"""Key lifecycle management for crypto-shredding and keyed tombstones.

Separating keys from the immutable ledger makes erasure real:
  * Single-subject memories encrypt under that subject's CEK (`subject:<id>`).
  * Shared or derived memories encrypt under a unique per-record DEK (`record:<id>`)
    indexed under every contributing subject — erasing ANY contributing subject shreds
    the shared DEK, ensuring conservative zero-leak erasure while unshared bystander
    memories remain readable.
  * Subject tombstone tokens are computed via HMAC-SHA256 under a secret installation
    `tombstone_secret` so append-only `ERASE_SUBJECT` events never expose plaintext names
    or fall to unsalted dictionary attacks.

In production, implement `KeyStore` over your KMS/HSM (`AWS KMS`, `GCP KMS`,
`HashiCorp Vault Transit`). `FileKeyStore` enforces atomic writes and `0o600` permissions
and should live in a directory/volume excluded from immutable ledger backups.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading

from .crypto import new_key


@dataclass(frozen=True)
class KeyRef:
    kind: str  # "subject" or "record"
    key_id: str

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.key_id}"


@dataclass(frozen=True)
class ShredReport:
    subject_id: str
    subject_key_destroyed: bool
    record_keys_destroyed: int


class KeyStore(ABC):
    """Abstract mutable keystore controlling decryption and tombstone capabilities."""

    @abstractmethod
    def get_or_create_subject_key(self, subject_id: str) -> bytes: ...

    @abstractmethod
    def get_subject_key(self, subject_id: str) -> bytes | None: ...

    @abstractmethod
    def create_record_key(self, record_id: str, subjects: Iterable[str]) -> bytes: ...

    @abstractmethod
    def get_record_key(self, record_id: str) -> bytes | None: ...

    @abstractmethod
    def shred_subject(self, subject_id: str) -> ShredReport: ...

    @abstractmethod
    def tombstone_token(self, identifier: str) -> str: ...

    @abstractmethod
    def subject_ref(self, subject_id: str) -> str: ...

    def content_tombstone_token(self, identifier: str) -> str:
        """Domain-separated keyed HMAC token used for content-level zombie ingest blocking."""
        norm = _normalize_identifier(identifier)
        if not norm:
            return ""
        return self.tombstone_token("\x00content:" + norm)

    def blind_token(self, token: str) -> str:
        """Domain-separated keyed HMAC search token for Zero-Plaintext blind-index search."""
        norm = _normalize_identifier(token)
        if not norm:
            return ""
        raw = self.tombstone_token("\x00blind:" + norm)
        _, _, hex_part = raw.partition(":")
        digest = hex_part or hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"bt{digest[:24]}"

    def key_for(self, key_kind: str, key_id: str) -> bytes | None:
        if key_kind == "subject":
            return self.get_subject_key(key_id)
        if key_kind == "record":
            return self.get_record_key(key_id)
        return None


def _normalize_identifier(value: str) -> str:
    return " ".join(value.strip().lower().split())


_FILE_KEYSTORE_LOCK = threading.RLock()


class MemoryKeyStore(KeyStore):
    """In-memory mutable keystore for tests, benchmarks, and ephemeral runs."""

    def __init__(self, tombstone_secret: bytes | None = None) -> None:
        self._lock = threading.RLock()
        self._subject_keys: dict[str, bytes] = {}
        self._record_keys: dict[str, tuple[bytes, frozenset[str]]] = {}
        self._tombstone_secret = tombstone_secret or secrets.token_bytes(32)

    def _on_mutate(self) -> None:
        pass

    def get_or_create_subject_key(self, subject_id: str) -> bytes:
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_id must be a non-empty string")
        with self._lock:
            key = self._subject_keys.get(subject_id)
            if key is None:
                key = new_key()
                self._subject_keys[subject_id] = key
                self._on_mutate()
            return key

    def get_subject_key(self, subject_id: str) -> bytes | None:
        with self._lock:
            return self._subject_keys.get(subject_id)

    def create_record_key(self, record_id: str, subjects: Iterable[str]) -> bytes:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        owners = frozenset(str(s) for s in subjects if str(s).strip())
        if not owners:
            raise ValueError("a record key must be associated with at least one non-empty subject")
        with self._lock:
            if record_id in self._record_keys:
                key, _ = self._record_keys[record_id]
                return key
            key = new_key()
            self._record_keys[record_id] = (key, owners)
            self._on_mutate()
            return key

    def get_record_key(self, record_id: str) -> bytes | None:
        with self._lock:
            entry = self._record_keys.get(record_id)
            return entry[0] if entry is not None else None

    def shred_subject(self, subject_id: str) -> ShredReport:
        if not isinstance(subject_id, str) or not subject_id.strip():
            return ShredReport(subject_id=str(subject_id), subject_key_destroyed=False, record_keys_destroyed=0)
        with self._lock:
            subject_destroyed = self._subject_keys.pop(subject_id, None) is not None
            doomed = [rid for rid, (_, owners) in self._record_keys.items() if subject_id in owners]
            for rid in doomed:
                self._record_keys.pop(rid, None)
            if subject_destroyed or doomed:
                self._on_mutate()
            return ShredReport(
                subject_id=subject_id,
                subject_key_destroyed=subject_destroyed,
                record_keys_destroyed=len(doomed),
            )

    def tombstone_token(self, identifier: str) -> str:
        norm = _normalize_identifier(identifier)
        if not norm:
            return ""
        digest = hmac.new(self._tombstone_secret, norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"

    def subject_ref(self, subject_id: str) -> str:
        """Deterministic opaque reference for `subject_id` keyed by vault secret."""
        norm = _normalize_identifier(subject_id)
        if not norm:
            raise ValueError("subject_id must be non-empty")
        digest = hmac.new(self._tombstone_secret, b"ref:" + norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"subj_{digest[:24]}"

    def export_state(self) -> dict:
        with self._lock:
            return {
                "version": 1,
                "tombstone_secret": base64.b64encode(self._tombstone_secret).decode("ascii"),
                "subject_keys": {
                    sid: base64.b64encode(key).decode("ascii")
                    for sid, key in sorted(self._subject_keys.items())
                },
                "record_keys": {
                    rid: {
                        "key": base64.b64encode(key).decode("ascii"),
                        "subjects": sorted(owners),
                    }
                    for rid, (key, owners) in sorted(self._record_keys.items())
                },
            }

    @classmethod
    def from_state(cls, data: dict) -> "MemoryKeyStore":
        ks = cls(tombstone_secret=base64.b64decode(data["tombstone_secret"]))
        for sid, b64 in data.get("subject_keys", {}).items():
            ks._subject_keys[sid] = base64.b64decode(b64)
        for rid, rec in data.get("record_keys", {}).items():
            ks._record_keys[rid] = (base64.b64decode(rec["key"]), frozenset(rec["subjects"]))
        return ks


class FileKeyStore(MemoryKeyStore):
    """Atomic, 0o600 file-backed mutable KeyStore."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with _FILE_KEYSTORE_LOCK:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                super().__init__(tombstone_secret=base64.b64decode(raw["tombstone_secret"]))
                for sid, b64 in raw.get("subject_keys", {}).items():
                    self._subject_keys[sid] = base64.b64decode(b64)
                for rid, rec in raw.get("record_keys", {}).items():
                    self._record_keys[rid] = (base64.b64decode(rec["key"]), frozenset(rec["subjects"]))
            else:
                super().__init__()
                self._persist()

    def _on_mutate(self) -> None:
        self._persist()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(6)}.tmp"
        )
        payload = json.dumps(self.export_state(), sort_keys=True, indent=2).encode("utf-8")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


_SQLITE_KEYSTORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS keystore_meta (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS subject_keys (
    subject_id TEXT PRIMARY KEY,
    key_bytes  BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS record_keys (
    record_id TEXT PRIMARY KEY,
    key_bytes BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS record_owners (
    record_id  TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    PRIMARY KEY (record_id, subject_id),
    FOREIGN KEY (record_id) REFERENCES record_keys(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_record_owners_subject ON record_owners(subject_id);
"""


class SQLiteKeyStore(KeyStore):
    """ACID SQLite-backed mutable KeyStore with `PRAGMA secure_delete = ON` and 0o600 file perms."""

    def __init__(self, path: str | Path = ":memory:", *, tombstone_secret: bytes | None = None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(_SQLITE_KEYSTORE_SCHEMA)
        if self.path != ":memory:":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM keystore_meta WHERE key = 'tombstone_secret'"
            ).fetchone()
            if row is not None:
                self._tombstone_secret = bytes(row[0])
            else:
                secret = tombstone_secret or secrets.token_bytes(32)
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO keystore_meta (key, value) VALUES ('tombstone_secret', ?)",
                        (sqlite3.Binary(secret),),
                    )
                    row2 = self._conn.execute(
                        "SELECT value FROM keystore_meta WHERE key = 'tombstone_secret'"
                    ).fetchone()
                    self._tombstone_secret = bytes(row2[0]) if row2 is not None else secret
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteKeyStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_or_create_subject_key(self, subject_id: str) -> bytes:
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_id must be a non-empty string")
        with self._lock:
            row = self._conn.execute(
                "SELECT key_bytes FROM subject_keys WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is not None:
                return bytes(row[0])
            key = new_key()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO subject_keys (subject_id, key_bytes) VALUES (?, ?)",
                    (subject_id, sqlite3.Binary(key)),
                )
                row2 = self._conn.execute(
                    "SELECT key_bytes FROM subject_keys WHERE subject_id = ?",
                    (subject_id,),
                ).fetchone()
                self._conn.execute("COMMIT")
                return bytes(row2[0]) if row2 is not None else key
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def get_subject_key(self, subject_id: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key_bytes FROM subject_keys WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            return bytes(row[0]) if row is not None else None

    def create_record_key(self, record_id: str, subjects: Iterable[str]) -> bytes:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        owners = sorted({str(s) for s in subjects if str(s).strip()})
        if not owners:
            raise ValueError("a record key must be associated with at least one non-empty subject")
        with self._lock:
            row = self._conn.execute(
                "SELECT key_bytes FROM record_keys WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is not None:
                return bytes(row[0])
            key = new_key()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO record_keys (record_id, key_bytes) VALUES (?, ?)",
                    (record_id, sqlite3.Binary(key)),
                )
                self._conn.executemany(
                    "INSERT OR IGNORE INTO record_owners (record_id, subject_id) VALUES (?, ?)",
                    [(record_id, sid) for sid in owners],
                )
                row2 = self._conn.execute(
                    "SELECT key_bytes FROM record_keys WHERE record_id = ?",
                    (record_id,),
                ).fetchone()
                self._conn.execute("COMMIT")
                return bytes(row2[0]) if row2 is not None else key
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def get_record_key(self, record_id: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key_bytes FROM record_keys WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            return bytes(row[0]) if row is not None else None

    def shred_subject(self, subject_id: str) -> ShredReport:
        if not isinstance(subject_id, str) or not subject_id.strip():
            return ShredReport(subject_id=str(subject_id), subject_key_destroyed=False, record_keys_destroyed=0)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur_subj = self._conn.execute(
                    "DELETE FROM subject_keys WHERE subject_id = ?",
                    (subject_id,),
                )
                subject_destroyed = cur_subj.rowcount > 0
                doomed_rows = self._conn.execute(
                    "SELECT record_id FROM record_owners WHERE subject_id = ?",
                    (subject_id,),
                ).fetchall()
                doomed = [r[0] for r in doomed_rows]
                rec_deleted = 0
                if doomed:
                    cur_rec = self._conn.executemany(
                        "DELETE FROM record_keys WHERE record_id = ?",
                        [(rid,) for rid in doomed],
                    )
                    rec_deleted = cur_rec.rowcount if cur_rec.rowcount >= 0 else len(doomed)
                self._conn.execute("COMMIT")
                return ShredReport(
                    subject_id=subject_id,
                    subject_key_destroyed=subject_destroyed,
                    record_keys_destroyed=rec_deleted,
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def tombstone_token(self, identifier: str) -> str:
        norm = _normalize_identifier(identifier)
        if not norm:
            return ""
        digest = hmac.new(self._tombstone_secret, norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"

    def subject_ref(self, subject_id: str) -> str:
        norm = _normalize_identifier(subject_id)
        if not norm:
            raise ValueError("subject_id must be non-empty")
        digest = hmac.new(self._tombstone_secret, b"ref:" + norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"subj_{digest[:24]}"
