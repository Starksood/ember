"""Enterprise KMS / HSM KeyStore adapters (AWS KMS, GCP KMS, HashiCorp Vault Transit).

Architecture:
  * Raw 256-bit subject CEKs, per-record DEKs, and the installation `tombstone_secret` are
    encrypted (envelope-wrapped) under the configured KMS key before persisting to local
    ACID storage (`SQLite` with `PRAGMA secure_delete = ON`, `0o600` file perms).
  * Plaintext DEKs never sit unencrypted on disk: dumping the keystore DB without KMS access
    exposes only KMS-wrapped ciphertext blobs (`EncryptionContext` / AAD bound per `key_ref`).
  * Cryptographic erasure purges the in-memory plaintext key cache AND securely deletes the
    sole KMS-wrapped DEK ciphertext rows on disk, making deleted records irrecoverable even
    to someone holding both the ledger DB and the KMS master key.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Mapping
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any
from urllib.parse import urlparse

from .crypto import new_key
from .keystore import KeyStore, ShredReport, _normalize_identifier

_KMS_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS kms_meta (
    key          TEXT PRIMARY KEY,
    wrapped_blob BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS kms_subject_keys (
    subject_id   TEXT PRIMARY KEY,
    wrapped_blob BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS kms_record_keys (
    record_id    TEXT PRIMARY KEY,
    wrapped_blob BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS kms_record_owners (
    record_id  TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    PRIMARY KEY (record_id, subject_id),
    FOREIGN KEY (record_id) REFERENCES kms_record_keys(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kms_record_owners_subject ON kms_record_owners(subject_id);
"""


class KmsOperationError(RuntimeError):
    """Raised when a remote KMS / HSM wrap or unwrap call fails."""


class EnvelopeKmsKeyStore(KeyStore):
    """Envelope-encrypting `KeyStore` delegating root key wrapping/unwrapping to an HSM/KMS backend."""

    def __init__(
        self,
        store_path: str | Path = ":memory:",
        *,
        tenant_id: str = "default",
        cache_enabled: bool = True,
    ) -> None:
        self.path = str(store_path)
        self.tenant_id = tenant_id
        self.cache_enabled = cache_enabled
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(_KMS_STORE_SCHEMA)
        if self.path != ":memory:":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        self._subject_cache: dict[str, bytes] = {}
        self._record_cache: dict[str, bytes] = {}
        self._tombstone_secret: bytes | None = None

    def _wrap(self, plaintext: bytes, context: dict[str, str]) -> bytes:
        raise NotImplementedError

    def _unwrap(self, wrapped_blob: bytes, context: dict[str, str]) -> bytes:
        raise NotImplementedError

    def clear_cache(self) -> None:
        """Drop all cached plaintext keys from memory (forces KMS unwrap on next read)."""
        with self._lock:
            self._subject_cache.clear()
            self._record_cache.clear()
            self._tombstone_secret = None

    def close(self) -> None:
        with self._lock:
            self.clear_cache()
            self._conn.close()

    def __enter__(self) -> "EnvelopeKmsKeyStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_tombstone_secret(self) -> bytes:
        with self._lock:
            if self.cache_enabled and self._tombstone_secret is not None:
                return self._tombstone_secret
            ctx = {"ember_tenant": self.tenant_id, "ember_purpose": "tombstone_secret"}
            row = self._conn.execute(
                "SELECT wrapped_blob FROM kms_meta WHERE key = 'tombstone_secret'"
            ).fetchone()
            if row is not None:
                secret = self._unwrap(bytes(row[0]), ctx)
            else:
                secret = secrets.token_bytes(32)
                wrapped = self._wrap(secret, ctx)
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO kms_meta (key, wrapped_blob) VALUES ('tombstone_secret', ?)",
                        (sqlite3.Binary(wrapped),),
                    )
                    row2 = self._conn.execute(
                        "SELECT wrapped_blob FROM kms_meta WHERE key = 'tombstone_secret'"
                    ).fetchone()
                    self._conn.execute("COMMIT")
                    if row2 is not None and bytes(row2[0]) != wrapped:
                        secret = self._unwrap(bytes(row2[0]), ctx)
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            if self.cache_enabled:
                self._tombstone_secret = secret
            return secret

    def get_or_create_subject_key(self, subject_id: str) -> bytes:
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_id must be a non-empty string")
        with self._lock:
            if self.cache_enabled and subject_id in self._subject_cache:
                return self._subject_cache[subject_id]
            ctx = {
                "ember_tenant": self.tenant_id,
                "ember_purpose": "subject_cek",
                "ember_key_ref": f"subject:{subject_id}",
            }
            row = self._conn.execute(
                "SELECT wrapped_blob FROM kms_subject_keys WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is not None:
                key = self._unwrap(bytes(row[0]), ctx)
            else:
                key = new_key()
                wrapped = self._wrap(key, ctx)
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO kms_subject_keys (subject_id, wrapped_blob) VALUES (?, ?)",
                        (subject_id, sqlite3.Binary(wrapped)),
                    )
                    row2 = self._conn.execute(
                        "SELECT wrapped_blob FROM kms_subject_keys WHERE subject_id = ?",
                        (subject_id,),
                    ).fetchone()
                    self._conn.execute("COMMIT")
                    if row2 is not None and bytes(row2[0]) != wrapped:
                        key = self._unwrap(bytes(row2[0]), ctx)
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            if self.cache_enabled:
                self._subject_cache[subject_id] = key
            return key

    def get_subject_key(self, subject_id: str) -> bytes | None:
        with self._lock:
            if self.cache_enabled and subject_id in self._subject_cache:
                return self._subject_cache[subject_id]
            row = self._conn.execute(
                "SELECT wrapped_blob FROM kms_subject_keys WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
            if row is None:
                return None
            ctx = {
                "ember_tenant": self.tenant_id,
                "ember_purpose": "subject_cek",
                "ember_key_ref": f"subject:{subject_id}",
            }
            key = self._unwrap(bytes(row[0]), ctx)
            if self.cache_enabled:
                self._subject_cache[subject_id] = key
            return key

    def create_record_key(self, record_id: str, subjects: Iterable[str]) -> bytes:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        owners = sorted({str(s) for s in subjects if str(s).strip()})
        if not owners:
            raise ValueError("a record key must be associated with at least one non-empty subject")
        with self._lock:
            if self.cache_enabled and record_id in self._record_cache:
                return self._record_cache[record_id]
            ctx = {
                "ember_tenant": self.tenant_id,
                "ember_purpose": "record_dek",
                "ember_key_ref": f"record:{record_id}",
            }
            row = self._conn.execute(
                "SELECT wrapped_blob FROM kms_record_keys WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is not None:
                key = self._unwrap(bytes(row[0]), ctx)
            else:
                key = new_key()
                wrapped = self._wrap(key, ctx)
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO kms_record_keys (record_id, wrapped_blob) VALUES (?, ?)",
                        (record_id, sqlite3.Binary(wrapped)),
                    )
                    self._conn.executemany(
                        "INSERT OR IGNORE INTO kms_record_owners (record_id, subject_id) VALUES (?, ?)",
                        [(record_id, sid) for sid in owners],
                    )
                    row2 = self._conn.execute(
                        "SELECT wrapped_blob FROM kms_record_keys WHERE record_id = ?",
                        (record_id,),
                    ).fetchone()
                    self._conn.execute("COMMIT")
                    if row2 is not None and bytes(row2[0]) != wrapped:
                        key = self._unwrap(bytes(row2[0]), ctx)
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            if self.cache_enabled:
                self._record_cache[record_id] = key
            return key

    def get_record_key(self, record_id: str) -> bytes | None:
        with self._lock:
            if self.cache_enabled and record_id in self._record_cache:
                return self._record_cache[record_id]
            row = self._conn.execute(
                "SELECT wrapped_blob FROM kms_record_keys WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            ctx = {
                "ember_tenant": self.tenant_id,
                "ember_purpose": "record_dek",
                "ember_key_ref": f"record:{record_id}",
            }
            key = self._unwrap(bytes(row[0]), ctx)
            if self.cache_enabled:
                self._record_cache[record_id] = key
            return key

    def shred_subject(self, subject_id: str) -> ShredReport:
        if not isinstance(subject_id, str) or not subject_id.strip():
            return ShredReport(subject_id=str(subject_id), subject_key_destroyed=False, record_keys_destroyed=0)
        with self._lock:
            self._subject_cache.pop(subject_id, None)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur_subj = self._conn.execute(
                    "DELETE FROM kms_subject_keys WHERE subject_id = ?",
                    (subject_id,),
                )
                subject_destroyed = cur_subj.rowcount > 0
                doomed_rows = self._conn.execute(
                    "SELECT record_id FROM kms_record_owners WHERE subject_id = ?",
                    (subject_id,),
                ).fetchall()
                doomed = [r[0] for r in doomed_rows]
                rec_deleted = 0
                for rid in doomed:
                    self._record_cache.pop(rid, None)
                if doomed:
                    cur_rec = self._conn.executemany(
                        "DELETE FROM kms_record_keys WHERE record_id = ?",
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
        secret = self._get_tombstone_secret()
        digest = hmac.new(secret, norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"

    def subject_ref(self, subject_id: str) -> str:
        norm = _normalize_identifier(subject_id)
        if not norm:
            raise ValueError("subject_id must be non-empty")
        secret = self._get_tombstone_secret()
        digest = hmac.new(secret, b"ref:" + norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"subj_{digest[:24]}"


class AwsKmsKeyStore(EnvelopeKmsKeyStore):
    """AWS KMS envelope KeyStore (`Encrypt` / `Decrypt` bound with `EncryptionContext`)."""

    def __init__(
        self,
        key_id: str,
        store_path: str | Path = ":memory:",
        *,
        tenant_id: str = "default",
        region_name: str | None = None,
        kms_client: Any = None,
        cache_enabled: bool = True,
    ) -> None:
        if not key_id or not str(key_id).strip():
            raise ValueError("AwsKmsKeyStore requires a non-empty KMS key_id or alias")
        super().__init__(store_path, tenant_id=tenant_id, cache_enabled=cache_enabled)
        self.key_id = key_id
        if kms_client is not None:
            self.client = kms_client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "AwsKmsKeyStore requires 'boto3'. Install with: pip install 'ember-custody[aws]'"
                ) from exc
            self.client = boto3.client("kms", region_name=region_name)

    def _wrap(self, plaintext: bytes, context: dict[str, str]) -> bytes:
        resp = self.client.encrypt(
            KeyId=self.key_id,
            Plaintext=plaintext,
            EncryptionContext=dict(context),
        )
        return bytes(resp["CiphertextBlob"])

    def _unwrap(self, wrapped_blob: bytes, context: dict[str, str]) -> bytes:
        resp = self.client.decrypt(
            CiphertextBlob=wrapped_blob,
            EncryptionContext=dict(context),
        )
        return bytes(resp["Plaintext"])


class GcpKmsKeyStore(EnvelopeKmsKeyStore):
    """Google Cloud KMS envelope KeyStore binding AAD context per wrapped DEK."""

    def __init__(
        self,
        key_name: str,
        store_path: str | Path = ":memory:",
        *,
        tenant_id: str = "default",
        kms_client: Any = None,
        cache_enabled: bool = True,
    ) -> None:
        if not key_name or not str(key_name).strip():
            raise ValueError("GcpKmsKeyStore requires a full KMS key_name resource path")
        super().__init__(store_path, tenant_id=tenant_id, cache_enabled=cache_enabled)
        self.key_name = key_name
        if kms_client is not None:
            self.client = kms_client
        else:
            try:
                from google.cloud import kms
            except ImportError as exc:
                raise ImportError(
                    "GcpKmsKeyStore requires 'google-cloud-kms'. Install with: pip install 'ember-custody[gcp]'"
                ) from exc
            self.client = kms.KeyManagementServiceClient()

    @staticmethod
    def _aad(context: dict[str, str]) -> bytes:
        return json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _wrap(self, plaintext: bytes, context: dict[str, str]) -> bytes:
        resp = self.client.encrypt(
            request={
                "name": self.key_name,
                "plaintext": plaintext,
                "additional_authenticated_data": self._aad(context),
            }
        )
        ciphertext = resp["ciphertext"] if isinstance(resp, Mapping) else resp.ciphertext
        return bytes(ciphertext)

    def _unwrap(self, wrapped_blob: bytes, context: dict[str, str]) -> bytes:
        resp = self.client.decrypt(
            request={
                "name": self.key_name,
                "ciphertext": wrapped_blob,
                "additional_authenticated_data": self._aad(context),
            }
        )
        plaintext = resp["plaintext"] if isinstance(resp, Mapping) else resp.plaintext
        return bytes(plaintext)


class VaultTransitKeyStore(EnvelopeKmsKeyStore):
    """HashiCorp Vault Transit envelope KeyStore over stdlib HTTPS/HTTP (zero third-party SDK needed)."""

    def __init__(
        self,
        vault_addr: str,
        key_name: str,
        token: str,
        store_path: str | Path = ":memory:",
        *,
        tenant_id: str = "default",
        mount_point: str = "transit",
        namespace: str | None = None,
        timeout: float = 10.0,
        http_fn: Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] | None = None,
        cache_enabled: bool = True,
    ) -> None:
        parsed = urlparse(vault_addr)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("vault_addr must be a valid http:// or https:// URL")
        if not key_name or not str(key_name).strip():
            raise ValueError("key_name must be non-empty")
        if not token or not str(token).strip():
            raise ValueError("token must be non-empty")
        super().__init__(store_path, tenant_id=tenant_id, cache_enabled=cache_enabled)
        self.vault_addr = vault_addr.rstrip("/")
        self.mount_point = mount_point.strip("/")
        self.key_name = key_name.strip()
        self.token = token
        self.namespace = namespace
        self.timeout = float(timeout)
        self._http_fn = http_fn

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.vault_addr}/{path.lstrip('/')}"
        headers = {
            "X-Vault-Token": self.token,
            "Content-Type": "application/json",
        }
        if self.namespace:
            headers["X-Vault-Namespace"] = self.namespace
        if self._http_fn is not None:
            return self._http_fn(url, headers, body)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"disallowed Vault URL scheme or host: {url!r}")
        target_path = parsed.path or "/"
        if parsed.query:
            target_path = f"{target_path}?{parsed.query}"
        payload_bytes = json.dumps(body).encode("utf-8")
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(parsed.netloc, timeout=self.timeout)
        try:
            conn.request("POST", target_path, body=payload_bytes, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise KmsOperationError(f"Vault Transit HTTP {resp.status} at {url}: {raw.decode('utf-8', 'replace')}")
            return json.loads(raw.decode("utf-8"))
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            if isinstance(exc, KmsOperationError):
                raise
            raise KmsOperationError(f"Vault Transit call failed at {url}: {exc}") from exc
        finally:
            conn.close()

    @staticmethod
    def _ctx_b64(context: dict[str, str]) -> str:
        raw = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(raw).decode("ascii")

    def _wrap(self, plaintext: bytes, context: dict[str, str]) -> bytes:
        res = self._post_json(
            f"v1/{self.mount_point}/encrypt/{self.key_name}",
            {
                "plaintext": base64.b64encode(plaintext).decode("ascii"),
                "context": self._ctx_b64(context),
            },
        )
        ct = res["data"]["ciphertext"]
        return str(ct).encode("utf-8")

    def _unwrap(self, wrapped_blob: bytes, context: dict[str, str]) -> bytes:
        res = self._post_json(
            f"v1/{self.mount_point}/decrypt/{self.key_name}",
            {
                "ciphertext": wrapped_blob.decode("utf-8"),
                "context": self._ctx_b64(context),
            },
        )
        pt_b64 = res["data"]["plaintext"]
        return base64.b64decode(pt_b64)


__all__ = [
    "AwsKmsKeyStore",
    "EnvelopeKmsKeyStore",
    "GcpKmsKeyStore",
    "KmsOperationError",
    "VaultTransitKeyStore",
]
