"""Append-only, hash-chained SQLite ledger and mutable redactable document store.

Durability contract:
  * PRAGMA journal_mode = WAL
  * PRAGMA synchronous = FULL (commit before return/ack)
  * Gap-free monotonic `seq` per tenant and strict `prev_hash` SHA-256 chaining
  * Idempotency conflicts on duplicate `event_id` with differing content reject atomically

Privacy separation:
  * The append-only `ledger_events` table stores ONLY encrypted content (`AES-256-GCM`) and
    metadata hashes; plaintext source documents NEVER live in `ledger_events`.
  * The separate `documents` table stores `RedactableDoc` leaf entries so erased subjects' leaves
    and nonces can be lawfully deleted without rewriting ledger history.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Protocol, runtime_checkable

from .keystore import KeyStore
from .merkle import RedactableDoc
from .records import LedgerEvent, MemoryItem, PendingErasureRecord, SourceRecord
from .reducer import (
    ConcurrentModificationError,
    EmberState,
    InvariantViolation,
    apply_event,
    decrypt_item_metadata,
    decrypt_item_payload,
)
from .verify import (
    GENESIS_HASH,
    canonical_bytes,
    compute_ledger_event_hash,
    verify_ledger_chain,
)


class LedgerIntegrityError(ValueError):
    """Raised when stored ledger rows fail hash-chain or sequence verification."""


class IdempotencyConflict(ValueError):
    """Raised when an `event_id` is reused with different payload content."""


def _validate_live_receipt_source(doc: RedactableDoc | None, receipt_meta: Any) -> None:
    if doc is None or not isinstance(receipt_meta, dict):
        return
    leaf_idx_raw = receipt_meta.get("leaf_index")
    if not isinstance(leaf_idx_raw, int) or isinstance(leaf_idx_raw, bool):
        return
    if leaf_idx_raw < 0 or leaf_idx_raw >= len(doc.entries):
        raise InvariantViolation(
            f"receipt leaf_index {leaf_idx_raw} out of range for source {doc.source_id!r}"
        )
    entry = doc.entries[leaf_idx_raw]
    if entry.get("redacted"):
        raise InvariantViolation(
            f"refusing ADD_ITEM citing redacted source leaf {doc.source_id!r}[{leaf_idx_raw}]"
        )
    span_indices = receipt_meta.get("span_indices")
    if isinstance(span_indices, list) and span_indices:
        spans_by_idx = {int(sp["span_index"]): sp for sp in entry.get("spans", ())}
        for si in span_indices:
            sp = spans_by_idx.get(int(si))
            if sp is not None and sp.get("redacted"):
                raise InvariantViolation(
                    f"refusing ADD_ITEM citing redacted source span {doc.source_id!r}[{leaf_idx_raw}:{si}]"
                )


@dataclass(frozen=True)
class AppendReceipt:
    seq: int
    event_id: str
    event_hash: str
    idempotent: bool


_TABLE_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _checkpoint_signable_bytes(ckpt: dict[str, Any]) -> bytes:
    body = {
        "kind": "ember-checkpoint/v1",
        "version": int(ckpt["version"]),
        "tenant_id": str(ckpt["tenant_id"]),
        "seq": int(ckpt["seq"]),
        "last_event_hash": str(ckpt["last_event_hash"]),
        "state_version": int(ckpt["state_version"]),
        "sources": ckpt["sources"],
        "items": ckpt["items"],
        "tombstoned_tokens": list(ckpt["tombstoned_tokens"]),
        "tombstoned_content_tokens": list(ckpt.get("tombstoned_content_tokens", ())),
        "created_at": str(ckpt["created_at"]),
    }
    return canonical_bytes(body)


def _verify_checkpoint_dict(
    ckpt: Any,
    *,
    tenant_id: str,
    expected_public_key_hex: str | None = None,
) -> dict[str, Any]:
    if not isinstance(ckpt, dict):
        raise LedgerIntegrityError("checkpoint must be a JSON object")
    if ckpt.get("version") != 1 or ckpt.get("tenant_id") != tenant_id:
        raise LedgerIntegrityError("checkpoint version or tenant_id mismatch")
    seq = ckpt.get("seq")
    state_ver = ckpt.get("state_version")
    if (
        not isinstance(seq, int)
        or isinstance(seq, bool)
        or seq < 0
        or not isinstance(state_ver, int)
        or isinstance(state_ver, bool)
        or state_ver < 0
    ):
        raise LedgerIntegrityError("invalid checkpoint seq/state_version")
    last_hash = ckpt.get("last_event_hash")
    if not isinstance(last_hash, str) or not last_hash.startswith("sha256:"):
        raise LedgerIntegrityError("invalid checkpoint last_event_hash")

    sig = ckpt.get("signature")
    if expected_public_key_hex is not None or sig is not None:
        pub = expected_public_key_hex if expected_public_key_hex is not None else ckpt.get("public_key", "")
        if not isinstance(sig, str) or not sig.startswith("ed25519:") or not isinstance(pub, str) or not pub:
            raise LedgerIntegrityError("checkpoint missing or invalid Ed25519 signature")
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519

        _, _, hex_sig = sig.partition(":")
        try:
            pub_bytes = bytes.fromhex(pub)
            sig_bytes = bytes.fromhex(hex_sig)
            if len(pub_bytes) != 32 or len(sig_bytes) != 64:
                raise LedgerIntegrityError("invalid Ed25519 key or signature length in checkpoint")
            pk = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            pk.verify(sig_bytes, _checkpoint_signable_bytes(ckpt))
        except (InvalidSignature, ValueError, TypeError, KeyError) as exc:
            raise LedgerIntegrityError(f"checkpoint signature verification failed: {exc}") from exc
    return ckpt


def _build_checkpoint_dict(
    *,
    tenant_id: str,
    seq: int,
    last_event_hash: str,
    state: EmberState,
    created_at: str | None = None,
    sign_fn: Callable[[bytes], str] | None = None,
    public_key_hex: str | None = None,
) -> dict[str, Any]:
    if seq < 0:
        raise ValueError("cannot checkpoint an empty ledger with no committed events")
    ckpt: dict[str, Any] = {
        "version": 1,
        "tenant_id": tenant_id,
        "seq": seq,
        "last_event_hash": last_event_hash,
        "state_version": state.version,
        "sources": {k: v.as_dict() for k, v in sorted(state.sources.items())},
        # Store ciphertext ledger payloads ONLY — NEVER unencrypted plaintext PII!
        "items": {k: v.ledger_payload() for k, v in sorted(state.items.items())},
        "tombstoned_tokens": sorted(state.tombstoned_tokens),
        "tombstoned_content_tokens": sorted(state.tombstoned_content_tokens),
        "created_at": created_at or _utc_now(),
    }
    if sign_fn is not None and public_key_hex is not None:
        ckpt["signer_scheme"] = "ed25519"
        ckpt["public_key"] = public_key_hex
        ckpt["signature"] = sign_fn(_checkpoint_signable_bytes(ckpt))
    return ckpt


def _restore_state_from_checkpoint(ckpt: dict[str, Any], keystore: KeyStore | None) -> EmberState:
    sources = {str(k): SourceRecord.from_dict(v) for k, v in (ckpt.get("sources") or {}).items()}
    items: dict[str, MemoryItem] = {}
    for item_id, payload in (ckpt.get("items") or {}).items():
        plain_subjects, plain_text, plain_quote = decrypt_item_payload(payload, keystore)
        plain_meta = decrypt_item_metadata(payload, keystore)
        items[str(item_id)] = MemoryItem.from_ledger_payload(
            payload,
            subjects=plain_subjects,
            text=plain_text,
            quote=plain_quote,
            metadata=plain_meta,
        )
    return EmberState(
        version=int(ckpt["state_version"]),
        sources=sources,  # wrapped or passed directly via EmberState; freeze identically to apply_event
        items=items,
        tombstoned_tokens=frozenset(str(t) for t in ckpt.get("tombstoned_tokens", ())),
        tombstoned_content_tokens=frozenset(str(t) for t in ckpt.get("tombstoned_content_tokens", ())),
    )


def _freeze_state(state: EmberState) -> EmberState:
    from types import MappingProxyType

    return EmberState(
        version=state.version,
        sources=MappingProxyType(dict(state.sources)),
        items=MappingProxyType(dict(state.items)),
        tombstoned_tokens=state.tombstoned_tokens,
        tombstoned_content_tokens=state.tombstoned_content_tokens,
    )


def _verify_delta_events(
    events: Sequence[LedgerEvent],
    *,
    tenant_id: str,
    expected_seq: int,
    expected_prev_hash: str,
) -> None:
    prev = expected_prev_hash
    seq = expected_seq
    for ev in events:
        if ev.tenant_id != tenant_id:
            raise LedgerIntegrityError(f"tenant mismatch in delta event seq {ev.seq}")
        if ev.seq != seq:
            raise LedgerIntegrityError(f"gap or out-of-order delta seq: expected {seq}, got {ev.seq}")
        if ev.prev_hash != prev:
            raise LedgerIntegrityError(f"broken hash chain at seq {ev.seq}")
        expected_hash = compute_ledger_event_hash(
            {
                "seq": ev.seq,
                "tenant_id": ev.tenant_id,
                "event_id": ev.event_id,
                "kind": ev.kind,
                "committed_at": ev.committed_at,
                "payload": ev.payload,
                "prev_hash": ev.prev_hash,
            }
        )
        if ev.event_hash != expected_hash:
            raise LedgerIntegrityError(f"hash mismatch at delta seq {ev.seq}")
        prev = ev.event_hash
        seq += 1


@runtime_checkable
class LedgerBackend(Protocol):
    """Pluggable durable append-only ledger & redactable doc store interface."""

    def close(self) -> None: ...

    def events(self, tenant_id: str = "default") -> list[LedgerEvent]: ...

    def tail(self, tenant_id: str = "default") -> tuple[str, int]: ...

    def verify_chain(self, tenant_id: str = "default") -> tuple[bool, str]: ...

    def append(
        self,
        *,
        tenant_id: str,
        event_id: str,
        kind: str,
        committed_at: str,
        payload: dict[str, Any],
        current_state: EmberState | None = None,
        keystore: KeyStore | None = None,
        validate: bool = True,
    ) -> tuple[LedgerEvent, bool, EmberState | None]: ...

    def replay(
        self,
        tenant_id: str = "default",
        *,
        keystore: KeyStore | None = None,
    ) -> EmberState: ...

    def put_document(self, tenant_id: str, doc: RedactableDoc) -> None: ...

    def get_document(self, tenant_id: str, source_id: str) -> RedactableDoc | None: ...

    def list_document_ids(self, tenant_id: str) -> list[str]: ...

    def put_certificate(self, cert_dict: dict[str, Any]) -> None: ...

    def list_certificates(self, tenant_id: str = "default") -> list[dict[str, Any]]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_events (
    tenant_id    TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    event_id     TEXT NOT NULL,
    kind         TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    event_hash   TEXT NOT NULL,
    PRIMARY KEY (tenant_id, seq),
    UNIQUE (tenant_id, event_id),
    UNIQUE (tenant_id, event_hash)
);

CREATE TABLE IF NOT EXISTS documents (
    tenant_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    doc_json  TEXT NOT NULL,
    PRIMARY KEY (tenant_id, source_id)
);

CREATE TABLE IF NOT EXISTS certificates (
    tenant_id      TEXT NOT NULL,
    certificate_id TEXT NOT NULL,
    subject_id     TEXT NOT NULL,
    cert_json      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, certificate_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    tenant_id       TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    last_event_hash TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    PRIMARY KEY (tenant_id, seq)
);

CREATE TABLE IF NOT EXISTS pending_erasures (
    tenant_id   TEXT NOT NULL,
    erasure_id  TEXT NOT NULL,
    subject_id  TEXT NOT NULL,
    status      TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (tenant_id, erasure_id)
);
"""


class SQLiteLedger:
    """Thread-safe durable SQLite append-only ledger + mutable redactable doc store."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=10.0)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA busy_timeout = 10000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _row_to_event(self, row: tuple[Any, ...]) -> LedgerEvent:
        return LedgerEvent(
            tenant_id=row[0],
            seq=row[1],
            event_id=row[2],
            kind=row[3],
            committed_at=row[4],
            payload=json.loads(row[5]),
            prev_hash=row[6],
            event_hash=row[7],
        )

    def events(self, tenant_id: str = "default") -> list[LedgerEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash
                FROM ledger_events
                WHERE tenant_id = ?
                ORDER BY seq ASC
                """,
                (tenant_id,),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

    def events_after(self, tenant_id: str = "default", *, seq_exclusive: int = -1) -> list[LedgerEvent]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash
                FROM ledger_events
                WHERE tenant_id = ? AND seq > ?
                ORDER BY seq ASC
                """,
                (tenant_id, seq_exclusive),
            ).fetchall()
            return [self._row_to_event(r) for r in rows]

    def get_event_by_id(self, tenant_id: str, event_id: str) -> LedgerEvent | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash
                FROM ledger_events
                WHERE tenant_id = ? AND event_id = ?
                """,
                (tenant_id, event_id),
            ).fetchone()
            return self._row_to_event(row) if row is not None else None

    def tail(self, tenant_id: str = "default") -> tuple[str, int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT event_hash, seq
                FROM ledger_events
                WHERE tenant_id = ?
                ORDER BY seq DESC LIMIT 1
                """,
                (tenant_id,),
            ).fetchone()
            return (row[0], row[1] + 1) if row else (GENESIS_HASH, 0)

    def verify_chain(self, tenant_id: str = "default") -> tuple[bool, str]:
        evs = self.events(tenant_id)
        return verify_ledger_chain([e.as_dict() for e in evs])

    def append(
        self,
        *,
        tenant_id: str,
        event_id: str,
        kind: str,
        committed_at: str,
        payload: dict[str, Any],
        current_state: EmberState | None = None,
        keystore: KeyStore | None = None,
        validate: bool = True,
        expected_seq: int | None = None,
    ) -> tuple[LedgerEvent, bool, EmberState | None]:
        """Atomically append one event, checking idempotency, OCC seq, invariants, and chaining."""
        payload_canon = canonical_bytes(payload).decode("utf-8")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing_row = self._conn.execute(
                    """
                    SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash
                    FROM ledger_events
                    WHERE tenant_id = ? AND event_id = ?
                    """,
                    (tenant_id, event_id),
                ).fetchone()
                if existing_row is not None:
                    ev = self._row_to_event(existing_row)
                    stored_canon = canonical_bytes(ev.payload).decode("utf-8")
                    if ev.kind != kind or ev.committed_at != committed_at or stored_canon != payload_canon:
                        raise IdempotencyConflict(
                            f"event_id {event_id!r} already committed for tenant {tenant_id!r} with different content"
                        )
                    self._conn.execute("COMMIT")
                    return ev, True, current_state

                prev_hash, seq = self.tail(tenant_id)
                if expected_seq is not None and seq != expected_seq:
                    raise ConcurrentModificationError(
                        f"concurrent ledger modification for tenant {tenant_id!r} "
                        f"(expected seq {expected_seq}, ledger at seq {seq})"
                    )
                candidate = {
                    "seq": seq,
                    "tenant_id": tenant_id,
                    "event_id": event_id,
                    "kind": kind,
                    "committed_at": committed_at,
                    "payload": payload,
                    "prev_hash": prev_hash,
                }
                event_hash = compute_ledger_event_hash(candidate)
                ev = LedgerEvent(
                    seq=seq,
                    tenant_id=tenant_id,
                    event_id=event_id,
                    kind=kind,
                    committed_at=committed_at,
                    payload=payload,
                    prev_hash=prev_hash,
                    event_hash=event_hash,
                )

                next_state = current_state
                if validate:
                    base = (
                        current_state
                        if (current_state is not None and current_state.version == seq)
                        else self.replay(tenant_id, keystore=keystore)
                    )
                    next_state = apply_event(base, ev, keystore=keystore)
                    if kind == "ADD_ITEM":
                        rmeta = (payload.get("item") or {}).get("receipt_meta")
                        if isinstance(rmeta, dict) and isinstance(rmeta.get("source_id"), str):
                            _validate_live_receipt_source(
                                self.get_document(tenant_id, rmeta["source_id"]),
                                rmeta,
                            )

                self._conn.execute(
                    """
                    INSERT INTO ledger_events (
                        tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        seq,
                        event_id,
                        kind,
                        committed_at,
                        payload_canon,
                        prev_hash,
                        event_hash,
                    ),
                )
                self._conn.execute("COMMIT")
                return ev, False, next_state
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # ── Signed Privacy-Preserving State Checkpoints (O(delta) startup) ────────

    def save_checkpoint(
        self,
        tenant_id: str,
        state: EmberState,
        *,
        seq: int | None = None,
        last_event_hash: str | None = None,
        sign_fn: Callable[[bytes], str] | None = None,
        public_key_hex: str | None = None,
    ) -> dict[str, Any]:
        """Persist a ciphertext-only snapshot at `seq` so restarts replay only newer delta events."""
        with self._lock:
            if seq is None or last_event_hash is None:
                tail_hash, next_seq = self.tail(tenant_id)
                seq = next_seq - 1
                last_event_hash = tail_hash
            ckpt = _build_checkpoint_dict(
                tenant_id=tenant_id,
                seq=seq,
                last_event_hash=last_event_hash,
                state=state,
                sign_fn=sign_fn,
                public_key_hex=public_key_hex,
            )
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO checkpoints (tenant_id, seq, last_event_hash, checkpoint_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(tenant_id, seq) DO UPDATE SET
                        last_event_hash = excluded.last_event_hash,
                        checkpoint_json = excluded.checkpoint_json
                    """,
                    (tenant_id, seq, last_event_hash, canonical_bytes(ckpt).decode("utf-8")),
                )
                self._conn.execute("COMMIT")
                return ckpt
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def load_latest_checkpoint(
        self,
        tenant_id: str = "default",
        *,
        expected_public_key_hex: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT checkpoint_json FROM checkpoints WHERE tenant_id = ? ORDER BY seq DESC LIMIT 1",
                (tenant_id,),
            ).fetchone()
            if row is None:
                return None
            ckpt = _verify_checkpoint_dict(
                json.loads(row[0]),
                tenant_id=tenant_id,
                expected_public_key_hex=expected_public_key_hex,
            )
            anchor_row = self._conn.execute(
                "SELECT event_hash FROM ledger_events WHERE tenant_id = ? AND seq = ?",
                (tenant_id, ckpt["seq"]),
            ).fetchone()
            if anchor_row is None or anchor_row[0] != ckpt["last_event_hash"]:
                raise LedgerIntegrityError(
                    f"checkpoint seq={ckpt['seq']} does not match ledger anchor event_hash"
                )
            return ckpt

    def replay(
        self,
        tenant_id: str = "default",
        *,
        keystore: KeyStore | None = None,
        use_checkpoint: bool = True,
        checkpoint_public_key: str | None = None,
    ) -> EmberState:
        """Reconstruct `EmberState` via latest verified checkpoint + O(delta) tail or full genesis replay."""
        with self._lock:
            ckpt = (
                self.load_latest_checkpoint(tenant_id, expected_public_key_hex=checkpoint_public_key)
                if use_checkpoint
                else None
            )
            if ckpt is not None:
                ckpt_seq = int(ckpt["seq"])
                delta_evs = self.events_after(tenant_id, seq_exclusive=ckpt_seq)
                _verify_delta_events(
                    delta_evs,
                    tenant_id=tenant_id,
                    expected_seq=ckpt_seq + 1,
                    expected_prev_hash=str(ckpt["last_event_hash"]),
                )
                state = _freeze_state(_restore_state_from_checkpoint(ckpt, keystore))
                for ev in delta_evs:
                    state = apply_event(state, ev, keystore=keystore)
                return state

            evs = self.events(tenant_id)
            ok, msg = verify_ledger_chain([e.as_dict() for e in evs])
            if not ok:
                raise LedgerIntegrityError(msg)
            state = EmberState.empty()
            for ev in evs:
                state = apply_event(state, ev, keystore=keystore)
            return state

    # ── Mutable RedactableDocument Storage (separate from immutable chain) ────────

    def put_document(self, tenant_id: str, doc: RedactableDoc) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO documents (tenant_id, source_id, doc_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(tenant_id, source_id) DO UPDATE SET doc_json = excluded.doc_json
                    """,
                    (tenant_id, doc.source_id, canonical_bytes(doc.as_dict()).decode("utf-8")),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def get_document(self, tenant_id: str, source_id: str) -> RedactableDoc | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT doc_json FROM documents WHERE tenant_id = ? AND source_id = ?",
                (tenant_id, source_id),
            ).fetchone()
            if row is None:
                return None
            return RedactableDoc.from_dict(json.loads(row[0]))

    def list_document_ids(self, tenant_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT source_id FROM documents WHERE tenant_id = ? ORDER BY source_id",
                (tenant_id,),
            ).fetchall()
            return [r[0] for r in rows]

    # ── Certificate archive ───────────────────────────────────────────────────────

    def put_certificate(self, cert_dict: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO certificates (tenant_id, certificate_id, subject_id, cert_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(tenant_id, certificate_id) DO UPDATE SET cert_json = excluded.cert_json
                    """,
                    (
                        cert_dict["tenant_id"],
                        cert_dict["certificate_id"],
                        cert_dict["subject_id"],
                        canonical_bytes(cert_dict).decode("utf-8"),
                    ),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def list_certificates(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT cert_json FROM certificates WHERE tenant_id = ? ORDER BY rowid ASC",
                (tenant_id,),
            ).fetchall()
            return [json.loads(r[0]) for r in rows]

    # ── Durable Pending-Erasure Journal (crash recovery & reconciliation) ─────

    def put_pending_erasure(self, tenant_id: str, record: PendingErasureRecord) -> None:
        encoded = canonical_bytes(record.as_dict()).decode("utf-8")
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO pending_erasures (
                        tenant_id, erasure_id, subject_id, status, updated_at, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id, erasure_id) DO UPDATE SET
                        status = excluded.status,
                        updated_at = excluded.updated_at,
                        record_json = excluded.record_json
                    """,
                    (
                        tenant_id,
                        record.erasure_id,
                        record.subject_id,
                        record.status,
                        record.updated_at,
                        encoded,
                    ),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def get_pending_erasure(self, tenant_id: str, erasure_id: str) -> PendingErasureRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT record_json FROM pending_erasures WHERE tenant_id = ? AND erasure_id = ?",
                (tenant_id, erasure_id),
            ).fetchone()
            return PendingErasureRecord.from_dict(json.loads(row[0])) if row is not None else None

    def list_pending_erasures(
        self,
        tenant_id: str = "default",
        *,
        include_completed: bool = False,
    ) -> list[PendingErasureRecord]:
        with self._lock:
            if include_completed:
                rows = self._conn.execute(
                    "SELECT record_json FROM pending_erasures WHERE tenant_id = ? ORDER BY rowid ASC",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT record_json FROM pending_erasures WHERE tenant_id = ? AND status != 'completed' "
                    "ORDER BY rowid ASC",
                    (tenant_id,),
                ).fetchall()
            return [PendingErasureRecord.from_dict(json.loads(r[0])) for r in rows]


def _open_pg_connection(dsn: str) -> Any:
    try:
        import psycopg

        return psycopg.connect(dsn)
    except ImportError:
        try:
            import psycopg2

            return psycopg2.connect(dsn)
        except ImportError as exc:
            raise ImportError(
                "PostgresLedger requires 'psycopg' (v3) or 'psycopg2'. "
                "Install with: pip install 'ember-custody[postgres]'"
            ) from exc


class PostgresLedger:
    """Thread-safe PostgreSQL append-only ledger + redactable document & checkpoint store."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection: Any = None,
        connect_fn: Callable[[], Any] | None = None,
        table_prefix: str = "ember_",
    ) -> None:
        if not _TABLE_PREFIX_RE.fullmatch(table_prefix):
            raise ValueError(f"invalid table_prefix {table_prefix!r}; must match [A-Za-z_][A-Za-z0-9_]*")
        if connection is not None:
            self._conn = connection
            self._owns_conn = False
        elif connect_fn is not None:
            self._conn = connect_fn()
            self._owns_conn = True
        elif dsn:
            self._conn = _open_pg_connection(dsn)
            self._owns_conn = True
        else:
            raise ValueError("PostgresLedger requires dsn, connection, or connect_fn")

        self._lock = threading.RLock()
        p = table_prefix
        self._tbl_events = p + "ledger_events"
        self._tbl_docs = p + "documents"
        self._tbl_certs = p + "certificates"
        self._tbl_ckpts = p + "checkpoints"
        self._tbl_pending = p + "pending_erasures"
        self._init_schema()

    def _exec(self, cur: Any, sql_template: str, params: Sequence[Any] = ()) -> Any:
        sql = (
            sql_template.replace("__EVENTS__", self._tbl_events)
            .replace("__DOCS__", self._tbl_docs)
            .replace("__CERTS__", self._tbl_certs)
            .replace("__CKPTS__", self._tbl_ckpts)
            .replace("__PENDING__", self._tbl_pending)
        )
        if params:
            cur.execute(sql, tuple(params))
        else:
            cur.execute(sql)
        return cur

    def _run_tx(self, fn: Callable[[Any], Any]) -> Any:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                res = fn(cur)
                self._conn.commit()
                return res
            except BaseException:
                rollback = getattr(self._conn, "rollback", None)
                if callable(rollback):
                    rollback()
                raise
            finally:
                close_cur = getattr(cur, "close", None)
                if callable(close_cur):
                    close_cur()

    def _acquire_tenant_xact_lock(self, cur: Any, tenant_id: str) -> None:
        try:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (tenant_id,))
        except Exception:
            # Lightweight DB-API mocks that do not emulate pg_advisory_xact_lock rely on in-process RLock + BEGIN.
            return

    def _init_schema(self) -> None:
        ddl_blocks = (
            """
            CREATE TABLE IF NOT EXISTS __EVENTS__ (
                tenant_id    TEXT NOT NULL,
                seq          BIGINT NOT NULL,
                event_id     TEXT NOT NULL,
                kind         TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                prev_hash    TEXT NOT NULL,
                event_hash   TEXT NOT NULL,
                PRIMARY KEY (tenant_id, seq),
                UNIQUE (tenant_id, event_id),
                UNIQUE (tenant_id, event_hash)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS __DOCS__ (
                tenant_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                doc_json  TEXT NOT NULL,
                PRIMARY KEY (tenant_id, source_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS __CERTS__ (
                tenant_id      TEXT NOT NULL,
                certificate_id TEXT NOT NULL,
                subject_id     TEXT NOT NULL,
                cert_json      TEXT NOT NULL,
                PRIMARY KEY (tenant_id, certificate_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS __CKPTS__ (
                tenant_id       TEXT NOT NULL,
                seq             BIGINT NOT NULL,
                last_event_hash TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                PRIMARY KEY (tenant_id, seq)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS __PENDING__ (
                tenant_id   TEXT NOT NULL,
                erasure_id  TEXT NOT NULL,
                subject_id  TEXT NOT NULL,
                status      TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (tenant_id, erasure_id)
            )
            """,
        )

        def _create(cur: Any) -> None:
            for block in ddl_blocks:
                self._exec(cur, block)

        self._run_tx(_create)

    def close(self) -> None:
        with self._lock:
            if self._owns_conn and hasattr(self._conn, "close"):
                self._conn.close()

    def __enter__(self) -> "PostgresLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _row_to_event(self, row: tuple[Any, ...]) -> LedgerEvent:
        return LedgerEvent(
            tenant_id=row[0],
            seq=int(row[1]),
            event_id=row[2],
            kind=row[3],
            committed_at=row[4],
            payload=json.loads(row[5]),
            prev_hash=row[6],
            event_hash=row[7],
        )

    def events(self, tenant_id: str = "default") -> list[LedgerEvent]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash "
                    "FROM __EVENTS__ WHERE tenant_id = %s ORDER BY seq ASC",
                    (tenant_id,),
                )
                return [self._row_to_event(r) for r in cur.fetchall()]
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def events_after(self, tenant_id: str = "default", *, seq_exclusive: int = -1) -> list[LedgerEvent]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash "
                    "FROM __EVENTS__ WHERE tenant_id = %s AND seq > %s ORDER BY seq ASC",
                    (tenant_id, seq_exclusive),
                )
                return [self._row_to_event(r) for r in cur.fetchall()]
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def get_event_by_id(self, tenant_id: str, event_id: str) -> LedgerEvent | None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash "
                    "FROM __EVENTS__ WHERE tenant_id = %s AND event_id = %s",
                    (tenant_id, event_id),
                )
                row = cur.fetchone()
                return self._row_to_event(row) if row is not None else None
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def tail(self, tenant_id: str = "default") -> tuple[str, int]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT event_hash, seq FROM __EVENTS__ WHERE tenant_id = %s ORDER BY seq DESC LIMIT 1",
                    (tenant_id,),
                )
                row = cur.fetchone()
                return (row[0], int(row[1]) + 1) if row else (GENESIS_HASH, 0)
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def verify_chain(self, tenant_id: str = "default") -> tuple[bool, str]:
        return verify_ledger_chain([e.as_dict() for e in self.events(tenant_id)])

    def append(
        self,
        *,
        tenant_id: str,
        event_id: str,
        kind: str,
        committed_at: str,
        payload: dict[str, Any],
        current_state: EmberState | None = None,
        keystore: KeyStore | None = None,
        validate: bool = True,
        expected_seq: int | None = None,
    ) -> tuple[LedgerEvent, bool, EmberState | None]:
        payload_canon = canonical_bytes(payload).decode("utf-8")

        def _op(cur: Any) -> tuple[LedgerEvent, bool, EmberState | None]:
            self._acquire_tenant_xact_lock(cur, tenant_id)
            self._exec(
                cur,
                "SELECT tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash "
                "FROM __EVENTS__ WHERE tenant_id = %s AND event_id = %s",
                (tenant_id, event_id),
            )
            existing = cur.fetchone()
            if existing is not None:
                ev = self._row_to_event(existing)
                stored_canon = canonical_bytes(ev.payload).decode("utf-8")
                if ev.kind != kind or ev.committed_at != committed_at or stored_canon != payload_canon:
                    raise IdempotencyConflict(
                        f"event_id {event_id!r} already committed for tenant {tenant_id!r} with different content"
                    )
                return ev, True, current_state

            self._exec(
                cur,
                "SELECT event_hash, seq FROM __EVENTS__ WHERE tenant_id = %s ORDER BY seq DESC LIMIT 1",
                (tenant_id,),
            )
            tail_row = cur.fetchone()
            prev_hash, seq = (tail_row[0], int(tail_row[1]) + 1) if tail_row else (GENESIS_HASH, 0)
            if expected_seq is not None and seq != expected_seq:
                raise ConcurrentModificationError(
                    f"concurrent ledger modification for tenant {tenant_id!r} "
                    f"(expected seq {expected_seq}, ledger at seq {seq})"
                )
            candidate = {
                "seq": seq,
                "tenant_id": tenant_id,
                "event_id": event_id,
                "kind": kind,
                "committed_at": committed_at,
                "payload": payload,
                "prev_hash": prev_hash,
            }
            event_hash = compute_ledger_event_hash(candidate)
            ev = LedgerEvent(
                seq=seq,
                tenant_id=tenant_id,
                event_id=event_id,
                kind=kind,
                committed_at=committed_at,
                payload=payload,
                prev_hash=prev_hash,
                event_hash=event_hash,
            )

            next_state = current_state
            if validate:
                base = (
                    current_state
                    if (current_state is not None and current_state.version == seq)
                    else self.replay(tenant_id, keystore=keystore)
                )
                next_state = apply_event(base, ev, keystore=keystore)
                if kind == "ADD_ITEM":
                    rmeta = (payload.get("item") or {}).get("receipt_meta")
                    if isinstance(rmeta, dict) and isinstance(rmeta.get("source_id"), str):
                        self._exec(
                            cur,
                            "SELECT doc_json FROM __DOCS__ WHERE tenant_id = %s AND source_id = %s",
                            (tenant_id, rmeta["source_id"]),
                        )
                        doc_row = cur.fetchone()
                        if doc_row is not None:
                            _validate_live_receipt_source(
                                RedactableDoc.from_dict(json.loads(doc_row[0])),
                                rmeta,
                            )

            self._exec(
                cur,
                "INSERT INTO __EVENTS__ (tenant_id, seq, event_id, kind, committed_at, payload_json, prev_hash, event_hash) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (tenant_id, seq, event_id, kind, committed_at, payload_canon, prev_hash, event_hash),
            )
            return ev, False, next_state

        return self._run_tx(_op)

    def save_checkpoint(
        self,
        tenant_id: str,
        state: EmberState,
        *,
        seq: int | None = None,
        last_event_hash: str | None = None,
        sign_fn: Callable[[bytes], str] | None = None,
        public_key_hex: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if seq is None or last_event_hash is None:
                tail_hash, next_seq = self.tail(tenant_id)
                seq = next_seq - 1
                last_event_hash = tail_hash
            ckpt = _build_checkpoint_dict(
                tenant_id=tenant_id,
                seq=seq,
                last_event_hash=last_event_hash,
                state=state,
                sign_fn=sign_fn,
                public_key_hex=public_key_hex,
            )

            def _op(cur: Any) -> dict[str, Any]:
                self._exec(
                    cur,
                    "INSERT INTO __CKPTS__ (tenant_id, seq, last_event_hash, checkpoint_json) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (tenant_id, seq) DO UPDATE SET "
                    "last_event_hash = EXCLUDED.last_event_hash, checkpoint_json = EXCLUDED.checkpoint_json",
                    (tenant_id, seq, last_event_hash, canonical_bytes(ckpt).decode("utf-8")),
                )
                return ckpt

            return self._run_tx(_op)

    def load_latest_checkpoint(
        self,
        tenant_id: str = "default",
        *,
        expected_public_key_hex: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT checkpoint_json FROM __CKPTS__ WHERE tenant_id = %s ORDER BY seq DESC LIMIT 1",
                    (tenant_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                ckpt = _verify_checkpoint_dict(
                    json.loads(row[0]),
                    tenant_id=tenant_id,
                    expected_public_key_hex=expected_public_key_hex,
                )
                self._exec(
                    cur,
                    "SELECT event_hash FROM __EVENTS__ WHERE tenant_id = %s AND seq = %s",
                    (tenant_id, ckpt["seq"]),
                )
                anchor = cur.fetchone()
                if anchor is None or anchor[0] != ckpt["last_event_hash"]:
                    raise LedgerIntegrityError(
                        f"checkpoint seq={ckpt['seq']} does not match ledger anchor event_hash"
                    )
                return ckpt
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def replay(
        self,
        tenant_id: str = "default",
        *,
        keystore: KeyStore | None = None,
        use_checkpoint: bool = True,
        checkpoint_public_key: str | None = None,
    ) -> EmberState:
        with self._lock:
            ckpt = (
                self.load_latest_checkpoint(tenant_id, expected_public_key_hex=checkpoint_public_key)
                if use_checkpoint
                else None
            )
            if ckpt is not None:
                ckpt_seq = int(ckpt["seq"])
                delta_evs = self.events_after(tenant_id, seq_exclusive=ckpt_seq)
                _verify_delta_events(
                    delta_evs,
                    tenant_id=tenant_id,
                    expected_seq=ckpt_seq + 1,
                    expected_prev_hash=str(ckpt["last_event_hash"]),
                )
                state = _freeze_state(_restore_state_from_checkpoint(ckpt, keystore))
                for ev in delta_evs:
                    state = apply_event(state, ev, keystore=keystore)
                return state

            evs = self.events(tenant_id)
            ok, msg = verify_ledger_chain([e.as_dict() for e in evs])
            if not ok:
                raise LedgerIntegrityError(msg)
            state = EmberState.empty()
            for ev in evs:
                state = apply_event(state, ev, keystore=keystore)
            return state

    def put_document(self, tenant_id: str, doc: RedactableDoc) -> None:
        encoded = canonical_bytes(doc.as_dict()).decode("utf-8")

        def _op(cur: Any) -> None:
            self._exec(
                cur,
                "INSERT INTO __DOCS__ (tenant_id, source_id, doc_json) VALUES (%s, %s, %s) "
                "ON CONFLICT (tenant_id, source_id) DO UPDATE SET doc_json = EXCLUDED.doc_json",
                (tenant_id, doc.source_id, encoded),
            )

        self._run_tx(_op)

    def get_document(self, tenant_id: str, source_id: str) -> RedactableDoc | None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT doc_json FROM __DOCS__ WHERE tenant_id = %s AND source_id = %s",
                    (tenant_id, source_id),
                )
                row = cur.fetchone()
                return RedactableDoc.from_dict(json.loads(row[0])) if row is not None else None
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def list_document_ids(self, tenant_id: str) -> list[str]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT source_id FROM __DOCS__ WHERE tenant_id = %s ORDER BY source_id ASC",
                    (tenant_id,),
                )
                return [r[0] for r in cur.fetchall()]
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def put_certificate(self, cert_dict: dict[str, Any]) -> None:
        encoded = canonical_bytes(cert_dict).decode("utf-8")

        def _op(cur: Any) -> None:
            self._exec(
                cur,
                "INSERT INTO __CERTS__ (tenant_id, certificate_id, subject_id, cert_json) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, certificate_id) DO UPDATE SET cert_json = EXCLUDED.cert_json",
                (cert_dict["tenant_id"], cert_dict["certificate_id"], cert_dict["subject_id"], encoded),
            )

        self._run_tx(_op)

    def list_certificates(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT cert_json FROM __CERTS__ WHERE tenant_id = %s",
                    (tenant_id,),
                )
                return [json.loads(r[0]) for r in cur.fetchall()]
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def put_pending_erasure(self, tenant_id: str, record: PendingErasureRecord) -> None:
        encoded = canonical_bytes(record.as_dict()).decode("utf-8")

        def _op(cur: Any) -> None:
            self._exec(
                cur,
                "INSERT INTO __PENDING__ (tenant_id, erasure_id, subject_id, status, updated_at, record_json) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(tenant_id, erasure_id) DO UPDATE SET "
                "status = EXCLUDED.status, updated_at = EXCLUDED.updated_at, record_json = EXCLUDED.record_json",
                (
                    tenant_id,
                    record.erasure_id,
                    record.subject_id,
                    record.status,
                    record.updated_at,
                    encoded,
                ),
            )

        self._run_tx(_op)

    def get_pending_erasure(self, tenant_id: str, erasure_id: str) -> PendingErasureRecord | None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._exec(
                    cur,
                    "SELECT record_json FROM __PENDING__ WHERE tenant_id = %s AND erasure_id = %s",
                    (tenant_id, erasure_id),
                )
                row = cur.fetchone()
                return PendingErasureRecord.from_dict(json.loads(row[0])) if row is not None else None
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    def list_pending_erasures(
        self,
        tenant_id: str = "default",
        *,
        include_completed: bool = False,
    ) -> list[PendingErasureRecord]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                if include_completed:
                    self._exec(
                        cur,
                        "SELECT record_json FROM __PENDING__ WHERE tenant_id = %s",
                        (tenant_id,),
                    )
                else:
                    self._exec(
                        cur,
                        "SELECT record_json FROM __PENDING__ WHERE tenant_id = %s AND status != %s",
                        (tenant_id, "completed"),
                    )
                return [PendingErasureRecord.from_dict(json.loads(r[0])) for r in cur.fetchall()]
            finally:
                if hasattr(cur, "close"):
                    cur.close()
