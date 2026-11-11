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

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from .keystore import KeyStore
from .merkle import RedactableDoc
from .records import LedgerEvent
from .reducer import EmberState, apply_event
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


@dataclass(frozen=True)
class AppendReceipt:
    seq: int
    event_id: str
    event_hash: str
    idempotent: bool


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
"""


class SQLiteLedger:
    """Thread-safe durable SQLite append-only ledger + mutable redactable doc store."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
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
    ) -> tuple[LedgerEvent, bool, EmberState | None]:
        """Atomically append one event, checking idempotency, optional reduction invariants, and chaining."""
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
                    base = current_state if current_state is not None else self.replay(tenant_id, keystore=keystore)
                    next_state = apply_event(base, ev, keystore=keystore)

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

    def replay(self, tenant_id: str = "default", *, keystore: KeyStore | None = None) -> EmberState:
        """Verify ledger chain and fold every committed event into an `EmberState`."""
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
