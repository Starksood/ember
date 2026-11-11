"""Typed immutable records for Ember custody ledger and verification certificates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .verify import canonical_bytes, sha256_prefixed

ORIGIN_KINDS = ("conversation", "file", "url", "api", "user_submitted", "unknown")
SOURCE_ATTESTED_FIELDS = ("source_id", "merkle_root", "flat_doc_hash", "byte_length", "leaf_count", "chunk_mode")
SOURCE_DECLARED_FIELDS = ("origin", "origin_kind", "supplied_by", "validated_by", "ingested_at")


@dataclass(frozen=True)
class SourceRecord:
    """Attested document arrival fingerprint + explicitly labelled caller declarations.

    The append-only ledger stores ONLY this fingerprint record (hashes, lengths, chunking),
    never raw source document text — allowing lawful leaf redaction in DocumentStore.
    """

    source_id: str
    merkle_root: str
    flat_doc_hash: str
    byte_length: int
    leaf_count: int
    chunk_mode: str
    ingested_at: str
    origin: str = "undeclared"
    origin_kind: str = "unknown"
    supplied_by: str = "undeclared"
    validated_by: str = "undeclared"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRecord":
        kind = str(data.get("origin_kind", "unknown"))
        if kind not in ORIGIN_KINDS:
            kind = "unknown"
        return cls(
            source_id=str(data["source_id"]),
            merkle_root=str(data["merkle_root"]),
            flat_doc_hash=str(data["flat_doc_hash"]),
            byte_length=int(data["byte_length"]),
            leaf_count=int(data["leaf_count"]),
            chunk_mode=str(data.get("chunk_mode", "sentence")),
            ingested_at=str(data["ingested_at"]),
            origin=str(data.get("origin") or "undeclared"),
            origin_kind=kind,
            supplied_by=str(data.get("supplied_by") or "undeclared"),
            validated_by=str(data.get("validated_by") or "undeclared"),
        )


@dataclass(frozen=True)
class MemoryItem:
    """Materialized memory record whose sensitive fields are encrypted in the immutable ledger."""

    item_id: str
    tenant_id: str
    subject_tokens: tuple[str, ...]
    key_kind: str  # "subject" or "record"
    key_id: str
    text_ciphertext: str
    subjects_ciphertext: str
    quote_ciphertext: str | None = None
    metadata_ciphertext: str | None = None
    receipt_meta: dict[str, Any] | None = None
    derived_from: tuple[str, ...] = ()
    adapter_record_ids: tuple[str, ...] = ()
    created_at: str = ""
    reconsent_id: str | None = None
    # Transient decrypted values held in materialized state (never serialized into ledger)
    subjects: tuple[str, ...] = ()
    text: str | None = None
    quote: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def ledger_payload(self) -> dict[str, Any]:
        """Serialize ONLY encrypted fields and structural coordinates for append-only storage."""
        return {
            "item_id": self.item_id,
            "tenant_id": self.tenant_id,
            "subject_tokens": list(self.subject_tokens),
            "key_kind": self.key_kind,
            "key_id": self.key_id,
            "text_ciphertext": self.text_ciphertext,
            "subjects_ciphertext": self.subjects_ciphertext,
            "quote_ciphertext": self.quote_ciphertext,
            "metadata_ciphertext": self.metadata_ciphertext,
            "receipt_meta": self.receipt_meta,
            "derived_from": list(self.derived_from),
            "adapter_record_ids": list(self.adapter_record_ids),
            "created_at": self.created_at,
            "reconsent_id": self.reconsent_id,
        }

    @classmethod
    def from_ledger_payload(
        cls,
        data: dict[str, Any],
        *,
        subjects: tuple[str, ...] = (),
        text: str | None = None,
        quote: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MemoryItem":
        resolved_meta = dict(metadata) if metadata is not None else dict(data.get("metadata") or {})
        return cls(
            item_id=str(data["item_id"]),
            tenant_id=str(data["tenant_id"]),
            subject_tokens=tuple(str(t) for t in data["subject_tokens"]),
            key_kind=str(data["key_kind"]),
            key_id=str(data["key_id"]),
            text_ciphertext=str(data["text_ciphertext"]),
            subjects_ciphertext=str(data["subjects_ciphertext"]),
            quote_ciphertext=data.get("quote_ciphertext"),
            metadata_ciphertext=data.get("metadata_ciphertext"),
            receipt_meta=dict(data["receipt_meta"]) if data.get("receipt_meta") else None,
            derived_from=tuple(str(d) for d in data.get("derived_from", ())),
            adapter_record_ids=tuple(str(r) for r in data.get("adapter_record_ids", ())),
            created_at=str(data.get("created_at", "")),
            reconsent_id=data.get("reconsent_id"),
            subjects=subjects,
            text=text,
            quote=quote,
            metadata=resolved_meta,
        )


@dataclass(frozen=True)
class LedgerEvent:
    """One sealed event row in the append-only SHA-256 hash chain."""

    seq: int
    tenant_id: str
    event_id: str
    kind: str
    committed_at: str
    payload: dict[str, Any]
    prev_hash: str
    event_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "tenant_id": self.tenant_id,
            "event_id": self.event_id,
            "kind": self.kind,
            "committed_at": self.committed_at,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "event_hash": self.event_hash,
        }


@dataclass(frozen=True)
class ClosureManifest:
    """Exact structural closure computed prior to executing an erasure."""

    subject_id: str
    item_ids: tuple[str, ...]
    derived_invalidated: tuple[str, ...]
    source_leaves_redacted: dict[str, list[int]]
    adapter_record_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "item_ids": list(self.item_ids),
            "derived_invalidated": list(self.derived_invalidated),
            "source_leaves_redacted": {
                k: sorted(v) for k, v in sorted(self.source_leaves_redacted.items())
            },
            "adapter_record_ids": list(self.adapter_record_ids),
        }


@dataclass(frozen=True)
class ErasureCertificate:
    """Ed25519-signed erasure attestation with honest trust-boundary metadata."""

    certificate_id: str
    tenant_id: str
    subject_id: str
    subject_tokens: tuple[str, ...]
    state_hash_before: str
    state_hash_after: str
    ledger_tail_hash: str
    closure: dict[str, Any]
    probe_count: int
    battery_all_abstained: bool
    cipher: str
    key_destroyed: bool
    record_keys_destroyed: int
    key_destroyed_at: str | None
    issued_at: str
    trust_boundary: dict[str, Any]
    anchor: dict[str, str]
    signer_scheme: str = ""
    public_key: str = ""
    signature: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "tenant_id": self.tenant_id,
            "subject_id": self.subject_id,
            "subject_tokens": list(self.subject_tokens),
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "ledger_tail_hash": self.ledger_tail_hash,
            "closure": self.closure,
            "probe_count": self.probe_count,
            "battery_all_abstained": self.battery_all_abstained,
            "cipher": self.cipher,
            "key_destroyed": self.key_destroyed,
            "record_keys_destroyed": self.record_keys_destroyed,
            "key_destroyed_at": self.key_destroyed_at,
            "issued_at": self.issued_at,
            "trust_boundary": self.trust_boundary,
            "anchor": self.anchor,
            "signer_scheme": self.signer_scheme,
            "public_key": self.public_key,
            "signature": self.signature,
        }

    def digest(self) -> str:
        return sha256_prefixed(canonical_bytes(self.as_dict()))
