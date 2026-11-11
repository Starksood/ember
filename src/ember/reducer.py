"""Pure deterministic state reduction and exact erasure closure computation.

Zero I/O, zero wall-clock calls, zero randomness. Given an identical sequence of
validated `LedgerEvent`s (and the same `KeyStore` decryption view), `apply_event`
reconstructs a byte-identical `EmberState`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any

from .crypto import build_aad, decrypt_text
from .keystore import KeyStore
from .records import ClosureManifest, LedgerEvent, MemoryItem, SourceRecord
from .verify import canonical_bytes, sha256_prefixed

EVENT_KINDS = frozenset({"ADD_SOURCE", "ADD_ITEM", "ERASE_SUBJECT"})


class ReductionError(ValueError):
    """Base error raised on invalid state transition or invariant failure."""


class InvariantViolation(ReductionError):
    """Raised when an event references missing parents/sources or collides on IDs."""


class TombstonedSubjectError(ReductionError):
    """Raised fail-closed when attempting to ingest data for a previously erased subject."""


class ErasureIncomplete(RuntimeError):
    """Raised when post-erasure probes still leak the erased subject; withholds certificate."""

    def __init__(self, subject_id: str, leaking_probes: list[str], closure: ClosureManifest) -> None:
        self.subject_id = subject_id
        self.leaking_probes = leaking_probes
        self.closure = closure
        super().__init__(
            f"erasure of {subject_id!r} incomplete: {len(leaking_probes)} probe(s) still leaked "
            f"({', '.join(map(repr, leaking_probes[:4]))}). Certificate withheld."
        )


def _frozen(mapping: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class EmberState:
    """Immutable materialized custody projection reconstructed solely from the ledger."""

    version: int = 0
    sources: Mapping[str, SourceRecord] = field(default_factory=_frozen)
    items: Mapping[str, MemoryItem] = field(default_factory=_frozen)
    tombstoned_tokens: frozenset[str] = frozenset()

    @classmethod
    def empty(cls) -> "EmberState":
        return cls()

    def structural_dict(self) -> dict[str, Any]:
        """Serializable representation of structural custody state (never stores plaintext keys)."""
        return {
            "version": self.version,
            "sources": {k: v.as_dict() for k, v in sorted(self.sources.items())},
            "items": {
                k: {
                    **v.ledger_payload(),
                    "subjects": list(v.subjects),
                    "text": v.text,
                    "quote": v.quote,
                }
                for k, v in sorted(self.items.items())
            },
            "tombstoned_tokens": sorted(self.tombstoned_tokens),
        }

    def digest(self) -> str:
        return sha256_prefixed(canonical_bytes(self.structural_dict()))

    def adapter_id_map(self) -> dict[str, str]:
        """Map external adapter record IDs back to Ember `item_id`."""
        out: dict[str, str] = {}
        for item in self.items.values():
            for ext_id in item.adapter_record_ids:
                out[ext_id] = item.item_id
        return out


def compute_closure(
    state: EmberState,
    subject_id: str,
    *,
    subject_tokens: tuple[str, ...] = (),
    extra_source_leaves: dict[str, list[int]] | None = None,
) -> ClosureManifest:
    """Compute exact closure over subjects, keyed tokens, transitive derived items, and source leaves."""
    token_set = set(subject_tokens)
    direct_ids: set[str] = set()
    for item_id, item in state.items.items():
        if subject_id in item.subjects or (token_set and bool(token_set & set(item.subject_tokens))):
            direct_ids.add(item_id)

    doomed_ids = set(direct_ids)
    derived_ids: set[str] = set()
    while True:
        added = {
            item_id
            for item_id, item in state.items.items()
            if item_id not in doomed_ids and any(parent in doomed_ids for parent in item.derived_from)
        }
        if not added:
            break
        derived_ids |= added
        doomed_ids |= added

    leaf_redactions: dict[str, set[int]] = {}
    if extra_source_leaves:
        for sid, indices in extra_source_leaves.items():
            leaf_redactions.setdefault(sid, set()).update(indices)

    adapter_ids: list[str] = []
    seen_adapter: set[str] = set()
    for item_id in sorted(doomed_ids):
        item = state.items[item_id]
        if item.receipt_meta:
            sid = item.receipt_meta.get("source_id")
            leaf_idx = item.receipt_meta.get("leaf_index")
            if isinstance(sid, str) and isinstance(leaf_idx, int) and leaf_idx >= 0:
                leaf_redactions.setdefault(sid, set()).add(leaf_idx)
        for ext_id in item.adapter_record_ids:
            if ext_id not in seen_adapter:
                seen_adapter.add(ext_id)
                adapter_ids.append(ext_id)

    return ClosureManifest(
        subject_id=subject_id,
        item_ids=tuple(sorted(doomed_ids)),
        derived_invalidated=tuple(sorted(derived_ids)),
        source_leaves_redacted={k: sorted(v) for k, v in sorted(leaf_redactions.items())},
        adapter_record_ids=tuple(adapter_ids),
    )


def decrypt_item_payload(
    payload: dict[str, Any],
    keystore: KeyStore | None,
) -> tuple[tuple[str, ...], str, str | None]:
    """Decrypt subjects, text, and optional evidence quote via KeyStore; yield `[erased]` if shredded."""
    key_kind = str(payload["key_kind"])
    key_id = str(payload["key_id"])
    key_ref = f"{key_kind}:{key_id}"
    tenant_id = str(payload["tenant_id"])
    item_id = str(payload["item_id"])
    key = keystore.key_for(key_kind, key_id) if keystore is not None else None

    subj_aad = build_aad(tenant_id=tenant_id, record_id=item_id, field="subjects", key_ref=key_ref)
    plain_subj_json = decrypt_text(key, str(payload["subjects_ciphertext"]), aad=subj_aad, strict=False)
    if plain_subj_json == "[erased]":
        plain_subjects: tuple[str, ...] = ()
    else:
        try:
            loaded = json.loads(plain_subj_json)
            plain_subjects = tuple(str(x) for x in loaded) if isinstance(loaded, list) else ()
        except ValueError:
            plain_subjects = ()

    text_aad = build_aad(tenant_id=tenant_id, record_id=item_id, field="text", key_ref=key_ref)
    plain_text = decrypt_text(key, str(payload["text_ciphertext"]), aad=text_aad, strict=False)

    quote_ct = payload.get("quote_ciphertext")
    plain_quote: str | None = None
    if isinstance(quote_ct, str) and quote_ct:
        quote_aad = build_aad(tenant_id=tenant_id, record_id=item_id, field="quote", key_ref=key_ref)
        plain_quote = decrypt_text(key, quote_ct, aad=quote_aad, strict=False)
    return plain_subjects, plain_text, plain_quote


def apply_event(
    state: EmberState,
    event: LedgerEvent,
    *,
    keystore: KeyStore | None = None,
    decrypted_override: tuple[tuple[str, ...], str | None, str | None] | None = None,
) -> EmberState:
    """Validate and apply one committed/replayed `LedgerEvent` to produce the next `EmberState`."""
    if event.kind not in EVENT_KINDS:
        raise InvariantViolation(f"unsupported ledger event kind: {event.kind!r}")

    if event.kind == "ADD_SOURCE":
        rec = SourceRecord.from_dict(event.payload["source"])
        if rec.source_id in state.sources:
            raise InvariantViolation(f"source already registered: {rec.source_id!r}")
        sources = dict(state.sources)
        sources[rec.source_id] = rec
        return EmberState(
            version=state.version + 1,
            sources=_frozen(sources),
            items=state.items,
            tombstoned_tokens=state.tombstoned_tokens,
        )

    if event.kind == "ADD_ITEM":
        item_data = event.payload["item"]
        item_id = str(item_data["item_id"])
        if item_id in state.items:
            raise InvariantViolation(f"item already exists: {item_id!r}")
        item_tokens = {t for t in item_data.get("subject_tokens", ()) if t}
        overlap = item_tokens & state.tombstoned_tokens
        if overlap and not item_data.get("reconsent_id"):
            raise TombstonedSubjectError(
                f"refusing ingest for tombstoned subject token(s) without explicit reconsent_id"
            )
        receipt_meta = item_data.get("receipt_meta")
        if isinstance(receipt_meta, dict) and receipt_meta.get("source_id"):
            sid = receipt_meta["source_id"]
            if sid not in state.sources:
                raise InvariantViolation(f"receipt references unknown source_id: {sid!r}")
        for parent_id in item_data.get("derived_from", ()):
            if parent_id not in state.items:
                raise InvariantViolation(f"item references unknown/erased parent {parent_id!r}")

        if decrypted_override is not None:
            plain_subjects, plain_text, plain_quote = decrypted_override
        else:
            plain_subjects, plain_text, plain_quote = decrypt_item_payload(item_data, keystore)
        item = MemoryItem.from_ledger_payload(
            item_data,
            subjects=plain_subjects,
            text=plain_text,
            quote=plain_quote,
        )
        items = dict(state.items)
        items[item.item_id] = item
        return EmberState(
            version=state.version + 1,
            sources=state.sources,
            items=_frozen(items),
            tombstoned_tokens=state.tombstoned_tokens,
        )

    if event.kind == "ERASE_SUBJECT":
        closure = event.payload["closure"]
        doomed = set(closure.get("item_ids", ()))
        tokens = {str(t) for t in event.payload.get("subject_tokens", ()) if t}
        items = {k: v for k, v in state.items.items() if k not in doomed}
        return EmberState(
            version=state.version + 1,
            sources=state.sources,
            items=_frozen(items),
            tombstoned_tokens=state.tombstoned_tokens | frozenset(tokens),
        )

    raise InvariantViolation(f"unhandled event kind: {event.kind!r}")
