"""CustodyVault: Ember's unified developer API combining ledger, keystore, Merkle redaction, signing, and adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

from .adapters import AdapterHit, InMemoryStoreAdapter, MemoryStoreAdapter, _content_tokens, _tokenize
from .crypto import CIPHER_NAME, build_aad, encrypt_text
from .keystore import FileKeyStore, KeyStore, MemoryKeyStore
from .ledger import SQLiteLedger
from .merkle import ChunkMode, ChunkerFn, RedactableDoc
from .receipts import Receipt, mint_redactable_receipt
from .records import ErasureCertificate, MemoryItem, SourceRecord
from .reducer import (
    EmberState,
    ErasureIncomplete,
    InvariantViolation,
    TombstonedSubjectError,
    compute_closure,
)
from .signing import Ed25519Signer, NullAnchor, TimestampAnchor, sign_certificate_dict
from .verify import canonical_bytes, sha256_prefixed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class QueryHit:
    item_id: str
    record_id: str
    text: str
    score: float
    subjects: tuple[str, ...]
    receipt: Receipt | None
    receipt_verified: bool | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: tuple[QueryHit, ...]
    filtered_orphans: int = 0

    @property
    def abstained(self) -> bool:
        return len(self.hits) == 0


@dataclass(frozen=True)
class RememberResult:
    item: MemoryItem
    receipt: Receipt | None
    adapter_record_ids: tuple[str, ...]
    event_hash: str


class CustodyVault:
    """Stateful cryptographic custody facade for AI memory and RAG stores."""

    def __init__(
        self,
        *,
        ledger: SQLiteLedger,
        keystore: KeyStore,
        signer: Ed25519Signer,
        tenant_id: str = "default",
        adapter: MemoryStoreAdapter | None = None,
        anchor: TimestampAnchor | None = None,
        keys_separated_from_ledger: bool = False,
    ) -> None:
        self.ledger = ledger
        self.keystore = keystore
        self.signer = signer
        self.tenant_id = tenant_id
        self.adapter: MemoryStoreAdapter = adapter if adapter is not None else InMemoryStoreAdapter()
        self.anchor: TimestampAnchor = anchor if anchor is not None else NullAnchor()
        self.keys_separated_from_ledger = keys_separated_from_ledger
        self.state: EmberState = self.ledger.replay(self.tenant_id, keystore=self.keystore)

    @classmethod
    def open(
        cls,
        data_dir: str | Path,
        *,
        key_dir: str | Path | None = None,
        tenant_id: str = "default",
        adapter: MemoryStoreAdapter | None = None,
        signer: Ed25519Signer | None = None,
        anchor: TimestampAnchor | None = None,
    ) -> "CustodyVault":
        """Open or create a persistent vault on disk.

        For real deployments, pass an explicit `key_dir` outside `data_dir` (or pass a KMS-backed
        `KeyStore`) so ledger backups never contain live content decryption keys.
        """
        data_path = Path(data_dir).resolve()
        data_path.mkdir(parents=True, exist_ok=True)
        resolved_key_dir = Path(key_dir).resolve() if key_dir is not None else (data_path / "keys")
        separated = key_dir is not None and not resolved_key_dir.is_relative_to(data_path)

        ledger = SQLiteLedger(data_path / "ledger.db")
        keystore = FileKeyStore(resolved_key_dir / "keyring.json")
        resolved_signer = signer if signer is not None else Ed25519Signer.load_or_create(resolved_key_dir / "ed25519.json")
        return cls(
            ledger=ledger,
            keystore=keystore,
            signer=resolved_signer,
            tenant_id=tenant_id,
            adapter=adapter,
            anchor=anchor,
            keys_separated_from_ledger=separated,
        )

    @classmethod
    def ephemeral(
        cls,
        *,
        tenant_id: str = "default",
        adapter: MemoryStoreAdapter | None = None,
        signer: Ed25519Signer | None = None,
        anchor: TimestampAnchor | None = None,
    ) -> "CustodyVault":
        """Open a clean in-memory vault for fast tests, notebooks, and demos."""
        return cls(
            ledger=SQLiteLedger(":memory:"),
            keystore=MemoryKeyStore(),
            signer=signer if signer is not None else Ed25519Signer.generate(),
            tenant_id=tenant_id,
            adapter=adapter,
            anchor=anchor,
            keys_separated_from_ledger=True,
        )

    def close(self) -> None:
        self.ledger.close()

    def register_source(
        self,
        source_id: str,
        text: str,
        *,
        mode: ChunkMode | ChunkerFn = "sentence",
        origin: str = "",
        origin_kind: str = "unknown",
        supplied_by: str = "",
        validated_by: str = "",
        ingested_at: str | None = None,
        event_id: str | None = None,
    ) -> tuple[SourceRecord, RedactableDoc]:
        """Hash, Merkle-index, store mutable leaves, and commit an attested SourceRecord to the chain."""
        doc = RedactableDoc.from_text(text, source_id=source_id, mode=mode)
        raw_bytes = text.encode("utf-8")
        ts = ingested_at or _utc_now()
        record = SourceRecord(
            source_id=source_id,
            merkle_root=doc.root(),
            flat_doc_hash=sha256_prefixed(raw_bytes),
            byte_length=len(raw_bytes),
            leaf_count=len(doc.entries),
            chunk_mode=doc.chunk_mode,
            ingested_at=ts,
            origin=origin or "undeclared",
            origin_kind=origin_kind,
            supplied_by=supplied_by or "undeclared",
            validated_by=validated_by or "undeclared",
        )
        ev_id = event_id or f"src:{source_id}:{uuid.uuid4().hex[:8]}"
        _, _, next_state = self.ledger.append(
            tenant_id=self.tenant_id,
            event_id=ev_id,
            kind="ADD_SOURCE",
            committed_at=ts,
            payload={"source": record.as_dict()},
            current_state=self.state,
            keystore=self.keystore,
            validate=True,
        )
        assert next_state is not None
        self.ledger.put_document(self.tenant_id, doc)
        self.state = next_state
        return record, doc

    def _effective_owners(self, subjects: Sequence[str], derived_from: Sequence[str]) -> set[str]:
        owners = {s for s in subjects if s}
        visited: set[str] = set()
        stack = list(derived_from)
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            parent = self.state.items.get(pid)
            if parent is None:
                raise InvariantViolation(f"cannot derive from unknown or erased parent item {pid!r}")
            owners.update(parent.subjects)
            stack.extend(parent.derived_from)
        return owners

    def remember(
        self,
        text: str,
        *,
        subjects: Iterable[str],
        source_id: str | None = None,
        quote: str | None = None,
        leaf_index: int | None = None,
        derived_from: Iterable[str] = (),
        aliases: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        reconsent_id: str | None = None,
        item_id: str | None = None,
        committed_at: str | None = None,
        event_id: str | None = None,
    ) -> RememberResult:
        """Encrypt, bind provenance receipt, commit to ledger, and upsert into the backing store."""
        subj_tuple = tuple(dict.fromkeys(str(s) for s in subjects if str(s).strip()))
        if not subj_tuple:
            raise ValueError("remember() requires at least one non-empty subject ID")
        derived_tuple = tuple(dict.fromkeys(str(d) for d in derived_from if str(d).strip()))
        alias_tuple = tuple(dict.fromkeys(str(a) for a in aliases if str(a).strip()))

        all_tokens = tuple(
            sorted(
                {
                    tok
                    for tok in (self.keystore.tombstone_token(val) for val in (*subj_tuple, *alias_tuple))
                    if tok
                }
            )
        )
        tomb_overlap = set(all_tokens) & self.state.tombstoned_tokens
        if tomb_overlap and not reconsent_id:
            raise TombstonedSubjectError(
                f"refusing ingest: subject has an active erasure tombstone and no reconsent_id"
            )

        effective_owners = self._effective_owners(subj_tuple, derived_tuple)
        resolved_item_id = item_id or f"item_{uuid.uuid4().hex[:12]}"
        ts = committed_at or _utc_now()

        receipt: Receipt | None = None
        receipt_meta: dict[str, Any] | None = None
        plain_quote: str | None = quote
        if source_id is not None:
            doc = self.ledger.get_document(self.tenant_id, source_id)
            if doc is None:
                raise InvariantViolation(f"unknown source_id {source_id!r}; call register_source() first")
            src_rec = self.state.sources.get(source_id)
            target_quote = quote if quote is not None else text
            receipt = mint_redactable_receipt(
                doc,
                target_quote,
                leaf_index=leaf_index,
                source_id=source_id,
            )
            if src_rec is not None:
                receipt = Receipt(
                    source_id=receipt.source_id,
                    quote=receipt.quote,
                    flat_doc_hash=src_rec.flat_doc_hash,
                    byte_start=receipt.byte_start,
                    byte_end=receipt.byte_end,
                    merkle_root=receipt.merkle_root,
                    leaf_index=receipt.leaf_index,
                    leaf_count=receipt.leaf_count,
                    proof=receipt.proof,
                )
            plain_quote = receipt.quote
            receipt_meta = {
                "source_id": receipt.source_id,
                "flat_doc_hash": receipt.flat_doc_hash,
                "byte_start": receipt.byte_start,
                "byte_end": receipt.byte_end,
                "merkle_root": receipt.merkle_root,
                "leaf_index": receipt.leaf_index,
                "leaf_count": receipt.leaf_count,
                "proof": [list(step) for step in receipt.proof],
            }

        # Key routing: direct single-subject items use opaque subject CEK (`subject_ref`);
        # derived/shared items use tracked DEK linked to every contributing opaque subject ref
        # so erasing ANY contributor shreds the derived item cleanly without storing raw names.
        owner_refs = {self.keystore.subject_ref(owner) for owner in effective_owners}
        if len(owner_refs) == 1 and not derived_tuple:
            key_kind = "subject"
            key_id = next(iter(owner_refs))
            cek = self.keystore.get_or_create_subject_key(key_id)
        else:
            key_kind = "record"
            key_id = resolved_item_id
            cek = self.keystore.create_record_key(resolved_item_id, owner_refs)

        key_ref = f"{key_kind}:{key_id}"
        subjects_ct = encrypt_text(
            cek,
            canonical_bytes(list(subj_tuple)).decode("utf-8"),
            aad=build_aad(tenant_id=self.tenant_id, record_id=resolved_item_id, field="subjects", key_ref=key_ref),
        )
        text_ct = encrypt_text(
            cek,
            text,
            aad=build_aad(tenant_id=self.tenant_id, record_id=resolved_item_id, field="text", key_ref=key_ref),
        )
        quote_ct: str | None = None
        if plain_quote is not None:
            quote_ct = encrypt_text(
                cek,
                plain_quote,
                aad=build_aad(tenant_id=self.tenant_id, record_id=resolved_item_id, field="quote", key_ref=key_ref),
            )

        meta = dict(metadata or {})
        ext_ids = self.adapter.upsert(
            item_id=resolved_item_id,
            text=text,
            subjects=tuple(sorted(effective_owners)),
            metadata=meta,
        )

        item_obj = MemoryItem(
            item_id=resolved_item_id,
            tenant_id=self.tenant_id,
            subject_tokens=all_tokens,
            key_kind=key_kind,
            key_id=key_id,
            text_ciphertext=text_ct,
            subjects_ciphertext=subjects_ct,
            quote_ciphertext=quote_ct,
            receipt_meta=receipt_meta,
            derived_from=derived_tuple,
            adapter_record_ids=tuple(ext_ids),
            created_at=ts,
            metadata=meta,
            reconsent_id=reconsent_id,
            subjects=subj_tuple,
            text=text,
            quote=plain_quote,
        )

        ev_id = event_id or f"item:{resolved_item_id}"
        try:
            ev, _, next_state = self.ledger.append(
                tenant_id=self.tenant_id,
                event_id=ev_id,
                kind="ADD_ITEM",
                committed_at=ts,
                payload={"item": item_obj.ledger_payload()},
                current_state=self.state,
                keystore=self.keystore,
                validate=True,
            )
        except BaseException:
            if ext_ids:
                self.adapter.delete(ext_ids)
            raise

        assert next_state is not None
        self.state = next_state
        return RememberResult(
            item=next_state.items[resolved_item_id],
            receipt=receipt,
            adapter_record_ids=tuple(ext_ids),
            event_hash=ev.event_hash,
        )

    def receipt_for(self, item: MemoryItem) -> Receipt | None:
        if item.receipt_meta is None or item.quote is None or item.quote == "[erased]":
            return None
        meta = item.receipt_meta
        return Receipt(
            source_id=str(meta["source_id"]),
            quote=item.quote,
            flat_doc_hash=str(meta.get("flat_doc_hash", "")),
            byte_start=int(meta["byte_start"]),
            byte_end=int(meta["byte_end"]),
            merkle_root=str(meta["merkle_root"]),
            leaf_index=int(meta["leaf_index"]),
            leaf_count=int(meta["leaf_count"]),
            proof=tuple((str(s), str(h)) for s, h in meta.get("proof", ())),
        )

    def verify_item_receipt(self, item_id: str) -> bool:
        item = self.state.items.get(item_id)
        if item is None:
            return False
        rec = self.receipt_for(item)
        if rec is None:
            return False
        doc = self.ledger.get_document(self.tenant_id, rec.source_id)
        if doc is None:
            return False
        return rec.verify_redactable(doc)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        verify_receipts: bool = True,
    ) -> SearchResult:
        """Query wrapped backend store, enforce live custody membership, and attach verified receipts."""
        raw_hits = self.adapter.search(query, limit=max(limit * 2, limit), filters=filters)
        ext_map = self.state.adapter_id_map()
        hits: list[QueryHit] = []
        filtered_orphans = 0

        for hit in raw_hits:
            item_id = ext_map.get(hit.record_id) or hit.metadata.get("ember_item_id")
            if not item_id or item_id not in self.state.items:
                filtered_orphans += 1
                continue
            item = self.state.items[item_id]
            rec = self.receipt_for(item)
            rec_ok: bool | None = None
            if verify_receipts and rec is not None:
                doc = self.ledger.get_document(self.tenant_id, rec.source_id)
                rec_ok = rec.verify_redactable(doc) if doc is not None else False
            hits.append(
                QueryHit(
                    item_id=item.item_id,
                    record_id=hit.record_id,
                    text=item.text if item.text is not None else hit.text,
                    score=hit.score,
                    subjects=item.subjects,
                    receipt=rec,
                    receipt_verified=rec_ok,
                    metadata=dict(hit.metadata),
                )
            )
            if len(hits) >= limit:
                break

        return SearchResult(query=query, hits=tuple(hits), filtered_orphans=filtered_orphans)

    def _probe_raw_leaks(
        self,
        subject_id: str,
        aliases: Sequence[str],
        doomed_item_ids: set[str],
        doomed_adapter_ids: set[str],
        probes: Sequence[str],
        markers: Sequence[str],
    ) -> list[str]:
        """Inspect raw adapter AND live materialized state after erasure to catch ANY residue."""
        leaking: list[str] = []
        lower_markers = [m.lower() for m in markers if m]
        # Defensive check against live adapter directly (do NOT mask stale vendor entries)
        for probe in probes:
            leaked = False
            raw_hits = self.adapter.search(probe, limit=25)
            for hit in raw_hits:
                if hit.record_id in doomed_adapter_ids:
                    leaked = True
                    break
                if hit.metadata.get("ember_item_id") in doomed_item_ids:
                    leaked = True
                    break
                if any(tok in hit.text.lower() for tok in lower_markers):
                    leaked = True
                    break
            if not leaked:
                q_toks = _content_tokens(probe)
                for item in self.state.items.values():
                    text_val = item.text or ""
                    if q_toks & _tokenize(text_val):
                        if subject_id in item.subjects or any(tok in text_val.lower() for tok in lower_markers):
                            leaked = True
                            break
            if leaked:
                leaking.append(probe)
        return leaking

    def erase(
        self,
        subject_id: str,
        *,
        aliases: Iterable[str] = (),
        probes: Iterable[str] = (),
        markers: Iterable[str] = (),
        leaf_predicate: Callable[[str], bool] | None = None,
        committed_at: str | None = None,
    ) -> ErasureCertificate:
        """Execute irreversible cascade erasure, key shredding, doc redaction, and probe certification."""
        alias_list = [str(a) for a in aliases if str(a).strip()]
        probe_list = [str(p) for p in probes if str(p).strip()]
        marker_list = list(dict.fromkeys([subject_id, *alias_list, *(str(m) for m in markers if str(m).strip())]))

        before_hash = self.state.digest()
        subject_tokens = tuple(
            sorted(
                {
                    tok
                    for tok in (self.keystore.tombstone_token(val) for val in (subject_id, *alias_list))
                    if tok
                }
            )
        )

        # Scan source docs for explicit leaf predicate if requested (in addition to cited closure leaves)
        extra_leaves: dict[str, list[int]] = {}
        if leaf_predicate is not None:
            for sid in self.ledger.list_document_ids(self.tenant_id):
                doc = self.ledger.get_document(self.tenant_id, sid)
                if doc is None:
                    continue
                matched = [
                    e["index"]
                    for e in doc.entries
                    if not e.get("redacted") and leaf_predicate(e["text"])
                ]
                if matched:
                    extra_leaves[sid] = matched

        closure = compute_closure(
            self.state,
            subject_id,
            subject_tokens=subject_tokens,
            extra_source_leaves=extra_leaves or None,
        )

        ts = committed_at or _utc_now()
        erase_ev_id = f"erase:{subject_id}:{uuid.uuid4().hex[:8]}"
        ev, _, next_state = self.ledger.append(
            tenant_id=self.tenant_id,
            event_id=erase_ev_id,
            kind="ERASE_SUBJECT",
            committed_at=ts,
            payload={
                "subject_tokens": list(subject_tokens),
                "closure": closure.as_dict(),
            },
            current_state=self.state,
            keystore=self.keystore,
            validate=True,
        )
        assert next_state is not None
        self.state = next_state

        # Redact leaves in mutable DocumentStore (bystander leaves and proofs survive untouched)
        for sid, leaf_indices in closure.source_leaves_redacted.items():
            doc = self.ledger.get_document(self.tenant_id, sid)
            if doc is not None:
                redacted_doc = doc.redact_indices(set(leaf_indices))
                self.ledger.put_document(self.tenant_id, redacted_doc)

        # Cascade deletion in backing store adapter (external Mem0 / vector DB IDs)
        if closure.adapter_record_ids:
            self.adapter.delete(closure.adapter_record_ids)

        # Crypto-shred subject CEK + shared/derived record DEKs (covering both direct ID and opaque refs)
        targets_to_shred = list(dict.fromkeys([subject_id, *alias_list]))
        key_destroyed = False
        record_keys_destroyed = 0
        for target in targets_to_shred:
            rep_direct = self.keystore.shred_subject(target)
            rep_ref = self.keystore.shred_subject(self.keystore.subject_ref(target))
            key_destroyed = key_destroyed or rep_direct.subject_key_destroyed or rep_ref.subject_key_destroyed
            record_keys_destroyed += rep_direct.record_keys_destroyed + rep_ref.record_keys_destroyed
        destroyed_at = _utc_now()

        # Verify probes against raw backing store & surviving state BEFORE issuing certificate
        leaking = self._probe_raw_leaks(
            subject_id,
            alias_list,
            set(closure.item_ids),
            set(closure.adapter_record_ids),
            probe_list,
            marker_list,
        )
        if leaking:
            raise ErasureIncomplete(subject_id, leaking, closure)

        after_hash = self.state.digest()
        issued_at = _utc_now()
        cert_id = f"cert_{uuid.uuid4().hex[:12]}"

        anchor_digest = sha256_prefixed(
            json.dumps(
                {
                    "certificate_id": cert_id,
                    "before": before_hash,
                    "after": after_hash,
                    "tail": ev.event_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        anchor_receipt = self.anchor.anchor(anchor_digest)

        trust_boundary = {
            "adversary_checkable_signature": True,
            "externally_timestamped": anchor_receipt.scheme != "none",
            "keys_separated_from_ledger": self.keys_separated_from_ledger,
            "live_store_covered": True,
            "pre_erasure_backups_covered": False,
            "notes": (
                "Certifies live projection closure, backing adapter ID deletion, salted Merkle leaf redaction, "
                "and AES-256-GCM key shredding on this store. Backups created prior to erasure that copied "
                "unseparated keys remain outside technical coverage."
            ),
        }

        unsigned = {
            "certificate_id": cert_id,
            "tenant_id": self.tenant_id,
            "subject_id": subject_id,
            "subject_tokens": list(subject_tokens),
            "state_hash_before": before_hash,
            "state_hash_after": after_hash,
            "ledger_tail_hash": ev.event_hash,
            "closure": closure.as_dict(),
            "probe_count": len(probe_list),
            "battery_all_abstained": True,
            "cipher": CIPHER_NAME,
            "key_destroyed": key_destroyed or record_keys_destroyed > 0,
            "record_keys_destroyed": record_keys_destroyed,
            "key_destroyed_at": destroyed_at,
            "issued_at": issued_at,
            "trust_boundary": trust_boundary,
            "anchor": anchor_receipt.as_dict(),
        }
        signed_dict = sign_certificate_dict(unsigned, self.signer)
        cert = ErasureCertificate(
            certificate_id=signed_dict["certificate_id"],
            tenant_id=signed_dict["tenant_id"],
            subject_id=signed_dict["subject_id"],
            subject_tokens=tuple(signed_dict["subject_tokens"]),
            state_hash_before=signed_dict["state_hash_before"],
            state_hash_after=signed_dict["state_hash_after"],
            ledger_tail_hash=signed_dict["ledger_tail_hash"],
            closure=signed_dict["closure"],
            probe_count=signed_dict["probe_count"],
            battery_all_abstained=signed_dict["battery_all_abstained"],
            cipher=signed_dict["cipher"],
            key_destroyed=signed_dict["key_destroyed"],
            record_keys_destroyed=signed_dict["record_keys_destroyed"],
            key_destroyed_at=signed_dict["key_destroyed_at"],
            issued_at=signed_dict["issued_at"],
            trust_boundary=signed_dict["trust_boundary"],
            anchor=signed_dict["anchor"],
            signer_scheme=signed_dict["signer_scheme"],
            public_key=signed_dict["public_key"],
            signature=signed_dict["signature"],
        )
        self.ledger.put_certificate(cert.as_dict())
        return cert

    def replay(self, *, keystore: KeyStore | None = None) -> EmberState:
        """Reconstruct materialized custody state from zero via ledger replay."""
        return self.ledger.replay(self.tenant_id, keystore=keystore if keystore is not None else self.keystore)

    def export_bundle(self, *, include_ledger: bool = True) -> dict[str, Any]:
        """Export self-contained redacted verification bundle checkable offline by `ember.verify`."""
        docs: dict[str, Any] = {}
        for sid in self.ledger.list_document_ids(self.tenant_id):
            doc = self.ledger.get_document(self.tenant_id, sid)
            if doc is not None:
                docs[sid] = doc.as_dict()

        receipts: list[dict[str, Any]] = []
        for item in self.state.items.values():
            rec = self.receipt_for(item)
            if rec is not None and rec.redaction_safe:
                receipts.append(rec.as_dict())

        bundle: dict[str, Any] = {
            "format": "ember-bundle/v1",
            "tenant_id": self.tenant_id,
            "documents": docs,
            "receipts": receipts,
            "certificates": self.ledger.list_certificates(self.tenant_id),
        }
        if include_ledger:
            bundle["ledger_events"] = [ev.as_dict() for ev in self.ledger.events(self.tenant_id)]
        return bundle
