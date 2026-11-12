"""First-class erasure coverage & lineage completeness auditing (`vault.audit_coverage()`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .crypto import ERASED_TOMBSTONE
from .records import LedgerEvent
from .reducer import EmberState, compute_closure


@dataclass(frozen=True)
class CoverageReport:
    """Falsifiable pre-erasure readiness and lineage completeness report."""

    tenant_id: str
    subject_id: str | None
    verdict: str  # "complete" | "bounded_with_warnings" | "incomplete"
    summary: str
    ledger_chain_ok: bool
    source_count: int
    sources_linked_count: int
    orphan_source_ids: tuple[str, ...]
    source_coverage_rate: float
    live_item_count: int
    attributed_item_count: int
    unattributed_item_count: int
    unattributed_item_ids: tuple[str, ...]
    traced_item_count: int
    derived_item_count: int
    derived_dependency_count: int
    lineage_capture_rate: float
    verified_stores: tuple[dict[str, Any], ...]
    unverified_locations: tuple[str, ...]
    pending_erasure_count: int
    pending_erasures: tuple[dict[str, Any], ...]
    subject_preview: dict[str, Any] | None = None
    recommendations: tuple[str, ...] = ()

    @property
    def erasure_certifiable(self) -> bool:
        return self.verdict != "incomplete"

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "subject_id": self.subject_id,
            "verdict": self.verdict,
            "erasure_certifiable": self.erasure_certifiable,
            "summary": self.summary,
            "ledger_chain_ok": self.ledger_chain_ok,
            "source_count": self.source_count,
            "sources_linked_count": self.sources_linked_count,
            "orphan_source_ids": list(self.orphan_source_ids),
            "source_coverage_rate": round(self.source_coverage_rate, 4),
            "live_item_count": self.live_item_count,
            "attributed_item_count": self.attributed_item_count,
            "unattributed_item_count": self.unattributed_item_count,
            "unattributed_item_ids": list(self.unattributed_item_ids),
            "traced_item_count": self.traced_item_count,
            "derived_item_count": self.derived_item_count,
            "derived_dependency_count": self.derived_dependency_count,
            "lineage_capture_rate": round(self.lineage_capture_rate, 4),
            "verified_stores": [dict(s) for s in self.verified_stores],
            "unverified_locations": list(self.unverified_locations),
            "pending_erasure_count": self.pending_erasure_count,
            "pending_erasures": [dict(p) for p in self.pending_erasures],
            "recommendations": list(self.recommendations),
        }
        if self.subject_preview is not None:
            out["subject_preview"] = dict(self.subject_preview)
        return out


def analyze_coverage(
    *,
    tenant_id: str,
    state: EmberState,
    events: Sequence[LedgerEvent],
    ledger_chain_ok: bool,
    verified_stores: Sequence[dict[str, Any]],
    unverified_locations: Sequence[str] = (),
    pending_erasures: Sequence[dict[str, Any]] = (),
    traced_item_ids: set[str] | frozenset[str] = frozenset(),
    subject_id: str | None = None,
    subject_tokens: tuple[str, ...] = (),
    subject_certificates_count: int = 0,
    subject_tombstoned: bool = False,
) -> CoverageReport:
    """Compute bounded erasure readiness and lineage capture metrics from live custody state."""
    linked_sources: set[str] = set()
    for ev in events:
        if ev.kind == "ADD_ITEM":
            rmeta = (ev.payload.get("item") or {}).get("receipt_meta")
            if isinstance(rmeta, dict) and rmeta.get("source_id"):
                linked_sources.add(str(rmeta["source_id"]))
        elif ev.kind == "ERASE_SUBJECT":
            closure = ev.payload.get("closure") or {}
            for sid in (closure.get("source_leaves_redacted") or {}).keys():
                linked_sources.add(str(sid))
            for sid in (closure.get("source_spans_redacted") or {}).keys():
                linked_sources.add(str(sid))

    live_items = [
        item for item in state.items.values() if item.text != ERASED_TOMBSTONE
    ]
    for item in live_items:
        if item.receipt_meta and item.receipt_meta.get("source_id"):
            linked_sources.add(str(item.receipt_meta["source_id"]))

    source_ids = sorted(state.sources.keys())
    linked_present = {sid for sid in source_ids if sid in linked_sources}
    orphan_sources = tuple(sid for sid in source_ids if sid not in linked_sources)
    source_count = len(source_ids)
    sources_linked_count = len(linked_present)
    source_coverage_rate = (sources_linked_count / source_count) if source_count > 0 else 1.0

    unattributed_ids: list[str] = []
    attributed_count = 0
    traced_count = 0
    derived_item_count = 0
    derived_edge_count = 0

    for item in sorted(live_items, key=lambda x: x.item_id):
        has_receipt = item.receipt_meta is not None
        has_derived = bool(item.derived_from)
        if item.item_id in traced_item_ids:
            traced_count += 1
        if has_derived:
            derived_item_count += 1
            derived_edge_count += len(item.derived_from)
        if has_receipt or has_derived:
            attributed_count += 1
        else:
            unattributed_ids.append(item.item_id)

    live_item_count = len(live_items)
    unattributed_tuple = tuple(unattributed_ids)
    unattributed_item_count = len(unattributed_tuple)
    lineage_capture_rate = (attributed_count / live_item_count) if live_item_count > 0 else 1.0

    uniq_stores = tuple(dict(s) for s in verified_stores)
    uniq_unverified = tuple(dict.fromkeys(str(x) for x in unverified_locations if str(x).strip()))
    pending_list = tuple(dict(p) for p in pending_erasures)
    pending_count = len(pending_list)

    stores_unreachable = any(not bool(s.get("reachable", True)) for s in uniq_stores)

    subject_preview: dict[str, Any] | None = None
    if subject_id is not None:
        closure = compute_closure(state, subject_id, subject_tokens=subject_tokens)
        norm_sub = " ".join(subject_id.strip().lower().split())
        potential_unattr = [
            item.item_id
            for item in sorted(live_items, key=lambda x: x.item_id)
            if item.item_id in unattributed_ids
            and item.item_id not in set(closure.item_ids)
            and item.text
            and norm_sub
            and norm_sub in item.text.lower()
        ]
        subject_preview = {
            "subject_id": subject_id,
            "records_identified": len(closure.item_ids),
            "derived_dependencies_found": len(closure.derived_invalidated),
            "source_leaves_to_redact": sum(len(v) for v in closure.source_leaves_redacted.values()),
            "source_spans_to_redact": sum(
                len(sp)
                for l_map in closure.source_spans_redacted.values()
                for sp in l_map.values()
            ),
            "adapter_records_to_delete": len(closure.adapter_record_ids),
            "tombstoned": subject_tombstoned,
            "certificates_issued": subject_certificates_count,
            "potential_unattributed_matches": len(potential_unattr),
        }

    recommendations: list[str] = []
    if not ledger_chain_ok:
        recommendations.append("Ledger hash-chain verification failed; inspect storage integrity before certifying.")
    if pending_count > 0:
        recommendations.append(
            f"{pending_count} pending/failed erasure operation(s) require reconciliation: run vault.reconcile_erasures()."
        )
    if stores_unreachable:
        recommendations.append("Backing store probe failed; verify store connectivity before issuing erasure certificates.")
    if orphan_sources:
        recommendations.append(
            f"{len(orphan_sources)} registered source(s) have zero linked memory items/receipts; derived memories may be unattributed."
        )
    if unattributed_item_count > 0:
        recommendations.append(
            f"{unattributed_item_count} live memory item(s) lack both source_id receipts and derived_from/trace_turn() lineage."
        )
    if "mem0_internal_history_db" in uniq_unverified:
        recommendations.append(
            "Enable encrypt_vendor_payload=True (or prune Mem0 SQLite history.db) to eliminate plaintext vendor DELETE logs."
        )
    if "colocated_keystore_and_ledger" in uniq_unverified:
        recommendations.append(
            "Configure a separate key_dir outside ledger backups or use AWS/GCP/Vault KMS for stronger key isolation."
        )

    if not ledger_chain_ok or pending_count > 0 or stores_unreachable:
        verdict = "incomplete"
        summary = "Incomplete — verification blockers detected (reconciliation, store reachability, or ledger integrity)."
    elif orphan_sources or unattributed_item_count > 0 or uniq_unverified:
        verdict = "bounded_with_warnings"
        summary = (
            "Bounded with warnings — structural custody is healthy inside traced boundaries, "
            "but unattributed writes, orphan sources, or unverified locations exist."
        )
    else:
        verdict = "complete"
        summary = "Complete — all sources, live items, lineage edges, separated keys, and stores verified."

    return CoverageReport(
        tenant_id=tenant_id,
        subject_id=subject_id,
        verdict=verdict,
        summary=summary,
        ledger_chain_ok=ledger_chain_ok,
        source_count=source_count,
        sources_linked_count=sources_linked_count,
        orphan_source_ids=orphan_sources,
        source_coverage_rate=source_coverage_rate,
        live_item_count=live_item_count,
        attributed_item_count=attributed_count,
        unattributed_item_count=unattributed_item_count,
        unattributed_item_ids=unattributed_tuple,
        traced_item_count=traced_count,
        derived_item_count=derived_item_count,
        derived_dependency_count=derived_edge_count,
        lineage_capture_rate=lineage_capture_rate,
        verified_stores=uniq_stores,
        unverified_locations=uniq_unverified,
        pending_erasure_count=pending_count,
        pending_erasures=pending_list,
        subject_preview=subject_preview,
        recommendations=tuple(recommendations),
    )
