"""Adversarial erasure benchmark & two-direction calibration suite (Step 2 empirical rigor).

Encodes the Phase 0 lessons learned from Fireweed:
  1. Never trust a leakage metric alone — deleting *everything* scores 0 leaks, so
     bystander survival AND surviving bystander Merkle receipt validity are co-primary gates.
  2. Enforce corpus hygiene before running: every leak marker must actually occur in its
     subject's facts and must never collide with bystander facts or other subjects.
  3. Benchmark a naive `user_id`-only store deletion baseline side-by-side against Ember's
     structural closure + crypto-shredding + Merkle redaction + tombstone enforcement.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unicodedata
from typing import Any

from .adapters import InMemoryStoreAdapter, Mem0Adapter, MemoryStoreAdapter, SQLiteStoreAdapter
from .reducer import TombstonedSubjectError, decrypt_item_metadata, decrypt_item_payload
from .vault import CustodyVault
from .verify import verify_ed25519_certificate


@dataclass(frozen=True)
class ScenarioBackendPair:
    """Baseline and Ember adapter pair provisioned for one scenario run."""

    baseline: MemoryStoreAdapter
    baseline_delete_naive: Callable[[str], Any]
    ember_adapter: MemoryStoreAdapter
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class BenchmarkBackendSpec:
    """Adapter and search configuration evaluated by the adversarial erasure suite."""

    adapter_name: str
    retrieval_mode: str
    embedding_model: str | None = None
    vector_store: str | None = None
    search_settings: dict[str, Any] = field(default_factory=dict)
    pair_factory: Callable[["SubjectScenario"], ScenarioBackendPair] | None = None

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.adapter_name,
            "retrieval_mode": self.retrieval_mode,
            "embedding_model": self.embedding_model,
            "vector_store": self.vector_store,
            "search_settings": dict(self.search_settings),
        }


def in_memory_backend_spec() -> BenchmarkBackendSpec:
    """Default deterministic lexical-overlap fixture (`InMemoryStoreAdapter`)."""

    def _make_pair(_scenario: "SubjectScenario") -> ScenarioBackendPair:
        baseline = InMemoryStoreAdapter()
        ember_adapter = InMemoryStoreAdapter()
        return ScenarioBackendPair(
            baseline=baseline,
            baseline_delete_naive=baseline.delete_by_user_id_naive,
            ember_adapter=ember_adapter,
        )

    return BenchmarkBackendSpec(
        adapter_name="InMemoryStoreAdapter",
        retrieval_mode="keyword_overlap",
        embedding_model=None,
        vector_store=None,
        search_settings={
            "probe_limit": 10,
            "bystander_limit": 5,
            "erase_probe_limit": 25,
            "naive_delete": "user_id_metadata_exact",
        },
        pair_factory=_make_pair,
    )


def mem0_backend_spec(
    memory_factory: Callable[["SubjectScenario", str], Any],
    *,
    embedding_model: str,
    vector_store: str = "qdrant",
    retrieval_mode: str = "vector_semantic",
    probe_limit: int = 10,
    bystander_limit: int = 5,
    threshold: float | None = 0.0,
    infer: bool = False,
    extra_search_settings: dict[str, Any] | None = None,
) -> BenchmarkBackendSpec:
    """Build a `BenchmarkBackendSpec` backed by real `mem0.Memory` instances."""

    def _make_pair(scenario: "SubjectScenario") -> ScenarioBackendPair:
        baseline_mem = memory_factory(scenario, "baseline")
        try:
            ember_mem = memory_factory(scenario, "ember")
        except BaseException:
            if hasattr(baseline_mem, "close"):
                baseline_mem.close()
            raise
        baseline_adapter = Mem0Adapter(
            baseline_mem,
            infer=infer,
            search_threshold=threshold,
        )
        ember_adapter = Mem0Adapter(
            ember_mem,
            infer=infer,
            search_threshold=threshold,
        )

        def _delete_naive(subject_id: str) -> Any:
            norm_id = baseline_adapter.normalize_entity_id(subject_id) or subject_id
            return baseline_mem.delete_all(user_id=norm_id)

        def _close() -> None:
            for client in (baseline_mem, ember_mem):
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    close_fn()

        return ScenarioBackendPair(
            baseline=baseline_adapter,
            baseline_delete_naive=_delete_naive,
            ember_adapter=ember_adapter,
            close=_close,
        )

    settings: dict[str, Any] = {
        "probe_limit": probe_limit,
        "bystander_limit": bystander_limit,
        "erase_probe_limit": 25,
        "threshold": threshold,
        "infer": infer,
        "naive_delete": "mem0.delete_all(user_id=subject)",
    }
    if extra_search_settings:
        settings.update(extra_search_settings)

    return BenchmarkBackendSpec(
        adapter_name="Mem0Adapter",
        retrieval_mode=retrieval_mode,
        embedding_model=embedding_model,
        vector_store=vector_store,
        search_settings=settings,
        pair_factory=_make_pair,
    )


@dataclass(frozen=True)
class SubjectScenario:
    subject: str
    peer: str
    third: str
    org: str
    city: str
    facts: tuple[str, ...]
    bystander_facts: tuple[str, ...]
    bystander_queries: tuple[str, ...]
    derived_summary: str
    leak_markers: tuple[str, ...]
    attacks: dict[str, tuple[str, ...]]
    compound_sentence: str = ""
    compound_subject_clause: str = ""
    compound_bystander_clause: str = ""


def _mk(
    name: str,
    role: str,
    org: str,
    city: str,
    year: str,
    peer: str,
    peer_role: str,
    third: str,
    third_role: str,
    relation: str,
) -> SubjectScenario:
    """Create one subject, two innocent bystanders sharing an org, and a derived team summary."""
    subj_clause = f"{name} leads operations in {city}"
    bys_clause = f"and {peer} manages {peer_role} duties at {org}."
    return SubjectScenario(
        subject=name,
        peer=peer,
        third=third,
        org=org,
        city=city,
        facts=(
            f"{name} is a {role} at {org}.",
            f"{name} lives in {city}.",
            f"{name} joined {org} in {year}.",
            f"{name} {relation} {peer} at {org}.",
        ),
        bystander_facts=(
            f"{peer} is a {peer_role} at {org}.",
            f"{third} is a {third_role} at {org}.",
        ),
        bystander_queries=(
            f"What does {peer} do at {org}?",
            f"Who is the {third_role} at {org}?",
        ),
        derived_summary=f"Staff summary at {org}: {name} in {city} collaborates with {peer}.",
        leak_markers=(name, city),
        compound_sentence=f"{subj_clause}, {bys_clause}",
        compound_subject_clause=f"{subj_clause},",
        compound_bystander_clause=bys_clause,
        attacks={
            "direct": (
                f"What does {name} do?",
                f"Where does {name} live?",
                f"When did {name} join {org}?",
            ),
            "paraphrase": (
                f"Tell me about {name}",
                f"What is {name}'s profession?",
                f"Which city is {name} based in?",
            ),
            "inference": (
                f"Who is the {role} living in {city}?",
                f"Which {role} works at {org}?",
            ),
            "multi_hop": (
                f"Who does {peer} work with at {org}?",
                f"Who {relation} {peer}?",
            ),
            "attribute": (
                f"Who lives in {city}?",
                f"Who is a {role} at {org}?",
            ),
            "temporal": (
                f"Who joined {org} in {year}?",
                f"Who was at {org} before {peer}?",
            ),
            "aggregate": (
                f"How many people work at {org}?",
                f"List everyone who works at {org}",
                f"List everyone living in {city}",
            ),
        },
    )


SUITE_V2: tuple[SubjectScenario, ...] = (
    _mk("Priya", "nurse", "Riverside", "Portland", "2019", "Dana", "pharmacist", "Marcus", "porter", "works with"),
    _mk("Theo", "data engineer", "Acme", "Denver", "2021", "Nina", "designer", "Omar", "recruiter", "mentors"),
    _mk("Rosalind", "cardiologist", "Beaumont", "Halifax", "2018", "Felix", "resident", "Ingrid", "lab manager", "supervises"),
    _mk("Kwame", "architect", "Sandow", "Accra", "2017", "Lena", "engineer", "Tomas", "draughtsman", "partners with"),
    _mk("Yuki", "translator", "Kirin Press", "Sapporo", "2020", "Marco", "editor", "Sofia", "proofreader", "edits with"),
    _mk("Dmitri", "welder", "Halvorsen Yard", "Tromso", "2016", "Beatrix", "apprentice", "Sven", "inspector", "trains"),
    _mk("Amara", "epidemiologist", "Larkfield", "Kampala", "2022", "Gil", "statistician", "Petra", "director", "co-authors with"),
    _mk("Bruno", "sommelier", "Verdant", "Mendoza", "2015", "Clara", "server", "Nadia", "owner", "trained"),
    _mk("Ingeborg", "glaciologist", "Vatna Institute", "Reykjavik", "2014", "Ravi", "technician", "Elsa", "trustee", "surveys with"),
    _mk("Tariq", "luthier", "Cedarline", "Fes", "2013", "Mira", "apprentice", "Hugo", "supplier", "teaches"),
    _mk("Solveig", "actuary", "Northmark", "Bergen", "2023", "Anton", "analyst", "Pilar", "chair", "reviews with"),
    _mk("Nkechi", "botanist", "Ferngate", "Lagos", "2012", "Owen", "curator", "Rita", "cataloguer", "collaborates with"),
)

SUITE_V3: tuple[SubjectScenario, ...] = SUITE_V2
SUITE_V4: tuple[SubjectScenario, ...] = SUITE_V2


def verify_suite_hygiene(suite: tuple[SubjectScenario, ...] = SUITE_V2) -> dict[str, Any]:
    """Refuse to run a corpus that cannot express failures honestly."""
    problems: list[str] = []
    seen_markers: dict[str, str] = {}
    for scenario in suite:
        subj_blob = " ".join(scenario.facts).lower()
        bys_blob = " ".join(scenario.bystander_facts).lower()
        for marker in scenario.leak_markers:
            ml = marker.lower()
            if ml not in subj_blob:
                problems.append(f"{scenario.subject}: marker {marker!r} never appears in subject facts")
            if ml in bys_blob:
                problems.append(f"{scenario.subject}: marker {marker!r} appears in bystander facts (confounded)")
            if ml in seen_markers and seen_markers[ml] != scenario.subject:
                problems.append(f"marker {marker!r} shared between {seen_markers[ml]} and {scenario.subject}")
            seen_markers[ml] = scenario.subject
        if not scenario.facts or not scenario.bystander_facts:
            problems.append(f"{scenario.subject}: must include both subject facts and bystander facts")
        if not scenario.attacks.get("direct"):
            problems.append(f"{scenario.subject}: missing required 'direct' probes")
        if scenario.compound_sentence:
            if scenario.subject.lower() not in scenario.compound_subject_clause.lower():
                problems.append(f"{scenario.subject}: compound_subject_clause missing subject")
            b_cl_low = scenario.compound_bystander_clause.lower()
            for marker in scenario.leak_markers:
                if marker.lower() in b_cl_low:
                    problems.append(
                        f"{scenario.subject}: compound_bystander_clause leaked subject marker {marker!r}"
                    )
    if problems:
        raise AssertionError("suite hygiene check failed:\n  - " + "\n  - ".join(problems))

    by_category: dict[str, int] = {}
    for scenario in suite:
        for cat, probes in scenario.attacks.items():
            by_category[cat] = by_category.get(cat, 0) + len(probes)
    return {
        "subjects": len(suite),
        "probes": sum(by_category.values()),
        "by_category": by_category,
        "unique_markers": len(seen_markers),
    }


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in markers)


def run_subject_scenario(
    scenario: SubjectScenario,
    *,
    backend: BenchmarkBackendSpec | None = None,
) -> dict[str, Any]:
    """Run baseline naive deletion vs Ember custody wrapper on one subject scenario."""
    resolved_backend = backend if backend is not None else in_memory_backend_spec()
    adapter_meta = resolved_backend.metadata_dict()
    probe_limit = int(resolved_backend.search_settings.get("probe_limit", 10))
    bystander_limit = int(resolved_backend.search_settings.get("bystander_limit", 5))

    if resolved_backend.pair_factory is not None:
        pair = resolved_backend.pair_factory(scenario)
    else:
        default_spec = in_memory_backend_spec()
        if default_spec.pair_factory is None:
            raise RuntimeError("in_memory_backend_spec missing pair_factory")
        pair = default_spec.pair_factory(scenario)

    vault: CustodyVault | None = None
    try:
        # 1. Baseline naive vector/KV store (`delete_all(user_id=subject)` only)
        baseline = pair.baseline
        for i, fact in enumerate(scenario.facts):
            baseline.upsert(
                item_id=f"b_s_{i}",
                text=fact,
                subjects=(scenario.subject,),
                metadata={"user_id": scenario.subject},
            )
        for i, bfact in enumerate(scenario.bystander_facts):
            owner = scenario.peer if i == 0 else scenario.third
            baseline.upsert(
                item_id=f"b_b_{i}",
                text=bfact,
                subjects=(owner,),
                metadata={"user_id": owner},
            )
        baseline.upsert(
            item_id="b_derived",
            text=scenario.derived_summary,
            subjects=(scenario.org,),
            metadata={"user_id": scenario.org},
        )
        pair.baseline_delete_naive(scenario.subject)

        baseline_leaks = 0
        for probes in scenario.attacks.values():
            for probe in probes:
                hits = baseline.search(probe, limit=probe_limit)
                if any(_contains_marker(h.text, scenario.leak_markers) for h in hits):
                    baseline_leaks += 1

        # 2. Ember-wrapped store with shared RedactableDoc + derived lineage + crypto-shredding
        adapter = pair.ember_adapter
        vault = CustodyVault.ephemeral(adapter=adapter)
        shared_text = " ".join([*scenario.facts, *scenario.bystander_facts])
        vault.register_source("shared_intake", shared_text, origin="audit_suite", origin_kind="file")

        subject_items = []
        for fact in scenario.facts:
            res = vault.remember(fact, subjects=[scenario.subject], source_id="shared_intake", quote=fact)
            subject_items.append(res.item.item_id)

        bystander_items = []
        for idx, bfact in enumerate(scenario.bystander_facts):
            owner = scenario.peer if idx == 0 else scenario.third
            res = vault.remember(bfact, subjects=[owner], source_id="shared_intake", quote=bfact)
            bystander_items.append(res.item.item_id)

        # Derived team summary under separate org tag referencing BOTH subject & bystander items
        vault.remember(
            scenario.derived_summary,
            subjects=[scenario.org],
            derived_from=[subject_items[0], bystander_items[0]],
            metadata={"user_id": scenario.org},
        )

        # Execute full Ember erasure + Ed25519 certification
        cert = vault.erase(
            scenario.subject,
            probes=scenario.attacks["direct"],
            markers=scenario.leak_markers,
        )

        # Evaluate all 7 attack categories against BOTH raw adapter and wrapped vault.search()
        ember_leaks = 0
        per_cat: dict[str, dict[str, int]] = {}
        total_probes = 0
        for cat, probes in scenario.attacks.items():
            cat_leaks = 0
            for probe in probes:
                total_probes += 1
                res = vault.search(probe, limit=probe_limit)
                raw_hits = adapter.search(probe, limit=probe_limit)
                if any(_contains_marker(h.text, scenario.leak_markers) for h in res.hits) or any(
                    _contains_marker(h.text, scenario.leak_markers) for h in raw_hits
                ):
                    cat_leaks += 1
                    ember_leaks += 1
            per_cat[cat] = {"probes": len(probes), "leaks": cat_leaks}

        # Check bystander survival (anti-over-deletion gate!)
        bystander_retrievable = all(
            not vault.search(bq, limit=bystander_limit).abstained for bq in scenario.bystander_queries
        )

        # Check surviving bystander receipts verify against redacted RedactableDoc
        bystander_receipts_valid = all(vault.verify_item_receipt(bid) for bid in bystander_items)
        redacted_doc = vault.ledger.get_document(vault.tenant_id, "shared_intake")
        doc_clean = redacted_doc is not None and not _contains_marker(redacted_doc.text(), scenario.leak_markers)

        # Check raw historical ledger payloads yield zero leak markers when decrypted post-shred
        raw_history_clean = True
        for ev in vault.ledger.events(vault.tenant_id):
            if ev.kind == "ADD_ITEM":
                _, pt, pq = decrypt_item_payload(ev.payload["item"], vault.keystore)
                pmeta = decrypt_item_metadata(ev.payload["item"], vault.keystore)
                combined = f"{pt} {pq or ''} {json.dumps(pmeta)}"
                if _contains_marker(combined, scenario.leak_markers):
                    raw_history_clean = False
                    break

        # Check tombstone prevents zombie re-ingestion
        tombstone_blocked = False
        try:
            vault.remember(scenario.facts[0], subjects=[scenario.subject])
        except TombstonedSubjectError:
            tombstone_blocked = True

        sig_ok, _ = verify_ed25519_certificate(cert.as_dict())

        return {
            "subject": scenario.subject,
            "adapter": adapter_meta,
            "total_probes": total_probes,
            "baseline_naive_leaks": baseline_leaks,
            "ember_leaks": ember_leaks,
            "by_category": per_cat,
            "bystander_retrievable": bystander_retrievable,
            "bystander_receipts_valid": bystander_receipts_valid,
            "redacted_source_clean": doc_clean,
            "raw_history_content_free": raw_history_clean,
            "tombstone_blocked": tombstone_blocked,
            "certificate_valid": sig_ok is True,
            "derived_invalidated": len(cert.closure["derived_invalidated"]),
        }
    finally:
        if vault is not None:
            vault.close()
        if pair.close is not None:
            pair.close()


def run_benchmark(
    suite: tuple[SubjectScenario, ...] = SUITE_V2,
    *,
    backend: BenchmarkBackendSpec | None = None,
) -> dict[str, Any]:
    """Execute the full adversarial suite with enforced hygiene and aggregate scorecard."""
    resolved_backend = backend if backend is not None else in_memory_backend_spec()
    hygiene = verify_suite_hygiene(suite)
    rows = [run_subject_scenario(s, backend=resolved_backend) for s in suite]
    total_probes = sum(r["total_probes"] for r in rows)
    total_ember_leaks = sum(r["ember_leaks"] for r in rows)
    total_baseline_leaks = sum(r["baseline_naive_leaks"] for r in rows)
    passed = (
        total_ember_leaks == 0
        and all(r["bystander_retrievable"] for r in rows)
        and all(r["bystander_receipts_valid"] for r in rows)
        and all(r["redacted_source_clean"] for r in rows)
        and all(r["raw_history_content_free"] for r in rows)
        and all(r["tombstone_blocked"] for r in rows)
        and all(r["certificate_valid"] for r in rows)
    )
    return {
        "suite": "ember_erasure_v2",
        "adapter": resolved_backend.metadata_dict(),
        "hygiene": hygiene,
        "subjects": len(rows),
        "total_probes": total_probes,
        "baseline_naive_leaks": total_baseline_leaks,
        "ember_leaks": total_ember_leaks,
        "all_bystanders_survived": all(r["bystander_retrievable"] for r in rows),
        "all_bystander_receipts_valid": all(r["bystander_receipts_valid"] for r in rows),
        "all_redacted_sources_clean": all(r["redacted_source_clean"] for r in rows),
        "all_replays_content_free": all(r["raw_history_content_free"] for r in rows),
        "all_tombstones_blocked": all(r["tombstone_blocked"] for r in rows),
        "all_certificates_valid": all(r["certificate_valid"] for r in rows),
        "passed": passed,
        "results": rows,
    }


def _adapter_contains_plaintext(adapter: MemoryStoreAdapter, markers: tuple[str, ...]) -> bool:
    if isinstance(adapter, InMemoryStoreAdapter):
        for rec in adapter._records.values():
            rec_text = str(rec.get("text", "")) if isinstance(rec, dict) else str(getattr(rec, "text", ""))
            rec_meta = rec.get("metadata", {}) if isinstance(rec, dict) else getattr(rec, "metadata", {})
            rec_subj = rec.get("subjects", ()) if isinstance(rec, dict) else getattr(rec, "subjects", ())
            if (
                _contains_marker(rec_text, markers)
                or _contains_marker(json.dumps(rec_meta), markers)
                or _contains_marker(json.dumps(list(rec_subj)), markers)
            ):
                return True
        return False
    for m in markers:
        for hit in adapter.search(m, limit=25):
            if _contains_marker(hit.text, markers) or _contains_marker(json.dumps(hit.metadata), markers):
                return True
    return False


def run_v3_subject_scenario(
    scenario: SubjectScenario,
    *,
    backend: BenchmarkBackendSpec | None = None,
    encrypt_vendor_payload: bool = True,
) -> dict[str, Any]:
    """Run all V2 attack classes plus Phase 2 sub-leaf, TurnTrace, encrypted-vendor, and content-tombstone checks."""
    base = run_subject_scenario(scenario, backend=backend)

    # Phase 2 deep hardening audit (clause co-tenancy, TurnTrace auto-lineage, zero-plaintext vendor, content zombie block)
    p2_adapter = InMemoryStoreAdapter()
    with CustodyVault.ephemeral(
        adapter=p2_adapter,
        encrypt_vendor_payload=encrypt_vendor_payload,
    ) as v_p2:
        compound_text = scenario.compound_sentence or f"{scenario.facts[0]} {scenario.bystander_facts[0]}"
        subj_quote = scenario.compound_subject_clause or scenario.facts[0]
        bys_quote = scenario.compound_bystander_clause or scenario.bystander_facts[0]
        v_p2.register_source("compound_doc", compound_text, mode="sentence", span_mode="clause", origin="audit_v3")
        r_subj = v_p2.remember(
            subj_quote.rstrip(","),
            subjects=[scenario.subject],
            source_id="compound_doc",
            quote=subj_quote,
        )
        r_bys = v_p2.remember(
            bys_quote.rstrip("."),
            subjects=[scenario.peer],
            source_id="compound_doc",
            quote=bys_quote,
        )

        # Auto TurnTrace read -> write lineage without manual derived_from
        with v_p2.trace_turn():
            _ = v_p2.search(scenario.subject)
            r_trace = v_p2.remember(
                f"Follow-up coordination note referencing {scenario.subject} at {scenario.org}",
                subjects=[scenario.org],
                metadata={"user_id": scenario.org},
            )

        vendor_plaintext_leaked = (
            _adapter_contains_plaintext(p2_adapter, scenario.leak_markers)
            if encrypt_vendor_payload
            else False
        )

        cert_p2 = v_p2.erase(scenario.subject, markers=scenario.leak_markers)
        comp_doc = v_p2.ledger.get_document(v_p2.tenant_id, "compound_doc")
        comp_clean = comp_doc is not None and not _contains_marker(comp_doc.text(), scenario.leak_markers)
        bys_subleaf_valid = v_p2.verify_item_receipt(r_bys.item.item_id)
        bys_searchable = not v_p2.search(scenario.peer, limit=5).abstained
        subleaf_ok = comp_clean and bys_subleaf_valid and bys_searchable

        trace_derived_ok = (
            r_trace.item.item_id not in v_p2.state.items
            and r_subj.item.item_id in r_trace.item.derived_from
            and r_trace.item.item_id in cert_p2.closure.get("derived_invalidated", [])
        )

        zombie_content_blocked = False
        try:
            v_p2.remember(
                f"{scenario.subject} emergency profile update in {scenario.city}",
                subjects=["crm_sync_bot"],
            )
        except TombstonedSubjectError:
            zombie_content_blocked = True

        p2_leaks = 0
        for probe in scenario.attacks["direct"]:
            res = v_p2.search(probe, limit=10)
            if any(_contains_marker(h.text, scenario.leak_markers) for h in res.hits):
                p2_leaks += 1

    base["subleaf_redaction_and_bystander_ok"] = subleaf_ok
    base["turn_trace_auto_derived_ok"] = trace_derived_ok
    base["encrypted_vendor_zero_plaintext"] = not vendor_plaintext_leaked
    base["zombie_content_blocked"] = zombie_content_blocked
    base["phase2_direct_leaks"] = p2_leaks
    return base


def run_v3_benchmark(
    suite: tuple[SubjectScenario, ...] = SUITE_V3,
    *,
    backend: BenchmarkBackendSpec | None = None,
    encrypt_vendor_payload: bool = True,
) -> dict[str, Any]:
    """Execute the extended V3 benchmark adding Phase 2 hardening gates across all 12 subjects."""
    resolved_backend = backend if backend is not None else in_memory_backend_spec()
    hygiene = verify_suite_hygiene(suite)
    rows = [
        run_v3_subject_scenario(s, backend=resolved_backend, encrypt_vendor_payload=encrypt_vendor_payload)
        for s in suite
    ]
    total_probes = sum(r["total_probes"] for r in rows)
    total_ember_leaks = sum(r["ember_leaks"] + r["phase2_direct_leaks"] for r in rows)
    total_baseline_leaks = sum(r["baseline_naive_leaks"] for r in rows)

    all_subleaf = all(r["subleaf_redaction_and_bystander_ok"] for r in rows)
    all_trace = all(r["turn_trace_auto_derived_ok"] for r in rows)
    all_enc_vendor = all(r["encrypted_vendor_zero_plaintext"] for r in rows)
    all_zombie_content = all(r["zombie_content_blocked"] for r in rows)
    passed = (
        total_ember_leaks == 0
        and all(r["bystander_retrievable"] for r in rows)
        and all(r["bystander_receipts_valid"] for r in rows)
        and all(r["redacted_source_clean"] for r in rows)
        and all(r["raw_history_content_free"] for r in rows)
        and all(r["tombstone_blocked"] for r in rows)
        and all(r["certificate_valid"] for r in rows)
        and all_subleaf
        and all_trace
        and all_enc_vendor
        and all_zombie_content
    )
    return {
        "suite": "ember_erasure_v3",
        "adapter": resolved_backend.metadata_dict(),
        "hygiene": hygiene,
        "subjects": len(rows),
        "total_probes": total_probes,
        "baseline_naive_leaks": total_baseline_leaks,
        "ember_leaks": total_ember_leaks,
        "all_bystanders_survived": all(r["bystander_retrievable"] for r in rows),
        "all_bystander_receipts_valid": all(r["bystander_receipts_valid"] for r in rows),
        "all_redacted_sources_clean": all(r["redacted_source_clean"] for r in rows),
        "all_replays_content_free": all(r["raw_history_content_free"] for r in rows),
        "all_tombstones_blocked": all(r["tombstone_blocked"] for r in rows),
        "all_certificates_valid": all(r["certificate_valid"] for r in rows),
        "all_subleaf_redactions_ok": all_subleaf,
        "all_turn_trace_auto_derived_ok": all_trace,
        "all_encrypted_vendor_zero_plaintext": all_enc_vendor,
        "all_zombie_content_blocked": all_zombie_content,
        "passed": passed,
        "results": rows,
    }


def _percentile_ms(samples_ms: list[float], pct: float) -> float:
    if not samples_ms:
        return 0.0
    ordered = sorted(samples_ms)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 3)


class _FaultyDeleteAdapter:
    """Adapter proxy injecting a transient `ConnectionError` on the first `delete()` call."""

    def __init__(self, inner: MemoryStoreAdapter, *, fail_count: int = 1) -> None:
        self._inner = inner
        self._remaining_fails = fail_count
        self.delete_calls = 0

    def upsert(self, **kwargs: Any) -> tuple[str, ...]:
        return self._inner.upsert(**kwargs)

    def search(self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None):
        return self._inner.search(query, limit=limit, filters=filters)

    def delete(self, record_ids: Iterable[str]) -> int:
        self.delete_calls += 1
        if self._remaining_fails > 0:
            self._remaining_fails -= 1
            raise ConnectionError("injected transient store delete timeout")
        return self._inner.delete(record_ids)


def _run_v4_paraphrase_track(scenario: SubjectScenario) -> dict[str, Any]:
    """Compare observed trace_turn() vs unobserved out-of-band entity-free paraphrase writes."""
    tag = hashlib.sha256(scenario.subject.encode("utf-8")).hexdigest()[:8]
    unique_paraphrase_token = f"indirectbadge_{tag}"
    entity_free_paraphrase = (
        f"Staff briefing note ({unique_paraphrase_token}): longtime specialist handling evening intake shifts."
    )
    email_alias = f"{scenario.subject.lower()}@example.org"
    nfd_alias = unicodedata.normalize("NFD", f"{scenario.subject}\u00e9")

    # Track 1A: Observed write inside trace_turn() -> entity-free paraphrase MUST cascade-delete cleanly.
    obs_adapter = InMemoryStoreAdapter()
    obs_leaks = 0
    obs_cert_ok = False
    obs_bystander_ok = False
    obs_email_blocked = False
    obs_unicode_blocked = False
    with CustodyVault.ephemeral(adapter=obs_adapter, encrypt_vendor_payload=True) as v_obs:
        v_obs.register_source(
            "doc_v4_obs",
            " ".join([*scenario.facts, *scenario.bystander_facts]),
            span_mode="clause",
            origin="v4_paraphrase",
        )
        v_obs.remember(
            scenario.facts[0],
            subjects=[scenario.subject],
            aliases=[email_alias, nfd_alias],
            source_id="doc_v4_obs",
            quote=scenario.facts[0],
        )
        bys_res = v_obs.remember(
            scenario.bystander_facts[0],
            subjects=[scenario.peer],
            source_id="doc_v4_obs",
            quote=scenario.bystander_facts[0],
        )
        with v_obs.trace_turn():
            _ = v_obs.search(scenario.facts[0], limit=5)
            v_obs.remember(entity_free_paraphrase, subjects=["summary_bot"])

        cert = v_obs.erase(
            scenario.subject,
            aliases=[email_alias, nfd_alias],
            probes=[scenario.facts[0], unique_paraphrase_token],
            markers=[*scenario.leak_markers, unique_paraphrase_token],
        )
        obs_cert_ok = verify_ed25519_certificate(cert.as_dict())[0] is True
        res_para = v_obs.search(unique_paraphrase_token, limit=5)
        if not res_para.abstained:
            obs_leaks += 1
        obs_bystander_ok = (
            not v_obs.search(scenario.peer, limit=5).abstained
            and v_obs.verify_item_receipt(bys_res.item.item_id)
        )
        try:
            v_obs.remember(f"Ticket follow-up for {email_alias}", subjects=["crm_bot"])
        except TombstonedSubjectError:
            obs_email_blocked = True
        try:
            nfc_alias = unicodedata.normalize("NFC", nfd_alias)
            v_obs.remember(f"Note mentioning {nfc_alias}", subjects=["crm_bot"])
        except TombstonedSubjectError:
            obs_unicode_blocked = True

    # Track 1B: Unobserved out-of-band write OUTSIDE trace_turn() -> honestly leaks,
    # AND `audit_coverage()` MUST detect the unattributed write 100% of the time.
    unobs_adapter = InMemoryStoreAdapter()
    unobs_leaked_honestly = False
    coverage_caught_gap = False
    with CustodyVault.ephemeral(adapter=unobs_adapter) as v_unobs:
        v_unobs.register_source(
            "doc_v4_unobs",
            scenario.facts[0],
            origin="v4_paraphrase",
        )
        v_unobs.remember(
            scenario.facts[0],
            subjects=[scenario.subject],
            source_id="doc_v4_unobs",
            quote=scenario.facts[0],
        )
        # Unobserved out-of-band summary (outside trace_turn, no derived_from, no source_id)
        v_unobs.remember(entity_free_paraphrase, subjects=["rogue_summarizer"])
        cov = v_unobs.audit_coverage()
        coverage_caught_gap = (
            cov.verdict == "bounded_with_warnings"
            and cov.unattributed_item_count >= 1
            and cov.lineage_capture_rate < 1.0
        )
        v_unobs.erase(scenario.subject, markers=scenario.leak_markers)
        res_unobs = v_unobs.search(unique_paraphrase_token, limit=5)
        unobs_leaked_honestly = any(unique_paraphrase_token in h.text for h in res_unobs.hits)

    return {
        "subject": scenario.subject,
        "observed_traced_paraphrase_leaks": obs_leaks,
        "observed_cert_valid": obs_cert_ok,
        "observed_bystander_ok": obs_bystander_ok,
        "email_zombie_blocked": obs_email_blocked,
        "unicode_zombie_blocked": obs_unicode_blocked,
        "unobserved_out_of_band_leaks": 1 if unobs_leaked_honestly else 0,
        "coverage_caught_unattributed_gap": coverage_caught_gap,
    }


def _run_v4_chaos_track() -> dict[str, Any]:
    """Execute crash & fault-injection chaos tests: mid-erase outage, cold restart reconcile, barrier race."""
    # 1. Transient mid-erase vendor failure -> incomplete verdict -> manual reconcile_erasures()
    inner = InMemoryStoreAdapter()
    faulty = _FaultyDeleteAdapter(inner, fail_count=1)
    transient_ok = False
    with CustodyVault.ephemeral(adapter=faulty) as v:
        v.register_source("c1", "Elena Vance takes Eliquis daily. Sam Patel likes tea.")
        v.remember("Elena Vance takes Eliquis daily.", subjects=["elena"], source_id="c1")
        v.remember("Sam Patel likes tea.", subjects=["sam"], source_id="c1")
        failed_caught = False
        try:
            v.erase("elena", markers=["Elena", "Eliquis"])
        except ConnectionError:
            failed_caught = True
        cov_during_outage = v.audit_coverage()
        rep = v.reconcile_erasures()
        cov_after = v.audit_coverage()
        transient_ok = (
            failed_caught
            and cov_during_outage.verdict == "incomplete"
            and cov_during_outage.pending_erasure_count == 1
            and rep.all_clean
            and rep.attempted == 1
            and cov_after.verdict == "complete"
            and v.search("Eliquis").abstained
            and not v.search("tea").abstained
        )

    # 2. Disk-backed crash recovery on fresh CustodyVault.open(..., auto_reconcile=True)
    disk_reconcile_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "vault_data"
        key_dir = root / "vault_keys"
        store_path = root / "store.db"
        store1 = SQLiteStoreAdapter(str(store_path))
        faulty1 = _FaultyDeleteAdapter(store1, fail_count=1)
        v1 = CustodyVault.open(data_dir, key_dir=key_dir, adapter=faulty1)
        try:
            v1.register_source("disk_src", "Marta Novak has peanut allergy. Leo Kim plays chess.")
            v1.remember("Marta Novak has peanut allergy.", subjects=["marta"], source_id="disk_src")
            v1.remember("Leo Kim plays chess.", subjects=["leo"], source_id="disk_src")
            try:
                v1.erase("marta", markers=["Marta", "peanut"])
            except ConnectionError:
                pass
        finally:
            v1.close()
            store1.close()

        # Reopen fresh handle simulating process restart after vendor recovered
        store2 = SQLiteStoreAdapter(str(store_path))
        try:
            with CustodyVault.open(
                data_dir,
                key_dir=key_dir,
                adapter=store2,
                auto_reconcile=True,
            ) as v2:
                cov2 = v2.audit_coverage()
                certs = v2.ledger.list_certificates(v2.tenant_id)
                disk_reconcile_ok = (
                    cov2.verdict == "complete"
                    and len(v2.pending_erasures()) == 0
                    and len(certs) == 1
                    and verify_ed25519_certificate(certs[0])[0] is True
                    and v2.search("peanut").abstained
                    and not v2.search("chess").abstained
                )
        finally:
            store2.close()

    # 3. Cross-handle concurrency barrier chaos (read before erase, write after erase fails + retry stays blocked)
    barrier_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "race_data"
        key_dir = root / "race_keys"
        store_a = SQLiteStoreAdapter(str(root / "race_store.db"))
        store_b = SQLiteStoreAdapter(str(root / "race_store.db"))
        va = CustodyVault.open(data_dir, key_dir=key_dir, adapter=store_a)
        vb = CustodyVault.open(data_dir, key_dir=key_dir, adapter=store_b)
        try:
            va.register_source("race_doc", "Nora Lin takes lithium. Omar Aziz bakes bread.")
            src_item = va.remember("Nora Lin takes lithium.", subjects=["nora"], source_id="race_doc")
            va.remember("Omar Aziz bakes bread.", subjects=["omar"], source_id="race_doc")

            barrier = threading.Barrier(2, timeout=5.0)
            writer_errors: list[str] = []

            def _delayed_writer() -> None:
                try:
                    barrier.wait()
                    time.sleep(0.02)
                    vb.remember(
                        "Delayed summary: Nora Lin takes lithium",
                        subjects=["nora"],
                        derived_from=[src_item.item.item_id],
                    )
                except Exception as exc:
                    writer_errors.append(type(exc).__name__)

            t = threading.Thread(target=_delayed_writer)
            t.start()
            barrier.wait()
            va.erase("nora", markers=["Nora", "lithium"])
            t.join(timeout=5.0)

            # Retry delayed write once more synchronously to verify tombstone holds
            retry_blocked = False
            try:
                vb.remember("Retry summary: Nora Lin takes lithium", subjects=["nora"])
            except TombstonedSubjectError:
                retry_blocked = True

            barrier_ok = (
                bool(writer_errors)
                and retry_blocked
                and va.search("lithium").abstained
                and not va.search("bread").abstained
            )
        finally:
            va.close()
            vb.close()
            store_a.close()
            store_b.close()

    return {
        "transient_outage_reconciled": transient_ok,
        "restart_auto_reconcile_ok": disk_reconcile_ok,
        "cross_handle_barrier_race_blocked": barrier_ok,
        "passed": transient_ok and disk_reconcile_ok and barrier_ok,
    }


def _run_v4_latency_benchmark(*, iterations: int = 25) -> dict[str, Any]:
    """Measure real wall-clock p50/p95 overhead with SQLiteLedger + FileKeyStore + SQLiteStoreAdapter."""
    reg_ms: list[float] = []
    rem_ms: list[float] = []
    search_ms: list[float] = []
    erase_ms: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = SQLiteStoreAdapter(str(root / "bench_store.db"))
        try:
            with CustodyVault.open(root / "bench_data", key_dir=root / "bench_keys", adapter=store) as vault:
                for idx in range(iterations):
                    sid = f"doc_{idx}"
                    subj = f"subj_{idx}"
                    text = f"Subject {subj} recorded clinical note number {idx} with unique token tok_{idx}."
                    t0 = time.perf_counter()
                    vault.register_source(sid, text, origin="v4_latency")
                    reg_ms.append((time.perf_counter() - t0) * 1000.0)

                    t1 = time.perf_counter()
                    vault.remember(text, subjects=[subj], source_id=sid, quote=text)
                    rem_ms.append((time.perf_counter() - t1) * 1000.0)

                    t2 = time.perf_counter()
                    vault.search(f"tok_{idx}", limit=5)
                    search_ms.append((time.perf_counter() - t2) * 1000.0)

                    t3 = time.perf_counter()
                    vault.erase(subj, probes=[f"tok_{idx}"], markers=[subj, f"tok_{idx}"])
                    erase_ms.append((time.perf_counter() - t3) * 1000.0)
        finally:
            store.close()

    return {
        "iterations": iterations,
        "register_source": {"p50_ms": _percentile_ms(reg_ms, 50), "p95_ms": _percentile_ms(reg_ms, 95)},
        "remember": {"p50_ms": _percentile_ms(rem_ms, 50), "p95_ms": _percentile_ms(rem_ms, 95)},
        "search": {"p50_ms": _percentile_ms(search_ms, 50), "p95_ms": _percentile_ms(search_ms, 95)},
        "erase": {"p50_ms": _percentile_ms(erase_ms, 50), "p95_ms": _percentile_ms(erase_ms, 95)},
    }


def run_v4_benchmark(
    suite: tuple[SubjectScenario, ...] = SUITE_V4,
    *,
    latency_iterations: int = 25,
) -> dict[str, Any]:
    """Execute Ember V4 adversarial suite: paraphrase matrix, crash/chaos recovery, and real latency telemetry."""
    hygiene = verify_suite_hygiene(suite)
    paraphrase_rows = [_run_v4_paraphrase_track(s) for s in suite]
    chaos = _run_v4_chaos_track()
    latency = _run_v4_latency_benchmark(iterations=latency_iterations)

    obs_leaks = sum(r["observed_traced_paraphrase_leaks"] for r in paraphrase_rows)
    unobs_leaks = sum(r["unobserved_out_of_band_leaks"] for r in paraphrase_rows)
    cov_detection_rate = sum(1 for r in paraphrase_rows if r["coverage_caught_unattributed_gap"]) / max(
        1, len(paraphrase_rows)
    )
    all_bystanders = all(r["observed_bystander_ok"] for r in paraphrase_rows)
    all_certs = all(r["observed_cert_valid"] for r in paraphrase_rows)
    all_email_blocked = all(r["email_zombie_blocked"] for r in paraphrase_rows)
    all_unicode_blocked = all(r["unicode_zombie_blocked"] for r in paraphrase_rows)

    passed = (
        obs_leaks == 0
        and unobs_leaks > 0
        and cov_detection_rate == 1.0
        and all_bystanders
        and all_certs
        and all_email_blocked
        and all_unicode_blocked
        and chaos["passed"]
    )
    return {
        "suite": "ember_erasure_v4",
        "hygiene": hygiene,
        "subjects": len(paraphrase_rows),
        "paraphrase_matrix": {
            "observed_trace_turn_leaks": obs_leaks,
            "unobserved_out_of_band_leaks": unobs_leaks,
            "coverage_audit_detection_rate": round(cov_detection_rate, 4),
            "all_bystanders_survived": all_bystanders,
            "all_certificates_valid": all_certs,
            "all_email_zombies_blocked": all_email_blocked,
            "all_unicode_zombies_blocked": all_unicode_blocked,
            "rows": paraphrase_rows,
        },
        "chaos_recovery": chaos,
        "latency_ms": latency,
        "passed": passed,
    }
