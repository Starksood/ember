"""Full 12-subject x 7-attack-class adversarial erasure suite (SUITE_V2)."""

from __future__ import annotations

import uuid
from typing import Any

from ember.adapters import InMemoryStoreAdapter
from ember.audit import (
    SUITE_V2,
    BenchmarkBackendSpec,
    ScenarioBackendPair,
    mem0_backend_spec,
    run_benchmark,
    run_subject_scenario,
    verify_suite_hygiene,
)


def test_suite_hygiene_passes_cleanly():
    summary = verify_suite_hygiene(SUITE_V2)
    assert summary["subjects"] == 12
    assert summary["probes"] == 204
    assert summary["unique_markers"] == 24


def test_full_adversarial_erasure_benchmark_beats_naive_baseline():
    report = run_benchmark(SUITE_V2)
    assert report["passed"] is True
    assert report["adapter"]["name"] == "InMemoryStoreAdapter"
    assert report["adapter"]["retrieval_mode"] == "keyword_overlap"
    assert report["adapter"]["embedding_model"] is None
    assert report["adapter"]["search_settings"]["probe_limit"] == 10
    assert report["subjects"] == 12
    assert report["total_probes"] == 204
    # Naive user_id-only deletion leaks via derived workspace records on every single subject
    assert report["baseline_naive_leaks"] > 0
    # Ember achieves zero leaks while preserving bystander survival, proofs, and content-free replays
    assert report["ember_leaks"] == 0
    assert report["all_bystanders_survived"] is True
    assert report["all_bystander_receipts_valid"] is True
    assert report["all_redacted_sources_clean"] is True
    assert report["all_replays_content_free"] is True
    assert report["all_tombstones_blocked"] is True
    assert report["all_certificates_valid"] is True

    single = run_subject_scenario(SUITE_V2[0])
    assert single["adapter"]["name"] == "InMemoryStoreAdapter"


def test_suite_runs_cleanly_through_strict_mem0_backend_spec_with_bystander_checks():
    closed_count = 0

    class _ClosableStrictMem0:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}

        def add(self, messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True):
            for name, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
                if val is not None and any(ch.isspace() for ch in str(val).strip()):
                    raise ValueError(f"Invalid {name}: whitespace rejected")
            mid = f"m2_{uuid.uuid4().hex[:10]}"
            self.rows[mid] = {
                "id": mid,
                "memory": str(messages),
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": dict(metadata or {}),
            }
            return {"results": [{"id": mid, "memory": str(messages), "event": "ADD"}]}

        def search(self, query: str, *, top_k: int = 20, filters=None, threshold: float = 0.1, **_kwargs):
            if not filters or not any(k in filters for k in ("user_id", "agent_id", "run_id")):
                raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
            q_toks = {t.strip(".,?!") for t in query.lower().split() if t.strip(".,?!")}
            hits = []
            for r in self.rows.values():
                if "user_id" in filters and r.get("user_id") != filters["user_id"]:
                    continue
                words = {t.strip(".,?!") for t in r["memory"].lower().split() if t.strip(".,?!")}
                score = len(q_toks & words) / max(1, len(q_toks))
                if score >= threshold:
                    hits.append(
                        {
                            "id": r["id"],
                            "memory": r["memory"],
                            "score": score,
                            "user_id": r["user_id"],
                            "metadata": dict(r["metadata"]),
                        }
                    )
            hits.sort(key=lambda h: -h["score"])
            return {"results": hits[:top_k]}

        def delete(self, memory_id: str):
            if memory_id not in self.rows:
                raise ValueError(f"Memory with id {memory_id} not found")
            del self.rows[memory_id]

        def delete_all(self, user_id: str | None = None):
            for mid in [k for k, v in self.rows.items() if v.get("user_id") == user_id]:
                del self.rows[mid]

        def close(self) -> None:
            nonlocal closed_count
            closed_count += 1

    spec = mem0_backend_spec(
        lambda _scenario, _role: _ClosableStrictMem0(),
        embedding_model="mock-strict-v2",
        vector_store="in-memory-strict-mock",
        retrieval_mode="token_overlap_strict_filters",
        probe_limit=10,
        bystander_limit=5,
        threshold=0.0,
        infer=False,
    )
    report = run_benchmark(SUITE_V2, backend=spec)
    assert report["passed"] is True
    assert report["baseline_naive_leaks"] == 204
    assert report["ember_leaks"] == 0
    assert report["all_bystanders_survived"] is True
    assert report["all_bystander_receipts_valid"] is True
    assert report["adapter"]["name"] == "Mem0Adapter"
    assert report["adapter"]["embedding_model"] == "mock-strict-v2"
    assert report["adapter"]["search_settings"]["threshold"] == 0.0
    assert closed_count == 24


def test_custom_backend_overdeletion_fails_bystander_gate():
    class WipesEverythingAdapter(InMemoryStoreAdapter):
        def delete(self, record_ids):
            with self._lock:
                wiped = len(self._records)
                self._records.clear()
                return wiped

    def _pair(_scenario):
        b = InMemoryStoreAdapter()
        return ScenarioBackendPair(
            baseline=b,
            baseline_delete_naive=b.delete_by_user_id_naive,
            ember_adapter=WipesEverythingAdapter(),
        )

    bad_spec = BenchmarkBackendSpec(
        adapter_name="WipesEverythingAdapter",
        retrieval_mode="keyword_overlap",
        pair_factory=_pair,
    )
    report = run_benchmark(SUITE_V2[:2], backend=bad_spec)
    assert report["ember_leaks"] == 0
    assert report["all_bystanders_survived"] is False
    assert report["passed"] is False
