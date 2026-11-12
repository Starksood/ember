"""Stateful 5-node Research & Triage Agent Graph wired with Ember custody & coverage audits.

Simulates a multi-step agent pipeline:
  Node 1 (Ingest)             -> registers source case notes + indexes encrypted LangChain vectors
  Node 2 (Retrieve)           -> queries via EmberLangChainRetriever inside `vault.trace_turn()`
  Node 3 (ObservedSummary)    -> writes escalation memo INSIDE `trace_turn()` (lineage captured)
  Node 4 (RogueToolWrite)     -> writes unattributed sidecar note OUTSIDE `trace_turn()` (lineage gap!)
  Node 5 (CoverageAndErase)   -> `audit_coverage()` catches the unattributed memo BEFORE & DURING erasure

Usage:
  python examples/integrations/langgraph_research_agent.py
  python examples/integrations/langgraph_research_agent.py --benchmark
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from ember import (
    CustodyVault,
    EmberLangChainRetriever,
    EmberLangChainVectorStore,
    SQLiteStoreAdapter,
    verify_ed25519_certificate,
)


def deterministic_embedder(text: str) -> list[float]:
    dims = 64
    vec = [0.0] * dims
    toks = re.findall(r"[a-z0-9_]+", text.lower())
    for tok in toks:
        h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
        vec[h % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


@dataclass
class AgentState:
    case_id: str
    subject_id: str
    retrieved_docs: list[Any] = field(default_factory=list)
    observed_memo_id: str = ""
    rogue_memo_id: str = ""
    coverage_before_erase: dict[str, Any] = field(default_factory=dict)
    certificate: dict[str, Any] = field(default_factory=dict)
    post_erase_observed_hits: int = -1
    post_erase_rogue_hits: int = -1
    bystander_surviving_hits: int = 0


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def run_agent_graph(*, benchmark: bool = False, iterations: int = 25) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = SQLiteStoreAdapter(str(root / "research_vectors.db"))
        try:
            with CustodyVault.open(
                root / "custody_data",
                key_dir=root / "custody_keys",
                tenant_id="clinical_research",
                adapter=store,
                embedder=deterministic_embedder,
                encrypt_vendor_payload=True,
            ) as vault:
                lc_store = EmberLangChainVectorStore(vault, subject_key="user_id")
                retriever: EmberLangChainRetriever = lc_store.as_retriever(search_kwargs={"k": 4})

                # Node 1: Ingest primary clinical notes tied to attested source document
                vault.register_source(
                    "trial_batch_9",
                    "Patient Nora takes daily lithium therapy. Patient Omar enrolled in sleep study.",
                    mode="sentence",
                    origin="ehr_export",
                    origin_kind="api",
                )
                lc_store.add_texts(
                    ["Patient Nora takes daily lithium therapy.", "Patient Omar enrolled in sleep study."],
                    metadatas=[
                        {"user_id": "patient_nora", "source_id": "trial_batch_9", "quote": "Patient Nora takes daily lithium therapy."},
                        {"user_id": "patient_omar", "source_id": "trial_batch_9", "quote": "Patient Omar enrolled in sleep study."},
                    ],
                )

                state = AgentState(case_id="trial_batch_9", subject_id="patient_nora")

                # Node 2 + Node 3: Retrieve + Summarize inside `trace_turn()` (lineage captured)
                with vault.trace_turn():
                    state.retrieved_docs = retriever.invoke("lithium therapy")
                    _check(len(state.retrieved_docs) == 1, "expected 1 retrieved document inside trace_turn")
                    _check(
                        state.retrieved_docs[0].metadata["ember_receipt_verified"] is True,
                        "expected retrieved document receipt to verify",
                    )

                    observed_ids = lc_store.add_texts(
                        ["Pharmacy protocol memo trace_tag_alpha: adjust electrolyte monitoring cadence."],
                        metadatas=[{"user_id": "pharmacy_ops"}],
                    )
                    state.observed_memo_id = observed_ids[0]

                # Node 4: Unobserved external tool writes WITHOUT trace_turn(), source_id, or derived_from!
                rogue_ids = lc_store.add_texts(
                    ["Background sidecar memo rogue_tag_beta: monitor hydration weekly."],
                    metadatas=[{"user_id": "rogue_background_tool"}],
                )
                state.rogue_memo_id = rogue_ids[0]

                # Node 5: audit_coverage() detects the unattributed write explicitly BEFORE certifying erasure
                cov_pre = vault.audit_coverage(subject_id="patient_nora")
                state.coverage_before_erase = cov_pre.as_dict()
                _check(
                    cov_pre.verdict == "bounded_with_warnings",
                    f"expected bounded_with_warnings coverage verdict, got {cov_pre.verdict}",
                )
                _check(cov_pre.unattributed_item_count == 1, "expected 1 unattributed write flagged")
                _check(
                    state.rogue_memo_id in cov_pre.unattributed_item_ids,
                    "expected rogue sidecar memo ID in unattributed_item_ids",
                )

                cert = vault.erase(
                    "patient_nora",
                    markers=["Nora", "lithium", "trace_tag_alpha"],
                )
                state.certificate = cert.as_dict()
                _check(verify_ed25519_certificate(state.certificate)[0] is True, "expected valid Ed25519 certificate")
                # Certificate trust_boundary honestly records the bounded coverage gap!
                _check(
                    cert.trust_boundary["coverage"]["verdict"] == "bounded_with_warnings",
                    "expected certificate trust_boundary coverage verdict to match pre-erasure warning",
                )
                _check(
                    cert.trust_boundary["coverage"]["unattributed_item_count"] == 1,
                    "expected certificate trust_boundary to record unattributed_item_count=1",
                )

                # Observed paraphrase (trace_tag_alpha) is GONE; unobserved rogue write honestly remained,
                # exactly matching the unattributed gap flagged by audit_coverage().
                observed_docs = retriever.invoke("trace_tag_alpha")
                rogue_docs = retriever.invoke("rogue_tag_beta")
                bystander_docs = retriever.invoke("sleep study")

                state.post_erase_observed_hits = sum(1 for d in observed_docs if "trace_tag_alpha" in d.page_content)
                state.post_erase_rogue_hits = sum(1 for d in rogue_docs if "rogue_tag_beta" in d.page_content)
                state.bystander_surviving_hits = sum(1 for d in bystander_docs if "sleep study" in d.page_content)

                _check(state.post_erase_observed_hits == 0, "expected traced summary memo to be erased")
                _check(state.post_erase_rogue_hits == 1, "expected unobserved rogue memo to stay accurately flagged")
                _check(state.bystander_surviving_hits == 1, "expected bystander clinical record to survive")

                out: dict[str, Any] = {
                    "app": "langgraph_research_agent",
                    "coverage_verdict": cov_pre.verdict,
                    "unattributed_items_detected": cov_pre.unattributed_item_ids,
                    "lineage_capture_rate": round(cov_pre.lineage_capture_rate, 4),
                    "observed_summary_post_erase_hits": state.post_erase_observed_hits,
                    "unobserved_tool_post_erase_hits": state.post_erase_rogue_hits,
                    "bystander_surviving_hits": state.bystander_surviving_hits,
                    "certificate_trust_boundary_coverage": cert.trust_boundary["coverage"],
                }

                if benchmark:
                    add_ms: list[float] = []
                    ret_ms: list[float] = []
                    for i in range(iterations):
                        t0 = time.perf_counter()
                        lc_store.add_texts(
                            [f"Research observation obs_tok_{i} for protocol validation."],
                            metadatas=[{"user_id": f"subj_b_{i}", "source_id": "trial_batch_9", "quote": "Patient Omar enrolled in sleep study."}],
                        )
                        add_ms.append((time.perf_counter() - t0) * 1000.0)

                        t1 = time.perf_counter()
                        _ = retriever.invoke(f"obs_tok_{i}")
                        ret_ms.append((time.perf_counter() - t1) * 1000.0)

                    def _p(vals: list[float], p: float) -> float:
                        s = sorted(vals)
                        return round(s[min(len(s) - 1, max(0, math.ceil((p / 100.0) * len(s)) - 1))], 3)

                    out["benchmark"] = {
                        "iterations": iterations,
                        "add_texts_p50_ms": _p(add_ms, 50),
                        "add_texts_p95_ms": _p(add_ms, 95),
                        "retriever_invoke_p50_ms": _p(ret_ms, 50),
                        "retriever_invoke_p95_ms": _p(ret_ms, 95),
                    }
                return out
        finally:
            store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ember LangGraph-Style Research Agent Integration Example")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(run_agent_graph(benchmark=args.benchmark, iterations=args.iterations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
