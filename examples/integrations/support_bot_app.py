"""Multi-tenant Customer Support Ticket Assistant integrated with Ember custody, coverage & crash recovery.

Demonstrates:
  1. Sub-leaf clause provenance (`span_mode="clause"`) on shared ticket transcripts.
  2. Automatic read->write lineage capture (`with mem.trace_turn():`) on escalation summaries.
  3. Pre-erasure readiness check (`GET /v1/coverage`) over `EmberHttpApp`.
  4. Transient store outage mid-`erase()`, `incomplete` verdict detection, and `POST /v1/reconcile`.
  5. Offline bundle + HTML compliance dossier export and real p50/p95 overhead measurement.

Usage:
  python examples/integrations/support_bot_app.py
  python examples/integrations/support_bot_app.py --benchmark --html /tmp/support_dossier.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from ember import (
    CustodyVault,
    EmberHttpApp,
    EmberMem0,
    Mem0Adapter,
    verify_bundle,
    verify_ed25519_certificate,
)
from ember.report import generate_html_report


class LocalSupportMem0Client:
    """In-process Mem0-compatible store supporting transient delete failures for chaos testing."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self._seq = 0
        self.inject_delete_failure = False

    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = False,
    ) -> dict[str, Any]:
        del infer
        self._seq += 1
        rid = f"ticket_mem_{self._seq}"
        text = (
            messages[0]["content"]
            if isinstance(messages, list) and messages and isinstance(messages[0], dict)
            else str(messages)
        )
        meta = dict(metadata or {})
        if user_id:
            meta["user_id"] = user_id
        if agent_id:
            meta["agent_id"] = agent_id
        if run_id:
            meta["run_id"] = run_id
        self.records[rid] = {"id": rid, "memory": text, "metadata": meta, "user_id": user_id}
        return {"results": [{"id": rid, "memory": text, "event": "ADD"}]}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[dict[str, Any]] = []
        for rec in self.records.values():
            meta = rec["metadata"]
            if user_id is not None and meta.get("user_id") != user_id and rec.get("user_id") != user_id:
                continue
            if agent_id is not None and meta.get("agent_id") != agent_id:
                continue
            if run_id is not None and meta.get("run_id") != run_id:
                continue
            r_toks = set(re.findall(r"[a-z0-9]+", rec["memory"].lower()))
            ov = len(q_toks & r_toks)
            if ov > 0:
                scored.append(
                    {
                        "id": rec["id"],
                        "memory": rec["memory"],
                        "score": ov / max(1, len(q_toks)),
                        "metadata": dict(meta),
                    }
                )
        scored.sort(key=lambda r: -r["score"])
        return {"results": scored[:limit]}

    def delete(self, memory_id: str) -> None:
        if self.inject_delete_failure:
            raise ConnectionError(f"upstream vector backend unavailable while deleting {memory_id}")
        self.records.pop(memory_id, None)


def _http_json(app: EmberHttpApp, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode("utf-8") if body is not None else b""
    status, _, resp_bytes = app.handle_request(method, path, raw)
    return status, json.loads(resp_bytes.decode("utf-8"))


def _pct(samples_ms: list[float], pct: float) -> float:
    ordered = sorted(samples_ms)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 3)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def run_support_bot_demo(*, html_out: Path | None = None, benchmark_iters: int = 0) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        client = LocalSupportMem0Client()
        adapter = Mem0Adapter(client, infer=False)

        # Keep keys in a separate directory from ledger backups & encrypt vendor payloads
        # so mem0 internal history logs receive zero plaintext PII.
        with CustodyVault.open(
            root / "vault_data",
            key_dir=root / "vault_keys",
            tenant_id="acme_support",
            adapter=adapter,
            encrypt_vendor_payload=True,
        ) as vault:
            mem = EmberMem0(vault)
            app = EmberHttpApp(vault)

            # 1. Shared ticket transcript with sub-leaf clause provenance
            transcript = (
                "Alice Chen reported billing double-charge on invoice INV-8821, "
                "while Bob Ross requested a Dark Mode desktop UI shortcut."
            )
            mem.register_source(
                "ticket_batch_42",
                transcript,
                mode="sentence",
                span_mode="clause",
                origin="zendesk_webhook",
                origin_kind="api",
            )

            # 2. Extract separate subject memories tied to exact clauses
            alice_mem = mem.add(
                "Alice Chen reported billing double-charge on invoice INV-8821",
                user_id="customer_alice",
                source_id="ticket_batch_42",
                quote="Alice Chen reported billing double-charge on invoice INV-8821,",
                aliases=["Alice Chen", "INV-8821"],
            )
            mem.add(
                "Bob Ross requested a Dark Mode desktop UI shortcut.",
                user_id="customer_bob",
                source_id="ticket_batch_42",
                quote="while Bob Ross requested a Dark Mode desktop UI shortcut.",
            )

            # 3. Observed escalation triage summary inside trace_turn()
            with mem.trace_turn():
                _ = mem.search("INV-8821", user_id="customer_alice")
                esc_mem = mem.add(
                    "Tier-2 refund escalation queue item #409 pending ledger credit",
                    user_id="support_queue_triage",
                )

            _check(
                alice_mem["ember_item_id"] in esc_mem["ember_derived_from"],
                "expected escalation memo to record dependency on Alice's memory",
            )

            # 4. Pre-erasure coverage check over HTTP
            cov_status, pre_cov = _http_json(app, "GET", "/v1/coverage?subject_id=customer_alice")
            _check(cov_status == 200, f"expected 200 from /v1/coverage, got {cov_status}")
            _check(pre_cov["verdict"] == "complete", f"expected complete verdict, got {pre_cov['verdict']}")
            _check(
                pre_cov["subject_preview"]["records_identified"] == 2,
                "expected both primary memory and escalation memo in closure preview",
            )

            # 5. Simulate transient backend timeout mid-erase, verify incomplete verdict, then reconcile
            client.inject_delete_failure = True
            erase_failed = False
            try:
                mem.erase("customer_alice", aliases=["Alice Chen", "INV-8821"])
            except ConnectionError:
                erase_failed = True
            _check(erase_failed, "expected ConnectionError during injected backend outage")

            _, outage_cov = _http_json(app, "GET", "/v1/coverage")
            _check(outage_cov["verdict"] == "incomplete", f"expected incomplete verdict, got {outage_cov['verdict']}")
            _check(outage_cov["pending_erasure_count"] == 1, "expected 1 pending erasure during outage")

            # Backend recovers; HTTP /v1/reconcile finishes cascade deletion & signs Ed25519 cert
            client.inject_delete_failure = False
            rec_status, rec_rep = _http_json(app, "POST", "/v1/reconcile", {})
            _check(rec_status == 200, f"expected 200 from /v1/reconcile, got {rec_status}")
            _check(rec_rep["all_clean"] is True, "expected reconcile_erasures to report all_clean=True")
            _check(len(rec_rep["certificate_ids"]) == 1, "expected 1 minted certificate after reconcile")

            _, post_cov = _http_json(app, "GET", "/v1/coverage")
            _check(post_cov["verdict"] == "complete", f"expected complete post-reconcile verdict, got {post_cov['verdict']}")

            # 6. Verify Alice abstains completely while Bob + Merkle clause receipt remain intact
            alice_post_res = mem.search("INV-8821", user_id="customer_alice")["results"]
            _check(alice_post_res == [], "expected erased Alice memory to abstain completely")
            _check(
                mem.search("refund escalation queue", user_id="support_queue_triage")["results"] == [],
                "expected derived escalation memo to cascade-erase",
            )
            bob_res = mem.search("Dark Mode", user_id="customer_bob")["results"]
            _check(len(bob_res) == 1, "expected bystander Bob memory to survive")
            _check(bob_res[0]["ember_receipt_verified"] is True, "expected surviving Merkle clause receipt to verify")

            # 7. Export offline verification bundle & verify certificate
            _, bundle = _http_json(app, "GET", "/v1/bundle")
            v_rep = verify_bundle(bundle, require_signature_check=True)
            _check(v_rep["ok"] is True, "expected offline bundle verification to succeed")
            _check(
                verify_ed25519_certificate(bundle["certificates"][0])[0] is True,
                "expected Ed25519 deletion certificate to verify",
            )

            if html_out is not None:
                html_out.parent.mkdir(parents=True, exist_ok=True)
                html_out.write_text(generate_html_report(bundle, report=v_rep), encoding="utf-8")

        summary: dict[str, Any] = {
            "app": "support_bot_app",
            "pre_erasure_verdict": pre_cov["verdict"],
            "outage_verdict": outage_cov["verdict"],
            "reconciled": rec_rep["all_clean"],
            "post_erasure_verdict": post_cov["verdict"],
            "alice_post_erase_hits": len(alice_post_res),
            "bob_surviving_receipt_verified": bob_res[0]["ember_receipt_verified"],
            "bundle_verified": v_rep["ok"],
        }

        if benchmark_iters > 0:
            raw_add_ms: list[float] = []
            raw_search_ms: list[float] = []
            wrap_add_ms: list[float] = []
            wrap_search_ms: list[float] = []

            raw_client = LocalSupportMem0Client()
            wrap_client = LocalSupportMem0Client()
            with CustodyVault.open(
                root / "bench_data",
                key_dir=root / "bench_keys",
                adapter=Mem0Adapter(wrap_client, infer=False),
                encrypt_vendor_payload=True,
            ) as b_vault:
                b_mem = EmberMem0(b_vault)
                for idx in range(benchmark_iters):
                    msg = f"Support inquiry ticket token_s_{idx} billing refund check"
                    t0 = time.perf_counter()
                    raw_client.add(msg, user_id=f"u_{idx}")
                    raw_add_ms.append((time.perf_counter() - t0) * 1000.0)

                    t1 = time.perf_counter()
                    raw_client.search(f"token_s_{idx}", user_id=f"u_{idx}")
                    raw_search_ms.append((time.perf_counter() - t1) * 1000.0)

                    t2 = time.perf_counter()
                    b_mem.add(msg, user_id=f"u_{idx}")
                    wrap_add_ms.append((time.perf_counter() - t2) * 1000.0)

                    t3 = time.perf_counter()
                    b_mem.search(f"token_s_{idx}", user_id=f"u_{idx}")
                    wrap_search_ms.append((time.perf_counter() - t3) * 1000.0)

            summary["benchmark"] = {
                "iterations": benchmark_iters,
                "raw_mem0_add_p50_ms": _pct(raw_add_ms, 50),
                "ember_mem0_add_p50_ms": _pct(wrap_add_ms, 50),
                "ember_mem0_add_p95_ms": _pct(wrap_add_ms, 95),
                "raw_mem0_search_p50_ms": _pct(raw_search_ms, 50),
                "ember_mem0_search_p50_ms": _pct(wrap_search_ms, 50),
                "ember_mem0_search_p95_ms": _pct(wrap_search_ms, 95),
            }
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Ember Customer Support Bot Integration Example")
    parser.add_argument("--benchmark", action="store_true", help="Measure baseline vs wrapped p50/p95 overhead")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--html", default=None, help="Optional output path for compliance dossier HTML")
    args = parser.parse_args()

    res = run_support_bot_demo(
        html_out=Path(args.html) if args.html else None,
        benchmark_iters=args.iterations if args.benchmark else 0,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
