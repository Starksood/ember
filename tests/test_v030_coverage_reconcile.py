"""Comprehensive tests for Ember 0.3.0: coverage auditing, crash reconciliation, V4 chaos, and E2E apps."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any

import pytest

from ember import (
    AsyncCustodyVault,
    CustodyVault,
    EmberHttpApp,
    EmberMcpServer,
    InMemoryStoreAdapter,
    Mem0Adapter,
    PostgresLedger,
    SQLiteStoreAdapter,
    generate_html_report,
    run_v4_benchmark,
    verify_bundle,
    verify_ed25519_certificate,
)
from ember.audit import _FaultyDeleteAdapter
from ember.cli import main as cli_main


class _FakeMem0Client:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self._seq = 0

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
        del infer, agent_id, run_id
        self._seq += 1
        rid = f"m_{self._seq}"
        text = messages[0]["content"] if isinstance(messages, list) else str(messages)
        meta = dict(metadata or {})
        if user_id:
            meta["user_id"] = user_id
        self.records[rid] = {"id": rid, "memory": text, "metadata": meta, "user_id": user_id}
        return {"results": [{"id": rid, "memory": text, "event": "ADD"}]}

    def search(self, query: str, *, user_id: str | None = None, limit: int = 5, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        hits = []
        for rec in self.records.values():
            if user_id is not None and rec.get("user_id") != user_id:
                continue
            if any(tok in rec["memory"].lower() for tok in query.lower().split()):
                hits.append({"id": rec["id"], "memory": rec["memory"], "score": 1.0, "metadata": dict(rec["metadata"])})
        return {"results": hits[:limit]}

    def delete(self, memory_id: str) -> None:
        self.records.pop(memory_id, None)


class _MockPgConn:
    def __init__(self) -> None:
        self._db = sqlite3.connect(":memory:", check_same_thread=False)

    def cursor(self) -> "_MockPgCursor":
        return _MockPgCursor(self)

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()

    def close(self) -> None:
        self._db.close()


class _MockPgCursor:
    def __init__(self, conn: _MockPgConn) -> None:
        self.w = conn
        self._cur = conn._db.cursor()
        self.rowcount = -1

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        cleaned = " ".join(sql.split())
        if cleaned == "BEGIN":
            if not self.w._db.in_transaction:
                self._cur.execute("BEGIN IMMEDIATE")
            return
        if cleaned.startswith("SELECT pg_advisory_xact_lock"):
            self._cur.execute("SELECT 1")
            return
        sqlite_sql = (
            cleaned.replace("BIGINT", "INTEGER")
            .replace("ON CONFLICT (tenant_id, seq) DO UPDATE", "ON CONFLICT(tenant_id, seq) DO UPDATE")
            .replace("ON CONFLICT (tenant_id, source_id) DO UPDATE", "ON CONFLICT(tenant_id, source_id) DO UPDATE")
            .replace("ON CONFLICT (tenant_id, certificate_id) DO UPDATE", "ON CONFLICT(tenant_id, certificate_id) DO UPDATE")
            .replace("EXCLUDED.", "excluded.")
            .replace("%s", "?")
        )
        self._cur.execute(sqlite_sql, tuple(params))
        self.rowcount = self._cur.rowcount

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()

    def close(self) -> None:
        self._cur.close()


class _DownStoreAdapter(InMemoryStoreAdapter):
    def search(self, query: str, *, limit: int = 5, filters: dict[str, Any] | None = None):
        raise RuntimeError("backing vector cluster unreachable")


def test_coverage_verdicts_complete_warnings_and_incomplete(tmp_path: Path) -> None:
    # 1. Clean separated vault -> complete
    with CustodyVault.open(tmp_path / "clean_data", key_dir=tmp_path / "clean_keys") as v:
        v.register_source("s1", "Alice lives in Oslo. Bob plays chess.", span_mode="clause")
        v.remember("Alice lives in Oslo.", subjects=["alice"], source_id="s1", quote="Alice lives in Oslo.")
        v.remember("Bob plays chess.", subjects=["bob"], source_id="s1", quote="Bob plays chess.")
        cov = v.audit_coverage("alice")
        assert cov.verdict == "complete"
        assert cov.erasure_certifiable is True
        assert cov.source_coverage_rate == 1.0
        assert cov.lineage_capture_rate == 1.0
        assert cov.unverified_locations == ()
        assert cov.subject_preview is not None
        assert cov.subject_preview["records_identified"] == 1

    # 2. Colocated keystore, orphan source, unattributed item, and custom unverified location -> bounded_with_warnings
    with CustodyVault.open(
        tmp_path / "colocated_data",
        unverified_locations=["nightly_s3_snapshot"],
    ) as v_warn:
        v_warn.register_source("orphan_src", "Orphan document with zero items.")
        v_warn.register_source("used_src", "Nina likes tea. Dana plays flute.")
        v_warn.remember("Nina likes tea.", subjects=["nina"], source_id="used_src")
        v_warn.remember("Dana plays flute.", subjects=["dana"], source_id="used_src")
        unattr = v_warn.remember("Shadow note mentioning nina outside pipeline", subjects=["shadow_bot"])

        cov_w = v_warn.audit_coverage("nina", unverified_locations=["legacy_bi_cache"])
        assert cov_w.verdict == "bounded_with_warnings"
        assert cov_w.erasure_certifiable is True
        assert "orphan_src" in cov_w.orphan_source_ids
        assert unattr.item.item_id in cov_w.unattributed_item_ids
        assert "colocated_keystore_and_ledger" in cov_w.unverified_locations
        assert "nightly_s3_snapshot" in cov_w.unverified_locations
        assert "legacy_bi_cache" in cov_w.unverified_locations
        assert cov_w.subject_preview is not None
        assert cov_w.subject_preview["potential_unattributed_matches"] == 1
        assert "Shadow note" not in json.dumps(cov_w.as_dict())

        # Erase clean subject first while unattributed shadow note exists -> honest bounded_with_warnings
        cert = v_warn.erase("dana")
        tb_cov = cert.trust_boundary["coverage"]
        assert tb_cov["verdict"] == "bounded_with_warnings"
        assert tb_cov["unattributed_item_count"] >= 1

        # Failing-closed probe refusal when unattributed note leaks literal subject identifier -> incomplete
        from ember.reducer import ErasureIncomplete

        with pytest.raises(ErasureIncomplete):
            v_warn.erase("nina")
        assert v_warn.audit_coverage().verdict == "incomplete"

    # 3. Mem0Adapter without encrypt_vendor_payload flags mem0_internal_history_db
    m0 = _FakeMem0Client()
    with CustodyVault.ephemeral(adapter=Mem0Adapter(m0, infer=False), encrypt_vendor_payload=False) as v_m0:
        v_m0.register_source("msrc", "Kai writes Rust.")
        v_m0.remember("Kai writes Rust.", subjects=["kai"], source_id="msrc")
        cov_m0 = v_m0.audit_coverage()
        assert cov_m0.verdict == "bounded_with_warnings"
        assert "mem0_internal_history_db" in cov_m0.unverified_locations

    # 4. Unreachable adapter -> incomplete
    with CustodyVault.ephemeral(adapter=_DownStoreAdapter()) as v_down:
        cov_down = v_down.audit_coverage()
        assert cov_down.verdict == "incomplete"
        assert cov_down.erasure_certifiable is False
        assert cov_down.verified_stores[0]["reachable"] is False


def test_ledger_integrity_failure_triggers_incomplete_coverage(tmp_path: Path) -> None:
    data_dir = tmp_path / "tamper_data"
    key_dir = tmp_path / "tamper_keys"
    with CustodyVault.open(data_dir, key_dir=key_dir) as v:
        v.register_source("s_t", "Tamper baseline note.")
        v.remember("Tamper baseline note.", subjects=["tamper_subj"], source_id="s_t")

    # Tamper directly with sqlite ledger_events hash
    with sqlite3.connect(data_dir / "ledger.db") as conn:
        conn.execute("UPDATE ledger_events SET event_hash = 'sha256:0000' WHERE seq = 0")
        conn.commit()

    # Use skip replay validation by calling ledger directly via a fresh vault with clean replay?
    # Opening a tampered ledger raises LedgerIntegrityError on replay, so test analyze_coverage or inspect chain:
    from ember.coverage import analyze_coverage
    from ember.reducer import EmberState

    rep = analyze_coverage(
        tenant_id="default",
        state=EmberState.empty(),
        events=(),
        ledger_chain_ok=False,
        verified_stores=[{"name": "InMemoryStoreAdapter", "reachable": True}],
    )
    assert rep.verdict == "incomplete"
    assert rep.erasure_certifiable is False


def test_durable_pending_erasure_reconcile_and_scrubbing(tmp_path: Path) -> None:
    data_dir = tmp_path / "crash_data"
    key_dir = tmp_path / "crash_keys"
    store_path = tmp_path / "store.db"

    sql_store1 = SQLiteStoreAdapter(str(store_path))
    faulty = _FaultyDeleteAdapter(sql_store1, fail_count=1)
    v1 = CustodyVault.open(data_dir, key_dir=key_dir, adapter=faulty)
    try:
        v1.register_source("doc1", "Maya Lin allergic to shellfish. Leo Park plays bass.")
        v1.remember("Maya Lin allergic to shellfish.", subjects=["maya"], source_id="doc1")
        v1.remember("Leo Park plays bass.", subjects=["leo"], source_id="doc1")

        with pytest.raises(ConnectionError, match="injected transient store delete timeout"):
            v1.erase("maya", aliases=["Maya Lin"], probes=["shellfish secret probe"], markers=["Maya", "shellfish"])

        pending = v1.pending_erasures()
        assert len(pending) == 1
        assert pending[0].status == "failed"
        assert "ConnectionError" in (pending[0].last_error or "")
        # Safe summary MUST strip plaintext probe strings
        assert "shellfish secret probe" not in json.dumps(pending[0].safe_summary())
        assert v1.audit_coverage().verdict == "incomplete"
    finally:
        v1.close()
        sql_store1.close()

    # Reopen with healthy adapter & auto_reconcile=True
    sql_store2 = SQLiteStoreAdapter(str(store_path))
    try:
        with CustodyVault.open(data_dir, key_dir=key_dir, adapter=sql_store2, auto_reconcile=True) as v2:
            assert v2.pending_erasures() == []
            assert v2.audit_coverage().verdict == "complete"
            certs = v2.ledger.list_certificates(v2.tenant_id)
            assert len(certs) == 1
            assert verify_ed25519_certificate(certs[0])[0] is True
            assert v2.search("shellfish").abstained
            assert not v2.search("bass").abstained

            # Completed journal rows MUST have probes, aliases, markers scrubbed
            all_rows = v2.ledger.list_pending_erasures(v2.tenant_id, include_completed=True)
            assert len(all_rows) == 1
            assert all_rows[0].status == "completed"
            assert all_rows[0].probes == ()
            assert all_rows[0].aliases == ()
            assert all_rows[0].markers == ()
            assert all_rows[0].certificate_id == certs[0]["certificate_id"]
    finally:
        sql_store2.close()


def test_postgres_pending_erasure_and_async_vault(tmp_path: Path) -> None:
    pg_conn = _MockPgConn()
    ledger = PostgresLedger(connection=pg_conn, table_prefix="t030_")
    inner = InMemoryStoreAdapter()
    faulty = _FaultyDeleteAdapter(inner, fail_count=1)

    async def _run() -> None:
        vault = CustodyVault(
            ledger=ledger,
            keystore=__import__("ember").MemoryKeyStore(),
            signer=__import__("ember").Ed25519Signer.generate(),
            adapter=faulty,
            keys_separated_from_ledger=True,
        )
        av = AsyncCustodyVault(vault)
        try:
            await av.register_source("pgdoc", "Zara Chen works on optics.")
            await av.remember("Zara Chen works on optics.", subjects=["zara"], source_id="pgdoc")
            with pytest.raises(ConnectionError):
                await av.erase("zara", markers=["Zara", "optics"])
            p_list = await av.pending_erasures()
            assert len(p_list) == 1
            cov1 = await av.audit_coverage()
            assert cov1.verdict == "incomplete"

            rec_rep = await av.reconcile_erasures()
            assert rec_rep.all_clean is True
            assert len(rec_rep.certificates) == 1
            cov2 = await av.audit_coverage()
            assert cov2.verdict == "complete"

            pg_done = ledger.list_pending_erasures("default", include_completed=True)
            assert len(pg_done) == 1
            assert pg_done[0].status == "completed"
            assert pg_done[0].probes == ()
        finally:
            await av.close()

    asyncio.run(_run())


def test_surfaces_http_mcp_cli_and_html_report(tmp_path: Path) -> None:
    inner = InMemoryStoreAdapter()
    faulty = _FaultyDeleteAdapter(inner, fail_count=1)
    xss_loc = "<script>alert('xss')</script>"
    with CustodyVault.open(
        tmp_path / "srv_data",
        key_dir=tmp_path / "srv_keys",
        adapter=faulty,
        unverified_locations=[xss_loc],
    ) as vault:
        app = EmberHttpApp(vault)
        mcp = EmberMcpServer(vault)

        tool_names = {t["name"] for t in mcp.list_tools()}
        assert "ember_audit_coverage" in tool_names
        assert "ember_reconcile_erasures" in tool_names

        mcp.call_tool("ember_register_source", {"source_id": "s_http", "text": "Owen Reid studies corals."})
        mcp.call_tool(
            "ember_remember",
            {"text": "Owen Reid studies corals.", "subjects": ["owen"], "source_id": "s_http"},
        )

        # First erase fails due to injected ConnectionError
        with pytest.raises(ConnectionError):
            vault.erase("owen", markers=["Owen", "corals"])

        # MCP audit_coverage sees incomplete
        cov_mcp, is_err = mcp.call_tool("ember_audit_coverage", {"subject_id": "owen"})
        assert not is_err
        assert cov_mcp["verdict"] == "incomplete"

        # HTTP GET /v1/coverage also sees incomplete
        st_c, _, body_c = app.handle_request("GET", "/v1/coverage?unverified_location=extra_cache")
        assert st_c == 200
        http_cov = json.loads(body_c)
        assert http_cov["verdict"] == "incomplete"
        assert "extra_cache" in http_cov["unverified_locations"]

        # Reconcile over HTTP POST /v1/reconcile
        st_r, _, body_r = app.handle_request("POST", "/v1/reconcile", b"{}")
        assert st_r == 200
        rec = json.loads(body_r)
        assert rec["all_clean"] is True
        assert len(rec["certificate_ids"]) == 1

        # Bundle & HTML report include coverage Section 5 and properly escape XSS string
        bundle = vault.export_bundle()
        assert bundle["coverage"]["verdict"] == "bounded_with_warnings"
        v_rep = verify_bundle(bundle, require_signature_check=True)
        assert v_rep["ok"] is True
        html_doc = generate_html_report(bundle, report=v_rep)
        assert "5. Bounded Erasure Coverage" in html_doc
        assert "<script>alert" not in html_doc
        assert "&lt;script&gt;alert" in html_doc

    # CLI coverage + reconcile + benchmark --suite v4
    cli_dir = tmp_path / "cli_v"
    cli_keys = tmp_path / "cli_k"
    assert cli_main(["--vault", str(cli_dir), "--key-dir", str(cli_keys), "add-source", "cs1", "--text", "Iris Tan runs marathons."]) == 0
    assert cli_main(["--vault", str(cli_dir), "--key-dir", str(cli_keys), "remember", "Iris Tan runs marathons.", "--subject", "iris", "--source-id", "cs1"]) == 0
    assert cli_main(["--vault", str(cli_dir), "--key-dir", str(cli_keys), "coverage", "--subject", "iris"]) == 0
    assert cli_main(["--vault", str(cli_dir), "--key-dir", str(cli_keys), "reconcile"]) == 0


def test_suite_v4_and_integration_templates(tmp_path: Path) -> None:
    report = run_v4_benchmark(latency_iterations=10)
    assert report["passed"] is True
    assert report["paraphrase_matrix"]["observed_trace_turn_leaks"] == 0
    assert report["paraphrase_matrix"]["unobserved_out_of_band_leaks"] == report["subjects"]
    assert report["paraphrase_matrix"]["coverage_audit_detection_rate"] == 1.0
    assert report["chaos_recovery"]["passed"] is True
    assert report["latency_ms"]["erase"]["p50_ms"] > 0

    repo_root = Path(__file__).resolve().parent.parent
    html_out = tmp_path / "support_dossier.html"
    p1 = subprocess.run(
        [sys.executable, str(repo_root / "examples/integrations/support_bot_app.py"), "--html", str(html_out)],
        capture_output=True,
        text=True,
        check=True,
    )
    res1 = json.loads(p1.stdout)
    assert res1["pre_erasure_verdict"] == "complete"
    assert res1["outage_verdict"] == "incomplete"
    assert res1["reconciled"] is True
    assert res1["post_erasure_verdict"] == "complete"
    assert res1["alice_post_erase_hits"] == 0
    assert html_out.exists()
    assert "Bounded Erasure Coverage" in html_out.read_text(encoding="utf-8")

    p2 = subprocess.run(
        [sys.executable, str(repo_root / "examples/integrations/langgraph_research_agent.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    res2 = json.loads(p2.stdout)
    assert res2["coverage_verdict"] == "bounded_with_warnings"
    assert len(res2["unattributed_items_detected"]) == 1
    assert res2["observed_summary_post_erase_hits"] == 0
    assert res2["unobserved_tool_post_erase_hits"] == 1
    assert res2["bystander_surviving_hits"] == 1
