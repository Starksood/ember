"""Verification suite for Ember Phase 2 Shift C (HTTP sidecar, MCP server, TS SDK, HTML dossier, SUITE_V3)."""

from __future__ import annotations

import http.client
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any

import pytest

from ember import (
    CustodyVault,
    EmberHttpApp,
    EmberMcpServer,
    SUITE_V3,
    CustodyVault as VaultCls,
    generate_html_report,
    make_http_server,
    run_v3_benchmark,
    start_background_server,
    verify_bundle,
    verify_suite_hygiene,
)
from ember.adapters import InMemoryStoreAdapter
from ember.cli import main as cli_main


# ── C1: HTTP Sidecar + Auth + Live TypeScript SDK ────────────────────────────


def test_c1_http_sidecar_auth_tenant_limits_and_fail_closed(tmp_path: Path) -> None:
    # Non-loopback host without auth MUST refuse to start
    v_mem = CustodyVault.ephemeral()
    with pytest.raises(ValueError, match="non-loopback"):
        make_http_server(v_mem, host="0.0.0.0", port=0)  # noqa: S104
    v_mem.close()

    api_token = "tok_" + secrets.token_hex(12)
    vault = CustodyVault.open(tmp_path / "srv_vault", key_dir=tmp_path / "srv_keys", tenant_id="acme")
    app = EmberHttpApp(vault, bearer_token=api_token, require_tenant_header=True, max_body_bytes=4096)

    # /health works without auth
    st, _, raw = app.handle_request("GET", "/health")
    assert st == 200
    assert json.loads(raw)["tenant_id"] == "acme"

    # Missing Bearer token -> 401
    st, _, _ = app.handle_request("POST", "/v1/search", b'{"query":"test"}', {"X-Ember-Tenant": "acme"})
    assert st == 401

    # Wrong Bearer token -> 401
    st, _, _ = app.handle_request(
        "POST",
        "/v1/search",
        b'{"query":"test"}',
        {"Authorization": "Bearer wrong", "X-Ember-Tenant": "acme"},
    )
    assert st == 401

    # Missing tenant header when required -> 400
    st, _, _ = app.handle_request("POST", "/v1/search", b'{"query":"test"}', {"Authorization": f"Bearer {api_token}"})
    assert st == 400

    # Wrong tenant header -> 403
    st, _, _ = app.handle_request(
        "POST",
        "/v1/search",
        b'{"query":"test"}',
        {"Authorization": f"Bearer {api_token}", "X-Ember-Tenant": "evil_tenant"},
    )
    assert st == 403

    # Oversized body -> 413
    st, _, _ = app.handle_request(
        "POST",
        "/v1/sources",
        b"x" * 5000,
        {"Authorization": f"Bearer {api_token}", "X-Ember-Tenant": "acme"},
    )
    assert st == 413

    auth_hdrs = {"Authorization": f"Bearer {api_token}", "X-Ember-Tenant": "acme"}

    # Bare string for subjects must reject cleanly (400) rather than iterating characters
    st, _, _ = app.handle_request(
        "POST",
        "/v1/remember",
        json.dumps({"text": "Alice lives in Oslo", "subjects": "alice"}).encode(),
        auth_hdrs,
    )
    assert st == 400

    st, _, raw = app.handle_request(
        "POST",
        "/v1/sources",
        json.dumps(
            {
                "source_id": "s_http",
                "text": "Alice lives in Oslo, and Ben lives in Cork.",
                "span_mode": "clause",
            }
        ).encode(),
        auth_hdrs,
    )
    assert st == 200
    assert len(json.loads(raw)["merkle_root"]) == 64

    st, _, _ = app.handle_request(
        "POST",
        "/v1/remember",
        json.dumps({"text": "Alice lives in Oslo", "subjects": ["alice"], "source_id": "s_http"}).encode(),
        auth_hdrs,
    )
    assert st == 200

    st, _, _ = app.handle_request(
        "POST",
        "/v1/remember",
        json.dumps({"text": "Ben lives in Cork", "subjects": ["ben"], "source_id": "s_http"}).encode(),
        auth_hdrs,
    )
    assert st == 200

    st, _, raw = app.handle_request("POST", "/v1/erase", json.dumps({"subject_id": "alice"}).encode(), auth_hdrs)
    assert st == 200
    assert json.loads(raw)["key_destroyed"] is True

    # Tombstone blocks re-ingest -> 403
    st, _, _ = app.handle_request(
        "POST",
        "/v1/remember",
        json.dumps({"text": "Alice returned to Oslo", "subjects": ["alice"]}).encode(),
        auth_hdrs,
    )
    assert st == 403

    st, _, raw = app.handle_request("POST", "/v1/checkpoint", b"{}", auth_hdrs)
    assert st == 200
    assert json.loads(raw)["status"] == "checkpointed"

    st, _, raw = app.handle_request("GET", "/v1/bundle", b"", auth_hdrs)
    assert st == 200
    assert verify_bundle(json.loads(raw), require_signature_check=True)["ok"] is True
    vault.close()


def test_c1_http_sidecar_live_tcp_and_typescript_sdk(tmp_path: Path) -> None:
    class _BrokenDeleteAdapter(InMemoryStoreAdapter):
        def delete(self, record_ids):
            return 0

    # 1. Verify 422 ErasureIncomplete over HTTP when backing adapter leaks
    bad_vault = VaultCls.ephemeral(adapter=_BrokenDeleteAdapter())
    bad_vault.remember("Secret token XYZ", subjects=["alice"])
    bad_app = EmberHttpApp(bad_vault)
    st, _, raw = bad_app.handle_request("POST", "/v1/erase", json.dumps({"subject_id": "alice"}).encode())
    assert st == 422
    assert "leaking_probes" in json.loads(raw)
    bad_vault.close()

    # 2. Spin up real HTTP server and exercise with both Python http.client and TypeScript SDK!
    live_secret = "tok_" + secrets.token_hex(12)
    vault = CustodyVault.open(tmp_path / "live_vault", key_dir=tmp_path / "live_keys", tenant_id="ts_demo")
    httpd, thread = start_background_server(vault, host="127.0.0.1", port=0, bearer_token=live_secret)
    port = httpd.server_address[1]
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read())["tenant_id"] == "ts_demo"

        conn.close()

        # Invalid Content-Length and unsupported chunked transfer encoding return proper 400/411
        conn_bad_len = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn_bad_len.putrequest("POST", "/v1/search")
        conn_bad_len.putheader("Content-Length", "not-an-int")
        conn_bad_len.endheaders()
        r400 = conn_bad_len.getresponse()
        assert r400.status == 400
        r400.read()
        conn_bad_len.close()

        conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn2.putrequest("POST", "/v1/search")
        conn2.putheader("Transfer-Encoding", "chunked")
        conn2.endheaders()
        r411 = conn2.getresponse()
        assert r411.status == 411
        r411.read()
        conn2.close()

        node_bin = shutil.which("node")
        if node_bin is not None:
            ver_proc = subprocess.run(
                [node_bin, "-p", "process.versions.node"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            parts = tuple(int(p) for p in ver_proc.stdout.strip().split(".")[:2] if p.isdigit())
            node_flags: list[str] | None = None
            if len(parts) == 2 and (parts[0] > 23 or (parts[0] == 23 and parts[1] >= 6)):
                node_flags = ["--test"]
            elif len(parts) == 2 and ((parts[0] == 22 and parts[1] >= 6) or (parts[0] == 23 and parts[1] < 6)):
                node_flags = ["--experimental-strip-types", "--test"]

            if node_flags is not None:
                sdk_dir = Path(__file__).resolve().parent.parent / "sdk" / "ts"
                env = dict(os.environ)
                env["EMBER_LIVE_TEST_URL"] = f"http://127.0.0.1:{port}"
                env["EMBER_LIVE_TEST_TOKEN"] = live_secret  # noqa: S105
                env["EMBER_LIVE_TEST_TENANT"] = "ts_demo"
                proc = subprocess.run(
                    [node_bin, *node_flags, "test/client.test.mts"],
                    cwd=str(sdk_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=20,
                )
                assert proc.returncode == 0, f"TS SDK test failed:\n{proc.stdout}\n{proc.stderr}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)
        vault.close()


# ── C1: MCP Server (JSON-RPC 2.0 over stdio) ─────────────────────────────────


def test_c1_mcp_stdio_server_protocol_and_custody_flow(tmp_path: Path) -> None:
    vault = CustodyVault.open(tmp_path / "mcp_vault", key_dir=tmp_path / "mcp_keys")
    srv = EmberMcpServer(vault)

    requests: list[Any] = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": {"name": "test"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ember_register_source",
                "arguments": {
                    "source_id": "mcp_doc",
                    "text": "Clara works at CERN, and Devon plays oboe.",
                    "span_mode": "clause",
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "ember_remember",
                "arguments": {
                    "text": "Clara works at CERN",
                    "subjects": ["clara"],
                    "source_id": "mcp_doc",
                    "quote": "Clara works at CERN,",
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "ember_remember",
                "arguments": {
                    "text": "Devon plays oboe",
                    "subjects": ["devon"],
                    "source_id": "mcp_doc",
                    "quote": "and Devon plays oboe.",
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "ember_erase", "arguments": {"subject_id": "clara", "probes": ["CERN"]}},
        },
        # Re-ingest must return isError=True cleanly without crashing stdio loop
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "ember_remember", "arguments": {"text": "Clara returned", "subjects": ["clara"]}},
        },
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "ember_search", "arguments": {"query": "oboe"}},
        },
        {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "ember_export_bundle", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 10, "method": "unknown/method"},
        # Missing required argument returns isError=True ValueError rather than -32601 Unknown tool
        {"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "ember_remember", "arguments": {"text": "x"}}},
    ]

    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\nnot-valid-json\n")
    stdout = io.StringIO()
    assert srv.run_stdio(stdin=stdin, stdout=stdout) == 0

    lines = [json.loads(line) for line in stdout.getvalue().strip().splitlines()]
    # notification (id=None) produces no response, plus invalid-json yields 1 response -> 12 lines
    assert len(lines) == 12
    by_id = {row["id"]: row for row in lines if row.get("id") is not None}

    assert by_id[1]["result"]["serverInfo"]["name"] == "ember-mcp"
    tool_names = {t["name"] for t in by_id[2]["result"]["tools"]}
    assert {"ember_register_source", "ember_remember", "ember_search", "ember_erase", "ember_verify_receipt", "ember_export_bundle"} <= tool_names

    erase_res = json.loads(by_id[6]["result"]["content"][0]["text"])
    assert erase_res["key_destroyed"] is True
    assert by_id[7]["result"]["isError"] is True
    assert "TombstonedSubjectError" in by_id[7]["result"]["content"][0]["text"]
    assert by_id[11]["result"]["isError"] is True
    assert "ValueError" in by_id[11]["result"]["content"][0]["text"]

    search_res = json.loads(by_id[8]["result"]["content"][0]["text"])
    assert len(search_res["hits"]) == 1
    assert search_res["hits"][0]["receipt_verified"] is True

    bundle_res = json.loads(by_id[9]["result"]["content"][0]["text"])
    assert verify_bundle(bundle_res, require_signature_check=True)["ok"] is True

    assert by_id[10]["error"]["code"] == -32601
    assert any(row.get("error", {}).get("code") == -32700 for row in lines)
    vault.close()


# ── C2: Compliance Dossier HTML Generator & XSS Safety ───────────────────────


def test_c2_html_compliance_dossier_and_xss_escaping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vault = CustodyVault.open(tmp_path / "html_vault", key_dir=tmp_path / "html_keys", tenant_id="med<corp>")
    vault.register_source(
        "doc_xss",
        "Alice secret record. Bob notes: <script>alert('xss')</script> \" onerror=\"alert(1).",
    )
    vault.remember("Alice secret record.", subjects=["alice"], source_id="doc_xss", quote="Alice secret record.")
    vault.remember(
        "Bob notes: <script>alert('xss')</script> \" onerror=\"alert(1).",
        subjects=["bob<tag>"],
        source_id="doc_xss",
        quote="Bob notes: <script>alert('xss')</script> \" onerror=\"alert(1).",
    )
    vault.erase("alice")

    bundle = vault.export_bundle()
    vault.close()

    bundle_path = tmp_path / "bundle.json"
    html_path = tmp_path / "dossier.html"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    assert cli_main(["verify", str(bundle_path), "--require-signature", "--html", str(html_path)]) == 0
    _ = capsys.readouterr()

    rendered = html_path.read_text(encoding="utf-8")
    assert "VERIFICATION PASSED" in rendered
    assert "[redacted]" in rendered
    # Must be self-contained (zero external CDN URLs or scripts)
    assert "http://" not in rendered
    assert "https://" not in rendered
    assert "<script" not in rendered.lower()
    # Dynamic untrusted XSS payload must be strictly HTML-escaped!
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in rendered
    assert "med&lt;corp&gt;" in rendered

    # Tampered bundle renders FAIL dossier cleanly
    tampered = json.loads(json.dumps(bundle))
    tampered["certificates"][0]["signature"] = "ed25519:" + ("00" * 64)
    bad_html = generate_html_report(tampered, require_signature_check=True)
    assert "VERIFICATION FAILED" in bad_html


# ── C3: SUITE_V3 Benchmark & CLI Integration ─────────────────────────────────


def test_c3_suite_v3_benchmark_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    hygiene = verify_suite_hygiene(SUITE_V3)
    assert hygiene["subjects"] == 12
    assert hygiene["probes"] == 204

    report = run_v3_benchmark(SUITE_V3)
    assert report["suite"] == "ember_erasure_v3"
    assert report["passed"] is True
    assert report["baseline_naive_leaks"] == 204
    assert report["ember_leaks"] == 0
    assert report["all_bystanders_survived"] is True
    assert report["all_bystander_receipts_valid"] is True
    assert report["all_subleaf_redactions_ok"] is True
    assert report["all_turn_trace_auto_derived_ok"] is True
    assert report["all_encrypted_vendor_zero_plaintext"] is True
    assert report["all_zombie_content_blocked"] is True
    assert report["all_certificates_valid"] is True

    assert cli_main(["benchmark", "--suite", "v3"]) == 0
    text_out = capsys.readouterr().out
    assert "SUITE V3" in text_out
    assert "Sub-leaf redactions  : True" in text_out
    assert "Vendor zero-plaintext: True" in text_out
