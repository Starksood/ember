"""Zero-CDN self-contained HTML compliance dossier generator (`ember verify --html`)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import html
import json
from typing import Any

from .verify import _render_spans_text, verify_bundle


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=True)


def _badge(ok: bool | None, *, true_label: str = "PASS", false_label: str = "FAIL", none_label: str = "UNCHECKED") -> str:
    if ok is True:
        return f'<span class="badge pass">{_esc(true_label)}</span>'
    if ok is False:
        return f'<span class="badge fail">{_esc(false_label)}</span>'
    return f'<span class="badge warn">{_esc(none_label)}</span>'


def _render_doc_preview(doc_dict: Mapping[str, Any]) -> str:
    entries = doc_dict.get("entries")
    if not isinstance(entries, list):
        return '<p class="muted">Invalid document entry structure.</p>'
    chunks: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("redacted"):
            chunks.append('<span class="redacted">[redacted]</span>')
        elif "spans" in entry and isinstance(entry["spans"], list):
            rendered = _render_spans_text(entry["spans"])
            # Render redacted spans cleanly as styled chips while HTML-escaping every surviving character
            pieces = rendered.split("[redacted]")
            buf: list[str] = []
            for idx, piece in enumerate(pieces):
                if idx > 0:
                    buf.append('<span class="redacted">[redacted]</span>')
                if piece:
                    buf.append(f'<span class="surviving">{_esc(piece)}</span>')
            chunks.append("".join(buf))
        else:
            chunks.append(f'<span class="surviving">{_esc(entry.get("text", ""))}</span>')
    return " ".join(chunks) if chunks else '<p class="muted">Empty document.</p>'


def generate_html_report(
    bundle: Mapping[str, Any],
    report: Mapping[str, Any] | None = None,
    *,
    require_signature_check: bool = False,
    generated_at: str | None = None,
) -> str:
    """Render an offline, air-gap-safe, XSS-escaped HTML compliance dossier for an Ember bundle."""
    verified = dict(report) if report is not None else verify_bundle(bundle, require_signature_check=require_signature_check)
    overall_ok = bool(verified.get("ok"))
    checks = verified.get("checks") or {}

    tenant_id = str(bundle.get("tenant_id", "unknown"))
    ts = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    docs = bundle.get("documents") if isinstance(bundle.get("documents"), Mapping) else {}
    receipts = bundle.get("receipts") if isinstance(bundle.get("receipts"), list) else []
    certs = bundle.get("certificates") if isinstance(bundle.get("certificates"), list) else []
    events = bundle.get("ledger_events") if isinstance(bundle.get("ledger_events"), list) else []

    doc_checks = (checks.get("documents") or {}).get("items") or {}
    rec_checks = (checks.get("receipts") or {}).get("items") or []
    cert_checks = (checks.get("certificates") or {}).get("items") or []
    ledger_check = checks.get("ledger_chain") or {}

    cert_status_by_id: dict[str, tuple[bool | None, str]] = {}
    for row in cert_checks:
        if isinstance(row, Mapping):
            cert_status_by_id[str(row.get("certificate_id", ""))] = (
                row.get("verified"),
                str(row.get("detail", "")),
            )

    cert_rows_html: list[str] = []
    for cert in certs:
        if not isinstance(cert, Mapping):
            continue
        cid = str(cert.get("certificate_id", ""))
        sig_ok, sig_detail = cert_status_by_id.get(cid, (None, "not evaluated"))
        closure = cert.get("closure") if isinstance(cert.get("closure"), Mapping) else {}
        derived_cnt = len(closure.get("derived_invalidated") or ())
        leaves_cnt = sum(len(v) for v in (closure.get("source_leaves_redacted") or {}).values() if isinstance(v, list))
        spans_cnt = sum(
            len(sp)
            for l_map in (closure.get("source_spans_redacted") or {}).values()
            if isinstance(l_map, Mapping)
            for sp in l_map.values()
            if isinstance(sp, list)
        )
        anchor = cert.get("anchor") if isinstance(cert.get("anchor"), Mapping) else {}
        tb = cert.get("trust_boundary") if isinstance(cert.get("trust_boundary"), Mapping) else {}
        cert_rows_html.append(
            "<tr>"
            f"<td><code>{_esc(cid)}</code></td>"
            f"<td>{_esc(cert.get('subject_id', ''))}</td>"
            f"<td><code>{_esc(cert.get('issued_at', ''))}</code></td>"
            f"<td>{_esc(cert.get('probe_count', 0))} (abstained={_esc(cert.get('battery_all_abstained'))})</td>"
            f"<td>key={_badge(bool(cert.get('key_destroyed')), true_label='SHREDDED', false_label='NO')}<br>"
            f"<small>shared_deks={_esc(cert.get('record_keys_destroyed', 0))}, derived={derived_cnt}, leaves={leaves_cnt}, spans={spans_cnt}</small></td>"
            f"<td>{_badge(sig_ok, true_label='ED25519 VALID', false_label='INVALID')}<br><small>{_esc(sig_detail)}</small></td>"
            f"<td><code>{_esc(anchor.get('scheme', 'none'))}</code><br><small>sep_keys={_esc(tb.get('keys_separated_from_ledger'))}, enc_vendor={_esc(tb.get('vendor_payload_encrypted'))}</small></td>"
            "</tr>"
        )
    if not cert_rows_html:
        cert_rows_html.append('<tr><td colspan="7" class="muted">No erasure certificates in bundle.</td></tr>')

    doc_panels: list[str] = []
    for sid, doc_dict in docs.items():
        d_info = doc_checks.get(str(sid)) if isinstance(doc_checks, Mapping) else {}
        d_ok = d_info.get("ok") if isinstance(d_info, Mapping) else None
        entries = doc_dict.get("entries") if isinstance(doc_dict, Mapping) and isinstance(doc_dict.get("entries"), list) else []
        red_cnt = sum(1 for e in entries if isinstance(e, Mapping) and e.get("redacted"))
        doc_panels.append(
            '<div class="card">'
            f'<div class="card-head"><strong>Document: <code>{_esc(sid)}</code></strong> {_badge(d_ok)}</div>'
            f'<div class="meta-line">Root: <code>{_esc((doc_dict or {}).get("merkle_root", ""))}</code> · Mode: <code>{_esc((doc_dict or {}).get("chunk_mode", ""))}</code> · Leaves: {len(entries)} ({red_cnt} full leaf redacted)</div>'
            f'<div class="doc-preview">{_render_doc_preview(doc_dict if isinstance(doc_dict, Mapping) else {})}</div>'
            "</div>"
        )
    if not doc_panels:
        doc_panels.append('<p class="muted">No source documents in bundle.</p>')

    rec_rows_html: list[str] = []
    for idx, rec in enumerate(receipts):
        if not isinstance(rec, Mapping):
            continue
        r_ok = rec_checks[idx].get("ok") if idx < len(rec_checks) and isinstance(rec_checks[idx], Mapping) else None
        span_part = (
            f"spans {rec.get('span_indices')}" if isinstance(rec.get("span_indices"), list) and rec.get("span_indices") else "full leaf"
        )
        rec_rows_html.append(
            "<tr>"
            f"<td><code>{_esc(rec.get('source_id', ''))}</code></td>"
            f"<td>{_esc(rec.get('leaf_index', ''))} ({_esc(span_part)})</td>"
            f"<td>{_esc(rec.get('quote', ''))}</td>"
            f"<td>{_badge(r_ok)}</td>"
            "</tr>"
        )
    if not rec_rows_html:
        rec_rows_html.append('<tr><td colspan="4" class="muted">No surviving receipts in bundle.</td></tr>')

    kind_counts = Counter(str(ev.get("kind", "UNKNOWN")) for ev in events if isinstance(ev, Mapping))
    kind_summary = ", ".join(f"{k}: {v}" for k, v in sorted(kind_counts.items())) or "none"
    tail_hash = str(events[-1].get("event_hash", "")) if events and isinstance(events[-1], Mapping) else "genesis"

    coverage = bundle.get("coverage") if isinstance(bundle.get("coverage"), Mapping) else None
    coverage_section_html = ""
    if coverage is not None:
        cov_verdict = str(coverage.get("verdict", "unknown"))
        if cov_verdict == "complete":
            cov_badge = '<span class="badge pass">COMPLETE</span>'
        elif cov_verdict == "bounded_with_warnings":
            cov_badge = '<span class="badge warn">BOUNDED WITH WARNINGS</span>'
        else:
            cov_badge = f'<span class="badge fail">{_esc(cov_verdict.upper())}</span>'
        unverified_locs = coverage.get("unverified_locations") or ()
        orphan_srcs = coverage.get("orphan_source_ids") or ()
        unattr_items = coverage.get("unattributed_item_ids") or ()
        recs = coverage.get("recommendations") or ()
        src_rate = float(coverage.get("source_coverage_rate", 1.0)) * 100.0
        lin_rate = float(coverage.get("lineage_capture_rate", 1.0)) * 100.0
        unverified_str = ", ".join(f"<code>{_esc(x)}</code>" for x in unverified_locs) or '<span class="muted">none</span>'
        orphan_str = ", ".join(f"<code>{_esc(x)}</code>" for x in orphan_srcs) or '<span class="muted">none</span>'
        unattr_str = ", ".join(f"<code>{_esc(x)}</code>" for x in unattr_items) or '<span class="muted">none</span>'
        rec_items_html = "".join(f"<li>{_esc(r)}</li>" for r in recs)
        rec_block = f"<ul>{rec_items_html}</ul>" if rec_items_html else '<div class="muted">No coverage warnings.</div>'
        coverage_section_html = (
            "\n  <h2>5. Bounded Erasure Coverage &amp; Lineage Completeness</h2>\n"
            '  <div class="card">\n'
            f"    <div>Verdict: {cov_badge} · {_esc(coverage.get('summary', ''))}</div>\n"
            f'    <div class="meta-line" style="margin-top: 6px;">'
            f"Source linkage: <strong>{src_rate:.1f}%</strong> ({_esc(coverage.get('sources_linked_count', 0))}/{_esc(coverage.get('source_count', 0))}) · "
            f"Lineage capture: <strong>{lin_rate:.1f}%</strong> ({_esc(coverage.get('attributed_item_count', 0))}/{_esc(coverage.get('live_item_count', 0))}) · "
            f"Pending erasures: <strong>{_esc(coverage.get('pending_erasure_count', 0))}</strong>"
            f"</div>\n"
            f"    <div><small>Orphan sources: {orphan_str} · Unattributed items: {unattr_str} · Unverified locations: {unverified_str}</small></div>\n"
            f'    <div style="margin-top: 8px;"><small><strong>Recommendations:</strong></small>{rec_block}</div>\n'
            "  </div>\n"
        )

    status_banner = (
        '<div class="banner pass">VERIFICATION PASSED — CRYPTOGRAPHIC PROOF INTACT</div>'
        if overall_ok
        else '<div class="banner fail">VERIFICATION FAILED — INTEGRITY OR SIGNATURE ERROR</div>'
    )

    escaped_json = _esc(json.dumps(verified, indent=2, sort_keys=True))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ember Compliance Dossier — {_esc(tenant_id)}</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #171f27;
    --border: #283442;
    --text: #e6edf3;
    --muted: #8b9bb0;
    --pass-bg: #0f3a25;
    --pass-fg: #3fb950;
    --fail-bg: #44161b;
    --fail-fg: #f85149;
    --warn-bg: #3b2e0e;
    --warn-fg: #d29922;
    --accent: #f0883e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; }}
  h1, h2 {{ margin: 0 0 12px; font-weight: 600; }}
  h2 {{ margin-top: 28px; font-size: 1.15rem; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; color: #a5d6ff; word-break: break-all; }}
  .banner {{ padding: 14px 18px; border-radius: 8px; font-weight: 700; letter-spacing: 0.03em; margin: 16px 0; }}
  .banner.pass {{ background: var(--pass-bg); color: var(--pass-fg); border: 1px solid #238636; }}
  .banner.fail {{ background: var(--fail-bg); color: var(--fail-fg); border: 1px solid #da3633; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }}
  .kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .kpi .label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi .val {{ font-size: 1.35rem; font-weight: 700; margin-top: 4px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
  .badge.pass {{ background: var(--pass-bg); color: var(--pass-fg); border: 1px solid #238636; }}
  .badge.fail {{ background: var(--fail-bg); color: var(--fail-fg); border: 1px solid #da3633; }}
  .badge.warn {{ background: var(--warn-bg); color: var(--warn-fg); border: 1px solid #9e6a03; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-top: 8px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; font-size: 0.9rem; }}
  th {{ background: #1c2630; color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-top: 10px; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
  .meta-line {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 10px; }}
  .doc-preview {{ background: #0b1015; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 0.94rem; }}
  .redacted {{ background: #3d1d22; color: #ff9492; padding: 1px 6px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 0.85em; font-weight: 600; }}
  .surviving {{ color: var(--text); }}
  .muted {{ color: var(--muted); }}
  details {{ margin-top: 24px; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  pre {{ overflow-x: auto; font-size: 0.82rem; color: #c9d1d9; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Ember Cryptographic Custody &amp; Erasure Dossier</h1>
  <div class="muted">Tenant: <code>{_esc(tenant_id)}</code> · Format: <code>{_esc(bundle.get('format', 'ember-bundle/v1'))}</code> · Rendered: <code>{_esc(ts)}</code></div>
  {status_banner}

  <div class="grid">
    <div class="kpi"><div class="label">Source Documents</div><div class="val">{len(docs)} {_badge((checks.get('documents') or {}).get('ok'))}</div></div>
    <div class="kpi"><div class="label">Provenance Receipts</div><div class="val">{len(receipts)} {_badge((checks.get('receipts') or {}).get('ok'))}</div></div>
    <div class="kpi"><div class="label">Erasure Certificates</div><div class="val">{len(certs)} {_badge((checks.get('certificates') or {}).get('ok'))}</div></div>
    <div class="kpi"><div class="label">Ledger Events</div><div class="val">{len(events)} {_badge(ledger_check.get('ok'))}</div></div>
  </div>

  <h2>1. Erasure Certificates</h2>
  <table>
    <thead><tr><th>Certificate ID</th><th>Subject</th><th>Issued At</th><th>Probes</th><th>Shred &amp; Closure</th><th>Signature</th><th>Anchor &amp; Boundary</th></tr></thead>
    <tbody>
      {"".join(cert_rows_html)}
    </tbody>
  </table>

  <h2>2. Redacted Merkle Source Documents</h2>
  {"".join(doc_panels)}

  <h2>3. Verified Provenance Receipts</h2>
  <table>
    <thead><tr><th>Source ID</th><th>Leaf / Spans</th><th>Certified Quote</th><th>Status</th></tr></thead>
    <tbody>
      {"".join(rec_rows_html)}
    </tbody>
  </table>

  <h2>4. Append-Only Ledger Chain Audit</h2>
  <div class="card">
    <div>Status: {_badge(ledger_check.get('ok'))} · Detail: {_esc(ledger_check.get('detail', 'not included in bundle'))}</div>
    <div class="meta-line" style="margin-top: 6px;">Events by kind: <code>{_esc(kind_summary)}</code> · Tail hash: <code>{_esc(tail_hash)}</code></div>
  </div>
{coverage_section_html}
  <details>
    <summary>Raw Verification Report JSON</summary>
    <pre>{escaped_json}</pre>
  </details>
</div>
</body>
</html>
"""


__all__ = ["generate_html_report"]
