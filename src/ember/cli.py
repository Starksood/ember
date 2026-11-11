"""Command-line interface for Ember cryptographic custody and adversarial erasure benchmarking."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from . import __version__
from .adapters import SQLiteStoreAdapter
from .audit import run_benchmark
from .crypto import DecryptionError, ERASED_TOMBSTONE
from .ledger import IdempotencyConflict, LedgerIntegrityError
from .reducer import ErasureIncomplete, InvariantViolation, TombstonedSubjectError
from .vault import CustodyVault
from .verify import verify_bundle


def _vault(args: argparse.Namespace) -> tuple[CustodyVault, SQLiteStoreAdapter]:
    data_dir = Path(args.vault)
    key_dir = Path(args.key_dir) if args.key_dir else None
    store = SQLiteStoreAdapter(str(data_dir / "store.db"))
    try:
        vault = CustodyVault.open(
            data_dir,
            key_dir=key_dir,
            tenant_id=args.tenant,
            adapter=store,
        )
    except BaseException:
        store.close()
        raise
    return vault, store


def _parse_metadata_pairs(pairs: Sequence[str] | None) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if not pairs:
        return meta
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"invalid --meta entry {pair!r}; expected KEY=VALUE")
        meta[key.strip()] = value
    return meta


def _cmd_init(args: argparse.Namespace) -> int:
    vault, store = _vault(args)
    try:
        print(
            json.dumps(
                {
                    "status": "initialized",
                    "vault_dir": str(Path(args.vault).resolve()),
                    "tenant_id": vault.tenant_id,
                    "public_key_hex": vault.signer.public_key_hex,
                    "keys_separated_from_ledger": vault.keys_separated_from_ledger,
                },
                indent=2,
            )
        )
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_add_source(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if text is None or not text:
        raise ValueError("provide either --text or --file with non-empty content")
    vault, store = _vault(args)
    try:
        rec, _ = vault.register_source(
            args.source_id,
            text,
            mode=args.mode,
            origin=args.origin,
            origin_kind=args.origin_kind,
            supplied_by=args.supplied_by,
        )
        print(json.dumps(rec.as_dict(), indent=2))
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_remember(args: argparse.Namespace) -> int:
    metadata = _parse_metadata_pairs(args.meta)
    vault, store = _vault(args)
    try:
        res = vault.remember(
            args.text,
            subjects=args.subject,
            source_id=args.source_id,
            quote=args.quote,
            derived_from=args.derived_from or (),
            aliases=args.alias or (),
            metadata=metadata or None,
            reconsent_id=args.reconsent_id,
        )
        print(
            json.dumps(
                {
                    "item_id": res.item.item_id,
                    "event_hash": res.event_hash,
                    "adapter_record_ids": list(res.adapter_record_ids),
                    "receipt": res.receipt.as_dict() if res.receipt else None,
                },
                indent=2,
            )
        )
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_search(args: argparse.Namespace) -> int:
    vault, store = _vault(args)
    try:
        res = vault.search(args.query, limit=args.limit)
        print(
            json.dumps(
                {
                    "query": res.query,
                    "abstained": res.abstained,
                    "filtered_orphans": res.filtered_orphans,
                    "hits": [
                        {
                            "item_id": h.item_id,
                            "score": h.score,
                            "text": h.text,
                            "subjects": list(h.subjects),
                            "receipt_verified": h.receipt_verified,
                        }
                        for h in res.hits
                    ],
                },
                indent=2,
            )
        )
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_erase(args: argparse.Namespace) -> int:
    vault, store = _vault(args)
    try:
        cert = vault.erase(
            args.subject,
            aliases=args.alias or (),
            probes=args.probe or (),
            markers=args.marker or (),
        )
        print(json.dumps(cert.as_dict(), indent=2))
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_replay(args: argparse.Namespace) -> int:
    vault, store = _vault(args)
    try:
        chain_ok, chain_msg = vault.ledger.verify_chain(vault.tenant_id)
        replayed = vault.replay()
        events = vault.ledger.events(vault.tenant_id)
        certs = vault.ledger.list_certificates(vault.tenant_id)
        erased_tombstones = sum(1 for item in replayed.items.values() if item.text == ERASED_TOMBSTONE)
        print(
            json.dumps(
                {
                    "tenant_id": vault.tenant_id,
                    "chain_ok": chain_ok,
                    "chain_detail": chain_msg,
                    "event_count": len(events),
                    "source_count": len(replayed.sources),
                    "live_item_count": len(replayed.items),
                    "erased_tombstone_tokens": len(replayed.tombstoned_tokens),
                    "shredded_items_remaining": erased_tombstones,
                    "certificate_count": len(certs),
                    "state_digest": replayed.digest(),
                },
                indent=2,
            )
        )
        return 0 if chain_ok else 1
    finally:
        vault.close()
        store.close()


def _cmd_export(args: argparse.Namespace) -> int:
    vault, store = _vault(args)
    try:
        bundle = vault.export_bundle()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "exported": str(out),
                    "documents": len(bundle["documents"]),
                    "receipts": len(bundle["receipts"]),
                    "certificates": len(bundle["certificates"]),
                },
                indent=2,
            )
        )
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_verify(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    report = verify_bundle(data, require_signature_check=args.require_signature)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def format_benchmark_report(report: dict[str, Any]) -> str:
    """Format an adversarial erasure benchmark report with explicit adapter & search settings."""
    adapter = report.get("adapter") or {}
    adapter_name = adapter.get("name", "unknown")
    retrieval_mode = adapter.get("retrieval_mode", "unknown")
    embedding_model = adapter.get("embedding_model") or "none"
    vector_store = adapter.get("vector_store") or "in-process"
    search_settings = adapter.get("search_settings") or {}
    settings_str = ", ".join(f"{k}={v}" for k, v in sorted(search_settings.items())) or "default"

    lines = [
        f"\nEMBER ERASURE ADVERSARIAL SUITE — {report['subjects']} subjects, "
        f"{report['total_probes']} probes (7 attack classes)",
        f"  Adapter   : {adapter_name} (retrieval={retrieval_mode}, store={vector_store})",
        f"  Embedding : {embedding_model}",
        f"  Search    : {settings_str}",
        "=" * 78,
    ]
    for row in report["results"]:
        lines.append(
            f"  {row['subject']:10} baseline_leaks={row['baseline_naive_leaks']:2d}  "
            f"ember_leaks={row['ember_leaks']:2d}  derived={row['derived_invalidated']}  "
            f"bystander_ok={row['bystander_retrievable']}  receipt_ok={row['bystander_receipts_valid']}  "
            f"replay_clean={row['raw_history_content_free']}"
        )
    lines.append("=" * 78)
    lines.extend(
        [
            f"  Adapter              : {adapter_name} ({retrieval_mode}, embed={embedding_model})",
            f"  Baseline naive leaks : {report['baseline_naive_leaks']}/{report['total_probes']}",
            f"  Ember leaks          : {report['ember_leaks']}/{report['total_probes']}",
            f"  Bystanders survived  : {report['all_bystanders_survived']}",
            f"  Receipts intact      : {report['all_bystander_receipts_valid']}",
            f"  Replays content-free : {report['all_replays_content_free']}",
            f"  Ed25519 certs valid  : {report['all_certificates_valid']}",
        ]
    )
    return "\n".join(lines)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_benchmark()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_benchmark_report(report))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ember",
        description="Cryptographic custody, redactable Merkle receipts, and verifiable cascade erasure.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--vault", default=".ember", help="Path to ledger/vault data directory")
    parser.add_argument("--key-dir", default=None, help="Optional separate key storage directory")
    parser.add_argument("--tenant", default="default", help="Tenant identifier")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize an Ember vault")
    p_init.set_defaults(func=_cmd_init)

    p_src = sub.add_parser("add-source", help="Register a Merkle-indexed redactable source document")
    p_src.add_argument("source_id")
    p_src.add_argument("--text", default=None)
    p_src.add_argument("--file", default=None)
    p_src.add_argument("--mode", choices=("sentence", "line", "whole"), default="sentence")
    p_src.add_argument("--origin", default="")
    p_src.add_argument("--origin-kind", default="unknown")
    p_src.add_argument("--supplied-by", default="")
    p_src.set_defaults(func=_cmd_add_source)

    p_rem = sub.add_parser("remember", help="Encrypt and store a custody-backed memory item")
    p_rem.add_argument("text")
    p_rem.add_argument("--subject", action="append", required=True, help="Subject ID (repeatable)")
    p_rem.add_argument("--alias", action="append", default=None, help="Subject alias (repeatable)")
    p_rem.add_argument("--source-id", default=None)
    p_rem.add_argument("--quote", default=None)
    p_rem.add_argument("--derived-from", action="append", default=None)
    p_rem.add_argument("--meta", action="append", default=None, help="Encrypted metadata KEY=VALUE (repeatable)")
    p_rem.add_argument("--reconsent-id", default=None)
    p_rem.set_defaults(func=_cmd_remember)

    p_search = sub.add_parser("search", help="Query vault store and verify provenance receipts")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.set_defaults(func=_cmd_search)

    p_erase = sub.add_parser("erase", help="Cascade-erase subject, redact sources, shred keys, and sign cert")
    p_erase.add_argument("subject")
    p_erase.add_argument("--alias", action="append", default=None)
    p_erase.add_argument("--probe", action="append", default=None)
    p_erase.add_argument("--marker", action="append", default=None)
    p_erase.set_defaults(func=_cmd_erase)

    p_rep = sub.add_parser("replay", help="Verify hash chain and replay ledger state from zero")
    p_rep.set_defaults(func=_cmd_replay)

    p_exp = sub.add_parser("export", help="Export offline verification bundle JSON")
    p_exp.add_argument("--out", default="ember-bundle.json")
    p_exp.set_defaults(func=_cmd_export)

    p_ver = sub.add_parser("verify", help="Verify an exported verification bundle offline")
    p_ver.add_argument("bundle")
    p_ver.add_argument("--require-signature", action="store_true")
    p_ver.set_defaults(func=_cmd_verify)

    p_bench = sub.add_parser("benchmark", help="Run the 12-subject x 7-class adversarial erasure suite")
    p_bench.add_argument("--json", action="store_true")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (
        TombstonedSubjectError,
        ErasureIncomplete,
        LedgerIntegrityError,
        IdempotencyConflict,
        InvariantViolation,
        DecryptionError,
        FileNotFoundError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
        OSError,
    ) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
