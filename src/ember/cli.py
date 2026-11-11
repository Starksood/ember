"""Command-line interface for Ember cryptographic custody and adversarial erasure benchmarking."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from .adapters import SQLiteStoreAdapter
from .audit import run_benchmark
from .vault import CustodyVault
from .verify import verify_bundle


def _vault(args: argparse.Namespace) -> tuple[CustodyVault, SQLiteStoreAdapter]:
    data_dir = Path(args.vault)
    key_dir = Path(args.key_dir) if args.key_dir else None
    store = SQLiteStoreAdapter(str(data_dir / "store.db"))
    vault = CustodyVault.open(
        data_dir,
        key_dir=key_dir,
        tenant_id=args.tenant,
        adapter=store,
    )
    return vault, store


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
    if not text:
        raise SystemExit("provide either --text or --file")
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
    vault, store = _vault(args)
    try:
        res = vault.remember(
            args.text,
            subjects=args.subject,
            source_id=args.source_id,
            quote=args.quote,
            derived_from=args.derived_from or (),
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


def _cmd_export(args: argparse.Namespace) -> int:
    vault, store = _vault(args)
    try:
        bundle = vault.export_bundle()
        out = Path(args.out)
        out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(json.dumps({"exported": str(out), "documents": len(bundle["documents"]), "receipts": len(bundle["receipts"]), "certificates": len(bundle["certificates"])}, indent=2))
        return 0
    finally:
        vault.close()
        store.close()


def _cmd_verify(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    report = verify_bundle(data, require_signature_check=args.require_signature)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    report = run_benchmark()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"\nEMBER ERASURE ADVERSARIAL SUITE — {report['subjects']} subjects, "
            f"{report['total_probes']} probes (7 attack classes)\n"
            + "=" * 78
        )
        for row in report["results"]:
            print(
                f"  {row['subject']:10} baseline_leaks={row['baseline_naive_leaks']:2d}  "
                f"ember_leaks={row['ember_leaks']:2d}  derived={row['derived_invalidated']}  "
                f"bystander_ok={row['bystander_retrievable']}  receipt_ok={row['bystander_receipts_valid']}  "
                f"replay_clean={row['raw_history_content_free']}"
            )
        print("=" * 78)
        print(
            f"  Baseline naive leaks : {report['baseline_naive_leaks']}/{report['total_probes']}\n"
            f"  Ember leaks          : {report['ember_leaks']}/{report['total_probes']}\n"
            f"  Bystanders survived  : {report['all_bystanders_survived']}\n"
            f"  Receipts intact      : {report['all_bystander_receipts_valid']}\n"
            f"  Replays content-free : {report['all_replays_content_free']}\n"
            f"  Ed25519 certs valid  : {report['all_certificates_valid']}"
        )
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ember",
        description="Cryptographic custody, redactable Merkle receipts, and verifiable cascade erasure.",
    )
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
    p_rem.add_argument("--source-id", default=None)
    p_rem.add_argument("--quote", default=None)
    p_rem.add_argument("--derived-from", action="append", default=None)
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
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
