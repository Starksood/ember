"""Zero-dependency Model Context Protocol (MCP) stdio JSON-RPC 2.0 server (`ember-mcp`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys
from typing import Any, TextIO

from . import __version__
from .cli import _close_vault_pair, _vault
from .crypto import DecryptionError
from .kms import KmsOperationError
from .ledger import IdempotencyConflict, LedgerIntegrityError
from .reducer import ErasureIncomplete, InvariantViolation, TombstonedSubjectError
from .vault import CustodyVault

MCP_PROTOCOL_VERSION = "2024-11-05"

_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "ember_register_source",
        "description": "Register a Merkle-indexed redactable source document for verifiable provenance receipts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "text": {"type": "string"},
                "mode": {"type": "string", "enum": ["sentence", "line", "whole"], "default": "sentence"},
                "span_mode": {"type": ["string", "null"], "enum": ["word", "clause", None], "default": None},
                "origin": {"type": "string", "default": ""},
                "origin_kind": {"type": "string", "default": "mcp"},
                "supplied_by": {"type": "string", "default": ""},
            },
            "required": ["source_id", "text"],
        },
    },
    {
        "name": "ember_remember",
        "description": "Encrypt with per-subject/record AES-256-GCM, bind provenance receipt, and commit to ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "subjects": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "source_id": {"type": ["string", "null"], "default": None},
                "quote": {"type": ["string", "null"], "default": None},
                "derived_from": {"type": "array", "items": {"type": "string"}},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
                "reconsent_id": {"type": ["string", "null"], "default": None},
            },
            "required": ["text", "subjects"],
        },
    },
    {
        "name": "ember_search",
        "description": "Search memories, filter orphans against live custody state, and verify Merkle receipts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 100},
                "filters": {"type": "object"},
                "verify_receipts": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ember_erase",
        "description": "Irreversibly crypto-shred subject keys, redact source leaves/spans, cascade delete, and sign Ed25519 cert.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "probes": {"type": "array", "items": {"type": "string"}},
                "markers": {"type": "array", "items": {"type": "string"}},
                "block_identifiers": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["subject_id"],
        },
    },
    {
        "name": "ember_verify_receipt",
        "description": "Verify the redacted Merkle provenance proof for a stored memory item.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string"},
            },
            "required": ["item_id"],
        },
    },
    {
        "name": "ember_export_bundle",
        "description": "Export self-contained redacted verification bundle JSON checkable offline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_ledger": {"type": "boolean", "default": True},
            },
        },
    },
)


class _UnknownToolError(KeyError):
    pass


class EmberMcpServer:
    """Thread-safe MCP JSON-RPC 2.0 server exposing Ember custody primitives."""

    def __init__(self, vault: CustodyVault) -> None:
        self.vault = vault

    def list_tools(self) -> list[dict[str, Any]]:
        return list(_TOOLS)

    @staticmethod
    def _req_str(args: dict[str, Any], key: str) -> str:
        val = args.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"missing or invalid required string argument {key!r}")
        return val

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
        args = dict(arguments or {})
        try:
            if name == "ember_register_source":
                source_id = self._req_str(args, "source_id")
                text = args.get("text")
                if not isinstance(text, str):
                    raise ValueError("missing or invalid required string argument 'text'")
                rec, doc = self.vault.register_source(
                    source_id,
                    text,
                    mode=args.get("mode", "sentence"),
                    span_mode=args.get("span_mode"),
                    origin=str(args.get("origin", "")),
                    origin_kind=str(args.get("origin_kind", "mcp")),
                    supplied_by=str(args.get("supplied_by", "")),
                )
                return {"source": rec.as_dict(), "merkle_root": doc.root()}, False

            if name == "ember_remember":
                text = args.get("text")
                subjects = args.get("subjects")
                if not isinstance(text, str) or not isinstance(subjects, list):
                    raise ValueError("'text' (string) and 'subjects' (array of strings) are required")
                res = self.vault.remember(
                    text,
                    subjects=subjects,
                    source_id=args.get("source_id"),
                    quote=args.get("quote"),
                    derived_from=args.get("derived_from") or (),
                    aliases=args.get("aliases") or (),
                    metadata=args.get("metadata"),
                    reconsent_id=args.get("reconsent_id"),
                )
                return (
                    {
                        "item_id": res.item.item_id,
                        "event_hash": res.event_hash,
                        "adapter_record_ids": list(res.adapter_record_ids),
                        "subjects": list(res.item.subjects),
                        "derived_from": list(res.item.derived_from),
                        "receipt": res.receipt.as_dict() if res.receipt else None,
                    },
                    False,
                )

            if name == "ember_search":
                query = self._req_str(args, "query")
                res = self.vault.search(
                    query,
                    limit=int(args.get("limit", 5)),
                    filters=args.get("filters"),
                    verify_receipts=bool(args.get("verify_receipts", True)),
                )
                return (
                    {
                        "query": res.query,
                        "abstained": res.abstained,
                        "filtered_orphans": res.filtered_orphans,
                        "hits": [
                            {
                                "item_id": h.item_id,
                                "record_id": h.record_id,
                                "text": h.text,
                                "score": h.score,
                                "subjects": list(h.subjects),
                                "receipt_verified": h.receipt_verified,
                                "receipt": h.receipt.as_dict() if h.receipt else None,
                                "metadata": h.metadata,
                            }
                            for h in res.hits
                        ],
                    },
                    False,
                )

            if name == "ember_erase":
                subject_id = self._req_str(args, "subject_id")
                cert = self.vault.erase(
                    subject_id,
                    aliases=args.get("aliases") or (),
                    probes=args.get("probes") or (),
                    markers=args.get("markers") or (),
                    block_identifiers=args.get("block_identifiers") or (),
                )
                return cert.as_dict(), False

            if name == "ember_verify_receipt":
                item_id = self._req_str(args, "item_id")
                return {"item_id": item_id, "verified": self.vault.verify_item_receipt(item_id)}, False

            if name == "ember_export_bundle":
                return self.vault.export_bundle(include_ledger=bool(args.get("include_ledger", True))), False

            raise _UnknownToolError(name)
        except (
            TombstonedSubjectError,
            ErasureIncomplete,
            IdempotencyConflict,
            InvariantViolation,
            LedgerIntegrityError,
            DecryptionError,
            KmsOperationError,
            ValueError,
            TypeError,
        ) as exc:
            err_payload: dict[str, Any] = {"error": type(exc).__name__, "detail": str(exc)}
            if isinstance(exc, ErasureIncomplete):
                err_payload["leaking_probes"] = list(exc.leaking_probes)
            return err_payload, True

    def handle_jsonrpc(self, msg: Any) -> dict[str, Any] | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id") if isinstance(msg, dict) else None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        method = msg["method"]
        req_id = msg.get("id")
        is_notification = "id" not in msg or req_id is None
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        if method == "notifications/initialized":
            return None

        if method == "initialize":
            result = {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ember-mcp", "version": __version__},
            }
            return None if is_notification else {"jsonrpc": "2.0", "id": req_id, "result": result}

        if method == "ping":
            return None if is_notification else {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return None if is_notification else {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}

        if method == "tools/call":
            tool_name = params.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                return (
                    None
                    if is_notification
                    else {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Missing tool name"}}
                )
            tool_args = params.get("arguments")
            if tool_args is not None and not isinstance(tool_args, dict):
                return (
                    None
                    if is_notification
                    else {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Tool arguments must be an object"},
                    }
                )
            try:
                payload, is_err = self.call_tool(tool_name, tool_args)
            except _UnknownToolError:
                return (
                    None
                    if is_notification
                    else {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
                )
            text = json.dumps(payload, ensure_ascii=True)
            res = {"content": [{"type": "text", "text": text}], "isError": is_err}
            return None if is_notification else {"jsonrpc": "2.0", "id": req_id, "result": res}

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def run_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        in_stream = stdin if stdin is not None else sys.stdin
        out_stream = stdout if stdout is not None else sys.stdout
        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                resp: dict[str, Any] | None = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
            else:
                resp = self.handle_jsonrpc(req)
            if resp is not None:
                out_stream.write(json.dumps(resp, separators=(",", ":"), ensure_ascii=True) + "\n")
                out_stream.flush()
        return 0


def build_mcp_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ember-mcp",
        description="Run Ember cryptographic custody tools as a Model Context Protocol (MCP) stdio server.",
    )
    parser.add_argument("--config", default=None, help="Path to ember.toml config file")
    parser.add_argument("--vault", default=".ember", help="Path to ledger/vault data directory")
    parser.add_argument("--key-dir", default=None, help="Optional separate key storage directory")
    parser.add_argument("--tenant", default="default", help="Tenant identifier")
    parser.add_argument("--adapter", choices=("sqlite", "memory"), default=None, help="Backing store adapter")
    parser.add_argument("--keystore-backend", choices=("file", "sqlite"), default=None, help="Local KeyStore format")
    parser.add_argument("--encrypt-vendor-payload", action="store_true")
    parser.add_argument("--tsa-url", default=None)
    parser.add_argument("--rekor-url", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_mcp_parser()
    args = parser.parse_args(raw_argv)
    args.vault_explicit = any(tok == "--vault" or tok.startswith("--vault=") for tok in raw_argv)
    args.tenant_explicit = any(tok == "--tenant" or tok.startswith("--tenant=") for tok in raw_argv)
    vault, store = _vault(args)
    try:
        return EmberMcpServer(vault).run_stdio()
    finally:
        _close_vault_pair(vault, store)


__all__ = [
    "EmberMcpServer",
    "MCP_PROTOCOL_VERSION",
    "build_mcp_parser",
    "main",
]
