"""Zero-dependency HTTP custody sidecar server (`ember serve`) + optional Starlette ASGI adapter."""

from __future__ import annotations

from collections.abc import Mapping
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .crypto import DecryptionError
from .kms import KmsOperationError
from .ledger import IdempotencyConflict, LedgerIntegrityError
from .reducer import ConcurrentModificationError, ErasureIncomplete, InvariantViolation, TombstonedSubjectError
from .vault import CustodyVault

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _hdr_get(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for k, v in headers.items():
        if str(k).lower() == target:
            return str(v)
    return None


class EmberHttpApp:
    """Protocol-clean request dispatcher backing both stdlib HTTPServer and Starlette ASGI."""

    def __init__(
        self,
        vault: CustodyVault,
        *,
        bearer_token: str | None = None,
        require_tenant_header: bool = False,
        max_body_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.vault = vault
        self.bearer_token = bearer_token if bearer_token else None
        self.require_tenant_header = require_tenant_header
        self.max_body_bytes = int(max_body_bytes)

    def _authenticate(self, headers: Mapping[str, str]) -> tuple[int, dict[str, Any]] | None:
        if not self.bearer_token:
            return None
        auth = _hdr_get(headers, "authorization") or ""
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, self.bearer_token):
            return 401, {"error": "Unauthorized", "detail": "missing or invalid Bearer token"}
        return None

    def _check_tenant(self, headers: Mapping[str, str]) -> tuple[int, dict[str, Any]] | None:
        tenant = _hdr_get(headers, "x-ember-tenant")
        if tenant is None:
            if self.require_tenant_header:
                return 400, {"error": "BadRequest", "detail": "missing required X-Ember-Tenant header"}
            return None
        if tenant != self.vault.tenant_id:
            return 403, {
                "error": "Forbidden",
                "detail": f"tenant mismatch: header {tenant!r} != vault tenant {self.vault.tenant_id!r}",
            }
        return None

    @staticmethod
    def _parse_json_body(body: bytes) -> dict[str, Any]:
        if not body:
            return {}
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def handle_request(
        self,
        method: str,
        raw_path: str,
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        hdrs = headers or {}
        if len(body) > self.max_body_bytes:
            return (
                413,
                {"Content-Type": "application/json"},
                _json_bytes({"error": "PayloadTooLarge", "detail": f"body exceeds {self.max_body_bytes} bytes"}),
            )

        split = urlsplit(raw_path)
        path = split.path.rstrip("/") or "/"
        query = parse_qs(split.query)

        if method == "GET" and path == "/health":
            return (
                200,
                {"Content-Type": "application/json"},
                _json_bytes(
                    {
                        "status": "ok",
                        "version": __version__,
                        "tenant_id": self.vault.tenant_id,
                        "keys_separated_from_ledger": self.vault.keys_separated_from_ledger,
                        "encrypt_vendor_payload": self.vault.encrypt_vendor_payload,
                    }
                ),
            )

        auth_err = self._authenticate(hdrs)
        if auth_err is not None:
            return (
                auth_err[0],
                {"Content-Type": "application/json", "WWW-Authenticate": "Bearer"},
                _json_bytes(auth_err[1]),
            )

        tenant_err = self._check_tenant(hdrs)
        if tenant_err is not None:
            return tenant_err[0], {"Content-Type": "application/json"}, _json_bytes(tenant_err[1])

        try:
            if method == "POST" and path == "/v1/sources":
                data = self._parse_json_body(body)
                if not isinstance(data.get("source_id"), str) or not isinstance(data.get("text"), str):
                    raise ValueError("'source_id' and 'text' strings are required")
                rec, doc = self.vault.register_source(
                    data["source_id"],
                    data["text"],
                    mode=data.get("mode", "sentence"),
                    span_mode=data.get("span_mode"),
                    origin=str(data.get("origin", "")),
                    origin_kind=str(data.get("origin_kind", "unknown")),
                    supplied_by=str(data.get("supplied_by", "")),
                    validated_by=str(data.get("validated_by", "")),
                    event_id=data.get("event_id"),
                )
                return (
                    200,
                    {"Content-Type": "application/json"},
                    _json_bytes({"source": rec.as_dict(), "merkle_root": doc.root()}),
                )

            if method == "POST" and path == "/v1/remember":
                data = self._parse_json_body(body)
                if not isinstance(data.get("text"), str) or not isinstance(data.get("subjects"), list):
                    raise ValueError("'text' (string) and 'subjects' (array of strings) are required")
                res = self.vault.remember(
                    data["text"],
                    subjects=data["subjects"],
                    source_id=data.get("source_id"),
                    quote=data.get("quote"),
                    leaf_index=data.get("leaf_index"),
                    derived_from=data.get("derived_from") or (),
                    aliases=data.get("aliases") or (),
                    metadata=data.get("metadata"),
                    reconsent_id=data.get("reconsent_id"),
                    item_id=data.get("item_id"),
                    event_id=data.get("event_id"),
                    auto_derive=bool(data.get("auto_derive", True)),
                    content_guard=bool(data.get("content_guard", True)),
                )
                return (
                    200,
                    {"Content-Type": "application/json"},
                    _json_bytes(
                        {
                            "item_id": res.item.item_id,
                            "event_hash": res.event_hash,
                            "adapter_record_ids": list(res.adapter_record_ids),
                            "subjects": list(res.item.subjects),
                            "derived_from": list(res.item.derived_from),
                            "receipt": res.receipt.as_dict() if res.receipt else None,
                        }
                    ),
                )

            if method == "POST" and path == "/v1/search":
                data = self._parse_json_body(body)
                if "query" not in data:
                    raise ValueError("'query' is required")
                res = self.vault.search(
                    str(data["query"]),
                    limit=int(data.get("limit", 5)),
                    filters=data.get("filters"),
                    verify_receipts=bool(data.get("verify_receipts", True)),
                )
                return (
                    200,
                    {"Content-Type": "application/json"},
                    _json_bytes(
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
                        }
                    ),
                )

            if method == "POST" and path == "/v1/erase":
                data = self._parse_json_body(body)
                if "subject_id" not in data:
                    raise ValueError("'subject_id' is required")
                cert = self.vault.erase(
                    str(data["subject_id"]),
                    aliases=data.get("aliases") or (),
                    probes=data.get("probes") or (),
                    markers=data.get("markers") or (),
                    block_identifiers=data.get("block_identifiers") or (),
                    auto_probes=bool(data.get("auto_probes", True)),
                )
                return 200, {"Content-Type": "application/json"}, _json_bytes(cert.as_dict())

            if method == "POST" and path == "/v1/checkpoint":
                ckpt = self.vault.checkpoint()
                return (
                    200,
                    {"Content-Type": "application/json"},
                    _json_bytes(
                        {
                            "status": "checkpointed",
                            "tenant_id": ckpt["tenant_id"],
                            "seq": ckpt["seq"],
                            "last_event_hash": ckpt["last_event_hash"],
                            "state_version": ckpt["state_version"],
                            "created_at": ckpt["created_at"],
                            "signature": ckpt.get("signature"),
                        }
                    ),
                )

            if method == "GET" and path == "/v1/bundle":
                inc_raw = (query.get("include_ledger") or ["true"])[0].lower()
                include_ledger = inc_raw not in ("0", "false", "no")
                bundle = self.vault.export_bundle(include_ledger=include_ledger)
                return 200, {"Content-Type": "application/json"}, _json_bytes(bundle)

            return (
                404,
                {"Content-Type": "application/json"},
                _json_bytes({"error": "NotFound", "detail": f"no route for {method} {path}"}),
            )
        except TombstonedSubjectError as exc:
            return 403, {"Content-Type": "application/json"}, _json_bytes({"error": type(exc).__name__, "detail": str(exc)})
        except (IdempotencyConflict, ConcurrentModificationError) as exc:
            return 409, {"Content-Type": "application/json"}, _json_bytes({"error": type(exc).__name__, "detail": str(exc)})
        except ErasureIncomplete as exc:
            return (
                422,
                {"Content-Type": "application/json"},
                _json_bytes(
                    {
                        "error": type(exc).__name__,
                        "detail": str(exc),
                        "leaking_probes": list(exc.leaking_probes),
                    }
                ),
            )
        except (InvariantViolation, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return 400, {"Content-Type": "application/json"}, _json_bytes({"error": type(exc).__name__, "detail": str(exc)})
        except (LedgerIntegrityError, DecryptionError, KmsOperationError, NotImplementedError) as exc:
            return 500, {"Content-Type": "application/json"}, _json_bytes({"error": type(exc).__name__, "detail": str(exc)})


def make_http_server(
    vault: CustodyVault,
    *,
    host: str = "127.0.0.1",
    port: int = 8420,
    bearer_token: str | None = None,
    require_tenant_header: bool = False,
    allow_insecure_no_auth: bool = False,
) -> ThreadingHTTPServer:
    """Create a configured stdlib `ThreadingHTTPServer` serving `EmberHttpApp`."""
    if host not in _LOOPBACK_HOSTS and not bearer_token and not allow_insecure_no_auth:
        raise ValueError(
            f"refusing to bind Ember HTTP sidecar on non-loopback host {host!r} without bearer_token "
            "(set --bearer-token or allow_insecure_no_auth=True)"
        )
    app = EmberHttpApp(
        vault,
        bearer_token=bearer_token,
        require_tenant_header=require_tenant_header,
    )

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _write_json(self, status: int, obj: Any, *, close: bool = False) -> None:
            payload = _json_bytes(obj)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(payload)

        def _dispatch(self, method: str) -> None:
            transfer_enc = str(self.headers.get("Transfer-Encoding") or "").lower()
            content_len = self.headers.get("Content-Length")
            if "chunked" in transfer_enc and content_len is None:
                self._write_json(
                    411,
                    {"error": "LengthRequired", "detail": "Transfer-Encoding: chunked is not supported; send Content-Length"},
                    close=True,
                )
                return
            if content_len is not None:
                try:
                    n_bytes = int(content_len)
                except ValueError:
                    self._write_json(400, {"error": "BadRequest", "detail": "invalid Content-Length header"}, close=True)
                    return
                if n_bytes < 0 or n_bytes > app.max_body_bytes:
                    self._write_json(
                        413,
                        {"error": "PayloadTooLarge", "detail": "invalid or oversized Content-Length"},
                        close=True,
                    )
                    return
                body = self.rfile.read(n_bytes) if n_bytes > 0 else b""
                if len(body) != n_bytes:
                    self._write_json(400, {"error": "BadRequest", "detail": "incomplete request body"}, close=True)
                    return
            else:
                body = b""
            hdr_dict = {k: v for k, v in self.headers.items()}
            status, resp_hdrs, payload = app.handle_request(method, self.path, body, hdr_dict)
            self.send_response(status)
            for k, v in resp_hdrs.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # Keep stdout/stderr clean for sidecar JSON/CLI usage

    return ThreadingHTTPServer((host, int(port)), _Handler)


def start_background_server(
    vault: CustodyVault,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    bearer_token: str | None = None,
    require_tenant_header: bool = False,
    allow_insecure_no_auth: bool = False,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start `ThreadingHTTPServer` in a daemon thread cleanly (returns `(httpd, thread)`)."""
    httpd = make_http_server(
        vault,
        host=host,
        port=port,
        bearer_token=bearer_token,
        require_tenant_header=require_tenant_header,
        allow_insecure_no_auth=allow_insecure_no_auth,
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def create_asgi_app(
    vault: CustodyVault,
    *,
    bearer_token: str | None = None,
    require_tenant_header: bool = False,
) -> Any:
    """Create a Starlette ASGI application wrapping `EmberHttpApp` (requires `starlette`)."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Route
    except ImportError as exc:
        raise ImportError(
            "ASGI mode requires 'starlette' and 'uvicorn'. Install with: pip install 'ember-custody[server]'"
        ) from exc

    core = EmberHttpApp(
        vault,
        bearer_token=bearer_token,
        require_tenant_header=require_tenant_header,
    )

    async def _bridge(request: Request) -> Response:
        body = await request.body()
        raw_path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        status, hdrs, payload = core.handle_request(request.method, raw_path, body, request.headers)
        return Response(content=payload, status_code=status, headers=hdrs, media_type="application/json")

    routes = [
        Route("/health", _bridge, methods=["GET"]),
        Route("/v1/sources", _bridge, methods=["POST"]),
        Route("/v1/remember", _bridge, methods=["POST"]),
        Route("/v1/search", _bridge, methods=["POST"]),
        Route("/v1/erase", _bridge, methods=["POST"]),
        Route("/v1/checkpoint", _bridge, methods=["POST"]),
        Route("/v1/bundle", _bridge, methods=["GET"]),
    ]
    return Starlette(routes=routes)


__all__ = [
    "EmberHttpApp",
    "create_asgi_app",
    "make_http_server",
    "start_background_server",
]
