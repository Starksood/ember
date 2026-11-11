"""Asymmetric Ed25519 certificate signing and external timestamp anchor interfaces."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Any, Protocol
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .verify import certificate_signable_bytes, verify_ed25519_certificate


@dataclass(frozen=True)
class AnchorReceipt:
    """External timestamp / transparency-log attestation over a certificate digest."""

    scheme: str
    digest: str
    token: str

    def as_dict(self) -> dict[str, str]:
        return {"scheme": self.scheme, "digest": self.digest, "token": self.token}


class TimestampAnchor(Protocol):
    """Protocol for binding certificate digests to an external time/order authority."""

    scheme: str

    def anchor(self, digest_hex: str) -> AnchorReceipt: ...


class NullAnchor:
    """Default local-only anchor. Explicitly discloses absence of third-party timing proof."""

    scheme = "none"

    def anchor(self, digest_hex: str) -> AnchorReceipt:
        return AnchorReceipt(scheme="none", digest=digest_hex, token="")


class CallbackAnchor:
    """Adapter wrapping a user-provided TSA / transparency-log submission callback."""

    def __init__(self, scheme: str, submit_fn: Callable[[str], str]) -> None:
        self.scheme = scheme
        self._submit_fn = submit_fn

    def anchor(self, digest_hex: str) -> AnchorReceipt:
        token = self._submit_fn(digest_hex)
        return AnchorReceipt(scheme=self.scheme, digest=digest_hex, token=token)


_SIGNER_FILE_LOCK = threading.RLock()


class Ed25519Signer:
    """Adversary-checkable Ed25519 signer (private key signs, hex public key verifies)."""

    scheme = "ed25519"
    adversary_checkable = True

    def __init__(self, private_bytes: bytes) -> None:
        self._sk = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)

    @classmethod
    def generate(cls) -> "Ed25519Signer":
        sk = ed25519.Ed25519PrivateKey.generate()
        raw = sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return cls(raw)

    @classmethod
    def load_or_create(cls, path: str | Path) -> "Ed25519Signer":
        p = Path(path)
        with _SIGNER_FILE_LOCK:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return cls(bytes.fromhex(data["private_key_hex"]))
            signer = cls.generate()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(6)}.tmp")
            payload = json.dumps(
                {
                    "scheme": cls.scheme,
                    "private_key_hex": signer.private_bytes().hex(),
                    "public_key_hex": signer.public_key_hex,
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, payload)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp, p)
                os.chmod(p, 0o600)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return signer

    def private_bytes(self) -> bytes:
        return self._sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_key_hex(self) -> str:
        return self._sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

    def sign(self, payload: bytes) -> str:
        return f"{self.scheme}:" + self._sk.sign(payload).hex()

    def verify_certificate(self, cert_dict: dict[str, Any]) -> bool:
        ok, _ = verify_ed25519_certificate(cert_dict, public_key_hex=self.public_key_hex)
        return ok is True


def sign_certificate_dict(cert_dict: dict[str, Any], signer: Ed25519Signer) -> dict[str, Any]:
    """Return copy of `cert_dict` signed with `signer`."""
    out = dict(cert_dict)
    out["signer_scheme"] = signer.scheme
    out["public_key"] = signer.public_key_hex
    out["signature"] = signer.sign(certificate_signable_bytes(out))
    return out


# ── DER helpers for zero-dep RFC 3161 TimeStampReq / TimeStampResp ──────────

_SHA256_OID_DER = bytes.fromhex("0609608648016503040201")


def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    encoded = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(encoded)]) + encoded


def _der_tlv(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(value)) + value


def _der_read_tlv(data: bytes, offset: int = 0) -> tuple[int, bytes, int]:
    if offset >= len(data):
        raise ValueError("unexpected end of DER input")
    tag = data[offset]
    offset += 1
    if offset >= len(data):
        raise ValueError("truncated DER length byte")
    first_len = data[offset]
    offset += 1
    if first_len < 0x80:
        length = first_len
    else:
        num_bytes = first_len & 0x7F
        if num_bytes == 0 or offset + num_bytes > len(data):
            raise ValueError("invalid or unsupported DER length")
        length = int.from_bytes(data[offset : offset + num_bytes], "big")
        offset += num_bytes
    if offset + length > len(data):
        raise ValueError("truncated DER value")
    return tag, data[offset : offset + length], offset + length


def _digest_hex32(digest_hex: str) -> bytes:
    raw_hex = digest_hex[7:] if digest_hex.startswith("sha256:") else digest_hex
    digest_bytes = bytes.fromhex(raw_hex.strip())
    if len(digest_bytes) != 32:
        raise ValueError(f"expected 32-byte SHA-256 digest, got {len(digest_bytes)} bytes")
    return digest_bytes


def build_rfc3161_request(digest_bytes: bytes, *, cert_req: bool = True, nonce: bytes | None = None) -> bytes:
    """Build a DER-encoded RFC 3161 `TimeStampReq` for a 32-byte SHA-256 digest."""
    if len(digest_bytes) != 32:
        raise ValueError("digest_bytes must be 32 bytes")
    alg_id = _der_tlv(0x30, _SHA256_OID_DER + b"\x05\x00")
    msg_imprint = _der_tlv(0x30, alg_id + _der_tlv(0x04, digest_bytes))
    version = b"\x02\x01\x01"
    raw_nonce = nonce if nonce is not None else secrets.token_bytes(8)
    if not raw_nonce or (raw_nonce[0] & 0x80):
        raw_nonce = b"\x00" + raw_nonce
    nonce_der = _der_tlv(0x02, raw_nonce)
    cert_der = b"\x01\x01\xff" if cert_req else b""
    return _der_tlv(0x30, version + msg_imprint + nonce_der + cert_der)


def parse_rfc3161_response_status(resp_bytes: bytes) -> tuple[int, bool]:
    """Validate minimal RFC 3161 `TimeStampResp` structure; return `(pki_status, has_token)`."""
    tag, seq_body, _ = _der_read_tlv(resp_bytes, 0)
    if tag != 0x30:
        raise ValueError("TimeStampResp is not a DER SEQUENCE")
    s_tag, status_info, pos = _der_read_tlv(seq_body, 0)
    if s_tag != 0x30:
        raise ValueError("PKIStatusInfo is not a DER SEQUENCE")
    i_tag, int_val, _ = _der_read_tlv(status_info, 0)
    if i_tag != 0x02 or not int_val:
        raise ValueError("PKIStatus is missing INTEGER")
    status = int.from_bytes(int_val, "big", signed=True)
    has_token = pos < len(seq_body)
    return status, has_token


def _http_post_raw(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"disallowed URL scheme or missing host: {url!r}")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.netloc, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers=dict(headers))
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, raw
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"HTTP POST failed for {url}: {exc}") from exc
    finally:
        conn.close()


class Rfc3161Anchor:
    """RFC 3161 Time-Stamp Authority anchor producing verifiable TSR tokens over certificate digests."""

    scheme = "rfc3161"

    def __init__(
        self,
        tsa_url: str,
        *,
        cert_req: bool = True,
        timeout: float = 10.0,
        transport: Callable[[str, bytes, dict[str, str]], bytes] | None = None,
    ) -> None:
        parsed = urlparse(tsa_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("tsa_url must be a valid http:// or https:// URL")
        self.tsa_url = tsa_url
        self.cert_req = cert_req
        self.timeout = float(timeout)
        self._transport = transport

    def anchor(self, digest_hex: str) -> AnchorReceipt:
        digest_bytes = _digest_hex32(digest_hex)
        req_der = build_rfc3161_request(digest_bytes, cert_req=self.cert_req)
        headers = {
            "Content-Type": "application/timestamp-query",
            "Accept": "application/timestamp-reply",
        }
        if self._transport is not None:
            resp_der = self._transport(self.tsa_url, req_der, headers)
        else:
            status_code, resp_der = _http_post_raw(self.tsa_url, req_der, headers, self.timeout)
            if status_code >= 400:
                raise RuntimeError(f"RFC 3161 TSA HTTP {status_code} from {self.tsa_url}")

        pki_status, has_token = parse_rfc3161_response_status(resp_der)
        if pki_status not in (0, 1) or not has_token:
            raise RuntimeError(f"RFC 3161 TSA rejected timestamp request (PKIStatus={pki_status})")
        token = f"rfc3161:tsr:{base64.b64encode(resp_der).decode('ascii')}"
        return AnchorReceipt(scheme=self.scheme, digest=digest_hex, token=token)


class RekorAnchor:
    """Sigstore Rekor transparency-log anchor submitting signed `hashedrekord` inclusion entries."""

    scheme = "rekor"

    def __init__(
        self,
        rekor_url: str = "https://rekor.sigstore.dev",
        *,
        signer: Ed25519Signer | None = None,
        timeout: float = 10.0,
        transport: Callable[[str, bytes, dict[str, str]], tuple[int, bytes]] | None = None,
    ) -> None:
        parsed = urlparse(rekor_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("rekor_url must be a valid http:// or https:// URL")
        self.rekor_url = rekor_url.rstrip("/")
        self.signer = signer if signer is not None else Ed25519Signer.generate()
        self.timeout = float(timeout)
        self._transport = transport

    def anchor(self, digest_hex: str) -> AnchorReceipt:
        digest_bytes = _digest_hex32(digest_hex)
        hex32 = digest_bytes.hex()
        pub_pem = self.signer._sk.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        sig_bytes = self.signer._sk.sign(digest_bytes)
        entry_payload = {
            "apiVersion": "0.0.1",
            "kind": "hashedrekord",
            "spec": {
                "data": {"hash": {"algorithm": "sha256", "value": hex32}},
                "signature": {
                    "content": base64.b64encode(sig_bytes).decode("ascii"),
                    "publicKey": {"content": base64.b64encode(pub_pem).decode("ascii")},
                },
            },
        }
        url = f"{self.rekor_url}/api/v1/log/entries"
        body = json.dumps(entry_payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._transport is not None:
            status_code, raw_resp = self._transport(url, body, headers)
        else:
            status_code, raw_resp = _http_post_raw(url, body, headers, self.timeout)

        if status_code >= 400:
            raise RuntimeError(f"Rekor HTTP {status_code} at {url}: {raw_resp.decode('utf-8', 'replace')}")
        parsed = json.loads(raw_resp.decode("utf-8"))
        if not isinstance(parsed, dict) or not parsed:
            raise RuntimeError("unexpected empty Rekor response")
        entry_uuid, meta = next(iter(parsed.items()))
        if not isinstance(meta, Mapping) or "logIndex" not in meta:
            raise RuntimeError(f"Rekor response missing logIndex for entry {entry_uuid!r}")
        token_info = {
            "uuid": str(entry_uuid),
            "logIndex": int(meta["logIndex"]),
            "integratedTime": meta.get("integratedTime"),
            "logID": meta.get("logID", ""),
            "url": f"{url}/{entry_uuid}",
        }
        token = "rekor:" + json.dumps(token_info, sort_keys=True, separators=(",", ":"))
        return AnchorReceipt(scheme=self.scheme, digest=digest_hex, token=token)


__all__ = [
    "AnchorReceipt",
    "CallbackAnchor",
    "Ed25519Signer",
    "NullAnchor",
    "RekorAnchor",
    "Rfc3161Anchor",
    "TimestampAnchor",
    "build_rfc3161_request",
    "parse_rfc3161_response_status",
    "sign_certificate_dict",
]
