"""Asymmetric Ed25519 certificate signing and external timestamp anchor interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol

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
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(bytes.fromhex(data["private_key_hex"]))
        signer = cls.generate()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        payload = json.dumps(
            {
                "scheme": cls.scheme,
                "private_key_hex": signer.private_bytes().hex(),
                "public_key_hex": signer.public_key_hex,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, p)
        os.chmod(p, 0o600)
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
