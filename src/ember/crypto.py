"""Authenticated per-subject/per-record encryption for crypto-shredding.

No silent weak cipher fallback: AES-256-GCM via `cryptography` is required.
Every ciphertext binds structured Associated Authenticated Data (AAD) so an
encrypted blob cannot be moved across tenants, record IDs, or fields.
"""

from __future__ import annotations

import base64
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .verify import canonical_bytes

CIPHER_NAME = "AES-256-GCM"
PREFIX = "enc2:"
ERASED_TOMBSTONE = "[erased]"


class DecryptionError(ValueError):
    """Raised when strict decryption fails due to tampering, wrong key, or AAD mismatch."""


def new_key() -> bytes:
    """Return a fresh 256-bit key suitable for AES-256-GCM."""
    return secrets.token_bytes(32)


def build_aad(*, tenant_id: str, record_id: str, field: str, key_ref: str) -> bytes:
    """Build canonical AAD binding a ciphertext to its exact structural slot."""
    return canonical_bytes(
        {
            "tenant_id": tenant_id,
            "record_id": record_id,
            "field": field,
            "key_ref": key_ref,
        }
    )


def encrypt_text(key: bytes, plaintext: str, *, aad: bytes = b"") -> str:
    """Encrypt UTF-8 plaintext with AES-256-GCM under a 96-bit random nonce."""
    if len(key) != 32:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), aad)
    return PREFIX + base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_text(
    key: bytes | None,
    blob: str,
    *,
    aad: bytes = b"",
    strict: bool = False,
    tombstone: str = ERASED_TOMBSTONE,
) -> str:
    """Decrypt a ciphertext blob, returning `tombstone` if key is shredded/missing.

    When `strict=True`, invalid tags or missing keys raise `DecryptionError` instead of
    tombstoning.
    """
    if not blob.startswith(PREFIX):
        if strict:
            raise DecryptionError("unexpected non-enc2 payload")
        return blob
    if key is None:
        if strict:
            raise DecryptionError("key absent or shredded")
        return tombstone
    try:
        raw = base64.b64decode(blob[len(PREFIX) :], validate=True)
    except Exception as exc:
        if strict:
            raise DecryptionError("malformed base64 ciphertext") from exc
        return tombstone
    if len(raw) < 12 + 16:
        if strict:
            raise DecryptionError("truncated AES-256-GCM payload")
        return tombstone
    nonce, ciphertext = raw[:12], raw[12:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad).decode("utf-8")
    except InvalidTag as exc:
        if strict:
            raise DecryptionError("AES-GCM authentication failed") from exc
        return tombstone
    except UnicodeDecodeError as exc:
        if strict:
            raise DecryptionError("decrypted bytes are not valid UTF-8") from exc
        return tombstone
