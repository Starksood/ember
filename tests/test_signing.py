"""Ed25519 certificate verification and external timestamp anchor behavior."""

from __future__ import annotations

from ember.signing import CallbackAnchor, Ed25519Signer, NullAnchor, sign_certificate_dict
from ember.verify import verify_ed25519_certificate


def _sample_unsigned_cert(anchor) -> dict:
    ar = anchor.anchor("sha256:abcd")
    return {
        "certificate_id": "cert_001",
        "tenant_id": "acme",
        "subject_id": "priya",
        "subject_tokens": ["hmac-sha256:1234"],
        "state_hash_before": "sha256:11",
        "state_hash_after": "sha256:22",
        "ledger_tail_hash": "sha256:33",
        "closure": {"subject_id": "priya", "item_ids": ["item_1"]},
        "probe_count": 3,
        "battery_all_abstained": True,
        "cipher": "AES-256-GCM",
        "key_destroyed": True,
        "record_keys_destroyed": 1,
        "key_destroyed_at": "2026-11-10T00:00:00Z",
        "issued_at": "2026-11-10T00:00:01Z",
        "trust_boundary": {"externally_timestamped": ar.scheme != "none"},
        "anchor": ar.as_dict(),
    }


def test_ed25519_certificate_verifies_with_public_key_and_catches_tamper(tmp_path):
    signer = Ed25519Signer.load_or_create(tmp_path / "signer.json")
    signed = sign_certificate_dict(_sample_unsigned_cert(NullAnchor()), signer)

    ok, _ = verify_ed25519_certificate(signed)
    assert ok is True

    # Tampering any attested field breaks signature under public-key-only verification
    tampered = dict(signed)
    tampered["probe_count"] = 99
    tampered_ok, _ = verify_ed25519_certificate(tampered, public_key_hex=signer.public_key_hex)
    assert tampered_ok is False


def test_callback_timestamp_anchor_is_covered_by_certificate_signature():
    signer = Ed25519Signer.generate()
    seen: list[str] = []
    anchor = CallbackAnchor("rfc3161-demo", lambda d: (seen.append(d) or f"tsa-tok:{d[:12]}"))
    signed = sign_certificate_dict(_sample_unsigned_cert(anchor), signer)

    assert len(seen) == 1
    assert signed["anchor"]["scheme"] == "rfc3161-demo"
    assert verify_ed25519_certificate(signed)[0] is True

    forged_anchor = dict(signed)
    forged_anchor["anchor"] = {"scheme": "rfc3161-demo", "digest": "sha256:abcd", "token": "spoofed"}
    assert verify_ed25519_certificate(forged_anchor)[0] is False
