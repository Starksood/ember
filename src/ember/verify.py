"""Offline, zero-dependency verification suite for Ember bundles, receipts, Merkle redactions, and ledger chains.

INTENTIONALLY STANDARD-LIBRARY ONLY for all structural, Merkle, byte-range, and hash-chain proofs.
Anyone with Python 3.11+ (and no third-party packages) can copy this single file and verify:
  * flat SHA-256 byte-range receipts against original bytes
  * RFC 6962 salted Merkle inclusion proofs against partially redacted documents
  * append-only SHA-256 hash-chained ledger integrity

Ed25519 signature verification on ErasureCertificates uses `cryptography` if installed, and
explicitly reports `signature_checked=False` rather than silently claiming pass when absent.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"
EMPTY_TREE_ROOT = hashlib.sha256(LEAF_PREFIX).hexdigest()
GENESIS_HASH = "sha256:" + ("0" * 64)


def canonical_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 ASCII-escaped canonical JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_prefixed(data: bytes) -> str:
    """Compute canonical `sha256:<hex>` digest."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def leaf_hash(part: str, nonce: str = "") -> str:
    """Compute length-delimited domain-separated leaf digest.

    H(0x00 || uint32_be(len(nonce_bytes)) || nonce_bytes || part_bytes)

    Length-prefixing `nonce_bytes` ensures a surviving leaf `(nonce, part)` and a
    redacted/unnonced candidate `("", nonce + part)` can never collide.
    """
    nonce_bytes = nonce.encode("utf-8")
    part_bytes = part.encode("utf-8")
    return hashlib.sha256(
        LEAF_PREFIX + len(nonce_bytes).to_bytes(4, "big") + nonce_bytes + part_bytes
    ).hexdigest()


def interior_hash(left_hex: str, right_hex: str) -> str:
    """Compute RFC 6962 interior node digest: H(0x01 || left_32 || right_32)."""
    left = bytes.fromhex(left_hex)
    right = bytes.fromhex(right_hex)
    if len(left) != 32 or len(right) != 32:
        raise ValueError("interior digests must be 32-byte SHA-256 hex strings")
    return hashlib.sha256(NODE_PREFIX + left + right).hexdigest()


def merkle_root(leaf_digests: Sequence[str]) -> str:
    """Compute root over ordered leaf digests; odd nodes promote unchanged."""
    if not leaf_digests:
        return EMPTY_TREE_ROOT
    level = list(leaf_digests)
    while len(level) > 1:
        nxt: list[str] = []
        for idx in range(0, len(level) - 1, 2):
            nxt.append(interior_hash(level[idx], level[idx + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def verify_inclusion(
    leaf_digest: str,
    proof: Sequence[Sequence[str]],
    expected_root: str,
    *,
    leaf_index: int | None = None,
    total_leaves: int | None = None,
) -> bool:
    """Verify an inclusion proof against `expected_root` (odd promotion semantics).

    When `leaf_index` and `total_leaves` are both provided, also verifies that each
    proof sibling side strictly matches the tree walk for that leaf position.
    """
    try:
        raw_leaf = bytes.fromhex(leaf_digest)
        raw_root = bytes.fromhex(expected_root)
    except ValueError:
        return False
    if len(raw_leaf) != 32 or len(raw_root) != 32:
        return False

    if leaf_index is not None and total_leaves is not None:
        if total_leaves <= 0 or not (0 <= leaf_index < total_leaves):
            return False
        idx = leaf_index
        width = total_leaves
        proof_pos = 0
        cur = leaf_digest
        while width > 1:
            promoted_last = (width % 2 == 1) and (idx == width - 1)
            if promoted_last:
                idx = (width - 1) // 2
            else:
                if proof_pos >= len(proof):
                    return False
                step = proof[proof_pos]
                if len(step) != 2:
                    return False
                side, sibling = step[0], step[1]
                expected_side = "right" if idx % 2 == 0 else "left"
                if side != expected_side:
                    return False
                try:
                    cur = interior_hash(cur, sibling) if side == "right" else interior_hash(sibling, cur)
                except ValueError:
                    return False
                proof_pos += 1
                idx //= 2
            width = (width + 1) // 2
        if proof_pos != len(proof):
            return False
        return cur == expected_root

    cur = leaf_digest
    for step in proof:
        if len(step) != 2:
            return False
        side, sibling = step[0], step[1]
        if side not in ("left", "right"):
            return False
        try:
            cur = interior_hash(cur, sibling) if side == "right" else interior_hash(sibling, cur)
        except ValueError:
            return False
    return cur == expected_root


def verify_flat_receipt(receipt: Mapping[str, Any], raw_doc: bytes) -> bool:
    """Verify flat SHA-256 byte-range receipt against full unredacted document bytes."""
    flat_hash = receipt.get("flat_doc_hash") or receipt.get("doc_hash")
    quote = receipt.get("quote")
    start = receipt.get("byte_start")
    end = receipt.get("byte_end")
    if not isinstance(flat_hash, str) or not isinstance(quote, str):
        return False
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if sha256_prefixed(raw_doc) != flat_hash:
        return False
    if not (0 <= start <= end <= len(raw_doc)):
        return False
    try:
        return raw_doc[start:end].decode("utf-8") == quote
    except UnicodeDecodeError:
        return False


def redactable_doc_hashes(doc_dict: Mapping[str, Any]) -> list[str] | None:
    """Recompute ordered leaf hashes for a serialized RedactableDoc dict cleanly."""
    entries = doc_dict.get("entries")
    if not isinstance(entries, list):
        return None
    hashes: list[str] = []
    for expected_idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            return None
        if entry.get("index", expected_idx) != expected_idx:
            return None
        if entry.get("redacted"):
            if "text" in entry or "nonce" in entry:
                # Fail closed: a redacted entry must have stripped plaintext and nonce.
                return None
            h = entry.get("hash")
            if not isinstance(h, str):
                return None
            try:
                if len(bytes.fromhex(h)) != 32:
                    return None
            except ValueError:
                return None
            hashes.append(h)
        else:
            text = entry.get("text")
            nonce = entry.get("nonce")
            if not isinstance(text, str) or not isinstance(nonce, str) or not nonce:
                return None
            hashes.append(leaf_hash(text, nonce))
    return hashes


def verify_redactable_doc(doc_dict: Mapping[str, Any], expected_root: str | None = None) -> tuple[bool, str]:
    """Validate structure, redaction hygiene, and Merkle root of a RedactableDoc dict."""
    hashes = redactable_doc_hashes(doc_dict)
    if hashes is None:
        return False, "invalid entries or incomplete redaction"
    computed_root = merkle_root(hashes)
    stored_root = doc_dict.get("merkle_root")
    if isinstance(stored_root, str) and stored_root != computed_root:
        return False, "stored merkle_root mismatches recomputed root"
    if expected_root is not None and computed_root != expected_root:
        return False, "recomputed root mismatches expected merkle_root"
    return True, computed_root


def verify_redactable_receipt(receipt: Mapping[str, Any], doc_dict: Mapping[str, Any]) -> bool:
    """Verify a receipt against a (possibly redacted) RedactableDoc dictionary."""
    expected_root = receipt.get("merkle_root")
    leaf_index = receipt.get("leaf_index")
    quote = receipt.get("quote")
    proof = receipt.get("proof")
    if not isinstance(expected_root, str) or not expected_root:
        return False
    if not isinstance(leaf_index, int) or leaf_index < 0:
        return False
    if not isinstance(quote, str) or not quote:
        return False
    if not isinstance(proof, list):
        return False

    ok, _ = verify_redactable_doc(doc_dict, expected_root=expected_root)
    if not ok:
        return False
    entries = doc_dict["entries"]
    if not (0 <= leaf_index < len(entries)):
        return False
    entry = entries[leaf_index]
    if entry.get("redacted") or "text" not in entry:
        return False
    leaf_text = entry["text"]
    if quote != leaf_text and quote not in leaf_text:
        return False
    computed_leaf = leaf_hash(leaf_text, entry["nonce"])
    return verify_inclusion(
        computed_leaf,
        proof,
        expected_root,
        leaf_index=leaf_index,
        total_leaves=len(entries),
    )


def compute_ledger_event_hash(event: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 event hash covering all fields except `event_hash`."""
    hashable = {
        "seq": event["seq"],
        "tenant_id": event["tenant_id"],
        "event_id": event["event_id"],
        "kind": event["kind"],
        "committed_at": event["committed_at"],
        "payload": event["payload"],
        "prev_hash": event["prev_hash"],
    }
    return sha256_prefixed(canonical_bytes(hashable))


def verify_ledger_chain(events: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    """Verify monotonic gap-free seq, `prev_hash` linkage, and per-event SHA-256 digests."""
    prev_by_tenant: dict[str, str] = {}
    seq_by_tenant: dict[str, int] = {}
    for row_idx, event in enumerate(events):
        try:
            tenant = event["tenant_id"]
            seq = event["seq"]
            prev_hash = event["prev_hash"]
            stored_hash = event["event_hash"]
        except KeyError as exc:
            return False, f"row {row_idx} missing field {exc}"
        expected_seq = seq_by_tenant.get(tenant, 0)
        if seq != expected_seq:
            return False, f"sequence gap for tenant {tenant!r}: expected {expected_seq}, got {seq}"
        expected_prev = prev_by_tenant.get(tenant, GENESIS_HASH)
        if prev_hash != expected_prev:
            return False, f"broken prev_hash for tenant {tenant!r} at seq {seq}"
        expected_hash = compute_ledger_event_hash(event)
        if stored_hash != expected_hash:
            return False, f"invalid event_hash for tenant {tenant!r} at seq {seq}"
        seq_by_tenant[tenant] = expected_seq + 1
        prev_by_tenant[tenant] = stored_hash
    return True, f"verified {len(events)} event(s)"


def certificate_signable_bytes(cert: Mapping[str, Any]) -> bytes:
    """Canonical payload covered by an ErasureCertificate's Ed25519 signature."""
    anchor = cert.get("anchor")
    anchor_payload = (
        {"scheme": anchor.get("scheme", "none"), "digest": anchor.get("digest", ""), "token": anchor.get("token", "")}
        if isinstance(anchor, Mapping)
        else {"scheme": "none", "digest": "", "token": ""}
    )
    body = {
        "certificate_id": cert["certificate_id"],
        "tenant_id": cert["tenant_id"],
        "subject_id": cert["subject_id"],
        "subject_tokens": list(cert["subject_tokens"]),
        "state_hash_before": cert["state_hash_before"],
        "state_hash_after": cert["state_hash_after"],
        "ledger_tail_hash": cert["ledger_tail_hash"],
        "closure": cert["closure"],
        "probe_count": cert["probe_count"],
        "battery_all_abstained": cert["battery_all_abstained"],
        "cipher": cert["cipher"],
        "key_destroyed": cert["key_destroyed"],
        "record_keys_destroyed": cert["record_keys_destroyed"],
        "key_destroyed_at": cert.get("key_destroyed_at"),
        "issued_at": cert["issued_at"],
        "trust_boundary": cert["trust_boundary"],
        "anchor": anchor_payload,
    }
    return canonical_bytes(body)


def verify_ed25519_certificate(cert: Mapping[str, Any], public_key_hex: str | None = None) -> tuple[bool | None, str]:
    """Verify Ed25519 signature on a serialized certificate dict if `cryptography` is available.

    Returns `(True, reason)` when valid, `(False, reason)` on tamper/invalid signature, or
    `(None, reason)` when `cryptography` is not installed (refusing to silently pass).
    """
    sig = cert.get("signature", "")
    pub = public_key_hex or cert.get("public_key", "")
    if not isinstance(sig, str) or not sig.startswith("ed25519:") or not isinstance(pub, str) or not pub:
        return False, "missing or non-ed25519 signature/public_key"
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:
        return None, "cryptography package unavailable — Ed25519 signature NOT checked"

    _, _, hex_sig = sig.partition(":")
    try:
        pk = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub))
        pk.verify(bytes.fromhex(hex_sig), certificate_signable_bytes(cert))
        return True, "ed25519 signature valid"
    except (InvalidSignature, ValueError) as exc:
        return False, f"invalid ed25519 signature: {exc}"


def verify_bundle(bundle: Mapping[str, Any], *, require_signature_check: bool = False) -> dict[str, Any]:
    """Verify a self-contained Ember verification bundle JSON document."""
    results: dict[str, Any] = {"ok": True, "checks": {}}

    docs = bundle.get("documents", {})
    docs_ok = True
    doc_details: dict[str, Any] = {}
    for source_id, doc_dict in docs.items():
        ok, info = verify_redactable_doc(doc_dict)
        doc_details[source_id] = {"ok": ok, "info": info}
        if not ok:
            docs_ok = False
    results["checks"]["documents"] = {"ok": docs_ok, "items": doc_details}
    if not docs_ok:
        results["ok"] = False

    receipts = bundle.get("receipts", [])
    receipts_ok = True
    receipt_details: list[dict[str, Any]] = []
    for item in receipts:
        sid = item.get("source_id", "")
        doc_dict = docs.get(sid)
        ok = bool(doc_dict is not None and verify_redactable_receipt(item, doc_dict))
        receipt_details.append({"source_id": sid, "leaf_index": item.get("leaf_index"), "ok": ok})
        if not ok:
            receipts_ok = False
    results["checks"]["receipts"] = {"ok": receipts_ok, "items": receipt_details}
    if not receipts_ok:
        results["ok"] = False

    if "ledger_events" in bundle:
        chain_ok, chain_msg = verify_ledger_chain(bundle["ledger_events"])
        results["checks"]["ledger_chain"] = {"ok": chain_ok, "detail": chain_msg}
        if not chain_ok:
            results["ok"] = False

    if "certificates" in bundle:
        certs_ok = True
        cert_details: list[dict[str, Any]] = []
        for cert in bundle["certificates"]:
            sig_ok, sig_msg = verify_ed25519_certificate(cert)
            item_ok = (sig_ok is True) if require_signature_check else (sig_ok is not False)
            cert_details.append(
                {
                    "certificate_id": cert.get("certificate_id"),
                    "verified": sig_ok,
                    "detail": sig_msg,
                }
            )
            if not item_ok:
                certs_ok = False
        results["checks"]["certificates"] = {"ok": certs_ok, "items": cert_details}
        if not certs_ok:
            results["ok"] = False

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ember-verify",
        description="Verify an Ember verification bundle JSON file offline (stdlib-only for Merkle/ledger checks).",
    )
    parser.add_argument("bundle", type=Path, help="Path to exported bundle.json")
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Fail if Ed25519 signatures cannot be verified (requires cryptography)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    data = json.loads(args.bundle.read_text(encoding="utf-8"))
    report = verify_bundle(data, require_signature_check=args.require_signature)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
