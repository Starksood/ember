"""Offline, zero-dependency verification suite for Ember bundles, receipts, Merkle redactions, and ledger chains.

INTENTIONALLY STANDARD-LIBRARY ONLY for all structural, Merkle, byte-range, and hash-chain proofs.
Anyone with Python 3.11+ (and no third-party packages) can copy this single file and verify:
  * flat SHA-256 byte-range receipts against original bytes
  * RFC 6962 salted Merkle inclusion proofs against partially redacted documents
  * append-only SHA-256 hash-chained ledger integrity

Ed25519 signature verification on ErasureCertificates uses `cryptography` if installed, and
explicitly reports `signature_checked=False` / `verified=None` rather than silently claiming pass
when absent.
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


def _is_strict_int(value: Any) -> bool:
    """Return True only for real ints (never `bool`, since bool subclasses int in Python)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def canonical_bytes(value: Any) -> bytes:
    """Deterministic UTF-8 ASCII-escaped canonical JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


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
    if not isinstance(left_hex, str) or not isinstance(right_hex, str):
        raise ValueError("interior digests must be hex strings")
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
    if not isinstance(leaf_digest, str) or not isinstance(expected_root, str):
        return False
    if not _is_non_string_sequence(proof):
        return False
    try:
        raw_leaf = bytes.fromhex(leaf_digest)
        raw_root = bytes.fromhex(expected_root)
    except ValueError:
        return False
    if len(raw_leaf) != 32 or len(raw_root) != 32:
        return False

    if leaf_index is not None and total_leaves is not None:
        if not _is_strict_int(leaf_index) or not _is_strict_int(total_leaves):
            return False
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
                if not _is_non_string_sequence(step) or len(step) != 2:
                    return False
                side, sibling = step[0], step[1]
                expected_side = "right" if idx % 2 == 0 else "left"
                if side != expected_side or not isinstance(sibling, str):
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
        if not _is_non_string_sequence(step) or len(step) != 2:
            return False
        side, sibling = step[0], step[1]
        if side not in ("left", "right") or not isinstance(sibling, str):
            return False
        try:
            cur = interior_hash(cur, sibling) if side == "right" else interior_hash(sibling, cur)
        except ValueError:
            return False
    return cur == expected_root


def verify_flat_receipt(receipt: Mapping[str, Any], raw_doc: bytes) -> bool:
    """Verify flat SHA-256 byte-range receipt against full unredacted document bytes."""
    if not isinstance(receipt, Mapping) or not isinstance(raw_doc, (bytes, bytearray)):
        return False
    raw_bytes = bytes(raw_doc)
    flat_hash = receipt.get("flat_doc_hash") or receipt.get("doc_hash")
    quote = receipt.get("quote")
    start = receipt.get("byte_start")
    end = receipt.get("byte_end")
    if not isinstance(flat_hash, str) or not isinstance(quote, str) or not quote:
        return False
    if not _is_strict_int(start) or not _is_strict_int(end):
        return False
    if sha256_prefixed(raw_bytes) != flat_hash:
        return False
    if not (0 <= start <= end <= len(raw_bytes)):
        return False
    try:
        return raw_bytes[start:end].decode("utf-8") == quote
    except UnicodeDecodeError:
        return False


def _redactable_leaf_spans_hashes(entry: Mapping[str, Any]) -> list[str] | None:
    """Recompute ordered span hashes for a v2 compound leaf entry cleanly."""
    spans = entry.get("spans")
    if not isinstance(spans, list) or not spans:
        return None
    if any(forbidden in entry for forbidden in ("text", "nonce", "hash")):
        return None
    has_leaf_bounds = "byte_start" in entry and "byte_end" in entry
    leaf_b_start = entry.get("byte_start", 0)
    leaf_b_end = entry.get("byte_end", 0)

    span_hashes: list[str] = []
    for expected_sidx, span in enumerate(spans):
        if not isinstance(span, Mapping):
            return None
        if "span_index" in span:
            sidx = span["span_index"]
            if not _is_strict_int(sidx) or sidx != expected_sidx:
                return None
        if "byte_start" in span or "byte_end" in span:
            b_start = span.get("byte_start")
            b_end = span.get("byte_end")
            if not _is_strict_int(b_start) or not _is_strict_int(b_end) or not (0 <= b_start <= b_end):
                return None
            if has_leaf_bounds and not (leaf_b_start <= b_start <= b_end <= leaf_b_end):
                return None
        s_redacted = span.get("redacted", False)
        if not isinstance(s_redacted, bool):
            return None
        if s_redacted:
            if "text" in span or "nonce" in span:
                return None
            h = span.get("hash")
            if not isinstance(h, str):
                return None
            try:
                if len(bytes.fromhex(h)) != 32:
                    return None
            except ValueError:
                return None
            span_hashes.append(h)
        else:
            text = span.get("text")
            nonce = span.get("nonce")
            if not isinstance(text, str) or not isinstance(nonce, str) or not nonce:
                return None
            span_hashes.append(leaf_hash(text, nonce))
    return span_hashes


def _render_spans_text(spans: Sequence[Mapping[str, Any]], redaction_marker: str = "[redacted]") -> str:
    """Render surviving/redacted sub-leaf spans preserving direct punctuation attachments."""
    parts: list[str] = []
    prev_end: int | None = None
    prev_was_redacted = False
    for span in spans:
        is_redacted = bool(span.get("redacted"))
        b_start = span.get("byte_start")
        b_end = span.get("byte_end")
        if is_redacted:
            if not prev_was_redacted:
                if parts:
                    parts.append(" ")
                parts.append(redaction_marker)
            prev_was_redacted = True
            prev_end = b_end if _is_strict_int(b_end) else None
            continue

        piece = str(span.get("text", ""))
        if parts:
            if _is_strict_int(b_start) and prev_end is not None and b_start <= prev_end and not prev_was_redacted:
                pass
            elif _is_strict_int(b_start) and prev_end is not None and b_start <= prev_end and piece and not piece[0].isalnum():
                pass
            else:
                parts.append(" ")
        parts.append(piece)
        prev_was_redacted = False
        prev_end = b_end if _is_strict_int(b_end) else None
    return "".join(parts)


def redactable_doc_hashes(doc_dict: Mapping[str, Any]) -> list[str] | None:
    """Recompute ordered leaf hashes for a serialized RedactableDoc dict cleanly."""
    if not isinstance(doc_dict, Mapping):
        return None
    entries = doc_dict.get("entries")
    if not isinstance(entries, list):
        return None
    hashes: list[str] = []
    for expected_idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            return None
        if "index" in entry:
            idx = entry["index"]
            if not _is_strict_int(idx) or idx != expected_idx:
                return None
        if "byte_start" in entry or "byte_end" in entry:
            b_start = entry.get("byte_start")
            b_end = entry.get("byte_end")
            if not _is_strict_int(b_start) or not _is_strict_int(b_end) or not (0 <= b_start <= b_end):
                return None
        redacted_flag = entry.get("redacted", False)
        if not isinstance(redacted_flag, bool):
            return None
        if redacted_flag:
            if "text" in entry or "nonce" in entry or "spans" in entry:
                # Fail closed: a redacted entry must have stripped plaintext, nonce, and spans.
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
        elif "spans" in entry:
            span_hashes = _redactable_leaf_spans_hashes(entry)
            if span_hashes is None:
                return None
            hashes.append(merkle_root(span_hashes))
        else:
            text = entry.get("text")
            nonce = entry.get("nonce")
            if not isinstance(text, str) or not isinstance(nonce, str) or not nonce:
                return None
            hashes.append(leaf_hash(text, nonce))
    return hashes


def verify_redactable_doc(doc_dict: Mapping[str, Any], expected_root: str | None = None) -> tuple[bool, str]:
    """Validate structure, redaction hygiene, and Merkle root of a RedactableDoc dict."""
    if not isinstance(doc_dict, Mapping):
        return False, "document must be a JSON object"
    hashes = redactable_doc_hashes(doc_dict)
    if hashes is None:
        return False, "invalid entries or incomplete redaction"
    computed_root = merkle_root(hashes)
    if "merkle_root" in doc_dict:
        stored_root = doc_dict["merkle_root"]
        if not isinstance(stored_root, str) or stored_root != computed_root:
            return False, "stored merkle_root mismatches recomputed root"
    if expected_root is not None and computed_root != expected_root:
        return False, "recomputed root mismatches expected merkle_root"
    return True, computed_root


def verify_redactable_receipt(receipt: Mapping[str, Any], doc_dict: Mapping[str, Any]) -> bool:
    """Verify a receipt against a (possibly redacted) RedactableDoc dictionary (v1 flat & v2 span leaves)."""
    if not isinstance(receipt, Mapping) or not isinstance(doc_dict, Mapping):
        return False
    expected_root = receipt.get("merkle_root")
    leaf_index = receipt.get("leaf_index")
    quote = receipt.get("quote")
    proof = receipt.get("proof")
    if not isinstance(expected_root, str) or not expected_root:
        return False
    if not _is_strict_int(leaf_index) or leaf_index < 0:
        return False
    if not isinstance(quote, str) or not quote.strip():
        return False
    if not isinstance(proof, list):
        return False

    rec_sid = receipt.get("source_id")
    doc_sid = doc_dict.get("source_id")
    if isinstance(rec_sid, str) and isinstance(doc_sid, str) and rec_sid and doc_sid and rec_sid != doc_sid:
        return False

    ok, _ = verify_redactable_doc(doc_dict, expected_root=expected_root)
    if not ok:
        return False
    entries = doc_dict["entries"]
    if "leaf_count" in receipt:
        leaf_count = receipt["leaf_count"]
        if not _is_strict_int(leaf_count) or leaf_count != len(entries):
            return False
    if not (0 <= leaf_index < len(entries)):
        return False
    entry = entries[leaf_index]
    if entry.get("redacted"):
        return False

    if "spans" in entry:
        spans = entry["spans"]
        span_indices = receipt.get("span_indices")
        span_proofs = receipt.get("span_proofs")
        if not isinstance(span_indices, list) or not span_indices:
            return False
        if not all(_is_strict_int(idx) for idx in span_indices):
            return False
        if span_indices != list(range(span_indices[0], span_indices[0] + len(span_indices))):
            return False
        if not (0 <= span_indices[0] and span_indices[-1] < len(spans)):
            return False
        if "span_count" in receipt:
            sc = receipt["span_count"]
            if not _is_strict_int(sc) or sc != len(spans):
                return False
        if not isinstance(span_proofs, list) or len(span_proofs) != len(span_indices):
            return False

        selected_spans: list[Mapping[str, Any]] = []
        for s_idx in span_indices:
            sp = spans[s_idx]
            if sp.get("redacted") or "text" not in sp or "nonce" not in sp:
                return False
            selected_spans.append(sp)

        joined_text = _render_spans_text(selected_spans)
        if quote != joined_text and quote not in joined_text:
            return False

        if "byte_start" in receipt or "byte_end" in receipt:
            r_start = receipt.get("byte_start")
            r_end = receipt.get("byte_end")
            if not _is_strict_int(r_start) or not _is_strict_int(r_end) or not (0 <= r_start <= r_end):
                return False
            first_span = selected_spans[0]
            last_span = selected_spans[-1]
            if "byte_start" in first_span and "byte_end" in last_span:
                if not (first_span["byte_start"] <= r_start <= r_end <= last_span["byte_end"]):
                    return False

        span_hashes = _redactable_leaf_spans_hashes(entry)
        if span_hashes is None:
            return False
        leaf_digest = merkle_root(span_hashes)
        for s_idx, s_proof in zip(span_indices, span_proofs, strict=True):
            sp = spans[s_idx]
            sp_digest = leaf_hash(sp["text"], sp["nonce"])
            if not verify_inclusion(
                sp_digest,
                s_proof,
                leaf_digest,
                leaf_index=s_idx,
                total_leaves=len(span_hashes),
            ):
                return False
        return verify_inclusion(
            leaf_digest,
            proof,
            expected_root,
            leaf_index=leaf_index,
            total_leaves=len(entries),
        )

    # Flat v1 leaf verification path
    if "span_indices" in receipt or "span_proofs" in receipt or receipt.get("span_count", 0) not in (0, None):
        return False
    if "text" not in entry:
        return False
    leaf_text = entry["text"]
    if quote != leaf_text and quote not in leaf_text:
        return False

    if "byte_start" in receipt or "byte_end" in receipt:
        r_start = receipt.get("byte_start")
        r_end = receipt.get("byte_end")
        if not _is_strict_int(r_start) or not _is_strict_int(r_end) or not (0 <= r_start <= r_end):
            return False
        if "byte_start" in entry and "byte_end" in entry:
            e_start = entry["byte_start"]
            e_end = entry["byte_end"]
            if not (e_start <= r_start <= r_end <= e_end):
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
    if not isinstance(event, Mapping):
        raise ValueError("ledger event must be a mapping")
    required = ("seq", "tenant_id", "event_id", "kind", "committed_at", "payload", "prev_hash")
    for key in required:
        if key not in event:
            raise KeyError(key)
    if not _is_strict_int(event["seq"]):
        raise ValueError("event seq must be an integer")
    if not isinstance(event["payload"], Mapping):
        raise ValueError("event payload must be a JSON object")
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
    if not _is_non_string_sequence(events):
        return False, "ledger_events must be a list of event objects"
    prev_by_tenant: dict[str, str] = {}
    seq_by_tenant: dict[str, int] = {}
    for row_idx, event in enumerate(events):
        if not isinstance(event, Mapping):
            return False, f"row {row_idx} is not a JSON object"
        try:
            tenant = event["tenant_id"]
            seq = event["seq"]
            prev_hash = event["prev_hash"]
            stored_hash = event["event_hash"]
            event_id = event["event_id"]
            kind = event["kind"]
            committed_at = event["committed_at"]
            payload = event["payload"]
        except KeyError as exc:
            return False, f"row {row_idx} missing field {exc}"
        if not isinstance(tenant, str) or not tenant:
            return False, f"row {row_idx} has invalid tenant_id"
        if not _is_strict_int(seq):
            return False, f"row {row_idx} seq must be an integer"
        if not all(isinstance(x, str) for x in (prev_hash, stored_hash, event_id, kind, committed_at)):
            return False, f"row {row_idx} has non-string required metadata"
        if not isinstance(payload, Mapping):
            return False, f"row {row_idx} payload must be a JSON object"

        expected_seq = seq_by_tenant.get(tenant, 0)
        if seq != expected_seq:
            return False, f"sequence gap for tenant {tenant!r}: expected {expected_seq}, got {seq}"
        expected_prev = prev_by_tenant.get(tenant, GENESIS_HASH)
        if prev_hash != expected_prev:
            return False, f"broken prev_hash for tenant {tenant!r} at seq {seq}"
        try:
            expected_hash = compute_ledger_event_hash(event)
        except (TypeError, ValueError, KeyError) as exc:
            return False, f"malformed event at seq {seq}: {exc}"
        if stored_hash != expected_hash:
            return False, f"invalid event_hash for tenant {tenant!r} at seq {seq}"
        seq_by_tenant[tenant] = expected_seq + 1
        prev_by_tenant[tenant] = stored_hash
    return True, f"verified {len(events)} event(s)"


def certificate_signable_bytes(cert: Mapping[str, Any]) -> bytes:
    """Canonical payload covered by an ErasureCertificate's Ed25519 signature."""
    if not isinstance(cert, Mapping):
        raise ValueError("certificate must be a JSON object")
    required = (
        "certificate_id",
        "tenant_id",
        "subject_id",
        "subject_tokens",
        "state_hash_before",
        "state_hash_after",
        "ledger_tail_hash",
        "closure",
        "probe_count",
        "battery_all_abstained",
        "cipher",
        "key_destroyed",
        "record_keys_destroyed",
        "issued_at",
        "trust_boundary",
    )
    for key in required:
        if key not in cert:
            raise ValueError(f"certificate missing required field {key!r}")
    if not _is_non_string_sequence(cert["subject_tokens"]):
        raise ValueError("certificate subject_tokens must be a list")

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
    if not isinstance(cert, Mapping):
        return False, "certificate must be a JSON object"
    sig = cert.get("signature", "")
    pub = public_key_hex if public_key_hex is not None else cert.get("public_key", "")
    if not isinstance(sig, str) or not sig.startswith("ed25519:") or not isinstance(pub, str) or not pub:
        return False, "missing or non-ed25519 signature/public_key"
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:
        return None, "cryptography package unavailable — Ed25519 signature NOT checked"

    _, _, hex_sig = sig.partition(":")
    try:
        pub_bytes = bytes.fromhex(pub)
        sig_bytes = bytes.fromhex(hex_sig)
        if len(pub_bytes) != 32 or len(sig_bytes) != 64:
            return False, "invalid Ed25519 key or signature byte length"
        pk = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        pk.verify(sig_bytes, certificate_signable_bytes(cert))
        return True, "ed25519 signature valid"
    except (InvalidSignature, ValueError, TypeError, KeyError) as exc:
        return False, f"invalid ed25519 signature: {exc}"


def verify_bundle(bundle: Mapping[str, Any], *, require_signature_check: bool = False) -> dict[str, Any]:
    """Verify a self-contained Ember verification bundle JSON document cleanly (fail-closed)."""
    if not isinstance(bundle, Mapping):
        return {"ok": False, "error": "bundle must be a JSON object", "checks": {}}

    results: dict[str, Any] = {"ok": True, "checks": {}}

    docs = bundle.get("documents", {})
    if not isinstance(docs, Mapping):
        results["checks"]["documents"] = {"ok": False, "error": "documents must be a JSON object", "items": {}}
        results["ok"] = False
        docs_map: Mapping[str, Any] = {}
    else:
        docs_map = docs
        docs_ok = True
        doc_details: dict[str, Any] = {}
        for source_id, doc_dict in docs.items():
            sid_key = str(source_id)
            ok, info = verify_redactable_doc(doc_dict)
            if ok and isinstance(doc_dict, Mapping):
                doc_sid = doc_dict.get("source_id")
                if isinstance(doc_sid, str) and doc_sid and doc_sid != sid_key:
                    ok, info = False, f"source_id mismatch: key={sid_key!r} != doc.source_id={doc_sid!r}"
            doc_details[sid_key] = {"ok": ok, "info": info}
            if not ok:
                docs_ok = False
        results["checks"]["documents"] = {"ok": docs_ok, "items": doc_details}
        if not docs_ok:
            results["ok"] = False

    receipts = bundle.get("receipts", [])
    if not _is_non_string_sequence(receipts):
        results["checks"]["receipts"] = {"ok": False, "error": "receipts must be a list", "items": []}
        results["ok"] = False
    else:
        receipts_ok = True
        receipt_details: list[dict[str, Any]] = []
        for item in receipts:
            if not isinstance(item, Mapping):
                receipt_details.append({"source_id": None, "leaf_index": None, "ok": False})
                receipts_ok = False
                continue
            sid = item.get("source_id", "")
            doc_dict = docs_map.get(sid) if isinstance(sid, str) else None
            ok = bool(isinstance(doc_dict, Mapping) and verify_redactable_receipt(item, doc_dict))
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
        certs = bundle["certificates"]
        if not _is_non_string_sequence(certs):
            results["checks"]["certificates"] = {"ok": False, "error": "certificates must be a list", "items": []}
            results["ok"] = False
        else:
            certs_ok = True
            cert_details: list[dict[str, Any]] = []
            for cert in certs:
                if not isinstance(cert, Mapping):
                    cert_details.append({"certificate_id": None, "verified": False, "detail": "certificate must be an object"})
                    certs_ok = False
                    continue
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
    try:
        raw_text = args.bundle.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": f"failed to read bundle: {exc}", "checks": {}}, indent=2))
        return 1
    report = verify_bundle(data, require_signature_check=args.require_signature)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
