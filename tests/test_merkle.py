"""Cryptographic & geometric properties of redactable salted Merkle documents."""

from __future__ import annotations

import pytest

from ember.merkle import (
    RedactableDoc,
    inclusion_proof,
    leaf_hash,
    merkle_root,
    split_parts,
    split_parts_with_spans,
    verify_inclusion,
)
from ember.receipts import mint_flat_receipt, mint_redactable_receipt
from ember.verify import EMPTY_TREE_ROOT, interior_hash, verify_redactable_receipt

SHARED_DOC = (
    "Priya Raman joined Acme in 2019 as a logistics analyst. "
    "Marcus Webb owns a bookshop in Leeds. "
    "Dana Kim has a cat named Pepper."
)


def test_all_three_horns_of_trilemma_hold_simultaneously():
    doc = RedactableDoc.from_text(SHARED_DOC, source_id="intake_1")
    bystander_receipt = mint_redactable_receipt(
        doc,
        "Marcus Webb owns a bookshop in Leeds.",
        raw_doc=SHARED_DOC.encode("utf-8"),
    )
    assert bystander_receipt.verify_flat(SHARED_DOC.encode("utf-8"))
    assert bystander_receipt.verify_redactable(doc)

    redacted = doc.redact(lambda text: "Priya" in text)

    assert redacted.root() == doc.root()
    assert "Priya" not in redacted.text()
    assert "logistics analyst" not in redacted.text()
    assert "Marcus Webb" in redacted.text()
    assert bystander_receipt.verify_redactable(redacted)
    assert verify_redactable_receipt(bystander_receipt.as_dict(), redacted.as_dict())


def test_redacted_receipt_fails_after_its_own_leaf_is_erased():
    doc = RedactableDoc.from_text(SHARED_DOC, source_id="intake_1")
    priya_receipt = mint_redactable_receipt(doc, "Priya Raman joined Acme in 2019 as a logistics analyst.")
    assert priya_receipt.verify_redactable(doc)

    redacted = doc.redact(lambda text: "Priya" in text)
    assert not priya_receipt.verify_redactable(redacted)


def test_redacted_entry_resists_dictionary_guessing_and_strips_nonce():
    doc = RedactableDoc.from_text(SHARED_DOC, source_id="intake_1")
    redacted = doc.redact(lambda text: "Priya" in text)

    redacted_entry = redacted.entries[0]
    assert redacted_entry["redacted"] is True
    assert "text" not in redacted_entry
    assert "nonce" not in redacted_entry

    guess = "Priya Raman joined Acme in 2019 as a logistics analyst."
    assert leaf_hash(guess, "") != redacted_entry["hash"]


def test_length_prefixed_nonce_delimiting_prevents_boundary_collision():
    assert leaf_hash("bc", "a") != leaf_hash("abc", "")
    sample = "abcdef"
    digests = {leaf_hash(sample[i:], sample[:i]) for i in range(len(sample) + 1)}
    assert len(digests) == len(sample) + 1


def test_rfc6962_domain_separation_and_empty_tree_distinct_from_empty_leaf():
    a, b = leaf_hash("a", "n1"), leaf_hash("b", "n2")
    assert interior_hash(a, b) != leaf_hash(a + b, "")
    assert EMPTY_TREE_ROOT != leaf_hash("", "")


def test_every_leaf_verifies_across_odd_and_even_tree_sizes_1_to_33():
    for n in range(1, 34):
        hashes = [leaf_hash(f"part-{i}", f"nonce-{i}") for i in range(n)]
        root = merkle_root(hashes)
        for idx in range(n):
            proof = inclusion_proof(hashes, idx)
            assert verify_inclusion(hashes[idx], proof, root, leaf_index=idx, total_leaves=n), f"n={n} idx={idx}"
            # Swapping metadata leaf_index must fail strict positional check when mismatched
            wrong_idx = (idx + 1) % n
            if n > 1 and wrong_idx != idx:
                assert not verify_inclusion(
                    hashes[idx],
                    proof,
                    root,
                    leaf_index=wrong_idx,
                    total_leaves=n,
                )


def test_tampering_surviving_leaf_breaks_verification():
    doc = RedactableDoc.from_text(SHARED_DOC, source_id="intake_1")
    bystander_receipt = mint_redactable_receipt(doc, "Marcus Webb owns a bookshop in Leeds.")
    tampered_dict = doc.redact(lambda text: "Priya" in text).as_dict()
    tampered_dict["entries"][1]["text"] = "Marcus Webb owns a casino in Leeds."
    assert not verify_redactable_receipt(bystander_receipt.as_dict(), tampered_dict)


def test_line_chunk_mode_and_exact_utf8_byte_offsets():
    multiline = "Priya works in Portland — first line.\nMarcus lives in Leeds — naïve café.\n"
    spans = split_parts_with_spans(multiline, mode="line")
    assert len(spans) == 2
    raw = multiline.encode("utf-8")
    for b_start, b_end, slice_text in spans:
        assert raw[b_start:b_end].decode("utf-8") == slice_text

    doc = RedactableDoc.from_text(multiline, source_id="lines", mode="line")
    rec = mint_redactable_receipt(doc, "naïve café", raw_doc=raw)
    assert rec.verify_flat(raw)
    assert rec.verify_redactable(doc.redact(lambda s: "Priya" in s))


def test_flat_receipt_rejects_absent_or_modified_document():
    raw = b"Priya joined Acme in 2019."
    rec = mint_flat_receipt(raw, "Acme in 2019", source_id="doc1")
    assert rec.verify_flat(raw)
    assert not rec.verify_flat(raw.replace(b"2019", b"2018"))
    with pytest.raises(ValueError):
        mint_flat_receipt(raw, "nonexistent quote")
