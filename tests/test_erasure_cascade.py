"""Exact cascade erasure, shared-doc redaction, derived summary shredding, and tombstones."""

from __future__ import annotations

import pytest

from ember.reducer import ErasureIncomplete, TombstonedSubjectError, decrypt_item_payload
from ember.vault import CustodyVault
from ember.verify import verify_redactable_receipt


def test_transitive_derived_invalidation_and_historical_key_shredding():
    vault = CustodyVault.ephemeral()
    doc_text = "Priya Raman works at Riverside. Dana Kim leads pharmacy."
    vault.register_source("intake", doc_text)

    r_priya = vault.remember(
        "Priya Raman works at Riverside.",
        subjects=["priya"],
        source_id="intake",
        quote="Priya Raman works at Riverside.",
    )
    r_dana = vault.remember(
        "Dana Kim leads pharmacy.",
        subjects=["dana"],
        source_id="intake",
        quote="Dana Kim leads pharmacy.",
    )
    # Two-hop derived chain containing ZERO explicit subject tag or name mention of Priya!
    r_mid = vault.remember(
        "Two clinical staff coordinate ward shifts.",
        subjects=["team_ops"],
        derived_from=[r_priya.item.item_id, r_dana.item.item_id],
    )
    r_top = vault.remember(
        "Riverside morning shift coverage is staffed.",
        subjects=["exec_brief"],
        derived_from=[r_mid.item.item_id],
    )

    cert = vault.erase("priya", probes=["Who works at Riverside?", "Who coordinates ward shifts?"])

    assert cert.key_destroyed is True
    assert r_mid.item.item_id in cert.closure["derived_invalidated"]
    assert r_top.item.item_id in cert.closure["derived_invalidated"]
    assert r_dana.item.item_id not in cert.closure["item_ids"]

    # Bystander (Dana) stays searchable and her Merkle receipt STILL verifies against redacted doc!
    dana_res = vault.search("Dana pharmacy")
    assert not dana_res.abstained
    assert dana_res.hits[0].receipt_verified is True
    redacted_doc = vault.ledger.get_document(vault.tenant_id, "intake")
    assert redacted_doc is not None
    assert "Priya" not in redacted_doc.text()
    assert verify_redactable_receipt(r_dana.receipt.as_dict(), redacted_doc.as_dict())

    # Inspecting EVERY raw historical ADD_ITEM event post-shred proves Priya & derived items are [erased]
    erased_ids = set(cert.closure["item_ids"])
    for ev in vault.ledger.events(vault.tenant_id):
        if ev.kind == "ADD_ITEM":
            item_payload = ev.payload["item"]
            subjs, pt, pq = decrypt_item_payload(item_payload, vault.keystore)
            if item_payload["item_id"] in erased_ids:
                assert pt == "[erased]" and subjs == ()
                assert pq in (None, "[erased]")
            else:
                assert item_payload["item_id"] == r_dana.item.item_id
                assert "Dana Kim" in pt

    vault.close()


def test_tombstone_blocks_silent_reingest_including_aliases_and_requires_reconsent():
    vault = CustodyVault.ephemeral()
    vault.remember("Priya lives in Portland.", subjects=["usr_priya"], aliases=["Priya Raman"])
    vault.erase("usr_priya", aliases=["Priya Raman"], probes=["Where does Priya live?"])

    with pytest.raises(TombstonedSubjectError):
        vault.remember("Priya lives in Portland.", subjects=["usr_priya"])

    with pytest.raises(TombstonedSubjectError):
        vault.remember("Priya Raman joined Acme.", subjects=["different_bucket_id"], aliases=["priya   raman"])

    # Explicit audited re-consent is allowed cleanly and works with new keys
    reconsented = vault.remember(
        "Priya opted back into concierge updates.",
        subjects=["usr_priya"],
        reconsent_id="consent_ticket_2026_11",
    )
    assert reconsented.item.reconsent_id == "consent_ticket_2026_11"
    assert not vault.search("concierge updates").abstained
    vault.close()


def test_erasure_incomplete_withholds_certificate_without_rolling_back_shred():
    """If an unlinked duplicate/residue still leaks on direct probes, fail closed and issue NO certificate."""
    vault = CustodyVault.ephemeral()
    vault.remember("Priya works at Riverside.", subjects=["priya"])
    # Simulate an unlinked mention filed under a different subject with no derived_from link
    vault.remember("Priya also manages inventory at Riverside.", subjects=["unlinked_ops"])

    with pytest.raises(ErasureIncomplete) as exc_info:
        vault.erase("priya", probes=["Where does Priya work?"])

    assert "Where does Priya work?" in exc_info.value.leaking_probes
    # Certificate was NOT issued / stored
    assert vault.ledger.list_certificates(vault.tenant_id) == []
    # Yet the destructive crypto-shred of priya's own key DID proceed (never rolled back)
    assert vault.keystore.get_subject_key(vault.keystore.subject_ref("priya")) is None
    vault.close()
