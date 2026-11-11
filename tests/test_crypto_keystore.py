"""AES-256-GCM AEAD, AAD record-swap protection, and FileKeyStore shredding lifecycle."""

from __future__ import annotations

import stat

import pytest

from ember.crypto import DecryptionError, ERASED_TOMBSTONE, build_aad, decrypt_text, encrypt_text, new_key
from ember.keystore import FileKeyStore, MemoryKeyStore


def test_encrypt_decrypt_roundtrip_and_aad_binding():
    key = new_key()
    aad_one = build_aad(tenant_id="acme", record_id="item_1", field="text", key_ref="subject:priya")
    aad_two = build_aad(tenant_id="acme", record_id="item_2", field="text", key_ref="subject:priya")

    ct = encrypt_text(key, "Priya lives in Portland", aad=aad_one)
    assert ct.startswith("enc2:") and "Priya" not in ct and "Portland" not in ct
    assert decrypt_text(key, ct, aad=aad_one, strict=True) == "Priya lives in Portland"

    # Copying ciphertext to a different record's AAD fails AES-GCM authentication
    assert decrypt_text(key, ct, aad=aad_two, strict=False) == ERASED_TOMBSTONE
    with pytest.raises(DecryptionError):
        decrypt_text(key, ct, aad=aad_two, strict=True)


def test_missing_or_wrong_key_tombstones_instead_of_leaking():
    aad = build_aad(tenant_id="t", record_id="r", field="text", key_ref="subject:s1")
    ct = encrypt_text(new_key(), "sensitive fact", aad=aad)
    assert decrypt_text(None, ct, aad=aad) == ERASED_TOMBSTONE
    assert decrypt_text(new_key(), ct, aad=aad) == ERASED_TOMBSTONE


def test_memory_keystore_subject_and_shared_record_shredding():
    ks = MemoryKeyStore()
    k_priya = ks.get_or_create_subject_key("priya")
    k_dana = ks.get_or_create_subject_key("dana")
    k_shared = ks.create_record_key("rec_summary", ["priya", "dana"])
    k_dana_rec = ks.create_record_key("rec_dana_only", ["dana"])

    assert len(k_priya) == 32 and len(k_shared) == 32
    report = ks.shred_subject("priya")
    assert report.subject_key_destroyed is True
    assert report.record_keys_destroyed == 1

    assert ks.get_subject_key("priya") is None
    assert ks.get_record_key("rec_summary") is None
    # Unshared bystander keys stay completely intact
    assert ks.get_subject_key("dana") == k_dana
    assert ks.get_record_key("rec_dana_only") == k_dana_rec


def test_file_keystore_persists_with_0600_and_tombstone_tokens_are_salted(tmp_path):
    key_file = tmp_path / "keys" / "keyring.json"
    ks = FileKeyStore(key_file)
    ks.get_or_create_subject_key("subj_a")
    tok1 = ks.tombstone_token("Priya Raman")

    mode = stat.S_IMODE(key_file.stat().st_mode)
    assert mode == 0o600

    reloaded = FileKeyStore(key_file)
    assert reloaded.get_subject_key("subj_a") == ks.get_subject_key("subj_a")
    assert reloaded.tombstone_token("  priya   raman ") == tok1

    # Different installation with different tombstone secret produces completely unrelated HMAC token
    other = MemoryKeyStore()
    assert other.tombstone_token("Priya Raman") != tok1
