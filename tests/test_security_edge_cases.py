"""Comprehensive pre-launch security, concurrency, adapter compatibility, and hostile-input tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from typing import Any

import pytest

from ember import (
    CustodyVault,
    DecryptionError,
    ERASED_TOMBSTONE,
    Ed25519Signer,
    FileKeyStore,
    InvariantViolation,
    Mem0Adapter,
    MemoryKeyStore,
    RedactableDoc,
    SQLiteStoreAdapter,
    TombstonedSubjectError,
    VectorStoreAdapter,
    cli,
    decrypt_item_metadata,
    mint_redactable_receipt,
    verify_bundle,
    verify_ed25519_certificate,
)
from ember.crypto import build_aad, decrypt_text, encrypt_text, new_key
import ember.verify as verify_mod


def test_caller_metadata_is_encrypted_in_ledger_and_shredded_on_erase(tmp_path):
    vault_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"
    vault = CustodyVault.open(vault_dir, key_dir=key_dir)

    res = vault.remember(
        "Priya works triage in Portland.",
        subjects=["priya"],
        metadata={
            "user_id": "priya_internal_99",
            "email": "priya.secret@clinic.example",
            "national_id": "nid-987654-secret",
        },
    )
    assert res.item.metadata["email"] == "priya.secret@clinic.example"
    vault.close()

    # Raw SQLite ledger must never hold caller metadata PII in plaintext
    conn = sqlite3.connect(vault_dir / "ledger.db")
    blob = " ".join(r[0] for r in conn.execute("SELECT payload_json FROM ledger_events").fetchall())
    conn.close()
    assert "priya.secret@clinic.example" not in blob
    assert "priya_internal_99" not in blob
    assert "nid-987654-secret" not in blob

    # Reopening vault reconstructs caller metadata while key exists
    reopened = CustodyVault.open(vault_dir, key_dir=key_dir)
    assert reopened.state.items[res.item.item_id].metadata["email"] == "priya.secret@clinic.example"

    reopened.erase("priya", probes=["Where does Priya work?", "nid-987654-secret"])
    # Historical ADD_ITEM event now yields empty metadata when key is shredded
    ev = next(e for e in reopened.ledger.events(reopened.tenant_id) if e.kind == "ADD_ITEM")
    assert decrypt_item_metadata(ev.payload["item"], reopened.keystore) == {}
    reopened.close()


def test_aes_gcm_cross_field_and_cross_tenant_aad_isolation():
    key = new_key()
    aad_subjects = build_aad(tenant_id="acme", record_id="i1", field="subjects", key_ref="subject:p1")
    aad_text = build_aad(tenant_id="acme", record_id="i1", field="text", key_ref="subject:p1")
    aad_other_tenant = build_aad(tenant_id="globex", record_id="i1", field="text", key_ref="subject:p1")

    ct = encrypt_text(key, '["priya"]', aad=aad_subjects)
    assert decrypt_text(key, ct, aad=aad_text, strict=False) == ERASED_TOMBSTONE
    assert decrypt_text(key, ct, aad=aad_other_tenant, strict=False) == ERASED_TOMBSTONE
    with pytest.raises(DecryptionError):
        decrypt_text(key, ct, aad=aad_text, strict=True)
    with pytest.raises(DecryptionError):
        decrypt_text(key, "enc2:not_valid_base64!!!", aad=aad_subjects, strict=True)
    with pytest.raises(DecryptionError):
        decrypt_text(key, "enc2:c2hvcnQ=", aad=aad_subjects, strict=True)


def test_verifier_rejects_type_confusion_spoofing_and_malformed_bundles(tmp_path):
    vault = CustodyVault.ephemeral()
    vault.register_source("doc1", "Priya lives in Portland. Marcus lives in Leeds.")
    vault.remember("Priya lives in Portland.", subjects=["priya"], source_id="doc1", quote="Priya lives in Portland.")
    vault.remember("Marcus lives in Leeds.", subjects=["marcus"], source_id="doc1", quote="Marcus lives in Leeds.")
    vault.erase("priya", probes=["Where does Priya live?"])
    bundle = vault.export_bundle()
    vault.close()

    # 1. Non-dict top level & wrong container types fail closed without crashing
    for bad in ([], "bad", 123, None):
        assert verify_bundle(bad)["ok"] is False  # type: ignore[arg-type]
    for patch in (
        {"documents": []},
        {"receipts": "not-a-list"},
        {"receipts": [None, 42]},
        {"ledger_events": "bad"},
        {"ledger_events": [None, {}]},
        {"certificates": "bad"},
        {"certificates": [None, {}]},
    ):
        mutated = {**bundle, **patch}
        assert verify_bundle(mutated)["ok"] is False

    # 2. Bool-is-int confusion attacks (Python isinstance(True, int) == True)
    bool_leaf = json.loads(json.dumps(bundle))
    bool_leaf["receipts"][0]["leaf_index"] = False
    assert verify_bundle(bool_leaf)["ok"] is False

    bool_seq = json.loads(json.dumps(bundle))
    bool_seq["ledger_events"][0]["seq"] = False
    assert verify_bundle(bool_seq)["ok"] is False

    bool_start = json.loads(json.dumps(bundle))
    bool_start["receipts"][0]["byte_start"] = False
    assert verify_mod.verify_flat_receipt(
        {"flat_doc_hash": verify_mod.sha256_prefixed(b"abc"), "quote": "a", "byte_start": False, "byte_end": 1},
        b"abc",
    ) is False

    # 3. Spoofed leaf_count, whitespace quote, byte_start outside leaf bounds, cross-doc replay
    bad_lc = json.loads(json.dumps(bundle))
    bad_lc["receipts"][0]["leaf_count"] = 999
    assert verify_bundle(bad_lc)["ok"] is False

    ws_quote = json.loads(json.dumps(bundle))
    ws_quote["receipts"][0]["quote"] = "   "
    assert verify_bundle(ws_quote)["ok"] is False

    shifted_bytes = json.loads(json.dumps(bundle))
    shifted_bytes["receipts"][0]["byte_start"] = 0
    assert verify_bundle(shifted_bytes)["ok"] is False

    # 4. Redacted entry retaining nonce or text fails closed; missing or invalid signature bytes fail closed
    leaky_doc = json.loads(json.dumps(bundle))
    leaky_doc["documents"]["doc1"]["entries"][0]["nonce"] = "stolen-nonce"
    assert verify_bundle(leaky_doc)["ok"] is False

    bad_redacted_flag = json.loads(json.dumps(bundle))
    bad_redacted_flag["documents"]["doc1"]["entries"][1]["redacted"] = "no"
    assert verify_bundle(bad_redacted_flag)["ok"] is False

    short_sig = json.loads(json.dumps(bundle))
    short_sig["certificates"][0]["signature"] = "ed25519:abcd"
    assert verify_bundle(short_sig, require_signature_check=True)["ok"] is False

    # 5. Cross-source receipt replay rejected even for identical text documents
    d1 = RedactableDoc.from_text("Same fact here.", source_id="source_A")
    d2 = RedactableDoc.from_text("Same fact here.", source_id="source_B")
    r_from_a = mint_redactable_receipt(d1, "Same fact here.")
    assert verify_mod.verify_redactable_receipt(r_from_a.as_dict(), d2.as_dict()) is False

    # 6. CLI verifier handles unreadable/invalid JSON gracefully with exit 1
    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    assert verify_mod.main([str(bad_json)]) == 1
    assert verify_mod.main([str(tmp_path / "missing.json")]) == 1


def test_short_id_marker_boundaries_case_normalization_unicode_and_duplicate_leaves():
    vault = CustodyVault.ephemeral()
    vault.remember("Dana joined Leeds in 2020.", subjects=["dana"])
    vault.remember("Ed lives in York.", subjects=["  Ed  "])

    # Erasing short ID 'ed' must NOT false-fail on substring 'ed' inside 'joined' or 'Leeds'!
    cert = vault.erase("ed", probes=["Where did Dana join?", "Where does Ed live?"])
    assert cert.key_destroyed is True
    assert not vault.search("Leeds").abstained

    # Tombstone blocks re-ingest even when casing/whitespace differs
    with pytest.raises(TombstonedSubjectError):
        vault.remember("Ed came back.", subjects=["ED"])
    vault.close()

    # Unicode emojis, accents, Windows CRLF newlines, and duplicate identical sentences in same doc
    multiline = "Priya ordered ☕️ at naïve café.\r\nMarcus runs Leeds shop.\r\nMarcus runs Leeds shop.\r\n"
    raw = multiline.encode("utf-8")
    doc = RedactableDoc.from_text(multiline, source_id="crlf_uni", mode="line")
    rec0 = mint_redactable_receipt(doc, "Marcus runs Leeds shop.", raw_doc=raw, leaf_index=1)
    rec1 = mint_redactable_receipt(doc, "Marcus runs Leeds shop.", raw_doc=raw, leaf_index=2)
    assert rec0.verify_flat(raw) and rec1.verify_flat(raw)
    assert rec0.byte_start < rec1.byte_start

    # Redact only occurrence 1 (index=1): rec0 fails, rec1 survives cleanly!
    partial = doc.redact_indices({0, 1})
    assert not rec0.verify_redactable(partial)
    assert rec1.verify_redactable(partial)


def test_input_validation_and_empty_nonce_or_chunker_guards():
    with pytest.raises(ValueError):
        RedactableDoc.from_text("Hello world.", nonces=[""])
    with pytest.raises(ValueError):
        RedactableDoc.from_text("Hello world.", mode=lambda _t: [(0, 999, "overflow")])  # type: ignore[arg-type]

    ks = MemoryKeyStore()
    with pytest.raises(ValueError):
        ks.get_or_create_subject_key("   ")
    with pytest.raises(ValueError):
        ks.subject_ref("")
    with pytest.raises(ValueError):
        ks.create_record_key("  ", ["alice"])

    vault = CustodyVault.ephemeral()
    with pytest.raises(ValueError):
        vault.register_source("   ", "Some text")
    with pytest.raises(ValueError):
        vault.remember("Hello", subjects=[""])
    with pytest.raises(ValueError):
        vault.erase("   ")

    # Duplicate explicit item_id rejects BEFORE touching adapter!
    vault.remember("Alpha", subjects=["alice"], item_id="fixed_id")
    before_count = len(vault.adapter.all_records())  # type: ignore[attr-defined]
    with pytest.raises(InvariantViolation):
        vault.remember("Duplicate", subjects=["alice"], item_id="fixed_id")
    assert len(vault.adapter.all_records()) == before_count  # type: ignore[attr-defined]
    assert vault.search("Alpha", limit=0).abstained
    assert vault.search("Alpha", limit=-5).abstained
    vault.close()


class _FlakySQLiteAdapter(SQLiteStoreAdapter):
    """Simulates transient network failure on first delete call, then recovers cleanly."""

    def __init__(self, path: str, *, fail_next_delete: bool = True) -> None:
        super().__init__(path)
        self.fail_next_delete = fail_next_delete

    def delete(self, record_ids):
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise ConnectionError("simulated transient vector DB timeout")
        return super().delete(record_ids)


def test_transient_adapter_failure_still_shreds_keys_and_completes_on_restart_retry(tmp_path):
    data_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"
    ext_db = tmp_path / "ext_store.db"

    ext1 = _FlakySQLiteAdapter(str(ext_db), fail_next_delete=True)
    v1 = CustodyVault.open(data_dir, key_dir=key_dir, adapter=ext1)
    v1.register_source("doc", "Priya works at Riverside. Dana works at Northmark.")
    r_priya = v1.remember("Priya works at Riverside.", subjects=["priya"], source_id="doc", quote="Priya works at Riverside.")
    r_dana = v1.remember("Dana works at Northmark.", subjects=["dana"], source_id="doc", quote="Dana works at Northmark.")
    v1.remember("Riverside triage handoff.", subjects=["team"], derived_from=[r_priya.item.item_id, r_dana.item.item_id])

    with pytest.raises(ConnectionError):
        v1.erase("priya", probes=["Where does Priya work?"])

    # Local security guarantees ALREADY fired despite external network error:
    # 1) Subject key & shared record DEK are shredded
    assert v1.keystore.get_subject_key(v1.keystore.subject_ref("priya")) is None
    # 2) RedactableDoc leaf for Priya is ALREADY stripped
    redacted_doc = v1.ledger.get_document(v1.tenant_id, "doc")
    assert redacted_doc is not None and "Priya" not in redacted_doc.text()
    # 3) No certificate was emitted prematurely
    assert v1.ledger.list_certificates(v1.tenant_id) == []
    v1.close()
    ext1.close()

    # Simulate process restart + network recovery: retry sweeps orphaned external IDs and certifies!
    ext2 = _FlakySQLiteAdapter(str(ext_db), fail_next_delete=False)
    v2 = CustodyVault.open(data_dir, key_dir=key_dir, adapter=ext2)
    cert = v2.erase("priya", probes=["Where does Priya work?", "triage handoff"])
    assert cert.key_destroyed is True
    assert len(cert.closure["adapter_record_ids"]) == 2
    assert verify_ed25519_certificate(cert.as_dict())[0] is True
    assert not v2.search("Northmark").abstained
    assert v2.verify_item_receipt(r_dana.item.item_id) is True
    assert v2.search("Riverside").abstained
    v2.close()
    ext2.close()


def test_multithreaded_concurrent_writes_keystore_and_chain_integrity(tmp_path):
    data_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"

    # Concurrent Ed25519Signer / FileKeyStore creation stress
    def _load_keys(i: int) -> str:
        s = Ed25519Signer.load_or_create(key_dir / "ed25519.json")
        ks = FileKeyStore(key_dir / "concurrent_ks.json")
        ks.get_or_create_subject_key(f"init_{i}")
        return s.public_key_hex

    with ThreadPoolExecutor(max_workers=6) as pool:
        pubs = list(pool.map(_load_keys, range(12)))
    assert len(set(pubs)) == 1

    # Concurrent remember / search / register_source on one CustodyVault
    ext = SQLiteStoreAdapter(str(tmp_path / "ext.db"))
    vault = CustodyVault.open(data_dir, key_dir=key_dir, adapter=ext)
    vault.register_source("shared", " ".join(f"Worker{i} handles Shift{i}." for i in range(24)))

    def _worker(idx: int) -> str:
        res = vault.remember(
            f"Worker{idx} handles Shift{idx}.",
            subjects=[f"subj_{idx % 4}"],
            source_id="shared",
            quote=f"Worker{idx} handles Shift{idx}.",
            metadata={"worker": idx},
        )
        _ = vault.search(f"Shift{idx}")
        return res.item.item_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        item_ids = list(pool.map(_worker, range(24)))
    assert len(set(item_ids)) == 24

    vault.erase("subj_0", probes=["Worker0 handles Shift0?"])
    ok, msg = vault.ledger.verify_chain(vault.tenant_id)
    assert ok is True, msg
    assert verify_bundle(vault.export_bundle(), require_signature_check=True)["ok"] is True
    vault.close()
    ext.close()


def test_minimal_duck_typed_adapters_and_unicode_search(tmp_path):
    class MinimalMem0:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}

        def add(self, text: str, user_id: str | None = None):
            mid = f"m_{len(self.rows)}"
            self.rows[mid] = {"id": mid, "memory": text, "user_id": user_id}
            return {"id": mid}

        def search(self, query: str):
            return [r for r in self.rows.values() if query.lower() in r["memory"].lower()]

        def delete(self, mid: str):
            self.rows.pop(mid, None)

    class MinimalVec:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def add_texts(self, texts: list[str]):
            self.items.extend(texts)

        def similarity_search(self, query: str, k: int = 4):
            class _D:
                def __init__(self, txt: str) -> None:
                    self.page_content = txt
                    self.metadata = {}

            return [_D(t) for t in self.items if query.lower() in t.lower()][:k]

        def delete(self, ids: list[str]):
            pass

    v_m0 = CustodyVault.ephemeral(adapter=Mem0Adapter(MinimalMem0()))
    r1 = v_m0.remember("José visited Zürich café.", subjects=["jose"])
    assert len(r1.adapter_record_ids) == 1
    v_m0.erase("jose", probes=["Zürich"])
    v_m0.close()

    # Minimal vector store without ids/filter parameters does not crash with TypeError
    v_vec = CustodyVault.ephemeral(adapter=VectorStoreAdapter(MinimalVec()))
    v_vec.remember("José works on graphs.", subjects=["jose"])
    assert len(v_vec.adapter.search("José", limit=0)) == 0
    v_vec.close()

    # SQLiteStoreAdapter Unicode search & exact delete count across SQLite versions
    sql = SQLiteStoreAdapter(str(tmp_path / "uni.db"))
    ids = sql.upsert(item_id="i1", text="José built a naïve café.", subjects=("jose",), metadata={})
    assert len(sql.search("Zürich")) == 0
    assert len(sql.search("naïve café")) == 1
    assert sql.delete(ids) == 1
    assert sql.delete(ids) == 0
    sql.close()


def test_cli_replay_meta_version_and_clean_error_handling(tmp_path, capsys):
    vault_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"
    base = ["--vault", str(vault_dir), "--key-dir", str(key_dir)]

    with pytest.raises(SystemExit) as ver_exit:
        cli.main(["--version"])
    assert ver_exit.value.code == 0
    assert "ember" in capsys.readouterr().out

    assert cli.main([*base, "remember", "Priya runs clinic.", "--subject", "priya", "--meta", "email=priya@clinic.org"]) == 0
    capsys.readouterr()

    # Verify CLI --meta is encrypted inside raw SQLite ledger
    conn = sqlite3.connect(vault_dir / "ledger.db")
    blob = " ".join(r[0] for r in conn.execute("SELECT payload_json FROM ledger_events").fetchall())
    conn.close()
    assert "priya@clinic.org" not in blob

    assert cli.main([*base, "replay"]) == 0
    replay_out = json.loads(capsys.readouterr().out)
    assert replay_out["chain_ok"] is True
    assert replay_out["live_item_count"] == 1

    assert cli.main([*base, "erase", "priya", "--probe", "clinic"]) == 0
    capsys.readouterr()

    # Tombstoned re-ingest & missing verify file fail cleanly with exit code 1 + JSON stderr (no traceback)
    assert cli.main([*base, "remember", "Priya again", "--subject", "priya"]) == 1
    err_out = json.loads(capsys.readouterr().err)
    assert err_out["error"] == "TombstonedSubjectError"

    assert cli.main([*base, "add-source", "bad_src"]) == 1
    assert cli.main(["verify", str(tmp_path / "missing_bundle.json")]) == 1
    capsys.readouterr()
