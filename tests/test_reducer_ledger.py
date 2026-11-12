"""Hash-chained SQLiteLedger durability, idempotency, tamper detection, and zero-PII log guarantees."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ember.ledger import IdempotencyConflict, LedgerIntegrityError, SQLiteLedger
from ember.reducer import ConcurrentModificationError, InvariantViolation, TombstonedSubjectError
from ember.vault import CustodyVault


def test_sqlite_ledger_idempotency_and_conflict_rejection():
    ledger = SQLiteLedger(":memory:")
    try:
        ev1, idem1, _ = ledger.append(
            tenant_id="t1",
            event_id="evt_1",
            kind="ADD_SOURCE",
            committed_at="2026-11-10T00:00:00Z",
            payload={
                "source": {
                    "source_id": "s1",
                    "merkle_root": "a" * 64,
                    "flat_doc_hash": "sha256:" + "b" * 64,
                    "byte_length": 12,
                    "leaf_count": 1,
                    "chunk_mode": "sentence",
                    "ingested_at": "2026-11-10T00:00:00Z",
                }
            },
        )
        assert not idem1 and ev1.seq == 0

        ev2, idem2, _ = ledger.append(
            tenant_id="t1",
            event_id="evt_1",
            kind="ADD_SOURCE",
            committed_at="2026-11-10T00:00:00Z",
            payload=ev1.payload,
        )
        assert idem2 and ev2.event_hash == ev1.event_hash

        with pytest.raises(IdempotencyConflict):
            ledger.append(
                tenant_id="t1",
                event_id="evt_1",
                kind="ADD_SOURCE",
                committed_at="2026-11-10T00:00:00Z",
                payload={"source": {**ev1.payload["source"], "byte_length": 999}},
            )
    finally:
        ledger.close()


def test_sqlite_ledger_detects_tampering_deletion_and_reordering(tmp_path):
    db_path = tmp_path / "ledger.db"
    vault = CustodyVault.open(tmp_path, key_dir=tmp_path.parent / "sep_keys")
    assert vault.keys_separated_from_ledger is True
    vault.register_source("s1", "Priya joined Acme in 2019. Dana works in Seattle.")
    vault.remember(
        "Priya joined Acme in 2019.",
        subjects=["priya"],
        source_id="s1",
        quote="Priya joined Acme in 2019.",
        metadata={"user_id": "priya", "email": "priya.raman@acme.example"},
    )
    vault.remember(
        "Dana works in Seattle.",
        subjects=["dana"],
        source_id="s1",
        quote="Dana works in Seattle.",
        metadata={"user_id": "dana", "email": "dana.kim@acme.example"},
    )
    vault.close()

    # Inspect raw DB to confirm ZERO plaintext claim/quote/subject/metadata strings sit inside ledger_events
    conn = sqlite3.connect(db_path)
    raw_rows = conn.execute("SELECT payload_json FROM ledger_events").fetchall()
    blob = " ".join(r[0] for r in raw_rows)
    assert "Priya" not in blob
    assert "priya.raman@acme.example" not in blob
    assert "Dana" not in blob
    assert "dana.kim@acme.example" not in blob
    assert "Seattle" not in blob
    assert "Acme" not in blob

    # Tamper payload of seq=1 directly in SQLite -> chain check must reject
    row1 = json.loads(raw_rows[1][0])
    assert "metadata" not in row1["item"]
    assert row1["item"]["metadata_ciphertext"].startswith("enc2:")
    row1["item"]["tampered"] = True
    conn.execute("UPDATE ledger_events SET payload_json = ? WHERE seq = 1", (json.dumps(row1),))
    conn.commit()
    conn.close()

    with pytest.raises(LedgerIntegrityError):
        CustodyVault.open(tmp_path, key_dir=tmp_path.parent / "sep_keys")


def test_crash_replay_reconstructs_byte_identical_state(tmp_path):
    key_dir = tmp_path / "external_keys"
    data_dir = tmp_path / "vault_data"
    v1 = CustodyVault.open(data_dir, key_dir=key_dir)
    v1.register_source("src", "Alice runs SRE. Bob runs ML.")
    r1 = v1.remember("Alice runs SRE.", subjects=["alice"], source_id="src", quote="Alice runs SRE.")
    v1.remember("Bob runs ML.", subjects=["bob"], source_id="src", quote="Bob runs ML.")
    v1.remember("SRE & ML pair on incidents.", subjects=["ops"], derived_from=[r1.item.item_id])
    v1.erase("alice", probes=["Who runs SRE?"])
    digest_live = v1.state.digest()
    v1.close()

    v2 = CustodyVault.open(data_dir, key_dir=key_dir)
    assert v2.state.digest() == digest_live
    assert v2.ledger.verify_chain(v2.tenant_id)[0] is True
    v2.close()


def test_ledger_rejects_stale_state_occ_and_redacted_leaf_backfill(tmp_path):
    v_writer = CustodyVault.open(tmp_path / "vault", key_dir=tmp_path / "keys")
    v_reader = CustodyVault.open(tmp_path / "vault", key_dir=tmp_path / "keys")
    try:
        v_writer.register_source("s1", "Priya lives in Denver. Dana lives in Austin.")
        v_writer.remember("Priya lives in Denver.", subjects=["priya"], source_id="s1", quote="Priya lives in Denver.")
        v_writer.remember("Dana lives in Austin.", subjects=["dana"], source_id="s1", quote="Dana lives in Austin.")

        # Refresh v_reader pre-erasure, then capture stale snapshot before v_writer erases Priya
        v_reader.search("Denver")
        stale_state = v_reader.state
        stale_priya_item = next(it for it in stale_state.items.values() if "priya" in it.subjects)
        stale_payload = {
            "item": {
                **stale_priya_item.ledger_payload(),
                "item_id": "item_stale_injection",
                "receipt_meta": {
                    "source_id": "s1",
                    "flat_doc_hash": stale_state.sources["s1"].flat_doc_hash,
                    "byte_start": 0,
                    "byte_end": 22,
                    "merkle_root": stale_state.sources["s1"].merkle_root,
                    "leaf_index": 0,
                    "leaf_count": 2,
                    "proof": [],
                },
            }
        }

        v_writer.erase("priya", probes=["Denver"])

        # 1) Passing expected_seq from stale reader must raise ConcurrentModificationError
        with pytest.raises(ConcurrentModificationError):
            v_writer.ledger.append(
                tenant_id=v_writer.tenant_id,
                event_id="evt_occ_stale",
                kind="ADD_ITEM",
                committed_at="2026-11-12T00:00:00Z",
                payload=stale_payload,
                current_state=stale_state,
                keystore=v_writer.keystore,
                validate=True,
                expected_seq=stale_state.version,
            )

        # 2) Passing stale current_state WITHOUT expected_seq must still ignore stale version and enforce tombstones
        with pytest.raises(TombstonedSubjectError):
            v_writer.ledger.append(
                tenant_id=v_writer.tenant_id,
                event_id="evt_stale_bypass_attempt",
                kind="ADD_ITEM",
                committed_at="2026-11-12T00:00:00Z",
                payload=stale_payload,
                current_state=stale_state,
                keystore=v_writer.keystore,
                validate=True,
            )

        # 3) Even an untagged item citing a redacted leaf (s1[0]) directly at the ledger layer is rejected
        untagged_redacted_leaf = {
            "item": {
                **stale_payload["item"],
                "item_id": "item_redacted_leaf_cite",
                "subject_tokens": [],
            }
        }
        with pytest.raises(InvariantViolation):
            v_writer.ledger.append(
                tenant_id=v_writer.tenant_id,
                event_id="evt_redacted_leaf_cite",
                kind="ADD_ITEM",
                committed_at="2026-11-12T00:00:00Z",
                payload=untagged_redacted_leaf,
                current_state=v_writer.state,
                keystore=v_writer.keystore,
                validate=True,
            )
    finally:
        v_reader.close()
        v_writer.close()
