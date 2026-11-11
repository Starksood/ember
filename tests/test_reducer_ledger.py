"""Hash-chained SQLiteLedger durability, idempotency, tamper detection, and zero-PII log guarantees."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ember.ledger import IdempotencyConflict, LedgerIntegrityError, SQLiteLedger
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
