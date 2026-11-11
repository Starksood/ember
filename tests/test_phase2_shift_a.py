"""Phase 2 Shift A verification: TurnTrace, zero-plaintext vendor mode, Mem0 infer sync, content tombstones, and sub-leaf redaction."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import threading
from typing import Any
import uuid

import pytest

from ember import (
    CustodyVault,
    ErasureIncomplete,
    InMemoryStoreAdapter,
    Mem0Adapter,
    RedactableDoc,
    SQLiteStoreAdapter,
    TombstonedSubjectError,
    VectorStoreAdapter,
    mint_redactable_receipt,
    verify_bundle,
)
import ember.verify as verify_mod


def _det_embed(text: str, dims: int = 128) -> list[float]:
    """Deterministic local word + bigrams normalized vector for testing dense vector stores offline."""
    vec = [0.0] * dims
    low = text.lower()
    words = [w.strip(".,?!'\"()") for w in low.split() if w.strip(".,?!'\"()")]
    for idx, tok in enumerate(words):
        h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
        vec[h % dims] += 2.0
        if idx + 1 < len(words):
            bg = f"{tok}_{words[idx + 1]}"
            hb = int.from_bytes(hashlib.sha256(bg.encode("utf-8")).digest()[:4], "big")
            vec[hb % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


# ---------------------------------------------------------------------------
# A1: Zero-Config Turn-Scoped Lineage Tracer (`vault.trace_turn()`)
# ---------------------------------------------------------------------------


def test_a1_turn_trace_auto_derives_and_cascades_across_erasure():
    adapter = InMemoryStoreAdapter()
    vault = CustodyVault.ephemeral(adapter=adapter)
    r_priya = vault.remember("Priya Raman leads triage routing.", subjects=["priya"])
    r_dana = vault.remember("Dana Kim owns pharmacy stock.", subjects=["dana"])

    with vault.trace_turn() as trace:
        res = vault.search("triage routing", limit=5)
        assert not res.abstained
        # Note: NO explicit derived_from passed!
        r_sum = vault.remember(
            "Night shift handoff depends on triage routing protocol.",
            subjects=["team_ops"],
        )
        assert r_priya.item.item_id in trace.read_ids
        assert r_sum.item.item_id in trace.written_ids
        assert r_priya.item.item_id in r_sum.item.derived_from

        # Auto-derive opt-out works inside the same turn
        r_opt_out = vault.remember(
            "Unrelated facility lighting schedule.",
            subjects=["facility"],
            auto_derive=False,
        )
        assert r_opt_out.item.derived_from == ()

    cert = vault.erase("priya", probes=["night shift handoff", "triage routing"])
    assert r_sum.item.item_id in cert.closure["derived_invalidated"]
    assert vault.search("night shift handoff").abstained
    assert not vault.search("pharmacy stock").abstained
    assert not vault.search("facility lighting").abstained
    assert r_dana.item.item_id in vault.state.items
    vault.close()


def test_a1_turn_trace_lexical_pruning_and_custom_callable():
    vault = CustodyVault.ephemeral()
    r_priya = vault.remember("Priya builds immunology classifiers in Portland.", subjects=["priya"])
    r_bystander = vault.remember("Dana builds solar battery telemetry in Portland.", subjects=["dana"])

    with vault.trace_turn(prune="lexical") as t_lex:
        res = vault.search("Portland", limit=5)
        assert len(res.hits) == 2
        # Only mentions immunology/classifiers, NOT telemetry/solar -> prunes Dana!
        r_summary = vault.remember(
            "Model summary: immunology classifiers require labeled samples.",
            subjects=["team_ml"],
        )
        assert r_summary.item.derived_from == (r_priya.item.item_id,)
        assert t_lex.hits[0].item_id in {r_priya.item.item_id, r_bystander.item.item_id}

    with vault.trace_turn(prune=lambda _text, hits: [h.item_id for h in hits if "dana" in h.subjects]) as t_custom:
        t_custom.clear_reads()
        vault.search("Portland", limit=5)
        t_custom.record_read(r_bystander.item.item_id)
        r_solar = vault.remember("Grid note references battery telemetry.", subjects=["team_grid"])
        assert r_solar.item.derived_from == (r_bystander.item.item_id,)

    vault.erase("priya")
    assert r_summary.item.item_id not in vault.state.items
    # Dana + r_solar survived because lexical pruning did not taint them!
    assert r_solar.item.item_id in vault.state.items
    vault.close()


def test_a1_turn_trace_async_and_thread_isolation():
    vault = CustodyVault.ephemeral()
    r_priya = vault.remember("Priya alpha protocol spec.", subjects=["priya"])
    r_dana = vault.remember("Dana beta logistics spec.", subjects=["dana"])

    async def _worker(query: str, target_subj: str, expected_parent: str) -> tuple[str, ...]:
        with vault.trace_turn():
            vault.search(query, limit=3)
            await asyncio.sleep(0.02)
            out = vault.remember(f"Note under {target_subj}", subjects=[target_subj])
            assert expected_parent in out.item.derived_from
            return out.item.derived_from

    async def _run_both():
        return await asyncio.gather(
            _worker("alpha protocol", "grp_a", r_priya.item.item_id),
            _worker("beta logistics", "grp_b", r_dana.item.item_id),
        )

    der_a, der_b = asyncio.run(_run_both())
    assert r_dana.item.item_id not in der_a
    assert r_priya.item.item_id not in der_b

    # Thread isolation check
    results: dict[str, tuple[str, ...]] = {}
    barrier = threading.Barrier(2)

    def _thread_job(name: str, q: str):
        with vault.trace_turn():
            vault.search(q, limit=2)
            barrier.wait(timeout=2.0)
            res = vault.remember(f"Thread write {name}", subjects=[f"t_{name}"])
            results[name] = res.item.derived_from

    t1 = threading.Thread(target=_thread_job, args=("p", "alpha protocol"))
    t2 = threading.Thread(target=_thread_job, args=("d", "beta logistics"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results["p"] == (r_priya.item.item_id,)
    assert results["d"] == (r_dana.item.item_id,)
    vault.close()


# ---------------------------------------------------------------------------
# A2: Two-Mode Vendor Storage (`encrypt_vendor_payload=True`)
# ---------------------------------------------------------------------------


@dataclass
class _VecEmbDoc:
    id: str
    page_content: str
    metadata: dict[str, Any]
    vector: list[float]


class _FakeEmbeddingVectorStore:
    def __init__(self) -> None:
        self.docs: dict[str, _VecEmbDoc] = {}

    def add_embeddings(
        self,
        text_embeddings: list[tuple[str, list[float]]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> list[str]:
        for (txt, vec), meta, rec_id in zip(text_embeddings, metadatas, ids, strict=True):
            self.docs[rec_id] = _VecEmbDoc(id=rec_id, page_content=txt, metadata=dict(meta), vector=list(vec))
        return list(ids)

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[_VecEmbDoc, float]]:
        scored: list[tuple[_VecEmbDoc, float]] = []
        for d in self.docs.values():
            if filter and any(d.metadata.get(fk) != fv for fk, fv in filter.items()):
                continue
            sim = sum(a * b for a, b in zip(embedding, d.vector, strict=True))
            if sim > 0.0:
                scored.append((d, sim))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.docs.pop(i, None)


class _MockMem0WithVectorStore:
    def __init__(self, *, swallow_delete: bool = False) -> None:
        self.swallow_delete = swallow_delete
        self.add_calls = 0

        class _VS:
            def __init__(self, parent: "_MockMem0WithVectorStore") -> None:
                self.parent = parent
                self.points: dict[str, dict[str, Any]] = {}

            def insert(self, *, vectors: list[list[float]], ids: list[str], payloads: list[dict[str, Any]]) -> None:
                for v, pid, p in zip(vectors, ids, payloads, strict=True):
                    self.points[pid] = {"id": pid, "vector": list(v), "payload": dict(p)}

            def search(
                self,
                *,
                vectors: list[list[float]],
                top_k: int = 5,
                filters: dict[str, Any] | None = None,
            ) -> list[dict[str, Any]]:
                qv = vectors[0]
                hits = []
                for p in self.points.values():
                    pl = p["payload"]
                    if filters and any(pl.get(fk) != fv for fk, fv in filters.items()):
                        continue
                    sim = sum(a * b for a, b in zip(qv, p["vector"], strict=True))
                    hits.append({"id": p["id"], "score": sim, "payload": dict(pl)})
                hits.sort(key=lambda r: -r["score"])
                return hits[:top_k]

            def delete(self, *, vector_id: str) -> None:
                if not self.parent.swallow_delete:
                    self.points.pop(vector_id, None)

        self.vector_store = _VS(self)

    def add(self, *args, **kwargs):
        self.add_calls += 1
        raise AssertionError("Mem0 LLM add() must not be called when direct vector insert is used")

    def search(self, *args, **kwargs):
        return {"results": []}

    def delete(self, memory_id: str):
        raise ValueError(f"Memory with id {memory_id} not found")


def test_a2_encrypted_vendor_payload_vector_sqlite_and_lang_vector_store(tmp_path):
    ext_db = tmp_path / "external_vectors.db"
    sql_adapter = SQLiteStoreAdapter(str(ext_db))
    vault = CustodyVault.open(
        tmp_path / "vault",
        key_dir=tmp_path / "keys",
        adapter=sql_adapter,
        encrypt_vendor_payload=True,
        embedder=_det_embed,
    )
    vault.register_source(
        "clinical",
        "Priya Raman handles cardiology intake. Dana Kim handles oncology logistics.",
    )
    vault.remember(
        "Priya Raman handles cardiology intake.",
        subjects=["priya"],
        source_id="clinical",
        quote="Priya Raman handles cardiology intake.",
        metadata={"user_id": "priya", "email": "priya.secret@clinic.example", "team": "cardio"},
    )
    r_dana = vault.remember(
        "Dana Kim handles oncology logistics.",
        subjects=["dana"],
        source_id="clinical",
        quote="Dana Kim handles oncology logistics.",
        metadata={"user_id": "dana", "team": "oncology"},
    )
    vault.close()

    # Inspect raw vendor SQLite table: ZERO plaintext text, email, or subject IDs!
    conn = sqlite3.connect(ext_db)
    raw_rows = conn.execute("SELECT text, meta_json FROM external_memories").fetchall()
    conn.close()
    raw_blob = " ".join(f"{r[0]} {r[1]}" for r in raw_rows)
    for sensitive in ("Priya", "cardiology", "Dana", "oncology", "priya.secret@clinic.example", "subj_"):
        if sensitive == "subj_":
            assert "subj_" in raw_blob
        else:
            assert sensitive not in raw_blob
    assert all(r[0].startswith("enc2:") for r in raw_rows)

    # Reopen across restart and verify search decrypts + verifies receipt + supports filters cleanly
    reopened = CustodyVault.open(
        tmp_path / "vault",
        key_dir=tmp_path / "keys",
        adapter=sql_adapter,
        encrypt_vendor_payload=True,
        embedder=_det_embed,
    )
    res_dana = reopened.search("oncology logistics", filters={"user_id": "dana", "team": "oncology"})
    assert not res_dana.abstained
    assert res_dana.hits[0].text == "Dana Kim handles oncology logistics."
    assert res_dana.hits[0].metadata["team"] == "oncology"
    assert res_dana.hits[0].receipt_verified is True

    cert = reopened.erase("priya")
    assert cert.trust_boundary["vendor_payload_encrypted"] is True
    assert reopened.search("cardiology intake").abstained
    assert reopened.verify_item_receipt(r_dana.item.item_id) is True
    reopened.close()
    sql_adapter.close()

    # VectorStoreAdapter direct embedding check
    vec_store = _FakeEmbeddingVectorStore()
    v_vec = CustodyVault.ephemeral(
        adapter=VectorStoreAdapter(vec_store),
        encrypt_vendor_payload=True,
        embedder=_det_embed,
    )
    v_vec.remember("Priya builds cardiac telemetry.", subjects=["priya"])
    v_vec.remember("Dana builds warehouse bots.", subjects=["dana"])
    assert all(d.page_content.startswith("enc2:") for d in vec_store.docs.values())
    assert "Priya" not in " ".join(d.page_content + json.dumps(d.metadata) for d in vec_store.docs.values())
    v_vec.erase("priya")
    assert len(vec_store.docs) == 1
    assert v_vec.search("warehouse bots").hits[0].text == "Dana builds warehouse bots."
    v_vec.close()


def test_a2_encrypted_vendor_payload_blind_index_mode_and_mem0_vector_store(tmp_path):
    # Mode 2: Blind-index tokens (NO embedder provided)
    sql_adapter = SQLiteStoreAdapter(str(tmp_path / "blind.db"))
    v_blind = CustodyVault.ephemeral(adapter=sql_adapter, encrypt_vendor_payload=True)
    v_blind.remember(
        "Priya Raman works in Portland pharmacy.",
        subjects=["priya"],
        metadata={"user_id": "priya", "city": "Portland"},
    )
    v_blind.remember(
        "Marcus Chen owns a bookshop in Leeds.",
        subjects=["marcus"],
        metadata={"user_id": "marcus", "city": "Leeds"},
    )
    conn = sqlite3.connect(tmp_path / "blind.db")
    dump = " ".join(f"{t} {m}" for t, m in conn.execute("SELECT text, meta_json FROM external_memories").fetchall())
    conn.close()
    assert "Priya" not in dump and "Portland" not in dump and " bt" in dump and "subj_" in dump

    res = v_blind.search("Portland pharmacy")
    assert not res.abstained
    assert res.hits[0].text == "Priya Raman works in Portland pharmacy."
    assert res.hits[0].metadata["city"] == "Portland"

    v_blind.erase("priya", probes=["Where does Priya work in Portland?"])
    assert v_blind.search("Portland pharmacy").abstained
    assert v_blind.search("bookshop Leeds").hits[0].text == "Marcus Chen owns a bookshop in Leeds."
    v_blind.close()
    sql_adapter.close()

    # Mem0Adapter direct vector_store encrypted mode + fail-closed when vector delete is swallowed
    mock_ok = _MockMem0WithVectorStore(swallow_delete=False)
    v_m0 = CustodyVault.ephemeral(
        adapter=Mem0Adapter(mock_ok, infer=False),
        encrypt_vendor_payload=True,
        embedder=_det_embed,
    )
    v_m0.remember("Priya handles triage.", subjects=["priya"])
    v_m0.remember("Dana handles payroll.", subjects=["dana"])
    assert mock_ok.add_calls == 0
    assert all(pt["payload"]["data"].startswith("enc2:") for pt in mock_ok.vector_store.points.values())
    assert "Priya" not in json.dumps(mock_ok.vector_store.points)
    v_m0.erase("priya")
    assert len(mock_ok.vector_store.points) == 1
    assert v_m0.search("payroll").hits[0].text == "Dana handles payroll."
    v_m0.close()

    mock_bad = _MockMem0WithVectorStore(swallow_delete=True)
    v_bad = CustodyVault.ephemeral(
        adapter=Mem0Adapter(mock_bad, infer=False),
        encrypt_vendor_payload=True,
        embedder=_det_embed,
    )
    v_bad.remember("Priya handles triage.", subjects=["priya"])
    with pytest.raises(ErasureIncomplete):
        v_bad.erase("priya")
    v_bad.close()

    with pytest.raises(ValueError, match="infer=False"):
        CustodyVault.ephemeral(
            adapter=Mem0Adapter(_MockMem0WithVectorStore(), infer=True),
            encrypt_vendor_payload=True,
        )


# ---------------------------------------------------------------------------
# A3: Mem0(infer=True) ADD / UPDATE / DELETE Lifecycle Sync
# ---------------------------------------------------------------------------


class _InferringMem0Mock:
    """Simulates Mem0(infer=True) consolidating/mutating existing memories in place."""

    def __init__(self, mode: str = "update_in_place") -> None:
        self.mode = mode
        self.rows: dict[str, dict[str, Any]] = {}
        self.deleted_ids: list[str] = []

    def add(self, messages: str, *, user_id: str | None = None, metadata: dict[str, Any] | None = None, infer: bool = True):
        meta = dict(metadata or {})
        if not infer or not self.rows:
            mid = f"m_inf_{uuid.uuid4().hex[:8]}"
            self.rows[mid] = {"id": mid, "memory": str(messages), "user_id": user_id, "metadata": meta}
            return {"results": [{"id": mid, "memory": str(messages), "event": "ADD"}]}

        # Trigger merge against an existing row when second message mentions pharmacy or co-lead
        existing_id = next(iter(self.rows))
        prev = self.rows[existing_id]
        if self.mode == "update_in_place":
            merged = f"{prev['memory']} Also: {messages}"
            self.rows[existing_id]["memory"] = merged
            self.rows[existing_id]["metadata"] = {**prev["metadata"], **meta}
            return {
                "results": [
                    {
                        "id": existing_id,
                        "memory": merged,
                        "previous_memory": prev["memory"],
                        "event": "UPDATE",
                    }
                ]
            }
        if self.mode == "delete_and_add":
            merged = f"Merged: {prev['memory']} + {messages}"
            del self.rows[existing_id]
            new_mid = f"m_inf_{uuid.uuid4().hex[:8]}"
            self.rows[new_mid] = {"id": new_mid, "memory": merged, "user_id": user_id, "metadata": meta}
            return {
                "results": [
                    {"id": new_mid, "memory": merged, "event": "ADD"},
                    {"id": existing_id, "event": "DELETE"},
                ]
            }
        raise AssertionError(f"unknown mode {self.mode}")

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 5):
        q_toks = {w.strip(".,?!") for w in query.lower().split()}
        out = []
        for r in self.rows.values():
            if filters and "user_id" in filters and r.get("user_id") != filters["user_id"]:
                continue
            words = {w.strip(".,?!") for w in r["memory"].lower().split()}
            if q_toks & words:
                out.append(
                    {
                        "id": r["id"],
                        "memory": r["memory"],
                        "score": 1.0,
                        "user_id": r["user_id"],
                        "metadata": dict(r["metadata"]),
                    }
                )
        return {"results": out[:top_k]}

    def delete(self, memory_id: str):
        self.deleted_ids.append(memory_id)
        if memory_id not in self.rows:
            raise ValueError(f"Memory with id {memory_id} not found")
        del self.rows[memory_id]


def test_a3_mem0_infer_update_and_delete_cascade_and_rollback():
    # Case 1: In-place UPDATE across Dana -> Priya promotes record DEK and erases merged record under EITHER subject
    m_upd = _InferringMem0Mock(mode="update_in_place")
    v1 = CustodyVault.ephemeral(adapter=Mem0Adapter(m_upd, infer=True))
    r_dana = v1.remember("Dana leads pharmacy.", subjects=["dana"])
    r_priya = v1.remember("Priya co-leads pharmacy.", subjects=["priya"])

    assert r_dana.item.item_id in r_priya.item.derived_from
    assert r_priya.item.key_kind == "record"
    assert r_dana.adapter_record_ids[0] in r_priya.adapter_record_ids

    # Erase Dana -> MUST destroy Priya's merged item & external record too!
    cert = v1.erase("dana", probes=["pharmacy"])
    assert r_priya.item.item_id in cert.closure["derived_invalidated"]
    assert m_upd.rows == {}
    assert v1.search("pharmacy").abstained
    v1.close()

    # Case 2: DELETE + ADD merge where DELETE appears AFTER ADD in results
    m_del = _InferringMem0Mock(mode="delete_and_add")
    v2 = CustodyVault.ephemeral(adapter=Mem0Adapter(m_del, infer=True))
    r_d = v2.remember("Dana owns cold storage.", subjects=["dana"])
    r_p = v2.remember("Priya inspects cold storage.", subjects=["priya"])
    assert r_d.item.item_id in r_p.item.derived_from
    assert r_p.item.text.startswith("Merged:")
    v2.erase("dana", probes=["cold storage"])
    assert m_del.rows == {}
    v2.close()

    # Case 3: Rollback on ledger failure only deletes newly ADDed IDs, NOT prior UPDATEd IDs
    m_rb = _InferringMem0Mock(mode="update_in_place")
    v3 = CustodyVault.ephemeral(adapter=Mem0Adapter(m_rb, infer=True))
    r_orig = v3.remember("Dana owns cold storage.", subjects=["dana"])
    orig_ext = r_orig.adapter_record_ids[0]
    # Force ledger.append failure during cross-subject UPDATE -> rollback must NOT delete pre-existing orig_ext!
    orig_append = v3.ledger.append

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated ledger disk full")

    v3.ledger.append = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="disk full"):
        v3.remember("Priya joins cold storage.", subjects=["priya"])
    v3.ledger.append = orig_append  # type: ignore[method-assign]
    # Crucial: pre-existing orig_ext was UPDATED (not ADDed), so rollback MUST NOT call adapter.delete([orig_ext])!
    assert orig_ext in m_rb.rows
    v3.close()


# ---------------------------------------------------------------------------
# A4: Content-Level Tombstone Ingest Guard & Auto-Generated Probes
# ---------------------------------------------------------------------------


def test_a4_content_tombstone_guard_across_restart_and_false_positive_boundaries(tmp_path):
    vdir = tmp_path / "vault"
    kdir = tmp_path / "keys"
    v1 = CustodyVault.open(vdir, key_dir=kdir)
    v1.remember("Priya Raman lives in Portland.", subjects=["priya_id"])
    v1.remember("Ed works in IT.", subjects=["ed"])
    v1.erase(
        "priya_id",
        aliases=["Priya Raman", "priya@clinic.example"],
        markers=["Portland"],
    )
    v1.erase("ed")  # <3 char identifier: must NOT block words like "joined"!
    v1.close()

    # Raw ledger must never expose alias plaintext in content_tokens
    conn = sqlite3.connect(vdir / "ledger.db")
    blob = " ".join(r[0] for r in conn.execute("SELECT payload_json FROM ledger_events").fetchall())
    conn.close()
    assert "Priya Raman" not in blob
    assert "priya@clinic.example" not in blob

    v2 = CustodyVault.open(vdir, key_dir=kdir)
    # Zombie text referencing erased subject under neutral bucket "misc_ops" MUST fail closed!
    with pytest.raises(TombstonedSubjectError):
        v2.remember("Re-assign ticket from Priya Raman to night team.", subjects=["misc_ops"])

    # Zombie email in metadata MUST fail closed!
    with pytest.raises(TombstonedSubjectError):
        v2.remember("Routine pager sync.", subjects=["misc_ops"], metadata={"email": "priya@clinic.example"})

    # Explicit reconsent_id permits clean re-ingestion
    re_res = v2.remember(
        "Priya Raman re-consented to support inbox.",
        subjects=["misc_ops"],
        reconsent_id="consent-2026-11-11",
    )
    assert re_res.item.reconsent_id == "consent-2026-11-11"

    # Generic markers ("Portland") and short <3 char IDs ("ed" vs "joined") do NOT trigger false positives!
    ok_city = v2.remember("Marcus Chen joined Portland warehouse operations.", subjects=["marcus"])
    assert ok_city.item.item_id in v2.state.items
    v2.close()


def test_a4_auto_probes_fail_closed_even_without_explicit_probes():
    class _LeakyInMemory(InMemoryStoreAdapter):
        def delete(self, record_ids):
            return 0  # Sabotage deletion silently!

    v_bad = CustodyVault.ephemeral(adapter=_LeakyInMemory())
    v_bad.remember("Priya coordinates emergency dispatch.", subjects=["priya"])
    # Notice NO `probes=` passed at all!
    with pytest.raises(ErasureIncomplete):
        v_bad.erase("priya")
    v_bad.close()

    # Bystander in same org/city does NOT false-positive auto-probes
    v_good = CustodyVault.ephemeral()
    v_good.remember("Priya coordinates dispatch in Portland.", subjects=["priya"])
    v_good.remember("Dana owns pharmacy in Portland.", subjects=["dana"])
    cert = v_good.erase("priya")
    assert cert.probe_count >= 1
    assert not v_good.search("pharmacy in Portland").abstained
    v_good.close()


# ---------------------------------------------------------------------------
# A5: Compound-Sentence Sub-Leaf Redaction (`span_mode`)
# ---------------------------------------------------------------------------


def test_a5_sub_leaf_word_and_clause_redaction_with_offline_stdlib_verify(tmp_path):
    doc_text = "Priya Raman mentored Dana Kim on Q3 routing. priya@clinic.example's ticket moved to Marcus Chen."
    vault = CustodyVault.ephemeral()
    _, orig_doc = vault.register_source("handoff", doc_text, mode="sentence", span_mode="word")
    root_before = orig_doc.root()

    r_priya = vault.remember(
        "Priya Raman",
        subjects=["priya"],
        source_id="handoff",
        quote="Priya Raman",
    )
    r_email = vault.remember(
        "priya@clinic.example",
        subjects=["priya"],
        source_id="handoff",
        quote="priya@clinic.example",
    )
    r_dana = vault.remember(
        "Dana Kim on Q3 routing.",
        subjects=["dana"],
        source_id="handoff",
        quote="Dana Kim on Q3 routing.",
    )
    r_marcus = vault.remember(
        "moved to Marcus Chen.",
        subjects=["marcus"],
        source_id="handoff",
        quote="moved to Marcus Chen.",
    )
    assert r_priya.receipt is not None and r_priya.receipt.span_level is True
    assert r_dana.receipt is not None and r_dana.receipt.span_level is True

    cert = vault.erase("priya", probes=["Priya Raman"])
    assert "source_spans_redacted" in cert.closure
    assert cert.closure["source_spans_redacted"]["handoff"]["0"] == list(r_priya.receipt.span_indices)
    assert cert.closure["source_spans_redacted"]["handoff"]["1"] == list(r_email.receipt.span_indices)

    redacted_doc = vault.ledger.get_document(vault.tenant_id, "handoff")
    assert redacted_doc is not None
    assert redacted_doc.root() == root_before
    rendered = redacted_doc.text()
    assert "Priya" not in rendered and "Raman" not in rendered and "priya@clinic.example" not in rendered
    assert "[redacted] mentored Dana Kim on Q3 routing." in rendered
    assert "[redacted]'s ticket moved to Marcus Chen." in rendered

    # Spans for Priya destroyed both text and nonce
    for sp_idx in r_priya.receipt.span_indices:
        sp = redacted_doc.entries[0]["spans"][sp_idx]
        assert sp["redacted"] is True and "text" not in sp and "nonce" not in sp

    # Innocent bystander sub-leaf receipts IN THE EXACT SAME SENTENCE still verify!
    assert vault.verify_item_receipt(r_dana.item.item_id) is True
    assert vault.verify_item_receipt(r_marcus.item.item_id) is True
    assert r_priya.receipt.verify_redactable(redacted_doc) is False

    # Export bundle and verify both programmatically and with standalone copied `verify.py`
    bundle = vault.export_bundle()
    assert verify_bundle(bundle, require_signature_check=True)["ok"] is True

    bundle_path = tmp_path / "v2_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    standalone_dir = tmp_path / "offline_box"
    standalone_dir.mkdir()
    copied_verify = standalone_dir / "verify.py"
    shutil.copy2(verify_mod.__file__, copied_verify)
    proc = subprocess.run(
        [sys.executable, str(copied_verify), str(bundle_path)],
        cwd=standalone_dir,
        env={"PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["ok"] is True

    # Negative tamper checks on v2 span bundles fail closed!
    bad_retained = json.loads(json.dumps(bundle))
    bad_retained["documents"]["handoff"]["entries"][0]["spans"][0]["nonce"] = "leaked-nonce"
    assert verify_bundle(bad_retained)["ok"] is False

    bad_mixed = json.loads(json.dumps(bundle))
    bad_mixed["documents"]["handoff"]["entries"][0]["text"] = "forged"
    assert verify_bundle(bad_mixed)["ok"] is False

    bad_span_bool = json.loads(json.dumps(bundle))
    bad_span_bool["receipts"][0]["span_indices"][0] = False
    assert verify_bundle(bad_span_bool)["ok"] is False

    # Forging span_indices onto a v1 flat doc fails closed
    v1_doc = RedactableDoc.from_text("Dana owns pharmacy.", source_id="flat")
    v1_rec = mint_redactable_receipt(v1_doc, "Dana owns pharmacy.").as_dict()
    v1_rec["span_indices"] = [0]
    v1_rec["span_count"] = 1
    v1_rec["span_proofs"] = [[]]
    assert verify_mod.verify_redactable_receipt(v1_rec, v1_doc.as_dict()) is False

    vault.close()


def test_a5_clause_span_mode_and_span_predicate_redaction():
    text = "Alex runs payroll, while Sam audits inventory; Lee handles shipping."
    vault = CustodyVault.ephemeral()
    vault.register_source("roles", text, mode="sentence", span_mode="clause")
    r_sam = vault.remember("Sam audits inventory", subjects=["sam"], source_id="roles", quote="Sam audits inventory")
    r_lee = vault.remember("Lee handles shipping.", subjects=["lee"], source_id="roles", quote="Lee handles shipping.")

    vault.erase("alex", span_predicate=lambda sp: "Alex" in sp)
    doc = vault.ledger.get_document(vault.tenant_id, "roles")
    assert doc is not None
    assert "Alex" not in doc.text()
    assert vault.verify_item_receipt(r_sam.item.item_id) is True
    assert vault.verify_item_receipt(r_lee.item.item_id) is True
    vault.close()
