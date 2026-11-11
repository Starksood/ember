"""Comprehensive verification suite for Ember Phase 2 Shift B (ecosystem, KMS, Postgres, checkpoints, anchors, config)."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import pytest

from ember import (
    AsyncCustodyVault,
    AwsKmsKeyStore,
    ChromaAdapter,
    CustodyVault,
    Ed25519Signer,
    EmberLangChainRetriever,
    EmberLangChainVectorStore,
    EmberLlamaIndexRetriever,
    EmberMem0,
    ErasureIncomplete,
    GcpKmsKeyStore,
    IdempotencyConflict,
    LedgerIntegrityError,
    PgVectorAdapter,
    PineconeAdapter,
    PostgresLedger,
    QdrantAdapter,
    RekorAnchor,
    Rfc3161Anchor,
    SQLiteKeyStore,
    SQLiteStoreAdapter,
    VaultTransitKeyStore,
    decrypt_item_payload,
    find_ember_config,
    load_ember_config,
    open_vault_from_config,
    verify_bundle,
    verify_ed25519_certificate,
)
from ember.cli import main as cli_main
from ember.crypto import ERASED_TOMBSTONE
from ember.signing import _der_tlv, parse_rfc3161_response_status


def _det_embed(text: str) -> list[float]:
    dims = 64
    vec = [0.0] * dims
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2]
    for idx, tok in enumerate(toks):
        h = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:4], "big")
        vec[h % dims] += 1.0
        if idx + 1 < len(toks):
            bg = f"{tok}_{toks[idx + 1]}"
            bh = int.from_bytes(hashlib.sha256(bg.encode("utf-8")).digest()[:4], "big")
            vec[bh % dims] += 1.5
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _cos_sim(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# ── Mock Framework Clients & External Stores ─────────────────────────────────


class FakeMem0Client:
    """Minimal Mem0-compatible client with filter-aware vector store & delete tracking."""

    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.records: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def add(self, messages: Any, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None, metadata: dict[str, Any] | None = None, infer: bool = False) -> dict[str, Any]:
        del infer
        self._counter += 1
        rid = f"m0_{self._counter}"
        text = messages[0]["content"] if isinstance(messages, list) and isinstance(messages[0], dict) else str(messages)
        meta = dict(metadata or {})
        if user_id:
            meta["user_id"] = user_id
        if agent_id:
            meta["agent_id"] = agent_id
        if run_id:
            meta["run_id"] = run_id
        self.records[rid] = {"id": rid, "memory": text, "metadata": meta, "user_id": user_id}
        return {"results": [{"id": rid, "memory": text, "event": "ADD"}]}

    def search(self, query: str, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None, limit: int = 5, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[dict[str, Any]] = []
        for rec in self.records.values():
            meta = rec["metadata"]
            if user_id is not None and meta.get("user_id") != user_id and rec.get("user_id") != user_id:
                continue
            if agent_id is not None and meta.get("agent_id") != agent_id:
                continue
            if run_id is not None and meta.get("run_id") != run_id:
                continue
            r_toks = set(re.findall(r"[a-z0-9]+", rec["memory"].lower()))
            ov = len(q_toks & r_toks)
            if ov > 0:
                scored.append({"id": rec["id"], "memory": rec["memory"], "score": ov / max(1, len(q_toks)), "metadata": dict(meta)})
        scored.sort(key=lambda r: -r["score"])
        return {"results": scored[:limit]}

    def delete(self, memory_id: str) -> None:
        if self.fail_delete:
            return
        self.records.pop(memory_id, None)


class FakeQdrantClient:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.points: dict[str, dict[str, Any]] = {}

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        del collection_name
        for pt in points:
            if isinstance(pt, dict):
                pid, vec, payload = str(pt["id"]), list(pt["vector"]), dict(pt["payload"])
            else:
                pid, vec, payload = str(pt.id), list(pt.vector), dict(pt.payload)
            self.points[pid] = {"id": pid, "vector": vec, "payload": payload}

    def _matches_filter(self, payload: dict[str, Any], q_filter: Any) -> bool:
        if q_filter is None:
            return True
        must = q_filter.get("must", []) if isinstance(q_filter, dict) else getattr(q_filter, "must", [])
        for cond in must:
            key = cond["key"] if isinstance(cond, dict) else cond.key
            val = cond["match"]["value"] if isinstance(cond, dict) else cond.match.value
            p_val = payload.get(key)
            if isinstance(p_val, list):
                if val not in p_val:
                    return False
            elif p_val != val:
                return False
        return True

    def query_points(self, collection_name: str, query: list[float], query_filter: Any = None, limit: int = 5, with_payload: bool = True) -> dict[str, Any]:
        del collection_name, with_payload
        rows = []
        for pt in self.points.values():
            if not self._matches_filter(pt["payload"], query_filter):
                continue
            sc = _cos_sim(query, pt["vector"])
            rows.append({"id": pt["id"], "score": sc, "payload": dict(pt["payload"])})
        rows.sort(key=lambda r: -r["score"])
        return {"points": rows[:limit]}

    def delete(self, collection_name: str, points_selector: Any) -> None:
        del collection_name
        if self.fail_delete:
            return
        ids = points_selector.get("points", []) if isinstance(points_selector, dict) else points_selector.points
        for pid in ids:
            self.points.pop(str(pid), None)


class FakeChromaCollection:
    """Enforces Chroma's scalar-only metadata constraint strictly."""

    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.docs: dict[str, dict[str, Any]] = {}

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], embeddings: list[list[float]] | None = None) -> None:
        for idx, rid in enumerate(ids):
            meta = metadatas[idx]
            for k, v in meta.items():
                if v is not None and not isinstance(v, (str, int, float, bool)):
                    raise ValueError(f"Chroma metadata {k!r} must be scalar, got {type(v).__name__}")
            self.docs[rid] = {
                "doc": documents[idx],
                "meta": dict(meta),
                "emb": list(embeddings[idx]) if embeddings else _det_embed(documents[idx]),
            }

    @staticmethod
    def _match_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(FakeChromaCollection._match_where(meta, clause) for clause in where["$and"])
        return all(meta.get(k) == v for k, v in where.items())

    def query(self, *, query_embeddings: list[list[float]] | None = None, query_texts: list[str] | None = None, n_results: int = 5, where: dict[str, Any] | None = None, include: list[str] | None = None) -> dict[str, Any]:
        del include
        q_vec = query_embeddings[0] if query_embeddings else _det_embed(query_texts[0])
        ranked = []
        for rid, rec in self.docs.items():
            if not self._match_where(rec["meta"], where):
                continue
            sim = _cos_sim(q_vec, rec["emb"])
            dist = max(0.0, 1.0 - sim)
            ranked.append((dist, rid, rec["doc"], rec["meta"]))
        ranked.sort(key=lambda x: x[0])
        top = ranked[:n_results]
        return {
            "ids": [[x[1] for x in top]],
            "documents": [[x[2] for x in top]],
            "metadatas": [[x[3] for x in top]],
            "distances": [[x[0] for x in top]],
        }

    def delete(self, *, ids: list[str]) -> None:
        if self.fail_delete:
            return
        for rid in ids:
            self.docs.pop(rid, None)


class FakePineconeIndex:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.namespaces: dict[str, dict[str, dict[str, Any]]] = {}

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str = "") -> None:
        ns = self.namespaces.setdefault(namespace, {})
        for row in vectors:
            meta = row["metadata"]
            for k, v in meta.items():
                if not isinstance(v, (str, int, float, bool, list)):
                    raise ValueError(f"unsupported Pinecone metadata type for {k!r}")
                if isinstance(v, list) and not all(isinstance(el, str) for el in v):
                    raise ValueError(f"Pinecone list metadata for {k!r} must be list[str]")
            ns[row["id"]] = {"id": row["id"], "values": list(row["values"]), "metadata": dict(meta)}

    @staticmethod
    def _match(meta: dict[str, Any], flt: dict[str, Any] | None) -> bool:
        if not flt:
            return True
        for k, expr in flt.items():
            target = expr["$eq"] if isinstance(expr, dict) and "$eq" in expr else expr
            val = meta.get(k)
            if isinstance(val, list):
                if target not in val:
                    return False
            elif val != target:
                return False
        return True

    def query(self, *, vector: list[float], top_k: int = 5, include_metadata: bool = True, namespace: str = "", filter: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: A002
        del include_metadata
        ns = self.namespaces.get(namespace, {})
        ranked = []
        for row in ns.values():
            if not self._match(row["metadata"], filter):
                continue
            sc = _cos_sim(vector, row["values"])
            ranked.append({"id": row["id"], "score": sc, "metadata": dict(row["metadata"])})
        ranked.sort(key=lambda r: -r["score"])
        return {"matches": ranked[:top_k]}

    def delete(self, *, ids: list[str], namespace: str = "") -> None:
        if self.fail_delete:
            return
        ns = self.namespaces.get(namespace, {})
        for rid in ids:
            ns.pop(rid, None)


class MockPgVectorConnection:
    """In-memory DB-API mock for PgVectorAdapter enforcing parameterized queries."""

    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.rows: dict[str, dict[str, Any]] = {}
        self.queries_seen: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self) -> "MockPgVectorCursor":
        return MockPgVectorCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class MockPgVectorCursor:
    def __init__(self, conn: MockPgVectorConnection) -> None:
        self.conn = conn
        self._result: list[tuple[Any, ...]] = []
        self.rowcount: int = -1

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.conn.queries_seen.append((sql, tuple(params)))
        if sql.startswith("CREATE EXTENSION") or sql.startswith("CREATE TABLE"):
            return
        if sql.startswith("INSERT INTO "):
            rec_id, item_id, text, meta_str, vec_lit = params
            vec = json.loads(vec_lit) if vec_lit is not None else None
            self.conn.rows[rec_id] = {
                "record_id": rec_id,
                "item_id": item_id,
                "text": text,
                "meta": json.loads(meta_str),
                "embedding": vec,
            }
            self.rowcount = 1
            return
        if "ORDER BY embedding <=>" in sql:
            if "meta_json @>" in sql:
                vec_lit, flt_json, _, limit = params
                flt = json.loads(flt_json)
            else:
                vec_lit, _, limit = params
                flt = None
            q_vec = json.loads(vec_lit)
            scored: list[tuple[Any, ...]] = []
            for row in self.conn.rows.values():
                if row["embedding"] is None:
                    continue
                if flt and any(row["meta"].get(k) != v for k, v in flt.items()):
                    continue
                sim = _cos_sim(q_vec, row["embedding"])
                scored.append((row["record_id"], row["text"], row["meta"], sim))
            scored.sort(key=lambda r: -r[3])
            self._result = scored[: int(limit)]
            return
        if sql.startswith("DELETE FROM ") and "ANY(%s)" in sql:
            (ids_list,) = params
            if self.conn.fail_delete:
                self.rowcount = 0
                return
            count = 0
            for rid in ids_list:
                if self.conn.rows.pop(rid, None) is not None:
                    count += 1
            self.rowcount = count
            return
        raise NotImplementedError(f"unexpected SQL in MockPgVectorCursor: {sql}")

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._result)

    def close(self) -> None:
        pass


class MockPostgresLedgerConn:
    """Full stateful PostgreSQL DB-API emulator backed by private SQLite for testing PostgresLedger."""

    def __init__(self) -> None:
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self.executed_sql: list[str] = []

    def cursor(self) -> "MockPostgresLedgerCursor":
        return MockPostgresLedgerCursor(self)

    def commit(self) -> None:
        self._db.commit()

    def rollback(self) -> None:
        self._db.rollback()

    def close(self) -> None:
        self._db.close()


class MockPostgresLedgerCursor:
    def __init__(self, wrapper: MockPostgresLedgerConn) -> None:
        self.w = wrapper
        self._cur = wrapper._db.cursor()
        self.rowcount = -1

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        cleaned = " ".join(sql.split())
        self.w.executed_sql.append(cleaned)
        if cleaned == "BEGIN":
            if not self.w._db.in_transaction:
                self._cur.execute("BEGIN IMMEDIATE")
            return
        if cleaned.startswith("SELECT pg_advisory_xact_lock"):
            self._cur.execute("SELECT 1")
            return
        # Translate PostgreSQL UPSERT EXCLUDED / BIGINT idioms cleanly to SQLite dialect
        sqlite_sql = cleaned.replace("BIGINT", "INTEGER")
        sqlite_sql = sqlite_sql.replace("ON CONFLICT (tenant_id, seq) DO UPDATE", "ON CONFLICT(tenant_id, seq) DO UPDATE")
        sqlite_sql = sqlite_sql.replace("ON CONFLICT (tenant_id, source_id) DO UPDATE", "ON CONFLICT(tenant_id, source_id) DO UPDATE")
        sqlite_sql = sqlite_sql.replace("ON CONFLICT (tenant_id, certificate_id) DO UPDATE", "ON CONFLICT(tenant_id, certificate_id) DO UPDATE")
        sqlite_sql = sqlite_sql.replace("EXCLUDED.", "excluded.")
        sqlite_sql = sqlite_sql.replace("%s", "?")
        self._cur.execute(sqlite_sql, tuple(params))
        self.rowcount = self._cur.rowcount

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()

    def close(self) -> None:
        self._cur.close()


# ── B1 Tests: Framework Drop-ins & Native Adapters ───────────────────────────


def test_b1_ember_mem0_wrap_trace_and_erase(tmp_path: Path) -> None:
    client = FakeMem0Client()
    mem = EmberMem0.wrap(
        client,
        data_dir=tmp_path / "vault",
        key_dir=tmp_path / "keys",
        default_user_id="alice",
    )
    mem.register_source(
        "doc_hr",
        "Alice Mercer joined on Jan 5. Alice has severe walnut allergy.",
        span_mode="clause",
        origin="hr",
    )
    a1 = mem.add(
        [{"role": "user", "content": "Alice has severe walnut allergy"}],
        user_id="alice",
        agent_id="concierge",
        run_id="r1",
        source_id="doc_hr",
        quote="Alice has severe walnut allergy",
    )
    assert a1["results"][0]["event"] == "ADD"
    assert a1["ember_receipt"] is not None

    mem.add("Bob prefers dark roast espresso", user_id="bob", agent_id="concierge", run_id="r1")

    with mem.trace_turn():
        found = mem.search("walnut allergy", user_id="alice")
        assert len(found["results"]) == 1
        assert found["results"][0]["ember_receipt_verified"] is True
        a2 = mem.add("Avoid serving pralines to Alice", user_id="alice")
        assert a1["ember_item_id"] in a2["ember_derived_from"]

    assert len(mem.get_all(user_id="alice")["results"]) == 2
    cert = mem.erase("alice", aliases=["Alice Mercer"])
    assert verify_ed25519_certificate(cert.as_dict())

    assert mem.search("pralines", user_id="alice")["results"] == []
    assert mem.get_all(user_id="alice")["results"] == []
    bob_hits = mem.search("espresso", user_id="bob")
    assert len(bob_hits["results"]) == 1
    mem.close()


def test_b1_langchain_retriever_and_vectorstore_flow(tmp_path: Path) -> None:
    vault = CustodyVault.open(tmp_path / "lc_vault", key_dir=tmp_path / "lc_keys", embedder=_det_embed)
    store = EmberLangChainVectorStore(vault, subject_key="user_id")
    vault.register_source("src_lc", "Nora Lin takes beta blockers. Liam Cho plays cello.")

    ids = store.add_texts(
        ["Nora Lin takes beta blockers.", "Liam Cho plays cello."],
        metadatas=[
            {"user_id": "nora", "source_id": "src_lc", "quote": "Nora Lin takes beta blockers."},
            {"user_id": "liam", "source_id": "src_lc", "quote": "Liam Cho plays cello."},
        ],
    )
    assert len(ids) == 2

    retriever = store.as_retriever(search_kwargs={"k": 3})
    assert isinstance(retriever, EmberLangChainRetriever)
    docs = retriever.invoke("beta blockers")
    assert len(docs) == 1
    assert docs[0].metadata["ember_receipt_verified"] is True

    async def _run_async() -> list[Any]:
        return await retriever.ainvoke("cello")

    async_docs = asyncio.run(_run_async())
    assert len(async_docs) == 1
    assert "cello" in async_docs[0].page_content

    with pytest.raises(ValueError, match="subject_id"):
        store.delete(ids=ids[:1])

    cert = store.erase("nora", markers=["beta blockers"])
    assert verify_ed25519_certificate(cert.as_dict())
    assert store.similarity_search("beta blockers", k=3) == []
    assert len(retriever.invoke("cello")) == 1
    vault.close()


def test_b1_llamaindex_retriever() -> None:
    @dataclass
    class _FakeBundle:
        query_str: str

    vault = CustodyVault.ephemeral()
    vault.register_source("li_src", "Elena Costa lives in Lisbon. Kenji Sato lives in Kyoto.")
    vault.remember("Elena Costa lives in Lisbon", subjects=["elena"], source_id="li_src")
    vault.remember("Kenji Sato lives in Kyoto", subjects=["kenji"], source_id="li_src")

    retriever = EmberLlamaIndexRetriever(vault, similarity_top_k=4)
    nodes = retriever.retrieve(_FakeBundle(query_str="Lisbon"))
    assert len(nodes) == 1
    assert nodes[0].node.metadata["ember_receipt_verified"] is True

    async def _aret() -> list[Any]:
        return await retriever.aretrieve("Kyoto")

    anodes = asyncio.run(_aret())
    assert len(anodes) == 1

    vault.erase("elena")
    assert retriever.retrieve("Lisbon") == []
    assert len(retriever.retrieve("Kyoto")) == 1
    vault.close()


@pytest.mark.parametrize("backend_name", ["qdrant", "chroma", "pgvector", "pinecone"])
def test_b1_native_filtered_vector_adapters_lifecycle_and_fail_closed(backend_name: str) -> None:
    def _make_adapter(fail_delete: bool) -> tuple[Any, Any]:
        if backend_name == "qdrant":
            raw = FakeQdrantClient(fail_delete=fail_delete)
            return QdrantAdapter(raw, "mem_col", embed_fn=_det_embed), raw
        if backend_name == "chroma":
            raw = FakeChromaCollection(fail_delete=fail_delete)
            return ChromaAdapter(raw, embed_fn=_det_embed), raw
        if backend_name == "pgvector":
            raw = MockPgVectorConnection(fail_delete=fail_delete)
            return PgVectorAdapter(raw, table_name="ember_vec_test", dimensions=64, embed_fn=_det_embed), raw
        raw = FakePineconeIndex(fail_delete=fail_delete)
        return PineconeAdapter(raw, namespace="prod", embed_fn=_det_embed), raw

    adapter, raw_backend = _make_adapter(fail_delete=False)
    vault = CustodyVault.ephemeral(adapter=adapter, encrypt_vendor_payload=True, embedder=_det_embed)
    vault.register_source("s1", "Doctor Maya Patel prescribed lithium. Jonas Weber swims daily.")
    m_maya = vault.remember(
        "Doctor Maya Patel prescribed lithium",
        subjects=["maya"],
        source_id="s1",
        metadata={"user_id": "maya", "dept": "psychiatry", "tags": ["rx", "vip"]},
    )
    m_jonas = vault.remember(
        "Jonas Weber swims daily",
        subjects=["jonas"],
        source_id="s1",
        metadata={"user_id": "jonas", "dept": "cardio", "tags": ["fitness"]},
    )

    # Zero plaintext PII leaked to vendor storage!
    raw_dump = json.dumps(getattr(raw_backend, "points", getattr(raw_backend, "docs", getattr(raw_backend, "namespaces", getattr(raw_backend, "rows", {})))))
    assert "lithium" not in raw_dump
    assert "Maya Patel" not in raw_dump

    hits = vault.search("prescribed lithium", filters={"user_id": "maya", "dept": "psychiatry"})
    assert len(hits.hits) == 1
    assert hits.hits[0].item_id == m_maya.item.item_id
    assert hits.hits[0].metadata["tags"] == ["rx", "vip"]

    if backend_name == "pgvector":
        # Ensure all queries with values were parameterized (no raw text concatenated into SQL)
        for sql_stmt, _ in raw_backend.queries_seen:
            assert "lithium" not in sql_stmt

    cert = vault.erase("maya", probes=["lithium", "Maya Patel"])
    assert verify_ed25519_certificate(cert.as_dict())
    assert vault.search("lithium", filters={"user_id": "maya"}).abstained
    jonas_hits = vault.search("swims daily", filters={"user_id": "jonas"})
    assert len(jonas_hits.hits) == 1
    assert jonas_hits.hits[0].item_id == m_jonas.item.item_id
    vault.close()

    # Fail-closed verification: if backing vector store silently ignores delete, erase() MUST raise ErasureIncomplete!
    bad_adapter, _ = _make_adapter(fail_delete=True)
    v_bad = CustodyVault.ephemeral(adapter=bad_adapter, encrypt_vendor_payload=True, embedder=_det_embed)
    v_bad.remember("Secret lithium record", subjects=["maya"])
    with pytest.raises(ErasureIncomplete):
        v_bad.erase("maya", probes=["lithium"])
    v_bad.close()


# ── B2 Tests: SQLiteKeyStore, KMS, Checkpoints, PostgresLedger, AsyncVault ────


def test_b2_sqlite_keystore_secure_delete_and_persistence(tmp_path: Path) -> None:
    ks_path = tmp_path / "keys.db"
    ks = SQLiteKeyStore(ks_path)
    mode = stat.S_IMODE(os.stat(ks_path).st_mode)
    assert mode == 0o600
    assert ks._conn.execute("PRAGMA secure_delete").fetchone()[0] == 1

    s_key = ks.get_or_create_subject_key("alice")
    r_key = ks.create_record_key("rec_1", ["alice", "bob"])
    tok1 = ks.tombstone_token("Alice Mercer")
    ks.close()

    reopened = SQLiteKeyStore(ks_path)
    assert reopened.get_subject_key("alice") == s_key
    assert reopened.get_record_key("rec_1") == r_key
    assert reopened.tombstone_token("  alice   mercer ") == tok1

    rep = reopened.shred_subject("alice")
    assert rep.subject_key_destroyed is True
    assert rep.record_keys_destroyed == 1
    assert reopened.get_subject_key("alice") is None
    assert reopened.get_record_key("rec_1") is None
    reopened.close()


def test_b2_kms_keystores_envelope_isolation_and_irreversible_shred(tmp_path: Path) -> None:
    master = os.urandom(32)

    class _FakeAwsClient:
        def encrypt(self, *, KeyId: str, Plaintext: bytes, EncryptionContext: dict[str, str]) -> dict[str, Any]:
            blob = {
                "kid": KeyId,
                "ctx": EncryptionContext,
                "pt_xor": base64.b64encode(bytes(b ^ master[i % 32] for i, b in enumerate(Plaintext))).decode(),
            }
            return {"CiphertextBlob": b"AWSKMS:" + json.dumps(blob, sort_keys=True).encode()}

        def decrypt(self, *, CiphertextBlob: bytes, EncryptionContext: dict[str, str]) -> dict[str, Any]:
            assert CiphertextBlob.startswith(b"AWSKMS:")
            parsed = json.loads(CiphertextBlob[7:].decode())
            if parsed["ctx"] != EncryptionContext:
                raise RuntimeError("KMS InvalidCiphertextException: context mismatch")
            raw = base64.b64decode(parsed["pt_xor"])
            return {"Plaintext": bytes(b ^ master[i % 32] for i, b in enumerate(raw))}

    class _FakeGcpClient:
        def encrypt(self, *, request: dict[str, Any]) -> Any:
            blob = {
                "name": request["name"],
                "aad": base64.b64encode(request["additional_authenticated_data"]).decode(),
                "pt": base64.b64encode(bytes(b ^ master[i % 32] for i, b in enumerate(request["plaintext"]))).decode(),
            }
            return {"ciphertext": b"GCPKMS:" + json.dumps(blob, sort_keys=True).encode()}

        def decrypt(self, *, request: dict[str, Any]) -> Any:
            parsed = json.loads(request["ciphertext"][7:].decode())
            if base64.b64decode(parsed["aad"]) != request["additional_authenticated_data"]:
                raise RuntimeError("GCP AAD mismatch")
            raw = base64.b64decode(parsed["pt"])
            return {"plaintext": bytes(b ^ master[i % 32] for i, b in enumerate(raw))}

    def _fake_vault_http(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        assert headers["X-Vault-Token"] == "hvs.test"
        if "/encrypt/" in url:
            pt = base64.b64decode(body["plaintext"])
            envelope = base64.b64encode(
                json.dumps({"ctx": body["context"], "d": base64.b64encode(pt).decode()}).encode()
            ).decode()
            return {"data": {"ciphertext": f"vault:v1:{envelope}"}}
        if "/decrypt/" in url:
            _, _, enc = body["ciphertext"].partition("vault:v1:")
            parsed = json.loads(base64.b64decode(enc).decode())
            if parsed["ctx"] != body["context"]:
                raise RuntimeError("Vault Transit context mismatch")
            return {"data": {"plaintext": parsed["d"]}}
        raise AssertionError(url)

    aws_ks = AwsKmsKeyStore("arn:aws:kms:us-east-1:123:key/abc", tmp_path / "aws.db", kms_client=_FakeAwsClient())
    gcp_ks = GcpKmsKeyStore("projects/p/locations/global/keyRings/r/cryptoKeys/k", tmp_path / "gcp.db", kms_client=_FakeGcpClient())
    vt_ks = VaultTransitKeyStore("https://vault.internal:8200", "ember-key", "hvs.test", tmp_path / "vault.db", http_fn=_fake_vault_http)

    for idx, ks in enumerate((aws_ks, gcp_ks, vt_ks)):
        v = CustodyVault.open(tmp_path / f"v_{idx}", keystore=ks)
        assert v.keys_separated_from_ledger is True
        res = v.remember(f"Sensitive secret #{idx}", subjects=["claire"])
        claire_ref = ks.subject_ref("claire")
        raw_dek = ks.get_subject_key(claire_ref)
        assert raw_dek is not None and len(raw_dek) == 32

        # SQLite keystore file MUST NEVER contain raw plaintext DEK bytes
        db_bytes = Path(ks.path).read_bytes()
        assert raw_dek not in db_bytes

        # Wrong KMS context must be rejected on unwrap
        row = ks._conn.execute("SELECT wrapped_blob FROM kms_subject_keys WHERE subject_id = ?", (claire_ref,)).fetchone()
        assert row is not None
        with pytest.raises(RuntimeError, match="mismatch"):
            ks._unwrap(bytes(row[0]), {"ember_tenant": "other", "ember_purpose": "subject_cek"})

        ks.clear_cache()
        # Re-read forces real unwrap from KMS
        assert ks.get_subject_key(claire_ref) == raw_dek

        # Erase purges both cache & wrapped DB blob -> permanent crypto-shred!
        cert = v.erase("claire")
        assert cert.key_destroyed is True
        assert ks.get_subject_key(claire_ref) is None
        replayed = v.replay()
        assert res.item.item_id not in replayed.items
        raw_ev_item = v.ledger.events()[0].payload["item"]
        assert decrypt_item_payload(raw_ev_item, ks) == ((), ERASED_TOMBSTONE, None)
        assert v.search("Secret").abstained is True
        v.close()


def test_b2_signed_checkpoints_privacy_o_delta_and_tamper_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "ckpt_vault"
    key_dir = tmp_path / "ckpt_keys"
    store1 = SQLiteStoreAdapter(data_dir / "store.db")
    v1 = CustodyVault.open(data_dir, key_dir=key_dir, adapter=store1, auto_checkpoint_every=3)
    v1.register_source("d1", "Patient Maya takes aspirin. Patient Leo takes insulin.")
    m1 = v1.remember("Patient Maya takes aspirin", subjects=["maya"], source_id="d1")
    v1.remember("Patient Leo takes insulin", subjects=["leo"], source_id="d1")
    # 3 ops committed -> auto_checkpoint triggered at seq=2!

    # Inspect raw checkpoint row on disk -> MUST contain zero plaintext PII!
    ckpt_row = v1.ledger._conn.execute("SELECT seq, checkpoint_json FROM checkpoints").fetchone()
    assert ckpt_row is not None
    assert ckpt_row[0] == 2
    assert "aspirin" not in ckpt_row[1]
    assert "insulin" not in ckpt_row[1]
    assert m1.item.text_ciphertext in ckpt_row[1]

    # Add 1 delta item AFTER checkpoint
    v1.remember("Leo runs marathons", subjects=["leo"])
    v1.close()
    store1.close()

    # Track calls to `decrypt_item_payload` in BOTH ledger restore and delta event reducer!
    import ember.ledger as ledger_mod
    import ember.reducer as reducer_mod

    restore_calls: list[str] = []
    delta_calls: list[str] = []
    orig_ledger_dec = ledger_mod.decrypt_item_payload
    orig_reducer_dec = reducer_mod.decrypt_item_payload

    def _spy_restore(payload: dict[str, Any], keystore: Any) -> Any:
        restore_calls.append(str(payload.get("item_id")))
        return orig_ledger_dec(payload, keystore)

    def _spy_delta(payload: dict[str, Any], keystore: Any) -> Any:
        delta_calls.append(str(payload.get("item_id")))
        return orig_reducer_dec(payload, keystore)

    monkeypatch.setattr(ledger_mod, "decrypt_item_payload", _spy_restore)
    monkeypatch.setattr(reducer_mod, "decrypt_item_payload", _spy_delta)

    store2 = SQLiteStoreAdapter(data_dir / "store.db")
    v2 = CustodyVault.open(data_dir, key_dir=key_dir, adapter=store2, use_checkpoint=True)
    # Checkpointed startup restores the 2 snapshot items + applies ONLY the 1 post-checkpoint delta event!
    assert len(restore_calls) == 2
    assert len(delta_calls) == 1
    full_state = v2.replay(use_checkpoint=False)
    assert v2.state.digest() == full_state.digest()
    assert len(v2.search("marathons").hits) == 1

    # Tamper checkpoint JSON -> MUST reject with LedgerIntegrityError!
    v2.ledger._conn.execute(
        "UPDATE checkpoints SET checkpoint_json = REPLACE(checkpoint_json, '\"state_version\":3', '\"state_version\":99')"
    )
    v2.close()
    store2.close()
    store3 = SQLiteStoreAdapter(data_dir / "store.db")
    try:
        with pytest.raises(LedgerIntegrityError, match="signature"):
            CustodyVault.open(data_dir, key_dir=key_dir, adapter=store3, use_checkpoint=True)
    finally:
        store3.close()


def test_b2_postgres_ledger_backend_full_custody_lifecycle(tmp_path: Path) -> None:
    pg_conn = MockPostgresLedgerConn()
    pg_ledger = PostgresLedger(connection=pg_conn, table_prefix="prod_ember_")
    ks = SQLiteKeyStore(tmp_path / "pg_keys.db")
    signer = Ed25519Signer.generate()

    v = CustodyVault(
        ledger=pg_ledger,
        keystore=ks,
        signer=signer,
        tenant_id="acme",
        keys_separated_from_ledger=True,
    )
    assert isinstance(pg_ledger, type(v.ledger))
    v.register_source("pg_doc", "Aria Chen holds keycard 9912.")
    r1 = v.remember("Aria Chen holds keycard 9912", subjects=["aria"], source_id="pg_doc", event_id="ev-pg-1")
    v.remember("Zed Park visits Tokyo", subjects=["zed"])

    # Idempotency conflict check on PostgresLedger
    with pytest.raises(IdempotencyConflict):
        v.remember("Different text", subjects=["aria"], event_id="ev-pg-1")

    ckpt = v.checkpoint()
    assert ckpt["seq"] == 2
    assert pg_ledger.verify_chain("acme")[0] is True

    # Reopen against same PostgresLedger + keystore using checkpoint
    v2 = CustodyVault(
        ledger=pg_ledger,
        keystore=ks,
        signer=signer,
        tenant_id="acme",
        keys_separated_from_ledger=True,
    )
    assert v2.state.items[r1.item.item_id].text == "Aria Chen holds keycard 9912"
    cert = v2.erase("aria", probes=["9912", "Aria Chen"])
    assert verify_ed25519_certificate(cert.as_dict())
    assert len(pg_ledger.list_certificates("acme")) == 1
    bundle = v2.export_bundle()
    assert verify_bundle(bundle, require_signature_check=True)["ok"] is True

    v2.close()
    ks.close()
    pg_conn.close()


def test_b2_async_custody_vault_concurrent_turn_trace(tmp_path: Path) -> None:
    async def _scenario() -> None:
        async with await AsyncCustodyVault.open(tmp_path / "async_vault", key_dir=tmp_path / "async_keys") as av:
            await av.register_source("src_a", "Nina Patel works on Project Helios. Owen Reed plays chess.")
            p1 = await av.remember("Nina Patel works on Project Helios", subjects=["nina"], source_id="src_a")
            await av.remember("Owen Reed plays chess", subjects=["owen"], source_id="src_a")

            async with av.trace_turn() as trace:
                r1, r2 = await asyncio.gather(
                    av.search("Project Helios"),
                    av.search("chess"),
                )
                assert len(r1.hits) == 1 and len(r2.hits) == 1
                assert p1.item.item_id in trace.read_ids
                derived = await av.remember("Summary mentions Project Helios status", subjects=["nina"])
                assert p1.item.item_id in derived.item.derived_from

            cert = await av.erase("nina", probes=["Helios", "Summary"])
            assert verify_ed25519_certificate(cert.as_dict())
            assert (await av.search("Helios")).abstained is True
            assert len((await av.search("chess")).hits) == 1

    asyncio.run(_scenario())


# ── B3 Tests: RFC 3161, Rekor, ember.toml, & CLI Integration ─────────────────


def test_b3_rfc3161_and_rekor_anchors_verify_protocol_and_fail_closed() -> None:
    digest = "sha256:" + ("ab" * 32)
    seen_tsa: list[tuple[str, bytes, dict[str, str]]] = []

    def _ok_tsa(url: str, req_der: bytes, headers: dict[str, str]) -> bytes:
        seen_tsa.append((url, req_der, headers))
        # Minimal valid TimeStampResp SEQUENCE { PKIStatusInfo { INTEGER 0 }, ContentInfo { NULL } }
        status_info = _der_tlv(0x30, b"\x02\x01\x00")
        token_dummy = _der_tlv(0x30, b"\x05\x00")
        return _der_tlv(0x30, status_info + token_dummy)

    tsa = Rfc3161Anchor("https://tsa.example.org/tsr", transport=_ok_tsa)
    receipt = tsa.anchor(digest)
    assert receipt.scheme == "rfc3161"
    assert receipt.token.startswith("rfc3161:tsr:")
    assert seen_tsa[0][2]["Content-Type"] == "application/timestamp-query"
    # Verify digest bytes are embedded inside TimeStampReq DER
    assert bytes.fromhex("ab" * 32) in seen_tsa[0][1]
    assert parse_rfc3161_response_status(base64.b64decode(receipt.token.split(":", 2)[2])) == (0, True)

    def _reject_tsa(url: str, req_der: bytes, headers: dict[str, str]) -> bytes:
        del url, req_der, headers
        return _der_tlv(0x30, _der_tlv(0x30, b"\x02\x01\x02"))

    bad_tsa = Rfc3161Anchor("https://tsa.example.org/tsr", transport=_reject_tsa)
    with pytest.raises(RuntimeError, match="PKIStatus=2"):
        bad_tsa.anchor(digest)

    # Rekor anchor: validate actual Ed25519 signature sent in hashedrekord payload!
    posted_rekor: list[dict[str, Any]] = []

    def _ok_rekor(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
        del url, headers
        payload = json.loads(body.decode("utf-8"))
        posted_rekor.append(payload)
        sig_bytes = base64.b64decode(payload["spec"]["signature"]["content"])
        pub_pem = base64.b64decode(payload["spec"]["signature"]["publicKey"]["content"])
        pub = serialization.load_pem_public_key(pub_pem)
        assert isinstance(pub, Ed25519PublicKey)
        pub.verify(sig_bytes, bytes.fromhex(payload["spec"]["data"]["hash"]["value"]))
        return 201, json.dumps({"uuid-123": {"logIndex": 42, "integratedTime": 1730000000, "logID": "c0d2"}}).encode()

    rekor = RekorAnchor("https://rekor.sigstore.dev", transport=_ok_rekor)
    rk_receipt = rekor.anchor(digest)
    assert rk_receipt.scheme == "rekor"
    assert json.loads(rk_receipt.token.partition(":")[2])["logIndex"] == 42
    assert posted_rekor[0]["kind"] == "hashedrekord"

    bad_rekor = RekorAnchor("https://rekor.sigstore.dev", transport=lambda u, b, h: (503, b"unavailable"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        bad_rekor.anchor(digest)


def test_b3_ember_toml_config_and_cli_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    proj = tmp_path / "myapp"
    proj.mkdir()
    monkeypatch.setenv("EMBER_TEST_TENANT", "tenant_cfg")
    cfg_file = proj / "ember.toml"
    cfg_file.write_text(
        """
[vault]
data_dir = "var/ledger_data"
key_dir = "var/isolated_keys"
tenant_id = "${EMBER_TEST_TENANT}"
encrypt_vendor_payload = true
auto_checkpoint_every = 2

[keystore]
backend = "sqlite"

[adapter]
backend = "sqlite"

[anchor]
backend = "none"
""".lstrip(),
        encoding="utf-8",
    )

    assert find_ember_config(proj) == cfg_file.resolve()
    cfg = load_ember_config(cfg_file)
    assert cfg.tenant_id == "tenant_cfg"
    assert cfg.encrypt_vendor_payload is True

    # Open programmatically via ember.toml and confirm keys are separated + encrypted vendor mode active
    vault = open_vault_from_config(cfg)
    assert vault.keys_separated_from_ledger is True
    assert isinstance(vault.keystore, SQLiteKeyStore)
    assert (proj / "var/isolated_keys/keys.db").exists()
    vault.close()

    # Exercise CLI commands end-to-end with `--config` + `--span-mode clause` + `checkpoint` + `replay --use-checkpoint`
    assert cli_main(["--config", str(cfg_file), "init"]) == 0
    out_init = json.loads(capsys.readouterr().out)
    assert out_init["tenant_id"] == "tenant_cfg"
    assert out_init["keys_separated_from_ledger"] is True
    assert out_init["encrypt_vendor_payload"] is True

    assert (
        cli_main(
            [
                "--config",
                str(cfg_file),
                "add-source",
                "policy_doc",
                "--span-mode",
                "clause",
                "--text",
                "Alice takes warfarin daily, while Ben swims at noon.",
            ]
        )
        == 0
    )
    _ = capsys.readouterr()

    assert (
        cli_main(
            [
                "--config",
                str(cfg_file),
                "remember",
                "Alice takes warfarin daily",
                "--subject",
                "alice",
                "--source-id",
                "policy_doc",
            ]
        )
        == 0
    )
    _ = capsys.readouterr()

    assert cli_main(["--config", str(cfg_file), "checkpoint"]) == 0
    ckpt_out = json.loads(capsys.readouterr().out)
    assert ckpt_out["status"] == "checkpointed"
    assert ckpt_out["signature"].startswith("ed25519:")

    assert cli_main(["--config", str(cfg_file), "replay", "--use-checkpoint"]) == 0
    rep_out = json.loads(capsys.readouterr().out)
    assert rep_out["chain_ok"] is True
    assert rep_out["live_item_count"] == 1
