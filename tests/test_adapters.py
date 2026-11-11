"""Integration tests for Mem0Adapter, VectorStoreAdapter, and SQLiteStoreAdapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

import pytest

from ember.adapters import Mem0Adapter, SQLiteStoreAdapter, VectorStoreAdapter
from ember.reducer import ErasureIncomplete
from ember.vault import CustodyVault


class FakeMem0V3Client:
    """Mimics `mem0.Memory` / `MemoryClient` wire signatures cleanly for unit integration testing."""

    def __init__(self, *, swallow_delete: bool = False) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.deleted_calls: list[str] = []
        self.swallow_delete = swallow_delete

    def add(
        self,
        messages,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = False,
    ) -> dict[str, Any]:
        # Simulate Mem0 creating a record per sentence when multiple lines/clauses arrive
        clauses = [c.strip() for c in str(messages).split(" | ") if c.strip()]
        created = []
        for clause in clauses:
            mid = f"m0_{uuid.uuid4().hex[:10]}"
            self.rows[mid] = {
                "id": mid,
                "memory": clause,
                "user_id": user_id,
                "agent_id": agent_id,
                "metadata": dict(metadata or {}),
            }
            created.append({"id": mid, "memory": clause, "event": "ADD"})
        return {"results": created}

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        q_tokens = {tok.strip(".,?!") for tok in query.lower().split() if tok.strip(".,?!")}
        out = []
        for row in self.rows.values():
            if filters:
                if "user_id" in filters and row.get("user_id") != filters["user_id"]:
                    continue
            words = {tok.strip(".,?!") for tok in row["memory"].lower().split()}
            overlap = len(q_tokens & words)
            if overlap > 0 or query.lower() in row["memory"].lower():
                out.append(
                    {
                        "id": row["id"],
                        "memory": row["memory"],
                        "score": overlap / max(1, len(q_tokens)),
                        "metadata": dict(row["metadata"]),
                    }
                )
        out.sort(key=lambda r: -r["score"])
        return {"results": out[:limit]}

    def delete(self, memory_id: str) -> None:
        self.deleted_calls.append(memory_id)
        if not self.swallow_delete:
            self.rows.pop(memory_id, None)

    def delete_all(self, filters: dict[str, Any]) -> None:
        target = filters.get("user_id")
        doomed = [mid for mid, row in self.rows.items() if row.get("user_id") == target]
        for mid in doomed:
            self.rows.pop(mid, None)


@dataclass
class _VecDoc:
    id: str
    page_content: str
    metadata: dict[str, Any]


class FakeVectorStore:
    """Mimics LangChain / Chroma vector store interface."""

    def __init__(self) -> None:
        self.docs: dict[str, _VecDoc] = {}

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]], ids: list[str]) -> list[str]:
        for t, m, i in zip(texts, metadatas, ids, strict=True):
            self.docs[i] = _VecDoc(id=i, page_content=t, metadata=dict(m))
        return list(ids)

    def similarity_search_with_score(self, query: str, k: int = 5, filter: dict[str, Any] | None = None):
        q = {w.strip(".,?!") for w in query.lower().split()}
        scored = []
        for doc in self.docs.values():
            if filter and any(doc.metadata.get(fk) != fv for fk, fv in filter.items()):
                continue
            words = {w.strip(".,?!") for w in doc.page_content.lower().split()}
            overlap = len(q & words)
            if overlap > 0:
                scored.append((doc, float(overlap)))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.docs.pop(i, None)


def test_mem0_adapter_cascades_across_multi_id_and_team_buckets():
    mem0 = FakeMem0V3Client()
    vault = CustodyVault.ephemeral(adapter=Mem0Adapter(mem0, default_user_id="team_acme"))

    vault.register_source(
        "onboarding",
        "Priya Raman works on triage routing. Dana Kim owns pharmacy inventory.",
    )
    r_priya = vault.remember(
        "Priya Raman works on triage routing | Priya joined Acme in 2019",
        subjects=["priya"],
        source_id="onboarding",
        quote="Priya Raman works on triage routing.",
        metadata={"user_id": "priya"},
    )
    assert len(r_priya.adapter_record_ids) == 2

    r_dana = vault.remember(
        "Dana Kim owns pharmacy inventory.",
        subjects=["dana"],
        source_id="onboarding",
        quote="Dana Kim owns pharmacy inventory.",
        metadata={"user_id": "dana"},
    )
    # Derived summary written into shared workspace bucket `team_acme` (naive delete_all(user_id="priya") misses it!)
    vault.remember(
        "Triage routing connects Priya and pharmacy inventory under Dana.",
        subjects=["team_acme"],
        derived_from=[r_priya.item.item_id, r_dana.item.item_id],
        metadata={"user_id": "team_acme"},
    )

    cert = vault.erase("priya", probes=["Where does Priya work?", "Who works on triage routing?"])
    assert len(cert.closure["adapter_record_ids"]) == 3  # 2 direct Mem0 rows + 1 derived team row

    # Raw Mem0 store has ZERO rows mentioning Priya, while Dana's row survives cleanly!
    remaining_texts = [row["memory"] for row in mem0.rows.values()]
    assert len(remaining_texts) == 1
    assert "Dana Kim" in remaining_texts[0]
    assert "Priya" not in " ".join(remaining_texts)

    res = vault.search("Dana pharmacy")
    assert not res.abstained
    assert res.hits[0].receipt_verified is True
    vault.close()


def test_mem0_vendor_delete_failure_fails_closed_on_erasure_cert():
    mem0 = FakeMem0V3Client(swallow_delete=True)
    vault = CustodyVault.ephemeral(adapter=Mem0Adapter(mem0))
    vault.remember("Priya lives in Portland.", subjects=["priya"])

    with pytest.raises(ErasureIncomplete):
        vault.erase("priya", probes=["Where does Priya live?"])
    # Certificate withheld when underlying vendor fails deletion, while live search defensively shields app
    assert vault.ledger.list_certificates(vault.tenant_id) == []
    guarded = vault.search("Portland")
    assert guarded.abstained and guarded.filtered_orphans >= 1
    vault.close()


def test_vector_store_and_sqlite_adapters(tmp_path):
    vec = FakeVectorStore()
    v1 = CustodyVault.ephemeral(adapter=VectorStoreAdapter(vec))
    v1.remember("Theo builds Spark pipelines at Acme.", subjects=["theo"])
    v1.remember("Nina designs UI at Acme.", subjects=["nina"])
    v1.erase("theo", probes=["Who builds Spark pipelines?"])
    assert len(vec.docs) == 1
    assert not v1.search("Nina UI").abstained
    v1.close()

    sql_adapter = SQLiteStoreAdapter(str(tmp_path / "ext.db"))
    v2 = CustodyVault.open(tmp_path / "vault", key_dir=tmp_path / "keys", adapter=sql_adapter)
    v2.remember("Amara runs epidemiology.", subjects=["amara"])
    v2.remember("Gil handles biostatistics.", subjects=["gil"])
    v2.erase("amara", probes=["Who runs epidemiology?"])
    assert v2.search("epidemiology").abstained
    assert not v2.search("biostatistics").abstained
    v2.close()
    sql_adapter.close()
