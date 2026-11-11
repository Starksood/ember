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


class StrictMem0V2Mock:
    """Enforces Mem0 2.x exact runtime contract discovered against mem0ai==2.2.1."""

    def __init__(self, *, fail_once_on_delete: bool = False) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.fail_once_on_delete = fail_once_on_delete
        self.delete_attempts = 0

    def add(
        self,
        messages,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> dict[str, Any]:
        meta = dict(metadata or {})
        for key in ("user_id", "agent_id", "run_id"):
            if key in meta:
                self.warnings.append(f"add(): ignoring metadata[{key!r}]")
                meta.pop(key, None)
        for name, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if val is not None:
                if not str(val).strip() or any(ch.isspace() for ch in str(val).strip()):
                    raise ValueError(f"Invalid {name}: cannot contain whitespace.")
        if not any((user_id, agent_id, run_id)):
            raise ValueError("At least one of user_id, agent_id, or run_id must be provided.")
        mid = f"m2_{uuid.uuid4().hex[:10]}"
        self.rows[mid] = {
            "id": mid,
            "memory": str(messages),
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": meta,
            "infer": infer,
        }
        return {"results": [{"id": mid, "memory": str(messages), "event": "ADD"}]}

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        **kwargs,
    ) -> dict[str, Any]:
        if any(k in kwargs for k in ("user_id", "agent_id", "run_id")):
            raise ValueError("Top-level entity params rejected in search(); use filters.")
        if not filters or not any(k in filters for k in ("user_id", "agent_id", "run_id")):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        self.search_calls.append({"query": query, "top_k": top_k, "filters": dict(filters), "threshold": threshold})

        q_tokens = {tok.strip(".,?!") for tok in query.lower().split() if tok.strip(".,?!")}
        results = []
        for row in self.rows.values():
            if "user_id" in filters and row.get("user_id") != filters["user_id"]:
                continue
            if "agent_id" in filters and row.get("agent_id") != filters["agent_id"]:
                continue
            if "run_id" in filters and row.get("run_id") != filters["run_id"]:
                continue
            words = {tok.strip(".,?!") for tok in row["memory"].lower().split() if tok.strip(".,?!")}
            overlap = len(q_tokens & words)
            score = overlap / max(1, len(q_tokens))
            if score >= threshold:
                # Mirror real Mem0 2.x output shape: top-level user_id, metadata separate
                results.append(
                    {
                        "id": row["id"],
                        "memory": row["memory"],
                        "score": score,
                        "user_id": row["user_id"],
                        **({"agent_id": row["agent_id"]} if row.get("agent_id") else {}),
                        **({"run_id": row["run_id"]} if row.get("run_id") else {}),
                        "metadata": dict(row["metadata"]),
                    }
                )
        results.sort(key=lambda r: -r["score"])
        return {"results": results[:top_k]}

    def delete(self, memory_id: str) -> dict[str, str]:
        self.delete_attempts += 1
        if memory_id not in self.rows:
            raise ValueError(f"Memory with id {memory_id} not found")
        del self.rows[memory_id]
        if self.fail_once_on_delete:
            self.fail_once_on_delete = False
            raise ConnectionError("simulated mid-batch Mem0 timeout after deleting one ID")
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None) -> None:
        doomed = [
            mid
            for mid, row in self.rows.items()
            if (user_id is None or row.get("user_id") == user_id)
            and (agent_id is None or row.get("agent_id") == agent_id)
            and (run_id is None or row.get("run_id") == run_id)
        ]
        for mid in doomed:
            del self.rows[mid]


def test_mem0_v2_strict_scopes_whitespace_restart_and_idempotent_retry(tmp_path):
    strict = StrictMem0V2Mock(fail_once_on_delete=True)
    vault_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"

    adapter1 = Mem0Adapter(strict, search_threshold=0.0)
    v1 = CustodyVault.open(vault_dir, key_dir=key_dir, adapter=adapter1)
    v1.register_source("intake", "Yuki translates novels in Sapporo. Marco edits poetry at Kirin Press.")
    r_yuki = v1.remember(
        "Yuki is a translator in Sapporo.",
        subjects=["Yuki"],
        source_id="intake",
        quote="Yuki translates novels in Sapporo.",
        metadata={"user_id": "Yuki"},
    )
    r_marco = v1.remember(
        "Marco is an editor at Kirin Press.",
        subjects=["Marco"],
        source_id="intake",
        quote="Marco edits poetry at Kirin Press.",
        metadata={"user_id": "Marco"},
    )
    # Whitespace org name (`Kirin Press`) previously triggered ValueError in real Mem0 2.x!
    v1.remember(
        "Staff summary at Kirin Press: Yuki in Sapporo collaborates with Marco.",
        subjects=["Kirin Press"],
        derived_from=[r_yuki.item.item_id, r_marco.item.item_id],
        metadata={"user_id": "Kirin Press"},
    )

    assert strict.warnings == []

    # First erase deletes Yuki's ID, then raises simulated timeout on the derived ID
    with pytest.raises(ConnectionError):
        v1.erase("Yuki", probes=["Where does Yuki work?"])
    v1.close()

    # Reopen on fresh Mem0Adapter after restart: scopes restore from state, retry deletes remaining ID,
    # and already-deleted Yuki ID raises "Memory with id ... not found" which Mem0Adapter handles idempotently!
    adapter2 = Mem0Adapter(strict, search_threshold=0.0)
    v2 = CustodyVault.open(vault_dir, key_dir=key_dir, adapter=adapter2)
    cert = v2.erase("Yuki", probes=["Where does Yuki work?", "translator Sapporo"])
    assert len(cert.closure["adapter_record_ids"]) == 2
    assert len(strict.rows) == 1
    # Ensure top_k and threshold=0.0 reached Mem0 search() rather than falling back to limit/0.1
    assert any(c["top_k"] == 25 and c["threshold"] == 0.0 for c in strict.search_calls)

    marco_res = v2.search("Marco edits poetry", limit=5)
    assert not marco_res.abstained
    assert marco_res.hits[0].metadata.get("user_id") == "Marco"
    assert v2.verify_item_receipt(r_marco.item.item_id) is True
    v2.close()


def test_mem0_adapter_default_agent_id_queries_in_single_scope_when_uniform():
    strict = StrictMem0V2Mock()
    adapter = Mem0Adapter(strict, default_agent_id="shared_agent", search_threshold=0.1)
    v = CustodyVault.ephemeral(adapter=adapter)
    v.remember("Priya works in Portland.", subjects=["priya"])
    v.remember("Dana works in Seattle.", subjects=["dana"])
    strict.search_calls.clear()

    res = v.search("Seattle", limit=5)
    assert not res.abstained and len(res.hits) == 1
    assert res.hits[0].text == "Dana works in Seattle."
    assert len(strict.search_calls) == 1
    assert strict.search_calls[0]["filters"] == {"agent_id": "shared_agent"}
    v.close()
