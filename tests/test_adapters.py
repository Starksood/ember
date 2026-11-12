"""Integration tests for Mem0Adapter, VectorStoreAdapter, and SQLiteStoreAdapter."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any
import uuid

import pytest

from ember.adapters import Mem0Adapter, SQLiteStoreAdapter, VectorStoreAdapter
from ember.reducer import (
    ConcurrentModificationError,
    ErasureIncomplete,
    InvariantViolation,
    TombstonedSubjectError,
)
from ember.vault import CustodyVault
from ember.verify import verify_bundle


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


def _make_race_backend(kind: str, tmp_path):
    if kind == "mem0":
        strict = StrictMem0V2Mock()
        return Mem0Adapter(strict, search_threshold=0.1), lambda: [
            row["memory"] for row in strict.rows.values()
        ]
    if kind == "vector":
        vec = FakeVectorStore()
        return VectorStoreAdapter(vec), lambda: [doc.page_content for doc in vec.docs.values()]
    sql = SQLiteStoreAdapter(str(tmp_path / "ext_race.db"))
    return sql, lambda: [
        r[0]
        for r in sql._conn.execute("SELECT text FROM external_memories").fetchall()
    ]


@pytest.mark.parametrize("backend_kind", ["mem0", "vector", "sqlite"])
def test_delayed_summarization_after_erasure_and_retry_cannot_recreate_erased_subject(backend_kind, tmp_path):
    adapter, raw_texts = _make_race_backend(backend_kind, tmp_path)
    vault = CustodyVault.ephemeral(adapter=adapter)
    try:
        vault.register_source(
            "intake",
            "Priya Raman works on triage routing, and prefers async handoffs. Dana Kim owns pharmacy inventory.",
            span_mode="clause",
        )
        r_priya = vault.remember(
            "Priya Raman works on triage routing.",
            subjects=["priya"],
            source_id="intake",
            quote="Priya Raman works on triage routing",
            metadata={"user_id": "priya"},
        )
        r_dana = vault.remember(
            "Dana Kim owns pharmacy inventory.",
            subjects=["dana"],
            source_id="intake",
            quote="Dana Kim owns pharmacy inventory.",
            metadata={"user_id": "dana"},
        )

        # Summarization job reads sources and search hits BEFORE erasure
        with vault.trace_turn():
            pre_hits = vault.search("triage routing pharmacy inventory", limit=5)
            assert len(pre_hits.hits) == 2

            # Erasure completes and issues deletion receipt BEFORE the slow summarization job writes
            cert = vault.erase(
                "priya",
                aliases=["Priya Raman"],
                probes=["Where does Priya work?", "triage routing", "async handoffs"],
            )
            assert cert.battery_all_abstained is True

            # 1) Delayed trace-attached summary + retry
            for _ in range(2):
                with pytest.raises(InvariantViolation):
                    vault.remember(
                        "Clinic handoff note: triage routing updated for next shift.",
                        subjects=["team_acme"],
                        metadata={"user_id": "team_acme"},
                    )

        # 2) Delayed explicit derived_from summary + retry
        for _ in range(2):
            with pytest.raises(InvariantViolation):
                vault.remember(
                    "Cross-team staffing digest",
                    subjects=["team_acme"],
                    derived_from=[r_priya.item.item_id, r_dana.item.item_id],
                    metadata={"user_id": "team_acme"},
                )

        # 3) Delayed source-cited compound span summary + retry (redacted span blocks recreation)
        for _ in range(2):
            with pytest.raises(ValueError):
                vault.remember(
                    "Priya prefers async handoffs",
                    subjects=["team_acme"],
                    source_id="intake",
                    leaf_index=0,
                    metadata={"user_id": "team_acme"},
                )

        # 4) Delayed untagged summary mentioning erased subject content + retry
        for _ in range(2):
            with pytest.raises(TombstonedSubjectError):
                vault.remember(
                    "Shift digest: Priya Raman handles async handoffs.",
                    subjects=["team_acme"],
                    metadata={"user_id": "team_acme"},
                )

        # Raw store must hold ONLY Dana's unrelated content; zero leaked rows for Priya or delayed summaries
        texts = raw_texts()
        assert len(texts) == 1
        assert "Dana Kim" in texts[0]
        assert "Priya" not in " ".join(texts)

        # Retrieval for erased subject abstains while unrelated user content & receipt still verify
        assert vault.search("Priya").abstained
        assert vault.search("triage routing").abstained
        dana_res = vault.search("Dana pharmacy")
        assert not dana_res.abstained
        assert dana_res.hits[0].item_id == r_dana.item.item_id
        assert dana_res.hits[0].receipt_verified is True
        assert vault.verify_item_receipt(r_dana.item.item_id) is True

        audit = verify_bundle(vault.export_bundle())
        assert audit["ok"] is True
    finally:
        vault.close()
        if hasattr(adapter, "close"):
            adapter.close()


class _BlockingMem0Adapter(Mem0Adapter):
    """Wraps Mem0Adapter to simulate an async summarization call landing across an erasure barrier."""

    def __init__(self, client: Any, *, before_write_barrier: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(client, **kwargs)
        self._race = before_write_barrier

    def upsert_detailed(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector=None,
    ):
        if self._race.get("arm_text") and self._race["arm_text"] in text:
            self._race["arm_text"] = None
            self._race["entered"].wait(timeout=5.0)
            self._race["release"].wait(timeout=5.0)
        return super().upsert_detailed(
            item_id=item_id,
            text=text,
            subjects=subjects,
            metadata=metadata,
            vector=vector,
        )


@pytest.mark.parametrize("keystore_backend", ["file", "sqlite"])
def test_cross_handle_midflight_barrier_race_and_retry(keystore_backend, tmp_path):
    vault_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"
    strict = StrictMem0V2Mock()

    race_state: dict[str, Any] = {
        "arm_text": None,
        "entered": threading.Barrier(2),
        "release": threading.Barrier(2),
    }

    worker_adapter = _BlockingMem0Adapter(strict, before_write_barrier=race_state, search_threshold=0.1)
    api_adapter = Mem0Adapter(strict, search_threshold=0.1)

    v_api = CustodyVault.open(
        vault_dir,
        key_dir=key_dir,
        adapter=api_adapter,
        keystore_backend=keystore_backend,
    )
    v_worker = CustodyVault.open(
        vault_dir,
        key_dir=key_dir,
        adapter=worker_adapter,
        keystore_backend=keystore_backend,
    )
    try:
        v_api.register_source(
            "handoff",
            "Priya works on oncology routing. Nina staffs pediatrics desk. Dana covers pharmacy.",
        )
        r_priya = v_api.remember(
            "Priya works on oncology routing.",
            subjects=["priya"],
            source_id="handoff",
            quote="Priya works on oncology routing.",
            metadata={"user_id": "priya"},
        )
        r_nina = v_api.remember(
            "Nina staffs pediatrics desk.",
            subjects=["nina"],
            source_id="handoff",
            quote="Nina staffs pediatrics desk.",
            metadata={"user_id": "nina"},
        )
        r_dana = v_api.remember(
            "Dana covers pharmacy.",
            subjects=["dana"],
            source_id="handoff",
            quote="Dana covers pharmacy.",
            metadata={"user_id": "dana"},
        )

        # Refresh worker handle so it sees pre-erasure state (stale wrt upcoming erase!)
        pre_search = v_worker.search("oncology routing")
        assert len(pre_search.hits) == 1

        # API handle erases Priya while worker handle holds stale state in RAM
        cert = v_api.erase("priya", probes=["Where does Priya work?", "oncology routing"])
        assert cert.battery_all_abstained is True
        # File/SQLite KeyStore shreds must be immediately visible across separate vault handles!
        assert v_worker.keystore.get_subject_key("priya") is None

        # Stale worker attempts delayed write + retry BEFORE entering adapter: rejected without writing a row
        rows_before = len(strict.rows)
        for _ in range(2):
            with pytest.raises(InvariantViolation):
                v_worker.remember(
                    "Delayed routing summary from stale worker",
                    subjects=["team_acme"],
                    derived_from=[r_priya.item.item_id],
                    metadata={"user_id": "team_acme"},
                )
        assert len(strict.rows) == rows_before

        # Unrelated user write on FileKeyStore must NOT resurrect shredded Priya keys on disk
        v_worker.remember("Dana renewed ACLS certification.", subjects=["dana"], metadata={"user_id": "dana"})
        assert v_api.keystore.get_subject_key("priya") is None
        assert v_worker.keystore.get_subject_key("priya") is None

        # True mid-flight Barrier race: worker passes pre-checks BEFORE erase("nina"), pauses mid-adapter,
        # API finishes erase("nina") + issues certificate, worker finishes slow write -> OCC aborts & rolls back!
        race_state["arm_text"] = "Mid-flight summary referencing pediatrics"
        worker_exc: list[BaseException] = []

        def _slow_job() -> None:
            try:
                v_worker.remember(
                    "Mid-flight summary referencing pediatrics desk",
                    subjects=["team_acme"],
                    derived_from=[r_nina.item.item_id],
                    metadata={"user_id": "team_acme"},
                )
            except BaseException as exc:
                worker_exc.append(exc)

        t = threading.Thread(target=_slow_job)
        t.start()
        race_state["entered"].wait(timeout=5.0)
        nina_cert = v_api.erase("nina", probes=["Who staffs pediatrics?", "pediatrics desk"])
        assert nina_cert.battery_all_abstained is True
        race_state["release"].wait(timeout=5.0)
        t.join(timeout=5.0)

        assert len(worker_exc) == 1
        assert isinstance(worker_exc[0], ConcurrentModificationError)
        # Mid-flight row written by slow adapter call MUST have been rolled back immediately
        assert all("Mid-flight summary" not in row["memory"] for row in strict.rows.values())

        # Retry of the delayed mid-flight job ALSO fails closed without touching adapter
        with pytest.raises(InvariantViolation):
            v_worker.remember(
                "Mid-flight summary referencing pediatrics desk",
                subjects=["team_acme"],
                derived_from=[r_nina.item.item_id],
                metadata={"user_id": "team_acme"},
            )
        assert v_api.search("pediatrics").abstained
        assert v_worker.search("pediatrics").abstained

        # Meanwhile a mid-flight bystander retry referencing only Dana succeeds after OCC catch-up!
        race_state["entered"].reset()
        race_state["release"].reset()
        v_api.remember("Rohan manages lab supplies.", subjects=["rohan"], metadata={"user_id": "rohan"})
        race_state["arm_text"] = "Dana bystander concurrent note"
        bystander_exc: list[BaseException] = []

        def _slow_bystander() -> None:
            try:
                v_worker.remember(
                    "Dana bystander concurrent note",
                    subjects=["dana"],
                    derived_from=[r_dana.item.item_id],
                    metadata={"user_id": "dana"},
                )
            except BaseException as exc:
                bystander_exc.append(exc)

        t2 = threading.Thread(target=_slow_bystander)
        t2.start()
        race_state["entered"].wait(timeout=5.0)
        v_api.erase("rohan", probes=["Who manages lab supplies?"])
        race_state["release"].wait(timeout=5.0)
        t2.join(timeout=5.0)
        assert len(bystander_exc) == 1 and isinstance(bystander_exc[0], ConcurrentModificationError)

        # Retry of unrelated bystander write succeeds cleanly!
        retried_dana = v_worker.remember(
            "Dana bystander concurrent note",
            subjects=["dana"],
            derived_from=[r_dana.item.item_id],
            metadata={"user_id": "dana"},
        )
        assert not v_api.search("bystander concurrent").abstained
        assert retried_dana.item.subjects == ("dana",)
        assert v_api.verify_item_receipt(r_dana.item.item_id) is True
    finally:
        v_worker.close()
        v_api.close()


def test_committed_pre_erasure_event_id_retry_after_erasure_cannot_recreate_record():
    strict = StrictMem0V2Mock()
    vault = CustodyVault.ephemeral(adapter=Mem0Adapter(strict, search_threshold=0.1))
    try:
        r_priya = vault.remember(
            "Priya works on billing.",
            subjects=["priya"],
            metadata={"user_id": "priya"},
        )
        vault.remember(
            "Dana leads ops.",
            subjects=["dana"],
            metadata={"user_id": "dana"},
            event_id="job-dana-1",
        )
        rows_before_retry = len(strict.rows)
        # Live idempotent event_id retry must NOT create duplicate vendor records
        idem_res = vault.remember(
            "Dana leads ops.",
            subjects=["dana"],
            metadata={"user_id": "dana"},
            event_id="job-dana-1",
        )
        assert len(strict.rows) == rows_before_retry
        assert idem_res.item.text == "Dana leads ops."

        vault.remember(
            "Billing coverage shift note",
            subjects=["team_acme"],
            derived_from=[r_priya.item.item_id],
            metadata={"user_id": "team_acme"},
            event_id="job-sum-1",
        )

        vault.erase("priya", probes=["billing", "Priya"])
        rows_after_erase = len(strict.rows)
        assert rows_after_erase == 1

        # Delayed queue retry of committed pre-erasure event_id must fail WITHOUT recreating Mem0 row
        for kwargs in (
            {"derived_from": [r_priya.item.item_id]},
            {},
        ):
            with pytest.raises(TombstonedSubjectError):
                vault.remember(
                    "Billing coverage shift note",
                    subjects=["team_acme"],
                    metadata={"user_id": "team_acme"},
                    event_id="job-sum-1",
                    **kwargs,
                )
        assert len(strict.rows) == rows_after_erase
        assert vault.search("Billing coverage").abstained
        assert not vault.search("Dana leads").abstained
    finally:
        vault.close()
