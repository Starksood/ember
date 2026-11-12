"""One-line framework integrations: EmberMem0, LangChain, and LlamaIndex drop-ins.

Designed with zero required third-party runtime dependencies:
  * If `langchain_core` or `llama_index.core` is installed, Ember's wrappers subclass their
    native base classes and emit real framework `Document` / `NodeWithScore` objects.
  * When omitted, compatible lightweight shims expose the exact `.invoke()` / `.ainvoke()` /
    `.retrieve()` / `.aretrieve()` / `.similarity_search()` APIs so apps and tests run clean.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import Mem0Adapter
from .coverage import CoverageReport
from .records import ErasureCertificate, PendingErasureRecord, ReconcileReport
from .signing import Ed25519Signer, TimestampAnchor
from .vault import CustodyVault, QueryHit


def _extract_mem0_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Mapping):
        content = messages.get("content") or messages.get("text") or messages.get("memory")
        return str(content) if content is not None else str(messages)
    if isinstance(messages, Sequence):
        parts: list[str] = []
        for entry in messages:
            if isinstance(entry, str) and entry.strip():
                parts.append(entry.strip())
            elif isinstance(entry, Mapping):
                content = entry.get("content") or entry.get("text") or entry.get("memory")
                if content is not None and str(content).strip():
                    parts.append(str(content).strip())
        return "\n".join(parts)
    return str(messages)


class EmberMem0:
    """1-line Mem0 drop-in proxy providing cryptographic custody, receipts, and verified cascade erasure."""

    def __init__(self, vault: CustodyVault) -> None:
        self.vault = vault

    @classmethod
    def wrap(
        cls,
        client: Any,
        *,
        data_dir: str | Path,
        key_dir: str | Path | None = None,
        tenant_id: str = "default",
        default_user_id: str | None = None,
        default_agent_id: str | None = None,
        default_run_id: str | None = None,
        infer: bool = False,
        search_threshold: float | None = None,
        encrypt_vendor_payload: bool = False,
        embedder: Any = None,
        signer: Ed25519Signer | None = None,
        anchor: TimestampAnchor | None = None,
    ) -> "EmberMem0":
        """Wrap an existing `mem0.Memory` or `mem0.MemoryClient` in a single persistent Ember vault."""
        adapter = Mem0Adapter(
            client,
            default_user_id=default_user_id,
            default_agent_id=default_agent_id,
            default_run_id=default_run_id,
            infer=infer,
            search_threshold=search_threshold,
        )
        vault = CustodyVault.open(
            data_dir,
            key_dir=key_dir,
            tenant_id=tenant_id,
            adapter=adapter,
            signer=signer,
            anchor=anchor,
            encrypt_vendor_payload=encrypt_vendor_payload,
            embedder=embedder,
        )
        return cls(vault)

    @classmethod
    def ephemeral(
        cls,
        client: Any,
        *,
        tenant_id: str = "default",
        default_user_id: str | None = None,
        default_agent_id: str | None = None,
        default_run_id: str | None = None,
        infer: bool = False,
        search_threshold: float | None = None,
        encrypt_vendor_payload: bool = False,
        embedder: Any = None,
        signer: Ed25519Signer | None = None,
        anchor: TimestampAnchor | None = None,
    ) -> "EmberMem0":
        """Wrap a `mem0` client with an ephemeral in-memory ledger & keystore."""
        adapter = Mem0Adapter(
            client,
            default_user_id=default_user_id,
            default_agent_id=default_agent_id,
            default_run_id=default_run_id,
            infer=infer,
            search_threshold=search_threshold,
        )
        vault = CustodyVault.ephemeral(
            tenant_id=tenant_id,
            adapter=adapter,
            signer=signer,
            anchor=anchor,
            encrypt_vendor_payload=encrypt_vendor_payload,
            embedder=embedder,
        )
        return cls(vault)

    def __enter__(self) -> "EmberMem0":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.vault.close()

    def trace_turn(self, *, auto_cite: bool = True, prune: Any = "all"):
        return self.vault.trace_turn(auto_cite=auto_cite, prune=prune)

    def register_source(self, source_id: str, text: str, **kwargs: Any):
        return self.vault.register_source(source_id, text, **kwargs)

    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        infer: bool | None = None,
        subjects: Iterable[str] | None = None,
        source_id: str | None = None,
        quote: str | None = None,
        derived_from: Iterable[str] = (),
        aliases: Iterable[str] = (),
        reconsent_id: str | None = None,
        auto_derive: bool = True,
    ) -> dict[str, Any]:
        del infer  # Adapter's `infer` policy controls Mem0 add semantics cleanly.
        text = _extract_mem0_text(messages)
        meta = dict(metadata or {})
        if user_id is not None:
            meta["user_id"] = user_id
        if agent_id is not None:
            meta["agent_id"] = agent_id
        if run_id is not None:
            meta["run_id"] = run_id

        if subjects is not None:
            resolved_subjects = tuple(str(s) for s in subjects if str(s).strip())
        else:
            candidates = [user_id, agent_id, run_id]
            resolved_subjects = tuple(str(x) for x in candidates if x is not None and str(x).strip())
            if not resolved_subjects and isinstance(self.vault.adapter, Mem0Adapter):
                fallbacks = [
                    self.vault.adapter.default_user_id,
                    self.vault.adapter.default_agent_id,
                    self.vault.adapter.default_run_id,
                ]
                resolved_subjects = tuple(str(x) for x in fallbacks if x is not None and str(x).strip())
        if not resolved_subjects:
            raise ValueError("EmberMem0.add() requires subjects, user_id, agent_id, or run_id")

        res = self.vault.remember(
            text,
            subjects=resolved_subjects,
            source_id=source_id,
            quote=quote,
            derived_from=derived_from,
            aliases=aliases,
            metadata=meta,
            reconsent_id=reconsent_id,
            auto_derive=auto_derive,
        )
        return {
            "results": [
                {
                    "id": ext_id,
                    "memory": res.item.text,
                    "event": "ADD",
                    "ember_item_id": res.item.item_id,
                }
                for ext_id in res.adapter_record_ids
            ],
            "ember_item_id": res.item.item_id,
            "ember_derived_from": list(res.item.derived_from),
            "ember_event_hash": res.event_hash,
            "ember_receipt": res.receipt.as_dict() if res.receipt else None,
        }

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List live non-erased memories matching scope filters from materialized custody state."""
        from .crypto import ERASED_TOMBSTONE

        results: list[dict[str, Any]] = []
        with self.vault._lock:
            for item in self.vault.state.items.values():
                if item.text is None or item.text == ERASED_TOMBSTONE:
                    continue
                meta = item.metadata or {}
                item_uid = meta.get("user_id") or (item.subjects[0] if item.subjects else None)
                if user_id is not None and item_uid != user_id and user_id not in item.subjects:
                    continue
                if agent_id is not None and meta.get("agent_id") != agent_id and agent_id not in item.subjects:
                    continue
                if run_id is not None and meta.get("run_id") != run_id and run_id not in item.subjects:
                    continue
                results.append(
                    {
                        "id": item.adapter_record_ids[0] if item.adapter_record_ids else item.item_id,
                        "memory": item.text,
                        "metadata": dict(meta),
                        "ember_item_id": item.item_id,
                        "ember_subjects": list(item.subjects),
                        "ember_derived_from": list(item.derived_from),
                    }
                )
                if len(results) >= max(0, int(limit)):
                    break
        return {"results": results}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int = 5,
        top_k: int | None = None,
        verify_receipts: bool = True,
    ) -> dict[str, Any]:
        eff_filters = dict(filters or {})
        if user_id is not None:
            eff_filters.setdefault("user_id", user_id)
        if agent_id is not None:
            eff_filters.setdefault("agent_id", agent_id)
        if run_id is not None:
            eff_filters.setdefault("run_id", run_id)
        eff_limit = int(top_k if top_k is not None else limit)

        res = self.vault.search(
            query,
            limit=eff_limit,
            filters=eff_filters or None,
            verify_receipts=verify_receipts,
        )
        return {
            "results": [
                {
                    "id": h.record_id,
                    "memory": h.text,
                    "score": h.score,
                    "metadata": h.metadata,
                    "ember_item_id": h.item_id,
                    "ember_subjects": list(h.subjects),
                    "ember_receipt_verified": h.receipt_verified,
                    "ember_receipt": h.receipt.as_dict() if h.receipt else None,
                }
                for h in res.hits
            ],
            "filtered_orphans": res.filtered_orphans,
        }

    def erase(
        self,
        subject_id: str,
        *,
        aliases: Iterable[str] = (),
        probes: Iterable[str] = (),
        markers: Iterable[str] = (),
        **kwargs: Any,
    ) -> ErasureCertificate:
        return self.vault.erase(
            subject_id,
            aliases=aliases,
            probes=probes,
            markers=markers,
            **kwargs,
        )

    def delete_user(self, user_id: str, **kwargs: Any) -> ErasureCertificate:
        return self.erase(user_id, **kwargs)

    def delete_all(self, *, user_id: str | None = None, **kwargs: Any) -> ErasureCertificate:
        if not user_id or not str(user_id).strip():
            raise ValueError("EmberMem0.delete_all() requires user_id for targeted cryptographic erasure")
        return self.erase(str(user_id), **kwargs)

    def audit_coverage(
        self,
        subject_id: str | None = None,
        *,
        unverified_locations: Iterable[str] = (),
    ) -> CoverageReport:
        return self.vault.audit_coverage(subject_id=subject_id, unverified_locations=unverified_locations)

    def pending_erasures(self) -> list[PendingErasureRecord]:
        return self.vault.pending_erasures()

    def reconcile_erasures(self) -> ReconcileReport:
        return self.vault.reconcile_erasures()


# ── Optional LangChain Base / Document Resolution ────────────────────────────

try:
    from langchain_core.documents import Document as _LCDocument
    from langchain_core.retrievers import BaseRetriever as _LCBaseRetriever
    from langchain_core.vectorstores import VectorStore as _LCVectorStore

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


@dataclass
class EmberDocument:
    """Lightweight LangChain-compatible Document fallback when `langchain_core` is absent."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


def _to_lc_document(hit: QueryHit) -> Any:
    meta = dict(hit.metadata)
    meta["ember_item_id"] = hit.item_id
    meta["ember_record_id"] = hit.record_id
    meta["ember_subjects"] = list(hit.subjects)
    meta["ember_score"] = hit.score
    if hit.receipt is not None:
        meta["ember_receipt"] = hit.receipt.as_dict()
    if hit.receipt_verified is not None:
        meta["ember_receipt_verified"] = hit.receipt_verified
    if _HAS_LANGCHAIN:
        return _LCDocument(page_content=hit.text, metadata=meta, id=hit.item_id)
    return EmberDocument(page_content=hit.text, metadata=meta, id=hit.item_id)


if _HAS_LANGCHAIN:

    class EmberLangChainRetriever(_LCBaseRetriever):  # type: ignore[misc]
        """Native `langchain_core.retrievers.BaseRetriever` backed by `CustodyVault`."""

        vault: Any
        k: int = 4
        default_filters: dict[str, Any] | None = None
        verify_receipts: bool = True

        model_config = {"arbitrary_types_allowed": True}

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            del run_manager
            v = getattr(self.vault, "_vault", self.vault)
            res = v.search(
                query,
                limit=self.k,
                filters=self.default_filters,
                verify_receipts=self.verify_receipts,
            )
            return [_to_lc_document(hit) for hit in res.hits]

        async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Any]:
            del run_manager
            return await asyncio.to_thread(self._get_relevant_documents, query)

else:

    class EmberLangChainRetriever:  # type: ignore[no-redef]
        """LangChain-compatible `BaseRetriever` drop-in (`invoke`, `ainvoke`, `get_relevant_documents`)."""

        def __init__(
            self,
            vault: Any,
            *,
            k: int = 4,
            default_filters: dict[str, Any] | None = None,
            verify_receipts: bool = True,
        ) -> None:
            self.vault = getattr(vault, "_vault", vault)
            self.k = k
            self.default_filters = dict(default_filters) if default_filters else None
            self.verify_receipts = verify_receipts

        def get_relevant_documents(self, query: str, *, filter: dict[str, Any] | None = None) -> list[Any]:
            merged_filters = dict(self.default_filters or {})
            if filter:
                merged_filters.update(filter)
            res = self.vault.search(
                query,
                limit=self.k,
                filters=merged_filters or None,
                verify_receipts=self.verify_receipts,
            )
            return [_to_lc_document(hit) for hit in res.hits]

        def invoke(self, input: str, config: Any = None, **kwargs: Any) -> list[Any]:  # noqa: A002
            del config
            return self.get_relevant_documents(str(input), filter=kwargs.get("filter"))

        async def ainvoke(self, input: str, config: Any = None, **kwargs: Any) -> list[Any]:  # noqa: A002
            del config
            return await asyncio.to_thread(self.get_relevant_documents, str(input), filter=kwargs.get("filter"))


_LCVectorStoreBase = _LCVectorStore if _HAS_LANGCHAIN else object


class EmberLangChainVectorStore(_LCVectorStoreBase):  # type: ignore[misc,valid-type]
    """Drop-in LangChain `VectorStore` delegating storage, search, and erasure to `CustodyVault`."""

    def __init__(
        self,
        vault: Any,
        *,
        subject_key: str = "user_id",
        default_subject: str | None = None,
        verify_receipts: bool = True,
    ) -> None:
        self.vault = getattr(vault, "_vault", vault)
        self.subject_key = subject_key
        self.default_subject = default_subject
        self.verify_receipts = verify_receipts

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        subjects: Iterable[str] | None = None,
        derived_from: Iterable[str] = (),
        **kwargs: Any,
    ) -> list[str]:
        del kwargs
        text_list = list(texts)
        metas = metadatas or [{} for _ in text_list]
        out_ids: list[str] = []
        for idx, text in enumerate(text_list):
            meta = dict(metas[idx]) if idx < len(metas) else {}
            item_subjects = list(subjects) if subjects is not None else []
            if not item_subjects:
                subj_val = meta.get(self.subject_key) or meta.get("subject")
                if isinstance(subj_val, str) and subj_val.strip():
                    item_subjects.append(subj_val.strip())
                elif isinstance(meta.get("subjects"), (list, tuple)):
                    item_subjects.extend(str(s) for s in meta["subjects"] if str(s).strip())
                elif self.default_subject:
                    item_subjects.append(self.default_subject)
            if not item_subjects:
                raise ValueError(
                    f"EmberLangChainVectorStore.add_texts() could not resolve subjects from "
                    f"metadata[{self.subject_key!r}], subjects=, or default_subject="
                )
            res = self.vault.remember(
                text,
                subjects=item_subjects,
                source_id=meta.get("source_id"),
                quote=meta.get("quote"),
                derived_from=derived_from,
                metadata=meta or None,
                item_id=ids[idx] if (ids is not None and idx < len(ids)) else None,
            )
            out_ids.append(res.item.item_id)
        return out_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[tuple[Any, float]]:
        del kwargs
        res = self.vault.search(
            query,
            limit=k,
            filters=filter,
            verify_receipts=self.verify_receipts,
        )
        return [(_to_lc_document(hit), float(hit.score)) for hit in res.hits]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> list[Any]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter, **kwargs)]

    def erase(self, subject_id: str, **kwargs: Any) -> ErasureCertificate:
        return self.vault.erase(subject_id, **kwargs)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        subject_id = kwargs.pop("subject_id", None)
        if subject_id:
            self.vault.erase(str(subject_id), **kwargs)
            return True
        if ids is None:
            return False
        raise ValueError(
            "EmberLangChainVectorStore requires erase(subject_id) or delete(subject_id=...) "
            "so cryptographic DEKs are shredded and the custody ledger stays consistent."
        )

    def as_retriever(self, **kwargs: Any) -> EmberLangChainRetriever:
        search_kwargs = dict(kwargs.get("search_kwargs") or {})
        k = int(search_kwargs.get("k", kwargs.get("k", 4)))
        flt = search_kwargs.get("filter", kwargs.get("filter"))
        return EmberLangChainRetriever(
            vault=self.vault,
            k=k,
            default_filters=flt,
            verify_receipts=self.verify_receipts,
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any = None,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "EmberLangChainVectorStore":
        vault = kwargs.pop("vault", None)
        if vault is None:
            vault = CustodyVault.ephemeral(embedder=embedding)
        store = cls(vault, **{k: v for k, v in kwargs.items() if k in ("subject_key", "default_subject", "verify_receipts")})
        store.add_texts(texts, metadatas=metadatas)
        return store


# ── Optional LlamaIndex BaseRetriever / NodeWithScore Wrapper ────────────────

try:
    from llama_index.core.retrievers import BaseRetriever as _LIBaseRetriever
    from llama_index.core.schema import NodeWithScore as _LINodeWithScore
    from llama_index.core.schema import QueryBundle as _LIQueryBundle
    from llama_index.core.schema import TextNode as _LITextNode

    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False


@dataclass
class EmberTextNode:
    """Lightweight LlamaIndex-compatible TextNode fallback when `llama_index.core` is absent."""

    text: str
    id_: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return self.id_

    def get_content(self) -> str:
        return self.text


@dataclass
class EmberNodeWithScore:
    """Lightweight LlamaIndex-compatible NodeWithScore fallback."""

    node: EmberTextNode
    score: float

    @property
    def text(self) -> str:
        return self.node.text


def _to_li_node(hit: QueryHit) -> Any:
    meta = dict(hit.metadata)
    meta["ember_item_id"] = hit.item_id
    meta["ember_subjects"] = list(hit.subjects)
    if hit.receipt is not None:
        meta["ember_receipt"] = hit.receipt.as_dict()
    if hit.receipt_verified is not None:
        meta["ember_receipt_verified"] = hit.receipt_verified
    if _HAS_LLAMAINDEX:
        node = _LITextNode(text=hit.text, id_=hit.item_id, metadata=meta)
        return _LINodeWithScore(node=node, score=float(hit.score))
    return EmberNodeWithScore(
        node=EmberTextNode(text=hit.text, id_=hit.item_id, metadata=meta),
        score=float(hit.score),
    )


_LIBase = _LIBaseRetriever if _HAS_LLAMAINDEX else object


class EmberLlamaIndexRetriever(_LIBase):  # type: ignore[misc,valid-type]
    """Drop-in LlamaIndex `BaseRetriever` integrating custody search & turn lineage."""

    def __init__(
        self,
        vault: Any,
        *,
        similarity_top_k: int = 5,
        filters: dict[str, Any] | None = None,
        verify_receipts: bool = True,
    ) -> None:
        if _HAS_LLAMAINDEX:
            super().__init__()
        self.vault = getattr(vault, "_vault", vault)
        self.similarity_top_k = similarity_top_k
        self.filters = dict(filters) if filters else None
        self.verify_receipts = verify_receipts

    def _query_str(self, str_or_query_bundle: Any) -> str:
        if _HAS_LLAMAINDEX and isinstance(str_or_query_bundle, _LIQueryBundle):
            return str_or_query_bundle.query_str
        return str(getattr(str_or_query_bundle, "query_str", str_or_query_bundle))

    def _retrieve(self, query_bundle: Any) -> list[Any]:
        q_str = self._query_str(query_bundle)
        res = self.vault.search(
            q_str,
            limit=self.similarity_top_k,
            filters=self.filters,
            verify_receipts=self.verify_receipts,
        )
        return [_to_li_node(hit) for hit in res.hits]

    def retrieve(self, str_or_query_bundle: Any) -> list[Any]:
        return self._retrieve(str_or_query_bundle)

    async def aretrieve(self, str_or_query_bundle: Any) -> list[Any]:
        return await asyncio.to_thread(self._retrieve, str_or_query_bundle)


__all__ = [
    "EmberDocument",
    "EmberLangChainRetriever",
    "EmberLangChainVectorStore",
    "EmberLlamaIndexRetriever",
    "EmberMem0",
    "EmberNodeWithScore",
    "EmberTextNode",
]
