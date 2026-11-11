"""Pluggable store adapters for Ember cryptographic custody.

Enables developers to wrap an existing memory engine or vector store (Mem0, LangChain/Chroma/
Qdrant-style stores, SQLite keyword stores) without rewriting their application interface.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import inspect
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Protocol, runtime_checkable
import uuid


@dataclass(frozen=True)
class AdapterHit:
    """Normalized record returned by any MemoryStoreAdapter search call."""

    record_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryStoreAdapter(Protocol):
    """Minimal contract any backing vector store or agent memory engine satisfies."""

    name: str

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[str, ...]: ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]: ...

    def delete(self, record_ids: Sequence[str]) -> int: ...


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "based",
        "before",
        "did",
        "do",
        "does",
        "everyone",
        "how",
        "in",
        "is",
        "list",
        "living",
        "many",
        "me",
        "of",
        "people",
        "tell",
        "the",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "work",
        "works",
    }
)


def _content_tokens(text: str) -> set[str]:
    toks = _tokenize(text)
    stripped = toks - _STOPWORDS
    return stripped or toks


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        if metadata.get(k) != v:
            return False
    return True


class InMemoryStoreAdapter:
    """Deterministic keyword-overlap store useful for unit tests, benchmarks, and baselines."""

    name = "in-memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[str, ...]:
        rec_id = f"mem_{uuid.uuid4().hex[:12]}"
        meta = dict(metadata)
        meta.setdefault("ember_item_id", item_id)
        if subjects and "user_id" not in meta:
            meta["user_id"] = subjects[0]
        with self._lock:
            self._records[rec_id] = {
                "record_id": rec_id,
                "text": text,
                "subjects": list(subjects),
                "metadata": meta,
            }
        return (rec_id,)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        q_toks = _content_tokens(query)
        scored: list[AdapterHit] = []
        with self._lock:
            for rec in self._records.values():
                if not _matches_filters(rec["metadata"], filters):
                    continue
                d_toks = _tokenize(rec["text"])
                overlap = len(q_toks & d_toks)
                if overlap == 0 and query.strip().lower() not in rec["text"].lower():
                    continue
                score = overlap / max(1, len(q_toks))
                scored.append(
                    AdapterHit(
                        record_id=rec["record_id"],
                        text=rec["text"],
                        score=score,
                        metadata=dict(rec["metadata"]),
                    )
                )
        scored.sort(key=lambda h: (-h.score, h.record_id))
        return scored[:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        removed = 0
        with self._lock:
            for rid in record_ids:
                if self._records.pop(rid, None) is not None:
                    removed += 1
        return removed

    def delete_by_user_id_naive(self, user_id: str) -> int:
        """Deliberately naive user_id-only delete used as baseline control in benchmarks."""
        with self._lock:
            doomed = [
                rid
                for rid, rec in self._records.items()
                if rec["metadata"].get("user_id") == user_id
            ]
            for rid in doomed:
                del self._records[rid]
            return len(doomed)

    def all_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records.values()]


class SQLiteStoreAdapter:
    """Persistent single-file SQLite store implementing `MemoryStoreAdapter` with zero extra deps."""

    name = "sqlite-store"

    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS external_memories (
                record_id TEXT PRIMARY KEY,
                item_id   TEXT NOT NULL,
                text      TEXT NOT NULL,
                meta_json TEXT NOT NULL
            )
            """
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[str, ...]:
        import json

        rec_id = f"sql_{uuid.uuid4().hex[:12]}"
        meta = dict(metadata)
        meta.setdefault("ember_item_id", item_id)
        if subjects and "user_id" not in meta:
            meta["user_id"] = subjects[0]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO external_memories (record_id, item_id, text, meta_json) VALUES (?, ?, ?, ?)",
                    (rec_id, item_id, text, json.dumps(meta, sort_keys=True)),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return (rec_id,)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        import json

        if limit <= 0:
            return []
        q_toks = _content_tokens(query)
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, text, meta_json FROM external_memories ORDER BY rowid ASC"
            ).fetchall()
        scored: list[AdapterHit] = []
        for rec_id, text, meta_json in rows:
            meta = json.loads(meta_json)
            if not _matches_filters(meta, filters):
                continue
            overlap = len(q_toks & _tokenize(text))
            if overlap == 0 and query.strip().lower() not in text.lower():
                continue
            scored.append(
                AdapterHit(
                    record_id=rec_id,
                    text=text,
                    score=overlap / max(1, len(q_toks)),
                    metadata=meta,
                )
            )
        scored.sort(key=lambda h: (-h.score, h.record_id))
        return scored[:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        if not record_ids:
            return 0
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                before = self._conn.total_changes
                self._conn.executemany(
                    "DELETE FROM external_memories WHERE record_id = ?",
                    [(rid,) for rid in record_ids],
                )
                deleted = max(0, self._conn.total_changes - before)
                self._conn.execute("COMMIT")
                return deleted
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise


class Mem0Adapter:
    """Drop-in custody adapter wrapping `mem0.Memory` or `mem0.MemoryClient`.

    Tracks every returned Mem0 `memory_id` in Ember's structural closure so erasing a
    subject deletes direct facts, shared workspace notes, and downstream derived
    summaries across Mem0 regardless of `user_id` bucket fragmentation.
    """

    name = "mem0"

    def __init__(
        self,
        client: Any,
        *,
        default_user_id: str | None = None,
        default_agent_id: str | None = None,
        infer: bool = False,
    ) -> None:
        self.client = client
        self.default_user_id = default_user_id
        self.default_agent_id = default_agent_id
        self.infer = infer

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[str, ...]:
        meta = dict(metadata)
        meta["ember_item_id"] = item_id
        meta["ember_subjects"] = list(subjects)

        kwargs: dict[str, Any] = {"metadata": meta, "infer": self.infer}
        user_id = meta.get("user_id") or self.default_user_id or (subjects[0] if subjects else None)
        agent_id = meta.get("agent_id") or self.default_agent_id
        if user_id is not None:
            kwargs["user_id"] = user_id
        if agent_id is not None:
            kwargs["agent_id"] = agent_id

        sig = inspect.signature(self.client.add)
        accepted = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted or has_varkw}
        if "messages" in accepted:
            raw = self.client.add(messages=text, **filtered_kwargs)
        else:
            raw = self.client.add(text, **filtered_kwargs)

        if raw is None:
            return ()
        rows = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        ids: list[str] = []
        for entry in rows:
            if isinstance(entry, dict) and entry.get("id"):
                ids.append(str(entry["id"]))
        if not ids and isinstance(raw, dict) and raw.get("id"):
            ids.append(str(raw["id"]))
        return tuple(ids)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        sig = inspect.signature(self.client.search)
        params = set(sig.parameters)
        kwargs: dict[str, Any] = {}
        if "limit" in params:
            kwargs["limit"] = limit
        if filters:
            if "filters" in params:
                kwargs["filters"] = filters
            else:
                for k, v in filters.items():
                    if k in params:
                        kwargs[k] = v
        raw = self.client.search(query, **kwargs)
        if raw is None:
            return []
        rows = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        hits: list[AdapterHit] = []
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            hits.append(
                AdapterHit(
                    record_id=str(entry.get("id", "")),
                    text=str(entry.get("memory") or entry.get("text") or entry.get("content") or ""),
                    score=float(entry.get("score", 0.0) or 0.0),
                    metadata=dict(entry.get("metadata") or {}),
                )
            )
        return hits[:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        deleted = 0
        for rid in record_ids:
            if not rid:
                continue
            try:
                sig = inspect.signature(self.client.delete)
                if "memory_id" in sig.parameters:
                    self.client.delete(memory_id=rid)
                else:
                    self.client.delete(rid)
                deleted += 1
            except TypeError:
                self.client.delete(rid)
                deleted += 1
        return deleted


class VectorStoreAdapter:
    """Adapter for LangChain / Chroma / Qdrant-compatible `.add_texts` / `.similarity_search` / `.delete`."""

    name = "vector-store"

    def __init__(self, store: Any) -> None:
        self.store = store

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> tuple[str, ...]:
        meta = {**metadata, "ember_item_id": item_id, "ember_subjects": list(subjects)}
        rec_id = f"vec_{uuid.uuid4().hex[:12]}"
        sig = inspect.signature(self.store.add_texts)
        params = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs: dict[str, Any] = {}
        if "metadatas" in params or has_varkw:
            kwargs["metadatas"] = [meta]
        if "ids" in params or has_varkw:
            kwargs["ids"] = [rec_id]
        returned = self.store.add_texts(texts=[text], **kwargs)
        if isinstance(returned, (list, tuple)) and returned:
            return tuple(str(x) for x in returned)
        return (rec_id,) if ("ids" in kwargs) else ()

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        if hasattr(self.store, "similarity_search_with_score"):
            sig = inspect.signature(self.store.similarity_search_with_score)
            params = set(sig.parameters)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            kwargs: dict[str, Any] = {}
            if "k" in params or has_varkw:
                kwargs["k"] = limit
            if filters is not None and ("filter" in params or has_varkw):
                kwargs["filter"] = filters
            pairs = self.store.similarity_search_with_score(query, **kwargs)
            hits: list[AdapterHit] = []
            for doc, score in pairs:
                meta = dict(getattr(doc, "metadata", {}) or {})
                hits.append(
                    AdapterHit(
                        record_id=str(getattr(doc, "id", None) or meta.get("id") or meta.get("ember_item_id") or ""),
                        text=str(getattr(doc, "page_content", "")),
                        score=float(score),
                        metadata=meta,
                    )
                )
            return hits[:limit]
        sig = inspect.signature(self.store.similarity_search)
        params = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs = {}
        if "k" in params or has_varkw:
            kwargs["k"] = limit
        if filters is not None and ("filter" in params or has_varkw):
            kwargs["filter"] = filters
        docs = self.store.similarity_search(query, **kwargs)
        return [
            AdapterHit(
                record_id=str(getattr(doc, "id", None) or (getattr(doc, "metadata", {}) or {}).get("ember_item_id", "")),
                text=str(getattr(doc, "page_content", "")),
                score=1.0,
                metadata=dict(getattr(doc, "metadata", {}) or {}),
            )
            for doc in docs
        ][:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(rid for rid in record_ids if rid))
        if not ids:
            return 0
        sig = inspect.signature(self.store.delete)
        params = set(sig.parameters)
        if "ids" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            self.store.delete(ids=ids)
        else:
            self.store.delete(ids)
        return len(ids)


ProbeFilterFn = Callable[[AdapterHit], bool]
