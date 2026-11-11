"""Pluggable store adapters for Ember cryptographic custody.

Enables developers to wrap an existing memory engine or vector store (Mem0, LangChain/Chroma/
Qdrant-style stores, SQLite keyword stores) without rewriting their application interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
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


@dataclass(frozen=True)
class AdapterOp:
    """Lifecycle operation emitted by adapters supporting in-place mutations/consolidations."""

    record_id: str
    event: str = "ADD"  # "ADD", "UPDATE", "DELETE", or "NONE"
    memory: str = ""
    previous_memory: str | None = None


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


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
    """Deterministic keyword-overlap & dense-vector store useful for tests, benchmarks, and baselines."""

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
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        rec_id = f"mem_{uuid.uuid4().hex[:12]}"
        meta = dict(metadata)
        meta.setdefault("ember_item_id", item_id)
        if subjects and "user_id" not in meta and not meta.get("ember_encrypted"):
            meta["user_id"] = subjects[0]
        with self._lock:
            self._records[rec_id] = {
                "record_id": rec_id,
                "text": text,
                "subjects": list(subjects),
                "metadata": meta,
                "vector": [float(x) for x in vector] if vector is not None else None,
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

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        q_vec = [float(x) for x in query_vector]
        scored: list[AdapterHit] = []
        with self._lock:
            for rec in self._records.values():
                if not _matches_filters(rec["metadata"], filters):
                    continue
                rec_vec = rec.get("vector")
                if not rec_vec:
                    continue
                sim = _cosine_similarity(q_vec, rec_vec)
                if sim <= min_score:
                    continue
                scored.append(
                    AdapterHit(
                        record_id=rec["record_id"],
                        text=rec["text"],
                        score=sim,
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
                record_id   TEXT PRIMARY KEY,
                item_id     TEXT NOT NULL,
                text        TEXT NOT NULL,
                meta_json   TEXT NOT NULL,
                vector_json TEXT
            )
            """
        )
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(external_memories)").fetchall()}
        if "vector_json" not in cols:
            self._conn.execute("ALTER TABLE external_memories ADD COLUMN vector_json TEXT")

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
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        rec_id = f"sql_{uuid.uuid4().hex[:12]}"
        meta = dict(metadata)
        meta.setdefault("ember_item_id", item_id)
        if subjects and "user_id" not in meta and not meta.get("ember_encrypted"):
            meta["user_id"] = subjects[0]
        vec_json = json.dumps([float(x) for x in vector]) if vector is not None else None
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO external_memories (record_id, item_id, text, meta_json, vector_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (rec_id, item_id, text, json.dumps(meta, sort_keys=True), vec_json),
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

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        q_vec = [float(x) for x in query_vector]
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id, text, meta_json, vector_json FROM external_memories "
                "WHERE vector_json IS NOT NULL ORDER BY rowid ASC"
            ).fetchall()
        scored: list[AdapterHit] = []
        for rec_id, text, meta_json, vec_json in rows:
            meta = json.loads(meta_json)
            if not _matches_filters(meta, filters):
                continue
            rec_vec = json.loads(vec_json)
            sim = _cosine_similarity(q_vec, rec_vec)
            if sim <= min_score:
                continue
            scored.append(
                AdapterHit(
                    record_id=rec_id,
                    text=text,
                    score=sim,
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


def _normalize_mem0_entity_id(value: Any) -> str | None:
    """Trim and sanitize Mem0 entity IDs (`user_id`, `agent_id`, `run_id`) when needed.

    Mem0 2.x rejects any identifier containing whitespace (`_validate_and_trim_entity_id`).
    IDs without internal whitespace pass through unchanged so exact names (`"priya"`,
    `"team_acme"`) match native callers; whitespace-containing IDs (`"Kirin Press"`) map
    deterministically to a collision-resistant slug (`"Kirin_Press__<8hex>"`).
    """
    if value is None:
        return None
    trimmed = str(value).strip()
    if not trimmed:
        return None
    if not any(ch.isspace() for ch in trimmed):
        return trimmed
    slug = re.sub(r"\s+", "_", trimmed)
    digest = hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def _promote_mem0_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Combine entry `metadata` with top-level Mem0 identity fields without losing keys."""
    meta = dict(entry.get("metadata") or {})
    for key in ("user_id", "agent_id", "run_id", "actor_id"):
        if key not in meta and entry.get(key) is not None:
            meta[key] = entry[key]
    return meta


class Mem0Adapter:
    """Drop-in custody adapter wrapping `mem0.Memory` or `mem0.MemoryClient`.

    Tracks every returned Mem0 `memory_id` in Ember's structural closure so erasing a
    subject deletes direct facts, shared workspace notes, and downstream derived
    summaries across Mem0 regardless of `user_id` bucket fragmentation. Also tracks
    written Mem0 identity scopes so unscoped `search()` calls and post-erasure leak
    probes work cleanly with Mem0 2.x's strict `filters` requirement.
    """

    name = "mem0"

    def __init__(
        self,
        client: Any,
        *,
        default_user_id: str | None = None,
        default_agent_id: str | None = None,
        default_run_id: str | None = None,
        infer: bool = False,
        search_threshold: float | None = None,
        known_scopes: Sequence[Mapping[str, Any]] = (),
        sanitize_ids: bool = True,
    ) -> None:
        self.client = client
        self.default_user_id = default_user_id
        self.default_agent_id = default_agent_id
        self.default_run_id = default_run_id
        self.infer = infer
        self.search_threshold = search_threshold
        self.sanitize_ids = sanitize_ids
        self._lock = threading.RLock()
        self._scopes: list[dict[str, str]] = []
        self._scope_keys: set[tuple[tuple[str, str], ...]] = set()
        self._direct_vector_ids: set[str] = set()
        if default_user_id or default_agent_id or default_run_id:
            self.register_scope(
                user_id=default_user_id,
                agent_id=default_agent_id,
                run_id=default_run_id,
            )
        for scope in known_scopes:
            self.register_scope(scope)

    def normalize_entity_id(self, value: Any) -> str | None:
        if value is None:
            return None
        if self.sanitize_ids:
            return _normalize_mem0_entity_id(value)
        trimmed = str(value).strip()
        return trimmed or None

    def register_scope(
        self,
        scope: Mapping[str, Any] | None = None,
        *,
        user_id: Any = None,
        agent_id: Any = None,
        run_id: Any = None,
    ) -> None:
        """Record a `(user_id, agent_id, run_id)` scope so unscoped searches cover it."""
        merged: dict[str, Any] = dict(scope) if scope else {}
        if user_id is not None:
            merged["user_id"] = user_id
        if agent_id is not None:
            merged["agent_id"] = agent_id
        if run_id is not None:
            merged["run_id"] = run_id
        normalized: dict[str, str] = {}
        for key in ("user_id", "agent_id", "run_id"):
            if key in merged and merged[key] is not None:
                norm_val = self.normalize_entity_id(merged[key])
                if norm_val:
                    normalized[key] = norm_val
        if not normalized:
            return
        scope_key = tuple(sorted(normalized.items()))
        with self._lock:
            if scope_key not in self._scope_keys:
                self._scope_keys.add(scope_key)
                self._scopes.append(normalized)

    def registered_scopes(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(scope) for scope in self._scopes]

    def upsert_detailed(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> tuple[AdapterOp, ...]:
        meta = dict(metadata)
        encrypted_mode = bool(meta.get("ember_encrypted"))
        raw_user_id = meta.pop("user_id", None)
        raw_agent_id = meta.pop("agent_id", None)
        raw_run_id = meta.pop("run_id", None)
        meta["ember_item_id"] = item_id
        if encrypted_mode:
            meta["ember_subject_refs"] = list(subjects)
        else:
            meta["ember_subjects"] = list(subjects)

        eff_raw_user_id = raw_user_id if raw_user_id is not None else (
            self.default_user_id if self.default_user_id is not None else (subjects[0] if subjects else None)
        )
        eff_raw_agent_id = raw_agent_id if raw_agent_id is not None else self.default_agent_id
        eff_raw_run_id = raw_run_id if raw_run_id is not None else self.default_run_id

        user_id = self.normalize_entity_id(eff_raw_user_id)
        agent_id = self.normalize_entity_id(eff_raw_agent_id)
        run_id = self.normalize_entity_id(eff_raw_run_id)

        if not encrypted_mode:
            if eff_raw_user_id is not None and user_id != str(eff_raw_user_id).strip():
                meta.setdefault("ember_user_id", str(eff_raw_user_id).strip())
            if eff_raw_agent_id is not None and agent_id != str(eff_raw_agent_id).strip():
                meta.setdefault("ember_agent_id", str(eff_raw_agent_id).strip())
            if eff_raw_run_id is not None and run_id != str(eff_raw_run_id).strip():
                meta.setdefault("ember_run_id", str(eff_raw_run_id).strip())

        if vector is not None and hasattr(self.client, "vector_store"):
            vs = self.client.vector_store
            insert_fn = getattr(vs, "insert", None)
            if callable(insert_fn):
                rec_id = str(uuid.uuid4())
                payload = {**meta, "data": text}
                if user_id is not None:
                    payload["user_id"] = user_id
                if agent_id is not None:
                    payload["agent_id"] = agent_id
                if run_id is not None:
                    payload["run_id"] = run_id
                insert_fn(vectors=[list(float(x) for x in vector)], ids=[rec_id], payloads=[payload])
                with self._lock:
                    self._direct_vector_ids.add(rec_id)
                self.register_scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
                return (AdapterOp(record_id=rec_id, event="ADD", memory=text),)

        effective_infer = False if encrypted_mode else self.infer
        kwargs: dict[str, Any] = {"metadata": meta, "infer": effective_infer}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if agent_id is not None:
            kwargs["agent_id"] = agent_id
        if run_id is not None:
            kwargs["run_id"] = run_id

        sig = inspect.signature(self.client.add)
        accepted = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted or has_varkw}
        if "messages" in accepted:
            raw = self.client.add(messages=text, **filtered_kwargs)
        else:
            raw = self.client.add(text, **filtered_kwargs)

        self.register_scope(user_id=user_id, agent_id=agent_id, run_id=run_id)

        if raw is None:
            return ()
        rows = raw.get("results", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        ops: list[AdapterOp] = []
        for entry in rows:
            if not isinstance(entry, dict):
                continue
            rid = entry.get("id")
            if not rid:
                continue
            ops.append(
                AdapterOp(
                    record_id=str(rid),
                    event=str(entry.get("event") or "ADD").upper(),
                    memory=str(entry.get("memory") or entry.get("text") or ""),
                    previous_memory=(
                        str(entry["previous_memory"])
                        if entry.get("previous_memory") is not None
                        else (str(entry["prev_memory"]) if entry.get("prev_memory") is not None else None)
                    ),
                )
            )
        if not ops and isinstance(raw, dict) and raw.get("id"):
            ops.append(
                AdapterOp(
                    record_id=str(raw["id"]),
                    event=str(raw.get("event") or "ADD").upper(),
                    memory=str(raw.get("memory") or raw.get("text") or ""),
                )
            )
        return tuple(ops)

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        ops = self.upsert_detailed(
            item_id=item_id,
            text=text,
            subjects=subjects,
            metadata=metadata,
            vector=vector,
        )
        return tuple(op.record_id for op in ops if op.record_id and op.event != "DELETE")

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0 or not hasattr(self.client, "vector_store"):
            return []
        vs = self.client.vector_store
        search_fn = getattr(vs, "search", None)
        if not callable(search_fn):
            return []
        norm_filters = self._normalize_filters(filters)
        sig = inspect.signature(search_fn)
        params = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        q_vec = [list(float(x) for x in query_vector)]

        scopes = self.registered_scopes()
        query_scopes: list[dict[str, Any]]
        if any(norm_filters.get(k) is not None for k in ("user_id", "agent_id", "run_id")):
            query_scopes = [norm_filters]
        elif scopes:
            query_scopes = [{**scope, **norm_filters} for scope in scopes]
        else:
            query_scopes = [norm_filters] if norm_filters else [{}]

        merged: dict[str, AdapterHit] = {}
        for qf in query_scopes:
            kwargs: dict[str, Any] = {}
            if "vectors" in params or has_varkw:
                kwargs["vectors"] = q_vec
            if "query" in params:
                kwargs["query"] = ""
            if "top_k" in params or has_varkw:
                kwargs["top_k"] = limit
            elif "limit" in params:
                kwargs["limit"] = limit
            if qf and ("filters" in params or has_varkw):
                kwargs["filters"] = qf
            rows = search_fn(**kwargs) or []
            for row in rows:
                rid = str(getattr(row, "id", "") or (row.get("id", "") if isinstance(row, Mapping) else ""))
                payload = getattr(row, "payload", None)
                if payload is None and isinstance(row, Mapping):
                    payload = row.get("payload") or row
                payload_dict = dict(payload or {})
                score = float(getattr(row, "score", 0.0) if not isinstance(row, Mapping) else row.get("score", 0.0))
                txt = str(payload_dict.get("data") or payload_dict.get("memory") or payload_dict.get("text") or "")
                hit = AdapterHit(record_id=rid, text=txt, score=score, metadata=payload_dict)
                prev = merged.get(rid)
                if prev is None or hit.score > prev.score:
                    merged[rid] = hit
        hits = list(merged.values())
        hits.sort(key=lambda h: (-h.score, h.record_id))
        return hits[:limit]

    def _rows_to_hits(self, raw: Any) -> list[AdapterHit]:
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
                    metadata=_promote_mem0_metadata(entry),
                )
            )
        return hits

    def _normalize_filters(self, filters: Mapping[str, Any] | None) -> dict[str, Any]:
        if not filters:
            return {}
        out = dict(filters)
        if self.sanitize_ids:
            for key in ("user_id", "agent_id", "run_id"):
                if key in out and isinstance(out[key], str):
                    norm_val = self.normalize_entity_id(out[key])
                    if norm_val is not None:
                        out[key] = norm_val
        return out

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
        base_kwargs: dict[str, Any] = {}
        if "top_k" in params:
            base_kwargs["top_k"] = limit
        elif "limit" in params:
            base_kwargs["limit"] = limit
        if self.search_threshold is not None and "threshold" in params:
            base_kwargs["threshold"] = self.search_threshold

        norm_filters = self._normalize_filters(filters)
        has_identity_filter = any(
            norm_filters.get(k) is not None for k in ("user_id", "agent_id", "run_id")
        )

        # Single direct search when explicit identity filters are supplied or client doesn't take `filters`.
        if has_identity_filter or "filters" not in params:
            kwargs = dict(base_kwargs)
            if norm_filters:
                if "filters" in params:
                    kwargs["filters"] = norm_filters
                else:
                    for k, v in norm_filters.items():
                        if k in params:
                            kwargs[k] = v
            raw = self.client.search(query, **kwargs)
            return self._rows_to_hits(raw)[:limit]

        # Unscoped search against a `filters`-aware client (e.g., mem0.Memory 2.x):
        # fan out across tracked identity scopes so Mem0 never rejects `filters=None`.
        scopes = self.registered_scopes()
        query_scopes: list[dict[str, Any]]
        if scopes:
            norm_default_agent = self.normalize_entity_id(self.default_agent_id)
            if (
                norm_default_agent
                and "agent_id" not in norm_filters
                and all(s.get("agent_id") == norm_default_agent for s in scopes)
            ):
                query_scopes = [{"agent_id": norm_default_agent, **norm_filters}]
            else:
                query_scopes = [{**scope, **norm_filters} for scope in scopes]
        else:
            kwargs = dict(base_kwargs)
            if norm_filters:
                kwargs["filters"] = norm_filters
            try:
                raw = self.client.search(query, **kwargs)
                return self._rows_to_hits(raw)[:limit]
            except ValueError as exc:
                if "filters must contain at least one of" in str(exc):
                    return []
                raise

        merged: dict[str, AdapterHit] = {}
        for idx, qf in enumerate(query_scopes):
            raw = self.client.search(query, filters=qf, **base_kwargs)
            for hit in self._rows_to_hits(raw):
                dedup_key = hit.record_id or f"anon:{idx}:{hit.text}"
                prev = merged.get(dedup_key)
                if prev is None or hit.score > prev.score:
                    merged[dedup_key] = hit
        hits = list(merged.values())
        hits.sort(key=lambda h: (-h.score, h.record_id))
        return hits[:limit]

    def _delete_direct_vector(self, rid: str) -> bool:
        vs = getattr(self.client, "vector_store", None)
        vs_del = getattr(vs, "delete", None) if vs is not None else None
        if not callable(vs_del):
            return False
        vs_sig = inspect.signature(vs_del)
        params = set(vs_sig.parameters)
        try:
            if "vector_id" in params:
                vs_del(vector_id=rid)
            elif "ids" in params:
                vs_del(ids=[rid])
            elif "id" in params:
                vs_del(id=rid)
            else:
                vs_del(rid)
            return True
        except ValueError as exc:
            if "not found" in str(exc).lower():
                return False
            raise

    def delete(self, record_ids: Sequence[str]) -> int:
        deleted = 0
        ids = list(dict.fromkeys(str(rid) for rid in record_ids if rid))
        if not ids:
            return 0
        delete_fn = getattr(self.client, "delete", None)
        use_kw = False
        if callable(delete_fn):
            sig = inspect.signature(delete_fn)
            use_kw = "memory_id" in sig.parameters
        for rid in ids:
            with self._lock:
                is_direct = rid in self._direct_vector_ids
                self._direct_vector_ids.discard(rid)
            if is_direct and self._delete_direct_vector(rid):
                deleted += 1
                continue
            if callable(delete_fn):
                try:
                    try:
                        if use_kw:
                            delete_fn(memory_id=rid)
                        else:
                            delete_fn(rid)
                    except TypeError:
                        delete_fn(rid)
                    deleted += 1
                    continue
                except ValueError as exc:
                    if "not found" not in str(exc).lower():
                        raise
            if self._delete_direct_vector(rid):
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
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        encrypted_mode = bool(metadata.get("ember_encrypted"))
        meta = dict(metadata)
        meta["ember_item_id"] = item_id
        if encrypted_mode:
            meta["ember_subject_refs"] = list(subjects)
        else:
            meta["ember_subjects"] = list(subjects)
        rec_id = f"vec_{uuid.uuid4().hex[:12]}"

        if vector is not None and hasattr(self.store, "add_embeddings"):
            sig = inspect.signature(self.store.add_embeddings)
            params = set(sig.parameters)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            kwargs: dict[str, Any] = {}
            if "metadatas" in params or has_varkw:
                kwargs["metadatas"] = [meta]
            if "ids" in params or has_varkw:
                kwargs["ids"] = [rec_id]
            returned = self.store.add_embeddings(text_embeddings=[(text, [float(x) for x in vector])], **kwargs)
            if isinstance(returned, (list, tuple)) and returned:
                return tuple(str(x) for x in returned)
            return (rec_id,) if ("ids" in kwargs) else ()

        sig = inspect.signature(self.store.add_texts)
        params = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs = {}
        if "metadatas" in params or has_varkw:
            kwargs["metadatas"] = [meta]
        if "ids" in params or has_varkw:
            kwargs["ids"] = [rec_id]
        returned = self.store.add_texts(texts=[text], **kwargs)
        if isinstance(returned, (list, tuple)) and returned:
            return tuple(str(x) for x in returned)
        return (rec_id,) if ("ids" in kwargs) else ()

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        vec_list = [float(x) for x in query_vector]
        for method_name, with_scores in (
            ("similarity_search_by_vector_with_relevance_scores", True),
            ("similarity_search_by_vector_with_score", True),
            ("similarity_search_by_vector", False),
        ):
            fn = getattr(self.store, method_name, None)
            if not callable(fn):
                continue
            sig = inspect.signature(fn)
            params = set(sig.parameters)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            kwargs: dict[str, Any] = {}
            if "k" in params or has_varkw:
                kwargs["k"] = limit
            if filters is not None and ("filter" in params or has_varkw):
                kwargs["filter"] = filters
            raw = fn(vec_list, **kwargs)
            hits: list[AdapterHit] = []
            for item in raw:
                if with_scores:
                    doc, score = item
                    sc = float(score)
                else:
                    doc, sc = item, 1.0
                meta = dict(getattr(doc, "metadata", {}) or {})
                hits.append(
                    AdapterHit(
                        record_id=str(getattr(doc, "id", None) or meta.get("id") or meta.get("ember_item_id") or ""),
                        text=str(getattr(doc, "page_content", "")),
                        score=sc,
                        metadata=meta,
                    )
                )
            return hits[:limit]
        return []

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


_VALID_SQL_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _normalize_adapter_metadata(
    *,
    item_id: str,
    subjects: tuple[str, ...],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    encrypted_mode = bool(metadata.get("ember_encrypted"))
    meta = dict(metadata)
    meta["ember_item_id"] = item_id
    if encrypted_mode:
        meta["ember_subject_refs"] = list(subjects)
    else:
        meta["ember_subjects"] = list(subjects)
        if subjects and "user_id" not in meta:
            meta["user_id"] = subjects[0]
    return meta


def _scalar_only_chroma_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Encode non-scalar metadata lists into JSON strings for Chroma metadata constraints."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[f"__json_{k}"] = json.dumps(list(v))
        else:
            out[k] = str(v)
    return out


def _unpack_chroma_metadata(raw_meta: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (raw_meta or {}).items():
        if isinstance(k, str) and k.startswith("__json_") and isinstance(v, str):
            try:
                out[k[7:]] = json.loads(v)
                continue
            except ValueError:
                pass
        out[k] = v
    return out


def _flatten_pinecone_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure metadata values satisfy Pinecone scalar/list[str] constraints cleanly."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v if x is not None]
        else:
            out[k] = json.dumps(v) if isinstance(v, Mapping) else str(v)
    return out


class QdrantAdapter:
    """Native O(1)-filtered custody adapter for `qdrant_client.QdrantClient`."""

    name = "qdrant"
    supports_vector = True

    def __init__(
        self,
        client: Any,
        collection_name: str,
        *,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        text_key: str = "text",
    ) -> None:
        if not collection_name or not str(collection_name).strip():
            raise ValueError("collection_name must be non-empty")
        self.client = client
        self.collection_name = collection_name
        self.embed_fn = embed_fn
        self.text_key = text_key

    def _build_qdrant_filter(self, filters: Mapping[str, Any] | None) -> Any:
        if not filters:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            return Filter(
                must=[
                    FieldCondition(key=str(k), match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )
        except ImportError:
            return {"must": [{"key": str(k), "match": {"value": v}} for k, v in filters.items()]}

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        vec = list(float(x) for x in vector) if vector is not None else (
            list(float(x) for x in self.embed_fn(text)) if self.embed_fn is not None else None
        )
        if vec is None:
            raise ValueError(
                "QdrantAdapter requires an embedding vector via CustodyVault(embedder=...) or QdrantAdapter(embed_fn=...)"
            )
        rec_id = str(uuid.uuid4())
        payload = _normalize_adapter_metadata(item_id=item_id, subjects=subjects, metadata=metadata)
        payload[self.text_key] = text

        try:
            from qdrant_client.models import PointStruct

            point: Any = PointStruct(id=rec_id, vector=vec, payload=payload)
        except ImportError:
            point = {"id": rec_id, "vector": vec, "payload": payload}

        self.client.upsert(collection_name=self.collection_name, points=[point])
        return (rec_id,)

    def _parse_qdrant_points(self, rows: Sequence[Any]) -> list[AdapterHit]:
        hits: list[AdapterHit] = []
        for row in rows:
            if isinstance(row, Mapping):
                rid = str(row.get("id", ""))
                score = float(row.get("score", 0.0) or 0.0)
                payload = dict(row.get("payload") or {})
            else:
                rid = str(getattr(row, "id", ""))
                score = float(getattr(row, "score", 0.0) or 0.0)
                payload = dict(getattr(row, "payload", None) or {})
            txt = str(payload.get(self.text_key) or payload.get("memory") or payload.get("text") or "")
            hits.append(AdapterHit(record_id=rid, text=txt, score=score, metadata=payload))
        return hits

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        q_filter = self._build_qdrant_filter(filters)
        q_vec = [float(x) for x in query_vector]
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
            points = getattr(resp, "points", None)
            if points is None and isinstance(resp, Mapping):
                points = resp.get("points", [])
            return self._parse_qdrant_points(points or [])[:limit]
        rows = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            query_filter=q_filter,
            limit=limit,
            with_payload=True,
        )
        return self._parse_qdrant_points(rows or [])[:limit]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        if self.embed_fn is not None:
            return self.search_vector(self.embed_fn(query), limit=limit, filters=filters)
        # Token-overlap fallback over Qdrant scroll if no embed_fn was configured
        scroll_fn = getattr(self.client, "scroll", None)
        if not callable(scroll_fn):
            return []
        q_filter = self._build_qdrant_filter(filters)
        raw = scroll_fn(
            collection_name=self.collection_name,
            scroll_filter=q_filter,
            limit=max(limit * 4, 25),
            with_payload=True,
        )
        points = raw[0] if isinstance(raw, tuple) else (getattr(raw, "points", None) or raw)
        q_toks = _content_tokens(query)
        scored: list[AdapterHit] = []
        for hit in self._parse_qdrant_points(points or []):
            overlap = len(q_toks & _tokenize(hit.text))
            if overlap > 0 or query.strip().lower() in hit.text.lower():
                scored.append(
                    AdapterHit(
                        record_id=hit.record_id,
                        text=hit.text,
                        score=overlap / max(1, len(q_toks)),
                        metadata=hit.metadata,
                    )
                )
        scored.sort(key=lambda h: (-h.score, h.record_id))
        return scored[:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(str(rid) for rid in record_ids if rid))
        if not ids:
            return 0
        try:
            from qdrant_client.models import PointIdsList

            selector: Any = PointIdsList(points=ids)
        except ImportError:
            selector = {"points": ids}
        self.client.delete(collection_name=self.collection_name, points_selector=selector)
        return len(ids)


class ChromaAdapter:
    """Native O(1)-filtered custody adapter wrapping a `chromadb.Collection`."""

    name = "chroma"
    supports_vector = True

    def __init__(
        self,
        collection: Any,
        *,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self.collection = collection
        self.embed_fn = embed_fn

    @staticmethod
    def _build_where(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        items = list(filters.items())
        if len(items) == 1:
            k, v = items[0]
            return {str(k): v}
        return {"$and": [{str(k): v} for k, v in items]}

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        rec_id = f"chr_{uuid.uuid4().hex[:12]}"
        meta = _scalar_only_chroma_metadata(
            _normalize_adapter_metadata(item_id=item_id, subjects=subjects, metadata=metadata)
        )
        vec = list(float(x) for x in vector) if vector is not None else (
            list(float(x) for x in self.embed_fn(text)) if self.embed_fn is not None else None
        )
        kwargs: dict[str, Any] = {
            "ids": [rec_id],
            "documents": [text],
            "metadatas": [meta],
        }
        if vec is not None:
            kwargs["embeddings"] = [vec]
        upsert_fn = getattr(self.collection, "upsert", None)
        if callable(upsert_fn):
            upsert_fn(**kwargs)
        else:
            self.collection.add(**kwargs)
        return (rec_id,)

    def _parse_query_result(self, raw: Mapping[str, Any]) -> list[AdapterHit]:
        ids_batch = raw.get("ids") or [[]]
        docs_batch = raw.get("documents") or [[]]
        metas_batch = raw.get("metadatas") or [[]]
        dists_batch = raw.get("distances") or [[]]
        ids = ids_batch[0] if ids_batch else []
        docs = docs_batch[0] if docs_batch else []
        metas = metas_batch[0] if metas_batch else []
        dists = dists_batch[0] if dists_batch else []
        hits: list[AdapterHit] = []
        for idx, rid in enumerate(ids):
            txt = str(docs[idx] if idx < len(docs) and docs[idx] is not None else "")
            meta = _unpack_chroma_metadata(metas[idx] if idx < len(metas) else None)
            dist = float(dists[idx]) if idx < len(dists) and dists[idx] is not None else 0.0
            score = 1.0 / (1.0 + max(0.0, dist))
            hits.append(AdapterHit(record_id=str(rid), text=txt, score=score, metadata=meta))
        return hits

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [[float(x) for x in query_vector]],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._build_where(filters)
        if where is not None:
            kwargs["where"] = where
        return self._parse_query_result(self.collection.query(**kwargs))[:limit]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        if self.embed_fn is not None:
            return self.search_vector(self.embed_fn(query), limit=limit, filters=filters)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._build_where(filters)
        if where is not None:
            kwargs["where"] = where
        return self._parse_query_result(self.collection.query(**kwargs))[:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(str(rid) for rid in record_ids if rid))
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)


class PgVectorAdapter:
    """Native O(1)-filtered PostgreSQL + pgvector adapter with parameterized SQL & JSONB filters."""

    name = "pgvector"
    supports_vector = True

    def __init__(
        self,
        connection: Any,
        *,
        table_name: str = "ember_vectors",
        dimensions: int = 384,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        create_table: bool = True,
    ) -> None:
        if not _VALID_SQL_IDENT.fullmatch(table_name):
            raise ValueError(f"invalid SQL table_name {table_name!r}")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.conn = connection
        self.table_name = table_name
        self.dimensions = int(dimensions)
        self.embed_fn = embed_fn
        self._lock = threading.RLock()
        if create_table:
            self._init_table()

    def _run(self, sql_template: str, params: Sequence[Any] = ()) -> Any:
        sql = sql_template.replace("__TBL__", self.table_name).replace("__DIMS__", str(self.dimensions))
        cur = self.conn.cursor()
        try:
            if params:
                cur.execute(sql, tuple(params))
            else:
                cur.execute(sql)
            return cur
        finally:
            if hasattr(cur, "close"):
                cur.close()

    def _init_table(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                ddl = (
                    "CREATE TABLE IF NOT EXISTS __TBL__ ("
                    "record_id TEXT PRIMARY KEY, "
                    "item_id TEXT NOT NULL, "
                    "text TEXT NOT NULL, "
                    "meta_json JSONB NOT NULL, "
                    "embedding vector(__DIMS__)"
                    ")"
                ).replace("__TBL__", self.table_name).replace("__DIMS__", str(self.dimensions))
                cur.execute(ddl)
                self.conn.commit()
            except BaseException:
                if hasattr(self.conn, "rollback"):
                    self.conn.rollback()
                raise
            finally:
                if hasattr(cur, "close"):
                    cur.close()

    @staticmethod
    def _vec_literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        rec_id = f"pgv_{uuid.uuid4().hex[:12]}"
        meta = _normalize_adapter_metadata(item_id=item_id, subjects=subjects, metadata=metadata)
        vec = list(float(x) for x in vector) if vector is not None else (
            list(float(x) for x in self.embed_fn(text)) if self.embed_fn is not None else None
        )
        vec_lit = self._vec_literal(vec) if vec is not None else None
        meta_str = json.dumps(meta, sort_keys=True)
        with self._lock:
            cur = self.conn.cursor()
            try:
                sql = (
                    "INSERT INTO __TBL__ (record_id, item_id, text, meta_json, embedding) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s::vector) "
                    "ON CONFLICT (record_id) DO UPDATE SET "
                    "item_id = EXCLUDED.item_id, text = EXCLUDED.text, "
                    "meta_json = EXCLUDED.meta_json, embedding = EXCLUDED.embedding"
                ).replace("__TBL__", self.table_name)
                cur.execute(sql, (rec_id, item_id, text, meta_str, vec_lit))
                self.conn.commit()
            except BaseException:
                if hasattr(self.conn, "rollback"):
                    self.conn.rollback()
                raise
            finally:
                if hasattr(cur, "close"):
                    cur.close()
        return (rec_id,)

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        vec_lit = self._vec_literal(query_vector)
        with self._lock:
            cur = self.conn.cursor()
            try:
                if filters:
                    sql = (
                        "SELECT record_id, text, meta_json, (1 - (embedding <=> %s::vector)) AS score "
                        "FROM __TBL__ "
                        "WHERE embedding IS NOT NULL AND meta_json @> %s::jsonb "
                        "ORDER BY embedding <=> %s::vector ASC LIMIT %s"
                    ).replace("__TBL__", self.table_name)
                    cur.execute(sql, (vec_lit, json.dumps(dict(filters), sort_keys=True), vec_lit, limit))
                else:
                    sql = (
                        "SELECT record_id, text, meta_json, (1 - (embedding <=> %s::vector)) AS score "
                        "FROM __TBL__ "
                        "WHERE embedding IS NOT NULL "
                        "ORDER BY embedding <=> %s::vector ASC LIMIT %s"
                    ).replace("__TBL__", self.table_name)
                    cur.execute(sql, (vec_lit, vec_lit, limit))
                rows = cur.fetchall()
            finally:
                if hasattr(cur, "close"):
                    cur.close()
        hits: list[AdapterHit] = []
        for rec_id, text, meta_val, score in rows:
            meta = dict(meta_val) if isinstance(meta_val, Mapping) else json.loads(meta_val)
            hits.append(AdapterHit(record_id=str(rec_id), text=str(text), score=float(score), metadata=meta))
        return hits[:limit]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        if self.embed_fn is not None:
            return self.search_vector(self.embed_fn(query), limit=limit, filters=filters)
        q_toks = sorted(_content_tokens(query))
        pattern = "|".join(re.escape(tok) for tok in q_toks) if q_toks else ""
        filter_json = json.dumps(dict(filters), sort_keys=True) if filters else None
        with self._lock:
            cur = self.conn.cursor()
            try:
                if filters and pattern:
                    sql = (
                        "SELECT record_id, text, meta_json FROM __TBL__ "
                        "WHERE meta_json @> %s::jsonb AND text ~* %s LIMIT %s"
                    ).replace("__TBL__", self.table_name)
                    cur.execute(sql, (filter_json, pattern, limit))
                elif filters:
                    sql = (
                        "SELECT record_id, text, meta_json FROM __TBL__ "
                        "WHERE meta_json @> %s::jsonb LIMIT %s"
                    ).replace("__TBL__", self.table_name)
                    cur.execute(sql, (filter_json, limit))
                elif pattern:
                    sql = (
                        "SELECT record_id, text, meta_json FROM __TBL__ "
                        "WHERE text ~* %s LIMIT %s"
                    ).replace("__TBL__", self.table_name)
                    cur.execute(sql, (pattern, limit))
                else:
                    sql = ("SELECT record_id, text, meta_json FROM __TBL__ LIMIT %s").replace("__TBL__", self.table_name)
                    cur.execute(sql, (limit,))
                rows = cur.fetchall()
            finally:
                if hasattr(cur, "close"):
                    cur.close()
        q_set = _content_tokens(query)
        hits: list[AdapterHit] = []
        for rec_id, text, meta_val in rows:
            meta = dict(meta_val) if isinstance(meta_val, Mapping) else json.loads(meta_val)
            overlap = len(q_set & _tokenize(str(text)))
            score = overlap / max(1, len(q_set))
            hits.append(AdapterHit(record_id=str(rec_id), text=str(text), score=score, metadata=meta))
        hits.sort(key=lambda h: (-h.score, h.record_id))
        return hits[:limit]

    def delete(self, record_ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(str(rid) for rid in record_ids if rid))
        if not ids:
            return 0
        sql = ("DELETE FROM __TBL__ WHERE record_id = ANY(%s)").replace("__TBL__", self.table_name)
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(sql, (ids,))
                deleted = cur.rowcount if getattr(cur, "rowcount", -1) >= 0 else len(ids)
                self.conn.commit()
                return int(deleted)
            except BaseException:
                if hasattr(self.conn, "rollback"):
                    self.conn.rollback()
                raise
            finally:
                if hasattr(cur, "close"):
                    cur.close()


class PineconeAdapter:
    """Native O(1)-filtered custody adapter wrapping a Pinecone `Index`."""

    name = "pinecone"
    supports_vector = True

    def __init__(
        self,
        index: Any,
        *,
        namespace: str = "",
        embed_fn: Callable[[str], Sequence[float]] | None = None,
        text_key: str = "text",
    ) -> None:
        self.index = index
        self.namespace = namespace
        self.embed_fn = embed_fn
        self.text_key = text_key

    @staticmethod
    def _build_filter(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        return {str(k): {"$eq": v} for k, v in filters.items()}

    def upsert(
        self,
        *,
        item_id: str,
        text: str,
        subjects: tuple[str, ...],
        metadata: dict[str, Any],
        vector: Sequence[float] | None = None,
    ) -> tuple[str, ...]:
        vec = list(float(x) for x in vector) if vector is not None else (
            list(float(x) for x in self.embed_fn(text)) if self.embed_fn is not None else None
        )
        if vec is None:
            raise ValueError(
                "PineconeAdapter requires vector embeddings via CustodyVault(embedder=...) or PineconeAdapter(embed_fn=...)"
            )
        rec_id = f"pc_{uuid.uuid4().hex[:12]}"
        meta = _flatten_pinecone_metadata(
            _normalize_adapter_metadata(item_id=item_id, subjects=subjects, metadata=metadata)
        )
        meta[self.text_key] = text
        self.index.upsert(
            vectors=[{"id": rec_id, "values": vec, "metadata": meta}],
            namespace=self.namespace,
        )
        return (rec_id,)

    def _parse_matches(self, resp: Any) -> list[AdapterHit]:
        matches = resp.get("matches", []) if isinstance(resp, Mapping) else (getattr(resp, "matches", None) or [])
        hits: list[AdapterHit] = []
        for m in matches:
            if isinstance(m, Mapping):
                rid = str(m.get("id", ""))
                score = float(m.get("score", 0.0) or 0.0)
                meta = dict(m.get("metadata") or {})
            else:
                rid = str(getattr(m, "id", ""))
                score = float(getattr(m, "score", 0.0) or 0.0)
                meta = dict(getattr(m, "metadata", None) or {})
            txt = str(meta.get(self.text_key) or meta.get("memory") or meta.get("text") or "")
            hits.append(AdapterHit(record_id=rid, text=txt, score=score, metadata=meta))
        return hits

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        kwargs: dict[str, Any] = {
            "vector": [float(x) for x in query_vector],
            "top_k": limit,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        pc_filter = self._build_filter(filters)
        if pc_filter is not None:
            kwargs["filter"] = pc_filter
        return self._parse_matches(self.index.query(**kwargs))[:limit]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[AdapterHit]:
        if limit <= 0:
            return []
        if self.embed_fn is not None:
            return self.search_vector(self.embed_fn(query), limit=limit, filters=filters)
        raise ValueError(
            "PineconeAdapter text search requires an embed_fn (pass embedder= to CustodyVault or embed_fn= to PineconeAdapter)"
        )

    def delete(self, record_ids: Sequence[str]) -> int:
        ids = list(dict.fromkeys(str(rid) for rid in record_ids if rid))
        if not ids:
            return 0
        self.index.delete(ids=ids, namespace=self.namespace)
        return len(ids)


ProbeFilterFn = Callable[[AdapterHit], bool]
