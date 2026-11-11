"""CustodyVault: Ember's unified developer API combining ledger, keystore, Merkle redaction, signing, and adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
import threading
from typing import Any
import uuid

from .adapters import (
    AdapterHit,
    AdapterOp,
    InMemoryStoreAdapter,
    Mem0Adapter,
    MemoryStoreAdapter,
    SQLiteStoreAdapter,
    VectorStoreAdapter,
    _content_tokens,
    _tokenize,
)
from .crypto import CIPHER_NAME, ERASED_TOMBSTONE, build_aad, encrypt_text
from .keystore import FileKeyStore, KeyStore, MemoryKeyStore, SQLiteKeyStore
from .ledger import LedgerBackend, SQLiteLedger
from .merkle import ChunkMode, ChunkerFn, RedactableDoc, SpanChunkerFn, SpanMode, _entry_text
from .receipts import Receipt, mint_redactable_receipt
from .records import ErasureCertificate, MemoryItem, SourceRecord
from .reducer import (
    EmberState,
    ErasureIncomplete,
    InvariantViolation,
    TombstonedSubjectError,
    compute_closure,
)
from .signing import Ed25519Signer, NullAnchor, TimestampAnchor, sign_certificate_dict
from .verify import canonical_bytes, sha256_prefixed

_WORD_CHAR_NO_UNDERSCORE = re.compile(r"[^\W_]", re.UNICODE)
_ACTIVE_TRACES: contextvars.ContextVar[dict[str, "TurnTrace"] | None] = contextvars.ContextVar(
    "ember_active_traces",
    default=None,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _marker_matches(text: str, markers: Sequence[str]) -> bool:
    """Boundary-aware case-insensitive marker detection (avoids 'ed' matching 'joined')."""
    if not text:
        return False
    for marker in markers:
        m = marker.strip()
        if not m:
            continue
        left = r"(?<![^\W_])" if _WORD_CHAR_NO_UNDERSCORE.match(m[0]) else ""
        right = r"(?![^\W_])" if _WORD_CHAR_NO_UNDERSCORE.match(m[-1]) else ""
        if re.search(f"{left}{re.escape(m)}{right}", text, flags=re.IGNORECASE | re.UNICODE):
            return True
    return False


_PUNCT_STRIP = " \t\r\n.,!?;:'\"()[]{}"
_WORD_OR_ID_RE = re.compile(r"[^\W_]+(?:[@.+_-][^\W_]+)*", re.UNICODE)


def _scalar_metadata_strings(metadata: Mapping[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    out: list[str] = []
    for k, v in metadata.items():
        if str(k).startswith("ember_"):
            continue
        if isinstance(v, str) and v.strip():
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for entry in v:
                if isinstance(entry, str) and entry.strip():
                    out.append(entry)
    return out


def _extract_content_candidates(
    text: str,
    quote: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    *,
    max_ngram: int = 4,
) -> set[str]:
    candidates: set[str] = set()
    sources = [text]
    if isinstance(quote, str) and quote.strip():
        sources.append(quote)
    sources.extend(_scalar_metadata_strings(metadata))

    for src in sources:
        norm_full = " ".join(src.strip().lower().split())
        if norm_full:
            candidates.add(norm_full)
        ws_tokens = [tok.strip(_PUNCT_STRIP).lower() for tok in src.split()]
        ws_tokens = [t for t in ws_tokens if t]
        for size in range(1, min(max_ngram, len(ws_tokens)) + 1):
            for idx in range(len(ws_tokens) - size + 1):
                candidates.add(" ".join(ws_tokens[idx : idx + size]))
        id_tokens = [tok.lower() for tok in _WORD_OR_ID_RE.findall(src)]
        for size in range(1, min(max_ngram, len(id_tokens)) + 1):
            for idx in range(len(id_tokens) - size + 1):
                candidates.add(" ".join(id_tokens[idx : idx + size]))
    return candidates


def _is_auto_block_identifier(val: str) -> bool:
    norm = " ".join(val.strip().lower().split())
    if not norm:
        return False
    if any(sep in norm for sep in (" ", "@", "-", "_", ".")):
        return True
    return len(norm) >= 3


def _resolve_embed_fn(embedder: Any) -> Callable[[str], list[float]] | None:
    if embedder is None:
        return None
    raw_fn: Callable[[str], Any]
    if callable(embedder):
        raw_fn = embedder
    elif hasattr(embedder, "embed_query") and callable(embedder.embed_query):
        raw_fn = embedder.embed_query
    elif hasattr(embedder, "embed") and callable(embedder.embed):
        raw_fn = embedder.embed
    elif hasattr(embedder, "encode") and callable(embedder.encode):
        raw_fn = embedder.encode
    else:
        raise TypeError("embedder must be callable or expose embed_query(), embed(), or encode()")

    def _call(text: str) -> list[float]:
        vec = raw_fn(text)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(x) for x in vec]

    return _call


def _adapter_supports_vector(adapter: MemoryStoreAdapter) -> bool:
    if isinstance(adapter, (InMemoryStoreAdapter, SQLiteStoreAdapter)):
        return True
    if isinstance(adapter, VectorStoreAdapter):
        return callable(getattr(adapter.store, "add_embeddings", None)) and any(
            callable(getattr(adapter.store, name, None))
            for name in (
                "similarity_search_by_vector_with_relevance_scores",
                "similarity_search_by_vector_with_score",
                "similarity_search_by_vector",
            )
        )
    if isinstance(adapter, Mem0Adapter):
        vs = getattr(adapter.client, "vector_store", None)
        return vs is not None and callable(getattr(vs, "insert", None)) and callable(getattr(vs, "search", None))
    return bool(getattr(adapter, "supports_vector", False) and callable(getattr(adapter, "search_vector", None)))


def _synthesize_auto_probes(
    *,
    subject_id: str,
    aliases: Sequence[str],
    markers: Sequence[str],
    doomed_items: Sequence[MemoryItem],
    explicit_probes: Sequence[str],
    max_total: int = 16,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(val: str) -> None:
        cleaned = str(val).strip()
        if not cleaned or cleaned == ERASED_TOMBSTONE:
            return
        low = cleaned.lower()
        if low in seen:
            return
        seen.add(low)
        ordered.append(cleaned)

    for p in explicit_probes:
        _add(p)
    target = max(max_total, len(ordered))
    _add(subject_id)
    for a in aliases:
        _add(a)
    for m in markers:
        _add(m)
    for item in doomed_items:
        if len(ordered) >= target:
            break
        if item.text:
            _add(item.text)
        if item.quote and item.quote != item.text:
            _add(item.quote)
        if len(ordered) < target and item.text:
            toks = sorted(_content_tokens(item.text), key=lambda t: (-len(t), t))
            for tok in toks[:2]:
                if len(tok) >= 4:
                    _add(tok)
                if len(ordered) >= target:
                    break
    return ordered[:target]


@dataclass(frozen=True)
class QueryHit:
    item_id: str
    record_id: str
    text: str
    score: float
    subjects: tuple[str, ...]
    receipt: Receipt | None
    receipt_verified: bool | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: tuple[QueryHit, ...]
    filtered_orphans: int = 0

    @property
    def abstained(self) -> bool:
        return len(self.hits) == 0


@dataclass(frozen=True)
class RememberResult:
    item: MemoryItem
    receipt: Receipt | None
    adapter_record_ids: tuple[str, ...]
    event_hash: str


@dataclass
class TurnTrace:
    """Turn-scoped lineage tracker recording reads (`search`) and auto-attaching `derived_from` on writes."""

    vault: "CustodyVault"
    auto_cite: bool = True
    prune: str | Callable[[str, list[QueryHit]], Sequence[str]] = "all"
    _hits: list[QueryHit] = field(default_factory=list)
    _manual_reads: list[str] = field(default_factory=list)
    _written: list[str] = field(default_factory=list)
    _token: contextvars.Token[dict[str, "TurnTrace"] | None] | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def __enter__(self) -> "TurnTrace":
        cur = _ACTIVE_TRACES.get()
        nxt = dict(cur) if cur is not None else {}
        nxt[self.vault._trace_key] = self
        self._token = _ACTIVE_TRACES.set(nxt)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _ACTIVE_TRACES.reset(self._token)
            self._token = None

    async def __aenter__(self) -> "TurnTrace":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.__exit__(exc_type, exc, tb)

    @property
    def hits(self) -> tuple[QueryHit, ...]:
        with self._lock:
            return tuple(self._hits)

    @property
    def read_ids(self) -> tuple[str, ...]:
        with self._lock:
            combined = [h.item_id for h in self._hits]
            combined.extend(self._manual_reads)
            return tuple(dict.fromkeys(combined))

    @property
    def written_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._written)

    def record_read(self, hit_or_id: QueryHit | str) -> None:
        with self._lock:
            if isinstance(hit_or_id, QueryHit):
                self._hits.append(hit_or_id)
            elif isinstance(hit_or_id, str) and hit_or_id.strip():
                self._manual_reads.append(hit_or_id.strip())

    def record_write(self, item_id: str) -> None:
        with self._lock:
            if item_id:
                self._written.append(item_id)

    def clear_reads(self) -> None:
        with self._lock:
            self._hits.clear()
            self._manual_reads.clear()

    def select_citations(self, text: str) -> tuple[str, ...]:
        if not self.auto_cite:
            return ()
        with self._lock:
            all_reads = self.read_ids
            if not all_reads:
                return ()
            hits_copy = list(self._hits)
            manual_copy = list(self._manual_reads)
        if callable(self.prune):
            chosen = [str(x).strip() for x in self.prune(text, hits_copy) if str(x).strip()]
            return tuple(dict.fromkeys(chosen))
        if self.prune == "lexical":
            written_toks = _content_tokens(text)
            written_all = _tokenize(text)
            matched: list[str] = []
            for h in hits_copy:
                h_toks = _content_tokens(h.text)
                if (written_toks & h_toks) or any(
                    _tokenize(subj) & written_all for subj in h.subjects
                ):
                    matched.append(h.item_id)
            matched.extend(manual_copy)
            return tuple(dict.fromkeys(matched))
        if self.prune in ("all", None):
            return all_reads
        raise ValueError(f"unsupported trace prune strategy: {self.prune!r}")


class CustodyVault:
    """Thread-safe cryptographic custody facade for AI memory and RAG stores."""

    def __init__(
        self,
        *,
        ledger: LedgerBackend,
        keystore: KeyStore,
        signer: Ed25519Signer,
        tenant_id: str = "default",
        adapter: MemoryStoreAdapter | None = None,
        anchor: TimestampAnchor | None = None,
        keys_separated_from_ledger: bool = False,
        encrypt_vendor_payload: bool = False,
        embedder: Any = None,
        auto_checkpoint_every: int | None = None,
        use_checkpoint: bool = True,
    ) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        if auto_checkpoint_every is not None and auto_checkpoint_every < 1:
            raise ValueError("auto_checkpoint_every must be >= 1")
        resolved_adapter: MemoryStoreAdapter = adapter if adapter is not None else InMemoryStoreAdapter()
        if encrypt_vendor_payload and isinstance(resolved_adapter, Mem0Adapter) and resolved_adapter.infer:
            raise ValueError(
                "Mem0Adapter(infer=True) cannot be used with encrypt_vendor_payload=True "
                "because server-side LLM fact extraction requires plaintext; set infer=False"
            )
        self._lock = threading.RLock()
        self._trace_key = f"vault_{uuid.uuid4().hex}"
        self.ledger = ledger
        self.keystore = keystore
        self._owns_keystore = False
        self.signer = signer
        self.tenant_id = tenant_id
        self.adapter = resolved_adapter
        self.anchor: TimestampAnchor = anchor if anchor is not None else NullAnchor()
        self.keys_separated_from_ledger = keys_separated_from_ledger
        self.encrypt_vendor_payload = encrypt_vendor_payload
        self.embedder = embedder
        self._embed_fn = _resolve_embed_fn(embedder)
        self.auto_checkpoint_every = auto_checkpoint_every
        self._ops_since_checkpoint = 0
        self.state: EmberState = self._replay_state(
            keystore=self.keystore,
            use_checkpoint=use_checkpoint,
        )
        self._sync_adapter_scopes()

    def _replay_state(self, *, keystore: KeyStore | None, use_checkpoint: bool) -> EmberState:
        sig = inspect.signature(self.ledger.replay)
        params = set(sig.parameters)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        kwargs: dict[str, Any] = {"keystore": keystore}
        if "use_checkpoint" in params or has_varkw:
            kwargs["use_checkpoint"] = use_checkpoint
        if "checkpoint_public_key" in params or has_varkw:
            kwargs["checkpoint_public_key"] = self.signer.public_key_hex
        return self.ledger.replay(self.tenant_id, **kwargs)

    def checkpoint(self) -> dict[str, Any]:
        """Persist an Ed25519-signed ciphertext-only state snapshot for O(delta) restarts."""
        with self._lock:
            save_fn = getattr(self.ledger, "save_checkpoint", None)
            if not callable(save_fn):
                raise NotImplementedError(f"{type(self.ledger).__name__} does not support checkpoints")
            ckpt = save_fn(
                self.tenant_id,
                self.state,
                sign_fn=self.signer.sign,
                public_key_hex=self.signer.public_key_hex,
            )
            self._ops_since_checkpoint = 0
            return ckpt

    def _maybe_auto_checkpoint(self) -> None:
        if self.auto_checkpoint_every is None:
            return
        self._ops_since_checkpoint += 1
        if self._ops_since_checkpoint >= self.auto_checkpoint_every:
            if callable(getattr(self.ledger, "save_checkpoint", None)):
                self.checkpoint()

    def _sync_adapter_scopes(self) -> None:
        """Seed scope-aware adapters (e.g., Mem0Adapter) from replayed state after restart."""
        register = getattr(self.adapter, "register_scope", None)
        if not callable(register):
            return
        for item in self.state.items.values():
            if item.text == ERASED_TOMBSTONE:
                continue
            meta = item.metadata or {}
            uid = meta.get("user_id") or meta.get("ember_user_id") or (item.subjects[0] if item.subjects else None)
            aid = meta.get("agent_id") or meta.get("ember_agent_id")
            rid = meta.get("run_id") or meta.get("ember_run_id")
            if self.encrypt_vendor_payload:
                uid = self.keystore.subject_ref(str(uid)) if uid is not None else None
                aid = self.keystore.subject_ref(str(aid)) if aid is not None else None
                rid = self.keystore.subject_ref(str(rid)) if rid is not None else None
            if uid is not None or aid is not None or rid is not None:
                register(user_id=uid, agent_id=aid, run_id=rid)

    def trace_turn(
        self,
        *,
        auto_cite: bool = True,
        prune: str | Callable[[str, list[QueryHit]], Sequence[str]] = "all",
    ) -> TurnTrace:
        """Open a thread- and async-safe turn scope capturing `search()` reads -> `remember()` `derived_from`."""
        return TurnTrace(vault=self, auto_cite=auto_cite, prune=prune)

    def _active_trace(self) -> TurnTrace | None:
        traces = _ACTIVE_TRACES.get()
        return traces.get(self._trace_key) if traces else None

    def __enter__(self) -> "CustodyVault":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def open(
        cls,
        data_dir: str | Path,
        *,
        key_dir: str | Path | None = None,
        tenant_id: str = "default",
        adapter: MemoryStoreAdapter | None = None,
        signer: Ed25519Signer | None = None,
        anchor: TimestampAnchor | None = None,
        encrypt_vendor_payload: bool = False,
        embedder: Any = None,
        ledger: LedgerBackend | None = None,
        keystore: KeyStore | None = None,
        keystore_backend: str = "file",
        auto_checkpoint_every: int | None = None,
        use_checkpoint: bool = True,
    ) -> "CustodyVault":
        """Open or create a persistent vault on disk (or plug custom `LedgerBackend` / `KeyStore`).

        For real deployments, pass an explicit `key_dir` outside `data_dir` (or pass a KMS-backed
        `KeyStore` / `PostgresLedger`) so ledger backups never contain live content decryption keys.
        """
        data_path = Path(data_dir).resolve()
        data_path.mkdir(parents=True, exist_ok=True)
        resolved_key_dir = Path(key_dir).resolve() if key_dir is not None else (data_path / "keys")

        resolved_ledger: LedgerBackend = ledger if ledger is not None else SQLiteLedger(data_path / "ledger.db")
        resolved_keystore: KeyStore | None = keystore
        owns_keystore = False
        try:
            if resolved_keystore is None:
                if keystore_backend == "sqlite":
                    resolved_keystore = SQLiteKeyStore(resolved_key_dir / "keys.db")
                    owns_keystore = True
                elif keystore_backend == "file":
                    resolved_keystore = FileKeyStore(resolved_key_dir / "keyring.json")
                else:
                    raise ValueError(f"unsupported keystore_backend: {keystore_backend!r}")

            resolved_signer = (
                signer if signer is not None else Ed25519Signer.load_or_create(resolved_key_dir / "ed25519.json")
            )

            ledger_path = getattr(resolved_ledger, "path", None)
            ks_path = getattr(resolved_keystore, "path", None)
            if isinstance(resolved_keystore, MemoryKeyStore) and not isinstance(resolved_keystore, FileKeyStore):
                separated = True
            elif hasattr(resolved_keystore, "key_id") or hasattr(resolved_keystore, "key_name"):
                separated = True
            elif ledger_path is None or str(ledger_path) == ":memory:":
                separated = True
            elif ks_path is not None and str(ks_path) != ":memory:":
                separated = not Path(str(ks_path)).resolve().is_relative_to(Path(str(ledger_path)).resolve().parent)
            else:
                separated = key_dir is not None and not resolved_key_dir.is_relative_to(data_path)

            vault = cls(
                ledger=resolved_ledger,
                keystore=resolved_keystore,
                signer=resolved_signer,
                tenant_id=tenant_id,
                adapter=adapter,
                anchor=anchor,
                keys_separated_from_ledger=separated,
                encrypt_vendor_payload=encrypt_vendor_payload,
                embedder=embedder,
                auto_checkpoint_every=auto_checkpoint_every,
                use_checkpoint=use_checkpoint,
            )
            vault._owns_keystore = owns_keystore
            return vault
        except BaseException:
            if ledger is None:
                resolved_ledger.close()
            if owns_keystore and resolved_keystore is not None:
                close_ks = getattr(resolved_keystore, "close", None)
                if callable(close_ks):
                    close_ks()
            raise

    @classmethod
    def ephemeral(
        cls,
        *,
        tenant_id: str = "default",
        adapter: MemoryStoreAdapter | None = None,
        signer: Ed25519Signer | None = None,
        anchor: TimestampAnchor | None = None,
        encrypt_vendor_payload: bool = False,
        embedder: Any = None,
        auto_checkpoint_every: int | None = None,
        use_checkpoint: bool = True,
    ) -> "CustodyVault":
        """Open a clean in-memory vault for fast tests, notebooks, and demos."""
        ledger = SQLiteLedger(":memory:")
        try:
            return cls(
                ledger=ledger,
                keystore=MemoryKeyStore(),
                signer=signer if signer is not None else Ed25519Signer.generate(),
                tenant_id=tenant_id,
                adapter=adapter,
                anchor=anchor,
                keys_separated_from_ledger=True,
                encrypt_vendor_payload=encrypt_vendor_payload,
                embedder=embedder,
                auto_checkpoint_every=auto_checkpoint_every,
                use_checkpoint=use_checkpoint,
            )
        except BaseException:
            ledger.close()
            raise

    def close(self) -> None:
        with self._lock:
            self.ledger.close()
            if self._owns_keystore:
                close_ks = getattr(self.keystore, "close", None)
                if callable(close_ks):
                    close_ks()

    def register_source(
        self,
        source_id: str,
        text: str,
        *,
        mode: ChunkMode | ChunkerFn = "sentence",
        span_mode: SpanMode | SpanChunkerFn = None,
        origin: str = "",
        origin_kind: str = "unknown",
        supplied_by: str = "",
        validated_by: str = "",
        ingested_at: str | None = None,
        event_id: str | None = None,
    ) -> tuple[SourceRecord, RedactableDoc]:
        """Hash, Merkle-index, store mutable leaves/spans, and commit an attested SourceRecord to the chain."""
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        with self._lock:
            doc = RedactableDoc.from_text(text, source_id=source_id, mode=mode, span_mode=span_mode)
            raw_bytes = text.encode("utf-8")
            ts = ingested_at or _utc_now()
            chunk_label = f"{doc.chunk_mode}+{doc.span_mode}" if doc.span_mode else doc.chunk_mode
            record = SourceRecord(
                source_id=source_id,
                merkle_root=doc.root(),
                flat_doc_hash=sha256_prefixed(raw_bytes),
                byte_length=len(raw_bytes),
                leaf_count=len(doc.entries),
                chunk_mode=chunk_label,
                ingested_at=ts,
                origin=origin or "undeclared",
                origin_kind=origin_kind,
                supplied_by=supplied_by or "undeclared",
                validated_by=validated_by or "undeclared",
            )
            ev_id = event_id or f"src:{source_id}:{uuid.uuid4().hex[:8]}"
            _, _, next_state = self.ledger.append(
                tenant_id=self.tenant_id,
                event_id=ev_id,
                kind="ADD_SOURCE",
                committed_at=ts,
                payload={"source": record.as_dict()},
                current_state=self.state,
                keystore=self.keystore,
                validate=True,
            )
            if next_state is None:
                raise RuntimeError("ledger.append unexpectedly returned None state")
            self.ledger.put_document(self.tenant_id, doc)
            self.state = next_state
            self._maybe_auto_checkpoint()
            return record, doc

    def _effective_owners(self, subjects: Sequence[str], derived_from: Sequence[str]) -> set[str]:
        owners = {s for s in subjects if s}
        visited: set[str] = set()
        stack = list(derived_from)
        while stack:
            pid = stack.pop()
            if pid in visited:
                continue
            visited.add(pid)
            parent = self.state.items.get(pid)
            if parent is None:
                raise InvariantViolation(f"cannot derive from unknown or erased parent item {pid!r}")
            owners.update(parent.subjects)
            stack.extend(parent.derived_from)
        return owners

    def _encrypt_item_fields(
        self,
        *,
        item_id: str,
        subj_tuple: tuple[str, ...],
        owner_refs: set[str],
        force_record_key: bool,
        store_text: str,
        plain_quote: str | None,
        meta: dict[str, Any],
    ) -> tuple[str, str, str, str, str | None, str]:
        if len(owner_refs) == 1 and not force_record_key:
            key_kind = "subject"
            key_id = next(iter(owner_refs))
            cek = self.keystore.get_or_create_subject_key(key_id)
        else:
            key_kind = "record"
            key_id = item_id
            cek = self.keystore.create_record_key(item_id, owner_refs)

        key_ref = f"{key_kind}:{key_id}"
        subjects_ct = encrypt_text(
            cek,
            canonical_bytes(list(subj_tuple)).decode("utf-8"),
            aad=build_aad(tenant_id=self.tenant_id, record_id=item_id, field="subjects", key_ref=key_ref),
        )
        text_ct = encrypt_text(
            cek,
            store_text,
            aad=build_aad(tenant_id=self.tenant_id, record_id=item_id, field="text", key_ref=key_ref),
        )
        quote_ct: str | None = None
        if plain_quote is not None:
            quote_ct = encrypt_text(
                cek,
                plain_quote,
                aad=build_aad(tenant_id=self.tenant_id, record_id=item_id, field="quote", key_ref=key_ref),
            )
        metadata_ct = encrypt_text(
            cek,
            canonical_bytes(meta).decode("utf-8"),
            aad=build_aad(tenant_id=self.tenant_id, record_id=item_id, field="metadata", key_ref=key_ref),
        )
        return key_kind, key_id, subjects_ct, text_ct, quote_ct, metadata_ct

    def _build_blind_tokens(self, text: str) -> str:
        toks = sorted(
            {
                bt
                for bt in (self.keystore.blind_token(w) for w in _content_tokens(text))
                if bt
            }
        )
        return " ".join(toks)

    def _translate_filters(self, filters: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if not filters:
            return None, {}
        if not self.encrypt_vendor_payload:
            return dict(filters), {}
        adapter_filters: dict[str, Any] = {}
        local_filters: dict[str, Any] = {}
        for key, val in filters.items():
            if key in ("user_id", "agent_id", "run_id"):
                if isinstance(val, str) and val.strip():
                    adapter_filters[key] = self.keystore.subject_ref(val)
                else:
                    adapter_filters[key] = val
            elif key == "ember_item_id":
                adapter_filters[key] = val
            else:
                local_filters[key] = val
        return (adapter_filters or None), local_filters

    def remember(
        self,
        text: str,
        *,
        subjects: Iterable[str],
        source_id: str | None = None,
        quote: str | None = None,
        leaf_index: int | None = None,
        derived_from: Iterable[str] = (),
        aliases: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        reconsent_id: str | None = None,
        item_id: str | None = None,
        committed_at: str | None = None,
        event_id: str | None = None,
        auto_derive: bool = True,
        content_guard: bool = True,
    ) -> RememberResult:
        """Encrypt, bind provenance receipt, commit to ledger, and upsert into the backing store."""
        subj_tuple = tuple(dict.fromkeys(str(s) for s in subjects if str(s).strip()))
        if not subj_tuple:
            raise ValueError("remember() requires at least one non-empty subject ID")
        if item_id is not None and (not isinstance(item_id, str) or not item_id.strip()):
            raise ValueError("item_id must be a non-empty string")
        if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
            raise ValueError("source_id must be a non-empty string")

        explicit_derived = [str(d) for d in derived_from if str(d).strip()]
        trace = self._active_trace()
        if auto_derive and trace is not None:
            traced_reads = trace.select_citations(text)
            explicit_derived = [*traced_reads, *explicit_derived]
        derived_tuple = tuple(dict.fromkeys(explicit_derived))
        alias_tuple = tuple(dict.fromkeys(str(a) for a in aliases if str(a).strip()))

        with self._lock:
            resolved_item_id = item_id or f"item_{uuid.uuid4().hex[:12]}"
            if event_id is None and resolved_item_id in self.state.items:
                raise InvariantViolation(f"item already exists: {resolved_item_id!r}")

            all_tokens = tuple(
                sorted(
                    {
                        tok
                        for tok in (self.keystore.tombstone_token(val) for val in (*subj_tuple, *alias_tuple))
                        if tok
                    }
                )
            )
            tomb_overlap = set(all_tokens) & self.state.tombstoned_tokens
            if tomb_overlap and not reconsent_id:
                raise TombstonedSubjectError(
                    "refusing ingest: subject has an active erasure tombstone and no reconsent_id"
                )

            if content_guard and self.state.tombstoned_content_tokens and not reconsent_id:
                cands = _extract_content_candidates(text, quote, metadata)
                c_tokens = {self.keystore.content_tombstone_token(c) for c in cands if c}
                if c_tokens & self.state.tombstoned_content_tokens:
                    raise TombstonedSubjectError(
                        "refusing ingest: content mentions a tombstoned subject identifier without explicit reconsent_id"
                    )

            effective_owners = self._effective_owners(subj_tuple, derived_tuple)
            ts = committed_at or _utc_now()

            receipt: Receipt | None = None
            receipt_meta: dict[str, Any] | None = None
            plain_quote: str | None = quote
            if source_id is not None:
                doc = self.ledger.get_document(self.tenant_id, source_id)
                if doc is None:
                    raise InvariantViolation(f"unknown source_id {source_id!r}; call register_source() first")
                src_rec = self.state.sources.get(source_id)
                target_quote = quote if quote is not None else text
                receipt = mint_redactable_receipt(
                    doc,
                    target_quote,
                    leaf_index=leaf_index,
                    source_id=source_id,
                )
                if src_rec is not None:
                    receipt = Receipt(
                        source_id=receipt.source_id,
                        quote=receipt.quote,
                        flat_doc_hash=src_rec.flat_doc_hash,
                        byte_start=receipt.byte_start,
                        byte_end=receipt.byte_end,
                        merkle_root=receipt.merkle_root,
                        leaf_index=receipt.leaf_index,
                        leaf_count=receipt.leaf_count,
                        proof=receipt.proof,
                        span_indices=receipt.span_indices,
                        span_count=receipt.span_count,
                        span_proofs=receipt.span_proofs,
                    )
                plain_quote = receipt.quote
                receipt_meta = {
                    "source_id": receipt.source_id,
                    "flat_doc_hash": receipt.flat_doc_hash,
                    "byte_start": receipt.byte_start,
                    "byte_end": receipt.byte_end,
                    "merkle_root": receipt.merkle_root,
                    "leaf_index": receipt.leaf_index,
                    "leaf_count": receipt.leaf_count,
                    "proof": [list(step) for step in receipt.proof],
                }
                if receipt.span_level:
                    receipt_meta["span_indices"] = list(receipt.span_indices)
                    receipt_meta["span_count"] = receipt.span_count
                    receipt_meta["span_proofs"] = [[list(step) for step in sp_p] for sp_p in receipt.span_proofs]

            meta = dict(metadata or {})
            owner_refs = {self.keystore.subject_ref(owner) for owner in effective_owners}
            store_text = text
            key_kind, key_id, subjects_ct, text_ct, quote_ct, metadata_ct = self._encrypt_item_fields(
                item_id=resolved_item_id,
                subj_tuple=subj_tuple,
                owner_refs=owner_refs,
                force_record_key=bool(derived_tuple),
                store_text=store_text,
                plain_quote=plain_quote,
                meta=meta,
            )

            pre_ext_map = self.state.adapter_id_map()
            vec_to_store: list[float] | None = None
            supports_vec = _adapter_supports_vector(self.adapter)
            if self.encrypt_vendor_payload:
                if self._embed_fn is not None and supports_vec:
                    vec_to_store = self._embed_fn(text)
                    adapter_text = text_ct
                else:
                    b_str = self._build_blind_tokens(f"{text} {' '.join(subj_tuple)} {' '.join(alias_tuple)}")
                    adapter_text = f"{text_ct} {b_str}".strip()
                adapter_subjects = tuple(sorted(owner_refs))
                adapter_meta: dict[str, Any] = {
                    "ember_item_id": resolved_item_id,
                    "ember_encrypted": True,
                    "ember_subject_refs": sorted(owner_refs),
                }
                for route_key in ("user_id", "agent_id", "run_id"):
                    if route_key in meta and meta[route_key] is not None and str(meta[route_key]).strip():
                        adapter_meta[route_key] = self.keystore.subject_ref(str(meta[route_key]))
            else:
                adapter_text = text
                adapter_subjects = tuple(sorted(effective_owners))
                adapter_meta = meta
                if self._embed_fn is not None and supports_vec:
                    vec_to_store = self._embed_fn(text)

            upsert_kw: dict[str, Any] = {
                "item_id": resolved_item_id,
                "text": adapter_text,
                "subjects": adapter_subjects,
                "metadata": adapter_meta,
            }
            if vec_to_store is not None and supports_vec:
                upsert_kw["vector"] = vec_to_store

            upsert_detailed_fn = getattr(self.adapter, "upsert_detailed", None)
            if callable(upsert_detailed_fn):
                ops: tuple[AdapterOp, ...] = tuple(upsert_detailed_fn(**upsert_kw))
            else:
                raw_ids = self.adapter.upsert(**upsert_kw)
                ops = tuple(AdapterOp(record_id=str(rid), event="ADD", memory=adapter_text) for rid in raw_ids if rid)

            ext_ids = tuple(dict.fromkeys(op.record_id for op in ops if op.record_id and op.event != "DELETE"))
            added_ids = tuple(dict.fromkeys(op.record_id for op in ops if op.record_id and op.event == "ADD"))

            # A3: If Mem0(infer=True) consolidated/updated or deleted existing records owned by prior items,
            # propagate those prior items into `derived_from`, promote to a shared `record` DEK, and re-encrypt!
            mutated_priors: list[str] = []
            consolidated_texts: list[str] = []
            has_delete_op = any(op.event == "DELETE" for op in ops)
            for op in ops:
                if op.event in ("UPDATE", "DELETE") and op.record_id:
                    prior_id = pre_ext_map.get(op.record_id)
                    if prior_id and prior_id != resolved_item_id and prior_id in self.state.items:
                        if prior_id not in derived_tuple:
                            mutated_priors.append(prior_id)
            if not self.encrypt_vendor_payload:
                for op in ops:
                    if not op.memory:
                        continue
                    if op.event == "UPDATE" or (op.event == "ADD" and has_delete_op and mutated_priors):
                        consolidated_texts.append(op.memory)

            if mutated_priors or consolidated_texts:
                derived_tuple = tuple(dict.fromkeys([*derived_tuple, *mutated_priors]))
                effective_owners = self._effective_owners(subj_tuple, derived_tuple)
                owner_refs = {self.keystore.subject_ref(owner) for owner in effective_owners}
                if consolidated_texts:
                    joined_mem = " ".join(dict.fromkeys(consolidated_texts))
                    if joined_mem != store_text:
                        meta.setdefault("ember_input_text", store_text)
                        store_text = joined_mem
                key_kind, key_id, subjects_ct, text_ct, quote_ct, metadata_ct = self._encrypt_item_fields(
                    item_id=resolved_item_id,
                    subj_tuple=subj_tuple,
                    owner_refs=owner_refs,
                    force_record_key=bool(derived_tuple or mutated_priors),
                    store_text=store_text,
                    plain_quote=plain_quote,
                    meta=meta,
                )

            item_obj = MemoryItem(
                item_id=resolved_item_id,
                tenant_id=self.tenant_id,
                subject_tokens=all_tokens,
                key_kind=key_kind,
                key_id=key_id,
                text_ciphertext=text_ct,
                subjects_ciphertext=subjects_ct,
                quote_ciphertext=quote_ct,
                metadata_ciphertext=metadata_ct,
                receipt_meta=receipt_meta,
                derived_from=derived_tuple,
                adapter_record_ids=ext_ids,
                created_at=ts,
                reconsent_id=reconsent_id,
                subjects=subj_tuple,
                text=store_text,
                quote=plain_quote,
                metadata=meta,
            )

            ev_id = event_id or f"item:{resolved_item_id}"
            try:
                ev, _, next_state = self.ledger.append(
                    tenant_id=self.tenant_id,
                    event_id=ev_id,
                    kind="ADD_ITEM",
                    committed_at=ts,
                    payload={"item": item_obj.ledger_payload()},
                    current_state=self.state,
                    keystore=self.keystore,
                    validate=True,
                )
            except BaseException:
                if added_ids:
                    self.adapter.delete(added_ids)
                raise

            if next_state is None:
                raise RuntimeError("ledger.append unexpectedly returned None state")
            self.state = next_state
            self._maybe_auto_checkpoint()
            if trace is not None:
                trace.record_write(resolved_item_id)
            return RememberResult(
                item=next_state.items[resolved_item_id],
                receipt=receipt,
                adapter_record_ids=ext_ids,
                event_hash=ev.event_hash,
            )

    def receipt_for(self, item: MemoryItem) -> Receipt | None:
        if item.receipt_meta is None or item.quote is None or item.quote == ERASED_TOMBSTONE:
            return None
        meta = item.receipt_meta
        raw_span_indices = meta.get("span_indices", ())
        raw_span_proofs = meta.get("span_proofs", ())
        return Receipt(
            source_id=str(meta["source_id"]),
            quote=item.quote,
            flat_doc_hash=str(meta.get("flat_doc_hash", "")),
            byte_start=int(meta["byte_start"]),
            byte_end=int(meta["byte_end"]),
            merkle_root=str(meta["merkle_root"]),
            leaf_index=int(meta["leaf_index"]),
            leaf_count=int(meta["leaf_count"]),
            proof=tuple((str(s), str(h)) for s, h in meta.get("proof", ())),
            span_indices=tuple(int(i) for i in raw_span_indices),
            span_count=int(meta.get("span_count", 0)),
            span_proofs=tuple(
                tuple((str(s), str(h)) for s, h in sp_proof)
                for sp_proof in raw_span_proofs
            ),
        )

    def verify_item_receipt(self, item_id: str) -> bool:
        with self._lock:
            item = self.state.items.get(item_id)
            if item is None:
                return False
            rec = self.receipt_for(item)
            if rec is None:
                return False
            doc = self.ledger.get_document(self.tenant_id, rec.source_id)
            if doc is None:
                return False
            return rec.verify_redactable(doc)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        verify_receipts: bool = True,
    ) -> SearchResult:
        """Query wrapped backend store, enforce live custody membership, and attach verified receipts."""
        if limit <= 0:
            return SearchResult(query=query, hits=(), filtered_orphans=0)
        with self._lock:
            adapter_filters, local_filters = self._translate_filters(filters)
            fetch_limit = max(limit * 4, limit) if local_filters else max(limit * 2, limit)
            supports_vec = _adapter_supports_vector(self.adapter)
            if self.encrypt_vendor_payload:
                if self._embed_fn is not None and supports_vec:
                    q_vec = self._embed_fn(query)
                    raw_hits = self.adapter.search_vector(q_vec, limit=fetch_limit, filters=adapter_filters)  # type: ignore[attr-defined]
                else:
                    blind_query = self._build_blind_tokens(query)
                    raw_hits = (
                        self.adapter.search(blind_query, limit=fetch_limit, filters=adapter_filters)
                        if blind_query
                        else []
                    )
            else:
                if self._embed_fn is not None and supports_vec:
                    q_vec = self._embed_fn(query)
                    raw_hits = self.adapter.search_vector(q_vec, limit=fetch_limit, filters=adapter_filters)  # type: ignore[attr-defined]
                else:
                    raw_hits = self.adapter.search(query, limit=fetch_limit, filters=adapter_filters)

            ext_map = self.state.adapter_id_map()
            hits: list[QueryHit] = []
            filtered_orphans = 0

            for hit in raw_hits:
                item_id = ext_map.get(hit.record_id) or hit.metadata.get("ember_item_id")
                if not item_id or item_id not in self.state.items:
                    filtered_orphans += 1
                    continue
                item = self.state.items[item_id]
                if item.text == ERASED_TOMBSTONE:
                    filtered_orphans += 1
                    continue
                if local_filters and any(item.metadata.get(k) != v for k, v in local_filters.items()):
                    continue
                rec = self.receipt_for(item)
                rec_ok: bool | None = None
                if verify_receipts and rec is not None:
                    doc = self.ledger.get_document(self.tenant_id, rec.source_id)
                    rec_ok = rec.verify_redactable(doc) if doc is not None else False
                if self.encrypt_vendor_payload or hit.metadata.get("ember_encrypted"):
                    clean_meta = dict(item.metadata)
                    clean_meta.setdefault("ember_item_id", item.item_id)
                else:
                    clean_meta = dict(hit.metadata)
                hits.append(
                    QueryHit(
                        item_id=item.item_id,
                        record_id=hit.record_id,
                        text=item.text if item.text is not None else hit.text,
                        score=hit.score,
                        subjects=item.subjects,
                        receipt=rec,
                        receipt_verified=rec_ok,
                        metadata=clean_meta,
                    )
                )
                if len(hits) >= limit:
                    break

            trace = self._active_trace()
            if trace is not None:
                for qh in hits:
                    trace.record_read(qh)
            return SearchResult(query=query, hits=tuple(hits), filtered_orphans=filtered_orphans)

    def _probe_raw_leaks(
        self,
        subject_id: str,
        aliases: Sequence[str],
        doomed_item_ids: set[str],
        doomed_adapter_ids: set[str],
        probes: Sequence[str],
        markers: Sequence[str],
    ) -> list[str]:
        """Inspect raw adapter AND live materialized state after erasure to catch ANY residue."""
        leaking: list[str] = []
        target_ids_norm = {" ".join(x.strip().lower().split()) for x in (subject_id, *aliases) if str(x).strip()}
        target_refs = {self.keystore.subject_ref(x) for x in (subject_id, *aliases) if str(x).strip()}
        supports_vec = _adapter_supports_vector(self.adapter)

        # Defensive check against live adapter directly (do NOT mask stale vendor entries)
        for probe in probes:
            leaked = False
            candidate_hits: list[AdapterHit] = []
            if self._embed_fn is not None and supports_vec:
                candidate_hits.extend(self.adapter.search_vector(self._embed_fn(probe), limit=25))  # type: ignore[attr-defined]
            elif self.encrypt_vendor_payload:
                blind_q = self._build_blind_tokens(probe)
                if blind_q:
                    candidate_hits.extend(self.adapter.search(blind_q, limit=25))
            else:
                candidate_hits.extend(self.adapter.search(probe, limit=25))

            for hit in candidate_hits:
                if hit.record_id in doomed_adapter_ids:
                    leaked = True
                    break
                meta_item_id = hit.metadata.get("ember_item_id")
                if meta_item_id in doomed_item_ids:
                    leaked = True
                    break
                for id_key in ("user_id", "ember_user_id"):
                    uid = hit.metadata.get(id_key)
                    if isinstance(uid, str):
                        if " ".join(uid.strip().lower().split()) in target_ids_norm or uid in target_refs:
                            leaked = True
                            break
                if leaked:
                    break
                hit_subjects = hit.metadata.get("ember_subjects")
                if isinstance(hit_subjects, (list, tuple)) and any(
                    " ".join(str(s).strip().lower().split()) in target_ids_norm for s in hit_subjects
                ):
                    leaked = True
                    break
                hit_refs = hit.metadata.get("ember_subject_refs")
                if isinstance(hit_refs, (list, tuple)) and any(str(r) in target_refs for r in hit_refs):
                    leaked = True
                    break
                if _marker_matches(hit.text, markers):
                    leaked = True
                    break
                if isinstance(meta_item_id, str) and meta_item_id in self.state.items:
                    surviving_item = self.state.items[meta_item_id]
                    if surviving_item.text and _marker_matches(surviving_item.text, markers):
                        leaked = True
                        break
            if not leaked:
                q_toks = _content_tokens(probe)
                for item in self.state.items.values():
                    text_val = item.text or ""
                    if q_toks & _tokenize(text_val):
                        if any(
                            " ".join(s.strip().lower().split()) in target_ids_norm for s in item.subjects
                        ) or _marker_matches(text_val, markers):
                            leaked = True
                            break
            if leaked:
                leaking.append(probe)
        return leaking

    def erase(
        self,
        subject_id: str,
        *,
        aliases: Iterable[str] = (),
        probes: Iterable[str] = (),
        markers: Iterable[str] = (),
        block_identifiers: Iterable[str] = (),
        leaf_predicate: Callable[[str], bool] | None = None,
        span_predicate: Callable[[str], bool] | None = None,
        auto_probes: bool = True,
        committed_at: str | None = None,
    ) -> ErasureCertificate:
        """Execute irreversible cascade erasure, key shredding, doc redaction, and probe certification."""
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("subject_id must be a non-empty string")
        alias_list = [str(a) for a in aliases if str(a).strip()]
        probe_list = [str(p) for p in probes if str(p).strip()]
        marker_list = list(dict.fromkeys([subject_id, *alias_list, *(str(m) for m in markers if str(m).strip())]))

        with self._lock:
            before_hash = self.state.digest()
            subject_tokens = tuple(
                sorted(
                    {
                        tok
                        for tok in (self.keystore.tombstone_token(val) for val in (subject_id, *alias_list))
                        if tok
                    }
                )
            )
            token_set = set(subject_tokens)

            id_candidates = [val for val in (subject_id, *alias_list) if _is_auto_block_identifier(val)]
            id_candidates.extend(str(b) for b in block_identifiers if str(b).strip())
            content_tokens = tuple(
                sorted(
                    {
                        tok
                        for tok in (self.keystore.content_tombstone_token(val) for val in id_candidates)
                        if tok
                    }
                )
            )

            # Gather any prior matching ERASE_SUBJECT closures (enables safe retry after transient adapter failures)
            prior_events = self.ledger.events(self.tenant_id)
            prior_erase_found = False
            prior_item_ids: list[str] = []
            prior_derived_ids: list[str] = []
            prior_adapter_ids: list[str] = []
            prior_source_leaves: dict[str, set[int]] = {}
            prior_source_spans: dict[str, dict[int, set[int]]] = {}
            for pev in prior_events:
                if pev.kind != "ERASE_SUBJECT":
                    continue
                ev_tokens = {str(t) for t in pev.payload.get("subject_tokens", ()) if t}
                if not (ev_tokens & token_set):
                    continue
                prior_erase_found = True
                pclosure = pev.payload.get("closure") or {}
                prior_item_ids.extend(str(x) for x in pclosure.get("item_ids", ()))
                prior_derived_ids.extend(str(x) for x in pclosure.get("derived_invalidated", ()))
                prior_adapter_ids.extend(str(x) for x in pclosure.get("adapter_record_ids", ()))
                for sid, idxs in (pclosure.get("source_leaves_redacted") or {}).items():
                    prior_source_leaves.setdefault(str(sid), set()).update(int(i) for i in idxs)
                for sid, leaf_map in (pclosure.get("source_spans_redacted") or {}).items():
                    target = prior_source_spans.setdefault(str(sid), {})
                    for leaf_k, sp_idxs in leaf_map.items():
                        target.setdefault(int(leaf_k), set()).update(int(i) for i in sp_idxs)

            # Scan source docs for explicit leaf/span predicates if requested (in addition to cited closure leaves/spans)
            extra_leaves: dict[str, list[int]] = {}
            extra_spans: dict[str, dict[int, list[int]]] = {}
            if leaf_predicate is not None or span_predicate is not None:
                for sid in self.ledger.list_document_ids(self.tenant_id):
                    doc = self.ledger.get_document(self.tenant_id, sid)
                    if doc is None:
                        continue
                    matched_leaves: list[int] = []
                    matched_doc_spans: dict[int, list[int]] = {}
                    for e in doc.entries:
                        if e.get("redacted"):
                            continue
                        l_idx = int(e["index"])
                        if leaf_predicate is not None and leaf_predicate(_entry_text(e)):
                            matched_leaves.append(l_idx)
                        elif span_predicate is not None and "spans" in e:
                            s_hits = [
                                int(sp["span_index"])
                                for sp in e["spans"]
                                if not sp.get("redacted") and span_predicate(str(sp["text"]))
                            ]
                            if s_hits:
                                matched_doc_spans[l_idx] = s_hits
                    if matched_leaves:
                        extra_leaves[sid] = matched_leaves
                    if matched_doc_spans:
                        extra_spans[sid] = matched_doc_spans

            closure = compute_closure(
                self.state,
                subject_id,
                subject_tokens=subject_tokens,
                extra_source_leaves=extra_leaves or None,
                extra_source_spans=extra_spans or None,
            )

            doomed_items = [self.state.items[iid] for iid in closure.item_ids if iid in self.state.items]
            effective_probes = (
                _synthesize_auto_probes(
                    subject_id=subject_id,
                    aliases=alias_list,
                    markers=marker_list,
                    doomed_items=doomed_items,
                    explicit_probes=probe_list,
                )
                if auto_probes
                else probe_list
            )

            ts = committed_at or _utc_now()
            erase_ev_id = f"erase:{subject_id}:{uuid.uuid4().hex[:8]}"
            ev, _, next_state = self.ledger.append(
                tenant_id=self.tenant_id,
                event_id=erase_ev_id,
                kind="ERASE_SUBJECT",
                committed_at=ts,
                payload={
                    "subject_tokens": list(subject_tokens),
                    "content_tokens": list(content_tokens),
                    "closure": closure.as_dict(),
                },
                current_state=self.state,
                keystore=self.keystore,
                validate=True,
            )
            if next_state is None:
                raise RuntimeError("ledger.append unexpectedly returned None state")
            self.state = next_state

            # 1. Local irreversible leaf/span redaction in DocumentStore FIRST (survives transient external failures)
            merged_source_leaves: dict[str, set[int]] = {k: set(v) for k, v in prior_source_leaves.items()}
            for sid, leaf_indices in closure.source_leaves_redacted.items():
                merged_source_leaves.setdefault(sid, set()).update(leaf_indices)

            merged_source_spans: dict[str, dict[int, set[int]]] = {
                sid: {l_idx: set(s_idxs) for l_idx, s_idxs in l_map.items()}
                for sid, l_map in prior_source_spans.items()
            }
            for sid, l_map in closure.source_spans_redacted.items():
                target_sid = merged_source_spans.setdefault(sid, {})
                for l_idx, s_idxs in l_map.items():
                    target_sid.setdefault(int(l_idx), set()).update(int(i) for i in s_idxs)

            for sid in sorted(set(merged_source_leaves) | set(merged_source_spans)):
                doc = self.ledger.get_document(self.tenant_id, sid)
                if doc is None:
                    continue
                updated_doc = doc
                whole = merged_source_leaves.get(sid)
                if whole:
                    updated_doc = updated_doc.redact_indices(whole)
                sp_map = merged_source_spans.get(sid)
                if sp_map:
                    updated_doc = updated_doc.redact_leaf_spans(sp_map)
                    if whole:
                        for l_idx in whole:
                            sp_map.pop(l_idx, None)
                    if not sp_map:
                        merged_source_spans.pop(sid, None)
                if updated_doc is not doc:
                    self.ledger.put_document(self.tenant_id, updated_doc)

            # 2. Local irreversible AES-256-GCM key shredding BEFORE external network calls!
            targets_to_shred = list(dict.fromkeys([subject_id, *alias_list]))
            key_destroyed = False
            record_keys_destroyed = 0
            for target in targets_to_shred:
                rep_direct = self.keystore.shred_subject(target)
                rep_ref = self.keystore.shred_subject(self.keystore.subject_ref(target))
                key_destroyed = key_destroyed or rep_direct.subject_key_destroyed or rep_ref.subject_key_destroyed
                record_keys_destroyed += rep_direct.record_keys_destroyed + rep_ref.record_keys_destroyed
            destroyed_at = _utc_now()

            # Check prior interrupted erase cycles so retries report key destruction accurately when keys are gone
            all_target_keys_absent = all(
                self.keystore.get_subject_key(k) is None
                for target in targets_to_shred
                for k in (target, self.keystore.subject_ref(target))
            )
            prior_records_shredded = all(
                self.keystore.get_record_key(rid) is None for rid in prior_item_ids
            )
            if prior_erase_found and prior_item_ids and all_target_keys_absent and prior_records_shredded:
                key_destroyed = True

            # 3. External store cascade deletion (safe to fail/retry cleanly without leaving keys alive!)
            merged_adapter_ids = tuple(dict.fromkeys([*prior_adapter_ids, *closure.adapter_record_ids]))
            if merged_adapter_ids:
                self.adapter.delete(merged_adapter_ids)

            # Ensure scope-aware stores (such as Mem0 2.x) probe target subject/alias scopes even on cold retry.
            register_scope = getattr(self.adapter, "register_scope", None)
            if callable(register_scope):
                for target in (subject_id, *alias_list):
                    scope_uid = self.keystore.subject_ref(target) if self.encrypt_vendor_payload else target
                    register_scope(user_id=scope_uid)

            merged_item_ids = tuple(sorted(set(prior_item_ids) | set(closure.item_ids)))
            merged_derived_ids = tuple(sorted(set(prior_derived_ids) | set(closure.derived_invalidated)))
            effective_closure: dict[str, Any] = {
                "subject_id": subject_id,
                "item_ids": list(merged_item_ids),
                "derived_invalidated": list(merged_derived_ids),
                "source_leaves_redacted": {k: sorted(v) for k, v in sorted(merged_source_leaves.items())},
                "adapter_record_ids": list(merged_adapter_ids),
            }
            if merged_source_spans:
                effective_closure["source_spans_redacted"] = {
                    sid: {str(l_idx): sorted(s_idxs) for l_idx, s_idxs in sorted(l_map.items())}
                    for sid, l_map in sorted(merged_source_spans.items())
                }

            # 4. Verify probes against raw backing store & surviving state BEFORE issuing certificate
            leaking = self._probe_raw_leaks(
                subject_id,
                alias_list,
                set(merged_item_ids),
                set(merged_adapter_ids),
                effective_probes,
                marker_list,
            )
            if leaking:
                raise ErasureIncomplete(subject_id, leaking, closure)

            after_hash = self.state.digest()
            issued_at = _utc_now()
            cert_id = f"cert_{uuid.uuid4().hex[:12]}"

            anchor_digest = sha256_prefixed(
                json.dumps(
                    {
                        "certificate_id": cert_id,
                        "before": before_hash,
                        "after": after_hash,
                        "tail": ev.event_hash,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            anchor_receipt = self.anchor.anchor(anchor_digest)

            trust_boundary = {
                "adversary_checkable_signature": True,
                "externally_timestamped": anchor_receipt.scheme != "none",
                "keys_separated_from_ledger": self.keys_separated_from_ledger,
                "live_store_covered": True,
                "pre_erasure_backups_covered": False,
                "vendor_payload_encrypted": self.encrypt_vendor_payload,
                "notes": (
                    "Certifies live projection closure, backing adapter ID deletion, salted Merkle leaf/span redaction, "
                    "and AES-256-GCM key shredding on this store. Backups created prior to erasure that copied "
                    "unseparated keys remain outside technical coverage."
                ),
            }

            unsigned = {
                "certificate_id": cert_id,
                "tenant_id": self.tenant_id,
                "subject_id": subject_id,
                "subject_tokens": list(subject_tokens),
                "state_hash_before": before_hash,
                "state_hash_after": after_hash,
                "ledger_tail_hash": ev.event_hash,
                "closure": effective_closure,
                "probe_count": len(effective_probes),
                "battery_all_abstained": True,
                "cipher": CIPHER_NAME,
                "key_destroyed": key_destroyed or record_keys_destroyed > 0,
                "record_keys_destroyed": record_keys_destroyed,
                "key_destroyed_at": destroyed_at,
                "issued_at": issued_at,
                "trust_boundary": trust_boundary,
                "anchor": anchor_receipt.as_dict(),
            }
            signed_dict = sign_certificate_dict(unsigned, self.signer)
            cert = ErasureCertificate(
                certificate_id=signed_dict["certificate_id"],
                tenant_id=signed_dict["tenant_id"],
                subject_id=signed_dict["subject_id"],
                subject_tokens=tuple(signed_dict["subject_tokens"]),
                state_hash_before=signed_dict["state_hash_before"],
                state_hash_after=signed_dict["state_hash_after"],
                ledger_tail_hash=signed_dict["ledger_tail_hash"],
                closure=signed_dict["closure"],
                probe_count=signed_dict["probe_count"],
                battery_all_abstained=signed_dict["battery_all_abstained"],
                cipher=signed_dict["cipher"],
                key_destroyed=signed_dict["key_destroyed"],
                record_keys_destroyed=signed_dict["record_keys_destroyed"],
                key_destroyed_at=signed_dict["key_destroyed_at"],
                issued_at=signed_dict["issued_at"],
                trust_boundary=signed_dict["trust_boundary"],
                anchor=signed_dict["anchor"],
                signer_scheme=signed_dict["signer_scheme"],
                public_key=signed_dict["public_key"],
                signature=signed_dict["signature"],
            )
            self.ledger.put_certificate(cert.as_dict())
            self._maybe_auto_checkpoint()
            return cert

    def replay(
        self,
        *,
        keystore: KeyStore | None = None,
        use_checkpoint: bool = False,
    ) -> EmberState:
        """Reconstruct materialized custody state via full genesis replay (or checkpoint + delta)."""
        with self._lock:
            return self._replay_state(
                keystore=keystore if keystore is not None else self.keystore,
                use_checkpoint=use_checkpoint,
            )

    def export_bundle(self, *, include_ledger: bool = True) -> dict[str, Any]:
        """Export self-contained redacted verification bundle checkable offline by `ember.verify`."""
        with self._lock:
            docs: dict[str, Any] = {}
            for sid in self.ledger.list_document_ids(self.tenant_id):
                doc = self.ledger.get_document(self.tenant_id, sid)
                if doc is not None:
                    docs[sid] = doc.as_dict()

            receipts: list[dict[str, Any]] = []
            for item in self.state.items.values():
                rec = self.receipt_for(item)
                if rec is not None and rec.redaction_safe:
                    receipts.append(rec.as_dict())

            bundle: dict[str, Any] = {
                "format": "ember-bundle/v1",
                "tenant_id": self.tenant_id,
                "documents": docs,
                "receipts": receipts,
                "certificates": self.ledger.list_certificates(self.tenant_id),
            }
            if include_ledger:
                bundle["ledger_events"] = [ev.as_dict() for ev in self.ledger.events(self.tenant_id)]
            return bundle


class AsyncCustodyVault:
    """Native `asyncio` wrapper around `CustodyVault` for async agent frameworks & FastAPI services."""

    def __init__(self, vault: CustodyVault) -> None:
        self._vault = vault

    @property
    def sync(self) -> CustodyVault:
        return self._vault

    @property
    def tenant_id(self) -> str:
        return self._vault.tenant_id

    @property
    def signer(self) -> Ed25519Signer:
        return self._vault.signer

    @property
    def state(self) -> EmberState:
        return self._vault.state

    @property
    def ledger(self) -> LedgerBackend:
        return self._vault.ledger

    @property
    def keystore(self) -> KeyStore:
        return self._vault.keystore

    @property
    def adapter(self) -> MemoryStoreAdapter:
        return self._vault.adapter

    @classmethod
    async def open(cls, data_dir: str | Path, **kwargs: Any) -> "AsyncCustodyVault":
        vault = await asyncio.to_thread(CustodyVault.open, data_dir, **kwargs)
        return cls(vault)

    @classmethod
    async def ephemeral(cls, **kwargs: Any) -> "AsyncCustodyVault":
        vault = await asyncio.to_thread(CustodyVault.ephemeral, **kwargs)
        return cls(vault)

    async def __aenter__(self) -> "AsyncCustodyVault":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await asyncio.to_thread(self._vault.close)

    def trace_turn(
        self,
        *,
        auto_cite: bool = True,
        prune: str | Callable[[str, list[QueryHit]], Sequence[str]] = "all",
    ) -> TurnTrace:
        """Return an async context manager tracking concurrent `search()` -> `remember()` lineage."""
        return self._vault.trace_turn(auto_cite=auto_cite, prune=prune)

    async def register_source(
        self,
        source_id: str,
        text: str,
        **kwargs: Any,
    ) -> tuple[SourceRecord, RedactableDoc]:
        return await asyncio.to_thread(self._vault.register_source, source_id, text, **kwargs)

    async def remember(
        self,
        text: str,
        *,
        subjects: Iterable[str],
        **kwargs: Any,
    ) -> RememberResult:
        return await asyncio.to_thread(self._vault.remember, text, subjects=subjects, **kwargs)

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        verify_receipts: bool = True,
    ) -> SearchResult:
        return await asyncio.to_thread(
            self._vault.search,
            query,
            limit=limit,
            filters=filters,
            verify_receipts=verify_receipts,
        )

    async def erase(self, subject_id: str, **kwargs: Any) -> ErasureCertificate:
        return await asyncio.to_thread(self._vault.erase, subject_id, **kwargs)

    async def checkpoint(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._vault.checkpoint)

    async def replay(
        self,
        *,
        keystore: KeyStore | None = None,
        use_checkpoint: bool = False,
    ) -> EmberState:
        return await asyncio.to_thread(self._vault.replay, keystore=keystore, use_checkpoint=use_checkpoint)

    async def verify_item_receipt(self, item_id: str) -> bool:
        return await asyncio.to_thread(self._vault.verify_item_receipt, item_id)

    async def export_bundle(self, *, include_ledger: bool = True) -> dict[str, Any]:
        return await asyncio.to_thread(self._vault.export_bundle, include_ledger=include_ledger)
