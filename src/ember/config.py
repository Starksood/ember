"""Zero-dependency `ember.toml` declarative configuration loader and vault factory (`tomllib`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import tomllib
from typing import Any

from .adapters import (
    ChromaAdapter,
    InMemoryStoreAdapter,
    MemoryStoreAdapter,
    PgVectorAdapter,
    PineconeAdapter,
    QdrantAdapter,
    SQLiteStoreAdapter,
)
from .keystore import FileKeyStore, KeyStore, MemoryKeyStore, SQLiteKeyStore
from .kms import AwsKmsKeyStore, GcpKmsKeyStore, VaultTransitKeyStore
from .ledger import LedgerBackend, PostgresLedger, SQLiteLedger
from .signing import Ed25519Signer, NullAnchor, RekorAnchor, Rfc3161Anchor, TimestampAnchor
from .vault import CustodyVault

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_str(raw: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        val = os.environ.get(name)
        if val is not None and val != "":
            return val
        if default is not None:
            return default
        raise ValueError(f"missing required environment variable ${{{name}}} referenced in config")

    return _ENV_PATTERN.sub(_sub, raw)


def _expand_tree(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env_str(obj)
    if isinstance(obj, dict):
        return {k: _expand_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_tree(v) for v in obj]
    return obj


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p).resolve()


@dataclass(frozen=True)
class EmberConfig:
    """Typed declarative representation of `ember.toml`."""

    data_dir: Path = field(default_factory=lambda: Path(".ember"))
    key_dir: Path | None = None
    tenant_id: str = "default"
    encrypt_vendor_payload: bool = False
    auto_checkpoint_every: int | None = None
    use_checkpoint: bool = True
    ledger: dict[str, Any] = field(default_factory=lambda: {"backend": "sqlite"})
    keystore: dict[str, Any] = field(default_factory=lambda: {"backend": "file"})
    adapter: dict[str, Any] = field(default_factory=lambda: {"backend": "sqlite"})
    anchor: dict[str, Any] = field(default_factory=lambda: {"backend": "none"})
    server: dict[str, Any] = field(default_factory=lambda: {"host": "127.0.0.1", "port": 8420})
    source_path: Path | None = None

    @property
    def base_dir(self) -> Path:
        return self.source_path.parent if self.source_path is not None else Path.cwd()


def find_ember_config(start: str | Path | None = None) -> Path | None:
    """Locate `ember.toml` in `start` or its nearest ancestor directory."""
    cur = Path(start).resolve() if start is not None else Path.cwd().resolve()
    if cur.is_file():
        return cur
    for candidate_dir in (cur, *cur.parents):
        cfg = candidate_dir / "ember.toml"
        if cfg.is_file():
            return cfg
    return None


def parse_ember_config_dict(raw: Mapping[str, Any], *, base_dir: Path | None = None, source_path: Path | None = None) -> EmberConfig:
    """Validate and expand an in-memory config mapping (parsed from `ember.toml`)."""
    expanded = _expand_tree(dict(raw))
    resolved_base = (base_dir or (source_path.parent if source_path is not None else Path.cwd())).resolve()

    allowed_sections = {"vault", "ledger", "keystore", "adapter", "anchor", "server"}
    unknown = set(expanded) - allowed_sections
    if unknown:
        raise ValueError(f"unknown ember.toml top-level section(s): {', '.join(sorted(unknown))}")

    vault_sec = dict(expanded.get("vault") or {})
    data_dir_raw = vault_sec.get("data_dir", ".ember")
    data_dir = _resolve_path(resolved_base, data_dir_raw)

    key_dir_raw = vault_sec.get("key_dir")
    key_dir = _resolve_path(resolved_base, key_dir_raw) if key_dir_raw else None

    tenant_id = str(vault_sec.get("tenant_id", "default"))
    encrypt_vendor_payload = bool(vault_sec.get("encrypt_vendor_payload", False))
    auto_ckpt_raw = vault_sec.get("auto_checkpoint_every")
    auto_checkpoint_every = int(auto_ckpt_raw) if auto_ckpt_raw is not None else None
    use_checkpoint = bool(vault_sec.get("use_checkpoint", True))

    ledger_sec = dict(expanded.get("ledger") or {"backend": "sqlite"})
    if "path" in ledger_sec and str(ledger_sec["path"]) != ":memory:":
        ledger_sec["path"] = str(_resolve_path(resolved_base, ledger_sec["path"]))

    ks_sec = dict(expanded.get("keystore") or {"backend": "file"})
    if "path" in ks_sec and str(ks_sec["path"]) != ":memory:":
        ks_sec["path"] = str(_resolve_path(resolved_base, ks_sec["path"]))

    adapter_sec = dict(expanded.get("adapter") or {"backend": "sqlite"})
    if "path" in adapter_sec and str(adapter_sec["path"]) != ":memory:":
        adapter_sec["path"] = str(_resolve_path(resolved_base, adapter_sec["path"]))
    if "path_dir" in adapter_sec:
        adapter_sec["path_dir"] = str(_resolve_path(resolved_base, adapter_sec["path_dir"]))

    anchor_sec = dict(expanded.get("anchor") or {"backend": "none"})
    server_sec = dict(expanded.get("server") or {"host": "127.0.0.1", "port": 8420})

    return EmberConfig(
        data_dir=data_dir,
        key_dir=key_dir,
        tenant_id=tenant_id,
        encrypt_vendor_payload=encrypt_vendor_payload,
        auto_checkpoint_every=auto_checkpoint_every,
        use_checkpoint=use_checkpoint,
        ledger=ledger_sec,
        keystore=ks_sec,
        adapter=adapter_sec,
        anchor=anchor_sec,
        server=server_sec,
        source_path=source_path,
    )


def load_ember_config(path: str | Path) -> EmberConfig:
    """Read and parse `ember.toml` from disk."""
    cfg_path = Path(path).resolve()
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    return parse_ember_config_dict(raw, base_dir=cfg_path.parent, source_path=cfg_path)


def build_anchor(cfg_anchor: Mapping[str, Any], *, signer: Ed25519Signer | None = None) -> TimestampAnchor:
    backend = str(cfg_anchor.get("backend", "none")).lower()
    timeout = float(cfg_anchor.get("timeout", 10.0))
    if backend in ("none", "null", ""):
        return NullAnchor()
    if backend == "rfc3161":
        tsa_url = str(cfg_anchor.get("tsa_url", "")).strip()
        if not tsa_url:
            raise ValueError("[anchor] backend='rfc3161' requires non-empty tsa_url")
        return Rfc3161Anchor(
            tsa_url,
            cert_req=bool(cfg_anchor.get("cert_req", True)),
            timeout=timeout,
        )
    if backend == "rekor":
        rekor_url = str(cfg_anchor.get("rekor_url", "https://rekor.sigstore.dev")).strip()
        return RekorAnchor(rekor_url=rekor_url, signer=signer, timeout=timeout)
    raise ValueError(f"unsupported anchor backend in config: {backend!r}")


def build_keystore(
    cfg_ks: Mapping[str, Any],
    *,
    key_dir: Path,
    tenant_id: str,
) -> tuple[KeyStore, bool]:
    backend = str(cfg_ks.get("backend", "file")).lower()
    if backend == "memory":
        return MemoryKeyStore(), False
    if backend == "file":
        path = Path(str(cfg_ks.get("path", key_dir / "keyring.json")))
        return FileKeyStore(path), False
    if backend == "sqlite":
        path = str(cfg_ks.get("path", key_dir / "keys.db"))
        return SQLiteKeyStore(path), True
    if backend == "aws_kms":
        key_id = str(cfg_ks.get("key_id", "")).strip()
        region = cfg_ks.get("region_name") or None
        store_path = str(cfg_ks.get("path", key_dir / "aws_kms_wrapped.db"))
        return AwsKmsKeyStore(key_id, store_path, tenant_id=tenant_id, region_name=region), True
    if backend == "gcp_kms":
        key_name = str(cfg_ks.get("key_name", "")).strip()
        store_path = str(cfg_ks.get("path", key_dir / "gcp_kms_wrapped.db"))
        return GcpKmsKeyStore(key_name, store_path, tenant_id=tenant_id), True
    if backend in ("vault_transit", "vault"):
        vault_addr = str(cfg_ks.get("vault_addr", "")).strip()
        key_name = str(cfg_ks.get("key_name", "")).strip()
        token = str(cfg_ks.get("token", "")).strip()
        mount_point = str(cfg_ks.get("mount_point", "transit")).strip()
        namespace = cfg_ks.get("namespace") or None
        store_path = str(cfg_ks.get("path", key_dir / "vault_wrapped.db"))
        return (
            VaultTransitKeyStore(
                vault_addr,
                key_name,
                token,
                store_path,
                tenant_id=tenant_id,
                mount_point=mount_point,
                namespace=str(namespace) if namespace else None,
            ),
            True,
        )
    raise ValueError(f"unsupported keystore backend in config: {backend!r}")


def build_ledger(cfg_ledger: Mapping[str, Any], *, data_dir: Path) -> LedgerBackend:
    backend = str(cfg_ledger.get("backend", "sqlite")).lower()
    if backend == "sqlite":
        path = str(cfg_ledger.get("path", data_dir / "ledger.db"))
        return SQLiteLedger(path)
    if backend == "postgres":
        dsn = str(cfg_ledger.get("dsn", "")).strip()
        prefix = str(cfg_ledger.get("table_prefix", "ember_"))
        return PostgresLedger(dsn, table_prefix=prefix)
    raise ValueError(f"unsupported ledger backend in config: {backend!r}")


def build_adapter(
    cfg_adapter: Mapping[str, Any],
    *,
    data_dir: Path,
    embed_fn: Callable[[str], Any] | None = None,
) -> tuple[MemoryStoreAdapter, Callable[[], None] | None]:
    backend = str(cfg_adapter.get("backend", "sqlite")).lower()
    if backend == "memory":
        return InMemoryStoreAdapter(), None
    if backend == "sqlite":
        path = str(cfg_adapter.get("path", data_dir / "store.db"))
        store = SQLiteStoreAdapter(path)
        return store, store.close
    if backend == "qdrant":
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("Qdrant adapter config requires 'qdrant-client' (`pip install 'ember-custody[qdrant]'`)") from exc
        url = cfg_adapter.get("url")
        api_key = cfg_adapter.get("api_key")
        path_dir = cfg_adapter.get("path_dir")
        client = QdrantClient(url=url, api_key=api_key, path=path_dir) if path_dir else QdrantClient(url=url, api_key=api_key)
        col = str(cfg_adapter.get("collection_name", "ember"))
        return QdrantAdapter(client, col, embed_fn=embed_fn), getattr(client, "close", None)
    if backend == "chroma":
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Chroma adapter config requires 'chromadb' (`pip install 'ember-custody[chroma]'`)") from exc
        path_dir = cfg_adapter.get("path_dir")
        client = chromadb.PersistentClient(path=str(path_dir)) if path_dir else chromadb.Client()
        col_name = str(cfg_adapter.get("collection_name", "ember"))
        collection = client.get_or_create_collection(name=col_name)
        return ChromaAdapter(collection, embed_fn=embed_fn), None
    if backend == "pgvector":
        dsn = str(cfg_adapter.get("dsn", "")).strip()
        if not dsn:
            raise ValueError("[adapter] backend='pgvector' requires non-empty dsn")
        from .ledger import _open_pg_connection

        conn = _open_pg_connection(dsn)
        tbl = str(cfg_adapter.get("table_name", "ember_vectors"))
        dims = int(cfg_adapter.get("dimensions", 384))
        return PgVectorAdapter(conn, table_name=tbl, dimensions=dims, embed_fn=embed_fn), conn.close
    if backend == "pinecone":
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError("Pinecone adapter config requires 'pinecone' (`pip install 'ember-custody[pinecone]'`)") from exc
        api_key = str(cfg_adapter.get("api_key", "")).strip()
        index_name = str(cfg_adapter.get("index_name", "")).strip()
        if not api_key or not index_name:
            raise ValueError("[adapter] backend='pinecone' requires api_key and index_name")
        pc = Pinecone(api_key=api_key)
        idx = pc.Index(index_name)
        ns = str(cfg_adapter.get("namespace", ""))
        return PineconeAdapter(idx, namespace=ns, embed_fn=embed_fn), None
    raise ValueError(f"unsupported adapter backend in config: {backend!r}")


def open_vault_from_config(
    config_or_path: EmberConfig | str | Path,
    *,
    embedder: Any = None,
    signer: Ed25519Signer | None = None,
    adapter: MemoryStoreAdapter | None = None,
    anchor: TimestampAnchor | None = None,
) -> CustodyVault:
    """Open a ready-to-use `CustodyVault` wired from `EmberConfig` or an `ember.toml` path."""
    cfg = config_or_path if isinstance(config_or_path, EmberConfig) else load_ember_config(config_or_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    resolved_key_dir = cfg.key_dir if cfg.key_dir is not None else (cfg.data_dir / "keys")

    resolved_signer = signer if signer is not None else Ed25519Signer.load_or_create(resolved_key_dir / "ed25519.json")
    resolved_anchor = anchor if anchor is not None else build_anchor(cfg.anchor, signer=resolved_signer)

    ledger = build_ledger(cfg.ledger, data_dir=cfg.data_dir)
    keystore: KeyStore | None = None
    owns_ks = False
    adapter_closer: Callable[[], None] | None = None
    try:
        keystore, owns_ks = build_keystore(cfg.keystore, key_dir=resolved_key_dir, tenant_id=cfg.tenant_id)
        if adapter is None:
            resolved_adapter, adapter_closer = build_adapter(cfg.adapter, data_dir=cfg.data_dir)
        else:
            resolved_adapter = adapter

        vault = CustodyVault.open(
            cfg.data_dir,
            key_dir=cfg.key_dir,
            tenant_id=cfg.tenant_id,
            adapter=resolved_adapter,
            signer=resolved_signer,
            anchor=resolved_anchor,
            encrypt_vendor_payload=cfg.encrypt_vendor_payload,
            embedder=embedder,
            ledger=ledger,
            keystore=keystore,
            auto_checkpoint_every=cfg.auto_checkpoint_every,
            use_checkpoint=cfg.use_checkpoint,
        )
        vault._owns_keystore = owns_ks
        if adapter_closer is not None:
            orig_close = vault.close

            def _close_all() -> None:
                try:
                    orig_close()
                finally:
                    adapter_closer()

            vault.close = _close_all  # type: ignore[method-assign]
        return vault
    except BaseException:
        ledger.close()
        if owns_ks and keystore is not None:
            close_ks = getattr(keystore, "close", None)
            if callable(close_ks):
                close_ks()
        if adapter_closer is not None:
            adapter_closer()
        raise


__all__ = [
    "EmberConfig",
    "build_adapter",
    "build_anchor",
    "build_keystore",
    "build_ledger",
    "find_ember_config",
    "load_ember_config",
    "open_vault_from_config",
    "parse_ember_config_dict",
]
