# Changelog

All notable changes to `ember-custody` are documented here.

## [Unreleased]

### Added
- **Optimistic Concurrency Control & Multi-Handle State Synchronization (`ember.vault`, `ember.ledger`, `ember.keystore`, `ember.reducer`):**
  - Added `ConcurrentModificationError` (subclass of `InvariantViolation`, mapped to HTTP `409 Conflict` in `EmberHttpApp`) and `expected_seq` optimistic sequence verification on `SQLiteLedger.append()` / `PostgresLedger.append()` plus automatic rollback (`adapter.delete(added_ids)`) of in-flight adapter writes when an `erase()` commits concurrently.
  - Added `CustodyVault._sync_state_from_ledger()` delta catch-up across vault handles sharing a persistent ledger, `get_event_by_id()` pre-upsert `event_id` idempotency/erasure validation before calling vendor `adapter.upsert()`, redacted-leaf validation on `ADD_ITEM` ledger appends, `FileKeyStore` on-disk change sync (`_sync_from_disk`) so cross-handle key shreds never stay cached or get resurrected, and SQLite `timeout=10.0` / `PRAGMA busy_timeout = 10000`.
  - Added concurrency, stale-handle, `threading.Barrier` mid-flight TOCTOU race, and delayed-write retry tests across `Mem0Adapter`, `VectorStoreAdapter`, and `SQLiteStoreAdapter` (`tests/test_adapters.py`, `tests/test_reducer_ledger.py`).

## [0.2.0] — Phase 2: Fail-Proof Core, Production Integrations & Sidecar Ecosystem

### Added
- **Turn-Scoped Read -> Write Auto-Lineage (`ember.vault.TurnTrace`):**
  - `vault.trace_turn()` (`contextvars`-backed, thread- and async-safe) automatically records `search()` hits and injects `derived_from` on subsequent `remember()` calls within the turn (`prune="all" | "lexical_overlap" | callable`).
- **Encrypted Vendor Storage Modes (`encrypt_vendor_payload=True`):**
  - Writes AES-256-GCM ciphertext (`enc2:...`), opaque HMAC `subject_ref` filters, and either external float embeddings (`embedder`) or keyed HMAC blind-index search tokens (`KeyStore.blind_token()`) so third-party vector DBs never store plaintext PII or raw `user_id` labels.
- **Mem0 `infer=True` Lifecycle Sync & Content-Level Zombie Blocking (`ember.adapters`, `ember.vault`, `ember.keystore`):**
  - `Mem0Adapter` handles Mem0's `ADD`/`UPDATE`/`DELETE`/`NOOP` consolidation events during `remember()` and exposes `get_all` + `ember_derived_from` support.
  - `erase()` records domain-separated keyed HMAC `tombstoned_content_tokens` alongside subject tokens so un-tagged zombie re-ingestion naming an erased subject fails closed (`TombstonedSubjectError`) unless `reconsent_id` is supplied; synthesizes `auto_probes=True` from doomed item content when explicit `probes` are omitted.
- **Sub-Leaf Redactable Merkle Trees (`ember.merkle`, `ember.receipts`, `ember.verify`):**
  - Compound sentences sharing multiple subjects (`span_mode="word" | "clause"` or custom `SpanChunkerFn`) compute leaf digests as internal roots over independently salted `SpanEntry` leaves, letting Subject A's clause be redacted + nonce-destroyed while Subject B's surviving clause in the exact same sentence remains verifiable offline via `verify.py`.
- **1-Line Framework Wrappers & Native Vector DB Adapters (`ember.frameworks`, `ember.adapters`):**
  - Added `EmberMem0`, `EmberLangChainRetriever`, `EmberLangChainVectorStore`, and `EmberLlamaIndexRetriever`.
  - Added native O(1)-ID adapters `QdrantAdapter`, `ChromaAdapter`, `PgVectorAdapter`, and `PineconeAdapter` alongside `InMemoryStoreAdapter` and `SQLiteStoreAdapter`.
- **Postgres Ledger, Ed25519 Signed Checkpoints, SQLite/Cloud KMS & Async Support (`ember.ledger`, `ember.keystore`, `ember.kms`, `ember.vault`):**
  - Extracted `LedgerBackend` protocol and implemented `PostgresLedger` (psycopg3) plus Ed25519-signed ciphertext-only `checkpoints` (`vault.checkpoint()`, `replay(use_checkpoint=True)`, `auto_checkpoint_every`) for `O(delta)` restarts without storing plaintext in snapshots.
  - Added `SQLiteKeyStore` (`PRAGMA secure_delete=ON`, `0o600` perms), envelope-wrapped KMS backends (`AwsKmsKeyStore`, `GcpKmsKeyStore`, `VaultTransitKeyStore`), and `AsyncCustodyVault`.
- **External Timestamp Anchoring & Declarative Config (`ember.signing`, `ember.config`, `ember.cli`):**
  - Added `Rfc3161Anchor` (DER TSQ/TSR over stdlib `urllib` + `cryptography`) and `RekorAnchor` (Sigstore hashedrekord v0.0.1 over Ed25519).
  - Added zero-dependency `ember.toml` loader (`tomllib` with `${ENV:-default}` expansion) and CLI `--config`, `--keystore-backend`, `--encrypt-vendor-payload`, `--tsa-url`, `--rekor-url`, `add-source --span-mode`, `checkpoint`, and `replay --use-checkpoint`.
- **HTTP Sidecar, MCP Server & TypeScript SDK (`ember.server`, `ember.mcp_server`, `sdk/ts/`):**
  - `ember serve` (`EmberHttpApp`, `ThreadingHTTPServer`, optional Starlette ASGI) with constant-time Bearer token auth, fail-closed non-loopback bind protection, tenant header enforcement, and body size caps.
  - `ember mcp` / `ember-mcp` zero-dependency stdio JSON-RPC 2.0 MCP server (`ember_register_source`, `ember_remember`, `ember_search`, `ember_erase`, `ember_verify_receipt`, `ember_export_bundle`).
  - Zero-dependency `@ember-custody/client` TypeScript SDK (`sdk/ts/`) with strict types verified live against the Python sidecar.
- **Compliance HTML Dossier & `SUITE_V3` Adversarial Benchmark (`ember.report`, `ember.audit`):**
  - `ember verify bundle.json --html report.html` generates an air-gapped, zero-CDN, HTML5 dossier with strict HTML/attribute escaping and `[redacted]` visual chips.
  - Added `SUITE_V3` (`run_v3_benchmark`, `ember benchmark --suite v3`) verifying all 12 subjects × 204 probes plus sub-leaf compound redaction, `TurnTrace` auto-lineage, zero-plaintext encrypted vendor storage, and content-level zombie blocking.

### Fixed
- **`Mem0Adapter` compatibility with live `mem0ai` 2.x (`ember.adapters`, `ember.vault`):**
  - Track written `(user_id, agent_id, run_id)` scopes (plus persistence/restoration across restarts via `CustodyVault._sync_adapter_scopes()`) and fan out unscoped `search(filters=None)` queries so Mem0 2.x's required identity filter check never crashes `_probe_raw_leaks()` or `vault.search()`. Optimized uniform `default_agent_id` scopes to query in a single pass.
  - Map `limit` to Mem0 2.x `top_k` (falling back to `limit` for v1/`MemoryClient`), forward optional `search_threshold`, strip identity keys from outgoing `metadata` payloads to prevent `ignoring metadata['user_id']` warnings, promote top-level `user_id`/`agent_id`/`run_id`/`actor_id` back into `AdapterHit.metadata`, deterministically sanitize whitespace in entity IDs (e.g., `"Kirin Press"`), and treat missing-ID `ValueError` during `delete()` retries idempotently.
  - Guard against accidental bare-string `subjects="alice"` arguments in `CustodyVault.remember()`, HTTP sidecar, and MCP server.

### Changed
- `run_subject_scenario()`, `run_benchmark()`, `run_v3_benchmark()`, and `ember benchmark` report the evaluated `Adapter`, `Retrieval`, `Embedding`, and `Search` settings explicitly next to every score (`examples/benchmark_mem0.py` runs `SUITE_V2` against live `mem0ai` + `Qdrant` + `sentence-transformers`).

## [0.1.0] — Initial Public Release

### Added
- **CustodyVault & CLI (`ember.vault`, `ember.cli`):** Unified Python wrapper and CLI (`init`, `add-source`, `remember`, `search`, `erase`, `replay`, `export`, `verify`, `benchmark`) across `Mem0Adapter`, `VectorStoreAdapter`, `SQLiteStoreAdapter`, and `InMemoryStoreAdapter`.
- **Redactable Salted Merkle Provenance (`ember.merkle`, `ember.receipts`):** RFC 6962 domain-separated (`0x00` vs `0x01`) trees with length-prefixed per-leaf 128-bit nonces so erasing Subject A destroys A's text and nonce while keeping Bystander B's inclusion proofs intact and dictionary-resistant.
- **Zero-Plaintext Ledger & Crypto-Shredding (`ember.ledger`, `ember.crypto`, `ember.keystore`, `ember.records`):** Append-only SQLite WAL (`synchronous=FULL`) SHA-256 hash chain encrypting `subjects`, `text`, `quote`, and caller `metadata` via AES-256-GCM with AAD field/record/tenant binding, two-tier subject CEKs + shared/derived record DEKs, and keyed HMAC tombstones blocking zombie re-ingestion.
- **Fail-Closed Erasure Certificates (`ember.signing`, `ember.reducer`):** Local leaf redaction and key shredding execute before external adapter delete calls (preserving retry recovery on transient failures), followed by raw probe checks (`ErasureIncomplete`), Ed25519 signing, and pluggable `TimestampAnchor`.
- **Zero-Dependency Offline Verifier (`ember/verify.py`, `ember-verify`):** Pure Python standard-library verifier checking Merkle redactions, bystander receipts, and ledger hash chains without third-party dependencies.
- **Adversarial Two-Direction Benchmark (`ember.audit`):** 12-subject × 7-attack-class suite (`204` probes) enforcing zero leaks alongside bystander survival, valid bystander receipts, content-free historical replays, tombstone enforcement, and Ed25519 signatures.
