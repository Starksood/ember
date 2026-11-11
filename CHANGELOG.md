# Changelog

All notable changes to `ember-custody` are documented here.

## [Unreleased]

### Fixed
- **`Mem0Adapter` compatibility with live `mem0ai` 2.x (`ember.adapters`, `ember.vault`):**
  - Track written `(user_id, agent_id, run_id)` scopes (plus persistence/restoration across restarts via `CustodyVault._sync_adapter_scopes()`) and fan out unscoped `search(filters=None)` queries so Mem0 2.x's required identity filter check never crashes `_probe_raw_leaks()` or `vault.search()`. Optimized uniform `default_agent_id` scopes to query in a single pass.
  - Map `limit` to Mem0 2.x `top_k` (falling back to `limit` for v1/`MemoryClient`), forward optional `search_threshold`, strip identity keys from outgoing `metadata` payloads to prevent `ignoring metadata['user_id']` warnings, promote top-level `user_id`/`agent_id`/`run_id`/`actor_id` back into `AdapterHit.metadata`, deterministically sanitize whitespace in entity IDs (e.g., `"Kirin Press"`), and treat missing-ID `ValueError` during `delete()` retries idempotently.

### Added & Changed
- **Pluggable, self-documenting benchmark reporting (`ember.audit`, `ember.cli`):**
  - Added `BenchmarkBackendSpec`, `ScenarioBackendPair`, `in_memory_backend_spec()`, and `mem0_backend_spec()` so `SUITE_V2` runs identically with all bystander-survival, redacted Merkle receipt, content-free replay, tombstone, and Ed25519 gates against either `InMemoryStoreAdapter` or a real `Mem0` vector backend.
  - `run_subject_scenario()`, `run_benchmark()`, and `ember benchmark` now explicitly record and print the evaluated `Adapter`, `Retrieval`, `Embedding`, and `Search` settings alongside scores.
  - Added [`examples/benchmark_mem0.py`](examples/benchmark_mem0.py) and verified `SUITE_V2` locally against `mem0ai==2.2.1` + `Qdrant` + `sentence-transformers/all-MiniLM-L6-v2`.
  - Documented Mem0's internal plaintext `history.db` (`history_db_path`) boundary in [`SECURITY.md`](SECURITY.md), [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), [`docs/TRILEMMA.md`](docs/TRILEMMA.md), and [`README.md`](README.md).

## [0.1.0] — Initial Public Release

### Added
- **CustodyVault & CLI (`ember.vault`, `ember.cli`):** Unified Python wrapper and CLI (`init`, `add-source`, `remember`, `search`, `erase`, `replay`, `export`, `verify`, `benchmark`) across `Mem0Adapter`, `VectorStoreAdapter`, `SQLiteStoreAdapter`, and `InMemoryStoreAdapter`.
- **Redactable Salted Merkle Provenance (`ember.merkle`, `ember.receipts`):** RFC 6962 domain-separated (`0x00` vs `0x01`) trees with length-prefixed per-leaf 128-bit nonces so erasing Subject A destroys A's text and nonce while keeping Bystander B's inclusion proofs intact and dictionary-resistant.
- **Zero-Plaintext Ledger & Crypto-Shredding (`ember.ledger`, `ember.crypto`, `ember.keystore`, `ember.records`):** Append-only SQLite WAL (`synchronous=FULL`) SHA-256 hash chain encrypting `subjects`, `text`, `quote`, and caller `metadata` via AES-256-GCM with AAD field/record/tenant binding, two-tier subject CEKs + shared/derived record DEKs, and keyed HMAC tombstones blocking zombie re-ingestion.
- **Fail-Closed Erasure Certificates (`ember.signing`, `ember.reducer`):** Local leaf redaction and key shredding execute before external adapter delete calls (preserving retry recovery on transient failures), followed by raw probe checks (`ErasureIncomplete`), Ed25519 signing, and pluggable `TimestampAnchor`.
- **Zero-Dependency Offline Verifier (`ember/verify.py`, `ember-verify`):** Pure Python standard-library verifier checking Merkle redactions, bystander receipts, and ledger hash chains without third-party dependencies.
- **Adversarial Two-Direction Benchmark (`ember.audit`):** 12-subject × 7-attack-class suite (`204` probes) enforcing zero leaks alongside bystander survival, valid bystander receipts, content-free historical replays, tombstone enforcement, and Ed25519 signatures.
