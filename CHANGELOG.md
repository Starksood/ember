# Changelog

All notable changes to `ember-custody` are documented here.

## [0.1.0] — Initial Public Release

### Added
- **CustodyVault & CLI (`ember.vault`, `ember.cli`):** Unified Python wrapper and CLI (`init`, `add-source`, `remember`, `search`, `erase`, `replay`, `export`, `verify`, `benchmark`) across `Mem0Adapter`, `VectorStoreAdapter`, `SQLiteStoreAdapter`, and `InMemoryStoreAdapter`.
- **Redactable Salted Merkle Provenance (`ember.merkle`, `ember.receipts`):** RFC 6962 domain-separated (`0x00` vs `0x01`) trees with length-prefixed per-leaf 128-bit nonces so erasing Subject A destroys A's text and nonce while keeping Bystander B's inclusion proofs intact and dictionary-resistant.
- **Zero-Plaintext Ledger & Crypto-Shredding (`ember.ledger`, `ember.crypto`, `ember.keystore`, `ember.records`):** Append-only SQLite WAL (`synchronous=FULL`) SHA-256 hash chain encrypting `subjects`, `text`, `quote`, and caller `metadata` via AES-256-GCM with AAD field/record/tenant binding, two-tier subject CEKs + shared/derived record DEKs, and keyed HMAC tombstones blocking zombie re-ingestion.
- **Fail-Closed Erasure Certificates (`ember.signing`, `ember.reducer`):** Local leaf redaction and key shredding execute before external adapter delete calls (preserving retry recovery on transient failures), followed by raw probe checks (`ErasureIncomplete`), Ed25519 signing, and pluggable `TimestampAnchor`.
- **Zero-Dependency Offline Verifier (`ember/verify.py`, `ember-verify`):** Pure Python standard-library verifier checking Merkle redactions, bystander receipts, and ledger hash chains without third-party dependencies.
- **Adversarial Two-Direction Benchmark (`ember.audit`):** 12-subject × 7-attack-class suite (`204` probes) enforcing zero leaks alongside bystander survival, valid bystander receipts, content-free historical replays, tombstone enforcement, and Ed25519 signatures.
