# Security Model & Threat Boundaries

Ember aims to make security guarantees **explicit and falsifiable**, never aspirational.

## What Ember Proves

1. **Verifiable Provenance Under Partial Tampering & Redaction (`ember.merkle`, `ember.receipts`, `ember.verify`):**
   - A `Receipt` binds a verbatim quote to its UTF-8 byte range and a salted RFC 6962 Merkle tree (`0x00 || len32(nonce) || nonce || leaf_bytes` for leaves; `0x01 || left || right` for internal nodes). With `span_mode="word" | "clause"`, multi-subject leaves further commit an internal salted span Merkle root so one clause in a single sentence can be redacted while a bystander's clause in the same sentence stays verifiable.
   - Anyone holding the surviving document bundle and `ember.verify` (zero third-party dependencies) can verify surviving quotes and proofs **after** another subject's leaves or spans have been redacted, and render an air-gapped HTML dossier (`--html`, with strict `html.escape` XSS protection).
   - Redacted leaves and spans retain only their salted hash (`nonce` destroyed), preventing dictionary confirmation attacks on low-entropy statements.

2. **Tamper-Evident Append-Only History & Signed Checkpoints (`ember.ledger`):**
   - Events form a gap-free per-tenant SHA-256 chain (`SQLiteLedger` with `WAL`, `PRAGMA synchronous=FULL`, or `PostgresLedger`). Editing, dropping, or reordering any event breaks validation unless an attacker controls and re-seals the entire tail from the edited event onward.
   - `checkpoint()` writes Ed25519-signed snapshots containing **only encrypted ciphertexts** and opaque tokens for `O(delta)` restarts without duplicating plaintext.

3. **Durable Crypto-Shredding, KMS Isolation & Cascade Closure (`ember.crypto`, `ember.keystore`, `ember.kms`, `ember.reducer`, `ember.vault`):**
   - Content fields (`subjects`, `text`), caller `metadata`, and quoted spans are encrypted with `AES-256-GCM` bound via AAD to `(tenant, record_id, field, key_ref)`.
   - `KeyStore` supports `FileKeyStore`, `SQLiteKeyStore` (`PRAGMA secure_delete=ON`), and envelope-wrapped KMS backends (`AwsKmsKeyStore`, `GcpKmsKeyStore`, `VaultTransitKeyStore`).
   - Shared or derived records (`derived_from` chains and `vault.trace_turn()` read->write capture) receive tracked per-record keys linked to every contributing subject; erasing any contributing subject destroys the record key so from-zero replay produces `[erased]` tombstones and `{}` metadata instead of plaintext.
   - Erasure computes exact structural + transitive closure, persists a durable `pending_erasures` journal row (`SQLiteLedger` / `PostgresLedger`) *before* touching external stores, commits keyed `ERASE_SUBJECT` event tokens, redacts source-document leaves/spans and shreds encryption keys **before** external adapter network deletion (preventing transient SaaS/vector DB timeouts from leaving local keys alive, and enabling automated crash recovery via `reconcile_erasures()` / `auto_reconcile=True`), persists NFKC-normalized domain-separated subject and content HMAC tombstone tokens blocking silent re-ingestion, and gates certificate signing on post-erasure adversarial probes (`ErasureIncomplete` fails closed without issuing a certificate).
   - **Coverage & Lineage Completeness (`vault.audit_coverage()`):** Every signed certificate records its exact `trust_boundary["coverage"]` verdict (`"complete" | "bounded_with_warnings" | "incomplete"`), surfacing orphan sources, unattributed writes made outside `trace_turn()`, declared `unverified_locations` (e.g. cold snapshots), and unfinished `pending_erasures` rather than implying invisible universal deletion.

4. **Sidecar & MCP Service Boundaries (`ember.server`, `ember.mcp_server`):**
   - `ember serve` refuses to bind non-loopback addresses without a `bearer_token` (verified with `hmac.compare_digest`), enforces optional `X-Ember-Tenant` header binding, bounds request body sizes (`413`) and transfer encoding (`411`), and maps custody/erasure errors fail-closed (`403`/`409`/`422`).
   - `ember-mcp` exposes strictly scoped custody primitives over JSON-RPC 2.0 stdio without shelling out or returning raw key material.

## Explicit Boundaries (Honest Limitations)

1. **Pre-Erasure Backups & Key Co-Location:**
   - Crypto-shredding protects live logs and post-erasure snapshots **only if** unshredded local keys were not backed up alongside ciphertexts prior to erasure. In production, place `key_dir` outside ledger backups or configure `AwsKmsKeyStore` / `GcpKmsKeyStore` / `VaultTransitKeyStore` + `PostgresLedger`.
2. **Whole-Chain Rewriting & Clock Backdating:**
   - A malicious operator who possesses the entire ledger and Ed25519 signing key can rewrite history coherently unless certificate digests are anchored externally via `Rfc3161Anchor` or `RekorAnchor` (Sigstore Rekor). Unanchored local certificates explicitly report `externally_timestamped: false`.
3. **External Vector Store Consistency, Embeddings & Internal History Logs:**
   - When wrapping third-party stores (`QdrantAdapter`, `ChromaAdapter`, `PgVectorAdapter`, `PineconeAdapter`, `Mem0Adapter`, `VectorStoreAdapter`), Ember tracks all IDs returned at write time and issues concrete delete calls during erasure, then verifies the live query surface with synthesized + explicit probe abstention. Using `encrypt_vendor_payload=True` prevents vendor DB plaintext exposure; omit `embedder` if dense float vector inversion must also be avoided.
   - Specifically for OSS `mem0.Memory`: while `delete(memory_id)` removes the vector record and entity-store links, Mem0's internal `SQLiteManager` (`history_db_path`) writes a `"DELETE"` row holding `old_memory` in plaintext (`m.history(memory_id)`). Configure `encrypt_vendor_payload=True` + an `embedder` (or isolate/prune `history_db_path`) if on-disk Mem0 history is within your threat boundary.

## Reporting Vulnerabilities

Please report cryptographic or isolation defects via GitHub Security Advisories or directly to `sasood01@wsc.edu`.
