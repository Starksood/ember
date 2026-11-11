# Security Model & Threat Boundaries

Ember aims to make security guarantees **explicit and falsifiable**, never aspirational.

## What Ember Proves

1. **Verifiable Provenance Under Partial Tampering & Redaction (`ember.merkle`, `ember.receipts`, `ember.verify`):**
   - A `Receipt` binds a verbatim quote to its UTF-8 byte range and a salted RFC 6962 Merkle tree (`0x00 || len32(nonce) || nonce || leaf_bytes` for leaves; `0x01 || left || right` for internal nodes).
   - Anyone holding the surviving document bundle and `ember.verify` (zero third-party dependencies) can verify surviving quotes and proofs **after** another subject's leaves have been redacted.
   - Redacted leaves retain only their salted hash (`nonce` destroyed), preventing dictionary confirmation attacks on low-entropy statements.

2. **Tamper-Evident Append-Only History (`ember.ledger`):**
   - Events form a gap-free per-tenant SHA-256 chain (`WAL`, `PRAGMA synchronous=FULL`). Editing, dropping, or reordering any event breaks validation unless an attacker controls and re-seals the entire tail from the edited event onward.

3. **Durable Crypto-Shredding & Cascade Closure (`ember.crypto`, `ember.keystore`, `ember.reducer`, `ember.vault`):**
   - Content fields (`subjects`, `text`), caller `metadata`, and quoted spans are encrypted with `AES-256-GCM` bound via AAD to `(tenant, record_id, field, key_ref)`.
   - Shared or derived records (`derived_from` chains) receive tracked per-record keys linked to every contributing subject; erasing any contributing subject destroys the record key so from-zero replay produces `[erased]` tombstones and `{}` metadata instead of plaintext.
   - Erasure computes exact structural + transitive closure, commits the keyed `ERASE_SUBJECT` event, redacts source-document leaves and shreds encryption keys **before** external adapter network deletion (preventing transient SaaS/vector DB timeouts from leaving local keys alive, and allowing safe retries), persists a keyed HMAC tombstone token blocking silent re-ingestion, and gates certificate signing on post-erasure adversarial probes (`ErasureIncomplete` fails closed without issuing a certificate).

## Explicit Boundaries (Honest Limitations)

1. **Pre-Erasure Backups & Key Co-Location:**
   - Crypto-shredding protects the live ledger and replay state **only if** the `KeyStore` was not backed up alongside ciphertexts prior to erasure. In production, store `key_dir` on a separate volume or implement `KeyStore` over KMS/HSM (`AWS KMS`, `GCP KMS`, `Vault Transit`).
2. **Whole-Chain Rewriting & Clock Backdating:**
   - A malicious operator who possesses the entire SQLite file and Ed25519 signing key can rewrite history coherently unless certificate digests are anchored externally via `TimestampAnchor` (RFC 3161 TSA or a public transparency log). Unanchored local certificates explicitly report `externally_timestamped: false`.
3. **External Vector Store Consistency & Internal History Logs:**
   - When wrapping third-party stores (`Mem0Adapter`, `VectorStoreAdapter`), Ember tracks all IDs returned at write time and issues concrete delete calls during erasure, then verifies the live query surface with probe abstention. Ember cannot cryptographically attest internal garbage-collection timing inside third-party SaaS caches (such as vendor ANN index rebuild lag); use `probes` to fail closed if vendor search still surfaces residue.
   - Specifically for `mem0.Memory`: while `delete(memory_id)` removes the vector record and entity-store links, Mem0's own internal `SQLiteManager` (`history_db_path`) writes a `"DELETE"` row holding `old_memory` in plaintext (`m.history(memory_id)`). Configure an isolated/disposable `history_db_path` or prune/encrypt/rotate Mem0's internal SQLite history file if on-disk Mem0 history is within your threat boundary.

## Reporting Vulnerabilities

Please report cryptographic or isolation defects via GitHub Security Advisories or directly to `sasood01@wsc.edu`.
