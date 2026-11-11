# Ember Architecture

Ember separates **immutable cryptographic accountability** from **mutable erasure boundaries** and **existing store ergonomics**.

```text
  Application / RAG / Agents (Python Sync & Async, LangChain, LlamaIndex, Mem0)
  HTTP Sidecar (`ember serve` / TS SDK `@ember-custody/client`) / MCP (`ember-mcp`)
                   │
                   ▼
           ┌───────────────┐   TurnTrace (contextvars) + Memory Adapters
           │ CustodyVault  │   (Qdrant, Chroma, pgvector, Pinecone, Mem0, SQLite)
           └───────┬───────┘
                   │
     ┌─────────────┼──────────────────────┐
     ▼             ▼                      ▼
┌──────────┐ ┌───────────┐         ┌──────────────┐
│ KeyStore │ │ Document  │         │ LedgerBackend│
│ File /   │ │ Store     │         │ SQLite /     │
│ SQLite / │ │ (Salted   │         │ Postgres     │
│ KMS(AWS/ │ │  Leaves & │         │ SHA-256 Chain│
│ GCP/Vlt) │ │  Spans)   │         │ + Checkpoints│
└────┬─────┘ └─────┬─────┘         └──────┬───────┘
     │             │                      │
     │             │                      ▼
     │             │               ┌──────────────┐
     └──────┬──────┴──────────────▶│ Pure Reducer │
            │                      │ (Zero-IO     │
            ▼                      │  State Fold) │
   ┌────────────────┐              └──────────────┘
   │ Ed25519 Signer │
   │ + RFC3161 /    │
   │   Rekor Anchor │
   └────────────────┘
```

## Key Separation Principles

1. **Append-Only Ledger & Signed Checkpoints Hold Zero Plaintext:**
   - `ADD_SOURCE` stores `merkle_root`, `flat_doc_hash`, `byte_length`, and `leaf_count`, alongside caller-declared `origin` metadata. Raw document text is **never** written to `ledger_events`.
   - `ADD_ITEM` encrypts `subjects`, `text`, quoted `evidence` spans, and caller `metadata` via AES-256-GCM bound to `(tenant_id, item_id, field, key_ref)` AAD.
   - `ERASE_SUBJECT` records only keyed subject/content HMAC tombstone tokens and structural `ClosureManifest` IDs.
   - Ed25519-signed `checkpoints` persist only ciphertext payloads and token sets so vault restarts run in `O(delta)` time without re-introducing plaintext into snapshots.

2. **Two-Tier Encryption Keys (`subject` CEK + `record` DEK) + Envelope KMS:**
   - Single-subject memories encrypt under an opaque subject CEK (`subject:<hmac_ref>`).
   - Shared or derived memories (`derived_from` DAG, including automatic read->write links captured inside `vault.trace_turn()`) inherit the transitive union of contributing subjects and encrypt under a tracked per-record DEK (`record:<item_id>`).
   - Erasing *any* contributing subject destroys both that subject's CEK and every shared/derived DEK referencing them (`FileKeyStore`, `SQLiteKeyStore` with `secure_delete=ON`, or wrapped DEKs under `AwsKmsKeyStore` / `GcpKmsKeyStore` / `VaultTransitKeyStore`).

3. **Two-Mode Vendor Privacy (`encrypt_vendor_payload=True`):**
   - Mode 2A (Encrypted Text + External Embedder): stores `AES-256-GCM` ciphertext and opaque HMAC `subject_ref` filters alongside caller-supplied dense embeddings (`Qdrant`, `Chroma`, `pgvector`, `Pinecone`, `Mem0`).
   - Mode 2B (Encrypted Text + Keyed Blind-Index): omits float vectors and indexes salted HMAC-SHA256 keyword tokens (`bt...`), protecting stores without an embedder from exposing raw terms or embeddings.

4. **Salted RFC 6962 Redactable Documents (Leaf + Sub-Leaf Span Trees):**
   - `RedactableDoc` splits sources into sentence, line, or custom leaves, and optionally splits compound multi-subject sentences into sub-leaf spans (`span_mode="word" | "clause"` or custom `SpanChunkerFn`), each carrying a 128-bit random nonce: `H(0x00 || uint32_be(len(nonce)) || nonce || part_bytes)`.
   - Redacting Subject A deletes `text` and `nonce` from A's leaves or spans while preserving the parent digest, Merkle root, and Bystander B's inclusion proofs intact without exposing A's text to dictionary guessing.

5. **Zero-Dependency Offline Verifier (`ember/verify.py`) & HTML Compliance Dossier (`ember/report.py`):**
   - Document integrity, span/leaf bystander receipts, and ledger chain validation require zero third-party packages; `ember verify --html` renders an air-gapped HTML5 dossier with strict XSS escaping.
