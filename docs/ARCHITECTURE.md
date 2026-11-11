# Ember Architecture

Ember separates **immutable cryptographic accountability** from **mutable erasure boundaries** and **existing store ergonomics**.

```text
  Application / RAG / Mem0 / Vector Store
                   │
                   ▼
           ┌───────────────┐
           │ CustodyVault  │  (Step 3 wrapper + CLI)
           └───────┬───────┘
                   │
     ┌─────────────┼──────────────────┐
     ▼             ▼                  ▼
┌──────────┐ ┌───────────┐     ┌──────────────┐
│ KeyStore │ │ Document  │     │ SQLiteLedger │
│ (Mutable)│ │ Store     │     │ (Append-only │
│ CEK+DEKs │ │ (Mutable  │     │  SHA-256     │
│ + HMAC   │ │  Salted   │     │  Hash Chain) │
│ Secrets  │ │  Leaves)  │     └──────┬───────┘
└────┬─────┘ └─────┬─────┘            │
     │             │                  ▼
     │             │           ┌──────────────┐
     └──────┬──────┴──────────▶│ Pure Reducer │
            │                  │ (Zero-IO     │
            ▼                  │  State Fold) │
   ┌────────────────┐          └──────────────┘
   │ Ed25519 Signer │
   │ + Timestamp    │
   │   Anchor       │
   └────────────────┘
```

## Key Separation Principles

1. **Append-Only Ledger Holds Zero Plaintext:**
   - `ADD_SOURCE` stores `merkle_root`, `flat_doc_hash`, `byte_length`, and `leaf_count`, alongside caller-declared `origin` metadata. Raw document text is **never** written to `ledger_events`.
   - `ADD_ITEM` encrypts `subjects`, `text`, quoted `evidence` spans, and caller `metadata` via AES-256-GCM bound to `(tenant_id, item_id, field, key_ref)` AAD.
   - `ERASE_SUBJECT` records only keyed HMAC tombstone tokens and structural `ClosureManifest` IDs.

2. **Two-Tier Encryption Keys (`subject` CEK + `record` DEK):**
   - Single-subject memories encrypt under an opaque subject CEK (`subject:<hmac_ref>`).
   - Shared or derived memories (`derived_from` DAG) automatically inherit a union of all upstream contributing subjects and encrypt under a tracked per-record DEK (`record:<item_id>`).
   - Erasing *any* contributing subject deletes both that subject's CEK and every shared/derived DEK referencing them, making historical ciphertext unrecoverable on from-zero replay (`[erased]`), while innocent bystanders' unshared records stay readable.

3. **Salted RFC 6962 Redactable Documents:**
   - `RedactableDoc` splits sources into sentence, line, or custom byte spans and assigns each leaf a fresh 128-bit random nonce: `H(0x00 || uint32_be(len(nonce)) || nonce || part_bytes)`.
   - Redacting Subject A deletes `text` and `nonce` from A's leaves while keeping the leaf digest, preserving the Merkle root and Bystander B's inclusion proofs intact without exposing A's text to dictionary guessing.

4. **Stdlib Offline Verifier (`ember/verify.py`):**
   - Document integrity, bystander receipts, and ledger chain validation require zero third-party dependencies so auditors do not have to trust the runtime environment.
