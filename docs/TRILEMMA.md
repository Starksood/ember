# Solving the Erasure / Receipt Trilemma in AI Memory and RAG

> Why `delete_all(user_id=...)` breaks GDPR compliance in practice — and how length-prefixed salted Merkle documents, two-tier AES-256-GCM crypto-shredding, and two-direction benchmarks close the gap.

## 1. The Hidden Failure Modes of AI Memory Deletion

Most vector stores (`Qdrant`, `Chroma`, `Pinecone`, `pgvector`) and agent memory layers (`Mem0`, `Letta`, `LangMem`) make storing facts and scoping simple queries by `user_id` straightforward. When a user invokes **GDPR Article 17 ("Right to be Forgotten")** or an enterprise client demands verifiable offboarding, three failure modes appear silently:

### Failure 1: The Erasure / Receipt Trilemma
Imagine an onboarding call transcript or clinical intake note containing facts about two people:

> *"Priya Raman joined Acme in 2019 as a logistics analyst. Marcus Webb owns a bookshop in Leeds."*

Your RAG pipeline indexes that transcript and gives Marcus a cryptographic provenance receipt binding his fact to the document's flat `SHA-256` hash and UTF-8 byte offsets.

Now Priya requests erasure. A flat document hash forces an impossible trilemma:
1. **Redact or delete the transcript** -> Priya's data is gone, **but Marcus's SHA-256 receipt breaks forever** because both the document digest and every byte offset shifted.
2. **Keep the transcript intact so Marcus's receipt verifies** -> **Priya's PII stays on disk permanently**.

### Failure 2: Derived & Cross-Bucket Memory Leaks
Suppose an agent synthesizes a weekly summary:

> *"Riverside handoff: triage routing pairs with pharmacy inventory."*

Because summaries belong to `"team_acme"` rather than `"priya"`, calling `delete_all(user_id="priya")` deletes Priya's raw memories — **and leaves derived summaries, graph links, and append-only audit logs surfacing her context under a team namespace.**

### Failure 3: Zombie Re-Ingestion and Fake Compliance Metrics
Even after deletion runs cleanly:
- **Zombie syncs** (webhook retries, backup imports, stale ETL jobs) re-ingest Priya minutes later under slightly different normalization (`"Priya Raman"` vs `"priya"`).
- **Trivial 0-leak metrics:** Any benchmark that only measures "leak rate" rewards wiping *everything* — destroying innocent bystanders' memories alongside the erased subject gives `0 leaks` on paper while corrupting your store.

---

## 2. How Ember Resolves All Three

Ember (`ember-custody`) wraps an existing store (`Mem0`, LangChain/Chroma/Qdrant vector stores, or SQLite) without rewriting your retrieval model:

### A. Salted Length-Prefixed RFC 6962 Merkle Documents ([`merkle.py`](../src/ember/merkle.py))
Instead of hashing the whole document flatly, Ember splits a source into sentence/line/custom leaves, generates a fresh **128-bit random nonce** per leaf, and hashes:

```text
leaf_hash  = SHA-256(0x00 || uint32_be(len(nonce)) || nonce || leaf_bytes)
node_hash  = SHA-256(0x01 || left_32 || right_32)
```

- **Length-prefixing `nonce`** prevents boundary ambiguity between `(nonce, text)` and `("", nonce + text)`.
- **RFC 6962 `0x00` vs `0x01` domain separation** + strict positional proof-walk checks block second-preimage tree-geometry spoofing.
- **On erasure:** Ember strips Priya's leaf `text` and `nonce`, retaining only Priya's `leaf_hash`. The tree root stays identical, offline dictionary confirmation on common sentences fails (`nonce` is gone), **and Marcus's receipt still verifies bit-for-bit against the redacted document** using zero external dependencies ([`verify.py`](../src/ember/verify.py)).

### B. Zero-Plaintext Append-Only Ledger + Two-Tier Crypto-Shredding ([`keystore.py`](../src/ember/keystore.py), [`vault.py`](../src/ember/vault.py))
- Append-only `SQLiteLedger` (`WAL`, `synchronous=FULL`) hashes an unbroken SHA-256 chain over **encrypted** `subjects`, `text`, `quote`, and caller `metadata` (`AES-256-GCM` bound via AAD to `(tenant_id, item_id, field, key_ref)`).
- **Direct records** encrypt under an opaque subject CEK (`subject:subj_<hmac>`).
- **Shared and `derived_from` records** automatically compute the transitive ancestor set of all contributors and encrypt under a tracked per-record DEK (`record:<item_id>`) indexed under every contributing subject.
- **Erasing Priya** redacts shared source leaves and destroys both Priya's CEK and every derived/shared DEK *before* calling external vendor `delete()` (so even a network timeout never leaves local decryption keys alive, and retries safely sweep pending IDs).
- **From-zero ledger replay** remains 100% tamper-evident while yielding `[erased]` and `{}` for all of Priya's items — leaving innocent bystanders decrypted and retrievable.

### C. Keyed Tombstones & Two-Direction Calibration ([`reducer.py`](../src/ember/reducer.py), [`audit.py`](../src/ember/audit.py))
- Vault-keyed HMAC-SHA256 tombstones block zombie re-ingestion with `TombstonedSubjectError` unless an explicit `reconsent_id` is recorded.
- `erase()` probes the **raw backing store directly** (`adapter.search()`) and refuses to issue an Ed25519 `ErasureCertificate` (`ErasureIncomplete`) if any residue still surfaces.
- The built-in adversarial suite tests **12 subjects × 7 attack categories (204 probes)** while co-gating **bystander retrieval survival** and **redacted Merkle receipt verification**:
  - Naive `user_id`-only baseline: **204/204 probes leak**
  - Ember wrapper: **0/204 probes leak**, **100% bystander memories survive**, **100% bystander receipts verify offline**, **100% from-zero replays stay content-free**.

---

## 3. Honest Threat Boundaries

Cryptography cannot erase data you already leaked out-of-band. Every certificate explicitly records trust boundaries (`trust_boundary` in [`ErasureCertificate`](../src/ember/records.py)):
1. **Separate your keys from ledger backups:** Crypto-shredding only protects pre-existing snapshots if `key_dir` (or your KMS/HSM `KeyStore`) was never backed up alongside `ledger.db`.
2. **External timestamp anchoring:** Without configuring a `TimestampAnchor` (RFC 3161 TSA or public transparency log), a malicious root operator holding both `ledger.db` and `ed25519.json` can rewrite history coherently. Unanchored certificates explicitly declare `externally_timestamped: false`.
3. **Unlinked copy-paste residue:** Structural lineage catches everything written via `subjects` and `derived_from` plus optional `leaf_predicate` sweeps; unlinked copies still trigger fail-closed rejection via `probes`.
