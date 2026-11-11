# Ember (`ember-custody`)

> **Proof that survives erasure.** Cryptographic custody, redactable Merkle provenance receipts, and adversarially tested cascade deletion for AI memory and RAG stores.

Existing memory engines (`Mem0`, `Letta`, `LangChain`, vector databases) know how to store embeddings and delete by top-level `user_id`. Under GDPR Article 17 ("Right to be Forgotten") and enterprise audit requirements, three failures still happen silently:

1. **The Erasure / Receipt Trilemma:** If two subjects appear in the same source transcript, a flat SHA-256 document hash forces an impossible choice: delete Subject A cleanly and break Bystander B's provenance receipt, or preserve B's receipt by leaving A's source file unredacted.
2. **Derived & Cross-Bucket Leaks:** Calling `delete_all(user_id="priya")` deletes raw memories keyed to `"priya"`, leaving downstream derived team summaries, shared workspace notes, and append-only audit logs surfacing `"Priya"` forever.
3. **Zombie Re-Ingestion & Fake Metrics:** Sync retries re-ingest erased subjects seconds later; naive deletion metrics pass `0 leaks` trivially if a system wipes bystander data along with the subject.

**Ember** wraps your existing memory or vector store (~10 lines of Python) to solve all three with explicit, falsifiable cryptographic mechanisms.

---

## Core Guarantees

- **Redactable Salted Merkle Provenance (`ember.merkle`, `ember.receipts`):** Documents split into sentence/line/custom leaves with length-prefixed per-leaf 128-bit random nonces (`0x00 || len32(nonce) || nonce || leaf_bytes`, RFC 6962 domain separation). Erasing Subject A destroys A's plaintext and nonce while keeping A's leaf digest so **Bystander B's inclusion proof still verifies** against the unchanged Merkle root — while offline dictionary guessing of A's redacted leaf fails.
- **Zero-Plaintext Hash-Chained Ledger + Crypto-Shredding (`ember.ledger`, `ember.crypto`, `ember.keystore`):** Append-only SQLite WAL chain (`synchronous=FULL`) encrypts `subjects`, `text`, `quote`, and caller `metadata` with `AES-256-GCM` bound via AAD to `(tenant, item_id, field, key_ref)`. Shared and `derived_from` records use tracked per-record keys linked to every upstream contributor — erasing any contributor destroys both their subject CEK and every shared/derived DEK before issuing external adapter calls, so from-zero replay produces `[erased]` even across transient store timeouts.
- **Keyed Tombstones & Fail-Closed Certificates (`ember.reducer`, `ember.signing`):** Vault-keyed HMAC-SHA256 tombstones block re-ingestion (`TombstonedSubjectError`) unless an explicit `reconsent_id` is supplied. `erase()` probes the **raw wrapped store directly** and raises `ErasureIncomplete` without issuing an Ed25519 certificate if anything still surfaces.
- **Zero-Dependency Offline Verifier (`ember/verify.py`):** Copy this single Python stdlib file anywhere to verify Merkle redactions, bystander receipts, and ledger hash chains without third-party packages (`cryptography` is used only when checking Ed25519 certificate signatures).

---

## Installation

```bash
pip install ember-custody
```

Optional Mem0 extra:

```bash
pip install "ember-custody[mem0]"
```

From a local clone (using [`uv`](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Starksood/ember.git
cd ember
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

See [Solving the Erasure / Receipt Trilemma](docs/TRILEMMA.md) for the cryptographic design and two-direction benchmark deep dive.

---

## 60-Second Quickstart

### 1. Wrap Mem0 (or any Vector Store) with Cascade Custody

```python
from mem0 import Memory
from ember import CustodyVault, Mem0Adapter, TombstonedSubjectError

vault = CustodyVault.open(
    "./ember_ledger",
    key_dir="./ember_keys",  # keep keys separate from ledger backups!
    adapter=Mem0Adapter(Memory(), default_user_id="team_acme"),
)

# Register a multi-party intake document (Merkle-indexed, zero raw text in ledger)
vault.register_source(
    "intake_notes",
    "Priya Raman works in Portland on triage routing. Dana Kim leads pharmacy inventory.",
    origin="onboarding_transcript",
    origin_kind="conversation",
)

r_priya = vault.remember(
    "Priya Raman works in Portland on triage routing.",
    subjects=["priya"],
    source_id="intake_notes",
    quote="Priya Raman works in Portland on triage routing.",
    metadata={"user_id": "priya"},
)
r_dana = vault.remember(
    "Dana Kim leads pharmacy inventory.",
    subjects=["dana"],
    source_id="intake_notes",
    quote="Dana Kim leads pharmacy inventory.",
    metadata={"user_id": "dana"},
)

# Derived team summary sits in a different user_id bucket — naive delete_all(user_id="priya") misses it
vault.remember(
    "Riverside handoff: triage routing pairs with pharmacy inventory.",
    subjects=["team_acme"],
    derived_from=[r_priya.item.item_id, r_dana.item.item_id],
    metadata={"user_id": "team_acme"},
)

# Right to be Forgotten: cascades to Mem0 records, derived summaries, shared leaves, and keys
cert = vault.erase(
    "priya",
    probes=["Where does Priya work?", "Who handles triage routing?"],
)

# Dana (innocent bystander) and her redacted source proof survive intact
assert vault.verify_item_receipt(r_dana.item.item_id) is True
assert vault.search("Dana pharmacy").abstained is False

# Re-ingesting Priya without explicit re-consent fails closed
try:
    vault.remember("Priya Raman works in Portland.", subjects=["priya"])
except TombstonedSubjectError:
    print("Blocked zombie re-ingestion")
```

Run the self-contained examples without external API keys:

```bash
python examples/wrap_mem0.py
python examples/redactable_transcript.py
python examples/offline_verify_bundle.py
```

### 2. CLI Workflow & Offline Audit

```bash
ember --vault .ember --key-dir .keys init
ember --vault .ember --key-dir .keys add-source doc1 --text "Priya lives in Portland. Marcus lives in Leeds."
ember --vault .ember --key-dir .keys remember "Priya lives in Portland." --subject priya --source-id doc1 --meta email=priya@example.org
ember --vault .ember --key-dir .keys remember "Marcus lives in Leeds." --subject marcus --source-id doc1
ember --vault .ember --key-dir .keys erase priya --probe "Where does Priya live?"
ember --vault .ember --key-dir .keys replay
ember --vault .ember --key-dir .keys export --out bundle.json
ember verify bundle.json --require-signature
```

Or copy `src/ember/verify.py` anywhere and run with Python standard library alone:

```bash
python3 verify.py bundle.json
```

---

## Adversarial Benchmark & Two-Direction Calibration

Run the deterministic keyword-overlap fixture (`InMemoryStoreAdapter`) across 12 subjects × 7 attack classes (`direct`, `paraphrase`, `inference`, `multi_hop`, `attribute`, `temporal`, `aggregate` — 204 attack probes total):

```bash
ember benchmark
```

Measured output comparing a naive `delete_all(user_id=subject)` baseline (where derived team summaries live under a workspace bucket) against Ember:

```text
EMBER ERASURE ADVERSARIAL SUITE — 12 subjects, 204 probes (7 attack classes)
  Adapter   : InMemoryStoreAdapter (retrieval=keyword_overlap, store=in-process)
  Embedding : none
  Search    : bystander_limit=5, erase_probe_limit=25, naive_delete=user_id_metadata_exact, probe_limit=10
==============================================================================
  Priya      baseline_leaks=17  ember_leaks= 0  derived=1  bystander_ok=True  receipt_ok=True  replay_clean=True
  ... (12 subjects total)
==============================================================================
  Adapter              : InMemoryStoreAdapter (keyword_overlap, embed=none)
  Baseline naive leaks : 204/204
  Ember leaks          : 0/204
  Bystanders survived  : True
  Receipts intact      : True
  Replays content-free : True
  Ed25519 certs valid  : True
```

### Real Vector-Store Comparison (`Mem0` + Qdrant + Dense Embeddings)

`run_benchmark()` accepts any `BenchmarkBackendSpec` (`mem0_backend_spec`) so the exact same 204-probe corpus, bystander survival checks, redacted Merkle receipt checks, content-free replay verification, and Ed25519 certificate verification can run against a live vector store:

```bash
pip install "ember-custody[mem0]" qdrant-client sentence-transformers
python examples/benchmark_mem0.py --threshold 0.0
```

Verified locally against `mem0ai==2.2.1` + local `Qdrant` + `sentence-transformers/all-MiniLM-L6-v2` (384-dim cosine similarity, `top_k=10`, `bystander_limit=5`, `infer=False`, tested at both `threshold=0.1` and `threshold=0.0`):

```text
EMBER ERASURE ADVERSARIAL SUITE — 12 subjects, 204 probes (7 attack classes)
  Adapter   : Mem0Adapter (retrieval=vector_semantic_cosine, store=qdrant (local))
  Embedding : sentence-transformers/all-MiniLM-L6-v2
  Search    : bystander_limit=5, embedding_dims=384, erase_probe_limit=25, infer=False, naive_delete=mem0.delete_all(user_id=subject), probe_limit=10, threshold=0.0
==============================================================================
  Adapter              : Mem0Adapter (vector_semantic_cosine, embed=sentence-transformers/all-MiniLM-L6-v2)
  Baseline naive leaks : 204/204
  Ember leaks          : 0/204
  Bystanders survived  : True
  Receipts intact      : True
  Replays content-free : True
  Ed25519 certs valid  : True
```

Why bystander checks are co-primary: leak rate alone is monotone in deletion — deleting *everything* scores `0/204` leaks. Both `tests/test_erasure_suite.py` and `tests/test_calibration.py` verify both failure directions: under-deletion triggers `ErasureIncomplete`, and over-deletion fails bystander retrieval survival (checked against both `InMemoryStoreAdapter` and live `Mem0`).

---

## Honest Threat Model Limits

Ember documents what cryptography cannot invent locally (see [`SECURITY.md`](SECURITY.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)):

1. **Key Separation & Backups:** Crypto-shredding protects live logs and snapshots taken *after* erasure only if pre-erasure backups did not copy unshredded keys alongside `ledger.db`. Configure a separate `key_dir` or implement `ember.keystore.KeyStore` over KMS/HSM (`AWS KMS`, `GCP KMS`, `Vault Transit`).
2. **External Time Anchoring:** Self-signed local `Ed25519` proofs establish internal ordering and integrity; preventing an operator holding both `ledger.db` and `ed25519.json` from backdating the whole chain requires plugging an RFC 3161 TSA or public transparency log via `ember.signing.TimestampAnchor`.
3. **Wrapped Store Internal History Tables (e.g., Mem0 `history.db`):** `Mem0Adapter` deletes live Mem0 vector points (and Mem0 strips entity links), while Ember's own ledger stays AES-256-GCM encrypted and crypto-shredded. However, `mem0.Memory` also maintains an internal plaintext SQLite history DB (`history_db_path`, logging `old_memory` even on `"DELETE"` events). Treat `history_db_path` as non-shredded vendor state: isolate or purge/rotate it alongside erasure if deployed on disk.

---

## Origin & License

Ember distills the verifiable provenance, salted Merkle redaction, crypto-shredding, and two-direction audit primitives developed in [Fireweed](https://github.com/Starksood) into a standalone Apache-2.0 library focused strictly on cryptographic custody.

Licensed under the [Apache License 2.0](LICENSE).
