# Ember (`ember-custody`)

> **Proof that survives erasure.** Cryptographic custody, redactable Merkle provenance receipts, and adversarially tested cascade deletion for AI memory and RAG stores.

Existing memory engines (`Mem0`, `Qdrant`, `Chroma`, `pgvector`, `Pinecone`, `LangChain`, `LlamaIndex`) know how to store embeddings and delete by top-level `user_id`. Under GDPR Article 17 ("Right to be Forgotten") and enterprise audit requirements, three failures still happen silently:

1. **The Erasure / Receipt Trilemma:** If two subjects appear in the same source transcript (or even the same compound sentence), a flat SHA-256 document hash forces an impossible choice: delete Subject A cleanly and break Bystander B's provenance receipt, or preserve B's receipt by leaving A's source file unredacted.
2. **Derived & Cross-Bucket Leaks:** Calling `delete_all(user_id="priya")` deletes raw memories keyed to `"priya"`, leaving downstream turn-derived team summaries, shared workspace notes, and append-only audit logs surfacing `"Priya"` forever.
3. **Zombie Re-Ingestion & Fake Metrics:** Sync retries or un-tagged follow-up writes re-ingest erased subjects seconds later; naive deletion metrics pass `0 leaks` trivially if a system wipes bystander data along with the subject.

**Ember** wraps your existing memory, vector DB, framework retriever, HTTP service, or MCP agent pipeline to solve all three with explicit, falsifiable cryptographic mechanisms.

---

## Core Guarantees

- **Redactable Salted Merkle Provenance — Leaf & Sub-Leaf (`ember.merkle`, `ember.receipts`):** Documents split into sentence/line/custom leaves—and optionally compound `span_mode="word" | "clause"` sub-leaves—with length-prefixed per-node 128-bit random nonces (`0x00 || len32(nonce) || nonce || leaf_bytes`, RFC 6962 domain separation). Erasing Subject A destroys A's plaintext and nonce while keeping A's salted digest so **Bystander B's inclusion proof still verifies** even inside the exact same compound sentence.
- **Automatic Turn-Scoped Lineage & Zero-Plaintext Vendor Storage (`ember.vault`, `ember.adapters`):**
  - `vault.trace_turn()` automatically tracks reads -> `derived_from` writes within an agent turn (`prune="all" | "lexical_overlap" | callable`) without manual wiring.
  - `encrypt_vendor_payload=True` sends AES-256-GCM ciphertext and opaque HMAC `subject_ref` filters (with your `embedder` or keyed HMAC blind-index search tokens) so third-party vector DBs never hold plaintext PII.
- **Hash-Chained Ledger, Signed Checkpoints & Enterprise KMS (`ember.ledger`, `ember.keystore`, `ember.kms`):**
  - Append-only `SQLiteLedger` (`WAL`, `synchronous=FULL`) or `PostgresLedger` encrypts `subjects`, `text`, `quote`, and caller `metadata` with `AES-256-GCM` bound via AAD to `(tenant, item_id, field, key_ref)`.
  - Ed25519-signed ciphertext-only `checkpoints` support `O(delta)` startup without writing plaintext to snapshots.
  - Key management spans `FileKeyStore`, `SQLiteKeyStore` (`PRAGMA secure_delete=ON`), and envelope-wrapped KMS (`AwsKmsKeyStore`, `GcpKmsKeyStore`, `VaultTransitKeyStore`).
- **Keyed Content Tombstones, Fail-Closed Certificates & External Anchoring (`ember.reducer`, `ember.signing`):**
  - Vault-keyed HMAC-SHA256 subject and content tombstones block silent/un-tagged zombie re-ingestion (`TombstonedSubjectError`) unless an explicit `reconsent_id` is supplied.
  - `erase()` synthesizes `auto_probes` alongside caller `probes`, queries the **raw wrapped store**, raises `ErasureIncomplete` if any residue remains, signs an `Ed25519` certificate, and optionally anchors digests to RFC 3161 TSAs (`Rfc3161Anchor`) or Sigstore Rekor (`RekorAnchor`).
- **Offline Verifier & Air-Gapped Compliance Dossier (`ember/verify.py`, `ember/report.py`):**
  - Copy [`src/ember/verify.py`](src/ember/verify.py) anywhere to verify Merkle redactions, bystander receipts, and ledger hash chains using Python's standard library alone (`python3 -S`).
  - Render self-contained zero-CDN HTML compliance dossiers with `ember verify bundle.json --html report.html`.

---

## Installation

```bash
pip install ember-custody
```

Optional extras for your stack:

```bash
pip install "ember-custody[mem0]"       # Mem0 wrapper + adapter
pip install "ember-custody[qdrant]"     # Qdrant native adapter
pip install "ember-custody[chroma]"     # Chroma native adapter
pip install "ember-custody[postgres]"   # PostgresLedger & PgVectorAdapter (psycopg3)
pip install "ember-custody[pinecone]"   # Pinecone native adapter
pip install "ember-custody[langchain]"  # EmberLangChainRetriever / VectorStore
pip install "ember-custody[llamaindex]" # EmberLlamaIndexRetriever
pip install "ember-custody[aws]"        # AwsKmsKeyStore (boto3)
pip install "ember-custody[gcp]"        # GcpKmsKeyStore (google-cloud-kms)
pip install "ember-custody[server]"     # Optional ASGI mode (starlette + uvicorn)
```

From a local clone (using [`uv`](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/Starksood/ember.git
cd ember
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

See [Architecture](docs/ARCHITECTURE.md), [Threat Model](docs/THREAT_MODEL.md), and [Solving the Erasure / Receipt Trilemma](docs/TRILEMMA.md) for the cryptographic details.

---

## 60-Second Quickstart

### 1. Drop-In Frameworks, Native Vector DBs & `TurnTrace`

Wrap `Mem0` in one line (`EmberMem0`), drop into `LangChain` (`EmberLangChainRetriever`, `EmberLangChainVectorStore`) or `LlamaIndex` (`EmberLlamaIndexRetriever`), or construct `CustodyVault` directly over `QdrantAdapter`, `ChromaAdapter`, `PgVectorAdapter`, `PineconeAdapter`, `Mem0Adapter`, `SQLiteStoreAdapter`, or `InMemoryStoreAdapter`:

```python
from mem0 import Memory
from ember import CustodyVault, Mem0Adapter, TombstonedSubjectError

vault = CustodyVault.open(
    "./ember_ledger",
    key_dir="./ember_keys",         # keep keys separate from ledger backups (or use AWS/GCP/Vault KMS)
    adapter=Mem0Adapter(Memory(), default_user_id="team_acme"),
    encrypt_vendor_payload=True,    # zero plaintext PII inside the vendor vector DB
)

# Register a compound multi-party intake sentence with clause-level sub-leaf redaction
vault.register_source(
    "intake_notes",
    "Priya Raman works in Portland on triage routing, and Dana Kim leads pharmacy inventory.",
    span_mode="clause",
    origin="onboarding_transcript",
    origin_kind="conversation",
)

r_priya = vault.remember(
    "Priya Raman works in Portland on triage routing.",
    subjects=["priya"],
    source_id="intake_notes",
    quote="Priya Raman works in Portland on triage routing,",
    metadata={"user_id": "priya"},
)
r_dana = vault.remember(
    "Dana Kim leads pharmacy inventory.",
    subjects=["dana"],
    source_id="intake_notes",
    quote="and Dana Kim leads pharmacy inventory.",
    metadata={"user_id": "dana"},
)

# TurnTrace automatically links search() reads -> remember() derived_from inside an agent turn
with vault.trace_turn():
    _ = vault.search("Priya triage")
    vault.remember(
        "Riverside handoff: triage routing pairs with pharmacy inventory.",
        subjects=["team_acme"],
        metadata={"user_id": "team_acme"},
    )

# Right to be Forgotten: cascades to vendor records, derived turn notes, compound clauses, and keys
cert = vault.erase(
    "priya",
    probes=["Where does Priya work?", "Who handles triage routing?"],
)

# Dana's clause in the SAME sentence + her redacted source proof survive intact
assert vault.verify_item_receipt(r_dana.item.item_id) is True
assert vault.search("Dana pharmacy").abstained is False

# Both explicit re-ingest AND untagged content mentioning Priya fail closed
try:
    vault.remember("Priya Raman needs urgent follow-up.", subjects=["crm_sync_bot"])
except TombstonedSubjectError:
    print("Blocked zombie re-ingestion")
```

Async agents can use `await AsyncCustodyVault.open(...)` or `async with vault.trace_turn():`. Run self-contained examples without external API keys:

```bash
python examples/wrap_mem0.py
python examples/redactable_transcript.py
python examples/offline_verify_bundle.py
```

### 2. CLI, `ember.toml` & Air-Gapped HTML Dossier

```bash
ember --vault .ember --key-dir .keys --keystore-backend sqlite init
ember --vault .ember --key-dir .keys add-source doc1 \
  --text "Priya lives in Portland, and Marcus lives in Leeds." \
  --span-mode clause
ember --vault .ember --key-dir .keys remember "Priya lives in Portland." --subject priya --source-id doc1 --meta email=priya@example.org
ember --vault .ember --key-dir .keys remember "Marcus lives in Leeds." --subject marcus --source-id doc1
ember --vault .ember --key-dir .keys erase priya --probe "Where does Priya live?"
ember --vault .ember --key-dir .keys checkpoint
ember --vault .ember --key-dir .keys replay --use-checkpoint
ember --vault .ember --key-dir .keys export --out bundle.json
ember verify bundle.json --require-signature --html compliance-dossier.html
```

Or copy `src/ember/verify.py` anywhere and run with Python standard library alone:

```bash
python3 -S verify.py bundle.json
```

All CLI and sidecar subcommands also accept `--config ember.toml` (`[vault]`, `[ledger]`, `[keystore]`, `[adapter]`, `[anchor]`, `[server]`, with `${ENV_VAR:-default}` expansion).

### 3. HTTP Custody Sidecar, TypeScript SDK & MCP Server

Run Ember as a polyglot HTTP sidecar or stdio MCP server for non-Python agents and IDE tools:

```bash
# HTTP custody sidecar (stdlib ThreadingHTTPServer, or add --asgi for Starlette/uvicorn)
export EMBER_API_TOKEN="your-secret-token"
ember --vault .ember --key-dir .keys serve --host 127.0.0.1 --port 8420 --require-tenant-header

# Model Context Protocol (MCP) JSON-RPC 2.0 stdio server
ember-mcp --vault .ember --key-dir .keys
```

TypeScript / Node.js / Bun / Deno clients use the zero-dependency SDK in [`sdk/ts/`](sdk/ts/src/index.ts):

```ts
import { EmberClient } from "@ember-custody/client";

const ember = new EmberClient({
  baseUrl: "http://127.0.0.1:8420",
  bearerToken: process.env.EMBER_API_TOKEN,
  tenantId: "default",
});

await ember.registerSource({
  source_id: "doc1",
  text: "Maya takes warfarin daily, and Leo plays violin.",
  span_mode: "clause",
});
await ember.remember({ text: "Maya takes warfarin daily", subjects: ["maya"], source_id: "doc1" });
const cert = await ember.erase({ subject_id: "maya", probes: ["warfarin"] });
```

---

## Adversarial Benchmark & Two-Direction Calibration

Run both the V2 suite and the expanded V3 Phase-2 suite across 12 subjects × 7 attack classes (`direct`, `paraphrase`, `inference`, `multi_hop`, `attribute`, `temporal`, `aggregate` — 204 attack probes total):

```bash
ember benchmark --suite v3
```

Measured output against `InMemoryStoreAdapter`:

```text
EMBER ERASURE ADVERSARIAL SUITE V3 — 12 subjects, 204 probes (7 attack classes)
  Adapter   : InMemoryStoreAdapter (retrieval=keyword_overlap, store=in-process)
  Embedding : none
  Search    : bystander_limit=5, erase_probe_limit=25, naive_delete=user_id_metadata_exact, probe_limit=10
==============================================================================
  Adapter              : InMemoryStoreAdapter (keyword_overlap, embed=none)
  Baseline naive leaks : 204/204
  Ember leaks          : 0/204
  Bystanders survived  : True
  Receipts intact      : True
  Replays content-free : True
  Ed25519 certs valid  : True
  Sub-leaf redactions  : True
  TurnTrace derived ok : True
  Vendor zero-plaintext: True
  Zombie content block : True
```

### Real Vector-Store Comparison (`Mem0` + Qdrant + Dense Embeddings)

`run_benchmark()` and `run_v3_benchmark()` accept any `BenchmarkBackendSpec` (`mem0_backend_spec`) so the exact same 204-probe corpus, bystander survival checks, redacted Merkle receipt checks, content-free replay verification, and Ed25519 certificate verification can run against a live vector store:

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

1. **Key Separation & Backups:** Crypto-shredding protects live logs and snapshots taken *after* erasure only if pre-erasure backups did not copy unshredded keys alongside ledger storage. Configure a separate `key_dir` or use `AwsKmsKeyStore`, `GcpKmsKeyStore`, or `VaultTransitKeyStore` + `PostgresLedger`.
2. **External Time Anchoring:** Self-signed local `Ed25519` proofs establish internal ordering and integrity; preventing an operator holding both ledger and signing key from backdating history requires `Rfc3161Anchor` or `RekorAnchor`.
3. **Wrapped Store Internal History Tables (e.g., Mem0 `history.db`):** `Mem0Adapter` deletes live Mem0 vector points (and Mem0 strips entity links), while Ember's own ledger stays AES-256-GCM encrypted and crypto-shredded. However, OSS `mem0.Memory` also maintains an internal plaintext SQLite history DB (`history_db_path`, logging `old_memory` even on `"DELETE"` events). Configure `encrypt_vendor_payload=True` + an `embedder` (bypassing `.add()`) or isolate/prune `history_db_path`.

---

## Origin & License

Ember distills the verifiable provenance, salted Merkle redaction, crypto-shredding, and two-direction audit primitives developed in [Fireweed](https://github.com/Starksood) into a standalone Apache-2.0 library focused strictly on cryptographic custody.

Licensed under the [Apache License 2.0](LICENSE).
