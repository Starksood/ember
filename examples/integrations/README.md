# Ember End-to-End Integration Templates

Two runnable, self-contained application integrations built against `ember-custody` `0.3.0`:

| Template | Stack / Pattern | What It Validates |
|---|---|---|
| [`support_bot_app.py`](support_bot_app.py) | `EmberMem0` + `CustodyVault` + `EmberHttpApp` sidecar | Clause-level `span_mode="clause"`, `trace_turn()`, HTTP `GET /v1/coverage`, mid-erase outage recovery via `POST /v1/reconcile`, HTML compliance dossier export |
| [`langgraph_research_agent.py`](langgraph_research_agent.py) | `EmberLangChainVectorStore` + `EmberLangChainRetriever` + `SQLiteStoreAdapter` | Multi-node stateful research agent graph where an observed node cascade-deletes while an out-of-band tool write is caught by `vault.audit_coverage()` & stamped into `cert.trust_boundary["coverage"]` |

## Run Locally

```bash
python examples/integrations/support_bot_app.py --benchmark
python examples/integrations/langgraph_research_agent.py --benchmark
```

## Integration Diff (~6-10 Lines)

### 1. Mem0-Backed Support Assistant
Before:
```python
mem = Memory()
mem.add("Alice reported billing error", user_id="customer_alice")
hits = mem.search("billing", user_id="customer_alice")
mem.delete_all(user_id="customer_alice")
```
After (preserving vendor payload zero-plaintext and HTTP reconciliation):
```python
vault = CustodyVault.open("./vault_data", key_dir="./vault_keys", adapter=Mem0Adapter(mem_client, infer=False), encrypt_vendor_payload=True)
mem = EmberMem0(vault)
with mem.trace_turn():
    hits = mem.search("billing", user_id="customer_alice")
    mem.add("Escalation triage memo", user_id="triage_queue")
cov = vault.audit_coverage("customer_alice")
cert = mem.erase("customer_alice")  # or vault.reconcile_erasures() on transient failure
```

### 2. LangChain / LangGraph Agent
```python
store = EmberLangChainVectorStore(vault, subject_key="user_id")
retriever = store.as_retriever(search_kwargs={"k": 4})
with vault.trace_turn():
    docs = retriever.invoke("lithium therapy")
    store.add_texts(["Pharmacy protocol memo"], metadatas=[{"user_id": "ops"}])
report = vault.audit_coverage("patient_nora")
```

## Observed Overhead Telemetry (Apple Silicon / SQLite WAL + FileKeyStore)

Measured directly from running both templates on 2026-11-12:

| Operation | Raw Store p50 | Ember Wrapped p50 | Ember Wrapped p95 |
|---|---:|---:|---:|
| `Mem0` `add` (`encrypt_vendor_payload=True`, n=30) | `0.001 ms` | `0.372 ms` | `0.490 ms` |
| `Mem0` `search` (`encrypt_vendor_payload=True`, n=30) | `0.004 ms` | `0.041 ms` | `0.056 ms` |
| `LangChain` `add_texts` (`SQLiteStoreAdapter`, 64d cosine, n=25) | — | `0.551 ms` | `0.688 ms` |
| `LangChain` `retriever.invoke` (64d cosine, n=25) | — | `0.165 ms` | `0.298 ms` |
| `SUITE_V4` full `erase()` (key shred + redaction + probes + Ed25519 sign, n=25) | — | `1.730 ms` | `2.664 ms` |

## Integration Friction Discovered & Fixed

1. **Unicode NFC vs. NFD Identifier Mismatch:** Erasing an alias in NFD form previously hashed to a different `tombstone_token` / `content_tombstone_token` than an NFC query. Resolved by applying Unicode `NFKC` normalization in `keystore._normalize_identifier()` and `vault._extract_content_candidates()`.
2. **Dense Vector Top-k Threshold vs. Substring Abstention:** Under dense vector similarity (`k=4`), irrelevant surviving documents still rank below the top match if similarity thresholding isn't applied, so application assertions should check content / subject filters or configure similarity cutoffs explicitly.
3. **Unobserved External Tools & Trust Warnings:** Any tool that writes without `trace_turn()`, `derived_from`, or `source_id` shows up immediately in `audit_coverage().unattributed_item_ids` and degrades the certificate's `trust_boundary["coverage"]["verdict"]` from `"complete"` to `"bounded_with_warnings"` instead of silently pretending total erasure.
