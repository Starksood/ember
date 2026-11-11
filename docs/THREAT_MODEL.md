# Threat Model & Scope Boundaries

Ember prefers explicit boundaries over security theater.

## Adversary Capabilities & Defenses

| Threat / Attack | Ember Defense | Mechanism |
|---|---|---|
| Silent partial edit, deletion, or reordering of historical events | **Detected** | Gap-free per-tenant `seq`, `prev_hash`, SHA-256 chain (`SQLiteLedger`) |
| Dictionary confirmation of a redacted sentence ("Priya joined Acme") | **Blocked** | Per-leaf 128-bit random nonce destroyed at redaction; length-prefixed hash (`0x00 \|\| len32(nonce) \|\| nonce \|\| text`) |
| Merkle second-preimage / tree-geometry spoofing | **Blocked** | RFC 6962 `0x00` vs `0x01` domain separation; strict odd-promotion geometry check in `verify_inclusion` |
| Redacting Subject A breaking Bystander B's provenance proof | **Solved** | Per-leaf Merkle inclusion proofs keep root invariant across bystander-disjoint redactions |
| Downstream derived summary / reflection leaking erased subject | **Blocked** | Exact transitive `derived_from` closure + shared `record` DEK shredding + external ID deletion |
| Cross-record or cross-tenant ciphertext splicing | **Blocked** | AES-256-GCM AAD binds `(tenant_id, record_id, field, key_ref)` |
| Caller `metadata` PII (`user_id`, emails) lingering in ledger logs | **Blocked** | Encrypted into `metadata_ciphertext` via AES-256-GCM; redacted to `{}` on key shred |
| Transient vendor API failure leaving keys unshredded mid-erasure | **Blocked** | Local source redaction + key shredding run before external `adapter.delete()`; retry sweeps prior pending closure IDs |
| Dictionary test on `ERASE_SUBJECT` tombstone names in ledger | **Blocked** | Vault-keyed `HMAC-SHA256(tombstone_secret, normalized_identifier)` |
| Silent re-ingestion of erased subject on retry/sync or un-tagged zombie prose | **Blocked** | Keyed subject AND content tombstone tokens (`HMAC-SHA256`) block matching identifiers, aliases, and n-grams in `remember()` unless `reconsent_id` is supplied |
| Sub-leaf multi-subject sentence redaction breaking bystander proofs | **Solved** | Compound Merkle leaves (`span_mode="word" \| "clause"`) strip redacted spans' text + nonces while preserving bystander spans & offline inclusion proofs |
| Plaintext text/PII stored on third-party vector DB host | **Mitigated** | `encrypt_vendor_payload=True` sends AES-256-GCM ciphertext (`text_ct`) and opaque HMAC `subject_ref` scopes with external embeddings or HMAC blind-index tokens |
| Fake / over-deleting erasure scoring 0 leaks by wiping bystander | **Rejected** | `CustodyVault.erase` and `audit.py` require surviving bystander retrieval & bystander receipt verification |
| Erasure claimed complete despite vendor store still answering | **Blocked** | `erase()` synthesizes auto-probes from doomed items + explicit `probes`, queries underlying adapter raw search, and raises `ErasureIncomplete` without issuing a certificate |
| Operator/root rewriting local SQLite + Ed25519 timestamps coherently | **Mitigated** | `Rfc3161Anchor` (DER TSA token) & `RekorAnchor` (Sigstore transparency log) bind certificate SHA-256 digests externally |
| Cross-tenant or unauthenticated HTTP sidecar (`ember serve`) calls | **Blocked** | Constant-time Bearer auth (`hmac.compare_digest`), non-loopback no-auth bind refusal, optional `X-Ember-Tenant` lock, body cap (`413`) |
| Malicious bundle payload injecting HTML/JS into audit reports | **Blocked** | `generate_html_report` runs strict `html.escape(..., quote=True)` across every tenant, quote, doc, and JSON field with zero external CDNs |

## Explicit Out-of-Scope Boundaries

1. **Snapshots taken prior to key separation:** If an operator backs up `ledger.db` and `keyring.json` together before erasure, the backup retains both ciphertexts and unshredded keys. Always separate `key_dir` onto an unbacked KMS/HSM or dedicated volume.
2. **Whole-log rewriting without external anchor:** Without configuring a `TimestampAnchor` (RFC 3161 TSA or external transparency log), an operator holding both `ledger.db` and the Ed25519 private signing key can coherently rewrite both timestamps and logs. Certificates report `trust_boundary.externally_timestamped` honestly (`False` under `NullAnchor`).
3. **Opaque LLM paraphrased leaks outside `derived_from` / `trace_turn()`:** Structural cascade erasure tracks `subjects`, explicit `derived_from`, turn-scoped `trace_turn()` read->write capture, `Mem0(infer=True)` `UPDATE`/`DELETE` consolidations, and optional `leaf_predicate`/`span_predicate` sweeps. Unlinked paraphrased writes that drop `derived_from` outside `trace_turn()` and rephrase identifiers rely on auto/explicit `probes` and caller discipline.
4. **Vendor-internal append tables outside the search index (e.g., Mem0 `history.db`):** `Mem0Adapter` removes live vectors via Mem0's `delete()` API and covers Mem0 scopes on live search probes. However, OSS `mem0.Memory` writes an internal SQLite change history (`history_db_path`, recording `old_memory` alongside `"DELETE"` events) when using `.add()`; either pass `encrypt_vendor_payload=True` with an `embedder` (bypassing `Mem0.add()` via direct `vector_store.insert`) or isolate/rotate `history_db_path`.
5. **Semantic inversion of dense embeddings (`encrypt_vendor_payload=True` + `embedder`):** Storing AES-256-GCM ciphertext (`enc2:...`) prevents plaintext/metadata exposure in vendor DB dumps, logs, and dashboards, and prevents vendor LLMs from reading data. However, raw float embeddings still inhabit semantic geometry and can reveal coarse topic neighborhoods to an adversary with vector access prior to deletion (after erasure, Ember deletes the vector record too). Where embedding-inversion exposure is unacceptable, omit `embedder` so Ember indexes keyed HMAC blind-index tokens instead of dense float vectors.
