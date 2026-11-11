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
| Silent re-ingestion of erased subject on retry/sync | **Blocked** | Tombstone tokens block `remember()` with `TombstonedSubjectError` unless `reconsent_id` is supplied |
| Fake / over-deleting erasure scoring 0 leaks by wiping bystander | **Rejected** | `CustodyVault.erase` and `audit.py` require surviving bystander retrieval & bystander receipt verification |
| Erasure claimed complete despite vendor store still answering | **Blocked** | `erase()` probes underlying adapter raw search and raises `ErasureIncomplete` without issuing a certificate |

## Explicit Out-of-Scope Boundaries

1. **Snapshots taken prior to key separation:** If an operator backs up `ledger.db` and `keyring.json` together before erasure, the backup retains both ciphertexts and unshredded keys. Always separate `key_dir` onto an unbacked KMS/HSM or dedicated volume.
2. **Whole-log rewriting without external anchor:** Without configuring a `TimestampAnchor` (RFC 3161 TSA or external transparency log), an operator holding both `ledger.db` and the Ed25519 private signing key can coherently rewrite both timestamps and logs. Certificates report `trust_boundary.externally_timestamped` honestly (`False` under `NullAnchor`).
3. **Opaque LLM paraphrased leaks outside `derived_from`:** Structural cascade erasure tracks explicit `subjects` and `derived_from` lineage plus optional `leaf_predicate` sweeps. If an unlinked process copies Subject A's PII into an untagged external item with no `derived_from` link, pass comprehensive `probes` so `ErasureIncomplete` fails closed.
