"""Ember: cryptographic custody, redactable Merkle receipts, and verifiable cascade erasure."""

from .adapters import (
    AdapterHit,
    InMemoryStoreAdapter,
    Mem0Adapter,
    MemoryStoreAdapter,
    SQLiteStoreAdapter,
    VectorStoreAdapter,
)
from .audit import (
    SUITE_V2,
    BenchmarkBackendSpec,
    ScenarioBackendPair,
    SubjectScenario,
    in_memory_backend_spec,
    mem0_backend_spec,
    run_benchmark,
    verify_suite_hygiene,
)
from .crypto import CIPHER_NAME, ERASED_TOMBSTONE, DecryptionError
from .keystore import FileKeyStore, KeyStore, MemoryKeyStore, ShredReport
from .ledger import IdempotencyConflict, LedgerIntegrityError, SQLiteLedger
from .merkle import RedactableDoc, inclusion_proof, leaf_hash, merkle_root, split_parts
from .receipts import Receipt, locate_span, mint_flat_receipt, mint_redactable_receipt
from .records import ClosureManifest, ErasureCertificate, LedgerEvent, MemoryItem, SourceRecord
from .reducer import (
    EmberState,
    ErasureIncomplete,
    InvariantViolation,
    ReductionError,
    TombstonedSubjectError,
    apply_event,
    compute_closure,
    decrypt_item_metadata,
    decrypt_item_payload,
)
from .signing import AnchorReceipt, CallbackAnchor, Ed25519Signer, NullAnchor, TimestampAnchor
from .vault import CustodyVault, QueryHit, RememberResult, SearchResult
from .verify import verify_bundle, verify_ed25519_certificate, verify_ledger_chain, verify_redactable_receipt

__version__ = "0.1.0"

__all__ = [
    "AdapterHit",
    "AnchorReceipt",
    "BenchmarkBackendSpec",
    "CIPHER_NAME",
    "CallbackAnchor",
    "ClosureManifest",
    "CustodyVault",
    "DecryptionError",
    "ERASED_TOMBSTONE",
    "Ed25519Signer",
    "EmberState",
    "ErasureCertificate",
    "ErasureIncomplete",
    "FileKeyStore",
    "IdempotencyConflict",
    "InMemoryStoreAdapter",
    "InvariantViolation",
    "KeyStore",
    "LedgerEvent",
    "LedgerIntegrityError",
    "Mem0Adapter",
    "MemoryItem",
    "MemoryKeyStore",
    "MemoryStoreAdapter",
    "NullAnchor",
    "QueryHit",
    "Receipt",
    "RedactableDoc",
    "ReductionError",
    "RememberResult",
    "SQLiteLedger",
    "SQLiteStoreAdapter",
    "SUITE_V2",
    "ScenarioBackendPair",
    "SearchResult",
    "ShredReport",
    "SourceRecord",
    "SubjectScenario",
    "TimestampAnchor",
    "TombstonedSubjectError",
    "VectorStoreAdapter",
    "__version__",
    "apply_event",
    "compute_closure",
    "decrypt_item_metadata",
    "decrypt_item_payload",
    "in_memory_backend_spec",
    "inclusion_proof",
    "leaf_hash",
    "locate_span",
    "mem0_backend_spec",
    "merkle_root",
    "mint_flat_receipt",
    "mint_redactable_receipt",
    "run_benchmark",
    "split_parts",
    "verify_bundle",
    "verify_ed25519_certificate",
    "verify_ledger_chain",
    "verify_redactable_receipt",
    "verify_suite_hygiene",
]
