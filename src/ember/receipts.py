"""Byte-range and redactable Merkle provenance receipts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .merkle import RedactableDoc, inclusion_proof
from .verify import sha256_prefixed, verify_flat_receipt, verify_redactable_receipt


def locate_span(doc: bytes, quote: str) -> tuple[int, int] | None:
    """Return exact UTF-8 `(byte_start, byte_end)` of `quote` inside `doc`, or None."""
    if not quote:
        return None
    q_bytes = quote.encode("utf-8")
    idx = doc.find(q_bytes)
    if idx != -1:
        return (idx, idx + len(q_bytes))
    try:
        text = doc.decode("utf-8")
    except UnicodeDecodeError:
        return None
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(tok) for tok in tokens)
    match = re.search(pattern, text)
    if match is None:
        return None
    start = len(text[: match.start()].encode("utf-8"))
    end = len(text[: match.end()].encode("utf-8"))
    return (start, end)


@dataclass(frozen=True)
class Receipt:
    """Dual-mode provenance binding: flat SHA-256 byte range + redactable Merkle proof."""

    source_id: str
    quote: str
    flat_doc_hash: str
    byte_start: int
    byte_end: int
    merkle_root: str = ""
    leaf_index: int = -1
    leaf_count: int = 0
    proof: tuple[tuple[str, str], ...] = ()

    @property
    def redaction_safe(self) -> bool:
        return bool(self.merkle_root) and self.leaf_index >= 0 and self.leaf_count > 0

    def verify_flat(self, raw_doc: bytes) -> bool:
        return verify_flat_receipt(self.as_dict(), raw_doc)

    def verify_redactable(self, doc: RedactableDoc | dict[str, Any]) -> bool:
        doc_dict = doc.as_dict() if isinstance(doc, RedactableDoc) else doc
        return verify_redactable_receipt(self.as_dict(), doc_dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source_id": self.source_id,
            "quote": self.quote,
            "flat_doc_hash": self.flat_doc_hash,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }
        if self.redaction_safe:
            data["merkle_root"] = self.merkle_root
            data["leaf_index"] = self.leaf_index
            data["leaf_count"] = self.leaf_count
            data["proof"] = [list(step) for step in self.proof]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        return cls(
            source_id=str(data.get("source_id", "")),
            quote=str(data["quote"]),
            flat_doc_hash=str(data.get("flat_doc_hash") or data.get("doc_hash", "")),
            byte_start=int(data["byte_start"]),
            byte_end=int(data["byte_end"]),
            merkle_root=str(data.get("merkle_root", "")),
            leaf_index=int(data.get("leaf_index", -1)),
            leaf_count=int(data.get("leaf_count", 0)),
            proof=tuple((str(s), str(h)) for s, h in data.get("proof", ())),
        )


def mint_flat_receipt(raw_doc: bytes, quote: str, *, source_id: str = "") -> Receipt:
    """Mint a flat byte-range receipt; raises ValueError if `quote` is absent."""
    span = locate_span(raw_doc, quote)
    if span is None:
        raise ValueError(f"quote is not a contiguous slice of source {source_id!r}")
    verbatim = raw_doc[span[0] : span[1]].decode("utf-8")
    return Receipt(
        source_id=source_id,
        quote=verbatim,
        flat_doc_hash=sha256_prefixed(raw_doc),
        byte_start=span[0],
        byte_end=span[1],
    )


def mint_redactable_receipt(
    doc: RedactableDoc,
    quote: str,
    *,
    raw_doc: bytes | None = None,
    leaf_index: int | None = None,
    source_id: str | None = None,
) -> Receipt:
    """Mint a receipt binding `quote` to both its exact bytes and its Merkle leaf proof."""
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("quote must be a non-empty string")
    chosen_idx = leaf_index
    matched_quote = quote

    if chosen_idx is None:
        for entry in doc.entries:
            if entry.get("redacted"):
                continue
            leaf_text = entry["text"]
            if leaf_text == quote:
                chosen_idx = entry["index"]
                matched_quote = leaf_text
                break
        if chosen_idx is None:
            for entry in doc.entries:
                if entry.get("redacted"):
                    continue
                leaf_text = entry["text"]
                span = locate_span(leaf_text.encode("utf-8"), quote)
                if span is not None:
                    chosen_idx = entry["index"]
                    matched_quote = leaf_text.encode("utf-8")[span[0] : span[1]].decode("utf-8")
                    break
    else:
        if not (0 <= chosen_idx < len(doc.entries)):
            raise IndexError(f"leaf_index {chosen_idx} out of range")
        entry = doc.entries[chosen_idx]
        if entry.get("redacted"):
            raise ValueError(f"cannot mint receipt on already-redacted leaf {chosen_idx}")
        span = locate_span(entry["text"].encode("utf-8"), quote)
        if span is None:
            raise ValueError(f"quote not found inside leaf {chosen_idx}")
        matched_quote = entry["text"].encode("utf-8")[span[0] : span[1]].decode("utf-8")

    if chosen_idx is None:
        raise ValueError(f"quote not located in any surviving leaf of source {source_id or doc.source_id!r}")

    entry = doc.entries[chosen_idx]
    leaf_bytes = entry["text"].encode("utf-8")
    rel_span = locate_span(leaf_bytes, matched_quote)
    if rel_span is None:
        raise ValueError(f"matched quote could not be aligned inside leaf {chosen_idx}")
    byte_start = int(entry["byte_start"]) + rel_span[0]
    byte_end = int(entry["byte_start"]) + rel_span[1]
    flat_hash = sha256_prefixed(raw_doc) if raw_doc is not None else ""
    if raw_doc is not None:
        # Validate flat slice matches verbatim bytes as well.
        if raw_doc[byte_start:byte_end].decode("utf-8", errors="replace") != matched_quote:
            flat_span = locate_span(raw_doc, matched_quote)
            if flat_span is None:
                raise ValueError("quote mismatch between RedactableDoc leaf and supplied raw_doc bytes")
            byte_start, byte_end = flat_span

    hashes = doc.hashes()
    return Receipt(
        source_id=source_id if source_id is not None else doc.source_id,
        quote=matched_quote,
        flat_doc_hash=flat_hash,
        byte_start=byte_start,
        byte_end=byte_end,
        merkle_root=doc.root(),
        leaf_index=chosen_idx,
        leaf_count=len(hashes),
        proof=tuple(inclusion_proof(hashes, chosen_idx)),
    )


__all__ = [
    "Receipt",
    "locate_span",
    "mint_flat_receipt",
    "mint_redactable_receipt",
]
