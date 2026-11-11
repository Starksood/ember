"""Byte-range and redactable Merkle provenance receipts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .merkle import RedactableDoc, inclusion_proof
from .verify import _is_strict_int, sha256_prefixed, verify_flat_receipt, verify_redactable_receipt


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
    span_indices: tuple[int, ...] = ()
    span_count: int = 0
    span_proofs: tuple[tuple[tuple[str, str], ...], ...] = ()

    @property
    def redaction_safe(self) -> bool:
        return bool(self.merkle_root) and self.leaf_index >= 0 and self.leaf_count > 0

    @property
    def span_level(self) -> bool:
        return bool(self.span_indices) and self.span_count > 0 and len(self.span_proofs) == len(self.span_indices)

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
            if self.span_level:
                data["span_indices"] = list(self.span_indices)
                data["span_count"] = self.span_count
                data["span_proofs"] = [[list(step) for step in sp_proof] for sp_proof in self.span_proofs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        raw_span_proofs = data.get("span_proofs", ())
        parsed_span_proofs = tuple(
            tuple((str(s), str(h)) for s, h in sp_proof)
            for sp_proof in raw_span_proofs
        )
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
            span_indices=tuple(int(i) for i in data.get("span_indices", ())),
            span_count=int(data.get("span_count", 0)),
            span_proofs=parsed_span_proofs,
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


def _locate_quote_in_compound_entry(
    entry: dict[str, Any],
    quote: str,
) -> tuple[tuple[int, ...], str, int, int] | None:
    """Locate `quote` inside surviving sub-leaf spans of `entry`, returning `(span_indices, quote, b_start, b_end)`."""
    spans = entry.get("spans") or []
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for sp in spans:
        if sp.get("redacted"):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(sp)
    if current:
        segments.append(current)

    tokens = quote.split()
    ws_pattern = re.compile(r"\s+".join(re.escape(tok) for tok in tokens)) if tokens else None

    for seg in segments:
        parts: list[str] = []
        char_ranges: list[tuple[int, int, dict[str, Any]]] = []
        cursor = 0
        prev_end: int | None = None
        for sp in seg:
            b_start = sp.get("byte_start")
            b_end = sp.get("byte_end")
            piece = str(sp["text"])
            if parts:
                if _is_strict_int(b_start) and prev_end is not None and b_start <= prev_end:
                    pass
                else:
                    parts.append(" ")
                    cursor += 1
            c_start = cursor
            parts.append(piece)
            cursor += len(piece)
            char_ranges.append((c_start, cursor, sp))
            prev_end = b_end if _is_strict_int(b_end) else None

        rendered = "".join(parts)
        pos = rendered.find(quote)
        if pos != -1:
            m_start, m_end = pos, pos + len(quote)
        elif ws_pattern is not None:
            match = ws_pattern.search(rendered)
            if match is None:
                continue
            m_start, m_end = match.span()
        else:
            continue

        overlapped = [
            (cs, ce, sp) for cs, ce, sp in char_ranges if cs < m_end and m_start < ce
        ]
        if not overlapped:
            continue
        first_cs, _, first_sp = overlapped[0]
        last_cs, last_ce, last_sp = overlapped[-1]
        matched_quote = rendered[m_start:m_end]
        head_bytes = len(first_sp["text"][: max(0, m_start - first_cs)].encode("utf-8"))
        tail_offset = max(0, last_ce - m_end)
        last_text = last_sp["text"]
        tail_bytes = len(last_text[len(last_text) - tail_offset :].encode("utf-8")) if tail_offset else 0
        b_start_out = int(first_sp["byte_start"]) + head_bytes
        b_end_out = int(last_sp["byte_end"]) - tail_bytes
        span_indices = tuple(int(sp["span_index"]) for _, _, sp in overlapped)
        return span_indices, matched_quote, b_start_out, b_end_out
    return None


def mint_redactable_receipt(
    doc: RedactableDoc,
    quote: str,
    *,
    raw_doc: bytes | None = None,
    leaf_index: int | None = None,
    source_id: str | None = None,
) -> Receipt:
    """Mint a receipt binding `quote` to both its exact bytes and its Merkle leaf/span proof."""
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("quote must be a non-empty string")
    chosen_idx = leaf_index
    matched_quote = quote
    matched_span_info: tuple[tuple[int, ...], str, int, int] | None = None

    if chosen_idx is None:
        for entry in doc.entries:
            if entry.get("redacted"):
                continue
            if "spans" in entry:
                found = _locate_quote_in_compound_entry(entry, quote)
                if found is not None:
                    chosen_idx = entry["index"]
                    matched_span_info = found
                    matched_quote = found[1]
                    if found[1] == quote:
                        break
            else:
                leaf_text = entry["text"]
                if leaf_text == quote:
                    chosen_idx = entry["index"]
                    matched_quote = leaf_text
                    break
        if chosen_idx is None:
            for entry in doc.entries:
                if entry.get("redacted") or "spans" in entry:
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
        if "spans" in entry:
            matched_span_info = _locate_quote_in_compound_entry(entry, quote)
            if matched_span_info is None:
                raise ValueError(f"quote not found inside compound leaf {chosen_idx}")
            matched_quote = matched_span_info[1]
        else:
            span = locate_span(entry["text"].encode("utf-8"), quote)
            if span is None:
                raise ValueError(f"quote not found inside leaf {chosen_idx}")
            matched_quote = entry["text"].encode("utf-8")[span[0] : span[1]].decode("utf-8")

    if chosen_idx is None:
        raise ValueError(f"quote not located in any surviving leaf of source {source_id or doc.source_id!r}")

    entry = doc.entries[chosen_idx]
    span_indices: tuple[int, ...] = ()
    span_count = 0
    span_proofs: tuple[tuple[tuple[str, str], ...], ...] = ()

    if "spans" in entry:
        if matched_span_info is None:
            matched_span_info = _locate_quote_in_compound_entry(entry, matched_quote)
        if matched_span_info is None:
            raise ValueError(f"matched quote could not be aligned inside compound leaf {chosen_idx}")
        span_indices, matched_quote, byte_start, byte_end = matched_span_info
        span_hashes = doc.span_hashes_for(chosen_idx) or []
        span_count = len(span_hashes)
        span_proofs = tuple(tuple(inclusion_proof(span_hashes, si)) for si in span_indices)
    else:
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
        span_indices=span_indices,
        span_count=span_count,
        span_proofs=span_proofs,
    )


__all__ = [
    "Receipt",
    "locate_span",
    "mint_flat_receipt",
    "mint_redactable_receipt",
]
