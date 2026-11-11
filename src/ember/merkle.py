"""Redactable RFC 6962 salted Merkle documents.

Resolving the Erasure / Receipt Trilemma:
A flat whole-document hash invalidates innocent bystander receipts as soon as any
sentence of another subject is erased. Splitting a source into parts, minting a
per-leaf 128-bit random nonce, and publishing length-delimited RFC 6962 hashes
(`0x00 || uint32_be(len(nonce)) || nonce || part_bytes`) allows Ember to erase a
subject's plaintext and nonce while retaining the leaf hash so:
  1. The erased leaf cannot be confirmed via offline dictionary guessing,
  2. Bystanders' text stays intact, and
  3. Bystanders' receipts still verify against the unchanged Merkle root.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
import secrets
from typing import Any, Literal

from .verify import (
    _is_strict_int,
    _render_spans_text,
    interior_hash,
    leaf_hash,
    merkle_root,
    redactable_doc_hashes,
    verify_inclusion,
)

ChunkMode = Literal["sentence", "line", "whole"]
ChunkerFn = Callable[[str], list[tuple[int, int, str]]]
SpanMode = Literal["word", "clause"] | None
SpanChunkerFn = Callable[[str], list[tuple[int, int, str]]]

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_WORD_SPAN_RE = re.compile(r"[^\W_]+(?:[@.+_-][^\W_]+)*|[^\w\s]", re.UNICODE)
_CLAUSE_SPLIT_RE = re.compile(r"(?:[;,—–]|\s+(?:and|or|but|while|with|after|before)\s+)", re.IGNORECASE)


def _strip_span(text: str, char_start: int, char_end: int) -> tuple[int, int, str] | None:
    while char_start < char_end and text[char_start].isspace():
        char_start += 1
    while char_end > char_start and text[char_end - 1].isspace():
        char_end -= 1
    if char_start >= char_end:
        return None
    byte_start = len(text[:char_start].encode("utf-8"))
    byte_end = len(text[:char_end].encode("utf-8"))
    return byte_start, byte_end, text[char_start:char_end]


def split_parts_with_spans(text: str, mode: ChunkMode | ChunkerFn = "sentence") -> list[tuple[int, int, str]]:
    """Split text into `(utf8_byte_start, utf8_byte_end, slice_text)` leaves."""
    if not text or not text.strip():
        return []
    if callable(mode):
        raw_parts = mode(text)
        text_byte_len = len(text.encode("utf-8"))
        validated: list[tuple[int, int, str]] = []
        for entry in raw_parts:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError("custom chunker must return tuples of (byte_start, byte_end, text)")
            b_start, b_end, slice_text = entry
            if not _is_strict_int(b_start) or not _is_strict_int(b_end):
                raise ValueError("custom chunker offsets must be ints")
            if not (0 <= b_start <= b_end <= text_byte_len):
                raise ValueError(f"custom chunker byte range {(b_start, b_end)} out of bounds [0, {text_byte_len}]")
            if not isinstance(slice_text, str) or not slice_text:
                raise ValueError("custom chunker leaf text must be a non-empty string")
            validated.append((b_start, b_end, slice_text))
        return validated

    parts: list[tuple[int, int, str]] = []
    if mode == "whole":
        span = _strip_span(text, 0, len(text))
        return [span] if span is not None else []

    if mode == "line":
        pos = 0
        for line in text.splitlines(keepends=True):
            end = pos + len(line)
            span = _strip_span(text, pos, end)
            if span is not None:
                parts.append(span)
            pos = end
        return parts

    if mode == "sentence":
        start = 0
        for match in _SENTENCE_BREAK.finditer(text):
            span = _strip_span(text, start, match.start())
            if span is not None:
                parts.append(span)
            start = match.end()
        tail = _strip_span(text, start, len(text))
        if tail is not None:
            parts.append(tail)
        return parts

    raise ValueError(f"unsupported chunk mode: {mode!r}")


def split_parts(text: str, mode: ChunkMode | ChunkerFn = "sentence") -> list[str]:
    """Convenience helper returning the ordered leaf strings."""
    return [part for _, _, part in split_parts_with_spans(text, mode=mode)]


def split_spans_with_offsets(
    leaf_text: str,
    base_byte_start: int = 0,
    *,
    span_mode: SpanMode | SpanChunkerFn = "word",
) -> list[tuple[int, int, str]]:
    """Split a leaf string into `(doc_byte_start, doc_byte_end, span_text)` sub-leaves."""
    if span_mode is None or not leaf_text:
        return []
    if callable(span_mode):
        raw_spans = span_mode(leaf_text)
        leaf_byte_len = len(leaf_text.encode("utf-8"))
        validated: list[tuple[int, int, str]] = []
        for entry in raw_spans:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError("custom span_chunker must return tuples of (byte_start, byte_end, text)")
            b_start, b_end, s_text = entry
            if not _is_strict_int(b_start) or not _is_strict_int(b_end):
                raise ValueError("custom span_chunker offsets must be ints")
            if not (0 <= b_start <= b_end <= leaf_byte_len):
                raise ValueError(f"custom span range {(b_start, b_end)} out of bounds [0, {leaf_byte_len}]")
            if not isinstance(s_text, str) or not s_text:
                raise ValueError("custom span_chunker text must be a non-empty string")
            validated.append((base_byte_start + b_start, base_byte_start + b_end, s_text))
        return validated

    if span_mode == "word":
        out: list[tuple[int, int, str]] = []
        for m in _WORD_SPAN_RE.finditer(leaf_text):
            rel = _strip_span(leaf_text, m.start(), m.end())
            if rel is not None:
                out.append((base_byte_start + rel[0], base_byte_start + rel[1], rel[2]))
        return out

    if span_mode == "clause":
        out = []
        pos = 0
        for m in _CLAUSE_SPLIT_RE.finditer(leaf_text):
            rel = _strip_span(leaf_text, pos, m.start())
            if rel is not None:
                out.append((base_byte_start + rel[0], base_byte_start + rel[1], rel[2]))
            delim = _strip_span(leaf_text, m.start(), m.end())
            if delim is not None:
                out.append((base_byte_start + delim[0], base_byte_start + delim[1], delim[2]))
            pos = m.end()
        tail = _strip_span(leaf_text, pos, len(leaf_text))
        if tail is not None:
            out.append((base_byte_start + tail[0], base_byte_start + tail[1], tail[2]))
        return out

    raise ValueError(f"unsupported span_mode: {span_mode!r}")


def new_nonce() -> str:
    """Generate a 128-bit random hex nonce for one document leaf."""
    return secrets.token_hex(16)


def inclusion_proof(hashes: Sequence[str], index: int) -> list[tuple[str, str]]:
    """Build `(side, sibling_hex)` Merkle proof for `index` with odd-node promotion."""
    if not (0 <= index < len(hashes)):
        raise IndexError(f"leaf index {index} out of range for {len(hashes)} leaves")
    proof: list[tuple[str, str]] = []
    level = list(hashes)
    i = index
    while len(level) > 1:
        nxt = [interior_hash(level[j], level[j + 1]) for j in range(0, len(level) - 1, 2)]
        promoted = len(level) % 2 == 1
        if promoted:
            nxt.append(level[-1])
        if promoted and i == len(level) - 1:
            i = len(nxt) - 1
        else:
            sib = i ^ 1
            side = "right" if sib > i else "left"
            proof.append((side, level[sib]))
            i //= 2
        level = nxt
    return proof


def _clone_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(entry)
    if "spans" in copied and isinstance(copied["spans"], list):
        copied["spans"] = [dict(sp) for sp in copied["spans"]]
    return copied


def _leaf_hashes_for_entry(entry: Mapping[str, Any]) -> list[str]:
    spans = entry["spans"]
    out: list[str] = []
    for sp in spans:
        if sp.get("redacted"):
            out.append(sp["hash"])
        else:
            out.append(leaf_hash(sp["text"], sp["nonce"]))
    return out


def _entry_digest(entry: Mapping[str, Any]) -> str:
    if entry.get("redacted"):
        return str(entry["hash"])
    if "spans" in entry:
        return merkle_root(_leaf_hashes_for_entry(entry))
    return leaf_hash(entry["text"], entry["nonce"])


def _entry_text(entry: Mapping[str, Any], redaction_marker: str = "[redacted]") -> str:
    if entry.get("redacted"):
        return redaction_marker
    if "spans" in entry:
        return _render_spans_text(entry["spans"], redaction_marker=redaction_marker)
    return str(entry["text"])


@dataclass(frozen=True)
class RedactableDoc:
    """Immutable Merkle-indexed document supporting lawful leaf and sub-leaf span redaction."""

    source_id: str
    chunk_mode: str
    entries: tuple[dict[str, Any], ...]
    span_mode: str | None = None

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source_id: str = "",
        mode: ChunkMode | ChunkerFn = "sentence",
        nonces: Sequence[str] | None = None,
        span_mode: SpanMode | SpanChunkerFn = None,
    ) -> "RedactableDoc":
        parts = split_parts_with_spans(text, mode=mode)
        mode_name = mode if isinstance(mode, str) else getattr(mode, "__name__", "custom")
        if span_mode is None:
            if nonces is None:
                leaf_nonces = [new_nonce() for _ in parts]
            else:
                if len(nonces) != len(parts):
                    raise ValueError(f"expected {len(parts)} nonces, got {len(nonces)}")
                leaf_nonces = list(nonces)
                if any(not isinstance(n, str) or not n for n in leaf_nonces):
                    raise ValueError("every leaf nonce must be a non-empty string")
            entries = tuple(
                {
                    "index": idx,
                    "text": part,
                    "nonce": nonce,
                    "byte_start": b_start,
                    "byte_end": b_end,
                }
                for idx, ((b_start, b_end, part), nonce) in enumerate(zip(parts, leaf_nonces, strict=True))
            )
            return cls(source_id=source_id, chunk_mode=mode_name, entries=entries, span_mode=None)

        if nonces is not None:
            raise ValueError("top-level nonces cannot be passed when span_mode is enabled")
        span_mode_name = span_mode if isinstance(span_mode, str) else getattr(span_mode, "__name__", "custom")
        compound_entries: list[dict[str, Any]] = []
        for idx, (b_start, b_end, part) in enumerate(parts):
            sub_spans = split_spans_with_offsets(part, b_start, span_mode=span_mode)
            if not sub_spans:
                sub_spans = [(b_start, b_end, part)]
            span_dicts = [
                {
                    "span_index": s_idx,
                    "text": s_text,
                    "nonce": new_nonce(),
                    "byte_start": s_start,
                    "byte_end": s_end,
                }
                for s_idx, (s_start, s_end, s_text) in enumerate(sub_spans)
            ]
            compound_entries.append(
                {
                    "index": idx,
                    "byte_start": b_start,
                    "byte_end": b_end,
                    "spans": span_dicts,
                }
            )
        return cls(
            source_id=source_id,
            chunk_mode=mode_name,
            entries=tuple(compound_entries),
            span_mode=span_mode_name,
        )

    def hashes(self) -> list[str]:
        return [_entry_digest(entry) for entry in self.entries]

    def root(self) -> str:
        return merkle_root(self.hashes())

    def proof_for(self, index: int) -> tuple[tuple[str, str], ...]:
        return tuple(inclusion_proof(self.hashes(), index))

    def span_hashes_for(self, index: int) -> list[str] | None:
        if not (0 <= index < len(self.entries)):
            raise IndexError(f"leaf index {index} out of range for {len(self.entries)} leaves")
        entry = self.entries[index]
        if entry.get("redacted") or "spans" not in entry:
            return None
        return _leaf_hashes_for_entry(entry)

    def span_proof_for(self, leaf_index: int, span_index: int) -> tuple[tuple[str, str], ...]:
        sh = self.span_hashes_for(leaf_index)
        if sh is None:
            raise ValueError(f"leaf {leaf_index} is flat or redacted; no sub-leaf spans")
        return tuple(inclusion_proof(sh, span_index))

    def redact(self, predicate: Callable[[str], bool]) -> "RedactableDoc":
        """Redact every surviving leaf where `predicate(text)` returns True."""
        indices = {
            entry["index"]
            for entry in self.entries
            if not entry.get("redacted") and predicate(_entry_text(entry))
        }
        return self.redact_indices(indices)

    def redact_indices(self, indices: set[int]) -> "RedactableDoc":
        """Redact exact leaf indices: destroy plaintext/spans/nonce, keep salted leaf hash."""
        out: list[dict[str, Any]] = []
        for entry in self.entries:
            if not entry.get("redacted") and entry["index"] in indices:
                digest = _entry_digest(entry)
                out.append(
                    {
                        "index": entry["index"],
                        "hash": digest,
                        "byte_start": entry["byte_start"],
                        "byte_end": entry["byte_end"],
                        "redacted": True,
                    }
                )
            else:
                out.append(_clone_entry(entry))
        return RedactableDoc(
            source_id=self.source_id,
            chunk_mode=self.chunk_mode,
            entries=tuple(out),
            span_mode=self.span_mode,
        )

    def redact_leaf_spans(self, leaf_spans: Mapping[int, Iterable[int]]) -> "RedactableDoc":
        """Redact specific sub-leaf `span_index` entries while keeping bystander spans intact."""
        normalized: dict[int, set[int]] = {
            int(leaf_idx): {int(si) for si in span_idxs}
            for leaf_idx, span_idxs in leaf_spans.items()
        }
        out: list[dict[str, Any]] = []
        for entry in self.entries:
            l_idx = int(entry["index"])
            target_spans = normalized.get(l_idx)
            if not target_spans or entry.get("redacted"):
                out.append(_clone_entry(entry))
                continue
            if "spans" not in entry:
                digest = _entry_digest(entry)
                out.append(
                    {
                        "index": l_idx,
                        "hash": digest,
                        "byte_start": entry["byte_start"],
                        "byte_end": entry["byte_end"],
                        "redacted": True,
                    }
                )
                continue
            new_spans: list[dict[str, Any]] = []
            for sp in entry["spans"]:
                if not sp.get("redacted") and int(sp["span_index"]) in target_spans:
                    new_spans.append(
                        {
                            "span_index": sp["span_index"],
                            "hash": leaf_hash(sp["text"], sp["nonce"]),
                            "byte_start": sp["byte_start"],
                            "byte_end": sp["byte_end"],
                            "redacted": True,
                        }
                    )
                else:
                    new_spans.append(dict(sp))
            cloned = dict(entry)
            cloned["spans"] = new_spans
            out.append(cloned)
        return RedactableDoc(
            source_id=self.source_id,
            chunk_mode=self.chunk_mode,
            entries=tuple(out),
            span_mode=self.span_mode,
        )

    def redact_spans(self, predicate: Callable[[str], bool]) -> "RedactableDoc":
        """Redact individual sub-leaf spans where `predicate(span_text)` is True (v1 leaves match whole leaf)."""
        whole_indices: set[int] = set()
        span_targets: dict[int, set[int]] = {}
        for entry in self.entries:
            if entry.get("redacted"):
                continue
            if "spans" not in entry:
                if predicate(entry["text"]):
                    whole_indices.add(int(entry["index"]))
            else:
                hits = {
                    int(sp["span_index"])
                    for sp in entry["spans"]
                    if not sp.get("redacted") and predicate(sp["text"])
                }
                if hits:
                    span_targets[int(entry["index"])] = hits
        doc = self.redact_indices(whole_indices) if whole_indices else self
        return doc.redact_leaf_spans(span_targets) if span_targets else doc

    def redact_matching_spans(self, markers: Sequence[str]) -> "RedactableDoc":
        """Redact only the sub-leaf spans matching any boundary-aware marker (supports multi-word markers)."""
        clean_markers = [m.strip() for m in markers if isinstance(m, str) and m.strip()]
        if not clean_markers:
            return self
        word_char = re.compile(r"[^\W_]", re.UNICODE)
        patterns = [
            re.compile(
                (r"(?<![^\W_])" if word_char.match(m[0]) else "")
                + re.escape(m)
                + (r"(?![^\W_])" if word_char.match(m[-1]) else ""),
                flags=re.IGNORECASE | re.UNICODE,
            )
            for m in clean_markers
        ]
        whole_indices: set[int] = set()
        span_targets: dict[int, set[int]] = {}
        for entry in self.entries:
            if entry.get("redacted"):
                continue
            if "spans" not in entry:
                if any(pat.search(entry["text"]) for pat in patterns):
                    whole_indices.add(int(entry["index"]))
                continue

            # Track exact character offsets of each span inside `_render_spans_text` so multi-word
            # markers like "Priya Raman" map precisely to the constituent spans [0, 1].
            spans = entry["spans"]
            rendered_parts: list[str] = []
            span_char_ranges: list[tuple[int, int, int]] = []
            cursor = 0
            prev_end: int | None = None
            prev_red = False
            for sp in spans:
                is_red = bool(sp.get("redacted"))
                b_start = sp.get("byte_start")
                b_end = sp.get("byte_end")
                if is_red:
                    if not prev_red:
                        if rendered_parts:
                            rendered_parts.append(" ")
                            cursor += 1
                        rendered_parts.append("[redacted]")
                        cursor += len("[redacted]")
                    prev_red = True
                    prev_end = b_end if _is_strict_int(b_end) else None
                    continue
                piece = str(sp["text"])
                if rendered_parts:
                    need_space = True
                    if _is_strict_int(b_start) and prev_end is not None and b_start <= prev_end and not prev_red:
                        need_space = False
                    elif (
                        _is_strict_int(b_start)
                        and prev_end is not None
                        and b_start <= prev_end
                        and piece
                        and not piece[0].isalnum()
                    ):
                        need_space = False
                    if need_space:
                        rendered_parts.append(" ")
                        cursor += 1
                s_cstart = cursor
                rendered_parts.append(piece)
                cursor += len(piece)
                span_char_ranges.append((s_cstart, cursor, int(sp["span_index"])))
                prev_red = False
                prev_end = b_end if _is_strict_int(b_end) else None

            rendered = "".join(rendered_parts)
            matched_spans: set[int] = set()
            for pat in patterns:
                for match in pat.finditer(rendered):
                    m_start, m_end = match.span()
                    for c_start, c_end, sp_idx in span_char_ranges:
                        if c_start < m_end and m_start < c_end:
                            matched_spans.add(sp_idx)
            if matched_spans:
                span_targets[int(entry["index"])] = matched_spans

        doc = self.redact_indices(whole_indices) if whole_indices else self
        return doc.redact_leaf_spans(span_targets) if span_targets else doc

    def surviving_indices(self) -> list[int]:
        return [entry["index"] for entry in self.entries if not entry.get("redacted")]

    def text(self, redaction_marker: str = "[redacted]") -> str:
        """Render surviving text with redacted leaves/spans replaced by `redaction_marker`."""
        return " ".join(_entry_text(entry, redaction_marker=redaction_marker) for entry in self.entries)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source_id": self.source_id,
            "chunk_mode": self.chunk_mode,
            "merkle_root": self.root(),
            "entries": [_clone_entry(entry) for entry in self.entries],
        }
        if self.span_mode is not None:
            out["span_mode"] = self.span_mode
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedactableDoc":
        if redactable_doc_hashes(data) is None:
            raise ValueError("invalid or corrupted RedactableDoc payload")
        raw_sm = data.get("span_mode")
        return cls(
            source_id=str(data.get("source_id", "")),
            chunk_mode=str(data.get("chunk_mode", "sentence")),
            entries=tuple(_clone_entry(entry) for entry in data.get("entries", [])),
            span_mode=str(raw_sm) if isinstance(raw_sm, str) and raw_sm else None,
        )


__all__ = [
    "ChunkMode",
    "ChunkerFn",
    "RedactableDoc",
    "SpanChunkerFn",
    "SpanMode",
    "inclusion_proof",
    "interior_hash",
    "leaf_hash",
    "merkle_root",
    "new_nonce",
    "split_parts",
    "split_parts_with_spans",
    "split_spans_with_offsets",
    "verify_inclusion",
]
