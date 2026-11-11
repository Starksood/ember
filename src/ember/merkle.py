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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
import secrets
from typing import Any, Literal

from .verify import interior_hash, leaf_hash, merkle_root, redactable_doc_hashes, verify_inclusion

ChunkMode = Literal["sentence", "line", "whole"]
ChunkerFn = Callable[[str], list[tuple[int, int, str]]]

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


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
        return mode(text)

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


@dataclass(frozen=True)
class RedactableDoc:
    """Immutable Merkle-indexed document supporting lawful leaf redaction."""

    source_id: str
    chunk_mode: str
    entries: tuple[dict[str, Any], ...]

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source_id: str = "",
        mode: ChunkMode | ChunkerFn = "sentence",
        nonces: Sequence[str] | None = None,
    ) -> "RedactableDoc":
        parts = split_parts_with_spans(text, mode=mode)
        if nonces is None:
            leaf_nonces = [new_nonce() for _ in parts]
        else:
            if len(nonces) != len(parts):
                raise ValueError(f"expected {len(parts)} nonces, got {len(nonces)}")
            leaf_nonces = list(nonces)
        entries = tuple(
            {
                "index": idx,
                "text": part,
                "nonce": nonce,
                "byte_start": b_start,
                "byte_end": b_end,
            }
            for idx, ((b_start, b_end, part), nonce) in enumerate(zip(parts, leaf_nonces))
        )
        mode_name = mode if isinstance(mode, str) else getattr(mode, "__name__", "custom")
        return cls(source_id=source_id, chunk_mode=mode_name, entries=entries)

    def hashes(self) -> list[str]:
        out: list[str] = []
        for entry in self.entries:
            if entry.get("redacted"):
                out.append(entry["hash"])
            else:
                out.append(leaf_hash(entry["text"], entry["nonce"]))
        return out

    def root(self) -> str:
        return merkle_root(self.hashes())

    def proof_for(self, index: int) -> tuple[tuple[str, str], ...]:
        return tuple(inclusion_proof(self.hashes(), index))

    def redact(self, predicate: Callable[[str], bool]) -> "RedactableDoc":
        """Redact every surviving leaf where `predicate(text)` returns True."""
        indices = {
            entry["index"]
            for entry in self.entries
            if not entry.get("redacted") and predicate(entry["text"])
        }
        return self.redact_indices(indices)

    def redact_indices(self, indices: set[int]) -> "RedactableDoc":
        """Redact exact leaf indices: destroy plaintext and nonce, keep salted leaf hash."""
        out: list[dict[str, Any]] = []
        for entry in self.entries:
            if not entry.get("redacted") and entry["index"] in indices:
                digest = leaf_hash(entry["text"], entry["nonce"])
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
                out.append(dict(entry))
        return RedactableDoc(source_id=self.source_id, chunk_mode=self.chunk_mode, entries=tuple(out))

    def surviving_indices(self) -> list[int]:
        return [entry["index"] for entry in self.entries if not entry.get("redacted")]

    def text(self, redaction_marker: str = "[redacted]") -> str:
        """Render surviving text with redacted leaves replaced by `redaction_marker`."""
        return " ".join(
            redaction_marker if entry.get("redacted") else entry["text"]
            for entry in self.entries
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_mode": self.chunk_mode,
            "merkle_root": self.root(),
            "entries": [dict(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedactableDoc":
        if redactable_doc_hashes(data) is None:
            raise ValueError("invalid or corrupted RedactableDoc payload")
        return cls(
            source_id=str(data.get("source_id", "")),
            chunk_mode=str(data.get("chunk_mode", "sentence")),
            entries=tuple(dict(entry) for entry in data.get("entries", [])),
        )


__all__ = [
    "ChunkMode",
    "ChunkerFn",
    "RedactableDoc",
    "inclusion_proof",
    "interior_hash",
    "leaf_hash",
    "merkle_root",
    "new_nonce",
    "split_parts",
    "split_parts_with_spans",
    "verify_inclusion",
]
