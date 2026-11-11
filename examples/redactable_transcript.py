"""Solve the Erasure / Receipt Trilemma on a shared multi-party source transcript."""

from __future__ import annotations

from ember import RedactableDoc, mint_redactable_receipt


def main() -> None:
    transcript = (
        "Priya Raman joined Acme in 2019 as a logistics analyst. "
        "Marcus Webb owns a bookshop in Leeds. "
        "Dana Kim has a cat named Pepper."
    )
    doc = RedactableDoc.from_text(transcript, source_id="hr_transcript_2026")
    marcus_receipt = mint_redactable_receipt(
        doc,
        "Marcus Webb owns a bookshop in Leeds.",
        raw_doc=transcript.encode("utf-8"),
    )

    redacted = doc.redact(lambda leaf: "Priya" in leaf)

    print("Original Merkle root :", doc.root())
    print("Redacted Merkle root :", redacted.root())
    print("Surviving text       :", redacted.text())
    print("Marcus receipt valid :", marcus_receipt.verify_redactable(redacted))


if __name__ == "__main__":
    main()
