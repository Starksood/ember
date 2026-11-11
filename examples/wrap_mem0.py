"""25-line drop-in example: wrap a Mem0 Memory instance with cryptographic custody & cascade erasure.

Run directly with zero external API keys (uses an in-process Mem0 v3-compatible stub),
or swap `DemoMem0Client()` for `from mem0 import Memory; client = Memory()`.
"""

from __future__ import annotations

import uuid

from ember import CustodyVault, Mem0Adapter, TombstonedSubjectError


class DemoMem0Client:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def add(self, messages, user_id=None, metadata=None, infer=False):
        mid = f"mem0_{uuid.uuid4().hex[:8]}"
        self.rows[mid] = {"id": mid, "memory": str(messages), "user_id": user_id, "metadata": metadata or {}}
        return {"results": [{"id": mid, "memory": str(messages), "event": "ADD"}]}

    def search(self, query, filters=None, limit=5):
        q = set(query.lower().split())
        hits = [
            {"id": r["id"], "memory": r["memory"], "score": len(q & set(r["memory"].lower().split())), "metadata": r["metadata"]}
            for r in self.rows.values()
            if (not filters or r.get("user_id") == filters.get("user_id"))
            and (q & set(r["memory"].lower().split()))
        ]
        return {"results": sorted(hits, key=lambda h: -h["score"])[:limit]}

    def delete(self, memory_id: str) -> None:
        self.rows.pop(memory_id, None)


def main() -> None:
    mem0 = DemoMem0Client()  # Or: from mem0 import Memory; mem0 = Memory()
    vault = CustodyVault.ephemeral(adapter=Mem0Adapter(mem0, default_user_id="team_acme"))

    vault.register_source(
        "intake_notes",
        "Priya Raman works in Portland on triage routing. Dana Kim leads pharmacy inventory.",
    )
    r_priya = vault.remember(
        "Priya Raman works in Portland on triage routing.",
        subjects=["priya"],
        source_id="intake_notes",
        quote="Priya Raman works in Portland on triage routing.",
        metadata={"user_id": "priya"},
    )
    r_dana = vault.remember(
        "Dana Kim leads pharmacy inventory.",
        subjects=["dana"],
        source_id="intake_notes",
        quote="Dana Kim leads pharmacy inventory.",
        metadata={"user_id": "dana"},
    )
    # Derived summary stored under team workspace bucket — naive delete_all(user_id="priya") misses this!
    vault.remember(
        "Riverside handoff: triage routing pairs with pharmacy inventory.",
        subjects=["team_acme"],
        derived_from=[r_priya.item.item_id, r_dana.item.item_id],
        metadata={"user_id": "team_acme"},
    )

    cert = vault.erase("priya", probes=["Where does Priya work?", "Who handles triage routing?"])
    print("Erased:", cert.subject_id)
    print("Cascade adapter IDs deleted:", cert.closure["adapter_record_ids"])
    print("Bystander search (Dana):", vault.search("Dana pharmacy").hits[0].text)
    print("Bystander receipt still valid?", vault.verify_item_receipt(r_dana.item.item_id))

    try:
        vault.remember("Priya Raman works in Portland.", subjects=["priya"])
    except TombstonedSubjectError as exc:
        print("Re-ingest blocked by tombstone:", exc)

    vault.close()


if __name__ == "__main__":
    main()
