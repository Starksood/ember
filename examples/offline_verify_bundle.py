"""Export an Ember verification bundle and verify it with the stdlib-only verifier."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from ember import CustodyVault
from ember.verify import verify_bundle


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vault = CustodyVault.open(root / "vault", key_dir=root / "keys")
        vault.register_source(
            "notes",
            "Priya lives in Portland. Dana lives in Seattle.",
            origin="support_ticket_42",
            origin_kind="file",
        )
        vault.remember("Priya lives in Portland.", subjects=["priya"], source_id="notes", quote="Priya lives in Portland.")
        vault.remember("Dana lives in Seattle.", subjects=["dana"], source_id="notes", quote="Dana lives in Seattle.")
        vault.erase("priya", probes=["Where does Priya live?"])

        bundle = vault.export_bundle()
        vault.close()

        bundle_path = root / "bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        report = verify_bundle(json.loads(bundle_path.read_text(encoding="utf-8")), require_signature_check=True)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
