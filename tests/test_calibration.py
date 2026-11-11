"""Two-direction calibration controls & true stdlib-only offline verifier tests."""

from __future__ import annotations

import builtins
import json
import sys

import pytest

from ember.adapters import InMemoryStoreAdapter
from ember.audit import SUITE_V2, SubjectScenario, verify_suite_hygiene
from ember.vault import CustodyVault
import ember.verify as verify_mod


def test_negative_control_overdeletion_fails_bystander_gate():
    """Deleting everything trivially achieves 0 leaks, so bystander gates MUST fail it."""
    s = SUITE_V2[0]
    adapter = InMemoryStoreAdapter()
    vault = CustodyVault.ephemeral(adapter=adapter)
    vault.register_source("shared", " ".join([*s.facts, *s.bystander_facts]))
    r_sub = vault.remember(s.facts[0], subjects=[s.subject], source_id="shared", quote=s.facts[0])
    r_bys = vault.remember(s.bystander_facts[0], subjects=[s.peer], source_id="shared", quote=s.bystander_facts[0])

    # Simulate an over-zealous "erase everything" wipe
    vault.erase(s.subject, probes=s.attacks["direct"])
    adapter.delete(r_bys.item.adapter_record_ids)

    # Leaks = 0, yet bystander survival fails strictly!
    bystander_survived = all(not vault.search(bq).abstained for bq in s.bystander_queries)
    assert bystander_survived is False
    assert r_sub.item.item_id != r_bys.item.item_id
    vault.close()


def test_hygiene_gate_rejects_confounded_or_unexpressive_markers():
    bad_bystander_overlap = SubjectScenario(
        subject="Alex",
        peer="Sam",
        third="Lee",
        org="Acme",
        city="Austin",
        facts=("Alex lives in Austin.",),
        bystander_facts=("Sam also lives in Austin.",),
        bystander_queries=("Where does Sam live?",),
        derived_summary="Summary",
        leak_markers=("Alex", "Austin"),
        attacks={"direct": ("Where does Alex live?",)},
    )
    with pytest.raises(AssertionError, match="confounded"):
        verify_suite_hygiene((bad_bystander_overlap,))

    missing_marker = SubjectScenario(
        subject="Alex",
        peer="Sam",
        third="Lee",
        org="Acme",
        city="Boston",
        facts=("Alex is a nurse.",),
        bystander_facts=("Sam is a pharmacist.",),
        bystander_queries=("What does Sam do?",),
        derived_summary="Summary",
        leak_markers=("Alex", "Boston"),
        attacks={"direct": ("What does Alex do?",)},
    )
    with pytest.raises(AssertionError, match="never appears"):
        verify_suite_hygiene((missing_marker,))


def test_verify_py_runs_stdlib_only_even_when_cryptography_import_blocked(monkeypatch, tmp_path):
    """Prove bundle verification of documents, receipts, and ledger chain needs zero third-party deps."""
    vault = CustodyVault.ephemeral()
    vault.register_source(
        "transcript",
        "Priya works at Riverside. Marcus owns a bookshop in Leeds.",
    )
    vault.remember("Priya works at Riverside.", subjects=["priya"], source_id="transcript", quote="Priya works at Riverside.")
    vault.remember("Marcus owns a bookshop in Leeds.", subjects=["marcus"], source_id="transcript", quote="Marcus owns a bookshop in Leeds.")
    vault.erase("priya", probes=["Where does Priya work?"])
    bundle = vault.export_bundle()
    vault.close()

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    # Simulate an environment WITHOUT `cryptography` installed
    real_import = builtins.__import__

    def _no_crypto(name, *args, **kwargs):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("simulated no-cryptography stdlib environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_crypto)

    # Structural, Merkle redaction, bystander receipts, and ledger chain ALL pass stdlib-only!
    report = verify_mod.verify_bundle(bundle, require_signature_check=False)
    assert report["ok"] is True
    assert report["checks"]["documents"]["ok"] is True
    assert report["checks"]["receipts"]["ok"] is True
    assert report["checks"]["ledger_chain"]["ok"] is True
    assert report["checks"]["certificates"]["items"][0]["verified"] is None

    # Strict `--require-signature` refuses to silently pass when Ed25519 cannot be checked!
    strict_report = verify_mod.verify_bundle(bundle, require_signature_check=True)
    assert strict_report["ok"] is False
    assert verify_mod.main([str(bundle_path)]) == 0
    assert verify_mod.main([str(bundle_path), "--require-signature"]) == 1


def test_verify_py_executes_as_standalone_copied_file_without_ember_package(tmp_path):
    """Verify `verify.py` can be copied alone onto an auditor machine and executed directly."""
    import shutil
    import subprocess
    from pathlib import Path

    vault = CustodyVault.ephemeral()
    vault.register_source("s", "Priya works in Portland. Marcus works in Leeds.")
    vault.remember("Priya works in Portland.", subjects=["priya"], source_id="s", quote="Priya works in Portland.")
    vault.remember("Marcus works in Leeds.", subjects=["marcus"], source_id="s", quote="Marcus works in Leeds.")
    vault.erase("priya", probes=["Where does Priya work?"])
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(vault.export_bundle()), encoding="utf-8")
    vault.close()

    standalone_dir = tmp_path / "auditor_box"
    standalone_dir.mkdir()
    copied = standalone_dir / "verify.py"
    shutil.copy2(Path(verify_mod.__file__), copied)

    proc = subprocess.run(
        [sys.executable, str(copied), str(bundle_path)],
        cwd=standalone_dir,
        env={"PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ok"] is True
    assert out["checks"]["documents"]["ok"] is True
    assert out["checks"]["receipts"]["ok"] is True
    assert out["checks"]["ledger_chain"]["ok"] is True

