"""Full 12-subject x 7-attack-class adversarial erasure suite (SUITE_V2)."""

from __future__ import annotations

from ember.audit import SUITE_V2, run_benchmark, verify_suite_hygiene


def test_suite_hygiene_passes_cleanly():
    summary = verify_suite_hygiene(SUITE_V2)
    assert summary["subjects"] == 12
    assert summary["probes"] == 204
    assert summary["unique_markers"] == 24


def test_full_adversarial_erasure_benchmark_beats_naive_baseline():
    report = run_benchmark(SUITE_V2)
    assert report["passed"] is True
    assert report["subjects"] == 12
    assert report["total_probes"] == 204
    # Naive user_id-only deletion leaks via derived workspace records on every single subject
    assert report["baseline_naive_leaks"] > 0
    # Ember achieves zero leaks while preserving bystander survival, proofs, and content-free replays
    assert report["ember_leaks"] == 0
    assert report["all_bystanders_survived"] is True
    assert report["all_bystander_receipts_valid"] is True
    assert report["all_redacted_sources_clean"] is True
    assert report["all_replays_content_free"] is True
    assert report["all_tombstones_blocked"] is True
    assert report["all_certificates_valid"] is True
