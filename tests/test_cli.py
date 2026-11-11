"""End-to-end CLI workflow tests (`init -> add-source -> remember -> search -> erase -> export -> verify -> benchmark`)."""

from __future__ import annotations

import json

from ember import cli


def test_cli_end_to_end_custody_lifecycle(tmp_path, capsys):
    vault_dir = tmp_path / "vault"
    key_dir = tmp_path / "keys"
    bundle_file = tmp_path / "bundle.json"

    base = ["--vault", str(vault_dir), "--key-dir", str(key_dir)]

    assert cli.main([*base, "init"]) == 0
    init_out = json.loads(capsys.readouterr().out)
    assert init_out["keys_separated_from_ledger"] is True

    assert (
        cli.main(
            [
                *base,
                "add-source",
                "doc1",
                "--text",
                "Priya lives in Portland. Marcus lives in Leeds.",
                "--origin",
                "hr_csv",
                "--origin-kind",
                "file",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        cli.main(
            [
                *base,
                "remember",
                "Priya lives in Portland.",
                "--subject",
                "priya",
                "--source-id",
                "doc1",
                "--quote",
                "Priya lives in Portland.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        cli.main(
            [
                *base,
                "remember",
                "Marcus lives in Leeds.",
                "--subject",
                "marcus",
                "--source-id",
                "doc1",
                "--quote",
                "Marcus lives in Leeds.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert cli.main([*base, "erase", "priya", "--probe", "Where does Priya live?"]) == 0
    cert_out = json.loads(capsys.readouterr().out)
    assert cert_out["key_destroyed"] is True

    assert cli.main([*base, "search", "Where does Priya live?"]) == 0
    assert json.loads(capsys.readouterr().out)["abstained"] is True

    assert cli.main([*base, "search", "Where does Marcus live?"]) == 0
    marcus_res = json.loads(capsys.readouterr().out)
    assert not marcus_res["abstained"]
    assert marcus_res["hits"][0]["receipt_verified"] is True

    assert cli.main([*base, "export", "--out", str(bundle_file)]) == 0
    capsys.readouterr()

    assert cli.main(["verify", str(bundle_file), "--require-signature"]) == 0
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["ok"] is True


def test_cli_benchmark_command(capsys):
    assert cli.main(["benchmark"]) == 0
    text_out = capsys.readouterr().out
    assert "InMemoryStoreAdapter" in text_out
    assert "keyword_overlap" in text_out
    assert "Embedding : none" in text_out

    assert cli.main(["benchmark", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True and report["ember_leaks"] == 0
    assert report["adapter"]["name"] == "InMemoryStoreAdapter"
    assert report["adapter"]["retrieval_mode"] == "keyword_overlap"
    assert report["adapter"]["embedding_model"] is None
