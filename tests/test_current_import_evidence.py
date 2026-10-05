from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "evidence" / "yarqa-import-20261005.json"
SOURCE_REVISION = "819b904f304a054cf78446aa43a5cb85e5149c4b"
KERNEL_REVISION = "4e0828e517e84377d0024ab3f9d0cdab7d31520d"
POSTMERGE_RECEIPT = ROOT / "evidence" / "yarqa-postmerge-import-20261005.json"
POSTMERGE_SOURCE_REVISION = "cc895b47e7a3d1e7d5f246812a573dfd1205b103"
MODEL_REVISION = "d6aeb4f24655c07acc155eed7acade5ad249724f"
WORKFLOWS = (
    ("cpu-tests.yml", "pytest-cpu"),
    ("chain-drift-guard.yml", "compat"),
    ("hf-card.yml", "card"),
)


def test_current_import_receipt_is_exact_and_bounded() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "szl.yarqa-immutable-import-receipt/v1"
    assert receipt["state"] == "MEASURED_IMPORT_PASS_BYTES_ALIGNED_SOURCE_BINDING_UNAVAILABLE"
    assert receipt["source"]["revision"] == SOURCE_REVISION
    assert receipt["provider"]["kernel_revision"] == KERNEL_REVISION
    assert receipt["provider"]["source_binding_file"] == "UNAVAILABLE"
    assert receipt["runtime"]["kernels"] == "0.16.1"
    assert receipt["runtime"]["device"] == "cpu"
    assert receipt["runtime"]["paid_compute"] is False
    assert receipt["selfcheck"]["ok"] is True
    assert receipt["selfcheck"]["chain_ok"] is True
    assert all(row["equal"] for row in receipt["package_parity"].values())
    assert receipt["claims"]["performance"] == "NOT_CLAIMED"
    assert receipt["claims"]["production_readiness"] == "NOT_CLAIMED"
    assert receipt["secret_values_recorded"] is False


def test_postmerge_receipt_uses_canonical_git_object_bytes() -> None:
    receipt = json.loads(POSTMERGE_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "szl.yarqa-postmerge-import-receipt/v1"
    assert receipt["state"] == (
        "MEASURED_CURRENT_MAIN_IMPORT_PASS_PACKAGE_ALIGNED_"
        "SOURCE_BINDING_UNAVAILABLE_HUB_CARDS_SOURCE_AHEAD"
    )
    assert receipt["source"]["revision"] == POSTMERGE_SOURCE_REVISION
    assert receipt["provider"]["kernel_revision"] == KERNEL_REVISION
    assert receipt["provider"]["model_revision"] == MODEL_REVISION
    assert receipt["provider"]["kernel_source_binding_file"] == "UNAVAILABLE"
    assert receipt["provider"]["model_source_binding_file"] == "UNAVAILABLE"
    assert receipt["card_projection"]["classification"] == "SOURCE_AHEAD"
    assert receipt["card_projection"]["kernel_matches_github_card"] is False
    assert receipt["card_projection"]["model_matches_github_card"] is False
    assert "canonical Git object bytes" in receipt["measurement_method"]
    assert receipt["supersedes"]["path"] == "evidence/yarqa-import-20261005.json"

    for row in receipt["package_parity"].values():
        canonical = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{POSTMERGE_SOURCE_REVISION}:{row['github_path']}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(canonical).hexdigest()
        assert row["github_sha256"] == digest
        assert row["all_variants_equal"] is True
        assert set(row["variants"]) == {"torch-cpu", "torch-universal"}
        assert all(
            variant["equal_to_github_source"] is True
            and variant["remote_sha256"] == digest
            for variant in row["variants"].values()
        )

    assert receipt["runtime"]["kernels"] == "0.16.1"
    assert receipt["runtime"]["device"] == "cpu"
    assert receipt["runtime"]["paid_compute"] is False
    assert receipt["selfcheck"]["ok"] is True
    assert receipt["claims"]["performance"] == "NOT_CLAIMED"
    assert receipt["claims"]["production_readiness"] == "NOT_CLAIMED"
    assert receipt["cost_budget_usd"] == 0
    assert receipt["secret_values_recorded"] is False


def test_readme_pins_the_measured_provider_revision() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert KERNEL_REVISION in readme
    assert "evidence/yarqa-import-20261005.json" in readme
    assert 'revision="main"' not in readme
    assert "source binding remains" in readme
    assert "production-readiness claim" in readme


def test_source_qualifying_workflows_use_owner_head_not_synthetic_merge() -> None:
    for name, job in WORKFLOWS:
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert f"  {job}:" in text
        assert "SOURCE_REVISION: ${{ github.event.pull_request.head.sha || github.sha }}" in text
        assert "ref: ${{ env.SOURCE_REVISION }}" in text
        assert "persist-credentials: false" in text
        assert "fetch-depth: 1" in text
        assert 'test "$(git rev-parse HEAD)" = "$SOURCE_REVISION"' in text
        assert "refs/pull" not in text
        assert "merge_commit_sha" not in text
        assert "pull_request_target" not in text
