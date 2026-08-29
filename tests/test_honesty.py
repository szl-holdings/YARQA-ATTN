# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Honesty: KERNEL voice, no cloned stacks, no fabricated LIVE/speed stamps."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.kernels_ci

BANNED = (
    "flash_attn",
    "sageattn",
    "sage_attn",
    "vllm",
    "cutlass",
    "cute",
    "torch.nn.attention.flex_attention",
)

_ROOT = Path(__file__).resolve().parents[1]


def test_selfcheck_no_speedup_claim(kernel_mod):
    report = kernel_mod.selfcheck()
    assert report["ok"] is True
    assert "Conjecture 1" in str(report.get("lambda", ""))
    note = str(report.get("note", "")).lower()
    assert "no speedup" in note
    assert report.get("python") == "present"
    assert "import-LIVE" not in str(report.get("status", ""))


def test_no_vendored_attention_imports():
    root = _ROOT / "torch-ext" / "yarqa_attn"
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for banned in BANNED:
                        assert not alias.name.startswith(banned), path.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for banned in BANNED:
                    assert not node.module.startswith(banned), path.name


def test_output_digest_does_not_call_numpy():
    path = _ROOT / "torch-ext" / "yarqa_attn" / "attn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "numpy":
            raise AssertionError("attn.py must not call Tensor.numpy (CPU CI has no numpy)")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "numpy"
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("numpy")


def test_readme_kernel_voice_no_metrics():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8").lower()
    card = (_ROOT / "CARD.md").read_text(encoding="utf-8").lower()
    blob = readme + "\n" + card
    assert "owner" in readme and "kernel" in readme
    assert "not a" in readme and "weight" in readme
    # Disclaimers are KERNEL voice. Fabricated numeric rates are not.
    assert "not claimed" in readme
    assert "software/kernel" in readme or "software" in readme
    assert not re.search(r"\d[\d.,]*\s*(tokens/s|tok/s|joules|j/token)", blob)
    # MEASURED import-LIVE (method/N/date) is allowed. Bare YAML flags without
    # MEASURED evidence are not a substitute for a timed Hub load.
    if "import-live" in blob or "import_live: true" in blob:
        assert "measured" in blob
    assert "python" in readme and "present" in readme
    assert "szl-receipt-attn" in readme
    assert "szl-maskmod" in readme
    assert "szl-block-kv" in readme
