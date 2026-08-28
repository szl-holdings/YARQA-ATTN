# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Kernel-builder layout checks. These are not performance tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_NAME_RE = re.compile(r"^[a-z][-a-z0-9]*[a-z0-9]$")


@pytest.mark.kernels_ci
def test_build_toml_edition_5_torch_noarch():
    text = (_ROOT / "build.toml").read_text(encoding="utf-8")
    assert "edition = 5" in text
    assert "[torch-noarch]" in text
    assert "[kernel." not in text
    assert 'name = "yarqa-attn"' in text
    assert _NAME_RE.match("yarqa-attn")
    assigned_triton = [
        line
        for line in text.splitlines()
        if "triton" in line.lower()
        and not line.lstrip().startswith("#")
        and "=" in line
    ]
    assert assigned_triton == []
    assert 'repo-id = "SZLHOLDINGS/YARQA-ATTN"' in text
    assert 'backends = ["cpu"]' in text


@pytest.mark.kernels_ci
def test_no_benchmarks_directory():
    """Honesty: this package does not ship fabricated benches."""
    assert not (_ROOT / "benchmarks").exists()


@pytest.mark.kernels_ci
def test_required_kernel_builder_files():
    for rel in (
        "build.toml",
        "CARD.md",
        "LICENSE",
        "README.md",
        "flake.nix",
        "torch-ext/yarqa_attn/__init__.py",
        "torch-ext/yarqa_attn/attn.py",
        "torch-ext/yarqa_attn/_chain.py",
        "tests/test_yarqa_attn.py",
    ):
        assert (_ROOT / rel).is_file(), rel


@pytest.mark.kernels_ci
def test_ops_py_not_authored():
    """kernel-builder generates _ops.py. Do not ship a handwritten one."""
    assert not (_ROOT / "torch-ext/yarqa_attn/_ops.py").exists()


@pytest.mark.kernels_ci
def test_flake_pytest_is_check_input_only():
    text = (_ROOT / "flake.nix").read_text(encoding="utf-8")
    assert "pythonCheckInputs" in text
    assert "pythonBuildInputs =" not in text
