# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""This repository's kernel publication config, checked offline against HEAD.

The generic publisher is tested in test_szl_kernel_publish.py. These tests pin what
hf-kernel-publish would put on kernels/SZLHOLDINGS/YARQA-ATTN and on the model twin
for the commit under test: the package layout, the card, and what the prune lists
may and may not remove.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "szl_kernel_publish.py"
REPO_ID = "SZLHOLDINGS/YARQA-ATTN"
PACKAGE_FILES = ("__init__.py", "_chain.py", "attn.py")


def _publisher():
    name = "szl_kernel_publish_config_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_config_names_the_kernel_repo_and_package():
    kp = _publisher()
    (cfg,) = kp.load_config(ROOT)
    assert cfg.repo_id == REPO_ID
    assert cfg.name == "yarqa-attn"  # kernels KernelName is lowercase
    assert cfg.version == 1
    assert cfg.source_repository == "szl-holdings/YARQA-ATTN"
    assert [(p.src, p.name) for p in cfg.packages] == [("torch-ext/yarqa_attn", "yarqa_attn")]
    assert cfg.card == "CARD.md" and cfg.license_file == "LICENSE"
    assert cfg.twin is not None and cfg.twin.repo_id == REPO_ID


def test_payload_at_head_is_the_torch_ext_package():
    kp = _publisher()
    cfg = kp.find_kernel(ROOT, REPO_ID)
    head = _head()
    payload = kp.build_payload(cfg, ROOT, head)
    expected_build = {f"build/{v}/yarqa_attn/{f}" for v in kp.VARIANTS for f in PACKAGE_FILES}
    expected_build |= {f"build/{v}/metadata.json" for v in kp.VARIANTS}
    assert {p for p in payload if p.startswith("build/")} == expected_build
    for name in PACKAGE_FILES:
        blob = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", f"{head}:torch-ext/yarqa_attn/{name}"],
                              check=True, capture_output=True).stdout.replace(b"\r\n", b"\n")
        for variant in kp.VARIANTS:
            assert payload[f"build/{variant}/yarqa_attn/{name}"] == blob
    meta = json.loads(payload["build/torch-cpu/metadata.json"])
    assert meta["name"] == "yarqa-attn" and meta["version"] == 1 and meta["license"] == "Apache-2.0"
    assert set(payload) - expected_build == {"README.md", "LICENSE", "NOTICE", kp.BINDING_PATH}
    card = payload["README.md"].decode("utf-8")
    assert f'get_kernel("{REPO_ID}", version=1, trust_remote_code=True)' in card
    assert "REPLACE_WITH" not in card and 'revision="main"' not in card
    assert "yarqa_attn.npz" in card and "SYNTHETIC" in card


def test_prune_lists_keep_receipts_and_disclosed_weights():
    kp = _publisher()
    cfg = kp.find_kernel(ROOT, REPO_ID)
    assert set(cfg.prune_paths) == {"OPERATIONAL.json", "BENCH.laptop-blackwell.json"}
    for kept in ("yarqa_attn.npz", "TRAINING_RECEIPT.json", "bom/model-bom.cdx.json", "LICENSE",
                 "og-card.png", ".gitattributes", "README.md"):
        assert not kp.is_pruned(kept, cfg.twin.prune_paths), kept
    for gone in ("torch-ext/yarqa_attn/attn.py", "yarqa.py", "tests/conftest.py", "OPERATIONAL.json", "CARD.md"):
        assert kp.is_pruned(gone, cfg.twin.prune_paths), gone


def test_twin_pointer_card_passes_the_publisher_lint():
    kp = _publisher()
    cfg = kp.find_kernel(ROOT, REPO_ID)
    text = kp.render_card(cfg, (ROOT / cfg.twin.card).read_text(encoding="utf-8"), cfg.twin.card)
    kp.lint_card(cfg, text, cfg.twin.card, require_usage=True)
    assert f"https://huggingface.co/kernels/{REPO_ID}" in text
    assert "yarqa_attn.npz" in text and "SYNTHETIC" in text


def test_no_floating_main_loads_in_source():
    for rel in ("README.md", "CARD.md", "tests/conftest.py", "tests/test_local_kernels.py", "example.py"):
        assert 'revision="main"' not in (ROOT / rel).read_text(encoding="utf-8"), rel
