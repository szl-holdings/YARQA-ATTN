# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Receipt-chain tests: partition + output digests, tamper detect."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.kernels_ci


def test_receipt_partition_and_output(kernel_mod, cpu_device):
    torch.manual_seed(4)
    q = k = v = torch.randn(1, 1, 8, 16, device=cpu_device)
    chain = kernel_mod.ReceiptChain()
    kernel_mod.yarqa_attn(q, k, v, 2, chain=chain)
    ok, depth, brk = chain.verify()
    assert ok is True
    assert depth == 2
    assert brk == -1
    assert chain._rows[0]["op"] == "yarqa_partition"
    assert chain._rows[0]["bounds"] == [0, 4, 8]
    assert chain._rows[1]["op"] == "yarqa_output"
    assert isinstance(chain._rows[1]["output_digest"], str)
    assert len(chain._rows[1]["output_digest"]) == 64


def test_receipt_tamper_breaks(kernel_mod, cpu_device):
    q = k = v = torch.randn(1, 1, 4, 8, device=cpu_device)
    chain = kernel_mod.ReceiptChain()
    kernel_mod.yarqa_attn(q, k, v, 2, chain=chain)
    ok, depth, brk = chain.verify()
    assert ok is True and depth == 2 and brk == -1
    chain._rows[0]["bounds"] = [0, 1, 4]
    ok2, _, first = chain.verify()
    assert ok2 is False
    assert first == 0


def test_output_digest_changes_with_values(kernel_mod, cpu_device):
    torch.manual_seed(5)
    q = torch.randn(1, 1, 8, 8, device=cpu_device)
    k = torch.randn(1, 1, 8, 8, device=cpu_device)
    v_a = torch.randn(1, 1, 8, 8, device=cpu_device)
    v_b = v_a + 1.0
    a = kernel_mod.ReceiptChain()
    b = kernel_mod.ReceiptChain()
    kernel_mod.yarqa_attn(q, k, v_a, 2, chain=a)
    kernel_mod.yarqa_attn(q, k, v_b, 2, chain=b)
    assert a._rows[0]["digest"] == b._rows[0]["digest"]
    assert a._rows[1]["output_digest"] != b._rows[1]["output_digest"]


def test_partition_receipt_changes_with_canal_count(kernel_mod, cpu_device):
    q = k = v = torch.randn(1, 1, 8, 8, device=cpu_device)
    a = kernel_mod.ReceiptChain()
    b = kernel_mod.ReceiptChain()
    kernel_mod.yarqa_attn(q, k, v, 2, chain=a)
    kernel_mod.yarqa_attn(q, k, v, 4, chain=b)
    assert a._rows[0]["bounds"] != b._rows[0]["bounds"]
    assert a._rows[0]["digest"] != b._rows[0]["digest"]
