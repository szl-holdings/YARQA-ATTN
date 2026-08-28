# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""API edges. No benchmarks."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.kernels_ci
def test_rejects_n_canals_gt_seq(kernel_mod, cpu_device):
    q = k = v = torch.randn(1, 1, 4, 8, device=cpu_device)
    with pytest.raises(ValueError, match="cannot exceed"):
        kernel_mod.yarqa_attn(q, k, v, 8)


@pytest.mark.kernels_ci
def test_rejects_n_canals_zero(kernel_mod, cpu_device):
    q = k = v = torch.randn(1, 1, 4, 8, device=cpu_device)
    with pytest.raises(ValueError, match="n_canals"):
        kernel_mod.yarqa_attn(q, k, v, 0)


@pytest.mark.kernels_ci
def test_rejects_mismatched_shapes(kernel_mod, cpu_device):
    q = torch.randn(1, 1, 4, 8, device=cpu_device)
    k = torch.randn(1, 1, 8, 8, device=cpu_device)
    v = torch.randn(1, 1, 8, 8, device=cpu_device)
    with pytest.raises(ValueError, match="shape"):
        kernel_mod.yarqa_attn(q, k, v, 2)


@pytest.mark.kernels_ci
def test_rejects_rank_3(kernel_mod, cpu_device):
    q = k = v = torch.randn(2, 4, 8, device=cpu_device)
    with pytest.raises(ValueError, match="rank-4"):
        kernel_mod.yarqa_attn(q, k, v, 2)


@pytest.mark.kernels_ci
def test_shape_preserved(kernel_mod, cpu_device):
    q = k = v = torch.randn(2, 3, 12, 16, device=cpu_device)
    y = kernel_mod.yarqa_attn(q, k, v, 3)
    assert y.shape == q.shape
    assert y.dtype == q.dtype
    assert y.device == q.device
