# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""CPU correctness: slice-and-attend vs naive within-compartment reference."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

_ATOL = 1.0e-5
_RTOL = 1.0e-5


def _naive_within_compartment(q, k, v, n_canals, compartment_mask):
    seq = q.shape[2]
    mask = compartment_mask(seq, n_canals, device=q.device)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )


@pytest.mark.kernels_ci
def test_matches_naive_compartment_cpu(kernel_mod, cpu_device):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 32, device=cpu_device)
    k = torch.randn(1, 2, 16, 32, device=cpu_device)
    v = torch.randn(1, 2, 16, 32, device=cpu_device)
    y = kernel_mod.yarqa_attn(q, k, v, 4)
    ref = _naive_within_compartment(q, k, v, 4, kernel_mod.compartment_mask)
    torch.testing.assert_close(y, ref, atol=_ATOL, rtol=_RTOL)


@pytest.mark.kernels_ci
def test_one_canal_matches_full_sdpa(kernel_mod, cpu_device):
    torch.manual_seed(1)
    q = torch.randn(2, 3, 8, 16, device=cpu_device)
    k = torch.randn(2, 3, 8, 16, device=cpu_device)
    v = torch.randn(2, 3, 8, 16, device=cpu_device)
    y = kernel_mod.yarqa_attn(q, k, v, 1)
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    torch.testing.assert_close(y, ref, atol=_ATOL, rtol=_RTOL)


@pytest.mark.kernels_ci
def test_n_canals_gt_1_actually_splits(kernel_mod, cpu_device):
    torch.manual_seed(2)
    q = torch.randn(1, 2, 16, 32, device=cpu_device)
    k = torch.randn(1, 2, 16, 32, device=cpu_device)
    v = torch.randn(1, 2, 16, 32, device=cpu_device)
    y4 = kernel_mod.yarqa_attn(q, k, v, 4)
    y1 = kernel_mod.yarqa_attn(q, k, v, 1)
    assert not torch.allclose(y4, y1, atol=_ATOL, rtol=_RTOL)
    bounds = kernel_mod.canal_bounds(16, 4)
    assert bounds == [0, 4, 8, 12, 16]
    assert len(bounds) == 5


@pytest.mark.kernels_ci
def test_cross_canal_values_do_not_mix(kernel_mod, cpu_device):
    torch.manual_seed(3)
    q = torch.randn(1, 1, 8, 16, device=cpu_device)
    k = torch.randn(1, 1, 8, 16, device=cpu_device)
    v = torch.randn(1, 1, 8, 16, device=cpu_device)
    y = kernel_mod.yarqa_attn(q, k, v, 2)
    v_zero_right = v.clone()
    v_zero_right[:, :, 4:, :] = 0
    y_left = kernel_mod.yarqa_attn(q, k, v_zero_right, 2)
    torch.testing.assert_close(y[:, :, :4, :], y_left[:, :, :4, :], atol=_ATOL, rtol=_RTOL)
    assert not torch.allclose(y[:, :, 4:, :], y_left[:, :, 4:, :], atol=_ATOL, rtol=_RTOL)


@pytest.mark.kernels_ci
def test_uneven_remainder_on_earlier_canals(kernel_mod):
    assert kernel_mod.canal_bounds(10, 3) == [0, 4, 7, 10]


@pytest.mark.kernels_ci
def test_selfcheck(kernel_mod):
    r = kernel_mod.selfcheck()
    assert r["ok"] is True
    assert r["lambda"] == "Conjecture 1"
    assert r["split"] is True
    assert r["tamper_detected"] is True
    assert r["path"] == "torch_compartment"


@pytest.mark.kernels_ci
def test_cpu_only_refuses_cuda_tensor(kernel_mod):
    if not torch.cuda.is_available():
        q = torch.randn(1, 1, 4, 8)
        # Honesty: no GPU in this environment — skip is not a fabricated pass.
        pytest.skip("No CUDA GPU — CUDA refusal path not exercised (honest skip)")
    q = torch.randn(1, 1, 4, 8, device="cuda")
    with pytest.raises(RuntimeError, match="CPU-only"):
        kernel_mod.yarqa_attn(q, q, q, 2)
