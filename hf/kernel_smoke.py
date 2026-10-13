# SPDX-License-Identifier: Apache-2.0
"""Smoke hook for hf-kernel-publish (YARQA-ATTN).

`smoke(kernel)` receives the module that the `kernels` loader imported, from a local
build (check-local) or from the Hub (verify). Raise, or return False, to fail the run.
CPU-only, deterministic and fast. The result's repr is written to the receipt.
"""
from __future__ import annotations


def smoke(kernel):
    import torch

    report = kernel.selfcheck()
    if report.get("ok") is not True:
        raise AssertionError(f"selfcheck() did not pass: {report}")
    if kernel.canal_bounds(16, 4) != [0, 4, 8, 12, 16]:
        raise AssertionError("canal_bounds(16, 4) must be [0, 4, 8, 12, 16]")
    generator = torch.Generator().manual_seed(20261013)
    q, k, v = (torch.randn(1, 2, 16, 32, generator=generator) for _ in range(3))
    chain = kernel.ReceiptChain()
    y = kernel.yarqa_attn(q, k, v, 4, chain=chain)
    if not isinstance(y, torch.Tensor) or y.shape != q.shape:
        raise AssertionError("yarqa_attn must return a tensor shaped like q")
    ok, depth, first_break = chain.verify()
    if not (ok and depth == 2 and first_break == -1):
        raise AssertionError(f"receipt chain must verify with depth 2: {(ok, depth, first_break)}")
    return {"selfcheck_ok": True, "chain_depth": depth, "max_abs_vs_compartment_ref":
            report.get("max_abs_vs_compartment_ref"), "version": getattr(kernel, "__version__", None)}
