# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""Compartment / plug-flow attention. Original SZL cut.

Named attn.py (not _ops.py): kernel-builder generates
torch-ext/<python_name>/_ops.py with add_op_namespace_prefix.

CPU only. No Dao hopper, Sage csrc, vLLM paged .cu, cuDNN FMHA,
TRT cubins, CuTeDSL, or flex_attention.py.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from ._chain import ReceiptChain, sha3_hex

PATH = "torch_compartment"
LAMBDA = "Conjecture 1"
_ATOL = 1.0e-5
_RTOL = 1.0e-5


def canal_bounds(seq_len: int, n_canals: int) -> List[int]:
    """Exclusive endpoints of contiguous canals. Remainder goes to earlier canals."""
    if not isinstance(n_canals, int) or isinstance(n_canals, bool):
        raise TypeError("n_canals must be an int")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if n_canals < 1:
        raise ValueError("n_canals must be >= 1")
    if n_canals > seq_len:
        raise ValueError("n_canals cannot exceed sequence length")
    base, rem = divmod(seq_len, n_canals)
    bounds = [0]
    for i in range(n_canals):
        width = base + (1 if i < rem else 0)
        bounds.append(bounds[-1] + width)
    return bounds


def compartment_mask(seq_len: int, n_canals: int, *, device=None) -> torch.Tensor:
    """Boolean keep-mask: True where query and key sit in the same canal."""
    bounds = canal_bounds(seq_len, n_canals)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for s, e in zip(bounds, bounds[1:]):
        mask[s:e, s:e] = True
    return mask


def _require_cpu(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t.device.type != "cpu":
            raise RuntimeError(
                "YARQA-ATTN v0 is CPU-only. GPU cubins are not claimed. "
                "This is not a Flash/Flex/paged stack."
            )


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 (batch, heads, seq, dim)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape (batch, heads, seq, dim) in v0")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v dtype must match")


def _within_canal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
    except Exception:
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


def _block_diag_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_canals: int
) -> torch.Tensor:
    """Naive full-attn-within-compartment: one SDPA with a block-diagonal keep-mask."""
    seq = q.shape[2]
    mask = compartment_mask(seq, n_canals, device=q.device)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )


def _output_digest(y: torch.Tensor) -> str:
    blob = y.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()
    return sha3_hex(blob)


def yarqa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_canals: int,
    *,
    chain: Optional[ReceiptChain] = None,
) -> torch.Tensor:
    """Attend inside each contiguous canal; concatenate along seq.

    Not Flash tiled fusion, not Flex score_mod, not paged KV gather.
    """
    _require_cpu(q, k, v)
    _validate_qkv(q, k, v)
    seq = int(q.shape[2])
    bounds = canal_bounds(seq, n_canals)
    pieces = []
    for start, end in zip(bounds, bounds[1:]):
        pieces.append(
            _within_canal(
                q[:, :, start:end, :],
                k[:, :, start:end, :],
                v[:, :, start:end, :],
            )
        )
    y = torch.cat(pieces, dim=2)
    if chain is not None:
        chain.emit(
            {
                "op": "yarqa_partition",
                "path": PATH,
                "n_canals": n_canals,
                "bounds": list(bounds),
                "q_shape": list(q.shape),
                "dtype": str(q.dtype).replace("torch.", ""),
                "lambda": LAMBDA,
            }
        )
        chain.emit(
            {
                "op": "yarqa_output",
                "path": PATH,
                "n_canals": n_canals,
                "output_digest": _output_digest(y),
                "out_shape": list(y.shape),
                "lambda": LAMBDA,
            }
        )
    return y


def _tamper_breaks(chain: ReceiptChain) -> bool:
    if not chain._rows:
        return False
    saved = chain._rows[0].get("bounds")
    chain._rows[0]["bounds"] = [0, 0]
    ok, _, first = chain.verify()
    chain._rows[0]["bounds"] = saved
    return (not ok) and first == 0


def _actually_splits(q, k, v, n_canals: int, y: torch.Tensor) -> bool:
    if n_canals <= 1:
        return False
    y_one = yarqa_attn(q, k, v, 1)
    if torch.allclose(y, y_one, atol=_ATOL, rtol=_RTOL):
        return False
    bounds = canal_bounds(int(q.shape[2]), n_canals)
    widths = [e - s for s, e in zip(bounds, bounds[1:])]
    return len(widths) == n_canals and min(widths) >= 1


def selfcheck() -> dict:
    torch.manual_seed(20260828)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    n_canals = 4
    chain = ReceiptChain()
    y = yarqa_attn(q, k, v, n_canals, chain=chain)
    ref = _block_diag_reference(q, k, v, n_canals)
    err = float((y - ref).abs().max().item())
    ok_chain, depth, brk = chain.verify()
    split = _actually_splits(q, k, v, n_canals, y)
    tamper = _tamper_breaks(chain)
    ok = bool(
        err < _ATOL
        and ok_chain
        and depth == 2
        and brk == -1
        and split
        and tamper
    )
    return {
        "ok": ok,
        "max_abs_vs_compartment_ref": err,
        "chain_ok": ok_chain,
        "chain_depth": depth,
        "chain_break": brk,
        "tamper_detected": tamper,
        "split": split,
        "n_canals": n_canals,
        "path": PATH,
        "lambda": LAMBDA,
        "python": "present",
        "note": (
            "CPU correctness vs naive within-compartment attn; "
            "no speedup claimed; GPU cubins not claimed; not import-LIVE"
        ),
    }
