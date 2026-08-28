# SPDX-FileCopyrightText: 2026 SZL Holdings
# SPDX-License-Identifier: Apache-2.0
"""Source-tree example of yarqa-attn. Not a benchmark. Shapes only.

Hub get_kernel is MEASURE after KERNEL bind. This script is not import-LIVE.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "torch-ext"))

import torch

from yarqa_attn import ReceiptChain, canal_bounds, selfcheck, yarqa_attn

q = torch.randn(1, 2, 16, 32)
k = torch.randn(1, 2, 16, 32)
v = torch.randn(1, 2, 16, 32)
chain = ReceiptChain()
out = yarqa_attn(q, k, v, 4, chain=chain)
ok, depth, first_break = chain.verify()
print("bounds", canal_bounds(16, 4))
print("out", tuple(out.shape), "verify", ok, depth, first_break)
print("selfcheck", selfcheck())
