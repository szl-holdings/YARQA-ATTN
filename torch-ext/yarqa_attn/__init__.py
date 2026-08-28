# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SZL Holdings
"""yarqa-attn public API. Compartment / plug-flow attention. Not a Flash/Flex/paged stack."""

from ._chain import ReceiptChain
from .attn import canal_bounds, compartment_mask, selfcheck, yarqa_attn

__all__ = [
    "ReceiptChain",
    "canal_bounds",
    "compartment_mask",
    "selfcheck",
    "yarqa_attn",
]
__version__ = "0.1.0"
