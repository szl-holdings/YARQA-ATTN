---
language:
- Python
license: apache-2.0
library_name: kernels
tags:
  - kernel
  - attention
  - yarqa
  - governed-ai
  - szl-holdings
  - kernel-lane
szl:
  owner: KERNEL
  not_a_weight: true
  not_an_alias: true
  collection: none
  python: present
  import_live: false
---

# YARQA-ATTN

KERNEL kernel card. Original SZL **compartment / plug-flow** attention cut. Receipt-aware. Honesty-labeled.

**Not a Fall 2026 ATELIER weight.** No tensors in this repo. Not an alias of [`szl-receipt-attn`](https://github.com/szl-holdings/szl-receipt-attn). Not a pointer at the Triton trio (`szl-receipt-attn`, `szl-maskmod`, `szl-block-kv`). Those three stay separate. a11oy-net does not list this as a fourth Flash / Flex / paged stack.

GitHub is source of truth: [`szl-holdings/YARQA-ATTN`](https://github.com/szl-holdings/YARQA-ATTN). KERNEL binds Hub bytes from that tree. Do not PUT an empty card.

| | |
|---|---|
| **Owner** | KERNEL |
| **Artifact** | kernel (Python present; no weights; GPU cubins not claimed) |
| **Status** | Python present · CPU selfcheck · not import-LIVE |
| **License** | Apache-2.0 |
| **Λ** | Conjecture 1 (advisory, never a theorem) |
| **Path** | `torch_compartment` (CPU) |
| **Serve studio** | not this repo. Live CPU lab is [`szl-model-inference-lab`](https://huggingface.co/spaces/SZLHOLDINGS/szl-model-inference-lab) (Khipu GGUF only) |

Silhouette: partition a sequence into canals (contiguous compartments), attend within a canal, emit SHA3-256 of the partition and of the attention output. We do not copy Dao hopper, Sage `csrc`, vLLM paged `.cu`, cuDNN FMHA, TRT cubins, CuTeDSL, or `flex_attention.py`. Metaphor only vs [`szl-holdings/yarqa`](https://github.com/szl-holdings/yarqa) (CFD; different product). Throughput is MEASURED only from a timed run on named hardware. Until then every speed claim is unstamped. No tokens/s. No joules.

Do not list this next to Chaski, Qantu, Waman, Chakana, or Tinku.

## Load

```python
from kernels import get_kernel
attn = get_kernel("SZLHOLDINGS/YARQA-ATTN", revision="main", trust_remote_code=True)
```

KERNEL stamps import-LIVE only after Kernel Hub `get_kernel` MEASURE. This card is not that stamp.

Apache-2.0. Copyright 2026 SZL Holdings.
