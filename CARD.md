---
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
  source_repo: szl-holdings/YARQA-ATTN
  proof_url: https://github.com/szl-holdings/YARQA-ATTN
  owner: KERNEL
  not_a_weight: true
  not_an_alias: true
  collection: none
  python: present
  import_live: true
---
<!-- hf-card: type=kernel source=szl-holdings/YARQA-ATTN vars=hf/card.yaml -->
<!-- Rendered by szl-holdings/.github hf-card/render.py. Edit hf/card.yaml in szl-holdings/YARQA-ATTN; do not edit this card on the Hub. -->

# YARQA-ATTN

KERNEL kernel card. Original SZL **compartment / plug-flow** attention cut: partition a sequence into canals (contiguous compartments), attend within each canal, and emit SHA3-256 receipts of the partition and of the attention output. Receipt-aware. Honesty-labeled. SOFTWARE/KERNEL: not a trained model, no weights.

## Kernel

| Field | Value |
| --- | --- |
| Hub id | `SZLHOLDINGS/YARQA-ATTN` |
| Library | `kernels` |
| Backends | cpu (torch_compartment) |

```python
from kernels import get_kernel

attn = get_kernel("SZLHOLDINGS/YARQA-ATTN", revision="main", trust_remote_code=True)
```

## What it is not

**Not a Fall 2026 ATELIER weight.** No tensors in this repo. Not an alias of
[`szl-receipt-attn`](https://github.com/szl-holdings/szl-receipt-attn). Not a pointer
at the Triton trio (`szl-receipt-attn`, `szl-maskmod`, `szl-block-kv`); those three
stay separate, and a11oy-net must not list this as a fourth Flash / Flex / paged stack.
Do not list this next to Chaski, Qantu, Waman, Chakana, or Tinku.

| This kernel | Not this kernel |
|---|---|
| Contiguous canals, attend inside each | Flash tiled fused attention → `szl-receipt-attn` |
| Block-diagonal by construction | Flex `score_mod` + block-mask → `szl-maskmod` |
| No paged KV gather | Paged KV → `szl-block-kv` |
| Attention kernel | CFD plug-flow at `github.com/szl-holdings/yarqa` (metaphor only; different product) |

We do not copy Dao hopper, Sage `csrc`, vLLM paged `.cu`, cuDNN FMHA, TRT cubins,
CuTeDSL, or `flex_attention.py`.

## Doctrine

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory; uniqueness OPEN; never a theorem). GitHub bytes are the artifact; the Hub is the publish mirror.

## API

`yarqa_attn(q, k, v, n_canals, chain=None)`: `q,k,v` are `(batch, heads, seq, dim)` on
**CPU**. Sequence length is split into `n_canals` contiguous canals (earlier canals
receive the remainder). Each canal is independent attention. Outputs are concatenated
along seq.

`ReceiptChain`: SHA3-256. One receipt for partition boundaries, one for the
attention-output digest. `verify()` returns `(ok, depth, first_break)`.

`canal_bounds(seq_len, n_canals)`: exclusive end-points `[0, ..., seq_len]`.

Source tree (put `torch-ext/` on `PYTHONPATH`):

```python
import torch
from yarqa_attn import yarqa_attn, ReceiptChain, selfcheck, canal_bounds

q = k = v = torch.randn(1, 2, 16, 32)
chain = ReceiptChain()
y = yarqa_attn(q, k, v, 4, chain=chain)
print(canal_bounds(16, 4), chain.verify(), selfcheck())
```

`selfcheck()` never fabricates a pass. It runs a small CPU check: slice-and-attend vs a
naive block-diagonal reference, receipt tamper detection, and that `n_canals > 1`
actually splits.

## Claims

| Label | Claim | Evidence |
| --- | --- | --- |
| MEASURED | Compartment correctness. `tests/test_yarqa_attn.py` asserts that fp32 output matches a naive within-compartment (block-diagonal) reference, and with one canal matches full SDPA, within atol=1e-5, rtol=1e-5; that more than one canal actually splits; that one canal's output does not depend on another canal's values; that earlier canals take the remainder; and that `selfcheck()` reports ok. CI runs it on every pull request and every push to main (`cpu-tests.yml`). | [receipt](https://github.com/szl-holdings/YARQA-ATTN/blob/176c5bc8af34a9eeac2dd28b2a4d6e70c7456828/tests/test_yarqa_attn.py) |
| MEASURED | Receipts. `tests/test_receipt.py` asserts one receipt for the partition and one for the output, that tampering breaks the chain at the first row, and that the digests change with the values and with the canal count. | [receipt](https://github.com/szl-holdings/YARQA-ATTN/blob/176c5bc8af34a9eeac2dd28b2a4d6e70c7456828/tests/test_receipt.py) |
| REPORTED | CPU `get_kernel` load from the Kernel Hub (kernels 0.16.1, `build/torch-universal` and `build/torch-cpu`, `selfcheck` ok, `path=torch_compartment`) and a local pytest run, recorded 2026-08-29 in the Hub-only `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` against source commit 160640bd. No receipt for them is committed in this repository. | none linked |
| UNAVAILABLE | GPU cubins. None shipped; the 2026-08-28 session had no CUDA device. v0 refuses CUDA tensors, and the refusal test is skipped where no GPU exists. | none linked |
| NOT_CLAIMED | Throughput, tokens/s, joules, or performance versus FlashAttention. A speed claim needs a timed run on named hardware; none exists. | none linked |

Labels follow the [SZL claim language](https://github.com/szl-holdings/.github/blob/main/docs/CLAIM_LANGUAGE.md). A claim is only as strong as the receipt it links.

## Limits

- Kernel, not weights. Live CPU inference lab is `SZLHOLDINGS/szl-model-inference-lab` (Khipu GGUF only), not this repo.
- Receipts prove integrity and declared origin only; they do not prove accuracy, readiness or performance.

## Source and provenance

| Field | Value |
| --- | --- |
| Source repository | [szl-holdings/YARQA-ATTN](https://github.com/szl-holdings/YARQA-ATTN) |
| Proof | <https://github.com/szl-holdings/YARQA-ATTN> |
| License | `apache-2.0` |

This card is written to the Hub only by the committed mirror workflow of szl-holdings/YARQA-ATTN.
