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
  import_live: false
---
<!-- hf-card: type=kernel source=szl-holdings/YARQA-ATTN vars=hf/card.yaml -->
<!-- Rendered by szl-holdings/.github hf-card/render.py. Edit hf/card.yaml in szl-holdings/YARQA-ATTN; do not edit this card on the Hub. -->

<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112" /></a></p>

# YARQA-ATTN

KERNEL kernel card. Original SZL **compartment / plug-flow** attention cut: partition a sequence into canals (contiguous compartments), attend within each canal, and emit SHA3-256 receipts of the partition and of the attention output. Receipt-aware. Honesty-labeled. SOFTWARE/KERNEL: not a trained model; the kernel ships no weights.

**Artifact:** CPU compartment attention kernel · **Stage:** Published from GitHub by hf-kernel-publish (v1 channel)

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/YARQA-ATTN) · [Evidence](https://github.com/szl-holdings/YARQA-ATTN/blob/2f15aded33e9d86407d8cec17abc7bfb1501bd0a/evidence/yarqa-postmerge-import-20261005.json)

## Before you use it

- Kernel, not weights. Live CPU inference lab is `SZLHOLDINGS/szl-model-inference-lab` (Khipu GGUF only), not this repo.
- Receipts prove integrity and declared origin only; they do not prove accuracy, readiness or performance.

<details>
<summary>Technical details and evidence</summary>

## Kernel

| Field | Value |
| --- | --- |
| Hub id | `SZLHOLDINGS/YARQA-ATTN` |
| Library | `kernels` |
| Backends | cpu (torch_compartment) |

```python
# pip install "kernels>=0.17.2" torch
# This loads and executes remote Python code; review the source first.
from kernels import get_kernel

attn = get_kernel("SZLHOLDINGS/YARQA-ATTN", version=1, trust_remote_code=True)
```

## What it is not

**Not a Fall 2026 ATELIER weight.** No tensors in this kernel. The model twin
`SZLHOLDINGS/YARQA-ATTN` keeps `yarqa_attn.npz`, a small SYNTHETIC attention-gate
silhouette described by its `TRAINING_RECEIPT.json`; the kernel never loads it, and it
carries no benchmark or capability claim. Not an alias of
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

## Source and Hub release scope

GitHub `szl-holdings/YARQA-ATTN` is the only writer of the Kernel Hub repository.
Its `hf-kernel-publish` workflow builds `build/torch-cpu` and `build/torch-universal`
(identical bytes) from `torch-ext/yarqa_attn` at one GitHub commit, writes them to the
`main` and `v1` branches, and then loads them anonymously with
`get_kernel("SZLHOLDINGS/YARQA-ATTN", version=1, trust_remote_code=True)` in a fresh
runner and compares the served files with the GitHub source. `SZL_KERNEL_SOURCE.json`
in the Kernel Hub repository names the GitHub commit each build came from.

`version=1` follows the `v1` branch, the stable API channel. Loading with
`trust_remote_code=True` executes code from the Kernel Hub repository; the narrowest
form is `trust_remote_code=["SZLHOLDINGS/YARQA-ATTN"]`. To pin an exact build, resolve
the `v1` branch to its commit and pass that as `revision`:

```python
from huggingface_hub import HfApi
from kernels import get_kernel

refs = HfApi().list_repo_refs("SZLHOLDINGS/YARQA-ATTN", repo_type="kernel")
pin = next(b.target_commit for b in refs.branches if b.name == "v1")
attn = get_kernel("SZLHOLDINGS/YARQA-ATTN", revision=pin, trust_remote_code=["SZLHOLDINGS/YARQA-ATTN"])
```

A GitHub source commit or a model-twin revision is not a Kernel Hub revision. The
model twin carries only a pointer to the Kernel Hub repository. Publication does not
qualify performance, GPU execution, the model-twin NPZ artifact or production use.

Publication context is recorded in [hf/README.md](https://github.com/szl-holdings/YARQA-ATTN/blob/main/hf/README.md).

## Evidence record

**MEASURED** — Compartment correctness. `tests/test_yarqa_attn.py` asserts that fp32 output matches a naive within-compartment (block-diagonal) reference, and with one canal matches full SDPA, within atol=1e-5, rtol=1e-5; that more than one canal actually splits; that one canal's output does not depend on another canal's values; that earlier canals take the remainder; and that `selfcheck()` reports ok. CI runs it on every pull request and every push to main (`cpu-tests.yml`). [Receipt](https://github.com/szl-holdings/YARQA-ATTN/blob/724208adc51c664f48aded57ee3c0d2cefc18361/tests/test_yarqa_attn.py).

**MEASURED** — Receipts. `tests/test_receipt.py` asserts one receipt for the partition and one for the output, that tampering breaks the chain at the first row, and that the digests change with the values and with the canal count. [Receipt](https://github.com/szl-holdings/YARQA-ATTN/blob/724208adc51c664f48aded57ee3c0d2cefc18361/tests/test_receipt.py).

**MEASURED** — Immutable CPU `get_kernel` load from Kernel Hub revision `4e0828e517e84377d0024ab3f9d0cdab7d31520d` using `kernels==0.16.1` on 2026-10-05 UTC. `selfcheck()` passed. The three canonical Git object blobs at protected source `cc895b47e7a3d1e7d5f246812a573dfd1205b103` matched all six provider copies across the CPU and universal build variants. Neither Hub twin exposes `SZL_SOURCE_BINDING.json`, and both provider cards remain behind the source-owned card. Source binding is UNAVAILABLE and card projection is SOURCE_AHEAD. No performance, GPU, provider-SLA or production claim is made. [Receipt](https://github.com/szl-holdings/YARQA-ATTN/blob/2f15aded33e9d86407d8cec17abc7bfb1501bd0a/evidence/yarqa-postmerge-import-20261005.json).

**UNAVAILABLE** — GPU cubins. None shipped; the 2026-08-28 session had no CUDA device. v0 refuses CUDA tensors, and the refusal test is skipped where no GPU exists. (no receipt linked).

**NOT_CLAIMED** — Throughput, tokens/s, joules, or performance versus FlashAttention. A speed claim needs a timed run on named hardware; none exists. (no receipt linked).

Labels follow the [SZL claim language](https://github.com/szl-holdings/.github/blob/main/docs/CLAIM_LANGUAGE.md). A claim is only as strong as the receipt it links.

## Source and provenance

| Field | Value |
| --- | --- |
| Source repository | [szl-holdings/YARQA-ATTN](https://github.com/szl-holdings/YARQA-ATTN) |
| Proof | <https://github.com/szl-holdings/YARQA-ATTN> |
| License | `apache-2.0` |

Rendering this card does not establish Hub publication. Verify any Hub copy against its source revision, publication evidence, and exact Hub revision.

</details>
