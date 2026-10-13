# YARQA-ATTN

Canonical GitHub source for Hub `SZLHOLDINGS/YARQA-ATTN`.

**Owner: KERNEL.** This is a kernel, not a Fall 2026 ATELIER weight, not a pointer at the Triton trio (`szl-receipt-attn`, `szl-maskmod`, `szl-block-kv`), and not an alias of receipt-attn. a11oy-net must not list it as a fourth Flash / Flex / paged stack. Those three stay separate.

<!-- szl:header v1 -->
<!-- Native identity: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon. -->
<p>
  <a href="https://github.com/szl-holdings">
    <img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="32" height="32" />
  </a>
</p>

**Control before action. Evidence after.**

[Product](https://a-11-oy.com) · [Proof](https://a11oy.net) · [GitHub organization](https://github.com/szl-holdings)
<!-- /szl:header -->

Quechua *yarqa* = irrigation canal that divides flow. Original SZL cut: **compartment / plug-flow attention**. Partition a sequence into canals (contiguous compartments), attend **within** a compartment, emit SHA3-256 receipts of the partition and of the attention output.

Distinct silhouettes:

| This kernel | Not this kernel |
|---|---|
| Contiguous canals, attend inside each | Flash tiled fused attention → `szl-receipt-attn` |
| Block-diagonal by construction | Flex `score_mod` + block-mask → `szl-maskmod` |
| No paged KV gather | Paged KV → `szl-block-kv` |
| Attention kernel | CFD plug-flow at `github.com/szl-holdings/yarqa` (metaphor only; different product) |

We do not copy Dao hopper, Sage `csrc`, vLLM paged `.cu`, cuDNN FMHA, TRT cubins, CuTeDSL, or `flex_attention.py`.

**SOFTWARE/KERNEL.** Not a trained model. No weights. Tokens/s and joules are not claimed.

Doctrine v11 LOCKED. Λ = Conjecture 1 (advisory; uniqueness OPEN; never a theorem). GitHub bytes are the artifact. Hub is the publish mirror. KERNEL binds Hub after this source lands. No Hub PUT of empty cards.

<!-- SZL-KERNEL-STATUS:import-LIVE:START -->
## Status

> **MEASURED immutable CPU import — 2026-10-05 UTC.** Kernel Hub revision
> [`4e0828e`](https://huggingface.co/kernels/SZLHOLDINGS/YARQA-ATTN/commit/4e0828e517e84377d0024ab3f9d0cdab7d31520d)
> loaded with `kernels==0.16.1`; `selfcheck()` passed. The three canonical Git
> object blobs at protected source `cc895b47e7a3d1e7d5f246812a573dfd1205b103` matched all six
> provider copies across the CPU and universal build variants. Neither Hub twin
> exposes `SZL_SOURCE_BINDING.json`, and both provider cards remain behind the
> source-owned card, so GitHub-to-Hub source binding remains **UNAVAILABLE** and card projection is
> **SOURCE_AHEAD**. This is not a floating `main` qualification, a performance
> claim, GPU evidence, provider-SLA claim, or production-readiness claim.

| Observation | Label | Exact evidence |
|---|---|---|
| Immutable Kernel Hub import and `selfcheck()` | **MEASURED** | [`evidence/yarqa-postmerge-import-20261005.json`](evidence/yarqa-postmerge-import-20261005.json) |
| Canonical Git-object parity across CPU and universal builds | **MEASURED 6/6** | Same receipt; SHA-256 equality against immutable provider bytes |
| GitHub-to-Hub source-binding files | **UNAVAILABLE** | No `SZL_SOURCE_BINDING.json` on either observed Hub twin |
| Provider card projection | **SOURCE_AHEAD** | Source-owned `CARD.md` differs from both immutable provider cards |
| GPU cubins | **UNAVAILABLE / NOT SHIPPED** | v0 refuses CUDA; no GPU or throughput claim |

<!-- SZL-KERNEL-STATUS:import-LIVE:END -->

| | |
|---|---|
| **Python** | present (`torch-ext/yarqa_attn`) |
| **Path** | `torch_compartment` — CPU torch SDPA, with a labeled manual matmul+softmax fallback |
| **License** | Apache-2.0 |

## Load

The Kernel Hub repository is written only by this repository's
[`hf-kernel-publish`](.github/workflows/hf-kernel-publish.yml) workflow. It publishes
`torch-ext/yarqa_attn` to the `main` and `v1` branches and then verifies, anonymously in a
fresh runner, that `get_kernel` serves exactly the GitHub source:

```python
# pip install "kernels>=0.17.2" torch
from kernels import get_kernel

attn = get_kernel("SZLHOLDINGS/YARQA-ATTN", version=1, trust_remote_code=True)
```

`version=1` follows the `v1` branch (stable API channel). To pin an exact build, resolve
the `v1` branch to its Kernel Hub commit (`HfApi().list_repo_refs(..., repo_type="kernel")`)
and pass it as `revision=`; `SZL_KERNEL_SOURCE.json` on the Hub names the GitHub commit it
was built from. The 2026-10-05 measurement above is historical: it pinned the pre-pipeline
Kernel Hub commit `4e0828e`. The model twin `SZLHOLDINGS/YARQA-ATTN` is a pointer only; it
keeps `yarqa_attn.npz`, a small SYNTHETIC silhouette that the kernel never loads.

Source tree (labeled; not a Hub load; not import-LIVE):

```python
import torch
from yarqa_attn import yarqa_attn, ReceiptChain, selfcheck, canal_bounds

q = k = v = torch.randn(1, 2, 16, 32)
chain = ReceiptChain()
y = yarqa_attn(q, k, v, 4, chain=chain)
print(canal_bounds(16, 4), chain.verify(), selfcheck())
```

`selfcheck()` never fabricates a pass. It runs a small CPU check: slice-and-attend vs a naive block-diagonal full-attn-within-compartment reference, receipt tamper detect, and `n_canals > 1` actually splits.

## API

`yarqa_attn(q, k, v, n_canals, chain=None)` — `q,k,v` are `(batch, heads, seq, dim)` on **CPU**. Sequence length is split into `n_canals` contiguous canals (earlier canals receive the remainder). Each canal is independent attention. Outputs are concatenated along seq.

`ReceiptChain` — SHA3-256. One receipt for partition boundaries, one for the attention-output digest. `verify()` returns `(ok, depth, first_break)`.

`canal_bounds(seq_len, n_canals)` — exclusive end-points `[0, ..., seq_len]`.

v0 refuses CUDA tensors. That is honesty, not a missing bench.

## Correctness band (documented, not a bench)

fp32 vs naive within-compartment SDPA (block-diagonal keep-mask): **atol=1e-5, rtol=1e-5**.

## Tests

- kernel-builder: `nix run .#testshell-torch-ext-local` (sets `LOCAL_KERNELS`; `get_kernel` must hard-fail if that env is ignored)
- source tree (labeled, not a Hub load): `SZL_SOURCE_TREE_TESTS=1 PYTHONPATH=torch-ext python -m pytest tests/ -q`

## License

Apache-2.0. Copyright 2026 SZL Holdings. Owner: Stephen P. Lutar Jr. / SZL Holdings. Homepage: https://a-11-oy.com
