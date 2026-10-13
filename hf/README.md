# Hugging Face card source and kernel publication

This folder is the GitHub source for the two Hub twins of this repository:

- the Kernel Hub repo `kernels/SZLHOLDINGS/YARQA-ATTN` (what `get_kernel` loads);
- the model-type twin `SZLHOLDINGS/YARQA-ATTN`.

| Path | What it is |
| --- | --- |
| `card.yaml` | The card source for **both** twins (plan decision D9: one source, one template). |
| `../CARD.md` | Its rendering by the shared [`hf-card`](https://github.com/szl-holdings/.github/tree/main/hf-card) toolkit. This is the file a mirror publishes as `README.md`. |
| `hub-import/` | The Hub cards exactly as the Hub served them, and `IMPORT.json`: the imported revisions, card digests, the full Hub file list of both twins, and how each Hub file compares with this repository. |

CI keeps them honest:

- `.github/workflows/hf-card.yml` fails when `CARD.md` drifts from `card.yaml`, or when the card breaks the card contract: front matter (`license`, `library_name: kernels`, `szl.source_repo`, `szl.proof_url`), and decision D10 (every `MEASURED` claim links a receipt pinned to a commit of this repository).
- `tests/test_hf_card_source.py` checks, offline, that `hub-import/` still holds the imported bytes and that `CARD.md` names this repository and its Apache-2.0 license.

To edit the card, change `card.yaml` and re-render:

```bash
python <hf-card>/render.py hf/card.yaml --out CARD.md
```

To refresh the import (read-only, anonymous public Hub API, no token):

```bash
python scripts/hf_hub_import.py fetch
```

## Current post-merge witness

The read-only witness in [`../evidence/yarqa-postmerge-import-20261005.json`](../evidence/yarqa-postmerge-import-20261005.json) binds protected
GitHub source `cc895b47e7a3d1e7d5f246812a573dfd1205b103` to immutable Kernel Hub revision
`4e0828e517e84377d0024ab3f9d0cdab7d31520d` and model-twin revision `d6aeb4f24655c07acc155eed7acade5ad249724f`.
It measures 6/6 canonical Git-object matches across the CPU and universal package
variants and an exact-revision CPU `selfcheck()` pass. Neither twin exposes
`SZL_SOURCE_BINDING.json`, and both provider cards differ from the source-owned card,
so the current disposition is **PACKAGE_ALIGNED / SOURCE_BINDING_UNAVAILABLE /
CARD_SOURCE_AHEAD**. No Hub mutation was performed.

The earlier `evidence/yarqa-import-20261005.json` remains preserved as historical
evidence. Its parity conclusion is retained, but its hash rows reflected Windows
checkout line endings. The successor receipt records canonical Git object bytes and
immutable provider bytes, preventing checkout-EOL contamination.

## Publication: `hf-kernel-publish`

[`.github/workflows/hf-kernel-publish.yml`](../.github/workflows/hf-kernel-publish.yml) with
[`scripts/szl_kernel_publish.py`](../scripts/szl_kernel_publish.py) is the only writer of both
twins. Its configuration is [`.github/hf-kernels.json`](../.github/hf-kernels.json).

- **Kernel repo** (`kernels/SZLHOLDINGS/YARQA-ATTN`, what `get_kernel` loads): `build/torch-cpu`
  and `build/torch-universal` (identical bytes, LF-normalised) from `torch-ext/yarqa_attn`, a
  generated `metadata.json` with a sha256 digest per variant, `CARD.md` as `README.md` (after the
  publisher's card lint), `LICENSE`, `NOTICE`, and `SZL_KERNEL_SOURCE.json` naming the GitHub
  commit. The same tree is written to `main` and `v1` by fast-forward Git pushes over HTTPS
  (`huggingface_hub` cannot commit to kernel repos). The stale Hub-only plates
  `OPERATIONAL.json` (it pinned `revision="main"`) and `BENCH.laptop-blackwell.json` are pruned.
- **Verification** runs anonymously in a fresh runner with `kernels==0.17.2` and CPU torch:
  `get_kernel("SZLHOLDINGS/YARQA-ATTN", version=1, trust_remote_code=True)`, the allowlist form,
  and `revision="main"` must load `build/<variant>/yarqa_attn` of the resolved commit, and every
  served file must equal the GitHub source.
- **Model twin** (`SZLHOLDINGS/YARQA-ATTN`), only after verification passes: `README.md` becomes
  [`model-twin-pointer.md`](model-twin-pointer.md), and the stale source copies (`torch-ext/`,
  `tests/`, `yarqa.py`, `example.py`, `build.toml`, `flake.nix`, `pyproject.toml`, `CARD.md`) and
  plates are pruned. `yarqa_attn.npz` (SYNTHETIC), `TRAINING_RECEIPT.json`, `bom/`, `LICENSE` and
  `og-card.png` are kept; the pointer card discloses the NPZ.

Triggers: a published release publishes its tag; `workflow_dispatch` (default branch only;
`dry_run` plans without writing) repairs or publishes a chosen ref; pull requests build and load
both variants locally; a weekly job re-verifies what `get_kernel` serves against the latest release.
The credential is the org secret `HF_TOKEN`, exposed only to the two writing steps.

`hub-import/` and the 2026-10-05 witnesses below are historical: they describe the Hub before
this pipeline existed.

## What the import shows

Read the numbers from `hub-import/IMPORT.json`; the points below are what they mean.

- **The Hub kernel build matches this tree's `torch-ext/yarqa_attn/` sources**; only the card differs. Of the three kernels in this batch, this one is closest to publishable.
- **Hub-only receipts.** `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` exist only on the Hub. The staged card cites them as `REPORTED`, not `MEASURED`.
- **Other writers.** The model twin also carries `yarqa.py`, uploaded by `szl-holdings/a11oy` `atelier-hub-publish.yml` (a cross-writer the plan retires), plus Hub-only `TRAINING_RECEIPT.json`, `yarqa_attn.npz`, `bom/model-bom.cdx.json` and a Hub copy of `CARD.md`. Its card thumbnail `og-card.png` exists only on the Hub; the staged card does not reference it.
