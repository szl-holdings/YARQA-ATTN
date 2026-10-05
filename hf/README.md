# Hugging Face card source: staged, not published

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

## Publish status: blocked on owner actions

**Nothing in this repository writes the Hub.** Both twins still serve the cards in `hub-import/`. Publishing `CARD.md` needs all of:

1. **An HF credential for this repository.** Preferred: Trusted Publishers for `SZLHOLDINGS/YARQA-ATTN` and `kernels/SZLHOLDINGS/YARQA-ATTN`, bound to this repository's mirror workflow on `main`. Fallback: a repository secret `HF_TOKEN` scoped to those two repos. This repository has no HF secret today. (Owner action.)
2. **The shared mirror.** `reusable-hf-mirror.yml` (model and kernel targets) is not merged in `szl-holdings/.github`; workflow changes there wait for the owner's trust-root review. No copy-pasted mirror is added here in the meantime.
3. **A kernel write path.** `huggingface_hub` 2.0.0 refuses commits to `repo_type="kernel"`; Git is the only proven transport.

When those exist, a thin caller of the shared mirror publishes `CARD.md` as `README.md` to both twins, ships `LICENSE` (the kernel repo has none on the Hub, and the model twin's `LICENSE` differs from this repository's), and keeps the Hub-only receipts listed in `hub-import/IMPORT.json`. Until then, nobody edits these cards on the Hub.

## What the import shows

Read the numbers from `hub-import/IMPORT.json`; the points below are what they mean.

- **The Hub kernel build matches this tree's `torch-ext/yarqa_attn/` sources**; only the card differs. Of the three kernels in this batch, this one is closest to publishable.
- **Hub-only receipts.** `BENCH.laptop-blackwell.json` and `OPERATIONAL.json` exist only on the Hub. The staged card cites them as `REPORTED`, not `MEASURED`.
- **Other writers.** The model twin also carries `yarqa.py`, uploaded by `szl-holdings/a11oy` `atelier-hub-publish.yml` (a cross-writer the plan retires), plus Hub-only `TRAINING_RECEIPT.json`, `yarqa_attn.npz`, `bom/model-bom.cdx.json` and a Hub copy of `CARD.md`. Its card thumbnail `og-card.png` exists only on the Hub; the staged card does not reference it.
