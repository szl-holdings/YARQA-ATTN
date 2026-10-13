---
license: apache-2.0
tags:
  - kernel
  - attention
  - yarqa
  - szl-holdings
---

# {{KERNEL_NAME}} has moved to the kernel repository

This model repository holds no loadable build. The `kernels` library (0.16 and later) loads
kernels only from the kernel repository:

**[{{KERNEL_URL}}]({{KERNEL_URL}})**

```python
# pip install "kernels>=0.17.2" torch
from kernels import get_kernel

kernel = get_kernel("{{REPO_ID}}", version={{KERNEL_VERSION}}, trust_remote_code=True)
```

The source of truth is [{{GITHUB_REPOSITORY}}](https://github.com/{{GITHUB_REPOSITORY}}).
Its `hf-kernel-publish` workflow builds the kernel, verifies it through `get_kernel`, and
then updates this pointer. Commits of this model repository are not kernel revisions.

## What else is in this repository

- `yarqa_attn.npz` is a small **SYNTHETIC** attention-gate silhouette (route vs refuse),
  described by `TRAINING_RECEIPT.json` next to it. It is not loaded by the kernel, is not a
  trained production model, and carries no benchmark or capability claim. The kernel itself
  ships no weights.
- `bom/model-bom.cdx.json`, `LICENSE` and `og-card.png` are kept as uploaded.
