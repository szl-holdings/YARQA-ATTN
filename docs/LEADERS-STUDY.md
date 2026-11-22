# Domain Leader Study — cohort-01 (OUROBOROS phase-3)

Date: 2026-11-22 · Method: web/arXiv research, read-only · Class: study card, PROPOSE-ONLY.

## Domain
Attention kernel (owner: KERNEL; distinct from the Triton trio receipt-attn / maskmod / block-kv; not a fourth Flash/Flex/paged stack — retained).

## Leaders and publications
- FlexAttention + FlashAttention-4 (PyTorch blog, "Fast and Flexible"; arXiv:2412.05496 Flex Attention: a Programming Model for Generating Optimized Attention Kernels)
- Triton Attention Backend: Portable High-Performance Paged Attention for Production-Grade LLM Inference (SC-W 2026)
- vLLM Triton Attention Backend Deep Dive (vllm.ai blog, Mar 2026)
- Portable Paged Attention: From Triton to Helion (IBM Research, PyTorch Conference 2026)
- Flash Attention Kernel Benchmark: CK vs Triton vs PyTorch on AMD MI300X (2026)

## Absorb actions (proposals, not yet applied)
1. Align kernel docs with the score-mod programming-model taxonomy (leader: FlexAttention) without changing identity.
2. Adopt the prefill/mixed/decode latency-subplot benchmark protocol for kernel receipts (leader: vLLM deep dive).
3. Cross-vendor portability notes (Triton-to-Helion class) in kernel upgrade proposals.
4. Benchmark comparison table vs leader kernels, honestly labeled as measurement debt until actually measured.

Study receipted in OUROBOROS phase-3 program logs.
