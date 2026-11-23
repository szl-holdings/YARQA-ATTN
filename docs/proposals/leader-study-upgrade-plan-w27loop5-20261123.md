# Leader-Study Upgrade Plan - W27 loop-5 - 2026-11-23

Status: PROPOSE-ONLY DRAFT - DO NOT MERGE. Owner-gated: merging requires governing approval and the estate signature-wall owner unlock.

Repo: YARQA-ATTN - KERNEL original SZL attention kernel (YARQA canal/compartment silhouette). Receipt-aware, honesty-labeled. Not a Flash/Sage/Flex clone, not a weight, not an alias of the Triton trio. Doctrine v11, Λ=Conjecture 1, Apache-2.0.

Frame: OUROBOROS loop-5 leader study. Governing seat: Kimi kimi-k3.1-pro (json_ok, think off). Advisory seats: Astra/Sol/Claude community rails (provenance unverified, advisory only). Leader evidence web-verified 2026-11-23.

## Leader moats to absorb
- FlashAttention-3: Hopper-optimized attention using asynchronous tensor-core execution; cited at 1.5-2x over FA2, 740 TFLOPS FP16, and approximately 1.2 PFLOPS FP8. Moat: Hardware-specific scheduling and measured throughput, with performance claims tied to a defined GPU target.
- FlashAttention-2: Established high-performance attention implementation and the baseline used by the cited FlashAttention-3 comparison. Moat: Recognized attention implementation lineage and a concrete predecessor baseline for performance evaluation.
- Triton: OpenAI GPU kernel language used for implementing GPU kernels. Moat: A programmable kernel-development path rather than only a fixed attention implementation.
- PyTorch scaled_dot_product_attention: PyTorch's standard scaled dot-product attention interface. Moat: A native framework entry point against which usability and numerical behavior can be compared.
- KernelBench: Stanford kernel-evaluation resource identified in the supplied facts. Moat: An external evaluation reference for separating kernel correctness and performance evidence from repository assertions.

## Honest gaps (grounded in repo facts)
- The supplied facts contain no GPU evidence, so the kernel has no established GPU performance or GPU correctness position relative to FlashAttention-3, FlashAttention-2, Triton implementations, or PyTorch scaled_dot_product_attention.
- Tokens/s and joules are explicitly not claimed; there is therefore no supplied throughput or energy benchmark to compare with the cited leaders.
- The measured evidence is an immutable CPU import with kernels==0.16.1 and a passing selfcheck(), not a floating-main qualification or production-readiness claim.
- GitHub-to-Hub source binding remains UNAVAILABLE because neither Hub twin exposes SZL_SOURCE_BINDING.json.
- Card projection remains SOURCE_AHEAD because both provider cards remain behind the source-owned card.
- The facts establish matching of three canonical Git object blobs across six provider copies for CPU and universal build variants, but do not establish broader release, hardware, or workload coverage.
- No numerical-equivalence results against PyTorch scaled_dot_product_attention or other attention implementations are supplied.
- No external KernelBench result is supplied, so independent kernel evaluation remains unestablished by these facts.
- The repository is described as an original kernel with no weights; model-quality or trained-weight comparisons would therefore be inappropriate.

## Dosed upgrade plan
- First preserve the locked identity and receipt contract: add no benchmark or compatibility wording that turns YARQA-ATTN into a Flash, Sage, Flex, paged-KV, weight, or Triton-trio alias.
- Propose a small correctness matrix for empty, single-token, single-compartment, multi-compartment, and uneven-partition cases, checking compartment isolation and deterministic SHA3-256 receipts; compare attention values with PyTorch scaled_dot_product_attention only where inputs and masking semantics are explicitly aligned.
- Record exact source revisions, dependency versions, device, dtypes, shapes, and commands for each result, keeping the existing immutable CPU import evidence distinct from any new execution evidence.
- After correctness evidence exists, propose a separately labeled benchmark protocol with PyTorch scaled_dot_product_attention and other suitable baselines; report only measured results and do not reuse leader figures as YARQA-ATTN results.
- Consider KernelBench only as an additional evaluation reference where applicable, and publish task selection and adaptations so a generic benchmark is not misrepresented as validation of canal semantics or receipts.
- In a later documentation-only change, propose adding the source-binding material needed to resolve UNAVAILABLE binding and bring provider cards toward the source-owned card; verify the result before changing either status.
- Keep GPU support, performance, energy, provider SLA, and production readiness unclaimed unless corresponding immutable evidence is subsequently measured and recorded.
- Risk note: The main risk is evidence overreach: converting a successful immutable CPU import and blob-match result into GPU, performance, numerical-equivalence, production, or source-binding claims. Benchmarking can also erode the product boundary if full-sequence attention baselines are compared without stating different semantics. Binding and card updates carry publication risk and should remain separate, reversible proposals with post-change verification.
- Propose-only note: This is a proposal only. No repository, Hub, provider-card, doctrine, or status change is authorized by this audit. Preserve Doctrine v11 LOCKED and Lambda = Conjecture 1 as advisory with uniqueness OPEN, never a theorem. Do not create empty Hub cards or claim that proposed tests or bindings have run.

## Advisory notes (unverified rails, non-binding)
- astra: YARQA-ATTN: Hardware-specific compiler optimizations for proprietary AI accelerators. curation-atlas: Real-time integration with private enterprise security telemetry. david-leads: Proprietary cross-referenced datasets of non-public government procurement.
- sol: YARQA-ATTN: Hardware-specific Triton/CUDA optimization primitives curation-atlas: Real-time GitHub event stream ingestion pipeline david-leads: Proprietary DSSE-evidence signal mapping schema
- claude: YARQA-ATTN: Hardware-specific compiler optimizations for proprietary AI accelerators. curation-atlas: Direct API integration with private enterprise security telemetry streams. david-leads: Exclusive access to non-public government procurement and intelligence feeds.

## DO NOT MERGE
This plan is a proposal artifact of the 10-pass OUROBOROS loop (loop-5 of 10). No code, no CI, no behavior changes in this file. A reviewer with authority may convert to a work order later.