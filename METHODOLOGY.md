# Methodology: Compile-Time Governance Overhead Measurement on BitNet

## Research question

What is the runtime overhead of compile-time invariant enforcement (MIND
governance) on BitNet b1.58 inference, compared to:

1. Bare BitNet inference (no governance)
2. BitNet + runtime guardrails (e.g. NeMo Guardrails, Guardrails AI)
3. BitNet + MIND compile-time invariants

And: how does cross-substrate determinism (bit-identical replay) compare
across the three configurations?

## Hypotheses

**H1**: MIND compile-time governance overhead on BitNet workloads is
statistically negligible (< 1%), because BitNet is memory-bandwidth-bound
while MIND invariant checks are CPU-bound — they run on disjoint resources.

**H2**: Runtime guardrails impose 5–15% overhead on BitNet because they add
sequential post-inference inference steps, some of which compete for the
same memory bus.

**H3**: BitNet + MIND produces bit-identical SHA-256 evidence chains across
x86_64, ARM64, and CUDA targets for identical inputs, enabling auditable
replay. Bare BitNet on FP16 substrates does not.

## Experimental setup

### Substrate

- **Model**: BitNet b1.58 2B4T (Microsoft, Apache 2.0)
  - 2.4B parameters, ternary weights `{-1, 0, +1}`, 4-bit activations
  - HuggingFace: `microsoft/bitnet-b1.58-2B-4T`

### Hardware targets

| Target | Class | Why |
|---|---|---|
| Intel Xeon (or equivalent) | x86_64 CPU | Baseline, deterministic substrate |
| Raspberry Pi 5 (8GB) | ARM64 CPU | Cross-arch validation |
| NVIDIA L4 or T4 | CUDA | GPU validation |
| NVIDIA P100 (16GB) | CUDA legacy | Memory-constrained validation |

### Three configurations under test

| Config | Description | Implementation |
|---|---|---|
| `baseline` | Bare BitNet inference, no governance | Reference HuggingFace transformers pipeline |
| `runtime` | BitNet + runtime guardrails | Wrapped with NeMo Guardrails or Guardrails AI |
| `compile_time` | BitNet + MIND compile-time invariants | Compiled via `mindc` from `src/bitnet.mind` |

### Workloads

For each configuration and hardware target, run:

| Workload | Batch | Seq Len | Tokens generated |
|---|---|---|---|
| Single token | 1 | 1 | 1 |
| Short prompt | 1 | 128 | 64 |
| Medium prompt | 1 | 512 | 128 |
| Long prompt | 1 | 2048 | 256 |
| Batched | 4, 8, 16 | 512 | 128 |

## Measurements

### Per-token metrics

- Latency (mean, p50, p95, p99)
- Throughput (tok/sec)
- Memory peak (host RAM, GPU VRAM)
- Energy per token (where measurable via NVML, RAPL, or external meter)

### Determinism metrics

- SHA-256 of the full evidence chain per inference
- Bit-identicality across hardware targets for the same input + seed
- Variance of TTFT across 100 reproductions per configuration
- Stability Index (SI) drift across runs

### Governance metrics

- Number of invariants enforced at compile time
- Number of refusals (when a generated token would violate an invariant)
- Time spent in invariant checking (where measurable)

## Reproducibility

- Fixed seed: `0xdeadbeefc0ffee` for all runs
- 100 reproductions per (config × hardware × workload) cell
- All raw measurements stored as JSON with hashes for verification
- Cross-architecture harness in `tests/test_determinism.py` validates
  hash-parity claim

## Statistical analysis

- Paired t-test for overhead deltas (baseline vs runtime, baseline vs compile_time)
- Bonferroni correction for multiple comparisons across hardware targets
- Bootstrap 95% CI on latency percentiles (n=10000)
- Effect size (Cohen's d) reported alongside p-values

## Threats to validity

| Threat | Mitigation |
|---|---|
| HuggingFace BitNet implementation may have framework overhead | Compare to a hand-rolled reference if discrepancy > 2% |
| Runtime guardrails not optimized for BitNet | Use multiple runtime systems (NeMo + Guardrails AI), report worst-case to be fair |
| Energy meter accuracy varies across hardware | Report all three (NVML, RAPL, external) where available |
| `mindc` compiler is STARGA-internal, not open-source | Pre-built binaries shipped with checksums for reproducibility audit |
| Determinism claim depends on Q16.16 precision | Document precision budget, verify against reference HF output |

## Expected outcomes

- **If H1 holds**: Compile-time governance is essentially free on ternary
  workloads, suggesting a structural advantage for high-stakes regulated
  inference where guardrails are mandatory.
- **If H1 fails**: Either MIND has unanticipated memory-bus contention, or
  BitNet is not as bandwidth-bound as theory predicts. Both are publishable.
- **If H3 holds**: This is the first demonstrated bit-identical replay of
  LLM inference across CPU/GPU/ARM substrates, enabling reproducible
  evaluation, regulatory replay, and Daubert-admissible AI evidence chains.

## Authorship and contribution

To be negotiated. Suggested split:

- **Experimental design and execution**: Lead author (university group)
- **Methodology and governance system**: STARGA Inc.
- **Senior author**: Faculty advisor
- **Affiliations**: As negotiated

## Target venues

- MLSys 2026 (deadline ~October)
- NeurIPS Workshop on Efficient ML
- USENIX OSDI
- Workshop on Trustworthy ML

## Open questions

- Should we include a 7B-class model (e.g. Llama-3.1-8B at FP16) as a
  contrast point to BitNet, to show governance overhead scales with the
  bandwidth profile of the substrate?
- Should we measure adversarial robustness of governance under malformed
  inputs, or scope this paper strictly to performance?
