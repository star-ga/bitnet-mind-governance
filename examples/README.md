# Examples

This directory holds one compiler-checked governance example and two
**research-direction skeletons** for distributed execution patterns proposed
for the methods paper. The skeletons are not compilable under current mindc
(0.10.x).

See `docs/parallel_pipeline.md` for the full discussion of why these
patterns matter and what's required to land them.

## Files

| File                              | What it shows                                                          |
|-----------------------------------|------------------------------------------------------------------------|
| `governance/rng_in_kernel.mind`   | A `#[deterministic]` kernel that draws from a PRNG; mindc must refuse it |
| `tensor_parallel_attention.mind`  | TP attention shard with lexicographic-order all-gather                 |
| `pipeline_parallel_forward.mind`  | PP 3-stage forward with compile-time evidence-chain continuity         |

## What's compilable today

| Compilable | File                                  |
|------------|---------------------------------------|
| Yes (built and run by `tools/determinism.py`) | `src/bitnet.mind` |
| Refused by design (governance negative control) | `examples/governance/rng_in_kernel.mind` |
| No (research direction) | `examples/tensor_parallel_attention.mind` |
| No (research direction) | `examples/pipeline_parallel_forward.mind` |

## How these skeletons fit the paper

The proposed paper has two core hypotheses:

- **H1**: Governance overhead is < 1% on BitNet workloads (vs 5-15% on FP16)
- **H4** (proposed addition): Cross-shard determinism is achievable on
  TP+PP at < 5% throughput overhead vs non-deterministic FP16 baseline

The skeletons in this directory are the spec for the H4 measurement.
Implementing them — adding the `mind.distributed.*` modules to mindc —
is the engineering scope STARGA contributes to the collaboration.

## What collaborators contribute

- Experimental design for measuring cross-shard determinism
- Hardware access (multi-GPU box for paper-grade validation)
- Production-pipeline mental model (what TP+PP actually looks like in
  vLLM / DeepSpeed / NVIDIA Triton)
- Paper writing and venue selection (MLSys / OSDI / NeurIPS workshop)

## Open questions

See `docs/parallel_pipeline.md` § "Open questions for collaboration".
