# BitNet + MIND Governance

A reference implementation and measurement harness for compile-time governance
on ternary-weight LLMs.

## What this is

This repository contains:

- A **700M-parameter BitNet b1.58 forward pass** written in pure MIND, using
  ternary weights `{-1, 0, +1}` and Q16.16 fixed-point activations
- A **cross-architecture determinism harness** that compiles the same `.mind`
  source for x86_64, ARM64, and CUDA targets, runs each binary against fixed
  inputs, and verifies that the SHA-256 evidence chain is bit-identical
- Reference implementations of `BitLinear` projection and ternary arithmetic
  primitives

## What this is for

A measurement-study paper concept on **compile-time governance overhead for
low-latency LLM inference**, comparing:

1. Bare BitNet inference (baseline)
2. BitNet + runtime guardrails (the typical pattern: NeMo Guardrails, Guardrails AI)
3. BitNet + MIND compile-time invariants (governance baked into the binary)

The hypothesis: governance overhead is statistically negligible on BitNet
workloads (vs 5–15% on FP16) because BitNet is memory-bandwidth-bound while
MIND invariant checks are CPU-bound — disjoint resources.

The secondary claim: **bit-identical replay across substrates** is achievable
with ternary × Q16.16, providing the deterministic substrate that current
FP16 inference cannot.

## Why ternary × Q16.16 is deterministic

Ternary weights `{-1, 0, +1}` collapse multiplication to addition, sign-flip,
or zero. There is no floating-point rounding, no reduction-order dependence,
no thermal-state drift. Combined with Q16.16 fixed-point activations
(integer-only `add`, `sub`, `shift`), the forward pass becomes a pure integer
computation — bit-identical across CPU, GPU, ARM, and x86.

This is the property that makes auditable, reproducible LLM inference
structurally possible.

## Quick start

```bash
git clone https://github.com/star-ga/bitnet-mind-governance
cd bitnet-mind-governance

# Install Python dependencies for the harness
pip install -r requirements.txt

# Run the cross-architecture determinism harness
cd tests
python test_determinism.py
```

If `mindc` is not installed locally, the harness will simulate the build
phase and verify the harness logic only. Real cross-arch validation requires
`mindc >= 0.2.4` (the first release with BitNet support; `0.2.5`
recommended for the stable `mic@1` IR contract) and target-architecture
access.

## Repository structure

```
bitnet-mind-governance/
├── README.md               (this file)
├── LICENSE                 (Apache 2.0)
├── METHODOLOGY.md          (governance overhead measurement protocol)
├── src/
│   ├── bitnet.mind         (700M BitNet forward pass, pure MIND)
│   ├── bitlinear.mind      (Q16.16 BitLinear projection)
│   └── ternary.mind        (ternary weight arithmetic primitives)
├── tests/
│   └── test_determinism.py (cross-architecture hash-parity harness)
├── docs/
│   ├── paper_concept.md    (paper outline and methodology)
│   └── citing.md           (BibTeX for citing this work)
└── requirements.txt
```

## Status

- BitNet forward pass in pure MIND: **scaffolded**, 235 lines
- Cross-architecture determinism harness: **scaffolded**, validation pending
  on actual cross-arch run
- Governance overhead measurement: **methodology drafted**, measurement TBD
- Toolchain: **tested against `mindc 0.2.5`** (current release; native
  BitNet support landed in `0.2.4`, Pratt parser + stable `mic@1` IR
  contract added in `0.2.5`, STARGA toolchain)

This is research-stage code intended for academic collaboration. The MIND
compiler (`mindc`) is STARGA-internal; binaries are reproducible with
appropriate access.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Citation

See [docs/citing.md](docs/citing.md). Once the paper is on arXiv, this section
will include a citable preprint reference.

## Contact

For research collaboration:
[STARGA Inc.](https://star.ga) — `ceo@star.ga`
