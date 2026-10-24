<p align="center">
  <img src="assets/logo.svg" alt="bitnet-mind-governance logo" width="96" height="96">
</p>

# BitNet + MIND Governance

A small, runnable reference for **deterministic ternary-weight inference with
compile-time governance**, written in [MIND](https://mindlang.dev), plus the
harness that checks it bit-for-bit.

Governance integration: MIND-Law — STARGA's implementation of the 512 constraint by J. M. Watson.

## What it is

- **`src/bitnet.mind`**: a BitNet-style forward pass in pure MIND. Weights
  are ternary `{-1, 0, +1}`, packed 2 bits each; activations are Q16.16
  fixed point. BitLinear projections, RMSNorm (exact integer square root),
  squared-ReLU FFN, residual adds and a ternary LM head are all integer-only,
  with reductions in fixed ascending order. Every recorded activation vector
  is folded into a SHA-256 evidence chain whose root the program prints.
- **`reference/bitnet_ref.py`**: an independent Python oracle that computes
  the same integers and the same chain with no shared code.
- **`tools/determinism.py`**: the harness. It compiles the MIND program with
  `mindc`, runs it, compares its output with the oracle and the pinned root in
  `results/expected.json`, checks the compiler's evidence artifact, and runs
  the governance negative control.
- **`examples/governance/rng_in_kernel.mind`**: a kernel marked
  `#[deterministic]` that draws from a PRNG; `mindc` must refuse to compile it.

The model is deliberately small and self-contained: hidden size 64, FFN 128,
2 layers, vocabulary 32, weights generated from a fixed seed by a 31-bit LCG.
It loads no real BitNet checkpoint. Inputs are single positions, so softmax
over one position is exactly 1 and attention reduces to `W_o * (W_v * h)`;
multi-position attention is not implemented.

## Status

| Item | State | Evidence |
|---|---|---|
| Ternary x Q16.16 forward pass in MIND | Runs, self-tests pass | `SELFTEST ... PASS` lines, exit code 0 |
| Output equals independent Python oracle | Verified | `cdylib` check in `tools/determinism.py` |
| Same evidence-chain root on x86_64 and aarch64 | Verified (aarch64 under QEMU, see below); CI compares native runners | `cross-arch` job in `.github/workflows/ci.yml` |
| Reproducible, verifiable compiler evidence artifact | Verified | `evidence` check (`mindc --emit-evidence`, `mindc verify`) |
| `#[deterministic]` refuses a PRNG call at compile time | Verified, with a positive control | `governance` check |
| Real BitNet b1.58 checkpoint (700M / 2B) | Not implemented | — |
| Multi-position attention, KV cache | Not implemented | — |
| CUDA / GPU run | Not measured | — |
| Governance-overhead measurement (paper H1/H2) | Methodology only | [METHODOLOGY.md](METHODOLOGY.md) |
| Tensor / pipeline parallel (paper H4) | Research direction, not compilable | [docs/parallel_pipeline.md](docs/parallel_pipeline.md) |

Pinned result, reproduced by the oracle on any machine:

```text
TOKEN t=1 argmax=9 logit=113230
TOKEN t=7 argmax=23 logit=93688
TOKEN t=19 argmax=14 logit=126190
TOKEN t=30 argmax=14 logit=116542
ROOT cd332f416ba263421b57f8f464f3bd70f479ee32bc39c43d3ead50b6669956cc
```

Logits are raw Q16.16 integers (113230 / 65536 = 1.7277...).

Recorded harness runs (`python3 tools/determinism.py`, all four checks PASS,
root as above):

| Host | Compiler | Notes |
|---|---|---|
| x86_64 Linux | published `mindc` v0.10.2 (linux-x64) | mic@3 `trace_hash` `c457cf7e...` |
| x86_64 Linux | newer `mindc` development build (reports 0.10.2) | `trace_hash` `d6c4124f...`; see [docs/mindc_alignment.md](docs/mindc_alignment.md) |
| aarch64 Linux, QEMU user-mode emulation (Ubuntu 24.04 container) | published `mindc` v0.10.2 (linux-arm64) + LLVM/MLIR 18 | `trace_hash` `c457cf7e...`, identical to x86_64. Under emulation clang exceeds mindc's fixed 60 s tool timeout at `-O3`, so this run forwarded clang at `-O0`; the root is unchanged. The CI arm64 job uses the normal `-O3` pipeline on a native runner. |

## Quick start

Requirements: Linux (x86_64 or aarch64), Python 3.12 or newer, a C toolchain
(`cc`, `ld`, `as`, `objcopy`; `build-essential` on Debian/Ubuntu), and
`mindc` 0.10.x. On both x86_64 and aarch64 the published `mindc` also calls
`mlir-opt`, `mlir-translate` and `clang` when it builds the cdylib (the
`std.sha256` substrate module is compiled through the MLIR pipeline; Ubuntu
24.04: `mlir-18-tools llvm-18 clang-18`, with `/usr/lib/llvm-18/bin` on
`PATH`, or set `MLIR_OPT`, `MLIR_TRANSLATE` and `CLANG` to the tool paths).

```bash
git clone https://github.com/star-ga/bitnet-mind-governance
cd bitnet-mind-governance
python3 -m pip install -r requirements.txt

# 1. Reference oracle only (no compiler needed)
python3 reference/bitnet_ref.py

# 2. Install mindc 0.10.2 from the public release (use linux-arm64 on aarch64)
curl -fsSLO https://github.com/star-ga/mind/releases/download/v0.10.2/mindc-linux-x64.tar.gz
echo "0c8beb4464f8c7d638a81c2dd8ad5eac81a0ddbc7556be711f00e029d01ef258  mindc-linux-x64.tar.gz" | sha256sum -c -
mkdir -p .mindc && tar -xzf mindc-linux-x64.tar.gz -C .mindc
export MINDC="$PWD/.mindc/mindc"

# 3. Full harness and tests
python3 tools/determinism.py --json-out report.json
python3 -m pytest -q
```

`tools/determinism.py` exits 0 only when every check passed; without a
compiler it runs the oracle alone and exits 2, so a missing toolchain never
reads as a pass. The tests skip (with a printed reason) the checks that need
`mindc` when it is not found.

To build the program directly:

```bash
"$MINDC" build --release --emit cdylib --out libbitnet.so
python3 -c "import ctypes, sys; sys.exit(ctypes.CDLL('./libbitnet.so').main())"
```

`mindc build --release` (an executable instead of a shared library) also
works, but needs the MIND CPU runtime library installed; the shared-library
path above does not.

## How it uses MIND

- **Q16.16 integer arithmetic.** MIND defines integer semantics exactly
  (wrapping two's complement, defined division and shifts), so the same
  source gives the same bits on every conforming target. The Python oracle
  additionally checks that no intermediate leaves the signed 64-bit range.
- **`#[deterministic]` call-graph check.** Every function in
  `src/bitnet.mind` except `main` (which prints) carries `#[deterministic]`.
  The type checker rejects such a function when it calls one of the
  compiler's known nondeterministic builtins, such as an unseeded PRNG draw.
  `examples/governance/rng_in_kernel.mind` is the negative control: it is
  refused with `determinism::nondeterministic_in_deterministic`, and the
  harness confirms that the same file without the annotation does not raise
  that diagnostic. The check has limits in the published 0.10.2 release
  (it does not follow every indirect call); see
  [docs/mindc_alignment.md](docs/mindc_alignment.md).
- **mic@3 evidence artifacts.** `mindc src/bitnet.mind --emit-evidence a.mic3`
  writes the canonical IR with an embedded evidence chain. The harness emits
  it twice, requires the two files to be byte-identical, and requires
  `mindc verify --require-deterministic --require-strict-fp` to accept it.
- **Standard library.** SHA-256 comes from MIND's `std.sha256`; output goes
  through `std.io`.

## Repository structure

```
bitnet-mind-governance/
├── Mind.toml                     MIND project manifest (entry: src/bitnet.mind)
├── src/bitnet.mind               ternary x Q16.16 forward pass + evidence chain
├── reference/bitnet_ref.py       independent Python oracle
├── tools/determinism.py          determinism + governance harness
├── tests/test_determinism.py     pytest suite (oracle known answers + harness checks)
├── results/expected.json         pinned output lines and evidence-chain root
├── examples/
│   ├── governance/rng_in_kernel.mind     must be refused by mindc
│   ├── tensor_parallel_attention.mind    research skeleton, not compilable
│   └── pipeline_parallel_forward.mind    research skeleton, not compilable
├── docs/
│   ├── mindc_alignment.md        toolchain notes for mindc 0.10.x
│   ├── paper_concept.md          proposed measurement paper
│   ├── parallel_pipeline.md      TP + PP research direction
│   └── citing.md                 BibTeX
├── METHODOLOGY.md                governance-overhead measurement protocol
├── pyproject.toml                ruff + pytest configuration
├── assets/logo.svg               project logo
└── .github/workflows/ci.yml      lint, tests and cross-arch root comparison
```

## Research context

The repository supports a proposed measurement study of compile-time
governance on ternary-weight models: bare inference vs. runtime guardrails
vs. compile-time invariants, plus cross-substrate bit-identical replay. The
hypotheses and protocol are in [METHODOLOGY.md](METHODOLOGY.md) and
[docs/paper_concept.md](docs/paper_concept.md). None of the overhead
hypotheses has been measured yet; the only measured claims are the ones in
the Status table above.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Links

- MIND language and docs: [mindlang.dev](https://mindlang.dev)
- MIND compiler (source and releases): [github.com/star-ga/mind](https://github.com/star-ga/mind)
- Citing this work: [docs/citing.md](docs/citing.md)
- Contact: [STARGA, Inc.](https://star.ga), `ceo@star.ga`
