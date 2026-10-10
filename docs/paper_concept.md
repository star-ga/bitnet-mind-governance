# Paper concept

## Working title

*"Compile-time governance for ternary-weight LLMs: a measurement study of
BitNet under invariant constraints"*

## One-paragraph abstract (draft)

> Draft wording: the abstract states the hypotheses as if confirmed. None
> of the overhead results below has been measured yet.

Production deployments of LLMs in regulated domains (medical, legal,
financial, defense) require governance: guardrails on what the model is
permitted to emit, audit trails of every decision, and reproducible replay
across substrates. Current practice bolts governance onto the inference
loop at runtime, imposing 5–15% latency overhead on standard FP16 stacks.
We present a measurement study of compile-time governance enforcement on
BitNet b1.58 2B4T, a ~2B-parameter ternary-weight LLM, demonstrating that
governance overhead is statistically negligible (< 1%) on
bandwidth-bound ternary workloads. We further show that the combination
of ternary weights and Q16.16 fixed-point activations enables
**bit-identical replay** across x86_64, ARM64, and CUDA substrates — a
property unattainable on FP16 inference due to non-associative
floating-point reductions. Our results suggest a structural separation
between intelligence (which scales with parameters) and accountability
(which scales with substrate determinism), and indicate that ternary
substrates may be the natural fit for high-stakes regulated AI.

## Sections

1. **Introduction**
   - The accountability gap in LLM inference
   - Why FP16 cannot provide bit-identical replay
   - Compile-time vs runtime governance

2. **Background**
   - BitNet b1.58 architecture and ternary arithmetic
   - Q16.16 fixed-point and integer determinism
   - Existing governance approaches: NeMo Guardrails, Guardrails AI,
     prompt-level constraints

3. **Compile-time governance with MIND invariants**
   - The MIND systems language and `mindc` compiler
   - Invariants as compile-time refusals (not runtime checks)
   - The `no_float_in_forward_pass` invariant on ternary substrates

4. **Experimental setup**
   - Hardware: Xeon CPU, ARM64 SBC, T4/L4/P100 GPU
   - Models: BitNet b1.58 2B4T (baseline) + Llama 3.1 8B FP16 (contrast)
   - Three configurations: baseline / runtime guardrails / compile-time

5. **Results**
   - Latency overhead per configuration per hardware
   - Throughput and memory-bus contention analysis
   - Energy per token (where measurable)
   - Cross-architecture hash-parity validation
   - Statistical significance (Bonferroni-corrected)

6. **Discussion**
   - Why ternary is structurally amenable to governance
   - Implications for high-stakes regulated inference
   - When FP16 substrates remain preferable

7. **Related work**
   - Hardware-aware LLM inference
   - Formal verification of neural networks
   - Audit and provenance for ML systems

8. **Limitations**
   - Results are tied to a specific `mindc` release; the harness records
     the compiler version and CI pins the release archive by sha256
   - BitNet b1.58 is one ternary model class; other quantization schemes
     may behave differently
   - Determinism property depends on Q16.16 precision budget

9. **Conclusion**

## Headline contributions

1. First measurement study of compile-time governance overhead on a
   ternary-weight LLM
2. Demonstration of bit-identical cross-substrate LLM replay
3. The "accountability vs intelligence" structural argument: that high-stakes
   AI may be served better by deterministic ternary substrates than by
   higher-precision floating-point ones, even if the floating-point models
   are larger

## Calibrated confidence as a sidecar (future-AI gap #3)

A ternary-weight forward pass produces per-token logits and, separately, a
calibrated confidence/utility score for each token. That score is
**stochastic**, so the determinism harness treats it as a **sidecar
annotation: never a kernel operand, and excluded from the SHA-256 evidence
chain / byte-identity check**. The cross-arch bit-identity claim is anchored
only on the deterministic Q16.16 logit bytes; the confidence score may differ
in how a consumer uses it without ever perturbing the hash. This keeps the
governance-overhead measurement clean — the confidence layer adds annotation
cost, not determinism cost, and the paper can report the two separately.

## What needs to happen to write this paper

| Item | Owner | Status |
|---|---|---|
| Reference ternary x Q16.16 forward pass in MIND | STARGA | Runs (small synthetic model, `src/bitnet.mind`); no real checkpoint yet |
| Cross-architecture determinism harness | STARGA | Runs (`tools/determinism.py`); same root on x86_64 and aarch64 (emulated) |
| Cross-arch run on a real BitNet checkpoint (x86 + ARM + CUDA) | TBD | Not yet executed |
| Runtime guardrails wrapper | TBD | Not yet built |
| Statistical analysis pipeline | TBD | Not yet built |
| Energy measurement setup | TBD | Hardware-dependent |
| Paper writing | TBD | Not started |

## Estimated timeline

- 6 weeks: hardware access, baseline runs, harness validation
- 4 weeks: runtime guardrails integration, statistical analysis
- 4 weeks: paper writing and revision
- **Total**: 14 weeks from green-light to first submission

## Target venues (in order of fit)

1. MLSys
2. NeurIPS Workshop on Efficient ML
3. USENIX OSDI
4. ACM Trustworthy ML workshops
