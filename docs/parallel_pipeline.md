# Pipeline & Tensor Parallelism in MIND — Research Direction

## What's true today (mindc 0.2.5)

`mindc` compiles to **single-device** targets (CPU, single GPU, Pi/edge). It
does not yet emit multi-device tensor-parallel (TP) or pipeline-parallel (PP)
plans as built-in primitives. There is no NCCL binding, no all-reduce, no
distributed scheduler in the compiler today.

This is an honest gap. **It is also the paper-shaped problem.**

## Why this gap is the contribution, not a blocker

When a 70B-class transformer is sharded across 3 GPUs using standard
TP+PP (vLLM, DeepSpeed, NVIDIA Triton, HuggingFace Accelerate), the
all-reduce step across shards is **non-deterministic by reduction order**.
Different runs produce slightly different logits because IEEE-754 floating-
point reductions are not associative under parallel scheduling.

In production this is tolerated. For audit, regulatory, and replay-required
workloads, it is a blocker. The existing solution to the memory problem
*creates* a reproducibility problem.

A deterministic TP+PP scheduler emitted by `mindc` over Q16.16 + ternary
weights solves both:

- **Memory**: same shard-then-merge structure as TP/PP today
- **Reproducibility**: integer-only math + compile-time-fixed reduction
  order → bit-identical replay across shards, runs, and hardware

## Architecture (research scope — not implemented)

The skeleton below is what the `.mind` source for a deterministic TP+PP
plan could look like. **It does not compile under mindc 0.2.5.** It is a
specification for the methods contribution of the proposed paper.

### Tensor-parallel attention head shard

```mind
[invariant deterministic_all_reduce]
[invariant reduction_order_lexicographic]
module attention_tp_shard {
    use mind.tensor.q16_16
    use mind.distributed.shard      // proposed: not in 0.2.5
    use mind.distributed.allreduce  // proposed: not in 0.2.5

    // World: 3 shards across 3 devices.
    // Shard k owns heads [k * HEADS_PER_SHARD : (k+1) * HEADS_PER_SHARD].
    const NUM_SHARDS:        u32 = 3u32
    const HEADS_PER_SHARD:   u32 = NUM_HEADS / NUM_SHARDS

    fn attention_shard(
        shard_id:  u32,
        x:         &[q16_16],          // local hidden state
        w_q_local: &TernaryMatrix,     // sharded query proj
        w_k_local: &TernaryMatrix,
        w_v_local: &TernaryMatrix,
        w_o:       &TernaryMatrix,     // output proj — full
    ) -> Vec<q16_16> {
        // 1. Local Q/K/V projection — no cross-shard work
        let q_local = ternary_matmul(w_q_local, x)
        let k_local = ternary_matmul(w_k_local, x)
        let v_local = ternary_matmul(w_v_local, x)

        // 2. Local attention over heads owned by this shard
        let head_out_local = local_attention(q_local, k_local, v_local)

        // 3. All-gather: collect all shards' head outputs
        //    Lexicographic gather order: shard 0, 1, 2, ... — fixed at compile time
        [invariant gather_order_lexicographic]
        let head_out_full = allgather_lexicographic(head_out_local)

        // 4. Output projection — full matrix, sharded only on input
        ternary_matmul(w_o, head_out_full)
    }
}
```

**Key invariants enforced by `mindc` (proposed):**
- `deterministic_all_reduce`: refuses to emit if any reduction across
  shards uses non-lexicographic order
- `reduction_order_lexicographic`: applies to local matmuls (already
  enforced in 0.2.5) and extends to cross-device gathers
- `gather_order_lexicographic`: shard contributions concatenated in
  fixed shard-id order, never timestamp-arrival order

### Pipeline-parallel forward pass

```mind
[invariant evidence_chain_continuous]
module pipeline_forward {
    use mind.distributed.pipeline    // proposed: not in 0.2.5
    use evidence_emitter

    // Stages: 24 transformer layers split into 3 stages of 8 layers each.
    const NUM_STAGES:        u32 = 3u32
    const LAYERS_PER_STAGE:  u32 = NUM_LAYERS / NUM_STAGES

    [pipeline_stage(0)]
    fn stage0(x: &[q16_16]) -> Vec<q16_16> {
        let mut h = x.clone()
        let mut layer: u32 = 0u32
        while layer < LAYERS_PER_STAGE {
            h = transformer_block(layer, &h)
            // Emit evidence at every layer boundary
            evidence_emit("layer_boundary", layer, hash::sha256(&h))
            layer += 1u32
        }
        h
    }

    [pipeline_stage(1)]
    fn stage1(h_in: &[q16_16]) -> Vec<q16_16> {
        // Receive from stage 0 — verify evidence chain continuous
        verify_evidence_received("stage0_output")
        let mut h = h_in.clone()
        let mut layer: u32 = LAYERS_PER_STAGE
        while layer < 2u32 * LAYERS_PER_STAGE {
            h = transformer_block(layer, &h)
            evidence_emit("layer_boundary", layer, hash::sha256(&h))
            layer += 1u32
        }
        h
    }

    [pipeline_stage(2)]
    fn stage2(h_in: &[q16_16]) -> Vec<q16_16> {
        verify_evidence_received("stage1_output")
        let mut h = h_in.clone()
        let mut layer: u32 = 2u32 * LAYERS_PER_STAGE
        while layer < NUM_LAYERS {
            h = transformer_block(layer, &h)
            evidence_emit("layer_boundary", layer, hash::sha256(&h))
            layer += 1u32
        }
        h
    }
}
```

**Key invariants enforced by `mindc` (proposed):**
- `evidence_chain_continuous`: each stage must verify the previous
  stage's evidence emit before processing — refuses to compile if
  evidence chain is bypassed
- Pipeline boundaries become natural audit points
- The merkle root of all stage boundaries forms the inference attestation

## Comparison: existing approach vs. proposed

| Property                         | vLLM / DeepSpeed TP+PP | mindc deterministic TP+PP (proposed) |
|----------------------------------|------------------------|--------------------------------------|
| Memory (shards 70B across N GPUs)| Yes                    | Yes                                  |
| Cross-shard determinism          | No (FP reduction race) | Yes (lexicographic + integer)        |
| Bit-identical replay across runs | No                     | Yes                                  |
| Cross-hardware bit-identicality  | No (BF16 vs FP16)      | Yes (Q16.16 invariant)               |
| Per-layer evidence chain         | No                     | Yes (compile-time enforced)          |
| Works on FP16 transformers       | Yes                    | No (BitNet ternary required)         |
| Production-ready today           | Yes                    | No (research direction)              |

The asymmetry is real: production teams accept non-determinism because
they have to. Audit-required teams cannot.

## Engineering scope to reach proof-of-concept

Six to eight weeks of focused work, scoped for a paper-validation prototype:

1. **Add `mind.distributed.shard` + `mind.distributed.allreduce` modules
   to mindc** (~3 weeks)
   - Direct CUDA peer-to-peer for proof-of-concept (no NCCL needed)
   - Or shared-memory IPC for single-machine multi-process validation
   - Lexicographic-order all-reduce, all-gather primitives

2. **Add `mind.distributed.pipeline` with `[pipeline_stage(N)]` attribute**
   (~2 weeks)
   - Send/recv between stages
   - Evidence-chain continuity verification at stage boundaries

3. **Port `bitnet.mind` to use the new primitives** (~1 week)
   - Re-shard along attention heads (TP)
   - Split layer groups into stages (PP)
   - Combine TP+PP for 3-way memory split

4. **Cross-shard determinism harness** (~1 week)
   - Run same input through 3-shard TP+PP, capture per-stage merkle roots
   - Compare to single-device run — must match bit-identically
   - Repeat across (x86 host + 3× consumer GPU) and (ARM host + 3× different
     GPU) for cross-hardware validation

This is the methodological contribution to the proposed paper.

## What this gives the proposed paper

Strengthens the H1 hypothesis (governance overhead < 1% on BitNet) by
adding a second hypothesis:

**H4 (proposed): Cross-shard determinism is achievable on TP+PP at the
cost of < 5% throughput overhead vs. non-deterministic FP16 baseline.**

Two measurable claims, both testable, both novel. Together this is the
shape of an MLSys / OSDI methods paper.

## What's not in scope

- NCCL integration — defer to production hardening after paper validation
- Multi-machine clusters — single-machine multi-GPU is sufficient for
  paper-grade demonstration
- Auto-sharding / search over shard topologies — manual specification
  is fine for the proof-of-concept

## Open questions for collaboration

1. Is the all-reduce lexicographic order over shard-id sufficient, or
   do we need a full deterministic-collective-communication library?
2. How do we handle KV cache distribution across pipeline stages
   without breaking the evidence chain on multi-token decode?
3. What's the minimum set of distributed primitives that need to be
   added to `mindc` to unblock the paper measurement, vs what's nice-to-have?

These are the questions the methods paper should answer.
