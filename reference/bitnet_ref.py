#!/usr/bin/env python3
"""Independent Python oracle for src/bitnet.mind.

This file re-implements, in plain Python integers, exactly the arithmetic
that src/bitnet.mind performs: ternary x Q16.16 BitLinear projections,
Q16.16 RMSNorm with an exact integer square root, squared-ReLU, residual
adds, a ternary LM head, and a SHA-256 evidence chain over every recorded
activation vector.

It shares no code with the MIND program. If the compiled MIND binary and
this oracle print the same evidence-chain root, the binary computed every
recorded activation bit-for-bit as specified.

Every intermediate value is range-checked against the signed 64-bit range,
so a silent wrap in the i64 MIND program can never be mistaken for a match.

Usage:
    python3 reference/bitnet_ref.py          # prints the same report as the binary
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass

# --- Model topology (must match src/bitnet.mind) ---------------------------

HIDDEN = 64  # d_model
FFN = 128  # FFN inner dimension
LAYERS = 2  # decoder blocks
VOCAB = 32  # LM-head rows
TOKENS = (1, 7, 19, 30)

SEED = 0xDEADBEEFC0FFEE & 0x7FFFFFFF  # fixed seed from METHODOLOGY.md

Q_ONE = 65536  # 1.0 in Q16.16
SCALE_D = 8192  # 1/sqrt(64)  in Q16.16 (exact)
SCALE_F = 5793  # 1/sqrt(128) in Q16.16 (rounded)
EPS_RAW = 1  # RMSNorm epsilon, raw Q16.16 units

TAG_EMBED = 1
TAG_ATTN = 2
TAG_FFN = 3
TAG_LOGITS = 4

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def _i64(v: int) -> int:
    """Return v unchanged, failing loudly if it left the signed 64-bit range."""
    if not I64_MIN <= v <= I64_MAX:
        raise OverflowError(f"value {v} leaves the i64 range")
    return v


# --- Deterministic parameter generation ------------------------------------


def lcg_next(s: int) -> int:
    """31-bit LCG; every state is non-negative, so no shift-sign ambiguity."""
    return (s * 1103515245 + 12345) & 0x7FFFFFFF


@dataclass(frozen=True)
class Ternary:
    rows: int
    cols: int
    weights: tuple[int, ...]  # row-major, each in {-1, 0, +1}


def gen_ternary(s: int, rows: int, cols: int) -> tuple[Ternary, int]:
    out = []
    for _ in range(rows * cols):
        s = lcg_next(s)
        out.append(((s >> 16) % 3) - 1)
    return Ternary(rows, cols, tuple(out)), s


def gen_embedding(s: int) -> tuple[tuple[tuple[int, ...], ...], int]:
    table = []
    for _ in range(VOCAB):
        row = []
        for _ in range(HIDDEN):
            s = lcg_next(s)
            row.append(((s >> 8) & 131071) - 65536)  # [-1.0, 1.0) in Q16.16
        table.append(tuple(row))
    return tuple(table), s


@dataclass(frozen=True)
class Layer:
    wv: Ternary
    wo: Ternary
    w_up: Ternary
    w_down: Ternary


@dataclass(frozen=True)
class Model:
    embedding: tuple[tuple[int, ...], ...]
    layers: tuple[Layer, ...]
    lm_head: Ternary


def build_model() -> Model:
    s = SEED
    embedding, s = gen_embedding(s)
    layers = []
    for _ in range(LAYERS):
        wv, s = gen_ternary(s, HIDDEN, HIDDEN)
        wo, s = gen_ternary(s, HIDDEN, HIDDEN)
        w_up, s = gen_ternary(s, FFN, HIDDEN)
        w_down, s = gen_ternary(s, HIDDEN, FFN)
        layers.append(Layer(wv, wo, w_up, w_down))
    lm_head, s = gen_ternary(s, VOCAB, HIDDEN)
    return Model(embedding, tuple(layers), lm_head)


# --- Kernels ----------------------------------------------------------------


def bitlinear(w: Ternary, scale: int, x: list[int]) -> list[int]:
    """y[i] = scale * (sum_{w=+1} x[j] - sum_{w=-1} x[j]), j ascending."""
    y = []
    for i in range(w.rows):
        acc = 0
        base = i * w.cols
        for j in range(w.cols):
            code = w.weights[base + j]
            if code == 1:
                acc = _i64(acc + x[j])
            elif code == -1:
                acc = _i64(acc - x[j])
        y.append(_i64(acc * scale) >> 16)
    return y


def rms_norm(x: list[int]) -> list[int]:
    sumsq = 0
    for v in x:
        sumsq = _i64(sumsq + (_i64(v * v) >> 16))
    mean = sumsq // len(x) + EPS_RAW  # sumsq >= 0: floor == trunc
    rms = math.isqrt(_i64(mean << 16))  # sqrt(Q32.32) -> Q16.16
    inv = (1 << 32) // rms  # 1/rms in Q16.16
    return [_i64(v * inv) >> 16 for v in x]


def relu2(v: int) -> int:
    return (_i64(v * v) >> 16) if v > 0 else 0


def residual_add(x: list[int], d: list[int]) -> list[int]:
    return [_i64(a + b) for a, b in zip(x, d, strict=True)]


# --- Evidence chain ---------------------------------------------------------


def chain_record(prev: bytes, tag: int, values: list[int]) -> bytes:
    payload = bytes([tag]) + b"".join(v.to_bytes(8, "little", signed=True) for v in values)
    return hashlib.sha256(prev + payload).digest()


@dataclass(frozen=True)
class TokenResult:
    token: int
    argmax: int
    logit: int


def forward(model: Model, token: int, chain: bytes) -> tuple[TokenResult, bytes]:
    x = list(model.embedding[token])
    chain = chain_record(chain, TAG_EMBED, x)
    for layer in model.layers:
        h = rms_norm(x)
        # Sequence length 1: softmax over a single position is exactly 1, so
        # attention reduces to W_o * (W_v * h).
        attn = bitlinear(layer.wo, SCALE_D, bitlinear(layer.wv, SCALE_D, h))
        x = residual_add(x, attn)
        chain = chain_record(chain, TAG_ATTN, x)
        h2 = rms_norm(x)
        up = [relu2(v) for v in bitlinear(layer.w_up, SCALE_D, h2)]
        x = residual_add(x, bitlinear(layer.w_down, SCALE_F, up))
        chain = chain_record(chain, TAG_FFN, x)
    logits = bitlinear(model.lm_head, SCALE_D, rms_norm(x))
    chain = chain_record(chain, TAG_LOGITS, logits)
    best = 0
    for k in range(1, len(logits)):
        if logits[k] > logits[best]:
            best = k
    return TokenResult(token, best, logits[best]), chain


@dataclass(frozen=True)
class Report:
    tokens: tuple[TokenResult, ...]
    root: str

    def lines(self) -> list[str]:
        out = [f"TOKEN t={r.token} argmax={r.argmax} logit={r.logit}" for r in self.tokens]
        out.append(f"ROOT {self.root}")
        return out


def run() -> Report:
    model = build_model()
    chain = bytes(32)
    results = []
    for t in TOKENS:
        r, chain = forward(model, t, chain)
        results.append(r)
    return Report(tuple(results), chain.hex())


def main() -> int:
    for line in run().lines():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
