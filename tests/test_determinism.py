"""Tests for the BitNet + MIND determinism harness.

The reference tests always run. The compiler tests need `mindc` (from
$MINDC or PATH) and are skipped, with the reason printed, when it is absent.
Run:  python3 -m pytest -q tests
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))
sys.path.insert(0, str(REPO / "tools"))

import bitnet_ref as ref  # noqa: E402
import determinism  # noqa: E402

MINDC = determinism.resolve_mindc(None)
needs_mindc = pytest.mark.skipif(MINDC is None, reason="mindc not found ($MINDC or PATH)")


# --- Reference kernels (hand-computed known answers) -----------------------


def test_bitlinear_known_row() -> None:
    w = ref.Ternary(1, 8, (1, -1, 0, 1, -1, -1, 0, 1))
    x = [(1 << k) * ref.Q_ONE for k in range(8)]
    assert ref.bitlinear(w, ref.Q_ONE, x) == [87 * ref.Q_ONE]


def test_bitlinear_scale_is_q16_multiply() -> None:
    w = ref.Ternary(2, 2, (1, 1, -1, 0))
    x = [3 * ref.Q_ONE, ref.Q_ONE]
    # row 0: (3 + 1) * 0.5 = 2.0 ; row 1: -3 * 0.5 = -1.5
    assert ref.bitlinear(w, ref.Q_ONE // 2, x) == [2 * ref.Q_ONE, -3 * ref.Q_ONE // 2]


def test_rms_norm_of_constant_vector_is_one() -> None:
    x = [2 * ref.Q_ONE] * 4
    y = ref.rms_norm(x)
    assert all(abs(v - ref.Q_ONE) <= 2 for v in y)


def test_relu2() -> None:
    assert ref.relu2(-5 * ref.Q_ONE) == 0
    assert ref.relu2(0) == 0
    assert ref.relu2(3 * ref.Q_ONE) == 9 * ref.Q_ONE


def test_i64_guard_rejects_overflow() -> None:
    with pytest.raises(OverflowError):
        ref._i64(1 << 63)


def test_chain_record_layout() -> None:
    prev = bytes(32)
    got = ref.chain_record(prev, 2, [1, -1])
    payload = bytes([2]) + (1).to_bytes(8, "little") + (-1).to_bytes(8, "little", signed=True)
    assert got == hashlib.sha256(prev + payload).digest()


def test_generated_weights_are_ternary() -> None:
    model = ref.build_model()
    mats = [model.lm_head] + [m for layer in model.layers for m in vars(layer).values()]
    for m in mats:
        assert set(m.weights) <= {-1, 0, 1}
        assert len(m.weights) == m.rows * m.cols


# --- Pinned result ----------------------------------------------------------


def test_reference_matches_pinned_root() -> None:
    expected = json.loads((REPO / "results" / "expected.json").read_text())
    report = ref.run()
    assert report.root == expected["root"]
    assert report.lines() == expected["lines"]


def test_root_depends_on_every_token() -> None:
    """Positive control: changing the input must change the root."""
    model = ref.build_model()
    chain_a, chain_b = bytes(32), bytes(32)
    for t in ref.TOKENS:
        _, chain_a = ref.forward(model, t, chain_a)
    for t in (*ref.TOKENS[:-1], ref.TOKENS[-1] - 1):
        _, chain_b = ref.forward(model, t, chain_b)
    assert chain_a != chain_b


def test_parse_output() -> None:
    text = "SELFTEST sha256 PASS\nSELFTEST isqrt FAIL\nTOKEN t=1 argmax=2 logit=3\nROOT ab\n"
    selftests, lines = determinism.parse_output(text)
    assert selftests == {"sha256": True, "isqrt": False}
    assert lines == ["TOKEN t=1 argmax=2 logit=3", "ROOT ab"]


# --- Compiled MIND program (needs mindc) ------------------------------------


@needs_mindc
def test_compiled_program_matches_reference(tmp_path: Path) -> None:
    check = determinism.check_cdylib(MINDC, tmp_path, ref.run().lines())
    assert check.passed, check.detail


@needs_mindc
def test_evidence_artifact_is_reproducible_and_verifies(tmp_path: Path) -> None:
    check = determinism.check_evidence(MINDC, tmp_path)
    assert check.passed, check.detail
    assert check.detail["determinism"] == "deterministic"
    assert check.detail["fp_mode"] == "strict"


@needs_mindc
def test_nondeterministic_kernel_is_refused(tmp_path: Path) -> None:
    check = determinism.check_governance(MINDC, tmp_path)
    assert check.passed, check.detail
