#!/usr/bin/env python3
"""Determinism harness for src/bitnet.mind.

What it checks, on whatever host it runs on:

1. reference   - reference/bitnet_ref.py (pure Python) produces the pinned
                 evidence-chain root in results/expected.json.
2. cdylib      - `mindc build --emit cdylib` compiles src/bitnet.mind; the
                 library's main() passes every in-binary self-test and prints
                 exactly the reference TOKEN/ROOT lines.
3. evidence    - two `mindc --emit-evidence` runs produce byte-identical
                 mic@3 artifacts, and `mindc verify --require-deterministic
                 --require-strict-fp` accepts them.
4. governance  - examples/governance/rng_in_kernel.mind is refused at compile
                 time with the `#[deterministic]` call-graph diagnostic.

Running the harness on two machines (for example an x86_64 and an aarch64
host) and comparing the `root` fields of their JSON reports is the
cross-architecture check; .github/workflows/ci.yml does that on every push.

Usage:
    python3 tools/determinism.py [--mindc PATH] [--json-out FILE]

The compiler is taken from --mindc, then $MINDC, then `mindc` on PATH.
Exit status is 0 only when every check that ran passed. Without a compiler
only check 1 runs and the exit status is 2, so a missing toolchain can never
read as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reference"))

import bitnet_ref  # noqa: E402  (path set up above)

EXPECTED = REPO / "results" / "expected.json"
NEGATIVE_EXAMPLE = REPO / "examples" / "governance" / "rng_in_kernel.mind"
GOVERNANCE_CODE = "determinism::nondeterministic_in_deterministic"
SELFTESTS = ("sha256", "isqrt", "linear", "reserved", "model")
TIMEOUT = int(os.environ.get("BITNET_HARNESS_TIMEOUT", "900"))

LOADER = (
    "import ctypes, sys\n"
    "lib = ctypes.CDLL(sys.argv[1])\n"
    "lib.main.restype = ctypes.c_int64\n"
    "lib.main.argtypes = []\n"
    "sys.exit(int(lib.main()))\n"
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)


@dataclass
class Report:
    host: dict
    mindc: str | None
    root: str | None
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, check=False
    )


def resolve_mindc(explicit: str | None) -> str | None:
    candidate = explicit or os.environ.get("MINDC") or shutil.which("mindc")
    if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
        return str(Path(candidate).resolve())
    return None


def mindc_version(mindc: str) -> str:
    out = run([mindc, "--version"]).stdout.strip().splitlines()
    return out[0] if out else "unknown"


def stage_project(dst: Path) -> Path:
    """Copy Mind.toml + src/ into dst so builds never write into the checkout."""
    dst.mkdir(parents=True)
    shutil.copy2(REPO / "Mind.toml", dst / "Mind.toml")
    shutil.copytree(REPO / "src", dst / "src")
    return dst


def parse_output(text: str) -> tuple[dict[str, bool], list[str]]:
    selftests: dict[str, bool] = {}
    lines: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "SELFTEST":
            selftests[parts[1]] = parts[2] == "PASS"
        elif parts and parts[0] in ("TOKEN", "ROOT"):
            lines.append(line)
    return selftests, lines


def check_reference() -> tuple[Check, list[str], str]:
    report = bitnet_ref.run()
    expected = json.loads(EXPECTED.read_text())
    passed = report.root == expected["root"] and report.lines() == expected["lines"]
    detail = {"root": report.root, "expected_root": expected["root"]}
    return Check("reference", passed, detail), report.lines(), report.root


def check_cdylib(mindc: str, work: Path, ref_lines: list[str]) -> Check:
    proj = stage_project(work / "cdylib")
    lib = work / "libbitnet_mind_governance.so"
    build = run(
        [mindc, "build", "--release", "--no-cache", "--emit", "cdylib", "--out", str(lib)],
        cwd=proj,
    )
    if build.returncode != 0 or not lib.is_file():
        tail = (build.stdout + build.stderr)[-1500:]
        return Check("cdylib", False, {"stage": "build", "rc": build.returncode, "log": tail})
    exe = run([sys.executable, "-c", LOADER, str(lib)])
    selftests, lines = parse_output(exe.stdout)
    detail = {
        "rc": exe.returncode,
        "selftests": selftests,
        "lines": lines,
        "lines_match_reference": lines == ref_lines,
    }
    passed = (
        exe.returncode == 0
        and all(selftests.get(name) is True for name in SELFTESTS)
        and lines == ref_lines
    )
    return Check("cdylib", passed, detail)


def check_evidence(mindc: str, work: Path) -> Check:
    proj = stage_project(work / "evidence")
    src = proj / "src" / "bitnet.mind"
    artifacts = [work / "a.mic3", work / "b.mic3"]
    for art in artifacts:
        emit = run([mindc, str(src), "--emit-evidence", str(art)], cwd=proj)
        if emit.returncode != 0 or not art.is_file():
            tail = (emit.stdout + emit.stderr)[-1500:]
            return Check("evidence", False, {"stage": "emit", "rc": emit.returncode, "log": tail})
    identical = artifacts[0].read_bytes() == artifacts[1].read_bytes()
    verify = run(
        [mindc, "verify", str(artifacts[0]), "--require-deterministic", "--require-strict-fp"]
    )
    fields = {}
    for line in verify.stdout.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in ("determinism", "fp_mode", "trace_hash", "trace_hash_valid"):
            fields[key.strip()] = value.strip()
    detail = {
        "artifacts_identical": identical,
        "verify_rc": verify.returncode,
        **fields,
    }
    return Check("evidence", identical and verify.returncode == 0, detail)


def check_governance(mindc: str, work: Path) -> Check:
    """Negative control: a #[deterministic] kernel that draws randomness.

    The annotated file must be refused with GOVERNANCE_CODE and write no
    artifact. A copy with the annotation stripped must NOT carry that
    diagnostic, which shows the refusal comes from the annotation.
    """
    out = work / "rejected.mic3"
    res = run([mindc, str(NEGATIVE_EXAMPLE), "--emit-evidence", str(out)], cwd=work)
    text = res.stdout + res.stderr

    stripped = work / "rng_in_kernel_unannotated.mind"
    stripped.write_text(NEGATIVE_EXAMPLE.read_text().replace("#[deterministic]\n", ""))
    control = run([mindc, str(stripped), "--emit-evidence", str(work / "control.mic3")], cwd=work)
    control_text = control.stdout + control.stderr

    refused = res.returncode != 0 and GOVERNANCE_CODE in text and not out.exists()
    control_ok = GOVERNANCE_CODE not in control_text
    return Check(
        "governance",
        refused and control_ok,
        {
            "rc": res.returncode,
            "diagnostic_present": GOVERNANCE_CODE in text,
            "artifact_written": out.exists(),
            "unannotated_copy_has_diagnostic": not control_ok,
        },
    )


def build_report(mindc: str | None) -> Report:
    host = {"machine": platform.machine(), "system": platform.system()}
    ref_check, ref_lines, root = check_reference()
    checks = [ref_check]
    if mindc is None:
        return Report(host, None, root, checks)
    with tempfile.TemporaryDirectory(prefix="bitnet_harness_") as tmp:
        work = Path(tmp)
        checks.append(check_cdylib(mindc, work, ref_lines))
        checks.append(check_evidence(mindc, work))
        checks.append(check_governance(mindc, work))
    return Report(host, mindc_version(mindc), root, checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mindc", help="path to the mindc compiler")
    parser.add_argument("--json-out", help="write the report as JSON to this file")
    args = parser.parse_args(argv)

    mindc = resolve_mindc(args.mindc)
    report = build_report(mindc)

    for check in report.checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}")
    print(f"host  {report.host['machine']} {report.host['system']}")
    print(f"mindc {report.mindc or 'not found'}")
    print(f"root  {report.root}")

    if args.json_out:
        payload = asdict(report)
        payload["ok"] = report.ok
        payload.pop("mindc", None)
        payload["mindc_version"] = report.mindc
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")

    if mindc is None:
        print("mindc not found: only the Python reference ran (exit 2)", file=sys.stderr)
        return 2
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
