"""Require real MLX evaluation through translated native array-creation kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host.packages import ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.portable_host.runtime import HostRuntime

UPSTREAM_TESTS = (
    "test_ops.TestOps.test_arange_overload_dispatch",
    "test_ops.TestOps.test_arange_inferred_dtype",
    "test_ops.TestOps.test_arange_corner_cases_cast",
)


def save(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if mx.metal.is_available():
        raise RuntimeError("This proof must not have a Metal backend")
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    runtime = None
    if args.worker != "cpu":
        runtime = HostRuntime(args.packages, output / "dispatch.jsonl")
        runtime.install()
        assert mx.default_device() == mx.gpu and mx.is_available(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    os.environ["DEVICE"] = "cpu" if args.worker == "cpu" else "gpu"
    if args.worker in {"unsupported", "over-limit", "missing"}:
        try:
            if args.worker == "unsupported":
                value = mx.abs(mx.array([-3.0, 2.0]), stream=mx.gpu)
            elif args.worker == "over-limit":
                value = mx.arange(65536, stream=mx.gpu)
            else:
                descriptor = runtime.descriptors["arangefloat32"]
                descriptor["artifact"]["packagePath"] = "artifacts/missing.glsl"
                value = mx.arange(2.0, 9.0, stream=mx.gpu)
            mx.eval(value)
        except (ValueError, RuntimeError) as error:
            message = str(error)
            expected = {
                "unsupported": "no GPU implementation",
                "over-limit": "65535",
                "missing": "artifact",
            }[args.worker]
            if expected not in message:
                raise
            save(output / "result.json", {"rejected": True, "message": message})
            return
        raise RuntimeError(f"{args.worker} unexpectedly executed")

    records = []
    for dtype in ("float32", "int32", "uint32", "int64", "uint64"):
        for count in (0, 1, 7, 257):
            expected = np.arange(2, 2 + 3 * count, 3, dtype=dtype)
            value = mx.arange(2, 2 + 3 * count, 3, dtype=getattr(mx, dtype))
            actual = np.array(value)
            np.testing.assert_array_equal(actual, expected)
            if str(actual.dtype) != dtype:
                raise RuntimeError("MLX readback dtype changed")
            records.append({"dtype": dtype, "count": count, "values": actual.tolist()})
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in UPSTREAM_TESTS
    )
    with (output / "upstream-tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if (
        not result.wasSuccessful()
        or result.testsRun != len(UPSTREAM_TESTS)
        or result.skipped
    ):
        raise RuntimeError(
            f"Unchanged upstream tests failed; inspect {output / 'upstream-tests.log'}"
        )
    save(
        output / "result.json",
        {"tests": result.testsRun, "skipped": len(result.skipped), "arrays": records},
    )


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    head = subprocess.check_output(
        ["git", "-C", str(args.mlx_root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if head != COMMIT:
        raise ValueError(f"Expected MLX {COMMIT}, got {head}")
    test_path = args.mlx_root / "python/tests/test_ops.py"
    pristine = subprocess.check_output(
        ["git", "-C", str(args.mlx_root), "show", "HEAD:python/tests/test_ops.py"],
        timeout=30,
    )
    if test_path.read_bytes() != pristine:
        raise ValueError("Upstream operation tests were modified")
    results = {}
    for mode in ("cpu", "native", "unsupported", "over-limit", "missing"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "180",
            "--label",
            f"MLX host {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        save(
            output / f"{mode}.command.json",
            {"command": command, "returncode": result.returncode},
        )
        if result.returncode:
            raise RuntimeError(
                f"Host proof {mode} failed; inspect {output / (mode + '.stderr')}"
            )
        results[mode] = json.loads((output / mode / "result.json").read_text())
    if results["cpu"] != results["native"]:
        raise RuntimeError("Original CPU and translated GPU host results differ")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    if {record["entry"] for record in trace} != set(ENTRIES):
        raise RuntimeError("Native trace does not cover every translated entry")
    index = json.loads((args.packages / "index.json").read_text())
    if any(record["target"] != index["target"] for record in trace):
        raise RuntimeError("Native trace used an unexpected target")
    evidence = {
        "commit": COMMIT,
        "target": index["target"],
        "upstreamTestSha256": hashlib.sha256(pristine).hexdigest(),
        "upstreamTests": list(UPSTREAM_TESTS),
        "original": results["cpu"],
        "translated": results["native"],
        "dispatchCount": len(trace),
        "entries": sorted({record["entry"] for record in trace}),
        "negativeChecks": {
            mode: results[mode] for mode in ("unsupported", "over-limit", "missing")
        },
        "fullUpstreamSuite": False,
        "fullTranslatedBackend": False,
    }
    save(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker", choices=["cpu", "native", "unsupported", "over-limit", "missing"]
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        verify(args)
