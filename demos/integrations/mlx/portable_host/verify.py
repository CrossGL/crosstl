"""Require real MLX evaluation through translated native compute kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    copy_workloads,
    unary_workloads,
    view_workloads,
)
from demos.integrations.mlx.portable_host.packages import ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime

UPSTREAM_TESTS = (
    "test_ops.TestOps.test_arange_overload_dispatch",
    "test_ops.TestOps.test_arange_inferred_dtype",
    "test_ops.TestOps.test_arange_corner_cases_cast",
    "test_ops.TestOps.test_abs",
    "test_ops.TestOps.test_negative",
    "test_ops.TestOps.test_floor",
    "test_ops.TestOps.test_ceil",
    "test_ops.TestOps.test_square",
    "test_ops.TestOps.test_sqrt",
    "test_ops.TestOps.test_rsqrt",
    "test_ops.TestOps.test_exp",
    "test_ops.TestOps.test_expm1",
    "test_ops.TestOps.test_erf",
    "test_ops.TestOps.test_sin",
    "test_ops.TestOps.test_cos",
    "test_ops.TestOps.test_transpose_noargs",
    "test_ops.TestOps.test_transpose_axis",
    "test_ops.TestOps.test_broadcast",
    "test_ops.TestOps.test_split",
)
DTYPES = ("float32", "int32", "uint32", "int64", "uint64")
COUNTS = (0, 1, 7, 257)
NEGATIVE_CHECKS = {
    "unsupported": "no GPU implementation",
    "over-limit": "65535",
    "missing": "artifact",
    "unary-dtype": "float32",
    "unary-layout": "contiguous",
    "unary-over-limit": "65535",
    "copy-dtype": "matching float32, int32 or uint32",
    "copy-limit": "65535",
    "copy-allocation": "exceeds its allocation",
}


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
    if args.worker in NEGATIVE_CHECKS:
        try:
            if args.worker == "unsupported":
                value = mx.add(
                    mx.array([-3.0, 2.0]), mx.array([1.0, 1.0]), stream=mx.gpu
                )
            elif args.worker == "over-limit":
                value = mx.arange(65536, stream=mx.gpu)
            elif args.worker == "unary-dtype":
                value = mx.abs(mx.array([-3, 2], dtype=mx.int32), stream=mx.gpu)
            elif args.worker == "unary-layout":
                source = mx.as_strided(
                    mx.array(np.arange(8, dtype=np.float32)), (4,), (2,), stream=mx.gpu
                )
                value = mx.abs(source, stream=mx.gpu)
            elif args.worker == "unary-over-limit":
                value = mx.abs(
                    mx.array(np.ones(65536, dtype=np.float32)), stream=mx.gpu
                )
            elif args.worker == "copy-dtype":
                source = mx.array(np.arange(12, dtype=np.int64).reshape(3, 4))
                value = mx.reshape(mx.transpose(source), (12,), stream=mx.gpu)
            elif args.worker == "copy-limit":
                source = mx.array(np.arange(65792, dtype=np.float32).reshape(257, 256))
                value = mx.contiguous(mx.transpose(source), stream=mx.gpu)
            elif args.worker == "copy-allocation":
                source = mx.as_strided(mx.array([1.0, 2.0, 3.0]), (2,), (-1,))
                value = mx.contiguous(source, stream=mx.gpu)
            else:
                descriptor = runtime.descriptors["arangefloat32"]
                descriptor["artifact"]["packagePath"] = "artifacts/missing.glsl"
                value = mx.arange(2.0, 9.0, stream=mx.gpu)
            mx.eval(value)
        except (ValueError, RuntimeError) as error:
            message = str(error)
            expected = NEGATIVE_CHECKS[args.worker]
            if expected not in message:
                raise
            save(output / "result.json", {"rejected": True, "message": message})
            return
        raise RuntimeError(f"{args.worker} unexpectedly executed")

    records = []
    for dtype in DTYPES:
        for count in COUNTS:
            expected = np.arange(2, 2 + 3 * count, 3, dtype=dtype)
            value = mx.arange(2, 2 + 3 * count, 3, dtype=getattr(mx, dtype))
            actual = np.array(value)
            np.testing.assert_array_equal(actual, expected)
            if str(actual.dtype) != dtype:
                raise RuntimeError("MLX readback dtype changed")
            records.append({"dtype": dtype, "count": count, "values": actual.tolist()})
    unary = unary_workloads.run(mx, np)
    views = view_workloads.run(mx, np)
    copies = copy_workloads.run(mx, np)
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name) for name in UPSTREAM_TESTS
    )
    with (output / "upstream-tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    save(
        output / "result.json",
        {
            "tests": result.testsRun,
            "skipped": len(result.skipped),
            "failures": len(result.failures),
            "errors": len(result.errors),
            "arrays": records,
            "unary": unary,
            "views": views,
            "copies": copies,
        },
    )
    if (
        not result.wasSuccessful()
        or result.testsRun != len(UPSTREAM_TESTS)
        or result.skipped
    ):
        raise RuntimeError(
            f"Unchanged upstream tests failed; inspect {output / 'upstream-tests.log'}"
        )


def verify_results(result, *, cpu=False):
    if (
        type(result.get("tests")) is not int
        or result["tests"] != len(UPSTREAM_TESTS)
        or type(result.get("skipped")) is not int
        or result["skipped"] != 0
        or any(
            type(result.get(key)) is not int or result[key] != 0
            for key in ("failures", "errors")
        )
    ):
        raise RuntimeError("Incomplete upstream test results")
    expected = [
        {"dtype": dtype, "count": count, "values": [2 + 3 * i for i in range(count)]}
        for dtype in DTYPES
        for count in COUNTS
    ]
    if result.get("arrays") != expected or any(
        type(record["count"]) is not int
        or any(type(value) not in {int, float} for value in record["values"])
        for record in result["arrays"]
    ):
        raise RuntimeError("Incomplete or incorrect array readbacks")
    unary_workloads.validate(result.get("unary"), cpu=cpu)
    view_workloads.validate(result.get("views"))
    copy_workloads.validate(result.get("copies"))


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    adaptation = verify_prepared(args.mlx_root)
    save(output / "adaptation-before.json", adaptation)
    test_path = args.mlx_root / "python/tests/test_ops.py"
    pristine = subprocess.check_output(
        ["git", "-C", str(args.mlx_root), "show", "HEAD:python/tests/test_ops.py"],
        timeout=30,
    )
    if test_path.read_bytes() != pristine:
        raise ValueError("Upstream operation tests were modified")
    results = {}
    failed = []
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
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
            failed.append(mode)
        else:
            results[mode] = json.loads((output / mode / "result.json").read_text())
    if failed:
        raise RuntimeError(
            f"Host proof workers failed: {', '.join(failed)}; inspect {output}"
        )
    if results["cpu"]["arrays"] != results["native"]["arrays"]:
        raise RuntimeError("Original CPU and translated GPU host results differ")
    verify_results(results["cpu"], cpu=True)
    verify_results(results["native"])
    for mode, expected in NEGATIVE_CHECKS.items():
        if (
            results[mode].get("rejected") is not True
            or not isinstance(results[mode].get("message"), str)
            or expected not in results[mode]["message"]
        ):
            raise RuntimeError(f"Missing required rejection evidence: {mode}")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    if {record["entry"] for record in trace} != set(ENTRIES):
        raise RuntimeError("Native trace does not cover every translated entry")
    expected_dispatches = (
        [(f"arange{dtype}", count) for dtype in DTYPES for count in COUNTS if count]
        + unary_workloads.dispatches()
        + view_workloads.dispatches()
        + copy_workloads.dispatches()
    )
    if [(record["entry"], record.get("threads")) for record in trace][
        : len(expected_dispatches)
    ] != expected_dispatches or any(
        type(record.get("threads")) is not int or not 1 <= record["threads"] <= 65535
        for record in trace
    ):
        raise RuntimeError("Native trace has incomplete or invalid dispatch sizes")
    index = json.loads((args.packages / "index.json").read_text())
    if any(record["target"] != index["target"] for record in trace):
        raise RuntimeError("Native trace used an unexpected target")
    after = verify_prepared(args.mlx_root)
    save(output / "adaptation-after.json", after)
    if after != adaptation or test_path.read_bytes() != pristine:
        raise ValueError("MLX sources changed during execution")
    evidence = {
        "schemaVersion": 2,
        "commit": COMMIT,
        "adaptation": adaptation,
        "target": index["target"],
        "cpuReferenceProfile": unary_workloads.cpu_reference_profile(),
        "upstreamTestSha256": hashlib.sha256(pristine).hexdigest(),
        "upstreamTests": list(UPSTREAM_TESTS),
        "original": results["cpu"],
        "translated": results["native"],
        "dispatchCount": len(trace),
        "entries": sorted({record["entry"] for record in trace}),
        "negativeChecks": {mode: results[mode] for mode in NEGATIVE_CHECKS},
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
    parser.add_argument("--worker", choices=["cpu", "native", *NEGATIVE_CHECKS])
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        verify(args)
