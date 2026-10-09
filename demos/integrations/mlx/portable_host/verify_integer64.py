"""Verify 64-bit host storage and unchanged upstream clip and meshgrid tests."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import integer64_workloads
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import load_index
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TESTS = ("test_ops.TestOps.test_clip", "test_ops.TestOps.test_meshgrid")
NEGATIVE_CHECKS = {
    "missing": "No translated package for v_Absint64int64",
    "int16": "unary dispatch requires float32 arrays",
    "large-allocation": "exceeds its allocation",
    "reduction": "matching float32/int32/uint32 numeric arrays",
    "bitwise": "supported dtype",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = HostRuntime(
            args.packages,
            output / "dispatch.jsonl",
            integer64=args.integer64,
            absolute=args.absolute,
            reductions=args.reductions,
        )
        if args.worker == "missing":
            host.descriptors.pop("v_Absint64int64")
        host.install()
    if args.worker in NEGATIVE_CHECKS:
        dtype = "int16" if args.worker == "int16" else "int64"
        operand = mx.array(np.arange(3, dtype=dtype))
        if args.worker == "large-allocation":
            operand = mx.as_strided(operand, (65536,), (1,))
        try:
            result = (
                mx.sum(operand)
                if args.worker == "reduction"
                else (
                    mx.bitwise_and(operand, operand)
                    if args.worker == "bitwise"
                    else mx.abs(operand)
                )
            )
            mx.eval(result)
        except (ValueError, RuntimeError) as error:
            record = {
                "check": args.worker,
                "error": str(error),
                "dispatchCount": host.dispatch_count,
            }
            write_json(output / "rejection.json", record)
            if NEGATIVE_CHECKS[args.worker] not in str(error) or host.dispatch_count:
                raise RuntimeError(
                    "Integer64 rejection did not occur before dispatch"
                ) from error
            return
        raise RuntimeError("Unsupported integer64 operation was accepted")
    integer64_workloads.run(
        mx, np, host, lambda value: write_json(output / "results.json", value)
    )
    count = host.dispatch_count if host else 0
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    records = []
    for name in UPSTREAM_TESTS:
        start = host.dispatch_count if host else 0
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName(name)
        )
        records.append(
            {
                "test": name,
                "testsRun": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skips": len(result.skipped),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        write_json(
            output / "upstream.json", {"tests": records, "workloadDispatchCount": count}
        )
        if not result.wasSuccessful() or result.skipped:
            raise RuntimeError(
                "An unchanged upstream integer64 composition test failed"
            )


def validate_upstream(record, *, native):
    tests = record.get("tests", [])
    if len(tests) != len(UPSTREAM_TESTS):
        raise ValueError("Integer64 upstream test coverage is incomplete")
    for item, name in zip(tests, UPSTREAM_TESTS):
        expected = {"test": name, "testsRun": 1, "failures": 0, "errors": 0, "skips": 0}
        count = item.get("dispatchCount")
        if (
            any(item.get(key) != value for key, value in expected.items())
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError("Integer64 upstream test evidence differs")
    count = record.get("workloadDispatchCount")
    if type(count) is not int or (count <= 0 if native else count != 0):
        raise ValueError("Integer64 workload dispatch count differs")


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before, sources = verify_prepared(args.mlx_root), upstream_test_sources(
        args.mlx_root
    )
    write_json(output / "adaptation-before.json", before)
    packages = [
        (directory, json.loads((directory / "index.json").read_text()), False)
        for directory in (args.packages, args.integer64, args.absolute)
    ]
    target = packages[0][1]["target"]
    for directory in args.reductions:
        packages.append((directory, load_index(directory, target), True))
    reductions = {
        entry
        for _, index, variants in packages
        if variants
        for entry in index["descriptors"]
    }
    if not {"w32/all_reduce_andbool_", "w1/init_reduce_andbool_"} <= reductions:
        raise ValueError(
            "Upstream meshgrid requires Boolean whole and empty reductions"
        )
    results, upstream, rejections = {}, {}, {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "1800" if mode == "native" else "120",
            "--label",
            f"MLX integer64 {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_integer64",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--integer64",
            str(args.integer64.resolve()),
            "--absolute",
            str(args.absolute.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        for directory in args.reductions:
            command.extend(("--reductions", str(directory.resolve())))
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        write_json(
            output / f"{mode}.command.json",
            {"command": command, "returncode": process.returncode},
        )
        if process.returncode:
            raise RuntimeError(f"Integer64 {mode} worker failed; inspect {output}")
        if mode in NEGATIVE_CHECKS:
            record = json.loads((output / mode / "rejection.json").read_text())
            if (
                record.get("check") != mode
                or record.get("dispatchCount") != 0
                or NEGATIVE_CHECKS[mode] not in record.get("error", "")
            ):
                raise ValueError("Integer64 rejection evidence is incomplete")
            rejections[mode] = record
        else:
            results[mode] = json.loads((output / mode / "results.json").read_text())
            upstream[mode] = json.loads((output / mode / "upstream.json").read_text())
            validate_upstream(upstream[mode], native=mode == "native")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    count = upstream["native"]["workloadDispatchCount"]
    if count + sum(
        item["dispatchCount"] for item in upstream["native"]["tests"]
    ) != len(trace):
        raise ValueError("Integer64 trace count differs")
    integer64_workloads.validate(results["cpu"], [], native=False)
    integer64_workloads.validate(results["native"], trace[:count], native=True)
    verify_native_identity(trace, target)
    checked = 0
    for directory, index, variants in packages:
        events = [
            event
            for event in trace
            if (
                f'w{event["workgroupSize"][0]}/{event["entry"]}'
                if variants
                else event["entry"]
            )
            in index["descriptors"]
        ]
        checked += len(events)
        verify_artifacts(events, directory, index, variants=variants)
    if checked != len(trace):
        raise ValueError("Integer64 trace references an unknown or duplicate artifact")
    if before != verify_prepared(args.mlx_root) or sources != upstream_test_sources(
        args.mlx_root
    ):
        raise ValueError("MLX sources changed during integer64 verification")
    evidence = {
        "commit": COMMIT,
        "target": target,
        "adaptation": before,
        "upstreamTestSources": sources,
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "upstream": upstream,
        "negativeChecks": rejections,
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    write_json(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "integer64", "absolute", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append", required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
