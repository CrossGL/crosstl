"""Verify general Gather through unchanged MLX Python operations and tests."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    gather_axis_workloads,
    gather_capacity_workloads,
    gather_evidence,
    gather_workloads,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources

UPSTREAM_TESTS = ("test_ops.TestOps.test_take",)
AXIS_UPSTREAM_TESTS = ("test_ops.TestOps.test_take_along_axis",)


def selected_workloads(*, axis=False, capacity=False):
    if axis and capacity:
        raise ValueError("Capacity workloads exercise general gather, not GatherAxis")
    return (
        gather_axis_workloads
        if axis
        else gather_capacity_workloads if capacity else gather_workloads
    )


def worker(args):
    import mlx.core as mx
    import numpy as np

    workloads = selected_workloads(axis=args.axis, capacity=args.capacity)
    upstream_tests = AXIS_UPSTREAM_TESTS if args.axis else UPSTREAM_TESTS
    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "native":
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            mlx_root=args.mlx_root,
            reductions=args.reductions,
            integer64=args.integer64,
        )
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    records = []
    for case in workloads.cases():
        start = host.dispatch_count if host else 0
        source, result = workloads.expression(mx, np, case)
        expected_source, _, expected = workloads.reference(np, case)
        actual = np.array(result)
        record = {
            **case,
            "actual": gather_workloads.words(np, actual),
            "expected": gather_workloads.words(np, expected),
            "shape": list(actual.shape),
            "dtype": str(result.dtype),
            "inputUnchanged": (
                gather_workloads.words(np, np.array(source))
                == gather_workloads.words(np, expected_source)
            ),
            "dispatchCount": host.dispatch_count - start if host else 0,
            "dispatchStart": start,
        }
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
        if (
            record["actual"] != record["expected"]
            or actual.shape != expected.shape
            or result.dtype != getattr(mx, case["dtype"])
            or not record["inputUnchanged"]
        ):
            raise RuntimeError(f"Gather numerical mismatch: {case['id']}")
        print(case["id"], flush=True)
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    suite = unittest.defaultTestLoader.loadTestsFromNames(upstream_tests)
    start = host.dispatch_count if host else 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "dispatchStart": start,
        "dispatchCount": host.dispatch_count - start if host else 0,
    }
    (args.output_dir / "upstream.json").write_text(json.dumps(summary))
    validate_upstream(summary, native=host is not None, tests=upstream_tests)


def validate_upstream(summary, *, native, tests=UPSTREAM_TESTS):
    expected = {
        "testsRun": len(tests),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    if any(
        type(summary.get(key)) is not int or summary[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("Unchanged upstream indexing tests did not pass")
    for key in ("dispatchStart", "dispatchCount"):
        if type(summary.get(key)) is not int or (
            summary[key] <= 0 if native else summary[key] != 0
        ):
            raise ValueError("Upstream gather dispatch accounting is incomplete")


def validate_records(np, records, *, native, axis=False, capacity=False):
    workloads = selected_workloads(axis=axis, capacity=capacity)
    cases = list(workloads.cases())
    if len(records) != len(cases):
        raise ValueError("Gather evidence does not cover every required workload")
    cursor = 0
    for case, record in zip(cases, records):
        _, _, expected = workloads.reference(np, case)
        if (
            any(
                record.get(key) != value
                for key, value in case.items()
                if key != "dtype"
            )
            or record.get("actual") != gather_workloads.words(np, expected)
            or record.get("expected") != record["actual"]
            or record.get("shape") != list(expected.shape)
            or record.get("dtype") != "mlx.core." + case["dtype"].removesuffix("_")
            or record.get("inputUnchanged") is not True
            or type(record.get("dispatchCount")) is not int
            or type(record.get("dispatchStart")) is not int
            or record["dispatchStart"] != cursor
            or (record["dispatchCount"] < 1 if native else record["dispatchCount"] != 0)
        ):
            raise ValueError("Gather results or dispatch evidence do not match")
        cursor += record["dispatchCount"]


def verify(args):
    import numpy as np

    workloads = selected_workloads(axis=args.axis, capacity=args.capacity)
    upstream_tests = AXIS_UPSTREAM_TESTS if args.axis else UPSTREAM_TESTS
    args.output_dir.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    test_sources = upstream_test_sources(args.mlx_root)
    records, upstream = {}, {}
    for mode in ("cpu", "native"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "180" if mode == "cpu" else "1800",
            "--label",
            f"MLX gather {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_gather",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--reductions",
            str(args.reductions.resolve()),
            "--integer64",
            str(args.integer64.resolve()),
            "--output-dir",
            str(args.output_dir.resolve() / mode),
        ]
        if args.axis:
            command.append("--axis")
        if args.capacity:
            command.append("--capacity")
        with (args.output_dir / f"{mode}.stdout").open("w") as stdout, (
            args.output_dir / f"{mode}.stderr"
        ).open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        (args.output_dir / f"{mode}.command.json").write_text(
            json.dumps({"command": command, "returncode": result.returncode})
        )
        if result.returncode:
            raise RuntimeError(
                f"Gather {mode} worker failed; inspect its retained output"
            )
        records[mode] = json.loads(
            (args.output_dir / mode / "results.json").read_text()
        )
        validate_records(
            np,
            records[mode],
            native=mode == "native",
            axis=args.axis,
            capacity=args.capacity,
        )
        upstream[mode] = json.loads(
            (args.output_dir / mode / "upstream.json").read_text()
        )
        validate_upstream(upstream[mode], native=mode == "native", tests=upstream_tests)
    trace = [
        json.loads(line)
        for line in (args.output_dir / "native/dispatch.jsonl").read_text().splitlines()
    ]
    audit = gather_evidence.validate(
        np, records["native"], trace, upstream["native"], workloads=workloads
    )
    if (
        verify_prepared(args.mlx_root) != before
        or upstream_test_sources(args.mlx_root) != test_sources
    ):
        raise ValueError("MLX sources changed during gather verification")
    summary = {
        "commit": COMMIT,
        **audit,
        "adaptation": before,
        "upstreamTestSources": test_sources,
        "upstreamTests": list(upstream_tests),
        "casesPerPath": len(records["native"]),
        "upstreamTestsPerPath": len(upstream_tests),
        "numericalParity": True,
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "reductions", "integer64", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--axis", action="store_true", help="Verify GatherAxis and take_along_axis"
    )
    selection.add_argument(
        "--capacity",
        action="store_true",
        help="Include large general-gather views and batch boundaries",
    )
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
