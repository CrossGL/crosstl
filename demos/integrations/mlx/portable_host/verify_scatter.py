"""Verify general Scatter through MLX indexed updates and unchanged upstream tests."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_gather import validate_upstream

UPSTREAM_TESTS = ("test_array.TestArray.test_setitem_with_list",)


def validate_records(np, records, *, native, float32=False):
    cases = list(workloads.float_cases() if float32 else workloads.cases())
    if len(records) != len(cases):
        raise ValueError("Scatter evidence does not cover every workload")
    cursor = 0
    for case, record in zip(cases, records):
        *_, expected = workloads.reference(np, case)
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("actual") != words(np, expected)
            or record.get("shape") != list(expected.shape)
            or record.get("resultDtype") != "mlx.core." + case["dtype"]
            or record.get("inputUnchanged") is not True
            or type(record.get("dispatchCount")) is not int
            or type(record.get("dispatchStart")) is not int
            or record["dispatchStart"] != cursor
            or (record["dispatchCount"] < 1 if native else record["dispatchCount"] != 0)
        ):
            raise ValueError("Scatter results or dispatch accounting do not match")
        cursor += record["dispatchCount"]


def worker(args):
    import mlx.core as mx
    import numpy as np

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "native":
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            mlx_root=args.mlx_root,
            integer64=args.integer64,
        )
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    records = []
    for case in workloads.float_cases() if args.float32 else workloads.cases():
        start = host.dispatch_count if host else 0
        source, result = workloads.expression(mx, np, case)
        expected_source, _, _, expected = workloads.reference(np, case)
        actual = np.array(result)
        record = {
            **case,
            "actual": words(np, actual),
            "shape": list(actual.shape),
            "resultDtype": str(result.dtype),
            "inputUnchanged": words(np, np.array(source)) == words(np, expected_source),
            "dispatchStart": start,
            "dispatchCount": host.dispatch_count - start if host else 0,
        }
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
        if (
            actual.shape != expected.shape
            or record["actual"] != words(np, expected)
            or not record["inputUnchanged"]
        ):
            raise RuntimeError(f"Scatter numerical mismatch: {case['id']}")
        print(case["id"], flush=True)
    validate_records(np, records, native=host is not None, float32=args.float32)
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    suite = unittest.defaultTestLoader.loadTestsFromNames(UPSTREAM_TESTS)
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
    validate_upstream(summary, native=host is not None, tests=UPSTREAM_TESTS)


def verify(args):
    import numpy as np

    from demos.integrations.mlx.portable_host.scatter_evidence import validate

    args.output_dir.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    test_sources = upstream_test_sources(args.mlx_root)
    records, upstream = {}, {}
    for mode in ("cpu", "native"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "180" if mode == "cpu" else "3300",
            "--label",
            f"MLX scatter {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_scatter",
            "--worker",
            mode,
        ]
        for name in ("mlx_root", "packages", "integer64"):
            command.extend(
                ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
            )
        command.extend(["--output-dir", str(args.output_dir.resolve() / mode)])
        if args.float32:
            command.append("--float32")
        with (args.output_dir / f"{mode}.stdout").open("w") as stdout, (
            args.output_dir / f"{mode}.stderr"
        ).open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        (args.output_dir / f"{mode}.command.json").write_text(
            json.dumps({"command": command, "returncode": result.returncode})
        )
        if result.returncode:
            raise RuntimeError(f"Scatter {mode} worker failed; inspect retained output")
        records[mode] = json.loads(
            (args.output_dir / mode / "results.json").read_text()
        )
        validate_records(
            np, records[mode], native=mode == "native", float32=args.float32
        )
        upstream[mode] = json.loads(
            (args.output_dir / mode / "upstream.json").read_text()
        )
        validate_upstream(upstream[mode], native=mode == "native", tests=UPSTREAM_TESTS)
    trace = [
        json.loads(line)
        for line in (args.output_dir / "native/dispatch.jsonl").read_text().splitlines()
    ]
    audit = validate(
        np, records["native"], trace, upstream["native"], float32=args.float32
    )
    if (
        verify_prepared(args.mlx_root) != before
        or upstream_test_sources(args.mlx_root) != test_sources
    ):
        raise ValueError("MLX sources changed during scatter verification")
    summary = {
        "commit": COMMIT,
        **audit,
        "adaptation": before,
        "upstreamTestSources": test_sources,
        "upstreamTests": list(UPSTREAM_TESTS),
        "upstreamTestsPerPath": len(UPSTREAM_TESTS),
        "casesPerPath": len(records["native"]),
        "workloadProfile": "float32" if args.float32 else "integer",
        "numericalParity": True,
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "integer64", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
