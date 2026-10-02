"""Verify translated selection through MLX and the unchanged upstream where test."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import absolute_workloads, selection_workloads
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import load_index
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TEST = "test_ops.TestOps.test_where"
NEGATIVE_CHECKS = {
    "missing": "No translated package for v_Selectint32",
    "int64": "matching float32/int32/uint32/bool values",
    "uint8": "matching float32/int32/uint32/bool values",
    "float16": "matching float32/int32/uint32/bool values",
    "over-limit": "at most 65535 elements",
    "absolute-missing": "No translated package for v_Absint32int32",
    "absolute-int16": "unary dispatch requires float32 arrays",
    "absolute-over-limit": "at most 65535 stored elements",
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
            selection=args.selection,
            absolute=args.absolute,
            reductions=args.reductions,
        )
        if args.worker == "missing":
            host.descriptors.pop("v_Selectint32")
        elif args.worker == "absolute-missing":
            host.descriptors.pop("v_Absint32int32")
        host.install()
    if args.worker in NEGATIVE_CHECKS:
        check = args.worker.removeprefix("absolute-")
        dtype = check if check in {"int64", "int16", "uint8", "float16"} else "int32"
        count = 65536 if check == "over-limit" else 3
        values = mx.array(np.arange(count).astype(dtype))
        condition = mx.array(np.ones(count, dtype="bool"))
        try:
            mx.eval(
                mx.abs(values)
                if args.worker.startswith("absolute-")
                else mx.where(condition, values, values)
            )
        except (ValueError, RuntimeError) as error:
            record = {
                "check": args.worker,
                "error": str(error),
                "dispatchCount": host.dispatch_count,
            }
            write_json(output / "rejection.json", record)
            if NEGATIVE_CHECKS[args.worker] not in str(error) or host.dispatch_count:
                raise RuntimeError(
                    "Selection was not rejected before dispatch"
                ) from error
            return
        raise RuntimeError("Unsupported selection was accepted")
    selection_workloads.run(
        mx, np, host, lambda value: write_json(output / "results.json", value)
    )
    selection_dispatches = host.dispatch_count if host else 0
    absolute_workloads.run(
        mx, np, host, lambda value: write_json(output / "absolute.json", value)
    )
    workload_dispatches = host.dispatch_count if host else 0
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName(UPSTREAM_TEST)
    )
    write_json(
        output / "upstream.json",
        {
            "test": UPSTREAM_TEST,
            "testsRun": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skips": len(result.skipped),
            "dispatchCount": host.dispatch_count - workload_dispatches if host else 0,
            "workloadDispatchCount": workload_dispatches,
            "selectionDispatchCount": selection_dispatches,
        },
    )
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("The unchanged upstream where test failed")


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    sources = upstream_test_sources(args.mlx_root)
    write_json(output / "adaptation-before.json", before)
    base = json.loads((args.packages / "index.json").read_text())
    selection = json.loads((args.selection / "index.json").read_text())
    absolute = json.loads((args.absolute / "index.json").read_text())
    reductions = load_index(args.reductions, base["target"])
    if "w32/all_reduce_andbool_" not in reductions["descriptors"]:
        raise ValueError(
            "The upstream where test requires Boolean reduction at width32"
        )
    results, absolute_results, upstream, rejections = {}, {}, {}, {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "900" if mode == "native" else "60",
            "--label",
            f"MLX selection {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_selection",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--selection",
            str(args.selection.resolve()),
            "--absolute",
            str(args.absolute.resolve()),
            "--reductions",
            str(args.reductions.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        write_json(
            output / f"{mode}.command.json",
            {"command": command, "returncode": process.returncode},
        )
        if process.returncode:
            raise RuntimeError(f"Selection {mode} worker failed; inspect {output}")
        if mode in NEGATIVE_CHECKS:
            record = json.loads((output / mode / "rejection.json").read_text())
            if (
                record.get("check") != mode
                or record.get("dispatchCount") != 0
                or NEGATIVE_CHECKS[mode] not in record.get("error", "")
            ):
                raise ValueError("Selection rejection evidence is incomplete")
            rejections[mode] = record
        else:
            results[mode] = json.loads((output / mode / "results.json").read_text())
            absolute_results[mode] = json.loads(
                (output / mode / "absolute.json").read_text()
            )
            record = json.loads((output / mode / "upstream.json").read_text())
            required = {
                "test": UPSTREAM_TEST,
                "testsRun": 1,
                "failures": 0,
                "errors": 0,
                "skips": 0,
            }
            if any(record.get(key) != value for key, value in required.items()):
                raise ValueError("Upstream where evidence is incomplete")
            if mode == "cpu" and (
                record.get("dispatchCount") != 0
                or record.get("workloadDispatchCount") != 0
            ):
                raise ValueError("CPU selection unexpectedly dispatched")
            if mode == "native" and (
                type(record.get("dispatchCount")) is not int
                or record["dispatchCount"] <= 0
            ):
                raise ValueError("Upstream where dispatch evidence is incomplete")
            upstream[mode] = record
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    count = upstream["native"]["workloadDispatchCount"]
    if (
        type(count) is not int
        or count < 0
        or count + upstream["native"]["dispatchCount"] != len(trace)
    ):
        raise ValueError("Selection dispatch count differs")
    selection_workloads.validate(results["cpu"], [], native=False)
    selection_count = upstream["native"]["selectionDispatchCount"]
    if type(selection_count) is not int or not 0 <= selection_count <= count:
        raise ValueError("Selection workload dispatch count differs")
    selection_workloads.validate(
        results["native"], trace[:selection_count], native=True
    )
    absolute_workloads.validate(absolute_results["cpu"], [], native=False)
    absolute_workloads.validate(
        absolute_results["native"], trace[selection_count:count], native=True
    )
    verify_native_identity(trace, base["target"])
    checked_events = 0
    for index, directory, variants in (
        (base, args.packages, False),
        (selection, args.selection, False),
        (absolute, args.absolute, False),
        (reductions, args.reductions, True),
    ):
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
        checked_events += len(events)
        verify_artifacts(events, directory, index, variants=variants)
    if checked_events != len(trace):
        raise ValueError("Selection trace references an unknown artifact")
    after = verify_prepared(args.mlx_root)
    if before != after or sources != upstream_test_sources(args.mlx_root):
        raise ValueError("MLX sources changed during selection verification")
    evidence = {
        "commit": COMMIT,
        "target": base["target"],
        "adaptation": before,
        "upstreamTestSources": sources,
        "casesPerPath": len(results["native"]),
        "absoluteCasesPerPath": len(absolute_results["native"]),
        "dispatchCount": len(trace),
        "upstream": upstream,
        "negativeChecks": rejections,
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    write_json(output / "adaptation-after.json", after)
    write_json(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "mlx-root",
        "packages",
        "selection",
        "absolute",
        "reductions",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
