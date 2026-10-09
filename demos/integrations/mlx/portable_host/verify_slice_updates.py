"""Execute slice reductions and unchanged upstream MLX indexing tests."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import slice_capacity_workloads as capacity
from demos.integrations.mlx.portable_host import slice_update_workloads as workloads
from demos.integrations.mlx.portable_host.gather_evidence import audit_native_execution
from demos.integrations.mlx.portable_host.packages import (
    SLICE_UPDATE_ENTRIES,
    SLICE_UPDATE_TYPES,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import load_index
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.slice_update_layout import (
    MAX_STORAGE_ELEMENTS,
)
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TESTS = ("test_array.TestArray.test_array_at_slice_update_extensive",)
NEGATIVE_CHECKS = {
    **{
        f"missing-{dtype}": "No translated package for slice_update_sum" + dtype
        for dtype in SLICE_UPDATE_TYPES
    },
    "int16": "slice updates require matching supported arrays",
    "limit": "slice updates require matching supported arrays",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


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
            slice_updates=args.slice_updates,
            reductions=args.reductions,
            retain_native_modules=True,
        )
        if args.worker.startswith("missing-"):
            host.descriptors.pop(
                "slice_update_sum" + args.worker.removeprefix("missing-")
            )
        host.install()
    if args.worker in NEGATIVE_CHECKS:
        dtype = (
            args.worker.removeprefix("missing-")
            if args.worker.startswith("missing-")
            else ("int16" if args.worker == "int16" else "float32")
        )
        operand = mx.array(np.ones(4, dtype=dtype))
        if args.worker == "limit":
            operand = mx.broadcast_to(operand[:1], (MAX_STORAGE_ELEMENTS + 1,))
        update = mx.array(np.ones(1, dtype=dtype))
        try:
            mx.eval(operand.at[1:2].add(update))
        except (ValueError, RuntimeError) as error:
            record = {
                "check": args.worker,
                "error": str(error),
                "dispatchCount": host.dispatch_count,
            }
            write_json(output / "rejection.json", record)
            if NEGATIVE_CHECKS[args.worker] not in str(error) or host.dispatch_count:
                raise RuntimeError(
                    "Slice update rejection did not precede native dispatch"
                ) from error
            return
        raise RuntimeError("Unsupported slice update was accepted")
    workloads.run(
        mx, np, host, lambda record: write_json(output / "results.json", record)
    )
    small_count = host.dispatch_count if host else 0
    capacity.run(
        mx, np, host, lambda record: write_json(output / "capacity.json", record)
    )
    write_json(output / "capacity-start.json", {"dispatchCount": small_count})
    count = host.dispatch_count if host else 0
    os.environ["DEVICE"], os.environ["CI"] = ("gpu" if host else "cpu"), "1"
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
        if not result.wasSuccessful() or result.skipped or result.testsRun != 1:
            raise RuntimeError("An unchanged upstream slice update test failed")


def validate_upstream(record, *, native):
    tests = record.get("tests", [])
    if len(tests) != len(UPSTREAM_TESTS):
        raise ValueError("Slice update upstream coverage is incomplete")
    for item, name in zip(tests, UPSTREAM_TESTS):
        expected = {"test": name, "testsRun": 1, "failures": 0, "errors": 0, "skips": 0}
        count = item.get("dispatchCount")
        if (
            any(
                item.get(key) != value or type(item.get(key)) is not type(value)
                for key, value in expected.items()
            )
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError("Slice update upstream test evidence differs")
    count = record.get("workloadDispatchCount")
    if type(count) is not int or (count <= 0 if native else count != 0):
        raise ValueError("Slice update workload dispatch count differs")


def validate_native_event(event, package_directory, target, output):
    details = event["details"]
    request = details["request"]
    artifact = event["artifact"]
    if (
        event["target"] != target
        or request.get("target") != target
        or request.get("entryPoint")
        != {"metal": event["entry"], "opengl": "main", "directx": "CSMain"}[target]
        or request.get("artifact", {}).get("target") != target
        or request.get("artifact", {}).get("packagePath") != artifact["packagePath"]
        or any(
            request.get("dispatch", {}).get(key) != event[key]
            for key in ("workgroupSize", "workgroupCount")
        )
    ):
        raise ValueError("Slice update native dispatch identity differs")
    retained = (output / "native/native-modules").resolve()
    for module in [details["module"], *details["validationModules"]]:
        if not Path(module["file"]).resolve().is_relative_to(retained):
            raise ValueError("Slice update native module is outside retained evidence")
    audit_native_execution(dict(event, packageRoot=str(package_directory)))


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    sources = upstream_test_sources(args.mlx_root)
    write_json(output / "adaptation-before.json", before)
    packages = [
        (directory, json.loads((directory / "index.json").read_text()), False)
        for directory in (args.packages, args.integer64, args.slice_updates)
    ]
    target = packages[0][1]["target"]
    for _, index, _ in packages:
        if index["target"] != target:
            raise ValueError("Slice update package identities differ")
    if set(packages[-1][1]["descriptors"]) != set(SLICE_UPDATE_ENTRIES):
        raise ValueError("Slice update package family is incomplete")
    for directory in args.reductions:
        packages.append((directory, load_index(directory, target), True))
    entries = {
        entry
        for _, index, variants in packages
        if variants
        for entry in index["descriptors"]
    }
    if not {"w32/all_reduce_andbool_", "w64/all_reduce_andbool_"} <= entries:
        raise ValueError(
            "Upstream slice update tests require Boolean reductions at widths 32 and 64"
        )
    results, upstream, rejections = {}, {}, {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "1800" if mode == "native" else "120",
            "--label",
            f"MLX slice updates {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_slice_updates",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--integer64",
            str(args.integer64.resolve()),
            "--slice-updates",
            str(args.slice_updates.resolve()),
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
            raise RuntimeError(f"Slice update {mode} worker failed; inspect {output}")
        if mode in NEGATIVE_CHECKS:
            record = json.loads((output / mode / "rejection.json").read_text())
            if (
                record.get("check") != mode
                or record.get("dispatchCount") != 0
                or (NEGATIVE_CHECKS[mode] not in record.get("error", ""))
            ):
                raise ValueError("Slice update rejection evidence is incomplete")
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
        raise ValueError("Slice update trace count differs")
    workloads.validate(results["cpu"], [], native=False)
    small_count = json.loads((output / "native/capacity-start.json").read_text())[
        "dispatchCount"
    ]
    workloads.validate(results["native"], trace[:small_count], native=True)
    for mode in ("cpu", "native"):
        capacity.validate(
            json.loads((output / mode / "capacity.json").read_text()),
            trace[small_count:count] if mode == "native" else [],
            native=mode == "native",
        )
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
        for event in events:
            validate_native_event(event, directory / "package", target, output)
    if checked != len(trace):
        raise ValueError(
            "Slice update trace references an unknown or duplicate artifact"
        )
    if before != verify_prepared(args.mlx_root) or sources != upstream_test_sources(
        args.mlx_root
    ):
        raise ValueError("MLX sources changed during slice update verification")
    evidence = {
        "commit": COMMIT,
        "target": target,
        "adaptation": before,
        "upstreamTestSources": sources,
        "casesPerPath": len(results["native"]),
        "capacityCasesPerPath": len(list(capacity.cases())),
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
    for name in ("mlx-root", "packages", "integer64", "slice-updates", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append", required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
