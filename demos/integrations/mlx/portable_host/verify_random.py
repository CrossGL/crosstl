"""Verify random generation through unchanged MLX APIs and native packages."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import random_evidence, random_workloads
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.random_packages import ENTRIES, load_index
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reductions,
)
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TESTS = tuple(
    "test_random.TestRandom." + name
    for name in (
        "test_global_rng",
        "test_key",
        "test_key_split",
        "test_uniform",
        "test_gumbel",
    )
)
REQUIRED_REDUCTIONS = frozenset(
    ("w32/all_reduce_andbool_", "w256/all_reduce_andbool_", "w32/all_reduce_sumfloat32")
)
NEGATIVE_CHECKS = {
    "missing": "No translated package for rbitsc",
    "over-limit": "CrossTL random output or key count exceeds its bounds",
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


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
            random=args.random,
            reductions=[args.reductions],
        )
        if args.worker == "missing":
            for entry in ENTRIES:
                host.descriptors.pop(entry)
        host.install()
    if args.worker in NEGATIVE_CHECKS:
        try:
            mx.eval(
                mx.random.split(
                    mx.random.key(42), 16384 if args.worker == "over-limit" else 3
                )
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
                    "Random rejection did not occur before native dispatch"
                ) from error
            return
        raise RuntimeError("Unsupported random dispatch was accepted")
    random_workloads.run(
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
    validate_upstream(
        {"tests": records, "workloadDispatchCount": count}, native=host is not None
    )


def validate_upstream(record, *, native):
    tests = record.get("tests", [])
    if len(tests) != len(UPSTREAM_TESTS):
        raise ValueError("Random upstream test inventory is incomplete")
    for name, item in zip(UPSTREAM_TESTS, tests):
        expected = {"test": name, "testsRun": 1, "failures": 0, "errors": 0, "skips": 0}
        count = item.get("dispatchCount")
        if (
            any(item.get(key) != value for key, value in expected.items())
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError(
                "An unchanged upstream random test failed or did not execute"
            )
    count = record.get("workloadDispatchCount")
    if type(count) is not int or (count <= 0 if native else count != 0):
        raise ValueError("Random workload dispatch count differs")


def verify(args):
    import numpy as np

    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    sources = upstream_test_sources(args.mlx_root, UPSTREAM_TESTS)
    write_json(output / "adaptation-before.json", before)
    base = json.loads((args.packages / "index.json").read_text())
    target = base["target"]
    random = {"target": target, "descriptors": load_index(args.random, target)}
    reductions = load_reductions(args.reductions, target)
    missing = REQUIRED_REDUCTIONS - reductions["descriptors"].keys()
    if missing:
        raise ValueError(
            "Random upstream assertions require reduction packages: "
            + ", ".join(sorted(missing))
        )
    evidence = {
        "commit": COMMIT,
        "target": target,
        "passed": False,
        "adaptation": before,
        "upstreamTestSources": sources,
        "workers": {},
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    write_json(output / "evidence.json", evidence)
    try:
        for mode in ("cpu", "native", *NEGATIVE_CHECKS):
            command = [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"
                ),
                "--timeout-seconds",
                "1800" if mode == "native" else "120",
                "--label",
                f"MLX random {mode}",
                "--",
                sys.executable,
                "-m",
                __package__ + ".verify_random",
                "--worker",
                mode,
            ]
            for name in ("mlx_root", "packages", "random", "reductions"):
                command.extend(
                    ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
                )
            command.extend(("--output-dir", str(output / mode)))
            with (output / f"{mode}.stdout").open("w") as stdout, (
                output / f"{mode}.stderr"
            ).open("w") as stderr:
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
            record = {"command": command, "returncode": process.returncode}
            evidence["workers"][mode] = record
            write_json(output / f"{mode}.command.json", record)
            write_json(output / "evidence.json", evidence)
        if any(value["returncode"] for value in evidence["workers"].values()):
            raise RuntimeError("A random worker failed; inspect retained logs")
        results, upstream = {}, {}
        trace = [
            json.loads(line)
            for line in (output / "native/dispatch.jsonl").read_text().splitlines()
        ]
        for mode in ("cpu", "native"):
            results[mode] = json.loads((output / mode / "results.json").read_text())
            upstream[mode] = json.loads((output / mode / "upstream.json").read_text())
            validate_upstream(upstream[mode], native=mode == "native")
            count = upstream[mode]["workloadDispatchCount"]
            random_workloads.validate(
                np,
                results[mode],
                trace[:count] if mode == "native" else [],
                native=mode == "native",
            )
        if upstream["native"]["workloadDispatchCount"] + sum(
            test["dispatchCount"] for test in upstream["native"]["tests"]
        ) != len(trace):
            raise ValueError("Random upstream trace count differs")
        random_events = [event for event in trace if event["entry"] in ENTRIES]
        if {event["entry"] for event in random_events} != set(ENTRIES):
            raise ValueError("Random proof requires both key layout entries")
        for event in random_events:
            random_evidence.audit_event(np, event)
        verify_native_identity(
            [event for event in trace if event["entry"] not in ENTRIES], target
        )
        checked = 0
        for directory, index, variants in (
            (args.packages, base, False),
            (args.random, random, False),
            (args.reductions, reductions, True),
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
            verify_artifacts(events, directory, index, variants=variants)
            checked += len(events)
        if checked != len(trace):
            raise ValueError("Random trace contains an unknown or duplicate package")
        rejections = {}
        for mode, error in NEGATIVE_CHECKS.items():
            item = json.loads((output / mode / "rejection.json").read_text())
            if (
                item.get("check") != mode
                or item.get("dispatchCount") != 0
                or error not in item.get("error", "")
            ):
                raise ValueError("Random rejection evidence is incomplete")
            rejections[mode] = item
        if before != verify_prepared(args.mlx_root) or sources != upstream_test_sources(
            args.mlx_root, UPSTREAM_TESTS
        ):
            raise ValueError("MLX source changed during random verification")
        evidence.update(
            passed=True,
            casesPerPath=len(results["native"]),
            dispatchCount=len(trace),
            randomDispatchCount=len(random_events),
            upstream=upstream,
            negativeChecks=rejections,
        )
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        write_json(output / "evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "random", "reductions", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
