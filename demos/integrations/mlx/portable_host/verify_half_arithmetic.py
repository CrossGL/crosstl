"""Verify half arithmetic and an unchanged upstream random-distribution test."""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    half_arithmetic_workloads,
    random_evidence,
)
from demos.integrations.mlx.portable_host.packages import (
    HALF_ARITHMETIC_ENTRIES,
    HALF_ENTRIES,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES as RANDOM_ENTRIES,
)
from demos.integrations.mlx.portable_host.random_packages import load_index
from demos.integrations.mlx.portable_host.runtime import ALL_HALF_ENTRIES, HostRuntime
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_gather import validate_upstream
from demos.integrations.mlx.portable_host.verify_half import (
    audit_half_events,
    write_json,
)
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TESTS = ("test_random.TestRandom.test_broadcastable_scale_loc",)


def worker(args):
    import mlx.core as mx
    import numpy as np

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            half=args.half,
            half_arithmetic=args.arithmetic,
            random=args.random,
        )
        if args.worker == "missing":
            host.descriptors.pop("vv_Addfloat16")
        host.install()
    if args.worker == "missing":
        try:
            mx.eval(
                mx.add(
                    mx.array([1.0], dtype=mx.float16), mx.array([2.0], dtype=mx.float16)
                )
            )
        except (ValueError, RuntimeError) as error:
            if (
                "No translated package for vv_Addfloat16" not in str(error)
                or host.dispatch_count
            ):
                raise RuntimeError(
                    "Missing half arithmetic package was not rejected before dispatch"
                ) from error
            write_json(
                args.output_dir / "rejection.json",
                {"error": str(error), "dispatchCount": 0},
            )
            return
        raise RuntimeError("Missing half arithmetic package was accepted")
    half_arithmetic_workloads.run(
        mx,
        np,
        host,
        lambda records: write_json(args.output_dir / "results.json", records),
    )
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    start = host.dispatch_count if host else 0
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(UPSTREAM_TESTS)
    )
    summary = {
        "testsRun": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(result.skipped),
        "dispatchStart": start,
        "dispatchCount": host.dispatch_count - start if host else 0,
    }
    write_json(args.output_dir / "upstream.json", summary)
    validate_upstream(summary, native=host is not None, tests=UPSTREAM_TESTS)


def verify(args):
    import numpy as np

    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    sources = upstream_test_sources(args.mlx_root, UPSTREAM_TESTS)
    write_json(output / "adaptation-before.json", before)
    base = json.loads((args.packages / "index.json").read_text())
    families = [(args.packages, base)]
    for directory, family, entries in (
        (args.half, "half", HALF_ENTRIES),
        (args.arithmetic, "half-arithmetic", HALF_ARITHMETIC_ENTRIES),
    ):
        index = json.loads((directory / "index.json").read_text())
        if (
            index.get("target") != base["target"]
            or index.get("family") != family
            or set(index.get("descriptors", {})) != set(entries)
        ):
            raise ValueError(
                "Half arithmetic requires the exact package inventory and target"
            )
        families.append((directory, index))
    families.append(
        (
            args.random,
            {
                "target": base["target"],
                "descriptors": load_index(args.random, base["target"]),
            },
        )
    )
    evidence = {
        "passed": False,
        "commit": COMMIT,
        "target": base["target"],
        "upstreamTestSources": sources,
        "fullUpstreamSuite": False,
    }
    write_json(output / "summary.json", evidence)
    try:
        for mode in ("cpu", "native", "missing"):
            command = [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"
                ),
                "--timeout-seconds",
                "1800" if mode == "native" else "120",
                "--label",
                "MLX half arithmetic " + mode,
                "--",
                sys.executable,
                "-m",
                __package__ + ".verify_half_arithmetic",
                "--worker",
                mode,
            ]
            for name in ("mlx_root", "packages", "half", "arithmetic", "random"):
                command.extend(
                    ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
                )
            command.extend(("--output-dir", str(output / mode)))
            with (output / (mode + ".stdout")).open("w") as stdout, (
                output / (mode + ".stderr")
            ).open("w") as stderr:
                process = subprocess.run(
                    command, stdout=stdout, stderr=stderr, check=False
                )
            write_json(
                output / (mode + ".command.json"),
                {"command": command, "returncode": process.returncode},
            )
            if process.returncode:
                raise RuntimeError(
                    f"Half arithmetic {mode} worker failed; inspect {output}"
                )
        rejection = json.loads((output / "missing/rejection.json").read_text())
        if rejection.get(
            "dispatchCount"
        ) != 0 or "No translated package for vv_Addfloat16" not in rejection.get(
            "error", ""
        ):
            raise ValueError("Half arithmetic rejection evidence differs")
        trace = [
            json.loads(line)
            for line in (output / "native/dispatch.jsonl").read_text().splitlines()
        ]
        upstream = {}
        for mode in ("cpu", "native"):
            upstream[mode] = json.loads((output / mode / "upstream.json").read_text())
            validate_upstream(
                upstream[mode], native=mode == "native", tests=UPSTREAM_TESTS
            )
            records = json.loads((output / mode / "results.json").read_text())
            half_arithmetic_workloads.validate(
                records,
                trace[: upstream[mode]["dispatchStart"]] if mode == "native" else [],
                native=mode == "native",
            )
        if upstream["native"]["dispatchStart"] + upstream["native"][
            "dispatchCount"
        ] != len(trace):
            raise ValueError("Half arithmetic upstream dispatch accounting differs")
        audit_half_events(
            [event for event in trace if event["entry"] in ALL_HALF_ENTRIES], output
        )
        for event in trace:
            if event["entry"] in RANDOM_ENTRIES:
                random_evidence.audit_event(np, event)
        verify_native_identity(
            [
                event
                for event in trace
                if event["entry"] not in (*ALL_HALF_ENTRIES, *RANDOM_ENTRIES)
            ],
            base["target"],
        )
        checked = 0
        for directory, index in families:
            events = [
                event for event in trace if event["entry"] in index["descriptors"]
            ]
            verify_artifacts(events, directory, index, variants=False)
            checked += len(events)
        if checked != len(trace):
            raise ValueError(
                "Half arithmetic trace contains unknown or duplicate packages"
            )
        after = verify_prepared(args.mlx_root)
        write_json(output / "adaptation-after.json", after)
        if before != after or sources != upstream_test_sources(
            args.mlx_root, UPSTREAM_TESTS
        ):
            raise ValueError("Half arithmetic verification changed pinned sources")
        evidence.update(
            passed=True,
            workloads=len(half_arithmetic_workloads.cases()),
            nativeDispatches=len(trace),
            upstream=upstream,
        )
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        write_json(output / "summary.json", evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "half", "arithmetic", "random", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", "missing"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
