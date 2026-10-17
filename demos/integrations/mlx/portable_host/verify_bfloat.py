"""Validate bfloat host operations against CPU and independent bit references."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import bfloat_workloads
from demos.integrations.mlx.portable_host.packages import BFLOAT_ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_block_quantization import compile_entry
from demos.integrations.mlx.portable_host.verify_half import (
    audit_half_events,
    write_json,
)
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts


def worker(args):
    import mlx.core as mx
    import numpy as np

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("An original GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            bfloat=args.bfloat,
            retain_native_modules=True,
        )
        if args.worker == "missing":
            host.descriptors.pop("vv_Addbfloat16")
        host.install()
    if args.worker == "missing":
        try:
            mx.eval(
                mx.add(
                    mx.array([1.0], dtype=mx.bfloat16),
                    mx.array([2.0], dtype=mx.bfloat16),
                )
            )
        except (ValueError, RuntimeError) as error:
            if (
                "No translated package for vv_Addbfloat16" not in str(error)
                or host.dispatch_count
            ):
                raise RuntimeError(
                    "Missing bfloat package was not rejected before dispatch"
                ) from error
            write_json(
                args.output_dir / "rejection.json",
                {"error": str(error), "dispatchCount": 0},
            )
            return
        raise RuntimeError("Missing bfloat package was accepted")
    bfloat_workloads.run(
        mx,
        np,
        host,
        lambda records: write_json(args.output_dir / "results.json", records),
    )


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    base = json.loads((args.packages / "index.json").read_text())
    bfloat = json.loads((args.bfloat / "index.json").read_text())
    if (
        bfloat.get("family") != "bfloat"
        or bfloat.get("target") != base["target"]
        or set(bfloat.get("descriptors", {})) != set(BFLOAT_ENTRIES)
    ):
        raise ValueError("Bfloat proof requires the exact package inventory and target")
    evidence = {
        "passed": False,
        "commit": COMMIT,
        "target": base["target"],
        "fullUpstreamSuite": False,
    }
    write_json(output / "summary.json", evidence)
    write_json(output / "adaptation-before.json", before)
    for entry, descriptor in bfloat["descriptors"].items():
        compile_entry(
            base["target"],
            entry,
            descriptor,
            args.bfloat / "package",
            output / "compilation",
        )
    records = {}
    for mode in ("cpu", "native", "missing"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "600" if mode == "native" else "120",
            "--label",
            "MLX bfloat " + mode,
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_bfloat",
            "--worker",
            mode,
        ]
        for name in ("mlx_root", "packages", "bfloat"):
            command.extend(
                ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
            )
        command.extend(("--output-dir", str(output / mode)))
        with (output / (mode + ".stdout")).open("w") as stdout, (
            output / (mode + ".stderr")
        ).open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        write_json(
            output / (mode + ".command.json"),
            {"command": command, "returncode": result.returncode},
        )
        if result.returncode:
            raise RuntimeError(f"Bfloat {mode} worker failed; inspect {output}")
        if mode != "missing":
            records[mode] = json.loads((output / mode / "results.json").read_text())
    rejection = json.loads((output / "missing/rejection.json").read_text())
    if rejection.get(
        "dispatchCount"
    ) != 0 or "No translated package for vv_Addbfloat16" not in rejection.get(
        "error", ""
    ):
        raise ValueError("Bfloat rejection evidence differs")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    bfloat_workloads.validate(records["cpu"], [], native=False)
    bfloat_workloads.validate(records["native"], trace, native=True)
    audit_half_events(trace, output)
    covered = 0
    for directory, index in ((args.packages, base), (args.bfloat, bfloat)):
        events = [event for event in trace if event["entry"] in index["descriptors"]]
        verify_artifacts(events, directory, index, variants=False)
        covered += len(events)
    if covered != len(trace):
        raise ValueError(
            "Bfloat proof contains unknown or duplicate packaged artifacts"
        )
    if verify_prepared(args.mlx_root) != before:
        raise ValueError("Bfloat proof changed pinned sources")
    evidence.update(
        passed=True,
        workloads=len(records["native"]),
        nativeDispatches=len(trace),
        compiledEntries=len(bfloat["descriptors"]),
        adaptation=before,
    )
    write_json(output / "summary.json", evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "bfloat", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", "missing"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
