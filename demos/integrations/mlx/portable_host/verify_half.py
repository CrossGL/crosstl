"""Verify pinned MLX half storage on CPU and a generated native backend."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import half_workloads
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_input_bindings,
    audit_native_execution,
)
from demos.integrations.mlx.portable_host.packages import HALF_COPY_ENTRY, HALF_ENTRIES
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


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
            args.packages, args.output_dir / "dispatch.jsonl", half=args.half
        )
        if args.worker == "missing":
            host.descriptors.pop(HALF_COPY_ENTRY)
        host.install()
    if args.worker == "missing":
        try:
            mx.eval(
                mx.contiguous(
                    mx.array(np.asarray([1.0, 2.0, 3.0], dtype=np.float16))[::-1]
                )
            )
        except (ValueError, RuntimeError) as error:
            record = {"error": str(error), "dispatchCount": host.dispatch_count}
            write_json(args.output_dir / "rejection.json", record)
            if (
                "No translated package for " + HALF_COPY_ENTRY not in str(error)
                or host.dispatch_count
            ):
                raise RuntimeError(
                    "Missing half package did not fail before dispatch"
                ) from error
            return
        raise RuntimeError("Missing half package was accepted")
    half_workloads.run(
        mx,
        np,
        host,
        lambda records: write_json(args.output_dir / "results.json", records),
    )


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    write_json(output / "adaptation-before.json", before)
    base = json.loads((args.packages / "index.json").read_text())
    half = json.loads((args.half / "index.json").read_text())
    if (
        half.get("family") != "half"
        or half.get("target") != base["target"]
        or set(half.get("descriptors", {})) != set(HALF_ENTRIES)
    ):
        raise ValueError("Half proof requires every package for the base target")
    records = {}
    for mode in ("cpu", "native", "missing"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "900" if mode == "native" else "120",
            "--label",
            "MLX half " + mode,
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_half",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--half",
            str(args.half.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        with (output / (mode + ".stdout")).open("w") as stdout, (
            output / (mode + ".stderr")
        ).open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        write_json(
            output / (mode + ".command.json"),
            {"command": command, "returncode": process.returncode},
        )
        if process.returncode:
            raise RuntimeError(f"Half {mode} worker failed; inspect {output}")
        if mode != "missing":
            records[mode] = json.loads((output / mode / "results.json").read_text())
    rejection = json.loads((output / "missing/rejection.json").read_text())
    if rejection.get(
        "dispatchCount"
    ) != 0 or "No translated package for " + HALF_COPY_ENTRY not in rejection.get(
        "error", ""
    ):
        raise ValueError("Half rejection evidence differs")
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    half_workloads.validate(records["cpu"], [], native=False)
    half_workloads.validate(records["native"], trace, native=True)
    verify_native_identity(
        [event for event in trace if event["entry"] not in HALF_ENTRIES],
        base["target"],
    )
    for index, directory in ((base, args.packages), (half, args.half)):
        verify_artifacts(
            [event for event in trace if event["entry"] in index["descriptors"]],
            directory,
            index,
            variants=False,
        )
    for event in trace:
        if event["entry"] not in HALF_ENTRIES:
            continue
        audit_input_bindings(event)
        audit_native_execution(event)
        details = event["details"]
        request = details["request"]
        if request["target"] != event["target"] or any(
            request["dispatch"][key] != event[key]
            for key in ("workgroupSize", "workgroupCount")
        ):
            raise ValueError("Half native dispatch identity differs")
        if (
            request["entryPoint"]
            != {"directx": "CSMain", "opengl": "main", "metal": event["entry"]}[
                event["target"]
            ]
        ):
            raise ValueError("Half native entry differs")
        for name, value in event["inputs"].items():
            binding = request["buffers"][name]
            layout = binding["binding"]["metadata"]["scalarLayout"]
            if (
                binding["dtype"] != value["dtype"]
                or binding["shape"] != value["shape"]
                or binding.get("encoding") != value.get("encoding")
                or layout["elementType"] != value["dtype"]
                or value["shape"] != [len(value["values"])]
            ):
                raise ValueError("Half native upload layout differs")
        for module in [details["module"], *details["validationModules"]]:
            path = Path(module["file"]).resolve()
            if (
                not path.is_relative_to(output / "native/native-modules")
                or hashlib.sha256(path.read_bytes()).hexdigest() != module["sha256"]
            ):
                raise ValueError("Half retained module identity differs")
    after = verify_prepared(args.mlx_root)
    write_json(output / "adaptation-after.json", after)
    if before != after:
        raise ValueError("Half verification changed pinned sources")
    write_json(
        output / "summary.json",
        {
            "passed": True,
            "commit": COMMIT,
            "target": base["target"],
            "workloads": len(half_workloads.cases()),
            "nativeDispatches": len(trace),
            "allBinary16WordsCopied": True,
            "fullUpstreamSuite": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "half", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", "missing"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
