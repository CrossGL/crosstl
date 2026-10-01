"""Verify bounded row plans on CPU and the explicitly registered native backend."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import row_workloads
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import (
    ROW_ENTRIES,
    load_index,
)
from demos.integrations.mlx.portable_host.runtime import HostRuntime


def verify_artifacts(trace, directory, index):
    checked = set()
    for record in trace:
        key = f'w{record["workgroupSize"][0]}/{record["entry"]}'
        artifact = index["descriptors"][key]["artifact"]
        if (
            record.get("artifact") != artifact
            or record.get("target") != index["target"]
        ):
            raise ValueError("Row trace does not identify its packaged artifact")
        if key not in checked:
            package = (directory / "package").resolve()
            path = (package / artifact["packagePath"]).resolve()
            if not path.is_relative_to(package):
                raise ValueError("Row artifact escapes its package directory")
            payload = path.read_bytes()
            if artifact.get("hash") != {
                "algorithm": "sha256",
                "value": hashlib.sha256(payload).hexdigest(),
            } or artifact.get("sizeBytes") != len(payload):
                raise ValueError("Row artifact changed after translation")
            checked.add(key)


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True)
    index = json.loads((args.reductions / "index.json").read_text())
    runtime = None
    if args.worker == "native":
        runtime = HostRuntime(
            args.packages, output / "dispatch.jsonl", reductions=args.reductions
        )
        runtime.install()
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    def observe(record):
        with (output / "readbacks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    records = row_workloads.collect(
        mx,
        np,
        index["widths"],
        observe=observe,
        dispatch_count=(
            (lambda: runtime.dispatch_count) if runtime is not None else None
        ),
    )
    row_workloads.validate(records, index["widths"])
    (output / "result.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    target = json.loads((args.packages / "index.json").read_text())["target"]
    index = load_index(args.reductions, target)
    if (
        index.get("family") != "row"
        or set(index["entries"]) != set(ROW_ENTRIES)
        or not {32, 128}.issubset(index["widths"])
    ):
        raise ValueError("Row proof requires every row entry and both threshold widths")
    results = {}
    for mode in ("cpu", "native"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "3600",
            "--label",
            f"MLX row host {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_rows",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--reductions",
            str(args.reductions.resolve()),
            "--output-dir",
            str(output / mode),
        ]
        with (output / f"{mode}.stdout").open("w") as stdout, (
            output / f"{mode}.stderr"
        ).open("w") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        (output / f"{mode}.command.json").write_text(
            json.dumps({"command": command, "returncode": result.returncode}, indent=2)
        )
        if result.returncode:
            raise RuntimeError(f"Row verification failed for {mode}; inspect {output}")
        results[mode] = json.loads((output / mode / "result.json").read_text())
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    row_workloads.validate(results["cpu"], index["widths"])
    row_workloads.validate(results["native"], index["widths"], trace=trace)
    verify_artifacts(trace, args.reductions, index)
    if len(trace) != len(results["native"]) or any(
        record["target"] != target for record in trace
    ):
        raise ValueError("Row trace contains missing or unexpected dispatches")
    if [record["actual"] for record in results["cpu"]] != [
        record["actual"] for record in results["native"]
    ]:
        raise ValueError("CPU and native row results differ")
    if verify_prepared(args.mlx_root) != before:
        raise ValueError("MLX sources changed during row verification")
    evidence = {
        "commit": COMMIT,
        "target": target,
        "adaptation": before,
        "widths": index["widths"],
        "entries": sorted(ROW_ENTRIES),
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
        "results": results,
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reductions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        verify(args)
