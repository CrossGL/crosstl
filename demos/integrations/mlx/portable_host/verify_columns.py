"""Verify unchanged column kernels through actual MLX CPU and native operations."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import column_small_layout, column_workloads
from demos.integrations.mlx.portable_host.column_small_layout import (
    SMALL_COLUMN_ENTRIES,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    CACHE_DIRECTORY,
    SmallColumnPackageCache,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import (
    COLUMN_ENTRIES,
    load_index,
)
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

NEGATIVE_CHECKS = {
    "allocation": "column reduction source view exceeds its bounds",
    "long": "long-column reduction plans are not implemented",
    "limit": "column reduction inner shape exceeds its bounds",
}


def verify_small_artifacts(trace, root, packages, target):
    """Each small column dispatch names its cached package's unchanged artifact."""
    cache = SmallColumnPackageCache(root, packages / CACHE_DIRECTORY, target)
    checked = {}
    for record in trace:
        key = column_small_layout.variant(
            record["entry"], tuple(record["workgroupSize"])
        )
        if key not in checked:
            descriptor, package = cache.get(key)
            artifact = descriptor["artifact"]
            path = (package / artifact["packagePath"]).resolve()
            if not path.is_relative_to(package.resolve()):
                raise ValueError("Small column artifact escapes its package directory")
            payload = path.read_bytes()
            if artifact.get("hash") != {
                "algorithm": "sha256",
                "value": hashlib.sha256(payload).hexdigest(),
            } or artifact.get("sizeBytes") != len(payload):
                raise ValueError("Small column artifact changed after translation")
            checked[key] = artifact
        if record.get("artifact") != checked[key] or record.get("target") != target:
            raise ValueError("Small column trace does not identify its cached artifact")
    return sorted(checked)


def reject_invalid_plan(mx, np, mode, runtime):
    try:
        if mode == "allocation":
            source = mx.as_strided(
                mx.array(np.ones(4, dtype=np.float32)), (33, 33), (33, 1)
            )
        else:
            # An axis of 65,536 columns exceeds the bound on one shape axis.
            shape = {"long": (1024, 2), "limit": (2, 65536)}[mode]
            source = mx.array(np.ones(shape, dtype=np.float32))
        mx.eval(mx.sum(source, axis=0))
    except (ValueError, RuntimeError) as error:
        if NEGATIVE_CHECKS[mode] not in str(error) or runtime.dispatch_count != 0:
            raise RuntimeError(
                "Column rejection did not preserve its dispatch boundary"
            ) from error
        return {"rejected": True, "message": str(error), "dispatchCount": 0}
    raise RuntimeError("Unsupported column plan unexpectedly executed")


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True)
    runtime = None
    if args.worker != "cpu":
        runtime = HostRuntime(
            args.packages,
            output / "dispatch.jsonl",
            mlx_root=args.mlx_root,
            reductions=args.reductions,
        )
        runtime.install()
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    if args.worker in NEGATIVE_CHECKS:
        record = reject_invalid_plan(mx, np, args.worker, runtime)
        (output / "result.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        return

    def observe(record):
        with (output / "readbacks.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    records = column_workloads.collect(
        mx,
        np,
        observe=observe,
        dispatch_count=(
            (lambda: runtime.dispatch_count) if runtime is not None else None
        ),
    )
    column_workloads.validate(records)
    (output / "result.json").write_text(
        json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
    )


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    target = json.loads((args.packages / "index.json").read_text())["target"]
    index = load_index(args.reductions, target)
    if (
        index.get("family") != "column"
        or set(index["entries"]) != set(COLUMN_ENTRIES)
        or index["widths"] != [256]
    ):
        raise ValueError("Column proof requires every column entry at width 256")
    results = {}
    for mode in ("cpu", "native", *NEGATIVE_CHECKS):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "3600",
            "--label",
            f"MLX column host {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_columns",
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
            raise RuntimeError(
                f"Column verification failed for {mode}; inspect {output}"
            )
        results[mode] = json.loads((output / mode / "result.json").read_text())
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    column_workloads.validate(results["cpu"])
    column_workloads.validate(results["native"], trace=trace)
    small = [record for record in trace if record["entry"] in SMALL_COLUMN_ENTRIES]
    verify_artifacts(
        [record for record in trace if record not in small], args.reductions, index
    )
    verify_small_artifacts(small, args.mlx_root, args.packages, target)
    if [record["actual"] for record in results["cpu"]] != [
        record["actual"] for record in results["native"]
    ]:
        raise ValueError("CPU and native column results differ")
    if verify_prepared(args.mlx_root) != before:
        raise ValueError("MLX sources changed during column verification")
    negative = {mode: results.pop(mode) for mode in NEGATIVE_CHECKS}
    for mode, record in negative.items():
        if (
            record.get("rejected") is not True
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != 0
            or NEGATIVE_CHECKS[mode] not in record.get("message", "")
            or (output / mode / "dispatch.jsonl").exists()
        ):
            raise ValueError("Column rejection evidence is missing or inconsistent")
    evidence = {
        "commit": COMMIT,
        "target": target,
        "adaptation": before,
        "widths": index["widths"],
        "entries": sorted(COLUMN_ENTRIES),
        "smallEntries": sorted(SMALL_COLUMN_ENTRIES),
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
        "results": results,
        "negativeChecks": negative,
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8"
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--reductions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        verify(args)
