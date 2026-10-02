"""Run small-row reductions through MLX's CPU and generated native backends."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

from crosstl.project import select_native_loader_dispatch_regions
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import ENTRIES
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
    HostRuntime,
)
from demos.integrations.mlx.portable_host.small_row_packages import SmallRowPackageCache


def cases():
    for entry, dtype in ENTRIES.items():
        operation = entry.removeprefix("all_reduce_").removesuffix(dtype)
        for label, shape, axes in (
            ("scalar-tail", (1, 1031, 7), (0, 2)),
            ("cooperative", (9, 5, 9), (0, 2)),
        ):
            yield dict(
                id=f"{label}-{operation}-{dtype}",
                dtype=dtype,
                operation=operation,
                shape=list(shape),
                axes=list(axes),
            )
    for label, shape, axes in (
        ("scalar-boundary", (31, 5, 8), (0, 2)),
        ("cooperative-boundary", (32, 5, 8), (0, 2)),
        ("scalar-width64", (8, 33, 64), (0, 2)),
        ("cooperative-width64", (9, 33, 64), (0, 2)),
        ("rank2", (2, 3, 3, 5, 7), (0, 2, 4)),
        ("rank5", (2, 3, 2, 5, 2, 3, 7), (0, 2, 4, 6)),
    ):
        yield dict(
            id=label,
            dtype="float32",
            operation="sum",
            shape=list(shape),
            axes=list(axes),
        )


def reference(np, case):
    indices = np.arange(np.prod(case["shape"])).reshape(case["shape"])
    coordinates = np.indices(case["shape"])
    row = np.zeros(case["shape"], dtype=np.int64)
    reduction_index = np.zeros(case["shape"], dtype=np.int64)
    for axis, size in enumerate(case["shape"]):
        if axis in case["axes"]:
            reduction_index = reduction_index * size + coordinates[axis]
        else:
            row = row * size + coordinates[axis]
    if case["dtype"] == "bool_":
        data = (
            (row % 2 == 0) | (reduction_index != 0)
            if case["operation"] == "and"
            else (row % 2 == 0) & (reduction_index == 0)
        )
    elif case["operation"] == "prod":
        data = np.ones(case["shape"], dtype=np.int64)
        data[(row % 3 == 0) & (reduction_index == 0)] = 0
        if case["dtype"] != "uint32":
            data[(row % 3 == 1) & (reduction_index == 0)] = -1
    else:
        data = indices % 19 + row * 32
        if case["dtype"] != "uint32":
            data = data - 9
        if case["dtype"] == "float32":
            data = data / 4
    data = data.astype(case["dtype"])
    name = {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
    kwargs = {"dtype": data.dtype} if name in {"sum", "prod"} else {}
    expected = getattr(np, name)(data, axis=tuple(case["axes"]), **kwargs)
    return data, expected


def worker(args):
    import mlx.core as mx
    import numpy as np

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "native":
        host = HostRuntime(
            args.packages, args.output_dir / "dispatch.jsonl", mlx_root=args.mlx_root
        )
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    records = []
    for case in cases():
        data, expected = reference(np, case)
        source = mx.array(data)
        start = host.dispatch_count if host else 0
        name = {"and": "all", "or": "any"}.get(case["operation"], case["operation"])
        result = getattr(mx, name)(source, axis=case["axes"])
        actual = np.array(result)
        record = {
            **case,
            "actual": actual.tolist(),
            "expected": expected.tolist(),
            "resultDtype": str(result.dtype),
            "resultShape": list(actual.shape),
            "inputUnchanged": bool(np.array_equal(np.array(source), data)),
            "dispatchCount": host.dispatch_count - start if host else 0,
            "inputHash": hashlib.sha256(data.tobytes()).hexdigest(),
        }
        records.append(record)
        (args.output_dir / "results.json").write_text(json.dumps(records, indent=2))
        if (
            result.dtype != getattr(mx, case["dtype"])
            or actual.shape != expected.shape
            or not np.array_equal(actual, expected)
        ):
            raise RuntimeError(f"Small-row numerical mismatch: {case['id']}")
        if not record["inputUnchanged"]:
            raise RuntimeError(f"Small-row input changed: {case['id']}")
        print(case["id"], flush=True)


def validate(records, trace, *, native):
    import numpy as np

    expected_cases = list(cases())
    if len(records) != len(expected_cases) or len(trace) != (
        len(records) if native else 0
    ):
        raise ValueError("Small-row evidence does not cover every required case")
    for index, (record, case) in enumerate(zip(records, expected_cases)):
        data, expected = reference(np, case)
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("inputHash") != hashlib.sha256(data.tobytes()).hexdigest()
            or record.get("actual") != expected.tolist()
            or record.get("expected") != expected.tolist()
            or record.get("resultDtype")
            != "mlx.core." + case["dtype"].removesuffix("_")
            or record.get("resultShape") != list(expected.shape)
            or record.get("inputUnchanged") is not True
            or type(record.get("dispatchCount")) is not int
            or record["dispatchCount"] != int(native)
        ):
            raise ValueError("Small-row results or dispatch count do not match")
        if native:
            event = trace[index]
            values = expected.reshape(-1).tolist()
            row_size = case["shape"][-1]
            rank = len(case["axes"]) - 1
            dimension = 1 if rank <= 1 else 2 if rank == 2 else 5
            entry = (
                f"row_reduce_small_{dimension}_reduce_"
                + case["operation"]
                + case["dtype"]
            )
            nonrows = math.prod(case["shape"][axis] for axis in case["axes"][:-1])
            scalar = (nonrows < 32 and row_size <= 8) or nonrows <= 8
            group = min(len(values), 1024) if scalar else 32
            exact = [len(values), 1, 1] if scalar else [32, len(values), 1]
            groups = (
                [(len(values) + group - 1) // group, 1, 1]
                if scalar
                else [1, len(values), 1]
            )
            if case["dtype"] == "bool_":
                guard = (
                    BOOLEAN_GUARD
                    if event.get("target") == "metal"
                    else [int(v) for v in BOOLEAN_GUARD]
                )
            elif case["dtype"] == "float32":
                guard = [
                    struct.unpack("<f", struct.pack("<I", value))[0]
                    for value in COPY_GUARD
                ]
            else:
                guard = COPY_GUARD
            if (
                type(event.get("dispatchVersion")) is not int
                or event["dispatchVersion"] != DISPATCH_VERSION
                or event.get("entry") != entry
                or event.get("reductionValues") != values
                or event.get("reductionGuardValues") != guard
                or event.get("threadGridSize") != exact
                or event.get("workgroupSize") != [group, 1, 1]
                or event.get("workgroupCount") != groups
                or event.get("threads") != data.size
            ):
                raise ValueError("Small-row native trace does not match its results")


def verify(args):
    args.output_dir.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    (args.output_dir / "adaptation-before.json").write_text(
        json.dumps(before, indent=2)
    )
    results = {}
    for mode in ("cpu", "native"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "180" if mode == "cpu" else "3600",
            "--label",
            f"MLX small-row host {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_small_rows",
            "--worker",
            mode,
            "--mlx-root",
            str(args.mlx_root.resolve()),
            "--packages",
            str(args.packages.resolve()),
            "--output-dir",
            str(args.output_dir.resolve() / mode),
        ]
        with (args.output_dir / f"{mode}.stdout").open("w") as out, (
            args.output_dir / f"{mode}.stderr"
        ).open("w") as err:
            process = subprocess.run(command, stdout=out, stderr=err, check=False)
        (args.output_dir / f"{mode}.command.json").write_text(
            json.dumps({"command": command, "returncode": process.returncode})
        )
        if process.returncode:
            raise RuntimeError(f"Small-row {mode} worker failed; see {args.output_dir}")
        results[mode] = json.loads(
            (args.output_dir / mode / "results.json").read_text()
        )
    trace = [
        json.loads(line)
        for line in (args.output_dir / "native/dispatch.jsonl").read_text().splitlines()
    ]
    validate(results["cpu"], [], native=False)
    validate(results["native"], trace, native=True)
    target = json.loads((args.packages / "index.json").read_text())["target"]
    cache = SmallRowPackageCache(args.mlx_root, args.packages / "small-rows", target)
    for event in trace:
        if event["target"] != target:
            raise ValueError("Small-row execution target changed")
        packages = cache.get(
            event["entry"],
            thread_grid_size=event["threadGridSize"],
            workgroup_size=event["workgroupSize"],
        )
        if event["artifact"] != packages[0][0]["artifact"]:
            raise ValueError("Small-row artifact changed after execution")
        if target != "metal":
            select_native_loader_dispatch_regions(
                packages,
                thread_grid_size=event["threadGridSize"],
                source_workgroup_size=event["workgroupSize"],
            )
            regions = event["details"].get("regions", [])
            if len(regions) != len(packages):
                raise ValueError("Small-row native region evidence is incomplete")
            for region, (descriptor, root) in zip(regions, packages):
                dimensions = descriptor["provenance"]["dispatchRegion"]
                if (
                    region["artifact"] != descriptor["artifact"]
                    or region["provenance"] != descriptor["provenance"]
                    or region["packageRoot"] != str(root)
                    or region["workgroupCount"] != dimensions["workgroupCount"]
                    or region["workgroupSize"] != dimensions["workgroupSize"]
                    or region["moduleHash"]
                    != hashlib.sha256(
                        Path(region["moduleFile"]).read_bytes()
                    ).hexdigest()
                ):
                    raise ValueError(
                        "Small-row region identity changed after execution"
                    )
    after = verify_prepared(args.mlx_root)
    (args.output_dir / "adaptation-after.json").write_text(json.dumps(after, indent=2))
    if after != before:
        raise ValueError("MLX sources changed during small-row verification")
    summary = {
        "commit": COMMIT,
        "target": target,
        "adaptation": before,
        "casesPerPath": len(results["native"]),
        "dispatchCount": len(trace),
        "dispatchVersion": DISPATCH_VERSION,
        "numericalParity": True,
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
