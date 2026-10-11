"""Verify source-typed bfloat extrema with exact references and native receipts."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

from demos.integrations.mlx.portable_host import bfloat_storage, reduction_layout
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import (
    BFLOAT_ENTRIES,
    load_index,
)
from demos.integrations.mlx.portable_host.runtime import HostRuntime, physical_dtype
from demos.integrations.mlx.portable_host.small_row_packages import SmallRowPackageCache
from demos.integrations.mlx.portable_host.verify_block_quantization import compile_entry
from demos.integrations.mlx.portable_host.verify_half import (
    audit_half_events,
    write_json,
)

WIDTHS = (32, 128, 256)


def cases():
    for operation in ("min", "max"):
        for count in (125, 128, 509, 512, 1021, 1024, 4097, 65536):
            yield {"operation": operation, "shape": [count], "profile": "negative"}
        for profile in ("mixed", "nan-first", "nan-last", "infinity", "negative-zero"):
            yield {"operation": operation, "shape": [509], "profile": profile}
        for shape in ([3, 32], [33, 32], [512, 32], [1031, 7], [9, 5, 9]):
            yield {"operation": operation, "shape": shape, "profile": "negative"}


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def number(value):
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def reduce_words(values, operation):
    decoded = [number(value) for value in values]
    if any(math.isnan(value) for value in decoded):
        return 0x7FC0
    return word(
        (min if operation == "min" else max)(
            decoded, default=math.inf if operation == "min" else -math.inf
        )
    )


def reference(case):
    shape, profile = case["shape"], case["profile"]
    values = [
        float((i * 17) % 53 - (54 if profile == "negative" else 26))
        for i in range(math.prod(shape))
    ]
    if profile == "nan-first":
        values[0] = math.nan
    elif profile == "nan-last":
        values[-1] = math.nan
    elif profile == "infinity":
        values[0], values[-1] = -math.inf, math.inf
    elif profile == "negative-zero":
        values = [-0.0] * len(values)
    source = list(map(word, values))
    if len(shape) == 1:
        groups = [source]
    elif len(shape) == 2:
        groups = [source[i : i + shape[1]] for i in range(0, len(source), shape[1])]
    else:
        groups = [
            [
                source[(outer * shape[1] + row) * shape[2] + column]
                for outer in range(shape[0])
                for column in range(shape[2])
            ]
            for row in range(shape[1])
        ]
    return source, [reduce_words(group, case["operation"]) for group in groups]


def canonical(words):
    if not isinstance(words, list) or any(
        type(value) is not int or not 0 <= value <= 0xFFFF for value in words
    ):
        raise ValueError("Bfloat reduction storage words are invalid")
    return [
        0x7FC0 if value & 0x7F80 == 0x7F80 and value & 0x7F else value
        for value in words
    ]


def worker(args):
    import mlx.core as mx

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("An original GPU backend is already available")
    host = None
    if args.worker == "native":
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            mlx_root=args.mlx_root,
            bfloat=args.bfloat,
            reductions=args.reductions,
            retain_native_modules=True,
        )
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    records = []
    for case in cases():
        source, expected = reference(case)
        value = mx.array(list(map(number, source)), dtype=mx.bfloat16).reshape(
            case["shape"]
        )
        axes = (
            None
            if len(case["shape"]) == 1
            else (1 if len(case["shape"]) == 2 else (0, 2))
        )
        before = host.dispatch_count if host else 0
        output = getattr(mx, case["operation"])(value, axis=axes)
        mx.eval(output)
        actual = output.tolist()
        actual = list(map(word, actual if isinstance(actual, list) else [actual]))
        original = value.reshape(-1).tolist()
        record = {
            **case,
            "actualWords": actual,
            "dtype": str(output.dtype),
            "inputWords": list(map(word, original)),
            "shapeOut": list(output.shape),
            "dispatchCount": host.dispatch_count - before if host else 0,
        }
        records.append(record)
        write_json(args.output_dir / "results.json", records)
        if canonical(actual) != expected or canonical(
            record["inputWords"]
        ) != canonical(source):
            raise ValueError(
                f"Bfloat reduction differs from independent reference: {case}"
            )


def validate(records, trace, *, native):
    required = list(cases())
    if len(records) != len(required):
        raise ValueError("Bfloat reduction inventory differs")
    cursor = 0
    for case, record in zip(required, records):
        source, expected = reference(case)
        whole = len(case["shape"]) == 1
        dispatches = (2 if whole and len(source) > 4096 else 1) if native else 0
        if (
            any(record.get(key) != value for key, value in case.items())
            or canonical(record["actualWords"]) != expected
            or canonical(record["inputWords"]) != canonical(source)
            or record["dtype"] != "mlx.core.bfloat16"
            or record["shapeOut"] != ([] if whole else [len(expected)])
            or type(record["dispatchCount"]) is not int
            or record["dispatchCount"] != dispatches
        ):
            raise ValueError("Bfloat reduction results differ")
        partials = source
        for event in trace[cursor : cursor + dispatches]:
            target = event["target"]
            input_packet = event["inputs"].get(
                "in_Buffer" if target == "opengl" else "in_"
            )
            source_words = partials if whole else source
            packet = {
                "dtype": physical_dtype("bfloat16", target),
                "shape": [len(source_words)],
                "values": bfloat_storage.pack(source_words, target),
            }
            if bfloat_storage.encoding(target):
                packet["encoding"] = bfloat_storage.encoding(target)
            if input_packet != packet:
                raise ValueError("Bfloat reduction input storage differs")
            if event["dispatchVersion"] != 3 or event[
                "reductionGuardValues"
            ] != bfloat_storage.pack(bfloat_storage.GUARD, target):
                raise ValueError("Bfloat reduction guards or dispatch version differ")
            if whole:
                plan = reduction_layout.stage(len(partials), "bfloat16")
                if (
                    event["entry"] != f'all_reduce_{case["operation"]}bfloat16'
                    or event["threads"] != len(partials)
                    or event["workgroupCount"] != plan["workgroupCount"]
                    or event["workgroupSize"] != plan["workgroupSize"]
                    or event["reductionMetadata"] != {
                        "in_size": len(partials),
                        "row_size": plan["rowSize"],
                    }
                ):
                    raise ValueError("Bfloat reduction pass plan differs")
                row_size = plan["rowSize"]
                partials = [
                    reduce_words(
                        partials[row * row_size : (row + 1) * row_size],
                        case["operation"],
                    )
                    for row in range(plan["workgroupCount"][1])
                ]
            else:
                if event[
                    "entry"
                ] != f'row_reduce_small_1_reduce_{case["operation"]}bfloat16' or event[
                    "threads"
                ] != len(
                    source
                ):
                    raise ValueError("Bfloat small-row specialization differs")
                rows, width = len(expected), case["shape"][-1]
                nonrows = case["shape"][0] if len(case["shape"]) == 3 else 1
                scalar = (nonrows < 32 and width <= 8) or nonrows <= 8
                group = min(rows, 1024) if scalar else 32
                groups = [(rows + group - 1) // group, 1, 1] if scalar else [1, rows, 1]
                exact = [rows, 1, 1] if scalar else [32, rows, 1]
                metadata = {
                    "row_size": [width],
                    "non_row_reductions": [nonrows],
                    "shape": [rows],
                    "strides": [width],
                    "ndim": [1],
                    "reduce_shape": [nonrows] if nonrows > 1 else [0],
                    "reduce_strides": [rows * width] if nonrows > 1 else [0],
                    "reduce_ndim": [int(nonrows > 1)],
                }
                if (
                    event["workgroupCount"] != groups
                    or event["workgroupSize"] != [group, 1, 1]
                    or event["threadGridSize"] != exact
                    or event["reductionMetadata"] != metadata
                ):
                    raise ValueError("Bfloat small-row geometry or metadata differs")
                partials = expected
            actual = bfloat_storage.unpack(event["reductionValues"], target)
            if canonical(actual) != partials:
                raise ValueError("Bfloat native partial reduction differs")
        cursor += dispatches
    if cursor != len(trace):
        raise ValueError("Bfloat reduction dispatch coverage differs")


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    target = json.loads((args.packages / "index.json").read_text())["target"]
    index = load_index(args.reductions, target)
    if (
        index["family"] != "bfloat"
        or set(index["entries"]) != set(BFLOAT_ENTRIES)
        or set(index["widths"]) != set(WIDTHS)
    ):
        raise ValueError(
            "Bfloat reduction proof requires its complete package inventory"
        )
    write_json(output / "summary.json", {"passed": False, "fullUpstreamSuite": False})
    for key, descriptor in index["descriptors"].items():
        compile_entry(
            target,
            key.replace("/", "-"),
            descriptor,
            args.reductions / "package",
            output / "compilation",
        )
    records = {}
    for mode in ("cpu", "native"):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
            "--timeout-seconds",
            "1200" if mode == "native" else "120",
            "--label",
            "Bfloat extrema " + mode,
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_bfloat_reductions",
            "--worker",
            mode,
        ]
        for name in ("mlx_root", "packages", "bfloat", "reductions"):
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
            raise RuntimeError(f"Bfloat extrema {mode} failed; inspect {output}")
        records[mode] = json.loads((output / mode / "results.json").read_text())
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    validate(records["cpu"], [], native=False)
    validate(records["native"], trace, native=True)
    cache = SmallRowPackageCache(args.mlx_root, args.packages / "small-rows", target)
    for event in trace:
        if event["target"] != target:
            raise ValueError("Bfloat reduction target differs")
        if event["entry"].startswith("all_reduce_"):
            key = f'w{event["workgroupSize"][0]}/{event["entry"]}'
            packages = [
                (index["descriptors"][key], args.reductions.resolve() / "package")
            ]
        else:
            packages = cache.get(
                event["entry"],
                thread_grid_size=event["threadGridSize"],
                workgroup_size=event["workgroupSize"],
            )
        if event["artifact"] != packages[0][0]["artifact"]:
            raise ValueError("Bfloat reduction artifact differs")
        for descriptor, root in packages:
            artifact = descriptor["artifact"]
            if (
                hashlib.sha256(
                    (root / artifact["packagePath"]).read_bytes()
                ).hexdigest()
                != artifact["hash"]["value"]
            ):
                raise ValueError("Bfloat reduction source identity differs")
        if "regions" not in event["details"]:
            audit_half_events([event], output)
        else:
            regions = event["details"]["regions"]
            if len(regions) != len(packages):
                raise ValueError("Bfloat region coverage differs")
            for region, (descriptor, root) in zip(regions, packages):
                dimensions = descriptor["provenance"]["dispatchRegion"]
                module = Path(region["moduleFile"]).resolve()
                if (
                    region["artifact"] != descriptor["artifact"]
                    or region["provenance"] != descriptor["provenance"]
                    or region["packageRoot"] != str(root)
                    or region["workgroupCount"] != dimensions["workgroupCount"]
                    or region["workgroupSize"] != dimensions["workgroupSize"]
                    or not module.is_relative_to(output / "native/native-modules")
                    or hashlib.sha256(module.read_bytes()).hexdigest()
                    != region["moduleHash"]
                ):
                    raise ValueError("Bfloat region identity differs")
    if before != verify_prepared(args.mlx_root):
        raise ValueError("Bfloat reduction proof changed pinned sources")
    write_json(
        output / "summary.json",
        {
            "passed": True,
            "commit": COMMIT,
            "target": target,
            "workloads": len(records["native"]),
            "nativeDispatches": len(trace),
            "compiledEntries": len(index["descriptors"]),
            "adaptation": before,
            "fullUpstreamSuite": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "bfloat", "reductions", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
