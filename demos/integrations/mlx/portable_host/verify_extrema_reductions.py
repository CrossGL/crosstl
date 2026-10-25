"""Verify source-typed 16-bit float extrema with exact references and receipts.

``mx.min`` and ``mx.max`` over bfloat16 and float16 arrays run the translated
whole-array and small-row reduction kernels. Extrema select an input element,
so every native result must equal the independent reference word for word,
NaN payloads aside.
"""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    bfloat_storage,
    half_storage,
    reduction_layout,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.reduction_packages import (
    BFLOAT_ENTRIES,
    HALF_ENTRIES,
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


@dataclass(frozen=True)
class Format:
    """A 16-bit float storage format and the reduction packages that serve it."""

    dtype: str
    family: str
    entries: dict
    storage: object
    # The bits of a canonical quiet NaN and the exponent field.
    nan: int
    exponent: int
    # Profiles beyond the common ones; bfloat subnormal comparisons are
    # refused off Metal (CrossGL/crosstl#2000).
    profiles: tuple = ()

    def word(self, value):
        if self.dtype == "float16":
            return struct.unpack("<H", struct.pack("<e", value))[0]
        return struct.unpack("<I", struct.pack("<f", value))[0] >> 16

    def number(self, word):
        if self.dtype == "float16":
            return struct.unpack("<e", struct.pack("<H", word))[0]
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]


FORMATS = {
    "bfloat16": Format(
        "bfloat16", "bfloat", BFLOAT_ENTRIES, bfloat_storage, 0x7FC0, 0x7F80
    ),
    "float16": Format(
        "float16",
        "half",
        HALF_ENTRIES,
        half_storage,
        0x7E00,
        0x7C00,
        ("subnormal",),
    ),
}
BFLOAT = FORMATS["bfloat16"]


def cases(fmt=BFLOAT):
    for operation in ("min", "max"):
        for count in (125, 128, 509, 512, 1021, 1024, 4097, 65536):
            yield {"operation": operation, "shape": [count], "profile": "negative"}
        for profile in (
            "mixed",
            "nan-first",
            "nan-last",
            "infinity",
            "negative-zero",
            *fmt.profiles,
        ):
            yield {"operation": operation, "shape": [509], "profile": profile}
        for shape in ([3, 32], [33, 32], [512, 32], [1031, 7], [9, 5, 9]):
            yield {"operation": operation, "shape": shape, "profile": "negative"}


def reduce_words(values, operation, fmt=BFLOAT):
    decoded = [fmt.number(value) for value in values]
    if any(math.isnan(value) for value in decoded):
        return fmt.nan
    return fmt.word(
        (min if operation == "min" else max)(
            decoded, default=math.inf if operation == "min" else -math.inf
        )
    )


def reference(case, fmt=BFLOAT):
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
    elif profile == "subnormal":
        # Signed binary16 subnormals and zeros: multiples of 2**-24 below 2**-14.
        values = [math.ldexp(value, -24) for value in values]
    source = list(map(fmt.word, values))
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
    return source, [reduce_words(group, case["operation"], fmt) for group in groups]


def canonical(words, fmt=BFLOAT):
    if not isinstance(words, list) or any(
        type(value) is not int or not 0 <= value <= 0xFFFF for value in words
    ):
        raise ValueError("16-bit float reduction storage words are invalid")
    return [
        (
            fmt.nan
            if value & fmt.exponent == fmt.exponent and value & ~fmt.exponent & 0x7FFF
            else value
        )
        for value in words
    ]


def source_array(mx, words, shape, fmt):
    """An MLX array that holds exactly ``words``.

    Off Apple silicon MLX converts a Python float to float16 in software, which
    stores a NaN as 0x7D00, so the input storage would not be the reference's.
    A numpy binary16 buffer carries the reference bits unchanged. MLX stores
    the bfloat16 quiet NaN 0x7FC0 on every platform.
    """
    if fmt.dtype == "float16":
        import numpy as np

        bits = np.array(words, dtype=np.uint16)
        return mx.array(bits.view(np.float16)).reshape(shape)
    return mx.array(list(map(fmt.number, words)), dtype=getattr(mx, fmt.dtype)).reshape(
        shape
    )


def worker(args):
    import mlx.core as mx

    args.output_dir.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("An original GPU backend is already available")
    fmt, host = FORMATS[args.dtype], None
    if args.worker == "native":
        host = HostRuntime(
            args.packages,
            args.output_dir / "dispatch.jsonl",
            mlx_root=args.mlx_root,
            **{fmt.family: args.elementwise},
            reductions=args.reductions,
            retain_native_modules=True,
        )
        host.install()
    else:
        mx.set_default_device(mx.cpu)
    records = []
    for case in cases(fmt):
        source, expected = reference(case, fmt)
        value = source_array(mx, source, case["shape"], fmt)
        axes = (
            None
            if len(case["shape"]) == 1
            else (1 if len(case["shape"]) == 2 else (0, 2))
        )
        before = host.dispatch_count if host else 0
        output = getattr(mx, case["operation"])(value, axis=axes)
        mx.eval(output)
        actual = output.tolist()
        actual = list(map(fmt.word, actual if isinstance(actual, list) else [actual]))
        original = value.reshape(-1).tolist()
        record = {
            **case,
            "actualWords": actual,
            "dtype": str(output.dtype),
            "inputWords": list(map(fmt.word, original)),
            "shapeOut": list(output.shape),
            "dispatchCount": host.dispatch_count - before if host else 0,
        }
        records.append(record)
        write_json(args.output_dir / "results.json", records)
        if canonical(actual, fmt) != expected or canonical(
            record["inputWords"], fmt
        ) != canonical(source, fmt):
            raise ValueError(
                f"{fmt.dtype} reduction differs from independent reference: {case}"
            )


def validate(records, trace, *, native, fmt=BFLOAT):
    required, dtype = list(cases(fmt)), fmt.dtype
    if len(records) != len(required):
        raise ValueError(f"{dtype} reduction inventory differs")
    cursor = 0
    for case, record in zip(required, records):
        source, expected = reference(case, fmt)
        whole = len(case["shape"]) == 1
        dispatches = (2 if whole and len(source) > 4096 else 1) if native else 0
        if (
            any(record.get(key) != value for key, value in case.items())
            or canonical(record["actualWords"], fmt) != expected
            or canonical(record["inputWords"], fmt) != canonical(source, fmt)
            or record["dtype"] != f"mlx.core.{dtype}"
            or record["shapeOut"] != ([] if whole else [len(expected)])
            or type(record["dispatchCount"]) is not int
            or record["dispatchCount"] != dispatches
        ):
            raise ValueError(f"{dtype} reduction results differ")
        partials = source
        for event in trace[cursor : cursor + dispatches]:
            target = event["target"]
            input_packet = event["inputs"].get(
                "in_Buffer" if target == "opengl" else "in_"
            )
            source_words = partials if whole else source
            packet = {
                "dtype": physical_dtype(dtype, target),
                "shape": [len(source_words)],
                "values": fmt.storage.pack(source_words, target),
            }
            if fmt.storage.encoding(target):
                packet["encoding"] = fmt.storage.encoding(target)
            if input_packet != packet:
                raise ValueError(f"{dtype} reduction input storage differs")
            if event["dispatchVersion"] != 3 or event[
                "reductionGuardValues"
            ] != fmt.storage.pack(fmt.storage.GUARD, target):
                raise ValueError(f"{dtype} reduction guards or dispatch version differ")
            if whole:
                plan = reduction_layout.stage(len(partials), dtype)
                if (
                    event["entry"] != f'all_reduce_{case["operation"]}{dtype}'
                    or event["threads"] != len(partials)
                    or event["workgroupCount"] != plan["workgroupCount"]
                    or event["workgroupSize"] != plan["workgroupSize"]
                    or event["reductionMetadata"] != {
                        "in_size": len(partials),
                        "row_size": plan["rowSize"],
                    }
                ):
                    raise ValueError(f"{dtype} reduction pass plan differs")
                row_size = plan["rowSize"]
                partials = [
                    reduce_words(
                        partials[row * row_size : (row + 1) * row_size],
                        case["operation"],
                        fmt,
                    )
                    for row in range(plan["workgroupCount"][1])
                ]
            else:
                if event[
                    "entry"
                ] != f'row_reduce_small_1_reduce_{case["operation"]}{dtype}' or event[
                    "threads"
                ] != len(
                    source
                ):
                    raise ValueError(f"{dtype} small-row specialization differs")
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
                    raise ValueError(f"{dtype} small-row geometry or metadata differs")
                partials = expected
            actual = fmt.storage.unpack(event["reductionValues"], target)
            if canonical(actual, fmt) != partials:
                raise ValueError(f"{dtype} native partial reduction differs")
        cursor += dispatches
    if cursor != len(trace):
        raise ValueError(f"{dtype} reduction dispatch coverage differs")


def verify(args):
    fmt, output = FORMATS[args.dtype], args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    target = json.loads((args.packages / "index.json").read_text())["target"]
    index = load_index(args.reductions, target)
    if (
        index["family"] != fmt.family
        or set(index["entries"]) != set(fmt.entries)
        or set(index["widths"]) != set(WIDTHS)
    ):
        raise ValueError(
            f"{fmt.dtype} reduction proof requires its complete package inventory"
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
            f"{fmt.dtype} extrema {mode}",
            "--",
            sys.executable,
            "-m",
            __package__ + ".verify_extrema_reductions",
            "--worker",
            mode,
            "--dtype",
            fmt.dtype,
        ]
        for name in ("mlx_root", "packages", "elementwise", "reductions"):
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
            raise RuntimeError(f"{fmt.dtype} extrema {mode} failed; inspect {output}")
        records[mode] = json.loads((output / mode / "results.json").read_text())
    trace = [
        json.loads(line)
        for line in (output / "native/dispatch.jsonl").read_text().splitlines()
    ]
    validate(records["cpu"], [], native=False, fmt=fmt)
    validate(records["native"], trace, native=True, fmt=fmt)
    cache = SmallRowPackageCache(args.mlx_root, args.packages / "small-rows", target)
    for event in trace:
        if event["target"] != target:
            raise ValueError(f"{fmt.dtype} reduction target differs")
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
            raise ValueError(f"{fmt.dtype} reduction artifact differs")
        for descriptor, root in packages:
            artifact = descriptor["artifact"]
            if (
                hashlib.sha256(
                    (root / artifact["packagePath"]).read_bytes()
                ).hexdigest()
                != artifact["hash"]["value"]
            ):
                raise ValueError(f"{fmt.dtype} reduction source identity differs")
        if "regions" not in event["details"]:
            audit_half_events([event], output)
        else:
            regions = event["details"]["regions"]
            if len(regions) != len(packages):
                raise ValueError(f"{fmt.dtype} region coverage differs")
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
                    raise ValueError(f"{fmt.dtype} region identity differs")
    if before != verify_prepared(args.mlx_root):
        raise ValueError(f"{fmt.dtype} reduction proof changed pinned sources")
    write_json(
        output / "summary.json",
        {
            "passed": True,
            "commit": COMMIT,
            "target": target,
            "dtype": fmt.dtype,
            "workloads": len(records["native"]),
            "nativeDispatches": len(trace),
            "compiledEntries": len(index["descriptors"]),
            "adaptation": before,
            "fullUpstreamSuite": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx-root", "packages", "reductions", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dtype", choices=tuple(FORMATS), default="bfloat16")
    # The elementwise packages of the dtype's host family (--family bfloat or
    # --family half of packages.py).
    parser.add_argument("--elementwise", type=Path, required=True)
    parser.add_argument("--worker", choices=("cpu", "native"))
    args = parser.parse_args()
    worker(args) if args.worker else verify(args)
