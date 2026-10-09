"""Exercise public block-quantization APIs with independently encoded references."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantization import packed_reference

FORMATS = (
    ("mxfp4", 32, 4, False),
    ("mxfp8", 32, 8, False),
    ("nvfp4", 16, 4, False),
    ("nvfp4", 16, 4, True),
)


def reference(mode, groups=7, global_scale=False):
    """Use exact format values and power-of-two scales, not translated arithmetic."""
    if mode not in {"mxfp4", "mxfp8", "nvfp4"} or groups < 1:
        raise ValueError("Unsupported block reference")
    if global_scale and mode != "nvfp4":
        raise ValueError("Global scale requires nvfp4")
    group = 16 if mode == "nvfp4" else 32
    bits = 8 if mode == "mxfp8" else 4
    table = (
        [
            (448, 126),
            (-448, 254),
            (0.5, 48),
            (-0.5, 176),
            (1, 56),
            (-1, 184),
            (2, 64),
            (-2, 192),
        ]
        if bits == 8
        else [
            (0, 0),
            (0.5, 1),
            (1, 2),
            (1.5, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (6, 7),
            (0, 0),
            (-0.5, 9),
            (-1, 10),
            (-1.5, 11),
            (-2, 12),
            (-3, 13),
            (-4, 14),
            (-6, 15),
        ]
    )
    values, codes, scales = [], [], []
    for index in range(groups):
        exponent = index % 5 - 2
        zero = index == groups - 1
        values.append(
            [
                0.0 if zero else table[i % len(table)][0] * 2.0**exponent
                for i in range(group)
            ]
        )
        codes.extend(0 if zero else table[i % len(table)][1] for i in range(group))
        scales.append(
            0
            if zero
            else (
                56 + 8 * (exponent + int(global_scale))
                if mode == "nvfp4"
                else 127 + exponent
            )
        )
    return values, packed_reference(codes, bits), bytes(scales)


def cases():
    result = [
        {
            "name": (
                f"{mode}-{dtype}-global-{str(global_scale).lower()}-offset-{offset}"
            ),
            "mode": mode,
            "group": group,
            "bits": bits,
            "globalScale": global_scale,
            "dtype": dtype,
            "groups": 7,
            "offset": offset,
        }
        for mode, group, bits, global_scale in FORMATS
        for dtype in ("float32", "float16", "bfloat16")
        for offset in (0, 1)
    ]
    result.append(
        {
            "name": "nvfp4-float32-batch-boundary",
            "mode": "nvfp4",
            "group": 16,
            "bits": 4,
            "globalScale": True,
            "dtype": "float32",
            "groups": 65536,
            "offset": 1,
        }
    )
    return result


def verify(root, packages, output):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    (output / "adaptation.json").write_text(json.dumps(preparation, indent=2))
    host = HostRuntime(packages, output / "trace.jsonl", mlx_root=root)
    host.install()
    records = []
    compiled = set()
    try:
        for case in cases():
            for operation in ("quantize", "dequantize"):
                entry = (
                    f"{case['mode']}_{operation}_{SOURCE_TYPES[case['dtype']]}"
                    f"_gs_{case['group']}_b_{case['bits']}"
                    f"_hgs_{str(case['globalScale']).lower()}"
                )
                if entry not in compiled:
                    descriptor, package = host.quantization.get(entry)
                    compile_entry(
                        host.target, entry, descriptor, package, output / "compilation"
                    )
                    compiled.add(entry)
            values, expected_packed, expected_scales = reference(
                case["mode"], case["groups"], case["globalScale"]
            )
            dtype = getattr(mx, case["dtype"])
            storage_values = (
                [[123.0] * case["group"]] + values + [[-123.0] * case["group"]]
                if case["offset"]
                else values
            )
            storage = mx.array(storage_values, dtype=dtype)
            source = storage[1:-1] if case["offset"] else storage
            global_scale = (
                mx.array(1344.0, dtype=mx.float32) if case["globalScale"] else None
            )
            mx.eval(source)
            if global_scale is not None:
                mx.eval(global_scale)
            before = source.tolist()
            options = {
                "mode": case["mode"],
                "group_size": case["group"],
                "bits": case["bits"],
                "stream": mx.gpu,
            }
            if global_scale is not None:
                options["global_scale"] = global_scale
            start = host.dispatch_count
            packed, scales = mx.quantize(source, **options)
            restored = mx.dequantize(packed, scales, dtype=dtype, **options)
            mx.eval(packed, scales, restored)
            actual_packed = np.asarray(packed).astype("<u4", copy=False).tobytes()
            actual_scales = np.asarray(scales, dtype=np.uint8).tobytes()
            actual_values = restored.tolist()
            # Retain full input and output data even when a numerical check fails.
            (output / f"{case['name']}.json").write_text(
                json.dumps(
                    {
                        "case": case,
                        "input": before,
                        "globalScale": 1344.0 if global_scale is not None else None,
                        "packed": actual_packed.hex(),
                        "expectedPacked": expected_packed.hex(),
                        "scales": actual_scales.hex(),
                        "expectedScales": expected_scales.hex(),
                        "restored": actual_values,
                        "expectedRestored": values,
                    },
                    indent=2,
                )
            )
            if (
                packed.dtype != mx.uint32
                or scales.dtype != mx.uint8
                or restored.dtype != dtype
            ):
                raise AssertionError(f"Block output dtypes differ: {case['name']}")
            if (
                packed.shape != (case["groups"], case["group"] * case["bits"] // 32)
                or scales.shape != (case["groups"], 1)
                or restored.shape != source.shape
            ):
                raise AssertionError(f"Block output shapes differ: {case['name']}")
            if actual_packed != expected_packed or actual_scales != expected_scales:
                raise AssertionError(f"Block encoding differs: {case['name']}")
            if (
                actual_values != values
                or source.tolist() != before
                or storage.tolist() != storage_values
            ):
                raise AssertionError(f"Block values or source differ: {case['name']}")
            if global_scale is not None and global_scale.item() != 1344.0:
                raise AssertionError("Global scale was modified")
            count = host.dispatch_count - start
            if count != 2 * ((case["groups"] + 65534) // 65535):
                raise AssertionError(f"Missing block native dispatches: {case['name']}")
            record = {
                **case,
                "dispatchStart": start,
                "dispatchCount": count,
                "packedHash": hashlib.sha256(actual_packed).hexdigest(),
                "scalesHash": hashlib.sha256(actual_scales).hexdigest(),
                "inputUnchanged": True,
            }
            records.append(record)
            (output / "results.json").write_text(
                json.dumps(
                    {
                        "target": host.target,
                        "fullUpstreamSuite": False,
                        "cases": records,
                    },
                    indent=2,
                )
            )
            print(json.dumps(record), flush=True)
        if verify_prepared(root) != preparation:
            raise AssertionError("Prepared source changed during block execution")
    finally:
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    return records


def compile_entry(target, entry, descriptor, package, output):
    """Validate the exact packaged artifact before executing the public API."""
    output.mkdir(parents=True, exist_ok=True)
    source = package / descriptor["artifact"]["packagePath"]
    expected = descriptor["artifact"]["hash"]["value"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Block compilation source identity differs")
    module = output / (
        entry + {"metal": ".air", "opengl": ".spv", "directx": ".dxil"}[target]
    )
    if target == "metal":
        commands = [
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metal",
                "-Werror",
                "-fno-fast-math",
                "-c",
                str(source),
                "-o",
                str(module),
            ]
        ]
    elif target == "directx":
        commands = [
            [
                "dxc",
                "-T",
                "cs_6_6",
                "-E",
                descriptor["entryPoint"]["name"],
                "-HV",
                "2021",
                "-WX",
                *dxc_compiler_arguments_for_source(source.read_text()),
                str(source),
                "-Fo",
                str(module),
            ]
        ]
    else:
        commands = [
            [
                "glslangValidator",
                "--target-env",
                "opengl",
                "--target-env",
                "spirv1.3",
                "-S",
                "comp",
                str(source),
                "-o",
                str(module),
            ],
            ["spirv-val", "--target-env", "spv1.3", str(module)],
        ]
    record = {
        "entry": entry,
        "target": target,
        "artifact": descriptor["artifact"],
        "steps": [],
        "status": "failed",
    }
    try:
        for command in commands:
            step = {"command": command}
            record["steps"].append(step)
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=120, check=False
            )
            step.update(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            )
            if result.returncode:
                raise RuntimeError(f"Block toolchain validation failed: {entry}")
        if not module.is_file() or not module.stat().st_size:
            raise RuntimeError("Block compiler did not produce a module")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("Block compilation source changed")
        record.update(
            status="passed",
            module=module.name,
            moduleHash=hashlib.sha256(module.read_bytes()).hexdigest(),
        )
    finally:
        (output / (entry + ".json")).write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
