"""Exercise MLX affine APIs through translated kernels and retain native evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime


def packed_reference(codes, bits):
    """Pack little-endian quantization codes without using a shader implementation."""
    result, accumulator, occupied = bytearray(), 0, 0
    for code in codes:
        if type(code) is not int or not 0 <= code < 1 << bits:
            raise ValueError("Invalid affine reference code")
        accumulator |= code << occupied
        occupied += bits
        while occupied >= 8:
            result.append(accumulator & 255)
            accumulator >>= 8
            occupied -= 8
    if occupied:
        raise ValueError("Affine reference ends with an incomplete packed byte")
    return bytes(result)


def cases():
    result = [
        {
            "name": f"{dtype}-{rows}x512",
            "dtype": dtype,
            "shape": (rows, 512),
            "group": 32,
            "bits": 2,
            "strided": False,
        }
        for dtype in ("float32", "float16", "bfloat16")
        for rows in (128, 256)
    ]
    result.extend(
        {
            "name": f"float32-gs{group}-b{bits}",
            "dtype": "float32",
            "shape": (2, 256),
            "group": group,
            "bits": bits,
            "strided": False,
        }
        for group, bits in ((64, 3), (128, 4), (128, 5), (64, 6), (32, 8))
    )
    result.append(
        {
            "name": "strided-float32",
            "dtype": "float32",
            "shape": (2, 256),
            "group": 64,
            "bits": 3,
            "strided": True,
        }
    )
    result.append(
        {
            "name": "batch-boundary-float32",
            "dtype": "float32",
            "shape": (4096, 512),
            "group": 32,
            "bits": 2,
            "strided": False,
            "tailScale": 2,
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
    try:
        for case in cases():
            bins = (1 << case["bits"]) - 1
            size = int(np.prod(case["shape"]))
            raw = np.tile(
                np.array([0, 0.5, 1.5, bins], dtype=np.float32), size // 4
            ).reshape(case["shape"])
            raw.reshape(-1)[-case["group"] :] *= case.get("tailScale", 1)
            if case["strided"]:
                storage = np.full(
                    (case["shape"][0], case["shape"][1] * 2), 123.0, dtype=np.float32
                )
                storage[:, ::2] = raw
                source = mx.array(storage.tolist(), dtype=getattr(mx, case["dtype"]))[
                    :, ::2
                ]
            else:
                source = mx.array(raw.tolist(), dtype=getattr(mx, case["dtype"]))
            # Input construction and readback are host operations; the two MLX APIs
            # below must use the registered native backend, never their CPU fallback.
            mx.eval(source)
            before = np.asarray(source.tolist(), dtype=np.float32).tobytes()
            start = host.dispatch_count
            packed, scales, biases = mx.quantize(
                source, group_size=case["group"], bits=case["bits"]
            )
            restored = mx.dequantize(
                packed, scales, biases, group_size=case["group"], bits=case["bits"]
            )
            mx.eval(packed, scales, biases, restored)
            actual = np.asarray(packed).astype("<u4", copy=False).tobytes()
            expected = packed_reference(
                [bins, bins, bins - 1, 0] * (size // 4), case["bits"]
            )
            if actual != expected:
                raise AssertionError(f"Packed affine codes differ: {case['name']}")
            scale_values = np.asarray(scales.tolist(), dtype=np.float32)
            bias_values = np.asarray(biases.tolist(), dtype=np.float32)
            reconstructed = np.asarray(restored.tolist(), dtype=np.float32)
            expected_shape = (*case["shape"][:-1], case["shape"][-1] // case["group"])
            if (
                scales.shape != expected_shape
                or biases.shape != expected_shape
                or restored.shape != case["shape"]
            ):
                raise AssertionError(f"Affine output shapes differ: {case['name']}")
            if packed.dtype != mx.uint32 or any(
                value.dtype != getattr(mx, case["dtype"])
                for value in (scales, biases, restored)
            ):
                raise AssertionError(f"Affine output dtypes differ: {case['name']}")
            expected_scales = np.full(expected_shape, -1, dtype=np.float32)
            expected_biases = np.full(expected_shape, bins, dtype=np.float32)
            expected_scales.reshape(-1)[-1] *= case.get("tailScale", 1)
            expected_biases.reshape(-1)[-1] *= case.get("tailScale", 1)
            if not np.array_equal(scale_values, expected_scales) or not np.array_equal(
                bias_values, expected_biases
            ):
                raise AssertionError(f"Affine scale or bias differs: {case['name']}")
            expected_values = np.tile(
                np.array([0, 0, 1, bins], dtype=np.float32), size // 4
            ).reshape(case["shape"])
            expected_values.reshape(-1)[-case["group"] :] *= case.get("tailScale", 1)
            if not np.array_equal(
                reconstructed.view(np.uint32), expected_values.view(np.uint32)
            ):
                raise AssertionError(f"Affine reconstruction differs: {case['name']}")
            if before != np.asarray(source.tolist(), dtype=np.float32).tobytes():
                raise AssertionError(f"Affine source was modified: {case['name']}")
            chunks = (size // case["group"] + 65534) // 65535
            count = host.dispatch_count - start
            if count != chunks * 2 + int(case["strided"]):
                raise AssertionError(
                    f"Missing affine native dispatches: {case['name']}"
                )
            record = {
                **case,
                "dispatchStart": start,
                "dispatchCount": count,
                "values": size,
                "packedHash": hashlib.sha256(actual).hexdigest(),
                "restoredHash": hashlib.sha256(reconstructed.tobytes()).hexdigest(),
                "inputUnchanged": True,
            }
            records.append(record)
            (output / "results.json").write_text(
                json.dumps({"target": host.target, "cases": records}, indent=2)
            )
            print(json.dumps(record), flush=True)
    finally:
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
