"""Exercise MLX quantized_matmul through translated affine matrix-vector kernels.

Transposed cases reach the qmv kernels and untransposed ones qvm.

Every case routes one ``mx.quantized_matmul`` call to one translated entry and
checks each output bit against an exact float64 reference: inputs lie on
dyadic grids that keep every float32 partial sum exact, whatever its order.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from demos.integrations.mlx.portable_host import matmul_layout
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantization import packed_reference

# (name, dtype, group, bits, K, N, x shape before K, w batch shape, layout,
# transpose). Layouts: "dense" binds x as given, "strided" takes every other x
# batch.
CASES = (
    ("quad-d64-partial-tile", "float32", 64, 4, 64, 72, (1,), (), "dense", True),
    ("quad-d128-rows", "float16", 32, 2, 128, 64, (3, 1), (), "dense", True),
    ("quad-d128-stacked", "bfloat16", 128, 8, 128, 100, (2, 1), (2,), "dense", True),
    ("fast-rows", "float32", 128, 4, 512, 16, (2,), (), "dense", True),
    ("fast-stacked", "float16", 64, 6, 256, 24, (3, 1), (3,), "dense", True),
    ("qmv-unaligned-rows", "bfloat16", 32, 3, 256, 67, (1,), (), "dense", True),
    (
        "qmv-broadcast-weights",
        "float32",
        32,
        5,
        96,
        40,
        (2, 2, 1),
        (2, 1),
        "dense",
        True,
    ),
    ("qmv-shared-weights", "float16", 64, 8, 192, 33, (3, 1), (), "strided", True),
    (
        "qmv-two-batch-axes",
        "float32",
        128,
        6,
        384,
        16,
        (2, 3, 1),
        (2, 3),
        "dense",
        True,
    ),
    # qvm: a K tail with a half-empty last tile, a K shorter than one block
    # over broadcast weights, whole blocks, shared weights and two batch axes.
    ("qvm-tail-rows", "float32", 32, 3, 67, 96, (3,), (), "dense", False),
    (
        "qvm-short-k-broadcast",
        "bfloat16",
        128,
        6,
        20,
        256,
        (3, 1),
        (1,),
        "dense",
        False,
    ),
    ("qvm-stacked", "float16", 64, 4, 256, 128, (2, 1), (2,), "dense", False),
    ("qvm-two-bit-rows", "bfloat16", 64, 2, 512, 64, (2,), (), "dense", False),
    ("qvm-shared-weights", "float32", 32, 8, 128, 64, (2, 1), (), "strided", False),
    ("qvm-two-batch-axes", "float16", 32, 5, 96, 32, (2, 3, 2), (2, 3), "dense", False),
)
SCALES = (1.0, 0.5, 0.25, 0.125)


def case_record(case):
    name, dtype, group, bits, k, n, x_prefix, w_batch, layout, transpose = case
    m = x_prefix[-1]
    batches = math.prod(x_prefix[:-1]) if len(x_prefix) > 1 else 1
    if not w_batch and layout == "dense":
        # w.ndim() == 2 and a row-contiguous x: MLX folds every x row into M.
        m, batches = math.prod(x_prefix), 1
    return {
        "name": name,
        "dtype": dtype,
        "groupSize": group,
        "bits": bits,
        "K": k,
        "N": n,
        "xShape": [*x_prefix, k],
        "wBatchShape": list(w_batch),
        "layout": layout,
        "transpose": transpose,
        "M": m,
        "batches": batches,
        "entry": matmul_layout.select(dtype, group, bits, m, n, k, batches, transpose),
    }


def case_data(record, seed):
    """Codes, scales, biases and x on dyadic grids, and the exact outputs."""
    import numpy as np

    rng = np.random.default_rng(seed)
    k, n, group, bits = record["K"], record["N"], record["groupSize"], record["bits"]
    bins = (1 << bits) - 1
    w_batch = tuple(record["wBatchShape"])
    x_shape = tuple(record["xShape"])
    if record["layout"] == "strided":
        x_shape = (2 * x_shape[0], *x_shape[1:])
    # Quantization groups run along w's last axis: K when transposed, N not.
    rows, columns = (n, k) if record["transpose"] else (k, n)
    codes = rng.integers(0, bins + 1, size=(*w_batch, rows, columns))
    scales = rng.choice(SCALES, size=(*w_batch, rows, columns // group)) * rng.choice(
        (-1.0, 1.0), size=(*w_batch, rows, columns // group)
    )
    biases = -rng.integers(0, bins + 1, size=scales.shape) * scales
    x = rng.integers(-4, 5, size=x_shape) / 4.0
    packed = np.stack(
        [
            np.frombuffer(packed_reference([int(c) for c in row], bits), dtype="<u4")
            for row in codes.reshape(-1, columns)
        ]
    ).reshape(*w_batch, rows, columns * bits // 32)
    weights = codes * np.repeat(scales, group, axis=-1) + np.repeat(
        biases, group, axis=-1
    )
    if record["transpose"]:
        weights = np.swapaxes(weights, -1, -2)
    bound = np.abs(x).max() * k * (bins + bins)
    if bound >= 2**19:
        raise ValueError("Quantized matmul case exceeds the exact float32 budget")
    bound_x = x[::2] if record["layout"] == "strided" else x
    total = np.matmul(bound_x, weights)
    return {
        "w": packed.astype(np.uint32),
        "scales": scales,
        "biases": biases,
        "x": x,
        "expected": rounded(total, record["dtype"]),
    }


def rounded(values, dtype):
    """Round exact float32 totals to ``dtype`` as the kernels' final cast does."""
    import numpy as np

    single = values.astype(np.float32)
    if not np.array_equal(single.astype(np.float64), values):
        raise ValueError("Quantized matmul reference is not exact in float32")
    if dtype == "float32":
        return single
    if dtype == "float16":
        return single.astype(np.float16).astype(np.float32)
    bits = single.view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return bits.astype(np.uint32).view(np.float32)


def float_bits(values):
    return b"".join(struct.pack("<f", float(value)) for value in values)


def run_case(mx, np, host, record, seed):
    data = case_data(record, seed)
    dtype = getattr(mx, record["dtype"])
    arrays = {
        name: mx.array(data[name].tolist(), dtype=dtype)
        for name in ("scales", "biases", "x")
    }
    arrays["w"] = mx.array(data["w"].tolist(), dtype=mx.uint32)
    mx.eval(*arrays.values())
    before = {
        name: float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for name, value in arrays.items()
        if name != "w"
    }
    x = arrays["x"][::2] if record["layout"] == "strided" else arrays["x"]
    start = host.dispatch_count
    y = mx.quantized_matmul(
        x,
        arrays["w"],
        arrays["scales"],
        arrays["biases"],
        transpose=record["transpose"],
        group_size=record["groupSize"],
        bits=record["bits"],
    )
    mx.eval(y)
    count = host.dispatch_count - start
    actual = np.asarray(y.tolist(), dtype=np.float32)
    expected = data["expected"]
    if actual.shape != expected.shape or y.dtype != dtype:
        raise AssertionError(
            f"Quantized matmul output layout differs: {record['name']}"
        )
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"Quantized matmul output differs: {record['name']}")
    if count != 1:
        raise AssertionError(
            f"Quantized matmul dispatch count differs: {record['name']}"
        )
    for name, value in arrays.items():
        if name != "w" and before[name] != float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"Quantized matmul input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        # MLX's CPU kernel accumulates in the element type; float32 is exact.
        reference = mx.quantized_matmul(
            x,
            arrays["w"],
            arrays["scales"],
            arrays["biases"],
            transpose=record["transpose"],
            group_size=record["groupSize"],
            bits=record["bits"],
            stream=mx.cpu,
        )
        cpu = np.asarray(reference.tolist(), dtype=np.float32)
        if cpu.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(
                f"Quantized matmul differs from MLX CPU: {record['name']}"
            )
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu is not None,
        "inputUnchanged": True,
    }


def validate_trace(trace, records, target):
    expected = [record["entry"] for record in records]
    if [event.get("entry") for event in trace] != expected or any(
        event.get("target") != target for event in trace
    ):
        raise RuntimeError("Quantized matmul trace does not match the routed entries")
    for event, record in zip(trace, records):
        metadata = event["matmulMetadata"]
        if (
            metadata["M"],
            metadata["N"],
            metadata["K"],
            metadata["batches"],
        ) != (record["M"], record["N"], record["K"], record["batches"]):
            raise RuntimeError(f"Quantized matmul launch differs: {record['name']}")
    kernels = {matmul_layout.signature(entry)["kernel"] for entry in expected}
    batched = {matmul_layout.signature(entry)["batched"] for entry in expected}
    if kernels != {"qmv_quad", "qmv_fast", "qmv", "qvm"} or batched != {False, True}:
        raise RuntimeError("Quantized matmul cases do not cover every route")


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
        for seed, case in enumerate(CASES):
            record = run_case(mx, np, host, case_record(case), seed)
            records.append(record)
            (output / "results.json").write_text(
                json.dumps({"target": host.target, "cases": records}, indent=2)
            )
            print(json.dumps(record), flush=True)
    finally:
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    trace = [
        json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()
    ]
    validate_trace(trace, records, host.target)
    if verify_prepared(root) != preparation:
        raise RuntimeError("MLX sources changed during quantized matmul verification")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
