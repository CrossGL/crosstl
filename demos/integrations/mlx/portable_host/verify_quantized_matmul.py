"""Exercise MLX quantized_matmul through translated affine matrix-vector kernels.

Transposed cases reach the qmv kernels and untransposed ones qvm, or for ``K``
of 1024 and more qvm_split_k and the column sum over its partitions.

Every case routes one ``mx.quantized_matmul`` call to one translated entry and
checks each output bit against an exact float64 reference: inputs lie on
dyadic grids that keep every float32 partial sum exact, whatever its order.
16-bit float split-K products round each partition's product to their type and
sum the partitions in it. Below 32 partitions their reference rounds at the
same points, in the order the small column sum adds. From 32 partitions the
looped column sum adds through simd reductions, whose order Metal leaves
unspecified, so those cases keep every partial product and every sum of them
an integer that the element type holds exactly (``order_free``).
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
    # qvm_split_k: 8 whole partitions over three folded rows, a short final
    # partition, 32 partitions of stacked weights (a looped column sum) and
    # stacked weights under a strided batch (a small column sum over an outer
    # batch axis).
    ("qvm-split-k-rows", "float32", 32, 2, 2048, 128, (3,), (), "dense", False),
    ("qvm-split-k-tail", "float32", 64, 4, 1100, 64, (1,), (), "dense", False),
    ("qvm-split-k-wide", "float32", 128, 2, 9000, 256, (2, 1), (2,), "dense", False),
    ("qvm-split-k-strided", "float32", 32, 8, 1024, 32, (3, 1), (3,), "strided", False),
    # 16-bit float qvm_split_k: 8 partitions summed in float16 over folded
    # rows, and in bfloat16 over an outer batch axis; 32 partitions (looped
    # sums) of stacked float16 weights, and of bfloat16 folded rows with a
    # short final partition.
    ("qvm-split-k-half", "float16", 64, 4, 2048, 128, (3,), (), "dense", False),
    (
        "qvm-split-k-bfloat-strided",
        "bfloat16",
        32,
        3,
        1024,
        64,
        (3, 1),
        (3,),
        "strided",
        False,
    ),
    (
        "qvm-split-k-half-wide",
        "float16",
        128,
        2,
        9000,
        256,
        (2, 1),
        (2,),
        "dense",
        False,
    ),
    ("qvm-split-k-bfloat-wide", "bfloat16", 64, 3, 8200, 64, (3,), (), "dense", False),
)
# Products the host refuses, as (case, the error it raises). quantized.cpp
# sends a strided x batch over 2-D weights to qvm_split_k, whose single weight
# batch axis (the partitions) adjust_matrix_offsets never wraps: every batch
# after the first would read past w.
REFUSALS = (
    (
        (
            "qvm-split-k-shared-weights",
            "float32",
            32,
            8,
            1024,
            32,
            (3, 1),
            (),
            "strided",
            False,
        ),
        "batched inputs and 2-D weights",
    ),
)
SCALES = (1.0, 0.5, 0.25, 0.125)
# The significant bits of each 16-bit float type: it holds every integer up
# to 2 ** PRECISION[dtype] in magnitude exactly.
PRECISION = {"float16": 11, "bfloat16": 8}


def case_record(case):
    name, dtype, group, bits, k, n, x_prefix, w_batch, layout, transpose = case
    m = x_prefix[-1]
    batches = math.prod(x_prefix[:-1]) if len(x_prefix) > 1 else 1
    if not w_batch and layout == "dense":
        # w.ndim() == 2 and a row-contiguous x: MLX folds every x row into M.
        m, batches = math.prod(x_prefix), 1
    entry = matmul_layout.select(dtype, group, bits, m, n, k, batches, transpose)
    split = matmul_layout.signature(entry).get("splitK")
    width = -(-k // split) if split else k
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
        "entry": entry,
        "dispatches": [entry, *split_sum(entry, m, n)],
        # qvm_split_k's partitions: split_d columns each, the last shorter.
        "partitions": [[start, min(start + width, k)] for start in range(0, k, width)],
    }


def split_sum(entry, m, n):
    """The column sum reduce.cpp runs over a split-K product's partitions."""
    info = matmul_layout.signature(entry)
    if "splitK" not in info:
        return []
    # strided_reduce_small below 32 partitions, otherwise the looped pass.
    kind = "small_1" if info["splitK"] < 32 else "looped_1_32_32"
    return [f"col_reduce_{kind}_reduce_sum{info['dtype']}"]


def small_column_variants():
    """The small column sum variants the split-K cases launch, as cache keys."""
    from demos.integrations.mlx.portable_host import column_small_layout

    keys = set()
    for case in CASES:
        record = case_record(case)
        for entry in record["dispatches"][1:]:
            if entry in column_small_layout.ALL_SMALL_COLUMN_ENTRIES:
                split = matmul_layout.signature(record["entry"])["splitK"]
                lanes, rows = column_small_layout.geometry(
                    record["M"] * record["N"], split
                )
                keys.add(column_small_layout.variant(entry, (lanes, rows, 1)))
    return keys


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
    magnitudes = rng.choice(SCALES, size=(*w_batch, rows, columns // group))
    signs = rng.choice((-1.0, 1.0), size=magnitudes.shape)
    # Unit scales keep an order-free case's dequantized weights integers.
    scales = signs if order_free(record) else magnitudes * signs
    biases = -rng.integers(0, bins + 1, size=scales.shape) * scales
    x = rng.integers(-4, 5, size=x_shape) / 4.0
    if order_free(record):
        x = sparse_rows(
            rng, x_shape, record["partitions"], 2 ** PRECISION[record["dtype"]] // bins
        )
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
    if order_free(record):
        expected = exact_sum(np, record, bound_x, weights)
    elif record["dtype"] != "float32" and len(record["partitions"]) > 1:
        expected = partition_sum(np, record, bound_x, weights)
    else:
        expected = rounded(np.matmul(bound_x, weights), record["dtype"])
    return {
        "w": packed.astype(np.uint32),
        "scales": scales,
        "biases": biases,
        "x": x,
        "expected": expected,
    }


def order_free(record):
    """Whether a 16-bit float split-K case's partition sum has no fixed order.

    From 32 partitions reduce.cpp sums them in the looped column pass, and
    above 256 in the two-pass reduction; both add through simd reductions.
    """
    return record["dtype"] != "float32" and len(record["partitions"]) >= 32


def sparse_rows(rng, shape, partitions, budget):
    """x rows that hold one -1 or 1 in each active partition, zeros elsewhere.

    With unit scales each active partition's product is then one dequantized
    weight, an integer of at most ``2 ** bits - 1`` in magnitude. ``budget``
    partitions at most, spread evenly over K, are active.
    """
    import numpy as np

    count = min(len(partitions), budget)
    active = [partitions[index * len(partitions) // count] for index in range(count)]
    x = np.zeros(shape)
    for row in x.reshape(-1, shape[-1]):
        for start, stop in active:
            row[rng.integers(start, stop)] = rng.choice((-1.0, 1.0))
    return x


def exact_sum(np, record, x, weights):
    """A 16-bit float split-K product that every summation order gives exactly.

    Each partition's product is an integer and, for each output, their
    magnitudes total at most ``2 ** PRECISION[dtype]``: every partial product
    and every sum of some of them is an integer the element type holds, so
    no store or addition rounds.
    """
    dtype = record["dtype"]
    partials = np.stack(
        [np.matmul(x[..., a:b], weights[..., a:b, :]) for a, b in record["partitions"]]
    )
    if (
        not np.array_equal(partials, np.rint(partials))
        or np.abs(partials).sum(axis=0).max() > 2 ** PRECISION[dtype]
    ):
        raise ValueError("16-bit float partition sums are not exact in every order")
    return rounded(partials.sum(axis=0), dtype)


def partition_sum(np, record, x, weights):
    """A 16-bit float split-K product as the kernels round and sum it.

    Each partition's exact float32 product rounds to the element type when the
    product kernel stores it. strided_reduce_small then gives threadgroup row
    ``j`` the partitions ``j``, ``j + rows``, ... to add into an accumulator of
    the element type from zero, and row 0 adds rows 1, 2, ... into its own.
    Every addition rounds: a float32 sum of two such values rounds once to
    float16 or bfloat16 correctly, as the targets' 16-bit additions do.
    """
    from demos.integrations.mlx.portable_host import column_small_layout

    dtype, partitions = record["dtype"], record["partitions"]
    if len(partitions) >= 32:
        raise ValueError("16-bit float partition sums take the small column pass")
    partials = [
        rounded(np.matmul(x[..., a:b], weights[..., a:b, :]), dtype)
        for a, b in partitions
    ]
    _, rows = column_small_layout.geometry(record["M"] * record["N"], len(partials))
    sums = []
    for row in range(rows):
        total = np.zeros_like(partials[0])
        for partial in partials[row::rows]:
            total = rounded(partial + total, dtype)
        sums.append(total)
    result = sums[0]
    for total in sums[1:]:
        result = rounded(total + result, dtype)
    return result


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
    if count != len(record["dispatches"]):
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


def run_refusal(mx, np, host, refusal, seed, record_for=None):
    """Check that one product raises ``message`` before any dispatch.

    ``record_for`` describes the case; it defaults to ``case_record``.
    """
    case, message = refusal
    record = (record_for or case_record)(case)
    data = case_data(record, seed)
    dtype = getattr(mx, record["dtype"])
    arrays = {
        name: mx.array(data[name].tolist(), dtype=dtype)
        for name in ("scales", "biases", "x")
    }
    arrays["w"] = mx.array(data["w"].tolist(), dtype=mx.uint32)
    mx.eval(*arrays.values())
    x = arrays["x"][::2] if record["layout"] == "strided" else arrays["x"]
    start = host.dispatch_count
    try:
        mx.eval(
            mx.quantized_matmul(
                x,
                arrays["w"],
                arrays["scales"],
                arrays["biases"],
                transpose=record["transpose"],
                group_size=record["groupSize"],
                bits=record["bits"],
            )
        )
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(
                f"Quantized matmul refusal differs: {record['name']}: {error}"
            ) from error
    else:
        raise AssertionError(
            f"Quantized matmul refusal did not raise: {record['name']}"
        )
    if host.dispatch_count != start:
        raise AssertionError(f"Quantized matmul refusal dispatched: {record['name']}")
    return {"name": record["name"], "entry": record["entry"], "message": message}


def validate_trace(trace, records, target):
    expected = [entry for record in records for entry in record["dispatches"]]
    if [event.get("entry") for event in trace] != expected or any(
        event.get("target") != target for event in trace
    ):
        raise RuntimeError("Quantized matmul trace does not match the routed entries")
    products = [event for event in trace if event["entry"].startswith("affine_")]
    for event, record in zip(products, records):
        metadata = event["matmulMetadata"]
        if (
            metadata["M"],
            metadata["N"],
            metadata["K"],
            metadata["batches"],
        ) != (record["M"], record["N"], record["K"], record["batches"]):
            raise RuntimeError(f"Quantized matmul launch differs: {record['name']}")
    entries = [record["entry"] for record in records]
    kernels = {matmul_layout.signature(entry)["kernel"] for entry in entries}
    batched = {matmul_layout.signature(entry)["batched"] for entry in entries}
    sums = {entry for entry in expected if entry.startswith("col_reduce_")}
    if (
        kernels != {"qmv_quad", "qmv_fast", "qmv", "qvm", "qvm_split_k"}
        or batched != {False, True}
        or sums != {
            f"col_reduce_{kind}_reduce_sum{dtype}"
            for kind in ("small_1", "looped_1_32_32")
            for dtype in ("float32", "float16", "bfloat16")
        }
    ):
        raise RuntimeError("Quantized matmul cases do not cover every route")


def verify(root, packages, output, reductions=None):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    (output / "adaptation.json").write_text(json.dumps(preparation, indent=2))
    host = HostRuntime(
        packages, output / "trace.jsonl", mlx_root=root, reductions=reductions
    )
    host.install()
    records, refusals = [], []
    try:
        for seed, case in enumerate(CASES):
            record = run_case(mx, np, host, case_record(case), seed)
            records.append(record)
            (output / "results.json").write_text(
                json.dumps({"target": host.target, "cases": records}, indent=2)
            )
            print(json.dumps(record), flush=True)
        for seed, refusal in enumerate(REFUSALS, start=len(CASES)):
            refusals.append(run_refusal(mx, np, host, refusal, seed))
        (output / "results.json").write_text(
            json.dumps(
                {"target": host.target, "cases": records, "refusals": refusals},
                indent=2,
            )
        )
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
    parser.add_argument(
        "--reductions", type=Path, help="the looped column sum's packages"
    )
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir, args.reductions)
