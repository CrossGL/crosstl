"""Exercise MLX matmul through translated floating-point matmul kernels.

Every case routes one ``mx.matmul`` call to one translated ``gemv``,
``gemv_t`` or ``steel_gemm_fused`` entry, or to a ``gemm_splitk`` entry and
the accumulation that sums its partitions, and checks each output bit against
an exact float64 reference: inputs lie on dyadic grids that keep every float32
partial sum exact, whatever its order. Products MLX sends to other kernels
must fail with the host's explicit error.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    gemm_layout,
    gemm_splitk_layout,
    gemv_layout,
)
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

# (name, dtype, a, b). Each operand is (stored shape, view): "dense" binds the
# stored array, "swap" its last two axes swapped, "step" every other element
# of the stored last axis, ("broadcast", shape) a broadcast of it and
# ("strided", shape, strides) an as_strided view.
CASES = (
    ("vector-matrix-rows", "float32", ((1, 96), "dense"), ((96, 40), "dense")),
    ("vector-matrix-narrow", "float16", ((1, 33), "dense"), ((33, 3), "dense")),
    ("vector-matrix-wide", "bfloat16", ((1, 64), "dense"), ((64, 600), "dense")),
    ("vector-matrix-widest", "float32", ((1, 40), "dense"), ((40, 2050), "dense")),
    (
        "vector-matrix-batched",
        "float32",
        ((3, 1, 128), "dense"),
        ((3, 128, 64), "dense"),
    ),
    (
        "vector-matrix-shared-vector",
        "bfloat16",
        ((96,), ("strided", (4, 1, 96), (0, 96, 1))),
        ((4, 96, 32), "dense"),
    ),
    # MLX's broadcast zeroes the strides of size-1 axes, so check_transpose
    # copies the broadcast vector.
    (
        "vector-matrix-broadcast-vector",
        "float32",
        ((1, 1, 48), ("broadcast", (3, 1, 48))),
        ((3, 48, 20), "dense"),
    ),
    (
        "vector-matrix-two-batch-axes",
        "float16",
        ((2, 3, 1, 64), "dense"),
        ((1, 3, 64, 32), ("broadcast", (2, 3, 64, 32))),
    ),
    ("transposed-weights", "float32", ((1, 256), "dense"), ((67, 256), "swap")),
    ("transposed-weights-short-k", "float16", ((1, 64), "dense"), ((100, 64), "swap")),
    (
        "transposed-weights-long-k",
        "bfloat16",
        ((1, 1024), "dense"),
        ((48, 1024), "swap"),
    ),
    ("transposed-weights-tiny", "float32", ((1, 40), "dense"), ((2, 40), "swap")),
    (
        "transposed-weights-tiny-long-k",
        "float16",
        ((1, 128), "dense"),
        ((3, 128), "swap"),
    ),
    ("transposed-weights-tall", "float32", ((1, 72), "dense"), ((4100, 72), "swap")),
    (
        "transposed-weights-batched",
        "float32",
        ((8, 1, 64), "dense"),
        ((8, 256, 64), "swap"),
    ),
    # The base host packages copy 32-bit words only.
    ("strided-vector", "float32", ((1, 160), "step"), ((80, 24), "dense")),
    ("matrix-column", "float32", ((70, 128), "dense"), ((128, 1), "dense")),
    ("transposed-matrix-column", "float16", ((128, 50), "swap"), ((128, 1), "dense")),
    (
        "batch-folded-into-rows",
        "bfloat16",
        ((3, 4, 64), "dense"),
        ((64,), ("strided", (3, 64, 1), (0, 1, 1))),
    ),
    # Steel GEMM: every transposition and type, partial and whole tiles,
    # several tiles per axis, batches with and without has_batch, a batch
    # folded into M and a copied operand.
    ("gemm-nn-partial", "float32", ((33, 40), "dense"), ((40, 35), "dense")),
    ("gemm-nn-aligned", "float16", ((128, 64), "dense"), ((64, 64), "dense")),
    ("gemm-nt", "bfloat16", ((70, 48), "dense"), ((69, 48), "swap")),
    ("gemm-tn", "float32", ((40, 65), "swap"), ((40, 66), "dense")),
    ("gemm-tt", "float16", ((24, 130), "swap"), ((70, 24), "swap")),
    ("gemm-wide", "float32", ((130, 70), "dense"), ((70, 200), "dense")),
    ("gemm-batched", "float32", ((3, 20, 30), "dense"), ((3, 30, 40), "dense")),
    (
        "gemm-batched-aligned",
        "bfloat16",
        ((2, 64, 32), "dense"),
        ((2, 32, 128), "dense"),
    ),
    (
        "gemm-two-batch-axes",
        "bfloat16",
        ((2, 3, 17, 24), "dense"),
        ((1, 3, 24, 19), ("broadcast", (2, 3, 24, 19))),
    ),
    (
        "gemm-batch-folded-into-rows",
        "float16",
        ((3, 5, 32), "dense"),
        ((32, 70), ("strided", (3, 32, 70), (0, 70, 1))),
    ),
    ("gemm-copied-operand", "float32", ((20, 64), "step"), ((32, 9), "dense")),
    # Split-K: every transposition, type and tile, aligned and unaligned M x N
    # and K, 4 to 32 partitions, partitions without a whole K tile, a batch
    # folded into M and a copied operand.
    ("splitk-nn-whole", "float32", ((16, 256), "dense"), ((256, 16), "dense")),
    ("splitk-nt-partial", "float16", ((33, 300), "dense"), ((35, 300), "swap")),
    ("splitk-tn-wide-tile", "bfloat16", ((200, 48), "swap"), ((200, 64), "dense")),
    ("splitk-tt-aligned", "float32", ((512, 64), "swap"), ((96, 512), "swap")),
    ("splitk-mixed-tile", "float16", ((20, 640), "dense"), ((640, 72), "dense")),
    ("splitk-tall-tile", "bfloat16", ((64, 640), "dense"), ((640, 32), "dense")),
    ("splitk-deep", "float32", ((8, 1024), "dense"), ((1024, 8), "dense")),
    (
        "splitk-batch-folded-into-rows",
        "float32",
        ((2, 8, 256), "dense"),
        ((256, 24), ("strided", (2, 256, 24), (0, 24, 1))),
    ),
    # 208 / 16 = 13 K tiles over 16 partitions: only the last one has work.
    ("splitk-copied-operand", "float32", ((24, 416), "step"), ((208, 24), "dense")),
)
# Products the host refuses, with the error it raises.
REFUSALS = (("dot-product", (1, 16), (16, 1), "single dot products"),)
# Translated kernels each family dispatches per product.
KERNELS = {"gemv": 1, "gemm": 1, "splitk": 2}


def _row_strides(shape):
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.insert(0, stride)
        stride *= extent
    return strides


def view_layout(stored, view):
    """The shape and strides MLX gives the view of a row-major array."""
    shape, strides = list(stored), _row_strides(stored)
    if view == "dense":
        return shape, strides
    if view == "swap":
        shape[-2:], strides[-2:] = shape[:-3:-1], strides[:-3:-1]
        return shape, strides
    if view == "step":
        shape[-1], strides[-1] = -(-shape[-1] // 2), 2 * strides[-1]
        return shape, strides
    if view[0] == "strided":
        return list(view[1]), list(view[2])
    kind, target = view
    if kind != "broadcast" or len(target) != len(shape):
        raise ValueError("Unsupported matmul operand view")
    # backend/common/broadcasting.cpp broadcast.
    return list(target), [
        0 if extent == 1 else stride for extent, stride in zip(shape, strides)
    ]


def case_record(case):
    name, dtype, (a_stored, a_view), (b_stored, b_view) = case
    a_shape, a_strides = view_layout(a_stored, a_view)
    b_shape, b_strides = view_layout(b_stored, b_view)
    family, route = gemm_layout.matmul_route(
        dtype, a_shape, a_strides, b_shape, b_strides
    )
    return {
        "name": name,
        "family": family,
        "dtype": dtype,
        "a": {"stored": list(a_stored), "view": a_view, "shape": a_shape},
        "b": {"stored": list(b_stored), "view": b_view, "shape": b_shape},
        "entry": route["entry"],
        "route": route,
        "copies": sum(route["copies"].values()),
    }


def _operand(mx, np, values, view, dtype):
    array = mx.array(values.tolist(), dtype=dtype)
    if view == "dense":
        return array, array
    if view == "swap":
        return array, mx.swapaxes(array, -1, -2)
    if view == "step":
        return array, array[..., ::2]
    if view[0] == "strided":
        return array, mx.as_strided(array, view[1], view[2])
    return array, mx.broadcast_to(array, view[1])


def _view(np, values, view):
    if view == "dense":
        return values
    if view == "swap":
        return np.swapaxes(values, -1, -2)
    if view == "step":
        return values[..., ::2]
    if view[0] == "strided":
        return np.lib.stride_tricks.as_strided(
            values, view[1], [stride * values.itemsize for stride in view[2]]
        )
    return np.broadcast_to(values, view[1])


def case_data(np, record, seed):
    """Stored operands on dyadic grids and the exact rounded product."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=record["a"]["stored"]) / 4.0
    b = rng.integers(-4, 5, size=record["b"]["stored"]) / 8.0
    total = np.matmul(
        _view(np, a, record["a"]["view"]), _view(np, b, record["b"]["view"])
    )
    if np.abs(total).max() >= 2**19:
        raise ValueError("Matmul case exceeds the exact float32 budget")
    return {"a": a, "b": b, "expected": rounded(total, record["dtype"])}


def float_bits(values):
    return b"".join(struct.pack("<f", float(value)) for value in values)


def run_case(mx, np, host, record, seed):
    data = case_data(np, record, seed)
    dtype = getattr(mx, record["dtype"])
    a_stored, a = _operand(mx, np, data["a"], record["a"]["view"], dtype)
    b_stored, b = _operand(mx, np, data["b"], record["b"]["view"], dtype)
    mx.eval(a_stored, b_stored)
    stored = {"a": a_stored, "b": b_stored}
    before = {
        name: float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for name, value in stored.items()
    }
    start = host.dispatch_count
    y = mx.matmul(a, b)
    mx.eval(y)
    count = host.dispatch_count - start
    actual = np.asarray(y.tolist(), dtype=np.float32)
    expected = data["expected"]
    if actual.shape != expected.shape or y.dtype != dtype:
        raise AssertionError(f"Matmul output layout differs: {record['name']}")
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"Matmul output differs: {record['name']}")
    # One translated copy for each operand matmul.cpp copies, then the product.
    if count != KERNELS[record["family"]] + record["copies"]:
        raise AssertionError(f"Matmul dispatch count differs: {record['name']}")
    for name, value in stored.items():
        if before[name] != float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"Matmul input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        reference = mx.matmul(a, b, stream=mx.cpu)
        cpu = np.asarray(reference.tolist(), dtype=np.float32)
        if cpu.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(f"Matmul differs from MLX CPU: {record['name']}")
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu is not None,
        "inputUnchanged": True,
    }


def run_refusal(mx, host, refusal):
    name, a_shape, b_shape, message = refusal
    a = mx.ones(a_shape, dtype=mx.float32)
    b = mx.ones(b_shape, dtype=mx.float32)
    mx.eval(a, b)
    start = host.dispatch_count
    try:
        mx.eval(mx.matmul(a, b))
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f"Matmul refusal differs: {name}: {error}") from error
    else:
        raise AssertionError(f"Matmul refusal did not raise: {name}")
    if host.dispatch_count != start:
        raise AssertionError(f"Matmul refusal dispatched: {name}")
    return {"name": name, "message": message}


def _gemv_launch_matches(event, record):
    route, metadata = record["route"], event["gemvMetadata"]
    parameters = event["parameters"]
    return (
        metadata["K"],
        metadata["outVecSize"],
        metadata["matrixLd"],
        metadata["batches"],
    ) == (route["K"], route["outVecSize"], route["matrixLd"], route["batches"]) and (
        parameters["batch_shape"],
        parameters["vector_batch_stride"],
        parameters["matrix_batch_stride"],
    ) == (
        route["batchShape"],
        route["vectorBatchStride"],
        route["matrixBatchStride"],
    )


def _gemm_launch_matches(event, record):
    route, metadata = record["route"], event["gemmMetadata"]
    parameters = event["parameters"]
    return (
        all(metadata[key] == route[key] for key in ("M", "N", "K", "batches"))
        and parameters["params"] == route["params"]
        and parameters["batch_shape"] == route["batchShape"]
        and parameters["batch_strides"] == route["batchStrides"]
    )


def _splitk_launch_matches(events, record):
    product, accum = events
    route = record["route"]
    metadata = product["splitkMetadata"]
    return (
        all(metadata[key] == route[key] for key in ("M", "N", "K", "partitions"))
        and product["parameters"]["params"] == route["params"]
        and accum["parameters"] == route["accumParams"]
        and accum["splitkMetadata"]["partitions"] == route["partitions"]
        # The accumulation reads exactly the partitions the product wrote.
        and accum["inputHashes"]["C_split"] == product["outputHash"]
    )


def _splitk_covered(records):
    signatures = [gemm_splitk_layout.signature(record["entry"]) for record in records]
    routes = [record["route"] for record in records]
    return (
        {(info["transposeA"], info["transposeB"]) for info in signatures}
        == {(False, False), (False, True), (True, False), (True, True)}
        and {info["dtype"] for info in signatures} == set(gemm_splitk_layout.DTYPES)
        and {(info["bm"], info["bn"]) for info in signatures}
        == {(16, 16), (16, 32), (32, 16), (32, 32)}
        and {info["mnAligned"] for info in signatures} == {False, True}
        and {info["kAligned"] for info in signatures} == {False, True}
        and {route["partitions"] for route in routes} >= {4, 8, 16, 32}
        and any(route["params"]["gemm_k_iterations_aligned"] == 0 for route in routes)
        and any(record["copies"] for record in records)
        and any(
            route["M"] > record["a"]["shape"][-2]
            for route, record in zip(routes, records)
        )
    )


def _gemv_covered(records):
    signatures = [gemv_layout.signature(record["entry"]) for record in records]
    tiles = {
        tuple(info[key] for key in ("kind", "bm", "bn", "sm", "sn", "tm", "tn"))
        for info in signatures
    }
    return (
        {info["kind"] for info in signatures} == {"gemv", "gemv_t"}
        and {info["nc"] for info in signatures} == {False, True}
        and {info["dtype"] for info in signatures} == set(gemv_layout.DTYPES)
        and len(tiles) >= 10
        and any(record["copies"] for record in records)
        and any(record["route"]["M"] > 1 for record in records)
    )


def _gemm_covered(records):
    signatures = [gemm_layout.signature(record["entry"]) for record in records]
    constants = [info["constants"] for info in signatures]
    routes = [record["route"] for record in records]
    return (
        {(info["transposeA"], info["transposeB"]) for info in signatures}
        == {(False, False), (False, True), (True, False), (True, True)}
        and {info["dtype"] for info in signatures} == set(gemm_layout.DTYPES)
        and all(
            {values[name] for values in constants} == {False, True}
            for name in ("has_batch", "align_M", "align_N", "align_K")
        )
        and any(
            min(route["params"]["tiles_m"], route["params"]["tiles_n"]) > 1
            for route in routes
        )
        and any(record["copies"] for record in records)
        and any(
            route["M"] > record["a"]["shape"][-2]
            for route, record in zip(routes, records)
        )
        and any(
            route["batches"] > 1 and len(route["batchShape"]) == 1 for route in routes
        )
    )


def _single(matches):
    return lambda events, record: matches(events[0], record)


def _entries(record):
    """The translated entries one product dispatches, in order."""
    if record["family"] == "splitk":
        return [record["entry"], record["route"]["accum"]]
    return [record["entry"]]


# (family, entry prefix, launch check, coverage check)
FAMILIES = (
    ("gemv", gemv_layout.ENTRY_PREFIXES, _single(_gemv_launch_matches), _gemv_covered),
    ("gemm", gemm_layout.ENTRY_PREFIXES, _single(_gemm_launch_matches), _gemm_covered),
    (
        "splitk",
        gemm_splitk_layout.ENTRY_PREFIXES,
        _splitk_launch_matches,
        _splitk_covered,
    ),
)


def validate_trace(trace, records, target):
    if any(event.get("target") != target for event in trace):
        raise RuntimeError("Matmul trace does not match the routed entries")
    for family, prefixes, matches, covered in FAMILIES:
        chosen = [record for record in records if record["family"] == family]
        products = [
            event for event in trace if event.get("entry", "").startswith(prefixes)
        ]
        if [event["entry"] for event in products] != [
            entry for record in chosen for entry in _entries(record)
        ]:
            raise RuntimeError("Matmul trace does not match the routed entries")
        start = 0
        for record in chosen:
            size = KERNELS[family]
            if not matches(products[start : start + size], record):
                raise RuntimeError(f"Matmul launch differs: {record['name']}")
            start += size
        if not covered(chosen):
            raise RuntimeError("Matmul cases do not cover every route")


def verify(root, packages, output):
    import mlx.core as mx
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    preparation = verify_prepared(root)
    (output / "adaptation.json").write_text(json.dumps(preparation, indent=2))
    host = HostRuntime(packages, output / "trace.jsonl", mlx_root=root)
    host.install()
    records, refusals = [], []
    try:
        for seed, case in enumerate(CASES):
            record = run_case(mx, np, host, case_record(case), seed)
            records.append(record)
            print(
                json.dumps({key: record[key] for key in ("name", "entry")}), flush=True
            )
        for refusal in REFUSALS:
            refusals.append(run_refusal(mx, host, refusal))
    finally:
        (output / "results.json").write_text(
            json.dumps(
                {"target": host.target, "cases": records, "refusals": refusals},
                indent=2,
            )
        )
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    trace = [
        json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()
    ]
    validate_trace(trace, records, host.target)
    if verify_prepared(root) != preparation:
        raise RuntimeError("MLX sources changed during matmul verification")
    print(
        json.dumps(
            {
                "cases": len(records),
                "refusals": len(refusals),
                "dispatches": host.dispatch_count,
                "elements": sum(
                    record["route"]["batches"]
                    * (
                        record["route"]["outVecSize"]
                        if record["family"] == "gemv"
                        else record["route"]["M"] * record["route"]["N"]
                    )
                    for record in records
                ),
            }
        ),
        flush=True,
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
