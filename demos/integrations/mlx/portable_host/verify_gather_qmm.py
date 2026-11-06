"""Exercise MLX gather_qmm through translated affine gathered quantized kernels.

Every case routes one ``mx.gather_qmm`` call to one translated entry of
quantized.cpp's GatherQMM routes: ``affine_gather_qmv_fast`` and
``affine_gather_qmv`` for transposed products of few rows,
``affine_gather_qvm`` for untransposed ones, and ``affine_gather_qmm_t`` and
``affine_gather_qmm_n`` from the vector limit. The cases cover one and two
index dimensions, broadcast indices and two-dimensional, one-dimensional and
stacked activation and weight batches, so adjust_matrix_offsets takes each
of its paths. Each output bit is checked against an exact float64
reference: inputs lie on dyadic grids that keep every dequantized weight
exact in the element type and every float32 partial sum exact, whatever its
order, so only the final cast rounds. float32 products must also match
MLX's CPU bit for bit.

The refusals check that sorted single-row batches, which quantized.cpp sends
to the untranslated ``gather_qmm_rhs``, and indices that name no matrix of
their operand raise before any gathered product launches. The block-scaled
modes are ``verify_fp_gather_qmm``'s.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from demos.integrations.mlx.portable_host import gather_qmm_layout
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantization import packed_reference
from demos.integrations.mlx.portable_host.verify_quantized_matmul import (
    SCALES,
    float_bits,
    rounded,
)

# (name, dtype, group, bits, K, N, M, x batch shape, w batch shape,
# lhs indices, rhs indices, transpose). Indices are nested tuples or a scalar
# that broadcasts against the other.
CASES = (
    # gather_qmv_fast over stacked activations and experts.
    (
        "qmv-fast-experts",
        "float16",
        64,
        4,
        512,
        64,
        1,
        (2,),
        (3,),
        (1, 0, 1, 1),
        (2, 0, 1, 2),
        True,
    ),
    # gather_qmv with two index dimensions over a two-dimensional w batch.
    (
        "qmv-broadcast",
        "bfloat16",
        32,
        3,
        96,
        40,
        3,
        (2,),
        (2, 2),
        ((0,), (1,)),
        ((3, 1, 2),),
        True,
    ),
    # gather_qmv with N off the 8-row tile, a two-dimensional x batch and
    # weights every product shares through a broadcast scalar index.
    (
        "qmv-shared-weights",
        "float32",
        128,
        6,
        256,
        33,
        2,
        (2, 3),
        (),
        ((5, 0), (2, 4)),
        0,
        True,
    ),
    # gather_qmv_fast one row below the vector limit.
    (
        "qmv-fast-limit",
        "float32",
        32,
        8,
        256,
        32,
        17,
        (1,),
        (2,),
        (0, 0),
        (1, 0),
        True,
    ),
    # gather_qvm over experts, and over a broadcast lhs index.
    (
        "qvm-experts",
        "float32",
        64,
        8,
        64,
        128,
        3,
        (3,),
        (2,),
        (2, 0, 1),
        (1, 1, 0),
        False,
    ),
    (
        "qvm-broadcast",
        "bfloat16",
        32,
        2,
        64,
        96,
        1,
        (1,),
        (3,),
        0,
        (2, 1),
        False,
    ),
    # gather_qmm_t at the vector limit and over two row and column tiles.
    (
        "qmm-t-limit",
        "float32",
        32,
        8,
        64,
        32,
        18,
        (1,),
        (2,),
        (0, 0),
        (1, 0),
        True,
    ),
    (
        "qmm-t-experts",
        "float16",
        64,
        2,
        128,
        64,
        40,
        (2,),
        (3,),
        (1, 0, 1),
        (2, 0, 1),
        True,
    ),
    # gather_qmm_t with unaligned N and two index dimensions.
    (
        "qmm-t-unaligned-broadcast",
        "bfloat16",
        32,
        5,
        64,
        40,
        33,
        (2, 2),
        (3,),
        ((0, 3), (2, 1)),
        ((2,), (0,)),
        True,
    ),
    # gather_qmm_n with a K tail over experts, and with shared weights.
    (
        "qmm-n-experts",
        "float32",
        32,
        6,
        40,
        64,
        33,
        (2,),
        (2,),
        (1, 0),
        (1, 0),
        False,
    ),
    (
        "qmm-n-shared",
        "float16",
        128,
        4,
        33,
        128,
        8,
        (3,),
        (),
        (2, 0),
        0,
        False,
    ),
)
KERNELS = {"qmv_fast", "qmv", "qvm", "qmm_t", "qmm_n"}
# The arange quantized.cpp's caller dispatches for an omitted lhs index.
DEFAULT_INDEX_ENTRY = "arangeuint32"


def shape_of(value):
    """The shape of a nested tuple of indices; a scalar has shape ()."""
    shape = []
    while isinstance(value, (tuple, list)):
        shape.append(len(value))
        value = value[0]
    return tuple(shape)


def broadcast_shape(*shapes):
    """NumPy's broadcast of ``shapes``."""
    rank = max(len(shape) for shape in shapes)
    result = []
    for axis in range(rank):
        extents = {
            shape[axis - rank + len(shape)]
            for shape in shapes
            if axis - rank + len(shape) >= 0
        } - {1}
        if len(extents) > 1:
            raise ValueError("Gathered quantized indices do not broadcast")
        result.append(extents.pop() if extents else 1)
    return tuple(result)


def case_record(case):
    (
        name,
        dtype,
        group,
        bits,
        k,
        n,
        m,
        x_batch,
        w_batch,
        lhs,
        rhs,
        transpose,
    ) = case
    shape = broadcast_shape(shape_of(lhs), shape_of(rhs))
    entry = gather_qmm_layout.select(dtype, group, bits, m, n, k, transpose)
    return {
        "name": name,
        "dtype": dtype,
        "groupSize": group,
        "bits": bits,
        "K": k,
        "N": n,
        "M": m,
        "xBatchShape": list(x_batch),
        "wBatchShape": list(w_batch),
        "lhsIndices": lhs,
        "rhsIndices": rhs,
        "indexShape": list(shape),
        "transpose": transpose,
        "batches": math.prod(shape),
        "entry": entry,
        "dispatches": [entry],
    }


def matrix_pairs(np, record):
    """Each output batch's ``(x, w)`` matrices, in row-major index order."""
    lhs, rhs = np.broadcast_arrays(
        np.array(record["lhsIndices"]), np.array(record["rhsIndices"])
    )
    return [[int(i), int(j)] for i, j in zip(lhs.reshape(-1), rhs.reshape(-1))]


def case_data(record, seed):
    """Codes, scales, biases and x on dyadic grids, and the exact outputs."""
    import numpy as np

    rng = np.random.default_rng(seed)
    k, n, m = record["K"], record["N"], record["M"]
    group, bits = record["groupSize"], record["bits"]
    bins = (1 << bits) - 1
    w_batch, x_batch = tuple(record["wBatchShape"]), tuple(record["xBatchShape"])
    # Quantization groups run along w's last axis: K when transposed, N not.
    rows, columns = (n, k) if record["transpose"] else (k, n)
    codes = rng.integers(0, bins + 1, size=(*w_batch, rows, columns))
    magnitudes = rng.choice(SCALES, size=(*w_batch, rows, columns // group))
    scales = magnitudes * rng.choice((-1.0, 1.0), size=magnitudes.shape)
    biases = -rng.integers(0, bins + 1, size=scales.shape) * scales
    x = rng.integers(-4, 5, size=(*x_batch, m, k)) / 4.0
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
    if np.abs(x).max() * k * (bins + bins) >= 2**19:
        raise ValueError("Gathered quantized case exceeds the exact float32 budget")
    matrices = x.reshape(-1, m, k)
    experts = weights.reshape(-1, k, n)
    pairs = matrix_pairs(np, record)
    if any(i >= len(matrices) or j >= len(experts) for i, j in pairs):
        raise ValueError("Gathered quantized case indexes past its operands")
    products = np.stack([np.matmul(matrices[i], experts[j]) for i, j in pairs])
    expected = rounded(products.reshape(*record["indexShape"], m, n), record["dtype"])
    return {
        "w": packed.astype(np.uint32),
        "scales": scales,
        "biases": biases,
        "x": x,
        "expected": expected,
        "pairs": pairs,
    }


def _arrays(mx, record, data):
    dtype = getattr(mx, record["dtype"])
    arrays = {
        name: mx.array(data[name].tolist(), dtype=dtype)
        for name in ("scales", "biases", "x")
    }
    arrays["w"] = mx.array(data["w"].tolist(), dtype=mx.uint32)
    for name, key in (("lhs", "lhsIndices"), ("rhs", "rhsIndices")):
        arrays[name] = mx.array(record[key], dtype=mx.uint32)
    mx.eval(*arrays.values())
    return arrays


def _product(mx, record, arrays, **kwargs):
    return mx.gather_qmm(
        arrays["x"],
        arrays["w"],
        arrays["scales"],
        arrays["biases"],
        lhs_indices=kwargs.pop("lhs", arrays["lhs"]),
        rhs_indices=kwargs.pop("rhs", arrays["rhs"]),
        transpose=record["transpose"],
        group_size=record["groupSize"],
        bits=record["bits"],
        **kwargs,
    )


def run_case(mx, np, host, record, seed):
    data = case_data(record, seed)
    arrays = _arrays(mx, record, data)
    before = {
        name: float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for name, value in arrays.items()
        if name != "w"
    }
    start = host.dispatch_count
    y = _product(mx, record, arrays)
    mx.eval(y)
    count = host.dispatch_count - start
    actual = np.asarray(y.tolist(), dtype=np.float32)
    expected = data["expected"]
    if actual.shape != expected.shape or y.dtype != getattr(mx, record["dtype"]):
        raise AssertionError(f"gather_qmm output layout differs: {record['name']}")
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"gather_qmm output differs: {record['name']}")
    if count != len(record["dispatches"]):
        raise AssertionError(f"gather_qmm dispatch count differs: {record['name']}")
    for name, value in arrays.items():
        if name != "w" and before[name] != float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"gather_qmm input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        # MLX's CPU kernel accumulates in float32, which these inputs keep exact.
        reference = _product(mx, record, arrays, stream=mx.cpu)
        cpu = np.asarray(reference.tolist(), dtype=np.float32)
        if cpu.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(f"gather_qmm differs from MLX CPU: {record['name']}")
    return {
        **record,
        "seed": seed,
        "pairs": data["pairs"],
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu is not None,
        "inputUnchanged": True,
    }


def _refuse(mx, host, name, message, product, allowed=()):
    """Require ``product()`` to raise ``message`` before a gathered launch."""
    count = host.dispatch_count
    try:
        mx.eval(product())
    except (RuntimeError, ValueError) as error:
        if message not in str(error):
            raise AssertionError(f"gather_qmm refusal differs: {name}: {error}")
    else:
        raise AssertionError(f"gather_qmm refusal did not raise: {name}")
    if host.dispatch_count - count != len(allowed):
        raise AssertionError(f"gather_qmm refusal dispatched: {name}")
    return {"name": name, "message": message, "dispatches": list(allowed)}


def run_refusals(mx, np, host, seed):
    """Sorted single-row batches, block-scaled weights and stray indices."""
    record = case_record(CASES[0])
    data = case_data(record, seed)
    arrays = _arrays(mx, record, data)
    x = mx.array(np.zeros((16, 1, record["K"])).tolist(), dtype=mx.float16)
    sorted_rhs = mx.array([0] * 4 + [1] * 4 + [2] * 8, dtype=mx.uint32)
    mx.eval(x, sorted_rhs)
    refusals = [
        # right_sorted with M == 1, B == 16 and B / E >= 4: gather_qmm_rhs.
        # quantized.cpp's caller first fills the omitted lhs indices.
        _refuse(
            mx,
            host,
            "sorted-rhs",
            "gather_qmm_rhs",
            lambda: mx.gather_qmm(
                x,
                arrays["w"],
                arrays["scales"],
                arrays["biases"],
                rhs_indices=sorted_rhs,
                transpose=True,
                group_size=record["groupSize"],
                bits=record["bits"],
                sorted_indices=True,
            ),
            allowed=(DEFAULT_INDEX_ENTRY,),
        ),
        # An rhs index past the three experts.
        _refuse(
            mx,
            host,
            "rhs-out-of-range",
            "index names no matrix",
            lambda: _product(
                mx,
                record,
                arrays,
                rhs=mx.array([2, 0, 3, 1], dtype=mx.uint32),
            ),
        ),
    ]
    return refusals


def validate_trace(trace, records, refusals, target):
    expected = [entry for record in records for entry in record["dispatches"]]
    expected += [entry for refusal in refusals for entry in refusal["dispatches"]]
    if [event.get("entry") for event in trace] != expected or any(
        event.get("target") != target for event in trace
    ):
        raise RuntimeError("gather_qmm trace does not match the routed entries")
    products = [
        event
        for event in trace
        if event["entry"].startswith(gather_qmm_layout.ENTRY_PREFIXES)
    ]
    for event, record in zip(products, records):
        metadata = event["matmulMetadata"]
        if (
            metadata["M"],
            metadata["N"],
            metadata["K"],
            metadata["batches"],
            metadata["matrixIndices"],
        ) != (
            record["M"],
            record["N"],
            record["K"],
            record["batches"],
            record["pairs"],
        ):
            raise RuntimeError(f"gather_qmm launch differs: {record['name']}")
    signatures = [gather_qmm_layout.signature(record["entry"]) for record in records]
    parameters = [event["parameters"] for event in products]
    if (
        {info["kernel"] for info in signatures} != KERNELS
        or {info["aligned"] for info in signatures} != {None, False, True}
        or {info["dtype"] for info in signatures} != {"float32", "float16", "bfloat16"}
        or {value["batch_ndims"] for value in parameters} != {1, 2}
        or {value["x_batch_ndims"] for value in parameters} != {1, 2}
        or {value["w_batch_ndims"] for value in parameters} != {0, 1, 2}
    ):
        raise RuntimeError("gather_qmm cases do not cover every route")


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
            (output / "results.json").write_text(
                json.dumps({"target": host.target, "cases": records}, indent=2)
            )
            print(json.dumps(record), flush=True)
        refusals = run_refusals(mx, np, host, len(CASES))
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
    validate_trace(trace, records, refusals, host.target)
    if verify_prepared(root) != preparation:
        raise RuntimeError("MLX sources changed during gather_qmm verification")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.mlx_root, args.packages, args.output_dir)
