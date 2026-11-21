"""Exercise MLX gather_qmm through translated block-scaled gathered quantized kernels.

Every case routes one ``mx.gather_qmm`` call with mxfp4, mxfp8 or nvfp4
weights to one translated entry of quantized.cpp's GatherQMM routes
(``fp_gather_qmm_layout``): ``gather_qmv_fast`` and ``gather_qmv`` for
transposed products of few rows, ``gather_qvm`` for untransposed ones, and
``gather_qmm_t`` and ``gather_qmm_n`` from the vector limit, with and without
nvfp4's global scale per expert. The cases cover one and two index
dimensions, broadcast indices and zero-, one- and two-dimensional weight
batches, so adjust_matrix_offsets takes each of its paths and moves each
global scale to its gathered expert's.

Each output bit is checked against an exact float64 reference. The weights
take fp4 codes, or fp8 codes of moderate exponent, under power-of-two group
scales, and the global scales are 2688 times powers of two, so every
dequantized weight is exact in the element type; ``case_data`` proves every
float32 partial sum exact, whatever its order, so only the final cast
rounds. MLX's CPU dequantization must reproduce the reference weights, and
float32 products without a global scale, which MLX computes on the GPU only,
must match MLX's CPU product bit for bit.

The refusals check that a sorted single-row batch, which quantized.cpp sends
to the untranslated ``gather_qmm_rhs``, and an index past the experts raise
before any gathered product launches.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from demos.integrations.mlx.portable_host import fp_gather_qmm_layout
from demos.integrations.mlx.portable_host.fp_matmul_layout import MODES
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_gather_qmm import (
    DEFAULT_INDEX_ENTRY,
    _refuse,
    broadcast_shape,
    matrix_pairs,
    shape_of,
)
from demos.integrations.mlx.portable_host.verify_quantization import packed_reference
from demos.integrations.mlx.portable_host.verify_quantized_matmul import (
    float_bits,
    rounded,
)

# (name, mode, dtype, K, N, M, x batch shape, w batch shape, lhs indices,
# rhs indices, transpose, global scale exponents). Indices are nested tuples
# or a scalar that broadcasts against the other; each expert's global scale
# is 2688 * 2 ** exponent, shaped as the weights' batch, or None for none.
CASES = (
    # gather_qmv_fast over stacked activations and experts.
    (
        "mxfp4-qmv-fast-experts",
        "mxfp4",
        "float16",
        512,
        64,
        1,
        (2,),
        (3,),
        (1, 0, 1, 1),
        (2, 0, 1, 2),
        True,
        None,
    ),
    # gather_qmv with two index dimensions over a two-dimensional w batch,
    # each expert with its own global scale.
    (
        "nvfp4-qmv-global-broadcast",
        "nvfp4",
        "bfloat16",
        96,
        40,
        3,
        (2,),
        (2, 2),
        ((0,), (1,)),
        ((3, 1, 2),),
        True,
        ((-1, 0), (1, 2)),
    ),
    # gather_qmv with N off the 8-row tile, a two-dimensional x batch and
    # weights every product shares through a broadcast scalar index.
    (
        "mxfp8-qmv-shared-weights",
        "mxfp8",
        "float32",
        256,
        33,
        2,
        (2, 3),
        (),
        ((5, 0), (2, 4)),
        0,
        True,
        None,
    ),
    # gather_qmv_fast one row below the vector limit, with global scales.
    (
        "nvfp4-qmv-fast-global-limit",
        "nvfp4",
        "float32",
        512,
        32,
        17,
        (1,),
        (2,),
        (0, 0),
        (1, 0),
        True,
        (1, -1),
    ),
    # gather_qvm over experts, and over a broadcast lhs index with global
    # scales.
    (
        "mxfp4-qvm-experts",
        "mxfp4",
        "float32",
        64,
        128,
        3,
        (3,),
        (2,),
        (2, 0, 1),
        (1, 1, 0),
        False,
        None,
    ),
    (
        "nvfp4-qvm-global-broadcast",
        "nvfp4",
        "bfloat16",
        64,
        96,
        1,
        (1,),
        (3,),
        0,
        (2, 1),
        False,
        (0, 1, -1),
    ),
    # gather_qmm_t at the vector limit and over two row and column tiles.
    (
        "mxfp8-qmm-t-limit",
        "mxfp8",
        "float32",
        64,
        32,
        18,
        (1,),
        (2,),
        (0, 0),
        (1, 0),
        True,
        None,
    ),
    # gather_qmm_t with unaligned N, two index dimensions and global scales.
    (
        "nvfp4-qmm-t-global-unaligned",
        "nvfp4",
        "float16",
        64,
        40,
        33,
        (2, 2),
        (3,),
        ((0, 3), (2, 1)),
        ((2,), (0,)),
        True,
        (1, 0, -1),
    ),
    # gather_qmm_n with a K tail over experts, and with shared weights under
    # one global scale.
    (
        "mxfp4-qmm-n-experts",
        "mxfp4",
        "bfloat16",
        40,
        64,
        33,
        (2,),
        (2,),
        (1, 0),
        (1, 0),
        False,
        None,
    ),
    (
        "nvfp4-qmm-n-global-shared",
        "nvfp4",
        "float32",
        33,
        128,
        8,
        (3,),
        (),
        (2, 0),
        0,
        False,
        1,
    ),
)
KERNELS = {"qmv_fast", "qmv", "qvm", "qmm_t", "qmm_n"}
# fp_quantized.h's F8E4M3_MAX * F4E2M1_MAX: a global scale g multiplies each
# product by g / ENCODED_RANGE.
ENCODED_RANGE = 448 * 6
# fp4_e2m1's codes, by value; code 8 is negative zero.
FP4_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
# fp8_e4m3 codes of exponent fields 5 through 9: magnitudes in [0.25, 7.5]
# on a 2 ** -5 grid.
FP8_CODES = tuple(
    sign << 7 | exponent << 3 | mantissa
    for sign in (0, 1)
    for exponent in range(5, 10)
    for mantissa in range(8)
)
# The group scales' exponents: e8m0 codes 125 to 128, e4m3 codes 40 to 64.
SCALE_EXPONENTS = (-2, -1, 0, 1)
# float32 holds every multiple of a grid step below this many steps.
EXACT_STEPS = 2**24


def fp8_value(code):
    """fp8_e4m3's value of a normal code."""
    sign, exponent, mantissa = code >> 7, (code >> 3) & 15, code & 7
    return (-1.0) ** sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def scale_code(mode, exponent):
    """The uint8 scale of ``2 ** exponent``: e4m3 for nvfp4, e8m0 otherwise."""
    return 56 + 8 * exponent if mode == "nvfp4" else 127 + exponent


def case_record(case):
    """One case's routing: its entry, index shape and dispatches."""
    (
        name,
        mode,
        dtype,
        k,
        n,
        m,
        x_batch,
        w_batch,
        lhs,
        rhs,
        transpose,
        exponents,
    ) = case
    group, bits = MODES[mode]
    shape = broadcast_shape(shape_of(lhs), shape_of(rhs))
    if exponents is not None and shape_of(exponents) != tuple(w_batch):
        raise ValueError("Global scales must hold one value per expert")
    entry = fp_gather_qmm_layout.select(
        mode, dtype, m, n, k, transpose, exponents is not None
    )
    return {
        "name": name,
        "mode": mode,
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
        "globalScaleExponents": exponents,
        "batches": math.prod(shape),
        "entry": entry,
        "dispatches": [entry],
    }


def case_data(record, seed):
    """Codes, scales, activations, dequantized weights and the exact product."""
    import numpy as np

    rng = np.random.default_rng(seed)
    k, n, m = record["K"], record["N"], record["M"]
    mode, group, bits = record["mode"], record["groupSize"], record["bits"]
    w_batch, x_batch = tuple(record["wBatchShape"]), tuple(record["xBatchShape"])
    # Quantization groups run along w's last axis: K when transposed, N not.
    rows, columns = (n, k) if record["transpose"] else (k, n)
    if bits == 4:
        codes = rng.integers(0, 16, size=(*w_batch, rows, columns))
        values = np.array(FP4_VALUES)[codes]
        step = 0.5
    else:
        codes = rng.choice(FP8_CODES, size=(*w_batch, rows, columns))
        values = np.vectorize(fp8_value)(codes)
        step = 2.0**-5
    exponents = rng.choice(SCALE_EXPONENTS, size=(*w_batch, rows, columns // group))
    scales = np.vectorize(lambda e: scale_code(mode, int(e)))(exponents)
    weights = values * np.repeat(2.0**exponents, group, axis=-1)
    step *= 2.0 ** exponents.min()
    global_exponents = record["globalScaleExponents"]
    global_scale = None
    if global_exponents is not None:
        factors = 2.0 ** np.array(global_exponents, dtype=np.float64)
        global_scale = ENCODED_RANGE * factors
        weights = weights * factors.reshape(*w_batch, 1, 1)
        step *= factors.min()
    x = rng.integers(-4, 5, size=(*x_batch, m, k)) / 4.0
    step /= 4
    packed = np.stack(
        [
            np.frombuffer(packed_reference([int(c) for c in row], bits), dtype="<u4")
            for row in codes.reshape(-1, columns)
        ]
    ).reshape(*w_batch, rows, columns * bits // 32)
    if record["transpose"]:
        weights = np.swapaxes(weights, -1, -2)
    matrices = x.reshape(-1, m, k)
    experts = weights.reshape(-1, k, n)
    pairs = matrix_pairs(np, record)
    if any(i >= len(matrices) or j >= len(experts) for i, j in pairs):
        raise ValueError("Gathered quantized case indexes past its operands")
    # Every term is a multiple of ``step``, so every partial sum below
    # EXACT_STEPS steps is exact in float32, in any order.
    budget = max(
        float((np.abs(matrices[i]) @ np.abs(experts[j])).max()) for i, j in pairs
    )
    if budget >= EXACT_STEPS * step:
        raise ValueError("Gathered quantized case exceeds the exact float32 budget")
    products = np.stack([np.matmul(matrices[i], experts[j]) for i, j in pairs])
    expected = rounded(products.reshape(*record["indexShape"], m, n), record["dtype"])
    return {
        "w": packed,
        "scales": scales,
        "globalScale": global_scale,
        "x": x,
        "weights": weights,
        "expected": expected,
        "pairs": pairs,
    }


def _arrays(mx, record, data):
    arrays = {
        "x": mx.array(data["x"].tolist(), dtype=getattr(mx, record["dtype"])),
        "w": mx.array(data["w"].tolist(), dtype=mx.uint32),
        "scales": mx.array(data["scales"].tolist(), dtype=mx.uint8),
    }
    if data["globalScale"] is not None:
        arrays["globalScale"] = mx.array(data["globalScale"].tolist(), dtype=mx.float32)
    for name, key in (("lhs", "lhsIndices"), ("rhs", "rhsIndices")):
        arrays[name] = mx.array(record[key], dtype=mx.uint32)
    mx.eval(*arrays.values())
    return arrays


def _product(mx, record, arrays, **kwargs):
    options = {"global_scale": arrays["globalScale"]} if "globalScale" in arrays else {}
    return mx.gather_qmm(
        arrays["x"],
        arrays["w"],
        arrays["scales"],
        lhs_indices=kwargs.pop("lhs", arrays["lhs"]),
        rhs_indices=kwargs.pop("rhs", arrays["rhs"]),
        transpose=record["transpose"],
        group_size=record["groupSize"],
        bits=record["bits"],
        mode=record["mode"],
        **options,
        **kwargs,
    )


def _check_dequantized(mx, np, record, arrays, data):
    """MLX's CPU dequantization reproduces the reference weights."""
    options = {"global_scale": arrays["globalScale"]} if "globalScale" in arrays else {}
    weights = mx.dequantize(
        arrays["w"],
        arrays["scales"],
        group_size=record["groupSize"],
        bits=record["bits"],
        mode=record["mode"],
        dtype=mx.float32,
        stream=mx.cpu,
        **options,
    )
    actual = np.asarray(weights.tolist(), dtype=np.float64)
    expected = data["weights"]
    if record["transpose"]:
        expected = np.swapaxes(expected, -1, -2)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise AssertionError(f"gather_qmm reference weights differ: {record['name']}")


def run_case(mx, np, host, record, seed):
    data = case_data(record, seed)
    arrays = _arrays(mx, record, data)
    if "globalScale" not in arrays:
        # MLX's CPU dequantize takes no global scale.
        _check_dequantized(mx, np, record, arrays, data)
    before = {
        name: float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for name, value in arrays.items()
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
        if before[name] != float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"gather_qmm input changed: {record['name']}")
    cpu = record["dtype"] == "float32" and "globalScale" not in arrays
    if cpu:
        # MLX's CPU kernel accumulates in float32, which these inputs keep exact.
        reference = _product(mx, record, arrays, stream=mx.cpu)
        values = np.asarray(reference.tolist(), dtype=np.float32)
        if values.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(f"gather_qmm differs from MLX CPU: {record['name']}")
    return {
        **record,
        "seed": seed,
        "pairs": data["pairs"],
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu,
        "inputUnchanged": True,
    }


def run_refusals(mx, np, host, seed):
    record = case_record(CASES[1])
    data = case_data(record, seed)
    arrays = _arrays(mx, record, data)
    experts = mx.array(data["w"].reshape(-1, *data["w"].shape[-2:]).tolist(), mx.uint32)
    scales = mx.array(
        data["scales"].reshape(-1, *data["scales"].shape[-2:]).tolist(), mx.uint8
    )
    global_scale = mx.array(data["globalScale"].reshape(-1).tolist(), mx.float32)
    x = mx.array(np.zeros((16, 1, record["K"])).tolist(), dtype=mx.bfloat16)
    sorted_rhs = mx.array([0] * 4 + [1] * 4 + [2] * 8, dtype=mx.uint32)
    mx.eval(experts, scales, global_scale, x, sorted_rhs)
    common = {"transpose": True, "mode": "nvfp4", "global_scale": global_scale}
    return [
        # right_sorted with M == 1, B == 16 and B / E >= 4: gather_qmm_rhs.
        # quantized.cpp's caller first fills the omitted lhs indices.
        _refuse(
            mx,
            host,
            "sorted-rhs",
            "gather_qmm_rhs",
            lambda: mx.gather_qmm(
                x,
                experts,
                scales,
                rhs_indices=sorted_rhs,
                sorted_indices=True,
                **common,
            ),
            allowed=(DEFAULT_INDEX_ENTRY,),
        ),
        # An rhs index past the four experts.
        _refuse(
            mx,
            host,
            "rhs-out-of-range",
            "index names no matrix",
            lambda: _product(
                mx, record, arrays, rhs=mx.array(((3, 4, 2),), dtype=mx.uint32)
            ),
        ),
    ]


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
        if event["entry"].startswith(fp_gather_qmm_layout.ENTRY_PREFIXES)
    ]
    for event, record in zip(products, records):
        metadata = event["matmulMetadata"]
        if (
            metadata["M"],
            metadata["N"],
            metadata["K"],
            metadata["batches"],
            metadata["matrixIndices"],
            metadata["globalScale"],
        ) != (
            record["M"],
            record["N"],
            record["K"],
            record["batches"],
            record["pairs"],
            record["globalScaleExponents"] is not None,
        ):
            raise RuntimeError(f"gather_qmm launch differs: {record['name']}")
    signatures = [fp_gather_qmm_layout.signature(record["entry"]) for record in records]
    parameters = [event["parameters"] for event in products]
    if (
        {info["kernel"] for info in signatures} != KERNELS
        or {info["mode"] for info in signatures} != set(MODES)
        or {info["aligned"] for info in signatures} != {None, False, True}
        or {info["dtype"] for info in signatures} != {"float32", "float16", "bfloat16"}
        or {info["globalScale"] for info in signatures} != {False, True}
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
