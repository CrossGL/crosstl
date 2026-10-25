"""Exercise MLX AddMM through translated matrix kernels and their epilogues.

Every case routes one ``mx.addmm`` call as ``AddMM::eval_gpu`` does: to a
``gemv`` or ``gemv_t`` entry with or without ``axpby``, to a
``steel_gemm_fused`` entry that adds or scales ``C``, to a ``gemm_splitk``
entry and the plain or ``_axbpy`` accumulation, to a translated copy or
multiplication of ``C`` when ``K`` is 0, or to nothing for an empty output.
Inputs lie on dyadic grids and ``alpha`` and ``beta`` are powers of two, so
the product, each scaled term and their sum are exact in the output type:
every output bit is checked against the float64 reference, and float32
outputs against MLX's CPU as well.
"""

import argparse
import hashlib
import json
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    addmm_layout,
    gemm_layout,
    gemm_splitk_layout,
    gemv_layout,
)
from demos.integrations.mlx.portable_host import verify_matmul as matmul
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

# (name, dtype, a, b, c, alpha, beta). Operands are (stored shape, view) as in
# verify_matmul, plus "flip", the stored array with its rows reversed, and
# ("bias", shape), a lower-rank C that ops.cpp addmm broadcasts to ``shape``.
CASES = (
    # gemv_axbpy: both kernels, contiguous and strided batches, a bias
    # broadcast over rows, scaled and plain epilogues.
    (
        "vector-bias",
        "float32",
        ((1, 96), "dense"),
        ((96, 40), "dense"),
        ((40,), ("bias", (1, 40))),
        1.0,
        1.0,
    ),
    (
        "vector-axpby",
        "float16",
        ((1, 33), "dense"),
        ((33, 3), "dense"),
        ((1, 3), "dense"),
        0.5,
        2.0,
    ),
    (
        "vector-transposed-weights",
        "bfloat16",
        ((1, 64), "dense"),
        ((48, 64), "swap"),
        ((48,), ("bias", (1, 48))),
        1.0,
        1.0,
    ),
    (
        "vector-wide",
        "float32",
        ((1, 40), "dense"),
        ((40, 2050), "dense"),
        ((2050,), ("bias", (1, 2050))),
        2.0,
        -1.0,
    ),
    (
        "matrix-vector",
        "float32",
        ((40, 72), "dense"),
        ((72, 1), "dense"),
        ((40, 1), "dense"),
        2.0,
        -1.0,
    ),
    (
        "matrix-vector-flipped-c",
        "float16",
        ((24, 16), "dense"),
        ((16, 1), "dense"),
        ((24, 1), "flip"),
        1.0,
        0.5,
    ),
    (
        "vector-batched",
        "float32",
        ((3, 1, 128), "dense"),
        ((3, 128, 64), "dense"),
        ((3, 1, 64), "dense"),
        0.25,
        1.0,
    ),
    (
        "vector-two-batch-axes",
        "float16",
        ((2, 3, 1, 64), "dense"),
        ((1, 3, 64, 32), ("broadcast", (2, 3, 64, 32))),
        ((2, 3, 1, 32), "dense"),
        1.0,
        1.0,
    ),
    (
        "vector-plain",
        "bfloat16",
        ((1, 48), "dense"),
        ((48, 24), "dense"),
        ((1, 24), "dense"),
        1.0,
        0.0,
    ),
    # steel_gemm_fused: C added or scaled, aligned and unaligned tiles,
    # transposed and broadcast C, batches folded, collapsed and strided.
    (
        "gemm-bias",
        "float32",
        ((64, 48), "dense"),
        ((48, 64), "dense"),
        ((64,), ("bias", (64, 64))),
        1.0,
        1.0,
    ),
    (
        "gemm-axpby",
        "float16",
        ((70, 41), "dense"),
        ((41, 69), "dense"),
        ((70, 69), "dense"),
        0.5,
        2.0,
    ),
    (
        "gemm-transposed",
        "float32",
        ((35, 33), "swap"),
        ((17, 35), "swap"),
        ((17, 33), "swap"),
        2.0,
        0.5,
    ),
    (
        "gemm-batched-bias",
        "bfloat16",
        ((2, 3, 64, 16), "dense"),
        ((2, 3, 16, 64), "dense"),
        ((64,), ("bias", (2, 3, 64, 64))),
        1.0,
        1.0,
    ),
    (
        "gemm-has-batch",
        "float32",
        ((2, 3, 16, 48), "dense"),
        ((1, 3, 48, 40), ("broadcast", (2, 3, 48, 40))),
        ((2, 3, 16, 40), "dense"),
        1.0,
        1.0,
    ),
    (
        "gemm-folded",
        "float16",
        ((3, 16, 32), "dense"),
        ((1, 32, 48), ("broadcast", (3, 32, 48))),
        ((3, 16, 48), "dense"),
        1.0,
        0.0,
    ),
    (
        "gemm-copied",
        "float32",
        ((40, 66), "step"),
        ((33, 20), "dense"),
        ((40, 20), "flip"),
        -1.0,
        1.0,
    ),
    # Split-K: plain and _axbpy accumulations, transposed and broadcast C.
    (
        "splitk-bias",
        "float32",
        ((24, 416), "dense"),
        ((416, 24), "dense"),
        ((24,), ("bias", (24, 24))),
        1.0,
        1.0,
    ),
    (
        "splitk-axpby",
        "float16",
        ((16, 256), "dense"),
        ((256, 16), "dense"),
        ((16, 16), "dense"),
        0.5,
        2.0,
    ),
    (
        "splitk-transposed",
        "bfloat16",
        ((128, 32), "swap"),
        ((128, 32), "dense"),
        ((32, 32), "swap"),
        1.0,
        1.0,
    ),
    (
        "splitk-plain",
        "float32",
        ((32, 512), "dense"),
        ((512, 32), "dense"),
        ((32, 32), "dense"),
        1.0,
        0.0,
    ),
    # K == 0 copies C or multiplies it by beta; an empty output dispatches
    # nothing.
    (
        "empty-k-copy",
        "float32",
        ((5, 0), "dense"),
        ((0, 4), "dense"),
        ((5, 4), "dense"),
        0.5,
        1.0,
    ),
    (
        "empty-k-scale",
        "float16",
        ((2, 5, 0), "dense"),
        ((2, 0, 5), "dense"),
        ((2, 5, 5), "dense"),
        0.0,
        3.0,
    ),
    (
        "empty-output",
        "float32",
        ((0, 8), "dense"),
        ((8, 4), "dense"),
        ((0, 4), "dense"),
        1.0,
        1.0,
    ),
)
# Translated kernels each family dispatches per product.
KERNELS = {"gemv": 1, "gemm": 1, "splitk": 2, "scale": 1, "empty": 0}


def view_layout(stored, view):
    """The shape and strides AddMM receives for one operand."""
    if view == "flip":
        shape, strides = matmul.view_layout(stored, "dense")
        strides[-2] = -strides[-2]
        return shape, strides
    if view[0] == "bias":
        target = list(view[1])
        strides = [0] * (len(target) - len(stored)) + [
            0 if extent == 1 else stride
            for extent, stride in zip(stored, matmul._row_strides(stored))
        ]
        return target, strides
    return matmul.view_layout(stored, view)


def case_record(case):
    name, dtype, *operands, alpha, beta = case
    views = {}
    for key, (stored, view) in zip("abc", operands):
        shape, strides = view_layout(stored, view)
        views[key] = {
            "stored": list(stored),
            "view": view,
            "shape": shape,
            "strides": strides,
        }
    family, route = addmm_layout.route(
        dtype,
        *((views[key]["shape"], views[key]["strides"]) for key in "abc"),
        alpha,
        beta,
    )
    return {
        "name": name,
        "family": family,
        "dtype": dtype,
        **views,
        "alpha": alpha,
        "beta": beta,
        "entry": route.get("entry"),
        "route": route,
        "copies": sum(route["copies"].values()),
    }


def _view(np, values, view):
    if view == "flip":
        return values[..., ::-1, :]
    if view[0] == "bias":
        return np.broadcast_to(values, view[1])
    return matmul._view(np, values, view)


def _operand(mx, np, values, view, dtype):
    """The stored array and the view passed to mx.addmm.

    Arrays are built from float32 storage, which holds every grid value
    exactly, so empty operands keep their shapes.
    """
    array = mx.array(values.astype(np.float32), dtype=dtype)
    if view == "dense" or view[0] == "bias":
        return array, array
    if view == "flip":
        return array, array[..., ::-1, :]
    if view == "swap":
        return array, mx.swapaxes(array, -1, -2)
    if view == "step":
        return array, array[..., ::2]
    if view[0] == "strided":
        return array, mx.as_strided(array, view[1], view[2])
    return array, mx.broadcast_to(array, view[1])


def _exact(np, values, dtype):
    return np.array_equal(rounded(values, dtype).astype(np.float64), values)


def case_data(np, record, seed):
    """Stored operands on dyadic grids and the exact rounded result."""
    rng = np.random.default_rng(seed)
    wide = record["dtype"] == "float32"
    a = rng.integers(-4, 5, size=record["a"]["stored"]) / (4.0 if wide else 2.0)
    b = rng.integers(-4, 5, size=record["b"]["stored"]) / (8.0 if wide else 4.0)
    c = rng.integers(-4, 5, size=record["c"]["stored"]) / 2.0
    product = np.matmul(
        _view(np, a, record["a"]["view"]), _view(np, b, record["b"]["view"])
    )
    addend = _view(np, c, record["c"]["view"])
    alpha, beta = record["alpha"], record["beta"]
    total = alpha * product + beta * addend
    # The gemv epilogue rounds each term in the output type, so every term
    # must be exact there as well as the total.
    terms = (product, alpha * product, beta * addend, total)
    if not all(_exact(np, term, record["dtype"]) for term in terms):
        raise ValueError(f"AddMM case is not exact in its type: {record['name']}")
    return {"a": a, "b": b, "c": c, "expected": rounded(total, record["dtype"])}


def run_case(mx, np, host, record, seed):
    data = case_data(np, record, seed)
    dtype = getattr(mx, record["dtype"])
    stored, views = {}, {}
    for key in "abc":
        stored[key], views[key] = _operand(
            mx, np, data[key], record[key]["view"], dtype
        )
    mx.eval(*stored.values())
    before = {
        key: matmul.float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for key, value in stored.items()
    }
    arguments = (views["c"], views["a"], views["b"], record["alpha"], record["beta"])
    start = host.dispatch_count
    y = mx.addmm(*arguments)
    mx.eval(y)
    count = host.dispatch_count - start
    actual = np.asarray(y.tolist(), dtype=np.float32).reshape(data["expected"].shape)
    expected = data["expected"]
    if list(y.shape) != list(expected.shape) or y.dtype != dtype:
        raise AssertionError(f"AddMM output layout differs: {record['name']}")
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"AddMM output differs: {record['name']}")
    # One translated copy for each operand the host copies, then the kernels.
    if count != KERNELS[record["family"]] + record["copies"]:
        raise AssertionError(f"AddMM dispatch count differs: {record['name']}")
    for key, value in stored.items():
        if before[key] != matmul.float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"AddMM input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        reference = mx.addmm(*arguments, stream=mx.cpu)
        cpu = np.asarray(reference.tolist(), dtype=np.float32).reshape(expected.shape)
        if cpu.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(f"AddMM differs from MLX CPU: {record['name']}")
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu is not None,
        "inputUnchanged": True,
    }


def _gemv_matches(events, record):
    (event,) = events
    route, metadata = record["route"], event["gemvMetadata"]
    parameters = event["parameters"]
    matches = matmul._gemv_launch_matches(event, record)
    if "axpby" in route:
        return (
            matches
            and metadata.get("axpby") == route["axpby"]
            and parameters["bias_batch_stride"] == route["biasBatchStride"]
        )
    return matches and "axpby" not in metadata


def _gemm_matches(events, record):
    (event,) = events
    route = record["route"]
    return (
        matmul._gemm_launch_matches(event, record)
        and event["gemmMetadata"].get("addmm") == route.get("addmm")
        and event["parameters"].get("addmm_params") == route.get("addmm")
    )


def _splitk_matches(events, record):
    return matmul._splitk_launch_matches(events, record)


def _gemv_covered(records):
    signatures = [gemv_layout.signature(record["entry"]) for record in records]
    return (
        {info["kind"] for info in signatures} == {"gemv", "gemv_t"}
        and {info["nc"] for info in signatures} == {False, True}
        and {info["axpby"] for info in signatures} == {False, True}
        and {info["dtype"] for info in signatures} == set(gemv_layout.DTYPES)
        and any(record["route"]["M"] > 1 for record in records)
        and any(record["route"]["copies"]["c"] for record in records)
    )


def _gemm_covered(records):
    signatures = [gemm_layout.signature(record["entry"]) for record in records]
    constants = [info["constants"] for info in signatures]
    addmm = [record["route"].get("addmm") for record in records]
    return (
        {info["dtype"] for info in signatures} == set(gemm_layout.DTYPES)
        and all(values["use_out_source"] for values in constants)
        and all(
            {values[name] for values in constants} == {False, True}
            for name in ("do_axpby", "has_batch", "align_M", "align_N", "align_K")
        )
        and {(info["transposeA"], info["transposeB"]) for info in signatures}
        >= {(False, False), (True, True)}
        and any(params["ldc"] == 0 for params in addmm)
        and any(params["fdc"] > 1 for params in addmm)
        and any(record["route"]["copies"]["c"] for record in records)
        and any(record["route"]["copies"]["a"] for record in records)
        and any(record["route"]["M"] > record["a"]["shape"][-2] for record in records)
    )


def _splitk_covered(records):
    accums = [
        gemm_splitk_layout.signature(record["route"]["accum"]) for record in records
    ]
    params = [record["route"]["accumParams"] for record in records]
    return (
        {info["kind"] for info in accums} == {"accum", "axbpy"}
        and {info["dtype"] for info in accums} == set(gemm_splitk_layout.DTYPES)
        and any(item.get("ldc") == 0 for item in params)
        and any(item.get("fdc", 1) > 1 for item in params)
    )


def _scale_covered(records):
    return {record["route"]["copy"] for record in records} == {False, True}


def _scale_dispatches(trace, records):
    """Each K == 0 case copies C, or copies operands and multiplies by beta."""
    for record in records:
        if record["family"] != "scale":
            continue
        start = record["dispatchStart"]
        events = trace[start : start + record["dispatchCount"]]
        last = "copy" if record["route"]["copy"] else f"vv_Multiply{record['dtype']}"
        entries = [event.get("entry", "") for event in events]
        if (
            len(entries) != KERNELS["scale"] + record["copies"]
            or any("copy" not in entry for entry in entries[:-1])
            or (
                last not in entries[-1]
                if record["route"]["copy"]
                else entries[-1] != last
            )
        ):
            return False
    return True


def _empty_covered(records):
    return bool(records)


# (family, entry prefixes, launch check, coverage check)
FAMILIES = (
    ("gemv", gemv_layout.ENTRY_PREFIXES, _gemv_matches, _gemv_covered),
    ("gemm", gemm_layout.ENTRY_PREFIXES, _gemm_matches, _gemm_covered),
    ("splitk", gemm_splitk_layout.ENTRY_PREFIXES, _splitk_matches, _splitk_covered),
)


def _entries(record):
    if record["family"] == "splitk":
        return [record["entry"], record["route"]["accum"]]
    return [record["entry"]]


def validate_trace(trace, records, target):
    if any(event.get("target") != target for event in trace):
        raise RuntimeError("AddMM trace does not match the routed entries")
    for family, prefixes, matches, covered in FAMILIES:
        chosen = [record for record in records if record["family"] == family]
        products = [
            event for event in trace if event.get("entry", "").startswith(prefixes)
        ]
        if [event["entry"] for event in products] != [
            entry for record in chosen for entry in _entries(record)
        ]:
            raise RuntimeError("AddMM trace does not match the routed entries")
        start = 0
        for record in chosen:
            size = KERNELS[family]
            if not matches(products[start : start + size], record):
                raise RuntimeError(f"AddMM launch differs: {record['name']}")
            start += size
        if not covered(chosen):
            raise RuntimeError("AddMM cases do not cover every route")
    for family, covered in (("scale", _scale_covered), ("empty", _empty_covered)):
        if not covered([record for record in records if record["family"] == family]):
            raise RuntimeError("AddMM cases do not cover every route")
    if not _scale_dispatches(trace, records):
        raise RuntimeError("AddMM scaling of C differs from matmul.cpp's")


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
            print(
                json.dumps({key: record[key] for key in ("name", "family", "entry")}),
                flush=True,
            )
    finally:
        (output / "results.json").write_text(
            json.dumps({"target": host.target, "cases": records}, indent=2)
        )
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    trace = [
        json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()
    ]
    validate_trace(trace, records, host.target)
    if verify_prepared(root) != preparation:
        raise RuntimeError("MLX sources changed during AddMM verification")
    print(
        json.dumps({"cases": len(records), "dispatches": host.dispatch_count}),
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
