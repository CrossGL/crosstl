"""Exercise MLX BlockMaskedMM through the translated masked gemv and steel kernels.

Every case routes one ``mx.block_masked_mm`` call as ``BlockMaskedMM::eval_gpu``
does: products with one output row or column to ``gemv_masked`` or
``gemv_t_masked``, every other one to the steel ``block_masked_gemm``, after
copying operands the kernels cannot read in place; to a translated fill of
zeros for an empty contraction; or to no launch for an empty output. One case
names a masked gemv tile that gemv_masked.metal does not instantiate, which
MLX cannot run either: the host refuses it before any launch.

Operands lie on dyadic grids and masks hold 0, 1, -1 or 2, so every product
is exact in float32 and rounds once to the output type. Every output bit is
checked against the float64 reference of block_masked_mm's definition, and
float32 outputs against MLX's CPU values as well.
"""

import argparse
import hashlib
import json
from pathlib import Path

from demos.integrations.mlx.portable_host import block_masked_mm_layout as layout
from demos.integrations.mlx.portable_host import verify_matmul as matmul
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

# (name, dtype, block size, a, b, (output mask, operand masks)). Operands are
# (stored shape, view) as in verify_matmul; masks are None, "bool" or
# "scale" (masks of the element type).
CASES = (
    # Partial tiles and a K tail, boolean masks everywhere.
    (
        "steel-bool-unaligned",
        "float32",
        32,
        ((40, 50), "dense"),
        ((50, 70), "dense"),
        ("bool", "bool"),
    ),
    # Whole tiles, multiplicative masks everywhere.
    (
        "steel-scale-aligned",
        "float16",
        64,
        ((64, 32), "dense"),
        ((32, 128), "dense"),
        ("scale", "scale"),
    ),
    # A transposed: a boolean output mask over scaled operands.
    (
        "steel-transposed-a",
        "bfloat16",
        32,
        ((48, 64), "swap"),
        ((48, 96), "dense"),
        ("bool", "scale"),
    ),
    # B transposed and broadcast over a batch of A.
    (
        "steel-transposed-b-batched",
        "float32",
        64,
        ((3, 70, 40), "dense"),
        ((1, 66, 40), "swap"),
        ("scale", "bool"),
    ),
    # Both transposed over a broadcast two-dimensional batch.
    (
        "steel-transposed-grid",
        "float16",
        32,
        ((2, 1, 40, 33), "swap"),
        ((1, 2, 50, 40), "swap"),
        (None, "bool"),
    ),
    # Scaled operands of a batch against one B, whole tiles.
    (
        "steel-operands-batched",
        "bfloat16",
        64,
        ((2, 64, 64), "dense"),
        ((64, 64), "dense"),
        (None, "scale"),
    ),
    # A copied from a strided view; a boolean output mask alone.
    (
        "steel-copied-a",
        "float32",
        32,
        ((40, 66), "step"),
        ((33, 40), "dense"),
        ("bool", None),
    ),
    # A scaled output mask alone over unaligned tiles.
    (
        "steel-output-scale",
        "float16",
        64,
        ((130, 20), "dense"),
        ((20, 70), "dense"),
        ("scale", None),
    ),
    # One output row: gemv_t over B.
    (
        "gemv-t-row",
        "float32",
        32,
        ((1, 100), "dense"),
        ((100, 70), "dense"),
        ("bool", "bool"),
    ),
    # One output column over a tall A: gemv with four simdgroup rows.
    (
        "gemv-column-tall",
        "float16",
        64,
        ((600, 80), "dense"),
        ((80, 1), "dense"),
        ("scale", "scale"),
    ),
    # One output row over a broadcast grid of wide B: gemv_t, four columns.
    (
        "gemv-t-grid",
        "bfloat16",
        64,
        ((2, 1, 1, 130), "dense"),
        ((1, 2, 130, 2100), "dense"),
        (None, "scale"),
    ),
    # One output column per batch with an output mask alone.
    (
        "gemv-column-batched",
        "float32",
        32,
        ((3, 50, 40), "dense"),
        ((3, 40, 1), "dense"),
        ("bool", None),
    ),
    # Three outputs: the single-row thread tile.
    (
        "gemv-short",
        "float16",
        32,
        ((3, 45), "dense"),
        ((45, 1), "dense"),
        (None, "bool"),
    ),
    # One row of three outputs at block size 64 needs a gemv_t tile that
    # gemv_masked.metal never instantiates.
    (
        "gemv-t-uninstantiated",
        "float32",
        64,
        ((1, 20), "dense"),
        ((20, 3), "dense"),
        ("bool", None),
    ),
    # An empty contraction fills the output with zeros.
    ("empty-k", "float32", 32, ((3, 0), "dense"), ((0, 4), "dense"), ("bool", "bool")),
    # No row: the output is empty and nothing launches.
    (
        "empty-output",
        "float16",
        32,
        ((0, 6), "dense"),
        ((6, 7), "dense"),
        ("scale", None),
    ),
)
# Translated dispatches per kind besides operand copies.
KERNELS = {"masked-mv": 1, "masked-mm": 1, "missing": 0, "empty": 1, "none": 0}
MASK_SCALES = (0, 1, -1, 2)


def _prod(values):
    total = 1
    for value in values:
        total *= value
    return total


def _broadcast_shape(*shapes):
    ndim = max(len(shape) for shape in shapes)
    padded = [[1] * (ndim - len(shape)) + list(shape) for shape in shapes]
    result = []
    for extents in zip(*padded):
        sizes = {extent for extent in extents if extent != 1}
        if len(sizes) > 1:
            raise ValueError("BlockMaskedMM batch shapes do not broadcast")
        result.append(sizes.pop() if sizes else 1)
    return result


def _broadcast(shape, strides, target):
    """broadcasting.cpp broadcast: every extent-one axis gets stride zero."""
    if list(shape) == list(target):
        return list(shape), list(strides)
    diff = len(target) - len(shape)
    result = [0] * len(target)
    for axis, (extent, stride) in enumerate(zip(shape, strides)):
        result[axis + diff] = 0 if extent == 1 else stride
    return list(target), result


def _token(mask, dtype):
    return {None: "nomask", "bool": "bool_"}.get(mask, dtype)


def case_record(case):
    name, dtype, block, a, b, (out_mask, op_mask) = case
    views = {}
    for key, (stored, view) in zip("ab", (a, b)):
        shape, strides = matmul.view_layout(stored, view)
        views[key] = {
            "stored": list(stored),
            "view": view,
            "shape": shape,
            "strides": strides,
        }
    m, k = views["a"]["shape"][-2:]
    n = views["b"]["shape"][-1]
    batch = _broadcast_shape(views["a"]["shape"][:-2], views["b"]["shape"][:-2])
    out_shape = [*batch, m, n]
    tiles = {"m": -(-m // block), "n": -(-n // block), "k": -(-k // block)}
    masks = {}
    if out_mask:
        masks["out"] = [*batch, tiles["m"], tiles["n"]]
    if op_mask:
        masks["lhs"] = [*views["a"]["shape"][:-2], tiles["m"], tiles["k"]]
        masks["rhs"] = [*views["b"]["shape"][:-2], tiles["k"], tiles["n"]]
    if _prod(out_shape) == 0:
        route = {"kind": "none", "entries": [], "copies": {}}
    elif k == 0:
        route = {"kind": "empty", "entries": [], "copies": {}}
    else:
        operands = [
            _broadcast(views[key]["shape"], views[key]["strides"], [*batch, *dims])
            for key, dims in (("a", (m, k)), ("b", (k, n)))
        ]
        route = layout.route(
            dtype,
            *operands,
            block,
            _token(out_mask, dtype),
            _token(op_mask, dtype),
        )
        if not _instantiated(route):
            # MLX would look up a kernel its library never built.
            route = {
                **route,
                "kind": "missing",
                "missing": route["entries"],
                "entries": [],
            }
    return {
        "name": name,
        "kind": route["kind"],
        "dtype": dtype,
        "blockSize": block,
        **views,
        "masks": {"out": out_mask, "op": op_mask, "shapes": masks},
        "outShape": out_shape,
        "entries": route["entries"],
        "route": route,
        "copies": sum(route["copies"].values()) if route["kind"] != "missing" else 0,
    }


def _instantiated(route):
    try:
        layout.signature(route["entries"][0])
    except ValueError:
        return False
    return True


def _expand(np, mask, block, rows, columns):
    """A block mask repeated over its blocks and cut to the operand's extent."""
    expanded = np.repeat(np.repeat(mask, block, axis=-2), block, axis=-1)
    return expanded[..., :rows, :columns]


def _mask_values(np, rng, shape, kind):
    """Block values with at least one zero block when there is any block."""
    if kind == "bool":
        values = rng.integers(0, 4, size=shape) > 0
    else:
        values = rng.choice(MASK_SCALES, size=shape).astype(np.float64)
    if values.size:
        values.reshape(-1)[0] = 0
    return values


def _nonzero(rng, size):
    """Nonzero integers in [-4, 4]: no product is a signed zero."""
    return rng.integers(1, 5, size=size) * rng.choice((-1, 1), size=size)


def reference(np, a, b, block, masks, kinds):
    """block_masked_mm: operand masks scale or drop K blocks, the output mask
    zeroes or scales tiles. Accumulators start at +0 and a zero output mask
    writes +0, as the kernels do."""
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    if "lhs" in masks:
        a = a * _expand(np, masks["lhs"], block, m, k)
        b = b * _expand(np, masks["rhs"], block, k, n)
    total = (a @ b) + 0.0
    if "out" in masks:
        out = _expand(np, masks["out"], block, m, n).astype(np.float64)
        total = np.where(
            out == 0, 0.0, total * (out if kinds["out"] == "scale" else 1.0)
        )
    return total


def case_data(np, record, seed):
    rng = np.random.default_rng(seed)
    a = _nonzero(rng, record["a"]["stored"]) / 4.0
    b = _nonzero(rng, record["b"]["stored"]) / 8.0
    kinds = {"out": record["masks"]["out"], "op": record["masks"]["op"]}
    masks = {}
    for which, shape in record["masks"]["shapes"].items():
        masks[which] = _mask_values(
            np, rng, shape, kinds["out" if which == "out" else "op"]
        )
    total = reference(
        np,
        matmul._view(np, a, record["a"]["view"]),
        matmul._view(np, b, record["b"]["view"]),
        record["blockSize"],
        masks,
        kinds,
    )
    total = np.broadcast_to(total, record["outShape"])
    if total.size and np.abs(total).max() >= 2**19:
        raise ValueError("BlockMaskedMM case exceeds the exact float32 budget")
    return {"a": a, "b": b, "masks": masks, "expected": rounded(total, record["dtype"])}


def _operand(mx, np, values, view, dtype):
    array = mx.array(values.astype(np.float32), dtype=dtype)
    if view == "dense":
        return array, array
    if view == "swap":
        return array, mx.swapaxes(array, -1, -2)
    if view == "step":
        return array, array[..., ::2]
    raise ValueError("Unsupported BlockMaskedMM operand view")


def _masks(mx, record, data, dtype):
    arrays = {}
    for which, values in data["masks"].items():
        kind = record["masks"]["out" if which == "out" else "op"]
        arrays[which] = (
            mx.array(values)
            if kind == "bool"
            else mx.array(values.astype("float32"), dtype=dtype)
        )
    return arrays


def run_case(mx, np, host, record, seed):
    data = case_data(np, record, seed)
    dtype = getattr(mx, record["dtype"])
    stored, views = {}, {}
    for key in "ab":
        stored[key], views[key] = _operand(
            mx, np, data[key], record[key]["view"], dtype
        )
    masks = _masks(mx, record, data, dtype)
    mx.eval(*stored.values(), *masks.values())
    before = {
        key: matmul.float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for key, value in {**stored, **masks}.items()
    }
    arguments = (views["a"], views["b"], record["blockSize"])
    keywords = {
        "mask_out": masks.get("out"),
        "mask_lhs": masks.get("lhs"),
        "mask_rhs": masks.get("rhs"),
    }
    start = host.dispatch_count
    if record["kind"] == "missing":
        try:
            mx.eval(mx.block_masked_mm(*arguments, **keywords))
        except ValueError as error:
            if f"No translated package for {record['route']['missing'][0]}" not in str(
                error
            ):
                raise
        else:
            raise AssertionError(
                f"BlockMaskedMM ran an uninstantiated kernel: {record['name']}"
            )
        if host.dispatch_count != start:
            raise AssertionError(
                f"BlockMaskedMM launched before refusing: {record['name']}"
            )
        return {
            **record,
            "seed": seed,
            "dispatchStart": start,
            "dispatchCount": 0,
            "refused": True,
        }
    y = mx.block_masked_mm(*arguments, **keywords)
    mx.eval(y)
    count = host.dispatch_count - start
    expected = data["expected"]
    actual = np.asarray(y.tolist(), dtype=np.float32).reshape(expected.shape)
    if list(y.shape) != list(expected.shape) or y.dtype != dtype:
        raise AssertionError(f"BlockMaskedMM output layout differs: {record['name']}")
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"BlockMaskedMM output differs: {record['name']}")
    if count != KERNELS[record["kind"]] + record["copies"]:
        raise AssertionError(f"BlockMaskedMM dispatch count differs: {record['name']}")
    for key, value in {**stored, **masks}.items():
        if before[key] != matmul.float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"BlockMaskedMM input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        reference_value = mx.block_masked_mm(*arguments, **keywords, stream=mx.cpu)
        cpu = np.asarray(reference_value.tolist(), dtype=np.float32).reshape(
            expected.shape
        )
        # MLX's CPU masks by multiplying, so only the sign of zeros may differ.
        if not np.array_equal(cpu, actual):
            raise AssertionError(
                f"BlockMaskedMM differs from MLX CPU: {record['name']}"
            )
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuEqual": cpu is not None,
        "inputUnchanged": True,
    }


def _launch_matches(event, record):
    """The traced launch has the record's shape, block size and masks."""
    metadata, route = event["maskedMetadata"], record["route"]
    kinds = {None: "nomask", "bool": "bool", "scale": "scale"}
    same = (
        metadata["kind"] == route["kind"]
        and metadata["blockSize"] == record["blockSize"]
        and metadata["outMask"] == kinds[record["masks"]["out"]]
        and metadata["opMask"] == kinds[record["masks"]["op"]]
        and metadata["batches"] == _prod(record["outShape"][:-2])
    )
    if route["kind"] == "masked-mm":
        return same and all(
            metadata[key] == route[key] for key in ("M", "N", "K", "lda", "ldb")
        )
    out_size = route["N"] if route["N"] != 1 else route["M"]
    return same and metadata["K"] == route["K"] and metadata["outVecSize"] == out_size


def _covered(records):
    infos = [
        layout.signature(entry) for record in records for entry in record["entries"]
    ]
    steel = [info for info in infos if info["kind"] == "masked-mm"]
    gemv = [info for info in infos if info["kind"] == "masked-mv"]
    return (
        {info["dtype"] for info in infos} == set(layout.DTYPES)
        and {info["block"] for info in steel} == set(layout.BLOCK_SIZES)
        and {(info["transposeA"], info["transposeB"]) for info in steel}
        == {(False, False), (False, True), (True, False), (True, True)}
        and {info["mnAligned"] for info in steel} == {False, True}
        and {info["kAligned"] for info in steel} == {False, True}
        and len({(info["outMask"], info["opMask"]) for info in steel})
        == len(layout.MASK_PAIRS)
        and {(info["kernel"], info["nc"]) for info in gemv}
        >= {("gemv", False), ("gemv_t", False), ("gemv_t", True)}
        and {
            block
            for record in records
            if record["kind"] == "masked-mv"
            for block in [record["blockSize"]]
        }
        == set(layout.BLOCK_SIZES)
        and {record["kind"] for record in records} == set(KERNELS)
        and any(record["route"]["copies"].get("a") for record in records)
        and all("outputHash" in record or record.get("refused") for record in records)
    )


def validate_trace(trace, records, target):
    if any(event.get("target") != target for event in trace):
        raise RuntimeError("BlockMaskedMM trace does not match the routed entries")
    for record in records:
        start, count = record["dispatchStart"], record["dispatchCount"]
        events = trace[start : start + count]
        products = [
            event
            for event in events
            if event.get("entry", "").startswith(layout.ENTRY_PREFIXES)
        ]
        if [event["entry"] for event in products] != record["entries"]:
            raise RuntimeError(
                f"BlockMaskedMM trace differs from its route: {record['name']}"
            )
        others = [event for event in events if event not in products]
        copies = [event for event in others if "copy" in event.get("entry", "")]
        expected = record["copies"] + (record["kind"] == "empty")
        if len(copies) != expected or len(others) != len(copies):
            raise RuntimeError(f"BlockMaskedMM copies differ: {record['name']}")
        if products and not _launch_matches(products[0], record):
            raise RuntimeError(f"BlockMaskedMM launch differs: {record['name']}")
    if not _covered(records):
        raise RuntimeError("BlockMaskedMM cases do not cover every route")


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
                json.dumps({key: record[key] for key in ("name", "kind", "entries")}),
                flush=True,
            )
    finally:
        (output / "results.json").write_text(
            json.dumps({"target": host.target, "cases": records}, indent=2)
        )
        close = getattr(host.executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
    trace_path = output / "trace.jsonl"
    trace = (
        [json.loads(line) for line in trace_path.read_text().splitlines()]
        if trace_path.exists()
        else []
    )
    validate_trace(trace, records, host.target)
    if verify_prepared(root) != preparation:
        raise RuntimeError("MLX sources changed during BlockMaskedMM verification")
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
