"""Exercise MLX GatherMM through translated gathered matrix kernels.

Every case routes one ``mx.gather_mm`` call as ``GatherMM::eval_gpu`` does: to
``steel_gather_mm`` with and without an index batch, to ``gemv_gather`` or
``gemv_t_gather`` for one output row or column, to ``gather_mm_offsets`` and
``steel_gather_mm_rhs`` for sorted right indices, or to a translated fill of
zeros for an empty contraction. Inputs lie on dyadic grids, so every product
is exact in float32 and rounds once to the output type: every output bit is
checked against the float64 reference, and float32 outputs against MLX's CPU
as well. The sorted route's software-subgroup tile scheduler does not yet
translate to DirectX or OpenGL (CrossGL/crosstl#2546); on those targets its
cases must fail before any kernel runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

from demos.integrations.mlx.portable_host import gather_mm_layout, gather_mv_layout
from demos.integrations.mlx.portable_host import verify_matmul as matmul
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

# Targets whose translation of the sorted route fails closed, with its issue.
RHS_UNSUPPORTED = {"directx": "CrossGL/crosstl#2546", "opengl": "CrossGL/crosstl#2546"}
# (name, dtype, a, b, lhs, rhs, sorted). Operands are (stored shape, view) as in
# verify_matmul; indices are (shape, values) or None for MLX's default.
CASES = (
    # gather_mm: a one-dimensional index batch reads each index by its stride.
    (
        "matrix-vector-batch",
        "float32",
        ((4, 33, 17), "dense"),
        ((3, 17, 35), "dense"),
        ((3,), (3, 0, 2)),
        ((3,), (2, 2, 0)),
        False,
    ),
    # has_batch: a two-dimensional index batch, broadcast right indices, a
    # transposed B and whole tiles.
    (
        "matrix-grid-aligned",
        "float16",
        ((2, 2, 64, 32), "dense"),
        ((3, 64, 32), "swap"),
        ((2, 2), (3, 0, 1, 2)),
        ((1, 2), (2, 1)),
        False,
    ),
    # Scalar indices into a transposed A.
    (
        "matrix-scalar-transposed",
        "bfloat16",
        ((4, 40, 70), "swap"),
        ((2, 40, 9), "dense"),
        ((), (3,)),
        ((), (1,)),
        False,
    ),
    # A copied from a strided view; MLX's default left indices.
    (
        "matrix-copied-default",
        "float32",
        ((3, 20, 36), "step"),
        ((5, 18, 66), "dense"),
        None,
        ((3,), (4, 0, 4)),
        False,
    ),
    # gather_mv: one output row, so the gathered matrix is B; gemv_t_gather.
    (
        "row-transposed-gemv",
        "float32",
        ((5, 1, 96), "dense"),
        ((3, 96, 40), "dense"),
        ((4,), (4, 1, 0, 4)),
        ((4,), (2, 0, 1, 2)),
        False,
    ),
    # One output row of x @ W.T: gemv_gather over transposed weights.
    (
        "row-transposed-weights",
        "float16",
        ((6, 1, 64), "dense"),
        ((4, 48, 64), "swap"),
        ((3,), (5, 0, 2)),
        ((3,), (3, 3, 1)),
        False,
    ),
    # One output column: gemv_gather over A with a broadcast index grid.
    (
        "column-grid",
        "bfloat16",
        ((3, 50, 64), "dense"),
        ((2, 64, 1), "dense"),
        ((2, 2), (2, 0, 1, 2)),
        ((2, 1), (1, 0)),
        False,
    ),
    # gather_mm_rhs: rows sorted by expert, a group with no rows.
    (
        "sorted-rows",
        "float32",
        ((8, 1, 32), "dense"),
        ((4, 32, 64), "dense"),
        None,
        ((8,), (0, 0, 1, 1, 1, 3, 3, 3)),
        True,
    ),
    # gather_mm_rhs over a broadcast left operand and transposed experts.
    (
        "sorted-broadcast-transposed",
        "float16",
        ((2, 1, 1, 48), "dense"),
        ((3, 70, 48), "swap"),
        None,
        ((2, 5), (0, 0, 1, 1, 1, 1, 2, 2, 2, 2)),
        True,
    ),
    # An empty contraction fills the output with zeros.
    (
        "empty-k",
        "float32",
        ((2, 3, 0), "dense"),
        ((2, 0, 4), "dense"),
        ((2,), (1, 0)),
        ((2,), (0, 1)),
        False,
    ),
)
# Translated dispatches per kind besides operand copies.
KERNELS = {"mm": 1, "mv": 1, "rhs": 2, "empty": 1}


def _indices(index, batch):
    """The broadcast index shape and its values; None is ops.cpp's arange."""
    if index is None:
        return list(batch), list(range(_prod(batch)))
    shape, values = index
    return list(shape), list(values)


def _prod(values):
    total = 1
    for value in values:
        total *= value
    return total


def _broadcast_shape(*shapes):
    rank = max(len(shape) for shape in shapes)
    padded = [[1] * (rank - len(shape)) + list(shape) for shape in shapes]
    return [max(extents) for extents in zip(*padded)]


def case_record(case):
    name, dtype, a, b, lhs, rhs, sort = case
    views = {}
    for key, (stored, view) in zip("ab", (a, b)):
        shape, strides = matmul.view_layout(stored, view)
        views[key] = {
            "stored": list(stored),
            "view": view,
            "shape": shape,
            "strides": strides,
        }
    lhs_shape, lhs_values = _indices(lhs, views["a"]["shape"][:-2])
    rhs_shape, rhs_values = _indices(rhs, views["b"]["shape"][:-2])
    index_shape = _broadcast_shape(lhs_shape, rhs_shape)
    empty = 0 in views["a"]["shape"] or 0 in views["b"]["shape"]
    if empty:
        route = {"kind": "empty", "entries": [], "copies": {}}
    else:
        route = gather_mm_layout.route(
            dtype,
            views["a"]["shape"],
            views["a"]["strides"],
            views["b"]["shape"],
            views["b"]["strides"],
            index_shape,
            sort and lhs is None,
        )
    return {
        "name": name,
        "kind": route["kind"],
        "dtype": dtype,
        **views,
        "lhs": None if lhs is None else {"shape": lhs_shape, "values": lhs_values},
        "rhs": {"shape": rhs_shape, "values": rhs_values},
        "indexShape": index_shape,
        "sorted": sort,
        "entries": route["entries"],
        "route": route,
        "copies": sum(route["copies"].values()),
        # ops.cpp builds each missing index array with one arange.
        "defaults": int(lhs is None),
    }


def _gathered_reference(np, a, b, lhs, rhs, index_shape):
    """ops.cpp gather_mm: out[i] = a[lhs[i]] @ b[rhs[i]] over the index batch."""
    a_batches = a.reshape((_prod(a.shape[:-2]), *a.shape[-2:]))
    b_batches = b.reshape((_prod(b.shape[:-2]), *b.shape[-2:]))
    lhs = np.broadcast_to(lhs, index_shape).reshape(-1)
    rhs = np.broadcast_to(rhs, index_shape).reshape(-1)
    products = [a_batches[i] @ b_batches[j] for i, j in zip(lhs, rhs)]
    out_shape = (*index_shape, a.shape[-2], b.shape[-1])
    if not products:
        return np.zeros(out_shape)
    return np.stack(products).reshape(out_shape)


def case_data(np, record, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, size=record["a"]["stored"]) / 4.0
    b = rng.integers(-4, 5, size=record["b"]["stored"]) / 8.0
    a_view = matmul._view(np, a, record["a"]["view"])
    b_view = matmul._view(np, b, record["b"]["view"])
    lhs = record["lhs"] or {
        "shape": list(a_view.shape[:-2]),
        "values": list(range(_prod(a_view.shape[:-2]))),
    }
    total = _gathered_reference(
        np,
        a_view,
        b_view,
        np.array(lhs["values"], dtype=np.int64).reshape(lhs["shape"]),
        np.array(record["rhs"]["values"], dtype=np.int64).reshape(
            record["rhs"]["shape"]
        ),
        record["indexShape"],
    )
    if total.size and np.abs(total).max() >= 2**19:
        raise ValueError("GatherMM case exceeds the exact float32 budget")
    return {"a": a, "b": b, "expected": rounded(total, record["dtype"])}


def _operand(mx, np, values, view, dtype):
    array = mx.array(values.astype(np.float32), dtype=dtype)
    if view == "dense":
        return array, array
    if view == "swap":
        return array, mx.swapaxes(array, -1, -2)
    if view == "step":
        return array, array[..., ::2]
    raise ValueError("Unsupported GatherMM operand view")


def _call(mx, record, a, b, stream=None):
    def index(spec):
        if spec is None:
            return None
        values = mx.array(spec["values"], dtype=mx.uint32)
        return values.reshape(spec["shape"])

    keywords = {"sorted_indices": record["sorted"]}
    if stream is not None:
        keywords["stream"] = stream
    return mx.gather_mm(a, b, index(record["lhs"]), index(record["rhs"]), **keywords)


def run_case(mx, np, host, record, seed):
    data = case_data(np, record, seed)
    dtype = getattr(mx, record["dtype"])
    stored, views = {}, {}
    for key in "ab":
        stored[key], views[key] = _operand(
            mx, np, data[key], record[key]["view"], dtype
        )
    mx.eval(*stored.values())
    before = {
        key: matmul.float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for key, value in stored.items()
    }
    start = host.dispatch_count
    unsupported = record["kind"] == "rhs" and host.target in RHS_UNSUPPORTED
    if unsupported:
        # The default indices, copies and group offsets run; the product's
        # translation then fails before it launches.
        try:
            mx.eval(_call(mx, record, views["a"], views["b"]))
        except RuntimeError as error:
            count = host.dispatch_count - start
            if (
                "software subgroup" not in str(error)
                or count != record["defaults"] + record["copies"] + 1
            ):
                raise
            return {
                **record,
                "seed": seed,
                "dispatchStart": start,
                "dispatchCount": count,
                "unsupported": RHS_UNSUPPORTED[host.target],
            }
        raise AssertionError(
            f"GatherMM sorted route ran untranslated: {record['name']}"
        )
    y = _call(mx, record, views["a"], views["b"])
    mx.eval(y)
    count = host.dispatch_count - start
    expected = data["expected"]
    actual = np.asarray(y.tolist(), dtype=np.float32).reshape(expected.shape)
    if list(y.shape) != list(expected.shape) or y.dtype != dtype:
        raise AssertionError(f"GatherMM output layout differs: {record['name']}")
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"GatherMM output differs: {record['name']}")
    # The default indices' aranges, one translated copy for each operand the
    # host copies, then the kernels.
    if count != KERNELS[record["kind"]] + record["copies"] + record["defaults"]:
        raise AssertionError(f"GatherMM dispatch count differs: {record['name']}")
    for key, value in stored.items():
        if before[key] != matmul.float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"GatherMM input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        reference = _call(mx, record, views["a"], views["b"], stream=mx.cpu)
        cpu = np.asarray(reference.tolist(), dtype=np.float32).reshape(expected.shape)
        if cpu.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(f"GatherMM differs from MLX CPU: {record['name']}")
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu is not None,
        "inputUnchanged": True,
    }


def _flat(values, shape):
    """Row-major values of ``values`` broadcast to ``shape``."""
    rank = len(shape)
    source = [1] * (rank - len(values["shape"])) + list(values["shape"])
    strides, stride = [0] * rank, 1
    for axis in reversed(range(rank)):
        strides[axis] = 0 if source[axis] == 1 else stride
        stride *= source[axis]
    result = []
    for flat in range(_prod(shape)):
        offset, rest = 0, flat
        for axis in reversed(range(rank)):
            offset += (rest % shape[axis]) * strides[axis]
            rest //= shape[axis]
        result.append(values["values"][offset])
    return result


def _launch_matches(events, record):
    """The traced launches gather the record's indices with its geometry."""
    route = record["route"]
    if record["kind"] == "rhs":
        offsets, product = events
        groups = record["b"]["shape"][0] if len(record["b"]["shape"]) > 2 else 1
        rows = _flat(record["rhs"], record["indexShape"])
        return (
            offsets["gatherMetadata"]["groups"] == groups
            and product["gatherMetadata"]["offsets"]
            == gather_mm_layout.offsets_reference(rows, groups)
            and product["gatherMetadata"]["M"] == route["M"]
            and product["gatherMetadata"]["ldb"] == route["ldb"]
        )
    (event,) = events
    metadata = event["gatherMetadata"]
    lhs = record["lhs"] or {
        "shape": record["a"]["shape"][:-2],
        "values": list(range(_prod(record["a"]["shape"][:-2]))),
    }
    lhs_values = _flat(lhs, record["indexShape"])
    rhs_values = _flat(record["rhs"], record["indexShape"])
    if record["kind"] == "mv":
        mat, vec = (
            (lhs_values, rhs_values)
            if record["a"]["shape"][-2] != 1
            else (rhs_values, lhs_values)
        )
        return (
            metadata["matrixIndices"] == mat
            and metadata["vectorIndices"] == vec
            and metadata["K"] == route["K"]
            and metadata["outVecSize"] == route["outVecSize"]
            and metadata["matrixLd"] == route["matrixLd"]
        )
    return (
        metadata["lhsIndices"] == lhs_values
        and metadata["rhsIndices"] == rhs_values
        and (metadata["lda"], metadata["ldb"]) == (route["lda"], route["ldb"])
    )


def _covered(records, target):
    entries = [entry for record in records for entry in record["entries"]]
    mm = [
        gather_mm_layout.signature(entry)
        for entry in entries
        if entry.startswith("steel_gather_mm_") and "_rhs_" not in entry
    ]
    mv = [
        gather_mv_layout.signature(entry)
        for entry in entries
        if entry.startswith(gather_mv_layout.ENTRY_PREFIXES)
    ]
    rhs = [gather_mm_layout.signature(entry) for entry in entries if "_rhs_" in entry]
    return (
        {info["dtype"] for info in mm} == set(gather_mm_layout.DTYPES)
        and all(
            {info["constants"][name] for info in mm} == {False, True}
            for name in gather_mm_layout.CONSTANTS
        )
        and {(info["transposeA"], info["transposeB"]) for info in mm}
        >= {(False, False), (False, True), (True, False)}
        and {info["kind"] for info in mv} == {"gemv_gather", "gemv_t_gather"}
        and {info["transposeB"] for info in rhs} == {False, True}
        and {record["kind"] for record in records} == set(KERNELS)
        and any(record["copies"] for record in records)
        and any(record["lhs"] is None for record in records if record["kind"] == "mm")
        and (
            target in RHS_UNSUPPORTED
            or all("outputHash" in record for record in records)
        )
    )


def validate_trace(trace, records, target):
    if any(event.get("target") != target for event in trace):
        raise RuntimeError("GatherMM trace does not match the routed entries")
    for record in records:
        start, count = record["dispatchStart"], record["dispatchCount"]
        events = trace[start : start + count]
        products = [
            event
            for event in events
            if event.get("entry", "").startswith(
                (
                    *gather_mv_layout.ENTRY_PREFIXES,
                    "steel_gather_mm_",
                    "gather_mm_offsets",
                )
            )
        ]
        expected = (
            record["entries"][:1] if record.get("unsupported") else record["entries"]
        )
        if [event["entry"] for event in products] != expected:
            raise RuntimeError(
                f"GatherMM trace differs from its route: {record['name']}"
            )
        others = [event for event in events if event not in products]
        aranges = [e for e in others if e.get("entry", "").startswith("arange")]
        copies = [e for e in others if "copy" in e.get("entry", "")]
        if record["kind"] == "empty":
            if len(others) != 1 or len(copies) != 1:
                raise RuntimeError("GatherMM empty contraction is not a zero fill")
            continue
        if (
            len(aranges) != record["defaults"]
            or len(copies) != record["copies"]
            or len(others) != len(aranges) + len(copies)
        ):
            raise RuntimeError(f"GatherMM copies differ: {record['name']}")
        if record.get("unsupported"):
            continue
        if not _launch_matches(products, record):
            raise RuntimeError(f"GatherMM launch differs: {record['name']}")
    if not _covered(records, target):
        raise RuntimeError("GatherMM cases do not cover every route")


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
        raise RuntimeError("MLX sources changed during GatherMM verification")
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
