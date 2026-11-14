"""Exercise MLX SegmentedMM through the translated segmented steel GEMM kernel.

Every case routes one ``mx.segmented_mm`` call as ``SegmentedMM::eval_gpu``
does: to ``steel_segmented_mm`` with one ``(k_start, k_end)`` pair per output
batch, or with batch boundaries for a view whose trailing strides are both
one, after copying operands and segments the kernel cannot read in place; to
a translated fill of zeros for an empty contraction; or to no launch for an
empty output. Inputs lie on dyadic grids, so every product is exact in
float32 and rounds once to the output type: every output bit is checked
against the float64 reference, and float32 outputs against MLX's CPU as well.
"""

import argparse
import hashlib
import json
from pathlib import Path

from demos.integrations.mlx.portable_host import segmented_mm_layout
from demos.integrations.mlx.portable_host import verify_matmul as matmul
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

BK = segmented_mm_layout.HOST_TILE[2]
# (name, dtype, a, b, segments). Operands are (stored shape, view) as in
# verify_matmul. Segments are (layout, shape, words): ``pairs`` binds a dense
# (..., 2) array, ``boundaries`` the (S, 2) view with strides (1, 1) over S + 1
# boundary words, and ``transposed`` the transpose of a dense (2, S) array.
CASES = (
    # Unaligned rows and columns; whole K tiles with a tail, an empty segment,
    # a tail alone and tiles that end exactly on a segment's end.
    (
        "pairs-unaligned",
        "float32",
        ((37, 50), "dense"),
        ((50, 70), "dense"),
        ("pairs", (5, 2), (5, 50, 17, 17, 3, 10, 0, 32, 1, 49)),
    ),
    # Whole tiles in both dimensions.
    (
        "pairs-aligned",
        "float16",
        ((64, 40), "dense"),
        ((40, 128), "dense"),
        ("pairs", (2, 2), (0, 40, 8, 24)),
    ),
    # Batch boundaries over a transposed A: consecutive segments, one empty.
    (
        "boundaries-transposed-a",
        "bfloat16",
        ((48, 64), "swap"),
        ((48, 30), "dense"),
        ("boundaries", (4, 2), (0, 16, 16, 40, 48)),
    ),
    # A two-dimensional segment batch over a transposed B with whole columns.
    (
        "pairs-batched-transposed-b",
        "float32",
        ((20, 33), "dense"),
        ((64, 33), "swap"),
        ("pairs", (2, 3, 2), (0, 33, 0, 1, 2, 18, 18, 18, 30, 33, 7, 29)),
    ),
    # Both operands transposed, segments copied from a transposed view.
    (
        "copied-segments-transposed",
        "float16",
        ((40, 70), "swap"),
        ((66, 40), "swap"),
        ("transposed", (3, 2), (0, 9, 20, 40, 1, 36)),
    ),
    # A copied from a strided view, boundaries ending before the contraction.
    (
        "copied-a-boundaries",
        "bfloat16",
        ((65, 76), "step"),
        ((38, 64), "dense"),
        ("boundaries", (3, 2), (2, 7, 23, 37)),
    ),
    # An empty contraction fills the output with zeros.
    (
        "empty-k",
        "float32",
        ((3, 0), "dense"),
        ((0, 4), "dense"),
        ("pairs", (2, 2), (0, 0, 0, 0)),
    ),
    # No segment: the output is empty and nothing launches.
    (
        "empty-output",
        "float16",
        ((5, 6), "dense"),
        ((6, 7), "dense"),
        ("pairs", (0, 2), ()),
    ),
)
# Translated dispatches per kind besides operand and segment copies.
KERNELS = {"segmented": 1, "empty": 1, "none": 0}


def _prod(values):
    total = 1
    for value in values:
        total *= value
    return total


def segments_view(spec):
    """``(shape, strides, row_contiguous, pairs)`` of a segments spec."""
    layout, shape, words = spec
    shape = list(shape)
    batches = _prod(shape[:-1])
    if layout == "pairs":
        strides = [_prod(shape[axis + 1 :]) for axis in range(len(shape))]
        pairs = [tuple(words[2 * z : 2 * z + 2]) for z in range(batches)]
        return shape, strides, True, pairs
    if layout == "boundaries":
        pairs = [(words[z], words[z + 1]) for z in range(batches)]
        # MLX marks a (1, 2) view of two words row-contiguous.
        return shape, [1, 1], batches == 1, pairs
    if layout == "transposed":
        pairs = [(words[z], words[batches + z]) for z in range(batches)]
        return shape, [1, batches], False, pairs
    raise ValueError("Unsupported SegmentedMM segments layout")


def case_record(case):
    name, dtype, a, b, segments = case
    views = {}
    for key, (stored, view) in zip("ab", (a, b)):
        shape, strides = matmul.view_layout(stored, view)
        views[key] = {
            "stored": list(stored),
            "view": view,
            "shape": shape,
            "strides": strides,
        }
    seg_shape, seg_strides, row_contiguous, pairs = segments_view(segments)
    m, k, n = views["a"]["shape"][-2], views["a"]["shape"][-1], views["b"]["shape"][-1]
    out_shape = [*seg_shape[:-1], m, n]
    if _prod(out_shape) == 0:
        route = {"kind": "none", "entries": [], "copies": {}}
    elif k == 0:
        route = {"kind": "empty", "entries": [], "copies": {}}
    else:
        route = segmented_mm_layout.route(
            dtype,
            views["a"]["shape"],
            views["a"]["strides"],
            views["b"]["shape"],
            views["b"]["strides"],
            (seg_shape, seg_strides, row_contiguous),
        )
    return {
        "name": name,
        "kind": route["kind"],
        "dtype": dtype,
        **views,
        "segments": {
            "layout": segments[0],
            "shape": seg_shape,
            "words": list(segments[2]),
            "pairs": [list(pair) for pair in pairs],
        },
        "outShape": out_shape,
        "entries": route["entries"],
        "route": route,
        "copies": sum(route["copies"].values()),
    }


def _reference(np, a, b, pairs, out_shape):
    """cpu/masked_mm.cpp segmented_mm: batch z is a[:, s:e] @ b[s:e]."""
    m, n = out_shape[-2:]
    products = [
        a[:, start:end] @ b[start:end] if end > start else np.zeros((m, n))
        for start, end in pairs
    ]
    if not products:
        return np.zeros(out_shape)
    return np.stack(products).reshape(out_shape)


def _nonzero(rng, size):
    """Nonzero integers in [-4, 4]: no product is a signed zero, so a sum is
    -0 in neither the reference, the kernel's +0 accumulators nor MLX's CPU."""
    return rng.integers(1, 5, size=size) * rng.choice((-1, 1), size=size)


def case_data(np, record, seed):
    rng = np.random.default_rng(seed)
    a = _nonzero(rng, record["a"]["stored"]) / 4.0
    b = _nonzero(rng, record["b"]["stored"]) / 8.0
    total = _reference(
        np,
        matmul._view(np, a, record["a"]["view"]),
        matmul._view(np, b, record["b"]["view"]),
        record["segments"]["pairs"],
        record["outShape"],
    )
    if total.size and np.abs(total).max() >= 2**19:
        raise ValueError("SegmentedMM case exceeds the exact float32 budget")
    # The kernel's accumulators start at +0.
    return {"a": a, "b": b, "expected": rounded(total + 0.0, record["dtype"])}


def _operand(mx, np, values, view, dtype):
    array = mx.array(values.astype(np.float32), dtype=dtype)
    if view == "dense":
        return array, array
    if view == "swap":
        return array, mx.swapaxes(array, -1, -2)
    if view == "step":
        return array, array[..., ::2]
    raise ValueError("Unsupported SegmentedMM operand view")


def _segments(mx, spec):
    """The stored words and the view of them ``mx.segmented_mm`` receives."""
    layout, shape, words = spec["layout"], spec["shape"], spec["words"]
    stored = mx.array(words, dtype=mx.uint32)
    if layout == "pairs":
        return stored, stored.reshape(shape)
    if layout == "boundaries":
        return stored, mx.as_strided(stored, shape, [1, 1])
    return stored, stored.reshape([2, shape[0]]).T


def run_case(mx, np, host, record, seed):
    data = case_data(np, record, seed)
    dtype = getattr(mx, record["dtype"])
    stored, views = {}, {}
    for key in "ab":
        stored[key], views[key] = _operand(
            mx, np, data[key], record[key]["view"], dtype
        )
    stored["segments"], segments = _segments(mx, record["segments"])
    # The views are evaluated first: MLX gives them shared buffers, so the
    # counted dispatches are the copies SegmentedMM makes and its kernel.
    mx.eval(*stored.values(), segments)
    pairs = record["segments"]["pairs"]
    if (
        list(segments.shape) != record["segments"]["shape"]
        or [list(pair) for pair in segments.reshape([len(pairs), 2]).tolist()] != pairs
    ):
        raise AssertionError(f"SegmentedMM segments view differs: {record['name']}")
    before = {
        key: matmul.float_bits(np.asarray(value.tolist(), dtype=np.float64).reshape(-1))
        for key, value in stored.items()
    }
    start = host.dispatch_count
    y = mx.segmented_mm(views["a"], views["b"], segments)
    mx.eval(y)
    count = host.dispatch_count - start
    expected = data["expected"]
    actual = np.asarray(y.tolist(), dtype=np.float32).reshape(expected.shape)
    if list(y.shape) != list(expected.shape) or y.dtype != dtype:
        raise AssertionError(f"SegmentedMM output layout differs: {record['name']}")
    if actual.view(np.uint32).tobytes() != expected.view(np.uint32).tobytes():
        raise AssertionError(f"SegmentedMM output differs: {record['name']}")
    if count != KERNELS[record["kind"]] + record["copies"]:
        raise AssertionError(f"SegmentedMM dispatch count differs: {record['name']}")
    for key, value in stored.items():
        if before[key] != matmul.float_bits(
            np.asarray(value.tolist(), dtype=np.float64).reshape(-1)
        ):
            raise AssertionError(f"SegmentedMM input changed: {record['name']}")
    cpu = None
    if record["dtype"] == "float32":
        reference = mx.segmented_mm(views["a"], views["b"], segments, stream=mx.cpu)
        cpu = np.asarray(reference.tolist(), dtype=np.float32).reshape(expected.shape)
        if cpu.view(np.uint32).tobytes() != actual.view(np.uint32).tobytes():
            raise AssertionError(f"SegmentedMM differs from MLX CPU: {record['name']}")
    return {
        **record,
        "seed": seed,
        "dispatchStart": start,
        "dispatchCount": count,
        "outputHash": hashlib.sha256(actual.tobytes()).hexdigest(),
        "cpuBitwiseEqual": cpu is not None,
        "inputUnchanged": True,
    }


def _launch_matches(event, record):
    """The traced launch reads the record's segments with its geometry."""
    route, metadata = record["route"], event["gatherMetadata"]
    loops = [
        list(segmented_mm_layout.k_loop(BK, start, end)[:2])
        for start, end in record["segments"]["pairs"]
    ]
    return (
        metadata["segments"] == record["segments"]["pairs"]
        and metadata["kLoops"] == loops
        and metadata["batches"] == len(record["segments"]["pairs"])
        and all(metadata[key] == route[key] for key in ("M", "N", "K", "lda", "ldb"))
        and metadata["constants"]["segments_contiguous"] == route["contiguous"]
    )


def _covered(records):
    infos = [
        segmented_mm_layout.signature(entry)
        for record in records
        for entry in record["entries"]
    ]
    loops = [
        segmented_mm_layout.k_loop(BK, start, end)[:2]
        for record in records
        if record["kind"] == "segmented"
        for start, end in record["segments"]["pairs"]
    ]
    return (
        {info["dtype"] for info in infos} == set(segmented_mm_layout.DTYPES)
        and all(
            {info["constants"][name] for info in infos} == {False, True}
            for name in segmented_mm_layout.CONSTANTS
        )
        and {(info["transposeA"], info["transposeB"]) for info in infos}
        == {(False, False), (False, True), (True, False), (True, True)}
        and {record["kind"] for record in records} == set(KERNELS)
        and any(record["route"]["copies"].get("a") for record in records)
        and any(record["route"]["copies"].get("segments") for record in records)
        # Whole tiles with and without a tail, a tail alone, and no read.
        and any(tiles and tail > 0 for tiles, tail in loops)
        and any(tiles and tail <= 0 for tiles, tail in loops)
        and any(not tiles and tail > 0 for tiles, tail in loops)
        and any(not tiles and tail <= 0 for tiles, tail in loops)
        and all("outputHash" in record for record in records)
    )


def validate_trace(trace, records, target):
    if any(event.get("target") != target for event in trace):
        raise RuntimeError("SegmentedMM trace does not match the routed entries")
    for record in records:
        start, count = record["dispatchStart"], record["dispatchCount"]
        events = trace[start : start + count]
        products = [
            event
            for event in events
            if event.get("entry", "").startswith(segmented_mm_layout.ENTRY_PREFIXES)
        ]
        if [event["entry"] for event in products] != record["entries"]:
            raise RuntimeError(
                f"SegmentedMM trace differs from its route: {record['name']}"
            )
        others = [event for event in events if event not in products]
        copies = [event for event in others if "copy" in event.get("entry", "")]
        expected = record["copies"] + (record["kind"] == "empty")
        if len(copies) != expected or len(others) != len(copies):
            raise RuntimeError(f"SegmentedMM copies differ: {record['name']}")
        if products and not _launch_matches(products[0], record):
            raise RuntimeError(f"SegmentedMM launch differs: {record['name']}")
    if not _covered(records):
        raise RuntimeError("SegmentedMM cases do not cover every route")


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
        raise RuntimeError("MLX sources changed during SegmentedMM verification")
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
