"""Exercise MLX quantized_matmul through translated affine matrix-matrix kernels.

Products with at least ``matmul_layout.vector_limit`` rows reach quantized.cpp's
matrix-matrix routes: batched transposed products ``affine_qmm_t``, single
transposed matrices ``affine_qmm_t_splitk`` and the column sum over its
partitions, or ``affine_qmm_t`` when ``K`` admits no split, and untransposed
products ``affine_qmm_n``.

Every case routes one ``mx.quantized_matmul`` call to its translated entries
and checks each output bit against an exact float64 reference. Inputs lie on
dyadic grids with power-of-two scales, so every dequantized weight is exact in
the element type the kernels dequantize into and every float32 partial sum is
exact, whatever its order; only the final cast rounds. 16-bit float split-K
products also round each partition's product and every addition of their
partition sum. Below 32 partitions the reference repeats those roundings in
the small column sum's order; the looped and two-pass sums' simd reductions
have no fixed order, so their cases keep every partial product and sum exact
in the element type (``verify_quantized_matmul.order_free``).
"""

import argparse
import json
import math
from pathlib import Path

from demos.integrations.mlx.portable_host import column_small_layout, qmm_layout
from demos.integrations.mlx.portable_host.prepare import verify_prepared
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.verify_quantized_matmul import run_case

# (name, dtype, group, bits, K, N, x shape before K, w batch shape, layout,
# transpose), as in verify_quantized_matmul. Layouts: "dense" binds x as
# given, "strided" takes every other x batch.
CASES = (
    # qmm_t: stacked aligned weights, broadcast unaligned weights, 2-D weights
    # under a strided x batch, and single matrices whose K admits no split,
    # one of them folding a 3-D x into its rows.
    ("qmm-t-stacked", "float16", 64, 4, 128, 64, (2, 40), (2,), "dense", True),
    (
        "qmm-t-broadcast-unaligned",
        "bfloat16",
        32,
        8,
        96,
        40,
        (3, 33),
        (1,),
        "dense",
        True,
    ),
    ("qmm-t-shared-weights", "float32", 128, 3, 256, 72, (3, 20), (), "strided", True),
    ("qmm-t-unsplit", "float16", 128, 6, 128, 96, (40,), (), "dense", True),
    ("qmm-t-unsplit-folded", "float32", 128, 2, 128, 33, (2, 20), (), "dense", True),
    # qmm_t_splitk: 4 partitions of one unaligned column tile (a small sum),
    # 64 partitions of one partial row tile (a looped sum), 4 partitions of
    # partial row and column tiles, and 64 partitions holding 65,536 partial
    # products.
    ("qmm-t-split-small", "float32", 64, 4, 256, 8, (3, 10), (), "dense", True),
    ("qmm-t-split-looped", "float32", 32, 2, 2048, 32, (20,), (), "dense", True),
    ("qmm-t-split-tiles", "float32", 128, 5, 512, 99, (129,), (), "dense", True),
    ("qmm-t-split-wide-sum", "float32", 32, 6, 2048, 32, (32,), (), "dense", True),
    # 512 partitions of one tile: the two-pass sum.
    ("qmm-t-split-two-pass", "float32", 32, 4, 16384, 32, (20,), (), "dense", True),
    # 16-bit float partitions, summed in their type: 4 partitions, 16
    # partitions that each threadgroup row adds two of, test_qmm's 8
    # partitions of 131,072 float16 partial products, 64 partitions of one
    # and of four tiles (looped sums), and 512 and 384 partitions of one tile
    # (two-pass sums), the bfloat16 one over folded rows.
    ("qmm-t-split-half", "float16", 64, 4, 256, 8, (30,), (), "dense", True),
    (
        "qmm-t-split-bfloat-interleaved",
        "bfloat16",
        32,
        3,
        512,
        32,
        (32,),
        (),
        "dense",
        True,
    ),
    ("qmm-t-split-half-wide", "float16", 32, 4, 256, 256, (64,), (), "dense", True),
    ("qmm-t-split-half-looped", "float16", 32, 3, 2048, 32, (20,), (), "dense", True),
    (
        "qmm-t-split-bfloat-looped",
        "bfloat16",
        64,
        2,
        4096,
        40,
        (33,),
        (),
        "dense",
        True,
    ),
    (
        "qmm-t-split-half-two-pass",
        "float16",
        32,
        2,
        16384,
        32,
        (20,),
        (),
        "dense",
        True,
    ),
    (
        "qmm-t-split-bfloat-two-pass",
        "bfloat16",
        64,
        3,
        24576,
        24,
        (2, 6),
        (),
        "dense",
        True,
    ),
    # qmm_n: a K tail over two row tiles, stacked, broadcast and shared
    # weights, and a single row tile at the vector limit. MLX gives every
    # broadcast unit axis a zero stride, so broadcast weights keep two groups
    # per row: one would send their scales through ensure_row_contiguous_matrix's
    # copy.
    ("qmm-n-k-tail", "float32", 32, 3, 45, 64, (36,), (), "dense", False),
    ("qmm-n-stacked", "float16", 64, 8, 40, 128, (2, 33), (2,), "dense", False),
    ("qmm-n-broadcast", "bfloat16", 64, 2, 64, 128, (3, 8), (1,), "dense", False),
    ("qmm-n-shared-weights", "float16", 32, 5, 33, 32, (3, 10), (), "strided", False),
    ("qmm-n-limit", "float32", 64, 6, 13, 64, (4,), (), "dense", False),
)
# The column sums of each qmm_layout.partition_sum plan, in dispatch order:
# col_reduce_2pass leaves 32 partials of each output for a looped pass.
SUM_KINDS = {
    "small": ("col_reduce_small_1",),
    "looped": ("col_reduce_looped_1_32_32",),
    "two-pass": ("col_reduce_2pass_1_32_32", "col_reduce_looped_1_32_32"),
}
SUM_ENTRIES = {
    f"{kind}_reduce_sum{dtype}"
    for kind in (
        "col_reduce_small_1",
        "col_reduce_looped_1_32_32",
        "col_reduce_2pass_1_32_32",
    )
    for dtype in ("float32", "float16", "bfloat16")
}


def case_record(case):
    name, dtype, group, bits, k, n, x_prefix, w_batch, layout, transpose = case
    m = x_prefix[-1]
    batches = math.prod(x_prefix[:-1]) if len(x_prefix) > 1 else 1
    if not w_batch and layout == "dense":
        # w.ndim() == 2 and a row-contiguous x: MLX folds every x row into M.
        m, batches = math.prod(x_prefix), 1
    entry = qmm_layout.select(dtype, group, bits, m, n, k, batches, transpose)
    split = (
        qmm_layout.split_k(m, n, k, group)
        if qmm_layout.signature(entry)["kernel"] == "qmm_t_splitk"
        else 1
    )
    sums = []
    if split > 1:
        kinds = SUM_KINDS[qmm_layout.partition_sum(m, n, split)]
        sums = [f"{kind}_reduce_sum{dtype}" for kind in kinds]
    width = k // split
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
        "splitK": split,
        "entry": entry,
        "dispatches": [entry, *sums],
        # qmm_splitk's partitions: k_partition_size = K / split_k columns each.
        "partitions": [[start, start + width] for start in range(0, k, width)],
    }


def small_column_variants():
    """The small column sum variants the split-K cases launch, as cache keys."""
    keys = set()
    for case in CASES:
        record = case_record(case)
        for entry in record["dispatches"][1:]:
            if entry in column_small_layout.ALL_SMALL_COLUMN_ENTRIES:
                lanes, rows = column_small_layout.geometry(
                    record["M"] * record["N"], record["splitK"]
                )
                keys.add(column_small_layout.variant(entry, (lanes, rows, 1)))
    return keys


def validate_trace(trace, records, target):
    expected = [entry for record in records for entry in record["dispatches"]]
    if [event.get("entry") for event in trace] != expected or any(
        event.get("target") != target for event in trace
    ):
        raise RuntimeError("Quantized qmm trace does not match the routed entries")
    products = [event for event in trace if event["entry"].startswith("affine_")]
    for event, record in zip(products, records):
        metadata = event["matmulMetadata"]
        if (
            metadata["M"],
            metadata["N"],
            metadata["K"],
            metadata["batches"],
            metadata["splitK"],
        ) != (
            record["M"],
            record["N"],
            record["K"],
            record["batches"],
            record["splitK"],
        ):
            raise RuntimeError(f"Quantized qmm launch differs: {record['name']}")
    signatures = [qmm_layout.signature(record["entry"]) for record in records]
    sums = {entry for entry in expected if entry.startswith("col_reduce_")}
    if (
        {info["kernel"] for info in signatures} != {"qmm_t", "qmm_t_splitk", "qmm_n"}
        or {info["batched"] for info in signatures} != {False, True}
        or {info["aligned"] for info in signatures} != {None, False, True}
        or {info["dtype"] for info in signatures} != {"float32", "float16", "bfloat16"}
        or sums != SUM_ENTRIES
    ):
        raise RuntimeError("Quantized qmm cases do not cover every route")


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
        raise RuntimeError("MLX sources changed during quantized qmm verification")
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
