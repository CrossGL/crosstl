"""Tile-level model of the pinned MLX ``fp_qmm_n_impl``.

``fp_qmm_n_impl`` multiplies activations by weights that are not transposed:
``K`` rows of ``N`` codes, quantized along ``N``. Its four-element activation
``BlockLoader``, its last pass over the ``K % BK`` remaining rows and its
stores are ``quantized_qmm_n_model``'s, and ``BlockMMA``'s float32 fragment
products ``quantized_qmm_t_model``'s. It differs from the affine kernel where
this module follows ``fp_quantized.h`` at the pinned commit:

- the fp ``QuantizedBlockLoader`` reads one E4M3 or E8M0 scale per thread's
  eight codes, ``(bj * pack_factor) / group_size`` groups into its row, and
  rounds every dequantized weight to the element type;
- groups of 16 make ``partial_n``: ``N`` need only be a multiple of 16, so a
  column tile may hold 16 columns. Its weight loads bound only rows, so the
  threads past ``N`` dequantize the next weight row's first bytes and scale,
  or the guards after the last row, into columns the safe store never
  writes. ``simulate`` records those reads apart from the in-row ones.

The ``_hgs`` instantiations pass ``fp_qmm_n_impl`` no global scale, so they
compute exactly what the plain ones do. Named mutations show which defects
the expectations reject.
"""

import math

from demos.integrations.mlx.tests.kernels import fp_qmm_t_model as fp
from demos.integrations.mlx.tests.kernels import quantized_qmm_n_model as affine
from demos.integrations.mlx.tests.kernels import quantized_qmm_t_model as base
from demos.integrations.mlx.tests.kernels.fp_qmv_model import _checked, _scale_value
from demos.integrations.mlx.tests.kernels.test_current_quantized import _to_type

BM, BN, BK, THREADS = base.BM, base.BN, base.BK, base.THREADS


def geometry(bits, group_size):
    """The fp QuantizedBlockLoader's constants with ``reduction_dim = 0``."""
    shared = base.geometry(bits)
    assert shared["packFactor"] == 8 // bits and shared["bytesPerPack"] == 1
    return {**shared, "partialN": group_size % BN != 0}


MUTATIONS = {
    "order": "pairs each byte's codes with its columns in reverse",
    "sign": "decodes every code without its sign bit",
    "scale-split": "gives every thread its row's first group in the tile",
    "scale-columns": "ignores the column tile's group offset",
    "scale-step": "keeps the first block's scales",
    "scale-stride": "steps the scales one group per block, as along K",
    "k-step": "keeps the activation loader on the first block",
    "k-tail": "drops the last partial block of the reduction",
    "tail-mask": "loads the last partial block without load_safe's bounds",
    "rows": "ignores the threadgroup's row offset",
    "columns": "ignores the threadgroup's column offset",
    "transpose": "reads the weight tile as (BN, BK)",
    "store": "stores whole tiles past M and N",
    "batch": "ignores the weight batch offsets",
}


def _loaders(context, wl, scales, mutation):
    """Each thread's QuantizedBlockLoader state after its constructor."""
    shape, n = context["geometry"], context["N"]
    reads, packed = shape["reads"], shape["packedColumns"]
    pack_factor, group_size = shape["packFactor"], context["groupSize"]
    row_bytes = n * shape["bytesPerPack"] // pack_factor
    row_groups = n // group_size
    loaders = []
    for thread in range(THREADS):
        bi, bj = reads * thread // packed, reads * thread % packed
        # BCOLS_PACKED * BROWS reaches the threadgroup size at both widths.
        assert packed * BK >= THREADS
        group = 0 if mutation == "scale-split" else bj * pack_factor // group_size
        loaders.append(
            {
                "bi": bi,
                "bj": bj,
                "src": wl + bi * row_bytes + bj * shape["bytesPerPack"],
                "scales": scales + bi * row_groups + group,
            }
        )
    return loaders


def _load_w(context, state, loaders, rows, first_column, tiles, mutation):
    """load_safe(short2(num_outs, rows)) or load_unsafe of one weight block.

    Both bound the rows only: a thread whose codes start at or past ``N``
    reads past its row.
    """
    storage, extra = state["storage"], state["extra"]
    shape, ld, ws = context["geometry"], context["ld"], context["Ws"]
    pack_factor, reads = shape["packFactor"], shape["reads"]
    for loader in loaders:
        bi = loader["bi"]
        dst = bi * ld + loader["bj"] * pack_factor
        if bi >= rows:
            for i in range(reads * pack_factor):
                tiles["Ws"].add(dst + i)
                ws[dst + i] = 0.0
            continue
        # A thread's eight codes lie in one group, wholly before or past N.
        over = first_column + loader["bj"] * pack_factor >= context["N"]
        scale_byte = fp._read(context, storage, extra, "scales", loader["scales"], over)
        if "bytes" in context["records"]:
            context["records"]["bytes"].add(("scales", loader["scales"]))
        if scale_byte != scale_byte:
            scale_byte = 0
        # float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc,
        # whose factor stays 1 without a global scale.
        scale = _scale_value(context["mode"], scale_byte)
        try:
            narrowed = _to_type(context["dtype"], scale)
        except OverflowError:
            # A mutation's read of a guard: E8M0 2^38 overflows float16.
            if context["strict"]:
                raise
            narrowed = math.inf
        if context["strict"] and not over:
            assert narrowed == scale, scale_byte
        for i in range(reads):
            byte = fp._byte(context, storage, extra, loader["src"] + i, over)
            weights = fp._weights(context, byte, narrowed, mutation, not over)
            for j, weight in enumerate(weights):
                tiles["Ws"].add(dst + i * pack_factor + j)
                ws[dst + i * pack_factor + j] = weight


def _next_w(context, loaders, mutation):
    """QuantizedBlockLoader::next across ``BK`` weight rows."""
    shape, n = context["geometry"], context["N"]
    tile_stride = BK * n * shape["bytesPerPack"] // shape["packFactor"]
    group_stride = 1 if mutation == "scale-stride" else BK * n // context["groupSize"]
    for loader in loaders:
        loader["src"] += tile_stride
        if mutation != "scale-step":
            loader["scales"] += group_stride


def _block(context, state, loaders, accumulator, x, bounds, offset, mutation):
    """One block: both loads, then BlockMMA::mma."""
    rows, k_rows, x_safe, first_column = bounds
    tiles = context["tiles"]
    affine._load_x(context, x, rows, k_rows, offset, tiles, x_safe)
    _load_w(context, state, loaders, k_rows, first_column, tiles, mutation)
    base._mma(context, accumulator, mutation)


def _tile(state, context, bases, tid_x, tid_y, mutation):
    """fp_qmm_n_impl for one threadgroup."""
    x_base, w_base, s_base, _, y_base = bases
    shape = geometry(context["bits"], context["groupSize"])
    # The fp inputs are exact in float32 but not multiples of 2^-5.
    context = {**context, "mode": state["mode"], "geometry": shape, "check": _checked}
    k, n, group_size = context["K"], context["N"], context["groupSize"]
    y_row = 0 if mutation == "rows" else tid_y * BM
    y_col = 0 if mutation == "columns" else tid_x * BN
    x = x_base + y_row * k
    wl = w_base * 4 + y_col * shape["bytesPerPack"] // shape["packFactor"]
    scales = s_base + (0 if mutation == "scale-columns" else y_col // group_size)
    y = y_base + y_row * n + y_col
    rows = min(BM, context["M"] - y_row)
    columns = min(BN, n - y_col) if shape["partialN"] else BN
    x_safe = rows < BM
    blocks, tail = divmod(k, BK)
    context["paths"].add(
        f"rows-{'partial' if x_safe else 'whole'}/k-{'tail' if tail else 'whole'}"
        f"/n-{'partial' if columns < BN else 'whole'}"
    )
    loaders = _loaders(context, wl, scales, mutation)
    accumulator = [[0.0] * BN for _ in range(BM)]
    for block in range(blocks):
        offset = 0 if mutation == "k-step" else block * BK
        bounds = (rows, BK, x_safe, y_col)
        _block(context, state, loaders, accumulator, x, bounds, offset, mutation)
        _next_w(context, loaders, mutation)
    if tail and mutation != "k-tail":
        context["paths"].add("tail")
        offset = 0 if mutation == "k-step" else blocks * BK
        bounds = (rows, tail, True, y_col)
        if mutation == "tail-mask":
            bounds = (rows, BK, x_safe, y_col)
        _block(context, state, loaders, accumulator, x, bounds, offset, mutation)
    base._store(context, accumulator, y, rows, columns, mutation)


def simulate(info, data, families, mutation=None):
    """The kernel tile by tile; returns outputs, stores, reads and paths.

    ``overreads`` holds the weight words and scales the loads past ``N``
    read, which the in-row ``reads`` leave out.
    """
    storage = {name: values for name, (_, values) in data["values"].items()}
    extra = {"overreads": {"w": set(), "scales": set()}, "strides": set()}
    state = {"storage": storage, "extra": extra, "mode": info["mode"]}

    def tile(context, bases, tid_x, tid_y, mutation):
        _tile(state, context, bases, tid_x, tid_y, mutation)

    run = base.simulate(
        info,
        data,
        families,
        mutation,
        tile=tile,
        transpose_b=False,
        offsets=fp._offsets(info, extra),
    )
    if "strides" in run["records"]:
        run["records"]["strides"] = set(extra["strides"])
    run["overreads"] = extra["overreads"]
    return run


def expected_paths(m, n, k, group_size):
    """The loop and store paths an ``M`` by ``N`` by ``K`` product reaches."""
    tail = "tail" if k % BK else "whole"
    paths = {"tail"} if k % BK else set()
    for row in range(0, m, BM):
        rows = min(BM, m - row)
        for column in range(0, n, BN):
            columns = min(BN, n - column) if group_size % BN else BN
            paths.add(
                f"rows-{'partial' if rows < BM else 'whole'}/k-{tail}"
                f"/n-{'partial' if columns < BN else 'whole'}"
            )
            whole = rows == BM and columns == BN
            paths.add("store-whole" if whole else "store-safe")
    return paths
