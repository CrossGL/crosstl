"""Tile-level model of the pinned MLX ``fp_gather_qmm_rhs``.

The helpers follow ``fp_quantized.h``'s ``fp_gather_qmm_rhs`` and
``QuantizedBlockLoader``, ``utils.h``'s ``schedule_row_tile``,
``quantized_utils.h``'s ``gemm_loop_aligned``, ``gemm_loop_unaligned`` and
``gemm_loop_finalize``, and ``steel/gemm/loader.h``'s ``BlockLoader`` at the
pinned commit, for ``instantiate_quantized_all_rhs``'s one tile: ``BM = 16``,
``BN = BK = 32`` and a ``1 x 2`` simdgroup grid of 64 threads.

``simulate`` replays one dispatch threadgroup by threadgroup. Each ``tid.y``
first runs ``schedule_row_tile`` over ``offsets`` 32 experts at a time, as
its lanes do, to find the expert that owns its row tile, the tile's first row
and its valid rows; a threadgroup past the last tile returns. The tile then
takes the branch ``fp_gather_qmm_rhs`` takes for its rows, its columns and
``align_N``, runs the whole blocks of ``K`` through both loaders and the
float32 products, the partial block when ``align_K`` is false, and stores the
whole tile or its valid part. A weight loader bounds only its rows: past ``K``
in a transposed tail, or past ``N`` in an untransposed partial column tile, a
thread dequantizes the next row's first bytes and scale, or the guards after
the last expert; ``simulate`` records those reads apart from the in-row ones.
Global-scale entries fold the expert's ``inv_scale_enc`` into each group scale
and round every dequantized weight to the element type, as the loader does.
Named mutations show which defects the expectations reject.
"""

import math
from collections import Counter
from operator import mul

from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    GUARD_COUNT,
    _checked,
    _code_value,
    _inverse_scale,
    _scale_value,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import padded
from demos.integrations.mlx.tests.kernels.test_current_quantized import _to_type
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    SIMD_SIZE,
    _f32,
)

# instantiate_quantized_all_rhs: bm 16, bn 32, bk 32, wm 1, wn 2.
BM, BN, BK, WM, WN = 16, 32, 32, 1, 2
THREADS = WM * WN * SIMD_SIZE
# BlockLoader<T, BM, BK, BK_padded, 1, 64>: n_reads, TCOLS and TROWS.
X_READS = BM * BK // THREADS
X_COLUMNS = BK // X_READS
X_ROWS = THREADS // X_COLUMNS

MUTATIONS = {
    "order": "pairs each byte's codes with its columns in reverse",
    "sign": "decodes every code without its sign bit",
    "scale-step": "keeps the first block's scales",
    "k-step": "keeps the activation loader on the first block",
    "columns": "ignores the threadgroup's column offset",
    "transpose": "reads the weight tile the other way round",
    "store": "stores whole tiles past the valid rows and columns",
    "tail": "drops the partial block past the whole blocks of K",
    "tail-zero": "keeps the tail's activations past K",
    "group": "reads the first expert's weights for every row tile",
    "chunk": "restarts the tile count in every chunk of 32 experts",
    "round": "counts only an expert's whole row tiles",
    "row": "starts every row tile at its expert's first row",
    "rows": "takes every row tile as BM rows",
    "global-scale": "drops the global scale's factor",
    "expert-scale": "reads the first expert's global scale for every expert",
    "fold-scale": "applies the global scale to the sums, not to each weight",
}


def w_geometry(bits, group_size, transpose):
    """QuantizedBlockLoader's constants for the rhs weight tile.

    Transposed weights load a ``(BN, BK)`` tile along their reduction
    dimension, untransposed ones a ``(BK, BN)`` tile across it.
    """
    pack_factor = 8 // bits
    rows, columns = (BN, BK) if transpose else (BK, BN)
    packed = columns // pack_factor
    reads = 1 if packed * rows < THREADS else packed * rows // THREADS
    # The loader's static_asserts: one group per thread's reads.
    assert reads * pack_factor <= group_size and group_size % (reads * pack_factor) == 0
    return {
        "packFactor": pack_factor,
        "rows": rows,
        "columns": columns,
        "packedColumns": packed,
        "reads": reads,
        "groupSteps": 1 if group_size < columns else group_size // columns,
        "scaleStep": columns // group_size if group_size < columns else 1,
    }


def schedule(offsets, num_groups, m, tile, mutation=None):
    """schedule_row_tile<BM>: ``(row, group, rows)`` of ``tile``, or ``None``.

    Each pass gives one lane per expert of a chunk of 32: its rows run from
    ``offsets[g]`` to ``offsets[g + 1]``, or to ``M`` for the last expert,
    and the inclusive prefix sum of the experts' row tiles finds the lane
    that owns ``tile``.
    """
    tiles_before = 0
    for first in range(0, num_groups, SIMD_SIZE):
        lanes = []
        for lane in range(SIMD_SIZE):
            group = first + lane
            start = end = m
            if group < num_groups:
                start = offsets(group)
                end = offsets(group + 1) if group + 1 < num_groups else m
            count = end - start
            tiles = count // BM if mutation == "round" else -(-count // BM)
            lanes.append((start, end, tiles))
        through, total = [], tiles_before
        for _, _, tiles in lanes:
            total += tiles
            through.append(total)
        owner = sum(1 for value in through if value <= tile)
        if owner < SIMD_SIZE:
            start, end, tiles = lanes[owner]
            row = start
            if mutation != "row":
                row += (tile - (through[owner] - tiles)) * BM
            rows = BM if mutation == "rows" else min(BM, end - row)
            return row, first + owner, rows
        if mutation != "chunk":
            tiles_before = through[-1]
    return None


def _narrow(context, value):
    """``value`` in the element type; a mutated model's overflow is infinite."""
    try:
        return _to_type(context["dtype"], value)
    except OverflowError:
        if context["strict"]:
            raise
        return math.copysign(math.inf, value)


def _raw(context, name, index):
    """An element a weight loader reads past its row: guards allowed."""
    values = context["storage"][name]
    if not 0 <= index < len(values):
        assert not context["strict"], (name, index)
        return math.nan
    context["overreads"][name].add(index)
    return values[index]


def _byte(context, index, over):
    context["records"]["bytes"].add(("w", index))
    if over:
        word = _raw(context, "w", index // 4)
    else:
        word = context["load"]("w", index // 4)
    if word != word:
        return 0
    return (word >> (8 * (index % 4))) & 0xFF


def _weights(context, byte, scale, mutation):
    """dequantize<T, bits>: static_cast<T>(scale * Dequantize<bits, float>)."""
    bits = context["bits"]
    codes = [byte, byte >> 4] if bits == 4 else [byte]
    if mutation == "order":
        codes.reverse()
    if mutation == "sign":
        codes = [code & (0x7F if bits == 8 else 0xF7) for code in codes]
    weights = []
    for code in codes:
        weight = _narrow(context, _f32(scale * _code_value(bits, code)))
        if context["strict"]:
            assert math.isfinite(weight), (byte, scale)
        weights.append(weight)
    return weights


def _w_loaders(context, src, scales):
    """Each thread's QuantizedBlockLoader state after its constructor."""
    shape, group_size = context["wShape"], context["groupSize"]
    pack_factor, reads = shape["packFactor"], shape["reads"]
    src_ld = context["K"] if context["transpose"] else context["N"]
    loaders = []
    for thread in range(THREADS):
        bi = reads * thread // shape["packedColumns"]
        bj = reads * thread % shape["packedColumns"]
        loaders.append(
            {
                "bi": bi,
                "bj": bj,
                "dst": bi * context["ld"] + bj * pack_factor,
                "src": src + bi * src_ld // pack_factor + bj,
                "scales": (
                    scales + bi * src_ld // group_size + bj * pack_factor // group_size
                ),
                "count": 0,
            }
        )
    return loaders


def _load_w(context, tile, loaders, bound_rows, block, mutation):
    """load_unsafe, or load_safe that zeroes the rows at or past ``bound_rows``."""
    shape, ws, slots = context["wShape"], context["Ws"], context["tiles"]["Ws"]
    pack_factor, reads = shape["packFactor"], shape["reads"]
    for loader in loaders:
        dst = loader["dst"]
        if loader["bi"] >= bound_rows:
            for i in range(reads * pack_factor):
                slots.add(dst + i)
                ws[dst + i] = 0.0
            continue
        # A thread's codes lie in one group, wholly inside or past its row.
        if context["transpose"]:
            over = block * BK + loader["bj"] * pack_factor >= context["K"]
        else:
            over = tile["column"] + loader["bj"] * pack_factor >= context["N"]
        index = loader["scales"]
        context["records"]["bytes"].add(("scales", index))
        byte = (
            _raw(context, "scales", index) if over else context["load"]("scales", index)
        )
        if byte != byte:
            byte = 0
        # float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc.
        value = _scale_value(context["mode"], byte)
        narrowed = _narrow(context, value)
        if context["strict"] and not over:
            assert narrowed == value, byte
        scale = _f32(narrowed * tile["inverse"])
        for i in range(reads):
            weights = _weights(
                context, _byte(context, loader["src"] + i, over), scale, mutation
            )
            for j, weight in enumerate(weights):
                slots.add(dst + i * pack_factor + j)
                ws[dst + i * pack_factor + j] = weight


def _next_w(context, loaders, mutation):
    """QuantizedBlockLoader::next, along or across the reduction dimension."""
    shape, group_size = context["wShape"], context["groupSize"]
    if context["transpose"]:
        stride, scale_stride = shape["packedColumns"], shape["scaleStep"]
    else:
        stride = BK * context["N"] // shape["packFactor"]
        scale_stride = BK * context["N"] // group_size
    for loader in loaders:
        loader["src"] += stride
        if mutation == "scale-step":
            continue
        if context["transpose"] and shape["groupSteps"] > 1:
            loader["count"] += 1
            if loader["count"] == shape["groupSteps"]:
                loader["count"] = 0
                loader["scales"] += 1
        else:
            loader["scales"] += scale_stride


def _load_x(context, x, rows, columns, mutation):
    """BlockLoader::load_unsafe, or load_safe(short2(columns, rows))."""
    xs, ld, k = context["Xs"], context["ld"], context["K"]
    load, records = context["load"], context["records"]
    for thread in range(THREADS):
        bi, bj = thread // X_COLUMNS, X_READS * (thread % X_COLUMNS)
        source = x + bi * k + bj
        for row in range(0, BM, X_ROWS):
            for j in range(X_READS):
                slot = (bi + row) * ld + bj + j
                context["tiles"]["Xs"].add(slot)
                index = source + row * k + j
                if columns is None:
                    records["x-vector"].add(("x", index))
                    records["elements"].add(("x", index))
                    xs[slot] = load("x", index)
                    continue
                inside = row < rows - bi and j < columns - bj
                if mutation == "tail-zero" and row < rows - bi:
                    inside = True
                if inside:
                    records["elements"].add(("x", index))
                    xs[slot] = load("x", index)
                else:
                    xs[slot] = 0.0


def _mma(context, accumulator, mutation):
    """BlockMMA::mma: float32 products of the Xs and Ws tiles.

    The transposed kernel's BlockMMA reads Ws as (BN, BK), the other as
    (BK, BN).
    """
    ld, xs, ws = context["ld"], context["Xs"], context["Ws"]
    x_rows = [xs[row * ld : row * ld + BK] for row in range(BM)]
    if context["transpose"] != (mutation == "transpose"):
        w_columns = [ws[column * ld : column * ld + BK] for column in range(BN)]
    else:
        w_columns = [[ws[k * ld + column] for k in range(BK)] for column in range(BN)]
    check = _checked if context["strict"] else _f32
    for row, values in enumerate(x_rows):
        line = accumulator[row]
        for column, weights in enumerate(w_columns):
            # Exact in float32 in any order: every product is a multiple of
            # 2^-f and their magnitudes sum below 2^(24 - f).
            line[column] = check(line[column] + sum(map(mul, values, weights)))


def _store(context, accumulator, y, rows, columns, scale, mutation):
    """store_result, or store_result_safe(y, N, short2(columns, rows))."""
    outputs, n = context["outputs"]["y"], context["N"]
    total = len(outputs) - GUARD_COUNT
    for row in range(BM):
        for column in range(BN):
            index = y + row * n + column
            value = accumulator[row][column]
            if scale != 1.0:
                value = _f32(value * scale)
            if row < rows and column < columns:
                if not 0 <= index < total:
                    # Only a mutated schedule stores past the matrix: into
                    # the guards, or past the buffer.
                    assert not context["strict"], index
                    if 0 <= index < len(outputs):
                        outputs[index] = _narrow(context, value)
                    continue
                context["stores"]["y", index] += 1
                context["records"]["y"].add(("y", index))
                outputs[index] = _narrow(context, value)
            elif mutation == "store" and total <= index < len(outputs):
                # Writes inside the matrix land on another tile's outputs in
                # an unspecified order; past it they reach the guards.
                outputs[index] = _narrow(context, value)


def _tile(context, owner, tid_x, mutation):
    """fp_gather_qmm_rhs for one scheduled threadgroup."""
    y_row, group, tgp_bm = owner
    k, n, transpose = context["K"], context["N"], context["transpose"]
    pack_factor, group_size = context["wShape"]["packFactor"], context["groupSize"]
    y_col = 0 if mutation == "columns" else tid_x * BN
    align_n, align_k = context["alignN"], context["alignK"]
    tgp_bn = BN if align_n else min(BN, n - y_col)
    k_blocks = k // BK
    k_remain = k - k_blocks * BK
    expert = 0 if mutation == "group" else group
    k_w, n_w = k // pack_factor, n // pack_factor
    if transpose:
        wl = y_col * k_w + expert * n * k_w
        scales = y_col * (k // group_size) + expert * n * (k // group_size)
    else:
        wl = y_col // pack_factor + expert * k * n_w
        scales = y_col // group_size + expert * k * (n // group_size)
    inverse = 1.0
    if context["globalScale"]:
        # The loader's constructor reads global_scale + group.
        index = 0 if mutation == "expert-scale" else group
        context["records"][context["globalScaleFamily"]].add(("global_scale", index))
        inverse = _inverse_scale(context["load"]("global_scale", index))
    folded = inverse if mutation == "fold-scale" else 1.0
    if mutation in ("global-scale", "fold-scale"):
        inverse = 1.0
    tile = {"column": y_col, "inverse": inverse}
    loaders = _w_loaders(context, wl, scales)
    rows_whole = tgp_bm == BM
    columns_whole = align_n or tgp_bn == BN
    if rows_whole and columns_whole:
        branch = "aligned"
    elif columns_whole:
        branch = "rows"
    elif rows_whole:
        branch = "columns"
    else:
        branch = "both"
    context["paths"].add(branch)
    # load_safe's bounds: the x loader's (BK, tgp_bm), the weight loader's
    # (BK, tgp_bn) rows when transposed, else all BK of its rows.
    x_rows = None if rows_whole else tgp_bm
    w_rows = context["wShape"]["rows"]
    if not columns_whole:
        w_rows = tgp_bn if transpose else BK
    x_base = y_row * k
    accumulator = [[0.0] * BN for _ in range(BM)]
    for block in range(k_blocks):
        offset = 0 if mutation == "k-step" else block * BK
        _load_x(
            context, x_base + offset, x_rows, None if x_rows is None else BK, mutation
        )
        _load_w(context, tile, loaders, w_rows, block, mutation)
        _mma(context, accumulator, mutation)
        _next_w(context, loaders, mutation)
    if not align_k and mutation != "tail":
        context["paths"].add("finalize")
        offset = 0 if mutation == "k-step" else k_blocks * BK
        _load_x(context, x_base + offset, tgp_bm, k_remain, mutation)
        # tile_w: (k_remain, tgp_bn) transposed, (tgp_bn, k_remain) otherwise.
        _load_w(
            context,
            tile,
            loaders,
            tgp_bn if transpose else k_remain,
            k_blocks,
            mutation,
        )
        _mma(context, accumulator, mutation)
    store_rows, store_columns = (BM, BN) if branch == "aligned" else (tgp_bm, tgp_bn)
    context["paths"].add("store-whole" if branch == "aligned" else "store-safe")
    _store(
        context,
        accumulator,
        y_row * n + y_col,
        store_rows,
        store_columns,
        folded,
        mutation,
    )


def simulate(info, constants, data, mutation=None):
    """The kernel threadgroup by threadgroup; outputs, stores, reads, paths.

    ``constants`` holds the frozen ``align_N`` and ``align_K``. ``schedule``
    gives each ``tid.y``'s owner or ``None``; ``overreads`` holds the weight
    words and scales read past a row, which the in-row ``reads`` leave out.
    """
    storage = {name: values for name, (_, values) in data["values"].items()}
    reads = {name: set() for name in storage}
    strict = mutation is None
    families = ("x-vector", "elements", "y", "bytes", "offsets", "global-scales")
    records = {family: set() for family in families}

    def load(name, index):
        if strict:
            assert 0 <= index < len(storage[name]) - GUARD_COUNT, (name, index)
        elif not 0 <= index < len(storage[name]):
            return math.nan
        reads[name].add(index)
        return storage[name][index]

    ld = padded(info["itype"])
    context = {
        "load": load,
        "storage": storage,
        "strict": strict,
        "dtype": info["dtype"],
        "mode": info["mode"],
        "bits": info["bits"],
        "groupSize": info["groupSize"],
        "transpose": info["transpose"],
        "globalScale": info["globalScale"],
        "globalScaleFamily": (
            "elements" if info["dtype"] == "float32" else "global-scales"
        ),
        "alignN": constants["align_N"],
        "alignK": constants["align_K"],
        "wShape": w_geometry(info["bits"], info["groupSize"], info["transpose"]),
        "ld": ld,
        "K": storage["K"][0],
        "N": storage["N"][0],
        "Xs": [None] * (BM * ld),
        "Ws": [None] * (BN * ld),
        "outputs": {"y": list(data["initial"]["y"])},
        "stores": Counter(),
        "paths": set(),
        "tiles": {"Xs": set(), "Ws": set()},
        "records": records,
        "overreads": {"w": set(), "scales": set()},
    }
    m, num_groups = storage["M"][0], storage["num_groups"][0]
    owners = []
    grid_x, grid_y, grid_z = data["grid"]
    assert grid_z == 1
    for tid_y in range(grid_y):
        owner = schedule(
            lambda group: load("offsets", group), num_groups, m, tid_y, mutation
        )
        owners.append(owner)
        if owner is None:
            context["paths"].add("idle")
            continue
        for tid_x in range(grid_x):
            _tile(context, owner, tid_x, mutation)
    records["offsets"] = {("offsets", index) for index in reads["offsets"]}
    records["Xs"] = {("Xs", slot) for slot in context["tiles"]["Xs"]}
    records["Ws"] = {("Ws", slot) for slot in context["tiles"]["Ws"]}
    return {
        "outputs": context["outputs"],
        "stores": context["stores"],
        "reads": reads,
        "overreads": context["overreads"],
        "paths": context["paths"],
        "records": records,
        "schedule": owners,
    }
