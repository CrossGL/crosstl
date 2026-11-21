"""Tile-level model of the pinned MLX ``affine_gather_qmm_rhs``.

``quantized.h``'s ``affine_gather_qmm_rhs`` shares ``fp_gather_qmm_rhs``'s
tile, ``schedule_row_tile``, activation ``BlockLoader``, BlockMMA products,
branches and stores, which ``fp_gather_qmm_rhs_model`` models. This module
follows the parts that differ at the pinned commit:

- the affine ``QuantizedBlockLoader`` dequantizes ``scale * code + bias`` in
  float32 from packs of ``bytes_per_pack`` bytes, so 3-, 5- and 6-bit codes
  span bytes, and rounds each weight to the element type;
- its group size is at least 32, a multiple of the loader's columns, so each
  thread takes one scale and bias per row and the transposed loader steps
  them once per ``group_size / BK`` blocks;
- ``biases`` takes the global scale's buffer slot, and no entry has a global
  scale.

The quantized dimension, ``K`` for ``nt`` and ``N`` for ``nn``, is a
multiple of the group size, hence of 32: the transposed kernel never runs
its partial block of ``K`` and the untransposed one never has a partial
column tile. Every weight loader thread therefore stays inside its weight
row, which ``simulate`` asserts.
"""

import math
from collections import Counter

from demos.integrations.mlx.tests.kernels import fp_gather_qmm_rhs_model as fp
from demos.integrations.mlx.tests.kernels.fp_gather_qmm_rhs_model import (
    BK,
    BM,
    BN,
    THREADS,
    WM,
    WN,
    X_COLUMNS,
    X_READS,
    X_ROWS,
    schedule,
)
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import (
    _dequantize,
    padded,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    GUARD_COUNT,
)

__all__ = (
    "BK",
    "BM",
    "BN",
    "MUTATIONS",
    "THREADS",
    "WM",
    "WN",
    "X_COLUMNS",
    "X_READS",
    "X_ROWS",
    "schedule",
    "simulate",
    "w_geometry",
)

MUTATIONS = {
    "bias": "drops the bias term",
    "order": "pairs each pack's codes with its columns in reverse",
    "scale-step": "keeps the first block's scales and biases",
    "group-steps": "steps the scales and biases every block",
    "k-step": "keeps the activation loader on the first block",
    "columns": "ignores the threadgroup's column offset",
    "transpose": "reads the weight tile the other way round",
    "store": "stores whole tiles past the valid rows and columns",
    "tail": "drops the partial block past the whole blocks of K",
    "group": "reads the first expert's weights for every row tile",
    "chunk": "restarts the tile count in every chunk of 32 experts",
    "round": "counts only an expert's whole row tiles",
    "row": "starts every row tile at its expert's first row",
    "rows": "takes every row tile as BM rows",
}


def w_geometry(bits, group_size, transpose):
    """The affine QuantizedBlockLoader's constants for the rhs weight tile.

    Transposed weights load a ``(BN, BK)`` tile along their reduction
    dimension, untransposed ones a ``(BK, BN)`` tile across it.
    """
    # get_pack_factor<bits, 8>() and get_bytes_per_pack<bits>().
    pack_factor = 8 if bits in (3, 5) else 4 if bits == 6 else 8 // bits
    bytes_per_pack = 1 if bits & (bits - 1) == 0 else 5 if bits == 5 else 3
    rows, columns = (BN, BK) if transpose else (BK, BN)
    packed = columns // pack_factor
    reads = 1 if packed * rows < THREADS else packed * rows // THREADS
    # The loader's static_asserts.
    assert columns <= group_size and group_size % columns == 0
    assert group_size % (reads * pack_factor) == 0
    return {
        "packFactor": pack_factor,
        "bytesPerPack": bytes_per_pack,
        "rows": rows,
        "columns": columns,
        "packedColumns": packed,
        "reads": reads,
        "groupSteps": group_size // columns,
    }


def _row_layout(context):
    """Bytes and groups of one weight row: ``K`` codes nt, ``N`` codes nn."""
    shape = context["wShape"]
    codes = context["K"] if context["transpose"] else context["N"]
    return (
        codes * shape["bytesPerPack"] // shape["packFactor"],
        codes // context["groupSize"],
    )


def _w_loaders(context, src, scales, line, lines):
    """Each thread's QuantizedBlockLoader state after its constructor.

    ``line`` is the weight row of the tile's first loader row and ``lines``
    the expert's weight rows; each loader keeps its own row so that
    ``_load_w`` can check it stays inside it and inside the expert.
    """
    shape = context["wShape"]
    reads, packed = shape["reads"], shape["packedColumns"]
    row_bytes, row_groups = _row_layout(context)
    loaders = []
    for thread in range(THREADS):
        bi, bj = reads * thread // packed, reads * thread % packed
        if packed * shape["rows"] < THREADS and bi >= shape["rows"]:
            continue
        loaders.append(
            {
                "bi": bi,
                "bj": bj,
                "dst": bi * context["ld"] + bj * shape["packFactor"],
                "src": src + bi * row_bytes + bj * shape["bytesPerPack"],
                "scales": scales + bi * row_groups,
                "line": line + bi,
                "lines": lines,
                "count": 0,
            }
        )
    return loaders


def _load_w(context, loaders, bound_rows, mutation):
    """load_unsafe, or load_safe that zeroes the rows at or past ``bound_rows``."""
    shape, ws, slots = context["wShape"], context["Ws"], context["tiles"]["Ws"]
    pack_factor, reads = shape["packFactor"], shape["reads"]
    bytes_per_pack = shape["bytesPerPack"]
    row_bytes, row_groups = _row_layout(context)
    for loader in loaders:
        dst = loader["dst"]
        if loader["bi"] >= bound_rows:
            for i in range(reads * pack_factor):
                slots.add(dst + i)
                ws[dst + i] = 0.0
            continue
        line, last = loader["line"], loader["src"] + reads * bytes_per_pack - 1
        if context["strict"]:
            # Every byte, scale and bias the thread reads lies in its row.
            assert line in loader["lines"], loader
            assert loader["src"] // row_bytes == line == last // row_bytes, loader
            assert loader["scales"] // row_groups == line, loader
        index = loader["scales"]
        context["records"]["elements"].update((("scales", index), ("biases", index)))
        scale = context["load"]("scales", index)
        bias = context["load"]("biases", index)
        if scale != scale or bias != bias:
            # Only a mutated model reads past the buffers.
            scale = bias = 0.0
        for i in range(reads):
            weights = _dequantize(
                context, loader["src"] + i * bytes_per_pack, scale, bias, mutation
            )
            for j, weight in enumerate(weights):
                slots.add(dst + i * pack_factor + j)
                ws[dst + i * pack_factor + j] = fp._narrow(context, weight)


def _next_w(context, loaders, mutation):
    """QuantizedBlockLoader::next, along or across the reduction dimension."""
    shape, transpose = context["wShape"], context["transpose"]
    row_bytes, row_groups = _row_layout(context)
    if transpose:
        stride = shape["packedColumns"] * shape["bytesPerPack"]
    else:
        stride, group_stride = BK * row_bytes, BK * row_groups
    for loader in loaders:
        loader["src"] += stride
        if not transpose:
            loader["line"] += BK
        if mutation == "scale-step":
            continue
        if not transpose:
            loader["scales"] += group_stride
            continue
        if shape["groupSteps"] > 1 and mutation != "group-steps":
            loader["count"] += 1
            if loader["count"] != shape["groupSteps"]:
                continue
            loader["count"] = 0
        loader["scales"] += 1


def _branch(context, tgp_bm, tgp_bn):
    rows_whole = tgp_bm == BM
    columns_whole = context["alignN"] or tgp_bn == BN
    if rows_whole and columns_whole:
        return "aligned"
    if columns_whole:
        return "rows"
    return "columns" if rows_whole else "both"


def _tile(context, owner, tid_x, mutation):
    """affine_gather_qmm_rhs for one scheduled threadgroup."""
    y_row, group, tgp_bm = owner
    k, n, transpose = context["K"], context["N"], context["transpose"]
    shape, group_size = context["wShape"], context["groupSize"]
    pack_factor, bytes_per_pack = shape["packFactor"], shape["bytesPerPack"]
    y_col = 0 if mutation == "columns" else tid_x * BN
    tgp_bn = BN if context["alignN"] else min(BN, n - y_col)
    k_blocks = k // BK
    k_remain = k - k_blocks * BK
    expert = 0 if mutation == "group" else group
    k_w, n_w = k * bytes_per_pack // pack_factor, n * bytes_per_pack // pack_factor
    k_g, n_g = k // group_size, n // group_size
    if transpose:
        wl = y_col * k_w + expert * n * k_w
        scales = y_col * k_g + expert * n * k_g
        line = expert * n + y_col
    else:
        wl = y_col * bytes_per_pack // pack_factor + expert * k * n_w
        scales = y_col // group_size + expert * k * n_g
        line = expert * k
    # The weight rows of the tile's own expert: N rows nt, K rows nn.
    rows = n if transpose else k
    lines = range(group * rows, (group + 1) * rows)
    loaders = _w_loaders(context, wl, scales, line, lines)
    branch = _branch(context, tgp_bm, tgp_bn)
    context["paths"].add(branch)
    # load_safe's bounds: the x loader's (BK, tgp_bm), the weight loader's
    # (BK, tgp_bn) rows when transposed, else all BK of its rows.
    x_rows = None if tgp_bm == BM else tgp_bm
    w_rows = shape["rows"]
    if branch in ("columns", "both"):
        w_rows = tgp_bn if transpose else BK
    x_base = y_row * k
    accumulator = [[0.0] * BN for _ in range(BM)]
    for block in range(k_blocks):
        offset = 0 if mutation == "k-step" else block * BK
        fp._load_x(
            context, x_base + offset, x_rows, None if x_rows is None else BK, mutation
        )
        _load_w(context, loaders, w_rows, mutation)
        fp._mma(context, accumulator, mutation)
        _next_w(context, loaders, mutation)
    if not context["alignK"] and mutation != "tail":
        context["paths"].add("finalize")
        offset = 0 if mutation == "k-step" else k_blocks * BK
        fp._load_x(context, x_base + offset, tgp_bm, k_remain, mutation)
        # tile_w: (k_remain, tgp_bn) transposed, (tgp_bn, k_remain) otherwise.
        _load_w(context, loaders, tgp_bn if transpose else k_remain, mutation)
        fp._mma(context, accumulator, mutation)
    store_rows, store_columns = (BM, BN) if branch == "aligned" else (tgp_bm, tgp_bn)
    context["paths"].add("store-whole" if branch == "aligned" else "store-safe")
    fp._store(
        context,
        accumulator,
        y_row * n + y_col,
        store_rows,
        store_columns,
        1.0,
        mutation,
    )


def simulate(info, constants, data, mutation=None):
    """The kernel threadgroup by threadgroup; outputs, stores, reads, paths.

    ``constants`` holds the frozen ``align_N`` and ``align_K``. ``schedule``
    gives each ``tid.y``'s owner or ``None``.
    """
    storage = {name: values for name, (_, values) in data["values"].items()}
    reads = {name: set() for name in storage}
    strict = mutation is None
    families = ("x-vector", "elements", "y", "bytes", "offsets")
    records = {family: set() for family in families}

    def load(name, index):
        if strict:
            assert 0 <= index < len(storage[name]) - GUARD_COUNT, (name, index)
        elif not 0 <= index < len(storage[name]):
            return math.nan
        reads[name].add(index)
        return storage[name][index]

    def byte(index):
        records["bytes"].add(("w", index))
        word = load("w", index // 4)
        if word != word:
            return 0
        return (word >> (8 * (index % 4))) & 0xFF

    ld = padded(info["itype"])
    shape = w_geometry(info["bits"], info["groupSize"], info["transpose"])
    context = {
        "load": load,
        "byte": byte,
        "storage": storage,
        "strict": strict,
        "dtype": info["dtype"],
        "bits": info["bits"],
        "groupSize": info["groupSize"],
        "transpose": info["transpose"],
        "alignN": constants["align_N"],
        "alignK": constants["align_K"],
        "wShape": shape,
        # quantized_qmm_t_model._dequantize reads the pack geometry here.
        "geometry": shape,
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
        "paths": context["paths"],
        "records": records,
        "schedule": owners,
    }
