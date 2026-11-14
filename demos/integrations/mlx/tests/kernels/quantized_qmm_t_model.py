"""Tile-level model of the pinned MLX ``qmm_t_impl``.

The helpers follow ``quantized.h``'s ``QuantizedBlockLoader`` and
``qmm_t_impl`` and ``steel/gemm/loader.h`` and ``mma.h`` at the pinned commit.
``simulate`` replays one dispatch threadgroup by threadgroup: both loaders
per block of 32 codes, BlockMMA's float32 fragment products and the whole or
safe stores. It records every buffer element, weight byte, batch stride and
tile slot it touches, so that the native gate can justify the ranges it
asserts to the translator, and applies named mutations to show which defects
the expectations reject.
"""

import math
from collections import Counter
from operator import mul

from demos.integrations.mlx.tests.kernels.test_current_quantized import _to_type
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import (
    GUARD_COUNT,
    SIMD_SIZE,
    _adjust,
    _checked,
    _f32,
)

# qmm_t_impl's tile and BlockMMA's simdgroup grid.
BM = BN = BK = 32
WM = WN = 2
FRAGMENT = 8
THREADS = WM * WN * SIMD_SIZE


def padded(itype):
    """qmm_t_impl's BK_padded: BK plus 16 bytes of elements."""
    return BK + 16 // (4 if itype == "float" else 2)


def geometry(bits):
    """QuantizedBlockLoader's constants for one width."""
    # get_pack_factor<bits, 8>() and get_bytes_per_pack<bits>().
    pack_factor = 8 if bits in (3, 5) else 4 if bits == 6 else 8 // bits
    power_of_2_bits = bits & (bits - 1) == 0
    bytes_per_pack = 1 if power_of_2_bits else 5 if bits == 5 else 3
    packed = BK // pack_factor
    n_reads = 1 if packed * BN < THREADS else packed * BN // THREADS
    return {
        "packFactor": pack_factor,
        "bytesPerPack": bytes_per_pack,
        "packedColumns": packed,
        "reads": n_reads,
    }


MUTATIONS = {
    "bias": "drops the bias term",
    "order": "pairs each pack's codes with its columns in reverse",
    "scale-step": "keeps the first block's scales and biases",
    "group-steps": "steps the scales and biases every block",
    "k-step": "keeps the activation loader on the first block",
    "rows": "ignores the threadgroup's row offset",
    "columns": "ignores the threadgroup's column offset",
    "transpose": "reads the weight tile as (BK, BN)",
    "store": "stores whole tiles past M and N",
    "batch": "ignores the weight batch offsets",
}


def fragment_cells():
    """``(simdgroup, lane, row, column)`` of every accumulator element.

    BlockMMA places simdgroup ``s`` at ``(8 * (s / WN), 8 * (s % WN))`` and
    each lane at BaseMMAFrag::get_coord; TM = TN = 2 fragments step by
    ``8 * WM`` rows and ``8 * WN`` columns, two columns per lane.
    """
    cells = []
    for group in range(WM * WN):
        tm, tn = FRAGMENT * (group // WN), FRAGMENT * (group % WN)
        for lane in range(SIMD_SIZE):
            qid = lane // 4
            sm = (qid & 4) + (lane // 2) % 4
            sn = (qid & 2) * 2 + (lane % 2) * 2
            for i in range(BM // (FRAGMENT * WM)):
                for j in range(BN // (FRAGMENT * WN)):
                    for element in range(2):
                        row = tm + sm + i * FRAGMENT * WM
                        column = tn + sn + j * FRAGMENT * WN + element
                        cells.append((group, lane, row, column))
    return cells


def _load_x(context, x, rows, k_offset, tiles):
    """BlockLoader::load_safe or load_unsafe of one activation block."""
    load, ld, xs = context["load"], context["ld"], context["Xs"]
    reads = BK * BM // THREADS
    for thread in range(THREADS):
        bi, bj = thread // (BK // reads), reads * (thread % (BK // reads))
        source = x + bi * context["K"] + bj + k_offset
        for j in range(reads):
            slot = bi * ld + bj + j
            tiles["Xs"].add(slot)
            if bi < rows:
                index = source + j
                context["records"]["x-vector" if rows == BM else "elements"].add(
                    ("x", index)
                )
                xs[slot] = load("x", index)
            else:
                xs[slot] = 0.0


def _dequantize(context, byte, scale, bias, mutation):
    """dequantize<T, pack_factor, bits> of one pack."""
    bits, geometry = context["bits"], context["geometry"]
    value = 0
    for index in range(geometry["bytesPerPack"]):
        value |= context["byte"](byte + index) << (8 * index)
    mask = (1 << bits) - 1
    codes = [(value >> (bits * i)) & mask for i in range(geometry["packFactor"])]
    if mutation == "order":
        codes.reverse()
    if mutation == "bias":
        bias = 0.0
    weights = [_f32(scale * code + bias) for code in codes]
    if context["strict"]:
        for weight in weights:
            # Every dequantized weight is exact in T.
            assert _to_type(context["dtype"], weight) == weight, weight
    return weights


def _load_w(context, loaders, columns, tiles, mutation):
    """QuantizedBlockLoader::load_safe or load_unsafe of one weight block."""
    geometry, ld, ws = context["geometry"], context["ld"], context["Ws"]
    pack_factor, reads = geometry["packFactor"], geometry["reads"]
    for loader in loaders:
        bi = loader["bi"]
        dst = bi * ld + loader["bj"] * pack_factor
        if bi >= columns:
            for i in range(reads * pack_factor):
                tiles["Ws"].add(dst + i)
                ws[dst + i] = 0.0
            continue
        scale = context["load"]("scales", loader["scales"])
        bias = context["load"]("biases", loader["biases"])
        context["records"]["elements"].update(
            (("scales", loader["scales"]), ("biases", loader["biases"]))
        )
        for i in range(reads):
            byte = loader["src"] + i * geometry["bytesPerPack"]
            weights = _dequantize(context, byte, scale, bias, mutation)
            for j, weight in enumerate(weights):
                tiles["Ws"].add(dst + i * pack_factor + j)
                ws[dst + i * pack_factor + j] = weight


def _next_w(context, loaders, mutation):
    """QuantizedBlockLoader::next along the reduction dimension."""
    geometry = context["geometry"]
    group_steps = context["groupSize"] // BK
    for loader in loaders:
        loader["src"] += geometry["packedColumns"] * geometry["bytesPerPack"]
        if mutation == "scale-step":
            continue
        if group_steps > 1 and mutation != "group-steps":
            loader["count"] += 1
            if loader["count"] != group_steps:
                continue
            loader["count"] = 0
        loader["scales"] += 1
        loader["biases"] += 1


def _w_loaders(context, wl, scales, biases):
    """Each thread's QuantizedBlockLoader state after its constructor."""
    geometry = context["geometry"]
    reads, packed = geometry["reads"], geometry["packedColumns"]
    k_w = context["K"] * geometry["bytesPerPack"] // geometry["packFactor"]
    k_g = context["K"] // context["groupSize"]
    loaders = []
    for thread in range(THREADS):
        bi, bj = reads * thread // packed, reads * thread % packed
        if packed * BN < THREADS and bi >= BN:
            continue
        loaders.append(
            {
                "bi": bi,
                "bj": bj,
                "src": wl + bi * k_w + bj * geometry["bytesPerPack"],
                "scales": scales + bi * k_g,
                "biases": biases + bi * k_g,
                "count": 0,
            }
        )
    return loaders


def _mma(context, accumulator, mutation):
    """BlockMMA::mma: float32 fragment products of the Xs and Ws tiles.

    ``transpose_b`` BlockMMA reads Ws as (BN, BK), otherwise as (BK, BN).
    """
    ld, xs, ws = context["ld"], context["Xs"], context["Ws"]
    x_rows = [xs[row * ld : row * ld + BK] for row in range(BM)]
    if context["transposeB"] == (mutation == "transpose"):
        w_columns = [[ws[k * ld + column] for k in range(BK)] for column in range(BN)]
    else:
        w_columns = [ws[column * ld : column * ld + BK] for column in range(BN)]
    check = _checked if context["strict"] else _f32
    for row, values in enumerate(x_rows):
        line = accumulator[row]
        for column, weights in enumerate(w_columns):
            # Exact in float32 in any order: the products and partial sums
            # are multiples of 2^-5 below the case's budget.
            line[column] = check(line[column] + sum(map(mul, values, weights)))


def _store(context, accumulator, y, rows, columns, mutation):
    """store_result or store_result_safe of one tile."""
    outputs, n, cast = context["outputs"]["y"], context["N"], context["cast"]
    total = len(outputs) - GUARD_COUNT
    whole = rows == BM and columns == BN
    context["paths"].add("store-whole" if whole else "store-safe")
    for row in range(BM):
        for column in range(BN):
            index = y + row * n + column
            if row < rows and column < columns:
                if not context["strict"] and not 0 <= index < len(outputs):
                    continue
                assert 0 <= index < total, index
                context["stores"]["y", index] += 1
                context["records"]["y"].add(("y", index))
                outputs[index] = cast(accumulator[row][column])
            elif mutation == "store" and total <= index < len(outputs):
                # Writes inside the matrix land on another tile's outputs in
                # an unspecified order; past it they reach the guards.
                outputs[index] = cast(accumulator[row][column])


def _tile(context, bases, tid_x, tid_y, mutation):
    """qmm_t_impl for one threadgroup."""
    x_base, w_base, s_base, b_base, y_base = bases
    geometry, k = context["geometry"], context["K"]
    k_w = k * geometry["bytesPerPack"] // geometry["packFactor"]
    k_g = k // context["groupSize"]
    y_row = 0 if mutation == "rows" else tid_y * BM
    y_col = 0 if mutation == "columns" else tid_x * BN
    x = x_base + y_row * k
    wl = w_base * 4 + y_col * k_w
    scales, biases = s_base + y_col * k_g, b_base + y_col * k_g
    y = y_base + y_row * context["N"] + y_col
    rows = min(BM, context["M"] - y_row)
    columns = min(BN, context["N"] - y_col)
    x_safe = rows < BM
    w_safe = not context["aligned"] and columns < BN
    context["paths"].add(
        f"x-{'safe' if x_safe else 'unsafe'}/w-{'safe' if w_safe else 'unsafe'}"
    )
    loaders = _w_loaders(context, wl, scales, biases)
    accumulator = [[0.0] * BN for _ in range(BM)]
    tiles = context["tiles"]
    for block in range(0, context["Keff"], BK):
        offset = 0 if mutation == "k-step" else block
        _load_x(context, x, rows if x_safe else BM, offset, tiles)
        _load_w(context, loaders, columns if w_safe else BN, tiles, mutation)
        _mma(context, accumulator, mutation)
        _next_w(context, loaders, mutation)
    _store(context, accumulator, y, rows, columns, mutation)


def _batch_bases(variant, storage, z, mutation, records):
    """adjust_matrix_offsets' x, w word, scale, bias and y offsets for ``tid.z``.

    ``gather`` first reads ``tid.z``'s lhs and rhs indices; the ``indices``
    mutation takes ``tid.z`` itself, as the batched kernels do.
    """
    view = {**storage, "out_vec_size": storage["N"]}
    if variant == "gather" and mutation == "indices":
        variant = "batch_1"
    offsets = []
    bases = _adjust(variant, view, z, mutation, offsets)
    if "indices" in records:
        records["indices"].update(
            (buffer, index)
            for _, buffer, index in offsets
            if buffer in ("lhs_indices", "rhs_indices")
        )
    # adjust_matrix_offsets reads each batch dimension's stride.
    strides = [("x_strides", "x_batch_ndims")] + [
        (name, "w_batch_ndims") for name in ("w_strides", "s_strides", "b_strides")
    ]
    if variant == "gather":
        strides += [(name, "batch_ndims") for name in ("lhs_strides", "rhs_strides")]
    for name, ndims in strides:
        records["strides"].update((name, axis) for axis in range(storage[ndims][0]))
    return bases


def simulate(
    info, data, families, mutation=None, tile=None, transpose_b=True, offsets=None
):
    """The kernel tile by tile; returns outputs, stores, reads and paths.

    ``tile`` replays one threadgroup, qmm_t_impl's by default, and
    ``transpose_b`` is its BlockMMA's. ``offsets(storage, z, mutation)``
    gives ``tid.z``'s x, w word, scale and bias, and y offsets in place of
    adjust_matrix_offsets. A bound ``k_partition_size`` is qmm_t_impl's
    ``K_eff``, otherwise ``K`` is.
    """
    tile = tile or _tile
    storage = {name: values for name, (_, values) in data["values"].items()}
    reads = {name: set() for name in storage}
    records = {family: set() for family in families}
    # Mutations may read guards, or past a buffer, which gives no value.
    strict = mutation is None

    def load(name, index):
        if strict:
            assert 0 <= index < len(storage[name]) - GUARD_COUNT, (name, index)
        elif not 0 <= index < len(storage[name]):
            return math.nan
        reads[name].add(index)
        return storage[name][index]

    def byte(index):
        records["w-bytes"].add(("w", index))
        word = load("w", index // 4)
        if word != word:
            return 0
        return (word >> (8 * (index % 4))) & 0xFF

    def cast(value):
        try:
            return _to_type(info["dtype"], value)
        except OverflowError:
            if strict:
                raise
            return math.inf

    ld = padded(info["itype"])
    context = {
        "load": load,
        "byte": byte,
        "cast": cast,
        "strict": strict,
        "dtype": info["dtype"],
        "bits": info["bits"],
        "groupSize": info["groupSize"],
        "aligned": info.get("aligned", True),
        "transposeB": transpose_b,
        "geometry": geometry(info["bits"]),
        "ld": ld,
        "K": storage["K"][0],
        "Keff": storage.get("k_partition_size", storage["K"])[0],
        "N": storage["N"][0],
        "M": storage["M"][0],
        "Xs": [None] * (BM * ld),
        "Ws": [None] * (BN * ld),
        "outputs": {"y": list(data["initial"]["y"])},
        "stores": Counter(),
        "paths": set(),
        "tiles": {"Xs": set(), "Ws": set()},
        "records": records,
    }
    grid_x, grid_y, grid_z = data["grid"]
    for z in range(grid_z):
        bases = (0, 0, 0, 0, 0)
        if offsets is not None:
            bases = offsets(storage, z, mutation)
        elif info["variant"] in ("batch_1", "gather"):
            bases = _batch_bases(info["variant"], storage, z, mutation, records)
        for tid_y in range(grid_y):
            for tid_x in range(grid_x):
                tile(context, bases, tid_x, tid_y, mutation)
    records["Xs"] = {("Xs", slot) for slot in context["tiles"]["Xs"]}
    records["Ws"] = {("Ws", slot) for slot in context["tiles"]["Ws"]}
    return {
        "outputs": context["outputs"],
        "stores": context["stores"],
        "reads": reads,
        "paths": context["paths"],
        "records": records,
    }


def expected_paths(aligned, m, n):
    """The loop and store paths an ``M`` by ``N`` output reaches, from its tiles."""
    paths = set()
    for row in range(0, m, BM):
        for column in range(0, n, BN):
            rows, columns = min(BM, m - row), min(BN, n - column)
            w_safe = not aligned and columns < BN
            paths.add(
                f"x-{'safe' if rows < BM else 'unsafe'}"
                f"/w-{'safe' if w_safe else 'unsafe'}"
            )
            paths.add("store-whole" if (rows, columns) == (BM, BN) else "store-safe")
    return paths
