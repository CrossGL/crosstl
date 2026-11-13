"""Tile-level model of the pinned MLX ``fp_qmm_t_impl``.

The helpers follow ``fp_quantized.h``'s ``QuantizedBlockLoader`` and
``fp_qmm_t_impl`` at the pinned commit; the activation ``BlockLoader``,
``BlockMMA``'s float32 fragment products and the whole or safe stores are
``quantized_qmm_t_model``'s. ``simulate`` replays one dispatch threadgroup by
threadgroup: per block of 32 codes both loaders, the fragment products and the
scale steps, then, when the group size leaves ``K`` a partial block, the
``num_k`` tail that zeroes the activation columns past ``K``. The tail's
weight loader bounds only rows, so a thread whose codes start at or past ``K``
dequantizes the next row's first bytes and scale, or the guards after the
last row; ``simulate`` records those reads apart from the in-row ones and,
unmutated, requires the codes to decode finite, so the zeroed activations
cancel them. Global-scale entries fold ``inv_scale_enc`` into each group scale
and round every dequantized weight to the element type, as the loader does.
Named mutations show which defects the expectations reject.
"""

import functools
import math

from demos.integrations.mlx.tests.kernels import quantized_qmm_t_model as base
from demos.integrations.mlx.tests.kernels.fp_qmv_model import (
    _adjust,
    _checked,
    _code_value,
    _inverse_scale,
    _scale_value,
)
from demos.integrations.mlx.tests.kernels.test_current_quantized import _to_type
from demos.integrations.mlx.tests.kernels.test_current_quantized_qmv import _f32

BM, BN, BK, THREADS = base.BM, base.BN, base.BK, base.THREADS


def geometry(bits, group_size):
    """QuantizedBlockLoader's constants and fp_qmm_t_impl's partial_k."""
    # get_pack_factor<8, bits>() is 8 / bits and get_bytes_per_pack() one
    # byte, which quantized.h's affine geometry gives these widths too.
    shared = base.geometry(bits)
    assert shared["packFactor"] == 8 // bits and shared["bytesPerPack"] == 1
    return {
        **shared,
        "groupSteps": 1 if group_size < BK else group_size // BK,
        "scaleStep": BK // group_size if group_size < BK else 1,
        "partialK": group_size % BK != 0,
    }


MUTATIONS = {
    "order": "pairs each byte's codes with its columns in reverse",
    "sign": "decodes every code without its sign bit",
    "scale-split": "gives every thread its row's first group in the block",
    "scale-step": "keeps the first block's scales",
    "k-step": "keeps the activation loader on the first block",
    "rows": "ignores the threadgroup's row offset",
    "columns": "ignores the threadgroup's column offset",
    "transpose": "reads the weight tile as (BK, BN)",
    "store": "stores whole tiles past M and N",
    "batch": "ignores the weight batch offsets",
    "tail": "drops the partial block past the whole blocks of K",
    "tail-zero": "keeps the tail's activations past K",
    "global-scale": "drops the global scale's factor",
    "fold-scale": "applies the global scale to the sums, not to each weight",
}


def _read(context, storage, extra, name, index, over):
    """One element of ``name``: in-row through the checked loader, else raw."""
    if not over:
        return context["load"](name, index)
    values = storage[name]
    if not 0 <= index < len(values):
        assert not context["strict"], (name, index)
        return math.nan
    extra["overreads"][name].add(index)
    return values[index]


def _byte(context, storage, extra, index, over):
    word = _read(context, storage, extra, "w", index // 4, over)
    if "bytes" in context["records"]:
        context["records"]["bytes"].add(("w", index))
    if word != word:
        return 0
    return (word >> (8 * (index % 4))) & 0xFF


def _loaders(context, wl, scales, mutation):
    """Each thread's QuantizedBlockLoader state after its constructor."""
    shape = context["geometry"]
    reads, packed = shape["reads"], shape["packedColumns"]
    pack_factor, group_size = shape["packFactor"], context["groupSize"]
    k_w = context["K"] * shape["bytesPerPack"] // pack_factor
    k_g = context["K"] // group_size
    loaders = []
    for thread in range(THREADS):
        bi, bj = reads * thread // packed, reads * thread % packed
        # BCOLS_PACKED * BROWS reaches the threadgroup size at both widths.
        assert packed * BN >= THREADS
        group = 0 if mutation == "scale-split" else bj * pack_factor // group_size
        loaders.append(
            {
                "bi": bi,
                "bj": bj,
                "src": wl + bi * k_w + bj * shape["bytesPerPack"],
                "scales": scales + bi * k_g + group,
                "count": 0,
            }
        )
    return loaders


def _weights(context, byte, scale, mutation, exact):
    """dequantize<T, bits>: static_cast<T>(scale * Dequantize<bits, float>)."""
    bits = context["bits"]
    codes = [byte, byte >> 4] if bits == 4 else [byte]
    if mutation == "order":
        codes.reverse()
    if mutation == "sign":
        # The sign bit of the E4M3 byte or of the E2M1 nibble.
        codes = [code & (0x7F if bits == 8 else 0xF7) for code in codes]
    weights = []
    for code in codes:
        value = _f32(scale * _code_value(bits, code))
        weight = _to_type(context["dtype"], value)
        if context["strict"]:
            assert math.isfinite(weight), (byte, scale)
            if exact:
                assert weight == value, (byte, scale)
        weights.append(weight)
    return weights


def _load_w(context, state, loaders, columns, first_column, tiles, mutation):
    """load_safe or load_unsafe of the block whose codes start at ``first_column``."""
    storage, extra, inverse = state["storage"], state["extra"], state["inverse"]
    shape, ld, ws = context["geometry"], context["ld"], context["Ws"]
    pack_factor, reads = shape["packFactor"], shape["reads"]
    for loader in loaders:
        bi = loader["bi"]
        dst = bi * ld + loader["bj"] * pack_factor
        if bi >= columns:
            for i in range(reads * pack_factor):
                tiles["Ws"].add(dst + i)
                ws[dst + i] = 0.0
            continue
        # A thread's codes lie in one group, wholly before or wholly past K.
        over = first_column + loader["bj"] * pack_factor >= context["K"]
        scale_byte = _read(context, storage, extra, "scales", loader["scales"], over)
        if "bytes" in context["records"]:
            context["records"]["bytes"].add(("scales", loader["scales"]))
        if scale_byte != scale_byte:
            scale_byte = 0
        # float(dequantize_scale<T, group_size>(*scales)) * inv_scale_enc.
        scale = _scale_value(context["mode"], scale_byte)
        narrowed = _to_type(context["dtype"], scale)
        if context["strict"] and not over:
            assert narrowed == scale, scale_byte
        scale = _f32(narrowed * inverse)
        exact = not over and not state["globalScale"]
        for i in range(reads):
            byte = _byte(context, storage, extra, loader["src"] + i, over)
            for j, weight in enumerate(_weights(context, byte, scale, mutation, exact)):
                tiles["Ws"].add(dst + i * pack_factor + j)
                ws[dst + i * pack_factor + j] = weight


def _next_w(context, loaders, mutation):
    """QuantizedBlockLoader::next along the reduction dimension."""
    shape = context["geometry"]
    for loader in loaders:
        loader["src"] += shape["packedColumns"] * shape["bytesPerPack"]
        if mutation == "scale-step":
            continue
        if shape["groupSteps"] > 1:
            loader["count"] += 1
            if loader["count"] != shape["groupSteps"]:
                continue
            loader["count"] = 0
            loader["scales"] += 1
        else:
            loader["scales"] += shape["scaleStep"]


def _load_x_tail(context, x, rows, num_k, k_offset, tiles, mutation):
    """BlockLoader::load_safe(short2(num_k, rows)) of the partial block."""
    load, ld, xs = context["load"], context["ld"], context["Xs"]
    reads = BK * BM // THREADS
    for thread in range(THREADS):
        bi, bj = thread // (BK // reads), reads * (thread % (BK // reads))
        source = x + bi * context["K"] + bj + k_offset
        for j in range(reads):
            slot = bi * ld + bj + j
            tiles["Xs"].add(slot)
            inside = bi < rows and (bj + j < num_k or mutation == "tail-zero")
            if inside:
                context["records"]["elements"].add(("x", source + j))
                xs[slot] = load("x", source + j)
            else:
                xs[slot] = 0.0


def _tile(state, context, bases, tid_x, tid_y, mutation):
    """fp_qmm_t_impl for one threadgroup."""
    x_base, w_base, s_base, _, y_base = bases
    shape = geometry(context["bits"], context["groupSize"])
    # The fp inputs are exact in float32 but not multiples of 2^-5.
    context = {**context, "mode": state["mode"], "geometry": shape, "check": _checked}
    k = context["K"]
    k_w = k * shape["bytesPerPack"] // shape["packFactor"]
    k_g = k // context["groupSize"]
    y_row = 0 if mutation == "rows" else tid_y * BM
    y_col = 0 if mutation == "columns" else tid_x * BN
    x = x_base + y_row * k
    wl = w_base * 4 + y_col * k_w
    scales = s_base + y_col * k_g
    y = y_base + y_row * context["N"] + y_col
    rows = min(BM, context["M"] - y_row)
    columns = min(BN, context["N"] - y_col)
    x_safe = rows < BM
    w_safe = not context["aligned"] and columns < BN
    context["paths"].add(
        f"x-{'safe' if x_safe else 'unsafe'}/w-{'safe' if w_safe else 'unsafe'}"
    )
    inverse = 1.0
    if state["globalScale"]:
        # The loader's constructor: *global_scale / (F8E4M3_MAX * F4E2M1_MAX).
        inverse = _inverse_scale(context["load"]("global_scale", 0))
    folded = inverse if mutation == "fold-scale" else 1.0
    if mutation in ("global-scale", "fold-scale"):
        inverse = 1.0
    state = {**state, "inverse": inverse}
    loaders = _loaders(context, wl, scales, mutation)
    accumulator = [[0.0] * BN for _ in range(BM)]
    tiles = context["tiles"]
    k_blocks = context["Keff"] // BK
    num_k = context["Keff"] - k_blocks * BK if shape["partialK"] else 0
    if mutation == "tail":
        num_k = 0
    for block in range(k_blocks):
        offset = 0 if mutation == "k-step" else block * BK
        base._load_x(context, x, rows if x_safe else BM, offset, tiles)
        _load_w(
            context,
            state,
            loaders,
            columns if w_safe else BN,
            block * BK,
            tiles,
            mutation,
        )
        base._mma(context, accumulator, mutation)
        _next_w(context, loaders, mutation)
    if num_k > 0:
        context["paths"].add("tail")
        offset = 0 if mutation == "k-step" else k_blocks * BK
        _load_x_tail(context, x, rows, num_k, offset, tiles, mutation)
        # load_safe(short2(num_k, num_outs)) bounds the rows only.
        _load_w(context, state, loaders, columns, k_blocks * BK, tiles, mutation)
        base._mma(context, accumulator, mutation)
    if folded != 1.0:
        cast = context["cast"]
        context = {**context, "cast": lambda value: cast(_f32(value * folded))}
    base._store(context, accumulator, y, rows, columns, mutation)


def _offsets(info, extra):
    """fp adjust_matrix_offsets' x, w word, scale and y offsets for ``tid.z``."""

    def offsets(storage, z, mutation):
        if info["variant"] == "batch_0":
            return 0, 0, 0, None, 0
        view = {**storage, "out_vec_size": storage["N"]}
        x_base, w_base, s_base, _, y_base = _adjust(info, view, z, mutation, [])
        # adjust_matrix_offsets reads each batch dimension's stride.
        for name, ndims in (
            ("x_strides", "x_batch_ndims"),
            ("w_strides", "w_batch_ndims"),
            ("s_strides", "w_batch_ndims"),
        ):
            extra["strides"].update((name, axis) for axis in range(storage[ndims][0]))
        return x_base, w_base, s_base, None, y_base

    return offsets


def simulate(info, data, families, mutation=None):
    """The kernel tile by tile; returns outputs, stores, reads and paths.

    ``overreads`` holds the weight words and scales the tail reads past each
    row, which the in-row ``reads`` leave out.
    """
    storage = {name: values for name, (_, values) in data["values"].items()}
    extra = {"overreads": {"w": set(), "scales": set()}, "strides": set()}
    state = {
        "storage": storage,
        "extra": extra,
        "mode": info["mode"],
        "globalScale": info["globalScale"],
    }
    run = base.simulate(
        info,
        data,
        families,
        mutation,
        tile=functools.partial(_tile, state),
        offsets=_offsets(info, extra),
    )
    if "strides" in run["records"]:
        run["records"]["strides"] = set(extra["strides"])
    run["overreads"] = extra["overreads"]
    return run


def expected_paths(aligned, m, n, k, group_size):
    """The loop, tail and store paths an ``M`` by ``N`` by ``K`` product reaches."""
    paths = base.expected_paths(aligned, m, n)
    if group_size % BK and k % BK:
        paths.add("tail")
    return paths
