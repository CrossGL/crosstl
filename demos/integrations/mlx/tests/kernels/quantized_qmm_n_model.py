"""Tile-level model of the pinned MLX ``qmm_n_impl``.

``qmm_n_impl`` multiplies activations by weights that are not transposed:
``K`` rows of ``N`` codes, quantized along ``N``. It shares ``qmm_t_impl``'s
tile, dequantization and BlockMMA fragments, which ``quantized_qmm_t_model``
models, and differs in three places this module follows from ``quantized.h``
and ``steel/gemm/loader.h`` at the pinned commit:

- its activation ``BlockLoader`` takes ``n_reads = 4``, so each thread copies
  two rows of four elements, and bounds both the rows and the reduction;
- its ``QuantizedBlockLoader`` reduces along its rows (``reduction_dim = 0``):
  each thread dequantizes eight codes of one weight row and steps by ``BK``
  rows, and the scales by ``BK`` rows of groups, per block;
- ``K`` need not be a multiple of ``BK``: after the whole blocks, a last pass
  loads only the ``K % BK`` remaining rows of both tiles.

Every column tile is whole, because ``N`` is a multiple of the group size.
"""

from dataclasses import dataclass

from demos.integrations.mlx.tests.kernels import quantized_qmm_t_model as base
from demos.integrations.mlx.tests.kernels.gemm_fused_model import Loader
from demos.integrations.mlx.tests.kernels.quantized_qmm_t_model import (
    BK,
    BM,
    BN,
    THREADS,
)

# loader_x_t's n_reads, and the threads per row and rows per pass it implies.
X_READS = 4
X_THREAD_COLUMNS = BK // X_READS
X_THREAD_ROWS = THREADS // X_THREAD_COLUMNS


@dataclass(frozen=True)
class VectorLoader(Loader):
    """A ``BlockLoader`` whose ``n_reads`` template argument is explicit."""

    vector: int = X_READS

    @property
    def reads(self):
        return self.vector


MUTATIONS = {
    "bias": "drops the bias term",
    "order": "pairs each pack's codes with its columns in reverse",
    "scale-step": "keeps the first block's scales and biases",
    "scale-stride": "steps the scales and biases one group per block, as along K",
    "k-step": "keeps the activation loader on the first block",
    "k-tail": "drops the last partial block of the reduction",
    "tail-mask": "loads the last partial block without load_safe's bounds",
    "rows": "ignores the threadgroup's row offset",
    "columns": "ignores the threadgroup's column offset",
    "transpose": "reads the weight tile as (BN, BK)",
    "store": "stores whole tiles past M",
    "batch": "ignores the weight batch offsets",
}


def _load_x(context, x, rows, columns, offset, tiles, safe):
    """BlockLoader<T, BM, BK, BK_padded, 1, 128, 1, 4>, one activation block.

    ``load_unsafe`` copies whole vectors of four. ``load_safe(short2(columns,
    rows))`` zeroes a thread's slots when its first row or column lies
    outside, and otherwise reads ``src[0]`` in place of each element past the
    bounds before zeroing it.
    """
    load, ld, xs, k = context["load"], context["ld"], context["Xs"], context["K"]
    family = "elements" if safe else "x-vector"
    for thread in range(THREADS):
        bi = thread // X_THREAD_COLUMNS
        bj = X_READS * (thread % X_THREAD_COLUMNS)
        source = x + bi * k + bj + offset
        live = not safe or (rows - bi > 0 and columns - bj > 0)
        for i in range(0, BM, X_THREAD_ROWS):
            for j in range(X_READS):
                slot = (bi + i) * ld + bj + j
                tiles["Xs"].add(slot)
                if not live:
                    xs[slot] = 0.0
                    continue
                inside = not safe or (i < rows - bi and j < columns - bj)
                index = source + (i * k + j if inside else 0)
                context["records"][family].add(("x", index))
                value = load("x", index)
                xs[slot] = value if inside else 0.0


def _w_loaders(context, wl, scales, biases):
    """Each thread's QuantizedBlockLoader state after its constructor."""
    geometry = context["geometry"]
    reads, packed = geometry["reads"], geometry["packedColumns"]
    n = context["N"]
    row_bytes = n * geometry["bytesPerPack"] // geometry["packFactor"]
    row_groups = n // context["groupSize"]
    loaders = []
    for thread in range(THREADS):
        bi, bj = reads * thread // packed, reads * thread % packed
        if packed * BK < THREADS and bi >= BK:
            continue
        loaders.append(
            {
                "bi": bi,
                "bj": bj,
                "src": wl + bi * row_bytes + bj * geometry["bytesPerPack"],
                "scales": scales + bi * row_groups,
                "biases": biases + bi * row_groups,
            }
        )
    return loaders


def _next_w(context, loaders, mutation):
    """QuantizedBlockLoader::next across ``BK`` weight rows."""
    geometry, n = context["geometry"], context["N"]
    tile_stride = BK * n * geometry["bytesPerPack"] // geometry["packFactor"]
    group_stride = BK * n // context["groupSize"]
    if mutation == "scale-stride":
        group_stride = 1
    for loader in loaders:
        loader["src"] += tile_stride
        if mutation == "scale-step":
            continue
        loader["scales"] += group_stride
        loader["biases"] += group_stride


def _block(context, loaders, accumulator, x, rows, columns, offset, safe, mutation):
    """One block: both loads, then BlockMMA::mma."""
    tiles = context["tiles"]
    _load_x(context, x, rows, columns, offset, tiles, safe)
    # QuantizedBlockLoader::load_safe(short2(BN, columns)) bounds its rows;
    # load_unsafe is load_safe with every row inside.
    base._load_w(context, loaders, columns, tiles, mutation)
    base._mma(context, accumulator, mutation)


def _tile(context, bases, tid_x, tid_y, mutation):
    """qmm_n_impl for one threadgroup."""
    x_base, w_base, s_base, b_base, y_base = bases
    geometry, k, n = context["geometry"], context["K"], context["N"]
    y_row = 0 if mutation == "rows" else tid_y * BM
    y_col = 0 if mutation == "columns" else tid_x * BN
    x = x_base + y_row * k
    wl = w_base * 4 + y_col * geometry["bytesPerPack"] // geometry["packFactor"]
    scales = s_base + y_col // context["groupSize"]
    biases = b_base + y_col // context["groupSize"]
    y = y_base + y_row * n + y_col
    rows = min(BM, context["M"] - y_row)
    x_safe = rows < BM
    blocks, tail = divmod(k, BK)
    context["paths"].add(
        f"rows-{'partial' if x_safe else 'whole'}/k-{'tail' if tail else 'whole'}"
    )
    loaders = _w_loaders(context, wl, scales, biases)
    accumulator = [[0.0] * BN for _ in range(BM)]
    for block in range(blocks):
        offset = 0 if mutation == "k-step" else block * BK
        _block(context, loaders, accumulator, x, rows, BK, offset, x_safe, mutation)
        _next_w(context, loaders, mutation)
    if tail and mutation != "k-tail":
        offset = 0 if mutation == "k-step" else blocks * BK
        if mutation == "tail-mask":
            _block(context, loaders, accumulator, x, rows, BK, offset, x_safe, mutation)
        else:
            _block(context, loaders, accumulator, x, rows, tail, offset, True, mutation)
    base._store(context, accumulator, y, rows, BN, mutation)


def simulate(info, data, families, mutation=None):
    """The kernel tile by tile; returns outputs, stores, reads and paths."""
    return base.simulate(info, data, families, mutation, tile=_tile, transpose_b=False)


def expected_paths(m, k):
    """The loop and store paths an ``M`` row output reaches, from its tiles."""
    tail = "tail" if k % BK else "whole"
    paths = set()
    for row in range(0, m, BM):
        rows = min(BM, m - row)
        paths.add(f"rows-{'partial' if rows < BM else 'whole'}/k-{tail}")
        paths.add("store-whole" if rows == BM else "store-safe")
    return paths
