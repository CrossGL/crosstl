"""Geometry of the pinned MLX ``steel_gemm_fused`` instantiations.

The helpers here follow ``steel/gemm/gemm.h``, ``loader.h`` and ``mma.h`` at
the pinned commit. They name every instantiated entry and derive, per entry,
the threadgroup slots each ``BlockLoader`` writes and ``BlockMMA`` reads, so
that the native gate can state the ranges it asserts to the translator.
"""

from dataclasses import dataclass

# (iname, Metal type, element bytes), in the order the source instantiates.
TYPES = (
    ("float16", "half", 2),
    ("bfloat16", "bfloat16_t", 2),
    ("float32", "float", 4),
    ("complex64", "complex64_t", 8),
)
# (bm, bn, bk, wm, wn), as ``instantiate_gemm_shapes_helper`` lists them.
SHAPES = (
    (64, 64, 16, 2, 2),
    (64, 64, 16, 1, 2),
    (64, 32, 32, 2, 2),
    (32, 64, 16, 1, 2),
    (32, 32, 16, 2, 2),
    (64, 32, 8, 4, 1),
)
TRANSPOSES = (
    ("nn", False, False),
    ("nt", False, True),
    ("tn", True, False),
    ("tt", True, True),
)
FRAGMENT = 8
SIMD_WIDTH = 32


def _token(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


@dataclass(frozen=True)
class Loader:
    """One ``BlockLoader`` instantiation and the slots it writes."""

    function: str
    rows: int
    columns: int
    leading: int
    threads: int

    @property
    def reads(self):
        return self.rows * self.columns // self.threads

    def slots(self):
        """Threadgroup elements ``load_unsafe`` writes, one list per thread."""
        thread_columns = self.columns // self.reads
        thread_rows = self.threads // thread_columns
        return [
            [
                (lane // thread_columns + row) * self.leading
                + self.reads * (lane % thread_columns)
                + element
                for row in range(0, self.rows, thread_rows)
                for element in range(self.reads)
            ]
            for lane in range(self.threads)
        ]


@dataclass(frozen=True)
class GemmEntry:
    """One instantiated ``steel_gemm_fused`` entry point."""

    transpose: str
    transpose_a: bool
    transpose_b: bool
    iname: str
    metal_type: str
    element_bytes: int
    bm: int
    bn: int
    bk: int
    wm: int
    wn: int

    @property
    def dtype(self):
        return self.iname

    @property
    def name(self):
        return (
            f"steel_gemm_fused_{self.transpose}_{self.iname}_{self.iname}"
            f"_bm{self.bm}_bn{self.bn}_bk{self.bk}_wm{self.wm}_wn{self.wn}"
        )

    @property
    def threads(self):
        return self.wm * self.wn * SIMD_WIDTH

    @property
    def workgroup_size(self):
        return (SIMD_WIDTH, self.wn, self.wm)

    @property
    def padding(self):
        # gemm.h pads each threadgroup row by 16 bytes.
        return 16 // self.element_bytes

    @property
    def lda(self):
        return (self.bm if self.transpose_a else self.bk) + self.padding

    @property
    def ldb(self):
        return (self.bk if self.transpose_b else self.bn) + self.padding

    def _loader(self, flag, rows, columns, leading_base, reduction_flag):
        # Materialized names spell the template arguments as the source writes
        # them: ``flag ? x : y`` gives ``flag_x_y`` and ``x + pad`` gives
        # ``x_pad``; ``!flag`` keeps only the flag's value.
        a, b = rows
        c, d = columns
        e, f = leading_base
        pad = self.padding
        parts = (
            self.metal_type,
            flag,
            a,
            b,
            flag,
            c,
            d,
            flag,
            e,
            pad,
            f,
            pad,
            reduction_flag,
            self.threads,
        )
        name = (
            "BlockLoader_" + "_".join(_token(part) for part in parts) + "__load_unsafe"
        )
        actual_rows = a if flag else b
        actual_columns = c if flag else d
        actual_leading = (e if flag else f) + pad
        return Loader(name, actual_rows, actual_columns, actual_leading, self.threads)

    @property
    def loaders(self):
        ta, tb = self.transpose_a, self.transpose_b
        loader_a = self._loader(
            ta, (self.bk, self.bm), (self.bm, self.bk), (self.bm, self.bk), ta
        )
        loader_b = self._loader(
            tb, (self.bn, self.bk), (self.bk, self.bn), (self.bk, self.bn), tb
        )
        return loader_a, loader_b

    def tile_reads(self):
        """Threadgroup elements ``BlockMMA::mma`` reads from the A and B tiles."""
        a_str_m, a_str_k = (1, self.lda) if self.transpose_a else (self.lda, 1)
        b_str_k, b_str_n = (1, self.ldb) if self.transpose_b else (self.ldb, 1)
        tile_m = self.bm // (FRAGMENT * self.wm)
        tile_n = self.bn // (FRAGMENT * self.wn)
        a_slots, b_slots = set(), set()
        for group in range(self.wm * self.wn):
            tm = FRAGMENT * (group // self.wn)
            tn = FRAGMENT * (group % self.wn)
            for lane in range(SIMD_WIDTH):
                qid = lane // 4
                sm = (qid & 4) + (lane // 2) % 4
                sn = (qid & 2) * 2 + (lane % 2) * 2
                for kk in range(0, self.bk, FRAGMENT):
                    for element in range(2):
                        for tile in range(tile_m):
                            row = tm + sm + tile * FRAGMENT * self.wm
                            a_slots.add(row * a_str_m + (sn + kk + element) * a_str_k)
                        for tile in range(tile_n):
                            column = tn + sn + tile * FRAGMENT * self.wn + element
                            b_slots.add((sm + kk) * b_str_k + column * b_str_n)
        return a_slots, b_slots


def entries():
    """Every instantiated entry, in source order."""
    return tuple(
        GemmEntry(tname, ta, tb, iname, metal_type, size, *shape)
        for iname, metal_type, size in TYPES
        for shape in SHAPES
        for tname, ta, tb in TRANSPOSES
    )


BY_NAME = {entry.name: entry for entry in entries()}
