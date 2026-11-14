"""Geometry of the pinned MLX ``steel_gemm_splitk`` instantiations.

``steel_gemm_splitk.metal`` builds its kernels from the same ``GEMMKernel``
as ``steel_gemm_fused``, so each entry's ``BlockLoader`` and ``BlockMMA``
geometry follows ``gemm_fused_model``. This module names the split-K entries,
their accumulation kernels, and the partition plan ``matmul.cpp`` dispatches.
"""

from dataclasses import dataclass, replace

from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused

# (iname, Metal type, element bytes, oname), in the order the source instantiates.
TYPES = (
    ("float16", "half", 2, "float32"),
    ("bfloat16", "bfloat16_t", 2, "float32"),
    ("float32", "float", 4, "float32"),
    ("complex64", "complex64_t", 8, "complex64"),
)
# (bm, bn, bk, wm, wn), as ``instantiate_gemm_shapes_helper`` lists them.
SHAPES = (
    (16, 16, 16, 2, 2),
    (16, 32, 16, 2, 2),
    (32, 16, 16, 2, 2),
    (32, 32, 16, 2, 2),
)
# (MN_aligned, K_aligned), as ``instantiate_gemm_aligned_helper`` lists them.
ALIGNMENTS = ((True, True), (True, False), (False, True), (False, False))
# (oname, aname), as ``instantiate_accum`` lists them.
ACCUMULATORS = (
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float32", "float32"),
    ("complex64", "complex64"),
)


def _aligned(flag):
    return "taligned" if flag else "naligned"


@dataclass(frozen=True)
class SplitKEntry(fused.GemmEntry):
    """One instantiated ``gemm_splitk`` entry point."""

    oname: str = "float32"
    mn_aligned: bool = False
    k_aligned: bool = False

    @property
    def name(self):
        return (
            f"steel_gemm_splitk_{self.transpose}_{self.iname}_{self.oname}"
            f"_bm{self.bm}_bn{self.bn}_bk{self.bk}_wm{self.wm}_wn{self.wn}"
            f"_MN_{_aligned(self.mn_aligned)}_K_{_aligned(self.k_aligned)}"
        )

    @property
    def loaders(self):
        # A square tile with equal transposition gives both loaders the same
        # spelled arguments; ``loader_b_t``'s distinct specialization then
        # takes the first numbered suffix.
        loader_a, loader_b = super().loaders
        if loader_a.function == loader_b.function:
            loader_b = replace(
                loader_b,
                function=loader_b.function.replace("__load_unsafe", "_2__load_unsafe"),
            )
        return loader_a, loader_b


@dataclass(frozen=True)
class AccumEntry:
    """One instantiated ``gemm_splitk_accum`` or ``_axpby`` entry point."""

    oname: str
    aname: str
    axpby: bool

    @property
    def dtype(self):
        return self.oname

    @property
    def name(self):
        suffix = "_axbpy" if self.axpby else ""
        return f"steel_gemm_splitk_accum_{self.oname}_{self.aname}{suffix}"


def entries():
    """Every instantiated ``gemm_splitk`` entry, in source order."""
    return tuple(
        SplitKEntry(
            tname,
            ta,
            tb,
            iname,
            metal_type,
            size,
            *shape,
            oname=oname,
            mn_aligned=mn_aligned,
            k_aligned=k_aligned,
        )
        for iname, metal_type, size, oname in TYPES
        for shape in SHAPES
        for tname, ta, tb in fused.TRANSPOSES
        for mn_aligned, k_aligned in ALIGNMENTS
    )


def accum_entries():
    """Every instantiated accumulation entry, in source order."""
    return tuple(
        AccumEntry(oname, aname, axpby)
        for oname, aname in ACCUMULATORS
        for axpby in (False, True)
    )


def _next_power_of_2(value):
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def partition_plan(m, n, k, bk=16):
    """``steel_gemm_splitk_axpby``'s partition count, iterations and size."""
    tiles = (-(-m // 32)) * (-(-n // 32))
    partitions = min(max(2, _next_power_of_2((k // 16) // tiles)), 32)
    iterations = (k // bk) // partitions
    return partitions, iterations, iterations * bk


def partition_ranges(entry, k, partitions, size):
    """The ``K`` range each partition's threadgroups accumulate.

    Every partition but the last covers ``size`` columns from its start; the
    last also runs the remaining whole tiles and, unaligned, the tail.
    """
    ranges = []
    for partition in range(partitions):
        start = size * partition
        stop = start + size
        if partition + 1 == partitions:
            remaining = (k - stop) // entry.bk
            stop += remaining * entry.bk
            if not entry.k_aligned:
                stop = k
        ranges.append((start, stop))
    return ranges


BY_NAME = {entry.name: entry for entry in entries()}
ACCUM_BY_NAME = {entry.name: entry for entry in accum_entries()}
