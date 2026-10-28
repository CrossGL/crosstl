"""Entries and segment semantics of the pinned MLX ``steel_gemm_segmented``.

``steel_gemm_segmented.metal`` instantiates ``segmented_mm`` for three element
types, five tile shapes and four transpositions. It builds the kernel from the
same ``GEMMKernel`` as ``steel_gemm_fused`` with both alignment template
arguments ``true``, so each entry's ``BlockLoader`` and ``BlockMMA`` geometry
follows ``gemm_fused_model``. The helpers here follow
``steel_gemm_segmented.h`` and ``matmul.cpp``'s ``segmented_mm`` at the pinned
commit.

Output batch ``z`` is the product of one A and one B restricted to the inner
range ``[k_start, k_end)`` that ``segments`` gives for ``z``. With
``segments_contiguous`` the buffer holds one ``(k_start, k_end)`` pair per
batch; otherwise it holds boundaries, and batch ``z`` spans words ``z`` and
``z + 1``.
"""

from dataclasses import dataclass

from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused

# (iname, Metal type, element bytes), in the order the source instantiates.
TYPES = fused.TYPES[:3]
# (bm, bn, bk, wm, wn), as ``instantiate_segmented_mm_shapes_helper`` lists
# them.
SHAPES = (
    (64, 64, 16, 2, 2),
    (64, 64, 16, 1, 2),
    (64, 32, 32, 2, 2),
    (32, 64, 16, 1, 2),
    (32, 32, 16, 2, 2),
)


@dataclass(frozen=True)
class SegmentedEntry(fused.GemmEntry):
    """One instantiated ``segmented_mm`` entry point."""

    @property
    def name(self):
        return (
            f"steel_segmented_mm_{self.transpose}_{self.iname}_{self.iname}"
            f"_bm{self.bm}_bn{self.bn}_bk{self.bk}_wm{self.wm}_wn{self.wn}"
        )


def entries():
    """Every instantiated entry, in source order."""
    return tuple(
        SegmentedEntry(tname, ta, tb, iname, metal_type, size, *shape)
        for iname, metal_type, size in TYPES
        for shape in SHAPES
        for tname, ta, tb in fused.TRANSPOSES
    )


BY_NAME = {entry.name: entry for entry in entries()}


def segments(words, contiguous, batches):
    """The ``(k_start, k_end)`` range ``segmented_mm`` reads for each batch."""
    if contiguous:
        return [(words[2 * z], words[2 * z + 1]) for z in range(batches)]
    return [(words[z], words[z + 1]) for z in range(batches)]


def product(m, n, segment, a, b):
    """Output batch values for one segment, keyed by ``(row, column)``.

    ``a(row, inner)`` and ``b(inner, column)`` give operand values. The
    kernel's accumulators start at +0, so an empty segment writes zeros.
    """
    start, end = segment
    return {
        (row, column): sum(
            (a(row, inner) * b(inner, column) for inner in range(start, end)), 0.0
        )
        for row in range(m)
        for column in range(n)
    }
