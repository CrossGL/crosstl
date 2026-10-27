"""Geometry of the pinned MLX ``steel_gemm_gather`` instantiations.

``steel_gemm_gather.metal`` builds ``gather_mm`` and ``gather_mm_rhs`` from
the same ``GEMMKernel`` as ``steel_gemm_fused`` with both alignment template
arguments ``true``, so each entry's ``BlockLoader`` and ``BlockMMA``
geometry follows ``gemm_fused_model``. This module names the entries in
source order.
"""

from dataclasses import dataclass

from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused

# (iname, Metal type, element bytes); the gather kernels have no complex64.
TYPES = fused.TYPES[:3]
# (bm, bn, bk, wm, wn), as ``instantiate_gather_mm_shapes_helper`` lists them.
RHS_SHAPES = ((16, 64, 16, 1, 2),)
MM_SHAPES = (
    (64, 64, 16, 2, 2),
    (64, 64, 16, 1, 2),
    (64, 32, 32, 2, 2),
    (32, 64, 16, 1, 2),
    (32, 32, 16, 2, 2),
)
# ``gather_mm_rhs`` is instantiated without a transposed A.
RHS_TRANSPOSES = fused.TRANSPOSES[:2]


@dataclass(frozen=True)
class GatherEntry(fused.GemmEntry):
    """One instantiated ``gather_mm`` or ``gather_mm_rhs`` entry point."""

    rhs: bool = False

    @property
    def name(self):
        kernel = "steel_gather_mm_rhs" if self.rhs else "steel_gather_mm"
        return (
            f"{kernel}_{self.transpose}_{self.iname}_{self.iname}"
            f"_bm{self.bm}_bn{self.bn}_bk{self.bk}_wm{self.wm}_wn{self.wn}"
        )


def entries():
    """Every instantiated entry, in source order."""
    result = []
    for iname, metal_type, size in TYPES:
        for shapes, transposes, rhs in (
            (RHS_SHAPES, RHS_TRANSPOSES, True),
            (MM_SHAPES, fused.TRANSPOSES, False),
        ):
            for shape in shapes:
                for tname, ta, tb in transposes:
                    result.append(
                        GatherEntry(
                            tname, ta, tb, iname, metal_type, size, *shape, rhs=rhs
                        )
                    )
    return tuple(result)


BY_NAME = {entry.name: entry for entry in entries()}
