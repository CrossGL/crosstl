"""Entries and block-mask semantics of the pinned MLX ``steel_gemm_masked``.

``steel_gemm_masked.metal`` instantiates ``block_masked_gemm`` for three
element types, two tile shapes, four transpositions, four alignment pairs and
eight mask combinations. The helpers here follow ``steel_gemm_masked.h`` and
``matmul.cpp``'s ``BlockMaskedMM::eval_gpu`` at the pinned commit.

Masks hold one value per block. The output mask has one per ``BM x BN``
output tile. The operand masks have one per ``BM x BM`` block of A and B,
because ``k_mask_factor = BM / BK`` K iterations share one. A K block
contributes to an output tile only when its lhs and rhs values are both
non-zero. Boolean masks only select; masks of the element type also scale
(``has_mul_operand_mask`` and ``has_mul_output_mask``). A zero output mask
value writes zeros to the tile and skips the product.
"""

from dataclasses import dataclass

from demos.integrations.mlx.tests.kernels import gemm_fused_model as fused

# (iname, Metal type, element bytes), in the order the source instantiates.
TYPES = (
    ("float16", "half", 2),
    ("bfloat16", "bfloat16_t", 2),
    ("float32", "float", 4),
)
# (bm, bn, bk, wm, wn), as ``instantiate_gemm_shapes_helper`` lists them.
SHAPES = ((32, 32, 16, 2, 2), (64, 64, 16, 2, 2))
# (aname, MN_aligned, kname, K_aligned), as ``instantiate_gemm_aligned_helper``
# lists them.
ALIGNMENTS = (
    ("taligned", True, "taligned", True),
    ("taligned", True, "naligned", False),
    ("naligned", False, "taligned", True),
    ("naligned", False, "naligned", False),
)
# (output mask, operand mask), as ``instantiate_gemm_mask_helper`` lists
# them. ``"T"`` is the element type, which makes the mask multiplicative.
MASKS = (
    ("bool", "bool"),
    ("T", "T"),
    ("bool", "T"),
    ("T", "bool"),
    ("nomask", "bool"),
    ("nomask", "T"),
    ("bool", "nomask"),
    ("T", "nomask"),
)


@dataclass(frozen=True)
class MaskedEntry:
    """One instantiated ``block_masked_gemm`` entry point."""

    gemm: fused.GemmEntry
    out_mask: str
    op_mask: str
    mn_aligned: bool
    k_aligned: bool

    @property
    def dtype(self):
        return self.gemm.dtype

    @property
    def transpose(self):
        return self.gemm.transpose

    @property
    def transpose_a(self):
        return self.gemm.transpose_a

    @property
    def transpose_b(self):
        return self.gemm.transpose_b

    @property
    def block(self):
        assert self.gemm.bm == self.gemm.bn
        return self.gemm.bm

    @property
    def workgroup_size(self):
        return self.gemm.workgroup_size

    @property
    def loaders(self):
        return self.gemm.loaders

    def mask_kind(self, which):
        """``"nomask"``, ``"bool"`` or ``"scale"`` for ``"out"`` or ``"op"``."""
        kind = {"out": self.out_mask, "op": self.op_mask}[which]
        return "scale" if kind == "T" else kind

    def _mask_name(self, kind):
        return {"bool": "bool_", "nomask": "nomask", "T": self.dtype}[kind]

    @property
    def name(self):
        g = self.gemm
        return (
            f"steel_gemm_block_outmask_{self._mask_name(self.out_mask)}"
            f"_opmask_{self._mask_name(self.op_mask)}_{g.transpose}"
            f"_{g.iname}_{g.iname}_bm{g.bm}_bn{g.bn}_bk{g.bk}_wm{g.wm}_wn{g.wn}"
            f"_MN_{'t' if self.mn_aligned else 'n'}aligned"
            f"_K_{'t' if self.k_aligned else 'n'}aligned"
        )


def entries():
    """Every instantiated entry, in source order."""
    return tuple(
        MaskedEntry(
            fused.GemmEntry(tname, ta, tb, iname, metal_type, size, *shape),
            out_mask,
            op_mask,
            mn_aligned,
            k_aligned,
        )
        for iname, metal_type, size in TYPES
        for shape in SHAPES
        for tname, ta, tb in fused.TRANSPOSES
        for _, mn_aligned, _, k_aligned in ALIGNMENTS
        for out_mask, op_mask in MASKS
    )


BY_NAME = {entry.name: entry for entry in entries()}


def product(entry, m, n, k, a, b, out_mask, lhs_mask, rhs_mask):
    """The block-masked product ``block_masked_gemm`` writes, as a dict.

    ``a(row, inner)`` and ``b(inner, column)`` give operand values. The masks
    are callables of block coordinates, ``None`` when the entry has none:
    ``out_mask(tile_row, tile_column)``, ``lhs_mask(tile_row, k_block)`` and
    ``rhs_mask(k_block, tile_column)``. The result maps every output
    coordinate the kernel writes to its value; it writes all of them.
    """
    block = entry.block
    scale_operands = entry.mask_kind("op") == "scale"
    scale_output = entry.mask_kind("out") == "scale"
    result = {}
    for row in range(m):
        for column in range(n):
            tile_row, tile_column = row // block, column // block
            if out_mask is not None and not out_mask(tile_row, tile_column):
                result[row, column] = 0
                continue
            # The kernel's accumulators start at +0.
            total = 0.0
            for inner in range(k):
                k_block = inner // block
                left, right = a(row, inner), b(inner, column)
                if lhs_mask is not None:
                    lhs = lhs_mask(tile_row, k_block)
                    rhs = rhs_mask(k_block, tile_column)
                    if not (lhs and rhs):
                        continue
                    if scale_operands:
                        left, right = left * lhs, right * rhs
                total += left * right
            if scale_output:
                total *= out_mask(tile_row, tile_column)
            result[row, column] = total
    return result
