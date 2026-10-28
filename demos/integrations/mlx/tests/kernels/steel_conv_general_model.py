"""Geometry and host parameters of the pinned MLX ``steel_conv_general`` kernel.

``steel_conv_general.metal`` instantiates ``implicit_gemm_conv_2d_general``
for three element types and six block shapes. The function constant
``align_C`` (id 200) selects a loop over whole 16-channel blocks or one that
ends with a partial block. ``implicit_gemm_conv_2D_general_gpu`` in
``conv.cpp`` runs it for one-group 2-D convolutions that are input-dilated or
whose channels the specialized loaders cannot take. Grid ``z`` enumerates the
output row and column residues that share filter taps, and ``base_h`` and
``base_w`` give each residue's first tap and tap count. This module names the
entries and reproduces every parameter the host encodes.
"""

import math
from dataclasses import dataclass

from demos.integrations.mlx.tests.kernels import steel_conv_model as conv_model

SOURCE = "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.metal"
ALIGN_C_ID = 200
BK = 16


@dataclass(frozen=True)
class GeneralEntry(conv_model.ConvEntry):
    """One ``implicit_gemm_conv_2d_general`` entry.

    Its loaders share the large-channel loaders' tile geometry: four or
    eight adjacent elements per thread, and one for 8-column weight tiles.
    """

    @property
    def source(self):
        return SOURCE

    @property
    def name(self):
        return (
            f"implicit_gemm_conv_2d_general_{self.dtype}_bm{self.bm}_bn{self.bn}"
            f"_bk{self.bk}_wm{self.wm}_wn{self.wn}"
        )

    def accepts(self, conv):
        """The kernel indexes one group of 2-D channels-last data."""
        return conv.rank == 2 and conv.groups == 1


def entries():
    """Every instantiated entry, in source order."""
    return tuple(
        GeneralEntry(2, dtype, metal_type, size, *block, "l", 0, "l", False)
        for dtype, metal_type, size in conv_model.TYPES
        for block in conv_model.BLOCKS
    )


BY_NAME = {entry.name: entry for entry in entries()}


def align_c(conv):
    """The ``align_C`` value the host specializes for ``conv``."""
    return conv.channels % BK == 0


def _bases(residues, stride, pad, taps, kdil, idil, flip):
    """``Conv2DGeneralBaseInfo`` per output residue along one axis.

    Each residue starts at the first tap that lands on an input pixel and
    steps ``lcm(idil, kdil) / kdil`` taps at a time.
    """
    step = math.lcm(idil, kdil) // kdil
    jump = -kdil if flip else kdil
    first = (taps - 1) * kdil if flip else 0
    bases = []
    for residue in range(residues):
        position = residue * stride - pad + first
        base = 0
        while base < taps and position % idil:
            base += 1
            position += jump
        bases.append({"weight_base": base, "weight_size": -(-(taps - base) // step)})
    return tuple(bases)


@dataclass(frozen=True)
class Launch:
    """The buffers ``implicit_gemm_conv_2D_general_gpu`` encodes and its grid."""

    gemm: dict
    jump: dict
    base_h: tuple
    base_w: tuple
    grid: tuple
    align_c: bool


def launch(conv, entry):
    """The host parameters and grid for ``conv`` dispatched to ``entry``."""
    assert entry.accepts(conv), (entry.name, conv)
    out_h, out_w = conv.out_spatial
    idil, kdil, stride = conv.input_dilation, conv.dilation, conv.stride
    wgt_jump = [math.lcm(i, k) // k for i, k in zip(idil, kdil)]
    out_jump = [math.lcm(i, s) // s for i, s in zip(idil, stride)]
    adj_h, adj_w = -(-out_h // out_jump[0]), -(-out_w // out_jump[1])
    adj_m = conv.batch * adj_h * adj_w
    jump = dict(
        f_wgt_jump_h=wgt_jump[0],
        f_wgt_jump_w=wgt_jump[1],
        f_out_jump_h=out_jump[0],
        f_out_jump_w=out_jump[1],
        adj_out_h=adj_h,
        adj_out_w=adj_w,
        adj_out_hw=adj_h * adj_w,
        adj_implicit_m=adj_m,
    )
    base_h, base_w = (
        _bases(
            out_jump[axis],
            stride[axis],
            conv.padding[axis],
            conv.kernel[axis],
            kdil[axis],
            idil[axis],
            conv.flip,
        )
        for axis in range(2)
    )
    sign = -1 if conv.flip else 1
    in_strides = conv.params()["in_strides"]
    step_w = in_strides[2] * kdil[1]
    step_h = in_strides[1] * kdil[0]
    taps_h, taps_w = conv.kernel
    tiles_n = -(-conv.outputs // entry.bn)
    tiles_m = -(-adj_m // entry.bm)
    gemm = dict(
        M=conv.batch * out_h * out_w,
        N=conv.outputs,
        K=taps_h * taps_w * conv.channels,
        gemm_k_iterations=-(-conv.channels // BK),
        inp_jump_w=sign * step_w,
        inp_jump_h=sign * (step_h - (taps_w - 1) * step_w),
        inp_jump_c=BK - sign * (taps_h - 1) * step_h - sign * (taps_w - 1) * step_w,
        tiles_n=tiles_n,
        tiles_m=tiles_m,
        swizzle_log=0,
    )
    grid = (tiles_n, tiles_m, out_jump[0] * out_jump[1])
    return Launch(gemm, jump, base_h, base_w, grid, align_c(conv))


def host_entry(conv, dtype):
    """The entry ``implicit_gemm_conv_2D_general_gpu`` selects for ``dtype``."""
    out_jump = [math.lcm(i, s) // s for i, s in zip(conv.input_dilation, conv.stride)]
    adj_m = conv.batch * math.prod(
        -(-size // jump) for size, jump in zip(conv.out_spatial, out_jump)
    )
    bm = 64 if adj_m >= 8192 and conv.channels >= 64 else 32
    bn = 64 if bm == 64 and conv.outputs >= 64 else 32
    return BY_NAME[f"implicit_gemm_conv_2d_general_{dtype}_bm{bm}_bn{bn}_bk16_wm2_wn2"]
