"""Geometry and arithmetic of the pinned MLX ``steel_conv`` kernels.

``steel_conv.metal`` instantiates ``implicit_gemm_conv_2d`` for three element
types, six block shapes and six loader selections, and ``steel_conv_3d.metal``
instantiates ``implicit_gemm_conv_3d`` for the same types and shapes with a
small- or large-filter input loader. ``conv.cpp`` turns a convolution into an
implicit GEMM: ``M`` output pixels, ``N`` output channels per group and ``K``
filter taps times input channels per group. This module names the entries,
models the threadgroup slots each loader writes and reproduces the host
parameters ``implicit_gemm_conv_2D_gpu`` and ``implicit_gemm_conv_3D_gpu``
encode.
"""

import itertools
import math
from dataclasses import dataclass, field

SOURCES = {
    2: "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.metal",
    3: "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.metal",
}
TYPES = (
    ("float32", "float", 4),
    ("float16", "half", 2),
    ("bfloat16", "bfloat16_t", 2),
)
# (bm, bn, bk, wm, wn) in instantiation order.
BLOCKS = (
    (32, 8, 16, 4, 1),
    (64, 8, 16, 4, 1),
    (32, 32, 16, 2, 2),
    (32, 64, 16, 2, 2),
    (64, 32, 16, 2, 2),
    (64, 64, 16, 2, 2),
)
# (channel name, channels, filter name, small filter) in instantiation order.
FILTERS = {
    2: (
        ("l", 0, "s", True),
        ("l", 0, "l", False),
        ("1", 1, "l", False),
        ("2", 2, "l", False),
        ("3", 3, "l", False),
        ("4", 4, "l", False),
    ),
    3: (("l", 0, "s", True), ("l", 0, "l", False)),
}
SIMD_WIDTH = 32
# ``ChannelHelper`` widens 3 channels to 4 and keeps 1, 2 and 4.
CHANNEL_VECTORS = {1: 1, 2: 2, 3: 4, 4: 4}
# The small-filter loaders keep one ``short`` bit per tap and axis.
SMALL_FILTER_TAPS = 16


@dataclass(frozen=True)
class Loader:
    """One loader's threadgroup tile: ``rows`` x ``columns`` with row pitch
    ``leading``, written ``vector`` adjacent elements per thread and row."""

    rows: int
    columns: int
    leading: int
    vector: int
    threads: int
    # The small-channel weight loader returns early for threads whose row
    # lies past the tile; the other loaders cover the tile exactly.
    clip_rows: bool = False

    def slots(self):
        """Threadgroup elements ``load_unsafe`` writes, one list per thread."""
        thread_columns = self.columns // self.vector
        thread_rows = self.threads // thread_columns
        lanes = []
        for lane in range(self.threads):
            row, column = lane // thread_columns, self.vector * (lane % thread_columns)
            if self.clip_rows and row >= self.rows:
                lanes.append([])
                continue
            lanes.append(
                [
                    (row + step) * self.leading + column + element
                    for step in range(0, self.rows, thread_rows)
                    for element in range(self.vector)
                ]
            )
        return lanes


@dataclass(frozen=True)
class ConvEntry:
    """One instantiated ``implicit_gemm_conv_{2,3}d`` entry point."""

    rank: int
    dtype: str
    metal_type: str
    element_bytes: int
    bm: int
    bn: int
    bk: int
    wm: int
    wn: int
    channel_name: str
    channels: int
    filter_name: str
    small_filter: bool

    @property
    def source(self):
        return SOURCES[self.rank]

    @property
    def name(self):
        channel = "" if self.rank == 3 else f"_channel_{self.channel_name}"
        return (
            f"implicit_gemm_conv_{self.rank}d_{self.dtype}_bm{self.bm}_bn{self.bn}"
            f"_bk{self.bk}_wm{self.wm}_wn{self.wn}{channel}_filter_{self.filter_name}"
        )

    @property
    def threads(self):
        return self.wm * self.wn * SIMD_WIDTH

    @property
    def workgroup_size(self):
        return (SIMD_WIDTH, self.wn, self.wm)

    @property
    def padding(self):
        # Both kernels pad each threadgroup row by 16 bytes.
        return 16 // self.element_bytes

    @property
    def leading(self):
        # Both tiles hold BK reduction elements per row: A is not transposed
        # and B is.
        return self.bk + self.padding

    def _vector(self, rows, weight):
        if self.channels:
            return CHANNEL_VECTORS[self.channels]
        if weight and self.bn == 8:
            return 1
        return 8 if self.threads // (rows * self.bk) >= 8 else 4

    @property
    def loaders(self):
        """The input (A) and weight (B) loaders."""
        loader_a = Loader(
            self.bm, self.bk, self.leading, self._vector(self.bm, False), self.threads
        )
        loader_b = Loader(
            self.bn,
            self.bk,
            self.leading,
            self._vector(self.bn, True),
            self.threads,
            clip_rows=bool(self.channels),
        )
        return loader_a, loader_b

    def tile_reads(self):
        """Threadgroup elements ``BlockMMA::mma`` reads from the A and B tiles."""
        tile_m = self.bm // (8 * self.wm)
        tile_n = self.bn // (8 * self.wn)
        a_slots, b_slots = set(), set()
        for group in range(self.wm * self.wn):
            tm = 8 * (group // self.wn)
            tn = 8 * (group % self.wn)
            for lane in range(SIMD_WIDTH):
                qid = lane // 4
                sm = (qid & 4) + (lane // 2) % 4
                sn = (qid & 2) * 2 + (lane % 2) * 2
                for kk in range(0, self.bk, 8):
                    for element in range(2):
                        for tile in range(tile_m):
                            row = tm + sm + tile * 8 * self.wm
                            a_slots.add(row * self.leading + sn + kk + element)
                        for tile in range(tile_n):
                            column = tn + sn + tile * 8 * self.wn + element
                            b_slots.add(column * self.leading + sm + kk)
        return a_slots, b_slots

    def accepts(self, conv):
        """Whether ``conv`` meets this entry's loader preconditions."""
        if conv.rank != self.rank:
            return False
        if self.channels:
            return conv.per_group == self.channels
        if conv.per_group % self.bk:
            return False
        return not self.small_filter or max(conv.kernel) <= SMALL_FILTER_TAPS


def entries(rank=2):
    """Every instantiated entry of one rank, in source order."""
    return tuple(
        ConvEntry(rank, dtype, metal_type, size, *block, *selection)
        for dtype, metal_type, size in TYPES
        for block in BLOCKS
        for selection in FILTERS[rank]
    )


BY_NAME = {entry.name: entry for rank in SOURCES for entry in entries(rank)}


@dataclass(frozen=True)
class Conv:
    """One 2-D or 3-D convolution in MLX's channels-last layouts.

    ``dilation`` spaces the filter taps and ``input_dilation`` the input
    pixels, as if ``input_dilation - 1`` zeros sat between them. Unset strides
    and dilations are one and unset paddings zero.
    """

    batch: int
    spatial: tuple
    kernel: tuple
    channels: int
    outputs: int
    groups: int = 1
    stride: tuple = field(default=None)
    padding: tuple = field(default=None)
    dilation: tuple = field(default=None)
    flip: bool = False
    input_dilation: tuple = field(default=None)

    def __post_init__(self):
        rank = len(self.spatial)
        for name, default in (
            ("stride", 1),
            ("padding", 0),
            ("dilation", 1),
            ("input_dilation", 1),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, (default,) * rank)
        assert {len(self.kernel), len(self.stride), len(self.padding)} == {rank}
        assert len(self.dilation) == len(self.input_dilation) == rank

    @property
    def rank(self):
        return len(self.spatial)

    @property
    def out_spatial(self):
        return tuple(
            ((size - 1) * idil + 2 * pad - dil * (taps - 1)) // stride + 1
            for size, taps, stride, pad, dil, idil in zip(
                self.spatial,
                self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                self.input_dilation,
            )
        )

    @property
    def per_group(self):
        return self.channels // self.groups

    @property
    def outputs_per_group(self):
        return self.outputs // self.groups

    @property
    def input_shape(self):
        return (self.batch, *self.spatial, self.channels)

    @property
    def weight_shape(self):
        return (self.outputs, *self.kernel, self.per_group)

    @property
    def output_shape(self):
        return (self.batch, *self.out_spatial, self.outputs)

    def params(self):
        """``MLXConvParams<N>`` as ``conv.cpp`` fills it for row-major arrays."""
        return dict(
            N=self.batch,
            C=self.channels,
            O=self.outputs,
            iS=list(self.spatial),
            wS=list(self.kernel),
            oS=list(self.out_spatial),
            str=list(self.stride),
            pad=list(self.padding),
            kdil=list(self.dilation),
            idil=list(self.input_dilation),
            in_strides=_row_major(self.input_shape),
            wt_strides=_row_major(self.weight_shape),
            out_strides=_row_major(self.output_shape),
            groups=self.groups,
            flip=self.flip,
        )

    def _k_iterations(self, bk):
        taps = math.prod(self.kernel)
        if self.rank == 2 and self.per_group <= 2:
            return -(-taps * self.per_group // bk)
        if self.rank == 2 and self.per_group <= 4:
            return -(-(taps * 4) // bk)
        return taps * -(-self.per_group // bk)

    def gemm_params(self, entry):
        """``ImplicitGemmConv{2,3}DParams`` as the host builds them.

        Each jump moves the input pointer from the last tap along one axis to
        the first tap along the next, and ``inp_jump_c`` to the next channel
        block, so the fields appear innermost axis first.
        """
        implicit_m = self.batch * math.prod(self.out_spatial)
        implicit_n = self.outputs_per_group
        sign = -1 if self.flip else 1
        in_strides = _row_major(self.input_shape)
        # Jump per tap along each spatial axis, outermost first.
        steps = [
            in_strides[axis + 1] * self.dilation[axis] for axis in range(self.rank)
        ]
        jumps = {}
        names = ("inp_jump_w", "inp_jump_h", "inp_jump_d")
        for depth, name in enumerate(names[: self.rank]):
            axis = self.rank - 1 - depth
            inner = sum(
                (self.kernel[a] - 1) * steps[a] for a in range(axis + 1, self.rank)
            )
            jumps[name] = sign * (steps[axis] - inner)
        jumps["inp_jump_c"] = entry.bk - sign * sum(
            (taps - 1) * step for taps, step in zip(self.kernel, steps)
        )
        return dict(
            M=implicit_m,
            N=implicit_n,
            K=math.prod(self.kernel) * self.per_group,
            gemm_k_iterations=self._k_iterations(entry.bk),
            **jumps,
            tiles_n=-(-implicit_n // entry.bn),
            tiles_m=-(-implicit_m // entry.bm),
            swizzle_log=0,
        )

    def host_entry(self, dtype):
        """The entry ``implicit_gemm_conv_{2,3}D_gpu`` selects for ``dtype``."""
        implicit_m = self.batch * math.prod(self.out_spatial)
        implicit_n = self.outputs_per_group
        wm, wn = 2, 2
        bm = 64 if implicit_m >= 8192 and self.per_group >= 64 else 32
        bn = 64 if bm == 64 or implicit_n >= 64 else 32
        if implicit_n <= 16:
            bn, wm, wn = 8, 4, 1
        channels = self.per_group if self.rank == 2 and self.per_group <= 4 else 0
        small = not channels and max(self.kernel) <= SMALL_FILTER_TAPS
        channel = "" if self.rank == 3 else f"_channel_{channels or 'l'}"
        return BY_NAME[
            f"implicit_gemm_conv_{self.rank}d_{dtype}_bm{bm}_bn{bn}_bk16_wm{wm}"
            f"_wn{wn}{channel}_filter_{'s' if small else 'l'}"
        ]

    def reference(self, inputs, weights):
        """Exact channels-last outputs: every tap on an input pixel contributes.

        A tap lands on a pixel when its position in the dilated input is a
        multiple of ``input_dilation`` inside the input.
        """
        n_in = _row_major(self.input_shape)
        n_wt = _row_major(self.weight_shape)
        outputs = []
        for n in range(self.batch):
            for pixel in itertools.product(*(range(size) for size in self.out_spatial)):
                for o in range(self.outputs):
                    group = o // self.outputs_per_group
                    total = 0.0
                    for tap in itertools.product(*(range(k) for k in self.kernel)):
                        dilated = [
                            p * s - pad + t * d
                            for p, s, pad, t, d in zip(
                                pixel, self.stride, self.padding, tap, self.dilation
                            )
                        ]
                        if not all(
                            value >= 0 and value % idil == 0 and value // idil < size
                            for value, idil, size in zip(
                                dilated, self.input_dilation, self.spatial
                            )
                        ):
                            continue
                        position = [
                            value // idil
                            for value, idil in zip(dilated, self.input_dilation)
                        ]
                        weight_tap = [
                            k - 1 - t if self.flip else t
                            for k, t in zip(self.kernel, tap)
                        ]
                        base = n * n_in[0] + group * self.per_group
                        base += sum(v * s for v, s in zip(position, n_in[1:]))
                        wbase = o * n_wt[0]
                        wbase += sum(v * s for v, s in zip(weight_tap, n_wt[1:]))
                        for c in range(self.per_group):
                            total += inputs[base + c] * weights[wbase + c]
                    outputs.append(total)
        return outputs


def _row_major(shape):
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return list(reversed(strides))
