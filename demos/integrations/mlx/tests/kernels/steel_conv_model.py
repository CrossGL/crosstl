"""Geometry and arithmetic of the pinned MLX ``steel_conv`` kernels.

``steel_conv.metal`` instantiates ``implicit_gemm_conv_2d`` for three element
types, six block shapes and six loader selections. ``conv.cpp`` turns a 2-D
convolution into an implicit GEMM: ``M`` output pixels, ``N`` output channels
per group and ``K`` filter taps times input channels per group. This module
names the entries, models the threadgroup slots each loader writes and
reproduces the host parameters ``implicit_gemm_conv_2D_gpu`` encodes.
"""

from dataclasses import dataclass

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
FILTERS = (
    ("l", 0, "s", True),
    ("l", 0, "l", False),
    ("1", 1, "l", False),
    ("2", 2, "l", False),
    ("3", 3, "l", False),
    ("4", 4, "l", False),
)
SIMD_WIDTH = 32
# ``ChannelHelper`` widens 3 channels to 4 and keeps 1, 2 and 4.
CHANNEL_VECTORS = {1: 1, 2: 2, 3: 4, 4: 4}


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
    """One instantiated ``implicit_gemm_conv_2d`` entry point."""

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
    def name(self):
        return (
            f"implicit_gemm_conv_2d_{self.dtype}_bm{self.bm}_bn{self.bn}"
            f"_bk{self.bk}_wm{self.wm}_wn{self.wn}"
            f"_channel_{self.channel_name}_filter_{self.filter_name}"
        )

    @property
    def threads(self):
        return self.wm * self.wn * SIMD_WIDTH

    @property
    def workgroup_size(self):
        return (SIMD_WIDTH, self.wn, self.wm)

    @property
    def padding(self):
        # steel_conv.h pads each threadgroup row by 16 bytes.
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
        per_group = conv.channels // conv.groups
        if self.channels:
            return per_group == self.channels
        if per_group % self.bk:
            return False
        return not self.small_filter or max(conv.kernel) <= 16


def entries():
    """Every instantiated entry, in source order."""
    return tuple(
        ConvEntry(dtype, metal_type, size, *block, *selection)
        for dtype, metal_type, size in TYPES
        for block in BLOCKS
        for selection in FILTERS
    )


BY_NAME = {entry.name: entry for entry in entries()}


@dataclass(frozen=True)
class Conv:
    """One 2-D convolution in MLX's NHWC input and OHWI weight layouts."""

    batch: int
    spatial: tuple
    kernel: tuple
    channels: int
    outputs: int
    groups: int = 1
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    flip: bool = False

    @property
    def out_spatial(self):
        return tuple(
            (size + 2 * pad - dil * (taps - 1) - 1) // stride + 1
            for size, taps, stride, pad, dil in zip(
                self.spatial, self.kernel, self.stride, self.padding, self.dilation
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
        """``MLXConvParams<2>`` as ``conv.cpp`` fills it for row-major arrays."""
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
            idil=[1, 1],
            in_strides=_row_major(self.input_shape),
            wt_strides=_row_major(self.weight_shape),
            out_strides=_row_major(self.output_shape),
            groups=self.groups,
            flip=self.flip,
        )

    def gemm_params(self, entry):
        """``ImplicitGemmConv2DParams`` as ``implicit_gemm_conv_2D_gpu`` builds it."""
        oh, ow = self.out_spatial
        kh, kw = self.kernel
        implicit_m = self.batch * oh * ow
        implicit_n = self.outputs_per_group
        implicit_k = kh * kw * self.per_group
        bk = entry.bk
        if self.per_group <= 2:
            k_iterations = -(-implicit_k // bk)
        elif self.per_group <= 4:
            k_iterations = -(-(kh * kw * 4) // bk)
        else:
            k_iterations = kh * kw * -(-self.per_group // bk)
        sign = -1 if self.flip else 1
        in_strides = _row_major(self.input_shape)
        ijw = in_strides[2] * self.dilation[1]
        ijh = in_strides[1] * self.dilation[0]
        return dict(
            M=implicit_m,
            N=implicit_n,
            K=implicit_k,
            gemm_k_iterations=k_iterations,
            inp_jump_w=sign * ijw,
            inp_jump_h=sign * (ijh - (kw - 1) * ijw),
            inp_jump_c=bk - sign * (kh - 1) * ijh - sign * (kw - 1) * ijw,
            tiles_n=-(-implicit_n // entry.bn),
            tiles_m=-(-implicit_m // entry.bm),
            swizzle_log=0,
        )

    def host_entry(self, dtype):
        """The entry ``implicit_gemm_conv_2D_gpu`` selects for ``dtype``."""
        oh, ow = self.out_spatial
        implicit_m = self.batch * oh * ow
        implicit_n = self.outputs_per_group
        wm, wn = 2, 2
        bm = 64 if implicit_m >= 8192 and self.per_group >= 64 else 32
        bn = 64 if bm == 64 or implicit_n >= 64 else 32
        if implicit_n <= 16:
            bn, wm, wn = 8, 4, 1
        channels = self.per_group if self.per_group <= 4 else 0
        small = not channels and max(self.kernel) <= 16
        return BY_NAME[
            f"implicit_gemm_conv_2d_{dtype}_bm{bm}_bn{bn}_bk16_wm{wm}_wn{wn}"
            f"_channel_{channels or 'l'}_filter_{'s' if small else 'l'}"
        ]

    def reference(self, inputs, weights):
        """Exact NHWC outputs: every tap that lands in the input contributes."""
        n_in = _row_major(self.input_shape)
        n_wt = _row_major(self.weight_shape)
        oh, ow = self.out_spatial
        kh, kw = self.kernel
        outputs = []
        for n in range(self.batch):
            for y in range(oh):
                for x in range(ow):
                    for o in range(self.outputs):
                        group = o // self.outputs_per_group
                        total = 0.0
                        for ty in range(kh):
                            iy = y * self.stride[0] - self.padding[0]
                            iy += ty * self.dilation[0]
                            if not 0 <= iy < self.spatial[0]:
                                continue
                            wy = kh - 1 - ty if self.flip else ty
                            for tx in range(kw):
                                ix = x * self.stride[1] - self.padding[1]
                                ix += tx * self.dilation[1]
                                if not 0 <= ix < self.spatial[1]:
                                    continue
                                wx = kw - 1 - tx if self.flip else tx
                                base = n * n_in[0] + iy * n_in[1] + ix * n_in[2]
                                base += group * self.per_group
                                wbase = o * n_wt[0] + wy * n_wt[1] + wx * n_wt[2]
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
