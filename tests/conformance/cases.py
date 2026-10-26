"""Compute hazards with independent source-rule oracles.

Every case is a compute kernel with input buffers at bindings 0 through n-1 and an output at binding n, written in one
source language (cases/<id>/src.<ext>). The oracle models the SOURCE language's rule under its reference
configuration; `readings` records, per reading of the source semantics, whether the rule creates an obligation
(HLSL: reading A = FXC/DXBC rules of D3D11.3; reading B = DXC 1.9 / DXIL; every other language: "spec").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import (
    constants_cases,
    control_flow_cases,
    conversion_cases,
    half_cases,
    integer_cases,
    pack_cases,
    warp_cases,
)

CASES_DIR = Path(__file__).resolve().parent / "cases"
EXT_LANG = {
    ".comp": "glsl",
    ".hlsl": "hlsl",
    ".metal": "metal",
    ".spvasm": "spirv",
    ".cu": "cuda",
    ".cl": "opencl",
    ".cgl": "cgl",
}
I32_MAX, I32_MIN = 2**31 - 1, -(2**31)
NAN, INF = float("nan"), float("inf")
CANARY = 0x7F7F7F7F


@dataclass
class Case:
    id: str
    cls: str
    title: str
    rule: str
    inputs: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray]
    out_dtype: type
    oracle: Callable | None = None  # inputs -> expected output array
    compare: str = "exact"  # exact | bits | ulp:<k> | ulp:<k>+sign
    threads: int = 64  # global size (threads)
    group: int = 64  # threadgroup / work-group size of the source
    out_count: int | None = None  # elements of the output buffer (default: oracle size)
    out_init: np.ndarray | None = None  # initial output contents (canaries)
    check: Callable | None = (
        None  # custom (inputs, out) -> (ok, detail) for parametric oracles
    )
    readings: dict = field(default_factory=lambda: {"spec": True})
    constants: bytes | None = (
        None  # constants block the host uploads, in the source's layout
    )
    v1: str | None = None
    note: str = ""
    lang: str = ""
    src: Path | None = None

    @property
    def input_buffers(self):
        """Arrays in binding order; keep single-array oracle and check arguments unchanged."""
        return (
            (self.inputs,)
            if isinstance(self.inputs, np.ndarray)
            else tuple(self.inputs)
        )

    def __post_init__(self):
        if not self.input_buffers or not all(
            isinstance(buffer, np.ndarray) for buffer in self.input_buffers
        ):
            raise ValueError(
                "Case inputs must be an array or a non-empty sequence of arrays"
            )
        if self.src is None:
            d = CASES_DIR / self.id
            srcs = sorted(p for p in d.iterdir() if p.suffix in EXT_LANG)
            assert len(srcs) == 1, (self.id, srcs)
            self.src, self.lang = srcs[0], EXT_LANG[srcs[0].suffix]
        elif not self.src.is_file() or EXT_LANG.get(self.src.suffix) != self.lang:
            raise ValueError(f"Invalid source fixture: {self.src}")
        if self.out_count is None:
            self.out_count = (
                len(self.oracle(self.inputs)) if self.oracle else self.threads
            )

    def expected(self):
        return self.oracle(self.inputs) if self.oracle else None


# ---------------------------------------------------------------------------------------------- helpers
def _tile(arr, n=64, width=1, dtype=None):
    arr = np.asarray(arr, dtype=dtype)
    reps = int(np.ceil(n * width / arr.size))
    return np.tile(arr, reps)[: n * width]


def _pairs(pairs, dtype, n=64):
    return _tile(np.array([v for p in pairs for v in p], dtype=dtype), n, 2)


def _wrap32(x):
    return ((np.asarray(x, dtype=np.int64) + 2**31) % 2**32 - 2**31).astype(np.int64)


def _i32_edge(n=64):
    base = np.array(
        [
            0,
            1,
            5,
            -1,
            -7,
            2**30,
            2**30 + 7,
            2_000_000_000,
            I32_MAX,
            I32_MAX - 1,
            I32_MAX - 3,
            I32_MIN,
            I32_MIN + 1,
            123456789,
            -123456789,
            1 << 24,
        ],
        dtype=np.int64,
    )
    return _tile(base, n).astype(np.int32)


# The words of an MSL ``struct { float3 a; float b; float2 c; }`` that hold values: a (0-2), b (4) and c (6-7).
_METAL_STRUCT_WORDS = np.array([0, 1, 2, 4, 6, 7])


def check_metal_struct(inputs, out):
    """The value words of each 8-word MSL struct, bit for bit; the padding words are unspecified."""
    r = inputs.reshape(64, 8).astype(np.float32)
    want = np.stack(
        [
            r[:, 2] + r[:, 4],
            r[:, 1] + r[:, 4] + np.float32(0.5),
            r[:, 0] + r[:, 4],
            r[:, 0] - r[:, 7],
            r[:, 4] * np.float32(2.0),
            r[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    got = out.astype(np.float32).reshape(64, 8)[:, _METAL_STRUCT_WORDS]
    bad = np.argwhere(got.view(np.uint32) != want.view(np.uint32))
    detail = {"mismatches": int(len(bad)), "n": int(want.size)}
    if len(bad):
        element, word = (int(value) for value in bad[0])
        detail.update(
            element=element,
            word=int(_METAL_STRUCT_WORDS[word]),
            got=repr(float(got[element, word])),
            want=repr(float(want[element, word])),
        )
    return len(bad) == 0, detail


# The words of a CUDA ``struct { float3 a; float2 b; }`` that hold values: a (0-2) and b (4-5).
_CUDA_STRUCT_WORDS = np.array([0, 1, 2, 4, 5])


def check_cuda_struct(inputs, out):
    """u_struct_layout: the value words of each 6-word CUDA struct, bit for bit; the padding word is unspecified."""
    r = inputs.reshape(64, 6).astype(np.float32)
    want = np.stack(
        [
            r[:, 2] + r[:, 4],
            r[:, 1],
            r[:, 0] - r[:, 5],
            r[:, 0] * np.float32(2.0) + np.float32(1.0),
            r[:, 5] + np.float32(0.5),
        ],
        axis=1,
    ).astype(np.float32)
    got = out.astype(np.float32).reshape(64, 6)[:, _CUDA_STRUCT_WORDS]
    bad = np.argwhere(got.view(np.uint32) != want.view(np.uint32))
    detail = {"mismatches": int(len(bad)), "n": int(want.size)}
    if len(bad):
        element, word = (int(value) for value in bad[0])
        detail.update(
            element=element,
            word=int(_CUDA_STRUCT_WORDS[word]),
            got=repr(float(got[element, word])),
            want=repr(float(want[element, word])),
        )
    return len(bad) == 0, detail


# The words of an MSL ``struct { float3 a; float b; }`` that hold values: a (0-2) and b (4).
_METAL_FLOAT3_FLOAT_WORDS = np.array([0, 1, 2, 4])


def check_metal_helper_struct(inputs, out):
    """m_float3_helper: element i gets b = v and a = (v, 0, 1) with v = src[i].a.y + src[i].b, bit for bit; the
    padding words are unspecified."""
    r = inputs.reshape(64, 8).astype(np.float32)
    v = (r[:, 1] + r[:, 4]).astype(np.float32)
    zero, one = np.zeros_like(v), np.ones_like(v)
    want = np.stack([v, zero, one, v], axis=1).astype(np.float32)
    got = out.astype(np.float32).reshape(64, 8)[:, _METAL_FLOAT3_FLOAT_WORDS]
    bad = np.argwhere(got.view(np.uint32) != want.view(np.uint32))
    detail = {"mismatches": int(len(bad)), "n": int(want.size)}
    if len(bad):
        element, word = (int(value) for value in bad[0])
        detail.update(
            element=element,
            word=int(_METAL_FLOAT3_FLOAT_WORDS[word]),
            got=repr(float(got[element, word])),
            want=repr(float(want[element, word])),
        )
    return len(bad) == 0, detail


def o_packed_helper(inp):
    """m_packed_helper: element i of the packed_float3 output is (z, y + 0.5, x) of input element i."""
    r = inp.reshape(64, 3).astype(np.float32)
    return (
        np.stack([r[:, 2], r[:, 1] + np.float32(0.5), r[:, 0]], axis=1)
        .reshape(-1)
        .astype(np.float32)
    )


def o_struct_stride(inp):
    r = inp.reshape(64, 7).astype(np.float32)
    out = np.stack(
        [
            r[:, 6],
            r[:, 3],
            r[:, 2] + np.float32(0.5),
            r[:, 1],
            r[:, 0] + r[:, 4],
            r[:, 1],
            r[:, 4],
        ],
        axis=1,
    )
    return out.reshape(-1).astype(np.float32)


def o_packed_mixed(inp):
    # Words of an element: a at 0..2, padding 3, s at 4, b at 5..7, c at 8, padding 9, d at 10..11.
    r = inp.reshape(64, 12).astype(np.float32)
    out = np.stack(
        [
            r[:, 2],
            r[:, 5],
            r[:, 7] + np.float32(0.5),
            r[:, 8],
            r[:, 11],
            r[:, 0] + r[:, 10] + r[:, 4],
        ],
        axis=1,
    )
    return out.reshape(-1).astype(np.float32)


def o_matrix_orientation(inp):
    """GLSL 4.60 sections 5.4.2, 5.6, 5.10 and 8.6 on mat2x3 values built from six inputs per thread."""
    out = []
    for i in range(inp.size // 6):
        a = inp[6 * i : 6 * i + 6].astype(np.float64)
        k = i & 1
        m = a.reshape(2, 3).T.copy()  # 3 rows, 2 columns; column c is m[:, c]
        mv = m @ np.array([1.0, 10.0])
        vm = np.array([1.0, 10.0, 100.0]) @ m
        c1, ck = m[:, 1].copy(), m[:, k].copy()
        d = np.eye(3) * a[0]
        dm = d @ mv
        row = [*mv, *vm, c1[0], c1[2], m[2, 0], dm[0], dm[2] + d[1, 0], ck[1]]
        m[:, 1] = m[:, 0] + c1
        m[1, 0] = 7.0
        row += [m[0, 1], m[2, 1], m[1, 0]]
        s = np.column_stack([m[0:2, 0], m[1:3, 1]])
        row.append((s @ np.array([1.0, 1000.0]))[0])
        t = m.T
        row.append(t[1, 2] + 2 * 100.0)
        row.append((m @ t)[0, 2])
        row.append((t @ m)[0, 1])
        m[:, k] = [a[5], 2.0, 3.0]
        row.append(m[0, 1] + m[1, 0] * 10.0)
        m[:, k] += [1.0, 20.0, 300.0]
        row.append(m[0, k] + m[2, k])
        op = np.outer(m[:, 0], [1.0, 10.0])
        row += [(m * m)[2, 1], op[2, 1], op[1, 0], m[1, 1] + 1000.0]
        out += row
    return np.array(out, dtype=np.float32)


def o_hlsl_matrix_orientation(inp):
    """HLSL floatRxC rows, row-filling constructors and initializer lists, _mRC swizzles, mul and casts."""
    out = []
    for i in range(inp.size // 6):
        a = inp[6 * i : 6 * i + 6].astype(np.float64)
        k = i & 1
        m = a.reshape(2, 3).copy()  # HLSL float2x3: two rows of three
        mv = m @ np.array([1.0, 10.0, 100.0])
        vm = np.array([1.0, 10.0]) @ m
        r1, rk = m[1].copy(), m[k].copy()
        row = [*mv, *vm, r1[0], r1[2], m[0, 2], rk[1], m[1, 2] + m[0, 0] * 10.0]
        row.append(m[0, 2] + m[1, 0] * 100.0)
        m[1] = m[0] + r1
        m[0, 1] = 7.0
        row += [m[1, 0], m[1, 2], m[0, 1]]
        m[k] = [a[5], 2.0, 3.0]
        row.append(m[1, 0] + m[0, 1] * 10.0)
        m[k] += [1.0, 20.0, 300.0]
        row.append(m[k, 0] + m[k, 2])
        row.append((m * m)[1, 2])
        t = m.T
        row += [(m @ t)[1, 0], (t @ m)[2, 0]]
        tr = m[:2, :2]
        row.append(tr[1, 1] + tr[0, 1] * 10.0)
        il = np.array([[a[0], 2.0], [3.0, a[1]]])
        row.append(il[0, 1] + il[1, 0] * 10.0 + il[1, 1] * 100.0)
        row.append(a[0] + a[0] * 10.0)
        m[0, 1] = 9.0
        row.append(m[0, 1])
        fm = np.array([[1.0, 2.0], [3.0, a[0]]])
        row.append(fm[1, 0] + fm[0, 1] * 10.0 + fm[1, 1] * 100.0)
        out += row
    return np.array(out, dtype=np.float32)


def o_matrix_expression_columns(inp):
    """GLSL 4.60 sections 5.6, 5.9 and 8.6: columns of matrix-valued expressions built from six inputs per thread."""
    out = []
    for i in range(inp.size // 6):
        a = inp[6 * i : 6 * i + 6].astype(np.float64)
        k = i & 1
        m = a.reshape(2, 3).T  # 3 rows, 2 columns; column c is m[:, c]
        n = np.array([[1.0, 2.0, 3.0], [a[5], a[0], 4.0]]).T
        c = np.array([a[0], a[2], a[4]])
        r = np.array([a[1], 10.0])
        q, s, t = (m / 2.0)[:, 1], (m + 1.0)[:, k], (10.0 - m)[:, 1]
        u, v = (m * n)[:, 1], np.outer(c, r)[:, k]
        x, y = (m - 1.0) @ r, c @ (m + n)
        out += [q[0], q[2], s[1], s[2], t[0], t[1], u[0], u[2], v[0], v[1], x[0], x[2]]
        out += [y[0], y[1], (m * 2.0)[1, 1], (2.0 * m - n)[2, k]]
        d = np.array([[2.0, 4.0, 0.5], [8.0, 1.0, 0.25]]).T * float(1 << k)
        h = (m + 1.0 - n - 0.5) / d
        out += [-m[0, 1], h[1, k], (4.0 / d)[2, 1], (m / d)[0, k]]
    return np.array(out, dtype=np.float32)


def _matrix_storage_input(n=64):
    """``n`` std430 mat2x3 values: two columns of three floats, each padded to four with -1000."""
    columns = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    out = []
    for i in range(n):
        for column in columns + np.float32(i % 4):
            out += [*column, -1000.0]
    return np.array(out, dtype=np.float32)


def o_matrix_storage(inp):
    """GLSL 4.60 sections 4.1.11 and 7.6.2.2 on std430 mat2x3 values and matrix initializer lists."""
    out = []
    for i in range(inp.size // 8):
        c0 = inp[8 * i : 8 * i + 3].astype(np.float64)
        c1 = inp[8 * i + 4 : 8 * i + 7].astype(np.float64)
        mv = c0 + 10.0 * c1
        gv = np.array([c0[0], c1[0]]) + 100.0 * np.array([c0[2], 5.0])
        hv = (
            np.array([c1[0], c1[1]])
            + 10.0 * np.array([c1[2], c0[0]])
            + 100.0 * np.array([c0[1], c0[2]])
        )
        out += [*mv, *gv, *hv, c1[2] + c0[1] * 10.0]
    return np.array(out, dtype=np.float32)


def o_add_wrap(inp):
    x = inp.astype(np.int64)
    return (_wrap32(x + 1) > x).astype(np.int32)


def o_mul_wrap(inp):
    x = inp.astype(np.int64)
    y = _wrap32(x * 2)
    return (np.trunc(y / 2).astype(np.int64) == x).astype(np.int32)


def o_loop_wrap(inp):
    out = []
    for s in inp.astype(np.int64):
        b, k, c = int(_wrap32(s + 4)), int(s), 0
        while k < b and c < 8:
            c, k = c + 1, k + 1
        out.append(c)
    return np.array(out, np.int32)


def _shift_pairs():
    return _pairs(
        [
            (1, 0),
            (1, 31),
            (1, 32),
            (1, 33),
            (3, 36),
            (0x80000001, 63),
            (0xFFFFFFFF, 64),
            (5, 1),
            (0x12345678, 40),
            (0xDEADBEEF, 255),
        ],
        np.uint32,
    )


def o_shift_xor(inp):
    out = []
    for x, s in inp.reshape(-1, 2).astype(np.int64):
        k = int(s) & 31
        out.append(((int(x) << k) & 0xFFFFFFFF) ^ (int(x) >> k))
    return np.array(out, np.uint64).astype(np.uint32)


def o_shl_mask(inp):
    return np.array(
        [(int(x) << (int(s) & 31)) & 0xFFFFFFFF for x, s in inp.reshape(-1, 2)],
        np.uint64,
    ).astype(np.uint32)


def o_udiv_d3d(inp):
    return np.array(
        [0xFFFFFFFF if b == 0 else int(a) // int(b) for a, b in inp.reshape(-1, 2)],
        np.uint64,
    ).astype(np.uint32)


def _cmp_inputs():
    return _tile(
        np.array([-1, 0, 3, 4, 5, I32_MIN, I32_MAX, -5], np.int32), dtype=np.int32
    )


def o_cmp_mixed(inp):
    return (inp.astype(np.int64).astype(np.uint32) < 4).astype(np.int32)


def _uchar_inputs():
    return _tile(
        np.array([0, 1, 254, 255, 511, -1, 127, 128, 300, 1000], np.int32),
        dtype=np.int32,
    )


def o_uchar(inp):
    return (((inp.astype(np.int64) & 255) + 1) & 255).astype(np.int32)


def _u64_inputs():
    return _tile(
        np.array(
            [0, 1, 2, 3, 0xFFFFFFFF, 0x80000000, 123456789, 4000000000], np.uint32
        ),
        dtype=np.uint32,
    )


def o_u64(inp):
    return np.array([(int(x) * 0xFFFFFFFF) >> 32 for x in inp], np.uint64).astype(
        np.uint32
    )


def _srem_pairs():
    return _pairs(
        [
            (-7, 3),
            (7, -3),
            (-7, -3),
            (7, 3),
            (-1, 5),
            (I32_MIN, 7),
            (I32_MAX, -10),
            (0, -3),
            (-9, 9),
            (5, 1),
        ],
        np.int32,
    )


def o_srem(inp):
    out = []
    for a, b in inp.reshape(-1, 2).astype(np.int64):
        q = abs(int(a)) // abs(int(b)) * (1 if (a >= 0) == (b > 0) else -1)
        out.append(int(a) - int(b) * q)
    return np.array(out, np.int32)


def _ftoi_inputs():
    return _tile(
        np.array(
            [NAN, INF, -INF, 3e9, -3e9, 2147483520.0, -2147483648.0, 2.7, -2.7, 0.5],
            np.float32,
        ),
        dtype=np.float32,
    )


def o_ftoi_d3d(inp):
    out = []
    for v in inp.astype(np.float64):
        if math.isnan(v):
            out.append(0)
        elif v >= 2147483648.0:
            out.append(I32_MAX)
        elif v <= -2147483649.0:
            out.append(I32_MIN)
        else:
            out.append(int(v))
    return np.array(out, np.int64).astype(np.int32)


def _ftoi_nan_inputs():
    neg_nan = np.array([0xFFC00000], np.uint32).view(np.float32)[0]
    return _tile(
        np.array(
            [NAN, neg_nan, 2.7, -2.7, 1e6, -1e6, 0.5, 2147483520.0, -2147483648.0, 0.0],
            np.float32,
        ),
        dtype=np.float32,
    )


def o_ftoi_nan0(inp):
    return np.array(
        [0 if math.isnan(v) else int(v) for v in inp.astype(np.float64)], np.int64
    ).astype(np.int32)


def _nan_pairs():
    return _pairs(
        [
            (NAN, 1.0),
            (1.0, NAN),
            (2.0, 3.0),
            (-4.0, NAN),
            (NAN, -2.5),
            (INF, 1.0),
            (5.0, -INF),
            (0.25, 0.5),
        ],
        np.float32,
    )


def o_fmin(inp):
    return np.fmin(inp[0::2], inp[1::2]).astype(np.float32)


def _contraction_inputs(n=64, seed=7):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 2.0, n).astype(np.float32)
    b = rng.uniform(1.0, 2.0, n).astype(np.float32)
    c = (a * b).astype(np.float32)
    return np.stack([a, b, c], axis=1).reshape(-1).astype(np.float32)


def o_unfused(inp):
    abc = inp.reshape(-1, 3)
    return ((abc[:, 0] * abc[:, 1]).astype(np.float32) - abc[:, 2]).astype(np.float32)


def o_sin(inp):
    return np.sin(inp.astype(np.float64)).astype(np.float32)


def _atan2_pairs():
    return _pairs(
        [
            (0.0, -1.0),
            (-0.0, -1.0),
            (0.0, 1.0),
            (-0.0, 1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.5, -2.0),
            (-0.5, -2.0),
        ],
        np.float32,
    )


def _atan2_zero_pairs():
    return _pairs(
        [
            (0.0, -0.0),
            (-0.0, -0.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (0.0, -2.0),
            (-0.0, -2.0),
            (0.0, 2.0),
            (-0.0, 2.0),
            (-3.0, 0.0),
            (3.0, -0.0),
            (1.0, 1.0),
            (-1.0, -1.0),
        ],
        np.float32,
    )


def o_atan2(inp):
    return np.array(
        [math.atan2(y, x) for y, x in inp.reshape(-1, 2).astype(np.float64)], np.float32
    )


def _denorm_inputs():
    return _tile(
        np.array(
            [
                1e-39,
                -1e-39,
                3e-40,
                2.35e-38,
                1.17549435e-38,
                1e-44,
                5.0,
                -2.5e-39,
                0.0,
                1.0,
            ],
            np.float32,
        ),
        dtype=np.float32,
    )


def o_half_exact(inp):
    return (inp.astype(np.float32) * np.float32(0.5)).astype(np.float32)


def _ftz(x):
    x = np.asarray(x, np.float32).copy()
    sub = (x != 0) & (np.abs(x) < np.float32(1.17549435e-38))
    x[sub] = np.copysign(np.float32(0.0), x[sub])
    return x


def o_half_ftz(inp):
    return _ftz(_ftz(inp) * np.float32(0.5))


def _isnan_inputs():
    return _tile(
        np.array([0.0, NAN, INF, -INF, -0.0, 1e-40, NAN, 3.5], np.float32),
        dtype=np.float32,
    )


def _fmod_pairs():
    return _pairs(
        [
            (-5.5, 2.0),
            (5.5, -2.0),
            (-7.0, 3.0),
            (7.25, 2.5),
            (-1e-3, 1.0),
            (9.5, 4.0),
            (-9.5, -4.0),
            (3.0, 7.0),
            (-0.0, 3.0),
            (1e7, 3.0),
        ],
        np.float32,
    )


def o_fmod(inp):
    return np.fmod(inp[0::2].astype(np.float64), inp[1::2].astype(np.float64)).astype(
        np.float32
    )


def o_half_mul(inp):
    return (
        (inp.astype(np.float16) * np.float16(3.0)).astype(np.float16).astype(np.float32)
    )


def o_fib(inp):
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    return np.array([fib(int(v)) for v in inp], np.int32)


def o_partial(inp, n=100, total=128):
    out = np.full(total, CANARY, np.int64)
    out[:n] = np.arange(n) * 3 + int(inp[0])
    return out.astype(np.int32)


def o_buffer_length(inp, header=4):
    """The input block's array starts ``header`` elements in; its length follows from the bound size."""
    n = len(inp) - header
    i = np.arange(len(inp))
    return (inp[header + i % n].astype(np.int64) + (n << 16)).astype(np.int32)


def o_buffer_count(inp):
    n = len(inp)
    return (np.roll(inp, -1).astype(np.int64) + (n << 16)).astype(np.int32)


def o_mirror(inp):
    """Each 64-element group reversed: workgroup memory is shared by the group."""
    return inp.reshape(-1, 64)[:, ::-1].reshape(-1).astype(np.int32)


def o_array_constants(inp):
    """Constant arrays of floats, of structs and of two-element vector arrays, indexed by bits of the input."""
    weights = [1.0, 2.0, 4.0, 8.0]
    table = [(1.5, 3), (-2.5, 7)]
    grid = [[(1.0, 0.5), (2.0, 0.25)], [(3.0, 4.0), (5.0, 6.0)]]
    out = []
    for k in inp.astype(np.int64):
        scale, bias = table[(k >> 2) & 1]
        gx, gy = grid[(k >> 3) & 1][(k >> 4) & 1]
        out.append(weights[k & 3] * scale + bias + gx * 16.0 + gy * 64.0)
    return np.array(out, dtype=np.float32)


def o_warp(inp):
    return inp.astype(np.int64).reshape(-1, 32).sum(axis=1).astype(np.int32)


def o_array_list_assignment(inp):
    """Brace lists that read the array they replace see its old elements: a rotation, a struct swap, a row swap."""
    out = []
    for x in inp.astype(np.int64):
        a = [x + 2, x, x + 1]
        q = [(x + 5) * 100 + 1, x * 100 + 1]
        g = 2 * 1000 + 3 * 100 + x * 10 + 1
        out.extend(a + q + [g, x * 4, x * 2 + x * 3])
    return np.array(out, dtype=np.int32)


def _width_check(inp, out, group=64, divergent=False):
    """Parametric oracle for SIMD/wave reductions: the reported width w must be uniform and every lane's sum must
    equal the sum over its SIMD-group (consecutive chunks of w threads within each threadgroup); with
    divergent=True, only lanes < 16 are active and the others keep -1."""
    out = out.astype(np.int64).reshape(-1, 2)
    widths = set(out[:, 1].tolist())
    if len(widths) != 1:
        return False, {"reason": f"non-uniform reported width {sorted(widths)[:5]}"}
    w = widths.pop()
    if w <= 0:
        return False, {"reason": f"reported width {w}"}
    x = inp.astype(np.int64)
    bad = 0
    for i in range(len(x)):
        t0 = (i // group) * group
        lane = (i - t0) % w
        g0 = t0 + ((i - t0) // w) * w
        g1 = min(g0 + w, t0 + group)
        if divergent:
            want = x[g0 : min(g1, g0 + 16)].sum() if lane < 16 else -1
        else:
            want = x[g0:g1].sum()
        bad += int(out[i, 0] != want)
    return bad == 0, {"width": int(w), "mismatches": bad, "n": len(x)}


RNG = np.random.default_rng(20261016)
SMALL = RNG.integers(-1000, 1000, 128).astype(np.int32)
R_SIN = np.concatenate([RNG.uniform(-1e5, 1e5, 48), RNG.uniform(-10, 10, 16)]).astype(
    np.float32
)

AB = {"A": True, "B": False}  # obligation under reading A (DXBC) only
BOTH = {"A": True, "B": True}  # obligation under both HLSL readings

CASES = [
    # ---- integer rules ---------------------------------------------------------------------------------
    Case(
        "g_add_wrap",
        "H1",
        "x + 1 > x with x near INT_MAX",
        "GLSL 4.60 §4.1.3 WRAP",
        _i32_edge(),
        np.int32,
        o_add_wrap,
        v1="int_add_overflow",
    ),
    Case(
        "g_mul_wrap",
        "H1",
        "(x * 2) / 2 == x",
        "GLSL 4.60 §4.1.3 WRAP",
        _i32_edge(),
        np.int32,
        o_mul_wrap,
        v1="int_mul_overflow",
    ),
    Case(
        "g_loop_wrap",
        "H1",
        "for (k = s; k < s + 4; ++k)",
        "GLSL 4.60 §4.1.3 WRAP",
        _i32_edge(),
        np.int32,
        o_loop_wrap,
        v1="int_loop_overflow",
    ),
    Case(
        "h_add_wrap",
        "H1",
        "x + 1 > x",
        "A: D3D11.3 §22.12.1 iadd WRAP; B: DXIL add nsw",
        _i32_edge(),
        np.int32,
        o_add_wrap,
        readings=AB,
        v1="int_add_overflow",
    ),
    Case(
        "h_mul_wrap",
        "H1",
        "(x * 2) / 2 == x",
        "A: D3D11.3 §22.12.6 imul WRAP; B: DXIL mul nsw",
        _i32_edge(),
        np.int32,
        o_mul_wrap,
        readings=AB,
        v1="int_mul_overflow",
    ),
    Case(
        "h_loop_wrap",
        "H1",
        "loop bound s + 4",
        "A: DXBC WRAP; B: DXIL add nsw",
        _i32_edge(),
        np.int32,
        o_loop_wrap,
        readings=AB,
        v1="int_loop_overflow",
    ),
    Case(
        "s_add_wrap",
        "H1",
        "OpIAdd without NoSignedWrap",
        "SPIR-V §3.3.13 OpIAdd",
        _i32_edge(),
        np.int32,
        o_add_wrap,
    ),
    Case(
        "s_mul_wrap",
        "H1",
        "OpIMul without NoSignedWrap",
        "SPIR-V §3.3.13 OpIMul",
        _i32_edge(),
        np.int32,
        o_mul_wrap,
    ),
    Case(
        "h_shift_mask",
        "H2",
        "x << s with s >= 32",
        "D3D11.3 §22.11.7 ishl; DXC masks (SPIR-V.rst)",
        _shift_pairs(),
        np.uint32,
        o_shl_mask,
        readings=BOTH,
        v1="h_shift_mask",
    ),
    Case(
        "m_shift_mask",
        "H2",
        "(x << s) ^ (x >> s) with s >= 32",
        "MSL 4.1 §3.1 item 11 MASK",
        _shift_pairs(),
        np.uint32,
        o_shift_xor,
    ),
    Case(
        "c_shift_mask",
        "H2",
        "(x << s) ^ (x >> s) with s >= 32",
        "OpenCL C §6.5.10 MASK",
        _shift_pairs(),
        np.uint32,
        o_shift_xor,
    ),
    Case(
        "h_udiv_zero",
        "H3",
        "a / 0u",
        "A: D3D11.3 §22.12.9 0xFFFFFFFF; B: DXIL udiv (n/f)",
        _pairs(
            [
                (7, 0),
                (0, 0),
                (10, 3),
                (0xFFFFFFFF, 0),
                (100, 7),
                (5, 0),
                (1, 1),
                (9, 2),
            ],
            np.uint32,
        ),
        np.uint32,
        o_udiv_d3d,
        readings=AB,
        v1="h_udiv_zero",
    ),
    Case(
        "m_cmp_mixed",
        "H4",
        "int < uint",
        "C++17 usual arithmetic conversions (MSL §3.1)",
        _cmp_inputs(),
        np.int32,
        o_cmp_mixed,
        v1="m_signed_unsigned_cmp",
    ),
    Case(
        "g_cmp_mixed",
        "H4",
        "int < uint",
        "GLSL 4.60 §4.1.10",
        _cmp_inputs(),
        np.int32,
        o_cmp_mixed,
    ),
    Case(
        "h_cmp_mixed",
        "H4",
        "int < uint",
        "HLSL draft §5.1 [Expr.conv]",
        _cmp_inputs(),
        np.int32,
        o_cmp_mixed,
        readings=BOTH,
    ),
    Case(
        "u_cmp_mixed",
        "H4",
        "int < unsigned",
        "C++ [expr.arith.conv]",
        _cmp_inputs(),
        np.int32,
        o_cmp_mixed,
    ),
    Case(
        "c_cmp_mixed",
        "H4",
        "int < uint",
        "C99 §6.3.1.8",
        _cmp_inputs(),
        np.int32,
        o_cmp_mixed,
    ),
    Case(
        "m_uchar_wrap",
        "H15",
        "uchar v = x; v = v + 1",
        "C++ integral conversion mod 256",
        _uchar_inputs(),
        np.int32,
        o_uchar,
        v1="m_uchar_wrap",
    ),
    Case(
        "u_uchar_wrap",
        "H15",
        "unsigned char v = x; v = v + 1",
        "C++ integral conversion mod 256",
        _uchar_inputs(),
        np.int32,
        o_uchar,
    ),
    Case(
        "c_uchar_wrap",
        "H15",
        "uchar v = x; v = v + 1",
        "C99 §6.3.1.3",
        _uchar_inputs(),
        np.int32,
        o_uchar,
    ),
    Case(
        "m_ulong_mul",
        "H15",
        "high word of ulong(x) * 0xFFFFFFFF",
        "C++ unsigned 64-bit arithmetic",
        _u64_inputs(),
        np.uint32,
        o_u64,
        v1="m_ulong_mul",
    ),
    Case(
        "u_u64_mul",
        "H15",
        "high word of x * 0xFFFFFFFFull",
        "C++ unsigned 64-bit arithmetic",
        _u64_inputs(),
        np.uint32,
        o_u64,
    ),
    Case(
        "u_srem_sign",
        "H3",
        "a % b with negative operands",
        "C++ [expr.mul]/4",
        _srem_pairs(),
        np.int32,
        o_srem,
    ),
    Case(
        "m_srem_sign",
        "H3",
        "a % b with negative operands",
        "MSL §3.1 (C++ rules)",
        _srem_pairs(),
        np.int32,
        o_srem,
    ),
    # ---- floating-point rules --------------------------------------------------------------------------
    Case(
        "h_ftoi_sat",
        "H5",
        "(int)x for NaN, Inf, out of range",
        "A: D3D11.3 §22.13.3 ftoi; B: DXIL fptosi",
        _ftoi_inputs(),
        np.int32,
        o_ftoi_d3d,
        readings=AB,
        v1="h_ftoi_sat",
    ),
    Case(
        "m_ftoi_nan",
        "H5",
        "int(NaN) == 0",
        "MSL 4.1 §8.6",
        _ftoi_nan_inputs(),
        np.int32,
        o_ftoi_nan0,
    ),
    Case(
        "c_convert_sat",
        "H5",
        "convert_int_sat(x)",
        "OpenCL C §6.4.3.3",
        _ftoi_inputs(),
        np.int32,
        o_ftoi_d3d,
    ),
    Case(
        "c_convert_uint_sat_rte",
        "H5",
        "convert_uint_sat_rte(x)",
        "OpenCL C §6.4.3.3",
        conversion_cases.uint_rte_inputs(),
        np.uint32,
        conversion_cases.uint_rte_reference,
    ),
    Case(
        "c_convert_char_sat",
        "H5",
        "convert_char_sat(x)",
        "OpenCL C §6.4.3.3",
        conversion_cases.char_inputs(),
        np.int32,
        conversion_cases.char_reference,
    ),
    Case(
        "h_min_nan",
        "H6",
        "min(x, NaN)",
        "D3D11.3 §3.1.3.1 minimumNumber; DXIL FMin",
        _nan_pairs(),
        np.float32,
        o_fmin,
        compare="bits",
        readings=BOTH,
        v1="h_min_nan",
    ),
    Case(
        "m_fmin_nan",
        "H6",
        "fmin(x, NaN)",
        "MSL 4.1 Table 6.4 (safe math)",
        _nan_pairs(),
        np.float32,
        o_fmin,
        compare="bits",
        v1="m_fmin_nan",
    ),
    Case(
        "u_fminf_nan",
        "H6",
        "fminf(x, NaN)",
        "CUDA Math API fminf",
        _nan_pairs(),
        np.float32,
        o_fmin,
        compare="bits",
    ),
    Case(
        "c_fmin_nan",
        "H6",
        "fmin(x, NaN)",
        "OpenCL C §6.15.2",
        _nan_pairs(),
        np.float32,
        o_fmin,
        compare="bits",
    ),
    Case(
        "s_nmin_nan",
        "H6",
        "NMin(x, NaN) with SignedZeroInfNanPreserve",
        "GLSL.std.450 NMin; Vulkan float controls",
        _nan_pairs(),
        np.float32,
        o_fmin,
        compare="bits",
    ),
    Case(
        "g_precise",
        "H7",
        "precise r = a*b - c",
        "GLSL 4.60 §4.9",
        _contraction_inputs(),
        np.float32,
        o_unfused,
        v1="fp_precise_contraction",
    ),
    Case(
        "h_precise",
        "H7",
        "precise r = a*b - c",
        "HLSL precise (DXIL)",
        _contraction_inputs(),
        np.float32,
        o_unfused,
        readings=BOTH,
        v1="fp_precise_contraction",
    ),
    Case(
        "m_contract_off",
        "H7",
        "#pragma METAL fp contract(off): a*b - c",
        "MSL 4.1 §1.6.3",
        _contraction_inputs(),
        np.float32,
        o_unfused,
    ),
    Case(
        "s_nocontraction",
        "H7",
        "NoContraction a*b - c",
        "SPIR-V NoContraction",
        _contraction_inputs(),
        np.float32,
        o_unfused,
    ),
    Case(
        "u_fmul_rn",
        "H7",
        "__fadd_rn(__fmul_rn(a, b), -c)",
        "CUDA Math API intrinsics",
        _contraction_inputs(),
        np.float32,
        o_unfused,
    ),
    Case(
        "c_fp_contract_off",
        "H7",
        "FP_CONTRACT OFF: a*b - c",
        "OpenCL C §6.15.2.1",
        _contraction_inputs(),
        np.float32,
        o_unfused,
    ),
    Case(
        "m_precise_sin",
        "H8",
        "precise::sin(x), x in [1e2, 1e5]",
        "MSL precise sin (bound used: 4 ulp)",
        np.random.default_rng(3).uniform(1e2, 1e5, 64).astype(np.float32),
        np.float32,
        o_sin,
        compare="ulp:4",
        v1="m_precise_sin",
    ),
    Case(
        "m_atan2",
        "H8/H10",
        "precise::atan2(y, x) with signed zeros",
        "MSL precise atan2 (6 ulp); C99 Annex F",
        _atan2_pairs(),
        np.float32,
        o_atan2,
        compare="ulp:6+sign",
        v1="m_atan2_signed_zero",
    ),
    Case(
        "u_sinf",
        "H8",
        "sinf(x), |x| <= 1e5",
        "CUDA Math API sinf: 2 ulp",
        R_SIN,
        np.float32,
        o_sin,
        compare="ulp:2",
    ),
    Case(
        "c_sin",
        "H8",
        "sin(x), |x| <= 1e5",
        "OpenCL C §7.4: 4 ulp",
        R_SIN,
        np.float32,
        o_sin,
        compare="ulp:4",
    ),
    Case(
        "u_atan2f_zero",
        "H10",
        "atan2f with signed zeros",
        "CUDA Math API atan2f special values",
        _atan2_zero_pairs(),
        np.float32,
        o_atan2,
        compare="ulp:2+sign",
    ),
    Case(
        "u_denorm",
        "H9",
        "x * 0.5f for subnormal x",
        "nvcc default -ftz=false",
        _denorm_inputs(),
        np.float32,
        o_half_exact,
        compare="bits",
    ),
    Case(
        "h_denorm",
        "H9",
        "x * 0.5 for subnormal x",
        "A: D3D11.3 fp32 flush-to-zero; B: DXIL default 'any'",
        _denorm_inputs(),
        np.float32,
        o_half_ftz,
        compare="bits",
        readings=AB,
    ),
    Case(
        "m_isnan",
        "isnan",
        "isnan(x)",
        "MSL (safe math)",
        _isnan_inputs(),
        np.int32,
        lambda inp: np.isnan(inp).astype(np.int32),
    ),
    Case(
        "g_isnan",
        "isnan",
        "isnan(x)",
        "GLSL 4.60 §8.3: false if NaNs are not implemented (negative control)",
        _isnan_inputs(),
        np.int32,
        lambda inp: np.isnan(inp).astype(np.int32),
        readings={"spec": False},
        v1="g_isnan",
    ),
    Case(
        "m_fmod_sign",
        "fmod",
        "fmod(x, y) sign of x",
        "C/MSL fmod",
        _fmod_pairs(),
        np.float32,
        o_fmod,
        compare="bits",
        v1="m_fmod_sign",
    ),
    Case(
        "u_fmodf_sign",
        "fmod",
        "fmodf(x, y) sign of x",
        "CUDA fmodf (C99)",
        _fmod_pairs(),
        np.float32,
        o_fmod,
        compare="bits",
    ),
    Case(
        "m_half_mul",
        "H15",
        "half(x) * 3.0h",
        "MSL half (binary16)",
        _tile(
            np.array(
                [
                    0.1,
                    1 / 3,
                    1000.7,
                    21844.3,
                    1e-5,
                    -2.5,
                    3.14159,
                    0.7,
                    65000.0 / 3,
                    7e-3,
                ],
                np.float32,
            ),
            dtype=np.float32,
        ),
        np.float32,
        o_half_mul,
        compare="ulp:0",
        v1="m_half_mul",
    ),
    Case(
        "m_half_div",
        "H15",
        "half(x) / half(y), quotients next to a binary16 midpoint",
        "MSL half (binary16), correctly rounded division",
        half_cases.div_inputs(),
        np.float32,
        None,
        threads=512,
        out_count=512,
        check=half_cases.check(half_cases.div_reference),
    ),
    Case(
        "m_half_sqrt",
        "H15",
        "sqrt(half(x)) over [1, 4) and random normal values",
        "MSL half (binary16), correctly rounded square root",
        half_cases.sqrt_inputs(),
        np.float32,
        None,
        threads=2560,
        out_count=2560,
        check=half_cases.check(half_cases.sqrt_reference),
    ),
    Case(
        "m_half_fma",
        "H15",
        "fma(half(x), half(y), half(z)), sums next to a binary16 midpoint",
        "MSL half (binary16), IEEE 754 fusedMultiplyAdd",
        half_cases.fma_inputs(),
        np.float32,
        None,
        threads=512,
        out_count=512,
        check=half_cases.check(half_cases.fma_reference),
    ),
    Case(
        "m_half_round",
        "H15",
        "half(x) for x on and next to binary16 midpoints",
        "MSL half (binary16), conversion rounds to nearest even",
        half_cases.round_inputs(),
        np.float32,
        None,
        threads=512,
        out_count=512,
        check=half_cases.check(half_cases.round_reference),
    ),
    # ---- layout and memory -----------------------------------------------------------------------------
    Case(
        "m_packed_float3",
        "H16",
        "packed_float3: 12-byte stride",
        "MSL packed_float3",
        np.arange(64 * 3, dtype=np.float32),
        np.float32,
        lambda inp: inp[1::3].astype(np.float32),
        compare="ulp:0",
        v1="m_packed_float3",
    ),
    Case(
        "m_packed_struct",
        "H16",
        "struct with packed_float3 and packed_float2 members: 28-byte elements",
        "MSL packed vector alignment",
        np.arange(64 * 7, dtype=np.float32),
        np.float32,
        o_struct_stride,
        compare="ulp:0",
    ),
    Case(
        "m_packed_mixed",
        "H16",
        "struct mixing float3 and packed_float3 members: 48-byte elements, packed member at 20",
        "MSL float3 and packed vector alignment",
        np.arange(64 * 12, dtype=np.float32),
        np.float32,
        o_packed_mixed,
        compare="ulp:0",
    ),
    Case(
        "m_float3_stride",
        "H16",
        "float3: 16-byte stride",
        "MSL float3 size and alignment",
        np.arange(64 * 4, dtype=np.float32),
        np.float32,
        lambda inp: inp[1::4].astype(np.float32),
        compare="ulp:0",
    ),
    Case(
        "m_float3x3_stride",
        "H16",
        "float3x3: 16-byte columns, 48-byte stride; a column read and a matrix-vector product",
        "MSL matrix size, alignment and column order",
        np.arange(64 * 12, dtype=np.float32),
        np.float32,
        lambda inp: (
            lambda r: r[:, 8]
            + r[:, 1]
            + np.float32(2.0) * r[:, 5]
            + np.float32(3.0) * r[:, 9]
        )(inp.reshape(64, 12).astype(np.float32)).astype(np.float32),
        compare="ulp:0",
    ),
    Case(
        "m_packed_helper",
        "H16",
        "packed_float3 pointers passed to helper functions, with an element offset: 12-byte stride",
        "MSL packed_float3",
        np.arange(64 * 3, dtype=np.float32),
        np.float32,
        o_packed_helper,
        compare="ulp:0",
    ),
    Case(
        "m_float3_helper",
        "H16",
        "struct {float3 a; float b;} pointers passed to helper functions, with an element offset: 32-byte elements",
        "MSL natural struct layout",
        np.arange(64 * 8, dtype=np.float32),
        np.float32,
        None,
        out_count=64 * 8,
        check=check_metal_helper_struct,
    ),
    Case(
        "m_float3_struct",
        "H16",
        "struct {float3 a; float b; float2 c;}: b at offset 16, c at 24, 32-byte elements",
        "MSL natural struct layout",
        np.arange(64 * 8, dtype=np.float32),
        np.float32,
        None,
        out_count=64 * 8,
        check=check_metal_struct,
    ),
    Case(
        "h_float3_stride",
        "H16",
        "StructuredBuffer<float3>: 12-byte stride",
        "HLSL structured buffer layout",
        np.arange(64 * 3, dtype=np.float32),
        np.float32,
        lambda inp: inp[1::3].astype(np.float32),
        compare="ulp:0",
        readings=BOTH,
    ),
    Case(
        "h_struct_stride",
        "H16",
        "StructuredBuffer<P>: float3 and float2 members at 4-byte offsets, 28-byte elements",
        "HLSL structured buffer layout",
        np.arange(64 * 7, dtype=np.float32),
        np.float32,
        o_struct_stride,
        compare="ulp:0",
        readings=BOTH,
    ),
    Case(
        "u_float3_stride",
        "H16",
        "float3 array: 12-byte elements",
        "CUDA float3",
        np.arange(64 * 3, dtype=np.float32),
        np.float32,
        lambda inp: inp[1::3].astype(np.float32),
        compare="ulp:0",
    ),
    Case(
        "u_struct_layout",
        "H16",
        "struct {float3 a; float2 b;}: b at offset 16, 24-byte elements",
        "CUDA built-in vector alignment",
        np.arange(64 * 6, dtype=np.float32),
        np.float32,
        None,
        out_count=64 * 6,
        check=check_cuda_struct,
    ),
    Case(
        "g_std430_vec3_float",
        "H16",
        "struct {vec3 a; float b;}: b at offset 12",
        "GLSL std430 rules",
        np.arange(64 * 4, dtype=np.float32),
        np.float32,
        lambda inp: inp[3::4].astype(np.float32),
        compare="ulp:0",
    ),
    Case(
        "g_matrix_orientation",
        "S11",
        "mat2x3 constructors, columns, products, built-ins and mat3(s)",
        "GLSL 4.60 sections 5.4.2, 5.6, 5.10 and 8.6",
        np.tile(np.arange(1, 7, dtype=np.float32), 64)
        + np.repeat(np.arange(64) % 4, 6).astype(np.float32),
        np.float32,
        o_matrix_orientation,
        compare="ulp:0",
    ),
    Case(
        "h_matrix_orientation",
        "S11",
        "float2x3 rows, row-filling constructors, _mRC swizzles, mul, * and casts",
        "HLSL matrix type, per-component math and mul references",
        np.tile(np.arange(1, 7, dtype=np.float32), 64)
        + np.repeat(np.arange(64) % 4, 6).astype(np.float32),
        np.float32,
        o_hlsl_matrix_orientation,
        compare="ulp:0",
    ),
    Case(
        "g_matrix_expression_columns",
        "S11",
        "columns of mat2x3 sums, quotients, negations, compound updates, matrixCompMult and outerProduct results",
        "GLSL 4.60 sections 5.6, 5.9 and 8.6",
        np.tile(np.arange(1, 7, dtype=np.float32), 64)
        + np.repeat(np.arange(64) % 4, 6).astype(np.float32),
        np.float32,
        o_matrix_expression_columns,
        compare="ulp:0",
    ),
    Case(
        "g_matrix_storage",
        "S11",
        "std430 mat2x3 storage loads and matrix initializer lists",
        "GLSL 4.60 sections 4.1.11 and 7.6.2.2",
        _matrix_storage_input(),
        np.float32,
        o_matrix_storage,
        compare="ulp:0",
    ),
    Case(
        "h_oob_read",
        "H11",
        "a[i + 60] with 64 elements",
        "A: D3D11.3 out-of-bounds reads return 0",
        np.arange(1, 65, dtype=np.float32),
        np.float32,
        lambda inp: np.array(
            [inp[i + 60] if i + 60 < 64 else 0.0 for i in range(64)], np.float32
        ),
        compare="ulp:0",
        readings=AB,
    ),
    Case(
        "u_ptr_arith",
        "H17",
        "p = inp + 3; p[i]",
        "C++ pointer arithmetic",
        np.arange(67, dtype=np.int32) * 7,
        np.int32,
        lambda inp: inp[3:67].astype(np.int32),
    ),
    Case(
        "m_ptr_arith",
        "H17",
        "p = inp + 3; p[i]",
        "MSL device pointers",
        np.arange(67, dtype=np.int32) * 7,
        np.int32,
        lambda inp: inp[3:67].astype(np.int32),
    ),
    Case(
        "c_ptr_arith",
        "H17",
        "p = inp + 3; p[i]",
        "OpenCL C global pointers",
        np.arange(67, dtype=np.int32) * 7,
        np.int32,
        lambda inp: inp[3:67].astype(np.int32),
    ),
    # ---- execution model ---------------------------------------------------------------------------------
    Case(
        "m_partial_tg",
        "H12",
        "dispatchThreads(100) with threadgroup 64",
        "MSL non-uniform threadgroups",
        np.array([1] + [0] * 63, np.int32),
        np.int32,
        o_partial,
        threads=100,
        out_count=128,
        out_init=np.full(128, CANARY, np.int32),
    ),
    Case(
        "c_partial_wg",
        "H12",
        "global size 100, work-group 64",
        "OpenCL C 2.0 non-uniform work-groups",
        np.array([1] + [0] * 63, np.int32),
        np.int32,
        o_partial,
        threads=100,
        out_count=128,
        out_init=np.full(128, CANARY, np.int32),
    ),
    Case(
        "u_warp_reduce",
        "H13",
        "__shfl_down_sync warp reduction",
        "CUDA warpSize = 32",
        SMALL,
        np.int32,
        o_warp,
        threads=128,
        out_count=4,
    ),
    Case(
        "h_groupshared",
        "H13",
        "groupshared array written and read across the thread group",
        "HLSL groupshared: one copy per thread group",
        SMALL,
        np.int32,
        o_mirror,
        threads=128,
    ),
    Case(
        "g_shared_exchange",
        "H13",
        "shared array written and read across the work group",
        "GLSL 4.60 section 4.3.8: shared variables are shared by the work group",
        SMALL,
        np.int32,
        o_mirror,
        threads=128,
    ),
    Case(
        "g_buffer_length",
        "H13",
        "length() of a runtime array after a buffer block's header",
        "GLSL 4.60 section 4.1.9: a runtime array's length follows from the buffer size",
        SMALL,
        np.int32,
        o_buffer_length,
        threads=128,
    ),
    Case(
        "s_buffer_length",
        "H13",
        "OpArrayLength on storage blocks",
        "SPIR-V OpArrayLength: the element count of the bound buffer's runtime array",
        SMALL,
        np.int32,
        o_buffer_length,
        threads=128,
    ),
    Case(
        "g_buffer_count",
        "H13",
        "length() of a buffer block that holds only a runtime array",
        "GLSL 4.60 section 4.1.9: a runtime array's length follows from the buffer size",
        SMALL,
        np.int32,
        o_buffer_count,
        threads=128,
    ),
    Case(
        "s_buffer_count",
        "H13",
        "OpArrayLength on a storage block that holds only a runtime array",
        "SPIR-V OpArrayLength: the element count of the bound buffer's runtime array",
        SMALL,
        np.int32,
        o_buffer_count,
        threads=128,
    ),
    Case(
        "x_warp_shuffle",
        "H13",
        "CUDA warp shuffles called from CrossGL in a one-warp workgroup",
        "CUDA warp = 32 consecutive threads; width segments; own value out of range",
        SMALL,
        np.int32,
        warp_cases.shuffle_reference,
        threads=128,
        group=32,
        out_count=128 * warp_cases.RESULTS,
    ),
    Case(
        "x_array_list_assignment",
        "composite",
        "Brace lists assigned to int, struct and nested arrays whose elements read the array being assigned",
        "GLSL 4.60 section 5.8: an assignment stores the value of its right operand, so a list's elements "
        "are all computed before the array changes, as for the equivalent array constructor",
        np.arange(-20, 44, dtype=np.int32),
        np.int32,
        o_array_list_assignment,
    ),
    Case(
        "m_simd_sum",
        "H13",
        "simd_sum and threads_per_simdgroup",
        "MSL SIMD-group functions (width impl.-defined)",
        SMALL,
        np.int32,
        None,
        threads=128,
        out_count=256,
        check=lambda i, o: _width_check(i, o),
    ),
    Case(
        "h_wave_sum",
        "H13",
        "WaveActiveSum and WaveGetLaneCount",
        "HLSL SM 6.0 wave intrinsics",
        SMALL,
        np.int32,
        None,
        threads=128,
        out_count=256,
        check=lambda i, o: _width_check(i, o),
        readings=BOTH,
    ),
    Case(
        "m_simd_divergent",
        "H14",
        "simd_sum in SIMD-divergent control flow",
        "MSL: reduces over active threads",
        SMALL,
        np.int32,
        None,
        threads=128,
        out_count=256,
        check=lambda i, o: _width_check(i, o, divergent=True),
    ),
    Case(
        "u_recursion",
        "H18",
        "recursive __device__ fib",
        "CUDA recursion",
        _tile(np.arange(16, dtype=np.int32), dtype=np.int32),
        np.int32,
        o_fib,
    ),
    Case(
        "g_pack",
        "pack",
        "packUnorm/packSnorm 4x8 and 2x16, packHalf2x16",
        "GLSL 4.60 §8.4: round(clamp(c) * scale), first component in the low bits",
        pack_cases.pack_inputs(),
        np.uint32,
        pack_cases.pack_reference,
    ),
    Case(
        "g_unpack",
        "pack",
        "unpackUnorm/unpackSnorm 4x8 and 2x16, unpackHalf2x16",
        "GLSL 4.60 §8.4: f / scale clamped to [-1, 1]; division within 2.5 ULP",
        pack_cases.unpack_inputs(),
        np.float32,
        pack_cases.unpack_reference,
        compare="ulp:3",
    ),
    Case(
        "g_bitfield",
        "bitfield",
        "bitfieldExtract and bitfieldInsert with run-time offsets and widths",
        "GLSL 4.60 §8.8: width 0 extracts 0 and inserts nothing; signed extracts sign-extend",
        pack_cases.bitfield_inputs(),
        np.uint32,
        pack_cases.bitfield_reference,
    ),
    Case(
        "g_mix_select",
        "select",
        "mix(x, y, bvec) with infinite and NaN operands",
        "GLSL 4.60 §8.3: a boolean selector selects; the other operand does not reach the result",
        pack_cases.mix_select_inputs(),
        np.float32,
        pack_cases.mix_select_reference,
        compare="bits",
    ),
    Case(
        "g_frexp_ldexp_modf",
        "exponent",
        "frexp, ldexp and modf on normal values and zero",
        "GLSL 4.60 §8.3: significand in [0.5, 1) and exponent, x * 2^e, parts with the sign of x",
        pack_cases.exponent_inputs(),
        np.uint32,
        pack_cases.exponent_reference,
    ),
    Case(
        "s_frexp_ldexp_modf",
        "exponent",
        "FrexpStruct, ModfStruct and Ldexp with struct results read by OpCompositeExtract",
        "GLSL.std.450: member 0 is the significand or fraction, member 1 the exponent or whole part",
        pack_cases.exponent_inputs(),
        np.uint32,
        pack_cases.exponent_reference,
    ),
    Case(
        "h_frexp_ldexp_modf",
        "exponent",
        "frexp and ldexp with float exponents, modf into an int",
        "HLSL intrinsics: frexp exponent and ldexp x * 2^e as floats; exp2 within 3 + 2|e| ULP (Vulkan 1.3)",
        pack_cases.hlsl_exponent_inputs(),
        np.float32,
        pack_cases.hlsl_exponent_reference,
        compare="ulp:8",
    ),
    Case(
        "m_bit_scans",
        "integer",
        "clz and ctz of uint, int and vector components, qualified and unqualified",
        "MSL 3.2 §6.13: leading or trailing zero bits per component; a zero component counts its bit width",
        integer_cases.bit_scan_inputs(),
        np.uint32,
        integer_cases.metal_bit_scan_reference,
    ),
    Case(
        "g_bit_scans",
        "integer",
        "findMSB and findLSB of uint, int and vector components",
        "GLSL 4.60 §8.8: bit index or -1 for zero; findMSB of a negative int finds the highest clear bit",
        integer_cases.bit_scan_inputs(),
        np.uint32,
        integer_cases.glsl_bit_scan_reference,
    ),
    Case(
        "g_extended_arithmetic",
        "integer",
        "uaddCarry, usubBorrow, umulExtended and imulExtended, scalar and vector",
        "GLSL 4.60 §8.8: sum and difference modulo 2^32 with carry or borrow; high and low words of the product",
        integer_cases.integer_pair_inputs(),
        np.uint32,
        integer_cases.extended_arithmetic_reference,
    ),
    Case(
        "s_extended_arithmetic",
        "integer",
        "OpIAddCarry, OpISubBorrow, OpUMulExtended and OpSMulExtended read by OpCompositeExtract",
        "SPIR-V 1.6 §3.52.13: member 0 is the low word, sum or difference, member 1 the high word, carry or borrow",
        integer_cases.integer_pair_inputs(),
        np.uint32,
        integer_cases.extended_arithmetic_reference,
    ),
    Case(
        "s_integer_signedness",
        "integer",
        "Shifts, comparisons, division, remainder, modulus and conversions whose opcode signedness differs from the operand type",
        "SPIR-V 1.6 §3.3.6: the opcode, not the operand type, decides the signedness of an integer operation",
        integer_cases.integer_pair_inputs(),
        np.uint32,
        integer_cases.signedness_reference,
    ),
    Case(
        "s_structured_control_flow",
        "control",
        "Nested loops with continue and break, a fall-through switch, a do-while, swapped loop values, a "
        "read-then-overwrite and a return from an endless loop, in function variables",
        "SPIR-V 1.6 §2.11: merge instructions declare the selection, switch and loop constructs",
        control_flow_cases.structured_control_flow_inputs(),
        np.uint32,
        control_flow_cases.structured_control_flow_reference,
    ),
    Case(
        "s_structured_control_flow_phi",
        "control",
        "The s_structured_control_flow kernel after spirv-opt -O, with its loop values and switch result in OpPhi",
        "SPIR-V 1.6 §2.11 and §3.52.17: an OpPhi takes the value of the edge control arrives by",
        control_flow_cases.structured_control_flow_inputs(),
        np.uint32,
        control_flow_cases.structured_control_flow_reference,
    ),
    Case(
        "s_array_constants",
        "composite",
        "OpConstantComposite of array, struct and nested array types, read with dynamic indices",
        "SPIR-V 1.6 §3.52.7: OpConstantComposite of an array or structure type takes its constituents in order",
        np.arange(64, dtype=np.uint32) % 32,
        np.float32,
        o_array_constants,
    ),
    Case(
        "g_short_circuit",
        "control",
        "&&, || and ?: whose unselected operands call a function that writes its inout argument, in "
        "straight-line code, a while condition and a do-while condition, and an index guarded by a bounds test",
        "GLSL 4.60 section 5.9: && and || evaluate the right operand only when the left one leaves the "
        "result open; ?: evaluates only the selected expression",
        control_flow_cases.short_circuit_inputs(),
        np.int32,
        control_flow_cases.short_circuit_reference,
        threads=control_flow_cases.SHORT_CIRCUIT_THREADS,
    ),
    Case(
        "u_atomic_add",
        "atomics",
        "atomicAdd(&out[0], 1) by 256 threads",
        "CUDA atomicAdd (sanity)",
        np.ones(256, np.int32),
        np.int32,
        lambda inp: np.array([int(inp.sum())], np.int32),
        threads=256,
        out_count=1,
        out_init=np.zeros(1, np.int32),
    ),
    Case(
        "g_std140_uniform",
        "S20",
        "std140 uniform block: vec3+float slot, 16-byte array strides, mat3 columns, slot-aligned tail",
        "GLSL 4.60 section 7.6.2.2 (std140)",
        constants_cases.inputs(),
        np.float32,
        constants_cases.glsl_reference,
        compare="ulp:0",
        constants=constants_cases.block(constants_cases.STD140),
    ),
    Case(
        "s_std140_uniform",
        "S20",
        "std140 Uniform Block: Offset, ArrayStride and MatrixStride decorations from glslang",
        "GLSL 4.60 section 7.6.2.2 (std140) as decorated by glslang",
        constants_cases.inputs(),
        np.float32,
        constants_cases.glsl_reference,
        compare="ulp:0",
        constants=constants_cases.block(constants_cases.STD140),
    ),
    Case(
        "s_push_constant_std430",
        "S20",
        "std430 push constants: 8-byte vec2 and 4-byte float strides, mat3 columns, member after an array",
        "GLSL 4.60 section 7.6.2.3 (std430) as decorated by glslang",
        constants_cases.inputs(),
        np.float32,
        constants_cases.glsl_reference,
        compare="ulp:0",
        constants=constants_cases.block(constants_cases.STD430),
    ),
    Case(
        "h_cbuffer_packing",
        "S20",
        "cbuffer packing: unpadded last array element, scalar after float3x3, register-aligned arrays",
        "HLSL packing rules for constant variables",
        constants_cases.inputs(),
        np.float32,
        constants_cases.hlsl_reference,
        compare="ulp:0",
        constants=constants_cases.block(constants_cases.HLSL),
    ),
    Case(
        "h_cbuffer_row_major",
        "S20",
        "cbuffer packing: row_major float3x3 rows, scalar packed after the last row, unpadded array tail",
        "HLSL packing rules for constant variables; row_major matrix packing",
        constants_cases.inputs(),
        np.float32,
        constants_cases.hlsl_reference,
        compare="ulp:0",
        constants=constants_cases.block(constants_cases.HLSL_ROW_MAJOR),
    ),
    Case(
        "m_constant_struct",
        "S20",
        "constant struct: 16-byte float3, packed_float3, 8-byte float2 stride, float3x3 columns",
        "MSL 4.1 chapter 2 size and alignment of vector, packed vector and matrix types",
        constants_cases.inputs(),
        np.float32,
        constants_cases.msl_reference,
        compare="ulp:0",
        constants=constants_cases.block(constants_cases.MSL),
    ),
]

# MSL 4.1 section 3.1 leaves negative remainder undefined. Vulkan's float-control
# rules do not establish NMin's NaN result merely from SignedZeroInfNanPreserve.
# Keep both as observational controls rather than claiming source obligations.
CATALOGUE_CORRECTIONS = {
    "m_srem_sign": {"A": False, "B": False},
    "s_nmin_nan": {"A": False, "B": False},
}
for _c in CASES:
    if _c.id in CATALOGUE_CORRECTIONS:
        _c.readings = dict(CATALOGUE_CORRECTIONS[_c.id])

BY_ID = {c.id: c for c in CASES}


def compare(case: Case, got: bytes):
    """Return (ok, detail) for a raw output buffer."""
    arr = np.frombuffer(
        got[: case.out_count * np.dtype(case.out_dtype).itemsize], dtype=case.out_dtype
    )
    if arr.size < case.out_count:
        return False, {"reason": "short output"}
    if case.check is not None:
        return case.check(case.inputs, arr)
    want = case.expected().astype(case.out_dtype)
    mode = case.compare
    sign = mode.endswith("+sign")
    mode = mode.replace("+sign", "")
    if mode in ("exact",):
        bad = arr != want
    elif mode == "bits":
        a32, b32 = arr.astype(np.float32), want.astype(np.float32)
        bad = (a32.view(np.uint32) != b32.view(np.uint32)) & ~(
            np.isnan(a32) & np.isnan(b32)
        )
    else:
        k = int(mode.split(":")[1])
        a32, b32 = arr.astype(np.float32), want.astype(np.float32)
        a = a32.view(np.int32).astype(np.int64)
        b = b32.view(np.int32).astype(np.int64)
        # ULP distance on the monotone integer line (sign-magnitude -> two's complement ordering)
        a = np.where(a < 0, -(a & 0x7FFFFFFF), a)
        b = np.where(b < 0, -(b & 0x7FFFFFFF), b)
        both_nan = np.isnan(a32) & np.isnan(b32)
        bad = (np.abs(a - b) > k) & ~both_nan
        bad |= np.isnan(a32) != np.isnan(b32)
    if sign:  # a zero result must carry the oracle's sign bit
        a32, b32 = arr.astype(np.float32), want.astype(np.float32)
        zero = b32 == 0
        bad |= zero & (np.signbit(a32) != np.signbit(b32))
    bad = np.nonzero(bad)[0]
    detail = {"mismatches": int(bad.size), "n": int(want.size)}
    if bad.size:
        i = int(bad[0])
        detail.update(first_index=i, got=repr(arr[i].item()), want=repr(want[i].item()))
    return bad.size == 0, detail
