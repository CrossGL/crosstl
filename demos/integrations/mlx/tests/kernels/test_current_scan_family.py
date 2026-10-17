"""Native parity for every pinned MLX ``scan.metal`` kernel on all three backends.

``scan.metal`` instantiates ``contiguous_scan`` and ``strided_scan`` for 57
element-type and operation helpers, each inclusive and exclusive, forward and
reverse. A model of both kernels, which follows their block loops, SIMD-group
scans, threadgroup carries and guarded loads and stores step by step, gives
the expected words of every case. Sums and products whose SIMD-group scans use
Apple's prefix intrinsics take inputs whose every partial result is exact, so
any association order gives the same words; the explicit shuffle scans of
8-byte types, extrema and ``logaddexp`` run in the kernel's own order. The
model is checked against an independent left fold of each scanned line.
"""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
)
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _element_type,
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/scan.metal"
SOURCE_SHA256 = "49faeb1a36f346b2aa0f68a5f5fe2544af2109fec5357425550f97a79926b6f3"
HEADERS = {
    "mlx/backend/metal/kernels/scan.h": (
        "735e545d1296f48d0bddffaff289c4e4ef77c5f76cb2bace861a706c43fc24a0"
    ),
    "mlx/backend/metal/kernels/binary_ops.h": (
        "6dbe789faafbf7259f411c7fab7c0fcdd192df47a78f8f0bd408228f7a30e5c7"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
    "mlx/backend/metal/kernels/complex.h": (
        "16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1"
    ),
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SCAN_FAMILY"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/scan.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")

# instantiate_scan_helper(name, itype, otype, op, nreads) rows of scan.metal,
# in source order. N_READS is 2 for the 8-byte types and 4 otherwise.
HELPERS = (
    ("sum_bool__int32", "bool_", "int32", "sum"),
    ("sum_bool__uint32", "bool_", "uint32", "sum"),
    ("sum_uint8_uint8", "uint8", "uint8", "sum"),
    ("sum_uint16_uint16", "uint16", "uint16", "sum"),
    ("sum_uint32_uint32", "uint32", "uint32", "sum"),
    ("sum_uint64_uint64", "uint64", "uint64", "sum"),
    ("sum_int8_int8", "int8", "int8", "sum"),
    ("sum_int16_int16", "int16", "int16", "sum"),
    ("sum_int32_int32", "int32", "int32", "sum"),
    ("sum_int64_int64", "int64", "int64", "sum"),
    ("sum_float16_float16", "float16", "float16", "sum"),
    ("sum_float32_float32", "float32", "float32", "sum"),
    ("sum_bfloat16_bfloat16", "bfloat16", "bfloat16", "sum"),
    ("sum_complex64_complex64", "complex64", "complex64", "sum"),
    ("prod_bool__bool_", "bool_", "bool_", "prod"),
    ("prod_uint8_uint8", "uint8", "uint8", "prod"),
    ("prod_uint16_uint16", "uint16", "uint16", "prod"),
    ("prod_uint32_uint32", "uint32", "uint32", "prod"),
    ("prod_uint64_uint64", "uint64", "uint64", "prod"),
    ("prod_int8_int8", "int8", "int8", "prod"),
    ("prod_int16_int16", "int16", "int16", "prod"),
    ("prod_int32_int32", "int32", "int32", "prod"),
    ("prod_int64_int64", "int64", "int64", "prod"),
    ("prod_float16_float16", "float16", "float16", "prod"),
    ("prod_float32_float32", "float32", "float32", "prod"),
    ("prod_bfloat16_bfloat16", "bfloat16", "bfloat16", "prod"),
    ("prod_complex64_complex64", "complex64", "complex64", "prod"),
    ("max_bool__bool_", "bool_", "bool_", "max"),
    ("max_uint8_uint8", "uint8", "uint8", "max"),
    ("max_uint16_uint16", "uint16", "uint16", "max"),
    ("max_uint32_uint32", "uint32", "uint32", "max"),
    ("max_uint64_uint64", "uint64", "uint64", "max"),
    ("max_int8_int8", "int8", "int8", "max"),
    ("max_int16_int16", "int16", "int16", "max"),
    ("max_int32_int32", "int32", "int32", "max"),
    ("max_int64_int64", "int64", "int64", "max"),
    ("max_float16_float16", "float16", "float16", "max"),
    ("max_float32_float32", "float32", "float32", "max"),
    ("max_bfloat16_bfloat16", "bfloat16", "bfloat16", "max"),
    ("max_complex64_complex64", "complex64", "complex64", "max"),
    ("min_bool__bool_", "bool_", "bool_", "min"),
    ("min_uint8_uint8", "uint8", "uint8", "min"),
    ("min_uint16_uint16", "uint16", "uint16", "min"),
    ("min_uint32_uint32", "uint32", "uint32", "min"),
    ("min_uint64_uint64", "uint64", "uint64", "min"),
    ("min_int8_int8", "int8", "int8", "min"),
    ("min_int16_int16", "int16", "int16", "min"),
    ("min_int32_int32", "int32", "int32", "min"),
    ("min_int64_int64", "int64", "int64", "min"),
    ("min_float16_float16", "float16", "float16", "min"),
    ("min_float32_float32", "float32", "float32", "min"),
    ("min_bfloat16_bfloat16", "bfloat16", "bfloat16", "min"),
    ("min_complex64_complex64", "complex64", "complex64", "min"),
    ("logaddexp_float16_float16", "float16", "float16", "logaddexp"),
    ("logaddexp_float32_float32", "float32", "float32", "logaddexp"),
    ("logaddexp_bfloat16_bfloat16", "bfloat16", "bfloat16", "logaddexp"),
    ("logaddexp_complex64_complex64", "complex64", "complex64", "logaddexp"),
)
KINDS = ("contig", "strided")
# instantiate_scan_helper order: (inclusive, reverse) per kernel.
VARIANTS = (
    ("inclusive_", True, False),
    ("exclusive_", False, False),
    ("reverse_inclusive_", True, True),
    ("reverse_exclusive_", False, True),
)
WIDE_DTYPES = ("uint64", "int64", "complex64")
ITEM_SIZES = {
    "bool_": 1,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "float32": 4,
    "bfloat16": 2,
    "complex64": 8,
}
INTEGER_DTYPES = {
    "uint8": (8, False),
    "uint16": (16, False),
    "uint32": (32, False),
    "uint64": (64, False),
    "int8": (8, True),
    "int16": (16, True),
    "int32": (32, True),
    "int64": (64, True),
}
FLOAT_DTYPES = ("float16", "float32", "bfloat16")
# Sums and products of types narrower than 8 bytes use Apple's
# simd_prefix_{inclusive,exclusive}_{sum,product}; everything else scans
# with the kernel's explicit simd_shuffle_and_fill_up loop.
SIMD_SIZE = 32
CONTIG_WIDTH = 128
SIGNED_MAX = 2**31 - 1
GRID = (1, 3, 1)
GUARD_COUNT = 16
# Host bounds every translation checks before a dispatch.
METADATA_BOUNDS = {"axis_size": 4096, "stride": 4096, "stride_blocks": 128}
# OpenGL indexes 32-bit storage arrays. The reverse kernels address their
# blocks from the end of each scanned line, so a pointer may sit before the
# line; only guarded accesses dereference it. These host preconditions state
# that every executed access index fits signed int range, which the kernel
# model checks access by access.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "contig-reverse": ("input + i", "out_ + i"),
    "strided-reverse": (
        "index_y * stride + i",
        "in_ + index_y * stride + i",
        "out_ + index_y * stride + i",
    ),
}
# Cases per kernel. Contiguous kernels scan three rows of ``axis_size`` with a
# 128-thread group: one element, a partial block, exactly one block of
# N_READS * 128 elements and two blocks plus a tail, which carries the prefix
# across the block loop. Strided kernels scan axis 1 of an
# ``(outer, axis_size, stride)`` array with MLX's group of
# (32 / N_READS) * 32 threads; ``outer * ceil(stride / 32)`` is three groups.
CONTIG_CASES = ("single", "short", "block", "carry")
STRIDED_CASES = {
    "single": (3, 1, 7),
    "narrow": (3, 37, 5),
    "tile": (3, 64, 32),
    "wide": (1, 33, 70),
}
# logaddexp rounds inexactly; one case per kernel takes dense inputs and a
# derived error bound, the others values whose results are exact.
DENSE_CASES = {"contig": "short", "strided": "narrow"}


def _helper_entries(name):
    return tuple(
        f"{kind}_scan_{variant}{name}" for kind in KINDS for variant, *_ in VARIANTS
    )


ENTRIES = tuple(entry for name, *_ in HELPERS for entry in _helper_entries(name))
INFO = {}
for _name, _itype, _otype, _op in HELPERS:
    for _kind in KINDS:
        for _variant, _inclusive, _reverse in VARIANTS:
            INFO[f"{_kind}_scan_{_variant}{_name}"] = {
                "helper": _name,
                "kind": _kind,
                "itype": _itype,
                "otype": _otype,
                "op": _op,
                "inclusive": _inclusive,
                "reverse": _reverse,
                "nreads": 2 if _otype in WIDE_DTYPES else 4,
            }


def _workgroup(entry):
    info = INFO[entry]
    if info["kind"] == "contig":
        return [CONTIG_WIDTH, 1, 1]
    return [(SIMD_SIZE // info["nreads"]) * SIMD_SIZE, 1, 1]


def _contig_axis(info, case):
    block = info["nreads"] * CONTIG_WIDTH
    return {"single": 1, "short": 37, "block": block, "carry": 2 * block + 5}[case]


def _cases(entry):
    return CONTIG_CASES if INFO[entry]["kind"] == "contig" else tuple(STRIDED_CASES)


def _geometry(entry, case):
    """``(lines, axis_size, stride, stride_blocks, element_count)``."""
    info = INFO[entry]
    if info["kind"] == "contig":
        axis = _contig_axis(info, case)
        return GRID[1], axis, 1, 1, GRID[1] * axis
    outer, axis, stride = STRIDED_CASES[case]
    blocks = -(-stride // 32)
    assert outer * blocks == GRID[1]
    return outer * stride, axis, stride, blocks, outer * axis * stride


def _parameters(entry):
    if INFO[entry]["kind"] == "contig":
        return ("axis_size",)
    return ("axis_size", "stride", "stride_blocks")


def _index_assertions(entry):
    info = INFO[entry]
    if not info["reverse"]:
        return ()
    return OPENGL_INDEX_RANGE_ASSERTIONS[f"{info['kind']}-reverse"]


def _config(directory, target, entry):
    """The translation of one entry, as the gate and its contract run it."""
    width = _workgroup(entry)[0]
    region = plan_dispatch_regions((width * GRID[0], GRID[1], GRID[2]), (width, 1, 1))[
        0
    ].to_json()
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        workgroup_size_rules={SOURCE: (width, 1, 1)},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=SIGNED_MAX)
            for expression in (_index_assertions(entry) if target == "opengl" else ())
        ],
        metadata_preconditions=[
            dict(
                source=SOURCE,
                entryPoint=entry,
                parameter=parameter,
                minimum=1,
                maximum=METADATA_BOUNDS[parameter],
            )
            for parameter in _parameters(entry)
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "target_options": {
                    "directx": {
                        "software_subgroup_width": SIMD_SIZE,
                        "relative_wave_shuffle_out_of_range": "self",
                        "binary32_additive_profile": "native-precise",
                        # contiguous_scan reads all 32 simdgroup_sums slots
                        # after writing one per SIMD-group.
                        "groupshared_bool_storage": "uint",
                        "dispatch_region": region,
                    },
                    "opengl": {
                        "software_subgroup_width": SIMD_SIZE,
                        "dispatch_region": region,
                    },
                },
            }
        },
    )


# ---------------------------------------------------------------------------
# Element arithmetic with Metal semantics
# ---------------------------------------------------------------------------


class _Garbage:
    """Unspecified threadgroup memory; every combination with it stays so."""

    def __repr__(self):
        return "GARBAGE"


GARBAGE = _Garbage()
NAN = float("nan")
INF = float("inf")


def _wrap(dtype, value):
    bits, signed = INTEGER_DTYPES[dtype]
    value &= (1 << bits) - 1
    return value - (1 << bits) if signed and value >> (bits - 1) else value


def _bfloat(value):
    """Round to bfloat16 (ties to even), kept as a float32 scalar."""
    word = int(np.array(np.float32(value)).view(np.uint32))
    if math.isnan(float(value)):
        word = 0x7FC00000
    else:
        word = (word + 0x7FFF + ((word >> 16) & 1)) >> 16 << 16
    return np.array(word & 0xFFFFFFFF, dtype=np.uint32).view(np.float32)[()]


def _real(dtype, value):
    """A logical floating value rounded to ``dtype``."""
    with np.errstate(all="ignore"):
        if dtype == "float16":
            return np.float16(value)
        if dtype == "float32":
            return np.float32(value)
        return _bfloat(np.float32(value))


def _quiet_nan(dtype):
    return _real(dtype, NAN)


def _is_nan(value):
    if isinstance(value, tuple):
        return any(_is_nan(part) for part in value)
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _float_add(dtype, a, b):
    with np.errstate(all="ignore"):
        if dtype == "bfloat16":
            return _bfloat(np.float32(a) + np.float32(b))
        return a + b


def _float_mul(dtype, a, b):
    with np.errstate(all="ignore"):
        if dtype == "bfloat16":
            return _bfloat(np.float32(a) * np.float32(b))
        return a * b


def _complex_mul(a, b):
    with np.errstate(all="ignore"):
        return (
            a[0] * b[0] - a[1] * b[1],
            a[0] * b[1] + a[1] * b[0],
        )


def _complex_ge(a, b):
    return a[0] > b[0] or (a[0] == b[0] and a[1] >= b[1])


def _complex_gt(a, b):
    return a[0] > b[0] or (a[0] == b[0] and a[1] > b[1])


def _limits(dtype, maximum):
    if dtype == "bool_":
        return maximum
    if dtype in INTEGER_DTYPES:
        bits, signed = INTEGER_DTYPES[dtype]
        if signed:
            return (1 << (bits - 1)) - 1 if maximum else -(1 << (bits - 1))
        return (1 << bits) - 1 if maximum else 0
    if dtype == "complex64":
        part = np.float32(INF if maximum else -INF)
        return (part, part)
    return _real(dtype, INF if maximum else -INF)


def _init(info):
    dtype, op = info["otype"], info["op"]
    if op == "sum":
        if dtype == "complex64":
            return (np.float32(0.0), np.float32(0.0))
        return _real(dtype, 0.0) if dtype in FLOAT_DTYPES else 0
    if op == "prod":
        if dtype == "bool_":
            return True
        if dtype == "complex64":
            return (np.float32(1.0), np.float32(0.0))
        return _real(dtype, 1.0) if dtype in FLOAT_DTYPES else 1
    # CumMax, CumLogaddexp: Limits<U>::min; CumMin: Limits<U>::max.
    return _limits(dtype, op == "min")


def _convert(info, value):
    """The ``T -> U`` conversion of ``values[i] = input[i]``."""
    if info["itype"] == "bool_" and info["otype"] != "bool_":
        return int(bool(value))
    if info["otype"] in FLOAT_DTYPES:
        return _real(info["otype"], value)
    if info["otype"] == "complex64":
        return (np.float32(value[0]), np.float32(value[1]))
    return value


def _logaddexp(dtype, x, y):
    """LogAddExp from binary_ops.h, evaluated in binary64 and rounded."""
    if math.isnan(float(x)) or math.isnan(float(y)):
        return _quiet_nan(dtype)
    maximum, minimum = max(float(x), float(y)), min(float(x), float(y))
    if minimum == -INF or maximum == INF:
        return _real(dtype, maximum)
    return _real(dtype, maximum + math.log1p(math.exp(minimum - maximum)))


def _complex_log1p(x, y):
    """``log1p(complex64_t)`` from utils.h, in binary64."""
    theta = math.atan2(y, x + 1)
    if math.sqrt(x * x + y * y) < 0.5:
        r = x * (2 + x) + y * y
        if r == 0:
            return x, theta
        return 0.5 * math.log1p(r), theta
    return math.log(math.sqrt((x + 1) * (x + 1) + y * y)), theta


def _complex_logaddexp(x, y):
    if any(math.isnan(float(part)) for part in (*x, *y)):
        # quiet_NaN() converts to complex64_t(NaN, 0).
        return (np.float32(NAN), np.float32(0.0))
    maximum = x if _complex_gt(x, y) else y
    minimum = x if _complex_gt(y, x) else y
    if float(minimum[0]) == -INF or float(maximum[0]) == INF:
        return maximum
    m = math.exp(float(minimum[0]) - float(maximum[0]))
    delta = float(minimum[1]) - float(maximum[1])
    real, imag = _complex_log1p(m * math.cos(delta), m * math.sin(delta))
    return (
        np.float32(float(maximum[0]) + real),
        np.float32(float(maximum[1]) + imag),
    )


def _operation(info):
    """``Op::operator()`` and ``combine`` for one entry, Metal semantics."""
    dtype, op = info["otype"], info["op"]

    def guarded(function):
        def apply(a, b):
            if a is GARBAGE or b is GARBAGE:
                return GARBAGE
            return function(a, b)

        return apply

    if op == "sum":
        if dtype in INTEGER_DTYPES:
            return guarded(lambda a, b: _wrap(dtype, a + b))
        if dtype == "complex64":
            return guarded(
                lambda a, b: (
                    _float_add("float32", a[0], b[0]),
                    _float_add("float32", a[1], b[1]),
                )
            )
        return guarded(lambda a, b: _float_add(dtype, a, b))
    if op == "prod":
        if dtype == "bool_":
            return guarded(lambda a, b: bool(a) and bool(b))
        if dtype in INTEGER_DTYPES:
            return guarded(lambda a, b: _wrap(dtype, a * b))
        if dtype == "complex64":
            return guarded(_complex_mul)
        return guarded(lambda a, b: _float_mul(dtype, a, b))
    if op in {"max", "min"}:
        maximum = op == "max"

        def combine(a, b):
            if dtype in FLOAT_DTYPES and (_is_nan(a) or _is_nan(b)):
                return _quiet_nan(dtype)
            if dtype == "complex64":
                keep = _complex_ge(a, b) if maximum else _complex_ge(b, a)
            else:
                keep = a >= b if maximum else a <= b
            return a if keep else b

        return guarded(combine)
    assert op == "logaddexp"
    if dtype == "complex64":
        return guarded(_complex_logaddexp)
    return guarded(lambda a, b: _logaddexp(dtype, a, b))


def _simd_inclusive(op, init, lanes):
    """``simd_scan``: the explicit fill-up loop (also exact for intrinsics)."""
    values = list(lanes)
    shift = 1
    while shift <= 16:
        values = [
            op(values[lane], values[lane - shift] if lane >= shift else init)
            for lane in range(SIMD_SIZE)
        ]
        shift *= 2
    return values


def _simd_exclusive(op, init, lanes):
    values = _simd_inclusive(op, init, lanes)
    return [init] + values[:-1]


# ---------------------------------------------------------------------------
# Kernel models
# ---------------------------------------------------------------------------


def _record(log, kind, index, limit):
    assert 0 <= index < limit, (kind, index, limit)
    log.append((kind, index))


def _contiguous_scan(info, arithmetic, inputs, axis, width):
    """``contiguous_scan`` for the three rows of the gate's dispatch grid."""
    n_reads = info["nreads"]
    op, init, convert = arithmetic
    simd_groups = width // SIMD_SIZE
    rows = GRID[1]
    size = rows * axis
    output = [None] * size
    log = []
    for row in range(rows):
        base = row * axis
        prefix = init
        sums = [GARBAGE] * SIMD_SIZE
        for block in range(-(-axis // (n_reads * width))):
            offsets = [block * width * n_reads + lid * n_reads for lid in range(width)]
            values = []
            for offset in offsets:
                loaded = [None] * n_reads
                for i in range(n_reads):
                    if info["reverse"]:
                        # load_unsafe/load_safe at in + axis - offset - N_READS.
                        if offset + n_reads - i - 1 < axis:
                            index = base + axis - offset - n_reads + i
                            _record(log, "read", index, size)
                            loaded[n_reads - i - 1] = convert(inputs[index])
                        else:
                            loaded[n_reads - i - 1] = init
                    elif offset + i < axis:
                        _record(log, "read", base + offset + i, size)
                        loaded[i] = convert(inputs[base + offset + i])
                    else:
                        loaded[i] = init
                for i in range(1, n_reads):
                    loaded[i] = op(loaded[i], loaded[i - 1])
                values.append(loaded)
            prev_thread = []
            for group in range(simd_groups):
                lanes = [values[group * SIMD_SIZE + lane][-1] for lane in range(32)]
                prev_thread += _simd_exclusive(op, init, lanes)
            for group in range(simd_groups):
                last = group * SIMD_SIZE + SIMD_SIZE - 1
                sums[group] = op(prev_thread[last], values[last][-1])
            sums = _simd_exclusive(op, init, sums)
            for lid, offset in enumerate(offsets):
                group = lid // SIMD_SIZE
                for i in range(n_reads):
                    value = op(values[lid][i], prefix)
                    value = op(value, sums[group])
                    values[lid][i] = op(value, prev_thread[lid])
                    assert values[lid][i] is not GARBAGE
                if info["reverse"]:
                    if not info["inclusive"] and lid == 0 and offset == 0:
                        _record(log, "write", base + axis - 1, size)
                        output[base + axis - 1] = init
                    shift = 0 if info["inclusive"] else 1
                    start = base + axis - offset - shift - n_reads
                    for i in range(n_reads):
                        if offset + shift + n_reads - i - 1 < axis:
                            _record(log, "write", start + i, size)
                            output[start + i] = values[lid][n_reads - i - 1]
                else:
                    if not info["inclusive"] and lid == 0 and offset == 0:
                        _record(log, "write", base, size)
                        output[base] = init
                    shift = 0 if info["inclusive"] else 1
                    for i in range(n_reads):
                        if offset + shift + i < axis:
                            _record(log, "write", base + offset + shift + i, size)
                            output[base + offset + shift + i] = values[lid][i]
            # The last thread shares its inclusive total through sums[0].
            prefix = values[-1][-1]
    return output, log


def _strided_scan(info, arithmetic, inputs, outer, axis, stride, width):
    """``strided_scan`` for the gate's three threadgroups."""
    n_reads = info["nreads"]
    op, init, convert = arithmetic
    bm = bn = 32
    n_simds = bn // n_reads
    n_scans = bn // n_simds
    bn_pad = 32 + 16 // ITEM_SIZES[info["otype"]]
    assert width == n_simds * SIMD_SIZE
    stride_blocks = -(-stride // bn)
    size = outer * axis * stride
    output = [None] * size
    log = []
    for full_gid in range(GRID[1]):
        offset = full_gid // stride_blocks * axis * stride
        global_index_x = full_gid % stride_blocks * bn
        stride_limit = stride - global_index_x
        shared = [GARBAGE] * (bm * bn_pad)
        prefix = [[init] * n_scans for _ in range(width)]
        for j in range(0, axis, bm):
            positions = []
            for lid in range(width):
                read_y, read_x = divmod(lid * n_reads, bn)
                index_y = j + read_y
                check = index_y
                if info["reverse"]:
                    index_y = axis - 1 - index_y
                base = offset + global_index_x + read_x
                positions.append((read_y, read_x, index_y, check, base))
                for i in range(n_reads):
                    slot = read_y * bn_pad + read_x + i
                    if check < axis and read_x + i < stride_limit:
                        index = base + index_y * stride + i
                        _record(log, "read", index, size)
                        shared[slot] = convert(inputs[index])
                    else:
                        shared[slot] = init
            for group in range(n_simds):
                for i in range(n_scans):
                    column = group * n_scans + i
                    lanes = [shared[lane * bn_pad + column] for lane in range(32)]
                    scanned = _simd_inclusive(op, init, lanes)
                    lid = group * SIMD_SIZE
                    scanned = [op(value, prefix[lid][i]) for value in scanned]
                    for lane in range(SIMD_SIZE):
                        prefix[lid + lane][i] = scanned[-1]
                        shared[lane * bn_pad + column] = scanned[lane]
            for read_y, read_x, index_y, check, base in positions:
                if not info["inclusive"]:
                    if check == 0:
                        for i in range(n_reads):
                            if read_x + i < stride_limit:
                                index = base + index_y * stride + i
                                _record(log, "write", index, size)
                                output[index] = init
                    index_y += -1 if info["reverse"] else 1
                    check += 1
                for i in range(n_reads):
                    if check < axis and read_x + i < stride_limit:
                        index = base + index_y * stride + i
                        _record(log, "write", index, size)
                        value = shared[read_y * bn_pad + read_x + i]
                        assert value is not GARBAGE
                        output[index] = value
    return output, log


# ---------------------------------------------------------------------------
# logaddexp error bounds
# ---------------------------------------------------------------------------

# Each logaddexp combination may differ from its binary64 value rounded to the
# storage type by this many units in the last place of max(1, |result|): the
# exp, log1p and final rounding of every implementation. A combination whose
# smaller operand is -inf, whose larger one is +inf or which is dominated by
# DOMINATED_GAP returns its larger operand exactly. Real logaddexp is
# 1-Lipschitz in its operands, so bounds propagate as their maximum; complex
# bounds scale by the combination's own first-order sensitivity.
LOGADDEXP_OP_ULPS = 16
COMPLEX_LOGADDEXP_OP_ULPS = 64
DOMINATED_GAP = 104
PRECISION = {"float16": 11, "float32": 24, "bfloat16": 8}


class _Bounded:
    """A model value and a bound on its distance to any native result."""

    __slots__ = ("value", "error")

    def __init__(self, value, error=0.0):
        self.value = value
        self.error = error

    def __repr__(self):
        return f"_Bounded({self.value!r}, {self.error!r})"


def _ulp(dtype, magnitude):
    exponent = math.frexp(max(float(magnitude), 1.0))[1]
    return 2.0 ** (exponent - PRECISION[dtype])


def _bounded_operation(info):
    dtype = info["otype"]
    plain = _operation(info)

    def apply(a, b):
        if a is GARBAGE or b is GARBAGE:
            return GARBAGE
        value = plain(a.value, b.value)
        if _is_nan(a.value) or _is_nan(b.value):
            return _Bounded(value)
        error = max(a.error, b.error)
        if dtype == "complex64":
            larger = a.value if _complex_gt(a.value, b.value) else b.value
            smaller = a.value if _complex_gt(b.value, a.value) else b.value
            gap = float(smaller[0]) - float(larger[0])
            if float(smaller[0]) == -INF or float(larger[0]) == INF:
                return _Bounded(value, error)
            if gap <= -DOMINATED_GAP:
                return _Bounded(value, error)
            ratio = complex(
                math.exp(gap) * math.cos(float(smaller[1]) - float(larger[1])),
                math.exp(gap) * math.sin(float(smaller[1]) - float(larger[1])),
            )
            sensitivity = 1.01 * (1 + abs(ratio)) / abs(1 + ratio)
            magnitude = max(abs(float(part)) for part in (*value, *larger))
            return _Bounded(
                value,
                sensitivity * error
                + COMPLEX_LOGADDEXP_OP_ULPS * _ulp("float32", magnitude),
            )
        larger = max(float(a.value), float(b.value))
        smaller = min(float(a.value), float(b.value))
        if smaller == -INF or larger == INF or smaller - larger <= -DOMINATED_GAP:
            return _Bounded(value, error)
        magnitude = max(abs(larger), abs(float(value)))
        return _Bounded(value, error + LOGADDEXP_OP_ULPS * _ulp(dtype, magnitude))

    return apply


def _arithmetic(info, bounded=False):
    """``(op, init, convert)`` for the kernel models."""
    op, init = _operation(info), _init(info)
    if not bounded:
        return op, init, lambda value: _convert(info, value)
    return (
        _bounded_operation(info),
        _Bounded(init),
        lambda value: _Bounded(_convert(info, value)),
    )


def _lines(entry, case):
    """Element indices of every scanned line, in memory order."""
    info = INFO[entry]
    lines, axis, stride, _, _ = _geometry(entry, case)
    result = []
    for line in range(lines):
        if info["kind"] == "contig":
            result.append([line * axis + k for k in range(axis)])
        else:
            plane, column = divmod(line, stride)
            result.append(
                [plane * axis * stride + k * stride + column for k in range(axis)]
            )
    return result


def _definition(entry, case, inputs):
    """An independent left fold along every scanned line."""
    info = INFO[entry]
    op, init, convert = _arithmetic(info)
    _, _, _, _, size = _geometry(entry, case)
    output = [None] * size
    for indices in _lines(entry, case):
        total = init
        for index in indices[::-1] if info["reverse"] else indices:
            value = op(convert(inputs[index]), total)
            output[index] = value if info["inclusive"] else total
            total = value
    return output


def _simulate(entry, case, inputs, bounded=False):
    info = INFO[entry]
    lines, axis, stride, _, size = _geometry(entry, case)
    assert len(inputs) == size
    width = _workgroup(entry)[0]
    arithmetic = _arithmetic(info, bounded)
    if info["kind"] == "contig":
        output, log = _contiguous_scan(info, arithmetic, inputs, axis, width)
    else:
        output, log = _strided_scan(
            info, arithmetic, inputs, lines // stride, axis, stride, width
        )
    writes = sorted(index for kind, index in log if kind == "write")
    reads = sorted(index for kind, index in log if kind == "read")
    # Every input is read and every output written exactly once.
    assert writes == list(range(size))
    assert reads == list(range(size))
    assert None not in output
    return output, log


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Bounds on |prefix sum| for exact floating sums: any partial result of any
# association is a sum over at most four contiguous runs, at most 8 * B in
# magnitude, and every value is a multiple of 1/4.
FLOAT_SUM_BOUNDS = {"bfloat16": 8, "float16": 64, "float32": 2**16}
SIGNED_SUM_BOUNDS = {"int32": 2**27, "int64": 2**59}
# Products whose source operands promote to int must not overflow: uint16
# products stay below 2**16, int32 and int64 products below their maxima.
PRODUCT_BUDGETS = {"uint16": 2**16 - 1, "int32": 2**31 - 1, "int64": 2**63 - 1}
PRODUCT_FACTORS = {
    "uint16": (2, 3, 5, 7, 11, 13),
    "int32": (3, -5, 7, -11, 2**20 + 1),
    "int64": (3, -5, 7, 2**31 + 1, -(2**20 + 7)),
}
LARGE_PRODUCT_FACTORS = {"uint16": 65521, "int32": 2**30 - 1, "int64": 2**62 - 57}
# Distinct extrema operands rounded to each type, with infinities.
EXTREMA_POOLS = {
    "float32": (
        -INF,
        -3.0e38,
        -1.5e10,
        -2.5,
        -1.0,
        -0.5,
        1.0e-30,
        0.5,
        1.0,
        1.5,
        2.5,
        1000.0,
        1.5e10,
        3.0e38,
        INF,
    ),
    "float16": (
        -INF,
        -65504.0,
        -1000.0,
        -2.5,
        -1.0,
        -0.5,
        2.0**-14,
        0.5,
        1.0,
        1.5,
        2.5,
        1000.0,
        65504.0,
        INF,
    ),
    "bfloat16": (
        -INF,
        -(2.0**100),
        -1024.0,
        -2.5,
        -1.0,
        -0.5,
        2.0**-100,
        0.5,
        1.0,
        1.5,
        2.5,
        1024.0,
        2.0**100,
        INF,
    ),
}
COMPLEX_EXTREMA = tuple(
    (real, imag)
    for real in (-INF, -2.5, -1.0, 0.0, 1.0, 2.5, INF)
    for imag in (-1.0, 0.0, 1.0, 2.0)
)
# logaddexp operands 64 * odd differ by at least 128, so exp() of their
# difference underflows and each combination returns its larger operand.
LOGADDEXP_ODD_LIMITS = {"bfloat16": 255, "float16": 1023, "float32": 4095}


def _draws(*key):
    from demos.integrations.mlx.tests.kernels.test_current_sort import _lcg

    digest = hashlib.sha256(":".join(map(str, key)).encode()).digest()
    return _lcg(int.from_bytes(digest[:8], "little"))


def _bits(draws, bits):
    value = 0
    for _ in range(-(-bits // 31)):
        value = (value << 31) | next(draws)
    return value & ((1 << bits) - 1)


def _below(draws, count):
    return _bits(draws, 62) % count


def _signed(draws, magnitude):
    return _below(draws, 2 * magnitude + 1) - magnitude


def _walk(draws, length, bound, step):
    """Steps whose running sum stays within ``[-bound, bound]``."""
    total, steps = 0, []
    for _ in range(length):
        move = _signed(draws, step)
        if abs(total + move) > bound:
            move = -move
        steps.append(move)
        total += move
    return steps


def _integer_pool(dtype):
    bits, signed = INTEGER_DTYPES[dtype]
    low, high = _limits(dtype, False), _limits(dtype, True)
    pool = {low, low + 1, 0, 1, 2, high - 1, high}
    pool |= {-1, -2} if signed else {2 ** (bits - 1) - 1, 2 ** (bits - 1)}
    return sorted(pool)


def _sum_values(dtype, draws, pattern, length):
    if dtype == "bool_":
        return [bool(next(draws) & 1) for _ in range(length)]
    if dtype in SIGNED_SUM_BOUNDS:
        bound = SIGNED_SUM_BOUNDS[dtype]
        return _walk(draws, length, bound, bound // 2)
    if dtype in INTEGER_DTYPES:
        bits = INTEGER_DTYPES[dtype][0]
        pool = _integer_pool(dtype)
        return [
            (
                pool[_below(draws, len(pool))]
                if next(draws) & 3 == 0
                else _wrap(dtype, _bits(draws, bits))
            )
            for _ in range(length)
        ]
    if dtype == "complex64":
        if pattern == 1:
            # Cancellation around 2**24 rounds; the order is the kernel's own.
            cycle = [(2.0**24, 1.0), (1.0, -(2.0**24)), (-(2.0**24), 1.0)]
            cycle.append((1.0, 2.0**24))
            return [
                (
                    cycle[_below(draws, 4)]
                    if next(draws) & 1
                    else (_signed(draws, 8) / 4, _signed(draws, 8) / 4)
                )
                for _ in range(length)
            ]
        if pattern == 2:
            zeros = [(-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.125, -0.125)]
            zeros.append((-0.125, 0.125))
            return [zeros[_below(draws, len(zeros))] for _ in range(length)]
        bound = FLOAT_SUM_BOUNDS["float32"] * 4
        return list(
            zip(
                (step / 4 for step in _walk(draws, length, bound, bound // 2)),
                (step / 4 for step in _walk(draws, length, bound, bound // 2)),
            )
        )
    bound = FLOAT_SUM_BOUNDS[dtype] * 4
    if pattern == 1:
        # Signed zeros and exact cancellation back to zero.
        values, total = [], 0
        for _ in range(length):
            if next(draws) & 1:
                values.append(-0.0 if next(draws) & 1 else 0.0)
                continue
            move = -total if total and next(draws) & 1 else _signed(draws, 8)
            if abs(total + move) > bound:
                move = -move
            values.append(move / 4)
            total += move
        return values
    values = [step / 4 for step in _walk(draws, length, bound, bound // 2)]
    if pattern == 2:
        values[_below(draws, length)] = INF if next(draws) & 1 else -INF
    return values


def _prod_values(dtype, draws, pattern, length):
    if dtype == "bool_":
        rate = (64, None, 8)[pattern]
        return [rate is None or _below(draws, rate) != 0 for _ in range(length)]
    if dtype in PRODUCT_BUDGETS:
        signed = INTEGER_DTYPES[dtype][1]
        values = [-1 if signed and next(draws) & 1 else 1 for _ in range(length)]
        budget = PRODUCT_BUDGETS[dtype]
        if pattern == 1:
            factors = [LARGE_PRODUCT_FACTORS[dtype]]
        else:
            factors = [
                PRODUCT_FACTORS[dtype][_below(draws, len(PRODUCT_FACTORS[dtype]))]
                for _ in range(length // 4 + 2)
            ]
        product = 1
        for factor in factors:
            if product * abs(factor) <= budget:
                values[_below(draws, length)] *= factor
                product *= abs(factor)
        assert math.prod(abs(value) for value in values) <= budget
        if pattern == 2:
            values[_below(draws, length)] = 0
        return values
    if dtype in INTEGER_DTYPES:
        bits = INTEGER_DTYPES[dtype][0]
        values = []
        for _ in range(length):
            word = _bits(draws, bits)
            values.append(_wrap(dtype, word if _below(draws, 16) == 0 else word | 1))
        if pattern == 2:
            values[_below(draws, length)] = 0
        return values
    if dtype == "complex64":
        units = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
        values, level = [], 0
        for _ in range(length):
            following = _below(draws, 5) - 2
            step, level = following - level, following
            if step % 2 == 0:
                scale = 2.0 ** (step // 2)
                unit = units[_below(draws, 4)]
                values.append((unit[0] * scale, unit[1] * scale))
            else:
                scale = 2.0 ** ((step - 1) // 2)
                values.append(
                    (
                        scale if next(draws) & 1 else -scale,
                        scale if next(draws) & 1 else -scale,
                    )
                )
        if pattern == 1:
            values[_below(draws, length)] = (0.0, -0.0)
        return values
    values, level = [], 0
    for _ in range(length):
        following = _below(draws, 3) - 1
        magnitude = 2.0 ** (following - level)
        level = following
        values.append(-magnitude if next(draws) & 1 else magnitude)
    if pattern == 1:
        for _ in range(max(1, length // 16)):
            values[_below(draws, length)] = -0.0 if next(draws) & 1 else 0.0
    if pattern == 2:
        values[_below(draws, length)] = INF if next(draws) & 1 else -INF
    return values


def _extrema_values(op, dtype, draws, pattern, length):
    if dtype == "bool_":
        rate = (2, 32, 32)[pattern]
        values = [_below(draws, rate) == 0 for _ in range(length)]
        return [not value for value in values] if pattern == 2 else values
    if dtype in INTEGER_DTYPES:
        bits = INTEGER_DTYPES[dtype][0]
        pool = _integer_pool(dtype)
        return [
            (
                pool[_below(draws, len(pool))]
                if next(draws) & 1
                else _wrap(dtype, _bits(draws, bits))
            )
            for _ in range(length)
        ]
    if dtype == "complex64":
        return [
            COMPLEX_EXTREMA[_below(draws, len(COMPLEX_EXTREMA))] for _ in range(length)
        ]
    # One zero sign per line: Metal keeps the left operand of a tie.
    pool = list(EXTREMA_POOLS[dtype]) + [-0.0 if pattern == 1 else 0.0]
    values = [pool[_below(draws, len(pool))] for _ in range(length)]
    if pattern == 2:
        values[_below(draws, length)] = NAN
    return values


def _logaddexp_values(dtype, draws, pattern, length, dense):
    if dense:
        if dtype == "complex64":
            return [
                (_signed(draws, 8) / 4, _signed(draws, 2) / 8) for _ in range(length)
            ]
        return [_signed(draws, 12) / 4 for _ in range(length)]
    limit = LOGADDEXP_ODD_LIMITS["float32" if dtype == "complex64" else dtype]
    pool = [sign * 64.0 * odd for odd in range(1, limit + 1, 2) for sign in (1, -1)]
    for index in range(len(pool) - 1, 0, -1):
        other = _below(draws, index + 1)
        pool[index], pool[other] = pool[other], pool[index]
    positions = list(range(length))
    for index in range(len(positions) - 1, 0, -1):
        other = _below(draws, index + 1)
        positions[index], positions[other] = positions[other], positions[index]
    count = min(length // 5 + 1, len(pool))
    values = [-INF] * length
    for position, value in zip(positions[:count], pool):
        values[position] = value
    if dtype == "complex64":
        # Nonzero imaginary parts: a +0.0 or -0.0 increment leaves them as is.
        values = [(value, (2 * _below(draws, 32) - 31) / 8) for value in values]
    if pattern == 1:
        special = (INF, 1.5) if dtype == "complex64" else INF
        values[_below(draws, length)] = special
    if pattern == 2:
        special = (NAN, 0.0) if dtype == "complex64" else NAN
        values[_below(draws, length)] = special
    return values


def _line_values(entry, case, line, length):
    info = INFO[entry]
    op, dtype = info["op"], info["itype"]
    draws = _draws(info["kind"], case, op, dtype, line)
    pattern = line % 3
    if op == "sum":
        return _sum_values(dtype, draws, pattern, length)
    if op == "prod":
        return _prod_values(dtype, draws, pattern, length)
    if op in {"max", "min"}:
        return _extrema_values(op, dtype, draws, pattern, length)
    dense = case == DENSE_CASES[info["kind"]]
    return _logaddexp_values(dtype, draws, pattern, length, dense)


def _inputs(entry, case):
    """Logical input elements of one case, in memory order."""
    _, axis, _, _, size = _geometry(entry, case)
    inputs = [None] * size
    for line, indices in enumerate(_lines(entry, case)):
        for index, value in zip(indices, _line_values(entry, case, line, axis)):
            inputs[index] = value
    assert None not in inputs
    return inputs


def _poison(info):
    """Input padding that would change any result that read it."""
    dtype, op = info["itype"], info["op"]
    if dtype == "bool_":
        return op in {"sum", "max"}
    if dtype in INTEGER_DTYPES:
        return {"sum": 1, "prod": 0, "max": _limits(dtype, True)}.get(
            op, _limits(dtype, False)
        )
    if dtype == "complex64":
        if op == "max":
            return (INF, INF)
        if op == "min":
            return (-INF, -INF)
        return (NAN, NAN)
    return NAN


def _float_guard(value):
    value = float(value)
    return 1000.0 if math.isfinite(value) and abs(value + 1000.0) < 500.0 else -1000.0


def _complement(dtype, value):
    """An output guard that differs from the expected element."""
    if dtype == "bool_":
        return not value
    if dtype in INTEGER_DTYPES:
        return _wrap(dtype, value ^ 0x5A5A5A5A5A5A5A5A)
    if dtype == "complex64":
        return (_float_guard(value[0]), _float_guard(value[1]))
    return _float_guard(value)


def _trailing_guards(dtype):
    if dtype == "bool_":
        return [index % 2 == 0 for index in range(GUARD_COUNT)]
    if dtype in INTEGER_DTYPES:
        return [_wrap(dtype, 0x5A5A5A5A5A5A5A5A)] * GUARD_COUNT
    if dtype == "complex64":
        return [(-1000.0, 1000.0)] * GUARD_COUNT
    return [-1000.0] * GUARD_COUNT


def _logical(dtype, value):
    """A model value as a plain logical value for the runtime encoders."""
    if isinstance(value, _Bounded):
        value = value.value
    if dtype == "complex64":
        return (float(value[0]), float(value[1]))
    if dtype in FLOAT_DTYPES:
        return float(value)
    return value


def _case_data(entry, case):
    """Inputs, expected outputs, bounds and initial buffers of one case."""
    info = INFO[entry]
    inputs = _inputs(entry, case)
    bounded = info["op"] == "logaddexp"
    output, log = _simulate(entry, case, inputs, bounded)
    dtype = info["otype"]
    expected = [_logical(dtype, value) for value in output]
    bounds = [value.error for value in output] if bounded else [0.0] * len(output)
    lines, axis, stride, stride_blocks, size = _geometry(entry, case)
    return {
        "inputs": inputs,
        "storage": inputs + [_poison(info)] * GUARD_COUNT,
        "expected": expected + _trailing_guards(dtype),
        "bounds": bounds + [0.0] * GUARD_COUNT,
        "initial": (
            [_complement(dtype, value) for value in expected] + _trailing_guards(dtype)
        ),
        "accesses": log,
        "metadata": {
            "axis_size": axis,
            "stride": stride,
            "stride_blocks": stride_blocks,
        },
        "size": size,
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _element_count():
    return sum(_geometry(entry, case)[4] for entry in ENTRIES for case in _cases(entry))


def _expected_validation_contract():
    return {
        "model": (
            "contiguous_scan and strided_scan step by step: block loop, "
            "SIMD-group scans, threadgroup carries, guarded loads and stores"
        ),
        "comparison": {
            "default": "exact words",
            "logaddexp": (
                "exact words where the model's bound is zero, otherwise "
                "|native - model| <= bound per element (complex: modulus)"
            ),
        },
        "orderFreeScans": (
            "sums and products of types narrower than 8 bytes use Apple's "
            "prefix intrinsics; their inputs make every partial result exact"
        ),
        "nReads": {"8-byte": 2, "other": 4},
        "workgroupSize": {
            "contig": [CONTIG_WIDTH, 1, 1],
            "strided": "[(32 / N_READS) * 32, 1, 1]",
        },
        "grid": list(GRID),
        "dispatchRegion": {
            "targets": ["opengl", "directx"],
            "threads": "[workgroup width, 3, 1]",
        },
        "cases": {
            "contig": {
                "rows": GRID[1],
                "axisSize": {
                    "single": 1,
                    "short": 37,
                    "block": "N_READS * 128",
                    "carry": "2 * N_READS * 128 + 5",
                },
            },
            "strided": {
                name: {
                    "outer": outer,
                    "axisSize": axis,
                    "stride": stride,
                    "strideBlocks": -(-stride // 32),
                }
                for name, (outer, axis, stride) in STRIDED_CASES.items()
            },
        },
        "metadataPreconditions": {
            name: {"minimum": 1, "maximum": maximum}
            for name, maximum in METADATA_BOUNDS.items()
        },
        "openglIndexRangeAssertions": {
            group: [
                {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                for expression in expressions
            ]
            for group, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
        },
        "softwareSubgroupWidth": {"targets": ["opengl", "directx"], "width": 32},
        "directx": {
            "binary32AdditiveProfile": "native-precise",
            "groupsharedBoolStorage": "uint",
            "relativeWaveShuffleOutOfRange": "self",
        },
        "logaddexpBounds": {
            "opUlps": LOGADDEXP_OP_ULPS,
            "complexOpUlps": COMPLEX_LOGADDEXP_OP_ULPS,
            "dominatedGap": DOMINATED_GAP,
            "denseCases": DENSE_CASES,
        },
        "inputs": {
            "floatSumPrefixBounds": FLOAT_SUM_BOUNDS,
            "signedSumPrefixBounds": SIGNED_SUM_BOUNDS,
            "productBudgets": PRODUCT_BUDGETS,
            "floatProducts": "signed powers of two with prefix exponents in [-1, 1]",
            "complexProducts": "axis and diagonal Gaussian factors, exact",
            "extrema": "one zero sign per scanned line",
            "logaddexpExact": "64 * odd values or -inf; combinations underflow",
        },
        "guards": {
            "output": "every element starts different from its expected result",
            "trailingOutputGuards": GUARD_COUNT,
            "inputPadding": GUARD_COUNT,
        },
        "elementTypes": {
            target: {
                dtype: _element_type(target, dtype)
                for dtype in sorted(ITEM_SIZES, key=list(ITEM_SIZES).index)
            }
            for target in TARGETS
        },
        "dispatchCountByTarget": sum(len(_cases(entry)) for entry in ENTRIES),
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": sum(len(_cases(entry)) for entry in ENTRIES),
        "checkedElementCountByTarget": _element_count(),
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_scan_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-scan-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "headers": HEADERS,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
        assert len({record["sha256"] for record in artifacts.values()}) == len(ENTRIES)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_scan_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "scan.current-tree.translation.json" in readme
    assert len(HELPERS) == 57
    assert len(ENTRIES) == len(set(ENTRIES)) == 456
    assert ENTRIES[:8] == (
        "contig_scan_inclusive_sum_bool__int32",
        "contig_scan_exclusive_sum_bool__int32",
        "contig_scan_reverse_inclusive_sum_bool__int32",
        "contig_scan_reverse_exclusive_sum_bool__int32",
        "strided_scan_inclusive_sum_bool__int32",
        "strided_scan_exclusive_sum_bool__int32",
        "strided_scan_reverse_inclusive_sum_bool__int32",
        "strided_scan_reverse_exclusive_sum_bool__int32",
    )
    assert ENTRIES[-1] == "strided_scan_reverse_exclusive_logaddexp_complex64_complex64"
    for entry, info in INFO.items():
        # scan.cpp: "{contig,strided}_scan_[reverse_]{in,ex}clusive_{op}_{in}_{out}".
        name = (
            f"{info['kind']}_scan_{'reverse_' if info['reverse'] else ''}"
            f"{'inclusive' if info['inclusive'] else 'exclusive'}_{info['op']}_"
            f"{info['itype']}_{info['otype']}"
        )
        assert entry == name
        # n_reads = (in.itemsize() <= 4) ? 4 : 2.
        assert info["nreads"] == (4 if ITEM_SIZES[info["itype"]] <= 4 else 2)


@pytest.mark.parametrize("entry", ENTRIES[::8] + ENTRIES[4::8])
def test_scan_cases_follow_mlx_dispatch(entry):
    info = INFO[entry]
    width = _workgroup(entry)[0]
    for case in _cases(entry):
        lines, axis, stride, stride_blocks, size = _geometry(entry, case)
        if info["kind"] == "strided":
            # scan.cpp: bn = 32, n_simdgroups = bn / n_reads and the grid's
            # height carries every 32-column block of every outer slice.
            assert width == (32 // info["nreads"]) * 32
            assert stride_blocks == (stride + 31) // 32
            assert (lines // stride) * stride_blocks == GRID[1]
            assert axis <= METADATA_BOUNDS["axis_size"]
            assert stride <= METADATA_BOUNDS["stride"]
            assert stride_blocks <= METADATA_BOUNDS["stride_blocks"]
        else:
            assert width == CONTIG_WIDTH and lines == GRID[1]
            assert size == GRID[1] * axis
    if info["kind"] == "contig":
        # scan.cpp chooses 128 threads for these rows; longer and shorter
        # rows run the same block loop with more or fewer blocks.
        block = info["nreads"] * CONTIG_WIDTH
        elements_per_simd = info["nreads"] * SIMD_SIZE
        assert ((block + elements_per_simd - 1) // elements_per_simd) * 32 == width
        assert -(-_contig_axis(info, "carry") // block) == 3


def _same(info, actual, expected, bound=0.0):
    """Model comparison: words, or bounded values for logaddexp."""
    dtype = info["otype"]
    actual, expected = _logical(dtype, actual), _logical(dtype, expected)
    parts = (actual, expected) if dtype == "complex64" else ((actual,), (expected,))
    finite = all(math.isfinite(value) for value in (*parts[0], *parts[1]))
    if bound and finite:
        return math.dist(parts[0], parts[1]) <= bound
    return all(
        (math.isnan(a) and math.isnan(b))
        or struct.pack("<d", a) == struct.pack("<d", b)
        for a, b in zip(*parts)
    )


# complex64 sums of the cancellation pattern round; their words come from the
# kernel's own association order, so the left fold differs.
INEXACT_LINES = {("sum", "complex64"): 1}


@pytest.mark.parametrize("helper", [name for name, *_ in HELPERS])
def test_scan_model_matches_left_fold(helper):
    """The kernel model reaches the definition for every variant and case."""
    for entry in _helper_entries(helper):
        info = INFO[entry]
        bounded = info["op"] == "logaddexp"
        for case in _cases(entry):
            inputs = _inputs(entry, case)
            output, _ = _simulate(entry, case, inputs, bounded)
            if bounded:
                op, init, convert = _arithmetic(info, True)
                reference = [None] * len(inputs)
                for indices in _lines(entry, case):
                    total = init
                    for index in indices[::-1] if info["reverse"] else indices:
                        value = op(convert(inputs[index]), total)
                        reference[index] = value if info["inclusive"] else total
                        total = value
            else:
                reference = _definition(entry, case, inputs)
            inexact = INEXACT_LINES.get((info["op"], info["otype"]))
            for line, indices in enumerate(_lines(entry, case)):
                if inexact is not None and line % 3 == inexact:
                    continue
                for index in indices:
                    got, want = output[index], reference[index]
                    bound = got.error + want.error if bounded else 0.0
                    if info["op"] == "prod" and info["otype"] == "complex64":
                        # Exact, but zero signs follow the association order.
                        assert [float(part) for part in got] == [
                            float(part) for part in want
                        ], (entry, case, index)
                        continue
                    assert _same(info, got, want, bound), (entry, case, index, got)


@pytest.mark.parametrize(
    "dtype,bound", [(dtype, bound) for dtype, bound in FLOAT_SUM_BOUNDS.items()]
)
def test_floating_sum_inputs_keep_every_partial_result_exact(dtype, bound):
    """Four runs of quarters bounded by B need at most 8B * 4 significand units."""
    assert 8 * bound * 4 <= 2 ** PRECISION[dtype]
    entry = f"contig_scan_inclusive_sum_{dtype}_{dtype}"
    for case in CONTIG_CASES:
        for line, indices in enumerate(_lines(entry, case)):
            if line % 3 == 2:
                continue
            values = [Fraction(_inputs(entry, case)[index]) for index in indices]
            assert all((value * 4).denominator == 1 for value in values)
            totals = [sum(values[: k + 1]) for k in range(len(values))]
            assert max(abs(total) for total in totals) <= bound


def test_scan_inputs_exercise_signed_zero_infinity_and_nan():
    seen = set()
    for entry in ENTRIES[::4]:
        info = INFO[entry]
        for case in _cases(entry):
            for value in _inputs(entry, case):
                parts = value if isinstance(value, tuple) else (value,)
                for part in parts:
                    if isinstance(part, float):
                        if math.isnan(part):
                            seen.add((info["op"], "nan"))
                        elif math.isinf(part):
                            seen.add((info["op"], "inf"))
                        elif part == 0 and math.copysign(1, part) < 0:
                            seen.add((info["op"], "-0"))
    assert {
        ("sum", "inf"),
        ("sum", "-0"),
        ("prod", "inf"),
        ("prod", "-0"),
        ("max", "nan"),
        ("min", "-0"),
        ("logaddexp", "nan"),
        ("logaddexp", "inf"),
    } <= seen


@pytest.mark.parametrize(
    "entry",
    [entry for entry in ENTRIES if INFO[entry]["reverse"]][::4],
)
def test_scan_model_justifies_opengl_index_assertions(entry):
    """Every executed access of a reverse kernel lies inside its buffer."""
    for case in _cases(entry):
        data = _case_data(entry, case)
        indices = [index for _, index in data["accesses"]]
        assert min(indices) >= 0 and max(indices) < data["size"] <= SIGNED_MAX
        # The kernels form some of these addresses before their lines start.
        if INFO[entry]["kind"] == "contig" and case == "carry":
            n_reads = INFO[entry]["nreads"]
            axis = data["metadata"]["axis_size"]
            last = -(-axis // (n_reads * CONTIG_WIDTH)) - 1
            start = axis - (
                last * CONTIG_WIDTH * n_reads + (CONTIG_WIDTH - 1) * n_reads
            )
            assert start - n_reads < 0


def _mutated_words(entry, case, mutation):
    info = dict(INFO[entry])
    inputs = _inputs(entry, case)
    if mutation == "inclusive":
        info["inclusive"] = not info["inclusive"]
    elif mutation == "reverse":
        info["reverse"] = not info["reverse"]
    width = _workgroup(entry)[0]
    _, axis, stride, _, _ = _geometry(entry, case)
    bounded = info["op"] == "logaddexp"
    arithmetic = _arithmetic(info, bounded)
    if mutation == "init":
        op, init, convert = arithmetic
        arithmetic = (op, convert(inputs[0]), convert)
    if mutation == "carry":
        op, init, convert = arithmetic

        def reset(a, b, op=op):
            # Drop every combination with a previous block's total.
            return a if b is GARBAGE else op(a, b)

        arithmetic = (reset, init, convert)
    if info["kind"] == "contig":
        output, _ = _contiguous_scan(info, arithmetic, inputs, axis, width)
    else:
        lines = _geometry(entry, case)[0]
        output, _ = _strided_scan(
            info, arithmetic, inputs, lines // stride, axis, stride, width
        )
    return [_logical(info["otype"], value) for value in output]


@pytest.mark.parametrize("mutation", ("inclusive", "reverse", "init"))
@pytest.mark.parametrize(
    "entry",
    (
        "contig_scan_inclusive_sum_float32_float32",
        "strided_scan_reverse_exclusive_prod_int64_int64",
        "contig_scan_reverse_inclusive_max_bfloat16_bfloat16",
        "strided_scan_exclusive_min_uint8_uint8",
        "contig_scan_exclusive_logaddexp_complex64_complex64",
        "strided_scan_inclusive_sum_bool__uint32",
    ),
)
def test_scan_expectations_reject_defects(entry, mutation):
    info = INFO[entry]
    exposed = False
    for case in _cases(entry):
        data = _case_data(entry, case)
        expected = data["expected"][: data["size"]]
        mutated = _mutated_words(entry, case, mutation)
        exposed |= any(
            not _same(info, got, want, bound)
            for got, want, bound in zip(mutated, expected, data["bounds"])
        )
    assert exposed, (entry, mutation)


@pytest.mark.parametrize("target", TARGETS)
def test_scan_runtime_values_match_physical_storage(target):
    expected = {
        "metal": {"bool_": "bool", "float16": "float16", "bfloat16": "bfloat16"},
        "opengl": {
            "bool_": "uint32",
            "uint8": "uint32",
            "uint16": "uint32",
            "int8": "int32",
            "int16": "int32",
            "float16": "float32",
            "bfloat16": "float32",
        },
        "directx": {
            "bool_": "uint32",
            "uint8": "uint32",
            "int8": "int32",
            "float16": "uint16",
            "bfloat16": "uint16",
        },
    }[target]
    for dtype in ITEM_SIZES:
        physical = expected.get(dtype, "float32" if dtype == "complex64" else dtype)
        assert _element_type(target, dtype) == physical


@pytest.mark.parametrize("target", TARGETS)
def test_scan_configs_scope_opengl_index_preconditions(tmp_path, target):
    for entry in ENTRIES[:16]:
        config = _config(tmp_path / "work", target, entry)
        assertions = [item.to_json() for item in config.index_range_assertions]
        preconditions = [item.to_json() for item in config.metadata_preconditions]
        assert [item["parameter"] for item in preconditions] == list(_parameters(entry))
        if target != "opengl" or not INFO[entry]["reverse"]:
            assert assertions == []
            continue
        assert [item["expression"] for item in assertions] == list(
            _index_assertions(entry)
        )
        assert {(item["minimum"], item["maximum"]) for item in assertions} == {
            (0, SIGNED_MAX)
        }


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------

C_TYPES = {
    "bool": "bool_",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "int8_t": "int8",
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "half": "float16",
    "float": "float32",
    "bfloat16_t": "bfloat16",
    "complex64_t": "complex64",
}
OPERATIONS = {
    "CumSum": "sum",
    "CumProd": "prod",
    "CumMax": "max",
    "CumMin": "min",
    "CumLogaddexp": "logaddexp",
}


def _source_helpers(text):
    rows = re.findall(
        r"^instantiate_scan_helper\((\w+),\s*(\w+),\s*(\w+),\s*(\w+),\s*(\d)\)",
        text,
        flags=re.MULTILINE,
    )
    return [
        ((name, C_TYPES[itype], C_TYPES[otype], OPERATIONS[op]), int(nreads))
        for name, itype, otype, op, nreads in rows
    ]


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native scan gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in HEADERS.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # The helper table above is scan.metal's own instantiation list.
    helpers = _source_helpers(source.decode("utf-8"))
    assert [row for row, _ in helpers] == list(HELPERS)
    assert [nreads for _, nreads in helpers] == [
        INFO[_helper_entries(name)[0]]["nreads"] for name, *_ in HELPERS
    ]
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("scan-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _translate(root, work, target, entry):
    """Translate one entry and check it against its pinned artifact."""
    record = _contract()["artifacts"][target][entry]
    translation = translate_project(
        _config(work, target, entry),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    validation = validate_project_report(work / "report.json")
    assert validation["success"] is True, validation
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    assert report["summary"]["failedCount"] == 0
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    assert artifact["generatedHash"] == {
        "algorithm": "sha256",
        "value": record["sha256"],
    }
    assert artifact["generatedSizeBytes"] == record["sizeBytes"]
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == _workgroup(entry)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    descriptor, package = _prepare_native_package(translation, work)
    return descriptor, package, record


def _scalar(value):
    return {"dtype": "uint64", "shape": [1], "values": [value]}


def _request(descriptor, package, target, entry, data):
    """Bind the case by source parameter and build one dispatch."""
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    info = INFO[entry]
    values = {
        "in": _runtime_value(target, info["itype"], data["storage"]),
        "out": _runtime_value(target, info["otype"], data["initial"]),
        **{name: _scalar(data["metadata"][name]) for name in _parameters(entry)},
    }
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        source = binding.get("provenance", {}).get("sourceResource")
        assert source is not None, binding
        name = source["parameter"]
        value = values[name]
        layout = binding["scalarLayout"]
        storage = layout.get("storageEncoding")
        if storage is None:
            assert layout["elementType"] == value["dtype"], (name, layout)
        else:
            assert layout["elementType"] == "uint16", (name, layout)
            assert storage == {
                "encoding": FLOAT16_BITS,
                "logicalElementType": value["dtype"],
            }
            assert value["encoding"] == FLOAT16_BITS
        if name in _parameters(entry):
            assert binding["provenance"]["metadataPrecondition"]["maximum"] == (
                METADATA_BOUNDS[name]
            )
        inputs[binding["name"]] = value
        if name == "out":
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert output is not None and len(inputs) == len(values)
    expected = _runtime_value(target, info["otype"], data["expected"])
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: expected},
        {"workgroupCount": list(GRID), "workgroupSize": _workgroup(entry)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, {"out": output}, len(observed)


def _decode(target, dtype, words):
    """Logical values of runtime output words."""
    if dtype == "complex64":
        values = [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
        return list(zip(values[::2], values[1::2]))
    if dtype == "float32" or (target == "opengl" and dtype in FLOAT_DTYPES):
        return [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
    if dtype == "float16":
        return [struct.unpack("<e", struct.pack("<H", word))[0] for word in words]
    assert dtype == "bfloat16"
    return [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words]


def _check(label, target, entry, actual, data):
    """Compare output words; logaddexp elements with a bound numerically."""
    info = INFO[entry]
    dtype = info["otype"]
    expected = _runtime_value(target, dtype, data["expected"])["values"]
    assert len(actual) == len(expected), label
    if info["op"] != "logaddexp":
        mismatches = [
            (index, want, got)
            for index, (got, want) in enumerate(zip(actual, expected))
            if got != want
        ]
        assert not mismatches, (label, len(mismatches), mismatches[:8])
        return 0.0
    got_values = _decode(target, dtype, actual)
    want_values = _decode(target, dtype, expected)
    words = 2 if dtype == "complex64" else 1
    worst, mismatches = 0.0, []
    for index, (got, want, bound) in enumerate(
        zip(got_values, want_values, data["bounds"])
    ):
        got_words = actual[index * words : (index + 1) * words]
        want_words = expected[index * words : (index + 1) * words]
        parts = (got, want) if dtype == "complex64" else ((got,), (want,))
        if bound and all(math.isfinite(value) for value in (*parts[0], *parts[1])):
            distance = math.dist(*parts)
            worst = max(worst, distance / bound)
            if distance > bound:
                mismatches.append((index, want, got, bound))
        elif got_words != want_words:
            mismatches.append((index, want_words, got_words, bound))
    assert not mismatches, (label, len(mismatches), mismatches[:8])
    return worst


# Apple's Metal compiler on the gate's development machine (M2 Max, Metal
# 32023.918) miscompiles MLX's reverse contiguous scans of one-byte inputs:
# both the unchanged upstream kernel and its translation drop or misplace
# elements addressed through ``in + axis_size - offset - N_READS``, while the
# same pointers formed with signed integer offsets give the model's words.
# These entries must still reproduce the upstream words exactly on Metal.
METAL_BYTE_REVERSE_ENTRIES = frozenset(
    entry
    for entry, info in INFO.items()
    if info["kind"] == "contig" and info["reverse"] and ITEM_SIZES[info["itype"]] == 1
)


def _assert_identical(label, actual, expected):
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


def _model_deviations(target, entry, words, data):
    try:
        _check("model", target, entry, words, data)
    except AssertionError as error:
        return error.args[0][1] if isinstance(error.args[0], tuple) else -1
    return 0


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    info = INFO[entry]
    payloads = [
        _metal_bytes(info["itype"], data["storage"]),
        _metal_bytes(info["otype"], data["initial"]),
        *(struct.pack("<Q", data["metadata"][name]) for name in _parameters(entry)),
    ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": list(GRID),
                "workgroupSize": _workgroup(entry),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    for index, payload in enumerate(payloads):
        if index != 1:
            assert (readback / f"buffer-{index}.bin").read_bytes() == payload
    raw = (readback / "buffer-1.bin").read_bytes()
    dtype = info["otype"]
    if dtype == "bool_":
        words = list(raw)
        assert set(words) <= {0, 1}
        words = [bool(word) for word in words]
    elif dtype in INTEGER_DTYPES:
        from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
            METAL_FORMATS,
        )

        code = METAL_FORMATS[dtype]
        words = list(struct.unpack(f"<{len(raw) // struct.calcsize(code)}{code}", raw))
    elif dtype in {"float16", "bfloat16"}:
        words = list(struct.unpack(f"<{len(raw) // 2}H", raw))
    else:
        words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return device, words


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_scan_native_parity(current_mlx, metal_reference, tmp_path, entry):
    root, target = current_mlx
    with tempfile.TemporaryDirectory(prefix=".current-scan-", dir=root) as directory:
        work = Path(directory)
        try:
            descriptor, package, record = _translate(root, work, target, entry)
            executor = _executor(target)
            evidence = []
            try:
                for case in _cases(entry):
                    data = _case_data(entry, case)
                    request, output_names, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    outputs = _execute(
                        executor, work, case, request, output_names, observed
                    )
                    output = outputs["out"]
                    upstream = None
                    if target == "metal":
                        device, words = _upstream(
                            work, metal_reference, entry, case, data
                        )
                        # The translation reproduces the unchanged upstream
                        # kernel word for word on the same device.
                        _assert_identical(case, output["values"], words)
                        deviations = _model_deviations(target, entry, words, data)
                        upstream = {
                            "device": device,
                            "identicalToTranslation": True,
                            "modelDeviationCount": deviations,
                        }
                        if deviations:
                            # Only Apple's miscompilation of the byte-element
                            # reverse pointers may separate both from the model.
                            assert entry in METAL_BYTE_REVERSE_ENTRIES, (
                                case,
                                deviations,
                            )
                            evidence.append(
                                {
                                    "case": case,
                                    **data["metadata"],
                                    "elementCount": data["size"],
                                    "observedInputCount": observed,
                                    "upstream": upstream,
                                }
                            )
                            continue
                    worst = _check(case, target, entry, output["values"], data)
                    evidence.append(
                        {
                            "case": case,
                            **data["metadata"],
                            "elementCount": data["size"],
                            "worstBoundFraction": worst,
                            "observedInputCount": observed,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "comparison": _expected_validation_contract()["comparison"],
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.air", "*.metallib", "out"),
            )


def test_ci_requires_native_scan_family_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX scan family native parity", group="scan"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/scan-family",
        "--junitxml=mlx-current-results/scan-family.xml",
        "demos/integrations/mlx/tests/kernels/test_current_scan_family.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_scan_family.py"
    )
