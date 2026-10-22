"""Translate and execute the pinned MLX rotary position embedding kernels everywhere."""

import hashlib
import itertools
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_cross_entropy import (
    GUARD_BITS,
    GUARD_COUNT,
    _bits,
    _decode,
    _element_layout,
    _from_bits,
    _guard_values,
    _quantize,
    _runtime_value,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/rope.metal"
SOURCE_SHA256 = "2b35221ebea033ff43100b85eaaddec5c0e34e3cfd52e7c692573764b2b7d0ab"
HEADER = "mlx/backend/metal/kernels/utils.h"
HEADER_SHA256 = "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ROPE"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/rope.current-tree.translation.json"
)
# instantiate_rope's order in rope.metal: each type instantiates the single
# kernels, then the general kernels with int32 and int64 (large) indices.
DTYPES = ("float16", "bfloat16", "float32")
KERNELS = (
    "rope_single_",
    "rope_single_freqs_",
    "rope_",
    "rope_freqs_",
    "rope_large_",
    "rope_freqs_large_",
)
ENTRIES = tuple(kernel + dtype for dtype in DTYPES for kernel in KERNELS)
TARGETS = ("metal", "opengl", "directx")
# mlx/backend/metal/rope.cpp specializes three function constants.
CONSTANT_IDS = {"forward": 1, "traditional": 2, "hs_transpose": 3}
# rope.cpp's n_per_thread, rope_impl's N: each general thread rotates up to
# four heads.
HEADS_PER_THREAD = 4
# rope.cpp dispatches exactly one thread per rotated pair; one-thread
# workgroups give OpenGL and DirectX the same threads_per_grid.
WORKGROUP_SIZE = (1, 1, 1)
SIMD_SIZE = 32
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# composed element indices to signed int range.
SIGNED_MAX = 2**31 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "single": ("freq_stride * pos.x",),
    "general": (
        "batch_idx * offset_stride",
        "freq_stride * pos.x",
        "in_index_1",
        "in_index_2",
        "out_index_1",
        "out_index_2",
    ),
}


def _dtype(entry):
    return entry.rsplit("_", 1)[1]


def _kernel(entry):
    return entry[: -len(_dtype(entry))]


def _kind(entry):
    return "single" if _kernel(entry).startswith("rope_single_") else "general"


def _freqs(entry):
    return "_freqs_" in _kernel(entry)


def _large(entry):
    return "_large_" in _kernel(entry)


def _constant_names(entry):
    """Return the function constants ``entry`` reads.

    The single kernels never read hs_transpose; rope.cpp only sets it for
    strided four-dimensional inputs, which take the general kernels.
    """
    names = ("forward", "traditional")
    return names + (("hs_transpose",) if _kind(entry) == "general" else ())


def _variant_key(constants):
    """Name a specialization after MLX's pipeline hash."""
    parts = ["forward" if constants["forward"] else "vjp"]
    if constants["traditional"]:
        parts.append("traditional")
    if constants.get("hs_transpose"):
        parts.append("transpose")
    return "-".join(parts)


def _combinations(entry):
    names = _constant_names(entry)
    return tuple(
        dict(zip(names, values))
        for values in itertools.product((True, False), repeat=len(names))
    )


VARIANTS = tuple(
    (entry, _variant_key(constants), constants)
    for entry in ENTRIES
    for constants in _combinations(entry)
)

# rope.cpp takes the single kernels for one contiguous decode step (T = 1, one
# offset): name -> (rows B * N, row length D, rotated dims, offset, scale,
# base, freq stride). With dims < D MLX first copies the input into the output
# and rotates that copy in place.
SINGLE_CASES = {
    "decode": (3, 8, 8, 5, 1.0, 10000.0, 1),
    "partial": (2, 12, 8, 17, 0.5, 10000.0, 2),
    "start": (2, 32, 32, 0, 1.0, 10000.0, 1),
    "negative-offset": (1, 16, 16, -3, 2.0, 1000000.0, 2),
    "long-context": (2, 64, 64, 4000, 1.0, 10000.0, 1),
}
# The general kernels: name -> (layout, batch B, heads N, sequence T, row
# length D, rotated dims, offsets, scale, base, freq stride). One offset is a
# scalar (offset_stride 0); one per batch has offset_stride 1. "contiguous"
# is a row-contiguous [B, N, T, D] input, "strided" a non-contiguous 3-D
# [B, T, D] view reading every second element of [B, T, 2D] storage, and
# "transpose" a [B, T, N, D] buffer viewed as [B, N, T, D], the one layout for
# which rope.cpp sets hs_transpose.
GENERAL_CASES = {
    "prefill": ("contiguous", 2, 3, 5, 8, 8, (3,), 1.0, 10000.0, 1),
    "heads": ("contiguous", 1, 9, 3, 16, 16, (0,), 1.0, 10000.0, 2),
    "batch-offsets": ("contiguous", 3, 2, 4, 32, 32, (1, 2, 3), 1.0, 10000.0, 1),
    "partial": ("contiguous", 2, 1, 6, 12, 8, (7,), 0.25, 10000.0, 1),
    "strided": ("strided", 2, 1, 4, 8, 8, (2,), 2.0, 1000000.0, 2),
    "long-context": ("contiguous", 1, 2, 3, 64, 64, (4000,), 1.0, 10000.0, 1),
    "transpose": ("transpose", 2, 3, 5, 8, 8, (2,), 1.0, 10000.0, 1),
    "transpose-offsets": ("transpose", 2, 6, 4, 16, 16, (5, 9), 0.5, 10000.0, 2),
}
CASES = {"single": SINGLE_CASES, "general": GENERAL_CASES}
INPUT_GUARD_COUNT = 16
# Reading past the offsets yields a position far from any case.
OFFSET_GUARD = 1 << 20
# Each output is checked against a float64 rotation with a per-element bound on
# the kernel's own rounding. Additions, subtractions, products and int-to-float
# conversions are within OP_ERROR (one binary32 unit in the last place,
# relative), divisions within DIVISION_ERROR (2.5 units), exp2 within the
# larger of four and 3 + 2|y| units (GLSL's exp2 bound, D3D's 21 bits, Metal's
# precise exp2) and the angle error propagates through exp2. Metal's
# fast::cos and fast::sin, which translated Metal keeps and OpenGL and DirectX
# lower to their default cos and sin, may be TRIG_ABSOLUTE + TRIG_RELATIVE *
# |theta| from the exact value. That is at least twice the worst error measured
# on the pinned upstream kernel's arguments for Apple's fast functions (4.0e-7
# on [-pi, pi], growing by 2.7e-7 per radian), Mesa llvmpipe's cos/sin and
# WARP's, whose sin of small negative angles is up to 2.4e-6 from the exact
# value, about five units in the last place of 2 pi.
# Results must then be a correct rounding to the storage type of a value within
# the bound; elements rope does not rotate must keep their exact bits.
OP_ERROR = 2.0**-23
DIVISION_ERROR = 2.5 * 2.0**-23
EXP2_UNITS = 4.0
TRIG_ABSOLUTE = 2.0**-17
TRIG_RELATIVE = 2.0**-20
TINY = 2.0**-149
BOUND_SAFETY = 1.0 + 2.0**-10
SIGNIFICAND_BITS = {"float32": 24, "float16": 11, "bfloat16": 8}
# frexp exponent of the smallest normal value of each type.
MIN_NORMAL_EXPONENT = {"float32": -125, "float16": -13, "bfloat16": -125}


def _transposed(name):
    return GENERAL_CASES.get(name, ("",))[0] == "transpose"


def _variant_cases(entry, constants):
    """Return the cases of one specialization.

    rope.cpp sets hs_transpose exactly for the transposed layout, so the
    general kernels run transposed cases only with it set.
    """
    if _kind(entry) == "single":
        return tuple(SINGLE_CASES)
    return tuple(
        name for name in GENERAL_CASES if _transposed(name) == constants["hs_transpose"]
    )


def _stream(name, salt):
    state = (zlib.crc32(f"{name}:{salt}".encode()) << 1) | 1
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        yield (state >> 11) / 2**53


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _geometry(kind, name):
    """Return the logical shape [B, N, T, D] and the rotated dims of a case."""
    case = CASES[kind][name]
    if kind == "single":
        rows, length, dims = case[:3]
        return (1, rows, 1, length), dims
    _, batch, heads, sequence, length, dims = case[:6]
    return (batch, heads, sequence, length), dims


def _logical_index(shape, b, h, t, i):
    _, heads, sequence, length = shape
    return ((b * heads + h) * sequence + t) * length + i


def _layout(kind, name):
    """Return the input's kernel strides and each logical element's address."""
    shape, _ = _geometry(kind, name)
    batch, heads, sequence, length = shape
    layout = "contiguous" if kind == "single" else GENERAL_CASES[name][0]
    if layout == "contiguous":
        strides = (sequence * length, length, 1)
        size = batch * heads * sequence * length

        def address(b, h, t, i):
            return _logical_index(shape, b, h, t, i)

    elif layout == "strided":
        assert heads == 1  # a 3-D input has a single head
        strides = (sequence * 2 * length, 2 * length, 2)
        size = batch * sequence * 2 * length

        def address(b, h, t, i):
            return b * strides[0] + t * strides[1] + i * strides[2]

    else:
        assert layout == "transpose"
        strides = (length, heads * length, 1)
        size = batch * sequence * heads * length

        def address(b, h, t, i):
            return ((b * sequence + t) * heads + h) * length + i

    return strides, size, address


def _case(entry, name):
    """Return the host arguments and buffers of one case as MLX binds them."""
    kind, dtype = _kind(entry), _dtype(entry)
    case = CASES[kind][name]
    shape, dims = _geometry(kind, name)
    batch, heads, sequence, length = shape
    strides, size, address = _layout(kind, name)
    values = _stream(f"{kind}:{name}", f"x:{dtype}")
    logical = [
        _quantize(dtype, 2.0 * next(values) - 1.0)
        for _ in range(batch * heads * sequence * length)
    ]
    physical = [math.nan] * size
    for b in range(batch):
        for h in range(heads):
            for t in range(sequence):
                for i in range(length):
                    physical[address(b, h, t, i)] = logical[
                        _logical_index(shape, b, h, t, i)
                    ]
    if kind == "single":
        offsets, scale, base, freq_stride = (case[3],), case[4], case[5], case[6]
        grid = (dims // 2, heads, 1)
    else:
        offsets, scale, base, freq_stride = case[6:10]
        grid = (dims // 2, sequence, batch * -(-heads // HEADS_PER_THREAD))
    half = dims // 2
    noise = _stream(f"{kind}:{name}", "freqs")
    freqs = [
        _float32(base ** (pair / half) * (0.75 + 0.5 * next(noise)))
        for pair in range(half)
    ]
    freq_buffer = []
    for value in freqs:
        freq_buffer.extend([value] + [math.nan] * (freq_stride - 1))
    # Partial rotations rotate a copy of the input; other outputs start as
    # guards, so an unwritten element fails.
    guard = _from_bits(dtype, GUARD_BITS[dtype])
    initial = list(logical) if dims < length else [guard] * len(logical)
    return {
        "shape": shape,
        "dims": dims,
        "grid": grid,
        "logical": logical,
        "input": physical,
        "initial": initial,
        "strides": strides,
        "outStrides": (sequence * length, length, 1),
        "offsets": offsets,
        "offsetStride": 0 if len(offsets) == 1 else 1,
        "scale": _float32(scale),
        # rope.cpp passes log2(base) as a float.
        "base": _float32(math.log2(base)),
        "freqs": freqs,
        "freqBuffer": freq_buffer,
        "freqStride": freq_stride,
        "heads": heads,
    }


def _pairs(constants, dims):
    half = dims // 2
    for pair in range(half):
        if constants["traditional"]:
            yield pair, 2 * pair, 2 * pair + 1
        else:
            yield pair, pair, pair + half


def _inverse_frequency(entry, data, pair):
    """Return the exact inverse frequency and its relative error bound."""
    if _freqs(entry):
        return 1.0 / data["freqs"][pair], DIVISION_ERROR
    half = data["dims"] // 2
    exponent = -(pair / half) * data["base"]
    # d = pos.x / grid.x divides and -d * base rounds once; exp2 adds its own
    # error at the rounded argument.
    shift = abs(exponent) * ((1 + DIVISION_ERROR) * (1 + OP_ERROR) - 1)
    growth = math.expm1(math.log(2.0) * shift)
    units = max(EXP2_UNITS, 3.0 + 2.0 * (abs(exponent) + shift))
    return 2.0**exponent, (1 + growth) * (1 + units * OP_ERROR) - 1


def _rotation(entry, constants, data, b, t, pair):
    """Return the exact angle, cos, sin and their absolute error bound."""
    if _kind(entry) == "single":
        position = data["offsets"][0]
    else:
        position = t + data["offsets"][b * data["offsetStride"]]
    inverse, inverse_error = _inverse_frequency(entry, data, pair)
    theta = data["scale"] * position * inverse
    # L = scale * position and theta = L * inv_freq each round once.
    angle_error = abs(theta) * ((1 + OP_ERROR) ** 2 * (1 + inverse_error) - 1)
    trig = TRIG_ABSOLUTE + TRIG_RELATIVE * (abs(theta) + angle_error)
    return theta, math.cos(theta), math.sin(theta), angle_error + trig


def _combine(a, u, b, v, error):
    """Bound a * u + b * v computed from approximations within ``error``.

    Each product and the sum round once, or the sum fuses with one product.
    """
    exact = a * u + b * v
    approximation = (abs(a) + abs(b)) * error
    rounding = OP_ERROR * (abs(a) * (abs(u) + error) + abs(b) * (abs(v) + error))
    rounding += OP_ERROR * (abs(exact) + approximation)
    return exact, (approximation + rounding + 3 * TINY) * BOUND_SAFETY


def _reference(entry, constants, name):
    """Return the case, float64 outputs and each output's bound.

    A bound of None requires the element's exact initial bits.
    """
    data = _case(entry, name)
    shape, dims = data["shape"], data["dims"]
    batch, heads, sequence, _ = shape
    expected = list(data["initial"])
    bounds = [None] * len(expected)
    for b in range(batch):
        for h in range(heads):
            for t in range(sequence):
                for pair, first, second in _pairs(constants, dims):
                    _, cos, sin, error = _rotation(entry, constants, data, b, t, pair)
                    i1 = _logical_index(shape, b, h, t, first)
                    i2 = _logical_index(shape, b, h, t, second)
                    x1, x2 = data["logical"][i1], data["logical"][i2]
                    if constants["forward"]:
                        out1 = _combine(x1, cos, x2, -sin, error)
                        out2 = _combine(x1, sin, x2, cos, error)
                    else:
                        out1 = _combine(x2, sin, x1, cos, error)
                        out2 = _combine(x2, cos, x1, -sin, error)
                    (expected[i1], bounds[i1]), (expected[i2], bounds[i2]) = (
                        out1,
                        out2,
                    )
    return data, expected, bounds


def _simulate(entry, constants, name, mutation=None):
    """Run rope.metal's index arithmetic on the bound buffers in float64.

    ``mutation`` injects one defect; results round to the storage type.
    """
    kind, dtype = _kind(entry), _dtype(entry)
    data = _case(entry, name)
    constants = dict(constants)
    flips = {"direction": "forward", "pairing": "traditional", "layout": "hs_transpose"}
    if mutation in flips:
        constants[flips[mutation]] = not constants[flips[mutation]]
    source = data["input"] + [math.nan] * INPUT_GUARD_COUNT
    offsets = list(data["offsets"]) + [OFFSET_GUARD] * INPUT_GUARD_COUNT
    freq_buffer = data["freqBuffer"] + [math.nan] * INPUT_GUARD_COUNT
    freq_stride = 1 if mutation == "freq-stride" else data["freqStride"]
    offset_stride = 0 if mutation == "batch-offset" else data["offsetStride"]
    scale = 1.0 if mutation == "scale" else data["scale"]
    strides = data["outStrides"] if mutation == "strides" else data["strides"]
    out_strides = data["outStrides"]
    output = list(data["initial"])
    grid = data["grid"]

    def angle(step, offset, x):
        if _freqs(entry):
            inverse = 1.0 / freq_buffer[freq_stride * x]
        else:
            numerator = x + 1 if mutation == "frequency" else x
            inverse = 2.0 ** (-(numerator / grid[0]) * data["base"])
        position = step + (0 if mutation == "offset" else offset)
        theta = scale * position * inverse
        if mutation == "binary16-angle":
            theta = struct.unpack("<e", struct.pack("<e", theta))[0]
        return math.cos(theta), math.sin(theta)

    def rotate(cos, sin, x1, x2):
        if constants["forward"]:
            return x1 * cos - x2 * sin, x1 * sin + x2 * cos
        return x2 * sin + x1 * cos, x2 * cos - x1 * sin

    def store(index, value):
        output[index] = _quantize(dtype, value)

    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                if kind == "single":
                    cos, sin = angle(0, data["offsets"][0], x)
                    if constants["traditional"]:
                        index_1 = 2 * x + y * data["outStrides"][1]
                        index_2 = index_1 + 1
                    else:
                        index_1 = x + y * data["outStrides"][1]
                        index_2 = index_1 + grid[0]
                    rx1, rx2 = rotate(cos, sin, source[index_1], source[index_2])
                    store(index_1, rx1)
                    store(index_2, rx2)
                    continue
                n_head = data["heads"]
                n_head_up = HEADS_PER_THREAD * -(-n_head // HEADS_PER_THREAD)
                head = (z * HEADS_PER_THREAD) % n_head_up
                batch = (z * HEADS_PER_THREAD) // n_head_up
                cos, sin = angle(y, offsets[batch * offset_stride], x)
                matrix = batch * n_head + head
                if constants["hs_transpose"]:
                    in_1 = batch * grid[1] * strides[1] + y * strides[1]
                    in_1 += head * strides[0]
                else:
                    in_1 = y * strides[1] + matrix * strides[0]
                out_1 = y * out_strides[1] + matrix * out_strides[0]
                if constants["traditional"]:
                    out_1 += 2 * x * out_strides[2]
                    out_2 = out_1 + 1
                    in_1 += 2 * x * strides[2]
                    in_2 = in_1 + strides[2]
                else:
                    out_1 += x * out_strides[2]
                    out_2 = out_1 + grid[0] * out_strides[2]
                    in_1 += x * strides[2]
                    in_2 = in_1 + grid[0] * strides[2]
                for i in range(HEADS_PER_THREAD):
                    if head + i >= n_head:
                        break
                    rx1, rx2 = rotate(cos, sin, source[in_1], source[in_2])
                    store(out_1, rx1)
                    store(out_2, rx2)
                    in_1 += strides[0]
                    in_2 += strides[0]
                    out_1 += out_strides[0]
                    out_2 += out_strides[0]
    return output


def _round(dtype, value):
    """Round a float64 value to ``dtype`` with ties to even, in one step."""
    if value == 0 or not math.isfinite(value):
        return value
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])
    return round(value / quantum) * quantum


def _ulp(dtype, value):
    exponent = max(math.frexp(value)[1], MIN_NORMAL_EXPONENT[dtype])
    return 2.0 ** (exponent - SIGNIFICAND_BITS[dtype])


def _check(dtype, actual, expected, bounds):
    """Check results against the reference.

    Return the largest fraction of an output's bound that the results use
    beyond the final rounding's half unit in the last place, so correctly
    rounded results report zero.
    """
    assert len(actual) == len(expected) == len(bounds)
    bad, worst = [], 0.0
    for index, (got, want, bound) in enumerate(zip(actual, expected, bounds)):
        if bound is None:
            if _bits(dtype, got) != _bits(dtype, want) or math.isnan(got):
                bad.append((index, got, want, "unrotated"))
            continue
        low, high = _round(dtype, want - bound), _round(dtype, want + bound)
        if math.isnan(got) or not low <= got <= high:
            bad.append((index, got, want, low, high))
        elif got != want:
            half = max(_ulp(dtype, want), _ulp(dtype, got)) / 2
            excess = max(0.0, abs(got - want) - half)
            if excess:
                worst = max(worst, excess / bound)
    assert not bad, bad[:8]
    return worst


def _config(output, target, entry, constants):
    options = "\n[project.specialization_constants]\n"
    for name, value in constants.items():
        options += f'"{CONSTANT_IDS[name]}" = {str(value).lower()}\n'
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS[_kind(entry)]:
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    size = ", ".join(str(item) for item in WORKGROUP_SIZE)
    return f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [{size}]
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_contract():
    single = {
        name: {
            "rows": rows,
            "rowLength": length,
            "dims": dims,
            "offset": offset,
            "scale": scale,
            "base": base,
            "freqStride": freq_stride,
        }
        for name, (
            rows,
            length,
            dims,
            offset,
            scale,
            base,
            freq_stride,
        ) in SINGLE_CASES.items()
    }
    general = {
        name: {
            "layout": layout,
            "batch": batch,
            "heads": heads,
            "sequence": sequence,
            "rowLength": length,
            "dims": dims,
            "offsets": list(offsets),
            "scale": scale,
            "base": base,
            "freqStride": freq_stride,
            "hsTranspose": layout == "transpose",
        }
        for name, (
            layout,
            batch,
            heads,
            sequence,
            length,
            dims,
            offsets,
            scale,
            base,
            freq_stride,
        ) in GENERAL_CASES.items()
    }
    return {"single": single, "general": general}


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "headsPerThread": HEADS_PER_THREAD,
        "specializationConstants": {
            name: {"id": identifier, "values": [True, False]}
            for name, identifier in CONSTANT_IDS.items()
        },
        "indexRangeAssertions": {
            "opengl": {
                kind: [
                    {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                    for expression in expressions
                ]
                for kind, expressions in OPENGL_INDEX_RANGE_ASSERTIONS.items()
            },
            "metal": {},
            "directx": {},
        },
        "cases": _case_contract(),
        "errorModel": {
            "operationError": OP_ERROR,
            "divisionError": DIVISION_ERROR,
            "exp2Units": "max(4, 3 + 2|y|)",
            "trigAbsolute": TRIG_ABSOLUTE,
            "trigRelative": TRIG_RELATIVE,
            "tiny": TINY,
        },
        "fastMathLowering": "metal-keeps-fast-cos-sin-others-use-default-cos-sin",
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "unrotatedElements": "exact-bits-of-the-input-copy",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _dispatch_count():
    return sum(
        len(_variant_cases(entry, constants)) for entry, _, constants in VARIANTS
    )


def _checked_elements(entry):
    total = 0
    for item, _, constants in VARIANTS:
        if item != entry:
            continue
        for name in _variant_cases(entry, constants):
            shape, _ = _geometry(_kind(entry), name)
            total += math.prod(shape)
    return total


def _expected_scope():
    return {
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(VARIANTS),
        "dispatchCountByTarget": _dispatch_count(),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        # rope.cpp selects the int64-index kernels above INT32_MAX elements;
        # the gate runs them on the same small cases as their int32 forms.
        "largeIndexKernelsUseSmallCases": True,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_rope_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-rope-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
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
        for entry, records in artifacts.items():
            assert tuple(records) == tuple(
                key for item, key, _ in VARIANTS if item == entry
            )
            for record in records.values():
                assert set(record) == {"sha256", "sizeBytes"}
                assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    # Metal keeps function constants and OpenGL specialization constants, so
    # every specialization of an entry shares one artifact; DirectX
    # materializes each.
    for entry in ENTRIES:
        for target in ("metal", "opengl"):
            records = contract["artifacts"][target][entry]
            assert len({record["sha256"] for record in records.values()}) == 1
        records = contract["artifacts"]["directx"][entry]
        assert len({record["sha256"] for record in records.values()}) == len(records)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_rope_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "rope.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 18
    assert ENTRIES[:6] == (
        "rope_single_float16",
        "rope_single_freqs_float16",
        "rope_float16",
        "rope_freqs_float16",
        "rope_large_float16",
        "rope_freqs_large_float16",
    )
    assert [_dtype(entry) for entry in ENTRIES[::6]] == list(DTYPES)
    assert [_kind(entry) for entry in ENTRIES[:6]] == ["single"] * 2 + ["general"] * 4
    assert [_large(entry) for entry in ENTRIES[:6]] == [False] * 4 + [True] * 2
    assert [_freqs(entry) for entry in ENTRIES[:6]] == [False, True] * 3
    # Four forward/traditional specializations per single kernel and eight
    # with hs_transpose per general kernel.
    assert len(VARIANTS) == 3 * (2 * 4 + 4 * 8)
    assert len({(entry, key) for entry, key, _ in VARIANTS}) == len(VARIANTS)
    assert {key for _, key, _ in VARIANTS} == {
        direction + traditional + transpose
        for direction in ("forward", "vjp")
        for traditional in ("", "-traditional")
        for transpose in ("", "-transpose")
    }
    assert {key for entry, key, _ in VARIANTS if _kind(entry) == "single"} == {
        "forward",
        "forward-traditional",
        "vjp",
        "vjp-traditional",
    }


def test_rope_cases_follow_mlx_dispatch():
    for entry, _, constants in VARIANTS:
        assert len(_variant_cases(entry, constants)) >= 2, (entry, constants)
    assert _dispatch_count() == 3 * (2 * 4 * 5 + 4 * (4 * 6 + 4 * 2))
    for kind, cases in CASES.items():
        for name in cases:
            data = _case(f"rope_{'single_' if kind == 'single' else ''}float32", name)
            batch, heads, sequence, length = data["shape"]
            dims = data["dims"]
            assert dims % 2 == 0 and 2 <= dims <= length
            if kind == "single":
                assert data["grid"] == (dims // 2, batch * heads, 1)
                continue
            assert data["grid"] == (
                dims // 2,
                sequence,
                batch * math.ceil(heads / HEADS_PER_THREAD),
            )
            assert len(data["offsets"]) in {1, batch}
            if _transposed(name):
                # rope.cpp's condition: batch stride T * N * D, the head axis
                # at stride D and the sequence axis at N * D.
                assert data["strides"] == (length, heads * length, 1)
    # Ragged head groups, per-batch offsets, partial rotations, strided and
    # transposed inputs, negative and zero offsets, a long context, scales
    # other than one and strided frequencies all appear.
    general = GENERAL_CASES.values()
    assert {case[2] % HEADS_PER_THREAD for case in general} >= {1, 2, 3}
    assert any(len(case[6]) > 1 for case in general if case[0] == "contiguous")
    assert any(len(case[6]) > 1 for case in general if case[0] == "transpose")
    assert {case[0] for case in general} == {"contiguous", "strided", "transpose"}
    for cases, dims_index, length_index in (
        (SINGLE_CASES.values(), 2, 1),
        (general, 5, 4),
    ):
        cases = tuple(cases)
        assert any(case[dims_index] < case[length_index] for case in cases)
        assert any(case[-1] == 2 for case in cases)
        assert any(case[-1] == 1 for case in cases)
        assert any(case[-3] != 1.0 for case in cases)
    assert min(case[3] for case in SINGLE_CASES.values()) < 0
    assert {0, 4000} <= {case[3] for case in SINGLE_CASES.values()}
    assert max(max(case[6]) for case in general) == 4000


def test_rope_reference_matches_kernel_index_arithmetic():
    # The reference rotates the logical tensor; the simulation runs the
    # kernel's own index arithmetic on the bound buffers. Exact rotations,
    # rounded to binary32 and then to the storage type as the kernel's final
    # conversion does, use a negligible part of the bound.
    for entry, _, constants in VARIANTS:
        for name in _variant_cases(entry, constants):
            data, expected, bounds = _reference(entry, constants, name)
            simulated = _simulate(entry, constants, name)
            assert _check(_dtype(entry), simulated, expected, bounds) < 2.0**-10
            rotated = sum(bound is not None for bound in bounds)
            batch, heads, sequence, _ = data["shape"]
            assert rotated == batch * heads * sequence * data["dims"]


def _mutations(entry, name):
    """Return the defects that change ``entry``'s results on case ``name``."""
    kind = _kind(entry)
    data = _case(entry, name)
    batch, heads, sequence, _ = data["shape"]
    # Some position, and so some angle, is nonzero; a zero angle is the
    # identity under any pairing or direction.
    moving = any(data["offsets"]) or sequence > 1
    mutations = []
    if moving:
        mutations += ["pairing", "direction", "binary16-angle"]
        if not _freqs(entry):
            mutations.append("frequency")
    if any(data["offsets"]):
        mutations.append("offset")
    if data["scale"] != 1.0 and moving:
        mutations.append("scale")
    if _freqs(entry) and data["freqStride"] > 1:
        mutations.append("freq-stride")
    if kind == "general":
        if len(set(data["offsets"])) > 1:
            mutations.append("batch-offset")
        if batch > 1 and heads > 1:
            mutations.append("layout")
        if data["strides"] != data["outStrides"]:
            mutations.append("strides")
    return mutations


@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_checks_reject_defects(dtype):
    rejected = set()
    for entry, _, constants in VARIANTS:
        if _dtype(entry) != dtype:
            continue
        for name in _variant_cases(entry, constants):
            _, expected, bounds = _reference(entry, constants, name)
            exact = _simulate(entry, constants, name)
            guard = _from_bits(dtype, GUARD_BITS[dtype])
            # An unwritten output and a changed unrotated element fail.
            first = next(index for index, bound in enumerate(bounds) if bound)
            unwritten = exact[:first] + [guard] + exact[first + 1 :]
            with pytest.raises(AssertionError):
                _check(dtype, unwritten, expected, bounds)
            if None in bounds:
                tail = bounds.index(None)
                changed = list(exact)
                changed[tail] = -changed[tail] if changed[tail] else 1.0
                with pytest.raises(AssertionError):
                    _check(dtype, changed, expected, bounds)
            for mutation in _mutations(entry, name):
                if mutation == "binary16-angle" and dtype != "float32":
                    continue  # within a 16-bit output's own rounding
                wrong = _simulate(entry, constants, name, mutation)
                with pytest.raises(AssertionError):
                    _check(dtype, wrong, expected, bounds)
                rejected.add(mutation)
    expected_mutations = {
        "pairing",
        "direction",
        "frequency",
        "offset",
        "scale",
        "freq-stride",
        "batch-offset",
        "layout",
        "strides",
    }
    if dtype == "float32":
        expected_mutations.add("binary16-angle")
    assert rejected == expected_mutations


def test_rope_bounds_stay_meaningful():
    # Angles up to 10 radians keep every bound below 2^-14 for these unit-range
    # pairs, under one binary16 unit; a 4,000-position context widens it to
    # 2^-6 because the exp2 error scales with the angle and the fast trig error
    # with its argument, as for upstream MLX.
    for entry, _, constants in VARIANTS:
        for name in _variant_cases(entry, constants):
            _, _, bounds = _reference(entry, constants, name)
            limit = 2.0**-6 if name == "long-context" else 2.0**-14
            assert max(bound for bound in bounds if bound) < limit, (entry, name)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_rope_configs_scope_target_options(tmp_path, target, entry, key, constants):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry, constants), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert [item.to_json() for item in config.index_range_assertions] == (
        [
            {
                "source": SOURCE,
                "expression": expression,
                "minimum": 0,
                "maximum": SIGNED_MAX,
            }
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS[_kind(entry)]
        ]
        if target == "opengl"
        else []
    )
    text = path.read_text(encoding="utf-8")
    assert f'"{entry}" = [1, 1, 1]' in text
    for name, identifier in CONSTANT_IDS.items():
        line = f'"{identifier}" = '
        if name in constants:
            assert line + str(constants[name]).lower() in text
        else:
            assert line not in text
    assert "software_subgroup_width" not in text


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native rope gate")
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
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256((root / HEADER).read_bytes()).hexdigest() == HEADER_SHA256
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("rope-metal-reference")
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


def _padded(values):
    return list(values) + [math.nan] * INPUT_GUARD_COUNT


def _metal_payload(dtype, values):
    if dtype == "float32":
        return struct.pack(f"<{len(values)}I", *(_bits(dtype, v) for v in values))
    return struct.pack(f"<{len(values)}H", *(_bits(dtype, v) for v in values))


def _metal_words(dtype, raw):
    if dtype == "float32":
        return struct.unpack(f"<{len(raw) // 4}I", raw)
    return struct.unpack(f"<{len(raw) // 2}H", raw)


def _offsets(entry, data):
    if _kind(entry) == "single":
        return [data["offsets"][0]]
    return list(data["offsets"]) + [OFFSET_GUARD] * INPUT_GUARD_COUNT


def _upstream_metal(work, reference, entry, constants, name, data):
    """Run the unchanged upstream kernel with rope.cpp's buffer indices."""
    runner, library = reference
    dtype = _dtype(entry)
    offsets = _offsets(entry, data)
    payloads = {
        0: _metal_payload(dtype, _padded(data["input"])),
        1: _metal_payload(
            dtype,
            data["initial"] + [_from_bits(dtype, GUARD_BITS[dtype])] * GUARD_COUNT,
        ),
        2: struct.pack(f"<{len(offsets)}i", *offsets),
        3: struct.pack("<f", data["scale"]),
    }
    if _kind(entry) == "single":
        payloads[4] = struct.pack("<q", data["outStrides"][1])
    else:
        payloads[4] = struct.pack("<3q", *data["strides"])
        payloads[5] = struct.pack("<3q", *data["outStrides"])
        payloads[6] = struct.pack("<q", data["offsetStride"])
        payloads[7] = struct.pack("<i", data["heads"])
    if _freqs(entry):
        payloads[10] = _metal_payload("float32", _padded(data["freqBuffer"]))
        payloads[11] = struct.pack("<q", data["freqStride"])
    else:
        payloads[10] = struct.pack("<f", data["base"])
    files = []
    for index in range(max(payloads) + 1):
        # The runner binds buffer i at index i; unused indices hold zeros.
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payloads.get(index, bytes(16)))
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": list(data["grid"]),
        "workgroupSize": list(WORKGROUP_SIZE),
        "simdWidth": SIMD_SIZE,
        # rope.cpp always specializes all three constants.
        "boolConstants": [
            {"index": identifier, "value": constants.get(constant, False)}
            for constant, identifier in CONSTANT_IDS.items()
        ],
    }
    request_path = work / f"{name}-upstream-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request_path, output], work, f"{name}-upstream")
    )
    for index, path in enumerate(files):
        if index != 1:
            assert (output / f"buffer-{index}.bin").read_bytes() == Path(
                path
            ).read_bytes()
    words = _metal_words(dtype, (output / "buffer-1.bin").read_bytes())
    count = len(data["initial"])
    assert list(words[count:]) == [GUARD_BITS[dtype]] * GUARD_COUNT
    return device, [_from_bits(dtype, word) for word in words[:count]]


def _member(binding, entry):
    layout = binding["scalarLayout"]
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry) + "_"
    member = layout.get("memberName", binding["name"]).removeprefix(prefix)
    return member.removesuffix("Buffer").rstrip("_")


def _dispatch(descriptor, package, target, entry, data):
    dtype = _dtype(entry)
    count = len(data["initial"])
    integers = {"offset": ("int32", _offsets(entry, data))}
    if _kind(entry) == "single":
        integers["stride"] = ("int64", [data["outStrides"][1]])
    else:
        integers.update(
            strides=("int64", list(data["strides"])),
            out_strides=("int64", list(data["outStrides"])),
            offset_stride=("int64", [data["offsetStride"]]),
            n_head=("int32", [data["heads"]]),
        )
    floats = {"scale": data["scale"]}
    if _freqs(entry):
        integers["freq_stride"] = ("int64", [data["freqStride"]])
    else:
        floats["base"] = data["base"]
    values, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        member = _member(binding, entry)
        element_type, encoding = _element_layout(binding)
        if member == "out":
            value = _runtime_value(dtype, element_type, data["initial"], encoding)
            guards = _guard_values(dtype, element_type)
            value["values"] = value["values"][:count] + guards
            value["shape"] = [count + GUARD_COUNT]
            output = (binding["name"], value["dtype"], count, guards)
        elif member == "in":
            value = _runtime_value(
                dtype, element_type, _padded(data["input"]), encoding
            )
        elif member == "freqs":
            assert element_type == "float32" and encoding is None, binding
            value = _runtime_value(
                "float32", element_type, _padded(data["freqBuffer"]), None
            )
        elif member in floats:
            assert element_type == "float32" and encoding is None, binding
            value = _runtime_value("float32", element_type, [floats[member]], None)
        else:
            expected_type, words = integers[member]
            assert element_type == expected_type, (member, element_type)
            value = {"dtype": element_type, "shape": [len(words)], "values": words}
        if member in {"in", "out"} and encoding is not None:
            assert encoding["logicalElementType"] == dtype, (member, encoding)
        elif member in {"in", "out"}:
            assert element_type in {dtype, "float32", "uint16"}, (member, element_type)
        values[binding["name"]] = value
        if member != "out":
            observed.append(binding["name"])
    assert output is not None
    assert len(observed) == len(integers) + len(floats) + 1 + _freqs(entry)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        {output[0]: dict(values[output[0]])},
        {"workgroupCount": list(data["grid"]), "workgroupSize": list(WORKGROUP_SIZE)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, len(observed)


def _assert_lowering(text, target, entry, constants):
    assert "unsupported" not in text
    if target == "metal":
        # rope.metal's fast::cos and fast::sin keep their namespace.
        assert text.count("metal::fast::cos(theta)") == 1
        assert text.count("metal::fast::sin(theta)") == 1
        assert text.count("fast::") == 2
    else:
        # OpenGL and DirectX call their default cos and sin.
        assert "fast::" not in text
    assert text.count("cos(theta)") == text.count("sin(theta)") == 1
    assert ("exp2(" in text) is not _freqs(entry)
    for name, value in constants.items():
        identifier = CONSTANT_IDS[name]
        if target == "metal":
            assert f"constant bool {name} [[function_constant({identifier})]];" in text
        elif target == "opengl":
            # Specialized when the pipeline is created, as on Metal.
            assert f"layout(constant_id = {identifier}) const bool {name} = " in text
        else:
            assert f"static const bool {name} = {str(value).lower()};" in text
    if "hs_transpose" not in constants:
        assert "function_constant(3)" not in text and "hs_transpose" not in text


@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_current_mlx_rope_native_parity(
    current_mlx, metal_reference, tmp_path, entry, key, constants
):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry][key]
    with tempfile.TemporaryDirectory(prefix=".current-rope-", dir=root) as path:
        work = Path(path)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry, constants),
                encoding="utf-8",
            )
            translation = translate_project(
                load_project_config(root, config),
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
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            check_generated_artifact(
                artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
            )
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
            generated = root / artifact["path"]
            _assert_lowering(
                generated.read_text(encoding="utf-8"), target, entry, constants
            )
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            assert {
                item["id"]: item["value"]
                for item in descriptor["specializationConstants"]
            } == {CONSTANT_IDS[name]: value for name, value in constants.items()}
            executor = _executor(target)
            evidence = []
            try:
                for name in _variant_cases(entry, constants):
                    data, expected, bounds = _reference(entry, constants, name)
                    request, output, observed = _dispatch(
                        descriptor, package, target, entry, data
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    (work / f"{name}-result.json").write_text(
                        json.dumps(
                            {
                                "status": result.status,
                                "outputs": result.outputs,
                                "details": result.details,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    assert result.status == "ok"
                    binding, element_type, written, guards = output
                    value = result.outputs[binding]
                    assert value["dtype"] == element_type
                    assert value["values"][written:] == guards
                    actual = _decode(
                        dtype, {**value, "values": value["values"][:written]}
                    )
                    used = _check(dtype, actual, expected, bounds)
                    observations = result.details["inputObservations"]
                    assert len(observations) == observed
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, upstream_values = _upstream_metal(
                            work, metal_reference, entry, constants, name, data
                        )
                        upstream = {
                            "device": device,
                            "boundFractionUsed": _check(
                                dtype, upstream_values, expected, bounds
                            ),
                            "bitIdenticalToTranslated": (
                                [_bits(dtype, item) for item in upstream_values]
                                == [_bits(dtype, item) for item in actual]
                            ),
                        }
                    evidence.append(
                        {
                            "case": name,
                            "shape": list(data["shape"]),
                            "dims": data["dims"],
                            "grid": list(data["grid"]),
                            "largestBound": max(bound for bound in bounds if bound),
                            "boundFractionUsed": used,
                            "outputGuardCount": GUARD_COUNT,
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
                        "dtype": dtype,
                        "specialization": key,
                        "constants": constants,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS[_kind(entry)])
                            if target == "opengl"
                            else []
                        ),
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if target == "metal":
                # The round trip keeps fast::cos and fast::sin, so the
                # translated kernel reproduces the upstream kernel bit for bit.
                assert [
                    case["case"]
                    for case in evidence
                    if not case["upstream"]["bitIdenticalToTranslated"]
                ] == []
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_rope_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX rope native parity", group="sequence"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/rope",
        "--junitxml=mlx-current-results/rope.xml",
        "demos/integrations/mlx/tests/kernels/test_current_rope.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_rope.py"
    )
