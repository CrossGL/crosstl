"""Natively prove fp_quantized.metal's 36 floating-point block quantization kernels.

``fp_quantized.metal`` instantiates ``fp_quantize_dequantize``, ``fp_quantize``
and ``fp_dequantize`` for float, bfloat16_t and float16_t in four modes:
``nvfp4`` (16-value groups, FP4 E2M1 codes, FP8 E4M3 scales) with and without
a global scale, ``mxfp8`` (32-value groups, E4M3 codes, E8M0 scales) and
``mxfp4`` (32-value groups, E2M1 codes, E8M0 scales): 36 of its 518 entries.
Each is translated and pinned per target, then runs two cases on 64-thread
workgroups, as ``quantize_impl`` and ``fp_quantize_dequantize`` in
``quantized.cpp`` bind them: a row of workgroups and a two-by-two grid, whose
``threads_per_grid`` row stride the kernels' flattened index uses.

Quantize groups put values on the element grid, exactly on every E2M1
rounding threshold (or between adjacent E4M3 values), at free dyadic values,
all negative and all signed zero. ``nvfp4`` groups also saturate the E4M3
scale and take its subnormal path, and ``mx`` groups need the E8M0 scale
rounded up. A lane-level model replays the simdgroup maxima (per 16-lane half
for ``nvfp4``), every correctly rounded binary32 operation, the scale and code
encoders bit for bit, the neighbouring-lane shuffle that packs two E2M1 codes
per byte and the guarded byte writes. Dequantize unpacks random codes,
including signed zeros and E4M3 subnormals, with E4M3 scales of either sign
or E8M0 scales. Every word is compared exactly, guards and inputs must be
unchanged, and Metal also runs the unchanged upstream kernels. The
matrix-vector and matrix-matrix kernels are outside this gate.
"""

import functools
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
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
from demos.integrations.mlx.tests.kernels.test_current_quantized import (
    _assert_identical,
    _f32,
    _kernel_parameters,
    _macro,
    _metal_round,
    _to_type,
    _words,
)
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _below,
    _draws,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FP_QUANTIZED"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/fp_quantized.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
SOURCE = "mlx/backend/metal/kernels/fp_quantized.metal"
SOURCE_SHA256 = "18ba5ebc68cb0ffa1a992f36a941d810bfdcf0163731a6351bd17bee076e2425"
HEADERS = {
    "mlx/backend/metal/kernels/fp_quantized.h": (
        "359a1ea9008f74cc15c61e9d39c6300dcffb9cfb9cc0ebd59ec7c00b789d719f"
    ),
    "mlx/backend/metal/kernels/fp4.h": (
        "e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a"
    ),
    "mlx/backend/metal/kernels/fp8.h": (
        "9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae"
    ),
    "mlx/backend/metal/kernels/quantized_utils.h": (
        "a12841b57d505f6cca81631901b845dbf76bc90dfc099e7f40c763b2cc838c51"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
}
# quantize_impl, fp_quantize_dequantize and get_quantize_kernel_dims, whose
# bindings and grid the cases follow.
HOST_SOURCE = "mlx/backend/metal/quantized.cpp"
HOST_SHA256 = "36b79607ee01bc213511dcdad590b2dc3b96099300551cbc3e7e6c7e13ba5982"
UNIT_ENTRY_COUNT = 518

# fp_quantized.metal's instantiation order: instantiate_quantized_types walks
# the element types, then instantiate_quantize_dequantize these modes, each
# with these three kernels.
KINDS = ("quantize_dequantize", "quantize", "dequantize")
DTYPES = (("float", "float32"), ("bfloat16_t", "bfloat16"), ("float16_t", "float16"))
# (mode, group size, bits, has_global_scale)
MODES = (
    ("nvfp4", 16, 4, False),
    ("nvfp4", 16, 4, True),
    ("mxfp8", 32, 8, False),
    ("mxfp4", 32, 4, False),
)
# Kernels of fp_quantized.metal this gate does not cover.
EXCLUDED_KERNELS = (
    "fp_qmv_fast",
    "fp_qmv",
    "fp_qvm",
    "fp_qmm_n",
    "fp_gather_qmv_fast",
    "fp_gather_qmv",
    "fp_gather_qvm",
    "fp_gather_qmm_n",
    "fp_qmv_quad",
    "fp_qmv_wide",
    "fp_qvm_split_k",
    "fp_qmm_t_splitk",
    "fp_gather_qmm_t",
    "fp_qmm_t",
    "fp_gather_qmm_rhs",
)

SIMD_SIZE = 32
WORKGROUP = (64, 1, 1)
CASES = ("row", "grid")
GRIDS = {
    "quantize_dequantize": {"row": (3, 1, 1), "grid": (2, 2, 1)},
    "quantize": {"row": (3, 1, 1), "grid": (2, 2, 1)},
    "dequantize": {"row": (2, 1, 1), "grid": (2, 2, 1)},
}
# [[buffer(i)]] parameters of the kernels, as (name, index).
PARAMETERS = {
    "quantize_dequantize": (("w", 0), ("global_scale", 1), ("out", 2)),
    "quantize": (("w", 0), ("out", 1), ("scales", 2), ("global_scale", 3)),
    "dequantize": (("w", 0), ("scales", 1), ("global_scale", 2), ("out", 3)),
}
OUTPUTS = {
    "quantize_dequantize": ("out",),
    "quantize": ("out", "scales"),
    "dequantize": ("out",),
}
# One scenario per quantization group, in grid order.
SCENARIOS = {
    "nvfp4": {
        "row": (
            "grid",
            "ties",
            "free",
            "negative",
            "zero",
            "saturate",
            "tiny",
            "free",
            "ties",
            "grid",
            "negative",
            "free",
        ),
        "grid": (
            "free",
            "grid",
            "tiny",
            "ties",
            "saturate",
            "zero",
            "negative",
            "free",
            "grid",
            "free",
            "ties",
            "tiny",
            "negative",
            "saturate",
            "free",
            "grid",
        ),
    },
    "mx": {
        "row": ("grid", "ties", "roundup", "free", "zero", "negative"),
        "grid": ("free", "roundup", "grid", "negative", "ties", "free", "zero", "grid"),
    },
}
# Global scales of the has_global_scale entries per case: quantize's
# F8E4M3_MAX * F4E2M1_MAX / g is 4 and 2.688, dequantize's g / 2688 is 4 and
# 3000 / 2688.
GLOBAL_SCALES = {
    "quantize": {"row": 672.0, "grid": 1000.0},
    "dequantize": {"row": 10752.0, "grid": 3000.0},
}
F8E4M3_MAX = 448.0
F4E2M1_MAX = 6.0
GUARD_COUNT = 8
SENTINEL = -3.25
BYTE_SENTINEL = 0xA5
POISON = 1.5

# Expressions OpenGL may evaluate in 32 bits: the flattened thread index and
# the group and packed-output indices quantize derives from it, and
# dequantize's group index, packed-input offset, rebased output offset and its
# subscripts. The model shows each names an element inside its buffer, and
# every case's buffers hold far fewer than INDEX_BOUND elements.
INDEX_BOUND = 2**20 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "quantize_dequantize": ("index",),
    "quantize": ("index", "gindex", "index / pack_factor"),
    "dequantize": ("gindex", "oindex", "offset", "out_ + i"),
}
INDEX_BUFFERS = {
    "index": "w",
    "gindex": "scales",
    "index / pack_factor": "out",
    "oindex": "out",
    "offset": "w",
    "out_ + i": "out",
}
# The portable host's arithmetic profiles for these kernels
# (portable_host/quantization_packages.py) and a subnormal-flushing comparison.
# For an all-zero mx group fp8_e8m0's float conversion returns the subnormal
# 2^-127. The Metal source flushes it: log2 of it is -inf, so the scale byte is
# 0 (#2143), and ``scale == 0`` holds, so codes stay signed zeros instead of
# the NaN code a subnormal-preserving comparison and flushing division give.
SOURCE_PROFILES = {
    "binary32_multiplication_profile": "rne-flush",
    "binary32_comparison_profile": "flush-subnormals",
    "binary32_log2_operand_profile": "flush-subnormals",
    "binary32_log2_accuracy_profile": "portable-finite",
}


def _instantiations():
    """``(entry, info)`` in fp_quantized.metal's instantiation order."""
    for itype, dtype in DTYPES:
        for mode, group_size, bits, global_scale in MODES:
            for kind in KINDS:
                hgs = "true" if global_scale else "false"
                yield f"{mode}_{kind}_{itype}_gs_{group_size}_b_{bits}_hgs_{hgs}", {
                    "kind": kind,
                    "mode": mode,
                    "itype": itype,
                    "dtype": dtype,
                    "groupSize": group_size,
                    "bits": bits,
                    "globalScale": global_scale,
                }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _config(directory, target, entry):
    """The translation of one entry, as the gate and contract run it."""
    kind = INFO[entry]["kind"]
    # Correctly rounded binary32 division keeps every scale and code exact.
    options = {
        "directx": {"binary32_division_profile": "rne-flush"},
        "opengl": {"binary32_division_profile": "rne-flush"},
    }
    if kind != "dequantize":
        # simd_max, and quantize's simd_shuffle_down; dequantize has no
        # subgroup operation.
        options["directx"].update(
            software_subgroup_width=SIMD_SIZE,
            relative_wave_shuffle_out_of_range="self",
        )
        options["opengl"]["software_subgroup_width"] = SIMD_SIZE
    assertions = OPENGL_INDEX_RANGE_ASSERTIONS[kind] if target == "opengl" else ()
    return ProjectConfig(
        root=Path(directory).parent,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=Path(directory).name + "/out",
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: WORKGROUP}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=INDEX_BOUND)
            for expression in assertions
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                **SOURCE_PROFILES,
                "target_options": options,
            }
        },
    )


# ---------------------------------------------------------------------------
# Scale and code formats, bit for bit as fp4.h and fp8.h compute them
# ---------------------------------------------------------------------------


def _bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _half(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def _flushed(value):
    """Metal's float32 arithmetic flushes subnormal operands and results."""
    return 0.0 * value if 0 < abs(value) < 2.0**-126 else value


def _div(left, right):
    """Correctly rounded binary32 division with subnormals flushed."""
    return _flushed(_f32(_flushed(left) / _flushed(right)))


def _mul(left, right):
    return _flushed(_f32(left * right))


def _e4m3_encode(value):
    """``fp8_e4m3(float)``: PyTorch's saturating round to nearest even."""
    f_bits = _bits32(value)
    sign = f_bits & 0x80000000
    f_bits ^= sign
    if f_bits >= 543 << 21:
        bits = 0x7E
    elif f_bits < 121 << 23:
        # The subnormal path rounds by adding 2^14, whose ulp is 2^-9.
        f_bits = _bits32(_f32(_float32(f_bits) + _float32(141 << 23)))
        bits = (f_bits - (141 << 23)) & 0xFF
    else:
        mant_odd = (f_bits >> 20) & 1
        f_bits = (f_bits + (((7 - 127) & 0xFFFFFFFF) << 23) + 0x7FFFF) & 0xFFFFFFFF
        bits = ((f_bits + mant_odd) >> 20) & 0xFF
    return bits | (sign >> 24)


def _e4m3_decode(bits):
    """``float(fp8_e4m3)``, through the binary16 bit pattern times 256."""
    v = bits & 127
    u = (v << 7) | (((v + 1) >> 7) << 14) | (((bits >> 7) & 1) << 15)
    return _half(u) * 256.0


def _e8m0_encode(value):
    """``fp8_e8m0(float)``: the rounded base-2 logarithm, saturated."""
    if not math.isfinite(value):
        return 0xFF
    if value < 0.0:
        return 0x00
    value = _flushed(value)
    # log2(0) is -inf, and the float-to-int conversion saturates.
    n = -(2**31) if value == 0 else int(_metal_round(math.log2(value)))
    return min(max(n, -127), 127) + 127


def _e8m0_decode(bits):
    return _float32(0x400000 if bits == 0 else bits << 23)


def _mx_scale_round_up(value):
    """``mx_scale_round_up``: the smallest E8M0 value not below ``value``."""
    bits = _e8m0_encode(value)
    if bits < 0xFE and _flushed(_e8m0_decode(bits)) < value:
        bits += 1
    return _e8m0_decode(bits)


def _e2m1_encode(value):
    """``fp4_e2m1(float)``: E2M1 with ties to the even code."""
    if math.isnan(value):
        return 0x7
    sign = 0x8 if math.copysign(1.0, value) < 0 else 0x0
    x = abs(value)
    for bound, inclusive, code in (
        (5.0, False, 0x7),
        (3.5, True, 0x6),
        (2.5, False, 0x5),
        (1.75, True, 0x4),
        (1.25, False, 0x3),
        (0.75, True, 0x2),
        (0.25, False, 0x1),
    ):
        if x > bound or (inclusive and x == bound):
            return code | sign
    return sign


def _e2m1_decode(bits):
    """``float(fp4_e2m1)``: the binary16 pattern ``(bits & 7) << 9`` times 2^14."""
    value = _half((bits & 7) << 9) * 16384.0
    return -value if bits & 8 else value


E2M1_VALUES = tuple(_e2m1_decode(code) for code in range(8))
E4M3_VALUES = tuple(_e4m3_decode(code) for code in range(127))
E2M1_TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _code(bits, value):
    return _e4m3_encode(value) if bits == 8 else _e2m1_encode(value)


def _decode(bits, code):
    return _e4m3_decode(code) if bits == 8 else _e2m1_decode(code)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _threads(entry, case):
    x, y, _ = GRIDS[INFO[entry]["kind"]][case]
    return x * y * WORKGROUP[0]


def _family(info):
    return "nvfp4" if info["mode"] == "nvfp4" else "mx"


def _signs(draws, values):
    return [value if _below(draws, 2) else -value for value in values]


def _place_max(draws, values, peak):
    """Put the group's largest magnitude ``peak`` at a random lane."""
    values[_below(draws, len(values))] = peak if _below(draws, 2) else -peak
    return values


def _group_values(info, case, group, scenario):
    """The ``group_size`` input values of one quantization group."""
    draws = _draws("fp-quantize", info["mode"], info["dtype"], case, group)
    size, bits = info["groupSize"], info["bits"]
    top = F8E4M3_MAX if bits == 8 else F4E2M1_MAX
    grid = E4M3_VALUES[1:] if bits == 8 else E2M1_VALUES
    if scenario in ("grid", "ties", "roundup"):
        if info["mode"] == "nvfp4":
            # An E4M3 scale 2^e (1 + m / 8); ties need a power of two.
            mantissa = 0 if scenario == "ties" else _below(draws, 8)
            unit = 2.0 ** (_below(draws, 7) - 3) * (1 + mantissa / 8)
        else:
            unit = 2.0 ** (_below(draws, 5) - 2)
        if scenario == "grid":
            values = [unit * grid[_below(draws, len(grid))] for _ in range(size)]
            return _place_max(draws, _signs(draws, values), top * unit)
        if scenario == "ties":
            if bits == 8:
                # Midpoints of adjacent E4M3 values.
                ties = [
                    (E4M3_VALUES[k] + E4M3_VALUES[k + 1]) / 2
                    for k in (1 + _below(draws, 125) for _ in range(size))
                ]
            else:
                ties = [E2M1_TIES[index % len(E2M1_TIES)] for index in range(size)]
            values = [unit * tie for tie in ties]
            return _place_max(draws, _signs(draws, values), top * unit)
        # amax / top is 17/16 of a power of two, so the E8M0 scale rounds up.
        step = top * unit / 16
        values = [step * _below(draws, 18) for _ in range(size)]
        return _place_max(draws, _signs(draws, values), step * 17)
    if scenario == "zero":
        return _signs(draws, [0.0] * size)
    if scenario == "saturate":
        # amax / 6 >= 464: the E4M3 scale saturates to 448.
        values = [16.0 * _below(draws, 256) for _ in range(size)]
        return _place_max(draws, _signs(draws, values), 16.0 * 255)
    if scenario == "tiny":
        # amax / 6 < 2^-6: the E4M3 scale takes its subnormal path.
        values = [2.0**-14 * _below(draws, 256) for _ in range(size)]
        return _place_max(draws, _signs(draws, values), 2.0**-14 * 255)
    values = [2.0**-5 * (1 + _below(draws, 255)) for _ in range(size)]
    return values if scenario == "free" else [-value for value in values]


def _global_scale(info, case):
    if not info["globalScale"]:
        return POISON
    return GLOBAL_SCALES["dequantize" if info["kind"] == "dequantize" else "quantize"][
        case
    ]


def _dequantize_inputs(entry, case):
    """Packed codes and scale bytes; every code and scale class appears."""
    info = INFO[entry]
    draws = _draws("fp-dequantize", entry, case)
    bits, size = info["bits"], info["groupSize"]
    pack_factor = 1 if bits == 8 else 2
    count = _threads(entry, case)
    if bits == 8:
        # NaN codes 0x7F and 0xFF aside: zeros, subnormals, both signs.
        valid = [code for code in range(256) if code & 127 != 127]
        payload = list(range(0, 8)) + [0x80, 0x87, 0x7E, 0xFE]
        payload += [valid[_below(draws, len(valid))] for _ in range(count - 12)]
    else:
        payload = [0x10 * high + low for low, high in zip(range(16), range(15, -1, -1))]
        payload += [_below(draws, 256) for _ in range(count - 16)]
    groups = count * pack_factor // size
    if info["mode"] == "nvfp4":
        # E4M3 scales: zero, signed zero, subnormal, normal, either sign.
        first = [0x00, 0x80, 0x01, 0x07, 0x08, 0x7E, 0xC5, 0xBA]
        valid = [code for code in range(256) if code & 127 != 127]
        scales = first + [valid[_below(draws, len(valid))] for _ in range(groups - 8)]
    else:
        # E8M0 scales 2^-5 .. 2^7 keep every output normal and finite in float16.
        scales = [122, 134] + [122 + _below(draws, 13) for _ in range(groups - 2)]
    return payload, scales[:groups]


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    info = INFO[entry]
    dtype, kind, bits = info["dtype"], info["kind"], info["bits"]
    size = info["groupSize"]
    pack_factor = 1 if bits == 8 else 2
    threads = _threads(entry, case)
    scale = [_global_scale(info, case)] + [POISON] * GUARD_COUNT
    if kind == "dequantize":
        payload, scales = _dequantize_inputs(entry, case)
        kinds = {"w": "uint8", "scales": "uint8", "global_scale": "float32"}
        kinds["out"] = dtype
        initial = {"out": [SENTINEL] * (threads * pack_factor + GUARD_COUNT)}
        storage = {
            "w": payload + [BYTE_SENTINEL] * GUARD_COUNT,
            "scales": scales + [BYTE_SENTINEL] * GUARD_COUNT,
        }
    else:
        scenarios = SCENARIOS[_family(info)][case]
        assert len(scenarios) * size == threads, (entry, case)
        w = []
        for group, scenario in enumerate(scenarios):
            w += _group_values(info, case, group, scenario)
        assert all(_to_type(dtype, value) == value for value in w), entry
        kinds = {"w": dtype, "global_scale": "float32", "out": dtype}
        initial = {"out": [SENTINEL] * (threads + GUARD_COUNT)}
        if kind == "quantize":
            kinds.update(out="uint8", scales="uint8")
            initial = {
                "out": [BYTE_SENTINEL] * (threads // pack_factor + GUARD_COUNT),
                "scales": [BYTE_SENTINEL] * (threads // size + GUARD_COUNT),
            }
        storage = {"w": w + [POISON] * GUARD_COUNT}
    storage.update(global_scale=scale, **initial)
    data = {
        "kinds": kinds,
        "values": {name: (kinds[name], values) for name, values in storage.items()},
        "initial": initial,
        "grid": GRIDS[kind][case],
        "threads": threads,
    }
    data["expected"] = _simulate(entry, data)["outputs"]
    return data


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


MUTATIONS = {
    "quantize": {
        "shuffle": "drops the neighbouring lane's code from each packed byte",
        "halves": "takes one maximum across all 32 lanes",
        "roundup": "rounds the E8M0 scale to nearest instead of up",
        "ties": "rounds E2M1 ties away from zero",
        "sign": "drops the codes' sign bits",
    },
    "quantize_dequantize": {
        "halves": "takes one maximum across all 32 lanes",
        "roundup": "rounds the E8M0 scale to nearest instead of up",
        "ties": "rounds E2M1 ties away from zero",
        "sign": "drops the codes' sign bits",
    },
    "dequantize": {
        "order": "reverses the codes of each byte",
        "group": "indexes scales by packed byte instead of output element",
        "global": "ignores the global scale",
    },
}


def _applies(entry, mutation):
    """Whether ``mutation`` can change ``entry``'s outputs at all."""
    info = INFO[entry]
    return {
        "shuffle": info["bits"] == 4,
        "halves": info["mode"] == "nvfp4",
        "roundup": info["mode"] != "nvfp4",
        "ties": info["bits"] == 4,
        "order": info["bits"] == 4,
        "group": info["bits"] == 4,
        "global": info["globalScale"],
    }.get(mutation, True)


def _quantize_scale(info, amax, global_scale, mutation):
    """quantize's scale code, decoded scale and encoding factor for one lane."""
    scale_enc = 1.0
    if info["globalScale"]:
        scale_enc = _div(F8E4M3_MAX * F4E2M1_MAX, global_scale)
    scale_dec_b = _div(amax, F4E2M1_MAX if info["bits"] == 4 else F8E4M3_MAX)
    path = None
    if info["globalScale"]:
        scale_dec_b = _mul(scale_dec_b, scale_enc)
    elif info["mode"] != "nvfp4":
        rounded = _mx_scale_round_up(scale_dec_b)
        path = "up" if rounded != scale_dec_b else "exact"
        if mutation == "roundup":
            rounded = _e8m0_decode(_e8m0_encode(scale_dec_b))
        scale_dec_b = rounded
    if info["mode"] == "nvfp4":
        scale_bits = _e4m3_encode(scale_dec_b)
        magnitude = _bits32(abs(scale_dec_b))
        path = (
            "saturated"
            if magnitude >= 543 << 21
            else "subnormal" if magnitude < 121 << 23 else "normal"
        )
        scale = _e4m3_decode(scale_bits)
    else:
        scale_bits = _e8m0_encode(scale_dec_b)
        scale = _e8m0_decode(scale_bits)
    return scale_bits, scale, scale_enc, path


def _quantize_code(info, value, mutation):
    code = _code(info["bits"], value)
    if mutation == "ties" and info["bits"] == 4 and abs(value) in E2M1_TIES:
        code = (code & 8) | min((code & 7) + (abs(value) not in (0.75, 1.75, 3.5)), 7)
    if mutation == "sign":
        code &= 0x7F if info["bits"] == 8 else 0x7
    return code


def _simulate_quantize(entry, data, mutation=None):
    """Replay fp_quantize or fp_quantize_dequantize lane by lane."""
    info = INFO[entry]
    kind, size, bits = info["kind"], info["groupSize"], info["bits"]
    pack_factor = 1 if bits == 8 else 2
    w = data["values"]["w"][1]
    global_scale = data["values"]["global_scale"][1][0]
    outputs = {name: list(values) for name, values in data["initial"].items()}
    stores, indices, groups = Counter(), {}, []
    record = indices.setdefault
    for base in range(0, data["threads"], SIMD_SIZE):
        lanes = [_f32(w[base + lane]) for lane in range(SIMD_SIZE)]
        codes = []
        for lane, value in enumerate(lanes):
            index = base + lane
            if info["mode"] == "nvfp4" and mutation != "halves":
                half = lanes[:16] if lane < 16 else lanes[16:]
            else:
                half = lanes
            amax = max(abs(item) for item in half)
            scale_bits, scale, scale_enc, path = _quantize_scale(
                info, amax, global_scale, mutation
            )
            scale_enc_b = 0.0 if _flushed(scale) == 0 else _div(scale_enc, scale)
            scaled = _mul(value, scale_enc_b)
            code = _quantize_code(info, scaled, mutation)
            codes.append(code)
            record("index", []).append(index)
            if index % size == 0:
                groups.append(
                    {
                        "scale": scale,
                        "scaleBits": scale_bits,
                        "scaleEnc": scale_enc,
                        "scaleEncB": scale_enc_b,
                        "path": path,
                        "amax": amax,
                    }
                )
            if kind == "quantize_dequantize":
                value = _mul(_div(scale, scale_enc), _decode(bits, code))
                outputs["out"][index] = _to_type(info["dtype"], value)
                stores["out", index] += 1
                continue
            record("gindex", []).append(index // size)
            if index % size == 0:
                outputs["scales"][index // size] = scale_bits
                stores["scales", index // size] += 1
        if kind != "quantize":
            continue
        for lane in range(0, SIMD_SIZE, pack_factor):
            output = codes[lane]
            if bits == 4 and mutation != "shuffle":
                # simd_shuffle_down(output, 1) from the next lane.
                output = (output | (codes[lane + 1] << bits)) & 0xFF
            slot = (base + lane) // pack_factor
            record("index / pack_factor", []).append(slot)
            outputs["out"][slot] = output
            stores["out", slot] += 1
    return {"outputs": outputs, "stores": stores, "indices": indices, "groups": groups}


def _simulate_dequantize(entry, data, mutation=None):
    """Replay fp_dequantize thread by thread."""
    info = INFO[entry]
    size, bits = info["groupSize"], info["bits"]
    pack_factor = 1 if bits == 8 else 2
    w = data["values"]["w"][1]
    scales = data["values"]["scales"][1]
    global_scale = data["values"]["global_scale"][1][0]
    outputs = {name: list(values) for name, values in data["initial"].items()}
    stores, indices = Counter(), {}
    inv_scale_enc = 1.0
    if info["globalScale"] and mutation != "global":
        inv_scale_enc = _div(global_scale, F8E4M3_MAX * F4E2M1_MAX)
    for offset in range(data["threads"]):
        oindex = offset * pack_factor
        gindex = (offset if mutation == "group" else oindex) // size
        for name, value in (("offset", offset), ("oindex", oindex), ("gindex", gindex)):
            indices.setdefault(name, []).append(value)
        if info["mode"] == "nvfp4":
            scale = _e4m3_decode(scales[gindex])
        else:
            scale = _e8m0_decode(scales[gindex])
        if info["globalScale"]:
            scale = _mul(scale, inv_scale_enc)
        val = w[offset]
        for i in range(pack_factor):
            shift = bits * (pack_factor - 1 - i if mutation == "order" else i)
            code = (val >> shift) & 0x0F if bits == 4 else val
            value = _mul(scale, _decode(bits, code))
            outputs["out"][oindex + i] = _to_type(info["dtype"], value)
            stores["out", oindex + i] += 1
            indices.setdefault("out_ + i", []).append(oindex + i)
    return {"outputs": outputs, "stores": stores, "indices": indices, "groups": []}


def _simulate(entry, data, mutation=None):
    if INFO[entry]["kind"] == "dequantize":
        return _simulate_dequantize(entry, data, mutation)
    return _simulate_quantize(entry, data, mutation)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _checked_element_count():
    """Output elements every target compares, across all entries and cases."""
    return sum(
        len(values)
        for entry in ENTRIES
        for case in CASES
        for values in _case_data(entry, case)["expected"].values()
    )


def _expected_validation_contract():
    return {
        "model": (
            "fp_quantize and fp_quantize_dequantize lane by lane: simdgroup "
            "maxima (per 16-lane half for nvfp4), correctly rounded binary32 "
            "scale arithmetic, the E4M3, E8M0 and E2M1 encoders bit for bit, "
            "the shuffled packing and guarded byte writes; fp_dequantize "
            "thread by thread: unpacking, scale decoding and scale * code"
        ),
        "comparison": "exact words",
        "workgroupSize": list(WORKGROUP),
        "grids": {
            kind: {case: list(GRIDS[kind][case]) for case in CASES} for kind in KINDS
        },
        "quantizeGroups": {
            family: {case: list(SCENARIOS[family][case]) for case in CASES}
            for family in SCENARIOS
        },
        "globalScales": GLOBAL_SCALES,
        "inputs": {
            "quantize": (
                "element-grid multiples of an E4M3 or power-of-two scale, every "
                "E2M1 threshold or E4M3 midpoint, mx groups whose scale rounds "
                "up, multiples of 1/32 in [1/32, 8] (all negative in one "
                "group), signed zeros, and for nvfp4 saturated and subnormal "
                "E4M3 scales"
            ),
            "dequantize": (
                "every E2M1 code, E4M3 codes other than NaN with zeros and "
                "subnormals, E4M3 scales of either sign with zeros and "
                "subnormals, E8M0 scales 2^-5 to 2^7"
            ),
        },
        "guards": {
            "count": GUARD_COUNT,
            "sentinel": SENTINEL,
            "byteSentinel": BYTE_SENTINEL,
        },
        "softwareSubgroups": {
            "kernels": ["fp_quantize_dequantize", "fp_quantize"],
            "width": SIMD_SIZE,
            "directxShuffleOutOfRange": "self",
        },
        "division": {
            "targets": ["directx", "opengl"],
            "binary32_division_profile": "rne-flush",
        },
        "sourceProfiles": dict(SOURCE_PROFILES),
        "openglIndexRangeAssertions": {
            kind: {
                "expressions": list(OPENGL_INDEX_RANGE_ASSERTIONS[kind]),
                "minimum": 0,
                "maximum": INDEX_BOUND,
            }
            for kind in KINDS
        },
        "metalUpstreamComparison": "unchanged fp_quantized.metal, word for word",
    }


def _expected_scope():
    return {
        "instantiatedEntryCount": UNIT_ENTRY_COUNT,
        "coveredEntryCount": len(ENTRIES),
        "excludedEntries": {
            "kernels": list(EXCLUDED_KERNELS),
            "count": UNIT_ENTRY_COUNT - len(ENTRIES),
            "reason": (
                "the matrix-vector and matrix-matrix kernels are outside this gate"
            ),
        },
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "checkedElementCountByTarget": _checked_element_count(),
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def expected_contract(artifacts=None):
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-fp-quantized-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "headers": dict(HEADERS),
            "hostSource": {"path": HOST_SOURCE, "sha256": HOST_SHA256},
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        "artifacts": artifacts,
        "validationContract": _expected_validation_contract(),
        "scope": _expected_scope(),
    }


def test_current_mlx_fp_quantized_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    artifacts = contract["artifacts"]
    assert set(artifacts) == set(TARGETS)
    for target in TARGETS:
        assert list(artifacts[target]) == list(ENTRIES)
        for record in artifacts[target].values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            assert record["sizeBytes"] > 0


def test_fp_quantized_entries_follow_mlx_instantiation():
    assert len(ENTRIES) == 36 == len(set(ENTRIES))
    assert Counter(info["kind"] for info in INFO.values()) == dict.fromkeys(KINDS, 12)
    assert ENTRIES[:3] == (
        "nvfp4_quantize_dequantize_float_gs_16_b_4_hgs_false",
        "nvfp4_quantize_float_gs_16_b_4_hgs_false",
        "nvfp4_dequantize_float_gs_16_b_4_hgs_false",
    )
    assert ENTRIES[-1] == "mxfp4_dequantize_float16_t_gs_32_b_4_hgs_false"
    assert not {f"fp_{kind}" for kind in KINDS} & set(EXCLUDED_KERNELS)


# ---------------------------------------------------------------------------
# Model checks
# ---------------------------------------------------------------------------


def test_fp_formats_match_their_definitions():
    # E4M3: every non-NaN code decodes to its value and re-encodes to itself.
    for code in range(256):
        if code & 127 == 127:
            continue
        exponent, mantissa = (code >> 3) & 15, code & 7
        magnitude = (
            mantissa * 2.0**-9
            if exponent == 0
            else 2.0 ** (exponent - 7) * (1 + mantissa / 8)
        )
        value = _e4m3_decode(code)
        assert value == (-magnitude if code & 0x80 else magnitude), code
        assert math.copysign(1, value) == (-1 if code & 0x80 else 1)
        assert _e4m3_encode(value) == code, code
    assert _e4m3_encode(464.0) == 0x7E and _e4m3_encode(463.9) == 0x7E
    assert _e4m3_encode(3 * 2.0**-10) == 0x02  # a subnormal tie rounds to even
    # E2M1: codes decode to the format's eight magnitudes with signed zero.
    assert E2M1_VALUES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert math.copysign(1, _e2m1_decode(8)) == -1
    for code in range(16):
        assert _e2m1_encode(_e2m1_decode(code)) == code
    assert [_e2m1_encode(tie) for tie in E2M1_TIES] == [0, 2, 2, 4, 4, 6, 6]
    # E8M0 rounds up to a power of two however log2 rounds near half steps.
    for exponent in range(-20, 20):
        for fraction in (1.0, 1.0001, 1.414, 1.4143, 1.5, 1.999):
            value = _f32(2.0**exponent * fraction)
            expected = 2.0 ** (exponent + (fraction > 1))
            assert _mx_scale_round_up(value) == expected, value


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_model_matches_mlx_layout(entry):
    info = INFO[entry]
    size, bits = info["groupSize"], info["bits"]
    pack_factor = 1 if bits == 8 else 2
    for case in CASES:
        data = _case_data(entry, case)
        run = _simulate(entry, data)
        outputs = run["outputs"]
        # Every output element is written exactly once and the guards stay.
        for name, values in outputs.items():
            written = len(values) - GUARD_COUNT
            assert all(run["stores"][name, index] == 1 for index in range(written))
            assert values[written:] == data["initial"][name][written:]
        assert len(run["stores"]) == sum(
            len(values) - GUARD_COUNT for values in outputs.values()
        )
        if info["kind"] == "dequantize":
            # Element e takes code e % pack_factor of byte e / pack_factor,
            # least significant first, and the scale of group e / group_size.
            payload = data["values"]["w"][1]
            scales = data["values"]["scales"][1]
            decode = _e4m3_decode if info["mode"] == "nvfp4" else _e8m0_decode
            factor = 1.0
            if info["globalScale"]:
                factor = data["values"]["global_scale"][1][0] / 2688
            for element in range(data["threads"] * pack_factor):
                byte = payload[element // pack_factor]
                code = byte >> (bits * (element % pack_factor)) & (2**bits - 1)
                value = decode(scales[element // size]) * factor * _decode(bits, code)
                expected = outputs["out"][element]
                assert abs(value - expected) <= abs(value) * 2.0**-7, element
            continue
        w = data["values"]["w"][1]
        codes = [
            _code(bits, _mul(w[index], run["groups"][index // size]["scaleEncB"]))
            for index in range(data["threads"])
        ]
        if info["kind"] == "quantize":
            # Byte j holds element 2j's code low and element 2j + 1's high.
            stream = outputs["out"][:-GUARD_COUNT]
            if bits == 4:
                stream = [
                    nibble for byte in stream for nibble in (byte & 15, byte >> 4)
                ]
            assert stream == codes
            assert outputs["scales"][:-GUARD_COUNT] == [
                group["scaleBits"] for group in run["groups"]
            ]
        else:
            for index, code in enumerate(codes):
                group = run["groups"][index // size]
                value = group["scale"] / group["scaleEnc"] * _decode(bits, code)
                expected = outputs["out"][index]
                assert abs(value - expected) <= abs(value) * 2.0**-7, index


def test_fp_quantize_groups_reach_every_branch():
    paths, codes, ties = Counter(), Counter(), Counter()
    for entry in ENTRIES:
        info = INFO[entry]
        if info["kind"] == "dequantize":
            continue
        for case in CASES:
            data = _case_data(entry, case)
            run = _simulate(entry, data)
            scenarios = SCENARIOS[_family(info)][case]
            assert len(run["groups"]) == len(scenarios)
            for scenario, group in zip(scenarios, run["groups"]):
                paths[info["mode"], info["globalScale"], group["path"]] += 1
                if scenario == "zero":
                    assert group["amax"] == 0 and group["scaleBits"] == 0, entry
                if scenario == "saturate":
                    assert group["scaleBits"] == 0x7E, entry
                if scenario == "roundup":
                    assert group["path"] == "up", entry
                if scenario in ("grid", "ties") and info["mode"] != "nvfp4":
                    assert group["path"] == "exact", entry
            if info["kind"] == "quantize" and info["bits"] == 4:
                for byte in run["outputs"]["out"][:-GUARD_COUNT]:
                    codes[info["mode"], byte & 15] += 1
                    codes[info["mode"], byte >> 4] += 1
            w = data["values"]["w"][1][: data["threads"]]
            ties[info["mode"]] += sum(abs(value) in E2M1_TIES for value in w)
    for global_scale in (False, True):
        for path in ("normal", "saturated", "subnormal"):
            assert paths["nvfp4", global_scale, path], (global_scale, path)
    for mode in ("mxfp8", "mxfp4"):
        assert paths[mode, False, "up"] and paths[mode, False, "exact"], mode
    for mode in ("nvfp4", "mxfp4"):
        assert {code for name, code in codes if name == mode} == set(range(16)), mode
        assert ties[mode], mode


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_fp_quantized_inputs_are_exact_in_their_element_type(dtype):
    for entry in ENTRIES:
        if INFO[entry]["dtype"] != dtype:
            continue
        for case in CASES:
            data = _case_data(entry, case)
            for name, (kind, values) in data["values"].items():
                if kind == dtype:
                    assert all(_to_type(dtype, value) == value for value in values)
                    assert all(
                        value == 0 or abs(value) >= 2.0**-14 for value in values
                    ), (entry, name)
            for values in data["expected"].values():
                assert all(math.isfinite(value) for value in values), entry


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in ENTRIES
        for mutation in MUTATIONS[INFO[entry]["kind"]]
        if _applies(entry, mutation)
    ],
)
def test_fp_quantized_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[INFO[entry]["kind"]][mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_fp_quantized_opengl_assertions_name_buffer_elements(entry):
    kind = INFO[entry]["kind"]
    used = set()
    for case in CASES:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        for expression, values in _simulate(entry, data)["indices"].items():
            used.add(expression)
            name = INDEX_BUFFERS[expression]
            assert 0 <= min(values) and max(values) < lengths[name], expression
            assert max(values) <= INDEX_BOUND
    assert used == set(OPENGL_INDEX_RANGE_ASSERTIONS[kind])


@pytest.mark.parametrize("target", TARGETS)
def test_fp_quantized_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES:
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        source = config.source_options["metal"]
        assert {key: source[key] for key in SOURCE_PROFILES} == SOURCE_PROFILES
        options = source["target_options"]
        division = {"binary32_division_profile": "rne-flush"}
        if INFO[entry]["kind"] == "dequantize":
            assert options == {"directx": division, "opengl": division}
        else:
            assert options == {
                "directx": {
                    **division,
                    "software_subgroup_width": SIMD_SIZE,
                    "relative_wave_shuffle_out_of_range": "self",
                },
                "opengl": {**division, "software_subgroup_width": SIMD_SIZE},
            }
        expressions = [item.expression for item in config.index_range_assertions]
        if target == "opengl":
            assert expressions == list(
                OPENGL_INDEX_RANGE_ASSERTIONS[INFO[entry]["kind"]]
            )
        else:
            assert expressions == []


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native fp quantized gate")
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
            "mlx/backend/metal",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    source = (root / SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    for header, digest in {**HEADERS, HOST_SOURCE: HOST_SHA256}.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    # ENTRIES follow fp_quantized.metal's own instantiation macros, and the
    # parameter tuples fp_quantized.h's [[buffer(i)]] signatures.
    text = source.decode("utf-8")
    kernels = _macro(text, "instantiate_quantize_dequantize")
    assert re.findall(r"(fp_\w+),\s*\\", kernels) == [f"fp_{kind}" for kind in KINDS]
    types = re.findall(r"^instantiate_quantized_types\((\w+)\)", text, re.MULTILINE)
    assert types == [itype for itype, _ in DTYPES]
    modes = re.findall(
        r"instantiate_quantize_dequantize\(type, (\w+), (\d+), (\d+), (\w+)\)",
        _macro(text, "instantiate_quantized_types"),
    )
    assert modes == [
        (mode, str(size), str(bits), "true" if scale else "false")
        for mode, size, bits, scale in MODES
    ]
    header = (root / "mlx/backend/metal/kernels/fp_quantized.h").read_text(
        encoding="utf-8"
    )
    for kind in KINDS:
        assert _kernel_parameters(header, f"fp_{kind}") == PARAMETERS[kind]
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fp-quantized-metal-reference")
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


def _translate(root, work, target, entry, record):
    """Translate one entry and check it against its pinned artifact."""
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
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    (execution,) = artifact["execution"]["entryPoints"]
    assert execution["workgroupSize"] == list(WORKGROUP)
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    return _prepare_native_package(translation, work)


def _request(descriptor, package, target, entry, data):
    """Bind every reflected binding by source parameter and build one dispatch."""
    from crosstl.project.runtime_value_encoding import FLOAT16_BITS

    info = INFO[entry]
    kind = info["kind"]
    values = {
        name: _runtime_value(target, dtype, storage)
        for name, (dtype, storage) in data["values"].items()
    }
    inputs, observed, outputs, generated = {}, [], {}, []
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            # HLSL has no threads_per_grid builtin: the flattened index's row
            # stride comes from a constant buffer the loader fills from the
            # dispatch.
            assert provenance["executionInput"]["valueSource"] == (
                "dispatch.workgroupCount"
            ), binding
            generated.append(binding["name"])
            continue
        source = provenance.get("sourceResource")
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
        inputs[binding["name"]] = value
        if name in OUTPUTS[kind]:
            assert binding["access"] == "read_write", binding
            outputs[name] = binding["name"]
        else:
            observed.append((name, binding["name"]))
    assert len(generated) == (1 if target == "directx" else 0), generated
    assert set(outputs) == set(OUTPUTS[kind]), outputs
    # Only the global-scale entries read global_scale; the others may drop it.
    names = {name for name, _ in observed}
    reads = {name for name, _ in PARAMETERS[kind]} - set(OUTPUTS[kind])
    assert names == reads or (
        not info["globalScale"] and names == reads - {"global_scale"}
    ), names
    expected = {
        outputs[name]: _runtime_value(
            target, data["kinds"][name], data["expected"][name]
        )
        for name in outputs
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": list(data["grid"]), "workgroupSize": list(WORKGROUP)},
        expected_target=target,
        observe_inputs=[binding for _, binding in observed],
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    derived = [
        item.value.values
        for item in request.execution_plan.resource_bindings
        if item.value is not None and "executionInput" in item.value.metadata
    ]
    assert derived == [list(data["grid"])] * len(generated)
    return request, outputs, len(observed)


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    kind = INFO[entry]["kind"]
    files, payloads = [], {}
    for name, index in sorted(PARAMETERS[kind], key=lambda item: item[1]):
        payload = _metal_bytes(*data["values"][name])
        payloads[name] = payload
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": list(data["grid"]),
                "workgroupSize": list(WORKGROUP),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    words = {}
    for name, index in PARAMETERS[kind]:
        raw = (readback / f"buffer-{index}.bin").read_bytes()
        if name in OUTPUTS[kind]:
            words[name] = _words(data["kinds"][name], raw)
        else:
            assert raw == payloads[name], name
    return device, words


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_fp_quantized_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-fpq-", dir=root) as path:
        work = Path(path)
        try:
            descriptor, package = _translate(root, work, target, entry, record)
            executor = _executor(target)
            evidence = []
            try:
                for case in CASES:
                    data = _case_data(entry, case)
                    request, outputs, observed = _request(
                        descriptor, package, target, entry, data
                    )
                    results = _execute(executor, work, case, request, outputs, observed)
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream(
                            work, metal_reference, entry, case, data
                        )
                        upstream = {"device": device, "identicalToTranslation": True}
                    for name in outputs:
                        words = results[name]["values"]
                        if target == "metal":
                            # The translation reproduces the unchanged upstream
                            # kernel word for word on the same device.
                            _assert_identical(
                                f"{case}-{name}-upstream", words, upstream_words[name]
                            )
                        expected = _runtime_value(
                            target, data["kinds"][name], data["expected"][name]
                        )["values"]
                        _assert_identical(f"{case}-{name}", words, expected)
                    evidence.append(
                        {
                            "case": case,
                            "grid": list(data["grid"]),
                            "threads": data["threads"],
                            "checkedElementCount": sum(
                                len(values) for values in data["expected"].values()
                            ),
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


def test_ci_requires_native_fp_quantized_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow,
        "Prove current MLX fp quantize and dequantize native parity",
        group="quantized",
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/fp-quantized",
        "--junitxml=mlx-current-results/fp-quantized.xml",
        "demos/integrations/mlx/tests/kernels/test_current_fp_quantized.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_fp_quantized.py"
    )
