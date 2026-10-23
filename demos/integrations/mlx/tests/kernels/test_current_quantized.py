"""Natively prove quantized.metal's 108 affine quantize and dequantize kernels.

``quantized.metal`` instantiates ``affine_quantize`` and ``affine_dequantize``
for float, float16_t and bfloat16_t, group sizes 128, 64 and 32 and 2, 3, 4,
5, 6 and 8 bits: 108 of its 2,052 entries. Each is translated and pinned per
target, then runs two cases on 64-thread workgroups: a row of three
(quantize) or two (dequantize) workgroups and a two-by-two grid, whose
``threads_per_grid`` row stride the kernels' flattened offset uses.
``quantize_impl`` dispatches ``min(maxTotalThreadsPerThreadgroup, nthreads)``
threads per group; any multiple of the 32-lane simdgroup keeps each
quantization group inside one simdgroup, as the kernel's ``simd_min`` and
``simd_max`` require.

Quantize groups hold values ``(k + d - a) * 2^e`` for codes ``k``, quarter
offsets ``d`` and an integer ``a`` that puts the larger magnitude at either
end, so every scale, edge and code is exact. Further groups are all zero (the
``eps`` scale), all negative (``w_max`` starts at zero), a positive constant
(``eps`` with a rounded division) and free dyadic values, whose scales and
codes need correctly rounded float32 division. A model replays each lane's
reduction, rounding, shuffles and byte writes and must reproduce MLX's
bit-stream packing. Dequantize unpacks random codes with power-of-two scales
of either sign and biases ``-a * scale``, so every output is exact in the
element type. Each target must produce the model's words and leave every
guard untouched; on Metal the translation must also match the unchanged
upstream kernel word for word.
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
from demos.integrations.mlx.tests.kernels.test_current_scan_family import (
    _below,
    _draws,
    _signed,
)
from demos.integrations.mlx.tests.kernels.test_current_searchsorted import (
    _metal_bytes,
    _runtime_value,
)
from demos.integrations.mlx.tests.kernels.test_current_sort import _execute
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_QUANTIZED"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/quantized.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
SOURCE = "mlx/backend/metal/kernels/quantized.metal"
SOURCE_SHA256 = "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
HEADERS = {
    "mlx/backend/metal/kernels/quantized.h": (
        "8e2ca65d8d8dafe67b97120d1127d54ce57032f621bdba161e54a0bc4a948b8c"
    ),
    "mlx/backend/metal/kernels/quantized_utils.h": (
        "a12841b57d505f6cca81631901b845dbf76bc90dfc099e7f40c763b2cc838c51"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
    "mlx/backend/metal/kernels/bf16.h": (
        "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb"
    ),
}
# quantize_impl and get_quantize_kernel_dims, whose bindings and grid the
# cases follow.
HOST_SOURCE = "mlx/backend/metal/quantized.cpp"
HOST_SHA256 = "36b79607ee01bc213511dcdad590b2dc3b96099300551cbc3e7e6c7e13ba5982"
UNIT_ENTRY_COUNT = 2052

# quantized.metal's instantiation order: instantiate_quantized_all() walks
# the bit widths, instantiate_quantized_groups the group sizes,
# instantiate_quantized_types the element types, and
# instantiate_quantized_all_single starts with these two kernels.
KINDS = ("affine_quantize", "affine_dequantize")
DTYPES = (("float", "float32"), ("float16_t", "float16"), ("bfloat16_t", "bfloat16"))
GROUP_SIZES = (128, 64, 32)
BITS = (2, 3, 4, 5, 6, 8)
# Kernels of quantized.metal this gate does not cover yet.
EXCLUDED_KERNELS = (
    "affine_gather_qmv_fast",
    "affine_gather_qmv",
    "affine_gather_qvm",
    "affine_gather_qmm_n",
    "affine_qmv_fast",
    "affine_qmv",
    "affine_qvm",
    "affine_qmm_n",
    "affine_gather_qmm_t",
    "affine_qmm_t",
    "affine_qmv_quad",
    "affine_qmv_wide",
    "affine_qvm_split_k",
    "affine_qmm_t_splitk",
    "affine_gather_qmm_rhs_nt",
    "affine_gather_qmm_rhs_nn",
)

SIMD_SIZE = 32
WORKGROUP = (64, 1, 1)
CASES = ("row", "grid")
GRIDS = {
    "affine_quantize": {"row": (3, 1, 1), "grid": (2, 2, 1)},
    # A row of two keeps 8-bit, 128-value groups whole.
    "affine_dequantize": {"row": (2, 1, 1), "grid": (2, 2, 1)},
}
# [[buffer(i)]] parameters of the kernels, as (name, index).
PARAMETERS = {
    "affine_quantize": (("w", 0), ("out", 1), ("scales", 2), ("biases", 3)),
    "affine_dequantize": (("w", 0), ("scales", 1), ("biases", 2), ("out", 3)),
}
OUTPUTS = {
    "affine_quantize": ("out", "scales", "biases"),
    "affine_dequantize": ("out",),
}
# One quantization group per simdgroup; the row and grid cases hold six and
# eight of them.
SCENARIOS = {
    "row": ("upper", "lower", "zero", "negative", "constant", "inexact"),
    "grid": (
        "lower",
        "inexact",
        "upper",
        "negative",
        "inexact",
        "zero",
        "upper",
        "lower",
    ),
}
EPS = struct.unpack("<f", struct.pack("<f", 1e-7))[0]
GUARD_COUNT = 8
SENTINEL = -3.25
BYTE_SENTINEL = 0xA5
POISON = 1.5
BYTE_POISON = 0x5A

# Expressions OpenGL may evaluate in 32 bits: quantize's input, group and
# packed-output indices, and dequantize's group, output and packed-input
# offsets with the complete rebased pointer subscripts. The model shows each
# names an element inside its buffer, and every case's buffers hold far fewer
# than INDEX_BOUND elements.
INDEX_BOUND = 2**20 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "affine_quantize": (
        "in_index + i",
        "gindex",
        "out_index",
        "out_index / writes_per_reduce",
        "out_index + i / pack_factor",
    ),
    "affine_dequantize": (
        "gindex",
        "oindex",
        "offset",
        "offset * bytes_per_pack",
        "out_ + i",
        *(f"out_ + {index}" for index in range(8)),
        *(f"w + {index}" for index in range(5)),
    ),
}


def _instantiations():
    """``(entry, info)`` in quantized.metal's instantiation order."""
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for itype, dtype in DTYPES:
                for kind in KINDS:
                    yield f"{kind}_{itype}_gs_{group_size}_b_{bits}", {
                        "kind": kind,
                        "itype": itype,
                        "dtype": dtype,
                        "groupSize": group_size,
                        "bits": bits,
                    }


INFO = dict(_instantiations())
ENTRIES = tuple(INFO)


def _geometry(bits, group_size):
    """quantize's and dequantize's packing constants for one entry."""
    pack_factor = 8 if bits in (3, 5) else 4 if bits == 6 else 8 // bits
    power_of_2_bits = bits & (bits - 1) == 0
    bytes_per_pack = 1 if power_of_2_bits else 5 if bits == 5 else 3
    values_per_reduce = group_size // SIMD_SIZE
    writes_per_reduce = pack_factor // values_per_reduce
    writes_per_pack = 1 if writes_per_reduce > 1 else values_per_reduce // pack_factor
    return {
        "packFactor": pack_factor,
        "bytesPerPack": bytes_per_pack,
        "valuesPerReduce": values_per_reduce,
        "writesPerReduce": writes_per_reduce,
        "writesPerPack": writes_per_pack,
        "powerOf2Bits": power_of_2_bits,
    }


def _config(directory, target, entry):
    """The translation of one entry, as the gate and contract run it."""
    info = INFO[entry]
    options = {}
    if info["kind"] == "affine_quantize":
        # simd_min, simd_max and simd_shuffle_down; dequantize has no
        # subgroup operation.
        options = {
            "directx": {
                "software_subgroup_width": SIMD_SIZE,
                "relative_wave_shuffle_out_of_range": "self",
                "binary32_division_profile": "rne-flush",
            },
            "opengl": {"software_subgroup_width": SIMD_SIZE},
        }
    assertions = (
        OPENGL_INDEX_RANGE_ASSERTIONS[info["kind"]] if target == "opengl" else ()
    )
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
                "target_options": options,
            }
        },
    )


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def _f32(value):
    """Round to binary32; one double operation then this is exact IEEE."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _to_type(dtype, value):
    """``static_cast<T>`` of a float32 value, rounding to nearest even."""
    value = _f32(value)
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<e", value))[0]
    if dtype == "bfloat16":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return value


def _exact(dtype, value):
    return math.isfinite(value) and _to_type(dtype, value) == value


def _metal_round(value):
    """Metal ``round``: halfway cases away from zero, keeping the sign."""
    return math.copysign(math.floor(abs(value) + 0.5), value)


def _quantize_group(values, bits, mutation=None):
    """affine_quantize's arithmetic for one group: scale, bias and codes."""
    n_bins = float((1 << bits) - 1)
    # w_min starts at Limits<T>::max (infinity) but w_max at zero.
    w_min, w_max = math.inf, 0.0
    for value in values:
        w_min, w_max = min(w_min, value), max(w_max, value)
    scale = max(_f32(_f32(w_max - w_min) / n_bins), EPS)
    side = abs(w_min) > abs(w_max)
    if mutation == "edge":
        side = False
    scale = scale if side else -scale
    edge = w_min if side else w_max
    q0 = _metal_round(_f32(edge / scale))
    at_zero = q0 == 0.0
    scale = scale if at_zero else _f32(edge / q0)
    bias = 0.0 if at_zero else edge
    rounding = math.trunc if mutation == "trunc" else _metal_round
    codes = []
    for value in values:
        code = min(rounding(_f32(_f32(value - bias) / scale)), n_bins)
        # The uint8_t conversion needs a code in [0, n_bins]; a defect may not.
        assert mutation is not None or 0 <= code <= n_bins, (values, code)
        codes.append(int(code) & 0xFF)
    return {
        "scale": scale,
        "bias": bias,
        "codes": codes,
        "side": side,
        "atZero": at_zero,
    }


def _pack_bitstream(codes, bits):
    """MLX's packed layout: codes as one least-significant-bit-first stream."""
    stream = 0
    for index, code in enumerate(codes):
        stream |= code << (bits * index)
    return list(stream.to_bytes(len(codes) * bits // 8, "little"))


def _unpack_bitstream(payload, bits, count):
    stream = int.from_bytes(bytes(payload), "little")
    return [(stream >> (bits * index)) & ((1 << bits) - 1) for index in range(count)]


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _threads(entry, case):
    x, y, z = GRIDS[INFO[entry]["kind"]][case]
    return x * y * z * WORKGROUP[0]


def _group_values(entry, case, group, scenario):
    """One quantization group's values for ``scenario``."""
    info = INFO[entry]
    dtype, size = info["dtype"], info["groupSize"]
    n_bins = (1 << info["bits"]) - 1
    draws = _draws("quantized", entry, case, group)
    scale = 2.0 ** (_below(draws, 6) - 3)
    if scenario == "zero":
        return [0.0] * size
    if scenario == "constant":
        return [2.0 ** -(4 + _below(draws, 6))] * size
    if scenario == "inexact":
        return [_signed(draws, 128) / 64 for _ in range(size)]
    if scenario == "upper":
        offset = n_bins - _below(draws, n_bins // 2 + 1)
    elif scenario == "lower":
        offset = _below(draws, n_bins // 2 + 1)
    else:
        assert scenario == "negative", scenario
        offset = n_bins
    top = n_bins - 1 if scenario == "negative" else n_bins
    codes = [_below(draws, top + 1) for _ in range(size)]
    low = _below(draws, size)
    high = (low + 1 + _below(draws, size - 1)) % size
    codes[low], codes[high] = 0, top
    values = []
    for index, code in enumerate(codes):
        quarter = 0.0 if index in (low, high) else _signed(draws, 1) / 4
        if code == 0:
            quarter = abs(quarter)
        elif code == top:
            quarter = -abs(quarter)
        value = (code + quarter - offset) * scale
        values.append(value if _exact(dtype, value) else (code - offset) * scale)
    return values


def _dequantize_inputs(entry, case):
    """Random codes per group, packed, with exact scales and biases."""
    info = INFO[entry]
    bits, size = info["bits"], info["groupSize"]
    n_bins = (1 << bits) - 1
    count = _threads(entry, case) * _geometry(bits, size)["packFactor"]
    codes, scales, biases = [], [], []
    for group in range(count // size):
        draws = _draws("dequantized", entry, case, group)
        values = [_below(draws, n_bins + 1) for _ in range(size)]
        low = _below(draws, size)
        values[low], values[(low + 1 + _below(draws, size - 1)) % size] = 0, n_bins
        codes += values
        scale = (2.0 ** (_below(draws, 6) - 3)) * (1 if _below(draws, 2) else -1)
        scales.append(scale)
        biases.append(-_below(draws, n_bins + 1) * scale)
    return codes, scales, biases


@functools.lru_cache(maxsize=None)
def _case_data(entry, case):
    """Logical storage of every parameter, the grid and the expected outputs."""
    info = INFO[entry]
    dtype, kind = info["dtype"], info["kind"]
    if kind == "affine_quantize":
        scenarios = SCENARIOS[case]
        assert len(scenarios) * SIMD_SIZE == _threads(entry, case), (entry, case)
        w = []
        for group, scenario in enumerate(scenarios):
            w += _group_values(entry, case, group, scenario)
        assert all(_exact(dtype, value) for value in w), entry
        initial = {
            "out": [BYTE_SENTINEL] * (len(w) * info["bits"] // 8 + GUARD_COUNT),
            "scales": [SENTINEL] * (len(scenarios) + GUARD_COUNT),
            "biases": [SENTINEL] * (len(scenarios) + GUARD_COUNT),
        }
        kinds = {"w": dtype, "out": "uint8", "scales": dtype, "biases": dtype}
        storage = {"w": w + [POISON] * GUARD_COUNT, **initial}
    else:
        codes, scales, biases = _dequantize_inputs(entry, case)
        initial = {"out": [SENTINEL] * (len(codes) + GUARD_COUNT)}
        kinds = {"w": "uint8", "scales": dtype, "biases": dtype, "out": dtype}
        storage = {
            "w": _pack_bitstream(codes, info["bits"]) + [BYTE_POISON] * GUARD_COUNT,
            "scales": scales + [POISON] * GUARD_COUNT,
            "biases": biases + [POISON] * GUARD_COUNT,
            **initial,
        }
    data = {
        "kinds": kinds,
        "values": {name: (kinds[name], values) for name, values in storage.items()},
        "initial": initial,
        "grid": GRIDS[kind][case],
        "threads": _threads(entry, case),
    }
    data["expected"] = _simulate(entry, data)["outputs"]
    return data


# ---------------------------------------------------------------------------
# Kernel model
# ---------------------------------------------------------------------------


MUTATIONS = {
    "affine_quantize": {
        "shuffle": "drops the neighbouring lanes' codes",
        "trunc": "truncates codes instead of rounding them",
        "edge": "always takes the upper edge",
    },
    "affine_dequantize": {
        "bias": "drops the bias",
        "order": "reverses the codes of each pack",
    },
}


def _simulate(entry, data, mutation=None):
    """The kernel thread by thread; returns outputs, stores and indices."""
    kind = INFO[entry]["kind"]
    if kind == "affine_quantize":
        return _simulate_quantize(entry, data, mutation)
    return _simulate_dequantize(entry, data, mutation)


def _simulate_quantize(entry, data, mutation):
    info = INFO[entry]
    dtype, bits, size = info["dtype"], info["bits"], info["groupSize"]
    geometry = _geometry(bits, size)
    pack_factor, bytes_per_pack = geometry["packFactor"], geometry["bytesPerPack"]
    per_reduce, writes = geometry["valuesPerReduce"], geometry["writesPerReduce"]
    w = data["values"]["w"][1]
    outputs = {name: list(values) for name, values in data["initial"].items()}
    stores = Counter()
    indices = {
        expression: [] for expression in OPENGL_INDEX_RANGE_ASSERTIONS[info["kind"]]
    }
    groups = []

    def store(name, index, value):
        assert 0 <= index < len(outputs[name]) - GUARD_COUNT, (name, index)
        stores[name, index] += 1
        outputs[name][index] = value

    for base in range(0, data["threads"], SIMD_SIZE):
        # One simdgroup holds one group: simd_min and simd_max reduce it.
        group = _quantize_group(
            w[base * per_reduce : (base + SIMD_SIZE) * per_reduce], bits, mutation
        )
        groups.append(group)
        codes = group["codes"]

        def code(lane, i):
            return codes[(lane - base) * per_reduce + i]

        for lane in range(base, base + SIMD_SIZE):
            in_index = lane * per_reduce
            out_index = (
                lane * geometry["writesPerPack"]
                if geometry["powerOf2Bits"]
                else lane * bytes_per_pack // writes
            )
            gindex = in_index // size
            indices["in_index + i"] += [in_index + i for i in range(per_reduce)]
            indices["gindex"].append(gindex)
            if bits in (3, 5, 6):
                # Only these widths subscript out with out_index itself.
                indices["out_index"].append(out_index)
            if in_index % size == 0:
                store("scales", gindex, _to_type(dtype, group["scale"]))
                store("biases", gindex, _to_type(dtype, group["bias"]))
            output = 0
            for i in range(per_reduce):
                value = code(lane, i)
                if bits == 8:
                    output = value
                else:
                    output |= value << (bits * (i % pack_factor))
                if pack_factor < per_reduce and i % pack_factor == pack_factor - 1:
                    indices["out_index + i / pack_factor"].append(
                        out_index + i // pack_factor
                    )
                    store("out", out_index + i // pack_factor, output & 0xFF)
                    output = 0
                else:
                    for j in range(1, writes):
                        # simd_shuffle_down leaves the upper lanes unchanged;
                        # the lanes that write never read past the simdgroup.
                        source = lane + j if lane + j < base + SIMD_SIZE else lane
                        shuffled = 0 if mutation == "shuffle" else code(source, i)
                        output |= shuffled << (bits * (j * per_reduce + i))
            if bits in (3, 5, 6):
                if in_index % pack_factor == 0 and out_index % bytes_per_pack == 0:
                    for byte in range(bytes_per_pack):
                        store("out", out_index + byte, (output >> (8 * byte)) & 0xFF)
            elif writes > 0 and out_index % writes == 0:
                indices["out_index / writes_per_reduce"].append(out_index // writes)
                store("out", out_index // writes, output & 0xFF)
    return {"outputs": outputs, "stores": stores, "indices": indices, "groups": groups}


def _dequantized_codes(bits, payload, offset, pack_factor, bytes_per_pack):
    """affine_dequantize's per-thread unpacking, expression for expression."""
    if bits in (3, 5, 6):
        b = payload[offset * bytes_per_pack : (offset + 1) * bytes_per_pack]
        if bits == 3:
            return [
                b[0] & 0x7,
                (b[0] & 0x38) >> 3,
                ((b[0] & 0xC0) >> 6) + ((b[1] & 0x1) << 2),
                (b[1] & 0xE) >> 1,
                (b[1] & 0x70) >> 4,
                ((b[1] & 0x80) >> 7) + ((b[2] & 0x3) << 1),
                (b[2] & 0x1C) >> 2,
                (b[2] & 0xE0) >> 5,
            ]
        if bits == 5:
            return [
                b[0] & 0x1F,
                ((b[0] & 0xE0) >> 5) + ((b[1] & 0x3) << 3),
                (b[1] & 0x7C) >> 2,
                ((b[1] & 0x80) >> 7) + ((b[2] & 0xF) << 1),
                ((b[2] & 0xF0) >> 4) + ((b[3] & 0x1) << 4),
                (b[3] & 0x3E) >> 1,
                ((b[3] & 0xC0) >> 6) + ((b[4] & 0x7) << 2),
                (b[4] & 0xF8) >> 3,
            ]
        return [
            b[0] & 0x3F,
            ((b[0] >> 6) & 0x03) + ((b[1] & 0x0F) << 2),
            ((b[1] >> 4) & 0x0F) + ((b[2] & 0x03) << 4),
            (b[2] >> 2) & 0x3F,
        ]
    value = payload[offset]
    return [(value >> (bits * i)) & ((1 << bits) - 1) for i in range(pack_factor)]


def _simulate_dequantize(entry, data, mutation):
    info = INFO[entry]
    dtype, bits, size = info["dtype"], info["bits"], info["groupSize"]
    geometry = _geometry(bits, size)
    pack_factor, bytes_per_pack = geometry["packFactor"], geometry["bytesPerPack"]
    payload = data["values"]["w"][1]
    scales, biases = data["values"]["scales"][1], data["values"]["biases"][1]
    outputs = {"out": list(data["initial"]["out"])}
    stores = Counter()
    indices = {
        expression: [] for expression in OPENGL_INDEX_RANGE_ASSERTIONS[info["kind"]]
    }
    for offset in range(data["threads"]):
        oindex = offset * pack_factor
        gindex = oindex // size
        indices["gindex"].append(gindex)
        indices["oindex"].append(oindex)
        if bits in (3, 5, 6):
            indices["offset * bytes_per_pack"].append(offset * bytes_per_pack)
            for byte in range(bytes_per_pack):
                indices[f"w + {byte}"].append(offset * bytes_per_pack + byte)
        else:
            indices["offset"].append(offset)
        codes = _dequantized_codes(bits, payload, offset, pack_factor, bytes_per_pack)
        assert len(codes) == pack_factor
        if mutation == "order":
            codes = codes[::-1]
        bias = 0.0 if mutation == "bias" else biases[gindex]
        for i, code in enumerate(codes):
            value = scales[gindex] * code + bias
            # Every product and sum is exact in the element type, whatever the
            # order or contraction.
            assert _exact(dtype, value), (entry, offset, i, value)
            indices["out_ + i" if bits in (2, 4, 8) else f"out_ + {i}"].append(
                oindex + i
            )
            assert 0 <= oindex + i < len(outputs["out"]) - GUARD_COUNT
            stores["out", oindex + i] += 1
            outputs["out"][oindex + i] = value
    return {"outputs": outputs, "stores": stores, "indices": indices, "groups": []}


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


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
            "affine_quantize lane by lane: the simdgroup minimum and maximum, "
            "scale, edge and zero-point arithmetic, Metal rounding, shuffled "
            "packing and guarded byte writes; affine_dequantize thread by "
            "thread: its unpacking expressions and scale * code + bias"
        ),
        "packing": "least-significant-bit-first code stream, checked per entry",
        "comparison": "exact words",
        "workgroupSize": list(WORKGROUP),
        "grids": {
            kind: {case: list(GRIDS[kind][case]) for case in CASES} for kind in KINDS
        },
        "quantizeGroups": {case: list(SCENARIOS[case]) for case in CASES},
        "inputs": {
            "quantize": (
                "(k + d - a) * 2^e with d in {-1/4, 0, 1/4}, e in [-3, 2] and a "
                "placing the larger magnitude at either end; zero, all-negative, "
                "positive constant and free multiples of 1/64 in [-2, 2]"
            ),
            "dequantize": (
                "random codes with 0 and the top code in every group, scales "
                "+-2^e with e in [-3, 2] and biases -a * scale"
            ),
        },
        "guards": {
            "count": GUARD_COUNT,
            "sentinel": SENTINEL,
            "byteSentinel": BYTE_SENTINEL,
        },
        "softwareSubgroups": {
            "kernels": ["affine_quantize"],
            "width": SIMD_SIZE,
            "directxShuffleOutOfRange": "self",
        },
        "directxDivision": {
            "kernels": ["affine_quantize"],
            "binary32_division_profile": "rne-flush",
        },
        "openglIndexRangeAssertions": {
            kind: {
                "expressions": list(OPENGL_INDEX_RANGE_ASSERTIONS[kind]),
                "minimum": 0,
                "maximum": INDEX_BOUND,
            }
            for kind in KINDS
        },
        "metalUpstreamComparison": "unchanged quantized.metal, word for word",
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
        "kind": "crosstl-mlx-current-tree-quantized-contract",
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


def test_current_mlx_quantized_contract_is_exact():
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


def test_quantized_entries_follow_mlx_instantiation():
    assert len(ENTRIES) == 108 == len(set(ENTRIES))
    assert Counter(info["kind"] for info in INFO.values()) == {
        "affine_quantize": 54,
        "affine_dequantize": 54,
    }
    assert ENTRIES[:4] == (
        "affine_quantize_float_gs_128_b_2",
        "affine_dequantize_float_gs_128_b_2",
        "affine_quantize_float16_t_gs_128_b_2",
        "affine_dequantize_float16_t_gs_128_b_2",
    )
    assert ENTRIES[-1] == "affine_dequantize_bfloat16_t_gs_32_b_8"
    assert not set(KINDS) & set(EXCLUDED_KERNELS)


# ---------------------------------------------------------------------------
# Model checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_model_matches_mlx_layout(entry):
    info = INFO[entry]
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
        if info["kind"] == "affine_quantize":
            codes = [code for group in run["groups"] for code in group["codes"]]
            # The lanes' shuffled writes reproduce MLX's packed stream.
            assert outputs["out"][:-GUARD_COUNT] == _pack_bitstream(codes, info["bits"])
            for output, name in (("scales", "scale"), ("biases", "bias")):
                assert outputs[output][:-GUARD_COUNT] == [
                    _to_type(info["dtype"], group[name]) for group in run["groups"]
                ]
        else:
            payload = data["values"]["w"][1]
            count = len(outputs["out"]) - GUARD_COUNT
            codes = _unpack_bitstream(
                payload[: count * info["bits"] // 8], info["bits"], count
            )
            scales, biases = data["values"]["scales"][1], data["values"]["biases"][1]
            size = info["groupSize"]
            assert outputs["out"][:count] == [
                code * scales[index // size] + biases[index // size]
                for index, code in enumerate(codes)
            ]


def test_quantize_groups_reach_every_branch():
    seen = Counter()
    for entry in ENTRIES:
        if INFO[entry]["kind"] != "affine_quantize":
            continue
        for case in CASES:
            data = _case_data(entry, case)
            for scenario, group in zip(
                SCENARIOS[case], _simulate(entry, data)["groups"]
            ):
                seen[scenario, group["side"], group["atZero"]] += 1
                if scenario in ("upper", "negative"):
                    assert group["side"] and not group["atZero"], entry
                elif scenario in ("lower", "constant"):
                    assert not group["side"] and not group["atZero"], entry
                elif scenario == "zero":
                    assert group["atZero"] and group["scale"] == -EPS, entry
                if scenario == "constant":
                    # max(0 / n_bins, eps), then the rounded edge / q0.
                    assert set(group["codes"]) == {0}
    assert {key[0] for key in seen} == set(SCENARIOS["row"])


@pytest.mark.parametrize("dtype", [dtype for _, dtype in DTYPES])
def test_quantized_inputs_are_exact_in_their_element_type(dtype):
    entries = [entry for entry in ENTRIES if INFO[entry]["dtype"] == dtype]
    for entry in entries:
        for case in CASES:
            data = _case_data(entry, case)
            for name, (kind, values) in data["values"].items():
                if kind == dtype:
                    assert all(_exact(dtype, value) for value in values), (entry, name)
            for name, values in data["expected"].items():
                if data["kinds"][name] == dtype:
                    assert all(_exact(dtype, value) for value in values), (entry, name)


@pytest.mark.parametrize(
    "entry,mutation",
    [
        (entry, mutation)
        for entry in ENTRIES
        for mutation in MUTATIONS[INFO[entry]["kind"]]
        # Shuffles only carry codes when a lane packs other lanes' values,
        # and an 8-bit pack holds one code.
        if (
            mutation != "shuffle"
            or _geometry(INFO[entry]["bits"], INFO[entry]["groupSize"])[
                "writesPerReduce"
            ]
            > 1
        )
        and (mutation != "order" or INFO[entry]["bits"] != 8)
    ],
)
def test_quantized_expectations_reject_defects(entry, mutation):
    changed = False
    for case in CASES:
        data = _case_data(entry, case)
        changed |= _simulate(entry, data, mutation)["outputs"] != data["expected"]
    assert changed, (entry, mutation, MUTATIONS[INFO[entry]["kind"]][mutation])


@pytest.mark.parametrize("entry", ENTRIES)
def test_quantized_opengl_assertions_name_buffer_elements(entry):
    kind = INFO[entry]["kind"]
    used = set()
    for case in CASES:
        data = _case_data(entry, case)
        lengths = {name: len(values) for name, (_, values) in data["values"].items()}
        buffers = {
            "in_index + i": "w",
            "gindex": "scales",
            "out_index": "out",
            "out_index / writes_per_reduce": "out",
            "out_index + i / pack_factor": "out",
            "oindex": "out",
            "offset": "w",
            "offset * bytes_per_pack": "w",
        }
        for expression, values in _simulate(entry, data)["indices"].items():
            if not values:
                continue
            used.add(expression)
            name = buffers.get(expression, expression.split(" + ")[0].rstrip("_"))
            assert 0 <= min(values) and max(values) < lengths[name], (expression, name)
            assert max(values) <= INDEX_BOUND
    assert used <= set(OPENGL_INDEX_RANGE_ASSERTIONS[kind])


@pytest.mark.parametrize("target", TARGETS)
def test_quantized_configs_scope_target_options(tmp_path, target):
    for entry in ENTRIES:
        config = _config(tmp_path / "work", target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[SOURCE].values()
        assert tuple(int(value) for value in rule) == WORKGROUP
        options = config.source_options["metal"]["target_options"]
        if INFO[entry]["kind"] == "affine_quantize":
            assert options == {
                "directx": {
                    "software_subgroup_width": SIMD_SIZE,
                    "relative_wave_shuffle_out_of_range": "self",
                    "binary32_division_profile": "rne-flush",
                },
                "opengl": {"software_subgroup_width": SIMD_SIZE},
            }
        else:
            assert options == {}
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


def _macro(text, name):
    match = re.search(rf"#define {name}\((.*?)\)(.*?)(?:\n\n|\n#|\Z)", text, re.DOTALL)
    assert match, name
    return match.group(2)


def _kernel_parameters(text, kernel):
    match = re.search(rf"\]\]\s*void\s+{kernel}\s*\((.*?)\)\s*\{{", text, re.DOTALL)
    assert match, kernel
    found = re.findall(r"(\w+)\s*\[\[buffer\((\d+)\)", match.group(1))
    return tuple((name, int(index)) for name, index in found)


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native quantized gate")
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
    # ENTRIES follow quantized.metal's own instantiation macros, and the
    # parameter tuples quantized.h's [[buffer(i)]] signatures.
    text = source.decode("utf-8")
    single = _macro(text, "instantiate_quantized_all_single")
    assert re.findall(r"instantiate_quantized\((\w+),", single)[:2] == list(KINDS)
    types = _macro(text, "instantiate_quantized_types")
    assert re.findall(r"instantiate_quantized_funcs\((\w+),", types) == [
        itype for itype, _ in DTYPES
    ]
    groups = _macro(text, "instantiate_quantized_groups")
    assert re.findall(r"instantiate_quantized_types\((\d+),", groups) == [
        str(size) for size in GROUP_SIZES
    ]
    widths = _macro(text, "instantiate_quantized_all")
    assert re.findall(r"instantiate_quantized_groups\((\d+)\)", widths) == [
        str(bits) for bits in BITS
    ]
    header = (root / "mlx/backend/metal/kernels/quantized.h").read_text(
        encoding="utf-8"
    )
    for kind in KINDS:
        assert _kernel_parameters(header, kind) == PARAMETERS[kind]
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("quantized-metal-reference")
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

    kind = INFO[entry]["kind"]
    values = {
        name: _runtime_value(target, dtype, storage)
        for name, (dtype, storage) in data["values"].items()
    }
    inputs, observed, outputs, generated = {}, [], {}, []
    for binding in descriptor["bindings"]:
        provenance = binding.get("provenance", {})
        if "executionInput" in provenance:
            # HLSL has no threads_per_grid builtin: the flattened offset's row
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
            observed.append(binding["name"])
    assert len(generated) == (1 if target == "directx" else 0), generated
    assert set(outputs) == set(OUTPUTS[kind]), outputs
    assert len(observed) == len(PARAMETERS[kind]) - len(OUTPUTS[kind])
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
        observe_inputs=observed,
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


def _words(dtype, raw):
    """Raw Metal readback in the runtime's logical word encoding."""
    if dtype == "uint8":
        return list(raw)
    if dtype == "float32":
        return list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


def _upstream(work, reference, entry, label, data):
    """Run the unchanged upstream kernel on raw ``[[buffer(i)]]`` storage."""
    runner, library = reference
    kind = INFO[entry]["kind"]
    files, payloads = [], {}
    for name, index in PARAMETERS[kind]:
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


def _assert_identical(label, actual, expected):
    assert len(actual) == len(expected), label
    mismatches = [
        (index, want, got)
        for index, (got, want) in enumerate(zip(actual, expected))
        if got != want
    ]
    assert not mismatches, (label, len(mismatches), mismatches[:8])


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_quantized_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-quantized-", dir=root) as path:
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
                                f"{case}-{name}", words, upstream_words[name]
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


def test_ci_requires_native_quantized_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow,
        "Prove current MLX quantize and dequantize native parity",
        group="quantized",
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/quantized",
        "--junitxml=mlx-current-results/quantized.xml",
        "demos/integrations/mlx/tests/kernels/test_current_quantized.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_quantized.py"
    )
