"""Translate and execute the pinned MLX scaled dot-product attention kernels."""

import functools
import hashlib
import json
import math
import operator
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
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
SOURCE_SHA256 = "21a6c5a69a6c98e2b353fadc0cef00c46d26fac4aa6b55637ec454520ea7bc56"
HEADERS = {
    "mlx/backend/metal/kernels/sdpa_vector.h": (
        "f52c8da21f2c31f3f77168d326a49815fc29bde4b573883896a8eb3298f11508"
    ),
    "mlx/backend/metal/kernels/sdpa_blocked.h": (
        "665d98c389b042a2cbdfae9d95263d7d4ae56d9fc1dfe903b914940c82c8a4f3"
    ),
    "mlx/backend/metal/kernels/utils.h": (
        "e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf"
    ),
}
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SDPA"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts"
    / "scaled_dot_product_attention.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# scaled_dot_product_attention.metal's instantiation macros, in order.
TYPE_NAMES = {"float": "float32", "bfloat16_t": "bfloat16", "float16_t": "float16"}
HEAD_DIMS = (
    (64, 64),
    (96, 64),
    (96, 96),
    (128, 128),
    (192, 128),
    (192, 192),
    (256, 256),
    (512, 512),
)
GQA_HEADS = ((64, 64, 8), (128, 128, 8), (128, 128, 12), (128, 128, 16))
AGGREGATION_DIMS = (64, 96, 128, 192, 256, 512)
BLOCKED_ENTRIES = (
    "sdpa_blocked_scale_copy_bfloat16",
    "sdpa_blocked_scale_copy_float16",
    "sdpa_blocked_causal_mask_bfloat16",
    "sdpa_blocked_causal_mask_float16",
)


def _instantiated():
    entries = list(BLOCKED_ENTRIES)
    for type_name in TYPE_NAMES:
        for qk_dim, value_dim in HEAD_DIMS:
            entries.append(f"sdpa_vector_{type_name}_{qk_dim}_{value_dim}")
            entries.append(f"sdpa_vector_2pass_1_{type_name}_{qk_dim}_{value_dim}")
        for qk_dim, value_dim, factor in GQA_HEADS:
            entries.append(
                f"sdpa_vector_2pass_1_gqa_{factor}_{type_name}_{qk_dim}_{value_dim}"
            )
        for value_dim in AGGREGATION_DIMS:
            entries.append(f"sdpa_vector_2pass_2_{type_name}_{value_dim}")
    return tuple(entries)


INSTANTIATED = _instantiated()


def _kind(entry):
    if entry.startswith("sdpa_blocked_scale_copy_"):
        return "scale-copy"
    if entry.startswith("sdpa_blocked_causal_mask_"):
        return "causal-mask"
    if entry.startswith("sdpa_vector_2pass_1_gqa_"):
        return "gqa"
    if entry.startswith("sdpa_vector_2pass_1_"):
        return "pass1"
    if entry.startswith("sdpa_vector_2pass_2_"):
        return "pass2"
    return "vector"


# The first pass of the two-pass kernels does not translate to OpenGL or
# DirectX yet: sdpa_vector_2pass_1 strides its key loop by function constant 26
# from threadgroup_position_in_grid.z, and sdpa_vector_2pass_1_gqa bounds it by
# per-SIMD-group token chunks, so neither software-subgroup lowering can prove
# the loop uniform across the workgroup. Both translate back to Metal.
PENDING_KINDS = {
    "pass1": "key loop starts at block_idx and steps by function constant 26",
    "gqa": "key loop bounds s0 and s1 depend on the SIMD group",
}
ENTRIES = tuple(entry for entry in INSTANTIATED if _kind(entry) not in PENDING_KINDS)
PENDING_ENTRIES = tuple(
    entry for entry in INSTANTIATED if _kind(entry) in PENDING_KINDS
)


def _dtype(entry):
    kind = _kind(entry)
    if kind in {"scale-copy", "causal-mask"}:
        return entry.rsplit("_", 1)[1]
    match = re.match(r"sdpa_vector_(?:2pass_[12]_(?:gqa_\d+_)?)?(.+?)_\d+", entry)
    return TYPE_NAMES[match.group(1)]


def _dims(entry):
    """Return (query/key head dim, value head dim) of a vector kernel."""
    if _kind(entry) == "pass2":
        value_dim = int(entry.rsplit("_", 1)[1])
        return value_dim, value_dim
    return tuple(int(item) for item in entry.split("_")[-2:])


# mlx/backend/metal/scaled_dot_product_attention.cpp specializes six function
# constants for sdpa_vector; the other translated kernels have none.
CONSTANT_IDS = {
    "has_mask": 20,
    "query_transposed": 21,
    "do_causal": 22,
    "bool_mask": 23,
    "float_mask": 24,
    "has_sinks": 25,
}
# A pairwise-covering set of the specializations sdpa_vector() can select:
# every mask mode (none, bool, float or causal) with both query layouts and
# both sink settings, and every query layout with both sink settings.
VECTOR_SPECIALIZATIONS = (
    ("none", False, False),
    ("none", True, True),
    ("bool", False, True),
    ("bool", True, False),
    ("float", False, True),
    ("float", True, False),
    ("causal", False, False),
    ("causal", True, True),
)


def _vector_constants(mask, transposed, sinks):
    return {
        "has_mask": mask in {"bool", "float"},
        "query_transposed": transposed,
        "do_causal": mask == "causal",
        "bool_mask": mask == "bool",
        "float_mask": mask == "float",
        "has_sinks": sinks,
    }


def _variant_key(constants):
    """Name a specialization after sdpa_vector()'s pipeline hash suffix."""
    if not constants:
        return "default"
    if constants["has_mask"]:
        mask = "boolmask" if constants["bool_mask"] else "floatmask"
    else:
        mask = "nomask"
    return "_".join(
        (
            mask,
            "qt" if constants["query_transposed"] else "qnt",
            "c" if constants["do_causal"] else "nc",
            "sinks" if constants["has_sinks"] else "nosinks",
        )
    )


def _combinations(entry):
    if _kind(entry) != "vector":
        return ({},)
    return tuple(_vector_constants(*item) for item in VECTOR_SPECIALIZATIONS)


VARIANTS = tuple(
    (entry, _variant_key(constants), constants)
    for entry in ENTRIES
    for constants in _combinations(entry)
)
SIMD_SIZE = 32
# sdpa_vector and sdpa_vector_2pass_2 run 32 SIMD groups of 32 threads; the
# blocked kernels dispatch one thread per element, so one-thread workgroups
# give OpenGL and DirectX the same threads_per_grid.
WORKGROUP_SIZES = {
    "vector": (1024, 1, 1),
    "pass2": (1024, 1, 1),
    "scale-copy": (1, 1, 1),
    "causal-mask": (1, 1, 1),
}
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
SOFTWARE_SUBGROUP_KINDS = ("vector", "pass2")
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# composed element indices and pointer offsets to signed int range.
SIGNED_MAX = 2**31 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "scale-copy": ("q_index", "q_block_index"),
    "causal-mask": ("index",),
    "vector": (
        "kv_head_idx * k_head_stride + simd_gid * k_seq_stride"
        " + simd_lid * qk_per_thread",
        "kv_head_idx * v_head_stride + simd_gid * v_seq_stride"
        " + simd_lid * v_per_thread",
        "keys + j",
        "values + j",
    ),
    "pass2": (),
}
# name -> (batch, query heads, key/value heads, query length, key length,
# key/value layout, mask layout, scale or None for 1/sqrt(D), values). Every
# case takes sdpa_vector: at most eight queries and fewer than 1024 keys.
# "cache" keys and values are the first N rows of a longer cache, and
# "transposed" ones a [B, N, H_kv, D] buffer viewed as [B, H_kv, N, D]. Array
# masks are [B, H, qL, N] ("full"), a [qL, N] mask broadcast over batch and
# heads ("broadcast"), or a [B, H, N, qL] buffer viewed as [B, H, qL, N]
# ("transposed"). "rising" keys make every later key score higher, so each
# SIMD group rescales its accumulators at every key.
VECTOR_CASES = {
    "decode": (2, 4, 2, 1, 37, "contiguous", "full", None, "random"),
    "short": (1, 2, 2, 1, 5, "cache", "broadcast", 1.0, "random"),
    "batches": (3, 1, 1, 3, 33, "cache", "full", None, "random"),
    "prefill": (1, 4, 2, 4, 70, "transposed", "transposed", 0.08, "random"),
    "rising": (1, 2, 1, 2, 96, "contiguous", "broadcast", None, "rising"),
}
# (flattened batch-head, query) rows whose array masks drop every key.
MASKED_ROWS = {"decode": ((1, 0),), "batches": ((2, 1),), "prefill": ((3, 2),)}
CACHE_SLACK = 7
# name -> (rows B * H, query length, blocks, values): synthetic first-pass
# partials. "sparse" blocks may hold no keys, one row has none at all and one
# block dominates each other row; "wide" maxima span forty units.
PASS2_CASES = {
    "one-chunk": (3, 1, 32, "random"),
    "two-chunks": (2, 2, 64, "sparse"),
    "four-chunks": (1, 3, 128, "wide"),
}
# sdpa_full_self_attention_blocked runs these kernels only for 512-wide heads,
# 256 query rows at a time.
BLOCK_ROWS = 256
BLOCKED_HEAD_DIM = 512
# name -> (batch, heads, query length, q_offset, rows, layout, scale or None
# for 1/sqrt(512)). Queries need only a unit-stride head dimension:
# "sequence-major" is a [B, L, H, D] buffer and "head-major" an [H, B, L, D]
# buffer, both viewed as [B, H, L, D].
SCALE_COPY_CASES = {
    "first-block": (1, 2, 3, 0, 3, "contiguous", None),
    "second-block": (1, 2, 259, 256, 3, "sequence-major", 0.125),
    "head-major": (2, 2, 4, 0, 4, "head-major", 0.3),
}
# name -> (score rows B * H * rows, tile rows, columns).
CAUSAL_MASK_CASES = {
    "square": (8, 4, 4),
    "offset": (15, 5, 9),
    "block": (256, 256, 300),
}
CASES = {
    "vector": VECTOR_CASES,
    "pass2": PASS2_CASES,
    "scale-copy": SCALE_COPY_CASES,
    "causal-mask": CAUSAL_MASK_CASES,
}
INPUT_GUARD_COUNT = 16
FLT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
FINITE_MIN = {
    "float32": -FLT_MAX,
    "float16": -65504.0,
    "bfloat16": -struct.unpack("<f", struct.pack("<I", 0x7F7F0000))[0],
}
# Each softmax output is checked against a float64 reference with a
# per-element bound on the kernel's own rounding. Every binary32 addition,
# product and conversion is within OP_ERROR (one unit in the last place,
# relative) and divisions within DIVISION_ERROR. Each exponential is within the
# larger of four and 3 + 2|x| units of exp(x) (Metal's fast::exp, GLSL's exp,
# D3D's 21-bit exp2) and its argument is a rounded difference, rounded again
# when DirectX scales it for exp2. Scores are dot products of D / 32 products
# per lane summed across 32 lanes in any order. Every key's weight passes
# through its own exponential, the rescaling factors of each later key of its
# SIMD group and the cross-group factor; the accumulators round at every key
# and the 32 partial results are summed in any order. Results must then be a
# correct rounding to the storage type of a value within the bound. The
# blocked kernels round once, so their results must be bit-exact.
OP_ERROR = 2.0**-23
DIVISION_ERROR = 2.5 * 2.0**-23
EXP_UNITS = 4.0
# The smallest normal binary32 value bounds each flushed underflow.
TINY = 2.0**-126
BOUND_SAFETY = 1.0 + 2.0**-10
SIGNIFICAND_BITS = {"float32": 24, "float16": 11, "bfloat16": 8}
# frexp exponent of the smallest normal value of each type.
MIN_NORMAL_EXPONENT = {"float32": -125, "float16": -13, "bfloat16": -125}


def _variant_cases(entry, constants):
    """Return the cases of one specialization.

    MLX transposes queries only for more than one query when the batch or
    the head count is one, and only applies the causal mask to more than one
    query, so those specializations take the multi-query cases.
    """
    kind = _kind(entry)
    if kind != "vector":
        return tuple(CASES[kind])
    if constants["query_transposed"] or constants["do_causal"]:
        return ("batches", "prefill", "rising")
    return ("decode", "short", "batches")


def _stream(name, salt):
    state = (zlib.crc32(f"{name}:{salt}".encode()) << 1) | 1
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        yield (state >> 11) / 2**53


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _gamma(count):
    return count * OP_ERROR / (1.0 - count * OP_ERROR)


def _exp_units(argument):
    """Units in the last place one exponential may add at ``argument``.

    The argument is a rounded difference, which DirectX rounds again when it
    scales the argument for exp2: each adds |x| units.
    """
    return max(EXP_UNITS, 3.0 + 2.0 * abs(argument)) + 2.0 * abs(argument)


def _read(buffer, index):
    return buffer[index] if 0 <= index < len(buffer) else math.nan


class _VectorCase:
    """Logical tensors of one sdpa_vector case and their MLX strides."""

    def __init__(self, dtype, qk_dim, value_dim, name):
        case = VECTOR_CASES[name]
        batch, heads, kv_heads, queries, keys, kv_layout, mask_layout = case[:7]
        scale, pattern = case[7:]
        self.name = name
        self.dtype = dtype
        self.qk_dim, self.value_dim = qk_dim, value_dim
        self.batch, self.heads, self.kv_heads = batch, heads, kv_heads
        self.queries, self.keys = queries, keys
        self.kv_layout, self.mask_layout = kv_layout, mask_layout
        self.scale = _float32(qk_dim**-0.5 if scale is None else scale)
        self.gqa = heads // kv_heads
        self.rows = batch * heads * queries
        noise = _stream(f"vector:{name}", f"{dtype}:{qk_dim}:{value_dim}")

        def uniform(low, high):
            return low + (high - low) * next(noise)

        quantize = functools.partial(_quantize, dtype)
        kv_count = batch * kv_heads
        if pattern == "rising":
            base = [uniform(-1.0, 1.0) for _ in range(qk_dim)]
            self.q = [
                [quantize(item + 0.05 * uniform(-1.0, 1.0)) for item in base]
                for _ in range(self.rows)
            ]
            self.k = [
                [
                    [quantize(item * (0.25 + 2.0 * key / keys)) for item in base]
                    for key in range(keys)
                ]
                for _ in range(kv_count)
            ]
        else:
            self.q = [
                [quantize(uniform(-1.0, 1.0)) for _ in range(qk_dim)]
                for _ in range(self.rows)
            ]
            self.k = [
                [
                    [quantize(uniform(-1.0, 1.0)) for _ in range(qk_dim)]
                    for _ in range(keys)
                ]
                for _ in range(kv_count)
            ]
        self.v = [
            [
                [quantize(uniform(-1.0, 1.0)) for _ in range(value_dim)]
                for _ in range(keys)
            ]
            for _ in range(kv_count)
        ]
        # Value columns of each key/value head, for the reference's sums.
        self.columns = [
            [[row[index] for row in head] for index in range(value_dim)]
            for head in self.v
        ]
        self.sinks = [quantize(uniform(-2.0, 2.0)) for _ in range(heads)]
        masked = set(MASKED_ROWS.get(name, ()))
        choices = (0.0, 0.0, 0.0, -0.75, 0.5, -3.0, 1.25, -math.inf, -math.inf, -12.0)
        self.bool_mask, self.float_mask = [], []
        shared = None
        for bh in range(batch * heads):
            bools, floats = [], []
            for query in range(queries):
                if mask_layout == "broadcast" and shared is not None:
                    bools.append(list(shared[0][query]))
                    floats.append(list(shared[1][query]))
                    continue
                if (bh, query) in masked:
                    bools.append([False] * keys)
                    floats.append([-math.inf] * keys)
                    continue
                bools.append([next(noise) < 0.7 for _ in range(keys)])
                floats.append(
                    [
                        quantize(choices[int(next(noise) * len(choices))])
                        for _ in range(keys)
                    ]
                )
            if mask_layout == "broadcast" and shared is None:
                shared = (bools, floats)
            self.bool_mask.append(bools)
            self.float_mask.append(floats)
        # Exact scores without masks and the sums of their absolute products.
        self.dots, self.abs_dots = [], []
        for row in range(self.rows):
            kv = self._kv_head(row // queries)
            query = self.q[row]
            self.dots.append(
                [
                    self.scale * math.fsum(map(operator.mul, query, key))
                    for key in self.k[kv]
                ]
            )
            self.abs_dots.append(
                [
                    abs(self.scale)
                    * math.fsum(abs(item) for item in map(operator.mul, query, key))
                    for key in self.k[kv]
                ]
            )

    def _kv_head(self, bh):
        return bh // self.gqa

    def kv_strides(self, dim):
        """Return (head stride, sequence stride, buffer size) as MLX binds them."""
        if self.kv_layout == "contiguous":
            return self.keys * dim, dim, self.batch * self.kv_heads * self.keys * dim
        if self.kv_layout == "cache":
            capacity = self.keys + CACHE_SLACK
            size = self.batch * self.kv_heads * capacity * dim
            return capacity * dim, dim, size
        assert self.kv_layout == "transposed" and self.batch == 1
        return dim, self.kv_heads * dim, self.keys * self.kv_heads * dim

    def mask_strides(self):
        """Return (kv, query, head) strides and the size of the mask buffer."""
        keys, queries = self.keys, self.queries
        q_stride = keys if queries > 1 else 0
        if self.mask_layout == "full":
            return 1, q_stride, queries * keys, self.batch * self.heads * queries * keys
        if self.mask_layout == "broadcast":
            return 1, q_stride, 0, queries * keys
        assert self.mask_layout == "transposed"
        assert self.batch == 1 or self.heads == 1
        return (
            queries,
            1 if queries > 1 else 0,
            keys * queries,
            self.batch * self.heads * keys * queries,
        )

    def mask_index(self, bh, query, key):
        kv, q_stride, head = self.mask_strides()[:3]
        return bh * head + key * kv + query * q_stride


@functools.lru_cache(maxsize=None)
def _vector_case(dtype, qk_dim, value_dim, name):
    return _VectorCase(dtype, qk_dim, value_dim, name)


def _vector_data(entry, name):
    return _vector_case(_dtype(entry), *_dims(entry), name)


def _uses_key(constants, data, bh, query, key):
    if constants["do_causal"]:
        return key <= data.keys - data.queries + query
    if constants["bool_mask"]:
        return data.bool_mask[bh][query][key]
    if constants["float_mask"]:
        return data.float_mask[bh][query][key] >= FINITE_MIN[data.dtype]
    return True


def _vector_reference(entry, constants, name):
    """Return the float64 attention outputs and each output's error bound."""
    data = _vector_data(entry, name)
    qk_per_thread = data.qk_dim // SIMD_SIZE
    expected, bounds = [], []
    for row in range(data.rows):
        bh, query = divmod(row, data.queries)
        kv = data._kv_head(bh)
        used = [
            key
            for key in range(data.keys)
            if _uses_key(constants, data, bh, query, key)
        ]
        sink = data.sinks[bh % data.heads] if constants["has_sinks"] else None
        if not used:
            # No key: the accumulators stay zero, and so do the outputs.
            expected += [0.0] * data.value_dim
            bounds += [0.0] * data.value_dim
            continue
        scores, errors = {}, {}
        for key in used:
            score = data.dots[row][key]
            error = (
                (1 + OP_ERROR) * _gamma(qk_per_thread + 34) * data.abs_dots[row][key]
            )
            if constants["float_mask"]:
                score += data.float_mask[bh][query][key]
                error += (1 + OP_ERROR) * OP_ERROR * abs(score)
            scores[key], errors[key] = score, error
        largest_error = max(errors.values())
        top = max(scores.values())
        if sink is not None:
            top = max(top, sink)
        counts = [0] * SIMD_SIZE
        for key in used:
            counts[key % SIMD_SIZE] += 1
        later = {}
        seen = [0] * SIMD_SIZE
        for key in used:
            group = key % SIMD_SIZE
            seen[group] += 1
            later[key] = counts[group] - seen[group]
        weights = [0.0] * data.keys
        numerator_error = [0.0] * data.keys
        denominator_error = 0.0
        for key in used:
            distance = top - scores[key]
            weight = math.exp(-distance)
            if not weight:
                continue
            # Each link's exponential adds at most 4 + 4|x| units, and the
            # arguments along one key's links sum to its distance from the
            # final maximum, up to the computed scores' errors.
            links = later[key] + 2
            exponentials = OP_ERROR * (
                EXP_UNITS * links + 4.0 * (distance + 2.0 * largest_error)
            )
            relative = math.expm1(errors[key] + exponentials)
            group = counts[key % SIMD_SIZE]
            weights[key] = weight
            numerator_error[key] = (
                (1 + relative) * (1 + OP_ERROR) ** (3 * group + 34) - 1
            ) * weight
            denominator_error += (
                (1 + relative) * (1 + OP_ERROR) ** (2 * group + 34) - 1
            ) * weight
        denominator = math.fsum(weights)
        if sink is not None:
            distance = top - sink
            links = counts[0] + 1
            relative = math.expm1(
                OP_ERROR * (EXP_UNITS * links + 4.0 * (distance + 2.0 * largest_error))
            )
            weight = math.exp(-distance)
            denominator += weight
            denominator_error += (
                (1 + relative) * (1 + OP_ERROR) ** (2 * counts[0] + 34) - 1
            ) * weight
        slack = TINY * (4 * len(used) + 64) / denominator
        for column in data.columns[kv]:
            numerator = math.fsum(map(operator.mul, weights, column))
            spread = math.fsum(map(operator.mul, numerator_error, map(abs, column)))
            value = numerator / denominator
            bound = (spread + abs(value) * denominator_error) / (
                denominator - denominator_error
            )
            bound = bound * (1 + DIVISION_ERROR) + DIVISION_ERROR * abs(value)
            expected.append(value)
            bounds.append((bound + slack) * BOUND_SAFETY)
    return expected, bounds


def _vector_buffers(entry, constants, name):
    """Return the buffers and scalars sdpa_vector() binds for one case.

    Buffers that a specialization disables hold decoys that would change
    every result: an all-false boolean mask, an all -inf float mask and sinks
    far above every score.
    """
    data = _vector_data(entry, name)
    dtype, qk_dim, value_dim = data.dtype, data.qk_dim, data.value_dim
    bh_count = data.batch * data.heads
    if constants["query_transposed"]:
        assert data.queries > 1 and (data.batch == 1 or data.heads == 1)
    queries = [math.nan] * (data.rows * qk_dim)
    for row in range(data.rows):
        bh, query = divmod(row, data.queries)
        base = (
            query * bh_count + bh if constants["query_transposed"] else row
        ) * qk_dim
        queries[base : base + qk_dim] = data.q[row]

    def kv_buffer(tensor, dim):
        head_stride, seq_stride, size = data.kv_strides(dim)
        buffer = [math.nan] * size
        for kv, head in enumerate(tensor):
            for key, row in enumerate(head):
                base = kv * head_stride + key * seq_stride
                buffer[base : base + dim] = row
        return buffer, head_stride, seq_stride

    keys, k_head_stride, k_seq_stride = kv_buffer(data.k, qk_dim)
    values, v_head_stride, v_seq_stride = kv_buffer(data.v, value_dim)
    kv_stride, q_stride, head_stride, mask_size = data.mask_strides()
    bool_mask = [False] * mask_size
    float_mask = [0.0] * mask_size
    for bh in range(bh_count):
        for query in range(data.queries):
            for key in range(data.keys):
                index = data.mask_index(bh, query, key)
                bool_mask[index] = data.bool_mask[bh][query][key]
                float_mask[index] = data.float_mask[bh][query][key]
    if not constants["bool_mask"]:
        bool_mask = [False] * INPUT_GUARD_COUNT
    if not constants["float_mask"]:
        float_mask = [-math.inf] * INPUT_GUARD_COUNT
    if not constants["has_mask"]:
        kv_stride = q_stride = head_stride = 0
    sinks = list(data.sinks) if constants["has_sinks"] else [20.0] * data.heads
    return {
        "queries": queries + [math.nan] * INPUT_GUARD_COUNT,
        "keys": keys + [math.nan] * INPUT_GUARD_COUNT,
        "values": values + [math.nan] * INPUT_GUARD_COUNT,
        "out": [_from_bits(dtype, GUARD_BITS[dtype])] * (data.rows * value_dim),
        "gqa_factor": data.gqa,
        "N": data.keys,
        "k_head_stride": k_head_stride,
        "k_seq_stride": k_seq_stride,
        "v_head_stride": v_head_stride,
        "v_seq_stride": v_seq_stride,
        "scale": data.scale,
        "bmask": bool_mask,
        "fmask": float_mask + [math.nan] * INPUT_GUARD_COUNT,
        "mask_kv_seq_stride": kv_stride,
        "mask_q_seq_stride": q_stride,
        "mask_head_stride": head_stride,
        "sinks": (
            [_quantize(dtype, item) for item in sinks] + [math.nan] * INPUT_GUARD_COUNT
        ),
        "num_q_heads": data.heads,
        "grid": (bh_count, data.queries, 1),
    }


def _vector_simulate(entry, constants, name, mutation=None):
    """Run sdpa_vector's index arithmetic on the bound buffers in float64.

    ``mutation`` injects one defect; results round to the storage type.
    """
    data = _vector_data(entry, name)
    dtype, qk_dim, value_dim = data.dtype, data.qk_dim, data.value_dim
    buffers = _vector_buffers(entry, constants, name)
    grid = buffers["grid"]
    transposed = constants["query_transposed"] != (mutation == "transpose")
    scale = 1.0 if mutation == "scale" else buffers["scale"]
    keys = data.keys - 1 if mutation == "last-key" else data.keys
    k_head_stride = buffers["k_head_stride"]
    k_seq_stride = buffers["k_seq_stride"]
    if mutation == "head-stride":
        k_head_stride = data.keys * qk_dim
    if mutation == "kv-stride":
        k_seq_stride = qk_dim
    mask_head_stride = 0 if mutation == "mask-head" else buffers["mask_head_stride"]
    output = list(buffers["out"])
    for bh in range(grid[0]):
        for query in range(grid[1]):
            kv = bh if mutation == "gqa" else bh // buffers["gqa_factor"]
            o_offset = bh * grid[1] + query
            q_offset = grid[0] * query + bh if transposed else o_offset
            q = [
                scale * _read(buffers["queries"], q_offset * qk_dim + index)
                for index in range(qk_dim)
            ]
            states = []
            for group in range(SIMD_SIZE):
                top, total = -FLT_MAX, 0.0
                accumulator = [0.0] * value_dim
                if constants["has_sinks"] and group == 0 and mutation != "sinks":
                    head = (
                        bh if mutation == "sink-head" else bh % buffers["num_q_heads"]
                    )
                    top, total = _read(buffers["sinks"], head), 1.0
                for key in range(group, keys, SIMD_SIZE):
                    mask = (
                        bh * mask_head_stride
                        + key * buffers["mask_kv_seq_stride"]
                        + query * buffers["mask_q_seq_stride"]
                    )
                    use = True
                    if constants["do_causal"]:
                        limit = data.keys - grid[1] + query
                        use = key < limit if mutation == "causal" else key <= limit
                    elif constants["bool_mask"]:
                        use = bool(_read(buffers["bmask"], mask))
                        use = use != (mutation == "mask-polarity")
                    elif constants["float_mask"]:
                        use = _read(buffers["fmask"], mask) >= FINITE_MIN[dtype]
                    if not use:
                        continue
                    base = kv * k_head_stride + key * k_seq_stride
                    score = math.fsum(
                        item * _read(buffers["keys"], base + index)
                        for index, item in enumerate(q)
                    )
                    if mutation == "binary16-score":
                        score = struct.unpack("<e", struct.pack("<e", score))[0]
                    if constants["float_mask"] and mutation != "mask-add":
                        score += _read(buffers["fmask"], mask)
                    new_top = max(top, score)
                    factor = math.exp(top - new_top)
                    weight = math.exp(score - new_top)
                    top = new_top
                    total = total * factor + weight
                    base = kv * buffers["v_head_stride"] + key * buffers["v_seq_stride"]
                    accumulator = [
                        item * factor + weight * _read(buffers["values"], base + index)
                        for index, item in enumerate(accumulator)
                    ]
                states.append((top, total, accumulator))
            top = max(state[0] for state in states)
            factors = [math.exp(state[0] - top) for state in states]
            total = math.fsum(
                state[1] * factor for state, factor in zip(states, factors)
            )
            for index in range(value_dim):
                value = math.fsum(
                    state[2][index] * factor for state, factor in zip(states, factors)
                )
                if total != 0 and mutation != "normalize":
                    value /= total
                output[o_offset * value_dim + index] = _quantize(dtype, value)
    return output


class _Pass2Case:
    """Synthetic first-pass partials of one sdpa_vector_2pass_2 case."""

    def __init__(self, dtype, value_dim, name):
        heads, queries, blocks, pattern = PASS2_CASES[name]
        self.dtype, self.value_dim = dtype, value_dim
        self.heads, self.queries, self.blocks = heads, queries, blocks
        self.rows = heads * queries
        noise = _stream(f"pass2:{name}", f"{dtype}:{value_dim}")
        span = 20.0 if pattern == "wide" else 4.0
        self.maxs, self.sums, self.partials = [], [], []
        for row in range(self.rows):
            maxs, sums, partials = [], [], []
            dominant = int(next(noise) * blocks)
            for block in range(blocks):
                empty = pattern == "sparse" and (row == 1 or next(noise) < 0.3)
                if empty:
                    maxs.append(-FLT_MAX)
                    sums.append(0.0)
                    partials.append([0.0] * value_dim)
                    continue
                top = _float32(span * (2.0 * next(noise) - 1.0))
                if pattern == "sparse" and block == dominant:
                    top = _float32(top + 25.0)
                total = _float32(1.0 + 7.0 * next(noise))
                maxs.append(top)
                sums.append(total)
                partials.append(
                    [
                        _quantize(dtype, total * (2.0 * next(noise) - 1.0))
                        for _ in range(value_dim)
                    ]
                )
            self.maxs.append(maxs)
            self.sums.append(sums)
            self.partials.append(partials)


@functools.lru_cache(maxsize=None)
def _pass2_case(dtype, value_dim, name):
    return _Pass2Case(dtype, value_dim, name)


def _pass2_data(entry, name):
    return _pass2_case(_dtype(entry), _dims(entry)[1], name)


def _pass2_reference(entry, name):
    data = _pass2_data(entry, name)
    chunks = data.blocks // SIMD_SIZE
    expected, bounds = [], []
    for row in range(data.rows):
        maxs, sums, partials = data.maxs[row], data.sums[row], data.partials[row]
        if not any(sums):
            expected += [0.0] * data.value_dim
            bounds += [0.0] * data.value_dim
            continue
        top = max(maxs)
        weights, relative = [], []
        for item in maxs:
            distance = top - item
            weight = math.exp(-distance)
            weights.append(weight)
            # Blocks without keys hold zero sums and partials.
            relative.append(
                math.expm1(OP_ERROR * _exp_units(distance)) if weight else 0.0
            )
        growth = (1 + OP_ERROR) ** (chunks + 33)
        denominator = math.fsum(w * s for w, s in zip(weights, sums))
        denominator_error = math.fsum(
            ((1 + r) * growth - 1) * w * s for w, r, s in zip(weights, relative, sums)
        )
        coefficients = [((1 + r) * growth - 1) * w for w, r in zip(weights, relative)]
        slack = TINY * (4 * data.blocks + 64) / denominator
        for index in range(data.value_dim):
            column = [block[index] for block in partials]
            value = math.fsum(map(operator.mul, weights, column)) / denominator
            spread = math.fsum(map(operator.mul, coefficients, map(abs, column)))
            bound = (spread + abs(value) * denominator_error) / (
                denominator - denominator_error
            )
            bound = bound * (1 + DIVISION_ERROR) + DIVISION_ERROR * abs(value)
            expected.append(value)
            bounds.append((bound + slack) * BOUND_SAFETY)
    return expected, bounds


def _pass2_buffers(entry, name):
    data = _pass2_data(entry, name)
    partials = [item for row in data.partials for block in row for item in block] + [
        math.nan
    ] * INPUT_GUARD_COUNT
    return {
        "partials": partials,
        "sums": (
            [item for row in data.sums for item in row] + [math.nan] * INPUT_GUARD_COUNT
        ),
        "maxs": (
            [item for row in data.maxs for item in row] + [math.nan] * INPUT_GUARD_COUNT
        ),
        "out": (
            [_from_bits(data.dtype, GUARD_BITS[data.dtype])]
            * (data.rows * data.value_dim)
        ),
        "blocks": data.blocks,
        "grid": (data.heads, data.queries, 1),
    }


def _pass2_simulate(entry, name, mutation=None):
    """Run sdpa_vector_2pass_2's index arithmetic in float64."""
    data = _pass2_data(entry, name)
    buffers = _pass2_buffers(entry, name)
    dim, blocks = data.value_dim, data.blocks
    per_lane = dim // SIMD_SIZE
    chunks = blocks // SIMD_SIZE - (1 if mutation == "chunk" else 0)
    grid = buffers["grid"]
    output = list(buffers["out"])
    maxs, sums, partials = buffers["maxs"], buffers["sums"], buffers["partials"]
    for head in range(grid[0]):
        for query in range(grid[1]):
            offset = head * grid[1] + query
            row = offset * blocks
            partial_row = offset * (dim if mutation == "stride" else blocks * dim)
            top = -FLT_MAX
            for lane in range(SIMD_SIZE):
                for chunk in range(chunks):
                    top = max(top, _read(maxs, row + lane + SIMD_SIZE * chunk))
            total = math.fsum(
                math.exp(_read(maxs, row + lane + SIMD_SIZE * chunk) - top)
                * (
                    1.0
                    if mutation == "sums"
                    else _read(sums, row + lane + SIMD_SIZE * chunk)
                )
                for lane in range(SIMD_SIZE)
                for chunk in range(chunks)
            )
            for index in range(dim):
                lane = index // per_lane
                value = 0.0
                for group in range(SIMD_SIZE):
                    for chunk in range(chunks):
                        block = group + SIMD_SIZE * chunk
                        source = lane if mutation == "max-lane" else group
                        factor = math.exp(
                            _read(maxs, row + source + SIMD_SIZE * chunk) - top
                        )
                        value += factor * _read(
                            partials, partial_row + block * dim + index
                        )
                if total != 0 and mutation != "normalize":
                    value /= total
                output[offset * dim + index] = _quantize(data.dtype, value)
    return output


class _ScaleCopyCase:
    def __init__(self, dtype, name):
        batch, heads, length, q_offset, rows, layout, scale = SCALE_COPY_CASES[name]
        self.dtype = dtype
        self.batch, self.heads, self.length = batch, heads, length
        self.q_offset, self.rows, self.layout = q_offset, rows, layout
        self.dim = BLOCKED_HEAD_DIM
        self.scale = _float32(self.dim**-0.5 if scale is None else scale)
        dim = self.dim
        if layout == "contiguous":
            self.strides = (heads * length * dim, length * dim, dim, 1)
        elif layout == "sequence-major":
            self.strides = (length * heads * dim, dim, heads * dim, 1)
        else:
            assert layout == "head-major"
            self.strides = (length * dim, batch * length * dim, dim, 1)
        size = batch * heads * length * dim
        noise = _stream(f"scale-copy:{name}", dtype)
        # Signed zeros and magnitudes from 2^-6 to 4: every product stays in
        # the normal range, so flushing subnormals cannot change a result.
        self.logical = []
        for _ in range(size):
            pick = next(noise)
            if pick < 0.02:
                value = 0.0
            elif pick < 0.04:
                value = -0.0
            else:
                magnitude = 2.0 ** (-6.0 + 8.0 * next(noise))
                value = magnitude if next(noise) < 0.5 else -magnitude
            self.logical.append(_quantize(dtype, value))
        self.q = [math.nan] * size
        for b in range(batch):
            for h in range(heads):
                for t in range(length):
                    for d in range(dim):
                        self.q[self.address(b, h, t, d)] = self.value(b, h, t, d)

    def value(self, b, h, t, d):
        """Return logical element [b, h, t, d] of the [B, H, L, D] queries."""
        return self.logical[((b * self.heads + h) * self.length + t) * self.dim + d]

    def address(self, b, h, t, d):
        s = self.strides
        return b * s[0] + h * s[1] + t * s[2] + d * s[3]


@functools.lru_cache(maxsize=None)
def _scale_copy_case(dtype, name):
    return _ScaleCopyCase(dtype, name)


def _scale_copy_reference(entry, name):
    """Return each q_block element: one rounding of an exact 16-bit product."""
    data = _scale_copy_case(_dtype(entry), name)
    scale = _quantize(data.dtype, data.scale)
    return [
        _quantize(data.dtype, data.value(b, h, data.q_offset + row, d) * scale)
        for b in range(data.batch)
        for h in range(data.heads)
        for row in range(data.rows)
        for d in range(data.dim)
    ]


def _scale_copy_simulate(entry, name, mutation=None):
    """Run sdpa_blocked_scale_copy's index arithmetic on the bound buffers.

    ``mutation`` injects one defect.
    """
    data = _scale_copy_case(_dtype(entry), name)
    buffers = _scale_copy_buffers(entry, name)
    dtype, heads, rows, dim = data.dtype, data.heads, data.rows, data.dim
    scale = buffers["scale"]
    if mutation != "scale-rounding":
        scale = _quantize(dtype, scale)
    q_offset = 0 if mutation == "offset" else buffers["q_offset"]
    strides = buffers["q_strides"]
    if mutation == "strides":
        strides = [heads * data.length * dim, data.length * dim, dim, 1]
    output = list(buffers["q_block"])
    width, height, depth = buffers["grid"]
    for z in range(depth):
        head, batch = z % heads, z // heads
        if mutation == "head-batch":
            head, batch = z // data.batch, z % data.batch
        for row in range(height):
            for d in range(width):
                q_index = (
                    batch * strides[0]
                    + head * strides[1]
                    + (q_offset + row) * strides[2]
                    + d
                )
                product = _read(buffers["q"], q_index) * scale
                output[(z * rows + row) * dim + d] = _quantize(dtype, product)
    return output


def _scale_copy_buffers(entry, name):
    data = _scale_copy_case(_dtype(entry), name)
    return {
        "q": data.q + [math.nan] * INPUT_GUARD_COUNT,
        "q_block": (
            [_from_bits(data.dtype, GUARD_BITS[data.dtype])]
            * (data.batch * data.heads * data.rows * data.dim)
        ),
        "scale": data.scale,
        "q_strides": list(data.strides),
        "heads": data.heads,
        "rows": data.rows,
        "head_dim": data.dim,
        "q_offset": data.q_offset,
        "grid": (data.dim, data.rows, data.batch * data.heads),
    }


def _causal_mask_initial(dtype, name):
    total_rows, _, columns = CAUSAL_MASK_CASES[name]
    noise = _stream(f"causal-mask:{name}", dtype)
    return [
        _quantize(dtype, 8.0 * next(noise) - 4.0) for _ in range(total_rows * columns)
    ]


def _causal_mask_reference(entry, name):
    """Mask the strict upper triangle of each score row's trailing tile."""
    dtype = _dtype(entry)
    total_rows, tile_rows, columns = CAUSAL_MASK_CASES[name]
    scores = _causal_mask_initial(dtype, name)
    for score_row in range(total_rows):
        for column in range(tile_rows):
            if column > score_row % tile_rows:
                scores[score_row * columns + columns - tile_rows + column] = FINITE_MIN[
                    dtype
                ]
    return scores


def _causal_mask_simulate(entry, name, mutation=None):
    """Run sdpa_blocked_causal_mask over its grid; ``mutation`` adds a defect."""
    dtype = _dtype(entry)
    buffers = _causal_mask_buffers(entry, name)
    scores = list(buffers["scores"])
    tile_rows, columns = buffers["tile_rows"], buffers["columns"]
    width, height, _ = buffers["grid"]
    for score_row in range(height):
        row = score_row if mutation == "row" else score_row % tile_rows
        for column in range(width):
            masked = column >= row if mutation == "diagonal" else column > row
            if not masked:
                continue
            index = score_row * columns + column
            if mutation != "alignment":
                index += columns - tile_rows
            scores[index] = FINITE_MIN[dtype]
    return scores


def _causal_mask_buffers(entry, name):
    total_rows, tile_rows, columns = CAUSAL_MASK_CASES[name]
    return {
        "scores": _causal_mask_initial(_dtype(entry), name),
        "tile_rows": tile_rows,
        "columns": columns,
        "grid": (tile_rows, total_rows, 1),
    }


def _buffers(entry, constants, name):
    kind = _kind(entry)
    if kind == "vector":
        return _vector_buffers(entry, constants, name)
    if kind == "pass2":
        return _pass2_buffers(entry, name)
    if kind == "scale-copy":
        return _scale_copy_buffers(entry, name)
    return _causal_mask_buffers(entry, name)


def _reference(entry, constants, name):
    """Return the expected outputs and bounds; None bounds require exact bits."""
    kind = _kind(entry)
    if kind == "vector":
        return _vector_reference(entry, constants, name)
    if kind == "pass2":
        return _pass2_reference(entry, name)
    if kind == "scale-copy":
        expected = _scale_copy_reference(entry, name)
    else:
        expected = _causal_mask_reference(entry, name)
    return expected, [None] * len(expected)


def _simulate(entry, constants, name, mutation=None):
    kind = _kind(entry)
    if kind == "vector":
        return _vector_simulate(entry, constants, name, mutation)
    if kind == "pass2":
        return _pass2_simulate(entry, name, mutation)
    if kind == "scale-copy":
        return _scale_copy_simulate(entry, name, mutation)
    return _causal_mask_simulate(entry, name, mutation)


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
            if math.isnan(got) or _bits(dtype, got) != _bits(dtype, want):
                bad.append((index, got, want, "exact"))
            continue
        low, high = _round(dtype, want - bound), _round(dtype, want + bound)
        if math.isnan(got) or not low <= got <= high:
            bad.append((index, got, want, low, high))
        elif got != want and bound:
            half = max(_ulp(dtype, want), _ulp(dtype, got)) / 2
            excess = max(0.0, abs(got - want) - half)
            if excess:
                worst = max(worst, excess / bound)
    assert not bad, bad[:8]
    return worst


def _mutations(entry, constants, name):
    """Return the defects that change ``entry``'s results on case ``name``."""
    kind = _kind(entry)
    if kind == "scale-copy":
        data = _scale_copy_case(_dtype(entry), name)
        mutations = []
        if _quantize(data.dtype, data.scale) != data.scale:
            mutations.append("scale-rounding")
        if data.q_offset:
            mutations.append("offset")
        if data.layout != "contiguous":
            mutations.append("strides")
        if data.batch > 1 and data.heads > 1:
            mutations.append("head-batch")
        return mutations
    if kind == "causal-mask":
        total_rows, tile_rows, columns = CAUSAL_MASK_CASES[name]
        mutations = ["diagonal"]
        if total_rows > tile_rows:
            mutations.append("row")
        if columns > tile_rows:
            mutations.append("alignment")
        return mutations
    if kind == "pass2":
        return ["normalize", "chunk", "max-lane", "stride", "sums"]
    data = _vector_data(entry, name)
    weights = _vector_weights(entry, constants, name)
    mutations = ["normalize"]
    # Dropping a key or the sink only matters where it carries weight.
    if any(row.get(data.keys - 1, 0.0) > 2.0**-5 for row, _ in weights):
        mutations.append("last-key")
    if data.scale != 1.0:
        mutations.append("scale")
    if data.gqa > 1:
        mutations.append("gqa")
    if data.queries > 1 and data.batch * data.heads > 1:
        mutations.append("transpose")
    if data.kv_layout != "contiguous":
        mutations.append("head-stride")
    if data.kv_layout == "transposed":
        mutations.append("kv-stride")
    if constants["do_causal"]:
        mutations.append("causal")
    if constants["bool_mask"]:
        mutations.append("mask-polarity")
    if constants["has_mask"] and data.mask_strides()[2]:
        mutations.append("mask-head")
    if constants["float_mask"]:
        mutations.append("mask-add")
    if constants["has_sinks"]:
        if any(row and (sink or 0.0) > 2.0**-5 for row, sink in weights):
            mutations.append("sinks")
        if data.batch > 1:
            mutations.append("sink-head")
    # Unscaled scores are large enough that binary16 rounding of them stays
    # within their own binary32 bound.
    if data.dtype == "float32" and data.scale != 1.0:
        mutations.append("binary16-score")
    return mutations


def _vector_weights(entry, constants, name):
    """Return each row's exact softmax weight per used key and of the sink."""
    data = _vector_data(entry, name)
    rows = []
    for row in range(data.rows):
        bh, query = divmod(row, data.queries)
        scores = {}
        for key in range(data.keys):
            if _uses_key(constants, data, bh, query, key):
                score = data.dots[row][key]
                if constants["float_mask"]:
                    score += data.float_mask[bh][query][key]
                scores[key] = score
        sink = data.sinks[bh % data.heads] if constants["has_sinks"] else None
        if not scores:
            rows.append(({}, None))
            continue
        top = max([*scores.values(), *([] if sink is None else [sink])])
        weights = {key: math.exp(score - top) for key, score in scores.items()}
        total = math.fsum(weights.values())
        sink_weight = None if sink is None else math.exp(sink - top)
        total += sink_weight or 0.0
        rows.append(
            (
                {key: weight / total for key, weight in weights.items()},
                None if sink_weight is None else sink_weight / total,
            )
        )
    return rows


def _config(output, target, entry, constants):
    kind = _kind(entry)
    options = ""
    if constants:
        options += "\n[project.specialization_constants]\n"
        for name, value in constants.items():
            options += f'"{CONSTANT_IDS[name]}" = {str(value).lower()}\n'
    if target in SOFTWARE_SUBGROUP_TARGETS and kind in SOFTWARE_SUBGROUP_KINDS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
        if target == "directx":
            options += 'relative_wave_shuffle_out_of_range = "self"\n'
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS[kind]:
            options += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = {SIGNED_MAX}
"""
    size = ", ".join(str(item) for item in WORKGROUP_SIZES[kind])
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
[project.source_options.metal]
max_template_specializations = 256
max_template_materialization_work = 16384
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_contract():
    vector = {
        name: {
            "batch": batch,
            "heads": heads,
            "kvHeads": kv_heads,
            "queryLength": queries,
            "keyLength": keys,
            "kvLayout": kv_layout,
            "maskLayout": mask_layout,
            "scale": scale,
            "values": pattern,
            "maskedRows": [list(row) for row in MASKED_ROWS.get(name, ())],
        }
        for name, (
            batch,
            heads,
            kv_heads,
            queries,
            keys,
            kv_layout,
            mask_layout,
            scale,
            pattern,
        ) in VECTOR_CASES.items()
    }
    return {
        "vector": vector,
        "vectorCaseSelection": {
            "queryTransposedOrCausal": ["batches", "prefill", "rising"],
            "other": ["decode", "short", "batches"],
        },
        "cacheSlack": CACHE_SLACK,
        "pass2": {
            name: {
                "rows": rows,
                "queryLength": queries,
                "blocks": blocks,
                "values": pattern,
            }
            for name, (rows, queries, blocks, pattern) in PASS2_CASES.items()
        },
        "scaleCopy": {
            name: {
                "batch": batch,
                "heads": heads,
                "queryLength": length,
                "qOffset": q_offset,
                "rows": rows,
                "layout": layout,
                "scale": scale,
                "headDim": BLOCKED_HEAD_DIM,
            }
            for name, (
                batch,
                heads,
                length,
                q_offset,
                rows,
                layout,
                scale,
            ) in SCALE_COPY_CASES.items()
        },
        "causalMask": {
            name: {"scoreRows": total, "tileRows": tile, "columns": columns}
            for name, (total, tile, columns) in CAUSAL_MASK_CASES.items()
        },
    }


def _expected_validation_contract():
    return {
        "workgroupSizes": {kind: list(size) for kind, size in WORKGROUP_SIZES.items()},
        "simdSize": SIMD_SIZE,
        "softwareSubgroups": {
            target: list(SOFTWARE_SUBGROUP_KINDS)
            for target in SOFTWARE_SUBGROUP_TARGETS
        },
        "specializationConstants": {
            name: {"id": identifier, "values": [True, False]}
            for name, identifier in CONSTANT_IDS.items()
        },
        "vectorSpecializations": [
            _variant_key(_vector_constants(*item)) for item in VECTOR_SPECIALIZATIONS
        ],
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
            "expUnits": "max(4, 3 + 2|x|) + 2|x|",
            "scoreError": "gamma(D / 32 + 34) * scale * sum|q * k|",
            "accumulatorRoundings": {"numerator": "3n + 34", "denominator": "2n + 34"},
            "tiny": TINY,
            "boundSafety": BOUND_SAFETY,
        },
        "fastMathLowering": "metal-keeps-fast-exp-others-use-default-exp",
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "blockedResults": "bit-exact",
        "keylessRows": "exact-zero",
        "disabledBufferDecoys": {
            "bmask": "all-false",
            "fmask": "all-negative-infinity",
            "sinks": 20.0,
        },
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _output_count(entry, name):
    kind = _kind(entry)
    if kind == "vector":
        data = _vector_data(entry, name)
        return data.rows * data.value_dim
    if kind == "pass2":
        data = _pass2_data(entry, name)
        return data.rows * data.value_dim
    if kind == "scale-copy":
        data = _scale_copy_case(_dtype(entry), name)
        return data.batch * data.heads * data.rows * data.dim
    total_rows, _, columns = CAUSAL_MASK_CASES[name]
    return total_rows * columns


def _dispatch_count():
    return sum(
        len(_variant_cases(entry, constants)) for entry, _, constants in VARIANTS
    )


def _checked_elements(entry):
    return sum(
        _output_count(entry, name)
        for item, _, constants in VARIANTS
        if item == entry
        for name in _variant_cases(entry, constants)
    )


def _expected_scope():
    return {
        "instantiatedEntryCount": len(INSTANTIATED),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "pendingEntryCount": len(PENDING_ENTRIES),
        "artifactCountByTarget": len(VARIANTS),
        "dispatchCountByTarget": _dispatch_count(),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_sdpa_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-sdpa-contract"
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
    pending = contract["pendingEntries"]
    assert tuple(pending) == PENDING_ENTRIES
    for entry, record in pending.items():
        assert record == {
            "kind": _kind(entry),
            "reason": PENDING_KINDS[_kind(entry)],
            "translatesTo": ["metal"],
            "diagnostics": {
                "opengl": "project.translate.opengl-software-subgroup-invalid",
                "directx": "project.translate.directx-software-subgroup-invalid",
            },
        }
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


def test_sdpa_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "scaled_dot_product_attention.current-tree.translation.json" in readme
    assert len(INSTANTIATED) == len(set(INSTANTIATED)) == 82
    assert INSTANTIATED[:4] == BLOCKED_ENTRIES
    assert INSTANTIATED[4:8] == (
        "sdpa_vector_float_64_64",
        "sdpa_vector_2pass_1_float_64_64",
        "sdpa_vector_float_96_64",
        "sdpa_vector_2pass_1_float_96_64",
    )
    assert len(ENTRIES) == 46 and len(PENDING_ENTRIES) == 36
    counts = {}
    for entry in ENTRIES:
        counts[_kind(entry)] = counts.get(_kind(entry), 0) + 1
    assert counts == {"scale-copy": 2, "causal-mask": 2, "vector": 24, "pass2": 18}
    assert {_kind(entry) for entry in PENDING_ENTRIES} == set(PENDING_KINDS)
    assert {_dtype(entry) for entry in ENTRIES} == set(TYPE_NAMES.values())
    assert {_dtype(entry) for entry in BLOCKED_ENTRIES} == {"float16", "bfloat16"}
    assert _dims("sdpa_vector_bfloat16_t_192_128") == (192, 128)
    assert _dims("sdpa_vector_2pass_2_float16_t_96") == (96, 96)
    assert _dtype("sdpa_vector_2pass_1_gqa_12_float16_t_128_128") == "float16"
    # Eight specializations per vector kernel and one per other kernel.
    assert len(VARIANTS) == 24 * 8 + 4 + 18
    assert len({(entry, key) for entry, key, _ in VARIANTS}) == len(VARIANTS)
    keys = {key for entry, key, _ in VARIANTS if _kind(entry) == "vector"}
    assert len(keys) == 8
    pairs = set()
    for mask, transposed, sinks in VECTOR_SPECIALIZATIONS:
        pairs |= {("mask", mask, "qt", transposed), ("mask", mask, "sinks", sinks)}
        pairs.add(("qt", transposed, "sinks", sinks))
    assert len(pairs) == 4 * 2 + 4 * 2 + 2 * 2


def test_sdpa_cases_follow_mlx_dispatch():
    for entry, _, constants in VARIANTS:
        assert len(_variant_cases(entry, constants)) == 3, (entry, constants)
    assert _dispatch_count() == 3 * len(VARIANTS)
    for name, case in VECTOR_CASES.items():
        batch, heads, kv_heads, queries, keys, kv_layout, mask_layout = case[:7]
        # sdpa_vector: at most eight queries, fewer than 1024 keys (the
        # two-pass threshold on large devices), whole query-head groups.
        assert 1 <= queries <= 8 and 1 < keys < 1024
        assert heads % kv_heads == 0
        if kv_layout == "transposed":
            assert batch == 1 or kv_heads == 1  # kv_copy_unless keeps it
        if mask_layout == "transposed":
            assert batch == 1 or heads == 1  # mask_copy_unless keeps it
        for bh, query in MASKED_ROWS.get(name, ()):
            assert bh < batch * heads and query < queries
            assert mask_layout != "broadcast"
    for name in ("batches", "prefill", "rising"):
        batch, heads, _, queries = VECTOR_CASES[name][:4]
        # MLX transposes queries only for B == 1 or H == 1, and only applies
        # the causal mask to more than one query.
        assert queries > 1 and (batch == 1 or heads == 1)
    assert {case[5] for case in VECTOR_CASES.values()} == {
        "contiguous",
        "cache",
        "transposed",
    }
    assert {case[6] for case in VECTOR_CASES.values()} == {
        "full",
        "broadcast",
        "transposed",
    }
    assert {case[3] for case in VECTOR_CASES.values()} >= {1, 2, 3, 4}
    # Fewer keys than SIMD groups, partial and whole rounds of 32 keys, GQA
    # and multi-query attention all appear.
    assert {case[4] % SIMD_SIZE == 0 for case in VECTOR_CASES.values()} == {
        True,
        False,
    }
    assert min(case[4] for case in VECTOR_CASES.values()) < SIMD_SIZE
    assert {case[1] // case[2] for case in VECTOR_CASES.values()} >= {1, 2}
    for rows, queries, blocks, _ in PASS2_CASES.values():
        # The aggregation reads partials in SIMD-width chunks; MLX picks 32
        # to 1024 blocks and rounds MLX_SDPA_BLOCKS up to a multiple of 32.
        assert blocks % SIMD_SIZE == 0 and 32 <= blocks <= 1024
        assert rows * queries > 1
    assert {case[2] // SIMD_SIZE for case in PASS2_CASES.values()} == {1, 2, 4}
    for case in SCALE_COPY_CASES.values():
        batch, heads, length, q_offset, rows, layout, _ = case
        assert q_offset % BLOCK_ROWS == 0
        assert rows == min(BLOCK_ROWS, length - q_offset)
    for total_rows, tile_rows, columns in CAUSAL_MASK_CASES.values():
        assert total_rows % tile_rows == 0 and tile_rows <= BLOCK_ROWS
        assert columns >= tile_rows
    assert max(case[1] for case in CAUSAL_MASK_CASES.values()) == BLOCK_ROWS


def _representative(entry):
    """Run every head shape of float32 and the smallest of the 16-bit types."""
    if _kind(entry) not in {"vector", "pass2"}:
        return True
    return _dtype(entry) == "float32" or _dims(entry)[1] == 64


def test_sdpa_reference_matches_kernel_index_arithmetic():
    # The reference attends over the logical tensors; the simulation runs the
    # kernels' own index arithmetic, SIMD-group partitioning and online
    # softmax on the bound buffers in float64, rounding only the results.
    # Both use a negligible part of the bound.
    for entry, _, constants in VARIANTS:
        if not _representative(entry):
            continue
        for name in _variant_cases(entry, constants):
            expected, bounds = _reference(entry, constants, name)
            simulated = _simulate(entry, constants, name)
            assert _check(_dtype(entry), simulated, expected, bounds) < 2.0**-8, (
                entry,
                name,
            )


@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_sdpa_checks_reject_defects(dtype):
    rejected = set()
    for entry, _, constants in VARIANTS:
        if _dtype(entry) != dtype or not _representative(entry):
            continue
        for name in _variant_cases(entry, constants):
            expected, bounds = _reference(entry, constants, name)
            exact = _simulate(entry, constants, name)
            # An unwritten output fails.
            index = next(
                (i for i, bound in enumerate(bounds) if bound is None or bound),
                0,
            )
            guard = _from_bits(dtype, GUARD_BITS[dtype])
            unwritten = exact[:index] + [guard] + exact[index + 1 :]
            with pytest.raises(AssertionError):
                _check(dtype, unwritten, expected, bounds)
            for mutation in _mutations(entry, constants, name):
                wrong = _simulate(entry, constants, name, mutation)
                with pytest.raises(AssertionError):
                    _check(dtype, wrong, expected, bounds)
                rejected.add(mutation)
    expected_mutations = {
        "normalize",
        "last-key",
        "scale",
        "gqa",
        "transpose",
        "head-stride",
        "kv-stride",
        "causal",
        "mask-polarity",
        "mask-head",
        "mask-add",
        "sinks",
        "sink-head",
        "chunk",
        "max-lane",
        "stride",
        "sums",
    }
    if dtype == "float32":
        expected_mutations.add("binary16-score")
    else:
        expected_mutations |= {
            "scale-rounding",
            "offset",
            "strides",
            "head-batch",
            "diagonal",
            "row",
            "alignment",
        }
    assert rejected == expected_mutations


def test_sdpa_bounds_stay_meaningful():
    # Scores of unit-range inputs keep every softmax bound under 2^-12, far
    # below the change a dropped key, a wrong mask or a misplaced head causes;
    # unscaled 512-wide dot products widen it to 2^-9, still under one
    # bfloat16 unit at one half.
    for entry, _, constants in VARIANTS:
        if _kind(entry) not in {"vector", "pass2"}:
            continue
        for name in _variant_cases(entry, constants):
            _, bounds = _reference(entry, constants, name)
            limit = 2.0**-9 if name == "short" else 2.0**-12
            assert max(bounds) < limit, (entry, name, max(bounds))
            assert any(bounds), (entry, name)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_sdpa_configs_scope_target_options(tmp_path, target, entry, key, constants):
    kind = _kind(entry)
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
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS[kind]
        ]
        if target == "opengl"
        else []
    )
    text = path.read_text(encoding="utf-8")
    size = ", ".join(str(item) for item in WORKGROUP_SIZES[kind])
    assert f'"{entry}" = [{size}]' in text
    for name, identifier in CONSTANT_IDS.items():
        line = f'"{identifier}" = '
        if name in constants:
            assert line + str(constants[name]).lower() in text
        else:
            assert line not in text
    software = target in SOFTWARE_SUBGROUP_TARGETS and kind in SOFTWARE_SUBGROUP_KINDS
    assert ("software_subgroup_width = 32" in text) is software
    assert ("relative_wave_shuffle_out_of_range" in text) is (
        software and target == "directx"
    )


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native SDPA gate")
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
    for header, digest in HEADERS.items():
        assert hashlib.sha256((root / header).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("sdpa-metal-reference")
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


# Each kernel's bindings in scaled_dot_product_attention.cpp's buffer-index
# order: member -> payload kind. "T" buffers hold the storage type.
BINDINGS = {
    "vector": {
        "queries": "T",
        "keys": "T",
        "values": "T",
        "out": "out",
        "gqa_factor": "int32",
        "N": "int32",
        "k_head_stride": "uint64",
        "k_seq_stride": "uint64",
        "v_head_stride": "uint64",
        "v_seq_stride": "uint64",
        "scale": "float32",
        "bmask": "bool",
        "fmask": "T",
        "mask_kv_seq_stride": "int32",
        "mask_q_seq_stride": "int32",
        "mask_head_stride": "int32",
        "sinks": "T",
        "num_q_heads": "int32",
    },
    "pass2": {
        "partials": "T",
        "sums": "float32[]",
        "maxs": "float32[]",
        "out": "out",
        "blocks": "int32",
    },
    "scale-copy": {
        "q": "T",
        "q_block": "out",
        "scale": "float32",
        "q_strides": "int64[]",
        "heads": "int32",
        "rows": "int32",
        "head_dim": "int32",
        "q_offset": "int32",
    },
    "causal-mask": {"scores": "out", "tile_rows": "int32", "columns": "int32"},
}
OUTPUTS = {
    "vector": "out",
    "pass2": "out",
    "scale-copy": "q_block",
    "causal-mask": "scores",
}


def _metal_payload(dtype, values):
    if dtype == "float32":
        return struct.pack(f"<{len(values)}I", *(_bits(dtype, v) for v in values))
    return struct.pack(f"<{len(values)}H", *(_bits(dtype, v) for v in values))


def _metal_words(dtype, raw):
    if dtype == "float32":
        return struct.unpack(f"<{len(raw) // 4}I", raw)
    return struct.unpack(f"<{len(raw) // 2}H", raw)


def _upstream_payload(dtype, payload, value):
    if payload == "T":
        return _metal_payload(dtype, value)
    if payload == "out":
        return _metal_payload(
            dtype, list(value) + [_from_bits(dtype, GUARD_BITS[dtype])] * GUARD_COUNT
        )
    if payload == "float32":
        return struct.pack("<f", value)
    if payload == "float32[]":
        return _metal_payload("float32", value)
    if payload == "int32":
        return struct.pack("<i", value)
    if payload == "uint64":
        return struct.pack("<Q", value)
    if payload == "int64[]":
        return struct.pack(f"<{len(value)}q", *value)
    assert payload == "bool"
    return struct.pack(f"<{len(value)}B", *(int(bool(item)) for item in value))


def _upstream_metal(work, reference, entry, constants, name, buffers):
    """Run the unchanged upstream kernel with MLX's buffer indices."""
    runner, library = reference
    kind, dtype = _kind(entry), _dtype(entry)
    files = []
    for index, (member, payload) in enumerate(BINDINGS[kind].items()):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(_upstream_payload(dtype, payload, buffers[member]))
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": list(buffers["grid"]),
        "workgroupSize": list(WORKGROUP_SIZES[kind]),
        "simdWidth": SIMD_SIZE,
        # sdpa_vector() always specializes all six constants.
        "boolConstants": (
            [
                {"index": identifier, "value": constants[constant]}
                for constant, identifier in CONSTANT_IDS.items()
            ]
            if constants
            else []
        ),
    }
    request_path = work / f"{name}-upstream-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request_path, output], work, f"{name}-upstream")
    )
    out_index = list(BINDINGS[kind]).index(OUTPUTS[kind])
    for index, path in enumerate(files):
        if index != out_index:
            assert (output / f"buffer-{index}.bin").read_bytes() == Path(
                path
            ).read_bytes()
    words = _metal_words(dtype, (output / f"buffer-{out_index}.bin").read_bytes())
    count = len(buffers[OUTPUTS[kind]])
    assert list(words[count:]) == [GUARD_BITS[dtype]] * GUARD_COUNT
    return device, [_from_bits(dtype, word) for word in words[:count]]


def _pipeline_limits(work, reference, translated, entry, constants):
    """Return both pipelines' maxTotalThreadsPerThreadgroup for ``constants``.

    MLX's check_kernel_threadgroup_size throws when this limit is below the
    threadgroup it requests, so a kernel whose upstream pipeline cannot run
    1,024 threads on the device cannot run in MLX either.
    """
    runner, upstream = reference
    placeholder = work / "pipeline-placeholder.bin"
    placeholder.write_bytes(b"\0" * 4)
    request = work / "pipeline-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": [str(placeholder)],
                "workgroupCount": [1, 1, 1],
                "workgroupSize": list(WORKGROUP_SIZES[_kind(entry)]),
                "simdWidth": SIMD_SIZE,
                "boolConstants": (
                    [
                        {"index": identifier, "value": constants[constant]}
                        for constant, identifier in CONSTANT_IDS.items()
                    ]
                    if constants
                    else []
                ),
                "pipelineOnly": True,
            }
        ),
        encoding="utf-8",
    )
    limits = {}
    for label, library in (("translated", translated), ("upstream", upstream)):
        record = json.loads(
            _run(
                [runner, library, entry, request, work / f"pipeline-{label}"],
                work,
                f"pipeline-{label}",
            )
        )
        assert record["simdWidth"] == SIMD_SIZE, record
        limits[label] = record["maxTotalThreadsPerThreadgroup"]
        limits["device"] = record["device"]
    limits["requested"] = math.prod(WORKGROUP_SIZES[_kind(entry)])
    return limits


def _member(binding, entry):
    layout = binding["scalarLayout"]
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry) + "_"
    member = layout.get("memberName", binding["name"]).removeprefix(prefix)
    return member.removesuffix("Buffer").rstrip("_")


def _dispatch(descriptor, package, target, entry, buffers):
    kind, dtype = _kind(entry), _dtype(entry)
    plan = BINDINGS[kind]
    values, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        member = _member(binding, entry)
        payload = plan[member]
        element_type, encoding = _element_layout(binding)
        data = buffers[member]
        if payload in {"T", "out"}:
            if encoding is not None:
                assert encoding["logicalElementType"] == dtype, (member, encoding)
            else:
                assert element_type in {dtype, "float32", "uint16"}, (
                    member,
                    element_type,
                )
            value = _runtime_value(dtype, element_type, data, encoding)
            if payload == "out":
                count = len(data)
                guards = _guard_values(dtype, element_type)
                value["values"] = value["values"][:count] + guards
                value["shape"] = [count + GUARD_COUNT]
                output = (binding["name"], value["dtype"], count, guards)
        elif payload in {"float32", "float32[]"}:
            assert element_type == "float32" and encoding is None, binding
            items = data if payload == "float32[]" else [data]
            value = _runtime_value("float32", element_type, items, None)
        elif payload == "bool":
            if target == "metal":
                assert element_type == "bool", binding
                value = {
                    "dtype": "bool",
                    "shape": [len(data)],
                    "values": [bool(item) for item in data],
                }
            else:
                # OpenGL and DirectX store booleans as 32-bit words.
                assert element_type == "uint32", binding
                value = {
                    "dtype": "uint32",
                    "shape": [len(data)],
                    "values": [int(bool(item)) for item in data],
                }
        else:
            expected_type = payload.removesuffix("[]")
            assert element_type == expected_type, (member, element_type)
            items = list(data) if payload.endswith("[]") else [data]
            value = {"dtype": element_type, "shape": [len(items)], "values": items}
        values[binding["name"]] = value
        if payload != "out":
            observed.append(binding["name"])
    assert output is not None
    assert len(observed) == len(plan) - 1, (sorted(observed), sorted(plan))
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        {output[0]: dict(values[output[0]])},
        {
            "workgroupCount": list(buffers["grid"]),
            "workgroupSize": list(WORKGROUP_SIZES[kind]),
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output, len(observed)


def _assert_lowering(text, target, entry, constants):
    kind = _kind(entry)
    assert "unsupported" not in text
    # sdpa_vector calls fast::exp three times, sdpa_vector_2pass_2 twice.
    calls = {"vector": 3, "pass2": 2}.get(kind, 0)
    if target == "metal":
        # Translated Metal keeps every fast::exp.
        assert text.count("metal::fast::exp(") == text.count("fast::") == calls
    else:
        # OpenGL and DirectX call their default exp.
        assert "fast::" not in text
    if kind in SOFTWARE_SUBGROUP_KINDS:
        assert text.count("exp(") >= calls
        if target == "metal":
            assert "simd_sum(" in text and "simd_max(" in text
        elif target == "opengl":
            assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in text
        else:
            assert "__crossgl_software_subgroup_sum_float(" in text
            assert "__crossgl_software_subgroup_max_float(" in text
    for name, value in constants.items():
        identifier = CONSTANT_IDS[name]
        if target == "metal":
            assert f"constant bool {name} [[function_constant({identifier})]];" in text
        elif target == "opengl":
            # Specialized when the pipeline is created, as on Metal.
            assert f"layout(constant_id = {identifier}) const bool {name} = " in text
        else:
            assert f"static const bool {name} = {str(value).lower()};" in text
    if not constants:
        assert "function_constant(" not in text and "constant_id" not in text


@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_current_mlx_sdpa_native_parity(
    current_mlx, metal_reference, tmp_path, entry, key, constants
):
    root, target = current_mlx
    kind, dtype = _kind(entry), _dtype(entry)
    record = _contract()["artifacts"][target][entry][key]
    with tempfile.TemporaryDirectory(prefix=".current-sdpa-", dir=root) as path:
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
            assert artifact["generatedHash"] == {
                "algorithm": "sha256",
                "value": record["sha256"],
            }
            assert artifact["generatedSizeBytes"] == record["sizeBytes"]
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == list(WORKGROUP_SIZES[kind])
            generated = root / artifact["path"]
            _assert_lowering(
                generated.read_text(encoding="utf-8"), target, entry, constants
            )
            limits = None
            if target == "metal":
                translated_library = _metal_library(
                    generated, work / "translated.metallib", root
                )
                limits = _pipeline_limits(
                    work, metal_reference, translated_library, entry, constants
                )
                if limits["upstream"] >= limits["requested"]:
                    # MLX dispatches its own kernel on this device, so the
                    # translation must fit the same threadgroup.
                    assert limits["translated"] >= limits["requested"], limits
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            assert {
                item["id"]: item["value"]
                for item in descriptor["specializationConstants"]
            } == {CONSTANT_IDS[name]: value for name, value in constants.items()}
            if limits is not None and limits["translated"] < limits["requested"]:
                # Neither pipeline runs MLX's threadgroup on this device, where
                # MLX's check_kernel_threadgroup_size throws for its own kernel.
                (work / "parity.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "entry": entry,
                            "target": target,
                            "specialization": key,
                            "constants": constants,
                            "artifactSha256": record["sha256"],
                            "pipelineLimits": limits,
                            "cases": [],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                pytest.skip(
                    f"{limits['device']} runs at most {limits['upstream']} threads "
                    f"of MLX's {entry} and {limits['translated']} of the "
                    f"translation; MLX requests {limits['requested']}"
                )
            executor = _executor(target)
            evidence = []
            try:
                for name in _variant_cases(entry, constants):
                    buffers = _buffers(entry, constants, name)
                    expected, bounds = _reference(entry, constants, name)
                    request, output, observed = _dispatch(
                        descriptor, package, target, entry, buffers
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
                    if target == "metal" and limits["upstream"] < limits["requested"]:
                        # Only MLX's own pipeline is too small to dispatch.
                        upstream = {"dispatched": False, "pipelineLimits": limits}
                    elif target == "metal":
                        device, upstream_values = _upstream_metal(
                            work, metal_reference, entry, constants, name, buffers
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
                    finite = [bound for bound in bounds if bound is not None]
                    evidence.append(
                        {
                            "case": name,
                            "grid": list(buffers["grid"]),
                            "outputCount": written,
                            "largestBound": max(finite) if finite else None,
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
                        "pipelineLimits": limits,
                        "cases": evidence,
                        "indexRangeAssertions": (
                            list(OPENGL_INDEX_RANGE_ASSERTIONS[kind])
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
                # The round trip keeps fast::exp, so wherever MLX's own kernel
                # runs, the translated kernel reproduces it bit for bit.
                assert [
                    case["case"]
                    for case in evidence
                    if case["upstream"].get("dispatched", True)
                    and not case["upstream"]["bitIdenticalToTranslated"]
                ] == []
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_sdpa_parity():
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )
    from tools import ci_coverage

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX scaled dot-product attention native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/sdpa",
        "--junitxml=mlx-current-results/sdpa.xml",
        "demos/integrations/mlx/tests/kernels/test_current_sdpa.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_sdpa.py"
    )
    upload = ci_coverage.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX rope native parity") < (
        workflow.index("Prove current MLX scaled dot-product attention native parity")
    )
    assert workflow.index(
        "Prove current MLX scaled dot-product attention native parity"
    ) < workflow.index("Upload current MLX runtime diagnostics")
