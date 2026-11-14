"""Translate and execute the pinned MLX gated-delta forward kernels on all backends."""

import hashlib
import json
import math
import os
import re
import shutil
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
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
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
    _element_layout,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/gated_delta_update.metal"
SOURCE_SHA256 = "8e3a37c7e272dd7fd9aadcce28e210bb5b9a26d1c2aae1fd34823007b1ff1df7"
HEADER = "mlx/backend/metal/kernels/gated_delta_update.h"
HEADER_SHA256 = "ca081d5bf13e2a5379cfc12306da779cbe1ea73e346aa68e6d68ab814a284597"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GATED_DELTA"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/gated_delta_update.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# instantiate_gated_delta's order: each storage type, each (Dk, Dv, Hk, Hv)
# that supported_gated_delta_shape() accepts, then the sequential kernel for
# every checkpoint interval gated_delta_ckpt() can return and the eight-step
# fused chunk kernel. The host names keep the C++ type spellings.
STORAGE_TYPES = {"float": "float32", "bfloat16_t": "bfloat16", "float16_t": "float16"}
SHAPES = (
    (128, 128, 24, 24),
    (128, 128, 32, 32),
    (128, 128, 16, 32),
    (128, 128, 16, 48),
    (128, 128, 16, 16),
    (128, 128, 16, 64),
)
CHECKPOINTS = (1, 4, 8, 16)
CHUNK = 8
INSTANTIATED = tuple(
    name
    for type_name in STORAGE_TYPES
    for dk, dv, hk, hv in SHAPES
    for name in (
        *(
            f"seq_gated_delta_{type_name}_{dk}_{dv}_{hk}_{hv}_{ckpt}"
            for ckpt in CHECKPOINTS
        ),
        f"gated_delta_fused_chunk_{type_name}_{dk}_{dv}_{hk}_{hv}_{CHUNK}",
    )
)
# The fused chunk kernel holds its tiles in simdgroup_float8x8 values and
# addresses their fragments through thread_elements(): the registered
# tile_4x4_row_pair coordinates, which MLX spells out as fm and fn. OpenGL and
# DirectX lower the tiles to 32-lane software subgroups, each lane holding its
# own two fragment elements.
SEQ_ENTRIES = tuple(entry for entry in INSTANTIATED if entry.startswith("seq_"))
FUSED_ENTRIES = tuple(
    entry for entry in INSTANTIATED if entry.startswith("gated_delta_fused_chunk_")
)
ENTRIES = INSTANTIATED
FRAGMENT_MAPPING = "tile_4x4_row_pair"
FRAGMENT_MAPPING_PROVENANCE = "mlx_gated_delta_lane_coordinates"
# for (; t + C <= T; t += C) must not overflow t + C, so OpenGL and DirectX
# take T from a host precondition when proving the chunk loop uniform.
T_MAXIMUM = 2**31 - 1 - CHUNK
SAVE_STATE_ID = 200
# gated_delta_update.cpp clears save_state (function constant 200) for
# inference and sets it for the backward's sequential save pass. The fused
# chunk kernel runs only for inference and never reads the constant.
SPECIALIZATIONS = {"inference": False, "save": True}
FUSED_SPECIALIZATIONS = {"inference": False}
VARIANTS = tuple(
    (entry, key, {"save_state": value})
    for entry in SEQ_ENTRIES
    for key, value in SPECIALIZATIONS.items()
) + tuple(
    (entry, key, {"save_state": value})
    for entry in FUSED_ENTRIES
    for key, value in FUSED_SPECIALIZATIONS.items()
)
SIMD_SIZE = 32
# gated_delta_update.cpp dispatches a (32, Dv, B * Hv) grid of (32, 4, 1)
# groups: one SIMD group per state row, each thread holding Dk / 32 entries.
WORKGROUP_SIZE = (32, 4, 1)
N_PER_T = 4
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
SUBGROUPS = {
    "metal": "native-simdgroup-32",
    "opengl": "software-subgroup-32",
    "directx": "software-subgroup-32",
}
# name -> (batch B, sequence T). The decode case is MLX's one-token step;
# eight steps are the longest sequence MLX sends to the sequential kernel when
# inferring; the checkpoint case runs 2 * Ckpt + 1 steps (None), so the save
# pass stores three checkpoint slots.
CASES = {
    "decode": (1, 1),
    "prefill": (2, 8),
    "checkpoints": (1, None),
}
SPECIALIZATION_CASES = {
    "inference": ("decode", "prefill"),
    "save": ("decode", "checkpoints"),
}
# gated_delta_chunk_size() picks the eight-step fused kernel for T > 8 on
# devices without NAX and dispatches a (32, Dv / 8, B * Hv) grid of
# (32, 4, 1) groups: each SIMD group holds an eight-column block of every
# state row of one head. The shortest case is the shortest fused sequence,
# one chunk and a one-step tail; the tail case leaves a five-step tail in
# each of two batches; the chunk case carries the state through three chunks.
FUSED_CASES = {
    "shortest": (1, 9),
    "tail": (2, 13),
    "chunks": (1, 24),
}
FUSED_SPECIALIZATION_CASES = {"inference": tuple(FUSED_CASES)}
FUSED_COLUMNS_PER_GROUP = 8
# Fused error model. Every 8 x 8 matrix product may sum in any order, so a sum
# of n products errs by at most gamma(n) of the sum of absolute products;
# fast::exp and fast::log follow the accuracy Metal's fast math and GLSL
# specify: exp within 3 + 2 |x| ulp and log within 2^-21 absolute on
# [0.5, 2]. Rounding committed in a chunk is tracked element by element; the
# state error it leaves is carried as a row 2-norm through the chunk's
# product of g (I - beta k k^T), as in the sequential kernel.
EXP_ULPS = (3.0, 2.0)
ULP_FRACTION = 2.0**-23
LOG_ABSOLUTE_ERROR = 2.0**-21
INPUT_GUARD_COUNT = 16
OUTPUT_GUARD_BITS = GUARD_BITS
# Error model. The kernel computes in binary32: one rounding (u = 2^-24) per
# operation, which may also flush a subnormal result (FTZ per operation),
# fused multiply-adds where the compiler contracts them, and a 32-lane
# reduction in any order, so every key and query dot product of Dk terms is
# within gamma(REDUCTION_ROUNDINGS) of its exact value relative to the sum of
# the absolute products. Rounding errors carried by the state propagate
# through g (I - beta k k^T), whose 2-norm is at most g while beta |k|^2 <= 2;
# the bound tracks the 2-norm of every state row's error from step to step.
UNIT_ROUNDOFF = 2.0**-24
FTZ = 2.0**-126
REDUCTION_ROUNDINGS = 1 + N_PER_T + (SIMD_SIZE - 1)
BOUND_SAFETY = 1.0 + 2.0**-10
TINY = 2.0**-140
SIGNIFICAND_BITS = {"float32": 24, "float16": 11, "bfloat16": 8}
# frexp exponent of the smallest normal value of each type.
MIN_NORMAL_EXPONENT = {"float32": -125, "float16": -13, "bfloat16": -125}
STATE_MAGNITUDE = 0.5


def _np():
    if os.environ.get(REQUIRE_ENV) == "1":
        import numpy

        return numpy
    return pytest.importorskip("numpy")


def _parse(entry):
    """Return (dtype, Dk, Dv, Hk, Hv, Ckpt or C) of a sequential or fused entry."""
    match = re.fullmatch(
        r"(?:seq_gated_delta|gated_delta_fused_chunk)_"
        r"(float|bfloat16_t|float16_t)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)",
        entry,
    )
    assert match, entry
    type_name, *numbers = match.groups()
    dk, dv, hk, hv, last = (int(item) for item in numbers)
    return STORAGE_TYPES[type_name], dk, dv, hk, hv, last


def _fused(entry):
    return entry.startswith("gated_delta_fused_chunk_")


def _dtype(entry):
    return _parse(entry)[0]


def _geometry(entry, name):
    batch, steps = (FUSED_CASES if _fused(entry) else CASES)[name]
    if steps is None:
        steps = 2 * _parse(entry)[5] + 1
    return batch, steps


def _cases(entry, key):
    return (FUSED_SPECIALIZATION_CASES if _fused(entry) else SPECIALIZATION_CASES)[key]


def _slots(entry, name):
    """Return n_ckpt, the checkpoint slots gated_delta_seq stores."""
    ckpt = _parse(entry)[5]
    return (_geometry(entry, name)[1] + ckpt - 1) // ckpt


def _gamma(count):
    return count * UNIT_ROUNDOFF / (1.0 - count * UNIT_ROUNDOFF)


def _uniform(np, label, shape):
    """Return platform-independent uniform [0, 1) values for ``label``."""
    count = math.prod(shape)
    raw = np.random.PCG64(zlib.crc32(label.encode("utf-8"))).random_raw(count)
    return ((raw >> np.uint64(11)).astype(np.float64) * 2.0**-53).reshape(shape)


def _quantize(np, dtype, values):
    """Round to ``dtype`` storage (through binary32, ties to even)."""
    single = np.asarray(values, dtype=np.float64).astype(np.float32)
    if dtype == "float32":
        return single.astype(np.float64)
    if dtype == "float16":
        return single.astype(np.float16).astype(np.float64)
    bits = single.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return (rounded.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _round(np, dtype, values):
    """Round float64 values to ``dtype`` with ties to even, in one step."""
    exponent = np.maximum(np.frexp(values)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = np.ldexp(1.0, exponent - SIGNIFICAND_BITS[dtype])
    return np.where(values == 0, values, np.rint(values / quantum) * quantum)


def _ulp(np, dtype, values):
    exponent = np.maximum(np.frexp(values)[1], MIN_NORMAL_EXPONENT[dtype])
    return np.ldexp(1.0, exponent - SIGNIFICAND_BITS[dtype])


def _inputs(entry, name):
    """Return the logical input tensors of one case as MLX lays them out."""
    np = _np()
    dtype, dk, dv, hk, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    label = f"{dtype}:{dk}:{dv}:{hk}:{hv}:{name}:{steps}"

    def uniform(tag, shape):
        return _uniform(np, f"{label}:{tag}", shape)

    def unit(tag, shape):
        # MLX normalizes queries and keys; |k| stays 1 within the storage
        # rounding, so beta |k|^2 <= 2 holds for every beta in [0, 1].
        raw = 2.0 * uniform(tag, shape) - 1.0
        return raw / np.linalg.norm(raw, axis=-1, keepdims=True)

    keys = unit("k", (batch, steps, hk, dk))
    if _fused(entry):
        # A head's keys share a direction, so k_i . k_j is near 1/2 and the
        # chunk's WY inverse (I + beta K K^T)^-1 departs far from I.
        mixed = keys + unit("k-direction", (batch, 1, hk, dk))
        keys = mixed / np.linalg.norm(mixed, axis=-1, keepdims=True)

    return {
        # q, k: [B, T, Hk, Dk]; v: [B, T, Hv, Dv]; g, beta: [B, T, Hv];
        # state_in: [B, Hv, Dv, Dk] in float.
        "q": _quantize(np, dtype, unit("q", (batch, steps, hk, dk))),
        "k": _quantize(np, dtype, keys),
        "v": _quantize(np, dtype, 2.0 * uniform("v", (batch, steps, hv, dv)) - 1.0),
        "g": _quantize(np, dtype, 0.5 + 0.5 * uniform("g", (batch, steps, hv))),
        "beta": _quantize(np, dtype, uniform("beta", (batch, steps, hv))),
        "state_in": _quantize(
            np,
            "float32",
            STATE_MAGNITUDE * (2.0 * uniform("state", (batch, hv, dv, dk)) - 1.0),
        ),
    }


def _reference(entry, name, save):
    """Return exact outputs and per-element bounds on the kernel's rounding.

    The state recurrence is s1 = g s, d = beta (v - k . s1), s' = s1 + d k and
    y = q . s'. For a computed state with error e, the computed step equals
    the exact step applied to the computed state plus that step's own
    rounding, so |e'| <= g |I - beta k k^T| |e| + |rho| row by row, where rho
    collects the rounding of s1, of both reductions, of d and of the update.
    """
    if _fused(entry):
        assert not save, entry
        return _fused_reference(entry, name)
    np = _np()
    dtype, dk, dv, hk, hv, ckpt = _parse(entry)
    data = _inputs(entry, name)
    batch, steps = _geometry(entry, name)
    heads = np.arange(hv) // (hv // hk)
    gamma = _gamma(REDUCTION_ROUNDINGS)
    unit = UNIT_ROUNDOFF
    root = math.sqrt(dk)
    state = data["state_in"].copy()
    error = np.zeros((batch, hv, dv))
    outputs = np.empty((batch, steps, hv, dv))
    output_bounds = np.empty((batch, steps, hv, dv))
    slots, slot_bounds = [], []
    for t in range(steps):
        if save and t % ckpt == 0:
            # Stored before the decay: the state at entry to step t.
            slots.append(state.copy())
            slot_bounds.append(error.copy())
        keys = data["k"][:, t][:, heads]
        queries = data["q"][:, t][:, heads]
        decay = data["g"][:, t][..., None]
        beta = data["beta"][:, t][..., None]
        values = data["v"][:, t]
        key_norm = np.linalg.norm(keys, axis=-1)[..., None]
        query_norm = np.linalg.norm(queries, axis=-1)[..., None]
        contraction = np.maximum(1.0, np.abs(1.0 - beta * key_norm**2))
        decayed = state * decay[..., None]
        # |g s_hat| <= g (|s| + e); s1_hat = g s_hat + r1.
        magnitude = decay * (np.linalg.norm(state, axis=-1) + error)
        r1 = unit * magnitude + FTZ * root
        decayed_norm = magnitude + r1
        decayed_error = decay * error + r1
        key_sum = np.einsum("bhvk,bhk->bhv", np.abs(decayed), np.abs(keys))
        r_kv = gamma * (key_sum + key_norm * decayed_error) + 2 * dk * FTZ
        kv = np.einsum("bhvk,bhk->bhv", decayed, keys)
        residual = np.abs(values - kv) + key_norm * decayed_error + r_kv
        r_d = (2 * unit + unit * unit) * np.abs(beta) * residual + (
            1.0 + (1.0 + unit) * np.abs(beta)
        ) * FTZ
        delta_magnitude = np.abs(beta) * residual + r_d
        r2 = (
            unit * key_norm * delta_magnitude
            + unit * (decayed_norm + key_norm * delta_magnitude)
            + 2 * FTZ * root
        ) / (1.0 - unit)
        rho = contraction * r1 + key_norm * (r_d + np.abs(beta) * r_kv) + r2
        state = decayed + ((values - kv) * beta)[..., None] * keys[:, :, None, :]
        error = decay * contraction * error + rho
        outputs[:, t] = np.einsum("bhvk,bhk->bhv", state, queries)
        query_sum = np.einsum("bhvk,bhk->bhv", np.abs(state), np.abs(queries))
        output_bounds[:, t] = (
            query_norm * error * (1.0 + gamma) + gamma * query_sum + 2 * dk * FTZ
        )
    result = {
        "data": data,
        "y": outputs,
        "yBound": output_bounds * BOUND_SAFETY + TINY,
        "state": state,
        "stateBound": np.broadcast_to(
            (error * BOUND_SAFETY + TINY)[..., None], state.shape
        ),
    }
    if save:
        # state_cache: [B * Hv, n_ckpt, Dv, Dk].
        cache = np.stack(slots, axis=2).reshape(batch * hv, len(slots), dv, dk)
        bounds = np.stack(slot_bounds, axis=2).reshape(batch * hv, len(slots), dv)
        result["cache"] = cache
        result["cacheBound"] = np.broadcast_to(
            (bounds * BOUND_SAFETY + TINY)[..., None], cache.shape
        )
    return result


def _simulate(entry, name, save, mutation=None):
    """Run gated_delta_seq's own index arithmetic on the flat buffers.

    Every thread (lane, dv, n) is evaluated exactly in float64; y takes the
    kernel's final conversion to the storage type. ``mutation`` injects one
    defect.
    """
    np = _np()
    dtype, dk, dv, hk, hv, ckpt = _parse(entry)
    data = _inputs(entry, name)
    batch, steps = _geometry(entry, name)
    flat = {key: value.reshape(-1) for key, value in data.items()}
    n = np.arange(batch * hv)[:, None, None]
    dv_idx = np.arange(dv)[None, :, None]
    lane = np.arange(SIMD_SIZE)[None, None, :]
    b_idx = n // hv
    hv_idx = n % hv
    hk_idx = hv_idx // (hv // hk)
    if mutation == "key-head":
        hk_idx = hv_idx % hk
    t_stride = 1 if mutation == "batch-stride" else steps
    q_ptr = b_idx * t_stride * hk * dk + hk_idx * dk
    k_ptr = b_idx * t_stride * hk * dk + hk_idx * dk
    v_ptr = b_idx * steps * hv * dv + hv_idx * dv
    y_ptr = b_idx * steps * hv * dv + hv_idx * dv
    g_ptr = b_idx * steps * hv
    i = np.arange(N_PER_T)
    # s_idx = n_per_t * dk_idx + i: [threads..., lanes, i].
    s_idx = (N_PER_T * lane)[..., None] + i
    row = ((n * dv + dv_idx) * dk)[..., None]
    state = flat["state_in"][row + s_idx]
    if mutation == "zero-state":
        state = np.zeros_like(state)
    n_ckpt = (steps + ckpt - 1) // ckpt
    cache = np.full(batch * hv * n_ckpt * dv * dk, np.nan)
    cache_base = (n * n_ckpt * dv * dk + dv_idx * dk)[..., None]
    y = np.full(batch * steps * hv * dv, np.nan)
    for t in range(steps):
        if save and t % ckpt == 0:
            slot = t // ckpt
            if mutation == "checkpoint-slot":
                slot = (slot + 1) % n_ckpt
            stored = state
            if mutation == "checkpoint-after-decay":
                stored = state * flat["g"][g_ptr + hv_idx][..., None]
            cache[cache_base + slot * dv * dk + s_idx] = stored
        decay = flat["g"][g_ptr + hv_idx][..., None]
        if mutation == "gate-head":
            decay = flat["g"][g_ptr + hk_idx][..., None]
        keys = flat["k"][(k_ptr + 0)[..., None] + s_idx]
        if mutation == "stale-keys":
            keys = flat["k"][(k_ptr - t * hk * dk)[..., None] + s_idx]
        queries = flat["q"][q_ptr[..., None] + s_idx]
        decayed = state if mutation == "no-decay" else state * decay
        lanes = (decayed * keys).sum(axis=-1)
        kv = lanes if mutation == "lane-sum" else lanes.sum(axis=-1)[..., None]
        value = flat["v"][v_ptr + dv_idx]
        beta = flat["beta"][g_ptr + hv_idx]
        delta = (value - kv) if mutation == "missing-beta" else (value - kv) * beta
        update_keys = queries if mutation == "query-for-key" else keys
        updated = decayed + update_keys * delta[..., None]
        read = decayed if mutation == "output-before-update" else updated
        y[(y_ptr + dv_idx)[..., 0]] = (read * queries).sum(axis=(-2, -1))
        if mutation == "checkpoint-exit" and save and t % ckpt == 0:
            cache[cache_base + (t // ckpt) * dv * dk + s_idx] = updated
        state = updated
        q_ptr = q_ptr + hk * dk
        k_ptr = k_ptr + hk * dk
        v_ptr = v_ptr + hv * dv
        y_ptr = y_ptr + hv * dv
        g_ptr = g_ptr + hv
    state_out = np.full(batch * hv * dv * dk, np.nan)
    state_out[row + s_idx] = state
    # Exact results rounded once to the stored type.
    rounded = _round(np, dtype, y)
    if mutation == "truncate-output" and dtype != "float32":
        rounded = _truncate(np, dtype, y)
    result = {
        "y": rounded.reshape(batch, steps, hv, dv),
        "state": _round(np, "float32", state_out).reshape(batch, hv, dv, dk),
    }
    if save:
        result["cache"] = _round(np, "float32", cache).reshape(
            batch * hv, n_ckpt, dv, dk
        )
    return result


# --- Fused chunk kernel -------------------------------------------------------


def _exp_error(np, x, x_error):
    """Bound |fast::exp(x_hat) - exp(x)| for any |x_hat - x| <= x_error."""
    reach = np.abs(x) + x_error
    return np.exp(x) * np.expm1(x_error) + (
        EXP_ULPS[0] + EXP_ULPS[1] * reach
    ) * ULP_FRACTION * np.exp(x + x_error)


def _scale_error(x, x_error, scale, scale_error):
    """Bound fl(x_hat * s_hat) - x * s from magnitudes |x| and |s|."""
    return (
        x_error * (scale + scale_error)
        + x * scale_error
        + UNIT_ROUNDOFF * (x + x_error) * (scale + scale_error)
        + FTZ
    )


def _sum_error(result, *errors):
    """Bound fl(x_hat +- y_hat) - (x +- y) when |x +- y| <= ``result``."""
    carried = sum(errors)
    return carried + UNIT_ROUNDOFF * (result + carried) + FTZ


def _product_error(np, left, left_error, right, right_error, acc=None, acc_error=None):
    """Bound a matrix product (plus an accumulator) summed in any order.

    ``left`` and ``right`` bound the magnitudes of exact operands; with n
    products and an accumulator the sum of n + 1 terms errs by at most
    gamma(n + 1) of its absolute terms, and every operation may flush.
    """
    terms = left.shape[-1]
    if left_error is None:
        left_error = np.zeros_like(left)
    if right_error is None:
        right_error = np.zeros_like(right)
    error = left_error @ right + left @ right_error + left_error @ right_error
    reach = (left + left_error) @ (right + right_error)
    extra = 0
    if acc is not None:
        extra = 1
        error = error + acc_error
        reach = reach + acc + acc_error
    return error + _gamma(terms + extra) * reach + (2 * terms + extra) * FTZ


def _swap(np, values):
    return np.swapaxes(values, -1, -2)


def _fused_chunk(np, keys, queries, values, decays, betas, valid, state, reach):
    """Run one chunk of PROCESS_CHUNK_SG exactly and bound its rounding.

    keys and queries [N, C, Dk], values [N, C, Dv], decays and betas [N, C]
    hold the chunk's rows, zero past ``valid``. ``state`` [N, Dv, Dk] is the
    exact state and ``reach`` [N, Dv] bounds each row of the computed state's
    error, so |s_hat| <= |s| + reach element by element. Returns the exact
    chunk outputs [N, C, Dv] and new state [N, Dv, Dk] the WY form computes,
    and bounds on the rounding the kernel commits from the computed state.
    """
    unit = UNIT_ROUNDOFF
    chunk, dk = keys.shape[-2:]
    rows = np.arange(chunk)
    live = rows < valid
    lower = rows[:, None] >= rows[None, :]
    strict = rows[:, None] > rows[None, :]
    # gamma: the inclusive prefix sums of fast::log(g), zero past the tail.
    # Every use reads the same stored sums, so the log errors of the rows a
    # difference gamma_m - gamma_n shares cancel; only the scan's own
    # rounding differs between its ends.
    logs = np.where(live, np.log(np.where(live, decays, 1.0)), 0.0)
    log_error = np.where(live, LOG_ABSOLUTE_ERROR, 0.0) * np.ones_like(logs)
    prefix = np.cumsum(logs, axis=-1)
    log_reach = np.cumsum(log_error, axis=-1)
    scan_error = (
        _gamma(rows) * np.cumsum(np.abs(logs) + log_error, axis=-1) + rows * FTZ
    )

    def difference(high, low):
        value = prefix[..., high] - prefix[..., low]
        error = (
            np.abs(log_reach[..., high] - log_reach[..., low])
            + scan_error[..., high]
            + scan_error[..., low]
        )
        return value, error + unit * (np.abs(value) + error) + FTZ

    def factor(argument, argument_error):
        return np.exp(argument), _exp_error(np, argument, argument_error)

    row_decay, row_decay_error = factor(prefix, log_reach + scan_error)
    pair, pair_error = factor(*difference(rows[:, None], rows[None, :]))
    column_decay, column_decay_error = factor(*difference(np.full(chunk, -1), rows))
    chunk_decay, chunk_decay_error = factor(
        prefix[..., -1], log_reach[..., -1] + scan_error[..., -1]
    )
    # T^-1 = (I + strict_lower(beta K K^T))^-1 through the Neumann product.
    keys_t = _swap(np, keys)
    kkt = keys @ keys_t
    kkt_error = _gamma(dk) * (np.abs(keys) @ np.abs(keys_t)) + 2 * dk * FTZ
    beta_rows = np.abs(betas)[..., :, None]
    power = np.where(strict, betas[..., :, None] * kkt, 0.0)
    power_error = np.where(
        strict, _scale_error(np.abs(kkt), kkt_error, beta_rows, 0.0), 0.0
    )
    inverse = np.eye(chunk) - power
    inverse_error = power_error.copy()
    step = 1
    while (1 << step) < chunk:
        squared = power @ power
        power_error = _product_error(
            np, np.abs(power), power_error, np.abs(power), power_error
        )
        power = squared
        product = inverse + inverse @ power
        inverse_error = _product_error(
            np,
            np.abs(inverse),
            inverse_error,
            np.abs(power),
            power_error,
            np.abs(inverse),
            inverse_error,
        )
        inverse = product
        step += 1
    # Magnitudes of state-dependent values at the computed state: the exact
    # value plus the linear map's absolute value applied to the error.
    state_t = _swap(np, state)
    state_reach = np.abs(state_t) + reach[..., None, :]

    def shifted(weights):
        return np.abs(weights).sum(axis=-1)[..., :, None] * reach[..., None, :]

    # W S^T with W = diag(e^gamma) T^-1 diag(beta) K.
    scaled_keys = betas[..., None] * keys
    scaled_keys_error = unit * np.abs(scaled_keys) + FTZ
    weights = inverse @ scaled_keys
    weights_error = _product_error(
        np, np.abs(inverse), inverse_error, np.abs(scaled_keys), scaled_keys_error
    )
    decayed_weights = row_decay[..., None] * weights
    decayed_weights_error = _scale_error(
        np.abs(weights),
        weights_error,
        row_decay[..., None],
        row_decay_error[..., None],
    )
    projected = decayed_weights @ state_t
    projected_error = _product_error(
        np, np.abs(decayed_weights), decayed_weights_error, state_reach, None
    )
    # U = (Gamma o T^-1) diag(beta) V and delta = U - W S^T.
    gated = np.where(lower, inverse * pair, 0.0)
    gated_error = np.where(
        lower, _scale_error(np.abs(inverse), inverse_error, pair, pair_error), 0.0
    )
    scaled_values = betas[..., None] * values
    scaled_values_error = unit * np.abs(scaled_values) + FTZ
    targets = gated @ scaled_values
    targets_error = _product_error(
        np, np.abs(gated), gated_error, np.abs(scaled_values), scaled_values_error
    )
    delta = targets - projected
    delta_reach = np.abs(delta) + shifted(decayed_weights)
    delta_error = _sum_error(delta_reach, targets_error, projected_error)
    # y = (Gamma o Q K^T) delta + diag(e^gamma) Q S^T.
    qkt = queries @ keys_t
    qkt_error = _gamma(dk) * (np.abs(queries) @ np.abs(keys_t)) + 2 * dk * FTZ
    decayed_queries = row_decay[..., None] * queries
    decayed_queries_error = _scale_error(
        np.abs(queries), 0.0, row_decay[..., None], row_decay_error[..., None]
    )
    history = decayed_queries @ state_t
    history_error = _product_error(
        np, np.abs(decayed_queries), decayed_queries_error, state_reach, None
    )
    history_reach = np.abs(history) + shifted(decayed_queries)
    attention = np.where(lower, qkt * pair, 0.0)
    attention_error = np.where(
        lower, _scale_error(np.abs(qkt), qkt_error, pair, pair_error), 0.0
    )
    outputs = attention @ delta + history
    outputs_error = _product_error(
        np,
        np.abs(attention),
        attention_error,
        delta_reach,
        delta_error,
        history_reach,
        history_error,
    )
    # S^T' = e^gamma_C S^T + (K^T diag(e^(gamma_C - gamma))) delta.
    keys_decayed = keys_t * column_decay[..., None, :]
    keys_decayed_error = _scale_error(
        np.abs(keys_t),
        0.0,
        column_decay[..., None, :],
        column_decay_error[..., None, :],
    )
    update = keys_decayed @ delta
    update_error = _product_error(
        np, np.abs(keys_decayed), keys_decayed_error, delta_reach, delta_error
    )
    carried = chunk_decay[..., None, None] * state_t
    carried_error = _scale_error(
        state_reach,
        0.0,
        chunk_decay[..., None, None],
        chunk_decay_error[..., None, None],
    )
    new_state = carried + update
    new_reach = np.abs(new_state) + (
        chunk_decay[..., None, None] * reach[..., None, :]
        + np.abs(keys_decayed) @ shifted(decayed_weights)
    )
    new_state_error = _sum_error(new_reach, carried_error, update_error)
    return outputs, outputs_error, _swap(np, new_state), _swap(np, new_state_error)


def _rows(np, entry, data):
    """Return per-head rows [B * Hv, T, D] of the logical inputs."""
    _, dk, dv, hk, hv, _ = _parse(entry)
    heads = np.arange(hv) // (hv // hk)
    batch, steps = data["g"].shape[:2]

    def heads_first(values):
        return np.moveaxis(values, 2, 1).reshape(batch * hv, steps, -1)

    return {
        "k": heads_first(data["k"][:, :, heads]),
        "q": heads_first(data["q"][:, :, heads]),
        "v": heads_first(data["v"]),
        "g": heads_first(data["g"][..., None])[..., 0],
        "beta": heads_first(data["beta"][..., None])[..., 0],
        "state": data["state_in"].reshape(batch * hv, dv, dk).copy(),
    }


def _chunk_rows(np, values, start, valid, chunk, fill=0.0):
    rows = np.full(values.shape[:1] + (chunk,) + values.shape[2:], fill)
    rows[:, :valid] = values[:, start : start + valid]
    return rows


def _fused_reference(entry, name):
    """Return exact outputs and per-element bounds for the fused chunk kernel.

    Exact results come from the recurrence step by step. Within a chunk the
    kernel's WY form rounds element by element (``_fused_chunk``); the state
    error it carries is a row 2-norm, which each step's g (I - beta k k^T)
    cannot grow while beta |k|^2 <= 2, so an output row sees the carried error
    through |q| and every step before it.
    """
    np = _np()
    dtype, dk, dv, hk, hv, chunk = _parse(entry)
    data = _inputs(entry, name)
    batch, steps = _geometry(entry, name)
    rows = _rows(np, entry, data)
    heads = batch * hv
    state = rows["state"]
    error = np.zeros((heads, dv))
    outputs = np.empty((heads, steps, dv))
    output_bounds = np.empty((heads, steps, dv))
    for start in range(0, steps, chunk):
        valid = min(chunk, steps - start)
        _, rho_y, _, rho_s = _fused_chunk(
            np,
            _chunk_rows(np, rows["k"], start, valid, chunk),
            _chunk_rows(np, rows["q"], start, valid, chunk),
            _chunk_rows(np, rows["v"], start, valid, chunk),
            _chunk_rows(np, rows["g"], start, valid, chunk, fill=1.0),
            _chunk_rows(np, rows["beta"], start, valid, chunk),
            valid,
            state,
            error,
        )
        carried = error
        for offset in range(valid):
            t = start + offset
            keys, queries = rows["k"][:, t], rows["q"][:, t]
            decay, beta = rows["g"][:, t], rows["beta"][:, t]
            decayed = state * decay[:, None, None]
            kv = np.einsum("nvk,nk->nv", decayed, keys)
            state = (
                decayed
                + ((rows["v"][:, t] - kv) * beta[:, None])[..., None] * keys[:, None, :]
            )
            outputs[:, t] = np.einsum("nvk,nk->nv", state, queries)
            contraction = np.maximum(
                1.0, np.abs(1.0 - beta * np.linalg.norm(keys, axis=-1) ** 2)
            )
            carried = carried * (decay * contraction)[:, None]
            output_bounds[:, t] = (
                rho_y[:, offset] + np.linalg.norm(queries, axis=-1)[:, None] * carried
            )
        error = carried + np.linalg.norm(rho_s, axis=-1)

    def logical(values):
        return np.moveaxis(values.reshape(batch, hv, steps, dv), 1, 2)

    return {
        "data": data,
        "y": logical(outputs),
        "yBound": logical(output_bounds) * BOUND_SAFETY + TINY,
        "state": state.reshape(batch, hv, dv, dk),
        "stateBound": np.broadcast_to(
            (error * BOUND_SAFETY + TINY).reshape(batch, hv, dv)[..., None],
            (batch, hv, dv, dk),
        ),
    }


def _fused_simulate(entry, name, mutation=None):
    """Run gated_delta_fused_chunk's own index arithmetic on the flat buffers.

    Every SIMD group's tiles are evaluated whole and exactly in float64: the
    lanes' (fm, fn) and (fm, fn + 1) pairs cover each 8 x 8 tile once, LOAD_M
    reads SRC[fm * LD + fn] and LOAD_MT reads SRC[fn * LD + fm]. y takes the
    kernel's final conversion to the storage type. ``mutation`` injects one
    defect.
    """
    np = _np()
    dtype, dk, dv, hk, hv, chunk = _parse(entry)
    data = _inputs(entry, name)
    batch, steps = _geometry(entry, name)
    # Reads past a buffer see the dispatch's NaN guards.
    flat = {
        key: np.concatenate([value.reshape(-1), np.full(4 * chunk * hv * dv, np.nan)])
        for key, value in data.items()
    }
    n = np.arange(batch * hv)
    b_idx, hv_idx = n // hv, n % hv
    hk_idx = hv_idx // (hv // hk)
    if mutation == "key-head":
        hk_idx = hv_idx % hk
    t_stride = 1 if mutation == "batch-stride" else steps
    gate_idx = hk_idx if mutation == "gate-head" else hv_idx
    g_ptr = b_idx * t_stride * hv
    qk_ptr = b_idx * t_stride * hk * dk + hk_idx * dk
    v_ptr = b_idx * steps * hv * dv + hv_idx * dv
    y_ptr = b_idx * steps * hv * dv + hv_idx * dv
    rows = np.arange(chunk)
    # i_state + kk, loaded transposed: S_tile[kk / 8][r][c] = state[dv + c][kk + r].
    state_rows = (n[:, None, None] * dv + np.arange(dv)[None, :, None]) * dk
    state = flat["state_in"][state_rows + np.arange(dk)[None, None, :]]
    if mutation == "zero-state":
        state = np.zeros_like(state)
    if mutation == "untransposed-state":
        blocks = state.reshape(-1, dv // 8, 8, dk // 8, 8)
        state = np.swapaxes(blocks, 2, 4).reshape(state.shape)
    y = np.full(batch * steps * hv * dv, np.nan)
    t = 0
    while t < steps:
        valid = min(chunk, steps - t)
        live = rows < valid
        if mutation == "unmasked-tail":
            live = np.ones_like(live)
        step_rows = (t + rows)[None, :, None]
        key_rows = step_rows if mutation != "stale-keys" else rows[None, :, None]
        width_k = np.arange(dk)[None, None, :]
        width_v = np.arange(dv)[None, None, :]
        mask = live[None, :, None]
        keys = np.where(
            mask, flat["k"][qk_ptr[:, None, None] + key_rows * hk * dk + width_k], 0.0
        )
        queries = np.where(
            mask, flat["q"][qk_ptr[:, None, None] + step_rows * hk * dk + width_k], 0.0
        )
        values = np.where(
            mask, flat["v"][v_ptr[:, None, None] + step_rows * hv * dv + width_v], 0.0
        )
        scalar = g_ptr[:, None] + (t + rows)[None, :] * hv
        decays = np.where(live, flat["g"][scalar + gate_idx[:, None]], 1.0)
        betas = np.where(live, flat["beta"][scalar + hv_idx[:, None]], 0.0)
        if mutation == "missing-beta":
            betas = np.where(live, 1.0, 0.0)
        outputs, new_state = _fused_exact(
            np, keys, queries, values, decays, betas, live, state, mutation
        )
        y_index = (
            y_ptr[:, None, None] + (t + rows[:valid])[None, :, None] * hv * dv + width_v
        )
        y[y_index.reshape(-1)] = outputs[:, :valid].reshape(-1)
        state = new_state
        t += chunk
    state_out = np.full(batch * hv * dv * dk, np.nan)
    state_out[(state_rows + np.arange(dk)[None, None, :]).reshape(-1)] = state.reshape(
        -1
    )
    rounded = _round(np, dtype, y)
    if mutation == "truncate-output" and dtype != "float32":
        rounded = _truncate(np, dtype, y)
    return {
        "y": rounded.reshape(batch, steps, hv, dv),
        "state": _round(np, "float32", state_out).reshape(batch, hv, dv, dk),
    }


def _fused_exact(np, keys, queries, values, decays, betas, live, state, mutation=None):
    """PROCESS_CHUNK_SG in exact arithmetic on whole tiles, with defects."""
    chunk = keys.shape[-2]
    rows = np.arange(chunk)
    lower = rows[:, None] >= rows[None, :]
    strict = rows[:, None] > rows[None, :]
    logs = np.where(live, np.log(decays), 0.0)
    prefix = np.cumsum(logs, axis=-1)
    row_decay = np.exp(prefix)
    pair = np.exp(prefix[..., :, None] - prefix[..., None, :])
    column_decay = np.exp(prefix[..., -1:] - prefix)
    chunk_decay = np.exp(prefix[..., -1])
    keys_t = _swap(np, keys)
    kkt = keys @ keys_t
    inverse_mask = lower if mutation == "inclusive-inverse-mask" else strict
    power = np.where(inverse_mask, betas[..., :, None] * kkt, 0.0)
    inverse = np.eye(chunk) - power
    step = 1
    while (1 << step) < chunk and not (mutation == "one-inverse-step" and step > 1):
        power = power @ power
        inverse = inverse + inverse @ power
        step += 1
    state_t = _swap(np, state)
    weights = row_decay[..., None] * (inverse @ (betas[..., None] * keys))
    projected = weights @ state_t
    gated = (
        inverse
        if mutation == "ungated-inverse"
        else np.where(lower, inverse * pair, 0.0)
    )
    delta = gated @ (betas[..., None] * values) - projected
    output_mask = strict if mutation == "strict-output-mask" else lower
    attention = np.where(output_mask, (queries @ keys_t) * pair, 0.0)
    outputs = attention @ delta + (row_decay[..., None] * queries) @ state_t
    decay = 1.0 if mutation == "chunk-decay" else chunk_decay[..., None, None]
    new_state = decay * state_t + (keys_t * column_decay[..., None, :]) @ delta
    return outputs, _swap(np, new_state)


def _truncate(np, dtype, values):
    """Convert to ``dtype`` by truncating the significand toward zero."""
    single = np.asarray(values, dtype=np.float64).astype(np.float32)
    if dtype == "bfloat16":
        bits = single.view(np.uint32) & np.uint32(0xFFFF0000)
        return bits.view(np.float32).astype(np.float64)
    exponent = np.maximum(np.frexp(single)[1], MIN_NORMAL_EXPONENT[dtype])
    quantum = np.ldexp(1.0, exponent - SIGNIFICAND_BITS[dtype])
    return np.trunc(single / quantum) * quantum


def _check(np, dtype, actual, expected, bounds):
    """Check results against the reference.

    Results must be a correct rounding to ``dtype`` of a value within the
    bound. Return the largest fraction of a bound used beyond the final
    rounding's half unit in the last place.
    """
    actual = np.asarray(actual, dtype=np.float64)
    assert actual.shape == expected.shape == bounds.shape
    low = _round(np, dtype, expected - bounds)
    high = _round(np, dtype, expected + bounds)
    good = (actual >= low) & (actual <= high)
    if not good.all():
        bad = np.flatnonzero(~good.reshape(-1))[:8]
        raise AssertionError(
            [
                (
                    int(index),
                    float(actual.reshape(-1)[index]),
                    float(expected.reshape(-1)[index]),
                    float(low.reshape(-1)[index]),
                    float(high.reshape(-1)[index]),
                )
                for index in bad
            ]
        )
    half = np.maximum(_ulp(np, dtype, expected), _ulp(np, dtype, actual)) / 2
    excess = np.maximum(0.0, np.abs(actual - expected) - half)
    return float(np.max(excess / bounds)) if excess.size else 0.0


def _check_outputs(entry, reference, outputs):
    """Check every output of one dispatch; return the bound fractions used."""
    np = _np()
    used = {
        "y": _check(
            np, _dtype(entry), outputs["y"], reference["y"], reference["yBound"]
        ),
        "state": _check(
            np,
            "float32",
            outputs["state"],
            reference["state"],
            reference["stateBound"],
        ),
    }
    if "cache" in reference:
        used["cache"] = _check(
            np,
            "float32",
            outputs["cache"],
            reference["cache"],
            reference["cacheBound"],
        )
    return used


def _config(output, target, entry, constants):
    fused = _fused(entry)
    options = "\n[project.specialization_constants]\n"
    options += f'"{SAVE_STATE_ID}" = {str(constants["save_state"]).lower()}\n'
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
        if fused:
            options += "cooperative_matrix_software_lowering = true\n"
    if target == "directx":
        options += 'relative_wave_shuffle_out_of_range = "self"\n'
    matrices = ""
    if fused:
        # The fused kernel's tiles move between buffers and lanes through
        # MLX's own (fm, fn) lane coordinates.
        matrices = f"""preserve_resource_origins = true
cooperative_matrix_fragment_mapping = "{FRAGMENT_MAPPING}"
cooperative_matrix_fragment_mapping_provenance = "{FRAGMENT_MAPPING_PROVENANCE}"
"""
    if fused and target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[[project.metadata_preconditions]]
source = "{SOURCE}"
entryPoint = "{entry}"
parameter = "T"
minimum = 0
maximum = {T_MAXIMUM}
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
[project.source_options.metal]
max_template_specializations = 256
max_template_materialization_work = 16384
{matrices}{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_contract():
    return {
        name: {
            "batch": batch,
            "steps": steps if steps is not None else "2 * Ckpt + 1",
            "specializations": [
                key for key, names in SPECIALIZATION_CASES.items() if name in names
            ],
        }
        for name, (batch, steps) in CASES.items()
    }


def _fused_validation_contract():
    return {
        "chunk": CHUNK,
        "grid": "[32, Dv / 8, B * Hv]",
        "workgroupSize": list(WORKGROUP_SIZE),
        "columnsPerSimdGroup": FUSED_COLUMNS_PER_GROUP,
        "matrixFragments": {
            "mapping": FRAGMENT_MAPPING,
            "provenance": FRAGMENT_MAPPING_PROVENANCE,
            "softwareLoweringTargets": list(SOFTWARE_SUBGROUP_TARGETS),
        },
        "metadataPreconditions": {
            target: (
                [{"parameter": "T", "minimum": 0, "maximum": T_MAXIMUM}]
                if target in SOFTWARE_SUBGROUP_TARGETS
                else []
            )
            for target in TARGETS
        },
        "specializationConstants": {
            "save_state": {"id": SAVE_STATE_ID, "values": dict(FUSED_SPECIALIZATIONS)}
        },
        "cases": {
            name: {"batch": batch, "steps": steps, "specializations": ["inference"]}
            for name, (batch, steps) in FUSED_CASES.items()
        },
        "errorModel": {
            "unitRoundoff": UNIT_ROUNDOFF,
            "flushAllowancePerOperation": FTZ,
            "productSums": "any-order-gamma(n)-per-element",
            "expUlps": "3 + 2|x|",
            "logAbsoluteError": LOG_ABSOLUTE_ERROR,
            "stateError": "row-2-norm-through-each-step's-g-(I-beta-k-kT)",
            "boundSafety": BOUND_SAFETY,
            "tiny": TINY,
        },
    }


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "grid": "[32, Dv, B * Hv]",
        "stateElementsPerThread": N_PER_T,
        "subgroups": dict(SUBGROUPS),
        "specializationConstants": {
            "save_state": {"id": SAVE_STATE_ID, "values": dict(SPECIALIZATIONS)}
        },
        "indexRangeAssertions": {target: [] for target in TARGETS},
        "cases": _case_contract(),
        "inputs": {
            "queriesAndKeys": "unit-norm-rows-rounded-to-storage",
            "values": "uniform(-1, 1)",
            "decay": "uniform(0.5, 1)",
            "beta": "uniform(0, 1)",
            "stateIn": f"uniform(-{STATE_MAGNITUDE}, {STATE_MAGNITUDE})",
        },
        "errorModel": {
            "unitRoundoff": UNIT_ROUNDOFF,
            "flushAllowancePerOperation": FTZ,
            "reductionRoundings": REDUCTION_ROUNDINGS,
            "stateError": "row-2-norm-through-g-(I-beta-k-kT)",
            "boundSafety": BOUND_SAFETY,
            "tiny": TINY,
        },
        "fusedChunk": _fused_validation_contract(),
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _dispatch_count():
    return sum(len(_cases(entry, key)) for entry, key, _ in VARIANTS)


def _checked_elements(entry):
    """Count the y, final-state and checkpoint elements checked for ``entry``."""
    _, dk, dv, _, hv, _ = _parse(entry)
    total = 0
    specializations = FUSED_SPECIALIZATIONS if _fused(entry) else SPECIALIZATIONS
    for key, save in specializations.items():
        for name in _cases(entry, key):
            batch, steps = _geometry(entry, name)
            total += batch * steps * hv * dv + batch * hv * dv * dk
            if save:
                total += batch * hv * _slots(entry, name) * dv * dk
    return total


def _expected_scope():
    return {
        "instantiatedEntryCount": len(INSTANTIATED),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "sequentialEntryCount": len(SEQ_ENTRIES),
        "fusedChunkEntryCount": len(FUSED_ENTRIES),
        "artifactCountByTarget": len(VARIANTS),
        "dispatchCountByTarget": _dispatch_count(),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_gated_delta_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-gated-delta-contract"
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
    assert "pendingEntries" not in contract
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for entry, records in artifacts.items():
            expected = FUSED_SPECIALIZATIONS if _fused(entry) else SPECIALIZATIONS
            assert tuple(records) == tuple(expected)
            for record in records.values():
                assert set(record) == {"sha256", "sizeBytes"}
                assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    # Metal keeps the function constant and OpenGL a specialization constant,
    # so both specializations of a sequential entry share one artifact;
    # DirectX materializes each.
    for entry in SEQ_ENTRIES:
        for target in ("metal", "opengl"):
            records = contract["artifacts"][target][entry]
            assert len({record["sha256"] for record in records.values()}) == 1
        records = contract["artifacts"]["directx"][entry]
        assert len({record["sha256"] for record in records.values()}) == len(records)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_gated_delta_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "gated_delta_update.current-tree.translation.json" in readme
    assert len(INSTANTIATED) == len(set(INSTANTIATED)) == 90
    assert len(SEQ_ENTRIES) == 72 and len(FUSED_ENTRIES) == 18
    assert ENTRIES == INSTANTIATED
    assert INSTANTIATED[:5] == (
        "seq_gated_delta_float_128_128_24_24_1",
        "seq_gated_delta_float_128_128_24_24_4",
        "seq_gated_delta_float_128_128_24_24_8",
        "seq_gated_delta_float_128_128_24_24_16",
        "gated_delta_fused_chunk_float_128_128_24_24_8",
    )
    assert INSTANTIATED[-1] == "gated_delta_fused_chunk_float16_t_128_128_16_64_8"
    for entries in (SEQ_ENTRIES, FUSED_ENTRIES):
        assert {_dtype(entry) for entry in entries} == set(STORAGE_TYPES.values())
    assert {_parse(entry)[5] for entry in FUSED_ENTRIES} == {CHUNK}
    for dk, dv, hk, hv in SHAPES:
        # supported_gated_delta_shape(): Dk = Dv = 128 and Hv a multiple of Hk.
        assert dk == dv == 128 and hv % hk == 0
        # Four SIMD groups of eight columns tile Dv in the fused kernel.
        assert dv % (WORKGROUP_SIZE[1] * FUSED_COLUMNS_PER_GROUP) == 0
    assert len(VARIANTS) == 2 * len(SEQ_ENTRIES) + len(FUSED_ENTRIES)
    assert len({(entry, key) for entry, key, _ in VARIANTS}) == len(VARIANTS)


def test_gated_delta_cases_follow_mlx_dispatch():
    assert _dispatch_count() == 4 * len(SEQ_ENTRIES) + 3 * len(FUSED_ENTRIES)
    for entry in SEQ_ENTRIES:
        ckpt = _parse(entry)[5]
        assert _geometry(entry, "decode") == (1, 1)
        # gated_delta_chunk_size() takes the sequential kernel up to T = 8.
        assert _geometry(entry, "prefill") == (2, 8)
        assert _geometry(entry, "checkpoints") == (1, 2 * ckpt + 1)
        # The save pass stores n_ckpt slots: one for a single step, three when
        # T = 2 * Ckpt + 1.
        assert _slots(entry, "decode") == 1
        assert _slots(entry, "checkpoints") == 3
    assert set(CHECKPOINTS) == {1, 4, 8, 16}
    for entry in FUSED_ENTRIES:
        chunk = _parse(entry)[5]
        shapes = [_geometry(entry, name) for name in FUSED_CASES]
        # The fused kernel runs only for T > 8: each case has a full chunk.
        assert all(steps > chunk for _, steps in shapes)
        # A one-step tail, a five-step tail over two batches, three chunks.
        assert [steps % chunk for _, steps in shapes] == [1, 5, 0]
        assert max(batch for batch, _ in shapes) == 2
        assert max(steps // chunk for _, steps in shapes) == 3


REPRESENTATIVES = (
    "seq_gated_delta_float_128_128_24_24_1",
    "seq_gated_delta_float_128_128_16_64_4",
    "seq_gated_delta_bfloat16_t_128_128_16_48_8",
    "seq_gated_delta_float16_t_128_128_16_32_16",
    "seq_gated_delta_float16_t_128_128_32_32_4",
)


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_reference_matches_kernel_index_arithmetic(entry):
    # The reference runs the recurrence on the logical tensors; the simulation
    # runs the kernel's own pointer arithmetic on the flat buffers. Exact
    # results, with the kernel's final conversion of y, use a negligible part
    # of the bound.
    for key, save in SPECIALIZATIONS.items():
        for name in SPECIALIZATION_CASES[key]:
            reference = _reference(entry, name, save)
            simulated = _simulate(entry, name, save)
            used = _check_outputs(entry, reference, simulated)
            assert max(used.values()) < 2.0**-10, (name, used)


def _mutations(entry, name, save):
    """Return the defects that change ``entry``'s results on case ``name``."""
    _, _, _, hk, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    mutations = [
        "no-decay",
        "missing-beta",
        "query-for-key",
        "output-before-update",
        "lane-sum",
        "zero-state",
    ]
    if hv != hk:
        mutations += ["key-head", "gate-head"]
    if steps > 1:
        mutations.append("stale-keys")
        if batch > 1:
            mutations.append("batch-stride")
    if save:
        mutations.append("checkpoint-after-decay")
        if _slots(entry, name) > 1:
            mutations += ["checkpoint-slot", "checkpoint-exit"]
    if _dtype(entry) != "float32":
        mutations.append("truncate-output")
    return mutations


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_checks_reject_defects(entry):
    np = _np()
    rejected = set()
    for key, save in SPECIALIZATIONS.items():
        for name in SPECIALIZATION_CASES[key]:
            reference = _reference(entry, name, save)
            exact = _simulate(entry, name, save)
            # An unwritten output fails.
            for output in exact:
                unwritten = {item: value.copy() for item, value in exact.items()}
                unwritten[output].reshape(-1)[7] = np.nan
                with pytest.raises(AssertionError):
                    _check_outputs(entry, reference, unwritten)
            for mutation in _mutations(entry, name, save):
                wrong = _simulate(entry, name, save, mutation)
                with pytest.raises(AssertionError):
                    _check_outputs(entry, reference, wrong)
                rejected.add(mutation)
    expected = {
        "no-decay",
        "missing-beta",
        "query-for-key",
        "output-before-update",
        "lane-sum",
        "zero-state",
        "stale-keys",
        "batch-stride",
        "checkpoint-after-decay",
        "checkpoint-slot",
        "checkpoint-exit",
    }
    _, _, _, hk, hv, _ = _parse(entry)
    if hv != hk:
        expected |= {"key-head", "gate-head"}
    if _dtype(entry) != "float32":
        expected.add("truncate-output")
    assert rejected == expected


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_bounds_stay_meaningful(entry):
    # The bounds stay far below the outputs' own scale: under 2^-12 of the
    # outputs' root mean square over the longest sequence (2^-14 for one
    # step), so a bfloat16 output must be the correctly rounded value except
    # within that distance of a rounding boundary.
    np = _np()
    for key, save in SPECIALIZATIONS.items():
        for name in SPECIALIZATION_CASES[key]:
            reference = _reference(entry, name, save)
            batch, steps = _geometry(entry, name)
            # beta |k|^2 <= 2 keeps I - beta k k^T non-expansive.
            data = reference["data"]
            norms = np.linalg.norm(data["k"], axis=-1) ** 2
            assert np.all(data["beta"][..., None] * norms[:, :, None, :] <= 2.0)
            y_scale = float(np.sqrt(np.mean(reference["y"] ** 2)))
            state_scale = float(np.sqrt(np.mean(reference["state"] ** 2)))
            assert float(np.max(reference["yBound"])) < 2.0**-12 * y_scale, name
            assert float(np.max(reference["stateBound"])) < 2.0**-12 * state_scale, name
            if save:
                assert float(np.max(reference["cacheBound"])) < 2.0**-12 * (state_scale)


SCOPED_VARIANTS = VARIANTS[:8] + VARIANTS[-2:]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,key,constants",
    SCOPED_VARIANTS,
    ids=[f"{e}-{k}" for e, k, _ in SCOPED_VARIANTS],
)
def test_gated_delta_configs_scope_target_options(
    tmp_path, target, entry, key, constants
):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry, constants), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    assert list(config.index_range_assertions) == []
    text = path.read_text(encoding="utf-8")
    assert f'"{entry}" = [32, 4, 1]' in text
    assert f'"{SAVE_STATE_ID}" = {str(SPECIALIZATIONS[key]).lower()}' in text
    software = target in SOFTWARE_SUBGROUP_TARGETS
    fused = _fused(entry)
    assert ("software_subgroup_width = 32" in text) is software
    assert ('relative_wave_shuffle_out_of_range = "self"' in text) is (
        target == "directx"
    )
    # Only the fused kernel holds matrices, and only software subgroups need
    # its lowering and the chunk loop's bound on T.
    assert (f'cooperative_matrix_fragment_mapping = "{FRAGMENT_MAPPING}"' in text) is (
        fused
    )
    assert ("cooperative_matrix_software_lowering = true" in text) is (
        fused and software
    )
    preconditions = [
        (item.entry_point, item.parameter, item.minimum, item.maximum)
        for item in config.metadata_preconditions
    ]
    assert preconditions == ([(entry, "T", 0, T_MAXIMUM)] if fused and software else [])


FUSED_REPRESENTATIVES = (
    "gated_delta_fused_chunk_float_128_128_24_24_8",
    "gated_delta_fused_chunk_float_128_128_16_64_8",
    "gated_delta_fused_chunk_bfloat16_t_128_128_16_48_8",
    "gated_delta_fused_chunk_float16_t_128_128_16_32_8",
    "gated_delta_fused_chunk_float16_t_128_128_32_32_8",
)


@pytest.mark.parametrize("entry", FUSED_REPRESENTATIVES)
def test_gated_delta_fused_reference_matches_kernel_index_arithmetic(entry):
    # The reference steps the recurrence on the logical tensors; the
    # simulation runs the fused kernel's WY form with its own pointers, masks
    # and tile transposes on the flat buffers. Both are exact, so the
    # simulation, with the kernel's final conversion of y, uses a negligible
    # part of the bound.
    for name in FUSED_CASES:
        reference = _reference(entry, name, False)
        simulated = _fused_simulate(entry, name)
        used = _check_outputs(entry, reference, simulated)
        assert max(used.values()) < 2.0**-10, (name, used)


def _fused_mutations(entry, name):
    """Return the defects that change ``entry``'s results on case ``name``."""
    _, _, _, hk, hv, chunk = _parse(entry)
    batch, steps = _geometry(entry, name)
    mutations = [
        "missing-beta",
        "inclusive-inverse-mask",
        "one-inverse-step",
        "ungated-inverse",
        "strict-output-mask",
        "chunk-decay",
        "untransposed-state",
        "zero-state",
        "stale-keys",
    ]
    if hv != hk:
        mutations += ["key-head", "gate-head"]
    if batch > 1:
        mutations.append("batch-stride")
    if steps % chunk:
        mutations.append("unmasked-tail")
    if _dtype(entry) != "float32":
        mutations.append("truncate-output")
    return mutations


@pytest.mark.parametrize("entry", FUSED_REPRESENTATIVES)
def test_gated_delta_fused_checks_reject_defects(entry):
    np = _np()
    rejected = set()
    for name in FUSED_CASES:
        reference = _reference(entry, name, False)
        exact = _fused_simulate(entry, name)
        for output in exact:
            unwritten = {item: value.copy() for item, value in exact.items()}
            unwritten[output].reshape(-1)[7] = np.nan
            with pytest.raises(AssertionError):
                _check_outputs(entry, reference, unwritten)
        for mutation in _fused_mutations(entry, name):
            wrong = _fused_simulate(entry, name, mutation)
            with pytest.raises(AssertionError):
                _check_outputs(entry, reference, wrong)
            rejected.add(mutation)
    expected = {
        "missing-beta",
        "inclusive-inverse-mask",
        "one-inverse-step",
        "ungated-inverse",
        "strict-output-mask",
        "chunk-decay",
        "untransposed-state",
        "zero-state",
        "stale-keys",
        "batch-stride",
        "unmasked-tail",
    }
    _, _, _, hk, hv, _ = _parse(entry)
    if hv != hk:
        expected |= {"key-head", "gate-head"}
    if _dtype(entry) != "float32":
        expected.add("truncate-output")
    assert rejected == expected


@pytest.mark.parametrize("entry", FUSED_REPRESENTATIVES)
def test_gated_delta_fused_bounds_stay_meaningful(entry):
    # The bound allows fast::exp and fast::log their specified error and every
    # tile product any summation order, yet stays under 2^-7 of each output's
    # root mean square. The keys of a head share a direction, so the WY
    # inverse departs far from the identity and every Neumann factor counts.
    np = _np()
    for name in FUSED_CASES:
        reference = _reference(entry, name, False)
        data = reference["data"]
        norms = np.linalg.norm(data["k"], axis=-1) ** 2
        assert np.all(data["beta"][..., None] * norms[:, :, None, :] <= 2.0)
        assert np.all((data["g"] >= 0.5) & (data["g"] <= 1.0))
        for output, bound in (("y", "yBound"), ("state", "stateBound")):
            scale = float(np.sqrt(np.mean(reference[output] ** 2)))
            assert float(np.max(reference[bound])) < 2.0**-7 * scale, (name, output)
        keys = _rows(np, entry, data)["k"][:, :CHUNK]
        products = keys @ np.swapaxes(keys, -1, -2)
        off_diagonal = products[:, ~np.eye(CHUNK, dtype=bool)]
        assert float(np.median(off_diagonal)) > 0.25, name


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native gated-delta gate")
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
    work = tmp_path_factory.mktemp("gated-delta-metal-reference")
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


def _bits16(np, dtype, values):
    """Return the 16-bit storage words of values exact in ``dtype``."""
    values = np.asarray(values, dtype=np.float64)
    if dtype == "float16":
        return values.astype(np.float16).view(np.uint16)
    single = values.astype(np.float32)
    words = (single.view(np.uint32) >> 16).astype(np.uint16)
    return np.where(np.isnan(single), np.uint16(0x7FC0), words)


def _padded(np, values):
    return np.concatenate(
        [np.asarray(values, dtype=np.float64).reshape(-1)]
        + [np.full(INPUT_GUARD_COUNT, np.nan)]
    )


def _encode(np, logical, element_type, encoding, values):
    """Encode logical values for the reflected physical storage."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if element_type == "float32":
        assert encoding is None
        return {
            "dtype": "float32",
            "shape": [values.size],
            "encoding": FLOAT32_BITS,
            "values": values.astype(np.float32).view(np.uint32).tolist(),
        }
    words = _bits16(np, logical, values).tolist()
    if encoding is not None:
        # DirectX binary16 storage takes logical float16 words in uint16 buffers.
        assert element_type == "uint16"
        assert encoding == {"encoding": FLOAT16_BITS, "logicalElementType": logical}
        return {
            "dtype": logical,
            "shape": [values.size],
            "encoding": FLOAT16_BITS,
            "values": words,
        }
    if element_type == "uint16":
        return {"dtype": "uint16", "shape": [values.size], "values": words}
    assert element_type == logical, (element_type, logical)
    return {
        "dtype": logical,
        "shape": [values.size],
        "encoding": FLOAT16_BITS if logical == "float16" else BFLOAT16_BITS,
        "values": words,
    }


def _guard_words(np, logical, element_type, count):
    """Return ``count`` output guard words in the physical storage."""
    if element_type == "float32" and logical != "float32":
        # Widened 16-bit storage holds the guard's exact binary32 value.
        word = int(
            np.asarray(
                _decode_words(np, logical, "uint16", [GUARD_BITS[logical]])[0],
                dtype=np.float32,
            ).view(np.uint32)
        )
    else:
        word = GUARD_BITS[logical]
    return [word] * count


def _decode_words(np, logical, dtype, words):
    words = np.asarray(words, dtype=np.uint64)
    if dtype == "float32":
        return words.astype(np.uint32).view(np.float32).astype(np.float64)
    words = words.astype(np.uint16)
    if logical == "float16":
        return words.view(np.float16).astype(np.float64)
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _decode(np, logical, value):
    assert all(isinstance(word, int) for word in value["values"][:4])
    if value["dtype"] == "float32":
        floats = _decode_words(np, logical, "float32", value["values"])
        if logical != "float32":
            # Widened 16-bit storage must still hold exactly representable
            # values.
            finite = floats[~np.isnan(floats)]
            assert np.array_equal(_quantize(np, logical, finite), finite)
        return floats
    assert value["dtype"] in {logical, "uint16"}, (value["dtype"], logical)
    return _decode_words(np, logical, "uint16", value["values"])


def _member(binding, entry):
    layout = binding["scalarLayout"]
    # DirectX constant members carry the sanitized entry name as a prefix.
    prefix = re.sub(r"_+", "_", entry) + "_"
    member = layout.get("memberName", binding["name"]).removeprefix(prefix)
    return member.removesuffix("Buffer").rstrip("_")


INPUT_MEMBERS = ("q", "k", "v", "state_in", "g", "beta")
OUTPUT_MEMBERS = ("y", "state_out", "state_cache")


def _output_sizes(entry, name, save):
    _, dk, dv, _, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    sizes = {"y": batch * steps * hv * dv, "state_out": batch * hv * dv * dk}
    if not _fused(entry):
        # Without save_state nothing may be stored in the cache.
        sizes["state_cache"] = batch * hv * _slots(entry, name) * dv * dk if save else 0
    return sizes


def _workgroup_count(entry, name):
    """MLX's grid in (32, 4, 1) groups: (32, Dv, B * Hv) or (32, Dv / 8, B * Hv)."""
    _, _, dv, _, hv, _ = _parse(entry)
    batch, _ = _geometry(entry, name)
    rows = dv // FUSED_COLUMNS_PER_GROUP if _fused(entry) else dv
    return [1, rows // WORKGROUP_SIZE[1], batch * hv]


def _dispatch(descriptor, package, target, entry, name, save, data):
    np = _np()
    dtype, _, dv, _, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    sizes = _output_sizes(entry, name, save)
    values, observed, outputs = {}, [], {}
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        member = _member(binding, entry)
        element_type, encoding = _element_layout(binding)
        logical = (
            "float32" if member in {"state_in", "state_out", "state_cache"} else dtype
        )
        if member in INPUT_MEMBERS:
            value = _encode(
                np, logical, element_type, encoding, _padded(np, data[member])
            )
            if logical == "float32" or element_type == "float32":
                assert element_type == "float32" and encoding is None, binding
            observed.append(binding["name"])
        elif member in OUTPUT_MEMBERS:
            count = sizes[member] + GUARD_COUNT
            if element_type == "float32":
                value = {
                    "dtype": "float32",
                    "shape": [count],
                    "encoding": FLOAT32_BITS,
                    "values": _guard_words(np, logical, element_type, count),
                }
            else:
                value = _encode(
                    np,
                    logical,
                    element_type,
                    encoding,
                    _decode_words(np, logical, "uint16", [GUARD_BITS[logical]] * count),
                )
            outputs[member] = (binding["name"], value["dtype"], sizes[member])
        else:
            assert member == "T", member
            assert element_type == "int32", binding
            value = {"dtype": "int32", "shape": [1], "values": [steps]}
            observed.append(binding["name"])
        values[binding["name"]] = value
    assert set(observed) and len(observed) == len(INPUT_MEMBERS) + 1, observed
    assert {"y", "state_out"} <= set(outputs), outputs
    if save:
        assert "state_cache" in outputs, outputs
    if _fused(entry):
        assert set(outputs) == {"y", "state_out"}, outputs
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        {binding: dict(values[binding]) for binding, _, _ in outputs.values()},
        {
            "workgroupCount": _workgroup_count(entry, name),
            "workgroupSize": list(WORKGROUP_SIZE),
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, outputs, len(observed)


def _read_outputs(np, entry, name, save, result, outputs):
    """Decode the written outputs and check every output guard."""
    dtype, dk, dv, _, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    decoded = {}
    for member, (binding, element_type, count) in outputs.items():
        value = result.outputs[binding]
        assert value["dtype"] == element_type
        logical = "float32" if member != "y" else dtype
        words = _decode(np, logical, value)
        guard = _decode_words(
            np,
            logical,
            "uint16" if logical != "float32" else "float32",
            [GUARD_BITS[logical]],
        )[0]
        tail = words[count:]
        assert tail.size == GUARD_COUNT and np.all(tail == guard), (member, tail[:4])
        decoded[member] = words[:count]
    shaped = {
        "y": decoded["y"].reshape(batch, steps, hv, dv),
        "state": decoded["state_out"].reshape(batch, hv, dv, dk),
    }
    if save:
        shaped["cache"] = decoded["state_cache"].reshape(
            batch * hv, _slots(entry, name), dv, dk
        )
    return shaped


def _metal_payload(np, dtype, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if dtype == "float32":
        return values.astype(np.float32).tobytes()
    return _bits16(np, dtype, values).astype("<u2").tobytes()


def _upstream_metal(np, work, reference, entry, name, save, data):
    """Run the unchanged upstream kernel with gated_delta_update.cpp's slots."""
    runner, library = reference
    dtype, dk, dv, _, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    sizes = _output_sizes(entry, name, save)
    guard16 = _decode_words(np, dtype, "uint16", [GUARD_BITS[dtype]])[0]
    guard32 = _decode_words(np, "float32", "float32", [GUARD_BITS["float32"]])[0]
    payloads = {
        index: _metal_payload(
            np,
            "float32" if member == "state_in" else dtype,
            _padded(np, data[member]),
        )
        for index, member in enumerate(INPUT_MEMBERS)
    }
    payloads[6] = _metal_payload(
        np,
        dtype,
        np.full(sizes["y"] + GUARD_COUNT, guard16 if dtype != "float32" else guard32),
    )
    payloads[7] = _metal_payload(
        np, "float32", np.full(sizes["state_out"] + GUARD_COUNT, guard32)
    )
    payloads[8] = np.asarray([steps], dtype="<i4").tobytes()
    if save:
        # Slot 9 is gated off by the function constant for inference.
        payloads[9] = _metal_payload(
            np, "float32", np.full(sizes["state_cache"] + GUARD_COUNT, guard32)
        )
    files = []
    for index in range(max(payloads) + 1):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payloads[index])
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": _workgroup_count(entry, name),
        "workgroupSize": list(WORKGROUP_SIZE),
        "simdWidth": SIMD_SIZE,
        "boolConstants": [{"index": SAVE_STATE_ID, "value": save}],
    }
    request_path = work / f"{name}-upstream-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request_path, output], work, f"{name}-upstream")
    )
    for index in (*range(len(INPUT_MEMBERS)), 8):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    words = {}
    for index, member, logical in (
        (6, "y", dtype),
        (7, "state_out", "float32"),
        *(((9, "state_cache", "float32"),) if save else ()),
    ):
        raw = (output / f"buffer-{index}.bin").read_bytes()
        width = "<u4" if logical == "float32" else "<u2"
        stored = np.frombuffer(raw, dtype=width).astype(np.uint64)
        values = _decode_words(
            np, logical, "float32" if logical == "float32" else "uint16", stored
        )
        count = sizes[member]
        guard = guard32 if logical == "float32" else guard16
        assert np.all(values[count:] == guard), member
        words[member] = stored[:count]
    for path in files:
        Path(path).unlink()
    return device, words


def _assert_fused_lowering(text, target):
    matrix = f"crossgl_cooperative_matrix_fragment_float_8x8_32x2_{FRAGMENT_MAPPING}"
    if target == "metal":
        # The fused kernel never reads save_state, so no constant survives.
        assert "function_constant" not in text
        for spelling in (
            "simdgroup_matrix<float, 8, 8>",
            "make_filled_simdgroup_matrix<float, 8, 8>(",
            "simdgroup_multiply_accumulate(",
            "simdgroup_load(",
            "simdgroup_store(",
            ".thread_elements()[",
            "simd_prefix_inclusive_sum(",
            "fast::exp(",
            "fast::log(",
        ):
            assert spelling in text, spelling
    elif target == "opengl":
        assert "simdgroup" not in text and "gl_Subgroup" not in text
        assert "subgroupInclusiveAdd" not in text and "constant_id" not in text
        for helper in (
            "crossglSoftwareSubgroupPrefixInclusiveSumFloat(",
            f"{matrix}_fill(",
            f"{matrix}_fragment_offset(",
            f"{matrix}_multiply_accumulate(",
        ):
            assert helper in text, helper
    else:
        assert "Wave" not in text and "save_state" not in text
        for helper in (
            "__crossgl_software_subgroup_PrefixInclusiveSum_float(",
            "__crossgl_software_matrix_mma(",
            "__crossgl_software_matrix_fragment_offset(",
        ):
            assert helper in text, helper


def _assert_lowering(text, target, entry, constants):
    assert "unsupported" not in text
    if _fused(entry):
        _assert_fused_lowering(text, target)
        return
    save = constants["save_state"]
    if target == "metal":
        assert (
            "constant bool save_state [[function_constant(200)]];" in text
        ), "function constant"
        assert "[[function_constant(save_state)]]" in text
        assert text.count("simd_sum(") == 2
    elif target == "opengl":
        # Specialized when the pipeline is created, as on Metal.
        assert "layout(constant_id = 200) const bool save_state = " in text
        assert "subgroupAdd" not in text and "gl_Subgroup" not in text
        assert text.count("crossglSoftwareSubgroupSumFloat(") == 3
    else:
        assert f"static const bool save_state = {str(save).lower()};" in text
        assert "WaveActiveSum" not in text and "[WaveSize" not in text
        assert text.count("__crossgl_software_subgroup_sum_float(") == 3


def _words(np, dtype, values):
    """Return the flat storage words of decoded values for bit comparisons."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if dtype == "float32":
        return values.astype(np.float32).view(np.uint32).astype(np.uint64)
    return _bits16(np, dtype, values).astype(np.uint64)


@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS, ids=[f"{e}-{k}" for e, k, _ in VARIANTS]
)
def test_current_mlx_gated_delta_native_parity(
    current_mlx, metal_reference, tmp_path, entry, key, constants
):
    np = _np()
    root, target = current_mlx
    dtype = _dtype(entry)
    save = constants["save_state"]
    record = _contract()["artifacts"][target][entry][key]
    with tempfile.TemporaryDirectory(prefix=".current-gdu-", dir=root) as path:
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
            } == ({} if _fused(entry) else {SAVE_STATE_ID: save})
            executor = _executor(target)
            evidence = []
            try:
                for name in _cases(entry, key):
                    reference = _reference(entry, name, save)
                    request, outputs, observed = _dispatch(
                        descriptor,
                        package,
                        target,
                        entry,
                        name,
                        save,
                        reference["data"],
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    assert result.status == "ok", (result.status, result.details)
                    actual = _read_outputs(np, entry, name, save, result, outputs)
                    used = _check_outputs(entry, reference, actual)
                    observations = result.details["inputObservations"]
                    assert len(observations) == observed
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, words = _upstream_metal(
                            np,
                            work,
                            metal_reference,
                            entry,
                            name,
                            save,
                            reference["data"],
                        )
                        identical = bool(
                            np.array_equal(words["y"], _words(np, dtype, actual["y"]))
                            and np.array_equal(
                                words["state_out"],
                                _words(np, "float32", actual["state"]),
                            )
                            and (
                                not save
                                or np.array_equal(
                                    words["state_cache"],
                                    _words(np, "float32", actual["cache"]),
                                )
                            )
                        )
                        upstream = {
                            "device": device,
                            "bitIdenticalToTranslated": identical,
                        }
                    batch, steps = _geometry(entry, name)
                    evidence.append(
                        {
                            "case": name,
                            "batch": batch,
                            "steps": steps,
                            "checkpointSlots": _slots(entry, name) if save else 0,
                            "largestBounds": {
                                "y": float(np.max(reference["yBound"])),
                                "state": float(np.max(reference["stateBound"])),
                            },
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
                        "subgroups": SUBGROUPS[target],
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if target == "metal":
                # The same arithmetic on the same SIMD reduction: the
                # translated kernel reproduces the upstream kernel bit for bit.
                assert [
                    case["case"]
                    for case in evidence
                    if not case["upstream"]["bitIdenticalToTranslated"]
                ] == []
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.bin"),
            )


def test_ci_requires_native_gated_delta_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        CURRENT_PARITY_JOB,
        assert_workflow_triggers,
        current_parity_step,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX gated delta native parity", group="sequence"
    )
    # The 144 sequential dispatch sets alone took 3076 s on WARP; the fused
    # chunk entries need the rest of the sequence group's budget.
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "--timeout-seconds 7200",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/gated-delta",
        "--junitxml=mlx-current-results/gated-delta.xml",
        "demos/integrations/mlx/tests/kernels/test_current_gated_delta.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_gated_delta.py"
    )
    install = workflow_text.workflow_job_step_section(
        workflow, CURRENT_PARITY_JOB, "Install CrossTL"
    )
    # The reference and the checks run on numpy arrays.
    assert "numpy==" in install
