"""Translate and execute the pinned MLX gated-delta backward kernels on all backends."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
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
from demos.integrations.mlx.tests.kernels.test_current_gated_delta import (
    BOUND_SAFETY,
    CHECKPOINTS,
    FTZ,
    INPUT_GUARD_COUNT,
    N_PER_T,
    REDUCTION_ROUNDINGS,
    SHAPES,
    SIMD_SIZE,
    SOFTWARE_SUBGROUP_TARGETS,
    STORAGE_TYPES,
    SUBGROUPS,
    TINY,
    UNIT_ROUNDOFF,
    WORKGROUP_SIZE,
    _check,
    _decode,
    _decode_words,
    _encode,
    _gamma,
    _member,
    _metal_payload,
    _padded,
    _quantize,
    _uniform,
    _words,
)
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/gated_delta_update_vjp.metal"
SOURCE_SHA256 = "59590c0bc3dbeb6ce1c808660b43abdf462a3f43f16abcb484220605b8aa86e7"
HEADER = "mlx/backend/metal/kernels/gated_delta_update_vjp.h"
HEADER_SHA256 = "717aa81794efca14c589371c83a16e86f2d0b5e0660778b0e09e34abe46e63c6"
ATOMIC_HEADER = "mlx/backend/metal/kernels/atomic.h"
ATOMIC_HEADER_SHA256 = (
    "4c35ea2798a2335502865247aee878149fc9ada0d7e84c05d771baef0c7fcc60"
)
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GATED_DELTA_VJP"
CONTRACT_PATH = (
    ROOT
    / "demos/integrations/mlx/contracts/gated_delta_update_vjp.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# instantiate_gated_delta_vjp's order: each storage type, each (Dk, Dv, Hk, Hv)
# that supported_gated_delta_shape() accepts, then every checkpoint interval.
# The host names keep the C++ type spellings.
ENTRIES = tuple(
    f"seq_gated_delta_vjp_{type_name}_{dk}_{dv}_{hk}_{hv}_{ckpt}"
    for type_name in STORAGE_TYPES
    for dk, dv, hk, hv in SHAPES
    for ckpt in CHECKPOINTS
)
# gated_delta_update.cpp runs the sequential backward only when no NAX chunk
# is available, and gated_delta_ckpt() then returns 1: the host stores the
# state entering every step. The other intervals are instantiated in the
# metallib and replay their segments from sparser checkpoints.
HOST_CHECKPOINT_INTERVALS = (1,)
# The kernel adds its query, key, decay and beta gradients to zero-filled
# buffers through mlx_atomic<float>, so their summation order is not fixed.
ATOMIC_OUTPUTS = ("dq", "dk", "dg", "db")
# The value gradient is stored by lane 0 of each row's SIMD group and the
# initial-state gradient by every thread; neither depends on thread order.
STORED_OUTPUTS = ("dv", "dh")
OUTPUT_MEMBERS = ("dq", "dk", "dv", "dg", "db", "dh")
INPUT_MEMBERS = ("q", "k", "v", "g", "b", "cot_o", "cot_h", "state_cache")
# The storage-typed inputs; the cotangent of the final state, the checkpoint
# cache and every gradient are float.
TYPED_INPUTS = ("q", "k", "v", "g", "b", "cot_o")
# OpenGL indexes 32-bit storage arrays. mlx_atomic_fetch_add_explicit takes a
# size_t offset into a rebased atomic pointer; this host precondition bounds
# the complete element offset of every atomic update to signed int range.
SIGNED_MAX = 2**31 - 1
OPENGL_INDEX_RANGE_ASSERTIONS = ("object + offset",)
# name -> (batch B, sequence T). None means 2 * Ckpt + 1 steps: the replay
# walks three checkpoint segments, the last holding a single step.
CASES = {
    "single": (1, 1),
    "batched": (2, 3),
    "segments": (1, None),
}
COTANGENT_STATE_MAGNITUDE = 0.5
INITIAL_STATE_MAGNITUDE = 0.5


def _np():
    if os.environ.get(REQUIRE_ENV) == "1":
        import numpy

        return numpy
    return pytest.importorskip("numpy")


def _parse(entry):
    match = re.fullmatch(
        r"seq_gated_delta_vjp_(float|bfloat16_t|float16_t)"
        r"_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)",
        entry,
    )
    assert match, entry
    type_name, *numbers = match.groups()
    dk, dv, hk, hv, ckpt = (int(item) for item in numbers)
    return STORAGE_TYPES[type_name], dk, dv, hk, hv, ckpt


def _dtype(entry):
    return _parse(entry)[0]


def _geometry(entry, name):
    batch, steps = CASES[name]
    if steps is None:
        steps = 2 * _parse(entry)[5] + 1
    return batch, steps


def _slots(entry, name):
    """Return n_ckpt, the checkpoint slots gated_delta_vjp_seq replays from."""
    ckpt = _parse(entry)[5]
    return (_geometry(entry, name)[1] + ckpt - 1) // ckpt


def _heads(np, entry):
    _, _, _, hk, hv, _ = _parse(entry)
    return np.arange(hv) // (hv // hk)


def _inputs(entry, name):
    """Return the logical input tensors of one case as MLX lays them out."""
    np = _np()
    dtype, dk, dv, hk, hv, ckpt = _parse(entry)
    batch, steps = _geometry(entry, name)
    label = f"vjp:{dtype}:{dk}:{dv}:{hk}:{hv}:{ckpt}:{name}:{steps}"

    def uniform(tag, shape):
        return _uniform(np, f"{label}:{tag}", shape)

    def unit(tag, shape):
        # MLX normalizes queries and keys; |k| stays 1 within the storage
        # rounding, so beta |k|^2 <= 2 holds for every beta in [0, 1].
        raw = 2.0 * uniform(tag, shape) - 1.0
        return raw / np.linalg.norm(raw, axis=-1, keepdims=True)

    data = {
        # q, k: [B, T, Hk, Dk]; v, cot_o: [B, T, Hv, Dv]; g, b: [B, T, Hv];
        # cot_h: [B, Hv, Dv, Dk] in float.
        "q": _quantize(np, dtype, unit("q", (batch, steps, hk, dk))),
        "k": _quantize(np, dtype, unit("k", (batch, steps, hk, dk))),
        "v": _quantize(np, dtype, 2.0 * uniform("v", (batch, steps, hv, dv)) - 1.0),
        "g": _quantize(np, dtype, 0.5 + 0.5 * uniform("g", (batch, steps, hv))),
        "b": _quantize(np, dtype, uniform("b", (batch, steps, hv))),
        "cot_o": _quantize(
            np, dtype, 2.0 * uniform("cot_o", (batch, steps, hv, dv)) - 1.0
        ),
        "cot_h": _quantize(
            np,
            "float32",
            COTANGENT_STATE_MAGNITUDE
            * (2.0 * uniform("cot_h", (batch, hv, dv, dk)) - 1.0),
        ),
    }
    # The forward save pass stores the state entering every Ckpt-th step;
    # here the exact recurrence from a random initial state, rounded to float.
    heads = _heads(np, entry)
    state = INITIAL_STATE_MAGNITUDE * (2.0 * uniform("h0", (batch, hv, dv, dk)) - 1.0)
    slots = []
    for t in range(steps):
        if t % ckpt == 0:
            slots.append(state.copy())
        keys = data["k"][:, t][:, heads]
        decayed = state * data["g"][:, t][..., None, None]
        kv = np.einsum("bhvk,bhk->bhv", decayed, keys)
        delta = (data["v"][:, t] - kv) * data["b"][:, t][..., None]
        state = decayed + delta[..., None] * keys[:, :, None, :]
    # state_cache: [B * Hv, n_ckpt, Dv, Dk].
    data["state_cache"] = _quantize(np, "float32", np.stack(slots, axis=2)).reshape(
        batch * hv, len(slots), dv, dk
    )
    return data


def _step(np, data, heads, t, state, error, dk):
    """Return one exact forward step and bounds on its computed rounding.

    ``state`` is the exact state entering step ``t``; the computed state has a
    row 2-norm error of at most ``error``. As in the forward gate, the
    computed step equals the exact step of the computed state plus its own
    rounding, and ``g (I - beta k k^T)`` does not amplify the carried error
    while ``beta |k|^2 <= 2``.
    """
    unit = UNIT_ROUNDOFF
    gamma = _gamma(REDUCTION_ROUNDINGS)
    root = math.sqrt(dk)
    keys = data["k"][:, t][:, heads]
    decay = data["g"][:, t][..., None]
    beta = data["b"][:, t][..., None]
    values = data["v"][:, t]
    key_norm = np.linalg.norm(keys, axis=-1)[..., None]
    contraction = np.maximum(1.0, np.abs(1.0 - beta * key_norm**2))
    decayed = state * decay[..., None]
    magnitude = decay * (np.linalg.norm(state, axis=-1) + error)
    r1 = unit * magnitude + FTZ * root
    decayed_norm = magnitude + r1
    decayed_error = decay * error + r1
    key_sum = np.einsum("bhvk,bhk->bhv", np.abs(decayed), np.abs(keys))
    r_kv = gamma * (key_sum + key_norm * decayed_error) + 2 * dk * FTZ
    kv = np.einsum("bhvk,bhk->bhv", decayed, keys)
    kv_error = key_norm * decayed_error + r_kv
    # |v - kv_hat| before the subtraction rounds.
    difference = np.abs(values - kv) + kv_error
    r_d = (2 * unit + unit * unit) * np.abs(beta) * difference + (
        1.0 + (1.0 + unit) * np.abs(beta)
    ) * FTZ
    delta_magnitude = np.abs(beta) * difference + r_d
    r2 = (
        unit * key_norm * delta_magnitude
        + unit * (decayed_norm + key_norm * delta_magnitude)
        + 2 * FTZ * root
    ) / (1.0 - unit)
    rho = contraction * r1 + key_norm * (r_d + np.abs(beta) * r_kv) + r2
    residual = values - kv
    return {
        "decayed": decayed,
        "decayedError": decayed_error,
        "residual": residual,
        # u_hat = fl(v - kv_hat).
        "residualError": kv_error + unit * difference + FTZ,
        "state": decayed + (residual * beta)[..., None] * keys[:, :, None, :],
        "stateError": decay * contraction * error + rho,
    }


def _accumulated(np, terms, errors, axes, count):
    """Sum ``terms`` over ``axes`` as ``count`` atomic adds to a zero buffer.

    Each computed term is within ``errors`` of its exact value. Recursive
    summation in any order adds count - 1 roundings to every term (the first
    add to zero is exact), and each add may flush a subnormal sum.
    """
    total = terms.sum(axis=axes)
    bound = (
        errors.sum(axis=axes)
        + _gamma(count - 1) * (np.abs(terms) + errors).sum(axis=axes)
        + count * FTZ
    )
    return total, bound


def _reference(entry, name):
    """Return exact gradients and per-element bounds on the kernel's rounding.

    Each segment replays forward from its own checkpoint slot, so the replay
    error restarts at zero there, while the cotangent state s_hat carries its
    error through every step. Its update s_hat <- g (I - beta k k^T)
    (s_hat + co q) is the transpose of the forward step and does not amplify
    the carried error either. The four atomic gradients add the rounding of
    any summation order of their contributions.
    """
    np = _np()
    _, dk, dv, hk, hv, ckpt = _parse(entry)
    data = _inputs(entry, name)
    batch, steps = _geometry(entry, name)
    repeats = hv // hk
    heads = _heads(np, entry)
    unit = UNIT_ROUNDOFF
    gamma = _gamma(REDUCTION_ROUNDINGS)
    root = math.sqrt(dk)
    slots = _slots(entry, name)
    cache = data["state_cache"].reshape(batch, hv, slots, dv, dk)
    adjoint = data["cot_h"].copy()
    adjoint_error = np.zeros((batch, hv, dv))
    gradients = {
        "dq": np.zeros((batch, steps, hk, dk)),
        "dk": np.zeros((batch, steps, hk, dk)),
        "dv": np.zeros((batch, steps, hv, dv)),
        "dg": np.zeros((batch, steps, hv)),
        "db": np.zeros((batch, steps, hv)),
    }
    bounds = {key: np.zeros_like(value) for key, value in gradients.items()}
    for slot in reversed(range(slots)):
        t0 = slot * ckpt
        length = min(ckpt, steps - t0)
        state = cache[:, :, slot].copy()
        error = np.zeros((batch, hv, dv))
        segment = []
        for t in range(t0, t0 + length):
            segment.append((state, error))
            step = _step(np, data, heads, t, state, error, dk)
            state, error = step["state"], step["stateError"]
        for j in reversed(range(length)):
            t = t0 + j
            entry_state, entry_error = segment[j]
            # The backward recomputes the step from the replayed entry state.
            step = _step(np, data, heads, t, entry_state, entry_error, dk)
            queries = data["q"][:, t][:, heads]
            keys = data["k"][:, t][:, heads]
            decay = data["g"][:, t][..., None]
            beta = data["b"][:, t][..., None]
            cot = data["cot_o"][:, t]
            key_norm = np.linalg.norm(keys, axis=-1)[..., None]
            query_norm = np.linalg.norm(queries, axis=-1)[..., None]
            contraction = np.maximum(1.0, np.abs(1.0 - beta * key_norm**2))
            # s_hat += co q.
            h1 = adjoint + cot[..., None] * queries[:, :, None, :]
            h1_norm = np.linalg.norm(h1, axis=-1)
            h1_error = (
                (adjoint_error + unit * np.abs(cot) * query_norm) * (1.0 + unit)
                + unit * h1_norm
                + 3 * FTZ * root
            )
            # w = s_hat . k over the 32 lanes.
            w = np.einsum("bhvk,bhk->bhv", h1, keys)
            w_sum = np.einsum("bhvk,bhk->bhv", np.abs(h1), np.abs(keys))
            r_w = gamma * (w_sum + key_norm * h1_error) + 2 * dk * FTZ
            w_error = key_norm * h1_error + r_w
            w_max = np.abs(w) + w_error
            # dv = beta w, stored by lane 0.
            gradients["dv"][:, t] = beta * w
            bounds["dv"][:, t] = (
                np.abs(beta) * w_error + unit * np.abs(beta) * w_max + FTZ
            )
            residual = step["residual"]
            residual_error = step["residualError"]
            residual_max = np.abs(residual) + residual_error
            # db += w u from every row of the head.
            term = w * residual
            term_error = (
                w_max * residual_error
                + np.abs(residual) * w_error
                + unit * w_max * residual_max
                + FTZ
            )
            gradients["db"][:, t], bounds["db"][:, t] = _accumulated(
                np, term, term_error, (2,), dv
            )
            # dq += co s_t from every row of every value head of the key head.
            computed = step["state"]
            computed_error = step["stateError"][..., None]
            terms = cot[..., None] * computed
            errors = (
                np.abs(cot)[..., None] * computed_error
                + unit * np.abs(cot)[..., None] * (np.abs(computed) + computed_error)
                + FTZ
            )
            shape = (batch, hk, repeats * dv, dk)
            gradients["dq"][:, t], bounds["dq"][:, t] = _accumulated(
                np,
                terms.reshape(shape),
                errors.reshape(shape),
                (2,),
                repeats * dv,
            )
            # dk += beta (u s_hat - w s_dec).
            decayed = step["decayed"]
            decayed_error = step["decayedError"][..., None]
            first = residual[..., None] * h1
            first_error = (
                residual_max[..., None] * h1_error[..., None]
                + np.abs(h1) * residual_error[..., None]
                + unit * residual_max[..., None] * (np.abs(h1) + h1_error[..., None])
                + FTZ
            )
            second = w[..., None] * decayed
            second_error = (
                w_max[..., None] * decayed_error
                + np.abs(decayed) * w_error[..., None]
                + unit * w_max[..., None] * (np.abs(decayed) + decayed_error)
                + FTZ
            )
            difference = first - second
            difference_error = (
                first_error
                + second_error
                + unit * (np.abs(difference) + first_error + second_error)
                + FTZ
            )
            factor = np.abs(beta)[..., None]
            terms = beta[..., None] * difference
            errors = (
                factor * difference_error
                + unit * factor * (np.abs(difference) + difference_error)
                + FTZ
            )
            gradients["dk"][:, t], bounds["dk"][:, t] = _accumulated(
                np,
                terms.reshape(shape),
                errors.reshape(shape),
                (2,),
                repeats * dv,
            )
            # s_hat -= beta w k.
            h2 = h1 - (beta * w)[..., None] * keys[:, :, None, :]
            h2_norm = np.linalg.norm(h2, axis=-1)
            scaled_max = np.abs(beta) * w_max * (1.0 + unit) + FTZ
            h2_carried = (
                contraction * h1_error
                + key_norm * (np.abs(beta) * r_w + unit * np.abs(beta) * w_max + FTZ)
                + unit * scaled_max * key_norm
            )
            h2_error = h2_carried + unit * (h2_norm + h2_carried) + 3 * FTZ * root
            # dg += s_hat . s_prev from every row of the head.
            row = np.einsum("bhvk,bhvk->bhv", h2, entry_state)
            row_sum = np.einsum("bhvk,bhvk->bhv", np.abs(h2), np.abs(entry_state))
            cross = (
                h2_error * np.linalg.norm(entry_state, axis=-1)
                + h2_norm * entry_error
                + h2_error * entry_error
            )
            row_error = cross + gamma * (row_sum + cross) + 2 * dk * FTZ
            gradients["dg"][:, t], bounds["dg"][:, t] = _accumulated(
                np, row, row_error, (2,), dv
            )
            # s_hat *= g.
            adjoint = h2 * decay[..., None]
            adjoint_error = (
                decay * h2_error + unit * decay * (h2_norm + h2_error) + FTZ * root
            )
    gradients["dh"] = adjoint
    bounds["dh"] = np.broadcast_to(adjoint_error[..., None], adjoint.shape)
    return {
        "data": data,
        "gradients": gradients,
        "bounds": {key: value * BOUND_SAFETY + TINY for key, value in bounds.items()},
    }


def _reduce(np, values, single):
    """Sum the last axis left to right, rounding each add when ``single``."""
    if not single:
        return values.sum(axis=-1)
    total = values[..., 0]
    for index in range(1, values.shape[-1]):
        total = total + values[..., index]
    return total


def _simulate(entry, name, mutation=None, single=False):
    """Run gated_delta_vjp_seq's own index arithmetic on the flat buffers.

    Every thread (lane, dv, n) is evaluated in float64, or with ``single`` in
    binary32 with every operation rounded, each lane summing its four terms
    in order, lanes reduced in order and atomic adds applied in thread order.
    ``mutation`` injects one defect.
    """
    np = _np()
    _, dk, dv, hk, hv, ckpt = _parse(entry)
    data = _inputs(entry, name)
    batch, steps = _geometry(entry, name)
    real = np.float32 if single else np.float64
    flat = {key: value.reshape(-1).astype(real) for key, value in data.items()}
    n = np.arange(batch * hv)[:, None, None]
    dv_idx = np.arange(dv)[None, :, None]
    lane = np.arange(SIMD_SIZE)[None, None, :]
    b_idx = n // hv
    hv_idx = n % hv
    hk_idx = hv_idx // (hv // hk)
    if mutation == "key-head":
        hk_idx = hv_idx % hk
    t_stride = 1 if mutation == "batch-stride" else steps
    qk_stride, v_stride, g_stride = hk * dk, hv * dv, hv
    q_base = b_idx * t_stride * qk_stride + hk_idx * dk
    v_base = b_idx * t_stride * v_stride + hv_idx * dv
    g_base = b_idx * t_stride * g_stride + hv_idx
    gate_base = g_base
    if mutation == "gate-head":
        gate_base = b_idx * t_stride * g_stride + hk_idx
    n_ckpt = (steps + ckpt - 1) // ckpt
    i = np.arange(N_PER_T)
    # s_idx = n_per_t * dk_idx + i: [threads..., lanes, i].
    s_idx = (N_PER_T * lane)[..., None] + i
    row = ((n * dv + dv_idx) * dk)[..., None]
    s_hat = flat["cot_h"][row + s_idx]
    if mutation == "zero-cotangent-state":
        s_hat = np.zeros_like(s_hat)
    out = {
        "dq": np.zeros(batch * steps * hk * dk, real),
        "dk": np.zeros(batch * steps * hk * dk, real),
        "dv": np.full(batch * steps * hv * dv, np.nan, real),
        "dg": np.zeros(batch * steps * hv, real),
        "db": np.zeros(batch * steps * hv, real),
        "dh": np.full(batch * hv * dv * dk, np.nan, real),
    }

    def dot(left, right):
        # Per-lane partial sums: [threads..., lanes].
        lanes = _reduce(np, left * right, single)
        if mutation == "lane-sum":
            return lanes[..., None]
        return _reduce(np, lanes, single)[..., None, None]

    def add(key, index, values):
        index = np.broadcast_to(index, values.shape).reshape(-1)
        if mutation == "overwrite-dq" and key == "dq":
            out[key][index] = values.reshape(-1)
        else:
            np.add.at(out[key], index, values.reshape(-1))

    for seg_idx in range(n_ckpt - 1, -1, -1):
        t0 = seg_idx * ckpt
        seg_len = min(ckpt, steps - t0)
        slot = seg_idx
        if mutation == "wrong-slot":
            slot = (seg_idx + 1) % n_ckpt
        c_state = (n * n_ckpt * dv * dk + slot * dv * dk + dv_idx * dk)[..., None]
        s = flat["state_cache"][c_state + s_idx]
        seg = []
        for j in range(seg_len):
            t = t0 + j
            if mutation != "exit-state-for-entry":
                seg.append(s)
            gamma = flat["g"][gate_base + t * g_stride][..., None]
            beta = flat["b"][gate_base + t * g_stride][..., None]
            k_t = flat["k"][(q_base + t * qk_stride)[..., None] + s_idx]
            s = s * gamma
            kv = dot(s, k_t)
            value = flat["v"][v_base + t * v_stride + dv_idx][..., None]
            delta = beta * (value - kv)
            s = s + delta * k_t
            if mutation == "exit-state-for-entry":
                seg.append(s)
        order = range(seg_len - 1, -1, -1)
        if mutation == "forward-backward":
            order = range(seg_len)
        for j in order:
            t = t0 + j
            gamma = flat["g"][gate_base + t * g_stride][..., None]
            beta = flat["b"][gate_base + t * g_stride][..., None]
            q_t = flat["q"][(q_base + t * qk_stride)[..., None] + s_idx]
            k_t = flat["k"][(q_base + t * qk_stride)[..., None] + s_idx]
            dq_t = (q_base + t * qk_stride)[..., None] + s_idx
            co = flat["cot_o"][v_base + t * v_stride + dv_idx][..., None]
            s_prev = seg[j]
            s_dec = s_prev * gamma
            kv = dot(s_dec, k_t)
            if mutation != "missing-output-cotangent":
                s_hat = s_hat + co * q_t
            w = dot(s_hat, q_t if mutation == "query-for-key" else k_t)
            out["dv"][(v_base + t * v_stride + dv_idx)[..., 0]] = (
                w if mutation == "dv-without-beta" else beta * w
            )[..., 0, 0]
            value = flat["v"][v_base + t * v_stride + dv_idx][..., None]
            u = value - kv
            delta = beta * u
            add(
                "db",
                (g_base + t * g_stride)[..., 0],
                (w if mutation == "db-without-residual" else w * u)[..., 0, 0],
            )
            s_t = s_dec + delta * k_t
            add("dq", dq_t, co * (s_dec if mutation == "dq-from-decayed" else s_t))
            if mutation == "dk-sign":
                contrib = beta * (w * s_dec - u * s_hat)
            else:
                contrib = beta * (u * s_hat - w * s_dec)
            updated = s_hat - beta * w * k_t
            if mutation == "dk-after-update":
                contrib = beta * (u * updated - w * s_dec)
            add("dk", dq_t, contrib)
            read = s_hat if mutation == "dgamma-before-update" else updated
            s_hat = updated
            dgamma = _reduce(np, _reduce(np, read * s_prev, single), single)
            add("dg", (g_base + t * g_stride)[..., 0], dgamma)
            if mutation != "no-cotangent-decay":
                s_hat = s_hat * gamma
    out["dh"][row + s_idx] = s_hat
    return {
        "dq": out["dq"].astype(np.float64).reshape(batch, steps, hk, dk),
        "dk": out["dk"].astype(np.float64).reshape(batch, steps, hk, dk),
        "dv": out["dv"].astype(np.float64).reshape(batch, steps, hv, dv),
        "dg": out["dg"].astype(np.float64).reshape(batch, steps, hv),
        "db": out["db"].astype(np.float64).reshape(batch, steps, hv),
        "dh": out["dh"].astype(np.float64).reshape(batch, hv, dv, dk),
    }


def _check_outputs(reference, outputs):
    """Check every gradient of one dispatch; return the bound fractions used."""
    np = _np()
    return {
        key: _check(
            np,
            "float32",
            outputs[key],
            reference["gradients"][key],
            reference["bounds"][key],
        )
        for key in OUTPUT_MEMBERS
    }


def _config(output, target, entry):
    options = ""
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
    if target == "directx":
        options += 'relative_wave_shuffle_out_of_range = "self"\n'
    if target == "opengl":
        for expression in OPENGL_INDEX_RANGE_ASSERTIONS:
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
[project.source_options.metal]
max_template_specializations = 256
max_template_materialization_work = 16384
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_contract():
    return {
        name: {
            "batch": batch,
            "steps": steps if steps is not None else "2 * Ckpt + 1",
        }
        for name, (batch, steps) in CASES.items()
    }


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "grid": "[32, Dv, B * Hv]",
        "stateElementsPerThread": N_PER_T,
        "subgroups": dict(SUBGROUPS),
        "specializationConstants": {},
        "hostCheckpointIntervals": list(HOST_CHECKPOINT_INTERVALS),
        "atomicOutputs": {
            "outputs": list(ATOMIC_OUTPUTS),
            "initialValue": 0.0,
            "summationOrder": "unspecified",
            "lowering": {
                "metal": "atomic_fetch_add_explicit on atomic_float",
                "opengl": "atomicCompSwap loop on uint words",
                "directx": "InterlockedCompareExchangeFloatBitwise loop",
            },
        },
        "storedOutputs": list(STORED_OUTPUTS),
        "indexRangeAssertions": {
            "metal": [],
            "opengl": [
                {"expression": expression, "minimum": 0, "maximum": SIGNED_MAX}
                for expression in OPENGL_INDEX_RANGE_ASSERTIONS
            ],
            "directx": [],
        },
        "cases": _case_contract(),
        "inputs": {
            "queriesAndKeys": "unit-norm-rows-rounded-to-storage",
            "values": "uniform(-1, 1)",
            "decay": "uniform(0.5, 1)",
            "beta": "uniform(0, 1)",
            "outputCotangent": "uniform(-1, 1)",
            "stateCotangent": (
                f"uniform(-{COTANGENT_STATE_MAGNITUDE}, {COTANGENT_STATE_MAGNITUDE})"
            ),
            "stateCache": (
                "exact-forward-states-every-Ckpt-steps-from-uniform("
                f"-{INITIAL_STATE_MAGNITUDE}, {INITIAL_STATE_MAGNITUDE})"
                "-rounded-to-float"
            ),
        },
        "errorModel": {
            "unitRoundoff": UNIT_ROUNDOFF,
            "flushAllowancePerOperation": FTZ,
            "reductionRoundings": REDUCTION_ROUNDINGS,
            "replayError": "row-2-norm-through-g-(I-beta-k-kT)-from-each-slot",
            "cotangentStateError": "row-2-norm-through-g-(I-beta-k-kT)",
            "atomicAccumulation": "recursive-summation-in-any-order",
            "boundSafety": BOUND_SAFETY,
            "tiny": TINY,
        },
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
        "upstreamBitIdenticalOutputs": list(STORED_OUTPUTS),
    }


def _output_sizes(entry, name):
    _, dk, dv, hk, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    return {
        "dq": batch * steps * hk * dk,
        "dk": batch * steps * hk * dk,
        "dv": batch * steps * hv * dv,
        "dg": batch * steps * hv,
        "db": batch * steps * hv,
        "dh": batch * hv * dv * dk,
    }


def _checked_elements(entry):
    """Count the gradient elements checked for ``entry``."""
    return sum(sum(_output_sizes(entry, name).values()) for name in CASES)


def _expected_scope():
    return {
        "instantiatedEntryCount": len(ENTRIES),
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "artifactCountByTarget": len(ENTRIES),
        "dispatchCountByTarget": len(ENTRIES) * len(CASES),
        "checkedElementCountByEntry": {
            entry: _checked_elements(entry) for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_gated_delta_vjp_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-gated-delta-vjp-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "headers": {HEADER: HEADER_SHA256, ATOMIC_HEADER: ATOMIC_HEADER_SHA256},
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
        # Every entry is its own artifact.
        assert len({record["sha256"] for record in artifacts.values()}) == len(ENTRIES)
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_gated_delta_vjp_entries_follow_mlx_instantiation():
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "gated_delta_update_vjp.current-tree.translation.json" in readme
    assert len(ENTRIES) == len(set(ENTRIES)) == 72
    assert ENTRIES[:5] == (
        "seq_gated_delta_vjp_float_128_128_24_24_1",
        "seq_gated_delta_vjp_float_128_128_24_24_4",
        "seq_gated_delta_vjp_float_128_128_24_24_8",
        "seq_gated_delta_vjp_float_128_128_24_24_16",
        "seq_gated_delta_vjp_float_128_128_32_32_1",
    )
    assert ENTRIES[-1] == "seq_gated_delta_vjp_float16_t_128_128_16_64_16"
    assert {_dtype(entry) for entry in ENTRIES} == set(STORAGE_TYPES.values())
    for dk, dv, hk, hv in SHAPES:
        # supported_gated_delta_shape(): Dk = Dv = 128 and Hv a multiple of Hk.
        assert dk == dv == 128 and hv % hk == 0
    assert set(HOST_CHECKPOINT_INTERVALS) <= set(CHECKPOINTS)


def test_gated_delta_vjp_cases_follow_mlx_dispatch():
    for entry in ENTRIES:
        ckpt = _parse(entry)[5]
        assert _geometry(entry, "single") == (1, 1)
        assert _geometry(entry, "batched") == (2, 3)
        assert _geometry(entry, "segments") == (1, 2 * ckpt + 1)
        assert _slots(entry, "single") == 1
        # The batched case crosses segment boundaries at Ckpt = 1 and fills
        # part of one segment otherwise; the segments case replays three.
        assert _slots(entry, "batched") == (3 if ckpt == 1 else 1)
        assert _slots(entry, "segments") == 3
        data = _inputs(entry, "segments")
        _, dk, dv, _, hv, _ = _parse(entry)
        assert data["state_cache"].shape == (hv, 3, dv, dk)


REPRESENTATIVES = (
    "seq_gated_delta_vjp_float_128_128_16_16_1",
    "seq_gated_delta_vjp_float_128_128_16_32_4",
    "seq_gated_delta_vjp_bfloat16_t_128_128_16_48_8",
    "seq_gated_delta_vjp_float16_t_128_128_24_24_16",
    "seq_gated_delta_vjp_float16_t_128_128_16_64_4",
)


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_vjp_reference_matches_kernel_index_arithmetic(entry):
    # The reference runs the recurrence on the logical tensors; the simulation
    # runs the kernel's own pointer arithmetic on the flat buffers. Exact
    # results use a negligible part of the bound.
    for name in CASES:
        reference = _reference(entry, name)
        simulated = _simulate(entry, name)
        used = _check_outputs(reference, simulated)
        assert max(used.values()) < 2.0**-10, (name, used)


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_vjp_bounds_cover_binary32_evaluation(entry):
    # The kernel's arithmetic in binary32, one rounding per operation in the
    # kernel's own order, stays well inside the bounds.
    for name in CASES:
        reference = _reference(entry, name)
        single = _simulate(entry, name, single=True)
        used = _check_outputs(reference, single)
        assert max(used.values()) < 0.25, (name, used)
        # A binary32 evaluation is not exact, so the bounds are not vacuous.
        assert max(used.values()) > 0.0, (name, used)


def _mutations(entry, name):
    """Return the defects that change ``entry``'s results on case ``name``."""
    _, _, _, hk, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    mutations = [
        "no-cotangent-decay",
        "missing-output-cotangent",
        "query-for-key",
        "dv-without-beta",
        "db-without-residual",
        "dq-from-decayed",
        "dk-sign",
        "dk-after-update",
        "dgamma-before-update",
        "lane-sum",
        "zero-cotangent-state",
        "overwrite-dq",
        "exit-state-for-entry",
    ]
    if hv != hk:
        mutations += ["key-head", "gate-head"]
    if steps > 1 and batch > 1:
        mutations.append("batch-stride")
    if _slots(entry, name) > 1:
        mutations.append("wrong-slot")
    if min(_parse(entry)[5], steps) > 1:
        mutations.append("forward-backward")
    return mutations


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_vjp_checks_reject_defects(entry):
    np = _np()
    rejected = set()
    for name in CASES:
        reference = _reference(entry, name)
        exact = _simulate(entry, name)
        # An unwritten or corrupted gradient fails.
        for output in exact:
            unwritten = {item: value.copy() for item, value in exact.items()}
            unwritten[output].reshape(-1)[3] = np.nan
            with pytest.raises(AssertionError):
                _check_outputs(reference, unwritten)
        for mutation in _mutations(entry, name):
            wrong = _simulate(entry, name, mutation)
            with pytest.raises(AssertionError):
                _check_outputs(reference, wrong)
            rejected.add(mutation)
    expected = {
        "no-cotangent-decay",
        "missing-output-cotangent",
        "query-for-key",
        "dv-without-beta",
        "db-without-residual",
        "dq-from-decayed",
        "dk-sign",
        "dk-after-update",
        "dgamma-before-update",
        "lane-sum",
        "zero-cotangent-state",
        "overwrite-dq",
        "exit-state-for-entry",
        "batch-stride",
        "wrong-slot",
    }
    _, _, _, hk, hv, ckpt = _parse(entry)
    if hv != hk:
        expected |= {"key-head", "gate-head"}
    if ckpt > 1:
        expected.add("forward-backward")
    assert rejected == expected


# Largest bound over each gradient's root mean square. The atomic gradients
# sum 128 to 512 contributions in an unspecified order and the decay
# gradient adds 128 dot products of 128 terms, so their worst-case bounds
# are wider than those of the value and state gradients.
BOUND_SCALE = {
    "dq": 2.0**-7,
    "dk": 2.0**-7,
    "dv": 2.0**-11,
    "dg": 2.0**-7,
    "db": 2.0**-8,
    "dh": 2.0**-11,
}


@pytest.mark.parametrize("entry", REPRESENTATIVES)
def test_gated_delta_vjp_bounds_stay_meaningful(entry):
    np = _np()
    for name in CASES:
        reference = _reference(entry, name)
        data = reference["data"]
        heads = _heads(np, entry)
        # beta |k|^2 <= 2 keeps I - beta k k^T non-expansive.
        norms = np.linalg.norm(data["k"][:, :, heads], axis=-1) ** 2
        assert np.all(data["b"] * norms <= 2.0)
        for key, scale in BOUND_SCALE.items():
            value = reference["gradients"][key]
            rms = float(np.sqrt(np.mean(value**2)))
            assert float(np.max(reference["bounds"][key])) < scale * rms, (
                name,
                key,
            )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ENTRIES[:8])
def test_gated_delta_vjp_configs_scope_target_options(tmp_path, target, entry):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry), encoding="utf-8")
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
            for expression in OPENGL_INDEX_RANGE_ASSERTIONS
        ]
        if target == "opengl"
        else []
    )
    text = path.read_text(encoding="utf-8")
    assert f'"{entry}" = [32, 4, 1]' in text
    assert "specialization_constants" not in text
    software = target in SOFTWARE_SUBGROUP_TARGETS
    assert ("software_subgroup_width = 32" in text) is software
    assert ('relative_wave_shuffle_out_of_range = "self"' in text) is (
        target == "directx"
    )


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
    for path, digest in (
        (SOURCE, SOURCE_SHA256),
        (HEADER, HEADER_SHA256),
        (ATOMIC_HEADER, ATOMIC_HEADER_SHA256),
    ):
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("gated-delta-vjp-metal-reference")
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


def _initial_words(member, size):
    """Return an output's initial float32 words: zeros or guards, then guards."""
    guard = GUARD_BITS["float32"]
    # The atomic gradients accumulate into zero-filled buffers, as in
    # gated_delta_update.cpp; the stored gradients overwrite every element,
    # so an unwritten one keeps its guard value.
    payload = [0] * size if member in ATOMIC_OUTPUTS else [guard] * size
    return payload + [guard] * GUARD_COUNT


def _dispatch(descriptor, package, target, entry, name, data):
    np = _np()
    dtype, _, dv, _, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    sizes = _output_sizes(entry, name)
    values, observed, outputs = {}, [], {}
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        member = _member(binding, entry)
        element_type, encoding = _element_layout(binding)
        if member in INPUT_MEMBERS:
            logical = dtype if member in TYPED_INPUTS else "float32"
            value = _encode(
                np, logical, element_type, encoding, _padded(np, data[member])
            )
            if logical == "float32":
                assert element_type == "float32" and encoding is None, binding
            observed.append(binding["name"])
        elif member in OUTPUT_MEMBERS:
            words = _initial_words(member, sizes[member])
            assert encoding is None, binding
            if member in ATOMIC_OUTPUTS:
                assert binding["scalarLayout"]["componentCount"] == 1, binding
                (field,) = binding["scalarLayout"]["structMembers"]
                assert field["name"] == "val" and field["offsetBytes"] == 0
            if element_type == "uint32":
                # OpenGL backs mlx_atomic<float> with uint words that hold
                # the float bits.
                assert target == "opengl" and member in ATOMIC_OUTPUTS, binding
                value = {"dtype": "uint32", "shape": [len(words)], "values": words}
            else:
                assert element_type == "float32", binding
                value = {
                    "dtype": "float32",
                    "shape": [len(words)],
                    "encoding": FLOAT32_BITS,
                    "values": words,
                }
            outputs[member] = (binding["name"], value["dtype"], sizes[member])
        else:
            assert member == "T", member
            assert element_type == "int32", binding
            value = {"dtype": "int32", "shape": [1], "values": [steps]}
            observed.append(binding["name"])
        values[binding["name"]] = value
    assert len(observed) == len(INPUT_MEMBERS) + 1, observed
    assert set(outputs) == set(OUTPUT_MEMBERS), outputs
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        values,
        {binding: dict(values[binding]) for binding, _, _ in outputs.values()},
        {
            "workgroupCount": [1, dv // WORKGROUP_SIZE[1], batch * hv],
            "workgroupSize": list(WORKGROUP_SIZE),
        },
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, outputs, len(observed)


def _float_words(np, value):
    """Decode float32 words carried as float32 or uint32 runtime values."""
    assert all(isinstance(word, int) for word in value["values"][:4])
    if value["dtype"] == "uint32":
        return _decode_words(np, "float32", "float32", value["values"])
    return _decode(np, "float32", value)


def _shaped(entry, name, decoded):
    _, dk, dv, hk, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    shapes = {
        "dq": (batch, steps, hk, dk),
        "dk": (batch, steps, hk, dk),
        "dv": (batch, steps, hv, dv),
        "dg": (batch, steps, hv),
        "db": (batch, steps, hv),
        "dh": (batch, hv, dv, dk),
    }
    return {key: decoded[key].reshape(shapes[key]) for key in OUTPUT_MEMBERS}


def _read_outputs(np, entry, name, result, outputs):
    """Decode the written gradients and check every output guard."""
    guard = _decode_words(np, "float32", "float32", [GUARD_BITS["float32"]])[0]
    decoded = {}
    for member, (binding, element_type, count) in outputs.items():
        value = result.outputs[binding]
        assert value["dtype"] == element_type
        words = _float_words(np, value)
        tail = words[count:]
        assert tail.size == GUARD_COUNT and np.all(tail == guard), (member, tail[:4])
        decoded[member] = words[:count]
    return _shaped(entry, name, decoded)


def _upstream_metal(np, work, reference, entry, name, data):
    """Run the unchanged upstream kernel with gated_delta_update.cpp's slots."""
    runner, library = reference
    dtype, _, dv, _, hv, _ = _parse(entry)
    batch, steps = _geometry(entry, name)
    sizes = _output_sizes(entry, name)
    payloads = {
        index: _metal_payload(
            np,
            dtype if member in TYPED_INPUTS else "float32",
            _padded(np, data[member]),
        )
        for index, member in enumerate(INPUT_MEMBERS)
    }
    payloads[8] = np.asarray([steps], dtype="<i4").tobytes()
    for index, member in enumerate(OUTPUT_MEMBERS, start=9):
        payloads[index] = np.asarray(
            _initial_words(member, sizes[member]), dtype="<u4"
        ).tobytes()
    files = []
    for index in range(len(payloads)):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payloads[index])
        files.append(str(path))
    request = {
        "buffers": files,
        "workgroupCount": [1, dv // WORKGROUP_SIZE[1], batch * hv],
        "workgroupSize": list(WORKGROUP_SIZE),
        "simdWidth": SIMD_SIZE,
    }
    request_path = work / f"{name}-upstream-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request_path, output], work, f"{name}-upstream")
    )
    for index in range(len(INPUT_MEMBERS) + 1):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    guard = _decode_words(np, "float32", "float32", [GUARD_BITS["float32"]])[0]
    words, decoded = {}, {}
    for index, member in enumerate(OUTPUT_MEMBERS, start=9):
        raw = (output / f"buffer-{index}.bin").read_bytes()
        stored = np.frombuffer(raw, dtype="<u4").astype(np.uint64)
        values = _decode_words(np, "float32", "float32", stored)
        count = sizes[member]
        assert values.size == count + GUARD_COUNT, member
        assert np.all(values[count:] == guard), member
        words[member] = stored[:count]
        decoded[member] = values[:count]
    for path in files:
        Path(path).unlink()
    return device, words, _shaped(entry, name, decoded)


def _assert_lowering(text, target, entry):
    assert "unsupported" not in text
    ckpt = _parse(entry)[5]
    # Each thread keeps its slice of every entry state of a segment in
    # registers.
    assert f"float seg[{ckpt}][4];" in text
    if target == "metal":
        assert (
            "atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>"
            "(&object[offset].val), val, memory_order_relaxed);"
        ) in text
        assert text.count("mlx_atomic_fetch_add_explicit_float(") == 5
        assert text.count("simd_sum(") == 4
    elif target == "opengl":
        assert "subgroupAdd" not in text and "gl_Subgroup" not in text
        assert text.count("crossglSoftwareSubgroupSumFloat(") == 5
        # One compare-and-swap loop per atomic gradient buffer.
        assert text.count("atomicCompSwap(") == 4
        for member in ATOMIC_OUTPUTS:
            assert (
                f"buffer {member}Buffer {{ crossgl_words_mlx_atomic_float_void "
                f"{member}[]; }};"
            ) in text
    else:
        assert "WaveActiveSum" not in text and "[WaveSize" not in text
        assert text.count("__crossgl_software_subgroup_sum_float(") == 5
        # One bitwise compare-exchange loop shared by the four atomic buffers.
        assert text.count("InterlockedCompareExchangeFloatBitwise(") == 2
        for member in ATOMIC_OUTPUTS:
            assert f"RWStructuredBuffer<mlx_atomic_float_void> {member}" in text


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_gated_delta_vjp_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    np = _np()
    root, target = current_mlx
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-gduv-", dir=root) as path:
        work = Path(path)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry), encoding="utf-8"
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
            assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
            generated = root / artifact["path"]
            _assert_lowering(generated.read_text(encoding="utf-8"), target, entry)
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            assert descriptor["specializationConstants"] == []
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES:
                    reference = _reference(entry, name)
                    request, outputs, observed = _dispatch(
                        descriptor, package, target, entry, name, reference["data"]
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    assert result.status == "ok", (result.status, result.details)
                    actual = _read_outputs(np, entry, name, result, outputs)
                    used = _check_outputs(reference, actual)
                    observations = result.details["inputObservations"]
                    assert len(observations) == observed
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, words, upstream_outputs = _upstream_metal(
                            np, work, metal_reference, entry, name, reference["data"]
                        )
                        # The unchanged kernel meets the same bounds.
                        upstream_used = _check_outputs(reference, upstream_outputs)
                        identical = {
                            key: bool(
                                np.array_equal(
                                    words[key], _words(np, "float32", actual[key])
                                )
                            )
                            for key in OUTPUT_MEMBERS
                        }
                        upstream = {
                            "device": device,
                            "bitIdenticalToTranslated": identical,
                            "boundFractionUsed": upstream_used,
                        }
                    batch, steps = _geometry(entry, name)
                    evidence.append(
                        {
                            "case": name,
                            "batch": batch,
                            "steps": steps,
                            "checkpointSlots": _slots(entry, name),
                            "largestBounds": {
                                key: float(np.max(reference["bounds"][key]))
                                for key in OUTPUT_MEMBERS
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
                        "dtype": _dtype(entry),
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
                # The stored gradients do not depend on thread order: the
                # translated kernel reproduces the upstream kernel bit for bit.
                assert [
                    (case["case"], key)
                    for case in evidence
                    for key in STORED_OUTPUTS
                    if not case["upstream"]["bitIdenticalToTranslated"][key]
                ] == []
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.bin"),
            )


def test_ci_requires_native_gated_delta_vjp_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow,
        "Prove current MLX gated delta backward native parity",
        group="backward",
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/gated-delta-vjp",
        "--junitxml=mlx-current-results/gated-delta-vjp.xml",
        "demos/integrations/mlx/tests/kernels/test_current_gated_delta_vjp.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow,
        "demos/integrations/mlx/tests/kernels/test_current_gated_delta_vjp.py",
    )
