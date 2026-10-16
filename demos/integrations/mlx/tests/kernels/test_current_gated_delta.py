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
# addresses their fragments through thread_elements(); the Metal frontend does
# not yet resolve that typedef as a cooperative matrix, so no target
# translates it.
PENDING_REASON = (
    "simdgroup_float8x8 tiles accessed through thread_elements() do not "
    "translate yet"
)
PENDING_DIAGNOSTIC = "project.translate.metal-struct-method"
ENTRIES = tuple(entry for entry in INSTANTIATED if entry.startswith("seq_"))
PENDING_ENTRIES = tuple(entry for entry in INSTANTIATED if not entry.startswith("seq_"))
SAVE_STATE_ID = 200
# gated_delta_update.cpp clears save_state (function constant 200) for
# inference and sets it for the backward's sequential save pass.
SPECIALIZATIONS = {"inference": False, "save": True}
VARIANTS = tuple(
    (entry, key, {"save_state": value})
    for entry in ENTRIES
    for key, value in SPECIALIZATIONS.items()
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
    match = re.fullmatch(
        r"seq_gated_delta_(float|bfloat16_t|float16_t)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)",
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

    return {
        # q, k: [B, T, Hk, Dk]; v: [B, T, Hv, Dv]; g, beta: [B, T, Hv];
        # state_in: [B, Hv, Dv, Dk] in float.
        "q": _quantize(np, dtype, unit("q", (batch, steps, hk, dk))),
        "k": _quantize(np, dtype, unit("k", (batch, steps, hk, dk))),
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
    options = "\n[project.specialization_constants]\n"
    options += f'"{SAVE_STATE_ID}" = {str(constants["save_state"]).lower()}\n'
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
    if target == "directx":
        options += 'relative_wave_shuffle_out_of_range = "self"\n'
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
            "specializations": [
                key for key, names in SPECIALIZATION_CASES.items() if name in names
            ],
        }
        for name, (batch, steps) in CASES.items()
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
        "results": "correct-rounding-of-a-value-within-the-error-bound",
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": GUARD_COUNT,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _dispatch_count():
    return sum(len(SPECIALIZATION_CASES[key]) for _, key, _ in VARIANTS)


def _checked_elements(entry):
    """Count the y, final-state and checkpoint elements checked for ``entry``."""
    _, dk, dv, _, hv, _ = _parse(entry)
    total = 0
    for key, names in SPECIALIZATION_CASES.items():
        for name in names:
            batch, steps = _geometry(entry, name)
            total += batch * steps * hv * dv + batch * hv * dv * dk
            if SPECIALIZATIONS[key]:
                total += batch * hv * _slots(entry, name) * dv * dk
    return total


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
    pending = contract["pendingEntries"]
    assert tuple(pending) == PENDING_ENTRIES
    for record in pending.values():
        assert record == {
            "kind": "fused-chunk",
            "reason": PENDING_REASON,
            "translatesTo": [],
            "diagnostics": {target: PENDING_DIAGNOSTIC for target in TARGETS},
        }
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for records in artifacts.values():
            assert tuple(records) == tuple(SPECIALIZATIONS)
            for record in records.values():
                assert set(record) == {"sha256", "sizeBytes"}
                assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    # Metal keeps the function constant and OpenGL a specialization constant,
    # so both specializations of an entry share one artifact; DirectX
    # materializes each.
    for entry in ENTRIES:
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
    assert len(ENTRIES) == 72 and len(PENDING_ENTRIES) == 18
    assert INSTANTIATED[:5] == (
        "seq_gated_delta_float_128_128_24_24_1",
        "seq_gated_delta_float_128_128_24_24_4",
        "seq_gated_delta_float_128_128_24_24_8",
        "seq_gated_delta_float_128_128_24_24_16",
        "gated_delta_fused_chunk_float_128_128_24_24_8",
    )
    assert INSTANTIATED[-1] == "gated_delta_fused_chunk_float16_t_128_128_16_64_8"
    assert {_dtype(entry) for entry in ENTRIES} == set(STORAGE_TYPES.values())
    for dk, dv, hk, hv in SHAPES:
        # supported_gated_delta_shape(): Dk = Dv = 128 and Hv a multiple of Hk.
        assert dk == dv == 128 and hv % hk == 0
    assert len(VARIANTS) == 2 * len(ENTRIES)
    assert len({(entry, key) for entry, key, _ in VARIANTS}) == len(VARIANTS)


def test_gated_delta_cases_follow_mlx_dispatch():
    assert _dispatch_count() == 4 * len(ENTRIES)
    for entry in ENTRIES:
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


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "entry,key,constants", VARIANTS[:8], ids=[f"{e}-{k}" for e, k, _ in VARIANTS[:8]]
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
    return {
        "y": batch * steps * hv * dv,
        "state_out": batch * hv * dv * dk,
        # Without save_state nothing may be stored in the cache.
        "state_cache": batch * hv * _slots(entry, name) * dv * dk if save else 0,
    }


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
        "workgroupCount": [1, dv // WORKGROUP_SIZE[1], batch * hv],
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


def _assert_lowering(text, target, constants):
    assert "unsupported" not in text
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
            assert artifact["generatedHash"] == {
                "algorithm": "sha256",
                "value": record["sha256"],
            }
            assert artifact["generatedSizeBytes"] == record["sizeBytes"]
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
            generated = root / artifact["path"]
            _assert_lowering(generated.read_text(encoding="utf-8"), target, constants)
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            assert {
                item["id"]: item["value"]
                for item in descriptor["specializationConstants"]
            } == {SAVE_STATE_ID: save}
            executor = _executor(target)
            evidence = []
            try:
                for name in SPECIALIZATION_CASES[key]:
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


@pytest.mark.parametrize("entry", PENDING_ENTRIES)
def test_current_mlx_gated_delta_fused_chunk_stays_pending(current_mlx, entry):
    # The contract records why the fused chunk entries are not covered; the
    # record must stay true, so a translator that starts accepting them fails
    # here until the entries join the gate.
    root, target = current_mlx
    record = _contract()["pendingEntries"][entry]
    with tempfile.TemporaryDirectory(prefix=".current-gdu-", dir=root) as path:
        work = Path(path)
        config = work / "crosstl.toml"
        config.write_text(
            _config(f"{work.name}/out", target, entry, {"save_state": False}),
            encoding="utf-8",
        )
        report = translate_project(
            load_project_config(root, config),
            format_output=False,
            validate=True,
            run_toolchains=False,
        ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    codes = {item["code"] for item in report["diagnostics"]}
    assert record["diagnostics"][target] in codes, codes
    (diagnostic,) = (
        item
        for item in report["diagnostics"]
        if item["code"] == record["diagnostics"][target]
    )
    assert "thread_elements" in diagnostic["message"]
    assert "simdgroup_float8x8" in diagnostic["message"]


def test_ci_requires_native_gated_delta_parity():
    from tests.ci_helpers import (
        CURRENT_PARITY_JOB,
        assert_workflow_triggers,
        current_parity_step,
    )
    from tools import ci_coverage

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX gated delta native parity", group="sequence"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
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
    install = ci_coverage.workflow_job_step_section(
        workflow, CURRENT_PARITY_JOB, "Install CrossTL"
    )
    # The reference and the checks run on numpy arrays.
    assert "numpy==" in install
