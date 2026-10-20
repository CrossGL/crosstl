"""Translate the pinned MLX steel attention entry and compare native results.

``steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat32`` computes
``softmax(scale * Q K^T + mask) V`` with threadgroup Q, K and V tiles, simdgroup
matrix fragments and an online base-2 softmax. The gate freezes the aligned,
float-mask constants that ``scaled_dot_product_attention.cpp`` selects for
sequence lengths that are multiples of the 32-row blocks, fills ``AttnParams``
and ``AttnMaskParams`` as that host code does, and dispatches the upstream
``(NQ, H, B)`` grid of ``(32, 4, 1)`` threadgroups.

Every case must match a float64 evaluation of the definition on the decoded
float32 inputs at ``rtol=atol=1e-5``, the tolerance of the other attention gates,
and must leave the output guards and the unused ``lse`` buffer unchanged. On
Metal the original MLX source, compiled and dispatched with the same constants
and buffers, must meet the same reference.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    pack_storage_records,
    translate_project,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeConstantBinding,
    RuntimeExecutionState,
    RuntimeSpecializationConstant,
)
from tests.runtime_helpers import _prepare_native_package, compile_metal, run
from tests.test_translator.test_native_loader_dispatch_integration import _executor

np = pytest.importorskip("numpy")

MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal"
SOURCE_SHA256 = "ade46f59394199b17ae4a04323d2840838d594d22173490df732f1c3b0b8f605"
HEADER_HASHES = {
    "kernels/steel_attention.h": (
        "0519f2ec5f6b2917a7685df85ecc5555ca8acd7e070a14c9945ad28adadec36c"
    ),
    "attn.h": "67722214ff487ec95df675e2f135ccd7fbe639b6eb6ab98b14bcefa58d66ba95",
    "loader.h": "e435f413d71d44ca4ad72a59d192046be77388f63936179c58882286f4cc5dab",
    "mma.h": "6f22866c76a1f15cf27681d3a8700f3419412a954d00e0810c87f7dac82ddeab",
    "params.h": "f03c6ed73d7298fa69b75a49e7eec54954b6fafd59530a1f175934215412ed80",
    "transforms.h": "6479114663c5923a7ee2f9fc1cf9d5e99e680d47d793251749989056a8a63141",
}
ENTRY = "steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat32"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_ATTENTION"
BQ, BK, HEAD_DIM = 32, 32, 64
WORKGROUP = (32, 4, 1)
CONSTANT_IDS = {
    "align_Q": 200,
    "align_K": 201,
    "has_mask": 300,
    "do_causal": 301,
    "has_sinks": 302,
    "save_lse": 303,
}
SPECIALIZATIONS = {
    "align_Q": True,
    "align_K": True,
    "has_mask": True,
    "do_causal": False,
    "has_sinks": False,
    "save_lse": False,
}
SIGNED_MAX = 2**31 - 1
PLATFORM_TARGETS = {"darwin": "metal", "win32": "directx", "linux": "opengl"}
GATE_TOOLS = {
    "metal": ("xcrun",),
    "directx": ("dxc",),
    "opengl": ("glslangValidator",),
}
GUARD = -98765.0
GUARDS = 4
TOLERANCE = 1e-5
ORIGINAL_FLAGS = (
    "-std=metal3.2",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
    "-Wno-c++17-extensions",
    "-Wno-c++20-extensions",
)
BINDINGS = ("Q", "K", "V", "O", "params", "mask_params", "mask", "sinks", "lse")


@dataclass(frozen=True)
class Case:
    """One dispatch: shapes, the Q/O layout and how the mask broadcasts."""

    batches: int
    heads: int
    kv_heads: int
    query_length: int
    key_length: int
    sequence_major: bool
    mask_per_head: bool
    seed: int

    @property
    def name(self):
        layout = "blhd" if self.sequence_major else "bhld"
        mask = "head-mask" if self.mask_per_head else "shared-mask"
        return (
            f"b{self.batches}-h{self.heads}-kv{self.kv_heads}-"
            f"q{self.query_length}-k{self.key_length}-{layout}-{mask}"
        )


CASES = (
    Case(1, 2, 2, 64, 64, False, True, 7),
    # Grouped query heads, a longer key sequence (qL_off = 64), the
    # sequence-major Q/O layout MLX passes through strides and a mask
    # broadcast over heads with a zero head stride.
    Case(2, 4, 2, 32, 96, True, False, 11),
)


def _opengl_index_assertions(source=SOURCE):
    """GLSL indexes with 32-bit ints, so 64-bit buffer offsets need ranges.

    The kernel offsets Q, K, V, O and the mask by ``int64_t`` strides from
    ``AttnParams``; MLX buffers stay below 2^31 elements. The threadgroup tiles
    need no assertion: their view offsets are proven from the loaders and the
    fragment coordinates.
    """
    return tuple(
        {
            "source": source,
            "expression": "reference.offset + index",
            "function": f"crosstl_resource_{access}",
            "minimum": 0,
            "maximum": SIGNED_MAX,
        }
        for access in ("load_float", "store_float")
    )


def _config(root, output, target="metal"):
    assert target in {"metal", "directx", "opengl"}
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output,
        entry_points={SOURCE: (ENTRY,)},
        specialization_constants=dict(SPECIALIZATIONS),
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: WORKGROUP},
        index_range_assertions=(
            _opengl_index_assertions() if target == "opengl" else ()
        ),
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
                "target_options": {
                    "directx": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": 32,
                        "relative_wave_shuffle_out_of_range": "self",
                    },
                    "opengl": {
                        "cooperative_matrix_software_lowering": True,
                        "software_subgroup_width": 32,
                    },
                },
            }
        },
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_attention_target_contract_is_explicit(tmp_path, target):
    config = _config(tmp_path, "out", target)
    assert list(config.targets) == [target]
    assert config.workgroup_size_rules == {SOURCE: ("32", "4", "1")}
    assert config.freeze_specialization_constants is True
    assert config.specialization_constants == SPECIALIZATIONS
    assert not config.workgroup_access_assertions
    ranges = [item.to_json() for item in config.index_range_assertions]
    if target != "opengl":
        assert not ranges
        return
    assert ranges == list(_opengl_index_assertions())
    assert {item["function"] for item in ranges} == {
        "crosstl_resource_load_float",
        "crosstl_resource_store_float",
    }
    assert all(
        item["minimum"] == 0 and item["maximum"] == SIGNED_MAX for item in ranges
    )


def _strides(shape):
    strides, size = [], 1
    for extent in reversed(shape):
        strides.insert(0, size)
        size *= extent
    return strides


def _case_data(case):
    """Inputs, AttnParams as the host fills them, and the float64 reference."""
    assert case.query_length % BQ == 0 and case.key_length % BK == 0
    assert case.heads % case.kv_heads == 0
    rng = np.random.default_rng(case.seed)
    b, h, kv, ql, kl, d = (
        case.batches,
        case.heads,
        case.kv_heads,
        case.query_length,
        case.key_length,
        HEAD_DIM,
    )
    q = rng.uniform(-1, 1, (b, h, ql, d)).astype(np.float32)
    k = rng.uniform(-1, 1, (b, kv, kl, d)).astype(np.float32)
    v = rng.uniform(-1, 1, (b, kv, kl, d)).astype(np.float32)
    mask_heads = h if case.mask_per_head else 1
    mask = rng.uniform(-2, 0, (b, mask_heads, ql, kl)).astype(np.float32)
    scale = np.float32(1.0 / math.sqrt(d))
    group = h // kv
    keys = np.repeat(k.astype(np.float64), group, axis=1)
    values = np.repeat(v.astype(np.float64), group, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), keys)
    scores = scores * np.float64(scale) + mask.astype(np.float64)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    reference = np.einsum("bhqk,bhkd->bhqd", weights, values)
    if case.sequence_major:
        q_storage = np.ascontiguousarray(q.transpose(0, 2, 1, 3))
        reference = reference.transpose(0, 2, 1, 3)
        q_strides = [ql * h * d, d, h * d]
    else:
        q_storage = q
        q_strides = _strides([b, h, ql, d])[:3]
    kv_strides = _strides([b, kv, kl, d])[:3]
    mask_strides = _strides([b, mask_heads, ql, kl])[:3]
    if not case.mask_per_head:
        mask_strides[1] = 0
    params = dict(
        B=b,
        H=h,
        D=d,
        qL=ql,
        kL=kl,
        gqa_factor=group,
        scale=float(scale),
        NQ=ql // BQ,
        NK=kl // BK,
        NQ_aligned=ql // BQ,
        NK_aligned=kl // BK,
        qL_rem=0,
        kL_rem=0,
        qL_off=kl - ql,
        Q_strides=q_strides,
        K_strides=kv_strides,
        V_strides=kv_strides,
        O_strides=q_strides,
    )
    resources = {
        "Q": q_storage.ravel(),
        "K": k.ravel(),
        "V": v.ravel(),
        "mask": mask.ravel(),
    }
    geometry = {
        "workgroupCount": [ql // BQ, h, b],
        "workgroupSize": list(WORKGROUP),
    }
    return resources, params, {"M_strides": mask_strides}, reference.ravel(), geometry


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_attention_reference_follows_the_definition(case):
    resources, params, mask_params, reference, geometry = _case_data(case)
    assert reference.shape == (
        case.batches * case.heads * case.query_length * HEAD_DIM,
    )
    # Weighted averages of values in [-1, 1] stay in [-1, 1].
    assert np.all(np.abs(reference) <= 1.0)
    assert params["qL_off"] == case.key_length - case.query_length
    assert params["gqa_factor"] * case.kv_heads == case.heads
    assert geometry["workgroupCount"][0] * BQ == case.query_length
    assert (mask_params["M_strides"][1] == 0) is (not case.mask_per_head)
    if case.sequence_major:
        assert params["Q_strides"][1] == HEAD_DIM
        assert params["Q_strides"][2] == case.heads * HEAD_DIM


def _f32(values):
    values = [float(value) for value in values]
    return {"dtype": "float32", "shape": [len(values)], "values": values}


def _record(binding, values):
    words = pack_storage_records(binding["scalarLayout"], [values])
    return {"dtype": "uint32", "shape": [len(words)], "values": words}


def _dispatch_values(bindings, case):
    resources, params, mask_params, reference, geometry = _case_data(case)
    initial_output = [GUARD] * (len(reference) + GUARDS)
    values = {name: _f32(resources[name]) for name in ("Q", "K", "V", "mask")}
    values.update(
        O=_f32(initial_output),
        sinks=_f32([GUARD]),
        lse=_f32([GUARD]),
        params=_record(bindings["params"], params),
        mask_params=_record(bindings["mask_params"], mask_params),
    )
    inputs = {bindings[name]["name"]: value for name, value in values.items()}
    outputs = {
        bindings["O"]["name"]: _f32(list(reference) + [GUARD] * GUARDS),
        bindings["lse"]["name"]: _f32([GUARD]),
    }
    return inputs, outputs, geometry


def _check(label, actual, expected, bindings):
    output = np.asarray(actual[bindings["O"]["name"]]["values"], dtype=np.float64)
    reference = np.asarray(expected[bindings["O"]["name"]]["values"], dtype=np.float64)
    assert output.shape == reference.shape, label
    assert output[-GUARDS:].tolist() == [GUARD] * GUARDS, label
    assert actual[bindings["lse"]["name"]]["values"] == [GUARD], label
    np.testing.assert_allclose(
        output[:-GUARDS],
        reference[:-GUARDS],
        rtol=TOLERANCE,
        atol=TOLERANCE,
        err_msg=label,
    )
    return float(np.max(np.abs(output[:-GUARDS] - reference[:-GUARDS])))


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def attention_source(tmp_path_factory):
    """The pinned tree, the original Metal module (on macOS) and the target."""
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned attention native gate")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET", PLATFORM_TARGETS[sys.platform])
    assert PLATFORM_TARGETS[sys.platform] == target
    for tool in GATE_TOOLS[target]:
        assert shutil.which(tool), f"Required tool is missing: {tool}"
    work = tmp_path_factory.mktemp("attention-original")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    assert not subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    ).strip()
    source = root / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    for name, digest in HEADER_HASHES.items():
        header = root / "mlx/backend/metal/kernels/steel/attn" / name
        assert hashlib.sha256(header.read_bytes()).hexdigest() == digest
    original = None
    if target == "metal":
        run(["xcrun", "--sdk", "macosx", "metal", "--version"], work, "metal-version")
        original = compile_metal(source, root, work / "original", flags=ORIGINAL_FLAGS)
    else:
        tool = GATE_TOOLS[target][0]
        run([tool, "--version"], work, f"{tool}-version")
    return root, original, target


@pytest.fixture(scope="module")
def translated_attention(attention_source, tmp_path_factory):
    """One translation, package and binding table shared by every case."""
    root, _original, target = attention_source
    work = tmp_path_factory.mktemp("attention-translation")
    with tempfile.TemporaryDirectory(prefix=".current-attention-", dir=root) as output:
        try:
            report = translate_project(
                _config(root, Path(output).name, target), format_output=False
            )
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            assert payload["project"]["indexRangeAssertions"] == (
                list(_opengl_index_assertions()) if target == "opengl" else []
            )
            descriptor, package = _prepare_native_package(report, work)
        finally:
            shutil.copytree(output, work / "translation", dirs_exist_ok=True)
    constants = descriptor["specializationConstants"]
    assert {item["name"]: item["id"] for item in constants} == CONSTANT_IDS
    assert all(
        item["frozen"] is True and item["value"] is SPECIALIZATIONS[item["name"]]
        for item in constants
    )
    bindings = {
        item["provenance"]["sourceResource"]["parameter"]: item
        for item in descriptor["bindings"]
    }
    assert len(descriptor["bindings"]) == len(bindings) == len(BINDINGS)
    assert set(bindings) == set(BINDINGS)
    params = bindings["params"]["scalarLayout"]
    assert params["elementStrideBytes"] == 152
    assert len(params["structMembers"]) == 18
    assert bindings["mask_params"]["scalarLayout"]["elementStrideBytes"] == 24
    return descriptor, package, bindings


def _dispatch_original(executor, state, native, source, module):
    constants = {
        name: NativeRuntimeConstantBinding(
            name=name,
            constant=RuntimeSpecializationConstant(
                name=name,
                constant_id=identifier,
                dtype="bool",
                value=SPECIALIZATIONS[name],
                kind="function-constant",
            ),
            value=SPECIALIZATIONS[name],
            source="original-source-specialization",
        )
        for name, identifier in CONSTANT_IDS.items()
    }
    control = replace(
        native,
        artifact_path=source,
        module_path=module,
        entry_point=ENTRY,
        loaded_artifact=None,
        constants=constants,
    )
    actual = executor.runtime_adapter.runtime.dispatch(None, state, control)
    assert state.details["metalRuntime"]["librarySHA256"] == (
        hashlib.sha256(module.read_bytes()).hexdigest()
    )
    return actual


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_current_attention_executes(
    tmp_path, attention_source, translated_attention, case
):
    root, original, target = attention_source
    descriptor, package, bindings = translated_attention
    inputs, outputs, geometry = _dispatch_values(bindings, case)
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, geometry, expected_target=target
    )
    assert not request.execution_plan.diagnostics
    assert not request.adapter_contract.specialization_constants
    executor = _executor(target)
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        actual = executor.runtime_adapter.dispatch(state, native)
        result = {
            "case": case.name,
            "target": target,
            "checkedValues": len(outputs[bindings["O"]["name"]]["values"]) - GUARDS,
            "maxAbsErrorGenerated": _check("generated", actual, outputs, bindings),
            "moduleSha256": hashlib.sha256(native.module_path.read_bytes()).hexdigest(),
        }
        if target == "metal":
            original_outputs = _dispatch_original(
                executor, state, native, root / SOURCE, original
            )
            result["maxAbsErrorOriginalMetal"] = _check(
                "original Metal", original_outputs, outputs, bindings
            )
        _write_json(
            tmp_path / "parity.json",
            {
                "sourceCommit": MLX_COMMIT,
                "entryPoint": ENTRY,
                "specializations": SPECIALIZATIONS,
                "sourceSha256": SOURCE_SHA256,
                "headerSha256": HEADER_HASHES,
                "tolerance": {"rtol": TOLERANCE, "atol": TOLERANCE},
                "result": result,
                "scope": (
                    "One float32 aligned float-mask entry with frozen constants; "
                    "not the upstream suite or the full attention family."
                ),
            },
        )
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
