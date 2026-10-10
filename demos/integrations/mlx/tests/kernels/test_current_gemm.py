"""Round-trip a complete pinned MLX GEMM entry and compare native results."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
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
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor

MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal"
SOURCE_SHA256 = "6c6e67f5c7fc6b378ca2c849c36a79d696c93e65b3a60d3f5b1f13928c79029e"
ENTRY = "steel_gemm_fused_nn_float32_float32_bm32_bn32_bk16_wm2_wn2"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_GEMM"
CONSTANT_IDS = {
    "has_batch": 10,
    "use_out_source": 100,
    "do_axpby": 110,
    "align_M": 200,
    "align_N": 201,
    "align_K": 202,
}
GUARD = -98765.0
CASES = (
    (1, 1, 1, 1, 0, 0),
    (4, 5, 7, 1, 0, 0),
    (32, 32, 16, 1, 0, 0),
    (33, 35, 17, 2, 0, 0),
    (33, 35, 17, 2, 3, 1),
)
ORIGINAL_FLAGS = (
    "-std=metal3.2",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
    "-Wno-c++17-extensions",
    "-Wno-c++20-extensions",
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _config(root, output):
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=("metal",),
        output_dir=output,
        entry_points={SOURCE: (ENTRY,)},
        specialization_constants={name: False for name in CONSTANT_IDS},
        freeze_specialization_constants=True,
        workgroup_size_rules={SOURCE: (32, 2, 2)},
        source_options={
            "metal": {
                "max_template_specializations": 256,
                "max_template_materialization_work": 8192,
                "cooperative_matrix_fragment_mapping": "tile_4x4_row_pair",
                "cooperative_matrix_fragment_mapping_provenance": (
                    "mlx_steel_BaseMMAFrag_get_coord"
                ),
                "preserve_resource_origins": True,
            }
        },
    )


def _case_data(case):
    m, n, k, batches, padding, extra_groups = case
    lda, ldb, ldd = k + padding, n + padding, n + padding
    stride_a, stride_b, stride_d = (
        m * lda + padding,
        k * ldb + padding,
        m * ldd + padding,
    )
    a = [GUARD] * (batches * stride_a + 8)
    b = [GUARD] * (batches * stride_b + 8)
    expected = [GUARD] * (batches * stride_d + 8)
    for batch in range(batches):
        for row in range(m):
            for inner in range(k):
                index = (batch * m + row) * k + inner
                a[batch * stride_a + row * lda + inner] = ((index * 3 + 1) % 11 - 5) / 4
        for inner in range(k):
            for col in range(n):
                index = (batch * k + inner) * n + col
                b[batch * stride_b + inner * ldb + col] = ((index * 7 + 2) % 13 - 6) / 4
        for row in range(m):
            for col in range(n):
                # Quarter-integer operands keep every product and sum exact in float32.
                expected[batch * stride_d + row * ldd + col] = sum(
                    a[batch * stride_a + row * lda + inner]
                    * b[batch * stride_b + inner * ldb + col]
                    for inner in range(k)
                )
    params = dict(
        M=m,
        N=n,
        K=k,
        lda=lda,
        ldb=ldb,
        ldd=ldd,
        tiles_n=(n + 31) // 32,
        tiles_m=(m + 31) // 32,
        batch_stride_a=stride_a,
        batch_stride_b=stride_b,
        batch_stride_d=stride_d,
        swizzle_log=0,
        gemm_k_iterations_aligned=k // 16,
        batch_ndim=0,
    )
    geometry = {
        "workgroupCount": [
            params["tiles_n"] + extra_groups,
            params["tiles_m"] + extra_groups,
            batches,
        ],
        "workgroupSize": [32, 2, 2],
    }
    return a, b, expected, params, geometry


def _dispatch_values(bindings, case):
    a, b, expected, params, geometry = _case_data(case)
    addmm = dict(
        ldc=params["N"],
        fdc=1,
        batch_stride_c=params["M"] * params["N"],
        alpha=1.0,
        beta=0.0,
    )
    values = {
        "A": ("float32", a),
        "B": ("float32", b),
        "C": ("float32", [0.0]),
        "D": ("float32", [GUARD] * len(expected)),
        "params": (
            "uint32",
            pack_storage_records(bindings["params"]["scalarLayout"], [params]),
        ),
        "addmm_params": (
            "uint32",
            pack_storage_records(bindings["addmm_params"]["scalarLayout"], [addmm]),
        ),
        "batch_shape": ("int32", [1]),
        "batch_strides": ("int64", [0]),
    }
    inputs = {
        bindings[name]["name"]: {"dtype": dtype, "shape": [len(items)], "values": items}
        for name, (dtype, items) in values.items()
    }
    outputs = {
        bindings["D"]["name"]: {
            "dtype": "float32",
            "shape": [len(expected)],
            "values": expected,
        }
    }
    return inputs, outputs, geometry


def _original_outputs(request, source, module, work):
    executor = _executor("metal")
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    try:
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        native = executor.runtime_adapter.prepare_buffers(state)
        constants = {
            name: NativeRuntimeConstantBinding(
                name=name,
                constant=RuntimeSpecializationConstant(
                    name=name,
                    constant_id=identifier,
                    dtype="bool",
                    value=False,
                    kind="function-constant",
                ),
                value=False,
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
        module_hash = hashlib.sha256(module.read_bytes()).hexdigest()
        assert state.details["metalRuntime"]["librarySHA256"] == module_hash
        _write_json(
            work / "original-evidence.json",
            {
                "outputs": actual,
                "request": control.to_json(),
                "sourceCommit": MLX_COMMIT,
                "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "moduleSha256": module_hash,
                "runtime": state.details["metalRuntime"],
                "compileFlags": ORIGINAL_FLAGS,
            },
        )
        return actual
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


@pytest.mark.parametrize("case", CASES)
def test_gemm_case_shapes_and_guards(case):
    m, n, k, batches, padding, extra_groups = case
    a, b, expected, params, geometry = _case_data(case)
    assert len(a) == batches * (m * (k + padding) + padding) + 8
    assert len(b) == batches * (k * (n + padding) + padding) + 8
    assert len(expected) == batches * (m * (n + padding) + padding) + 8
    live = {
        batch * params["batch_stride_d"] + row * params["ldd"] + col
        for batch in range(batches)
        for row in range(m)
        for col in range(n)
    }
    assert sum(value != GUARD for value in expected) == m * n * batches
    assert all(
        expected[index] == GUARD for index in range(len(expected)) if index not in live
    )
    assert geometry["workgroupCount"] == [
        (n + 31) // 32 + extra_groups,
        (m + 31) // 32 + extra_groups,
        batches,
    ]
    assert geometry["workgroupSize"] == [32, 2, 2]
    assert params["gemm_k_iterations_aligned"] == k // 16


def test_gemm_scalar_reference():
    a, b, expected, _, _ = _case_data(CASES[0])
    assert a[0] == -1.0 and b[0] == -1.0
    assert expected == [1.0] + [GUARD] * 8


def test_current_gemm_metal_roundtrip(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned GEMM native gate")
    assert sys.platform == "darwin" and shutil.which("xcrun")
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
    run(["xcrun", "--sdk", "macosx", "metal", "--version"], tmp_path, "metal-version")
    original = compile_metal(source, root, tmp_path / "original", flags=ORIGINAL_FLAGS)
    with tempfile.TemporaryDirectory(prefix=".current-gemm-", dir=root) as directory:
        work = Path(directory)
        try:
            report = translate_project(_config(root, work.name), format_output=False)
            report.write_json(tmp_path / "report.json")
            payload = report.to_json()
            assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
            assert not payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            assert payload["project"]["indexRangeAssertions"] == []
            assert payload["project"]["workgroupAccessAssertions"] == []
            descriptor, package = _prepare_native_package(report, tmp_path)
            constants = descriptor["specializationConstants"]
            assert len(constants) == len(CONSTANT_IDS)
            assert {item["name"]: item["id"] for item in constants} == CONSTANT_IDS
            assert all(
                item["frozen"] is True and item["value"] is False for item in constants
            )
            bindings = {
                item["provenance"]["sourceResource"]["parameter"]: item
                for item in descriptor["bindings"]
            }
            assert len(descriptor["bindings"]) == len(bindings) == 8
            assert set(bindings) == {
                "A",
                "B",
                "C",
                "D",
                "params",
                "addmm_params",
                "batch_shape",
                "batch_strides",
            }
            assert bindings["params"]["scalarLayout"]["elementStrideBytes"] == 72
            assert len(bindings["params"]["scalarLayout"]["structMembers"]) == 14
            assert bindings["addmm_params"]["scalarLayout"]["elementStrideBytes"] == 24
            results = []
            for case in CASES:
                case_work = tmp_path / ("case-" + "-".join(map(str, case)))
                case_work.mkdir()
                inputs, outputs, geometry = _dispatch_values(bindings, case)
                _write_json(
                    case_work / "inputs.json",
                    {"inputs": inputs, "outputs": outputs, "geometry": geometry},
                )
                request = build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    inputs,
                    outputs,
                    geometry,
                    expected_target="metal",
                )
                assert not request.execution_plan.diagnostics
                assert not request.adapter_contract.specialization_constants
                _execute(
                    request,
                    outputs,
                    case_work,
                    validate=lambda path, destination, target: compile_metal(
                        path,
                        root,
                        destination,
                        flags=("-std=metal3.1", "-fno-fast-math"),
                    ),
                )
                original_outputs = _original_outputs(
                    request, source, original, case_work
                )
                generated = json.loads((case_work / "evidence.json").read_text())
                record = generated["records"]["generated"]
                assert (
                    record["details"]["metalRuntime"]["librarySHA256"]
                    == record["moduleSha256"]
                )
                assert record["details"]["metalRuntime"]["threadExecutionWidth"] == 32
                assert not record["request"]["constants"]
                assert (
                    original_outputs
                    == generated["records"]["generated"]["outputs"]
                    == outputs
                )
                results.append(
                    {
                        "shape": list(case[:4]),
                        "padding": case[4],
                        "extraWorkgroups": case[5],
                        "checkedValues": len(next(iter(outputs.values()))["values"]),
                        "equalOriginalMetal": True,
                        "equalExactReference": True,
                    }
                )
                _write_json(
                    tmp_path / "parity.json",
                    {
                        "sourceCommit": MLX_COMMIT,
                        "entryPoint": ENTRY,
                        "sourceSha256": SOURCE_SHA256,
                        "cases": results,
                        "complete": len(results) == len(CASES),
                        "scope": (
                            "One float32 NN entry with six frozen false constants; not the upstream suite or full GEMM family."
                        ),
                    },
                )
        finally:
            shutil.copytree(work, tmp_path / "translation", dirs_exist_ok=True)
