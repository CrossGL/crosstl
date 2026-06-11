import json
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl.project as project_api
from crosstl.project.runtime_verification import (
    COMPARISON_FAILED,
    PASSED,
    RUNTIME_FAILED,
    RUNTIME_TEST_MANIFEST_KIND,
    RUNTIME_TEST_PLAN_KIND,
    RUNTIME_TEST_REPORT_KIND,
    RUNTIME_VERIFICATION_REPORT_KIND,
    SKIPPED,
    TRANSLATION_FAILED,
    UNAVAILABLE,
    RuntimeAdapterContract,
    RuntimeDispatchGeometry,
    RuntimeEntryPoint,
    RuntimeExecutionAdapter,
    RuntimeExecutionError,
    RuntimeExecutionRequest,
    RuntimeExecutorAvailability,
    RuntimeExecutorResult,
    RuntimeExecutorSkipped,
    RuntimeParityAdapter,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
    RuntimeVerificationError,
    build_runtime_test_manifest,
    compare_runtime_outputs,
    default_runtime_test_adapters,
    load_runtime_verification_fixtures,
    parse_runtime_test_manifest,
    parse_runtime_verification_fixtures,
    plan_runtime_test_manifest,
    prepare_runtime_execution,
    verify_runtime_fixtures,
    verify_runtime_test_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def _artifact_report(tmp_path, artifacts):
    return {
        "kind": "crosstl-project-portability-report",
        "project": {"root": str(tmp_path), "targets": ["opengl", "directx"]},
        "artifacts": artifacts,
    }


def _translated_artifact(**overrides):
    artifact = {
        "source": "kernels/add.cgl",
        "path": "out/opengl/debug/add.glsl",
        "target": "opengl",
        "sourceBackend": "cgl",
        "variant": "debug",
        "status": "translated",
    }
    artifact.update(overrides)
    return artifact


def _runtime_fixture(**overrides):
    fixture = {
        "id": "add-debug",
        "selector": {
            "source": "kernels/add.cgl",
            "target": "OpenGL",
            "variant": "debug",
        },
        "inputs": [
            {
                "name": "lhs",
                "kind": "buffer",
                "dtype": "float32",
                "shape": [2],
                "values": [1.0, 2.0],
            }
        ],
        "expectedOutputs": [
            {
                "name": "out",
                "kind": "buffer",
                "dtype": "float32",
                "shape": [2],
                "values": [2.0, 4.0],
            }
        ],
        "defaultTolerance": {"absolute": 0.01, "relative": 0.001},
    }
    fixture.update(overrides)
    return fixture


def test_load_runtime_verification_fixtures_from_json(tmp_path):
    fixture_path = tmp_path / "runtime-fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "kind": "crosstl-runtime-verification-fixtures",
                "fixtures": [_runtime_fixture()],
            }
        ),
        encoding="utf-8",
    )

    fixtures = load_runtime_verification_fixtures(fixture_path)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.id == "add-debug"
    assert fixture.selector.source == "kernels/add.cgl"
    assert fixture.selector.target == "opengl"
    assert fixture.selector.variant == "debug"
    assert fixture.inputs[0].values == [1.0, 2.0]
    assert fixture.expected_outputs[0].tolerance.absolute == 0.01


def test_load_runtime_verification_fixtures_parses_adapter_contract(tmp_path):
    fixture_path = tmp_path / "runtime-fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "kind": "crosstl-runtime-verification-fixtures",
                "fixtures": [
                    _runtime_fixture(
                        runtimeAdapter={
                            "id": "adapter-contract-v1",
                            "entryPoints": [
                                {
                                    "name": "main",
                                    "stage": "compute",
                                    "executionConfig": {"numthreads": [4, 1, 1]},
                                }
                            ],
                            "resourceBindings": [
                                {
                                    "id": "buffer|0|0|lhs",
                                    "name": "lhs",
                                    "kind": "buffer",
                                    "set": 0,
                                    "binding": 0,
                                    "value": "lhs",
                                }
                            ],
                            "specializationConstants": [
                                {
                                    "name": "tile_size",
                                    "id": 1,
                                    "dtype": "uint32",
                                    "value": 4,
                                    "required": True,
                                }
                            ],
                            "dispatch": {
                                "entryPoint": "main",
                                "workgroupSize": [4, 1, 1],
                                "workgroupCount": [1, 1, 1],
                            },
                            "validationHooks": [
                                {"name": "adapter-contract", "phase": "pre-run"}
                            ],
                        }
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    fixture = load_runtime_verification_fixtures(fixture_path)[0]

    contract = fixture.adapter_contract
    assert isinstance(contract, RuntimeAdapterContract)
    assert contract.contract_id == "adapter-contract-v1"
    assert contract.entry_points[0].name == "main"
    assert contract.resource_bindings[0].binding == 0
    assert contract.specialization_constants[0].required is True
    assert contract.dispatch.workgroup_count == (1, 1, 1)
    assert contract.validation_hooks[0].phase == "pre-run"
    assert fixture.to_json()["runtimeAdapter"]["dispatch"]["workgroupSize"] == [
        4,
        1,
        1,
    ]


def test_parse_runtime_verification_fixtures_rejects_missing_selector():
    with pytest.raises(RuntimeVerificationError, match="selector must identify"):
        parse_runtime_verification_fixtures({"fixtures": [{"id": "missing"}]})


def test_compare_runtime_outputs_applies_tolerance():
    comparisons = compare_runtime_outputs(
        [
            {
                "name": "out",
                "dtype": "float32",
                "shape": [3],
                "values": [1.0, 0.0, 100.0],
            }
        ],
        {"out": {"dtype": "float32", "values": [1.009, 0.005, 104.0]}},
        default_tolerance={"absolute": 0.01, "relative": 0.05},
    )

    assert comparisons[0]["status"] == PASSED
    assert comparisons[0]["maxAbsoluteError"] == 4.0

    failed = compare_runtime_outputs(
        [{"name": "out", "values": [1.0]}],
        {"out": [1.02]},
        default_tolerance={"absolute": 0.01},
    )

    assert failed[0]["status"] == COMPARISON_FAILED
    assert failed[0]["mismatchCount"] == 1


def test_prepare_runtime_execution_maps_artifact_metadata_to_plan(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("// translated shader", encoding="utf-8")
    artifact = _translated_artifact(
        path="out/opengl/debug/add.glsl",
        compileSteps=[
            {
                "kind": "compile-source",
                "command": ["compiler", "add.glsl"],
                "metadata": {"profile": "compute"},
            }
        ],
        loadSteps=[
            {
                "kind": "load-module",
                "metadata": {"apiCalls": ["createModule"]},
            }
        ],
    )
    fixture = parse_runtime_verification_fixtures(
        {
            "fixtures": [
                _runtime_fixture(
                    inputs=[
                        {
                            "name": "lhs",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [8],
                            "values": [1.0] * 8,
                        }
                    ],
                    expectedOutputs=[
                        {
                            "name": "out",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [8],
                            "values": [2.0] * 8,
                        }
                    ],
                )
            ]
        }
    )[0]
    contract = RuntimeAdapterContract(
        entry_points=(
            RuntimeEntryPoint(
                name="main",
                stage="compute",
                workgroup_size=(4, 1, 1),
            ),
        ),
        resource_bindings=(
            RuntimeResourceBinding(name="lhs", kind="buffer", binding=0),
            RuntimeResourceBinding(name="out", kind="buffer", binding=1),
        ),
        specialization_constants=(
            RuntimeSpecializationConstant(
                name="tile_size",
                dtype="uint32",
                value=4,
                required=True,
            ),
        ),
        dispatch=RuntimeDispatchGeometry(entry_point="main", global_size=(8, 1, 1)),
    )
    request = RuntimeExecutionRequest(
        fixture=fixture,
        artifact=artifact,
        artifact_path=artifact_path,
        project_root=tmp_path,
        adapter_contract=contract,
    )

    plan = prepare_runtime_execution(request)

    assert plan.compile_steps[0].action == "compile-source"
    assert plan.compile_steps[0].command == ("compiler", "add.glsl")
    assert plan.load_steps[0].action == "load-module"
    assert plan.resource_bindings[0].source == "input"
    assert plan.resource_bindings[1].source == "expectedOutput"
    assert plan.constant_bindings[0].value == 4
    assert plan.dispatch.workgroup_count == (2, 1, 1)
    assert plan.to_json()["resourceBindings"][0]["value"] == {
        "name": "lhs",
        "kind": "buffer",
        "dtype": "float32",
        "shape": [8],
    }


def test_verify_runtime_fixtures_reports_pass_and_writes_json(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("// translated shader", encoding="utf-8")

    class FakeOpenGLExecutor(project_api.RuntimeExecutor):
        name = "fake-opengl"

        def run(self, request):
            assert request.artifact_path == artifact_path.resolve()
            assert request.fixture.inputs[0].values == [1.0, 2.0]
            return RuntimeExecutorResult(
                outputs={
                    "out": {
                        "dtype": "float32",
                        "shape": [2],
                        "values": [2.0, 4.005],
                    }
                }
            )

    output_path = tmp_path / "runtime-report.json"

    report = verify_runtime_fixtures(
        _artifact_report(tmp_path, [_translated_artifact()]),
        {"fixtures": [_runtime_fixture()]},
        executors={"opengl": FakeOpenGLExecutor()},
        output_path=output_path,
    )

    assert report["kind"] == RUNTIME_VERIFICATION_REPORT_KIND
    assert report["success"] is True
    assert report["summary"] == {
        "fixtureCount": 1,
        "passedCount": 1,
        "skippedCount": 0,
        "unavailableCount": 0,
        "translationFailedCount": 0,
        "runtimeFailedCount": 0,
        "comparisonFailedCount": 0,
        "failedCount": 0,
    }
    assert report["results"][0]["status"] == PASSED
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["summary"] == report["summary"]


def test_verify_runtime_fixtures_runs_execution_adapter_pipeline(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("// translated shader", encoding="utf-8")

    class VectorAddAdapter(RuntimeExecutionAdapter):
        name = "vector-add-adapter"

        def dispatch_fixture(self, state):
            assert state.plan.dispatch.entry_point == "main"
            assert state.plan.dispatch.workgroup_count == (1, 1, 1)
            assert state.resource_values["out"] is None
            state.record_step("dispatch", "deterministic-vector-add")
            return {
                "out": {
                    "dtype": "float32",
                    "shape": [2],
                    "values": [
                        lhs + rhs
                        for lhs, rhs in zip(
                            state.resource_values["lhs"],
                            state.resource_values["rhs"],
                        )
                    ],
                }
            }

    report = verify_runtime_fixtures(
        _artifact_report(
            tmp_path,
            [
                _translated_artifact(
                    entryPoints=[
                        {
                            "name": "main",
                            "stage": "compute",
                            "workgroupSize": [2, 1, 1],
                        }
                    ],
                    resourceBindings=[
                        {"name": "lhs", "kind": "buffer", "binding": 0},
                        {"name": "rhs", "kind": "buffer", "binding": 1},
                        {"name": "out", "kind": "buffer", "binding": 2},
                    ],
                    dispatch={
                        "entryPoint": "main",
                        "globalSize": [2, 1, 1],
                    },
                )
            ],
        ),
        {
            "fixtures": [
                _runtime_fixture(
                    inputs=[
                        {
                            "name": "lhs",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [1.0, 2.0],
                        },
                        {
                            "name": "rhs",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [10.0, 20.0],
                        },
                    ],
                    expectedOutputs=[
                        {
                            "name": "out",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [11.0, 22.0],
                        }
                    ],
                )
            ]
        },
        executors={"opengl": VectorAddAdapter()},
    )

    result = report["results"][0]
    assert result["status"] == PASSED
    assert result["runtimeExecution"]["resourceBindings"][2]["source"] == (
        "expectedOutput"
    )
    assert result["runtimeExecution"]["dispatch"]["workgroupCount"] == [1, 1, 1]
    assert [
        step["phase"] for step in result["executor"]["details"]["adapterSteps"]
    ] == [
        "compile",
        "load",
        "bind",
        "bind",
        "dispatch",
        "collect",
    ]


def test_verify_runtime_fixtures_reports_setup_diagnostic_with_source_span(tmp_path):
    source_span = {
        "file": "kernels/add.cgl",
        "line": 4,
        "column": 5,
        "offset": 12,
        "length": 8,
        "endLine": 4,
        "endColumn": 13,
        "endOffset": 20,
    }
    generated_span = {
        "file": "out/opengl/debug/add.glsl",
        "line": 8,
        "column": 1,
        "offset": 32,
        "length": 8,
        "endLine": 8,
        "endColumn": 9,
        "endOffset": 40,
    }

    report = verify_runtime_fixtures(
        _artifact_report(
            tmp_path,
            [
                _translated_artifact(
                    resourceBindings=[
                        {
                            "name": "missing",
                            "kind": "buffer",
                            "binding": 0,
                        }
                    ],
                    sourceMap={
                        "source": source_span,
                        "generated": generated_span,
                    },
                )
            ],
        ),
        {"fixtures": [_runtime_fixture()]},
    )

    result = report["results"][0]
    diagnostic = result["diagnostics"][0]
    assert result["status"] == RUNTIME_FAILED
    assert result["failurePhase"] == "runtime-setup"
    assert diagnostic["code"] == "project.runtime-verification.resource-unbound"
    assert diagnostic["artifact"]["source"] == "kernels/add.cgl"
    assert diagnostic["sourceSpan"] == source_span
    assert diagnostic["generatedSpan"] == generated_span
    assert diagnostic["location"] == source_span


def test_verify_runtime_fixtures_runs_executable_adapter_contract_fixture():
    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification"

    class FakeMLXExecutor(project_api.RuntimeExecutor):
        name = "fake-mlx-runtime"

        def is_available(self, request):
            assert isinstance(
                request.adapter_contract, project_api.RuntimeAdapterContract
            )
            assert request.adapter_contract.validation_hooks[0].name == (
                "adapter-contract"
            )
            return RuntimeExecutorAvailability(True)

        def run(self, request):
            contract = request.adapter_contract
            assert request.artifact_path == (fixture_dir / "vector_add.metal").resolve()
            assert contract.contract_id == "vector-add-contract-v1"
            assert contract.entry_points[0].name == "vector_add"
            assert contract.entry_points[0].stage == "compute"
            assert [binding.name for binding in contract.resource_bindings] == [
                "lhs",
                "rhs",
                "out",
            ]
            assert contract.specialization_constants[0].kind == "function-constant"
            assert contract.specialization_constants[0].value == 4
            assert contract.dispatch.entry_point == "vector_add"
            assert contract.dispatch.workgroup_size == (4, 1, 1)
            assert contract.dispatch.workgroup_count == (1, 1, 1)
            assert [hook.phase for hook in contract.validation_hooks] == [
                "pre-run",
                "post-run",
            ]
            inputs = {value.name: value.values for value in request.fixture.inputs}
            output = [lhs + rhs for lhs, rhs in zip(inputs["lhs"], inputs["rhs"])]
            return RuntimeExecutorResult(
                outputs={
                    "out": {
                        "dtype": "float32",
                        "shape": [4],
                        "values": output,
                    }
                },
                details={"entryPoint": contract.dispatch.entry_point},
            )

    report = verify_runtime_fixtures(
        fixture_dir / "vector_add.artifacts.json",
        fixture_dir / "vector_add.runtime-fixtures.json",
        executors={"mlx": FakeMLXExecutor()},
        project_root=ROOT,
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == PASSED
    assert result["executor"]["key"] == "mlx"
    assert result["runtimeAdapter"]["entryPoints"][0]["name"] == "vector_add"
    assert len(result["runtimeAdapter"]["resourceBindings"]) == 3
    assert result["runtimeAdapter"]["specializationConstants"][0]["kind"] == (
        "function-constant"
    )
    assert result["runtimeAdapter"]["dispatch"] == {
        "entryPoint": "vector_add",
        "workgroupSize": [4, 1, 1],
        "workgroupCount": [1, 1, 1],
        "globalSize": [4, 1, 1],
        "metadata": {"stage": "compute", "source": "fixture", "status": "available"},
    }
    assert result["comparisons"][0]["status"] == PASSED


def test_verify_runtime_fixtures_reports_unavailable_executor_without_failure(tmp_path):
    report = verify_runtime_fixtures(
        _artifact_report(tmp_path, [_translated_artifact()]),
        {"fixtures": [_runtime_fixture()]},
    )

    assert report["success"] is True
    assert report["summary"]["unavailableCount"] == 1
    assert report["summary"]["failedCount"] == 0
    assert report["results"][0]["status"] == UNAVAILABLE
    assert report["results"][0]["failurePhase"] == "runtime"
    assert report["results"][0]["diagnostics"][0]["code"].endswith(
        "executor-unavailable"
    )


def test_verify_runtime_fixtures_keeps_skip_separate_from_unavailable(tmp_path):
    class SkippingExecutor(project_api.RuntimeExecutor):
        def run(self, request):
            raise RuntimeExecutorSkipped("fixture requires a runtime feature")

    report = verify_runtime_fixtures(
        _artifact_report(tmp_path, [_translated_artifact()]),
        {"fixtures": [_runtime_fixture()]},
        executors={"opengl": SkippingExecutor()},
    )

    assert report["success"] is True
    assert report["summary"]["skippedCount"] == 1
    assert report["summary"]["unavailableCount"] == 0
    assert report["results"][0]["status"] == SKIPPED


def test_verify_runtime_fixtures_records_executor_availability_details(tmp_path):
    class UnavailableExecutor(project_api.RuntimeExecutor):
        name = "fake-directx"

        def is_available(self, request):
            return RuntimeExecutorAvailability(
                False,
                reason="dxc runtime probe unavailable",
                details={"tool": "dxc"},
            )

        def run(self, request):
            raise AssertionError("unavailable executors should not run")

    report = verify_runtime_fixtures(
        _artifact_report(tmp_path, [_translated_artifact(target="directx")]),
        {"fixtures": [_runtime_fixture(selector={"target": "directx"})]},
        executors={"directx": UnavailableExecutor()},
    )

    result = report["results"][0]
    assert result["status"] == UNAVAILABLE
    assert result["executor"]["message"] == "dxc runtime probe unavailable"
    assert result["executor"]["details"] == {"tool": "dxc"}


def test_verify_runtime_fixtures_separates_translation_and_runtime_failures(tmp_path):
    class FailingRuntimeExecutor(project_api.RuntimeExecutor):
        def run(self, request):
            raise RuntimeExecutionError("pipeline state creation failed")

    fixtures = {
        "fixtures": [
            _runtime_fixture(id="translation-failed"),
            _runtime_fixture(
                id="runtime-failed",
                selector={"source": "kernels/runtime.cgl", "target": "opengl"},
            ),
        ]
    }
    artifacts = [
        _translated_artifact(status="failed", error="parser rejected source"),
        _translated_artifact(source="kernels/runtime.cgl", variant=None),
    ]

    report = verify_runtime_fixtures(
        _artifact_report(tmp_path, artifacts),
        fixtures,
        executors={"opengl": FailingRuntimeExecutor()},
    )

    assert report["success"] is False
    assert report["summary"]["translationFailedCount"] == 1
    assert report["summary"]["runtimeFailedCount"] == 1
    assert report["summary"]["comparisonFailedCount"] == 0
    results_by_fixture = {result["fixture"]: result for result in report["results"]}
    assert results_by_fixture["translation-failed"]["status"] == TRANSLATION_FAILED
    assert results_by_fixture["translation-failed"]["failurePhase"] == "translation"
    assert results_by_fixture["runtime-failed"]["status"] == RUNTIME_FAILED
    assert results_by_fixture["runtime-failed"]["failurePhase"] == "runtime"


def test_parse_runtime_test_manifest_maps_adapters_and_platform_requirements():
    missing_tool = "crosstl-runtime-tool-that-does-not-exist-1007"

    manifest = parse_runtime_test_manifest(
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "opengl-native",
                    "target": "opengl",
                    "executor": "opengl-native",
                    "platformRequirements": {
                        "platformClass": "native-graphics",
                        "requiredTools": [missing_tool],
                        "requiredEnvironment": ["CROSSTL_RUNTIME_TEST_DEVICE"],
                    },
                }
            ],
            "tests": [
                _runtime_fixture(id="add-runtime", adapter="opengl-native"),
            ],
        }
    )

    assert manifest.adapters[0].adapter_id == "opengl-native"
    assert manifest.adapters[0].platform_requirements.platform_class == (
        "native-graphics"
    )
    assert missing_tool in manifest.test_cases[0].platform_requirements.required_tools
    assert manifest.test_cases[0].fixture.executor == "opengl-native"
    assert manifest.to_json()["tests"][0]["adapter"] == "opengl-native"


def test_build_runtime_test_manifest_from_mlx_fixture_metadata():
    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification" / "mlx"
    artifact_report = fixture_dir / "reduced_binary_add.artifacts.json"
    fixture_metadata = fixture_dir / "reduced_binary_add.fixture-metadata.json"

    manifest = build_runtime_test_manifest(
        artifact_report,
        fixture_metadata,
        project_root=ROOT,
    )

    assert manifest["kind"] == RUNTIME_TEST_MANIFEST_KIND
    assert manifest["success"] is True
    assert manifest["artifactManifest"] == str(artifact_report)
    assert manifest["projectRoot"] == str(ROOT)
    assert manifest["summary"]["testCount"] == 1
    assert manifest["summary"]["testsByTarget"] == {"metal": 1}
    assert manifest["diagnostics"] == []
    assert manifest["metadata"]["repository"] == "mlx"
    assert manifest["metadata"]["fixtureMetadataKind"] == (
        "crosstl-project-runtime-fixture-metadata"
    )
    assert manifest["adapters"] == [
        {
            "id": "metal-runtime-probe",
            "target": "metal",
            "executor": "metal",
            "adapterKind": "metal-runtime-probe",
            "platformRequirements": {
                "platformClass": "native-graphics",
                "requiredTools": ["xcrun"],
                "metadata": {"source": "default-runtime-test-adapter"},
            },
        }
    ]
    test_case = manifest["tests"][0]
    assert test_case["adapter"] == "metal-runtime-probe"
    assert test_case["selector"] == {
        "source": "mlx/backend/metal/kernels/binary.metal",
        "target": "metal",
        "variant": "reduced-add",
        "path": "tests/fixtures/runtime_verification/mlx/reduced_binary_add.metal",
    }
    assert test_case["runtimeAdapter"]["entryPoints"][0]["name"] == (
        "mlx_binary_add_f32"
    )
    assert test_case["runtimeAdapter"]["resourceBindings"][2]["value"] == "out"
    assert test_case["runtimeAdapter"]["specializationConstants"] == [
        {
            "kind": "function-constant",
            "required": True,
            "name": "element_count",
            "id": 0,
            "dtype": "uint32",
            "value": 4,
        }
    ]
    assert test_case["runtimeAdapter"]["dispatch"] == {
        "entryPoint": "mlx_binary_add_f32",
        "globalSize": [4, 1, 1],
    }
    parsed = parse_runtime_test_manifest(manifest)
    assert parsed.test_cases[0].fixture.id == "mlx-reduced-binary-add-f32"

    plan = plan_runtime_test_manifest(manifest)

    assert plan["kind"] == RUNTIME_TEST_PLAN_KIND
    assert plan["testCases"][0]["fixture"] == "mlx-reduced-binary-add-f32"
    assert plan["testCases"][0]["artifact"]["target"] == "metal"
    assert plan["testCases"][0]["runtimeExecution"]["dispatch"] == {
        "entryPoint": "mlx_binary_add_f32",
        "workgroupSize": [4, 1, 1],
        "workgroupCount": [1, 1, 1],
        "globalSize": [4, 1, 1],
        "metadata": {"source": "fixture", "stage": "compute", "status": "available"},
    }


def test_build_runtime_test_manifest_reports_ambiguous_fixture_selector(tmp_path):
    metadata = {
        "kind": "crosstl-project-runtime-fixture-metadata",
        "fixtures": [
            {
                "id": "ambiguous-add",
                "selector": {
                    "source": "kernels/add.cgl",
                    "target": "metal",
                },
                "inputs": [{"name": "lhs", "values": [1.0]}],
                "expectedOutputs": [{"name": "out", "values": [2.0]}],
            }
        ],
    }
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(target="metal", path="out/metal/add-a.metal"),
            _translated_artifact(target="metal", path="out/metal/add-b.metal"),
        ],
    )

    manifest = build_runtime_test_manifest(artifact_report, metadata)

    assert manifest["success"] is False
    assert manifest["diagnosticCounts"]["error"] == 1
    assert manifest["diagnostics"][0]["code"] == (
        "project.runtime-test-manifest.artifact-ambiguous"
    )
    assert manifest["diagnostics"][0]["fixture"] == "ambiguous-add"
    parse_runtime_test_manifest(manifest)


def test_build_runtime_test_manifest_reports_incomplete_fixture_data(tmp_path):
    metadata = {
        "kind": "crosstl-project-runtime-fixture-metadata",
        "fixtures": [
            {
                "id": "missing-outputs",
                "selector": {
                    "source": "kernels/add.cgl",
                    "target": "opengl",
                    "variant": "debug",
                },
                "inputs": [{"name": "lhs", "values": [1.0]}],
            }
        ],
    }

    manifest = build_runtime_test_manifest(
        _artifact_report(tmp_path, [_translated_artifact()]),
        metadata,
    )

    codes = {diagnostic["code"] for diagnostic in manifest["diagnostics"]}
    assert manifest["success"] is False
    assert "project.runtime-test-manifest.fixture-expected-outputs-missing" in codes
    assert "project.runtime-test-manifest.entry-points-unavailable" in codes
    parse_runtime_test_manifest(manifest)


def test_project_cli_runtime_test_manifest_text_outputs_generated_tests():
    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification" / "mlx"
    artifact_report = fixture_dir / "reduced_binary_add.artifacts.json"
    fixture_metadata = fixture_dir / "reduced_binary_add.fixture-metadata.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "runtime-test-manifest",
            str(artifact_report),
            str(fixture_metadata),
            "--project-root",
            str(ROOT),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"Project runtime test manifest: {artifact_report}" in result.stdout
    assert f"Fixture metadata: {fixture_metadata}" in result.stdout
    assert "Summary: 1 runtime tests, 1 adapters, 0 diagnostics" in result.stdout
    assert "Runtime tests by target: metal=1" in result.stdout
    assert "- mlx-reduced-binary-add-f32" in result.stdout


def test_plan_runtime_test_manifest_records_structured_skip_and_toolchain_logs(
    tmp_path,
):
    missing_tool = "crosstl-runtime-tool-that-does-not-exist-1007"
    artifact = _translated_artifact(
        toolchain={"status": "failed", "tool": "glslangValidator"},
        toolchainRuns=[
            {
                "status": "failed",
                "command": ["glslangValidator", "-S", "comp"],
                "stdout": "",
                "stderr": "validation failed",
            }
        ],
    )

    plan = plan_runtime_test_manifest(
        _artifact_report(tmp_path, [artifact]),
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "opengl-native",
                    "target": "opengl",
                    "executor": "opengl-native",
                    "platformRequirements": {"requiredTools": [missing_tool]},
                }
            ],
            "tests": [
                _runtime_fixture(id="add-runtime", adapter="opengl-native"),
            ],
        },
    )

    planned = plan["testCases"][0]
    diagnostic = planned["diagnostics"][0]
    assert plan["kind"] == RUNTIME_TEST_PLAN_KIND
    assert planned["status"] == SKIPPED
    assert planned["failurePhase"] == "platform-requirements"
    assert planned["artifact"]["source"] == "kernels/add.cgl"
    assert planned["artifact"]["target"] == "opengl"
    assert planned["artifact"]["path"] == "out/opengl/debug/add.glsl"
    assert planned["artifact"]["toolchainRuns"][0]["stderr"] == "validation failed"
    assert diagnostic["code"] == (
        "project.runtime-test.platform-requirements-unavailable"
    )
    assert diagnostic["fixture"] == "add-runtime"
    assert missing_tool in diagnostic["missingTools"]


def test_verify_runtime_test_manifest_reports_skipped_dependency_record(tmp_path):
    missing_tool = "crosstl-runtime-tool-that-does-not-exist-1007"
    output_path = tmp_path / "runtime-test-report.json"

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_translated_artifact()]),
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "opengl-native",
                    "target": "opengl",
                    "executor": "opengl-native",
                    "platformRequirements": {"requiredTools": [missing_tool]},
                }
            ],
            "tests": [
                _runtime_fixture(id="add-runtime", adapter="opengl-native"),
            ],
        },
        output_path=output_path,
    )

    result = report["results"][0]
    assert report["kind"] == RUNTIME_TEST_REPORT_KIND
    assert report["success"] is True
    assert report["summary"]["skippedCount"] == 1
    assert result["status"] == SKIPPED
    assert result["failurePhase"] == "platform-requirements"
    assert result["executor"]["status"] == SKIPPED
    assert missing_tool in result["executor"]["details"]["missingTools"]
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["summary"] == report["summary"]


def test_verify_runtime_test_manifest_runs_executor_and_links_failed_check(tmp_path):
    artifact = _translated_artifact(
        toolchainRuns=[
            {
                "status": "passed",
                "command": ["glslangValidator", "-S", "comp"],
                "stdout": "ok",
                "stderr": "",
            }
        ],
    )

    class WrongValueExecutor(project_api.RuntimeExecutor):
        def run(self, request):
            assert request.fixture.id == "add-runtime"
            return RuntimeExecutorResult(
                outputs={
                    "out": {
                        "dtype": "float32",
                        "shape": [2],
                        "values": [2.0, 99.0],
                    }
                }
            )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [artifact]),
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "runtime-check",
                    "executor": "opengl",
                    "platformRequirements": {
                        "platformClass": "native-graphics",
                        "requiredTools": [],
                    },
                }
            ],
            "tests": [
                _runtime_fixture(id="add-runtime", adapter="runtime-check"),
            ],
        },
        executors={"opengl": WrongValueExecutor()},
    )

    result = report["results"][0]
    diagnostic = result["diagnostics"][0]
    assert report["success"] is False
    assert result["status"] == COMPARISON_FAILED
    assert result["failurePhase"] == "comparison"
    assert result["artifact"]["toolchainRuns"][0]["stdout"] == "ok"
    assert diagnostic["artifact"]["source"] == "kernels/add.cgl"
    assert diagnostic["artifact"]["toolchainRuns"][0]["command"][0] == (
        "glslangValidator"
    )
    assert diagnostic["output"] == "out"


def test_verify_runtime_test_manifest_runs_runtime_parity_adapter_pipeline(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("// translated shader", encoding="utf-8")
    artifact = _translated_artifact(
        path="out/opengl/debug/add.glsl",
        entryPoints=[
            {
                "name": "main",
                "stage": "compute",
                "workgroupSize": [2, 1, 1],
            }
        ],
        resourceBindings=[
            {"name": "lhs", "kind": "buffer", "binding": 0},
            {"name": "rhs", "kind": "buffer", "binding": 1},
            {"name": "out", "kind": "buffer", "binding": 2},
        ],
        dispatch={
            "entryPoint": "main",
            "globalSize": [2, 1, 1],
        },
    )

    class VectorAddParityAdapter(RuntimeParityAdapter):
        name = "vector-add-runtime"
        target = "opengl"

        def prepare_buffers(self, state):
            assert state.plan.artifact_path == artifact_path.resolve()
            assert state.plan.dispatch.workgroup_count == (1, 1, 1)
            return dict(state.resource_values)

        def dispatch(self, state, prepared_buffers):
            assert state.plan.dispatch.entry_point == "main"
            assert prepared_buffers["out"] is None
            return [
                lhs + rhs
                for lhs, rhs in zip(
                    prepared_buffers["lhs"],
                    prepared_buffers["rhs"],
                )
            ]

        def collect_outputs(self, state, dispatch_result):
            return {
                "out": {
                    "dtype": "float32",
                    "shape": [2],
                    "values": dispatch_result,
                }
            }

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [artifact]),
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "runtime-check",
                    "executor": "opengl",
                    "adapterKind": "opengl-runtime-parity",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                _runtime_fixture(
                    id="add-runtime",
                    adapter="runtime-check",
                    inputs=[
                        {
                            "name": "lhs",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [1.0, 2.0],
                        },
                        {
                            "name": "rhs",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [10.0, 20.0],
                        },
                    ],
                    expectedOutputs=[
                        {
                            "name": "out",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [11.0, 22.0],
                        }
                    ],
                ),
            ],
        },
        executors={"opengl": VectorAddParityAdapter()},
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == PASSED
    assert result["executor"]["name"] == "opengl-runtime-parity"
    assert (
        result["executor"]["details"]["runtimeParityAdapter"]["runtimeAdapter"]
        == "vector-add-runtime"
    )
    assert [
        step["phase"] for step in result["executor"]["details"]["adapterSteps"]
    ] == [
        "compile",
        "load",
        "bind",
        "bind",
        "prepare",
        "dispatch",
        "collect",
    ]
    assert result["comparisons"][0]["status"] == PASSED


def test_verify_runtime_test_manifest_reports_runtime_adapter_gap_as_unavailable(
    tmp_path,
):
    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_translated_artifact()]),
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "runtime-check",
                    "executor": "opengl",
                    "adapterKind": "opengl-runtime-parity",
                    "platformRequirements": {"requiredTools": []},
                }
            ],
            "tests": [
                _runtime_fixture(id="add-runtime", adapter="runtime-check"),
            ],
        },
    )

    result = report["results"][0]
    assert report["success"] is True
    assert report["summary"]["unavailableCount"] == 1
    assert report["summary"]["failedCount"] == 0
    assert result["status"] == UNAVAILABLE
    assert result["failurePhase"] == "runtime"
    assert result["executor"]["name"] == "opengl-runtime-parity"
    assert "Runtime parity adapter is not implemented" in result["executor"]["message"]
    assert result["executor"]["details"]["adapter"] == "runtime-check"
    assert result["diagnostics"][0]["code"] == (
        "project.runtime-verification.executor-unavailable"
    )


def test_default_runtime_test_adapters_cover_native_platform_classes():
    adapters = default_runtime_test_adapters()

    platform_classes = {
        adapter.platform_requirements.platform_class for adapter in adapters
    }
    assert "native-graphics" in platform_classes
    assert "native-compute" in platform_classes
    assert any(adapter.target == "opengl" for adapter in adapters)
    assert any(adapter.target == "cuda" for adapter in adapters)
