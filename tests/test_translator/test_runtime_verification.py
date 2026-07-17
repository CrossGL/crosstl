import json
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl.project as project_api
from crosstl._crosstl import translate
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
    DirectXRuntimeParityAdapter,
    NativeRuntimeDispatchRequest,
    NativeRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeAdapterContract,
    RuntimeArtifactSelector,
    RuntimeDispatchGeometry,
    RuntimeEntryPoint,
    RuntimeExecutionAdapter,
    RuntimeExecutionError,
    RuntimeExecutionRequest,
    RuntimeExecutorAvailability,
    RuntimeExecutorResult,
    RuntimeExecutorSkipped,
    RuntimeFixture,
    RuntimeParityAdapter,
    RuntimeResourceBinding,
    RuntimeSpecializationConstant,
    RuntimeVerificationError,
    VulkanRuntimeParityAdapter,
    build_runtime_test_manifest,
    compare_runtime_outputs,
    default_runtime_test_adapters,
    load_runtime_verification_fixtures,
    native_runtime_parity_adapter,
    native_runtime_parity_adapters,
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


def _native_runtime_artifact(**overrides):
    artifact = _translated_artifact(
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
        specializationConstants=[
            {
                "name": "tile_size",
                "id": 0,
                "dtype": "uint32",
                "value": 2,
            }
        ],
        dispatch={"entryPoint": "main", "globalSize": [2, 1, 1]},
    )
    artifact.update(overrides)
    return artifact


def _native_runtime_fixture(**overrides):
    fixture = _runtime_fixture(
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
    )
    fixture.update(overrides)
    return fixture


def _native_runtime_manifest(target="opengl", fixture=None):
    return {
        "kind": RUNTIME_TEST_MANIFEST_KIND,
        "adapters": [
            {
                "id": "runtime-check",
                "executor": target,
                "adapterKind": f"{target}-runtime-parity",
                "platformRequirements": {"requiredTools": []},
            }
        ],
        "tests": [
            fixture
            or _native_runtime_fixture(
                selector={
                    "source": "kernels/add.cgl",
                    "target": target,
                    "variant": "debug",
                }
            )
        ],
    }


class FakeNativeRuntime:
    name = "fake-native-runtime"

    def __init__(self, *, outputs=None, dispatch_error=None):
        self.outputs = outputs
        self.dispatch_error = dispatch_error
        self.prepared = None

    def is_available(self, adapter, request):
        assert isinstance(adapter, NativeRuntimeParityAdapter)
        assert request.fixture.id == "add-runtime"
        return RuntimeExecutorAvailability(True)

    def load_artifact(self, adapter, state, module_path):
        return {"target": adapter.target, "modulePath": str(module_path)}

    def dispatch(self, adapter, state, prepared):
        assert isinstance(prepared, NativeRuntimeDispatchRequest)
        assert prepared.entry_point == "main"
        assert prepared.buffers["lhs"].value == [1.0, 2.0]
        assert prepared.buffers["rhs"].value == [10.0, 20.0]
        assert prepared.buffers["out"].source == "expectedOutput"
        assert prepared.constants["tile_size"].value == 2
        assert state.loaded_artifact["target"] == adapter.target
        self.prepared = prepared
        if self.dispatch_error is not None:
            raise self.dispatch_error
        return self.outputs or {
            "out": {
                "dtype": "float32",
                "shape": [2],
                "values": [11.0, 22.0],
            }
        }


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


def test_runtime_fixture_round_trips_independent_entry_point():
    fixture = RuntimeFixture(
        id="entry-point-fixture",
        selector=RuntimeArtifactSelector(
            source="kernels/add.cgl",
            target="opengl",
        ),
        entry_point="vector_add",
    )

    assert fixture.entry_point == "vector_add"
    assert fixture.to_json()["entryPoint"] == "vector_add"

    parsed = parse_runtime_verification_fixtures(
        {
            "fixtures": [
                _runtime_fixture(
                    id="parsed-entry-point-fixture",
                    entryPoint="reduce_sum",
                )
            ]
        }
    )[0]

    assert parsed.entry_point == "reduce_sum"
    assert parsed.to_json()["entryPoint"] == "reduce_sum"


def test_runtime_fixture_accepts_scalar_value_alias():
    parsed = parse_runtime_verification_fixtures(
        {
            "fixtures": [
                _runtime_fixture(
                    id="scalar-input-fixture",
                    inputs=[
                        {
                            "name": "start",
                            "kind": "scalar",
                            "dtype": "uint32",
                            "value": 3,
                        }
                    ],
                )
            ]
        }
    )[0]

    assert parsed.inputs[0].values == 3
    assert parsed.to_json()["inputs"][0]["values"] == 3


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


def test_verify_runtime_fixtures_resolves_generated_resource_names(tmp_path):
    class AliasAdapter(RuntimeExecutionAdapter):
        name = "alias-adapter"

        def dispatch_fixture(self, state):
            assert state.resource_values["lhs_Buffer"] == [1.0, 2.0]
            assert state.resource_values["rhsUniform"] == [10.0, 20.0]
            assert state.resource_values["arangeuint8_outBuffer"] is None
            state.record_step("dispatch", "alias-aware-dispatch")
            return {
                "out": {
                    "dtype": "float32",
                    "shape": [2],
                    "values": [11.0, 22.0],
                }
            }

    report = verify_runtime_fixtures(
        _artifact_report(
            tmp_path,
            [
                _translated_artifact(
                    entryPoints=[{"name": "main", "stage": "compute"}],
                    resourceBindings=[
                        {"name": "lhs_Buffer", "kind": "buffer", "binding": 0},
                        {"name": "rhsUniform", "kind": "buffer", "binding": 1},
                        {
                            "name": "arangeuint8_outBuffer",
                            "kind": "buffer",
                            "binding": 2,
                        },
                    ],
                    dispatch={"entryPoint": "main", "workgroupCount": [1, 1, 1]},
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
        executors={"opengl": AliasAdapter()},
    )

    result = report["results"][0]
    assert result["status"] == PASSED
    bindings = result["runtimeExecution"]["resourceBindings"]
    assert [binding["source"] for binding in bindings] == [
        "input",
        "input",
        "expectedOutput",
    ]


def test_verify_runtime_fixtures_resolves_explicit_resource_aliases(tmp_path):
    class ExplicitAliasAdapter(RuntimeExecutionAdapter):
        name = "explicit-alias-adapter"

        def dispatch_fixture(self, state):
            assert state.resource_values["generatedOutputBuffer"] is None
            state.record_step("dispatch", "explicit-alias-dispatch")
            return {
                "out": {
                    "dtype": "float32",
                    "shape": [2],
                    "values": [2.0, 4.0],
                }
            }

    report = verify_runtime_fixtures(
        _artifact_report(
            tmp_path,
            [
                _translated_artifact(
                    entryPoints=[{"name": "main", "stage": "compute"}],
                    resourceBindings=[
                        {
                            "name": "generatedOutputBuffer",
                            "kind": "buffer",
                            "binding": 0,
                        },
                    ],
                    dispatch={"entryPoint": "main", "workgroupCount": [1, 1, 1]},
                )
            ],
        ),
        {
            "fixtures": [
                _runtime_fixture(
                    inputs=[],
                    expectedOutputs=[
                        {
                            "name": "out",
                            "kind": "buffer",
                            "dtype": "float32",
                            "shape": [2],
                            "values": [2.0, 4.0],
                            "aliases": ["generatedOutputBuffer"],
                        }
                    ],
                )
            ]
        },
        executors={"opengl": ExplicitAliasAdapter()},
    )

    result = report["results"][0]
    assert result["status"] == PASSED
    binding = result["runtimeExecution"]["resourceBindings"][0]
    assert binding["value"]["name"] == "out"
    assert binding["value"]["metadata"] == {"aliases": ["generatedOutputBuffer"]}
    assert binding["source"] == "expectedOutput"


def test_plan_runtime_test_manifest_warns_for_unbound_incomplete_layout_resource(
    tmp_path,
):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                entryPoints=[{"name": "main", "stage": "compute"}],
                resourceBindings=[
                    {
                        "name": "float16_t",
                        "kind": "uniform",
                        "binding": None,
                        "status": "layout-missing",
                    }
                ],
                dispatch={"entryPoint": "main", "workgroupCount": [1, 1, 1]},
            )
        ],
    )
    manifest = build_runtime_test_manifest(
        artifact_report,
        {
            "adapters": [
                {
                    "id": "metadata-runtime-probe",
                    "executor": "metadata",
                    "adapterKind": "metadata-runtime-probe",
                }
            ],
            "fixtures": [
                {
                    "adapter": "metadata-runtime-probe",
                    "fixture": _runtime_fixture(
                        inputs=[],
                        expectedOutputs=[{"name": "out", "values": []}],
                    ),
                }
            ],
        },
        project_root=tmp_path,
    )

    plan = plan_runtime_test_manifest(artifact_report, manifest, project_root=tmp_path)

    case = plan["testCases"][0]
    assert case["status"] in {"planned", "skipped"}
    assert case["diagnostics"][0]["severity"] == "warning"
    assert case["diagnostics"][0]["code"] == (
        "project.runtime-verification.resource-unbound"
    )
    assert plan["summary"]["failedCount"] == 0


def test_plan_runtime_test_manifest_accepts_matching_compiled_workgroup_size(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                entryPoints=[
                    {
                        "name": "vector_add",
                        "stage": "compute",
                        "workgroupSize": [8, 1, 1],
                    }
                ]
            )
        ],
    )
    fixture = _runtime_fixture(
        adapter="runtime-check",
        entryPoint="vector_add",
        runtimeAdapter={
            "dispatch": {
                "entryPoint": "vector_add",
                "workgroupSize": [8, 1, 1],
                "globalSize": [16, 1, 1],
            }
        },
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        _native_runtime_manifest(fixture=fixture),
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    assert case["status"] == "planned"
    assert case["runtimeExecution"]["dispatch"]["workgroupSize"] == [8, 1, 1]
    assert all(
        diagnostic["code"] != "project.runtime-verification.workgroup-size-mismatch"
        for diagnostic in case["diagnostics"]
    )


def test_plan_runtime_test_manifest_rejects_compiled_workgroup_size_mismatch(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                id="compiled-vector-add",
                entryPoints=[
                    {
                        "name": "vector_add",
                        "stage": "compute",
                        "workgroupSize": [8, 1, 1],
                        "metadata": {
                            "workgroupSizeProvenance": "project.workgroupSize"
                        },
                    }
                ],
            )
        ],
    )
    fixture = _runtime_fixture(
        adapter="runtime-check",
        entryPoint="vector_add",
        runtimeAdapter={
            "dispatch": {
                "entryPoint": "vector_add",
                "workgroupSize": [4, 1, 1],
                "globalSize": [16, 1, 1],
            }
        },
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        _native_runtime_manifest(fixture=fixture),
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    diagnostic = next(
        item
        for item in case["diagnostics"]
        if item["code"] == "project.runtime-verification.workgroup-size-mismatch"
    )
    assert case["status"] == RUNTIME_FAILED
    assert case["failurePhase"] == "runtime-setup"
    assert plan["summary"]["plannedCount"] == 0
    assert diagnostic["requestedWorkgroupSize"] == [4, 1, 1]
    assert diagnostic["compiledWorkgroupSize"] == [8, 1, 1]
    assert diagnostic["selectedEntryPoint"] == "vector_add"
    assert diagnostic["entryPointSelectionSource"] == "fixture.entryPoint"
    assert diagnostic["selectedEntryPointProvenance"] == {
        "source": "fixture.entryPoint",
        "artifactEntry": {
            "name": "vector_add",
            "stage": "compute",
            "workgroupSize": [8, 1, 1],
            "metadata": {"workgroupSizeProvenance": "project.workgroupSize"},
        },
    }


def test_plan_runtime_test_manifest_checks_selected_entry_workgroup_size(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                entryPoints=[
                    {
                        "name": "vector_add",
                        "stage": "compute",
                        "workgroupSize": [4, 1, 1],
                    },
                    {
                        "name": "reduce_sum",
                        "stage": "compute",
                        "workgroupSize": [16, 1, 1],
                    },
                ]
            )
        ],
    )
    fixture = _runtime_fixture(
        adapter="runtime-check",
        entryPoint="reduce_sum",
        runtimeAdapter={
            "dispatch": {
                "entryPoint": "reduce_sum",
                "workgroupSize": [4, 1, 1],
                "globalSize": [16, 1, 1],
            }
        },
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        _native_runtime_manifest(fixture=fixture),
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    diagnostic = next(
        item
        for item in case["diagnostics"]
        if item["code"] == "project.runtime-verification.workgroup-size-mismatch"
    )
    assert case["status"] == RUNTIME_FAILED
    assert diagnostic["selectedEntryPoint"] == "reduce_sum"
    assert diagnostic["compiledWorkgroupSize"] == [16, 1, 1]
    assert diagnostic["requestedWorkgroupSize"] == [4, 1, 1]
    assert diagnostic["entryPointSelectionSource"] == "fixture.entryPoint"


@pytest.mark.parametrize(
    ("compiled_size", "requested_size", "expected_size"),
    [
        ([8, 1, 1], None, [8, 1, 1]),
        (None, [4, 1, 1], [4, 1, 1]),
    ],
)
def test_plan_runtime_test_manifest_completes_missing_workgroup_size_side(
    tmp_path,
    compiled_size,
    requested_size,
    expected_size,
):
    entry_point = {"name": "vector_add", "stage": "compute"}
    if compiled_size is not None:
        entry_point["workgroupSize"] = compiled_size
    dispatch = {"entryPoint": "vector_add", "globalSize": [16, 1, 1]}
    if requested_size is not None:
        dispatch["workgroupSize"] = requested_size
    artifact_report = _artifact_report(
        tmp_path,
        [_translated_artifact(entryPoints=[entry_point])],
    )
    fixture = _runtime_fixture(
        adapter="runtime-check",
        entryPoint="vector_add",
        runtimeAdapter={"dispatch": dispatch},
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        _native_runtime_manifest(fixture=fixture),
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    assert case["status"] == "planned"
    assert case["runtimeExecution"]["dispatch"]["workgroupSize"] == expected_size
    assert all(
        diagnostic["code"] != "project.runtime-verification.workgroup-size-mismatch"
        for diagnostic in case["diagnostics"]
    )


def test_plan_runtime_test_manifest_scopes_contract_to_fixture_entry_point(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                id="multi-entry-artifact",
                entryPoints=[
                    {"name": "vector_add", "stage": "compute"},
                    {"name": "reduce_sum", "stage": "compute"},
                ],
                resourceBindings=[
                    {
                        "name": "lhs",
                        "kind": "buffer",
                        "binding": 0,
                        "metadata": {"entryPoint": "vector_add"},
                    },
                    {
                        "name": "scratch",
                        "kind": "buffer",
                        "binding": 1,
                        "metadata": {"entryPoints": ["reduce_sum"]},
                    },
                    {
                        "name": "out",
                        "kind": "buffer",
                        "binding": 2,
                        "metadata": {"entryPoints": ["vector_add"]},
                    },
                ],
                specializationConstants=[
                    {
                        "name": "vector_width",
                        "value": 4,
                        "metadata": {"entryPoint": "vector_add"},
                    },
                    {
                        "name": "reduction_width",
                        "value": 8,
                        "metadata": {"entryPoints": ["reduce_sum"]},
                    },
                ],
                validationHooks=[
                    {
                        "name": "validate-vector-layout",
                        "metadata": {"entryPoint": "vector_add"},
                    },
                    {
                        "name": "validate-reduction-layout",
                        "metadata": {"entryPoints": ["reduce_sum"]},
                    },
                ],
                dispatch={"entryPoint": "reduce_sum", "workgroupCount": [1, 1, 1]},
            )
        ],
    )
    manifest = {
        "kind": RUNTIME_TEST_MANIFEST_KIND,
        "tests": [_runtime_fixture(entryPoint="vector_add")],
    }

    plan = plan_runtime_test_manifest(artifact_report, manifest, project_root=tmp_path)

    contract = plan["testCases"][0]["runtimeAdapter"]
    assert [item["name"] for item in contract["resourceBindings"]] == ["lhs", "out"]
    assert [item["name"] for item in contract["specializationConstants"]] == [
        "vector_width"
    ]
    assert [item["name"] for item in contract["validationHooks"]] == [
        "validate-vector-layout"
    ]
    assert contract["dispatch"]["entryPoint"] == "vector_add"


def test_plan_runtime_test_manifest_can_replace_ambiguous_artifact_contract(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                id="ambiguous-aggregate",
                hostInterface={"status": "ambiguous"},
                entryPoints=[{"name": "CSMain", "stage": "compute"}],
                resourceBindings=[{"name": "unscoped", "kind": "buffer", "binding": 0}],
                dispatch={"entryPoint": "CSMain", "workgroupCount": [1, 1, 1]},
            )
        ],
    )
    fixture = _runtime_fixture(
        entryPoint="CSMain_3",
        runtimeAdapter={
            "entryPoints": [
                {
                    "name": "CSMain_3",
                    "stage": "compute",
                    "workgroupSize": [1, 1, 1],
                }
            ],
            "resourceBindings": [
                {
                    "name": "lhs",
                    "kind": "buffer",
                    "binding": 4,
                    "value": "lhs",
                },
                {
                    "name": "out",
                    "kind": "buffer",
                    "binding": 2,
                    "value": "out",
                },
            ],
            "dispatch": {"entryPoint": "CSMain_3", "globalSize": [2, 1, 1]},
            "metadata": {"artifactContractMode": "replace"},
        },
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        {"kind": RUNTIME_TEST_MANIFEST_KIND, "tests": [fixture]},
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    assert case["status"] in {"planned", "skipped"}
    assert [
        binding["name"] for binding in case["runtimeAdapter"]["resourceBindings"]
    ] == ["lhs", "out"]
    assert case["runtimeAdapter"]["dispatch"]["entryPoint"] == "CSMain_3"
    assert all(
        diagnostic["code"] != "project.runtime-verification.entry-point-missing"
        for diagnostic in case["diagnostics"]
    )


def test_plan_runtime_test_manifest_rejects_invalid_artifact_contract_mode(tmp_path):
    artifact_report = _artifact_report(tmp_path, [_translated_artifact()])
    fixture = _runtime_fixture(
        adapter="unavailable-runtime",
        runtimeAdapter={"metadata": {"artifactContractMode": "overlay"}},
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "adapters": [
                {
                    "id": "unavailable-runtime",
                    "target": "opengl",
                    "executor": "opengl",
                    "platformRequirements": {
                        "requiredTools": ["crosstl-guaranteed-missing-runtime-tool"]
                    },
                }
            ],
            "tests": [fixture],
        },
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    diagnostic = next(
        item
        for item in case["diagnostics"]
        if item["code"] == "project.runtime-verification.artifact-contract-mode-invalid"
    )
    assert case["status"] == RUNTIME_FAILED
    assert diagnostic["reasonKind"] == "artifact-contract-mode-invalid"
    assert diagnostic["artifactContractMode"] == "overlay"
    assert diagnostic["supportedArtifactContractModes"] == ["merge", "replace"]


def test_mlx_arange_directx_generated_manifest_plans_curated_interface(tmp_path):
    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification" / "mlx"
    artifact_path = tmp_path / "out" / "directx" / "arange" / "arangeuint32.hlsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("// generated standalone entry", encoding="utf-8")
    artifact_report = {
        "kind": "crosstl-runtime-artifact-manifest",
        "project": {"root": str(tmp_path), "targets": ["directx"]},
        "artifacts": [
            {
                "id": "mlx-arange-generated-directx",
                "source": "mlx/backend/metal/kernels/arange.metal",
                "path": "out/directx/arange/arangeuint32.hlsl",
                "target": "directx",
                "sourceBackend": "metal",
                "status": "translated",
                "hostInterface": {"status": "ambiguous"},
                "entryPoints": [
                    {
                        "name": "CSMain",
                        "stage": "compute",
                        "executionConfig": {"numthreads": [1, 1, 1]},
                    }
                ],
                "resourceBindings": [
                    {
                        "name": "arangeuint8_start_Constants",
                        "kind": "constant-buffer",
                        "set": 0,
                        "binding": 0,
                    },
                    {
                        "name": "out_",
                        "kind": "buffer",
                        "set": 0,
                        "binding": 2,
                        "access": "read_write",
                    },
                ],
                "parameterBlocks": [
                    {
                        "name": "arangeuint8_start_Constants",
                        "kind": "constant-buffer",
                        "set": 0,
                        "binding": 0,
                        "status": "layout-ready",
                    }
                ],
                "dispatch": {
                    "status": "available",
                    "workgroups": [
                        {
                            "entryPoint": "CSMain",
                            "stage": "compute",
                            "workgroupSize": [1, 1, 1],
                        }
                    ],
                },
                "runtimeDataStatus": {
                    "hostInterface": "ambiguous",
                    "entryPoints": "available",
                    "resourceBindings": "available",
                    "parameterBlocks": "available",
                    "dispatch": "available",
                },
            }
        ],
    }

    manifest = build_runtime_test_manifest(
        artifact_report,
        fixture_dir / "arange_directx.fixture-metadata.json",
        project_root=tmp_path,
    )
    plan = plan_runtime_test_manifest(
        artifact_report,
        manifest,
        project_root=tmp_path,
    )

    assert manifest["success"] is True
    assert manifest["tests"][0]["entryPoint"] == "CSMain"
    assert manifest["tests"][0]["metadata"]["runtimeMetadata"]["status"] == (
        "incomplete"
    )
    case = plan["testCases"][0]
    assert case["status"] == "planned"
    assert case["runtimeExecution"]["dispatch"] == {
        "entryPoint": "CSMain",
        "workgroupSize": [1, 1, 1],
        "workgroupCount": [7, 1, 1],
        "globalSize": [7, 1, 1],
    }
    bindings = case["runtimeExecution"]["resourceBindings"]
    assert [binding["binding"]["name"] for binding in bindings] == [
        "arangeuint32_start_Constants",
        "arangeuint32_step_Constants",
        "out_",
    ]
    assert [binding["binding"]["binding"] for binding in bindings] == [0, 1, 2]
    assert bindings[0]["binding"]["metadata"]["parameterBlock"] == {
        "field": "arangeuint32_start",
        "byteOffset": 0,
        "byteSize": 4,
    }
    assert bindings[1]["binding"]["metadata"]["parameterBlock"] == {
        "field": "arangeuint32_step",
        "byteOffset": 0,
        "byteSize": 4,
    }
    assert [binding["source"] for binding in bindings] == [
        "input",
        "input",
        "expectedOutput",
    ]
    assert all(binding["status"] == "bound" for binding in bindings)


def test_plan_runtime_test_manifest_infers_entry_point_from_dispatch(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                entryPoints=[
                    {"name": "vector_add", "stage": "compute"},
                    {"name": "reduce_sum", "stage": "compute"},
                ],
                resourceBindings=[
                    {
                        "name": "lhs",
                        "kind": "buffer",
                        "binding": 0,
                        "metadata": {"entryPoint": "vector_add"},
                    },
                    {
                        "name": "out",
                        "kind": "buffer",
                        "binding": 1,
                        "metadata": {"entryPoints": ["vector_add"]},
                    },
                    {
                        "name": "scratch",
                        "kind": "buffer",
                        "binding": 2,
                        "metadata": {"entryPoint": "reduce_sum"},
                    },
                ],
                dispatch={"entryPoint": "vector_add", "workgroupCount": [1, 1, 1]},
            )
        ],
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        {"kind": RUNTIME_TEST_MANIFEST_KIND, "tests": [_runtime_fixture()]},
        project_root=tmp_path,
    )

    contract = plan["testCases"][0]["runtimeAdapter"]
    assert [item["name"] for item in contract["resourceBindings"]] == ["lhs", "out"]
    assert contract["dispatch"]["entryPoint"] == "vector_add"


def test_plan_runtime_test_manifest_reports_ambiguous_entry_point_resources(tmp_path):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                id="multi-entry-artifact",
                entryPoints=[
                    {"name": "vector_add", "stage": "compute"},
                    {"name": "reduce_sum", "stage": "compute"},
                ],
                resourceBindings=[
                    {"name": "lhs", "kind": "buffer", "binding": 0},
                    {"name": "scratch", "kind": "buffer", "binding": 1},
                    {"name": "out", "kind": "buffer", "binding": 2},
                ],
                dispatch={"entryPoint": "vector_add", "workgroupCount": [1, 1, 1]},
            )
        ],
    )

    plan = plan_runtime_test_manifest(
        artifact_report,
        {
            "kind": RUNTIME_TEST_MANIFEST_KIND,
            "tests": [_runtime_fixture(entryPoint="vector_add")],
        },
        project_root=tmp_path,
    )

    case = plan["testCases"][0]
    diagnostic = next(
        item
        for item in case["diagnostics"]
        if item["code"]
        == "project.runtime-verification.entry-point-resource-scope-ambiguous"
    )
    assert case["status"] in {"planned", "skipped"}
    scoped_resources = case["runtimeAdapter"].get("resourceBindings", [])
    assert len(scoped_resources) < 3
    assert "scratch" not in {item["name"] for item in scoped_resources}
    assert all(
        item["code"] != "project.runtime-verification.resource-unbound"
        for item in case["diagnostics"]
    )
    assert diagnostic["fixture"] == "add-debug"
    assert diagnostic["artifact"]["id"] == "multi-entry-artifact"
    assert diagnostic["selectedEntryPoint"] == "vector_add"
    assert set(diagnostic["candidateEntryPoints"]) == {"vector_add", "reduce_sum"}
    assert diagnostic["target"] == "opengl"


def test_runtime_test_manifest_preserves_ambiguous_scope_with_fixture_dispatch(
    tmp_path,
):
    artifact_report = _artifact_report(
        tmp_path,
        [
            _translated_artifact(
                id="multi-entry-artifact",
                entryPoints=[
                    {"name": "vector_add", "stage": "compute"},
                    {"name": "reduce_sum", "stage": "compute"},
                ],
                resourceBindings=[
                    {"name": "lhs", "kind": "buffer", "binding": 0},
                    {"name": "scratch", "kind": "buffer", "binding": 1},
                    {"name": "out", "kind": "buffer", "binding": 2},
                ],
                dispatch={"entryPoint": "reduce_sum", "workgroupCount": [1, 1, 1]},
            )
        ],
    )
    fixture = _runtime_fixture(
        entryPoint="vector_add",
        runtimeAdapter={"dispatch": {"globalSize": [2, 1, 1]}},
    )

    manifest = build_runtime_test_manifest(
        artifact_report,
        {"kind": "crosstl-project-runtime-fixture-metadata", "fixtures": [fixture]},
        project_root=tmp_path,
    )
    plan = plan_runtime_test_manifest(
        artifact_report,
        {"kind": RUNTIME_TEST_MANIFEST_KIND, "tests": [fixture]},
        project_root=tmp_path,
    )

    manifest_codes = {item["code"] for item in manifest["diagnostics"]}
    assert (
        "project.runtime-verification.entry-point-resource-scope-ambiguous"
        in manifest_codes
    )
    assert all(
        item["code"] != "project.runtime-test-manifest.resource-bindings-unavailable"
        for item in manifest["diagnostics"]
    )
    assert plan["testCases"][0]["runtimeAdapter"]["dispatch"]["entryPoint"] == (
        "vector_add"
    )
    assert plan["testCases"][0]["runtimeAdapter"]["dispatch"]["globalSize"] == [
        2,
        1,
        1,
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
    assert manifest["summary"]["runtimeMetadataStatusCounts"] == {"ready": 1}
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
    assert test_case["metadata"]["runtimeMetadata"]["status"] == "ready"
    assert test_case["metadata"]["runtimeMetadata"]["source"] == (
        "derived-runtime-adapter-contract"
    )
    assert test_case["metadata"]["runtimeMetadata"]["fields"] == {
        "entryPoints": "available",
        "resourceBindings": "available",
        "dispatch": "available",
    }
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


def test_mlx_file_scope_immutable_lookup_fixture_is_value_sensitive():
    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification" / "mlx"
    source_path = fixture_dir / "file_scope_immutable_lookup.metal"
    artifact_report = fixture_dir / "file_scope_immutable_lookup.artifacts.json"
    fixture_metadata = fixture_dir / "file_scope_immutable_lookup.fixture-metadata.json"

    generated = translate(
        str(source_path),
        backend="directx",
        source_backend="metal",
        format_output=False,
    )
    assert (
        "static const uint lookup_table[2][4] = "
        "{{3u, 5u, 7u, 11u}, {13u, 17u, 19u, 23u}};"
    ) in generated
    assert "static const uint lookup_table[2][4];" not in generated
    assert "output[index] = lookup_table[row][column];" in generated

    manifest = build_runtime_test_manifest(
        artifact_report,
        fixture_metadata,
        project_root=ROOT,
    )

    assert manifest["success"] is True
    assert manifest["summary"]["testCount"] == 1
    assert manifest["summary"]["testsByTarget"] == {"directx": 1}
    assert manifest["summary"]["runtimeMetadataStatusCounts"] == {"ready": 1}
    assert manifest["adapters"][0]["executor"] == "directx"
    assert manifest["adapters"][0]["platformRequirements"]["requiredTools"] == ["dxc"]
    test_case = manifest["tests"][0]
    assert test_case["id"] == "mlx-file-scope-immutable-lookup-u32"
    assert test_case["expectedOutputs"] == [
        {
            "name": "output",
            "kind": "buffer",
            "dtype": "uint32",
            "shape": [4],
            "values": [5, 19, 11, 13],
            "tolerance": {"absolute": 0.0, "relative": 0.0},
        }
    ]
    assert test_case["runtimeAdapter"]["dispatch"] == {
        "entryPoint": "CSMain",
        "globalSize": [4, 1, 1],
    }

    plan = plan_runtime_test_manifest(manifest)
    runtime_execution = plan["testCases"][0]["runtimeExecution"]
    assert runtime_execution["dispatch"] == {
        "entryPoint": "CSMain",
        "workgroupSize": [1, 1, 1],
        "workgroupCount": [4, 1, 1],
        "globalSize": [4, 1, 1],
        "metadata": {"stage": "compute", "source": "fixture", "status": "available"},
    }
    assert runtime_execution["resourceBindings"][0]["binding"] == {
        "id": "buffer|0|output",
        "name": "output",
        "kind": "buffer",
        "type": "RWStructuredBuffer<uint>",
        "set": 0,
        "binding": 0,
        "access": "write",
    }


def test_mlx_workflow_requires_directx_lookup_numerical_execution():
    workflow = (ROOT / ".github" / "workflows" / "mlx-project-porting.yml").read_text(
        encoding="utf-8"
    )
    for watched_path in (
        "tests/fixtures/runtime_verification/**",
        "tests/test_translator/test_native_runtime_drivers.py",
        "tests/test_translator/test_runtime_verification.py",
    ):
        assert workflow.count(f'- "{watched_path}"') == 2

    compile_start = workflow.index("- name: Run MLX project-porting checks")
    runtime_start = workflow.index(
        "- name: Prove Direct3D immutable lookup numerical execution"
    )
    runtime_end = workflow.index("- name: Verify MLX frontier accounting")
    assert compile_start < runtime_start < runtime_end

    compile_step = workflow[compile_start:runtime_start]
    runtime_step = workflow[runtime_start:runtime_end]
    assert "--require-directx-toolchain" in compile_step
    assert "CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST" not in compile_step
    assert "if: runner.os == 'Windows'" in runtime_step
    assert 'CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST: "1"' in runtime_step
    assert "python -m pytest -q -n auto" in runtime_step
    assert "tests/test_translator/test_native_runtime_drivers.py" in runtime_step
    assert (
        "directx_compute_runtime_executes_mlx_file_scope_lookup_on_device"
        in runtime_step
    )
    assert "continue-on-error" not in runtime_step
    assert "|| true" not in runtime_step
    assert 'python -m pip install -e ".[directx-runtime]" pytest-xdist' in workflow


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


def test_build_runtime_test_manifest_records_runtime_metadata_readiness(tmp_path):
    artifact = _native_runtime_artifact(
        runtimeDataStatus={
            "hostInterface": "available",
            "entryPoints": "available",
            "resourceBindings": "partial",
            "parameterBlocks": "not-applicable",
            "dispatch": "unavailable",
        }
    )

    manifest = build_runtime_test_manifest(
        _artifact_report(tmp_path, [artifact]),
        {
            "kind": "crosstl-project-runtime-fixture-metadata",
            "fixtures": [_runtime_fixture(id="runtime-metadata-status")],
        },
    )

    runtime_metadata = manifest["tests"][0]["metadata"]["runtimeMetadata"]
    assert manifest["success"] is True
    assert manifest["diagnosticCounts"] == {"note": 0, "warning": 1, "error": 0}
    assert manifest["summary"]["runtimeMetadataStatusCounts"] == {"incomplete": 1}
    assert runtime_metadata["status"] == "incomplete"
    assert runtime_metadata["source"] == "artifact.runtimeDataStatus"
    assert runtime_metadata["fields"]["parameterBlocks"] == "not-applicable"
    assert runtime_metadata["missingFields"] == {
        "resourceBindings": "partial",
        "dispatch": "unavailable",
    }
    assert manifest["diagnostics"][0]["code"] == (
        "project.runtime-test-manifest.runtime-metadata-incomplete"
    )
    assert manifest["diagnostics"][0]["missingFields"] == {
        "resourceBindings": "partial",
        "dispatch": "unavailable",
    }

    parsed = parse_runtime_test_manifest(manifest)
    assert parsed.test_cases[0].metadata["runtimeMetadata"]["status"] == "incomplete"


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


def test_runtime_parity_native_factories_create_target_adapters():
    adapters = native_runtime_parity_adapters()

    assert set(adapters) == {"directx", "opengl", "vulkan"}
    assert isinstance(adapters["directx"], DirectXRuntimeParityAdapter)
    assert adapters["directx"].runtime.name == "directx-compute-runtime"
    assert isinstance(adapters["opengl"], OpenGLRuntimeParityAdapter)
    assert adapters["opengl"].runtime.name == "opengl-compute-runtime"
    assert isinstance(adapters["vulkan"], VulkanRuntimeParityAdapter)
    assert isinstance(
        native_runtime_parity_adapter("DirectX"), DirectXRuntimeParityAdapter
    )
    with pytest.raises(RuntimeVerificationError, match="not available"):
        native_runtime_parity_adapter("metal")


def test_runtime_parity_native_adapter_reports_unavailable_tooling(tmp_path):
    missing_tool = "crosstl-runtime-tool-that-does-not-exist-1302"
    adapter = OpenGLRuntimeParityAdapter(
        required_tools=(missing_tool,),
        tool_resolver=lambda _tool: None,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == UNAVAILABLE
    assert result["failurePhase"] == "runtime"
    assert result["executor"]["details"]["reasonKind"] == "tool-unavailable"
    assert result["executor"]["details"]["missingTools"] == [missing_tool]


def test_runtime_parity_native_opengl_adapter_compiles_specialization_to_spirv(
    tmp_path,
):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        "#version 450\nlayout(constant_id = 0) const uint tile_size = 1u;\n"
        "void main() {}\n",
        encoding="utf-8",
    )
    calls = []

    def passing_command(command, *, input_text=None):
        assert input_text is None
        calls.append(command)
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(b"\x03\x02#\x07\x00\x00\x00\x00")
        return {"returncode": 0}

    runtime = FakeNativeRuntime()
    adapter = OpenGLRuntimeParityAdapter(
        runtime=runtime,
        required_tools=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=passing_command,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    assert result["status"] == PASSED
    assert calls[0][:7] == (
        "/fake/glslangValidator",
        "--target-env",
        "opengl",
        "-S",
        "comp",
        "-e",
        "main",
    )
    assert calls[0][-3] == "-o"
    assert calls[0][-1] == str(artifact_path.resolve())
    assert runtime.prepared.module_path.suffix == ".spv"
    details = result["executor"]["details"]
    assert details["nativeRuntimeDispatch"]["modulePath"].endswith(".spv")
    assert details["openglRuntimeConstants"] == {
        "specializationConstantCount": 1,
        "specializationConstantIds": [0],
        "uniformConstantCount": 0,
        "uniformConstants": [],
    }
    assert any(
        step["action"] == "compile-glsl-to-opengl-spirv-for-runtime"
        for step in details["adapterSteps"]
    )


def test_runtime_parity_native_opengl_adapter_preserves_uniform_source_path(
    tmp_path,
):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        "#version 450\nuniform uint tile_size;\nvoid main() {}\n",
        encoding="utf-8",
    )
    calls = []

    def passing_command(command, *, input_text=None):
        assert input_text is None
        calls.append(command)
        return {"returncode": 0}

    artifact = _native_runtime_artifact(
        specializationConstants=[
            {
                "name": "tile_size",
                "kind": "uniform",
                "dtype": "uint32",
                "value": 2,
            }
        ]
    )
    runtime = FakeNativeRuntime()
    adapter = OpenGLRuntimeParityAdapter(
        runtime=runtime,
        required_tools=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=passing_command,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [artifact]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    assert result["status"] == PASSED
    assert calls == [
        (
            "/fake/glslangValidator",
            "-S",
            "comp",
            str(artifact_path.resolve()),
        )
    ]
    assert runtime.prepared.module_path == artifact_path.resolve()
    assert result["executor"]["details"]["openglRuntimeConstants"] == {
        "specializationConstantCount": 0,
        "specializationConstantIds": [],
        "uniformConstantCount": 1,
        "uniformConstants": ["tile_size"],
    }


def test_runtime_parity_native_adapter_skips_validation_tooling_when_disabled(
    tmp_path,
):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    adapter = OpenGLRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=("crosstl-runtime-tool-that-does-not-exist-1302",),
        tool_resolver=lambda _tool: None,
        validate=False,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == PASSED
    validation_steps = [
        step
        for step in result["executor"]["details"]["adapterSteps"]
        if step["action"] == "validate-native-runtime-artifact"
    ]
    assert len(validation_steps) == 1
    assert validation_steps[0]["status"] == SKIPPED
    assert validation_steps[0]["details"]["reason"] == "validation-disabled"


def test_runtime_parity_native_adapter_reports_unavailable_platform(tmp_path):
    adapter = OpenGLRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=(),
        supported_platforms=("missing-platform-1302",),
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    assert report["success"] is True
    assert result["status"] == UNAVAILABLE
    assert result["executor"]["details"]["reasonKind"] == "platform-unavailable"
    assert result["executor"]["details"]["requiredPlatforms"] == [
        "missing-platform-1302"
    ]


def test_runtime_parity_native_adapter_reports_setup_failure(tmp_path):
    adapter = OpenGLRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=(),
        validate=False,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(
            tmp_path,
            [_native_runtime_artifact(path="out/opengl/debug/missing.glsl")],
        ),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    diagnostic = result["diagnostics"][0]
    assert report["success"] is False
    assert result["status"] == RUNTIME_FAILED
    assert result["failurePhase"] == "runtime-setup"
    assert result["executor"]["details"]["failurePhase"] == "runtime-setup"
    assert diagnostic["code"] == ("project.runtime-verification.adapter-setup-failed")
    assert "does not exist" in diagnostic["message"]


def test_runtime_parity_native_adapter_reports_validation_failure(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")

    def failing_command(command, *, input_text=None):
        assert input_text is None
        assert command[0] == "/fake/glslangValidator"
        return {"returncode": 1, "stderr": "shader validation failed"}

    adapter = OpenGLRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=failing_command,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    diagnostic = result["diagnostics"][0]
    assert report["success"] is False
    assert result["status"] == RUNTIME_FAILED
    assert result["failurePhase"] == "runtime-validation"
    assert result["executor"]["details"]["returncode"] == 1
    assert result["executor"]["details"]["stderr"] == "shader validation failed"
    assert diagnostic["code"] == (
        "project.runtime-verification.adapter-validation-failed"
    )


def test_runtime_parity_native_adapter_reports_dispatch_failure(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    adapter = OpenGLRuntimeParityAdapter(
        runtime=FakeNativeRuntime(dispatch_error=ValueError("dispatch rejected")),
        required_tools=(),
        validate=False,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    diagnostic = result["diagnostics"][0]
    assert report["success"] is False
    assert result["status"] == RUNTIME_FAILED
    assert result["failurePhase"] == "runtime-dispatch"
    assert result["executor"]["details"]["failurePhase"] == "runtime-dispatch"
    assert diagnostic["code"] == (
        "project.runtime-verification.adapter-dispatch-failed"
    )
    assert "dispatch rejected" in diagnostic["message"]


def test_runtime_parity_native_adapter_reports_numerical_mismatch(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    runtime = FakeNativeRuntime(
        outputs={
            "out": {
                "dtype": "float32",
                "shape": [2],
                "values": [11.0, 23.0],
            }
        }
    )
    adapter = OpenGLRuntimeParityAdapter(
        runtime=runtime,
        required_tools=(),
        validate=False,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )

    result = report["results"][0]
    assert report["success"] is False
    assert result["status"] == COMPARISON_FAILED
    assert result["failurePhase"] == "comparison"
    assert result["comparisons"][0]["firstMismatch"]["index"] == 1
    assert runtime.prepared.buffers["out"].source == "expectedOutput"
    assert result["diagnostics"][0]["code"] == (
        "project.runtime-verification.output-mismatch"
    )


def test_runtime_parity_native_vulkan_adapter_assembles_and_validates_spirv(
    tmp_path,
):
    artifact_path = tmp_path / "out" / "vulkan" / "debug" / "add.spvasm"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("; SPIR-V\n", encoding="utf-8")
    calls = []

    def passing_command(command, *, input_text=None):
        assert input_text is None
        calls.append(command)
        return {"returncode": 0}

    adapter = VulkanRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=passing_command,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(
            tmp_path,
            [
                _native_runtime_artifact(
                    path="out/vulkan/debug/add.spvasm", target="vulkan"
                )
            ],
        ),
        _native_runtime_manifest(target="vulkan"),
        executors={"vulkan": adapter},
    )

    assert report["results"][0]["status"] == PASSED
    assert calls[0][:3] == (
        "/fake/spirv-as",
        str(artifact_path.resolve()),
        "-o",
    )
    assert calls[1][0] == "/fake/spirv-val"
    assert calls[1][1].endswith(".spv")


def test_runtime_parity_native_directx_adapter_compiles_hlsl(tmp_path):
    artifact_path = tmp_path / "out" / "directx" / "debug" / "add.hlsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("[numthreads(2,1,1)] void main() {}\n", encoding="utf-8")
    calls = []

    def passing_command(command, *, input_text=None):
        assert input_text is None
        calls.append(command)
        return {"returncode": 0}

    adapter = DirectXRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=(),
        supported_platforms=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=passing_command,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(
            tmp_path,
            [
                _native_runtime_artifact(
                    path="out/directx/debug/add.hlsl", target="directx"
                )
            ],
        ),
        _native_runtime_manifest(target="directx"),
        executors={"directx": adapter},
    )

    assert report["results"][0]["status"] == PASSED
    assert calls[0][:-2] == ("/fake/dxc", "-T", "cs_6_0", "-E", "main", "-Fo")
    assert "-enable-16bit-types" not in calls[0]
    assert Path(calls[0][-2]).name == "add.dxil"
    assert calls[0][-1] == str(artifact_path.resolve())


def test_runtime_parity_native_directx_adapter_enables_native_16bit_hlsl(tmp_path):
    artifact_path = tmp_path / "out" / "directx" / "debug" / "add.hlsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        "RWStructuredBuffer<uint16_t> output : register(u0);\n"
        "[numthreads(2,1,1)] void main() { output[0] = uint16_t(1); }\n",
        encoding="utf-8",
    )
    calls = []

    def passing_command(command, *, input_text=None):
        assert input_text is None
        calls.append(command)
        return {"returncode": 0}

    adapter = DirectXRuntimeParityAdapter(
        runtime=FakeNativeRuntime(),
        required_tools=(),
        supported_platforms=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=passing_command,
    )

    report = verify_runtime_test_manifest(
        _artifact_report(
            tmp_path,
            [
                _native_runtime_artifact(
                    path="out/directx/debug/add.hlsl", target="directx"
                )
            ],
        ),
        _native_runtime_manifest(target="directx"),
        executors={"directx": adapter},
    )

    assert report["results"][0]["status"] == PASSED
    assert calls[0][:-2] == (
        "/fake/dxc",
        "-T",
        "cs_6_2",
        "-enable-16bit-types",
        "-E",
        "main",
        "-Fo",
    )
    assert Path(calls[0][-2]).name == "add.dxil"
    assert calls[0][-1] == str(artifact_path.resolve())


def test_default_runtime_test_adapters_cover_native_platform_classes():
    adapters = default_runtime_test_adapters()

    platform_classes = {
        adapter.platform_requirements.platform_class for adapter in adapters
    }
    assert "native-graphics" in platform_classes
    assert "native-compute" in platform_classes
    assert any(adapter.target == "opengl" for adapter in adapters)
    assert any(adapter.target == "cuda" for adapter in adapters)
