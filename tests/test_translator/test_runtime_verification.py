import json
from pathlib import Path

import pytest

import crosstl.project as project_api
from crosstl.project.runtime_verification import (
    COMPARISON_FAILED,
    PASSED,
    RUNTIME_FAILED,
    RUNTIME_VERIFICATION_REPORT_KIND,
    SKIPPED,
    TRANSLATION_FAILED,
    UNAVAILABLE,
    RuntimeAdapterContract,
    RuntimeExecutionError,
    RuntimeExecutorAvailability,
    RuntimeExecutorResult,
    RuntimeExecutorSkipped,
    RuntimeVerificationError,
    compare_runtime_outputs,
    load_runtime_verification_fixtures,
    parse_runtime_verification_fixtures,
    verify_runtime_fixtures,
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
