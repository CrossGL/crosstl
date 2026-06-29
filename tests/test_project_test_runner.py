import json
import subprocess
import sys
from pathlib import Path

from crosstl.project import (
    PROJECT_TEST_RUNNER_INSPECTION_KIND,
    PROJECT_TEST_RUNNER_PLAN_KIND,
    PROJECT_TEST_RUNNER_REPORT_KIND,
    build_project_test_runner_plan,
    execute_project_test_runner_plan,
    inspect_project_test_runner_plan,
)
from crosstl.project.runtime_verification import RuntimeParityAdapter

ROOT = Path(__file__).resolve().parents[1]


def _artifact_report(tmp_path, artifacts):
    return {
        "kind": "crosstl-runtime-artifact-manifest",
        "project": {"root": str(tmp_path), "targets": ["opengl"]},
        "artifacts": artifacts,
    }


def _artifact(**overrides):
    artifact = {
        "id": "opengl|kernels/add.cgl|debug",
        "source": "kernels/add.cgl",
        "path": "out/opengl/add.glsl",
        "target": "opengl",
        "stage": "compute",
        "variant": "debug",
        "status": "translated",
        "entryPoints": [{"name": "main", "stage": "compute"}],
        "resourceBindings": [
            {"name": "lhs", "kind": "buffer", "binding": 0},
            {"name": "out", "kind": "buffer", "binding": 1},
        ],
        "dispatch": {
            "status": "available",
            "workgroups": [{"entryPoint": "main", "workgroupSize": [1, 1, 1]}],
        },
    }
    artifact.update(overrides)
    return artifact


def _manifest(missing_tool="crosstl-test-runner-tool-that-does-not-exist"):
    return {
        "kind": "crosstl-project-runtime-test-manifest",
        "adapters": [
            {
                "id": "opengl-native",
                "target": "opengl",
                "executor": "opengl-native",
                "platformRequirements": {"requiredTools": [missing_tool]},
            }
        ],
        "tests": [
            {
                "id": "add-fixture",
                "selector": {
                    "source": "kernels/add.cgl",
                    "target": "opengl",
                    "variant": "debug",
                },
                "adapter": "opengl-native",
                "inputs": [{"name": "lhs", "values": [1.0]}],
                "expectedOutputs": [{"name": "out", "values": [2.0]}],
                "metadata": {"upstreamTestName": "project.tests.test_add"},
            }
        ],
    }


def test_project_test_runner_plan_records_handoff_commands_and_provenance(tmp_path):
    missing_tool = "crosstl-test-runner-tool-that-does-not-exist"
    package_path = tmp_path / "runtime-package.json"

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact(toolchainRuns=[{"stderr": "warn"}])]),
        _manifest(missing_tool),
        handoff_packages=[package_path],
        selected_targets=["OpenGL"],
        test_commands=[
            {
                "name": "upstream add",
                "command": [sys.executable, "-c", "print('ok')"],
                "targets": ["opengl"],
                "adapter": "opengl-native",
                "fixture": "add-fixture",
                "expectedArtifacts": ["out/opengl/add.glsl"],
            }
        ],
        expected_artifacts=["out/opengl/add.glsl"],
        project_root=tmp_path,
    )

    assert plan["kind"] == PROJECT_TEST_RUNNER_PLAN_KIND
    assert plan["selectedTargets"] == ["opengl"]
    assert plan["runtimeHandoffPackages"][0]["available"] is False
    assert plan["testCommands"][0]["status"] == "skipped"
    assert missing_tool in plan["testCommands"][0]["diagnostics"][0]["missingTools"]
    runtime_test = plan["runtimeTests"][0]
    assert runtime_test["provenance"]["sourceFile"] == "kernels/add.cgl"
    assert runtime_test["provenance"]["generatedArtifact"] == "out/opengl/add.glsl"
    assert runtime_test["provenance"]["backend"] == "opengl"
    assert runtime_test["provenance"]["upstreamTestName"] == "project.tests.test_add"


def test_runtime_derived_command_inherits_unavailable_adapter_requirements(tmp_path):
    missing_tool = "crosstl-test-runner-runtime-tool-that-does-not-exist"
    manifest = _manifest(missing_tool)
    manifest["tests"][0]["metadata"]["testCommand"] = ["true"]

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        manifest,
        selected_targets=["opengl"],
        project_root=tmp_path,
    )

    runtime_test = plan["runtimeTests"][0]
    command = plan["testCommands"][0]

    assert runtime_test["status"] == "skipped"
    assert command["adapter"] == "opengl-native"
    assert command["requiredAdapters"] == ["opengl-native"]
    assert command["status"] == "skipped"
    assert missing_tool in command["requiredTools"]
    assert missing_tool in command["diagnostics"][0]["missingTools"]
    assert set(command["diagnostics"][0]["missingTools"]) == set(
        runtime_test["diagnostics"][0]["missingTools"]
    )


def test_project_test_runner_plans_mapping_adapter_references(tmp_path):
    missing_tool = "crosstl-test-runner-mapping-tool-that-does-not-exist"

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "mapping adapter command",
                "command": ["true"],
                "targets": ["opengl"],
                "adapter": {
                    "id": "opengl-mapped",
                    "platformRequirements": {"requiredTools": [missing_tool]},
                },
            }
        ],
        project_root=tmp_path,
    )

    command = plan["testCommands"][0]

    assert command["adapter"] == "opengl-mapped"
    assert command["requiredAdapters"] == ["opengl-mapped"]
    assert command["status"] == "skipped"
    assert command["requiredTools"] == [missing_tool]
    assert command["diagnostics"][0]["adapter"] == "opengl-mapped"


def test_project_test_runner_executes_available_project_command(tmp_path):
    output_file = tmp_path / "command-output.txt"
    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "write output",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(output_file)!r}).write_text('ok')",
                ],
                "targets": ["opengl"],
                "expectedArtifacts": [str(output_file)],
            }
        ],
        project_root=tmp_path,
    )

    report = execute_project_test_runner_plan(
        plan,
        project_root=tmp_path,
        run_runtime_tests=False,
    )

    assert report["kind"] == PROJECT_TEST_RUNNER_REPORT_KIND
    assert report["success"] is True
    assert report["results"][0]["status"] == "passed"
    assert report["results"][0]["logs"][0]["returncode"] == 0
    assert output_file.read_text(encoding="utf-8") == "ok"


def test_project_test_runner_executes_runtime_tests_with_supplied_executor(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    artifact_report_path = tmp_path / "artifact-report.json"
    artifact_report_path.write_text(
        json.dumps(
            _artifact_report(
                tmp_path,
                [
                    _artifact(
                        path="out/opengl/add.glsl",
                        dispatch={
                            "entryPoint": "main",
                            "globalSize": [1, 1, 1],
                            "workgroupSize": [1, 1, 1],
                        },
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "kind": "crosstl-project-runtime-test-manifest",
        "adapters": [
            {
                "id": "fixture-runtime",
                "executor": "fixture-runtime",
                "adapterKind": "fixture-runtime-parity",
                "platformRequirements": {"requiredTools": []},
            }
        ],
        "tests": [
            {
                "id": "increment-fixture",
                "selector": {
                    "source": "kernels/add.cgl",
                    "target": "opengl",
                    "variant": "debug",
                },
                "adapter": "fixture-runtime",
                "inputs": [
                    {
                        "name": "lhs",
                        "dtype": "float32",
                        "shape": [1],
                        "values": [1.0],
                    }
                ],
                "expectedOutputs": [
                    {
                        "name": "out",
                        "dtype": "float32",
                        "shape": [1],
                        "values": [2.0],
                    }
                ],
            }
        ],
    }

    class IncrementRuntime(RuntimeParityAdapter):
        name = "increment-runtime"
        target = "opengl"

        def prepare_buffers(self, state):
            return dict(state.resource_values)

        def dispatch(self, state, prepared_buffers):
            assert state.plan.dispatch.entry_point == "main"
            assert prepared_buffers["out"] is None
            return {"out": [value + 1.0 for value in prepared_buffers["lhs"]]}

        def collect_outputs(self, state, dispatch_result):
            return {
                name: {
                    "dtype": "float32",
                    "shape": [len(values)],
                    "values": values,
                }
                for name, values in dispatch_result.items()
            }

    plan = build_project_test_runner_plan(
        artifact_report_path,
        manifest,
        project_root=tmp_path,
    )

    report = execute_project_test_runner_plan(
        plan,
        project_root=tmp_path,
        runtime_executors={"fixture-runtime": IncrementRuntime()},
    )

    runtime_report = report["runtimeTestReport"]
    runtime_result = runtime_report["results"][0]
    assert report["success"] is True
    assert runtime_report["success"] is True
    assert plan["runtimeTestPlan"]["testCases"][0]["platformRequirements"] == {}
    assert runtime_result["status"] == "passed"
    assert (
        runtime_result["executor"]["details"]["runtimeParityAdapter"]["runtimeAdapter"]
        == "increment-runtime"
    )
    assert report["summary"]["passedCount"] == 1


def test_project_cli_execute_test_runner_loads_runtime_executor(tmp_path):
    artifact_path = tmp_path / "out" / "opengl" / "add.glsl"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    artifact_report_path = tmp_path / "artifact-report.json"
    artifact_report_path.write_text(
        json.dumps(
            _artifact_report(
                tmp_path,
                [
                    _artifact(
                        path="out/opengl/add.glsl",
                        dispatch={
                            "entryPoint": "main",
                            "globalSize": [1, 1, 1],
                            "workgroupSize": [1, 1, 1],
                        },
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "kind": "crosstl-project-runtime-test-manifest",
        "adapters": [
            {
                "id": "fixture-runtime",
                "executor": "fixture-runtime",
                "adapterKind": "fixture-runtime-parity",
                "platformRequirements": {"requiredTools": []},
            }
        ],
        "tests": [
            {
                "id": "increment-fixture",
                "selector": {
                    "source": "kernels/add.cgl",
                    "target": "opengl",
                    "variant": "debug",
                },
                "adapter": "fixture-runtime",
                "inputs": [
                    {
                        "name": "lhs",
                        "dtype": "float32",
                        "shape": [1],
                        "values": [1.0],
                    }
                ],
                "expectedOutputs": [
                    {
                        "name": "out",
                        "dtype": "float32",
                        "shape": [1],
                        "values": [2.0],
                    }
                ],
            }
        ],
    }
    plan = build_project_test_runner_plan(
        artifact_report_path,
        manifest,
        project_root=tmp_path,
    )
    plan_path = tmp_path / "test-runner-plan.json"
    report_path = tmp_path / "test-runner-report.json"
    executor_path = tmp_path / "fixture_runtime.py"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    executor_path.write_text(
        """
class IncrementRuntime:
    name = "increment-runtime"

    def prepare_buffers(self, state):
        return dict(state.resource_values)

    def dispatch(self, state, prepared_buffers):
        return {"out": [value + 1.0 for value in prepared_buffers["lhs"]]}

    def collect_outputs(self, state, dispatch_result):
        return {
            name: {
                "dtype": "float32",
                "shape": [len(values)],
                "values": values,
            }
            for name, values in dispatch_result.items()
        }
""".lstrip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "execute-test-runner",
            str(plan_path),
            "--project-root",
            str(tmp_path),
            "--runtime-executor",
            f"fixture-runtime={executor_path}:IncrementRuntime",
            "--output",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runtime_result = report["runtimeTestReport"]["results"][0]
    assert report["success"] is True
    assert runtime_result["status"] == "passed"
    assert (
        runtime_result["executor"]["details"]["runtimeParityAdapter"]["runtimeAdapter"]
        == "increment-runtime"
    )


def test_project_test_runner_inspection_summarizes_unavailable_adapters(tmp_path):
    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        _manifest(),
        project_root=tmp_path,
    )

    inspection = inspect_project_test_runner_plan(plan)

    assert inspection["kind"] == PROJECT_TEST_RUNNER_INSPECTION_KIND
    assert inspection["summary"]["unavailableAdapterCount"] == 1
    assert inspection["summary"]["runtimeTestCount"] == 1


def test_project_cli_test_runner_plan_text_uses_mlx_fixture(tmp_path):
    fixture_dir = ROOT / "tests" / "fixtures" / "runtime_verification" / "mlx"
    artifact_report = fixture_dir / "reduced_binary_add.artifacts.json"
    fixture_metadata = fixture_dir / "reduced_binary_add.fixture-metadata.json"
    manifest_path = tmp_path / "runtime-test-manifest.json"

    manifest_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "runtime-test-manifest",
            str(artifact_report),
            str(fixture_metadata),
            "--project-root",
            str(ROOT),
            "--output",
            str(manifest_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert manifest_result.returncode == 0, manifest_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "test-runner-plan",
            str(artifact_report),
            "--runtime-test-manifest",
            str(manifest_path),
            "--handoff-package",
            str(tmp_path / "host-integration"),
            "--target",
            "metal",
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
    assert "Project test-runner plan:" in result.stdout
    assert "Selected targets: metal" in result.stdout
    assert "mlx.core.add reduced f32" in result.stdout
    assert "Runtime handoff packages:" in result.stdout
