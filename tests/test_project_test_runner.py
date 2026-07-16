import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    PROJECT_TEST_RUNNER_INSPECTION_KIND,
    PROJECT_TEST_RUNNER_PLAN_KIND,
    PROJECT_TEST_RUNNER_REPORT_KIND,
    build_project_test_runner_plan,
    execute_project_test_runner_plan,
    inspect_project_test_runner_plan,
)
from crosstl.project.runtime_verification import (
    RuntimeParityAdapter,
    RuntimeVerificationError,
)

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


def test_runtime_adapter_requirements_use_plan_environment(tmp_path, monkeypatch):
    variable_name = "CROSSTL_RUNTIME_ADAPTER_ENV_1722"
    monkeypatch.delenv(variable_name, raising=False)
    manifest = _manifest()
    manifest["adapters"][0].pop("target")
    manifest["adapters"][0]["platformRequirements"] = {
        "requiredEnvironment": [variable_name]
    }
    manifest["tests"][0]["metadata"]["testCommand"] = [
        sys.executable,
        "-c",
        "",
    ]

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        manifest,
        environment={"variables": {variable_name: "configured"}},
        project_root=tmp_path,
    )

    assert plan["adapters"][0]["availability"]["available"] is True
    assert plan["adapters"][0]["availability"]["missingEnvironment"] == []
    assert plan["runtimeTests"][0]["status"] == "planned"
    assert plan["testCommands"][0]["status"] == "planned"


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


def test_project_test_runner_applies_environment_and_ordered_setup(
    tmp_path, monkeypatch
):
    variable_name = "CROSSTL_TEST_RUNNER_VALUE_1722"
    variable_value = "runner-secret-value-1722"
    monkeypatch.delenv(variable_name, raising=False)
    work_dir = tmp_path / "upstream"
    work_dir.mkdir()
    setup_script = (
        "import os, sys; "
        "from pathlib import Path; "
        "path = Path('setup-order.txt'); "
        "path.write_text(path.read_text() + sys.argv[1] + ':' + "
        f"os.environ[{variable_name!r}] + '\\n' if path.exists() else "
        f"sys.argv[1] + ':' + os.environ[{variable_name!r}] + '\\n')"
    )
    test_script = (
        "import json, os; "
        "from pathlib import Path; "
        "Path('result.json').write_text(json.dumps({"
        f"'value': os.environ[{variable_name!r}], "
        "'order': Path('setup-order.txt').read_text()}))"
    )
    first_arg = "first argument; $SHELL remains data"

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "consume configured environment",
                "command": [sys.executable, "-c", test_script],
                "requiredEnvironment": [variable_name],
            }
        ],
        environment={
            "cwd": "upstream",
            "variables": {variable_name: variable_value},
            "setupCommands": [
                [sys.executable, "-c", setup_script, first_arg],
                [sys.executable, "-c", setup_script, "second"],
            ],
        },
        project_root=tmp_path,
    )

    assert plan["testCommands"][0]["status"] == "planned"
    report = execute_project_test_runner_plan(
        plan,
        project_root=tmp_path,
        run_runtime_tests=False,
    )

    result_payload = json.loads((work_dir / "result.json").read_text(encoding="utf-8"))
    assert result_payload == {
        "value": variable_value,
        "order": f"{first_arg}:{variable_value}\nsecond:{variable_value}\n",
    }
    assert report["success"] is True
    assert report["results"][0]["status"] == "passed"
    setup = report["environmentSetup"]
    assert setup["status"] == "passed"
    assert setup["cwd"] == str(work_dir)
    assert setup["appliedVariables"] == [variable_name]
    assert [log["returncode"] for log in setup["logs"]] == [0, 0]
    assert setup["logs"][0]["command"][-1] == first_arg
    assert variable_value not in json.dumps(report)
    assert variable_name not in os.environ


def test_project_test_runner_plan_environment_satisfies_required_variables(
    tmp_path, monkeypatch
):
    supplied = "CROSSTL_TEST_RUNNER_SUPPLIED_1722"
    missing = "CROSSTL_TEST_RUNNER_MISSING_1722"
    monkeypatch.delenv(supplied, raising=False)
    monkeypatch.delenv(missing, raising=False)
    output_file = tmp_path / "supplied.txt"

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "supplied variable",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(output_file)!r}).write_text('ok')",
                ],
                "requiredEnvironment": [supplied],
            },
            {
                "name": "missing variable",
                "command": [sys.executable, "-c", "raise SystemExit(99)"],
                "requiredEnvironment": [missing],
            },
        ],
        environment={"variables": {supplied: "configured"}},
        project_root=tmp_path,
    )

    assert [command["status"] for command in plan["testCommands"]] == [
        "planned",
        "skipped",
    ]
    assert plan["testCommands"][1]["diagnostics"][0]["missingEnvironment"] == [missing]

    report = execute_project_test_runner_plan(
        plan,
        project_root=tmp_path,
        run_runtime_tests=False,
    )

    assert output_file.read_text(encoding="utf-8") == "ok"
    assert [result["status"] for result in report["results"]] == [
        "passed",
        "skipped",
    ]
    assert report["results"][1]["failurePhase"] == "adapter-availability"


def test_project_test_runner_reports_setup_failure_and_skips_dependent_commands(
    tmp_path,
):
    secret = "setup-secret-value-1722"
    order_file = tmp_path / "setup-order.txt"
    skipped_setup_file = tmp_path / "setup-after-failure.txt"
    skipped_test_file = tmp_path / "test-after-failure.txt"
    append_script = (
        "from pathlib import Path; "
        f"path = Path({str(order_file)!r}); "
        "path.write_text(path.read_text() + 'first\\n' if path.exists() else 'first\\n')"
    )
    failure_script = (
        "import os, sys; "
        "print('setup stdout'); "
        "print(os.environ['CROSSTL_SETUP_SECRET_1722'], file=sys.stderr); "
        "raise SystemExit(7)"
    )

    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "must not run",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(skipped_test_file)!r}).touch()",
                ],
            }
        ],
        environment={
            "variables": {"CROSSTL_SETUP_SECRET_1722": secret},
            "setupCommands": [
                [sys.executable, "-c", append_script],
                [sys.executable, "-c", failure_script],
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(skipped_setup_file)!r}).touch()",
                ],
            ],
        },
        project_root=tmp_path,
    )

    report = execute_project_test_runner_plan(
        plan,
        project_root=tmp_path,
        run_runtime_tests=False,
    )

    setup = report["environmentSetup"]
    assert report["success"] is False
    assert report["summary"]["environmentSetupFailedCount"] == 1
    assert setup["status"] == "runtime-failed"
    assert setup["failurePhase"] == "environment-setup"
    assert setup["returncode"] == 7
    assert setup["stdout"] == "setup stdout\n"
    assert setup["stderr"] == "<redacted>\n"
    assert len(setup["logs"]) == 2
    assert report["results"][0]["status"] == "skipped"
    assert report["results"][0]["failurePhase"] == "environment-setup"
    assert order_file.read_text(encoding="utf-8") == "first\n"
    assert not skipped_setup_file.exists()
    assert not skipped_test_file.exists()
    assert secret not in json.dumps(report)


def test_project_test_runner_does_not_interpret_string_setup_commands_with_shell(
    tmp_path,
):
    shell_output = tmp_path / "implicit-shell-output.txt"
    command = (
        f'{sys.executable} -c "from pathlib import Path; '
        f'Path({str(shell_output)!r}).touch()"'
    )
    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        environment={"setupCommands": [command]},
        project_root=tmp_path,
    )

    report = execute_project_test_runner_plan(
        plan,
        project_root=tmp_path,
        run_runtime_tests=False,
    )

    assert report["success"] is False
    assert report["environmentSetup"]["command"] == [command]
    assert report["environmentSetup"]["returncode"] is None
    assert not shell_output.exists()


@pytest.mark.parametrize("escape_kind", ["environment", "command"])
def test_project_test_runner_rejects_working_directory_escape(tmp_path, escape_kind):
    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[{"name": "safe", "command": [sys.executable, "-c", ""]}],
        project_root=tmp_path,
    )
    if escape_kind == "environment":
        plan["environment"]["cwd"] = ".."
    else:
        plan["testCommands"][0]["cwd"] = "../outside"

    with pytest.raises(RuntimeVerificationError, match="escapes project root"):
        execute_project_test_runner_plan(
            plan,
            project_root=tmp_path,
            run_runtime_tests=False,
        )


def test_project_cli_round_trips_test_runner_environment(tmp_path, monkeypatch):
    variable_name = "CROSSTL_CLI_TEST_RUNNER_VALUE_1722"
    variable_value = "cli-runner-secret-value-1722"
    monkeypatch.delenv(variable_name, raising=False)
    work_dir = tmp_path / "upstream"
    work_dir.mkdir()
    artifact_report_path = tmp_path / "artifact-report.json"
    config_path = tmp_path / "test-runner-config.json"
    plan_path = tmp_path / "test-runner-plan.json"
    report_path = tmp_path / "test-runner-report.json"
    artifact_report_path.write_text(
        json.dumps(_artifact_report(tmp_path, [_artifact()])), encoding="utf-8"
    )
    setup_script = (
        "import sys; from pathlib import Path; "
        "path = Path('setup-order.txt'); "
        "path.write_text((path.read_text() if path.exists() else '') + "
        "sys.argv[1] + '\\n')"
    )
    test_script = (
        "import json, os; from pathlib import Path; "
        "Path('result.json').write_text(json.dumps({"
        f"'value': os.environ[{variable_name!r}], "
        "'order': Path('setup-order.txt').read_text()}))"
    )
    config_path.write_text(
        json.dumps(
            {
                "environment": {
                    "cwd": "upstream",
                    "variables": {variable_name: variable_value},
                    "setupCommands": [
                        [sys.executable, "-c", setup_script, "first"],
                        [sys.executable, "-c", setup_script, "second"],
                    ],
                },
                "testCommands": [
                    {
                        "name": "serialized environment",
                        "command": [sys.executable, "-c", test_script],
                        "requiredEnvironment": [variable_name],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "test-runner-plan",
            str(artifact_report_path),
            "--test-config",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--output",
            str(plan_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert plan_result.returncode == 0, plan_result.stdout or plan_result.stderr
    serialized_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert serialized_plan["testCommands"][0]["status"] == "planned"

    execute_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "execute-test-runner",
            str(plan_path),
            "--project-root",
            str(tmp_path),
            "--no-runtime-tests",
            "--output",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert execute_result.returncode == 0, (
        execute_result.stdout or execute_result.stderr
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["success"] is True
    assert report["environmentSetup"]["appliedVariables"] == [variable_name]
    assert [log["returncode"] for log in report["environmentSetup"]["logs"]] == [
        0,
        0,
    ]
    assert json.loads((work_dir / "result.json").read_text(encoding="utf-8")) == {
        "value": variable_value,
        "order": "first\nsecond\n",
    }
    assert variable_value not in json.dumps(report)


def test_project_cli_serialized_plan_reports_environment_setup_failure(tmp_path):
    skipped_file = tmp_path / "skipped-command.txt"
    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "must not run",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(skipped_file)!r}).touch()",
                ],
            }
        ],
        environment={
            "setupCommands": [
                [
                    sys.executable,
                    "-c",
                    "import sys; print('failed setup'); raise SystemExit(9)",
                ]
            ]
        },
        project_root=tmp_path,
    )
    plan_path = tmp_path / "test-runner-plan.json"
    report_path = tmp_path / "test-runner-report.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "execute-test-runner",
            str(plan_path),
            "--project-root",
            str(tmp_path),
            "--no-runtime-tests",
            "--output",
            str(report_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["environmentSetup"]["failurePhase"] == "environment-setup"
    assert report["environmentSetup"]["returncode"] == 9
    assert report["environmentSetup"]["stdout"] == "failed setup\n"
    assert report["results"][0]["status"] == "skipped"
    assert not skipped_file.exists()


def test_project_cli_serialized_plan_rejects_environment_path_escape(tmp_path):
    plan = build_project_test_runner_plan(
        _artifact_report(tmp_path, [_artifact()]),
        test_commands=[
            {
                "name": "must not run",
                "command": [sys.executable, "-c", "raise SystemExit(99)"],
            }
        ],
        project_root=tmp_path,
    )
    plan["environment"]["cwd"] = ".."
    plan_path = tmp_path / "escaped-test-runner-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "execute-test-runner",
            str(plan_path),
            "--project-root",
            str(tmp_path),
            "--no-runtime-tests",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "escapes project root" in result.stdout


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


def test_project_cli_execute_test_runner_loads_native_runtime_driver(tmp_path):
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
                "id": "native-opengl",
                "executor": "opengl",
                "adapterKind": "opengl-native-runtime",
                "platformRequirements": {"requiredTools": []},
            }
        ],
        "tests": [
            {
                "id": "native-increment-fixture",
                "selector": {
                    "source": "kernels/add.cgl",
                    "target": "opengl",
                    "variant": "debug",
                },
                "adapter": "native-opengl",
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
    runtime_path = tmp_path / "native_runtime.py"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runtime_path.write_text(
        """
class OpenGLFixtureRuntime:
    name = "fixture-opengl-driver"

    def dispatch(self, adapter, state, request):
        assert adapter.target == "opengl"
        assert request.entry_point == "main"
        assert request.buffers["out"].source == "expectedOutput"
        return {"out": [value + 1.0 for value in request.buffers["lhs"].value]}
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
            "--native-runtime-adapter",
            f"opengl={runtime_path}:OpenGLFixtureRuntime",
            "--no-native-runtime-validation",
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
    details = runtime_result["executor"]["details"]
    assert report["success"] is True
    assert runtime_result["status"] == "passed"
    assert details["runtimeParityAdapter"]["runtimeAdapter"] == "opengl-native-runtime"
    assert details["nativeRuntimeDispatch"]["target"] == "opengl"


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
