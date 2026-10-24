"""Checks that the MLX workflow keeps every required native test and fits its runner limits.

These guards moved here from ``tests/test_ci_workflows.py`` unchanged, apart from importing the workflow-text
helpers from ``tests.workflow_text``.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

from tests import workflow_text
from tests.ci_helpers import bounded_step_deadline

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = ROOT / ".github" / "workflows"


@pytest.mark.parametrize(
    "job,overhead,runner_os",
    [
        ("opengl", 300, "Linux"),
        ("portable-host", 1800, "Linux"),
        ("portable-host", 1800, "macOS"),
        ("portable-host", 1800, "Windows"),
    ],
)
def test_project_native_job_deadlines_fit_runner_limit(job, overhead, runner_os):
    workflow = yaml.safe_load((WORKFLOW_DIR / "demo-project-testing.yml").read_text())
    definition = workflow["jobs"][job]

    def deadline(step):
        return bounded_step_deadline(step, runner_os)

    steps = definition["steps"]
    failure_steps = [step for step in steps if "failure()" in step.get("if", "")]
    normal_deadline = sum(deadline(step) for step in steps if step not in failure_steps)
    paths = [normal_deadline]
    if job == "portable-host":
        assert len(failure_steps) == 1
        comparison = failure_steps[0]
        assert (
            comparison["if"]
            == "failure() && runner.os == 'Windows' && !matrix.warp_qualification && steps.collective-helpers.outcome == 'failure'"
        )
        failed_index = next(
            index
            for index, step in enumerate(steps)
            if step.get("id") == "collective-helpers"
        )
        assert steps.index(comparison) > failed_index
        # Independent numerical and host batches retain their deadlines after a failure.
        independent = [
            step
            for step in steps[failed_index + 1 :]
            if "!cancelled()" in step.get("if", "")
        ]
        assert len(independent) == 34
        assert independent == [
            step for step in steps[failed_index + 1 : -1] if step != comparison
        ]
        assert steps[-1]["name"] == "Retain native execution evidence"
        assert steps[-1]["if"] == "always()"
        paths.append(
            sum(deadline(step) for step in steps[: failed_index + 1])
            + (comparison["timeout-minutes"] * 60 if runner_os == "Windows" else 0)
            + sum(deadline(step) for step in independent)
        )
    else:
        assert not failure_steps
    assert normal_deadline
    assert max(paths) + overhead < definition["timeout-minutes"] * 60 <= 360 * 60


def test_directx_runtime_comparison_preserves_required_runtime_and_workload():
    workflow = yaml.safe_load((WORKFLOW_DIR / "demo-project-testing.yml").read_text())
    steps = workflow["jobs"]["portable-host"]["steps"]
    comparison = next(step for step in steps if "failure()" in step.get("if", ""))
    assert comparison["timeout-minutes"] == 8
    assert comparison["shell"] == "pwsh"
    required_gate = next(
        step for step in steps if step.get("id") == "collective-helpers"
    )
    assert comparison["env"] == {
        "CROSTL_REQUIRE_DIRECTX_NARROW_BUFFERS": "1",
        "CROSTL_REQUIRE_DIRECTX_COOPERATIVE_MATRIX": "1",
    }
    assert "if" not in required_gate and "continue-on-error" not in required_gate
    assert not comparison.get("continue-on-error")
    script = comparison["run"]
    for value in (
        'version = "1.0.13"',
        "63231c48b0573ba4c078f69cd10a4059a0fee3427107b8219e5e80ab75bd304b",
        "0621056518e047fd2fa9f75f03a49fed42d7158dee77c11c441cd34d32e93638",
        "$remaining = 360 - [int][Math]::Ceiling($clock.Elapsed.TotalSeconds)",
        "--timeout-seconds $remaining",
        "python -m pytest -q -n auto",
        "@tests",
        "$PSNativeCommandUseErrorActionPreference = $false",
        "$caseResult = $LASTEXITCODE",
        "if ($caseResult -ne 0) { $result = $caseResult }",
        "replacesRequiredGate = $false",
        "exit $result",
    ):
        assert value in script
    resource_command = required_gate["run"].split(
        '--label "Resource values and project execution"', 1
    )[1]
    required_tests = re.findall(r"tests/[^\s\\]+", resource_command)
    compared_tests = re.findall(r'"(tests/[^"\s]+)"', script)
    assert len(required_tests) == 34
    assert "tests/test_translator/test_metadata_preconditions.py" in required_tests
    assert "tests/test_translator/test_metadata_elements.py" in required_tests
    assert "tests/test_translator/test_scalar_return_bounds.py" in required_tests
    assert "tests/test_translator/test_aggregate_state_bounds.py" in required_tests
    assert "tests/test_translator/test_helper_local_returns.py" in required_tests
    assert "tests/test_translator/test_helper_array_returns.py" in required_tests
    assert "tests/test_translator/test_loop_guard_bounds.py" in required_tests
    assert "tests/test_translator/test_pointer_index_bounds.py" in required_tests
    assert "tests/test_translator/test_runtime_input_observations.py" in required_tests
    assert compared_tests == [
        "tests/test_translator/test_directx_cooperative_matrix.py::test_precise_scalar_control_executes",
        "tests/test_translator/test_directx_cooperative_matrix.py::test_software_matrix_executes",
        "tests/test_translator/test_directx_narrow_buffer_extents.py::test_narrow_buffer_extent_execute",
    ]
    assert all(
        test.split("::", 1)[0] in required_gate["run"] for test in compared_tests
    )
    assert 'version = "system"' in script
    assert 'Join-Path $env:SystemRoot "System32/d3d10warp.dll"' in script
    assert "sourceLibrary = $case.dll" in script
    assert "fileVersion = (Get-Item $case.dll).VersionInfo.FileVersion" in script
    pytest_arguments = script.split("-- python -m pytest", 1)[1].splitlines()[0]
    assert not re.search(
        r"(?:^|\s)(?:-k|-m|--keyword|--markexpr)(?:\s|=|$)", pytest_arguments
    )
    assert re.findall(r'@\{ version = "([^"]+)"', script) == [
        "1.0.13",
        "1.0.14.1",
        "1.0.14.2",
        "1.0.15",
        "1.0.15.1",
        "1.0.16",
        "1.0.16.1",
        "1.0.17",
        "1.0.18",
        "1.0.19",
    ]
    assert 'version = "1.0.14"' not in script
    assert script.count("[System.Diagnostics.Stopwatch]::StartNew()") == 1
    assert script.index("[System.Diagnostics.Stopwatch]::StartNew()") < script.index(
        "Invoke-WebRequest"
    )
    assert "-TimeoutSec ([Math]::Min(30, $remaining))" in script
    assert script.index("archive checksum mismatch") < script.index(
        "Copy-Item $installed $backup"
    )
    assert script.index("library checksum mismatch") < script.index(
        "Copy-Item $installed $backup"
    )
    restoration = script.split("} finally {", 1)[1]
    assert "Copy-Item $backup $installed -Force" in restoration
    assert "$restoredDigest -ne $originalDigest" in restoration
    assert "restoration.json" in restoration
    required = next(
        step for step in steps if step.get("name") == "Install pinned DirectX tools"
    )
    assert "microsoft.direct3d.warp/1.0.21/" in required["run"]
    assert "e79c10550449365adf0a9393d97a0df69941e671ab6e952a78d92da066517ca3" in script


def test_native_collective_batches_limit_software_driver_threads():
    workflow = yaml.safe_load((WORKFLOW_DIR / "demo-project-testing.yml").read_text())
    job = workflow["jobs"]["portable-host"]
    step = next(item for item in job["steps"] if item.get("id") == "collective-helpers")
    assert job["env"]["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "2"
    assert step["env"]["LP_NUM_THREADS"] == "2"
    assert "if" not in step and "continue-on-error" not in step
    assert step["run"].count("--timeout-seconds 600") == 2
    assert step["run"].count("-n auto -vv --durations=25") == 2
    batches = step["run"].split("python -m pytest ")[1:]
    assert len(batches) == 2
    for index, (batch, count) in enumerate(zip(batches, (19, 34))):
        selectors = re.findall(r"tests/[^\s\\]+", batch)
        assert len(selectors) == len(set(selectors)) == count
        assert not re.search(r"(?:^|\s)(?:-m|--maxfail)(?:\s|=|$)", batch)
        if index == 0:
            assert not re.search(r"(?:^|\s)-k(?:\s|=|$)", batch)


def test_resource_selection_retains_every_native_opt_in():
    workflow = yaml.safe_load((WORKFLOW_DIR / "demo-project-testing.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["portable-host"]["steps"]
        if step.get("id") == "collective-helpers"
    )
    batch = step["run"].split('--label "Resource values and project execution"', 1)[1]
    expression = "execute or native or original_metal or prepared_uploads or dxc_compilation or fxc_comparison_preserves"
    assert f'-k "{expression}"' in batch
    selectors = re.findall(r"tests/[^\s\\]+", batch)
    required, selected = set(), set()
    for selector in selectors:
        module = selector.split("::")[0]
        tree = ast.parse((ROOT / module).read_text())
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert not any(
            isinstance(node, ast.ClassDef) and node.name.startswith("Test")
            for node in tree.body
        )
        guarded = {
            name
            for name, function in functions.items()
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                for node in ast.walk(function)
            )
        }
        # Include local helper calls and fixtures, not only guards in test bodies.
        while True:
            inherited = {
                name
                for name, function in functions.items()
                if any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in guarded
                    for node in ast.walk(function)
                )
                or any(arg.arg in guarded for arg in function.args.args)
            }
            if inherited <= guarded:
                break
            guarded.update(inherited)
        for name in functions:
            if not name.startswith("test_") or (
                "::" in selector and selector != f"{module}::{name}"
            ):
                continue
            identity = (module, name)
            if name in guarded:
                required.add(identity)
            if any(term in name for term in expression.split(" or ")):
                selected.add(identity)
    assert len(required) == 73
    assert required <= selected
    for name in (
        "test_record_members_execute_natively",
        "test_member_bound_loop_executes_natively",
    ):
        assert ("tests/test_translator/test_metadata_members.py", name) in selected
    assert (
        "tests/test_translator/test_frozen_specializations.py",
        "test_frozen_int32_specialization_native_execution",
    ) in selected
    assert (
        "tests/test_translator/test_pointer_index_bounds.py",
        "test_pointer_array_recurrence_executes_natively",
    ) in selected
    assert (
        "tests/test_translator/test_helper_local_returns.py",
        "test_scalar_local_helpers_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_pointer_index_bounds.py",
        "test_signed_relative_indices_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_pointer_index_bounds.py",
        "test_local_pointer_loops_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_pointer_index_bounds.py",
        "test_pointer_array_bounds_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_pointer_index_bounds.py",
        "test_typed_pointer_operands_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_loop_guard_bounds.py",
        "test_tight_tiled_layout_executes_natively",
    ) in selected
    assert (
        "tests/test_translator/test_loop_guard_bounds.py",
        "test_tail_counter_executes_natively",
    ) in selected
    assert (
        "tests/test_translator/test_helper_array_returns.py",
        "test_vector_array_helpers_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_elements.py",
        "test_array_metadata_executes_natively",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_elements.py",
        "test_checked_element_indices_execute_natively",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_combined_contracts",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_entry_scoped_metadata",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_metadata_bounded_collective_loop",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_metadata_bounded_indices",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_metadata_loop_offsets",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_metadata_bounded_division",
    ) in selected
    assert (
        "tests/test_translator/test_metadata_preconditions.py",
        "test_native_region_metadata_indices",
    ) in selected
    assert (
        "tests/test_translator/test_directx_cooperative_matrix.py",
        "test_precise_scalar_control_executes",
    ) in selected
    assert (
        "tests/test_translator/test_runtime_input_ownership.py",
        "test_prepared_uploads_ignore_late_input_mutation",
    ) in selected


def test_portable_host_subgroup_products_budget_covers_llvmpipe_runs():
    workflow = (WORKFLOW_DIR / "demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate software subgroup products"
    )
    # The 632 OpenGL software-subgroup tests passed in 114-122 s, and two of
    # the last four OpenGL host runs exceeded the earlier 180 s budget at 91%.
    assert '--timeout-seconds 300 --label "Software subgroup products"' in step
    assert "tests/test_translator/test_software_subgroup_product.py" in step
    assert 'CROSTL_REQUIRE_SOFTWARE_SUBGROUP_PRODUCT: "1"' in step


def test_project_arithmetic_selection_retains_every_device_test():
    workflow = (WORKFLOW_DIR / "demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Validate binary32 arithmetic"
    )
    expression = "execute or original_metal or modules_link_together"
    assert f'-k "{expression}"' in step
    # Two llvmpipe workers on Linux took 67-125 s across recent runs.
    assert "--timeout-seconds 300" in step
    assert 'PYTEST_XDIST_AUTO_NUM_WORKERS: "2"' in step
    assert 'LP_NUM_THREADS: "2"' in step
    assert "--dist loadgroup -vv --durations=10" in step
    assert "if:" not in step
    assert "continue-on-error" not in step
    assert 'CROSTL_REQUIRE_FUSED_MATH: "1"' in step
    assert 'CROSTL_REQUIRE_HALF_REMAINDER: "1"' in step
    assert 'CROSTL_REQUIRE_HALF_WIDENING: "1"' in step
    assert 'CROSTL_REQUIRE_ADDITIVE_PROFILE: "1"' in step
    selectors = re.findall(
        r"tests/test_translator/test_[a-z_]+\.py(?:::[a-z_]+)?", step
    )
    assert len(selectors) == len(set(selectors)) == 9
    assert selectors[0] == (
        "tests/test_translator/test_metal_arithmetic_ownership.py::"
        "test_arithmetic_builtin_ownership_executes"
    )
    modules = {selector.split("::")[0] for selector in selectors}
    assert len(modules) == 8
    selected = set()
    required = set()
    for module in modules:
        tree = ast.parse((ROOT / module).read_text())
        for function in tree.body:
            if not isinstance(
                function, ast.FunctionDef
            ) or not function.name.startswith("test_"):
                continue
            identity = (module, function.name)
            if (
                module in selectors or f"{module}::{function.name}" in selectors
            ) and any(term in function.name for term in expression.split(" or ")):
                selected.add(identity)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and any(
                    isinstance(argument, ast.Name) and argument.id == "REQUIRE_ENV"
                    for argument in node.args
                )
                for node in ast.walk(function)
            ):
                required.add(identity)
    assert len(required) == 13
    assert selected == required


def test_native_arithmetic_selection_retains_every_device_test():
    workflow = (WORKFLOW_DIR / "demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate binary32 arithmetic"
    )
    expression = "execute or original_metal or modules_link_together"
    assert f'-k "{expression}"' in step
    assert '-m "not extended_power"' in step
    power_step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned real power"
    )
    assert "-m extended_power" in power_step
    assert "tests/test_translator/test_metal_power.py" in power_step
    assert 'CROSTL_REQUIRE_METAL_POWER: "1"' in power_step
    assert "--timeout-seconds 180" in power_step
    assert "--timeout-seconds 480" in step
    modules = re.findall(r"tests/test_translator/test_[a-z_]+\.py", step)
    assert len(modules) == len(set(modules)) == 11
    assert 'CROSTL_REQUIRE_METAL_POWER: "1"' in step
    assert "tests/test_translator/test_metal_power.py" in modules
    assert 'CROSTL_REQUIRE_MULTIPLICATION_PROFILE: "1"' in step
    assert "tests/test_translator/test_metal_multiplication_profile.py" in modules
    assert 'CROSTL_REQUIRE_CONTRACTION_PROFILE: "1"' in step
    assert "tests/test_translator/test_metal_contraction_profile.py" in modules
    selected = set()
    required = set()
    for module in modules:
        tree = ast.parse((ROOT / module).read_text())
        for function in tree.body:
            if not isinstance(
                function, ast.FunctionDef
            ) or not function.name.startswith("test_"):
                continue
            identity = (module, function.name)
            if any(term in function.name for term in expression.split(" or ")):
                selected.add(identity)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and any(
                    isinstance(argument, ast.Name) and argument.id == "REQUIRE_ENV"
                    for argument in node.args
                )
                for node in ast.walk(function)
            ):
                required.add(identity)
    assert len(required) == 20
    assert (
        "tests/test_translator/test_metal_contraction_profile.py",
        "test_contraction_profiles_execute",
    ) in required
    assert (
        "tests/test_translator/test_metal_power.py",
        "test_power_executes_exact_identity",
    ) in required
    assert (
        "tests/test_translator/test_metal_power.py",
        "test_power_executes_profiled_subnormal_operands",
    ) in required
    assert (
        "tests/test_translator/test_metal_power.py",
        "test_power_accuracy_profile_executes_unchanged_metal_range_boundaries",
    ) in required
    assert selected == required
    power_tree = ast.parse(
        (ROOT / "tests/test_translator/test_metal_power.py").read_text()
    )
    functions = {
        node.name: node for node in power_tree.body if isinstance(node, ast.FunctionDef)
    }
    boundary = functions[
        "test_power_accuracy_profile_executes_unchanged_metal_range_boundaries"
    ]
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "extended_power"
        for node in boundary.decorator_list
    )
    domain = functions["test_power_executes_native_domains"]
    grouped = next(
        node.args[1]
        for node in domain.decorator_list
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "dataset,source_control"
    )
    assert [
        ast.literal_eval(node) for node in grouped.elts if isinstance(node, ast.Tuple)
    ] == [("domain", False), ("domain", True)]
    extended = [node for node in grouped.elts if isinstance(node, ast.Call)]
    assert [tuple(ast.literal_eval(arg) for arg in node.args) for node in extended] == [
        ("finite", False),
        ("finite", True),
        ("extended", False),
        ("portable-domain", False),
    ]
    assert all(
        any(
            keyword.arg == "marks"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "extended_power"
            for keyword in node.keywords
        )
        for node in extended
    )
    full_suite = (WORKFLOW_DIR / "full-tests.yml").read_text()
    assert "runs-on: ubuntu-latest" in full_suite
    run = workflow_text.workflow_job_step_section(
        full_suite, "pytest", "Run complete test suite"
    )
    assert "python -m pytest tests demos/integrations" in run
    arguments = run.split("python -m pytest", 1)[1]
    assert " -m " not in arguments and " -k " not in arguments
