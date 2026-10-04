"""Explicit constrained specializations must execute their own computations."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_backend.test_metal.test_constrained_specialization import CASES, source
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_CONSTRAINED_SPECIALIZATION_RUNTIME"


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_project_selects_constrained_specialization(tmp_path, target, scoped, case):
    (tmp_path / "select.metal").write_text(source(case), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("select.metal",),
            targets=(target,),
            entry_points={"select.metal": ("select_body",)} if scoped else {},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert not report["diagnostics"], report["diagnostics"]
    (artifact,) = report["artifacts"]
    assert artifact["status"] == "translated"
    generated = (tmp_path / artifact["path"]).read_text()
    assert "update_value_int" in generated or "update_value_float" in generated


def _request(root, target, case):
    kind = "int" if case in {"integer", "explicit-integer"} else "float"
    dtype = kind + "32"
    original, descriptor, package = _package(
        root, target, kind, (1, 1, 1), source=source(case), software_subgroups=False
    )
    values = [-4096, -7, 0, 13, 4096]
    guard = 123456
    expected_values = [
        value + (7 if kind == "int" or case.startswith("nondefault") else -7)
        for value in values
    ]
    inputs = {
        "values": {"dtype": dtype, "shape": [5], "values": values},
        "results": {"dtype": dtype, "shape": [8], "values": [guard] * 8},
    }
    expected = _bound_values(
        descriptor,
        {
            "values": inputs["values"],
            "results": {
                "dtype": dtype,
                "shape": [8],
                "values": [guard, *expected_values, guard, guard],
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [5, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("case", CASES)
def test_constrained_specializations_compile(tmp_path, target, case):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _request(tmp_path, target, case)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_constrained_specializations_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required specialization execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="select_body",
    )


def test_constrained_specialization_execution_is_required_on_each_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_constrained_specialization_runtime.py" in step
        timeout = 1800 if name == "Validate general gather and empty arrays" else 1200
        assert f"--timeout-seconds {timeout}" in step and "-n auto" in step
        assert "if:" not in step and "continue-on-error" not in workflow
    for event in ("pull_request", "push"):
        assert "tests/test_translator/**" in ci_coverage.workflow_event_path_filters(
            workflow, event
        )
