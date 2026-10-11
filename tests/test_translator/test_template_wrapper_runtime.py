"""Deduced wrapper pointers preserve native buffer writes and default slots."""

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
from tests.ci_helpers import assert_paths_covered
from tests.test_backend.test_metal.test_template_wrapper_deduction import CASES, source
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_TEMPLATE_WRAPPER_RUNTIME"


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("case", ("dependent-plain", "non-type-plain"))
def test_project_retains_struct_only_materialization(tmp_path, target, scoped, case):
    (tmp_path / "wrapper.metal").write_text(source(case), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("wrapper.metal",),
            targets=(target,),
            entry_points={"wrapper.metal": ("copy_wrapped",)} if scoped else {},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert not report["diagnostics"]
    (artifact,) = report["artifacts"]
    assert artifact["status"] == "translated"
    assert artifact["templateMaterialization"]["status"] == "materialized"
    generated = (tmp_path / artifact["path"]).read_text()
    assert "struct Box_int_" in generated
    assert "Box<" not in generated


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "argument", ("Box<float>", "Box<int, float>", "Box<int, void, int>")
)
def test_project_rejects_incompatible_wrapper_arguments(tmp_path, target, argument):
    (tmp_path / "wrapper.metal").write_text(
        source(argument_type=argument), encoding="utf-8"
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("wrapper.metal",),
            targets=(target,),
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.metal-struct-method"
        for item in report["diagnostics"]
    )
    for artifact in report["artifacts"]:
        assert not (tmp_path / artifact["path"]).exists()


def _request(root, target, case):
    original, descriptor, package = _package(
        root, target, "int", (1, 1, 1), source=source(case), software_subgroups=False
    )
    values = [-7, 2147483647, -2147483648]
    guard = 123456789
    inputs = {
        "values": {"dtype": "int32", "shape": [3], "values": values},
        "results": {"dtype": "int32", "shape": [7, 1], "values": [guard] * 7},
    }
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "int32",
                "shape": [7, 1],
                "values": [guard] * 2 + values + [guard] * 2,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("case", CASES)
def test_template_wrapper_compiles(tmp_path, target, case):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _request(tmp_path, target, case)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_template_wrapper_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required template-wrapper execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="copy_wrapped",
    )


def test_template_wrapper_execution_is_required_on_each_native_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    for name in (
        "Validate general gather and empty arrays",
        "Validate indexed DirectX gather and resource aggregates",
        "Validate indexed OpenGL gather and resource aggregates",
    ):
        step = ci_coverage.workflow_step_section(workflow, name)
        assert f'{REQUIRE_ENV}: "1"' in step
        assert "test_template_wrapper_runtime.py" in step
        timeout = 1800 if name == "Validate general gather and empty arrays" else 1200
        assert f"--timeout-seconds {timeout}" in step
        assert "if:" not in step and "continue-on-error" not in workflow
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_backend/test_metal/**",
        )
