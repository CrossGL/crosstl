"""Concrete functors preserve member writes through specialized helper calls."""

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_backend.test_metal.test_functor_member_forwarding import CASES, source
from tests.test_backend.test_metal.test_unused_call_operators import SOURCE as OUTLINED
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_FUNCTOR_MEMBER_RUNTIME"


def _outlined_source(selected):
    owner = "Offset" if selected else "Root"
    return (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + OUTLINED.split("kernel void", 1)[0]
        + f"""kernel void calculate(
        const device uint* values [[buffer(0)]],
        device uint* results [[buffer(1)]],
        uint i [[thread_position_in_grid]]) {{
    float input = as_type<float>(values[i]);
    results[4u + 2u * i] = as_type<uint>({owner}{{}}(input));
    results[5u + 2u * i] = values[i];
}}
"""
    )


def _outlined_project(root, target, selected, profile):
    original = _outlined_source(selected)
    (root / "outlined.metal").write_text(original, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("outlined.metal",),
            targets=(target,),
            entry_points={"outlined.metal": ("calculate",)},
            source_options=(
                {"metal": {"binary32_additive_profile": profile}} if profile else {}
            ),
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert not report["diagnostics"]
    (artifact,) = report["artifacts"]
    assert artifact["status"] == "translated"
    return original, (root / artifact["path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("selected", (False, True))
@pytest.mark.parametrize("profile", (None, "rne-flush"))
def test_outlined_functor_project_compiles(tmp_path, target, selected, profile):
    original, generated = _outlined_project(tmp_path, target, selected, profile)
    assert "float Offset::operator()" in original
    assert ("Offset_operator_call" in generated.replace("__", "_")) == selected
    if selected and profile:
        assert "crossgl_metal_add_float" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("selected", (False, True))
@pytest.mark.parametrize("profile", (None, "rne-flush"))
def test_outlined_functor_executes_natively(tmp_path, selected, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required functor-member execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, generated = _outlined_project(tmp_path, target, selected, profile)
    values = [float(i) / 8.0 for i in range(-64, 65)]

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    inputs = [(bits(value),) for value in values]
    expected = [0x1937A5C3] * (8 + 2 * len(values))
    factor = 3.0 if selected else 2.0
    for i, value in enumerate(values):
        expected[4 + 2 * i : 6 + 2 * i] = [bits(value * factor), bits(value)]
    initial = [0x1937A5C3] * len(expected)
    execution = tmp_path / "native"
    execution.mkdir()
    actual, evidence = _dispatch(
        execution,
        target,
        generated,
        inputs,
        len(expected),
        initial_output=initial,
        entry="calculate" if target == "metal" else None,
    )
    assert actual == expected, evidence
    (execution / "audit.json").write_text(json.dumps(evidence), encoding="utf-8")
    if target == "metal":
        control = tmp_path / "original"
        control.mkdir()
        baseline, evidence = _dispatch(
            control,
            target,
            original,
            inputs,
            len(expected),
            entry="calculate",
            initial_output=initial,
        )
        assert baseline == expected, evidence


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("scoped", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_project_materializes_forwarded_functor_members(tmp_path, target, scoped, case):
    (tmp_path / "forward.metal").write_text(source(case), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("forward.metal",),
            targets=(target,),
            entry_points={"forward.metal": ("forward",)} if scoped else {},
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
    assert "op.apply(" not in generated
    if case in {"both", "stateful", "effects"}:
        assert "operator_call" in generated


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("failure", ("dependent", "ambiguous"))
def test_project_does_not_publish_unresolved_functor_members(tmp_path, target, failure):
    text = source()
    if failure == "dependent":
        text = text.replace("invoke<int, Update<int>>", "invoke<int>")
    else:
        text = text.replace(
            "void apply(device T* results, T value, uint index)",
            "void apply(device T* results, short value, uint index)",
        ).replace(
            "};",
            "void apply(device T* results, ushort value, uint index) { results[index] -= value; }\n};",
            1,
        )
    (tmp_path / "forward.metal").write_text(text, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("forward.metal",),
            targets=(target,),
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    assert any(item["severity"] == "error" for item in report["diagnostics"])
    for artifact in report["artifacts"]:
        assert not (tmp_path / artifact["path"]).exists()


def _request(root, target, case):
    original, descriptor, package = _package(
        root, target, "int", (1, 1, 1), source=source(case), software_subgroups=False
    )
    values = [-20000000, -7, 0, 13, 20000000]
    initial = [11, 22, 33, 44, 55]
    adjustment = {"member": 0, "transitive": 0, "both": 3, "stateful": 13}
    results = [
        base + (2 * value + 4 if case == "effects" else value + adjustment[case])
        for base, value in zip(initial, values)
    ]
    guard = 123456789
    inputs = {
        "values": {"dtype": "int32", "shape": [5], "values": values},
        "results": {
            "dtype": "int32",
            "shape": [8],
            "values": [guard, *initial, guard, guard],
        },
    }
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "int32",
                "shape": [8],
                "values": [guard, *results, guard, guard],
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
def test_forwarded_functor_members_compile(tmp_path, target, case):
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if not shutil.which(tool):
        pytest.skip(f"{tool} is not installed")
    _, request, _ = _request(tmp_path, target, case)
    _, module = _compile(request.artifact_path.read_text(), target, tmp_path)
    assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_forwarded_functor_members_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required functor-member execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case)
    _execute(
        request, expected, tmp_path, original_source=original, original_entry="forward"
    )


def test_functor_member_execution_is_required_on_each_native_target():
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
        assert "test_functor_member_runtime.py" in step
        timeout = 1800 if name == "Validate general gather and empty arrays" else 1200
        assert f"--timeout-seconds {timeout}" in step and "-n auto" in step
        assert "if:" not in step and "continue-on-error" not in workflow
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/**",
        )
