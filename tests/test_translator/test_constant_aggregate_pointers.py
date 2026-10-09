"""Constant aggregate pointers retain indexed storage and native buffer bindings."""

import json
import os
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_CONSTANT_AGGREGATE_POINTERS"
TARGETS = ("metal", "directx", "opengl")
GUARD = 0x5A1B2C3D
ROWS = [
    (99, 101),
    (0, 3),
    (11, 17),
    (0xFFFFFFFF, 2),
    (0x7FFFFFFF, 0x80000001),
    (31, 0),
    (701, 709),
]
COUNT = len(ROWS) - 2
BODIES = {
    "arrow": "uint value = params->bias + params->scale;",
    "index": "uint value = params[i + 1].bias + params[i + 1].scale;",
    "offset": "uint value = (params + i + 1)->bias + (params + i + 1)->scale;",
    "dereference": "uint value = (*(params + i + 1)).bias + (*(params + i + 1)).scale;",
    "helper": "uint value = read_parameter(params + i + 1);",
    "alias": (
        "constant Params* row = params + i + 1; uint value = row->bias + row->scale;"
    ),
    "once": (
        "uint cursor = i + 1; uint value = (params + cursor++)->bias; value += cursor;"
    ),
    "constructor": "Snapshot snapshot(params + i + 1); uint value = snapshot.value;",
}


def _source(case, conditional=False):
    condition = ", function_constant(enabled)" if conditional else ""
    declaration = (
        "constant bool enabled [[function_constant(2)]];" if conditional else ""
    )
    body = BODIES[case]
    if conditional:
        body = f"uint value = 0; if (enabled) {{ {body} results[i + 4] = value; }} else {{ results[i + 4] = value; }}"
    else:
        body += "\n    results[i + 4] = value;"
    return f"""#include <metal_stdlib>
using namespace metal;
struct Params {{ uint bias; uint scale; }};
{declaration}
uint read_parameter(constant Params* p) {{ return p->bias + p->scale; }}
struct Snapshot {{
    uint value;
    Snapshot(constant Params* p) {{ this->value = p->bias + p->scale; }}
}};
kernel void aggregate_parameters(constant Params* params [[buffer(5){condition}]],
    device uint* results [[buffer(7)]], uint i [[thread_position_in_grid]]) {{
    {body}
}}
"""


def _project(root, target, case, enabled=None):
    source = _source(case, conditional=enabled is not None)
    path = root / "parameters.metal"
    path.write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=(path.name,),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
            specialization_constants=(
                {"enabled": enabled} if enabled is not None else {}
            ),
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    payload = report.to_json()
    assert not payload["diagnostics"], payload
    assert payload["summary"]["translatedCount"] == 1
    return source, root / payload["artifacts"][0]["path"]


def _request(root, target, case, enabled=None):
    source, _ = _project(root, target, case, enabled)
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    (root / "descriptor.json").write_text(json.dumps(descriptor, indent=2))
    (params,) = (
        binding
        for binding in descriptor["bindings"]
        if binding["coordinates"]["binding"] == 5
    )
    layout = params["scalarLayout"]
    assert params["access"] == "read"
    assert layout["runtimeSized"] is True
    assert layout["componentCount"] == 2
    assert layout["elementType"] == "uint32"
    assert layout["elementSizeBytes"] == layout["elementStrideBytes"] == 8
    assert [member["offsetBytes"] for member in layout["structMembers"]] == [0, 4]
    if enabled is False:
        values = [0] * COUNT
    elif case == "arrow":
        values = [sum(ROWS[0])] * COUNT
    elif case == "once":
        values = [ROWS[i + 1][0] + i + 2 for i in range(COUNT)]
    else:
        values = [sum(row) for row in ROWS[1:-1]]
    expected_words = (
        [GUARD] * 4 + [value & 0xFFFFFFFF for value in values] + [GUARD] * 4
    )
    inputs = _bound_values(
        descriptor,
        {
            "params": {
                "dtype": "uint32",
                "shape": [len(ROWS), 2],
                "values": [list(row) for row in ROWS],
            },
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * COUNT + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": expected_words,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [COUNT, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", BODIES)
def test_constant_aggregate_pointer_project_contract(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    assert "ConstantBuffer<Params>" not in generated
    if target == "metal":
        assert "constant Params* params [[buffer(5)]]" in generated
    elif target == "directx":
        assert "StructuredBuffer<Params> params : register(t5);" in generated
    else:
        assert "binding = 5) readonly buffer" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", ("arrow", "offset", "helper", "once"))
def test_constant_aggregate_pointer_saved_intermediate(tmp_path, target, case):
    original = tmp_path / "source.metal"
    original.write_text(_source(case))
    intermediate = tmp_path / "source.cgl"
    code = translate(str(original), backend="crossgl", format_output=False)
    assert "StructuredBuffer<Params> params @buffer(5) @constant" in code
    assert "ConstantBuffer<Params>" not in code
    intermediate.write_text(code)
    generated = translate(str(intermediate), backend=target, format_output=False)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("enabled", (False, True))
def test_conditional_constant_aggregate_pointer_binding(tmp_path, target, enabled):
    _, request, _ = _request(tmp_path, target, "offset", enabled)
    generated = request.artifact_path.read_text()
    if target == "metal":
        assert (
            "constant Params* params [[buffer(5)]] [[function_constant(enabled)]]"
            in generated
        )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("case", BODIES)
@pytest.mark.parametrize("enabled", (None, False, True))
def test_constant_aggregate_pointers_execute(tmp_path, case, enabled):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required aggregate pointer execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case, enabled)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="aggregate_parameters",
    )


def test_constant_aggregate_pointer_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_constant_aggregate_pointers.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
