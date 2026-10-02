"""Pointer-member addresses retain source storage during overload binding."""

import os
import shutil
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_MEMBER_POINTER_METAL"


def _source(space, form):
    qualifier = f"const {space}"
    field = f"{qualifier} int* values;"
    declaration, expression = "Cursor cursor{base};", "cursor.values"
    if form == "alias":
        field = f"using Pointer = {qualifier} int*; Pointer values;"
    elif form == "alias-chain":
        field = f"using Pointer = {qualifier} int*; using Storage = Pointer; Storage values;"
    elif form == "nested":
        declaration, expression = "Envelope envelope{{base}};", "envelope.cursor.values"
    elif form == "array":
        field = f"array<{qualifier} int*, 1> values;"
        declaration, expression = "Cursor cursor{{base}};", "cursor.values[0]"
    elif form == "declarator":
        field = f"{qualifier} int* values[1];"
        declaration, expression = "Cursor cursor{{base}};", "cursor.values[0]"
    elif form == "const-owner":
        declaration = "const Cursor cursor{base};"
    aggregate = f"struct Cursor {{ {field} }};"
    if form == "nested":
        aggregate += "struct Envelope { Cursor cursor; };"
    buffer_space = "constant" if space == "constant" else "device"
    setup = f"const {space} int* base = src;"
    if space in {"thread", "threadgroup"}:
        setup = f"""{space} int local_values[8];
    for (uint i = 0; i < 8; ++i) {{ local_values[i] = src[i]; }}
    const {space} int* base = local_values;"""
    return f"""#include <metal_stdlib>
using namespace metal;
{aggregate}
int load({qualifier} int* values, int index) {{ return values[index] + 11; }}
int load({qualifier} int* values, int2 index) {{ return values[index.x] + 101; }}
kernel void member_pointer(const {buffer_space} int* src [[buffer(0)]],
                           device int* results [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {{
    {setup}
    {declaration}
    uint offset = tid + 1;
    auto pointer = &{expression}[offset++];
    results[4 + tid * 4] = load(pointer, 1);
    results[5 + tid * 4] = load(&{expression}[offset++], 2);
    results[6 + tid * 4] = load(pointer, int2(0, 0));
    results[7 + tid * 4] = int(offset);
}}
"""


@pytest.mark.parametrize("space", ["constant", "device", "thread", "threadgroup"])
@pytest.mark.parametrize(
    "form",
    ["direct", "alias", "alias-chain", "nested", "array", "declarator", "const-owner"],
)
def test_member_pointer_address_round_trip_compiles(tmp_path, space, form):
    source, _, package = _package(
        tmp_path,
        "metal",
        "int",
        (1, 1, 1),
        source=_source(space, form),
        software_subgroups=False,
    )
    (generated,) = package.rglob("*.metal")
    validation = tmp_path / "validation"
    validation.mkdir()
    _compile(generated.read_text(), "metal", validation)
    _compile(source, "metal", tmp_path, metal_compile_flags=("-fno-fast-math",))


@pytest.mark.parametrize("space", ["constant", "device", "thread", "threadgroup"])
@pytest.mark.parametrize(
    "form",
    ["direct", "alias", "alias-chain", "nested", "array", "declarator", "const-owner"],
)
def test_member_pointer_address_executes_natively(tmp_path, space, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required Metal pointer-member execution")
    source, descriptor, package = _package(
        tmp_path,
        "metal",
        "int",
        (1, 1, 1),
        source=_source(space, form),
        software_subgroups=False,
    )
    values = [-13, 4, 71, -19, 103, 53, 27, -6]
    expected = [-123456] * 4
    for tid in range(3):
        expected.extend(
            (values[tid + 2] + 11, values[tid + 4] + 11, values[tid + 1] + 101, tid + 3)
        )
    expected.extend([-123456] * 4)
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "src": {"dtype": "int32", "shape": [8], "values": values},
                "results": {
                    "dtype": "int32",
                    "shape": [len(expected)],
                    "values": [-123456] * len(expected),
                },
            },
        ),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="member_pointer",
    )


@pytest.mark.parametrize("space", ["device", "thread", "threadgroup"])
def test_const_pointer_member_keeps_writable_pointee(tmp_path, space):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required Metal pointer-member execution")
    setup = f"{space} int* base = src;"
    if space != "device":
        setup = f"""{space} int local_values[8];
    for (uint i = 0; i < 8; ++i) {{ local_values[i] = src[i]; }}
    {space} int* base = local_values;"""
    source = f"""#include <metal_stdlib>
using namespace metal;
struct Cursor {{ {space} int* const values; }};
int adjust({space} int* values, int index) {{ values[index] += 3; return values[index]; }}
int adjust({space} int* values, int2 index) {{ values[index.x] += 7; return values[index.x]; }}
kernel void member_pointer(device int* src [[buffer(0)]],
                           device int* results [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {{
    {setup}
    const Cursor cursor{{base}};
    auto pointer = &cursor.values[tid * 2 + 1];
    results[4 + tid * 4] = adjust(pointer, 0);
    results[5 + tid * 4] = adjust(&cursor.values[tid * 2 + 1], int2(1, 0));
    results[6 + tid * 4] = base[tid * 2 + 1];
    results[7 + tid * 4] = base[tid * 2 + 2];
}}
"""
    source, descriptor, package = _package(
        tmp_path, "metal", "int", (1, 1, 1), source=source, software_subgroups=False
    )
    values = [-13, 4, 71, -19, 103, 53, 27, -6]
    expected = [-123456] * 4
    for tid in range(3):
        expected.extend([values[tid * 2 + 1] + 3, values[tid * 2 + 2] + 7] * 2)
    expected.extend([-123456] * 4)
    expected_source = list(values)
    if space == "device":
        for tid in range(3):
            expected_source[tid * 2 + 1] += 3
            expected_source[tid * 2 + 2] += 7
    outputs = _bound_values(
        descriptor,
        {
            "results": {"dtype": "int32", "shape": [len(expected)], "values": expected},
            "src": {"dtype": "int32", "shape": [8], "values": expected_source},
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "src": {"dtype": "int32", "shape": [8], "values": values},
                "results": {
                    "dtype": "int32",
                    "shape": [len(expected)],
                    "values": [-123456] * len(expected),
                },
            },
        ),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="member_pointer",
    )


@pytest.mark.parametrize(
    "target, code",
    [
        ("opengl", "project.translate.opengl-storage-pointer-unsupported"),
    ],
)
def test_pointer_free_targets_retain_aggregate_storage_diagnostics(
    tmp_path, target, code
):
    (tmp_path / "source.metal").write_text(_source("constant", "direct"))
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("source.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(1, 1, 1),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert [item["code"] for item in report["diagnostics"]] == [code]


def test_directx_member_pointer_address_compiles(tmp_path):
    if not shutil.which("dxc"):
        pytest.skip("the optional DirectX compiler is unavailable")
    _, _, package = _package(
        tmp_path,
        "directx",
        "int",
        (1, 1, 1),
        source=_source("constant", "direct"),
        software_subgroups=False,
    )
    (generated,) = package.rglob("*.hlsl")
    validation = tmp_path / "validation"
    validation.mkdir()
    _, module = _compile(generated.read_text(), "directx", validation)
    assert module.is_file() and module.stat().st_size


def test_member_pointer_metal_gate_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/mlx-metal-host.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate pointer member round trips"
    )
    assert (
        "test_metal_member_pointer_provenance.py" in step
        and f'{REQUIRE_ENV}: "1"' in step
    )
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "cancel-in-progress: false" in workflow
    for event in ("push", "pull_request"):
        assert (
            "tests/test_translator/test_metal_member_pointer_provenance.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
