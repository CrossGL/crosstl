"""Concrete Metal vector constructors through native project packages."""

import os
import shutil
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.ci_helpers import assert_paths_covered
from tests.runtime_helpers import _validate
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_VECTOR_CONSTRUCTOR_RUNTIME"
KINDS = ("float", "half", "int")
FORMS = ("unqualified", "qualified", "alias", "conversion")
VALUES = (1.0007, -2.0009, 4095.5)


def _source(kind, width, form):
    vector = f"vec<{kind}, {width}>"
    constructor = f"metal::{vector}" if form == "qualified" else vector
    alias = ""
    if form == "alias":
        alias = f"using Value = {vector};"
        constructor = "Value"
    arguments = ", ".join(f"base + {kind}({lane})" for lane in range(width))
    helpers = (
        f"{vector} make_lanes({kind} base) {{ return {constructor}({arguments}); }}"
    )
    if form == "conversion":
        helpers = f"""struct Components {{
    {kind} base;
    operator {vector}() const {{ return {vector}({arguments}); }}
}};
{vector} make_lanes({kind} base) {{
    Components item{{base}};
    return static_cast<{vector}>(item);
}}"""
    mixed = [f"vec<{kind}, 2>(base, base + {kind}(1))"]
    mixed.extend(f"base + {kind}({lane})" for lane in range(2, width))
    return f"""#include <metal_stdlib>
using namespace metal;
{alias}
{helpers}
kernel void vector_constructors(const device float* src [[buffer(0)]],
                                device float* results [[buffer(1)]],
                                uint tid [[thread_position_in_grid]]) {{
    {kind} base = {kind}(src[tid]);
    uint cursor = tid;
    {vector} lanes = make_lanes(base);
    {vector} splat = {constructor}({kind}(src[cursor++]));
    {vector} copied = {constructor}(lanes);
    {vector} mixed = {constructor}({', '.join(mixed)});
    uint offset = 4 + tid * {width * 4 + 1};
    for (uint lane = 0; lane < {width}; ++lane) {{
        results[offset + lane] = float(lanes[lane]);
        results[offset + {width} + lane] = float(splat[lane]);
        results[offset + {width * 2} + lane] = float(copied[lane]);
        results[offset + {width * 3} + lane] = float(mixed[lane]);
    }}
    results[offset + {width * 4}] = float(cursor);
}}
"""


def _cast(kind, value):
    if kind == "int":
        return int(value)
    fmt = "<e" if kind == "half" else "<f"
    return struct.unpack(fmt, struct.pack(fmt, value))[0]


def _request(root, target, kind, width, form):
    source, descriptor, package = _package(
        root,
        target,
        "float",
        (1, 1, 1),
        source=_source(kind, width, form),
        software_subgroups=False,
    )
    values = [_cast("float", value) for value in VALUES]
    expected = [-123456.0] * 4
    for tid, value in enumerate(values):
        base = _cast(kind, value)
        lanes = [float(_cast(kind, base + lane)) for lane in range(width)]
        expected.extend(
            lanes + [float(base)] * width + lanes + lanes + [float(tid + 1)]
        )
    expected.extend([-123456.0] * 4)
    outputs = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "float32",
                "shape": [len(expected)],
                "values": expected,
            }
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "src": {"dtype": "float32", "shape": [len(values)], "values": values},
                "results": {
                    "dtype": "float32",
                    "shape": [len(expected)],
                    "values": [-123456.0] * len(expected),
                },
            },
        ),
        outputs,
        {"workgroupCount": [len(values), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("form", FORMS)
def test_generic_vector_constructors_compile(tmp_path, target, kind, width, form):
    _, request, _ = _request(tmp_path, target, kind, width, form)
    required_tools = {
        "metal": ("xcrun",),
        "directx": ("dxc",),
        "opengl": ("glslangValidator", "spirv-val"),
    }[target]
    missing = [tool for tool in required_tools if not shutil.which(tool)]
    native_target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}.get(
        sys.platform
    )
    if missing:
        if os.environ.get(REQUIRE_ENV) == "1" and target == native_target:
            pytest.fail(f"required native compiler unavailable: {', '.join(missing)}")
        pytest.skip(f"compiler unavailable: {', '.join(missing)}")
    _validate(request.artifact_path, tmp_path, target)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("form", FORMS)
def test_generic_vector_constructors_execute_natively(tmp_path, kind, width, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required vector-constructor execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, outputs = _request(tmp_path, target, kind, width, form)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="vector_constructors",
        validate=_validate,
    )


CONVERSION_CASES = (
    "named",
    "functional",
    "alias",
    "temporary",
    "mutating",
    "overloaded",
    "const-overloaded",
    "template",
    "parenthesized",
    "const-alias",
)


def _conversion_request(root, target, case):
    conversion = "operator float2() const { return float2(base + 1.0f, base + 7.0f); }"
    declaration = "Components item{base};"
    call = "static_cast<vec<float, 2>>(item)"
    offset = 1.0
    observed_base_offset = 0.0
    if case == "functional":
        call = "metal::vec<float, 2>(item)"
    elif case == "alias":
        call = "static_cast<Pair>(item)"
    elif case == "temporary":
        call = "static_cast<float2>(Components{base++})"
    elif case == "mutating":
        conversion = "operator float2() { base += 2.0f; return float2(base + 1.0f, base + 7.0f); }"
        offset = 3.0
        observed_base_offset = 2.0
    elif case in {"overloaded", "const-overloaded"}:
        conversion += " operator float2() { return float2(base + 3.0f, base + 9.0f); }"
        if case == "const-overloaded":
            declaration = "const Components item{base++};"
        else:
            offset = 3.0
    elif case == "parenthesized":
        call = "static_cast<float2>((item))"
    elif case == "const-alias":
        declaration = "using ReadOnly = const Components; ReadOnly item{base};"
    prefix = ""
    if case == "template":
        prefix = "template <typename T>"
        conversion = (
            "operator vec<T, 2>() const { return vec<T, 2>(base + T(1), base + T(7)); }"
        )
        declaration = "Components<float> item{base};"
    source = f"""#include <metal_stdlib>
using namespace metal;
using Pair = vec<float, 2>;
{prefix}
struct Components {{ float base; {conversion} }};
kernel void vector_conversions(const device float* src [[buffer(0)]],
                               device float* results [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {{
    float base = src[tid];
    {declaration}
    float2 converted = {call};
    uint offset = 4 + 4 * tid;
    results[offset] = converted.x;
    results[offset + 1] = converted.y;
    results[offset + 2] = base;
    results[offset + 3] = item.base;
}}
"""
    source, descriptor, package = _package(
        root, target, "float", (1, 1, 1), source=source, software_subgroups=False
    )
    values = [1.25, -2.5, 4095.5]
    expected = [-123456.0] * 4
    for value in values:
        expected.extend(
            [
                value + offset,
                value + offset + 6,
                value + (case in {"temporary", "const-overloaded"}),
                value + observed_base_offset,
            ]
        )
    expected.extend([-123456.0] * 4)
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "float32", "shape": [len(expected)], "values": expected}},
    )
    inputs = _bound_values(
        descriptor,
        {
            "src": {"dtype": "float32", "shape": [3], "values": values},
            "results": {
                "dtype": "float32",
                "shape": [len(expected)],
                "values": [-123456.0] * len(expected),
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", CONVERSION_CASES)
def test_selected_vector_conversion_executes_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required vector-conversion execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, outputs = _conversion_request(tmp_path, target, case)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="vector_conversions",
        validate=_validate,
    )


def test_generic_vector_constructor_gate_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate generic vector constructors"
    )
    assert "test_metal_vector_constructors.py" in step and f'{REQUIRE_ENV}: "1"' in step
    assert (
        "--timeout-seconds 300" in step and "--junitxml" in step and "-n auto" in step
    )
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metal_vector_constructors.py",
        )
