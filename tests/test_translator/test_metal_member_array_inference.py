"""Concrete member-array deduction through project packages and native execution."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project import pipeline as project_pipeline
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_MEMBER_ARRAY_RUNTIME"
DECLARATIONS = {
    "declarator": "T values[3];",
    "standard": "array<T, 3> values;",
    "qualified": "metal::array<T, 3> values;",
    "alias": "using Storage = array<T, 3>; Storage values;",
}


def _source(declaration, member_call, kind):
    operation = "op.adjust" if member_call else "adjust"
    helper = "template <typename U> U adjust(U value) { return value * U(3) + U(1); }"
    helpers = f"struct Operation {{ {helper} }};" if member_call else helper
    local = "Operation op;" if member_call else ""
    return f"""#include <metal_stdlib>
using namespace metal;
template <typename T> struct Values {{ {DECLARATIONS[declaration]} }};
{helpers}
kernel void member_arrays(const device {kind}* src [[buffer(0)]],
                          device {kind}* results [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {{
    Values<{kind}> data;
    data.values[0] = src[tid];
    data.values[1] = src[tid] + {kind}(2);
    data.values[2] = src[tid] - {kind}(4);
    int index = 0;
    {local}
    results[4 + tid * 4] = {operation}(data.values[index++]);
    results[5 + tid * 4] = {operation}(data.values[index++]);
    results[6 + tid * 4] = {operation}(data.values[index++]);
    results[7 + tid * 4] = {kind}(index);
}}
"""


def _request(root, target, declaration, member_call, kind):
    source, descriptor, package = _package(
        root,
        target,
        kind,
        (1, 1, 1),
        source=_source(declaration, member_call, kind),
        software_subgroups=False,
    )
    values = [-7, 0, 19, 43] if kind == "int" else [-7.25, 0.5, 19.125, 43.75]
    dtype = "int32" if kind == "int" else "float32"
    expected = [-123456] * 4
    for value in values:
        expected.extend((value * 3 + 1, (value + 2) * 3 + 1, (value - 4) * 3 + 1, 3))
    expected.extend([-123456] * 4)
    inputs = {
        "src": {"dtype": dtype, "shape": [len(values)], "values": values},
        "results": {
            "dtype": dtype,
            "shape": [len(expected)],
            "values": [-123456] * len(expected),
        },
    }
    outputs = {
        "results": {"dtype": dtype, "shape": [len(expected)], "values": expected}
    }
    bound_outputs = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        bound_outputs,
        {"workgroupCount": [len(values), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, bound_outputs


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("declaration", DECLARATIONS)
@pytest.mark.parametrize("member_call", [False, True], ids=["free", "member"])
@pytest.mark.parametrize("kind", ["int", "float"])
def test_member_array_deduction_compiles(
    tmp_path, target, declaration, member_call, kind
):
    _, request, _ = _request(tmp_path, target, declaration, member_call, kind)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("declaration", DECLARATIONS)
@pytest.mark.parametrize("member_call", [False, True], ids=["free", "member"])
@pytest.mark.parametrize("kind", ["int", "float"])
def test_member_array_deduction_executes_natively(
    tmp_path, declaration, member_call, kind
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required member-array execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(
        tmp_path, target, declaration, member_call, kind
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="member_arrays",
    )


def test_member_array_native_gate_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate member array deduction"
    )
    assert (
        "test_metal_member_array_inference.py" in step and f'{REQUIRE_ENV}: "1"' in step
    )
    assert "--timeout-seconds" in step and "--junitxml" in step and "-n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
    for event in ("push", "pull_request"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metal_member_array_inference.py",
        )


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "member, initializer, expression",
    [
        ("const array<T, N> values;", "{{7}}", "refs.values[0]"),
        ("array<array<T, N>, N> values;", "{{{{7}}}}", "refs.values[0][0]"),
        ("T values[N][N];", "{{{7}}}", "refs.values[0][0]"),
        ("array<const device T*, N> values;", "{{src}}", "refs.values[0][tid]"),
        ("const device T* values[N];", "{{src}}", "refs.values[0][tid]"),
    ],
    ids=[
        "const",
        "nested-standard",
        "nested-declarator",
        "standard-pointer",
        "declarator-pointer",
    ],
)
def test_member_array_materialization_preserves_concrete_element_type(
    tmp_path, target, member, initializer, expression
):
    source = f"""#include <metal_stdlib>
using namespace metal;
template <typename T, int N> struct References {{ {member} }};
template <typename T> T echo_value(T value) {{ return value; }}
kernel void indexed_value(const device int* src [[buffer(0)]],
                          device int* results [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {{
    References<int, 1> refs{initializer};
    results[tid] = echo_value({expression});
}}
"""
    path = tmp_path / "source.metal"
    path.write_text(source)
    unit = project_pipeline.ProjectTranslationUnit(
        path=path,
        relative_path=path.name,
        source_backend="metal",
        extension=".metal",
        source_hash={"sha256": "test"},
        source_size_bytes=len(source.encode()),
    )
    result = project_pipeline._project_template_materialization_for_artifact(
        unit=unit,
        target=target,
        variant=None,
        defines={},
        define_sources={},
        include_paths=[],
        source_options={},
    )
    assert result is not None and not result.diagnostics
    calls = [
        item
        for item in result.metadata["specializations"]
        if item["name"] == "echo_value"
    ]
    assert len(calls) == 1 and calls[0]["parameters"] == {"T": "int"}
    assert f"echo_value_int({expression})" in result.text
    # Pointer-bearing aggregates still need resource lowering after deduction;
    # this assertion covers materialization, not native pointer-array support.
    _compile(source, "metal", tmp_path, metal_compile_flags=("-fno-fast-math",))
