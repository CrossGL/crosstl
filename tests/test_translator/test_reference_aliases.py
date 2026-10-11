"""Shared source references retain sequential reads and writes on each target."""

import os
import pickle
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.ast import (
    BlockNode,
    ForInNode,
    IdentifierNode,
    IdentifierPatternNode,
    LambdaNode,
    MatchArmNode,
    ReturnNode,
)
from crosstl.translator.codegen.reference_aliases import (
    ReferenceAliasError,
    lower_reference_aliases,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_REFERENCE_ALIASES"
CASES = (
    "scalar",
    "readonly",
    "readonly-first",
    "array",
    "vector",
    "aggregate",
    "nested",
    "shadow",
    "distinct",
    "return",
    "by-value",
    "index-mutation",
    "name-collision",
)
INPUTS = [0.0, 1.0, -2.0, 16.0, 128.0]
GUARD = [98765.0] * 4


def _source(case):
    helper = "void update(thread float& a, thread float& b) { a += 1.0f; b += a; }"
    setup = "float first = inputs[tid]; float second = 7.0f;"
    call = "update(first, first);"
    outputs = "first, second"
    if case == "readonly":
        helper = (
            "void update(thread float& a, const thread float& b) { a += 1.0f; a += b; }"
        )
    elif case == "readonly-first":
        helper = (
            "void update(const thread float& a, thread float& b) { b += 1.0f; b += a; }"
        )
    elif case in ("array", "vector"):
        setup = (
            "float values[2]; values[0] = inputs[tid]; values[1] = 7.0f;"
            if case == "array"
            else "float2 values = float2(inputs[tid], 7.0f);"
        )
        call = "uint index = 0u; update(values[index], values[index]);"
        outputs = "values[0], values[1]"
        if case == "vector":
            helper = "void update(thread float2& a, thread float2& b) { a[0] += 1.0f; b[0] += a[0]; }"
            call = "update(values, values);"
    elif case in ("aggregate", "nested"):
        setup = "Tile tile; tile.values[0] = inputs[tid]; tile.values[1] = 7.0f;"
        if case == "aggregate":
            helper = "void update(thread Tile& a, thread Tile& b) { a.values[0] += 1.0f; b.values[0] += a.values[0]; }"
            call = "update(tile, tile);"
        else:
            helper += """
void outer(thread Tile& destination, thread Tile& accumulator) {
    for (int i = 0; i < 2; ++i) {
        update(destination.values[i], accumulator.values[i]);
    }
}
"""
            call = "outer(tile, tile);"
        outputs = "tile.values[0], tile.values[1]"
    elif case == "shadow":
        helper = """
void update(thread float& a, thread float& b) {
    a += 1.0f;
    { float b = 3.0f; a += b; }
    { float a = 2.0f; b += a; }
    b += a;
}
"""
    elif case == "distinct":
        call = "update(first, second);"
    elif case == "return":
        helper = "float update(thread float& a, thread float& b) { a += 1.0f; b += a; return a; }"
        call = "second = update(first, first);"
    elif case == "by-value":
        helper = "void update(thread float& a, float b) { a += 1.0f; a += b; }"
    elif case == "index-mutation":
        helper = "void update(thread float& a, thread float& b, thread uint& index) { index = 1u; a += 1.0f; b += a; }"
        setup = "float values[2]; values[0] = inputs[tid]; values[1] = 7.0f; uint index = 0u;"
        call = "update(values[index], values[index], index);"
        outputs = "values[0], values[1]"
    elif case == "name-collision":
        helper += "\nfloat update_shared_references(float x) { return x; }"
        setup += " float crosstl_reference_0 = 0.0f;"
        call += " second += update_shared_references(crosstl_reference_0);"
    first, second = outputs.split(", ")
    return f"""#include <metal_stdlib>
using namespace metal;
struct Tile {{ float values[2]; }};
{helper}
kernel void reference_aliases(const device float* inputs [[buffer(0)]],
                              device float* results [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {{
    if (tid >= 5u) {{ return; }}
    {setup}
    {call}
    results[4u + 2u * tid] = {first};
    results[5u + 2u * tid] = {second};
}}
"""


def _expected(case, value):
    first, second = 2 * (value + 1), 7.0
    if case == "nested":
        second = 16.0
    elif case == "shadow":
        first = 2 * (value + 6)
    elif case == "distinct":
        first, second = value + 1, value + 8
    elif case == "return":
        second = first
    elif case == "by-value":
        first = 2 * value + 1
    return [first, second]


def _request(root, target, case):
    source, descriptor, package = _package(
        root, target, "float", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    expected = (
        GUARD + [item for value in INPUTS for item in _expected(case, value)] + GUARD
    )
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {"dtype": "float32", "shape": [len(INPUTS)], "values": INPUTS},
            "results": {
                "dtype": "float32",
                "shape": [len(expected)],
                "values": GUARD + [-12345.0] * 10 + GUARD,
            },
        },
    )
    outputs = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "float32",
                "shape": [len(expected)],
                "values": expected,
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [6, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_reference_aliases_project_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_reference_aliases_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native shared references")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="reference_aliases",
    )


def _canonical(tmp_path, case="nested"):
    source = tmp_path / "source.metal"
    source.write_text(_source(case))
    return translate(str(source), backend="crossgl", format_output=False)


def test_reference_aliases_preserve_source_ast_and_reuse(tmp_path):
    ast = parse(_canonical(tmp_path))
    before = pickle.dumps(ast)
    first = lower_reference_aliases(ast)
    second = lower_reference_aliases(ast)
    assert pickle.dumps(ast) == before
    assert pickle.dumps(first) == pickle.dumps(second)
    assert len([fn for fn in first.functions if "_shared_references" in fn.name]) == 2


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_reference_aliases_saved_intermediate(tmp_path, target):
    intermediate = tmp_path / "source.cgl"
    intermediate.write_text(_canonical(tmp_path))
    generated = translate(str(intermediate), backend=target, format_output=False)
    assert "_shared_references" in generated
    _compile(generated, target, tmp_path)


def test_reference_aliases_leave_value_result_parameters_unchanged(tmp_path):
    ast = parse(_canonical(tmp_path, "scalar").replace("inout thread", "inout"))
    generated = lower_reference_aliases(ast)
    assert not any("_shared_references" in fn.name for fn in generated.functions)


@pytest.mark.parametrize("kind", ("range", "match"))
def test_reference_aliases_preserve_pattern_binding_scope(tmp_path, kind):
    ast = parse(_canonical(tmp_path, "scalar"))
    helper = next(fn for fn in ast.functions if fn.name == "update")
    body = BlockNode([ReturnNode(IdentifierNode("b"))])
    if kind == "range":
        scoped = ForInNode("b", IdentifierNode("a"), body)
    else:
        scoped = MatchArmNode(IdentifierPatternNode("b"), IdentifierNode("a"), body)
    helper.body.statements = [scoped]
    result = lower_reference_aliases(ast)
    specialized = next(fn for fn in result.functions if "_shared_references" in fn.name)
    scope = specialized.body.statements[0]
    assert scope.body.statements[0].value.name == "b"
    observed = scope.iterable if kind == "range" else scope.guard
    assert observed.name == specialized.parameters[0].name


def test_reference_aliases_reject_unproven_closure_captures(tmp_path):
    ast = parse(_canonical(tmp_path, "scalar"))
    helper = next(fn for fn in ast.functions if fn.name == "update")
    helper.body.statements = [LambdaNode([], IdentifierNode("a"), captures=["a", "b"])]
    with pytest.raises(
        ReferenceAliasError, match="nested-callable-capture-unsupported"
    ):
        lower_reference_aliases(ast)


def test_reference_aliases_reject_side_effecting_sibling_argument(tmp_path):
    text = (
        _canonical(tmp_path, "scalar")
        .replace("inout thread float b)", "inout thread float b, float extra)")
        .replace("update(first, first)", "update(first, first, second++)")
    )
    with pytest.raises(ReferenceAliasError, match="argument-evaluation-unsupported"):
        lower_reference_aliases(parse(text))


def test_reference_aliases_required_on_native_platforms():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_reference_aliases.py" in step
    assert "pytest -q -n auto" in step
    assert "if:" not in step and "continue-on-error" not in step
