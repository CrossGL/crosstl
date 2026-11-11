"""The DirectX generator scans each function body once per render, and rescans a function whose body is replaced."""

from collections import Counter

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen import hlsl_function_body_scans as scans_module
from crosstl.translator.codegen.ast_walk import walk_ast_values
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen


def _call(name, *arguments):
    return A.FunctionCallNode(name, list(arguments))


def _function(name, body, parameters=()):
    return A.FunctionNode(
        name,
        "void",
        [A.ParameterNode(parameter, "int") for parameter in parameters],
        body,
    )


def test_scan_records_calls_dispatch_mesh_arguments_and_declared_names_in_walk_order():
    dispatch_arguments = [A.LiteralNode(1, "int")] * 3
    body = [
        A.VariableNode("local", "int"),
        A.ExpressionStatementNode(_call("second", _call("first"))),
        A.ExpressionStatementNode(_call("DispatchMesh", *dispatch_arguments)),
    ]
    func = _function("f", body, parameters=("count",))
    scan = scans_module.scan_function_body(
        func, walk_ast_values, HLSLCodeGen().function_call_name
    )
    assert scan.call_names == ("second", "first", "DispatchMesh")
    assert len(scan.dispatch_mesh_args) == 1
    assert scan.dispatch_mesh_args[0] is body[2].expression.arguments
    assert scan.identifier_names == frozenset({"count", "local"})


@pytest.mark.parametrize("replace", ["body", "parameters"])
def test_a_function_whose_body_or_parameters_are_replaced_is_scanned_again(replace):
    generator = HLSLCodeGen()
    func = _function("f", [A.ExpressionStatementNode(_call("first"))])
    assert generator.hlsl_function_call_names(func) == ["first"]
    assert generator.hlsl_function_body_scan(func) is generator.hlsl_function_body_scan(
        func
    )
    if replace == "body":
        func.body = [A.ExpressionStatementNode(_call("second"))]
        assert generator.hlsl_function_call_names(func) == ["second"]
    else:
        func.parameters = [A.ParameterNode("renamed", "int")]
        assert generator.collect_hlsl_function_identifier_names(func) == {"renamed"}


def test_callers_receive_their_own_containers():
    generator = HLSLCodeGen()
    func = _function("f", [A.VariableNode("local", "int")], parameters=("count",))
    names = generator.collect_hlsl_function_identifier_names(func)
    names.add("caller_added")
    assert generator.collect_hlsl_function_identifier_names(func) == {"count", "local"}
    calls = generator.hlsl_function_call_names(func)
    calls.append("caller_added")
    assert generator.hlsl_function_call_names(func) == []


def _metal_kernels(count):
    kernels = "\n".join(f"""kernel void k{index}(device float* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) {{
    out[id] = outer(out[id]) + {index}.0f;
}}""" for index in range(count))
    return f"""#include <metal_stdlib>
using namespace metal;

float inner(float x) {{ return x * 2.0f; }}
float outer(float x) {{ float doubled = inner(x); return doubled + inner(doubled); }}

{kernels}
"""


def _scans_by_function(monkeypatch, tmp_path, count):
    scanned = Counter()
    scan = scans_module.scan_function_body

    def counting(func, walk, call_name):
        scanned[getattr(func, "name", None)] += 1
        return scan(func, walk, call_name)

    path = tmp_path / f"kernels{count}.metal"
    path.write_text(_metal_kernels(count))
    with monkeypatch.context() as patch:
        patch.setattr(scans_module, "scan_function_body", counting)
        code = crosstl.translate(str(path), backend="directx")
    return scanned, code


def test_shared_helper_scans_do_not_grow_with_the_number_of_entry_points(
    monkeypatch, tmp_path
):
    small, small_code = _scans_by_function(monkeypatch, tmp_path, 2)
    large, large_code = _scans_by_function(monkeypatch, tmp_path, 12)
    assert small_code.count("[numthreads(") == 2
    assert large_code.count("[numthreads(") == 12
    for helper in ("inner", "outer"):
        assert small[helper] >= 1
        assert large[helper] == small[helper]
