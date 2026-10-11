"""Discarded compile-time branches must not become runtime conditionals."""

import os
import shutil
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_backend.test_metal.test_codegen import (
    convert,
    convert_without_preprocessing,
)
from tests.test_translator.test_atomic_load_runtime import GUARD
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_CONSTEXPR_RUNTIME"
CASES = (
    "trait-int",
    "trait-float",
    "chain-true",
    "chain-false",
    "adjacent",
    "nested",
    "scope",
    "local-bool",
    "value-template",
    "runtime",
)
EXPECTED = {
    "trait-int": [-2, 1, 6],
    "trait-float": [-1, 2, 7],
    "chain-true": [11, 22, 22],
    "chain-false": [11, 44, 33],
    "adjacent": [12, 22, 22],
    "nested": [-3, 0, 5],
    "scope": [9, 9, 9],
    "local-bool": [4, 7, 12],
    "value-template": [5, 0, 5],
    "runtime": [8, 2, 8],
}


def _source(case):
    argument = "inputs[tid]"
    instantiation = "int"
    parameters = "typename T"
    if case.startswith("trait-"):
        instantiation = case[len("trait-") :]
        argument = f"{instantiation}(inputs[tid])"
        body = """if constexpr (metal::is_floating_point_v<T>) {
            return isnan(value) ? T(7) : value + T(2);
        } else {
            return (value | T(0)) + T(1);
        }"""
    elif case.startswith("chain-"):
        instantiation = "float" if case == "chain-true" else "int"
        argument = f"{instantiation}(inputs[tid])"
        body = """if (value < T(0)) { return T(11); }
        else if constexpr (metal::is_floating_point_v<T>) { return T(22); }
        else if (value > T(0)) { return T(33); }
        else { return T(44); }"""
    elif case == "adjacent":
        body = """T result = T(0);
        if constexpr (true) { result += T(2); }
        if (value < T(0)) { result += T(10); }
        else if constexpr (true) { result += T(20); }
        else { result += T(40); }
        return result;"""
    elif case == "nested":
        body = """if constexpr (false) { return value.missing(); }
        else if constexpr ((2 * 3 == 6) && !false) {
            if constexpr (true || false) { return value; }
            else { return value.missing(); }
        } else { return value.missing(); }"""
    elif case == "scope":
        body = """T result = T(0);
        if constexpr (true) { const T selected = T(4); result += selected; }
        if constexpr (false) { result += value.missing(); }
        else { const T selected = T(5); result += selected; }
        return result + value * T(0);"""
    elif case == "local-bool":
        body = """constexpr bool enabled = true;
        T result = value;
        if constexpr (enabled) {
            constexpr bool enabled = false;
            if constexpr (enabled) { result += value.missing(); }
            else { result += T(3); }
        }
        if constexpr (enabled) { result += T(4); }
        return result;"""
    elif case == "value-template":
        parameters = "typename T, int group_size"
        instantiation = "int, 16"
        body = """if constexpr (group_size == 16) { return value & T(7); }
        else { return value.missing(); }"""
    else:
        body = """T result = T(0);
        if (value != T(0)) { result = T(8); } else { result = T(2); }
        return result;"""
    return f"""#include <metal_stdlib>
using namespace metal;
template <{parameters}> T select_value(T value) {{ {body} }}
kernel void select_values(const device int* inputs [[buffer(0)]],
    device int* results [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{
    results[tid] = int(select_value<{instantiation}>({argument}));
}}
"""


def _request(root, target, case):
    original, descriptor, package = _package(
        root, target, "int", (1, 1, 1), source=_source(case), software_subgroups=False
    )
    inputs = {"dtype": "int32", "shape": [3], "values": [-3, 0, 5]}
    initial = {"dtype": "int32", "shape": [6], "values": [GUARD] * 6}
    result = {**initial, "values": [*EXPECTED[case], *([GUARD] * 3)]}
    expected = _bound_values(descriptor, {"results": result})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, {"inputs": inputs, "results": initial}),
        expected,
        {"workgroupCount": [3, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return original, request, expected


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("case", CASES)
def test_constexpr_branches_translate(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    assert "missing" not in generated and "if (false)" not in generated
    if case == "trait-int":
        assert "isnan" not in generated
    tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[target]
    if shutil.which(tool):
        _, module = _compile(generated, target, tmp_path)
        assert module.is_file() and module.stat().st_size


@pytest.mark.parametrize("case", CASES)
def test_constexpr_branches_execute_natively(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required compile-time branch execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=original,
        original_entry="select_values",
    )


def test_parser_keeps_adjacent_if_statements_and_branch_markers():
    from crosstl.backend.Metal.MetalLexer import MetalLexer
    from crosstl.backend.Metal.MetalParser import MetalParser

    source = """void run(bool active) {
        if constexpr (true) {}
        if (active) {} else if constexpr (false) {} else {}
        if constexpr (false) {} else if (active) {}
    }"""
    nodes = (
        MetalParser(MetalLexer(source, preprocess=False).tokenize())
        .parse()
        .functions[0]
        .body
    )
    assert len(nodes) == 3
    assert [node.if_constexpr for node in nodes] == [[True], [False], [True]]
    assert [node.else_if_constexpr for node in nodes] == [[], [True], [False]]


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_unresolved_constexpr_condition_reports_source_location(tmp_path, target):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void unresolved(device int* output [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if constexpr (tid != 0u) { output[tid] = 1; }
}
"""
    (tmp_path / "source.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=("source.metal",),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    diagnostics = [
        item for item in report["diagnostics"] if item["severity"] == "error"
    ]
    assert report["summary"]["translatedCount"] == 0
    assert len(diagnostics) == 1
    assert (
        diagnostics[0]["code"] == "project.translate.metal-constexpr-branch-unresolved"
    )
    assert diagnostics[0]["location"]["line"] == 4


def test_ordinary_false_branch_is_not_discarded():
    generated = convert("void run() { if (false) { retained(); } }")
    assert "if (false)" in generated and "retained();" in generated


def test_discarded_branch_is_not_lowered():
    generated = convert_without_preprocessing(
        "void run() { if constexpr (false) { sizeof(Unknown); } else { retained(); } }"
    )
    assert "Unknown" not in generated and "retained();" in generated


def test_instantiated_primary_is_retained_for_dependent_references():
    from crosstl.backend.Metal.preprocessor import MetalPreprocessor

    source = """template <typename T, int N> T choose(T value) { return value + T(N); }
    template <typename T> T deferred(T value) { return choose<T, 4>(value); }
    kernel void run(device int* result [[buffer(0)]]) { result[0] = choose<int, 2>(3); }
    """
    processed = MetalPreprocessor().preprocess(source)
    assert "choose_int_2(3)" in processed
    assert "T choose(T value)" in processed and "choose<T, 4>(value)" in processed
