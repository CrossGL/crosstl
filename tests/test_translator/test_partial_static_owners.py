"""Partial template static owners survive constructors and saved intermediates."""

import os
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_PARTIAL_STATIC_OWNERS"
TARGETS = ("metal", "directx", "opengl")
CASES = (
    "forward-alias",
    "forward-qualified",
    "defined-alias",
    "defined-qualified",
    "late-forward",
    "default-argument",
    "extent",
    "typedef",
    "alias-chain",
    "shadow",
    "initializer",
    "method",
    "namespace",
)
INPUTS = [0, 11, 0xFFFFFFFF, 0x80000000, 0x89ABCDEF]
GUARD = 0x5A1B2C3D


def _source(case):
    primary = " {}" if case.startswith("defined") or case == "late-forward" else ""
    defaults = (
        ", int Rows, int Cols = 8"
        if case == "default-argument"
        else ", int Rows, int Cols"
    )
    alias_target = (
        "Fragment<T, extent, extent>" if case == "extent" else "Fragment<T, 8, 8>"
    )
    if case == "default-argument":
        alias_target = "Fragment<T, 8>"
    alias = f"using fragment_t = {alias_target};"
    if case == "typedef":
        alias = f"typedef {alias_target} fragment_t;"
    if case == "alias-chain":
        alias += " using second_t = fragment_t;"
    owner = (
        alias_target
        if case.endswith("qualified")
        else "second_t" if case == "alias-chain" else "fragment_t"
    )
    constructor = f"Block(uint lane) thread {{ value = {owner}::get_coord(lane); }}"
    if case == "shadow":
        constructor = """Block(uint lane) thread {
            value = fragment_t::get_coord(lane);
            { using fragment_t = Alternate; value += fragment_t::get_coord(lane); }
            value += fragment_t::get_coord(lane);
        }"""
    if case == "initializer":
        constructor = f"Block(uint lane) thread : value({owner}::get_coord(lane)) {{}}"
    method = ""
    if case == "method":
        constructor = "Block(uint lane) thread { value = lane; }"
        method = "uint evaluate() const thread { return fragment_t::get_coord(value); }"
    declaration = f"""
template<typename T{defaults}> struct Fragment{primary};
template<typename T> struct Fragment<T, 8, 8> {{
    static constexpr uint get_coord(uint lane) {{ return lane + 7u; }}
}};
"""
    if case == "late-forward":
        declaration += "template<typename T, int Rows, int Cols> struct Fragment;"
    declaration += f"""
struct Alternate {{
    static constexpr uint get_coord(uint lane) {{ return lane + 101u; }}
}};
template<typename T> struct Block {{
    static constant constexpr int extent = 8;
    {alias}
    uint value;
    {constructor}
    {method}
}};
"""
    block = "Block<uint>"
    if case == "namespace":
        declaration = f"namespace Geometry {{ {declaration} }}"
        block = "Geometry::Block<uint>"
    result = "block.evaluate()" if case == "method" else "block.value"
    return f"""#include <metal_stdlib>
using namespace metal;
{declaration}
kernel void partial_owners(const device uint* inputs [[buffer(0)]],
                           device uint* results [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {{
    uint cursor = inputs[tid];
    {block} block(cursor++);
    results[4 + 2 * tid] = {result};
    results[5 + 2 * tid] = cursor;
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    output = [GUARD] * 4
    for word in INPUTS:
        value = 3 * word + 115 if case == "shadow" else word + 7
        output.extend([value & 0xFFFFFFFF, (word + 1) & 0xFFFFFFFF])
    output.extend([GUARD] * 4)
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {"dtype": "uint32", "shape": [len(INPUTS)], "values": INPUTS},
            "results": {
                "dtype": "uint32",
                "shape": [len(output)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * (2 * len(INPUTS)) + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {"dtype": "uint32", "shape": [len(output)], "values": output},
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [len(INPUTS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_partial_static_owners_project_compile(tmp_path, case, target):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    separator = "_" if target == "opengl" else "__"
    assert f"Fragment_uint_8_8{separator}get_coord(" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_partial_static_owners_saved_intermediate(tmp_path, case, target):
    original = tmp_path / "source.metal"
    original.write_text(_source(case))
    intermediate = tmp_path / "source.cgl"
    intermediate.write_text(
        translate(str(original), backend="crossgl", format_output=False)
    )
    generated = translate(str(intermediate), backend=target, format_output=False)
    separator = "_" if target == "opengl" else "__"
    assert f"Fragment_uint_8_8{separator}get_coord(" in generated
    _compile(generated, target, tmp_path)


def test_forward_primary_retains_signature_without_inventing_definition():
    source = "template<typename T, int Count = 8> struct Incomplete;"
    preprocessor = MetalPreprocessor()
    assert not preprocessor._find_template_structs(source)
    (declaration,) = preprocessor._find_template_structs(
        source,
        include_forward_declarations=True,
    )
    assert declaration.is_forward_declaration
    assert declaration.template_parameters == ["T", "Count"]
    assert declaration.template_parameter_defaults == {"Count": "8"}
    assert (
        preprocessor._materialize_template_struct_with_name(
            declaration, ["uint", "8"], "Incomplete_uint_8"
        )
        == ""
    )
    output = preprocessor._materialize_explicit_template_struct_instantiations(
        source + "\nIncomplete<uint, 8>* value;"
    )
    assert "Incomplete<uint, 8>* value;" in output
    assert "struct Incomplete_uint_8" not in output


def test_forward_primary_does_not_select_ambiguous_partial():
    source = """
        template<typename T, int R, int C> struct Fragment;
        template<typename T, int C> struct Fragment<T, 8, C> {
            static uint get_coord(uint lane) { return lane + 1u; }
        };
        template<typename T, int R> struct Fragment<T, R, 8> {
            static uint get_coord(uint lane) { return lane + 2u; }
        };
        uint probe(uint lane) { return Fragment<uint, 8, 8>::get_coord(lane); }
    """
    output = MetalPreprocessor()._materialize_explicit_template_struct_instantiations(
        source
    )
    assert "Fragment<uint, 8, 8>::get_coord(lane)" in output
    assert "struct Fragment_uint_8_8" not in output


def test_static_alias_resolution_ignores_comments_and_string_literals():
    source = """
        template<typename T, int N> struct Fragment;
        struct Owner {
            using fragment_t = Fragment<uint, 8>;
            uint evaluate(uint value) const {
                // fragment_t::get_coord(value)
                const char* text = "fragment_t::get_coord(value)";
                return value;
            }
        };
    """
    assert (
        MetalPreprocessor()._canonicalize_struct_static_alias_owners(source) == source
    )


@pytest.mark.parametrize("case", CASES)
def test_partial_static_owners_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required partial static owner execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="partial_owners",
    )


def test_partial_static_owner_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_partial_static_owners.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
