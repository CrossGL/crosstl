"""Source overload selection survives private resource-handle lowering."""

import os
import pickle
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode, FunctionNode, IdentifierNode
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    lower_resource_aggregates,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RESOURCE_OVERLOADS"
TARGETS = ("metal", "directx", "opengl")
STRICT_METAL = ("-std=metal3.2", "-Wall", "-Wextra", "-fno-fast-math")
GUARD = 0x5A1B2C3D
INPUTS = [0, 11, 0xFFFFFFFF, 0x80000000, 0x89ABCDEF]
SCALAR = """
uint select_value(uint value) { return value + 7u; }
uint select_value(float value) { return uint(value) + 9u; }
"""
POINTER = """
uint select_value(const device uint* value, uint index) { return value[index] + 7u; }
uint select_value(device uint* value, uint index) { return value[index] + 19u; }
"""
CASES = {
    "exact": (
        SCALAR,
        "uint value = select_value(c.input[tid]) + select_value(1.0f);",
        17,
        0,
    ),
    "arithmetic": (SCALAR, "uint value = select_value((word + 3u) ^ 0u);", 10, 0),
    "constructor": (SCALAR, "uint value = word + select_value(float(1u));", 10, 0),
    "nested": (SCALAR, "uint value = select_value(select_value(word));", 14, 0),
    "conditional": (
        SCALAR,
        "uint value = select_value(tid == 0u ? word : word + 1u) - uint(tid != 0u);",
        7,
        0,
    ),
    "aggregate": (
        "uint select_value(Cursor c, uint index) { return c.input[index] + 7u; }"
        "uint select_value(uint value, uint index) { return value + index + 19u; }",
        "uint value = select_value(c, tid);",
        7,
        0,
    ),
    "vector": (
        "uint select_value(uint2 value) { return value.x + value.y + 7u; }"
        "uint select_value(float2 value) { return uint(value.x + value.y) + 19u; }",
        "uint2 pair = uint2(word, 3u); uint value = select_value(pair);",
        10,
        0,
    ),
    "swizzle": (
        SCALAR,
        "uint2 pair = uint2(word, 3u); uint value = select_value(pair.x);",
        7,
        0,
    ),
    "shadow": (
        SCALAR,
        "uint value = select_value(word); { float word = 1.0f; value += select_value(word); }",
        17,
        0,
    ),
    "for": (
        SCALAR,
        "uint value = select_value(word); for (float word = 1.0f; word < 2.0f; word += 1.0f) { value += select_value(word); }",
        17,
        0,
    ),
    "while": (
        SCALAR,
        "uint value = select_value(word); while (count < 1u) { float word = 1.0f; value += select_value(word); count++; }",
        17,
        1,
    ),
    "do": (
        SCALAR,
        "uint value = select_value(word); do { float word = 1.0f; value += select_value(word); count++; } while (count < 1u);",
        17,
        1,
    ),
    "switch": (
        SCALAR,
        "uint value = select_value(word); switch (tid) { case 0u: { float word = 1.0f; value += select_value(word); break; } default: { uint word = 3u; value += select_value(word); break; } }",
        17,
        0,
    ),
    "readonly": (POINTER, "uint value = select_value(c.input, tid);", 7, 0),
    "writable": (POINTER, "uint value = select_value(c.input, tid);", 19, 0),
    "constant": (
        "uint select_value(const constant uint* value, uint index) { return value[index] + 7u; }"
        "uint select_value(const device uint* value, uint index) { return value[index] + 19u; }",
        "uint value = select_value(c.input, tid);",
        7,
        0,
    ),
    "effects": (
        SCALAR,
        "uint value = select_value(word++); count = word - c.input[tid];",
        7,
        1,
    ),
    "boolean-promotion": (
        "uint select_value(int x) { return uint(x) + 7u; }"
        "uint select_value(float x) { return uint(x) + 19u; }",
        "uint value = word + select_value(true);",
        8,
        0,
    ),
    "signed": (
        "uint select_value(int x) { return uint(x) + 7u; }"
        "uint select_value(uint x) { return x + 19u; }",
        "uint value = word + select_value(-1);",
        6,
        0,
    ),
    "forwarded": (
        SCALAR + "uint relay(uint x) { return select_value(x); }",
        "uint value = relay(word) + select_value(1.0f);",
        17,
        0,
    ),
    "vector-index": (
        SCALAR,
        "uint2 pair = uint2(word, 3u); uint value = select_value(pair[0]);",
        7,
        0,
    ),
    "mutation": (
        "uint select_value(thread uint& x) { x += 7u; return x; }"
        "uint select_value(float x) { return uint(x) + 19u; }",
        "uint value = select_value(word); count = word - c.input[tid];",
        7,
        7,
    ),
    "array": (
        SCALAR,
        "uint pair[2] = {word, 3u}; uint value = select_value(pair[0]);",
        7,
        0,
    ),
}


def _source(case):
    declarations, body, _, _ = CASES[case]
    space = (
        "device"
        if case == "writable"
        else "const constant" if case == "constant" else "const device"
    )
    return f"""#include <metal_stdlib>
using namespace metal;
struct Cursor {{ {space} uint* input; }};
{declarations}
kernel void resources({space} uint* inputs [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {{
    Cursor c;
    c.input = inputs;
    uint word = c.input[tid];
    (void)word;
    uint count = 0u;
    {body}
    results[4 + 2 * tid] = value;
    results[5 + 2 * tid] = count;
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
        index_range_assertions=(
            (
                {
                    "source": "products.metal",
                    "expression": "reference.offset + index",
                    "minimum": 0,
                    "maximum": len(INPUTS) - 1,
                },
            )
            if target == "opengl"
            else ()
        ),
    )
    _, _, bias, count = CASES[case]
    output = [GUARD] * 4
    for word in INPUTS:
        output.extend([(word + bias) & 0xFFFFFFFF, count])
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
    outputs = {
        "results": {"dtype": "uint32", "shape": [len(output)], "values": output},
    }
    if case == "writable":
        outputs["inputs"] = {
            "dtype": "uint32",
            "shape": [len(INPUTS)],
            "values": INPUTS,
        }
    expected = _bound_values(descriptor, outputs)
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


def _validate(artifact, work, target):
    return _compile(
        artifact.read_text(), target, work, metal_compile_flags=STRICT_METAL
    )[1]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_resource_overload_project_compile(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _validate(request.artifact_path, tmp_path, target)


@pytest.mark.parametrize("case", CASES)
def test_resource_overload_binds_before_erasure(tmp_path, case):
    source = tmp_path / "source.metal"
    source.write_text(_source(case))
    ast = parse(translate(str(source), backend="crossgl", format_output=False))
    before = pickle.dumps(ast)
    lowered = lower_resource_aggregates(ast)
    assert pickle.dumps(ast) == before
    functions = {node.name for node in lowered.walk() if isinstance(node, FunctionNode)}
    calls = [
        node.function.name
        for node in lowered.walk()
        if isinstance(node, FunctionCallNode)
        and isinstance(node.function, IdentifierNode)
        and node.function.name.startswith("select_value")
    ]
    assert calls and all(
        name.startswith("select_value_resource_overload") for name in calls
    )
    assert set(calls) <= functions
    assert pickle.dumps(lower_resource_aggregates(ast)) == pickle.dumps(lowered)


@pytest.mark.parametrize("case", CASES)
def test_resource_overload_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required resource overload execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="resources",
        metal_compile_flags=STRICT_METAL,
        validate=_validate,
    )


@pytest.mark.parametrize(
    "declarations, call",
    [
        (
            "uint pick(int x) { return uint(x); } uint pick(float x) { return uint(x); }",
            "pick(1u)",
        ),
        (
            "uint pick(int x, uint y) { return uint(x) + y; } uint pick(uint x, int y) { return x + uint(y); }",
            "pick(1u, 1u)",
        ),
        (
            "uint pick(uint x) { return x; } uint pick(float x) { return uint(x); }",
            "pick(unknown())",
        ),
    ],
)
def test_unproven_overload_is_not_selected(declarations, call):
    ast = parse(f"""shader T {{
        struct Cursor {{ device uint* data; }}
        {declarations}
        compute {{ void main(RWStructuredBuffer<uint> output @buffer(0)) {{
            Cursor c = Cursor(output); c.data[0] = {call};
        }} }}
    }}""")
    before = pickle.dumps(ast)
    with pytest.raises(ResourceAggregateError, match="overloaded-resource-call"):
        lower_resource_aggregates(ast)
    assert pickle.dumps(ast) == before


def test_resource_overload_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_resource_overloads.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


@pytest.mark.parametrize("rows, cols", [(8, 8), (16, 8)])
def test_cooperative_matrix_overload_keeps_fragment_identity(rows, cols):
    matrix = f"CooperativeMatrix<float, {rows}, {cols}, subgroup, accumulator, unspecified, metal_thread_elements, 32, {rows * cols // 32}, metal_thread_elements_reference_view>"
    ast = parse(f"""shader T {{
        struct Cursor {{ device uint* data; }}
        uint pick(vec2 value) {{ return uint(value.x) + 7u; }}
        uint pick({matrix} value) {{ return 19u; }}
        compute {{ void main(RWStructuredBuffer<uint> output @buffer(0)) {{
            Cursor c = Cursor(output);
            {matrix} value;
            c.data[0] = pick(value);
        }} }}
    }}""")
    lowered = lower_resource_aggregates(ast)
    selected = [
        f for f in lowered.functions if f.name.startswith("pick_resource_overload")
    ]
    assert len(selected) == 1
    matrix_type = selected[0].parameters[0].param_type
    assert matrix_type.rows == rows and matrix_type.cols == cols
    assert matrix_type.fragment_provenance == "metal_thread_elements_reference_view"


def test_cooperative_matrix_provenance_is_not_inferred_from_vector_size():
    from crosstl.translator.ast import CooperativeMatrixType, PrimitiveType
    from crosstl.translator.codegen.resource_aggregates import _conversion_rank

    scalar = PrimitiveType("float")
    source = CooperativeMatrixType(
        scalar,
        8,
        8,
        fragment_layout="metal_thread_elements",
        subgroup_size=32,
        elements_per_lane=2,
        fragment_provenance="metal_thread_elements_reference_view",
    )
    target = CooperativeMatrixType(
        scalar,
        8,
        8,
        fragment_layout="dense",
        subgroup_size=32,
        elements_per_lane=2,
        fragment_provenance="source_fixture",
    )
    assert _conversion_rank(source, target) is None


def test_overload_matching_does_not_erase_floating_width_metadata():
    from crosstl.translator.ast import PrimitiveType
    from crosstl.translator.codegen.resource_aggregates import _conversion_rank

    narrow = PrimitiveType("float", size_bits=16)
    assert _conversion_rank(narrow, PrimitiveType("float", size_bits=16)) == 0
    assert _conversion_rank(PrimitiveType("float"), narrow) is None


@pytest.mark.parametrize("expression", ["small", "pair.x", "pair[0]"])
def test_narrow_source_promotion_selects_integer_overload(expression):
    ast = parse(f"""shader T {{
        struct Cursor {{ device uint* data; }}
        uint pick(int value) {{ return uint(value) + 7u; }}
        uint pick(float value) {{ return uint(value) + 19u; }}
        compute {{ void main(RWStructuredBuffer<uint> output @buffer(0)) {{
            Cursor c = Cursor(output);
            uint16 small = 1u;
            u16vec2 pair;
            c.data[0] = pick({expression});
        }} }}
    }}""")
    lowered = lower_resource_aggregates(ast)
    selected = [
        f for f in lowered.functions if f.name.startswith("pick_resource_overload")
    ]
    assert len(selected) == 1
    assert selected[0].parameters[0].param_type.name == "int"
