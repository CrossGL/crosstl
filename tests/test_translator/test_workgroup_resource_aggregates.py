"""Private aggregates retain shared allocation identity and element offsets."""

import os
import pickle
import struct
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import translate
from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.ast import (
    ArrayType,
    IdentifierNode,
    LiteralNode,
    ParameterNode,
    PointerType,
    PrimitiveType,
    VariableNode,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.resource_aggregates import (
    ResourceAggregateError,
    lower_resource_aggregates,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_workgroup_record_transfers import GUARD, WORDS

REQUIRE_ENV = "CROSTL_REQUIRE_WORKGROUP_RESOURCE_AGGREGATES"
CASES = [
    (kind, mode)
    for kind in ("uint", "int", "float")
    for mode in (
        "direct",
        "helper",
        "nested",
        "copy",
        "rebase",
        "selection",
        "shadow",
        "shared-only",
    )
]


def _source(kind, mode):
    declarations = (
        f"struct Cursor {{ const device {kind}* input; threadgroup {kind}* output; }};"
    )
    setup = "Cursor cursor{inputs + base, left + 2};"
    operation = "cursor.output[lane] = cursor.input[lane];"
    if mode == "helper":
        declarations += f"""
void store_value(Cursor cursor, uint lane) {{ cursor.output[lane] = cursor.input[lane]; }}
{kind} read_value(const threadgroup {kind}* values, uint lane) {{ return values[lane]; }}
void relay(Cursor cursor, uint lane) {{ store_value(cursor, lane); cursor.output[lane] = read_value(cursor.output, lane); }}
"""
        operation = "relay(cursor, lane);"
    elif mode == "nested":
        declarations += "struct Envelope { Cursor cursor; };"
        setup = "Envelope envelope{{inputs + base, left + 2}};"
        operation = "envelope.cursor.output[lane] = envelope.cursor.input[lane];"
    elif mode == "copy":
        setup += "Cursor snapshot = cursor; cursor.output = right + 3;"
        operation = "snapshot.output[lane] = snapshot.input[lane]; cursor.output[lane] = cursor.input[lane];"
    elif mode == "rebase":
        setup = "Cursor cursor{inputs + base + 1, left + 1}; cursor.input -= 1; cursor.output += 1;"
    elif mode == "selection":
        setup += "if ((lane & 1u) != 0u) { cursor.output = right + 3; }"
    elif mode == "shadow":
        declarations += "struct Holder { Cursor cursor; };"
        setup += """
Holder holder{cursor};
{
    threadgroup KIND left[10];
    Cursor inner{inputs + base, left + 2};
    inner.output[lane] = inner.input[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    holder.cursor.output[lane] = inner.output[(lane + 1u) % 4u];
}
""".replace("KIND", kind)
        operation = ""
    elif mode == "shared-only":
        declarations = f"struct Cursor {{ threadgroup {kind}* output; }};"
        setup = "Cursor cursor{left + 2};"
        operation = "cursor.output[lane] = inputs[base + lane];"
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void shared_resources(const device {kind}* inputs [[buffer(0)]],
                             device uint* results [[buffer(1)]],
                             uint lane [[thread_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]]) {{
    threadgroup {kind} left[10];
    threadgroup {kind} right[10];
    uint base = 4u + 4u * group.x;
    for (uint i = lane; i < 10u; i += 4u) {{
        left[i] = as_type<{kind}>(0x5A1B2C3Du);
        right[i] = as_type<{kind}>(0x5A1B2C3Du);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {setup}
    {operation}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 10u; i += 4u) {{
        results[4u + 20u * group.x + i] = as_type<uint>(left[i]);
        results[14u + 20u * group.x + i] = as_type<uint>(right[i]);
    }}
}}
"""


def _request(root, target, case):
    kind, mode = case
    words = [WORDS[i % len(WORDS)] for i in range(24)]
    if kind == "float":
        words = [0x00800123 if word == 0x7FC00123 else word for word in words]
    values = [
        struct.unpack(
            "<" + {"uint": "I", "int": "i", "float": "f"}[kind], struct.pack("<I", word)
        )[0]
        for word in words
    ]
    expected_words = [GUARD] * 4
    for group in range(5):
        left, right = [GUARD] * 10, [GUARD] * 10
        for lane in range(4):
            value = words[
                4 + 4 * group + ((lane + 1) % 4 if mode == "shadow" else lane)
            ]
            if mode == "selection" and lane % 2:
                right[3 + lane] = value
            else:
                left[2 + lane] = value
            if mode == "copy":
                right[3 + lane] = value
        expected_words.extend(left + right)
    expected_words.extend([GUARD] * 4)
    source, descriptor, package = _package(
        root,
        target,
        kind,
        (4, 1, 1),
        source=_source(*case),
        software_subgroups=False,
        index_range_assertions=(
            [
                {
                    "source": "products.metal",
                    "function": f"crosstl_{storage}_{operation}_{element}{suffix}",
                    "expression": "reference.offset + index",
                    "minimum": 0,
                    "maximum": 9 if storage == "workgroup" else 107,
                }
                for storage in ("resource", "workgroup")
                for operation in ("load", "store")
                for element in dict.fromkeys((kind, "uint"))
                for suffix in ("", "_2")
            ]
            if target == "opengl"
            else ()
        ),
    )
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {
                "dtype": {"uint": "uint32", "int": "int32", "float": "float32"}[kind],
                "shape": [len(values)],
                "values": values,
            },
            "results": {
                "dtype": "uint32",
                "shape": [len(expected_words)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * 100 + [GUARD] * 4,
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
        {"workgroupCount": [5, 1, 1], "workgroupSize": [4, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_shared_resource_aggregate_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_shared_resource_aggregate_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required shared aggregate execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="shared_resources",
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


def test_shared_resource_aggregate_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_workgroup_resource_aggregates.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


def _ast(tmp_path):
    source = tmp_path / "shared.metal"
    source.write_text(_source("uint", "direct"))
    return parse(translate(str(source), backend="cgl", format_output=False))


@pytest.mark.parametrize(
    "mode", ["initializer", "volatile", "narrow", "vector", "nested", "extent", "zero"]
)
def test_shared_allocation_requires_exact_layout(tmp_path, mode):
    ast = _ast(tmp_path)
    left = next(
        node
        for node in ast.walk()
        if isinstance(node, VariableNode) and node.name == "left"
    )
    reason = "workgroup-allocation-layout"
    if mode == "initializer":
        left.initial_value = LiteralNode(0, "uint")
        reason = "workgroup-allocation-initializer-or-qualifier"
    elif mode == "volatile":
        left.qualifiers.append("volatile")
        reason = "workgroup-allocation-initializer-or-qualifier"
    elif mode == "narrow":
        left.var_type.element_type = PrimitiveType("uint", size_bits=16)
    elif mode == "vector":
        left.var_type.element_type = PrimitiveType("uint2")
    elif mode == "nested":
        left.var_type.element_type = ArrayType(
            PrimitiveType("uint"), LiteralNode(2, "int")
        )
    elif mode == "extent":
        left.var_type.size = IdentifierNode("unknown_size")
    else:
        left.var_type.size = LiteralNode(0, "int")
    with pytest.raises(ResourceAggregateError, match=reason):
        lower_resource_aggregates(ast)


def test_shared_allocation_keeps_source_ast_and_distinct_backings(tmp_path):
    source = tmp_path / "shadow.metal"
    source.write_text(_source("uint", "shadow"))
    ast = parse(translate(str(source), backend="cgl", format_output=False))
    before = pickle.dumps(ast)
    lowered = lower_resource_aggregates(ast)
    assert pickle.dumps(ast) == before
    roots = [
        node
        for node in lowered.global_variables
        if node.name.startswith("crosstl_workgroup_")
    ]
    assert len(roots) == 3
    assert len({node.name for node in roots}) == 3
    assert all(isinstance(node.var_type, ArrayType) for node in roots)


def test_shared_allocation_rejects_dynamic_entry_binding(tmp_path):
    ast = _ast(tmp_path)
    next(iter(ast.stages.values())).entry_point.parameters.append(
        ParameterNode(
            "dynamic", PointerType(PrimitiveType("uint"), address_space="threadgroup")
        )
    )
    with pytest.raises(ResourceAggregateError, match="dynamic-workgroup-binding"):
        lower_resource_aggregates(ast)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "mode,reason",
    [
        ("readonly", "write-through-readonly-pointer"),
        ("cast", "pointer-contract-mismatch"),
    ],
)
def test_shared_pointer_contracts_remain_diagnostic(tmp_path, target, mode, reason):
    source = _source("uint", "direct")
    if mode == "readonly":
        source = source.replace(
            "threadgroup uint* output", "const threadgroup uint* output"
        )
    else:
        source = source.replace(
            "cursor.output[lane] = cursor.input[lane];",
            "cursor.input = (const device uint*)cursor.output;",
        )
    path = tmp_path / "shared.metal"
    path.write_text(source)
    ast = parse(translate(str(path), backend="cgl", format_output=False))
    with pytest.raises(ResourceAggregateError, match=reason):
        lower_resource_aggregates(ast, storage_pointer_parameters=target == "opengl")


def test_shared_pointer_index_narrowing_requires_range_proof(tmp_path):
    (tmp_path / "shared.metal").write_text(_source("uint", "direct"))
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("shared.metal",),
            targets=("opengl",),
            workgroup_size=(4, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 1
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )
    assert not list((tmp_path / "out").rglob("*.glsl"))


def test_nested_single_member_constructor_uses_member_type(tmp_path):
    ast = parse("""shader Nested {
        struct Pair { uint left; uint right; }
        struct Holder { Pair value; }
        compute { void main(RWStructuredBuffer<uint> output @buffer(0)) {
            Holder item = Holder({17u, 29u});
            output[0] = item.value.left + item.value.right;
        } }
    }""")
    generated = GLSLCodeGen().generate(ast)
    assert "Holder(Pair(17u, 29u))" in generated
    _compile(generated, "opengl", tmp_path)
