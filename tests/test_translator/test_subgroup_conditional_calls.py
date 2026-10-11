"""Subgroup-conditional wrappers keep guarded memory and converged barriers."""

import json
import math
import os
import pickle
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.ast import (
    ArrayAccessNode,
    BlockNode,
    FunctionCallNode,
    IfNode,
    VariableNode,
)
from crosstl.translator.codegen.subgroup_control_flow import (
    converge_subgroup_conditional_calls,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_CONDITIONAL_CALLS"


def _source(body=None, helpers=None, condition="subgroup == 1u"):
    helpers = helpers or """float inner(float value) { return WavePrefixSum(value); }
    float wrapped(float value) { return inner(value); }"""
    body = body or f"""uint counter = 0u;
        if ({condition}) {{
            float result = wrapped(inputs[invocation + counter++]);
            results[invocation] = asuint(result);
        }}"""
    source = f"""shader ConditionalScan {{
        StructuredBuffer<float> inputs @register(t0);
        RWStructuredBuffer<uint> results @register(u1);
        {helpers}
        compute {{
            @numthreads(32, 4, 1)
            void main(uint invocation @gl_LocalInvocationIndex, uint subgroup @gl_SubgroupID) {{
                {body}
            }}
        }}
    }}"""
    return source.replace("subgroup", "simd_group_id")


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_conditional_scan_stages_inputs_and_stores_without_mutating_ast(
    tmp_path, target
):
    ast = parse(_source())
    generator = _codegen(target)
    entries = (
        generator.hlsl_software_subgroup_stage_entries(ast)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(ast)
    )
    before = pickle.dumps(ast)
    lowered = converge_subgroup_conditional_calls(
        ast, entries, generator.walk_ast, generator.map_operator
    )
    assert pickle.dumps(ast) == before
    assert lowered is not ast
    lowered_entries = (
        generator.hlsl_software_subgroup_stage_entries(lowered)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(lowered)
    )
    assert (
        converge_subgroup_conditional_calls(
            lowered, lowered_entries, generator.walk_ast, generator.map_operator
        )
        is lowered
    )
    entry = next(iter(lowered.stages.values())).entry_point
    block = next(node for node in entry.body.statements if isinstance(node, BlockNode))
    assert [type(node) for node in block.statements] == [
        VariableNode,
        VariableNode,
        IfNode,
        VariableNode,
        IfNode,
    ]
    reads = list(generator.walk_ast(block.statements[2]))
    assert any(isinstance(node, ArrayAccessNode) for node in reads)
    call = block.statements[3].initial_value
    assert isinstance(call, FunctionCallNode) and call.args is call.arguments
    assert not any(
        isinstance(node, ArrayAccessNode) for node in generator.walk_ast(call)
    )
    generated = generator.generate_stage(ast, "compute")
    assert generated.count("counter++") == 1
    assert "subgroup_enabled" in generated and "subgroup_input" in generated
    _compile(generated, target, tmp_path)
    assert generator.generate_stage(ast, "compute") == generated


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_unused_empty_receiver_is_forwarded_without_scalar_reference_effects(
    tmp_path, target
):
    helpers = """struct Operation {};
    float inner(inout Operation self, float value) { return WavePrefixSum(value); }
    float wrapped(inout Operation self, float value) { return inner(self, value); }"""
    body = """Operation op;
    if (subgroup == 0u) { float value = wrapped(op, inputs[invocation]); results[invocation] = asuint(value); }"""
    generated = _codegen(target).generate_stage(
        parse(_source(body, helpers)), "compute"
    )
    assert "subgroup_enabled" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "helpers",
    (
        "float wrapped(float value) { results[0] = 1u; return WavePrefixSum(value); }",
        "float wrapped(float value) { return WavePrefixSum(inputs[0]); }",
        "float wrapped(inout float value) { return WavePrefixSum(value); }",
        "float wrapped(float value) { return wrapped(value); }",
        "float wrapped(float value) { return WavePrefixSum(value); } float wrapped(int value) { return float(WavePrefixSum(value)); }",
        "float inner(float ignored, float value) { return WavePrefixSum(value); } float wrapped(float value) { return inner(inputs[0], value); }",
        "float inner(float ignored, float value) { return WavePrefixSum(value); } float wrapped(float value) { return inner(INFINITY, value); }",
        "float wrapped(float value) { return WavePrefixSum(1.0); }",
        "float wrapped(out float value) { return WavePrefixSum(1.0); }",
    ),
)
def test_effectful_or_unresolved_wrappers_stay_guarded(target, helpers):
    ast = parse(_source(helpers=helpers))
    generator = _codegen(target)
    entries = (
        generator.hlsl_software_subgroup_stage_entries(ast)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(ast)
    )
    assert (
        converge_subgroup_conditional_calls(
            ast, entries, generator.walk_ast, generator.map_operator
        )
        is ast
    )
    with pytest.raises(ValueError):
        generator.generate_stage(ast, "compute")


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "body",
    (
        "if (invocation < 32u) { float value = wrapped(inputs[invocation]); results[invocation] = asuint(value); }",
        "subgroup = invocation; if (subgroup == 0u) { float value = wrapped(inputs[invocation]); results[invocation] = asuint(value); }",
        "uint& alias = subgroup; alias = invocation; if (subgroup == 0u) { float value = wrapped(inputs[invocation]); results[invocation] = asuint(value); }",
        "{ uint subgroup = invocation; if (subgroup == 0u) { float value = wrapped(inputs[invocation]); results[invocation] = asuint(value); } }",
        "if (subgroup == 0u) { threadgroup float value = wrapped(inputs[invocation]); results[invocation] = asuint(value); }",
        "if (subgroup == 0u) { float value = wrapped(inputs[invocation]); results[invocation] = asuint(value); } else { results[invocation] = 0u; }",
        "if (subgroup == 0u) { float value = wrapped(WavePrefixSum(inputs[invocation])); results[invocation] = asuint(value); }",
    ),
)
def test_unproven_predicates_and_shared_results_are_not_lifted(target, body):
    ast = parse(_source(body))
    generator = _codegen(target)
    entries = (
        generator.hlsl_software_subgroup_stage_entries(ast)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(ast)
    )
    assert (
        converge_subgroup_conditional_calls(
            ast, entries, generator.walk_ast, generator.map_operator
        )
        is ast
    )
    with pytest.raises(ValueError):
        generator.generate_stage(ast, "compute")


def _native_source(size, selected):
    return f"""#include <metal_stdlib>
using namespace metal;
struct Operation {{
    float scan(float value) {{ return simd_prefix_exclusive_sum(value); }}
}};
float wrapped(thread Operation& op, float value, float ignored) {{ return op.scan(value); }}
kernel void conditional_scan(device float* inputWords [[buffer(0)]],
                             device uint* outputWords [[buffer(1)]],
                             uint invocation [[thread_index_in_threadgroup]],
                             uint simd_group_id [[simdgroup_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]]) {{
    Operation op;
    uint counter = 0u;
    uint ignoredCounter = 0u;
    uint outputIndex = (group.x * {size}u + invocation) * 4u;
    for (uint round = 0u; round < 2u; ++round) {{
        if (simd_group_id == {selected}u) {{
            float result = wrapped(op, inputWords[group.x * 32u + invocation - {selected * 32}u] + float(counter++), float(ignoredCounter++));
            outputWords[outputIndex + round] = as_type<uint>(result);
        }}
    }}
    outputWords[outputIndex + 2u] = counter;
    outputWords[outputIndex + 3u] = ignoredCounter;
}}
"""


@pytest.mark.parametrize("shape", ((7, 5, 1), (32, 4, 1)))
@pytest.mark.parametrize("selected", (0, 1, 3))
def test_conditional_calls_preserve_guarded_effects_on_device(
    tmp_path, shape, selected
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for conditional collective execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_native_source(size, selected)
    )
    values = [(index % 13 - 6) * 0.125 for index in range(96)]
    marker = 0xBAD01234
    expected = [marker] * (3 * size * 4)
    for group in range(3):
        for invocation in range(size):
            offset = (group * size + invocation) * 4
            active = invocation // 32 == selected
            expected[offset + 2] = 2 if active else 0
            expected[offset + 3] = 2 if active else 0
            if active:
                lane = invocation % 32
                for round in range(2):
                    value = sum(
                        (values[group * 32 + i] + round for i in range(lane)), -0.0
                    )
                    expected[offset + round] = struct.unpack(
                        "<I", struct.pack("<f", value)
                    )[0]
    guards = [marker] * 17
    expected += guards

    def packet(dtype, data):
        return {"dtype": dtype, "shape": [len(data)], "values": data}

    inputs = {
        "inputWords": packet("float32", values),
        "outputWords": packet("uint32", [marker] * len(expected)),
    }
    outputs = _bound_values(
        descriptor,
        {"inputWords": inputs["inputWords"], "outputWords": packet("uint32", expected)},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        outputs,
        {"workgroupCount": [3, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "shape": shape,
                "descriptor": descriptor,
                "inputs": inputs,
                "outputs": outputs,
            },
            indent=2,
        )
    )
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="conditional_scan",
    )


def test_conditional_calls_share_the_required_native_gate():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_subgroup_conditional_calls.py" in step
    assert "--timeout-seconds 180" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step
