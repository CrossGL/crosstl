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
    "header",
    (
        "int step = 1; step <= 16; step *= 2",
        "uint step = 1u; step <= 16u; step <<= 1u",
        "int step = 1; step < 4; step += 1",
    ),
)
def test_private_accumulators_and_pure_combiners_converge(tmp_path, target, header):
    helpers = f"""float combine(float a, float b) {{
        if (isnan(a) || isnan(b)) {{ return asfloat(2143289344u); }}
        return a <= b ? a : b;
    }}
    float inner(float value) {{
        for ({header}) {{
            float other = WaveShuffleAndFillUp(value, asfloat(2139095040u), uint(step));
            value = combine(value, other);
        }}
        return value;
    }}
    float wrapped(float value) {{
        value = inner(value);
        return WaveShuffleAndFillUp(value, asfloat(2139095040u), 1u);
    }}"""
    generator = _codegen(target)
    ast = parse(_source(helpers=helpers))
    before = pickle.dumps(ast)
    generated = generator.generate_stage(ast, "compute")
    assert "subgroup_enabled" in generated
    assert pickle.dumps(ast) == before
    assert generator.generate_stage(ast, "compute") == generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_narrowed_wide_integer_identity_retains_its_value(tmp_path, target):
    helpers = """int wrapped(int value) {
        for (int step = 1; step <= 16; step *= 2) {
            int other = WaveShuffleAndFillUp(value, int(-2147483648l), uint(step));
            value = value >= other ? value : other;
        }
        return value;
    }"""
    generated = _codegen(target).generate_stage(
        parse(_source(helpers=helpers)), "compute"
    )
    assert "subgroup_enabled" in generated
    _compile(generated, target, tmp_path)


# Staged zero inputs only run in disabled subgroups, and both targets define
# integer +, - and * as wrapping, so these wrappers add no trap or effect.
@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "helpers",
    (
        "float wrapped(float value) { int local = asint(value) + 2147483647; return WavePrefixSum(float(local)); }",
        "float wrapped(float value) { int local = -asint(value); return WavePrefixSum(float(local)); }",
        # MLX's two-word 64-bit scan: a bit-preserving uvec2 shuffle per step.
        """int64 shuffle(int64 data, int64 fill, uint delta) {
            return as_type<int64>(WaveShuffleAndFillUp(as_type<uvec2>(data), as_type<uvec2>(fill), delta));
        }
        uint64 shuffle(uint64 data, uint64 fill, uint delta) {
            return as_type<uint64>(WaveShuffleAndFillUp(as_type<uvec2>(data), as_type<uvec2>(fill), delta));
        }
        int64 scan(int64 value) {
            for (int step = 1; step <= 16; step *= 2) {
                value = value + shuffle(value, (int64)(0), uint(step));
            }
            return value;
        }
        float wrapped(float value) { return float(scan(int64(asint(value)))); }""",
        # Compound updates read the whole current value.
        """bool scan(bool value) {
            for (int step = 1; step <= 16; step *= 2) {
                bool other = WaveShuffleAndFillUp(uint(value), 1u, uint(step)) != 0u;
                value &= other;
            }
            return value;
        }
        float wrapped(float value) { bool all = scan(value > 0.0); return all ? 1.0 : 0.0; }""",
        # MLX's CumProd spells its identity static_cast<U>(1.0f): an exactly
        # integral literal converted to the integer type it fits.
        """uint64 shuffle(uint64 data, uint64 fill, uint delta) {
            return as_type<uint64>(WaveShuffleAndFillUp(as_type<uvec2>(data), as_type<uvec2>(fill), delta));
        }
        uint64 scan(uint64 value) {
            for (int step = 1; step <= 16; step *= 2) {
                value = value * shuffle(value, (uint64)(1.0f), uint(step));
            }
            return value;
        }
        float wrapped(float value) { return float(scan(uint64(asuint(value))) + (uint64)(-0.0)); }""",
        # Binary16 extrema and mixed-precision arms take the usual conversion.
        """float16 combine(float16 a, float16 b) {
            if (isnan(a) || isnan(b)) { return float16(asfloat(2143289344u)); }
            return a >= b ? a : b;
        }
        float16 widened(float16 a, float16 b) {
            float16 high = max(a, b);
            return high == float16(0.0) ? high : high + exp(float(min(a, b) - high));
        }
        float wrapped(float value) {
            float16 half_value = float16(value);
            float16 other = WaveShuffleAndFillUp(half_value, -float16(asfloat(2139095040u)), 1u);
            return float(widened(combine(half_value, other), other));
        }""",
    ),
)
def test_wrapping_integer_and_narrow_float_wrappers_converge(tmp_path, target, helpers):
    ast = parse(_source(helpers=helpers))
    generator = _codegen(target)
    entries = (
        generator.hlsl_software_subgroup_stage_entries(ast)
        if target == "directx"
        else generator.glsl_software_subgroup_stage_entries(ast)
    )
    lowered = converge_subgroup_conditional_calls(
        ast, entries, generator.walk_ast, generator.map_operator
    )
    assert lowered is not ast
    generated = generator.generate_stage(ast, "compute")
    assert "subgroup_enabled" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "helpers",
    (
        "float wrapped(float value) { if (value > 0.0) { return value; } return WavePrefixSum(value); }",
        "float wrapped(float value) { return value > 0.0 ? WavePrefixSum(value) : value; }",
        "bool wrapped(float value) { return value > 0.0 && WaveActiveAnyTrue(value < 1.0); }",
    ),
)
def test_pure_wrappers_still_require_uniform_barrier_participation(target, helpers):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(parse(_source(helpers=helpers)), "compute")


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
        "float wrapped(float value) { float local = inputs[0]; return WavePrefixSum(value + local); }",
        "float wrapped(float value) { float* alias = &value; *alias = 0.0; return WavePrefixSum(value); }",
        "float wrapped(float value) { for (int i = 0; i < 16; i *= 2) { value = WavePrefixSum(value); } return value; }",
        "float wrapped(float value) { for (int i = 1; i <= 1073741824; i *= 2) { value = WavePrefixSum(value); } return value; }",
        "float wrapped(float value) { for (int i = 1; i <= 16; i *= 2) { i = int(value); value = WavePrefixSum(value); } return value; }",
        "float wrapped(float value) { for (int i = 1; i <= 16; i *= 2) { value = WavePrefixSum(value); return value; } return value; }",
        "float wrapped(float value) { volatile float local = value; return WavePrefixSum(local); }",
        "float wrapped(float value) { threadgroup float local = value; return WavePrefixSum(local); }",
        "float wrapped(float value) { int local = value; return WavePrefixSum(float(local)); }",
        "int narrowed(float value) { return value; } float wrapped(float value) { return WavePrefixSum(float(narrowed(value))); }",
        "int inner(int value) { return value; } float wrapped(float value) { return WavePrefixSum(float(inner(value))); }",
        "int ignored(int value, int other) { return other; } float wrapped(float value) { return WavePrefixSum(float(ignored(value, 1))); }",
        "float wrapped(float value) { int local = 1 / asint(value); return WavePrefixSum(float(local)); }",
        "float wrapped(float value) { int local = 2147483647 + 1; return WavePrefixSum(value + float(local)); }",
        "float wrapped(float value) { int64 local = as_type<int64>(asuint(value)); return WavePrefixSum(float(local)); }",
        "float wrapped(float value) { float local = value; while (local > 0.0) { local = WavePrefixSum(local); } return local; }",
        "float wrapped(float value) { float local = value; if (value > 0.0) { local = WavePrefixSum(local); } return local; }",
        "float wrapped(float value) { return WavePrefixSum(float(1 << asint(value))); }",
        "float wrapped(float value) { int local = int(9223372036854775807l); return WavePrefixSum(value + float(local)); }",
        "float wrapped(float value) { int local = int(-9223372036854775807l - 1l); return WavePrefixSum(value + float(local)); }",
        # Only exactly integral, finite literals that fit convert exactly.
        "float wrapped(float value) { int local = int(1.5); return WavePrefixSum(value + float(local)); }",
        "float wrapped(float value) { uint local = uint(-1.0); return WavePrefixSum(value + float(local)); }",
        "float wrapped(float value) { int local = int(4294967296.0); return WavePrefixSum(value + float(local)); }",
        "float wrapped(float value) { int local = int(INFINITY); return WavePrefixSum(value + float(local)); }",
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


def _native_source(size, selected, operation="sum"):
    scan = "return simd_prefix_exclusive_sum(value);"
    combine = ""
    if operation != "sum":
        comparison = "<=" if operation == "min" else ">="
        fill = "0x7f800000u" if operation == "min" else "0xff800000u"
        combine = f"""float combine(float a, float b) {{
            if (isnan(a) || isnan(b)) {{ return as_type<float>(0x7fc00000u); }}
            return a {comparison} b ? a : b;
        }}"""
        scan = f"""for (int step = 1; step <= 16; step *= 2) {{
            float other = simd_shuffle_and_fill_up(value, as_type<float>({fill}), step);
            if (lane < uint(step)) {{ other = as_type<float>({fill}); }}
            value = combine(value, other);
        }}
        float prior = simd_shuffle_and_fill_up(value, as_type<float>({fill}), 1);
        return lane == 0u ? as_type<float>({fill}) : prior;"""
    return f"""#include <metal_stdlib>
using namespace metal;
{combine}
struct Operation {{
    float scan(float value, uint lane) {{ {scan} }}
}};
float wrapped(thread Operation& op, float value, float ignored, uint lane) {{ return op.scan(value, lane); }}
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
            float result = wrapped(op, inputWords[group.x * 32u + invocation - {selected * 32}u] + float(counter++), float(ignoredCounter++), invocation % 32u);
            outputWords[outputIndex + round] = as_type<uint>(result);
        }}
    }}
    outputWords[outputIndex + 2u] = counter;
    outputWords[outputIndex + 3u] = ignoredCounter;
}}
"""


@pytest.mark.parametrize("shape", ((7, 5, 1), (32, 4, 1)))
@pytest.mark.parametrize("selected", (0, 1, 3))
@pytest.mark.parametrize("operation", ("sum", "min", "max"))
def test_conditional_calls_preserve_guarded_effects_on_device(
    tmp_path, shape, selected, operation
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for conditional collective execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path,
        target,
        "float",
        shape,
        source=_native_source(size, selected, operation),
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
                    prefix = [values[group * 32 + i] + round for i in range(lane)]
                    value = (
                        sum(prefix, -0.0)
                        if operation == "sum"
                        else (
                            min(prefix, default=math.inf)
                            if operation == "min"
                            else max(prefix, default=-math.inf)
                        )
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
                "operation": operation,
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
    assert "--timeout-seconds 300" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step
