"""Pure floating helpers retain guarded effects and collective participation."""

import json
import math
import os
import pickle
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.translator import parse
from crosstl.translator.ast import AttributeNode, NamedType, StructNode
from crosstl.translator.codegen.subgroup_control_flow import (
    converge_subgroup_conditional_calls,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _codegen
from tests.test_translator.test_subgroup_conditional_calls import REQUIRE_ENV, _source

HELPERS = """struct Operation {};
float corrected_log(float x) {
    float xp1 = 1.0 + x;
    if (xp1 == asfloat(2139095040u)) { return xp1; }
    if (xp1 == 1.0) { return x; }
    return x * (log(xp1) / (xp1 - 1.0));
}
float combine(inout Operation self, float x, float y) {
    if (isnan(x) || isnan(y)) { return asfloat(2143289344u); }
    const float inf = asfloat(2139095040u);
    float high = max(x, y);
    float low = min(x, y);
    return low == -inf || high == inf ? high : high + corrected_log(exp(low - high));
}
float temporary(float x, float y) {
    Operation self;
    return combine(self, x, y);
}
float wrapped(float value) {
    for (int step = 1; step <= 16; step *= 2) {
        float other = WaveShuffleAndFillUp(value, -asfloat(2139095040u), uint(step));
        value = temporary(value, other);
    }
    return WaveShuffleAndFillUp(value, -asfloat(2139095040u), 1u);
}"""


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_floating_helper_chain_converges_without_changing_ast(tmp_path, target):
    ast = parse(_source(helpers=HELPERS))
    original = pickle.dumps(ast)
    generator = _codegen(target)
    generated = generator.generate_stage(ast, "compute")
    assert "subgroup_input" in generated and "subgroup_enabled" in generated
    assert generated.count("counter++") == 1
    assert "log(" in generated and "exp(" in generated
    assert pickle.dumps(ast) == original
    assert generator.generate_stage(ast, "compute") == generated
    _compile(generated, target, tmp_path)


def _assert_not_converged(ast, target):
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


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_precise_float_locals_retain_qualifiers_during_convergence(tmp_path, target):
    helpers = """float wrapped(float value) {
        precise float rounded = value + 1.0;
        return WavePrefixSum(rounded);
    }"""
    ast = parse(_source(helpers=helpers))
    original = pickle.dumps(ast)
    generated = _codegen(target).generate_stage(ast, "compute")
    assert "subgroup_input" in generated and "subgroup_enabled" in generated
    assert "precise float rounded" in generated
    assert pickle.dumps(ast) == original
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "declaration",
    (
        "precise volatile float rounded = value;",
        "precise device float rounded = value;",
        "precise float rounded;",
        "precise uint rounded = 1u;",
    ),
)
def test_precise_locals_do_not_hide_storage_or_initialization(target, declaration):
    helpers = (
        f"float wrapped(float value) {{ {declaration} return WavePrefixSum(value); }}"
    )
    _assert_not_converged(parse(_source(helpers=helpers)), target)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "expression",
    (
        "exp(inputs[0])",
        "log(inputs[0])",
        "min(value, inputs[0])",
        "max(value, float(results[0]))",
        "float(int(value) / 0)",
        "float(1 / int(value))",
        "float(uint(value) / 1u)",
        "value / int(value)",
        "value / inputs[0]",
        "float(int(log(value)))",
        "exp(value, value)",
        "min(value)",
        "min(value, value, value)",
        "exp<int>(value)",
        "log(1)",
        "min(value, 1)",
        "max(half(value), value)",
        "max(value, half(value))",
        "exp(float2(value, value))",
        "tan(value)",
        "sqrt(value, value)",
        "atan2(value)",
        "cos(inputs[0])",
        "sin(int(value))",
    ),
)
def test_unproved_math_arguments_and_conversions_remain_guarded(target, expression):
    helpers = f"float wrapped(float value) {{ return WavePrefixSum({expression}); }}"
    _assert_not_converged(parse(_source(helpers=helpers)), target)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("operation", ("min", "max", "exp", "log"))
def test_owned_math_names_do_not_inherit_builtin_purity(target, operation):
    arguments = "float x, float y" if operation in {"min", "max"} else "float x"
    call = f"{operation}(value, value)" if "," in arguments else f"{operation}(value)"
    helpers = f"""float {operation}({arguments}) {{ results[0] = 1u; return x; }}
    float wrapped(float value) {{ return WavePrefixSum({call}); }}"""
    _assert_not_converged(parse(_source(helpers=helpers)), target)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("metadata", ("attributes", "inheritance", "generic_params"))
def test_empty_receivers_require_an_inert_declaration(target, metadata):
    ast = parse(_source(helpers=HELPERS))
    structure = next(
        node
        for node in _codegen(target).walk_ast(ast)
        if isinstance(node, StructNode) and node.name == "Operation"
    )
    setattr(
        structure,
        metadata,
        (
            [AttributeNode("unproven")]
            if metadata == "attributes"
            else [NamedType("Base")]
        ),
    )
    _assert_not_converged(ast, target)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "declaration",
    (
        "device Operation self;",
        "threadgroup Operation self;",
        "volatile Operation self;",
        "Operation self = make_operation();",
        "Operation self; Operation self;",
    ),
)
def test_empty_local_receiver_does_not_hide_storage_or_initialization_effects(
    target, declaration
):
    _assert_not_converged(
        parse(_source(helpers=HELPERS.replace("Operation self;", declaration))), target
    )


def _native_source(size, selected):
    return f"""#include <metal_stdlib>
using namespace metal;
struct Operation {{}};
float evaluate(thread Operation& self, float value) {{
    if (isnan(value)) {{ return as_type<float>(0x7fc00000u); }}
    if (isinf(value)) {{ return value; }}
    float bounded = max(-4.0f, min(value, 4.0f));
    return log(exp(bounded / 2.0f)) + value / value;
}}
float temporary(float value) {{
    Operation self;
    return evaluate(self, value);
}}
float wrapped(float value, float ignored, uint lane) {{
    float shifted = simd_shuffle_and_fill_up(temporary(value), 0.0f, 1);
    return lane == 0u ? 0.0f : shifted;
}}
kernel void conditional_math(device float* inputWords [[buffer(0)]],
                             device uint* outputWords [[buffer(1)]],
                             uint invocation [[thread_index_in_threadgroup]],
                             uint simd_group_id [[simdgroup_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]]) {{
    uint counter = 0u;
    uint ignoredCounter = 0u;
    uint lane = invocation % 32u;
    uint index = group.x * 32u + invocation - {selected * 32}u;
    uint outputIndex = (group.x * {size}u + invocation) * 4u;
    for (uint round = 0u; round < 2u; ++round) {{
        if (simd_group_id == {selected}u) {{
            float result = wrapped(inputWords[index] + float(counter++), float(ignoredCounter++), lane);
            float reference = lane == 0u ? 0.0f : temporary(inputWords[index - 1u] + float(round));
            outputWords[outputIndex + round] = (isnan(result) && isnan(reference)) ||
                as_type<uint>(result) == as_type<uint>(reference);
        }}
    }}
    outputWords[outputIndex + 2u] = counter;
    outputWords[outputIndex + 3u] = ignoredCounter;
}}"""


@pytest.mark.parametrize("shape", ((7, 5, 1), (32, 4, 1)))
@pytest.mark.parametrize("selected", (0, 1, 3))
def test_floating_helpers_preserve_native_guarded_execution(tmp_path, shape, selected):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for conditional collective execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_native_source(size, selected)
    )
    # Compare the converged result to a lane-local evaluation on the same device.
    # This proves scheduling and effects, not cross-device transcendental accuracy.
    words = [
        word
        for _ in range(12)
        for word in (
            0x00000000,
            0x80000000,
            0x3F800000,
            0xBF800000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0x3EAAAAAB,
        )
    ]
    marker = 0xBAD01234
    expected = [marker] * (3 * size * 4)
    for group in range(3):
        for invocation in range(size):
            offset = (group * size + invocation) * 4
            active = invocation // 32 == selected
            expected[offset + 2] = expected[offset + 3] = 2 if active else 0
            if active:
                expected[offset] = expected[offset + 1] = 1
    expected += [marker] * 17
    inputs = {
        "inputWords": {
            "dtype": "float32",
            "shape": [len(words)],
            "values": words,
            "encoding": FLOAT32_BITS,
        },
        "outputWords": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [marker] * len(expected),
        },
    }
    outputs = _bound_values(
        descriptor,
        {
            "inputWords": inputs["inputWords"],
            "outputWords": {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": expected,
            },
        },
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
                "shape": shape,
                "selected": selected,
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
        original_entry="conditional_math",
    )


def test_floating_helpers_share_the_required_native_gate():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_subgroup_float_helpers.py" in step
    assert "--timeout-seconds 300" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step
