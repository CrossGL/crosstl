"""Private value structs retain initialization and effects across collectives."""

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
from crosstl.translator.ast import FunctionNode
from crosstl.translator.codegen.subgroup_control_flow import (
    converge_subgroup_conditional_calls,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package
from tests.test_translator.test_software_subgroup_votes import _codegen
from tests.test_translator.test_subgroup_conditional_calls import REQUIRE_ENV, _source


def _helpers(constructor=None):
    constructor = constructor or """Pair value;
        value.real = real;
        value.imag = imag;
        return value;"""
    return f"""struct Pair {{ float real; float imag; }};
    Pair make_value(float real, float imag) {{ {constructor} }}
    bool shuffle(bool value, bool fill, uint16 delta) {{
        return WaveShuffleAndFillUp(uint(value), uint(fill), delta) != 0u;
    }}
    Pair shuffle(Pair value, Pair fill, uint16 delta) {{
        return make_value(WaveShuffleAndFillUp(value.real, fill.real, delta),
                          WaveShuffleAndFillUp(value.imag, fill.imag, delta));
    }}
    Pair wrapped(Pair value) {{
        for (int step = 1; step <= 16; step *= 2) {{
            Pair other = shuffle(value, make_value(0.0, 0.0), step);
            value = make_value(value.real + other.real, value.imag + other.imag);
        }}
        return shuffle(value, make_value(0.0, 0.0), 1);
    }}"""


def _program(helpers):
    return _source(
        """uint counter = 0u;
        if (subgroup == 1u) {
            Pair result = wrapped(make_value(inputs[invocation + counter++], inputs[invocation]));
            results[invocation] = asuint(result.real) ^ asuint(result.imag);
        }""",
        helpers,
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_private_struct_constructors_and_typed_overloads_converge(tmp_path, target):
    ast = parse(_program(_helpers()))
    original = pickle.dumps(ast)
    generator = _codegen(target)
    generated = generator.generate_stage(ast, "compute")
    assert pickle.dumps(ast) == original
    assert generated.count("counter++") == 1
    assert "subgroup_enabled" in generated and "subgroup_input" in generated
    _compile(generated, target, tmp_path)
    assert generator.generate_stage(ast, "compute") == generated


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "construction",
    ("return Pair(real, imag);", "return {real, imag};"),
)
def test_private_struct_constructor_forms_converge(tmp_path, target, construction):
    generated = _codegen(target).generate_stage(
        parse(_program(_helpers(construction))), "compute"
    )
    assert "subgroup_input" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize("tag_type", ("uint16", "uint", "int"))
def test_private_struct_mixed_scalar_fields_keep_typed_initializers(
    tmp_path, target, tag_type
):
    helpers = f"""struct Pair {{ float real; {tag_type} tag; bool enabled; }};
    Pair make_value(float real, {tag_type} tag, bool enabled) {{
        Pair value;
        value.real = real; value.tag = tag; value.enabled = enabled;
        return value;
    }}
    Pair wrapped(Pair value) {{ value.real = WavePrefixSum(value.real); return value; }}"""
    body = f"""if (subgroup == 1u) {{
        Pair result = wrapped(make_value(inputs[invocation], {tag_type}(17), true));
        results[invocation] = result.enabled ? uint(result.tag) : asuint(result.real);
    }}"""
    generated = _codegen(target).generate_stage(
        parse(_source(body, helpers)), "compute"
    )
    assert "subgroup_input" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "constructor",
    (
        "Pair value; value.real = real; return value;",
        "Pair value; value.real = value.imag; value.imag = imag; return value;",
        "Pair value; if (real > 0.0) { value.real = real; value.imag = imag; } return value;",
        "Pair value; for (int i = 0; i < 0; ++i) { value.real = real; value.imag = imag; } return value;",
        "Pair value; value.real = inputs[0]; value.imag = imag; return value;",
        "Pair value; value.real = real; value.imag = imag; results[0] = 1u; return value;",
        "Pair value; value.real = real; value.imag = float(int(imag)); return value;",
        "return Pair(real);",
        "return Pair(real, imag, 1.0);",
        "return {real};",
        "return {real, imag, 1.0};",
    ),
)
def test_uninitialized_or_effectful_value_structs_are_not_speculated(
    target, constructor
):
    ast = parse(_program(_helpers(constructor)))
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
    "qualifier", ("inout", "out", "volatile", "threadgroup", "device")
)
def test_value_struct_reference_and_storage_parameters_remain_rejected(
    target, qualifier
):
    helpers = _helpers().replace(
        "Pair wrapped(Pair value)", f"Pair wrapped({qualifier} Pair value)"
    )
    ast = parse(_program(helpers))
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
def test_scalar_boolean_staging_uses_a_boolean_identity(tmp_path, target):
    helpers = "float wrapped(bool value) { return WavePrefixSum(value ? 1.0 : 0.0); }"
    generated = _codegen(target).generate_stage(
        parse(_source(helpers=helpers)), "compute"
    )
    assert "subgroup_input = false" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_conversion_safety_does_not_choose_an_ambiguous_overload(target):
    helpers = """float choose(int value) { results[0] = 1u; return float(value); }
    float choose(float value) { return value; }
    float wrapped(float value) { return WavePrefixSum(value + choose(4294967295u)); }"""
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
@pytest.mark.parametrize("qualifier", ("device", "constant", "threadgroup", "volatile"))
def test_type_level_storage_qualifiers_are_not_private_values(target, qualifier):
    ast = parse(_program(_helpers()))
    generator = _codegen(target)
    function = next(
        node
        for node in generator.walk_ast(ast)
        if isinstance(node, FunctionNode) and node.name == "wrapped"
    )
    function.parameters[0].param_type.qualifiers = [qualifier]
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


def _native_source(size, selected):
    return f"""#include <metal_stdlib>
using namespace metal;
struct Pair {{ float real; float imag; }};
Pair shuffle(Pair value, Pair fill, ushort delta) {{
    return {{simd_shuffle_and_fill_up(value.real, fill.real, delta),
             simd_shuffle_and_fill_up(value.imag, fill.imag, delta)}};
}}
bool shuffle(bool value, bool fill, ushort delta) {{
    return simd_shuffle_and_fill_up(uint(value), uint(fill), delta) != 0u;
}}
Pair scan(Pair value, uint lane) {{
    for (int step = 1; step <= 16; step *= 2) {{
        Pair other = shuffle(value, Pair{{0.0f, 0.0f}}, step);
        if (lane < uint(step)) {{ other = Pair{{0.0f, 0.0f}}; }}
        value = Pair{{value.real + other.real, value.imag + other.imag}};
    }}
    Pair previous = shuffle(value, Pair{{0.0f, 0.0f}}, 1);
    return lane == 0u ? Pair{{0.0f, 0.0f}} : previous;
}}
Pair wrapped(Pair value, float ignored, uint lane) {{
    return scan(value, lane);
}}
kernel void conditional_pair(device float* inputWords [[buffer(0)]],
                             device uint* outputWords [[buffer(1)]],
                             uint invocation [[thread_index_in_threadgroup]],
                             uint simd_group_id [[simdgroup_index_in_threadgroup]],
                             uint3 group [[threadgroup_position_in_grid]]) {{
    uint counter = 0u;
    uint ignoredCounter = 0u;
    uint index = group.x * 64u + 2u * (invocation - {selected * 32}u);
    uint outputIndex = (group.x * {size}u + invocation) * 6u;
    for (uint round = 0u; round < 2u; ++round) {{
        if (simd_group_id == {selected}u) {{
            Pair result = wrapped(Pair{{inputWords[index] + float(counter++), inputWords[index + 1u]}}, float(ignoredCounter++), invocation % 32u);
            outputWords[outputIndex + 2u * round] = as_type<uint>(result.real);
            outputWords[outputIndex + 2u * round + 1u] = as_type<uint>(result.imag);
        }}
    }}
    outputWords[outputIndex + 4u] = counter;
    outputWords[outputIndex + 5u] = ignoredCounter;
}}"""


@pytest.mark.parametrize("shape", ((7, 5, 1), (32, 4, 1)))
@pytest.mark.parametrize("selected", (0, 1, 3))
def test_value_struct_collectives_preserve_fields_and_guarded_effects(
    tmp_path, shape, selected
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for conditional collective execution")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, "float", shape, source=_native_source(size, selected)
    )
    values = [(index % 19 - 9) * 0.125 for index in range(192)]
    marker = 0xBAD01234
    expected = [marker] * (3 * size * 6)
    for group in range(3):
        for invocation in range(size):
            offset = (group * size + invocation) * 6
            active = invocation // 32 == selected
            expected[offset + 4] = expected[offset + 5] = 2 if active else 0
            if active:
                for round in range(2):
                    for component in range(2):
                        value = sum(
                            values[group * 64 + i * 2 + component]
                            + (round if component == 0 else 0)
                            for i in range(invocation % 32)
                        )
                        expected[offset + round * 2 + component] = struct.unpack(
                            "<I", struct.pack("<f", value)
                        )[0]
    expected += [marker] * 17

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
        original_entry="conditional_pair",
    )


def test_value_structs_share_the_required_native_gate():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate software subgroup products"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_subgroup_value_structs.py" in step
    assert "--timeout-seconds 180" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step
