"""Slang generation looks stage, semantic and call names up exactly and refuses names Slang cannot write."""

import pytest

from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.stage_utils import UnknownShaderStageError

COMPUTE = """shader Probe {{
{functions}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* o;
        void main(uvec3 gid @builtin(global_invocation_id)) {{
            float x = o[gid.x];
            o[gid.x] = {value};
        }}
    }}
}}
"""


def generate(source):
    return SlangCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def stored_value(slang):
    lines = [line.strip() for line in slang.splitlines() if "o[gid.x] =" in line]
    assert len(lines) == 1, slang
    return lines[0]


@pytest.mark.parametrize(
    "stage, spelling",
    [
        ("tessellation_control", "hull"),
        ("tessellation_evaluation", "domain"),
        ("task", "amplification"),
        ("object", "amplification"),
        ("ray_generation", "raygeneration"),
        ("ray_closest_hit", "closesthit"),
        ("fragment", "fragment"),
    ],
)
def test_stage_names_have_their_slang_shader_attribute_spelling(stage, spelling):
    assert SlangCodeGen().slang_shader_stage_name(stage) == spelling


def test_a_name_that_is_no_stage_is_refused():
    # The table returned an unknown name unchanged, so [shader("pixle")] reached slangc.
    with pytest.raises(UnknownShaderStageError, match="'pixle' names no shader stage"):
        SlangCodeGen().slang_shader_stage_name("pixle")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("uintBitsToFloat(floatBitsToUint(x))", "o[gid.x] = asfloat(asuint(x));"),
        ("intBitsToFloat(floatBitsToInt(x))", "o[gid.x] = asfloat(asint(x));"),
        ("acosh(x + 2.0)", "o[gid.x] = acosh(x + 2.0);"),
        ("lit(x, x, x).y", "o[gid.x] = lit(x, x, x).y;"),
    ],
)
def test_slang_and_hlsl_built_ins_are_written(value, expected):
    assert stored_value(generate(COMPUTE.format(functions="", value=value))) == expected


@pytest.mark.parametrize(
    "value", ["simd_sum(x)", "x; EmitVertex()", "x; threadgroup_barrier()"]
)
def test_another_languages_built_in_function_is_refused(value):
    # simd_sum and threadgroup_barrier are MSL, EmitVertex is a GLSL geometry built-in; Slang declares none.
    with pytest.raises(
        SlangLoweringRefused, match="Slang declares no function of that name"
    ):
        generate(COMPUTE.format(functions="", value=value))


def test_a_declared_function_with_a_built_in_name_is_called():
    slang = generate(
        COMPUTE.format(
            functions="    float simd_sum(float v) { return v; }", value="simd_sum(x)"
        )
    )
    assert stored_value(slang) == "o[gid.x] = simd_sum(x);"


def test_a_semantic_built_in_without_a_slang_system_value_is_refused():
    # @gl_NumWorkGroups on a varying became a user semantic no stage writes.
    source = """shader Probe {
        struct VertexInput { vec4 position @gl_NumWorkGroups; }
        vertex {
            vec4 main(VertexInput input) @gl_Position { return input.position; }
        }
    }"""
    with pytest.raises(SlangLoweringRefused, match="no system value for the built-in"):
        generate(source)


def test_user_semantics_keep_their_spelling():
    source = """shader Probe {
        struct VertexInput { vec4 position @POSITION; vec2 uv @TEXCOORD3; }
        vertex {
            vec4 main(VertexInput input) @gl_Position { return input.position; }
        }
    }"""
    slang = generate(source)
    assert ": POSITION" in slang
    assert ": TEXCOORD3" in slang
    assert ": SV_Position" in slang
