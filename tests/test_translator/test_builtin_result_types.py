"""Untyped locals take the CrossGL type of built-in calls and of component access, in every C-family emitter.

Before, ``let r = smoothstep(0.0, 1.0, v);`` was declared ``float`` by the GLSL, HLSL and MSL emitters whenever
their own expression typing did not know the built-in (dropping two components of a ``vec3``); ``let c = m[0];``
was declared with the matrix type and ``let x = v[1];`` (HLSL) with the vector type; and ``let n = bitCount(iv);``
was declared ``uint3`` by HLSL although GLSL 4.60 section 8.8 gives ``bitCount`` a signed result.  A local whose
type still cannot be inferred is refused instead of being declared ``float``.
"""

import textwrap

import pytest

from crosstl.translator.codegen.builtin_result_types import (
    C_FAMILY,
    GLSL_FAMILY,
    builtin_call_result_type,
    component_access_type,
    type_shape,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


@pytest.mark.parametrize(
    "name,arguments,family,expected",
    [
        ("smoothstep", ["float", "float", "vec3"], GLSL_FAMILY, "vec3"),
        ("smoothstep", ["float", "float", "float3"], C_FAMILY, "float3"),
        ("step", ["float", "vec4"], GLSL_FAMILY, "vec4"),
        ("min", ["vec3", "float"], GLSL_FAMILY, "vec3"),
        ("clamp", ["ivec2", "int", "int"], GLSL_FAMILY, "ivec2"),
        ("mix", ["float3", "float3", "bool3"], C_FAMILY, "float3"),
        ("pow", ["float", "float3"], C_FAMILY, "float3"),
        ("ldexp", ["vec3", "ivec3"], GLSL_FAMILY, "vec3"),
        ("abs", ["int3"], C_FAMILY, "int3"),
        ("length", ["half3"], C_FAMILY, "half"),
        ("dot", ["dvec2", "dvec2"], GLSL_FAMILY, "double"),
        ("distance", ["vec4", "vec4"], GLSL_FAMILY, "float"),
        ("lessThan", ["vec3", "vec3"], GLSL_FAMILY, "bvec3"),
        ("isnan", ["float2"], C_FAMILY, "bool2"),
        ("any", ["bvec3"], GLSL_FAMILY, "bool"),
        ("bitCount", ["uvec4"], GLSL_FAMILY, "ivec4"),
        ("bitCount", ["uint3"], C_FAMILY, "int3"),
        ("floatBitsToUint", ["vec2"], GLSL_FAMILY, "uvec2"),
        ("intBitsToFloat", ["int"], GLSL_FAMILY, "float"),
        ("packHalf2x16", ["vec2"], GLSL_FAMILY, "uint"),
        ("unpackUnorm4x8", ["uint"], C_FAMILY, "float4"),
        ("unpackSnorm2x16", ["uint"], GLSL_FAMILY, "vec2"),
        ("determinant", ["mat3"], GLSL_FAMILY, "float"),
        ("inverse", ["float4x4"], C_FAMILY, "float4x4"),
        ("transpose", ["mat2x3"], GLSL_FAMILY, "mat3x2"),
        ("transpose", ["dmat4"], GLSL_FAMILY, "dmat4"),
        ("outerProduct", ["vec3", "vec2"], GLSL_FAMILY, "mat2x3"),
        ("outerProduct", ["float3", "float3"], C_FAMILY, "float3x3"),
        ("textureSize", ["sampler2DArray", "int"], GLSL_FAMILY, "ivec3"),
        ("textureSize", ["samplerCube", "int"], C_FAMILY, "int2"),
        ("imageSize", ["image3D"], GLSL_FAMILY, "ivec3"),
    ],
)
def test_builtin_result_types_follow_the_glsl_overloads(
    name, arguments, family, expected
):
    assert builtin_call_result_type(name, arguments, family) == expected


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("smoothstep", ["float", None, "vec3"]),  # an operand of unknown type
        ("normalize", ["MyStruct"]),  # not a scalar, vector or matrix
        ("length", ["mat3"]),  # no overload takes a matrix
        (
            "outerProduct",
            ["float3", "float2"],
        ),  # C-family non-square layout is emitter-defined
        ("notABuiltin", ["vec3"]),
        ("textureSize", ["MyTexture"]),
    ],
)
def test_builtin_result_types_leave_undetermined_calls_undecided(name, arguments):
    assert builtin_call_result_type(name, arguments, GLSL_FAMILY) is None


@pytest.mark.parametrize(
    "type_name,family,expected",
    [
        ("vec3", GLSL_FAMILY, "float"),
        ("uint4", C_FAMILY, "uint"),
        ("bvec2", GLSL_FAMILY, "bool"),
        ("mat3", GLSL_FAMILY, "vec3"),
        ("mat2x4", GLSL_FAMILY, "vec4"),  # a mat2x4 has two columns of four rows
        ("dmat3x2", GLSL_FAMILY, "dvec2"),
        ("float4x4", C_FAMILY, "float4"),
        ("float2x3", C_FAMILY, None),
        ("float", GLSL_FAMILY, None),
        ("Light", GLSL_FAMILY, None),
    ],
)
def test_component_access_selects_a_component_or_a_column(type_name, family, expected):
    assert component_access_type(type_name, family) == expected


def test_type_shapes_parse_both_spelling_families():
    assert type_shape("f16vec3").scalar == "half"
    assert type_shape("half3").family == C_FAMILY
    assert type_shape("mat4x2").matrix == (4, 2)
    assert type_shape("double3x3").matrix == (3, 3)
    assert type_shape("float16_t").scalar == "half"
    assert type_shape("vec5") is None


def let_program(expression):
    return textwrap.dedent(f"""
        shader S {{
            fragment {{
                uniform sampler2D tex;
                vec4 main(vec3 v @ TEXCOORD0) @ gl_FragColor {{
                    ivec3 iv = ivec3(1);
                    mat3 m = mat3(1.0);
                    let r = {expression};
                    return vec4(0.0);
                }}
            }}
        }}
        """)


def declared_type_of_r(emitter, expression):
    ast = Parser(
        Lexer(let_program(expression)).get_tokens(), strict_function_bodies=True
    ).parse()
    output = emitter().generate(ast)
    lines = [line for line in output.splitlines() if " r = " in line]
    assert lines, output
    return lines[0].split(" r = ")[0].split()[-1]


EMITTER_CASES = [
    # expression, GLSL, HLSL, MSL
    ("smoothstep(0.0, 1.0, v)", "vec3", "float3", "float3"),
    ("step(0.5, v)", "vec3", "float3", "float3"),
    ("pow(v, v)", "vec3", "float3", "float3"),
    ("length(v)", "float", "float", "float"),
    ("abs(iv)", "ivec3", "int3", "int3"),
    ("bitCount(iv)", "ivec3", "int3", "int3"),
    ("lessThan(v, v)", "bvec3", "bool3", "bool3"),
    ("m[0]", "vec3", "float3", "float3"),
    ("v[1]", "float", "float", "float"),
    ("length(smoothstep(0.0, 1.0, v))", "float", "float", "float"),
    ("determinant(m)", "float", "float", "float"),
]


@pytest.mark.parametrize("expression,glsl,hlsl,msl", EMITTER_CASES)
def test_emitters_declare_lets_with_the_crossgl_result_type(
    expression, glsl, hlsl, msl
):
    assert declared_type_of_r(GLSLCodeGen, expression) == glsl
    assert declared_type_of_r(HLSLCodeGen, expression) == hlsl
    assert declared_type_of_r(MetalCodeGen, expression) == msl


@pytest.mark.parametrize("emitter", [GLSLCodeGen, HLSLCodeGen, MetalCodeGen])
def test_emitters_refuse_a_let_whose_type_cannot_be_inferred(emitter):
    program = textwrap.dedent("""
        shader S {
            struct Stuff { float a; }
            fragment {
                vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                    Stuff s;
                    let t = s.missing;
                    return vec4(1.0);
                }
            }
        }
        """)
    ast = Parser(Lexer(program).get_tokens(), strict_function_bodies=True).parse()
    with pytest.raises(LoweringRefused) as raised:
        emitter().generate(ast)
    assert "the type of 't' is not written" in str(raised.value)
