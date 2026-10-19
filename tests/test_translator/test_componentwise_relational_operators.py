"""CrossGL relational and equality operators are componentwise on vectors (ADR 0002).

GLSL is the only source language whose ``==`` and ``!=`` compare vectors and matrices as a whole (GLSL 4.60 section
5.9); its componentwise forms are the built-ins ``lessThan`` ... ``notEqual`` and ``not`` (section 8.7).  Before:

- the GLSL importer kept ``bool b = (v == w);`` verbatim, which HLSL then truncated to ``v.x == w.x``;
- the HLSL, MSL, WGSL, Slang and SPIR-V emitters passed ``lessThan(a, b)`` and ``not(b)`` through as calls to
  functions that do not exist on those targets, and SPIR-V had no ``any`` or ``all``;
- the GLSL emitter refused ``!`` on a vector, and MSL and WGSL rejected ``!`` on a numeric vector.
"""

import shutil
import subprocess
import textwrap

import pytest

from crosstl.backend.GLSL.aggregate_equality import (
    GlslAggregateEqualityError,
    GlslMatrixEqualityOperandError,
)
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.relational_builtins import (
    lower_relational_builtins,
    metal_logical_not_operand,
    wgsl_logical_not_operand,
)
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

RELATIONAL_PROGRAM = textwrap.dedent(
    """
    shader S {
      struct FragmentInput { vec3 v @ TEXCOORD0; }
      struct FragmentOutput { vec4 color @ gl_FragColor; }
      fragment {
        FragmentOutput main(FragmentInput input) {
          FragmentOutput output;
          vec3 v = input.v;
          bvec3 lt = lessThan(v, vec3(0.5));
          bvec3 ge = greaterThanEqual(v, vec3(0.1));
          bvec3 eq = equal(v, vec3(0.25));
          bvec3 ne = not(eq);
          bvec3 nv = !v;
          bvec3 nb = !lt;
          bool q = any(notEqual(lt, ge)) || all(lessThanEqual(v, vec3(1.0))) || any(greaterThan(v, vec3(2.0)));
          output.color = vec4(float(q), float(ne.x), float(nv.y), float(nb.z));
          return output;
        }
      }
    }
    """
)

GLSL_RELATIONAL_BUILTINS = (
    "lessThan(",
    "lessThanEqual(",
    "greaterThan(",
    "greaterThanEqual(",
    "equal(",
    "notEqual(",
    "not(",
)


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def import_glsl(source, stage="fragment"):
    ast = GLSLParser(GLSLLexer(source).tokenize(), stage).parse()
    return GLSLToCrossGLConverter(shader_type=stage).generate(ast)


def glsl_fragment(declarations, body):
    return (
        "#version 450\n"
        + textwrap.dedent(declarations)
        + "layout(location = 0) out vec4 color;\nvoid main() {\n"
        + textwrap.dedent(body)
        + "color = vec4(1.0);\n}\n"
    )


def statement_lines(code, needle):
    return [line.strip() for line in code.splitlines() if needle in line]


def run_tool(command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, **kwargs
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0 and "missing Metal Toolchain" in diagnostics:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, diagnostics


def require_tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


# GLSL importer: aggregate equality becomes all()/any() over CrossGL's componentwise operators.


@pytest.mark.parametrize(
    "declarations,expression,expected",
    [
        ("uniform vec3 v;\n", "v == vec3(0.25)", "all(v == vec3(0.25))"),
        ("uniform vec3 v; uniform vec2 uv;\n", "v.xy != uv", "any(v.xy != uv)"),
        (
            "uniform vec3 v;\nvec3 f(vec3 x) { return x * 2.0; }\n",
            "f(v) == v",
            "all(f(v) == v)",
        ),
        ("uniform vec3 v;\n", "normalize(v) == v * 2.0", "all(normalize(v) == (v * 2.0))"),
        ("uniform vec3 v;\n", "(v + 1.0) != -v", "any((v + 1.0) != (-v))"),
        ("uniform ivec4 i;\n", "i == ivec4(0)", "all(i == ivec4(0))"),
        ("uniform bvec2 b;\n", "b != bvec2(true)", "any(b != bvec2(true))"),
        (
            "struct S { vec3 a; float b; };\nuniform S s; uniform vec3 v;\n",
            "s.a == v",
            "all(s.a == v)",
        ),
        (
            "uniform sampler2D t; uniform vec2 uv;\n",
            "texture(t, uv) == vec4(0.0)",
            "all(texture(t, uv) == vec4(0.0))",
        ),
        ("uniform vec3 v; uniform vec2 uv;\n", "v.x == uv.y", "(v.x == uv.y)"),
        ("uniform float x;\n", "x != 1.0", "(x != 1.0)"),
    ],
)
def test_glsl_importer_reduces_vector_equality_to_one_bool(
    declarations, expression, expected
):
    code = import_glsl(
        glsl_fragment(declarations, f"bool r = {expression};\n")
    )
    assert statement_lines(code, "bool r =") == [f"bool r = {expected};"]


def test_glsl_importer_reduces_a_local_array_element_comparison():
    code = import_glsl(
        glsl_fragment(
            "uniform vec3 v;\n",
            "vec3 arr[2]; arr[0] = v; arr[1] = v;\nbool r = arr[0] == arr[1];\n",
        )
    )
    assert statement_lines(code, "bool r =") == ["bool r = all(arr[0] == arr[1]);"]


def test_glsl_importer_compares_matrices_column_by_column():
    code = import_glsl(
        glsl_fragment(
            "uniform mat3x2 M; uniform mat3x2 N;\n",
            "bool same = M == N;\nbool differs = M != N;\n",
        )
    )
    assert statement_lines(code, "bool same =") == [
        "bool same = (all(M[0] == N[0]) && all(M[1] == N[1]) && all(M[2] == N[2]));"
    ]
    assert statement_lines(code, "bool differs =") == [
        "bool differs = (any(M[0] != N[0]) || any(M[1] != N[1]) || any(M[2] != N[2]));"
    ]


def test_glsl_importer_reduces_vector_equality_in_conditions():
    code = import_glsl(
        glsl_fragment("uniform vec3 v;\n", "if (v == vec3(0.0)) { discard; }\n")
    )
    assert any("if (all(v == vec3(0.0)))" in line for line in code.splitlines())


def test_glsl_importer_keeps_structure_equality_whole():
    code = import_glsl(
        glsl_fragment(
            "struct S { vec3 a; float b; };\nuniform S s; uniform S t;\n",
            "bool r = s == t;\n",
        )
    )
    assert statement_lines(code, "bool r =") == ["bool r = (s == t);"]


def test_glsl_importer_types_extension_builtins_in_equalities():
    # GL_EXT_fragment_shading_rate declares gl_ShadingRateEXT and its flags as int; GL_EXT_ray_query's
    # intersection-type getter and its constants are uint.
    code = import_glsl(
        "#version 460\n#extension GL_EXT_fragment_shading_rate : require\n"
        "layout(location = 0) out vec4 color;\nvoid main() {\n"
        "bool vertical = (gl_ShadingRateEXT & gl_ShadingRateFlag2VerticalPixelsEXT) "
        "== gl_ShadingRateFlag2VerticalPixelsEXT;\n"
        "color = vec4(float(vertical));\n}\n"
    )
    lines = statement_lines(code, "bool vertical =")
    assert len(lines) == 1 and "all(" not in lines[0]


@pytest.mark.parametrize(
    "type_name,structs,expected",
    [
        ("i16vec2", (), ("vector", 0)),
        ("f64vec3", (), ("vector", 0)),
        ("u8vec4", (), ("vector", 0)),
        ("int64_t", (), ("scalar", 0)),
        ("float16_t", (), ("scalar", 0)),
        ("f64mat2x3", (), ("matrix", 2)),
        ("dmat4", (), ("matrix", 4)),
        ("vec3[]", (), ("aggregate", 0)),
        ("S", ("S",), ("aggregate", 0)),
        ("UnknownOpaque", (), (None, 0)),
    ],
)
def test_glsl_equality_operand_kinds(type_name, structs, expected):
    from crosstl.backend.GLSL.aggregate_equality import _operand_kind

    assert _operand_kind(type_name, structs) == expected


def test_glsl_importer_refuses_equality_whose_operand_types_are_unknown():
    source = glsl_fragment(
        "", "bool r = externalValue(1.0) == externalValue(2.0);\n"
    )
    with pytest.raises(GlslAggregateEqualityError, match="operand types"):
        import_glsl(source)


def test_glsl_importer_refuses_matrix_equality_with_side_effects():
    source = glsl_fragment(
        "uniform mat2 M;\nmat2 g() { return M; }\n", "bool r = g() == M;\n"
    )
    with pytest.raises(GlslMatrixEqualityOperandError, match="side effects"):
        import_glsl(source)


def test_glsl_importer_output_round_trips_to_valid_glsl_and_hlsl(tmp_path):
    source = glsl_fragment(
        "layout(location = 0) in vec3 v;\nuniform mat2 M;\n",
        "bool a = v == vec3(0.25);\nbool b = M != mat2(1.0);\n"
        "if (a || b) { discard; }\n",
    )
    crossgl = import_glsl(source)
    glsl = GLSLCodeGen().generate(parse(crossgl))
    assert "all(equal(" in glsl
    assert "any(notEqual(" in glsl
    hlsl = HLSLCodeGen().generate(parse(crossgl))
    assert "all((v == float3(0.25" in hlsl.replace("input.", "")
    glslang = shutil.which("glslangValidator")
    if glslang is not None:
        run_tool([glslang, "--stdin", "-S", "frag"], input=glsl)


# Emitters other than GLSL lower GLSL's relational built-ins into operators.


@pytest.mark.parametrize(
    "emitter,expected",
    [
        (
            HLSLCodeGen,
            [
                "bool3 lt = (v < float3(0.5, 0.5, 0.5));",
                "bool3 ge = (v >= float3(0.1, 0.1, 0.1));",
                "bool3 eq = (v == float3(0.25, 0.25, 0.25));",
                "bool3 ne = !eq;",
                "bool3 nv = !v;",
            ],
        ),
        (
            MetalCodeGen,
            [
                "bool3 lt = v < float3(0.5);",
                "bool3 eq = v == float3(0.25);",
                "bool3 ne = !eq;",
                "bool3 nv = !bool3(v);",
                "bool3 nb = !lt;",
            ],
        ),
        (
            WGSLCodeGen,
            [
                "var lt: vec3<bool> = (v < vec3<f32>(0.5));",
                "var ne: vec3<bool> = !eq;",
                "var nv: vec3<bool> = !vec3<bool>(v);",
            ],
        ),
        (
            SlangCodeGen,
            ["bool3 lt = v < float3(0.5);", "bool3 ne = !eq;"],
        ),
    ],
)
def test_emitters_spell_relational_builtins_as_operators(emitter, expected):
    code = emitter().generate(parse(RELATIONAL_PROGRAM))
    lines = {line.strip() for line in code.splitlines()}
    for statement in expected:
        assert statement in lines
    for builtin in GLSL_RELATIONAL_BUILTINS:
        assert f" {builtin}" not in code and f"({builtin}" not in code


def test_spirv_lowers_relational_builtins_and_boolean_reductions():
    code = VulkanSPIRVCodeGen().generate(parse(RELATIONAL_PROGRAM))
    for instruction in (
        "OpFOrdLessThan",
        "OpFOrdGreaterThanEqual",
        "OpFOrdLessThanEqual",
        "OpFOrdGreaterThan",
        "OpFOrdEqual",
        "OpLogicalNot",
        "OpAny",
        "OpAll",
    ):
        assert instruction in code
    assert "OpFunctionCall" not in code


def test_spirv_reduces_numeric_vectors_and_scalars():
    code = VulkanSPIRVCodeGen().generate(
        parse(
            textwrap.dedent(
                """
                shader S {
                  struct FragmentInput { vec3 v @ TEXCOORD0; }
                  struct FragmentOutput { vec4 color @ gl_FragColor; }
                  fragment {
                    FragmentOutput main(FragmentInput input) {
                      FragmentOutput output;
                      bool a = any(input.v);
                      bool b = all(input.v.x);
                      output.color = vec4(float(a), float(b), 0.0, 1.0);
                      return output;
                    }
                  }
                }
                """
            )
        )
    )
    assert code.count("OpAny") == 1
    assert "OpAll" not in code
    # float(bool) selects 1.0 or 0.0 (GLSL 4.60 section 5.4.1).
    assert "OpSelect" in code


def test_glsl_emitter_spells_vector_logical_not_with_not():
    code = GLSLCodeGen().generate(parse(RELATIONAL_PROGRAM))
    lines = {line.strip() for line in code.splitlines()}
    assert "bvec3 ne = not(eq);" in lines
    assert "bvec3 nv = not(bvec3(v));" in lines
    assert "bvec3 nb = not(lt);" in lines
    assert "bvec3 lt = lessThan(v, vec3(0.5));" in lines


def test_user_functions_named_like_relational_builtins_are_not_lowered():
    program = parse(
        textwrap.dedent(
            """
            shader S {
              struct FragmentInput { vec3 v @ TEXCOORD0; }
              struct FragmentOutput { vec4 color @ gl_FragColor; }
              bool equal(vec3 a, vec3 b) { return all(a == b); }
              fragment {
                FragmentOutput main(FragmentInput input) {
                  FragmentOutput output;
                  bool same = equal(input.v, vec3(0.0));
                  bvec3 lt = lessThan(input.v, vec3(1.0));
                  output.color = vec4(float(same), float(lt.x), 0.0, 1.0);
                  return output;
                }
              }
            }
            """
        )
    )
    assert lower_relational_builtins(program) == 1
    hlsl = HLSLCodeGen().generate(program)
    assert "equal(input.v, float3(0.0, 0.0, 0.0))" in hlsl
    assert "lessThan" not in hlsl


@pytest.mark.parametrize(
    "operand_type,expected",
    [
        ("float3", "bool3(v)"),
        ("int2", "bool2(v)"),
        ("bool4", "v"),
        ("float", "v"),
        (None, "v"),
    ],
)
def test_metal_logical_not_converts_numeric_vectors(operand_type, expected):
    assert metal_logical_not_operand("v", operand_type) == expected


@pytest.mark.parametrize(
    "operand_type,expected",
    [
        ("vec3<f32>", "vec3<bool>(v)"),
        ("vec2<u32>", "vec2<bool>(v)"),
        ("vec4<bool>", "v"),
        ("i32", "bool(v)"),
        ("bool", "v"),
    ],
)
def test_wgsl_logical_not_converts_numeric_operands(operand_type, expected):
    assert wgsl_logical_not_operand("v", operand_type) == expected


# The lowered output is accepted by each target's own compiler or validator.


def test_relational_program_validates_with_glslang(tmp_path):
    glslang = require_tool("glslangValidator")
    run_tool(
        [glslang, "--stdin", "-S", "frag"],
        input=GLSLCodeGen().generate(parse(RELATIONAL_PROGRAM)),
    )


def test_relational_program_validates_with_naga(tmp_path):
    naga = require_tool("naga")
    path = tmp_path / "relational.wgsl"
    path.write_text(WGSLCodeGen().generate(parse(RELATIONAL_PROGRAM)), "utf-8")
    run_tool([naga, str(path)])


def test_relational_program_validates_with_spirv_val(tmp_path):
    spirv_as = require_tool("spirv-as")
    spirv_val = require_tool("spirv-val")
    asm = tmp_path / "relational.spvasm"
    binary = tmp_path / "relational.spv"
    asm.write_text(VulkanSPIRVCodeGen().generate(parse(RELATIONAL_PROGRAM)), "utf-8")
    run_tool([spirv_as, str(asm), "-o", str(binary)])
    run_tool([spirv_val, str(binary)])


def test_relational_program_compiles_with_xcrun_metal(tmp_path):
    xcrun = require_tool("xcrun")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, timeout=30
    )
    if probe.returncode != 0:
        pytest.skip("xcrun cannot locate metal")
    source = tmp_path / "relational.metal"
    source.write_text(MetalCodeGen().generate(parse(RELATIONAL_PROGRAM)), "utf-8")
    run_tool(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(tmp_path / "r.air")]
    )


def test_relational_program_compiles_with_dxc(tmp_path):
    dxc = require_tool("dxc")
    source = tmp_path / "relational.hlsl"
    source.write_text(HLSLCodeGen().generate(parse(RELATIONAL_PROGRAM)), "utf-8")
    run_tool([dxc, "-T", "ps_6_0", "-E", "PSMain", str(source)])
