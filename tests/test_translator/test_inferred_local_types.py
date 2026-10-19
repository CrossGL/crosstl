"""Locals declared without a written type take their initializer's type, or are refused.

Before:
* ``var name = expr;`` (the Mojo and Slang importers emit it) did not parse, so the function lost its body;
* the HLSL emitter typed ``let i = gl_GlobalInvocationID.x;`` as ``float`` and
  ``let c = gl_GlobalInvocationID;`` as ``float`` (dropping two components), because it fell back on ``float``
  whenever it could not infer an initializer's type.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.ast import VariableNode
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

VAR_PROGRAM = textwrap.dedent("""
    shader S {
        compute {
            buffer float data[];
            @numthreads(64, 1, 1)
            void main() {
                var i = gl_GlobalInvocationID.x;
                var y = data[i] * 2.0;
                i = i + 1u;
                data[i] = y;
            }
        }
    }
    """)

BUILTIN_LET_PROGRAM = textwrap.dedent("""
    shader S {
        compute {
            buffer float data[];
            @numthreads(64, 1, 1)
            void main() {
                let a = gl_GlobalInvocationID.x;
                let b = gl_LocalInvocationIndex;
                let c = gl_GlobalInvocationID;
                let w = gl_WorkGroupID.yz;
                data[a] = float(b + c.y + w.x);
            }
        }
    }
    """)


def translate(tmp_path, text, backend):
    source = tmp_path / "shader.cgl"
    source.write_text(text)
    return crosstl.translate(str(source), backend=backend)


def function_body(program, name="main"):
    ast = Parser(Lexer(program).tokens, strict_function_bodies=True).parse()
    for stage in ast.stages.values():
        entry = getattr(stage, "entry_point", None)
        if entry is not None and entry.name == name:
            return entry.body.statements
    raise AssertionError(f"no entry point {name}")


def test_an_inferred_var_parses_as_a_mutable_untyped_local():
    first = function_body(VAR_PROGRAM)[0]
    assert isinstance(first, VariableNode)
    assert first.name == "i"
    assert first.var_type is None
    assert "mut" in first.qualifiers


@pytest.mark.parametrize(
    "backend,declaration",
    [
        ("metal", "uint i = thread_position_in_grid.x;"),
        ("opengl", "uint i = gl_GlobalInvocationID.x;"),
        ("directx", "uint i = dispatchThreadID.x;"),
    ],
)
def test_an_inferred_var_takes_its_initializers_type(tmp_path, backend, declaration):
    output = translate(tmp_path, VAR_PROGRAM, backend)
    assert declaration in output
    assert "float y =" in output


def test_an_inferred_var_reaches_spirv_as_a_uint_function_variable(tmp_path):
    output = translate(tmp_path, VAR_PROGRAM, "vulkan")
    assert "%i = OpVariable %_ptr_Function_uint Function" in output


def test_hlsl_types_lets_of_builtin_variables(tmp_path):
    output = translate(tmp_path, BUILTIN_LET_PROGRAM, "directx")
    assert "uint a = dispatchThreadID.x;" in output
    assert "uint b = groupIndex;" in output
    assert "uint3 c = dispatchThreadID;" in output
    assert "uint2 w = " in output
    assert "float a" not in output
    assert "float c" not in output


def test_hlsl_refuses_a_local_whose_type_cannot_be_inferred(tmp_path):
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
    with pytest.raises(DirectXLoweringRefused) as raised:
        translate(tmp_path, program, "directx")
    assert "the type of 't' is not written" in str(raised.value)
