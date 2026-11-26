"""A helper declared in one stage block does not shadow built-ins in another stage."""

from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader P {
    vertex {
        void subgroupBarrier() { }
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            subgroupBarrier();
            return p;
        }
    }
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            subgroupBarrier();
            return vec4(pos.x);
        }
    }
}
"""


def _generate(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return GLSLCodeGen().generate(ast)


def _stage_section(code, macro):
    start = code.index(f"#ifdef {macro}\n// ")
    return code[start : code.index("#endif", start)]


def test_other_stage_keeps_the_builtin_call():
    code = _generate(SOURCE)
    fragment = _stage_section(code, "GL_FRAGMENT_SHADER")
    assert "subgroupBarrier();" in fragment
    assert "subgroupBarrier_" not in fragment
    assert "#extension GL_KHR_shader_subgroup_basic : require" in code


def test_declaring_stage_still_calls_its_own_helper():
    vertex = _stage_section(_generate(SOURCE), "GL_VERTEX_SHADER")
    assert "void subgroupBarrier_() {" in vertex
    assert "    subgroupBarrier_();" in vertex


def test_stage_without_a_helper_does_not_require_the_extension():
    source = SOURCE.replace(
        "    fragment {\n        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {\n"
        "            subgroupBarrier();\n",
        "    fragment {\n        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {\n",
    )
    assert "subgroupBarrier();" not in source.split("fragment")[1]
    code = _generate(source)
    assert "GL_KHR_shader_subgroup_basic" not in code


def test_helper_shared_by_all_stages_is_still_called():
    source = """
    shader P {
        void subgroupBarrier() { }
        vertex {
            vec4 main(vec4 p @ POSITION) @ gl_Position {
                subgroupBarrier();
                return p;
            }
        }
        fragment {
            vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
                subgroupBarrier();
                return vec4(pos.x);
            }
        }
    }
    """
    code = _generate(source)
    assert "subgroupBarrier_();" in _stage_section(code, "GL_FRAGMENT_SHADER")
    assert "GL_KHR_shader_subgroup_basic" not in code
