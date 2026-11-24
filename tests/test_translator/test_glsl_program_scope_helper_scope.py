"""A program-scope function sees only program-scope helpers (#5046).

A helper declared at program scope that calls a built-in such as
``subgroupBarrier()`` keeps the built-in call even when one stage block
declares a helper of that name. The stage-local helper does not shadow the
built-in in program scope, the program-scope function is not copied into
the stage that declares the helper, and the output still requires
``GL_KHR_shader_subgroup_basic``.
"""

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.stage_bound_functions import bind_stage_functions

SOURCE = """
shader P {
    void sync() { subgroupBarrier(); }
    vertex {
        void subgroupBarrier() { }
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            sync();
            return p;
        }
    }
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            sync();
            return vec4(pos.x);
        }
    }
}
"""

STAGE_LOCAL_CALL = """
shader P {
    void helper() { localFn(); }
    vertex {
        void localFn() { }
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            helper();
            return p;
        }
    }
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            return vec4(pos.x);
        }
    }
}
"""

PROGRAM_SCOPE_HELPER = """
shader P {
    void subgroupBarrier() { }
    void sync() { subgroupBarrier(); }
    vertex {
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            sync();
            return p;
        }
    }
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            sync();
            return vec4(pos.x);
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _generate(source):
    return GLSLCodeGen().generate(_parse(source))


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _stage_section(code, macro):
    start = code.index(f"#ifdef {macro}\n// ")
    return code[start : code.index("#endif", start)]


def _stage(program, name):
    return next(
        stage
        for stage in program.stages.values()
        if str(getattr(stage.stage, "value", stage.stage)).lower() == name
    )


def test_builtin_call_does_not_bind_the_function_to_a_stage():
    program = _parse(SOURCE)
    assert bind_stage_functions(program) == 0
    assert [function.name for function in program.functions] == ["sync"]
    assert [
        function.name for function in _stage(program, "vertex").local_functions
    ] == ["subgroupBarrier"]


def test_program_scope_function_keeps_the_builtin_call(tmp_path):
    code = _translate(tmp_path, SOURCE)
    assert "subgroupBarrier();" in code
    assert "    subgroupBarrier_();\n" not in code
    assert code.count("void sync() {") == 1
    assert "void sync() {" not in _stage_section(code, "GL_VERTEX_SHADER")
    assert "#extension GL_KHR_shader_subgroup_basic : require" in code


def test_direct_codegen_keeps_the_builtin_call():
    code = _generate(SOURCE)
    assert "subgroupBarrier();" in code
    assert "    subgroupBarrier_();\n" not in code
    assert code.count("void sync() {") == 1
    assert "#extension GL_KHR_shader_subgroup_basic : require" in code


def test_program_scope_helper_calling_stage_local_helper_is_still_bound():
    program = _parse(STAGE_LOCAL_CALL)
    assert bind_stage_functions(program) == 1
    assert [
        function.name for function in _stage(program, "vertex").local_functions
    ] == ["helper", "localFn"]
    assert program.functions == []


def test_program_scope_helper_named_like_a_builtin_still_wins(tmp_path):
    code = _translate(tmp_path, PROGRAM_SCOPE_HELPER)
    assert "subgroupBarrier_();" in code
    assert "GL_KHR_shader_subgroup_basic" not in code
