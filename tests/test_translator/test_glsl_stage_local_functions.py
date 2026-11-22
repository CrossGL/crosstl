"""GLSL output keeps a stage-local helper's rename inside its stage (issue #5026).

A helper declared in a stage block is visible in that stage only. A vertex helper ``void subgroupBarrier()`` is
renamed because it shadows the built-in, and the fragment stage's ``subgroupBarrier()`` call, which names the
built-in, used to be renamed with it, so the fragment section called a function it never declares.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_stage_local_functions import (
    rename_stage_local_functions_called_elsewhere,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GLSLANG = shutil.which("glslangValidator")

OTHER_STAGE_BUILTIN = """shader P {
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

OWN_HELPERS = """shader P {
    vertex {
        float shade(float x) { return x * 2.0; }
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            return p * shade(p.x);
        }
    }
    fragment {
        float shade(float x) { return x + 1.0; }
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            return vec4(shade(pos.x));
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "stages.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _section(code, macro):
    start = code.index(f"#ifdef {macro}\n// ")
    return code[start : code.index("#endif", start)]


def _assert_glslang_accepts(tmp_path, code, *stages):
    # glslang defines GL_VERTEX_SHADER or GL_FRAGMENT_SHADER for the stage it compiles.
    if GLSLANG is None:
        return
    (tmp_path / "out.glsl").write_text(code, encoding="utf-8")
    for stage in stages:
        result = subprocess.run(
            [GLSLANG, "-S", stage, "out.glsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, stage + "\n" + result.stdout + result.stderr


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def test_other_stage_builtin_call_keeps_the_builtin(tmp_path):
    code = _translate(tmp_path, OTHER_STAGE_BUILTIN, "opengl")

    vertex = _section(code, "GL_VERTEX_SHADER")
    fragment = _section(code, "GL_FRAGMENT_SHADER")
    assert "void subgroupBarrier_fn() {" in vertex
    assert "subgroupBarrier_fn();" in vertex.split("void main()", 1)[1]
    assert "subgroupBarrier();" in fragment
    assert "subgroupBarrier_" not in fragment
    _assert_glslang_accepts(tmp_path, code, "vert", "frag")


def test_each_stage_keeps_its_own_helper_name(tmp_path):
    code = _translate(tmp_path, OWN_HELPERS, "opengl")

    vertex = _section(code, "GL_VERTEX_SHADER")
    fragment = _section(code, "GL_FRAGMENT_SHADER")
    assert "float shade(float x) {" in vertex and "(x * 2.0)" in vertex
    assert "float shade(float x) {" in fragment and "(x + 1.0)" in fragment
    _assert_glslang_accepts(tmp_path, code, "vert", "frag")


def test_program_without_foreign_calls_is_not_copied():
    program = _parse(OWN_HELPERS)

    assert rename_stage_local_functions_called_elsewhere(program) is program


def test_rename_leaves_the_source_program_unchanged():
    program = _parse(OTHER_STAGE_BUILTIN)

    renamed = rename_stage_local_functions_called_elsewhere(program)

    assert renamed is not program

    def helper_names(ast):
        return [
            function.name
            for stage in ast.stages.values()
            for function in stage.local_functions
        ]

    assert helper_names(program) == ["subgroupBarrier"]
    assert helper_names(renamed) == ["subgroupBarrier_fn"]


@pytest.mark.parametrize("backend", ("opengl", "webgl"))
def test_derivative_builtin_of_another_stage_keeps_its_name(tmp_path, backend):
    # A vertex helper named like the built-in inversesqrt; WebGL 2 has no subgroup functions.
    code = _translate(
        tmp_path,
        """shader P {
    vertex {
        float inversesqrt(float x) { return x; }
        vec4 main(vec4 p @ POSITION) @ gl_Position {
            return p * inversesqrt(p.x);
        }
    }
    fragment {
        vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor {
            return vec4(inversesqrt(pos.x));
        }
    }
}
""",
        backend,
    )

    vertex = _section(code, "GL_VERTEX_SHADER")
    fragment = _section(code, "GL_FRAGMENT_SHADER")
    assert "inversesqrt_fn(p.x)" in vertex
    assert "inversesqrt(gl_FragCoord.x)" in fragment
    assert "inversesqrt_" not in fragment
    _assert_glslang_accepts(tmp_path, code, "vert", "frag")
