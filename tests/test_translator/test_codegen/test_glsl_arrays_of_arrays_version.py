"""An array of arrays raises the GLSL ``#version`` line, and WebGL 2 refuses one.

Arrays of arrays arrived in GLSL 4.30 and GLSL ES 3.10 (section 4.1.9 of both specifications), so GLSL ES 3.00,
the only version WebGL 2 compiles, has none. Generated programs are compiled with glslangValidator when it is
installed.
"""

import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import WebGLLoweringRefused
from crosstl.translator.codegen.glsl_version_requirements import (
    ARRAYS_OF_ARRAYS,
    declares_array_of_arrays,
    generated_program_requirements,
)
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen


def compile_with_glslang(tmp_path, code, stage="frag"):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    source = tmp_path / f"program.{stage}"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [glslang, str(source)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def generate(source, codegen=GLSLCodeGen):
    return codegen().generate(crosstl.translator.parse(source))


FRAGMENT = """
{version}
shader Probe {{
    {declarations}
    struct FragmentOutput {{ vec4 color @ gl_FragColor; }}
    fragment {{
        FragmentOutput main() {{
            FragmentOutput output;
            float x = gl_FragCoord.x;
            {body}
            return output;
        }}
    }}
}}
"""

ARRAY_OF_ARRAYS_PROGRAMS = {
    "local": (
        "",
        "float a[2][2]; a[1][1] = x; output.color = vec4(a[1][1]);",
    ),
    "constructor": (
        "",
        "float a[2][2]; a = float[2][2](float[2](x, 2.0), float[2](x, x));"
        " output.color = vec4(a[1][0]);",
    ),
    "struct-member": (
        "struct Cell { float w[2][3]; }",
        "Cell c; c.w[1][2] = x; output.color = vec4(c.w[1][2]);",
    ),
    "parameter": (
        "float pick(float t[2][2]) { return t[1][0]; }",
        "float a[2][2]; a[1][0] = x; output.color = vec4(pick(a));",
    ),
}


def program(name, version):
    declarations, body = ARRAY_OF_ARRAYS_PROGRAMS[name]
    return FRAGMENT.format(version=version, declarations=declarations, body=body)


@pytest.mark.parametrize("name", sorted(ARRAY_OF_ARRAYS_PROGRAMS))
@pytest.mark.parametrize(
    "version,emitted",
    [
        ("#version 300 es", "#version 310 es"),
        ("#version 420 core", "#version 430 core"),
        ("#version 330 core", "#version 430 core"),
    ],
)
def test_array_of_arrays_raises_version(tmp_path, name, version, emitted):
    code = generate(program(name, version))

    assert code.startswith(f"{emitted}\n")
    compile_with_glslang(tmp_path, code)


def test_arrays_of_arrays_extension_keeps_desktop_version():
    # GL_ARB_arrays_of_arrays gives a GLSL 4.20 program arrays of arrays without a raise.
    code = generate(
        program(
            "local", "#version 420 core\n#extension GL_ARB_arrays_of_arrays : enable"
        )
    )

    assert code.startswith("#version 420 core\n")
    assert "#extension GL_ARB_arrays_of_arrays : enable" in code


@pytest.mark.parametrize("name", sorted(ARRAY_OF_ARRAYS_PROGRAMS))
def test_webgl_refuses_array_of_arrays(name):
    with pytest.raises(WebGLLoweringRefused) as refusal:
        generate(program(name, "#version 300 es"), codegen=WebGLCodeGen)

    message = str(refusal.value)
    assert "an array of arrays requires GLSL ES 3.10" in message
    assert "project.translate.webgl-lowering-unsupported" in message


def test_matrix_and_vector_array_subscripts_keep_glsl_es_300(tmp_path):
    # ``m[1][1]`` and ``v[1][0]`` index a matrix and an array of vectors, which GLSL ES 3.00 has.
    source = FRAGMENT.format(
        version="#version 300 es",
        declarations="",
        body="mat2 m = mat2(x); vec2 v[2]; v[1] = vec2(x);"
        " output.color = vec4(m[1][1] + v[1][0]);",
    )

    code = generate(source)
    assert code.startswith("#version 300 es\n")
    compile_with_glslang(tmp_path, code)
    assert generate(source, codegen=WebGLCodeGen).startswith("#version 300 es\n")


@pytest.mark.parametrize(
    "line",
    [
        "float a[2][2];",
        "highp float a[2][2] = float[2][2](float[2](1.0, 2.0), float[2](3.0, 4.0));",
        "a = float[2][2](b, c);",
        "float[2][3] rows() {",
        "void f(in vec4 t[N][2]) {",
        "Cell cells[2][3];",
        "float data[][4];",
        "} blocks[2][2];",
    ],
)
def test_declares_array_of_arrays(line):
    assert declares_array_of_arrays(line, frozenset({"Cell"}))


@pytest.mark.parametrize(
    "line",
    [
        "a[1][1] = x;",
        "float y = m[1][1];",
        "return t[1][0];",
        "else a[1][1] = x;",
        "vec4 v[2] = vec4[2](a[0][1], b);",
        "color = vec4(q[1][1]);",
        "Other cells[2][3];",
    ],
)
def test_subscripts_are_not_arrays_of_arrays(line):
    assert not declares_array_of_arrays(line, frozenset({"Cell"}))


def test_requirement_found_in_rendered_program():
    code = "#version 300 es\nstruct Cell {\n    float w[2][3];\n};\nvoid main() {}\n"

    assert ARRAYS_OF_ARRAYS in generated_program_requirements(code, ("fragment",))
