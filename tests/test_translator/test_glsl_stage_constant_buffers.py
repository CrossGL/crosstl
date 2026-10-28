"""A ``ConstantBuffer<T>`` declared in a stage is a uniform block of that stage in GLSL output."""

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """shader S {
    struct P {
        vec4 v;
    };
    fragment {
        ConstantBuffer<P> params @binding(1);
        ConstantBuffer<P> lights[2] @binding(2);
        vec4 main() @ gl_FragColor {
            return params.v + lights[1].v;
        }
    }
}
"""


def _glsl(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def test_a_stage_constant_buffer_declares_a_uniform_block(tmp_path):
    code = " ".join(_glsl(tmp_path).split())
    assert "layout(std140, binding = 1) uniform P { vec4 v; } params;" in code
    assert "layout(std140, binding = 2) uniform P_lights { vec4 v; } lights[2];" in code
    assert "(params.v + lights[1].v)" in code


def test_the_uniform_blocks_compile(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "shader.frag"
    shader.write_text(_glsl(tmp_path))
    result = subprocess.run(
        [glslang, "-S", "frag", str(shader)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
