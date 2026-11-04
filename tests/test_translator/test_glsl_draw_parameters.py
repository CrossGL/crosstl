"""``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_DrawID`` for OpenGL and OpenGL ES targets."""

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """shader Probe {
    struct VertexInput {
        vec4 position;
        int base @ gl_BaseVertex;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0 + vec4(float(input.base), 0.0, 0.0, 0.0);
            return output;
        }
    }
}
"""


def translate(tmp_path, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate_glsl(tmp_path, glsl):
    if shutil.which("glslangValidator"):
        (tmp_path / "out.vert").write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "vert", "out.vert"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_requires_the_draw_parameters_extension(tmp_path):
    glsl = translate(tmp_path, "opengl")
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert "gl_BaseVertexARB" in glsl
    _validate_glsl(tmp_path, glsl)


def test_glsl_es_has_no_base_vertex(tmp_path):
    with pytest.raises(Exception, match="base vertex"):
        translate(tmp_path, "webgl")
