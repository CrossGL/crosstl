"""The draw parameters ``gl_BaseVertex``, ``gl_BaseInstance`` and ``gl_DrawID`` in OpenGL GLSL output.

They are core from GLSL 4.60 (section 7.1.1); before it GL_ARB_shader_draw_parameters declares them with an ``ARB``
suffix. OpenGL output used to copy the 4.60 names into ``#version 450 core``, which glslang rejects as undeclared.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

STRUCT_MEMBERS = """shader Probe {
    struct VertexInput {
        vec4 position;
        int base @ gl_BaseVertex;
        int first @ gl_BaseInstance;
        int draw @ gl_DrawID;
        int instance @ gl_InstanceIndex;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position + vec4(float(input.base), float(input.first),
                float(input.draw), float(input.instance));
            return output;
        }
    }
}
"""

PARAMETERS = """shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(vec4 position @ POSITION, uint base @ gl_BaseVertex,
                          int first @ gl_BaseInstance, uint instance @ gl_InstanceIndex) {
            VertexOutput output;
            output.position = position + vec4(float(base), float(first), float(instance), 0.0);
            return output;
        }
    }
}
"""

IDENTIFIERS = """shader Probe {
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(vec4 position @ POSITION) {
            VertexOutput output;
            output.position = position + vec4(float(gl_BaseVertex), float(gl_BaseInstance),
                float(gl_DrawID), 0.0);
            return output;
        }
    }
}
"""

GLSL_460 = """#version 460
layout(location = 0) in vec4 position;
void main() {
    gl_Position = position + vec4(float(gl_BaseVertex), float(gl_BaseInstance), float(gl_DrawID), 0.0);
}
"""

_SUFFIXED = re.compile(r"\bgl_(?:BaseVertex|BaseInstance|DrawID)ARB\b")
_CORE = re.compile(r"\bgl_(?:BaseVertex|BaseInstance|DrawID)\b")


def _translate(tmp_path, source, backend="opengl", suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(tmp_path, glsl):
    if not shutil.which("glslangValidator"):
        return
    (tmp_path / "out.vert").write_text(glsl)
    result = subprocess.run(
        ["glslangValidator", "-S", "vert", "out.vert"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_struct_member_semantics_read_the_extension_names_before_glsl_460(tmp_path):
    glsl = _translate(tmp_path, STRUCT_MEMBERS)
    assert glsl.startswith("#version 450 core\n")
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert sorted(set(_SUFFIXED.findall(glsl))) == [
        "gl_BaseInstanceARB",
        "gl_BaseVertexARB",
        "gl_DrawIDARB",
    ]
    assert not _CORE.search(glsl)
    # gl_InstanceIndex starts at the draw's first instance; OpenGL's gl_InstanceID starts at 0.
    assert "(gl_InstanceID + gl_BaseInstanceARB)" in glsl
    assert "gl_InstanceIndex" not in glsl
    _validate(tmp_path, glsl)


def test_parameter_semantics_convert_the_extension_names(tmp_path):
    glsl = _translate(tmp_path, PARAMETERS)
    assert "uint base = uint(gl_BaseVertexARB);" in glsl
    assert "uint instance = uint((gl_InstanceID + gl_BaseInstanceARB));" in glsl
    assert "float(gl_BaseInstanceARB)" in glsl
    assert not _CORE.search(glsl) and "gl_InstanceIndex" not in glsl
    _validate(tmp_path, glsl)


def test_builtin_identifiers_read_the_extension_names(tmp_path):
    glsl = _translate(tmp_path, IDENTIFIERS)
    assert "#extension GL_ARB_shader_draw_parameters : require" in glsl
    assert "float(gl_BaseVertexARB)" in glsl
    assert "float(gl_BaseInstanceARB)" in glsl
    assert "float(gl_DrawIDARB)" in glsl
    assert not _CORE.search(glsl)
    _validate(tmp_path, glsl)


def test_glsl_460_keeps_the_core_names(tmp_path):
    glsl = _translate(tmp_path, GLSL_460, suffix=".vert")
    assert glsl.startswith("#version 460")
    assert "GL_ARB_shader_draw_parameters" not in glsl
    assert not _SUFFIXED.search(glsl)
    assert sorted(set(_CORE.findall(glsl))) == [
        "gl_BaseInstance",
        "gl_BaseVertex",
        "gl_DrawID",
    ]
    _validate(tmp_path, glsl)


def test_glsl_es_refuses_the_draw_parameters(tmp_path):
    with pytest.raises(Exception) as raised:
        _translate(tmp_path, IDENTIFIERS, backend="webgl")
    message = str(raised.value)
    assert "GLSL ES 3.00 has no base vertex" in message
    assert "GLSL ES 3.00 has no base instance" in message
    assert "GLSL ES 3.00 has no draw index" in message
