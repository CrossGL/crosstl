"""WebGL 2 input: GLSL ES 3.00 source read through the GLSL frontend.

GLSL ES 3.00 is the shading language of WebGL 2. Its shaders start with ``#version 300 es``, which the GLSL
preprocessor reads as the ES profile (it defines ``GL_ES``), so the source names ``webgl``, ``webgl2``, ``essl`` and
``glsl-es`` select the GLSL frontend. ``.webgl.glsl`` is the extension CrossTL writes WebGL output under, so a file
with it is read as source only when the source backend is named.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import codegen
from crosstl.translator.source_registry import (
    SOURCE_REGISTRY,
    WEBGL_SOURCE_UNSUPPORTED_MESSAGE,
    register_default_sources,
)

GLSLANG = shutil.which("glslangValidator")

VERTEX = """#version 300 es
precision highp float;
layout(location = 0) in vec3 position;
layout(location = 1) in mediump vec2 uv;
uniform Camera {
    mat4 viewProjection;
    vec4 tint;
} camera;
out mediump vec2 vUv;
flat out int vInstance;
void main() {
    vUv = uv;
    vInstance = gl_InstanceID;
    gl_PointSize = 1.0;
    gl_Position = camera.viewProjection * vec4(position, 1.0);
}
"""

FRAGMENT = """#version 300 es
precision mediump float;
precision lowp sampler2DArray;
in mediump vec2 vUv;
flat in int vInstance;
uniform sampler2DArray layers;
uniform highp sampler2DShadow shadow;
layout(location = 0) out vec4 color;
layout(location = 1) out highp vec4 extra;
void main() {
    lowp vec4 c = texture(layers, vec3(vUv, float(vInstance)));
    float s = texture(shadow, vec3(vUv, 0.5));
    if (c.a < 0.1) discard;
    color = c * s;
    extra = vec4(gl_FragCoord.xy, gl_FrontFacing ? 1.0 : 0.0, 1.0);
}
"""

WEBGL_SOURCE_NAMES = ("webgl", "webgl2", "essl", "glsl-es", "WebGL")


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _translate(path, backend, source_backend="webgl"):
    return crosstl.translate(
        str(path), backend=backend, source_backend=source_backend, format_output=False
    )


def _glslang(tmp_path, name, text):
    path = _write(tmp_path, name, text)
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", WEBGL_SOURCE_NAMES)
def test_webgl_source_names_select_the_glsl_frontend(name):
    register_default_sources()
    assert SOURCE_REGISTRY.resolve_name(name) == "opengl"


def test_webgl_backend_names_the_glsl_frontend_as_its_source():
    spec = codegen.get_backend("webgl")
    assert spec.source_registry_name == "opengl"
    assert "webgl" in codegen.source_backend_names()


@pytest.mark.parametrize("name", WEBGL_SOURCE_NAMES)
@pytest.mark.parametrize("backend", ["cgl", "metal", "directx"])
def test_webgl_source_imports_as_glsl_does(tmp_path, name, backend):
    path = _write(tmp_path, "shader.frag", FRAGMENT)
    assert _translate(path, backend, name) == _translate(path, backend, "opengl")


def test_webgl_source_keeps_the_es_version_in_crossgl(tmp_path):
    crossgl = _translate(_write(tmp_path, "shader.frag", FRAGMENT), "cgl")
    assert "#version 300 es" in crossgl.splitlines()


def test_a_webgl_glsl_file_needs_the_source_backend_named(tmp_path):
    path = _write(tmp_path, "shader.webgl.glsl", FRAGMENT)
    with pytest.raises(Exception, match="source_backend='webgl'") as error:
        crosstl.translate(str(path), backend="metal")
    assert str(error.value) == WEBGL_SOURCE_UNSUPPORTED_MESSAGE
    named = _translate(path, "metal")
    assert named == _translate(_write(tmp_path, "shader.frag", FRAGMENT), "metal")


@pytest.mark.parametrize("version", ["300 es", "450"])
def test_a_uniform_block_without_layout_keeps_its_block_name(tmp_path, version):
    # The host finds a uniform block by its block name (glGetUniformBlockIndex); the instance name is local to the
    # shader. GLSL ES 3.00 has no binding qualifier, so its blocks usually have no layout qualifier at all.
    source = VERTEX.replace("#version 300 es", f"#version {version}")
    path = _write(tmp_path, "shader.vert", source)
    crossgl = _translate(path, "cgl", "opengl")
    assert "cbuffer Camera @buffer_layout(std140) {" in crossgl
    assert "cbuffer camera" not in crossgl
    webgl = _translate(path, "webgl", "opengl")
    assert "uniform Camera {" in webgl


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
@pytest.mark.parametrize(
    ("name", "source"),
    [("shader.vert", VERTEX), ("shader.frag", FRAGMENT)],
    ids=["vertex", "fragment"],
)
def test_webgl_round_trip_is_glsl_es_300_that_glslang_accepts(tmp_path, name, source):
    _glslang(tmp_path, "native-" + name, source)
    translated = _translate(_write(tmp_path, name, source), "webgl")
    assert translated.startswith("#version 300 es\n")
    _glslang(tmp_path, "translated-" + name, translated)


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
@pytest.mark.parametrize(
    ("name", "source"),
    [("shader.vert", VERTEX), ("shader.frag", FRAGMENT)],
    ids=["vertex", "fragment"],
)
def test_webgl_source_to_glsl_output_is_accepted(tmp_path, name, source):
    translated = _translate(_write(tmp_path, name, source), "opengl")
    _glslang(tmp_path, "translated-" + name, translated)
