"""The target chooses the GLSL profile: ``opengl`` writes desktop GLSL, ``gles`` GLSL ES 3.x, ``webgl`` GLSL ES 3.00.

A CrossGL ``#version`` line, which a GLSL ES source (WebGL 2 or OpenGL ES) passes on, chooses the version output
starts from; a line of the other profile maps to the first version of the output's profile with its constructs
(:mod:`crosstl.translator.codegen.glsl_source_directives`). Each profile leaves out the other's ``#extension`` lines,
starting from the version that has the extension's constructs, or refuses an extension it has no form of.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen import GLSLCodeGen, get_codegen
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.glsl_source_directives import (
    DESKTOP_EXTENSIONS_IN_ES_CORE,
    ES_EXTENSIONS_IN_DESKTOP_CORE,
    ES_EXTENSIONS_WITHOUT_DESKTOP_FORM,
)
from crosstl.translator.refusals import OpenGLRefusal

GLSLANG = shutil.which("glslangValidator")
needs_glslang = pytest.mark.skipif(
    GLSLANG is None, reason="glslangValidator is not installed"
)

FRAGMENT_300 = """#version 300 es
precision highp float;
struct S { float a; vec2 b; };
void bump(inout S s) { s.a += 1.0; s.b *= 2.0; }
uniform vec4 tint;
out vec4 fragColor;
void main() {
    S s = S(tint.x, tint.yz);
    bump(s);
    fragColor = vec4(s.a, s.b, 1.0);
}
"""

PACKING_300 = """#version 300 es
precision highp float;
uniform vec4 tint;
out vec4 fragColor;
void main() {
    uint p = packHalf2x16(tint.xy);
    fragColor = vec4(float(p & 1u));
}
"""

COMPUTE_310 = """#version 310 es
layout(local_size_x = 8) in;
layout(std430, binding = 0) buffer Data { uint v[]; } data;
void main() {
    uint i = gl_GlobalInvocationID.x;
    data.v[i] = data.v[i] * 2u + uint(mix(1, 3, i > 4u));
}
"""

SAMPLE_VARIABLES_300 = """#version 300 es
#extension GL_OES_sample_variables : require
precision highp float;
out vec4 fragColor;
void main() { fragColor = vec4(float(gl_SampleID)); }
"""

EXTERNAL_IMAGE_300 = """#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES image;
out vec4 fragColor;
void main() { fragColor = texture(image, vec2(0.5)); }
"""


def _crossgl_program(header):
    return header + """
shader main {
    fragment {
        vec4 main() @location(0) {
            return vec4(1.0);
        }
    }
}
"""


def _translate(tmp_path, name, text, backend):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang(tmp_path, name, text):
    path = tmp_path / f"out-{name}"
    path.write_text(text, encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_extension_tables_do_not_overlap():
    assert not set(ES_EXTENSIONS_IN_DESKTOP_CORE) & set(
        ES_EXTENSIONS_WITHOUT_DESKTOP_FORM
    )
    assert all(name.startswith("GL_ARB_") for name in DESKTOP_EXTENSIONS_IN_ES_CORE)


@pytest.mark.parametrize("name", ["gles", "opengles", "opengl-es", "GLES"])
def test_the_gles_targets_select_glsl_es_output(name):
    generator = get_codegen(name)
    assert type(generator) is GLSLCodeGen
    assert generator.glsl_es_target
    assert not get_codegen("opengl").glsl_es_target


def test_an_unknown_glsl_target_profile_is_refused():
    with pytest.raises(OpenGLRefusal, match="unknown GLSL target profile"):
        GLSLCodeGen(target_profile="gles4")


@pytest.mark.parametrize(
    ("name", "source", "version"),
    [
        ("shader.frag", FRAGMENT_300, "#version 330 core"),
        ("pack.frag", PACKING_300, "#version 420 core"),
        ("shader.comp", COMPUTE_310, "#version 450 core"),
        ("sample.frag", SAMPLE_VARIABLES_300, "#version 400 core"),
    ],
    ids=["es300", "es300-packing", "es310-compute", "es300-sample-variables"],
)
def test_glsl_output_of_a_glsl_es_source_is_desktop_glsl(
    tmp_path, name, source, version
):
    translated = _translate(tmp_path, name, source, "opengl")
    assert translated.splitlines()[0] == version
    assert "#extension GL_OES_" not in translated
    if GLSLANG is not None:
        _glslang(tmp_path, name, translated)


@pytest.mark.parametrize(
    ("name", "source", "version", "extension"),
    [
        ("shader.frag", FRAGMENT_300, "#version 300 es", None),
        ("shader.comp", COMPUTE_310, "#version 310 es", None),
        (
            "external.frag",
            EXTERNAL_IMAGE_300,
            "#version 300 es",
            "#extension GL_OES_EGL_image_external_essl3 : require",
        ),
    ],
    ids=["es300", "es310-compute", "es300-external-image"],
)
def test_gles_output_of_a_glsl_es_source_keeps_its_version(
    tmp_path, name, source, version, extension
):
    translated = _translate(tmp_path, name, source, "gles")
    lines = translated.splitlines()
    assert lines[0] == version
    if extension is not None:
        assert extension in lines
    if GLSLANG is not None:
        _glslang(tmp_path, name, translated)


@needs_glslang
def test_webgl_output_of_a_glsl_es_source_stays_glsl_es_300(tmp_path):
    translated = _translate(tmp_path, "shader.frag", FRAGMENT_300, "webgl")
    assert translated.startswith("#version 300 es\n")
    _glslang(tmp_path, "webgl.frag", translated)


def test_a_glsl_es_extension_desktop_glsl_has_no_form_of_is_refused(tmp_path):
    with pytest.raises(OpenGLLoweringRefused, match="GL_OES_EGL_image_external_essl3"):
        _translate(tmp_path, "external.frag", EXTERNAL_IMAGE_300, "opengl")


@pytest.mark.parametrize(
    ("header", "version"),
    [
        ("#version 300 es\n", "#version 330 core"),
        ("#version 310 es\n", "#version 450 core"),
        ("#version 320 es\n", "#version 450 core"),
        ("#version 100\n", "#version 330 core"),
        (
            "#version 330 core\n#extension GL_EXT_gpu_shader5 : enable\n",
            "#version 400 core",
        ),
        (
            "#version 330 core\n#extension GL_EXT_clip_cull_distance : require\n",
            "#version 450 core",
        ),
        (
            "#version 430 core\n#extension GL_EXT_texture_buffer : require\n",
            "#version 430 core",
        ),
        (
            "#version 330 core\n#extension GL_EXT_gpu_shader5 : disable\n",
            "#version 330 core",
        ),
        (
            "#version 330 core\n#extension GL_EXT_shader_framebuffer_fetch : disable\n",
            "#version 330 core",
        ),
    ],
    ids=[
        "es300",
        "es310",
        "es320",
        "es100",
        "gpu-shader5",
        "cull-distance",
        "texture-buffer",
        "disabled",
        "disabled-es-only",
    ],
)
def test_desktop_output_maps_a_glsl_es_version_and_its_extensions(
    tmp_path, header, version
):
    translated = _translate(tmp_path, "d.cgl", _crossgl_program(header), "opengl")
    lines = translated.splitlines()
    assert lines[0] == version
    assert not any(line.startswith("#extension GL_EXT_") for line in lines)
    if GLSLANG is not None:
        _glslang(tmp_path, "d.frag", translated)


def test_desktop_output_refuses_a_crossgl_glsl_es_only_extension(tmp_path):
    header = "#version 450 core\n#extension GL_EXT_shader_framebuffer_fetch : require\n"
    with pytest.raises(OpenGLLoweringRefused, match="GL_EXT_shader_framebuffer_fetch"):
        _translate(tmp_path, "f.cgl", _crossgl_program(header), "opengl")


@pytest.mark.parametrize(
    ("header", "version"),
    [
        ("", "#version 300 es"),
        ("#version 450 core\n", "#version 300 es"),
        ("#version 320 es\n", "#version 320 es"),
        (
            "#version 450 core\n#extension GL_ARB_separate_shader_objects : enable\n",
            "#version 300 es",
        ),
        (
            "#version 450 core\n#extension GL_ARB_shader_clock : disable\n",
            "#version 300 es",
        ),
    ],
    ids=["no-version", "desktop", "es320", "in-es-core", "disabled"],
)
def test_gles_output_starts_from_glsl_es_300_and_leaves_out_arb_extensions(
    tmp_path, header, version
):
    translated = _translate(tmp_path, "e.cgl", _crossgl_program(header), "gles")
    lines = translated.splitlines()
    assert lines[0] == version
    assert not any(line.startswith("#extension GL_ARB_") for line in lines)
    if GLSLANG is not None:
        _glslang(tmp_path, "e.frag", translated)


@pytest.mark.parametrize("backend", ["gles", "webgl"])
def test_glsl_es_output_refuses_a_desktop_only_extension(tmp_path, backend):
    header = "#version 450 core\n#extension GL_ARB_shader_clock : require\n"
    with pytest.raises(Exception, match="GL_ARB_shader_clock"):
        _translate(tmp_path, "c.cgl", _crossgl_program(header), backend)
