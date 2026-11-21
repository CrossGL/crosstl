"""GLSL output gives an unlocated varying a location that only a SPIR-V compilation reads.

OpenGL links varyings by name, so GLSL output leaves a varying without a fixed location unlocated. glslang ``-V``
(Vulkan) and ``-G`` (GL_ARB_gl_spirv) reject such a declaration with "SPIR-V requires location for user
input/output". The varying now takes ``CROSSGL_LOCATION(n)``, which expands to ``layout(location = n)`` only when
``GL_SPIRV`` or ``VULKAN`` is defined, with the location SPIR-V output gives the same value. See
:mod:`crosstl.translator.codegen.glsl_spirv_varying_locations`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_spirv_varying_locations import (
    spirv_location_macro_definition,
    strip_spirv_location_macro,
)

VERTEX_FRAGMENT = """shader Probe {
    struct VSIn { vec3 pos; vec2 uv; }
    struct VSOut { vec4 position @gl_Position; vec2 uv; }
    struct FSOut { vec4 color; }
    vertex {
        VSOut main(VSIn input) {
            VSOut o;
            o.position = vec4(input.pos, 1.0);
            o.uv = input.uv;
            return o;
        }
    }
    fragment {
        FSOut main(VSOut input) {
            FSOut o;
            o.color = vec4(input.uv, 0.0, 1.0);
            return o;
        }
    }
}
"""

LOOSE_INPUTS = """shader Probe {
    struct VSOut { vec4 position @gl_Position; vec2 uv; vec3 n @ Normal; float w; }
    fragment {
        vec4 main(VSOut input, float extra, vec2 more) @gl_FragColor {
            return vec4(input.uv, input.w + extra, more.x);
        }
    }
}
"""

HLSL_LOCATED = """struct PSIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; float3 n : NORMAL; };
float4 main(PSIn i) : SV_Target { return float4(i.uv, i.n.x, 1.0); }
"""

SUBPASS_INPUT = """shader Probe {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        buffer vec4* outp;
        layout(input_attachment_index = 0) uniform subpassInput u0;
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            outp[0] = subpassLoad(u0);
            output.color = vec4(input.uv, 0.0, 1.0);
            return output;
        }
    }
}
"""


GLSL_330 = """#version 330 core
in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(uv, 0.0, 1.0);
}
"""

GLSL_150 = """#version 150
in vec2 uv;
out vec4 color;
void main() {
    color = vec4(uv, 0.0, 1.0);
}
"""


def _glslang():
    glslang = shutil.which("glslangValidator")
    if not glslang:
        pytest.skip("glslangValidator is required")
    return glslang


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _stages(source):
    ast = crosstl.translator.parse(source)
    return (
        GLSLCodeGen().generate_stage(ast, "vertex"),
        GLSLCodeGen().generate_stage(ast, "fragment"),
    )


def _write(tmp_path, **stages):
    paths = []
    for stage, glsl in stages.items():
        path = tmp_path / f"out.{stage}"
        path.write_text(glsl)
        paths.append(str(path))
    return paths


def _spirv_locations(tmp_path, glsl, stage):
    """``{name: location}`` of the user varyings in a Vulkan compilation of ``glsl``."""
    glslang = _glslang()
    if not shutil.which("spirv-dis"):
        pytest.skip("spirv-dis is required")
    (source,) = _write(tmp_path, **{stage: glsl})
    _run([glslang, "-V", source, "-o", str(tmp_path / "out.spv")])
    assembly = _run(["spirv-dis", str(tmp_path / "out.spv")])
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', assembly))
    return {
        names.get(target, target): int(location)
        for target, location in re.findall(
            r"OpDecorate (%\w+) Location (\d+)", assembly
        )
    }


def test_vertex_output_and_fragment_input_share_a_spirv_location(tmp_path):
    vertex, fragment = _stages(VERTEX_FRAGMENT)
    assert spirv_location_macro_definition() in vertex
    assert spirv_location_macro_definition() in fragment
    assert "CROSSGL_LOCATION(0) out vec2 out_uv;" in vertex
    assert "CROSSGL_LOCATION(0) in vec2 out_uv;" in fragment
    # Vertex inputs and fragment outputs keep the locations they already had.
    assert "layout(location = 1) in vec2 uv;" in vertex
    assert "layout(location = 0) out vec4 color;" in fragment

    glslang = _glslang()
    paths = _write(tmp_path, vert=vertex, frag=fragment)
    # OpenGL links the two stages by name, Vulkan and GL_ARB_gl_spirv by location.
    _run([glslang, "-l", *paths])
    _run([glslang, "-G", "-l", *paths, "-o", str(tmp_path / "gl.spv")])
    _run([glslang, "-V", "-l", *paths, "-o", str(tmp_path / "vk.spv")])


def test_opengl_compilation_reads_the_unlocated_declaration():
    vertex, _fragment = _stages(VERTEX_FRAGMENT)
    assert strip_spirv_location_macro("CROSSGL_LOCATION(0) out vec2 out_uv;") == (
        "out vec2 out_uv;"
    )
    # A varying location needs GLSL 4.10 or GL_ARB_separate_shader_objects, but only the SPIR-V branch has it.
    assert vertex.startswith("#version 330 core\n")
    assert "#if defined(GL_SPIRV) || defined(VULKAN)\n" in vertex
    assert "#define CROSSGL_LOCATION(n)\n#endif\n" in vertex


def test_loose_inputs_take_the_locations_after_their_structure(tmp_path):
    glsl = crosstl.translate(
        str(_source(tmp_path, LOOSE_INPUTS, "probe.cgl")),
        backend="opengl",
        format_output=False,
    )
    assert "CROSSGL_LOCATION(0) in vec2 uv;" in glsl
    assert "layout(location = 1) in vec3 n;" in glsl
    assert "CROSSGL_LOCATION(2) in float w;" in glsl
    assert "CROSSGL_LOCATION(3) in float extra;" in glsl
    assert "CROSSGL_LOCATION(4) in vec2 more;" in glsl
    assert _spirv_locations(tmp_path, glsl, "frag") == {
        "uv": 0,
        "n": 1,
        "w": 2,
        "extra": 3,
        "more": 4,
        "fragColor": 0,
    }
    # The SPIR-V output of the same source places them alike.
    spirv = crosstl.translate(str(tmp_path / "probe.cgl"), backend="vulkan")
    for name, location in (("uv", 0), ("n", 1), ("w", 2), ("extra", 3), ("more", 4)):
        assert re.search(rf"OpDecorate %\w*_{name} Location {location}\n", spirv), name


def test_located_varyings_need_no_macro(tmp_path):
    glsl = crosstl.translate(
        str(_source(tmp_path, HLSL_LOCATED, "probe.hlsl")),
        backend="opengl",
        format_output=False,
    )
    assert "CROSSGL_LOCATION" not in glsl
    assert re.search(r"layout\(location = \d+\) in vec2 uv;", glsl)


def test_webgl_output_has_no_varying_locations(tmp_path):
    glsl = crosstl.translate(
        str(_source(tmp_path, VERTEX_FRAGMENT, "probe.cgl")),
        backend="webgl",
        format_output=False,
    )
    assert "CROSSGL_LOCATION" not in glsl
    assert "GL_SPIRV" not in glsl


def test_subpass_input_program_compiles_for_opengl_and_vulkan(tmp_path):
    glsl = crosstl.translate(
        str(_source(tmp_path, SUBPASS_INPUT, "probe.cgl")),
        backend="opengl",
        format_output=False,
    )
    assert "CROSSGL_LOCATION(0) in vec2 uv;" in glsl
    glslang = _glslang()
    (path,) = _write(tmp_path, frag=glsl)
    _run([glslang, path])
    assert _spirv_locations(tmp_path, glsl, "frag")["uv"] == 0


def test_glsl_330_program_keeps_its_version_and_compiles_for_spirv(tmp_path):
    glsl = crosstl.translate(
        str(_source(tmp_path, GLSL_330, "probe.frag")),
        backend="opengl",
        format_output=False,
    )
    # Locations of varyings are core in 4.10; the SPIR-V branch enables them with GL_ARB_separate_shader_objects.
    assert glsl.startswith("#version 330 core\n")
    assert "CROSSGL_LOCATION(0) in vec2 uv;" in glsl
    glslang = _glslang()
    (path,) = _write(tmp_path, frag=glsl)
    _run([glslang, path])
    assert _spirv_locations(tmp_path, glsl, "frag")["uv"] == 0


def test_layouts_still_raise_a_glsl_150_program(tmp_path):
    glsl = crosstl.translate(
        str(_source(tmp_path, GLSL_150, "probe.frag")),
        backend="opengl",
        format_output=False,
    )
    # The fragment output location needs 3.30 in OpenGL, whatever the SPIR-V branch enables.
    assert glsl.startswith("#version 330 core\n")
    glslang = _glslang()
    (path,) = _write(tmp_path, frag=glsl)
    _run([glslang, path])


def _source(tmp_path, text, name):
    path = tmp_path / name
    path.write_text(text)
    return path
