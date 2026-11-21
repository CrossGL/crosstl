"""Rasterizer-ordered resources become a GL_ARB_fragment_shader_interlock critical section in OpenGL output.

The output used to declare the resources as plain storage buffers and images, so the accesses of overlapping
fragments ran in any order and the SPIR-V glslang compiled from it had no ``PixelInterlockOrderedEXT`` mode.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.glsl_version_requirements import (
    generated_program_requirements,
)

BUFFER = """shader Probe {
    RWStructuredBuffer<vec4> accum @rasterizer_ordered;
    struct FragmentInput {
        vec4 coord @ gl_FragCoord;
        vec2 uv;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            uint k = uint(input.coord.y) * 64u + uint(input.coord.x);
            accum[k] = accum[k] + vec4(input.uv, 0.0, 1.0);
            output.color = vec4(input.uv, 0.0, 1.0);
            return output;
        }
    }
}
"""

IMAGE = """shader Probe {
    image2D accum @rasterizer_ordered @rgba32f;
    struct FragmentInput {
        vec4 coord @ gl_FragCoord;
    }
    fragment {
        vec4 main(FragmentInput input) @location(0) {
            ivec2 p = ivec2(input.coord.xy);
            imageStore(accum, p, imageLoad(accum, p) + vec4(1.0));
            return imageLoad(accum, p);
        }
    }
}
"""

# Metal import writes ``[[raster_order_group(N)]]`` on a fragment argument as an ordered entry parameter.
ENTRY_BUFFER = """shader Probe {
    struct FragmentIn {
        vec4 coord @gl_Position;
        vec2 uv;
    }
    fragment {
        @ stage_entry
        vec4 probe(FragmentIn in_, RWStructuredBuffer<vec4> accum @buffer(0) @rasterizer_ordered) {
            uint k = uint(in_.coord.y) * 64u + uint(in_.coord.x);
            buffer_store(accum, k, buffer_load(accum, k) + vec4(in_.uv, 0.0, 1.0));
            return vec4(in_.uv, 0.0, 1.0);
        }
    }
}
"""

ENTRY_IMAGE = """shader Probe {
    struct FragmentIn {
        vec4 coord @gl_Position;
    }
    fragment {
        @ stage_entry
        vec4 probe(FragmentIn in_, image2D accum @rgba32f @rasterizer_ordered) {
            ivec2 p = ivec2(in_.coord.xy);
            imageStore(accum, p, imageLoad(accum, p) + vec4(1.0));
            return imageLoad(accum, p);
        }
    }
}
"""

HLSL = """RasterizerOrderedBuffer<float4> accum : register(u1);
RasterizerOrderedTexture2D<float4> img : register(u2);
float4 main(float4 pos : SV_Position) : SV_Target
{
    uint k = uint(pos.y) * 64u + uint(pos.x);
    accum[k] = accum[k] + float4(1, 0, 0, 1);
    img[uint2(pos.xy)] = img[uint2(pos.xy)] * 0.5;
    EARLY_RETURN
    return float4(1, 1, 1, 1);
}
"""

GLSL_MODE = """#version 460
#extension GL_ARB_fragment_shader_interlock : require
layout(sample_interlock_unordered) in;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(std430, binding = 0) buffer Accum { vec4 accum[]; };
void main() {
    beginInvocationInterlockARB();
    accum[0] += vec4(uv, 0.0, 1.0);
    endInvocationInterlockARB();
    color = vec4(uv, 0.0, 1.0);
}
"""

GLSL_ES = """#version 310 es
precision highp float;
#extension GL_ARB_fragment_shader_interlock : require
layout(location = 0) out vec4 color;
layout(std430, binding = 0) buffer Accum { vec4 accum[]; };
void main() {
    beginInvocationInterlockARB();
    accum[0] += vec4(1.0);
    endInvocationInterlockARB();
    color = vec4(1.0);
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.frag"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-S", "frag", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _main_body(glsl):
    return glsl.split("void main() {\n", 1)[1].split("\n}\n", 1)[0].splitlines()


def _assert_critical_section(glsl):
    assert "#extension GL_ARB_fragment_shader_interlock : require" in glsl
    assert "layout(pixel_interlock_ordered) in;" in glsl
    body = _main_body(glsl)
    assert body[0] == "    beginInvocationInterlockARB();"
    assert body[-2:] == ["    endInvocationInterlockARB();", "    return;"]
    assert glsl.count("beginInvocationInterlockARB()") == 1
    assert glsl.count("endInvocationInterlockARB()") == 1


@pytest.mark.parametrize(
    "source",
    [BUFFER, IMAGE, ENTRY_BUFFER, ENTRY_IMAGE],
    ids=["buffer", "image", "entry-buffer", "entry-image"],
)
def test_rasterizer_ordered_resource_opens_a_critical_section(tmp_path, source):
    glsl = _translate(tmp_path, "probe.cgl", source)
    _assert_critical_section(glsl)
    _glslang(tmp_path, glsl)


def test_image_read_in_the_return_value_stays_inside_the_section(tmp_path):
    body = _main_body(_translate(tmp_path, "probe.cgl", IMAGE))
    read = next(i for i, line in enumerate(body) if "= imageLoad(accum, p);" in line)
    assert read < body.index("    endInvocationInterlockARB();")


def test_hlsl_rasterizer_ordered_views_open_a_critical_section(tmp_path):
    glsl = _translate(tmp_path, "rov.hlsl", HLSL.replace("EARLY_RETURN", ""))
    _assert_critical_section(glsl)
    _glslang(tmp_path, glsl)


def test_early_return_from_an_ordered_fragment_is_refused(tmp_path):
    source = HLSL.replace(
        "EARLY_RETURN", "if (pos.x > 10.0)\n        return float4(0, 0, 0, 1);"
    )
    with pytest.raises(OpenGLLoweringRefused, match="returns early"):
        _translate(tmp_path, "rov.hlsl", source)


def test_declared_interlock_mode_and_calls_are_kept(tmp_path):
    glsl = _translate(tmp_path, "probe.frag", GLSL_MODE)
    assert "layout(sample_interlock_unordered) in;" in glsl
    assert "pixel_interlock_ordered" not in glsl
    assert glsl.count("beginInvocationInterlockARB()") == 1
    assert glsl.count("#extension GL_ARB_fragment_shader_interlock") == 1
    _glslang(tmp_path, glsl)


def test_glsl_es_has_no_interlock(tmp_path):
    with pytest.raises(OpenGLLoweringRefused, match="does not exist in GLSL ES"):
        _translate(tmp_path, "probe.frag", GLSL_ES)


def test_interlock_layout_needs_glsl_450():
    code = "layout(pixel_interlock_ordered) in;\nvoid main() {\n}\n"
    features = {
        requirement.feature: requirement.minimum(False)
        for requirement in generated_program_requirements(code, ("fragment",))
    }
    assert features["a fragment shader interlock layout"] == 450


@pytest.mark.parametrize("source", [BUFFER, ENTRY_BUFFER], ids=["global", "entry"])
def test_stage_without_ordered_resources_is_unchanged(tmp_path, source):
    source = source.replace(" @rasterizer_ordered", "")
    glsl = _translate(tmp_path, "probe.cgl", source)
    assert not re.search(r"[Ii]nterlock", glsl)
