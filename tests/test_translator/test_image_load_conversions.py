"""Integer image loads convert implicitly to float and unsigned contexts, as GLSL 4.60 section 4.1.10 does."""

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """#version 450
layout(binding = 3, rg8ui) uniform readonly uimage2D frequency;
layout(binding = 4, r32i) uniform readonly iimage2D counts;
layout(location = 0) out vec4 outColor;
void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uvec4 total = imageLoad(counts, coord);
    outColor = imageLoad(frequency, coord) + vec4(total);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "shader.frag"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("backend", "pattern"),
    [
        ("metal", r"outColor = float4\(frequency\.read\("),
        ("directx", r"outColor = \(?__crossgl_uint_to_float4\("),
        ("opengl", r"outColor = \(?imageLoad\(frequency, coord\)"),
    ],
)
def test_unsigned_image_load_assigns_to_a_float_vector(tmp_path, backend, pattern):
    # The load used to be refused: "requires uint result context for rg8ui images: expected float".
    code = _translate(tmp_path, backend)
    assert re.search(pattern, code), code


def test_metal_output_compiles(tmp_path):
    code = _translate(tmp_path, "metal")
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def test_float_load_into_an_integer_context_is_still_refused(tmp_path):
    source = """shader S {
    image2D target @r32f;
    fragment {
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            uint bits = imageLoad(target, ivec2(0));
            return vec4(float(bits));
        }
    }
}
"""
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    with pytest.raises(Exception, match="requires float result context"):
        crosstl.translate(str(path), backend="metal", format_output=False)
