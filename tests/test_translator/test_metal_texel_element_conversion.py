"""Metal ``write`` receives a texel of the texture's own element type.

GLSL converts an ``f16vec4`` ``imageStore`` argument to the image's ``vec4``
(GL_EXT_shader_explicit_arithmetic_types_float16 adds the implicit float16 to
float conversion), but ``texture2d<float>::write`` takes ``vec<float, 4>`` and
MSL converts between vector types only explicitly.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f, binding = 0) writeonly uniform image2D color;
layout(r32i, binding = 1) writeonly uniform iimage2D index_image;
layout(set = 0, binding = 2) readonly buffer Texels { f16vec4 texels[]; };

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    f16vec4 texel = texels[p.x] * float16_t(0.5);
    imageStore(color, p, texel);
    imageStore(index_image, p, ivec4(p, 0, 1));
}
"""


@pytest.fixture(scope="module")
def metal(tmp_path_factory):
    path = tmp_path_factory.mktemp("texel") / "store.comp"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_half_texel_is_converted_to_the_texture_element_type(metal):
    store = re.search(r"\bcolor\.write\((.*)\);", metal)
    assert store, metal
    assert store.group(1).startswith("float4(texel"), store.group(0)


def test_texel_of_the_texture_element_type_is_written_as_is(metal):
    store = re.search(r"\bindex_image\.write\((.*)\);", metal)
    assert store, metal
    assert not store.group(1).startswith("int4(int4("), store.group(0)


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="xcrun is not on PATH")
def test_converted_store_compiles(metal, tmp_path):
    path = tmp_path / "store.metal"
    path.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(path) + ".air"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]


HALF_TEXTURE_KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void fill(texture2d<half, access::write> target [[texture(0)]],
                 uint2 gid [[thread_position_in_grid]]) {
    half4 texel = half4(half(0.25), half(0.5), half(0.75), half(1.0));
    target.write(texel, gid);
}
"""


def test_half_texture_keeps_its_half_texel(tmp_path):
    # ``texture2d<half>::write`` takes ``half4``; converting the texel to the
    # format's ``float4`` would select no ``write`` overload.
    path = tmp_path / "fill.metal"
    path.write_text(HALF_TEXTURE_KERNEL, encoding="utf-8")
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    store = re.search(r"\btarget\.write\((.*)\);", metal)
    assert store, metal
    assert not store.group(1).startswith("float4("), store.group(0)
