"""Keep every channel of unformatted storage image loads in Metal stores."""

import re

import pytest

from crosstl import translate
from tests.test_translator.test_metal_builtin_ownership import _compile

HLSL = """
RWTexture2D<float4> source : register(u0);
RWTexture2D<float4> copy : register(u1);
RWTexture2D<float4> scaled : register(u2);
RWTexture2D<int4> signedCopy : register(u3);
RWTexture2D<int4> signedSource : register(u4);
RWTexture2D<float> single : register(u5);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    copy[id.xy] = source[id.xy];
    scaled[id.xy] = scaled[id.xy] * 2.0;
    signedCopy[id.xy] = signedSource[id.xy];
    single[id.xy] = single[id.yx];
}
"""

CROSSGL = """
shader main {
    image2D source;
    image2D copy;
    compute {
        @ numthreads(8, 8, 1)
        void copyImage(uvec3 id @ gl_GlobalInvocationID) @ stage_entry {
            imageStore(copy, ivec2(id.xy), imageLoad(source, ivec2(id.xy)) + vec4(1.0));
        }
    }
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"images.{suffix}"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend="metal", format_output=False)


def _writes(generated):
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"\b(\w+)\.write\((.*), uint2\(", generated)
    }


def test_hlsl_texture_copies_store_all_channels(tmp_path):
    generated = _translate(tmp_path, HLSL, "hlsl")
    writes = _writes(generated)

    assert writes["copy"] == "source.read(uint2(id.xy))"
    assert writes["scaled"] == "scaled.read(uint2(id.xy)) * 2.0"
    assert writes["signedCopy"] == "signedSource.read(uint2(id.xy))"
    # Scalar textures still select their single channel explicitly.
    assert writes["single"] == "float4(single.read(uint2(id.yx)).x, 0.0, 0.0, 0.0)"
    _compile(generated, "metal", tmp_path)


def test_crossgl_unformatted_image_load_is_a_four_channel_value(tmp_path):
    generated = _translate(tmp_path, CROSSGL, "cgl")

    assert _writes(generated)["copy"] == (
        "source.read(uint2(int2(id.xy))) + float4(1.0)"
    ), generated
    _compile(generated, "metal", tmp_path)


@pytest.mark.parametrize("load", ["source", "scaled"])
def test_no_channel_broadcast_from_unformatted_loads(tmp_path, load):
    generated = _translate(tmp_path, HLSL, "hlsl")

    assert f"float4({load}.read(uint2(id.xy)).x" not in generated
