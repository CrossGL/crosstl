"""HLSL and Slang storage images keep their texel types through CrossGL.

A typed UAV such as ``RWTexture2D<float2>`` declares its texel type while the
view bound at run time decides the format. The importers write
``@format(unknown)`` and, for a texel type other than a four-component one,
``@texel(T)`` or ``@texel(unorm|snorm, T)``. DirectX and Slang output declare
that texel type again, widen its loads to CrossGL's four components and narrow
the stores. DXC's ``[[vk::image_format(F)]]`` and Slang's ``[format(F)]``
import as ``@format(F)``.

HLSL and Slang pass a resource only to a parameter of its type, so a storage
image parameter takes the type of the globals passed to it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from crosstl.backend.import_refusals import DirectXImportRefused, SlangImportRefused
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from tests.test_translator.test_spirv_separate_textures import _translate, _validate
from tests.test_translator.test_texel_buffers import (
    _compile_compute_hlsl,
    _validate_compute_glsl,
)

HLSL_TYPED_UAVS = """RWTexture2D<unorm float4> colors : register(u0);
RWTexture2D<float2> pairs : register(u1);
[[vk::image_format("rg16f")]] RWTexture2D<float2> halves : register(u2);
[[vk::image_format("rg8snorm")]] RWTexture2D<snorm float2> signedPairs : register(u3);
RWTexture2DArray<uint2> layers : register(u4);
RWTexture3D<int> volume : register(u5);
RWTexture1D<snorm float> ramp : register(u6);
RWTexture2D<float3> rgb : register(u7);
RWTexture2D<float4> plain : register(u8);
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float4 c = colors[id.xy];
    colors[id.xy] = c * 0.5;
    float2 p = pairs[id.xy];
    pairs[id.xy] = p + halves[id.xy];
    halves[id.xy] = signedPairs[id.xy];
    signedPairs[id.xy] = -p;
    uint2 l = layers[id];
    layers[id] = l + 1u;
    int v = volume[id];
    volume[id] = v * 2;
    ramp[id.x] = ramp[id.x] * 0.5;
    float3 t = rgb[id.xy];
    rgb[id.xy] = t.zyx;
    plain[id.xy] = plain[id.xy] + c;
}
"""

SLANG_TYPED_UAVS = """[format("rgba8")] RWTexture2D<float4> colors;
[[vk::image_format("r32ui")]] RWTexture2D<uint> counters;
RWTexture2D<snorm float2> pairs;
RWTexture2DArray<int2> layers;
RWTexture2D<float4> plain;
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float4 c = colors.Load(int2(id.xy));
    plain.Store(int2(id.xy), c);
}
"""

HLSL_PARAMETERS = """RWTexture2D<float2> pairs : register(u0);
RWTexture2D<uint> counts : register(u1);
float2 readPair(RWTexture2D<float2> img, uint2 p) { return img[p]; }
float2 twice(RWTexture2D<float2> img, uint2 p) { return readPair(img, p) * 2.0; }
void bump(RWTexture2D<uint> img, uint2 p) { uint o; InterlockedAdd(img[p], 1u, o); }
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    pairs[id.xy] = twice(pairs, id.xy);
    bump(counts, id.xy);
}
"""

FORMATTED_PARAMETER = """shader FormattedParameter {
    @ format(r32f) image2D scalars;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        vec4 readTexel(image2D image, ivec2 p) { return imageLoad(image, p); }
        void main() {
            imageStore(scalars, ivec2(0, 0), readTexel(scalars, ivec2(1, 1)));
        }
    }
}
"""

MIXED_PARAMETER = """shader MixedParameter {
    @ format(r32f) image2D scalars;
    @ format(rgba32f) image2D colors;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        vec4 readTexel(image2D image, ivec2 p) { return imageLoad(image, p); }
        void main() {
            imageStore(colors, ivec2(0, 0), readTexel(scalars, ivec2(1, 1)));
            imageStore(scalars, ivec2(0, 0), readTexel(colors, ivec2(1, 1)));
        }
    }
}
"""

INVALID_TEXELS = """shader InvalidTexels {
    @ format(unknown)
    @ texel(vec2)
    uimage2D kindMismatch;
    @ format(rg32f)
    @ texel(vec2)
    image2D formatted;
    @ format(unknown)
    @ texel(half4)
    image2D unknownType;
    @ format(unknown)
    @ texel(unorm, ivec2)
    iimage2D normalizedInteger;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            imageStore(kindMismatch, ivec2(0, 0), uvec4(1u));
            imageStore(formatted, ivec2(0, 0), vec4(1.0));
            imageStore(unknownType, ivec2(0, 0), vec4(1.0));
            imageStore(normalizedInteger, ivec2(0, 0), ivec4(1));
        }
    }
}
"""


def _declaration_attributes(crossgl, name):
    """The prefix ``@`` attributes of the declaration of ``name``."""
    match = re.search(rf"((?:^\s*@ [^\n]+\n)*)^\s*\w+ {name};", crossgl, re.M)
    assert match, (name, crossgl[:2000])
    return re.findall(r"@ ([^\n]+)", match.group(1))


@pytest.mark.parametrize(
    ("name", "attributes"),
    [
        ("colors", ["format(unknown)", "texel(unorm, vec4)"]),
        ("pairs", ["format(unknown)", "texel(vec2)"]),
        ("halves", ["format(rg16f)"]),
        ("signedPairs", ["format(rg8_snorm)"]),
        ("layers", ["format(unknown)", "texel(uvec2)"]),
        ("volume", ["format(unknown)", "texel(int)"]),
        ("ramp", ["format(unknown)", "texel(snorm, float)"]),
        ("rgb", ["format(unknown)", "texel(vec3)"]),
        ("plain", ["format(unknown)"]),
    ],
)
def test_hlsl_typed_uavs_import_their_format_or_texel_type(tmp_path, name, attributes):
    crossgl = _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "cgl")
    declared = [
        attribute
        for attribute in _declaration_attributes(crossgl, name)
        if attribute.startswith(("format", "texel"))
    ]
    assert declared == attributes, crossgl
    assert "vk_image_format" not in crossgl


def test_slang_typed_uavs_import_their_format_or_texel_type(tmp_path):
    crossgl = _translate(tmp_path, SLANG_TYPED_UAVS, ".slang", "cgl")
    for declaration in (
        "image2D colors @format(rgba8);",
        "uimage2D counters @format(r32ui);",
        "image2D pairs @format(unknown) @texel(snorm, vec2);",
        "iimage2DArray layers @format(unknown) @texel(ivec2);",
        "image2D plain @format(unknown);",
    ):
        assert declaration in crossgl, crossgl
    assert "vk_image_format" not in crossgl


@pytest.mark.parametrize(
    ("source", "suffix", "refusal"),
    [
        (
            '[[vk::image_format("bogus")]] RWTexture2D<float4> img;\n',
            ".hlsl",
            DirectXImportRefused,
        ),
        ('[format("bogus")] RWTexture2D<float4> img;\n', ".slang", SlangImportRefused),
    ],
)
def test_unknown_source_image_formats_are_refused(tmp_path, source, suffix, refusal):
    shader = source + (
        "[numthreads(1, 1, 1)]\n"
        "void CSMain(uint3 id : SV_DispatchThreadID) { img[id.xy] = float4(1, 0, 0, 1); }\n"
    )
    with pytest.raises(refusal, match="no such storage image format"):
        _translate(tmp_path, shader, suffix, "cgl")


def test_hlsl_output_declares_the_source_texel_types(tmp_path):
    hlsl = _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "directx")
    for declaration in (
        "RWTexture2D<unorm float4> colors : register(u0);",
        "RWTexture2D<float2> pairs : register(u1);",
        "RWTexture2D<float2> halves : register(u2);",
        "RWTexture2D<snorm float2> signedPairs : register(u3);",
        "RWTexture2DArray<uint2> layers : register(u4);",
        "RWTexture3D<int> volume : register(u5);",
        "RWTexture1D<snorm float> ramp : register(u6);",
        "RWTexture2D<float3> rgb : register(u7);",
        "RWTexture2D<float4> plain : register(u8);",
    ):
        assert declaration in hlsl, declaration
    assert '[[vk::image_format("rg16f")]]' in hlsl
    # Stores keep the components the texel holds, also for normalized ones.
    assert re.search(r"signedPairs\[id\.xy\] = \(float4\([^;]+\)\)\.xy;", hlsl), hlsl
    assert re.search(r"ramp\[id\.x\] = \(float4\([^;]+\)\)\.x;", hlsl), hlsl
    _compile_compute_hlsl(tmp_path, hlsl, "CSMain")


def test_slang_output_declares_the_source_texel_types(tmp_path):
    slang = _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "slang")
    for declaration in (
        "RWTexture2D<unorm float4> colors : register(u0);",
        "RWTexture2D<float2> pairs : register(u1);",
        '[format("rg16f")] RWTexture2D<float2> halves : register(u2);',
        "RWTexture3D<int> volume : register(u5);",
        "RWTexture2D<float3> rgb : register(u7);",
    ):
        assert declaration in slang, declaration
    # A load fills the components a narrow texel lacks; a store keeps its own.
    assert "float2 p = float4(pairs[id.xy], 0.0, 1.0).xy;" in slang
    assert "int v = int4(volume[id], 0, 0, 1).x;" in slang
    assert "float3 t = float4(rgb[id.xy], 1.0).xyz;" in slang
    assert "signedPairs[id.xy] = (float4(-p, 0.0, 1.0)).xy;" in slang
    assert "rgb[id.xy] = (float4(t.zyx, 1.0)).xyz;" in slang
    _compile_slang(tmp_path, slang, "CSMain")


def test_other_targets_compile_images_with_declared_texels(tmp_path):
    _validate_compute_glsl(
        tmp_path, _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "opengl")
    )
    _validate(tmp_path, _translate(tmp_path, HLSL_TYPED_UAVS, ".hlsl", "vulkan"))


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_storage_image_parameters_take_the_type_of_their_arguments(tmp_path, backend):
    code = _translate(tmp_path, HLSL_PARAMETERS, ".hlsl", backend)
    for declaration in (
        "float2 readPair(RWTexture2D<float2> img, uint2 p)",
        "float2 twice(RWTexture2D<float2> img, uint2 p)",
        "void bump(RWTexture2D<uint> img, uint2 p)",
    ):
        assert declaration in code, declaration
    assert "RWTexture2D<float4>" not in code
    if backend == "directx":
        _compile_compute_hlsl(tmp_path, code, "CSMain")
    else:
        _compile_slang(tmp_path, code, "CSMain")


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_formatted_image_parameters_take_the_format_texel(tmp_path, backend):
    code = _translate(tmp_path, FORMATTED_PARAMETER, ".cgl", backend)
    assert "float4 readTexel(RWTexture2D<float> image, int2 p)" in code
    assert "return float4(image[p], 0.0, 0.0, 1.0);" in code
    if backend == "directx":
        _compile_compute_hlsl(tmp_path, code, "CSMain")
    else:
        _compile_slang(tmp_path, code, "main")


@pytest.mark.parametrize(
    ("backend", "refusal"),
    [("directx", DirectXLoweringRefused), ("slang", SlangLoweringRefused)],
)
def test_parameters_receiving_different_image_types_are_refused(
    tmp_path, backend, refusal
):
    with pytest.raises(refusal) as raised:
        _translate(tmp_path, MIXED_PARAMETER, ".cgl", backend)
    message = str(raised.value)
    assert "storage image parameter 'image' receives images of different types" in (
        message
    )
    assert "colors as RWTexture2D<float4>; scalars as RWTexture2D<float>" in message


@pytest.mark.parametrize(
    ("backend", "refusal"),
    [("directx", DirectXLoweringRefused), ("slang", SlangLoweringRefused)],
)
def test_invalid_texel_attributes_are_refused(tmp_path, backend, refusal):
    with pytest.raises(refusal) as raised:
        _translate(tmp_path, INVALID_TEXELS, ".cgl", backend)
    message = str(raised.value)
    for expected in (
        "'kindMismatch': uimage2D holds uint texels, which @texel(vec2) does not",
        "'formatted': @texel(vec2) applies only to an image whose format is unknown",
        "'unknownType': @texel(half4) must name a scalar or vector texel type",
        "'normalizedInteger': iimage2D holds int texels, which @texel(unorm, ivec2)",
    ):
        assert expected in message, message


def _compile_slang(tmp_path, slang, entry):
    slangc = shutil.which("slangc")
    if slangc is None:
        return
    path = tmp_path / "shader.slang"
    path.write_text(slang, encoding="utf-8")
    result = subprocess.run(
        [
            slangc,
            str(path),
            "-target",
            "spirv",
            "-entry",
            entry,
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "shader.spv"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
