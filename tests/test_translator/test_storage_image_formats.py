"""A storage image keeps its texel format in DirectX and Slang output.

An 8- or 16-bit normalized format is a ``unorm``/``snorm`` texel type, and any
format that is not a 32-bit scalar type's carries an explicit SPIR-V format:
``[[vk::image_format]]`` under ``__spirv__`` for DXC, ``[format]`` for Slang.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_image_formats import (
    dxc_derived_format,
    dxc_vulkan_image_format_prefix,
    normalized_texel_type,
)
from crosstl.translator.codegen.image_access_contracts import SUPPORTED_IMAGE_FORMATS

FORMATS = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        layout(rgba8) uniform writeonly image2D img;
        layout(rg8_snorm) uniform image2D sn;
        layout(rgba16f) uniform image2D hf;
        layout(rgba32f) uniform image2D full;
        layout(rgba16) uniform writeonly image2D wide[2];
        void main() {
            uint i = gl_GlobalInvocationID.x;
            imageStore(img, ivec2(0, 0), vec4(fin[i]));
            imageStore(sn, ivec2(0, 0), imageLoad(sn, ivec2(1, 0)));
            imageStore(hf, ivec2(0, 0), imageLoad(hf, ivec2(1, 0)) + vec4(fin[i]));
            imageStore(full, ivec2(0, 0), imageLoad(full, ivec2(1, 0)));
            imageStore(wide[1], ivec2(0, 0), vec4(fin[i]));
        }
    }
}
"""

# The SPIR-V format each image must have, in declaration order.
SPIRV_FORMATS = ["Rgba8", "Rg8Snorm", "Rgba16f", "Rgba32f", "Rgba16"]


def _translate(tmp_path, backend):
    path = tmp_path / "formats.cgl"
    path.write_text(FORMATS, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _declaration(code, name):
    match = re.search(rf"(?m)^.*\b{name}\b(?:\[\d+\])? : register\(u\d+\);$", code)
    assert match, code
    return match.group(0)


def _image_formats(spirv_assembly):
    return re.findall(r"OpTypeImage %\w+ 2D \d \d \d 2 (\w+)", spirv_assembly)


def test_normalized_formats_are_unorm_or_snorm_texels():
    assert normalized_texel_type("rgba8", "float4") == "unorm float4"
    assert normalized_texel_type("r16", "float") == "unorm float"
    assert normalized_texel_type("rg8_snorm", "float2") == "snorm float2"
    assert normalized_texel_type("rgba16f", "float4") == "float4"
    assert normalized_texel_type("r32ui", "uint") == "uint"


def test_dxc_needs_an_explicit_format_unless_the_texel_is_32_bit():
    for image_format in SUPPORTED_IMAGE_FORMATS:
        prefix = dxc_vulkan_image_format_prefix(image_format)
        if "32" in image_format:
            assert dxc_derived_format(image_format) == image_format
            assert prefix == ""
        else:
            assert prefix.startswith('#ifdef __spirv__\n[[vk::image_format("')
            assert "_snorm" not in prefix
    assert dxc_vulkan_image_format_prefix(None) == ""


def test_directx_output_declares_each_image_format(tmp_path):
    hlsl = _translate(tmp_path, "directx")
    assert _declaration(hlsl, "img") == "RWTexture2D<unorm float4> img : register(u1);"
    assert _declaration(hlsl, "sn").startswith("RWTexture2D<snorm float2> sn ")
    assert _declaration(hlsl, "hf").startswith("RWTexture2D<float4> hf ")
    assert _declaration(hlsl, "wide").startswith("RWTexture2D<unorm float4> wide[2] ")
    for name, spelling in (
        ("img", "rgba8"),
        ("sn", "rg8snorm"),
        ("hf", "rgba16f"),
        ("wide", "rgba16"),
    ):
        guard = (
            f'#ifdef __spirv__\n[[vk::image_format("{spelling}")]]\n#endif\n'
            + _declaration(hlsl, name)
        )
        assert guard in hlsl
    # DXC derives rgba32f from float4 itself.
    assert "#endif\n" + _declaration(hlsl, "full") not in hlsl
    assert hlsl.count("vk::image_format") == 4


def test_slang_output_declares_each_image_format(tmp_path):
    slang = _translate(tmp_path, "slang")
    for name, image_format, texel in (
        ("img", "rgba8", "unorm float4"),
        ("sn", "rg8_snorm", "snorm float2"),
        ("hf", "rgba16f", "float4"),
        ("full", "rgba32f", "float4"),
        ("wide", "rgba16", "unorm float4"),
    ):
        assert re.search(
            rf'\[format\("{image_format}"\)\] uniform RWTexture2D<{texel}> {name}\b',
            slang,
        ), slang


def _run(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False, timeout=180
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    return result.stdout


def test_dxc_compiles_the_formats_for_dxil_and_spirv(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    (tmp_path / "formats.hlsl").write_text(
        _translate(tmp_path, "directx"), encoding="utf-8"
    )
    base = [dxc, "-WX", "-T", "cs_6_0", "-E", "CSMain", "formats.hlsl"]
    _run([*base, "-Fo", "formats.dxil"], tmp_path)
    assembly = _run([*base, "-spirv"], tmp_path)
    assert _image_formats(assembly) == SPIRV_FORMATS


def test_slangc_compiles_the_formats_for_spirv(tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    (tmp_path / "formats.slang").write_text(
        _translate(tmp_path, "slang"), encoding="utf-8"
    )
    _run(
        [
            slangc,
            "formats.slang",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-target",
            "spirv-asm",
            "-o",
            "formats.spvasm",
        ],
        tmp_path,
    )
    assembly = (tmp_path / "formats.spvasm").read_text(encoding="utf-8")
    assert sorted(_image_formats(assembly)) == sorted(SPIRV_FORMATS)
