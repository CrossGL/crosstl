"""Storage image formats CrossGL names, and the names it refuses.

GLSL's format layout qualifiers include three packed formats (GLSL 4.60 section
4.4.7): ``rgb10_a2``, ``rgb10_a2ui`` and ``r11f_g11f_b10f``. SPIR-V spells them
``Rgb10A2``, ``Rgb10a2ui`` and ``R11fG11fB10f`` (StorageImageExtendedFormats),
DXC's ``[[vk::image_format]]`` ``rgb10a2``, ``rgb10a2ui`` and ``r11g11b10f``, and
Slang's ``[format]`` as GLSL does. WGSL has no such storage texel formats.

A storage image attribute that names no CrossGL format or qualifier, such as
``layout(rgb9_e5)`` or ``@format(bogus)``, is refused for every target: targets
used to substitute another format, and HLSL output wrote the name as an array
size.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.image_formats import SpirvImageFormatUnsupportedError
from crosstl.translator.codegen.image_access_contracts import (
    image_format_channel_count,
    image_format_component_kind,
)
from crosstl.translator.storage_image_qualifiers import UnsupportedImageFormatError

PACKED_FORMATS = {
    "rgb10_a2": ("image2D", 4, "float"),
    "rgb10_a2ui": ("uimage2D", 4, "uint"),
    "r11f_g11f_b10f": ("image2D", 3, "float"),
}

THREE_FORMATS = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(rgb10_a2) uniform image2D a;
        layout(rgb10_a2ui) uniform uimage2D b;
        layout(r11f_g11f_b10f) uniform image2D c;
        void main() {
            ivec2 p = ivec2(gl_GlobalInvocationID.xy);
            imageStore(a, p, imageLoad(a, p) * 0.5);
            imageStore(b, p, imageLoad(b, p) + uvec4(1u));
            imageStore(c, p, vec4(imageLoad(c, p).rgb * 2.0, 1.0));
        }
    }
}
"""

ALL_BACKENDS = (
    "directx",
    "opengl",
    "vulkan",
    "metal",
    "slang",
    "wgsl",
    "cuda",
    "hip",
    "rust",
    "mojo",
)


def single_image(declaration, store="imageStore(img, ivec2(0, 0), vec4(fout[i]));"):
    return f"""shader Probe {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fout;
        {declaration}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {store}
        }}
    }}
}}
"""


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def run_tool(tmp_path, *command):
    if shutil.which(command[0]) is None:
        pytest.skip(f"{command[0]} is not installed")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("image_format", sorted(PACKED_FORMATS))
def test_packed_format_channels_and_component_kind(image_format):
    _image_type, channels, kind = PACKED_FORMATS[image_format]
    assert image_format_channel_count(image_format) == channels
    assert image_format_component_kind(image_format) == kind


def test_glsl_output_keeps_packed_formats(tmp_path):
    code = translate(tmp_path, THREE_FORMATS, "opengl")

    assert "layout(rgb10_a2, binding = 0) uniform image2D a;" in code
    assert "layout(rgb10_a2ui, binding = 1) uniform uimage2D b;" in code
    assert "layout(r11f_g11f_b10f, binding = 2) uniform image2D c;" in code
    (tmp_path / "out.comp").write_text(code)
    run_tool(tmp_path, "glslangValidator", "-V", "out.comp", "-o", "out.spv")


def test_spirv_output_uses_packed_image_formats(tmp_path):
    code = translate(tmp_path, THREE_FORMATS, "vulkan")

    formats = re.findall(r"OpTypeImage %\w+ 2D 0 0 0 2 (\w+)", code)
    assert sorted(formats) == ["R11fG11fB10f", "Rgb10A2", "Rgb10a2ui"]
    assert "OpCapability StorageImageExtendedFormats" in code
    assert "StorageImageWriteWithoutFormat" not in code
    (tmp_path / "out.spvasm").write_text(code)
    run_tool(
        tmp_path, "spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"
    )
    run_tool(tmp_path, "spirv-val", "--target-env", "vulkan1.1", "out.spv")


def test_spirv_output_writes_three_components_to_r11f_g11f_b10f(tmp_path):
    code = translate(tmp_path, THREE_FORMATS, "vulkan")

    image = re.search(r"%(\w+) = OpTypeImage %\w+ 2D 0 0 0 2 R11fG11fB10f", code)
    assert image is not None
    # Result ids are names when SPIRV-Tools formats the module and numbers when
    # it is not installed, so follow the image type rather than the name ``c``.
    loads = re.findall(rf"%(\w+) = OpLoad %{image.group(1)} %\w+", code)
    assert loads
    writes = re.findall(r"OpImageWrite %(\w+) %\w+ %(\w+)", code)
    texels = [texel for target, texel in writes if target in loads]
    assert len(texels) == 1
    texel_type = re.search(rf"%{texels[0]} = \w+ %(\w+)", code).group(1)
    assert re.search(rf"%{texel_type} = OpTypeVector %\w+ 3", code)


def test_hlsl_output_declares_packed_texels_and_dxc_formats(tmp_path):
    code = translate(tmp_path, THREE_FORMATS, "directx")

    assert re.search(
        r'#ifdef __spirv__\n\[\[vk::image_format\("rgb10a2"\)\]\]\n#endif\n'
        r"RWTexture2D<unorm float4> a : register\(u0\);",
        code,
    )
    assert re.search(
        r'\[\[vk::image_format\("rgb10a2ui"\)\]\]\n#endif\n'
        r"RWTexture2D<uint4> b : register\(u1\);",
        code,
    )
    assert re.search(
        r'\[\[vk::image_format\("r11g11b10f"\)\]\]\n#endif\n'
        r"RWTexture2D<float3> c : register\(u2\);",
        code,
    )
    # The stored four-component texel keeps its leading three components.
    assert re.search(r"c\[p\] = .*\)\.xyz;", code)


def test_slang_output_names_packed_formats(tmp_path):
    code = translate(tmp_path, THREE_FORMATS, "slang")

    assert '[format("rgb10_a2")] uniform RWTexture2D<unorm float4> a' in code
    assert '[format("rgb10_a2ui")] uniform RWTexture2D<uint4> b' in code
    assert '[format("r11f_g11f_b10f")] uniform RWTexture2D<float3> c' in code


def test_metal_output_types_packed_formats_by_component_kind(tmp_path):
    code = translate(tmp_path, THREE_FORMATS, "metal")

    assert "texture2d<float, access::read_write> a" in code
    assert "texture2d<uint, access::read_write> b" in code
    assert "texture2d<float, access::read_write> c" in code


def test_wgsl_output_refuses_packed_formats(tmp_path):
    with pytest.raises(ValueError, match="representable image format"):
        translate(tmp_path, THREE_FORMATS, "wgsl")


@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize(
    ("declaration", "named"),
    [
        ("layout(bogus_format) uniform image2D img;", "@bogus_format"),
        ("layout(rgb9_e5) uniform image2D img;", "@rgb9_e5"),
        ("layout(rgba8, srgb8) uniform image2D img;", "@srgb8"),
        ("uniform image2D img @format(bogus);", r"@format\(bogus\)"),
        ("uniform uimage2D img @r64ui;", "@r64ui"),
    ],
)
def test_unknown_storage_image_format_is_refused(tmp_path, backend, declaration, named):
    store = (
        "imageStore(img, ivec2(0, 0), uvec4(1u));"
        if "uimage" in declaration
        else "imageStore(img, ivec2(0, 0), vec4(fout[i]));"
    )
    with pytest.raises(
        UnsupportedImageFormatError,
        match=f"Unsupported storage image format on stage variable 'img': {named} ",
    ):
        translate(tmp_path, single_image(declaration, store), backend)


def test_unknown_format_on_storage_image_array_is_refused(tmp_path):
    source = single_image(
        "layout(bogus) uniform image2D imgs[2];",
        "imageStore(imgs[1], ivec2(0, 0), vec4(fout[i]));",
    )
    with pytest.raises(UnsupportedImageFormatError, match="@bogus "):
        translate(tmp_path, source, "directx")


def test_unknown_format_on_storage_image_parameter_is_refused(tmp_path):
    source = """shader Probe {
    void store(image2D target @bogus, ivec2 p) {
        imageStore(target, p, vec4(1.0));
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(rgba8) uniform image2D img;
        void main() {
            store(img, ivec2(0, 0));
        }
    }
}
"""
    with pytest.raises(
        UnsupportedImageFormatError,
        match="parameter 'target' of function 'store': @bogus ",
    ):
        translate(tmp_path, source, "opengl")


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [
        ("directx", "RWTexture2D<unorm float4> img : register(u1);"),
        (
            "opengl",
            "layout(rgba8, binding = 0) coherent restrict uniform highp image2D img;",
        ),
        ("vulkan", re.compile(r"OpTypeImage %\w+ 2D 0 0 0 2 Rgba8\b")),
        (
            "metal",
            re.compile(
                r"texture2d<float, access::read_write,\s*"
                r"memory_coherence::memory_coherence_device>\s*img\b"
            ),
        ),
    ],
)
def test_storage_image_qualifiers_are_not_formats(tmp_path, backend, declaration):
    source = single_image("uniform image2D img @rgba8 @coherent @restrict @highp;")
    code = translate(tmp_path, source, backend)

    if isinstance(declaration, re.Pattern):
        assert declaration.search(code)
    else:
        assert declaration in code


def test_glsl_input_keeps_packed_formats(tmp_path):
    source = """#version 450
layout(local_size_x = 1) in;
layout(rgb10_a2, binding = 0) uniform image2D a;
layout(rgb10_a2ui, binding = 1) uniform uimage2D b;
layout(r11f_g11f_b10f, binding = 2) uniform image2D c;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    imageStore(a, p, imageLoad(a, p) * 0.5);
    imageStore(b, p, imageLoad(b, p) + uvec4(1u));
    imageStore(c, p, vec4(imageLoad(c, p).rgb * 2.0, 1.0));
}
"""
    code = translate(tmp_path, source, "cgl", suffix=".comp")

    assert "image2D a @binding(0) @rgb10_a2;" in code
    assert "uimage2D b @binding(1) @rgb10_a2ui;" in code
    assert "image2D c @binding(2) @r11f_g11f_b10f;" in code


SPIRV_PACKED_MODULE = """               OpCapability Shader
               OpCapability StorageImageExtendedFormats
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpName %img "img"
               OpDecorate %img DescriptorSet 0
               OpDecorate %img Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
    %v4float = OpTypeVector %float 4
      %int_0 = OpConstant %int 0
    %float_1 = OpConstant %float 1
      %coord = OpConstantComposite %v2int %int_0 %int_0
      %texel = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
      %image = OpTypeImage %float 2D 0 0 0 2 FORMAT
  %ptr_image = OpTypePointer UniformConstant %image
        %img = OpVariable %ptr_image UniformConstant
       %main = OpFunction %void None %fn
      %entry = OpLabel
     %loaded = OpLoad %image %img
               OpImageWrite %loaded %coord %texel
               OpReturn
               OpFunctionEnd
"""


@pytest.mark.parametrize(
    ("spirv_format", "crossgl_format"),
    [
        ("Rgb10A2", "rgb10_a2"),
        ("R11fG11fB10f", "r11f_g11f_b10f"),
        ("Rgba16Snorm", "rgba16_snorm"),
    ],
)
def test_spirv_input_keeps_image_formats(tmp_path, spirv_format, crossgl_format):
    module = SPIRV_PACKED_MODULE.replace("FORMAT", spirv_format)
    code = translate(tmp_path, module, "cgl", suffix=".spvasm")

    assert f"image2D img @set(0) @binding(0) @{crossgl_format};" in code


def test_spirv_input_refuses_image_format_crossgl_lacks(tmp_path):
    module = SPIRV_PACKED_MODULE.replace(
        "%float 2D 0 0 0 2 FORMAT", "%float 2D 0 0 0 2 R64ui"
    )
    with pytest.raises(SpirvImageFormatUnsupportedError, match="R64ui"):
        translate(tmp_path, module, "cgl", suffix=".spvasm")


def test_hlsl_input_reads_dxc_packed_format_names(tmp_path):
    source = """[[vk::image_format("rgb10a2")]] RWTexture2D<unorm float4> a : register(u0);
[[vk::image_format("rgb10a2ui")]] RWTexture2D<uint4> b : register(u1);
[[vk::image_format("r11g11b10f")]] RWTexture2D<float3> c : register(u2);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    a[id.xy] = a[id.xy] * 0.5;
    b[id.xy] = b[id.xy] + 1u;
    c[id.xy] = c[id.xy] * 2.0;
}
"""
    code = translate(tmp_path, source, "vulkan", suffix=".hlsl")

    formats = re.findall(r"OpTypeImage %\w+ 2D 0 0 0 2 (\w+)", code)
    assert sorted(formats) == ["R11fG11fB10f", "Rgb10A2", "Rgb10a2ui"]
