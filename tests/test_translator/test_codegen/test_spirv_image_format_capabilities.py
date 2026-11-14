"""SPIR-V capabilities that storage-image formats require (SPIR-V 1.6, section 3.11 "Image Format")."""

import re
import shutil
import subprocess

import pytest

from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_image_texels import SHADER_IMAGE_FORMATS
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate(source_code):
    return VulkanSPIRVCodeGen().generate(Parser(Lexer(source_code).tokens).parse())


def storage_image_shader(declaration, store):
    return f"""
    shader Images {{
        {declaration}

        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {{
                {store}
            }}
        }}
    }}
    """


def assert_validates(spv_code, tmp_path):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    assemble = subprocess.run(
        [spirv_as, "--target-env", "vulkan1.1", str(asm_path), "-o", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assemble.returncode == 0, assemble.stderr
    validate = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


@pytest.mark.parametrize(
    "image_type,image_format,spirv_format,value",
    [
        ("image2D", "rg8", "Rg8", "vec4(0.5, 0.25, 0.0, 0.0)"),
        ("uimage2D", "rg8ui", "Rg8ui", "uvec4(1u, 2u, 0u, 0u)"),
        ("image2D", "r16f", "R16f", "vec4(0.5)"),
        ("iimage2D", "r8i", "R8i", "ivec4(-1)"),
        ("image2D", "rgba16", "Rgba16", "vec4(0.5)"),
    ],
)
def test_extended_storage_image_formats_declare_the_capability(
    image_type, image_format, spirv_format, value, tmp_path
):
    spv_code = generate(
        storage_image_shader(
            f"{image_type} target @{image_format};",
            f"imageStore(target, ivec2(0, 0), {value});",
        )
    )

    assert re.search(rf"OpTypeImage %\d+ 2D 0 0 0 2 {spirv_format}\b", spv_code)
    assert "OpCapability StorageImageExtendedFormats" in spv_code
    assert_validates(spv_code, tmp_path)


@pytest.mark.parametrize(
    "image_type,image_format,spirv_format,value",
    [
        ("image2D", "rgba8", "Rgba8", "vec4(0.5)"),
        ("image2D", "r32f", "R32f", "vec4(0.5)"),
        ("uimage2D", "rgba32ui", "Rgba32ui", "uvec4(1u)"),
    ],
)
def test_core_storage_image_formats_need_no_extra_capability(
    image_type, image_format, spirv_format, value, tmp_path
):
    spv_code = generate(
        storage_image_shader(
            f"{image_type} target @{image_format};",
            f"imageStore(target, ivec2(0, 0), {value});",
        )
    )

    assert re.search(rf"OpTypeImage %\d+ 2D 0 0 0 2 {spirv_format}\b", spv_code)
    assert "StorageImageExtendedFormats" not in spv_code
    assert_validates(spv_code, tmp_path)


def test_format_capability_table_covers_every_mapped_format():
    codegen = VulkanSPIRVCodeGen()
    core = {
        "Rgba32f",
        "Rgba16f",
        "R32f",
        "Rgba8",
        "Rgba8Snorm",
        "Rgba32i",
        "Rgba16i",
        "Rgba8i",
        "R32i",
        "Rgba32ui",
        "Rgba16ui",
        "Rgba8ui",
        "R32ui",
    }
    assert SHADER_IMAGE_FORMATS == core | {"Unknown"}
    mapped = set(codegen.spirv_image_format_map().values())
    assert core <= mapped
