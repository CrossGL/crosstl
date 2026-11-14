"""SPIR-V size queries of storage images import as ``imageSize``.

``OpImageQuerySize`` takes a sampled or a storage image (SPIR-V 1.6 section 3.49.10, Image Instructions). The importer
always wrote ``textureSize``, which GLSL does not overload for ``image2D``, so SPIR-V and Metal refused the query and
GLSL failed to compile it.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """; SPIR-V
; Version: 1.0
               OpCapability Shader
               OpCapability ImageQuery
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %resultImage "resultImage"
               OpDecorate %resultImage DescriptorSet 0
               OpDecorate %resultImage Binding 0
               OpDecorate %resultImage NonReadable
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
    %v4float = OpTypeVector %float 4
      %image = OpTypeImage %float 2D 0 0 0 2 Rgba8
%_ptr_UniformConstant_image = OpTypePointer UniformConstant %image
%resultImage = OpVariable %_ptr_UniformConstant_image UniformConstant
      %int_1 = OpConstant %int 1
     %v2_one = OpConstantComposite %v2int %int_1 %int_1
    %float_1 = OpConstant %float 1
     %v4_one = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %10 = OpLoad %image %resultImage
         %11 = OpImageQuerySize %v2int %10
         %12 = OpISub %v2int %11 %v2_one
         %13 = OpLoad %image %resultImage
               OpImageWrite %13 %12 %v4_one
               OpReturn
               OpFunctionEnd
"""


def _write(tmp_path, source):
    path = tmp_path / "shader.spvasm"
    path.write_text(source)
    return str(path)


def test_fixture_is_valid_spirv(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    binary = tmp_path / "shader.spv"
    subprocess.run(
        ["spirv-as", _write(tmp_path, COMPUTE), "-o", str(binary)], check=True
    )
    subprocess.run(["spirv-val", str(binary)], check=True)


def test_storage_image_size_query_imports_as_image_size(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    assert re.search(r"\bimageSize\(\s*resultImage\s*\)", crossgl)
    assert "textureSize" not in crossgl


def test_storage_image_size_query_reaches_spirv_and_glsl(tmp_path):
    source = _write(tmp_path, COMPUTE)
    spirv = crosstl.translate(source, backend="vulkan", format_output=False)
    assert "OpImageQuerySize" in spirv
    glsl = crosstl.translate(source, backend="opengl", format_output=False)
    assert re.search(r"\bimageSize\(\s*resultImage\s*\)", glsl)
    if shutil.which("glslangValidator"):
        path = tmp_path / "out.comp"
        path.write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "comp", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout
