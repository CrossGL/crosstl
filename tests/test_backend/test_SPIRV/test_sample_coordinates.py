"""SPIR-V sample coordinates import with exactly the components the image consumes.

A SPIR-V sample coordinate holds ``(u[, v][, w][, array layer])`` as the image needs, then ``q`` for a projective
sample, and further components are ignored (SPIR-V 1.6 section 3.49.10, Image Instructions). glslang passes the whole
``vec3`` of ``texture(sampler2DShadow, P)``, reference included, and the importer kept it, so CrossGL compared a
two-dimensional shadow map at a three-component coordinate, which MSL rejects. Integral floating-point constants,
which spirv-dis prints without a fraction, now import as floating-point literals, and GLSL accepts the HLSL spelling
of a constant integer offset vector.

``SHADOW`` is glslang's output for::

    #version 450
    layout(binding=0) uniform sampler2DShadow shadowMap;
    layout(binding=1) uniform samplerCubeShadow cubeShadow;
    layout(location=0) in vec2 uv;
    layout(location=0) out vec4 color;
    void main() {
        float a = textureOffset(shadowMap, vec3(uv, 0.5), ivec2(1, 0));
        float c = texture(cubeShadow, vec4(uv, 1.0, 0.5));
        float e = textureLod(shadowMap, vec3(uv, 0.5), 0.0);
        color = vec4(a + c + e);
    }

and ``PROJECTIVE`` for ``color = textureProj(colorMap, vec4(uv, 0.5, 2.0))`` with a ``sampler2D colorMap``, whose
coordinate glslang writes as ``(u, v, q, q)``.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import pytest

import crosstl

SHADOW = """               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %uv %color
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %a "a"
               OpName %shadowMap "shadowMap"
               OpName %uv "uv"
               OpName %c "c"
               OpName %cubeShadow "cubeShadow"
               OpName %e "e"
               OpName %color "color"
               OpDecorate %shadowMap Binding 0
               OpDecorate %shadowMap DescriptorSet 0
               OpDecorate %uv Location 0
               OpDecorate %cubeShadow Binding 1
               OpDecorate %cubeShadow DescriptorSet 0
               OpDecorate %color Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
          %9 = OpTypeImage %float 2D 1 0 0 1 Unknown
         %10 = OpTypeSampledImage %9
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
  %shadowMap = OpVariable %_ptr_UniformConstant_10 UniformConstant
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
         %uv = OpVariable %_ptr_Input_v2float Input
  %float_0_5 = OpConstant %float 0.5
    %v3float = OpTypeVector %float 3
        %int = OpTypeInt 32 1
      %v2int = OpTypeVector %int 2
      %int_1 = OpConstant %int 1
      %int_0 = OpConstant %int 0
         %27 = OpConstantComposite %v2int %int_1 %int_0
         %31 = OpTypeImage %float Cube 1 0 0 1 Unknown
         %32 = OpTypeSampledImage %31
%_ptr_UniformConstant_32 = OpTypePointer UniformConstant %32
 %cubeShadow = OpVariable %_ptr_UniformConstant_32 UniformConstant
    %float_1 = OpConstant %float 1
    %v4float = OpTypeVector %float 4
    %float_0 = OpConstant %float 0
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %color = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %3
          %5 = OpLabel
          %a = OpVariable %_ptr_Function_float Function
          %c = OpVariable %_ptr_Function_float Function
          %e = OpVariable %_ptr_Function_float Function
         %13 = OpLoad %10 %shadowMap
         %17 = OpLoad %v2float %uv
         %20 = OpCompositeExtract %float %17 0
         %21 = OpCompositeExtract %float %17 1
         %22 = OpCompositeConstruct %v3float %20 %21 %float_0_5
         %28 = OpCompositeExtract %float %22 2
         %29 = OpImageSampleDrefImplicitLod %float %13 %22 %28 ConstOffset %27
               OpStore %a %29
         %35 = OpLoad %32 %cubeShadow
         %36 = OpLoad %v2float %uv
         %39 = OpCompositeExtract %float %36 0
         %40 = OpCompositeExtract %float %36 1
         %41 = OpCompositeConstruct %v4float %39 %40 %float_1 %float_0_5
         %42 = OpCompositeExtract %float %41 3
         %43 = OpImageSampleDrefImplicitLod %float %35 %41 %42
               OpStore %c %43
         %45 = OpLoad %10 %shadowMap
         %46 = OpLoad %v2float %uv
         %47 = OpCompositeExtract %float %46 0
         %48 = OpCompositeExtract %float %46 1
         %49 = OpCompositeConstruct %v3float %47 %48 %float_0_5
         %51 = OpCompositeExtract %float %49 2
         %52 = OpImageSampleDrefExplicitLod %float %45 %49 %51 Lod %float_0
               OpStore %e %52
         %55 = OpLoad %float %a
         %56 = OpLoad %float %c
         %57 = OpFAdd %float %55 %56
         %58 = OpLoad %float %e
         %59 = OpFAdd %float %57 %58
         %60 = OpCompositeConstruct %v4float %59 %59 %59 %59
               OpStore %color %60
               OpReturn
               OpFunctionEnd
"""

PROJECTIVE = """               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %color %uv
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %color "color"
               OpName %colorMap "colorMap"
               OpName %uv "uv"
               OpDecorate %color Location 0
               OpDecorate %colorMap Binding 0
               OpDecorate %colorMap DescriptorSet 0
               OpDecorate %uv Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %color = OpVariable %_ptr_Output_v4float Output
         %10 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
   %colorMap = OpVariable %_ptr_UniformConstant_11 UniformConstant
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
         %uv = OpVariable %_ptr_Input_v2float Input
  %float_0_5 = OpConstant %float 0.5
    %float_2 = OpConstant %float 2
       %main = OpFunction %void None %3
          %5 = OpLabel
         %14 = OpLoad %11 %colorMap
         %18 = OpLoad %v2float %uv
         %21 = OpCompositeExtract %float %18 0
         %22 = OpCompositeExtract %float %18 1
         %23 = OpCompositeConstruct %v4float %21 %22 %float_0_5 %float_2
         %24 = OpCompositeExtract %float %23 3
         %25 = OpCompositeInsert %v4float %24 %23 2
         %26 = OpImageSampleProjImplicitLod %v4float %14 %25
               OpStore %color %26
               OpReturn
               OpFunctionEnd
"""


def _translate(tmp_path, backend, module=SHADOW):
    path = tmp_path / "shader.spvasm"
    path.write_text(module)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sample_coordinates_keep_only_the_consumed_components(tmp_path):
    crossgl = _translate(tmp_path, "cgl")

    assert "textureCompareOffset(shadowMap, float3(uv[0], uv[1], 0.5).xy," in crossgl
    assert "textureCompare(cubeShadow, float4(uv[0], uv[1], 1.0, 0.5).xyz," in crossgl
    assert "textureCompareLod(shadowMap, float3(uv[0], uv[1], 0.5).xy," in crossgl


def test_projective_coordinates_keep_q_after_the_consumed_components(tmp_path):
    crossgl = _translate(tmp_path, "cgl", PROJECTIVE)

    assert "textureProj(colorMap, float4(" in crossgl
    assert crossgl.count(".xyz)") == 1


def test_integral_float_constants_import_as_float_literals(tmp_path):
    crossgl = _translate(tmp_path, "cgl")

    assert "float3(uv[0], uv[1], 0.5)[2], 0.0)" in crossgl
    assert "float4(uv[0], uv[1], 1.0, 0.5)" in crossgl


def test_shadow_samples_round_trip_to_glsl(tmp_path):
    glsl = _translate(tmp_path, "opengl")

    if shutil.which("glslangValidator"):
        path = tmp_path / "out.frag"
        path.write_text(glsl)
        _run(["glslangValidator", "-S", "frag", str(path)])


def test_shadow_samples_round_trip_to_spirv(tmp_path):
    spirv = _translate(tmp_path, "vulkan")

    assert "OpImageSampleDrefExplicitLod" in spirv
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        asm.write_text(spirv)
        _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


@pytest.mark.skipif(platform.system() != "Darwin", reason="needs the Metal toolchain")
def test_shadow_samples_round_trip_to_metal(tmp_path):
    msl = _translate(tmp_path, "metal")

    assert "sample_compare(" in msl
    if (
        shutil.which("xcrun")
        and subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        == 0
    ):
        path = tmp_path / "out.metal"
        path.write_text(msl)
        _run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(path),
                "-o",
                str(tmp_path / "out.air"),
            ]
        )
