"""SPIR-V lookups through ``OpSampledImage`` import as CrossGL's separate-sampler texture calls.

A SPIR-V module with separate images and samplers (every module DXC writes, and Vulkan GLSL that declares
``texture2D`` and ``sampler``) combines them with ``OpSampledImage`` right before each lookup. The importer wrote
that as GLSL's constructor, ``texture(sampler2D(image, sampler), uv)``. The GLSL, HLSL and Metal emitters require a
declared texture as a lookup's first argument and raised, and the SPIR-V emitter refused the shadow lookups. CrossGL
spells such a lookup ``texture(image, sampler, uv)``, as the HLSL importer writes it, and ``OpImage`` takes the image
back out of the combination for ``texelFetch`` and ``textureSize``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.sampled_images import (
    sampled_image_arguments,
    sampled_image_call,
    sampled_image_image,
)
from crosstl.backend.SPIRV.VulkanAst import FunctionCallNode

# glslang 15 output, comments and debug names of locals trimmed, for this fragment shader:
#   layout(set=0, binding=0) uniform texture2D albedo;   layout(set=0, binding=1) uniform texture2D depthTex;
#   layout(set=0, binding=2) uniform textureCube env;    layout(set=0, binding=3) uniform sampler linearSampler;
#   layout(set=0, binding=4) uniform samplerShadow shadowSampler;
#   vec4 a = texture(sampler2D(albedo, linearSampler), uv) + textureLod(..., uv, 2.0) + textureGrad(..., uv, dx, dy)
#          + textureOffset(..., uv, ivec2(1, 0)) + texture(..., uv, 0.5) + textureGather(..., uv, 1)
#          + texture(samplerCube(env, linearSampler), vec3(uv, 1.0)) + texelFetch(..., ivec2(uv * 4.0), 0);
#   float s = texture(sampler2DShadow(depthTex, shadowSampler), vec3(uv, 0.5));
#   ivec2 size = textureSize(sampler2D(albedo, linearSampler), 0);
#   c = a * s + vec4(vec2(size), 0.0, 0.0);
SEPARATE_SAMPLERS = """OpCapability Shader
OpCapability ImageQuery
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %c
OpExecutionMode %main OriginUpperLeft
OpName %albedo "albedo"
OpName %linearSampler "linearSampler"
OpName %uv "uv"
OpName %env "env"
OpName %depthTex "depthTex"
OpName %shadowSampler "shadowSampler"
OpName %c "c"
OpDecorate %albedo Binding 0
OpDecorate %albedo DescriptorSet 0
OpDecorate %linearSampler Binding 3
OpDecorate %linearSampler DescriptorSet 0
OpDecorate %uv Location 0
OpDecorate %env Binding 2
OpDecorate %env DescriptorSet 0
OpDecorate %depthTex Binding 1
OpDecorate %depthTex DescriptorSet 0
OpDecorate %shadowSampler Binding 4
OpDecorate %shadowSampler DescriptorSet 0
OpDecorate %c Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
%albedo = OpVariable %_ptr_UniformConstant_10 UniformConstant
%14 = OpTypeSampler
%_ptr_UniformConstant_14 = OpTypePointer UniformConstant %14
%linearSampler = OpVariable %_ptr_UniformConstant_14 UniformConstant
%18 = OpTypeSampledImage %10
%v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%uv = OpVariable %_ptr_Input_v2float Input
%float_2 = OpConstant %float 2
%float_0_100000001 = OpConstant %float 0.100000001
%float_0 = OpConstant %float 0
%38 = OpConstantComposite %v2float %float_0_100000001 %float_0
%39 = OpConstantComposite %v2float %float_0 %float_0_100000001
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%int_1 = OpConstant %int 1
%int_0 = OpConstant %int 0
%50 = OpConstantComposite %v2int %int_1 %int_0
%float_0_5 = OpConstant %float 0.5
%66 = OpTypeImage %float Cube 0 0 0 1 Unknown
%_ptr_UniformConstant_66 = OpTypePointer UniformConstant %66
%env = OpVariable %_ptr_UniformConstant_66 UniformConstant
%71 = OpTypeSampledImage %66
%float_1 = OpConstant %float 1
%v3float = OpTypeVector %float 3
%float_4 = OpConstant %float 4
%_ptr_Function_float = OpTypePointer Function %float
%93 = OpTypeImage %float 2D 1 0 0 1 Unknown
%_ptr_UniformConstant_93 = OpTypePointer UniformConstant %93
%depthTex = OpVariable %_ptr_UniformConstant_93 UniformConstant
%shadowSampler = OpVariable %_ptr_UniformConstant_14 UniformConstant
%99 = OpTypeSampledImage %93
%_ptr_Function_v2int = OpTypePointer Function %v2int
%_ptr_Output_v4float = OpTypePointer Output %v4float
%c = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %3
%5 = OpLabel
%a = OpVariable %_ptr_Function_v4float Function
%s = OpVariable %_ptr_Function_float Function
%size = OpVariable %_ptr_Function_v2int Function
%13 = OpLoad %10 %albedo
%17 = OpLoad %14 %linearSampler
%19 = OpSampledImage %18 %13 %17
%23 = OpLoad %v2float %uv
%24 = OpImageSampleImplicitLod %v4float %19 %23
%25 = OpLoad %10 %albedo
%26 = OpLoad %14 %linearSampler
%27 = OpSampledImage %18 %25 %26
%28 = OpLoad %v2float %uv
%30 = OpImageSampleExplicitLod %v4float %27 %28 Lod %float_2
%31 = OpFAdd %v4float %24 %30
%32 = OpLoad %10 %albedo
%33 = OpLoad %14 %linearSampler
%34 = OpSampledImage %18 %32 %33
%35 = OpLoad %v2float %uv
%40 = OpImageSampleExplicitLod %v4float %34 %35 Grad %38 %39
%41 = OpFAdd %v4float %31 %40
%42 = OpLoad %10 %albedo
%43 = OpLoad %14 %linearSampler
%44 = OpSampledImage %18 %42 %43
%45 = OpLoad %v2float %uv
%51 = OpImageSampleImplicitLod %v4float %44 %45 ConstOffset %50
%52 = OpFAdd %v4float %41 %51
%53 = OpLoad %10 %albedo
%54 = OpLoad %14 %linearSampler
%55 = OpSampledImage %18 %53 %54
%56 = OpLoad %v2float %uv
%58 = OpImageSampleImplicitLod %v4float %55 %56 Bias %float_0_5
%59 = OpFAdd %v4float %52 %58
%60 = OpLoad %10 %albedo
%61 = OpLoad %14 %linearSampler
%62 = OpSampledImage %18 %60 %61
%63 = OpLoad %v2float %uv
%64 = OpImageGather %v4float %62 %63 %int_1
%65 = OpFAdd %v4float %59 %64
%69 = OpLoad %66 %env
%70 = OpLoad %14 %linearSampler
%72 = OpSampledImage %71 %69 %70
%73 = OpLoad %v2float %uv
%76 = OpCompositeExtract %float %73 0
%77 = OpCompositeExtract %float %73 1
%78 = OpCompositeConstruct %v3float %76 %77 %float_1
%79 = OpImageSampleImplicitLod %v4float %72 %78
%80 = OpFAdd %v4float %65 %79
%81 = OpLoad %10 %albedo
%82 = OpLoad %14 %linearSampler
%83 = OpSampledImage %18 %81 %82
%84 = OpLoad %v2float %uv
%86 = OpVectorTimesScalar %v2float %84 %float_4
%87 = OpConvertFToS %v2int %86
%88 = OpImage %10 %83
%89 = OpImageFetch %v4float %88 %87 Lod %int_0
%90 = OpFAdd %v4float %80 %89
OpStore %a %90
%96 = OpLoad %93 %depthTex
%98 = OpLoad %14 %shadowSampler
%100 = OpSampledImage %99 %96 %98
%101 = OpLoad %v2float %uv
%102 = OpCompositeExtract %float %101 0
%103 = OpCompositeExtract %float %101 1
%104 = OpCompositeConstruct %v3float %102 %103 %float_0_5
%105 = OpCompositeExtract %float %104 2
%106 = OpImageSampleDrefImplicitLod %float %100 %104 %105
OpStore %s %106
%109 = OpLoad %10 %albedo
%110 = OpLoad %14 %linearSampler
%111 = OpSampledImage %18 %109 %110
%112 = OpImage %10 %111
%113 = OpImageQuerySizeLod %v2int %112 %int_0
OpStore %size %113
%116 = OpLoad %v4float %a
%117 = OpLoad %float %s
%118 = OpVectorTimesScalar %v4float %116 %117
%119 = OpLoad %v2int %size
%120 = OpConvertSToF %v2float %119
%121 = OpCompositeExtract %float %120 0
%122 = OpCompositeExtract %float %120 1
%123 = OpCompositeConstruct %v4float %121 %122 %float_0 %float_0
%124 = OpFAdd %v4float %118 %123
OpStore %c %124
OpReturn
OpFunctionEnd
"""

# DXC 1.10 -spirv -T ps_6_0 output, comments trimmed, for this pixel shader:
#   Texture2D<float4> sep : register(t1); Texture2D<float> depthTex : register(t3);
#   TextureCube<float4> cube : register(t4); SamplerState s : register(s2); SamplerComparisonState cmp : register(s5);
#   float4 a = sep.Sample(s, uv) + sep.SampleLevel(s, uv, 2.0) + sep.SampleGrad(s, uv, float2(0.1, 0), float2(0, 0.1));
#   float b = depthTex.SampleCmp(cmp, uv, 0.5) + depthTex.SampleCmpLevelZero(cmp, uv, 0.25);
#   float4 g = sep.GatherRed(s, uv) + cube.Sample(s, float3(uv, 1.0)) + sep.Sample(s, uv, int2(1, 0))
#            + sep.SampleBias(s, uv, 0.5);
#   return a + b + g;
DXC_SEPARATE_SAMPLERS = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_TEXCOORD0 %out_var_SV_Target
OpExecutionMode %main OriginUpperLeft
OpName %type_2d_image "type.2d.image"
OpName %sep "sep"
OpName %depthTex "depthTex"
OpName %type_cube_image "type.cube.image"
OpName %cube "cube"
OpName %type_sampler "type.sampler"
OpName %s "s"
OpName %cmp "cmp"
OpName %in_var_TEXCOORD0 "in.var.TEXCOORD0"
OpName %out_var_SV_Target "out.var.SV_Target"
OpName %main "main"
OpName %type_sampled_image "type.sampled.image"
OpName %type_sampled_image_0 "type.sampled.image"
OpDecorate %in_var_TEXCOORD0 Location 0
OpDecorate %out_var_SV_Target Location 0
OpDecorate %sep DescriptorSet 0
OpDecorate %sep Binding 1
OpDecorate %depthTex DescriptorSet 0
OpDecorate %depthTex Binding 3
OpDecorate %cube DescriptorSet 0
OpDecorate %cube Binding 4
OpDecorate %s DescriptorSet 0
OpDecorate %s Binding 2
OpDecorate %cmp DescriptorSet 0
OpDecorate %cmp Binding 5
%float = OpTypeFloat 32
%float_2 = OpConstant %float 2
%float_0_100000001 = OpConstant %float 0.100000001
%float_0 = OpConstant %float 0
%v2float = OpTypeVector %float 2
%19 = OpConstantComposite %v2float %float_0_100000001 %float_0
%20 = OpConstantComposite %v2float %float_0 %float_0_100000001
%float_0_5 = OpConstant %float 0.5
%float_0_25 = OpConstant %float 0.25
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float_1 = OpConstant %float 1
%int_1 = OpConstant %int 1
%v2int = OpTypeVector %int 2
%28 = OpConstantComposite %v2int %int_1 %int_0
%type_2d_image = OpTypeImage %float 2D 2 0 0 1 Unknown
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
%type_cube_image = OpTypeImage %float Cube 2 0 0 1 Unknown
%_ptr_UniformConstant_type_cube_image = OpTypePointer UniformConstant %type_cube_image
%type_sampler = OpTypeSampler
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
%_ptr_Input_v2float = OpTypePointer Input %v2float
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%void = OpTypeVoid
%36 = OpTypeFunction %void
%type_sampled_image = OpTypeSampledImage %type_2d_image
%v3float = OpTypeVector %float 3
%type_sampled_image_0 = OpTypeSampledImage %type_cube_image
%sep = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant
%depthTex = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant
%cube = OpVariable %_ptr_UniformConstant_type_cube_image UniformConstant
%s = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
%cmp = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
%in_var_TEXCOORD0 = OpVariable %_ptr_Input_v2float Input
%out_var_SV_Target = OpVariable %_ptr_Output_v4float Output
%main = OpFunction %void None %36
%38 = OpLabel
%39 = OpLoad %v2float %in_var_TEXCOORD0
%40 = OpLoad %type_2d_image %sep
%41 = OpLoad %type_sampler %s
%42 = OpSampledImage %type_sampled_image %40 %41
%43 = OpImageSampleImplicitLod %v4float %42 %39 None
%44 = OpLoad %type_2d_image %sep
%45 = OpLoad %type_sampler %s
%46 = OpSampledImage %type_sampled_image %44 %45
%47 = OpImageSampleExplicitLod %v4float %46 %39 Lod %float_2
%48 = OpFAdd %v4float %43 %47
%49 = OpLoad %type_2d_image %sep
%50 = OpLoad %type_sampler %s
%51 = OpSampledImage %type_sampled_image %49 %50
%52 = OpImageSampleExplicitLod %v4float %51 %39 Grad %19 %20
%53 = OpFAdd %v4float %48 %52
%54 = OpLoad %type_2d_image %depthTex
%55 = OpLoad %type_sampler %cmp
%56 = OpSampledImage %type_sampled_image %54 %55
%57 = OpImageSampleDrefImplicitLod %float %56 %39 %float_0_5 None
%58 = OpLoad %type_2d_image %depthTex
%59 = OpLoad %type_sampler %cmp
%60 = OpSampledImage %type_sampled_image %58 %59
%61 = OpImageSampleDrefExplicitLod %float %60 %39 %float_0_25 Lod %float_0
%62 = OpFAdd %float %57 %61
%63 = OpLoad %type_2d_image %sep
%64 = OpLoad %type_sampler %s
%65 = OpSampledImage %type_sampled_image %63 %64
%66 = OpImageGather %v4float %65 %39 %int_0 None
%67 = OpLoad %type_cube_image %cube
%68 = OpLoad %type_sampler %s
%69 = OpCompositeExtract %float %39 0
%70 = OpCompositeExtract %float %39 1
%71 = OpCompositeConstruct %v3float %69 %70 %float_1
%72 = OpSampledImage %type_sampled_image_0 %67 %68
%73 = OpImageSampleImplicitLod %v4float %72 %71 None
%74 = OpFAdd %v4float %66 %73
%75 = OpLoad %type_2d_image %sep
%76 = OpLoad %type_sampler %s
%77 = OpSampledImage %type_sampled_image %75 %76
%78 = OpImageSampleImplicitLod %v4float %77 %39 ConstOffset %28
%79 = OpFAdd %v4float %74 %78
%80 = OpLoad %type_2d_image %sep
%81 = OpLoad %type_sampler %s
%82 = OpSampledImage %type_sampled_image %80 %81
%83 = OpImageSampleImplicitLod %v4float %82 %39 Bias %float_0_5
%84 = OpFAdd %v4float %79 %83
%85 = OpCompositeConstruct %v4float %62 %62 %62 %62
%86 = OpFAdd %v4float %53 %85
%87 = OpFAdd %v4float %86 %84
OpStore %out_var_SV_Target %87
OpReturn
OpFunctionEnd
"""

TOOLS = {
    "directx": "dxc",
    "opengl": "glslangValidator",
    "metal": "xcrun",
    "vulkan": "spirv-val",
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.spvasm"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile(tmp_path, code, backend):
    """Compile the output with the native tool when it is installed."""
    if shutil.which(TOOLS[backend]) is None:
        return
    if backend == "directx":
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"], tmp_path)
    elif backend == "opengl":
        (tmp_path / "out.frag").write_text(code)
        _run(["glslangValidator", "-S", "frag", "out.frag"], tmp_path)
    elif backend == "metal":
        (tmp_path / "out.metal").write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )
    elif shutil.which("spirv-as"):
        (tmp_path / "out.spvasm").write_text(code)
        _run(["spirv-as", "out.spvasm", "-o", "out.spv"], tmp_path)
        _run(["spirv-val", "out.spv"], tmp_path)


def _flat(code):
    return " ".join(code.split())


def test_glslang_separate_sampler_lookups_name_the_image_and_the_sampler(tmp_path):
    code = _flat(_translate(tmp_path, SEPARATE_SAMPLERS, "cgl"))
    for lookup in (
        "texture(albedo, linearSampler, uv)",
        "textureLod(albedo, linearSampler, uv, 2.0)",
        "textureGrad(albedo, linearSampler, uv,",
        "textureOffset(albedo, linearSampler, uv, int2(1, 0))",
        "texture(albedo, linearSampler, uv, 0.5)",
        "textureGather(albedo, linearSampler, uv, 1)",
        "texture(env, linearSampler, float3(",
        "textureCompare(depthTex, shadowSampler,",
    ):
        assert lookup in code
    # OpImage takes the image back out of the combination for the lookups without a sampler.
    assert "texelFetch(albedo, int2((uv * 4.0)), 0)" in code
    assert "textureSize(albedo, 0)" in code
    assert "sampler2D(" not in code
    assert "samplerCube(" not in code
    assert "sampler2DShadow(" not in code


def test_dxc_separate_sampler_lookups_name_the_image_and_the_sampler(tmp_path):
    code = _flat(_translate(tmp_path, DXC_SEPARATE_SAMPLERS, "cgl"))
    for lookup in (
        "texture(sep, s, in_var_TEXCOORD0)",
        "textureLod(sep, s, in_var_TEXCOORD0, 2.0)",
        "textureGrad(sep, s, in_var_TEXCOORD0,",
        "textureCompare(depthTex, cmp, in_var_TEXCOORD0, 0.5)",
        "textureCompareLod(depthTex, cmp, in_var_TEXCOORD0, 0.25, 0.0)",
        "textureGather(sep, s, in_var_TEXCOORD0, 0)",
        "texture(cube, s, float3(",
        "textureOffset(sep, s, in_var_TEXCOORD0, int2(1, 0))",
        "texture(sep, s, in_var_TEXCOORD0, 0.5)",
    ):
        assert lookup in code
    assert "sampler2D(" not in code
    assert "sampler2DShadow(" not in code


@pytest.mark.parametrize(
    ("backend", "spellings"),
    [
        (
            "directx",
            (
                "albedo.Sample(linearSampler",
                "albedo.GatherGreen(linearSampler",
                "SamplerComparisonState shadowSampler",
                "depthTex.SampleCmp(shadowSampler",
            ),
        ),
        ("opengl", ("uniform sampler2D albedo;", "uniform sampler2DShadow depthTex;")),
        (
            "metal",
            (
                "albedo.sample(linearSampler",
                "depth2d<float> depthTex",
                "sample_compare",
            ),
        ),
        ("vulkan", ("OpSampledImage", "OpImageSampleDrefImplicitLod", "OpImageGather")),
    ],
)
def test_glslang_separate_samplers_compile_for_each_target(
    tmp_path, backend, spellings
):
    code = _translate(tmp_path, SEPARATE_SAMPLERS, backend)
    for spelling in spellings:
        assert spelling in code
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize(
    ("backend", "spellings"),
    [
        (
            "directx",
            ("sep.Sample(s", "SamplerComparisonState cmp", "SampleCmpLevelZero"),
        ),
        ("opengl", ("uniform sampler2D sep;", "uniform sampler2DShadow depthTex;")),
        ("metal", ("sep.sample(s", "depth2d<float> depthTex", "sample_compare")),
        ("vulkan", ("OpImageSampleDrefImplicitLod", "OpImageSampleDrefExplicitLod")),
    ],
)
def test_dxc_separate_samplers_compile_for_each_target(tmp_path, backend, spellings):
    code = _translate(tmp_path, DXC_SEPARATE_SAMPLERS, backend)
    for spelling in spellings:
        assert spelling in code
    _compile(tmp_path, code, backend)


def test_sampled_image_helpers():
    combined = sampled_image_call(FunctionCallNode, "sampler2D", "tex", "samp")
    assert combined.name == "sampler2D"
    assert combined.args == ["tex", "samp"]
    assert sampled_image_arguments(combined) == ["tex", "samp"]
    assert sampled_image_image(combined) == "tex"
    # A combined image-sampler variable, such as a GLSL sampler2D uniform, stays one argument.
    assert sampled_image_arguments("combinedTex") == ["combinedTex"]
    assert sampled_image_image("combinedTex") == "combinedTex"
    plain = FunctionCallNode("sampler2D", ["tex", "samp"])
    assert sampled_image_arguments(plain) == [plain]
