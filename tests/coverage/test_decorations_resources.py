"""Ray tracing storage, descriptor sets, interlocked regions and image types a translation must keep."""

import dataclasses

import pytest

from tools.coverage import decorations, lowering, validators
from tools.coverage.model import Feature


def _ray(name, group, stage="ray-generation"):
    return Feature("spirv", "graphics", name, None, group=group, stage=stage)


def test_a_ray_tracing_translation_keeps_the_storage_and_calls_that_hand_data_over():
    # A storage class is no decoration, so a payload is kept only by a variable in ray-payload storage, and is
    # handed over only by the trace that takes it.
    payload = _ray("ray-payload", "StorageClass.RayPayloadKHR OpTraceRayKHR")
    variable = "%payload = OpVariable %_ptr_RayPayloadKHR_Payload RayPayloadKHR\n"
    trace = "OpTraceRayKHR %scene %uint_0 %uint_255 %uint_0 %uint_1 %uint_0 %o %t %d %m %payload\n"
    assert decorations.lost(payload, "spirv", variable + trace) == ""
    # SPV_NV_ray_tracing gave the storage classes their values first, so an older disassembler prints NV names.
    nv = variable.replace("RayPayloadKHR", "RayPayloadNV")
    assert decorations.lost(payload, "spirv", nv + trace) == ""
    assert decorations.lost(payload, "spirv", variable) == (
        "the translation loses OpTraceRayKHR"
    )
    private = variable.replace(" RayPayloadKHR\n", " Private\n")
    assert decorations.lost(payload, "spirv", private + trace) == (
        "the translation loses StorageClass.RayPayloadKHR"
    )
    record = _ray("ray-shader-record", "StorageClass.ShaderRecordBufferKHR", "miss")
    assert (
        decorations.lost(
            record, "spirv", "%record = OpVariable %_ptr_Record ShaderRecordBufferNV\n"
        )
        == ""
    )
    assert decorations.lost(
        record, "spirv", "%record = OpVariable %_ptr_Record StorageBuffer\n"
    ) == ("the translation loses StorageClass.ShaderRecordBufferKHR")
    reported = _ray(
        "ray-reported-attributes",
        "StorageClass.HitAttributeKHR OpReportIntersectionKHR",
        "intersection",
    )
    attributes = (
        "%Attributes = OpTypeStruct %v3float\n"
        "%_ptr_HitAttributeKHR_Attributes = OpTypePointer HitAttributeKHR %Attributes\n"
    )
    report = "%hit = OpReportIntersectionNV %bool %float_1 %uint_0\n"
    assert decorations.lost(reported, "spirv", attributes + report) == ""
    # The barycentrics a triangle hit carries are a vector, not attributes of the stage's own.
    barycentrics = (
        "%_ptr_HitAttributeKHR_v2float = OpTypePointer HitAttributeKHR %v2float\n"
    )
    assert decorations.lost(reported, "spirv", barycentrics + report) == (
        "the translation loses StorageClass.HitAttributeKHR"
    )


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_the_data_ray_tracing_stages_hand_over():
    callable_data = Feature(
        "glsl",
        "graphics",
        "ray-callable-data",
        None,
        group="StorageClass.CallableDataKHR OpExecuteCallableKHR",
        stage="ray-generation",
    )
    source = (
        "#version 460\n#extension GL_EXT_ray_tracing : require\n"
        "struct CallData {\n    vec4 color;\n};\nlayout(location = 0) callableDataEXT CallData data;\n"
        "layout(std430, binding = 0) buffer Output {\n    float outp[];\n};\n"
        "void main() {\n    data.color = vec4(0.0);\n    executeCallableEXT(0u, 0);\n"
        "    outp[0] = data.color.x;\n}\n"
    )
    assert decorations.lost(callable_data, "glsl", source) == ""
    assert decorations.lost(
        callable_data, "glsl", source.replace("    executeCallableEXT(0u, 0);\n", "")
    ) == ("the translation loses OpExecuteCallableKHR")
    attributes = Feature(
        "glsl",
        "graphics",
        "ray-custom-attributes",
        None,
        group="StorageClass.HitAttributeKHR OpTypeStruct",
        stage="closest-hit",
    )
    source = (
        "#version 460\n#extension GL_EXT_ray_tracing : require\n"
        "struct Payload {\n    vec4 color;\n};\nlayout(location = 0) rayPayloadInEXT Payload payload;\n"
        "struct Attributes {\n    vec3 normal;\n};\nhitAttributeEXT Attributes attributes;\n"
        "void main() {\n    payload.color = vec4(attributes.normal, 1.0);\n}\n"
    )
    assert decorations.lost(attributes, "glsl", source) == ""
    barycentrics = source.replace(
        "hitAttributeEXT Attributes attributes;",
        "hitAttributeEXT vec2 uv;\nAttributes attributes;",
    )
    assert decorations.lost(attributes, "glsl", barycentrics) == (
        "the translation loses StorageClass.HitAttributeKHR"
    )


def _resource(name, group):
    return Feature("spirv", "resource", name, None, group=group)


def test_a_binding_translation_keeps_its_push_constants_and_descriptor_set():
    # A storage class is no decoration, so push constants are kept only by a variable in push-constant storage.
    push = _resource("push-constant", "StorageClass.PushConstant")
    variable = "%pc = OpVariable %_ptr_PushConstant_Constants PushConstant\n"
    assert decorations.lost(push, "spirv", variable) == ""
    uniform = "%pc = OpVariable %_ptr_Uniform_Constants Uniform\n"
    assert decorations.lost(push, "spirv", uniform) == (
        "the translation loses StorageClass.PushConstant"
    )
    # Every buffer is decorated with its set, so only a set other than the first keeps the probe's.
    bound = _resource("descriptor-set", "Decoration.DescriptorSet")
    first = "OpDecorate %fin DescriptorSet 0\nOpDecorate %params DescriptorSet 0\n"
    assert decorations.lost(bound, "spirv", first) == (
        "the translation loses Decoration.DescriptorSet"
    )
    second = first.replace("%params DescriptorSet 0", "%params DescriptorSet 1")
    assert decorations.lost(bound, "spirv", second) == ""
    assert decorations.lost(bound, "spirv", second.replace("Set 1", "Set 10")) == ""


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_push_constants_and_a_second_descriptor_set():
    head = (
        "#version 460\nlayout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        "layout(std430, binding = 0) buffer B0 { float fin[]; };\n"
        "layout(std430, binding = 1) buffer B1 { float outp[]; };\n"
    )
    body = "void main() {\n    outp[0] = fin[0] * scale;\n}\n"
    push = Feature(
        "glsl", "resource", "push-constant", None, group="StorageClass.PushConstant"
    )
    pushed = (
        head
        + "layout(push_constant) uniform Constants {\n    float scale;\n};\n"
        + body
    )
    assert decorations.lost(push, "glsl", pushed) == ""
    uniform = pushed.replace("push_constant", "std140, binding = 3")
    assert decorations.lost(push, "glsl", uniform) == (
        "the translation loses StorageClass.PushConstant"
    )
    bound = Feature(
        "glsl", "resource", "descriptor-set", None, group="Decoration.DescriptorSet"
    )
    assert decorations.lost(bound, "glsl", uniform) == (
        "the translation loses Decoration.DescriptorSet"
    )
    second = uniform.replace("std140, binding = 3", "std140, set = 1, binding = 3")
    assert decorations.lost(bound, "glsl", second) == ""


_ORDERED = (
    "ExecutionMode.PixelInterlockOrderedEXT Capability.FragmentShaderPixelInterlockEXT"
    " OpBeginInvocationInterlockEXT"
)


def test_a_rasterizer_ordered_translation_keeps_its_interlocked_region(monkeypatch):
    ordered = Feature(
        "spirv", "graphics", "rasterizer-ordered-image", None, group=_ORDERED
    )
    interlocked = (
        "OpCapability FragmentShaderPixelInterlockEXT\n"
        "OpExecutionMode %main PixelInterlockOrderedEXT\n"
        "OpBeginInvocationInterlockEXT\nOpEndInvocationInterlockEXT\n"
    )
    assert decorations.lost(ordered, "spirv", interlocked) == ""
    unordered = interlocked.replace("OpBeginInvocationInterlockEXT\n", "")
    assert decorations.lost(ordered, "spirv", unordered) == (
        "the translation loses OpBeginInvocationInterlockEXT"
    )
    # Metal orders the accesses to an argument in a raster order group, and has no capability to check.
    monkeypatch.setattr(validators, "air", lambda source: source)
    argument = '!15 = !{i32 2, !"air.texture", !"air.raster_order_group", i32 0}\n'
    assert decorations.lost(ordered, "metal", argument) == ""
    assert decorations.lost(
        ordered, "metal", argument.replace("raster_order_group", "read_write")
    ) == ("the translation loses ExecutionMode.PixelInterlockOrderedEXT")


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_a_pixel_interlocked_region():
    ordered = Feature(
        "glsl",
        "graphics",
        "rasterizer-ordered-image",
        None,
        group=_ORDERED,
        stage="fragment",
    )
    source = (
        "#version 460\n#extension GL_ARB_fragment_shader_interlock : require\n"
        "layout(pixel_interlock_ordered) in;\n"
        "layout(rgba32f, binding = 0) uniform image2D accum;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n    ivec2 p = ivec2(gl_FragCoord.xy);\n"
        "    beginInvocationInterlockARB();\n"
        "    imageStore(accum, p, imageLoad(accum, p) + vec4(1.0));\n"
        "    endInvocationInterlockARB();\n    color = vec4(1.0);\n}\n"
    )
    assert decorations.lost(ordered, "glsl", source) == ""
    unordered = (
        source.replace("layout(pixel_interlock_ordered) in;\n", "")
        .replace("    beginInvocationInterlockARB();\n", "")
        .replace("    endInvocationInterlockARB();\n", "")
    )
    lost = decorations.lost(ordered, "glsl", unordered)
    assert "OpBeginInvocationInterlockEXT" in lost
    assert "ExecutionMode.PixelInterlockOrderedEXT" in lost


_ARRAY = "OpTypeArray OpTypeImage Dim.2D"


def test_an_array_of_textures_keeps_its_images_as_array_elements(monkeypatch):
    textures = Feature("spirv", "resource", "array-of-textures", None, group=_ARRAY)
    image = "%img = OpTypeImage %float 2D 2 0 0 1 Unknown\n"
    # HLSL's Texture2D tex[2] is an array of images, GLSL's sampler2D tex[2] one of sampled images.
    for elements in (
        image + "%arr = OpTypeArray %img %uint_2\n",
        image + "%si = OpTypeSampledImage %img\n%arr = OpTypeArray %si %uint_2\n",
    ):
        assert decorations.lost(textures, "spirv", elements) == ""
    # Two separate textures keep the shape but not the array.
    separate = (
        image + "%si = OpTypeSampledImage %img\n%arr = OpTypeArray %float %uint_2\n"
    )
    assert (
        decorations.lost(textures, "spirv", separate) == "the translation loses Dim.2D"
    )
    # Metal passes an array of textures as an array of texture pointers.
    monkeypatch.setattr(validators, "air", lambda source: source)
    argument = "define void @probe([2 x %struct._texture_2d_t addrspace(1)*]* %3)\n"
    assert decorations.lost(textures, "metal", argument) == ""
    single = "define void @probe(%struct._texture_2d_t addrspace(1)* %3)\n"
    assert decorations.lost(textures, "metal", single) == "the translation loses Dim.2D"
    # A texture that is not an array keeps the plain shape.
    helper = dataclasses.replace(textures, name="texture-in-helper")
    assert decorations.lost(helper, "metal", single) == ""


def test_an_unsized_array_of_textures_stays_unsized(monkeypatch):
    unsized = Feature(
        "spirv",
        "resource",
        "unsized-array-of-textures",
        None,
        group="Capability.RuntimeDescriptorArray OpTypeRuntimeArray Dim.2D",
    )
    image = "%img = OpTypeImage %float 2D 2 0 0 1 Unknown\n"
    runtime = "OpCapability RuntimeDescriptorArray\n" + image
    assert decorations.lost(unsized, "spirv", runtime) == ""
    assert decorations.lost(unsized, "spirv", image) == (
        "the translation loses Capability.RuntimeDescriptorArray"
    )
    # Metal reaches the array through an argument buffer, a structure that holds the texture; a sized array of
    # textures wraps each texture too, but in no structure of its own.
    monkeypatch.setattr(validators, "air", lambda source: source)
    texture = (
        '%"struct.metal::texture2d" = type { %struct._texture_2d_t addrspace(1)* }\n'
    )
    slot = '%struct.Slot = type { %"struct.metal::texture2d" }\n'
    assert decorations.lost(unsized, "metal", slot + texture) == ""
    assert decorations.lost(unsized, "metal", texture) == (
        "the translation loses Capability.RuntimeDescriptorArray"
    )


def test_a_storage_image_translation_keeps_its_texel_format():
    unorm = Feature(
        "spirv",
        "resource",
        "unorm-storage-image",
        None,
        group="OpImageWrite ImageFormat.Rgba8 Dim.2D",
    )
    image = "%img = OpTypeImage %float 2D 0 0 0 2 {}\n"
    assert decorations.lost(unorm, "spirv", image.format("Rgba8")) == ""
    assert decorations.lost(unorm, "spirv", image.format("Rgba32f")) == (
        "the translation loses ImageFormat.Rgba8"
    )
    # A format that only begins with the wanted one is another format.
    assert decorations.lost(unorm, "spirv", image.format("Rgba8Snorm")) == (
        "the translation loses ImageFormat.Rgba8"
    )
    # DXC writes Rgba32f for any float4 storage texture, so HLSL keeps the format in its declaration.
    assert decorations.lost(unorm, "hlsl", "RWTexture2D<unorm float4> img;\n") == ""
    assert (
        decorations.lost(
            unorm, "hlsl", '[[vk::image_format("rgba8")]] RWTexture2D<float4> img;\n'
        )
        == ""
    )
    assert decorations.lost(unorm, "hlsl", "RWTexture2D<float4> img;\n") == (
        "the translation loses ImageFormat.Rgba8"
    )
    # An OpenCL C kernel names no format, so the HLSL CrossTL writes from it is not checked for one.
    plain = "RWTexture2D<float4> img;\n"
    assert decorations.lost(unorm, "hlsl", plain, imported="opencl") == ""
    assert decorations.lost(unorm, "hlsl", plain, imported="hlsl") == (
        "the translation loses ImageFormat.Rgba8"
    )


def test_a_shared_sampler_translation_declares_the_sampler_apart():
    shared = Feature(
        "spirv",
        "resource",
        "shared-sampler",
        None,
        group="OpTypeSampler OpSampledImage Dim.2D",
    )
    image = "%img = OpTypeImage %float 2D 0 0 0 1 Unknown\n"
    assert decorations.lost(shared, "spirv", image + "%smp = OpTypeSampler\n") == ""
    combined = image + "%si = OpTypeSampledImage %img\n"
    assert decorations.lost(shared, "spirv", combined) == (
        "the translation loses OpTypeSampler"
    )
