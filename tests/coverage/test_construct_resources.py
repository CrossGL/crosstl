"""Probes for bindings, textures, images, memory qualifiers, specialization constants and subgroups."""

from tools.coverage import constructs, probes
from tools.coverage.templates import (
    bindings,
    images,
    qualifiers,
    specializations,
    storage,
    subgroups,
    textures,
)

from .construct_grammar import GRAMMAR, construct_feature


def test_binding_probes_push_constants_and_bind_a_block_in_another_set():
    feature = construct_feature("glsl", "push-constant")
    assert (feature.category, feature.target) == ("resource", "vulkan")
    assert feature.group == "StorageClass.PushConstant"
    assert construct_feature("hlsl", "push-constant-offset").group == (
        "StorageClass.PushConstant Decoration.Offset"
    )
    assert (
        construct_feature("wgsl", "descriptor-set").group == "Decoration.DescriptorSet"
    )
    assert construct_feature("glsl", "descriptor-set").target == "vulkan"
    for language, name, line in (
        ("glsl", "push-constant", "layout(push_constant) uniform Constants {\n"),
        ("glsl", "push-constant-offset", "    layout(offset = 16) float scale;\n"),
        (
            "glsl",
            "descriptor-set",
            "layout(std140, set = 1, binding = 3) uniform Params {\n",
        ),
        (
            "hlsl",
            "push-constant",
            "[[vk::push_constant]] ConstantBuffer<Constants> pc;\n",
        ),
        ("hlsl", "push-constant-offset", "    [[vk::offset(16)]] float scale;\n"),
        ("slang", "descriptor-set", "cbuffer Params : register(b3, space1) {\n"),
        (
            "wgsl",
            "descriptor-set",
            "@group(1) @binding(3) var<uniform> params: Params;\n",
        ),
        ("rust", "push-constant", "#[spirv(push_constant)] pc: &Constants"),
        (
            "rust",
            "descriptor-set",
            "#[spirv(uniform, descriptor_set = 1, binding = 3)] scale: &f32",
        ),
        ("glsl", "push-constant", "    outp[i] = fin[i] * pc.scale + pc.bias;\n"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    for name, line in (
        ("push-constant", "    cbuffer Constants @push_constant {\n"),
        ("push-constant-offset", "        float scale @offset(16);\n"),
        ("descriptor-set", "    cbuffer Params @set(1) @binding(3) {\n"),
    ):
        assert (
            line in probes.build(construct_feature("glsl", name), "crossgl").source
        ), name
    # Metal binds every argument at an index of one table per resource kind, WGSL has no push-constant address
    # space, rust-gpu places no push-constant member, and GLSL ES 3.00 and the kernel languages have neither.
    names = set(bindings.NAMES)
    for language in ("glsl", "hlsl", "slang"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names <= listed, language
    assert {f.name for f in constructs.features("wgsl", GRAMMAR)} & names == {
        "descriptor-set"
    }
    assert {f.name for f in constructs.features("rust", GRAMMAR)} & names == {
        "push-constant",
        "descriptor-set",
    }
    for language in ("metal", "webgl", "cuda", "hip", "opencl", "mojo"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert not listed & names, language


def test_image_probes_pass_sample_and_index_textures_and_update_an_image_in_place():
    assert (
        construct_feature("glsl", "array-of-textures").group
        == "OpTypeArray OpTypeImage Dim.2D"
    )
    assert construct_feature("hlsl", "read-write-image").category == "resource"
    for language, name, line in (
        ("glsl", "texture-in-helper", "float lookup(float x) {\n"),
        ("glsl", "texture-argument", "float lookup(sampler2D t, float x) {\n"),
        (
            "glsl",
            "array-of-textures",
            "layout(binding = 3) uniform sampler2D tex[2];\n",
        ),
        (
            "glsl",
            "read-write-image",
            "layout(r32f, binding = 4) uniform image2D img;\n",
        ),
        ("webgl", "texture-argument", "float lookup(highp sampler2D t, float x) {\n"),
        ("webgl", "array-of-textures", "uniform highp sampler2D tex[2];\n"),
        (
            "hlsl",
            "texture-argument",
            "float lookup(Texture2D<float4> t, SamplerState s, float x) {\n",
        ),
        ("hlsl", "array-of-textures", "Texture2D<float4> tex[2] : register(t1);\n"),
        ("slang", "read-write-image", "RWTexture2D<float> img;\n"),
        ("metal", "array-of-textures", "array<texture2d<float>, 2> tex [[texture(0)]]"),
        ("metal", "read-write-image", "texture2d<float, access::read_write> img"),
        (
            "wgsl",
            "texture-argument",
            "fn lookup(t: texture_2d<f32>, s: sampler, x: f32)",
        ),
        ("wgsl", "read-write-image", "texture_storage_2d<r32float, read_write>;\n"),
        ("cuda", "texture-in-helper", "__device__ float lookup(cudaTextureObject_t t"),
        ("hip", "read-write-image", "hipSurfaceObject_t img"),
        ("opencl", "read-write-image", "__read_write image2d_t img"),
        ("rust", "array-of-textures", "tex: &[Image!(2D, type=f32, sampled); 2]"),
        ("glsl", "array-of-textures-loop-index", "    acc += textureLod(tex[k], "),
        (
            "glsl",
            "array-of-textures-nonuniform-index",
            "#extension GL_EXT_nonuniform_qualifier : require\n",
        ),
        ("glsl", "array-of-textures-nonuniform-index", "tex[nonuniformEXT(k)]"),
        (
            "hlsl",
            "array-of-textures-nonuniform-index",
            "tex[NonUniformResourceIndex(k)]",
        ),
        (
            "metal",
            "array-of-textures-nonuniform-index",
            "uint k = uint(fin[i]) & 1u;\n",
        ),
        ("rust", "array-of-textures-loop-index", "for k in 0..2 {\n"),
        (
            "glsl",
            "unsized-array-of-textures",
            "layout(binding = 3) uniform sampler2D tex[];\n",
        ),
        ("glsl", "unsized-array-of-textures", "int k = int(fin[0]) & 1;\n"),
        (
            "hlsl",
            "unsized-array-of-textures",
            "Texture2D<float4> tex[] : register(t1);\n",
        ),
        (
            "metal",
            "unsized-array-of-textures",
            "device const Slot* slots [[buffer(2)]]",
        ),
        (
            "rust",
            "unsized-array-of-textures",
            "tex: &RuntimeArray<Image!(2D, type=f32, sampled)>",
        ),
        (
            "glsl",
            "array-of-images",
            "layout(r32f, binding = 4) uniform image2D img[2];\n",
        ),
        ("hlsl", "array-of-images", "RWTexture2D<float> img[2] : register(u3);\n"),
        (
            "metal",
            "array-of-images",
            "array<texture2d<float, access::read_write>, 2> img",
        ),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "array-of-textures"), "crossgl"
    ).source
    assert "        uniform sampler2D tex[2];\n" in crossgl
    # CrossGL spells the non-uniform index as GLSL does, with no extension to enable.
    crossgl = probes.build(
        construct_feature("glsl", "array-of-textures-nonuniform-index"), "crossgl"
    ).source
    assert "tex[nonuniformEXT(k)]" in crossgl and "#extension" not in crossgl
    # A helper samples a program-scope texture by name where the language has them, and takes it as a parameter
    # where textures are entry-point parameters.
    assert (
        "lookup(fin[i])" in probes.build(
            construct_feature("hlsl", "texture-in-helper"), "hlsl"
        ).source
    )
    assert (
        "lookup(tex, smp, fin[i])" in probes.build(
            construct_feature("metal", "texture-in-helper"), "metal"
        ).source
    )
    names = set(images.NAMES)
    for language in ("glsl", "hlsl", "slang", "metal"):
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names <= listed, language
    arrays = {name for name in names if "array-of-" in name}
    absent = {
        "wgsl": arrays,
        "webgl": {
            "read-write-image",
            "array-of-textures-loop-index",
            "array-of-textures-nonuniform-index",
            "unsized-array-of-textures",
            "array-of-images",
        },
        "cuda": arrays,
        "hip": arrays,
        "opencl": arrays,
        "rust": {"array-of-textures-nonuniform-index"},
        "mojo": names,
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert names & listed == names - missing, language


def test_subgroup_probes_read_each_languages_built_ins():
    feature = construct_feature("glsl", "subgroup-invocation-id")
    assert (feature.category, feature.stage) == ("compute", "compute")
    assert feature.group == "BuiltIn.SubgroupLocalInvocationId"
    for language, name, line in (
        ("glsl", "subgroup-size", "#extension GL_KHR_shader_subgroup_basic : require"),
        ("glsl", "subgroup-id", "outp[i] = float(gl_SubgroupID);"),
        ("glsl", "workgroup-size", "outp[i] = float(gl_WorkGroupSize.x);"),
        ("hlsl", "subgroup-invocation-id", "outp[i] = float(WaveGetLaneIndex());"),
        ("slang", "num-subgroups", "outp[i] = float(WaveGetNumWaves());"),
        ("slang", "workgroup-size", "outp[i] = float(WorkgroupSize().x);"),
        ("metal", "subgroup-size", "uint width [[threads_per_simdgroup]]"),
        ("metal", "subgroup-id", "uint group [[simdgroup_index_in_threadgroup]]"),
        ("metal", "workgroup-size", "uint3 size [[threads_per_threadgroup]]"),
        (
            "wgsl",
            "subgroup-invocation-id",
            "@builtin(subgroup_invocation_id) lane: u32",
        ),
        (
            "rust",
            "subgroup-invocation-id",
            "#[spirv(subgroup_local_invocation_id)] lane: u32",
        ),
        ("cuda", "subgroup-invocation-id", "outp[i] = float(index % warpSize);"),
        ("hip", "subgroup-invocation-id", "outp[i] = float(__lane_id());"),
        ("opencl", "num-subgroups", "outp[i] = (float)get_num_sub_groups();"),
        ("mojo", "subgroup-id", "outp[i] = Float32(warp_id())"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(construct_feature("glsl", "num-subgroups"), "crossgl").source
    assert "outp[i] = float(gl_NumSubgroups);" in crossgl
    # naga accepts the subgroup built-ins only without the enable directive it has not implemented.
    wgsl = probes.build(construct_feature("wgsl", "subgroup-size"), "wgsl").source
    assert "enable subgroups" not in wgsl
    # HLSL has no built-in for the waves in a group or the group's size, WGSL and Rust none for the group's size, and
    # WebGL 2 no compute stage.
    names = set(subgroups.NAMES)
    absent = {
        "hlsl": {"num-subgroups", "subgroup-id", "workgroup-size"},
        "wgsl": {"workgroup-size"},
        "rust": {"workgroup-size"},
        "webgl": names,
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_specialization_probes_declare_each_languages_pipeline_constants():
    feature = construct_feature("glsl", "specialization-constant")
    assert (feature.category, feature.stage) == ("compute", "compute")
    assert feature.group == "OpSpecConstant Decoration.SpecId"
    # Only GLSL for Vulkan has constant_id, so the GLSL probes are checked for Vulkan and the others for their usual
    # target.
    assert feature.target == "vulkan"
    assert construct_feature("slang", "specialization-constant").target == ""
    for language, name, line in (
        (
            "glsl",
            "specialization-constant",
            "layout(constant_id = 0) const int scale = 3;",
        ),
        (
            "glsl",
            "specialization-constant-expression",
            "const int doubled = scale * 2 + 1;",
        ),
        (
            "glsl",
            "specialization-workgroup-size",
            "layout(local_size_x_id = 0, local_size_y = 1",
        ),
        (
            "slang",
            "specialization-constant-bool",
            "[vk::constant_id(0)] const bool negate = true;",
        ),
        ("slang", "specialization-workgroup-size", "[numthreads(size, 1, 1)]"),
        (
            "metal",
            "specialization-constant",
            "constant int scale = is_function_constant_defined(scale_value) ? scale_value : 3;",
        ),
        (
            "metal",
            "specialization-constant-bool",
            "constant bool negate_value [[function_constant(0)]];",
        ),
        (
            "wgsl",
            "specialization-constant-expression",
            "override doubled: i32 = scale * 2 + 1;",
        ),
        (
            "wgsl",
            "specialization-workgroup-size",
            "@compute @workgroup_size(size, 1, 1)",
        ),
        (
            "rust",
            "specialization-constant",
            "#[spirv(spec_constant(id = 0, default = 3))] scale: u32",
        ),
        (
            "glsl",
            "specialization-constant-float",
            "layout(constant_id = 0) const float gain = 1.5;",
        ),
        (
            "glsl",
            "specialization-constant-vector",
            "const ivec2 steps = ivec2(scale, 1);",
        ),
        ("glsl", "specialization-array-size", "shared float tile[count];"),
        ("slang", "specialization-array-size", "groupshared float tile[count];"),
        ("wgsl", "specialization-constant-float", "override gain: f32 = 1.5;"),
        (
            "wgsl",
            "specialization-array-size",
            "var<workgroup> tile: array<f32, count>;",
        ),
        (
            "metal",
            "specialization-constant-vector",
            "constant int2 steps = int2(scale, 1);",
        ),
        (
            "metal",
            "specialization-constant-in-helper",
            "return x * float(scale) + 1.0;",
        ),
        (
            "rust",
            "specialization-constant-in-helper",
            "fn scaled(x: f32, scale: u32) -> f32",
        ),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "specialization-constant-bool"), "crossgl"
    ).source
    assert "const bool negate @constant_id(0) = true;" in crossgl
    crossgl = probes.build(
        construct_feature("glsl", "specialization-array-size"), "crossgl"
    ).source
    assert "const int count @constant_id(0) = 4;" in crossgl
    assert "shared float tile[count];" in crossgl
    # HLSL for Direct3D, GLSL ES 3.00 and the kernel languages have no specialization constants. Metal has no
    # specialized threadgroup size or array length, WGSL and Slang no vector computed from a constant, and rust-gpu
    # only u32 parameters.
    names = set(specializations.NAMES)
    vector = {"specialization-constant-vector"}
    absent = {
        "glsl": set(),
        "slang": vector,
        "wgsl": vector,
        "metal": {"specialization-workgroup-size", "specialization-array-size"},
        "rust": (
            names - {"specialization-constant", "specialization-constant-in-helper"}
        ),
        **dict.fromkeys(("hlsl", "webgl", "cuda", "hip", "opencl", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_qualifier_probes_give_their_buffers_each_languages_memory_qualifiers():
    feature = construct_feature("glsl", "coherent-buffer")
    assert (feature.category, feature.stage) == ("compute", "compute")
    assert feature.group == "StorageClass.StorageBuffer Decoration.Coherent"
    for language, name, line in (
        ("glsl", "restrict-buffer", "layout(std430, binding = 0) restrict buffer B0"),
        (
            "glsl",
            "write-only-buffer",
            "layout(std430, binding = 1) writeonly buffer B1",
        ),
        ("hlsl", "coherent-buffer", "globallycoherent RWStructuredBuffer<float> outp"),
        (
            "slang",
            "coherent-buffer",
            "globallycoherent RWStructuredBuffer<float> outp;",
        ),
        ("metal", "restrict-buffer", "device float* __restrict outp [[buffer(1)]]"),
        ("metal", "coherent-buffer", "coherent(device) device float* outp"),
        ("metal", "volatile-buffer", "volatile device float* outp"),
        (
            "cuda",
            "restrict-buffer",
            "float* __restrict__ fin, float* __restrict__ outp",
        ),
        ("hip", "volatile-buffer", "float* fin, volatile float* outp"),
        ("opencl", "restrict-buffer", "__global float* restrict outp"),
        ("opencl", "volatile-buffer", "__global volatile float* outp"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "volatile-buffer"), "crossgl"
    ).source
    assert "volatile buffer float* outp;" in crossgl
    # HLSL and Slang have only coherent buffers, Metal no write-only device memory, the kernel languages no coherent
    # or write-only buffers, and WGSL, GLSL ES 3.00, rust-gpu and Mojo no buffer memory qualifiers.
    names = set(qualifiers.NAMES)
    kernels = {"restrict-buffer", "volatile-buffer"}
    absent = {
        "glsl": set(),
        "hlsl": names - {"coherent-buffer"},
        "slang": names - {"coherent-buffer"},
        "metal": {"write-only-buffer"},
        **dict.fromkeys(("cuda", "hip", "opencl"), names - kernels),
        **dict.fromkeys(("wgsl", "webgl", "rust", "mojo"), names),
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_texture_probes_read_each_shape_of_texture_per_language():
    feature = construct_feature("glsl", "texture-1d")
    assert (feature.category, feature.stage) == ("resource", "compute")
    assert feature.group == "OpTypeImage Dim.1D Capability.Sampled1D"
    assert (
        construct_feature("hlsl", "texture-cube-array").group == "OpTypeImage Dim.Cube"
    )
    for language, name, line in (
        ("glsl", "texture-1d", "layout(binding = 3) uniform sampler1D tex;"),
        (
            "glsl",
            "texture-cube-array",
            "textureLod(tex, vec4(fin[i], 0.5, 1.0, 1.0), 0.0)",
        ),
        ("glsl", "texel-buffer", "texelFetch(tex, int(fin[i]))"),
        ("webgl", "texture-2d-array", "uniform highp sampler2DArray tex;"),
        ("hlsl", "texture-cube", "TextureCube<float4> tex : register(t1);"),
        ("hlsl", "texel-buffer", "tex.Load(int(fin[i]))"),
        ("slang", "texture-3d", "Texture3D<float4> tex;"),
        ("metal", "texture-1d", "tex.sample(smp, fin[i]).x"),
        ("metal", "texture-2d-array", "texture2d_array<float> tex [[texture(0)]]"),
        ("metal", "texel-buffer", "texture_buffer<float> tex [[texture(0)]]"),
        ("wgsl", "texture-1d", "textureLoad(tex, i32(fin[i]), 0)"),
        ("wgsl", "texture-cube-array", "var tex: texture_cube_array<f32>;"),
        (
            "cuda",
            "texture-2d-array",
            "tex2DLayeredLod<float4>(tex, fin[i], 0.5f, 1, 0.0f)",
        ),
        ("hip", "texel-buffer", "tex1Dfetch<float4>(tex, int(fin[i]))"),
        ("opencl", "texel-buffer", "__read_only image1d_buffer_t tex"),
        ("rust", "texture-cube-array", "&Image!(cube, type=f32, sampled, arrayed)"),
        ("rust", "texel-buffer", "OpImageFetch typeof*{color} %image %coordinate"),
        ("glsl", "texture-1d-array", "textureLod(tex, vec2(fin[i], 1.0), 0.0)"),
        ("hlsl", "texture-1d-array", "Texture1DArray<float4> tex : register(t1);"),
        ("cuda", "texture-1d-array", "tex1DLayeredLod<float4>(tex, fin[i], 1, 0.0f)"),
        ("opencl", "texture-1d-array", "__read_only image1d_array_t tex"),
        ("rust", "texture-1d-array", "OpImageSampleExplicitLod typeof*{color}"),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(construct_feature("glsl", "texture-cube"), "crossgl").source
    assert "uniform samplerCube tex;" in crossgl
    assert construct_feature("glsl", "texture-1d-array").group == (
        "OpTypeImage Dim.1D Capability.Sampled1D"
    )
    # WGSL has no buffer or 1D array textures, OpenCL C no cube images, GLSL ES 3.00 no 1D, 1D-array, cube-array or
    # buffer textures, and Mojo's kernels no textures.
    names = set(textures.NAMES)
    absent = {
        **dict.fromkeys(
            ("glsl", "hlsl", "slang", "metal", "cuda", "hip", "rust"), set()
        ),
        "wgsl": {"texel-buffer", "texture-1d-array"},
        "opencl": {"texture-cube", "texture-cube-array"},
        "webgl": {
            "texture-1d",
            "texture-1d-array",
            "texture-cube-array",
            "texel-buffer",
        },
        "mojo": names,
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language


def test_storage_probes_write_each_shape_of_storage_image_per_language():
    feature = construct_feature("glsl", "storage-image-1d")
    assert (feature.category, feature.stage) == ("resource", "compute")
    assert feature.group == "OpImageWrite Dim.1D Capability.Image1D"
    assert construct_feature("hlsl", "storage-texel-buffer").group == (
        "OpImageWrite Dim.Buffer Capability.ImageBuffer"
    )
    for language, name, line in (
        (
            "glsl",
            "storage-image-1d",
            "layout(rgba32f, binding = 4) uniform writeonly image1D img;",
        ),
        (
            "glsl",
            "storage-image-array",
            "imageStore(img, ivec3(0, 0, 1), vec4(fin[i]));",
        ),
        ("hlsl", "storage-texel-buffer", "RWBuffer<float4> img : register(u3);"),
        ("slang", "storage-image-3d", "img[uint3(0, 0, 0)] = float4("),
        ("metal", "storage-image-array", "img.write(float4(fin[i]), uint2(0, 0), 1);"),
        ("metal", "storage-texel-buffer", "texture_buffer<float, access::write> img"),
        (
            "wgsl",
            "storage-image-3d",
            "var img: texture_storage_3d<rgba32float, write>;",
        ),
        ("cuda", "storage-image-array", "surf2DLayeredwrite(make_float4("),
        ("hip", "storage-image-1d", "hipSurfaceObject_t img"),
        ("opencl", "storage-texel-buffer", "__write_only image1d_buffer_t img"),
        (
            "rust",
            "storage-image-array",
            "&Image!(2D, arrayed, format = rgba32f, sampled = false)",
        ),
    ):
        probe = probes.build(construct_feature(language, name), language)
        assert line in probe.source, (language, name)
    crossgl = probes.build(
        construct_feature("glsl", "storage-image-3d"), "crossgl"
    ).source
    assert "layout(rgba32f) uniform writeonly image3D img;" in crossgl
    # WGSL, CUDA and HIP have no storage texel buffers, and GLSL ES 3.00 and Mojo's kernels no storage images.
    names = set(storage.NAMES)
    absent = {
        **dict.fromkeys(("glsl", "hlsl", "slang", "metal", "opencl", "rust"), set()),
        **dict.fromkeys(("wgsl", "cuda", "hip"), {"storage-texel-buffer"}),
        "webgl": names,
        "mojo": names,
    }
    for language, missing in absent.items():
        listed = {f.name for f in constructs.features(language, GRAMMAR)}
        assert listed & names == names - missing, language
