import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen
from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen

FRAGMENT_SMOKE_SHADER = """
shader ExternalValidatorSmoke {
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
}
"""


CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER = """
shader ExternalValidatorSynchronization {
    compute {
        void main() {
            barrier();
            memoryBarrier();
            workgroupBarrier();
        }
    }
}
"""


CROSSGL_WAVE_QUAD_COMPUTE_SHADER = """
shader ExternalValidatorWaveQuad {
    RWBuffer<uint> outputValues @register(u0);

    compute {
        @numthreads(4, 1, 1)
        void main(uvec3 tid @gl_GlobalInvocationID) {
            uint lane = WaveGetLaneIndex();
            uint laneCount = WaveGetLaneCount();
            bool firstLane = WaveIsFirstLane();
            uint value = lane + tid.x;
            uint sumValue = WaveActiveSum(value);
            uint prefixValue = WavePrefixSum(sumValue);
            bool anyLane = WaveActiveAnyTrue(prefixValue >= lane);
            bool allLane = WaveActiveAllTrue(laneCount > 0u);
            uvec4 ballot = WaveActiveBallot(anyLane || allLane || firstLane);
            uvec4 matchMask = WaveMatch(prefixValue);
            uint broadcast = WaveReadLaneAt(prefixValue, 0u);
            uint firstValue = WaveReadLaneFirst(broadcast + ballot.x + matchMask.x);
            uint quadX = QuadReadAcrossX(firstValue);
            uint quadLane = QuadReadLaneAt(value, 3u);
            bool quadAny = QuadAny(anyLane);
            bool quadAll = QuadAll(allLane);
            outputValues[tid.x] = quadX + quadLane
                + (quadAny ? 1u : 0u) + (quadAll ? 1u : 0u);
        }
    }
}
"""


CROSSGL_WGSL_GRAPHICS_SHADER = """
shader ExternalValidatorWGSLGraphics {
    struct VSInput {
        vec3 position @ POSITION;
        vec2 uv @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.position = vec4(input.position, 1.0);
            output.uv = input.uv;
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            return vec4(input.uv, 0.0, 1.0);
        }
    }
}
"""


CROSSGL_WGSL_RESOURCE_SHADER = """
shader ExternalValidatorWGSLResource {
    samplerCube envMap;
    vec4 sampleEnv(samplerCube tex, vec3 direction) {
        uint2 size = textureSize(tex, 0);
        return textureLod(tex, direction + vec3(float(size.x) * 0.0), 1.0);
    }
    fragment {
        vec4 main(vec3 normal @ NORMAL) @ gl_FragColor {
            return sampleEnv(envMap, normal);
        }
    }
}
"""


CROSSGL_WGSL_COMPUTE_SHADER = """
shader ExternalValidatorWGSLCompute {
    RWStructuredBuffer<float> outputValues;
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        void main(uint3 gid @ gl_GlobalInvocationID) {
            barrier();
            outputValues[gid.x] = float(gid.x);
            return;
        }
    }
}
"""


CROSSGL_WGSL_BUFFER_BLOCK_SHADER = """
shader ExternalValidatorWGSLBufferBlock {
    layout(std430, set = 1, binding = 2) readonly buffer InputBlock {
        float values[];
    } inputBlock;
    layout(std430, binding = 3) buffer OutputBlock {
        float values[];
    } outputBlock;
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        void main(uint3 gid @ gl_GlobalInvocationID) {
            outputBlock.values[gid.x] = inputBlock.values[gid.x];
            return;
        }
    }
}
"""


GLSL_SPECIALIZATION_CONSTANT_VERTEX_SHADER = """
#version 400

layout(constant_id = 16) const int arraySize = 5;
in vec4 ucol[arraySize];

layout(constant_id = 17) const bool spBool = true;
layout(constant_id = 18) const float spFloat = 3.14;
layout(constant_id = 19) const double spDouble = 3.1415926535897932384626433832795;
layout(constant_id = 22) const uint scale = 2;

layout(constant_id = 24) gl_MaxImageUnits;

out vec4 color;
out int size;

void foo(vec4 p[arraySize]);

void main()
{
    color = ucol[2];
    size = arraySize;
    if (spBool)
        color *= scale;
    color += float(spDouble / spFloat);

    foo(ucol);
}

layout(constant_id = 116) const int dupArraySize = 12;
in vec4 dupUcol[dupArraySize];

layout(constant_id = 117) const bool spDupBool = true;
layout(constant_id = 118) const float spDupFloat = 3.14;
layout(constant_id = 119) const double spDupDouble = 3.1415926535897932384626433832795;
layout(constant_id = 122) const uint dupScale = 2;

void foo(vec4 p[arraySize])
{
    color += dupUcol[2];
    size += dupArraySize;
    if (spDupBool)
        color *= dupScale;
    color += float(spDupDouble / spDupFloat);
}

int builtin_spec_constant()
{
    int result = gl_MaxImageUnits;
    return result;
}
"""
CROSSGL_TEXTURE_RESOURCE_FRAGMENT_SHADER = """
shader ExternalValidatorTextureResources {
    sampler2D colorMap @register(t0);
    sampler linearSampler @register(s0);

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec2 ddxValue @ TEXCOORD1;
        vec2 ddyValue @ TEXCOORD2;
    };

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            int lod = 0;
            ivec2 pixel = ivec2(input.uv * 16.0);
            ivec2 offset = ivec2(1, -1);
            int component = int(input.uv.x);
            vec4 base = texture(colorMap, linearSampler, input.uv);
            vec4 biased = texture(colorMap, linearSampler, input.uv, 0.25);
            vec4 level = textureLod(colorMap, linearSampler, input.uv, lod);
            vec4 grad = textureGrad(
                colorMap,
                linearSampler,
                input.uv,
                input.ddxValue,
                input.ddyValue
            );
            vec4 offsetSample = textureOffset(
                colorMap,
                linearSampler,
                input.uv,
                offset
            );
            vec4 fetched = texelFetch(colorMap, pixel, lod);
            vec4 fetchedOffset = texelFetchOffset(colorMap, pixel, lod, offset);
            vec4 gathered = textureGather(
                colorMap,
                linearSampler,
                input.uv,
                component
            );
            vec4 gatheredOffset = textureGatherOffset(
                colorMap,
                linearSampler,
                input.uv,
                offset,
                1
            );
            ivec2 size = textureSize(colorMap, lod);
            int levels = textureQueryLevels(colorMap);
            vec2 lodInfo = textureQueryLod(colorMap, linearSampler, input.uv);
            float scalar = float(size.x + size.y + levels) + lodInfo.x + lodInfo.y;
            return base + biased + level + grad + offsetSample + fetched
                + fetchedOffset + gathered + gatheredOffset
                + vec4(scalar * 0.0001);
        }
    }
}
"""


CROSSGL_SHADOW_TEXTURE_FRAGMENT_SHADER = """
shader ExternalValidatorShadowTextures {
    sampler2DShadow shadowMap @register(t1, space2);
    sampler compareSampler @register(s0, space2);

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec3 projected @ TEXCOORD1;
        float depth @ TEXCOORD2;
    };

    float sampleShadow(sampler2DShadow tex, sampler cmp, vec2 uv, float depth) {
        return textureCompare(tex, cmp, uv, depth);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            ivec2 offset = ivec2(1, 0);
            float base = textureCompare(
                shadowMap,
                compareSampler,
                input.uv,
                input.depth
            );
            float offsetCmp = textureCompareOffset(
                shadowMap,
                compareSampler,
                input.uv,
                input.depth,
                offset
            );
            float projected = textureCompareProj(
                shadowMap,
                compareSampler,
                input.projected,
                input.depth
            );
            float projectedOffset = textureCompareProjOffset(
                shadowMap,
                compareSampler,
                input.projected,
                input.depth,
                offset
            );
            vec4 gathered = textureGatherCompare(
                shadowMap,
                compareSampler,
                input.uv,
                input.depth
            );
            vec4 gatheredOffset = textureGatherCompareOffset(
                shadowMap,
                compareSampler,
                input.uv,
                input.depth,
                offset
            );
            vec2 lodInfo = textureQueryLod(shadowMap, input.uv);
            float helper = sampleShadow(
                shadowMap,
                compareSampler,
                input.uv,
                input.depth
            );
            float sum = base + offsetCmp + projected + projectedOffset
                + gathered.x + gatheredOffset.y + lodInfo.x + lodInfo.y
                + helper;
            return vec4(sum, sum, sum, 1.0);
        }
    }
}
"""


CROSSGL_TYPED_BUFFER_ATOMICS_COMPUTE_SHADER = """
shader ExternalValidatorTypedBufferAtomics {
    struct Counter {
        uint value;
        int signedValue;
    };

    RWBuffer<uint> counters @register(u1);
    RWStructuredBuffer<Counter> structuredCounters @register(u2);
    RWBuffer<uint> counterArrays[2] @register(u4);
    RWStructuredBuffer<int> signedCounters @register(u6);

    uint fetchAndAdd(uint index) {
        return atomicAdd(counters[index], 1u);
    }

    uint compareAndSwap(uint index) {
        return atomicCompareExchange(counters[index], 2u, 3u);
    }

    uint addWithBias(uint index) {
        return atomicAdd(counters[index], 5u) + 3u;
    }

    compute {
        @numthreads(1, 1, 1)
        void main(uvec3 tid @gl_GlobalInvocationID) {
            uint original = atomicAdd(counters[tid.x], 1u);
            original = atomicCompareExchange(counters[tid.x], 2u, 3u);
            atomicXor(counters[tid.x], 4u, original);
            atomicMin(structuredCounters[tid.x].signedValue, -1);
            atomicAdd(counterArrays[1][tid.x], 1u, original);
            int oldSigned = atomicMax(signedCounters[tid.x], -1);
            uint combined = atomicAdd(counters[tid.x], 5u)
                + atomicCompareExchange(counters[tid.x], 2u, 3u);
            original = atomicAdd(counterArrays[1][tid.x], 2u) + 4u;
            original = fetchAndAdd(tid.x) + compareAndSwap(tid.x);
            original += addWithBias(tid.x) + combined;
        }
    }
}
"""


HLSL_RASTERIZER_ORDERED_VALIDATOR_SHADER = """
shader HLSLRasterizerOrderedValidator {
    uimage2D pixelCounts @rasterizer_ordered @register(u0);
    image2DArray layers @rasterizer_ordered @register(u1);
    RWBuffer<uint> bins @rasterizer_ordered @register(u2);
    RWStructuredBuffer<int> values @rasterizer_ordered @register(u3);
    RWByteAddressBuffer rawBytes @rasterizer_ordered @register(u4);

    fragment {
        vec4 main(uvec2 pixel @ TEXCOORD0, uint layer @ TEXCOORD1) @ SV_Target0 {
            uint oldCount = imageAtomicAdd(pixelCounts, pixel, 1u);
            imageStore(
                layers,
                uvec3(pixel, layer),
                vec4(float(oldCount), 0.0, 0.0, 1.0)
            );
            uint oldBin = atomicAdd(bins[0], oldCount);
            int oldValue = atomicMax(values[0], int(oldBin));
            buffer_store(rawBytes, 0, oldBin);
            return imageLoad(layers, uvec3(pixel, layer));
        }
    }
}
"""


HLSL_DXR_LIBRARY_VALIDATOR_SHADER = """
shader HLSLDXRLibraryValidator {
    RaytracingAccelerationStructure accel @register(t0);

    struct Payload {
        vec3 color;
    };

    struct CallableData {
        uint value;
    };

    ray_generation {
        void main() {
            RayDesc ray;
            ray.Origin = vec3(0.0, 0.0, 0.0);
            ray.TMin = 0.001;
            ray.Direction = vec3(0.0, 0.0, 1.0);
            ray.TMax = 100.0;

            Payload payload;
            payload.color = vec3(0.0, 0.0, 0.0);
            TraceRay(accel, 0, 0xFF, 0, 1, 0, ray, payload);

            CallableData data;
            data.value = 0u;
            CallShader(0, data);
        }
    }

    ray_miss {
        void main(Payload payload @ payload) {
            payload.color = vec3(0.0, 0.0, 1.0);
        }
    }

    ray_closest_hit {
        void main(
            Payload payload @ payload,
            BuiltInTriangleIntersectionAttributes attributes @ hit_attribute
        ) {
            payload.color = vec3(attributes.barycentrics, 1.0);
        }
    }

    ray_any_hit {
        void main(
            Payload payload @ payload,
            BuiltInTriangleIntersectionAttributes attributes @ hit_attribute
        ) {
            IgnoreHit();
            AcceptHitAndEndSearch();
        }
    }

    ray_callable {
        void main(CallableData data @ callable_data) {
            data.value = data.value + 1u;
        }
    }
}
"""


HLSL_MESH_AMPLIFICATION_VALIDATOR_SHADER = """
shader HLSLMeshAmplificationValidator {
    struct MeshPayload {
        uint meshlet;
    };

    struct MeshVertex {
        vec4 position @ SV_Position;
        vec2 uv @ TEXCOORD0;
    };

    groupshared MeshPayload payload;

    task {
        void main() @numthreads(1, 1, 1) {
            payload.meshlet = 7u;
            DispatchMesh(1, 1, 1, payload);
        }
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[3],
            @indices out uvec3 tris[1]
        ) @numthreads(32, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            verts[0].uv = vec2(0.0, 0.0);
            verts[1].position = vec4(1.0, 0.0, 0.0, 1.0);
            verts[1].uv = vec2(1.0, 0.0);
            verts[2].position = vec4(0.0, 1.0, 0.0, 1.0);
            verts[2].uv = vec2(0.0, 1.0);
            tris[0] = uvec3(0u, 1u, 2u);
        }
    }
}
"""


HLSL_TESSELLATION_VALIDATOR_SHADER = """
shader HLSLTessellationValidator {
    struct HSInput {
        vec3 position @ POSITION;
    };

    struct HSOutput {
        vec3 position @ POSITION;
    };

    struct HSConstData {
        vec3 edges @ SV_TessFactor;
        float inside @ SV_InsideTessFactor;
    };

    tessellation_control {
        HSConstData HSConst(InputPatch<HSInput, 3> patch) {
            HSConstData constants;
            constants.edges[0] = 1.0;
            constants.edges[1] = 1.0;
            constants.edges[2] = 1.0;
            constants.inside = 1.0;
            return constants;
        }

        HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
            @domain(tri)
            @partitioning(integer)
            @outputtopology(triangle_cw)
            @outputcontrolpoints(3)
            @patchconstantfunc(HSConst) {
            HSOutput output;
            output.position = patch[id].position;
            return output;
        }
    }

    tessellation_evaluation {
        vec4 main(OutputPatch<HSOutput, 3> patch, vec3 bary @ SV_DomainLocation)
            @domain(tri) @ SV_Position {
            vec3 position = patch[0].position * bary.x
                + patch[1].position * bary.y
                + patch[2].position * bary.z;
            return vec4(position, 1.0);
        }
    }
}
"""


SLANG_TESSELLATION_VALIDATOR_SHADER = """
shader SlangTessellationValidator {
    struct VSOut {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };

    struct HSOut {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };

    struct PatchConstants {
        float outer[3] @ gl_TessLevelOuter;
        float inner[1] @ gl_TessLevelInner;
    };

    tessellation_control {
        PatchConstants HSConst(
            InputPatch<VSOut, 3> inputPatch,
            uint patchID @ gl_PrimitiveID
        ) {
            PatchConstants patch;
            VSOut first = gl_in[0];
            patch.outer[0] = first.position.x + float(patchID);
            patch.outer[1] = first.position.y;
            patch.outer[2] = first.position.z;
            patch.inner[0] = 1.0;
            return patch;
        }

        HSOut main(InputPatch<VSOut, 3> inputPatch)
            @domain(tri)
            @partitioning(integer)
            @outputtopology(triangle_cw)
            @outputcontrolpoints(3)
            @patchconstantfunc(HSConst) {
            HSOut output;
            VSOut current = gl_in[gl_InvocationID];
            output.position = current.position;
            output.uv = current.uv;
            return output;
        }
    }

    tessellation_evaluation {
        vec4 main(OutputPatch<HSOut, 3> patch, vec3 bary @ gl_TessCoord)
            @domain(tri) @ gl_Position {
            vec4 p0 = patch[0].position * bary.x;
            vec4 p1 = patch[1].position * bary.y;
            vec4 p2 = patch[2].position * bary.z;
            return p0 + p1 + p2;
        }
    }
}
"""


SLANG_RAY_STAGE_VALIDATOR_SHADER = """
shader SlangRayStageValidator {
    struct RayPayload {
        vec3 color;
    };

    struct HitAttributes {
        vec2 barycentrics;
    };

    struct CallableData {
        uint value;
    };

    ray_generation {
        void main() {
            uvec3 launch = gl_LaunchIDEXT;
            uint launchSizeX = gl_LaunchSizeEXT.x;
        }
    }

    ray_closest_hit {
        void main(
            RayPayload payload @ payload,
            HitAttributes attributes @ hit_attribute
        ) {
            payload.color = vec3(attributes.barycentrics, 1.0);
        }
    }

    ray_any_hit {
        void main(
            RayPayload payload @ payload,
            HitAttributes attributes @ hit_attribute
        ) {
            payload.color = vec3(attributes.barycentrics, 0.5);
            AcceptHitAndEndSearch();
        }
    }

    ray_miss {
        void main(RayPayload payload @ rayPayloadInEXT) {
            payload.color = vec3(0.0, 0.0, 0.0);
        }
    }

    ray_callable {
        void main(CallableData data @ callableDataInEXT) {
            data.value = data.value + 1u;
        }
    }

    ray_intersection {
        void main() {
            HitAttributes attributes;
            attributes.barycentrics = vec2(0.25, 0.75);
            bool accepted = ReportHit(1.0, 0, attributes);
        }
    }
}
"""


SLANG_MESH_TASK_VALIDATOR_SHADER = """
shader SlangMeshTaskValidator {
    struct MeshPayload {
        uint meshlet;
    };

    struct MeshVertex {
        vec4 position @ SV_Position;
        vec2 uv @ TEXCOORD0;
    };

    struct MeshPrimitive {
        bool culled @ SV_CullPrimitive;
    };

    groupshared MeshPayload payload;

    task {
        void main(uvec3 groupId @ gl_WorkGroupID) @numthreads(1, 1, 1) {
            payload.meshlet = groupId.x;
            DispatchMesh(1, 1, 1, payload);
        }
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            uvec3 threadId @ gl_LocalInvocationID,
            @vertices out MeshVertex verts[3],
            @indices out uvec3 tris[1],
            @primitives out MeshPrimitive prims[1]
        ) @numthreads(32, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            verts[1].position = vec4(float(threadId.x), 1.0, 0.0, 1.0);
            verts[2].position = vec4(0.0, 0.0, 1.0, 1.0);
            verts[0].uv = vec2(0.0, 0.0);
            verts[1].uv = vec2(1.0, 0.0);
            verts[2].uv = vec2(0.0, 1.0);
            tris[0] = uvec3(0u, 1u, 2u);
            prims[0].culled = false;
        }
    }
}
"""


GLSL_MULTISAMPLE_STORAGE_COMPUTE_SHADER = """
shader GLSLMultisampleStorageValidator {
    image2DMS colorImage @rgba16f;
    uimage2DMS counters @r32ui;
    image2DMSArray layered @rgba16f;

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            int sampleIndex = 2;
            vec4 oldColor = imageLoad(colorImage, pixel, sampleIndex);
            uint oldCount = imageLoad(counters, pixel, sampleIndex);
            imageStore(colorImage, pixel, sampleIndex, oldColor + vec4(1.0));
            imageStore(counters, pixel, sampleIndex, oldCount + 3u);
            uint atomicOld = imageAtomicAdd(counters, pixel, sampleIndex, 3u);
            uint exchanged = imageAtomicExchange(counters, pixel, sampleIndex, atomicOld + 1u);
            uint swapped = imageAtomicCompSwap(counters, pixel, sampleIndex, exchanged, oldCount);

            ivec3 pixelLayer = ivec3(2, 3, 1);
            vec4 oldLayer = imageLoad(layered, pixelLayer, 0);
            imageStore(layered, pixelLayer, 0, oldLayer + oldColor + vec4(float(swapped)));
        }
    }
}
"""


GLSL_CUBE_STORAGE_COMPUTE_SHADER = """
shader GLSLCubeStorageValidator {
    imageCube cube;
    imageCubeArray cubeArray @rgba16f;
    iimageCube signedCube @r32i;
    uimageCubeArray unsignedCubeArray @r32ui;

    compute {
        void main() {
            ivec3 cubeCoord = ivec3(0, 1, 2);
            ivec3 layerCoord = ivec3(3, 4, 5);
            vec4 oldCube = imageLoad(cube, cubeCoord);
            imageStore(cube, cubeCoord, oldCube + vec4(0.25));

            vec4 oldLayer = imageLoad(cubeArray, layerCoord);
            imageStore(cubeArray, layerCoord, oldLayer + oldCube);

            int signedOld = imageAtomicExchange(signedCube, cubeCoord, -1);
            imageStore(signedCube, cubeCoord, signedOld + 1);

            uint unsignedOld = imageAtomicAdd(unsignedCubeArray, layerCoord, 2u);
            imageStore(unsignedCubeArray, layerCoord, unsignedOld + 1u);
        }
    }
}
"""


GLSL_PARAMETER_IMAGE_ATOMIC_COMPUTE_SHADER = """
shader GLSLParameterImageAtomicValidator {
    uimage2D counters @r32ui;

    uint addCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
        return imageAtomicAdd(image, pixel, value);
    }

    compute {
        void main() {
            uint oldValue = addCounter(counters, ivec2(0, 1), 2u);
            imageStore(counters, ivec2(0, 1), oldValue);
        }
    }
}
"""


GLSL_ARRAY_ELEMENT_IMAGE_SPECIALIZATION_COMPUTE_SHADER = """
shader GLSLArrayElementImageSpecializationValidator {
    image2D counters @r32ui[2];

    int queryElement(image2D image @r32ui) {
        return imageSize(image).x;
    }

    int queryViaArray(image2D images[] @r32ui) {
        return queryElement(images[0]);
    }

    compute {
        void main() {
            int directCount = queryElement(counters[0]);
            int nestedCount = queryViaArray(counters);
            imageStore(counters[1], ivec2(0, 0), uint(directCount + nestedCount));
        }
    }
}
"""


GLSL_DYNAMIC_IMAGE_ARRAY_HELPER_COMPUTE_SHADER = """
shader GLSLDynamicImageArrayHelperValidator {
    image2D counters @r32ui[2];

    int queryElement(image2D image @r32ui) {
        return imageSize(image).x;
    }

    int queryViaDynamic(image2D images[] @r32ui, int layer) {
        return queryElement(images[layer]);
    }

    int queryViaInitializer(image2D images[] @r32ui, int layer) {
        int count = queryElement(images[layer]);
        return count;
    }

    int queryViaAssignment(image2D images[] @r32ui, int layer) {
        int count = 0;
        count = queryElement(images[layer]);
        return count;
    }

    void storeElement(image2D image @r32ui, ivec2 pixel, uint value) {
        imageStore(image, pixel, value);
    }

    void storeViaExpression(image2D images[] @r32ui, int layer) {
        storeElement(images[layer], ivec2(0, 0), uint(layer));
    }

    compute {
        void main() {
            int directCount = queryElement(counters[0]);
            int nestedCount = queryViaDynamic(counters, 1);
            int initializedCount = queryViaInitializer(counters, 0);
            int assignedCount = queryViaAssignment(counters, 1);
            storeViaExpression(counters, 1);
            imageStore(
                counters[1],
                ivec2(0, 0),
                uint(directCount + nestedCount + initializedCount + assignedCount)
            );
        }
    }
}
"""


GLSL_ADVANCED_IMAGE_ARRAY_SPECIALIZATION_COMPUTE_SHADER = """
shader GLSLAdvancedImageArraySpecializationValidator {
    image2DMS msImages @rgba16f[2];
    uimage2DMS msCounters @r32ui[2];
    imageCube cubeImages @rgba16f[2];
    imageCubeArray cubeLayerImages @rgba16f[2];

    vec4 touchMS(image2DMS image @rgba16f, ivec2 pixel, int sampleIndex, vec4 value) {
        vec4 oldValue = imageLoad(image, pixel, sampleIndex);
        imageStore(image, pixel, sampleIndex, oldValue + value);
        return oldValue;
    }

    uint bumpMS(uimage2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
        return imageAtomicAdd(image, pixel, sampleIndex, value);
    }

    vec4 touchCube(imageCube image @rgba16f, ivec3 coord, vec4 value) {
        vec4 oldValue = imageLoad(image, coord);
        imageStore(image, coord, oldValue + value);
        return oldValue;
    }

    vec4 touchCubeLayer(imageCubeArray image @rgba16f, ivec3 coord, vec4 value) {
        vec4 oldValue = imageLoad(image, coord);
        imageStore(image, coord, oldValue + value);
        return oldValue;
    }

    vec4 viaMS(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex, vec4 value) {
        return touchMS(images[1], pixel, sampleIndex, value);
    }

    uint viaCounter(uimage2DMS counters[] @r32ui, ivec2 pixel, int sampleIndex, uint value) {
        return bumpMS(counters[1], pixel, sampleIndex, value);
    }

    vec4 viaCube(imageCube images[] @rgba16f, ivec3 coord, vec4 value) {
        return touchCube(images[1], coord, value);
    }

    vec4 viaCubeLayer(imageCubeArray images[] @rgba16f, ivec3 coord, vec4 value) {
        return touchCubeLayer(images[1], coord, value);
    }

    compute {
        void main() {
            vec4 directMS = touchMS(msImages[0], ivec2(0, 1), 2, vec4(1.0));
            vec4 nestedMS = viaMS(msImages, ivec2(2, 3), 0, directMS);
            uint directCounter = bumpMS(msCounters[0], ivec2(1, 2), 3, 4u);
            uint nestedCounter = viaCounter(msCounters, ivec2(3, 4), 1, directCounter);
            vec4 directCube = touchCube(cubeImages[0], ivec3(0, 1, 2), nestedMS);
            vec4 nestedCube = viaCube(cubeImages, ivec3(2, 3, 4), directCube);
            vec4 directLayer = touchCubeLayer(cubeLayerImages[0], ivec3(1, 2, 3), nestedCube);
            vec4 nestedLayer = viaCubeLayer(cubeLayerImages, ivec3(3, 4, 5), directLayer + vec4(float(nestedCounter)));
        }
    }
}
"""


GLSL_DYNAMIC_ADVANCED_IMAGE_ARRAY_HELPER_COMPUTE_SHADER = """
shader GLSLDynamicAdvancedImageArrayHelperValidator {
    image2DMS msImages @rgba16f[2];
    uimage2DMS msCounters @r32ui[2];
    imageCube cubeImages @rgba16f[2];
    imageCubeArray cubeLayerImages @rgba16f[2];

    vec4 touchMS(image2DMS image @rgba16f, ivec2 pixel, int sampleIndex, vec4 value) {
        vec4 oldValue = imageLoad(image, pixel, sampleIndex);
        imageStore(image, pixel, sampleIndex, oldValue + value);
        return oldValue;
    }

    uint bumpMS(uimage2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
        return imageAtomicAdd(image, pixel, sampleIndex, value);
    }

    vec4 touchCube(imageCube image @rgba16f, ivec3 coord, vec4 value) {
        vec4 oldValue = imageLoad(image, coord);
        imageStore(image, coord, oldValue + value);
        return oldValue;
    }

    vec4 touchCubeLayer(imageCubeArray image @rgba16f, ivec3 coord, vec4 value) {
        vec4 oldValue = imageLoad(image, coord);
        imageStore(image, coord, oldValue + value);
        return oldValue;
    }

    vec4 viaMS(image2DMS images[] @rgba16f, int layer, ivec2 pixel, int sampleIndex, vec4 value) {
        return touchMS(images[layer], pixel, sampleIndex, value);
    }

    uint viaCounter(uimage2DMS counters[] @r32ui, int layer, ivec2 pixel, int sampleIndex, uint value) {
        return bumpMS(counters[layer], pixel, sampleIndex, value);
    }

    vec4 viaCube(imageCube images[] @rgba16f, int layer, ivec3 coord, vec4 value) {
        return touchCube(images[layer], coord, value);
    }

    vec4 viaCubeLayer(imageCubeArray images[] @rgba16f, int layer, ivec3 coord, vec4 value) {
        return touchCubeLayer(images[layer], coord, value);
    }

    compute {
        void main() {
            vec4 msValue = viaMS(msImages, 1, ivec2(2, 3), 0, vec4(1.0));
            uint count = viaCounter(msCounters, 1, ivec2(3, 4), 1, 5u);
            vec4 cubeValue = viaCube(cubeImages, 1, ivec3(2, 3, 4), msValue);
            vec4 layerValue = viaCubeLayer(cubeLayerImages, 1, ivec3(3, 4, 5), cubeValue + vec4(float(count)));
        }
    }
}
"""


GLSL_STORAGE_IMAGE_ACCESS_COMPUTE_SHADER = """
shader GLSLStorageImageAccessValidator {
    image2D source @access(readonly);
    image2D target @access(writeonly);
    uimage2D counters @r32ui @access(readwrite);

    float readPixel(image2D image @access(readonly), ivec2 pixel) {
        return imageLoad(image, pixel);
    }

    void writePixel(image2D image @access(writeonly), ivec2 pixel, vec4 value) {
        imageStore(image, pixel, value);
    }

    uint bump(uimage2D image @r32ui @access(readwrite), ivec2 pixel, uint value) {
        uint oldValue = imageAtomicAdd(image, pixel, value);
        imageStore(image, pixel, oldValue + 1u);
        return oldValue;
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            float value = readPixel(source, pixel);
            writePixel(target, pixel, vec4(value));
            uint oldValue = bump(counters, pixel, 1u);
            imageStore(counters, pixel, oldValue + 2u);
        }
    }
}
"""


GLSL_BUFFER_BLOCK_ACCESS_COMPUTE_SHADER = """
shader GLSLBufferBlockAccessValidator {
    struct Counter {
        uint value;
    };
    struct ReadonlyData {
        uint value;
    };
    struct WriteonlyData {
        uint value;
    };
    struct ReadwriteData {
        uint value;
        Counter nested;
        Counter items[2];
        uint values[4];
    };

    ReadonlyData readonlyData @glsl_buffer_block(std430) @readonly;
    WriteonlyData writeonlyData @glsl_buffer_block(std430) @writeonly;
    ReadwriteData readwriteData @glsl_buffer_block(std430) @access(readwrite);

    compute {
        void main() {
            uint value = readonlyData.value;
            writeonlyData.value = value;
            uint oldValue = atomicAdd(readwriteData.value, 1u);
            uint nestedOld = atomicAdd(readwriteData.nested.value, oldValue);
            uint itemOld = atomicAdd(readwriteData.items[1].value, nestedOld);
            uint arrayOld = atomicAdd(readwriteData.values[0], itemOld);
            uint swapped = atomicCompSwap(readwriteData.values[1], uint(0), arrayOld);
            readwriteData.value += oldValue;
            readwriteData.values[2] = swapped;
        }
    }
}
"""


GLSL_GEOMETRY_TESSELLATION_LAYOUT_SHADER = """
shader GLSLGeometryTessellationLayoutValidator {
    geometry {
        void main() @points @outputtopology(triangle_strip) @max_vertices(3) { }
    }

    tessellation_control {
        void main() @outputcontrolpoints(4) { }
    }

    tessellation_evaluation {
        void main() @domain(triangle) @partitioning(fractional_odd) @cw { }
    }
}
"""


GLSL_RAY_GENERATION_VALIDATOR_SHADER = """
shader GLSLRayGenerationValidator {
    accelerationStructureEXT topLevelAS @binding(0);

    ray_generation {
        layout(location = 0) @rayPayloadEXT vec4 rayPayload;

        void main() {
            rayPayload = vec4(1.0);
            TraceRay(
                topLevelAS,
                gl_RayFlagsNoneEXT,
                255u,
                0u,
                0u,
                0u,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                100.0,
                0
            );
        }
    }
}
"""


GLSL_RAY_QUERY_COMPUTE_VALIDATOR_SHADER = """
shader GLSLRayQueryComputeValidator {
    accelerationStructureEXT topLevelAS @binding(0);

    compute {
        void main() {
            RayQuery<RAY_FLAG_NONE> rayQuery;
            rayQueryInitializeEXT(
                rayQuery,
                topLevelAS,
                gl_RayFlagsNoneEXT,
                255u,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                100.0
            );
            bool active = rayQuery.Proceed();
            uint hitType = rayQuery.CommittedType();
        }
    }
}
"""


GLSL_RAY_QUERY_TRACE_RAY_INLINE_VALIDATOR_SHADER = """
shader GLSLRayQueryTraceRayInlineValidator {
    struct RayDesc {
        vec3 Origin;
        float TMin;
        vec3 Direction;
        float TMax;
    };

    accelerationStructureEXT topLevelAS @binding(0);

    compute {
        void main() {
            RayDesc ray;
            ray.Origin = vec3(0.0, 0.0, 0.0);
            ray.TMin = 0.001;
            ray.Direction = vec3(0.0, 0.0, 1.0);
            ray.TMax = 100.0;

            RayQuery<RAY_FLAG_NONE> rayQuery;
            rayQuery.TraceRayInline(
                topLevelAS,
                gl_RayFlagsNoneEXT,
                255u,
                ray
            );
            bool active = rayQuery.Proceed();
            uint hitType = rayQuery.CommittedType();
        }
    }
}
"""


MIXED_GLSL_GEOMETRY_INTERFACE_ARRAY_SHADER = """
#version 450 core
layout(lines_adjacency, invocations = 2) in;
layout(triangle_strip, max_vertices = 6) out;

in vec3 vColor[];
flat in int vLayer[];
out vec3 gColor;
flat out int gLayer;

void main() {
    for (int i = 0; i < 4; i++) {
        gColor = vColor[i];
        gLayer = vLayer[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
"""


MIXED_GLSL_FRAGMENT_MULTIPLE_OUTPUTS_SHADER = """
#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 0, index = 0) out vec4 accum;
layout(location = 0, index = 1) out vec4 revealage;
layout(location = 2) out vec4 normal;

void main() {
    accum = vec4(uv, 0.0, 1.0);
    revealage = vec4(1.0);
    normal = vec4(0.0);
}
"""


MIXED_GLSL_FRAGMENT_COMPONENT_PACKING_SHADER = """
#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 0, component = 0) out float luminance;
layout(location = 0, component = 1) out vec2 velocity;
layout(location = 0, component = 3) out float coverage;
layout(location = 1) out vec4 color;

void main() {
    luminance = uv.x;
    velocity = uv;
    coverage = uv.y;
    color = vec4(uv, 0.0, 1.0);
}
"""


MIXED_GLSL_FRAGMENT_BLEND_SUPPORT_SHADER = """
#version 460 core
#extension GL_KHR_blend_equation_advanced : enable
layout(location = 0) in vec2 uv;
layout(location = 0, blend_support_colordodge) out highp vec4 outputColour;
layout(location = 1, blend_support_multiply) out vec4 overlayColour;
layout(blend_support_multiply, blend_support_screen) out;

void main() {
    outputColour = vec4(uv, 0.0, 1.0);
    overlayColour = vec4(0.25);
}
"""


MIXED_GLSL_FRAGMENT_COLOR_DEPTH_SHADER = """
#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main() {
    color = vec4(uv, 0.0, 1.0);
    gl_FragDepth = uv.x;
}
"""


MIXED_GLSL_150_FRAGMENT_OUTPUT_SHADER = """
#version 150
out vec4 outputColor;

void main() {
    outputColor = vec4(1.0);
}
"""


GLSL_GEOMETRY_INTERFACE_BLOCK_VALIDATOR_SHADER = """
shader GLSLGeometryInterfaceBlockValidator {
    @glsl_interface_block(in) @glsl_interface_instance(vertexIn) @glsl_interface_array
    struct VertexIn {
        flat vec2 inputUv;
    };

    @glsl_interface_block(out) @glsl_interface_instance(fragmentOut)
    struct FragmentOut {
        noperspective vec2 outUv;
    };

    geometry {
        layout(points) in;
        layout(points, max_vertices = 1) out;

        void main() {
            fragmentOut.outUv = vertexIn[0].inputUv;
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
        }
    }
}
"""


MIXED_GLSL_TESSELLATION_CONTROL_INTERFACE_ARRAY_SHADER = """
#version 450 core
layout(vertices = 4) out;

in vec3 vPosition[];
out vec3 tcPosition[];
patch out vec4 tcPatchColor;

void main() {
    tcPosition[gl_InvocationID] = vPosition[gl_InvocationID];
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    tcPatchColor = vec4(1.0);
    gl_TessLevelOuter[0] = 2.0;
    gl_TessLevelOuter[1] = 2.0;
    gl_TessLevelOuter[2] = 2.0;
    gl_TessLevelOuter[3] = 2.0;
    gl_TessLevelInner[0] = 2.0;
    gl_TessLevelInner[1] = 2.0;
}
"""


MIXED_GLSL_TESSELLATION_EVALUATION_INTERFACE_ARRAY_SHADER = """
#version 450 core
layout(quads, fractional_even_spacing, ccw, point_mode) in;

in vec3 tcPosition[];
patch in vec4 tcPatchColor;
out vec4 teColor;

void main() {
    vec3 p = mix(
        mix(tcPosition[0], tcPosition[1], gl_TessCoord.x),
        mix(tcPosition[2], tcPosition[3], gl_TessCoord.x),
        gl_TessCoord.y
    );
    teColor = tcPatchColor;
    gl_Position = vec4(p, 1.0);
}
"""


MIXED_GLSL_PREPROCESSOR_COMPUTE_SHADER = """
#version 300 es
#extension GL_ARB_separate_shader_objects : enable
precision highp float;

void main() { }
"""


MIXED_GLSL_SSBO_UINT_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 17) buffer AtomicBlock {
    uint counter;
    uint bins[4];
} atomicBlock;

void main() {
    uint oldCounter = atomicAdd(atomicBlock.counter, 1u);
    uint oldBin = atomicExchange(atomicBlock.bins[2], oldCounter);
    uint minBin = atomicMin(atomicBlock.bins[0], 2u);
    uint maxBin = atomicMax(atomicBlock.bins[0], minBin);
    uint andBin = atomicAnd(atomicBlock.bins[1], 15u);
    uint orBin = atomicOr(atomicBlock.bins[1], andBin);
    uint xorBin = atomicXor(atomicBlock.bins[2], orBin);
    uint casBin = atomicCompSwap(atomicBlock.bins[3], xorBin, 7u);
    atomicAdd(atomicBlock.bins[1], casBin);
}
"""


MIXED_GLSL_SSBO_INT_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 18) buffer SignedAtomicBlock {
    int counter;
    int bins[4];
} signedAtomicBlock;

void main() {
    int oldCounter = atomicAdd(signedAtomicBlock.counter, -1);
    int oldBin = atomicExchange(signedAtomicBlock.bins[2], oldCounter);
    int minBin = atomicMin(signedAtomicBlock.bins[0], -2);
    int maxBin = atomicMax(signedAtomicBlock.bins[0], minBin);
    int andBin = atomicAnd(signedAtomicBlock.bins[1], 15);
    int orBin = atomicOr(signedAtomicBlock.bins[1], andBin);
    int xorBin = atomicXor(signedAtomicBlock.bins[2], orBin);
    int casBin = atomicCompSwap(signedAtomicBlock.bins[3], xorBin, -7);
    atomicAdd(signedAtomicBlock.bins[1], casBin);
}
"""


MIXED_GLSL_SSBO_RUNTIME_ARRAY_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 19) buffer RuntimeAtomicBlock {
    uint count;
    uint values[];
} runtimeAtomicBlock;
layout(std430, binding = 20) buffer RuntimeSignedAtomicBlock {
    int count;
    int values[];
} runtimeSignedAtomicBlock;

void main() {
    uint i = runtimeAtomicBlock.count;
    uint oldValue = atomicAdd(runtimeAtomicBlock.values[i], 1u);
    uint swapped = atomicCompSwap(runtimeAtomicBlock.values[i + 1u], oldValue, 7u);
    int j = runtimeSignedAtomicBlock.count;
    int oldSigned = atomicMin(runtimeSignedAtomicBlock.values[j], -2);
    int exchanged = atomicExchange(runtimeSignedAtomicBlock.values[j + 1], oldSigned);
    atomicAdd(runtimeSignedAtomicBlock.values[j], exchanged);
}
"""


MIXED_GLSL_SSBO_UNSUPPORTED_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 21) readonly buffer ReadAtomicBlock {
    uint value;
} readAtomicBlock;
layout(std430, binding = 22) buffer FloatAtomicBlock {
    float value;
} floatAtomicBlock;
layout(std430, binding = 23) buffer VectorAtomicBlock {
    uvec2 value;
} vectorAtomicBlock;
layout(std430, binding = 24) buffer MatrixAtomicBlock {
    mat2 value;
} matrixAtomicBlock;

void main() {
    uint readonlyOld = atomicAdd(readAtomicBlock.value, 1u);
    float floatOld = atomicAdd(floatAtomicBlock.value, 1.0);
    uint vectorOld = atomicAdd(vectorAtomicBlock.value, 1u);
    float matrixOld = atomicAdd(matrixAtomicBlock.value, 1.0);
}
"""


MIXED_GLSL_SSBO_STD140_COMPUTE_SHADER = """
#version 450 core
layout(std140, binding = 25) buffer Std140Block {
    uint count;
    mat2 basis;
    float weights[3];
    float values[];
} std140Block;

void main() {
    uint i = std140Block.count;
    mat2 basis = std140Block.basis;
    float weight = std140Block.weights[2];
    float value = std140Block.values[i];
    std140Block.basis = basis;
    std140Block.weights[1] = weight;
    std140Block.values[i] = value;
}
"""


MIXED_GLSL_SSBO_BOOL_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 26) buffer BoolBlock {
    bool enabled;
    bool flags[2];
    float values[];
} boolBlock;

void main() {
    uint i = boolBlock.enabled ? 1u : 0u;
    bool first = boolBlock.flags[0];
    bool dynamicFlag = boolBlock.flags[i];
    if (first || dynamicFlag) {
        boolBlock.flags[1] = false;
    }
    boolBlock.values[i] = boolBlock.values[i] + (boolBlock.enabled ? 1.0 : 0.0);
}
"""


MIXED_GLSL_SSBO_BOOL_VECTOR_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 27) buffer BoolVectorBlock {
    bvec3 mask;
    bvec2 pairs[2];
    bvec4 values[];
} boolVectorBlock;

void main() {
    uint i = boolVectorBlock.mask.x ? 1u : 0u;
    bvec3 mask = boolVectorBlock.mask;
    bvec2 pair = boolVectorBlock.pairs[i];
    bvec4 dynamicFlags = boolVectorBlock.values[i];
    boolVectorBlock.mask = bvec3(dynamicFlags.x, pair.y, mask.z);
    boolVectorBlock.pairs[0] = bvec2(mask.x, dynamicFlags.y);
    boolVectorBlock.values[1] = bvec4(mask.x, pair.y, dynamicFlags.z, true);
}
"""


MIXED_GLSL_SSBO_NESTED_STRUCT_COMPUTE_SHADER = """
#version 450 core
struct InnerBlockData {
    float scale;
    bvec3 mask;
};

layout(std430, binding = 28) buffer NestedBlock {
    uint count;
    InnerBlockData inner;
    float values[];
} nestedBlock;

void main() {
    uint i = nestedBlock.count;
    float scale = nestedBlock.inner.scale;
    bvec3 mask = nestedBlock.inner.mask;
    nestedBlock.inner.scale = scale + 1.0;
    nestedBlock.inner.mask = bvec3(mask.y, mask.x, true);
    nestedBlock.values[i] = nestedBlock.inner.scale;
}
"""


MIXED_GLSL_SSBO_NESTED_STRUCT_ARRAY_COMPUTE_SHADER = """
#version 450 core
struct ArrayBlockData {
    uint id;
    vec3 normal;
    bvec2 flags;
};

layout(std430, binding = 29) buffer NestedArrayBlock {
    ArrayBlockData fixedItems[2];
    uint count;
    ArrayBlockData items[];
} nestedArrayBlock;

void main() {
    uint i = nestedArrayBlock.count;
    vec3 normal = nestedArrayBlock.fixedItems[1].normal;
    bvec2 flags = nestedArrayBlock.items[i].flags;
    nestedArrayBlock.fixedItems[0].id = nestedArrayBlock.items[i].id;
    nestedArrayBlock.items[i].normal = normal;
    nestedArrayBlock.items[i].flags = bvec2(flags.y, true);
}
"""


MIXED_GLSL_SSBO_NESTED_STRUCT_AGGREGATE_COMPUTE_SHADER = """
#version 450 core
struct AggregatePayload {
    float scale;
    bvec3 mask;
};

struct AggregateBlockData {
    AggregatePayload payload;
    uint id;
};

layout(std430, binding = 30) buffer AggregateBlock {
    AggregateBlockData inner;
    AggregateBlockData items[];
} aggregateBlock;

void main() {
    uint i = 1u;
    AggregateBlockData inner = aggregateBlock.inner;
    AggregateBlockData item = aggregateBlock.items[i];
    aggregateBlock.inner = item;
    aggregateBlock.items[i] = inner;
}
"""


MIXED_GLSL_SSBO_NESTED_STRUCT_AGGREGATE_ARRAY_COMPUTE_SHADER = """
#version 450 core
struct ArrayAggregateItem {
    vec2 uv;
    bvec2 flags;
};

struct ArrayAggregateData {
    float weights[2];
    ArrayAggregateItem items[2];
    uint id;
};

layout(std430, binding = 16) buffer ArrayAggregateBlock {
    ArrayAggregateData inner;
    ArrayAggregateData entries[];
} arrayAggregateBlock;

void main() {
    uint i = 1u;
    ArrayAggregateData inner = arrayAggregateBlock.inner;
    ArrayAggregateData entry = arrayAggregateBlock.entries[i];
    arrayAggregateBlock.inner = entry;
    arrayAggregateBlock.entries[i] = inner;
}
"""


MIXED_GLSL_SSBO_AGGREGATE_LAYOUT_HELPER_COMPUTE_SHADER = """
#version 450 core
struct LayoutSharedData {
    float weights[2];
    uint id;
};

layout(std430, binding = 14) buffer LayoutStd430Block {
    LayoutSharedData item;
} block430;

layout(std140, binding = 15) buffer LayoutStd140Block {
    LayoutSharedData item;
} block140;

void main() {
    LayoutSharedData a = block430.item;
    LayoutSharedData b = block140.item;
    block430.item = b;
    block140.item = a;
}
"""


MIXED_GLSL_SSBO_READONLY_AGGREGATE_COMPUTE_SHADER = """
#version 450 core
struct ReadOnlyAggregateItem {
    vec2 uv;
    bvec2 flags;
};

struct ReadOnlyAggregateData {
    float weights[2];
    ReadOnlyAggregateItem items[2];
    uint id;
};

layout(std430, binding = 13) readonly buffer ReadOnlyAggregateBlock {
    ReadOnlyAggregateData inner;
    ReadOnlyAggregateData entries[];
} readAggregateBlock;

ReadOnlyAggregateData readEntry(uint i) {
    return readAggregateBlock.entries[i];
}

void main() {
    uint i = 1u;
    ReadOnlyAggregateData inner = readAggregateBlock.inner;
    ReadOnlyAggregateData entry = readEntry(i);
    float weight = entry.weights[1] + inner.weights[0];
    bool flag = entry.items[1].flags.y;
}
"""


MIXED_GLSL_SSBO_NESTED_AGGREGATE_LEAF_COMPOUND_COMPUTE_SHADER = """
#version 450 core
struct CompoundItem {
    vec2 uv;
    bvec2 flags;
};

struct CompoundData {
    float weights[2];
    CompoundItem items[2];
    uint id;
};

layout(std430, binding = 12) buffer CompoundAggregateBlock {
    uint index;
    CompoundData entries[];
} compoundAggregateBlock;

void main() {
    uint i = compoundAggregateBlock.index;
    compoundAggregateBlock.entries[i].weights[1] += 1.0;
    compoundAggregateBlock.entries[i].items[1].uv += vec2(0.5);
    compoundAggregateBlock.entries[i].items[0].flags = bvec2(true, false);
}
"""


URP_TOON_LIGHTING_SOURCE_URL = (
    "https://github.com/TinyPlay/URPShadersCollection/blob/"
    "6e663fffccd00a4cce837644a29f6e8f82a6e372/"
    "Shaders/_Includes/ToonLighting.hlsl"
)


URP_TOON_LIGHTING_HLSL_INCLUDE = """
#ifndef CUSTOM_LIGHTING_INCLUDED
#define CUSTOM_LIGHTING_INCLUDED

void CalculateMainLight_float(
    float3 WorldPos,
    out float3 Direction,
    out float3 Color,
    out half DistanceAtten,
    out half ShadowAtten
)
{
    #ifdef SHADERGRAPH_PREVIEW
        Direction = float3(0.5,0.5,0);
        Color = 1;
        DistanceAtten = 1;
        ShadowAtten = 1;
    #else
        #if SHADOWS_SCREEN
            half4 clipPos = TransformWorldToHClip(WorldPos);
            half4 shadowCoord = ComputeScreenPos(clipPos);
        #else
            half4 shadowCoord = TransformWorldToShadowCoord(WorldPos);
        #endif

        Light mainLight = GetMainLight(0);
        Direction = mainLight.direction;
        Color = mainLight.color;
        DistanceAtten = mainLight.distanceAttenuation;
        ShadowAtten = mainLight.shadowAttenuation;
    #endif
}

#endif
"""


PMFX_SHADER_SOURCE_URL = (
    "https://github.com/polymonster/pmfx-shader/blob/"
    "79df2ad107dc35dd02f6f92ab4b38fcc05ba5fbf/"
    "examples/v2/v2_examples.hlsl"
)


PMFX_PERMUTATION_CONDITIONAL_HLSL = """
struct vs_output {
    float4 position : SV_POSITION;
};

struct vs_input {
    float4 position : POSITION;
};

vs_output vs_output_default() {
    vs_output output;
    output.position = float4(0, 0, 0, 1);
    return output;
}

vs_output vs_main_permutations(vs_input input) {
    if:(SKINNED) {
        return vs_output_default();
    }
    else if:(INSTANCED) {
        return vs_output_default();
    }
    return vs_output_default();
}
"""

PMFX_EFFECT_METADATA_HLSL = """
state default {
    DepthEnable = true;
}

program p0 {
    vs = vs_main;
    ps = ps_main;
}

fxgroup PostProcess {
    technique10 Render {
        pass P0 {
            PixelShader = compile ps_5_0 ps_main();
        }
    }
}

pass ExtractedPass {
    PixelShader = compile ps_5_0 ps_main();
}

float4 ps_main() : SV_Target {
    return float4(1, 1, 1, 1);
}
"""

DXC_STYLE_STRUCT_CONSTRUCTOR_HLSL = """
struct MaterialSample {
    float roughness;
    float3 normal;

    MaterialSample(float value, float3 n) : roughness(value), normal(n) {}

    ~MaterialSample() {
        roughness = 0.0;
    }
};

float4 ps_main(float3 n : NORMAL) : SV_Target {
    MaterialSample sample = MaterialSample(0.5, n);
    return float4(sample.normal * sample.roughness, 1);
}
"""


def _fragment_ast():
    return crosstl.translator.parse(FRAGMENT_SMOKE_SHADER)


def _hlsl_to_crossgl(source):
    tokens = HLSLLexer(source).tokenize()
    ast = HLSLParser(tokens).parse()
    crossgl = HLSLToCrossGLConverter().generate(ast)
    assert crosstl.translator.parse(crossgl) is not None
    return crossgl


def _mixed_glsl_ast(source, shader_type):
    tokens = GLSLLexer(source).tokenize()
    glsl_ast = GLSLParser(tokens, shader_type).parse()
    crossgl = GLSLToCrossGLConverter(shader_type=shader_type).generate(glsl_ast)
    return crosstl.translator.parse(crossgl)


def _glsl_specialization_constant_ast():
    return _mixed_glsl_ast(GLSL_SPECIALIZATION_CONSTANT_VERTEX_SHADER, "vertex")


def _require_tool(name):
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} is not installed")
    return path


def _require_glslang_stage(glslang, stage):
    result = subprocess.run(
        [glslang, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    help_text = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode != 0 and not help_text.strip():
        detail = help_text.strip() or "no diagnostic output"
        pytest.skip(f"glslangValidator stage probe failed: {detail}")
    if stage not in help_text:
        pytest.skip(f"glslangValidator does not advertise GLSL stage {stage!r}")


def _require_xcrun_tool(name):
    xcrun = _require_tool("xcrun")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        pytest.skip(f"xcrun cannot locate {name}: {detail}")
    return xcrun


def _run_validator(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    assert result.returncode == 0, diagnostics


def _run_glslang_mesh_task_validator(glslang, stage, source_path):
    output_path = source_path.with_suffix(source_path.suffix + ".spv")
    _run_validator(
        [
            glslang,
            "-S",
            stage,
            "--target-env",
            "vulkan1.3",
            "-o",
            str(output_path),
            str(source_path),
        ]
    )


def _run_hlsl_glslang_spirv_validator(glslang, spirv_val, stage, entry, source_path):
    output_path = source_path.with_suffix(source_path.suffix + ".spv")
    _run_validator(
        [
            glslang,
            "-D",
            "-V",
            "-e",
            entry,
            "-S",
            stage,
            str(source_path),
            "-o",
            str(output_path),
        ]
    )
    _run_validator([spirv_val, str(output_path)])


def test_mixed_glsl_specialization_constants_lower_for_target_codegen():
    shader_ast = _glsl_specialization_constant_ast()

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "layout(constant_id" not in glsl
    assert "CrossGL fallback: OpenGL source validation cannot preserve" in glsl
    assert "const int arraySize = 5;" in glsl
    assert glsl.count("void foo") == 1
    assert "output." not in glsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "constant_id" not in metal
    assert (
        "Metal source output cannot preserve GLSL specialization constant id" in metal
    )
    assert "Metal does not support double specialization constant" in metal
    assert "constant double spDouble" not in metal
    assert "thread VertexOutput& output" in metal
    assert "constant int gl_MaxImageUnits = 8;" in metal
    assert "Metal vertex entry points require a position output" in metal
    assert "float4 __crossgl_position [[position]];" in metal

    spirv = VulkanSPIRVCodeGen().generate(shader_ast)
    assert "WARNING" not in spirv
    assert "SpecId 16" in spirv
    assert "SpecId 116" in spirv
    assert re.search(r"OpFunctionCall %\d+ %\d+ %\d+ %\d+", spirv)


def test_mixed_glsl_specialization_constants_opengl_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "specialization_constants.vert"

    shader_path.write_text(
        GLSLCodeGen().generate(_glsl_specialization_constant_ast()),
        encoding="utf-8",
    )

    _run_validator([glslang, "-S", "vert", str(shader_path)])


def test_mixed_glsl_specialization_constants_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "specialization_constants.metal"
    output_path = tmp_path / "specialization_constants.air"

    shader_path.write_text(
        MetalCodeGen().generate(_glsl_specialization_constant_ast()),
        encoding="utf-8",
    )

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_specialization_constants_spirv_output_validates_with_spirv_tools(
    tmp_path,
):
    spirv_as = _require_tool("spirv-as")
    spirv_val = _require_tool("spirv-val")
    asm_path = tmp_path / "specialization_constants.spvasm"
    spv_path = tmp_path / "specialization_constants.spv"

    asm_path.write_text(
        VulkanSPIRVCodeGen().generate(_glsl_specialization_constant_ast()),
        encoding="utf-8",
    )

    _run_validator([spirv_as, str(asm_path), "-o", str(spv_path)])
    _run_validator([spirv_val, str(spv_path)])


def _compile_slang_hlsl_entry(
    slangc,
    source_path,
    output_path,
    entry,
    stage,
    profile,
):
    _run_validator(
        [
            slangc,
            "-target",
            "hlsl",
            "-entry",
            entry,
            "-stage",
            stage,
            "-profile",
            profile,
            "-o",
            str(output_path),
            str(source_path),
        ]
    )
    assert output_path.exists()


def _run_validator_or_skip_unsupported_extension(
    command, extension, unsupported_diagnostics=()
):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    unsupported_markers = (
        "not supported",
        "unsupported",
        "unrecognized extension",
        "extension not supported",
    )
    if (
        result.returncode != 0
        and extension in diagnostics
        and any(marker in diagnostics.lower() for marker in unsupported_markers)
    ):
        pytest.skip(f"{extension} is not supported by this validator build")
    if (
        result.returncode != 0
        and unsupported_diagnostics
        and all(marker in diagnostics for marker in unsupported_diagnostics)
    ):
        pytest.skip(
            f"{extension} qualifier path is not supported by this validator build"
        )
    assert result.returncode == 0, diagnostics


@pytest.mark.parametrize(
    "shader_source",
    (
        CROSSGL_WGSL_GRAPHICS_SHADER,
        CROSSGL_WGSL_RESOURCE_SHADER,
        CROSSGL_WGSL_COMPUTE_SHADER,
        CROSSGL_WGSL_BUFFER_BLOCK_SHADER,
    ),
)
def test_generated_wgsl_validates_with_naga(tmp_path, shader_source):
    naga = _require_tool("naga")
    ast = crosstl.translator.parse(shader_source)
    generated = WGSLCodeGen().generate(ast)
    source_path = tmp_path / "shader.wgsl"
    source_path.write_text(generated, encoding="utf-8")

    _run_validator([naga, "--input-kind", "wgsl", str(source_path)])


def test_real_world_urp_hlsl_include_imports_to_parseable_crossgl():
    crossgl = _hlsl_to_crossgl(URP_TOON_LIGHTING_HLSL_INCLUDE)

    assert URP_TOON_LIGHTING_SOURCE_URL.startswith("https://github.com/")
    assert (
        "void CalculateMainLight_float(vec3 WorldPos, out vec3 Direction, "
        "out vec3 Color, out float16 DistanceAtten, out float16 ShadowAtten)"
    ) in crossgl
    assert "f16vec4 shadowCoord = TransformWorldToShadowCoord(WorldPos);" in crossgl
    assert "Light mainLight = GetMainLight(0);" in crossgl
    assert "Direction = mainLight.direction;" in crossgl
    assert "Color = mainLight.color;" in crossgl


def test_real_world_pmfx_permutation_conditionals_import_to_parseable_crossgl():
    assert PMFX_SHADER_SOURCE_URL.startswith("https://github.com/")

    crossgl = _hlsl_to_crossgl(PMFX_PERMUTATION_CONDITIONAL_HLSL)

    assert "if (SKINNED)" in crossgl
    assert "else if (INSTANCED)" in crossgl
    assert "return vs_output_default();" in crossgl


def test_pmfx_style_effect_metadata_blocks_import_to_parseable_crossgl():
    crossgl = _hlsl_to_crossgl(PMFX_EFFECT_METADATA_HLSL)

    assert "vec4 ps_main() @ gl_FragColor" in crossgl
    assert "return vec4(1, 1, 1, 1);" in crossgl


def test_dxc_style_struct_constructors_import_to_parseable_crossgl():
    crossgl = _hlsl_to_crossgl(DXC_STYLE_STRUCT_CONSTRUCTOR_HLSL)

    assert "struct MaterialSample" in crossgl
    assert "MaterialSample sample = MaterialSample(0.5, n);" in crossgl
    assert "return vec4(sample.normal * sample.roughness, 1);" in crossgl


def test_generated_hlsl_vertex_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "validator_vertex_smoke.hlsl"
    output_path = tmp_path / "validator_vertex_smoke.dxil"

    shader_path.write_text(
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(CROSSGL_WGSL_GRAPHICS_SHADER),
            "vertex",
        ),
        encoding="utf-8",
    )

    _run_validator(
        [
            dxc,
            "-T",
            "vs_6_0",
            "-E",
            "VSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_fragment_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "validator_smoke.hlsl"
    output_path = tmp_path / "validator_smoke.dxil"

    shader_path.write_text(
        HLSLCodeGen().generate_stage(_fragment_ast(), "fragment"),
        encoding="utf-8",
    )

    _run_validator(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_compute_synchronization_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "synchronization.hlsl"
    output_path = tmp_path / "synchronization.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    assert code.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert "AllMemoryBarrier();" in code
    assert "workgroupBarrier();" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_wave_quad_intrinsics_compile_with_dxc(tmp_path):
    shader_path = tmp_path / "wave_quad.hlsl"
    output_path = tmp_path / "wave_quad.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(CROSSGL_WAVE_QUAD_COMPUTE_SHADER)
    )
    assert "[numthreads(4, 1, 1)]" in code
    for snippet in [
        "RWBuffer<uint> outputValues : register(u0);",
        "WaveGetLaneIndex()",
        "WaveGetLaneCount()",
        "WaveIsFirstLane()",
        "WaveActiveSum(value)",
        "WavePrefixSum(sumValue)",
        "WaveActiveAnyTrue((prefixValue >= lane))",
        "WaveActiveAllTrue((laneCount > 0u))",
        "WaveActiveBallot(((anyLane || allLane) || firstLane))",
        "WaveMatch(prefixValue)",
        "WaveReadLaneAt(prefixValue, 0u)",
        "WaveReadLaneFirst(((broadcast + ballot.x) + matchMask.x))",
        "QuadReadAcrossX(firstValue)",
        "QuadReadLaneAt(value, 3u)",
        "QuadAny(anyLane)",
        "QuadAll(allLane)",
    ]:
        assert snippet in code
    assert "uvec4" not in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_5",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_texture_resource_intrinsics_compile_with_dxc(tmp_path):
    shader_path = tmp_path / "texture_resources.hlsl"
    output_path = tmp_path / "texture_resources.dxil"

    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(CROSSGL_TEXTURE_RESOURCE_FRAGMENT_SHADER),
        "fragment",
    )
    for snippet in [
        "Texture2D colorMap : register(t0);",
        "SamplerState linearSampler : register(s0);",
        "int textureQueryLevels(Texture2D tex)",
        "int2 textureSize(Texture2D tex, int lod)",
        "colorMap.Sample(linearSampler, input.uv)",
        "colorMap.SampleBias(linearSampler, input.uv, 0.25)",
        "colorMap.SampleLevel(linearSampler, input.uv, lod)",
        "colorMap.SampleGrad(linearSampler, input.uv, input.ddxValue, input.ddyValue)",
        "colorMap.Sample(linearSampler, input.uv, offset)",
        "colorMap.Load(int3(pixel, lod))",
        "colorMap.Load(int3((pixel + offset), lod))",
        "component == 0 ? colorMap.GatherRed(linearSampler, input.uv)",
        "colorMap.GatherGreen(linearSampler, input.uv, offset)",
        "colorMap.CalculateLevelOfDetailUnclamped(linearSampler, input.uv)",
        "colorMap.CalculateLevelOfDetail(linearSampler, input.uv)",
    ]:
        assert snippet in code
    for unsupported in [
        "textureLod(",
        "textureGrad(",
        "textureOffset(",
        "textureGather(",
        "textureGatherOffset(",
        "texelFetch(",
        "texelFetchOffset(",
        "textureQueryLod(",
    ]:
        assert unsupported not in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_shadow_texture_intrinsics_compile_with_dxc(tmp_path):
    shader_path = tmp_path / "shadow_textures.hlsl"
    output_path = tmp_path / "shadow_textures.dxil"

    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(CROSSGL_SHADOW_TEXTURE_FRAGMENT_SHADER),
        "fragment",
    )
    for snippet in [
        "Texture2D shadowMap : register(t1, space2);",
        "SamplerComparisonState compareSampler : register(s0, space2);",
        "SamplerState shadowMapQuerySampler : register(s2, space2);",
        (
            "float sampleShadow(Texture2D tex, SamplerComparisonState cmp, "
            "float2 uv, float depth)"
        ),
        "tex.SampleCmp(cmp, uv, depth)",
        "shadowMap.SampleCmp(compareSampler, input.uv, input.depth)",
        "shadowMap.SampleCmp(compareSampler, input.uv, input.depth, offset)",
        (
            "shadowMap.SampleCmp(compareSampler, "
            "input.projected.xy / input.projected.z, input.depth)"
        ),
        (
            "shadowMap.SampleCmp(compareSampler, "
            "input.projected.xy / input.projected.z, input.depth, offset)"
        ),
        "shadowMap.GatherCmp(compareSampler, input.uv, input.depth)",
        "shadowMap.GatherCmp(compareSampler, input.uv, input.depth, offset)",
        (
            "shadowMap.CalculateLevelOfDetailUnclamped("
            "shadowMapQuerySampler, input.uv)"
        ),
        "shadowMap.CalculateLevelOfDetail(shadowMapQuerySampler, input.uv)",
    ]:
        assert snippet in code
    for unsupported in [
        "textureCompare(",
        "textureCompareOffset(",
        "textureCompareProj(",
        "textureCompareProjOffset(",
        "textureGatherCompare(",
        "textureGatherCompareOffset(",
        "textureQueryLod(",
    ]:
        assert unsupported not in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_typed_buffer_atomics_compile_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "typed_buffer_atomics.hlsl"
    output_path = tmp_path / "typed_buffer_atomics.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(CROSSGL_TYPED_BUFFER_ATOMICS_COMPUTE_SHADER)
    )
    assert "InterlockedAdd(counters[tid.x], 1u, original);" in code
    assert "InterlockedCompareExchange(counters[tid.x], 2u, 3u, original);" in code
    assert "InterlockedAdd(counterArrays[1][tid.x], 1u, original);" in code
    assert "InterlockedMax(signedCounters[tid.x], -1, oldSigned);" in code
    assert "InterlockedAdd(counters[index], 1u, __crossgl_atomic_return_0);" in code
    assert (
        "InterlockedCompareExchange(counters[index], 2u, 3u, __crossgl_atomic_return_1);"
        in code
    )
    assert "InterlockedAdd(counters[index], 5u, __crossgl_atomic_expr_2);" in code
    assert "InterlockedAdd(counters[tid.x], 5u, __crossgl_atomic_expr_3);" in code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_4);"
    ) in code
    assert (
        "InterlockedAdd(counterArrays[1][tid.x], 2u, __crossgl_atomic_expr_5);" in code
    )
    assert "atomicCompareExchange(counters" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_struct_buffer_atomics_validate_with_glslang_spirv_val(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    spirv_val = _require_tool("spirv-val")
    shader_path = tmp_path / "struct_buffer_atomics.hlsl"

    code = HLSLCodeGen().generate(crosstl.translator.parse("""
            shader ExternalValidatorStructBufferAtomics {
                struct Counters {
                    int active;
                }

                compute {
                    buffer Counters counters;

                    void main() {
                        atomicAdd(counters.active, 1);
                    }
                }
            }
            """))
    assert "RWStructuredBuffer<Counters> counters : register(u0);" in code
    assert "InterlockedAdd(counters[0].active, 1);" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_hlsl_glslang_spirv_validator(glslang, spirv_val, "comp", "CSMain", shader_path)


def test_generated_hlsl_rasterizer_ordered_resources_compile_with_dxc(tmp_path):
    shader_path = tmp_path / "rasterizer_ordered_resources.hlsl"
    output_path = tmp_path / "rasterizer_ordered_resources.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(HLSL_RASTERIZER_ORDERED_VALIDATOR_SHADER)
    )
    assert "RasterizerOrderedTexture2D<uint> pixelCounts : register(u0);" in code
    assert "RasterizerOrderedTexture2DArray<float4> layers : register(u1);" in code
    assert "RasterizerOrderedBuffer<uint> bins : register(u2);" in code
    assert "RasterizerOrderedStructuredBuffer<int> values : register(u3);" in code
    assert "RasterizerOrderedByteAddressBuffer rawBytes : register(u4);" in code
    assert (
        "uint imageAtomicAdd_uimage2D("
        "RasterizerOrderedTexture2D<uint> image, int2 coord, uint value)"
    ) in code
    assert "InterlockedAdd(bins[0], oldCount, oldBin);" in code
    assert "InterlockedMax(values[0], int(oldBin), oldValue);" in code
    assert "rawBytes.Store(0, oldBin);" in code
    assert "RWTexture2D<uint> pixelCounts" not in code
    assert "RWBuffer<uint> bins : register(u2);" not in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_dxr_library_compiles_with_dxc(tmp_path):
    shader_path = tmp_path / "dxr_library.hlsl"
    output_path = tmp_path / "dxr_library.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(HLSL_DXR_LIBRARY_VALIDATOR_SHADER)
    )
    assert '[shader("raygeneration")]' in code
    assert '[shader("miss")]' in code
    assert '[shader("closesthit")]' in code
    assert '[shader("anyhit")]' in code
    assert '[shader("callable")]' in code
    assert "void RayGenMain()" in code
    assert "void MissMain(inout Payload payload)" in code
    assert (
        "void ClosestHitMain(inout Payload payload, "
        "in BuiltInTriangleIntersectionAttributes attributes)"
    ) in code
    assert (
        "void AnyHitMain(inout Payload payload, "
        "in BuiltInTriangleIntersectionAttributes attributes)"
    ) in code
    assert "void CallableMain(inout CallableData data)" in code
    assert "TraceRay(accel, 0, 255, 0, 1, 0, ray, payload);" in code
    assert "CallShader(0, data);" in code
    assert ": payload" not in code
    assert ": hit_attribute" not in code
    assert ": callable_data" not in code
    assert "struct BuiltInTriangleIntersectionAttributes" not in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "lib_6_3",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_mesh_amplification_compile_with_dxc(tmp_path):
    shader_path = tmp_path / "mesh_amplification.hlsl"
    amplification_output = tmp_path / "mesh_amplification_as.dxil"
    mesh_output = tmp_path / "mesh_amplification_ms.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(HLSL_MESH_AMPLIFICATION_VALIDATOR_SHADER)
    )
    assert '[shader("amplification")]' in code
    assert '[shader("mesh")]' in code
    assert "[numthreads(1, 1, 1)]" in code
    assert "[numthreads(32, 1, 1)]" in code
    assert '[outputtopology("triangle")]' in code
    assert "void ASMain()" in code
    assert "groupshared MeshPayload payload;" in code
    assert code.index("groupshared MeshPayload payload;") < code.index("void ASMain()")
    as_body = code[code.index("void ASMain()") : code.index("[numthreads(32, 1, 1)]")]
    assert "groupshared MeshPayload payload;" not in as_body
    assert "DispatchMesh(1, 1, 1, payload);" in code
    assert (
        "void MSMain(in payload MeshPayload payload, "
        "out vertices MeshVertex verts[3], out indices uint3 tris[1])"
    ) in code
    assert "SetMeshOutputCounts(3, 1);" in code
    assert "tris[0] = uint3(0u, 1u, 2u);" in code
    assert ": mesh_payload" not in code
    assert "EmitMeshTasksEXT" not in code
    assert "SetMeshOutputsEXT" not in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "as_6_5",
            "-E",
            "ASMain",
            str(shader_path),
            "-Fo",
            str(amplification_output),
        ]
    )
    _run_validator(
        [
            dxc,
            "-T",
            "ms_6_5",
            "-E",
            "MSMain",
            str(shader_path),
            "-Fo",
            str(mesh_output),
        ]
    )
    assert amplification_output.exists()
    assert mesh_output.exists()


def test_generated_slang_mesh_task_pair_compiles_with_slangc(tmp_path):
    shader_path = tmp_path / "slang_mesh_task_pair.slang"
    amplification_output = tmp_path / "slang_mesh_task_pair_as.hlsl"
    mesh_output = tmp_path / "slang_mesh_task_pair_ms.hlsl"

    code = SlangCodeGen().generate(
        crosstl.translator.parse(SLANG_MESH_TASK_VALIDATOR_SHADER)
    )
    assert '[numthreads(1, 1, 1)]\n[shader("amplification")]' in code
    assert "void ASMain(uint3 groupId : SV_GroupID)" in code
    assert "groupshared MeshPayload payload;" in code
    assert "payload.meshlet = groupId.x;" in code
    assert "DispatchMesh(1, 1, 1, payload);" in code
    assert '[numthreads(32, 1, 1)]\n[outputtopology("triangle")]' in code
    assert '[shader("mesh")]' in code
    assert (
        "void MSMain(in payload MeshPayload payload, "
        "uint3 threadId : SV_GroupThreadID, "
        "out vertices MeshVertex verts[3], out indices uint3 tris[1], "
        "out primitives MeshPrimitive prims[1])" in code
    )
    assert "float4 position : SV_Position;" in code
    assert "float2 uv : TEXCOORD0;" in code
    assert "bool culled : SV_CullPrimitive;" in code
    assert "SetMeshOutputCounts(3, 1);" in code
    assert "verts[0].position = float4(float(payload.meshlet), 0.0, 0.0, 1.0);" in code
    assert "verts[1].position = float4(float(threadId.x), 1.0, 0.0, 1.0);" in code
    assert "tris[0] = uint3(0u, 1u, 2u);" in code
    assert "prims[0].culled = false;" in code
    assert ": gl_WorkGroupID" not in code
    assert ": gl_LocalInvocationID" not in code
    assert ": mesh_payload" not in code
    assert ": vertices" not in code
    assert ": indices" not in code
    assert ": primitives" not in code
    assert "unsupported Slang mesh intrinsic" not in code
    shader_path.write_text(code, encoding="utf-8")

    slangc = _require_tool("slangc")
    _compile_slang_hlsl_entry(
        slangc, shader_path, amplification_output, "ASMain", "amplification", "as_6_5"
    )
    _compile_slang_hlsl_entry(
        slangc, shader_path, mesh_output, "MSMain", "mesh", "ms_6_5"
    )


def test_generated_hlsl_tessellation_pair_compile_with_dxc(tmp_path):
    shader_path = tmp_path / "tessellation_pair.hlsl"
    hull_output = tmp_path / "tessellation_pair_hs.dxil"
    domain_output = tmp_path / "tessellation_pair_ds.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(HLSL_TESSELLATION_VALIDATOR_SHADER)
    )
    assert '[shader("hull")]' in code
    assert '[shader("domain")]' in code
    assert '[patchconstantfunc("HSConst")]' in code
    assert "[outputcontrolpoints(3)]" in code
    assert "float edges[3] : SV_TessFactor;" in code
    assert "constants.edges[0] = 1.0;" in code
    assert (
        "HSOutput HSMain(InputPatch<HSInput, 3> patch, "
        "uint id : SV_OutputControlPointID)"
    ) in code
    assert (
        "float4 DSMain(OutputPatch<HSOutput, 3> patch, "
        "float3 bary : SV_DomainLocation): SV_Position"
    ) in code
    shader_path.write_text(code, encoding="utf-8")

    dxc = _require_tool("dxc")
    _run_validator(
        [
            dxc,
            "-T",
            "hs_6_0",
            "-E",
            "HSMain",
            str(shader_path),
            "-Fo",
            str(hull_output),
        ]
    )
    _run_validator(
        [
            dxc,
            "-T",
            "ds_6_0",
            "-E",
            "DSMain",
            str(shader_path),
            "-Fo",
            str(domain_output),
        ]
    )
    assert hull_output.exists()
    assert domain_output.exists()


def test_generated_slang_tessellation_pair_compiles_with_slangc(tmp_path):
    shader_path = tmp_path / "slang_tessellation_pair.slang"
    hull_output = tmp_path / "slang_tessellation_pair_hs.hlsl"
    domain_output = tmp_path / "slang_tessellation_pair_ds.hlsl"

    code = SlangCodeGen().generate(
        crosstl.translator.parse(SLANG_TESSELLATION_VALIDATOR_SHADER)
    )
    assert "float4 position : SV_Position;" in code
    assert "float2 uv : TEXCOORD0;" in code
    assert "float outer[3] : SV_TessFactor;" in code
    assert "float inner[1] : SV_InsideTessFactor;" in code
    assert (
        "PatchConstants HSConst(InputPatch<VSOut, 3> inputPatch, "
        "uint patchID : SV_PrimitiveID)"
    ) in code
    assert '[domain("tri")]' in code
    assert '[partitioning("integer")]' in code
    assert '[outputtopology("triangle_cw")]' in code
    assert "[outputcontrolpoints(3)]" in code
    assert '[patchconstantfunc("HSConst")]' in code
    assert '[shader("hull")]' in code
    assert (
        "HSOut HSMain(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_InvocationID : SV_OutputControlPointID)"
    ) in code
    assert "VSOut first = inputPatch[0];" in code
    assert "VSOut current = inputPatch[gl_InvocationID];" in code
    assert '[shader("domain")]' in code
    assert (
        "float4 DSMain(OutputPatch<HSOut, 3> patch, "
        "float3 bary : SV_DomainLocation) : SV_Position"
    ) in code
    assert "return p0 + p1 + p2;" in code
    assert "gl_in" not in code
    assert ": gl_TessCoord" not in code
    assert ": gl_TessLevelOuter" not in code
    assert ": gl_TessLevelInner" not in code
    shader_path.write_text(code, encoding="utf-8")

    slangc = _require_tool("slangc")
    _compile_slang_hlsl_entry(
        slangc, shader_path, hull_output, "HSMain", "hull", "hs_6_0"
    )
    _compile_slang_hlsl_entry(
        slangc, shader_path, domain_output, "DSMain", "domain", "ds_6_0"
    )


def test_generated_slang_ray_stage_library_compiles_with_slangc(tmp_path):
    shader_path = tmp_path / "slang_ray_stage_library.slang"

    code = SlangCodeGen().generate(
        crosstl.translator.parse(SLANG_RAY_STAGE_VALIDATOR_SHADER)
    )
    assert '[shader("raygeneration")]' in code
    assert "void RayGenMain()" in code
    assert "uint3 launch = DispatchRaysIndex();" in code
    assert "uint launchSizeX = DispatchRaysDimensions().x;" in code
    assert '[shader("closesthit")]' in code
    assert (
        "void ClosestHitMain(inout RayPayload payload, "
        "in HitAttributes attributes)" in code
    )
    assert '[shader("anyhit")]' in code
    assert (
        "void AnyHitMain(inout RayPayload payload, "
        "in HitAttributes attributes)" in code
    )
    assert "AcceptHitAndEndSearch();" in code
    assert '[shader("miss")]' in code
    assert "void MissMain(inout RayPayload payload)" in code
    assert '[shader("callable")]' in code
    assert "void CallableMain(inout CallableData data)" in code
    assert '[shader("intersection")]' in code
    assert "void IntersectionMain()" in code
    assert "bool accepted = ReportHit(1.0, 0, attributes);" in code
    assert "payload.color = float3(attributes.barycentrics, 1.0);" in code
    assert "payload.color = float3(attributes.barycentrics, 0.5);" in code
    assert "payload.color = float3(0.0, 0.0, 0.0);" in code
    assert "data.value = data.value + 1u;" in code
    assert ": payload" not in code
    assert ": hit_attribute" not in code
    assert "rayPayloadInEXT" not in code
    assert "callableDataInEXT" not in code
    assert "gl_LaunchIDEXT" not in code
    assert "gl_LaunchSizeEXT" not in code
    shader_path.write_text(code, encoding="utf-8")

    slangc = _require_tool("slangc")
    for stage, entry in [
        ("raygeneration", "RayGenMain"),
        ("closesthit", "ClosestHitMain"),
        ("anyhit", "AnyHitMain"),
        ("miss", "MissMain"),
        ("callable", "CallableMain"),
        ("intersection", "IntersectionMain"),
    ]:
        _compile_slang_hlsl_entry(
            slangc,
            shader_path,
            tmp_path / f"slang_ray_stage_library_{stage}.hlsl",
            entry,
            stage,
            "sm_6_3",
        )


@pytest.mark.parametrize(
    ("case_name", "source", "expected_snippets", "forbidden_snippets"),
    [
        (
            "uint",
            MIXED_GLSL_SSBO_UINT_ATOMICS_COMPUTE_SHADER,
            (
                "RWByteAddressBuffer atomicBlock : register(u17);",
                "__crossgl_byteaddress_atomic_compare_exchange_uint",
                "InterlockedAdd",
            ),
            ("unsupported HLSL GLSL buffer block atomic", "#version"),
        ),
        (
            "int",
            MIXED_GLSL_SSBO_INT_ATOMICS_COMPUTE_SHADER,
            (
                "RWByteAddressBuffer signedAtomicBlock : register(u18);",
                "__crossgl_byteaddress_atomic_compare_exchange_int",
                "InterlockedMin",
            ),
            ("unsupported HLSL GLSL buffer block atomic", "#version"),
        ),
        (
            "runtime_array",
            MIXED_GLSL_SSBO_RUNTIME_ARRAY_ATOMICS_COMPUTE_SHADER,
            (
                "runtimeAtomicBlock, (4 + i * 4)",
                "runtimeSignedAtomicBlock, (4 + j * 4)",
                "__crossgl_byteaddress_atomic_compare_exchange_uint",
            ),
            ("unsupported HLSL GLSL buffer block atomic", "#version"),
        ),
        (
            "unsupported",
            MIXED_GLSL_SSBO_UNSUPPORTED_ATOMICS_COMPUTE_SHADER,
            (
                "ByteAddressBuffer readAtomicBlock : register(t21);",
                "unsupported HLSL GLSL buffer block atomic",
                "float floatOld = /* unsupported HLSL GLSL buffer block atomic",
                "*/ 0;",
            ),
            ("Interlocked", "__crossgl_byteaddress_atomic", "#version"),
        ),
    ],
)
def test_mixed_glsl_ssbo_atomics_hlsl_output_compiles_with_dxc(
    tmp_path,
    case_name,
    source,
    expected_snippets,
    forbidden_snippets,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / f"mixed_glsl_ssbo_{case_name}_atomics.hlsl"
    output_path = tmp_path / f"mixed_glsl_ssbo_{case_name}_atomics.dxil"

    code = HLSLCodeGen().generate(_mixed_glsl_ast(source, "compute"))
    for snippet in expected_snippets:
        assert snippet in code
    for snippet in forbidden_snippets:
        assert snippet not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_std140_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_std140.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_std140.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_STD140_COMPUTE_SHADER, "compute")
    )
    assert "RWByteAddressBuffer std140Block : register(u25);" in code
    assert "std140Block.Load2(16)" in code
    assert "std140Block.Load2(32)" in code
    assert "std140Block.Load(80)" in code
    assert "std140Block.Load((96 + i * 16))" in code
    assert "unsupported HLSL GLSL buffer block" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_bool_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_bool.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_bool.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_BOOL_COMPUTE_SHADER, "compute")
    )
    assert "RWByteAddressBuffer boolBlock : register(u26);" in code
    assert "boolBlock.Load(0) != 0u" in code
    assert "boolBlock.Load((4 + i * 4)) != 0u" in code
    assert "boolBlock.Store(8, ((false) ? 1u : 0u));" in code
    assert "boolBlock.Store((12 + i * 4), asuint" in code
    assert "unsupported HLSL GLSL buffer block" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_glsl_fragment_validates_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "validator_smoke.frag"

    shader_path.write_text(
        GLSLCodeGen().generate_stage(_fragment_ast(), "fragment"),
        encoding="utf-8",
    )

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_fragment_sample_builtins_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "sample_builtins.frag"
    shader = """
    shader GLSLFragmentSampleBuiltinsValidator {
        fragment {
            vec4 main(
                int sampleIndex @SV_SampleIndex,
                vec2 samplePosition @sample_position,
                int coverage @SV_Coverage
            ) @gl_FragColor {
                return vec4(samplePosition, float(sampleIndex), float(coverage));
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")
    assert "gl_SampleID" in code
    assert "gl_SamplePosition" in code
    assert "gl_SampleMaskIn[0]" in code
    assert "gl_SampleMask)" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_fragment_interpolation_helpers_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "interpolation_helpers.frag"
    shader = """
    shader GLSLFragmentInterpolationHelpersValidator {
        fragment {
            vec4 main(
                vec4 sampleColor @location(0) @sample,
                vec2 offset @location(1)
            ) @gl_FragColor {
                return interpolate_at_sample(sampleColor, 0)
                    + interpolate_at_offset(sampleColor, offset)
                    + interpolate_at_centroid(sampleColor);
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")
    assert "layout(location = 0) sample in vec4 sampleColor;" in code
    assert "interpolateAtSample(sampleColor, 0)" in code
    assert "interpolateAtOffset(sampleColor, offset)" in code
    assert "interpolateAtCentroid(sampleColor)" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_fragment_derivative_helpers_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "derivative_helpers.frag"
    shader = """
    shader GLSLFragmentDerivativeHelpersValidator {
        sampler2D colorMap @binding(0);

        vec2 wrappedGradientX(vec2 uv) {
            return gradientX(uv);
        }

        fragment {
            vec2 gradientX(vec2 uv) {
                return ddx(uv);
            }

            vec4 main(vec2 uv @location(0)) @gl_FragColor {
                float dx = wrappedGradientX(uv).x;
                float fineY = ddy_fine(uv.y);
                float coarseWidth = fwidth_coarse(uv.x);
                vec4 sampled = textureGrad(colorMap, uv, wrappedGradientX(uv), ddy(uv));
                return sampled + vec4(dx + fineY + coarseWidth);
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")
    assert "return dFdx(uv);" in code
    assert "return gradientX(uv);" in code
    assert "dFdyFine(uv.y)" in code
    assert "fwidthCoarse(uv.x)" in code
    assert "textureGrad(colorMap, uv, wrappedGradientX(uv), dFdy(uv))" in code
    assert "ddx(" not in code
    assert "ddy_fine" not in code
    assert "fwidth_coarse" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_projected_gradient_offsets_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "projected_gradient_offsets.frag"
    shader = """
    shader GLSLProjectedGradientOffsetValidator {
        sampler2D colorMap @binding(0);
        sampler3D volumeMap @binding(1);

        fragment {
            vec4 main(
                vec2 uv @location(0),
                vec4 uvq @location(1),
                vec4 xyzq @location(2),
                vec2 ddx2 @location(3),
                vec2 ddy2 @location(4),
                vec3 ddx3 @location(5),
                vec3 ddy3 @location(6)
            ) @gl_FragColor {
                const ivec2 offset2 = ivec2(1, 0);
                const ivec3 offset3 = ivec3(1, 0, -1);
                vec4 gradOffset = textureGradOffset(
                    colorMap,
                    uv,
                    ddx2,
                    ddy2,
                    offset2
                );
                vec4 projectedGrad = textureProjGrad(colorMap, uvq, ddx2, ddy2);
                vec4 projectedGradOffset = textureProjGradOffset(
                    colorMap,
                    uvq,
                    ddx2,
                    ddy2,
                    offset2
                );
                vec4 volumeProjectedGrad = textureProjGrad(
                    volumeMap,
                    xyzq,
                    ddx3,
                    ddy3
                );
                vec4 volumeProjectedGradOffset = textureProjGradOffset(
                    volumeMap,
                    xyzq,
                    ddx3,
                    ddy3,
                    offset3
                );
                return gradOffset
                    + projectedGrad
                    + projectedGradOffset
                    + volumeProjectedGrad
                    + volumeProjectedGradOffset;
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")
    assert "layout(binding = 0) uniform sampler2D colorMap;" in code
    assert "layout(binding = 1) uniform sampler3D volumeMap;" in code
    assert "const ivec2 offset2 = ivec2(1, 0);" in code
    assert "const ivec3 offset3 = ivec3(1, 0, (-1));" in code
    assert "textureGradOffset(colorMap, uv, ddx2, ddy2, offset2)" in code
    assert "textureProjGrad(colorMap, uvq, ddx2, ddy2)" in code
    assert "textureProjGradOffset(colorMap, uvq, ddx2, ddy2, offset2)" in code
    assert "textureProjGrad(volumeMap, xyzq, ddx3, ddy3)" in code
    assert "textureProjGradOffset(volumeMap, xyzq, ddx3, ddy3, offset3)" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_const_gather_offsets_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "const_gather_offsets.frag"
    shader = """
    shader ConstGatherOffsets {
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherColor(sampler2D tex, vec2 uv, int component) {
            const ivec2 offsets[4] = {
                ivec2(-1, -1),
                ivec2(1, -1),
                ivec2(-1, 1),
                ivec2(1, 1)
            };
            return textureGatherOffsets(tex, uv, offsets, component);
        }

        vec4 gatherShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth) {
            const int left = -1;
            const ivec2 offsets[4] = {
                ivec2(left, -1),
                ivec2(1, -1),
                ivec2(left, 1),
                ivec2(1, 1)
            };
            return textureGatherCompareOffsets(tex, s, uv, depth, offsets);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherColor(colorMap, input.uv, input.component)
                    + gatherShadow(shadowMap, compareSampler, input.uv, input.depth);
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")

    assert "textureGatherOffsets(" not in code
    assert "textureGatherCompareOffsets(" not in code
    assert "unsupported GLSL texture gather" not in code
    assert "textureGatherOffset" in code

    shader_path.write_text(code, encoding="utf-8")
    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_mixed_glsl_fragment_multiple_outputs_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "mixed_glsl_fragment_multiple_outputs.frag"

    code = GLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_FRAGMENT_MULTIPLE_OUTPUTS_SHADER, "fragment")
    )
    assert "layout(location = 0) in vec2 uv;" in code
    assert "layout(location = 0, index = 0) out vec4 accum;" in code
    assert "layout(location = 0, index = 1) out vec4 revealage;" in code
    assert "layout(location = 2) out vec4 normal;" in code
    assert "accum = vec4(uv, 0.0, 1.0);" in code
    assert "revealage = vec4(1.0);" in code
    assert "normal = vec4(0.0);" in code
    assert "fragColor" not in code
    assert "return accum" not in code
    assert "\n    vec4 accum;" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_mixed_glsl_fragment_component_packing_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "mixed_glsl_fragment_component_packing.frag"

    code = GLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_FRAGMENT_COMPONENT_PACKING_SHADER, "fragment")
    )
    assert "layout(location = 0) in vec2 uv;" in code
    assert "layout(location = 0, component = 0) out float luminance;" in code
    assert "layout(location = 0, component = 1) out vec2 velocity;" in code
    assert "layout(location = 0, component = 3) out float coverage;" in code
    assert "layout(location = 1) out vec4 color;" in code
    assert "luminance = uv.x;" in code
    assert "velocity = uv;" in code
    assert "coverage = uv.y;" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_mixed_glsl_fragment_blend_support_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "mixed_glsl_fragment_blend_support.frag"

    code = GLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_FRAGMENT_BLEND_SUPPORT_SHADER, "fragment")
    )
    assert "#extension GL_KHR_blend_equation_advanced : enable" in code
    assert (
        "layout(blend_support_colordodge, blend_support_multiply, "
        "blend_support_screen) out;" in code
    )
    assert "layout(location = 0) out highp vec4 outputColour;" in code
    assert "layout(location = 1) out vec4 overlayColour;" in code
    assert "outputColour = vec4(uv, 0.0, 1.0);" in code
    assert "overlayColour = vec4(0.25);" in code
    assert "fragColor" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator_or_skip_unsupported_extension(
        [glslang, "-S", "frag", str(shader_path)],
        "GL_KHR_blend_equation_advanced",
        unsupported_diagnostics=("invalid layout qualifier", "blend_support_"),
    )


def test_mixed_glsl_fragment_color_depth_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "mixed_glsl_fragment_color_depth.frag"

    code = GLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_FRAGMENT_COLOR_DEPTH_SHADER, "fragment")
    )
    assert "layout(location = 0) in vec2 uv;" in code
    assert "layout(location = 0) out vec4 color;" in code
    assert "color = vec4(uv, 0.0, 1.0);" in code
    assert "gl_FragDepth = uv.x;" in code
    assert "fragColor" not in code
    assert "return color" not in code
    assert "\n    vec4 color;" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_mixed_glsl_150_fragment_output_validates_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "mixed_glsl_150_fragment_output.frag"

    code = GLSLCodeGen().generate_stage(
        _mixed_glsl_ast(MIXED_GLSL_150_FRAGMENT_OUTPUT_SHADER, "fragment"),
        "fragment",
    )
    assert code.lstrip().startswith("#version 330 core\n")
    assert "#version 150" not in code
    assert "layout(location = 0) out vec4 fragColor;" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_compute_synchronization_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "synchronization.comp"

    code = GLSLCodeGen().generate(
        crosstl.translator.parse(CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    assert code.count("barrier();") == 2
    assert "memoryBarrier();" in code
    assert "workgroupBarrier();" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_wave_subgroups_validate_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "wave_subgroups.comp"
    shader = """
    shader ExternalValidatorWaveSubgroups {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint count = WaveGetLaneCount();
                uint sumValue = WaveActiveSum(lane);
                uint prefixValue = WavePrefixSum(sumValue);
                bool first = WaveIsFirstLane();
                bool anyLane = WaveActiveAnyTrue(prefixValue > 0u);
                bool allLane = WaveActiveAllTrue(count > 0u);
                uvec4 ballot = WaveActiveBallot(anyLane || allLane || first);
                uint broadcast = WaveReadLaneAt(prefixValue, 0u);
                uint firstValue = WaveReadLaneFirst(broadcast + ballot.x);
            }
        }
    }
    """

    code = GLSLCodeGen().generate(crosstl.translator.parse(shader))
    assert "#extension GL_KHR_shader_subgroup_basic : require" in code
    assert "#extension GL_KHR_shader_subgroup_arithmetic : require" in code
    assert "#extension GL_KHR_shader_subgroup_ballot : require" in code
    assert "#extension GL_KHR_shader_subgroup_shuffle : require" in code
    assert "Wave" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_geometry_tessellation_layouts_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    ast = crosstl.translator.parse(GLSL_GEOMETRY_TESSELLATION_LAYOUT_SHADER)
    generator = GLSLCodeGen()

    stage_cases = [
        ("geometry", "geom", "layout(points) in;"),
        ("tessellation_control", "tesc", "layout(vertices = 4) out;"),
        (
            "tessellation_evaluation",
            "tese",
            "layout(triangles, fractional_odd_spacing, cw) in;",
        ),
    ]

    for stage_name, validator_stage, expected_layout in stage_cases:
        shader_path = tmp_path / f"layout_{validator_stage}.glsl"
        code = generator.generate_stage(ast, stage_name)
        assert expected_layout in code
        shader_path.write_text(code, encoding="utf-8")

        _run_validator([glslang, "-S", validator_stage, str(shader_path)])


def test_generated_glsl_geometry_inputtopology_metadata_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "geometry_inputtopology.geom"
    ast = crosstl.translator.parse("""
        shader GLSLGeometryInputTopologyValidator {
            geometry {
                void main()
                    @inputtopology(triangle)
                    @invocations(2)
                    @outputtopology(line)
                    @maxvertexcount(2)
                {
                    gl_Position = gl_in[0].gl_Position;
                    EmitVertex();
                    gl_Position = gl_in[1].gl_Position;
                    EmitVertex();
                    EndPrimitive();
                }
            }
        }
        """)

    code = GLSLCodeGen().generate_stage(ast, "geometry")
    assert "layout(triangles, invocations = 2) in;" in code
    assert "layout(line_strip, max_vertices = 2) out;" in code
    assert "inputtopology" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "geom", str(shader_path)])


def test_generated_glsl_tessellation_outputtopology_metadata_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "tessellation_outputtopology.tese"
    ast = crosstl.translator.parse("""
        shader GLSLTessellationOutputTopologyValidator {
            tessellation_evaluation {
                void main()
                    @domain(triangle)
                    @partitioning(fractional_even)
                    @outputtopology(triangle_ccw)
                {
                    gl_Position = vec4(gl_TessCoord, 1.0);
                }
            }
        }
        """)

    code = GLSLCodeGen().generate_stage(ast, "tessellation_evaluation")
    assert "layout(triangles, fractional_even_spacing, ccw) in;" in code
    assert "outputtopology" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "tese", str(shader_path)])


@pytest.mark.parametrize(
    (
        "case_name",
        "source",
        "shader_type",
        "validator_stage",
        "expected_snippets",
        "forbidden_snippets",
    ),
    [
        (
            "geometry_interface_array",
            MIXED_GLSL_GEOMETRY_INTERFACE_ARRAY_SHADER,
            "geometry",
            "geom",
            (
                "layout(lines_adjacency, invocations = 2) in;",
                "layout(triangle_strip, max_vertices = 6) out;",
                "in vec3 vColor[];",
                "flat in int vLayer[];",
                "out vec3 gColor;",
                "flat out int gLayer;",
                "gl_Position = gl_in[i].gl_Position;",
            ),
            (
                "GeometryInput",
                "GeometryOutput",
                "// layout(",
                "input.",
                "output.",
                "return output;",
            ),
        ),
        (
            "tessellation_control_interface_array",
            MIXED_GLSL_TESSELLATION_CONTROL_INTERFACE_ARRAY_SHADER,
            "tessellation_control",
            "tesc",
            (
                "layout(vertices = 4) out;",
                "in vec3 vPosition[];",
                "out vec3 tcPosition[];",
                "patch out vec4 tcPatchColor;",
                "gl_out[gl_InvocationID].gl_Position = "
                "gl_in[gl_InvocationID].gl_Position;",
            ),
            (
                "TessellationControlInput",
                "TessellationControlOutput",
                "// layout(",
                "input.",
                "output.",
                "return output;",
            ),
        ),
        (
            "tessellation_evaluation_interface_array",
            MIXED_GLSL_TESSELLATION_EVALUATION_INTERFACE_ARRAY_SHADER,
            "tessellation_evaluation",
            "tese",
            (
                "layout(quads, fractional_even_spacing, ccw, point_mode) in;",
                "in vec3 tcPosition[];",
                "patch in vec4 tcPatchColor;",
                "out vec4 teColor;",
                "gl_Position = vec4(p, 1.0);",
            ),
            (
                "TessellationEvaluationInput",
                "TessellationEvaluationOutput",
                "// layout(",
                "input.",
                "output.",
                "return output;",
            ),
        ),
    ],
)
def test_mixed_glsl_geometry_tessellation_interfaces_validate_with_glslangvalidator(
    tmp_path,
    case_name,
    source,
    shader_type,
    validator_stage,
    expected_snippets,
    forbidden_snippets,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / f"mixed_glsl_{case_name}.{validator_stage}"

    code = GLSLCodeGen().generate(_mixed_glsl_ast(source, shader_type))
    for snippet in expected_snippets:
        assert snippet in code
    for snippet in forbidden_snippets:
        assert snippet not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", validator_stage, str(shader_path)])


def test_generated_glsl_geometry_interface_blocks_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "geometry_interface_block.geom"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_GEOMETRY_INTERFACE_BLOCK_VALIDATOR_SHADER),
        "geometry",
    )
    assert "in VertexIn {" in code
    assert "flat vec2 inputUv;" in code
    assert "} vertexIn[];" in code
    assert "out FragmentOut {" in code
    assert "noperspective vec2 outUv;" in code
    assert "} fragmentOut;" in code
    assert "in VertexIn vertexIn[]" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "geom", str(shader_path)])


def test_generated_glsl_geometry_interface_block_multidimensional_arrays_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "geometry_interface_block_multidimensional.geom"

    shader = """
    shader GLSLGeometryInterfaceBlockMultidimensionalArraysValidator {
        @glsl_interface_block(in) @glsl_interface_instance(vertexIn) @glsl_interface_array
        struct VertexIn {
            flat ivec2 ids[2][2];
            noperspective vec3 positions[2];
        };

        @glsl_interface_block(out) @glsl_interface_instance(fragmentOut)
        struct FragmentOut {
            noperspective vec4 colors[2][2];
        };

        geometry {
            layout(points) in;
            layout(points, max_vertices = 1) out;

            void main() {
                fragmentOut.colors[1][0] = vec4(vertexIn[0].positions[1], 1.0)
                    + vec4(vertexIn[0].ids[0][1], 0.0, 0.0);
                gl_Position = gl_in[0].gl_Position;
                EmitVertex();
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "geometry")
    assert "in VertexIn {" in code
    assert "flat ivec2 ids[2][2];" in code
    assert "noperspective vec3 positions[2];" in code
    assert "} vertexIn[];" in code
    assert "noperspective vec4 colors[2][2];" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "geom", str(shader_path)])


def test_mixed_glsl_tessellation_multidimensional_patch_arrays_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    cases = [
        (
            "tesc",
            "tessellation_control",
            """
            #version 450 core
            layout(vertices = 4) out;

            in vec3 vPosition[];
            out vec3 tcGrid[][2];
            patch out vec4 tcPatchColors[2][2];

            void main() {
                tcGrid[gl_InvocationID][0] = vPosition[gl_InvocationID];
                tcGrid[gl_InvocationID][1] = vec3(1.0);
                gl_out[gl_InvocationID].gl_Position =
                    gl_in[gl_InvocationID].gl_Position;
                tcPatchColors[0][1] = vec4(1.0);
                gl_TessLevelOuter[0] = 2.0;
                gl_TessLevelOuter[1] = 2.0;
                gl_TessLevelOuter[2] = 2.0;
                gl_TessLevelOuter[3] = 2.0;
                gl_TessLevelInner[0] = 2.0;
                gl_TessLevelInner[1] = 2.0;
            }
            """,
            (
                "out vec3 tcGrid[][2];",
                "patch out vec4 tcPatchColors[2][2];",
            ),
        ),
        (
            "tese",
            "tessellation_evaluation",
            """
            #version 450 core
            layout(quads, equal_spacing, ccw) in;

            in vec3 tcGrid[][2];
            patch in vec4 tcPatchColors[2][2];
            out vec4 teColor;

            void main() {
                vec3 p = mix(tcGrid[0][0], tcGrid[1][1], gl_TessCoord.x);
                teColor = tcPatchColors[0][1];
                gl_Position = vec4(p, 1.0);
            }
            """,
            (
                "in vec3 tcGrid[][2];",
                "patch in vec4 tcPatchColors[2][2];",
            ),
        ),
    ]

    for stage_suffix, shader_type, source, expected_snippets in cases:
        shader_path = tmp_path / f"tessellation_multidimensional_arrays.{stage_suffix}"
        code = GLSLCodeGen().generate(_mixed_glsl_ast(source, shader_type))
        for snippet in expected_snippets:
            assert snippet in code
        assert "vec3[] tcGrid" not in code
        assert "vec4[2][2] tcPatchColors" not in code
        shader_path.write_text(code, encoding="utf-8")

        _run_validator([glslang, "-S", stage_suffix, str(shader_path)])


def test_generated_webgl_graphics_validate_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    vertex_path = tmp_path / "webgl_graphics.vert"
    fragment_path = tmp_path / "webgl_graphics.frag"

    ast = crosstl.translator.parse(CROSSGL_WGSL_GRAPHICS_SHADER)
    vertex_code = WebGLCodeGen().generate_program(ast, target_stage="vertex")
    fragment_code = WebGLCodeGen().generate_program(ast, target_stage="fragment")

    assert "#version 300 es" in vertex_code
    assert "#version 300 es" in fragment_code
    assert "layout(location = 5) out vec2 out_uv;" not in vertex_code
    assert "layout(location = 5) in vec2 uv;" not in fragment_code
    vertex_path.write_text(vertex_code, encoding="utf-8")
    fragment_path.write_text(fragment_code, encoding="utf-8")

    _run_validator([glslang, "-S", "vert", str(vertex_path)])
    _run_validator([glslang, "-S", "frag", str(fragment_path)])


def test_generated_glsl_stage_io_multidimensional_arrays_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "stage_io_multidimensional_arrays.vert"

    shader = """
    shader GLSLStageIOMultidimensionalArraysValidator {
        vertex {
            struct VertexInput {
                vec3 positions[2][3];
                ivec2 ids[2][2];
            }

            struct VertexOutput {
                vec4 colors[2][3];
                vec4 position @ gl_Position;
            }

            VertexOutput main(VertexInput input) {
                VertexOutput output;
                output.colors[1][2] = vec4(input.positions[0][1], 1.0);
                output.position = output.colors[1][2];
                return output;
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "vertex")
    assert "in vec3 positions[2][3];" in code
    assert "in ivec2 ids[2][2];" in code
    assert "out vec4 colors[2][3];" in code
    assert "vec3[2][3] positions" not in code
    assert "ivec2[2][2] ids" not in code
    assert "vec4[2][3] colors" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "vert", str(shader_path)])


def test_generated_glsl_stage_io_interpolation_arrays_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    vertex_path = tmp_path / "stage_io_interpolation_arrays.vert"
    fragment_path = tmp_path / "stage_io_interpolation_arrays.frag"

    shader = """
    shader GLSLStageIOInterpolationArraysValidator {
        vertex {
            struct VertexInput {
                vec3 position @location(0);
            }

            struct VertexOutput {
                ivec2 ids[2][2] @location(1);
                vec4 samples[2] @location(5) @sample;
                vec3 centers[2] @location(7) @centroid;
                vec2 noPersp[2] @location(9) @noperspective;
                vec4 position @gl_Position;
            }

            VertexOutput main(VertexInput input) {
                VertexOutput output;
                output.ids[0][0] = ivec2(1, 2);
                output.samples[0] = vec4(input.position, 1.0);
                output.centers[0] = input.position;
                output.noPersp[0] = input.position.xy;
                output.position = vec4(input.position, 1.0);
                return output;
            }
        }

        fragment {
            vec4 main(
                ivec2 ids[2][2] @location(1),
                vec4 samples[2] @location(5) @sample,
                vec3 centers[2] @location(7) @centroid,
                vec2 noPersp[2] @location(9) @noperspective
            ) @gl_FragColor {
                return samples[0]
                    + vec4(centers[0], 1.0)
                    + vec4(noPersp[0], 0.0, 1.0)
                    + vec4(ids[0][0], 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    vertex_code = GLSLCodeGen().generate_stage(ast, "vertex")
    fragment_code = GLSLCodeGen().generate_stage(ast, "fragment")

    assert "layout(location = 1) flat out ivec2 ids[2][2];" in vertex_code
    assert "layout(location = 5) sample out vec4 samples[2];" in vertex_code
    assert "layout(location = 7) centroid out vec3 centers[2];" in vertex_code
    assert "layout(location = 1) flat in ivec2 ids[2][2];" in fragment_code
    assert "layout(location = 5) sample in vec4 samples[2];" in fragment_code
    assert "layout(location = 7) centroid in vec3 centers[2];" in fragment_code
    vertex_path.write_text(vertex_code, encoding="utf-8")
    fragment_path.write_text(fragment_code, encoding="utf-8")

    _run_validator([glslang, "-S", "vert", str(vertex_path)])
    _run_validator([glslang, "-S", "frag", str(fragment_path)])


@pytest.mark.parametrize(
    ("topology", "max_vertices", "index_assignment", "expected_layout"),
    [
        (
            "points",
            1,
            "gl_PrimitivePointIndicesEXT[0] = 0u;",
            "layout(points, max_vertices = 1, max_primitives = 1) out;",
        ),
        (
            "lines",
            2,
            "gl_PrimitiveLineIndicesEXT[0] = uvec2(0u, 1u);",
            "layout(lines, max_vertices = 2, max_primitives = 1) out;",
        ),
        (
            "triangles",
            3,
            "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);",
            "layout(triangles, max_vertices = 3, max_primitives = 1) out;",
        ),
    ],
)
def test_generated_glsl_mesh_shader_validates_with_glslangvalidator(
    tmp_path,
    topology,
    max_vertices,
    index_assignment,
    expected_layout,
):
    glslang = _require_tool("glslangValidator")
    _require_glslang_stage(glslang, "mesh")
    shader_path = tmp_path / f"mesh_shader_{topology}.mesh"
    shader = f"""
    shader GLSLMeshValidator {{
        mesh {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout({topology}, max_vertices = {max_vertices}, max_primitives = 1) out;

            void main() {{
                SetMeshOutputCounts({max_vertices}, 1);
                gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                {index_assignment}
            }}
        }}
    }}
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")
    assert "#extension GL_EXT_mesh_shader : require" in code
    assert expected_layout in code
    assert f"SetMeshOutputsEXT({max_vertices}, 1);" in code
    assert "SetMeshOutputCounts" not in code
    assert index_assignment in code
    shader_path.write_text(code, encoding="utf-8")

    _run_glslang_mesh_task_validator(glslang, "mesh", shader_path)


def test_generated_glsl_mesh_output_signature_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    _require_glslang_stage(glslang, "mesh")
    shader_path = tmp_path / "mesh_output_signature.mesh"
    shader = """
    shader GLSLMeshOutputSignatureValidator {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            int primitiveId @ gl_PrimitiveID;
            vec3 normal @ NORMAL;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                verts[0].uv = vec2(0.5, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
                prims[0].primitiveId = 7;
                prims[0].normal = vec3(0.0, 0.0, 1.0);
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")
    assert "layout(location = 5) out vec2 uv[3];" in code
    assert "layout(location = 1) perprimitiveEXT out vec3 normal[1];" in code
    assert "gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);" in code
    assert "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);" in code
    assert "gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 7;" in code
    assert "verts[0]" not in code
    assert "tris[0]" not in code
    assert "prims[0]" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_glslang_mesh_task_validator(glslang, "mesh", shader_path)


def test_generated_glsl_mesh_whole_output_constructor_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    _require_glslang_stage(glslang, "mesh")
    shader_path = tmp_path / "mesh_whole_output_constructor.mesh"
    shader = """
    shader GLSLMeshWholeOutputConstructorValidator {
        struct MeshVertex {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            int primitiveId @ gl_PrimitiveID;
            vec3 normal @ NORMAL;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1)
              @outputtopology(triangle)
              @max_vertices(3)
              @max_primitives(1)
            {
                SetMeshOutputCounts(3, 1);
                verts[1] = MeshVertex {
                    position: vec4(1.0, 0.0, 0.0, 1.0),
                    uv: vec2(0.25, 0.75)
                };
                gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);
                prims[0] = MeshPrimitive {
                    primitiveId: 11,
                    normal: vec3(0.0, 1.0, 0.0)
                };
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")
    assert "gl_MeshVerticesEXT[1].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);" in code
    assert "uv[1] = vec2(0.25, 0.75);" in code
    assert "gl_MeshPrimitivesEXT[0].gl_PrimitiveID = 11;" in code
    assert "normal[0] = vec3(0.0, 1.0, 0.0);" in code
    assert "MeshVertex(" not in code
    assert "MeshPrimitive(" not in code
    assert "verts[1]" not in code
    assert "prims[0]" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_glslang_mesh_task_validator(glslang, "mesh", shader_path)


def test_generated_glsl_mesh_helper_intrinsics_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    _require_glslang_stage(glslang, "mesh")
    shader_path = tmp_path / "mesh_helper_intrinsics.mesh"
    shader = """
    shader GLSLMeshHelperIntrinsicValidator {
        mesh {
            void main()
                @numthreads(1, 1, 1)
                @outputtopology(triangle)
                @max_vertices(3)
                @max_primitives(1)
            {
                vec3 position = vec3(0.0, 0.5, 1.0);
                SetMeshOutputCounts(3, 1);
                SetVertex(0, position);
                SetVertex(1, vec4(1.0, 0.0, 0.0, 1.0));
                SetVertex(2, vec3(0.0, 1.0, 0.0));
                SetPrimitive(0, uvec3(0u, 1u, 2u));
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")
    assert "gl_MeshVerticesEXT[0].gl_Position = vec4(position, 1.0);" in code
    assert "gl_MeshVerticesEXT[1].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);" in code
    assert "gl_MeshVerticesEXT[2].gl_Position = vec4(vec3(0.0, 1.0, 0.0), 1.0);" in code
    assert "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);" in code
    assert "SetVertex" not in code
    assert "SetPrimitive" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_glslang_mesh_task_validator(glslang, "mesh", shader_path)


def test_generated_glsl_task_dispatch_payload_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    _require_glslang_stage(glslang, "task")
    shader_path = tmp_path / "task_dispatch_payload.task"
    shader = """
    shader GLSLTaskDispatchPayloadValidator {
        struct TaskPayload {
            uint meshlet;
        };

        task {
            @taskPayloadSharedEXT TaskPayload payload;
            void main() @numthreads(1, 1, 1) {
                TaskPayload localPayload;
                localPayload.meshlet = 7u;
                DispatchMesh(2, 3, 4, localPayload);
            }
        }
    }
    """

    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")
    assert "taskPayloadSharedEXT TaskPayload payload;" in code
    assert "payload = localPayload;" in code
    assert "EmitMeshTasksEXT(2, 3, 4);" in code
    assert "EmitMeshTasksEXT(2, 3, 4, localPayload)" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_glslang_mesh_task_validator(glslang, "task", shader_path)


def test_generated_glsl_ray_generation_validates_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    _require_glslang_stage(glslang, "rgen")
    shader_path = tmp_path / "ray_generation.rgen"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_RAY_GENERATION_VALIDATOR_SHADER),
        "ray_generation",
    )
    assert code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in code
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in code
    assert "layout(location = 0) rayPayloadEXT vec4 rayPayload;" in code
    assert "traceRayEXT(" in code
    assert "TraceRay" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            glslang,
            "-V",
            "--target-env",
            "vulkan1.2",
            "-S",
            "rgen",
            "-o",
            str(tmp_path / "ray_generation.spv"),
            str(shader_path),
        ]
    )


def test_generated_glsl_ray_query_compute_validates_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "ray_query_compute.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_RAY_QUERY_COMPUTE_VALIDATOR_SHADER),
        "compute",
    )
    assert code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_query : require" in code
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in code
    assert "rayQueryEXT rayQuery;" in code
    assert "bool active_ = rayQueryProceedEXT(rayQuery);" in code
    assert "bool active =" not in code
    assert "rayQueryGetIntersectionTypeEXT(rayQuery, true)" in code
    assert ".Proceed(" not in code
    assert ".CommittedType(" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            glslang,
            "-V",
            "--target-env",
            "vulkan1.2",
            "-S",
            "comp",
            "-o",
            str(tmp_path / "ray_query_compute.spv"),
            str(shader_path),
        ]
    )


def test_generated_glsl_ray_query_trace_ray_inline_validates_with_glslangvalidator(
    tmp_path,
):
    shader_path = tmp_path / "ray_query_trace_ray_inline.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_RAY_QUERY_TRACE_RAY_INLINE_VALIDATOR_SHADER),
        "compute",
    )
    assert code.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_query : require" in code
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in code
    assert "rayQueryEXT rayQuery;" in code
    assert (
        "rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsNoneEXT, 255u, "
        "ray.Origin, ray.TMin, ray.Direction, ray.TMax);"
    ) in code
    assert "bool active_ = rayQueryProceedEXT(rayQuery);" in code
    assert "bool active =" not in code
    assert ".TraceRayInline(" not in code
    assert ".Proceed(" not in code

    glslang = _require_tool("glslangValidator")
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            glslang,
            "-V",
            "--target-env",
            "vulkan1.2",
            "-S",
            "comp",
            "-o",
            str(tmp_path / "ray_query_trace_ray_inline.spv"),
            str(shader_path),
        ]
    )


def test_generated_glsl_multisample_storage_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "multisample_storage.comp"

    shader_path.write_text(
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(GLSL_MULTISAMPLE_STORAGE_COMPUTE_SHADER),
            "compute",
        ),
        encoding="utf-8",
    )

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_cube_storage_validates_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "cube_storage.comp"

    shader_path.write_text(
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(GLSL_CUBE_STORAGE_COMPUTE_SHADER),
            "compute",
        ),
        encoding="utf-8",
    )

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_parameter_image_atomic_specialization_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "parameter_image_atomic.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_PARAMETER_IMAGE_ATOMIC_COMPUTE_SHADER),
        "compute",
    )
    assert "imageAtomicAdd(image, pixel, value)" not in code
    assert "imageAtomicAdd(counters, pixel, value)" in code
    assert "addCounter__glsl_image_counters(ivec2(0, 1), 2u)" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_array_element_image_specialization_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "array_element_image_specialization.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            GLSL_ARRAY_ELEMENT_IMAGE_SPECIALIZATION_COMPUTE_SHADER
        ),
        "compute",
    )
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[2];" in code
    assert "int queryElement__glsl_image_counters_0()" in code
    assert "return imageSize(counters[0]).x;" in code
    assert "int queryElement__glsl_image_counters()" not in code
    assert "return imageSize(counters).x;" not in code
    assert "return queryElement__glsl_image_counters_0();" in code
    assert "imageStore(counters[1], ivec2(0, 0), uvec4(uint(" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_dynamic_image_array_helper_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "dynamic_image_array_helper.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_DYNAMIC_IMAGE_ARRAY_HELPER_COMPUTE_SHADER),
        "compute",
    )
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[2];" in code
    assert "int queryElement__glsl_image_counters_0()" in code
    assert "int queryElement__glsl_image_counters_1()" in code
    assert "int queryViaDynamic__glsl_images_counters(int layer)" in code
    assert "switch (layer)" in code
    assert "return queryElement__glsl_image_counters_0();" in code
    assert "return queryElement__glsl_image_counters_1();" in code
    assert "int queryViaInitializer__glsl_images_counters(int layer)" in code
    assert "int count;\n    switch (layer)" in code
    assert "count = queryElement__glsl_image_counters_0();" in code
    assert "count = queryElement__glsl_image_counters_1();" in code
    assert "int queryViaAssignment__glsl_images_counters(int layer)" in code
    assert "void storeViaExpression__glsl_images_counters(int layer)" in code
    assert "storeElement__glsl_image_counters_0(" in code
    assert "storeElement__glsl_image_counters_1(" in code
    assert "storeElement__glsl_image_counters_layer" not in code
    assert "queryElement__glsl_image_counters_layer" not in code
    assert "return imageSize(counters[layer]).x;" not in code
    assert "return queryElement(counters[layer]);" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_advanced_image_array_specialization_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "advanced_image_array_specialization.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            GLSL_ADVANCED_IMAGE_ARRAY_SPECIALIZATION_COMPUTE_SHADER
        ),
        "compute",
    )
    assert "layout(rgba16f, binding = 0) uniform image2DMS msImages[2];" in code
    assert "layout(r32ui, binding = 2) uniform uimage2DMS msCounters[2];" in code
    assert "layout(rgba16f, binding = 4) uniform imageCube cubeImages[2];" in code
    assert (
        "layout(rgba16f, binding = 6) uniform imageCubeArray cubeLayerImages[2];"
        in code
    )
    assert "return touchMS__glsl_image_msImages_1(pixel, sampleIndex, value);" in code
    assert "return bumpMS__glsl_image_msCounters_1(pixel, sampleIndex, value);" in code
    assert "return touchCube__glsl_image_cubeImages_1(coord, value);" in code
    assert "return touchCubeLayer__glsl_image_cubeLayerImages_1(coord, value);" in code
    assert "imageAtomicAdd(msCounters[1], pixel, sampleIndex, value)" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_dynamic_advanced_image_array_helper_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "dynamic_advanced_image_array_helper.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            GLSL_DYNAMIC_ADVANCED_IMAGE_ARRAY_HELPER_COMPUTE_SHADER
        ),
        "compute",
    )
    assert "layout(rgba16f, binding = 0) uniform image2DMS msImages[2];" in code
    assert "layout(r32ui, binding = 2) uniform uimage2DMS msCounters[2];" in code
    assert "layout(rgba16f, binding = 4) uniform imageCube cubeImages[2];" in code
    assert (
        "layout(rgba16f, binding = 6) uniform imageCubeArray cubeLayerImages[2];"
        in code
    )
    assert "switch (layer)" in code
    assert "return touchMS__glsl_image_msImages_0(pixel, sampleIndex, value);" in code
    assert "return touchMS__glsl_image_msImages_1(pixel, sampleIndex, value);" in code
    assert "return bumpMS__glsl_image_msCounters_0(pixel, sampleIndex, value);" in code
    assert "return bumpMS__glsl_image_msCounters_1(pixel, sampleIndex, value);" in code
    assert "return touchCube__glsl_image_cubeImages_0(coord, value);" in code
    assert "return touchCube__glsl_image_cubeImages_1(coord, value);" in code
    assert "return touchCubeLayer__glsl_image_cubeLayerImages_0(coord, value);" in code
    assert "return touchCubeLayer__glsl_image_cubeLayerImages_1(coord, value);" in code
    assert "return touchMS(msImages[layer], pixel, sampleIndex, value);" not in code
    assert "return bumpMS(msCounters[layer], pixel, sampleIndex, value);" not in code
    assert "return touchCube(cubeImages[layer], coord, value);" not in code
    assert "return touchCubeLayer(cubeLayerImages[layer], coord, value);" not in code
    assert "touchMS__glsl_image_msImages_layer" not in code
    assert "imageLoad(msImages[layer], pixel, sampleIndex)" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_storage_image_access_qualifiers_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "storage_image_access.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_STORAGE_IMAGE_ACCESS_COMPUTE_SHADER),
        "compute",
    )
    assert "layout(rgba32f, binding = 0) readonly uniform image2D source;" in code
    assert "layout(rgba32f, binding = 1) writeonly uniform image2D target;" in code
    assert "layout(r32ui, binding = 2) uniform uimage2D counters;" in code
    assert "layout(r32ui, binding = 2) readwrite uniform uimage2D" not in code
    assert "float readPixel(readonly image2D image, ivec2 pixel)" in code
    assert "void writePixel(writeonly image2D image, ivec2 pixel, vec4 value)" in code
    assert "uint bump(uimage2D image, ivec2 pixel, uint value)" in code
    assert "uint oldValue = imageAtomicAdd(counters, pixel, value);" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_buffer_block_access_qualifiers_validate_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "buffer_block_access.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_BUFFER_BLOCK_ACCESS_COMPUTE_SHADER),
        "compute",
    )
    assert "layout(std430, binding = 0) readonly buffer ReadonlyData" in code
    assert "layout(std430, binding = 1) writeonly buffer WriteonlyData" in code
    assert "layout(std430, binding = 2) buffer ReadwriteData" in code
    assert "readwrite buffer" not in code
    assert "uint value = readonlyData.value;" in code
    assert "writeonlyData.value = value;" in code
    assert "uint oldValue = atomicAdd(readwriteData.value, 1u);" in code
    assert "uint nestedOld = atomicAdd(readwriteData.nested.value, oldValue);" in code
    assert "uint itemOld = atomicAdd(readwriteData.items[1].value, nestedOld);" in code
    assert "uint arrayOld = atomicAdd(readwriteData.values[0], itemOld);" in code
    assert (
        "uint swapped = atomicCompSwap(readwriteData.values[1], uint(0), arrayOld);"
        in code
    )
    assert "readwriteData.value += oldValue;" in code
    assert "readwriteData.values[2] = swapped;" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_mixed_glsl_ssbo_std140_glsl_output_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "mixed_glsl_ssbo_std140.comp"

    code = GLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_STD140_COMPUTE_SHADER, "compute")
    )
    assert "layout(std140, binding = 25) buffer Std140Block" in code
    assert "float weights[3];" in code
    assert "float values[];" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_metal_fragment_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "validator_smoke.metal"
    output_path = tmp_path / "validator_smoke.air"

    shader_path.write_text(
        MetalCodeGen().generate_stage(_fragment_ast(), "fragment"),
        encoding="utf-8",
    )

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_metal_compute_synchronization_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "synchronization.metal"
    output_path = tmp_path / "synchronization.air"

    code = MetalCodeGen().generate(
        crosstl.translator.parse(CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    assert code.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
    assert "threadgroup_barrier(mem_flags::mem_device);" in code
    assert "workgroupBarrier();" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_std140_metal_output_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_std140.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_std140.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_STD140_COMPUTE_SHADER, "compute")
    )
    assert "device uchar* std140Block [[buffer(25)]]" in code
    assert "std140Block + 16" in code
    assert "std140Block + 32" in code
    assert "std140Block + 80" in code
    assert "std140Block + (96 + i * 16)" in code
    assert "unsupported Metal GLSL buffer block" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_bool_vector_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_bool_vector.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_bool_vector.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_BOOL_VECTOR_COMPUTE_SHADER, "compute")
    )
    assert "RWByteAddressBuffer boolVectorBlock : register(u27);" in code
    assert "bool3((boolVectorBlock.Load(0) != 0u)" in code
    assert "bool2((boolVectorBlock.Load((16 + i * 8)) != 0u)" in code
    assert "bool4((boolVectorBlock.Load((32 + i * 16)) != 0u)" in code
    assert "bool3 __crossgl_bool_store_0" in code
    assert "boolVectorBlock.Store3(0, uint3" in code
    assert "boolVectorBlock.Store2(16, uint2" in code
    assert "boolVectorBlock.Store4(48, uint4" in code
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_NESTED_STRUCT_COMPUTE_SHADER, "compute")
    )
    assert "RWByteAddressBuffer nestedBlock : register(u28);" in code
    assert "uint i = nestedBlock.Load(0);" in code
    assert "float scale = asfloat(nestedBlock.Load(16));" in code
    assert "bool3((nestedBlock.Load(32) != 0u)" in code
    assert "nestedBlock.Store(16, asuint((scale + 1.0)));" in code
    assert "nestedBlock.Store3(32, uint3" in code
    assert (
        "nestedBlock.Store((48 + i * 4), asuint(asfloat(nestedBlock.Load(16))))" in code
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_array_hlsl_output_compiles_with_dxc(
    tmp_path,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct_array.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct_array.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_STRUCT_ARRAY_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "RWByteAddressBuffer nestedArrayBlock : register(u29);" in code
    assert "uint i = nestedArrayBlock.Load(96);" in code
    assert "float3 normal = asfloat(nestedArrayBlock.Load3(64));" in code
    assert "nestedArrayBlock.Load((112 + i * 48 + 32)) != 0u" in code
    assert "nestedArrayBlock.Store(0, nestedArrayBlock.Load((112 + i * 48)))" in code
    assert "nestedArrayBlock.Store3((112 + i * 48 + 16), asuint(normal))" in code
    assert "nestedArrayBlock.Store2((112 + i * 48 + 32), uint2" in code
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_aggregate_hlsl_output_compiles_with_dxc(
    tmp_path,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_STRUCT_AGGREGATE_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "RWByteAddressBuffer aggregateBlock : register(u30);" in code
    assert re.search(
        r"AggregateBlockData __crossgl_load_rw_glsl_buffer_AggregateBlockData_[0-9a-f]{10}"
        r"\(RWByteAddressBuffer buffer, uint offset\)",
        code,
    )
    assert "result.payload.scale = asfloat(buffer.Load(offset));" in code
    assert re.search(
        r"AggregateBlockData inner = "
        r"__crossgl_load_rw_glsl_buffer_AggregateBlockData_[0-9a-f]{10}"
        r"\(aggregateBlock, 0\);",
        code,
    )
    assert re.search(
        r"AggregateBlockData item = "
        r"__crossgl_load_rw_glsl_buffer_AggregateBlockData_[0-9a-f]{10}"
        r"\(aggregateBlock, \(48 \+ i \* 48\)\);",
        code,
    )
    assert "AggregateBlockData __crossgl_aggregate_store_0 = item;" in code
    assert (
        "aggregateBlock.Store(0, "
        "asuint(__crossgl_aggregate_store_0.payload.scale))" in code
    )
    assert (
        "bool3 __crossgl_bool_store_1 = "
        "__crossgl_aggregate_store_0.payload.mask" in code
    )
    assert "aggregateBlock.Store(32, __crossgl_aggregate_store_0.id)" in code
    assert (
        "aggregateBlock.Store((48 + i * 48), "
        "asuint(__crossgl_aggregate_store_2.payload.scale))" in code
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_aggregate_array_hlsl_output_compiles_with_dxc(
    tmp_path,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate_array.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate_array.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_STRUCT_AGGREGATE_ARRAY_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "RWByteAddressBuffer arrayAggregateBlock : register(u16);" in code
    assert re.search(
        r"ArrayAggregateData __crossgl_load_rw_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(RWByteAddressBuffer buffer, uint offset\)",
        code,
    )
    assert "result.weights[0] = asfloat(buffer.Load(offset));" in code
    assert "result.items[1].flags = bool2" in code
    assert re.search(
        r"ArrayAggregateData inner = "
        r"__crossgl_load_rw_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(arrayAggregateBlock, 0\);",
        code,
    )
    assert re.search(
        r"ArrayAggregateData entry = "
        r"__crossgl_load_rw_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(arrayAggregateBlock, \(48 \+ i \* 48\)\);",
        code,
    )
    assert "arrayAggregateBlock.Store2(8, asuint" in code
    assert "arrayAggregateBlock.Store((48 + i * 48 + 40)" in code
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_aggregate_layout_helper_hlsl_output_compiles_with_dxc(
    tmp_path,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_aggregate_layout_helper.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_aggregate_layout_helper.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_AGGREGATE_LAYOUT_HELPER_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "RWByteAddressBuffer block430 : register(u14);" in code
    assert "RWByteAddressBuffer block140 : register(u15);" in code
    helper_names = re.findall(
        r"LayoutSharedData "
        r"(__crossgl_load_rw_glsl_buffer_LayoutSharedData_[0-9a-f]{10})"
        r"\(RWByteAddressBuffer buffer, uint offset\)",
        code,
    )
    assert len(helper_names) == 2
    assert len(set(helper_names)) == 2
    assert "result.weights[1] = asfloat(buffer.Load((offset + 4)));" in code
    assert "result.id = buffer.Load((offset + 8));" in code
    assert "result.weights[1] = asfloat(buffer.Load((offset + 16)));" in code
    assert "result.id = buffer.Load((offset + 32));" in code
    assert "block430.Store(4, asuint(__crossgl_aggregate_store_0.weights[1]))" in code
    assert "block140.Store(16, asuint(__crossgl_aggregate_store_1.weights[1]))" in code
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_readonly_aggregate_hlsl_output_compiles_with_dxc(
    tmp_path,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_readonly_aggregate.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_readonly_aggregate.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_READONLY_AGGREGATE_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "ByteAddressBuffer readAggregateBlock : register(t13);" in code
    assert "RWByteAddressBuffer readAggregateBlock" not in code
    assert re.search(
        r"ReadOnlyAggregateData __crossgl_load_ro_glsl_buffer_"
        r"ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(ByteAddressBuffer buffer, uint offset\)",
        code,
    )
    assert re.search(
        r"return __crossgl_load_ro_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(readAggregateBlock, \(48 \+ i \* 48\)\);",
        code,
    )
    assert re.search(
        r"ReadOnlyAggregateData inner = "
        r"__crossgl_load_ro_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(readAggregateBlock, 0\);",
        code,
    )
    assert "result.items[1].flags = bool2" in code
    assert "readAggregateBlock.Store" not in code
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_aggregate_leaf_compound_hlsl_output_compiles_with_dxc(
    tmp_path,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_aggregate_leaf_compound.hlsl"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_aggregate_leaf_compound.dxil"

    code = HLSLCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_AGGREGATE_LEAF_COMPOUND_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "RWByteAddressBuffer compoundAggregateBlock : register(u12);" in code
    assert "uint i = compoundAggregateBlock.Load(0);" in code
    assert (
        "compoundAggregateBlock.Store((8 + i * 48 + 4), "
        "asuint((asfloat(compoundAggregateBlock.Load((8 + i * 48 + 4))) + 1.0)))"
        in code
    )
    assert (
        "compoundAggregateBlock.Store2((8 + i * 48 + 8 + 16), "
        "asuint((asfloat(compoundAggregateBlock.Load2((8 + i * 48 + 8 + 16))) "
        "+ float2(0.5, 0.5))))" in code
    )
    assert "compoundAggregateBlock.Store2((8 + i * 48 + 8 + 8), uint2" in code
    assert ("un" + "supported HLSL GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_bool_metal_output_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_bool.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_bool.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_BOOL_COMPUTE_SHADER, "compute")
    )
    assert "device uchar* boolBlock [[buffer(26)]]" in code
    assert "reinterpret_cast<const device uint*>(boolBlock + 0)" in code
    assert "reinterpret_cast<const device uint*>(boolBlock + (4 + i * 4))" in code
    assert (
        "(*reinterpret_cast<device uint*>(boolBlock + 8)) = "
        "((false) ? 1u : 0u);" in code
    )
    assert "reinterpret_cast<device float*>(boolBlock + (12 + i * 4))" in code
    assert "unsupported Metal GLSL buffer block" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_bool_vector_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_bool_vector.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_bool_vector.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_BOOL_VECTOR_COMPUTE_SHADER, "compute")
    )
    assert "device uchar* boolVectorBlock [[buffer(27)]]" in code
    assert "reinterpret_cast<const device uint*>(boolVectorBlock + 0)" in code
    assert (
        "reinterpret_cast<const device uint*>(boolVectorBlock + (16 + i * 8))" in code
    )
    assert (
        "reinterpret_cast<const device uint*>(boolVectorBlock + (32 + i * 16))" in code
    )
    assert "bool3 __crossgl_buffer_store_0" in code
    assert "reinterpret_cast<device uint*>(boolVectorBlock + 8)" in code
    assert "bool2 __crossgl_buffer_store_1" in code
    assert "reinterpret_cast<device uint*>(boolVectorBlock + 20)" in code
    assert "bool4 __crossgl_buffer_store_2" in code
    assert "reinterpret_cast<device uint*>(boolVectorBlock + 60)" in code
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_NESTED_STRUCT_COMPUTE_SHADER, "compute")
    )
    assert "device uchar* nestedBlock [[buffer(28)]]" in code
    assert "uint i = (*reinterpret_cast<const device uint*>(nestedBlock + 0));" in code
    assert "reinterpret_cast<const device float*>(nestedBlock + 16)" in code
    assert "reinterpret_cast<const device uint*>(nestedBlock + 32)" in code
    assert "reinterpret_cast<device float*>(nestedBlock + 16)" in code
    assert "bool3 __crossgl_buffer_store_0" in code
    assert "reinterpret_cast<device uint*>(nestedBlock + 40)" in code
    assert "reinterpret_cast<device float*>(nestedBlock + (48 + i * 4))" in code
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_array_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct_array.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct_array.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_STRUCT_ARRAY_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "device uchar* nestedArrayBlock [[buffer(29)]]" in code
    assert (
        "uint i = (*reinterpret_cast<const device uint*>"
        "(nestedArrayBlock + 96));" in code
    )
    assert "reinterpret_cast<const device float*>(nestedArrayBlock + 64)" in code
    assert (
        "reinterpret_cast<const device uint*>"
        "(nestedArrayBlock + (112 + i * 48))" in code
    )
    assert "reinterpret_cast<device uint*>(nestedArrayBlock + 0)" in code
    assert (
        "reinterpret_cast<device float*>"
        "(nestedArrayBlock + (112 + i * 48 + 16))" in code
    )
    assert "bool2 __crossgl_buffer_store_1" in code
    assert (
        "reinterpret_cast<device uint*>"
        "(nestedArrayBlock + (112 + i * 48 + 32 + 4))" in code
    )
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_aggregate_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_STRUCT_AGGREGATE_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "device uchar* aggregateBlock [[buffer(30)]]" in code
    assert "AggregateBlockData inner = AggregateBlockData{AggregatePayload" in code
    assert (
        "AggregateBlockData item = AggregateBlockData{"
        "AggregatePayload{(*reinterpret_cast<const device float*>"
        "(aggregateBlock + (48 + i * 48)))" in code
    )
    assert "AggregateBlockData __crossgl_aggregate_store_0 = item;" in code
    assert (
        "reinterpret_cast<device float*>(aggregateBlock + 0)"
        ") = __crossgl_aggregate_store_0.payload.scale" in code
    )
    assert (
        "bool3 __crossgl_buffer_store_1 = "
        "__crossgl_aggregate_store_0.payload.mask" in code
    )
    assert (
        "reinterpret_cast<device uint*>(aggregateBlock + 32)"
        ") = __crossgl_aggregate_store_0.id" in code
    )
    assert "AggregateBlockData __crossgl_aggregate_store_2 = inner;" in code
    assert (
        "reinterpret_cast<device float*>(aggregateBlock + (48 + i * 48))"
        ") = __crossgl_aggregate_store_2.payload.scale" in code
    )
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_struct_aggregate_array_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate_array.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_struct_aggregate_array.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_STRUCT_AGGREGATE_ARRAY_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "device uchar* arrayAggregateBlock [[buffer(16)]]" in code
    assert re.search(
        r"ArrayAggregateData __crossgl_load_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(const device uchar\* buffer, uint offset\)",
        code,
    )
    assert "result.weights[0] =" in code
    assert "result.items[1].flags = bool2" in code
    assert re.search(
        r"ArrayAggregateData inner = "
        r"__crossgl_load_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(arrayAggregateBlock, 0\);",
        code,
    )
    assert re.search(
        r"ArrayAggregateData entry = "
        r"__crossgl_load_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(arrayAggregateBlock, \(48 \+ i \* 48\)\);",
        code,
    )
    assert "float2 __crossgl_buffer_store_1" in code
    assert "bool2 __crossgl_buffer_store_2" in code
    assert "arrayAggregateBlock + (48 + i * 48 + 40)" in code
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_aggregate_layout_helper_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_aggregate_layout_helper.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_aggregate_layout_helper.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_AGGREGATE_LAYOUT_HELPER_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "device uchar* block430 [[buffer(14)]]" in code
    assert "device uchar* block140 [[buffer(15)]]" in code
    helper_names = re.findall(
        r"LayoutSharedData "
        r"(__crossgl_load_glsl_buffer_LayoutSharedData_[0-9a-f]{10})"
        r"\(const device uchar\* buffer, uint offset\)",
        code,
    )
    assert len(helper_names) == 2
    assert len(set(helper_names)) == 2
    assert "result.weights[1] =" in code
    assert "buffer + (offset + 4)" in code
    assert "buffer + (offset + 8)" in code
    assert "buffer + (offset + 16)" in code
    assert "buffer + (offset + 32)" in code
    assert "block430 + 4" in code
    assert "block140 + 16" in code
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_readonly_aggregate_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_readonly_aggregate.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_readonly_aggregate.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_READONLY_AGGREGATE_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "const device uchar* readAggregateBlock [[buffer(13)]]" in code
    assert "kernel void kernel_main(device uchar* readAggregateBlock" not in code
    assert re.search(
        r"ReadOnlyAggregateData __crossgl_load_glsl_buffer_"
        r"ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(const device uchar\* buffer, uint offset\)",
        code,
    )
    assert re.search(
        r"return __crossgl_load_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(readAggregateBlock, \(48 \+ i \* 48\)\);",
        code,
    )
    assert re.search(
        r"ReadOnlyAggregateData inner = "
        r"__crossgl_load_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(readAggregateBlock, 0\);",
        code,
    )
    assert "result.items[1].flags = bool2" in code
    assert "reinterpret_cast<device" not in code
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_nested_aggregate_leaf_compound_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_nested_aggregate_leaf_compound.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_nested_aggregate_leaf_compound.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_NESTED_AGGREGATE_LEAF_COMPOUND_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "device uchar* compoundAggregateBlock [[buffer(12)]]" in code
    assert (
        "uint i = (*reinterpret_cast<const device uint*>"
        "(compoundAggregateBlock + 0));" in code
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 4))) = "
        "((*reinterpret_cast<const device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 4))) + 1.0)" in code
    )
    assert (
        "float2 __crossgl_buffer_store_0 = "
        "(float2((*reinterpret_cast<const device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 8 + 16)))" in code
    )
    assert "compoundAggregateBlock + (8 + i * 48 + 8 + 16 + 4)" in code
    assert "compoundAggregateBlock + (8 + i * 48 + 8 + 8 + 4)" in code
    assert ("un" + "supported Metal GLSL buffer block") not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_preprocessor_metal_output_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_preprocessor.metal"
    output_path = tmp_path / "mixed_glsl_preprocessor.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_PREPROCESSOR_COMPUTE_SHADER, "compute")
    )
    assert "#version" not in code
    assert "#extension" not in code
    assert "precision highp float" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_uint_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_uint_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_uint_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_UINT_ATOMICS_COMPUTE_SHADER, "compute")
    )
    assert "#version" not in code
    assert "atomic_fetch_add_explicit" in code
    assert "__crossgl_buffer_atomic_compare_exchange_uint" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_int_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_int_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_int_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_INT_ATOMICS_COMPUTE_SHADER, "compute")
    )
    assert "#version" not in code
    assert "atomic_fetch_add_explicit" in code
    assert "reinterpret_cast<device atomic_int*>" in code
    assert "__crossgl_buffer_atomic_compare_exchange_int" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_runtime_array_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_runtime_array_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_runtime_array_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_RUNTIME_ARRAY_ATOMICS_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "#version" not in code
    assert "runtimeAtomicBlock + (4 + i * 4)" in code
    assert "runtimeSignedAtomicBlock + (4 + j * 4)" in code
    assert "__crossgl_buffer_atomic_compare_exchange_uint" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_unsupported_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_unsupported_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_unsupported_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_UNSUPPORTED_ATOMICS_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "#version" not in code
    assert "unsupported Metal GLSL buffer block atomic" in code
    assert "atomic_fetch_" not in code
    assert "float floatOld = /* unsupported Metal GLSL buffer block atomic" in code
    assert "*/ 0;" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()
