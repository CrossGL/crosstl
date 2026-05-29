import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

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


def _fragment_ast():
    return crosstl.translator.parse(FRAGMENT_SMOKE_SHADER)


def _mixed_glsl_ast(source, shader_type):
    tokens = GLSLLexer(source).tokenize()
    glsl_ast = GLSLParser(tokens, shader_type).parse()
    crossgl = GLSLToCrossGLConverter(shader_type=shader_type).generate(glsl_ast)
    return crosstl.translator.parse(crossgl)


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
    if result.returncode != 0:
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

    _run_validator([glslang, "-S", "mesh", str(shader_path)])


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

    _run_validator([glslang, "-S", "mesh", str(shader_path)])


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

    _run_validator([glslang, "-S", "mesh", str(shader_path)])


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

    _run_validator([glslang, "-S", "mesh", str(shader_path)])


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

    _run_validator([glslang, "-S", "task", str(shader_path)])


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
