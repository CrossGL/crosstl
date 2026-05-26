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
        [glslang, "-V", "--target-env", "vulkan1.2", "-S", "rgen", str(shader_path)]
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
        [glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", str(shader_path)]
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
