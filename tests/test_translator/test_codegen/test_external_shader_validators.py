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
        "+ float2(0.5))))" in code
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
