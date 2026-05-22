from textwrap import dedent

import pytest

from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.parser import Parser as CrossGLParser

RESOURCE_GLSL = """
#version 450 core
layout(binding = 0, rgba8) uniform image2D outputImage;
layout(binding = 1) uniform sampler2D tex;
layout(binding = 2) uniform isampler2D itex;
layout(binding = 3) uniform usampler2D utex;

layout(std430, binding = 4) buffer DataBlock {
    vec4 values[];
} dataBlock;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 fragColor;

uniform atomic_uint counter;

void main() {
    vec4 c = imageLoad(outputImage, ivec2(0, 0));
    vec4 t = texture(tex, vUV);
    imageStore(outputImage, ivec2(0, 0), c + t);
    uint prev = atomicAdd(counter, 1);
    memoryBarrier();
    barrier();
    fragColor = t + vec4(float(prev));
}
"""


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def generate_crossgl(code: str, shader_type: str):
    ast = parse_glsl(code, shader_type)
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(ast)


def parse_crossgl(code: str):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def normalize_codegen_snapshot(code: str):
    return "\n".join(line.rstrip() for line in dedent(code).strip().splitlines())


def test_parse_resources_and_atomics():
    ast = parse_glsl(RESOURCE_GLSL, "fragment")
    assert ast is not None


def test_parse_sampler_image_variants():
    code = """
    #version 450 core
    uniform sampler1D s1d;
    uniform sampler2DRect srect;
    uniform sampler2DArrayShadow s2da;
    uniform samplerCubeArray sca;
    uniform isampler2D is2d;
    uniform usampler2D us2d;
    layout(binding = 1) uniform image1D img1d;
    layout(binding = 2) uniform image2DArray img2da;
    layout(binding = 3) uniform imageBuffer imgBuf;
    void main() { }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_codegen_1d_storage_images_roundtrip():
    code = """
    #version 450 core
    layout(binding = 0, rgba32f) uniform image1D line;
    layout(binding = 1, rgba32f) uniform image1DArray layers;
    layout(binding = 2, r32ui) uniform uimage1D counters;
    layout(binding = 3, r32ui) uniform uimage1DArray layerCounters;

    void main() {
        vec4 c = imageLoad(line, 0);
        imageStore(layers, ivec2(1, 2), c);
        uint oldValue = imageAtomicAdd(counters, 3, 1u);
        imageAtomicExchange(layerCounters, ivec2(4, 5), oldValue);
    }
    """
    output = generate_crossgl(code, "fragment")
    assert "image1D line @binding(0) @rgba32f;" in output
    assert "image1DArray layers @binding(1) @rgba32f;" in output
    assert "uimage1D counters @binding(2) @r32ui;" in output
    assert "uimage1DArray layerCounters @binding(3) @r32ui;" in output
    assert "imageLoad(line, 0)" in output
    assert "imageStore(layers, ivec2(1, 2), c)" in output
    assert "imageAtomicAdd(counters, 3, 1u)" in output
    assert "imageAtomicExchange(layerCounters, ivec2(4, 5), oldValue)" in output

    shader_ast = parse_crossgl(output)
    assert shader_ast is not None


def test_codegen_preserves_image_layout_and_access_qualifiers():
    code = """
    #version 450 core
    layout(binding = 0, r32ui) coherent readonly uniform uimage2D counters;
    layout(binding = 1, rgba32f) writeonly uniform image2D outImage;

    void main() {
        uint value = imageLoad(counters, ivec2(0, 0)).x;
        imageStore(outImage, ivec2(0, 0), vec4(value));
    }
    """
    output = generate_crossgl(code, "fragment")

    assert "uimage2D counters @binding(0) @r32ui @coherent @readonly;" in output
    assert "image2D outImage @binding(1) @rgba32f @writeonly;" in output

    shader_ast = parse_crossgl(output)
    regenerated_glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "layout(r32ui, binding = 0) coherent readonly uniform uimage2D counters;"
        in regenerated_glsl
    )
    assert (
        "layout(rgba32f, binding = 1) writeonly uniform image2D outImage;"
        in regenerated_glsl
    )


def test_parse_image_atomics_and_counters():
    code = """
    #version 450 core
    layout(binding = 0, r32ui) uniform uimage2D img;
    layout(binding = 1) uniform atomic_uint counter;
    void main() {
        uint old = imageAtomicAdd(img, ivec2(0), 1u);
        uint next = atomicCounterIncrement(counter);
        atomicCounterDecrement(counter);
        memoryBarrier();
        barrier();
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_codegen_resources_roundtrip():
    output = generate_crossgl(RESOURCE_GLSL, "fragment")
    assert "imageLoad" in output
    assert "imageStore" in output
    assert "atomicAdd" in output
    shader_ast = parse_crossgl(output)
    assert shader_ast is not None


def test_codegen_ssbo_single_array_blocks_use_structured_buffer_contract():
    code = """
    #version 450 core
    layout(std430, binding = 0) readonly buffer InputBlock {
        float inputValues[];
    } inputBlock;
    layout(std430, binding = 1) buffer OutputBlock {
        float outputValues[];
    } outputBlock;

    void main() {
        float value = inputBlock.inputValues[0];
        outputBlock.outputValues[0] = value * 2.0;
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "StructuredBuffer<float> inputBlock @binding(0);" in crossgl
    assert "RWStructuredBuffer<float> outputBlock @binding(1);" in crossgl
    assert "float value = buffer_load(inputBlock, 0);" in crossgl
    assert "buffer_store(outputBlock, 0, (value * 2.0));" in crossgl
    assert "inputBlock.inputValues" not in crossgl
    assert "outputBlock.outputValues" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "StructuredBuffer<float> inputBlock : register(t0);" in hlsl
    assert "RWStructuredBuffer<float> outputBlock : register(u1);" in hlsl
    assert "float value = inputBlock.Load(0);" in hlsl
    assert "outputBlock.Store(0, (value * 2.0));" in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "const device float* inputBlock [[buffer(0)]]" in metal
    assert "device float* outputBlock [[buffer(1)]]" in metal
    assert "float value = inputBlock[0];" in metal
    assert "outputBlock[0] = value * 2.0;" in metal

    glsl = GLSLCodeGen().generate(shader_ast)
    assert (
        "layout(std430, binding = 0) readonly buffer inputBlockBuffer { float inputBlock[]; };"
        in glsl
    )
    assert (
        "layout(std430, binding = 1) buffer outputBlockBuffer { float outputBlock[]; };"
        in glsl
    )
    assert "float value = inputBlock[0];" in glsl
    assert "outputBlock[0] = (value * 2.0);" in glsl


def test_codegen_ssbo_scalar_blocks_remain_regular_blocks():
    code = """
    #version 450 core
    layout(std430, binding = 1) buffer Counter {
        uint value;
    } counter;

    void main() {
        atomicAdd(counter.value, 1);
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "RWStructuredBuffer" not in crossgl
    assert "struct Counter" in crossgl
    assert "Counter counter;" in crossgl
    assert "atomicAdd(counter.value, 1);" in crossgl


def test_codegen_ssbo_instance_arrays_use_structured_buffer_arrays():
    code = """
    #version 450 core
    layout(std430, binding = 2) readonly buffer ReadBlock {
        int inputs[];
    } readBuffers[2];
    layout(std430, binding = 4) buffer WriteBlock {
        int outputs[];
    } writeBuffers[2];

    void main() {
        int value = readBuffers[1].inputs[0];
        writeBuffers[0].outputs[0] = value + 1;
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "StructuredBuffer<int> readBuffers[2] @binding(2);" in crossgl
    assert "RWStructuredBuffer<int> writeBuffers[2] @binding(4);" in crossgl
    assert "int value = buffer_load(readBuffers[1], 0);" in crossgl
    assert "buffer_store(writeBuffers[0], 0, (value + 1));" in crossgl
    assert "buffer_load(buffer_load" not in crossgl
    assert "readBuffers[1].inputs" not in crossgl
    assert "writeBuffers[0].outputs" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "StructuredBuffer<int> readBuffers[2] : register(t2);" in hlsl
    assert "RWStructuredBuffer<int> writeBuffers[2] : register(u4);" in hlsl
    assert "int value = readBuffers[1].Load(0);" in hlsl
    assert "writeBuffers[0].Store(0, (value + 1));" in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "array<const device int*, 2> readBuffers [[buffer(2)]]" in metal
    assert "array<device int*, 2> writeBuffers [[buffer(4)]]" in metal
    assert "int value = readBuffers[1][0];" in metal
    assert "writeBuffers[0][0] = value + 1;" in metal

    glsl = GLSLCodeGen().generate(shader_ast)
    assert (
        "layout(std430, binding = 2) readonly buffer readBuffersBuffer { int data[]; } readBuffers[2];"
        in glsl
    )
    assert (
        "layout(std430, binding = 4) buffer writeBuffersBuffer { int data[]; } writeBuffers[2];"
        in glsl
    )
    assert "int value = readBuffers[1].data[0];" in glsl
    assert "writeBuffers[0].data[0] = (value + 1);" in glsl


def test_codegen_ssbo_compound_writes_use_load_store_contract():
    code = """
    #version 450 core
    layout(std430, binding = 0) buffer ValuesBlock {
        int values[];
    } valuesBlock;
    layout(std430, binding = 1) buffer LayersBlock {
        int data[];
    } layers[2];

    void main() {
        valuesBlock.values[0] += 3;
        layers[1].data[2] *= valuesBlock.values[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "RWStructuredBuffer<int> valuesBlock @binding(0);" in crossgl
    assert "RWStructuredBuffer<int> layers[2] @binding(1);" in crossgl
    assert (
        "buffer_store(valuesBlock, 0, buffer_load(valuesBlock, 0) + 3);"
        in crossgl
    )
    assert (
        "buffer_store(layers[1], 2, buffer_load(layers[1], 2) * buffer_load(valuesBlock, 0));"
        in crossgl
    )
    assert "buffer_load(buffer_load" not in crossgl
    assert "+=" not in crossgl
    assert "*=" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "valuesBlock.Store(0, (valuesBlock.Load(0) + 3));" in hlsl
    assert "layers[1].Store(2, (layers[1].Load(2) * valuesBlock.Load(0)));" in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "valuesBlock[0] = valuesBlock[0] + 3;" in metal
    assert "layers[1][2] = layers[1][2] * valuesBlock[0];" in metal

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "valuesBlock[0] = (valuesBlock[0] + 3);" in glsl
    assert "layers[1].data[2] = (layers[1].data[2] * valuesBlock[0]);" in glsl


def test_codegen_ssbo_length_uses_buffer_dimensions_contract():
    code = """
    #version 450 core
    layout(std430, binding = 0) buffer ValuesBlock {
        int values[];
    } valuesBlock;
    layout(std430, binding = 1) buffer LayersBlock {
        int data[];
    } layers[2];

    void main() {
        uint len;
        uint layerLen;
        len = valuesBlock.values.length();
        layerLen = layers[1].data.length();
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "RWStructuredBuffer<int> valuesBlock @binding(0);" in crossgl
    assert "RWStructuredBuffer<int> layers[2] @binding(1);" in crossgl
    assert "buffer_dimensions(valuesBlock, len);" in crossgl
    assert "buffer_dimensions(layers[1], layerLen);" in crossgl
    assert ".length(" not in crossgl
    assert "buffer_load(buffer_load" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "valuesBlock.GetDimensions(len);" in hlsl
    assert "layers[1].GetDimensions(layerLen);" in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert (
        "len = 0 /* unsupported Metal buffer dimensions: device buffers do not carry length */;"
        in metal
    )
    assert (
        "layerLen = 0 /* unsupported Metal buffer dimensions: device buffers do not carry length */;"
        in metal
    )

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "len = valuesBlock.length();" in glsl
    assert "layerLen = layers[1].data.length();" in glsl


def test_codegen_unsized_ssbo_instance_arrays_preserve_dynamic_receivers():
    code = """
    #version 450 core
    layout(std430, binding = 1) buffer DynamicBlock {
        int data[];
    } buffers[];

    void main() {
        uint dynamicIndex = 1u;
        int value = buffers[dynamicIndex].data[2];
        buffers[dynamicIndex].data[3] = value + 1;
        uint len;
        len = buffers[dynamicIndex].data.length();
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "RWStructuredBuffer<int> buffers[] @binding(1);" in crossgl
    assert "int value = buffer_load(buffers[dynamicIndex], 2);" in crossgl
    assert "buffer_store(buffers[dynamicIndex], 3, (value + 1));" in crossgl
    assert "buffer_dimensions(buffers[dynamicIndex], len);" in crossgl
    assert "buffer_load(buffer_load" not in crossgl
    assert "buffers[dynamicIndex].data" not in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWStructuredBuffer<int> buffers[] : register(u1);" in hlsl
    assert "int value = buffers[dynamicIndex].Load(2);" in hlsl
    assert "buffers[dynamicIndex].Store(3, (value + 1));" in hlsl
    assert "buffers[dynamicIndex].GetDimensions(len);" in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "int value = buffers[dynamicIndex][2];" in metal
    assert "buffers[dynamicIndex][3] = value + 1;" in metal
    assert (
        "len = 0 /* unsupported Metal buffer dimensions: device buffers do not carry length */;"
        in metal
    )

    glsl = GLSLCodeGen().generate(shader_ast)
    assert (
        "layout(std430, binding = 1) buffer buffersBuffer { int data[]; } buffers[];"
        in glsl
    )
    assert "int value = buffers[dynamicIndex].data[2];" in glsl
    assert "buffers[dynamicIndex].data[3] = (value + 1);" in glsl
    assert "len = buffers[dynamicIndex].data.length();" in glsl


def test_codegen_mixed_ssbo_runtime_array_blocks_preserve_shape_with_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 0) buffer ParticlesBlock {
        uint count;
        float data[];
    } particles;

    void main() {
        uint n = particles.count;
        float v = particles.data[0];
        particles.data[0] = v + float(n);
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "unsupported GLSL SSBO block ParticlesBlock" in crossgl
    assert "preserved as attributed block struct" in crossgl
    assert "struct ParticlesBlock" in crossgl
    assert "uint count;" in crossgl
    assert "float data[];" in crossgl
    assert (
        "ParticlesBlock particles @glsl_buffer_block(std430) @binding(0);" in crossgl
    )
    assert "RWStructuredBuffer<float> particles" not in crossgl
    assert "buffer_load(particles" not in crossgl
    assert "particles.data[0]" in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "struct ParticlesBlock" not in glsl
    assert "layout(std430, binding = 0) buffer ParticlesBlock" in glsl
    assert "uint count;" in glsl
    assert "float data[];" in glsl
    assert "} particles;" in glsl
    assert "float v = particles.data[0];" in glsl
    assert "particles.data[0] = (v + float(n));" in glsl

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer particles : register(u0);" in hlsl
    assert "struct ParticlesBlock" not in hlsl
    assert "unsupported HLSL GLSL buffer block ParticlesBlock" not in hlsl
    assert "uint n = particles.Load(0);" in hlsl
    assert "float v = asfloat(particles.Load(4));" in hlsl
    assert "particles.Store(4, asuint((v + float(n))));" in hlsl

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* particles [[buffer(0)]]" in metal
    assert "unsupported Metal GLSL buffer block ParticlesBlock" not in metal
    assert "uint n = (*reinterpret_cast<const device uint*>(particles + 0));" in metal
    assert "float v = (*reinterpret_cast<const device float*>(particles + 4));" in metal
    assert (
        "(*reinterpret_cast<device float*>(particles + 4)) = v + float(n);"
        in metal
    )


def test_codegen_mixed_ssbo_metal_vec3_metadata_uses_scalar_pointer_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 41) buffer MetalVecBlock {
        vec3 bounds;
        float data[];
    } metalVecBlock;

    void main() {
        float x = metalVecBlock.bounds.x;
        vec3 b = metalVecBlock.bounds;
        metalVecBlock.bounds = b;
        float tail = metalVecBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalVecBlock [[buffer(41)]]" in metal
    assert "unsupported Metal GLSL buffer block MetalVecBlock" not in metal
    assert (
        "float x = float3((*reinterpret_cast<const device float*>"
        "(metalVecBlock + 0)), (*reinterpret_cast<const device float*>"
        "(metalVecBlock + 4)), (*reinterpret_cast<const device float*>"
        "(metalVecBlock + 8))).x;"
        in metal
    )
    assert "float3 __crossgl_buffer_store_0 = b;" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalVecBlock + 0)) = "
        "__crossgl_buffer_store_0.x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalVecBlock + 4)) = "
        "__crossgl_buffer_store_0.y;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalVecBlock + 8)) = "
        "__crossgl_buffer_store_0.z;"
        in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalVecBlock + 12));"
        in metal
    )


def test_codegen_mixed_ssbo_metal_vec3_runtime_arrays_use_16_byte_stride():
    code = """
    #version 450 core
    layout(std430, binding = 42) buffer MetalDirectionsBlock {
        float scale;
        vec3 directions[];
    } metalDirectionsBlock;

    void main() {
        uint i = 2u;
        vec3 d = metalDirectionsBlock.directions[i];
        metalDirectionsBlock.directions[i] = d;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalDirectionsBlock [[buffer(42)]]" in metal
    assert (
        "float3 d = float3((*reinterpret_cast<const device float*>"
        "(metalDirectionsBlock + (16 + i * 16))), "
        "(*reinterpret_cast<const device float*>"
        "(metalDirectionsBlock + (16 + i * 16 + 4))), "
        "(*reinterpret_cast<const device float*>"
        "(metalDirectionsBlock + (16 + i * 16 + 8))));"
        in metal
    )
    assert "float3 __crossgl_buffer_store_0 = d;" in metal
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalDirectionsBlock + (16 + i * 16))) = "
        "__crossgl_buffer_store_0.x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalDirectionsBlock + (16 + i * 16 + 4))) = "
        "__crossgl_buffer_store_0.y;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalDirectionsBlock + (16 + i * 16 + 8))) = "
        "__crossgl_buffer_store_0.z;"
        in metal
    )


def test_codegen_mixed_ssbo_metal_readonly_blocks_use_const_device_pointer():
    code = """
    #version 450 core
    layout(std430, binding = 43) readonly buffer MetalReadBlock {
        uint count;
        float values[];
    } metalReadBlock;

    void main() {
        uint i = metalReadBlock.count;
        float v = metalReadBlock.values[i];
        metalReadBlock.values[i] = v;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "const device uchar* metalReadBlock [[buffer(43)]]" in metal
    assert (
        "uint i = (*reinterpret_cast<const device uint*>(metalReadBlock + 0));"
        in metal
    )
    assert (
        "float v = (*reinterpret_cast<const device float*>"
        "(metalReadBlock + (4 + i * 4)));"
        in metal
    )
    assert "readonly device buffer cannot be written" in metal


def test_codegen_mixed_ssbo_metal_float_compound_unsupported_op_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 62) buffer MetalFloatOpsBlock {
        uint count;
        float values[];
    } metalFloatOpsBlock;

    void main() {
        uint i = metalFloatOpsBlock.count;
        metalFloatOpsBlock.values[i] %= 2.0;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalFloatOpsBlock [[buffer(62)]]" in metal
    assert "unsupported Metal GLSL buffer block compound store" in metal
    assert "operator %= is not supported for float buffer members" in metal
    assert "reinterpret_cast<device float*>" not in metal


def test_codegen_mixed_ssbo_metal_integer_mod_compound_store_is_supported():
    code = """
    #version 450 core
    layout(std430, binding = 63) buffer MetalIntOpsBlock {
        uint count;
        uint values[];
    } metalIntOpsBlock;

    void main() {
        uint i = metalIntOpsBlock.count;
        metalIntOpsBlock.values[i] %= 3u;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalIntOpsBlock [[buffer(63)]]" in metal
    assert (
        "(*reinterpret_cast<device uint*>(metalIntOpsBlock + (4 + i * 4))) = "
        "((*reinterpret_cast<const device uint*>(metalIntOpsBlock + (4 + i * 4))) % 3u);"
        in metal
    )
    assert "unsupported Metal GLSL buffer block compound store" not in metal


def test_codegen_mixed_ssbo_metal_mat3_metadata_uses_column_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 44) buffer MetalMatrixBlock {
        float scale;
        mat3 transform;
        float data[];
    } metalMatrixBlock;

    void main() {
        float s = metalMatrixBlock.scale;
        mat3 t = metalMatrixBlock.transform;
        metalMatrixBlock.transform = t;
        float tail = metalMatrixBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalMatrixBlock [[buffer(44)]]" in metal
    assert "unsupported Metal GLSL buffer block MetalMatrixBlock" not in metal
    assert "float s = (*reinterpret_cast<const device float*>(metalMatrixBlock + 0));" in metal
    assert "float3x3 t = float3x3(" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixBlock + 16))" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixBlock + 32))" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixBlock + 56))" in metal
    assert "float3x3 __crossgl_matrix_store_0 = t;" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 16)) = "
        "__crossgl_matrix_store_0[0].x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 56)) = "
        "__crossgl_matrix_store_0[2].z;"
        in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalMatrixBlock + 64));"
        in metal
    )


def test_codegen_mixed_ssbo_metal_non_square_matrix_metadata_uses_column_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 45) buffer MetalNonSquareMatrixBlock {
        float scale;
        mat2x3 transform;
        float data[];
    } metalNonSquareMatrixBlock;

    void main() {
        mat2x3 t = metalNonSquareMatrixBlock.transform;
        float tail = metalNonSquareMatrixBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalNonSquareMatrixBlock [[buffer(45)]]" in metal
    assert "float3x2 t = float3x2(" in metal
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalNonSquareMatrixBlock + 16))"
        in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalNonSquareMatrixBlock + 40))"
        in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalNonSquareMatrixBlock + 48));"
        in metal
    )


def test_codegen_mixed_ssbo_metal_runtime_matrix_arrays_use_column_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 46) buffer MetalRuntimeMatrixBlock {
        float scale;
        mat4 transforms[];
    } metalRuntimeMatrixBlock;

    void main() {
        uint i = 1u;
        mat4 selected = metalRuntimeMatrixBlock.transforms[i];
        metalRuntimeMatrixBlock.transforms[i] = selected;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalRuntimeMatrixBlock [[buffer(46)]]" in metal
    assert "float4x4 selected = float4x4(" in metal
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64)))"
        in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64 + 48 + 12)))"
        in metal
    )
    assert "float4x4 __crossgl_matrix_store_0 = selected;" in metal
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64))) = "
        "__crossgl_matrix_store_0[0].x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64 + 48 + 12))) = "
        "__crossgl_matrix_store_0[3].w;"
        in metal
    )


def test_codegen_mixed_ssbo_metal_matrix_compound_store_uses_temp():
    code = """
    #version 450 core
    layout(std430, binding = 47) buffer MetalMatrixBlock {
        float scale;
        mat4 transform;
        float data[];
    } metalMatrixBlock;

    void main() {
        mat4 value = metalMatrixBlock.transform;
        metalMatrixBlock.transform += value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalMatrixBlock [[buffer(47)]]" in metal
    assert "float4x4 __crossgl_matrix_store_0 = (float4x4(" in metal
    assert ") + value);" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 16)) = "
        "__crossgl_matrix_store_0[0].x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 76)) = "
        "__crossgl_matrix_store_0[3].w;"
        in metal
    )
    assert "unsupported Metal GLSL buffer block matrix compound store" not in metal


def test_codegen_mixed_ssbo_metal_matrix_compound_unsupported_op_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 48) buffer MetalMatrixBlock {
        float scale;
        mat4 transform;
        float data[];
    } metalMatrixBlock;

    void main() {
        mat4 value = metalMatrixBlock.transform;
        metalMatrixBlock.transform %= value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalMatrixBlock [[buffer(48)]]" in metal
    assert "unsupported Metal GLSL buffer block matrix compound store" in metal
    assert "__crossgl_matrix_store_0" not in metal
    assert "(*reinterpret_cast<device float*>(metalMatrixBlock + 16)) =" not in metal


def test_codegen_mixed_ssbo_metal_fixed_matrix_arrays_use_column_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 49) buffer MetalMatrixArrayBlock {
        mat4 transforms[2];
        float data[];
    } metalMatrixArrayBlock;

    void main() {
        uint i = 1u;
        mat4 first = metalMatrixArrayBlock.transforms[0];
        mat4 selected = metalMatrixArrayBlock.transforms[i];
        metalMatrixArrayBlock.transforms[0] = selected;
        metalMatrixArrayBlock.transforms[i] = first;
        float tail = metalMatrixArrayBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalMatrixArrayBlock [[buffer(49)]]" in metal
    assert "float4x4 first = float4x4(" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixArrayBlock + 0))" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixArrayBlock + 60))" in metal
    assert "float4x4 selected = float4x4(" in metal
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalMatrixArrayBlock + (i * 64)))"
        in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalMatrixArrayBlock + (i * 64 + 48 + 12)))"
        in metal
    )
    assert "float4x4 __crossgl_matrix_store_0 = selected;" in metal
    assert "float4x4 __crossgl_matrix_store_1 = first;" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixArrayBlock + 0)) = "
        "__crossgl_matrix_store_0[0].x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalMatrixArrayBlock + (i * 64 + 48 + 12))) = "
        "__crossgl_matrix_store_1[3].w;"
        in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalMatrixArrayBlock + 128));"
        in metal
    )


def test_codegen_mixed_ssbo_metal_fixed_matrix_array_compound_store_uses_temp():
    code = """
    #version 450 core
    layout(std430, binding = 52) buffer MetalMatrixArrayBlock {
        mat4 transforms[2];
        float data[];
    } metalMatrixArrayBlock;

    void main() {
        mat4 value = metalMatrixArrayBlock.transforms[0];
        metalMatrixArrayBlock.transforms[1] += value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalMatrixArrayBlock [[buffer(52)]]" in metal
    assert "float4x4 __crossgl_matrix_store_0 = (float4x4(" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixArrayBlock + 64))" in metal
    assert ") + value);" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixArrayBlock + 64)) = "
        "__crossgl_matrix_store_0[0].x;"
        in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixArrayBlock + 124)) = "
        "__crossgl_matrix_store_0[3].w;"
        in metal
    )
    assert "unsupported Metal GLSL buffer block matrix compound store" not in metal


def test_codegen_mixed_ssbo_metal_readonly_fixed_matrix_array_write_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 50) readonly buffer MetalReadMatrixArrayBlock {
        mat4 transforms[2];
        float data[];
    } metalReadMatrixArrayBlock;

    void main() {
        uint i = 1u;
        mat4 selected = metalReadMatrixArrayBlock.transforms[i];
        metalReadMatrixArrayBlock.transforms[i] = selected;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "const device uchar* metalReadMatrixArrayBlock [[buffer(50)]]" in metal
    assert "float4x4 selected = float4x4(" in metal
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalReadMatrixArrayBlock + (i * 64)))"
        in metal
    )
    assert "readonly device buffer cannot be written" in metal
    assert "(*reinterpret_cast<device float*>" not in metal


def test_codegen_mixed_ssbo_metal_readonly_runtime_matrix_array_write_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 51) readonly buffer MetalRuntimeMatrixBlock {
        float scale;
        mat4 transforms[];
    } metalRuntimeMatrixBlock;

    void main() {
        uint i = 1u;
        mat4 selected = metalRuntimeMatrixBlock.transforms[i];
        metalRuntimeMatrixBlock.transforms[i] = selected;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "const device uchar* metalRuntimeMatrixBlock [[buffer(51)]]" in metal
    assert "float4x4 selected = float4x4(" in metal
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64)))"
        in metal
    )
    assert "readonly device buffer cannot be written" in metal
    assert "(*reinterpret_cast<device float*>" not in metal


def test_codegen_mixed_ssbo_directx_metal_fixed_vec3_snapshot():
    code = """
    #version 450 core
    layout(std430, binding = 60) buffer SnapshotBlock {
        uint count;
        vec3 axes[2];
        float data[];
    } snapshotBlock;

    void main() {
        uint i = snapshotBlock.count;
        vec3 axis = snapshotBlock.axes[1];
        snapshotBlock.axes[0] = axis;
        float tail = snapshotBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_hlsl = """
    #version 450 core
    RWByteAddressBuffer snapshotBlock : register(u60);
    // Compute Shader
    [numthreads(1, 1, 1)]
    void CSMain() {
        uint i = snapshotBlock.Load(0);
        float3 axis = asfloat(snapshotBlock.Load3(32));
        snapshotBlock.Store3(16, asuint(axis));
        float tail = asfloat(snapshotBlock.Load(48));
    }
    """
    expected_metal = """
    #version 450 core
    #include <metal_stdlib>
    using namespace metal;

    // Compute Shader
    kernel void kernel_main(device uchar* snapshotBlock [[buffer(60)]]) {
        uint i = (*reinterpret_cast<const device uint*>(snapshotBlock + 0));
        float3 axis = float3((*reinterpret_cast<const device float*>(snapshotBlock + 32)), (*reinterpret_cast<const device float*>(snapshotBlock + 36)), (*reinterpret_cast<const device float*>(snapshotBlock + 40)));
        float3 __crossgl_buffer_store_0 = axis;
        (*reinterpret_cast<device float*>(snapshotBlock + 16)) = __crossgl_buffer_store_0.x;
        (*reinterpret_cast<device float*>(snapshotBlock + 20)) = __crossgl_buffer_store_0.y;
        (*reinterpret_cast<device float*>(snapshotBlock + 24)) = __crossgl_buffer_store_0.z;
        float tail = (*reinterpret_cast<const device float*>(snapshotBlock + 48));
    }
    """

    assert normalize_codegen_snapshot(HLSLCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_hlsl)
    )
    assert normalize_codegen_snapshot(MetalCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_metal)
    )


def test_codegen_mixed_ssbo_directx_metal_readonly_mat2_snapshot():
    code = """
    #version 450 core
    layout(std430, binding = 61) readonly buffer SnapshotMatrixBlock {
        mat2 transform;
        float data[];
    } snapshotMatrixBlock;

    void main() {
        mat2 transform = snapshotMatrixBlock.transform;
        float tail = snapshotMatrixBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_hlsl = """
    #version 450 core
    ByteAddressBuffer snapshotMatrixBlock : register(t61);
    // Compute Shader
    [numthreads(1, 1, 1)]
    void CSMain() {
        float2x2 transform = float2x2(asfloat(snapshotMatrixBlock.Load2(0)), asfloat(snapshotMatrixBlock.Load2(8)));
        float tail = asfloat(snapshotMatrixBlock.Load(16));
    }
    """
    expected_metal = """
    #version 450 core
    #include <metal_stdlib>
    using namespace metal;

    // Compute Shader
    kernel void kernel_main(const device uchar* snapshotMatrixBlock [[buffer(61)]]) {
        float2x2 transform = float2x2(float2((*reinterpret_cast<const device float*>(snapshotMatrixBlock + 0)), (*reinterpret_cast<const device float*>(snapshotMatrixBlock + 4))), float2((*reinterpret_cast<const device float*>(snapshotMatrixBlock + 8)), (*reinterpret_cast<const device float*>(snapshotMatrixBlock + 12))));
        float tail = (*reinterpret_cast<const device float*>(snapshotMatrixBlock + 16));
    }
    """

    assert normalize_codegen_snapshot(HLSLCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_hlsl)
    )
    assert normalize_codegen_snapshot(MetalCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_metal)
    )


def test_codegen_mixed_ssbo_hlsl_vector_layout_uses_byte_address_methods():
    code = """
    #version 450 core
    layout(std430, binding = 3) buffer VectorBlock {
        vec2 bias;
        vec4 values[];
    } vectorBlock;

    void main() {
        vec2 bias = vectorBlock.bias;
        vec4 value = vectorBlock.values[1];
        vectorBlock.values[2] = value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer vectorBlock : register(u3);" in hlsl
    assert "struct VectorBlock" not in hlsl
    assert "float2 bias = asfloat(vectorBlock.Load2(0));" in hlsl
    assert "float4 value = asfloat(vectorBlock.Load4(32));" in hlsl
    assert "vectorBlock.Store4(48, asuint(value));" in hlsl


def test_codegen_mixed_ssbo_hlsl_dynamic_indices_and_compound_store():
    code = """
    #version 450 core
    layout(std430, binding = 4) buffer DynamicBlock {
        uint count;
        float data[];
    } dynamicBlock;

    void main() {
        uint i = dynamicBlock.count;
        float v = dynamicBlock.data[i];
        dynamicBlock.data[i] += 1.0;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer dynamicBlock : register(u4);" in hlsl
    assert "uint i = dynamicBlock.Load(0);" in hlsl
    assert "float v = asfloat(dynamicBlock.Load((4 + i * 4)));" in hlsl
    assert (
        "dynamicBlock.Store((4 + i * 4), "
        "asuint((asfloat(dynamicBlock.Load((4 + i * 4))) + 1.0)));"
        in hlsl
    )


def test_codegen_mixed_ssbo_hlsl_float_compound_unsupported_op_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 64) buffer FloatOpsBlock {
        uint count;
        float values[];
    } floatOpsBlock;

    void main() {
        uint i = floatOpsBlock.count;
        floatOpsBlock.values[i] %= 2.0;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer floatOpsBlock : register(u64);" in hlsl
    assert "unsupported HLSL GLSL buffer block compound store" in hlsl
    assert "operator %= is not supported for float buffer members" in hlsl
    assert "floatOpsBlock.Store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_integer_mod_compound_store_is_supported():
    code = """
    #version 450 core
    layout(std430, binding = 65) buffer IntOpsBlock {
        uint count;
        uint values[];
    } intOpsBlock;

    void main() {
        uint i = intOpsBlock.count;
        intOpsBlock.values[i] %= 3u;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer intOpsBlock : register(u65);" in hlsl
    assert (
        "intOpsBlock.Store((4 + i * 4), "
        "(intOpsBlock.Load((4 + i * 4)) % 3u));"
        in hlsl
    )
    assert "unsupported HLSL GLSL buffer block compound store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_readonly_blocks_use_srv_registers():
    code = """
    #version 450 core
    layout(std430, binding = 5) readonly buffer ReadBlock {
        uint count;
        float values[];
    } readBlock;

    void main() {
        uint i = readBlock.count;
        float v = readBlock.values[i];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    assert "ReadBlock readBlock @glsl_buffer_block(std430) @binding(5) @readonly;" in crossgl

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer readBlock : register(t5);" in hlsl
    assert "RWByteAddressBuffer readBlock" not in hlsl
    assert "uint i = readBlock.Load(0);" in hlsl
    assert "float v = asfloat(readBlock.Load((4 + i * 4)));" in hlsl


def test_codegen_mixed_ssbo_hlsl_vec3_metadata_packs_following_scalar_array():
    code = """
    #version 450 core
    layout(std430, binding = 2) buffer BoundsBlock {
        vec3 bounds;
        float data[];
    } boundsBlock;

    void main() {
        float x = boundsBlock.bounds.x;
        float v = boundsBlock.data[0] + x;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer boundsBlock : register(u2);" in hlsl
    assert "float x = asfloat(boundsBlock.Load3(0)).x;" in hlsl
    assert "float v = (asfloat(boundsBlock.Load(12)) + x);" in hlsl


def test_codegen_mixed_ssbo_hlsl_vec3_runtime_arrays_use_16_byte_stride():
    code = """
    #version 450 core
    layout(std430, binding = 6) buffer DirectionBlock {
        float scale;
        vec3 directions[];
    } directionBlock;

    void main() {
        uint i = 2u;
        vec3 d = directionBlock.directions[i];
        directionBlock.directions[i] = d;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer directionBlock : register(u6);" in hlsl
    assert "float3 d = asfloat(directionBlock.Load3((16 + i * 16)));" in hlsl
    assert "directionBlock.Store3((16 + i * 16), asuint(d));" in hlsl


def test_codegen_mixed_ssbo_hlsl_ivec3_metadata_load_store_uses_integer_casts():
    code = """
    #version 450 core
    layout(std430, binding = 38) buffer IntVectorBlock {
        ivec3 normal;
        uint data[];
    } intVectorBlock;

    void main() {
        int x = intVectorBlock.normal.x;
        ivec3 n = intVectorBlock.normal;
        intVectorBlock.normal = n;
        uint tail = intVectorBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer intVectorBlock : register(u38);" in hlsl
    assert "int x = asint(intVectorBlock.Load3(0)).x;" in hlsl
    assert "int3 n = asint(intVectorBlock.Load3(0));" in hlsl
    assert "intVectorBlock.Store3(0, asuint(n));" in hlsl
    assert "uint tail = intVectorBlock.Load(12);" in hlsl


def test_codegen_mixed_ssbo_hlsl_uvec3_runtime_arrays_use_raw_uint_stores():
    code = """
    #version 450 core
    layout(std430, binding = 39) buffer UIntVectorBlock {
        uint count;
        uvec3 values[];
    } uintVectorBlock;

    void main() {
        uint i = uintVectorBlock.count;
        uvec3 v = uintVectorBlock.values[i];
        uintVectorBlock.values[i] = v;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer uintVectorBlock : register(u39);" in hlsl
    assert "uint i = uintVectorBlock.Load(0);" in hlsl
    assert "uint3 v = uintVectorBlock.Load3((16 + i * 16));" in hlsl
    assert "uintVectorBlock.Store3((16 + i * 16), v);" in hlsl
    assert "asuint(v)" not in hlsl


def test_codegen_mixed_ssbo_hlsl_ivec3_fixed_arrays_use_16_byte_stride():
    code = """
    #version 450 core
    layout(std430, binding = 40) buffer IntFixedVectorBlock {
        ivec3 axes[2];
        uint data[];
    } intFixedVectorBlock;

    void main() {
        ivec3 axis = intFixedVectorBlock.axes[1];
        intFixedVectorBlock.axes[0] = axis;
        uint tail = intFixedVectorBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer intFixedVectorBlock : register(u40);" in hlsl
    assert "int3 axis = asint(intFixedVectorBlock.Load3(16));" in hlsl
    assert "intFixedVectorBlock.Store3(0, asuint(axis));" in hlsl
    assert "uint tail = intFixedVectorBlock.Load(32);" in hlsl


def test_codegen_mixed_ssbo_hlsl_fixed_scalar_arrays_before_runtime_array():
    code = """
    #version 450 core
    layout(std430, binding = 8) buffer FixedBlock {
        float weights[3];
        uint count;
        float data[];
    } fixedBlock;

    void main() {
        float w = fixedBlock.weights[2];
        uint n = fixedBlock.count;
        float v = fixedBlock.data[1];
        fixedBlock.weights[0] = v;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer fixedBlock : register(u8);" in hlsl
    assert "float w = asfloat(fixedBlock.Load(8));" in hlsl
    assert "uint n = fixedBlock.Load(12);" in hlsl
    assert "float v = asfloat(fixedBlock.Load(20));" in hlsl
    assert "fixedBlock.Store(0, asuint(v));" in hlsl


def test_codegen_mixed_ssbo_hlsl_fixed_vec3_arrays_use_16_byte_stride():
    code = """
    #version 450 core
    layout(std430, binding = 9) buffer FixedVec3Block {
        vec3 axes[2];
        float values[];
    } fixedVec3Block;

    void main() {
        vec3 axis = fixedVec3Block.axes[1];
        float value = fixedVec3Block.values[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer fixedVec3Block : register(u9);" in hlsl
    assert "float3 axis = asfloat(fixedVec3Block.Load3(16));" in hlsl
    assert "float value = asfloat(fixedVec3Block.Load(32));" in hlsl


def test_codegen_mixed_ssbo_hlsl_dynamic_indices_into_fixed_arrays():
    code = """
    #version 450 core
    layout(std430, binding = 10) buffer FixedDynamicBlock {
        uint index;
        vec2 offsets[4];
        float data[];
    } fixedDynamicBlock;

    void main() {
        uint i = fixedDynamicBlock.index;
        vec2 o = fixedDynamicBlock.offsets[i];
        fixedDynamicBlock.offsets[i] += vec2(1.0);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer fixedDynamicBlock : register(u10);" in hlsl
    assert "uint i = fixedDynamicBlock.Load(0);" in hlsl
    assert "float2 o = asfloat(fixedDynamicBlock.Load2((8 + i * 8)));" in hlsl
    assert (
        "fixedDynamicBlock.Store2((8 + i * 8), "
        "asuint((asfloat(fixedDynamicBlock.Load2((8 + i * 8))) + float2(1.0))));"
        in hlsl
    )


def test_codegen_mixed_ssbo_hlsl_readonly_fixed_arrays_use_srv_registers():
    code = """
    #version 450 core
    layout(std430, binding = 11) readonly buffer ReadFixedBlock {
        uint index;
        vec4 values[2];
        float tail[];
    } readFixedBlock;

    void main() {
        uint i = readFixedBlock.index;
        vec4 value = readFixedBlock.values[i];
        float tail = readFixedBlock.tail[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer readFixedBlock : register(t11);" in hlsl
    assert "RWByteAddressBuffer readFixedBlock" not in hlsl
    assert "uint i = readFixedBlock.Load(0);" in hlsl
    assert "float4 value = asfloat(readFixedBlock.Load4((16 + i * 16)));" in hlsl
    assert "float tail = asfloat(readFixedBlock.Load(48));" in hlsl


def test_codegen_mixed_ssbo_hlsl_mat2_metadata_before_runtime_array():
    code = """
    #version 450 core
    layout(std430, binding = 12) buffer Matrix2Block {
        mat2 transform;
        float data[];
    } matrix2Block;

    void main() {
        mat2 t = matrix2Block.transform;
        float v = matrix2Block.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrix2Block : register(u12);" in hlsl
    assert (
        "float2x2 t = float2x2(asfloat(matrix2Block.Load2(0)), "
        "asfloat(matrix2Block.Load2(8)));"
        in hlsl
    )
    assert "float v = asfloat(matrix2Block.Load(16));" in hlsl


def test_codegen_mixed_ssbo_hlsl_mat2_metadata_store_uses_column_stores():
    code = """
    #version 450 core
    layout(std430, binding = 12) buffer Matrix2Block {
        mat2 transform;
        float data[];
    } matrix2Block;

    void main() {
        mat2 t = matrix2Block.transform;
        matrix2Block.transform = t;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrix2Block : register(u12);" in hlsl
    assert "matrix2Block.Store2(0, asuint(t[0]));" in hlsl
    assert "matrix2Block.Store2(8, asuint(t[1]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_mat3_metadata_aligns_after_scalar():
    code = """
    #version 450 core
    layout(std430, binding = 14) buffer Matrix3Block {
        float scale;
        mat3 transform;
        float data[];
    } matrix3Block;

    void main() {
        float s = matrix3Block.scale;
        mat3 t = matrix3Block.transform;
        float v = matrix3Block.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrix3Block : register(u14);" in hlsl
    assert "float s = asfloat(matrix3Block.Load(0));" in hlsl
    assert (
        "float3x3 t = float3x3(asfloat(matrix3Block.Load3(16)), "
        "asfloat(matrix3Block.Load3(32)), asfloat(matrix3Block.Load3(48)));"
        in hlsl
    )
    assert "float v = asfloat(matrix3Block.Load(64));" in hlsl


def test_codegen_mixed_ssbo_hlsl_mat3_metadata_store_uses_column_stores():
    code = """
    #version 450 core
    layout(std430, binding = 14) buffer Matrix3Block {
        float scale;
        mat3 transform;
        float data[];
    } matrix3Block;

    void main() {
        mat3 t = matrix3Block.transform;
        matrix3Block.transform = t;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrix3Block : register(u14);" in hlsl
    assert "matrix3Block.Store3(16, asuint(t[0]));" in hlsl
    assert "matrix3Block.Store3(32, asuint(t[1]));" in hlsl
    assert "matrix3Block.Store3(48, asuint(t[2]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix store" not in hlsl


@pytest.mark.parametrize(
    (
        "glsl_type",
        "binding",
        "hlsl_type",
        "matrix_offset",
        "column_offsets",
        "components",
        "data_offset",
    ),
    [
        ("mat2x3", 15, "float3x2", 16, [16, 32], 3, 48),
        ("mat2x4", 16, "float4x2", 16, [16, 32], 4, 48),
        ("mat3x2", 17, "float2x3", 8, [8, 16, 24], 2, 32),
        ("mat3x4", 18, "float4x3", 16, [16, 32, 48], 4, 64),
        ("mat4x2", 19, "float2x4", 8, [8, 16, 24, 32], 2, 40),
        ("mat4x3", 20, "float3x4", 16, [16, 32, 48, 64], 3, 80),
    ],
)
def test_codegen_mixed_ssbo_hlsl_non_square_matrix_metadata_layout(
    glsl_type,
    binding,
    hlsl_type,
    matrix_offset,
    column_offsets,
    components,
    data_offset,
):
    code = f"""
    #version 450 core
    layout(std430, binding = {binding}) buffer NonSquareBlock {{
        float scale;
        {glsl_type} transform;
        float data[];
    }} nonSquareBlock;

    void main() {{
        float s = nonSquareBlock.scale;
        {glsl_type} t = nonSquareBlock.transform;
        float v = nonSquareBlock.data[0];
        nonSquareBlock.transform = t;
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    load_method = f"Load{components}"
    store_method = f"Store{components}"
    column_loads = ", ".join(
        f"asfloat(nonSquareBlock.{load_method}({offset}))"
        for offset in column_offsets
    )
    assert f"RWByteAddressBuffer nonSquareBlock : register(u{binding});" in hlsl
    assert "float s = asfloat(nonSquareBlock.Load(0));" in hlsl
    assert f"{hlsl_type} t = {hlsl_type}({column_loads});" in hlsl
    assert f"float v = asfloat(nonSquareBlock.Load({data_offset}));" in hlsl
    assert column_offsets[0] == matrix_offset
    for column, offset in enumerate(column_offsets):
        assert f"nonSquareBlock.{store_method}({offset}, asuint(t[{column}]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_mat4_metadata_aligns_after_scalar():
    code = """
    #version 450 core
    layout(std430, binding = 13) buffer Matrix4Block {
        float scale;
        mat4 transform;
        float data[];
    } matrix4Block;

    void main() {
        float s = matrix4Block.scale;
        mat4 t = matrix4Block.transform;
        float v = matrix4Block.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrix4Block : register(u13);" in hlsl
    assert "float s = asfloat(matrix4Block.Load(0));" in hlsl
    assert (
        "float4x4 t = float4x4(asfloat(matrix4Block.Load4(16)), "
        "asfloat(matrix4Block.Load4(32)), asfloat(matrix4Block.Load4(48)), "
        "asfloat(matrix4Block.Load4(64)));"
        in hlsl
    )
    assert "float v = asfloat(matrix4Block.Load(80));" in hlsl


def test_codegen_mixed_ssbo_hlsl_mat4_metadata_store_uses_column_stores():
    code = """
    #version 450 core
    layout(std430, binding = 13) buffer Matrix4Block {
        float scale;
        mat4 transform;
        float data[];
    } matrix4Block;

    void main() {
        mat4 t = matrix4Block.transform;
        matrix4Block.transform = t;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrix4Block : register(u13);" in hlsl
    assert "matrix4Block.Store4(16, asuint(t[0]));" in hlsl
    assert "matrix4Block.Store4(32, asuint(t[1]));" in hlsl
    assert "matrix4Block.Store4(48, asuint(t[2]));" in hlsl
    assert "matrix4Block.Store4(64, asuint(t[3]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix store" not in hlsl


@pytest.mark.parametrize(
    (
        "glsl_type",
        "binding",
        "hlsl_type",
        "components",
        "matrix_stride",
        "column_offsets",
        "data_offset",
    ),
    [
        ("mat4", 21, "float4x4", 4, 64, [0, 16, 32, 48], 128),
        ("mat3x2", 22, "float2x3", 2, 24, [0, 8, 16], 48),
    ],
)
def test_codegen_mixed_ssbo_hlsl_readonly_matrix_arrays_use_column_loads(
    glsl_type,
    binding,
    hlsl_type,
    components,
    matrix_stride,
    column_offsets,
    data_offset,
):
    code = f"""
    #version 450 core
    layout(std430, binding = {binding}) readonly buffer ReadMatrixArrayBlock {{
        {glsl_type} transforms[2];
        float data[];
    }} readMatrixArrayBlock;

    void main() {{
        uint i = 1u;
        {glsl_type} first = readMatrixArrayBlock.transforms[0];
        {glsl_type} selected = readMatrixArrayBlock.transforms[i];
        float tail = readMatrixArrayBlock.data[0];
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    load_method = f"Load{components}"
    first_loads = ", ".join(
        f"asfloat(readMatrixArrayBlock.{load_method}({offset}))"
        for offset in column_offsets
    )
    dynamic_loads = ", ".join(
        f"asfloat(readMatrixArrayBlock.{load_method}("
        f"{'(i * ' + str(matrix_stride) + ')' if offset == 0 else '(i * ' + str(matrix_stride) + ' + ' + str(offset) + ')'}"
        f"))"
        for offset in column_offsets
    )
    assert f"ByteAddressBuffer readMatrixArrayBlock : register(t{binding});" in hlsl
    assert "RWByteAddressBuffer readMatrixArrayBlock" not in hlsl
    assert f"{hlsl_type} first = {hlsl_type}({first_loads});" in hlsl
    assert f"{hlsl_type} selected = {hlsl_type}({dynamic_loads});" in hlsl
    assert f"float tail = asfloat(readMatrixArrayBlock.Load({data_offset}));" in hlsl


def test_codegen_mixed_ssbo_hlsl_readonly_matrix_metadata_write_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 35) readonly buffer ReadMatrixBlock {
        mat4 transform;
        float data[];
    } readMatrixBlock;

    void main() {
        mat4 value = readMatrixBlock.transform;
        readMatrixBlock.transform = value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer readMatrixBlock : register(t35);" in hlsl
    assert "RWByteAddressBuffer readMatrixBlock" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "readMatrixBlock.Store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_readonly_fixed_matrix_array_write_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 36) readonly buffer ReadMatrixArrayBlock {
        mat4 transforms[2];
        float data[];
    } readMatrixArrayBlock;

    void main() {
        uint i = 1u;
        mat4 selected = readMatrixArrayBlock.transforms[i];
        readMatrixArrayBlock.transforms[i] = selected;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer readMatrixArrayBlock : register(t36);" in hlsl
    assert "RWByteAddressBuffer readMatrixArrayBlock" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "readMatrixArrayBlock.Store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_readonly_runtime_matrix_array_write_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 37) readonly buffer RuntimeMatrixBlock {
        float scale;
        mat4 transforms[];
    } runtimeMatrixBlock;

    void main() {
        uint i = 1u;
        mat4 selected = runtimeMatrixBlock.transforms[i];
        runtimeMatrixBlock.transforms[i] = selected;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer runtimeMatrixBlock : register(t37);" in hlsl
    assert "RWByteAddressBuffer runtimeMatrixBlock" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "runtimeMatrixBlock.Store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_writable_matrix_arrays_use_column_stores():
    code = """
    #version 450 core
    layout(std430, binding = 23) buffer MatrixArrayBlock {
        mat4 transforms[2];
        float data[];
    } matrixArrayBlock;

    void main() {
        uint i = 1u;
        mat4 first = matrixArrayBlock.transforms[0];
        mat4 selected = matrixArrayBlock.transforms[i];
        matrixArrayBlock.transforms[0] = selected;
        matrixArrayBlock.transforms[i] = first;
        float tail = matrixArrayBlock.data[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrixArrayBlock : register(u23);" in hlsl
    assert (
        "float4x4 first = float4x4(asfloat(matrixArrayBlock.Load4(0)), "
        "asfloat(matrixArrayBlock.Load4(16)), asfloat(matrixArrayBlock.Load4(32)), "
        "asfloat(matrixArrayBlock.Load4(48)));"
        in hlsl
    )
    assert (
        "float4x4 selected = float4x4("
        "asfloat(matrixArrayBlock.Load4((i * 64))), "
        "asfloat(matrixArrayBlock.Load4((i * 64 + 16))), "
        "asfloat(matrixArrayBlock.Load4((i * 64 + 32))), "
        "asfloat(matrixArrayBlock.Load4((i * 64 + 48))));"
        in hlsl
    )
    assert "matrixArrayBlock.Store4(0, asuint(selected[0]));" in hlsl
    assert "matrixArrayBlock.Store4(16, asuint(selected[1]));" in hlsl
    assert "matrixArrayBlock.Store4(32, asuint(selected[2]));" in hlsl
    assert "matrixArrayBlock.Store4(48, asuint(selected[3]));" in hlsl
    assert "matrixArrayBlock.Store4((i * 64), asuint(first[0]));" in hlsl
    assert "matrixArrayBlock.Store4((i * 64 + 16), asuint(first[1]));" in hlsl
    assert "matrixArrayBlock.Store4((i * 64 + 32), asuint(first[2]));" in hlsl
    assert "matrixArrayBlock.Store4((i * 64 + 48), asuint(first[3]));" in hlsl
    assert "float tail = asfloat(matrixArrayBlock.Load(128));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix store" not in hlsl


@pytest.mark.parametrize(
    (
        "glsl_type",
        "binding",
        "hlsl_type",
        "components",
        "matrix_stride",
        "column_offsets",
        "data_offset",
    ),
    [
        ("mat3x2", 25, "float2x3", 2, 24, [0, 8, 16], 48),
        ("mat4x3", 26, "float3x4", 3, 64, [0, 16, 32, 48], 128),
    ],
)
def test_codegen_mixed_ssbo_hlsl_writable_non_square_matrix_arrays_use_column_stores(
    glsl_type,
    binding,
    hlsl_type,
    components,
    matrix_stride,
    column_offsets,
    data_offset,
):
    code = f"""
    #version 450 core
    layout(std430, binding = {binding}) buffer NonSquareMatrixArrayBlock {{
        {glsl_type} transforms[2];
        float data[];
    }} nonSquareMatrixArrayBlock;

    void main() {{
        uint i = 1u;
        {glsl_type} first = nonSquareMatrixArrayBlock.transforms[0];
        {glsl_type} selected = nonSquareMatrixArrayBlock.transforms[i];
        nonSquareMatrixArrayBlock.transforms[0] = selected;
        nonSquareMatrixArrayBlock.transforms[i] = first;
        float tail = nonSquareMatrixArrayBlock.data[0];
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    load_method = f"Load{components}"
    store_method = f"Store{components}"
    first_loads = ", ".join(
        f"asfloat(nonSquareMatrixArrayBlock.{load_method}({offset}))"
        for offset in column_offsets
    )
    dynamic_offsets = [
        f"(i * {matrix_stride})"
        if offset == 0
        else f"(i * {matrix_stride} + {offset})"
        for offset in column_offsets
    ]
    dynamic_loads = ", ".join(
        f"asfloat(nonSquareMatrixArrayBlock.{load_method}({offset}))"
        for offset in dynamic_offsets
    )

    assert f"RWByteAddressBuffer nonSquareMatrixArrayBlock : register(u{binding});" in hlsl
    assert f"{hlsl_type} first = {hlsl_type}({first_loads});" in hlsl
    assert f"{hlsl_type} selected = {hlsl_type}({dynamic_loads});" in hlsl
    for column, offset in enumerate(column_offsets):
        assert (
            f"nonSquareMatrixArrayBlock.{store_method}({offset}, "
            f"asuint(selected[{column}]));"
            in hlsl
        )
    for column, offset in enumerate(dynamic_offsets):
        assert (
            f"nonSquareMatrixArrayBlock.{store_method}({offset}, "
            f"asuint(first[{column}]));"
            in hlsl
        )
    assert f"float tail = asfloat(nonSquareMatrixArrayBlock.Load({data_offset}));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_matrix_metadata_compound_store_uses_temp():
    code = """
    #version 450 core
    layout(std430, binding = 32) buffer MatrixBlock {
        float scale;
        mat4 transform;
        float data[];
    } matrixBlock;

    void main() {
        mat4 value = matrixBlock.transform;
        matrixBlock.transform += value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrixBlock : register(u32);" in hlsl
    assert (
        "float4x4 __crossgl_matrix_store_0 = "
        "(float4x4(asfloat(matrixBlock.Load4(16)), "
        "asfloat(matrixBlock.Load4(32)), asfloat(matrixBlock.Load4(48)), "
        "asfloat(matrixBlock.Load4(64))) + value);"
        in hlsl
    )
    assert "matrixBlock.Store4(16, asuint(__crossgl_matrix_store_0[0]));" in hlsl
    assert "matrixBlock.Store4(32, asuint(__crossgl_matrix_store_0[1]));" in hlsl
    assert "matrixBlock.Store4(48, asuint(__crossgl_matrix_store_0[2]));" in hlsl
    assert "matrixBlock.Store4(64, asuint(__crossgl_matrix_store_0[3]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix compound store" not in hlsl


@pytest.mark.parametrize(
    ("operator", "rhs_decl", "rhs_expr", "binary_op"),
    [
        ("-=", "mat4 value = matrixBlock.transform;", "value", "-"),
        ("*=", "", "2.0", "*"),
        ("/=", "", "2.0", "/"),
    ],
)
def test_codegen_mixed_ssbo_hlsl_matrix_compound_supported_ops_use_temp(
    operator, rhs_decl, rhs_expr, binary_op
):
    code = f"""
    #version 450 core
    layout(std430, binding = 33) buffer MatrixBlock {{
        float scale;
        mat4 transform;
        float data[];
    }} matrixBlock;

    void main() {{
        {rhs_decl}
        matrixBlock.transform {operator} {rhs_expr};
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrixBlock : register(u33);" in hlsl
    assert (
        "float4x4 __crossgl_matrix_store_0 = "
        f"(float4x4(asfloat(matrixBlock.Load4(16)), "
        f"asfloat(matrixBlock.Load4(32)), asfloat(matrixBlock.Load4(48)), "
        f"asfloat(matrixBlock.Load4(64))) {binary_op} {rhs_expr});"
        in hlsl
    )
    assert "matrixBlock.Store4(16, asuint(__crossgl_matrix_store_0[0]));" in hlsl
    assert "matrixBlock.Store4(32, asuint(__crossgl_matrix_store_0[1]));" in hlsl
    assert "matrixBlock.Store4(48, asuint(__crossgl_matrix_store_0[2]));" in hlsl
    assert "matrixBlock.Store4(64, asuint(__crossgl_matrix_store_0[3]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix compound store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_matrix_compound_unsupported_op_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 34) buffer MatrixBlock {
        float scale;
        mat4 transform;
        float data[];
    } matrixBlock;

    void main() {
        mat4 value = matrixBlock.transform;
        matrixBlock.transform %= value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrixBlock : register(u34);" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix compound store" in hlsl
    assert "requires explicit matrix operation lowering" in hlsl
    assert "matrixBlock.Store4(16" not in hlsl


def test_codegen_mixed_ssbo_hlsl_matrix_array_compound_store_uses_temp():
    code = """
    #version 450 core
    layout(std430, binding = 24) buffer MatrixArrayBlock {
        mat4 transforms[2];
        float data[];
    } matrixArrayBlock;

    void main() {
        mat4 value = matrixArrayBlock.transforms[0];
        matrixArrayBlock.transforms[1] += value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer matrixArrayBlock : register(u24);" in hlsl
    assert (
        "float4x4 __crossgl_matrix_store_0 = "
        "(float4x4(asfloat(matrixArrayBlock.Load4(64)), "
        "asfloat(matrixArrayBlock.Load4(80)), "
        "asfloat(matrixArrayBlock.Load4(96)), "
        "asfloat(matrixArrayBlock.Load4(112))) + value);"
        in hlsl
    )
    assert "matrixArrayBlock.Store4(64, asuint(__crossgl_matrix_store_0[0]));" in hlsl
    assert "matrixArrayBlock.Store4(80, asuint(__crossgl_matrix_store_0[1]));" in hlsl
    assert "matrixArrayBlock.Store4(96, asuint(__crossgl_matrix_store_0[2]));" in hlsl
    assert "matrixArrayBlock.Store4(112, asuint(__crossgl_matrix_store_0[3]));" in hlsl
    assert "unsupported HLSL GLSL buffer block matrix compound store" not in hlsl


@pytest.mark.parametrize(
    (
        "glsl_type",
        "binding",
        "hlsl_type",
        "components",
        "runtime_offset",
        "matrix_stride",
        "column_offsets",
    ),
    [
        ("mat4", 27, "float4x4", 4, 16, 64, [16, 32, 48, 64]),
        ("mat3x2", 28, "float2x3", 2, 8, 24, [8, 16, 24]),
    ],
)
def test_codegen_mixed_ssbo_hlsl_runtime_matrix_arrays_use_column_loads(
    glsl_type,
    binding,
    hlsl_type,
    components,
    runtime_offset,
    matrix_stride,
    column_offsets,
):
    code = f"""
    #version 450 core
    layout(std430, binding = {binding}) readonly buffer RuntimeMatrixBlock {{
        float scale;
        {glsl_type} transforms[];
    }} runtimeMatrixBlock;

    void main() {{
        uint i = 1u;
        float s = runtimeMatrixBlock.scale;
        {glsl_type} first = runtimeMatrixBlock.transforms[0];
        {glsl_type} selected = runtimeMatrixBlock.transforms[i];
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    load_method = f"Load{components}"
    first_loads = ", ".join(
        f"asfloat(runtimeMatrixBlock.{load_method}({offset}))"
        for offset in column_offsets
    )
    dynamic_offsets = [
        f"({runtime_offset} + i * {matrix_stride})"
        if offset == runtime_offset
        else f"({runtime_offset} + i * {matrix_stride} + {offset - runtime_offset})"
        for offset in column_offsets
    ]
    dynamic_loads = ", ".join(
        f"asfloat(runtimeMatrixBlock.{load_method}({offset}))"
        for offset in dynamic_offsets
    )

    assert f"ByteAddressBuffer runtimeMatrixBlock : register(t{binding});" in hlsl
    assert "RWByteAddressBuffer runtimeMatrixBlock" not in hlsl
    assert "float s = asfloat(runtimeMatrixBlock.Load(0));" in hlsl
    assert f"{hlsl_type} first = {hlsl_type}({first_loads});" in hlsl
    assert f"{hlsl_type} selected = {hlsl_type}({dynamic_loads});" in hlsl


@pytest.mark.parametrize(
    (
        "glsl_type",
        "binding",
        "hlsl_type",
        "components",
        "runtime_offset",
        "matrix_stride",
        "column_offsets",
    ),
    [
        ("mat4", 29, "float4x4", 4, 16, 64, [16, 32, 48, 64]),
        ("mat3x2", 30, "float2x3", 2, 8, 24, [8, 16, 24]),
    ],
)
def test_codegen_mixed_ssbo_hlsl_runtime_matrix_array_stores_use_column_stores(
    glsl_type,
    binding,
    hlsl_type,
    components,
    runtime_offset,
    matrix_stride,
    column_offsets,
):
    code = f"""
    #version 450 core
    layout(std430, binding = {binding}) buffer RuntimeMatrixBlock {{
        float scale;
        {glsl_type} transforms[];
    }} runtimeMatrixBlock;

    void main() {{
        uint i = 1u;
        {glsl_type} first = runtimeMatrixBlock.transforms[0];
        {glsl_type} selected = runtimeMatrixBlock.transforms[i];
        runtimeMatrixBlock.transforms[i] = selected;
        runtimeMatrixBlock.transforms[0] = first;
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    load_method = f"Load{components}"
    store_method = f"Store{components}"
    first_loads = ", ".join(
        f"asfloat(runtimeMatrixBlock.{load_method}({offset}))"
        for offset in column_offsets
    )
    dynamic_offsets = [
        f"({runtime_offset} + i * {matrix_stride})"
        if offset == runtime_offset
        else f"({runtime_offset} + i * {matrix_stride} + {offset - runtime_offset})"
        for offset in column_offsets
    ]
    dynamic_loads = ", ".join(
        f"asfloat(runtimeMatrixBlock.{load_method}({offset}))"
        for offset in dynamic_offsets
    )

    assert f"RWByteAddressBuffer runtimeMatrixBlock : register(u{binding});" in hlsl
    assert f"{hlsl_type} first = {hlsl_type}({first_loads});" in hlsl
    assert f"{hlsl_type} selected = {hlsl_type}({dynamic_loads});" in hlsl
    for column, offset in enumerate(dynamic_offsets):
        assert (
            f"runtimeMatrixBlock.{store_method}({offset}, "
            f"asuint(selected[{column}]));"
            in hlsl
        )
    for column, offset in enumerate(column_offsets):
        assert (
            f"runtimeMatrixBlock.{store_method}({offset}, "
            f"asuint(first[{column}]));"
            in hlsl
        )
    assert "unsupported HLSL GLSL buffer block runtime matrix array store" not in hlsl


@pytest.mark.parametrize(
    (
        "glsl_type",
        "binding",
        "hlsl_type",
        "components",
        "runtime_offset",
        "matrix_stride",
        "column_offsets",
    ),
    [
        ("mat4", 31, "float4x4", 4, 16, 64, [16, 32, 48, 64]),
        ("mat3x2", 32, "float2x3", 2, 8, 24, [8, 16, 24]),
    ],
)
def test_codegen_mixed_ssbo_hlsl_runtime_matrix_array_compound_store_uses_temp(
    glsl_type,
    binding,
    hlsl_type,
    components,
    runtime_offset,
    matrix_stride,
    column_offsets,
):
    code = f"""
    #version 450 core
    layout(std430, binding = {binding}) buffer RuntimeMatrixBlock {{
        float scale;
        {glsl_type} transforms[];
    }} runtimeMatrixBlock;

    void main() {{
        uint i = 1u;
        {glsl_type} selected = runtimeMatrixBlock.transforms[i];
        runtimeMatrixBlock.transforms[i] += selected;
    }}
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    load_method = f"Load{components}"
    store_method = f"Store{components}"
    dynamic_offsets = [
        f"({runtime_offset} + i * {matrix_stride})"
        if offset == runtime_offset
        else f"({runtime_offset} + i * {matrix_stride} + {offset - runtime_offset})"
        for offset in column_offsets
    ]
    dynamic_loads = ", ".join(
        f"asfloat(runtimeMatrixBlock.{load_method}({offset}))"
        for offset in dynamic_offsets
    )

    assert f"RWByteAddressBuffer runtimeMatrixBlock : register(u{binding});" in hlsl
    assert (
        f"{hlsl_type} __crossgl_matrix_store_0 = "
        f"({hlsl_type}({dynamic_loads}) + selected);"
        in hlsl
    )
    for column, offset in enumerate(dynamic_offsets):
        assert (
            f"runtimeMatrixBlock.{store_method}({offset}, "
            f"asuint(__crossgl_matrix_store_0[{column}]));"
            in hlsl
        )
    assert "unsupported HLSL GLSL buffer block matrix compound store" not in hlsl


def test_codegen_mixed_ssbo_unsupported_hlsl_layout_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 7) buffer MatrixBlock {
        mat4 transforms[];
        float tail;
    } matrixBlock;

    void main() {
        float v = matrixBlock.tail;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "unsupported HLSL GLSL buffer block MatrixBlock" in hlsl
    assert "ByteAddressBuffer offset lowering" in hlsl
    assert (
        "unsupported member transforms: runtime arrays must be the final "
        "buffer block member"
        in hlsl
    )
    assert "RWByteAddressBuffer matrixBlock" not in hlsl


if __name__ == "__main__":
    pytest.main()
