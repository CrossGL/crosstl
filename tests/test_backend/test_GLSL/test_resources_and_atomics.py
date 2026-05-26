import re
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


def test_codegen_ssbo_scalar_blocks_preserve_block_attributes():
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
    assert "Counter counter @glsl_buffer_block(std430) @binding(1);" in crossgl
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
    assert "buffer_store(valuesBlock, 0, buffer_load(valuesBlock, 0) + 3);" in crossgl
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


def test_codegen_mixed_ssbo_runtime_array_blocks_preserve_shape_with_attribute():
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

    assert "unsupported GLSL SSBO block ParticlesBlock" not in crossgl
    assert "struct ParticlesBlock" in crossgl
    assert "uint count;" in crossgl
    assert "float data[];" in crossgl
    assert "ParticlesBlock particles @glsl_buffer_block(std430) @binding(0);" in crossgl
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
    assert "(*reinterpret_cast<device float*>(particles + 4)) = v + float(n);" in metal


def test_codegen_mixed_ssbo_fixed_member_arrays_accept_global_const_size():
    crossgl = """
    shader ConstSizedBlock {
        const int WIDTH = 3;

        struct Block {
            float weights[WIDTH];
            float data[];
        };

        Block block @glsl_buffer_block(std430) @binding(7);

        compute {
            void main(uint i) {
                float x = block.weights[2];
                float y = block.data[i];
                block.weights[1] = x + y;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "static const int WIDTH = 3;" in hlsl
    assert "RWByteAddressBuffer block : register(u7);" in hlsl
    assert "float x = asfloat(block.Load(8));" in hlsl
    assert "float y = asfloat(block.Load((12 + i * 4)));" in hlsl
    assert "block.Store(4, asuint((x + y)));" in hlsl
    assert "unsupported HLSL GLSL buffer block" not in hlsl

    assert "constant int WIDTH = 3;" in metal
    assert "device uchar* block [[buffer(7)]]" in metal
    assert "float x = (*reinterpret_cast<const device float*>(block + 8));" in metal
    assert (
        "float y = (*reinterpret_cast<const device float*>" "(block + (12 + i * 4)));"
    ) in metal
    assert "(*reinterpret_cast<device float*>(block + 4)) = x + y;" in metal
    assert "unsupported Metal GLSL buffer block" not in metal

    assert "layout(std430, binding = 7) buffer Block" in glsl
    assert "float weights[WIDTH];" in glsl
    assert "float data[];" in glsl
    assert "block.weights[1] = (x + y);" in glsl


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
        "(metalVecBlock + 8))).x;" in metal
    )
    assert "float3 __crossgl_buffer_store_0 = b;" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalVecBlock + 0)) = "
        "__crossgl_buffer_store_0.x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalVecBlock + 4)) = "
        "__crossgl_buffer_store_0.y;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalVecBlock + 8)) = "
        "__crossgl_buffer_store_0.z;" in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalVecBlock + 12));" in metal
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
        "(metalDirectionsBlock + (16 + i * 16 + 8))));" in metal
    )
    assert "float3 __crossgl_buffer_store_0 = d;" in metal
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalDirectionsBlock + (16 + i * 16))) = "
        "__crossgl_buffer_store_0.x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalDirectionsBlock + (16 + i * 16 + 4))) = "
        "__crossgl_buffer_store_0.y;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalDirectionsBlock + (16 + i * 16 + 8))) = "
        "__crossgl_buffer_store_0.z;" in metal
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
        "uint i = (*reinterpret_cast<const device uint*>(metalReadBlock + 0));" in metal
    )
    assert (
        "float v = (*reinterpret_cast<const device float*>"
        "(metalReadBlock + (4 + i * 4)));" in metal
    )
    assert "readonly device buffer cannot be written" in metal


def test_codegen_mixed_ssbo_metal_fixed_vector_array_compound_store_uses_temp():
    code = """
    #version 450 core
    layout(std430, binding = 66) buffer MetalFixedDynamicBlock {
        uint index;
        vec2 offsets[4];
        float data[];
    } metalFixedDynamicBlock;

    void main() {
        uint i = metalFixedDynamicBlock.index;
        metalFixedDynamicBlock.offsets[i] += vec2(1.0);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalFixedDynamicBlock [[buffer(66)]]" in metal
    assert (
        "uint i = (*reinterpret_cast<const device uint*>(metalFixedDynamicBlock + 0));"
        in metal
    )
    assert (
        "float2 __crossgl_buffer_store_0 = (float2("
        "(*reinterpret_cast<const device float*>(metalFixedDynamicBlock + (8 + i * 8))), "
        "(*reinterpret_cast<const device float*>(metalFixedDynamicBlock + (8 + i * 8 + 4)))) "
        "+ float2(1.0));" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalFixedDynamicBlock + (8 + i * 8))) = "
        "__crossgl_buffer_store_0.x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalFixedDynamicBlock + (8 + i * 8 + 4))) = "
        "__crossgl_buffer_store_0.y;" in metal
    )
    assert "unsupported Metal GLSL buffer block compound store" not in metal


def test_codegen_mixed_ssbo_metal_fixed_vector_float_mod_compound_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 67) buffer MetalFixedVectorOpsBlock {
        uint index;
        vec2 offsets[4];
        float data[];
    } metalFixedVectorOpsBlock;

    void main() {
        uint i = metalFixedVectorOpsBlock.index;
        metalFixedVectorOpsBlock.offsets[i] %= vec2(2.0);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "device uchar* metalFixedVectorOpsBlock [[buffer(67)]]" in metal
    assert "unsupported Metal GLSL buffer block compound store" in metal
    assert "operator %= is not supported for float buffer members" in metal
    assert "reinterpret_cast<device float*>" not in metal


def test_codegen_mixed_ssbo_metal_readonly_fixed_vector_write_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 68) readonly buffer MetalReadFixedVectorBlock {
        uint index;
        vec4 values[2];
        float tail[];
    } metalReadFixedVectorBlock;

    void main() {
        uint i = metalReadFixedVectorBlock.index;
        metalReadFixedVectorBlock.values[i] += vec4(1.0);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)
    assert "const device uchar* metalReadFixedVectorBlock [[buffer(68)]]" in metal
    assert "readonly device buffer cannot be written" in metal
    assert "reinterpret_cast<device float*>" not in metal


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
    assert (
        "float s = (*reinterpret_cast<const device float*>(metalMatrixBlock + 0));"
        in metal
    )
    assert "float3x3 t = float3x3(" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixBlock + 16))" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixBlock + 32))" in metal
    assert "(*reinterpret_cast<const device float*>(metalMatrixBlock + 56))" in metal
    assert "float3x3 __crossgl_matrix_store_0 = t;" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 16)) = "
        "__crossgl_matrix_store_0[0].x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 56)) = "
        "__crossgl_matrix_store_0[2].z;" in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalMatrixBlock + 64));" in metal
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
        "(metalNonSquareMatrixBlock + 16))" in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalNonSquareMatrixBlock + 40))" in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalNonSquareMatrixBlock + 48));" in metal
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
        "(metalRuntimeMatrixBlock + (16 + i * 64)))" in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64 + 48 + 12)))" in metal
    )
    assert "float4x4 __crossgl_matrix_store_0 = selected;" in metal
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64))) = "
        "__crossgl_matrix_store_0[0].x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalRuntimeMatrixBlock + (16 + i * 64 + 48 + 12))) = "
        "__crossgl_matrix_store_0[3].w;" in metal
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
        "__crossgl_matrix_store_0[0].x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixBlock + 76)) = "
        "__crossgl_matrix_store_0[3].w;" in metal
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
    assert (
        "(*reinterpret_cast<const device float*>(metalMatrixArrayBlock + 0))" in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>(metalMatrixArrayBlock + 60))" in metal
    )
    assert "float4x4 selected = float4x4(" in metal
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalMatrixArrayBlock + (i * 64)))" in metal
    )
    assert (
        "(*reinterpret_cast<const device float*>"
        "(metalMatrixArrayBlock + (i * 64 + 48 + 12)))" in metal
    )
    assert "float4x4 __crossgl_matrix_store_0 = selected;" in metal
    assert "float4x4 __crossgl_matrix_store_1 = first;" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixArrayBlock + 0)) = "
        "__crossgl_matrix_store_0[0].x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(metalMatrixArrayBlock + (i * 64 + 48 + 12))) = "
        "__crossgl_matrix_store_1[3].w;" in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(metalMatrixArrayBlock + 128));" in metal
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
    assert (
        "(*reinterpret_cast<const device float*>(metalMatrixArrayBlock + 64))" in metal
    )
    assert ") + value);" in metal
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixArrayBlock + 64)) = "
        "__crossgl_matrix_store_0[0].x;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(metalMatrixArrayBlock + 124)) = "
        "__crossgl_matrix_store_0[3].w;" in metal
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
        "(metalReadMatrixArrayBlock + (i * 64)))" in metal
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
        "(metalRuntimeMatrixBlock + (16 + i * 64)))" in metal
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
    expected_glsl = """
    #version 450 core
    layout(std430, binding = 60) buffer SnapshotBlock {
        uint count;
        vec3 axes[2];
        float data[];
    } snapshotBlock;
    // Compute Shader
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
        uint i = snapshotBlock.count;
        vec3 axis = snapshotBlock.axes[1];
        snapshotBlock.axes[0] = axis;
        float tail = snapshotBlock.data[0];
    }
    """

    assert normalize_codegen_snapshot(HLSLCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_hlsl)
    )
    assert normalize_codegen_snapshot(MetalCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_metal)
    )
    assert normalize_codegen_snapshot(GLSLCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_glsl)
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
    ByteAddressBuffer snapshotMatrixBlock : register(t61);
    // Compute Shader
    [numthreads(1, 1, 1)]
    void CSMain() {
        float2x2 transform = float2x2(asfloat(snapshotMatrixBlock.Load2(0)), asfloat(snapshotMatrixBlock.Load2(8)));
        float tail = asfloat(snapshotMatrixBlock.Load(16));
    }
    """
    expected_metal = """
    #include <metal_stdlib>
    using namespace metal;

    // Compute Shader
    kernel void kernel_main(const device uchar* snapshotMatrixBlock [[buffer(61)]]) {
        float2x2 transform = float2x2(float2((*reinterpret_cast<const device float*>(snapshotMatrixBlock + 0)), (*reinterpret_cast<const device float*>(snapshotMatrixBlock + 4))), float2((*reinterpret_cast<const device float*>(snapshotMatrixBlock + 8)), (*reinterpret_cast<const device float*>(snapshotMatrixBlock + 12))));
        float tail = (*reinterpret_cast<const device float*>(snapshotMatrixBlock + 16));
    }
    """
    expected_glsl = """
    #version 450 core
    layout(std430, binding = 61) readonly buffer SnapshotMatrixBlock {
        mat2 transform;
        float data[];
    } snapshotMatrixBlock;
    // Compute Shader
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
        mat2 transform = snapshotMatrixBlock.transform;
        float tail = snapshotMatrixBlock.data[0];
    }
    """

    assert normalize_codegen_snapshot(HLSLCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_hlsl)
    )
    assert normalize_codegen_snapshot(MetalCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_metal)
    )
    assert normalize_codegen_snapshot(GLSLCodeGen().generate(shader_ast)) == (
        normalize_codegen_snapshot(expected_glsl)
    )


def test_codegen_mixed_ssbo_bool_members_lower_as_uint_slots():
    crossgl = """
    shader main {
        struct BoolBlock {
            bool enabled;
            bool flags[2];
            float values[];
        };

        BoolBlock boolBlock @glsl_buffer_block(std430) @binding(53);

        bool readFlag(BoolBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.flags[i];
        }

        compute {
            void main() {
                bool enabled = boolBlock.enabled;
                bool first = boolBlock.flags[0];
                bool dynamic = readFlag(boolBlock, 1u);
                if (enabled && dynamic) {
                    boolBlock.flags[0] = false;
                }
                boolBlock.flags[1] = first;
                float tail = boolBlock.values[0];
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer boolBlock : register(u53);" in hlsl
    assert "bool readFlag(RWByteAddressBuffer localBlock, uint i)" in hlsl
    assert "return (localBlock.Load((4 + i * 4)) != 0u);" in hlsl
    assert "bool enabled = (boolBlock.Load(0) != 0u);" in hlsl
    assert "bool first = (boolBlock.Load(4) != 0u);" in hlsl
    assert "bool dynamic = readFlag(boolBlock, 1u);" in hlsl
    assert "boolBlock.Store(4, ((false) ? 1u : 0u));" in hlsl
    assert "boolBlock.Store(8, ((first) ? 1u : 0u));" in hlsl
    assert "float tail = asfloat(boolBlock.Load(12));" in hlsl
    assert "unsupported HLSL GLSL buffer block" not in hlsl

    assert "kernel void kernel_main(device uchar* boolBlock [[buffer(53)]])" in metal
    assert "bool readFlag(device uchar* localBlock, uint i)" in metal
    assert (
        "return ((*reinterpret_cast<const device uint*>"
        "(localBlock + (4 + i * 4))) != 0u);" in metal
    )
    assert (
        "bool enabled = ((*reinterpret_cast<const device uint*>"
        "(boolBlock + 0)) != 0u);" in metal
    )
    assert (
        "bool first = ((*reinterpret_cast<const device uint*>"
        "(boolBlock + 4)) != 0u);" in metal
    )
    assert "bool dynamic = readFlag(boolBlock, 1u);" in metal
    assert (
        "(*reinterpret_cast<device uint*>(boolBlock + 4)) = "
        "((false) ? 1u : 0u);" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(boolBlock + 8)) = "
        "((first) ? 1u : 0u);" in metal
    )
    assert (
        "float tail = (*reinterpret_cast<const device float*>"
        "(boolBlock + 12));" in metal
    )
    assert "unsupported Metal GLSL buffer block" not in metal

    assert "layout(std430, binding = 53) buffer BoolBlock" in glsl
    assert "bool enabled;" in glsl
    assert "bool flags[2];" in glsl
    assert "float values[];" in glsl
    assert "bool dynamic = readFlag(boolBlock, 1u);" in glsl


def test_codegen_mixed_ssbo_bool_vector_members_lower_as_uint_components():
    crossgl = """
    shader main {
        struct BoolVectorBlock {
            bvec3 mask;
            bvec2 pairs[2];
            bvec4 values[];
        };

        BoolVectorBlock boolVectorBlock @glsl_buffer_block(std430) @binding(55);

        bvec4 readValue(BoolVectorBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.values[i];
        }

        compute {
            void main() {
                bvec3 mask = boolVectorBlock.mask;
                bvec2 pair = boolVectorBlock.pairs[1];
                bvec4 dynamic = readValue(boolVectorBlock, 0u);
                boolVectorBlock.mask = bvec3(dynamic.x, pair.x, mask.z);
                boolVectorBlock.pairs[0] = bvec2(mask.x, dynamic.y);
                boolVectorBlock.values[1] = bvec4(mask.x, pair.y, dynamic.z, true);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer boolVectorBlock : register(u55);" in hlsl
    assert "bool4 readValue(RWByteAddressBuffer localBlock, uint i)" in hlsl
    assert (
        "return bool4((localBlock.Load((32 + i * 16)) != 0u), "
        "(localBlock.Load((32 + i * 16 + 4)) != 0u), "
        "(localBlock.Load((32 + i * 16 + 8)) != 0u), "
        "(localBlock.Load((32 + i * 16 + 12)) != 0u));" in hlsl
    )
    assert (
        "bool3 mask = bool3((boolVectorBlock.Load(0) != 0u), "
        "(boolVectorBlock.Load(4) != 0u), "
        "(boolVectorBlock.Load(8) != 0u));" in hlsl
    )
    assert (
        "bool2 pair = bool2((boolVectorBlock.Load(24) != 0u), "
        "(boolVectorBlock.Load(28) != 0u));" in hlsl
    )
    assert "bool3 __crossgl_bool_store_0 = bool3(dynamic.x, pair.x, mask.z);" in hlsl
    assert (
        "boolVectorBlock.Store3(0, uint3((__crossgl_bool_store_0.x ? 1u : 0u), "
        "(__crossgl_bool_store_0.y ? 1u : 0u), "
        "(__crossgl_bool_store_0.z ? 1u : 0u)));" in hlsl
    )
    assert "bool2 __crossgl_bool_store_1 = bool2(mask.x, dynamic.y);" in hlsl
    assert (
        "boolVectorBlock.Store2(16, uint2((__crossgl_bool_store_1.x ? 1u : 0u), "
        "(__crossgl_bool_store_1.y ? 1u : 0u)));" in hlsl
    )
    assert (
        "bool4 __crossgl_bool_store_2 = bool4(mask.x, pair.y, dynamic.z, true);" in hlsl
    )
    assert (
        "boolVectorBlock.Store4(48, uint4((__crossgl_bool_store_2.x ? 1u : 0u), "
        "(__crossgl_bool_store_2.y ? 1u : 0u), "
        "(__crossgl_bool_store_2.z ? 1u : 0u), "
        "(__crossgl_bool_store_2.w ? 1u : 0u)));" in hlsl
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert (
        "kernel void kernel_main(device uchar* boolVectorBlock [[buffer(55)]])" in metal
    )
    assert "bool4 readValue(device uchar* localBlock, uint i)" in metal
    assert (
        "reinterpret_cast<const device uint*>(localBlock + (32 + i * 16 + 12))" in metal
    )
    assert (
        "bool3 mask = bool3(((*reinterpret_cast<const device uint*>"
        "(boolVectorBlock + 0)) != 0u), "
        "((*reinterpret_cast<const device uint*>(boolVectorBlock + 4)) != 0u), "
        "((*reinterpret_cast<const device uint*>(boolVectorBlock + 8)) != 0u));"
        in metal
    )
    assert "bool3 __crossgl_buffer_store_0 = bool3(dynamic.x, pair.x, mask.z);" in metal
    assert (
        "(*reinterpret_cast<device uint*>(boolVectorBlock + 8)) = "
        "((__crossgl_buffer_store_0.z) ? 1u : 0u);" in metal
    )
    assert "bool2 __crossgl_buffer_store_1 = bool2(mask.x, dynamic.y);" in metal
    assert (
        "(*reinterpret_cast<device uint*>(boolVectorBlock + 20)) = "
        "((__crossgl_buffer_store_1.y) ? 1u : 0u);" in metal
    )
    assert (
        "bool4 __crossgl_buffer_store_2 = bool4(mask.x, pair.y, dynamic.z, true);"
        in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(boolVectorBlock + 60)) = "
        "((__crossgl_buffer_store_2.w) ? 1u : 0u);" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 55) buffer BoolVectorBlock" in glsl
    assert "bvec3 mask;" in glsl
    assert "bvec2 pairs[2];" in glsl
    assert "bvec4 values[];" in glsl
    assert "boolVectorBlock.values[1] = bvec4(mask.x, pair.y, dynamic.z, true);" in glsl


def test_codegen_mixed_ssbo_nested_struct_members_lower_as_leaf_offsets():
    crossgl = """
    shader main {
        struct InnerBlockData {
            float scale;
            bvec3 mask;
        };

        struct NestedBlock {
            uint count;
            InnerBlockData inner;
            float values[];
        };

        NestedBlock nestedBlock @glsl_buffer_block(std430) @binding(56);

        float readNested(NestedBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.inner.scale + localBlock.values[i];
        }

        compute {
            void main() {
                float scale = nestedBlock.inner.scale;
                bvec3 mask = nestedBlock.inner.mask;
                nestedBlock.inner.scale = scale + 1.0;
                nestedBlock.inner.mask = bvec3(mask.y, mask.x, true);
                nestedBlock.values[0] = readNested(nestedBlock, 0u);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer nestedBlock : register(u56);" in hlsl
    assert "float readNested(RWByteAddressBuffer localBlock, uint i)" in hlsl
    assert (
        "return (asfloat(localBlock.Load(16)) + "
        "asfloat(localBlock.Load((48 + i * 4))));" in hlsl
    )
    assert "float scale = asfloat(nestedBlock.Load(16));" in hlsl
    assert (
        "bool3 mask = bool3((nestedBlock.Load(32) != 0u), "
        "(nestedBlock.Load(36) != 0u), "
        "(nestedBlock.Load(40) != 0u));" in hlsl
    )
    assert "nestedBlock.Store(16, asuint((scale + 1.0)));" in hlsl
    assert "bool3 __crossgl_bool_store_0 = bool3(mask.y, mask.x, true);" in hlsl
    assert (
        "nestedBlock.Store3(32, uint3((__crossgl_bool_store_0.x ? 1u : 0u), "
        "(__crossgl_bool_store_0.y ? 1u : 0u), "
        "(__crossgl_bool_store_0.z ? 1u : 0u)));" in hlsl
    )
    assert "nestedBlock.Store(48, asuint(readNested(nestedBlock, 0u)));" in hlsl
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert "kernel void kernel_main(device uchar* nestedBlock [[buffer(56)]])" in metal
    assert "float readNested(device uchar* localBlock, uint i)" in metal
    assert (
        "return (*reinterpret_cast<const device float*>(localBlock + 16)) + "
        "(*reinterpret_cast<const device float*>(localBlock + (48 + i * 4)));" in metal
    )
    assert (
        "float scale = (*reinterpret_cast<const device float*>"
        "(nestedBlock + 16));" in metal
    )
    assert (
        "bool3 mask = bool3(((*reinterpret_cast<const device uint*>"
        "(nestedBlock + 32)) != 0u), "
        "((*reinterpret_cast<const device uint*>(nestedBlock + 36)) != 0u), "
        "((*reinterpret_cast<const device uint*>(nestedBlock + 40)) != 0u));" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(nestedBlock + 16)) = scale + 1.0;" in metal
    )
    assert "bool3 __crossgl_buffer_store_0 = bool3(mask.y, mask.x, true);" in metal
    assert (
        "(*reinterpret_cast<device uint*>(nestedBlock + 40)) = "
        "((__crossgl_buffer_store_0.z) ? 1u : 0u);" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(nestedBlock + 48)) = "
        "readNested(nestedBlock, 0u);" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 56) buffer NestedBlock" in glsl
    assert "InnerBlockData inner;" in glsl
    assert "nestedBlock.inner.scale = (scale + 1.0);" in glsl
    assert "nestedBlock.inner.mask = bvec3(mask.y, mask.x, true);" in glsl


def test_codegen_mixed_ssbo_nested_struct_arrays_lower_as_leaf_offsets():
    crossgl = """
    shader main {
        struct ArrayBlockData {
            uint id;
            vec3 normal;
            bvec2 flags;
        };

        struct NestedArrayBlock {
            ArrayBlockData fixedItems[2];
            uint count;
            ArrayBlockData items[];
        };

        NestedArrayBlock nestedArrayBlock @glsl_buffer_block(std430) @binding(57);

        float readNestedArray(NestedArrayBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.items[i].normal.x + float(localBlock.fixedItems[1].id);
        }

        compute {
            void main() {
                uint i = nestedArrayBlock.count;
                vec3 normal = nestedArrayBlock.fixedItems[1].normal;
                bvec2 flags = nestedArrayBlock.items[i].flags;
                nestedArrayBlock.fixedItems[0].id = nestedArrayBlock.items[i].id;
                nestedArrayBlock.items[i].normal = normal;
                nestedArrayBlock.items[i].flags = bvec2(flags.y, true);
                float value = readNestedArray(nestedArrayBlock, i);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer nestedArrayBlock : register(u57);" in hlsl
    assert "float readNestedArray(RWByteAddressBuffer localBlock, uint i)" in hlsl
    assert (
        "return (asfloat(localBlock.Load3((112 + i * 48 + 16))).x + "
        "float(localBlock.Load(48)));" in hlsl
    )
    assert "uint i = nestedArrayBlock.Load(96);" in hlsl
    assert "float3 normal = asfloat(nestedArrayBlock.Load3(64));" in hlsl
    assert (
        "bool2 flags = bool2((nestedArrayBlock.Load((112 + i * 48 + 32)) != 0u), "
        "(nestedArrayBlock.Load((112 + i * 48 + 32 + 4)) != 0u));" in hlsl
    )
    assert "nestedArrayBlock.Store(0, nestedArrayBlock.Load((112 + i * 48)));" in hlsl
    assert "nestedArrayBlock.Store3((112 + i * 48 + 16), asuint(normal));" in hlsl
    assert "bool2 __crossgl_bool_store_0 = bool2(flags.y, true);" in hlsl
    assert (
        "nestedArrayBlock.Store2((112 + i * 48 + 32), "
        "uint2((__crossgl_bool_store_0.x ? 1u : 0u), "
        "(__crossgl_bool_store_0.y ? 1u : 0u)));" in hlsl
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert (
        "kernel void kernel_main(device uchar* nestedArrayBlock [[buffer(57)]])"
        in metal
    )
    assert "float readNestedArray(device uchar* localBlock, uint i)" in metal
    assert "float((*reinterpret_cast<const device uint*>(localBlock + 48)))" in metal
    assert (
        "uint i = (*reinterpret_cast<const device uint*>"
        "(nestedArrayBlock + 96));" in metal
    )
    assert (
        "float3 normal = float3((*reinterpret_cast<const device float*>"
        "(nestedArrayBlock + 64)), "
        "(*reinterpret_cast<const device float*>(nestedArrayBlock + 68)), "
        "(*reinterpret_cast<const device float*>(nestedArrayBlock + 72)));" in metal
    )
    assert (
        "bool2 flags = bool2(((*reinterpret_cast<const device uint*>"
        "(nestedArrayBlock + (112 + i * 48 + 32))) != 0u), "
        "((*reinterpret_cast<const device uint*>"
        "(nestedArrayBlock + (112 + i * 48 + 32 + 4))) != 0u));" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(nestedArrayBlock + 0)) = "
        "(*reinterpret_cast<const device uint*>"
        "(nestedArrayBlock + (112 + i * 48)));" in metal
    )
    assert "float3 __crossgl_buffer_store_0 = normal;" in metal
    assert (
        "(*reinterpret_cast<device float*>"
        "(nestedArrayBlock + (112 + i * 48 + 16 + 8))) = "
        "__crossgl_buffer_store_0.z;" in metal
    )
    assert "bool2 __crossgl_buffer_store_1 = bool2(flags.y, true);" in metal
    assert (
        "(*reinterpret_cast<device uint*>"
        "(nestedArrayBlock + (112 + i * 48 + 32 + 4))) = "
        "((__crossgl_buffer_store_1.y) ? 1u : 0u);" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 57) buffer NestedArrayBlock" in glsl
    assert "ArrayBlockData fixedItems[2];" in glsl
    assert "ArrayBlockData items[];" in glsl
    assert "nestedArrayBlock.items[i].normal = normal;" in glsl
    assert "nestedArrayBlock.items[i].flags = bvec2(flags.y, true);" in glsl


def test_codegen_mixed_ssbo_nested_struct_aggregates_materialize_leaf_fields():
    crossgl = """
    shader main {
        struct AggregatePayload {
            float scale;
            bvec3 mask;
        };

        struct AggregateBlockData {
            AggregatePayload payload;
            uint id;
        };

        struct AggregateBlock {
            AggregateBlockData inner;
            AggregateBlockData items[];
        };

        AggregateBlock aggregateBlock @glsl_buffer_block(std430) @binding(58);

        AggregateBlockData passThrough(AggregateBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.items[i];
        }

        compute {
            void main() {
                uint i = 1u;
                AggregateBlockData inner = aggregateBlock.inner;
                AggregateBlockData item = passThrough(aggregateBlock, i);
                aggregateBlock.inner = item;
                aggregateBlock.items[i] = inner;
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer aggregateBlock : register(u58);" in hlsl
    assert re.search(
        r"AggregateBlockData __crossgl_load_rw_glsl_buffer_AggregateBlockData_[0-9a-f]{10}"
        r"\(RWByteAddressBuffer buffer, uint offset\)",
        hlsl,
    )
    assert "result.payload.scale = asfloat(buffer.Load(offset));" in hlsl
    assert (
        "result.payload.mask = bool3((buffer.Load((offset + 16)) != 0u), "
        "(buffer.Load((offset + 16 + 4)) != 0u), "
        "(buffer.Load((offset + 16 + 8)) != 0u));" in hlsl
    )
    assert "result.id = buffer.Load((offset + 32));" in hlsl
    assert (
        "AggregateBlockData passThrough(RWByteAddressBuffer localBlock, uint i)" in hlsl
    )
    assert re.search(
        r"return __crossgl_load_rw_glsl_buffer_AggregateBlockData_[0-9a-f]{10}"
        r"\(localBlock, \(48 \+ i \* 48\)\);",
        hlsl,
    )
    assert re.search(
        r"AggregateBlockData inner = "
        r"__crossgl_load_rw_glsl_buffer_AggregateBlockData_[0-9a-f]{10}"
        r"\(aggregateBlock, 0\);",
        hlsl,
    )
    assert "AggregateBlockData __crossgl_aggregate_store_0 = item;" in hlsl
    assert (
        "aggregateBlock.Store(0, "
        "asuint(__crossgl_aggregate_store_0.payload.scale));" in hlsl
    )
    assert (
        "bool3 __crossgl_bool_store_1 = "
        "__crossgl_aggregate_store_0.payload.mask;" in hlsl
    )
    assert (
        "aggregateBlock.Store3(16, uint3((__crossgl_bool_store_1.x ? 1u : 0u), "
        "(__crossgl_bool_store_1.y ? 1u : 0u), "
        "(__crossgl_bool_store_1.z ? 1u : 0u)));" in hlsl
    )
    assert "aggregateBlock.Store(32, __crossgl_aggregate_store_0.id);" in hlsl
    assert "AggregateBlockData __crossgl_aggregate_store_2 = inner;" in hlsl
    assert (
        "aggregateBlock.Store((48 + i * 48), "
        "asuint(__crossgl_aggregate_store_2.payload.scale));" in hlsl
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert (
        "kernel void kernel_main(device uchar* aggregateBlock [[buffer(58)]])" in metal
    )
    assert "AggregateBlockData passThrough(device uchar* localBlock, uint i)" in metal
    assert (
        "return AggregateBlockData{AggregatePayload{"
        "(*reinterpret_cast<const device float*>"
        "(localBlock + (48 + i * 48))), "
        "bool3(((*reinterpret_cast<const device uint*>"
        "(localBlock + (48 + i * 48 + 16))) != 0u), "
        "((*reinterpret_cast<const device uint*>"
        "(localBlock + (48 + i * 48 + 16 + 4))) != 0u), "
        "((*reinterpret_cast<const device uint*>"
        "(localBlock + (48 + i * 48 + 16 + 8))) != 0u))}, "
        "(*reinterpret_cast<const device uint*>"
        "(localBlock + (48 + i * 48 + 32)))};" in metal
    )
    assert (
        "AggregateBlockData inner = AggregateBlockData{"
        "AggregatePayload{"
        "(*reinterpret_cast<const device float*>(aggregateBlock + 0)), "
        "bool3(((*reinterpret_cast<const device uint*>(aggregateBlock + 16)) != 0u), "
        "((*reinterpret_cast<const device uint*>(aggregateBlock + 20)) != 0u), "
        "((*reinterpret_cast<const device uint*>(aggregateBlock + 24)) != 0u))}, "
        "(*reinterpret_cast<const device uint*>(aggregateBlock + 32))};" in metal
    )
    assert "AggregateBlockData __crossgl_aggregate_store_0 = item;" in metal
    assert (
        "(*reinterpret_cast<device float*>(aggregateBlock + 0)) = "
        "__crossgl_aggregate_store_0.payload.scale;" in metal
    )
    assert (
        "bool3 __crossgl_buffer_store_1 = "
        "__crossgl_aggregate_store_0.payload.mask;" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(aggregateBlock + 24)) = "
        "((__crossgl_buffer_store_1.z) ? 1u : 0u);" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(aggregateBlock + 32)) = "
        "__crossgl_aggregate_store_0.id;" in metal
    )
    assert "AggregateBlockData __crossgl_aggregate_store_2 = inner;" in metal
    assert (
        "(*reinterpret_cast<device float*>(aggregateBlock + (48 + i * 48))) = "
        "__crossgl_aggregate_store_2.payload.scale;" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 58) buffer AggregateBlock" in glsl
    assert "struct AggregatePayload" in glsl
    assert "AggregatePayload payload;" in glsl
    assert "AggregateBlockData inner;" in glsl
    assert "AggregateBlockData items[];" in glsl
    assert "aggregateBlock.inner = item;" in glsl
    assert "aggregateBlock.items[i] = inner;" in glsl


def test_codegen_mixed_ssbo_nested_struct_aggregate_arrays_materialize_leaf_fields():
    crossgl = """
    shader main {
        struct ArrayAggregateItem {
            vec2 uv;
            bvec2 flags;
        };

        struct ArrayAggregateData {
            float weights[2];
            ArrayAggregateItem items[2];
            uint id;
        };

        struct ArrayAggregateBlock {
            ArrayAggregateData inner;
            ArrayAggregateData entries[];
        };

        ArrayAggregateBlock arrayAggregateBlock @glsl_buffer_block(std430) @binding(59);

        ArrayAggregateData passArrayAggregate(ArrayAggregateBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.entries[i];
        }

        compute {
            void main() {
                uint i = 1u;
                ArrayAggregateData inner = arrayAggregateBlock.inner;
                ArrayAggregateData entry = passArrayAggregate(arrayAggregateBlock, i);
                arrayAggregateBlock.inner = entry;
                arrayAggregateBlock.entries[i] = inner;
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer arrayAggregateBlock : register(u59);" in hlsl
    assert re.search(
        r"ArrayAggregateData __crossgl_load_rw_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(RWByteAddressBuffer buffer, uint offset\)",
        hlsl,
    )
    assert "result.weights[0] = asfloat(buffer.Load(offset));" in hlsl
    assert "result.weights[1] = asfloat(buffer.Load((offset + 4)));" in hlsl
    assert "result.items[0].uv = asfloat(buffer.Load2((offset + 8)));" in hlsl
    assert (
        "result.items[1].flags = bool2((buffer.Load((offset + 8 + 16 + 8)) != 0u), "
        "(buffer.Load((offset + 8 + 16 + 8 + 4)) != 0u));" in hlsl
    )
    assert "result.id = buffer.Load((offset + 40));" in hlsl
    assert re.search(
        r"return __crossgl_load_rw_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(localBlock, \(48 \+ i \* 48\)\);",
        hlsl,
    )
    assert re.search(
        r"ArrayAggregateData inner = "
        r"__crossgl_load_rw_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(arrayAggregateBlock, 0\);",
        hlsl,
    )
    assert "ArrayAggregateData __crossgl_aggregate_store_0 = entry;" in hlsl
    assert (
        "arrayAggregateBlock.Store(0, "
        "asuint(__crossgl_aggregate_store_0.weights[0]));" in hlsl
    )
    assert (
        "arrayAggregateBlock.Store2(8, "
        "asuint(__crossgl_aggregate_store_0.items[0].uv));" in hlsl
    )
    assert "bool2 __crossgl_bool_store_1" in hlsl
    assert (
        "arrayAggregateBlock.Store((48 + i * 48 + 40), "
        "__crossgl_aggregate_store_3.id);" in hlsl
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert re.search(
        r"ArrayAggregateData __crossgl_load_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(const device uchar\* buffer, uint offset\)",
        metal,
    )
    assert (
        "result.weights[0] = "
        "(*reinterpret_cast<const device float*>(buffer + offset));" in metal
    )
    assert (
        "result.items[0].uv = float2((*reinterpret_cast<const device float*>"
        "(buffer + (offset + 8))), "
        "(*reinterpret_cast<const device float*>(buffer + (offset + 8 + 4))));" in metal
    )
    assert (
        "result.items[1].flags = bool2(((*reinterpret_cast<const device uint*>"
        "(buffer + (offset + 8 + 16 + 8))) != 0u), "
        "((*reinterpret_cast<const device uint*>"
        "(buffer + (offset + 8 + 16 + 8 + 4))) != 0u));" in metal
    )
    assert re.search(
        r"return __crossgl_load_glsl_buffer_ArrayAggregateData_[0-9a-f]{10}"
        r"\(localBlock, \(48 \+ i \* 48\)\);",
        metal,
    )
    assert "ArrayAggregateData __crossgl_aggregate_store_0 = entry;" in metal
    assert (
        "(*reinterpret_cast<device float*>(arrayAggregateBlock + 0)) = "
        "__crossgl_aggregate_store_0.weights[0];" in metal
    )
    assert "float2 __crossgl_buffer_store_1" in metal
    assert "bool2 __crossgl_buffer_store_2" in metal
    assert (
        "(*reinterpret_cast<device uint*>"
        "(arrayAggregateBlock + (48 + i * 48 + 40))) = "
        "__crossgl_aggregate_store_5.id;" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 59) buffer ArrayAggregateBlock" in glsl
    assert "float weights[2];" in glsl
    assert "ArrayAggregateItem items[2];" in glsl
    assert "ArrayAggregateData entries[];" in glsl
    assert "arrayAggregateBlock.inner = entry;" in glsl
    assert "arrayAggregateBlock.entries[i] = inner;" in glsl


def test_codegen_mixed_ssbo_readonly_aggregate_helpers_use_const_readers():
    crossgl = """
    shader main {
        struct ReadOnlyAggregateItem {
            vec2 uv;
            bvec2 flags;
        };

        struct ReadOnlyAggregateData {
            float weights[2];
            ReadOnlyAggregateItem items[2];
            uint id;
        };

        struct ReadOnlyAggregateBlock {
            ReadOnlyAggregateData inner;
            ReadOnlyAggregateData entries[];
        };

        ReadOnlyAggregateBlock readAggregateBlock @glsl_buffer_block(std430) @binding(95) @readonly;

        ReadOnlyAggregateData readEntry(ReadOnlyAggregateBlock localBlock @glsl_buffer_block(std430) @readonly, uint i) {
            return localBlock.entries[i];
        }

        compute {
            void main() {
                uint i = 1u;
                ReadOnlyAggregateData inner = readAggregateBlock.inner;
                ReadOnlyAggregateData entry = readEntry(readAggregateBlock, i);
                float weight = entry.weights[1] + inner.weights[0];
                bool flag = entry.items[1].flags.y;
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer readAggregateBlock : register(t95);" in hlsl
    assert "RWByteAddressBuffer readAggregateBlock" not in hlsl
    assert re.search(
        r"ReadOnlyAggregateData __crossgl_load_ro_glsl_buffer_"
        r"ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(ByteAddressBuffer buffer, uint offset\)",
        hlsl,
    )
    assert (
        "ReadOnlyAggregateData readEntry(ByteAddressBuffer localBlock, uint i)" in hlsl
    )
    assert re.search(
        r"return __crossgl_load_ro_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(localBlock, \(48 \+ i \* 48\)\);",
        hlsl,
    )
    assert re.search(
        r"ReadOnlyAggregateData inner = "
        r"__crossgl_load_ro_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(readAggregateBlock, 0\);",
        hlsl,
    )
    assert "result.items[1].flags = bool2" in hlsl
    assert "readAggregateBlock.Store" not in hlsl
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert "const device uchar* readAggregateBlock [[buffer(95)]]" in metal
    assert "kernel void kernel_main(device uchar* readAggregateBlock" not in metal
    assert re.search(
        r"ReadOnlyAggregateData __crossgl_load_glsl_buffer_"
        r"ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(const device uchar\* buffer, uint offset\)",
        metal,
    )
    assert (
        "ReadOnlyAggregateData readEntry(const device uchar* localBlock, uint i)"
        in metal
    )
    assert re.search(
        r"return __crossgl_load_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(localBlock, \(48 \+ i \* 48\)\);",
        metal,
    )
    assert re.search(
        r"ReadOnlyAggregateData inner = "
        r"__crossgl_load_glsl_buffer_ReadOnlyAggregateData_[0-9a-f]{10}"
        r"\(readAggregateBlock, 0\);",
        metal,
    )
    assert "result.items[1].flags = bool2" in metal
    assert "reinterpret_cast<device" not in metal
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 95) readonly buffer ReadOnlyAggregateBlock" in glsl
    assert "ReadOnlyAggregateData readEntry(ReadOnlyAggregateBlock localBlock" in glsl
    assert "ReadOnlyAggregateData inner = readAggregateBlock.inner;" in glsl
    assert "ReadOnlyAggregateData entry = readEntry(readAggregateBlock, i);" in glsl


def test_codegen_mixed_ssbo_nested_aggregate_leaf_compound_offsets():
    code = """
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

    layout(std430, binding = 98) buffer CompoundAggregateBlock {
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

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer compoundAggregateBlock : register(u98);" in hlsl
    assert "uint i = compoundAggregateBlock.Load(0);" in hlsl
    assert (
        "compoundAggregateBlock.Store((8 + i * 48 + 4), "
        "asuint((asfloat(compoundAggregateBlock.Load((8 + i * 48 + 4))) + 1.0)));"
        in hlsl
    )
    assert (
        "compoundAggregateBlock.Store2((8 + i * 48 + 8 + 16), "
        "asuint((asfloat(compoundAggregateBlock.Load2((8 + i * 48 + 8 + 16))) "
        "+ float2(0.5))));" in hlsl
    )
    assert "bool2 __crossgl_bool_store_0 = bool2(true, false);" in hlsl
    assert (
        "compoundAggregateBlock.Store2((8 + i * 48 + 8 + 8), "
        "uint2((__crossgl_bool_store_0.x ? 1u : 0u), "
        "(__crossgl_bool_store_0.y ? 1u : 0u)));" in hlsl
    )
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    assert "device uchar* compoundAggregateBlock [[buffer(98)]]" in metal
    assert (
        "uint i = (*reinterpret_cast<const device uint*>"
        "(compoundAggregateBlock + 0));" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 4))) = "
        "((*reinterpret_cast<const device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 4))) + 1.0);" in metal
    )
    assert (
        "float2 __crossgl_buffer_store_0 = "
        "(float2((*reinterpret_cast<const device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 8 + 16))), "
        "(*reinterpret_cast<const device float*>"
        "(compoundAggregateBlock + (8 + i * 48 + 8 + 16 + 4)))) "
        "+ float2(0.5));" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>"
        "(compoundAggregateBlock + (8 + i * 48 + 8 + 8 + 4))) = "
        "((__crossgl_buffer_store_1.y) ? 1u : 0u);" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 98) buffer CompoundAggregateBlock" in glsl
    assert "compoundAggregateBlock.entries[i].weights[1] += 1.0;" in glsl
    assert "compoundAggregateBlock.entries[i].items[1].uv += vec2(0.5);" in glsl
    assert (
        "compoundAggregateBlock.entries[i].items[0].flags = bvec2(true, false);" in glsl
    )


def test_codegen_mixed_ssbo_aggregate_helpers_distinguish_std140_and_std430_layouts():
    crossgl = """
    shader main {
        struct LayoutSharedData {
            float weights[2];
            uint id;
        };

        struct LayoutStd430Block {
            LayoutSharedData item;
        };

        struct LayoutStd140Block {
            LayoutSharedData item;
        };

        LayoutStd430Block block430 @glsl_buffer_block(std430) @binding(62);
        LayoutStd140Block block140 @glsl_buffer_block(std140) @binding(63);

        compute {
            void main() {
                LayoutSharedData a = block430.item;
                LayoutSharedData b = block140.item;
                block430.item = b;
                block140.item = a;
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    hlsl_helpers = re.findall(
        r"LayoutSharedData "
        r"(__crossgl_load_rw_glsl_buffer_LayoutSharedData_[0-9a-f]{10})"
        r"\(RWByteAddressBuffer buffer, uint offset\)",
        hlsl,
    )
    assert len(hlsl_helpers) == 2
    assert len(set(hlsl_helpers)) == 2
    assert "result.weights[1] = asfloat(buffer.Load((offset + 4)));" in hlsl
    assert "result.id = buffer.Load((offset + 8));" in hlsl
    assert "result.weights[1] = asfloat(buffer.Load((offset + 16)));" in hlsl
    assert "result.id = buffer.Load((offset + 32));" in hlsl
    assert re.search(
        r"LayoutSharedData a = "
        r"__crossgl_load_rw_glsl_buffer_LayoutSharedData_[0-9a-f]{10}"
        r"\(block430, 0\);",
        hlsl,
    )
    assert re.search(
        r"LayoutSharedData b = "
        r"__crossgl_load_rw_glsl_buffer_LayoutSharedData_[0-9a-f]{10}"
        r"\(block140, 0\);",
        hlsl,
    )
    assert "block430.Store(4, asuint(__crossgl_aggregate_store_0.weights[1]));" in hlsl
    assert "block430.Store(8, __crossgl_aggregate_store_0.id);" in hlsl
    assert "block140.Store(16, asuint(__crossgl_aggregate_store_1.weights[1]));" in hlsl
    assert "block140.Store(32, __crossgl_aggregate_store_1.id);" in hlsl
    assert ("un" + "supported HLSL GLSL buffer block") not in hlsl

    metal_helpers = re.findall(
        r"LayoutSharedData "
        r"(__crossgl_load_glsl_buffer_LayoutSharedData_[0-9a-f]{10})"
        r"\(const device uchar\* buffer, uint offset\)",
        metal,
    )
    assert len(metal_helpers) == 2
    assert len(set(metal_helpers)) == 2
    assert (
        "result.weights[1] = "
        "(*reinterpret_cast<const device float*>(buffer + (offset + 4)));" in metal
    )
    assert (
        "result.id = "
        "(*reinterpret_cast<const device uint*>(buffer + (offset + 8)));" in metal
    )
    assert (
        "result.weights[1] = "
        "(*reinterpret_cast<const device float*>(buffer + (offset + 16)));" in metal
    )
    assert (
        "result.id = "
        "(*reinterpret_cast<const device uint*>(buffer + (offset + 32)));" in metal
    )
    assert re.search(
        r"LayoutSharedData a = "
        r"__crossgl_load_glsl_buffer_LayoutSharedData_[0-9a-f]{10}"
        r"\(block430, 0\);",
        metal,
    )
    assert re.search(
        r"LayoutSharedData b = "
        r"__crossgl_load_glsl_buffer_LayoutSharedData_[0-9a-f]{10}"
        r"\(block140, 0\);",
        metal,
    )
    assert (
        "(*reinterpret_cast<device float*>(block430 + 4)) = "
        "__crossgl_aggregate_store_0.weights[1];" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(block430 + 8)) = "
        "__crossgl_aggregate_store_0.id;" in metal
    )
    assert (
        "(*reinterpret_cast<device float*>(block140 + 16)) = "
        "__crossgl_aggregate_store_1.weights[1];" in metal
    )
    assert (
        "(*reinterpret_cast<device uint*>(block140 + 32)) = "
        "__crossgl_aggregate_store_1.id;" in metal
    )
    assert ("un" + "supported Metal GLSL buffer block") not in metal

    assert "layout(std430, binding = 62) buffer LayoutStd430Block" in glsl
    assert "layout(std140, binding = 63) buffer LayoutStd140Block" in glsl
    assert "block430.item = b;" in glsl
    assert "block140.item = a;" in glsl


def test_codegen_mixed_ssbo_metal_store_parenthesizes_binary_ternary_operand():
    crossgl = """
    shader main {
        struct TernaryStoreBlock {
            bool enabled;
            float values[];
        };

        TernaryStoreBlock ternaryStoreBlock @glsl_buffer_block(std430) @binding(54);

        compute {
            void main() {
                uint i = 1u;
                ternaryStoreBlock.values[i] = ternaryStoreBlock.values[i] + (ternaryStoreBlock.enabled ? 1.0 : 0.0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    metal = MetalCodeGen().generate(shader_ast)

    assert (
        "(*reinterpret_cast<device float*>(ternaryStoreBlock + (4 + i * 4))) = "
        "(*reinterpret_cast<const device float*>"
        "(ternaryStoreBlock + (4 + i * 4))) + "
        "(((*reinterpret_cast<const device uint*>(ternaryStoreBlock + 0)) != 0u) "
        "? 1.0 : 0.0);" in metal
    )
    assert (
        "+ ((*reinterpret_cast<const device uint*>(ternaryStoreBlock + 0)) != 0u) "
        "? 1.0 : 0.0;" not in metal
    )


def test_codegen_mixed_ssbo_block_arrays_lower_to_byte_address_arrays():
    code = """
    #version 450 core
    layout(std430, binding = 72) coherent restrict buffer MixedArrayBlock {
        uint count;
        vec4 values[];
    } mixedBlocks[2];

    void main() {
        uint i = mixedBlocks[1].count;
        vec4 value = mixedBlocks[1].values[i];
        mixedBlocks[0].values[i] = value;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    assert (
        "MixedArrayBlock mixedBlocks[2] @glsl_buffer_block(std430) "
        "@binding(72) @coherent @restrict;" in crossgl
    )
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_hlsl = """
    globallycoherent RWByteAddressBuffer mixedBlocks[2] : register(u72);
    // Compute Shader
    [numthreads(1, 1, 1)]
    void CSMain() {
        uint i = mixedBlocks[1].Load(0);
        float4 value = asfloat(mixedBlocks[1].Load4((16 + i * 16)));
        mixedBlocks[0].Store4((16 + i * 16), asuint(value));
    }
    """
    expected_metal = """
    #include <metal_stdlib>
    using namespace metal;

    // Compute Shader
    kernel void kernel_main(array<device uchar*, 2> mixedBlocks [[buffer(72)]]) {
        uint i = (*reinterpret_cast<const device uint*>(mixedBlocks[1] + 0));
        float4 value = float4((*reinterpret_cast<const device float*>(mixedBlocks[1] + (16 + i * 16))), (*reinterpret_cast<const device float*>(mixedBlocks[1] + (16 + i * 16 + 4))), (*reinterpret_cast<const device float*>(mixedBlocks[1] + (16 + i * 16 + 8))), (*reinterpret_cast<const device float*>(mixedBlocks[1] + (16 + i * 16 + 12))));
        float4 __crossgl_buffer_store_0 = value;
        (*reinterpret_cast<device float*>(mixedBlocks[0] + (16 + i * 16))) = __crossgl_buffer_store_0.x;
        (*reinterpret_cast<device float*>(mixedBlocks[0] + (16 + i * 16 + 4))) = __crossgl_buffer_store_0.y;
        (*reinterpret_cast<device float*>(mixedBlocks[0] + (16 + i * 16 + 8))) = __crossgl_buffer_store_0.z;
        (*reinterpret_cast<device float*>(mixedBlocks[0] + (16 + i * 16 + 12))) = __crossgl_buffer_store_0.w;
    }
    """
    expected_glsl = """
    #version 450 core
    layout(std430, binding = 72) coherent restrict buffer MixedArrayBlock {
        uint count;
        vec4 values[];
    } mixedBlocks[2];
    // Compute Shader
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
        uint i = mixedBlocks[1].count;
        vec4 value = mixedBlocks[1].values[i];
        mixedBlocks[0].values[i] = value;
    }
    """

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "unsupported HLSL GLSL buffer block" not in hlsl
    assert "unsupported Metal GLSL buffer block" not in metal
    assert normalize_codegen_snapshot(hlsl) == normalize_codegen_snapshot(expected_hlsl)
    assert normalize_codegen_snapshot(metal) == normalize_codegen_snapshot(
        expected_metal
    )
    assert normalize_codegen_snapshot(glsl) == normalize_codegen_snapshot(expected_glsl)


def test_codegen_mixed_ssbo_readonly_block_arrays_lower_const_readers():
    code = """
    #version 450 core
    layout(std430, binding = 73) readonly buffer ReadMixedArrayBlock {
        mat2 transform;
        float values[];
    } readMixedBlocks[2];

    void main() {
        mat2 t = readMixedBlocks[1].transform;
        float value = readMixedBlocks[1].values[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_hlsl = """
    ByteAddressBuffer readMixedBlocks[2] : register(t73);
    // Compute Shader
    [numthreads(1, 1, 1)]
    void CSMain() {
        float2x2 t = float2x2(asfloat(readMixedBlocks[1].Load2(0)), asfloat(readMixedBlocks[1].Load2(8)));
        float value = asfloat(readMixedBlocks[1].Load(16));
    }
    """
    expected_metal = """
    #include <metal_stdlib>
    using namespace metal;

    // Compute Shader
    kernel void kernel_main(array<const device uchar*, 2> readMixedBlocks [[buffer(73)]]) {
        float2x2 t = float2x2(float2((*reinterpret_cast<const device float*>(readMixedBlocks[1] + 0)), (*reinterpret_cast<const device float*>(readMixedBlocks[1] + 4))), float2((*reinterpret_cast<const device float*>(readMixedBlocks[1] + 8)), (*reinterpret_cast<const device float*>(readMixedBlocks[1] + 12))));
        float value = (*reinterpret_cast<const device float*>(readMixedBlocks[1] + 16));
    }
    """
    expected_glsl = """
    #version 450 core
    layout(std430, binding = 73) readonly buffer ReadMixedArrayBlock {
        mat2 transform;
        float values[];
    } readMixedBlocks[2];
    // Compute Shader
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
        mat2 t = readMixedBlocks[1].transform;
        float value = readMixedBlocks[1].values[0];
    }
    """

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer readMixedBlocks" not in hlsl
    assert "device uchar* readMixedBlocks" not in metal
    assert normalize_codegen_snapshot(hlsl) == normalize_codegen_snapshot(expected_hlsl)
    assert normalize_codegen_snapshot(metal) == normalize_codegen_snapshot(
        expected_metal
    )
    assert normalize_codegen_snapshot(glsl) == normalize_codegen_snapshot(expected_glsl)


def test_codegen_mixed_ssbo_unsized_block_arrays_infer_literal_size():
    code = """
    #version 450 core
    layout(std430, binding = 74) buffer UnsizedMixedBlock {
        uint count;
        vec4 values[];
    } unsizedMixed[];

    void main() {
        uint i = unsizedMixed[2].count;
        vec4 value = unsizedMixed[2].values[i];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_hlsl = """
    RWByteAddressBuffer unsizedMixed[3] : register(u74);
    // Compute Shader
    [numthreads(1, 1, 1)]
    void CSMain() {
        uint i = unsizedMixed[2].Load(0);
        float4 value = asfloat(unsizedMixed[2].Load4((16 + i * 16)));
    }
    """
    expected_metal = """
    #include <metal_stdlib>
    using namespace metal;

    // Compute Shader
    kernel void kernel_main(array<device uchar*, 3> unsizedMixed [[buffer(74)]]) {
        uint i = (*reinterpret_cast<const device uint*>(unsizedMixed[2] + 0));
        float4 value = float4((*reinterpret_cast<const device float*>(unsizedMixed[2] + (16 + i * 16))), (*reinterpret_cast<const device float*>(unsizedMixed[2] + (16 + i * 16 + 4))), (*reinterpret_cast<const device float*>(unsizedMixed[2] + (16 + i * 16 + 8))), (*reinterpret_cast<const device float*>(unsizedMixed[2] + (16 + i * 16 + 12))));
    }
    """
    expected_glsl = """
    #version 450 core
    layout(std430, binding = 74) buffer UnsizedMixedBlock {
        uint count;
        vec4 values[];
    } unsizedMixed[];
    // Compute Shader
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
        uint i = unsizedMixed[2].count;
        vec4 value = unsizedMixed[2].values[i];
    }
    """

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "unsizedMixed[]" not in hlsl
    assert "array<device uchar*, 1> unsizedMixed" not in metal
    assert normalize_codegen_snapshot(hlsl) == normalize_codegen_snapshot(expected_hlsl)
    assert normalize_codegen_snapshot(metal) == normalize_codegen_snapshot(
        expected_metal
    )
    assert normalize_codegen_snapshot(glsl) == normalize_codegen_snapshot(expected_glsl)


def test_codegen_mixed_ssbo_unsized_block_arrays_propagate_to_metal_helpers():
    code = """
    #version 450 core
    layout(std430, binding = 75) readonly buffer HelperMixedBlock {
        mat2 transform;
        float values[];
    } helperBlocks[];

    float readValue(uint i) {
        mat2 t = helperBlocks[2].transform;
        return helperBlocks[2].values[i];
    }

    void main() {
        float value = readValue(0u);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer helperBlocks[3] : register(t75);" in hlsl
    assert "helperBlocks[2].Load2(0)" in hlsl
    assert "helperBlocks[2].Load((16 + i * 4))" in hlsl

    assert "array<const device uchar*, 3> helperBlocks [[buffer(75)]]" in metal
    assert "readValue(0u, helperBlocks)" in metal
    assert (
        "float readValue(uint i, array<const device uchar*, 3> helperBlocks)" in metal
    )
    assert "array<const device uchar*, 1> helperBlocks" not in metal
    assert "helperBlocks[2] + (16 + i * 4)" in metal

    assert "layout(std430, binding = 75) readonly buffer HelperMixedBlock" in glsl
    assert "float value = readValue(0u);" in glsl
    assert "helperBlocks[2].values[i]" in glsl


def test_codegen_mixed_ssbo_helper_array_store_uses_indexed_byte_address_receiver():
    code = """
    #version 450 core
    layout(std430, binding = 79) buffer HelperWriteBlock {
        uint count;
        vec4 values[];
    } helperWriteBlocks[];

    void writeValue(uint i, vec4 value) {
        helperWriteBlocks[2].values[i] = value;
    }

    void main() {
        uint i = helperWriteBlocks[2].count;
        writeValue(i, vec4(1.0));
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer helperWriteBlocks[3] : register(u79);" in hlsl
    assert "uint i = helperWriteBlocks[2].Load(0);" in hlsl
    assert "writeValue(i, float4(1.0));" in hlsl
    assert "helperWriteBlocks[2].Store4((16 + i * 16), asuint(value));" in hlsl

    assert "array<device uchar*, 3> helperWriteBlocks [[buffer(79)]]" in metal
    assert "writeValue(i, float4(1.0), helperWriteBlocks);" in metal
    assert (
        "void writeValue(uint i, float4 value, "
        "array<device uchar*, 3> helperWriteBlocks)" in metal
    )
    assert "helperWriteBlocks[2] + (16 + i * 16 + 12)" in metal

    assert "layout(std430, binding = 79) buffer HelperWriteBlock" in glsl
    assert "writeValue(i, vec4(1.0));" in glsl
    assert "helperWriteBlocks[2].values[i] = value;" in glsl


def test_codegen_mixed_ssbo_helper_array_compound_store_uses_indexed_receiver():
    code = """
    #version 450 core
    layout(std430, binding = 80) buffer HelperCompoundBlock {
        uint count;
        vec2 values[];
    } helperCompoundBlocks[];

    void addValue(uint i) {
        helperCompoundBlocks[1].values[i] += vec2(1.0);
    }

    void main() {
        uint i = helperCompoundBlocks[1].count;
        addValue(i);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer helperCompoundBlocks[2] : register(u80);" in hlsl
    assert "uint i = helperCompoundBlocks[1].Load(0);" in hlsl
    assert (
        "helperCompoundBlocks[1].Store2((8 + i * 8), "
        "asuint((asfloat(helperCompoundBlocks[1].Load2((8 + i * 8))) + float2(1.0))));"
        in hlsl
    )

    assert "array<device uchar*, 2> helperCompoundBlocks [[buffer(80)]]" in metal
    assert "addValue(i, helperCompoundBlocks);" in metal
    assert (
        "void addValue(uint i, array<device uchar*, 2> helperCompoundBlocks)" in metal
    )
    assert "helperCompoundBlocks[1] + (8 + i * 8 + 4)" in metal
    assert "unsupported Metal GLSL buffer block compound store" not in metal

    assert "layout(std430, binding = 80) buffer HelperCompoundBlock" in glsl
    assert "helperCompoundBlocks[1].values[i] += vec2(1.0);" in glsl


def test_codegen_mixed_ssbo_helper_readonly_array_write_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 81) readonly buffer HelperReadonlyBlock {
        uint count;
        float values[];
    } helperReadonlyBlocks[];

    void writeValue(uint i) {
        helperReadonlyBlocks[1].values[i] = 1.0;
    }

    void main() {
        writeValue(0u);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer helperReadonlyBlocks[2] : register(t81);" in hlsl
    assert "RWByteAddressBuffer helperReadonlyBlocks" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "helperReadonlyBlocks[1].Store" not in hlsl

    assert "array<const device uchar*, 2> helperReadonlyBlocks [[buffer(81)]]" in metal
    assert "array<device uchar*, 2> helperReadonlyBlocks" not in metal
    assert "readonly device buffer cannot be written" in metal
    assert "reinterpret_cast<device float*>" not in metal

    assert "layout(std430, binding = 81) readonly buffer HelperReadonlyBlock" in glsl
    assert "helperReadonlyBlocks[1].values[i] = 1.0;" in glsl


def test_codegen_mixed_ssbo_helper_matrix_array_store_uses_indexed_receiver():
    code = """
    #version 450 core
    layout(std430, binding = 82) buffer HelperMatrixBlock {
        uint count;
        mat4 transforms[];
    } helperMatrixBlocks[];

    void writeTransform(uint i, mat4 value) {
        helperMatrixBlocks[2].transforms[i] = value;
    }

    void main() {
        uint i = helperMatrixBlocks[2].count;
        mat4 value = helperMatrixBlocks[2].transforms[i];
        writeTransform(i, value);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer helperMatrixBlocks[3] : register(u82);" in hlsl
    assert "uint i = helperMatrixBlocks[2].Load(0);" in hlsl
    assert "writeTransform(i, value);" in hlsl
    assert "helperMatrixBlocks[2].Store4((16 + i * 64), asuint(value[0]));" in hlsl
    assert "helperMatrixBlocks[2].Store4((16 + i * 64 + 48), asuint(value[3]));" in hlsl

    assert "array<device uchar*, 3> helperMatrixBlocks [[buffer(82)]]" in metal
    assert "writeTransform(i, value, helperMatrixBlocks);" in metal
    assert (
        "void writeTransform(uint i, float4x4 value, "
        "array<device uchar*, 3> helperMatrixBlocks)" in metal
    )
    assert "float4x4 __crossgl_matrix_store_0 = value;" in metal
    assert "helperMatrixBlocks[2] + (16 + i * 64 + 48 + 12)" in metal

    assert "layout(std430, binding = 82) buffer HelperMatrixBlock" in glsl
    assert "helperMatrixBlocks[2].transforms[i] = value;" in glsl


def test_codegen_mixed_ssbo_helper_matrix_array_compound_store_uses_temp():
    code = """
    #version 450 core
    layout(std430, binding = 83) buffer HelperMatrixCompoundBlock {
        uint count;
        mat3 transforms[];
    } helperMatrixCompoundBlocks[];

    void addTransform(uint i, mat3 value) {
        helperMatrixCompoundBlocks[1].transforms[i] += value;
    }

    void main() {
        uint i = helperMatrixCompoundBlocks[1].count;
        mat3 value = helperMatrixCompoundBlocks[1].transforms[i];
        addTransform(i, value);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer helperMatrixCompoundBlocks[2] : register(u83);" in hlsl
    assert "uint i = helperMatrixCompoundBlocks[1].Load(0);" in hlsl
    assert "float3x3 __crossgl_matrix_store_0 = (" in hlsl
    assert "helperMatrixCompoundBlocks[1].Load3((16 + i * 48 + 32))" in hlsl
    assert (
        "helperMatrixCompoundBlocks[1].Store3((16 + i * 48 + 32), "
        "asuint(__crossgl_matrix_store_0[2]));" in hlsl
    )
    assert "unsupported HLSL GLSL buffer block matrix compound store" not in hlsl

    assert "array<device uchar*, 2> helperMatrixCompoundBlocks [[buffer(83)]]" in metal
    assert "addTransform(i, value, helperMatrixCompoundBlocks);" in metal
    assert (
        "void addTransform(uint i, float3x3 value, "
        "array<device uchar*, 2> helperMatrixCompoundBlocks)" in metal
    )
    assert "float3x3 __crossgl_matrix_store_0 = (" in metal
    assert "helperMatrixCompoundBlocks[1] + (16 + i * 48 + 32 + 8)" in metal
    assert "unsupported Metal GLSL buffer block matrix compound store" not in metal

    assert "layout(std430, binding = 83) buffer HelperMatrixCompoundBlock" in glsl
    assert "helperMatrixCompoundBlocks[1].transforms[i] += value;" in glsl


def test_codegen_mixed_ssbo_helper_readonly_matrix_array_write_stays_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 84) readonly buffer HelperReadMatrixBlock {
        uint count;
        mat4 transforms[];
    } helperReadMatrixBlocks[];

    void writeTransform(uint i, mat4 value) {
        helperReadMatrixBlocks[1].transforms[i] = value;
    }

    void main() {
        mat4 value = helperReadMatrixBlocks[1].transforms[0];
        writeTransform(0u, value);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer helperReadMatrixBlocks[2] : register(t84);" in hlsl
    assert "RWByteAddressBuffer helperReadMatrixBlocks" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "helperReadMatrixBlocks[1].Store" not in hlsl

    assert (
        "array<const device uchar*, 2> helperReadMatrixBlocks [[buffer(84)]]" in metal
    )
    assert "array<device uchar*, 2> helperReadMatrixBlocks" not in metal
    assert "readonly device buffer cannot be written" in metal
    assert "reinterpret_cast<device float*>" not in metal

    assert "layout(std430, binding = 84) readonly buffer HelperReadMatrixBlock" in glsl
    assert "helperReadMatrixBlocks[1].transforms[i] = value;" in glsl


def test_codegen_mixed_ssbo_helper_matrix_array_unsupported_compound_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 85) buffer HelperMatrixUnsupportedBlock {
        uint count;
        mat4 transforms[];
    } helperMatrixUnsupportedBlocks[];

    void modTransform(uint i, mat4 value) {
        helperMatrixUnsupportedBlocks[1].transforms[i] %= value;
    }

    void main() {
        mat4 value = helperMatrixUnsupportedBlocks[1].transforms[0];
        modTransform(0u, value);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "RWByteAddressBuffer helperMatrixUnsupportedBlocks[2] : register(u85);" in hlsl
    )
    assert "unsupported HLSL GLSL buffer block matrix compound store" in hlsl
    assert "helperMatrixUnsupportedBlocks[1].Store" not in hlsl

    assert (
        "array<device uchar*, 2> helperMatrixUnsupportedBlocks [[buffer(85)]]" in metal
    )
    assert "unsupported Metal GLSL buffer block matrix compound store" in metal
    assert "reinterpret_cast<device float*>" not in metal

    assert "layout(std430, binding = 85) buffer HelperMatrixUnsupportedBlock" in glsl
    assert "helperMatrixUnsupportedBlocks[1].transforms[i] %= value;" in glsl


def test_codegen_mixed_ssbo_nested_helper_read_threads_metal_resource_array():
    code = """
    #version 450 core
    layout(std430, binding = 86) readonly buffer NestedReadBlock {
        uint count;
        vec4 values[];
    } nestedReadBlocks[];

    vec4 leaf(uint i) {
        return nestedReadBlocks[2].values[i];
    }

    vec4 middle(uint i) {
        return leaf(i);
    }

    void main() {
        uint i = nestedReadBlocks[2].count;
        vec4 value = middle(i);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer nestedReadBlocks[3] : register(t86);" in hlsl
    assert "float4 value = middle(i);" in hlsl
    assert "return leaf(i);" in hlsl
    assert "nestedReadBlocks[2].Load4((16 + i * 16))" in hlsl

    assert "array<const device uchar*, 3> nestedReadBlocks [[buffer(86)]]" in metal
    assert "float4 value = middle(i, nestedReadBlocks);" in metal
    assert (
        "float4 leaf(uint i, array<const device uchar*, 3> nestedReadBlocks)" in metal
    )
    assert (
        "float4 middle(uint i, array<const device uchar*, 3> nestedReadBlocks)" in metal
    )
    assert "return leaf(i, nestedReadBlocks);" in metal
    assert "nestedReadBlocks[2] + (16 + i * 16 + 12)" in metal

    assert "layout(std430, binding = 86) readonly buffer NestedReadBlock" in glsl
    assert "vec4 value = middle(i);" in glsl
    assert "return leaf(i);" in glsl


def test_codegen_mixed_ssbo_nested_helper_write_threads_metal_resource_array():
    code = """
    #version 450 core
    layout(std430, binding = 87) buffer NestedWriteBlock {
        uint count;
        vec4 values[];
    } nestedWriteBlocks[];

    void leaf(uint i, vec4 value) {
        nestedWriteBlocks[2].values[i] = value;
    }

    void middle(uint i, vec4 value) {
        leaf(i, value);
    }

    void main() {
        uint i = nestedWriteBlocks[2].count;
        middle(i, vec4(1.0));
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer nestedWriteBlocks[3] : register(u87);" in hlsl
    assert "middle(i, float4(1.0));" in hlsl
    assert "leaf(i, value);" in hlsl
    assert "nestedWriteBlocks[2].Store4((16 + i * 16), asuint(value));" in hlsl

    assert "array<device uchar*, 3> nestedWriteBlocks [[buffer(87)]]" in metal
    assert "middle(i, float4(1.0), nestedWriteBlocks);" in metal
    assert (
        "void leaf(uint i, float4 value, array<device uchar*, 3> nestedWriteBlocks)"
        in metal
    )
    assert (
        "void middle(uint i, float4 value, array<device uchar*, 3> nestedWriteBlocks)"
        in metal
    )
    assert "leaf(i, value, nestedWriteBlocks);" in metal
    assert "nestedWriteBlocks[2] + (16 + i * 16 + 12)" in metal

    assert "layout(std430, binding = 87) buffer NestedWriteBlock" in glsl
    assert "middle(i, vec4(1.0));" in glsl
    assert "leaf(i, value);" in glsl


def test_codegen_mixed_ssbo_nested_helper_matrix_compound_threads_resource_array():
    code = """
    #version 450 core
    layout(std430, binding = 88) buffer NestedMatrixBlock {
        uint count;
        mat2 transforms[];
    } nestedMatrixBlocks[];

    void leaf(uint i, mat2 value) {
        nestedMatrixBlocks[2].transforms[i] += value;
    }

    void middle(uint i, mat2 value) {
        leaf(i, value);
    }

    void main() {
        uint i = nestedMatrixBlocks[2].count;
        mat2 value = nestedMatrixBlocks[2].transforms[i];
        middle(i, value);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer nestedMatrixBlocks[3] : register(u88);" in hlsl
    assert "float2x2 value = float2x2(" in hlsl
    assert "middle(i, value);" in hlsl
    assert "float2x2 __crossgl_matrix_store_0 = (" in hlsl
    assert (
        "nestedMatrixBlocks[2].Store2((8 + i * 16 + 8), asuint(__crossgl_matrix_store_0[1]));"
        in hlsl
    )

    assert "array<device uchar*, 3> nestedMatrixBlocks [[buffer(88)]]" in metal
    assert "middle(i, value, nestedMatrixBlocks);" in metal
    assert (
        "void leaf(uint i, float2x2 value, array<device uchar*, 3> nestedMatrixBlocks)"
        in metal
    )
    assert (
        "void middle(uint i, float2x2 value, array<device uchar*, 3> nestedMatrixBlocks)"
        in metal
    )
    assert "leaf(i, value, nestedMatrixBlocks);" in metal
    assert "float2x2 __crossgl_matrix_store_0 = (" in metal
    assert "nestedMatrixBlocks[2] + (8 + i * 16 + 8 + 4)" in metal

    assert "layout(std430, binding = 88) buffer NestedMatrixBlock" in glsl
    assert "middle(i, value);" in glsl
    assert "leaf(i, value);" in glsl


def test_codegen_mixed_ssbo_explicit_readonly_block_array_parameter_lowers():
    crossgl = """
    shader main {
        struct ParamMixedBlock {
            uint count;
            vec4 values[];
        };

        ParamMixedBlock blocks[3] @glsl_buffer_block(std430) @binding(89) @readonly;

        vec4 readFrom(ParamMixedBlock localBlocks[] @glsl_buffer_block(std430) @readonly, uint i) {
            return localBlocks[2].values[i];
        }

        compute {
            void main() {
                uint i = blocks[2].count;
                vec4 value = readFrom(blocks, i);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer blocks[3] : register(t89);" in hlsl
    assert "float4 readFrom(ByteAddressBuffer localBlocks[3], uint i)" in hlsl
    assert "return asfloat(localBlocks[2].Load4((16 + i * 16)));" in hlsl
    assert "ParamMixedBlock localBlocks[]" not in hlsl

    assert "array<const device uchar*, 3> blocks [[buffer(89)]]" in metal
    assert "float4 readFrom(array<const device uchar*, 3> localBlocks, uint i)" in metal
    assert "localBlocks[2] + (16 + i * 16 + 12)" in metal
    assert "ParamMixedBlock localBlocks[]" not in metal

    assert "vec4 readFrom(ParamMixedBlock localBlocks[], uint i)" in glsl
    assert "return localBlocks[2].values[i];" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_explicit_writable_block_array_parameter_lowers():
    crossgl = """
    shader main {
        struct ParamWriteBlock {
            uint count;
            vec4 values[];
        };

        ParamWriteBlock blocks[3] @glsl_buffer_block(std430) @binding(90);

        void writeTo(ParamWriteBlock localBlocks[] @glsl_buffer_block(std430), uint i, vec4 value) {
            localBlocks[2].values[i] = value;
        }

        compute {
            void main() {
                writeTo(blocks, 1u, vec4(1.0));
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer blocks[3] : register(u90);" in hlsl
    assert (
        "void writeTo(RWByteAddressBuffer localBlocks[3], uint i, float4 value)" in hlsl
    )
    assert "localBlocks[2].Store4((16 + i * 16), asuint(value));" in hlsl
    assert "ParamWriteBlock localBlocks[]" not in hlsl

    assert "array<device uchar*, 3> blocks [[buffer(90)]]" in metal
    assert (
        "void writeTo(array<device uchar*, 3> localBlocks, uint i, float4 value)"
        in metal
    )
    assert "localBlocks[2] + (16 + i * 16 + 12)" in metal
    assert "ParamWriteBlock localBlocks[]" not in metal

    assert "void writeTo(ParamWriteBlock localBlocks[], uint i, vec4 value)" in glsl
    assert "localBlocks[2].values[i] = value;" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_explicit_matrix_block_array_parameter_compound_store():
    crossgl = """
    shader main {
        struct ParamMatrixBlock {
            uint count;
            mat2 transforms[];
        };

        ParamMatrixBlock blocks[3] @glsl_buffer_block(std430) @binding(91);

        void addTo(ParamMatrixBlock localBlocks[] @glsl_buffer_block(std430), uint i, mat2 value) {
            localBlocks[2].transforms[i] += value;
        }

        compute {
            void main() {
                uint i = blocks[2].count;
                mat2 value = blocks[2].transforms[i];
                addTo(blocks, i, value);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer blocks[3] : register(u91);" in hlsl
    assert (
        "void addTo(RWByteAddressBuffer localBlocks[3], uint i, float2x2 value)" in hlsl
    )
    assert "float2x2 __crossgl_matrix_store_0 = (" in hlsl
    assert (
        "localBlocks[2].Store2((8 + i * 16 + 8), asuint(__crossgl_matrix_store_0[1]));"
        in hlsl
    )

    assert "array<device uchar*, 3> blocks [[buffer(91)]]" in metal
    assert (
        "void addTo(array<device uchar*, 3> localBlocks, uint i, float2x2 value)"
        in metal
    )
    assert "float2x2 __crossgl_matrix_store_0 = (" in metal
    assert "localBlocks[2] + (8 + i * 16 + 8 + 4)" in metal

    assert "void addTo(ParamMatrixBlock localBlocks[], uint i, mat2 value)" in glsl
    assert "localBlocks[2].transforms[i] += value;" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_explicit_readonly_block_array_parameter_write_stays_diagnostic():
    crossgl = """
    shader main {
        struct ParamReadOnlyBlock {
            uint count;
            vec4 values[];
        };

        ParamReadOnlyBlock blocks[2] @glsl_buffer_block(std430) @binding(92) @readonly;

        void writeTo(ParamReadOnlyBlock localBlocks[] @glsl_buffer_block(std430) @readonly, uint i) {
            localBlocks[1].values[i] = vec4(1.0);
        }

        compute {
            void main() {
                writeTo(blocks, 0u);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer blocks[2] : register(t92);" in hlsl
    assert "void writeTo(ByteAddressBuffer localBlocks[2], uint i)" in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "localBlocks[1].Store" not in hlsl

    assert "array<const device uchar*, 2> blocks [[buffer(92)]]" in metal
    assert "void writeTo(array<const device uchar*, 2> localBlocks, uint i)" in metal
    assert "readonly device buffer cannot be written" in metal
    assert "reinterpret_cast<device float*>" not in metal

    assert "void writeTo(ParamReadOnlyBlock localBlocks[], uint i)" in glsl
    assert "localBlocks[1].values[i] = vec4(1.0);" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_explicit_readonly_single_block_parameter_lowers():
    crossgl = """
    shader main {
        struct ParamSingleBlock {
            uint count;
            vec4 values[];
        };

        ParamSingleBlock block @glsl_buffer_block(std430) @binding(93) @readonly;

        vec4 readOne(ParamSingleBlock localBlock @glsl_buffer_block(std430) @readonly, uint i) {
            return localBlock.values[i];
        }

        compute {
            void main() {
                uint i = block.count;
                vec4 value = readOne(block, i);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "ByteAddressBuffer block : register(t93);" in hlsl
    assert "float4 readOne(ByteAddressBuffer localBlock, uint i)" in hlsl
    assert "return asfloat(localBlock.Load4((16 + i * 16)));" in hlsl
    assert "ParamSingleBlock localBlock" not in hlsl

    assert "const device uchar* block [[buffer(93)]]" in metal
    assert "float4 readOne(const device uchar* localBlock, uint i)" in metal
    assert "localBlock + (16 + i * 16 + 12)" in metal
    assert "ParamSingleBlock localBlock" not in metal

    assert "vec4 readOne(ParamSingleBlock localBlock, uint i)" in glsl
    assert "return localBlock.values[i];" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_explicit_single_block_parameter_forwards_array_element():
    crossgl = """
    shader main {
        struct ParamForwardElementBlock {
            uint count;
            mat2 transforms[];
        };

        ParamForwardElementBlock blocks[] @glsl_buffer_block(std430) @binding(94);

        void leaf(ParamForwardElementBlock localBlock @glsl_buffer_block(std430), uint i, mat2 value) {
            localBlock.transforms[i] += value;
        }

        void middle(ParamForwardElementBlock localBlock @glsl_buffer_block(std430), uint i, mat2 value) {
            leaf(localBlock, i, value);
        }

        compute {
            void main() {
                uint i = blocks[2].count;
                mat2 value = blocks[2].transforms[i];
                middle(blocks[2], i, value);
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer blocks[3] : register(u94);" in hlsl
    assert "void leaf(RWByteAddressBuffer localBlock, uint i, float2x2 value)" in hlsl
    assert "void middle(RWByteAddressBuffer localBlock, uint i, float2x2 value)" in hlsl
    assert "middle(blocks[2], i, value);" in hlsl
    assert "leaf(localBlock, i, value);" in hlsl
    assert "float2x2 __crossgl_matrix_store_0 = (" in hlsl
    assert (
        "localBlock.Store2((8 + i * 16 + 8), asuint(__crossgl_matrix_store_0[1]));"
        in hlsl
    )

    assert "array<device uchar*, 3> blocks [[buffer(94)]]" in metal
    assert "void leaf(device uchar* localBlock, uint i, float2x2 value)" in metal
    assert "void middle(device uchar* localBlock, uint i, float2x2 value)" in metal
    assert "middle(blocks[2], i, value);" in metal
    assert "leaf(localBlock, i, value);" in metal
    assert "float2x2 __crossgl_matrix_store_0 = (" in metal
    assert "localBlock + (8 + i * 16 + 8 + 4)" in metal

    assert "layout(std430, binding = 94) buffer ParamForwardElementBlock" in glsl
    assert "} blocks[];" in glsl
    assert "void leaf(ParamForwardElementBlock localBlock, uint i, mat2 value)" in glsl
    assert (
        "void middle(ParamForwardElementBlock localBlock, uint i, mat2 value)" in glsl
    )
    assert "middle(blocks[2], i, value);" in glsl
    assert "leaf(localBlock, i, value);" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_unsupported_explicit_parameters_are_diagnostic():
    crossgl = """
    shader main {
        struct UnsupportedParamBlock {
            double flag;
            float values[];
        };

        UnsupportedParamBlock block @glsl_buffer_block(std430) @binding(95);

        float readSingle(UnsupportedParamBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.values[i];
        }

        float readArray(UnsupportedParamBlock localBlocks[] @glsl_buffer_block(std430), uint i) {
            return localBlocks[0].values[i];
        }

        UnsupportedParamBlock makeBlock() {
            return block;
        }

        compute {
            void main() {
                float value = readSingle(block, 0u);
                UnsupportedParamBlock tmpBlock = makeBlock();
                float fromLocal = tmpBlock.values[0];
            }
        }
    }
    """

    shader_ast = parse_crossgl(dedent(crossgl))
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "unsupported HLSL GLSL buffer block parameter UnsupportedParamBlock "
        "localBlock (std430)" in hlsl
    )
    assert (
        "unsupported HLSL GLSL buffer block parameter UnsupportedParamBlock "
        "localBlocks (std430)" in hlsl
    )
    assert (
        "unsupported member flag: type is not supported by ByteAddressBuffer lowering"
        in hlsl
    )
    assert (
        "unsupported HLSL GLSL buffer block struct UnsupportedParamBlock omitted"
        in hlsl
    )
    assert (
        "unsupported HLSL GLSL buffer block variable UnsupportedParamBlock block omitted"
        in hlsl
    )
    assert "unsupported HLSL GLSL buffer block function readSingle omitted" in hlsl
    assert "unsupported HLSL GLSL buffer block function readArray omitted" in hlsl
    assert "unsupported HLSL GLSL buffer block function makeBlock omitted" in hlsl
    assert "struct UnsupportedParamBlock {" not in hlsl
    assert "UnsupportedParamBlock block;" not in hlsl
    assert "float readSingle(UnsupportedParamBlock localBlock, uint i)" not in hlsl
    assert "float readArray(UnsupportedParamBlock localBlocks[], uint i)" not in hlsl
    assert "UnsupportedParamBlock makeBlock()" not in hlsl
    assert (
        "float value = 0 /* unsupported HLSL GLSL buffer block function call "
        "readSingle: target function omitted */;" in hlsl
    )
    assert (
        "/* unsupported HLSL GLSL buffer block local variable UnsupportedParamBlock "
        "tmpBlock omitted: no target-side fallback declaration emitted */;" in hlsl
    )
    assert (
        "float fromLocal = 0 /* unsupported HLSL GLSL buffer block access tmpBlock: "
        "no target-side fallback declaration emitted */;" in hlsl
    )
    assert "return localBlock.values[i];" not in hlsl
    assert "return localBlocks[0].values[i];" not in hlsl
    assert "UnsupportedParamBlock tmpBlock = makeBlock();" not in hlsl
    assert "tmpBlock.values[0]" not in hlsl
    assert "readSingle(block, 0u)" not in hlsl

    assert (
        "unsupported Metal GLSL buffer block parameter UnsupportedParamBlock "
        "localBlock (std430)" in metal
    )
    assert (
        "unsupported Metal GLSL buffer block parameter UnsupportedParamBlock "
        "localBlocks (std430)" in metal
    )
    assert (
        "unsupported member flag: type is not supported by Metal pointer/offset lowering"
        in metal
    )
    assert (
        "unsupported Metal GLSL buffer block struct UnsupportedParamBlock omitted"
        in metal
    )
    assert (
        "unsupported Metal GLSL buffer block variable UnsupportedParamBlock block omitted"
        in metal
    )
    assert "unsupported Metal GLSL buffer block function readSingle omitted" in metal
    assert "unsupported Metal GLSL buffer block function readArray omitted" in metal
    assert "unsupported Metal GLSL buffer block function makeBlock omitted" in metal
    assert "struct UnsupportedParamBlock {" not in metal
    assert "UnsupportedParamBlock block;" not in metal
    assert "float readSingle(UnsupportedParamBlock localBlock, uint i)" not in metal
    assert "float readArray(UnsupportedParamBlock localBlocks[], uint i)" not in metal
    assert "UnsupportedParamBlock makeBlock()" not in metal
    assert (
        "float value = 0 /* unsupported Metal GLSL buffer block function call "
        "readSingle: target function omitted */;" in metal
    )
    assert (
        "/* unsupported Metal GLSL buffer block local variable UnsupportedParamBlock "
        "tmpBlock omitted: no target-side fallback declaration emitted */;" in metal
    )
    assert (
        "float fromLocal = 0 /* unsupported Metal GLSL buffer block access tmpBlock: "
        "no target-side fallback declaration emitted */;" in metal
    )
    assert "return localBlock.values[i];" not in metal
    assert "return localBlocks[0].values[i];" not in metal
    assert "UnsupportedParamBlock tmpBlock = makeBlock();" not in metal
    assert "tmpBlock.values[0]" not in metal
    assert "readSingle(block, 0u)" not in metal

    assert "layout(std430, binding = 95) buffer UnsupportedParamBlock" in glsl
    assert "} block;" in glsl
    assert "float readSingle(UnsupportedParamBlock localBlock, uint i)" in glsl
    assert "float readArray(UnsupportedParamBlock localBlocks[], uint i)" in glsl
    assert "UnsupportedParamBlock makeBlock()" in glsl
    assert "float value = readSingle(block, 0u);" in glsl
    assert "UnsupportedParamBlock tmpBlock = makeBlock();" in glsl
    assert "float fromLocal = tmpBlock.values[0];" in glsl
    assert "glsl_buffer_block" not in glsl


def test_codegen_mixed_ssbo_unsupported_nested_fallback_keeps_expression_type():
    crossgl = """
    shader main {
        struct UnsupportedVectorBlock {
            double flag;
            vec4 values[];
        };

        struct UnsupportedScalarBlock {
            double flag;
            float values[];
        };

        UnsupportedVectorBlock block @glsl_buffer_block(std430) @binding(96);
        UnsupportedScalarBlock scalarBlock @glsl_buffer_block(std430) @binding(97);

        vec4 readParam(UnsupportedVectorBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.values[i];
        }

        compute {
            void main() {
                bool choose = true;
                float viaAccess = dot(block.values[0], vec4(1.0));
                float viaCall = dot(readParam(block, 0u), vec4(1.0));
                float viaScalar = scalarBlock.values[0];
                vec4 viaVectorAdd = block.values[0] + scalarBlock.values[0];
                vec4 viaFunctionAdd = readParam(block, 0u) + scalarBlock.values[0];
                float viaScalarAdd = scalarBlock.values[0] + 1.0;
                vec4 viaTernaryAccess = choose ? block.values[0] : vec4(1.0);
                vec4 viaTernaryCall = choose ? vec4(1.0) : readParam(block, 0u);
                float viaTernaryScalar = choose ? scalarBlock.values[0] : 1.0;
                float viaTernarySwizzle = choose ? block.values[0].x : scalarBlock.values[0];
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "bool choose = true;" in hlsl
    assert (
        "dot(float4(0) /* unsupported HLSL GLSL buffer block access block: "
        "no target-side fallback declaration emitted */, float4(1.0))" in hlsl
    )
    assert (
        "dot(float4(0) /* unsupported HLSL GLSL buffer block function call "
        "readParam: target function omitted */, float4(1.0))" in hlsl
    )
    assert "dot(0 /* unsupported HLSL GLSL buffer block access block" not in hlsl
    assert (
        "dot(0 /* unsupported HLSL GLSL buffer block function call readParam"
        not in hlsl
    )
    assert (
        "float viaScalar = 0 /* unsupported HLSL GLSL buffer block access scalarBlock: "
        "no target-side fallback declaration emitted */;" in hlsl
    )
    assert (
        "float4 viaVectorAdd = (float4(0) /* unsupported HLSL GLSL buffer block "
        "access block: no target-side fallback declaration emitted */ + 0 "
        "/* unsupported HLSL GLSL buffer block access scalarBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "float4 viaFunctionAdd = (float4(0) /* unsupported HLSL GLSL buffer block "
        "function call readParam: target function omitted */ + 0 "
        "/* unsupported HLSL GLSL buffer block access scalarBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "float viaScalarAdd = (0 /* unsupported HLSL GLSL buffer block access "
        "scalarBlock: no target-side fallback declaration emitted */ + 1.0);" in hlsl
    )
    assert (
        "float4 viaTernaryAccess = (choose ? float4(0) /* unsupported HLSL "
        "GLSL buffer block access block: no target-side fallback declaration "
        "emitted */ : float4(1.0));" in hlsl
    )
    assert (
        "float4 viaTernaryCall = (choose ? float4(1.0) : float4(0) "
        "/* unsupported HLSL GLSL buffer block function call readParam: "
        "target function omitted */);" in hlsl
    )
    assert (
        "float viaTernaryScalar = (choose ? 0 /* unsupported HLSL GLSL buffer "
        "block access scalarBlock: no target-side fallback declaration emitted */ "
        ": 1.0);" in hlsl
    )
    assert (
        "float viaTernarySwizzle = (choose ? 0 /* unsupported HLSL GLSL buffer "
        "block access block: no target-side fallback declaration emitted */ : 0 "
        "/* unsupported HLSL GLSL buffer block access scalarBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )

    assert "bool choose = true;" in metal
    assert (
        "dot(float4(0) /* unsupported Metal GLSL buffer block access block: "
        "no target-side fallback declaration emitted */, float4(1.0))" in metal
    )
    assert (
        "dot(float4(0) /* unsupported Metal GLSL buffer block function call "
        "readParam: target function omitted */, float4(1.0))" in metal
    )
    assert "dot(0 /* unsupported Metal GLSL buffer block access block" not in metal
    assert (
        "dot(0 /* unsupported Metal GLSL buffer block function call readParam"
        not in metal
    )
    assert (
        "float viaScalar = 0 /* unsupported Metal GLSL buffer block access scalarBlock: "
        "no target-side fallback declaration emitted */;" in metal
    )
    assert (
        "float4 viaVectorAdd = float4(0) /* unsupported Metal GLSL buffer block "
        "access block: no target-side fallback declaration emitted */ + 0 "
        "/* unsupported Metal GLSL buffer block access scalarBlock: no target-side "
        "fallback declaration emitted */;" in metal
    )
    assert (
        "float4 viaFunctionAdd = float4(0) /* unsupported Metal GLSL buffer block "
        "function call readParam: target function omitted */ + 0 "
        "/* unsupported Metal GLSL buffer block access scalarBlock: no target-side "
        "fallback declaration emitted */;" in metal
    )
    assert (
        "float viaScalarAdd = 0 /* unsupported Metal GLSL buffer block access "
        "scalarBlock: no target-side fallback declaration emitted */ + 1.0;" in metal
    )
    assert (
        "float4 viaTernaryAccess = choose ? float4(0) /* unsupported Metal "
        "GLSL buffer block access block: no target-side fallback declaration "
        "emitted */ : float4(1.0);" in metal
    )
    assert (
        "float4 viaTernaryCall = choose ? float4(1.0) : float4(0) "
        "/* unsupported Metal GLSL buffer block function call readParam: "
        "target function omitted */;" in metal
    )
    assert (
        "float viaTernaryScalar = choose ? 0 /* unsupported Metal GLSL buffer "
        "block access scalarBlock: no target-side fallback declaration emitted */ "
        ": 1.0;" in metal
    )
    assert (
        "float viaTernarySwizzle = choose ? 0 /* unsupported Metal GLSL buffer "
        "block access block: no target-side fallback declaration emitted */ : 0 "
        "/* unsupported Metal GLSL buffer block access scalarBlock: no target-side "
        "fallback declaration emitted */;" in metal
    )

    assert "bool choose = true;" in glsl
    assert "float viaAccess = dot(block.values[0], vec4(1.0));" in glsl
    assert "float viaCall = dot(readParam(block, 0u), vec4(1.0));" in glsl
    assert "float viaScalar = scalarBlock.values[0];" in glsl
    assert "vec4 viaVectorAdd = (block.values[0] + scalarBlock.values[0]);" in glsl
    assert (
        "vec4 viaFunctionAdd = (readParam(block, 0u) + scalarBlock.values[0]);" in glsl
    )
    assert "float viaScalarAdd = (scalarBlock.values[0] + 1.0);" in glsl
    assert "vec4 viaTernaryAccess = (choose ? block.values[0] : vec4(1.0));" in glsl
    assert "vec4 viaTernaryCall = (choose ? vec4(1.0) : readParam(block, 0u));" in glsl
    assert "float viaTernaryScalar = (choose ? scalarBlock.values[0] : 1.0);" in glsl
    assert (
        "float viaTernarySwizzle = (choose ? block.values[0].x : "
        "scalarBlock.values[0]);" in glsl
    )


def test_codegen_mixed_ssbo_unsupported_bool_conditions_are_boolean_diagnostics():
    crossgl = """
    shader main {
        struct UnsupportedBoolBlock {
            double flag;
            bool flags[];
        };

        UnsupportedBoolBlock boolBlock @glsl_buffer_block(std430) @binding(98);

        bool readFlag(UnsupportedBoolBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.flags[i];
        }

        compute {
            void main() {
                if (boolBlock.flags[0]) {
                    float fromAccess = 1.0;
                }
                if (readFlag(boolBlock, 1u)) {
                    float fromCall = 2.0;
                }
                if (boolBlock.flags[2] && readFlag(boolBlock, 3u)) {
                    float fromLogical = 3.0;
                }
                while (boolBlock.flags[4]) {
                    break;
                }
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "if (false /* unsupported HLSL GLSL buffer block access boolBlock: "
        "no target-side fallback declaration emitted */)" in hlsl
    )
    assert (
        "if (false /* unsupported HLSL GLSL buffer block function call readFlag: "
        "target function omitted */)" in hlsl
    )
    assert (
        "if ((false /* unsupported HLSL GLSL buffer block access boolBlock: "
        "no target-side fallback declaration emitted */ && false /* unsupported "
        "HLSL GLSL buffer block function call readFlag: target function omitted */))"
        in hlsl
    )
    assert (
        "while (false /* unsupported HLSL GLSL buffer block access boolBlock: "
        "no target-side fallback declaration emitted */)" in hlsl
    )
    assert "if (0 /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "while (0 /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "boolBlock.flags" not in hlsl

    assert (
        "if (false /* unsupported Metal GLSL buffer block access boolBlock: "
        "no target-side fallback declaration emitted */)" in metal
    )
    assert (
        "if (false /* unsupported Metal GLSL buffer block function call readFlag: "
        "target function omitted */)" in metal
    )
    assert (
        "if (false /* unsupported Metal GLSL buffer block access boolBlock: "
        "no target-side fallback declaration emitted */ && false /* unsupported "
        "Metal GLSL buffer block function call readFlag: target function omitted */)"
        in metal
    )
    assert (
        "while (false /* unsupported Metal GLSL buffer block access boolBlock: "
        "no target-side fallback declaration emitted */)" in metal
    )
    assert "if (0 /* unsupported Metal GLSL buffer block" not in metal
    assert "while (0 /* unsupported Metal GLSL buffer block" not in metal
    assert "boolBlock.flags" not in metal

    assert "if (boolBlock.flags[0])" in glsl
    assert "if (readFlag(boolBlock, 1u))" in glsl
    assert "if ((boolBlock.flags[2] && readFlag(boolBlock, 3u)))" in glsl
    assert "while (boolBlock.flags[4])" in glsl


def test_codegen_mixed_ssbo_unsupported_integer_bounds_are_typed_diagnostics():
    crossgl = """
    shader main {
        struct UnsupportedIndexBlock {
            double flag;
            uint count;
            uint indices[];
        };

        struct UnsupportedSignedIndexBlock {
            double flag;
            int limit;
            int offsets[];
        };

        UnsupportedIndexBlock indexBlock @glsl_buffer_block(std430) @binding(99);
        UnsupportedSignedIndexBlock signedBlock @glsl_buffer_block(std430) @binding(100);

        uint readIndex(UnsupportedIndexBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.indices[i];
        }

        int readOffset(UnsupportedSignedIndexBlock localBlock @glsl_buffer_block(std430), uint i) {
            return localBlock.offsets[i];
        }

        compute {
            void main() {
                uint directIndex = indexBlock.indices[0];
                uint callIndex = readIndex(indexBlock, 1u);
                int signedIndex = signedBlock.offsets[0];
                int signedCall = readOffset(signedBlock, 1u);
                for (uint i = 0u; i < indexBlock.count; i = i + 1u) {
                    uint loopValue = i;
                }
                for (uint j = 0u; j < readIndex(indexBlock, 2u); j = j + 1u) {
                    uint helperBound = j;
                }
                for (int k = 0; k < signedBlock.limit; k = k + 1) {
                    int signedLoopValue = k;
                }
                for (int m = 0; m < readOffset(signedBlock, 2u); m = m + 1) {
                    int signedHelperBound = m;
                }
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "uint directIndex = 0u /* unsupported HLSL GLSL buffer block access "
        "indexBlock: no target-side fallback declaration emitted */;" in hlsl
    )
    assert (
        "uint callIndex = 0u /* unsupported HLSL GLSL buffer block function call "
        "readIndex: target function omitted */;" in hlsl
    )
    assert (
        "int signedIndex = 0 /* unsupported HLSL GLSL buffer block access "
        "signedBlock: no target-side fallback declaration emitted */;" in hlsl
    )
    assert (
        "int signedCall = 0 /* unsupported HLSL GLSL buffer block function call "
        "readOffset: target function omitted */;" in hlsl
    )
    assert (
        "for (uint i = 0u; (i < 0u /* unsupported HLSL GLSL buffer block "
        "access indexBlock: no target-side fallback declaration emitted */); "
        "i = (i + 1u))" in hlsl
    )
    assert (
        "for (uint j = 0u; (j < 0u /* unsupported HLSL GLSL buffer block "
        "function call readIndex: target function omitted */); j = (j + 1u))" in hlsl
    )
    assert (
        "for (int k = 0; (k < 0 /* unsupported HLSL GLSL buffer block access "
        "signedBlock: no target-side fallback declaration emitted */); "
        "k = (k + 1))" in hlsl
    )
    assert (
        "for (int m = 0; (m < 0 /* unsupported HLSL GLSL buffer block function "
        "call readOffset: target function omitted */); m = (m + 1))" in hlsl
    )
    assert "uint directIndex = 0 /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "uint callIndex = 0 /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "int signedIndex = 0u /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "int signedCall = 0u /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "indexBlock.indices" not in hlsl
    assert "signedBlock.offsets" not in hlsl

    assert (
        "uint directIndex = 0u /* unsupported Metal GLSL buffer block access "
        "indexBlock: no target-side fallback declaration emitted */;" in metal
    )
    assert (
        "uint callIndex = 0u /* unsupported Metal GLSL buffer block function call "
        "readIndex: target function omitted */;" in metal
    )
    assert (
        "int signedIndex = 0 /* unsupported Metal GLSL buffer block access "
        "signedBlock: no target-side fallback declaration emitted */;" in metal
    )
    assert (
        "int signedCall = 0 /* unsupported Metal GLSL buffer block function call "
        "readOffset: target function omitted */;" in metal
    )
    assert (
        "for (uint i = 0u; i < 0u /* unsupported Metal GLSL buffer block access "
        "indexBlock: no target-side fallback declaration emitted */; i = i + 1u)"
        in metal
    )
    assert (
        "for (uint j = 0u; j < 0u /* unsupported Metal GLSL buffer block "
        "function call readIndex: target function omitted */; j = j + 1u)" in metal
    )
    assert (
        "for (int k = 0; k < 0 /* unsupported Metal GLSL buffer block access "
        "signedBlock: no target-side fallback declaration emitted */; k = k + 1)"
        in metal
    )
    assert (
        "for (int m = 0; m < 0 /* unsupported Metal GLSL buffer block function "
        "call readOffset: target function omitted */; m = m + 1)" in metal
    )
    assert "uint directIndex = 0 /* unsupported Metal GLSL buffer block" not in metal
    assert "uint callIndex = 0 /* unsupported Metal GLSL buffer block" not in metal
    assert "int signedIndex = 0u /* unsupported Metal GLSL buffer block" not in metal
    assert "int signedCall = 0u /* unsupported Metal GLSL buffer block" not in metal
    assert "indexBlock.indices" not in metal
    assert "signedBlock.offsets" not in metal

    assert "layout(std430, binding = 99) buffer UnsupportedIndexBlock" in glsl
    assert "layout(std430, binding = 100) buffer UnsupportedSignedIndexBlock" in glsl
    assert "uint directIndex = indexBlock.indices[0];" in glsl
    assert "uint callIndex = readIndex(indexBlock, 1u);" in glsl
    assert "int signedIndex = signedBlock.offsets[0];" in glsl
    assert "int signedCall = readOffset(signedBlock, 1u);" in glsl
    assert "for (uint i = 0u; (i < indexBlock.count); i = (i + 1u))" in glsl
    assert "for (uint j = 0u; (j < readIndex(indexBlock, 2u)); j = (j + 1u))" in glsl
    assert "for (int k = 0; (k < signedBlock.limit); k = (k + 1))" in glsl
    assert "for (int m = 0; (m < readOffset(signedBlock, 2u)); m = (m + 1))" in glsl


def test_codegen_mixed_ssbo_unsupported_resource_indices_are_typed_diagnostics():
    crossgl = """
    shader ResourceFallbacks {
        sampler2D textures[4];
        sampler samplers[4];
        image2D outputImage;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedResourceIndexBlock {
            double flag;
            int layer;
            uint uLayer;
            int x;
            int y;
        };

        UnsupportedResourceIndexBlock resourceBlock @glsl_buffer_block(std430) @binding(101);

        int readLayer(UnsupportedResourceIndexBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.layer;
        }

        uint readULayer(UnsupportedResourceIndexBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uLayer;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 sampledDirect = texture(textures[resourceBlock.layer], samplers[resourceBlock.layer], input.uv);
                vec4 sampledUnsigned = texture(textures[resourceBlock.uLayer], samplers[readULayer(resourceBlock)], input.uv);
                vec4 sampledCall = texture(textures[readLayer(resourceBlock)], samplers[readLayer(resourceBlock)], input.uv);
                vec4 loadedDirect = imageLoad(outputImage, ivec2(resourceBlock.x, resourceBlock.y));
                vec4 loadedCall = imageLoad(outputImage, ivec2(readLayer(resourceBlock), resourceBlock.y));
                return sampledDirect + sampledUnsigned + sampledCall + loadedDirect + loadedCall;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert "RWTexture2D<float4> outputImage : register(u0);" in hlsl
    assert (
        "textures[0 /* unsupported HLSL GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */].Sample(samplers[0 "
        "/* unsupported HLSL GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */], input.uv)" in hlsl
    )
    assert (
        "textures[0u /* unsupported HLSL GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */].Sample(samplers[0u "
        "/* unsupported HLSL GLSL buffer block function call readULayer: "
        "target function omitted */], input.uv)" in hlsl
    )
    assert (
        "textures[0 /* unsupported HLSL GLSL buffer block function call readLayer: "
        "target function omitted */].Sample(samplers[0 /* unsupported HLSL GLSL "
        "buffer block function call readLayer: target function omitted */], input.uv)"
        in hlsl
    )
    assert (
        "outputImage[int2(0 /* unsupported HLSL GLSL buffer block access "
        "resourceBlock: no target-side fallback declaration emitted */, 0 "
        "/* unsupported HLSL GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */)]" in hlsl
    )
    assert (
        "outputImage[int2(0 /* unsupported HLSL GLSL buffer block function call "
        "readLayer: target function omitted */, 0 /* unsupported HLSL GLSL "
        "buffer block access resourceBlock: no target-side fallback declaration "
        "emitted */)]" in hlsl
    )
    assert "resourceBlock.layer" not in hlsl
    assert "resourceBlock.uLayer" not in hlsl
    assert "resourceBlock.x" not in hlsl
    assert "textures[float" not in hlsl
    assert "samplers[float" not in hlsl
    assert "outputImage[float" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert "texture2d<float, access::read_write> outputImage [[texture(4)]]" in metal
    assert (
        "textures[0 /* unsupported Metal GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */].sample(samplers[0 "
        "/* unsupported Metal GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */], input.uv)" in metal
    )
    assert (
        "textures[0u /* unsupported Metal GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */].sample(samplers[0u "
        "/* unsupported Metal GLSL buffer block function call readULayer: "
        "target function omitted */], input.uv)" in metal
    )
    assert (
        "textures[0 /* unsupported Metal GLSL buffer block function call readLayer: "
        "target function omitted */].sample(samplers[0 /* unsupported Metal GLSL "
        "buffer block function call readLayer: target function omitted */], input.uv)"
        in metal
    )
    assert (
        "outputImage.read(uint2(int2(0 /* unsupported Metal GLSL buffer block "
        "access resourceBlock: no target-side fallback declaration emitted */, "
        "0 /* unsupported Metal GLSL buffer block access resourceBlock: "
        "no target-side fallback declaration emitted */)))" in metal
    )
    assert (
        "outputImage.read(uint2(int2(0 /* unsupported Metal GLSL buffer block "
        "function call readLayer: target function omitted */, 0 /* unsupported "
        "Metal GLSL buffer block access resourceBlock: no target-side fallback "
        "declaration emitted */)))" in metal
    )
    assert "resourceBlock.layer" not in metal
    assert "resourceBlock.uLayer" not in metal
    assert "resourceBlock.x" not in metal
    assert "textures[float" not in metal
    assert "samplers[float" not in metal
    assert "outputImage.read(float" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "layout(rgba32f, binding = 0) uniform image2D outputImage;" in glsl
    assert "layout(std430, binding = 101) buffer UnsupportedResourceIndexBlock" in glsl
    assert "vec4 sampledDirect = texture(textures[resourceBlock.layer], uv);" in glsl
    assert "vec4 sampledUnsigned = texture(textures[resourceBlock.uLayer], uv);" in glsl
    assert "vec4 sampledCall = texture(textures[readLayer(resourceBlock)], uv);" in glsl
    assert (
        "vec4 loadedDirect = imageLoad(outputImage, "
        "ivec2(resourceBlock.x, resourceBlock.y));" in glsl
    )
    assert (
        "vec4 loadedCall = imageLoad(outputImage, "
        "ivec2(readLayer(resourceBlock), resourceBlock.y));" in glsl
    )


def test_codegen_mixed_ssbo_resource_array_helpers_infer_fallback_arg_types():
    crossgl = """
    shader ResourceArrayFallbacks {
        sampler2D textures[4];
        sampler samplers[4];
        uimage2D counters @r32ui[4];

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedArrayArgBlock {
            double flag;
            int layer;
            vec2 uv;
            ivec2 pixel;
            uint amount;
        };

        UnsupportedArrayArgBlock arrayBlock @glsl_buffer_block(std430) @binding(109);

        int readLayer(UnsupportedArrayArgBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.layer;
        }

        vec2 readUv(UnsupportedArrayArgBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        ivec2 readPixel(UnsupportedArrayArgBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        uint readAmount(UnsupportedArrayArgBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.amount;
        }

        vec4 sampleArray(sampler2D texs[], sampler sams[], int layer, vec2 uv) {
            return texture(texs[layer], sams[layer], uv);
        }

        uint incrementArray(uimage2D images[] @r32ui, int layer, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[layer], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 sampledDirect = sampleArray(textures, samplers, arrayBlock.layer, arrayBlock.uv);
                vec4 sampledCall = sampleArray(textures, samplers, readLayer(arrayBlock), readUv(arrayBlock));
                uint incrementedDirect = incrementArray(counters, arrayBlock.layer, arrayBlock.pixel, arrayBlock.amount);
                uint incrementedCall = incrementArray(counters, readLayer(arrayBlock), readPixel(arrayBlock), readAmount(arrayBlock));
                return sampledDirect + sampledCall + vec4(float(incrementedDirect + incrementedCall));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert "RWTexture2D<uint> counters[4] : register(u0);" in hlsl
    assert (
        "float4 sampleArray(Texture2D texs[4], SamplerState sams[4], int layer, "
        "float2 uv)" in hlsl
    )
    assert "return texs[layer].Sample(sams[layer], uv);" in hlsl
    assert (
        "uint incrementArray(RWTexture2D<uint> images[4], int layer, int2 pixel, "
        "uint value)" in hlsl
    )
    assert "uint oldValue = images[layer][pixel];" in hlsl
    assert "images[0][pixel] = (oldValue + value);" in hlsl
    assert (
        "float4 sampledDirect = sampleArray(textures, samplers, 0 /* unsupported "
        "HLSL GLSL buffer block access arrayBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported HLSL GLSL buffer "
        "block access arrayBlock: no target-side fallback declaration emitted */);"
        in hlsl
    )
    assert (
        "float4 sampledCall = sampleArray(textures, samplers, 0 /* unsupported "
        "HLSL GLSL buffer block function call readLayer: target function "
        "omitted */, float2(0) /* unsupported HLSL GLSL buffer block function "
        "call readUv: target function omitted */);" in hlsl
    )
    assert (
        "uint incrementedDirect = incrementArray(counters, 0 /* unsupported "
        "HLSL GLSL buffer block access arrayBlock: no target-side fallback "
        "declaration emitted */, int2(0) /* unsupported HLSL GLSL buffer block "
        "access arrayBlock: no target-side fallback declaration emitted */, 0u "
        "/* unsupported HLSL GLSL buffer block access arrayBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "uint incrementedCall = incrementArray(counters, 0 /* unsupported "
        "HLSL GLSL buffer block function call readLayer: target function "
        "omitted */, int2(0) /* unsupported HLSL GLSL buffer block function "
        "call readPixel: target function omitted */, 0u /* unsupported HLSL "
        "GLSL buffer block function call readAmount: target function omitted */);"
        in hlsl
    )
    assert "sampleArray(textures, samplers, 0u /* unsupported HLSL" not in hlsl
    assert (
        "incrementArray(counters, 0 /* unsupported HLSL GLSL buffer block access arrayBlock"
        in hlsl
    )
    assert "arrayBlock.layer" not in hlsl
    assert "arrayBlock.pixel" not in hlsl
    assert "imageLoad(" not in hlsl
    assert "imageStore(" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert (
        "array<texture2d<uint, access::read_write>, 4> counters [[texture(4)]]" in metal
    )
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert (
        "float4 sampleArray(array<texture2d<float>, 4> texs, array<sampler, 4> "
        "sams, int layer, float2 uv)" in metal
    )
    assert "return texs[layer].sample(sams[layer], uv);" in metal
    assert (
        "uint incrementArray(array<texture2d<uint, access::read_write>, 4> "
        "images, int layer, int2 pixel, uint value)" in metal
    )
    assert "uint oldValue = images[layer].read(uint2(pixel)).x;" in metal
    assert "images[0].write(uint4(oldValue + value), uint2(pixel));" in metal
    assert (
        "float4 sampledDirect = sampleArray(textures, samplers, 0 /* unsupported "
        "Metal GLSL buffer block access arrayBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported Metal GLSL buffer "
        "block access arrayBlock: no target-side fallback declaration emitted */);"
        in metal
    )
    assert (
        "float4 sampledCall = sampleArray(textures, samplers, 0 /* unsupported "
        "Metal GLSL buffer block function call readLayer: target function "
        "omitted */, float2(0) /* unsupported Metal GLSL buffer block function "
        "call readUv: target function omitted */);" in metal
    )
    assert (
        "uint incrementedDirect = incrementArray(counters, 0 /* unsupported "
        "Metal GLSL buffer block access arrayBlock: no target-side fallback "
        "declaration emitted */, int2(0) /* unsupported Metal GLSL buffer block "
        "access arrayBlock: no target-side fallback declaration emitted */, 0u "
        "/* unsupported Metal GLSL buffer block access arrayBlock: no target-side "
        "fallback declaration emitted */);" in metal
    )
    assert (
        "uint incrementedCall = incrementArray(counters, 0 /* unsupported "
        "Metal GLSL buffer block function call readLayer: target function "
        "omitted */, int2(0) /* unsupported Metal GLSL buffer block function "
        "call readPixel: target function omitted */, 0u /* unsupported Metal "
        "GLSL buffer block function call readAmount: target function omitted */);"
        in metal
    )
    assert "sampleArray(textures, samplers, 0u /* unsupported Metal" not in metal
    assert "arrayBlock.layer" not in metal
    assert "arrayBlock.pixel" not in metal
    assert "imageLoad(" not in metal
    assert "imageStore(" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "layout(r32ui, binding = 0) uniform uimage2D counters[4];" in glsl
    assert "vec4 sampleArray(sampler2D texs[4], int layer, vec2 uv)" in glsl
    assert "return texture(texs[layer], uv);" in glsl
    assert (
        "uint incrementArray(uimage2D images[4], int layer, ivec2 pixel, uint value)"
        in glsl
    )
    assert "uint oldValue = imageLoad(images[layer], pixel).x;" in glsl
    assert "imageStore(images[0], pixel, uvec4((oldValue + value)));" in glsl
    assert (
        "vec4 sampledDirect = sampleArray(textures, arrayBlock.layer, "
        "arrayBlock.uv);" in glsl
    )
    assert (
        "vec4 sampledCall = sampleArray(textures, readLayer(arrayBlock), "
        "readUv(arrayBlock));" in glsl
    )
    assert (
        "uint incrementedDirect = incrementArray__glsl_images_counters("
        "arrayBlock.layer, arrayBlock.pixel, arrayBlock.amount);" in glsl
    )
    assert (
        "uint incrementedCall = incrementArray__glsl_images_counters("
        "readLayer(arrayBlock), readPixel(arrayBlock), readAmount(arrayBlock));" in glsl
    )
    assert "sampler sams" not in glsl
    assert "samplers" not in glsl


def test_codegen_mixed_ssbo_shadow_resource_array_helpers_infer_fallback_arg_types():
    crossgl = """
    shader ShadowResourceArrayFallbacks {
        sampler2DShadow shadowMaps[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler shadowSamplers[4];

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedShadowArrayBlock {
            double flag;
            int layer;
            vec2 uv;
            vec4 cubeLayer;
            float depth;
        };

        UnsupportedShadowArrayBlock shadowBlock @glsl_buffer_block(std430) @binding(110);

        int readLayer(UnsupportedShadowArrayBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.layer;
        }

        vec2 readUv(UnsupportedShadowArrayBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        vec4 readCubeLayer(UnsupportedShadowArrayBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.cubeLayer;
        }

        float readDepth(UnsupportedShadowArrayBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.depth;
        }

        float shadowLayer(sampler2DShadow maps[], sampler samplers[], int layer, vec2 uv, float depth) {
            float compared = textureCompare(maps[layer], samplers[layer], uv, depth);
            vec4 gathered = textureGatherCompare(maps[1], samplers[1], uv, depth);
            return compared + gathered.x;
        }

        vec4 cubeGatherLayer(samplerCubeArrayShadow maps[], sampler samplers[], int layer, vec4 cubeLayer, float depth) {
            return textureGatherCompare(maps[layer], samplers[layer], cubeLayer, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float direct = shadowLayer(shadowMaps, shadowSamplers, shadowBlock.layer, shadowBlock.uv, shadowBlock.depth);
                float call = shadowLayer(shadowMaps, shadowSamplers, readLayer(shadowBlock), readUv(shadowBlock), readDepth(shadowBlock));
                vec4 cubeDirect = cubeGatherLayer(cubeShadowArrays, shadowSamplers, shadowBlock.layer, shadowBlock.cubeLayer, shadowBlock.depth);
                vec4 cubeCall = cubeGatherLayer(cubeShadowArrays, shadowSamplers, readLayer(shadowBlock), readCubeLayer(shadowBlock), readDepth(shadowBlock));
                return vec4(direct + call) + cubeDirect + cubeCall;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in hlsl
    assert "TextureCubeArray cubeShadowArrays[4] : register(t4);" in hlsl
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in hlsl
    assert (
        "float shadowLayer(Texture2D maps[4], SamplerComparisonState samplers[4], "
        "int layer, float2 uv, float depth)" in hlsl
    )
    assert "float compared = maps[layer].SampleCmp(samplers[layer], uv, depth);" in hlsl
    assert "float4 gathered = maps[1].GatherCmp(samplers[1], uv, depth);" in hlsl
    assert (
        "float4 cubeGatherLayer(TextureCubeArray maps[4], SamplerComparisonState "
        "samplers[4], int layer, float4 cubeLayer, float depth)" in hlsl
    )
    assert "return maps[layer].GatherCmp(samplers[layer], cubeLayer, depth);" in hlsl
    assert (
        "float direct = shadowLayer(shadowMaps, shadowSamplers, 0 /* unsupported "
        "HLSL GLSL buffer block access shadowBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported HLSL GLSL buffer block "
        "access shadowBlock: no target-side fallback declaration emitted */, 0 "
        "/* unsupported HLSL GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "float call = shadowLayer(shadowMaps, shadowSamplers, 0 /* unsupported "
        "HLSL GLSL buffer block function call readLayer: target function omitted */, "
        "float2(0) /* unsupported HLSL GLSL buffer block function call readUv: "
        "target function omitted */, 0 /* unsupported HLSL GLSL buffer block "
        "function call readDepth: target function omitted */);" in hlsl
    )
    assert (
        "float4 cubeDirect = cubeGatherLayer(cubeShadowArrays, shadowSamplers, 0 "
        "/* unsupported HLSL GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */, float4(0) /* unsupported HLSL GLSL "
        "buffer block access shadowBlock: no target-side fallback declaration "
        "emitted */, 0 /* unsupported HLSL GLSL buffer block access shadowBlock: "
        "no target-side fallback declaration emitted */);" in hlsl
    )
    assert (
        "float4 cubeCall = cubeGatherLayer(cubeShadowArrays, shadowSamplers, 0 "
        "/* unsupported HLSL GLSL buffer block function call readLayer: target "
        "function omitted */, float4(0) /* unsupported HLSL GLSL buffer block "
        "function call readCubeLayer: target function omitted */, 0 /* unsupported "
        "HLSL GLSL buffer block function call readDepth: target function omitted */);"
        in hlsl
    )
    assert "shadowLayer(shadowMaps, shadowSamplers, 0u /* unsupported HLSL" not in hlsl
    assert "shadowBlock.layer" not in hlsl
    assert "shadowBlock.cubeLayer" not in hlsl
    assert "textureCompare(" not in hlsl
    assert "textureGatherCompare(" not in hlsl

    assert "array<depth2d<float>, 4> shadowMaps [[texture(0)]]" in metal
    assert "array<depthcube_array<float>, 4> cubeShadowArrays [[texture(4)]]" in metal
    assert "array<sampler, 4> shadowSamplers [[sampler(0)]]" in metal
    assert (
        "float shadowLayer(array<depth2d<float>, 4> maps, array<sampler, 4> "
        "samplers, int layer, float2 uv, float depth)" in metal
    )
    assert (
        "float compared = maps[layer].sample_compare(samplers[layer], uv, depth);"
        in metal
    )
    assert "float4 gathered = maps[1].gather_compare(samplers[1], uv, depth);" in metal
    assert (
        "float4 cubeGatherLayer(array<depthcube_array<float>, 4> maps, "
        "array<sampler, 4> samplers, int layer, float4 cubeLayer, float depth)" in metal
    )
    assert (
        "return maps[layer].gather_compare(samplers[layer], cubeLayer.xyz, "
        "uint(cubeLayer.w), depth);" in metal
    )
    assert (
        "float direct = shadowLayer(shadowMaps, shadowSamplers, 0 /* unsupported "
        "Metal GLSL buffer block access shadowBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported Metal GLSL buffer block "
        "access shadowBlock: no target-side fallback declaration emitted */, 0 "
        "/* unsupported Metal GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */);" in metal
    )
    assert (
        "float call = shadowLayer(shadowMaps, shadowSamplers, 0 /* unsupported "
        "Metal GLSL buffer block function call readLayer: target function omitted */, "
        "float2(0) /* unsupported Metal GLSL buffer block function call readUv: "
        "target function omitted */, 0 /* unsupported Metal GLSL buffer block "
        "function call readDepth: target function omitted */);" in metal
    )
    assert (
        "float4 cubeDirect = cubeGatherLayer(cubeShadowArrays, shadowSamplers, 0 "
        "/* unsupported Metal GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */, float4(0) /* unsupported Metal GLSL "
        "buffer block access shadowBlock: no target-side fallback declaration "
        "emitted */, 0 /* unsupported Metal GLSL buffer block access shadowBlock: "
        "no target-side fallback declaration emitted */);" in metal
    )
    assert (
        "float4 cubeCall = cubeGatherLayer(cubeShadowArrays, shadowSamplers, 0 "
        "/* unsupported Metal GLSL buffer block function call readLayer: target "
        "function omitted */, float4(0) /* unsupported Metal GLSL buffer block "
        "function call readCubeLayer: target function omitted */, 0 /* unsupported "
        "Metal GLSL buffer block function call readDepth: target function omitted */);"
        in metal
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, 0u /* unsupported Metal" not in metal
    )
    assert "shadowBlock.layer" not in metal
    assert "shadowBlock.cubeLayer" not in metal
    assert "textureCompare(" not in metal
    assert "textureGatherCompare(" not in metal

    assert "layout(binding = 0) uniform sampler2DShadow shadowMaps[4];" in glsl
    assert (
        "layout(binding = 4) uniform samplerCubeArrayShadow cubeShadowArrays[4];"
        in glsl
    )
    assert (
        "float shadowLayer(sampler2DShadow maps[4], int layer, vec2 uv, float depth)"
        in glsl
    )
    assert "float compared = texture(maps[layer], vec3(uv, depth));" in glsl
    assert "vec4 gathered = textureGather(maps[1], uv, depth);" in glsl
    assert (
        "vec4 cubeGatherLayer(samplerCubeArrayShadow maps[4], int layer, "
        "vec4 cubeLayer, float depth)" in glsl
    )
    assert "return textureGather(maps[layer], cubeLayer, depth);" in glsl
    assert (
        "float direct = shadowLayer(shadowMaps, shadowBlock.layer, "
        "shadowBlock.uv, shadowBlock.depth);" in glsl
    )
    assert (
        "float call = shadowLayer(shadowMaps, readLayer(shadowBlock), "
        "readUv(shadowBlock), readDepth(shadowBlock));" in glsl
    )
    assert (
        "vec4 cubeDirect = cubeGatherLayer(cubeShadowArrays, shadowBlock.layer, "
        "shadowBlock.cubeLayer, shadowBlock.depth);" in glsl
    )
    assert (
        "vec4 cubeCall = cubeGatherLayer(cubeShadowArrays, readLayer(shadowBlock), "
        "readCubeLayer(shadowBlock), readDepth(shadowBlock));" in glsl
    )
    assert "sampler samplers" not in glsl
    assert "sampler shadowSamplers" not in glsl
    assert "shadowSamplers" not in glsl
    assert "textureCompare(" not in glsl
    assert "textureGatherCompare(" not in glsl


def test_codegen_mixed_ssbo_multisample_resource_arrays_infer_fallback_arg_types():
    crossgl = """
    shader MultisampleArrayFallbacks {
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedMultisampleBlock {
            double flag;
            int layer;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
        };

        UnsupportedMultisampleBlock msBlock @glsl_buffer_block(std430) @binding(111);

        int readLayer(UnsupportedMultisampleBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.layer;
        }

        ivec2 readPixel(UnsupportedMultisampleBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedMultisampleBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        int readSample(UnsupportedMultisampleBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.sampleIndex;
        }

        vec4 fetchSamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer, ivec2 pixel, ivec3 pixelLayer, int sampleIndex) {
            vec4 fetched2D = texelFetch(textures[layer], pixel, sampleIndex);
            vec4 fetchedArray = texelFetch(arrays[layer], pixelLayer, sampleIndex);
            int sampleCount = textureSamples(textures[layer]) + textureSamples(arrays[layer]);
            return fetched2D + fetchedArray + vec4(float(sampleCount));
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 direct = fetchSamples(msTextures, msArrays, msBlock.layer, msBlock.pixel, msBlock.pixelLayer, msBlock.sampleIndex);
                vec4 call = fetchSamples(msTextures, msArrays, readLayer(msBlock), readPixel(msBlock), readPixelLayer(msBlock), readSample(msBlock));
                vec4 inlineFetch = texelFetch(msTextures[msBlock.layer], msBlock.pixel, msBlock.sampleIndex);
                int inlineSamples = textureSamples(msArrays[msBlock.layer]);
                return direct + call + inlineFetch + vec4(float(inlineSamples));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> msTextures[4] : register(t0);" in hlsl
    assert "Texture2DMSArray<float4> msArrays[4] : register(t4);" in hlsl
    assert "int textureSamples(Texture2DMS<float4> tex)" in hlsl
    assert "int textureSamples(Texture2DMSArray<float4> tex)" in hlsl
    assert (
        "float4 fetchSamples(Texture2DMS<float4> textures[4], "
        "Texture2DMSArray<float4> arrays[4], int layer, int2 pixel, "
        "int3 pixelLayer, int sampleIndex)" in hlsl
    )
    assert "float4 fetched2D = textures[layer].Load(pixel, sampleIndex);" in hlsl
    assert "float4 fetchedArray = arrays[layer].Load(pixelLayer, sampleIndex);" in hlsl
    assert (
        "int sampleCount = (textureSamples(textures[layer]) + "
        "textureSamples(arrays[layer]));" in hlsl
    )
    assert (
        "float4 direct = fetchSamples(msTextures, msArrays, 0 /* unsupported "
        "HLSL GLSL buffer block access msBlock: no target-side fallback "
        "declaration emitted */, int2(0) /* unsupported HLSL GLSL buffer block "
        "access msBlock: no target-side fallback declaration emitted */, "
        "int3(0) /* unsupported HLSL GLSL buffer block access msBlock: "
        "no target-side fallback declaration emitted */, 0 /* unsupported HLSL "
        "GLSL buffer block access msBlock: no target-side fallback declaration "
        "emitted */);" in hlsl
    )
    assert (
        "float4 call = fetchSamples(msTextures, msArrays, 0 /* unsupported HLSL "
        "GLSL buffer block function call readLayer: target function omitted */, "
        "int2(0) /* unsupported HLSL GLSL buffer block function call readPixel: "
        "target function omitted */, int3(0) /* unsupported HLSL GLSL buffer "
        "block function call readPixelLayer: target function omitted */, 0 "
        "/* unsupported HLSL GLSL buffer block function call readSample: target "
        "function omitted */);" in hlsl
    )
    assert (
        "float4 inlineFetch = msTextures[0 /* unsupported HLSL GLSL buffer block "
        "access msBlock: no target-side fallback declaration emitted */].Load("
        "int2(0) /* unsupported HLSL GLSL buffer block access msBlock: "
        "no target-side fallback declaration emitted */, 0 /* unsupported HLSL "
        "GLSL buffer block access msBlock: no target-side fallback declaration "
        "emitted */);" in hlsl
    )
    assert (
        "int inlineSamples = textureSamples(msArrays[0 /* unsupported HLSL GLSL "
        "buffer block access msBlock: no target-side fallback declaration emitted */]);"
        in hlsl
    )
    assert ".Load(0 /* unsupported HLSL GLSL buffer block" not in hlsl
    assert ".Load(int2(0) /* unsupported HLSL GLSL buffer block access msBlock" in hlsl
    assert ".Load(int3(0) /* unsupported HLSL GLSL buffer block" not in hlsl
    assert "msBlock.layer" not in hlsl
    assert "msBlock.pixel" not in hlsl
    assert "msBlock.pixelLayer" not in hlsl
    assert "msBlock.sampleIndex" not in hlsl

    assert "array<texture2d_ms<float>, 4> msTextures [[texture(0)]]" in metal
    assert "array<texture2d_ms_array<float>, 4> msArrays [[texture(4)]]" in metal
    assert (
        "float4 fetchSamples(array<texture2d_ms<float>, 4> textures, "
        "array<texture2d_ms_array<float>, 4> arrays, int layer, int2 pixel, "
        "int3 pixelLayer, int sampleIndex)" in metal
    )
    assert "float4 fetched2D = textures[layer].read(pixel, uint(sampleIndex));" in metal
    assert (
        "float4 fetchedArray = arrays[layer].read(pixelLayer.xy, "
        "uint(pixelLayer.z), uint(sampleIndex));" in metal
    )
    assert (
        "int sampleCount = int(textures[layer].get_num_samples()) + "
        "int(arrays[layer].get_num_samples());" in metal
    )
    assert (
        "float4 direct = fetchSamples(msTextures, msArrays, 0 /* unsupported "
        "Metal GLSL buffer block access msBlock: no target-side fallback "
        "declaration emitted */, int2(0) /* unsupported Metal GLSL buffer block "
        "access msBlock: no target-side fallback declaration emitted */, "
        "int3(0) /* unsupported Metal GLSL buffer block access msBlock: "
        "no target-side fallback declaration emitted */, 0 /* unsupported Metal "
        "GLSL buffer block access msBlock: no target-side fallback declaration "
        "emitted */);" in metal
    )
    assert (
        "float4 call = fetchSamples(msTextures, msArrays, 0 /* unsupported Metal "
        "GLSL buffer block function call readLayer: target function omitted */, "
        "int2(0) /* unsupported Metal GLSL buffer block function call readPixel: "
        "target function omitted */, int3(0) /* unsupported Metal GLSL buffer "
        "block function call readPixelLayer: target function omitted */, 0 "
        "/* unsupported Metal GLSL buffer block function call readSample: target "
        "function omitted */);" in metal
    )
    assert (
        "float4 inlineFetch = msTextures[0 /* unsupported Metal GLSL buffer block "
        "access msBlock: no target-side fallback declaration emitted */].read("
        "int2(0) /* unsupported Metal GLSL buffer block access msBlock: "
        "no target-side fallback declaration emitted */, uint(0 /* unsupported "
        "Metal GLSL buffer block access msBlock: no target-side fallback "
        "declaration emitted */));" in metal
    )
    assert (
        "int inlineSamples = int(msArrays[0 /* unsupported Metal GLSL buffer "
        "block access msBlock: no target-side fallback declaration emitted */]"
        ".get_num_samples());" in metal
    )
    assert ".read(0 /* unsupported Metal GLSL buffer block" not in metal
    assert "uint(int2(0)" not in metal
    assert "msBlock.layer" not in metal
    assert "msBlock.pixel" not in metal
    assert "msBlock.pixelLayer" not in metal
    assert "msBlock.sampleIndex" not in metal

    assert "layout(binding = 0) uniform sampler2DMS msTextures[4];" in glsl
    assert "layout(binding = 4) uniform sampler2DMSArray msArrays[4];" in glsl
    assert (
        "vec4 fetchSamples(sampler2DMS textures[4], sampler2DMSArray arrays[4], "
        "int layer, ivec2 pixel, ivec3 pixelLayer, int sampleIndex)" in glsl
    )
    assert "vec4 fetched2D = texelFetch(textures[layer], pixel, sampleIndex);" in glsl
    assert (
        "vec4 fetchedArray = texelFetch(arrays[layer], pixelLayer, sampleIndex);"
        in glsl
    )
    assert (
        "int sampleCount = (textureSamples(textures[layer]) + "
        "textureSamples(arrays[layer]));" in glsl
    )
    assert (
        "vec4 direct = fetchSamples(msTextures, msArrays, msBlock.layer, "
        "msBlock.pixel, msBlock.pixelLayer, msBlock.sampleIndex);" in glsl
    )
    assert (
        "vec4 call = fetchSamples(msTextures, msArrays, readLayer(msBlock), "
        "readPixel(msBlock), readPixelLayer(msBlock), readSample(msBlock));" in glsl
    )
    assert (
        "vec4 inlineFetch = texelFetch(msTextures[msBlock.layer], msBlock.pixel, "
        "msBlock.sampleIndex);" in glsl
    )
    assert "int inlineSamples = textureSamples(msArrays[msBlock.layer]);" in glsl


def test_codegen_mixed_ssbo_multisample_diagnostics_preserve_fallback_arg_types():
    crossgl = """
    shader MultisampleDiagnosticFallbacks {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedMultisampleDiagnosticBlock {
            double flag;
            vec2 uv;
            vec3 uvLayer;
            vec2 ddx;
            vec2 ddy;
            ivec2 pixel;
            ivec3 pixelLayer;
            ivec2 offset;
            int sampleIndex;
            float depth;
        };

        UnsupportedMultisampleDiagnosticBlock msDiag @glsl_buffer_block(std430) @binding(112);

        vec2 readUv(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        vec3 readUvLayer(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uvLayer;
        }

        ivec2 readPixel(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        ivec2 readOffset(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.offset;
        }

        int readSample(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.sampleIndex;
        }

        float readDepth(UnsupportedMultisampleDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.depth;
        }

        vec4 invalid2D(sampler2DMS tex, vec2 uv, vec2 ddx, vec2 ddy, ivec2 pixel, int sampleIndex, ivec2 offset) {
            vec4 sampled = texture(tex, uv);
            vec4 grad = textureGrad(tex, uv, ddx, ddy);
            vec2 lod = textureQueryLod(tex, uv);
            vec4 offsetFetch = texelFetchOffset(tex, pixel, sampleIndex, offset);
            return sampled + grad + offsetFetch + vec4(lod, 0.0, 1.0);
        }

        vec4 invalidArray(sampler2DMSArray tex, vec3 uvLayer, ivec3 pixelLayer, int sampleIndex, ivec2 offset) {
            vec4 sampled = texture(tex, uvLayer);
            vec2 lod = textureQueryLod(tex, uvLayer);
            vec4 offsetFetch = texelFetchOffset(tex, pixelLayer, sampleIndex, offset);
            return sampled + offsetFetch + vec4(lod, 0.0, 1.0);
        }

        float invalidCompare(sampler2DMS tex, sampler s, vec2 uv, float depth, ivec2 offset) {
            return textureCompare(tex, s, uv, depth) + textureCompareOffset(tex, s, uv, depth, offset);
        }

        vec4 invalidGatherArray(sampler2DMSArray tex, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(tex, s, uvLayer, depth) + textureGatherCompareOffset(tex, s, uvLayer, depth, offset);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 direct2D = invalid2D(msTex, msDiag.uv, msDiag.ddx, msDiag.ddy, msDiag.pixel, msDiag.sampleIndex, msDiag.offset);
                vec4 call2D = invalid2D(msTex, readUv(msDiag), readUv(msDiag), readUv(msDiag), readPixel(msDiag), readSample(msDiag), readOffset(msDiag));
                vec4 directArray = invalidArray(msArray, msDiag.uvLayer, msDiag.pixelLayer, msDiag.sampleIndex, msDiag.offset);
                vec4 callArray = invalidArray(msArray, readUvLayer(msDiag), readPixelLayer(msDiag), readSample(msDiag), readOffset(msDiag));
                float compareValue = invalidCompare(msTex, linearSampler, msDiag.uv, msDiag.depth, msDiag.offset);
                vec4 gatherValue = invalidGatherArray(msArray, linearSampler, readUvLayer(msDiag), readDepth(msDiag), readOffset(msDiag));
                vec4 inlineDiag = texture(msTex, msDiag.uv) + texelFetchOffset(msArray, msDiag.pixelLayer, msDiag.sampleIndex, msDiag.offset);
                return direct2D + call2D + directArray + callArray + vec4(compareValue) + gatherValue + inlineDiag;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> msTex : register(t0);" in hlsl
    assert "Texture2DMSArray<float4> msArray : register(t1);" in hlsl
    assert "SamplerState linearSampler : register(s0);" in hlsl
    assert "SamplerComparisonState linearSampler" not in hlsl
    assert (
        "float4 invalid2D(Texture2DMS<float4> tex, float2 uv, float2 ddx, "
        "float2 ddy, int2 pixel, int sampleIndex, int2 offset)" in hlsl
    )
    assert (
        "float4 invalidArray(Texture2DMSArray<float4> tex, float3 uvLayer, "
        "int3 pixelLayer, int sampleIndex, int2 offset)" in hlsl
    )
    assert (
        "float invalidCompare(Texture2DMS<float4> tex, SamplerState s, "
        "float2 uv, float depth, int2 offset)" in hlsl
    )
    assert (
        "float4 invalidGatherArray(Texture2DMSArray<float4> tex, SamplerState s, "
        "float3 uvLayer, float depth, int2 offset)" in hlsl
    )
    assert (
        "unsupported DirectX multisample texture call: texture on "
        "Texture2DMS<float4> */ float4(0.0)" in hlsl
    )
    assert (
        "unsupported DirectX multisample texture call: textureGrad on "
        "Texture2DMS<float4> */ float4(0.0)" in hlsl
    )
    assert (
        "unsupported DirectX multisample texture query: textureQueryLod on "
        "Texture2DMS<float4> */ float2(0.0)" in hlsl
    )
    assert (
        "unsupported DirectX multisample texture query: textureQueryLod on "
        "Texture2DMSArray<float4> */ float2(0.0)" in hlsl
    )
    assert (
        "unsupported DirectX multisample texture comparison: textureCompare on "
        "Texture2DMS<float4> */ 0.0" in hlsl
    )
    assert (
        "unsupported DirectX multisample texture gather comparison: "
        "textureGatherCompare on Texture2DMSArray<float4> */ float4(0.0)" in hlsl
    )
    assert (
        hlsl.count(
            "unsupported DirectX texel fetch offset: multisample textures do not support offsets"
        )
        == 3
    )
    assert (
        "float4 direct2D = invalid2D(msTex, float2(0) /* unsupported HLSL "
        "GLSL buffer block access msDiag" in hlsl
    )
    assert (
        "float4 call2D = invalid2D(msTex, float2(0) /* unsupported HLSL GLSL "
        "buffer block function call readUv" in hlsl
    )
    assert (
        "float4 directArray = invalidArray(msArray, float3(0) /* unsupported "
        "HLSL GLSL buffer block access msDiag" in hlsl
    )
    assert (
        "float4 callArray = invalidArray(msArray, float3(0) /* unsupported "
        "HLSL GLSL buffer block function call readUvLayer" in hlsl
    )
    assert (
        "int3(0) /* unsupported HLSL GLSL buffer block function call "
        "readPixelLayer: target function omitted */" in hlsl
    )
    assert (
        "float compareValue = invalidCompare(msTex, linearSampler, float2(0) "
        "/* unsupported HLSL GLSL buffer block access msDiag" in hlsl
    )
    assert "texture(msTex," not in hlsl
    assert "texelFetchOffset(" not in hlsl
    assert "msDiag.uv" not in hlsl
    assert "msDiag.pixelLayer" not in hlsl

    assert "texture2d_ms<float> msTex [[texture(0)]]" in metal
    assert "texture2d_ms_array<float> msArray [[texture(1)]]" in metal
    assert "sampler linearSampler [[sampler(0)]]" in metal
    assert (
        "float4 invalid2D(texture2d_ms<float> tex, float2 uv, float2 ddx, "
        "float2 ddy, int2 pixel, int sampleIndex, int2 offset)" in metal
    )
    assert (
        "float4 invalidArray(texture2d_ms_array<float> tex, float3 uvLayer, "
        "int3 pixelLayer, int sampleIndex, int2 offset)" in metal
    )
    assert (
        "float invalidCompare(texture2d_ms<float> tex, sampler s, float2 uv, "
        "float depth, int2 offset)" in metal
    )
    assert (
        "float4 invalidGatherArray(texture2d_ms_array<float> tex, sampler s, "
        "float3 uvLayer, float depth, int2 offset)" in metal
    )
    assert (
        "unsupported Metal multisample texture call: texture on "
        "texture2d_ms<float> */ float4(0.0)" in metal
    )
    assert (
        "unsupported Metal multisample texture call: textureGrad on "
        "texture2d_ms<float> */ float4(0.0)" in metal
    )
    assert (
        "unsupported Metal multisample texture query: textureQueryLod on "
        "texture2d_ms<float> */ float2(0.0)" in metal
    )
    assert (
        "unsupported Metal multisample texture query: textureQueryLod on "
        "texture2d_ms_array<float> */ float2(0.0)" in metal
    )
    assert (
        "unsupported Metal multisample texture comparison: textureCompare on "
        "texture2d_ms<float> */ 0.0" in metal
    )
    assert (
        "unsupported Metal multisample texture gather comparison: "
        "textureGatherCompare on texture2d_ms_array<float> */ float4(0.0)" in metal
    )
    assert (
        metal.count(
            "unsupported Metal texel fetch offset: multisample textures do not support offsets"
        )
        == 3
    )
    assert (
        "float4 direct2D = invalid2D(msTex, float2(0) /* unsupported Metal "
        "GLSL buffer block access msDiag" in metal
    )
    assert (
        "float4 call2D = invalid2D(msTex, float2(0) /* unsupported Metal GLSL "
        "buffer block function call readUv" in metal
    )
    assert (
        "float4 directArray = invalidArray(msArray, float3(0) /* unsupported "
        "Metal GLSL buffer block access msDiag" in metal
    )
    assert (
        "float4 callArray = invalidArray(msArray, float3(0) /* unsupported "
        "Metal GLSL buffer block function call readUvLayer" in metal
    )
    assert (
        "int3(0) /* unsupported Metal GLSL buffer block function call "
        "readPixelLayer: target function omitted */" in metal
    )
    assert (
        "float compareValue = invalidCompare(msTex, linearSampler, float2(0) "
        "/* unsupported Metal GLSL buffer block access msDiag" in metal
    )
    assert "texture(msTex," not in metal
    assert "texelFetchOffset(" not in metal
    assert "msDiag.uv" not in metal
    assert "msDiag.pixelLayer" not in metal

    assert "layout(binding = 0) uniform sampler2DMS msTex;" in glsl
    assert "layout(binding = 1) uniform sampler2DMSArray msArray;" in glsl
    assert (
        "layout(std430, binding = 112) buffer UnsupportedMultisampleDiagnosticBlock"
        in glsl
    )
    assert (
        "float invalidCompare(sampler2DMS tex, vec2 uv, float depth, ivec2 offset)"
        in glsl
    )
    assert (
        "vec4 invalidGatherArray(sampler2DMSArray tex, vec3 uvLayer, float depth, "
        "ivec2 offset)" in glsl
    )
    assert "linearSampler" not in glsl
    assert (
        "unsupported GLSL multisample texture call: texture on sampler2DMS */ "
        "vec4(0.0)" in glsl
    )
    assert (
        "unsupported GLSL multisample texture query: textureQueryLod on "
        "sampler2DMSArray */ vec2(0.0)" in glsl
    )
    assert (
        "unsupported GLSL multisample texture comparison: textureCompare on "
        "sampler2DMS */ 0.0" in glsl
    )
    assert (
        "unsupported GLSL multisample texture gather comparison: "
        "textureGatherCompare on sampler2DMSArray */ vec4(0.0)" in glsl
    )
    assert glsl.count("unsupported GLSL texel fetch offset: multisample texture ") == 3
    assert (
        "vec4 direct2D = invalid2D(msTex, msDiag.uv, msDiag.ddx, msDiag.ddy, "
        "msDiag.pixel, msDiag.sampleIndex, msDiag.offset);" in glsl
    )
    assert (
        "vec4 call2D = invalid2D(msTex, readUv(msDiag), readUv(msDiag), "
        "readUv(msDiag), readPixel(msDiag), readSample(msDiag), "
        "readOffset(msDiag));" in glsl
    )
    assert (
        "float compareValue = invalidCompare(msTex, msDiag.uv, msDiag.depth, "
        "msDiag.offset);" in glsl
    )
    assert (
        "vec4 gatherValue = invalidGatherArray(msArray, readUvLayer(msDiag), "
        "readDepth(msDiag), readOffset(msDiag));" in glsl
    )
    assert "texture(msTex, msDiag.uv)" not in glsl
    assert "texelFetchOffset(msArray, msDiag.pixelLayer" not in glsl


def test_codegen_mixed_ssbo_multisample_image_args_infer_fallback_types():
    crossgl = """
    shader MultisampleImageFallbacks {
        image2DMS msColor @rgba16f;
        image2DMSArray msLayers @rgba8;
        uimage2DMS counters @r32ui;
        iimage2DMSArray signedLayers @r32i;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedMsImageBlock {
            double flag;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
            vec4 color;
            uint count;
            int signedValue;
        };

        UnsupportedMsImageBlock msImageBlock @glsl_buffer_block(std430) @binding(113);

        ivec2 readPixel(UnsupportedMsImageBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedMsImageBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        int readSample(UnsupportedMsImageBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.sampleIndex;
        }

        vec4 readColor(UnsupportedMsImageBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.color;
        }

        uint readCount(UnsupportedMsImageBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.count;
        }

        int readSigned(UnsupportedMsImageBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.signedValue;
        }

        vec4 touchImages(image2DMS colorImage @rgba16f, image2DMSArray layerImage @rgba8, uimage2DMS countImage @r32ui, iimage2DMSArray signedImage @r32i, ivec2 pixel, ivec3 pixelLayer, int sampleIndex, vec4 colorValue, uint countValue, int signedValue) {
            vec4 color = imageLoad(colorImage, pixel, sampleIndex);
            vec4 layer = imageLoad(layerImage, pixelLayer, sampleIndex);
            uint oldCount = imageLoad(countImage, pixel, sampleIndex);
            int oldSigned = imageLoad(signedImage, pixelLayer, sampleIndex);
            imageStore(colorImage, pixel, sampleIndex, colorValue + color);
            imageStore(layerImage, pixelLayer, sampleIndex, layer + colorValue);
            imageStore(countImage, pixel, sampleIndex, oldCount + countValue);
            imageStore(signedImage, pixelLayer, sampleIndex, oldSigned + signedValue);
            int sampleCount = imageSamples(colorImage) + imageSamples(layerImage) + imageSamples(countImage) + imageSamples(signedImage);
            return color + layer + vec4(float(oldCount + uint(oldSigned) + uint(sampleCount)));
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 direct = touchImages(msColor, msLayers, counters, signedLayers, msImageBlock.pixel, msImageBlock.pixelLayer, msImageBlock.sampleIndex, msImageBlock.color, msImageBlock.count, msImageBlock.signedValue);
                vec4 call = touchImages(msColor, msLayers, counters, signedLayers, readPixel(msImageBlock), readPixelLayer(msImageBlock), readSample(msImageBlock), readColor(msImageBlock), readCount(msImageBlock), readSigned(msImageBlock));
                vec4 inlineColor = imageLoad(msColor, msImageBlock.pixel, msImageBlock.sampleIndex);
                imageStore(counters, readPixel(msImageBlock), readSample(msImageBlock), readCount(msImageBlock));
                return direct + call + inlineColor;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> msColor : register(t0);" in hlsl
    assert "Texture2DMSArray<float4> msLayers : register(t1);" in hlsl
    assert "Texture2DMS<uint> counters : register(t2);" in hlsl
    assert "Texture2DMSArray<int> signedLayers : register(t3);" in hlsl
    assert "int textureSamples(Texture2DMS<uint> tex)" in hlsl
    assert "int textureSamples(Texture2DMSArray<int> tex)" in hlsl
    assert (
        "float4 touchImages(Texture2DMS<float4> colorImage, "
        "Texture2DMSArray<float4> layerImage, Texture2DMS<uint> "
        "countImage, Texture2DMSArray<int> signedImage, int2 pixel, "
        "int3 pixelLayer, int sampleIndex, float4 colorValue, uint "
        "countValue, int signedValue)" in hlsl
    )
    assert "float4 color = colorImage.Load(pixel, sampleIndex);" in hlsl
    assert "float4 layer = layerImage.Load(pixelLayer, sampleIndex);" in hlsl
    assert "uint oldCount = countImage.Load(pixel, sampleIndex);" in hlsl
    assert "int oldSigned = signedImage.Load(pixelLayer, sampleIndex);" in hlsl
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<float4>" in hlsl
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMSArray<float4>" in hlsl
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<uint>" in hlsl
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMSArray<int>" in hlsl
    )
    assert (
        "int sampleCount = (((textureSamples(colorImage) + "
        "textureSamples(layerImage)) + textureSamples(countImage)) + "
        "textureSamples(signedImage));" in hlsl
    )
    assert (
        "float4 direct = touchImages(msColor, msLayers, counters, signedLayers, "
        "int2(0) /* unsupported HLSL GLSL buffer block access msImageBlock" in hlsl
    )
    assert (
        "float4 call = touchImages(msColor, msLayers, counters, signedLayers, "
        "int2(0) /* unsupported HLSL GLSL buffer block function call readPixel" in hlsl
    )
    assert (
        "float4 inlineColor = msColor.Load(int2(0) /* unsupported HLSL GLSL "
        "buffer block access msImageBlock: no target-side fallback "
        "declaration emitted */, 0 /* unsupported HLSL GLSL buffer block "
        "access msImageBlock: no target-side fallback declaration emitted */);" in hlsl
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<uint>" in hlsl
    )
    assert (
        "float4(0) /* unsupported HLSL GLSL buffer block function call readColor"
        in hlsl
    )
    assert "0u /* unsupported HLSL GLSL buffer block function call readCount" in hlsl
    assert "imageLoad(" not in hlsl
    assert "imageStore(" not in hlsl
    assert "msImageBlock.pixel" not in hlsl
    assert "msImageBlock.sampleIndex" not in hlsl

    assert (
        "float4 touchImages(texture2d_ms<float, access::read> "
        "colorImage, texture2d_ms_array<float, access::read> "
        "layerImage, texture2d_ms<uint, access::read> countImage, "
        "texture2d_ms_array<int, access::read> signedImage, int2 "
        "pixel, int3 pixelLayer, int sampleIndex, float4 colorValue, uint "
        "countValue, int signedValue)" in metal
    )
    assert "float4 color = colorImage.read(uint2(pixel), uint(sampleIndex));" in metal
    assert (
        "float4 layer = layerImage.read(uint2(pixelLayer.xy), "
        "uint(pixelLayer.z), uint(sampleIndex));" in metal
    )
    assert (
        "uint oldCount = countImage.read(uint2(pixel), uint(sampleIndex)).x;" in metal
    )
    assert (
        "int oldSigned = signedImage.read(uint2(pixelLayer.xy), "
        "uint(pixelLayer.z), uint(sampleIndex)).x;" in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms<float, access::read>" in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms_array<float, access::read>" in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms<uint, access::read>" in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms_array<int, access::read>" in metal
    )
    assert (
        "colorImage.write(colorValue + color, uint2(pixel), uint(sampleIndex));"
        not in metal
    )
    assert (
        "layerImage.write(layer + colorValue, uint2(pixelLayer.xy), "
        "uint(pixelLayer.z), uint(sampleIndex));" not in metal
    )
    assert (
        "countImage.write(uint4(oldCount + countValue), uint2(pixel), uint(sampleIndex));"
        not in metal
    )
    assert (
        "signedImage.write(int4(oldSigned + signedValue), "
        "uint2(pixelLayer.xy), uint(pixelLayer.z), uint(sampleIndex));" not in metal
    )
    assert (
        "int sampleCount = int(colorImage.get_num_samples()) + "
        "int(layerImage.get_num_samples()) + int(countImage.get_num_samples()) "
        "+ int(signedImage.get_num_samples());" in metal
    )
    assert (
        "fragment float4 fragment_main(VSOutput input [[stage_in]], "
        "texture2d_ms<float, access::read> msColor [[texture(0)]], "
        "texture2d_ms_array<float, access::read> msLayers [[texture(1)]], "
        "texture2d_ms<uint, access::read> counters [[texture(2)]], "
        "texture2d_ms_array<int, access::read> signedLayers [[texture(3)]])" in metal
    )
    assert (
        "float4 direct = touchImages(msColor, msLayers, counters, signedLayers, "
        "int2(0) /* unsupported Metal GLSL buffer block access msImageBlock" in metal
    )
    assert (
        "float4 call = touchImages(msColor, msLayers, counters, signedLayers, "
        "int2(0) /* unsupported Metal GLSL buffer block function call readPixel"
        in metal
    )
    assert (
        "float4 inlineColor = msColor.read(uint2(int2(0) /* unsupported Metal "
        "GLSL buffer block access msImageBlock: no target-side fallback "
        "declaration emitted */), uint(0 /* unsupported Metal GLSL buffer "
        "block access msImageBlock: no target-side fallback declaration "
        "emitted */));" in metal
    )
    assert (
        "counters.write(uint4(0u /* unsupported Metal GLSL buffer block "
        "function call readCount: target function omitted */), uint2(int2(0) "
        "/* unsupported Metal GLSL buffer block function call readPixel: target "
        "function omitted */), uint(0 /* unsupported Metal GLSL buffer block "
        "function call readSample: target function omitted */));" not in metal
    )
    assert "imageLoad(" not in metal
    assert "imageStore(" not in metal
    assert "msImageBlock.pixel" not in metal
    assert "msImageBlock.sampleIndex" not in metal

    assert "layout(rgba16f, binding = 0) uniform image2DMS msColor;" in glsl
    assert "layout(rgba8, binding = 1) uniform image2DMSArray msLayers;" in glsl
    assert "layout(r32ui, binding = 2) uniform uimage2DMS counters;" in glsl
    assert "layout(r32i, binding = 3) uniform iimage2DMSArray signedLayers;" in glsl
    assert "layout(std430, binding = 113) buffer UnsupportedMsImageBlock" in glsl
    assert (
        "vec4 touchImages(image2DMS colorImage, image2DMSArray layerImage, "
        "uimage2DMS countImage, iimage2DMSArray signedImage, ivec2 pixel, "
        "ivec3 pixelLayer, int sampleIndex, vec4 colorValue, uint countValue, "
        "int signedValue)" in glsl
    )
    assert "vec4 color = imageLoad(colorImage, pixel, sampleIndex);" in glsl
    assert "vec4 layer = imageLoad(layerImage, pixelLayer, sampleIndex);" in glsl
    assert "uint oldCount = imageLoad(countImage, pixel, sampleIndex).x;" in glsl
    assert "int oldSigned = imageLoad(signedImage, pixelLayer, sampleIndex).x;" in glsl
    assert "imageStore(colorImage, pixel, sampleIndex, (colorValue + color));" in glsl
    assert (
        "imageStore(layerImage, pixelLayer, sampleIndex, (layer + colorValue));" in glsl
    )
    assert (
        "imageStore(countImage, pixel, sampleIndex, uvec4((oldCount + "
        "countValue)));" in glsl
    )
    assert (
        "imageStore(signedImage, pixelLayer, sampleIndex, ivec4((oldSigned + "
        "signedValue)));" in glsl
    )
    assert (
        "int sampleCount = (((textureSamples(colorImage) + "
        "textureSamples(layerImage)) + textureSamples(countImage)) + "
        "textureSamples(signedImage));" in glsl
    )
    assert (
        "vec4 direct = touchImages__glsl_colorImage_msColor_layerImage_msLayers_"
        "countImage_counters_signedImage_signedLayers(msImageBlock.pixel, "
        "msImageBlock.pixelLayer, msImageBlock.sampleIndex, msImageBlock.color, "
        "msImageBlock.count, msImageBlock.signedValue);" in glsl
    )
    assert (
        "vec4 call = touchImages__glsl_colorImage_msColor_layerImage_msLayers_"
        "countImage_counters_signedImage_signedLayers(readPixel(msImageBlock), "
        "readPixelLayer(msImageBlock), readSample(msImageBlock), "
        "readColor(msImageBlock), readCount(msImageBlock), "
        "readSigned(msImageBlock));" in glsl
    )
    assert (
        "vec4 inlineColor = imageLoad(msColor, msImageBlock.pixel, msImageBlock.sampleIndex);"
        in glsl
    )
    assert (
        "imageStore(counters, readPixel(msImageBlock), readSample(msImageBlock), "
        "uvec4(readCount(msImageBlock)));" in glsl
    )
    assert "imageSamples(" not in glsl


def test_codegen_multisample_image_query_helpers_cover_vector_integer_formats():
    crossgl = """
    shader MultisampleImageQueryFormats {
        iimage2DMS rgbaI @rgba32i;
        uimage2DMSArray rgbaU @rgba32ui;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                ivec2 intSize = imageSize(rgbaI);
                ivec3 uintArraySize = imageSize(rgbaU);
                int samples = imageSamples(rgbaI) + imageSamples(rgbaU);
                return vec4(float(intSize.x + uintArraySize.z + samples));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<int4> rgbaI : register(t0);" in hlsl
    assert "Texture2DMSArray<uint4> rgbaU : register(t1);" in hlsl
    assert "int2 imageSize(Texture2DMS<int4> image)" in hlsl
    assert "int3 imageSize(Texture2DMSArray<uint4> image)" in hlsl
    assert "image.GetDimensions(width, height, samples);" in hlsl
    assert "image.GetDimensions(width, height, elements, samples);" in hlsl
    assert "int textureSamples(Texture2DMS<int4> tex)" in hlsl
    assert "int textureSamples(Texture2DMSArray<uint4> tex)" in hlsl
    assert "int2 intSize = imageSize(rgbaI);" in hlsl
    assert "int3 uintArraySize = imageSize(rgbaU);" in hlsl
    assert "int samples = (textureSamples(rgbaI) + textureSamples(rgbaU));" in hlsl
    assert "unsupported DirectX texture samples query" not in hlsl

    assert "texture2d_ms<int, access::read> rgbaI [[texture(0)]]" in metal
    assert "texture2d_ms_array<uint, access::read> rgbaU [[texture(1)]]" in metal
    assert "int2 intSize = int2(rgbaI.get_width(), rgbaI.get_height());" in metal
    assert (
        "int3 uintArraySize = int3(rgbaU.get_width(), rgbaU.get_height(), "
        "rgbaU.get_array_size());" in metal
    )
    assert (
        "int samples = int(rgbaI.get_num_samples()) + "
        "int(rgbaU.get_num_samples());" in metal
    )

    assert "layout(rgba32i, binding = 0) uniform iimage2DMS rgbaI;" in glsl
    assert "layout(rgba32ui, binding = 1) uniform uimage2DMSArray rgbaU;" in glsl
    assert "ivec2 intSize = imageSize(rgbaI);" in glsl
    assert "ivec3 uintArraySize = imageSize(rgbaU);" in glsl
    assert "int samples = (textureSamples(rgbaI) + textureSamples(rgbaU));" in glsl
    assert "imageSamples(" not in glsl


def test_codegen_multisample_image_resource_array_queries_keep_explicit_formats():
    crossgl = """
    shader MultisampleImageResourceArrayQueries {
        image2DMS colorImages[4] @rgba16f;
        uimage2DMSArray counterLayers[4] @rgba32ui;

        struct VSInput {
            vec3 position @ POSITION;
        };

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        ivec2 queryColorSize(image2DMS images[] @rgba16f, int layer) {
            return imageSize(images[layer]);
        }

        int queryCounterSamples(uimage2DMSArray images[] @rgba32ui, int layer) {
            return imageSamples(images[layer]);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 colorSize = queryColorSize(colorImages, input.layer);
                int samples = queryCounterSamples(counterLayers, input.layer);
                uvec4 count = imageLoad(counterLayers[input.layer], ivec3(0, 1, 2), 3);
                imageStore(counterLayers[input.layer], ivec3(0, 1, 2), 3, count + uvec4(1u));
                return vec4(float(colorSize.x + samples + int(count.x)));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> colorImages[4] : register(t0);" in hlsl
    assert "Texture2DMSArray<uint4> counterLayers[4] : register(t4);" in hlsl
    assert "int2 imageSize(Texture2DMS<float4> image)" in hlsl
    assert "int textureSamples(Texture2DMSArray<uint4> tex)" in hlsl
    assert "int2 queryColorSize(Texture2DMS<float4> images[4], int layer)" in hlsl
    assert (
        "int queryCounterSamples(Texture2DMSArray<uint4> images[4], int layer)" in hlsl
    )
    assert "return imageSize(images[layer]);" in hlsl
    assert "return textureSamples(images[layer]);" in hlsl
    assert "uint4 count = counterLayers[input.layer].Load(int3(0, 1, 2), 3);" in hlsl
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMSArray<uint4>" in hlsl
    )
    assert "imageSamples(" not in hlsl
    assert "imageStore(" not in hlsl

    assert (
        "array<texture2d_ms<float, access::read>, 4> colorImages [[texture(0)]]"
        in metal
    )
    assert (
        "array<texture2d_ms_array<uint, access::read>, 4> counterLayers [[texture(4)]]"
        in metal
    )
    assert (
        "int2 queryColorSize(array<texture2d_ms<float, access::read>, 4> images, int layer)"
        in metal
    )
    assert (
        "int queryCounterSamples(array<texture2d_ms_array<uint, access::read>, 4> images, int layer)"
        in metal
    )
    assert (
        "return int2(images[layer].get_width(), images[layer].get_height());" in metal
    )
    assert "return int(images[layer].get_num_samples());" in metal
    assert (
        "uint4 count = counterLayers[input.layer].read(uint2((int3(0, 1, 2)).xy), uint((int3(0, 1, 2)).z), uint(3));"
        in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms_array<uint, access::read>" in metal
    )
    assert "texture2d_ms_array<uint, access::read_write>" not in metal
    assert ".write(" not in metal
    assert "imageSamples(" not in metal
    assert "imageStore(" not in metal

    assert "layout(rgba16f, binding = 0) uniform image2DMS colorImages[4];" in glsl
    assert (
        "layout(rgba32ui, binding = 4) uniform uimage2DMSArray counterLayers[4];"
        in glsl
    )
    assert "ivec2 queryColorSize(image2DMS images[4], int layer)" in glsl
    assert "int queryCounterSamples(uimage2DMSArray images[4], int layer)" in glsl
    assert "return imageSize(images[layer]);" in glsl
    assert "return textureSamples(images[layer]);" in glsl
    assert "uvec4 count = imageLoad(counterLayers[layer], ivec3(0, 1, 2), 3);" in glsl
    assert (
        "imageStore(counterLayers[layer], ivec3(0, 1, 2), 3, (count + uvec4(1u)));"
        in glsl
    )
    assert "imageSamples(" not in glsl


def test_codegen_for_in_do_while_sampled_query_arrays_infer_size():
    crossgl = """
    shader ForInDoWhileTransitiveSampledQueryArrays {
        sampler2D textures[];
        sampler afterSampler;
        sampler2D afterTexture;

        ivec2 leaf(sampler2D textures[], int limit) {
            ivec2 result = ivec2(0);
            for i in 0..1 {
                do {
                    result = result + textureSize(textures[4], 1);
                } while (false);
            }
            do {
                for j in limit {
                    result = result + ivec2(textureQueryLevels(textures[2]));
                }
            } while (false);
            return result;
        }

        ivec2 mid(sampler2D textures[], int mode, int limit) {
            ivec2 result = ivec2(0);
            switch (mode) {
                case 0:
                    result = leaf(textures, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(textures, limit);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode, int limit) @ gl_FragColor {
                ivec2 dims = mid(textures, mode, limit);
                return texture(afterTexture, afterSampler, uv) + vec4(float(dims.x + dims.y));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert "SamplerState afterSampler : register(s0);" in hlsl
    assert "int textureQueryLevels(Texture2D tex)" in hlsl
    assert "int2 textureSize(Texture2D tex, int lod)" in hlsl
    assert "int2 leaf(Texture2D textures[5], int limit)" in hlsl
    assert "int2 mid(Texture2D textures[5], int mode, int limit)" in hlsl
    assert "for (int i = 0; i < 1; ++i)" in hlsl
    assert "for (int j = 0; j < limit; ++j)" in hlsl
    assert hlsl.count("do {") == 2
    assert hlsl.count("} while (false);") == 2
    assert "textureSize(textures[4], 1)" in hlsl
    assert "int2(textureQueryLevels(textures[2]))" in hlsl
    assert "result = leaf(textures, limit);" in hlsl
    assert "result = (result + leaf(textures, limit));" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "ForInNode(" not in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[6]" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert "sampler afterSampler [[sampler(0)]]" in metal
    assert "int2 leaf(array<texture2d<float>, 5> textures, int limit)" in metal
    assert "int2 mid(array<texture2d<float>, 5> textures, int mode, int limit)" in metal
    assert "for (int i = 0; i < 1; ++i)" in metal
    assert "for (int j = 0; j < limit; ++j)" in metal
    assert metal.count("do {") == 2
    assert metal.count("} while (false);") == 2
    assert (
        "int2(textures[4].get_width(uint(1)), textures[4].get_height(uint(1)))" in metal
    )
    assert "int2(int(textures[2].get_num_mip_levels()))" in metal
    assert "result = leaf(textures, limit);" in metal
    assert "result = result + leaf(textures, limit);" in metal
    assert "DoWhileNode(" not in metal
    assert "ForInNode(" not in metal
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "array<texture2d<float>, 6> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "ivec2 leaf(sampler2D textures[5], int limit)" in glsl
    assert "ivec2 mid(sampler2D textures[5], int mode, int limit)" in glsl
    assert "for (int i = 0; i < 1; ++i)" in glsl
    assert "for (int j = 0; j < limit; ++j)" in glsl
    assert glsl.count("do {") == 2
    assert glsl.count("} while (false);") == 2
    assert "textureSize(textures[4], 1)" in glsl
    assert "ivec2(textureQueryLevels(textures[2]))" in glsl
    assert "result = leaf(textures, limit);" in glsl
    assert "result = (result + leaf(textures, limit));" in glsl
    assert "DoWhileNode(" not in glsl
    assert "ForInNode(" not in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[6]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];",
        "sampler2D textures[4];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_for_in_do_while_image_query_arrays_infer_size():
    crossgl = """
    shader ForInDoWhileTransitiveImageQueryArrays {
        image2DMS colorImages @rgba16f[];
        image2DMS afterImage @rgba16f;

        ivec2 leaf(image2DMS images[] @rgba16f, int limit) {
            ivec2 result = ivec2(0);
            for i in 0..1 {
                do {
                    result = result + imageSize(images[3]);
                } while (false);
            }
            do {
                for j in limit {
                    result = result + ivec2(imageSamples(images[1]));
                }
            } while (false);
            return result;
        }

        ivec2 mid(image2DMS images[] @rgba16f, int mode, int limit) {
            ivec2 result = ivec2(0);
            switch (mode) {
                case 0:
                    result = leaf(images, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(images, limit);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(int mode, int limit) @ gl_FragColor {
                ivec2 dims = mid(colorImages, mode, limit) + imageSize(afterImage);
                return vec4(float(dims.x + dims.y));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> colorImages[4] : register(t0);" in hlsl
    assert "Texture2DMS<float4> afterImage : register(t4);" in hlsl
    assert "int2 imageSize(Texture2DMS<float4> image)" in hlsl
    assert "int textureSamples(Texture2DMS<float4> tex)" in hlsl
    assert "int2 leaf(Texture2DMS<float4> images[4], int limit)" in hlsl
    assert "int2 mid(Texture2DMS<float4> images[4], int mode, int limit)" in hlsl
    assert "for (int i = 0; i < 1; ++i)" in hlsl
    assert "for (int j = 0; j < limit; ++j)" in hlsl
    assert hlsl.count("do {") == 2
    assert hlsl.count("} while (false);") == 2
    assert "imageSize(images[3])" in hlsl
    assert "int2(textureSamples(images[1]))" in hlsl
    assert "imageSize(afterImage)" in hlsl
    assert "result = leaf(images, limit);" in hlsl
    assert "result = (result + leaf(images, limit));" in hlsl
    assert "imageSamples(" not in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "ForInNode(" not in hlsl
    assert "Texture2DMS<float4> colorImages[]" not in hlsl
    assert "Texture2DMS<float4> colorImages[5]" not in hlsl

    assert (
        "array<texture2d_ms<float, access::read>, 4> colorImages [[texture(0)]]"
        in metal
    )
    assert "texture2d_ms<float, access::read> afterImage [[texture(4)]]" in metal
    assert (
        "int2 leaf(array<texture2d_ms<float, access::read>, 4> images, int limit)"
        in metal
    )
    assert (
        "int2 mid(array<texture2d_ms<float, access::read>, 4> images, int mode, int limit)"
        in metal
    )
    assert "for (int i = 0; i < 1; ++i)" in metal
    assert "for (int j = 0; j < limit; ++j)" in metal
    assert metal.count("do {") == 2
    assert metal.count("} while (false);") == 2
    assert "int2(images[3].get_width(), images[3].get_height())" in metal
    assert "int2(int(images[1].get_num_samples()))" in metal
    assert "int2(afterImage.get_width(), afterImage.get_height())" in metal
    assert "result = leaf(images, limit);" in metal
    assert "result = result + leaf(images, limit);" in metal
    assert "imageSamples(" not in metal
    assert "DoWhileNode(" not in metal
    assert "ForInNode(" not in metal
    assert "array<texture2d_ms<float, access::read>, 1> colorImages" not in metal
    assert "array<texture2d_ms<float, access::read>, 5> colorImages" not in metal

    assert "layout(rgba16f, binding = 0) uniform image2DMS colorImages[4];" in glsl
    assert "layout(rgba16f, binding = 4) uniform image2DMS afterImage;" in glsl
    assert "ivec2 leaf(image2DMS images[4], int limit)" in glsl
    assert "ivec2 mid(image2DMS images[4], int mode, int limit)" in glsl
    assert "for (int i = 0; i < 1; ++i)" in glsl
    assert "for (int j = 0; j < limit; ++j)" in glsl
    assert glsl.count("do {") == 4
    assert glsl.count("} while (false);") == 4
    assert "imageSize(images[3])" in glsl
    assert "ivec2(textureSamples(images[1]))" in glsl
    assert "imageSize(afterImage)" in glsl
    assert "imageSamples(" not in glsl
    assert "DoWhileNode(" not in glsl
    assert "ForInNode(" not in glsl
    assert "image2DMS colorImages[]" not in glsl
    assert "image2DMS colorImages[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2DMS colorImages @rgba16f[];",
        "image2DMS colorImages[3] @rgba16f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'colorImages': 3 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_cast_and_literal_swizzle_sampled_indices_infer_size():
    crossgl = """
    shader CastSwizzleSampledIndices {
        const int BASE = 1;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[int(BASE + 2)], samplers[int(BASE + 2)], uv)
                + texture(textures[ivec2(1, 3).y], samplers[ivec2(1, 3).y], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t4);" in hlsl
    assert "SamplerState afterTextureSampler : register(s4);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], "
        "float2 uv)" in hlsl
    )
    assert "textures[int((BASE + 2))].Sample(samplers[int((BASE + 2))], uv)" in hlsl
    assert "textures[int2(1, 3).y].Sample(samplers[int2(1, 3).y], uv)" in hlsl
    assert "sampleLayer(textures, samplers, uv)" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(4)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 4> textures, "
        "array<sampler, 4> samplers, float2 uv)" in metal
    )
    assert "textures[int(BASE + 2)].sample(samplers[int(BASE + 2)], uv)" in metal
    assert "textures[int2(1, 3).y].sample(samplers[int2(1, 3).y], uv)" in metal
    assert "sampleLayer(textures, samplers, uv)" in metal
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "array<texture2d<float>, 5> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[4], vec2 uv)" in glsl
    assert "texture(textures[int((BASE + 2))], uv)" in glsl
    assert "texture(textures[ivec2(1, 3).y], uv)" in glsl
    assert "sampleLayer(textures, uv)" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[3];\n        sampler samplers[3];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 3 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_cast_and_literal_swizzle_image_indices_infer_size():
    crossgl = """
    shader CastSwizzleImageIndices {
        const int BASE = 1;
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[int(BASE + 2)], pixel)
                + imageLoad(images[ivec2(1, 3).y], pixel);
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1));
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 readLayer(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "images[int((BASE + 2))][pixel]" in hlsl
    assert "images[int2(1, 3).y][pixel]" in hlsl
    assert "readLayer(rgFloatImages, int2(0, 1))" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 4> "
        "images, int2 pixel)" in metal
    )
    assert "images[int(BASE + 2)].read(uint2(pixel)).xy" in metal
    assert "images[int2(1, 3).y].read(uint2(pixel)).xy" in metal
    assert "readLayer(rgFloatImages, int2(0, 1))" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[4], ivec2 pixel)" in glsl
    assert "imageLoad(images[int((BASE + 2))], pixel).xy" in glsl
    assert "imageLoad(images[ivec2(1, 3).y], pixel).xy" in glsl
    assert "readLayer__glsl_images_rgFloatImages(ivec2(0, 1))" in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[3] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 3 and 4"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_alias_swizzle_sampled_index_infers_size():
    crossgl = """
    shader VectorAliasSampledIndex {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            ivec2 layers = ivec2(2, 6);
            return texture(textures[layers.y], samplers[layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv)" in hlsl
    )
    assert "int2 layers = int2(2, 6);" in hlsl
    assert "return textures[layers.y].Sample(samplers[layers.y], uv);" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv)" in metal
    )
    assert "int2 layers = int2(2, 6);" in metal
    assert "return textures[layers.y].sample(samplers[layers.y], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[7], vec2 uv)" in glsl
    assert "ivec2 layers = ivec2(2, 6);" in glsl
    assert "return texture(textures[layers.y], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_parameter_swizzle_image_index_infers_size():
    crossgl = """
    shader VectorParamImageIndex {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel, ivec2 layers) {
            return imageLoad(images[layers.y], pixel);
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel) {
            ivec2 layers = ivec2(1, 5);
            return leaf(images, pixel, layers);
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1));
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[6], int2 pixel, int2 layers)" in hlsl
    assert "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel)" in hlsl
    assert "int2 layers = int2(1, 5);" in hlsl
    assert "return images[layers.y][pixel];" in hlsl
    assert "return leaf(images, pixel, layers);" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, int2 layers)" in metal
    )
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel)" in metal
    )
    assert "int2 layers = int2(1, 5);" in metal
    assert "return images[layers.y].read(uint2(pixel)).xy;" in metal
    assert "return leaf(images, pixel, layers);" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[6], ivec2 pixel, ivec2 layers)" in glsl
    assert "vec2 readLayer(image2D images[6], ivec2 pixel)" in glsl
    assert "ivec2 layers = ivec2(1, 5);" in glsl
    assert "return imageLoad(images[layers.y], pixel).xy;" in glsl
    assert "return leaf(images, pixel, layers);" in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_component_if_assignment_infers_sampled_size():
    crossgl = """
    shader VectorComponentIfAssignSampledIndex {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers = ivec2(0, 1);
            if (choose) {
                layers.y = 6;
            } else {
                layers.y = 4;
            }
            return texture(textures[layers.y], samplers[layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "int2 layers = int2(0, 1);" in hlsl
    assert "layers.y = 6;" in hlsl
    assert "layers.y = 4;" in hlsl
    assert "return textures[layers.y].Sample(samplers[layers.y], uv);" in hlsl
    assert "Texture2D textures[2]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, bool choose)" in metal
    )
    assert "int2 layers = int2(0, 1);" in metal
    assert "layers.y = 6;" in metal
    assert "layers.y = 4;" in metal
    assert "return textures[layers.y].sample(samplers[layers.y], uv);" in metal
    assert "array<texture2d<float>, 2> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[7], vec2 uv, bool choose)" in glsl
    assert "ivec2 layers = ivec2(0, 1);" in glsl
    assert "layers.y = 6;" in glsl
    assert "layers.y = 4;" in glsl
    assert "return texture(textures[layers.y], uv);" in glsl
    assert "sampler2D textures[2]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_split_component_if_assignment_preserves_sampled_branch_correlation():
    crossgl = """
    shader SplitComponentIfSampledIndex {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers = ivec2(0, 1);
            if (choose) {
                layers.x = 5;
                layers.y = 2;
            } else {
                layers.x = 2;
                layers.y = 5;
            }
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "layers.x = 5;" in hlsl
    assert "layers.y = 2;" in hlsl
    assert "layers.x = 2;" in hlsl
    assert "layers.y = 5;" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[11]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv, bool choose)" in metal
    )
    assert "layers.x = 5;" in metal
    assert "layers.y = 2;" in metal
    assert "layers.x = 2;" in metal
    assert "layers.y = 5;" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 11> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv, bool choose)" in glsl
    assert "layers.x = 5;" in glsl
    assert "layers.y = 2;" in glsl
    assert "layers.x = 2;" in glsl
    assert "layers.y = 5;" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[11]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_component_compound_assignment_infers_sampled_size():
    crossgl = """
    shader VectorComponentCompoundSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            ivec2 layers = ivec2(0, 1);
            layers.y += 5;
            return texture(textures[layers.y + 1], samplers[layers.y + 1], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv)" in hlsl
    )
    assert "int2 layers = int2(0, 1);" in hlsl
    assert "layers.y += 5;" in hlsl
    assert (
        "return textures[(layers.y + 1)].Sample(samplers[(layers.y + 1)], uv);" in hlsl
    )
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv)" in metal
    )
    assert "int2 layers = int2(0, 1);" in metal
    assert "layers.y += 5;" in metal
    assert "return textures[layers.y + 1].sample(samplers[layers.y + 1], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv)" in glsl
    assert "ivec2 layers = ivec2(0, 1);" in glsl
    assert "layers.y += 5;" in glsl
    assert "return texture(textures[(layers.y + 1)], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_component_self_compound_helper_infers_image_size():
    crossgl = """
    shader VectorComponentCompoundHelperImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(ivec2 layers, int offset) {
            return layers.y + offset;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel) {
            ivec2 layers = ivec2(0, 2);
            layers.y += layers.y;
            int layer = pickLayer(layers, 3);
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1));
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[8] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u8);" in hlsl
    assert "float2 readLayer(RWTexture2D<float2> images[8], int2 pixel)" in hlsl
    assert "int2 layers = int2(0, 2);" in hlsl
    assert "layers.y += layers.y;" in hlsl
    assert "int layer = pickLayer(layers, 3);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 8> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(8)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 8> "
        "images, int2 pixel)" in metal
    )
    assert "int2 layers = int2(0, 2);" in metal
    assert "layers.y += layers.y;" in metal
    assert "int layer = pickLayer(layers, 3);" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[8];" in glsl
    assert "layout(rg32f, binding = 8) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[8], ivec2 pixel)" in glsl
    assert "ivec2 layers = ivec2(0, 2);" in glsl
    assert "layers.y += layers.y;" in glsl
    assert "int layer = pickLayer(layers, 3);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[7] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 7 and 8"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_bounded_for_component_compound_infers_sampled_size():
    crossgl = """
    shader ForLiteralComponentPlusAfterLoop {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            ivec2 layers = ivec2(0, 1);
            for (int i = 0; i < 3; i = i + 1) {
                layers.y += 2;
            }
            return texture(textures[layers.y], samplers[layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv)" in hlsl
    )
    assert "for (int i = 0; (i < 3); i = (i + 1))" in hlsl
    assert "layers.y += 2;" in hlsl
    assert "return textures[layers.y].Sample(samplers[layers.y], uv);" in hlsl
    assert "Texture2D textures[4]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv)" in metal
    )
    assert "for (int i = 0; i < 3; i = i + 1)" in metal
    assert "layers.y += 2;" in metal
    assert "return textures[layers.y].sample(samplers[layers.y], uv);" in metal
    assert "array<texture2d<float>, 4> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv)" in glsl
    assert "for (int i = 0; (i < 3); i = (i + 1))" in glsl
    assert "layers.y += 2;" in glsl
    assert "return texture(textures[layers.y], uv);" in glsl
    assert "sampler2D textures[4]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_bounded_for_in_scalar_compound_infers_image_size():
    crossgl = """
    shader ForInLiteralScalarPlusAfterLoop {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel) {
            int layer = 1;
            for i in 0..3 {
                layer += 2;
            }
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1));
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[8] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u8);" in hlsl
    assert "float2 readLayer(RWTexture2D<float2> images[8], int2 pixel)" in hlsl
    assert "for (int i = 0; i < 3; ++i)" in hlsl
    assert "layer += 2;" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[4]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 8> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(8)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 8> "
        "images, int2 pixel)" in metal
    )
    assert "for (int i = 0; i < 3; ++i)" in metal
    assert "layer += 2;" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 4> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[8];" in glsl
    assert "layout(rg32f, binding = 8) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[8], ivec2 pixel)" in glsl
    assert "for (int i = 0; i < 3; ++i)" in glsl
    assert "layer += 2;" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[4]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[7] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 7 and 8"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_nested_loop_induction_indices_infer_sampled_size():
    crossgl = """
    shader NestedForSampledDirectIndex {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            ivec2 layers = ivec2(0, 2);
            vec4 result = vec4(0.0);
            for (int i = 0; i < 2; i = i + 1) {
                for j in 0..3 {
                    result = result + texture(
                        textures[i + j + layers.y],
                        samplers[i + j + layers.y],
                        uv
                    );
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[6] : register(t0);" in hlsl
    assert "SamplerState samplers[6] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t6);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[6], SamplerState samplers[6], "
        "float2 uv)" in hlsl
    )
    assert "for (int i = 0; (i < 2); i = (i + 1))" in hlsl
    assert "for (int j = 0; j < 3; ++j)" in hlsl
    assert (
        "textures[((i + j) + layers.y)].Sample("
        "samplers[((i + j) + layers.y)], uv)" in hlsl
    )
    assert "Texture2D textures[5]" not in hlsl

    assert "array<texture2d<float>, 6> textures [[texture(0)]]" in metal
    assert "array<sampler, 6> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(6)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 6> textures, "
        "array<sampler, 6> samplers, float2 uv)" in metal
    )
    assert "for (int i = 0; i < 2; i = i + 1)" in metal
    assert "for (int j = 0; j < 3; ++j)" in metal
    assert "textures[i + j + layers.y].sample(samplers[i + j + layers.y], uv)" in metal
    assert "array<texture2d<float>, 5> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[6];" in glsl
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[6], vec2 uv)" in glsl
    assert "for (int i = 0; (i < 2); i = (i + 1))" in glsl
    assert "for (int j = 0; j < 3; ++j)" in glsl
    assert "texture(textures[((i + j) + layers.y)], uv)" in glsl
    assert "sampler2D textures[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[5];\n        sampler samplers[5];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 5 and 6"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_nested_loop_helper_indices_infer_image_size():
    crossgl = """
    shader NestedForImageHelperIndex {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(ivec2 layers, int outerIndex, int innerIndex) {
            return layers.y + outerIndex + innerIndex;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel) {
            ivec2 layers = ivec2(0, 2);
            vec2 result = vec2(0.0);
            for (int i = 0; i < 2; i = i + 1) {
                for j in 0..3 {
                    result = result + imageLoad(
                        images[pickLayer(layers, i, j)],
                        pixel
                    );
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1));
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel)" in hlsl
    assert "int pickLayer(int2 layers, int outerIndex, int innerIndex)" in hlsl
    assert "for (int i = 0; (i < 2); i = (i + 1))" in hlsl
    assert "for (int j = 0; j < 3; ++j)" in hlsl
    assert "images[pickLayer(layers, i, j)][pixel]" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel)" in metal
    )
    assert "int pickLayer(int2 layers, int outerIndex, int innerIndex)" in metal
    assert "for (int i = 0; i < 2; i = i + 1)" in metal
    assert "for (int j = 0; j < 3; ++j)" in metal
    assert "images[pickLayer(layers, i, j)].read(uint2(pixel)).xy" in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[6], ivec2 pixel)" in glsl
    assert "int pickLayer(ivec2 layers, int outerIndex, int innerIndex)" in glsl
    assert "for (int i = 0; (i < 2); i = (i + 1))" in glsl
    assert "for (int j = 0; j < 3; ++j)" in glsl
    assert "imageLoad(images[pickLayer(layers, i, j)], pixel).xy" in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_return_helper_assignment_infers_sampled_size():
    crossgl = """
    shader VectorFromHelperAssignSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        ivec2 bump(ivec2 layers, int i) {
            return ivec2(layers.x + i, layers.y + 2);
        }

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            ivec2 layers = ivec2(0, 1);
            for (int i = 0; i < 3; i = i + 1) {
                layers = bump(layers, i);
            }
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[11] : register(t0);" in hlsl
    assert "SamplerState samplers[11] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t11);" in hlsl
    assert "int2 bump(int2 layers, int i)" in hlsl
    assert "return int2((layers.x + i), (layers.y + 2));" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[11], SamplerState samplers[11], "
        "float2 uv)" in hlsl
    )
    assert "layers = bump(layers, i);" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 11> textures [[texture(0)]]" in metal
    assert "array<sampler, 11> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(11)]]" in metal
    assert "int2 bump(int2 layers, int i)" in metal
    assert "return int2(layers.x + i, layers.y + 2);" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 11> textures, "
        "array<sampler, 11> samplers, float2 uv)" in metal
    )
    assert "layers = bump(layers, i);" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[11];" in glsl
    assert "layout(binding = 11) uniform sampler2D afterTexture;" in glsl
    assert "ivec2 bump(ivec2 layers, int i)" in glsl
    assert "return ivec2((layers.x + i), (layers.y + 2));" in glsl
    assert "vec4 sampleLayer(sampler2D textures[11], vec2 uv)" in glsl
    assert "layers = bump(layers, i);" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[10];\n        sampler samplers[10];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 10 and 11"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_return_helper_assignment_infers_image_size():
    crossgl = """
    shader VectorFromNestedHelperImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        ivec2 bump(ivec2 layers, int i) {
            return ivec2(layers.x + i, layers.y + 2);
        }

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel) {
            ivec2 layers = ivec2(0, 1);
            for (int i = 0; i < 3; i = i + 1) {
                layers = bump(layers, i);
            }
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1));
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[12] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u12);" in hlsl
    assert "int2 bump(int2 layers, int i)" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert "float2 readLayer(RWTexture2D<float2> images[12], int2 pixel)" in hlsl
    assert "layers = bump(layers, i);" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 12> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(12)]]" in metal
    assert "int2 bump(int2 layers, int i)" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 12> "
        "images, int2 pixel)" in metal
    )
    assert "layers = bump(layers, i);" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[12];" in glsl
    assert "layout(rg32f, binding = 12) uniform image2D afterImage;" in glsl
    assert "ivec2 bump(ivec2 layers, int i)" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[12], ivec2 pixel)" in glsl
    assert "layers = bump(layers, i);" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[11] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 11 and 12"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_ternary_vector_return_helper_infers_sampled_size():
    crossgl = """
    shader TernaryVectorReturnSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        ivec2 chooseLayer(ivec2 layers, bool choose) {
            return choose
                ? ivec2(layers.x + 5, layers.y + 1)
                : ivec2(layers.x + 2, layers.y + 4);
        }

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers = ivec2(0, 1);
            layers = chooseLayer(layers, choose);
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert "int2 chooseLayer(int2 layers, bool choose)" in hlsl
    assert (
        "return (choose ? int2((layers.x + 5), (layers.y + 1)) : "
        "int2((layers.x + 2), (layers.y + 4)));" in hlsl
    )
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "layers = chooseLayer(layers, choose);" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert "int2 chooseLayer(int2 layers, bool choose)" in metal
    assert (
        "return choose ? int2(layers.x + 5, layers.y + 1) : "
        "int2(layers.x + 2, layers.y + 4);" in metal
    )
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv, bool choose)" in metal
    )
    assert "layers = chooseLayer(layers, choose);" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "ivec2 chooseLayer(ivec2 layers, bool choose)" in glsl
    assert (
        "return (choose ? ivec2((layers.x + 5), (layers.y + 1)) : "
        "ivec2((layers.x + 2), (layers.y + 4)));" in glsl
    )
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv, bool choose)" in glsl
    assert "layers = chooseLayer(layers, choose);" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_branch_vector_return_helper_infers_image_size():
    crossgl = """
    shader BranchVectorReturnImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        ivec2 chooseLayer(ivec2 layers, bool choose) {
            if (choose) {
                return ivec2(layers.x + 3, layers.y + 2);
            }
            return ivec2(layers.x + 1, layers.y + 5);
        }

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            ivec2 layers = ivec2(0, 1);
            layers = chooseLayer(layers, choose);
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), true);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[9] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u9);" in hlsl
    assert "int2 chooseLayer(int2 layers, bool choose)" in hlsl
    assert "return int2((layers.x + 3), (layers.y + 2));" in hlsl
    assert "return int2((layers.x + 1), (layers.y + 5));" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[9], int2 pixel, "
        "bool choose)" in hlsl
    )
    assert "layers = chooseLayer(layers, choose);" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 9> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(9)]]" in metal
    assert "int2 chooseLayer(int2 layers, bool choose)" in metal
    assert "return int2(layers.x + 3, layers.y + 2);" in metal
    assert "return int2(layers.x + 1, layers.y + 5);" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 9> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert "layers = chooseLayer(layers, choose);" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[9];" in glsl
    assert "layout(rg32f, binding = 9) uniform image2D afterImage;" in glsl
    assert "ivec2 chooseLayer(ivec2 layers, bool choose)" in glsl
    assert "return ivec2((layers.x + 3), (layers.y + 2));" in glsl
    assert "return ivec2((layers.x + 1), (layers.y + 5));" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[9], ivec2 pixel, bool choose)" in glsl
    assert "layers = chooseLayer(layers, choose);" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[8] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 8 and 9"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_component_update_after_ternary_preserves_sampled_correlation():
    crossgl = """
    shader ComponentUpdateAfterTernarySampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers = choose ? ivec2(5, 2) : ivec2(2, 5);
            layers.x += 1;
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[9] : register(t0);" in hlsl
    assert "SamplerState samplers[9] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t9);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[9], SamplerState samplers[9], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "layers.x += 1;" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[12]" not in hlsl

    assert "array<texture2d<float>, 9> textures [[texture(0)]]" in metal
    assert "array<sampler, 9> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(9)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 9> textures, "
        "array<sampler, 9> samplers, float2 uv, bool choose)" in metal
    )
    assert "layers.x += 1;" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 12> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[9];" in glsl
    assert "layout(binding = 9) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[9], vec2 uv, bool choose)" in glsl
    assert "layers.x += 1;" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[12]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[8];\n        sampler samplers[8];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 8 and 9"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_component_update_after_helper_preserves_image_correlation():
    crossgl = """
    shader ComponentUpdateAfterHelperImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        ivec2 chooseLayer(bool choose) {
            return choose ? ivec2(5, 2) : ivec2(2, 5);
        }

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            ivec2 layers = chooseLayer(choose);
            layers.x += 1;
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main(bool choose) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[10] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u10);" in hlsl
    assert "int2 chooseLayer(bool choose)" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[10], int2 pixel, "
        "bool choose)" in hlsl
    )
    assert "layers.x += 1;" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[13]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 10> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(10)]]" in metal
    assert "int2 chooseLayer(bool choose)" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 10> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert "layers.x += 1;" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 13> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[10];" in glsl
    assert "layout(rg32f, binding = 10) uniform image2D afterImage;" in glsl
    assert "ivec2 chooseLayer(bool choose)" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[10], ivec2 pixel, bool choose)" in glsl
    assert "layers.x += 1;" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[13]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[9] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 9 and 10"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_component_helper_assignment_preserves_sampled_correlation():
    crossgl = """
    shader ComponentHelperAssignSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        int mirrorLayer(int value) {
            return 7 - value;
        }

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers = choose ? ivec2(5, 2) : ivec2(2, 5);
            layers.x = mirrorLayer(layers.y);
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert "int mirrorLayer(int value)" in hlsl
    assert "return (7 - value);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "layers.x = mirrorLayer(layers.y);" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[11]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert "int mirrorLayer(int value)" in metal
    assert "return 7 - value;" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv, bool choose)" in metal
    )
    assert "layers.x = mirrorLayer(layers.y);" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 11> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "int mirrorLayer(int value)" in glsl
    assert "return (7 - value);" in glsl
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv, bool choose)" in glsl
    assert "layers.x = mirrorLayer(layers.y);" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[11]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_component_vector_helper_assignment_preserves_image_correlation():
    crossgl = """
    shader ComponentVectorHelperAssignImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int mirrorLayer(ivec2 layers) {
            return 7 - layers.y;
        }

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            ivec2 layers = choose ? ivec2(5, 2) : ivec2(2, 5);
            layers.x = mirrorLayer(layers);
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main(bool choose) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[9] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u9);" in hlsl
    assert "int mirrorLayer(int2 layers)" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[9], int2 pixel, "
        "bool choose)" in hlsl
    )
    assert "layers.x = mirrorLayer(layers);" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[12]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 9> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(9)]]" in metal
    assert "int mirrorLayer(int2 layers)" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 9> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert "layers.x = mirrorLayer(layers);" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 12> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[9];" in glsl
    assert "layout(rg32f, binding = 9) uniform image2D afterImage;" in glsl
    assert "int mirrorLayer(ivec2 layers)" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[9], ivec2 pixel, bool choose)" in glsl
    assert "layers.x = mirrorLayer(layers);" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[12]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[8] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 8 and 9"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_swizzle_assignment_preserves_sampled_correlation():
    crossgl = """
    shader SwizzleAssignSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers = choose ? ivec2(6, 1) : ivec2(1, 6);
            layers = layers.yx;
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "layers = layers.yx;" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[13]" not in hlsl
    assert "Texture2D textures[]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv, bool choose)" in metal
    )
    assert "layers = layers.yx;" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 13> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv, bool choose)" in glsl
    assert "layers = layers.yx;" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[13]" not in glsl
    assert "sampler2D textures[]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_constructor_reconstruction_preserves_image_correlation():
    crossgl = """
    shader ConstructorReconstructImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            ivec2 layers = choose ? ivec2(6, 1) : ivec2(1, 6);
            layers = ivec2(layers.y + 1, layers.x);
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main(bool choose) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[10] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u10);" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[10], int2 pixel, "
        "bool choose)" in hlsl
    )
    assert "layers = int2((layers.y + 1), layers.x);" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[15]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 10> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(10)]]" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 10> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert "layers = int2(layers.y + 1, layers.x);" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 15> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[10];" in glsl
    assert "layout(rg32f, binding = 10) uniform image2D afterImage;" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[10], ivec2 pixel, bool choose)" in glsl
    assert "layers = ivec2((layers.y + 1), layers.x);" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[15]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[9] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 9 and 10"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_nested_block_assignment_emits_scope_and_preserves_image_correlation():
    crossgl = """
    shader BlockAssignedImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            ivec2 layers;
            {
                layers = choose ? ivec2(5, 2) : ivec2(2, 5);
            }
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main(bool choose) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "BlockNode(" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[9] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u9);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[9], int2 pixel, "
        "bool choose)" in hlsl
    )
    assert (
        "    {\n"
        "        layers = (choose ? int2(5, 2) : int2(2, 5));\n"
        "    }\n"
        "    return images[pickLayer(layers, 1)][pixel];" in hlsl
    )
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[15]" not in hlsl

    assert "BlockNode(" not in metal
    assert (
        "array<texture2d<float, access::read_write>, 9> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(9)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 9> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert (
        "    {\n"
        "        layers = choose ? int2(5, 2) : int2(2, 5);\n"
        "    }\n"
        "    return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    )
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 15> rgFloatImages" not in metal

    assert "BlockNode(" not in glsl
    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[9];" in glsl
    assert "layout(rg32f, binding = 9) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[9], ivec2 pixel, bool choose)" in glsl
    assert (
        "    {\n"
        "        layers = (choose ? ivec2(5, 2) : ivec2(2, 5));\n"
        "    }\n"
        "    return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    )
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[15]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[8] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 8 and 9"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_nested_block_shadowed_value_type_does_not_leak_to_image_store():
    crossgl = """
    shader BlockShadowedScalarImageStore {
        image2D scalarImage @r32f;

        compute {
            void main() {
                float value = 0.5;
                {
                    vec2 value = vec2(1.0, 2.0);
                }
                imageStore(scalarImage, ivec2(0, 0), value);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "BlockNode(" not in hlsl
    assert "RWTexture2D<float> scalarImage : register(u0);" in hlsl
    assert (
        "    float value = 0.5;\n"
        "    {\n"
        "        float2 value = float2(1.0, 2.0);\n"
        "    }\n"
        "    scalarImage[int2(0, 0)] = value;" in hlsl
    )

    assert "BlockNode(" not in metal
    assert "texture2d<float, access::read_write> scalarImage [[texture(0)]]" in metal
    assert (
        "    float value = 0.5;\n"
        "    {\n"
        "        float2 value = float2(1.0, 2.0);\n"
        "    }\n"
        "    scalarImage.write(float4(value), uint2(int2(0, 0)));" in metal
    )

    assert "BlockNode(" not in glsl
    assert "layout(r32f, binding = 0) uniform image2D scalarImage;" in glsl
    assert (
        "    float value = 0.5;\n"
        "    {\n"
        "        vec2 value = vec2(1.0, 2.0);\n"
        "    }\n"
        "    imageStore(scalarImage, ivec2(0, 0), vec4(value));" in glsl
    )


def test_codegen_control_flow_shadowed_value_types_do_not_leak_to_image_store():
    cases = [
        (
            "BranchShadowedScalarImageStore",
            "void main(bool choose)",
            "if (choose) {\n                    vec2 value = vec2(1.0, 2.0);\n                }",
            "if (choose) {\n        float2 value = float2(1.0, 2.0);\n    }",
            "if (choose) {\n        vec2 value = vec2(1.0, 2.0);\n    }",
        ),
        (
            "LoopShadowedScalarImageStore",
            "void main()",
            (
                "for (int i = 0; i < 1; i = i + 1) {\n"
                "                    vec2 value = vec2(1.0, 2.0);\n"
                "                }"
            ),
            (
                "for (int i = 0; (i < 1); i = (i + 1)) {\n"
                "        float2 value = float2(1.0, 2.0);\n"
                "    }"
            ),
            (
                "for (int i = 0; (i < 1); i = (i + 1)) {\n"
                "        vec2 value = vec2(1.0, 2.0);\n"
                "    }"
            ),
        ),
    ]

    for shader_name, main_signature, control_source, hlsl_body, glsl_body in cases:
        crossgl = f"""
        shader {shader_name} {{
            image2D scalarImage @r32f;

            compute {{
                {main_signature} {{
                    float value = 0.5;
                    {control_source}
                    imageStore(scalarImage, ivec2(0, 0), value);
                }}
            }}
        }}
        """

        shader_ast = parse_crossgl(crossgl)
        assert shader_ast is not None

        hlsl = HLSLCodeGen().generate(shader_ast)
        metal = MetalCodeGen().generate(shader_ast)
        glsl = GLSLCodeGen().generate(shader_ast)

        assert "RWTexture2D<float> scalarImage : register(u0);" in hlsl
        assert hlsl_body in hlsl
        assert "scalarImage[int2(0, 0)] = value;" in hlsl

        assert (
            "texture2d<float, access::read_write> scalarImage [[texture(0)]]" in metal
        )
        assert (
            hlsl_body.replace("(i < 1)", "i < 1").replace("i = (i + 1)", "i = i + 1")
            in metal
        )
        assert "scalarImage.write(float4(value), uint2(int2(0, 0)));" in metal

        assert "layout(r32f, binding = 0) uniform image2D scalarImage;" in glsl
        assert glsl_body in glsl
        assert "imageStore(scalarImage, ivec2(0, 0), vec4(value));" in glsl


def test_codegen_for_initializer_shadowed_value_type_does_not_leak_to_image_store():
    crossgl = """
    shader ForInitializerShadowedScalarImageStore {
        image2D scalarImage @r32f;

        compute {
            void main() {
                float value = 0.5;
                for (vec2 value = vec2(0.0, 1.0); value.x < 1.0; value.x = value.x + 1.0) {
                }
                imageStore(scalarImage, ivec2(0, 0), value);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float> scalarImage : register(u0);" in hlsl
    assert (
        "for (float2 value = float2(0.0, 1.0); (value.x < 1.0); "
        "value.x = (value.x + 1.0)) {" in hlsl
    )
    assert "scalarImage[int2(0, 0)] = value;" in hlsl

    assert "texture2d<float, access::read_write> scalarImage [[texture(0)]]" in metal
    assert (
        "for (float2 value = float2(0.0, 1.0); value.x < 1.0; "
        "value.x = value.x + 1.0) {" in metal
    )
    assert "scalarImage.write(float4(value), uint2(int2(0, 0)));" in metal

    assert "layout(r32f, binding = 0) uniform image2D scalarImage;" in glsl
    assert (
        "for (vec2 value = vec2(0.0, 1.0); (value.x < 1.0); "
        "value.x = (value.x + 1.0)) {" in glsl
    )
    assert "imageStore(scalarImage, ivec2(0, 0), vec4(value));" in glsl


def test_codegen_for_in_shadowed_counter_type_is_scoped_to_loop_body():
    crossgl = """
    shader ForInShadowedScalarImageStore {
        image2D scalarImage @r32i;
        image2D vectorImage @rg32i;

        compute {
            void main() {
                ivec2 value = ivec2(3, 4);
                for value in 0..1 {
                    imageStore(scalarImage, ivec2(0, 0), value);
                }
                imageStore(vectorImage, ivec2(1, 0), value);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<int> scalarImage : register(u0);" in hlsl
    assert "RWTexture2D<int2> vectorImage : register(u1);" in hlsl
    assert "for (int value = 0; value < 1; ++value)" in hlsl
    assert "scalarImage[int2(0, 0)] = value;" in hlsl
    assert "vectorImage[int2(1, 0)] = value;" in hlsl

    assert "texture2d<int, access::read_write> scalarImage [[texture(0)]]" in metal
    assert "texture2d<int, access::read_write> vectorImage [[texture(1)]]" in metal
    assert "for (int value = 0; value < 1; ++value)" in metal
    assert "scalarImage.write(int4(value), uint2(int2(0, 0)));" in metal
    assert "vectorImage.write(int4(value, 0, 0), uint2(int2(1, 0)));" in metal

    assert "layout(r32i, binding = 0) uniform iimage2D scalarImage;" in glsl
    assert "layout(rg32i, binding = 1) uniform iimage2D vectorImage;" in glsl
    assert "for (int value = 0; value < 1; ++value)" in glsl
    assert "imageStore(scalarImage, ivec2(0, 0), ivec4(value));" in glsl
    assert "imageStore(vectorImage, ivec2(1, 0), ivec4(value, 0, 0));" in glsl


def test_codegen_loop_assigned_outer_vector_preserves_sampled_correlation():
    crossgl = """
    shader LoopAssignedSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            ivec2 layers;
            for (int i = 0; i < 1; i = i + 1) {
                layers = choose ? ivec2(6, 1) : ivec2(1, 6);
            }
            return texture(textures[layers.x + layers.y], samplers[layers.x + layers.y], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[8] : register(t0);" in hlsl
    assert "SamplerState samplers[8] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t8);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[8], SamplerState samplers[8], "
        "float2 uv, bool choose)" in hlsl
    )
    assert "for (int i = 0; (i < 1); i = (i + 1))" in hlsl
    assert "layers = (choose ? int2(6, 1) : int2(1, 6));" in hlsl
    assert (
        "return textures[(layers.x + layers.y)].Sample("
        "samplers[(layers.x + layers.y)], uv);" in hlsl
    )
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[13]" not in hlsl

    assert "array<texture2d<float>, 8> textures [[texture(0)]]" in metal
    assert "array<sampler, 8> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(8)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 8> textures, "
        "array<sampler, 8> samplers, float2 uv, bool choose)" in metal
    )
    assert "for (int i = 0; i < 1; i = i + 1)" in metal
    assert "layers = choose ? int2(6, 1) : int2(1, 6);" in metal
    assert (
        "return textures[layers.x + layers.y].sample("
        "samplers[layers.x + layers.y], uv);" in metal
    )
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "array<texture2d<float>, 13> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[8];" in glsl
    assert "layout(binding = 8) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[8], vec2 uv, bool choose)" in glsl
    assert "for (int i = 0; (i < 1); i = (i + 1))" in glsl
    assert "layers = (choose ? ivec2(6, 1) : ivec2(1, 6));" in glsl
    assert "return texture(textures[(layers.x + layers.y)], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[13]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[7];\n        sampler samplers[7];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 7 and 8"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_loop_assigned_outer_vector_preserves_image_correlation():
    crossgl = """
    shader LoopAssignedImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            ivec2 layers;
            for (int i = 0; i < 1; i = i + 1) {
                layers = choose ? ivec2(5, 2) : ivec2(2, 5);
            }
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main(bool choose) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[9] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u9);" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[9], int2 pixel, "
        "bool choose)" in hlsl
    )
    assert "for (int i = 0; (i < 1); i = (i + 1))" in hlsl
    assert "layers = (choose ? int2(5, 2) : int2(2, 5));" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[15]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 9> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(9)]]" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 9> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert "for (int i = 0; i < 1; i = i + 1)" in metal
    assert "layers = choose ? int2(5, 2) : int2(2, 5);" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 15> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[9];" in glsl
    assert "layout(rg32f, binding = 9) uniform image2D afterImage;" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[9], ivec2 pixel, bool choose)" in glsl
    assert "for (int i = 0; (i < 1); i = (i + 1))" in glsl
    assert "layers = (choose ? ivec2(5, 2) : ivec2(2, 5));" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[15]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[8] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 8 and 9"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_vector_component_match_assignment_infers_image_size():
    crossgl = """
    shader VectorComponentMatchImageIndex {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, int mode) {
            ivec2 layers = ivec2(0, 1);
            match mode {
                0 => { layers.y = 5; }
                _ => { layers.y = 3; }
            }
            return imageLoad(images[layers.y], pixel);
        }

        compute {
            void main(int mode) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), mode);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, int mode)" in hlsl
    )
    assert "int2 layers = int2(0, 1);" in hlsl
    assert "layers.y = 5;" in hlsl
    assert "layers.y = 3;" in hlsl
    assert "return images[layers.y][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[2]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, int mode)" in metal
    )
    assert "int2 layers = int2(0, 1);" in metal
    assert "layers.y = 5;" in metal
    assert "layers.y = 3;" in metal
    assert "return images[layers.y].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 2> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[6], ivec2 pixel, int mode)" in glsl
    assert "ivec2 layers = ivec2(0, 1);" in glsl
    assert "layers.y = 5;" in glsl
    assert "layers.y = 3;" in glsl
    assert "return imageLoad(images[layers.y], pixel).xy;" in glsl
    assert "image2D rgFloatImages[2]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_split_component_match_preserves_image_branch_correlation():
    crossgl = """
    shader SplitComponentMatchImageIndex {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(ivec2 layers, int extra) {
            return layers.x + layers.y + extra;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, int mode) {
            ivec2 layers = ivec2(0, 0);
            match mode {
                0 => {
                    layers.x = 4;
                    layers.y = 1;
                }
                _ => {
                    layers.x = 1;
                    layers.y = 4;
                }
            }
            return imageLoad(images[pickLayer(layers, 1)], pixel);
        }

        compute {
            void main(int mode) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), mode);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[7] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u7);" in hlsl
    assert "int pickLayer(int2 layers, int extra)" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[7], int2 pixel, int mode)" in hlsl
    )
    assert "layers.x = 4;" in hlsl
    assert "layers.y = 1;" in hlsl
    assert "layers.x = 1;" in hlsl
    assert "layers.y = 4;" in hlsl
    assert "return images[pickLayer(layers, 1)][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[10]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 7> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(7)]]" in metal
    assert "int pickLayer(int2 layers, int extra)" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 7> "
        "images, int2 pixel, int mode)" in metal
    )
    assert "layers.x = 4;" in metal
    assert "layers.y = 1;" in metal
    assert "layers.x = 1;" in metal
    assert "layers.y = 4;" in metal
    assert "return images[pickLayer(layers, 1)].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 10> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[7];" in glsl
    assert "layout(rg32f, binding = 7) uniform image2D afterImage;" in glsl
    assert "int pickLayer(ivec2 layers, int extra)" in glsl
    assert "vec2 readLayer(image2D images[7], ivec2 pixel, int mode)" in glsl
    assert "layers.x = 4;" in glsl
    assert "layers.y = 1;" in glsl
    assert "layers.x = 1;" in glsl
    assert "layers.y = 4;" in glsl
    assert "return imageLoad(images[pickLayer(layers, 1)], pixel).xy;" in glsl
    assert "image2D rgFloatImages[10]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[6] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 6 and 7"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_ternary_sampled_indices_in_function_args_infer_size():
    crossgl = """
    shader TernarySampledFunctionArgIndices {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 blend(vec4 a, vec4 b) {
            return a + b;
        }

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, bool choose) {
            return blend(
                texture(textures[choose ? 4 : 2], samplers[choose ? 4 : 2], uv),
                texture(textures[1], samplers[1], uv)
            );
        }

        fragment {
            vec4 main(vec2 uv, bool choose) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert "SamplerState afterTextureSampler : register(s5);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, bool choose)" in hlsl
    )
    assert (
        "blend(textures[(choose ? 4 : 2)].Sample(samplers[(choose ? 4 : 2)], uv)"
        in hlsl
    )
    assert "textures[1].Sample(samplers[1], uv)" in hlsl
    assert "sampleLayer(textures, samplers, uv, choose)" in hlsl
    assert "Texture2D textures[2]" not in hlsl
    assert "Texture2D textures[6]" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, bool choose)" in metal
    )
    assert (
        "blend(textures[choose ? 4 : 2].sample(samplers[choose ? 4 : 2], uv)" in metal
    )
    assert "textures[1].sample(samplers[1], uv)" in metal
    assert "sampleLayer(textures, samplers, uv, choose)" in metal
    assert "array<texture2d<float>, 2> textures" not in metal
    assert "array<texture2d<float>, 6> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[5], vec2 uv, bool choose)" in glsl
    assert "blend(texture(textures[(choose ? 4 : 2)], uv)" in glsl
    assert "texture(textures[1], uv)" in glsl
    assert "sampleLayer(textures, uv, choose)" in glsl
    assert "sampler2D textures[2]" not in glsl
    assert "sampler2D textures[6]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[4];\n        sampler samplers[4];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_ternary_image_indices_in_function_args_infer_size():
    crossgl = """
    shader TernaryImageFunctionArgIndices {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 combine(vec2 a, vec2 b) {
            return a + b;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose) {
            return combine(
                imageLoad(images[choose ? 3 : 1], pixel),
                imageLoad(images[1], pixel)
            );
        }

        compute {
            void main() {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), true);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[4], int2 pixel, bool choose)"
        in hlsl
    )
    assert "combine(images[(choose ? 3 : 1)][pixel], images[1][pixel])" in hlsl
    assert "readLayer(rgFloatImages, int2(0, 1), true)" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[2]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 4> "
        "images, int2 pixel, bool choose)" in metal
    )
    assert (
        "combine(images[choose ? 3 : 1].read(uint2(pixel)).xy, "
        "images[1].read(uint2(pixel)).xy)" in metal
    )
    assert "readLayer(rgFloatImages, int2(0, 1), true)" in metal
    assert "array<texture2d<float, access::read_write>, 2> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[4], ivec2 pixel, bool choose)" in glsl
    assert (
        "combine(imageLoad(images[(choose ? 3 : 1)], pixel).xy, "
        "imageLoad(images[1], pixel).xy)" in glsl
    )
    assert "readLayer__glsl_images_rgFloatImages(ivec2(0, 1), true)" in glsl
    assert "image2D rgFloatImages[2]" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[3] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 3 and 4"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_ternary_index_alias_initializers_infer_size():
    crossgl = """
    shader TernaryIndexAliasInitializers {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(
            sampler2D textures[],
            sampler samplers[],
            vec2 uv,
            bool outer,
            bool inner,
            int dynamicLayer
        ) {
            int layer = outer ? (inner ? dynamicLayer : 5) : 2;
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool outer, bool inner, int dynamicLayer) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, outer, inner, dynamicLayer) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[6] : register(t0);" in hlsl
    assert "SamplerState samplers[6] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t6);" in hlsl
    assert "SamplerState afterTextureSampler : register(s6);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[6], SamplerState samplers[6], "
        "float2 uv, bool outer, bool inner, int dynamicLayer)" in hlsl
    )
    assert "int layer = (outer ? (inner ? dynamicLayer : 5) : 2);" in hlsl
    assert "return textures[layer].Sample(samplers[layer], uv);" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 6> textures [[texture(0)]]" in metal
    assert "array<sampler, 6> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(6)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 6> textures, "
        "array<sampler, 6> samplers, float2 uv, bool outer, bool inner, "
        "int dynamicLayer)" in metal
    )
    assert "int layer = outer ? inner ? dynamicLayer : 5 : 2;" in metal
    assert "return textures[layer].sample(samplers[layer], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[6];" in glsl
    assert "layout(binding = 6) uniform sampler2D afterTexture;" in glsl
    assert (
        "vec4 sampleLayer(sampler2D textures[6], vec2 uv, bool outer, "
        "bool inner, int dynamicLayer)" in glsl
    )
    assert "int layer = (outer ? (inner ? dynamicLayer : 5) : 2);" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[5];\n        sampler samplers[5];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 5 and 6"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_ternary_index_alias_assignments_infer_size():
    crossgl = """
    shader TernaryIndexAliasAssignments {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(
            image2D images[] @rg32f,
            ivec2 pixel,
            bool outer,
            bool inner,
            int dynamicLayer
        ) {
            int layer = 0;
            layer = outer ? (inner ? dynamicLayer : 4) : 1;
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(bool outer, bool inner, int dynamicLayer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), outer, inner, dynamicLayer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[5] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u5);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[5], int2 pixel, "
        "bool outer, bool inner, int dynamicLayer)" in hlsl
    )
    assert "int layer = 0;" in hlsl
    assert "layer = (outer ? (inner ? dynamicLayer : 4) : 1);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 5> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(5)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 5> "
        "images, int2 pixel, bool outer, bool inner, int dynamicLayer)" in metal
    )
    assert "int layer = 0;" in metal
    assert "layer = outer ? inner ? dynamicLayer : 4 : 1;" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[5];" in glsl
    assert "layout(rg32f, binding = 5) uniform image2D afterImage;" in glsl
    assert (
        "vec2 readLayer(image2D images[5], ivec2 pixel, bool outer, "
        "bool inner, int dynamicLayer)" in glsl
    )
    assert "int layer = 0;" in glsl
    assert "layer = (outer ? (inner ? dynamicLayer : 4) : 1);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[4] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 4 and 5"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_branch_merged_index_aliases_infer_sampled_size():
    crossgl = """
    shader BranchMergedSampledIndexAliases {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(
            sampler2D textures[],
            sampler samplers[],
            vec2 uv,
            bool choose,
            bool inner,
            int dynamicLayer
        ) {
            int layer = 0;
            if (choose) {
                layer = inner ? dynamicLayer : 6;
            } else {
                layer = 3;
            }
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose, bool inner, int dynamicLayer) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose, inner, dynamicLayer) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert "SamplerState afterTextureSampler : register(s7);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, bool choose, bool inner, int dynamicLayer)" in hlsl
    )
    assert "layer = (inner ? dynamicLayer : 6);" in hlsl
    assert "layer = 3;" in hlsl
    assert "return textures[layer].Sample(samplers[layer], uv);" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[4]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, bool choose, bool inner, "
        "int dynamicLayer)" in metal
    )
    assert "layer = inner ? dynamicLayer : 6;" in metal
    assert "layer = 3;" in metal
    assert "return textures[layer].sample(samplers[layer], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert (
        "vec4 sampleLayer(sampler2D textures[7], vec2 uv, bool choose, "
        "bool inner, int dynamicLayer)" in glsl
    )
    assert "layer = (inner ? dynamicLayer : 6);" in glsl
    assert "layer = 3;" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[4]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_branch_merged_index_aliases_infer_image_size():
    crossgl = """
    shader BranchMergedImageIndexAliases {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(
            image2D images[] @rg32f,
            ivec2 pixel,
            bool choose,
            bool inner,
            int dynamicLayer
        ) {
            int layer = 2;
            if (choose) {
                layer = inner ? dynamicLayer : 5;
            }
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(bool choose, bool inner, int dynamicLayer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose, inner, dynamicLayer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, "
        "bool choose, bool inner, int dynamicLayer)" in hlsl
    )
    assert "int layer = 2;" in hlsl
    assert "layer = (inner ? dynamicLayer : 5);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[3]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, bool choose, bool inner, int dynamicLayer)" in metal
    )
    assert "int layer = 2;" in metal
    assert "layer = inner ? dynamicLayer : 5;" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 3> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert (
        "vec2 readLayer(image2D images[6], ivec2 pixel, bool choose, "
        "bool inner, int dynamicLayer)" in glsl
    )
    assert "int layer = 2;" in glsl
    assert "layer = (inner ? dynamicLayer : 5);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[3]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_switch_merged_index_aliases_infer_sampled_size():
    crossgl = """
    shader SwitchMergedSampledIndexAliases {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(
            sampler2D textures[],
            sampler samplers[],
            vec2 uv,
            int mode,
            bool choose,
            int dynamicLayer
        ) {
            int layer = 1;
            switch (mode) {
                case 0:
                    layer = choose ? dynamicLayer : 6;
                    break;
                case 1:
                    layer = 4;
                    break;
                default:
                    layer = 2;
                    break;
            }
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv, int mode, bool choose, int dynamicLayer) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, mode, choose, dynamicLayer) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert "SamplerState afterTextureSampler : register(s7);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, int mode, bool choose, int dynamicLayer)" in hlsl
    )
    assert "layer = (choose ? dynamicLayer : 6);" in hlsl
    assert "layer = 4;" in hlsl
    assert "layer = 2;" in hlsl
    assert "return textures[layer].Sample(samplers[layer], uv);" in hlsl
    assert "Texture2D textures[2]" not in hlsl
    assert "Texture2D textures[5]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, int mode, bool choose, "
        "int dynamicLayer)" in metal
    )
    assert "layer = choose ? dynamicLayer : 6;" in metal
    assert "layer = 4;" in metal
    assert "layer = 2;" in metal
    assert "return textures[layer].sample(samplers[layer], uv);" in metal
    assert "array<texture2d<float>, 2> textures" not in metal
    assert "array<texture2d<float>, 5> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert (
        "vec4 sampleLayer(sampler2D textures[7], vec2 uv, int mode, "
        "bool choose, int dynamicLayer)" in glsl
    )
    assert "layer = (choose ? dynamicLayer : 6);" in glsl
    assert "layer = 4;" in glsl
    assert "layer = 2;" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "sampler2D textures[2]" not in glsl
    assert "sampler2D textures[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_match_merged_index_aliases_infer_image_size():
    crossgl = """
    shader MatchMergedImageIndexAliases {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(
            image2D images[] @rg32f,
            ivec2 pixel,
            int mode,
            bool choose,
            int dynamicLayer
        ) {
            int layer = 1;
            match mode {
                0 => {
                    layer = choose ? dynamicLayer : 5;
                }
                _ => {
                    layer = 3;
                }
            }
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(int mode, bool choose, int dynamicLayer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), mode, choose, dynamicLayer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, int mode, "
        "bool choose, int dynamicLayer)" in hlsl
    )
    assert "layer = (choose ? dynamicLayer : 5);" in hlsl
    assert "layer = 3;" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[2]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[4]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, int mode, bool choose, int dynamicLayer)" in metal
    )
    assert "layer = choose ? dynamicLayer : 5;" in metal
    assert "layer = 3;" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 2> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 4> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert (
        "vec2 readLayer(image2D images[6], ivec2 pixel, int mode, "
        "bool choose, int dynamicLayer)" in glsl
    )
    assert "layer = (choose ? dynamicLayer : 5);" in glsl
    assert "layer = 3;" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[2]" not in glsl
    assert "image2D rgFloatImages[4]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_for_loop_carried_index_aliases_infer_sampled_size():
    crossgl = """
    shader ForLoopCarriedSampledIndexAliases {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(
            sampler2D textures[],
            sampler samplers[],
            vec2 uv,
            int limit,
            bool choose,
            int dynamicLayer
        ) {
            int layer = 1;
            for (int i = 0; i < limit; i = i + 1) {
                layer = choose ? dynamicLayer : 6;
            }
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv, int limit, bool choose, int dynamicLayer) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, limit, choose, dynamicLayer) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert "SamplerState afterTextureSampler : register(s7);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, int limit, bool choose, int dynamicLayer)" in hlsl
    )
    assert "for (int i = 0; (i < limit); i = (i + 1))" in hlsl
    assert "layer = (choose ? dynamicLayer : 6);" in hlsl
    assert "return textures[layer].Sample(samplers[layer], uv);" in hlsl
    assert "Texture2D textures[2]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, int limit, bool choose, "
        "int dynamicLayer)" in metal
    )
    assert "for (int i = 0; i < limit; i = i + 1)" in metal
    assert "layer = choose ? dynamicLayer : 6;" in metal
    assert "return textures[layer].sample(samplers[layer], uv);" in metal
    assert "array<texture2d<float>, 2> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert (
        "vec4 sampleLayer(sampler2D textures[7], vec2 uv, int limit, "
        "bool choose, int dynamicLayer)" in glsl
    )
    assert "for (int i = 0; (i < limit); i = (i + 1))" in glsl
    assert "layer = (choose ? dynamicLayer : 6);" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "sampler2D textures[2]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_while_do_loop_carried_index_aliases_infer_image_size():
    crossgl = """
    shader WhileDoLoopCarriedImageIndexAliases {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(
            image2D images[] @rg32f,
            ivec2 pixel,
            int limit,
            bool choose,
            int dynamicLayer
        ) {
            int layer = 1;
            int i = 0;
            while (i < limit) {
                layer = choose ? dynamicLayer : 4;
                i = i + 1;
            }
            do {
                layer = choose ? layer : 5;
            } while (false);
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(int limit, bool choose, int dynamicLayer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), limit, choose, dynamicLayer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, int limit, "
        "bool choose, int dynamicLayer)" in hlsl
    )
    assert "while ((i < limit))" in hlsl
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "layer = (choose ? dynamicLayer : 4);" in hlsl
    assert "layer = (choose ? layer : 5);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[2]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, int limit, bool choose, int dynamicLayer)" in metal
    )
    assert "while (i < limit)" in metal
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "layer = choose ? dynamicLayer : 4;" in metal
    assert "layer = choose ? layer : 5;" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 2> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert (
        "vec2 readLayer(image2D images[6], ivec2 pixel, int limit, "
        "bool choose, int dynamicLayer)" in glsl
    )
    assert "while ((i < limit))" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "layer = (choose ? dynamicLayer : 4);" in glsl
    assert "layer = (choose ? layer : 5);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[2]" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_helper_return_index_aliases_infer_sampled_size():
    crossgl = """
    shader HelperReturnSampledIndexAliases {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        int pickLayer(bool choose, int dynamicLayer) {
            return choose ? dynamicLayer : 6;
        }

        vec4 sampleLayer(
            sampler2D textures[],
            sampler samplers[],
            vec2 uv,
            bool choose,
            int dynamicLayer
        ) {
            int layer = pickLayer(choose, dynamicLayer);
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv, bool choose, int dynamicLayer) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, choose, dynamicLayer) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert "SamplerState afterTextureSampler : register(s7);" in hlsl
    assert "int pickLayer(bool choose, int dynamicLayer)" in hlsl
    assert "return (choose ? dynamicLayer : 6);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, bool choose, int dynamicLayer)" in hlsl
    )
    assert "int layer = pickLayer(choose, dynamicLayer);" in hlsl
    assert "return textures[layer].Sample(samplers[layer], uv);" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert "int pickLayer(bool choose, int dynamicLayer)" in metal
    assert "return choose ? dynamicLayer : 6;" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, bool choose, int dynamicLayer)" in metal
    )
    assert "int layer = pickLayer(choose, dynamicLayer);" in metal
    assert "return textures[layer].sample(samplers[layer], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert "int pickLayer(bool choose, int dynamicLayer)" in glsl
    assert "return (choose ? dynamicLayer : 6);" in glsl
    assert (
        "vec4 sampleLayer(sampler2D textures[7], vec2 uv, bool choose, "
        "int dynamicLayer)" in glsl
    )
    assert "int layer = pickLayer(choose, dynamicLayer);" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_nested_helper_return_index_aliases_infer_image_size():
    crossgl = """
    shader NestedHelperReturnImageIndexAliases {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int baseLayer(bool choose, int dynamicLayer) {
            return choose ? dynamicLayer : 4;
        }

        int pickLayer(bool choose, int dynamicLayer) {
            int layer = baseLayer(choose, dynamicLayer);
            return choose ? layer : 5;
        }

        vec2 readLayer(
            image2D images[] @rg32f,
            ivec2 pixel,
            bool choose,
            int dynamicLayer
        ) {
            int layer = pickLayer(choose, dynamicLayer);
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(bool choose, int dynamicLayer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose, dynamicLayer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert "int baseLayer(bool choose, int dynamicLayer)" in hlsl
    assert "return (choose ? dynamicLayer : 4);" in hlsl
    assert "int pickLayer(bool choose, int dynamicLayer)" in hlsl
    assert "int layer = baseLayer(choose, dynamicLayer);" in hlsl
    assert "return (choose ? layer : 5);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, "
        "bool choose, int dynamicLayer)" in hlsl
    )
    assert "int layer = pickLayer(choose, dynamicLayer);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert "int baseLayer(bool choose, int dynamicLayer)" in metal
    assert "return choose ? dynamicLayer : 4;" in metal
    assert "int pickLayer(bool choose, int dynamicLayer)" in metal
    assert "int layer = baseLayer(choose, dynamicLayer);" in metal
    assert "return choose ? layer : 5;" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, bool choose, int dynamicLayer)" in metal
    )
    assert "int layer = pickLayer(choose, dynamicLayer);" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert "int baseLayer(bool choose, int dynamicLayer)" in glsl
    assert "return (choose ? dynamicLayer : 4);" in glsl
    assert "int pickLayer(bool choose, int dynamicLayer)" in glsl
    assert "int layer = baseLayer(choose, dynamicLayer);" in glsl
    assert "return (choose ? layer : 5);" in glsl
    assert (
        "vec2 readLayer(image2D images[6], ivec2 pixel, bool choose, "
        "int dynamicLayer)" in glsl
    )
    assert "int layer = pickLayer(choose, dynamicLayer);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_control_flow_helper_return_literal_args_infer_sampled_size():
    crossgl = """
    shader ControlFlowHelperReturnLiteralArgsSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        int pickLayer(int mode, int fallbackLayer) {
            if (mode == 0) {
                return fallbackLayer;
            }
            switch (mode) {
                case 1:
                    return 4;
                default:
                    return 2;
            }
        }

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            int layer = pickLayer(mode, 6);
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, mode) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert "SamplerState afterTextureSampler : register(s7);" in hlsl
    assert "int pickLayer(int mode, int fallbackLayer)" in hlsl
    assert "return fallbackLayer;" in hlsl
    assert "return 4;" in hlsl
    assert "return 2;" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, int mode)" in hlsl
    )
    assert "int layer = pickLayer(mode, 6);" in hlsl
    assert "return textures[layer].Sample(samplers[layer], uv);" in hlsl
    assert "Texture2D textures[3]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert "int pickLayer(int mode, int fallbackLayer)" in metal
    assert "return fallbackLayer;" in metal
    assert "return 4;" in metal
    assert "return 2;" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, int mode)" in metal
    )
    assert "int layer = pickLayer(mode, 6);" in metal
    assert "return textures[layer].sample(samplers[layer], uv);" in metal
    assert "array<texture2d<float>, 3> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert "int pickLayer(int mode, int fallbackLayer)" in glsl
    assert "return fallbackLayer;" in glsl
    assert "return 4;" in glsl
    assert "return 2;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[7], vec2 uv, int mode)" in glsl
    assert "int layer = pickLayer(mode, 6);" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "sampler2D textures[3]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_match_helper_return_literal_args_infer_image_size():
    crossgl = """
    shader MatchHelperReturnLiteralArgsImage {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int pickLayer(int mode, int fallbackLayer) {
            match mode {
                0 => { return fallbackLayer; }
                1 => { return 3; }
                _ => { return 1; }
            }
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, int mode) {
            int layer = pickLayer(mode, 5);
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(int mode) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), mode);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert "int pickLayer(int mode, int fallbackLayer)" in hlsl
    assert "return fallbackLayer;" in hlsl
    assert "return 3;" in hlsl
    assert "return 1;" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, int mode)" in hlsl
    )
    assert "int layer = pickLayer(mode, 5);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[2]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert "int pickLayer(int mode, int fallbackLayer)" in metal
    assert "return fallbackLayer;" in metal
    assert "return 3;" in metal
    assert "return 1;" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, int mode)" in metal
    )
    assert "int layer = pickLayer(mode, 5);" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 2> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert "int pickLayer(int mode, int fallbackLayer)" in glsl
    assert "return fallbackLayer;" in glsl
    assert "return 3;" in glsl
    assert "return 1;" in glsl
    assert "vec2 readLayer(image2D images[6], ivec2 pixel, int mode)" in glsl
    assert "int layer = pickLayer(mode, 5);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[2]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_call_site_arithmetic_scalar_args_infer_sampled_size():
    crossgl = """
    shader CallSiteArithmeticScalarArgsSampled {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        int bumpLayer(int layer) {
            return layer + 1;
        }

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, int baseLayer) {
            int shifted = bumpLayer(baseLayer);
            return texture(textures[shifted], samplers[shifted], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, 5) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert "int bumpLayer(int layer)" in hlsl
    assert "return (layer + 1);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, int baseLayer)" in hlsl
    )
    assert "int shifted = bumpLayer(baseLayer);" in hlsl
    assert "return textures[shifted].Sample(samplers[shifted], uv);" in hlsl
    assert "sampleLayer(textures, samplers, uv, 5)" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert "int bumpLayer(int layer)" in metal
    assert "return layer + 1;" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, int baseLayer)" in metal
    )
    assert "int shifted = bumpLayer(baseLayer);" in metal
    assert "return textures[shifted].sample(samplers[shifted], uv);" in metal
    assert "sampleLayer(textures, samplers, uv, 5)" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert "int bumpLayer(int layer)" in glsl
    assert "return (layer + 1);" in glsl
    assert "vec4 sampleLayer(sampler2D textures[7], vec2 uv, int baseLayer)" in glsl
    assert "int shifted = bumpLayer(baseLayer);" in glsl
    assert "return texture(textures[shifted], uv);" in glsl
    assert "sampleLayer(textures, uv, 5)" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_arithmetic_alias_helper_return_infers_image_size():
    crossgl = """
    shader ArithmeticAliasHelperImageIndices {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        int offsetLayer(int baseLayer, bool choose, int dynamicLayer) {
            int fallback = choose ? dynamicLayer : baseLayer;
            return fallback + 2;
        }

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, bool choose, int dynamicLayer) {
            int layer = offsetLayer(5, choose, dynamicLayer);
            return imageLoad(images[layer], pixel);
        }

        compute {
            void main(bool choose, int dynamicLayer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), choose, dynamicLayer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[8] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u8);" in hlsl
    assert "int offsetLayer(int baseLayer, bool choose, int dynamicLayer)" in hlsl
    assert "int fallback = (choose ? dynamicLayer : baseLayer);" in hlsl
    assert "return (fallback + 2);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[8], int2 pixel, "
        "bool choose, int dynamicLayer)" in hlsl
    )
    assert "int layer = offsetLayer(5, choose, dynamicLayer);" in hlsl
    assert "return images[layer][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[6]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 8> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(8)]]" in metal
    assert "int offsetLayer(int baseLayer, bool choose, int dynamicLayer)" in metal
    assert "int fallback = choose ? dynamicLayer : baseLayer;" in metal
    assert "return fallback + 2;" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 8> "
        "images, int2 pixel, bool choose, int dynamicLayer)" in metal
    )
    assert "int layer = offsetLayer(5, choose, dynamicLayer);" in metal
    assert "return images[layer].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 6> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[8];" in glsl
    assert "layout(rg32f, binding = 8) uniform image2D afterImage;" in glsl
    assert "int offsetLayer(int baseLayer, bool choose, int dynamicLayer)" in glsl
    assert "int fallback = (choose ? dynamicLayer : baseLayer);" in glsl
    assert "return (fallback + 2);" in glsl
    assert (
        "vec2 readLayer(image2D images[8], ivec2 pixel, bool choose, int dynamicLayer)"
        in glsl
    )
    assert "int layer = offsetLayer(5, choose, dynamicLayer);" in glsl
    assert "return imageLoad(images[layer], pixel).xy;" in glsl
    assert "image2D rgFloatImages[1]" not in glsl
    assert "image2D rgFloatImages[6]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[7] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 7 and 8"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_builtin_min_index_infers_sampled_size():
    crossgl = """
    shader BuiltinMinSampledIndex {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv, int layer) {
            int bounded = min(layer, 6);
            return texture(textures[bounded], samplers[bounded], uv);
        }

        fragment {
            vec4 main(vec2 uv, int layer) @ gl_FragColor {
                return sampleLayer(textures, samplers, uv, layer) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[7] : register(t0);" in hlsl
    assert "SamplerState samplers[7] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t7);" in hlsl
    assert (
        "float4 sampleLayer(Texture2D textures[7], SamplerState samplers[7], "
        "float2 uv, int layer)" in hlsl
    )
    assert "int bounded = min(layer, 6);" in hlsl
    assert "return textures[bounded].Sample(samplers[bounded], uv);" in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[1]" not in hlsl

    assert "array<texture2d<float>, 7> textures [[texture(0)]]" in metal
    assert "array<sampler, 7> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(7)]]" in metal
    assert (
        "float4 sampleLayer(array<texture2d<float>, 7> textures, "
        "array<sampler, 7> samplers, float2 uv, int layer)" in metal
    )
    assert "int bounded = min(layer, 6);" in metal
    assert "return textures[bounded].sample(samplers[bounded], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[7];" in glsl
    assert "layout(binding = 7) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleLayer(sampler2D textures[7], vec2 uv, int layer)" in glsl
    assert "int bounded = min(layer, 6);" in glsl
    assert "return texture(textures[bounded], uv);" in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[6];\n        sampler samplers[6];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 6 and 7"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_builtin_clamp_index_infers_image_size():
    crossgl = """
    shader BuiltinClampImageIndex {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 readLayer(image2D images[] @rg32f, ivec2 pixel, int layer) {
            int bounded = clamp(layer, 0, 5);
            return imageLoad(images[bounded], pixel);
        }

        compute {
            void main(int layer) {
                vec2 result = readLayer(rgFloatImages, ivec2(0, 1), layer);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[6] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u6);" in hlsl
    assert (
        "float2 readLayer(RWTexture2D<float2> images[6], int2 pixel, int layer)" in hlsl
    )
    assert "int bounded = clamp(layer, 0, 5);" in hlsl
    assert "return images[bounded][pixel];" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[1]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 6> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(6)]]" in metal
    assert (
        "float2 readLayer(array<texture2d<float, access::read_write>, 6> "
        "images, int2 pixel, int layer)" in metal
    )
    assert "int bounded = clamp(layer, 0, 5);" in metal
    assert "return images[bounded].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[6];" in glsl
    assert "layout(rg32f, binding = 6) uniform image2D afterImage;" in glsl
    assert "vec2 readLayer(image2D images[6], ivec2 pixel, int layer)" in glsl
    assert "int bounded = clamp(layer, 0, 5);" in glsl
    assert "return imageLoad(images[bounded], pixel).xy;" in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[1]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[5] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 5 and 6"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_switch_match_unsized_sampled_arrays_infer_case_literal_size():
    crossgl = """
    shader SwitchMatchUnsizedSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int layer @ TEXCOORD1;
            int mode @ TEXCOORD2;
        };

        vec4 sampleCases(sampler2D textures[], sampler samplers[], int layer, vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    vec4 scoped = texture(textures[layer], samplers[layer], uv);
                    result = scoped;
                    break;
                case 1:
                    result = texture(textures[2], samplers[2], uv);
                    break;
                default:
                    result = texture(textures[4], samplers[4], uv);
                    break;
            }
            match mode {
                0 => {
                    result = result + texture(textures[layer], samplers[layer], uv);
                }
                _ => {
                    result = result + texture(textures[3], samplers[3], uv);
                }
            }
            return result;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleCases(textures, samplers, input.layer, input.uv, input.mode) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert "SamplerState afterTextureSampler : register(s5);" in hlsl
    assert (
        "float4 sampleCases(Texture2D textures[5], SamplerState samplers[5], "
        "int layer, float2 uv, int mode)" in hlsl
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in hlsl
    assert "textures[4].Sample(samplers[4], uv)" in hlsl
    assert "textures[3].Sample(samplers[3], uv)" in hlsl
    assert "case 0: {" in hlsl
    assert "Texture2D textures[] : register(t0);" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 sampleCases(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, int layer, float2 uv, int mode)" in metal
    )
    assert "textures[layer].sample(samplers[layer], uv)" in metal
    assert "textures[4].sample(samplers[4], uv)" in metal
    assert "textures[3].sample(samplers[3], uv)" in metal
    assert "case 0: {" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert (
        "vec4 sampleCases(sampler2D textures[5], int layer, vec2 uv, int mode)" in glsl
    )
    assert "texture(textures[layer], uv)" in glsl
    assert "texture(textures[4], uv)" in glsl
    assert "texture(textures[3], uv)" in glsl
    assert "case 0: {" in glsl
    assert "sampler2D textures[]" not in glsl


def test_codegen_switch_match_unsized_image_arrays_infer_case_literal_size():
    crossgl = """
    shader SwitchMatchUnsizedImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 imageCases(image2D images[] @rg32f, int layer, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    vec2 scoped = imageLoad(images[layer], pixel);
                    result = scoped;
                    break;
                case 1:
                    result = imageLoad(images[1], pixel);
                    break;
                default:
                    result = imageLoad(images[3], pixel);
                    break;
            }
            match mode {
                0 => {
                    result = result + imageLoad(images[layer], pixel);
                }
                _ => {
                    result = result + imageLoad(images[2], pixel);
                }
            }
            imageStore(images[0], pixel, result);
            return result;
        }

        compute {
            void main() {
                vec2 result = imageCases(rgFloatImages, 1, ivec2(0, 1), 2);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert (
        "float2 imageCases(RWTexture2D<float2> images[4], int layer, "
        "int2 pixel, int mode)" in hlsl
    )
    assert "images[layer][pixel]" in hlsl
    assert "images[3][pixel]" in hlsl
    assert "images[2][pixel]" in hlsl
    assert "case 0: {" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 imageCases(array<texture2d<float, access::read_write>, 4> "
        "images, int layer, int2 pixel, int mode)" in metal
    )
    assert "images[layer].read(uint2(pixel)).xy" in metal
    assert "images[3].read(uint2(pixel)).xy" in metal
    assert "images[2].read(uint2(pixel)).xy" in metal
    assert "case 0: {" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert (
        "vec2 imageCases(image2D images[4], int layer, ivec2 pixel, int mode)" in glsl
    )
    assert "imageLoad(images[layer], pixel).xy" in glsl
    assert "imageLoad(images[3], pixel).xy" in glsl
    assert "imageLoad(images[2], pixel).xy" in glsl
    assert "case 0: {" in glsl
    assert "image2D images[]" not in glsl


def test_codegen_switch_match_sampled_array_returns_infer_case_literal_size():
    crossgl = """
    shader SwitchMatchReturnSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int mode @ TEXCOORD1;
        };

        vec4 sampleReturns(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    return texture(textures[4], samplers[4], uv);
                case 1:
                    result = texture(textures[2], samplers[2], uv);
                    break;
                default:
                    break;
            }
            match mode {
                2 => {
                    return result + texture(textures[3], samplers[3], uv);
                }
                _ => {
                    result = result + texture(textures[1], samplers[1], uv);
                }
            }
            return result;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleReturns(textures, samplers, input.uv, input.mode) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 sampleReturns(Texture2D textures[5], "
        "SamplerState samplers[5], float2 uv, int mode)" in hlsl
    )
    assert "case 0: {" in hlsl
    assert "return textures[4].Sample(samplers[4], uv);" in hlsl
    assert "result = textures[2].Sample(samplers[2], uv);" in hlsl
    assert "return (result + textures[3].Sample(samplers[3], uv));" in hlsl
    assert "result = (result + textures[1].Sample(samplers[1], uv));" in hlsl
    assert "Texture2D textures[] : register(t0);" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 sampleReturns(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode)" in metal
    )
    assert "case 0: {" in metal
    assert "return textures[4].sample(samplers[4], uv);" in metal
    assert "result = textures[2].sample(samplers[2], uv);" in metal
    assert "return result + textures[3].sample(samplers[3], uv);" in metal
    assert "result = result + textures[1].sample(samplers[1], uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 sampleReturns(sampler2D textures[5], vec2 uv, int mode)" in glsl
    assert "case 0: {" in glsl
    assert "return texture(textures[4], uv);" in glsl
    assert "result = texture(textures[2], uv);" in glsl
    assert "return (result + texture(textures[3], uv));" in glsl
    assert "result = (result + texture(textures[1], uv));" in glsl
    assert "sampler2D textures[]" not in glsl


def test_codegen_switch_match_image_array_returns_infer_case_literal_size():
    crossgl = """
    shader SwitchMatchReturnImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 imageReturns(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    return imageLoad(images[3], pixel);
                case 1:
                    result = imageLoad(images[1], pixel);
                    break;
                default:
                    break;
            }
            match mode {
                2 => {
                    return result + imageLoad(images[2], pixel);
                }
                _ => {
                    result = result + imageLoad(images[0], pixel);
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = imageReturns(rgFloatImages, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert (
        "float2 imageReturns(RWTexture2D<float2> images[4], int2 pixel, int mode)"
        in hlsl
    )
    assert "case 0: {" in hlsl
    assert "return images[3][pixel];" in hlsl
    assert "result = images[1][pixel];" in hlsl
    assert "return (result + images[2][pixel]);" in hlsl
    assert "result = (result + images[0][pixel]);" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 imageReturns(array<texture2d<float, access::read_write>, 4> "
        "images, int2 pixel, int mode)" in metal
    )
    assert "case 0: {" in metal
    assert "return images[3].read(uint2(pixel)).xy;" in metal
    assert "result = images[1].read(uint2(pixel)).xy;" in metal
    assert "return result + images[2].read(uint2(pixel)).xy;" in metal
    assert "result = result + images[0].read(uint2(pixel)).xy;" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 imageReturns(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert "case 0: {" in glsl
    assert "return imageLoad(images[3], pixel).xy;" in glsl
    assert "result = imageLoad(images[1], pixel).xy;" in glsl
    assert "return (result + imageLoad(images[2], pixel).xy);" in glsl
    assert "result = (result + imageLoad(images[0], pixel).xy);" in glsl
    assert "image2D rgFloatImages[]" not in glsl


def test_codegen_switch_match_loop_transitive_sampled_arrays_infer_case_only_size():
    crossgl = """
    shader SwitchMatchLoopTransitiveSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int mode @ TEXCOORD1;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 result = vec4(0.0);
            for (int i = 0; i < 1; i = i + 1) {
                result = result + texture(textures[4], samplers[4], uv);
            }
            int j = 0;
            while (j < 1) {
                result = result + texture(textures[3], samplers[3], uv);
                j = j + 1;
            }
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(textures, samplers, uv);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return mid(textures, samplers, input.uv, input.mode) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode)" in hlsl
    )
    assert "for (int i = 0; (i < 1); i = (i + 1))" in hlsl
    assert "while ((j < 1))" in hlsl
    assert "result = (result + textures[4].Sample(samplers[4], uv));" in hlsl
    assert "result = (result + textures[3].Sample(samplers[3], uv));" in hlsl
    assert "result = leaf(textures, samplers, uv);" in hlsl
    assert "result = (result + leaf(textures, samplers, uv));" in hlsl
    assert "Texture2D textures[] : register(t0);" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode)" in metal
    )
    assert "for (int i = 0; i < 1; i = i + 1)" in metal
    assert "while (j < 1)" in metal
    assert "result = result + textures[4].sample(samplers[4], uv);" in metal
    assert "result = result + textures[3].sample(samplers[3], uv);" in metal
    assert "result = leaf(textures, samplers, uv);" in metal
    assert "result = result + leaf(textures, samplers, uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 leaf(sampler2D textures[5], vec2 uv)" in glsl
    assert "vec4 mid(sampler2D textures[5], vec2 uv, int mode)" in glsl
    assert "for (int i = 0; (i < 1); i = (i + 1))" in glsl
    assert "while ((j < 1))" in glsl
    assert "result = (result + texture(textures[4], uv));" in glsl
    assert "result = (result + texture(textures[3], uv));" in glsl
    assert "result = leaf(textures, uv);" in glsl
    assert "result = (result + leaf(textures, uv));" in glsl
    assert "sampler2D textures[]" not in glsl


def test_codegen_switch_match_loop_transitive_image_arrays_infer_case_only_size():
    crossgl = """
    shader SwitchMatchLoopTransitiveImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel) {
            vec2 result = vec2(0.0);
            for (int i = 0; i < 1; i = i + 1) {
                result = result + imageLoad(images[3], pixel);
            }
            int j = 0;
            while (j < 1) {
                result = result + imageLoad(images[2], pixel);
                j = j + 1;
            }
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(images, pixel);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "float2 mid(RWTexture2D<float2> images[4], int2 pixel, int mode)" in hlsl
    assert "for (int i = 0; (i < 1); i = (i + 1))" in hlsl
    assert "while ((j < 1))" in hlsl
    assert "result = (result + images[3][pixel]);" in hlsl
    assert "result = (result + images[2][pixel]);" in hlsl
    assert "result = leaf(images, pixel);" in hlsl
    assert "result = (result + leaf(images, pixel));" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode)" in metal
    )
    assert "for (int i = 0; i < 1; i = i + 1)" in metal
    assert "while (j < 1)" in metal
    assert "result = result + images[3].read(uint2(pixel)).xy;" in metal
    assert "result = result + images[2].read(uint2(pixel)).xy;" in metal
    assert "result = leaf(images, pixel);" in metal
    assert "result = result + leaf(images, pixel);" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert "for (int i = 0; (i < 1); i = (i + 1))" in glsl
    assert "while ((j < 1))" in glsl
    assert "result = (result + imageLoad(images[3], pixel).xy);" in glsl
    assert "result = (result + imageLoad(images[2], pixel).xy);" in glsl
    assert "result = leaf(images, pixel);" in glsl
    assert "result = (result + leaf(images, pixel));" in glsl
    assert "image2D rgFloatImages[]" not in glsl


def test_codegen_switch_match_fixed_loop_sampled_transitive_conflicts_are_rejected():
    crossgl = """
    shader SwitchMatchFixedLoopTransitiveSampledConflict {
        sampler2D smallTextures[2];
        sampler smallSamplers[2];
        sampler2D largeTextures[5];
        sampler largeSamplers[5];

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 result = vec4(0.0);
            for (int i = 0; i < 1; i = i + 1) {
                result = result + texture(textures[4], samplers[4], uv);
            }
            while (false) {
                result = result + texture(textures[3], samplers[3], uv);
            }
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(textures, samplers, uv);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return mid(smallTextures, smallSamplers, uv, mode) + mid(largeTextures, largeSamplers, uv, mode);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'smallTextures': 2 and 5"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_fixed_loop_image_transitive_conflicts_are_rejected():
    crossgl = """
    shader SwitchMatchFixedLoopTransitiveImageConflict {
        image2D smallImages[2] @rg32f;
        image2D largeImages[4] @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel) {
            vec2 result = vec2(0.0);
            for (int i = 0; i < 1; i = i + 1) {
                result = result + imageLoad(images[3], pixel);
            }
            while (false) {
                result = result + imageLoad(images[2], pixel);
            }
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(images, pixel);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 a = mid(smallImages, ivec2(0, 1), 0);
                vec2 b = mid(largeImages, ivec2(1, 2), 1);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'smallImages': 2 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_for_in_transitive_sampled_arrays_infer_size():
    crossgl = """
    shader SwitchMatchForInTransitiveSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int mode @ TEXCOORD1;
            int limit @ TEXCOORD2;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv, int limit) {
            vec4 result = vec4(0.0);
            for i in 0..1 {
                result = result + texture(textures[4], samplers[4], uv);
            }
            for j in limit {
                result = result + texture(textures[3], samplers[3], uv);
            }
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode, int limit) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(textures, samplers, uv, limit);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return mid(textures, samplers, input.uv, input.mode, input.limit) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int limit)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode, int limit)" in hlsl
    )
    assert "for (int i = 0; i < 1; ++i)" in hlsl
    assert "for (int j = 0; j < limit; ++j)" in hlsl
    assert "result = (result + textures[4].Sample(samplers[4], uv));" in hlsl
    assert "result = (result + textures[3].Sample(samplers[3], uv));" in hlsl
    assert "result = leaf(textures, samplers, uv, limit);" in hlsl
    assert "result = (result + leaf(textures, samplers, uv, limit));" in hlsl
    assert "Texture2D textures[] : register(t0);" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int limit)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode, int limit)" in metal
    )
    assert "for (int i = 0; i < 1; ++i)" in metal
    assert "for (int j = 0; j < limit; ++j)" in metal
    assert "result = result + textures[4].sample(samplers[4], uv);" in metal
    assert "result = result + textures[3].sample(samplers[3], uv);" in metal
    assert "result = leaf(textures, samplers, uv, limit);" in metal
    assert "result = result + leaf(textures, samplers, uv, limit);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 leaf(sampler2D textures[5], vec2 uv, int limit)" in glsl
    assert "vec4 mid(sampler2D textures[5], vec2 uv, int mode, int limit)" in glsl
    assert "for (int i = 0; i < 1; ++i)" in glsl
    assert "for (int j = 0; j < limit; ++j)" in glsl
    assert "result = (result + texture(textures[4], uv));" in glsl
    assert "result = (result + texture(textures[3], uv));" in glsl
    assert "result = leaf(textures, uv, limit);" in glsl
    assert "result = (result + leaf(textures, uv, limit));" in glsl
    assert "sampler2D textures[]" not in glsl


def test_codegen_switch_match_for_in_transitive_image_arrays_infer_size():
    crossgl = """
    shader SwitchMatchForInTransitiveImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel, int limit) {
            vec2 result = vec2(0.0);
            for i in 0..1 {
                result = result + imageLoad(images[3], pixel);
            }
            for j in limit {
                result = result + imageLoad(images[2], pixel);
            }
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode, int limit) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(images, pixel, limit);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1), 0, 2);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel, int limit)" in hlsl
    assert (
        "float2 mid(RWTexture2D<float2> images[4], int2 pixel, int mode, "
        "int limit)" in hlsl
    )
    assert "for (int i = 0; i < 1; ++i)" in hlsl
    assert "for (int j = 0; j < limit; ++j)" in hlsl
    assert "result = (result + images[3][pixel]);" in hlsl
    assert "result = (result + images[2][pixel]);" in hlsl
    assert "result = leaf(images, pixel, limit);" in hlsl
    assert "result = (result + leaf(images, pixel, limit));" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int limit)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode, int limit)" in metal
    )
    assert "for (int i = 0; i < 1; ++i)" in metal
    assert "for (int j = 0; j < limit; ++j)" in metal
    assert "result = result + images[3].read(uint2(pixel)).xy;" in metal
    assert "result = result + images[2].read(uint2(pixel)).xy;" in metal
    assert "result = leaf(images, pixel, limit);" in metal
    assert "result = result + leaf(images, pixel, limit);" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel, int limit)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel, int mode, int limit)" in glsl
    assert "for (int i = 0; i < 1; ++i)" in glsl
    assert "for (int j = 0; j < limit; ++j)" in glsl
    assert "result = (result + imageLoad(images[3], pixel).xy);" in glsl
    assert "result = (result + imageLoad(images[2], pixel).xy);" in glsl
    assert "result = leaf(images, pixel, limit);" in glsl
    assert "result = (result + leaf(images, pixel, limit));" in glsl
    assert "image2D rgFloatImages[]" not in glsl


def test_codegen_switch_match_fixed_for_in_sampled_transitive_conflicts_are_rejected():
    crossgl = """
    shader SwitchMatchFixedForInTransitiveSampledConflict {
        sampler2D smallTextures[2];
        sampler smallSamplers[2];
        sampler2D largeTextures[5];
        sampler largeSamplers[5];

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv, int limit) {
            vec4 result = vec4(0.0);
            for i in 0..1 {
                result = result + texture(textures[4], samplers[4], uv);
            }
            for j in limit {
                result = result + texture(textures[3], samplers[3], uv);
            }
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode, int limit) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(textures, samplers, uv, limit);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return mid(smallTextures, smallSamplers, uv, mode, 2) + mid(largeTextures, largeSamplers, uv, mode, 2);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'smallTextures': 2 and 5"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_fixed_for_in_image_transitive_conflicts_are_rejected():
    crossgl = """
    shader SwitchMatchFixedForInTransitiveImageConflict {
        image2D smallImages[2] @rg32f;
        image2D largeImages[4] @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel, int limit) {
            vec2 result = vec2(0.0);
            for i in 0..1 {
                result = result + imageLoad(images[3], pixel);
            }
            for j in limit {
                result = result + imageLoad(images[2], pixel);
            }
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode, int limit) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(images, pixel, limit);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 a = mid(smallImages, ivec2(0, 1), 0, 2);
                vec2 b = mid(largeImages, ivec2(1, 2), 1, 2);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'smallImages': 2 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_for_in_sampled_shadowed_const_restores_after_loop():
    safe_crossgl = """
    shader ForInShadowedConstSampledSafe {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        vec4 sampleLoop(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            vec4 result = vec4(0.0);
            for COUNT in 0..1 {
                result = result + texture(textures[COUNT], samplers[COUNT], uv);
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLoop(textures, samplers, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert (
        "float4 sampleLoop(Texture2D textures[4], SamplerState samplers[4], "
        "float2 uv)" in hlsl
    )
    assert "for (int COUNT = 0; COUNT < 1; ++COUNT)" in hlsl
    assert "Texture2D textures[5]" not in hlsl
    assert "SamplerState samplers[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert (
        "float4 sampleLoop(array<texture2d<float>, 4> textures, "
        "array<sampler, 4> samplers, float2 uv)" in metal
    )
    assert "for (int COUNT = 0; COUNT < 1; ++COUNT)" in metal
    assert "array<texture2d<float>, 5> textures" not in metal
    assert "array<sampler, 5> samplers" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "vec4 sampleLoop(sampler2D textures[4], vec2 uv)" in glsl
    assert "for (int COUNT = 0; COUNT < 1; ++COUNT)" in glsl
    assert "sampler2D textures[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + texture(textures[COUNT], samplers[COUNT], uv);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_for_in_image_shadowed_const_restores_after_loop():
    safe_crossgl = """
    shader ForInShadowedConstImageSafe {
        const int COUNT = 4;
        image2D rgFloatImages[4] @rg32f;

        vec2 imageLoop(image2D images[4] @rg32f, ivec2 pixel) {
            vec2 result = vec2(0.0);
            for COUNT in 0..1 {
                result = result + imageLoad(images[COUNT], pixel);
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = imageLoop(rgFloatImages, ivec2(0, 1));
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "float2 imageLoop(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "for (int COUNT = 0; COUNT < 1; ++COUNT)" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert (
        "float2 imageLoop(array<texture2d<float, access::read_write>, 4> "
        "images, int2 pixel)" in metal
    )
    assert "for (int COUNT = 0; COUNT < 1; ++COUNT)" in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "vec2 imageLoop(image2D images[4], ivec2 pixel)" in glsl
    assert "for (int COUNT = 0; COUNT < 1; ++COUNT)" in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + imageLoad(images[COUNT], pixel);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'images': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_for_in_switch_match_sampled_shadowed_const_restores_after_loop():
    safe_crossgl = """
    shader ForInSwitchMatchShadowedSampledSafe {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        vec4 sampleNested(sampler2D textures[4], sampler samplers[4], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            for COUNT in 0..1 {
                switch (mode) {
                    case 0:
                        result = result + texture(textures[COUNT], samplers[COUNT], uv);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + texture(textures[COUNT], samplers[COUNT], uv);
                    }
                    _ => {
                    }
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return sampleNested(textures, samplers, uv, mode);
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert (
        "float4 sampleNested(Texture2D textures[4], SamplerState samplers[4], "
        "float2 uv, int mode)" in hlsl
    )
    assert "case 0: {" in hlsl
    assert hlsl.count("textures[COUNT].Sample(samplers[COUNT], uv)") == 2
    assert "Texture2D textures[5]" not in hlsl
    assert "SamplerState samplers[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert (
        "float4 sampleNested(array<texture2d<float>, 4> textures, "
        "array<sampler, 4> samplers, float2 uv, int mode)" in metal
    )
    assert "case 0: {" in metal
    assert metal.count("textures[COUNT].sample(samplers[COUNT], uv)") == 2
    assert "array<texture2d<float>, 5> textures" not in metal
    assert "array<sampler, 5> samplers" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "vec4 sampleNested(sampler2D textures[4], vec2 uv, int mode)" in glsl
    assert "case 0: {" in glsl
    assert glsl.count("texture(textures[COUNT], uv)") == 2
    assert "sampler2D textures[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + texture(textures[COUNT], samplers[COUNT], uv);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_for_in_switch_match_image_shadowed_const_restores_after_loop():
    safe_crossgl = """
    shader ForInSwitchMatchShadowedImageSafe {
        const int COUNT = 4;
        image2D rgFloatImages[4] @rg32f;

        vec2 imageNested(image2D images[4] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            for COUNT in 0..1 {
                switch (mode) {
                    case 0:
                        result = result + imageLoad(images[COUNT], pixel);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + imageLoad(images[COUNT], pixel);
                    }
                    _ => {
                    }
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = imageNested(rgFloatImages, ivec2(0, 1), 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert (
        "float2 imageNested(RWTexture2D<float2> images[4], int2 pixel, int mode)"
        in hlsl
    )
    assert "case 0: {" in hlsl
    assert hlsl.count("images[COUNT][pixel]") == 2
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert (
        "float2 imageNested(array<texture2d<float, access::read_write>, 4> "
        "images, int2 pixel, int mode)" in metal
    )
    assert "case 0: {" in metal
    assert metal.count("images[COUNT].read(uint2(pixel)).xy") == 2
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "vec2 imageNested(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert "case 0: {" in glsl
    assert glsl.count("imageLoad(images[COUNT], pixel).xy") == 2
    assert "image2D rgFloatImages[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + imageLoad(images[COUNT], pixel);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'images': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_loop_forms_sampled_shadowed_const_restores_after_loop():
    safe_crossgl = """
    shader LoopShadowedConstSampledSafe {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        vec4 sampleLoops(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            vec4 result = vec4(0.0);
            while (false) {
                int COUNT = 0;
                result = result + texture(textures[COUNT], samplers[COUNT], uv);
            }
            do {
                int COUNT = 0;
                result = result + texture(textures[COUNT], samplers[COUNT], uv);
            } while (false);
            loop {
                int COUNT = 0;
                result = result + texture(textures[COUNT], samplers[COUNT], uv);
                break;
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleLoops(textures, samplers, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert hlsl.count("int COUNT = 0;") == 3
    assert hlsl.count("textures[COUNT].Sample(samplers[COUNT], uv)") == 3
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "while (true)" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "Texture2D textures[5]" not in hlsl
    assert "SamplerState samplers[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert metal.count("int COUNT = 0;") == 3
    assert metal.count("textures[COUNT].sample(samplers[COUNT], uv)") == 3
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "while (true)" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float>, 5> textures" not in metal
    assert "array<sampler, 5> samplers" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert glsl.count("int COUNT = 0;") == 3
    assert glsl.count("texture(textures[COUNT], uv)") == 3
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "while (true)" in glsl
    assert "DoWhileNode(" not in glsl
    assert "sampler2D textures[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + texture(textures[COUNT], samplers[COUNT], uv);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_loop_forms_image_shadowed_const_restores_after_loop():
    safe_crossgl = """
    shader LoopShadowedConstImageSafe {
        const int COUNT = 4;
        image2D rgFloatImages[4] @rg32f;

        vec2 imageLoops(image2D images[4] @rg32f, ivec2 pixel) {
            vec2 result = vec2(0.0);
            while (false) {
                int COUNT = 0;
                result = result + imageLoad(images[COUNT], pixel);
            }
            do {
                int COUNT = 0;
                result = result + imageLoad(images[COUNT], pixel);
            } while (false);
            loop {
                int COUNT = 0;
                result = result + imageLoad(images[COUNT], pixel);
                break;
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = imageLoops(rgFloatImages, ivec2(0, 1));
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert hlsl.count("int COUNT = 0;") == 3
    assert hlsl.count("images[COUNT][pixel]") == 3
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "while (true)" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert metal.count("int COUNT = 0;") == 3
    assert metal.count("images[COUNT].read(uint2(pixel)).xy") == 3
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "while (true)" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert glsl.count("int COUNT = 0;") == 6
    assert glsl.count("imageLoad(images[COUNT], pixel).xy") == 3
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "while (true)" in glsl
    assert "DoWhileNode(" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + imageLoad(images[COUNT], pixel);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'images': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_do_while_switch_match_sampled_arrays_infer_size():
    crossgl = """
    shader DoWhileSwitchMatchSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 sampleCases(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            do {
                switch (mode) {
                    case 0:
                        result = result + texture(textures[3], samplers[3], uv);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + texture(textures[2], samplers[2], uv);
                    }
                    _ => {
                    }
                }
            } while (false);
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return sampleCases(textures, samplers, uv, mode) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t4);" in hlsl
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "case 0: {" in hlsl
    assert "textures[3].Sample(samplers[3], uv)" in hlsl
    assert "textures[2].Sample(samplers[2], uv)" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(4)]]" in metal
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "case 0: {" in metal
    assert "textures[3].sample(samplers[3], uv)" in metal
    assert "textures[2].sample(samplers[2], uv)" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "array<texture2d<float>, 5> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "layout(binding = 4) uniform sampler2D afterTexture;" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "case 0: {" in glsl
    assert "texture(textures[3], uv)" in glsl
    assert "texture(textures[2], uv)" in glsl
    assert "DoWhileNode(" not in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[5]" not in glsl


def test_codegen_do_while_switch_match_image_arrays_infer_size():
    crossgl = """
    shader DoWhileSwitchMatchImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 imageCases(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            do {
                switch (mode) {
                    case 0:
                        result = result + imageLoad(images[3], pixel);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + imageLoad(images[2], pixel);
                    }
                    _ => {
                    }
                }
            } while (false);
            return result;
        }

        compute {
            void main() {
                vec2 result = imageCases(rgFloatImages, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "case 0: {" in hlsl
    assert "images[3][pixel]" in hlsl
    assert "images[2][pixel]" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "case 0: {" in metal
    assert "images[3].read(uint2(pixel)).xy" in metal
    assert "images[2].read(uint2(pixel)).xy" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "case 0: {" in glsl
    assert "imageLoad(images[3], pixel).xy" in glsl
    assert "imageLoad(images[2], pixel).xy" in glsl
    assert "DoWhileNode(" not in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl


def test_codegen_do_while_transitive_sampled_arrays_infer_size():
    crossgl = """
    shader DoWhileTransitiveSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            do {
                switch (mode) {
                    case 0:
                        result = result + texture(textures[4], samplers[4], uv);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + texture(textures[3], samplers[3], uv);
                    }
                    _ => {
                    }
                }
            } while (false);
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            return leaf(textures, samplers, uv, mode);
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return mid(textures, samplers, uv, mode) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode)" in hlsl
    )
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "case 0: {" in hlsl
    assert "textures[4].Sample(samplers[4], uv)" in hlsl
    assert "textures[3].Sample(samplers[3], uv)" in hlsl
    assert "return leaf(textures, samplers, uv, mode);" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[6]" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode)" in metal
    )
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "case 0: {" in metal
    assert "textures[4].sample(samplers[4], uv)" in metal
    assert "textures[3].sample(samplers[3], uv)" in metal
    assert "return leaf(textures, samplers, uv, mode);" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "array<texture2d<float>, 6> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 leaf(sampler2D textures[5], vec2 uv, int mode)" in glsl
    assert "vec4 mid(sampler2D textures[5], vec2 uv, int mode)" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "case 0: {" in glsl
    assert "texture(textures[4], uv)" in glsl
    assert "texture(textures[3], uv)" in glsl
    assert "return leaf(textures, uv, mode);" in glsl
    assert "DoWhileNode(" not in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[6]" not in glsl


def test_codegen_do_while_transitive_image_arrays_infer_size():
    crossgl = """
    shader DoWhileTransitiveImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            do {
                switch (mode) {
                    case 0:
                        result = result + imageLoad(images[3], pixel);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + imageLoad(images[2], pixel);
                    }
                    _ => {
                    }
                }
            } while (false);
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            return leaf(images, pixel, mode);
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel, int mode)" in hlsl
    assert "float2 mid(RWTexture2D<float2> images[4], int2 pixel, int mode)" in hlsl
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert "case 0: {" in hlsl
    assert "images[3][pixel]" in hlsl
    assert "images[2][pixel]" in hlsl
    assert "return leaf(images, pixel, mode);" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode)" in metal
    )
    assert "do {" in metal
    assert "} while (false);" in metal
    assert "case 0: {" in metal
    assert "images[3].read(uint2(pixel)).xy" in metal
    assert "images[2].read(uint2(pixel)).xy" in metal
    assert "return leaf(images, pixel, mode);" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert "case 0: {" in glsl
    assert "imageLoad(images[3], pixel).xy" in glsl
    assert "imageLoad(images[2], pixel).xy" in glsl
    assert "return leaf(images, pixel, mode);" in glsl
    assert "DoWhileNode(" not in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl


def test_codegen_for_in_do_while_transitive_sampled_arrays_infer_size():
    crossgl = """
    shader ForInDoWhileTransitiveSampledArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv, int limit) {
            vec4 result = vec4(0.0);
            for i in 0..1 {
                do {
                    result = result + texture(textures[4], samplers[4], uv);
                } while (false);
            }
            do {
                for j in limit {
                    result = result + texture(textures[2], samplers[2], uv);
                }
            } while (false);
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode, int limit) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(textures, samplers, uv, limit);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode, int limit) @ gl_FragColor {
                return mid(textures, samplers, uv, mode, limit) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int limit)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode, int limit)" in hlsl
    )
    assert "for (int i = 0; i < 1; ++i)" in hlsl
    assert "for (int j = 0; j < limit; ++j)" in hlsl
    assert hlsl.count("do {") == 2
    assert hlsl.count("} while (false);") == 2
    assert "textures[4].Sample(samplers[4], uv)" in hlsl
    assert "textures[2].Sample(samplers[2], uv)" in hlsl
    assert "result = leaf(textures, samplers, uv, limit);" in hlsl
    assert "result = (result + leaf(textures, samplers, uv, limit));" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "ForInNode(" not in hlsl
    assert "Texture2D textures[]" not in hlsl
    assert "Texture2D textures[6]" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int limit)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode, int limit)" in metal
    )
    assert "for (int i = 0; i < 1; ++i)" in metal
    assert "for (int j = 0; j < limit; ++j)" in metal
    assert metal.count("do {") == 2
    assert metal.count("} while (false);") == 2
    assert "textures[4].sample(samplers[4], uv)" in metal
    assert "textures[2].sample(samplers[2], uv)" in metal
    assert "result = leaf(textures, samplers, uv, limit);" in metal
    assert "result = result + leaf(textures, samplers, uv, limit);" in metal
    assert "DoWhileNode(" not in metal
    assert "ForInNode(" not in metal
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "array<texture2d<float>, 6> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 leaf(sampler2D textures[5], vec2 uv, int limit)" in glsl
    assert "vec4 mid(sampler2D textures[5], vec2 uv, int mode, int limit)" in glsl
    assert "for (int i = 0; i < 1; ++i)" in glsl
    assert "for (int j = 0; j < limit; ++j)" in glsl
    assert glsl.count("do {") == 2
    assert glsl.count("} while (false);") == 2
    assert "texture(textures[4], uv)" in glsl
    assert "texture(textures[2], uv)" in glsl
    assert "result = leaf(textures, uv, limit);" in glsl
    assert "result = (result + leaf(textures, uv, limit));" in glsl
    assert "DoWhileNode(" not in glsl
    assert "ForInNode(" not in glsl
    assert "sampler2D textures[]" not in glsl
    assert "sampler2D textures[6]" not in glsl

    fixed_crossgl = crossgl.replace(
        "sampler2D textures[];\n        sampler samplers[];",
        "sampler2D textures[4];\n        sampler samplers[4];",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_for_in_do_while_transitive_image_arrays_infer_size():
    crossgl = """
    shader ForInDoWhileTransitiveImageArrays {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel, int limit) {
            vec2 result = vec2(0.0);
            for i in 0..1 {
                do {
                    result = result + imageLoad(images[3], pixel);
                } while (false);
            }
            do {
                for j in limit {
                    result = result + imageLoad(images[1], pixel);
                }
            } while (false);
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode, int limit) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel, limit);
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    result = result + leaf(images, pixel, limit);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1), 0, 2);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel, int limit)" in hlsl
    assert (
        "float2 mid(RWTexture2D<float2> images[4], int2 pixel, int mode, "
        "int limit)" in hlsl
    )
    assert "for (int i = 0; i < 1; ++i)" in hlsl
    assert "for (int j = 0; j < limit; ++j)" in hlsl
    assert hlsl.count("do {") == 2
    assert hlsl.count("} while (false);") == 2
    assert "images[3][pixel]" in hlsl
    assert "images[1][pixel]" in hlsl
    assert "result = leaf(images, pixel, limit);" in hlsl
    assert "result = (result + leaf(images, pixel, limit));" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "ForInNode(" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[]" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int limit)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode, int limit)" in metal
    )
    assert "for (int i = 0; i < 1; ++i)" in metal
    assert "for (int j = 0; j < limit; ++j)" in metal
    assert metal.count("do {") == 2
    assert metal.count("} while (false);") == 2
    assert "images[3].read(uint2(pixel)).xy" in metal
    assert "images[1].read(uint2(pixel)).xy" in metal
    assert "result = leaf(images, pixel, limit);" in metal
    assert "result = result + leaf(images, pixel, limit);" in metal
    assert "DoWhileNode(" not in metal
    assert "ForInNode(" not in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel, int limit)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel, int mode, int limit)" in glsl
    assert "for (int i = 0; i < 1; ++i)" in glsl
    assert "for (int j = 0; j < limit; ++j)" in glsl
    assert glsl.count("do {") == 4
    assert glsl.count("} while (false);") == 4
    assert "imageLoad(images[3], pixel).xy" in glsl
    assert "imageLoad(images[1], pixel).xy" in glsl
    assert "result = leaf(images, pixel, limit);" in glsl
    assert "result = (result + leaf(images, pixel, limit));" in glsl
    assert "DoWhileNode(" not in glsl
    assert "ForInNode(" not in glsl
    assert "image2D rgFloatImages[]" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    fixed_crossgl = crossgl.replace(
        "image2D rgFloatImages @rg32f[];",
        "image2D rgFloatImages[3] @rg32f;",
    )
    fixed_ast = parse_crossgl(fixed_crossgl)
    assert fixed_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 3 and 4"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(fixed_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(fixed_ast)


def test_codegen_do_while_fixed_transitive_sampled_conflicts_are_rejected():
    crossgl = """
    shader DoWhileFixedTransitiveSampledConflict {
        sampler2D smallTextures[2];
        sampler smallSamplers[2];
        sampler2D largeTextures[5];
        sampler largeSamplers[5];

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            do {
                switch (mode) {
                    case 0:
                        result = result + texture(textures[4], samplers[4], uv);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + texture(textures[3], samplers[3], uv);
                    }
                    _ => {
                    }
                }
            } while (false);
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            return leaf(textures, samplers, uv, mode);
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return mid(smallTextures, smallSamplers, uv, mode) + mid(largeTextures, largeSamplers, uv, mode);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'smallTextures': 2 and 5"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_do_while_fixed_transitive_image_conflicts_are_rejected():
    crossgl = """
    shader DoWhileFixedTransitiveImageConflict {
        image2D smallImages[2] @rg32f;
        image2D largeImages[4] @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            do {
                switch (mode) {
                    case 0:
                        result = result + imageLoad(images[3], pixel);
                        break;
                    default:
                        break;
                }
                match mode {
                    1 => {
                        result = result + imageLoad(images[2], pixel);
                    }
                    _ => {
                    }
                }
            } while (false);
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            return leaf(images, pixel, mode);
        }

        compute {
            void main() {
                vec2 a = mid(smallImages, ivec2(0, 1), 0);
                vec2 b = mid(largeImages, ivec2(1, 2), 1);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'smallImages': 2 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_do_while_transitive_shadowed_sampled_const_restores_after_loop():
    safe_crossgl = """
    shader DoWhileTransitiveShadowedSampledSafe {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 result = vec4(0.0);
            do {
                int COUNT = 0;
                result = result + texture(textures[COUNT], samplers[COUNT], uv);
            } while (false);
            return result;
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv) {
            return leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return mid(textures, samplers, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[4], SamplerState samplers[4], "
        "float2 uv)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[4], SamplerState samplers[4], "
        "float2 uv)" in hlsl
    )
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert hlsl.count("int COUNT = 0;") == 1
    assert "textures[COUNT].Sample(samplers[COUNT], uv)" in hlsl
    assert "return leaf(textures, samplers, uv);" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "Texture2D textures[5]" not in hlsl
    assert "SamplerState samplers[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 4> textures, "
        "array<sampler, 4> samplers, float2 uv)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 4> textures, "
        "array<sampler, 4> samplers, float2 uv)" in metal
    )
    assert "do {" in metal
    assert "} while (false);" in metal
    assert metal.count("int COUNT = 0;") == 1
    assert "textures[COUNT].sample(samplers[COUNT], uv)" in metal
    assert "return leaf(textures, samplers, uv);" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float>, 5> textures" not in metal
    assert "array<sampler, 5> samplers" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "vec4 leaf(sampler2D textures[4], vec2 uv)" in glsl
    assert "vec4 mid(sampler2D textures[4], vec2 uv)" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert glsl.count("int COUNT = 0;") == 1
    assert "texture(textures[COUNT], uv)" in glsl
    assert "return leaf(textures, uv);" in glsl
    assert "DoWhileNode(" not in glsl
    assert "sampler2D textures[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + texture(textures[COUNT], samplers[COUNT], uv);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_do_while_transitive_shadowed_image_const_restores_after_loop():
    safe_crossgl = """
    shader DoWhileTransitiveShadowedImageSafe {
        const int COUNT = 4;
        image2D rgFloatImages[4] @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel) {
            vec2 result = vec2(0.0);
            do {
                int COUNT = 0;
                result = result + imageLoad(images[COUNT], pixel);
            } while (false);
            return result;
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel) {
            return leaf(images, pixel);
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1));
            }
        }
    }
    """

    shader_ast = parse_crossgl(safe_crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "float2 mid(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "do {" in hlsl
    assert "} while (false);" in hlsl
    assert hlsl.count("int COUNT = 0;") == 1
    assert "images[COUNT][pixel]" in hlsl
    assert "return leaf(images, pixel);" in hlsl
    assert "DoWhileNode(" not in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel)" in metal
    )
    assert "do {" in metal
    assert "} while (false);" in metal
    assert metal.count("int COUNT = 0;") == 1
    assert "images[COUNT].read(uint2(pixel)).xy" in metal
    assert "return leaf(images, pixel);" in metal
    assert "DoWhileNode(" not in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel)" in glsl
    assert "do {" in glsl
    assert "} while (false);" in glsl
    assert glsl.count("int COUNT = 0;") == 2
    assert "imageLoad(images[COUNT], pixel).xy" in glsl
    assert "return leaf(images, pixel);" in glsl
    assert "DoWhileNode(" not in glsl
    assert "image2D rgFloatImages[5]" not in glsl

    conflict_crossgl = safe_crossgl.replace(
        "return result;",
        "return result + imageLoad(images[COUNT], pixel);",
    )
    conflict_ast = parse_crossgl(conflict_crossgl)
    assert conflict_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'rgFloatImages': 4 and 5"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(conflict_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(conflict_ast)


def test_codegen_switch_match_transitive_sampled_arrays_infer_case_only_size():
    crossgl = """
    shader SwitchMatchTransitiveSampledCaseOnly {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int mode @ TEXCOORD1;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[4], samplers[4], uv);
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv);
                    break;
                default:
                    break;
            }
            match mode {
                0 => {
                    result = result + leaf(textures, samplers, uv);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return mid(textures, samplers, input.uv, input.mode) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert "SamplerState afterTextureSampler : register(s5);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode)" in hlsl
    )
    assert "result = leaf(textures, samplers, uv);" in hlsl
    assert "result = (result + leaf(textures, samplers, uv));" in hlsl
    assert "Texture2D textures[] : register(t0);" not in hlsl

    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode)" in metal
    )
    assert "result = leaf(textures, samplers, uv);" in metal
    assert "result = result + leaf(textures, samplers, uv);" in metal
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 leaf(sampler2D textures[5], vec2 uv)" in glsl
    assert "vec4 mid(sampler2D textures[5], vec2 uv, int mode)" in glsl
    assert "result = leaf(textures, uv);" in glsl
    assert "result = (result + leaf(textures, uv));" in glsl
    assert "sampler2D textures[]" not in glsl


def test_codegen_switch_match_transitive_image_arrays_infer_case_only_size():
    crossgl = """
    shader SwitchMatchTransitiveImageCaseOnly {
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[3], pixel);
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel);
                    break;
                default:
                    break;
            }
            match mode {
                0 => {
                    result = result + leaf(images, pixel);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "float2 mid(RWTexture2D<float2> images[4], int2 pixel, int mode)" in hlsl
    assert "return images[3][pixel];" in hlsl
    assert "result = leaf(images, pixel);" in hlsl
    assert "result = (result + leaf(images, pixel));" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode)" in metal
    )
    assert "return images[3].read(uint2(pixel)).xy;" in metal
    assert "result = leaf(images, pixel);" in metal
    assert "result = result + leaf(images, pixel);" in metal
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert "return imageLoad(images[3], pixel).xy;" in glsl
    assert "result = leaf(images, pixel);" in glsl
    assert "result = (result + leaf(images, pixel));" in glsl
    assert "image2D rgFloatImages[]" not in glsl


def test_codegen_switch_match_shadowed_transitive_sampled_arrays_infer_size():
    crossgl = """
    shader SwitchMatchShadowedTransitiveSampled {
        const int COUNT = 4;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    int COUNT = 0;
                    match mode {
                        0 => {
                            result = texture(textures[COUNT], samplers[COUNT], uv);
                            result = result + leaf(textures, samplers, uv);
                        }
                        _ => {
                        }
                    }
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    int COUNT = 0;
                    switch (mode) {
                        case 1:
                            result = result + texture(textures[COUNT], samplers[COUNT], uv);
                            result = result + leaf(textures, samplers, uv);
                            break;
                        default:
                            break;
                    }
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv, int mode) @ gl_FragColor {
                return mid(textures, samplers, uv, mode) + texture(afterTexture, uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "static const int COUNT = 4;" in hlsl
    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 leaf(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv)" in hlsl
    )
    assert (
        "float4 mid(Texture2D textures[5], SamplerState samplers[5], "
        "float2 uv, int mode)" in hlsl
    )
    assert hlsl.count("int COUNT = 0;") == 2
    assert hlsl.count("textures[COUNT].Sample(samplers[COUNT], uv)") == 3
    assert hlsl.count("leaf(textures, samplers, uv)") == 2
    assert "Texture2D textures[] : register(t0);" not in hlsl

    assert "constant int COUNT = 4;" in metal
    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 leaf(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv)" in metal
    )
    assert (
        "float4 mid(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, float2 uv, int mode)" in metal
    )
    assert metal.count("int COUNT = 0;") == 2
    assert metal.count("textures[COUNT].sample(samplers[COUNT], uv)") == 3
    assert metal.count("leaf(textures, samplers, uv)") == 2
    assert "array<texture2d<float>, 1> textures" not in metal

    assert "const int COUNT = 4;" in glsl
    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert "vec4 leaf(sampler2D textures[5], vec2 uv)" in glsl
    assert "vec4 mid(sampler2D textures[5], vec2 uv, int mode)" in glsl
    assert glsl.count("int COUNT = 0;") == 2
    assert glsl.count("texture(textures[COUNT], uv)") == 3
    assert glsl.count("leaf(textures, uv)") == 2
    assert "sampler2D textures[]" not in glsl


def test_codegen_switch_match_shadowed_transitive_image_arrays_infer_size():
    crossgl = """
    shader SwitchMatchShadowedTransitiveImage {
        const int COUNT = 3;
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    int COUNT = 0;
                    match mode {
                        0 => {
                            result = imageLoad(images[COUNT], pixel);
                            result = result + leaf(images, pixel);
                        }
                        _ => {
                        }
                    }
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    int COUNT = 0;
                    switch (mode) {
                        case 1:
                            result = result + imageLoad(images[COUNT], pixel);
                            result = result + leaf(images, pixel);
                            break;
                        default:
                            break;
                    }
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = mid(rgFloatImages, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "static const int COUNT = 3;" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert "float2 leaf(RWTexture2D<float2> images[4], int2 pixel)" in hlsl
    assert "float2 mid(RWTexture2D<float2> images[4], int2 pixel, int mode)" in hlsl
    assert hlsl.count("int COUNT = 0;") == 2
    assert hlsl.count("images[COUNT][pixel]") == 3
    assert hlsl.count("leaf(images, pixel)") == 2
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in hlsl

    assert "constant int COUNT = 3;" in metal
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 leaf(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel)" in metal
    )
    assert (
        "float2 mid(array<texture2d<float, access::read_write>, 4> images, "
        "int2 pixel, int mode)" in metal
    )
    assert metal.count("int COUNT = 0;") == 2
    assert metal.count("images[COUNT].read(uint2(pixel)).xy") == 3
    assert metal.count("leaf(images, pixel)") == 2
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal

    assert "const int COUNT = 3;" in glsl
    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert "vec2 leaf(image2D images[4], ivec2 pixel)" in glsl
    assert "vec2 mid(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert glsl.count("int COUNT = 0;") == 4
    assert glsl.count("imageLoad(images[COUNT], pixel).xy") == 3
    assert glsl.count("leaf(images, pixel)") == 2
    assert "image2D rgFloatImages[]" not in glsl


def test_codegen_switch_match_sampled_arrays_ignore_dynamic_negative_indices():
    crossgl = """
    shader SwitchMatchMixedComputedSampledIndices {
        const int BASE = 2;
        const int OFFSET = 2;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int layer @ TEXCOORD1;
            int mode @ TEXCOORD2;
        };

        vec4 sampleCases(sampler2D textures[], sampler samplers[], int layer, vec2 uv, int mode) {
            vec4 result = texture(textures[layer + 1], samplers[layer + 1], uv);
            switch (mode) {
                case 0:
                    result = result + texture(textures[-1], samplers[-1], uv);
                    break;
                default:
                    result = result + texture(textures[BASE + OFFSET], samplers[BASE + OFFSET], uv);
                    break;
            }
            match mode {
                1 => {
                    result = result + texture(textures[layer - 1], samplers[layer - 1], uv);
                }
                _ => {
                    result = result + texture(textures[BASE + OFFSET], samplers[BASE + OFFSET], uv);
                }
            }
            return result;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleCases(textures, samplers, input.layer, input.uv, input.mode) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "static const int BASE = 2;" in hlsl
    assert "static const int OFFSET = 2;" in hlsl
    assert "Texture2D textures[5] : register(t0);" in hlsl
    assert "SamplerState samplers[5] : register(s0);" in hlsl
    assert "Texture2D afterTexture : register(t5);" in hlsl
    assert (
        "float4 sampleCases(Texture2D textures[5], SamplerState samplers[5], "
        "int layer, float2 uv, int mode)" in hlsl
    )
    assert "textures[(layer + 1)].Sample(samplers[(layer + 1)], uv)" in hlsl
    assert "textures[-1].Sample(samplers[-1], uv)" in hlsl
    assert "textures[(layer - 1)].Sample(samplers[(layer - 1)], uv)" in hlsl
    assert (
        hlsl.count("textures[(BASE + OFFSET)].Sample(samplers[(BASE + OFFSET)], uv)")
        == 2
    )
    assert "Texture2D textures[1] : register(t0);" not in hlsl
    assert "Texture2D afterTexture : register(t1);" not in hlsl

    assert "constant int BASE = 2;" in metal
    assert "constant int OFFSET = 2;" in metal
    assert "array<texture2d<float>, 5> textures [[texture(0)]]" in metal
    assert "array<sampler, 5> samplers [[sampler(0)]]" in metal
    assert "texture2d<float> afterTexture [[texture(5)]]" in metal
    assert (
        "float4 sampleCases(array<texture2d<float>, 5> textures, "
        "array<sampler, 5> samplers, int layer, float2 uv, int mode)" in metal
    )
    assert "textures[layer + 1].sample(samplers[layer + 1], uv)" in metal
    assert "textures[-1].sample(samplers[-1], uv)" in metal
    assert "textures[layer - 1].sample(samplers[layer - 1], uv)" in metal
    assert (
        metal.count("textures[BASE + OFFSET].sample(samplers[BASE + OFFSET], uv)") == 2
    )
    assert "array<texture2d<float>, 1> textures" not in metal
    assert "texture2d<float> afterTexture [[texture(1)]]" not in metal

    assert "const int BASE = 2;" in glsl
    assert "const int OFFSET = 2;" in glsl
    assert "layout(binding = 0) uniform sampler2D textures[5];" in glsl
    assert "layout(binding = 5) uniform sampler2D afterTexture;" in glsl
    assert (
        "vec4 sampleCases(sampler2D textures[5], int layer, vec2 uv, int mode)" in glsl
    )
    assert "texture(textures[(layer + 1)], uv)" in glsl
    assert "texture(textures[(-1)], uv)" in glsl
    assert "texture(textures[(layer - 1)], uv)" in glsl
    assert glsl.count("texture(textures[(BASE + OFFSET)], uv)") == 2
    assert "layout(binding = 0) uniform sampler2D textures[1];" not in glsl
    assert "layout(binding = 1) uniform sampler2D afterTexture;" not in glsl


def test_codegen_switch_match_image_arrays_ignore_dynamic_negative_indices():
    crossgl = """
    shader SwitchMatchMixedComputedImageIndices {
        const int BASE = 1;
        const int OFFSET = 2;
        image2D rgFloatImages @rg32f[];
        image2D afterImage @rg32f;

        vec2 imageCases(image2D images[] @rg32f, int layer, ivec2 pixel, int mode) {
            vec2 result = imageLoad(images[layer + 1], pixel);
            switch (mode) {
                case 0:
                    result = result + imageLoad(images[-1], pixel);
                    break;
                default:
                    result = result + imageLoad(images[BASE + OFFSET], pixel);
                    break;
            }
            match mode {
                1 => {
                    result = result + imageLoad(images[layer - 1], pixel);
                }
                _ => {
                    result = result + imageLoad(images[BASE + OFFSET], pixel);
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = imageCases(rgFloatImages, 1, ivec2(0, 1), 0);
                imageStore(afterImage, ivec2(1, 2), result);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "static const int BASE = 1;" in hlsl
    assert "static const int OFFSET = 2;" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert "RWTexture2D<float2> afterImage : register(u4);" in hlsl
    assert (
        "float2 imageCases(RWTexture2D<float2> images[4], int layer, "
        "int2 pixel, int mode)" in hlsl
    )
    assert "images[(layer + 1)][pixel]" in hlsl
    assert "images[-1][pixel]" in hlsl
    assert "images[(layer - 1)][pixel]" in hlsl
    assert hlsl.count("images[(BASE + OFFSET)][pixel]") == 2
    assert "RWTexture2D<float2> rgFloatImages[1] : register(u0);" not in hlsl
    assert "RWTexture2D<float2> afterImage : register(u1);" not in hlsl

    assert "constant int BASE = 1;" in metal
    assert "constant int OFFSET = 2;" in metal
    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert "texture2d<float, access::read_write> afterImage [[texture(4)]]" in metal
    assert (
        "float2 imageCases(array<texture2d<float, access::read_write>, 4> "
        "images, int layer, int2 pixel, int mode)" in metal
    )
    assert "images[layer + 1].read(uint2(pixel)).xy" in metal
    assert "images[-1].read(uint2(pixel)).xy" in metal
    assert "images[layer - 1].read(uint2(pixel)).xy" in metal
    assert metal.count("images[BASE + OFFSET].read(uint2(pixel)).xy") == 2
    assert "array<texture2d<float, access::read_write>, 1> rgFloatImages" not in metal
    assert "texture2d<float, access::read_write> afterImage [[texture(1)]]" not in metal

    assert "const int BASE = 1;" in glsl
    assert "const int OFFSET = 2;" in glsl
    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "layout(rg32f, binding = 4) uniform image2D afterImage;" in glsl
    assert (
        "vec2 imageCases(image2D images[4], int layer, ivec2 pixel, int mode)" in glsl
    )
    assert "imageLoad(images[(layer + 1)], pixel).xy" in glsl
    assert "imageLoad(images[(-1)], pixel).xy" in glsl
    assert "imageLoad(images[(layer - 1)], pixel).xy" in glsl
    assert glsl.count("imageLoad(images[(BASE + OFFSET)], pixel).xy") == 2
    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[1];" not in glsl
    assert "layout(rg32f, binding = 1) uniform image2D afterImage;" not in glsl


def test_codegen_switch_match_fixed_sampled_transitive_conflicts_are_rejected():
    crossgl = """
    shader SwitchMatchFixedTransitiveSampledConflict {
        sampler2D smallTextures[2];
        sampler smallSamplers[2];
        sampler2D largeTextures[5];
        sampler largeSamplers[5];

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[4], samplers[4], uv);
        }

        vec4 mid(sampler2D textures[], sampler samplers[], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    result = leaf(textures, samplers, uv);
                    break;
                default:
                    break;
            }
            match mode {
                0 => {
                    result = result + leaf(textures, samplers, uv);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return mid(smallTextures, smallSamplers, uv, 0) + mid(largeTextures, largeSamplers, uv, 1);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = (
        "Conflicting fixed resource array sizes for 'smallTextures': 2 and 5"
    )
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_fixed_image_transitive_conflicts_are_rejected():
    crossgl = """
    shader SwitchMatchFixedTransitiveImageConflict {
        image2D smallImages[2] @rg32f;
        image2D largeImages[4] @rg32f;

        vec2 leaf(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[3], pixel);
        }

        vec2 mid(image2D images[] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    result = leaf(images, pixel);
                    break;
                default:
                    break;
            }
            match mode {
                0 => {
                    result = result + leaf(images, pixel);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 a = mid(smallImages, ivec2(0, 1), 0);
                vec2 b = mid(largeImages, ivec2(1, 2), 1);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'smallImages': 2 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_fixed_sampled_arrays_reject_case_out_of_bounds():
    crossgl = """
    shader SwitchMatchFixedSampledOutOfBounds {
        sampler2D textures[4];
        sampler samplers[4];

        vec4 sampleCases(sampler2D textures[4], sampler samplers[4], vec2 uv, int mode) {
            switch (mode) {
                case 0:
                    return texture(textures[4], samplers[4], uv);
                default:
                    break;
            }
            match mode {
                0 => { return texture(textures[0], samplers[0], uv); }
                _ => { return texture(textures[4], samplers[4], uv); }
            }
            return vec4(0.0);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleCases(textures, samplers, uv, 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'textures': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_fixed_image_arrays_reject_case_out_of_bounds():
    crossgl = """
    shader SwitchMatchFixedImageOutOfBounds {
        image2D rgFloatImages[4] @rg32f;

        vec2 imageCases(image2D images[4] @rg32f, ivec2 pixel, int mode) {
            switch (mode) {
                case 0:
                    return imageLoad(images[4], pixel);
                default:
                    break;
            }
            match mode {
                0 => { return imageLoad(images[0], pixel); }
                _ => { return imageLoad(images[4], pixel); }
            }
            return vec2(0.0);
        }

        compute {
            void main() {
                vec2 result = imageCases(rgFloatImages, ivec2(0, 1), 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'images': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_switch_match_fixed_sampled_arrays_shadowed_case_const_stays_dynamic():
    crossgl = """
    shader SwitchMatchFixedSampledShadowedConst {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        vec4 sampleCases(sampler2D textures[4], sampler samplers[4], vec2 uv, int mode) {
            vec4 result = vec4(0.0);
            switch (mode) {
                case 0:
                    int COUNT = 0;
                    vec4 scoped = texture(textures[COUNT], samplers[COUNT], uv);
                    result = scoped;
                    break;
                default:
                    break;
            }
            match mode {
                0 => {
                    int COUNT = 0;
                    result = result + texture(textures[COUNT], samplers[COUNT], uv);
                }
                _ => {
                }
            }
            return result;
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return sampleCases(textures, samplers, uv, 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2D textures[4] : register(t0);" in hlsl
    assert "SamplerState samplers[4] : register(s0);" in hlsl
    assert (
        "float4 sampleCases(Texture2D textures[4], SamplerState samplers[4], "
        "float2 uv, int mode)" in hlsl
    )
    assert hlsl.count("int COUNT = 0;") == 2
    assert "textures[COUNT].Sample(samplers[COUNT], uv)" in hlsl
    assert "Texture2D textures[5]" not in hlsl
    assert "SamplerState samplers[5]" not in hlsl

    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in metal
    assert "array<sampler, 4> samplers [[sampler(0)]]" in metal
    assert (
        "float4 sampleCases(array<texture2d<float>, 4> textures, "
        "array<sampler, 4> samplers, float2 uv, int mode)" in metal
    )
    assert metal.count("int COUNT = 0;") == 2
    assert "textures[COUNT].sample(samplers[COUNT], uv)" in metal
    assert "array<texture2d<float>, 5> textures" not in metal
    assert "array<sampler, 5> samplers" not in metal

    assert "layout(binding = 0) uniform sampler2D textures[4];" in glsl
    assert "vec4 sampleCases(sampler2D textures[4], vec2 uv, int mode)" in glsl
    assert glsl.count("int COUNT = 0;") == 2
    assert "texture(textures[COUNT], uv)" in glsl
    assert "sampler2D textures[5]" not in glsl


def test_codegen_switch_match_fixed_image_arrays_shadowed_case_const_stays_dynamic():
    crossgl = """
    shader SwitchMatchFixedImageShadowedConst {
        const int COUNT = 4;
        image2D rgFloatImages[4] @rg32f;

        vec2 imageCases(image2D images[4] @rg32f, ivec2 pixel, int mode) {
            vec2 result = vec2(0.0);
            switch (mode) {
                case 0:
                    int COUNT = 0;
                    vec2 scoped = imageLoad(images[COUNT], pixel);
                    result = scoped;
                    break;
                default:
                    break;
            }
            match mode {
                0 => {
                    int COUNT = 0;
                    result = result + imageLoad(images[COUNT], pixel);
                }
                _ => {
                }
            }
            return result;
        }

        compute {
            void main() {
                vec2 result = imageCases(rgFloatImages, ivec2(0, 1), 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in hlsl
    assert (
        "float2 imageCases(RWTexture2D<float2> images[4], int2 pixel, int mode)" in hlsl
    )
    assert hlsl.count("int COUNT = 0;") == 2
    assert "images[COUNT][pixel]" in hlsl
    assert "RWTexture2D<float2> rgFloatImages[5]" not in hlsl

    assert (
        "array<texture2d<float, access::read_write>, 4> rgFloatImages "
        "[[texture(0)]]" in metal
    )
    assert (
        "float2 imageCases(array<texture2d<float, access::read_write>, 4> "
        "images, int2 pixel, int mode)" in metal
    )
    assert metal.count("int COUNT = 0;") == 2
    assert "images[COUNT].read(uint2(pixel)).xy" in metal
    assert "array<texture2d<float, access::read_write>, 5> rgFloatImages" not in metal

    assert "layout(rg32f, binding = 0) uniform image2D rgFloatImages[4];" in glsl
    assert "vec2 imageCases(image2D images[4], ivec2 pixel, int mode)" in glsl
    assert glsl.count("int COUNT = 0;") == 4
    assert "imageLoad(images[COUNT], pixel).xy" in glsl
    assert "image2D rgFloatImages[5]" not in glsl


def test_codegen_dynamic_fixed_multisample_image_array_indices_keep_swizzles():
    crossgl = """
    shader DynamicFixedMultisampleImageArrayIndex {
        image2DMS colorImages[4] @rgba16f;
        uimage2DMSArray counterLayers[4] @rgba32ui;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        vec4 readColor(image2DMS images[4] @rgba16f, int layer, ivec2 pixel, int sampleIndex) {
            return imageLoad(images[layer], pixel, sampleIndex);
        }

        uint readCounter(uimage2DMSArray images[4] @rgba32ui, int layer, ivec3 pixelLayer, int sampleIndex) {
            return imageLoad(images[layer], pixelLayer, sampleIndex).x;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = readColor(colorImages, input.layer, ivec2(1, 2), 0);
                uint counter = readCounter(counterLayers, input.layer, ivec3(1, 2, 3), 0);
                return color + vec4(float(counter));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> colorImages[4] : register(t0);" in hlsl
    assert "Texture2DMSArray<uint4> counterLayers[4] : register(t4);" in hlsl
    assert (
        "float4 readColor(Texture2DMS<float4> images[4], int layer, int2 pixel, "
        "int sampleIndex)" in hlsl
    )
    assert (
        "uint readCounter(Texture2DMSArray<uint4> images[4], int layer, "
        "int3 pixelLayer, int sampleIndex)" in hlsl
    )
    assert "return images[layer].Load(pixel, sampleIndex);" in hlsl
    assert "return images[layer].Load(pixelLayer, sampleIndex).x;" in hlsl
    assert ".x.x" not in hlsl

    assert (
        "array<texture2d_ms<float, access::read>, 4> colorImages [[texture(0)]]"
        in metal
    )
    assert (
        "array<texture2d_ms_array<uint, access::read>, 4> counterLayers [[texture(4)]]"
        in metal
    )
    assert (
        "float4 readColor(array<texture2d_ms<float, access::read>, 4> images, "
        "int layer, int2 pixel, int sampleIndex)" in metal
    )
    assert (
        "uint readCounter(array<texture2d_ms_array<uint, access::read>, 4> images, "
        "int layer, int3 pixelLayer, int sampleIndex)" in metal
    )
    assert "return images[layer].read(uint2(pixel), uint(sampleIndex));" in metal
    assert (
        "return images[layer].read(uint2(pixelLayer.xy), uint(pixelLayer.z), "
        "uint(sampleIndex)).x;" in metal
    )
    assert ".x.x" not in metal

    assert "layout(rgba16f, binding = 0) uniform image2DMS colorImages[4];" in glsl
    assert (
        "layout(rgba32ui, binding = 4) uniform uimage2DMSArray counterLayers[4];"
        in glsl
    )
    assert (
        "vec4 readColor(image2DMS images[4], int layer, ivec2 pixel, int sampleIndex)"
        in glsl
    )
    assert (
        "uint readCounter(uimage2DMSArray images[4], int layer, "
        "ivec3 pixelLayer, int sampleIndex)" in glsl
    )
    assert "return imageLoad(images[layer], pixel, sampleIndex);" in glsl
    assert "return imageLoad(images[layer], pixelLayer, sampleIndex).x;" in glsl
    assert ".x.x" not in glsl


def test_codegen_multisample_image_later_shadowed_const_index_conflicts():
    crossgl = """
    shader LaterShadowMultisampleImageConstIndex {
        const int COUNT = 4;
        image2DMS colorImages[4] @rgba16f;

        vec4 leaf(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex) {
            vec4 value = imageLoad(images[COUNT], pixel, sampleIndex);
            int COUNT = 0;
            return value;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return leaf(colorImages, ivec2(1, 2), 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'colorImages': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_multisample_image_earlier_shadowed_const_index_stays_dynamic():
    crossgl = """
    shader EarlierShadowMultisampleImageConstIndex {
        const int COUNT = 4;
        image2DMS colorImages[4] @rgba16f;
        uimage2DMSArray counterLayers[4] @rgba32ui;

        vec4 leafColor(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel, sampleIndex);
        }

        uint leafCounter(uimage2DMSArray images[] @rgba32ui, ivec3 pixelLayer, int sampleIndex) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixelLayer, sampleIndex).x;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                vec4 color = leafColor(colorImages, ivec2(1, 2), 0);
                uint counter = leafCounter(counterLayers, ivec3(1, 2, 3), 0);
                return color + vec4(float(counter));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> colorImages[4] : register(t0);" in hlsl
    assert "Texture2DMSArray<uint4> counterLayers[4] : register(t4);" in hlsl
    assert (
        "float4 leafColor(Texture2DMS<float4> images[4], int2 pixel, int sampleIndex)"
        in hlsl
    )
    assert (
        "uint leafCounter(Texture2DMSArray<uint4> images[4], int3 pixelLayer, "
        "int sampleIndex)" in hlsl
    )
    assert "return images[COUNT].Load(pixel, sampleIndex);" in hlsl
    assert "return images[COUNT].Load(pixelLayer, sampleIndex).x;" in hlsl
    assert "images[5]" not in hlsl
    assert ".x.x" not in hlsl

    assert (
        "array<texture2d_ms<float, access::read>, 4> colorImages [[texture(0)]]"
        in metal
    )
    assert (
        "array<texture2d_ms_array<uint, access::read>, 4> counterLayers [[texture(4)]]"
        in metal
    )
    assert (
        "float4 leafColor(array<texture2d_ms<float, access::read>, 4> images, "
        "int2 pixel, int sampleIndex)" in metal
    )
    assert (
        "uint leafCounter(array<texture2d_ms_array<uint, access::read>, 4> images, "
        "int3 pixelLayer, int sampleIndex)" in metal
    )
    assert "return images[COUNT].read(uint2(pixel), uint(sampleIndex));" in metal
    assert (
        "return images[COUNT].read(uint2(pixelLayer.xy), uint(pixelLayer.z), "
        "uint(sampleIndex)).x;" in metal
    )
    assert "images[5]" not in metal
    assert ".x.x" not in metal

    assert "layout(rgba16f, binding = 0) uniform image2DMS colorImages[4];" in glsl
    assert (
        "layout(rgba32ui, binding = 4) uniform uimage2DMSArray counterLayers[4];"
        in glsl
    )
    assert "vec4 leafColor(image2DMS images[4], ivec2 pixel, int sampleIndex)" in glsl
    assert (
        "uint leafCounter(uimage2DMSArray images[4], ivec3 pixelLayer, "
        "int sampleIndex)" in glsl
    )
    assert "return imageLoad(images[COUNT], pixel, sampleIndex);" in glsl
    assert "return imageLoad(images[COUNT], pixelLayer, sampleIndex).x;" in glsl
    assert "images[5]" not in glsl
    assert ".x.x" not in glsl


def test_codegen_multisample_image_if_scope_restores_const_index_conflict():
    crossgl = """
    shader IfScopedMultisampleImageConstIndex {
        const int COUNT = 4;
        image2DMS colorImages[4] @rgba16f;

        vec4 leaf(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex, bool flag) {
            if (flag) {
                int COUNT = 0;
                vec4 dynamicValue = imageLoad(images[COUNT], pixel, sampleIndex);
            }
            return imageLoad(images[COUNT], pixel, sampleIndex);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return leaf(colorImages, ivec2(1, 2), 0, true);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'colorImages': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_multisample_image_loop_scope_restores_const_index_conflict():
    crossgl = """
    shader LoopScopedMultisampleImageConstIndex {
        const int COUNT = 4;
        image2DMS colorImages[4] @rgba16f;

        vec4 leaf(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex) {
            for (int i = 0; i < 1; i = i + 1) {
                int COUNT = 0;
                vec4 dynamicValue = imageLoad(images[COUNT], pixel, sampleIndex);
            }
            return imageLoad(images[COUNT], pixel, sampleIndex);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return leaf(colorImages, ivec2(1, 2), 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'colorImages': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_multisample_image_switch_scope_restores_const_index_conflict():
    crossgl = """
    shader SwitchScopedMultisampleImageConstIndex {
        const int COUNT = 4;
        image2DMS colorImages[4] @rgba16f;

        vec4 leaf(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex, int mode) {
            switch (mode) {
                case 0:
                    int COUNT = 0;
                    vec4 dynamicValue = imageLoad(images[COUNT], pixel, sampleIndex);
                    break;
                default:
                    break;
            }
            return imageLoad(images[COUNT], pixel, sampleIndex);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return leaf(colorImages, ivec2(1, 2), 0, 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'colorImages': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_multisample_image_match_scope_restores_const_index_conflict():
    crossgl = """
    shader MatchScopedMultisampleImageConstIndex {
        const int COUNT = 4;
        image2DMS colorImages[4] @rgba16f;

        vec4 leaf(image2DMS images[] @rgba16f, ivec2 pixel, int sampleIndex, int mode) {
            match mode {
                0 => {
                    int COUNT = 0;
                    vec4 dynamicValue = imageLoad(images[COUNT], pixel, sampleIndex);
                },
                _ => {
                },
            }
            return imageLoad(images[COUNT], pixel, sampleIndex);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return leaf(colorImages, ivec2(1, 2), 0, 0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'colorImages': 4 and 5"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_multisample_image_array_transitive_size_conflicts_are_rejected():
    crossgl = """
    shader MixedFixedMultisampleImageCallSites {
        image2DMS smallImages[2] @rgba16f;
        image2DMS largeImages[4] @rgba16f;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        ivec2 leafSize(image2DMS images[] @rgba16f, int layer) {
            return imageSize(images[layer]);
        }

        ivec2 midSize(image2DMS images[] @rgba16f, int layer) {
            return leafSize(images, layer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 a = midSize(smallImages, input.layer);
                ivec2 b = midSize(largeImages, input.layer);
                return vec4(float(a.x + b.x));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    expected_error = "Conflicting fixed resource array sizes for 'smallImages': 2 and 4"
    with pytest.raises(ValueError, match=expected_error):
        HLSLCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        MetalCodeGen().generate(shader_ast)
    with pytest.raises(ValueError, match=expected_error):
        GLSLCodeGen().generate(shader_ast)


def test_codegen_mixed_ssbo_multisample_integer_image_values_pack_fallbacks():
    crossgl = """
    shader MultisampleIntegerImageFallbacks {
        iimage2DMS signedImage @rgba32i;
        uimage2DMSArray unsignedLayers @rgba32ui;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedMsIntBlock {
            double flag;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
            ivec4 signedValue;
            uvec4 unsignedValue;
            int signedScalar;
            uint unsignedScalar;
        };

        UnsupportedMsIntBlock intBlock @glsl_buffer_block(std430) @binding(114);

        ivec2 readPixel(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        int readSample(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.sampleIndex;
        }

        ivec4 readSignedValue(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.signedValue;
        }

        uvec4 readUnsignedValue(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.unsignedValue;
        }

        int readSignedScalar(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.signedScalar;
        }

        uint readUnsignedScalar(UnsupportedMsIntBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.unsignedScalar;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                ivec4 signedDirect = imageLoad(signedImage, intBlock.pixel, intBlock.sampleIndex);
                uvec4 unsignedCall = imageLoad(unsignedLayers, readPixelLayer(intBlock), readSample(intBlock));
                int signedScalar = imageLoad(signedImage, readPixel(intBlock), readSample(intBlock));
                uint unsignedScalar = imageLoad(unsignedLayers, intBlock.pixelLayer, intBlock.sampleIndex);
                imageStore(signedImage, intBlock.pixel, intBlock.sampleIndex, intBlock.signedValue);
                imageStore(unsignedLayers, readPixelLayer(intBlock), readSample(intBlock), readUnsignedValue(intBlock));
                imageStore(signedImage, readPixel(intBlock), readSample(intBlock), readSignedScalar(intBlock));
                imageStore(unsignedLayers, intBlock.pixelLayer, intBlock.sampleIndex, intBlock.unsignedScalar);
                return vec4(float(signedDirect.x) + float(unsignedCall.x) + float(signedScalar) + float(unsignedScalar));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<int4> signedImage : register(t0);" in hlsl
    assert "Texture2DMSArray<uint4> unsignedLayers : register(t1);" in hlsl
    assert (
        "int4 signedDirect = signedImage.Load(int2(0) /* unsupported HLSL GLSL "
        "buffer block access intBlock: no target-side fallback declaration "
        "emitted */, 0 /* unsupported HLSL GLSL buffer block access intBlock: "
        "no target-side fallback declaration emitted */);" in hlsl
    )
    assert (
        "uint4 unsignedCall = unsignedLayers.Load(int3(0) /* unsupported HLSL "
        "GLSL buffer block function call readPixelLayer: target function "
        "omitted */, 0 /* unsupported HLSL GLSL buffer block function call "
        "readSample: target function omitted */);" in hlsl
    )
    assert (
        "int signedScalar = signedImage.Load(int2(0) /* unsupported HLSL GLSL "
        "buffer block function call readPixel: target function omitted */, 0 "
        "/* unsupported HLSL GLSL buffer block function call readSample: target "
        "function omitted */).x;" in hlsl
    )
    assert (
        "uint unsignedScalar = unsignedLayers.Load(int3(0) /* unsupported HLSL GLSL "
        "buffer block access intBlock: no target-side fallback declaration "
        "emitted */, 0 /* unsupported HLSL GLSL buffer block access intBlock: "
        "no target-side fallback declaration emitted */).x;" in hlsl
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<int4>" in hlsl
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMSArray<uint4>" in hlsl
    )
    assert "imageLoad(" not in hlsl
    assert "imageStore(" not in hlsl

    assert "texture2d_ms<int, access::read> signedImage [[texture(0)]]" in metal
    assert (
        "texture2d_ms_array<uint, access::read> unsignedLayers "
        "[[texture(1)]]" in metal
    )
    assert (
        "int4 signedDirect = signedImage.read(uint2(int2(0) /* unsupported "
        "Metal GLSL buffer block access intBlock: no target-side fallback "
        "declaration emitted */), uint(0 /* unsupported Metal GLSL buffer "
        "block access intBlock: no target-side fallback declaration emitted */));"
        in metal
    )
    assert (
        "uint4 unsignedCall = unsignedLayers.read(uint2((int3(0) "
        "/* unsupported Metal GLSL buffer block function call readPixelLayer: "
        "target function omitted */).xy), uint((int3(0) /* unsupported Metal "
        "GLSL buffer block function call readPixelLayer: target function "
        "omitted */).z), uint(0 /* unsupported Metal GLSL buffer block "
        "function call readSample: target function omitted */));" in metal
    )
    assert (
        "int signedScalar = signedImage.read(uint2(int2(0) /* unsupported "
        "Metal GLSL buffer block function call readPixel: target function "
        "omitted */), uint(0 /* unsupported Metal GLSL buffer block function "
        "call readSample: target function omitted */)).x;" in metal
    )
    assert (
        "uint unsignedScalar = unsignedLayers.read(uint2((int3(0) "
        "/* unsupported Metal GLSL buffer block access intBlock: no target-side "
        "fallback declaration emitted */).xy), uint((int3(0) /* unsupported "
        "Metal GLSL buffer block access intBlock: no target-side fallback "
        "declaration emitted */).z), uint(0 /* unsupported Metal GLSL buffer "
        "block access intBlock: no target-side fallback declaration emitted */)).x;"
        in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms<int, access::read>" in metal
    )
    assert (
        "unsupported Metal multisample image store: imageStore on "
        "texture2d_ms_array<uint, access::read>" in metal
    )
    assert (
        "signedImage.write(int4(0) /* unsupported Metal GLSL buffer block "
        "access intBlock: no target-side fallback declaration emitted */, "
        "uint2(int2(0) /* unsupported Metal GLSL buffer block access intBlock: "
        "no target-side fallback declaration emitted */), uint(0 /* unsupported "
        "Metal GLSL buffer block access intBlock: no target-side fallback "
        "declaration emitted */));" not in metal
    )
    assert (
        "unsignedLayers.write(uint4(0) /* unsupported Metal GLSL buffer block "
        "function call readUnsignedValue: target function omitted */, "
        "uint2((int3(0) /* unsupported Metal GLSL buffer block function call "
        "readPixelLayer: target function omitted */).xy), uint((int3(0) "
        "/* unsupported Metal GLSL buffer block function call readPixelLayer: "
        "target function omitted */).z), uint(0 /* unsupported Metal GLSL "
        "buffer block function call readSample: target function omitted */));"
        not in metal
    )
    assert (
        "signedImage.write(int4(0 /* unsupported Metal GLSL buffer block "
        "function call readSignedScalar: target function omitted */), "
        "uint2(int2(0) /* unsupported Metal GLSL buffer block function call "
        "readPixel: target function omitted */), uint(0 /* unsupported Metal "
        "GLSL buffer block function call readSample: target function omitted */));"
        not in metal
    )
    assert (
        "unsignedLayers.write(uint4(0u /* unsupported Metal GLSL buffer block "
        "access intBlock: no target-side fallback declaration emitted */), "
        "uint2((int3(0) /* unsupported Metal GLSL buffer block access intBlock: "
        "no target-side fallback declaration emitted */).xy), uint((int3(0) "
        "/* unsupported Metal GLSL buffer block access intBlock: no target-side "
        "fallback declaration emitted */).z), uint(0 /* unsupported Metal GLSL "
        "buffer block access intBlock: no target-side fallback declaration "
        "emitted */));" not in metal
    )
    assert "imageLoad(" not in metal
    assert "imageStore(" not in metal

    assert "layout(rgba32i, binding = 0) uniform iimage2DMS signedImage;" in glsl
    assert (
        "layout(rgba32ui, binding = 1) uniform uimage2DMSArray unsignedLayers;" in glsl
    )
    assert (
        "ivec4 signedDirect = imageLoad(signedImage, intBlock.pixel, "
        "intBlock.sampleIndex);" in glsl
    )
    assert (
        "uvec4 unsignedCall = imageLoad(unsignedLayers, "
        "readPixelLayer(intBlock), readSample(intBlock));" in glsl
    )
    assert (
        "int signedScalar = imageLoad(signedImage, readPixel(intBlock), "
        "readSample(intBlock)).x;" in glsl
    )
    assert (
        "uint unsignedScalar = imageLoad(unsignedLayers, intBlock.pixelLayer, "
        "intBlock.sampleIndex).x;" in glsl
    )
    assert (
        "imageStore(signedImage, intBlock.pixel, intBlock.sampleIndex, "
        "intBlock.signedValue);" in glsl
    )
    assert (
        "imageStore(unsignedLayers, readPixelLayer(intBlock), "
        "readSample(intBlock), readUnsignedValue(intBlock));" in glsl
    )
    assert (
        "imageStore(signedImage, readPixel(intBlock), readSample(intBlock), "
        "ivec4(readSignedScalar(intBlock)));" in glsl
    )
    assert (
        "imageStore(unsignedLayers, intBlock.pixelLayer, intBlock.sampleIndex, "
        "uvec4(intBlock.unsignedScalar));" in glsl
    )


def test_codegen_mixed_ssbo_multisample_image_atomics_use_sample_fallbacks():
    crossgl = """
    shader MultisampleImageAtomicFallbacks {
        uimage2DMS counters @r32ui;
        iimage2DMSArray signedLayers @r32i;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedMsAtomicBlock {
            double flag;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
            uint amount;
            int compareValue;
            int signedValue;
        };

        UnsupportedMsAtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(115);

        ivec2 readPixel(UnsupportedMsAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedMsAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        int readSample(UnsupportedMsAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.sampleIndex;
        }

        uint readAmount(UnsupportedMsAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.amount;
        }

        int readCompare(UnsupportedMsAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.compareValue;
        }

        int readSignedValue(UnsupportedMsAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.signedValue;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                uint atomicDirect = imageAtomicAdd(counters, atomicBlock.pixel, atomicBlock.sampleIndex, atomicBlock.amount);
                uint atomicCall = imageAtomicAdd(counters, readPixel(atomicBlock), readSample(atomicBlock), readAmount(atomicBlock));
                int swapDirect = imageAtomicCompSwap(signedLayers, atomicBlock.pixelLayer, atomicBlock.sampleIndex, atomicBlock.compareValue, atomicBlock.signedValue);
                int swapCall = imageAtomicCompSwap(signedLayers, readPixelLayer(atomicBlock), readSample(atomicBlock), readCompare(atomicBlock), readSignedValue(atomicBlock));
                return vec4(float(atomicDirect + atomicCall + uint(swapDirect) + uint(swapCall)));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<uint> counters : register(t0);" in hlsl
    assert "Texture2DMSArray<int> signedLayers : register(t1);" in hlsl
    assert (
        "uint atomicDirect = /* unsupported DirectX multisample image atomic: "
        "imageAtomicAdd on RWTexture2DMS<uint> */ 0u;" in hlsl
    )
    assert (
        "uint atomicCall = /* unsupported DirectX multisample image atomic: "
        "imageAtomicAdd on RWTexture2DMS<uint> */ 0u;" in hlsl
    )
    assert (
        "int swapDirect = /* unsupported DirectX multisample image atomic: "
        "imageAtomicCompSwap on RWTexture2DMSArray<int> */ 0;" in hlsl
    )
    assert (
        "int swapCall = /* unsupported DirectX multisample image atomic: "
        "imageAtomicCompSwap on RWTexture2DMSArray<int> */ 0;" in hlsl
    )
    assert "InterlockedAdd" not in hlsl
    assert "InterlockedCompareExchange" not in hlsl
    assert "imageAtomicAdd(counters" not in hlsl
    assert "imageAtomicCompSwap(signedLayers" not in hlsl

    assert "texture2d_ms<uint, access::read> counters [[texture(0)]]" in metal
    assert "texture2d_ms_array<int, access::read> signedLayers [[texture(1)]]" in metal
    assert (
        "uint atomicDirect = /* unsupported Metal multisample image atomic: "
        "imageAtomicAdd on texture2d_ms<uint, access::read> */ 0u;" in metal
    )
    assert (
        "uint atomicCall = /* unsupported Metal multisample image atomic: "
        "imageAtomicAdd on texture2d_ms<uint, access::read> */ 0u;" in metal
    )
    assert (
        "int swapDirect = /* unsupported Metal multisample image atomic: "
        "imageAtomicCompSwap on texture2d_ms_array<int, access::read> */ "
        "0;" in metal
    )
    assert (
        "int swapCall = /* unsupported Metal multisample image atomic: "
        "imageAtomicCompSwap on texture2d_ms_array<int, access::read> */ "
        "0;" in metal
    )
    assert "atomic_fetch_add" not in metal
    assert "imageAtomicAdd(counters" not in metal

    assert "layout(r32ui, binding = 0) uniform uimage2DMS counters;" in glsl
    assert "layout(r32i, binding = 1) uniform iimage2DMSArray signedLayers;" in glsl
    assert (
        "uint atomicDirect = imageAtomicAdd(counters, atomicBlock.pixel, "
        "atomicBlock.sampleIndex, atomicBlock.amount);" in glsl
    )
    assert (
        "uint atomicCall = imageAtomicAdd(counters, readPixel(atomicBlock), "
        "readSample(atomicBlock), readAmount(atomicBlock));" in glsl
    )
    assert (
        "int swapDirect = imageAtomicCompSwap(signedLayers, "
        "atomicBlock.pixelLayer, atomicBlock.sampleIndex, "
        "atomicBlock.compareValue, atomicBlock.signedValue);" in glsl
    )
    assert (
        "int swapCall = imageAtomicCompSwap(signedLayers, "
        "readPixelLayer(atomicBlock), readSample(atomicBlock), "
        "readCompare(atomicBlock), readSignedValue(atomicBlock));" in glsl
    )


def test_codegen_multisample_image_switch_match_fallbacks_stay_in_case_blocks():
    crossgl = """
    shader MultisampleSwitchMatchFallbacks {
        image2DMS colorImage @rgba16f;
        uimage2DMS counters @r32ui;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                uint total = 0u;
                switch (1) {
                    case 0:
                    case 1:
                        vec4 scoped = imageLoad(colorImage, ivec2(0), 0);
                        imageStore(colorImage, ivec2(0), 0, scoped + vec4(1.0));
                        total = total + imageAtomicAdd(counters, ivec2(0), 0, 1u);
                        break;
                    default:
                        vec4 scoped = imageLoad(colorImage, ivec2(1), 0);
                        imageStore(colorImage, ivec2(1), 0, scoped + vec4(2.0));
                        total = total + imageAtomicAdd(counters, ivec2(1), 0, 2u);
                        break;
                }
                match 2 {
                    0 => {
                        vec4 scoped = imageLoad(colorImage, ivec2(2), 0);
                        imageStore(colorImage, ivec2(2), 0, scoped + vec4(3.0));
                        total = total + imageAtomicAdd(counters, ivec2(2), 0, 3u);
                    }
                    _ => {
                        vec4 scoped = imageLoad(colorImage, ivec2(3), 0);
                        imageStore(colorImage, ivec2(3), 0, scoped + vec4(4.0));
                        total = total + imageAtomicAdd(counters, ivec2(3), 0, 4u);
                    }
                }
                return vec4(float(total));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<float4> colorImage : register(t0);" in hlsl
    assert "Texture2DMS<uint> counters : register(t1);" in hlsl
    assert "case 0:\n        case 1: {" in hlsl
    assert hlsl.count("case 0: {") == 1
    assert hlsl.count("float4 scoped") == 4
    assert hlsl.count("unsupported DirectX multisample image store") == 4
    assert hlsl.count("unsupported DirectX multisample image atomic") == 4
    assert "imageStore(" not in hlsl
    assert "imageAtomicAdd(counters" not in hlsl
    assert "InterlockedAdd" not in hlsl

    assert "texture2d_ms<float, access::read> colorImage [[texture(0)]]" in metal
    assert "texture2d_ms<uint, access::read> counters [[texture(1)]]" in metal
    assert "case 0:\n        case 1: {" in metal
    assert metal.count("case 0: {") == 1
    assert metal.count("float4 scoped") == 4
    assert metal.count("unsupported Metal multisample image store") == 4
    assert metal.count("unsupported Metal multisample image atomic") == 4
    assert "imageStore(" not in metal
    assert "imageAtomicAdd(counters" not in metal
    assert "atomic_fetch_add" not in metal
    assert ".write(" not in metal

    assert "layout(rgba16f, binding = 0) uniform image2DMS colorImage;" in glsl
    assert "layout(r32ui, binding = 1) uniform uimage2DMS counters;" in glsl
    assert "case 0:\n        case 1: {" in glsl
    assert glsl.count("case 0: {") == 1
    assert glsl.count("vec4 scoped") == 4
    assert glsl.count("imageStore(colorImage") == 4
    assert glsl.count("imageAtomicAdd(counters") == 4
    assert "unsupported" not in glsl


@pytest.mark.parametrize(
    ("operation", "intrinsic"),
    [
        ("imageAtomicMin", "InterlockedMin"),
        ("imageAtomicMax", "InterlockedMax"),
        ("imageAtomicAnd", "InterlockedAnd"),
        ("imageAtomicOr", "InterlockedOr"),
        ("imageAtomicXor", "InterlockedXor"),
        ("imageAtomicExchange", "InterlockedExchange"),
    ],
)
def test_codegen_multisample_image_atomic_variants_use_sample_argument(
    operation, intrinsic
):
    crossgl = f"""
    shader MultisampleImageAtomicVariant {{
        uimage2DMS counters @r32ui;
        struct VSOutput {{ vec2 uv; }};
        fragment {{
            vec4 main(VSOutput input) @ gl_FragColor {{
                uint old = {operation}(counters, ivec2(0), 1, 2u);
                return vec4(float(old));
            }}
        }}
    }}
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "Texture2DMS<uint> counters : register(t0);" in hlsl
    assert (
        "uint old = /* unsupported DirectX multisample image atomic: "
        f"{operation} on RWTexture2DMS<uint> */ 0u;" in hlsl
    )
    assert intrinsic not in hlsl
    assert f"{operation}(counters" not in hlsl

    assert (
        "uint old = /* unsupported Metal multisample image atomic: "
        f"{operation} on texture2d_ms<uint, access::read> */ 0u;" in metal
    )
    assert f"{operation}(counters" not in metal

    assert f"uint old = {operation}(counters, ivec2(0), 1, 2u);" in glsl


def test_codegen_float_image_atomic_exchange_uses_backend_diagnostics():
    crossgl = """
    shader FloatImageAtomicExchange {
        image2D scalarImage @r32f;
        image2DMS sampleImage @r32f;
        struct VSOutput { vec2 uv; };
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float oldScalar = imageAtomicExchange(scalarImage, ivec2(0), 1.0);
                float oldSample = imageAtomicExchange(sampleImage, ivec2(0), 1, 2.0);
                return vec4(oldScalar + oldSample);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float> scalarImage : register(u0);" in hlsl
    assert "Texture2DMS<float> sampleImage : register(t0);" in hlsl
    assert (
        "float oldScalar = /* unsupported DirectX image atomic resource call: "
        "imageAtomicExchange on RWTexture2D<float> */ 0.0;" in hlsl
    )
    assert (
        "float oldSample = /* unsupported DirectX multisample image atomic: "
        "imageAtomicExchange on RWTexture2DMS<float> */ 0.0;" in hlsl
    )
    assert "imageAtomicExchange(scalarImage" not in hlsl
    assert "imageAtomicExchange(sampleImage" not in hlsl

    assert "texture2d<float, access::read_write> scalarImage [[texture(0)]]" in metal
    assert "texture2d_ms<float, access::read> sampleImage [[texture(1)]]" in metal
    assert (
        "float oldScalar = /* unsupported Metal image atomic resource call: "
        "imageAtomicExchange on texture2d<float, access::read_write> */ 0.0;" in metal
    )
    assert (
        "float oldSample = /* unsupported Metal multisample image atomic: "
        "imageAtomicExchange on texture2d_ms<float, access::read> */ "
        "0.0;" in metal
    )
    assert "atomic_exchange" not in metal

    assert "layout(r32f, binding = 0) uniform image2D scalarImage;" in glsl
    assert "layout(r32f, binding = 1) uniform image2DMS sampleImage;" in glsl
    assert "float oldScalar = imageAtomicExchange(scalarImage, ivec2(0), 1.0);" in glsl
    assert (
        "float oldSample = imageAtomicExchange(sampleImage, ivec2(0), 1, 2.0);" in glsl
    )


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader VectorImageAtomicFormat {
                uimage2DMS counters @rgba32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = imageAtomicAdd(counters, ivec2(0), 1, 2u);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires r32i or r32ui image format, got rgba32ui",
        ),
        (
            """
            shader NarrowImageAtomicFormat {
                uimage2D counters @r16ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = imageAtomicAdd(counters, ivec2(0), 2u);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires r32i or r32ui image format, got r16ui",
        ),
        (
            """
            shader VectorFloatImageAtomicExchange {
                image2D values @rgba32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        float old = imageAtomicExchange(values, ivec2(0), 1.0);
                        return vec4(old);
                    }
                }
            }
            """,
            "requires r32i, r32ui, or r32f image format, got rgba32f",
        ),
        (
            """
            shader UnsignedFormatSignedAtomicData {
                iimage2D counters @r32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        int old = imageAtomicAdd(counters, ivec2(0), 1);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires uint data argument for r32ui images",
        ),
    ],
)
def test_codegen_rejects_image_atomic_format_and_value_mismatches(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader UnsignedAtomicAssignedToSigned {
                uimage2D counters @r32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        int old = imageAtomicAdd(counters, ivec2(0), 1u);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires uint result context for r32ui images: expected int",
        ),
        (
            """
            shader SignedAtomicAssignedToUnsigned {
                iimage2D counters @r32i;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = 0u;
                        old = imageAtomicAdd(counters, ivec2(0), 1);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires int result context for r32i images: expected uint",
        ),
        (
            """
            shader UnsignedAtomicReturnedAsSigned {
                uimage2D counters @r32ui;
                int readCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        return vec4(float(readCounter(counters, ivec2(0), 1u)));
                    }
                }
            }
            """,
            "requires uint result context for r32ui images: expected int",
        ),
        (
            """
            shader FloatExchangeAssignedToUnsigned {
                image2D values @r32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = imageAtomicExchange(values, ivec2(0), 1.0);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires float result context for r32f images: expected uint",
        ),
    ],
)
def test_codegen_rejects_image_atomic_result_context_mismatches(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


def test_codegen_allows_explicit_image_atomic_result_casts():
    crossgl = """
    shader ExplicitAtomicResultCasts {
        uimage2D counters @r32ui;
        image2D target @r32f;
        struct VSOutput { vec2 uv; };
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float scalarCast = float(imageAtomicAdd(counters, ivec2(0), 1u));
                imageStore(
                    target,
                    ivec2(0),
                    float(imageAtomicAdd(counters, ivec2(1), 2u))
                );
                return vec4(scalarCast);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "float scalarCast = float(imageAtomicAdd_uimage2D(counters, int2(0), 1u));"
        in hlsl
    )
    assert (
        "target[int2(0)] = float(imageAtomicAdd_uimage2D(counters, int2(1), 2u));"
        in hlsl
    )

    assert (
        "float scalarCast = float(counters.atomic_fetch_add(uint2(int2(0)), 1u).x);"
        in metal
    )
    assert (
        "target.write(float4(float(counters.atomic_fetch_add(uint2(int2(1)), 2u).x)), "
        "uint2(int2(0)));" in metal
    )

    assert "float scalarCast = float(imageAtomicAdd(counters, ivec2(0), 1u));" in glsl
    assert (
        "imageStore(target, ivec2(0), "
        "vec4(float(imageAtomicAdd(counters, ivec2(1), 2u))));" in glsl
    )


@pytest.mark.parametrize(
    "crossgl",
    [
        """
        shader HiddenAtomicBinaryFloatResult {
            uimage2D counters @r32ui;
            struct VSOutput { vec2 uv; };
            fragment {
                vec4 main(VSOutput input) @ gl_FragColor {
                    float old = imageAtomicAdd(counters, ivec2(0), 1u) + 1.0;
                    return vec4(old);
                }
            }
        }
        """,
        """
        shader HiddenAtomicTernaryFloatResult {
            uimage2D counters @r32ui;
            struct VSOutput { vec2 uv; };
            fragment {
                vec4 main(VSOutput input) @ gl_FragColor {
                    float old = input.uv.x > 0.0
                        ? imageAtomicAdd(counters, ivec2(0), 1u)
                        : 0.0;
                    return vec4(old);
                }
            }
        }
        """,
        """
        shader HiddenAtomicImageStoreFloatResult {
            uimage2D counters @r32ui;
            image2D target @r32f;
            struct VSOutput { vec2 uv; };
            fragment {
                vec4 main(VSOutput input) @ gl_FragColor {
                    imageStore(
                        target,
                        ivec2(0),
                        imageAtomicAdd(counters, ivec2(0), 1u)
                    );
                    return vec4(0.0);
                }
            }
        }
        """,
    ],
)
def test_codegen_rejects_hidden_image_atomic_result_contexts(crossgl):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(
            ValueError,
            match="requires uint result context for r32ui images: expected float",
        ):
            generator.generate(shader_ast)


def test_codegen_allows_explicit_image_load_result_casts():
    crossgl = """
    shader ExplicitImageLoadResultCasts {
        uimage2D counters @r32ui;
        image2D values @r32f;
        image2D target @r32f;
        struct VSOutput { vec2 uv; };
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float floatCast = float(imageLoad(counters, ivec2(0)));
                uint uintCast = uint(imageLoad(values, ivec2(1)));
                imageStore(target, ivec2(0), float(imageLoad(counters, ivec2(2))));
                return vec4(floatCast + float(uintCast));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "float floatCast = float(counters[int2(0)]);" in hlsl
    assert "uint uintCast = uint(values[int2(1)]);" in hlsl
    assert "target[int2(0)] = float(counters[int2(2)]);" in hlsl

    assert "float floatCast = float(counters.read(uint2(int2(0))).x);" in metal
    assert "uint uintCast = uint(values.read(uint2(int2(1))).x);" in metal
    assert (
        "target.write(float4(float(counters.read(uint2(int2(2))).x)), "
        "uint2(int2(0)));" in metal
    )

    assert "float floatCast = float(imageLoad(counters, ivec2(0)).x);" in glsl
    assert "uint uintCast = uint(imageLoad(values, ivec2(1)).x);" in glsl
    assert (
        "imageStore(target, ivec2(0), "
        "vec4(float(imageLoad(counters, ivec2(2)).x)));" in glsl
    )


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader HiddenUnsignedImageLoadFloatResult {
                uimage2D counters @r32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        float old = imageLoad(counters, ivec2(0));
                        return vec4(old);
                    }
                }
            }
            """,
            "requires uint result context for r32ui images: expected float",
        ),
        (
            """
            shader HiddenFloatImageLoadUnsignedResult {
                image2D values @r32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = imageLoad(values, ivec2(0));
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires float result context for r32f images: expected uint",
        ),
        (
            """
            shader HiddenImageLoadStoreFloatResult {
                uimage2D counters @r32ui;
                image2D target @r32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), imageLoad(counters, ivec2(0)));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires float value for r32f images",
        ),
    ],
)
def test_codegen_rejects_hidden_image_load_result_contexts(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


def test_codegen_allows_matching_image_load_result_contexts():
    crossgl = """
    shader ImageLoadResultContexts {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        image2D rgbaFloat @rgba32f;
        uimage2D rgUnsigned @rg32ui;
        struct VSOutput { vec2 uv; };
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float scalarFromRg = imageLoad(rgFloat, ivec2(0));
                vec2 rgExact = imageLoad(rgFloat, ivec2(1));
                vec4 rgbaExact = imageLoad(rgbaFloat, ivec2(2));
                vec2 explicitScalarCast = vec2(imageLoad(scalarFloat, ivec2(3)));
                uvec2 unsignedExact = imageLoad(rgUnsigned, ivec2(4));
                return vec4(
                    scalarFromRg
                    + rgExact.x
                    + rgbaExact.y
                    + explicitScalarCast.x
                    + float(unsignedExact.x)
                );
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "float scalarFromRg = rgFloat[int2(0)].x;" in hlsl
    assert "float2 rgExact = rgFloat[int2(1)];" in hlsl
    assert "float4 rgbaExact = rgbaFloat[int2(2)];" in hlsl
    assert "float2 explicitScalarCast = float2(scalarFloat[int2(3)]);" in hlsl
    assert "uint2 unsignedExact = rgUnsigned[int2(4)];" in hlsl

    assert "float scalarFromRg = rgFloat.read(uint2(int2(0))).x;" in metal
    assert "float2 rgExact = rgFloat.read(uint2(int2(1))).xy;" in metal
    assert "float4 rgbaExact = rgbaFloat.read(uint2(int2(2)));" in metal
    assert (
        "float2 explicitScalarCast = "
        "float2(scalarFloat.read(uint2(int2(3))).x);" in metal
    )
    assert "uint2 unsignedExact = rgUnsigned.read(uint2(int2(4))).xy;" in metal

    assert "float scalarFromRg = imageLoad(rgFloat, ivec2(0)).x;" in glsl
    assert "vec2 rgExact = imageLoad(rgFloat, ivec2(1)).xy;" in glsl
    assert "vec4 rgbaExact = imageLoad(rgbaFloat, ivec2(2));" in glsl
    assert "vec2 explicitScalarCast = vec2(imageLoad(scalarFloat, ivec2(3)).x);" in glsl
    assert "uvec2 unsignedExact = imageLoad(rgUnsigned, ivec2(4)).xy;" in glsl


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader ThreeComponentLoadFromRgImage {
                image2D source @rg32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        vec3 value = imageLoad(source, ivec2(0));
                        return vec4(value, 1.0);
                    }
                }
            }
            """,
            "requires scalar or 2-component result context for rg32f images",
        ),
        (
            """
            shader VectorLoadFromScalarImage {
                image2D source @r32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        vec2 value = imageLoad(source, ivec2(0));
                        return vec4(value, 0.0, 1.0);
                    }
                }
            }
            """,
            "requires scalar result context for r32f images",
        ),
        (
            """
            shader UnsignedVectorLoadFromFloatRgImage {
                image2D source @rg32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uvec2 value = imageLoad(source, ivec2(0));
                        return vec4(float(value.x));
                    }
                }
            }
            """,
            "requires float result context for rg32f images: expected uint",
        ),
        (
            """
            shader TwoComponentLoadFromRgbaImage {
                image2D source @rgba32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        vec2 value = imageLoad(source, ivec2(0));
                        return vec4(value, 0.0, 1.0);
                    }
                }
            }
            """,
            "requires scalar or 4-component result context for rgba32f images",
        ),
    ],
)
def test_codegen_rejects_mismatched_image_load_result_contexts(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


def test_codegen_allows_matching_multicomponent_image_store_values():
    crossgl = """
    shader MultiImageStoreValueKinds {
        image2D source @rg32f;
        image2D floatTarget @rg32f;
        image2D rgbaTarget @rgba32f;
        uimage2D unsignedTarget @rg32ui;
        struct VSOutput { vec2 uv; };
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                imageStore(unsignedTarget, ivec2(0), 1u);
                imageStore(unsignedTarget, ivec2(1), uvec2(2u));
                imageStore(floatTarget, ivec2(2), 1.0);
                imageStore(floatTarget, ivec2(3), vec2(2.0));
                imageStore(floatTarget, ivec2(4), imageLoad(source, ivec2(5)));
                imageStore(rgbaTarget, ivec2(6), vec4(3.0));
                imageStore(rgbaTarget, ivec2(7), 4.0);
                imageStore(floatTarget, ivec2(8), float2(5.0));
                return vec4(0.0);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "unsignedTarget[int2(0)] = uint2(1u, 0u);" in hlsl
    assert "unsignedTarget[int2(1)] = uint2(2u);" in hlsl
    assert "floatTarget[int2(2)] = float2(1.0, 0.0);" in hlsl
    assert "floatTarget[int2(3)] = float2(2.0);" in hlsl
    assert "floatTarget[int2(4)] = source[int2(5)];" in hlsl
    assert "rgbaTarget[int2(6)] = float4(3.0);" in hlsl
    assert "rgbaTarget[int2(7)] = float4(4.0);" in hlsl
    assert "floatTarget[int2(8)] = float2(5.0);" in hlsl

    assert "unsignedTarget.write(uint4(1u, 0u, 0u, 0u), uint2(int2(0)));" in metal
    assert "unsignedTarget.write(uint4(uint2(2u), 0u, 0u), uint2(int2(1)));" in metal
    assert "floatTarget.write(float4(1.0, 0.0, 0.0, 0.0), uint2(int2(2)));" in metal
    assert "floatTarget.write(float4(float2(2.0), 0.0, 0.0), uint2(int2(3)));" in metal
    assert (
        "floatTarget.write(float4(source.read(uint2(int2(5))).xy, 0.0, 0.0), "
        "uint2(int2(4)));" in metal
    )
    assert "rgbaTarget.write(float4(3.0), uint2(int2(6)));" in metal
    assert "rgbaTarget.write(float4(4.0), uint2(int2(7)));" in metal
    assert "floatTarget.write(float4(float2(5.0), 0.0, 0.0), uint2(int2(8)));" in metal

    assert "imageStore(unsignedTarget, ivec2(0), uvec4(1u, 0u, 0u, 0u));" in glsl
    assert "imageStore(unsignedTarget, ivec2(1), uvec4(uvec2(2u), 0u, 0u));" in glsl
    assert "imageStore(floatTarget, ivec2(2), vec4(1.0, 0.0, 0.0, 0.0));" in glsl
    assert "imageStore(floatTarget, ivec2(3), vec4(vec2(2.0), 0.0, 0.0));" in glsl
    assert (
        "imageStore(floatTarget, ivec2(4), "
        "vec4(imageLoad(source, ivec2(5)).xy, 0.0, 0.0));" in glsl
    )
    assert "imageStore(rgbaTarget, ivec2(6), vec4(3.0));" in glsl
    assert "imageStore(rgbaTarget, ivec2(7), vec4(4.0));" in glsl
    assert "imageStore(floatTarget, ivec2(8), vec4(vec2(5.0), 0.0, 0.0));" in glsl


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader FloatScalarStoredToUnsignedRgImage {
                uimage2D target @rg32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), 1.0);
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires uint value for rg32ui images",
        ),
        (
            """
            shader FloatVectorStoredToUnsignedRgImage {
                uimage2D target @rg32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), vec2(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires uint value for rg32ui images",
        ),
        (
            """
            shader FloatVectorStoredToSignedRgbaImage {
                iimage2D target @rgba32i;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), vec4(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires int value for rgba32i images",
        ),
        (
            """
            shader UnsignedVectorStoredToFloatRgImage {
                image2D target @rg32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), uvec2(1u));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires float value for rg32f images",
        ),
    ],
)
def test_codegen_rejects_mismatched_multicomponent_image_store_values(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader VectorStoredToScalarImage {
                image2D target @r32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), vec2(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires scalar value for r32f images",
        ),
        (
            """
            shader TernaryVectorStoredToScalarImage {
                image2D target @r32f;
                compute {
                    void main(bool choose) {
                        float value = 0.5;
                        imageStore(target, ivec2(0), choose ? value : vec2(1.0));
                    }
                }
            }
            """,
            "requires scalar value for r32f images",
        ),
        (
            """
            shader AliasVectorStoredToScalarImage {
                image2D target @r32f;
                compute {
                    void main() {
                        imageStore(target, ivec2(0), float2(1.0, 2.0));
                    }
                }
            }
            """,
            "requires scalar value for r32f images",
        ),
        (
            """
            shader ThreeComponentStoredToRgImage {
                image2D target @rg32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), vec3(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires scalar or 2-component value for rg32f images",
        ),
        (
            """
            shader FourComponentStoredToRgImage {
                uimage2D target @rg32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), uvec4(1u));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires scalar or 2-component value for rg32ui images",
        ),
        (
            """
            shader TwoComponentStoredToRgbaImage {
                image2D target @rgba32f;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(target, ivec2(0), vec2(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires scalar or 4-component value for rgba32f images",
        ),
    ],
)
def test_codegen_rejects_mismatched_image_store_value_shapes(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


@pytest.mark.parametrize(
    ("crossgl", "message"),
    [
        (
            """
            shader BadImageLoad {
                image2D img @rgba8;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        return imageLoad(img, ivec2(0), 0);
                    }
                }
            }
            """,
            "accepts at most 2 argument",
        ),
        (
            """
            shader BadImageStore {
                image2D img @rgba8;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(img, ivec2(0), 0, vec4(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "accepts at most 3 argument",
        ),
        (
            """
            shader MissingMultisampleLoad {
                image2DMS img @rgba8;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        return imageLoad(img, ivec2(0));
                    }
                }
            }
            """,
            "requires image, coordinate, and sample index arguments",
        ),
        (
            """
            shader MissingMultisampleStore {
                image2DMS img @rgba8;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        imageStore(img, ivec2(0), vec4(1.0));
                        return vec4(0.0);
                    }
                }
            }
            """,
            "requires image, coordinate, sample index, and value arguments",
        ),
        (
            """
            shader FloatMultisampleIndex {
                image2DMS img @rgba8;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        return imageLoad(img, ivec2(0), 0.5);
                    }
                }
            }
            """,
            "requires a scalar integer sample index argument",
        ),
        (
            """
            shader MissingMultisampleAtomic {
                uimage2DMS img @r32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = imageAtomicAdd(img, ivec2(0), 1u);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires image, coordinate, sample index, and value arguments",
        ),
        (
            """
            shader MissingMultisampleCompareSwap {
                iimage2DMS img @r32i;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        int old = imageAtomicCompSwap(img, ivec2(0), 1, 2);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires image, coordinate, sample index, compare, and value arguments",
        ),
        (
            """
            shader FloatMultisampleAtomicIndex {
                uimage2DMS img @r32ui;
                struct VSOutput { vec2 uv; };
                fragment {
                    vec4 main(VSOutput input) @ gl_FragColor {
                        uint old = imageAtomicAdd(img, ivec2(0), 0.5, 1u);
                        return vec4(float(old));
                    }
                }
            }
            """,
            "requires a scalar integer sample index argument",
        ),
    ],
)
def test_codegen_multisample_image_argument_validation(crossgl, message):
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    for generator in (HLSLCodeGen(), MetalCodeGen(), GLSLCodeGen()):
        with pytest.raises(ValueError, match=message):
            generator.generate(shader_ast)


def test_codegen_mixed_ssbo_unsupported_sampling_vectors_are_typed_diagnostics():
    crossgl = """
    shader ResourceVectorFallbacks {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedSamplerCoordBlock {
            double flag;
            vec2 uv;
            vec2 dx;
            vec2 dy;
            ivec2 pixel;
        };

        UnsupportedSamplerCoordBlock coordBlock @glsl_buffer_block(std430) @binding(102);

        vec2 readUv(UnsupportedSamplerCoordBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        ivec2 readPixel(UnsupportedSamplerCoordBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 lodDirect = textureLod(textures[0], samplers[0], coordBlock.uv, 2.0);
                vec4 lodCall = textureLod(textures[0], samplers[0], readUv(coordBlock), 2.0);
                vec4 gradDirect = textureGrad(textures[0], samplers[0], coordBlock.uv, coordBlock.dx, coordBlock.dy);
                vec4 gradCall = textureGrad(textures[0], samplers[0], readUv(coordBlock), readUv(coordBlock), coordBlock.dy);
                vec4 gatheredDirect = textureGather(textures[0], samplers[0], coordBlock.uv);
                vec4 gatheredCall = textureGather(textures[0], samplers[0], readUv(coordBlock));
                vec4 fetchedDirect = texelFetch(textures[0], coordBlock.pixel, 0);
                vec4 fetchedCall = texelFetch(textures[0], readPixel(coordBlock), 0);
                return lodDirect + lodCall + gradDirect + gradCall + gatheredDirect + gatheredCall + fetchedDirect + fetchedCall;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "float4 lodDirect = textures[0].SampleLevel(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */, 2.0);" in hlsl
    )
    assert (
        "float4 lodCall = textures[0].SampleLevel(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, 2.0);" in hlsl
    )
    assert (
        "float4 gradDirect = textures[0].SampleGrad(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */, float2(0) /* unsupported HLSL GLSL "
        "buffer block access coordBlock: no target-side fallback declaration "
        "emitted */, float2(0) /* unsupported HLSL GLSL buffer block access "
        "coordBlock: no target-side fallback declaration emitted */);" in hlsl
    )
    assert (
        "float4 gradCall = textures[0].SampleGrad(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, float2(0) /* unsupported HLSL GLSL buffer block "
        "function call readUv: target function omitted */, float2(0) "
        "/* unsupported HLSL GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "float4 gatheredDirect = textures[0].Gather(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "float4 gatheredCall = textures[0].Gather(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */);" in hlsl
    )
    assert (
        "float4 fetchedDirect = textures[0].Load(int3(int2(0) /* unsupported "
        "HLSL GLSL buffer block access coordBlock: no target-side fallback "
        "declaration emitted */, 0));" in hlsl
    )
    assert (
        "float4 fetchedCall = textures[0].Load(int3(int2(0) /* unsupported "
        "HLSL GLSL buffer block function call readPixel: target function "
        "omitted */, 0));" in hlsl
    )
    assert "SampleLevel(samplers[0], 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert "SampleGrad(samplers[0], 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert "Gather(samplers[0], 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert "Load(int3(0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert "coordBlock.uv" not in hlsl
    assert "coordBlock.pixel" not in hlsl

    assert (
        "float4 lodDirect = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */, level(2.0));" in metal
    )
    assert (
        "float4 lodCall = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, level(2.0));" in metal
    )
    assert (
        "float4 gradDirect = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */, gradient2d(float2(0) /* unsupported "
        "Metal GLSL buffer block access coordBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported Metal GLSL buffer "
        "block access coordBlock: no target-side fallback declaration emitted */));"
        in metal
    )
    assert (
        "float4 gradCall = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, gradient2d(float2(0) /* unsupported Metal GLSL "
        "buffer block function call readUv: target function omitted */, "
        "float2(0) /* unsupported Metal GLSL buffer block access coordBlock: "
        "no target-side fallback declaration emitted */));" in metal
    )
    assert (
        "float4 gatheredDirect = textures[0].gather(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block access coordBlock: no target-side "
        "fallback declaration emitted */);" in metal
    )
    assert (
        "float4 gatheredCall = textures[0].gather(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */);" in metal
    )
    assert (
        "float4 fetchedDirect = textures[0].read(int2(0) /* unsupported Metal "
        "GLSL buffer block access coordBlock: no target-side fallback declaration "
        "emitted */, 0);" in metal
    )
    assert (
        "float4 fetchedCall = textures[0].read(int2(0) /* unsupported Metal "
        "GLSL buffer block function call readPixel: target function omitted */, 0);"
        in metal
    )
    assert ".sample(samplers[0], 0 /* unsupported Metal GLSL buffer" not in metal
    assert ".gather(samplers[0], 0 /* unsupported Metal GLSL buffer" not in metal
    assert ".read(0 /* unsupported Metal GLSL buffer" not in metal
    assert "coordBlock.uv" not in metal
    assert "coordBlock.pixel" not in metal

    assert "vec4 lodDirect = textureLod(textures[0], coordBlock.uv, 2.0);" in glsl
    assert "vec4 lodCall = textureLod(textures[0], readUv(coordBlock), 2.0);" in glsl
    assert (
        "vec4 gradDirect = textureGrad(textures[0], coordBlock.uv, "
        "coordBlock.dx, coordBlock.dy);" in glsl
    )
    assert (
        "vec4 gradCall = textureGrad(textures[0], readUv(coordBlock), "
        "readUv(coordBlock), coordBlock.dy);" in glsl
    )
    assert "vec4 gatheredDirect = textureGather(textures[0], coordBlock.uv);" in glsl
    assert "vec4 gatheredCall = textureGather(textures[0], readUv(coordBlock));" in glsl
    assert "vec4 fetchedDirect = texelFetch(textures[0], coordBlock.pixel, 0);" in glsl
    assert (
        "vec4 fetchedCall = texelFetch(textures[0], readPixel(coordBlock), 0);" in glsl
    )


def test_codegen_mixed_ssbo_unsupported_sampling_offsets_are_typed_diagnostics():
    crossgl = """
    shader ResourceOffsetFallbacks {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedSamplerOffsetBlock {
            double flag;
            vec2 uv;
            vec2 dx;
            vec2 dy;
            ivec2 pixel;
            ivec2 offset;
        };

        UnsupportedSamplerOffsetBlock offsetBlock @glsl_buffer_block(std430) @binding(103);

        vec2 readUv(UnsupportedSamplerOffsetBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        ivec2 readOffset(UnsupportedSamplerOffsetBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.offset;
        }

        ivec2 readPixel(UnsupportedSamplerOffsetBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 lodDirect = textureLodOffset(textures[0], samplers[0], offsetBlock.uv, 2.0, offsetBlock.offset);
                vec4 lodCall = textureLodOffset(textures[0], samplers[0], readUv(offsetBlock), 2.0, readOffset(offsetBlock));
                vec4 gradDirect = textureGradOffset(textures[0], samplers[0], offsetBlock.uv, offsetBlock.dx, offsetBlock.dy, offsetBlock.offset);
                vec4 gradCall = textureGradOffset(textures[0], samplers[0], readUv(offsetBlock), readUv(offsetBlock), offsetBlock.dy, readOffset(offsetBlock));
                vec4 gatheredDirect = textureGatherOffset(textures[0], samplers[0], offsetBlock.uv, offsetBlock.offset);
                vec4 gatheredCall = textureGatherOffset(textures[0], samplers[0], readUv(offsetBlock), readOffset(offsetBlock));
                vec4 fetchedDirect = texelFetchOffset(textures[0], offsetBlock.pixel, 0, offsetBlock.offset);
                vec4 fetchedCall = texelFetchOffset(textures[0], readPixel(offsetBlock), 0, readOffset(offsetBlock));
                return lodDirect + lodCall + gradDirect + gradCall + gatheredDirect + gatheredCall + fetchedDirect + fetchedCall;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "float4 lodDirect = textures[0].SampleLevel(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, 2.0, int2(0) /* unsupported HLSL GLSL "
        "buffer block access offsetBlock: no target-side fallback declaration "
        "emitted */);" in hlsl
    )
    assert (
        "float4 lodCall = textures[0].SampleLevel(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, 2.0, int2(0) /* unsupported HLSL GLSL buffer block "
        "function call readOffset: target function omitted */);" in hlsl
    )
    assert (
        "float4 gradDirect = textures[0].SampleGrad(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, float2(0) /* unsupported HLSL GLSL "
        "buffer block access offsetBlock: no target-side fallback declaration "
        "emitted */, float2(0) /* unsupported HLSL GLSL buffer block access "
        "offsetBlock: no target-side fallback declaration emitted */, int2(0) "
        "/* unsupported HLSL GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */);" in hlsl
    )
    assert (
        "float4 gradCall = textures[0].SampleGrad(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, float2(0) /* unsupported HLSL GLSL buffer block "
        "function call readUv: target function omitted */, float2(0) "
        "/* unsupported HLSL GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, int2(0) /* unsupported HLSL GLSL "
        "buffer block function call readOffset: target function omitted */);" in hlsl
    )
    assert (
        "float4 gatheredDirect = textures[0].Gather(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, int2(0) /* unsupported HLSL GLSL "
        "buffer block access offsetBlock: no target-side fallback declaration "
        "emitted */);" in hlsl
    )
    assert (
        "float4 gatheredCall = textures[0].Gather(samplers[0], float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, int2(0) /* unsupported HLSL GLSL buffer block "
        "function call readOffset: target function omitted */);" in hlsl
    )
    assert (
        "float4 fetchedDirect = textures[0].Load(int3((int2(0) /* unsupported "
        "HLSL GLSL buffer block access offsetBlock: no target-side fallback "
        "declaration emitted */ + int2(0) /* unsupported HLSL GLSL buffer block "
        "access offsetBlock: no target-side fallback declaration emitted */), 0));"
        in hlsl
    )
    assert (
        "float4 fetchedCall = textures[0].Load(int3((int2(0) /* unsupported "
        "HLSL GLSL buffer block function call readPixel: target function omitted */ "
        "+ int2(0) /* unsupported HLSL GLSL buffer block function call readOffset: "
        "target function omitted */), 0));" in hlsl
    )
    assert "SampleLevel(samplers[0], 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert "SampleGrad(samplers[0], 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert "Gather(samplers[0], 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert ", 0 /* unsupported HLSL GLSL buffer block access offsetBlock" not in hlsl
    assert "offsetBlock.uv" not in hlsl
    assert "offsetBlock.offset" not in hlsl

    assert (
        "float4 lodDirect = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, level(2.0), int2(0) /* unsupported "
        "Metal GLSL buffer block access offsetBlock: no target-side fallback "
        "declaration emitted */);" in metal
    )
    assert (
        "float4 lodCall = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, level(2.0), int2(0) /* unsupported Metal GLSL "
        "buffer block function call readOffset: target function omitted */);" in metal
    )
    assert (
        "float4 gradDirect = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, gradient2d(float2(0) /* unsupported "
        "Metal GLSL buffer block access offsetBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported Metal GLSL buffer "
        "block access offsetBlock: no target-side fallback declaration emitted */), "
        "int2(0) /* unsupported Metal GLSL buffer block access offsetBlock: "
        "no target-side fallback declaration emitted */);" in metal
    )
    assert (
        "float4 gradCall = textures[0].sample(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, gradient2d(float2(0) /* unsupported Metal GLSL "
        "buffer block function call readUv: target function omitted */, "
        "float2(0) /* unsupported Metal GLSL buffer block access offsetBlock: "
        "no target-side fallback declaration emitted */), int2(0) /* unsupported "
        "Metal GLSL buffer block function call readOffset: target function "
        "omitted */);" in metal
    )
    assert (
        "float4 gatheredDirect = textures[0].gather(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block access offsetBlock: no target-side "
        "fallback declaration emitted */, int2(0) /* unsupported Metal GLSL "
        "buffer block access offsetBlock: no target-side fallback declaration "
        "emitted */);" in metal
    )
    assert (
        "float4 gatheredCall = textures[0].gather(samplers[0], float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, int2(0) /* unsupported Metal GLSL buffer block "
        "function call readOffset: target function omitted */);" in metal
    )
    assert (
        "float4 fetchedDirect = textures[0].read((int2(0) /* unsupported Metal "
        "GLSL buffer block access offsetBlock: no target-side fallback declaration "
        "emitted */ + int2(0) /* unsupported Metal GLSL buffer block access "
        "offsetBlock: no target-side fallback declaration emitted */), 0);" in metal
    )
    assert (
        "float4 fetchedCall = textures[0].read((int2(0) /* unsupported Metal "
        "GLSL buffer block function call readPixel: target function omitted */ "
        "+ int2(0) /* unsupported Metal GLSL buffer block function call readOffset: "
        "target function omitted */), 0);" in metal
    )
    assert ".sample(samplers[0], 0 /* unsupported Metal GLSL buffer" not in metal
    assert ".gather(samplers[0], 0 /* unsupported Metal GLSL buffer" not in metal
    assert ", 0 /* unsupported Metal GLSL buffer block access offsetBlock" not in metal
    assert "offsetBlock.uv" not in metal
    assert "offsetBlock.offset" not in metal

    assert (
        "vec4 lodDirect = textureLodOffset(textures[0], offsetBlock.uv, "
        "2.0, offsetBlock.offset);" in glsl
    )
    assert (
        "vec4 lodCall = textureLodOffset(textures[0], readUv(offsetBlock), "
        "2.0, readOffset(offsetBlock));" in glsl
    )
    assert (
        "vec4 gradDirect = textureGradOffset(textures[0], offsetBlock.uv, "
        "offsetBlock.dx, offsetBlock.dy, offsetBlock.offset);" in glsl
    )
    assert (
        "vec4 gradCall = textureGradOffset(textures[0], readUv(offsetBlock), "
        "readUv(offsetBlock), offsetBlock.dy, readOffset(offsetBlock));" in glsl
    )
    assert (
        "vec4 gatheredDirect = textureGatherOffset(textures[0], "
        "offsetBlock.uv, offsetBlock.offset);" in glsl
    )
    assert (
        "vec4 gatheredCall = textureGatherOffset(textures[0], "
        "readUv(offsetBlock), readOffset(offsetBlock));" in glsl
    )
    assert (
        "vec4 fetchedDirect = texelFetchOffset(textures[0], offsetBlock.pixel, "
        "0, offsetBlock.offset);" in glsl
    )
    assert (
        "vec4 fetchedCall = texelFetchOffset(textures[0], readPixel(offsetBlock), "
        "0, readOffset(offsetBlock));" in glsl
    )


def test_codegen_mixed_ssbo_unsupported_projected_compare_calls_infer_types():
    crossgl = """
    shader ShadowProjectedFallbacks {
        sampler2D colorMap;
        sampler linearSampler;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedProjectedShadowBlock {
            double flag;
            vec2 uv;
            vec3 uvq;
            float depth;
            vec2 dx;
            vec2 dy;
            ivec2 offset;
        };

        UnsupportedProjectedShadowBlock shadowBlock @glsl_buffer_block(std430) @binding(104);

        vec2 readUv(UnsupportedProjectedShadowBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        vec3 readUvq(UnsupportedProjectedShadowBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uvq;
        }

        float readDepth(UnsupportedProjectedShadowBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.depth;
        }

        ivec2 readOffset(UnsupportedProjectedShadowBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.offset;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 projectedDirect = textureProjOffset(colorMap, linearSampler, shadowBlock.uvq, shadowBlock.offset);
                vec4 projectedCall = textureProjGradOffset(colorMap, linearSampler, readUvq(shadowBlock), shadowBlock.dx, shadowBlock.dy, readOffset(shadowBlock));
                float compareDirect = textureCompareOffset(shadowMap, compareSampler, shadowBlock.uv, shadowBlock.depth, shadowBlock.offset);
                float compareCall = textureCompareGradOffset(shadowMap, compareSampler, readUv(shadowBlock), readDepth(shadowBlock), shadowBlock.dx, shadowBlock.dy, readOffset(shadowBlock));
                float compareProjDirect = textureCompareProjOffset(shadowMap, compareSampler, shadowBlock.uvq, shadowBlock.depth, shadowBlock.offset);
                float compareProjCall = textureCompareProjGradOffset(shadowMap, compareSampler, readUvq(shadowBlock), readDepth(shadowBlock), shadowBlock.dx, shadowBlock.dy, readOffset(shadowBlock));
                return projectedDirect + projectedCall + vec4(compareDirect + compareCall + compareProjDirect + compareProjCall);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "float4 projectedCall = colorMap.SampleGrad(linearSampler, (float3(0) "
        "/* unsupported HLSL GLSL buffer block function call readUvq: target "
        "function omitted */).xy / (float3(0) /* unsupported HLSL GLSL buffer "
        "block function call readUvq: target function omitted */).z, float2(0) "
        "/* unsupported HLSL GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */, float2(0) /* unsupported HLSL GLSL "
        "buffer block access shadowBlock: no target-side fallback declaration "
        "emitted */, int2(0) /* unsupported HLSL GLSL buffer block function call "
        "readOffset: target function omitted */);" in hlsl
    )
    assert (
        "float compareCall = shadowMap.SampleCmpGrad(compareSampler, float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, 0 /* unsupported HLSL GLSL buffer block function "
        "call readDepth: target function omitted */, float2(0) /* unsupported "
        "HLSL GLSL buffer block access shadowBlock: no target-side fallback "
        "declaration emitted */, float2(0) /* unsupported HLSL GLSL buffer block "
        "access shadowBlock: no target-side fallback declaration emitted */, "
        "int2(0) /* unsupported HLSL GLSL buffer block function call readOffset: "
        "target function omitted */);" in hlsl
    )
    assert (
        "float compareProjCall = shadowMap.SampleCmpGrad(compareSampler, "
        "(float3(0) /* unsupported HLSL GLSL buffer block function call readUvq: "
        "target function omitted */).xy / (float3(0) /* unsupported HLSL GLSL "
        "buffer block function call readUvq: target function omitted */).z, 0 "
        "/* unsupported HLSL GLSL buffer block function call readDepth: target "
        "function omitted */, float2(0) /* unsupported HLSL GLSL buffer block "
        "access shadowBlock: no target-side fallback declaration emitted */, "
        "float2(0) /* unsupported HLSL GLSL buffer block access shadowBlock: "
        "no target-side fallback declaration emitted */, int2(0) /* unsupported "
        "HLSL GLSL buffer block function call readOffset: target function omitted */);"
        in hlsl
    )
    assert "unsupported DirectX projected texture: textureProjGradOffset" not in hlsl
    assert (
        "unsupported DirectX texture compare: textureCompareProjGradOffset" not in hlsl
    )
    assert "shadowBlock.uvq" not in hlsl
    assert "shadowBlock.offset" not in hlsl

    assert (
        "float4 projectedCall = colorMap.sample(linearSampler, (float3(0) "
        "/* unsupported Metal GLSL buffer block function call readUvq: target "
        "function omitted */).xy / (float3(0) /* unsupported Metal GLSL buffer "
        "block function call readUvq: target function omitted */).z, "
        "gradient2d(float2(0) /* unsupported Metal GLSL buffer block access "
        "shadowBlock: no target-side fallback declaration emitted */, float2(0) "
        "/* unsupported Metal GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */), int2(0) /* unsupported Metal GLSL "
        "buffer block function call readOffset: target function omitted */);" in metal
    )
    assert (
        "float compareCall = shadowMap.sample_compare(compareSampler, float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, 0 /* unsupported Metal GLSL buffer block function "
        "call readDepth: target function omitted */, gradient2d(float2(0) "
        "/* unsupported Metal GLSL buffer block access shadowBlock: no target-side "
        "fallback declaration emitted */, float2(0) /* unsupported Metal GLSL "
        "buffer block access shadowBlock: no target-side fallback declaration "
        "emitted */), int2(0) /* unsupported Metal GLSL buffer block function "
        "call readOffset: target function omitted */);" in metal
    )
    assert (
        "float compareProjCall = shadowMap.sample_compare(compareSampler, "
        "(float3(0) /* unsupported Metal GLSL buffer block function call readUvq: "
        "target function omitted */).xy / (float3(0) /* unsupported Metal GLSL "
        "buffer block function call readUvq: target function omitted */).z, 0 "
        "/* unsupported Metal GLSL buffer block function call readDepth: target "
        "function omitted */, gradient2d(float2(0) /* unsupported Metal GLSL "
        "buffer block access shadowBlock: no target-side fallback declaration "
        "emitted */, float2(0) /* unsupported Metal GLSL buffer block access "
        "shadowBlock: no target-side fallback declaration emitted */), int2(0) "
        "/* unsupported Metal GLSL buffer block function call readOffset: target "
        "function omitted */);" in metal
    )
    assert "unsupported Metal projected texture: textureProjGradOffset" not in metal
    assert (
        "unsupported Metal texture compare: textureCompareProjGradOffset" not in metal
    )
    assert "shadowBlock.uvq" not in metal
    assert "shadowBlock.offset" not in metal

    assert (
        "vec4 projectedCall = textureProjGradOffset(colorMap, "
        "readUvq(shadowBlock), shadowBlock.dx, shadowBlock.dy, "
        "readOffset(shadowBlock));" in glsl
    )
    assert (
        "float compareCall = textureGradOffset(shadowMap, "
        "vec3(readUv(shadowBlock), readDepth(shadowBlock)), "
        "shadowBlock.dx, shadowBlock.dy, readOffset(shadowBlock));" in glsl
    )
    assert (
        "float compareProjCall = textureGradOffset(shadowMap, "
        "vec3((readUvq(shadowBlock)).xy / (readUvq(shadowBlock)).z, "
        "readDepth(shadowBlock)), shadowBlock.dx, shadowBlock.dy, "
        "readOffset(shadowBlock));" in glsl
    )
    assert "unsupported GLSL texture compare: textureCompareProjGradOffset" not in glsl


def test_codegen_mixed_ssbo_unsupported_gather_compare_calls_infer_types():
    crossgl = """
    shader GatherCompareFallbacks {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedGatherCompareBlock {
            double flag;
            vec2 uv;
            vec3 uvLayer;
            float depth;
            ivec2 offset;
        };

        UnsupportedGatherCompareBlock gatherBlock @glsl_buffer_block(std430) @binding(105);

        vec2 readUv(UnsupportedGatherCompareBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uv;
        }

        vec3 readUvLayer(UnsupportedGatherCompareBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uvLayer;
        }

        float readDepth(UnsupportedGatherCompareBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.depth;
        }

        ivec2 readOffset(UnsupportedGatherCompareBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.offset;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 planarDirect = textureGatherCompare(shadowMap, compareSampler, gatherBlock.uv, gatherBlock.depth);
                vec4 planarCall = textureGatherCompare(shadowMap, compareSampler, readUv(gatherBlock), readDepth(gatherBlock));
                vec4 planarOffsetDirect = textureGatherCompareOffset(shadowMap, compareSampler, gatherBlock.uv, gatherBlock.depth, gatherBlock.offset);
                vec4 planarOffsetCall = textureGatherCompareOffset(shadowMap, compareSampler, readUv(gatherBlock), readDepth(gatherBlock), readOffset(gatherBlock));
                vec4 arrayDirect = textureGatherCompare(shadowArray, compareSampler, gatherBlock.uvLayer, gatherBlock.depth);
                vec4 arrayCall = textureGatherCompare(shadowArray, compareSampler, readUvLayer(gatherBlock), readDepth(gatherBlock));
                vec4 arrayOffsetDirect = textureGatherCompareOffset(shadowArray, compareSampler, gatherBlock.uvLayer, gatherBlock.depth, gatherBlock.offset);
                vec4 arrayOffsetCall = textureGatherCompareOffset(shadowArray, compareSampler, readUvLayer(gatherBlock), readDepth(gatherBlock), readOffset(gatherBlock));
                return planarDirect + planarCall + planarOffsetDirect + planarOffsetCall + arrayDirect + arrayCall + arrayOffsetDirect + arrayOffsetCall;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "float4 planarCall = shadowMap.GatherCmp(compareSampler, float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, 0 /* unsupported HLSL GLSL buffer block function "
        "call readDepth: target function omitted */);" in hlsl
    )
    assert (
        "float4 planarOffsetCall = shadowMap.GatherCmp(compareSampler, float2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUv: target "
        "function omitted */, 0 /* unsupported HLSL GLSL buffer block function "
        "call readDepth: target function omitted */, int2(0) /* unsupported "
        "HLSL GLSL buffer block function call readOffset: target function omitted */);"
        in hlsl
    )
    assert (
        "float4 arrayCall = shadowArray.GatherCmp(compareSampler, float3(0) "
        "/* unsupported HLSL GLSL buffer block function call readUvLayer: target "
        "function omitted */, 0 /* unsupported HLSL GLSL buffer block function "
        "call readDepth: target function omitted */);" in hlsl
    )
    assert (
        "float4 arrayOffsetCall = shadowArray.GatherCmp(compareSampler, float3(0) "
        "/* unsupported HLSL GLSL buffer block function call readUvLayer: target "
        "function omitted */, 0 /* unsupported HLSL GLSL buffer block function "
        "call readDepth: target function omitted */, int2(0) /* unsupported "
        "HLSL GLSL buffer block function call readOffset: target function omitted */);"
        in hlsl
    )
    assert "unsupported DirectX texture gather compare" not in hlsl
    assert "GatherCmp(compareSampler, 0 /* unsupported HLSL GLSL buffer" not in hlsl
    assert (
        ", 0 /* unsupported HLSL GLSL buffer block function call readOffset" not in hlsl
    )
    assert "gatherBlock.uv" not in hlsl
    assert "gatherBlock.offset" not in hlsl

    assert (
        "float4 planarCall = shadowMap.gather_compare(compareSampler, float2(0) "
        "/* unsupported Metal GLSL buffer block function call readUv: target "
        "function omitted */, 0 /* unsupported Metal GLSL buffer block function "
        "call readDepth: target function omitted */);" in metal
    )
    assert (
        "float4 planarOffsetCall = shadowMap.gather_compare(compareSampler, "
        "float2(0) /* unsupported Metal GLSL buffer block function call readUv: "
        "target function omitted */, 0 /* unsupported Metal GLSL buffer block "
        "function call readDepth: target function omitted */, int2(0) "
        "/* unsupported Metal GLSL buffer block function call readOffset: target "
        "function omitted */);" in metal
    )
    assert (
        "float4 arrayCall = shadowArray.gather_compare(compareSampler, "
        "(float3(0) /* unsupported Metal GLSL buffer block function call "
        "readUvLayer: target function omitted */).xy, uint((float3(0) "
        "/* unsupported Metal GLSL buffer block function call readUvLayer: "
        "target function omitted */).z), 0 /* unsupported Metal GLSL buffer "
        "block function call readDepth: target function omitted */);" in metal
    )
    assert (
        "float4 arrayOffsetCall = shadowArray.gather_compare(compareSampler, "
        "(float3(0) /* unsupported Metal GLSL buffer block function call "
        "readUvLayer: target function omitted */).xy, uint((float3(0) "
        "/* unsupported Metal GLSL buffer block function call readUvLayer: "
        "target function omitted */).z), 0 /* unsupported Metal GLSL buffer "
        "block function call readDepth: target function omitted */, int2(0) "
        "/* unsupported Metal GLSL buffer block function call readOffset: target "
        "function omitted */);" in metal
    )
    assert "unsupported Metal texture gather compare" not in metal
    assert (
        "gather_compare(compareSampler, 0 /* unsupported Metal GLSL buffer" not in metal
    )
    assert (
        ", 0 /* unsupported Metal GLSL buffer block function call readOffset"
        not in metal
    )
    assert "gatherBlock.uv" not in metal
    assert "gatherBlock.offset" not in metal

    assert (
        "vec4 planarCall = textureGather(shadowMap, "
        "readUv(gatherBlock), readDepth(gatherBlock));" in glsl
    )
    assert (
        "vec4 planarOffsetCall = textureGatherOffset(shadowMap, "
        "readUv(gatherBlock), readDepth(gatherBlock), readOffset(gatherBlock));" in glsl
    )
    assert (
        "vec4 arrayCall = textureGather(shadowArray, "
        "readUvLayer(gatherBlock), readDepth(gatherBlock));" in glsl
    )
    assert (
        "vec4 arrayOffsetCall = textureGatherOffset(shadowArray, "
        "readUvLayer(gatherBlock), readDepth(gatherBlock), readOffset(gatherBlock));"
        in glsl
    )
    assert "textureGatherCompare(" not in glsl
    assert "textureGatherCompareOffset(" not in glsl


def test_codegen_mixed_ssbo_cube_gather_compare_calls_use_fixed_block_offsets():
    crossgl = """
    shader CubeGatherCompareFallbacks {
        samplerCubeShadow shadowCube;
        samplerCubeArrayShadow shadowCubeArray;
        sampler compareSampler;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedCubeGatherBlock {
            vec3 direction;
            vec4 cubeLayer;
            float depth;
            ivec2 offset;
        };

        UnsupportedCubeGatherBlock cubeBlock @glsl_buffer_block(std430) @binding(106);

        vec3 readDirection(UnsupportedCubeGatherBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.direction;
        }

        vec4 readCubeLayer(UnsupportedCubeGatherBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.cubeLayer;
        }

        float readDepth(UnsupportedCubeGatherBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.depth;
        }

        ivec2 readOffset(UnsupportedCubeGatherBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.offset;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 cubeDirect = textureGatherCompare(shadowCube, compareSampler, cubeBlock.direction, cubeBlock.depth);
                vec4 cubeCall = textureGatherCompare(shadowCube, compareSampler, readDirection(cubeBlock), readDepth(cubeBlock));
                vec4 cubeOffsetDirect = textureGatherCompareOffset(shadowCube, compareSampler, cubeBlock.direction, cubeBlock.depth, cubeBlock.offset);
                vec4 cubeOffsetCall = textureGatherCompareOffset(shadowCube, compareSampler, readDirection(cubeBlock), readDepth(cubeBlock), readOffset(cubeBlock));
                vec4 arrayDirect = textureGatherCompare(shadowCubeArray, compareSampler, cubeBlock.cubeLayer, cubeBlock.depth);
                vec4 arrayCall = textureGatherCompare(shadowCubeArray, compareSampler, readCubeLayer(cubeBlock), readDepth(cubeBlock));
                vec4 arrayOffsetDirect = textureGatherCompareOffset(shadowCubeArray, compareSampler, cubeBlock.cubeLayer, cubeBlock.depth, cubeBlock.offset);
                vec4 arrayOffsetCall = textureGatherCompareOffset(shadowCubeArray, compareSampler, readCubeLayer(cubeBlock), readDepth(cubeBlock), readOffset(cubeBlock));
                return cubeDirect + cubeCall + cubeOffsetDirect + cubeOffsetCall + arrayDirect + arrayCall + arrayOffsetDirect + arrayOffsetCall;
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer cubeBlock : register(u106);" in hlsl
    assert "float3 readDirection(RWByteAddressBuffer localBlock)" in hlsl
    assert "float readDepth(RWByteAddressBuffer localBlock)" in hlsl
    assert (
        "float4 cubeDirect = shadowCube.GatherCmp(compareSampler, "
        "asfloat(cubeBlock.Load3(0)), asfloat(cubeBlock.Load(32)));" in hlsl
    )
    assert (
        "float4 cubeCall = shadowCube.GatherCmp(compareSampler, "
        "readDirection(cubeBlock), readDepth(cubeBlock));" in hlsl
    )
    assert (
        "float4 arrayDirect = shadowCubeArray.GatherCmp(compareSampler, "
        "asfloat(cubeBlock.Load4(16)), asfloat(cubeBlock.Load(32)));" in hlsl
    )
    assert (
        "float4 arrayCall = shadowCubeArray.GatherCmp(compareSampler, "
        "readCubeLayer(cubeBlock), readDepth(cubeBlock));" in hlsl
    )
    assert (
        hlsl.count(
            "/* unsupported DirectX texture gather compare: "
            "textureGatherCompareOffset offsets require 2D or 2D-array textures */ "
            "float4(0.0)"
        )
        == 4
    )
    assert "unsupported HLSL GLSL buffer block" not in hlsl
    assert "cubeBlock.direction" not in hlsl
    assert "cubeBlock.cubeLayer" not in hlsl
    assert "cubeBlock.offset" not in hlsl

    assert "device uchar* cubeBlock [[buffer(106)]]" in metal
    assert "float3 readDirection(device uchar* localBlock)" in metal
    assert "float readDepth(device uchar* localBlock)" in metal
    assert (
        "float4 cubeCall = shadowCube.gather_compare(compareSampler, "
        "readDirection(cubeBlock), readDepth(cubeBlock));" in metal
    )
    assert (
        "float4 arrayCall = shadowCubeArray.gather_compare(compareSampler, "
        "(readCubeLayer(cubeBlock)).xyz, uint((readCubeLayer(cubeBlock)).w), "
        "readDepth(cubeBlock));" in metal
    )
    assert (
        metal.count(
            "/* unsupported Metal texture gather compare: "
            "textureGatherCompareOffset offsets require 2D or 2D-array depth "
            "textures */ float4(0.0)"
        )
        == 4
    )
    assert "unsupported Metal GLSL buffer block" not in metal
    assert "cubeBlock.direction" not in metal
    assert "cubeBlock.cubeLayer" not in metal
    assert "cubeBlock.offset" not in metal

    assert (
        "vec4 cubeCall = textureGather(shadowCube, "
        "readDirection(cubeBlock), readDepth(cubeBlock));" in glsl
    )
    assert (
        "vec4 arrayCall = textureGather(shadowCubeArray, "
        "readCubeLayer(cubeBlock), readDepth(cubeBlock));" in glsl
    )
    assert (
        glsl.count(
            "/* unsupported GLSL texture gather compare: "
            "textureGatherCompareOffset offsets require 2D or 2D-array shadow "
            "samplers */ vec4(0.0)"
        )
        == 4
    )
    assert "textureGatherCompare(" not in glsl
    assert "textureGatherCompareOffset(" not in glsl


def test_codegen_mixed_ssbo_unsupported_query_and_atomic_args_infer_types():
    crossgl = """
    shader QueryAtomicFallbacks {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;
        uimage2D counters @r32ui;
        uimage2DArray layerCounters @r32ui;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedQueryAtomicBlock {
            double flag;
            vec2 uv;
            vec3 uvLayer;
            vec4 cubeLayer;
            int lod;
            ivec2 pixel;
            ivec3 pixelLayer;
            uint amount;
        };

        UnsupportedQueryAtomicBlock queryBlock @glsl_buffer_block(std430) @binding(107);

        vec3 readUvLayer(UnsupportedQueryAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.uvLayer;
        }

        vec4 readCubeLayer(UnsupportedQueryAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.cubeLayer;
        }

        int readLod(UnsupportedQueryAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.lod;
        }

        ivec2 readPixel(UnsupportedQueryAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedQueryAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        uint readAmount(UnsupportedQueryAtomicBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.amount;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                ivec2 sizeDirect = textureSize(colorMap, queryBlock.lod);
                ivec3 layerSizeCall = textureSize(layerMap, readLod(queryBlock));
                vec2 lodDirect = textureQueryLod(layerMap, linearSampler, queryBlock.uvLayer);
                vec2 lodCall = textureQueryLod(cubeArray, linearSampler, readCubeLayer(queryBlock));
                uint atomicDirect = imageAtomicAdd(counters, queryBlock.pixel, queryBlock.amount);
                uint atomicCall = imageAtomicAdd(counters, readPixel(queryBlock), readAmount(queryBlock));
                uint swapDirect = imageAtomicCompSwap(layerCounters, queryBlock.pixelLayer, queryBlock.amount, queryBlock.amount);
                uint swapCall = imageAtomicCompSwap(layerCounters, readPixelLayer(queryBlock), queryBlock.amount, readAmount(queryBlock));
                return vec4(float(sizeDirect.x + layerSizeCall.z) + lodDirect.x + lodCall.y + float(atomicDirect + atomicCall + swapDirect + swapCall));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "int2 sizeDirect = textureSize(colorMap, 0 /* unsupported HLSL GLSL "
        "buffer block access queryBlock: no target-side fallback declaration "
        "emitted */);" in hlsl
    )
    assert (
        "int3 layerSizeCall = textureSize(layerMap, 0 /* unsupported HLSL "
        "GLSL buffer block function call readLod: target function omitted */);" in hlsl
    )
    assert (
        "float2 lodDirect = float2(layerMap.CalculateLevelOfDetailUnclamped"
        "(linearSampler, (float3(0) /* unsupported HLSL GLSL buffer block "
        "access queryBlock: no target-side fallback declaration emitted */).xy), "
        "layerMap.CalculateLevelOfDetail(linearSampler, (float3(0) "
        "/* unsupported HLSL GLSL buffer block access queryBlock: no target-side "
        "fallback declaration emitted */).xy));" in hlsl
    )
    assert (
        "float2 lodCall = float2(cubeArray.CalculateLevelOfDetailUnclamped"
        "(linearSampler, (float4(0) /* unsupported HLSL GLSL buffer block "
        "function call readCubeLayer: target function omitted */).xyz), "
        "cubeArray.CalculateLevelOfDetail(linearSampler, (float4(0) "
        "/* unsupported HLSL GLSL buffer block function call readCubeLayer: "
        "target function omitted */).xyz));" in hlsl
    )
    assert (
        "uint atomicDirect = imageAtomicAdd_uimage2D(counters, int2(0) "
        "/* unsupported HLSL GLSL buffer block access queryBlock: no target-side "
        "fallback declaration emitted */, 0u /* unsupported HLSL GLSL buffer "
        "block access queryBlock: no target-side fallback declaration emitted */);"
        in hlsl
    )
    assert (
        "uint atomicCall = imageAtomicAdd_uimage2D(counters, int2(0) "
        "/* unsupported HLSL GLSL buffer block function call readPixel: target "
        "function omitted */, 0u /* unsupported HLSL GLSL buffer block function "
        "call readAmount: target function omitted */);" in hlsl
    )
    assert (
        "uint swapDirect = imageAtomicCompSwap_uimage2DArray(layerCounters, "
        "int3(0) /* unsupported HLSL GLSL buffer block access queryBlock: "
        "no target-side fallback declaration emitted */, 0u /* unsupported "
        "HLSL GLSL buffer block access queryBlock: no target-side fallback "
        "declaration emitted */, 0u /* unsupported HLSL GLSL buffer block "
        "access queryBlock: no target-side fallback declaration emitted */);" in hlsl
    )
    assert (
        "uint swapCall = imageAtomicCompSwap_uimage2DArray(layerCounters, "
        "int3(0) /* unsupported HLSL GLSL buffer block function call "
        "readPixelLayer: target function omitted */, 0u /* unsupported HLSL "
        "GLSL buffer block access queryBlock: no target-side fallback "
        "declaration emitted */, 0u /* unsupported HLSL GLSL buffer block "
        "function call readAmount: target function omitted */);" in hlsl
    )
    assert "imageAtomicAdd(counters" not in hlsl
    assert "imageAtomicCompSwap(layerCounters" not in hlsl
    assert "textureQueryLod(" not in hlsl
    assert "queryBlock.uvLayer" not in hlsl
    assert "queryBlock.pixel" not in hlsl
    assert "queryBlock.amount" not in hlsl

    assert (
        "int2 sizeDirect = int2(colorMap.get_width(uint(0 /* unsupported Metal "
        "GLSL buffer block access queryBlock: no target-side fallback declaration "
        "emitted */)), colorMap.get_height(uint(0 /* unsupported Metal GLSL "
        "buffer block access queryBlock: no target-side fallback declaration "
        "emitted */)));" in metal
    )
    assert (
        "int3 layerSizeCall = int3(layerMap.get_width(uint(0 /* unsupported "
        "Metal GLSL buffer block function call readLod: target function omitted */)), "
        "layerMap.get_height(uint(0 /* unsupported Metal GLSL buffer block "
        "function call readLod: target function omitted */)), layerMap.get_array_size());"
        in metal
    )
    assert (
        "float2 lodDirect = float2(layerMap.calculate_unclamped_lod"
        "(linearSampler, (float3(0) /* unsupported Metal GLSL buffer block "
        "access queryBlock: no target-side fallback declaration emitted */).xy), "
        "layerMap.calculate_clamped_lod(linearSampler, (float3(0) "
        "/* unsupported Metal GLSL buffer block access queryBlock: no target-side "
        "fallback declaration emitted */).xy));" in metal
    )
    assert (
        "float2 lodCall = float2(cubeArray.calculate_unclamped_lod"
        "(linearSampler, (float4(0) /* unsupported Metal GLSL buffer block "
        "function call readCubeLayer: target function omitted */).xyz), "
        "cubeArray.calculate_clamped_lod(linearSampler, (float4(0) "
        "/* unsupported Metal GLSL buffer block function call readCubeLayer: "
        "target function omitted */).xyz));" in metal
    )
    assert (
        "uint atomicDirect = counters.atomic_fetch_add(uint2(int2(0) "
        "/* unsupported Metal GLSL buffer block access queryBlock: no target-side "
        "fallback declaration emitted */), 0u /* unsupported Metal GLSL buffer "
        "block access queryBlock: no target-side fallback declaration emitted */).x;"
        in metal
    )
    assert (
        "uint atomicCall = counters.atomic_fetch_add(uint2(int2(0) "
        "/* unsupported Metal GLSL buffer block function call readPixel: target "
        "function omitted */), 0u /* unsupported Metal GLSL buffer block function "
        "call readAmount: target function omitted */).x;" in metal
    )
    assert (
        "uint swapDirect = imageAtomicCompSwap_uimage2DArray(layerCounters, "
        "int3(0) /* unsupported Metal GLSL buffer block access queryBlock: "
        "no target-side fallback declaration emitted */, 0u /* unsupported "
        "Metal GLSL buffer block access queryBlock: no target-side fallback "
        "declaration emitted */, 0u /* unsupported Metal GLSL buffer block "
        "access queryBlock: no target-side fallback declaration emitted */);" in metal
    )
    assert (
        "uint swapCall = imageAtomicCompSwap_uimage2DArray(layerCounters, "
        "int3(0) /* unsupported Metal GLSL buffer block function call "
        "readPixelLayer: target function omitted */, 0u /* unsupported Metal "
        "GLSL buffer block access queryBlock: no target-side fallback "
        "declaration emitted */, 0u /* unsupported Metal GLSL buffer block "
        "function call readAmount: target function omitted */);" in metal
    )
    assert "imageAtomicAdd(counters" not in metal
    assert "imageAtomicCompSwap(layerCounters" not in metal
    assert "textureQueryLod(" not in metal
    assert "queryBlock.uvLayer" not in metal
    assert "queryBlock.pixel" not in metal
    assert "queryBlock.amount" not in metal

    assert "ivec2 sizeDirect = textureSize(colorMap, queryBlock.lod);" in glsl
    assert "ivec3 layerSizeCall = textureSize(layerMap, readLod(queryBlock));" in glsl
    assert "vec2 lodDirect = textureQueryLod(layerMap, queryBlock.uvLayer.xy);" in glsl
    assert (
        "vec2 lodCall = textureQueryLod(cubeArray, "
        "(readCubeLayer(queryBlock)).xyz);" in glsl
    )
    assert (
        "uint atomicDirect = imageAtomicAdd(counters, "
        "queryBlock.pixel, queryBlock.amount);" in glsl
    )
    assert (
        "uint atomicCall = imageAtomicAdd(counters, "
        "readPixel(queryBlock), readAmount(queryBlock));" in glsl
    )
    assert (
        "uint swapDirect = imageAtomicCompSwap(layerCounters, "
        "queryBlock.pixelLayer, queryBlock.amount, queryBlock.amount);" in glsl
    )
    assert (
        "uint swapCall = imageAtomicCompSwap(layerCounters, "
        "readPixelLayer(queryBlock), queryBlock.amount, readAmount(queryBlock));"
        in glsl
    )
    assert "textureQueryLod(layerMap, queryBlock.uvLayer);" not in glsl
    assert "textureQueryLod(cubeArray, readCubeLayer(queryBlock));" not in glsl


def test_codegen_mixed_ssbo_unsupported_image_format_args_infer_types():
    crossgl = """
    shader ImageFormatFallbacks {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        uimage2D unsignedScalar @r32ui;
        uimage2DArray unsignedLayers @rg32ui;

        struct VSOutput {
            vec2 uv;
        };

        struct UnsupportedImageFormatBlock {
            double flag;
            ivec2 pixel;
            ivec3 pixelLayer;
            float scalarValue;
            vec2 rgValue;
            uint unsignedValue;
            uvec2 unsignedRgValue;
        };

        UnsupportedImageFormatBlock imageBlock @glsl_buffer_block(std430) @binding(108);

        ivec2 readPixel(UnsupportedImageFormatBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixel;
        }

        ivec3 readPixelLayer(UnsupportedImageFormatBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.pixelLayer;
        }

        float readScalar(UnsupportedImageFormatBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.scalarValue;
        }

        vec2 readRg(UnsupportedImageFormatBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.rgValue;
        }

        uint readUnsigned(UnsupportedImageFormatBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.unsignedValue;
        }

        uvec2 readUnsignedRg(UnsupportedImageFormatBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.unsignedRgValue;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float scalarDirect = imageLoad(scalarFloat, imageBlock.pixel);
                float scalarCall = imageLoad(scalarFloat, readPixel(imageBlock));
                vec2 rgDirect = imageLoad(rgFloat, imageBlock.pixel);
                vec2 rgCall = imageLoad(rgFloat, readPixel(imageBlock));
                uint unsignedDirect = imageLoad(unsignedScalar, imageBlock.pixel);
                uvec2 unsignedRgCall = imageLoad(unsignedLayers, readPixelLayer(imageBlock));
                imageStore(scalarFloat, imageBlock.pixel, imageBlock.scalarValue);
                imageStore(scalarFloat, readPixel(imageBlock), readScalar(imageBlock));
                imageStore(rgFloat, imageBlock.pixel, imageBlock.scalarValue);
                imageStore(rgFloat, readPixel(imageBlock), readRg(imageBlock));
                imageStore(unsignedScalar, imageBlock.pixel, imageBlock.unsignedValue);
                imageStore(unsignedScalar, readPixel(imageBlock), readUnsigned(imageBlock));
                imageStore(unsignedLayers, imageBlock.pixelLayer, imageBlock.unsignedValue);
                imageStore(unsignedLayers, readPixelLayer(imageBlock), readUnsignedRg(imageBlock));
                return vec4(scalarDirect + scalarCall + rgDirect.x + rgCall.y + float(unsignedDirect + unsignedRgCall.x));
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWTexture2D<float> scalarFloat : register(u0);" in hlsl
    assert "RWTexture2D<float2> rgFloat : register(u1);" in hlsl
    assert "RWTexture2D<uint> unsignedScalar : register(u2);" in hlsl
    assert "RWTexture2DArray<uint2> unsignedLayers : register(u3);" in hlsl
    assert (
        "float scalarDirect = scalarFloat[int2(0) /* unsupported HLSL GLSL "
        "buffer block access imageBlock: no target-side fallback declaration "
        "emitted */];" in hlsl
    )
    assert (
        "float scalarCall = scalarFloat[int2(0) /* unsupported HLSL GLSL "
        "buffer block function call readPixel: target function omitted */];" in hlsl
    )
    assert (
        "float2 rgDirect = rgFloat[int2(0) /* unsupported HLSL GLSL buffer "
        "block access imageBlock: no target-side fallback declaration emitted */];"
        in hlsl
    )
    assert (
        "uint2 unsignedRgCall = unsignedLayers[int3(0) /* unsupported HLSL "
        "GLSL buffer block function call readPixelLayer: target function "
        "omitted */];" in hlsl
    )
    assert (
        "rgFloat[int2(0) /* unsupported HLSL GLSL buffer block access imageBlock: "
        "no target-side fallback declaration emitted */] = float2(0 "
        "/* unsupported HLSL GLSL buffer block access imageBlock: no target-side "
        "fallback declaration emitted */, 0.0);" in hlsl
    )
    assert (
        "rgFloat[int2(0) /* unsupported HLSL GLSL buffer block function call "
        "readPixel: target function omitted */] = float2(0) /* unsupported "
        "HLSL GLSL buffer block function call readRg: target function omitted */;"
        in hlsl
    )
    assert (
        "unsignedLayers[int3(0) /* unsupported HLSL GLSL buffer block access "
        "imageBlock: no target-side fallback declaration emitted */] = "
        "uint2(0u /* unsupported HLSL GLSL buffer block access imageBlock: "
        "no target-side fallback declaration emitted */, 0u);" in hlsl
    )
    assert (
        "unsignedLayers[int3(0) /* unsupported HLSL GLSL buffer block function "
        "call readPixelLayer: target function omitted */] = uint2(0) "
        "/* unsupported HLSL GLSL buffer block function call readUnsignedRg: "
        "target function omitted */;" in hlsl
    )
    assert "imageLoad(" not in hlsl
    assert "imageStore(" not in hlsl
    assert "imageBlock.pixel" not in hlsl
    assert "imageBlock.scalarValue" not in hlsl
    assert "imageBlock.unsignedRgValue" not in hlsl

    assert "texture2d<float, access::read_write> scalarFloat [[texture(0)]]" in metal
    assert "texture2d<float, access::read_write> rgFloat [[texture(1)]]" in metal
    assert "texture2d<uint, access::read_write> unsignedScalar [[texture(2)]]" in metal
    assert (
        "texture2d_array<uint, access::read_write> unsignedLayers [[texture(3)]]"
        in metal
    )
    assert (
        "float scalarDirect = scalarFloat.read(uint2(int2(0) /* unsupported "
        "Metal GLSL buffer block access imageBlock: no target-side fallback "
        "declaration emitted */)).x;" in metal
    )
    assert (
        "float scalarCall = scalarFloat.read(uint2(int2(0) /* unsupported "
        "Metal GLSL buffer block function call readPixel: target function "
        "omitted */)).x;" in metal
    )
    assert (
        "float2 rgDirect = rgFloat.read(uint2(int2(0) /* unsupported Metal "
        "GLSL buffer block access imageBlock: no target-side fallback declaration "
        "emitted */)).xy;" in metal
    )
    assert (
        "uint2 unsignedRgCall = unsignedLayers.read(uint2((int3(0) "
        "/* unsupported Metal GLSL buffer block function call readPixelLayer: "
        "target function omitted */).xy), uint((int3(0) /* unsupported Metal "
        "GLSL buffer block function call readPixelLayer: target function "
        "omitted */).z)).xy;" in metal
    )
    assert (
        "rgFloat.write(float4(0 /* unsupported Metal GLSL buffer block access "
        "imageBlock: no target-side fallback declaration emitted */, 0.0, 0.0, "
        "0.0), uint2(int2(0) /* unsupported Metal GLSL buffer block access "
        "imageBlock: no target-side fallback declaration emitted */));" in metal
    )
    assert (
        "rgFloat.write(float4(float2(0) /* unsupported Metal GLSL buffer block "
        "function call readRg: target function omitted */, 0.0, 0.0), "
        "uint2(int2(0) /* unsupported Metal GLSL buffer block function call "
        "readPixel: target function omitted */));" in metal
    )
    assert (
        "unsignedLayers.write(uint4(0u /* unsupported Metal GLSL buffer block "
        "access imageBlock: no target-side fallback declaration emitted */, 0u, "
        "0u, 0u), uint2((int3(0) /* unsupported Metal GLSL buffer block access "
        "imageBlock: no target-side fallback declaration emitted */).xy), "
        "uint((int3(0) /* unsupported Metal GLSL buffer block access imageBlock: "
        "no target-side fallback declaration emitted */).z));" in metal
    )
    assert (
        "unsignedLayers.write(uint4(uint2(0) /* unsupported Metal GLSL buffer "
        "block function call readUnsignedRg: target function omitted */, 0u, 0u), "
        "uint2((int3(0) /* unsupported Metal GLSL buffer block function call "
        "readPixelLayer: target function omitted */).xy), uint((int3(0) "
        "/* unsupported Metal GLSL buffer block function call readPixelLayer: "
        "target function omitted */).z));" in metal
    )
    assert "imageLoad(" not in metal
    assert "imageStore(" not in metal
    assert "imageBlock.pixel" not in metal
    assert "imageBlock.scalarValue" not in metal
    assert "imageBlock.unsignedRgValue" not in metal

    assert "layout(r32f, binding = 0) uniform image2D scalarFloat;" in glsl
    assert "layout(rg32f, binding = 1) uniform image2D rgFloat;" in glsl
    assert "layout(r32ui, binding = 2) uniform uimage2D unsignedScalar;" in glsl
    assert "layout(rg32ui, binding = 3) uniform uimage2DArray unsignedLayers;" in glsl
    assert "float scalarDirect = imageLoad(scalarFloat, imageBlock.pixel).x;" in glsl
    assert "vec2 rgDirect = imageLoad(rgFloat, imageBlock.pixel).xy;" in glsl
    assert (
        "uvec2 unsignedRgCall = imageLoad(unsignedLayers, "
        "readPixelLayer(imageBlock)).xy;" in glsl
    )
    assert (
        "imageStore(scalarFloat, imageBlock.pixel, vec4(imageBlock.scalarValue));"
        in glsl
    )
    assert (
        "imageStore(rgFloat, imageBlock.pixel, vec4(imageBlock.scalarValue, "
        "0.0, 0.0, 0.0));" in glsl
    )
    assert (
        "imageStore(rgFloat, readPixel(imageBlock), "
        "vec4(readRg(imageBlock), 0.0, 0.0));" in glsl
    )
    assert (
        "imageStore(unsignedLayers, imageBlock.pixelLayer, "
        "uvec4(imageBlock.unsignedValue, 0u, 0u, 0u));" in glsl
    )
    assert (
        "imageStore(unsignedLayers, readPixelLayer(imageBlock), "
        "uvec4(readUnsignedRg(imageBlock), 0u, 0u));" in glsl
    )


def test_codegen_mixed_ssbo_unsized_block_arrays_infer_named_constant_size():
    code = """
    #version 450 core
    const int BASE = 1;
    const int LAYER = BASE + 2;
    layout(std430, binding = 76) buffer ConstMixedBlock {
        uint count;
        vec4 values[];
    } constMixed[];

    void main() {
        uint i = constMixed[LAYER].count;
        vec4 value = constMixed[LAYER].values[i];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "static const int BASE = 1;" in hlsl
    assert "static const int LAYER = (BASE + 2);" in hlsl
    assert "RWByteAddressBuffer constMixed[4] : register(u76);" in hlsl
    assert "uint i = constMixed[LAYER].Load(0);" in hlsl
    assert "constMixed[LAYER].Load4((16 + i * 16))" in hlsl

    assert "constant int BASE = 1;" in metal
    assert "constant int LAYER = BASE + 2;" in metal
    assert "array<device uchar*, 4> constMixed [[buffer(76)]]" in metal
    assert "constMixed[LAYER] + (16 + i * 16)" in metal

    assert "const int LAYER = (BASE + 2);" in glsl
    assert "layout(std430, binding = 76) buffer ConstMixedBlock" in glsl
    assert "} constMixed[];" in glsl
    assert "vec4 value = constMixed[LAYER].values[i];" in glsl


def test_codegen_mixed_ssbo_fixed_block_array_literal_overflow_is_rejected():
    code = """
    #version 450 core
    layout(std430, binding = 77) buffer FixedMixedBlock {
        uint count;
        float values[];
    } fixedMixed[2];

    void main() {
        float value = fixedMixed[2].values[0];
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'fixedMixed': 2 and 3",
    ):
        HLSLCodeGen().generate(shader_ast)

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'fixedMixed': 2 and 3",
    ):
        MetalCodeGen().generate(shader_ast)

    glsl = GLSLCodeGen().generate(shader_ast)
    assert "layout(std430, binding = 77) buffer FixedMixedBlock" in glsl
    assert "} fixedMixed[2];" in glsl
    assert "float value = fixedMixed[2].values[0];" in glsl


def test_codegen_mixed_ssbo_fixed_block_array_call_propagated_conflict_is_rejected():
    crossgl = """
    #version 450 core
    shader main {
        struct ParamMixedBlock {
            uint count;
            float values[];
        };

        ParamMixedBlock fixedBlocks[2] @glsl_buffer_block(std430) @binding(78);

        float readFrom(ParamMixedBlock blocks[]) {
            return blocks[2].values[0];
        }

        compute {
            void main() {
                float value = readFrom(fixedBlocks);
            }
        }
    }
    """

    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'fixedBlocks': 2 and 3",
    ):
        HLSLCodeGen().generate(shader_ast)

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'fixedBlocks': 2 and 3",
    ):
        MetalCodeGen().generate(shader_ast)


def test_codegen_mixed_ssbo_fixed_only_scalar_vector_matrix_blocks_lower_to_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 116) buffer FixedPlainBlock {
        uint count;
        vec3 axis;
        mat2 transform;
    } fixedPlain;

    void main() {
        uint count = fixedPlain.count;
        vec3 axis = fixedPlain.axis;
        mat2 transform = fixedPlain.transform;
        fixedPlain.count = count + 1u;
        fixedPlain.axis = axis;
        fixedPlain.transform = transform;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    assert "unsupported GLSL SSBO block FixedPlainBlock" not in crossgl
    assert "struct FixedPlainBlock" in crossgl
    assert (
        "FixedPlainBlock fixedPlain @glsl_buffer_block(std430) @binding(116);"
        in crossgl
    )
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer fixedPlain : register(u116);" in hlsl
    assert "struct FixedPlainBlock" not in hlsl
    assert "uint count = fixedPlain.Load(0);" in hlsl
    assert "float3 axis = asfloat(fixedPlain.Load3(16));" in hlsl
    assert (
        "float2x2 transform = float2x2(asfloat(fixedPlain.Load2(32)), "
        "asfloat(fixedPlain.Load2(40)));" in hlsl
    )
    assert "fixedPlain.Store(0, (count + 1u));" in hlsl
    assert "fixedPlain.Store3(16, asuint(axis));" in hlsl
    assert "fixedPlain.Store2(32, asuint(transform[0]));" in hlsl
    assert "fixedPlain.Store2(40, asuint(transform[1]));" in hlsl
    assert "unsupported HLSL GLSL buffer block" not in hlsl

    assert "device uchar* fixedPlain [[buffer(116)]]" in metal
    assert "struct FixedPlainBlock" not in metal
    assert (
        "uint count = (*reinterpret_cast<const device uint*>(fixedPlain + 0));" in metal
    )
    assert (
        "float3 axis = float3((*reinterpret_cast<const device float*>"
        "(fixedPlain + 16)), (*reinterpret_cast<const device float*>"
        "(fixedPlain + 20)), (*reinterpret_cast<const device float*>"
        "(fixedPlain + 24)));" in metal
    )
    assert (
        "float2x2 transform = float2x2(float2((*reinterpret_cast<const device "
        "float*>(fixedPlain + 32)), (*reinterpret_cast<const device float*>"
        "(fixedPlain + 36))), float2((*reinterpret_cast<const device float*>"
        "(fixedPlain + 40)), (*reinterpret_cast<const device float*>"
        "(fixedPlain + 44))));" in metal
    )
    assert "(*reinterpret_cast<device uint*>(fixedPlain + 0)) = count + 1u;" in metal
    assert "(*reinterpret_cast<device float*>(fixedPlain + 16))" in metal
    assert "(*reinterpret_cast<device float*>(fixedPlain + 32))" in metal
    assert "unsupported Metal GLSL buffer block" not in metal

    assert "layout(std430, binding = 116) buffer FixedPlainBlock" in glsl
    assert "} fixedPlain;" in glsl
    assert "fixedPlain.transform = transform;" in glsl


def test_codegen_mixed_ssbo_std140_blocks_lower_to_explicit_offsets():
    code = """
    #version 450 core
    layout(std140, binding = 118) buffer Std140Block {
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

    crossgl = generate_crossgl(code, "compute")
    assert "unsupported GLSL SSBO block Std140Block" not in crossgl
    assert (
        "Std140Block std140Block @glsl_buffer_block(std140) @binding(118);" in crossgl
    )
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer std140Block : register(u118);" in hlsl
    assert "uint i = std140Block.Load(0);" in hlsl
    assert (
        "float2x2 basis = float2x2(asfloat(std140Block.Load2(16)), "
        "asfloat(std140Block.Load2(32)));" in hlsl
    )
    assert "float weight = asfloat(std140Block.Load(80));" in hlsl
    assert "float value = asfloat(std140Block.Load((96 + i * 16)));" in hlsl
    assert "std140Block.Store2(16, asuint(basis[0]));" in hlsl
    assert "std140Block.Store2(32, asuint(basis[1]));" in hlsl
    assert "std140Block.Store(64, asuint(weight));" in hlsl
    assert "std140Block.Store((96 + i * 16), asuint(value));" in hlsl
    assert "unsupported HLSL GLSL buffer block" not in hlsl

    assert "device uchar* std140Block [[buffer(118)]]" in metal
    assert "uint i = (*reinterpret_cast<const device uint*>(std140Block + 0));" in metal
    assert (
        "float2x2 basis = float2x2(float2((*reinterpret_cast<const device "
        "float*>(std140Block + 16)), (*reinterpret_cast<const device float*>"
        "(std140Block + 20))), float2((*reinterpret_cast<const device float*>"
        "(std140Block + 32)), (*reinterpret_cast<const device float*>"
        "(std140Block + 36))));" in metal
    )
    assert (
        "float weight = (*reinterpret_cast<const device float*>"
        "(std140Block + 80));" in metal
    )
    assert (
        "float value = (*reinterpret_cast<const device float*>"
        "(std140Block + (96 + i * 16)));" in metal
    )
    assert "(*reinterpret_cast<device float*>(std140Block + 16))" in metal
    assert "(*reinterpret_cast<device float*>(std140Block + 32))" in metal
    assert "(*reinterpret_cast<device float*>(std140Block + 64)) = weight;" in metal
    assert (
        "(*reinterpret_cast<device float*>(std140Block + (96 + i * 16))) = value;"
        in metal
    )
    assert "unsupported Metal GLSL buffer block" not in metal

    assert "layout(std140, binding = 118) buffer Std140Block" in glsl
    assert "float weights[3];" in glsl
    assert "float values[];" in glsl
    assert "std140Block.weights[1] = weight;" in glsl


def test_codegen_mixed_ssbo_uint_atomics_lower_to_byteaddress_and_device_atomics():
    code = """
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

    crossgl = generate_crossgl(code, "compute")
    assert "AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);" in crossgl
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer atomicBlock : register(u17);" in hlsl
    assert (
        "uint __crossgl_byteaddress_atomic_add_uint(RWByteAddressBuffer buffer, "
        "uint offset, uint value)" in hlsl
    )
    assert "buffer.InterlockedAdd(offset, value, original);" in hlsl
    assert (
        "uint __crossgl_byteaddress_atomic_exchange_uint("
        "RWByteAddressBuffer buffer, uint offset, uint value)" in hlsl
    )
    assert "buffer.InterlockedExchange(offset, value, original);" in hlsl
    assert (
        "uint oldCounter = __crossgl_byteaddress_atomic_add_uint("
        "atomicBlock, 0, 1u);" in hlsl
    )
    assert (
        "uint oldBin = __crossgl_byteaddress_atomic_exchange_uint("
        "atomicBlock, 12, oldCounter);" in hlsl
    )
    assert (
        "uint minBin = __crossgl_byteaddress_atomic_min_uint("
        "atomicBlock, 4, 2u);" in hlsl
    )
    assert (
        "uint maxBin = __crossgl_byteaddress_atomic_max_uint("
        "atomicBlock, 4, minBin);" in hlsl
    )
    assert (
        "uint andBin = __crossgl_byteaddress_atomic_and_uint("
        "atomicBlock, 8, 15u);" in hlsl
    )
    assert (
        "uint orBin = __crossgl_byteaddress_atomic_or_uint("
        "atomicBlock, 8, andBin);" in hlsl
    )
    assert (
        "uint xorBin = __crossgl_byteaddress_atomic_xor_uint("
        "atomicBlock, 12, orBin);" in hlsl
    )
    assert (
        "uint casBin = __crossgl_byteaddress_atomic_compare_exchange_uint("
        "atomicBlock, 16, xorBin, 7u);" in hlsl
    )
    assert "__crossgl_byteaddress_atomic_add_uint(atomicBlock, 8, casBin);" in hlsl
    assert "atomicAdd(atomicBlock.Load" not in hlsl
    assert "atomicExchange(atomicBlock.Load" not in hlsl

    assert "device uchar* atomicBlock [[buffer(17)]]" in metal
    assert (
        "uint oldCounter = atomic_fetch_add_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 0), "
        "1u, memory_order_relaxed);" in metal
    )
    assert (
        "uint oldBin = atomic_exchange_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 12), "
        "oldCounter, memory_order_relaxed);" in metal
    )
    assert (
        "uint minBin = atomic_fetch_min_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 4), "
        "2u, memory_order_relaxed);" in metal
    )
    assert (
        "uint maxBin = atomic_fetch_max_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 4), "
        "minBin, memory_order_relaxed);" in metal
    )
    assert (
        "uint andBin = atomic_fetch_and_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 8), "
        "15u, memory_order_relaxed);" in metal
    )
    assert (
        "uint orBin = atomic_fetch_or_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 8), "
        "andBin, memory_order_relaxed);" in metal
    )
    assert (
        "uint xorBin = atomic_fetch_xor_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 12), "
        "orBin, memory_order_relaxed);" in metal
    )
    assert (
        "uint casBin = __crossgl_buffer_atomic_compare_exchange_uint("
        "atomicBlock, 16, xorBin, 7u);" in metal
    )
    assert (
        "while (!atomic_compare_exchange_weak_explicit("
        "target, &original, value, memory_order_relaxed, memory_order_relaxed) "
        "&& original == compareValue);" in metal
    )
    assert (
        "atomic_fetch_add_explicit("
        "reinterpret_cast<device atomic_uint*>(atomicBlock + 8), "
        "casBin, memory_order_relaxed);" in metal
    )
    assert "atomicAdd((*reinterpret_cast" not in metal
    assert "atomicExchange((*reinterpret_cast" not in metal

    assert "layout(std430, binding = 17) buffer AtomicBlock" in glsl
    assert "uint oldCounter = atomicAdd(atomicBlock.counter, 1u);" in glsl
    assert "uint oldBin = atomicExchange(atomicBlock.bins[2], oldCounter);" in glsl
    assert "uint minBin = atomicMin(atomicBlock.bins[0], 2u);" in glsl
    assert "uint maxBin = atomicMax(atomicBlock.bins[0], minBin);" in glsl
    assert "uint andBin = atomicAnd(atomicBlock.bins[1], 15u);" in glsl
    assert "uint orBin = atomicOr(atomicBlock.bins[1], andBin);" in glsl
    assert "uint xorBin = atomicXor(atomicBlock.bins[2], orBin);" in glsl
    assert "uint casBin = atomicCompSwap(atomicBlock.bins[3], xorBin, 7u);" in glsl
    assert "atomicAdd(atomicBlock.bins[1], casBin);" in glsl


def test_codegen_mixed_ssbo_int_atomics_lower_to_byteaddress_and_device_atomics():
    code = """
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

    crossgl = generate_crossgl(code, "compute")
    assert (
        "SignedAtomicBlock signedAtomicBlock "
        "@glsl_buffer_block(std430) @binding(18);" in crossgl
    )
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer signedAtomicBlock : register(u18);" in hlsl
    assert (
        "int __crossgl_byteaddress_atomic_add_int(RWByteAddressBuffer buffer, "
        "uint offset, int value)" in hlsl
    )
    assert "buffer.InterlockedAdd(offset, asuint(value), original);" in hlsl
    assert "return asint(original);" in hlsl
    assert (
        "int __crossgl_byteaddress_atomic_min_int(RWByteAddressBuffer buffer, "
        "uint offset, int value)" in hlsl
    )
    assert "buffer.InterlockedMin(offset, value, original);" in hlsl
    assert (
        "int __crossgl_byteaddress_atomic_compare_exchange_int("
        "RWByteAddressBuffer buffer, uint offset, int compareValue, int value)" in hlsl
    )
    assert (
        "buffer.InterlockedCompareExchange("
        "offset, asuint(compareValue), asuint(value), original);" in hlsl
    )
    assert (
        "int oldCounter = __crossgl_byteaddress_atomic_add_int("
        "signedAtomicBlock, 0, -1);" in hlsl
    )
    assert (
        "int oldBin = __crossgl_byteaddress_atomic_exchange_int("
        "signedAtomicBlock, 12, oldCounter);" in hlsl
    )
    assert (
        "int minBin = __crossgl_byteaddress_atomic_min_int("
        "signedAtomicBlock, 4, -2);" in hlsl
    )
    assert (
        "int maxBin = __crossgl_byteaddress_atomic_max_int("
        "signedAtomicBlock, 4, minBin);" in hlsl
    )
    assert (
        "int andBin = __crossgl_byteaddress_atomic_and_int("
        "signedAtomicBlock, 8, 15);" in hlsl
    )
    assert (
        "int orBin = __crossgl_byteaddress_atomic_or_int("
        "signedAtomicBlock, 8, andBin);" in hlsl
    )
    assert (
        "int xorBin = __crossgl_byteaddress_atomic_xor_int("
        "signedAtomicBlock, 12, orBin);" in hlsl
    )
    assert (
        "int casBin = __crossgl_byteaddress_atomic_compare_exchange_int("
        "signedAtomicBlock, 16, xorBin, -7);" in hlsl
    )
    assert "__crossgl_byteaddress_atomic_add_int(signedAtomicBlock, 8, casBin);" in hlsl

    assert "device uchar* signedAtomicBlock [[buffer(18)]]" in metal
    assert (
        "int oldCounter = atomic_fetch_add_explicit("
        "reinterpret_cast<device atomic_int*>(signedAtomicBlock + 0), "
        "-1, memory_order_relaxed);" in metal
    )
    assert (
        "int oldBin = atomic_exchange_explicit("
        "reinterpret_cast<device atomic_int*>(signedAtomicBlock + 12), "
        "oldCounter, memory_order_relaxed);" in metal
    )
    assert (
        "int minBin = atomic_fetch_min_explicit("
        "reinterpret_cast<device atomic_int*>(signedAtomicBlock + 4), "
        "-2, memory_order_relaxed);" in metal
    )
    assert (
        "int casBin = __crossgl_buffer_atomic_compare_exchange_int("
        "signedAtomicBlock, 16, xorBin, -7);" in metal
    )
    assert (
        "atomic_fetch_add_explicit("
        "reinterpret_cast<device atomic_int*>(signedAtomicBlock + 8), "
        "casBin, memory_order_relaxed);" in metal
    )

    assert "layout(std430, binding = 18) buffer SignedAtomicBlock" in glsl
    assert "int oldCounter = atomicAdd(signedAtomicBlock.counter, (-1));" in glsl
    assert "int oldBin = atomicExchange(signedAtomicBlock.bins[2], oldCounter);" in glsl
    assert "int minBin = atomicMin(signedAtomicBlock.bins[0], (-2));" in glsl
    assert "int maxBin = atomicMax(signedAtomicBlock.bins[0], minBin);" in glsl
    assert "int andBin = atomicAnd(signedAtomicBlock.bins[1], 15);" in glsl
    assert "int orBin = atomicOr(signedAtomicBlock.bins[1], andBin);" in glsl
    assert "int xorBin = atomicXor(signedAtomicBlock.bins[2], orBin);" in glsl
    assert (
        "int casBin = atomicCompSwap(signedAtomicBlock.bins[3], xorBin, (-7));" in glsl
    )
    assert "atomicAdd(signedAtomicBlock.bins[1], casBin);" in glsl


def test_codegen_mixed_ssbo_runtime_array_atomics_lower_dynamic_offsets():
    code = """
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

    crossgl = generate_crossgl(code, "compute")
    assert (
        "RuntimeAtomicBlock runtimeAtomicBlock "
        "@glsl_buffer_block(std430) @binding(19);" in crossgl
    )
    assert (
        "RuntimeSignedAtomicBlock runtimeSignedAtomicBlock "
        "@glsl_buffer_block(std430) @binding(20);" in crossgl
    )
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer runtimeAtomicBlock : register(u19);" in hlsl
    assert "RWByteAddressBuffer runtimeSignedAtomicBlock : register(u20);" in hlsl
    assert "uint i = runtimeAtomicBlock.Load(0);" in hlsl
    assert (
        "uint oldValue = __crossgl_byteaddress_atomic_add_uint("
        "runtimeAtomicBlock, (4 + i * 4), 1u);" in hlsl
    )
    assert (
        "uint swapped = __crossgl_byteaddress_atomic_compare_exchange_uint("
        "runtimeAtomicBlock, (4 + (i + 1u) * 4), oldValue, 7u);" in hlsl
    )
    assert "int j = asint(runtimeSignedAtomicBlock.Load(0));" in hlsl
    assert (
        "int oldSigned = __crossgl_byteaddress_atomic_min_int("
        "runtimeSignedAtomicBlock, (4 + j * 4), -2);" in hlsl
    )
    assert (
        "int exchanged = __crossgl_byteaddress_atomic_exchange_int("
        "runtimeSignedAtomicBlock, (4 + (j + 1) * 4), oldSigned);" in hlsl
    )
    assert (
        "__crossgl_byteaddress_atomic_add_int("
        "runtimeSignedAtomicBlock, (4 + j * 4), exchanged);" in hlsl
    )

    assert "device uchar* runtimeAtomicBlock [[buffer(19)]]" in metal
    assert "device uchar* runtimeSignedAtomicBlock [[buffer(20)]]" in metal
    assert (
        "uint oldValue = atomic_fetch_add_explicit("
        "reinterpret_cast<device atomic_uint*>(runtimeAtomicBlock + (4 + i * 4)), "
        "1u, memory_order_relaxed);" in metal
    )
    assert (
        "uint swapped = __crossgl_buffer_atomic_compare_exchange_uint("
        "runtimeAtomicBlock, (4 + (i + 1u) * 4), oldValue, 7u);" in metal
    )
    assert (
        "int oldSigned = atomic_fetch_min_explicit("
        "reinterpret_cast<device atomic_int*>("
        "runtimeSignedAtomicBlock + (4 + j * 4)), -2, memory_order_relaxed);" in metal
    )
    assert (
        "int exchanged = atomic_exchange_explicit("
        "reinterpret_cast<device atomic_int*>("
        "runtimeSignedAtomicBlock + (4 + (j + 1) * 4)), "
        "oldSigned, memory_order_relaxed);" in metal
    )
    assert (
        "atomic_fetch_add_explicit("
        "reinterpret_cast<device atomic_int*>("
        "runtimeSignedAtomicBlock + (4 + j * 4)), "
        "exchanged, memory_order_relaxed);" in metal
    )

    assert "layout(std430, binding = 19) buffer RuntimeAtomicBlock" in glsl
    assert "layout(std430, binding = 20) buffer RuntimeSignedAtomicBlock" in glsl
    assert "uint oldValue = atomicAdd(runtimeAtomicBlock.values[i], 1u);" in glsl
    assert (
        "uint swapped = atomicCompSwap("
        "runtimeAtomicBlock.values[(i + 1u)], oldValue, 7u);" in glsl
    )
    assert (
        "int oldSigned = atomicMin(runtimeSignedAtomicBlock.values[j], (-2));" in glsl
    )
    assert (
        "int exchanged = atomicExchange("
        "runtimeSignedAtomicBlock.values[(j + 1)], oldSigned);" in glsl
    )
    assert "atomicAdd(runtimeSignedAtomicBlock.values[j], exchanged);" in glsl


def test_codegen_mixed_ssbo_invalid_atomics_emit_target_diagnostics():
    code = """
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

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert (
        "uint readonlyOld = /* unsupported HLSL GLSL buffer block atomic: "
        "atomicAdd cannot write readonly ByteAddressBuffer */ 0u;" in hlsl
    )
    assert (
        "float floatOld = /* unsupported HLSL GLSL buffer block atomic: "
        "atomicAdd currently supports only int or uint buffer members */ 0;" in hlsl
    )
    assert (
        "uint vectorOld = /* unsupported HLSL GLSL buffer block atomic: "
        "atomicAdd requires a scalar int or uint buffer member */ 0u;" in hlsl
    )
    assert (
        "float matrixOld = /* unsupported HLSL GLSL buffer block atomic: "
        "atomicAdd requires a scalar int or uint buffer member */ 0;" in hlsl
    )
    assert "Interlocked" not in hlsl
    assert "__crossgl_byteaddress_atomic" not in hlsl

    assert (
        "uint readonlyOld = /* unsupported Metal GLSL buffer block atomic: "
        "atomicAdd cannot write readonly device buffer */ 0u;" in metal
    )
    assert (
        "float floatOld = /* unsupported Metal GLSL buffer block atomic: "
        "atomicAdd currently supports only int or uint buffer members */ 0;" in metal
    )
    assert (
        "uint vectorOld = /* unsupported Metal GLSL buffer block atomic: "
        "atomicAdd requires a scalar int or uint buffer member */ 0u;" in metal
    )
    assert (
        "float matrixOld = /* unsupported Metal GLSL buffer block atomic: "
        "atomicAdd requires a scalar int or uint buffer member */ 0;" in metal
    )
    assert "atomic_fetch_" not in metal
    assert "__crossgl_buffer_atomic" not in metal

    assert "uint readonlyOld = atomicAdd(readAtomicBlock.value, 1u);" in glsl
    assert "float floatOld = atomicAdd(floatAtomicBlock.value, 1.0);" in glsl
    assert "uint vectorOld = atomicAdd(vectorAtomicBlock.value, 1u);" in glsl
    assert "float matrixOld = atomicAdd(matrixAtomicBlock.value, 1.0);" in glsl


def test_codegen_mixed_glsl_preprocessors_are_filtered_for_non_glsl_targets():
    code = """
    #version 300 es
    #extension GL_ARB_separate_shader_objects : enable
    precision highp float;

    void main() { }
    """

    crossgl = generate_crossgl(code, "compute")
    assert "#version 300 es" in crossgl
    assert "#extension GL_ARB_separate_shader_objects : enable" in crossgl
    assert "precision highp float;" in crossgl
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "#version" not in hlsl
    assert "#extension" not in hlsl
    assert "precision highp float" not in hlsl

    assert "#version" not in metal
    assert "#extension" not in metal
    assert "precision highp float" not in metal
    assert "#include <metal_stdlib>" in metal

    assert "#version 300 es" in glsl
    assert "#extension GL_ARB_separate_shader_objects : enable" in glsl
    assert "precision highp float;" in glsl


def test_codegen_mixed_ssbo_fixed_only_blocks_lower_to_explicit_offsets():
    code = """
    #version 450 core
    layout(std430, binding = 97) buffer FixedOnlyBlock {
        uint count;
        vec3 axis;
        mat2 transform;
        float weights[3];
    } fixedOnly;

    void main() {
        uint count = fixedOnly.count;
        vec3 axis = fixedOnly.axis;
        mat2 transform = fixedOnly.transform;
        float weight = fixedOnly.weights[2];
        fixedOnly.count = count + 1u;
        fixedOnly.axis = axis;
        fixedOnly.transform = transform;
        fixedOnly.weights[1] = weight;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "RWByteAddressBuffer fixedOnly : register(u97);" in hlsl
    assert "struct FixedOnlyBlock" not in hlsl
    assert "uint count = fixedOnly.Load(0);" in hlsl
    assert "float3 axis = asfloat(fixedOnly.Load3(16));" in hlsl
    assert (
        "float2x2 transform = float2x2(asfloat(fixedOnly.Load2(32)), "
        "asfloat(fixedOnly.Load2(40)));" in hlsl
    )
    assert "float weight = asfloat(fixedOnly.Load(56));" in hlsl
    assert "fixedOnly.Store(0, (count + 1u));" in hlsl
    assert "fixedOnly.Store3(16, asuint(axis));" in hlsl
    assert "fixedOnly.Store2(32, asuint(transform[0]));" in hlsl
    assert "fixedOnly.Store2(40, asuint(transform[1]));" in hlsl
    assert "fixedOnly.Store(52, asuint(weight));" in hlsl
    assert "unsupported HLSL GLSL buffer block" not in hlsl

    assert "device uchar* fixedOnly [[buffer(97)]]" in metal
    assert "struct FixedOnlyBlock" not in metal
    assert (
        "uint count = (*reinterpret_cast<const device uint*>(fixedOnly + 0));" in metal
    )
    assert (
        "float3 axis = float3((*reinterpret_cast<const device float*>"
        "(fixedOnly + 16)), (*reinterpret_cast<const device float*>"
        "(fixedOnly + 20)), (*reinterpret_cast<const device float*>"
        "(fixedOnly + 24)));" in metal
    )
    assert (
        "float2x2 transform = float2x2(float2((*reinterpret_cast<const device "
        "float*>(fixedOnly + 32)), (*reinterpret_cast<const device float*>"
        "(fixedOnly + 36))), float2((*reinterpret_cast<const device float*>"
        "(fixedOnly + 40)), (*reinterpret_cast<const device float*>"
        "(fixedOnly + 44))));" in metal
    )
    assert (
        "float weight = (*reinterpret_cast<const device float*>(fixedOnly + 56));"
        in metal
    )
    assert "(*reinterpret_cast<device uint*>(fixedOnly + 0)) = count + 1u;" in metal
    assert "(*reinterpret_cast<device float*>(fixedOnly + 16))" in metal
    assert "(*reinterpret_cast<device float*>(fixedOnly + 32))" in metal
    assert "(*reinterpret_cast<device float*>(fixedOnly + 52)) = weight;" in metal
    assert "unsupported Metal GLSL buffer block" not in metal

    assert "layout(std430, binding = 97) buffer FixedOnlyBlock" in glsl
    assert "} fixedOnly;" in glsl
    assert "fixedOnly.weights[1] = weight;" in glsl


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
        "asuint((asfloat(dynamicBlock.Load((4 + i * 4))) + 1.0)));" in hlsl
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
        "(intOpsBlock.Load((4 + i * 4)) % 3u));" in hlsl
    )
    assert "unsupported HLSL GLSL buffer block compound store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_fixed_vector_float_mod_compound_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 69) buffer FixedVectorOpsBlock {
        uint index;
        vec2 offsets[4];
        float data[];
    } fixedVectorOpsBlock;

    void main() {
        uint i = fixedVectorOpsBlock.index;
        fixedVectorOpsBlock.offsets[i] %= vec2(2.0);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "RWByteAddressBuffer fixedVectorOpsBlock : register(u69);" in hlsl
    assert "unsupported HLSL GLSL buffer block compound store" in hlsl
    assert "operator %= is not supported for float buffer members" in hlsl
    assert "fixedVectorOpsBlock.Store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_readonly_runtime_scalar_write_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 70) readonly buffer ReadWriteBlock {
        uint count;
        float values[];
    } readWriteBlock;

    void main() {
        uint i = readWriteBlock.count;
        readWriteBlock.values[i] = 1.0;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer readWriteBlock : register(t70);" in hlsl
    assert "RWByteAddressBuffer readWriteBlock" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "readWriteBlock.Store" not in hlsl


def test_codegen_mixed_ssbo_hlsl_readonly_fixed_vector_write_is_diagnostic():
    code = """
    #version 450 core
    layout(std430, binding = 71) readonly buffer ReadFixedVectorWriteBlock {
        uint index;
        vec4 values[2];
        float tail[];
    } readFixedVectorWriteBlock;

    void main() {
        uint i = readFixedVectorWriteBlock.index;
        readFixedVectorWriteBlock.values[i] += vec4(1.0);
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    assert "ByteAddressBuffer readFixedVectorWriteBlock : register(t71);" in hlsl
    assert "RWByteAddressBuffer readFixedVectorWriteBlock" not in hlsl
    assert "readonly ByteAddressBuffer cannot be written" in hlsl
    assert "readFixedVectorWriteBlock.Store" not in hlsl


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
    assert (
        "ReadBlock readBlock @glsl_buffer_block(std430) @binding(5) @readonly;"
        in crossgl
    )

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
        "asfloat(matrix2Block.Load2(8)));" in hlsl
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
        "asfloat(matrix3Block.Load3(32)), asfloat(matrix3Block.Load3(48)));" in hlsl
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
        f"asfloat(nonSquareBlock.{load_method}({offset}))" for offset in column_offsets
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
        "asfloat(matrix4Block.Load4(64)));" in hlsl
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
        "asfloat(matrixArrayBlock.Load4(48)));" in hlsl
    )
    assert (
        "float4x4 selected = float4x4("
        "asfloat(matrixArrayBlock.Load4((i * 64))), "
        "asfloat(matrixArrayBlock.Load4((i * 64 + 16))), "
        "asfloat(matrixArrayBlock.Load4((i * 64 + 32))), "
        "asfloat(matrixArrayBlock.Load4((i * 64 + 48))));" in hlsl
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
        f"(i * {matrix_stride})" if offset == 0 else f"(i * {matrix_stride} + {offset})"
        for offset in column_offsets
    ]
    dynamic_loads = ", ".join(
        f"asfloat(nonSquareMatrixArrayBlock.{load_method}({offset}))"
        for offset in dynamic_offsets
    )

    assert (
        f"RWByteAddressBuffer nonSquareMatrixArrayBlock : register(u{binding});" in hlsl
    )
    assert f"{hlsl_type} first = {hlsl_type}({first_loads});" in hlsl
    assert f"{hlsl_type} selected = {hlsl_type}({dynamic_loads});" in hlsl
    for column, offset in enumerate(column_offsets):
        assert (
            f"nonSquareMatrixArrayBlock.{store_method}({offset}, "
            f"asuint(selected[{column}]));" in hlsl
        )
    for column, offset in enumerate(dynamic_offsets):
        assert (
            f"nonSquareMatrixArrayBlock.{store_method}({offset}, "
            f"asuint(first[{column}]));" in hlsl
        )
    assert (
        f"float tail = asfloat(nonSquareMatrixArrayBlock.Load({data_offset}));" in hlsl
    )
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
        "asfloat(matrixBlock.Load4(64))) + value);" in hlsl
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
        f"asfloat(matrixBlock.Load4(64))) {binary_op} {rhs_expr});" in hlsl
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
        "asfloat(matrixArrayBlock.Load4(112))) + value);" in hlsl
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
        (
            f"({runtime_offset} + i * {matrix_stride})"
            if offset == runtime_offset
            else f"({runtime_offset} + i * {matrix_stride} + {offset - runtime_offset})"
        )
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
        (
            f"({runtime_offset} + i * {matrix_stride})"
            if offset == runtime_offset
            else f"({runtime_offset} + i * {matrix_stride} + {offset - runtime_offset})"
        )
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
            f"asuint(selected[{column}]));" in hlsl
        )
    for column, offset in enumerate(column_offsets):
        assert (
            f"runtimeMatrixBlock.{store_method}({offset}, "
            f"asuint(first[{column}]));" in hlsl
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
        (
            f"({runtime_offset} + i * {matrix_stride})"
            if offset == runtime_offset
            else f"({runtime_offset} + i * {matrix_stride} + {offset - runtime_offset})"
        )
        for offset in column_offsets
    ]
    dynamic_loads = ", ".join(
        f"asfloat(runtimeMatrixBlock.{load_method}({offset}))"
        for offset in dynamic_offsets
    )

    assert f"RWByteAddressBuffer runtimeMatrixBlock : register(u{binding});" in hlsl
    assert (
        f"{hlsl_type} __crossgl_matrix_store_0 = "
        f"({hlsl_type}({dynamic_loads}) + selected);" in hlsl
    )
    for column, offset in enumerate(dynamic_offsets):
        assert (
            f"runtimeMatrixBlock.{store_method}({offset}, "
            f"asuint(__crossgl_matrix_store_0[{column}]));" in hlsl
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
        matrixBlock.tail = v;
    }
    """

    crossgl = generate_crossgl(code, "compute")
    shader_ast = parse_crossgl(crossgl)
    assert shader_ast is not None

    hlsl = HLSLCodeGen().generate(shader_ast)
    metal = MetalCodeGen().generate(shader_ast)
    glsl = GLSLCodeGen().generate(shader_ast)

    assert "unsupported HLSL GLSL buffer block MatrixBlock" in hlsl
    assert "ByteAddressBuffer offset lowering" in hlsl
    assert (
        "unsupported member transforms: runtime arrays must be the final "
        "buffer block member" in hlsl
    )
    assert "unsupported HLSL GLSL buffer block struct MatrixBlock omitted" in hlsl
    assert (
        "unsupported HLSL GLSL buffer block variable MatrixBlock matrixBlock omitted"
        in hlsl
    )
    assert "struct MatrixBlock {" not in hlsl
    assert "MatrixBlock matrixBlock;" not in hlsl
    assert "RWByteAddressBuffer matrixBlock" not in hlsl
    assert (
        "float v = 0 /* unsupported HLSL GLSL buffer block access matrixBlock: "
        "no target-side fallback declaration emitted */;" in hlsl
    )
    assert (
        "/* unsupported HLSL GLSL buffer block assignment matrixBlock: "
        "no target-side fallback declaration emitted */;" in hlsl
    )
    assert "matrixBlock.tail" not in hlsl

    assert "unsupported Metal GLSL buffer block MatrixBlock" in metal
    assert "explicit pointer/offset lowering" in metal
    assert (
        "unsupported member transforms: runtime arrays must be the final "
        "buffer block member" in metal
    )
    assert "unsupported Metal GLSL buffer block struct MatrixBlock omitted" in metal
    assert (
        "unsupported Metal GLSL buffer block variable MatrixBlock matrixBlock omitted"
        in metal
    )
    assert "struct MatrixBlock {" not in metal
    assert "MatrixBlock matrixBlock;" not in metal
    assert "device uchar* matrixBlock" not in metal
    assert (
        "float v = 0 /* unsupported Metal GLSL buffer block access matrixBlock: "
        "no target-side fallback declaration emitted */;" in metal
    )
    assert (
        "/* unsupported Metal GLSL buffer block assignment matrixBlock: "
        "no target-side fallback declaration emitted */;" in metal
    )
    assert "matrixBlock.tail" not in metal

    assert "layout(std430, binding = 7) buffer MatrixBlock" in glsl
    assert "} matrixBlock;" in glsl
    assert "matrixBlock.tail = v;" in glsl


if __name__ == "__main__":
    pytest.main()
