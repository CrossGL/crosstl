import ast
from pathlib import Path
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

HELPER_RANGE_SHADER = """
shader RangeForInValidation {
    int helper(int limit) {
        int total = 0;
        for i in 1..=limit {
            total = total + i;
        }
        return total;
    }
}
"""


FRAGMENT_RANGE_SHADER = """
shader FragmentRangeValidation {
    fragment {
        vec4 main() @ gl_FragColor {
            int total = 0;
            for i in 1..=3 {
                total = total + i;
            }
            return vec4(float(total), 0.0, 0.0, 1.0);
        }
    }
}
"""


METAL_FUNCTION_CONSTANT_FRAGMENT_SHADER = """
shader MetalFunctionConstantValidation {
    bool useFast @function_constant(0) = true;
    int mode @constant_id(1) = 2;
    float scale @function_constant(2) = 0.5;
    uint flags @function_constant(3);

    fragment {
        vec4 main() @ gl_FragColor {
            float value = useFast ? scale : 0.0;
            return vec4(value + float(mode) + float(flags), 0.0, 0.0, 1.0);
        }
    }
}
"""


METAL_WAVE_INTRINSICS_COMPUTE_SHADER = """
shader MetalWaveIntrinsicsValidation {
    uint helperLane(uint seed) {
        return WaveGetLaneIndex() + seed;
    }

    uint helperBoth(uint seed) {
        return helperLane(seed) + WaveGetLaneCount();
    }

    uvec4 helperMatch(uint seed) {
        return WaveMatch(seed);
    }

    uint helperMulti(uint seed, uvec4 mask) {
        return WaveMultiPrefixSum(seed, mask);
    }

    uvec2 helperMultiVector(uvec2 seed, uvec4 mask) {
        return WaveMultiPrefixBitOr(seed, mask);
    }

    compute {
        void main() {
            uint value = 1u;
            uvec2 lanes = uvec2(value, value + 1u);
            uint lane = WaveGetLaneIndex();
            uint laneCount = WaveGetLaneCount();
            uint helperValue = helperBoth(value);
            uvec4 helperMatchValue = helperMatch(value);
            uint sumValue = WaveActiveSum(value);
            uint productValue = WaveActiveProduct(value + 1u);
            uint minValue = WaveActiveMin(sumValue);
            uint maxValue = WaveActiveMax(productValue);
            uint andValue = WaveActiveBitAnd(maxValue);
            uint orValue = WaveActiveBitOr(andValue);
            uint xorValue = WaveActiveBitXor(orValue);
            uint prefixSum = WavePrefixSum(xorValue);
            uint prefixProduct = WavePrefixProduct(value + 1u);
            bool anyLane = WaveActiveAnyTrue(prefixSum > 0u);
            bool allLane = WaveActiveAllTrue(prefixProduct > 0u);
            bool equalScalar = WaveActiveAllEqual(value);
            bool equalVector = WaveActiveAllEqual(lanes);
            uvec4 ballot = WaveActiveBallot(anyLane);
            uvec4 matchMask = WaveMatch(value);
            mat2 matrixValue = mat2(1.0);
            mat2 matrixDiagnostic = WaveMultiPrefixSum(matrixValue, ballot);
            uint multiSum = WaveMultiPrefixSum(value, ballot);
            uint multiProduct = WaveMultiPrefixProduct(value + 1u, ballot);
            uint multiCount = WaveMultiPrefixCountBits(anyLane, ballot);
            uint multiAnd = WaveMultiPrefixBitAnd(value, ballot);
            uint multiOr = WaveMultiPrefixBitOr(value, ballot);
            uint multiXor = WaveMultiPrefixBitXor(value, ballot);
            uint helperMultiValue = helperMulti(value, ballot);
            uvec2 vectorSum = WaveMultiPrefixSum(lanes, ballot);
            uvec2 vectorProduct = WaveMultiPrefixProduct(lanes + uvec2(1u, 1u), ballot);
            uvec2 vectorBits = WaveMultiPrefixBitXor(lanes, ballot);
            uvec2 helperVectorValue = helperMultiVector(lanes, ballot);
            uint count = WaveActiveCountBits(allLane);
            uint prefixCount = WavePrefixCountBits(anyLane);
            uint broadcast = WaveReadLaneAt(prefixSum, 0u);
            uint firstValue = WaveReadLaneFirst(broadcast);
            uint quadX = QuadReadAcrossX(firstValue);
            uint quadY = QuadReadAcrossY(quadX);
            uint quadDiagonal = QuadReadAcrossDiagonal(quadY);
            uint quadLane = QuadReadLaneAt(quadDiagonal, 0u);
            bool quadAny = QuadAny(anyLane);
            bool quadAll = QuadAll(allLane);
            uint folded = minValue + count + prefixCount + quadLane + ballot.x + matchMask.x + helperMatchValue.y + lane + helperValue + multiSum + multiProduct + multiCount + multiAnd + multiOr + multiXor + helperMultiValue;
            folded = folded + vectorSum.x + vectorProduct.y + vectorBits.x + helperVectorValue.y;
            folded = folded + (equalScalar ? value : 0u) + (equalVector ? lanes.x : 0u);
            folded = folded + (quadAny ? quadX : quadY);
            folded = folded + (quadAll ? quadDiagonal : firstValue) + laneCount;
        }
    }
}
"""


FRAGMENT_STRUCT_INPUT_SHADER = """
shader FragmentStructInputValidation {
    struct VSOutput {
        vec2 uv @ TEXCOORD0;
        vec3 normal @ NORMAL;
    };

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            return vec4(input.uv, input.normal.x, 1.0);
        }
    }
}
"""


SWITCH_MATCH_CASE_SCOPE_FRAGMENT_SHADER = """
shader SwitchMatchCaseScopeValidation {
    int chooseSwitch(int mode) {
        int value = 0;
        switch (mode) {
            case 0:
            case 1:
                int scoped = value + 1;
                value = scoped;
                break;
            default:
                int scoped = value + 2;
                value = scoped;
                break;
        }
        return value;
    }

    int chooseMatch(int mode) {
        int value = 0;
        match mode {
            0 => { int scoped = value + 3; value = scoped; }
            _ => { int scoped = value + 4; value = scoped; }
        }
        return value;
    }

    fragment {
        vec4 main() @ gl_FragColor {
            int total = chooseSwitch(1) + chooseMatch(2);
            return vec4(float(total), 0.0, 0.0, 1.0);
        }
    }
}
"""


SWITCH_MATCH_TEXTURE_CASE_SCOPE_FRAGMENT_SHADER = """
shader SwitchMatchTextureCaseScopeValidation {
    sampler2D textures[4];
    sampler samplers[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
    };

    vec4 chooseSwitch(sampler2D textures[4], sampler samplers[4], int mode, vec2 uv) {
        vec4 color = vec4(0.0);
        switch (mode) {
            case 0:
            case 1:
                vec4 scoped = texture(textures[0], samplers[0], uv);
                color = scoped;
                break;
            default:
                vec4 scoped = texture(textures[1], samplers[1], uv);
                color = scoped;
                break;
        }
        return color;
    }

    vec4 chooseMatch(sampler2D textures[4], sampler samplers[4], int mode, vec2 uv) {
        vec4 color = vec4(0.0);
        match mode {
            0 => { vec4 scoped = texture(textures[2], samplers[2], uv); color = scoped; }
            _ => { vec4 scoped = texture(textures[3], samplers[3], uv); color = scoped; }
        }
        return color;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return chooseSwitch(textures, samplers, 1, input.uv) + chooseMatch(textures, samplers, 2, input.uv);
        }
    }
}
"""


SWITCH_MATCH_IMAGE_CASE_SCOPE_COMPUTE_SHADER = """
shader SwitchMatchImageCaseScopeValidation {
    image2D rgFloatImages @rg32f[4];

    vec2 chooseSwitch(image2D images[4] @rg32f, int mode, ivec2 pixel) {
        vec2 color = vec2(0.0);
        switch (mode) {
            case 0:
            case 1:
                vec2 scoped = imageLoad(images[0], pixel);
                imageStore(images[1], pixel, scoped + vec2(1.0));
                color = scoped;
                break;
            default:
                vec2 scoped = imageLoad(images[2], pixel);
                imageStore(images[3], pixel, scoped + vec2(2.0));
                color = scoped;
                break;
        }
        return color;
    }

    vec2 chooseMatch(image2D images[4] @rg32f, int mode, ivec2 pixel) {
        vec2 color = vec2(0.0);
        match mode {
            0 => {
                vec2 scoped = imageLoad(images[0], pixel);
                imageStore(images[1], pixel, scoped + vec2(3.0));
                color = scoped;
            }
            _ => {
                vec2 scoped = imageLoad(images[2], pixel);
                imageStore(images[3], pixel, scoped + vec2(4.0));
                color = scoped;
            }
        }
        return color;
    }

    compute {
        void main() {
            vec2 first = chooseSwitch(rgFloatImages, 1, ivec2(0, 1));
            vec2 second = chooseMatch(rgFloatImages, 2, ivec2(2, 3));
            imageStore(rgFloatImages[0], ivec2(4, 5), first + second);
        }
    }
}
"""


DO_WHILE_SWITCH_MATCH_IMAGE_COMPUTE_SHADER = """
shader DoWhileSwitchMatchImageValidation {
    image2D rgFloatImages @rg32f[4];

    vec2 chooseImage(image2D images[4] @rg32f, int mode, ivec2 pixel) {
        vec2 color = vec2(0.0);
        do {
            switch (mode) {
                case 0:
                    vec2 scoped = imageLoad(images[0], pixel);
                    imageStore(images[1], pixel, scoped + vec2(1.0));
                    color = scoped;
                    break;
                default:
                    break;
            }
            match mode {
                1 => {
                    vec2 scoped = imageLoad(images[2], pixel);
                    imageStore(images[3], pixel, scoped + vec2(2.0));
                    color = color + scoped;
                }
                _ => {
                }
            }
        } while (false);
        return color;
    }

    compute {
        void main() {
            vec2 result = chooseImage(rgFloatImages, 1, ivec2(0, 1));
            imageStore(rgFloatImages[0], ivec2(2, 3), result);
        }
    }
}
"""


SPIRV_COMPLEX_RESOURCE_COMPUTE_SHADER = """
struct SampleEnvelope {
    vec4 color;
    vec4 scaled;
};

shader SpirvComplexResourceValidation {
    sampler2d textureGrid[2][3];
    image2D imageGrid @rgba16f[2][3];

    SampleEnvelope chooseSample(
        sampler2d dynTex[][3],
        image2D dynImages[][3] @rgba16f,
        int layer,
        int slot,
        bool preferSampled,
        float weight,
        vec2 uv,
        ivec2 pixel
    ) {
        SampleEnvelope result;
        result.color = vec4(0.0);
        vec4 sampled = texture(dynTex[layer][slot], uv);
        vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
        result.scaled = (preferSampled ? sampled : loaded) * weight;
        return result;
    }

    compute {
        void main() {
            vec2 uv = vec2(0.5, 0.25);
            ivec2 pixel = ivec2(0);
            SampleEnvelope envelope = chooseSample(
                textureGrid,
                imageGrid,
                1,
                2,
                true,
                0.5,
                uv,
                pixel
            );
        }
    }
}
"""


SPIRV_SYNCHRONIZATION_COMPUTE_SHADER = """
shader SpirvSynchronizationValidation {
    compute {
        void main() {
            barrier();
            workgroupBarrier();
            groupMemoryBarrier();
            memoryBarrierShared();
            memoryBarrierBuffer();
            memoryBarrierImage();
            allMemoryBarrier();
            memoryBarrier();
        }
    }
}
"""


SPIRV_WAVE_INTRINSICS_COMPUTE_SHADER = """
shader SpirvWaveValidation {
    compute {
        void main() {
            uint lane = WaveGetLaneIndex();
            uint count = WaveGetLaneCount();
            uint sumValue = WaveActiveSum(lane);
            uint productValue = WaveActiveProduct(lane + 1u);
            uint minValue = WaveActiveMin(sumValue);
            uint maxValue = WaveActiveMax(productValue);
            uint andValue = WaveActiveBitAnd(maxValue);
            uint orValue = WaveActiveBitOr(andValue);
            uint xorValue = WaveActiveBitXor(orValue);
            uint prefixSum = WavePrefixSum(xorValue);
            uint prefixProduct = WavePrefixProduct(lane + 1u);
            bool first = WaveIsFirstLane();
            bool anyLane = WaveActiveAnyTrue(prefixSum > 0u);
            bool allLane = WaveActiveAllTrue(prefixProduct > 0u);
            uvec4 ballot = WaveActiveBallot(anyLane);
            uint broadcast = WaveReadLaneAt(prefixSum, 0u);
            uint firstValue = WaveReadLaneFirst(broadcast + count);
        }
    }
}
"""


SPIRV_UNIFORM_BUFFER_COMPUTE_SHADER = """
shader SpirvUniformBufferValidation {
    cbuffer Camera @set(1) @binding(2) {
        mat4 viewProj;
        vec4 tint;
        float exposure;
        vec4 palette[2];
    }

    sampler2D colorMap;
    sampler linearSampler;

    compute {
        void main() {
            mat4 localView = viewProj;
            vec4 color = tint * exposure + palette[1];
            vec4 sampled = texture(colorMap, linearSampler, vec2(0.5, 0.5));
            vec4 result = color + sampled;
        }
    }
}
"""


SPIRV_STRUCTURED_BUFFER_COMPUTE_SHADER = """
shader SpirvStructuredBufferValidation {
    struct Particle {
        vec4 position;
        float mass;
    };

    RWStructuredBuffer<Particle> particles @set(1) @binding(4);
    StructuredBuffer<float> weights;

    compute {
        void main() {
            Particle p = buffer_load(particles, 0u);
            float weight = buffer_load(weights, 1u);
            p.mass = p.mass + weight;
            buffer_store(particles, 2u, p);
            particles[1u].mass = p.mass;
        }
    }
}
"""


SPIRV_GLSL_BUFFER_BLOCK_COMPUTE_SHADER = """
shader SpirvGlslBufferBlockValidation {
    struct Particle {
        vec4 position;
        float mass;
    };

    layout(std430, set = 1, binding = 4) buffer ParticleBlock {
        Particle particles[];
    } particleBlock;

    struct Std140Block {
        uint count;
        mat2 basis;
        float weights[3];
        float values[];
    };

    Std140Block std140Block @glsl_buffer_block(std140) @binding(6);

    struct Std140Leaf {
        float value;
        float weights[2];
    };

    struct Std140Aggregate {
        Std140Leaf item;
        Std140Leaf items[2];
    };

    Std140Leaf std140Scratch;
    Std140Aggregate std140Aggregate @glsl_buffer_block(std140) @binding(7);

    float readMass(ParticleBlock block @glsl_buffer_block(std430), uint index) {
        return block.particles[index].mass;
    }

    Std140Leaf readStd140Leaf() {
        return std140Aggregate.item;
    }

    void writeMass(
        ParticleBlock block @glsl_buffer_block(std430),
        uint index,
        float value
    ) {
        block.particles[index].mass = value;
    }

    compute {
        layout(std430, binding = 2) readonly buffer float values[];
        layout(std430, binding = 3) writeonly buffer float outValues[];

        void main() {
            float mass = particleBlock.particles[0u].mass;
            float value = buffer_load(values, 1u);
            float helperMass = readMass(particleBlock, 0u);
            writeMass(particleBlock, 1u, mass + value + helperMass);
            uint index = std140Block.count;
            mat2 basis = std140Block.basis;
            float weight = std140Block.weights[2];
            float dynamicValue = std140Block.values[index];
            std140Block.basis = basis;
            std140Block.weights[1] = weight + dynamicValue;
            std140Block.values[index] = weight + dynamicValue;
            std140Scratch = readStd140Leaf();
            std140Aggregate.items[1] = std140Scratch;
            buffer_store(outValues, 0u, mass);
        }
    }
}
"""


SPIRV_SCALAR_BUFFER_BLOCK_COMPUTE_SHADER = """
shader SpirvScalarBufferBlockValidation {
    struct ScalarBlock {
        float a;
        vec2 b;
        vec2 c;
        vec3 packed;
        float tail;
        mat2 basis;
        float weights[3];
        vec3 vectors[2];
    };

    ScalarBlock scalarBlock @glsl_buffer_block(scalar) @binding(11);

    compute {
        void main() {
            vec2 b = scalarBlock.b;
            mat2 basis = scalarBlock.basis;
            float weight = scalarBlock.weights[2];
            vec3 vectorValue = scalarBlock.vectors[1];
            scalarBlock.c = b;
            scalarBlock.basis = basis;
            scalarBlock.weights[1] = weight;
            scalarBlock.vectors[0] = vectorValue;
        }
    }
}
"""


SPIRV_GLSL_BUFFER_BLOCK_ARRAY_COMPUTE_SHADER = """
shader SpirvGlslBufferBlockArrayValidation {
    struct Particle {
        vec4 position;
        float mass;
    };

    struct ParticleBlock {
        uint count;
        Particle particles[];
    };

    struct FixedBlock {
        float a;
        vec3 packed;
        float tail;
        vec3 values[2];
    };

    ParticleBlock runtimeBlocks[] @glsl_buffer_block(std430)
        @binding(18) @readonly;
    ParticleBlock mutableRuntimeBlocks[] @glsl_buffer_block(std430) @binding(20);
    FixedBlock fixedBlocks[2] @glsl_buffer_block(scalar) @binding(19);

    compute {
        void main() {
            uint index = runtimeBlocks[0].count;
            float mass = runtimeBlocks[0].particles[index].mass;
            Particle particle = runtimeBlocks[0].particles[index];
            mutableRuntimeBlocks[0].particles[index] = particle;
            vec3 value = fixedBlocks[1].values[1];
            fixedBlocks[0].packed = value + vec3(mass, mass, mass);
        }
    }
}
"""


SPIRV_STORAGE_BUFFER_ATOMICS_COMPUTE_SHADER = """
shader SpirvStorageBufferAtomicsValidation {
    struct AtomicBlock {
        uint counter;
        uint bins[];
    };

    struct SignedAtomicBlock {
        int counter;
        int bins[];
    };

    AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);
    SignedAtomicBlock signedAtomicBlock @glsl_buffer_block(std430) @binding(18);

    uint bump(AtomicBlock block @glsl_buffer_block(std430), uint index) {
        return atomicAdd(block.bins[index], 1u);
    }

    compute {
        void main() {
            uint index = atomicBlock.counter;
            uint oldValue = bump(atomicBlock, index);
            uint minValue = atomicMin(atomicBlock.bins[0], oldValue);
            uint swapped = atomicCompSwap(atomicBlock.bins[1], minValue, 7u);
            int oldSigned = atomicMin(signedAtomicBlock.bins[0], -2);
            int exchanged = atomicExchange(
                signedAtomicBlock.bins[1],
                oldSigned
            );
            atomicAdd(signedAtomicBlock.counter, exchanged);
        }
    }
}
"""


SPIRV_RESOURCE_MEMORY_QUALIFIER_COMPUTE_SHADER = """
shader SpirvResourceMemoryQualifierValidation {
    image2D inputImage @rgba32f @readonly @coherent;
    image2D outputImage @rgba32f @writeonly;
    RWStructuredBuffer<float> coherentValues @binding(2) @coherent;
    StructuredBuffer<float> readOnlyValues @binding(3);
    RWStructuredBuffer<float> writeOnlyValues @binding(4) @writeonly;
    uimage2D counters @r32ui @binding(5);

    struct QualifiedReadBlock {
        float values[];
    };

    struct QualifiedWriteBlock {
        float values[];
    };

    QualifiedReadBlock qualifiedReadBlocks[2] @glsl_buffer_block(std430)
        @binding(6) @readonly @coherent @volatile @restrict;
    QualifiedWriteBlock qualifiedWriteBlocks[2] @glsl_buffer_block(std430)
        @binding(7) @writeonly @coherent;

    vec4 readLeaf(image2D image @rgba32f, ivec2 pixel) {
        return imageLoad(image, pixel);
    }

    vec4 readForward(image2D image @rgba32f, ivec2 pixel) {
        return readLeaf(image, pixel);
    }

    void writePixel(image2D image @rgba32f, ivec2 pixel, vec4 value) {
        imageStore(image, pixel, value);
    }

    uint addCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
        return imageAtomicAdd(image, pixel, value);
    }

    float readBuffer(StructuredBuffer<float> data, uint index) {
        return buffer_load(data, index);
    }

    void writeBuffer(RWStructuredBuffer<float> data, uint index, float value) {
        buffer_store(data, index, value);
    }

    float readBlock(
        QualifiedReadBlock blocks[] @glsl_buffer_block(std430) @readonly,
        uint blockIndex,
        uint valueIndex
    ) {
        return blocks[blockIndex].values[valueIndex];
    }

    void writeBlock(
        QualifiedWriteBlock blocks[] @glsl_buffer_block(std430) @writeonly,
        uint blockIndex,
        uint valueIndex,
        float value
    ) {
        blocks[blockIndex].values[valueIndex] = value;
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            vec4 texel = readForward(inputImage, pixel);
            writePixel(outputImage, pixel, texel);
            uint oldCounter = addCounter(counters, pixel, 1u);
            imageStore(counters, pixel, oldCounter);
            float value = readBuffer(readOnlyValues, 0u);
            writeBuffer(writeOnlyValues, 1u, value + texel.x);
            writeBuffer(coherentValues, 2u, value);
            float blockValue = readBlock(qualifiedReadBlocks, 1u, 0u);
            writeBlock(qualifiedWriteBlocks, 0u, 1u, blockValue + value);
        }
    }
}
"""


SPIRV_IMAGE_ATOMIC_FORWARDING_COMPUTE_SHADER = """
shader SpirvImageAtomicForwardingValidation {
    uimage2D counters @r32ui;

    uint atomicLeaf(uimage2D image @r32ui, ivec2 pixel, uint value) {
        return imageAtomicAdd(image, pixel, value);
    }

    uint atomicForward(uimage2D image @r32ui, ivec2 pixel, uint value) {
        return atomicLeaf(image, pixel, value);
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(1, 2);
            uint previous = atomicForward(counters, pixel, 1u);
            imageStore(counters, pixel, previous);
        }
    }
}
"""


SPIRV_ADVANCED_TEXTURE_COMPUTE_SHADER = """
shader SpirvAdvancedTextureComputeValidation {
    sampler2D colorMap;
    sampler2DArray layerMap;
    sampler linearSampler;

    compute {
        void main() {
            vec2 uv = vec2(0.25, 0.75);
            vec3 uvLayer = vec3(0.25, 0.75, 1.0);
            vec2 ddx = vec2(0.1, 0.0);
            vec2 ddy = vec2(0.0, 0.1);
            ivec2 pixel = ivec2(4, 8);
            ivec2 offset = ivec2(1, 0);
            vec4 lod = textureLod(colorMap, linearSampler, uv, 2.0);
            vec4 lodOffset = textureLodOffset(
                colorMap,
                linearSampler,
                uv,
                2.0,
                offset
            );
            vec4 grad = textureGrad(colorMap, linearSampler, uv, ddx, ddy);
            vec4 gradOffset = textureGradOffset(
                colorMap,
                linearSampler,
                uv,
                ddx,
                ddy,
                offset
            );
            vec4 shifted = textureOffset(colorMap, linearSampler, uv, offset);
            vec4 gathered = textureGather(colorMap, linearSampler, uv, 1);
            vec4 gatheredOffset = textureGatherOffset(
                colorMap,
                linearSampler,
                uv,
                offset,
                2
            );
            vec4 gatheredOffsets = textureGatherOffsets(
                layerMap,
                linearSampler,
                uvLayer,
                offset,
                offset,
                offset,
                offset,
                3
            );
            vec4 fetched = texelFetch(colorMap, linearSampler, pixel, 0);
            vec4 fetchedOffset = texelFetchOffset(
                colorMap,
                linearSampler,
                pixel,
                0,
                offset
            );
        }
    }
}
"""


SPIRV_TEXTURE_QUERY_COMPUTE_SHADER = """
shader SpirvTextureQueryComputeValidation {
    sampler2D colorMap;
    sampler2DArray layerMap;
    samplerCube cubeMap;
    sampler2DMS msMap;
    sampler2DMSArray msLayers;
    sampler linearSampler;

    compute {
        void main() {
            vec2 uv = vec2(0.25, 0.75);
            vec3 uvLayer = vec3(0.25, 0.75, 1.0);
            vec3 direction = vec3(1.0, 0.0, 0.0);
            ivec2 texSize = textureSize(colorMap, 0);
            ivec3 arraySize = textureSize(layerMap, 1);
            ivec2 cubeSize = textureSize(cubeMap, 0);
            ivec2 msSize = textureSize(msMap, 0);
            ivec3 msLayerSize = textureSize(msLayers, 0);
            vec2 lod = textureQueryLod(layerMap, linearSampler, uvLayer);
            vec2 cubeLod = textureQueryLod(cubeMap, direction);
            int levels = textureQueryLevels(colorMap);
            int samples = textureSamples(msMap);
        }
    }
}
"""


SPIRV_SHADOW_TEXTURE_COMPUTE_SHADER = """
shader SpirvShadowTextureComputeValidation {
    sampler2DShadow shadowMap;
    sampler2DArrayShadow shadowArray;
    sampler compareSampler;

    compute {
        void main() {
            vec2 uv = vec2(0.25, 0.75);
            vec3 uvLayer = vec3(0.25, 0.75, 1.0);
            vec2 ddx = vec2(0.1, 0.0);
            vec2 ddy = vec2(0.0, 0.1);
            ivec2 offset = ivec2(1, 0);
            float depth = 0.5;
            float base = textureCompare(shadowMap, compareSampler, uv, depth);
            float lod = textureCompareLod(
                shadowMap,
                compareSampler,
                uv,
                depth,
                2.0
            );
            float lodOffset = textureCompareLodOffset(
                shadowMap,
                compareSampler,
                uv,
                depth,
                2.0,
                offset
            );
            float grad = textureCompareGrad(
                shadowMap,
                compareSampler,
                uv,
                depth,
                ddx,
                ddy
            );
            float gradOffset = textureCompareGradOffset(
                shadowMap,
                compareSampler,
                uv,
                depth,
                ddx,
                ddy,
                offset
            );
            float shifted = textureCompareOffset(
                shadowArray,
                compareSampler,
                uvLayer,
                depth,
                offset
            );
            vec4 gathered = textureGatherCompare(
                shadowMap,
                compareSampler,
                uv,
                depth
            );
            vec4 gatheredOffset = textureGatherCompareOffset(
                shadowArray,
                compareSampler,
                uvLayer,
                depth,
                offset
            );
        }
    }
}
"""


SPIRV_PROJECTED_TEXTURE_COMPUTE_SHADER = """
shader SpirvProjectedTextureComputeValidation {
    sampler2D colorMap;
    sampler2DArray layerMap;
    sampler3D volumeMap;
    samplerCube cubeMap;
    sampler2DShadow shadowMap;
    sampler2DArrayShadow shadowArray;
    samplerCubeShadow shadowCube;
    sampler linearSampler;
    sampler compareSampler;

    compute {
        void main() {
            vec3 uvq = vec3(0.25, 0.75, 2.0);
            vec4 uvqw = vec4(0.25, 0.75, 0.0, 2.0);
            vec4 uvLayerQ = vec4(0.25, 0.75, 1.0, 2.0);
            vec4 xyzq = vec4(0.25, 0.5, 0.75, 2.0);
            vec4 dirQ = vec4(1.0, 0.0, 0.0, 2.0);
            vec2 ddx = vec2(0.1, 0.0);
            vec2 ddy = vec2(0.0, 0.1);
            vec3 dxyz = vec3(0.1, 0.0, 0.0);
            float depth = 0.5;
            vec4 projected = textureProj(colorMap, linearSampler, uvq);
            vec4 projectedOffset = textureProjOffset(
                colorMap,
                linearSampler,
                uvq,
                ivec2(1, 0)
            );
            vec4 projectedLod = textureProjLod(colorMap, uvqw, 2.0);
            vec4 projectedLodOffset = textureProjLodOffset(
                layerMap,
                linearSampler,
                uvLayerQ,
                2.0,
                ivec2(1, 0)
            );
            vec4 projectedGrad = textureProjGrad(volumeMap, xyzq, dxyz, dxyz);
            vec4 projectedGradOffset = textureProjGradOffset(
                colorMap,
                uvq,
                ddx,
                ddy,
                ivec2(-1, 0)
            );
            vec4 cubeProjected = textureProj(cubeMap, linearSampler, dirQ);
            vec4 cubeProjectedGrad = textureProjGrad(
                cubeMap,
                linearSampler,
                dirQ,
                dxyz,
                dxyz
            );
            float shadow = textureCompareProj(
                shadowMap,
                compareSampler,
                uvq,
                depth
            );
            float shadowOffset = textureCompareProjOffset(
                shadowMap,
                compareSampler,
                uvq,
                depth,
                ivec2(1, 0)
            );
            float shadowLod = textureCompareProjLod(shadowMap, uvq, depth, 2.0);
            float shadowLodOffset = textureCompareProjLodOffset(
                shadowMap,
                uvq,
                depth,
                2.0,
                ivec2(1, 0)
            );
            float shadowGrad = textureCompareProjGrad(
                shadowArray,
                compareSampler,
                uvLayerQ,
                depth,
                ddx,
                ddy
            );
            float shadowGradOffset = textureCompareProjGradOffset(
                shadowArray,
                compareSampler,
                uvLayerQ,
                depth,
                ddx,
                ddy,
                ivec2(-1, 0)
            );
            float cubeShadow = textureCompareProj(
                shadowCube,
                compareSampler,
                dirQ,
                depth
            );
            float cubeShadowGrad = textureCompareProjGrad(
                shadowCube,
                compareSampler,
                dirQ,
                depth,
                dxyz,
                dxyz
            );
        }
    }
}
"""


SAMPLED_TEXTURE_ARRAY_FRAGMENT_SHADER = """
shader SampledTextureArrayValidation {
    sampler2D textures[4];
    sampler samplers[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
    };

    vec4 sampleUnsized(sampler2D textures[], sampler samplers[], vec2 uv) {
        return texture(textures[2], samplers[2], uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sampleUnsized(textures, samplers, input.uv);
        }
    }
}
"""


IMPLICIT_SAMPLER_ARRAY_FRAGMENT_SHADER = """
shader ImplicitSamplerArrayValidation {
    sampler2D textures[4];
    sampler texturesSampler[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
    };

    vec4 sampleGlobal(int index, vec2 uv) {
        return texture(textures[index], uv);
    }

    vec4 sampleParam(sampler2D textures[4], sampler texturesSampler[4], int index, vec2 uv) {
        return texture(textures[index], uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sampleGlobal(2, input.uv) + sampleParam(textures, texturesSampler, 1, input.uv);
        }
    }
}
"""


TEXTURE_LOCAL_ALIAS_FRAGMENT_SHADER = """
shader TextureLocalAliasValidation {
    sampler2D colorMap;
    sampler colorMapSampler;
    sampler2D textures[4];
    sampler texturesSampler[4];
    sampler linearSampler;

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        int layer @ TEXCOORD1;
    };

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            let alias = colorMap;
            let sampleState = linearSampler;
            vec4 implicitSample = texture(alias, input.uv);
            vec4 explicitSample = texture(alias, sampleState, input.uv);
            let layerAlias = textures[input.layer];
            vec4 arraySample = texture(layerAlias, input.uv);
            return implicitSample + explicitSample + arraySample;
        }
    }
}
"""


RESOURCE_TEXTURE_ARRAY_LOCAL_ALIAS_FRAGMENT_SHADER = """
shader ResourceTextureArrayLocalAliasValidation {
    sampler2D textures[4];
    sampler texturesSampler[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        int layer @ TEXCOORD1;
    };

    vec4 sampleParam(sampler2D paramTextures[4], sampler paramTexturesSampler[4], int layer, vec2 uv) {
        let paramAlias = paramTextures;
        let chainedAlias = paramAlias;
        return texture(chainedAlias[layer], uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            let texAlias = textures;
            return texture(texAlias[input.layer], input.uv)
                + sampleParam(textures, texturesSampler, input.layer, input.uv);
        }
    }
}
"""


RESOURCE_IMAGE_ARRAY_LOCAL_ALIAS_COMPUTE_SHADER = """
shader ResourceImageArrayLocalAliasValidation {
    image2D images @rgba32f[4];

    void copyParam(image2D paramImages[4] @rgba32f) {
        let paramAlias = paramImages;
        let chainedAlias = paramAlias;
        vec4 paramValue = imageLoad(chainedAlias[1], ivec2(1, 1));
        imageStore(paramAlias[0], ivec2(1, 1), paramValue);
    }

    compute {
        void main() {
            let imageAlias = images;
            vec4 value = imageLoad(imageAlias[1], ivec2(0, 0));
            copyParam(images);
            imageStore(imageAlias[0], ivec2(0, 0), value);
        }
    }
}
"""


METAL_RESOURCE_ARRAY_ELEMENT_HELPER_COMPUTE_SHADER = """
shader MetalResourceArrayElementHelperValidation {
    sampler2D textures[4];
    sampler samplers[4];
    image2D images @rgba32f[4];

    vec4 sampleOne(sampler2D tex, sampler samp, vec2 uv) {
        return texture(tex, samp, uv);
    }

    vec4 sampleArray(sampler2D texs[4], sampler samps[4], int layer, vec2 uv) {
        return texture(texs[layer], samps[layer], uv);
    }

    vec4 readOne(image2D image @rgba32f, ivec2 pixel) {
        return imageLoad(image, pixel);
    }

    compute {
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            int layer = int(gid.x & 3u);
            vec4 sampled = sampleOne(textures[layer], samplers[layer], vec2(0.5));
            vec4 sampledArray = sampleArray(textures, samplers, layer, vec2(0.25));
            vec4 stored = readOne(images[layer], ivec2(0, 0));
            imageStore(images[layer], ivec2(1, 0), sampled + sampledArray + stored);
        }
    }
}
"""


SAMPLER_ARRAY_LOCAL_ALIAS_FRAGMENT_SHADER = """
shader SamplerArrayLocalAliasValidation {
    sampler2D textures[4];
    sampler samplers[4];

    struct TexturePack {
        sampler2D textures[4];
        sampler samplers[4];
    };

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        int layer @ TEXCOORD1;
    };

    vec4 sampleParam(
        sampler2D paramTextures[4],
        sampler paramSamplers[4],
        int layer,
        vec2 uv
    ) {
        let paramSamplerAlias = paramSamplers;
        let chainedSamplerAlias = paramSamplerAlias;
        return texture(paramTextures[layer], chainedSamplerAlias[layer], uv);
    }

    vec4 samplePack(TexturePack pack, int layer, vec2 uv) {
        let packSamplerAlias = pack.samplers;
        let chainedPackSamplerAlias = packSamplerAlias;
        return texture(pack.textures[layer], chainedPackSamplerAlias[layer], uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            TexturePack pack;
            let samplerAlias = samplers;
            let chainedAlias = samplerAlias;
            return texture(
                textures[input.layer],
                chainedAlias[input.layer],
                input.uv
            ) + sampleParam(textures, samplers, input.layer, input.uv)
                + samplePack(pack, input.layer, input.uv);
        }
    }
}
"""


STRUCT_MEMBER_TEXTURE_ARRAY_LOCAL_ALIAS_FRAGMENT_SHADER = """
shader StructMemberTextureArrayLocalAliasValidation {
    struct TexturePack {
        sampler2D textures[4];
        sampler texturesSampler[4];
    };

    vec4 samplePack(TexturePack pack, int layer, vec2 uv) {
        let texAlias = pack.textures;
        let chainedAlias = texAlias;
        return texture(chainedAlias[layer], uv);
    }

    vec4 sampleLayer(TexturePack pack, int layer, vec2 uv) {
        let layerAlias = pack.textures[layer];
        return texture(layerAlias, uv);
    }

    fragment {
        vec4 main() @ gl_FragColor {
            TexturePack pack;
            return samplePack(pack, 2, vec2(0.5, 0.25)) +
                sampleLayer(pack, 1, vec2(0.25, 0.75));
        }
    }
}
"""


SAMPLED_TEXTURE_ARRAY_CONST_INDEX_FRAGMENT_SHADER = """
shader SampledTextureArrayConstIndexValidation {
    const int COUNT = 4;
    sampler2D textures[4];
    sampler samplers[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
    };

    vec4 sampleConst(sampler2D textures[4], sampler samplers[4], vec2 uv) {
        return texture(textures[COUNT - 1], samplers[COUNT - 1], uv);
    }

    vec4 sampleShadowed(sampler2D textures[4], sampler samplers[4], vec2 uv) {
        int COUNT = 0;
        return texture(textures[COUNT], samplers[COUNT], uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            int COUNT = 0;
            vec4 direct = texture(textures[COUNT], samplers[COUNT], input.uv);
            return direct + sampleConst(textures, samplers, input.uv) + sampleShadowed(textures, samplers, input.uv);
        }
    }
}
"""


SAMPLED_TEXTURE_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER = """
shader TransitiveSampledShadowedConstIndexValidation {
    const int COUNT = 4;
    sampler2D textures[4];
    sampler samplers[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
    };

    vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
        int COUNT = 0;
        return texture(textures[COUNT], samplers[COUNT], uv);
    }

    vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
        int COUNT = 0;
        vec4 sampled = texture(textures[COUNT], samplers[COUNT], uv);
        return sampled + leaf(textures, samplers, uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return passThrough(textures, samplers, input.uv);
        }
    }
}
"""


SHADOW_SAMPLER_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER = """
shader TransitiveShadowSamplerShadowedConstIndexValidation {
    const int COUNT = 4;
    sampler2DShadow shadowMaps[4];
    sampler shadowSamplers[4];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        float depth;
    };

    float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
        int COUNT = 0;
        return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
    }

    float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
        int COUNT = 0;
        float sampled = textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        return sampled + leaf(shadowMaps, shadowSamplers, uv, depth);
    }

    fragment {
        float main(FSInput input) @ gl_FragDepth {
            return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
        }
    }
}
"""


ARRAY_SHADOW_TEXTURE_RESOURCE_ARRAY_FRAGMENT_SHADER = """
shader ArrayShadowTextureResourceArrayValidation {
    sampler2DArrayShadow shadowArrays[4];
    samplerCubeArrayShadow cubeShadowArrays[4];
    sampler shadowSamplers[4];

    struct FSInput {
        vec3 uvLayer @ TEXCOORD0;
        vec4 cubeLayer @ TEXCOORD1;
        float depth;
    };

    float sampleArrayLayer(sampler2DArrayShadow shadowArrays[], sampler shadowSamplers[], vec3 uvLayer, float depth) {
        return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
    }

    float sampleCubeLayer(samplerCubeArrayShadow cubeShadowArrays[], sampler shadowSamplers[], vec4 cubeLayer, float depth) {
        return textureCompare(cubeShadowArrays[3], shadowSamplers[3], cubeLayer, depth);
    }

    fragment {
        float main(FSInput input) @ gl_FragDepth {
            return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth) + sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth);
        }
    }
}
"""


ARRAY_SHADOW_TEXTURE_QUERY_FRAGMENT_SHADER = """
shader ArrayShadowTextureQueryValidation {
    sampler2DArrayShadow shadowArray;
    samplerCubeArrayShadow cubeShadowArray;
    sampler2DArrayShadow shadowArrays[4];
    samplerCubeArrayShadow cubeShadowArrays[4];

    ivec3 query2DArrayShadow(sampler2DArrayShadow tex) {
        ivec3 size = textureSize(tex, 1);
        int levels = textureQueryLevels(tex);
        return size + ivec3(levels);
    }

    ivec3 queryCubeArrayShadow(samplerCubeArrayShadow tex) {
        ivec3 size = textureSize(tex, 0);
        int levels = textureQueryLevels(tex);
        return size + ivec3(levels);
    }

    ivec3 queryArrayElements(sampler2DArrayShadow shadowArrays[], samplerCubeArrayShadow cubeShadowArrays[]) {
        ivec3 arraySize = textureSize(shadowArrays[2], 1);
        ivec3 cubeSize = textureSize(cubeShadowArrays[3], 0);
        int arrayLevels = textureQueryLevels(shadowArrays[2]);
        int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);
        return arraySize + cubeSize + ivec3(arrayLevels + cubeLevels);
    }

    fragment {
        vec4 main() @ gl_FragColor {
            ivec3 a = query2DArrayShadow(shadowArray);
            ivec3 b = queryCubeArrayShadow(cubeShadowArray);
            ivec3 c = queryArrayElements(shadowArrays, cubeShadowArrays);
            return vec4(float(a.x + b.y + c.z));
        }
    }
}
"""


ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER = """
shader ArrayTextureQueryLodValidation {
    sampler2DArray layerMap;
    samplerCubeArray cubeArray;
    sampler2DArray layerMaps[4];
    samplerCubeArray cubeArrays[4];
    sampler linearSampler;
    sampler linearSamplers[4];

    struct FSInput {
        vec3 uvLayer @ TEXCOORD0;
        vec4 cubeLayer @ TEXCOORD1;
    };

    vec2 queryArrayLod(sampler2DArray tex, sampler s, vec3 uvLayer) {
        return textureQueryLod(tex, s, uvLayer);
    }

    vec2 queryCubeArrayLod(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
        return textureQueryLod(tex, s, cubeLayer);
    }

    vec2 queryArrayElementLod(sampler2DArray layerMaps[], sampler linearSamplers[], vec3 uvLayer) {
        return textureQueryLod(layerMaps[2], linearSamplers[2], uvLayer);
    }

    vec2 queryCubeArrayElementLod(samplerCubeArray cubeArrays[], sampler linearSamplers[], vec4 cubeLayer) {
        return textureQueryLod(cubeArrays[3], linearSamplers[3], cubeLayer);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            vec2 a = queryArrayLod(layerMap, linearSampler, input.uvLayer);
            vec2 b = queryCubeArrayLod(cubeArray, linearSampler, input.cubeLayer);
            vec2 c = queryArrayElementLod(layerMaps, linearSamplers, input.uvLayer);
            vec2 d = queryCubeArrayElementLod(cubeArrays, linearSamplers, input.cubeLayer);
            return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
        }
    }
}
"""


SHADOW_ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER = """
shader ShadowArrayTextureQueryLodValidation {
    sampler2DArrayShadow shadowArray;
    samplerCubeArrayShadow cubeShadowArray;
    sampler2DArrayShadow shadowArrays[4];
    samplerCubeArrayShadow cubeShadowArrays[4];
    sampler linearSampler;
    sampler linearSamplers[4];

    struct FSInput {
        vec3 uvLayer @ TEXCOORD0;
        vec4 cubeLayer @ TEXCOORD1;
    };

    vec2 queryArrayLod(sampler2DArrayShadow tex, sampler s, vec3 uvLayer) {
        return textureQueryLod(tex, s, uvLayer);
    }

    vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer) {
        return textureQueryLod(tex, s, cubeLayer);
    }

    vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], sampler linearSamplers[], vec3 uvLayer) {
        return textureQueryLod(shadowArrays[2], linearSamplers[2], uvLayer);
    }

    vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], sampler linearSamplers[], vec4 cubeLayer) {
        return textureQueryLod(cubeShadowArrays[3], linearSamplers[3], cubeLayer);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            vec2 a = queryArrayLod(shadowArray, linearSampler, input.uvLayer);
            vec2 b = queryCubeArrayLod(cubeShadowArray, linearSampler, input.cubeLayer);
            vec2 c = queryArrayElementLod(shadowArrays, linearSamplers, input.uvLayer);
            vec2 d = queryCubeArrayElementLod(cubeShadowArrays, linearSamplers, input.cubeLayer);
            return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
        }
    }
}
"""


IMPLICIT_SHADOW_ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER = """
shader ImplicitShadowArrayTextureQueryLodValidation {
    sampler2DArrayShadow shadowArray;
    samplerCubeArrayShadow cubeShadowArray;
    sampler2DArrayShadow shadowArrays[4];
    samplerCubeArrayShadow cubeShadowArrays[4];

    struct FSInput {
        vec3 uvLayer @ TEXCOORD0;
        vec4 cubeLayer @ TEXCOORD1;
    };

    vec2 queryArrayLod(sampler2DArrayShadow tex, vec3 uvLayer) {
        return textureQueryLod(tex, uvLayer);
    }

    vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, vec4 cubeLayer) {
        return textureQueryLod(tex, cubeLayer);
    }

    vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], vec3 uvLayer) {
        return textureQueryLod(shadowArrays[2], uvLayer);
    }

    vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer) {
        return textureQueryLod(cubeShadowArrays[3], cubeLayer);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            vec2 a = textureQueryLod(shadowArray, input.uvLayer);
            vec2 b = textureQueryLod(cubeShadowArray, input.cubeLayer);
            vec2 c = queryArrayLod(shadowArray, input.uvLayer);
            vec2 d = queryCubeArrayLod(cubeShadowArray, input.cubeLayer);
            vec2 e = queryArrayElementLod(shadowArrays, input.uvLayer);
            vec2 f = queryCubeArrayElementLod(cubeShadowArrays, input.cubeLayer);
            return vec4(a.x + b.y, c.x + d.y, e.x + f.y, 1.0);
        }
    }
}
"""


CUBE_ARRAY_TEXTURE_GRAD_GATHER_FRAGMENT_SHADER = """
shader CubeArrayGradGatherValidation {
    samplerCubeArray cubeArray;
    samplerCubeArray cubeArrays[4];
    sampler cubeSampler;
    sampler cubeSamplers[4];

    struct FSInput {
        vec4 cubeLayer @ TEXCOORD0;
        vec3 ddx @ TEXCOORD1;
        vec3 ddy @ TEXCOORD2;
    };

    vec4 sampleCubeArrayOps(samplerCubeArray tex, sampler s, vec4 cubeLayer, vec3 ddx, vec3 ddy) {
        vec4 gradColor = textureGrad(tex, s, cubeLayer, ddx, ddy);
        vec4 gathered = textureGather(tex, s, cubeLayer);
        return gradColor + gathered;
    }

    vec4 sampleCubeArrayElements(samplerCubeArray cubeArrays[], sampler cubeSamplers[], vec4 cubeLayer, vec3 ddx, vec3 ddy) {
        vec4 gradColor = textureGrad(cubeArrays[2], cubeSamplers[2], cubeLayer, ddx, ddy);
        vec4 gathered = textureGather(cubeArrays[3], cubeSamplers[3], cubeLayer);
        return gradColor + gathered;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)
                + sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy);
        }
    }
}
"""


TEXTURE_GRADIENT_FAMILY_FRAGMENT_SHADER = """
shader TextureGradientFamilyValidation {
    sampler2D colorMap;
    samplerCube cubeMap;
    sampler3D volumeMap;
    sampler colorSampler;
    sampler cubeSampler;
    sampler volumeSampler;

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec3 direction @ TEXCOORD1;
        vec3 volumeUv @ TEXCOORD2;
        vec2 ddx2 @ TEXCOORD3;
        vec2 ddy2 @ TEXCOORD4;
        vec3 ddx3 @ TEXCOORD5;
        vec3 ddy3 @ TEXCOORD6;
    };

    vec4 sample2DGrad(sampler2D tex, sampler s, vec2 uv, vec2 ddx, vec2 ddy) {
        return textureGrad(tex, s, uv, ddx, ddy);
    }

    vec4 sampleCubeGrad(samplerCube tex, sampler s, vec3 direction, vec3 ddx, vec3 ddy) {
        return textureGrad(tex, s, direction, ddx, ddy);
    }

    vec4 sampleVolumeGrad(sampler3D tex, sampler s, vec3 volumeUv, vec3 ddx, vec3 ddy) {
        return textureGrad(tex, s, volumeUv, ddx, ddy);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sample2DGrad(colorMap, colorSampler, input.uv, input.ddx2, input.ddy2)
                + sampleCubeGrad(cubeMap, cubeSampler, input.direction, input.ddx3, input.ddy3)
                + sampleVolumeGrad(volumeMap, volumeSampler, input.volumeUv, input.ddx3, input.ddy3);
        }
    }
}
"""


TEXTURE_GATHER_OFFSET_FRAGMENT_SHADER = """
shader TextureGatherOffsetValidation {
    sampler2D colorMap;
    sampler2DArray layerMap;
    sampler linearSampler;

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec3 uvLayer @ TEXCOORD1;
        ivec2 offset @ TEXCOORD2;
        ivec2 offset0 @ TEXCOORD3;
        ivec2 offset1 @ TEXCOORD4;
        ivec2 offset2 @ TEXCOORD5;
        ivec2 offset3 @ TEXCOORD6;
        int component @ TEXCOORD7;
    };

    vec4 gatherOps(
        sampler2D tex,
        sampler2DArray layers,
        sampler s,
        vec2 uv,
        vec3 uvLayer,
        ivec2 offset,
        ivec2 offset0,
        ivec2 offset1,
        ivec2 offset2,
        ivec2 offset3,
        int component
    ) {
        vec4 green = textureGather(tex, s, uv, 1);
        vec4 dynamic = textureGather(tex, s, uv, component);
        vec4 offsetGather = textureGatherOffset(tex, s, uv, offset, 3);
        vec4 dynamicOffset = textureGatherOffset(tex, s, uv, offset, component);
        vec4 offsetsGather = textureGatherOffsets(
            layers,
            s,
            uvLayer,
            offset0,
            offset1,
            offset2,
            offset3,
            component
        );
        return green + dynamic + offsetGather + dynamicOffset + offsetsGather;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return gatherOps(
                colorMap,
                layerMap,
                linearSampler,
                input.uv,
                input.uvLayer,
                input.offset,
                input.offset0,
                input.offset1,
                input.offset2,
                input.offset3,
                input.component
            );
        }
    }
}
"""


TEXTURE_SAMPLE_OFFSET_FRAGMENT_SHADER = """
shader TextureSampleOffsetValidation {
    sampler2D colorMap;
    sampler2DArray layerMap;
    sampler linearSampler;

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec3 uvLayer @ TEXCOORD1;
        float lod;
        vec2 ddx @ TEXCOORD2;
        vec2 ddy @ TEXCOORD3;
    };

    vec4 sampleOffsets(
        sampler2D tex,
        sampler2DArray layers,
        sampler s,
        vec2 uv,
        vec3 uvLayer,
        float lod,
        vec2 ddx,
        vec2 ddy
    ) {
        vec4 plain = textureOffset(tex, s, uv, ivec2(1, 0));
        vec4 lodSample = textureLodOffset(tex, s, uv, lod, ivec2(1, 0));
        vec4 gradSample = textureGradOffset(tex, s, uv, ddx, ddy, ivec2(1, 0));
        vec4 arrayPlain = textureOffset(layers, s, uvLayer, ivec2(1, 0));
        vec4 arrayLod = textureLodOffset(layers, s, uvLayer, lod, ivec2(1, 0));
        vec4 arrayGrad = textureGradOffset(layers, s, uvLayer, ddx, ddy, ivec2(1, 0));
        return plain + lodSample + gradSample + arrayPlain + arrayLod + arrayGrad;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sampleOffsets(
                colorMap,
                layerMap,
                linearSampler,
                input.uv,
                input.uvLayer,
                input.lod,
                input.ddx,
                input.ddy
            );
        }
    }
}
"""


TEXTURE_3D_SAMPLE_OFFSET_FRAGMENT_SHADER = """
shader Texture3DSampleOffsetValidation {
    sampler3D volumeMap;
    sampler linearSampler;

    struct FSInput {
        vec3 uvw @ TEXCOORD0;
        float lod;
        vec3 ddx @ TEXCOORD1;
        vec3 ddy @ TEXCOORD2;
    };

    vec4 sampleVolumeOffsets(
        sampler3D volume,
        sampler s,
        vec3 uvw,
        float lod,
        vec3 ddx,
        vec3 ddy
    ) {
        vec4 plain = textureOffset(volume, s, uvw, ivec3(1, 0, -1));
        vec4 biased = textureOffset(volume, s, uvw, ivec3(1, 0, -1), 0.5);
        vec4 lodSample = textureLodOffset(volume, s, uvw, lod, ivec3(1, 0, -1));
        vec4 gradSample = textureGradOffset(
            volume,
            s,
            uvw,
            ddx,
            ddy,
            ivec3(1, 0, -1)
        );
        return plain + biased + lodSample + gradSample;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sampleVolumeOffsets(
                volumeMap,
                linearSampler,
                input.uvw,
                input.lod,
                input.ddx,
                input.ddy
            );
        }
    }
}
"""


TEXTURE_PROJECTED_FRAGMENT_SHADER = """
shader TextureProjectionValidation {
    sampler2D colorMap;
    sampler3D volumeMap;
    sampler linearSampler;

    struct FSInput {
        vec3 uvq @ TEXCOORD0;
        vec4 uvqw @ TEXCOORD1;
        vec4 xyzq @ TEXCOORD2;
        vec2 ddx @ TEXCOORD3;
        vec2 ddy @ TEXCOORD4;
    };

    vec4 projectedOps(
        sampler2D tex,
        sampler3D volume,
        sampler s,
        vec3 uvq,
        vec4 uvqw,
        vec4 xyzq,
        vec2 ddx,
        vec2 ddy
    ) {
        vec4 projected = textureProj(tex, s, uvq);
        vec4 projectedBias = textureProj(tex, s, uvqw, 0.25);
        vec4 volumeProjected = textureProj(volume, s, xyzq);
        vec4 projectedOffset = textureProjOffset(tex, s, uvq, ivec2(1, 0));
        vec4 projectedOffsetBias = textureProjOffset(tex, s, uvq, ivec2(1, 0), 0.5);
        vec4 projectedLod = textureProjLod(tex, s, uvq, 2.0);
        vec4 projectedLodOffset = textureProjLodOffset(tex, s, uvq, 3.0, ivec2(1, 0));
        vec4 projectedGrad = textureProjGrad(tex, s, uvq, ddx, ddy);
        vec4 projectedGradOffset = textureProjGradOffset(tex, s, uvq, ddx, ddy, ivec2(1, 0));
        return projected
            + projectedBias
            + volumeProjected
            + projectedOffset
            + projectedOffsetBias
            + projectedLod
            + projectedLodOffset
            + projectedGrad
            + projectedGradOffset;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return projectedOps(
                colorMap,
                volumeMap,
                linearSampler,
                input.uvq,
                input.uvqw,
                input.xyzq,
                input.ddx,
                input.ddy
            );
        }
    }
}
"""


VERTEX_STRUCT_IO_SHADER = """
shader VertexStructIOValidation {
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
}
"""


COMBINED_STAGE_IO_SHADER = """
shader CombinedStageIOValidation {
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


COMPUTE_STAGE_SHADER = """
shader ComputeStageValidation {
    compute {
        void main() {
            int value = 1;
            uint unsignedValue = 7u;
        }
    }
}
"""


COMPUTE_DO_WHILE_SHADER = """
shader ComputeDoWhileValidation {
    compute {
        void main() {
            int value = 0;
            do {
                value = value + 1;
            } while (value < 3);
        }
    }
}
"""


METAL_COMPUTE_BUILTINS_SHADER = """
shader MetalComputeBuiltinsValidation {
    compute {
        void main() {
            uint gx = gl_GlobalInvocationID.x;
            uint lx = gl_LocalInvocationID.x;
            uint group = gl_WorkGroupID.x;
            uint index = gl_LocalInvocationIndex;
            uint size = gl_WorkGroupSize.x;
            uint groups = gl_NumWorkGroups.x;
        }
    }
}
"""


METAL_THREADGROUP_HELPER_BARRIERS_SHADER = """
shader MetalThreadgroupHelperBarrierValidation {
    void writeScratch(threadgroup float scratch[64], uint index, float value) {
        scratch[index] = value;
    }

    float readScratch(threadgroup float scratch[64], uint index) {
        return scratch[index];
    }

    compute {
        void main() {
            shared float scratch[64];
            uint index = gl_LocalInvocationIndex;
            writeScratch(scratch, index, float(index));
            barrier();
            workgroupBarrier();
            groupMemoryBarrier();
            memoryBarrierShared();
            memoryBarrierBuffer();
            deviceMemoryBarrier();
            memoryBarrierImage();
            allMemoryBarrier();
            float value = readScratch(scratch, index);
            scratch[index] = value;
        }
    }
}
"""


METAL_THREADGROUP_ATOMIC_BARRIERS_SHADER = """
shader MetalThreadgroupAtomicBarrierValidation {
    uint bump(threadgroup atomic_uint counters[64], uint index) {
        return atomic_fetch_add_explicit(
            counters[index],
            1u,
            memory_order_relaxed
        );
    }

    compute {
        void main() {
            shared atomic_uint counters[64];
            uint index = gl_LocalInvocationIndex;
            atomic_store_explicit(counters[index], 0u, memory_order_relaxed);
            barrier();
            uint oldValue = bump(counters, index);
            groupMemoryBarrier();
            uint currentValue = atomic_load_explicit(
                counters[index],
                memory_order_relaxed
            );
            atomic_exchange_explicit(
                counters[index],
                oldValue + currentValue,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_ATOMIC_ARRAY_INITIALIZER_SHADER = """
shader MetalAtomicArrayInitializerValidation {
    compute {
        void main() {
            uint index = gl_LocalInvocationIndex;
            shared atomic_uint counters[4] = {0u, 1u, uint(index), 3u};
            atomic_int signedCounters[2] = {-1, 2};
            uint oldValue = atomic_fetch_add_explicit(
                counters[index],
                1u,
                memory_order_relaxed
            );
            int signedValue = atomic_load_explicit(
                signedCounters[index % 2u],
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_SYMBOLIC_ATOMIC_ARRAY_INITIALIZER_SHADER = """
shader MetalSymbolicAtomicArrayInitializerValidation {
    const int COUNT = 4;
    const int EXTRA = COUNT + 1;

    compute {
        void main() {
            uint index = gl_LocalInvocationIndex;
            shared atomic_uint counters[COUNT] = {0u, 1u};
            shared atomic_uint expressionCounters[COUNT + 1] = {2u};
            atomic_int signedCounters[EXTRA] = {-1, 2};
            uint oldValue = atomic_fetch_add_explicit(
                counters[index],
                1u,
                memory_order_relaxed
            );
            int signedValue = atomic_load_explicit(
                signedCounters[index % uint(EXTRA)],
                memory_order_relaxed
            );
            uint expressionValue = atomic_load_explicit(
                expressionCounters[index % uint(COUNT + 1)],
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_SCOPED_ATOMIC_SHADER = """
shader MetalScopedAtomicValidation {
    bool claim(
        threadgroup atomic_uint counters[4],
        thread uint expectedValues[4],
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_strong_explicit(
            counters[index],
            expectedValues[index],
            desired,
            memory_order_relaxed,
            memory_order_relaxed,
            memory_scope_device
        );
    }

    compute {
        void main(
            device atomic_uint* deviceCounters @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_uint counters[4];
            uint expectedValues[4];
            expectedValues[index] = 0u;
            atomic_store_explicit(
                counters[index],
                0u,
                memory_order_relaxed,
                memory_scope_workgroup
            );
            uint oldValue = atomic_fetch_add_explicit(
                counters[index],
                1u,
                memory_order_relaxed,
                memory_scope_workgroup
            );
            uint loaded = atomic_load_explicit(
                counters[index],
                memory_order_relaxed,
                memory_scope_workgroup
            );
            uint exchanged = atomic_exchange_explicit(
                deviceCounters[index],
                loaded + oldValue,
                memory_order_relaxed,
                memory_scope_device
            );
            bool claimed = claim(counters, expectedValues, index, exchanged);
        }
    }
}
"""


METAL_ATOMIC_POINTER_TARGETS_SHADER = """
shader MetalAtomicPointerTargetValidation {
    int bumpDevice(device atomic_int* counters, uint index, int delta) {
        return atomic_fetch_add_explicit(
            counters + index,
            delta,
            memory_order_relaxed
        );
    }

    int reduceThreadgroup(
        threadgroup atomic_int* counters,
        uint index,
        int value
    ) {
        return atomic_fetch_min_explicit(
            counters[index],
            value,
            memory_order_relaxed
        );
    }

    bool exchangeFlag(device atomic_bool* flags, uint index, bool value) {
        return atomic_exchange_explicit(
            flags + index,
            value,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_int* counters @buffer(0),
            device atomic_bool* flags @buffer(1),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_int scratch[64];
            atomic_store_explicit(scratch[index], 7, memory_order_relaxed);
            atomic_store_explicit(counters[index], 0, memory_order_relaxed);
            int oldDevice = bumpDevice(counters, index, 1);
            int oldScratch = reduceThreadgroup(scratch, index, oldDevice);
            int loaded = atomic_load_explicit(
                counters + index,
                memory_order_relaxed
            );
            bool wasSet = exchangeFlag(flags, index, true);
            bool isSet = atomic_load_explicit(
                flags + index,
                memory_order_relaxed
            );
            atomic_store_explicit(
                flags[index],
                wasSet && isSet,
                memory_order_relaxed
            );
            atomic_store_explicit(
                counters[index],
                oldScratch + loaded,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_THREADGROUP_ATOMIC_POINTER_ALIAS_SHADER = """
shader MetalThreadgroupAtomicAliasValidation {
    int bumpThreadgroup(threadgroup atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    int bumpDevice(device atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    compute {
        void main(uint index @gl_LocalInvocationIndex) {
            shared atomic_int scratch[64];
            atomic_int* alias = scratch + index;
            int oldValue = bumpThreadgroup(alias, 1);
            int nextValue = atomic_fetch_add_explicit(
                alias + 1,
                oldValue,
                memory_order_relaxed
            );
            int rejected = bumpDevice(alias + 2, nextValue);
            atomic_store_explicit(alias, rejected, memory_order_relaxed);
        }
    }
}
"""


METAL_THREADGROUP_ATOMIC_TERNARY_ALIAS_SHADER = """
shader MetalThreadgroupAtomicTernaryAliasValidation {
    int bumpThreadgroup(threadgroup atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_int* counters @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_int scratchA[64];
            shared atomic_int scratchB[64];
            bool useA = index == 0u;
            atomic_int* alias = useA ? scratchA + index : scratchB + index;
            int oldValue = bumpThreadgroup(alias, 1);
            atomic_store_explicit(alias, oldValue, memory_order_relaxed);

            bool useShared = oldValue == 0;
            atomic_int* mixedAlias = useShared ? scratchA + index : counters + index;
            int rejected = bumpThreadgroup(mixedAlias, oldValue);
            int directRejected = bumpThreadgroup(
                useShared ? scratchA + index : counters + index,
                rejected
            );
            atomic_store_explicit(counters + index, rejected, memory_order_relaxed);
            atomic_store_explicit(
                counters + index,
                directRejected,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_POINTER_ASSIGNMENT_ADDRESS_SPACE_SHADER = """
shader MetalPointerAssignmentAddressSpaceValidation {
    int bumpThreadgroup(threadgroup atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_int* counters @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_int scratchA[64];
            shared atomic_int scratchB[64];
            bool useA = index == 0u;
            atomic_int* alias = scratchA + index;
            alias = useA ? scratchA + index : scratchB + index;
            int first = bumpThreadgroup(alias, 1);
            bool useShared = first == 0;
            alias = useShared ? scratchA + index : counters + index;
            alias = counters + index;
            int second = bumpThreadgroup(alias, first);
            atomic_store_explicit(alias, second, memory_order_relaxed);
        }
    }
}
"""


METAL_THREADGROUP_REFERENCE_TERNARY_ALIAS_SHADER = """
shader MetalThreadgroupReferenceTernaryAliasValidation {
    struct Payload {
        float value;
    };

    void useThreadgroup(threadgroup Payload& payload, float delta) {
        payload.value = payload.value + delta;
    }

    compute {
        void main(
            device Payload* payloads @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            threadgroup Payload scratchA[64];
            threadgroup Payload scratchB[64];
            bool useA = index == 0u;
            Payload& alias = useA ? scratchA[index] : scratchB[index];
            useThreadgroup(alias, 1.0);

            bool useShared = alias.value == 0.0;
            Payload& mixedAlias = useShared ? scratchA[index] : payloads[index];
            useThreadgroup(mixedAlias, 2.0);
            useThreadgroup(
                useShared ? scratchA[index] : payloads[index],
                3.0
            );
        }
    }
}
"""


METAL_POINTER_MEMBER_ATOMIC_ADDRESS_SPACE_SHADER = """
shader MetalPointerMemberAddressSpaceValidation {
    struct PointerBank {
        device atomic_int* deviceCounters;
        threadgroup atomic_int* sharedCounters;
    };

    int bumpDevice(device atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    int bumpThreadgroup(threadgroup atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_int* counters @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_int scratch[64];
            PointerBank bank;
            bank.deviceCounters = counters;
            bank.sharedCounters = scratch;
            int deviceOld = bumpDevice(bank.deviceCounters + index, 1);
            int rejectedThread = bumpThreadgroup(
                bank.deviceCounters + index,
                deviceOld
            );
            int sharedOld = bumpThreadgroup(
                bank.sharedCounters + index,
                rejectedThread
            );
            atomic_store_explicit(
                bank.deviceCounters[index],
                sharedOld,
                memory_order_relaxed
            );
            atomic_store_explicit(
                bank.sharedCounters[index],
                deviceOld,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_NESTED_POINTER_MEMBER_ATOMIC_ALIAS_SHADER = """
shader MetalNestedPointerMemberAtomicValidation {
    struct InnerBank {
        threadgroup atomic_int* sharedCounters;
        device atomic_int* deviceCounters;
    };

    struct OuterBank {
        InnerBank inner;
    };

    int bumpThreadgroup(threadgroup atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    int bumpDevice(device atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_int* counters @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_int scratch[64];
            OuterBank bank;
            bank.inner.sharedCounters = scratch;
            bank.inner.deviceCounters = counters;
            OuterBank* bankPtr = &bank;
            atomic_int* sharedAlias = bankPtr->inner.sharedCounters + index;
            atomic_int* deviceAlias = bankPtr->inner.deviceCounters + index;
            int sharedOld = bumpThreadgroup(sharedAlias, 1);
            int rejectedShared = bumpDevice(sharedAlias + 1, sharedOld);
            int deviceOld = bumpDevice(deviceAlias, rejectedShared);
            int rejectedDevice = bumpThreadgroup(deviceAlias + 1, deviceOld);
            atomic_store_explicit(
                sharedAlias,
                rejectedDevice,
                memory_order_relaxed
            );
            atomic_store_explicit(
                deviceAlias,
                sharedOld + deviceOld,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_REFERENCE_MEMBER_ATOMIC_ADDRESS_SPACE_SHADER = """
shader MetalReferenceMemberAddressSpaceValidation {
    struct Bank {
        threadgroup atomic_int* sharedCounters;
        device atomic_int* deviceCounters;
    };

    int bumpThreadgroup(threadgroup atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    int bumpDevice(device atomic_int* counters, int delta) {
        return atomic_fetch_add_explicit(
            counters,
            delta,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_int* counters @buffer(0),
            uint index @gl_LocalInvocationIndex
        ) {
            shared atomic_int scratch[64];
            Bank bank;
            bank.sharedCounters = scratch;
            bank.deviceCounters = counters;
            Bank& ref = bank;
            int sharedOld = bumpThreadgroup(ref.sharedCounters + index, 1);
            int deviceOld = bumpDevice(ref.deviceCounters + index, sharedOld);
            int rejected = bumpDevice(ref.sharedCounters + index, deviceOld);
        }
    }
}
"""


METAL_ATOMIC_COMPARE_EXCHANGE_SHADER = """
shader MetalAtomicCompareExchangeValidation {
    bool claim(
        threadgroup atomic_uint counters[64],
        thread uint expectedValues[64],
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_weak_explicit(
            counters[index],
            expectedValues[index],
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    bool claimPtr(
        threadgroup atomic_uint* counters,
        thread uint* expectedValues,
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_strong_explicit(
            counters[index],
            expectedValues[index],
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    bool claimRaw(
        threadgroup atomic_uint* counter,
        thread uint* expected,
        uint desired
    ) {
        return atomic_compare_exchange_weak_explicit(
            counter,
            expected,
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    compute {
        void main() {
            shared atomic_uint counters[64];
            uint expectedValues[64];
            uint index = gl_LocalInvocationIndex;
            expectedValues[index] = 0u;
            atomic_store_explicit(counters[index], 0u, memory_order_relaxed);
            bool helperClaimed = claim(counters, expectedValues, index, 1u);
            bool pointerClaimed = claimPtr(counters, expectedValues, index, 2u);
            uint expected = helperClaimed ? 1u : 0u;
            bool claimed = atomic_compare_exchange_strong_explicit(
                counters[index],
                expected,
                2u,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_DEVICE_ATOMIC_COMPARE_EXCHANGE_SHADER = """
shader MetalDeviceAtomicCompareExchangeValidation {
    bool claimDevice(
        device atomic_uint* counters,
        thread uint* expectedValues,
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_strong_explicit(
            counters + index,
            expectedValues + index,
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    bool rejectDeviceExpected(
        device atomic_uint* counters,
        device uint* expectedValues,
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_weak_explicit(
            counters + index,
            expectedValues + index,
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_uint* counters @buffer(0),
            device uint* deviceExpected @buffer(1),
            uint index @gl_LocalInvocationIndex
        ) {
            uint expectedValues[64];
            expectedValues[index] = 0u;
            bool pointerClaimed = claimDevice(
                counters,
                expectedValues,
                index,
                1u
            );
            bool rejected = rejectDeviceExpected(
                counters,
                deviceExpected,
                index,
                2u
            );
            uint expected = pointerClaimed ? 1u : 0u;
            bool directClaimed = atomic_compare_exchange_strong_explicit(
                counters[index],
                expected,
                3u,
                memory_order_relaxed,
                memory_order_relaxed
            );
        }
    }
}
"""


METAL_STRUCT_MEMBER_ATOMIC_COMPARE_EXCHANGE_SHADER = """
shader MetalCompareExchangeMemberExpectedValidation {
    struct ExpectedBank {
        thread uint* threadExpected;
        device uint* deviceExpected;
    };

    bool claimDevice(
        device atomic_uint* counters,
        thread uint* expectedValues,
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_strong_explicit(
            counters + index,
            expectedValues + index,
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    bool rejectDeviceExpected(
        device atomic_uint* counters,
        device uint* expectedValues,
        uint index,
        uint desired
    ) {
        return atomic_compare_exchange_weak_explicit(
            counters + index,
            expectedValues + index,
            desired,
            memory_order_relaxed,
            memory_order_relaxed
        );
    }

    compute {
        void main(
            device atomic_uint* counters @buffer(0),
            device uint* deviceExpected @buffer(1),
            uint index @gl_LocalInvocationIndex
        ) {
            uint expectedValues[64];
            ExpectedBank bank;
            bank.threadExpected = expectedValues;
            bank.deviceExpected = deviceExpected;
            bank.threadExpected[index] = 0u;
            bool direct = atomic_compare_exchange_strong_explicit(
                counters[index],
                bank.threadExpected[index],
                1u,
                memory_order_relaxed,
                memory_order_relaxed
            );
            bool pointerDirect = atomic_compare_exchange_weak_explicit(
                counters + index,
                bank.threadExpected + index,
                2u,
                memory_order_relaxed,
                memory_order_relaxed
            );
            bool helper = claimDevice(
                counters,
                bank.threadExpected,
                index,
                3u
            );
            bool rejectedDirect = atomic_compare_exchange_weak_explicit(
                counters + index,
                bank.deviceExpected + index,
                4u,
                memory_order_relaxed,
                memory_order_relaxed
            );
            bool rejectedHelper = rejectDeviceExpected(
                counters,
                bank.deviceExpected,
                index,
                5u
            );
        }
    }
}
"""


METAL_BUFFER_BLOCK_ATOMIC_COMPARE_SHADER = """
shader MetalBufferBlockAtomicCompareValidation {
    struct AtomicBlock {
        uint counter;
        uint bins[];
    };

    struct SignedAtomicBlock {
        int counter;
        int bins[];
    };

    AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);
    SignedAtomicBlock signedAtomicBlock @glsl_buffer_block(std430) @binding(18);

    uint swapUnsigned(
        AtomicBlock block @glsl_buffer_block(std430),
        uint index,
        uint compareValue,
        uint value
    ) {
        return atomicCompSwap(block.bins[index], compareValue, value);
    }

    int swapSigned(
        SignedAtomicBlock block @glsl_buffer_block(std430),
        uint index,
        int compareValue,
        int value
    ) {
        return atomicCompSwap(block.bins[index], compareValue, value);
    }

    compute {
        void main() {
            uint resultU = swapUnsigned(
                atomicBlock,
                1u,
                atomicBlock.counter,
                9u
            );
            int resultS = swapSigned(
                signedAtomicBlock,
                1u,
                signedAtomicBlock.counter,
                -3
            );
            uint oldCounter = atomicCompSwap(atomicBlock.counter, 0u, resultU);
            int oldSignedCounter = atomicCompSwap(
                signedAtomicBlock.counter,
                0,
                resultS
            );
        }
    }
}
"""


METAL_ADDRESS_SPACE_PARAMETER_SHADER = """
shader MetalAddressSpaceParameterValidation {
    struct Payload {
        float value;
    };

    void update(threadgroup Payload& scratch, device float values[], constant uint& count) {
        if (count > 0u) {
            scratch.value = values[0] + 1.0;
            values[0] = scratch.value;
        }
    }

    compute {
        void main(device float values[] @buffer(0), constant uint& count @buffer(1)) {
            threadgroup Payload scratch;
            update(scratch, values, count);
        }
    }
}
"""


METAL_POINTER_MEMBER_ACCESS_SHADER = """
shader MetalPointerMemberAccessValidation {
    struct Payload {
        float value;
    };

    compute {
        void main(device Payload* payload @buffer(0), device float values[] @buffer(1)) {
            payload.value = values[0];
            float value = payload.value;
        }
    }
}
"""


METAL_INDEXED_POINTER_MEMBER_ACCESS_SHADER = """
shader MetalIndexedPointerMemberAccessValidation {
    struct Payload {
        float value;
    };

    compute {
        void main(device Payload* payloads @buffer(0), device float values[] @buffer(1)) {
            payloads[0].value = values[0];
            float value = payloads[0].value;
        }
    }
}
"""


METAL_READONLY_RAW_BUFFER_SHADER = """
shader MetalReadonlyRawBufferValidation {
    struct Payload {
        float value;
    };

    compute {
        void main(
            readonly device Payload* payload @buffer(0),
            readonly device float values[] @buffer(1),
            constant uint& count @buffer(2)
        ) {
            float value = payload.value + values[count];
        }
    }
}
"""


METAL_READONLY_RAW_BUFFER_DIAGNOSTIC_SHADER = """
shader MetalReadonlyRawBufferDiagnosticValidation {
    struct Payload {
        float value;
    };

    compute {
        void main(
            readonly device Payload* payload @buffer(0),
            readonly device float values[] @buffer(1)
        ) {
            payload.value = 1.0;
            values[0] = 2.0;
        }
    }
}
"""


METAL_READONLY_RAW_BUFFER_HELPER_SHADER = """
shader MetalReadonlyRawBufferHelperValidation {
    struct Payload {
        float value;
    };

    float readPayload(
        readonly device Payload* payload,
        readonly device float values[],
        constant uint& index
    ) {
        return payload.value + values[index];
    }

    void rejectWrite(readonly device Payload* payload, readonly device float values[]) {
        payload.value = 1.0;
        values[0] = 2.0;
    }

    compute {
        void main(
            readonly device Payload* payload @buffer(0),
            readonly device float values[] @buffer(1),
            constant uint& index @buffer(2)
        ) {
            float value = readPayload(payload, values, index);
            rejectWrite(payload, values);
        }
    }
}
"""


METAL_READONLY_RAW_BUFFER_MUTABLE_HELPER_CALL_SHADER = """
shader MetalReadonlyRawBufferMutableHelperCallValidation {
    struct Payload {
        float value;
    };

    void mutate(device Payload* payload, device float values[]) {
        payload.value = values[0];
    }

    compute {
        void main(
            readonly device Payload* payload @buffer(0),
            readonly device float values[] @buffer(1)
        ) {
            mutate(payload, values);
        }
    }
}
"""


METAL_CONST_REFERENCE_HELPER_SHADER = """
shader MetalConstReferenceHelperValidation {
    struct Payload {
        float value;
    };

    float readPayload(const Payload& payload) {
        return payload.value;
    }

    void mutate(Payload& payload) {
        payload.value = 2.0;
    }

    void rejectWrite(const Payload& payload) {
        payload.value = 1.0;
        mutate(payload);
    }

    compute {
        void main() {
            Payload payload;
            float value = readPayload(payload);
            rejectWrite(payload);
        }
    }
}
"""


METAL_CONST_POINTER_ARRAY_HELPER_SHADER = """
shader MetalConstPointerArrayHelperValidation {
    struct Payload {
        float value;
    };

    float readPointer(const Payload* payload) {
        return payload->value;
    }

    void mutatePointer(Payload* payload) {
        payload->value = 2.0;
    }

    void rejectPointerWrite(const Payload* payload) {
        payload->value = 1.0;
        mutatePointer(payload);
    }

    float readArray(const float values[], int index) {
        return values[index];
    }

    void mutateArray(float values[]) {
        values[0] = 3.0;
    }

    void rejectArrayWrite(const float values[]) {
        values[0] = 4.0;
        mutateArray(values);
    }

    compute {
        void main() {
            Payload payload;
            float values[2];
            values[0] = 1.0;
            values[1] = 2.0;
            float a = readPointer(&payload);
            rejectPointerWrite(&payload);
            float b = readArray(values, 1);
            rejectArrayWrite(values);
        }
    }
}
"""


METAL_CONST_THREADGROUP_POINTER_ALIAS_SHADER = """
shader MetalConstThreadgroupPointerAliasValidation {
    struct Payload {
        float value;
    };

    void mutate(threadgroup Payload* payload) {
        payload.value = 2.0;
    }

    float readValue(const threadgroup Payload* payload) {
        return payload.value;
    }

    compute {
        void main() {
            threadgroup Payload scratch;
            const Payload* alias = scratch;
            float value = readValue(alias);
            alias.value = value;
            mutate(alias);
        }
    }
}
"""


METAL_READONLY_DEVICE_POINTER_ALIAS_SHADER = """
shader MetalReadonlyDevicePointerAliasValidation {
    struct Payload {
        float value;
    };

    void mutate(device Payload* payload) {
        payload.value = 2.0;
    }

    float readValue(readonly device Payload* payload) {
        return payload.value;
    }

    compute {
        void main(device Payload* payload @buffer(0)) {
            readonly device Payload* alias = payload;
            float value = readValue(alias);
            alias.value = value;
            mutate(alias);
        }
    }
}
"""


METAL_CONST_LOCAL_ARRAY_ALIAS_SHADER = """
shader MetalConstLocalArrayAliasValidation {
    float readValue(const float values[2], int index) {
        return values[index];
    }

    void mutate(float values[2]) {
        values[0] = 2.0;
    }

    compute {
        void main() {
            const float values[2] = {1.0, 2.0};
            float value = readValue(values, 1);
            values[0] = value;
            mutate(values);
        }
    }
}
"""


METAL_CONSTANT_POINTER_REFERENCE_ALIAS_SHADER = """
shader MetalConstantPointerReferenceAliasValidation {
    struct Payload {
        float value;
    };

    void mutatePointer(device Payload* payload) {
        payload.value = 2.0;
    }

    void mutateReference(device Payload& payload) {
        payload.value = 3.0;
    }

    float readPointer(constant Payload* payload) {
        return payload.value;
    }

    float readReference(constant Payload& payload) {
        return payload.value;
    }

    compute {
        void main(
            constant Payload* pointerPayload @buffer(0),
            constant Payload& referencePayload @buffer(1)
        ) {
            constant Payload* pointerAlias = pointerPayload;
            constant Payload& referenceAlias = referencePayload;
            float pointerValue = readPointer(pointerAlias);
            float referenceValue = readReference(referenceAlias);
            pointerAlias.value = pointerValue;
            referenceAlias.value = referenceValue;
            mutatePointer(pointerAlias);
            mutateReference(referenceAlias);
        }
    }
}
"""


METAL_NESTED_CONSTANT_LOCAL_ARRAY_SHADER = """
shader MetalNestedConstantLocalArrayValidation {
    float readValue(const float values[2][2], int row, int col) {
        return values[row][col];
    }

    void mutate(float values[2][2]) {
        values[0][0] = 2.0;
    }

    compute {
        void main() {
            constant float values[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
            float value = readValue(values, 1, 0);
            values[0][1] = value;
            mutate(values);
        }
    }
}
"""


METAL_CONSTANT_SCALAR_VECTOR_RESOURCE_INDEX_SHADER = """
shader MetalConstantScalarVectorResourceIndexValidation {
    sampler2D textures[];
    sampler samplers[];

    struct FSInput {
        vec2 uv @ TEXCOORD0;
    };

    float helper(float input) {
        constant float scale = 2.0;
        constant vec2 bias = vec2(0.25, 0.5);
        float value = input * scale + bias.x;
        scale = value;
        bias.x = value;
        return value;
    }

    vec4 sampleLayer(vec2 uv, sampler2D textures[], sampler samplers[]) {
        constant int LAYER = 3;
        return texture(textures[LAYER], samplers[LAYER], uv);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            float value = helper(1.0);
            return sampleLayer(input.uv, textures, samplers) + vec4(value);
        }
    }
}
"""


METAL_STRUCT_RESOURCE_ARRAY_QUERY_SHADER = """
shader MetalStructResourceArrayQueryValidation {
    struct TexturePack {
        sampler2D textures[4];
        sampler2DArray layers[4];
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];
    };

    int query(TexturePack pack, vec3 uvLayer) {
        constant int LAYER = 2;
        ivec2 dims = textureSize(pack.textures[LAYER], LAYER);
        ivec3 layerDims = textureSize(pack.layers[LAYER], 1);
        int levels = textureQueryLevels(pack.textures[LAYER]);
        vec2 lod = textureQueryLod(pack.layers[LAYER], uvLayer);
        int samples = textureSamples(pack.msTextures[LAYER])
            + imageSamples(pack.msArrays[LAYER]);
        return dims.x + layerDims.z + levels + int(lod.x) + samples;
    }

    fragment {
        vec4 main() @ gl_FragColor {
            TexturePack pack;
            int value = query(pack, vec3(0.5, 0.25, 1.0));
            return vec4(float(value));
        }
    }
}
"""


METAL_NESTED_RESOURCE_CONTAINER_FORWARDING_SHADER = """
shader MetalNestedResourceContainerForwardingValidation {
    struct TexturePack {
        sampler2D textures[4];
        sampler2DArray layers[4];
    };

    struct ResourceBank {
        TexturePack pack;
        sampler samplers[4];
    };

    vec4 samplePack(
        TexturePack pack,
        sampler samplers[4],
        vec2 uv,
        vec3 uvLayer
    ) {
        constant int LAYER = 2;
        vec4 planar = texture(pack.textures[LAYER], samplers[LAYER], uv);
        vec4 layered = texture(pack.layers[LAYER], samplers[LAYER], uvLayer);
        return planar + layered;
    }

    vec4 sampleBank(ResourceBank bank, vec2 uv, vec3 uvLayer) {
        return samplePack(bank.pack, bank.samplers, uv, uvLayer);
    }

    fragment {
        vec4 main() @ gl_FragColor {
            ResourceBank bank;
            return sampleBank(
                bank,
                vec2(0.5, 0.25),
                vec3(0.25, 0.75, 1.0)
            );
        }
    }
}
"""


METAL_ADDRESS_SPACE_MISMATCH_CALL_SHADER = """
shader MetalAddressSpaceMismatchCallValidation {
    struct Payload {
        float value;
    };

    void useThreadgroup(threadgroup Payload& scratch) {
        scratch.value = 1.0;
    }

    void useDevice(device Payload& payload) {
        payload.value = 2.0;
    }

    compute {
        void main(device Payload* payload @buffer(0)) {
            threadgroup Payload scratch;
            useThreadgroup(payload[0]);
            useDevice(scratch);
        }
    }
}
"""


METAL_MESH_OBJECT_SHADER = """
shader MetalMeshObjectValidation {
    object {
        layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;
        void main() { }
    }

    mesh {
        void main()
            @max_total_threads_per_threadgroup(96)
            @max_vertices(64)
            @max_primitives(32)
            @outputtopology(triangle)
        {
            SetMeshOutputCounts(64, 12);
        }
    }
}
"""


METAL_MESH_DISPATCH_INVALID_GRID_SHADER = """
shader MetalMeshDispatchInvalidGridValidation {
    object {
        void main() @max_total_threads_per_threadgroup(32) {
            DispatchMesh(2);
        }
    }
}
"""


METAL_MESH_OUTPUT_SIGNATURE_SHADER = """
shader MetalMeshOutputSignatureValidation {
    struct MeshVertex {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };

    struct MeshPrimitive {
        uint layer @ gl_PrimitiveID;
    };

    mesh {
        void main(
            @vertices out MeshVertex verts[3],
            @indices out uvec3 tris[1],
            @primitives out MeshPrimitive prims[1]
        ) @numthreads(32, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            verts[0].uv = vec2(0.0);
            tris[0] = uvec3(0u, 1u, 2u);
            prims[0].layer = 0u;
        }
    }
}
"""


METAL_MESH_PRIMITIVE_SETTER_SHADER = """
shader MetalMeshPrimitiveSetterValidation {
    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    struct MeshPrimitive {
        uint layer @ gl_PrimitiveID;
    };

    mesh {
        void main(
            @vertices out MeshVertex verts[3],
            @indices out uvec3 tris[1],
            @primitives out MeshPrimitive prims[1]
        ) @numthreads(32, 1, 1) @outputtopology(triangle) {
            MeshVertex outVertex;
            outVertex.position = vec4(0.0, 0.0, 0.0, 1.0);
            MeshPrimitive outPrimitive;
            outPrimitive.layer = 7u;
            SetMeshOutputCounts(3, 1);
            SetVertex(0, outVertex);
            SetPrimitive(0, outPrimitive);
            SetIndex(0, uvec3(0u, 1u, 2u));
        }
    }
}
"""


METAL_MESH_OUTPUT_VARIABLE_MEMBER_WRITES_SHADER = """
shader MetalMeshOutputVariableMemberWritesValidation {
    struct MeshVertex {
        vec4 position @ gl_Position;
        vec2 uv @ TEXCOORD0;
    };

    struct MeshPrimitive {
        uint layer @ gl_PrimitiveID;
        vec2 bary @ TEXCOORD1;
    };

    mesh {
        void main(
            @vertices out MeshVertex verts[4],
            @indices out uvec3 tris[2],
            @primitives out MeshPrimitive prims[2]
        ) @numthreads(32, 1, 1) @outputtopology(triangle) {
            uint vertexIndex = 1u;
            uint primitiveIndex = 0u;
            SetMeshOutputCounts(4, 2);
            verts[vertexIndex].position = vec4(1.0, 0.0, 0.0, 1.0);
            verts[vertexIndex].position += vec4(0.0, 1.0, 0.0, 0.0);
            verts[vertexIndex].uv = vec2(0.5, 1.0);
            verts[vertexIndex].uv += vec2(0.25, 0.0);
            tris[primitiveIndex] = uvec3(0u, 1u, 2u);
            prims[primitiveIndex].layer = 2u;
            prims[primitiveIndex].layer += 3u;
            prims[primitiveIndex].bary = vec2(0.25, 0.75);
            prims[primitiveIndex].bary += vec2(0.5, 0.0);
        }
    }
}
"""


METAL_MESH_PAYLOAD_DISPATCH_SHADER = """
shader MetalMeshPayloadDispatchValidation {
    struct MeshPayload {
        uint meshlet;
    };

    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    task {
        void main() @numthreads(1, 1, 1) {
            groupshared MeshPayload payload;
            payload.meshlet = 7u;
            DispatchMesh(1, 1, 1, payload);
        }
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[1],
            @indices out uint points[1]
        ) @numthreads(1, 1, 1) @outputtopology(point) {
            SetMeshOutputCounts(1, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            points[0] = 0u;
        }
    }
}
"""


METAL_MESH_PAYLOAD_HELPER_DISPATCH_SHADER = """
shader MetalMeshPayloadHelperDispatchValidation {
    struct MeshPayload {
        uint meshlet;
    };

    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    task {
        void dispatchOne(threadgroup MeshPayload& payload) {
            DispatchMesh(1, 1, 1, payload);
        }

        void launch(threadgroup MeshPayload& payload) {
            dispatchOne(payload);
        }

        void main() @numthreads(1, 1, 1) {
            groupshared MeshPayload payload;
            payload.meshlet = 7u;
            launch(payload);
        }
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[1],
            @indices out uint points[1]
        ) @numthreads(1, 1, 1) @outputtopology(point) {
            SetMeshOutputCounts(1, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            points[0] = 0u;
        }
    }
}
"""


METAL_MESH_PAYLOAD_MEMBER_SOURCE_SHADER = """
shader MetalMeshPayloadMemberSourceValidation {
    struct MeshPayload {
        uint meshlet;
    };

    struct PayloadBlock {
        MeshPayload active;
    };

    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    task {
        void issue(threadgroup MeshPayload& payload) {
            DispatchMesh(1, 1, 1, payload);
        }

        void main() @numthreads(1, 1, 1) {
            groupshared PayloadBlock block;
            block.active.meshlet = 7u;
            issue(block.active);
        }
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[1],
            @indices out uint points[1]
        ) @numthreads(1, 1, 1) @outputtopology(point) {
            SetMeshOutputCounts(1, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            points[0] = 0u;
        }
    }
}
"""


METAL_MESH_PAYLOAD_ARRAY_SOURCE_SHADER = """
shader MetalMeshPayloadArraySourceValidation {
    struct MeshPayload {
        uint meshlet;
        uint lane;
    };

    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    task {
        void main() @numthreads(1, 1, 1) {
            groupshared MeshPayload payloads[2];
            groupshared MeshPayload copied;
            payloads[0].meshlet = 3u;
            payloads[0].lane = 4u;
            copied = payloads[0];
            payloads[1] = copied;
            DispatchMesh(1, 1, 1, payloads[1]);
        }
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[1],
            @indices out uint points[1]
        ) @numthreads(1, 1, 1) @outputtopology(point) {
            SetMeshOutputCounts(1, 1);
            verts[0].position =
                vec4(float(payload.meshlet + payload.lane), 0.0, 0.0, 1.0);
            points[0] = 0u;
        }
    }
}
"""


METAL_MESH_PAYLOAD_ADDRESS_SPACE_SHADER = """
shader MetalMeshPayloadAddressSpaceValidation {
    struct Payload {
        vec4 color;
    };

    void mutate(thread Payload& localPayload) {
        localPayload.color = vec4(0.5, 0.5, 0.5, 1.0);
    }

    object {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            mutate(payload);
            payload.color = vec4(1.0, 0.0, 0.0, 1.0);
            DispatchMesh(1, 1, 1);
        }
    }

    mesh {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            mutate(payload);
            payload.color = vec4(0.0, 1.0, 0.0, 1.0);
            vec4 color = payload.color;
        }
    }
}
"""


METAL_MESH_PAYLOAD_LOCAL_ALIAS_SHADER = """
shader MetalMeshPayloadLocalAliasValidation {
    struct Payload {
        vec4 color;
    };

    object {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Payload& alias = payload;
            alias.color = vec4(1.0, 0.0, 0.0, 1.0);
            DispatchMesh(1, 1, 1);
        }
    }

    mesh {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Payload& alias = payload;
            alias.color = vec4(0.0, 1.0, 0.0, 1.0);
            vec4 color = alias.color;
        }
    }
}
"""


METAL_MESH_PAYLOAD_POINTER_ALIAS_SHADER = """
shader MetalMeshPayloadPointerAliasValidation {
    struct Payload {
        vec4 color;
    };

    object {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Payload* alias = payload;
            alias.color = vec4(1.0, 0.0, 0.0, 1.0);
            DispatchMesh(1, 1, 1);
        }
    }

    mesh {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Payload* alias = payload;
            alias.color = vec4(0.0, 1.0, 0.0, 1.0);
            vec4 color = alias.color;
        }
    }
}
"""


METAL_MESH_PAYLOAD_POINTER_HELPER_SHADER = """
shader MetalMeshPayloadPointerHelperValidation {
    struct Payload {
        vec4 color;
    };

    void tint(Payload* payload @object_data) {
        payload.color = vec4(0.25, 0.5, 0.75, 1.0);
    }

    float read(const Payload* payload @object_data) {
        return payload.color.x;
    }

    object {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Payload* alias = payload;
            tint(alias);
            float value = read(alias);
            payload.color.x = value;
            DispatchMesh(1, 1, 1);
        }
    }

    mesh {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Payload* alias = payload;
            float value = read(alias);
            tint(alias);
            vec4 color = alias.color + vec4(value);
        }
    }
}
"""


METAL_MESH_PAYLOAD_MEMBER_POINTER_HELPER_SHADER = """
shader MetalMeshPayloadMemberPointerHelperValidation {
    struct Payload {
        vec4 color;
    };

    struct Wrapper {
        Payload* ptr @object_data;
    };

    void tint(Payload* payload @object_data) {
        payload.color = vec4(0.25, 0.5, 0.75, 1.0);
    }

    float read(const Payload* payload @object_data) {
        return payload.color.x;
    }

    object {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Wrapper wrapper;
            wrapper.ptr = payload;
            tint(wrapper.ptr);
            float value = read(wrapper.ptr);
            payload.color.x = value;
            DispatchMesh(1, 1, 1);
        }
    }

    mesh {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            Wrapper wrapper;
            wrapper.ptr = payload;
            float value = read(wrapper.ptr);
            tint(wrapper.ptr);
            vec4 color = wrapper.ptr.color + vec4(value);
        }
    }
}
"""


METAL_CONST_OBJECT_PAYLOAD_SHADER = """
shader MetalConstObjectPayloadValidation {
    struct Payload {
        vec4 color;
    };

    object {
        void main(const Payload& payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            payload.color = vec4(1.0, 0.0, 0.0, 1.0);
            DispatchMesh(1, 1, 1);
        }
    }

    mesh {
        void main(Payload payload @payload)
            @max_total_threads_per_threadgroup(32)
        {
            vec4 color = payload.color;
        }
    }
}
"""


METAL_MESH_PAYLOAD_INVALID_SOURCE_SHADER = """
shader MetalMeshPayloadInvalidSourceValidation {
    struct MeshPayload {
        uint meshlet;
    };

    struct PayloadBlock {
        MeshPayload active;
    };

    MeshPayload makePayload() {
        MeshPayload payload;
        payload.meshlet = 1u;
        return payload;
    }

    task {
        void main(
            device MeshPayload* devicePayloads @buffer(0),
            constant MeshPayload* constantPayloads @buffer(1),
            device PayloadBlock* deviceBlocks @buffer(2)
        ) @numthreads(1, 1, 1) {
            MeshPayload threadPayload;
            groupshared MeshPayload payloads[2];
            payloads[0].meshlet = 2u;
            threadgroup MeshPayload* alias = &payloads[0];
            DispatchMesh(1, 1, 1, makePayload());
            DispatchMesh(1, 1, 1, threadPayload);
            DispatchMesh(1, 1, 1, devicePayloads[0]);
            DispatchMesh(1, 1, 1, constantPayloads[0]);
            DispatchMesh(1, 1, 1, deviceBlocks[0].active);
            DispatchMesh(1, 1, 1, alias);
            DispatchMesh(1, 1, 1, payloads);
            DispatchMesh(1, 1, 1, alias[0]);
            DispatchMesh(1, 1, 1, payloads[0]);
        }
    }

    mesh {
        void main(MeshPayload payload @mesh_payload)
            @numthreads(1, 1, 1)
            @max_vertices(1)
            @max_primitives(1)
            @outputtopology(point)
        {
            SetMeshOutputCounts(1, 1);
        }
    }
}
"""


METAL_MESH_DISPATCH_WITHOUT_GRID_CONTEXT_SHADER = """
shader MetalMeshDispatchWithoutGridContextValidation {
    struct MeshPayload {
        uint meshlet;
    };

    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[1],
            @indices out uint points[1]
        ) @numthreads(1, 1, 1) @outputtopology(point) {
            DispatchMesh(1, 1, 1, payload);
            SetMeshOutputCounts(1, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            points[0] = 0u;
        }
    }
}
"""


METAL_MESH_DISPATCH_HELPER_WITHOUT_GRID_CONTEXT_SHADER = """
shader MetalMeshDispatchHelperWithoutGridContextValidation {
    struct MeshPayload {
        uint meshlet;
    };

    struct MeshVertex {
        vec4 position @ gl_Position;
    };

    void issue(threadgroup MeshPayload& payload) {
        DispatchMesh(1, 1, 1, payload);
    }

    mesh {
        void main(
            @mesh_payload in MeshPayload payload,
            @vertices out MeshVertex verts[1],
            @indices out uint points[1]
        ) @numthreads(1, 1, 1) @outputtopology(point) {
            issue(payload);
            SetMeshOutputCounts(1, 1);
            verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
            points[0] = 0u;
        }
    }
}
"""


METAL_RAY_TRACING_HELPER_SHADER = """
shader MetalRayTracingHelperValidation {
    accelerationStructureEXT topLevelAS @binding(0);

    void shoot(vec3 origin, vec3 direction) {
        let topLevelAlias = topLevelAS;
        TraceRay(
            topLevelAlias,
            0,
            0xff,
            0,
            1,
            0,
            origin,
            0.001,
            direction,
            1000.0,
            0
        );
    }

    ray_generation {
        void main() {
            shoot(vec3(0.0), vec3(0.0, 0.0, 1.0));
        }
    }
}
"""


METAL_RAY_TRACING_INTERSECTION_TABLE_TRACE_SHADER = """
shader MetalRayTracingIntersectionTableTraceValidation {
    accelerationStructureEXT topLevelAS @binding(0);
    intersection_function_table<instancing> intersectionFunctions @binding(1);

    void shoot(vec3 origin, vec3 direction) {
        TraceRay(
            topLevelAS,
            0,
            0xff,
            0,
            1,
            0,
            origin,
            0.001,
            direction,
            1000.0,
            0
        );
    }

    ray_generation {
        void main() {
            shoot(vec3(0.0), vec3(0.0, 0.0, 1.0));
        }
    }
}
"""


METAL_RAY_TRACING_PRIMITIVE_ACCELERATION_SHADER = """
shader MetalRayTracingPrimitiveAccelerationValidation {
    primitive_acceleration_structure primitiveAS @binding(0);
    intersection_function_table<triangle_data> intersectionFunctions @binding(1);

    ray_generation {
        void main() {
            TraceRay(
                primitiveAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                0
            );
        }
    }
}
"""


METAL_RAY_TRACING_PAYLOAD_TRACE_SHADER = """
shader MetalRayTracingPayloadTraceValidation {
    struct Payload {
        vec3 color;
    };

    accelerationStructureEXT topLevelAS @binding(0);
    intersection_function_table<instancing> intersectionFunctions @binding(1);

    ray_generation {
        void main() {
            Payload payload;
            payload.color = vec3(1.0, 0.0, 0.0);
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                payload
            );
        }
    }
}
"""


METAL_RAY_TRACING_INVALID_ACCELERATION_STRUCTURE_SHADER = """
shader MetalRayTracingInvalidAccelerationStructureValidation {
    struct Payload {
        vec3 color;
    };

    accelerationStructureEXT topLevelAS @binding(0);
    intersection_function_table<instancing> intersectionFunctions @binding(1);

    ray_generation {
        void main() {
            Payload payload;
            TraceRay(
                payload,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                payload
            );
        }
    }
}
"""


METAL_RAY_ACCELERATION_STRUCTURE_ARRAY_DIAGNOSTIC_SHADER = """
shader MetalRayAccelerationStructureArrayDiagnosticValidation {
    accelerationStructureEXT topLevelAS[2] @binding(0);
    primitive_acceleration_structure primitiveAS[2] @binding(3);

    ray_generation {
        void main(accelerationStructureEXT paramAS[2] @binding(5)) {
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                0
            );

            TraceRay(
                topLevelAS[1],
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                0
            );

            TraceRay(
                primitiveAS[0],
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                0
            );

            TraceRay(
                paramAS[0],
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                0
            );
        }
    }
}
"""


METAL_RAY_TRACING_PAYLOAD_DIAGNOSTIC_SHADER = """
shader MetalRayTracingPayloadDiagnosticValidation {
    struct Payload {
        vec3 color;
    };

    accelerationStructureEXT topLevelAS @binding(0);

    ray_generation {
        void main() {
            Payload payload;
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                payload
            );
        }
    }
}
"""


METAL_RAY_PAYLOAD_HELPER_ADDRESS_SPACE_SHADER = """
shader MetalRayPayloadHelperAddressSpaceValidation {
    struct Payload {
        vec3 color;
    };

    void tint(Payload& payload @ray_data) {
        payload.color = vec3(1.0, 0.0, 0.0);
    }

    void rejectThreadPayload(Payload& payload) {
        payload.color = vec3(0.0, 0.0, 0.0);
    }

    ray_any_hit {
        void main(Payload payload @ payload) {
            tint(payload);
            rejectThreadPayload(payload);
        }
    }
}
"""


METAL_RAY_PAYLOAD_ALIAS_ADDRESS_SPACE_SHADER = """
shader MetalRayPayloadAliasAddressSpaceValidation {
    struct Payload {
        vec3 color;
    };

    accelerationStructureEXT topLevelAS @binding(0);
    intersection_function_table<instancing> intersectionFunctions @binding(1);

    ray_generation {
        void main(Payload* external @device @buffer(2)) {
            Payload payload;
            Payload& threadAlias = payload;
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                threadAlias
            );

            Payload& deviceAlias = external[0];
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                deviceAlias
            );
        }
    }
}
"""


METAL_RAY_PAYLOAD_MEMBER_LVALUE_SHADER = """
shader MetalRayPayloadMemberLvalueValidation {
    struct Payload {
        vec3 color;
    };

    struct Wrapper {
        Payload payload;
        Payload payloads[2];
    };

    accelerationStructureEXT topLevelAS @binding(0);
    intersection_function_table<instancing> intersectionFunctions @binding(1);

    void rejectConst(const Wrapper& constWrapper) {
        TraceRay(
            topLevelAS,
            0,
            0xff,
            0,
            1,
            0,
            vec3(0.0),
            0.001,
            vec3(0.0, 0.0, 1.0),
            1000.0,
            constWrapper.payload
        );
    }

    ray_generation {
        void main() {
            Wrapper wrapper;
            Payload payloads[2];
            Payload* pointer = &payloads[0];
            rejectConst(wrapper);
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                wrapper.payload
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                payloads
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                payloads[0]
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                wrapper.payloads
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                pointer
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                pointer[0]
            );
        }
    }
}
"""


METAL_RAY_PAYLOAD_TYPE_VALIDATION_SHADER = """
shader MetalRayPayloadTypeValidation {
    struct Payload {
        vec3 color;
    };

    struct OtherPayload {
        vec3 color;
    };

    struct Wrapper {
        Payload payload;
        OtherPayload other;
        OtherPayload others[2];
    };

    accelerationStructureEXT topLevelAS @binding(0);
    intersection_function_table<instancing> intersectionFunctions @binding(1);

    ray_miss {
        void main(Payload payload @payload) {
            payload.color = vec3(0.0, 0.0, 1.0);
        }
    }

    ray_generation {
        void main() {
            Payload payload;
            Wrapper wrapper;
            OtherPayload other;
            OtherPayload others[2];
            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                payload
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                wrapper.payload
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                other
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                wrapper.other
            );

            TraceRay(
                topLevelAS,
                0,
                0xff,
                0,
                1,
                0,
                vec3(0.0),
                0.001,
                vec3(0.0, 0.0, 1.0),
                1000.0,
                others[0]
            );
        }
    }
}
"""


METAL_RAY_CALLABLE_DISPATCH_SHADER = """
shader MetalRayCallableDispatchValidation {
    struct CallableData {
        vec4 color;
    };

    visible_function_table<CallableData> callables @binding(1);

    ray_generation {
        void main() {
            CallableData data;
            data.color = vec4(1.0);
            CallShader(0, data);
        }
    }
}
"""


METAL_RAY_CALLABLE_INVALID_EXPLICIT_TABLE_SHADER = """
shader MetalRayCallableInvalidExplicitTableValidation {
    struct CallableData {
        vec4 color;
    };

    visible_function_table<CallableData> callables @binding(1);

    ray_generation {
        void main() {
            CallableData data;
            data.color = vec4(1.0);
            CallShader(data, 0, data);
        }
    }
}
"""


METAL_RAY_CALLABLE_ALIAS_ADDRESS_SPACE_SHADER = """
shader MetalRayCallableAliasAddressSpaceValidation {
    struct CallableData {
        vec4 color;
    };

    visible_function_table<CallableData> callables @binding(1);

    ray_generation {
        void main(CallableData* external @device @buffer(2)) {
            CallableData data;
            data.color = vec4(1.0);
            CallableData& threadAlias = data;
            CallShader(0, threadAlias);

            CallableData& deviceAlias = external[0];
            CallShader(1, deviceAlias);
        }
    }
}
"""


METAL_RAY_CALLABLE_POINTER_DEREF_SHADER = """
shader MetalRayCallablePointerDerefValidation {
    struct CallableData {
        vec4 color;
    };

    visible_function_table<CallableData> callables @binding(1);

    ray_generation {
        void main(CallableData* external @device @buffer(2)) {
            CallableData data;
            data.color = vec4(1.0);
            CallableData* alias = &data;
            CallShader(0, *alias);
            CallShader(1, *external);
        }
    }
}
"""


METAL_RAY_CALLABLE_HELPER_MEMBER_LVALUE_SHADER = """
shader MetalRayCallableHelperMemberLvalueValidation {
    struct CallableData {
        vec4 color;
    };

    struct Wrapper {
        CallableData data;
    };

    visible_function_table<CallableData> callables @binding(1);

    void invoke(uint shaderIndex, CallableData& data) {
        CallShader(shaderIndex, data);
    }

    void rejectConst(const Wrapper& wrapper) {
        CallShader(1, wrapper.data);
    }

    ray_generation {
        void main() {
            Wrapper wrapper;
            wrapper.data.color = vec4(1.0);
            invoke(2u, wrapper.data);
            CallShader(3u, wrapper.data);
            rejectConst(wrapper);
        }
    }
}
"""


METAL_RAY_STAGES_SHADER = """
shader MetalRayStagesValidation {
    struct Payload {
        vec3 color;
    };

    struct HitAttrib {
        vec2 bary;
    };

    struct BoundsHit {
        bool accept @ accept_intersection;
        float distance @ distance;
    };

    ray_any_hit {
        void main(Payload payload @ payload, HitAttrib attr @ hit_attribute) {
            payload.color = vec3(attr.bary, 0.0);
        }
    }

    ray_closest_hit {
        void main(Payload payload @ payload, HitAttrib attr @ hit_attribute) {
            payload.color = vec3(attr.bary, 1.0);
        }
    }

    ray_miss {
        void main(Payload payload @ payload) {
            payload.color = vec3(0.0, 0.0, 0.0);
        }
    }

    ray_callable {
        void main(Payload data @ callable_data) {
            data.color = vec3(1.0, 1.0, 1.0);
        }
    }

    ray_intersection {
        BoundsHit main(Payload payload @ payload) @bounding_box {
            payload.color = vec3(1.0, 0.0, 0.0);
            return BoundsHit { accept: true, distance: 1.0 };
        }
    }
}
"""


METAL_INTERSECTION_FUNCTION_TABLE_SHADER = """
shader MetalIntersectionFunctionTableValidation {
    intersection_function_table<instancing> intersectionFunctions @binding(3);

    uint tableSize() {
        return intersectionFunctions.size();
    }

    ray_generation {
        void main() {
            uint count = tableSize();
        }
    }
}
"""


METAL_RAY_FUNCTION_TABLE_PARAMETER_SHADER = """
shader MetalRayFunctionTableParameterValidation {
    struct CallableData {
        vec4 color;
    };

    ray_generation {
        void main(
            visible_function_table<CallableData> callables @binding(1),
            intersection_function_table<instancing> intersectionFunctions @binding(2)
        ) {
            CallableData data;
            data.color = vec4(1.0);
            CallShader(callables, 0, data);
            uint count = intersectionFunctions.size();
        }
    }
}
"""


METAL_RAY_FUNCTION_TABLE_ARRAY_DIAGNOSTIC_SHADER = """
shader MetalRayFunctionTableArrayDiagnosticValidation {
    struct CallableData {
        vec4 color;
    };

    visible_function_table<CallableData> callables[2] @binding(1);
    intersection_function_table<instancing> intersectionFunctions[2] @binding(3);

    void invoke(uint shaderIndex, CallableData data) {
        CallShader(callables[0], shaderIndex, data);
    }

    uint tableSize() {
        return intersectionFunctions[0].size();
    }

    ray_generation {
        void main(
            visible_function_table<CallableData> paramCallables[2] @binding(5),
            intersection_function_table<instancing> paramIntersections[2]
                @binding(7)
        ) {
            CallableData data;
            data.color = vec4(1.0);
            invoke(1u, data);
            uint count = tableSize();
            CallShader(paramCallables[0], 1u, data);
            uint paramCount = paramIntersections[0].size();
        }
    }
}
"""


METAL_TEXTURE_3D_PROJECTED_OFFSET_FRAGMENT_SHADER = """
shader MetalTexture3DProjectedOffsetValidation {
    sampler3D volumeMap;
    sampler linearSampler;

    struct FSInput {
        vec4 xyzq @ TEXCOORD0;
        float lod;
        vec3 ddx @ TEXCOORD1;
        vec3 ddy @ TEXCOORD2;
    };

    vec4 sampleProjectedVolumeOffsets(
        sampler3D volume,
        sampler s,
        vec4 xyzq,
        float lod,
        vec3 ddx,
        vec3 ddy
    ) {
        vec4 projected = textureProjOffset(volume, s, xyzq, ivec3(1, 0, -1));
        vec4 lodProjected = textureProjLodOffset(
            volume,
            s,
            xyzq,
            lod,
            ivec3(1, 0, -1)
        );
        vec4 gradProjected = textureProjGradOffset(
            volume,
            s,
            xyzq,
            ddx,
            ddy,
            ivec3(1, 0, -1)
        );
        return projected + lodProjected + gradProjected;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return sampleProjectedVolumeOffsets(
                volumeMap,
                linearSampler,
                input.xyzq,
                input.lod,
                input.ddx,
                input.ddy
            );
        }
    }
}
"""


SHADOW_GATHER_COMPARE_OFFSET_FRAGMENT_SHADER = """
shader ShadowGatherCompareOffsetValidation {
    sampler2DShadow shadowMap;
    sampler2DArrayShadow shadowArray;
    samplerCubeArrayShadow cubeShadowArray;
    sampler compareSampler;

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec3 uvLayer @ TEXCOORD1;
        vec4 cubeLayer @ TEXCOORD2;
        float depth;
        ivec2 offset @ TEXCOORD3;
    };

    vec4 gatherShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth, ivec2 offset) {
        vec4 gathered = textureGatherCompare(tex, s, uv, depth);
        vec4 offsetGathered = textureGatherCompareOffset(tex, s, uv, depth, offset);
        float offsetCompared = textureCompareOffset(tex, s, uv, depth, offset);
        return gathered + offsetGathered + vec4(offsetCompared);
    }

    vec4 gatherShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
        vec4 gathered = textureGatherCompare(tex, s, uvLayer, depth);
        vec4 offsetGathered = textureGatherCompareOffset(tex, s, uvLayer, depth, offset);
        float offsetCompared = textureCompareOffset(tex, s, uvLayer, depth, offset);
        return gathered + offsetGathered + vec4(offsetCompared);
    }

    vec4 gatherCubeShadowArray(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
        return textureGatherCompare(tex, s, cubeLayer, depth);
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            return gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)
                + gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset)
                + gatherCubeShadowArray(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
        }
    }
}
"""


SHADOW_COMPARE_LOD_GRAD_FRAGMENT_SHADER = """
shader ShadowCompareLodGradValidation {
    sampler2DShadow shadowMap;
    sampler2DArrayShadow shadowArray;
    samplerCubeArrayShadow cubeShadowArray;
    sampler compareSampler;

    struct FSInput {
        vec2 uv @ TEXCOORD0;
        vec3 uvLayer @ TEXCOORD1;
        float depth;
        float lod;
        vec2 ddx @ TEXCOORD2;
        vec2 ddy @ TEXCOORD3;
        vec4 cubeLayer @ TEXCOORD4;
        vec3 cubeDdx @ TEXCOORD5;
        vec3 cubeDdy @ TEXCOORD6;
    };

    float compareShadow(
        sampler2DShadow tex,
        sampler s,
        vec2 uv,
        float depth,
        float lod,
        vec2 ddx,
        vec2 ddy
    ) {
        float lodValue = textureCompareLod(tex, s, uv, depth, lod);
        float lodOffsetValue = textureCompareLodOffset(tex, s, uv, depth, lod, ivec2(1, 0));
        float gradValue = textureCompareGrad(tex, s, uv, depth, ddx, ddy);
        float gradOffsetValue = textureCompareGradOffset(tex, s, uv, depth, ddx, ddy, ivec2(1, 0));
        return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
    }

    float compareShadowArray(
        sampler2DArrayShadow tex,
        sampler s,
        vec3 uvLayer,
        float depth,
        float lod,
        vec2 ddx,
        vec2 ddy
    ) {
        float gradValue = textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy);
        float gradOffsetValue = textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, ivec2(1, 0));
        return gradValue + gradOffsetValue;
    }

    float compareCubeArrayShadow(
        samplerCubeArrayShadow tex,
        sampler s,
        vec4 cubeLayer,
        float depth,
        float lod,
        vec3 ddx,
        vec3 ddy
    ) {
        float lodValue = textureCompareLod(tex, s, cubeLayer, depth, lod);
        float gradValue = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
        return lodValue + gradValue;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            float shadow = compareShadow(
                shadowMap,
                compareSampler,
                input.uv,
                input.depth,
                input.lod,
                input.ddx,
                input.ddy
            );
            float arrayShadow = compareShadowArray(
                shadowArray,
                compareSampler,
                input.uvLayer,
                input.depth,
                input.lod,
                input.ddx,
                input.ddy
            );
            float cubeArrayShadow = compareCubeArrayShadow(
                cubeShadowArray,
                compareSampler,
                input.cubeLayer,
                input.depth,
                input.lod,
                input.cubeDdx,
                input.cubeDdy
            );
            return vec4(shadow + arrayShadow + cubeArrayShadow);
        }
    }
}
"""


PROJECTED_SHADOW_COMPARE_FRAGMENT_SHADER = """
shader ProjectedShadowCompareValidation {
    sampler2DShadow shadowMap;
    sampler2DArrayShadow shadowArray;
    sampler compareSampler;

    struct FSInput {
        vec3 uvq @ TEXCOORD0;
        vec4 uvqw @ TEXCOORD1;
        vec4 uvLayerQ @ TEXCOORD2;
        float depth;
        float lod;
        vec2 ddx @ TEXCOORD3;
        vec2 ddy @ TEXCOORD4;
    };

    float projectedShadow(
        sampler2DShadow tex,
        sampler s,
        vec3 uvq,
        vec4 uvqw,
        float depth,
        float lod,
        vec2 ddx,
        vec2 ddy
    ) {
        float projected = textureCompareProj(tex, s, uvq, depth);
        float projectedW = textureCompareProj(tex, s, uvqw, depth);
        float offsetProjected = textureCompareProjOffset(tex, s, uvq, depth, ivec2(1, 0));
        float lodProjected = textureCompareProjLod(tex, s, uvq, depth, lod);
        float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvq, depth, lod, ivec2(1, 0));
        float gradProjected = textureCompareProjGrad(tex, s, uvq, depth, ddx, ddy);
        float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvq, depth, ddx, ddy, ivec2(1, 0));
        return projected + projectedW + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
    }

    float projectedArrayShadow(
        sampler2DArrayShadow tex,
        sampler s,
        vec4 uvLayerQ,
        float depth,
        vec2 ddx,
        vec2 ddy
    ) {
        float projected = textureCompareProj(tex, s, uvLayerQ, depth);
        float offsetProjected = textureCompareProjOffset(tex, s, uvLayerQ, depth, ivec2(1, 0));
        float gradProjected = textureCompareProjGrad(tex, s, uvLayerQ, depth, ddx, ddy);
        float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvLayerQ, depth, ddx, ddy, ivec2(1, 0));
        return projected + offsetProjected + gradProjected + gradOffsetProjected;
    }

    fragment {
        vec4 main(FSInput input) @ gl_FragColor {
            float shadow = projectedShadow(
                shadowMap,
                compareSampler,
                input.uvq,
                input.uvqw,
                input.depth,
                input.lod,
                input.ddx,
                input.ddy
            );
            float arrayShadow = projectedArrayShadow(
                shadowArray,
                compareSampler,
                input.uvLayerQ,
                input.depth,
                input.ddx,
                input.ddy
            );
            return vec4(shadow + arrayShadow);
        }
    }
}
"""


DEFAULT_FLOAT_IMAGE_COMPUTE_SHADER = """
shader DefaultFloatImageValidation {
    image2D storageImage;

    float touchScalar(image2D image, ivec2 pixel, float value) {
        float oldValue = imageLoad(image, pixel);
        imageStore(image, pixel, oldValue + value);
        return oldValue;
    }

    vec4 touchVector(image2D image, ivec2 pixel, vec4 value) {
        vec4 oldValue = imageLoad(image, pixel);
        imageStore(image, pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float scalarValue = touchScalar(storageImage, ivec2(0, 1), 0.25);
            vec4 vectorValue = touchVector(storageImage, ivec2(2, 3), vec4(1.0));
            uint unsignedValue = 7u;
        }
    }
}
"""


METAL_SCALAR_IMAGE_COMPUTE_SHADER = DEFAULT_FLOAT_IMAGE_COMPUTE_SHADER


RG_IMAGE_COMPUTE_SHADER = """
shader RGImageValidation {
    image2D rgFloat @rg32f;
    uimage2D rgUnsigned @rg32ui;

    float scalarFloat(image2D image @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(image, pixel);
        imageStore(image, pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(image, pixel);
        imageStore(image, pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsigned(uimage2D image @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(image, pixel);
        imageStore(image, pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsigned(uimage2D image @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(image, pixel);
        imageStore(image, pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float a = scalarFloat(rgFloat, ivec2(0, 1), 0.25);
            vec2 b = vectorFloat(rgFloat, ivec2(2, 3), vec2(1.0));
            uint c = scalarUnsigned(rgUnsigned, ivec2(4, 5), 7u);
            uvec2 d = vectorUnsigned(rgUnsigned, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


INTEGER_IMAGE_ATOMICS_COMPUTE_SHADER = """
shader IntegerImageAtomicsValidation {
    uimage2D counters @r32ui;
    iimage2D signedCounters @r32i;
    uimage3D volumeCounters @r32ui;
    iimage2DArray layerCounters @r32i;

    uint touchUnsigned(
        uimage2D image @r32ui,
        ivec2 pixel,
        uint value,
        uint replacement
    ) {
        uint added = imageAtomicAdd(image, pixel, value);
        uint minValue = imageAtomicMin(image, pixel, value);
        uint maxValue = imageAtomicMax(image, pixel, value);
        uint andValue = imageAtomicAnd(image, pixel, value);
        uint orValue = imageAtomicOr(image, pixel, value);
        uint xorValue = imageAtomicXor(image, pixel, value);
        uint exchanged = imageAtomicExchange(image, pixel, replacement);
        uint swapped = imageAtomicCompSwap(image, pixel, exchanged, value);
        return added
            + minValue
            + maxValue
            + andValue
            + orValue
            + xorValue
            + exchanged
            + swapped;
    }

    int touchSigned(
        iimage2D image @r32i,
        ivec2 pixel,
        int value,
        int replacement
    ) {
        int added = imageAtomicAdd(image, pixel, value);
        int minValue = imageAtomicMin(image, pixel, value);
        int maxValue = imageAtomicMax(image, pixel, value);
        int exchanged = imageAtomicExchange(image, pixel, replacement);
        int swapped = imageAtomicCompSwap(image, pixel, exchanged, value);
        return added + minValue + maxValue + exchanged + swapped;
    }

    uint touchVolume(
        uimage3D image @r32ui,
        ivec3 voxel,
        uint value,
        uint replacement
    ) {
        uint added = imageAtomicAdd(image, voxel, value);
        uint swapped = imageAtomicCompSwap(image, voxel, added, replacement);
        return added + swapped;
    }

    int touchLayers(
        iimage2DArray image @r32i,
        ivec3 pixelLayer,
        int value,
        int replacement
    ) {
        int minValue = imageAtomicMin(image, pixelLayer, value);
        int swapped = imageAtomicCompSwap(image, pixelLayer, minValue, replacement);
        return minValue + swapped;
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(1, 2);
            ivec3 voxel = ivec3(1, 2, 3);
            ivec3 pixelLayer = ivec3(4, 5, 0);
            uint unsignedTotal = touchUnsigned(counters, pixel, 3u, 4u);
            int signedTotal = touchSigned(signedCounters, pixel, -3, 4);
            uint volumeTotal = touchVolume(volumeCounters, voxel, 5u, 6u);
            int layerTotal = touchLayers(layerCounters, pixelLayer, -5, 6);
            imageStore(counters, pixel, unsignedTotal + volumeTotal);
            imageStore(signedCounters, pixel, signedTotal + layerTotal);
        }
    }
}
"""


METAL_RESOURCE_ARRAY_IMAGE_ATOMICS_COMPUTE_SHADER = """
shader MetalResourceArrayImageAtomicsValidation {
    uimage2D counters @r32ui[2];
    iimage2D signedCounters @r32i[2];

    uint addCounter(
        uimage2D images[2] @r32ui,
        int index,
        ivec2 pixel,
        uint value
    ) {
        return imageAtomicAdd(images[index], pixel, value);
    }

    int swapSigned(
        iimage2D images[2] @r32i,
        int index,
        ivec2 pixel,
        int expected,
        int value
    ) {
        return imageAtomicCompSwap(images[index], pixel, expected, value);
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            uint oldValue = addCounter(counters, 1, pixel, 2u);
            int oldSigned = swapSigned(signedCounters, 0, pixel, 3, int(oldValue));
            imageStore(counters[0], pixel, oldValue + uint(oldSigned));
        }
    }
}
"""


METAL_STORAGE_IMAGE_ACCESS_QUALIFIERS_COMPUTE_SHADER = """
shader MetalStorageImageAccessQualifiersValidation {
    image2D source @rgba32f @readonly;
    image2D target @rgba32f @writeonly;
    uimage2D counters @r32ui @readwrite;

    vec4 readSource(image2D image @rgba32f @readonly, ivec2 pixel) {
        return imageLoad(image, pixel);
    }

    void writeTarget(image2D image @rgba32f @writeonly, ivec2 pixel, vec4 value) {
        imageStore(image, pixel, value);
    }

    uint addCounter(uimage2D image @r32ui @readwrite, ivec2 pixel, uint value) {
        return imageAtomicAdd(image, pixel, value);
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            vec4 color = readSource(source, pixel);
            uint oldValue = addCounter(counters, pixel, 2u);
            writeTarget(target, pixel, color + vec4(float(oldValue)));
        }
    }
}
"""


METAL_STRUCT_HELD_STORAGE_IMAGE_ARRAYS_COMPUTE_SHADER = """
shader MetalStructHeldStorageImageArraysValidation {
    struct ImagePack {
        image2D pairs[2] @rg16f @readonly;
        image2D targets[2] @rg16f @writeonly;
        uimage2D counters[2] @r32ui @readwrite;
    };

    vec2 readPair(ImagePack pack, int index, ivec2 pixel) {
        return imageLoad(pack.pairs[index], pixel);
    }

    void writePair(ImagePack pack, int index, ivec2 pixel, vec2 value) {
        imageStore(pack.targets[index], pixel, value);
    }

    uint addCounter(ImagePack pack, int index, ivec2 pixel, uint value) {
        return imageAtomicAdd(pack.counters[index], pixel, value);
    }

    vec2 readPairAlias(ImagePack pack, int index, ivec2 pixel) {
        let pairsAlias = pack.pairs;
        let chainedPairsAlias = pairsAlias;
        return imageLoad(chainedPairsAlias[index], pixel);
    }

    void writePairAlias(ImagePack pack, int index, ivec2 pixel, vec2 value) {
        let targetsAlias = pack.targets;
        let chainedTargetsAlias = targetsAlias;
        imageStore(chainedTargetsAlias[index], pixel, value);
    }

    uint addCounterAlias(ImagePack pack, int index, ivec2 pixel, uint value) {
        let countersAlias = pack.counters;
        let counterAlias = countersAlias[index];
        return imageAtomicAdd(counterAlias, pixel, value);
    }

    compute {
        void main() {
            ImagePack pack;
            ivec2 pixel = ivec2(0, 1);
            vec2 pair = readPair(pack, 0, pixel);
            writePair(pack, 1, pixel, pair);
            uint oldValue = addCounter(pack, 0, pixel, 1u);
            vec2 aliasPair = readPairAlias(pack, 1, pixel);
            writePairAlias(pack, 0, pixel, aliasPair);
            uint aliasOldValue = addCounterAlias(pack, 1, pixel, oldValue);
        }
    }
}
"""


METAL_MULTISAMPLE_STORAGE_IMAGE_ALIASES_COMPUTE_SHADER = """
shader MetalMultisampleStorageImageAliasesValidation {
    image2DMS colorImage @rgba16f;
    uimage2DMS counterImage @r32ui;
    image2DMS images @rgba16f[2];
    uimage2DMSArray layerCounters @r32ui[2];

    vec4 directAliases(
        image2DMS color @rgba16f,
        uimage2DMS counter @r32ui,
        ivec2 pixel,
        int sampleIndex
    ) {
        let colorAlias = color;
        let chainedColorAlias = colorAlias;
        let counterAlias = counter;
        vec4 colorValue = imageLoad(chainedColorAlias, pixel, sampleIndex);
        uint counterValue = imageLoad(counterAlias, pixel, sampleIndex);
        imageStore(chainedColorAlias, pixel, sampleIndex, colorValue);
        uint oldCounter = imageAtomicAdd(counterAlias, pixel, sampleIndex, counterValue);
        return colorValue + vec4(float(counterValue + oldCounter));
    }

    vec4 arrayAliases(
        image2DMS imageArray[2] @rgba16f,
        uimage2DMSArray counterArray[2] @r32ui,
        int index,
        ivec2 pixel,
        int layer,
        int sampleIndex
    ) {
        let imagesAlias = imageArray;
        let imageAlias = imagesAlias[index];
        let countersAlias = counterArray;
        let counterAlias = countersAlias[index];
        ivec3 pixelLayer = ivec3(pixel, layer);
        vec4 imageValue = imageLoad(imageAlias, pixel, sampleIndex);
        uint counterValue = imageLoad(counterAlias, pixelLayer, sampleIndex);
        imageStore(imageAlias, pixel, sampleIndex, imageValue);
        uint oldCounter = imageAtomicAdd(counterAlias, pixelLayer, sampleIndex, counterValue);
        return imageValue + vec4(float(counterValue + oldCounter));
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            vec4 directValue = directAliases(colorImage, counterImage, pixel, 0);
            vec4 arrayValue = arrayAliases(images, layerCounters, 1, pixel, 2, 0);
            if ((directValue.x + arrayValue.x) < -1.0) {
                return;
            }
        }
    }
}
"""


METAL_STORAGE_IMAGE_QUERY_DIAGNOSTICS_COMPUTE_SHADER = """
shader MetalStorageImageQueryDiagnosticsValidation {
    image2D colorImage @rgba32f;
    image3D volumeImage @rgba32f;
    image2DArray layerImage @rgba32f;
    image2D target @rgba32f @writeonly;

    int imageLevels(image2D image @rgba32f) {
        return textureQueryLevels(image);
    }

    vec2 imageLod(image2D image @rgba32f, vec2 uv) {
        return textureQueryLod(image, uv);
    }

    int imageSampleCount(image2D image @rgba32f) {
        return imageSamples(image) + textureSamples(image);
    }

    int volumeLevels(image3D image @rgba32f) {
        return textureQueryLevels(image);
    }

    int layerLevels(image2DArray image @rgba32f) {
        return textureQueryLevels(image);
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            vec2 uv = vec2(0.25, 0.75);
            vec3 uvw = vec3(0.25, 0.75, 1.0);
            int levels = imageLevels(colorImage)
                + volumeLevels(volumeImage)
                + layerLevels(layerImage)
                + textureQueryLevels(colorImage);
            vec2 lod = imageLod(colorImage, uv)
                + textureQueryLod(volumeImage, uvw);
            int samples = imageSampleCount(colorImage)
                + imageSamples(volumeImage)
                + textureSamples(layerImage);
            imageStore(target, pixel, vec4(float(levels + samples) + lod.x));
        }
    }
}
"""


RG_IMAGE_ARRAY_COMPUTE_SHADER = """
shader RGImageArrayValidation {
    image2D rgFloatImages @rg32f[3];
    uimage2D rgUnsignedImages @rg32ui[2];

    float scalarFloat(image2D images[3] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[1], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloat(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[1], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(images[1], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
            vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
            uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
            uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


RG_INFERRED_IMAGE_ARRAY_COMPUTE_SHADER = """
shader RGImageArrayInferredValidation {
    const int COUNT = 3;
    const int LAYER = COUNT - 1;
    image2D rgFloatImages @rg32f[];
    uimage2D rgUnsignedImages @rg32ui[COUNT];
    image2D afterImages @rg32f;

    float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[LAYER], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloat(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[LAYER], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
            vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
            uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
            uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


RG_TRANSITIVE_IMAGE_ARRAY_COMPUTE_SHADER = """
shader TransitiveRGImageArrayValidation {
    image2D rgFloatImages @rg32f[];
    uimage2D rgUnsignedImages @rg32ui[];

    float scalarFloatDeep(image2D images[] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[3], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    float scalarFloatMid(image2D images[] @rg32f, ivec2 pixel, float value) {
        return scalarFloatDeep(images, pixel, value);
    }

    vec2 vectorFloatDeep(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloatMid(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
        return vectorFloatDeep(images, pixel, value);
    }

    uint scalarUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[3], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
        return scalarUnsignedDeep(images, pixel, value);
    }

    uvec2 vectorUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
        return vectorUnsignedDeep(images, pixel, value);
    }

    compute {
        void main() {
            float sf = scalarFloatMid(rgFloatImages, ivec2(0, 1), 0.25);
            vec2 vf = vectorFloatMid(rgFloatImages, ivec2(2, 3), vec2(1.0));
            uint su = scalarUnsignedMid(rgUnsignedImages, ivec2(4, 5), 7u);
            uvec2 vu = vectorUnsignedMid(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


RG_FIXED_PARAM_IMAGE_ARRAY_COMPUTE_SHADER = """
shader FixedParamRGImageArrayValidation {
    image2D rgFloatImages @rg32f[];
    uimage2D rgUnsignedImages @rg32ui[];

    float scalarFloatFixed(image2D images[4] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[3], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloatFixed(image2D images[4] @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[3], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
            vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
            uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
            uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


RG_FIXED_CONST_PARAM_IMAGE_ARRAY_COMPUTE_SHADER = """
shader FixedConstParamRGImageArrayValidation {
    const int COUNT = 4;
    const int LAST = COUNT - 1;
    image2D rgFloatImages @rg32f[];
    uimage2D rgUnsignedImages @rg32ui[];

    float scalarFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[LAST], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[LAST], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
            vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
            uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
            uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


RG_FIXED_EXPR_PARAM_IMAGE_ARRAY_COMPUTE_SHADER = """
shader FixedExprParamRGImageArrayValidation {
    const int COUNT = 3;
    const int UINT_COUNT = 2;
    image2D rgFloatImages @rg32f[];
    uimage2D rgUnsignedImages @rg32ui[];

    float scalarFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[COUNT], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    vec2 vectorFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, vec2 value) {
        vec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uint scalarUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[3], pixel);
        imageStore(images[1], pixel, oldValue + value);
        return oldValue;
    }

    uvec2 vectorUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uvec2 value) {
        uvec2 oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
            vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
            uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
            uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
        }
    }
}
"""


RG_DIRECT_INDEX_WITHIN_FIXED_IMAGE_ARRAY_COMPUTE_SHADER = """
shader DirectIndexWithinFixedValidation {
    image2D rgFloatImages @rg32f[];
    uimage2D rgUnsignedImages @rg32ui[];

    float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[3], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uint touchUnsignedFour(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[3], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            float directFloat = imageLoad(rgFloatImages[2], pixel);
            uint directUint = imageLoad(rgUnsignedImages[1], pixel);
            float helperFloat = touchFour(rgFloatImages, pixel, directFloat);
            uint helperUint = touchUnsignedFour(rgUnsignedImages, pixel, directUint);
        }
    }
}
"""


RG_FIXED_GLOBAL_TO_UNSIZED_IMAGE_ARRAY_COMPUTE_SHADER = """
shader FixedGlobalToUnsizedValidation {
    image2D rgFloatImages @rg32f[4];
    uimage2D rgUnsignedImages @rg32ui[4];

    float touchFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
        float oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    uint touchUnsigned(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
        uint oldValue = imageLoad(images[2], pixel);
        imageStore(images[0], pixel, oldValue + value);
        return oldValue;
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            float sf = touchFloat(rgFloatImages, pixel, 0.25);
            uint su = touchUnsigned(rgUnsignedImages, pixel, 7u);
        }
    }
}
"""


RG_FIXED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER = """
shader FixedConstIndexValidation {
    const int COUNT = 4;
    image2D rgFloatImages @rg32f[4];
    uimage2D rgUnsignedImages @rg32ui[4];

    float touchFloat(image2D images[4] @rg32f, ivec2 pixel) {
        return imageLoad(images[COUNT - 1], pixel);
    }

    uint touchUnsigned(uimage2D images[4] @rg32ui, ivec2 pixel) {
        return imageLoad(images[COUNT - 1], pixel);
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            float directFloat = imageLoad(rgFloatImages[COUNT - 1], pixel);
            uint directUint = imageLoad(rgUnsignedImages[COUNT - 1], pixel);
            float helperFloat = touchFloat(rgFloatImages, pixel);
            uint helperUint = touchUnsigned(rgUnsignedImages, pixel);
        }
    }
}
"""


RG_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER = """
shader FixedShadowedConstIndexValidation {
    const int COUNT = 4;
    image2D rgFloatImages @rg32f[4];
    uimage2D rgUnsignedImages @rg32ui[4];

    float touchFloat(image2D images[4] @rg32f, ivec2 pixel) {
        int COUNT = 0;
        return imageLoad(images[COUNT], pixel);
    }

    uint touchUnsigned(uimage2D images[4] @rg32ui, ivec2 pixel) {
        int COUNT = 0;
        return imageLoad(images[COUNT], pixel);
    }

    compute {
        void main() {
            int COUNT = 0;
            ivec2 pixel = ivec2(0, 1);
            float directFloat = imageLoad(rgFloatImages[COUNT], pixel);
            uint directUint = imageLoad(rgUnsignedImages[COUNT], pixel);
            float helperFloat = touchFloat(rgFloatImages, pixel);
            uint helperUint = touchUnsigned(rgUnsignedImages, pixel);
        }
    }
}
"""


RG_TRANSITIVE_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER = """
shader TransitiveShadowedConstIndexValidation {
    const int COUNT = 4;
    image2D rgFloatImages @rg32f[4];
    uimage2D rgUnsignedImages @rg32ui[4];

    float leafFloat(image2D images[] @rg32f, ivec2 pixel) {
        int COUNT = 0;
        return imageLoad(images[COUNT], pixel);
    }

    uint leafUnsigned(uimage2D images[] @rg32ui, ivec2 pixel) {
        int COUNT = 0;
        return imageLoad(images[COUNT], pixel);
    }

    float passFloat(image2D images[] @rg32f, ivec2 pixel) {
        int COUNT = 0;
        float sampled = imageLoad(images[COUNT], pixel);
        return sampled + leafFloat(images, pixel);
    }

    uint passUnsigned(uimage2D images[] @rg32ui, ivec2 pixel) {
        int COUNT = 0;
        uint sampled = imageLoad(images[COUNT], pixel);
        return sampled + leafUnsigned(images, pixel);
    }

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            float sf = passFloat(rgFloatImages, pixel);
            uint su = passUnsigned(rgUnsignedImages, pixel);
        }
    }
}
"""


def run_validator(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"{' '.join(command)} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def validate_spirv_shader_source(
    tmp_path, stem, shader_source, validator_args=None, target_env=None
):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    source = tmp_path / f"{stem}.spvasm"
    output = tmp_path / f"{stem}.spv"
    code = VulkanSPIRVCodeGen().generate(crosstl.translator.parse(shader_source))
    assert "WARNING" not in code
    source.write_text(code, encoding="utf-8")

    target_args = ["--target-env", target_env] if target_env is not None else []
    run_validator([spirv_as, *target_args, str(source), "-o", str(output)])
    run_validator([spirv_val, *target_args, *(validator_args or []), str(output)])


def metal_supports_mesh_object_stage_attributes(xcrun, tmp_path):
    source = tmp_path / "metal_mesh_object_probe.metal"
    output = tmp_path / "metal_mesh_object_probe.air"
    source.write_text(
        """
#include <metal_stdlib>
using namespace metal;

[[object]]
void object_main() { }

[[mesh]]
void mesh_main() { }
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )


def test_generated_spirv_complex_resource_compute_validates_with_spirv_tools(
    tmp_path,
):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    source = tmp_path / "complex_resource_compute.spvasm"
    output = tmp_path / "complex_resource_compute.spv"
    code = VulkanSPIRVCodeGen().generate(
        crosstl.translator.parse(SPIRV_COMPLEX_RESOURCE_COMPUTE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator([spirv_as, str(source), "-o", str(output)])
    run_validator([spirv_val, str(output)])


def test_generated_spirv_synchronization_compute_validates_with_spirv_tools(
    tmp_path,
):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    source = tmp_path / "synchronization_compute.spvasm"
    output = tmp_path / "synchronization_compute.spv"
    code = VulkanSPIRVCodeGen().generate(
        crosstl.translator.parse(SPIRV_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator([spirv_as, str(source), "-o", str(output)])
    run_validator([spirv_val, str(output)])


def test_generated_spirv_wave_intrinsics_compute_validates_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "wave_intrinsics_compute",
        SPIRV_WAVE_INTRINSICS_COMPUTE_SHADER,
        target_env="vulkan1.1",
    )


def test_generated_spirv_uniform_buffer_compute_validates_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "uniform_buffer_compute",
        SPIRV_UNIFORM_BUFFER_COMPUTE_SHADER,
    )


def test_generated_spirv_structured_buffer_compute_validates_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "structured_buffer_compute",
        SPIRV_STRUCTURED_BUFFER_COMPUTE_SHADER,
    )


def test_generated_spirv_glsl_buffer_block_compute_validates_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "glsl_buffer_block_compute",
        SPIRV_GLSL_BUFFER_BLOCK_COMPUTE_SHADER,
    )


def test_generated_spirv_scalar_buffer_block_compute_validates_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "scalar_buffer_block_compute",
        SPIRV_SCALAR_BUFFER_BLOCK_COMPUTE_SHADER,
        validator_args=["--scalar-block-layout"],
    )


def test_generated_spirv_glsl_buffer_block_arrays_validate_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "glsl_buffer_block_array_compute",
        SPIRV_GLSL_BUFFER_BLOCK_ARRAY_COMPUTE_SHADER,
        validator_args=["--scalar-block-layout"],
        target_env="vulkan1.1",
    )


def test_generated_spirv_storage_buffer_atomics_validate_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "storage_buffer_atomics_compute",
        SPIRV_STORAGE_BUFFER_ATOMICS_COMPUTE_SHADER,
    )


def test_generated_spirv_resource_memory_qualifiers_validate_with_spirv_tools(
    tmp_path,
):
    validate_spirv_shader_source(
        tmp_path,
        "resource_memory_qualifier_compute",
        SPIRV_RESOURCE_MEMORY_QUALIFIER_COMPUTE_SHADER,
    )


def test_generated_spirv_integer_image_atomics_validates_with_spirv_tools(
    tmp_path,
):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    source = tmp_path / "integer_image_atomics.spvasm"
    output = tmp_path / "integer_image_atomics.spv"
    code = VulkanSPIRVCodeGen().generate(
        crosstl.translator.parse(INTEGER_IMAGE_ATOMICS_COMPUTE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator([spirv_as, str(source), "-o", str(output)])
    run_validator([spirv_val, str(output)])


def test_generated_spirv_forwarded_image_atomic_validates_with_spirv_tools(
    tmp_path,
):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    source = tmp_path / "forwarded_image_atomic.spvasm"
    output = tmp_path / "forwarded_image_atomic.spv"
    code = VulkanSPIRVCodeGen().generate(
        crosstl.translator.parse(SPIRV_IMAGE_ATOMIC_FORWARDING_COMPUTE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator([spirv_as, str(source), "-o", str(output)])
    run_validator([spirv_val, str(output)])


@pytest.mark.parametrize(
    ("stem", "shader_source"),
    [
        ("advanced_texture_compute", SPIRV_ADVANCED_TEXTURE_COMPUTE_SHADER),
        ("texture_query_compute", SPIRV_TEXTURE_QUERY_COMPUTE_SHADER),
        ("shadow_texture_compute", SPIRV_SHADOW_TEXTURE_COMPUTE_SHADER),
        ("projected_texture_compute", SPIRV_PROJECTED_TEXTURE_COMPUTE_SHADER),
    ],
)
def test_generated_spirv_texture_operations_validate_with_spirv_tools(
    tmp_path,
    stem,
    shader_source,
):
    validate_spirv_shader_source(tmp_path, stem, shader_source)


def test_generated_spirv_codegen_examples_validate_with_spirv_tools(tmp_path):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    def expects_codegen_error(function_node):
        for child in ast.walk(function_node):
            if not isinstance(child, ast.With):
                continue
            for item in child.items:
                context_expr = item.context_expr
                if not isinstance(context_expr, ast.Call):
                    continue
                callee = context_expr.func
                if isinstance(callee, ast.Attribute) and callee.attr == "raises":
                    return True
        return False

    test_file = Path(__file__).with_name("test_SPIRV_codegen.py")
    module = ast.parse(test_file.read_text(encoding="utf-8"))
    failures = []
    validated = 0

    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if expects_codegen_error(node):
            continue

        source_code = None
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "source_code"
                for target in stmt.targets
            ):
                continue
            if isinstance(stmt.value, ast.Constant) and isinstance(
                stmt.value.value, str
            ):
                source_code = stmt.value.value
                break

        if source_code is None or "shader" not in source_code:
            continue

        code = VulkanSPIRVCodeGen().generate(crosstl.translator.parse(source_code))
        if "OpEntryPoint " not in code:
            continue

        source = tmp_path / f"{node.name}.spvasm"
        output = tmp_path / f"{node.name}.spv"
        source.write_text(code, encoding="utf-8")

        assemble = subprocess.run(
            [spirv_as, str(source), "-o", str(output)],
            capture_output=True,
            text=True,
        )
        if assemble.returncode != 0:
            failures.append((node.name, "spirv-as", assemble.stderr))
            continue

        validate = subprocess.run(
            [spirv_val, str(output)],
            capture_output=True,
            text=True,
        )
        if validate.returncode != 0:
            failures.append((node.name, "spirv-val", validate.stderr))
            continue

        validated += 1

    assert not failures, "\n".join(
        f"{name} failed {stage}:\n{message}" for name, stage, message in failures
    )
    assert validated >= 50


def dxc_supports_sample_cmp_lod_grad(dxc, tmp_path):
    source = tmp_path / "sample_cmp_lod_grad_probe.hlsl"
    output = tmp_path / "sample_cmp_lod_grad_probe.dxil"
    source.write_text(
        """
Texture2D shadowMap : register(t0);
SamplerComparisonState shadowSampler : register(s0);

	float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
	    float2 ddx = float2(0.1, 0.0);
	    float2 ddy = float2(0.0, 0.1);
	    int2 offset = int2(1, 0);
	    float lodValue = shadowMap.SampleCmpLevel(shadowSampler, uv, 0.5, 1.0);
	    float lodOffsetValue = shadowMap.SampleCmpLevel(shadowSampler, uv, 0.5, 1.0, offset);
	    float gradValue = shadowMap.SampleCmpGrad(shadowSampler, uv, 0.5, ddx, ddy);
	    float gradOffsetValue = shadowMap.SampleCmpGrad(shadowSampler, uv, 0.5, ddx, ddy, offset);
	    return float4(lodValue + lodOffsetValue + gradValue + gradOffsetValue);
	}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [dxc, "-T", "ps_6_7", "-E", "PSMain", str(source), "-Fo", str(output)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_generated_metal_fragment_smoke_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_range.metal"
    output = tmp_path / "fragment_range.air"
    code = MetalCodeGen().generate(crosstl.translator.parse(FRAGMENT_RANGE_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_function_constants_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "function_constants.metal"
    output = tmp_path / "function_constants.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_FUNCTION_CONSTANT_FRAGMENT_SHADER),
        "fragment",
    )
    assert (
        "/* unsupported Metal function constant default: 'useFast' initializers "
        "are not allowed by MSL */"
    ) in code
    assert "constant bool useFast [[function_constant(0)]];" in code
    assert "constant int mode [[function_constant(1)]];" in code
    assert "constant float scale [[function_constant(2)]];" in code
    assert "constant uint flags [[function_constant(3)]];" in code
    assert "function constant default: 'flags'" not in code
    assert "bool useFast;" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_wave_intrinsics_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "wave_intrinsics.metal"
    output = tmp_path / "wave_intrinsics.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_WAVE_INTRINSICS_COMPUTE_SHADER),
        "compute",
    )
    assert "simd_sum(value)" in code
    assert "simd_all(value == simd_broadcast_first(value))" in code
    assert "simd_all(all(lanes == simd_broadcast_first(lanes)))" in code
    assert "uint crossglWaveLaneIndex [[thread_index_in_simdgroup]]" in code
    assert "uint crossglWaveLaneCount [[threads_per_simdgroup]]" in code
    assert "uint lane = crossglWaveLaneIndex;" in code
    assert "uint laneCount = crossglWaveLaneCount;" in code
    assert "uint helperLane(uint seed, uint crossglWaveLaneIndex)" in code
    assert (
        "uint helperBoth(uint seed, uint crossglWaveLaneIndex, "
        "uint crossglWaveLaneCount)" in code
    )
    assert "uint4 helperMatch(uint seed, uint crossglWaveLaneCount)" in code
    assert (
        "uint helperMulti(uint seed, uint4 mask, uint crossglWaveLaneIndex, "
        "uint crossglWaveLaneCount)" in code
    )
    assert (
        "uint2 helperMultiVector(uint2 seed, uint4 mask, uint crossglWaveLaneIndex, "
        "uint crossglWaveLaneCount)" in code
    )
    assert "helperBoth(value, crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    assert "helperMatch(value, crossglWaveLaneCount)" in code
    assert (
        "helperMulti(value, ballot, crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    )
    assert (
        "helperMultiVector(lanes, ballot, crossglWaveLaneIndex, "
        "crossglWaveLaneCount)" in code
    )
    assert "__crossgl_metal_wave_ballot(anyLane)" in code
    assert "__crossgl_metal_wave_match(value, crossglWaveLaneCount)" in code
    assert (
        "__crossgl_metal_wave_multi_prefix_sum(value, ballot, "
        "crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    )
    assert (
        "__crossgl_metal_wave_multi_prefix_count_bits(anyLane, ballot, "
        "crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    )
    assert (
        "__crossgl_metal_wave_multi_prefix_bit_xor(value, ballot, "
        "crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    )
    assert (
        "__crossgl_metal_wave_multi_prefix_sum(lanes, ballot, "
        "crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    )
    assert (
        "__crossgl_metal_wave_multi_prefix_bit_xor(lanes, ballot, "
        "crossglWaveLaneIndex, crossglWaveLaneCount)" in code
    )
    assert (
        "WaveMultiPrefixSum value argument must be numeric scalar or vector, got float2x2"
        in code
    )
    assert "__crossgl_metal_wave_multi_prefix_sum(matrixValue" not in code
    assert "quad_shuffle_xor(firstValue, ushort(1))" in code
    assert "WaveActiveSum(value)" not in code
    assert "WaveActiveAllEqual(value)" not in code
    assert "WaveActiveAllEqual(lanes)" not in code
    assert "WaveOpNode" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_switch_match_case_scope_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_switch_match_case_scope.metal"
    output = tmp_path / "fragment_switch_match_case_scope.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_CASE_SCOPE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_switch_match_texture_case_scope_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_switch_match_texture_case_scope.metal"
    output = tmp_path / "fragment_switch_match_texture_case_scope.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_TEXTURE_CASE_SCOPE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_sampled_texture_array_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_sampled_texture_array.metal"
    output = tmp_path / "fragment_sampled_texture_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLED_TEXTURE_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_implicit_sampler_array_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_implicit_sampler_array.metal"
    output = tmp_path / "fragment_implicit_sampler_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(IMPLICIT_SAMPLER_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    assert "textures[index].sample(texturesSampler[index], uv)" in code
    assert "textures[index].sample(texturesSampler, uv)" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_local_alias_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_local_alias.metal"
    output = tmp_path / "fragment_texture_local_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_LOCAL_ALIAS_FRAGMENT_SHADER),
        "fragment",
    )
    assert "alias.sample(colorMapSampler, input.uv)" in code
    assert "alias.sample(sampleState, input.uv)" in code
    assert "layerAlias.sample(texturesSampler[input.layer], input.uv)" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_resource_array_local_aliases_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    fragment_source = tmp_path / "fragment_resource_array_local_alias.metal"
    fragment_output = tmp_path / "fragment_resource_array_local_alias.air"
    fragment_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RESOURCE_TEXTURE_ARRAY_LOCAL_ALIAS_FRAGMENT_SHADER),
        "fragment",
    )
    assert "array<texture2d<float>, 4> texAlias = textures;" in fragment_code
    assert (
        "texAlias[input.layer].sample(texturesSampler[input.layer], input.uv)"
        in fragment_code
    )
    assert "chainedAlias[layer].sample(paramTexturesSampler[layer], uv)" in (
        fragment_code
    )
    fragment_source.write_text(fragment_code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(fragment_source),
            "-o",
            str(fragment_output),
        ]
    )

    compute_source = tmp_path / "compute_resource_array_local_alias.metal"
    compute_output = tmp_path / "compute_resource_array_local_alias.air"
    compute_code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RESOURCE_IMAGE_ARRAY_LOCAL_ALIAS_COMPUTE_SHADER),
        "compute",
    )
    assert (
        "array<texture2d<float, access::read_write>, 4> imageAlias = images;"
        in compute_code
    )
    assert (
        "array<texture2d<float, access::read_write>, 4> chainedAlias = paramAlias;"
        in compute_code
    )
    compute_source.write_text(compute_code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(compute_source),
            "-o",
            str(compute_output),
        ]
    )


def test_generated_metal_resource_array_element_helpers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_resource_array_element_helpers.metal"
    output = tmp_path / "compute_resource_array_element_helpers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_RESOURCE_ARRAY_ELEMENT_HELPER_COMPUTE_SHADER),
        "compute",
    )
    assert "sampleOne(textures[layer], samplers[layer], float2(0.5))" in code
    assert "sampleArray(textures, samplers, layer, float2(0.25))" in code
    assert "readOne(images[layer], int2(0, 0))" in code
    assert "array<texture2d<float>, 4> texs" in code
    assert "array<sampler, 4> samps" in code
    assert "array<texture2d<float, access::read_write>, 4> images" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_sampler_array_local_aliases_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_sampler_array_local_alias.metal"
    output = tmp_path / "fragment_sampler_array_local_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLER_ARRAY_LOCAL_ALIAS_FRAGMENT_SHADER),
        "fragment",
    )
    assert "array<sampler, 4> samplerAlias = samplers;" in code
    assert "array<sampler, 4> chainedAlias = samplerAlias;" in code
    assert "array<sampler, 4> paramSamplerAlias = paramSamplers;" in code
    assert "array<sampler, 4> chainedSamplerAlias = paramSamplerAlias;" in code
    assert "array<sampler, 4> packSamplerAlias = pack.samplers;" in code
    assert "array<sampler, 4> chainedPackSamplerAlias = packSamplerAlias;" in code
    assert "textures[input.layer].sample(chainedAlias[input.layer], input.uv)" in code
    assert "paramTextures[layer].sample(chainedSamplerAlias[layer], uv)" in code
    assert "pack.textures[layer].sample(chainedPackSamplerAlias[layer], uv)" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_struct_member_resource_array_local_aliases_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_struct_member_resource_array_local_alias.metal"
    output = tmp_path / "fragment_struct_member_resource_array_local_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            STRUCT_MEMBER_TEXTURE_ARRAY_LOCAL_ALIAS_FRAGMENT_SHADER
        ),
        "fragment",
    )
    assert "array<texture2d<float>, 4> texAlias = pack.textures;" in code
    assert "array<texture2d<float>, 4> chainedAlias = texAlias;" in code
    assert "chainedAlias[layer].sample(pack.texturesSampler[layer], uv)" in code
    assert "texture2d<float> layerAlias = pack.textures[layer];" in code
    assert "layerAlias.sample(pack.texturesSampler[layer], uv)" in code
    assert "sampler(mag_filter::linear, min_filter::linear)" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_sampled_texture_const_index_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_sampled_texture_const_index.metal"
    output = tmp_path / "fragment_sampled_texture_const_index.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLED_TEXTURE_ARRAY_CONST_INDEX_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_sampled_texture_transitive_shadowed_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_sampled_texture_transitive_shadowed.metal"
    output = tmp_path / "fragment_sampled_texture_transitive_shadowed.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            SAMPLED_TEXTURE_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_shadow_sampler_transitive_shadowed_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_shadow_sampler_transitive_shadowed.metal"
    output = tmp_path / "fragment_shadow_sampler_transitive_shadowed.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            SHADOW_SAMPLER_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_array_shadow_texture_resource_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_array_shadow_texture_resource_array.metal"
    output = tmp_path / "fragment_array_shadow_texture_resource_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_SHADOW_TEXTURE_RESOURCE_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_array_shadow_texture_query_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_array_shadow_texture_query.metal"
    output = tmp_path / "fragment_array_shadow_texture_query.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_SHADOW_TEXTURE_QUERY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_array_texture_query_lod_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_array_texture_query_lod.metal"
    output = tmp_path / "fragment_array_texture_query_lod.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_shadow_array_texture_query_lod_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_shadow_array_texture_query_lod.metal"
    output = tmp_path / "fragment_shadow_array_texture_query_lod.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_implicit_shadow_array_texture_query_lod_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_implicit_shadow_array_texture_query_lod.metal"
    output = tmp_path / "fragment_implicit_shadow_array_texture_query_lod.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            IMPLICIT_SHADOW_ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_cube_array_texture_grad_gather_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_cube_array_texture_grad_gather.metal"
    output = tmp_path / "fragment_cube_array_texture_grad_gather.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(CUBE_ARRAY_TEXTURE_GRAD_GATHER_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_gradient_family_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_gradient_family.metal"
    output = tmp_path / "fragment_texture_gradient_family.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_GRADIENT_FAMILY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_gather_offset_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_gather_offset.metal"
    output = tmp_path / "fragment_texture_gather_offset.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_GATHER_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_sample_offset_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_sample_offset.metal"
    output = tmp_path / "fragment_texture_sample_offset.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_SAMPLE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_3d_sample_offset_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_3d_sample_offset.metal"
    output = tmp_path / "fragment_texture_3d_sample_offset.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_3D_SAMPLE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_projection_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_projection.metal"
    output = tmp_path / "fragment_texture_projection.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_PROJECTED_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_texture_3d_projected_offset_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_texture_3d_projected_offset.metal"
    output = tmp_path / "fragment_texture_3d_projected_offset.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_TEXTURE_3D_PROJECTED_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_shadow_gather_compare_offset_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_shadow_gather_compare_offset.metal"
    output = tmp_path / "fragment_shadow_gather_compare_offset.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_GATHER_COMPARE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_shadow_compare_lod_grad_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_shadow_compare_lod_grad.metal"
    output = tmp_path / "fragment_shadow_compare_lod_grad.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_COMPARE_LOD_GRAD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_fragment_projected_shadow_compare_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "fragment_projected_shadow_compare.metal"
    output = tmp_path / "fragment_projected_shadow_compare.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(PROJECTED_SHADOW_COMPARE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_combined_stages_compile_separately_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    ast = crosstl.translator.parse(COMBINED_STAGE_IO_SHADER)
    generator = MetalCodeGen()

    for stage_name in ("vertex", "fragment"):
        source = tmp_path / f"combined_stage_{stage_name}.metal"
        output = tmp_path / f"combined_stage_{stage_name}.air"
        source.write_text(generator.generate_stage(ast, stage_name), encoding="utf-8")

        run_validator(
            [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
        )


def test_generated_metal_mesh_object_stages_compile_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_object.metal"
    output = tmp_path / "mesh_object.air"
    code = MetalCodeGen().generate(crosstl.translator.parse(METAL_MESH_OBJECT_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_dispatch_invalid_grid_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_dispatch_invalid_grid.metal"
    output = tmp_path / "mesh_dispatch_invalid_grid.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_DISPATCH_INVALID_GRID_SHADER)
    )
    assert (
        "unsupported Metal mesh dispatch: DispatchMesh grid argument must be "
        "a uint3-compatible vector"
    ) in code
    assert "set_threadgroups_per_grid(2)" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_output_signature_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_output_signature.metal"
    output = tmp_path / "mesh_output_signature.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_OUTPUT_SIGNATURE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "mesh<MeshVertex, MeshPrimitive, 3, 1, topology::triangle>" in code
    assert "MeshVertex verts[3]" not in code
    assert "uint3 tris[1]" not in code
    assert "MeshPrimitive prims[1]" not in code
    assert "MeshVertex _crossglMeshVertices_verts_i_0 = {};" in code
    assert "_crossglMeshVertices_verts_i_0.uv = float2(0.0);" in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_primitive_setter_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_primitive_setter.metal"
    output = tmp_path / "mesh_primitive_setter.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PRIMITIVE_SETTER_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "mesh<MeshVertex, MeshPrimitive, 3, 1, topology::triangle>" in code
    assert "_crossglMeshOut.set_vertex(0, outVertex);" in code
    assert "_crossglMeshOut.set_primitive(0, outPrimitive);" in code
    assert "_crossglMeshOut.set_index(0, uint3(0u, 1u, 2u).x);" in code
    assert "_crossglMeshOut.set_index(0, outPrimitive);" not in code
    assert "SetPrimitive" not in code
    assert "SetIndex" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_output_variable_member_writes_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_output_variable_member_writes.metal"
    output = tmp_path / "mesh_output_variable_member_writes.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_OUTPUT_VARIABLE_MEMBER_WRITES_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "MeshVertex _crossglMeshVertices_verts_vertexIndex = {};" in code
    assert "MeshPrimitive _crossglMeshPrimitives_prims_primitiveIndex = {};" in code
    assert (
        "_crossglMeshVertices_verts_vertexIndex.position += "
        "float4(0.0, 1.0, 0.0, 0.0);"
    ) in code
    assert ("_crossglMeshVertices_verts_vertexIndex.uv += float2(0.25, 0.0);") in code
    assert "_crossglMeshPrimitives_prims_primitiveIndex.layer += 3u;" in code
    assert (
        "_crossglMeshPrimitives_prims_primitiveIndex.bary += float2(0.5, 0.0);"
    ) in code
    assert (
        "_crossglMeshOut.set_vertex(vertexIndex, "
        "_crossglMeshVertices_verts_vertexIndex);"
    ) in code
    assert (
        "_crossglMeshOut.set_primitive(primitiveIndex, "
        "_crossglMeshPrimitives_prims_primitiveIndex);"
    ) in code
    assert "verts[vertexIndex]" not in code
    assert "tris[primitiveIndex]" not in code
    assert "prims[primitiveIndex]" not in code
    assert "unsupported Metal mesh output assignment" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_dispatch_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_dispatch.metal"
    output = tmp_path / "mesh_payload_dispatch.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_DISPATCH_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "object_data MeshPayload& _crossglMeshPayload [[payload]]" in code
    assert "const object_data MeshPayload& payload [[payload]]" in code
    assert "_crossglMeshPayload = payload;" in code
    assert "[[mesh_payload]]" not in code
    assert "DispatchMesh" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_helper_dispatch_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_helper_dispatch.metal"
    output = tmp_path / "mesh_payload_helper_dispatch.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_HELPER_DISPATCH_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "object_data MeshPayload& _crossglMeshPayload [[payload]]" in code
    assert "mesh_grid_properties _crossglMeshGrid" in code
    assert "dispatchOne(payload, _crossglMeshPayload, _crossglMeshGrid);" in code
    assert "launch(payload, _crossglMeshPayload, _crossglMeshGrid);" in code
    assert "_crossglMeshPayload = payload;" in code
    assert "DispatchMesh" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_member_source_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_member_source.metal"
    output = tmp_path / "mesh_payload_member_source.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_MEMBER_SOURCE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "threadgroup PayloadBlock block;" in code
    assert "issue(block.active, _crossglMeshPayload, _crossglMeshGrid);" in code
    assert "_crossglMeshPayload = payload;" in code
    assert "unsupported Metal mesh payload dispatch" not in code
    assert "DispatchMesh" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_array_source_compiles_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_array_source.metal"
    output = tmp_path / "mesh_payload_array_source.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_ARRAY_SOURCE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "threadgroup MeshPayload payloads[2];" in code
    assert "_crossglMeshPayload = payloads[1];" in code
    assert "unsupported Metal mesh payload dispatch" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_address_space_diagnostics_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_address_space.metal"
    output = tmp_path / "mesh_payload_address_space.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_ADDRESS_SPACE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "object_data Payload& payload [[payload]]" in code
    assert "const object_data Payload& payload [[payload]]" in code
    assert "unsupported Metal address-space call" in code
    assert "unsupported Metal mesh payload store" in code
    assert "mutate(payload)" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_local_aliases_compile_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_local_alias.metal"
    output = tmp_path / "mesh_payload_local_alias.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_LOCAL_ALIAS_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "object_data Payload& alias = payload;" in code
    assert "const object_data Payload& alias = payload;" in code
    assert "unsupported Metal mesh payload store" in code
    assert "\n    Payload& alias = payload;" not in code
    assert "\n    thread Payload& alias = payload;" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_pointer_aliases_compile_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_pointer_alias.metal"
    output = tmp_path / "mesh_payload_pointer_alias.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_POINTER_ALIAS_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "object_data Payload* alias = &payload;" in code
    assert "const object_data Payload* alias = &payload;" in code
    assert "alias->color = float4(1.0, 0.0, 0.0, 1.0);" in code
    assert "unsupported Metal mesh payload store" in code
    assert "\n    Payload* alias = payload;" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_pointer_helper_aliases_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_pointer_helper.metal"
    output = tmp_path / "mesh_payload_pointer_helper.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_POINTER_HELPER_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "void tint(object_data Payload* payload)" in code
    assert "float read(const object_data Payload* payload)" in code
    assert "[[object_data]]" not in code
    assert "object_data Payload* alias = &payload;" in code
    assert "const object_data Payload* alias = &payload;" in code
    assert "unsupported Metal mesh payload call" in code
    assert "unsupported Metal address-space call" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_member_pointer_helpers_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_member_pointer_helper.metal"
    output = tmp_path / "mesh_payload_member_pointer_helper.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_MEMBER_POINTER_HELPER_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "object_data Payload* ptr;" in code
    assert "wrapper.ptr = &payload;" in code
    assert "tint(wrapper.ptr);" in code
    assert "float value = read(wrapper.ptr);" in code
    assert "unsupported Metal mesh payload alias" in code
    assert "unsupported Metal mesh payload call" in code
    assert "float4 color = wrapper.ptr->color + float4(value);" in code
    assert "unsupported Metal address-space call" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_const_object_payload_diagnostic_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "const_object_payload_diagnostic.metal"
    output = tmp_path / "const_object_payload_diagnostic.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_CONST_OBJECT_PAYLOAD_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "const object_data Payload& payload [[payload]]" in code
    assert "unsupported Metal mesh payload store" in code
    assert "payload.color = float4(1.0, 0.0, 0.0, 1.0);" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_payload_invalid_sources_compile_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_payload_invalid_sources.metal"
    output = tmp_path / "mesh_payload_invalid_sources.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_PAYLOAD_INVALID_SOURCE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "unsupported Metal mesh payload dispatch" in code
    assert "device MeshPayload* devicePayloads [[buffer(0)]]" in code
    assert "constant MeshPayload* constantPayloads [[buffer(1)]]" in code
    assert "device PayloadBlock* deviceBlocks [[buffer(2)]]" in code
    assert "_crossglMeshPayload = makePayload();" not in code
    assert "_crossglMeshPayload = threadPayload;" not in code
    assert "_crossglMeshPayload = devicePayloads[0];" not in code
    assert "_crossglMeshPayload = constantPayloads[0];" not in code
    assert "_crossglMeshPayload = deviceBlocks[0].active;" not in code
    assert "_crossglMeshPayload = alias;" not in code
    assert "_crossglMeshPayload = payloads;" not in code
    assert "_crossglMeshPayload = alias[0];" in code
    assert "_crossglMeshPayload = payloads[0];" in code
    assert code.count("_crossglMeshGrid.set_threadgroups_per_grid(") == 9

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_dispatch_without_grid_context_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_dispatch_without_grid_context.metal"
    output = tmp_path / "mesh_dispatch_without_grid_context.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_DISPATCH_WITHOUT_GRID_CONTEXT_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "unsupported Metal mesh dispatch" in code
    assert "mesh_grid_properties context" in code
    assert "DispatchMesh(" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_mesh_dispatch_helper_without_grid_context_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    supported, diagnostics = metal_supports_mesh_object_stage_attributes(
        xcrun, tmp_path
    )
    if not supported:
        pytest.skip(
            "xcrun metal does not support Metal 3 mesh/object stage attributes: "
            f"{diagnostics}"
        )

    source = tmp_path / "mesh_dispatch_helper_without_grid_context.metal"
    output = tmp_path / "mesh_dispatch_helper_without_grid_context.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_MESH_DISPATCH_HELPER_WITHOUT_GRID_CONTEXT_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "unsupported Metal mesh dispatch helper call" in code
    assert "mesh_grid_properties context" in code
    assert "unsupported Metal address-space call" not in code
    assert "issue(payload)" not in code
    assert "DispatchMesh(" not in code

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_tracing_helper_trace_ray_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_tracing_helper.metal"
    output = tmp_path / "ray_tracing_helper.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_TRACING_HELPER_SHADER)
    )
    assert "instance_acceleration_structure topLevelAlias = topLevelAS;" in code
    assert "intersect(__crossgl_ray_0, topLevelAlias, 255)" in code
    assert "float topLevelAlias = topLevelAS;" not in code
    assert "unsupported Metal ray tracing intrinsic: TraceRay acceleration" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_tracing_intersection_table_trace_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_tracing_intersection_table_trace.metal"
    output = tmp_path / "ray_tracing_intersection_table_trace.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_TRACING_INTERSECTION_TABLE_TRACE_SHADER)
    )
    assert "intersect(__crossgl_ray_0, topLevelAS, 255, intersectionFunctions)" in (
        code
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_primitive_acceleration_trace_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_tracing_primitive_acceleration.metal"
    output = tmp_path / "ray_tracing_primitive_acceleration.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_TRACING_PRIMITIVE_ACCELERATION_SHADER)
    )
    assert "intersect(__crossgl_ray_0, primitiveAS, intersectionFunctions)" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_payload_trace_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_tracing_payload_trace.metal"
    output = tmp_path / "ray_tracing_payload_trace.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_TRACING_PAYLOAD_TRACE_SHADER)
    )
    assert (
        "intersect(" "__crossgl_ray_0, topLevelAS, 255, intersectionFunctions, payload)"
    ) in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_payload_diagnostic_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_tracing_payload_diagnostic.metal"
    output = tmp_path / "ray_tracing_payload_diagnostic.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_TRACING_PAYLOAD_DIAGNOSTIC_SHADER)
    )
    assert "payload forwarding requires a compatible intersection_function_table" in (
        code
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_payload_helper_address_spaces_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_payload_helper_address_spaces.metal"
    output = tmp_path / "ray_payload_helper_address_spaces.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_PAYLOAD_HELPER_ADDRESS_SPACE_SHADER)
    )
    assert "void tint(ray_data Payload& payload)" in code
    assert "tint(payload);" in code
    assert (
        "unsupported Metal address-space call: argument 'payload' uses ray_data "
        "address space but parameter 'payload' of 'rejectThreadPayload' "
        "requires thread"
    ) in code
    assert "rejectThreadPayload(payload);" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_payload_alias_address_spaces_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_payload_alias_address_spaces.metal"
    output = tmp_path / "ray_payload_alias_address_spaces.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_PAYLOAD_ALIAS_ADDRESS_SPACE_SHADER)
    )
    assert "thread Payload& threadAlias = payload;" in code
    assert "intersectionFunctions, threadAlias)" in code
    assert "device Payload& deviceAlias = external[0];" in code
    assert (
        "payload forwarding requires a thread-local payload lvalue; payload "
        "'deviceAlias' uses device address space"
    ) in code
    assert "intersectionFunctions, deviceAlias)" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_payload_member_lvalues_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_payload_member_lvalues.metal"
    output = tmp_path / "ray_payload_member_lvalues.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_PAYLOAD_MEMBER_LVALUE_SHADER)
    )
    assert "intersectionFunctions, wrapper.payload)" in code
    assert "intersectionFunctions, payloads[0])" in code
    assert "intersectionFunctions, pointer[0])" in code
    assert (
        "payload forwarding requires a thread-local struct payload lvalue; "
        "payload 'payloads' has pointer or array type"
    ) in code
    assert (
        "payload forwarding requires a thread-local struct payload lvalue; "
        "payload 'wrapper.payloads' has pointer or array type"
    ) in code
    assert (
        "payload forwarding requires a thread-local struct payload lvalue; "
        "payload 'pointer' has pointer or array type"
    ) in code
    assert (
        "payload forwarding requires a mutable thread-local struct payload "
        "lvalue; payload 'constWrapper.payload' is const-qualified"
    ) in code
    assert "intersectionFunctions, constWrapper.payload)" not in code
    assert "intersectionFunctions, payloads)" not in code
    assert "intersectionFunctions, wrapper.payloads)" not in code
    assert "intersectionFunctions, pointer)" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_payload_type_diagnostics_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_payload_type_diagnostics.metal"
    output = tmp_path / "ray_payload_type_diagnostics.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_PAYLOAD_TYPE_VALIDATION_SHADER)
    )
    assert "intersectionFunctions, payload)" in code
    assert "intersectionFunctions, wrapper.payload)" in code
    assert (
        "ray payload argument 'other' has type 'OtherPayload' but declared "
        "ray payload interface expects 'Payload'"
    ) in code
    assert (
        "ray payload argument 'wrapper.other' has type 'OtherPayload' but "
        "declared ray payload interface expects 'Payload'"
    ) in code
    assert (
        "ray payload argument 'others' has type 'OtherPayload' but declared "
        "ray payload interface expects 'Payload'"
    ) in code
    assert "intersectionFunctions, other)" not in code
    assert "intersectionFunctions, wrapper.other)" not in code
    assert "intersectionFunctions, others[0])" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_invalid_acceleration_structure_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_invalid_acceleration_structure.metal"
    output = tmp_path / "ray_invalid_acceleration_structure.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(
            METAL_RAY_TRACING_INVALID_ACCELERATION_STRUCTURE_SHADER
        )
    )
    assert (
        "acceleration structure argument 'payload' has type 'Payload' but "
        "TraceRay requires an acceleration_structure resource"
    ) in code
    assert "intersect(__crossgl_ray_0, payload" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_acceleration_structure_array_diagnostics_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_acceleration_structure_array_diagnostics.metal"
    output = tmp_path / "ray_acceleration_structure_array_diagnostics.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(
            METAL_RAY_ACCELERATION_STRUCTURE_ARRAY_DIAGNOSTIC_SHADER
        )
    )
    assert (
        "arrays of acceleration_structure are not valid Metal buffer parameters" in code
    )
    assert "acceleration structure argument 'topLevelAS' uses an" in code
    assert "acceleration structure argument 'primitiveAS' uses an" in code
    assert "acceleration structure argument 'paramAS' uses an" in code
    assert "array<instance_acceleration_structure" not in code
    assert "array<primitive_acceleration_structure" not in code
    assert ".intersect(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_callable_dispatch_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_callable_dispatch.metal"
    output = tmp_path / "ray_callable_dispatch.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_CALLABLE_DISPATCH_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_callable_invalid_explicit_table_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_callable_invalid_explicit_table.metal"
    output = tmp_path / "ray_callable_invalid_explicit_table.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_CALLABLE_INVALID_EXPLICIT_TABLE_SHADER)
    )
    assert (
        "explicit table argument 'data' must be a visible_function_table resource"
        in code
    )
    assert "data[0](data);" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_callable_alias_address_spaces_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_callable_alias_address_spaces.metal"
    output = tmp_path / "ray_callable_alias_address_spaces.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_CALLABLE_ALIAS_ADDRESS_SPACE_SHADER)
    )
    assert "thread CallableData& threadAlias = data;" in code
    assert "callables[0](threadAlias);" in code
    assert "device CallableData& deviceAlias = external[0];" in code
    assert (
        "callable data forwarding requires a thread-local callable-data "
        "lvalue; callable data 'deviceAlias' uses device address space"
    ) in code
    assert "callables[1](deviceAlias);" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_callable_pointer_deref_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_callable_pointer_deref.metal"
    output = tmp_path / "ray_callable_pointer_deref.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_CALLABLE_POINTER_DEREF_SHADER)
    )
    assert "thread CallableData* alias = &data;" in code
    assert "callables[0](*alias);" in code
    assert (
        "callable data forwarding requires a thread-local callable-data "
        "lvalue; callable data '*external' uses device address space"
    ) in code
    assert "callables[1](*external);" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_callable_helper_member_lvalues_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_callable_helper_member_lvalues.metal"
    output = tmp_path / "ray_callable_helper_member_lvalues.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_CALLABLE_HELPER_MEMBER_LVALUE_SHADER)
    )
    assert "callables[shaderIndex](data);" in code
    assert "invoke(2u, wrapper.data, callables);" in code
    assert "callables[3u](wrapper.data);" in code
    assert (
        "callable data forwarding requires a mutable thread-local callable-data "
        "lvalue; callable data 'wrapper.data' is const-qualified"
    ) in code
    assert "callables[1](wrapper.data);" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_stages_compile_with_metal3(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_stages.metal"
    output = tmp_path / "ray_stages.air"
    code = MetalCodeGen().generate(crosstl.translator.parse(METAL_RAY_STAGES_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_intersection_function_table_compiles_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "intersection_function_table.metal"
    output = tmp_path / "intersection_function_table.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_INTERSECTION_FUNCTION_TABLE_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_function_table_parameters_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_function_table_parameters.metal"
    output = tmp_path / "ray_function_table_parameters.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_FUNCTION_TABLE_PARAMETER_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_ray_function_table_array_diagnostics_compile_with_metal3(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "ray_function_table_array_diagnostics.metal"
    output = tmp_path / "ray_function_table_array_diagnostics.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_RAY_FUNCTION_TABLE_ARRAY_DIAGNOSTIC_SHADER)
    )
    assert "arrays of visible_function_table are not valid Metal buffer parameters" in (
        code
    )
    assert (
        "arrays of intersection_function_table are not valid Metal buffer parameters"
        in code
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(output),
        ]
    )


def test_generated_metal_compute_stage_with_builtins_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_builtins.metal"
    output = tmp_path / "compute_builtins.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_COMPUTE_BUILTINS_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_threadgroup_helper_barriers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "threadgroup_helper_barriers.metal"
    output = tmp_path / "threadgroup_helper_barriers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_THREADGROUP_HELPER_BARRIERS_SHADER),
        "compute",
    )
    assert "void writeScratch(threadgroup float scratch[64]" in code
    assert "float readScratch(threadgroup float scratch[64]" in code
    assert "threadgroup float scratch[64];" in code
    assert code.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 4
    assert code.count("threadgroup_barrier(mem_flags::mem_device);") == 2
    assert "threadgroup_barrier(mem_flags::mem_texture);" in code
    assert (
        "threadgroup_barrier(mem_flags::mem_device | "
        "mem_flags::mem_threadgroup | mem_flags::mem_texture);"
    ) in code
    assert "groupMemoryBarrier();" not in code
    assert "memoryBarrierShared();" not in code
    assert "memoryBarrierBuffer();" not in code
    assert "deviceMemoryBarrier();" not in code
    assert "memoryBarrierImage();" not in code
    assert "allMemoryBarrier();" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_threadgroup_atomic_barriers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "threadgroup_atomic_barriers.metal"
    output = tmp_path / "threadgroup_atomic_barriers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_THREADGROUP_ATOMIC_BARRIERS_SHADER),
        "compute",
    )
    assert (
        "return atomic_fetch_add_explicit(&counters[index], 1u, memory_order_relaxed);"
        in code
    )
    assert "atomic_store_explicit(&counters[index], 0u, memory_order_relaxed);" in code
    assert (
        "uint currentValue = atomic_load_explicit(&counters[index], memory_order_relaxed);"
        in code
    )
    assert "atomic_exchange_explicit(&counters[index]," in code
    assert code.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_atomic_array_initializers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "atomic_array_initializers.metal"
    output = tmp_path / "atomic_array_initializers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_ATOMIC_ARRAY_INITIALIZER_SHADER),
        "compute",
    )
    assert "threadgroup atomic_uint counters[4];" in code
    assert "threadgroup atomic_int signedCounters[2];" in code
    assert "atomic_store_explicit(&counters[0], 0u, memory_order_relaxed);" in code
    assert (
        "atomic_store_explicit(&counters[2], uint(index), memory_order_relaxed);"
        in code
    )
    assert (
        "atomic_store_explicit(&signedCounters[0], -1, memory_order_relaxed);" in code
    )
    assert "atomic_store_explicit(&counters, {" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_symbolic_atomic_array_initializers_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "symbolic_atomic_array_initializers.metal"
    output = tmp_path / "symbolic_atomic_array_initializers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_SYMBOLIC_ATOMIC_ARRAY_INITIALIZER_SHADER),
        "compute",
    )
    assert "threadgroup atomic_uint counters[COUNT];" in code
    assert "threadgroup atomic_uint expressionCounters[COUNT + 1];" in code
    assert "threadgroup atomic_int signedCounters[EXTRA];" in code
    assert "atomic_store_explicit(&counters[3], 0u, memory_order_relaxed);" in code
    assert (
        "atomic_store_explicit(&expressionCounters[4], 0u, memory_order_relaxed);"
        in code
    )
    assert "atomic_store_explicit(&signedCounters[4], 0, memory_order_relaxed);" in code
    assert "atomic_store_explicit(&counters, {" not in code
    assert "atomic_store_explicit(&expressionCounters, {" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_scoped_atomics_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "scoped_atomics.metal"
    output = tmp_path / "scoped_atomics.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_SCOPED_ATOMIC_SHADER),
        "compute",
    )
    assert "memory_scope_" not in code
    assert (
        "return atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expectedValues[index], desired, memory_order_relaxed, "
        "memory_order_relaxed);"
    ) in code
    assert "atomic_store_explicit(&counters[index], 0u, memory_order_relaxed);" in code
    assert (
        "uint oldValue = atomic_fetch_add_explicit(&counters[index], 1u, memory_order_relaxed);"
        in code
    )
    assert (
        "uint loaded = atomic_load_explicit(&counters[index], memory_order_relaxed);"
        in code
    )
    assert (
        "uint exchanged = atomic_exchange_explicit(&deviceCounters[index], loaded + oldValue, memory_order_relaxed);"
        in code
    )
    assert "atomic_compare_exchange_strong_explicit" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_atomic_pointer_targets_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "atomic_pointer_targets.metal"
    output = tmp_path / "atomic_pointer_targets.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_ATOMIC_POINTER_TARGETS_SHADER),
        "compute",
    )
    assert (
        "return atomic_fetch_add_explicit(counters + index, delta, memory_order_relaxed);"
        in code
    )
    assert (
        "return atomic_fetch_min_explicit(&counters[index], value, memory_order_relaxed);"
        in code
    )
    assert (
        "return atomic_exchange_explicit(flags + index, value, memory_order_relaxed);"
        in code
    )
    assert "atomic_store_explicit(&scratch[index], 7, memory_order_relaxed);" in code
    assert "atomic_store_explicit(&counters[index], 0, memory_order_relaxed);" in code
    assert (
        "int loaded = atomic_load_explicit(counters + index, memory_order_relaxed);"
        in code
    )
    assert (
        "atomic_store_explicit(&flags[index], wasSet && isSet, memory_order_relaxed);"
        in code
    )
    assert "atomic_fetch_add_explicit(&counters + index" not in code
    assert "atomic_exchange_explicit(&flags + index" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_threadgroup_atomic_pointer_alias_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "threadgroup_atomic_pointer_alias.metal"
    output = tmp_path / "threadgroup_atomic_pointer_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_THREADGROUP_ATOMIC_POINTER_ALIAS_SHADER),
        "compute",
    )
    assert "threadgroup atomic_int* alias = scratch + index;" in code
    assert "thread atomic_int* alias = scratch + index;" not in code
    assert "int oldValue = bumpThreadgroup(alias, 1);" in code
    assert (
        "int nextValue = atomic_fetch_add_explicit(alias + 1, oldValue, memory_order_relaxed);"
        in code
    )
    assert (
        "int rejected = 0 /* unsupported Metal address-space call: argument "
        "'alias' uses threadgroup address space but parameter 'counters' of "
        "'bumpDevice' requires device */;"
    ) in code
    assert "bumpDevice(alias + 2" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_threadgroup_atomic_ternary_alias_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "threadgroup_atomic_ternary_alias.metal"
    output = tmp_path / "threadgroup_atomic_ternary_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_THREADGROUP_ATOMIC_TERNARY_ALIAS_SHADER),
        "compute",
    )
    assert (
        "threadgroup atomic_int* alias = useA ? scratchA + index : scratchB + index;"
        in code
    )
    assert "thread atomic_int* alias = useA" not in code
    assert (
        "/* unsupported Metal address-space local alias: initializer branches "
        "'scratchA' (threadgroup) and 'counters' (device) use different address "
        "spaces; using uninitialized thread alias */"
    ) in code
    assert "thread atomic_int* mixedAlias;" in code
    assert "thread atomic_int* mixedAlias = useShared" not in code
    assert (
        "int rejected = 0 /* unsupported Metal address-space call: argument "
        "'mixedAlias' uses thread address space but parameter 'counters' of "
        "'bumpThreadgroup' requires threadgroup */;"
    ) in code
    assert (
        "int directRejected = 0 /* unsupported Metal address-space call: "
        "argument '<expr>' mixes branches 'scratchA' (threadgroup) and "
        "'counters' (device) but parameter 'counters' of 'bumpThreadgroup' "
        "requires threadgroup */;"
    ) in code
    assert "bumpThreadgroup(useShared ? scratchA" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_pointer_assignment_address_spaces_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "pointer_assignment_address_spaces.metal"
    output = tmp_path / "pointer_assignment_address_spaces.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_POINTER_ASSIGNMENT_ADDRESS_SPACE_SHADER),
        "compute",
    )
    assert "threadgroup atomic_int* alias = scratchA + index;" in code
    assert "alias = useA ? scratchA + index : scratchB + index;" in code
    assert (
        "/* unsupported Metal address-space assignment: value branches "
        "'scratchA' (threadgroup) and 'counters' (device) use different "
        "address spaces; assignment to 'alias' requires threadgroup */"
    ) in code
    assert (
        "/* unsupported Metal address-space assignment: value 'counters' uses "
        "device address space but target 'alias' uses threadgroup */"
    ) in code
    assert "alias = useShared ? scratchA + index : counters + index;" not in code
    assert "alias = counters + index;" not in code
    assert "int second = bumpThreadgroup(alias, first);" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_threadgroup_reference_ternary_alias_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "threadgroup_reference_ternary_alias.metal"
    output = tmp_path / "threadgroup_reference_ternary_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_THREADGROUP_REFERENCE_TERNARY_ALIAS_SHADER),
        "compute",
    )
    assert (
        "threadgroup Payload& alias = useA ? scratchA[index] : scratchB[index];" in code
    )
    assert (
        "/* unsupported Metal address-space local alias: initializer branches "
        "'scratchA' (threadgroup) and 'payloads' (device) use different address "
        "spaces; using uninitialized thread value */"
    ) in code
    assert "thread Payload mixedAlias;" in code
    assert "thread Payload& mixedAlias = useShared" not in code
    assert (
        "/* unsupported Metal address-space call: argument 'mixedAlias' uses "
        "thread address space but parameter 'payload' of 'useThreadgroup' "
        "requires threadgroup */"
    ) in code
    assert (
        "/* unsupported Metal address-space call: argument '<expr>' mixes "
        "branches 'scratchA' (threadgroup) and 'payloads' (device) but "
        "parameter 'payload' of 'useThreadgroup' requires threadgroup */"
    ) in code
    assert "useThreadgroup(mixedAlias, 2.0);" not in code
    assert "useThreadgroup(useShared ? scratchA" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_pointer_member_atomic_address_spaces_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "pointer_member_atomic_address_spaces.metal"
    output = tmp_path / "pointer_member_atomic_address_spaces.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_POINTER_MEMBER_ATOMIC_ADDRESS_SPACE_SHADER),
        "compute",
    )
    assert "device atomic_int* deviceCounters;" in code
    assert "threadgroup atomic_int* sharedCounters;" in code
    assert "deviceCounters [[device]]" not in code
    assert "sharedCounters [[threadgroup]]" not in code
    assert "int deviceOld = bumpDevice(bank.deviceCounters + index, 1);" in code
    assert (
        "int rejectedThread = 0 /* unsupported Metal address-space call: "
        "argument 'bank.deviceCounters' uses device address space but parameter "
        "'counters' of 'bumpThreadgroup' requires threadgroup */;"
    ) in code
    assert (
        "int sharedOld = bumpThreadgroup(bank.sharedCounters + index, rejectedThread);"
        in code
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_nested_pointer_member_atomic_alias_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "nested_pointer_member_atomic_alias.metal"
    output = tmp_path / "nested_pointer_member_atomic_alias.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_NESTED_POINTER_MEMBER_ATOMIC_ALIAS_SHADER),
        "compute",
    )
    assert "thread OuterBank* bankPtr = &bank;" in code
    assert (
        "threadgroup atomic_int* sharedAlias = bankPtr->inner.sharedCounters + index;"
        in code
    )
    assert (
        "device atomic_int* deviceAlias = bankPtr->inner.deviceCounters + index;"
        in code
    )
    assert (
        "int rejectedShared = 0 /* unsupported Metal address-space call: "
        "argument 'sharedAlias' uses threadgroup address space but parameter "
        "'counters' of 'bumpDevice' requires device */;"
    ) in code
    assert (
        "int rejectedDevice = 0 /* unsupported Metal address-space call: "
        "argument 'deviceAlias' uses device address space but parameter "
        "'counters' of 'bumpThreadgroup' requires threadgroup */;"
    ) in code
    assert "bumpDevice(sharedAlias + 1" not in code
    assert "bumpThreadgroup(deviceAlias + 1" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_reference_member_atomic_address_spaces_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "reference_member_atomic_address_spaces.metal"
    output = tmp_path / "reference_member_atomic_address_spaces.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_REFERENCE_MEMBER_ATOMIC_ADDRESS_SPACE_SHADER),
        "compute",
    )
    assert "thread Bank& ref = bank;" in code
    assert "Bank & ref = bank;" not in code
    assert "int sharedOld = bumpThreadgroup(ref.sharedCounters + index, 1);" in code
    assert "int deviceOld = bumpDevice(ref.deviceCounters + index, sharedOld);" in code
    assert (
        "int rejected = 0 /* unsupported Metal address-space call: argument "
        "'ref.sharedCounters' uses threadgroup address space but parameter "
        "'counters' of 'bumpDevice' requires device */;"
    ) in code
    assert "bumpDevice(ref.sharedCounters + index" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_atomic_compare_exchange_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "atomic_compare_exchange.metal"
    output = tmp_path / "atomic_compare_exchange.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_ATOMIC_COMPARE_EXCHANGE_SHADER),
        "compute",
    )
    indexed_compare_exchange = (
        "return atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expectedValues[index], desired, memory_order_relaxed, "
        "memory_order_relaxed);"
    )
    assert code.count(indexed_compare_exchange) == 2
    assert (
        "bool claimed = atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expected, 2u, memory_order_relaxed, memory_order_relaxed);"
    ) in code
    assert "atomic_compare_exchange_weak_explicit(&counter, &expected," not in code
    assert "atomic_compare_exchange_strong_explicit" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_device_atomic_compare_exchange_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "device_atomic_compare_exchange.metal"
    output = tmp_path / "device_atomic_compare_exchange.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_DEVICE_ATOMIC_COMPARE_EXCHANGE_SHADER),
        "compute",
    )
    assert (
        "return atomic_compare_exchange_weak_explicit(counters + index, "
        "expectedValues + index, desired, memory_order_relaxed, "
        "memory_order_relaxed);"
    ) in code
    assert (
        "return false /* unsupported Metal atomic compare-exchange expected pointer: "
        "expected storage 'expectedValues' uses device address space; Metal requires "
        "thread storage */;"
    ) in code
    assert (
        "bool directClaimed = atomic_compare_exchange_weak_explicit(&counters[index], "
        "&expected, 3u, memory_order_relaxed, memory_order_relaxed);"
    ) in code
    assert "atomic_compare_exchange_strong_explicit" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_struct_member_atomic_compare_exchange_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "struct_member_atomic_compare_exchange.metal"
    output = tmp_path / "struct_member_atomic_compare_exchange.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_STRUCT_MEMBER_ATOMIC_COMPARE_EXCHANGE_SHADER),
        "compute",
    )
    assert "thread uint* threadExpected;" in code
    assert "device uint* deviceExpected;" in code
    assert (
        "bool direct = atomic_compare_exchange_weak_explicit(&counters[index], "
        "&bank.threadExpected[index], 1u, memory_order_relaxed, "
        "memory_order_relaxed);"
    ) in code
    assert (
        "bool pointerDirect = atomic_compare_exchange_weak_explicit(counters + index, "
        "bank.threadExpected + index, 2u, memory_order_relaxed, "
        "memory_order_relaxed);"
    ) in code
    assert (
        "bool rejectedDirect = false /* unsupported Metal atomic compare-exchange "
        "expected pointer: expected storage 'bank.deviceExpected' uses device "
        "address space; Metal requires thread storage */;"
    ) in code
    assert (
        "atomic_compare_exchange_weak_explicit(counters + index, bank.deviceExpected + index"
        not in code
    )
    assert "atomic_compare_exchange_strong_explicit" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_buffer_block_atomic_compare_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "buffer_block_atomic_compare.metal"
    output = tmp_path / "buffer_block_atomic_compare.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_BUFFER_BLOCK_ATOMIC_COMPARE_SHADER),
        "compute",
    )
    assert "__crossgl_buffer_atomic_compare_exchange_uint" in code
    assert "__crossgl_buffer_atomic_compare_exchange_int" in code
    assert (
        "atomic_compare_exchange_weak_explicit(target, &original, value, "
        "memory_order_relaxed, memory_order_relaxed)"
    ) in code
    assert "atomicCompSwap(" not in code
    assert "atomic_compare_exchange_strong_explicit" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_address_space_parameters_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "address_space_parameters.metal"
    output = tmp_path / "address_space_parameters.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_ADDRESS_SPACE_PARAMETER_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_pointer_member_access_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "pointer_member_access.metal"
    output = tmp_path / "pointer_member_access.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_POINTER_MEMBER_ACCESS_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_indexed_pointer_member_access_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "indexed_pointer_member_access.metal"
    output = tmp_path / "indexed_pointer_member_access.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_INDEXED_POINTER_MEMBER_ACCESS_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_readonly_raw_buffers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "readonly_raw_buffers.metal"
    output = tmp_path / "readonly_raw_buffers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_READONLY_RAW_BUFFER_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_readonly_raw_buffer_diagnostics_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "readonly_raw_buffer_diagnostics.metal"
    output = tmp_path / "readonly_raw_buffer_diagnostics.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_READONLY_RAW_BUFFER_DIAGNOSTIC_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_readonly_raw_buffer_helpers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "readonly_raw_buffer_helpers.metal"
    output = tmp_path / "readonly_raw_buffer_helpers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_READONLY_RAW_BUFFER_HELPER_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_readonly_raw_buffer_mutable_helper_call_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "readonly_raw_buffer_mutable_helper_call.metal"
    output = tmp_path / "readonly_raw_buffer_mutable_helper_call.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_READONLY_RAW_BUFFER_MUTABLE_HELPER_CALL_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_const_reference_helpers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "const_reference_helpers.metal"
    output = tmp_path / "const_reference_helpers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_CONST_REFERENCE_HELPER_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    assert "float readPayload(const thread Payload& payload)" in code
    assert "void rejectWrite(const thread Payload& payload)" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "payload.value = 1.0;" not in code
    assert "mutate(payload);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_const_pointer_array_helpers_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "const_pointer_array_helpers.metal"
    output = tmp_path / "const_pointer_array_helpers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_CONST_POINTER_ARRAY_HELPER_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    assert "float readPointer(const thread Payload* payload)" in code
    assert "return payload->value;" in code
    assert "float readArray(const thread float values[], int index)" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "PointerAccessNode" not in code
    assert "payload->value = 1.0;" not in code
    assert "mutatePointer(payload);" not in code
    assert "values[0] = 4.0;" not in code
    assert "mutateArray(values);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_const_threadgroup_pointer_alias_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "const_threadgroup_pointer_alias.metal"
    output = tmp_path / "const_threadgroup_pointer_alias.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_CONST_THREADGROUP_POINTER_ALIAS_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "const threadgroup Payload* alias = &scratch;" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "alias->value = value;" not in code
    assert "mutate(alias);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_readonly_device_pointer_alias_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "readonly_device_pointer_alias.metal"
    output = tmp_path / "readonly_device_pointer_alias.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_READONLY_DEVICE_POINTER_ALIAS_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "const device Payload* alias = payload;" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "alias->value = value;" not in code
    assert "mutate(alias);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_const_local_array_alias_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "const_local_array_alias.metal"
    output = tmp_path / "const_local_array_alias.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_CONST_LOCAL_ARRAY_ALIAS_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "const float values[2] = {1.0, 2.0};" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "values[0] = value;" not in code
    assert "mutate(values);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_constant_pointer_reference_alias_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "constant_pointer_reference_alias.metal"
    output = tmp_path / "constant_pointer_reference_alias.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_CONSTANT_POINTER_REFERENCE_ALIAS_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "constant Payload* pointerAlias = pointerPayload;" in code
    assert "constant Payload& referenceAlias = referencePayload;" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "pointerAlias->value = pointerValue;" not in code
    assert "referenceAlias.value = referenceValue;" not in code
    assert "mutatePointer(pointerAlias);" not in code
    assert "mutateReference(referenceAlias);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_nested_constant_local_array_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "nested_constant_local_array.metal"
    output = tmp_path / "nested_constant_local_array.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_NESTED_CONSTANT_LOCAL_ARRAY_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "const float values[2][2] = {{1.0, 2.0}, {3.0, 4.0}};" in code
    assert "unsupported Metal parameter store" in code
    assert "unsupported Metal parameter call" in code
    assert "values[0][1] = value;" not in code
    assert "mutate(values);" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_constant_scalar_vector_resource_index_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "constant_scalar_vector_resource_index.metal"
    output = tmp_path / "constant_scalar_vector_resource_index.air"
    code = MetalCodeGen().generate(
        crosstl.translator.parse(METAL_CONSTANT_SCALAR_VECTOR_RESOURCE_INDEX_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    assert "const float scale = 2.0;" in code
    assert "const float2 bias = float2(0.25, 0.5);" in code
    assert "constexpr int LAYER = 3;" in code
    assert "array<texture2d<float>, 4> textures [[texture(0)]]" in code
    assert "array<sampler, 4> samplers [[sampler(0)]]" in code
    assert "textures[LAYER].sample(samplers[LAYER], uv)" in code
    assert "sampleLayer(input.uv, textures, samplers)" in code
    assert "unsupported Metal parameter store" in code
    assert "scale = value;" not in code
    assert "bias.x = value;" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_struct_resource_array_queries_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "struct_resource_array_queries.metal"
    output = tmp_path / "struct_resource_array_queries.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_STRUCT_RESOURCE_ARRAY_QUERY_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    assert "array<texture2d<float>, 4> textures;" in code
    assert "array<texture2d_array<float>, 4> layers;" in code
    assert "array<texture2d_ms<float>, 4> msTextures;" in code
    assert "array<texture2d_ms_array<float>, 4> msArrays;" in code
    assert "constexpr int LAYER = 2;" in code
    assert "pack.textures[LAYER].get_width(uint(LAYER))" in code
    assert "pack.layers[LAYER].get_array_size()" in code
    assert "pack.textures[LAYER].get_num_mip_levels()" in code
    assert "pack.layers[LAYER].calculate_unclamped_lod" in code
    assert "uvLayer.xy" in code
    assert "pack.msTextures[LAYER].get_num_samples()" in code
    assert "pack.msArrays[LAYER].get_num_samples()" in code
    assert "textureSize(" not in code
    assert "textureQueryLod(" not in code
    assert "textureSamples(" not in code
    assert "imageSamples(" not in code
    assert "unsupported Metal texture samples query" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_nested_resource_container_forwarding_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "nested_resource_container_forwarding.metal"
    output = tmp_path / "nested_resource_container_forwarding.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_NESTED_RESOURCE_CONTAINER_FORWARDING_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    assert "array<texture2d<float>, 4> textures;" in code
    assert "array<texture2d_array<float>, 4> layers;" in code
    assert "array<sampler, 4> samplers;" in code
    assert "samplePack(bank.pack, bank.samplers, uv, uvLayer)" in code
    assert "texture2d<float> textures[4];" not in code
    assert "sampler samplers[4];" not in code

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_address_space_mismatch_calls_compile_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "address_space_mismatch_calls.metal"
    output = tmp_path / "address_space_mismatch_calls.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_ADDRESS_SPACE_MISMATCH_CALL_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_do_while_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_do_while.metal"
    output = tmp_path / "compute_do_while.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(COMPUTE_DO_WHILE_SHADER), "compute"
    )
    assert "do {" in code
    assert "DoWhileNode(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_scalar_image_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_scalar_image.metal"
    output = tmp_path / "compute_scalar_image.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_SCALAR_IMAGE_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_rg_image_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_rg_image.metal"
    output = tmp_path / "compute_rg_image.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_IMAGE_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_integer_image_atomics_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_integer_image_atomics.metal"
    output = tmp_path / "compute_integer_image_atomics.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(INTEGER_IMAGE_ATOMICS_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_resource_array_image_atomics_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_resource_array_image_atomics.metal"
    output = tmp_path / "compute_resource_array_image_atomics.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_RESOURCE_ARRAY_IMAGE_ATOMICS_COMPUTE_SHADER),
        "compute",
    )
    assert "images[index].atomic_fetch_add(uint2(pixel), value).x" in code
    assert "imageAtomicCompSwap_iimage2D(images[index], pixel, expected, value)" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_storage_image_access_qualifiers_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_storage_image_access_qualifiers.metal"
    output = tmp_path / "compute_storage_image_access_qualifiers.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_STORAGE_IMAGE_ACCESS_QUALIFIERS_COMPUTE_SHADER),
        "compute",
    )
    assert "texture2d<float, access::read> source [[texture(0)]]" in code
    assert "texture2d<float, access::write> target [[texture(1)]]" in code
    assert "texture2d<uint, access::read_write> counters [[texture(2)]]" in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_struct_held_storage_image_arrays_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_struct_held_storage_image_arrays.metal"
    output = tmp_path / "compute_struct_held_storage_image_arrays.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_STRUCT_HELD_STORAGE_IMAGE_ARRAYS_COMPUTE_SHADER),
        "compute",
    )
    assert "array<texture2d<float, access::read>, 2> pairs;" in code
    assert "array<texture2d<float, access::write>, 2> targets;" in code
    assert "array<texture2d<uint, access::read_write>, 2> counters;" in code
    assert "pack.pairs[index].read(uint2(pixel)).xy" in code
    assert "pack.targets[index].write(float4(value, 0.0, 0.0), uint2(pixel))" in code
    assert "pack.counters[index].atomic_fetch_add(uint2(pixel), value).x" in code
    assert "array<texture2d<float, access::read>, 2> pairsAlias = pack.pairs" in code
    assert "chainedPairsAlias[index].read(uint2(pixel)).xy" in code
    assert (
        "array<texture2d<float, access::write>, 2> targetsAlias = pack.targets" in code
    )
    assert (
        "chainedTargetsAlias[index].write(float4(value, 0.0, 0.0), uint2(pixel))"
        in code
    )
    assert (
        "array<texture2d<uint, access::read_write>, 2> countersAlias = pack.counters"
        in code
    )
    assert "counterAlias.atomic_fetch_add(uint2(pixel), value).x" in code
    assert "thread texture2d" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_multisample_storage_image_aliases_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_multisample_storage_image_aliases.metal"
    output = tmp_path / "compute_multisample_storage_image_aliases.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            METAL_MULTISAMPLE_STORAGE_IMAGE_ALIASES_COMPUTE_SHADER
        ),
        "compute",
    )
    assert "texture2d_ms<float, access::read> colorAlias = color;" in code
    assert "texture2d_ms<float, access::read> chainedColorAlias = colorAlias" in code
    assert "texture2d_ms<uint, access::read> counterAlias = counter;" in code
    assert "chainedColorAlias.read(uint2(pixel), uint(sampleIndex))" in code
    assert (
        "array<texture2d_ms<float, access::read>, 2> imagesAlias = imageArray" in code
    )
    assert "texture2d_ms<float, access::read> imageAlias = imagesAlias[index]" in code
    assert (
        "array<texture2d_ms_array<uint, access::read>, 2> countersAlias = counterArray"
        in code
    )
    assert (
        "texture2d_ms_array<uint, access::read> counterAlias = countersAlias[index]"
        in code
    )
    assert (
        "counterAlias.read(uint2(pixelLayer.xy), uint(pixelLayer.z), uint(sampleIndex)).x"
        in code
    )
    assert "unsupported Metal multisample image store: imageStore" in code
    assert "unsupported Metal multisample image atomic: imageAtomicAdd" in code
    assert ".write(" not in code
    assert "atomic_fetch_add" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_storage_image_query_diagnostics_compile_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_storage_image_query_diagnostics.metal"
    output = tmp_path / "compute_storage_image_query_diagnostics.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(METAL_STORAGE_IMAGE_QUERY_DIAGNOSTICS_COMPUTE_SHADER),
        "compute",
    )
    assert "texture2d<float, access::read_write> colorImage [[texture(0)]]" in code
    assert "texture3d<float, access::read_write> volumeImage [[texture(1)]]" in code
    assert (
        "texture2d_array<float, access::read_write> layerImage [[texture(2)]]" in code
    )
    assert "texture2d<float, access::write> target [[texture(3)]]" in code
    assert (
        "unsupported Metal texture query: textureQueryLevels on "
        "texture2d<float, access::read_write>"
    ) in code
    assert (
        "unsupported Metal texture query: textureQueryLod on "
        "texture3d<float, access::read_write>"
    ) in code
    assert (
        "unsupported Metal texture samples query: requires multisample texture" in code
    )
    assert "textureQueryLevels(" not in code
    assert "textureQueryLod(" not in code
    assert "textureSamples(" not in code
    assert "imageSamples(" not in code
    assert "get_num_mip_levels" not in code
    assert "get_num_samples" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_rg_image_array_compiles_with_metal(tmp_path):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_rg_image_array.metal"
    output = tmp_path / "compute_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_switch_match_image_case_scope_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_switch_match_image_case_scope.metal"
    output = tmp_path / "compute_switch_match_image_case_scope.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_IMAGE_CASE_SCOPE_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_do_while_switch_match_image_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_do_while_switch_match_image.metal"
    output = tmp_path / "compute_do_while_switch_match_image.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(DO_WHILE_SWITCH_MATCH_IMAGE_COMPUTE_SHADER),
        "compute",
    )
    assert "do {" in code
    assert "case 0: {" in code
    assert "DoWhileNode(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_inferred_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_inferred_rg_image_array.metal"
    output = tmp_path / "compute_inferred_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_INFERRED_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_transitive_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_transitive_rg_image_array.metal"
    output = tmp_path / "compute_transitive_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_TRANSITIVE_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_fixed_param_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_fixed_param_rg_image_array.metal"
    output = tmp_path / "compute_fixed_param_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_fixed_const_param_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_fixed_const_param_rg_image_array.metal"
    output = tmp_path / "compute_fixed_const_param_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_CONST_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_fixed_expr_param_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_fixed_expr_param_rg_image_array.metal"
    output = tmp_path / "compute_fixed_expr_param_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_EXPR_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_direct_index_within_fixed_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_direct_index_within_fixed_rg_image_array.metal"
    output = tmp_path / "compute_direct_index_within_fixed_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            RG_DIRECT_INDEX_WITHIN_FIXED_IMAGE_ARRAY_COMPUTE_SHADER
        ),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_fixed_global_unsized_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_fixed_global_unsized_rg_image_array.metal"
    output = tmp_path / "compute_fixed_global_unsized_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_GLOBAL_TO_UNSIZED_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_fixed_const_index_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_fixed_const_index_rg_image_array.metal"
    output = tmp_path / "compute_fixed_const_index_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_shadowed_const_index_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_shadowed_const_index_rg_image_array.metal"
    output = tmp_path / "compute_shadowed_const_index_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(RG_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_metal_compute_transitive_shadowed_const_index_rg_image_array_compiles_with_metal(
    tmp_path,
):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not installed")

    source = tmp_path / "compute_transitive_shadowed_const_index_rg_image_array.metal"
    output = tmp_path / "compute_transitive_shadowed_const_index_rg_image_array.air"
    code = MetalCodeGen().generate_stage(
        crosstl.translator.parse(
            RG_TRANSITIVE_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER
        ),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(source), "-o", str(output)]
    )


def test_generated_glsl_fragment_smoke_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_range.frag"
    code = GLSLCodeGen().generate(crosstl.translator.parse(FRAGMENT_RANGE_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_switch_match_case_scope_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_switch_match_case_scope.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_CASE_SCOPE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_switch_match_texture_case_scope_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_switch_match_texture_case_scope.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_TEXTURE_CASE_SCOPE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_struct_input_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_struct_input.frag"
    code = GLSLCodeGen().generate(
        crosstl.translator.parse(FRAGMENT_STRUCT_INPUT_SHADER)
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_sampled_texture_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_sampled_texture_array.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLED_TEXTURE_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_sampled_texture_const_index_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_sampled_texture_const_index.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLED_TEXTURE_ARRAY_CONST_INDEX_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_sampled_texture_transitive_shadowed_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_sampled_texture_transitive_shadowed.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            SAMPLED_TEXTURE_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_shadow_sampler_transitive_shadowed_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_shadow_sampler_transitive_shadowed.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            SHADOW_SAMPLER_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_array_shadow_texture_resource_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_array_shadow_texture_resource_array.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_SHADOW_TEXTURE_RESOURCE_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_array_shadow_texture_query_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_array_shadow_texture_query.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_SHADOW_TEXTURE_QUERY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_array_texture_query_lod_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_array_texture_query_lod.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_shadow_array_texture_query_lod_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_shadow_array_texture_query_lod.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_cube_array_texture_grad_gather_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_cube_array_texture_grad_gather.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(CUBE_ARRAY_TEXTURE_GRAD_GATHER_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_texture_gather_offset_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_texture_gather_offset.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_GATHER_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_texture_sample_offset_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_texture_sample_offset.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_SAMPLE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_texture_3d_sample_offset_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_texture_3d_sample_offset.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_3D_SAMPLE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_texture_projection_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_texture_projection.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_PROJECTED_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_shadow_gather_compare_offset_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_shadow_gather_compare_offset.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_GATHER_COMPARE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_shadow_compare_lod_grad_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_shadow_compare_lod_grad.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_COMPARE_LOD_GRAD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_fragment_projected_shadow_compare_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "fragment_projected_shadow_compare.frag"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(PROJECTED_SHADOW_COMPARE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "frag", str(source)])


def test_generated_glsl_vertex_struct_io_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "vertex_struct_io.vert"
    code = GLSLCodeGen().generate(crosstl.translator.parse(VERTEX_STRUCT_IO_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "vert", str(source)])


def test_generated_glsl_combined_stages_validate_separately_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    ast = crosstl.translator.parse(COMBINED_STAGE_IO_SHADER)
    generator = GLSLCodeGen()

    vertex_source = tmp_path / "combined_stage.vert"
    vertex_source.write_text(generator.generate_stage(ast, "vertex"), encoding="utf-8")
    run_validator([glslang, "-S", "vert", str(vertex_source)])

    fragment_source = tmp_path / "combined_stage.frag"
    fragment_source.write_text(
        generator.generate_stage(ast, "fragment"), encoding="utf-8"
    )
    run_validator([glslang, "-S", "frag", str(fragment_source)])


def test_generated_glsl_compute_stage_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_stage.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(COMPUTE_STAGE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_do_while_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_do_while.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(COMPUTE_DO_WHILE_SHADER), "compute"
    )
    assert "do {" in code
    assert "DoWhileNode(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_default_float_image_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_default_float_image.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(DEFAULT_FLOAT_IMAGE_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_rg_image_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_rg_image.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_IMAGE_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_integer_image_atomics_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_integer_image_atomics.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(INTEGER_IMAGE_ATOMICS_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_rg_image_array_validates_with_glslang(tmp_path):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_switch_match_image_case_scope_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_switch_match_image_case_scope.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_IMAGE_CASE_SCOPE_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_do_while_switch_match_image_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_do_while_switch_match_image.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(DO_WHILE_SWITCH_MATCH_IMAGE_COMPUTE_SHADER),
        "compute",
    )
    assert "do {" in code
    assert "case 0: {" in code
    assert "DoWhileNode(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_inferred_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_inferred_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_INFERRED_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_transitive_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_transitive_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_TRANSITIVE_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_fixed_param_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_fixed_param_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_fixed_const_param_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_fixed_const_param_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_CONST_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_fixed_expr_param_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_fixed_expr_param_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_EXPR_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_direct_index_within_fixed_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_direct_index_within_fixed_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            RG_DIRECT_INDEX_WITHIN_FIXED_IMAGE_ARRAY_COMPUTE_SHADER
        ),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_fixed_global_unsized_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_fixed_global_unsized_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_GLOBAL_TO_UNSIZED_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_fixed_const_index_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_fixed_const_index_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_shadowed_const_index_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_shadowed_const_index_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_glsl_compute_transitive_shadowed_const_index_rg_image_array_validates_with_glslang(
    tmp_path,
):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not installed")

    source = tmp_path / "compute_transitive_shadowed_const_index_rg_image_array.comp"
    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            RG_TRANSITIVE_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER
        ),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator([glslang, "-S", "comp", str(source)])


def test_generated_hlsl_combined_stages_validate_separately_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    ast = crosstl.translator.parse(COMBINED_STAGE_IO_SHADER)
    generator = HLSLCodeGen()

    for stage_name, target, entry_point in (
        ("vertex", "vs_6_0", "VSMain"),
        ("fragment", "ps_6_0", "PSMain"),
    ):
        source = tmp_path / f"combined_stage_{stage_name}.hlsl"
        output = tmp_path / f"combined_stage_{stage_name}.dxil"
        source.write_text(generator.generate_stage(ast, stage_name), encoding="utf-8")

        run_validator(
            [dxc, "-T", target, "-E", entry_point, str(source), "-Fo", str(output)]
        )


def test_generated_hlsl_fragment_switch_match_case_scope_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_switch_match_case_scope.hlsl"
    output = tmp_path / "fragment_switch_match_case_scope.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_CASE_SCOPE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_switch_match_texture_case_scope_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_switch_match_texture_case_scope.hlsl"
    output = tmp_path / "fragment_switch_match_texture_case_scope.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_TEXTURE_CASE_SCOPE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_sampled_texture_array_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_sampled_texture_array.hlsl"
    output = tmp_path / "fragment_sampled_texture_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLED_TEXTURE_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_sampled_texture_const_index_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_sampled_texture_const_index.hlsl"
    output = tmp_path / "fragment_sampled_texture_const_index.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SAMPLED_TEXTURE_ARRAY_CONST_INDEX_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_sampled_texture_transitive_shadowed_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_sampled_texture_transitive_shadowed.hlsl"
    output = tmp_path / "fragment_sampled_texture_transitive_shadowed.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            SAMPLED_TEXTURE_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_shadow_sampler_transitive_shadowed_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_shadow_sampler_transitive_shadowed.hlsl"
    output = tmp_path / "fragment_shadow_sampler_transitive_shadowed.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            SHADOW_SAMPLER_ARRAY_TRANSITIVE_SHADOWED_FRAGMENT_SHADER
        ),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_array_shadow_texture_resource_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_array_shadow_texture_resource_array.hlsl"
    output = tmp_path / "fragment_array_shadow_texture_resource_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_SHADOW_TEXTURE_RESOURCE_ARRAY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_array_shadow_texture_query_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_array_shadow_texture_query.hlsl"
    output = tmp_path / "fragment_array_shadow_texture_query.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_SHADOW_TEXTURE_QUERY_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_array_texture_query_lod_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_array_texture_query_lod.hlsl"
    output = tmp_path / "fragment_array_texture_query_lod.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_shadow_array_texture_query_lod_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_shadow_array_texture_query_lod.hlsl"
    output = tmp_path / "fragment_shadow_array_texture_query_lod.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_ARRAY_TEXTURE_QUERY_LOD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_cube_array_texture_grad_gather_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_cube_array_texture_grad_gather.hlsl"
    output = tmp_path / "fragment_cube_array_texture_grad_gather.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(CUBE_ARRAY_TEXTURE_GRAD_GATHER_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_texture_gather_offset_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_texture_gather_offset.hlsl"
    output = tmp_path / "fragment_texture_gather_offset.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_GATHER_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_texture_sample_offset_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_texture_sample_offset.hlsl"
    output = tmp_path / "fragment_texture_sample_offset.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_SAMPLE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_texture_3d_sample_offset_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_texture_3d_sample_offset.hlsl"
    output = tmp_path / "fragment_texture_3d_sample_offset.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_3D_SAMPLE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_texture_projection_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_texture_projection.hlsl"
    output = tmp_path / "fragment_texture_projection.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(TEXTURE_PROJECTED_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_shadow_gather_compare_offset_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "fragment_shadow_gather_compare_offset.hlsl"
    output = tmp_path / "fragment_shadow_gather_compare_offset.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_GATHER_COMPARE_OFFSET_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_0", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_shadow_compare_lod_grad_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    if not dxc_supports_sample_cmp_lod_grad(dxc, tmp_path):
        pytest.skip("dxc does not support SampleCmpLevel/SampleCmpGrad variants")

    source = tmp_path / "fragment_shadow_compare_lod_grad.hlsl"
    output = tmp_path / "fragment_shadow_compare_lod_grad.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SHADOW_COMPARE_LOD_GRAD_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_7", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_fragment_projected_shadow_compare_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    if not dxc_supports_sample_cmp_lod_grad(dxc, tmp_path):
        pytest.skip("dxc does not support SampleCmpLevel/SampleCmpGrad variants")

    source = tmp_path / "fragment_projected_shadow_compare.hlsl"
    output = tmp_path / "fragment_projected_shadow_compare.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(PROJECTED_SHADOW_COMPARE_FRAGMENT_SHADER),
        "fragment",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "ps_6_7", "-E", "PSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_stage_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_stage.hlsl"
    output = tmp_path / "compute_stage.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(COMPUTE_STAGE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_do_while_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_do_while.hlsl"
    output = tmp_path / "compute_do_while.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(COMPUTE_DO_WHILE_SHADER), "compute"
    )
    assert "do {" in code
    assert "DoWhileNode(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_default_float_image_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_default_float_image.hlsl"
    output = tmp_path / "compute_default_float_image.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(DEFAULT_FLOAT_IMAGE_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_rg_image_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_rg_image.hlsl"
    output = tmp_path / "compute_rg_image.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_IMAGE_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_integer_image_atomics_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_integer_image_atomics.hlsl"
    output = tmp_path / "compute_integer_image_atomics.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(INTEGER_IMAGE_ATOMICS_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_rg_image_array_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_rg_image_array.hlsl"
    output = tmp_path / "compute_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_switch_match_image_case_scope_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_switch_match_image_case_scope.hlsl"
    output = tmp_path / "compute_switch_match_image_case_scope.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(SWITCH_MATCH_IMAGE_CASE_SCOPE_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_do_while_switch_match_image_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_do_while_switch_match_image.hlsl"
    output = tmp_path / "compute_do_while_switch_match_image.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(DO_WHILE_SWITCH_MATCH_IMAGE_COMPUTE_SHADER),
        "compute",
    )
    assert "do {" in code
    assert "case 0: {" in code
    assert "DoWhileNode(" not in code
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_inferred_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_inferred_rg_image_array.hlsl"
    output = tmp_path / "compute_inferred_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_INFERRED_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_transitive_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_transitive_rg_image_array.hlsl"
    output = tmp_path / "compute_transitive_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_TRANSITIVE_IMAGE_ARRAY_COMPUTE_SHADER), "compute"
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_fixed_param_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_fixed_param_rg_image_array.hlsl"
    output = tmp_path / "compute_fixed_param_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_fixed_const_param_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_fixed_const_param_rg_image_array.hlsl"
    output = tmp_path / "compute_fixed_const_param_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_CONST_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_fixed_expr_param_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_fixed_expr_param_rg_image_array.hlsl"
    output = tmp_path / "compute_fixed_expr_param_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_EXPR_PARAM_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_direct_index_within_fixed_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_direct_index_within_fixed_rg_image_array.hlsl"
    output = tmp_path / "compute_direct_index_within_fixed_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            RG_DIRECT_INDEX_WITHIN_FIXED_IMAGE_ARRAY_COMPUTE_SHADER
        ),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_fixed_global_unsized_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_fixed_global_unsized_rg_image_array.hlsl"
    output = tmp_path / "compute_fixed_global_unsized_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_GLOBAL_TO_UNSIZED_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_fixed_const_index_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_fixed_const_index_rg_image_array.hlsl"
    output = tmp_path / "compute_fixed_const_index_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_FIXED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_shadowed_const_index_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_shadowed_const_index_rg_image_array.hlsl"
    output = tmp_path / "compute_shadowed_const_index_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(RG_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_compute_transitive_shadowed_const_index_rg_image_array_validates_with_dxc(
    tmp_path,
):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "compute_transitive_shadowed_const_index_rg_image_array.hlsl"
    output = tmp_path / "compute_transitive_shadowed_const_index_rg_image_array.dxil"
    code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(
            RG_TRANSITIVE_SHADOWED_CONST_INDEX_IMAGE_ARRAY_COMPUTE_SHADER
        ),
        "compute",
    )
    source.write_text(code, encoding="utf-8")

    run_validator(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(source), "-Fo", str(output)]
    )


def test_generated_hlsl_helper_smoke_validates_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    source = tmp_path / "range_for_in.hlsl"
    output = tmp_path / "range_for_in.dxil"
    code = HLSLCodeGen().generate(crosstl.translator.parse(HELPER_RANGE_SHADER))
    source.write_text(code, encoding="utf-8")

    run_validator([dxc, "-T", "lib_6_3", str(source), "-Fo", str(output)])
