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
            memoryBarrier();
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

    compute {
        layout(std430, binding = 2) readonly buffer float values[];
        layout(std430, binding = 3) writeonly buffer float outValues[];

        void main() {
            float mass = particleBlock.particles[0u].mass;
            float value = buffer_load(values, 1u);
            particleBlock.particles[1u].mass = mass + value;
            buffer_store(outValues, 0u, mass);
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


METAL_RAY_TRACING_HELPER_SHADER = """
shader MetalRayTracingHelperValidation {
    accelerationStructureEXT topLevelAS @binding(0);

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


def validate_spirv_shader_source(tmp_path, stem, shader_source):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")

    source = tmp_path / f"{stem}.spvasm"
    output = tmp_path / f"{stem}.spv"
    code = VulkanSPIRVCodeGen().generate(crosstl.translator.parse(shader_source))
    assert "WARNING" not in code
    source.write_text(code, encoding="utf-8")

    run_validator([spirv_as, str(source), "-o", str(output)])
    run_validator([spirv_val, str(output)])


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
