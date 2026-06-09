import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen


def test_glsl_hlsl_style_texture_alias_arrays_and_storage_image_lower_to_glsl_resources():
    shader = """
    shader HlslStyleTextureAliases {
        Texture2D<float4> textures[4] @ binding(0);
        SamplerState samplers[4] @ binding(4);
        RWTexture2D<uint> counters @ binding(8) @r32ui;

        vec4 sampleLayer(
            Texture2D<float4> textures[4],
            SamplerState samplers[4],
            int layer,
            vec2 uv
        ) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(vec2 uv @ TEXCOORD0, int layer @ flat) @ gl_FragColor {
                imageStore(counters, ivec2(layer, layer), 7u);
                return sampleLayer(textures, samplers, layer, uv);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "layout(binding = 0) uniform sampler2D textures[4];" in generated
    assert "layout(r32ui, binding = 8) uniform uimage2D counters;" in generated
    assert "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in generated
    assert "return texture(textures[layer], uv);" in generated
    assert "imageStore(counters, ivec2(layer, layer), uvec4(7u));" in generated
    assert "fragColor = sampleLayer(textures, layer, uv);" in generated
    assert "Texture2D" not in generated
    assert "SamplerState" not in generated
    assert "RWTexture2D" not in generated
    assert "samplers" not in generated


def test_hip_generated_compute_kernel_preserves_long_expression_arrays_and_metadata():
    shader = """
    shader GeneratedKernelStress {
        const int TILE = 8;
        const int TILE2 = TILE * 2;
        RWStructuredBuffer<float> outValues @ binding(0);
        StructuredBuffer<float> inValues @ binding(1);

        compute {
            void main(
                uint3 gid @ gl_GlobalInvocationID,
                uint3 lid @ gl_LocalInvocationID
            ) @numthreads(TILE, 2, 1) {
                float localA[TILE2];
                float localB[(2 + 1) * 2];
                localA[lid.x] = buffer_load(inValues, gid.x);
                localB[0] = (
                    ((localA[lid.x] + 1.0) * 2.0) -
                    (sin(float(gid.x)) / (cos(float(lid.y)) + 1.0))
                );
                buffer_store(outValues, gid.x, localB[0]);
            }
        }
    }
    """

    generated = HipCodeGen().generate(crosstl.translator.parse(shader))

    assert "const int TILE = 8;" in generated
    assert "const int TILE2 = (TILE * 2);" in generated
    assert (
        "// CrossGL resource metadata: name=outValues kind=buffer set=0 "
        "binding=0 binding_source=explicit"
    ) in generated
    assert "float* outValues;" in generated
    assert "const float* inValues;" in generated
    assert (
        "__global__ void __launch_bounds__((TILE) * (2) * (1)) compute_main()"
        in generated
    )
    assert "float localA[TILE2];" in generated
    assert "float localB[((2 + 1) * 2)];" in generated
    assert (
        "localA[threadIdx.x] = inValues[(blockIdx.x * blockDim.x + threadIdx.x)];"
        in generated
    )
    assert "sinf(float((blockIdx.x * blockDim.x + threadIdx.x)))" in generated
    assert "cosf(float(threadIdx.y))" in generated
    assert (
        "outValues[(blockIdx.x * blockDim.x + threadIdx.x)] = localB[0];" in generated
    )
