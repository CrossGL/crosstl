import pytest

import crosstl.translator as cgl_translator
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser

EXTERNAL_REPOS = {
    "shader-slang/slang": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "adc996670ec281aa8a4ee131f30b324648cbbe60",
    },
    "shader-slang/slang-examples": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "bbc4b02787d427065d713c1820a33b66dc6c4117",
    },
    "shader-slang/slang-generated": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "3dacd6028f380f3f2ff735516d39a1bdc05aeed2",
    },
    "shader-slang/slang-generated-2026": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "a49a125809f7b491bfd54a3014021fa7d716bbdc",
    },
    "shader-slang/slang-wgsl-2026": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "726e0973b3f547c7729b86f122ff7aef8322bace",
    },
    "shader-slang/optix-examples": {
        "url": "https://github.com/shader-slang/optix-examples",
        "commit": "02fa85ae2f39400ced9a602531aa096589055076",
    },
    "shader-slang/slang-rhi": {
        "url": "https://github.com/shader-slang/slang-rhi",
        "commit": "9aa67530649c52b4a057a2fba185cf9247c33ec0",
    },
    "NVIDIAGameWorks/Falcor": {
        "url": "https://github.com/NVIDIAGameWorks/Falcor",
        "commit": "eb540f6748774680ce0039aaf3ac9279266ec521",
    },
    "HenriMichelon/vireo_samples": {
        "url": "https://github.com/HenriMichelon/vireo_samples",
        "commit": "e2d788909cc73a7f515380792796cebaaed53a7e",
    },
    "NVIDIAGameWorks/RTXGI": {
        "url": "https://github.com/NVIDIAGameWorks/RTXGI",
        "commit": "10b5770b8eaddfc1faab82b65f799ac6f47dcc44",
        "note": "searched; current tree has no .slang/.slangh files",
    },
    "nvpro-samples/vk_slang_editor": {
        "url": "https://github.com/nvpro-samples/vk_slang_editor",
        "commit": "620f60e4f724c4d06b1e7251c3fac6ee4d96cb54",
    },
}


EXTERNAL_FIXTURES = [
    {
        "id": "slang_default_parameter",
        "repo": "shader-slang/slang",
        "path": "tests/compute/default-parameter.slang",
        "source": (
            """
            RWStructuredBuffer<int> outputBuffer;

            int helper(int val, int a = 16)
            {
                return val + a;
            }

            int test(int val)
            {
                return helper(val) + helper(val, 256);
            }

            [numthreads(4, 1, 1)]
            void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                outputBuffer[dispatchThreadID.x] = test((int)dispatchThreadID.x);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "int helper(int val, int a)",
            "return helper(val) + helper(val, 256);",
        ],
        "not_contains": ["int a, = , 16", "int a = 16"],
    },
    {
        "id": "slang_autodiff_generic_where_clause",
        "repo": "shader-slang/slang",
        "path": "tests/autodiff/autodiff-generic-where-clause.slang",
        "source": (
            """
            [ForwardDifferentiable]
            T genericCalc<T : IDifferentiable & IFloat>(T val, T x)
            {
                return val * x * x + x;
            }

            void main()
            {
                let result = fwd_diff(genericCalc<float>)(
                    DifferentialPair<float>(2.0, 0.0),
                    DifferentialPair<float>(3.0, 1.0));
                printf("%f\\n", result.d);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "T genericCalc(T val, T x)",
            "let result = fwd_diff(genericCalc<float>)",
            'printf("%f\\n", result.d);',
        ],
    },
    {
        "id": "slang_tbuffer",
        "repo": "shader-slang/slang",
        "path": "tests/hlsl/tbuffer.slang",
        "source": (
            """
            tbuffer tbuf : register(t0)
            {
                float4 tb_val1;
            }

            tbuffer tbuf2 : register(t1)
            {
                Texture2D<float4> texture2D;
                float4 tb_val2;
            }

            RWStructuredBuffer<float4> outputBuffer;

            [numthreads(1, 1, 1)]
            void computeMain()
            {
                outputBuffer[0] = tb_val1 + texture2D[0] + tb_val2;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "cbuffer tbuf @register(t0)",
            "cbuffer tbuf2 @register(t1)",
            "sampler2D texture2D;",
        ],
    },
    {
        "id": "falcor_texture_load",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Tools/FalcorTest/Tests/Core/TextureLoadTests.cs.slang",
        "source": (
            """
            Texture2D<float4> gTex;
            RWStructuredBuffer<float4> result;

            [numthreads(1, 1, 1)]
            void main(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                result[dispatchThreadID.x] = gTex.Load(int3(0, 0, 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2D gTex;",
            "texelFetch(gTex, ivec2(0, 0), 0)",
        ],
        "not_contains": ["gTex.Load"],
    },
    {
        "id": "falcor_enum_class_underlying_type",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/RenderPasses/DebugPasses/ColorMapPass/ColorMapParams.slang",
        "source": (
            """
            enum class ColorMap : uint32_t
            {
                Grey,
                Jet,
                Viridis,
                Plasma,
                Magma,
                Inferno,
            };
        """
        ),
        "crossgl": True,
        "enum_underlying_types": {"ColorMap": "uint32_t"},
        "contains": [
            "enum ColorMap {",
            "Grey,",
            "Inferno,",
        ],
    },
    {
        "id": "falcor_texture_brace_vector_initializer",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Tools/FalcorTest/Tests/Core/TextureLoadTests.cs.slang",
        "source": (
            """
            RWStructuredBuffer<uint4> result;

            Texture2D<float4> texUnorm;
            Texture2D<uint4> texUnormAsUint;
            Texture2D<uint4> texUint;

            [numthreads(256, 1, 1)]
            void testLoadFormat(uint3 threadId: SV_DispatchThreadID)
            {
                float4 f = texUnorm[threadId.xy];
                uint4 u = texUnormAsUint[threadId.xy];
                uint4 v = texUint[threadId.xy];

                result[threadId.x] =
                    { asuint(f.x), u.x, (uint)(f.x * 255.f), v.x };
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2D texUnorm;",
            "result[threadId.x] = {asuint(f.x), u.x, uint(f.x * 255.0), v.x};",
        ],
        "not_contains": ["255.f"],
    },
    {
        "id": "slang_texture2darray_load_int4",
        "repo": "shader-slang/slang",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/texture2darray-load.slang"
        ),
        "source": (
            """
            Texture2DArray<float4> tex;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] = tex.Load(int4(int(tid.x), int(tid.y), 0, 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2DArray tex;",
            "texelFetch(tex, ivec4(int(tid.x), int(tid.y), 0, 0))",
        ],
        "not_contains": ["tex.Load"],
    },
    {
        "id": "slang_texture3d_load_int4",
        "repo": "shader-slang/slang-generated-2026",
        "path": "docs/generated/tests/cross-cutting/core-module/texture3d-load.slang",
        "source": (
            """
            Texture3D<float4> tex;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] =
                    tex.Load(int4(int(tid.x), int(tid.y), int(tid.z), 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler3D tex;",
            "texelFetch(tex, ivec3(int(tid.x), int(tid.y), int(tid.z)), 0)",
        ],
        "not_contains": [
            "tex.Load",
            "ivec4(int(tid.x), int(tid.y), int(tid.z), 0)",
        ],
    },
    {
        "id": "slang_wgsl_texture1d_load_int2",
        "repo": "shader-slang/slang-wgsl-2026",
        "path": "tests/wgsl/texture-load.slang",
        "source": (
            """
            Texture1D<float4> tex;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] = tex.Load(int2(int(tid.x), 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler1D tex;",
            "texelFetch(tex, int(tid.x), 0)",
        ],
        "not_contains": [
            "tex.Load",
            "ivec2(int(tid.x), 0)",
        ],
    },
    {
        "id": "slang_texturecubearray_samplelevel",
        "repo": "shader-slang/slang-generated",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/"
            "texturecubearray-samplelevel.slang"
        ),
        "source": (
            """
            TextureCubeArray<float4> tex;
            SamplerState samp;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] =
                    tex.SampleLevel(samp, float4(1.0, 0.0, 0.0, 0.0), 0.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "samplerCubeArray tex;",
            "sampler samp;",
            "textureLod(tex, samp, vec4(1.0, 0.0, 0.0, 0.0), 0.0)",
        ],
        "not_contains": ["tex.SampleLevel"],
    },
    {
        "id": "slang_hlsl_intrinsic_mul_matrix_vector",
        "repo": "shader-slang/slang-generated-2026",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/"
            "hlsl-intrinsic-mul-matrix-lowers-per-target.slang"
        ),
        "source": (
            """
            RWStructuredBuffer<float3> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                float3x3 m = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
                float3 v = float3(float(tid.x), float(tid.y), float(tid.z));
                outBuf[tid.x] = mul(m, v);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "mat3 m = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);",
            "vec3 v = vec3(float(tid.x), float(tid.y), float(tid.z));",
            "outBuf[tid.x] = (m * v);",
        ],
        "not_contains": ["mul("],
    },
    {
        "id": "slang_groupshared_atomic_add",
        "repo": "shader-slang/slang-generated",
        "path": (
            "docs/generated/tests/ir-reference/resources-and-atomics/"
            "atomic-groupshared-add.slang"
        ),
        "source": (
            """
            groupshared Atomic<int> gsa;
            uniform RWStructuredBuffer<int> rwbuf;

            [numthreads(8, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID,
                      uint3 gid : SV_GroupThreadID)
            {
                int prev = gsa.add((int)gid.x + 1);
                rwbuf[tid.x] = prev;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "Atomic<int> gsa;",
            "RWStructuredBuffer<int> rwbuf;",
            "int prev = gsa.add(int(gid.x) + 1);",
            "rwbuf[tid.x] = prev;",
        ],
    },
    {
        "id": "optix_examples_raygeneration_stages",
        "repo": "shader-slang/optix-examples",
        "path": "example02_pipelineAndRayGen/devicePrograms.slang",
        "source": (
            """
            int frameID;
            RWStructuredBuffer<uint> colorBuffer;
            int2 fbSize;

            [shader("closesthit")]
            void closesthit_radiance()
            {
            }

            [shader("anyhit")]
            void anyhit_radiance()
            {
            }

            [shader("miss")]
            void miss_radiance()
            {
            }

            [shader("raygeneration")]
            void renderFrame()
            {
                const int ix = DispatchRaysIndex().x;
                const int iy = DispatchRaysIndex().y;
                const int r = (ix % 256);
                const int g = (iy % 256);
                const int b = ((ix + iy) % 256);
                const uint rgba = 0xff000000
                    | (r << 0) | (g << 8) | (b << 16);
                const uint fbIndex = ix + iy * fbSize.x;
                colorBuffer[fbIndex] = rgba;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "ray_closest_hit {",
            "ray_any_hit {",
            "ray_miss {",
            "ray_generation {",
            "int b = (ix + iy) % 256;",
            "uint rgba = 0xff000000 | r << 0 | g << 8 | b << 16;",
            "colorBuffer[fbIndex] = rgba;",
        ],
    },
    {
        "id": "slang_examples_ray_payload_access_semantics",
        "repo": "shader-slang/slang-examples",
        "path": "examples/ray-tracing-pipeline/shaders.slang",
        "source": (
            """
            [raypayload] struct RayPayload
            {
                float4 color : read(caller) : write(caller, closesthit, miss);
            };

            RWTexture2D resultTexture;

            [shader("raygeneration")]
            void rayGenShader()
            {
                RayPayload payload = { float4(0, 0, 0, 0) };
                resultTexture[uint2(0, 0)] = payload.color;
            }

            [shader("miss")]
            void missShader(inout RayPayload payload)
            {
                payload.color = float4(0, 0, 0, 1);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "vec4 color @ ray_payload_read(caller) @ ray_payload_write(caller,closesthit,miss);",
            "ray_generation {",
            "resultTexture[uvec2(0, 0)] = payload.color;",
            "payload.color = vec4(0, 0, 0, 1);",
        ],
        "not_contains": ["@ read(caller):write"],
    },
    {
        "id": "falcor_pixel_stats_resource_array",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Falcor/Rendering/Utils/PixelStats.cs.slang",
        "source": (
            """
            import PixelStatsShared;

            cbuffer CB
            {
                uint2 gFrameDim;
            }

            Texture2D<uint> gStatsRayCount[(uint)PixelStatsRayType::Count];
            RWTexture2D<uint> gStatsRayCountTotal;

            [numthreads(16, 16, 1)]
            void main(uint3 dispatchThreadId : SV_DispatchThreadID)
            {
                uint2 pixel = dispatchThreadId.xy;
                if (any(pixel > gFrameDim)) return;

                uint totalRays = 0;
                for (uint i = 0; i < (uint)PixelStatsRayType::Count; i++)
                {
                    totalRays += gStatsRayCount[i][pixel];
                }
                gStatsRayCountTotal[pixel] = totalRays;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "import PixelStatsShared;",
            "sampler2D gStatsRayCount[uint(PixelStatsRayType::Count)];",
            "image2D gStatsRayCountTotal;",
            "totalRays += gStatsRayCount[i][pixel];",
        ],
    },
    {
        "id": "vireo_samples_constantbuffer_gbuffer_sampling",
        "repo": "HenriMichelon/vireo_samples",
        "path": "src/shaders/deferred_lighting.frag.slang",
        "source": (
            """
            struct VertexOutput {
                float4 position : SV_POSITION;
                float2 uv : TEXCOORD;
            };

            struct Global {
                float exposure;
            };

            struct Light {
                float3 color;
            };

            ConstantBuffer<Global> global : register(b0);
            ConstantBuffer<Light> light : register(b1);
            Texture2D positionBuffer : register(t2);
            SamplerState sampler : register(SAMPLER_NEAREST_BORDER, space1);

            float3 calcLighting(Global global, Light light, float3 worldPos)
            {
                return light.color * global.exposure;
            }

            float4 fragmentMain(VertexOutput input) : SV_TARGET
            {
                float3 worldPos =
                    positionBuffer.Sample(sampler, input.uv).rgb;
                float3 lit = calcLighting(global, light, worldPos);
                return float4(lit, 1.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "ConstantBuffer<Global> global_ @register(b0);",
            "ConstantBuffer<Light> light @register(b1);",
            "sampler2D positionBuffer @register(t2);",
            "sampler sampler_;",
            "vec3 worldPos = texture(positionBuffer, sampler_, input.uv).rgb;",
            "vec3 lit = calcLighting(global_, light, worldPos);",
            "return vec4(lit, 1.0);",
        ],
        "not_contains": [
            "@register(SAMPLER_NEAREST_BORDER, space1)",
            "positionBuffer.Sample",
            "global.exposure",
        ],
    },
    {
        "id": "vk_slang_editor_mandelbrot_inout_texture_write",
        "repo": "nvpro-samples/vk_slang_editor",
        "path": "examples/Basics/Mandelbrot.slang",
        "source": (
            """
            RWTexture2D<float4> texFrame;
            uniform float2 iResolution;

            bool mandelbrot(inout float2 z, out int i)
            {
                const float2 c = z;
                for(i = 0; i < 64; i++)
                {
                    if(dot(z, z) > 256.0)
                    {
                        return true;
                    }
                    z = c + float2(z.x * z.x - z.y * z.y, 2 * z.x * z.y);
                }
                return false;
            }

            [shader("compute")]
            [numthreads(16, 16, 1)]
            void render(uint2 thread: SV_DispatchThreadID)
            {
                float2 uv =
                    (float2(thread) - .5 * iResolution.xy) / iResolution.y;
                float2 z = 2.5 * uv - float2(.5, 0);
                int i;
                float3 color = float3(0.0);
                if(mandelbrot(z, i))
                {
                    color = 0.5 + 0.5 * sin(
                        i + float3(0, .5, 1) - log2(log2(dot(z,z))));
                }
                texFrame[thread] = float4(color, 1.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "image2D texFrame;",
            "bool mandelbrot(vec2 z, int i)",
            "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;",
            "texFrame[thread] = vec4(color, 1.0);",
        ],
    },
    {
        "id": "vk_slang_editor_rasterization_semantics",
        "repo": "nvpro-samples/vk_slang_editor",
        "path": "examples/Basics/Rasterization.slang",
        "source": (
            """
            uniform float4x4 iViewProjection;

            struct Vertex {
                float3 position : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 uv0 : TEXCOORD0;
            };
            struct VsOutput {
                Vertex vertex;
                float4 svPosition : SV_Position;
            };

            [shader("vertex")]
            VsOutput vertexMain(Vertex input)
            {
                VsOutput o;
                o.vertex = input;
                o.svPosition = mul(float4(o.vertex.position, 1.0),
                                   iViewProjection);
                return o;
            }

            [shader("fragment")]
            float4 fragmentMain(Vertex v,
                                float2 fragCoord : SV_Position,
                                float3 barycentrics : SV_Barycentrics,
                                uint triangleId : SV_PrimitiveID)
            {
                uint2 xy = uint2(fragCoord);
                bool sierpinski = ((xy.x & xy.y) == 0);
                float3 color = float3(sierpinski);
                return float4(color, 1.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct Vertex {",
            "vec3 position @ in_Position;",
            "vec4 svPosition @ Out_Position;",
            "vec3 barycentrics @ SV_Barycentrics",
            "bool sierpinski = (xy.x & xy.y) == 0;",
        ],
    },
    {
        "id": "slang_rhi_root_parameter_block_attributes",
        "repo": "shader-slang/slang-rhi",
        "path": "tests/test-root-shader-parameter.slang",
        "source": (
            """
            [__AttributeUsage(_AttributeTargets.Var)]
            struct rootAttribute {};

            struct S1
            {
                StructuredBuffer<uint> c0;
                [root] RWStructuredBuffer<uint> c1;
            }

            struct S0
            {
                StructuredBuffer<uint> b0;
                [root] StructuredBuffer<uint> b1;
                ParameterBlock<S1> s1;
                ConstantBuffer<S1> s2;
            }

            ParameterBlock<S0> g;
            [root] RWStructuredBuffer<uint> buffer;

            [shader("compute")]
            [numthreads(1,1,1)]
            void computeMain(uint3 sv_dispatchThreadID : SV_DispatchThreadID)
            {
                buffer[0] = g.b0[0] - g.b1[0] + g.s1.c0[0]
                    - g.s1.c1[0] + g.s2.c0[0];
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct rootAttribute {",
            "ParameterBlock<S0> g;",
            "RWStructuredBuffer<uint> buffer_;",
            "buffer_[0] = g.b0[0] - g.b1[0] + g.s1.c0[0] - g.s1.c1[0] + g.s2.c0[0];",
        ],
        "not_contains": ["[root]", "__AttributeUsage"],
    },
]


def parse_slang(source):
    tokens = SlangLexer(source).tokenize()
    return SlangParser(tokens).parse()


def generate_crossgl(ast):
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda item: item["id"])
def test_external_fixture_codegen_crossgl_reparse(fixture):
    ast = parse_slang(fixture["source"])
    generated = generate_crossgl(ast)

    enum_underlying_types = fixture.get("enum_underlying_types", {})
    if enum_underlying_types:
        enum_types = {
            enum.name: getattr(enum, "underlying_type", None)
            for enum in getattr(ast, "enums", [])
        }
        assert enum_types == enum_underlying_types

    for expected in fixture.get("contains", []):
        assert expected in generated
    for rejected in fixture.get("not_contains", []):
        assert rejected not in generated

    if fixture.get("crossgl", False):
        cgl_translator.parse(generated)


def test_falcor_exported_import_and_default_interface_parameter_parse():
    source = """
        __exported import Rendering.Materials.IMaterialInstance;
        __exported import Scene.Material.TextureSampler;
        import Rendering.Volumes.PhaseFunction;

        [anyValueSize(128)]
        interface IMaterial
        {
            associatedtype MaterialInstance : IMaterialInstance;

            MaterialInstance setupMaterialInstance(
                const MaterialSystem ms,
                const ShadingData sd,
                const ITextureSampler lod,
                const uint hints = (uint)MaterialInstanceHints::None);
        }
    """

    ast = parse_slang(source)
    method = ast.interfaces[0].methods[0]
    hints = method.params[-1]

    assert [node.module_name for node in ast.imports] == [
        "Rendering.Materials.IMaterialInstance",
        "Scene.Material.TextureSampler",
        "Rendering.Volumes.PhaseFunction",
    ]
    assert ast.imports[0].qualifiers == ["__exported"]
    assert ast.imports[1].qualifiers == ["__exported"]
    assert hints.vtype == "uint"
    assert hints.name == "hints"
    assert hints.value.target_type == "uint"
    assert hints.value.expression.name == "MaterialInstanceHints::None"
