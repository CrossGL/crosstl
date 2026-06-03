import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

APPLE_SAMPLE_REPO = "https://github.com/donaldwuid/apple_metal_sample_code"
APPLE_SAMPLE_COMMIT = "0bc50e5b3670b3169855ab260e8da5ff07b53749"
FILAMENT_REPO = "https://github.com/google/filament"
FILAMENT_COMMIT = "48881c840bca50da515f0df82b61c9a5b996b19a"
MOLTENVK_REPO = "https://github.com/KhronosGroup/MoltenVK"
MOLTENVK_COMMIT = "9d5d89a2a1954ae8fe73fd2a97a4f3d862319eee"
SAMPLE_METAL_REPO = "https://github.com/dehesa/sample-metal"
SAMPLE_METAL_COMMIT = "0003824a52516052f2d28503f576907e03425dd3"


EXTERNAL_FIXTURES = [
    {
        "name": "apple_modern_rendering_imported_typedef_cast",
        "repo_url": APPLE_SAMPLE_REPO,
        "commit": APPLE_SAMPLE_COMMIT,
        "source_path": (
            "MetalSampleCodeLibrary/MultipleTechniques/ModernRenderingWithMetal/"
            "Renderer/Shaders/AAPLMeshRenderer.metal"
        ),
        "roundtrip": True,
        "source": (
            """
            struct Camera { float4x4 invViewMatrix; };
            struct Input { float3 position; };
            struct Output { xhalf3 viewDir; };

            vertex Output main_vertex(Input in [[stage_in]],
                                      constant Camera& camera [[buffer(0)]]) {
                Output out;
                out.viewDir =
                    (xhalf3)normalize(camera.invViewMatrix[3].xyz - in.position);
                return out;
            }
        """
        ),
    },
    {
        "name": "apple_raytracing_dependent_types_and_ray_payloads",
        "repo_url": APPLE_SAMPLE_REPO,
        "commit": APPLE_SAMPLE_COMMIT,
        "source_path": (
            "MetalSampleCodeLibrary/RayTracing/"
            "ControlTheRayTracingProcessUsingIntersectionQueries/Renderer/Shaders.metal"
        ),
        "roundtrip": False,
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;
            using namespace raytracing;

            struct BoundingBoxIntersection {
                bool accept;
                float distance;
            };

            typedef BoundingBoxIntersection IntersectionFunction(
                float3, float3, float, float, unsigned int, unsigned int, device void *);
            typedef intersector<triangle_data, instancing, world_space_data>
                ::result_type IntersectionResult;

            IntersectionResult intersect(ray ray,
                                         instance_acceleration_structure accel) {
                intersection_params params;
                intersection_query<triangle_data, instancing> query;
                ray.origin = float3(0.0);

                struct ray shadowRay;
                shadowRay.origin = ray.origin;
                unsigned int keyframe0Index = min((unsigned int)1.0, 2u);
                return query.get_committed_intersection();
            }

            [[intersection(triangle, triangle_data, curve_data, instancing)]]
            bool triangleIntersectionFunction(float3 origin [[origin]],
                                              ray_data float3& normal [[payload]]) {
                return true;
            }
        """
        ),
    },
    {
        "name": "apple_dynamic_library_scoped_member_definition",
        "repo_url": APPLE_SAMPLE_REPO,
        "commit": APPLE_SAMPLE_COMMIT,
        "source_path": (
            "MetalSampleCodeLibrary/Shaders/CreatingAMetalDynamicLibrary/"
            "Renderer/AAPLUserDylib.metal"
        ),
        "roundtrip": True,
        "source": (
            """
            using namespace metal;

            float4 AAPLUserDylib::getFullScreenColor(float4 inColor) {
                return float4(inColor.r, inColor.g, inColor.b, 0);
            }
        """
        ),
    },
    {
        "name": "moltenvk_embedded_watermark_shader",
        "repo_url": MOLTENVK_REPO,
        "commit": MOLTENVK_COMMIT,
        "source_path": "MoltenVK/MoltenVK/Utility/MVKWatermarkShaderSource.h",
        "roundtrip": True,
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            typedef struct { float4x4 mvpMtx; float4 color; } Uniforms;
            typedef struct {
                float2 a_position [[attribute(0)]];
                float2 a_texCoord [[attribute(1)]];
            } Attributes;
            typedef struct {
                float4 v_position [[position]];
                float2 v_texCoord;
                float4 v_fragColor;
            } Varyings;

            vertex Varyings watermarkVertex(
                Attributes attributes [[stage_in]],
                constant Uniforms& uniforms [[buffer(0)]]) {
                Varyings varyings;
                varyings.v_position =
                    uniforms.mvpMtx * float4(attributes.a_position, 0.0, 1.0);
                varyings.v_fragColor = uniforms.color;
                varyings.v_texCoord = attributes.a_texCoord;
                return varyings;
            }

            fragment float4 watermarkFragment(
                Varyings varyings [[stage_in]],
                texture2d<float> texture [[texture(0)]],
                sampler sampler [[sampler(0)]]) {
                return varyings.v_fragColor
                    * texture.sample(sampler, varyings.v_texCoord);
            }
        """
        ),
    },
    {
        "name": "moltenvk_command_shader_typedefs_and_reinterpret_cast",
        "repo_url": MOLTENVK_REPO,
        "commit": MOLTENVK_COMMIT,
        "source_path": (
            "MoltenVK/MoltenVK/Commands/" "MVKCommandPipelineStateFactoryShaderSource.h"
        ),
        "roundtrip": False,
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            typedef struct {
                uint32_t width;
                uint32_t height;
                uint32_t depth;
            } __attribute__((packed)) VkExtent3D;

            typedef struct alignas(8) {
                uint32_t count;
                uint32_t countHigh;
            } VisibilityBuffer;

            kernel void cmdDrawIndirectConvertBuffers(
                const device char* srcBuff [[buffer(0)]],
                device MTLDrawIndexedPrimitivesIndirectArguments* destBuff
                    [[buffer(1)]],
                constant uint32_t& srcStride [[buffer(2)]],
                uint idx [[thread_position_in_grid]]) {
                const device auto& src =
                    *reinterpret_cast<const device
                        MTLDrawPrimitivesIndirectArguments*>(srcBuff + idx * srcStride);
                device auto& dst = destBuff[idx];
                dst.indexCount = src.vertexCount;
            }
        """
        ),
    },
    {
        "name": "filament_sdl_metal_texture_array_sample",
        "repo_url": FILAMENT_REPO,
        "commit": FILAMENT_COMMIT,
        "source_path": "third_party/libsdl2/src/render/metal/SDL_shaders_metal.metal",
        "roundtrip": True,
        "source": (
            """
            #include <metal_texture>
            using namespace metal;

            struct CopyVertexOutput {
                float4 position [[position]];
                float2 texcoord;
            };

            fragment float4 SDL_YUV_fragment(
                CopyVertexOutput vert [[stage_in]],
                constant float4& col [[buffer(0)]],
                texture2d<float> texY [[texture(0)]],
                texture2d_array<float> texUV [[texture(1)]],
                sampler s [[sampler(0)]]) {
                float3 yuv;
                yuv.x = texY.sample(s, vert.texcoord).r;
                yuv.y = texUV.sample(s, vert.texcoord, 0).r;
                yuv.z = texUV.sample(s, vert.texcoord, 1).r;
                return col * float4(yuv, 1.0);
            }
        """
        ),
    },
    {
        "name": "sample_metal_texturing_designated_initializers",
        "repo_url": SAMPLE_METAL_REPO,
        "commit": SAMPLE_METAL_COMMIT,
        "source_path": "Metal By Example/Texturing/Shader.metal",
        "roundtrip": True,
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct VertexInput {
                float4 position [[attribute(0)]];
                float2 texCoords [[attribute(2)]];
            };
            struct Uniforms { float4x4 modelViewProjectionMatrix; };
            struct VertexProjected {
                float4 position [[position]];
                float2 texCoords;
            };

            [[vertex]] VertexProjected main_vertex(
                const VertexInput v [[stage_in]],
                constant Uniforms& u [[buffer(1)]]) {
                return VertexProjected {
                    .position = u.modelViewProjectionMatrix * v.position,
                    .texCoords = v.texCoords
                };
            }

            [[fragment]] float4 main_fragment(
                VertexProjected v [[stage_in]],
                texture2d<float> diffuseTexture [[texture(0)]],
                sampler samplr [[sampler(0)]]) {
                float3 diffuseColor = diffuseTexture.sample(samplr, v.texCoords).rgb;
                return float4(diffuseColor, 1);
            }
        """
        ),
    },
]


def parse_metal(source):
    tokens = MetalLexer(source).tokenize()
    return MetalParser(tokens).parse()


def parse_crossgl(source):
    tokens = CrossGLLexer(source).get_tokens()
    return CrossGLParser(tokens).parse()


@pytest.mark.parametrize(
    "fixture", EXTERNAL_FIXTURES, ids=[fixture["name"] for fixture in EXTERNAL_FIXTURES]
)
def test_external_metal_fixture_parse_and_roundtrip(fixture):
    ast = parse_metal(fixture["source"])
    assert ast.functions or ast.structs or ast.global_variables or ast.typedefs
    assert fixture["repo_url"]
    assert fixture["commit"]
    assert fixture["source_path"]

    if not fixture["roundtrip"]:
        return

    crossgl = MetalToCrossGLConverter().generate(ast)
    assert crossgl.strip()
    parse_crossgl(crossgl)
