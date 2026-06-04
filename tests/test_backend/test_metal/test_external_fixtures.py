import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

APPLE_SAMPLE_REPO = "https://github.com/donaldwuid/apple_metal_sample_code"
APPLE_SAMPLE_COMMIT = "0bc50e5b3670b3169855ab260e8da5ff07b53749"
APPLE_MSL_SPEC_URL = (
    "https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf"
)
APPLE_MSL_SPEC_VERSION = "2025-10-23"
APPLE_REALITYKIT_DOC_URL = (
    "https://developer.apple.com/documentation/realitykit/"
    "custommaterial/surfaceshader/init(named:in:constantvalues:)"
)
APPLE_REALITYKIT_GEOMETRY_DOC_URL = (
    "https://developer.apple.com/documentation/realitykit/"
    "custommaterial/geometrymodifier/init(named:in:constantvalues:)"
)
APPLE_REALITYKIT_DOC_VERSION = "Apple Developer Documentation, accessed 2026-06-04"
FILAMENT_REPO = "https://github.com/google/filament"
FILAMENT_COMMIT = "48881c840bca50da515f0df82b61c9a5b996b19a"
MOLTENVK_REPO = "https://github.com/KhronosGroup/MoltenVK"
MOLTENVK_COMMIT = "9d5d89a2a1954ae8fe73fd2a97a4f3d862319eee"
SAMPLE_METAL_REPO = "https://github.com/dehesa/sample-metal"
SAMPLE_METAL_COMMIT = "0003824a52516052f2d28503f576907e03425dd3"
BOOK_OF_SHADERS_METAL_REPO = "https://github.com/metal-by-example/book-of-shaders-metal"
BOOK_OF_SHADERS_METAL_COMMIT = "12bb2366697cba9c5f660d54fead7bdcd73b6b8a"
MLX_REPO = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "e9e20fa69184bd38cc0ca12bd9a854c059e59588"
MLX_CURRENT_COMMIT = "b155224b9963cd9476363b464a559232a0868000"
MLX_FP_QUANTIZED_COMMIT = "c52b04b650be06291e3a6ff6e98b0ef1af3ff56b"
MLX_STEEL_DEPENDENT_TEMPLATE_COMMIT = "e1a3f2f31fc298cfd7f017d19e8165d88a0c3c59"
PYTORCH_REPO = "https://github.com/pytorch/pytorch"
PYTORCH_BUCKETIZATION_COMMIT = "5ee1f788c7098ae5e50e49543ee7822f73cd8990"
CANDLE_REPO = "https://github.com/huggingface/candle"
CANDLE_COMMIT = "39355c6c9187747e360a2d6ec9d67a2a501b2552"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_COMMIT = "94a220cd6745e6e3f8de62870b66fd5b9bc92700"
PMETAL_REPO = "https://github.com/Epistates/pmetal"
PMETAL_COMMIT = "089171635d1b9c9b7a58b575cf7d522834022cd3"
IMGUI_REPO = "https://github.com/ocornut/imgui"
IMGUI_COMMIT = "7950c96f0e86b761607a34601f19e90afa825bd6"
METALPETAL_REPO = "https://github.com/MetalPetal/MetalPetal"
METALPETAL_COMMIT = "f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb"
BLENDER_REPO = "https://github.com/blender/blender"
BLENDER_COMMIT = "2d196d20b93a9f6e596e6d451c5e845d84f21c89"
BLENDER_STRING_COMMIT = "e5fc656cdab0e682296f8dd024b942b548e788f4"
BLENDER_TEXTURE_READ_COMMIT = "38657e6c5ccb9968bfcc55b4fd384ca528c71d10"
METAL_RIPPLE_REPO = "https://github.com/swiftandcurious/MetalRipple"
METAL_RIPPLE_COMMIT = "125274960b1bf0184b6570afa97f097ee3d2c6b1"


EXTERNAL_FIXTURES = [
    {
        "name": "apple_msl_spec_multiline_barycentric_attribute",
        "repo_url": APPLE_MSL_SPEC_URL,
        "commit": APPLE_MSL_SPEC_VERSION,
        "source_path": (
            "Metal Shading Language Specification, section 2.19 Per-Vertex Values"
        ),
        "roundtrip": False,
        "contains": [
            "vec3 barycentric_coords @barycentric_coord @center_no_perspective;"
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct fragment_in {
                float4 position [[position]];
                float3 barycentric_coords [[barycentric_coord,
                                            center_no_perspective]];
            };
        """
        ),
    },
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
        "name": "apple_raytracing_template_and_function_constants",
        "repo_url": APPLE_SAMPLE_REPO,
        "commit": APPLE_SAMPLE_COMMIT,
        "source_path": (
            "MetalSampleCodeLibrary/RayTracing/"
            "ControlTheRayTracingProcessUsingIntersectionQueries/Renderer/Shaders.metal"
        ),
        "roundtrip": True,
        "contains": [
            "constant uint resourcesStride @function_constant(0);",
            "constant uint[] primes = {2, 3, 5, 7};",
            "T interpolateVertexAttribute(device T* attributes",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            constant unsigned int resourcesStride [[function_constant(0)]];
            constant bool useIntersectionFunctions [[function_constant(1)]];
            constant unsigned int primes[] = { 2, 3, 5, 7 };

            template<typename T>
            inline T interpolateVertexAttribute(device T *attributes,
                                                unsigned int primitiveIndex,
                                                float2 uv) {
                T T0 = attributes[primitiveIndex * 3 + 0];
                T T1 = attributes[primitiveIndex * 3 + 1];
                T T2 = attributes[primitiveIndex * 3 + 2];
                return (1.0f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
            }

            float halton(unsigned int i, unsigned int d) {
                return (float)primes[d];
            }
        """
        ),
    },
    {
        "name": "apple_realitykit_stitchable_scoped_parameter_types",
        "repo_url": APPLE_REALITYKIT_DOC_URL,
        "commit": APPLE_REALITYKIT_DOC_VERSION,
        "source_path": (
            "CustomMaterial.SurfaceShader init(named:in:constantValues:) and "
            f"{APPLE_REALITYKIT_GEOMETRY_DOC_URL}"
        ),
        "roundtrip": True,
        "contains": [
            "void surfaceShader(surface_parameters params) @stitchable",
            "void geometryModifier(geometry_parameters params) @stitchable",
        ],
        "not_contains": [
            "realitykit::surface_parameters",
            "realitykit::geometry_parameters",
        ],
        "source": (
            """
            #include <metal_stdlib>
            #include <RealityKit/RealityKit.h>
            using namespace metal;

            constant bool kEnableBasicLighting [[function_constant(0)]];
            constant bool kEnableVertexAnimation [[function_constant(1)]];

            [[stitchable]]
            void surfaceShader(realitykit::surface_parameters params) {
                float3 baseColor = float3(1, 0, 0);
                if (kEnableBasicLighting) {
                    float3 lighting = params.geometry().normal()
                        * params.geometry().view_direction() * baseColor;
                    params.surface().set_emissive_color(half3(lighting));
                } else {
                    params.surface().set_emissive_color(half3(baseColor));
                }
            }

            [[stitchable]]
            void geometryModifier(realitykit::geometry_parameters params) {
                if (kEnableVertexAnimation) {
                    float currentTime = params.uniforms().time() * 0.5;
                    params.geometry().set_model_position_offset(
                        sin(currentTime) * 0.1 * params.geometry().normal());
                }
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
        "name": "apple_imageblock_pixel_format_member_payload_type",
        "repo_url": APPLE_SAMPLE_REPO,
        "commit": APPLE_SAMPLE_COMMIT,
        "source_path": (
            "MetalSampleCodeLibrary/RenderWorkflows/"
            "ImplementingOrderIndependentTransparencyWithImageBlocks/"
            "Renderer/Shaders/AAPLShaders.metal"
        ),
        "roundtrip": True,
        "contains": [
            "f16vec4[kNumLayers] colors @raster_order_group(0);",
            "f16vec4 layerColor = fragmentValues.colors[0];",
        ],
        "not_contains": ["rgba8unorm<half4>"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            static constexpr constant short kNumLayers = 4;

            struct TransparentFragmentValues {
                rgba8unorm<half4> colors [[raster_order_group(0)]] [kNumLayers];
                half depths [[raster_order_group(0)]] [kNumLayers];
            };

            fragment half4 blendFragments(
                TransparentFragmentValues fragmentValues [[imageblock_data]]) {
                half4 layerColor = fragmentValues.colors[0];
                return layerColor;
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
        "name": "moltenvk_msaa_texture_access_qualifiers",
        "repo_url": MOLTENVK_REPO,
        "commit": MOLTENVK_COMMIT,
        "source_path": (
            "MoltenVK/MoltenVK/Commands/" "MVKCommandPipelineStateFactoryShaderSource.h"
        ),
        "roundtrip": True,
        "contains": [
            "image2D dst @texture(0) @writeonly",
            "sampler2DMS src @texture(1)",
            "imageStore(dst, pos, texelFetch(src, pos, 0));",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            kernel void cmdResolveColorImage2DFloat(
                texture2d<float, access::write> dst [[ texture(0) ]],
                texture2d_ms<float, access::read> src [[ texture(1) ]],
                uint2 pos [[thread_position_in_grid]]) {
                dst.write(src.read(pos, 0), pos);
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
        "name": "moltenvk_atomic_visibility_buffer_accumulation",
        "repo_url": MOLTENVK_REPO,
        "commit": MOLTENVK_COMMIT,
        "source_path": (
            "MoltenVK/MoltenVK/Commands/" "MVKCommandPipelineStateFactoryShaderSource.h"
        ),
        "roundtrip": True,
        "contains": [
            "alignas(8) struct AtomicVisibilityBuffer",
            "atomic_fetch_add_explicit((&dst.count), src.count, memory_order_relaxed);",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            typedef struct alignas(8) {
                uint32_t count;
                uint32_t countHigh;
            } VisibilityBuffer;

            typedef struct alignas(8) {
                atomic_uint count;
                atomic_uint countHigh;
            } AtomicVisibilityBuffer;

            typedef struct alignas(8) {
                uint32_t dst;
                uint32_t src;
            } QueryResultOffsets;

            kernel void accumulateOcclusionQueryResults(
                uint pos [[thread_position_in_grid]],
                const device QueryResultOffsets* offsets [[buffer(0)]],
                device AtomicVisibilityBuffer* dst_buffer [[buffer(1)]],
                const device VisibilityBuffer* src_buffer [[buffer(2)]]) {
                VisibilityBuffer src = src_buffer[offsets[pos].src];
                device AtomicVisibilityBuffer& dst = dst_buffer[offsets[pos].dst];
                uint32_t prev_lo = atomic_fetch_add_explicit(
                    &dst.count, src.count, memory_order_relaxed);
                uint32_t next_lo = prev_lo + src.count;
                atomic_fetch_add_explicit(
                    &dst.countHigh, src.countHigh, memory_order_relaxed);
                if (next_lo < prev_lo) {
                    atomic_fetch_add_explicit(
                        &dst.countHigh, 1, memory_order_relaxed);
                }
            }
        """
        ),
    },
    {
        "name": "filament_sdl_metal_nv12_swizzle_assignment",
        "repo_url": FILAMENT_REPO,
        "commit": FILAMENT_COMMIT,
        "source_path": "third_party/libsdl2/src/render/metal/SDL_shaders_metal.metal",
        "roundtrip": True,
        "contains": [
            "yuv.yz = texture(texUV, s, vert.texcoord).rg;",
            "return col * vec4(dot(yuv, decode.Rcoeff)",
        ],
        "source": (
            """
            #include <metal_texture>
            using namespace metal;

            struct CopyVertexOutput {
                float4 position [[position]];
                float2 texcoord;
            };
            struct YUVDecode {
                float3 offset;
                float3 Rcoeff;
                float3 Gcoeff;
                float3 Bcoeff;
            };

            fragment float4 SDL_NV12_fragment(
                CopyVertexOutput vert [[stage_in]],
                constant float4 &col [[buffer(0)]],
                constant YUVDecode &decode [[buffer(1)]],
                texture2d<float> texY [[texture(0)]],
                texture2d<float> texUV [[texture(1)]],
                sampler s [[sampler(0)]]) {
                float3 yuv;
                yuv.x = texY.sample(s, vert.texcoord).r;
                yuv.yz = texUV.sample(s, vert.texcoord).rg;
                yuv += decode.offset;
                return col * float4(dot(yuv, decode.Rcoeff),
                                    dot(yuv, decode.Gcoeff),
                                    dot(yuv, decode.Bcoeff),
                                    1.0);
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
    {
        "name": "book_of_shaders_global_vector_brace_initializers",
        "repo_url": BOOK_OF_SHADERS_METAL_REPO,
        "commit": BOOK_OF_SHADERS_METAL_COMMIT,
        "source_path": "BookOfShaders/Shaders/06a-color-mix.metal",
        "roundtrip": True,
        "contains": [
            "constant vec3 colorA = vec3(0.000f, 0.129f, 0.647f);",
            "vec3 color = mix(colorA, colorB, fraction);",
            "vec4 fragment_main(FragmentIn in_, constant Uniforms& uniforms @buffer(0))",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct Uniforms {
                float2 resolution;
                float2 mouse;
                float time;
            };
            struct FragmentIn {
                float4 position [[position]];
                float2 st;
            };

            constant float3 colorA { 0.000f, 0.129f, 0.647f };
            constant float3 colorB { 0.980f, 0.275f, 0.090f };

            [[fragment]]
            float4 fragment_main(FragmentIn in [[stage_in]],
                                 constant Uniforms &uniforms [[buffer(0)]]) {
                float fraction = sin(uniforms.time) * 0.5 + 0.5f;
                float3 color = mix(colorA, colorB, fraction);
                return float4(color, 1.0);
            }
        """
        ),
    },
    {
        "name": "mlx_axpby_template_static_cast_buffer_store",
        "repo_url": MLX_REPO,
        "commit": MLX_COMMIT,
        "source_path": "examples/extensions/axpby/axpby.metal",
        "roundtrip": True,
        "contains": [
            "StructuredBuffer<T> x @buffer(0)",
            "RWStructuredBuffer<T> out_ @buffer(2)",
            "buffer_store(out_, index, T(alpha) * buffer_load(x, index) + T(beta) * buffer_load(y, index));",
        ],
        "not_contains": ["static_cast", "device const"],
        "source": (
            """
            #include <metal_stdlib>

            template <typename T>
            [[kernel]] void axpby_contiguous(
                device const T* x [[buffer(0)]],
                device const T* y [[buffer(1)]],
                device T* out [[buffer(2)]],
                constant const float& alpha [[buffer(3)]],
                constant const float& beta [[buffer(4)]],
                uint index [[thread_position_in_grid]]) {
                out[index] = static_cast<T>(alpha) * x[index]
                    + static_cast<T>(beta) * y[index];
            }
        """
        ),
    },
    {
        "name": "mlx_rope_metal_fast_math_namespace_intrinsics",
        "repo_url": MLX_REPO,
        "commit": MLX_COMMIT,
        "source_path": "mlx/backend/metal/kernels/rope.metal",
        "roundtrip": True,
        "contains": [
            "float costheta = cos(theta);",
            "float sintheta = sin(theta);",
            "float inv_freq = exp2((-d) * base);",
        ],
        "not_contains": [
            "metal_u3a_u3afast_u3a_u3acos",
            "metal_u3a_u3afast_u3a_u3asin",
            "metal_u3a_u3aexp2",
        ],
        "source": (
            """
            #include <metal_math>
            using namespace metal;

            float2 rope_angles(float theta, float d, float base) {
                float costheta = metal::fast::cos(theta);
                float sintheta = metal::fast::sin(theta);
                float inv_freq = metal::exp2(-d * base);
                return float2(costheta, sintheta + inv_freq);
            }
        """
        ),
    },
    {
        "name": "mlx_fft_steel_const_function_constants",
        "repo_url": MLX_REPO,
        "commit": MLX_COMMIT,
        "source_path": "mlx/backend/metal/kernels/fft.h",
        "roundtrip": True,
        "contains": [
            "constant bool inv_ @function_constant(0);",
            "constant int elems_per_thread_ @function_constant(2);",
        ],
        "not_contains": ["STEEL_CONST"],
        "source": (
            """
            #include <metal_common>
            #include "mlx/backend/metal/kernels/steel/defines.h"
            using namespace metal;

            STEEL_CONST bool inv_ [[function_constant(0)]];
            STEEL_CONST int elems_per_thread_ [[function_constant(2)]];

            float2 apply_fft_flag(float2 value) {
                if (inv_) {
                    value.y = -value.y;
                }
                return value * float(elems_per_thread_);
            }
        """
        ),
    },
    {
        "name": "mlx_fp_quantized_if_constexpr_dequantize_scale",
        "repo_url": MLX_REPO,
        "commit": MLX_FP_QUANTIZED_COMMIT,
        "source_path": "mlx/backend/metal/kernels/fp_quantized.h",
        "roundtrip": True,
        "contains": [
            "T dequantize_scale(uint8 s)",
            "if (group_size == 16)",
        ],
        "not_contains": ["if constexpr"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template <typename T, int group_size>
            static inline T dequantize_scale(uint8_t s) {
                if constexpr (group_size == 16) {
                    return T(*(thread fp8_e4m3*)(&s));
                } else {
                    return T(*(thread fp8_e8m0*)(&s));
                }
            }
        """
        ),
    },
    {
        "name": "mlx_fp4_builtin_conversion_operators",
        "repo_url": MLX_REPO,
        "commit": MLX_CURRENT_COMMIT,
        "source_path": "mlx/backend/metal/kernels/fp4.h",
        "roundtrip": True,
        "struct_names": ["fp4_e2m1"],
        "contains": [
            "struct fp4_e2m1",
            "uint8 bits;",
        ],
        "not_contains": ["operator float"],
        "source": (
            """
            struct fp4_e2m1 {
                operator float16_t() {
                    half converted = as_type<half>(ushort((bits & 7) << 9));
                    return bits & 8 ? -converted : converted;
                }

                operator float() {
                    return static_cast<float>(this->operator float16_t());
                }

                operator bfloat16_t() {
                    return static_cast<bfloat16_t>(this->operator float16_t());
                }

                uint8_t bits;
            };
        """
        ),
    },
    {
        "name": "mlx_steel_dependent_template_type_disambiguator",
        "repo_url": MLX_REPO,
        "commit": MLX_STEEL_DEPENDENT_TEMPLATE_COMMIT,
        "source_path": (
            "mlx/backend/metal/kernels/steel/gemm/kernels/" "steel_gemm_fused_nax.h"
        ),
        "roundtrip": True,
        "struct_names": ["TransformNone"],
        "contains": [
            "struct TransformNone",
            "cfrag_t celems;",
        ],
        "not_contains": ["::template"],
        "source": (
            """
            // Reduced from MLX Steel GEMM fused NAX TransformNone:
            // using cfrag_t = typename CFrag::template dtype_frag_t<T>;
            using namespace mlx::steel;

            template <typename T, typename NAXTile_t>
            struct TransformNone {
                using CFrag = typename NAXTile_t::NAXFrag_t;
                using cfrag_t = typename CFrag::template dtype_frag_t<T>;

                cfrag_t celems;
            };
        """
        ),
    },
    {
        "name": "mlx_steel_attention_block_scope_decltype_alias",
        "repo_url": MLX_REPO,
        "commit": MLX_STEEL_DEPENDENT_TEMPLATE_COMMIT,
        "source_path": (
            "mlx/backend/metal/kernels/steel/attn/kernels/" "steel_attention.h"
        ),
        "roundtrip": True,
        "contains": [
            "const auto neg_inf = Limits_u3cselem_t_u3e_u3a_u3afinite_min;",
            "const int16 kRowsPT = stile_t_u3a_u3akRowsPerThread;",
        ],
        "source": (
            """
            void resolve_mask() {
                // Reduced from MLX Steel attention mask handling:
                // using stile_t = decltype(Stile);
                // constexpr auto neg_inf = Limits<selem_t>::finite_min;
                int Stile;
                using stile_t = decltype(Stile);
                using selem_t = typename stile_t::elem_type;
                constexpr auto neg_inf = Limits<selem_t>::finite_min;
                constexpr short kRowsPT = stile_t::kRowsPerThread;
            }
        """
        ),
    },
    {
        "name": "pytorch_bucketization_threadgroups_per_grid",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_BUCKETIZATION_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/Bucketization.metal",
        "roundtrip": True,
        "contains": [
            "uvec2 tgpg @gl_NumWorkGroups",
            "tid += tptg.x * tgpg.x",
        ],
        "not_contains": ["@threadgroups_per_grid"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template <typename input_t, typename output_t>
            kernel void searchsorted(
                constant input_t* data_in [[buffer(0)]],
                device output_t* data_out [[buffer(2)]],
                constant int64_t& numel_in [[buffer(5)]],
                uint2 tgid [[threadgroup_position_in_grid]],
                uint2 tid2 [[thread_position_in_threadgroup]],
                uint2 tptg [[threads_per_threadgroup]],
                uint2 tgpg [[threadgroups_per_grid]]) {
                for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in;
                     tid += tptg.x * tgpg.x) {
                    data_out[tid] = output_t(data_in[tid]);
                }
            }
        """
        ),
    },
    {
        "name": "mlx_gemv_template_struct_alias_specialization",
        "repo_url": MLX_REPO,
        "commit": MLX_COMMIT,
        "source_path": "mlx/backend/metal/kernels/gemv.metal",
        "roundtrip": False,
        "struct_names": ["DefaultAccT", "DefaultAccT<complex64_t>"],
        "source": (
            """
            template <typename U>
            struct DefaultAccT {
                using type = float;
            };

            template <>
            struct DefaultAccT<complex64_t> {
                using type = complex64_t;
            };
        """
        ),
    },
    {
        "name": "mlx_complex_top_level_operator_overloads",
        "repo_url": MLX_REPO,
        "commit": MLX_COMMIT,
        "source_path": "mlx/backend/metal/kernels/complex.h",
        "roundtrip": False,
        "struct_names": ["complex64_t"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct complex64_t {
                float real;
                float imag;
            };

            constexpr complex64_t operator-(complex64_t x) {
                return {-x.real, -x.imag};
            }

            constexpr bool operator>=(complex64_t a, complex64_t b) {
                return (a.real > b.real)
                    || (a.real == b.real && a.imag >= b.imag);
            }
        """
        ),
    },
    {
        "name": "candle_gemv_struct_static_assert",
        "repo_url": CANDLE_REPO,
        "commit": CANDLE_COMMIT,
        "source_path": "candle-metal-kernels/src/metal_src/gemv.metal",
        "roundtrip": True,
        "struct_names": ["GemvDefaultAccT", "GEMVKernel"],
        "contains": [
            "struct GEMVKernel",
            "constant int threadsM;",
            'static_assert(SM * SN == 32, "simdgroup must have 32 threads");',
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            #define MLX_MTL_CONST static constant constexpr const

            template <typename U>
            struct GemvDefaultAccT {
                using type = float;
            };

            template <
                typename T,
                const int BM,
                const int SM,
                const int SN,
                typename AccT = typename GemvDefaultAccT<T>::type>
            struct GEMVKernel {
                using acc_type = AccT;
                MLX_MTL_CONST int threadsM = BM * SM;
                static_assert(SM * SN == 32,
                              "simdgroup must have 32 threads");
            };
        """
        ),
    },
    {
        "name": "llama_cpp_bfloat_matrix_typedef_dimensions",
        "repo_url": LLAMA_CPP_REPO,
        "commit": LLAMA_CPP_COMMIT,
        "source_path": "ggml/src/ggml-metal/ggml-metal.metal",
        "roundtrip": True,
        "contains": [
            "typedef matrix<bfloat,4,4> bfloat4x4;",
            "typedef matrix<bfloat,2,4> bfloat2x4;",
            "void kernel_memset(constant ggml_metal_kargs_memset& args",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            typedef matrix<bfloat, 4, 4> bfloat4x4;
            typedef matrix<bfloat, 2, 4> bfloat2x4;

            template<typename T>
            kernel void kernel_memset(
                    constant ggml_metal_kargs_memset & args,
                    device T * dst,
                    uint tpig[[thread_position_in_grid]]) {
                dst[tpig] = args.val;
            }
        """
        ),
    },
    {
        "name": "imgui_backend_uchar_color_half_sample",
        "repo_url": IMGUI_REPO,
        "commit": IMGUI_COMMIT,
        "source_path": "backends/imgui_impl_metal.mm",
        "roundtrip": True,
        "contains": [
            "u8vec4 color @TANGENT;",
            "out_.color = vec4(in_.color) / vec4(255.0);",
            "f16vec4 texColor = texture(texture_, textureSampler, in_.texCoords);",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct Uniforms { float4x4 projectionMatrix; };
            struct VertexIn {
                float2 position [[attribute(0)]];
                float2 texCoords [[attribute(1)]];
                uchar4 color [[attribute(2)]];
            };
            struct VertexOut {
                float4 position [[position]];
                float2 texCoords;
                float4 color;
            };

            vertex VertexOut vertex_main(
                VertexIn in [[stage_in]],
                constant Uniforms &uniforms [[buffer(1)]]) {
                VertexOut out;
                out.position =
                    uniforms.projectionMatrix * float4(in.position, 0, 1);
                out.texCoords = in.texCoords;
                out.color = float4(in.color) / float4(255.0);
                return out;
            }

            fragment half4 fragment_main(
                VertexOut in [[stage_in]],
                texture2d<half, access::sample> texture [[texture(0)]],
                sampler textureSampler [[sampler(0)]]) {
                half4 texColor = texture.sample(textureSampler, in.texCoords);
                return half4(in.color) * texColor;
            }
        """
        ),
    },
    {
        "name": "metalpetal_yuv_conversion_namespace_sampler_textures",
        "repo_url": METALPETAL_REPO,
        "commit": METALPETAL_COMMIT,
        "source_path": "Frameworks/MetalPetal/Shaders/ColorConversionShaders.metal",
        "roundtrip": True,
        "contains": [
            "StructuredBuffer<Vertex> verticies @buffer(0)",
            "const sampler s = sampler(address::clamp_to_edge, filter::linear);",
            "image2D outTexture @texture(2) @writeonly",
            "imageStore(outTexture, gid, vec4(vec3(rgb), 1.0));",
        ],
        "source": (
            """
            #include <metal_stdlib>
            #include <simd/simd.h>
            #include "MTIShaderLib.h"

            using namespace metal;

            namespace metalpetal {
                namespace yuv2rgbconvert {
                    typedef struct {
                        packed_float2 position;
                        packed_float2 texcoord;
                    } Vertex;

                    typedef struct {
                        float3x3 matrix;
                        float3 offset;
                    } ColorConversion;

                    typedef struct {
                        float4 position [[ position ]];
                        float2 texcoord;
                    } Varyings;

                    vertex Varyings colorConversionVertex(
                        const device Vertex * verticies [[ buffer(0) ]],
                        unsigned int vid [[ vertex_id ]]) {
                        Varyings out;
                        Vertex v = verticies[vid];
                        out.position = float4(float2(v.position), 0.0, 1.0);
                        out.texcoord = v.texcoord;
                        return out;
                    }

                    fragment float4 colorConversionFragment(
                        Varyings in [[ stage_in ]],
                        texture2d<float, access::sample> yTexture [[ texture(0) ]],
                        texture2d<float, access::sample> cbcrTexture [[ texture(1) ]],
                        constant ColorConversion &colorConversion [[ buffer(0) ]],
                        constant bool &convertToLinearRGB [[ buffer(1) ]]) {
                        constexpr sampler s(
                            address::clamp_to_edge, filter::linear);
                        float3 ycbcr = float3(
                            yTexture.sample(s, in.texcoord).r,
                            cbcrTexture.sample(s, in.texcoord).rg);
                        float3 rgb =
                            colorConversion.matrix *
                            (ycbcr + colorConversion.offset);
                        return float4(
                            float3(
                                convertToLinearRGB ? sRGBToLinear(rgb) : rgb),
                            1.0);
                    }

                    kernel void colorConversion(
                        uint2 gid [[ thread_position_in_grid ]],
                        texture2d<float, access::read> yTexture [[ texture(0) ]],
                        texture2d<float, access::read> cbcrTexture [[ texture(1) ]],
                        texture2d<float, access::write> outTexture [[ texture(2) ]],
                        constant ColorConversion &colorConversion [[ buffer(0) ]]) {
                        uint2 cbcrCoordinates = uint2(gid.x / 2, gid.y / 2);
                        float y = yTexture.read(gid).r;
                        float2 cbcr = cbcrTexture.read(cbcrCoordinates).rg;
                        float3 ycbcr = float3(y, cbcr);
                        float3 rgb =
                            colorConversion.matrix *
                            (ycbcr + colorConversion.offset);
                        outTexture.write(float4(float3(rgb), 1.0), gid);
                    }
                }
            }
        """
        ),
    },
    {
        "name": "blender_msl_string_friend_operator",
        "repo_url": BLENDER_REPO,
        "commit": BLENDER_STRING_COMMIT,
        "source_path": "source/blender/gpu/shaders/gpu_shader_msl_string.msl",
        "roundtrip": True,
        "contains": [
            "struct string_t",
            "uint hash;",
            "uint as_uint(string_t str)",
            "return str.hash;",
        ],
        "not_contains": ["friend", "operator=="],
        "source": (
            """
            struct string_t {
              uint hash;

              string_t(uint hash_) : hash(hash_) {}

              friend bool operator==(string_t a, string_t b)
              {
                return a.hash == b.hash;
              }
            };

            uint as_uint(string_t str)
            {
              return str.hash;
            }
        """
        ),
    },
    {
        "name": "blender_texture_update_generic_vec_write",
        "repo_url": BLENDER_REPO,
        "commit": BLENDER_COMMIT,
        "source_path": "source/blender/gpu/metal/kernels/compute_texture_update.msl",
        "roundtrip": True,
        "contains": [
            "int[3] extent;",
            "StructuredBuffer<float> input_data @buffer(1)",
            "vec4 output;",
            "output[i] = float(buffer_load(input_data, index + i));",
            "imageStore(update_tex, uvec2(params.offset[0], params.offset[1]) + uvec2(xx, yy), output);",
        ],
        "not_contains": ["vec<float> output"],
        "source": (
            """
            using namespace metal;

            struct TextureUpdateParams {
                int mip_index;
                int extent[3];
                int offset[3];
                uint unpack_row_length;
            };

            kernel void compute_texture_update(
                constant TextureUpdateParams &params [[buffer(0)]],
                constant float *input_data [[buffer(1)]],
                texture2d<float, access::write> update_tex [[texture(0)]],
                uint2 position [[thread_position_in_grid]]) {
                uint xx = position[0];
                uint yy = position[1];
                int index = (yy * params.unpack_row_length + xx) * 4;

                vec<float, 4> output;
                for (int i = 0; i < 4; i++) {
                    output[i] = float(input_data[index + i]);
                }
                update_tex.write(
                    output,
                    uint2(params.offset[0], params.offset[1]) + uint2(xx, yy));
            }
        """
        ),
    },
    {
        "name": "blender_texture_read_template_specialization_array_read",
        "repo_url": BLENDER_REPO,
        "commit": BLENDER_TEXTURE_READ_COMMIT,
        "source_path": "source/blender/gpu/metal/kernels/compute_texture_read.msl",
        "roundtrip": True,
        "contains": [
            "uint convert_type_u3cuint_u3e(float val)",
            "image2DArray read_tex @texture(0) @readonly",
            "imageLoad(read_tex, uvec3(uvec2(params.offset[0], params.offset[1]) + uvec2(xx, yy), uint(params.offset[2] + layer)))",
            "buffer_store(output_data, index + i, convert_type_u3cuint_u3e(read_colour[i]));",
        ],
        "not_contains": ["convert_type<uint>"],
        "source": (
            """
            using namespace metal;

            template<typename T> T convert_type(float type)
            {
                return T(type);
            }

            template<> uint convert_type<uint>(float val)
            {
                return uint(val * float(0xFFFFFFFFu));
            }

            struct TextureReadParams {
                int extent[3];
                int offset[3];
            };

            kernel void compute_texture_read(
                constant TextureReadParams &params [[buffer(0)]],
                device uint *output_data [[buffer(1)]],
                texture2d_array<float, access::read> read_tex [[texture(0)]],
                uint3 position [[thread_position_in_grid]]) {
                uint xx = position[0];
                uint yy = position[1];
                uint layer = position[2];
                int index =
                    (layer * (params.extent[0] * params.extent[1])
                        + yy * params.extent[0] + xx) * 4;
                float4 read_colour = read_tex.read(
                    uint2(params.offset[0], params.offset[1]) + uint2(xx, yy),
                    uint(params.offset[2] + layer));
                for (int i = 0; i < 4; i++) {
                    output_data[index + i] =
                        convert_type<uint>(read_colour[i]);
                }
            }
        """
        ),
    },
    {
        "name": "pmetal_fused_lora_threadgroup_casted_half4_load",
        "repo_url": PMETAL_REPO,
        "commit": PMETAL_COMMIT,
        "source_path": "crates/pmetal-metal/src/kernels/metal/fused_lora.metal",
        "roundtrip": False,
        "contains": [
            "constant uint TILE_M @function_constant(0);",
            "threadgroup float* scratch @threadgroup(0)",
            "vec4((*(f16vec4*)(x_row + simd_lane_id)))",
            "vec4((*(f16vec4*)(A + simd_lane_id)))",
            "xA_tile[tid.x] = acc_a4.x;",
        ],
        "not_contains": [
            "(*(f16vec4*)x_row + simd_lane_id)",
            "(*(f16vec4*)A + simd_lane_id)",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct FusedLoraParams { uint rank; };
            constant uint TILE_M [[function_constant(0)]];
            constant uint SIMD_SIZE = 32;

            kernel void fused_lora_forward(
                device const half* x [[buffer(0)]],
                device const half* A [[buffer(2)]],
                device half* xA [[buffer(5)]],
                constant FusedLoraParams& params [[buffer(6)]],
                threadgroup float* scratch [[threadgroup(0)]],
                uint3 tid [[thread_position_in_threadgroup]],
                uint simd_lane_id [[thread_index_in_simdgroup]]) {
                threadgroup float* xA_tile = scratch;
                device const half* x_row = x + tid.x * TILE_M;
                float4 acc_a4 =
                    float4(*(device const half4*)(x_row + simd_lane_id))
                    * float4(*(device const half4*)(A + simd_lane_id));
                xA_tile[tid.x] = acc_a4.x;
                xA[tid.x] = half(params.rank);
            }
        """
        ),
    },
    {
        "name": "metal_ripple_swiftui_layer_samplerless_sample",
        "repo_url": METAL_RIPPLE_REPO,
        "commit": METAL_RIPPLE_COMMIT,
        "source_path": "MetalRipple/RippleEffect/Ripple.metal",
        "roundtrip": True,
        "contains": [
            "sampler2D layer",
            "f16vec4 color = texture(layer, newPosition);",
        ],
        "not_contains": [
            "SwiftUI::Layer",
            "texture(layer, newPosition, )",
        ],
        "source": (
            """
            #include <metal_stdlib>
            #include <SwiftUI/SwiftUI.h>
            using namespace metal;

            [[ stitchable ]]
            half4 Ripple(
                float2 position,
                SwiftUI::Layer layer,
                float2 origin,
                float time,
                float amplitude,
                float frequency,
                float decay,
                float speed) {
                float distance = length(position - origin);
                float delay = distance / speed;
                time -= delay;
                time = max(0.0, time);
                float rippleAmount =
                    amplitude * sin(frequency * time) * exp(-decay * time);
                float2 n = normalize(position - origin);
                float2 newPosition = position + rippleAmount * n;
                half4 color = layer.sample(newPosition);
                color.rgb += 0.3 * (rippleAmount / amplitude) * color.a;
                return color;
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
    struct_names = {struct.name for struct in ast.structs}
    for expected in fixture.get("struct_names", []):
        assert expected in struct_names

    should_generate = (
        fixture["roundtrip"] or fixture.get("contains") or fixture.get("not_contains")
    )
    if should_generate:
        crossgl = MetalToCrossGLConverter().generate(ast)
        assert crossgl.strip()
        for expected in fixture.get("contains", []):
            assert expected in crossgl
        for rejected in fixture.get("not_contains", []):
            assert rejected not in crossgl

    if fixture["roundtrip"]:
        parse_crossgl(crossgl)
