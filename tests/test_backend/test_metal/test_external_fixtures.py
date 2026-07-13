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
WGPU_REPO = "https://github.com/gfx-rs/wgpu"
WGPU_COMMIT = "26e2525f8dea477ef356b80efb6eb1bc1dec120d"
DAWN_REPO = "https://github.com/google/dawn"
DAWN_COMMIT = "78a171ad2ed7f7265cfc3dd52e4e7a637a099df0"
MLX_REPO = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "e9e20fa69184bd38cc0ca12bd9a854c059e59588"
MLX_CURRENT_COMMIT = "b155224b9963cd9476363b464a559232a0868000"
MLX_FP_QUANTIZED_COMMIT = "c52b04b650be06291e3a6ff6e98b0ef1af3ff56b"
MLX_FP_QUANTIZED_VALUE_TEMPLATE_COMMIT = "6ea7a00d05d548219864d10ff6c013b7544b13ea"
MLX_STEEL_DEPENDENT_TEMPLATE_COMMIT = "e1a3f2f31fc298cfd7f017d19e8165d88a0c3c59"
MLX_UNARY_OPS_COMMIT = "e1a3f2f31fc298cfd7f017d19e8165d88a0c3c59"
MLX_STEEL_ATTENTION_TYPE_TRAIT_COMMIT = "6ea7a00d05d548219864d10ff6c013b7544b13ea"
MLX_STEEL_GEMM_LOADER_COMMIT = "6ea7a00d05d548219864d10ff6c013b7544b13ea"
MLX_UTILS_TYPEDEF_COMMIT = "8f0e8b14e0fc028df8618684583af9bef44647b8"
MLX_SCAN_COMMIT = "8f0e8b14e0fc028df8618684583af9bef44647b8"
PYTORCH_REPO = "https://github.com/pytorch/pytorch"
PYTORCH_GRID_SAMPLER_COMMIT = "7168b60c0d3561d93aac7519d03d1bd95ee3e7a3"
PYTORCH_POOLING_COMMIT = "7168b60c0d3561d93aac7519d03d1bd95ee3e7a3"
PYTORCH_BUCKETIZATION_COMMIT = "5ee1f788c7098ae5e50e49543ee7822f73cd8990"
PYTORCH_ACTIVATION_COMMIT = "fa5cb72912c44b22acd9c26c69f3e933794ac501"
PYTORCH_C10_METAL_CONSTEXPR_COMMIT = "fa5cb72912c44b22acd9c26c69f3e933794ac501"
PYTORCH_CURRENT_COMMIT = "474a11a166e1313c37a9ad6f5ed0c887409d2cfc"
CANDLE_REPO = "https://github.com/huggingface/candle"
CANDLE_COMMIT = "39355c6c9187747e360a2d6ec9d67a2a501b2552"
TINYGRAD_REPO = "https://github.com/tinygrad/tinygrad"
TINYGRAD_THUNDER_COMMIT = "623b66e0e4e8e519038f9f5cd86a8ab6976032c8"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_COMMIT = "94a220cd6745e6e3f8de62870b66fd5b9bc92700"
LLAMA_CPP_GGML_COMMON_COMMIT = "e3471b3e7306fe120dc8f38a2263c1293fc2add7"
LLAMA_CPP_METAL_DEVICE_COMMIT = "f0152efe401acd5b329b5f62d87dc070a6d069f0"
LLAMA_CPP_METAL_CONTEXT_COMMIT = "d6d0ce8215a1c324e8de04b52f9dd65c5edc129f"
PMETAL_REPO = "https://github.com/Epistates/pmetal"
PMETAL_COMMIT = "089171635d1b9c9b7a58b575cf7d522834022cd3"
IMGUI_REPO = "https://github.com/ocornut/imgui"
IMGUI_COMMIT = "7950c96f0e86b761607a34601f19e90afa825bd6"
METALPETAL_REPO = "https://github.com/MetalPetal/MetalPetal"
METALPETAL_COMMIT = "f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb"
WEBKIT_REPO = "https://github.com/WebKit/WebKit"
WEBKIT_COMMIT = "7f16a4cd45bf6fe4c1b0bcdddec0b1bdee6859c1"
BLENDER_REPO = "https://github.com/blender/blender"
BLENDER_COMMIT = "2d196d20b93a9f6e596e6d451c5e845d84f21c89"
BLENDER_STRING_COMMIT = "e5fc656cdab0e682296f8dd024b942b548e788f4"
BLENDER_TEXTURE_READ_COMMIT = "38657e6c5ccb9968bfcc55b4fd384ca528c71d10"
METAL_RIPPLE_REPO = "https://github.com/swiftandcurious/MetalRipple"
METAL_RIPPLE_COMMIT = "125274960b1bf0184b6570afa97f097ee3d2c6b1"
TIMDECODE_SIMD_PARTITION_GIST = (
    "https://gist.github.com/timdecode/2aa10535b65dab08df78655d560983fb"
)
TIMDECODE_SIMD_PARTITION_VERSION = "808e31f98b1b72fd4cd322a7399dde40170240d0"
UNIXZII_GPU_PARTICLE_GIST = (
    "https://gist.github.com/unixzii/aeefe8edbd6a685cb3e230b5b30841db"
)
UNIXZII_GPU_PARTICLE_VERSION = "GitHub Gist, last active 2024-03-30"


def _dawn_deep_tint_array_initializer_source(depth=60):
    def nested_type(level):
        type_name = "int"
        for _ in range(level):
            type_name = f"tint_array<{type_name}, 1>"
        return type_name

    def nested_value(level):
        value = "-6"
        for index in range(level):
            type_name = nested_type(index + 1)
            value = f"{type_name}{{{value}}}"
        return value

    return f"""
        #include <metal_stdlib>
        using namespace metal;

        template<typename T, size_t N>
        struct tint_array {{
            T elements[N];
        }};

        kernel void f() {{
            thread {nested_type(depth)} arr = {nested_value(depth)};
        }}
    """


EXTERNAL_FIXTURES = [
    {
        "name": "apple_msl_spec_multiline_barycentric_attribute",
        "repo_url": APPLE_MSL_SPEC_URL,
        "commit": APPLE_MSL_SPEC_VERSION,
        "source_path": (
            "Metal Shading Language Specification, section 2.19 Per-Vertex Values"
        ),
        "roundtrip": False,
        "contains": ["vec3 barycentric_coords @gl_BaryCoordNoPerspEXT;"],
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
        "name": "metal_scoped_member_definition_with_trailing_const",
        "repo_url": APPLE_MSL_SPEC_URL,
        "commit": APPLE_MSL_SPEC_VERSION,
        "source_path": "Metal Shading Language Specification, C++ function syntax",
        "roundtrip": True,
        "contains": [
            "struct ToneMapper {",
            "float ToneMapper_u3a_u3aapply(float value)",
            "return value * exposure;",
        ],
        "source": (
            """
            using namespace metal;

            struct ToneMapper {
                float exposure;
                float apply(float value) const;
            };

            float ToneMapper::apply(float value) const {
                return value * exposure;
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
        "name": "apple_imageblock_symbolic_color_attachment_parameter",
        "repo_url": APPLE_SAMPLE_REPO,
        "commit": APPLE_SAMPLE_COMMIT,
        "source_path": (
            "MetalSampleCodeLibrary/RenderWorkflows/"
            "ImplementingOrderIndependentTransparencyWithImageBlocks/"
            "Renderer/Shaders/AAPLShaders.metal"
        ),
        "roundtrip": True,
        "contains": ["f16vec4 forwardOpaqueColor @gl_FragColor @raster_order_group(0)"],
        "not_contains": ["@color(AAPLRenderTargetColor)"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            typedef struct {
                float4 position [[position]];
            } ColorInOut;

            vertex ColorInOut quadPassVertex(uint vid [[vertex_id]]) {
                ColorInOut out;
                out.position = float4(0.0);
                return out;
            }

            struct TransparentFragmentValues {
                half4 colors [[raster_order_group(0)]] [4];
            };

            fragment half4 blendFragments(
                TransparentFragmentValues fragmentValues [[imageblock_data]],
                half4 forwardOpaqueColor [[color(AAPLRenderTargetColor),
                                           raster_order_group(0)]]) {
                return forwardOpaqueColor;
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
            "atomicAdd(dst.count, src.count);",
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
            "vec4 fragment_main(FragmentIn in_, constant Uniforms uniforms @buffer(0))",
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
        "name": "wgpu_naga_bare_coherent_parameter_qualifier",
        "repo_url": WGPU_REPO,
        "commit": WGPU_COMMIT,
        "source_path": "naga/tests/out/msl/wgsl-memory-decorations-coherent.metal",
        "roundtrip": True,
        "contains": [
            "void main_(inout device Data coherent_buf @user(fake0), "
            "device Data plain_buf @user(fake0))",
            "coherent_buf.values[0] = value;",
        ],
        "not_contains": ["coherent device"],
        "source": (
            """
            #include <metal_stdlib>
            using metal::uint;

            struct Data {
                uint values[1];
            };

            [[max_total_threads_per_threadgroup(1)]]
            kernel void main_(
                coherent device Data& coherent_buf [[user(fake0)]],
                device Data const& plain_buf [[user(fake0)]]) {
                uint value = plain_buf.values[0];
                coherent_buf.values[0] = value;
            }
        """
        ),
    },
    {
        "name": "wgpu_naga_int16_global_static_cast_literal_reparse",
        "repo_url": WGPU_REPO,
        "commit": WGPU_COMMIT,
        "source_path": "naga/tests/out/msl/wgsl-int16.metal",
        "roundtrip": True,
        "contains": [
            "constant uint16 constant_variable = (uint16)(20);",
            "constant int16 f16_to_i16_clamped = (int16)(32767);",
        ],
        "not_contains": ["(uint16)20", "(int16)32767"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            constant ushort constant_variable = static_cast<ushort>(20);
            constant short f16_to_i16_clamped = static_cast<short>(32767);
        """
        ),
    },
    {
        "name": "wgpu_naga_int64_integer_literal_suffix_reparse",
        "repo_url": WGPU_REPO,
        "commit": WGPU_COMMIT,
        "source_path": "naga/tests/out/msl/wgsl-int64.metal",
        "roundtrip": True,
        "contains": [
            "constant uint64 constant_variable = 20u;",
            "int64 val = 20;",
            "return val + 5;",
        ],
        "not_contains": ["20uL", "20L", "5L"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            constant ulong constant_variable = 20uL;

            long helper() {
                long val = 20L;
                return val + 5L;
            }
            """
        ),
    },
    {
        "name": "dawn_tint_subgroup_matrix_trailing_const_opaque_type",
        "repo_url": DAWN_REPO,
        "commit": DAWN_COMMIT,
        "source_path": (
            "test/tint/builtins/gen/var/subgroupMatrixStore/" "17ec3e.wgsl.expected.msl"
        ),
        "roundtrip": True,
        "contains": [
            "simdgroup_half8x8 arg_2 = "
            "make_filled_simdgroup_matrix_u3chalf_u2c8_u2c8_u3e(0.0);",
            "const simdgroup_half8x8 v_1 = arg_2;",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            void subgroupMatrixStore_17ec3e() {
                simdgroup_half8x8 arg_2 =
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
                simdgroup_half8x8 const v_1 = arg_2;
            }
        """
        ),
    },
    {
        "name": "dawn_tint_deep_braced_tint_array_initializer",
        "repo_url": DAWN_REPO,
        "commit": DAWN_COMMIT,
        "source_path": "test/tint/bug/chromium/1449474.wgsl.expected.msl",
        "roundtrip": True,
        "source": _dawn_deep_tint_array_initializer_source(),
    },
    {
        "name": "dawn_tint_warning_prologue_before_generated_msl",
        "repo_url": DAWN_REPO,
        "commit": DAWN_COMMIT,
        "source_path": "test/tint/bug/tint/2201.wgsl.expected.msl",
        "roundtrip": False,
        "source": (
            """
            <dawn>/test/tint/bug/tint/2201.wgsl:9:9 warning: code is unreachable
                    let _e16_ = vec2(false, false);
                    ^^^^^^^^^

            #include <metal_stdlib>
            using namespace metal;

            [[max_total_threads_per_threadgroup(1)]]
            kernel void v() {
              {
                uint2 tint_loop_idx = uint2(4294967295u);
                while(true) {
                  if (all((tint_loop_idx == uint2(0u)))) {
                    break;
                  }
                  if (true) {
                    break;
                  } else {
                    break;
                  }
                  /* unreachable */
                }
              }
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
        "name": "mlx_allocator_class_visibility_attribute",
        "repo_url": MLX_REPO,
        "commit": MLX_CURRENT_COMMIT,
        "source_path": "mlx/allocator.h",
        "roundtrip": False,
        "struct_names": ["Buffer"],
        "source": (
            """
            #define MLX_API __attribute__((visibility("default")))

            namespace mlx::core::allocator {
            class MLX_API Buffer {
             public:
                void* raw_ptr();
            };
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
        "name": "mlx_fp_quantized_block_scope_typedef_alias",
        "repo_url": MLX_REPO,
        "commit": MLX_CURRENT_COMMIT,
        "source_path": "mlx/backend/metal/kernels/fp_quantized.h",
        "roundtrip": False,
        "source": (
            """
            void load_quantized() {
                // Reduced from MLX qmv_quad_impl:
                // typedef float U;
                // thread U x_thread[values_per_thread];
                constexpr int values_per_thread = 4;
                typedef float U;
                thread U x_thread[values_per_thread];
                thread U result[8] = {0};
            }
            """
        ),
    },
    {
        "name": "mlx_fp_quantized_value_template_function_call",
        "repo_url": MLX_REPO,
        "commit": MLX_FP_QUANTIZED_VALUE_TEMPLATE_COMMIT,
        "source_path": "mlx/backend/metal/kernels/fp_quantized.h",
        "roundtrip": True,
        "contains": [
            "const int bytes_per_pack = get_bytes_per_pack_32();",
            "int16 get_bytes_per_pack_32()",
            "return 32 / 8;",
        ],
        "not_contains": ["get_bytes_per_pack<32>", "get_bytes_per_pack_u3c32_u3e"],
        "source": (
            """
            // Reduced from MLX fp_qmv_fast_impl:
            // constexpr int bytes_per_pack = get_bytes_per_pack<32>();
            template <int wsize = 8>
            inline constexpr short get_bytes_per_pack() {
                return wsize / 8;
            }

            void qmv_quad_impl() {
                constexpr int bytes_per_pack = get_bytes_per_pack<32>();
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
        "name": "mlx_steel_attention_unscoped_type_trait_variable_template",
        "repo_url": MLX_REPO,
        "commit": MLX_STEEL_ATTENTION_TYPE_TRAIT_COMMIT,
        "source_path": "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h",
        "roundtrip": True,
        "contains": [
            "const bool is_bool = is_same_v_u3cMaskType_u2cbool_u3e;",
        ],
        "not_contains": ["is_same_v<"],
        "source": (
            """
            using namespace mlx::steel;

            template <typename MaskType>
            void resolve_mask_type() {
                // Reduced from MLX Steel attention mask handling:
                // constexpr bool is_bool = is_same_v<MaskType, bool>;
                constexpr bool is_bool = is_same_v<MaskType, bool>;
                if (is_bool) {
                    return;
                }
            }
        """
        ),
    },
    {
        "name": "mlx_unary_ops_templated_call_operator_semicolon",
        "repo_url": MLX_REPO,
        "commit": MLX_UNARY_OPS_COMMIT,
        "source_path": "mlx/backend/metal/kernels/unary_ops.h",
        "roundtrip": True,
        "struct_names": ["Abs"],
        "contains": [
            "struct Abs",
            "uint8 value;",
        ],
        "source": (
            """
            struct Abs {
                template <typename T>
                T operator()(T x) {
                    return metal::abs(x);
                };

                uint8_t value;
            };
        """
        ),
    },
    {
        "name": "mlx_atomic_variable_template_expression",
        "repo_url": MLX_REPO,
        "commit": MLX_STEEL_DEPENDENT_TEMPLATE_COMMIT,
        "source_path": "mlx/backend/metal/kernels/atomic.h",
        "roundtrip": True,
        "contains": [
            "uint64 pack_offset = offset / packing_size_u3cT_u3e;",
            "uint64 elem_offset = offset % packing_size_u3cT_u3e;",
            "out_[0] = uint(pack_offset + elem_offset);",
        ],
        "not_contains": ["packing_size<T>"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template <typename T>
            constexpr constant uint packing_size = sizeof(uint) / sizeof(T);

            template <typename T>
            void update_atomic_offsets(size_t offset, device uint* out) {
                size_t pack_offset = offset / packing_size<T>;
                size_t elem_offset = offset % packing_size<T>;
                out[0] = uint(pack_offset + elem_offset);
            }
        """
        ),
    },
    {
        "name": "mlx_utils_half_float16_typedef_reparse",
        "repo_url": MLX_REPO,
        "commit": MLX_UTILS_TYPEDEF_COMMIT,
        "source_path": "mlx/backend/metal/kernels/utils.h",
        "roundtrip": True,
        "contains": ["typedef f16 float16_t;"],
        "not_contains": ["typedef float16 float16_t;"],
        "source": "typedef half float16_t;",
    },
    {
        "name": "mlx_bf16_scalar_typedef_reparse",
        "repo_url": MLX_REPO,
        "commit": MLX_CURRENT_COMMIT,
        "source_path": "mlx/backend/metal/kernels/bf16.h",
        "roundtrip": True,
        "contains": [
            "typedef bfloat16 bfloat16_t;",
            "return asuint(x);",
            "return as_type<bfloat16_t>(x);",
        ],
        "not_contains": [
            "typedef bfloat bfloat16_t;",
            "typedef f16 bfloat16_t;",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            typedef bfloat bfloat16_t;

            inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
                return as_type<uint16_t>(x);
            }

            inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
                return as_type<bfloat16_t>(x);
            }
        """
        ),
    },
    {
        "name": "mlx_scan_template_specialized_struct_reparse",
        "repo_url": MLX_REPO,
        "commit": MLX_SCAN_COMMIT,
        "source_path": "mlx/backend/metal/kernels/scan.h",
        "roundtrip": True,
        "struct_names": ["CumProd", "CumProd<bool>"],
        "contains": [
            "struct CumProd_u3cbool_u3e",
            "static constant bool init = true;",
        ],
        "not_contains": ["struct CumProd<bool>"],
        "source": (
            """
            template <typename U>
            struct CumProd {
                static constexpr constant U init = static_cast<U>(1.0f);
            };

            template <>
            struct CumProd<bool> {
                static constexpr constant bool init = true;

                template <typename T>
                bool operator()(bool a, T b) {
                    return a & static_cast<bool>(b);
                }
            };
        """
        ),
    },
    {
        "name": "mlx_steel_gemm_nested_aligned_read_vector",
        "repo_url": MLX_REPO,
        "commit": MLX_STEEL_GEMM_LOADER_COMMIT,
        "source_path": "mlx/backend/metal/kernels/steel/gemm/loader.h",
        "roundtrip": False,
        "struct_names": ["BlockLoader"],
        "source": (
            """
            #include "mlx/backend/metal/kernels/steel/defines.h"

            namespace mlx {
            namespace steel {

            template <
                typename T,
                short BROWS,
                short BCOLS,
                short dst_ld,
                short tgp_size,
                short alignment = 1,
                short n_reads = (BCOLS * BROWS) / (tgp_size),
                short TCOLS = BCOLS / n_reads>
            struct BlockLoader {
                const int src_ld;
                threadgroup T* dst;
                const device T* src;

                struct alignas(alignment * sizeof(T)) ReadVector {
                    uint8_t v[sizeof(T) * n_reads];
                };

                const short bj;
            };

            }
            }
        """
        ),
    },
    {
        "name": "mlx_reduction_parenthesized_template_argument",
        "repo_url": MLX_REPO,
        "commit": MLX_STEEL_DEPENDENT_TEMPLATE_COMMIT,
        "source_path": "mlx/backend/metal/kernels/reduction/reduce_col.h",
        "roundtrip": True,
        "contains": [
            "LoopedElemToLoc<NDIMS,IdxT,(NDIMS> 2)> loop = "
            "LoopedElemToLoc<NDIMS,IdxT,(NDIMS> 2)>(reduce_ndim);",
        ],
        "source": (
            """
            template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
            void col_reduce_looped(int reduce_ndim) {
                Op op;
                LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
            }
        """
        ),
    },
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Commit: 7168b60c0d3561d93aac7519d03d1bd95ee3e7a3
    # Path: aten/src/ATen/native/mps/kernels/GridSampler.h
    {
        "name": "pytorch_grid_sampler_leading_global_namespace_type",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_GRID_SAMPLER_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/GridSampler.h",
        "roundtrip": True,
        "struct_names": ["GridSamplerParams"],
        "contains": [
            "idx_t[N] output_sizes;",
            "idx_t[N] input_strides;",
        ],
        "not_contains": ["c10::metal::array"],
        "source": (
            """
            template <unsigned N = 5, typename idx_t = int32_t>
            struct GridSamplerParams {
                int32_t sampler_dims;
                ::c10::metal::array<idx_t, N> output_sizes;
                ::c10::metal::array<idx_t, N> input_strides;
            };
            """
        ),
    },
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Commit: 474a11a166e1313c37a9ad6f5ed0c887409d2cfc
    # Path: aten/src/ATen/native/mps/kernels/UpSample.h
    {
        "name": "pytorch_upsample_c10_array_expression_extent",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_CURRENT_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/UpSample.h",
        "roundtrip": True,
        "contains": [
            "uint64[N] input_strides;",
            "float[N-2] scales;",
        ],
        "not_contains": ["c10::metal::array"],
        "source": (
            """
            #include <c10/metal/common.h>

            template <unsigned N = 5>
            struct UpsampleParams {
                ::c10::metal::array<uint64_t, N> input_strides;
                ::c10::metal::array<float, N - 2> scales;
                bool align_corners;
            };
            """
        ),
    },
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Commit: 474a11a166e1313c37a9ad6f5ed0c887409d2cfc
    # Path: aten/src/ATen/native/mps/kernels/Embedding.h
    {
        "name": "pytorch_embedding_c10_array_scoped_extent",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_CURRENT_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/Embedding.h",
        "roundtrip": True,
        "contains": [
            "idx_type_t[c10_u3a_u3ametal_u3a_u3amax_ndim] outer_sizes;",
            "int64 padding_idx;",
        ],
        "not_contains": ["c10::metal::array", "::c10::metal::max_ndim"],
        "source": (
            """
            #include <c10/metal/common.h>

            template <typename idx_type_t = uint32_t>
            struct EmbeddingDenseBackwardParams {
                ::c10::metal::array<idx_type_t, ::c10::metal::max_ndim>
                    outer_sizes;
                int64_t padding_idx;
            };
            """
        ),
    },
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Commit: 474a11a166e1313c37a9ad6f5ed0c887409d2cfc
    # Path: c10/metal/atomic.h
    {
        "name": "pytorch_atomic_function_pointer_parameter",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_CURRENT_COMMIT,
        "source_path": "c10/metal/atomic.h",
        "roundtrip": True,
        "contains": [
            "void atomic_binary_op_helper(device AT* data",
            "T* op)",
            "val = op(value, value);",
        ],
        "not_contains": ["(*op)", "atomic<"],
        "source": (
            """
            #include <metal_atomic>

            template <typename AT, typename T>
            static inline void atomic_binary_op_helper(
                device ::metal::atomic<AT>* data,
                long offset,
                T value,
                T (*op)(T, T)) {
                T val;
                val = op(value, value);
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
        "name": "pytorch_pooling_templated_braced_constructor_return",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_POOLING_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/Pooling.metal",
        "roundtrip": True,
        "contains": [
            "IterBounds<int32_t> get_input_iter_bounds",
            "return IterBounds<int32_t>(start, end);",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template <typename T>
            struct IterBounds {
                T start;
                T end;
            };

            template <int32_t dim>
            IterBounds<int32_t> get_input_iter_bounds(
                constant int32_t* input_sizes,
                thread int32_t (&pooling_dim_indices)[3],
                constant int32_t* kernel_size,
                constant int32_t* stride,
                constant int32_t* padding,
                constant int32_t* dilation) {
                auto d = dilation[dim];
                auto start = stride[dim] * pooling_dim_indices[dim] - padding[dim];
                auto end = min(start + kernel_size[dim] * d, input_sizes[dim]);
                return IterBounds<int32_t>{start, end};
            }
        """
        ),
    },
    {
        "name": "pytorch_activation_if_constexpr_macro",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_ACTIVATION_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/ActivationKernel.metal",
        "roundtrip": False,
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template <typename T>
            static inline float gelu_dispatch_tanh(float x) {
                if IF_CONSTEXPR (::metal::is_same_v<T, float>) {
                    return ::metal::tanh(x);
                } else {
                    return tanh(x);
                }
            }
        """
        ),
    },
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Commit: fa5cb72912c44b22acd9c26c69f3e933794ac501
    # Path: aten/src/ATen/native/mps/kernels/ReduceOps.h
    {
        "name": "pytorch_c10_metal_constexpr_macro_constant",
        "repo_url": PYTORCH_REPO,
        "commit": PYTORCH_C10_METAL_CONSTEXPR_COMMIT,
        "source_path": "aten/src/ATen/native/mps/kernels/ReduceOps.h",
        "roundtrip": True,
        "contains": ["constant uint SUM_NCHAINS = 8;"],
        "source": (
            """
            #include <c10/metal/common.h>

            C10_METAL_CONSTEXPR uint32_t SUM_NCHAINS = 8;
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
            "static constant int threadsM = BM * SM;",
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
            "void kernel_memset(constant ggml_metal_kargs_memset args",
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
        "name": "llama_cpp_ggml_metal_keyword_member_name",
        "repo_url": LLAMA_CPP_REPO,
        "commit": LLAMA_CPP_METAL_DEVICE_COMMIT,
        "source_path": "ggml/src/ggml-metal/ggml-metal-device.h",
        "roundtrip": True,
        "contains": [
            "struct ggml_metal_buffer_id",
            "void* metal;",
            "uint64 offs;",
            "struct ggml_metal_pipeline_with_params",
            "int nsg;",
            "ggml_metal_device_id device_id;",
        ],
        "not_contains": ["enum ggml_metal_device_id device_id;"],
        "source": (
            """
            // Reduced from the ggml Metal device header. The host-side C API
            // uses `metal` as a struct member name for an id<MTLBuffer> handle
            // and declares prototypes with `struct tag` return types and
            // `enum tag` parameters.
            enum ggml_metal_device_id {
                GGML_METAL_DEVICE_ID_ANY = -1
            };

            struct ggml_metal_buffer_id {
                void * metal;
                size_t offs;
            };

            struct ggml_metal_device_props {
                enum ggml_metal_device_id device_id;
            };

            typedef struct ggml_metal_library * ggml_metal_library_t;

            struct ggml_metal_pipeline_with_params {
                int nsg;
            };

            struct ggml_metal_pipeline_with_params
            ggml_metal_library_get_pipeline(
                ggml_metal_library_t lib,
                const char * name);

            struct ggml_metal_pipeline_with_params
            ggml_metal_library_get_pipeline_base(
                ggml_metal_library_t lib,
                enum ggml_op op);
        """
        ),
    },
    {
        "name": "llama_cpp_ggml_metal_context_enum_return_prototypes",
        "repo_url": LLAMA_CPP_REPO,
        "commit": LLAMA_CPP_METAL_CONTEXT_COMMIT,
        "source_path": "ggml/src/ggml-metal/ggml-metal-context.h",
        "roundtrip": True,
        "source": (
            """
            typedef struct ggml_metal * ggml_metal_t;

            ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
            void ggml_metal_free(ggml_metal_t ctx);

            enum ggml_status ggml_metal_graph_compute(
                ggml_metal_t ctx,
                struct ggml_cgraph * gf);

            void ggml_metal_graph_optimize(
                ggml_metal_t ctx,
                struct ggml_cgraph * gf);
        """
        ),
    },
    {
        "name": "llama_cpp_ggml_extension_anonymous_union_member",
        "repo_url": LLAMA_CPP_REPO,
        "commit": LLAMA_CPP_GGML_COMMON_COMMIT,
        "source_path": "ggml/src/ggml-common.h included by ggml-metal.metal",
        "roundtrip": True,
        "contains": [
            "// Metal union iq1m_scale_t represented as struct-like layout; "
            "overlapping storage is not modeled",
            "struct iq1m_scale_t {",
            "struct block_q4_1 {",
            "uint8[32 / 2] qs;",
        ],
        "not_contains": ["__extension__", "GGML_EXTENSION"],
        "source": (
            """
            #define GGML_EXTENSION __extension__
            #define GGML_COMMON_AGGR_S
            #define GGML_COMMON_AGGR_U
            #define QK4_1 32

            typedef half ggml_half;
            typedef half2 ggml_half2;

            typedef struct {
                GGML_EXTENSION union {
                    struct {
                        ggml_half d;
                        ggml_half m;
                    } GGML_COMMON_AGGR_S;
                    ggml_half2 dm;
                } GGML_COMMON_AGGR_U;
                uint8_t qs[QK4_1 / 2];
            } block_q4_1;

            typedef union {
                ggml_half f16;
                uint16_t u16;
            } iq1m_scale_t;
        """
        ),
    },
    {
        "name": "llama_cpp_range_designated_array_initializer",
        "repo_url": LLAMA_CPP_REPO,
        "commit": LLAMA_CPP_COMMIT,
        "source_path": (
            "ggml/src/ggml-metal/ggml-metal.metal, " "kernel_mul_mv_ext_q4_f32_impl"
        ),
        "roundtrip": False,
        "contains": ["[0 ... r1ptg-1] = 0.0f"],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template<short r1ptg>
            void init_sums() {
                float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };
            }
        """
        ),
    },
    {
        "name": "llama_cpp_block_scope_static_assert_noop_struct",
        "repo_url": LLAMA_CPP_REPO,
        "commit": LLAMA_CPP_COMMIT,
        "source_path": (
            "ggml/src/ggml-metal/ggml-metal.metal:6140 after "
            "ggml-common.h static_assert fallback expansion"
        ),
        "roundtrip": False,
        "source": (
            """
            void flash_attn_ext_inner(threadgroup s_t * ss, short sgitg) {
                threadgroup s_t * ps = ss;
                ps += sgitg*(8*1);
                struct global_scope_noop_trick;
                constexpr short NC = (C/8)/NSG;
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
            "uint simd_lane_id @gl_SubgroupInvocationID",
            "uint simd_group_id @gl_SubgroupID",
            "vec4((*(f16vec4*)(x_row + simd_lane_id)))",
            "vec4((*(f16vec4*)(A + simd_lane_id)))",
            "xA_tile[tid.x] = acc_a4.x;",
        ],
        "not_contains": [
            "@thread_index_in_simdgroup",
            "@simdgroup_index_in_threadgroup",
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
                uint simd_lane_id [[thread_index_in_simdgroup]],
                uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
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
    {
        "name": "timdecode_simd_partition_conversion_operator_call",
        "repo_url": TIMDECODE_SIMD_PARTITION_GIST,
        "commit": TIMDECODE_SIMD_PARTITION_VERSION,
        "source_path": "simd_partition.metal",
        "roundtrip": True,
        "contains": [
            "uint unvisited = uint((uint64)simd_active_threads_mask());",
            "const uint v = uint((uint64)vote);",
        ],
        "not_contains": [
            "simd_active_threads_mask().operator",
            "vote.operator",
            "operator, unsigned",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template<typename T>
            static inline uint simd_partitionTD(T value) {
                uint unvisited =
                    uint(simd_active_threads_mask().operator unsigned long());
                uint result = value;
                while (unvisited != 0) {
                    const int activeLane = ctz(unvisited);
                    const T activeValue = simd_shuffle(value, activeLane);
                    const auto vote = simd_ballot(activeValue == value);
                    const uint v = uint(vote.operator unsigned long());
                    if (activeValue == value) {
                        result = v;
                    }
                    unvisited &= ~v;
                }
                return result;
            }
        """
        ),
    },
    {
        "name": "metalpetal_target_conditionals_import_default",
        "repo_url": METALPETAL_REPO,
        "commit": METALPETAL_COMMIT,
        "source_path": "Frameworks/MetalPetal/Shaders/Shaders.metal",
        "roundtrip": True,
        "contains": [
            "#include <TargetConditionals.h>",
            "vec4 passthrough(vec4 currentColor @gl_FragColor)",
        ],
        "not_contains": ["mti_haveColorArguments"],
        "source": (
            """
            #include <metal_stdlib>
            #include <TargetConditionals.h>

            #ifndef TARGET_OS_SIMULATOR
            #error TARGET_OS_SIMULATOR not defined. Check <TargetConditionals.h>
            #endif

            #if __HAVE_COLOR_ARGUMENTS__ && !TARGET_OS_SIMULATOR
            kernel void mti_haveColorArguments() {}
            #endif

            using namespace metal;

            namespace metalpetal {
                fragment float4 passthrough(float4 currentColor [[color(0)]]) {
                    return currentColor;
                }
            }
        """
        ),
    },
    {
        "name": "unixzii_gpu_particle_parenthesized_unary_position",
        "repo_url": UNIXZII_GPU_PARTICLE_GIST,
        "commit": UNIXZII_GPU_PARTICLE_VERSION,
        "source_path": "Shaders.metal",
        "roundtrip": True,
        "contains": [
            "v.position.y = (-(v.position.y + p.position.y - resolution.y / 2)) "
            "/ (resolution.y / 2);"
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            struct Particle {
                float2 position;
                float2 velocity;
                float life;
            };

            struct Vertex {
                float4 position [[position]];
                float2 uv;
                float opacity;
            };

            vertex Vertex particleVertex(
                const device Vertex *vertices [[buffer(0)]],
                const device float2 &resolution [[buffer(1)]],
                const device Particle *particles [[buffer(2)]],
                const device float2 &targetFrameSize [[buffer(3)]],
                unsigned int vid [[vertex_id]],
                unsigned int particleId [[instance_id]]) {
                int row = particleId / int(targetFrameSize.x);
                int col = particleId % int(targetFrameSize.x);
                Vertex v = vertices[vid];
                Particle p = particles[particleId];
                v.position.x =
                    ((v.position.x + p.position.x) - resolution.x / 2) /
                    (resolution.x / 2);
                v.position.y =
                    -((v.position.y + p.position.y) - resolution.y / 2) /
                    (resolution.y / 2);
                v.uv.x = float(col) / targetFrameSize.x;
                v.uv.y = float(row) / targetFrameSize.y;
                v.opacity = 1.0 - (p.life / 1000.0);
                return v;
            }
        """
        ),
    },
    {
        "name": "webkit_skia_interface_block_trailing_globals",
        "repo_url": WEBKIT_REPO,
        "commit": WEBKIT_COMMIT,
        "source_path": (
            "Source/ThirdParty/skia/tests/sksl/shared/InterfaceBlockInoutArray.metal"
        ),
        "roundtrip": True,
        "contains": [
            "InterfaceBlockIn[3] i;",
            "thread InterfaceBlockOut[3] o;",
        ],
        "source": (
            """
            #include <metal_stdlib>
            #include <simd/simd.h>
            using namespace metal;

            struct Inputs {
            };
            struct Outputs {
                half4 sk_FragColor [[color(0)]];
            };
            struct InterfaceBlockIn {
                int x;
            } i[3];
            thread struct InterfaceBlockOut {
                int x;
            } o[3];
            struct Globals {
                constant InterfaceBlockIn* i;
                constant InterfaceBlockOut* o;
            };
        """
        ),
    },
    {
        "name": "webkit_skia_restrict_local_identifier",
        "repo_url": WEBKIT_REPO,
        "commit": WEBKIT_COMMIT,
        "source_path": (
            "Source/ThirdParty/skia/tests/sksl/shared/ReservedInGLSLButAllowedInSkSL.metal"
        ),
        "roundtrip": True,
        "contains": [
            "f16vec4 restrict = _uniforms.colorGreen;",
            "return restrict * shared;",
        ],
        "source": (
            """
            #include <metal_stdlib>
            #include <simd/simd.h>
            using namespace metal;

            struct Uniforms {
                half4 colorGreen;
            };

            fragment half4 fragmentMain(
                constant Uniforms& _uniforms [[buffer(0)]]) {
                half4 restrict = _uniforms.colorGreen;
                half4 shared = _uniforms.colorGreen;
                return restrict * shared;
            }
        """
        ),
    },
    {
        "name": "tinygrad_thunder_struct_alignment_macro",
        "repo_url": TINYGRAD_REPO,
        "commit": TINYGRAD_THUNDER_COMMIT,
        "source_path": "extra/thunder/metal/include/types/shared/sv.metal",
        "roundtrip": True,
        "struct_names": ["sv", "alignment_dummy"],
        "contains": [
            "struct sv",
            "struct alignment_dummy",
            "dtype[_length] data;",
        ],
        "source": (
            """
            #include <metal_stdlib>
            using namespace metal;

            template<typename _T, size_t _length>
            struct mittens_DEFAULT_ALIGN sv {
                using dtype = _T;
                dtype data[_length];

                METAL_FUNC threadgroup dtype& operator[](size_t idx) threadgroup {
                    return data[idx];
                }
            };

            struct mittens_DEFAULT_ALIGN alignment_dummy {
                int dummy;
            };
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
