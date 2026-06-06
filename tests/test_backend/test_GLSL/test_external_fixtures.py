import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.GLSL.OpenglAst import (
    AssignmentNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    InitializerListNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

VULKAN_SAMPLES_REPO = "https://github.com/KhronosGroup/Vulkan-Samples"
VULKAN_SAMPLES_COMMIT = "ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a"


@dataclass(frozen=True)
class ExternalFixture:
    name: str
    repo: str
    commit: str
    path: str
    shader_type: str
    code: str

    @property
    def source_url(self):
        return f"{self.repo}/blob/{self.commit}/{self.path}"


EXTERNAL_FIXTURES = [
    ExternalFixture(
        name="glslang-spv-spec-constant-layout-only-builtin",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.specConstant.vert",
        shader_type="vertex",
        code=textwrap.dedent("""
            #version 400

            layout(constant_id = 16) const int arraySize = 5;
            in vec4 ucol[arraySize];

            layout(constant_id = 24) gl_MaxImageUnits;

            out vec4 color;

            void foo(vec4 p[arraySize]);

            void main()
            {
                color = ucol[2];
                foo(ucol);
            }

            int builtin_spec_constant()
            {
                int result = gl_MaxImageUnits;
                return result;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-spv-push-constant-switch",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.pushConstant.vert",
        shader_type="vertex",
        code=textwrap.dedent("""
            #version 400

            layout(push_constant) uniform Material {
                int kind;
                float fa[3];
            } matInst;

            out vec4 color;

            void main()
            {
                switch (matInst.kind) {
                case 1:  color = vec4(0.2); break;
                case 2:  color = vec4(0.5); break;
                default: color = vec4(0.0); break;
                }
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/spv.int16.amd.frag.
    # Reduced from AMD int16 literal and specialization-constant coverage.
    ExternalFixture(
        name="glslang-spv-int16-amd-short-literal-suffixes",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.int16.amd.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 450
            #extension GL_AMD_gpu_shader_int16 : require

            layout(location = 0) in flat int16_t ii16;
            layout(location = 0) out vec4 color;

            layout(constant_id = 105) const int16_t si16 = -5S;
            layout(constant_id = 106) const uint16_t su16 = 4US;

            void main()
            {
                const int16_t i16c[3] =
                {
                    0x111S,
                    -2s,
                    0400s,
                };
                const uint16_t u16c[] =
                {
                    0xFFFFus,
                    65535US,
                    0177777us,
                };
                int16_t signed_value = min(i16c[1], int16_t(-1s));
                uint16_t unsigned_value = max(u16c[2], uint16_t(0us));
                color = vec4(float(signed_value + int16_t(unsigned_value)
                    + si16 + int16_t(su16) + ii16));
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/spv.controlFlowAttributes.frag.
    # Reduced from GL_EXT_control_flow_attributes coverage for attributed control flow.
    ExternalFixture(
        name="glslang-spv-control-flow-statement-attributes",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.controlFlowAttributes.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 450

            #extension GL_EXT_control_flow_attributes : enable

            bool cond;
            layout(location = 0) out vec4 color;

            void main()
            {
                [[loop]] for (;;) { break; }
                [[dont_unroll]] while(cond) {
                    cond = false;
                }
                [[dependency_infinite]] do {
                    cond = false;
                } while(true);
                [[dependency_length(1+3)]] for (int i = 0; i < 8; ++i) {
                    cond = false;
                }
                [[flatten]] if (cond) {
                    color = vec4(1.0);
                } else {
                    color = vec4(0.0);
                }
                [[branch]] if (cond) cond = false;
                [ [ dont_flatten , branch ] ] switch(3) { case 3: break; }
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/spv.memoryScopeSemantics.comp.
    # Reduced from GL_KHR_memory_scope_semantics storage qualifier coverage.
    ExternalFixture(
        name="glslang-spv-memory-scope-semantics-qualifiers",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.memoryScopeSemantics.comp",
        shader_type="compute",
        code=textwrap.dedent("""
            #version 450
            #extension GL_KHR_memory_scope_semantics : require
            #pragma use_vulkan_memory_model

            layout(binding = 0, r32ui) workgroupcoherent uniform uimage2D imageu;
            layout(binding = 5, r32i) nonprivate uniform iimage2D imagej[2];
            layout(binding = 2) buffer BufferU {
                workgroupcoherent uint x;
            } bufferu;
            struct A { uint x[2]; };
            layout(binding = 4) volatile buffer BufferJ {
                subgroupcoherent A a;
            } bufferj[2];
            layout(binding = 6) nonprivate uniform sampler2D samp[2];
            layout(binding = 7) nonprivate uniform BufferK {
                uint x;
            } bufferk;

            void main()
            {
                uint y = bufferu.x;
                y = bufferj[0].a.x[1];
                imageLoad(imagej[0], ivec2(0,0));
                texture(samp[0], vec2(0,0));
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/spv.1.6.nontemporalimage.frag.
    # Reduced from GL_EXT_nontemporal_keyword image qualifier coverage.
    ExternalFixture(
        name="glslang-spv-nontemporal-image-qualifiers",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.1.6.nontemporalimage.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 460

            #pragma use_vulkan_memory_model

            #extension GL_EXT_nontemporal_keyword: require

            layout(location=0) in vec2 in_uv;

            layout(binding=1, rgba8) uniform nontemporal image2D u_nontempimage;
            layout(binding=2, rgba8) uniform image2D u_image;
            layout(binding=3) uniform writeonly nontemporal image2D u_nontempnoformatimage;
            layout(binding=4) uniform writeonly image2D u_noformatimage;

            void function_accepting_nontemporal(nontemporal writeonly image2D image)
            {
            }

            void function_not_accepting_nontemporal(writeonly image2D image)
            {
            }

            void main()
            {
                const ivec2 uv = ivec2(in_uv.x, in_uv.y);
                imageStore(u_nontempimage, uv, imageLoad(u_nontempimage, uv));
                imageStore(u_image, uv, imageLoad(u_image, uv));

                function_accepting_nontemporal(u_nontempnoformatimage);
                function_accepting_nontemporal(u_noformatimage);
                function_not_accepting_nontemporal(u_noformatimage);
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/spv.longVectorLiteral234.comp.
    # Reduced from GL_EXT_long_vector local template-type declarations.
    ExternalFixture(
        name="glslang-long-vector-local-template-declarations",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.longVectorLiteral234.comp",
        shader_type="compute",
        code=textwrap.dedent("""
            #version 450 core
            #extension GL_EXT_long_vector : enable

            layout (local_size_x = 1) in;

            void main()
            {
                vector<float, 2> v2 = vector<float, 2>(1.0, 2.0);
                vector<float, 3> v3 = vector<float, 3>(1.0, 2.0, 3.0);
                vector<float, 4> v4 = vector<float, 4>(1.0, 2.0, 3.0, 4.0);

                v2 += v2;
                v3 = v3 * 2.0;
                v4 = v4 + vector<float, 4>(0.0);

                int n2 = v2.length();
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/spv.coopmat.comp.
    # Reduced from GL_NV_cooperative_matrix declarations with expression template
    # arguments and templated constructor calls.
    ExternalFixture(
        name="glslang-spv-coopmat-expression-template-declarations",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.coopmat.comp",
        shader_type="compute",
        code=textwrap.dedent("""
            #version 450 core
            #extension GL_KHR_memory_scope_semantics : enable
            #extension GL_NV_cooperative_matrix : enable
            #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

            layout (local_size_x = 64) in;

            const int Z = 16;
            fcoopmatNV<32, gl_ScopeSubgroup, Z, 8> mD =
                fcoopmatNV<32, gl_ScopeSubgroup, Z, 8>(0.0);

            void main()
            {
                fcoopmatNV<32, gl_ScopeSubgroup, 16, (2>1?8:4)> m =
                    fcoopmatNV<32, gl_ScopeSubgroup, 16, (2>1?8:4)>(0.0);
                fcoopmatNV<16, gl_ScopeSubgroup, 8, 8> p1;
                p1 = fcoopmatNV<16, gl_ScopeSubgroup, 8, 8>(0.0);
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-420-tese-anonymous-struct-initializer",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/420.tese",
        shader_type="tessellation_evaluation",
        code=textwrap.dedent("""
            #version 420 core

            struct {
                float a;
                int b;
            } e = { 1.2, 2, };

            void main()
            {
                if (e.b > 0)
                    ;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-spv-perprimitive-nv-interface-block",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/spv.perprimitiveNV.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 460

            #extension GL_NV_mesh_shader: require

            layout(location=0)
            in B {
                perprimitiveNV float f;
            };

            layout(location=4)
            in C {
                flat centroid float h;
            };

            layout(location=8)
            out float g;

            void main()
            {
                g = f + h;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-150-frag-patch-contextual-identifier",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/150.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 150 core

            out vec4 color;

            float patch = 3.1;

            void main()
            {
                color = vec4(patch);
            }
        """).strip(),
    ),
    # Upstream source: https://github.com/KhronosGroup/glslang
    # Commit: 98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515
    # Path: Test/330.frag
    # Reduced from lines that mark "precise" as okay before it became a keyword.
    ExternalFixture(
        name="glslang-330-frag-precise-contextual-identifier",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/330.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 330 compatibility

            int precise;
            struct SKeyMem { int precise; } KeyMem;

            void main()
            {
                KeyMem.precise;
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/330.frag.
    # Reduced from GL_ARB_enhanced_layouts interface-block member location coverage.
    ExternalFixture(
        name="glslang-330-frag-interface-block-member-location",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/330.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 330 compatibility
            #extension GL_ARB_separate_shader_objects : enable
            #extension GL_ARB_enhanced_layouts : enable

            layout(location = 28) in InBlock {
                float f1;
                layout(location = 25) float f2;
                vec4 f3;
            } ininst;

            layout(location = 0) out vec4 outVar;

            void main()
            {
                outVar = vec4(ininst.f2) + ininst.f3;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-120-vert-invariant-builtin-list",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/120.vert",
        shader_type="vertex",
        code=textwrap.dedent("""
            #version 120

            attribute vec4 attv4;
            invariant varying vec2 centTexCoord;
            invariant gl_Position, gl_PointSize;

            void main()
            {
                centTexCoord = attv4.xy;
                gl_Position = attv4;
                gl_PointSize = 1.0;
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/120.frag.
    # Reduced from GLSL 1.20 coverage where future precision/interpolation/type
    # names are still accepted as ordinary identifiers.
    ExternalFixture(
        name="glslang-120-frag-contextual-keyword-identifiers",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/120.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 120

            float lowp;
            float mediump;
            float highp;
            float precision;

            void main()
            {
                float flat;
                float smooth;
                float noperspective;
                float uvec2;
                float uvec3;
                float uvec4;

                uvec2 = 2.0;
                gl_FragColor = vec4(lowp + mediump + highp + precision
                    + flat + smooth + noperspective + uvec2 + uvec3 + uvec4);
            }

            float imageBuffer;
            float uimage2DRect;
        """).strip(),
    ),
    ExternalFixture(
        name="learnopengl-deferred-shading-fragment",
        repo="https://github.com/JoeyDeVries/LearnOpenGL",
        commit="a545a703f95893258d16dbe32f5ccbb6400fd213",
        path="src/5.advanced_lighting/8.1.deferred_shading/8.1.deferred_shading.fs",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 330 core
            out vec4 FragColor;

            in vec2 TexCoords;

            uniform sampler2D gPosition;
            uniform sampler2D gNormal;
            uniform sampler2D gAlbedoSpec;

            struct Light {
                vec3 Position;
                vec3 Color;

                float Linear;
                float Quadratic;
            };
            const int NR_LIGHTS = 32;
            uniform Light lights[NR_LIGHTS];
            uniform vec3 viewPos;

            void main()
            {
                vec3 FragPos = texture(gPosition, TexCoords).rgb;
                vec3 Normal = texture(gNormal, TexCoords).rgb;
                vec3 Diffuse = texture(gAlbedoSpec, TexCoords).rgb;
                float Specular = texture(gAlbedoSpec, TexCoords).a;

                vec3 lighting = Diffuse * 0.1;
                vec3 viewDir = normalize(viewPos - FragPos);
                for(int i = 0; i < NR_LIGHTS; ++i)
                {
                    vec3 lightDir = normalize(lights[i].Position - FragPos);
                    vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Diffuse * lights[i].Color;
                    vec3 halfwayDir = normalize(lightDir + viewDir);
                    float spec = pow(max(dot(Normal, halfwayDir), 0.0), 16.0);
                    vec3 specular = lights[i].Color * spec * Specular;
                    float distance = length(lights[i].Position - FragPos);
                    float attenuation = 1.0 / (1.0 + lights[i].Linear * distance
                        + lights[i].Quadratic * distance * distance);
                    diffuse *= attenuation;
                    specular *= attenuation;
                    lighting += diffuse + specular;
                }
                FragColor = vec4(lighting, 1.0);
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-texture-frag-legacy-projected-samplers",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/texture.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 130

            uniform sampler2D texSampler2D;
            uniform sampler3D texSampler3D;

            varying vec2 coords2D;

            void main()
            {
                float bias = 2.0;
                vec3 coords3D = vec3(coords2D, 1.0);
                vec4 coords4D = vec4(coords2D, 1.0, 2.0);
                vec4 color = vec4(0.0);

                color += texture2DProj(texSampler2D, coords3D);
                color += texture3DProj(texSampler3D, coords4D, bias);

                gl_FragColor = color;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-texture-frag-legacy-varying-input",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/texture.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 130

            uniform sampler2D texSampler2D;

            varying vec2 coords2D;

            void main()
            {
                gl_FragColor = texture2D(texSampler2D, coords2D);
            }
        """).strip(),
    ),
    # Upstream source: KhronosGroup/glslang Test/400.frag.
    # Reduced from GLSL subroutine coverage with a comma-separated association list.
    ExternalFixture(
        name="glslang-400-frag-subroutine-association-list",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/400.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 400 core

            subroutine(subT1, subT2);
            subroutine float subT1() { return 1.0; }
            subroutine float subT2() { return 2.0; }

            out vec4 outp;

            void main()
            {
                outp = vec4(subT1() + subT2());
            }
        """).strip(),
    ),
    ExternalFixture(
        name="glslang-fragcoord-origin-layout-flags",
        repo="https://github.com/KhronosGroup/glslang",
        commit="98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515",
        path="Test/gl_FragCoord.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 150 core
            #extension GL_ARB_explicit_attrib_location : enable

            layout (origin_upper_left,pixel_center_integer) in vec4 gl_FragCoord;
            layout (location = 0) out vec4 myColor;

            void main() {
                myColor = vec4(gl_FragCoord.xy, 0.0, 1.0);
            }
        """).strip(),
    ),
    ExternalFixture(
        name="saschawillems-compute-particles-ssbo",
        repo="https://github.com/SaschaWillems/Vulkan",
        commit="180be3f9f9a0e86fff2a7de283a54063999f2b69",
        path="shaders/glsl/computeparticles/particle.comp",
        shader_type="compute",
        code=textwrap.dedent("""
            #version 450

            struct Particle
            {
                vec2 pos;
                vec2 vel;
                vec4 gradientPos;
            };

            layout(std140, binding = 0) readonly buffer ParticleSSBOIn {
               Particle particlesIn[ ];
            };

            layout(std140, binding = 1) buffer ParticleSSBOOut {
               Particle particlesOut[ ];
            };

            layout (local_size_x = 256) in;

            layout (binding = 2) uniform UBO
            {
                float deltaT;
                int particleCount;
            } ubo;

            vec2 attraction(vec2 pos, vec2 attractPos)
            {
                vec2 delta = attractPos - pos;
                const float damp = 0.5;
                float dDampedDot = dot(delta, delta) + damp;
                return delta * dDampedDot;
            }

            void main()
            {
                uint index = gl_GlobalInvocationID.x;
                if (index >= ubo.particleCount)
                    return;

                vec2 vVel = particlesIn[index].vel.xy;
                vec2 vPos = particlesIn[index].pos.xy;
                vPos += vVel * ubo.deltaT;
                particlesOut[index].pos.xy = vPos;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="godot-gles3-particles-precision-ubo-hash",
        repo="https://github.com/godotengine/godot",
        commit="3badc0ee23caf1e5269fe90f8ee17e09d4d2b682",
        path="drivers/gles3/shaders/particles.glsl",
        shader_type="vertex",
        code=textwrap.dedent("""
            #version 300 es
            precision highp float;
            #define MAX_ATTRACTORS 32

            struct Attractor {
                mat4 transform;
                vec4 extents;

                uint type;
                float strength;
            };

            layout(std140) uniform FrameData {
                uint random_seed;
                uint attractor_count;

                Attractor attractors[MAX_ATTRACTORS];
            };

            layout(location = 0) in highp vec4 color;
            out highp vec4 out_color;

            uint hash(uint x) {
                x = ((x >> uint(16)) ^ x) * uint(0x45d9f3b);
                x = (x >> uint(16)) ^ x;
                return x;
            }

            void main() {
                mat4 xform = mat4(1.0);
                xform[0] = color;
                out_color = vec4(float(hash(random_seed)));
            }
        """).strip(),
    ),
    ExternalFixture(
        name="godot-gles3-cube-to-dp-bracketed-stage-marker",
        repo="https://github.com/godotengine/godot",
        commit="bbd3f43b57db5008539e87bd86ef9e3cc7a44a23",
        path="drivers/gles3/shaders/cube_to_dp.glsl",
        shader_type="vertex",
        code=textwrap.dedent("""
            [vertex]

            precision mediump float;
            precision mediump int;

            layout(location = 0) in highp vec4 vertex_attrib;
            layout(location = 4) in vec2 uv_in;

            out vec2 uv_interp;

            void main() {
                uv_interp = uv_in;
                gl_Position = vertex_attrib;
            }
        """).strip(),
    ),
    ExternalFixture(
        name="godot-betsy-bc1-hash-section-preamble",
        repo="https://github.com/godotengine/godot",
        commit="72cc0fc9a75bf041e84b9d37e7e31e17cb114a9e",
        path="modules/betsy/bc1.glsl",
        shader_type="compute",
        code=textwrap.dedent("""
            #[versions]

            standard = "";
            dithered = "#define BC1_DITHER";

            #[compute]
            #version 450

            #VERSION_DEFINES

            layout(binding = 0) uniform sampler2D srcTex;
            layout(binding = 1, rg32ui) uniform restrict writeonly uimage2D dstTexture;

            layout(local_size_x = 8,
                    local_size_y = 8,
                    local_size_z = 1) in;

            void main() {
            }
        """).strip(),
    ),
    ExternalFixture(
        name="filament-surface-instancing-highp-object-uniforms",
        repo="https://github.com/google/filament",
        commit="48881c840bca50da515f0df82b61c9a5b996b19a",
        path="shaders/src/surface_instancing.glsl",
        shader_type="vertex",
        code=textwrap.dedent("""
            #version 300 es
            precision highp float;

            highp mat4 object_uniforms_worldFromModelMatrix;
            highp mat3 object_uniforms_worldFromModelNormalMatrix;
            highp int object_uniforms_flagsChannels;

            struct ObjectData {
                highp mat4 worldFromModelMatrix;
                highp int flagsChannels;
            };

            layout(std140) uniform ObjectUniforms {
                ObjectData data[4];
            } objectUniforms;

            highp mat4 getWorldFromModelMatrix() {
                return object_uniforms_worldFromModelMatrix;
            }

            void initObjectUniforms() {
                highp int i = 0;
                if ((objectUniforms.data[0].flagsChannels & 1) != 0) {
                    i = 1;
                }
                object_uniforms_worldFromModelMatrix =
                    objectUniforms.data[i].worldFromModelMatrix;
            }

            void main() {
                initObjectUniforms();
            }
        """).strip(),
    ),
    ExternalFixture(
        name="vulkan-samples-base-frag-set-bindings-push-constants",
        repo=VULKAN_SAMPLES_REPO,
        commit=VULKAN_SAMPLES_COMMIT,
        path="shaders/base.frag",
        shader_type="fragment",
        code=textwrap.dedent("""
            #version 320 es
            precision highp float;

            layout(set = 0, binding = 0) uniform sampler2D base_color_texture;

            layout(location = 0) in vec4 in_pos;
            layout(location = 1) in vec2 in_uv;
            layout(location = 2) in vec3 in_normal;

            layout(location = 0) out vec4 o_color;

            layout(set = 0, binding = 1) uniform GlobalUniform
            {
                mat4 model;
                mat4 view_proj;
                vec3 camera_position;
            }
            global_uniform;

            layout(push_constant, std430) uniform PBRMaterialUniform
            {
                vec4  base_color_factor;
                float metallic_factor;
                float roughness_factor;
            }
            pbr_material_uniform;

            layout(set = 0, binding = 4) uniform LightsInfo
            {
                Light directional_lights[48];
                Light point_lights[48];
                Light spot_lights[48];
            }
            lights_info;

            layout(constant_id = 0) const uint DIRECTIONAL_LIGHT_COUNT = 0U;
            layout(constant_id = 1) const uint POINT_LIGHT_COUNT       = 0U;
            layout(constant_id = 2) const uint SPOT_LIGHT_COUNT        = 0U;

            void main(void)
            {
                vec3 normal = normalize(in_normal);

                vec3 light_contribution = vec3(0.0);

                for (uint i = 0U; i < DIRECTIONAL_LIGHT_COUNT; ++i)
                {
                    light_contribution += apply_directional_light(lights_info.directional_lights[i], normal);
                }

                vec4 base_color = vec4(1.0, 0.0, 0.0, 1.0);

                base_color = texture(base_color_texture, in_uv);

                vec3 ambient_color = vec3(0.2) * base_color.xyz;

                o_color = vec4(ambient_color + light_contribution * base_color.xyz, base_color.w);
            }
        """).strip(),
    ),
    ExternalFixture(
        name="vulkan-samples-buffer-device-address-reference-compute",
        repo=VULKAN_SAMPLES_REPO,
        commit=VULKAN_SAMPLES_COMMIT,
        path="shaders/buffer_device_address/glsl/update_vbo.comp",
        shader_type="compute",
        code=textwrap.dedent("""
            #version 450
            #extension GL_EXT_buffer_reference : require

            layout(local_size_x = 8, local_size_y = 8) in;

            layout(buffer_reference) buffer Position;

            layout(std430, buffer_reference, buffer_reference_align = 8) writeonly buffer Position
            {
                vec2 positions[];
            };

            layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer PositionReferences
            {
                Position buffers[];
            };

            layout(push_constant) uniform Registers
            {
                PositionReferences references;
                float fract_time;
            } registers;

            void main()
            {
                uint local_index = gl_GlobalInvocationID.x;
                uint slice = gl_WorkGroupID.z;
                restrict Position positions = registers.references.buffers[slice];
                positions.positions[local_index] = vec2(fract(registers.fract_time));
            }
        """).strip(),
    ),
    ExternalFixture(
        name="ekmett-vr-scan-multi-declarator-for-init",
        repo="https://github.com/ekmett/vr",
        commit="e2b9ff4fbcd3f6bf885fbc60d4425a63c5e5f2a3",
        path="shaders/scan.glsl",
        shader_type="compute",
        code=textwrap.dedent("""
            #version 450

            uint warp_scan_inclusive(uint d, uint N) {
              for (uint i = 1, i_max = min(N >> 1, 5u); i < i_max; i <<= 1) {
                bool valid = false;
                uint t = shuffleUpNV(d, i, 32, valid);
                if (valid) d += t;
              }
              return d;
            }

            void main() {
              uint total = warp_scan_inclusive(1u, 8u);
            }
        """).strip(),
    ),
]


def parse_glsl(code, shader_type):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def generate_crossgl(code, shader_type):
    ast = parse_glsl(code, shader_type)
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(ast)


def parse_crossgl(code):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_parse_external_glsl_fixture(fixture):
    ast = parse_glsl(fixture.code, fixture.shader_type)

    assert ast is not None
    assert ast.shader_type == fixture.shader_type
    assert fixture.repo
    assert fixture.commit
    assert fixture.source_url.startswith(fixture.repo)


def test_parse_glslang_layout_only_builtin_spec_constant_fixture():
    fixture = EXTERNAL_FIXTURES[0]

    ast = parse_glsl(fixture.code, fixture.shader_type)
    builtin = next(
        var for var in ast.global_variables if var.name == "gl_MaxImageUnits"
    )

    assert builtin.vtype == ""
    assert builtin.layout == {"constant_id": "24"}


def test_parse_glslang_perprimitive_nv_interface_block_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-perprimitive-nv-interface-block"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    block = next(struct for struct in ast.structs if struct.name == "B")
    member = next(var for var in ast.io_variables if var.name == "f")
    output = next(var for var in ast.io_variables if var.name == "g")

    assert block.interface_qualifiers == ["in"]
    assert member.vtype == "float"
    assert member.qualifiers == ["perprimitiveNV", "in"]
    assert member.layout == {"location": "0"}
    assert output.qualifiers == ["out"]
    assert output.layout == {"location": "8"}


def test_parse_glslang_anonymous_struct_declarator_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-420-tese-anonymous-struct-initializer"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    anon_struct = next(
        struct for struct in ast.structs if struct.name == "AnonymousStruct0"
    )
    value = next(var for var in ast.global_variables if var.name == "e")

    assert [member.name for member in anon_struct.members] == ["a", "b"]
    assert value.vtype == "AnonymousStruct0"
    assert isinstance(value.value, InitializerListNode)


def test_parse_glslang_int16_short_literal_suffix_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-int16-amd-short-literal-suffixes"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    si16 = next(var for var in ast.constant if var.name == "si16")
    su16 = next(var for var in ast.constant if var.name == "su16")
    main = next(function for function in ast.functions if function.name == "main")
    i16c = next(var for var in main.body if getattr(var, "name", None) == "i16c")
    u16c = next(var for var in main.body if getattr(var, "name", None) == "u16c")

    assert isinstance(si16.value, UnaryOpNode)
    assert si16.value.operand.value == "5S"
    assert si16.layout == {"constant_id": "105"}
    assert su16.value.value == "4US"
    assert su16.layout == {"constant_id": "106"}
    assert [element.value for element in i16c.value.elements[0::2]] == [
        "0x111S",
        "0400s",
    ]
    assert isinstance(i16c.value.elements[1], UnaryOpNode)
    assert i16c.value.elements[1].operand.value == "2s"
    assert [element.value for element in u16c.value.elements] == [
        "0xFFFFus",
        "65535US",
        "0177777us",
    ]


def test_parse_glslang_control_flow_statement_attributes_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-control-flow-statement-attributes"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    main = next(function for function in ast.functions if function.name == "main")

    assert [type(statement) for statement in main.body] == [
        ForNode,
        WhileNode,
        DoWhileNode,
        ForNode,
        IfNode,
        IfNode,
        SwitchNode,
    ]


def test_parse_glslang_memory_scope_semantics_qualifier_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-memory-scope-semantics-qualifiers"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    imageu = next(var for var in ast.uniforms if var.name == "imageu")
    imagej = next(var for var in ast.uniforms if var.name == "imagej")
    samp = next(var for var in ast.uniforms if var.name == "samp")
    bufferk = next(var for var in ast.uniforms if var.name == "bufferk")
    buffer_u = next(struct for struct in ast.structs if struct.name == "BufferU")
    buffer_j = next(struct for struct in ast.structs if struct.name == "BufferJ")

    assert imageu.qualifiers == ["workgroupcoherent", "uniform"]
    assert imageu.layout == {"binding": "0", "r32ui": None}
    assert imagej.qualifiers == ["nonprivate", "uniform"]
    assert imagej.layout == {"binding": "5", "r32i": None}
    assert samp.qualifiers == ["nonprivate", "uniform"]
    assert bufferk.qualifiers == ["nonprivate", "uniform"]
    assert buffer_u.members[0].qualifiers == ["workgroupcoherent"]
    assert buffer_j.members[0].qualifiers == ["subgroupcoherent"]


def test_parse_glslang_nontemporal_image_qualifier_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-nontemporal-image-qualifiers"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    image = next(var for var in ast.uniforms if var.name == "u_nontempimage")
    writeonly_image = next(
        var for var in ast.uniforms if var.name == "u_nontempnoformatimage"
    )
    function = next(
        item for item in ast.functions if item.name == "function_accepting_nontemporal"
    )

    assert image.vtype == "image2D"
    assert image.qualifiers == ["uniform", "nontemporal"]
    assert image.layout == {"binding": "1", "rgba8": None}
    assert writeonly_image.qualifiers == ["uniform", "writeonly", "nontemporal"]
    assert function.params[0].qualifiers == ["nontemporal", "writeonly"]


def test_parse_glslang_long_vector_template_declarations_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-long-vector-local-template-declarations"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    main = next(function for function in ast.functions if function.name == "main")
    declarations = [
        statement for statement in main.body if isinstance(statement, VariableNode)
    ]

    assert [(decl.vtype, decl.name) for decl in declarations[:3]] == [
        ("vector<float, 2>", "v2"),
        ("vector<float, 3>", "v3"),
        ("vector<float, 4>", "v4"),
    ]
    assert all(isinstance(decl.value, FunctionCallNode) for decl in declarations[:3])
    assert [decl.value.name.name for decl in declarations[:3]] == [
        "vector<float, 2>",
        "vector<float, 3>",
        "vector<float, 4>",
    ]


def test_codegen_glslang_long_vector_template_declarations_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-long-vector-local-template-declarations"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "vector<float, 2> v2 = vector<float, 2>(1.0, 2.0);" in crossgl
    assert "vector<float, 3> v3 = vector<float, 3>(1.0, 2.0, 3.0);" in crossgl
    assert "vector<float, 4> v4 = vector<float, 4>(1.0, 2.0, 3.0, 4.0);" in crossgl
    assert "v4 = (v4 + vector<float, 4>(0.0));" in crossgl
    parse_crossgl(crossgl)


def test_parse_glslang_coopmat_expression_template_declarations_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-coopmat-expression-template-declarations"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    global_matrix = next(var for var in ast.global_variables if var.name == "mD")
    main = next(function for function in ast.functions if function.name == "main")
    declarations = [
        statement for statement in main.body if isinstance(statement, VariableNode)
    ]

    assert global_matrix.vtype == "fcoopmatNV<32, gl_ScopeSubgroup, Z, 8>"
    assert global_matrix.value.name.name == ("fcoopmatNV<32, gl_ScopeSubgroup, Z, 8>")
    assert [(decl.vtype, decl.name) for decl in declarations[:2]] == [
        ("fcoopmatNV<32, gl_ScopeSubgroup, 16, ( 2> 1 ? 8 : 4 )>", "m"),
        ("fcoopmatNV<16, gl_ScopeSubgroup, 8, 8>", "p1"),
    ]
    assert declarations[0].value.name.name == (
        "fcoopmatNV<32, gl_ScopeSubgroup, 16, ( 2> 1 ? 8 : 4 )>"
    )


def test_codegen_glslang_coopmat_expression_template_declarations_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-coopmat-expression-template-declarations"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "fcoopmatNV<32, gl_ScopeSubgroup, Z, 8> mD =" in crossgl
    assert "fcoopmatNV<32, gl_ScopeSubgroup, 16, ( 2> 1 ? 8 : 4 )> m =" in crossgl
    assert "fcoopmatNV<32, gl_ScopeSubgroup, 16, ( 2> 1 ? 8 : 4 )>(0.0)" in crossgl


def test_parse_glslang_patch_contextual_identifier_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-150-frag-patch-contextual-identifier"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    patch = next(var for var in ast.global_variables if var.name == "patch")

    assert patch.vtype == "float"
    assert patch.value.value == "3.1"


def test_parse_glslang_precise_contextual_identifier_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-330-frag-precise-contextual-identifier"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    precise_global = next(var for var in ast.global_variables if var.name == "precise")
    key_mem = next(struct for struct in ast.structs if struct.name == "SKeyMem")
    main = next(function for function in ast.functions if function.name == "main")

    assert precise_global.vtype == "int"
    assert key_mem.members[0].name == "precise"
    assert main.body[0].member == "precise"


def test_parse_glslang_interface_block_member_location_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-330-frag-interface-block-member-location"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    block = next(struct for struct in ast.structs if struct.name == "InBlock")
    instance = next(var for var in ast.io_variables if var.name == "ininst")
    member = next(var for var in block.members if var.name == "f2")

    assert block.interface_layout == {"location": "28"}
    assert block.interface_qualifiers == ["in"]
    assert instance.layout == {"location": "28"}
    assert member.layout == {"location": "25"}


def test_parse_glslang_invariant_builtin_list_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-120-vert-invariant-builtin-list"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    attribute = next(var for var in ast.io_variables if var.name == "attv4")
    redeclarations = [
        var
        for var in ast.global_variables
        if var.name in {"gl_Position", "gl_PointSize"}
    ]

    assert attribute.vtype == "vec4"
    assert attribute.qualifiers == ["attribute"]
    assert attribute.io_type == "IN"
    assert not any(var.name == "attv4" for var in ast.global_variables)
    assert [var.name for var in redeclarations] == ["gl_Position", "gl_PointSize"]
    assert [var.vtype for var in redeclarations] == ["", ""]
    assert [var.qualifiers for var in redeclarations] == [
        ["invariant"],
        ["invariant"],
    ]


def test_parse_glslang_contextual_keyword_identifier_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-120-frag-contextual-keyword-identifiers"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    global_names = [var.name for var in ast.global_variables]
    main = next(function for function in ast.functions if function.name == "main")
    local_names = [stmt.name for stmt in main.body if hasattr(stmt, "name")]
    assignment = next(stmt for stmt in main.body if isinstance(stmt, AssignmentNode))

    assert global_names == [
        "lowp",
        "mediump",
        "highp",
        "precision",
        "imageBuffer",
        "uimage2DRect",
    ]
    assert local_names == ["flat", "smooth", "noperspective", "uvec2", "uvec3", "uvec4"]
    assert assignment.left.name == "uvec2"


def test_codegen_glslang_contextual_keyword_identifier_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-120-frag-contextual-keyword-identifiers"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    for declaration in (
        "float lowp;",
        "float mediump;",
        "float highp;",
        "float precision;",
        "float flat;",
        "float smooth;",
        "float noperspective;",
        "float uvec2;",
        "float uvec3;",
        "float uvec4;",
        "float imageBuffer;",
        "float uimage2DRect;",
    ):
        assert declaration in crossgl
    assert "uvec2 = 2.0;" in crossgl
    parse_crossgl(crossgl)


def test_codegen_glslang_fragcoord_origin_layout_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-fragcoord-origin-layout-flags"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    frag_coord = next(var for var in ast.io_variables if var.name == "gl_FragCoord")

    assert frag_coord.layout == {
        "origin_upper_left": None,
        "pixel_center_integer": None,
    }

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "vec4 gl_FragCoord @origin_upper_left @pixel_center_integer;" in crossgl
    parse_crossgl(crossgl)


def test_parse_godot_particles_precision_ubo_hash_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "godot-gles3-particles-precision-ubo-hash"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    frame_data = next(struct for struct in ast.structs if struct.name == "FrameData")
    color = next(var for var in ast.io_variables if var.name == "color")
    out_color = next(var for var in ast.io_variables if var.name == "out_color")

    assert frame_data.interface_block is True
    assert frame_data.interface_layout == {"std140": None}
    assert frame_data.members[2].name == "attractors"
    assert frame_data.members[2].array_size.value == "32"
    assert color.qualifiers == ["in", "highp"]
    assert out_color.qualifiers == ["out", "highp"]


def test_parse_godot_bracketed_stage_marker_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "godot-gles3-cube-to-dp-bracketed-stage-marker"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    vertex = next(var for var in ast.io_variables if var.name == "vertex_attrib")
    uv = next(var for var in ast.io_variables if var.name == "uv_in")

    assert vertex.layout == {"location": "0"}
    assert vertex.qualifiers == ["in", "highp"]
    assert uv.layout == {"location": "4"}
    assert [function.name for function in ast.functions] == ["main"]


def test_parse_godot_hash_section_preamble_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "godot-betsy-bc1-hash-section-preamble"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    src_tex = next(uniform for uniform in ast.uniforms if uniform.name == "srcTex")
    dst_texture = next(
        uniform for uniform in ast.uniforms if uniform.name == "dstTexture"
    )

    assert ast.preprocessor == ["#version 450", "#VERSION_DEFINES"]
    assert ast.layouts == [
        {
            "layout": {"local_size_x": "8", "local_size_y": "8", "local_size_z": "1"},
            "qualifiers": ["in"],
        }
    ]
    assert src_tex.layout == {"binding": "0"}
    assert dst_texture.layout == {"binding": "1", "rg32ui": None}
    assert dst_texture.qualifiers == ["uniform", "restrict", "writeonly"]

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)
    assert "#[versions]" not in crossgl
    assert "dithered" not in crossgl
    assert "#VERSION_DEFINES" in crossgl
    assert "compute {" in crossgl
    parse_crossgl(crossgl)


def test_parse_filament_instancing_highp_object_uniforms_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "filament-surface-instancing-highp-object-uniforms"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    world_matrix = next(
        var
        for var in ast.global_variables
        if var.name == "object_uniforms_worldFromModelMatrix"
    )
    object_uniforms = next(var for var in ast.uniforms if var.name == "objectUniforms")
    object_data = next(struct for struct in ast.structs if struct.name == "ObjectData")

    assert world_matrix.qualifiers == ["highp"]
    assert object_uniforms.vtype == "ObjectUniforms"
    assert object_uniforms.layout == {"std140": None}
    assert object_data.members[0].qualifiers == ["highp"]


def test_parse_vulkan_samples_base_frag_push_constants_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "vulkan-samples-base-frag-set-bindings-push-constants"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    base_texture = next(var for var in ast.uniforms if var.name == "base_color_texture")
    material_block = next(
        var for var in ast.uniforms if var.name == "pbr_material_uniform"
    )
    lights_info = next(var for var in ast.uniforms if var.name == "lights_info")
    spot_count = next(var for var in ast.constant if var.name == "SPOT_LIGHT_COUNT")

    assert base_texture.layout == {"set": "0", "binding": "0"}
    assert material_block.layout == {"push_constant": None, "std430": None}
    assert lights_info.layout == {"set": "0", "binding": "4"}
    assert spot_count.layout == {"constant_id": "2"}
    assert spot_count.value.value == "0U"


def test_parse_vulkan_samples_buffer_device_address_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "vulkan-samples-buffer-device-address-reference-compute"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    forward_decl = next(var for var in ast.global_variables if var.name == "Position")
    position = next(struct for struct in ast.structs if struct.name == "Position")
    references = next(
        struct for struct in ast.structs if struct.name == "PositionReferences"
    )
    registers = next(var for var in ast.uniforms if var.name == "registers")
    local_alias = next(
        statement
        for statement in ast.functions[0].body
        if getattr(statement, "name", None) == "positions"
    )

    assert forward_decl.vtype == ""
    assert forward_decl.qualifiers == ["buffer"]
    assert forward_decl.layout == {"buffer_reference": None}
    assert position.interface_layout == {
        "std430": None,
        "buffer_reference": None,
        "buffer_reference_align": "8",
    }
    assert position.interface_qualifiers == ["writeonly", "buffer"]
    assert position.members[0].name == "positions"
    assert position.members[0].is_array is True
    assert references.interface_qualifiers == ["readonly", "buffer"]
    assert references.members[0].vtype == "Position"
    assert registers.vtype == "Registers"
    assert registers.layout == {"push_constant": None}
    assert local_alias.vtype == "Position"
    assert local_alias.qualifiers == ["restrict"]


def test_parse_ekmett_vr_scan_multi_declarator_for_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "ekmett-vr-scan-multi-declarator-for-init"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    scan = next(
        function for function in ast.functions if function.name == "warp_scan_inclusive"
    )
    loop = next(statement for statement in scan.body if isinstance(statement, ForNode))

    assert [declaration.name for declaration in loop.init] == ["i", "i_max"]
    assert [declaration.vtype for declaration in loop.init] == ["uint", "uint"]


def test_parse_glslang_subroutine_association_list_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-400-frag-subroutine-association-list"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)

    assert [function.name for function in ast.functions] == [
        "subT1",
        "subT2",
        "main",
    ]


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_glsl_fixture_to_parseable_crossgl(fixture):
    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    if "#version" in fixture.code:
        assert "#version" in crossgl
    assert fixture.shader_type in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_vulkan_samples_buffer_device_address_fixture_snippets():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "vulkan-samples-buffer-device-address-reference-compute"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "struct Position {" in crossgl
    assert "vec2 positions[];" in crossgl
    assert "struct PositionReferences {" in crossgl
    assert "Position buffers[];" in crossgl
    assert "cbuffer Registers @push_constant {" in crossgl
    assert "PositionReferences references;" in crossgl
    assert "restrict Position positions = references.buffers[slice];" in crossgl
    assert "\n     Position;\n" not in crossgl


def test_codegen_ekmett_vr_scan_multi_declarator_for_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "ekmett-vr-scan-multi-declarator-for-init"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert (
        "for (uint i = 1, i_max = min((N >> 1), 5u); (i < i_max); i <<= 1)" in crossgl
    )
    assert "for (uint i = 1; (i < i_max);" not in crossgl


def test_codegen_glslang_legacy_projected_texture_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-texture-frag-legacy-projected-samplers"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "textureProj(texSampler2D, coords3D)" in crossgl
    assert "textureProj(texSampler3D, coords4D, bias)" in crossgl
    assert "texture2DProj(" not in crossgl
    assert "texture3DProj(" not in crossgl


def test_codegen_glslang_anonymous_struct_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-420-tese-anonymous-struct-initializer"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "struct AnonymousStruct0 {" in crossgl
    assert "AnonymousStruct0 e = { 1.2, 2 };" in crossgl


def test_codegen_glslang_int16_short_literal_suffix_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-int16-amd-short-literal-suffixes"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "0x111S" not in crossgl
    assert "0x111" in crossgl
    assert "(-2s)" not in crossgl
    assert "(-2)" in crossgl
    assert "0400s" not in crossgl
    assert "0400" in crossgl
    assert "0xFFFFus" not in crossgl
    assert "0xFFFFu" in crossgl
    assert "65535US" not in crossgl
    assert "65535u" in crossgl
    assert "0177777us" not in crossgl
    assert "0177777u" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_control_flow_statement_attributes_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-control-flow-statement-attributes"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "[[" not in crossgl
    assert "for (; ; )" in crossgl
    assert "while (cond)" in crossgl
    assert "} while (true);" in crossgl
    assert "for (int i = 0; (i < 8); (++i))" in crossgl
    assert "if (cond)" in crossgl
    assert "switch (3)" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_memory_scope_semantics_qualifier_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-memory-scope-semantics-qualifiers"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "uimage2D imageu @binding(0) @r32ui @workgroupcoherent;" in crossgl
    assert "iimage2D imagej[2] @binding(5) @r32i @nonprivate;" in crossgl
    assert "uint x @workgroupcoherent;" in crossgl
    assert "A a @subgroupcoherent;" in crossgl
    assert "sampler2D samp[2] @binding(6) @nonprivate;" in crossgl
    assert "BufferK bufferk @nonprivate;" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_nontemporal_image_qualifier_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-nontemporal-image-qualifiers"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "image2D u_nontempimage @binding(1) @rgba8 @nontemporal;" in crossgl
    assert (
        "image2D u_nontempnoformatimage @binding(3) @writeonly @nontemporal;" in crossgl
    )
    assert (
        "void function_accepting_nontemporal(" "image2D image @writeonly @nontemporal)"
    ) in crossgl
    assert "nontemporal image2D" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_patch_contextual_identifier_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-150-frag-patch-contextual-identifier"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "float patch = 3.1;" in crossgl
    assert "vec4(patch)" in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_precise_contextual_identifier_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-330-frag-precise-contextual-identifier"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert crossgl.count("int precise;") == 2
    assert "KeyMem.precise;" in crossgl
    assert "@precise" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_interface_block_member_location_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-330-frag-interface-block-member-location"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert (
        "@glsl_interface_block(in) @location(28) @glsl_interface_instance(ininst)"
        in crossgl
    )
    assert "float f2 @location(25);" in crossgl
    assert "float f2;" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_invariant_builtin_list_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-120-vert-invariant-builtin-list"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "struct VertexInput" in crossgl
    assert "vec4 attv4;" in crossgl
    assert "VertexOutput main(VertexInput input)" in crossgl
    assert "vec4 attv4;\n\n    vertex" not in crossgl
    assert "output.centTexCoord = input.attv4.xy;" in crossgl
    assert "output.gl_Position = input.attv4;" in crossgl
    assert "vec4 gl_Position @invariant @ gl_Position;" in crossgl
    assert "float gl_PointSize @invariant @ gl_PointSize;" in crossgl
    assert " gl_Position @invariant;" not in crossgl
    assert " gl_PointSize @invariant;" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_legacy_varying_fragment_input_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-texture-frag-legacy-varying-input"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    coords = next(var for var in ast.io_variables if var.name == "coords2D")
    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert coords.qualifiers == ["varying"]
    assert coords.io_type == "IN"
    assert not any(var.name == "coords2D" for var in ast.global_variables)
    assert "struct FragmentInput" in crossgl
    assert "vec2 coords2D;" in crossgl
    assert "texture(texSampler2D, input.coords2D)" in crossgl
    assert "texture(texSampler2D, coords2D)" not in crossgl
    assert parse_crossgl(crossgl) is not None


def test_codegen_glslang_perprimitive_nv_fixture_canonical_qualifier():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-perprimitive-nv-interface-block"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "perprimitive float f;" in crossgl
    assert "perprimitiveNV float f;" not in crossgl
