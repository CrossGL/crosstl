import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.GLSL.OpenglAst import (
    DoWhileNode,
    ForNode,
    IfNode,
    InitializerListNode,
    SwitchNode,
    UnaryOpNode,
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


def test_parse_glslang_invariant_builtin_list_fixture():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-120-vert-invariant-builtin-list"
    )

    ast = parse_glsl(fixture.code, fixture.shader_type)
    redeclarations = [
        var
        for var in ast.global_variables
        if var.name in {"gl_Position", "gl_PointSize"}
    ]

    assert [var.name for var in redeclarations] == ["gl_Position", "gl_PointSize"]
    assert [var.vtype for var in redeclarations] == ["", ""]
    assert [var.qualifiers for var in redeclarations] == [
        ["invariant"],
        ["invariant"],
    ]


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


def test_codegen_glslang_invariant_builtin_list_fixture_snippet():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-120-vert-invariant-builtin-list"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

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
