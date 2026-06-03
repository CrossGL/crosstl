import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.GLSL.OpenglAst import ForNode
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


def test_codegen_glslang_perprimitive_nv_fixture_canonical_qualifier():
    fixture = next(
        item
        for item in EXTERNAL_FIXTURES
        if item.name == "glslang-spv-perprimitive-nv-interface-block"
    )

    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "perprimitive float f;" in crossgl
    assert "perprimitiveNV float f;" not in crossgl
