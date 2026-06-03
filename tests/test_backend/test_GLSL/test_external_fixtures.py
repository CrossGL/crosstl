import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


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


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_glsl_fixture_to_parseable_crossgl(fixture):
    crossgl = generate_crossgl(fixture.code, fixture.shader_type)

    assert "#version" in crossgl
    assert fixture.shader_type in crossgl
    assert parse_crossgl(crossgl) is not None
