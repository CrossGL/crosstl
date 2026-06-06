import textwrap

import pytest

import crosstl.translator
from crosstl.translator.ast import ShaderStage
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources


def reverse_plain_glsl(source: str):
    register_default_sources()
    spec = SOURCE_REGISTRY.get("glsl")
    ast = spec.parse(textwrap.dedent(source), file_path="/tmp/upstream-sample.glsl")
    crossgl = spec.reverse_codegen_factory().generate(ast)
    return ast, crossgl, crosstl.translator.parse(crossgl)


@pytest.mark.parametrize(
    (
        "source",
        "expected_shader_type",
        "expected_stage",
        "expected_crossgl",
    ),
    [
        pytest.param(
            """
            #version 310 es
            #extension GL_QCOM_tile_shading : enable

            layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

            layout(set = 0, binding = 2, tile_attachmentQCOM, rgba32i)
            uniform writeonly highp iimage2D color1;

            void main() {
                ivec2 offset = ivec2(gl_GlobalInvocationID.xy);
                imageStore(color1, offset, ivec4(1));
            }
            """,
            "compute",
            ShaderStage.COMPUTE,
            [
                "compute {",
                "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;",
                "iimage2D color1 @set(0) @binding(2) @rgba32i @writeonly;",
                "imageStore(color1, offset, ivec4(1));",
            ],
            id="glslang-spv-qcom-es-tile-shading-compute",
        ),
        pytest.param(
            """
            #version 150 core
            #extension GL_ARB_gpu_shader5 : require

            layout(points) in;
            layout(points, max_vertices = 1) out;
            layout(stream = 0) out float output1;

            void main() {
                output1 = 1.0;
                EmitStreamVertex(0);
                EndStreamPrimitive(0);
            }
            """,
            "geometry",
            ShaderStage.GEOMETRY,
            [
                "geometry {",
                "layout(points) in;",
                "layout(points, max_vertices = 1) out;",
                "out float output1 @stream(0);",
                "EmitStreamVertex(0);",
            ],
            id="glslang-end-stream-primitive-geometry",
        ),
        pytest.param(
            """
            #version 440

            layout(location = 0) in Primitive {
                vec2 texCoord;
            } IN[];

            layout(location = 0) out PrimitiveOut {
                vec2 texCoord;
            } OUT;

            layout(triangles, fractional_odd_spacing) in;
            layout(cw) in;

            void main() {
                OUT.texCoord = gl_TessCoord.xy;
                gl_Position = gl_in[gl_PatchVerticesIn].gl_Position;
            }
            """,
            "tessellation_evaluation",
            ShaderStage.TESSELLATION_EVALUATION,
            [
                "tessellation_evaluation {",
                "layout(triangles, fractional_odd_spacing) in;",
                "layout(cw) in;",
                "OUT.texCoord = gl_TessCoord.xy;",
            ],
            id="glslang-link-tessellation-evaluation",
        ),
        pytest.param(
            """
            #version 450 core

            layout(early_fragment_tests) in;

            void main() {
                gl_FragDepth = 0.5;
            }
            """,
            "fragment",
            ShaderStage.FRAGMENT,
            [
                "fragment {",
                "layout(early_fragment_tests) in;",
                "gl_FragDepth = 0.5;",
            ],
            id="glslang-fragment-layout-only",
        ),
        pytest.param(
            """
            #version 300 es
            precision highp float;

            uniform vec3 iResolution;
            uniform float iTime;

            void mainImage(out vec4 fragColor, in vec2 fragCoord)
            {
                vec2 uv = fragCoord / iResolution.xy;
                fragColor = vec4(uv, 0.5 + 0.5 * sin(iTime), 1.0);
            }
            """,
            "fragment",
            ShaderStage.FRAGMENT,
            [
                "fragment {",
                "void mainImage(out vec4 fragColor, in vec2 fragCoord)",
                "fragColor = vec4(uv, (0.5 + (0.5 * sin(iTime))), 1.0);",
                "vec4 main() @ gl_FragColor",
                "mainImage(shadertoyFragColor, gl_FragCoord.xy);",
                "return shadertoyFragColor;",
            ],
            id="shadertoy-main-image-fragment-entrypoint",
        ),
        pytest.param(
            """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
            #pragma shader_stage(fragment)

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = vec4(1.0);
            }
            """,
            "fragment",
            ShaderStage.FRAGMENT,
            [
                "fragment {",
                "#pragma shader_stage ( fragment )",
                "vec4 main() @location(0) @ outColor",
                "outColor = vec4(1.0);",
            ],
            id="glslc-pragma-shader-stage-fragment",
        ),
        pytest.param(
            """
            #version 460 core
            #extension GL_EXT_ray_tracing : require
            #pragma shader_stage(rgen)

            void main() {
            }
            """,
            "ray_generation",
            ShaderStage.RAY_GENERATION,
            [
                "ray_generation {",
                "#pragma shader_stage ( rgen )",
                "void main()",
            ],
            id="glslc-pragma-shader-stage-ray-generation",
        ),
        pytest.param(
            """
            #version 450 core
            #extension GL_EXT_mesh_shader : require
            #pragma shader_stage(mesh)

            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout(points, max_vertices = 1, max_primitives = 1) out;

            void main() {
                SetMeshOutputsEXT(1, 1);
            }
            """,
            "mesh",
            ShaderStage.MESH,
            [
                "mesh {",
                "#pragma shader_stage ( mesh )",
                "layout(points, max_vertices = 1, max_primitives = 1) out;",
                "SetMeshOutputCounts(1, 1);",
            ],
            id="glslc-pragma-shader-stage-mesh",
        ),
        pytest.param(
            """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
            layout(location = 0) in vec2 uv;
            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = vec4(uv, 0.0, 1.0);
            }
            """,
            "fragment",
            ShaderStage.FRAGMENT,
            [
                "fragment {",
                "#extension GL_ARB_separate_shader_objects : enable",
                "vec4 main(FragmentInput input) @location(0) @ outColor",
                "outColor = vec4(input.uv, 0.0, 1.0);",
            ],
            id="vulkan-fragment-output-location-infers-fragment",
        ),
    ],
)
def test_plain_glsl_registry_infers_stage_from_real_world_snippets(
    source,
    expected_shader_type,
    expected_stage,
    expected_crossgl,
):
    ast, crossgl, parsed = reverse_plain_glsl(source)

    assert ast.shader_type == expected_shader_type
    assert expected_stage in parsed.stages
    assert ShaderStage.VERTEX not in parsed.stages
    for expected in expected_crossgl:
        assert expected in crossgl


def test_plain_glsl_registry_keeps_vertex_output_varying_when_position_is_written():
    source = """
    #version 450
    layout(location = 0) in vec3 position;
    layout(location = 0) out vec2 uv;

    void main() {
        uv = position.xy;
        gl_Position = vec4(position, 1.0);
    }
    """

    ast, crossgl, parsed = reverse_plain_glsl(source)

    assert ast.shader_type == "vertex"
    assert ShaderStage.VERTEX in parsed.stages
    assert "vertex {" in crossgl
    assert "VertexOutput main(VertexInput input)" in crossgl
