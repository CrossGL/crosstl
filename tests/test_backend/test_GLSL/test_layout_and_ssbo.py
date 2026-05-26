import pytest

import crosstl.translator
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen


def parse_glsl(code: str, shader_type: str):
    tokens = GLSLLexer(code).tokenize()
    return GLSLParser(tokens, shader_type).parse()


def generate_crossgl(code: str, shader_type: str):
    return GLSLToCrossGLConverter(shader_type=shader_type).generate(
        parse_glsl(code, shader_type)
    )


def test_parse_ssbo_layout():
    code = """
    #version 450 core
    layout(std430, binding = 2) buffer DataBlock {
        vec4 values[];
    } dataBlock;
    void main() {
        vec4 v = dataBlock.values[0];
    }
    """
    ast = parse_glsl(code, "vertex")
    assert ast is not None


def test_parse_layout_locations_and_components():
    code = """
    #version 450 core
    layout(location = 1, component = 2) in vec4 color;
    layout(location = 0, index = 1) out vec4 fragColor;
    void main() {
        fragColor = color;
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_precision_qualifiers_on_variables():
    code = """
    #version 300 es
    precision mediump float;
    mediump vec2 uv;
    void main() {
        uv = vec2(0.0);
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None


def test_parse_early_fragment_tests_layout():
    code = """
    #version 450 core
    layout(early_fragment_tests) in;
    layout(location = 0) out vec4 fragColor;
    void main() {
        fragColor = vec4(1.0);
    }
    """
    ast = parse_glsl(code, "fragment")
    assert ast is not None

    crossgl = generate_crossgl(code, "fragment")
    assert "layout(early_fragment_tests) in;" in crossgl
    assert "// layout(" not in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))
    assert "layout(early_fragment_tests) in;" in glsl


def test_parse_ray_shader_record_layout_roundtrips_without_binding():
    code = """
    #version 460 core
    #extension GL_EXT_ray_tracing : require

    struct RayPayload {
        vec4 color;
    };

    layout(binding = 0) uniform accelerationStructureEXT topLevelAS;
    layout(location = 0) rayPayloadEXT RayPayload rayPayload;
    layout(shaderRecordEXT, std430) buffer ShaderRecordData {
        uint materialIndex;
    } shaderRecord;

    void main() {
        rayPayload.color = vec4(float(shaderRecord.materialIndex));
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsNoneEXT,
            0xff,
            0,
            1,
            0,
            vec3(0.0),
            0.001,
            vec3(0.0, 0.0, 1.0),
            1000.0,
            0
        );
    }
    """

    crossgl = generate_crossgl(code, "ray_generation")

    assert "accelerationStructureEXT topLevelAS @binding(0);" in crossgl
    assert "RayPayload rayPayload @location(0) @rayPayloadEXT;" in crossgl
    assert (
        "ShaderRecordData shaderRecord @glsl_buffer_block(shaderRecordEXT, std430);"
        in crossgl
    )
    assert "RayGenerationInput" not in crossgl
    assert "RayGenerationOutput" not in crossgl
    assert "void main()" in crossgl
    assert "TraceRay(" in crossgl
    assert "traceRayEXT(" not in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert glsl.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in glsl
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in glsl
    assert "layout(location = 0) rayPayloadEXT RayPayload rayPayload;" in glsl
    assert "layout(shaderRecordEXT, std430) buffer ShaderRecordData" in glsl
    assert "layout(shaderRecordEXT, std430, binding" not in glsl
    assert "rayPayload.color = vec4(float(shaderRecord.materialIndex));" in glsl
    assert "traceRayEXT(" in glsl


def test_parse_ray_shader_record_layout_rejects_binding():
    code = """
    #version 460 core
    #extension GL_EXT_ray_tracing : require

    layout(shaderRecordEXT, binding = 3) buffer ShaderRecordData {
        uint materialIndex;
    } shaderRecord;

    void main() { }
    """

    with pytest.raises(
        ValueError,
        match="shaderRecordEXT buffer blocks cannot declare binding layout qualifiers",
    ):
        generate_crossgl(code, "ray_generation")


@pytest.mark.parametrize(
    (
        "shader_type",
        "source",
        "expected_crossgl",
        "expected_glsl",
    ),
    [
        (
            "ray_closest_hit",
            """
            #version 460 core
            #extension GL_EXT_ray_tracing : require

            struct RayPayload {
                vec4 color;
            };

            layout(location = 0) rayPayloadInEXT RayPayload closestPayload;
            hitAttributeEXT vec2 hitAttributes;

            void main() {
                closestPayload.color = vec4(hitAttributes, 0.0, 1.0);
            }
            """,
            [
                "RayPayload closestPayload @location(0) @rayPayloadInEXT;",
                "vec2 hitAttributes @hitAttributeEXT;",
                "void main()",
            ],
            [
                "layout(location = 0) rayPayloadInEXT RayPayload closestPayload;",
                "hitAttributeEXT vec2 hitAttributes;",
                "closestPayload.color = vec4(hitAttributes, 0.0, 1.0);",
            ],
        ),
        (
            "ray_any_hit",
            """
            #version 460 core
            #extension GL_EXT_ray_tracing : require

            struct RayPayload {
                vec4 color;
            };

            layout(location = 0) rayPayloadInEXT RayPayload anyPayload;
            hitAttributeEXT vec2 anyHitAttributes;

            void main() {
                anyPayload.color = vec4(anyHitAttributes, 1.0, 1.0);
                ignoreIntersectionEXT;
            }
            """,
            [
                "RayPayload anyPayload @location(0) @rayPayloadInEXT;",
                "vec2 anyHitAttributes @hitAttributeEXT;",
                "IgnoreHit();",
            ],
            [
                "layout(location = 0) rayPayloadInEXT RayPayload anyPayload;",
                "hitAttributeEXT vec2 anyHitAttributes;",
                "ignoreIntersectionEXT;",
            ],
        ),
        (
            "ray_callable",
            """
            #version 460 core
            #extension GL_EXT_ray_tracing : require

            struct CallableData {
                vec4 value;
            };

            layout(location = 1) callableDataInEXT CallableData callableInput;

            void main() {
                callableInput.value = vec4(1.0);
            }
            """,
            [
                "CallableData callableInput @location(1) @callableDataInEXT;",
                "void main()",
            ],
            [
                "layout(location = 1) callableDataInEXT CallableData callableInput;",
                "callableInput.value = vec4(1.0);",
            ],
        ),
    ],
)
def test_parse_ray_stage_storage_qualifiers_roundtrip(
    shader_type,
    source,
    expected_crossgl,
    expected_glsl,
):
    crossgl = generate_crossgl(source, shader_type)

    assert "RayGenerationInput" not in crossgl
    assert "RayGenerationOutput" not in crossgl
    for expected in expected_crossgl:
        assert expected in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert glsl.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_tracing : require" in glsl
    for expected in expected_glsl:
        assert expected in glsl


def test_parse_ray_query_type_and_functions_roundtrip():
    code = """
    #version 460 core
    #extension GL_EXT_ray_query : require
    #extension GL_EXT_ray_tracing_position_fetch : require

    layout(binding = 0) uniform accelerationStructureEXT topLevelAS;

    void main() {
        rayQueryEXT rq;
        rayQueryInitializeEXT(
            rq,
            topLevelAS,
            gl_RayFlagsNoneEXT,
            255u,
            vec3(0.0),
            0.001,
            vec3(0.0, 0.0, 1.0),
            100.0
        );
        bool active = rayQueryProceedEXT(rq);
        uint hitType = rayQueryGetIntersectionTypeEXT(rq, true);
        uint candidatePrimitive = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
        uint committedInstance = rayQueryGetIntersectionInstanceIdEXT(rq, true);
        uint candidateGeometry = rayQueryGetIntersectionGeometryIndexEXT(rq, false);
        vec3 committedOrigin = rayQueryGetIntersectionObjectRayOriginEXT(rq, true);
        vec3 candidateDirection = rayQueryGetIntersectionObjectRayDirectionEXT(rq, false);
        float committedT = rayQueryGetIntersectionTEXT(rq, true);
        vec3 worldOrigin = rayQueryGetWorldRayOriginEXT(rq);
        vec3 worldDirection = rayQueryGetWorldRayDirectionEXT(rq);
        uint rayFlags = rayQueryGetRayFlagsEXT(rq);
        float rayTMin = rayQueryGetRayTMinEXT(rq);
        uint customIndex = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        uint sbtOffset =
            rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, false);
        vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(rq, false);
        bool frontFace = rayQueryGetIntersectionFrontFaceEXT(rq, true);
        bool aabbOpaque = rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq);
        vec3 trianglePositions[3];
        rayQueryGetIntersectionTriangleVertexPositionsEXT(
            rq, false, trianglePositions
        );
        rayQueryGetIntersectionTriangleVertexPositionsEXT(
            rq, true, trianglePositions
        );
        rayQueryGenerateIntersectionEXT(rq, 1.0);
        rayQueryConfirmIntersectionEXT(rq);
        rayQueryTerminateEXT(rq);
    }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "accelerationStructureEXT topLevelAS @binding(0);" in crossgl
    assert "rayQueryEXT rq;" in crossgl
    assert "rq.Initialize(" in crossgl
    assert "bool active = rq.Proceed();" in crossgl
    assert "uint hitType = rq.CommittedType();" in crossgl
    assert "uint candidatePrimitive = rq.CandidatePrimitiveIndex();" in crossgl
    assert "uint committedInstance = rq.CommittedInstanceID();" in crossgl
    assert "uint candidateGeometry = rq.CandidateGeometryIndex();" in crossgl
    assert "vec3 committedOrigin = rq.CommittedObjectRayOrigin();" in crossgl
    assert "vec3 candidateDirection = rq.CandidateObjectRayDirection();" in crossgl
    assert "float committedT = rq.CommittedRayT();" in crossgl
    assert "vec3 worldOrigin = rq.WorldRayOrigin();" in crossgl
    assert "vec3 worldDirection = rq.WorldRayDirection();" in crossgl
    assert "uint rayFlags = rq.RayFlags();" in crossgl
    assert "float rayTMin = rq.RayTMin();" in crossgl
    assert "uint customIndex = rq.CommittedInstanceCustomIndex();" in crossgl
    assert (
        "uint sbtOffset = rq.CandidateInstanceShaderBindingTableRecordOffset();"
        in crossgl
    )
    assert "vec2 barycentrics = rq.CandidateTriangleBarycentrics();" in crossgl
    assert "bool frontFace = rq.CommittedTriangleFrontFace();" in crossgl
    assert "bool aabbOpaque = rq.CandidateAABBOpaque();" in crossgl
    assert "vec3 trianglePositions[3];" in crossgl
    assert "rq.CandidateTriangleVertexPositions(trianglePositions);" in crossgl
    assert "rq.CommittedTriangleVertexPositions(trianglePositions);" in crossgl
    assert "rq.GenerateIntersection(1.0);" in crossgl
    assert "rq.ConfirmIntersection();" in crossgl
    assert "rq.Abort();" in crossgl
    assert "rayQueryInitializeEXT(" not in crossgl
    assert "rayQueryProceedEXT(rq)" not in crossgl
    assert "rayQueryGetIntersectionTypeEXT(rq, true)" not in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert glsl.lstrip().startswith("#version 460 core")
    assert "#extension GL_EXT_ray_query : require" in glsl
    assert "#extension GL_EXT_ray_tracing_position_fetch : require" in glsl
    assert "layout(binding = 0) uniform accelerationStructureEXT topLevelAS;" in glsl
    assert "rayQueryEXT rq;" in glsl
    assert "rayQueryInitializeEXT(" in glsl
    assert "bool active_ = rayQueryProceedEXT(rq);" in glsl
    assert "bool active =" not in glsl
    assert "uint hitType = rayQueryGetIntersectionTypeEXT(rq, true);" in glsl
    assert (
        "uint candidatePrimitive = "
        "rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);" in glsl
    )
    assert (
        "uint committedInstance = rayQueryGetIntersectionInstanceIdEXT(rq, true);"
        in glsl
    )
    assert (
        "uint candidateGeometry = rayQueryGetIntersectionGeometryIndexEXT(rq, false);"
        in glsl
    )
    assert (
        "vec3 committedOrigin = "
        "rayQueryGetIntersectionObjectRayOriginEXT(rq, true);" in glsl
    )
    assert (
        "vec3 candidateDirection = "
        "rayQueryGetIntersectionObjectRayDirectionEXT(rq, false);" in glsl
    )
    assert "float committedT = rayQueryGetIntersectionTEXT(rq, true);" in glsl
    assert "vec3 worldOrigin = rayQueryGetWorldRayOriginEXT(rq);" in glsl
    assert "vec3 worldDirection = rayQueryGetWorldRayDirectionEXT(rq);" in glsl
    assert "uint rayFlags = rayQueryGetRayFlagsEXT(rq);" in glsl
    assert "float rayTMin = rayQueryGetRayTMinEXT(rq);" in glsl
    assert (
        "uint customIndex = "
        "rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);" in glsl
    )
    assert (
        "uint sbtOffset = "
        "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, false);"
        in glsl
    )
    assert (
        "vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(rq, false);" in glsl
    )
    assert "bool frontFace = rayQueryGetIntersectionFrontFaceEXT(rq, true);" in glsl
    assert (
        "bool aabbOpaque = rayQueryGetIntersectionCandidateAABBOpaqueEXT(rq);" in glsl
    )
    assert "vec3 trianglePositions[3];" in glsl
    assert (
        "rayQueryGetIntersectionTriangleVertexPositionsEXT("
        "rq, false, trianglePositions);" in glsl
    )
    assert (
        "rayQueryGetIntersectionTriangleVertexPositionsEXT("
        "rq, true, trianglePositions);" in glsl
    )
    assert "rayQueryGenerateIntersectionEXT(rq, 1.0);" in glsl
    assert "rayQueryConfirmIntersectionEXT(rq);" in glsl
    assert "rayQueryTerminateEXT(rq);" in glsl
    assert ".Proceed(" not in glsl
    assert ".CommittedType(" not in glsl


def test_parse_compute_layout_roundtrips_as_stage_layout():
    code = """
    #version 450 core
    layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;

    void main() { }
    """

    crossgl = generate_crossgl(code, "compute")

    assert "// layout(" not in crossgl
    assert "compute {" in crossgl
    assert "layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;" in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert "layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;" in glsl


@pytest.mark.parametrize(
    ("shader_type", "glsl_call", "crossgl_call", "regenerated_call"),
    [
        (
            "mesh",
            "SetMeshOutputsEXT(3, 1)",
            "SetMeshOutputCounts(3, 1)",
            "SetMeshOutputsEXT(3, 1)",
        ),
        (
            "task",
            "EmitMeshTasksEXT(1, 1, 1)",
            "DispatchMesh(1, 1, 1)",
            "EmitMeshTasksEXT(1, 1, 1)",
        ),
    ],
)
def test_parse_mesh_intrinsics_roundtrip_through_canonical_crossgl(
    shader_type,
    glsl_call,
    crossgl_call,
    regenerated_call,
):
    code = f"""
    #version 450 core
    #extension GL_EXT_mesh_shader : require
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    void main() {{
        {glsl_call};
    }}
    """

    crossgl = generate_crossgl(code, shader_type)

    assert crossgl_call in crossgl
    assert glsl_call not in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert "#extension GL_EXT_mesh_shader : require" in glsl
    assert regenerated_call in glsl


def test_parse_mesh_stage_interface_qualifiers_roundtrip():
    code = """
    #version 450 core
    #extension GL_EXT_mesh_shader : require

    struct TaskPayload {
        uint meshlet;
    };

    taskPayloadSharedEXT TaskPayload payload;
    perprimitiveEXT out vec3 primitiveNormal[32];
    out vec4 vertexColor[64];
    layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    layout(triangles, max_vertices = 64, max_primitives = 32) out;

    void main() {
        SetMeshOutputsEXT(64, 32);
        primitiveNormal[0] = vec3(0.0, 0.0, 1.0);
        vertexColor[0] = vec4(float(payload.meshlet));
    }
    """

    crossgl = generate_crossgl(code, "mesh")

    assert "TaskPayload payload @taskPayloadSharedEXT;" in crossgl
    assert "perprimitive out vec3 primitiveNormal[32];" in crossgl
    assert "out vec4 vertexColor[64];" in crossgl
    assert "SetMeshOutputCounts(64, 32);" in crossgl
    assert "taskPayloadSharedEXT TaskPayload payload;" not in crossgl
    assert "perprimitiveEXT out vec3 primitiveNormal[32];" not in crossgl
    assert (
        "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;" in crossgl
    )
    assert "layout(triangles, max_vertices = 64, max_primitives = 32) out;" in crossgl
    assert "// layout(" not in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert "#extension GL_EXT_mesh_shader : require" in glsl
    assert "layout(triangles, max_vertices = 64, max_primitives = 32) out;" in glsl
    assert "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;" in glsl
    assert "taskPayloadSharedEXT TaskPayload payload;" in glsl
    assert "perprimitiveEXT out vec3 primitiveNormal[32];" in glsl
    assert "out vec4 vertexColor[64];" in glsl
    assert "SetMeshOutputsEXT(64, 32);" in glsl


@pytest.mark.parametrize(
    (
        "layout_topology",
        "expected_layout",
        "index_assignment",
        "crossgl_call",
        "regenerated_call",
    ),
    [
        (
            "points",
            "layout(points, max_vertices = 1, max_primitives = 1) out;",
            "gl_PrimitivePointIndicesEXT[0] = 0u;",
            "SetMeshOutputCounts(1, 1);",
            "SetMeshOutputsEXT(1, 1);",
        ),
        (
            "lines",
            "layout(lines, max_vertices = 2, max_primitives = 1) out;",
            "gl_PrimitiveLineIndicesEXT[0] = uvec2(0u, 1u);",
            "SetMeshOutputCounts(2, 1);",
            "SetMeshOutputsEXT(2, 1);",
        ),
        (
            "triangles",
            "layout(triangles, max_vertices = 3, max_primitives = 1) out;",
            "gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0u, 1u, 2u);",
            "SetMeshOutputCounts(3, 1);",
            "SetMeshOutputsEXT(3, 1);",
        ),
    ],
)
def test_parse_mesh_topology_builtin_index_arrays_roundtrip(
    layout_topology,
    expected_layout,
    index_assignment,
    crossgl_call,
    regenerated_call,
):
    max_vertices = {"points": 1, "lines": 2, "triangles": 3}[layout_topology]
    code = f"""
    #version 450 core
    #extension GL_EXT_mesh_shader : require
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    layout({layout_topology}, max_vertices = {max_vertices}, max_primitives = 1) out;

    void main() {{
        SetMeshOutputsEXT({max_vertices}, 1);
        gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        {index_assignment}
    }}
    """

    crossgl = generate_crossgl(code, "mesh")

    assert expected_layout in crossgl
    assert crossgl_call in crossgl
    assert "SetMeshOutputsEXT" not in crossgl
    assert "gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);" in crossgl
    assert index_assignment in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert "#extension GL_EXT_mesh_shader : require" in glsl
    assert "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in glsl
    assert expected_layout in glsl
    assert regenerated_call in glsl
    assert "gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);" in glsl
    assert index_assignment in glsl


@pytest.mark.parametrize(
    ("glsl_statement", "crossgl_call", "regenerated_statement"),
    [
        ("ignoreIntersectionEXT;", "IgnoreHit();", "ignoreIntersectionEXT;"),
        (
            "terminateRayEXT;",
            "AcceptHitAndEndSearch();",
            "terminateRayEXT;",
        ),
    ],
)
def test_parse_bare_ray_control_statements_roundtrip_canonically(
    glsl_statement,
    crossgl_call,
    regenerated_statement,
):
    code = f"""
    #version 460 core
    #extension GL_EXT_ray_tracing : require

    void main() {{
        {glsl_statement}
    }}
    """

    crossgl = generate_crossgl(code, "ray_any_hit")

    assert crossgl_call in crossgl
    assert glsl_statement not in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert "#extension GL_EXT_ray_tracing : require" in glsl
    assert regenerated_statement in glsl
    assert crossgl_call not in glsl


def test_parse_ray_hit_position_fetch_builtin_roundtrip():
    code = """
    #version 460 core
    #extension GL_EXT_ray_tracing : require
    #extension GL_EXT_ray_tracing_position_fetch : require

    void main() {
        vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];
        vec3 p2 = gl_HitTriangleVertexPositionsEXT[2];
        vec3 edge = p2 - p0;
    }
    """

    crossgl = generate_crossgl(code, "ray_closest_hit")

    assert "vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];" in crossgl
    assert "vec3 p2 = gl_HitTriangleVertexPositionsEXT[2];" in crossgl
    assert "vec3 edge = (p2 - p0);" in crossgl

    glsl = GLSLCodeGen().generate(crosstl.translator.parse(crossgl))

    assert "#extension GL_EXT_ray_tracing : require" in glsl
    assert "#extension GL_EXT_ray_tracing_position_fetch : require" in glsl
    assert "vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];" in glsl
    assert "vec3 p2 = gl_HitTriangleVertexPositionsEXT[2];" in glsl
    assert "vec3 edge = (p2 - p0);" in glsl


if __name__ == "__main__":
    pytest.main()
