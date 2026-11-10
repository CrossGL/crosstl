"""What a translation must keep of the geometry, tessellation, mesh and task stages."""

import re

import pytest

from tools.coverage import decorations, lowering
from tools.coverage.model import Feature


def _geometry(language, name, group):
    return Feature(language, "graphics", name, None, group=group, stage="geometry")


def test_an_instanced_geometry_translation_must_keep_more_than_one_invocation():
    # glslang, DXC and slangc write Invocations 1 for every geometry stage, so only a larger count keeps the mode.
    instanced = _geometry(
        "spirv",
        "geometry-invocations",
        "ExecutionMode.Invocations BuiltIn.InvocationId",
    )
    kept = (
        "OpExecutionMode %main InputPoints\nOpExecutionMode %main Invocations 2\n"
        "OpDecorate %gl_InvocationID BuiltIn InvocationId\n"
    )
    assert decorations.lost(instanced, "spirv", kept) == ""
    assert (
        decorations.lost(
            instanced, "spirv", kept.replace("Invocations 2", "Invocations 12")
        )
        == ""
    )
    assert decorations.lost(
        instanced, "spirv", kept.replace("Invocations 2", "Invocations 1")
    ) == ("the translation loses ExecutionMode.Invocations")
    lines = _geometry("spirv", "geometry-input-lines", "ExecutionMode.InputLines")
    assert decorations.lost(lines, "spirv", "OpExecutionMode %main InputLines\n") == ""
    assert decorations.lost(lines, "spirv", "OpExecutionMode %main InputPoints\n") == (
        "the translation loses ExecutionMode.InputLines"
    )


@pytest.mark.parametrize("language", ["hlsl", "slang"])
def test_an_hlsl_or_slang_stream_translation_keeps_a_second_stream_parameter(language):
    # DXC writes every HLSL output on stream 0 in SPIR-V and slangc cannot write a second stream there, so the
    # translation itself must declare a second point stream.
    streams = _geometry(
        language, "geometry-streams", "Decoration.Stream Capability.GeometryStreams"
    )
    source = (
        "[maxvertexcount(2)]\n"
        "void main(point float4 position[1] : POSITION, inout PointStream<GeometryOut> stream,\n"
        "          inout PointStream<ExtraOut> extras) {\n}\n"
    )
    assert decorations.lost(streams, language, source) == ""
    single = source.replace(",\n          inout PointStream<ExtraOut> extras", "")
    assert decorations.lost(streams, language, single) == (
        "the translation loses Decoration.Stream, Capability.GeometryStreams"
    )


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_geometry_primitives_instances_and_streams():
    lines = _geometry("glsl", "geometry-input-lines", "ExecutionMode.InputLines")
    source = (
        "#version 460\nlayout(lines) in;\nlayout(points, max_vertices = 1) out;\n"
        "layout(location = 0) in vec4 position[];\n"
        "void main() {\n    gl_Position = position[0];\n    EmitVertex();\n}\n"
    )
    assert decorations.lost(lines, "glsl", source) == ""
    assert decorations.lost(lines, "glsl", source.replace("(lines)", "(points)")) == (
        "the translation loses ExecutionMode.InputLines"
    )
    instanced = _geometry(
        "glsl",
        "geometry-invocations",
        "ExecutionMode.Invocations BuiltIn.InvocationId",
    )
    source = (
        "#version 460\nlayout(points, invocations = 2) in;\nlayout(points, max_vertices = 1) out;\n"
        "void main() {\n    gl_Position = vec4(float(gl_InvocationID));\n    EmitVertex();\n}\n"
    )
    assert decorations.lost(instanced, "glsl", source) == ""
    assert decorations.lost(
        instanced, "glsl", source.replace(", invocations = 2", "")
    ) == ("the translation loses ExecutionMode.Invocations")
    streams = _geometry(
        "glsl", "geometry-streams", "Decoration.Stream Capability.GeometryStreams"
    )
    source = (
        "#version 460\nlayout(points) in;\nlayout(points, max_vertices = 2) out;\n"
        "layout(stream = 1, location = 0) out vec4 extra;\n"
        "void main() {\n    gl_Position = vec4(1.0);\n    EmitStreamVertex(0);\n"
        "    extra = vec4(2.0);\n    EmitStreamVertex(1);\n}\n"
    )
    assert decorations.lost(streams, "glsl", source) == ""
    single = (
        "#version 460\nlayout(points) in;\nlayout(points, max_vertices = 2) out;\n"
        "layout(location = 0) out vec4 extra;\n"
        "void main() {\n    gl_Position = vec4(1.0);\n    extra = vec4(2.0);\n    EmitVertex();\n}\n"
    )
    assert decorations.lost(streams, "glsl", single) == (
        "the translation loses Decoration.Stream, Capability.GeometryStreams"
    )


def _patch(language, name, group, stage="tessellation-control"):
    return Feature(language, "graphics", name, None, group=group, stage=stage)


def test_a_per_patch_translation_must_keep_a_located_patch_variable():
    # Every tessellation stage decorates its levels Patch, so only a Patch variable with a Location keeps the value.
    output = _patch("spirv", "tessellation-patch-output", "Decoration.Patch")
    levels = "OpDecorate %gl_TessLevelOuter BuiltIn TessLevelOuter\nOpDecorate %gl_TessLevelOuter Patch\n"
    assert decorations.lost(output, "spirv", levels) == (
        "the translation loses Decoration.Patch"
    )
    assert decorations.lost(
        output, "spirv", levels + "OpDecorate %20 Location 1\nOpDecorate %24 Patch\n"
    ) == ("the translation loses Decoration.Patch")
    for kept in (
        "OpDecorate %extra Patch\nOpDecorate %extra Location 1\n",
        "OpDecorate %extra Location 1\nOpDecorate %extra Patch\n",
    ):
        assert decorations.lost(output, "spirv", levels + kept) == ""


def test_a_slang_per_patch_translation_keeps_a_user_semantic_beside_its_levels():
    # slangc writes no Patch decoration for a patch-constant member, so the translation itself must declare one.
    output = _patch("slang", "tessellation-patch-output", "Decoration.Patch")
    source = (
        "struct PatchConstants {\n    float edges[3] : SV_TessFactor;\n"
        "    float inside : SV_InsideTessFactor;\n    float4 extra : EXTRA;\n};\n"
    )
    assert decorations.lost(output, "slang", source) == ""
    assert decorations.lost(
        output, "slang", source.replace("    float4 extra : EXTRA;\n", "")
    ) == ("the translation loses Decoration.Patch")
    unrelated = "struct Other {\n    float4 extra : EXTRA;\n};\n"
    assert decorations.lost(
        output,
        "slang",
        unrelated + source.replace("    float4 extra : EXTRA;\n", ""),
    ) == ("the translation loses Decoration.Patch")


def test_a_metal_tessellation_translation_keeps_its_quad_patch_and_per_patch_input():
    quads = _patch(
        "metal",
        "tessellation-domain-quads",
        "ExecutionMode.Quads",
        "tessellation-evaluation",
    )
    patch_input = _patch(
        "metal",
        "tessellation-patch-input",
        "Decoration.Patch",
        "tessellation-evaluation",
    )
    assert decorations._wanted(quads, "metal") == [
        ("ExecutionMode.Quads", r'!"air\.patch", !"quad"')
    ]
    assert decorations._wanted(patch_input, "metal") == [
        ("Decoration.Patch", r'!"air\.patch_input"')
    ]


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_tessellation_domains_spacing_and_patch_values():
    quads = _patch(
        "glsl",
        "tessellation-domain-quads",
        "ExecutionMode.Quads",
        "tessellation-evaluation",
    )
    source = (
        "#version 460\nlayout(quads, equal_spacing, ccw) in;\nlayout(location = 0) in vec4 position[];\n"
        "void main() {\n    gl_Position = position[0] * gl_TessCoord.x;\n}\n"
    )
    assert decorations.lost(quads, "glsl", source) == ""
    assert decorations.lost(quads, "glsl", source.replace("quads", "triangles")) == (
        "the translation loses ExecutionMode.Quads"
    )
    odd = _patch(
        "glsl",
        "tessellation-spacing-fractional-odd",
        "ExecutionMode.SpacingFractionalOdd",
        "tessellation-evaluation",
    )
    spaced = source.replace("equal_spacing", "fractional_odd_spacing")
    assert decorations.lost(odd, "glsl", spaced) == ""
    assert decorations.lost(odd, "glsl", source) == (
        "the translation loses ExecutionMode.SpacingFractionalOdd"
    )
    output = _patch("glsl", "tessellation-patch-output", "Decoration.Patch")
    source = (
        "#version 460\nlayout(vertices = 3) out;\nlayout(location = 0) in vec4 position[];\n"
        "layout(location = 0) out vec4 outPosition[];\nlayout(location = 1) patch out vec4 extra;\n"
        "void main() {\n    outPosition[gl_InvocationID] = position[gl_InvocationID];\n"
        "    extra = position[0];\n    gl_TessLevelOuter[0] = 1.0;\n}\n"
    )
    assert decorations.lost(output, "glsl", source) == ""
    dropped = source.replace(
        "layout(location = 1) patch out vec4 extra;\n", ""
    ).replace("    extra = position[0];\n", "")
    assert decorations.lost(output, "glsl", dropped) == (
        "the translation loses Decoration.Patch"
    )


def _mesh(language, name, group):
    return Feature(language, "graphics", name, None, group=group, stage="mesh")


def test_a_mesh_translation_must_keep_its_topology_and_per_primitive_values():
    lines = _mesh(
        "spirv",
        "mesh-output-lines",
        "ExecutionMode.OutputLinesEXT BuiltIn.PrimitiveLineIndicesEXT",
    )
    kept = (
        "OpExecutionMode %main OutputLinesEXT\n"
        "OpDecorate %gl_PrimitiveLineIndicesEXT BuiltIn PrimitiveLineIndicesEXT\n"
    )
    assert decorations.lost(lines, "spirv", kept) == ""
    # A disassembler built from a grammar older than SPV_EXT_mesh_shader names the shared enumerants by their NV
    # spelling.
    assert (
        decorations.lost(lines, "spirv", kept.replace("LinesEXT\n", "LinesNV\n")) == ""
    )
    triangles = (
        "OpExecutionMode %main OutputTrianglesEXT\n"
        "OpDecorate %gl_PrimitiveTriangleIndicesEXT BuiltIn PrimitiveTriangleIndicesEXT\n"
    )
    assert decorations.lost(lines, "spirv", triangles) == (
        "the translation loses ExecutionMode.OutputLinesEXT, BuiltIn.PrimitiveLineIndicesEXT"
    )
    output = _mesh("spirv", "mesh-per-primitive-output", "Decoration.PerPrimitiveEXT")
    for spelling in ("PerPrimitiveEXT", "PerPrimitiveNV"):
        assert (
            decorations.lost(output, "spirv", f"OpDecorate %color {spelling}\n") == ""
        )
    assert decorations.lost(output, "spirv", "OpDecorate %color Location 0\n") == (
        "the translation loses Decoration.PerPrimitiveEXT"
    )


def test_a_mesh_translation_that_reads_the_payload_must_declare_task_payload_storage():
    # A storage class is no decoration, so the payload is kept only by a variable in task-payload storage.
    payload = _mesh(
        "spirv", "mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT"
    )
    pointer = "%_ptr_TaskPayloadWorkgroupEXT_Payload = OpTypePointer TaskPayloadWorkgroupEXT %Payload\n"
    variable = "%payload = OpVariable %_ptr_TaskPayloadWorkgroupEXT_Payload TaskPayloadWorkgroupEXT\n"
    assert decorations.lost(payload, "spirv", pointer + variable) == ""
    assert decorations.lost(payload, "spirv", pointer) == (
        "the translation loses StorageClass.TaskPayloadWorkgroupEXT"
    )
    private = variable.replace(" TaskPayloadWorkgroupEXT\n", " Private\n")
    assert decorations.lost(payload, "spirv", pointer + private) == (
        "the translation loses StorageClass.TaskPayloadWorkgroupEXT"
    )


def test_a_metal_mesh_translation_keeps_its_topology_primitive_data_and_payload_in_air():
    points = _mesh(
        "metal",
        "mesh-output-points",
        "ExecutionMode.OutputPoints BuiltIn.PrimitivePointIndicesEXT",
    )
    ((anchor, pattern),) = decorations._wanted(points, "metal")
    assert anchor == "ExecutionMode.OutputPoints"
    info = '!12 = !{!"air.mesh_type_info", !13, !14, i32 1, i32 1, !"air.point"}\n'
    assert re.search(pattern, info, re.M)
    assert not re.search(pattern, info.replace("air.point", "air.triangle"), re.M)
    for name, group, attribute in (
        (
            "mesh-per-primitive-output",
            "Decoration.PerPrimitiveEXT",
            "mesh_primitive_data",
        ),
        ("mesh-cull-primitive", "BuiltIn.CullPrimitiveEXT", "primitive_culled"),
        ("mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT", "payload"),
    ):
        assert decorations._wanted(_mesh("metal", name, group), "metal") == [
            (group, rf'!"air\.{attribute}"')
        ]


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_mesh_topologies_per_primitive_outputs_and_payloads():
    header = "#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x = 1) in;\n"
    lines = _mesh(
        "glsl",
        "mesh-output-lines",
        "ExecutionMode.OutputLinesEXT BuiltIn.PrimitiveLineIndicesEXT",
    )
    source = (
        header + "layout(lines, max_vertices = 2, max_primitives = 1) out;\n"
        "void main() {\n    SetMeshOutputsEXT(2, 1);\n"
        "    gl_PrimitiveLineIndicesEXT[0] = uvec2(0, 1);\n}\n"
    )
    assert decorations.lost(lines, "glsl", source) == ""
    triangles = (
        header + "layout(triangles, max_vertices = 3, max_primitives = 1) out;\n"
        "void main() {\n    SetMeshOutputsEXT(3, 1);\n"
        "    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);\n}\n"
    )
    assert decorations.lost(lines, "glsl", triangles) == (
        "the translation loses ExecutionMode.OutputLinesEXT, BuiltIn.PrimitiveLineIndicesEXT"
    )
    output = _mesh("glsl", "mesh-per-primitive-output", "Decoration.PerPrimitiveEXT")
    located = triangles.replace(
        "void main() {\n",
        "layout(location = 0) perprimitiveEXT out vec4 color[];\nvoid main() {\n",
    ).replace("}\n", "    color[0] = vec4(1.0);\n}\n")
    assert decorations.lost(output, "glsl", located) == ""
    assert decorations.lost(
        output, "glsl", located.replace("perprimitiveEXT ", "")
    ) == ("the translation loses Decoration.PerPrimitiveEXT")
    payload = _mesh("glsl", "mesh-task-payload", "StorageClass.TaskPayloadWorkgroupEXT")
    reads = triangles.replace(
        "void main() {\n",
        "struct Payload {\n    uint value;\n};\ntaskPayloadSharedEXT Payload payload;\n"
        "void main() {\n",
    ).replace(
        "}\n",
        "    gl_MeshVerticesEXT[0].gl_Position = vec4(float(payload.value));\n}\n",
    )
    assert decorations.lost(payload, "glsl", reads) == ""
    assert decorations.lost(payload, "glsl", triangles) == (
        "the translation loses StorageClass.TaskPayloadWorkgroupEXT"
    )


@pytest.mark.skipif(not lowering.can_lower("glsl"), reason="glslang is not installed")
def test_glslang_lowers_the_ids_a_task_stage_reads():
    feature = Feature(
        "glsl",
        "graphics",
        "task-workgroup-id",
        None,
        group="BuiltIn.WorkgroupId",
        stage="task",
    )
    source = (
        "#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x = 1) in;\n"
        "struct Payload {\n    uint value;\n};\ntaskPayloadSharedEXT Payload payload;\n"
        "void main() {\n    payload.value = gl_WorkGroupID.x;\n    EmitMeshTasksEXT(1, 1, 1);\n}\n"
    )
    assert decorations.lost(feature, "glsl", source) == ""
    assert decorations.lost(
        feature, "glsl", source.replace("gl_WorkGroupID.x", "0u")
    ) == ("the translation loses BuiltIn.WorkgroupId")
