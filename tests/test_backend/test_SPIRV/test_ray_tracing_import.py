"""SPIR-V ray tracing, mesh shading and per-vertex interface instructions import into CrossGL without losing effects.

The importer used to drop result-less instructions it did not know, so ``OpTraceRayKHR`` vanished from closest-hit
shaders and ``OpEmitMeshTasksEXT`` from task shaders; task and mesh entry points were read as fragment shaders, and
access chains into structs held in ray payloads or per-vertex interface arrays came out as numeric indices
(``payload[0]`` for ``payload.hitValue``). See :mod:`crosstl.backend.SPIRV.ray_tracing_import`.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.backend.SPIRV.ray_tracing_import import (
    SpirvInstructionUnsupportedError,
    ray_variable_locations,
)

HEADER = """; SPIR-V
; Version: 1.4
               OpCapability RayTracingKHR
               OpExtension "SPV_KHR_ray_tracing"
               OpMemoryModel Logical GLSL450
"""

CLOSEST_HIT = (
    HEADER
    + """               OpEntryPoint ClosestHitKHR %main "main" %payload %shadowed %as
               OpName %main "main"
               OpName %Payload "Payload"
               OpMemberName %Payload 0 "color"
               OpMemberName %Payload 1 "depth"
               OpName %payload "payload"
               OpName %shadowed "shadowed"
               OpName %as "topLevelAS"
               OpDecorate %as DescriptorSet 0
               OpDecorate %as Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
    %v3float = OpTypeVector %float 3
    %Payload = OpTypeStruct %v3float %float
%_ptr_IncomingRayPayloadKHR_Payload = OpTypePointer IncomingRayPayloadKHR %Payload
%_ptr_IncomingRayPayloadKHR_v3float = OpTypePointer IncomingRayPayloadKHR %v3float
%_ptr_RayPayloadKHR_bool = OpTypePointer RayPayloadKHR %bool
         %as_t = OpTypeAccelerationStructureKHR
%_ptr_UniformConstant_as = OpTypePointer UniformConstant %as_t
    %payload = OpVariable %_ptr_IncomingRayPayloadKHR_Payload IncomingRayPayloadKHR
   %shadowed = OpVariable %_ptr_RayPayloadKHR_bool RayPayloadKHR
         %as = OpVariable %_ptr_UniformConstant_as UniformConstant
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
   %uint_255 = OpConstant %uint 255
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
       %true = OpConstantTrue %bool
     %origin = OpConstantComposite %v3float %float_0 %float_0 %float_0
        %dir = OpConstantComposite %v3float %float_0 %float_1 %float_0
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpStore %shadowed %true
         %10 = OpLoad %as_t %as
               OpTraceRayKHR %10 %uint_1 %uint_255 %uint_0 %uint_0 %uint_1 %origin %float_0 %dir %float_1 %shadowed
         %11 = OpAccessChain %_ptr_IncomingRayPayloadKHR_v3float %payload %int_0
               OpStore %11 %dir
               OpReturn
               OpFunctionEnd
"""
)

ANY_HIT = HEADER + """               OpEntryPoint AnyHitKHR %main "main"
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpIgnoreIntersectionKHR
               OpFunctionEnd
"""

INTERSECTION = HEADER + """               OpEntryPoint IntersectionKHR %main "main"
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
    %float_1 = OpConstant %float 1
     %uint_0 = OpConstant %uint 0
       %main = OpFunction %void None %3
          %5 = OpLabel
          %6 = OpReportIntersectionKHR %bool %float_1 %uint_0
               OpReturn
               OpFunctionEnd
"""

CALLABLE_CALLER = (
    HEADER + """               OpEntryPoint RayGenerationKHR %main "main" %data
               OpName %main "main"
               OpName %data "callableData"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
    %v3float = OpTypeVector %float 3
%_ptr_CallableDataKHR_v3float = OpTypePointer CallableDataKHR %v3float
       %data = OpVariable %_ptr_CallableDataKHR_v3float CallableDataKHR
     %uint_2 = OpConstant %uint 2
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpExecuteCallableKHR %uint_2 %data
               OpReturn
               OpFunctionEnd
"""
)

UNSUPPORTED = HEADER + """               OpEntryPoint RayGenerationKHR %main "main"
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpBeginInvocationInterlockEXT
               OpReturn
               OpFunctionEnd
"""

TASK = """; SPIR-V
; Version: 1.4
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TaskEXT %main "main"
               OpExecutionMode %main LocalSize 4 1 1
               OpName %main "main"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
     %uint_1 = OpConstant %uint 1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpEmitMeshTasksEXT %uint_3 %uint_1 %uint_1
               OpFunctionEnd
"""

GEOMETRY = """; SPIR-V
; Version: 1.0
               OpCapability Geometry
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %gl_in %pos
               OpExecutionMode %main Triangles
               OpExecutionMode %main Invocations 1
               OpExecutionMode %main OutputTriangleStrip
               OpExecutionMode %main OutputVertices 3
               OpName %main "main"
               OpName %gl_PerVertex "gl_PerVertex"
               OpMemberName %gl_PerVertex 0 "gl_Position"
               OpName %gl_in "gl_in"
               OpName %pos "pos"
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpDecorate %gl_PerVertex Block
               OpDecorate %pos Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %uint_3 = OpConstant %uint 3
      %int_0 = OpConstant %int 0
      %int_2 = OpConstant %int 2
%gl_PerVertex = OpTypeStruct %v4float
%_arr_gl_PerVertex_uint_3 = OpTypeArray %gl_PerVertex %uint_3
%_ptr_Input__arr_gl_PerVertex_uint_3 = OpTypePointer Input %_arr_gl_PerVertex_uint_3
      %gl_in = OpVariable %_ptr_Input__arr_gl_PerVertex_uint_3 Input
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
        %pos = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %3
          %5 = OpLabel
         %10 = OpAccessChain %_ptr_Input_v4float %gl_in %int_2 %int_0
         %11 = OpLoad %v4float %10
               OpStore %pos %11
               OpEmitVertex
               OpReturn
               OpFunctionEnd
"""


def import_spirv(tmp_path, source):
    path = tmp_path / "shader.spvasm"
    path.write_text(source)
    return crosstl.translate(str(path), backend="cgl", format_output=False)


def test_trace_ray_names_the_payload_by_an_assigned_location(tmp_path):
    crossgl = import_spirv(tmp_path, CLOSEST_HIT)

    # Undecorated payloads get the lowest free location, in module order.
    assert "@location(0) @rayPayloadInEXT" in crossgl
    assert "bool shadowed @location(1) @rayPayloadEXT" in crossgl
    assert "TraceRay(topLevelAS, 1, 255, 0, 0, 1," in crossgl
    assert crossgl.count("TraceRay(") == 1
    assert ", 1);" in crossgl.split("TraceRay(", 1)[1].split("\n", 1)[0]


def test_struct_payload_access_chains_use_member_names(tmp_path):
    crossgl = import_spirv(tmp_path, CLOSEST_HIT)

    assert "payload.color = " in crossgl
    assert "payload[0]" not in crossgl


def test_hit_control_statements_become_intrinsics(tmp_path):
    assert "IgnoreHit()" in import_spirv(tmp_path, ANY_HIT)
    assert "ReportHit(1.0, 0);" in import_spirv(tmp_path, INTERSECTION)


def test_execute_callable_names_the_callable_data_location(tmp_path):
    crossgl = import_spirv(tmp_path, CALLABLE_CALLER)

    assert "@location(0) @callableDataEXT" in crossgl
    assert "CallShader(2, 0);" in crossgl


def test_unknown_statement_instructions_are_refused_not_dropped(tmp_path):
    with pytest.raises(SpirvInstructionUnsupportedError) as raised:
        import_spirv(tmp_path, UNSUPPORTED)

    assert "OpBeginInvocationInterlockEXT" in str(raised.value)
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.spirv-instruction-unsupported"
    )


def test_task_entry_keeps_its_stage_workgroup_size_and_mesh_dispatch(tmp_path):
    crossgl = import_spirv(tmp_path, TASK)

    assert "task {" in crossgl
    assert "fragment" not in crossgl
    assert "layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;" in crossgl
    assert "DispatchMesh(3, 1, 1);" in crossgl


def test_per_vertex_input_arrays_index_the_vertex_then_name_the_member(tmp_path):
    crossgl = import_spirv(tmp_path, GEOMETRY)

    assert "gl_in[2].gl_Position" in crossgl
    assert "gl_in[2][0]" not in crossgl


def test_ray_locations_keep_decorations_and_fill_gaps_per_family():
    variables = [
        {"id": "%a", "storage_class": "RayPayloadKHR"},
        {"id": "%b", "storage_class": "IncomingRayPayloadKHR"},
        {"id": "%c", "storage_class": "CallableDataKHR"},
        {"id": "%d", "storage_class": "HitAttributeKHR"},
        {"id": "%e", "storage_class": "RayPayloadKHR"},
    ]
    decorations = {"%b": [("Location", ["0"])], "%e": [("Location", ["2"])]}

    assert ray_variable_locations(variables, decorations) == {
        "%a": 1,
        "%b": 0,
        "%c": 0,
        "%e": 2,
    }
