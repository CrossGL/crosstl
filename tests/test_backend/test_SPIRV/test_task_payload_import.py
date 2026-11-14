"""The SPIR-V importer declares a task or mesh stage's TaskPayloadWorkgroupEXT variable.

A task stage writes its payload and a mesh stage reads it through one ``TaskPayloadWorkgroupEXT`` variable
(SPV_EXT_mesh_shader). The importer dropped the declaration while keeping its uses, so the CrossGL read an undeclared
``payload`` and no target could take the import. It now declares the variable as the GLSL importer does,
``Payload payload @taskPayloadSharedEXT``, which every emitter lowers: GLSL ``taskPayloadSharedEXT``, HLSL ``in
payload``/``groupshared`` and ``DispatchMesh``, Metal ``object_data``.

The modules are glslangValidator 16.6.0's output for GL_EXT_mesh_shader shaders (``--target-env vulkan1.2``).
"""

import re
import shutil
import subprocess

import pytest

import crosstl

MESH = """; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 46
; Schema: 0
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint MeshEXT %main "main" %gl_MeshVerticesEXT %payload %gl_PrimitiveTriangleIndicesEXT
               OpExecutionMode %main LocalSize 1 1 1
               OpExecutionMode %main OutputVertices 3
               OpExecutionMode %main OutputPrimitivesEXT 1
               OpExecutionMode %main OutputTrianglesEXT
               OpSource GLSL 460
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %gl_MeshPerVertexEXT "gl_MeshPerVertexEXT"
               OpMemberName %gl_MeshPerVertexEXT 0 "gl_Position"
               OpMemberName %gl_MeshPerVertexEXT 1 "gl_PointSize"
               OpMemberName %gl_MeshPerVertexEXT 2 "gl_ClipDistance"
               OpMemberName %gl_MeshPerVertexEXT 3 "gl_CullDistance"
               OpName %gl_MeshVerticesEXT "gl_MeshVerticesEXT"
               OpName %Payload "Payload"
               OpMemberName %Payload 0 "value"
               OpName %payload "payload"
               OpName %gl_PrimitiveTriangleIndicesEXT "gl_PrimitiveTriangleIndicesEXT"
               OpDecorate %gl_MeshPerVertexEXT Block
               OpMemberDecorate %gl_MeshPerVertexEXT 0 BuiltIn Position
               OpMemberDecorate %gl_MeshPerVertexEXT 1 BuiltIn PointSize
               OpMemberDecorate %gl_MeshPerVertexEXT 2 BuiltIn ClipDistance
               OpMemberDecorate %gl_MeshPerVertexEXT 3 BuiltIn CullDistance
               OpDecorate %gl_PrimitiveTriangleIndicesEXT BuiltIn PrimitiveTriangleIndicesEXT
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
     %uint_1 = OpConstant %uint 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_MeshPerVertexEXT = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_arr_gl_MeshPerVertexEXT_uint_3 = OpTypeArray %gl_MeshPerVertexEXT %uint_3
%_ptr_Output__arr_gl_MeshPerVertexEXT_uint_3 = OpTypePointer Output %_arr_gl_MeshPerVertexEXT_uint_3
%gl_MeshVerticesEXT = OpVariable %_ptr_Output__arr_gl_MeshPerVertexEXT_uint_3 Output
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
         %20 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %int_1 = OpConstant %int 1
    %Payload = OpTypeStruct %uint
%_ptr_TaskPayloadWorkgroupEXT_Payload = OpTypePointer TaskPayloadWorkgroupEXT %Payload
    %payload = OpVariable %_ptr_TaskPayloadWorkgroupEXT_Payload TaskPayloadWorkgroupEXT
%_ptr_TaskPayloadWorkgroupEXT_uint = OpTypePointer TaskPayloadWorkgroupEXT %uint
      %int_2 = OpConstant %int 2
         %34 = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_1
     %v3uint = OpTypeVector %uint 3
%_arr_v3uint_uint_1 = OpTypeArray %v3uint %uint_1
%_ptr_Output__arr_v3uint_uint_1 = OpTypePointer Output %_arr_v3uint_uint_1
%gl_PrimitiveTriangleIndicesEXT = OpVariable %_ptr_Output__arr_v3uint_uint_1 Output
     %uint_0 = OpConstant %uint 0
     %uint_2 = OpConstant %uint 2
         %42 = OpConstantComposite %v3uint %uint_0 %uint_1 %uint_2
%_ptr_Output_v3uint = OpTypePointer Output %v3uint
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_1 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpSetMeshOutputsEXT %uint_3 %uint_1
         %22 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_0 %int_0
               OpStore %22 %20
         %28 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_uint %payload %int_0
         %29 = OpLoad %uint %28
         %30 = OpConvertUToF %float %29
         %31 = OpCompositeConstruct %v4float %30 %float_0 %float_0 %float_1
         %32 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_1 %int_0
               OpStore %32 %31
         %35 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_2 %int_0
               OpStore %35 %34
         %44 = OpAccessChain %_ptr_Output_v3uint %gl_PrimitiveTriangleIndicesEXT %int_0
               OpStore %44 %42
               OpReturn
               OpFunctionEnd
"""

TASK = """; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 19
; Schema: 0
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TaskEXT %main "main" %payload
               OpExecutionMode %main LocalSize 1 1 1
               OpSource GLSL 460
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %Payload "Payload"
               OpMemberName %Payload 0 "value"
               OpName %payload "payload"
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
    %Payload = OpTypeStruct %uint
%_ptr_TaskPayloadWorkgroupEXT_Payload = OpTypePointer TaskPayloadWorkgroupEXT %Payload
    %payload = OpVariable %_ptr_TaskPayloadWorkgroupEXT_Payload TaskPayloadWorkgroupEXT
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
     %uint_7 = OpConstant %uint 7
%_ptr_TaskPayloadWorkgroupEXT_uint = OpTypePointer TaskPayloadWorkgroupEXT %uint
     %uint_1 = OpConstant %uint 1
     %v3uint = OpTypeVector %uint 3
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_1 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %14 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_uint %payload %int_0
               OpStore %14 %uint_7
               OpEmitMeshTasksEXT %uint_1 %uint_1 %uint_1 %payload
               OpFunctionEnd
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "module.spvasm"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("source", [MESH, TASK], ids=["mesh", "task"])
def test_import_declares_the_task_payload(tmp_path, source):
    crossgl = _translate(tmp_path, source, "cgl")
    assert re.search(
        r"\bPayload\s+payload\s*@\s*taskPayloadSharedEXT\s*;", crossgl
    ), crossgl


@pytest.mark.parametrize("source", [MESH, TASK], ids=["mesh", "task"])
def test_spirv_round_trip_keeps_the_task_payload(tmp_path, source):
    spirv = _translate(tmp_path, source, "vulkan")
    variable = re.search(
        r"^\s*(%\w+) = OpVariable %\w+ TaskPayloadWorkgroupEXT\b", spirv, flags=re.M
    )
    assert variable is not None, spirv
    if source is TASK:
        emit = re.search(r"^\s*OpEmitMeshTasksEXT\b(.*)$", spirv, flags=re.M)
        assert emit is not None and emit.group(1).split()[-1] == variable.group(
            1
        ), spirv
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler and validator:
        text, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        text.write_text(spirv)
        environment = ["--target-env", "vulkan1.2"]
        assembled = subprocess.run(
            [assembler, *environment, str(text), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = subprocess.run(
            [validator, *environment, str(binary)], capture_output=True, text=True
        )
        assert validated.returncode == 0, validated.stderr + spirv


def test_mesh_payload_reaches_hlsl_metal_and_glsl(tmp_path):
    hlsl = _translate(tmp_path, MESH, "directx")
    assert re.search(r"\bin\s+payload\s+Payload\s+payload\b", hlsl), hlsl
    metal = _translate(tmp_path, MESH, "metal")
    assert re.search(
        r"\bobject_data\s+Payload\s*&\s*payload\s*\[\[payload\]\]", metal
    ), metal
    glsl = _translate(tmp_path, MESH, "opengl")
    assert re.search(r"\btaskPayloadSharedEXT\s+Payload\s+payload\s*;", glsl), glsl


def test_task_payload_reaches_hlsl_and_glsl(tmp_path):
    hlsl = _translate(tmp_path, TASK, "directx")
    assert re.search(r"\bgroupshared\s+Payload\s+payload\s*;", hlsl), hlsl
    assert re.search(r"DispatchMesh\(1,\s*1,\s*1,\s*payload\)", hlsl), hlsl
    glsl = _translate(tmp_path, TASK, "opengl")
    assert re.search(r"\btaskPayloadSharedEXT\s+Payload\s+payload\s*;", glsl), glsl


GLSL_TASK = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
struct Payload { uint value; };
taskPayloadSharedEXT Payload payload;
void main() {
    payload.value = 7u;
    EmitMeshTasksEXT(1, 1, 1);
}
"""


def test_glsl_emit_mesh_tasks_passes_the_stage_payload(tmp_path):
    # EmitMeshTasksEXT(x, y, z) hands the stage's taskPayloadSharedEXT variable to the mesh workgroups; the SPIR-V
    # emitter wrote OpEmitMeshTasksEXT without it, so they read no payload.
    path = tmp_path / "shader.task"
    path.write_text(GLSL_TASK)
    spirv = crosstl.translate(str(path), backend="vulkan")
    variable = re.search(
        r"^\s*(%\w+) = OpVariable %\w+ TaskPayloadWorkgroupEXT\b", spirv, flags=re.M
    )
    emit = re.search(r"^\s*OpEmitMeshTasksEXT\b(.*)$", spirv, flags=re.M)
    assert variable is not None and emit is not None, spirv
    assert emit.group(1).split()[-1] == variable.group(1), spirv
