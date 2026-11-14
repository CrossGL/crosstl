"""Geometry stages that write several vertex streams, in SPIR-V output and input.

SPIR-V output refused ``EmitStreamVertex``/``EndStreamPrimitive``; it now writes ``OpEmitStreamVertex``/
``OpEndStreamPrimitive`` with a constant stream, the ``GeometryStreams`` capability and a ``Stream`` decoration on
every output. SPIR-V input refused ``OpEmitStreamVertex``, and its outputs lost their ``Stream`` decorations; it now
writes the calls and ``@stream(n)``. HLSL output still refuses a numbered stream.

``MODULE`` is glslangValidator 16.6.0's output (``--target-env vulkan1.2``) for ``SOURCE``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

CROSSGL = """shader main {
    in vec4 position @location(0)[];
    out vec4 extra @location(0) @stream(1);

    geometry {
    layout(points) in;
    layout(points, max_vertices = 2) out;

    void main() {
        gl_Position = position[0];
        EmitStreamVertex(0);
        EndStreamPrimitive(0);
        extra = (position[0] * 2.0);
        EmitStreamVertex(STREAM);
        EndStreamPrimitive(1);
    }
    }
}
"""

SOURCE = """#version 450
layout(points) in;
layout(points, max_vertices = 2) out;
layout(location = 0) in vec4 position[];
layout(stream = 1, location = 0) out vec4 extra;
void main() {
    gl_Position = position[0];
    EmitStreamVertex(0);
    EndStreamPrimitive(0);
    extra = position[0] * 2.0;
    EmitStreamVertex(1);
    EndStreamPrimitive(1);
}
"""

MODULE = """; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 30
; Schema: 0
               OpCapability Geometry
               OpCapability GeometryStreams
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Geometry %main "main" %_ %position %extra
               OpExecutionMode %main InputPoints
               OpExecutionMode %main Invocations 1
               OpExecutionMode %main OutputPoints
               OpExecutionMode %main OutputVertices 2
               OpSource GLSL 450
               OpName %main "main"
               OpName %gl_PerVertex "gl_PerVertex"
               OpMemberName %gl_PerVertex 0 "gl_Position"
               OpMemberName %gl_PerVertex 1 "gl_PointSize"
               OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
               OpMemberName %gl_PerVertex 3 "gl_CullDistance"
               OpName %_ ""
               OpName %position "position"
               OpName %extra "extra"
               OpDecorate %gl_PerVertex Block
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
               OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
               OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
               OpDecorate %_ Stream 0
               OpDecorate %position Location 0
               OpDecorate %extra Stream 1
               OpDecorate %extra Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_PerVertex = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
          %_ = OpVariable %_ptr_Output_gl_PerVertex Output
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_arr_v4float_uint_1 = OpTypeArray %v4float %uint_1
%_ptr_Input__arr_v4float_uint_1 = OpTypePointer Input %_arr_v4float_uint_1
   %position = OpVariable %_ptr_Input__arr_v4float_uint_1 Input
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %extra = OpVariable %_ptr_Output_v4float Output
    %float_2 = OpConstant %float 2
      %int_1 = OpConstant %int 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %20 = OpAccessChain %_ptr_Input_v4float %position %int_0
         %21 = OpLoad %v4float %20
         %23 = OpAccessChain %_ptr_Output_v4float %_ %int_0
               OpStore %23 %21
               OpEmitStreamVertex %int_0
               OpEndStreamPrimitive %int_0
         %25 = OpAccessChain %_ptr_Input_v4float %position %int_0
         %26 = OpLoad %v4float %25
         %28 = OpVectorTimesScalar %v4float %26 %float_2
               OpStore %extra %28
               OpEmitStreamVertex %int_1
               OpEndStreamPrimitive %int_1
               OpReturn
               OpFunctionEnd
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _validate(tmp_path, spirv):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        return
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


def _streams(spirv):
    """``{name: stream}``; without SPIRV-Tools the ids are numbers that only ``OpName`` names."""
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', spirv))
    return {
        names.get(target, target): stream
        for target, stream in re.findall(r"OpDecorate %(\w+) Stream (\d+)", spirv)
    }


def _check_stream_module(tmp_path, spirv):
    assert "OpCapability GeometryStreams" in spirv
    emitted = re.findall(
        r"^\s*(OpEmitStreamVertex|OpEndStreamPrimitive) %(\w+)", spirv, flags=re.M
    )
    assert [opcode for opcode, _ in emitted] == [
        "OpEmitStreamVertex",
        "OpEndStreamPrimitive",
        "OpEmitStreamVertex",
        "OpEndStreamPrimitive",
    ], spirv
    streams = _streams(spirv)
    assert streams.get("extra") == "1" and streams.get("gl_Position") == "0", streams
    _validate(tmp_path, spirv)


@pytest.mark.parametrize(
    "source, suffix",
    [(CROSSGL.replace("STREAM", "1"), ".cgl"), (SOURCE, ".geom")],
    ids=["crossgl", "glsl"],
)
def test_spirv_output_writes_numbered_streams(tmp_path, source, suffix):
    _check_stream_module(tmp_path, _translate(tmp_path, source, suffix, "vulkan"))


def test_spirv_output_refuses_a_stream_that_is_not_a_constant(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="constant, non-negative stream"):
        _translate(
            tmp_path, CROSSGL.replace("STREAM", "int(position[0].x)"), ".cgl", "vulkan"
        )


def test_spirv_input_keeps_streams(tmp_path):
    crossgl = _translate(tmp_path, MODULE, ".spvasm", "cgl")
    assert "EmitStreamVertex(1);" in crossgl and "EndStreamPrimitive(1);" in crossgl
    assert re.search(r"float4 extra @output\b[^;]*@stream\(1\)", crossgl), crossgl
    _check_stream_module(tmp_path, _translate(tmp_path, MODULE, ".spvasm", "vulkan"))
    glsl = _translate(tmp_path, MODULE, ".spvasm", "opengl")
    assert re.search(r"layout\(stream = 1, location = 0\) out vec4 extra;", glsl), glsl
