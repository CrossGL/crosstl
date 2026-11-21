"""SPIR-V fragment shader interlock imports as CrossGL's interlock calls and layouts.

SPV_EXT_fragment_shader_interlock brackets a critical section with ``OpBeginInvocationInterlockEXT`` and
``OpEndInvocationInterlockEXT`` and chooses its ordering with one of four execution modes. The importer refused both
instructions, which were the ``rasterizer-ordered-buffer`` and ``rasterizer-ordered-image`` refused rows of the SPIR-V
input coverage. CrossGL spells them as GLSL's GL_ARB_fragment_shader_interlock does. A device read inside the section
must stay inside it even when its value is used after the section ends, and a read of a Uniform ``BufferBlock`` is a
device read: the importer took it for a read of a read-only uniform block and wrote it out after the store that
overwrote it.

``INTERLOCK`` is glslang's output, with ``SampleInterlockUnorderedEXT`` in place of the pixel mode, for::

    #version 460
    #extension GL_ARB_fragment_shader_interlock : require
    layout(sample_interlock_unordered) in;
    layout(std430, binding = 0) buffer Counts { uint counts[]; };
    layout(location = 0) out vec4 color;
    void main() {
        uint index = uint(gl_FragCoord.x);
        beginInvocationInterlockARB();
        uint before = counts[index];
        counts[index] = before + 1u;
        endInvocationInterlockARB();
        color = vec4(float(before));
    }
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

INTERLOCK = """               OpCapability Shader
               OpCapability FragmentShaderSampleInterlockEXT
               OpExtension "SPV_EXT_fragment_shader_interlock"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %gl_FragCoord %color
               OpExecutionMode %main OriginUpperLeft
               OpExecutionMode %main SampleInterlockUnorderedEXT
               OpName %main "main"
               OpName %gl_FragCoord "gl_FragCoord"
               OpName %Counts "Counts"
               OpMemberName %Counts 0 "counts"
               OpName %counts_buffer "counts_buffer"
               OpName %color "color"
               OpDecorate %gl_FragCoord BuiltIn FragCoord
               OpDecorate %_runtimearr_uint ArrayStride 4
               OpMemberDecorate %Counts 0 Offset 0
               OpDecorate %Counts BufferBlock
               OpDecorate %counts_buffer DescriptorSet 0
               OpDecorate %counts_buffer Binding 0
               OpDecorate %color Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Input_v4float = OpTypePointer Input %v4float
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
%_runtimearr_uint = OpTypeRuntimeArray %uint
     %Counts = OpTypeStruct %_runtimearr_uint
%_ptr_Uniform_Counts = OpTypePointer Uniform %Counts
%counts_buffer = OpVariable %_ptr_Uniform_Counts Uniform
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
      %int_0 = OpConstant %int 0
     %uint_1 = OpConstant %uint 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %color = OpVariable %_ptr_Output_v4float Output
       %main = OpFunction %void None %3
          %5 = OpLabel
         %10 = OpLoad %v4float %gl_FragCoord
         %11 = OpCompositeExtract %float %10 0
      %index = OpConvertFToU %uint %11
               OpBeginInvocationInterlockEXT
         %20 = OpAccessChain %_ptr_Uniform_uint %counts_buffer %int_0 %index
     %before = OpLoad %uint %20
         %22 = OpIAdd %uint %before %uint_1
               OpStore %20 %22
               OpEndInvocationInterlockEXT
         %30 = OpConvertUToF %float %before
         %31 = OpCompositeConstruct %v4float %30 %30 %30 %30
               OpStore %color %31
               OpReturn
               OpFunctionEnd
"""


def _translate(tmp_path, backend):
    path = tmp_path / "shader.spvasm"
    path.write_text(INTERLOCK, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_interlock_instructions_import_as_interlock_calls(tmp_path):
    crossgl = _translate(tmp_path, "cgl")

    assert "layout(sample_interlock_unordered) in;" in crossgl
    begin = crossgl.index("beginInvocationInterlockARB();")
    end = crossgl.index("endInvocationInterlockARB();")
    assert begin < crossgl.index("counts_buffer[0].counts[") < end


def test_a_read_in_the_section_is_evaluated_in_the_section(tmp_path):
    crossgl = _translate(tmp_path, "cgl")

    end = crossgl.index("endInvocationInterlockARB();")
    assert crossgl.index("before = counts_buffer[0].counts[") < end
    assert "counts_buffer[0].counts[" not in crossgl[end:]


def test_a_buffer_block_read_before_a_store_to_it_keeps_its_value(tmp_path):
    # A Uniform BufferBlock is device memory, not a read-only uniform block: the read of ``before`` must be
    # evaluated before ``counts[index] = before + 1`` overwrites it.
    path = tmp_path / "shader.spvasm"
    module = INTERLOCK.replace("               OpBeginInvocationInterlockEXT\n", "")
    path.write_text(module.replace("               OpEndInvocationInterlockEXT\n", ""))
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)

    assert "uint before = counts_buffer[0].counts[" in crossgl
    assert (
        "float4(float(before), float(before), float(before), float(before))" in crossgl
    )


def test_spirv_round_trip_keeps_the_interlock_and_its_mode(tmp_path):
    spirv = _translate(tmp_path, "vulkan")

    assert "OpCapability FragmentShaderSampleInterlockEXT" in spirv
    assert 'OpExtension "SPV_EXT_fragment_shader_interlock"' in spirv
    assert "SampleInterlockUnorderedEXT" in spirv
    assert spirv.index("OpBeginInvocationInterlockEXT") < spirv.index(
        "OpEndInvocationInterlockEXT"
    )
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        pytest.skip("spirv-as and spirv-val are not installed")
    source, binary = tmp_path / "round.spvasm", tmp_path / "round.spv"
    source.write_text(spirv, encoding="utf-8")
    _run(["spirv-as", "--target-env", "vulkan1.0", str(source), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


def test_glsl_output_declares_the_interlock_layout(tmp_path):
    glsl = _translate(tmp_path, "opengl")

    assert "#extension GL_ARB_fragment_shader_interlock : require" in glsl
    assert "layout(sample_interlock_unordered) in;" in glsl
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    source = tmp_path / "shader.frag"
    source.write_text(glsl, encoding="utf-8")
    _run([validator, "-V", str(source), "-o", str(tmp_path / "glsl.spv")])
