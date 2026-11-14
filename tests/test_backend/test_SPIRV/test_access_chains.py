"""SPIR-V access chains import struct indices as member names at every step.

``OpAccessChain`` indexes through its pointee type: an index into a struct names a member, any other index selects
an element. The importer used to name only the first member of a flattened block and subscript the rest, so
``ubo.camera.pos`` came out as ``camera[0]`` and a struct reached through a function parameter as ``s[0][3]``.
See :mod:`crosstl.backend.SPIRV.access_chains`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.access_chains import chain_base_pointee, walk_indices

COMPUTE = """; SPIR-V
; Version: 1.0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %radius "radius(struct-Sphere-vf4-f11;"
               OpName %sphere "sphere"
               OpName %Sphere "Sphere"
               OpMemberName %Sphere 0 "center"
               OpMemberName %Sphere 1 "scale"
               OpName %Camera "Camera"
               OpMemberName %Camera 0 "pos"
               OpMemberName %Camera 1 "fov"
               OpName %UBO "UBO"
               OpMemberName %UBO 0 "aspect"
               OpMemberName %UBO 1 "camera"
               OpName %ubo ""
               OpName %Out "Out"
               OpMemberName %Out 0 "values"
               OpName %out ""
               OpName %local "local"
               OpMemberDecorate %Camera 0 Offset 0
               OpMemberDecorate %Camera 1 Offset 12
               OpMemberDecorate %UBO 0 Offset 0
               OpMemberDecorate %UBO 1 Offset 16
               OpDecorate %UBO Block
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %Out 0 Offset 0
               OpDecorate %Out BufferBlock
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 1
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %v3float = OpTypeVector %float 3
        %int = OpTypeInt 32 1
      %Sphere = OpTypeStruct %v4float %float
%_ptr_Function_Sphere = OpTypePointer Function %Sphere
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Function_float = OpTypePointer Function %float
   %radius_t = OpTypeFunction %float %_ptr_Function_Sphere
     %Camera = OpTypeStruct %v3float %float
        %UBO = OpTypeStruct %float %Camera
%_ptr_Uniform_UBO = OpTypePointer Uniform %UBO
%_ptr_Uniform_v3float = OpTypePointer Uniform %v3float
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_runtimearr_float = OpTypeRuntimeArray %float
        %Out = OpTypeStruct %_runtimearr_float
%_ptr_Uniform_Out = OpTypePointer Uniform %Out
        %ubo = OpVariable %_ptr_Uniform_UBO Uniform
        %out = OpVariable %_ptr_Uniform_Out Uniform
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_3 = OpConstant %int 3
    %float_1 = OpConstant %float 1
   %v4_one = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
       %main = OpFunction %void None %3
          %5 = OpLabel
      %local = OpVariable %_ptr_Function_Sphere Function
         %10 = OpAccessChain %_ptr_Function_v4float %local %int_0
               OpStore %10 %v4_one
         %11 = OpAccessChain %_ptr_Function_float %local %int_1
               OpStore %11 %float_1
         %12 = OpAccessChain %_ptr_Uniform_v3float %ubo %int_1 %int_0
         %13 = OpLoad %v3float %12
         %14 = OpCompositeExtract %float %13 0
         %15 = OpFunctionCall %float %radius %local
         %16 = OpFAdd %float %14 %15
         %17 = OpAccessChain %_ptr_Uniform_float %out %int_0 %int_0
               OpStore %17 %16
               OpReturn
               OpFunctionEnd
     %radius = OpFunction %float None %radius_t
     %sphere = OpFunctionParameter %_ptr_Function_Sphere
         %20 = OpLabel
         %21 = OpAccessChain %_ptr_Function_float %sphere %int_0 %int_3
         %22 = OpLoad %float %21
         %23 = OpAccessChain %_ptr_Function_float %sphere %int_1
         %24 = OpLoad %float %23
         %25 = OpFMul %float %22 %24
               OpReturnValue %25
               OpFunctionEnd
"""


def _write(tmp_path, source):
    path = tmp_path / "shader.spvasm"
    path.write_text(source)
    return str(path)


def test_fixture_is_valid_spirv(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    binary = tmp_path / "shader.spv"
    subprocess.run(
        ["spirv-as", _write(tmp_path, COMPUTE), "-o", str(binary)], check=True
    )
    subprocess.run(["spirv-val", str(binary)], check=True)


def test_nested_struct_member_of_flattened_block_keeps_its_name(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    assert re.search(r"\bcamera\.pos\b", crossgl)
    assert not re.search(r"\bcamera\s*\[", crossgl)


def test_struct_through_function_parameter_keeps_member_names(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    assert re.search(r"\bsphere\.center\s*\[\s*3\s*\]", crossgl)
    assert re.search(r"\bsphere\.scale\b", crossgl)
    assert not re.search(r"\bsphere\s*\[", crossgl)


def test_glsl_output_compiles(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    glsl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="glsl", format_output=False
    )
    path = tmp_path / "shader.comp"
    path.write_text(glsl)
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


class _Parser:
    def spirv_integer_constant_operand(self, operand, constants):
        try:
            return int(constants.get(operand, operand))
        except ValueError:
            return None

    def spirv_composite_index_type_id(self, type_id, index_operand, types):
        return types.get(type_id, {}).get("element_type")


TYPES = {
    "%S": {"kind": "struct", "member_types": ["%arr", "%float"]},
    "%arr": {"kind": "array", "element_type": "%float"},
    "%ptr": {"kind": "pointer", "type_id": "%S"},
}


def _walk(indices, type_id="%S"):
    return walk_indices(
        _Parser(),
        "base",
        type_id,
        indices,
        lambda op: op,
        {"%S": {"1": "w"}},
        TYPES,
        {"%c0": "0", "%c1": "1"},
    )


def test_unnamed_members_use_the_declared_fallback_name():
    access = _walk(["%c0", "%i"])
    assert access.index == "%i"
    assert access.array.member == "member0"


def test_struct_index_that_is_not_a_constant_is_refused():
    with pytest.raises(SyntaxError, match="in-range OpConstant"):
        _walk(["%i"])


def test_unknown_type_keeps_subscripts():
    access = _walk(["%c1"], type_id=None)
    assert access.index == "%c1"


def test_base_pointee_comes_from_variable_or_expression_type():
    assert (
        chain_base_pointee("%v", TYPES, {"%v": {"pointer_type_id": "%ptr"}}, {}) == "%S"
    )
    assert chain_base_pointee("%p", TYPES, {}, {"%p": "%ptr"}) == "%S"
    assert chain_base_pointee("%p", TYPES, {}, {}) is None
