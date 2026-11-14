"""SPIR-V built-ins and function-local names import into CrossGL names every emitter knows.

Ray tracing and mesh built-ins (``WorldRayOriginKHR``, ``PrimitiveTriangleIndicesEXT``) came out as stage inputs
named after the decoration, which the GLSL emitter declared as ``in vec3 WorldRayOriginKHR;``; and two ``Function``
variables sharing an ``OpName`` (glslang names every ``in`` argument temporary ``param``) became one variable
declared twice. See :mod:`crosstl.backend.SPIRV.ray_tracing_import` and :mod:`crosstl.backend.SPIRV.local_names`.
"""

from __future__ import annotations

import re

import crosstl
from crosstl.backend.SPIRV.local_names import function_local_names

CLOSEST_HIT = """; SPIR-V
; Version: 1.4
               OpCapability RayTracingKHR
               OpExtension "SPV_KHR_ray_tracing"
               OpMemoryModel Logical GLSL450
               OpEntryPoint ClosestHitKHR %main "main" %payload %origin %tmax
               OpName %main "main"
               OpName %helper "helper"
               OpName %x "x"
               OpName %payload "payload"
               OpName %p1 "param"
               OpName %p2 "param"
               OpDecorate %origin BuiltIn WorldRayOriginKHR
               OpDecorate %tmax BuiltIn RayTmaxKHR
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
%_ptr_Function_float = OpTypePointer Function %float
    %fn_type = OpTypeFunction %float %_ptr_Function_float
%_ptr_Input_v3float = OpTypePointer Input %v3float
%_ptr_Input_float = OpTypePointer Input %float
%_ptr_IncomingRayPayloadKHR_v3float = OpTypePointer IncomingRayPayloadKHR %v3float
     %origin = OpVariable %_ptr_Input_v3float Input
       %tmax = OpVariable %_ptr_Input_float Input
    %payload = OpVariable %_ptr_IncomingRayPayloadKHR_v3float IncomingRayPayloadKHR
     %helper = OpFunction %float None %fn_type
          %x = OpFunctionParameter %_ptr_Function_float
         %20 = OpLabel
         %21 = OpLoad %float %x
               OpReturnValue %21
               OpFunctionEnd
       %main = OpFunction %void None %3
          %5 = OpLabel
         %p1 = OpVariable %_ptr_Function_float Function
         %p2 = OpVariable %_ptr_Function_float Function
          %6 = OpLoad %v3float %origin
          %7 = OpLoad %float %tmax
               OpStore %p1 %7
          %8 = OpFunctionCall %float %helper %p1
               OpStore %p2 %8
          %9 = OpFunctionCall %float %helper %p2
         %10 = OpVectorTimesScalar %v3float %6 %9
               OpStore %payload %10
               OpReturn
               OpFunctionEnd
"""


def _write(tmp_path, source):
    path = tmp_path / "shader.spvasm"
    path.write_text(source)
    return str(path)


def test_ray_builtins_import_as_glsl_builtins(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, CLOSEST_HIT), backend="cgl", format_output=False
    )
    assert "@gl_WorldRayOriginEXT" in crossgl
    assert "@gl_RayTmaxEXT" in crossgl
    assert "builtin(worldrayoriginkhr)" not in crossgl


def test_ray_builtins_reach_glsl_as_builtins_not_inputs(tmp_path):
    glsl = crosstl.translate(
        _write(tmp_path, CLOSEST_HIT), backend="glsl", format_output=False
    )
    assert "gl_WorldRayOriginEXT" in glsl and "gl_RayTmaxEXT" in glsl
    assert not re.search(r"^\s*in\s+\w+\s+(?:gl_)?WorldRayOrigin", glsl, re.M)


def test_locals_sharing_an_opname_get_distinct_names(tmp_path):
    glsl = crosstl.translate(
        _write(tmp_path, CLOSEST_HIT), backend="glsl", format_output=False
    )
    assert len(re.findall(r"\bfloat param;", glsl)) == 1
    assert len(re.findall(r"\bfloat param_1;", glsl)) == 1
    assert "helper(param)" in glsl and "helper(param_1)" in glsl


def test_function_local_names_keeps_unique_names_and_avoids_module_names():
    instructions = [
        ("%f", "OpFunction", ["%void", "None", "%t"], 1),
        ("%a", "OpVariable", ["%ptr", "Function"], 2),
        ("%b", "OpVariable", ["%ptr", "Function"], 3),
        ("%c", "OpVariable", ["%ptr", "Function"], 4),
        ("%d", "OpVariable", ["%ptr", "Function"], 5),
    ]
    names = {"%a": "param", "%b": "param", "%c": "param", "%d": "v", "%g": "param_1"}
    local = function_local_names(instructions, names)
    assert [local[k] for k in ("%a", "%b", "%c", "%d")] == [
        "param",
        "param_2",
        "param_3",
        "v",
    ]
    assert names["%b"] == "param"  # the module table is not modified
    unique = {"%a": "a", "%b": "b"}
    assert function_local_names(instructions[:3], unique) is unique
