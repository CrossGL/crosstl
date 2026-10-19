"""Ray tracing payloads survive SPIR-V import and emission, and 1.4 entry points list every global they use.

The SPIR-V importer read ``MissKHR`` and ``CallableKHR`` entry points as fragment shaders and dropped their
``IncomingRayPayloadKHR``/``IncomingCallableDataKHR`` variables; the emitter gave ray payload declarations
``Private`` storage and left globals other than inputs, outputs and workgroup variables out of 1.4 entry-point
interfaces. See :mod:`crosstl.translator.codegen.spirv_entry_interfaces`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_entry_interfaces import (
    complete_entry_point_interfaces,
)

MISS = """; SPIR-V
; Version: 1.4
               OpCapability RayTracingKHR
               OpExtension "SPV_KHR_ray_tracing"
               OpMemoryModel Logical GLSL450
               OpEntryPoint MissKHR %main "main" %hitValue
               OpName %main "main"
               OpName %hitValue "hitValue"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
%_ptr_IncomingRayPayloadKHR_v3float = OpTypePointer IncomingRayPayloadKHR %v3float
   %hitValue = OpVariable %_ptr_IncomingRayPayloadKHR_v3float IncomingRayPayloadKHR
    %float_0 = OpConstant %float 0
  %float_0_2 = OpConstant %float 0.2
         %12 = OpConstantComposite %v3float %float_0 %float_0 %float_0_2
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpStore %hitValue %12
               OpReturn
               OpFunctionEnd
"""

CALLABLE = """; SPIR-V
; Version: 1.4
               OpCapability RayTracingKHR
               OpExtension "SPV_KHR_ray_tracing"
               OpMemoryModel Logical GLSL450
               OpEntryPoint CallableKHR %main "main" %outColor
               OpName %main "main"
               OpName %outColor "outColor"
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
%_ptr_IncomingCallableDataKHR_v3float = OpTypePointer IncomingCallableDataKHR %v3float
   %outColor = OpVariable %_ptr_IncomingCallableDataKHR_v3float IncomingCallableDataKHR
    %float_1 = OpConstant %float 1
         %12 = OpConstantComposite %v3float %float_1 %float_1 %float_1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpStore %outColor %12
               OpReturn
               OpFunctionEnd
"""

GLSL_MISS = """#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 hitValue;
vec3 background = vec3(0.1, 0.2, 0.3);
void main() { hitValue = background; }
"""


def translate(tmp_path, source, suffix, backend, source_backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, source_backend=source_backend, format_output=False
    )


def spirv_val(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    (tmp_path / "out.spvasm").write_text(code)
    assembled = subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.2", "out.spvasm", "-o", "out.spv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.2", "out.spv"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stderr


@pytest.mark.parametrize(
    "source, stage, qualifier",
    [
        (MISS, "ray_miss", "@rayPayloadInEXT"),
        (CALLABLE, "ray_callable", "@callableDataInEXT"),
    ],
    ids=["miss", "callable"],
)
def test_spirv_import_keeps_ray_stages_and_incoming_payloads(
    tmp_path, source, stage, qualifier
):
    cgl = translate(tmp_path, source, ".spvasm", "cgl", "vulkan")
    assert re.search(rf"\b{stage}\s*\{{", cgl)
    assert "fragment" not in cgl
    assert qualifier in cgl


@pytest.mark.parametrize(
    "source, storage",
    [(MISS, "IncomingRayPayloadKHR"), (CALLABLE, "IncomingCallableDataKHR")],
    ids=["miss", "callable"],
)
def test_spirv_round_trip_keeps_payload_storage(tmp_path, source, storage):
    spirv = translate(tmp_path, source, ".spvasm", "vulkan", "vulkan")
    variable = re.search(rf"%(\w+) = OpVariable %\w+ {storage}\b", spirv)
    assert variable, spirv
    entry = next(line for line in spirv.splitlines() if "OpEntryPoint" in line)
    assert f"%{variable.group(1)}" in entry.split('"main"', 1)[1]
    spirv_val(tmp_path, spirv)


def test_glsl_ray_payloads_and_private_globals_are_entry_interfaces(tmp_path):
    spirv = translate(tmp_path, GLSL_MISS, ".rmiss", "vulkan", "opengl")
    assert re.search(r"OpVariable %\w+ IncomingRayPayloadKHR", spirv)
    spirv_val(tmp_path, spirv)


def test_entry_point_interfaces_follow_the_call_tree():
    entry_points = ['OpEntryPoint MissKHR %1 "main" %5']
    global_variables = [
        "%5 = OpVariable %9 IncomingRayPayloadKHR",
        "%6 = OpVariable %10 Private",
        "%7 = OpVariable %11 Uniform",
        "%8 = OpVariable %12 Private",
    ]
    body = [
        "%1 = OpFunction %2 None %3",
        "%13 = OpLabel",
        "OpStore %5 %20",
        "%14 = OpFunctionCall %2 %4",
        "OpReturn",
        "OpFunctionEnd",
        "%4 = OpFunction %2 None %3",
        "%15 = OpLabel",
        "%16 = OpLoad %21 %7",
        "OpStore %6 %16",
        "OpReturn",
        "OpFunctionEnd",
    ]
    assert complete_entry_point_interfaces(entry_points, global_variables, body) == [
        'OpEntryPoint MissKHR %1 "main" %5 %6 %7'
    ]
