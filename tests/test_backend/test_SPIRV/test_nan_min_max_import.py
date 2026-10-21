"""GLSL.std.450 NMin, NMax and NClamp keep their NaN rule when imported.

NMin and NMax return the operand that is not a NaN, and NClamp is defined through them
(GLSL.std.450, instructions 79 to 81). FMin and FMax leave that case undefined. The importer
used to read both families as CrossGL ``min``/``max``, so the NaN rule was lost and a SPIR-V
round trip emitted ``FMin``. They now import as the C99 ``fmin``/``fmax`` (C99 7.12.12.2-3),
which every emitter lowers with that rule or refuses.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """; SPIR-V
; Version: 1.3
; Bound: 40
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %Buf "Buf"
               OpMemberName %Buf 0 "v"
               OpName %buf "buf"
               OpName %gid "gl_GlobalInvocationID"
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %v3uint = OpTypeVector %uint 3
        %arr = OpTypeRuntimeArray %float
        %Buf = OpTypeStruct %arr
    %ptr_buf = OpTypePointer StorageBuffer %Buf
        %buf = OpVariable %ptr_buf StorageBuffer
     %ptr_in = OpTypePointer Input %v3uint
        %gid = OpVariable %ptr_in Input
  %ptr_float = OpTypePointer StorageBuffer %float
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_2 = OpConstant %int 2
      %int_3 = OpConstant %int 3
      %int_4 = OpConstant %int 4
      %int_5 = OpConstant %int 5
       %main = OpFunction %void None %3
          %5 = OpLabel
         %p0 = OpAccessChain %ptr_float %buf %int_0 %int_0
         %a = OpLoad %float %p0
         %p1 = OpAccessChain %ptr_float %buf %int_0 %int_1
         %b = OpLoad %float %p1
         %p2 = OpAccessChain %ptr_float %buf %int_0 %int_2
         %nmin = OpExtInst %float %1 NMin %a %b
         %nmax = OpExtInst %float %1 NMax %a %b
         %nclamp = OpExtInst %float %1 NClamp %nmin %a %b
         %fmin = OpExtInst %float %1 FMin %nclamp %nmax
               OpStore %p2 %fmin
         %p3 = OpAccessChain %ptr_float %buf %int_0 %int_3
               OpStore %p3 %nmax
               OpReturn
               OpFunctionEnd
"""


def _write(tmp_path, source=COMPUTE, name="shader.spvasm"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _translate(tmp_path, backend, source=COMPUTE):
    return crosstl.translate(
        _write(tmp_path, source), backend=backend, format_output=False
    )


def test_fixture_is_valid_spirv(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", _write(tmp_path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_nmin_nmax_and_nclamp_import_as_nan_avoiding_calls(tmp_path):
    crossgl = re.sub(r"\s+", " ", _translate(tmp_path, "cgl"))
    # The stores to v[2] and v[3] may alias the loaded words, so both loads are read once first.
    assert "float a = buf[0].v[0]; float b = buf[0].v[1];" in crossgl
    a, b = "a", "b"
    nmin = f"fmin({a}, {b})"
    nclamp = f"fmin(fmax({nmin}, {a}), {b})"
    # FMin keeps CrossGL min: its NaN case is implementation-defined in GLSL.std.450.
    assert f"buf[0].v[2] = min({nclamp}, fmax({a}, {b}));" in crossgl
    assert f"buf[0].v[3] = fmax({a}, {b});" in crossgl
    assert "clamp(" not in crossgl


def test_nclamp_by_numeric_instruction_id_is_imported_the_same_way(tmp_path):
    numeric = COMPUTE.replace(" NMin ", " 79 ").replace(" NMax ", " 80 ")
    numeric = numeric.replace(" NClamp ", " 81 ")
    assert _translate(tmp_path, "cgl", numeric) == _translate(tmp_path, "cgl")


def test_glsl_output_tests_for_nan_and_compiles(tmp_path):
    code = _translate(tmp_path, "opengl")
    assert "isnan(" in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(code, encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_metal_output_uses_fmin_and_compiles(tmp_path):
    code = _translate(tmp_path, "metal")
    assert "fmin(" in code and "fmax(" in code
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not on PATH")
    path = tmp_path / "shader.metal"
    path.write_text(code, encoding="utf-8")
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(path) + ".air"]
    )


def test_spirv_round_trip_guards_the_nan_operands_and_validates(tmp_path):
    assembly = _translate(tmp_path, "vulkan")
    # fmin/fmax are lowered through integer NaN tests, so FMin/FMax never see a NaN operand.
    assert "OpBitwiseAnd" in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = _write(tmp_path, assembly, "round_trip.spvasm")
    binary = tmp_path / "round_trip.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", path, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
