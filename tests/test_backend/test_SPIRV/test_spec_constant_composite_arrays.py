"""SPIR-V input declares a named ``OpSpecConstantComposite`` array (SPIR-V 1.6 section 3.52.7).

The importer reads a named specialization-constant composite by name. It used to declare only those whose type has a
CrossGL name, so a composite array was left out while the function body still read it, and the CrossGL it wrote used
an undeclared constant.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

HEADER = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 1 1 1
               OpName %S "S"
NAMES
               OpDecorate %S SpecId 4
               OpDecorate %OutBuf BufferBlock
               OpMemberDecorate %OutBuf 0 Offset 0
               OpDecorate %rt ArrayStride 4
               OpDecorate %outp DescriptorSet 0
               OpDecorate %outp Binding 0
               OpDecorate %gid BuiltIn GlobalInvocationId
       %void = OpTypeVoid
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
      %v2int = OpTypeVector %int 2
     %v3uint = OpTypeVector %uint 3
          %S = OpSpecConstant %int 5
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
     %arr_i3 = OpTypeArray %int %uint_3
     %arr_v2 = OpTypeArray %v2int %uint_2
     %arr_i2 = OpTypeArray %int %uint_2
    %arr_i22 = OpTypeArray %arr_i2 %uint_2
        %arr = OpSpecConstantComposite %arr_i3 %S %int_1 %int_0
         %va = OpSpecConstantComposite %v2int %S %int_1
         %vb = OpConstantComposite %v2int %int_0 %int_1
      %pairs = OpSpecConstantComposite %arr_v2 %va %vb
         %r0 = OpSpecConstantComposite %arr_i2 %S %int_0
         %r1 = OpConstantComposite %arr_i2 %int_1 %int_0
       %grid = OpSpecConstantComposite %arr_i22 %r0 %r1
         %rt = OpTypeRuntimeArray %int
     %OutBuf = OpTypeStruct %rt
 %ptr_OutBuf = OpTypePointer Uniform %OutBuf
     %ptr_in = OpTypePointer Input %v3uint
   %ptr_fn_a = OpTypePointer Function %arr_i3
   %ptr_fn_p = OpTypePointer Function %arr_v2
   %ptr_fn_g = OpTypePointer Function %arr_i22
  %ptr_fn_v2 = OpTypePointer Function %v2int
 %ptr_fn_int = OpTypePointer Function %int
  %ptr_u_int = OpTypePointer Uniform %int
       %outp = OpVariable %ptr_OutBuf Uniform
        %gid = OpVariable %ptr_in Input
         %fn = OpTypeFunction %void
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %la = OpVariable %ptr_fn_a Function
         %lp = OpVariable %ptr_fn_p Function
         %lg = OpVariable %ptr_fn_g Function
          %g = OpLoad %v3uint %gid
          %x = OpCompositeExtract %uint %g 0
          %k = OpUMod %uint %x %uint_2
               OpStore %la %arr
               OpStore %lp %pairs
               OpStore %lg %grid
         %ap = OpAccessChain %ptr_fn_int %la %k
         %av = OpLoad %int %ap
         %pp = OpAccessChain %ptr_fn_v2 %lp %k
         %pv = OpLoad %v2int %pp
         %p0 = OpCompositeExtract %int %pv 0
         %gp = OpAccessChain %ptr_fn_int %lg %k %k
         %gv = OpLoad %int %gp
         %s0 = OpIAdd %int %av %p0
         %s1 = OpIAdd %int %s0 %gv
          %o = OpAccessChain %ptr_u_int %outp %int_0 %x
               OpStore %o %s1
               OpReturn
               OpFunctionEnd
"""

NAMED = "\n".join(
    f'               OpName %{name} "{name}"' for name in ("arr", "pairs", "grid")
)

DECLARATIONS = (
    "const int arr[3] = {S, 1, 0};",
    "const int2 pairs[2] = {int2(S, 1), int2(0, 1)};",
    "const int grid[2][2] = {{S, 0}, {1, 0}};",
)


def _import(tmp_path, names):
    path = tmp_path / "composites.spvasm"
    path.write_text(HEADER.replace("NAMES\n", names + "\n" if names else ""))
    return crosstl.translate(str(path), backend="cgl")


def _round_trip(tmp_path, crossgl, backend):
    path = tmp_path / "composites.cgl"
    path.write_text(crossgl)
    return crosstl.translate(str(path), backend=backend)


def test_named_composite_arrays_are_declared_where_the_body_reads_them(tmp_path):
    crossgl = _import(tmp_path, NAMED)
    for declaration in DECLARATIONS:
        assert declaration in crossgl
    for name in ("arr", "pairs", "grid"):
        assert re.search(rf"\b\w+ = {name};", crossgl)


def test_unnamed_composite_arrays_stay_inline(tmp_path):
    crossgl = _import(tmp_path, "")
    assert not re.search(r"const int\w* \w+\[", crossgl)
    assert "= {S, 1, 0};" in crossgl


def _named_id(spirv, name):
    # Result ids are numeric unless SPIRV-Tools reformats the module, so follow OpName to the id.
    match = re.search(rf'OpName (%\w+) "{name}"', spirv)
    assert match, name
    return re.escape(match.group(1))


def test_spirv_output_keeps_the_arrays_specialized(tmp_path):
    spirv = _round_trip(tmp_path, _import(tmp_path, NAMED), "vulkan")
    for name in ("arr", "pairs", "grid"):
        assert re.search(rf"{_named_id(spirv, name)} = OpSpecConstantComposite ", spirv)
    assert re.search(rf"{_named_id(spirv, 'S')} = OpSpecConstant %\w+ 5\b", spirv)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "round_trip.spvasm"
        assembly.write_text(spirv)
        binary = tmp_path / "round_trip.spv"
        for command in (
            ["spirv-as", "--target-env", "vulkan1.0", str(assembly), "-o", str(binary)],
            ["spirv-val", "--target-env", "vulkan1.0", str(binary)],
        ):
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "backend, declaration",
    [
        ("metal", "constant int arr[3] = {S, 1, 0};"),
        ("directx", "static const int arr[3] = {S, 1, 0};"),
    ],
)
def test_metal_and_hlsl_output_declare_the_array(tmp_path, backend, declaration):
    assert declaration in _round_trip(tmp_path, _import(tmp_path, NAMED), backend)
