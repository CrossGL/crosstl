"""SPIR-V Private variables import as CrossGL static module-scope variables.

SPIR-V's Private storage class is visible to every function of a module but belongs to one invocation (SPIR-V 1.6
section 3.7). The importer wrote such a variable as a plain module-scope CrossGL variable, which the emitters read
as uniform data: HLSL made it a read-only global (DXC: "read-only variable is not assignable"), GLSL a ``uniform``
(glslang: "can't modify a uniform") and Metal refused the store to its constant address space. CrossGL's per-thread
module-scope variable is ``static``, which every emitter lowers to its private storage.
"""

import shutil
import subprocess

import pytest

import crosstl

# glslang 15 output, comments and the debug names of locals trimmed, for:
#   layout(std430, set = 0, binding = 0) buffer Out { float values[]; } outBuf;
#   struct Accum { float avg[3]; };  Accum accum;  float scale = 2.0;
#   float total() { float s = 0.0; for (int i = 0; i < 3; ++i) s += accum.avg[i]; return s * scale; }
#   void main() {
#       for (int i = 0; i < 3; ++i) accum.avg[i] = float(gl_GlobalInvocationID.x + uint(i));
#       scale = scale + 1.0;
#       outBuf.values[gl_GlobalInvocationID.x] = total();
#   }
PRIVATE = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 1 1 1
OpName %total_ "total("
OpName %scale "scale"
OpName %Accum "Accum"
OpMemberName %Accum 0 "avg"
OpName %accum "accum"
OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
OpName %Out "Out"
OpMemberName %Out 0 "values"
OpName %outBuf "outBuf"
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
OpDecorate %_runtimearr_float ArrayStride 4
OpDecorate %Out BufferBlock
OpMemberDecorate %Out 0 Offset 0
OpDecorate %outBuf Binding 0
OpDecorate %outBuf DescriptorSet 0
OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%7 = OpTypeFunction %float
%_ptr_Private_float = OpTypePointer Private %float
%scale = OpVariable %_ptr_Private_float Private
%float_2 = OpConstant %float 2
%_ptr_Function_float = OpTypePointer Function %float
%float_0 = OpConstant %float 0
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_0 = OpConstant %int 0
%int_3 = OpConstant %int 3
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%_arr_float_uint_3 = OpTypeArray %float %uint_3
%Accum = OpTypeStruct %_arr_float_uint_3
%_ptr_Private_Accum = OpTypePointer Private %Accum
%accum = OpVariable %_ptr_Private_Accum Private
%int_1 = OpConstant %int 1
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%uint_0 = OpConstant %uint 0
%_ptr_Input_uint = OpTypePointer Input %uint
%float_1 = OpConstant %float 1
%_runtimearr_float = OpTypeRuntimeArray %float
%Out = OpTypeStruct %_runtimearr_float
%_ptr_Uniform_Out = OpTypePointer Uniform %Out
%outBuf = OpVariable %_ptr_Uniform_Out Uniform
%_ptr_Uniform_float = OpTypePointer Uniform %float
%uint_1 = OpConstant %uint 1
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_1 %uint_1 %uint_1
%main = OpFunction %void None %3
%5 = OpLabel
%i_0 = OpVariable %_ptr_Function_int Function
OpStore %scale %float_2
OpStore %i_0 %int_0
OpBranch %49
%49 = OpLabel
OpLoopMerge %51 %52 None
OpBranch %53
%53 = OpLabel
%54 = OpLoad %int %i_0
%55 = OpSLessThan %bool %54 %int_3
OpBranchConditional %55 %50 %51
%50 = OpLabel
%56 = OpLoad %int %i_0
%62 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
%63 = OpLoad %uint %62
%64 = OpLoad %int %i_0
%65 = OpBitcast %uint %64
%66 = OpIAdd %uint %63 %65
%67 = OpConvertUToF %float %66
%68 = OpAccessChain %_ptr_Private_float %accum %int_0 %56
OpStore %68 %67
OpBranch %52
%52 = OpLabel
%69 = OpLoad %int %i_0
%70 = OpIAdd %int %69 %int_1
OpStore %i_0 %70
OpBranch %49
%51 = OpLabel
%71 = OpLoad %float %scale
%73 = OpFAdd %float %71 %float_1
OpStore %scale %73
%78 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
%79 = OpLoad %uint %78
%80 = OpFunctionCall %float %total_
%82 = OpAccessChain %_ptr_Uniform_float %outBuf %int_0 %79
OpStore %82 %80
OpReturn
OpFunctionEnd
%total_ = OpFunction %float None %7
%9 = OpLabel
%s = OpVariable %_ptr_Function_float Function
%i = OpVariable %_ptr_Function_int Function
OpStore %s %float_0
OpStore %i %int_0
OpBranch %20
%20 = OpLabel
OpLoopMerge %22 %23 None
OpBranch %24
%24 = OpLabel
%25 = OpLoad %int %i
%28 = OpSLessThan %bool %25 %int_3
OpBranchConditional %28 %21 %22
%21 = OpLabel
%35 = OpLoad %int %i
%36 = OpAccessChain %_ptr_Private_float %accum %int_0 %35
%37 = OpLoad %float %36
%38 = OpLoad %float %s
%39 = OpFAdd %float %38 %37
OpStore %s %39
OpBranch %23
%23 = OpLabel
%40 = OpLoad %int %i
%42 = OpIAdd %int %40 %int_1
OpStore %i %42
OpBranch %20
%22 = OpLabel
%43 = OpLoad %float %s
%44 = OpLoad %float %scale
%45 = OpFMul %float %43 %44
OpReturnValue %45
OpFunctionEnd
"""


# The same module with scale's value as the OpVariable initializer, not a store at the start of main.
INITIALIZED = PRIVATE.replace(
    "%scale = OpVariable %_ptr_Private_float Private\n%float_2 = OpConstant %float 2\n",
    "%float_2 = OpConstant %float 2\n%scale = OpVariable %_ptr_Private_float Private %float_2\n",
).replace("OpStore %scale %float_2\n", "")

TOOLS = {
    "directx": "dxc",
    "opengl": "glslangValidator",
    "metal": "xcrun",
    "vulkan": "spirv-val",
    "wgsl": "naga",
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.spvasm"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile(tmp_path, code, backend):
    """Compile the output with the native tool when it is installed."""
    if shutil.which(TOOLS[backend]) is None:
        return
    if backend == "directx":
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)
    elif backend == "opengl":
        (tmp_path / "out.comp").write_text(code)
        _run(["glslangValidator", "-S", "comp", "out.comp"], tmp_path)
    elif backend == "metal":
        (tmp_path / "out.metal").write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )
    elif backend == "vulkan" and shutil.which("spirv-as"):
        (tmp_path / "out.spvasm").write_text(code)
        _run(["spirv-as", "out.spvasm", "-o", "out.spv"], tmp_path)
        _run(["spirv-val", "out.spv"], tmp_path)
    elif backend == "wgsl":
        (tmp_path / "out.wgsl").write_text(code)
        _run(["naga", "out.wgsl"], tmp_path)


def _lines(code):
    return [line.strip() for line in code.splitlines()]


def test_private_variables_import_as_static(tmp_path):
    lines = _lines(_translate(tmp_path, PRIVATE, "cgl"))
    assert "static float scale;" in lines
    assert "static Accum accum;" in lines
    assert "scale = 2.0;" in lines


def test_private_initializer_stays_on_the_declaration(tmp_path):
    lines = _lines(_translate(tmp_path, INITIALIZED, "cgl"))
    assert "static float scale = 2.0;" in lines
    assert "scale = 2.0;" not in lines


@pytest.mark.parametrize(
    ("backend", "forbidden"),
    [
        ("directx", "\nfloat scale;"),
        ("opengl", "uniform float scale"),
        ("metal", "constant float scale"),
        ("vulkan", "UniformConstant %float"),
        ("wgsl", "var<uniform> scale"),
    ],
)
@pytest.mark.parametrize(
    "source", [PRIVATE, INITIALIZED], ids=["stored", "initialized"]
)
def test_written_private_variables_compile_for_each_target(
    tmp_path, backend, forbidden, source
):
    code = _translate(tmp_path, source, backend)
    assert forbidden not in code
    _compile(tmp_path, code, backend)


def test_hlsl_declares_private_variables_static(tmp_path):
    code = _translate(tmp_path, PRIVATE, "directx")
    assert "static float scale;" in _lines(code)
    assert "static Accum accum;" in _lines(code)
