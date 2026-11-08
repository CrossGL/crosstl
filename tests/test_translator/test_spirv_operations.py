"""SPIR-V operations the importer keeps as calls are refused, and texel-pointer atomics import as image atomics.

The SPIR-V importer writes an instruction CrossGL has no operation for as a call to a reserved ``spirv*`` name
(``spirvImageSparseSample``, ``spirvImageTexelPointer``). No emitter defines those names, so every target used to
print the call verbatim. Atomics through ``OpImageTexelPointer`` now become CrossGL's ``imageAtomic*`` calls, storage
images import with CrossGL's image types (``uimage2D``, not HLSL's ``RWTexture2D<uint>``, which the Metal and WGSL
emitters do not read), and any reserved call that is left is refused with a typed diagnostic. WGSL, which has no
texture atomics, refuses ``imageAtomic*`` instead of printing it.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.image_texel_atomics import fold_image_texel_atomic
from crosstl.backend.SPIRV.VulkanAst import FunctionCallNode
from crosstl.translator.codegen.wgsl_image_atomics import (
    WGSLImageAtomicUnsupportedError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import SpirvOperationUnsupportedError
from crosstl.translator.semantics.spirv_instructions import (
    is_spirv_operation_name,
    refuse_spirv_instructions,
)

# glslang 15 output, debug names trimmed, for:
#   layout(r32ui) uniform uimage2D counts; layout(r32i) uniform iimage2D levels;
#   results.u[0] = imageAtomicAdd(counts, p, 3u); results.u[1] = imageAtomicCompSwap(counts, p, 7u, 9u);
#   results.i[0] = imageAtomicMin(levels, p, -2); results.i[1] = imageAtomicExchange(levels, p, 5);
ATOMICS = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 1 1 1
OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
OpName %Results "Results"
OpMemberName %Results 0 "u"
OpMemberName %Results 1 "i"
OpName %results "results"
OpName %counts "counts"
OpName %levels "levels"
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
OpDecorate %_arr_uint_uint_2 ArrayStride 4
OpDecorate %_arr_int_uint_2 ArrayStride 4
OpDecorate %Results BufferBlock
OpMemberDecorate %Results 0 Offset 0
OpMemberDecorate %Results 1 Offset 8
OpDecorate %results Binding 2
OpDecorate %results DescriptorSet 0
OpDecorate %counts Binding 0
OpDecorate %counts DescriptorSet 0
OpDecorate %levels Binding 1
OpDecorate %levels DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%_ptr_Function_v2int = OpTypePointer Function %v2int
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%v2uint = OpTypeVector %uint 2
%uint_2 = OpConstant %uint 2
%_arr_uint_uint_2 = OpTypeArray %uint %uint_2
%_arr_int_uint_2 = OpTypeArray %int %uint_2
%Results = OpTypeStruct %_arr_uint_uint_2 %_arr_int_uint_2
%_ptr_Uniform_Results = OpTypePointer Uniform %Results
%results = OpVariable %_ptr_Uniform_Results Uniform
%int_0 = OpConstant %int 0
%25 = OpTypeImage %uint 2D 0 0 0 2 R32ui
%_ptr_UniformConstant_25 = OpTypePointer UniformConstant %25
%counts = OpVariable %_ptr_UniformConstant_25 UniformConstant
%uint_3 = OpConstant %uint 3
%uint_0 = OpConstant %uint 0
%_ptr_Image_uint = OpTypePointer Image %uint
%uint_1 = OpConstant %uint 1
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%int_1 = OpConstant %int 1
%uint_7 = OpConstant %uint 7
%uint_9 = OpConstant %uint 9
%44 = OpTypeImage %int 2D 0 0 0 2 R32i
%_ptr_UniformConstant_44 = OpTypePointer UniformConstant %44
%levels = OpVariable %_ptr_UniformConstant_44 UniformConstant
%int_n2 = OpConstant %int -2
%_ptr_Image_int = OpTypePointer Image %int
%_ptr_Uniform_int = OpTypePointer Uniform %int
%int_5 = OpConstant %int 5
%main = OpFunction %void None %3
%5 = OpLabel
%p = OpVariable %_ptr_Function_v2int Function
%15 = OpLoad %v3uint %gl_GlobalInvocationID
%16 = OpVectorShuffle %v2uint %15 %15 0 1
%17 = OpBitcast %v2int %16
OpStore %p %17
%28 = OpLoad %v2int %p
%32 = OpImageTexelPointer %_ptr_Image_uint %counts %28 %uint_0
%34 = OpAtomicIAdd %uint %32 %uint_1 %uint_0 %uint_3
%36 = OpAccessChain %_ptr_Uniform_uint %results %int_0 %int_0
OpStore %36 %34
%38 = OpLoad %v2int %p
%41 = OpImageTexelPointer %_ptr_Image_uint %counts %38 %uint_0
%42 = OpAtomicCompareExchange %uint %41 %uint_1 %uint_0 %uint_0 %uint_9 %uint_7
%43 = OpAccessChain %_ptr_Uniform_uint %results %int_0 %int_1
OpStore %43 %42
%47 = OpLoad %v2int %p
%50 = OpImageTexelPointer %_ptr_Image_int %levels %47 %uint_0
%51 = OpAtomicSMin %int %50 %uint_1 %uint_0 %int_n2
%53 = OpAccessChain %_ptr_Uniform_int %results %int_1 %int_0
OpStore %53 %51
%54 = OpLoad %v2int %p
%56 = OpImageTexelPointer %_ptr_Image_int %levels %54 %uint_0
%57 = OpAtomicExchange %int %56 %uint_1 %uint_0 %int_5
%58 = OpAccessChain %_ptr_Uniform_int %results %int_1 %int_1
OpStore %58 %57
OpReturn
OpFunctionEnd
"""

STORAGE_IMAGES = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
OpExecutionMode %main LocalSize 1 1 1
OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
OpName %rw "rw"
OpName %outImg "outImg"
OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
OpDecorate %rw Binding 1
OpDecorate %rw DescriptorSet 0
OpDecorate %outImg NonReadable
OpDecorate %outImg Binding 0
OpDecorate %outImg DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%v2int = OpTypeVector %int 2
%_ptr_Function_v2int = OpTypePointer Function %v2int
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%v2uint = OpTypeVector %uint 2
%float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
%21 = OpTypeImage %float 2D 0 0 0 2 R32f
%_ptr_UniformConstant_21 = OpTypePointer UniformConstant %21
%rw = OpVariable %_ptr_UniformConstant_21 UniformConstant
%v4float = OpTypeVector %float 4
%uint_0 = OpConstant %uint 0
%30 = OpTypeImage %float 2D 0 0 0 2 Rgba8
%_ptr_UniformConstant_30 = OpTypePointer UniformConstant %30
%outImg = OpVariable %_ptr_UniformConstant_30 UniformConstant
%float_2 = OpConstant %float 2
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %3
%5 = OpLabel
%p = OpVariable %_ptr_Function_v2int Function
%v = OpVariable %_ptr_Function_float Function
%15 = OpLoad %v3uint %gl_GlobalInvocationID
%16 = OpVectorShuffle %v2uint %15 %15 0 1
%17 = OpBitcast %v2int %16
OpStore %p %17
%24 = OpLoad %21 %rw
%25 = OpLoad %v2int %p
%27 = OpImageRead %v4float %24 %25
%29 = OpCompositeExtract %float %27 0
OpStore %v %29
%33 = OpLoad %30 %outImg
%34 = OpLoad %v2int %p
%35 = OpLoad %float %v
%36 = OpCompositeConstruct %v4float %35 %35 %35 %35
OpImageWrite %33 %34 %36
%37 = OpLoad %21 %rw
%38 = OpLoad %v2int %p
%39 = OpLoad %float %v
%41 = OpFMul %float %39 %float_2
%42 = OpCompositeConstruct %v4float %41 %41 %41 %41
OpImageWrite %37 %38 %42
OpReturn
OpFunctionEnd
"""

SPARSE_SAMPLE = """OpCapability Shader
OpCapability SparseResidency
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %c
OpExecutionMode %main OriginUpperLeft
OpName %tex "tex"
OpName %uv "uv"
OpName %ResType "ResType"
OpName %c "c"
OpDecorate %tex Binding 0
OpDecorate %tex DescriptorSet 0
OpDecorate %uv Location 0
OpDecorate %c Location 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%float = OpTypeFloat 32
%10 = OpTypeImage %float 2D 0 0 0 1 Unknown
%11 = OpTypeSampledImage %10
%_ptr_UniformConstant_11 = OpTypePointer UniformConstant %11
%tex = OpVariable %_ptr_UniformConstant_11 UniformConstant
%v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%uv = OpVariable %_ptr_Input_v2float Input
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%ResType = OpTypeStruct %int %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%c = OpVariable %_ptr_Output_v4float Output
%bool = OpTypeBool
%float_0 = OpConstant %float 0
%33 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%v4bool = OpTypeVector %bool 4
%main = OpFunction %void None %3
%5 = OpLabel
%code = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %11 %tex
%18 = OpLoad %v2float %uv
%23 = OpImageSparseSampleImplicitLod %ResType %14 %18
%24 = OpCompositeExtract %v4float %23 1
OpStore %t %24
%25 = OpCompositeExtract %int %23 0
OpStore %code %25
%28 = OpLoad %int %code
%30 = OpImageSparseTexelsResident %bool %28
%31 = OpLoad %v4float %t
%35 = OpCompositeConstruct %v4bool %30 %30 %30 %30
%36 = OpSelect %v4float %35 %31 %33
OpStore %c %36
OpReturn
OpFunctionEnd
"""

# OpAtomicUMin on a signed texel: no image atomic compares a signed texel as unsigned.
UNSIGNED_MIN_ON_SIGNED = ATOMICS.replace("%51 = OpAtomicSMin", "%51 = OpAtomicUMin")
# OpAtomicISub on an unsigned texel: imageAtomicAdd with the negated value.
SUBTRACT = ATOMICS.replace("%34 = OpAtomicIAdd", "%34 = OpAtomicISub")

CROSSGL_IMAGE_ATOMIC = """
shader S {
    uimage2D counts @set(0) @binding(0) @r32ui;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            imageAtomicAdd(counts, int2(0, 0), 1u);
        }
    }
}
"""

RESERVED_SPELLING_FUNCTION = """
shader S {
    float spirvScale(float value) {
        return value * 0.5;
    }
    fragment {
        float4 color @output @location(0);
        void main() {
            color = float4(spirvScale(1.0), 0.0, 0.0, 1.0);
        }
    }
}
"""

COMPILERS = {
    "directx": "dxc",
    "opengl": "glslangValidator",
    "metal": "xcrun",
    "vulkan": "spirv-val",
    "wgsl": "naga",
}


def _translate(tmp_path, source, backend, suffix=".spvasm"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile(tmp_path, code, backend, stage="comp"):
    """Compile the output with the native tool when it is installed."""
    if shutil.which(COMPILERS[backend]) is None:
        return
    if backend == "directx":
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)
    elif backend == "opengl":
        (tmp_path / "out.glsl").write_text(code)
        _run(["glslangValidator", "-S", stage, "out.glsl"], tmp_path)
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


def test_texel_pointer_atomics_import_as_image_atomics(tmp_path):
    lines = _lines(_translate(tmp_path, ATOMICS, "cgl"))
    assert "uimage2D counts @set(0) @binding(0) @r32ui;" in lines
    assert "iimage2D levels @set(0) @binding(1) @r32i;" in lines
    assert "results[0].u[0] = imageAtomicAdd(counts, p, 3u);" in lines
    # OpAtomicCompareExchange takes Value before Comparator; imageAtomicCompSwap takes the comparator first.
    assert "results[0].u[1] = imageAtomicCompSwap(counts, p, 7u, 9u);" in lines
    assert "results[0].i[0] = imageAtomicMin(levels, p, -2);" in lines
    assert "results[0].i[1] = imageAtomicExchange(levels, p, 5);" in lines
    assert not any("spirv" in line for line in lines)


def test_subtraction_through_a_texel_pointer_adds_the_negated_unsigned_value(tmp_path):
    lines = _lines(_translate(tmp_path, SUBTRACT, "cgl"))
    assert "results[0].u[0] = imageAtomicAdd(counts, p, -3u);" in lines


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [
        ("directx", "InterlockedCompareExchange"),
        ("opengl", "imageAtomicCompSwap(counts"),
        ("metal", "atomic_compare_exchange_weak"),
        ("vulkan", "OpAtomicCompareExchange"),
    ],
)
@pytest.mark.parametrize("source", [ATOMICS, SUBTRACT], ids=["add", "subtract"])
def test_image_atomics_from_spirv_compile_for_each_target(
    tmp_path, backend, spelling, source
):
    code = _translate(tmp_path, source, backend)
    assert spelling in code
    assert "spirvImageTexelPointer" not in code
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "metal", "vulkan", "wgsl", "cuda"]
)
def test_unsigned_minimum_on_a_signed_texel_is_refused(tmp_path, backend):
    with pytest.raises(SpirvOperationUnsupportedError) as error:
        _translate(tmp_path, UNSIGNED_MIN_ON_SIGNED, backend)
    assert error.value.operation == "spirvImageTexelPointer"
    assert error.value.code.endswith(f"{backend}-spirv-operation-unsupported")


def test_crossgl_keeps_an_operation_it_cannot_spell(tmp_path):
    lines = _lines(_translate(tmp_path, UNSIGNED_MIN_ON_SIGNED, "cgl"))
    assert (
        "results[0].i[0] = atomicMin(spirvImageTexelPointer(levels, p, 0), -2);"
        in lines
    )


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal", "vulkan"])
def test_sparse_residency_sample_is_refused(tmp_path, backend):
    with pytest.raises(SpirvOperationUnsupportedError) as error:
        _translate(tmp_path, SPARSE_SAMPLE, backend)
    assert error.value.operation.startswith("spirvSparse")


def test_crossgl_keeps_the_sparse_residency_sample(tmp_path):
    code = _translate(tmp_path, SPARSE_SAMPLE, "cgl")
    assert "spirvImageSparseSample(tex, uv)" in code


def test_storage_images_import_with_crossgl_image_types(tmp_path):
    lines = _lines(_translate(tmp_path, STORAGE_IMAGES, "cgl"))
    assert "image2D rw @set(0) @binding(1) @r32f;" in lines
    assert "image2D outImg @set(0) @binding(0) @rgba8 @writeonly;" in lines
    assert not any("RWTexture" in line for line in lines)


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [
        ("directx", "RWTexture2D<float> rw"),
        ("opengl", "imageStore(rw"),
        ("metal", "texture2d<float, access::read_write> rw"),
        ("vulkan", "OpImageWrite"),
        ("wgsl", "texture_storage_2d<r32float, read_write>"),
    ],
)
def test_storage_image_reads_and_writes_from_spirv_compile_for_each_target(
    tmp_path, backend, spelling
):
    code = _translate(tmp_path, STORAGE_IMAGES, backend)
    assert spelling in code
    _compile(tmp_path, code, backend)


def test_wgsl_refuses_image_atomics(tmp_path):
    with pytest.raises(WGSLImageAtomicUnsupportedError) as error:
        _translate(tmp_path, CROSSGL_IMAGE_ATOMIC, "wgsl", suffix=".cgl")
    assert error.value.function_name == "imageAtomicAdd"
    assert error.value.missing_capabilities == ("wgsl.storage-texture-atomics",)


def test_wgsl_refuses_image_atomics_imported_from_spirv(tmp_path):
    with pytest.raises(WGSLImageAtomicUnsupportedError):
        _translate(tmp_path, ATOMICS, "wgsl")


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [
        ("directx", "spirvScale("),
        ("opengl", "spirvScale("),
        ("metal", "spirvScale("),
        ("vulkan", "OpFunctionCall"),
    ],
)
def test_a_program_function_with_a_reserved_spelling_is_not_an_operation(
    tmp_path, backend, spelling
):
    code = _translate(tmp_path, RESERVED_SPELLING_FUNCTION, backend, suffix=".cgl")
    assert spelling in code


def test_refusal_names_the_first_undeclared_operation():
    program = Parser(Lexer(RESERVED_SPELLING_FUNCTION).get_tokens()).parse()
    refuse_spirv_instructions(program, "directx")
    program = Parser(
        Lexer(
            RESERVED_SPELLING_FUNCTION.replace("spirvScale(1.0)", "spirvSDot(1, 2)")
        ).get_tokens()
    ).parse()
    refuse_spirv_instructions(program, "cgl")
    with pytest.raises(SpirvOperationUnsupportedError) as error:
        refuse_spirv_instructions(program, "directx")
    assert error.value.operation == "spirvSDot"


@pytest.mark.parametrize(
    ("name", "reserved"),
    [
        ("spirvSDot", True),
        ("spirvImageSparseSample", True),
        ("spirv_GLSL_std_450_Foo", True),
        ("spirv", False),
        ("spirvlower", False),
        ("mySpirvSDot", False),
        (None, False),
    ],
)
def test_reserved_operation_names(name, reserved):
    assert is_spirv_operation_name(name) is reserved


def test_texel_pointer_with_a_sample_index_is_not_folded():
    pointer = FunctionCallNode("spirvImageTexelPointer", ["img", "p", "2"])
    call = FunctionCallNode("atomicAdd", [pointer, "1"])
    assert fold_image_texel_atomic(call, "OpAtomicIAdd", "uint") is call


def test_atomics_on_other_pointers_are_not_folded():
    call = FunctionCallNode("atomicAdd", ["counter", "1"])
    assert fold_image_texel_atomic(call, "OpAtomicIAdd", "uint") is call
    pointer = FunctionCallNode("spirvImageTexelPointer", ["img", "p", "0"])
    load = FunctionCallNode("atomicLoad", [pointer])
    assert fold_image_texel_atomic(load, "OpAtomicLoad", "uint") is load
