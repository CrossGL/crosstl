"""SPIR-V input of workgroup sizes that the pipeline specializes, and of specialization constants without a name."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.workgroup_size_import import (
    SpirvWorkgroupSizeUnsupportedError,
)
from crosstl.translator.codegen.specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
)

# A compute entry that stores %value to outp[gl_GlobalInvocationID.x].
MODULE = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
MODES
OpName %main "main"
OpName %Out "Out"
OpMemberName %Out 0 "values"
OpName %outp "outp"
NAMES
OpDecorate %rt ArrayStride 4
OpDecorate %Out BufferBlock
OpMemberDecorate %Out 0 Offset 0
OpDecorate %outp DescriptorSet 0
OpDecorate %outp Binding 0
OpDecorate %gid BuiltIn GlobalInvocationId
DECORATIONS
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%v3uint = OpTypeVector %uint 3
%rt = OpTypeRuntimeArray %uint
%Out = OpTypeStruct %rt
%ptr_out = OpTypePointer Uniform %Out
%outp = OpVariable %ptr_out Uniform
%ptr_in = OpTypePointer Input %v3uint
%gid = OpVariable %ptr_in Input
%ptr_u = OpTypePointer Uniform %uint
%int_0 = OpConstant %int 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
CONSTANTS
%main = OpFunction %void None %fn
%entry = OpLabel
%g = OpLoad %v3uint %gid
%x = OpCompositeExtract %uint %g 0
BODY
%p = OpAccessChain %ptr_u %outp %int_0 %x
OpStore %p %value
OpReturn
OpFunctionEnd
"""

# glslang before SPIR-V 1.6, for layout(local_size_x = 64, local_size_x_id = 4) in; reading gl_WorkGroupSize.x.
GLSLANG_BUILTIN = {
    "MODES": "OpExecutionMode %main LocalSize 64 1 1",
    "DECORATIONS": "OpDecorate %21 SpecId 4\nOpDecorate %wgs BuiltIn WorkgroupSize",
    "CONSTANTS": (
        "%21 = OpSpecConstant %uint 64\n"
        "%wgs = OpSpecConstantComposite %v3uint %21 %uint_1 %uint_1"
    ),
    "BODY": "%value = OpCompositeExtract %uint %wgs 0",
}

# glslang for SPIR-V 1.6: LocalSizeId, and an undecorated composite of a second SpecId 4 constant.
GLSLANG_LOCAL_SIZE_ID = {
    "MODES": "OpExecutionModeId %main LocalSizeId %7 %uint_1 %uint_1",
    "DECORATIONS": "OpDecorate %7 SpecId 4\nOpDecorate %23 SpecId 4",
    "CONSTANTS": (
        "%7 = OpSpecConstant %uint 64\n%23 = OpSpecConstant %uint 64\n"
        "%24 = OpSpecConstantComposite %v3uint %23 %uint_1 %uint_1"
    ),
    "BODY": "%value = OpCompositeExtract %uint %24 0",
}


def _module(**parts):
    text = MODULE
    for key in ("MODES", "NAMES", "DECORATIONS", "CONSTANTS", "BODY"):
        text = text.replace(key, parts.get(key, ""))
    return text


def _translate(tmp_path, parts, backend="cgl"):
    path = tmp_path / "shader.spvasm"
    path.write_text(_module(**parts))
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def _validate_spirv(tmp_path, code):
    """Assemble and validate SPIR-V when SPIRV-Tools is installed."""
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _workgroup_size_composite(code):
    """(opcode, components) of the constant decorated BuiltIn WorkgroupSize."""
    name = re.search(r"OpDecorate %(\w+) BuiltIn WorkgroupSize", code).group(1)
    match = re.search(rf"%{name} = (Op\w+) %\w+ %(\w+) %(\w+) %(\w+)", code)
    return match.group(1), match.groups()[1:]


def _spec_constant(code, spec_id):
    """(name, default) of the constant decorated SpecId spec_id."""
    name = re.search(rf"OpDecorate %(\w+) SpecId {spec_id}\b", code).group(1)
    default = re.search(rf"%{name} = OpSpecConstant %\w+ (\d+)", code).group(1)
    return name, int(default)


def test_specialized_workgroup_size_builtin_keeps_its_specialization_id(tmp_path):
    code = _translate(tmp_path, GLSLANG_BUILTIN)

    assert (
        "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1, "
        "local_size_x_id = 4) in;" in code
    )
    # The read is the specialized built-in, not its default value, and nothing else declares the constant.
    assert "gl_WorkGroupSize[0]" in code
    assert "uint3(64, 1, 1)" not in code
    assert "@gl_WorkGroupSize" not in code
    assert "spec_constant_4" not in code


def test_specialized_workgroup_size_reaches_spirv_output(tmp_path):
    code = _translate(tmp_path, GLSLANG_BUILTIN, backend="vulkan")

    opcode, components = _workgroup_size_composite(code)
    name, default = _spec_constant(code, 4)
    assert opcode == "OpSpecConstantComposite"
    assert components[0] == name and default == 64
    builtin = re.search(r"OpDecorate %(\w+) BuiltIn WorkgroupSize", code).group(1)
    assert re.search(rf"OpCompositeExtract %\w+ %{builtin} 0", code)
    assert not re.search(rf"OpAccessChain %\w+ %{builtin}\b", code)
    _validate_spirv(tmp_path, code)


def test_specialized_workgroup_size_reaches_glsl_output(tmp_path):
    code = _translate(tmp_path, GLSLANG_BUILTIN, backend="opengl")

    assert "local_size_x_id = 4" in code
    if shutil.which("glslangValidator") and shutil.which("spirv-dis"):
        source, binary = tmp_path / "out.comp", tmp_path / "out.spv"
        source.write_text(code)
        _run(["glslangValidator", "-V", str(source), "-o", str(binary)])
        disassembly = _run(["spirv-dis", str(binary)])
        assert _workgroup_size_composite(disassembly)[0] == "OpSpecConstantComposite"
        assert _spec_constant(disassembly, 4)[1] == 64


def test_hlsl_output_refuses_the_specialized_workgroup_size(tmp_path):
    with pytest.raises(SpecializedWorkgroupSizeError) as error:
        _translate(tmp_path, GLSLANG_BUILTIN, backend="directx")
    assert error.value.reason == "target-has-no-specialized-workgroup-size"


def test_local_size_id_declares_one_constant_per_specialization_id(tmp_path):
    code = _translate(tmp_path, GLSLANG_LOCAL_SIZE_ID)

    assert (
        "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1, "
        "local_size_x_id = 4) in;" in code
    )
    assert code.count("@constant_id(4)") == 1
    assert "const uint spec_constant_4 @constant_id(4) = 64;" in code
    assert "uint3(spec_constant_4, 1, 1)" in code
    assert "uint3(64, 1, 1)" not in code


def test_named_local_size_id_operands_keep_their_declarations(tmp_path):
    parts = {
        "MODES": "OpExecutionModeId %main LocalSizeId %w %uint_2 %h",
        "NAMES": 'OpName %w "WIDTH"\nOpName %h "DEPTH"',
        "DECORATIONS": "OpDecorate %w SpecId 0\nOpDecorate %h SpecId 1",
        "CONSTANTS": "%w = OpSpecConstant %uint 8\n%h = OpSpecConstant %uint 4",
        "BODY": "%value = OpIAdd %uint %x %w",
    }
    code = _translate(tmp_path, parts)

    assert (
        "layout(local_size_x = 8, local_size_y = 2, local_size_z = 4, "
        "local_size_x_id = 0, local_size_z_id = 1) in;" in code
    )
    assert "const uint WIDTH @constant_id(0) = 8;" in code
    assert "const uint DEPTH @constant_id(1) = 4;" in code
    assert "(x + WIDTH)" in code or "+ WIDTH" in code


def test_unnamed_specialization_constant_is_read_by_name(tmp_path):
    # A module stripped of OpName: the constant was read as its default, 3, which fixed it.
    parts = {
        "MODES": "OpExecutionMode %main LocalSize 1 1 1",
        "DECORATIONS": "OpDecorate %s SpecId 2",
        "CONSTANTS": (
            "%s = OpSpecConstant %uint 3\n"
            "%sum = OpSpecConstantOp %uint IAdd %s %uint_1"
        ),
        "BODY": "%value = OpIAdd %uint %x %sum",
    }
    code = _translate(tmp_path, parts)

    assert "const uint spec_constant_2 @constant_id(2) = 3;" in code
    assert "(spec_constant_2 + 1)" in code
    assert "(3 + 1)" not in code


def test_unnamed_constants_of_one_id_with_different_defaults_get_their_own_names(
    tmp_path,
):
    parts = {
        "MODES": "OpExecutionMode %main LocalSize 1 1 1",
        "DECORATIONS": "OpDecorate %a SpecId 2\nOpDecorate %b SpecId 2",
        "CONSTANTS": "%a = OpSpecConstant %uint 3\n%b = OpSpecConstant %uint 5",
        "BODY": "%ab = OpIAdd %uint %a %b\n%value = OpIAdd %uint %x %ab",
    }
    code = _translate(tmp_path, parts)

    assert "const uint spec_constant_2 @constant_id(2) = 3;" in code
    assert "const uint spec_constant_2_1 @constant_id(2) = 5;" in code
    assert "(spec_constant_2 + spec_constant_2_1)" in code


def test_unread_unnamed_specialization_constant_stays_declared(tmp_path):
    parts = {
        "MODES": "OpExecutionMode %main LocalSize 1 1 1",
        "DECORATIONS": "OpDecorate %s SpecId 9",
        "CONSTANTS": "%s = OpSpecConstant %uint 3",
        "BODY": "%value = OpIAdd %uint %x %uint_1",
    }
    assert "const uint spec_constant_9 @constant_id(9) = 3;" in _translate(
        tmp_path, parts
    )


def test_workgroup_size_builtin_of_constants_overrides_local_size(tmp_path):
    parts = {
        "MODES": "OpExecutionMode %main LocalSize 1 1 1",
        "DECORATIONS": "OpDecorate %wgs BuiltIn WorkgroupSize",
        "CONSTANTS": "%wgs = OpConstantComposite %v3uint %uint_8 %uint_1 %uint_2",
        "BODY": "%value = OpCompositeExtract %uint %wgs 0",
    }
    code = _translate(tmp_path, parts)

    assert "layout(local_size_x = 8, local_size_y = 1, local_size_z = 2) in;" in code
    assert "_id" not in code.split("layout(local_size_x")[1].split(")")[0]


def test_specialization_op_of_the_builtin_reads_its_component(tmp_path):
    parts = dict(GLSLANG_BUILTIN)
    parts["CONSTANTS"] += "\n%wx = OpSpecConstantOp %uint CompositeExtract %wgs 0"
    parts["BODY"] = "%value = OpIMul %uint %x %wx"
    code = _translate(tmp_path, parts)

    assert "gl_WorkGroupSize.x" in code
    assert "64" not in code.split("void main")[1]


@pytest.mark.parametrize(
    "modes, constants",
    [
        (
            "OpExecutionModeId %main LocalSizeId %d %uint_1 %uint_1",
            "%s = OpSpecConstant %uint 4\n%d = OpSpecConstantOp %uint IMul %s %uint_2",
        ),
        (
            "OpExecutionMode %main LocalSize 8 1 1",
            "%s = OpSpecConstant %uint 4\n%d = OpSpecConstantOp %uint IMul %s %uint_2\n"
            "%wgs = OpSpecConstantComposite %v3uint %d %uint_1 %uint_1",
        ),
    ],
    ids=["local-size-id", "workgroup-size-builtin"],
)
def test_derived_workgroup_dimension_is_refused(tmp_path, modes, constants):
    decorations = "OpDecorate %s SpecId 3"
    if "%wgs" in constants:
        decorations += "\nOpDecorate %wgs BuiltIn WorkgroupSize"
    parts = {
        "MODES": modes,
        "DECORATIONS": decorations,
        "CONSTANTS": constants,
        "BODY": "%value = OpIAdd %uint %x %uint_1",
    }
    with pytest.raises(SpirvWorkgroupSizeUnsupportedError) as error:
        _translate(tmp_path, parts)
    assert error.value.axis == "x"
    assert error.value.code == "project.translate.spirv-workgroup-size-unsupported"
    assert "OpSpecConstantOp" in str(error.value)
