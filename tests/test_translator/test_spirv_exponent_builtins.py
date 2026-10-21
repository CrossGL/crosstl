"""frexp, modf and ldexp (GLSL 4.60 section 8.3) lowered to GLSL.std.450 for SPIR-V.

Execution against numpy's frexp, ldexp and modf is covered by the ``g_frexp_ldexp_modf`` conformance case; these
tests check the instructions SPIR-V receives, how the output operand is written, and the typed refusals.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.SPIRV_codegen import (
    SPIRVBuiltinLoweringError,
    VulkanSPIRVCodeGen,
)

EXPONENTS = """shader Exponents {
    RWStructuredBuffer<float> values @register(u0);
    RWStructuredBuffer<int> exponents @register(u1);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            int e;
            float m = frexp(values[i], e);
            ivec3 e3;
            vec3 m3 = frexp(vec3(values[i], 2.0, 3.0), e3);
            frexp(values[i + 1u], exponents[i]);
            float whole;
            float part = modf(values[i], whole);
            vec2 whole2;
            vec2 part2 = modf(vec2(values[i], 1.5), whole2);
            values[i] = m + m3.y + part + part2.x + whole + whole2.y
                + ldexp(values[i], e) + ldexp(m3, e3).z + ldexp(m3, e).x;
            exponents[i] = e + e3.z;
        }
    }
}"""

FLOAT_OUTPUTS = """shader FloatOutputs {
    RWStructuredBuffer<float> values @register(u0);
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            float e;
            float m = frexp(values[i], e);
            int whole;
            float part = modf(values[i], whole);
            values[i] = m + e + part + float(whole) + ldexp(values[i], e);
        }
    }
}"""


def _validate(assembly, tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    (tmp_path / "unit.spvasm").write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", "unit.spvasm", "-o", "unit.spv"],
        ["spirv-val", "--target-env", "vulkan1.1", "unit.spv"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr


def _instructions(assembly):
    return re.findall(r"OpExtInst %\w+ %\w+ (\w+)", assembly)


def _type_id(assembly, declaration):
    """The result id of a type declaration such as ``OpTypeInt 32 1``."""
    match = re.search(rf"(%\w+) = {re.escape(declaration)}\n", assembly)
    assert match is not None, declaration
    return match.group(1)


def _vector_id(assembly, component, count):
    return _type_id(assembly, f"OpTypeVector {component} {count}")


def test_frexp_and_modf_use_the_struct_returning_instructions(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(EXPONENTS))

    instructions = _instructions(assembly)
    assert instructions.count("FrexpStruct") == 3
    assert instructions.count("ModfStruct") == 2
    # The pointer forms would need the output operand's storage to match the instruction exactly.
    assert "Frexp " not in assembly and "Modf " not in assembly
    # GLSL.std.450 requires a 32-bit integer exponent member with the value's component count.
    float3 = _vector_id(assembly, _type_id(assembly, "OpTypeFloat 32"), 3)
    int3 = _vector_id(assembly, _type_id(assembly, "OpTypeInt 32 1"), 3)
    assert f"OpTypeStruct {float3} {int3}\n" in assembly
    _validate(assembly, tmp_path)


def test_frexp_writes_its_exponent_to_a_storage_buffer_element(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(EXPONENTS))

    frexp = re.findall(r"(%\w+) = OpExtInst %\w+ %\w+ FrexpStruct", assembly)[2]
    int_type = _type_id(assembly, "OpTypeInt 32 1")
    exponent = re.search(rf"(%\w+) = OpCompositeExtract {int_type} {frexp} 1", assembly)
    assert exponent is not None
    assert re.search(rf"OpStore %\w+ {exponent.group(1)}\b", assembly)
    _validate(assembly, tmp_path)


def test_ldexp_uses_ldexp_with_integer_exponents_and_splats_a_scalar_one(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(EXPONENTS))

    assert _instructions(assembly).count("Ldexp") == 3
    assert "Exp2" not in _instructions(assembly)
    _validate(assembly, tmp_path)


def test_hlsl_style_float_outputs_and_exponents_are_converted(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(FLOAT_OUTPUTS))

    # HLSL declares the frexp exponent with the value's type; the integer member is converted on the way out.
    float_type = _type_id(assembly, "OpTypeFloat 32")
    int_type = _type_id(assembly, "OpTypeInt 32 1")
    assert f"OpConvertSToF {float_type} " in assembly
    # modf's whole part is stored to an int through a float-to-int conversion.
    assert f"OpConvertFToS {int_type} " in assembly
    # A floating-point ldexp exponent means x * 2^e, as DXC lowers it.
    assert "Exp2" in _instructions(assembly)
    assert "Ldexp" not in _instructions(assembly)
    _validate(assembly, tmp_path)


@pytest.mark.parametrize(
    "replacement",
    [
        ("frexp(values[i], e)", "frexp(int(values[i]), e)"),
        ("frexp(values[i], e)", "frexp(values[i])"),
        ("ldexp(values[i], e)", "ldexp(int(values[i]), e)"),
        ("ldexp(values[i], e)", "ldexp(values[i], vec2(e))"),
    ],
)
def test_operands_without_a_glsl_std_450_form_are_refused(replacement):
    source = FLOAT_OUTPUTS.replace(*replacement, 1)
    with pytest.raises(SPIRVBuiltinLoweringError):
        VulkanSPIRVCodeGen().generate(parse(source))
