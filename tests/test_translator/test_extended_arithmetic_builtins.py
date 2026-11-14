"""uaddCarry, usubBorrow, umulExtended and imulExtended (GLSL 4.60 section 8.8) for every target.

GLSL calls them natively, SPIR-V emits ``OpIAddCarry``, ``OpISubBorrow``, ``OpUMulExtended`` and ``OpSMulExtended``
and stores the struct members to the out operands, and HLSL and Metal call helpers. Execution is covered by the
``g_extended_arithmetic`` and ``s_extended_arithmetic`` conformance cases; these tests check the spelling each
target receives, the typed refusals and the native checkers when they are installed.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.SPIRV_codegen import (
    SPIRVBuiltinLoweringError,
    VulkanSPIRVCodeGen,
)

SOURCE = """shader Extended {
    RWStructuredBuffer<uint> values @register(u0);
    groupshared uint shared_carry;
    compute {
        @numthreads(64, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            uint carry;
            uint sum = uaddCarry(values[i], values[i + 1u], carry);
            uint borrow;
            uint difference = usubBorrow(values[i], 7u, borrow);
            uint high;
            uint low;
            umulExtended(values[i], values[i + 1u], high, low);
            int signedHigh;
            int signedLow;
            imulExtended(int(values[i]), -3, signedHigh, signedLow);
            uvec2 carries;
            uvec2 sums = uaddCarry(uvec2(values[i], 1u), uvec2(2u, values[i]), carries);
            ivec3 highs;
            ivec3 lows;
            imulExtended(ivec3(1, 2, 3), ivec3(int(values[i])), highs, lows);
            uaddCarry(values[i], 1u, values[i + 2u]);
            uaddCarry(values[i], 1u, shared_carry);
            values[i] = sum + carry + difference + borrow + high + low + uint(signedHigh)
                + uint(signedLow) + sums.y + carries.x + uint(highs.z + lows.y) + shared_carry;
        }
    }
}"""


def _check(command, tmp_path):
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    if "unable to find utility" in result.stderr:
        pytest.skip("xcrun has no Metal compiler")
    assert result.returncode == 0, result.stdout + result.stderr


def test_glsl_calls_the_built_ins_natively(tmp_path):
    code = GLSLCodeGen().generate(parse(SOURCE))
    assert re.search(r"uaddCarry\(values\[i\], values\[\(?i \+ 1u\)?\], carry\)", code)
    assert re.search(r"umulExtended\([^;]*, high, low\)", code)
    assert "crossgl_" not in code.replace("crossgl_signed", "")
    if shutil.which("glslangValidator"):
        (tmp_path / "unit.comp").write_text(code, encoding="utf-8")
        _check(["glslangValidator", "unit.comp"], tmp_path)


def test_hlsl_calls_helpers_without_64_bit_integers(tmp_path):
    code = HLSLCodeGen().generate(parse(SOURCE))
    for helper in (
        "__crossgl_uaddCarry_uint(",
        "__crossgl_uaddCarry_uint2(",
        "__crossgl_usubBorrow_uint(",
        "__crossgl_umulExtended_uint(",
        "__crossgl_imulExtended_int(",
        "__crossgl_imulExtended_int3(",
    ):
        assert helper in code
    # The signed product corrects the unsigned one, so that helper is declared first.
    assert code.index("void __crossgl_umulExtended_uint3(") < code.index(
        "void __crossgl_imulExtended_int3("
    )
    assert "uint64_t" not in code
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "unit.hlsl").write_text(code, encoding="utf-8")
        entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code).group(1)
        _check(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-E",
                entry,
                str(tmp_path / "unit.hlsl"),
                "-Fo",
                str(tmp_path / "unit.dxil"),
            ],
            tmp_path,
        )


def test_metal_overloads_the_out_operands_for_each_address_space(tmp_path):
    code = MetalCodeGen().generate(parse(SOURCE))
    assert "mulhi(x, y)" in code
    for space in ("thread", "device", "threadgroup"):
        assert re.search(
            rf"__crossgl_uaddCarry_uint\(uint x, uint y,\s*{space} uint ?& ?carry\)",
            code,
        )
    if shutil.which("xcrun"):
        (tmp_path / "unit.metal").write_text(code, encoding="utf-8")
        _check(["xcrun", "-sdk", "macosx", "metal", "-c", "unit.metal"], tmp_path)


def test_spirv_emits_the_struct_instructions(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(SOURCE))
    for opcode in ("OpIAddCarry", "OpISubBorrow", "OpUMulExtended", "OpSMulExtended"):
        assert re.search(rf"= {opcode} %\w+ %\w+ %\w+", assembly), opcode
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "unit.spvasm").write_text(assembly, encoding="utf-8")
        _check(
            ["spirv-as", "--target-env", "vulkan1.1", "unit.spvasm", "-o", "unit.spv"],
            tmp_path,
        )
        _check(["spirv-val", "--target-env", "vulkan1.1", "unit.spv"], tmp_path)


@pytest.mark.parametrize(
    "replacement, reason",
    [
        (
            "umulExtended(values[i], values[i + 1u], high, low)",
            "umulExtended(values[i], values[i + 1u])",
        ),
        ("usubBorrow(values[i], 7u, borrow)", "usubBorrow(values[i], 7u)"),
        (
            "uaddCarry(values[i], values[i + 1u], carry)",
            "uaddCarry(float(values[i]), 1.0, carry)",
        ),
        (
            "usubBorrow(values[i], 7u, borrow)",
            "usubBorrow(uvec2(values[i]), 7u, borrow)",
        ),
    ],
)
@pytest.mark.parametrize(
    "codegen, refusal",
    [
        (GLSLCodeGen, OpenGLLoweringRefused),
        (HLSLCodeGen, DirectXLoweringRefused),
        (MetalCodeGen, MetalLoweringRefused),
        (VulkanSPIRVCodeGen, SPIRVBuiltinLoweringError),
    ],
)
def test_operand_counts_and_types_the_built_ins_do_not_take_are_refused(
    replacement, reason, codegen, refusal
):
    original, changed = replacement, reason
    source = SOURCE.replace(original, changed, 1)
    with pytest.raises(refusal):
        codegen().generate(parse(source))


def test_a_user_function_named_uaddCarry_is_called_unchanged():
    source = SOURCE.replace(
        "    compute {",
        "    uint uaddCarry(uint x, uint y) { return x; }\n    compute {",
        1,
    )
    body_start = source.index("void main")
    body = source[body_start:]
    body = re.sub(r"\n\s*uvec2 sums = [^\n]*", "", body)
    body = re.sub(r"\n\s*uaddCarry\(values\[i\], 1u, [^\n]*", "", body)
    body = body.replace(
        "uaddCarry(values[i], values[i + 1u], carry)", "uaddCarry(values[i], 1u)"
    )
    body = body.replace(" + sums.y + carries.x", "")
    source = source[:body_start] + body
    for codegen in (GLSLCodeGen, HLSLCodeGen, MetalCodeGen):
        code = codegen().generate(parse(source))
        assert "__crossgl_uaddCarry" not in code
        assert re.search(r"uaddCarry\(values\[i\], 1u\)", code)
