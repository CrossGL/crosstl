"""SPIR-V lowering of source-language arithmetic rules that SPIR-V leaves to the producer.

Each test pins the opcode that implements the source rule, not merely an opcode that validates:

- ``%`` truncates toward zero in C, C++, MSL and HLSL, so a signed remainder takes the
  dividend's sign (OpSRem, not the floored OpSMod); GLSL ``mod`` is floored (OpFMod);
- a signed ``>>`` sign-extends (OpShiftRightArithmetic);
- mixed signed/unsigned operands follow the usual arithmetic conversions (``int < uint``
  compares unsigned; ``i64 < uint`` stays signed 64-bit);
- ``isnan``/``isinf``/``isfinite``, ``fmin``/``fmax`` and the sign of a zero ``fmod``
  result are exact even where Vulkan lets drivers assume no NaN, Inf or signed zero.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def spirv_for(body: str, prelude: str = "") -> str:
    source = f"""
    shader ArithmeticSemantics {{
        {prelude}
        compute {{
            void main() {{
                {body}
            }}
        }}
    }}
    """
    return VulkanSPIRVCodeGen().generate(Parser(Lexer(source).tokens).parse())


def validate(spv_code: str, tmp_path) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    assemble = subprocess.run(
        [spirv_as, str(asm_path), "-o", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assemble.returncode == 0, assemble.stderr
    result = subprocess.run(
        [spirv_val, str(spv_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def type_id(spv_code: str, declaration: str) -> str:
    match = re.search(rf"(%\d+) = {re.escape(declaration)}\n", spv_code)
    assert match is not None, declaration
    return match.group(1)


def opcodes(spv_code: str):
    return re.findall(r"= (Op\w+)", spv_code)


def assert_no_fallback(spv_code: str) -> None:
    assert "WARNING" not in spv_code


class TestRemainder:
    def test_signed_remainder_truncates(self, tmp_path):
        spv = spirv_for("int a = -7; int b = 2; int r = a % b;")
        int_type = type_id(spv, "OpTypeInt 32 1")
        assert re.search(rf"= OpSRem {int_type} ", spv)
        assert "OpSMod" not in spv
        assert_no_fallback(spv)
        validate(spv, tmp_path)

    def test_unsigned_remainder(self, tmp_path):
        spv = spirv_for("uint a = 7u; uint b = 2u; uint r = a % b;")
        assert "OpUMod" in opcodes(spv)
        assert "OpSRem" not in spv
        validate(spv, tmp_path)

    def test_signed_vector_remainder_and_compound_assignment(self, tmp_path):
        spv = spirv_for(
            "ivec2 a = ivec2(-7, 7); ivec2 b = ivec2(2, -2); ivec2 r = a % b;"
            " int c = -9; c %= 4;"
        )
        assert opcodes(spv).count("OpSRem") == 2
        assert "OpSMod" not in spv
        validate(spv, tmp_path)

    def test_fmod_truncates_and_keeps_the_dividend_sign(self, tmp_path):
        spv = spirv_for("float x = -4.0; float y = 2.0; float r = fmod(x, y);")
        ops = opcodes(spv)
        assert "OpFRem" in ops
        assert "OpFMod" not in ops
        # The dividend's sign bit (0x80000000) is copied onto |OpFRem| so that
        # fmod(-0.0, y) and fmod(-4.0, 2.0) are -0.0 even without float controls.
        uint_type = type_id(spv, "OpTypeInt 32 0")
        assert re.search(rf"OpConstant {uint_type} 2147483648\b", spv)
        assert re.search(rf"OpConstant {uint_type} 2147483647\b", spv)
        assert "OpBitwiseOr" in ops
        validate(spv, tmp_path)

    def test_float_percent_operator_is_truncated(self, tmp_path):
        spv = spirv_for("vec3 x = vec3(-5.5); vec3 r = x % 2.0;")
        assert "OpFRem" in opcodes(spv)
        assert "OpFMod" not in opcodes(spv)
        validate(spv, tmp_path)

    def test_glsl_mod_is_floored(self, tmp_path):
        spv = spirv_for(
            "float x = -5.5; float r = mod(x, 2.0); vec2 v = mod(vec2(x), 3.0);"
        )
        ops = opcodes(spv)
        assert ops.count("OpFMod") == 2
        assert "OpFRem" not in ops
        validate(spv, tmp_path)

    def test_double_fmod_copies_sign_through_the_high_word(self, tmp_path):
        spv = spirv_for("double x = -4.0; double y = 2.0; double r = fmod(x, y);")
        ops = opcodes(spv)
        assert "OpFRem" in ops
        uint_type = type_id(spv, "OpTypeInt 32 0")
        uvec2 = type_id(spv, f"OpTypeVector {uint_type} 2")
        assert re.search(rf"= OpBitcast {uvec2} ", spv)
        assert "Int64" not in spv
        validate(spv, tmp_path)

    def test_specialization_constant_remainder_truncates(self):
        generator = VulkanSPIRVCodeGen()
        assert generator.specialization_constant_binary_opcode("%", "int") == "SRem"
        assert generator.specialization_constant_binary_opcode("%", "uint") == "UMod"


class TestShiftRight:
    def test_signed_shift_right_is_arithmetic(self, tmp_path):
        spv = spirv_for(
            "int a = -8; int r = a >> 1; ivec2 v = ivec2(-8, 8) >> 1; int c = -8; c >>= 2;"
        )
        ops = opcodes(spv)
        assert ops.count("OpShiftRightArithmetic") == 3
        assert "OpShiftRightLogical" not in ops
        validate(spv, tmp_path)

    def test_unsigned_shift_right_is_logical(self, tmp_path):
        spv = spirv_for(
            "uint a = 4294967288u; uint r = a >> 1u; uvec2 v = uvec2(a) >> 1u;"
        )
        ops = opcodes(spv)
        assert ops.count("OpShiftRightLogical") == 2
        assert "OpShiftRightArithmetic" not in ops
        validate(spv, tmp_path)


class TestUsualArithmeticConversions:
    @pytest.mark.parametrize(
        "types, expected",
        [
            (["int", "uint"], "uint"),
            (["uint", "int"], "uint"),
            (["int", "int"], "int"),
            (["i64", "uint"], "i64"),
            (["uint", "i64"], "i64"),
            (["u64", "int"], "u64"),
            (["i64", "u64"], "u64"),
            (["int", "i64"], "i64"),
            (["bool", "uint"], "uint"),
            (["int", "uint", "float"], "float"),
        ],
    )
    def test_promoted_type(self, types, expected):
        assert VulkanSPIRVCodeGen().promoted_numeric_type_name(types) == expected

    @pytest.mark.parametrize(
        "types, expected",
        [(["int", "uint"], "uint"), (["i64", "uint"], "i64"), (["u64", "int"], "u64")],
    )
    def test_promoted_bitwise_type(self, types, expected):
        generator = VulkanSPIRVCodeGen()
        assert generator.promoted_bitwise_integer_type_name(types) == expected

    def test_int_less_than_uint_compares_unsigned(self, tmp_path):
        spv = spirv_for("int x = -1; uint n = 4u; bool r = x < n; bool s = n > x;")
        ops = opcodes(spv)
        assert "OpULessThan" in ops
        assert "OpUGreaterThan" in ops
        assert "OpSLessThan" not in ops
        assert "OpSGreaterThan" not in ops
        validate(spv, tmp_path)

    def test_signed_comparison_stays_signed(self, tmp_path):
        spv = spirv_for("int x = -1; int n = 4; bool r = x < n;")
        assert "OpSLessThan" in opcodes(spv)
        validate(spv, tmp_path)

    def test_mixed_division_is_unsigned(self, tmp_path):
        spv = spirv_for("int x = -8; uint n = 2u; uint q = x / n;")
        ops = opcodes(spv)
        assert "OpUDiv" in ops
        assert "OpSDiv" not in ops
        validate(spv, tmp_path)


class TestFloatClassification:
    def test_isnan_isinf_isfinite_use_integer_bit_tests(self, tmp_path):
        spv = spirv_for(
            "float x = 1.0; bool a = isnan(x); bool b = isinf(x); bool c = isfinite(x);"
            " bvec3 d = isnan(vec3(x));"
        )
        ops = opcodes(spv)
        assert not {"OpIsNan", "OpIsInf", "OpOrdered", "OpUnordered"} & set(ops)
        uint_type = type_id(spv, "OpTypeInt 32 0")
        # 0x7F800000 is the binary32 exponent field and +Inf's magnitude.
        assert re.search(rf"OpConstant {uint_type} 2139095040\b", spv)
        assert ops.count("OpUGreaterThan") == 2  # scalar and vector isnan
        assert "OpIEqual" in ops  # isinf
        assert "OpINotEqual" in ops  # isfinite
        assert_no_fallback(spv)
        validate(spv, tmp_path)

    def test_double_classification_reads_both_words(self, tmp_path):
        spv = spirv_for("double x = 1.0; bool a = isnan(x); bool b = isinf(x);")
        uint_type = type_id(spv, "OpTypeInt 32 0")
        assert re.search(rf"OpConstant {uint_type} 2146435072\b", spv)  # 0x7FF00000
        assert "Int64" not in spv
        assert_no_fallback(spv)
        validate(spv, tmp_path)

    @pytest.mark.parametrize("function, std450", [("fmin", "FMin"), ("fmax", "FMax")])
    def test_fmin_fmax_return_the_non_nan_operand(self, function, std450, tmp_path):
        spv = spirv_for(
            f"float x = 1.0; float y = 2.0; float r = {function}(x, y);"
            f" vec2 v = {function}(vec2(x), vec2(y));"
        )
        assert spv.count(f" {std450} ") == 2
        assert opcodes(spv).count("OpSelect") == 4
        assert "NMin" not in spv and "NMax" not in spv
        assert_no_fallback(spv)
        validate(spv, tmp_path)


@pytest.mark.parametrize(
    "suffix, source",
    [
        (
            ".hlsl",
            "RWStructuredBuffer<float> o : register(u0);\n"
            "[numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
            "    o[id.x] = fmod(o[id.x], 2.0);\n}\n",
        ),
        (
            ".cu",
            "__global__ void k(float* o) { o[0] = fmodf(o[0], 2.0f) + fminf(o[0], o[1]); }\n",
        ),
        (
            ".hip",
            "__global__ void k(float* o) { o[0] = fmodf(o[0], 2.0f) + fmaxf(o[0], o[1]); }\n",
        ),
    ],
)
def test_importers_keep_c_fmod_and_nan_avoiding_min_max(suffix, source, tmp_path):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    crossgl = crosstl.translate(str(path), backend="cgl")
    assert "fmod(" in crossgl
    assert not re.search(r"(?<![\w.])mod\(", crossgl)
    assert not re.search(r"(?<![\w.])(min|max)\(", crossgl)
