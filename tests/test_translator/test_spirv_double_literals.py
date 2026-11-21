"""SPIR-V output of unsuffixed floating literals in binary64 contexts.

An unsuffixed CrossGL floating literal denotes its exact decimal value, rounded once to the type its context
requires; ``0.1f`` is binary32 (``ast.FLOAT_LITERAL_SUFFIX``). CUDA ``x * 0.1`` with a ``double`` ``x`` multiplies by
the binary64 value nearest 0.1, so SPIR-V output must write ``OpConstant %double 0.1``: an ``OpFConvert`` of
``OpConstant %float 0.1`` multiplies by 0.100000001490116 instead.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import FLOAT_LITERAL_SUFFIX, LiteralNode
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def spirv_for(body: str, prelude: str = "") -> str:
    source = f"""
    shader DoubleLiterals {{
        {prelude}
        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer double* fin;
            buffer float* ffin;
            buffer double* outp;
            void main() {{
                uint i = gl_GlobalInvocationID.x;
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
        [spirv_as, "--target-env", "vulkan1.0", str(asm_path), "-o", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assemble.returncode == 0, assemble.stderr
    result = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.0", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def type_id(spv_code: str, declaration: str) -> str:
    match = re.search(rf"(%\d+) = {re.escape(declaration)}\n", spv_code)
    assert match is not None, declaration
    return match.group(1)


def constant_ids(spv_code: str, type_name: str, value: str) -> set:
    return set(
        re.findall(rf"(%\d+) = OpConstant {type_name} {re.escape(value)}\n", spv_code)
    )


def converted_ids(spv_code: str, type_name: str) -> set:
    """The operands of every OpFConvert to ``type_name``."""
    return set(re.findall(rf"= OpFConvert {type_name} (%\d+)\n", spv_code))


def assert_double_constant(spv_code: str, value: str) -> None:
    double = type_id(spv_code, "OpTypeFloat 64")
    float_ = type_id(spv_code, "OpTypeFloat 32")
    assert constant_ids(spv_code, double, value), value
    narrow = constant_ids(spv_code, float_, value)
    assert not narrow & converted_ids(spv_code, double), value


class TestParser:
    @pytest.mark.parametrize(
        "text, value, suffix",
        [
            ("0.1f", 0.1, "f"),
            ("1e-3F", 0.001, "f"),
            ("0x1.8p1f", 3.0, "f"),
            ("0.1", 0.1, None),
        ],
    )
    def test_suffix_annotation(self, text, value, suffix):
        literal = Parser(Lexer(text).tokens).parse_literal()
        assert isinstance(literal, LiteralNode)
        assert literal.value == value
        assert literal.get_annotation(FLOAT_LITERAL_SUFFIX) == suffix


class TestDoubleContexts:
    @pytest.mark.parametrize(
        "body, value",
        [
            ("outp[i] = fin[i] * 0.1;", "0.1"),
            ("double a = 0.2; outp[i] = a;", "0.2"),
            ("double a = fin[i]; a += 0.3; outp[i] = a;", "0.3"),
            ("dvec2 v = dvec2(0.4, fin[i]); outp[i] = v.x;", "0.4"),
            ("dvec2 v = dvec2(fin[i]) * 0.6; outp[i] = v.y;", "0.6"),
            ("outp[i] = fin[i] > 0.7 ? 0.8 : fin[i];", "0.8"),
            ("outp[i] = max(0.9, fin[i]);", "0.9"),
            ("outp[i] = mix(1.1, fin[i], 0.5);", "1.1"),
            ("outp[i] = fma(1.3, fin[i], fin[i]);", "1.3"),
        ],
    )
    def test_unsuffixed_literal_is_double_constant(self, body, value, tmp_path):
        spv = spirv_for(body)
        assert_double_constant(spv, value)
        validate(spv, tmp_path)

    def test_negated_literal(self, tmp_path):
        spv = spirv_for("double e = -0.7; outp[i] = e + fin[i];")
        assert_double_constant(spv, "-0.7")
        validate(spv, tmp_path)

    def test_parameter_and_return(self, tmp_path):
        prelude = """
        double scale(double x, double y) { return x * y; }
        double tenth() { return 0.1; }
        """
        spv = spirv_for("outp[i] = scale(fin[i], 0.3) + tenth();", prelude)
        assert_double_constant(spv, "0.3")
        assert_double_constant(spv, "0.1")
        validate(spv, tmp_path)

    def test_struct_member(self, tmp_path):
        prelude = "struct S { double a; float b; };"
        spv = spirv_for("S s = S(0.35, 0.45); outp[i] = s.a + s.b;", prelude)
        assert_double_constant(spv, "0.35")
        validate(spv, tmp_path)

    def test_suffixed_literal_stays_binary32(self, tmp_path):
        spv = spirv_for("outp[i] = fin[i] * 0.1f;")
        double = type_id(spv, "OpTypeFloat 64")
        float_ = type_id(spv, "OpTypeFloat 32")
        assert not constant_ids(spv, double, "0.1")
        assert constant_ids(spv, float_, "0.1") & converted_ids(spv, double)
        validate(spv, tmp_path)

    def test_binary32_context_keeps_float(self, tmp_path):
        spv = spirv_for("outp[i] = double(ffin[i] * 0.1);")
        double = type_id(spv, "OpTypeFloat 64")
        float_ = type_id(spv, "OpTypeFloat 32")
        assert constant_ids(spv, float_, "0.1")
        assert not constant_ids(spv, double, "0.1")
        assert re.search(rf"= OpFMul {float_} ", spv)
        validate(spv, tmp_path)


class TestBuiltinOperandWidths:
    """A built-in with ``float`` and ``double`` operands takes the ``double`` overload (GLSL 4.60 6.1)."""

    @pytest.mark.parametrize(
        "call, instruction",
        [
            ("max(0.9, fin[i])", "FMax"),
            ("min(ffin[i], fin[i])", "FMin"),
            ("clamp(0.7, fin[i], 2.0)", "FClamp"),
            ("mix(ffin[i], fin[i], 0.5)", "FMix"),
            ("fma(0.4, fin[i], 0.3)", "Fma"),
            ("step(0.2, fin[i])", "Step"),
            ("smoothstep(0.1, 2.0, fin[i])", "SmoothStep"),
            ("distance(0.3, fin[i])", "Distance"),
        ],
    )
    def test_double_overload(self, call, instruction, tmp_path):
        spv = spirv_for(f"outp[i] = {call};")
        double = type_id(spv, "OpTypeFloat 64")
        assert re.search(rf"= OpExtInst {double} %\d+ {instruction} ", spv), call
        float_ = type_id(spv, "OpTypeFloat 32")
        assert not re.search(rf"= OpFConvert {float_} ", spv), call
        validate(spv, tmp_path)

    def test_vector_operands(self, tmp_path):
        spv = spirv_for(
            "vec2 f = vec2(ffin[i], 0.5f); dvec2 d = dvec2(fin[i]);"
            " dvec2 r = mix(f, d, 0.5); outp[i] = r.x + dot(f, d);"
        )
        v2double = type_id(spv, f"OpTypeVector {type_id(spv, 'OpTypeFloat 64')} 2")
        assert re.search(rf"= OpExtInst {v2double} %\d+ FMix ", spv)
        validate(spv, tmp_path)

    def test_binary32_operands_stay_binary32(self, tmp_path):
        spv = spirv_for("outp[i] = double(max(ffin[i], 0.5));")
        float_ = type_id(spv, "OpTypeFloat 32")
        assert re.search(rf"= OpExtInst {float_} %\d+ FMax ", spv)
        validate(spv, tmp_path)


def test_cuda_double_product(tmp_path):
    source = tmp_path / "dlit.cu"
    source.write_text(
        'extern "C" __global__ void probe(double* fin, double* outp) {\n'
        "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    outp[i] = fin[i] * 0.1 + fin[i] * 0.2f;\n"
        "}\n"
    )
    spv = crosstl.translate(str(source), backend="vulkan", format_output=False)
    assert_double_constant(spv, "0.1")
    double = type_id(spv, "OpTypeFloat 64")
    float_ = type_id(spv, "OpTypeFloat 32")
    assert constant_ids(spv, float_, "0.2") & converted_ids(spv, double)
    validate(spv, tmp_path)
