"""Malformed inputs found by the importer crash fuzzer (tests/conformance/fuzz) are rejected or refused.

Each case is a minimal, hand-reduced form of one finding. Before the fixes these inputs made an importer loop
forever, raise an internal ``IndexError``/``AttributeError``/``KeyError``, or exhaust memory.
"""

from __future__ import annotations

import textwrap
import threading
from decimal import Decimal, localcontext

import pytest

import crosstl
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Mojo.MojoParser import MojoParser
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser
from crosstl.backend.slang.float_literals import check_float_literal_range
from crosstl.backend.slang.SlangLexer import SlangLexer
from crosstl.backend.slang.SlangParser import SlangParser
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV.VulkanParser import VulkanParser

SECONDS = 20


def finishes(function):
    """Run ``function`` in a daemon thread; return the exception it raised, failing if it does not finish."""
    outcome = {}

    def run():
        try:
            function()
        except (
            BaseException
        ) as error:  # noqa: BLE001 - the test inspects what was raised
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(SECONDS)
    assert not worker.is_alive(), f"did not finish within {SECONDS} s"
    return outcome.get("error")


def tokens(lexer_class, code):
    lexer = lexer_class(code)
    return lexer.tokenize() if hasattr(lexer, "tokenize") else lexer.get_tokens()


PARSERS = {
    "cuda": (CudaLexer, CudaParser),
    "directx": (HLSLLexer, HLSLParser),
    "opengl": (GLSLLexer, GLSLParser),
    "metal": (MetalLexer, MetalParser),
    "mojo": (MojoLexer, MojoParser),
    "rust": (RustLexer, RustParser),
    "slang": (SlangLexer, SlangParser),
    "vulkan": (VulkanLexer, VulkanParser),
}


@pytest.mark.parametrize("language", sorted(PARSERS))
def test_eat_never_consumes_the_end_of_input(language):
    lexer_class, parser_class = PARSERS[language]
    parser = parser_class(tokens(lexer_class, ""))
    assert parser.current_token[0] == "EOF"
    with pytest.raises(SyntaxError, match="Unexpected end of input"):
        parser.eat("EOF")
    assert parser.current_token[0] == "EOF"


@pytest.mark.parametrize(
    "language, code",
    [
        # A template argument list cut off by the end of the file.
        ("cuda", "__global__ void hazard(const int *inp, int <outp) { }"),
        # A parameter list cut off inside its qualifiers.
        ("slang", "void main(in"),
    ],
)
def test_constructs_cut_off_by_the_end_of_input_are_rejected(language, code):
    lexer_class, parser_class = PARSERS[language]
    error = finishes(lambda: parser_class(tokens(lexer_class, code)).parse())
    assert isinstance(error, SyntaxError), repr(error)


def test_complete_programs_still_parse_to_the_end():
    for language, code in [
        ("slang", "void main(in float x) { }"),
        ("metal", "kernel void k(device int* a [[buffer(0)]]) { a[0] = 1; }"),
        ("mojo", "fn main():\n    var x = 1\n"),
        ("cuda", "__global__ void k(int* a) { a[0] = 1; }"),
    ]:
        lexer_class, parser_class = PARSERS[language]
        parser_class(tokens(lexer_class, code)).parse()


def test_directx_stray_hash_after_tokens_is_rejected(tmp_path):
    source = tmp_path / "stray.hlsl"
    source.write_text(
        "RWStructuredBuffer<int> a; #\n"
        "[numthreads(1, 1, 1)] void main() { a[0] = 1; }\n"
    )
    with pytest.raises(SyntaxError, match="Stray '#'"):
        crosstl.translate(str(source), backend="cgl", format_output=False)


def test_directx_null_directive_on_its_own_line_is_accepted(tmp_path):
    source = tmp_path / "null.hlsl"
    source.write_text(
        "#\nRWStructuredBuffer<int> a;\n"
        "[numthreads(1, 1, 1)] void main() { a[0] = 1; }\n"
    )
    assert "main" in crosstl.translate(str(source), backend="cgl", format_output=False)


@pytest.mark.parametrize("literal", ["5", '"text"'])
def test_mojo_transfer_of_a_literal_is_rejected(literal):
    # mojo build: "expression does not designate a value with a lifetime".
    code = f"fn main():\n    var x = {literal}^\n"
    with pytest.raises(SyntaxError, match="needs a value with a lifetime"):
        MojoParser(MojoLexer(code).tokenize()).parse()


def test_mojo_transfer_of_a_variable_still_parses():
    MojoParser(
        MojoLexer("fn main():\n    var x = 1\n    var y = x^\n").tokenize()
    ).parse()


def _slang_import(tmp_path, expression):
    source = tmp_path / "literal.slang"
    source.write_text(
        "RWStructuredBuffer<float> o;\n"
        f"[numthreads(1, 1, 1)] void main() {{ o[0] = {expression}; }}\n"
    )
    return lambda: crosstl.translate(str(source), backend="cgl", format_output=False)


# slangc 2025: "float literal '1.0e400' unrepresentable, converted to 'inf'" (E40009) and "'1.0e-400' is smaller
# than the smallest representable value for type float, converted to '0'" (E40010). An unsuffixed literal is a
# float even where a double is expected.
@pytest.mark.parametrize(
    "literal",
    [
        "1.0e400",
        "1.0e-400",
        "1e39f",
        "-3.4028236e38",
        "7e-46",
        "65520.0h",
        "1.0000000116860974e4294967296",
        "1e-4294967296",
    ],
)
def test_slang_unrepresentable_float_literals_are_refused(tmp_path, literal):
    error = finishes(_slang_import(tmp_path, literal))
    assert getattr(error, "project_diagnostic_code", None) == (
        "project.translate.slang-float-literal-unrepresentable"
    ), repr(error)


@pytest.mark.parametrize(
    "literal, digits",
    [("3.4028234e38", "340282340000000000000000000000000000000.0"), ("1e-45", "1")],
)
def test_slang_float_literals_at_the_limits_are_kept(tmp_path, literal, digits):
    assert digits in _slang_import(tmp_path, literal)()


def test_slang_float_literal_limits_are_exact():
    # 2**-150 is half the smallest float subnormal and rounds to zero (ties to even); anything above it does not.
    with localcontext() as context:
        context.prec = 200
        half_subnormal = Decimal(2) ** -150
        just_above = half_subnormal * (1 + Decimal(10) ** -100)
        overflow = Decimal(2**128 - 2**103)
        just_below = overflow - 1
    with pytest.raises(ValueError):
        check_float_literal_range("x", half_subnormal, "")
    check_float_literal_range("x", just_above, "")
    with pytest.raises(ValueError):
        check_float_literal_range("x", overflow, "")
    check_float_literal_range("x", just_below, "")


SPIRV_MODULE = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
{types}
%main = OpFunction %void None %fn
%entry = OpLabel
{body}
OpFunctionEnd
"""


def _spirv_import(tmp_path, body, types=""):
    source = tmp_path / "module.spvasm"
    source.write_text(SPIRV_MODULE.format(types=types, body=textwrap.dedent(body)))
    return lambda: crosstl.translate(
        str(source), backend="cgl", source_backend="vulkan", format_output=False
    )


INVALID_SPIRV = {
    # spirv-as: "Expected operand for OpBranch instruction, but found the next instruction instead."
    "branch-without-target": ("OpBranch\n%next = OpLabel\nOpReturn", ""),
    "switch-without-operands": (
        "OpSelectionMerge %next None\nOpSwitch\n%next = OpLabel\nOpReturn",
        "",
    ),
    "branch-with-extra-operand": (
        "OpBranch %next %next\n%next = OpLabel\nOpReturn",
        "",
    ),
    # spirv-val: "'Target Label' operands for OpBranch must be the ID of an OpLabel instruction".
    "branch-to-a-constant": ("OpBranch %true", ""),
    # spirv-val: "Block(s) ... are referenced but not defined in function".
    "merge-to-a-constant": (
        "OpSelectionMerge %true None\nOpBranchConditional %true %a %a\n%a = OpLabel\nOpReturn",
        "",
    ),
    # spirv-as: "Operand ... requires a previous definition".
    "undefined-pointee-type": (
        "%v = OpVariable %ptr Function\nOpReturn",
        "%ptr = OpTypePointer Function %float",
    ),
}


@pytest.mark.parametrize("case", sorted(INVALID_SPIRV))
def test_invalid_spirv_modules_are_refused(tmp_path, case):
    body, types = INVALID_SPIRV[case]
    error = finishes(_spirv_import(tmp_path, body, types))
    assert getattr(error, "project_diagnostic_code", None) == (
        "project.translate.spirv-invalid-module"
    ), repr(error)


def test_spirv_result_id_without_opcode_is_rejected(tmp_path):
    # spirv-as: "Invalid Opcode prefix '%true'".
    error = finishes(_spirv_import(tmp_path, "OpBranch %a\n%a = %true\nOpReturn"))
    assert isinstance(error, SyntaxError) and "after '='" in str(error), repr(error)


def test_valid_spirv_branches_with_weights_and_switch_still_import(tmp_path):
    body = """\
    OpSelectionMerge %merge None
    OpBranchConditional %true %then %merge 3 1
    %then = OpLabel
    OpBranch %merge
    %merge = OpLabel
    OpSelectionMerge %end None
    OpSwitch %zero %end 1 %one
    %one = OpLabel
    OpBranch %end
    %end = OpLabel
    OpReturn
    """
    output = _spirv_import(
        tmp_path, body, "%int = OpTypeInt 32 1\n%zero = OpConstant %int 0"
    )()
    assert "switch" in output
