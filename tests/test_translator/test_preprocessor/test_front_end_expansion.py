"""Macro replacement in the source front ends' preprocessors against the shared conforming preprocessor.

The HLSL, Metal, CUDA, HIP, OpenCL and Slang importers interpret directives line by line and replace macros
through the shared preprocessor's expander (``crosstl.backend.DirectX.macro_expansion``). On every macro case of
``cases.py`` (checked against Clang in ``test_differential.py``) each front end must give the tokens the shared
preprocessor gives in that compiler's dialect, or refuse with the same diagnostic code where the shared
preprocessor refuses.
"""

from __future__ import annotations

import pytest

from crosstl.backend.CUDA.preprocessor import CudaPreprocessor
from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor
from crosstl.backend.HIP.preprocessor import HipPreprocessor
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.OpenCL.preprocessor import OpenCLPreprocessor
from crosstl.backend.slang.preprocessor import SlangPreprocessor
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.preprocessor import preprocess
from crosstl.translator.preprocessor.lexer import lex_text

from .cases import CPP_CASES

FRONT_ENDS = {
    "hlsl": (HLSLPreprocessor, "hlsl"),
    "metal": (MetalPreprocessor, "msl"),
    "cuda": (CudaPreprocessor, "c++"),
    "hip": (HipPreprocessor, "c++"),
    "opencl": (OpenCLPreprocessor, "c++"),
    "slang": (SlangPreprocessor, "hlsl"),
}

MACRO_CASES = (
    "standard_example_3",
    "standard_example_4",
    "standard_example_5",
    "standard_example_7",
    "hash_hash",
    "va_opt",
    "gnu_comma_elision",
    "nested_arguments",
    "painted_blue",
    "deferred_invocation",
    "rescan_with_following_tokens",
    "stringize_spacing",
    "token_pasting",
    "object_vs_function_like",
    "avoid_accidental_pasting",
    "conditionals",
    "if_arithmetic",
    "redefinition_and_undef",
    "builtin_macros",
    "line_in_multiline_invocations",
    "pragma_operator_once",
    "pragmas",
    "digraphs_and_splices",
    "includes_absent",
)

NOT_COMPARABLE = {
    # The Metal importer also lowers C++ constructs while preprocessing, and refuses the call of a temporary
    # that this case produces; that is not a macro replacement difference.
    ("metal", "rescan_with_following_tokens"),
    # The reference dialect is C++; OpenCL C has no 'and'/'or' operator names or u8 character literals.
    ("opencl", "if_arithmetic"),
    # The Metal importer refuses '#pragma STDC FP_CONTRACT', whose lexical scope translation cannot keep.
    ("metal", "pragmas"),
    # The case leaves '# define NOT_A_DIRECTIVE 1' as text, which the Metal importer refuses as a directive
    # it cannot carry; Clang would hand the same text to the parser, which rejects it.
    ("metal", "digraphs_and_splices"),
    # These compilers search system header directories of their own, so the importers refuse a
    # __has_include(<header>) that the include path does not answer, where the reference takes it as absent.
    ("cuda", "includes_absent"),
    ("hip", "includes_absent"),
    ("opencl", "includes_absent"),
}


def tokens(text):
    return [token.text for token in lex_text(text, cpp=True) if token.text]


def source(name):
    case = CPP_CASES[name]
    return case[0] if isinstance(case, tuple) else case


@pytest.mark.parametrize("name", MACRO_CASES)
@pytest.mark.parametrize("front_end", sorted(FRONT_ENDS))
def test_front_end_replaces_macros_like_the_shared_preprocessor(front_end, name):
    if (front_end, name) in NOT_COMPARABLE:
        pytest.skip("not a macro replacement comparison for this front end")
    preprocessor_class, dialect = FRONT_ENDS[front_end]
    text = source(name)
    try:
        expected = tokens(preprocess(text, dialect).text)
    except TranslationDiagnostic as refusal:
        with pytest.raises(TranslationDiagnostic) as refused:
            preprocessor_class().preprocess(text)
        assert refused.value.code == refusal.code
        return
    assert tokens(preprocessor_class().preprocess(text)) == expected


def expand(text, preprocessor_class=HLSLPreprocessor):
    return tokens(preprocessor_class().preprocess(text))


def test_arguments_are_replaced_before_substitution():
    assert expand("#define F(x) ((x)+1)\nF(F(1))\n") == tokens("((((1)+1))+1)")


def test_pasting_through_an_indirection_pastes_replaced_arguments():
    text = (
        "#define CAT_I(a, b) a##b\n#define CAT(a, b) CAT_I(a, b)\n#define N 2\n"
        "CAT(v, N) CAT_I(v, N)\n"
    )
    assert expand(text, MetalPreprocessor) == ["v2", "vN"]


def test_self_referential_object_macro_in_an_argument_expands_once():
    text = "#define A A+1\n#define F(x) (x)\nF(A)\n"
    assert expand(text) == tokens("(A+1)")


def test_function_macro_name_at_the_end_of_a_replacement_takes_following_arguments():
    text = "#define F(x) [x]\n#define G F\nG(1)\n"
    assert expand(text, CudaPreprocessor) == tokens("[1]")


def test_painted_function_macro_name_stays_in_the_output():
    assert expand("#define F(x) x + F(x)\nF(1)\n") == tokens("1 + F(1)")


def test_gnu_comma_elision_and_named_variable_arguments():
    text = (
        "#define E(fmt, ...) f(fmt, ## __VA_ARGS__)\n"
        "#define N(fmt, rest...) g(fmt, ## rest)\n"
        "E(1) E(1, 2) N(3) N(3, 4, 5)\n"
    )
    assert expand(text) == tokens("f(1) f(1, 2) g(3) g(3, 4, 5)")


def test_va_opt_follows_the_compiler():
    text = "#define F(a, ...) f(a __VA_OPT__(,) __VA_ARGS__)\nF(1) F(1, 2)\n"
    for preprocessor_class in (MetalPreprocessor, CudaPreprocessor, HipPreprocessor):
        assert expand(text, preprocessor_class) == tokens("f(1 ) f(1 , 2)")
    # DXC is Clang 3.7, which has no __VA_OPT__: the name stays an identifier and DXC rejects the call.
    assert "__VA_OPT__" in expand(text, HLSLPreprocessor)


@pytest.mark.parametrize(
    "text",
    [
        "#define F(x) x\nF(1, 2)\n",
        "#define F(x, y) x\nF(1)\n",
        "#define F(x) x\nF(1\n",
        "#define P(a, b) a ## b\nP(+, -)\n",
    ],
)
def test_malformed_invocations_are_refused(text):
    with pytest.raises(TranslationDiagnostic) as refused:
        HLSLPreprocessor().preprocess(text)
    assert refused.value.code.startswith("project.translate.preprocessor-")


def test_condition_in_a_skipped_group_is_not_evaluated():
    text = "#if 0\n#if garbage (\n#endif\n#else\nkept\n#endif\n"
    assert expand(text) == ["kept"]


def test_if_expressions_use_the_conforming_evaluator():
    text = (
        "#define TWO 2\n"
        "#if TWO > 1 ? 1 : 1 / 0\nternary\n#endif\n"
        "#if -1 < 0u\n#else\nunsigned\n#endif\n"
        "#if defined(TWO) && !defined UNDEFINED\ndefined\n#endif\n"
    )
    assert expand(text) == ["ternary", "unsigned", "defined"]


def test_elifdef_follows_the_compiler():
    text = "#ifdef A\n#elifdef B\nb\n#else\nother\n#endif\n"
    preprocessor = MetalPreprocessor(defines={"B": "1"})
    assert tokens(preprocessor.preprocess(text)) == ["b"]


def test_multi_line_invocation_keeps_line_numbers_and_comments_end_with_their_line():
    text = "#define F(a, b) (a + b)\nint x = F(1, // one\n  2);\nint y = __LINE__;\n"
    lines = HLSLPreprocessor().preprocess(text).split("\n")
    first = next(index for index, line in enumerate(lines) if "int x" in line)
    assert tokens(lines[first]) == tokens("int x = (1 + 2);")
    assert lines[first + 1] == ""
    assert tokens(lines[first + 2]) == tokens("int y = 4;")
