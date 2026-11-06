"""Built-in macros and invocations that continue on later lines in the source front ends' preprocessors.

The expected values are Clang's (``clang -E``): ``__LINE__`` is the physical line where the outermost invocation
ends, line splices count as lines, ``__COUNTER__``, ``__INCLUDE_LEVEL__``, ``__BASE_FILE__`` and ``__FILE_NAME__``
exist where the compiler has them, the build-time macros are refused, and a function-like macro name at the end
of a line (written there or produced by a replacement) is invoked by a ``(`` on a following line.
"""

from __future__ import annotations

import pytest

from crosstl.backend.CUDA.preprocessor import CudaPreprocessor
from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor
from crosstl.backend.GLSL.preprocessor import GLSLPreprocessor
from crosstl.backend.HIP.preprocessor import HipPreprocessor
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.OpenCL.preprocessor import OpenCLPreprocessor
from crosstl.backend.slang.preprocessor import SlangPreprocessor
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.preprocessor.engine import BUILTIN_CODE
from crosstl.translator.preprocessor.lexer import lex_text
from crosstl.translator.preprocessor.macros import ARGUMENTS_CODE

C_FAMILY = (
    HLSLPreprocessor,
    MetalPreprocessor,
    CudaPreprocessor,
    HipPreprocessor,
    OpenCLPreprocessor,
    SlangPreprocessor,
)


def tokens(text):
    return [token.text for token in lex_text(text, cpp=True) if token.text]


def expand(text, preprocessor_class=HLSLPreprocessor, file_path=None):
    return tokens(preprocessor_class().preprocess(text, file_path=file_path))


@pytest.mark.parametrize("preprocessor_class", C_FAMILY)
def test_line_splices_count_as_physical_lines(preprocessor_class):
    text = "#define LONG 1 + \\\n2\nA \\\nB\n__LINE__\n"
    assert expand(text, preprocessor_class) == ["A", "B", "5"]


@pytest.mark.parametrize("preprocessor_class", C_FAMILY)
def test_line_is_where_the_outermost_invocation_ends(preprocessor_class):
    # clang -E: [5 7], the argument's own line and the line of the closing parenthesis.
    text = "#define F(x) [x __LINE__]\nA \\\nB\n__LINE__\nF(__LINE__ \\\n\n)\n"
    assert expand(text, preprocessor_class) == tokens("A B 4 [5 7]")


@pytest.mark.parametrize("preprocessor_class", C_FAMILY)
def test_function_macro_name_at_a_line_end_takes_a_parenthesis_on_a_later_line(
    preprocessor_class,
):
    # clang -E: [14 14]; blank and comment-only lines between the name and '(' are whitespace.
    text = (
        "#define F(x) [x __LINE__]\n"
        + "\n" * 7
        + "#define G F\nG\n\n// c\n(\n__LINE__)\n"
    )
    assert expand(text, preprocessor_class) == tokens("[14 14]")


def test_function_macro_name_not_followed_by_a_parenthesis_stays():
    text = "#define F(x) [x]\nF\nnext (1)\nF\n#define Q 1\n(2)\n"
    assert expand(text) == tokens("F next (1) F (2)")


def test_name_painted_by_its_own_expansion_does_not_take_a_later_parenthesis():
    text = "#define F(x) x F\nF(1)\n(2)\n"
    assert expand(text) == tokens("1 F (2)")


def test_object_like_replacement_opening_an_invocation_takes_later_lines():
    text = "#define F(x, y) [x y]\n#define H F(\nH 1,\n2\n) after\n"
    assert expand(text, CudaPreprocessor) == tokens("[1 2] after")


def test_invocation_interrupted_by_a_directive_is_refused():
    text = "#define F(x) [x]\n#define H F(\nH 1\n#define Z\n)\n"
    with pytest.raises(TranslationDiagnostic) as refused:
        HLSLPreprocessor().preprocess(text)
    assert refused.value.code == ARGUMENTS_CODE


@pytest.mark.parametrize("preprocessor_class", C_FAMILY)
def test_counter_and_include_level(preprocessor_class, tmp_path):
    header = tmp_path / "level.h"
    header.write_text("in __INCLUDE_LEVEL__ __COUNTER__\n")
    main = tmp_path / "main.src"
    text = '__COUNTER__ __INCLUDE_LEVEL__\n#include "level.h"\nout __COUNTER__\n'
    main.write_text(text)
    preprocessor = preprocessor_class()
    output = tokens(preprocessor.preprocess(text, file_path=str(main)))
    assert output == tokens("0 0 in 1 1 out 2")
    # Each translation unit counts from 0.
    assert tokens(preprocessor.preprocess("__COUNTER__\n")) == ["0"]


def test_base_file_names_the_main_file_in_an_included_file(tmp_path):
    (tmp_path / "base.h").write_text("__BASE_FILE__ __FILE_NAME__\n")
    main = tmp_path / "main.metal"
    output = expand('#include "base.h"\n', MetalPreprocessor, file_path=str(main))
    # The path is a string literal, so a Windows path has its backslashes escaped, as Clang spells it.
    literal = str(main).replace("\\", "\\\\").replace('"', '\\"')
    assert output == [f'"{literal}"', '"base.h"']


def test_file_name_exists_only_where_the_compiler_has_it():
    text = "#ifdef __FILE_NAME__\nyes\n#else\nno\n#endif\n"
    assert expand(text, MetalPreprocessor) == ["yes"]
    # DXC is Clang 3.7, before __FILE_NAME__ (Clang 9).
    assert expand(text, HLSLPreprocessor) == ["no"]
    assert expand("__FILE_NAME__\n", HLSLPreprocessor) == ["__FILE_NAME__"]


@pytest.mark.parametrize("preprocessor_class", C_FAMILY)
def test_builtin_macros_are_defined(preprocessor_class):
    text = (
        "#ifdef __LINE__\nline\n#endif\n"
        "#if defined(__COUNTER__) && defined __INCLUDE_LEVEL__ && defined(__DATE__)\nclang\n#endif\n"
        "#ifndef __BASE_FILE__\nmissing\n#endif\n"
    )
    assert expand(text, preprocessor_class) == ["line", "clang"]


@pytest.mark.parametrize("name", ["__DATE__", "__TIME__", "__TIMESTAMP__"])
@pytest.mark.parametrize("preprocessor_class", C_FAMILY)
def test_build_time_macros_are_refused(preprocessor_class, name):
    with pytest.raises(TranslationDiagnostic) as refused:
        preprocessor_class().preprocess(f"const char* t = {name};\n")
    assert refused.value.code == BUILTIN_CODE


def test_a_source_definition_takes_precedence_over_a_builtin_macro():
    # Clang warns (-Wbuiltin-macro-redefined) and uses the new definition.
    assert expand("#define __LINE__ 5\na __LINE__\n", CudaPreprocessor) == ["a", "5"]


def test_glsl_has_no_clang_builtin_macros():
    text = "#version 450\nint a = __COUNTER__ + __LINE__;\n"
    assert tokens(GLSLPreprocessor().preprocess(text)) == tokens(
        "#version 450\nint a = __COUNTER__ + 2;"
    )
