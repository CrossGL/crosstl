"""Directive handling in the source front ends' preprocessors: comments that span lines, the null directive,
``%:`` directives, the ``_Pragma`` operator and ``__has_include``."""

from __future__ import annotations

import pytest

from crosstl.backend.CUDA.preprocessor import CudaPreprocessor
from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor
from crosstl.backend.HIP.preprocessor import HipPreprocessor
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.OpenCL.preprocessor import OpenCLPreprocessor
from crosstl.backend.slang.preprocessor import SlangPreprocessor
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.preprocessor.lexer import lex_text

ALL = (
    HLSLPreprocessor,
    MetalPreprocessor,
    CudaPreprocessor,
    HipPreprocessor,
    OpenCLPreprocessor,
    SlangPreprocessor,
)
# Clang-based compilers whose language has digraphs ([lex.digraph]).
DIGRAPHS = (MetalPreprocessor, CudaPreprocessor, HipPreprocessor, OpenCLPreprocessor)
# Compilers that search system header directories of their own after the include path.
SYSTEM_HEADERS = (CudaPreprocessor, HipPreprocessor, OpenCLPreprocessor)
BUILTIN_CODE = "project.translate.preprocessor-unsupported-builtin"


def tokens(text):
    return [token.text for token in lex_text(text, cpp=True) if token.text]


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_block_comment_spanning_lines_continues_the_directive(cls):
    text = "#define C2 1 /* multi\nline */ + 2\nint a = C2;\nint b = __LINE__;\n"
    lines = cls().preprocess(text).split("\n")
    assert tokens("\n".join(lines)) == tokens("int a = 1 + 2; int b = 4;")
    # The comment's line break is kept as a line, so later lines keep their numbers.
    assert sum(1 for line in lines if "int" in line) == 2


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_hash_in_a_block_comment_starts_no_directive(cls):
    text = "/* start\n#define X 1\n*/ int X;\n"
    assert tokens(cls().preprocess(text)) == ["int", "X", ";"]


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_null_directive_has_no_effect(cls):
    assert tokens(cls().preprocess("#\n# /* nothing */\nkept\n")) == ["kept"]


@pytest.mark.parametrize("cls", DIGRAPHS, ids=lambda cls: cls.__name__)
def test_percent_colon_introduces_a_directive_where_the_language_has_digraphs(cls):
    assert tokens(cls().preprocess("%:define D 3\nint d = D;\n")) == tokens(
        "int d = 3;"
    )


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_pragma_operator_becomes_a_pragma_line_of_its_own(cls):
    lines = cls().preprocess('a _Pragma("omp parallel") b\n').split("\n")
    assert [line.strip() for line in lines if line.strip()] == [
        "a",
        "#pragma omp parallel",
        "b",
    ]


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_pragma_operator_from_a_stringized_macro_argument(cls):
    text = '#define DO_PRAGMA(x) _Pragma(#x)\nDO_PRAGMA(message("hi")) after\n'
    lines = [line.strip() for line in cls().preprocess(text).split("\n")]
    assert [line for line in lines if line] == ['#pragma message("hi")', "after"]


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_malformed_pragma_operator_is_refused(cls):
    with pytest.raises(TranslationDiagnostic) as refused:
        cls().preprocess("_Pragma(1)\n")
    assert refused.value.code == "project.translate.preprocessor-invalid-directive"


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_pragma_operator_once_stops_a_second_inclusion(cls, tmp_path):
    (tmp_path / "once.h").write_text('_Pragma("once")\nint guarded;\n')
    main = tmp_path / "main.src"
    text = '#include "once.h"\n#include "once.h"\nint after;\n'
    main.write_text(text)
    output = cls().preprocess(text, file_path=str(main))
    assert tokens(output) == tokens("int guarded; int after;")


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_has_include_looks_headers_up_on_the_include_path(cls, tmp_path):
    (tmp_path / "present.h").write_text("int present;\n")
    text = (
        '#if __has_include("present.h") && __has_include(<present.h>)\nfound\n#endif\n'
        '#if __has_include("absent.h")\nwrong\n#else\nabsent\n#endif\n'
        "#ifdef __has_include\ndefined\n#endif\n"
    )
    output = cls(include_paths=[str(tmp_path)]).preprocess(text)
    assert tokens(output) == ["found", "absent", "defined"]


def test_has_include_next_searches_only_later_include_directories(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    probe = "#if __has_include_next(<shared.h>)\nnext\n#else\nlast\n#endif\n"
    (first / "shared.h").write_text(probe)
    (second / "shared.h").write_text(probe)
    preprocessor = HLSLPreprocessor(include_paths=[str(first), str(second)])
    assert tokens(
        preprocessor.preprocess(probe, file_path=str(first / "shared.h"))
    ) == ["next"]
    preprocessor = HLSLPreprocessor(include_paths=[str(first), str(second)])
    assert tokens(
        preprocessor.preprocess(probe, file_path=str(second / "shared.h"))
    ) == ["last"]


@pytest.mark.parametrize("cls", SYSTEM_HEADERS, ids=lambda cls: cls.__name__)
def test_system_header_probe_is_refused_where_the_compiler_has_its_own(cls):
    text = "#if __has_include(<no_such_header.h>)\nyes\n#else\nno\n#endif\n"
    with pytest.raises(TranslationDiagnostic) as refused:
        cls().preprocess(text)
    assert refused.value.code == BUILTIN_CODE
    # A strict preprocessor takes the include path as every header there is.
    assert tokens(cls(strict=True).preprocess(text)) == ["no"]


@pytest.mark.parametrize(
    "cls", (HLSLPreprocessor, SlangPreprocessor), ids=lambda cls: cls.__name__
)
def test_system_header_probe_is_absent_where_only_the_include_path_is_searched(cls):
    text = "#if __has_include(<no_such_header.h>)\nyes\n#else\nno\n#endif\n"
    assert tokens(cls().preprocess(text)) == ["no"]


@pytest.mark.parametrize("cls", ALL, ids=lambda cls: cls.__name__)
def test_has_include_outside_a_condition_is_refused(cls):
    with pytest.raises(TranslationDiagnostic) as refused:
        cls().preprocess('int x = __has_include("a.h");\n')
    assert refused.value.code == BUILTIN_CODE
