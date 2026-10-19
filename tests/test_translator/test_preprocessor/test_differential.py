"""Differential tests: the shared preprocessor against Clang (C++20) and glslang on the same sources.

Outputs are compared as token sequences, since the reference tools space and break lines differently. Each case
must give the same tokens; each error case must be rejected by the reference tool and refused, with its typed
diagnostic code, by the shared preprocessor. A test is skipped when its reference tool is not installed.

Clang runs in ISO C++20 mode on every host. On Windows it defaults to Microsoft compatibility, which accepts
some ill-formed input (an invalid ``##`` paste is only a warning there), so ``-fno-ms-compatibility`` and
``-fno-ms-extensions`` are passed explicitly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.preprocessor import preprocess
from crosstl.translator.preprocessor.lexer import lex_text

from .cases import CPP_CASES, CPP_ERRORS, GLSL_CASES, GLSL_ERRORS, INCLUDE_TREE

CLANG = shutil.which("clang") or shutil.which("clang++")
GLSLANG = shutil.which("glslangValidator")
CLANG_ISO_CPP = [
    "-E",
    "-P",
    "-x",
    "c++",
    "-std=c++20",
    "-fno-ms-compatibility",
    "-fno-ms-extensions",
    "-Wno-everything",
]


def tokens(text, cpp=True):
    return [token.text for token in lex_text(text, cpp=cpp) if token.text]


def clang(path):
    return subprocess.run(
        [CLANG, *CLANG_ISO_CPP, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def glslang(path):
    return subprocess.run(
        [GLSLANG, "-E", str(path)], capture_output=True, text=True, timeout=60
    )


@pytest.mark.skipif(CLANG is None, reason="clang is not installed")
@pytest.mark.parametrize("name", sorted(CPP_CASES))
def test_cpp_case_matches_clang(name, tmp_path):
    path = tmp_path / f"{name}.cpp"
    path.write_text(CPP_CASES[name])
    reference = clang(path)
    assert reference.returncode == 0, reference.stderr
    ours = preprocess(CPP_CASES[name], "c++", path=str(path))
    assert tokens(ours.text) == tokens(reference.stdout)


@pytest.mark.skipif(CLANG is None, reason="clang is not installed")
@pytest.mark.parametrize("name", sorted(CPP_ERRORS))
def test_cpp_error_is_refused_like_clang(name, tmp_path):
    source, code = CPP_ERRORS[name]
    path = tmp_path / f"{name}.cpp"
    path.write_text(source)
    assert clang(path).returncode != 0
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess(source, "c++", path=str(path))
    assert raised.value.code == code


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
@pytest.mark.parametrize("name", sorted(GLSL_CASES))
def test_glsl_case_matches_glslang(name, tmp_path):
    path = tmp_path / f"{name}.vert"
    path.write_text(GLSL_CASES[name])
    reference = glslang(path)
    assert reference.returncode == 0, reference.stdout + reference.stderr
    ours = preprocess(GLSL_CASES[name], "glsl", path=str(path))
    assert tokens(ours.text, cpp=False) == tokens(reference.stdout, cpp=False)


@pytest.mark.skipif(GLSLANG is None, reason="glslangValidator is not installed")
@pytest.mark.parametrize("name", sorted(GLSL_ERRORS))
def test_glsl_error_is_refused_like_glslang(name, tmp_path):
    source, code = GLSL_ERRORS[name]
    path = tmp_path / f"{name}.vert"
    path.write_text(source)
    assert glslang(path).returncode != 0
    with pytest.raises(TranslationDiagnostic) as raised:
        preprocess(source, "glsl", path=str(path))
    assert raised.value.code == code


@pytest.mark.skipif(CLANG is None, reason="clang is not installed")
def test_include_closure_matches_clang(tmp_path):
    for name, text in INCLUDE_TREE.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    main = tmp_path / "main.cpp"
    paths = [str(tmp_path / "inc"), str(tmp_path / "inc2")]
    reference = subprocess.run(
        [CLANG, *CLANG_ISO_CPP] + [f"-I{path}" for path in paths] + [str(main)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert reference.returncode == 0, reference.stderr
    ours = preprocess(main.read_text(), "c++", path=str(main), include_paths=paths)
    assert tokens(ours.text) == tokens(reference.stdout)
    assert [Path(path).relative_to(tmp_path).as_posix() for path in ours.includes] == [
        "local.h",
        "inc/system.h",
        "inc2/system.h",
        "sub/nested.h",
        "sub/sibling.h",
        "once.h",
        "guarded.h",
    ]
    # Lines of an included file map back to that file.
    sibling = ours.text.splitlines().index("sibling_header 1") + 1
    assert ours.origin(sibling) == (str(tmp_path / "sub" / "sibling.h"), 1)
