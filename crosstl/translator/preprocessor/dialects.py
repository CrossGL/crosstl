"""Dialects of the shared preprocessor: what differs between the Metal, HLSL, GLSL and plain C++ front ends.

Metal and DXC are Clang-based, so their preprocessors are Clang's C++ preprocessor with each compiler's language
options and predefined macros; :mod:`.predefined` builds those dialects from data recorded from the toolchains
(``dialect("msl")`` and ``dialect("hlsl")`` give the default configurations). ``c++`` is Clang's C++20
preprocessor without predefined macros. GLSL (4.60 section 3.3, ES 3.20 section 3.4) is a reduced C preprocessor:
32-bit ``int`` expressions, no string or character literals, no variadic macros or ``#include`` (unless
GL_GOOGLE_include_directive or GL_ARB_shading_language_include enables it), and the ``#version`` and
``#extension`` directives, which are kept in the output for the GLSL front end.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from .lexer import CPP_MODE, GLSL_MODE, LexMode


@dataclass(frozen=True)
class HeaderMacros:
    """What including one compiler-provided header does to the macro table.

    ``defines`` are the definitions (the text after ``#define``) it adds, ``closure`` the compiler headers it
    includes (itself among them) and ``conditions`` the macro names its own conditional directives test: when the
    source has changed one of those, the header would expand differently and is refused. ``refusal`` says why the
    header cannot be included in this configuration; ``available`` is whether it exists for ``__has_include``.
    """

    defines: tuple[str, ...]
    closure: frozenset[str]
    conditions: frozenset[str]
    refusal: str | None = None
    available: bool = True


@dataclass(frozen=True)
class Dialect:
    name: str
    cpp: bool = True
    int_width: int = 64
    lex_mode: LexMode = CPP_MODE
    variadic: bool = True
    gnu_comma: bool = True
    va_opt: bool = True  # __VA_OPT__ (C++20; Clang 6 and later in every C++ mode)
    elifdef: bool = (
        True  # #elifdef and #elifndef (C++23; Clang 13 and later in every mode)
    )
    operator_names: bool = True  # 'and', 'or', 'not', ... are operators ([lex.digraph])
    file_name_macro: bool = True  # __FILE_NAME__ (Clang 9 and later)
    ident_directive: bool = True  # #ident and #sccs, kept in the output
    splice_whitespace: bool = True
    pragma_operator: bool = True
    include: bool = True
    include_extensions: frozenset[str] = frozenset()
    retained_directives: frozenset[str] = frozenset()
    retained_includes: frozenset[str] = frozenset()
    headers: Mapping[str, HeaderMacros] = field(default_factory=dict)
    implicit_headers: frozenset[str] = frozenset()
    implicit_conditions: frozenset[str] = frozenset()
    reserved_prefixes: tuple = ()
    redefinition_is_error: bool = False
    # Answers for Clang's feature-test macros (__has_feature, __has_attribute, ...); an unknown query is refused.
    feature_tests: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    # Full definitions ("NAME body" or "NAME(params) body"), then name-to-value definitions, in that order.
    predefined_definitions: tuple[str, ...] = ()
    predefined: Mapping[str, str] = field(default_factory=dict)
    toolchain: str = ""

    def with_predefined(self, macros: Mapping[str, str]) -> Dialect:
        merged: dict[str, str] = dict(self.predefined)
        merged.update(macros)
        return replace(self, predefined=merged)


CLANG_FEATURE_TESTS = (
    "__has_feature",
    "__has_extension",
    "__has_attribute",
    "__has_cpp_attribute",
    "__has_c_attribute",
    "__has_declspec_attribute",
    "__has_builtin",
    "__has_constexpr_builtin",
    "__has_warning",
    "__is_identifier",
    "__is_target_arch",
    "__is_target_os",
    "__is_target_environment",
    "__is_target_vendor",
)

CPP = Dialect("c++")
GLSL = Dialect(
    "glsl",
    cpp=False,
    int_width=32,
    lex_mode=GLSL_MODE,
    variadic=False,
    gnu_comma=False,
    va_opt=False,
    elifdef=False,
    operator_names=False,
    file_name_macro=False,
    ident_directive=False,
    splice_whitespace=False,
    pragma_operator=False,
    include=False,
    include_extensions=frozenset(
        {"GL_GOOGLE_include_directive", "GL_ARB_shading_language_include"}
    ),
    retained_directives=frozenset({"version", "extension"}),
    reserved_prefixes=("GL_",),
    redefinition_is_error=True,
)

DIALECTS = ("c++", "glsl", "msl", "hlsl")


def dialect(name_or_dialect, predefined: Mapping[str, str] | None = None) -> Dialect:
    """The named dialect in its default configuration, or ``name_or_dialect`` itself when it is a Dialect."""
    if isinstance(name_or_dialect, Dialect):
        chosen = name_or_dialect
    elif name_or_dialect == "c++":
        chosen = CPP
    elif name_or_dialect == "glsl":
        chosen = GLSL
    elif name_or_dialect in ("msl", "hlsl"):
        from .predefined import hlsl_dialect, msl_dialect

        chosen = msl_dialect() if name_or_dialect == "msl" else hlsl_dialect()
    else:
        raise KeyError(
            f"unknown preprocessor dialect {name_or_dialect!r}; known: {', '.join(DIALECTS)}"
        )
    return chosen.with_predefined(predefined) if predefined else chosen
