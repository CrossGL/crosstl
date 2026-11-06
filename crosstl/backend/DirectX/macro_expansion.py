"""Conforming macro replacement for the line-based source preprocessors.

:class:`~crosstl.backend.DirectX.preprocessor.HLSLPreprocessor` and its subclasses (Metal, GLSL, CUDA, HIP,
OpenCL and Slang) interpret directives line by line and keep their macro tables as replacement text. Replacing
macros in that text by text substitution is not what a compiler does: an argument such as ``F(1)`` in ``F(F(1))``
was substituted unexpanded and then could not be expanded at all, ``CAT(v, N)`` through ``CAT_I(a, b) a##b`` pasted
the unexpanded ``N``, a function-like macro name produced at the end of a replacement never met the ``(`` that
follows the invocation, and a self-referential object-like macro used in an argument expanded twice.

This module hands each line that names a macro to the shared preprocessor's
:class:`~crosstl.translator.preprocessor.macros.Expander`, which implements the rescanning rules of C++
[cpp.rescan] with hide sets: arguments are fully replaced before substitution unless they are operands of ``#`` or
``##``, a name produced by its own expansion is never expanded again, the rescan takes in the tokens that follow
the invocation, ``##`` works on placemarkers, GNU ``, ## __VA_ARGS__`` drops the comma for an empty variable
argument, and ``__VA_OPT__`` follows C++20 [cpp.subst] where the compiler implements it. A malformed invocation (a
wrong argument count, an unterminated argument list, a paste that forms no token) is refused with the shared
preprocessor's typed diagnostic instead of being substituted loosely.

The text before the first macro name of a line is kept verbatim; the rest of the line is spelled from its tokens
with the original spacing reduced to single spaces, plus a space wherever two adjacent tokens would otherwise lex
as one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from crosstl.translator.preprocessor.expression import evaluate
from crosstl.translator.preprocessor.lexer import (
    CPP_MODE,
    GLSL_MODE,
    IDENT,
    NEWLINE,
    NUMBER,
    PUNCT,
    STRING,
    LexMode,
    SourceFile,
    Token,
    lex,
    lex_text,
)
from crosstl.translator.preprocessor.macros import Expander
from crosstl.translator.preprocessor.macros import Macro as SharedMacro
from crosstl.translator.preprocessor.output import needs_space

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from .preprocessor import HLSLPreprocessor, Macro

BUILTIN_MACROS = ("__LINE__", "__FILE__")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
# A string or character literal, a comment, or the start of a block comment that the line does not close.
_SKIPPED_RE = re.compile(
    r"\"(?:\\.|[^\"\\\n])*\"?|'(?:\\.|[^'\\\n])*'?|//.*|/\*.*?\*/|(?P<open>/\*)", re.S
)


@dataclass(frozen=True)
class MacroDialect:
    """How a compiler's preprocessor replaces macros and evaluates ``#if``.

    ``mode`` is its lexical grammar. ``va_opt`` is C++20 ``__VA_OPT__`` (Clang 6 and later in every C and C++
    mode, GCC 8 and later); without it the name is an ordinary identifier. ``gnu_comma`` is the GNU
    ``, ## __VA_ARGS__`` comma elision, which Clang and GCC implement in every mode. ``int_width`` is the width
    of ``intmax_t`` that ``#if`` evaluates in, ``operator_names`` makes ``and``, ``or``, ``not`` and the rest
    operators in ``#if`` ([lex.digraph]), and ``elifdef`` accepts ``#elifdef`` and ``#elifndef`` (C++23; Clang 13
    and later, GCC 12 and later, in every mode).
    """

    mode: LexMode = CPP_MODE
    va_opt: bool = True
    gnu_comma: bool = True
    int_width: int = 64
    operator_names: bool = True
    elifdef: bool = True


# DXC is Clang 3.7 in an HLSL mode: no digraphs, raw strings, digit separators, operator names or __VA_OPT__.
HLSL_MACROS = MacroDialect(
    mode=replace(
        CPP_MODE,
        digraphs=False,
        raw_strings=False,
        digit_separators=False,
        unicode_literals=False,
        utf8_characters=False,
    ),
    va_opt=False,
    operator_names=False,
    elifdef=False,
)
# The Metal compiler is Clang in C++14 or C++17 mode without raw string literals; intmax_t is 128 bits on AIR.
METAL_MACROS = MacroDialect(mode=replace(CPP_MODE, raw_strings=False), int_width=128)
# OpenCL C is C99-based: no raw strings, digit separators or operator names.
OPENCL_MACROS = MacroDialect(
    mode=replace(CPP_MODE, raw_strings=False, digit_separators=False),
    operator_names=False,
)
# CUDA and HIP compile as C++ with Clang or the host compiler's preprocessor.
CPP_MACROS = MacroDialect()
# GLSL has no variadic macros, evaluates #if in 32-bit int and has no #elifdef.
GLSL_MACROS = MacroDialect(
    mode=GLSL_MODE,
    va_opt=False,
    gnu_comma=False,
    int_width=32,
    operator_names=False,
    elifdef=False,
)

_SPACE_CACHE: dict[tuple[str, str, LexMode], bool] = {}


def _needs_space(previous: Token, token: Token, mode: LexMode) -> bool:
    if previous.kind == PUNCT and token.kind == PUNCT:
        key = (previous.text, token.text, mode)
        cached = _SPACE_CACHE.get(key)
        if cached is None:
            cached = _SPACE_CACHE[key] = needs_space(previous, token, mode)
        return cached
    return needs_space(previous, token, mode)


class _SharedTable:
    """The shared expander's view of a text macro table, with ``__LINE__`` and ``__FILE__`` for one line."""

    def __init__(
        self,
        macros: dict[str, Macro],
        cache: dict[str, tuple[tuple, SharedMacro]],
        mode: LexMode,
        line_num: int,
        file_text: str,
    ):
        self._macros = macros
        self._cache = cache
        self._mode = mode
        self._line_num = line_num
        self._file_text = file_text

    def get(self, name: str) -> SharedMacro | None:
        if name == "__LINE__":
            return SharedMacro(
                name, builtin=lambda token: [self._literal(NUMBER, token)]
            )
        if name == "__FILE__":
            return SharedMacro(
                name, builtin=lambda token: [self._literal(STRING, token)]
            )
        macro = self._macros.get(name)
        if macro is None:
            return None
        params = tuple(macro.params) if macro.params is not None else None
        key = (params, macro.replacement, macro.is_variadic)
        cached = self._cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        converted = SharedMacro(
            name,
            body=lex_text(macro.replacement or "", mode=self._mode),
            params=list(params) if params is not None else None,
            variadic=bool(macro.is_variadic and params),
        )
        self._cache[name] = (key, converted)
        return converted

    def _literal(self, kind: str, origin: Token) -> Token:
        text = str(self._line_num) if kind == NUMBER else self._file_text
        return Token(kind, text, origin.source, origin.offset, origin.space)


def _code_extent(text: str) -> int:
    """Where a block comment that ``text`` opens but does not close begins, or ``len(text)``."""
    if "/*" not in text:
        return len(text)
    for match in _SKIPPED_RE.finditer(text):
        if match.group("open") is not None:
            return match.start()
    return len(text)


def _names_a_macro(text: str, macros: dict[str, Macro]) -> bool:
    return any(
        name in macros or name in BUILTIN_MACROS
        for name in _IDENTIFIER_RE.findall(text)
    )


def _spell(
    tokens: list[Token],
    mode: LexMode,
    previous: Token | None,
    at_space: bool,
) -> str:
    """Spell ``tokens`` after ``previous`` (the last token already written, if any); ``at_space`` says the text
    written so far ends in whitespace."""
    parts: list[str] = []
    for index, token in enumerate(tokens):
        if previous is not None:
            if index == 0:
                separate = not at_space and _needs_space(previous, token, mode)
            else:
                separate = token.space or _needs_space(previous, token, mode)
            if separate:
                parts.append(" ")
        parts.append(token.text)
        previous = token
    return "".join(parts)


def _lex_joined(code: str, mode: LexMode) -> list[Token]:
    """Tokens of ``code``; a line break of a joined invocation separates tokens like a space."""
    tokens: list[Token] = []
    broken = False
    for token in lex(SourceFile(None, code), mode=mode):
        if token.kind == NEWLINE:
            broken = True
            continue
        if broken:
            token.space = True
            broken = False
        tokens.append(token)
    return tokens


def expand_line(
    preprocessor: HLSLPreprocessor,
    text: str,
    line_num: int,
    file_path: str | None,
    disabled_macros: set[str] | None = None,
) -> str:
    """Replace the macros of one logical source line."""
    macros = preprocessor.macros
    if not _names_a_macro(text, macros):
        return text
    dialect: MacroDialect = preprocessor.macro_dialect
    extent = _code_extent(text)
    code, tail = text[:extent], text[extent:]
    tokens = _lex_joined(code, dialect.mode)
    first = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.kind == IDENT
            and (token.text in macros or token.text in BUILTIN_MACROS)
        ),
        None,
    )
    if first is None:
        return text
    hide = frozenset(disabled_macros or ())
    pending = [token.copy(hide=hide) if hide else token for token in tokens[first:]]
    expanded = _expander(preprocessor, line_num, file_path).expand(pending)
    prefix = code[: tokens[first].offset]
    previous = tokens[first - 1] if first else None
    at_space = not prefix or prefix[-1].isspace()
    spelled = _spell(expanded, dialect.mode, previous, at_space)
    # A joined multi-line invocation keeps its line count, so later lines keep their numbers.
    breaks = code.count("\n", tokens[first].offset)
    return prefix + spelled + "\n" * breaks + tail


def expand_condition(
    preprocessor: HLSLPreprocessor,
    text: str,
    line_num: int,
    file_path: str | None,
    disabled_macros: set[str] | None = None,
) -> str:
    """Replace the macros of a ``#if`` or ``#elif`` expression and evaluate ``defined``; the result is spelled
    with spaces between its tokens for :func:`evaluate_condition`."""
    dialect: MacroDialect = preprocessor.macro_dialect
    macros = preprocessor.macros
    hide = frozenset(disabled_macros or ())
    tokens = [
        token.copy(hide=hide) if hide else token
        for token in lex_text(text[: _code_extent(text)], mode=dialect.mode)
    ]

    def defined(token: Token, stack: list[Token]):
        if token.text != "defined":
            return None
        parenthesized = bool(stack) and stack[-1].text == "("
        if parenthesized:
            stack.pop()
        name = stack.pop() if stack else None
        if parenthesized and stack and stack[-1].text == ")":
            stack.pop()
        is_defined = name is not None and (
            name.text in macros or name.text in BUILTIN_MACROS
        )
        return [
            Token(
                NUMBER,
                "1" if is_defined else "0",
                token.source,
                token.offset,
                token.space,
            )
        ]

    expanded = _expander(preprocessor, line_num, file_path).expand(
        tokens, on_identifier=defined
    )
    return " ".join(token.text for token in expanded if token.text)


def evaluate_condition(preprocessor: HLSLPreprocessor, text: str) -> bool:
    """Truth of a macro-replaced ``#if`` expression, evaluated as the dialect's compiler does
    (:func:`crosstl.translator.preprocessor.expression.evaluate`): identifiers left are 0, ``true`` and
    ``false`` are 1 and 0 in C++, and a malformed expression is refused with a typed diagnostic.
    """
    dialect: MacroDialect = preprocessor.macro_dialect
    tokens = lex_text(text, mode=dialect.mode)
    anchor = Token(PUNCT, "#", SourceFile(None, text), 0)
    return evaluate(
        tokens,
        width=dialect.int_width,
        cpp=dialect.mode.cpp,
        directive=anchor,
        operator_names=dialect.operator_names,
    )


def _expander(
    preprocessor: HLSLPreprocessor, line_num: int, file_path: str | None
) -> Expander:
    dialect: MacroDialect = preprocessor.macro_dialect
    cache = preprocessor.__dict__.setdefault("_shared_macro_cache", {})
    table = _SharedTable(
        preprocessor.macros,
        cache,
        dialect.mode,
        line_num,
        preprocessor._stringize(file_path or ""),
    )
    return Expander(
        table,  # type: ignore[arg-type]  # the expander only calls get()
        cpp=dialect.mode.cpp,
        gnu_comma=dialect.gnu_comma,
        va_opt=dialect.va_opt,
        mode=dialect.mode,
    )
