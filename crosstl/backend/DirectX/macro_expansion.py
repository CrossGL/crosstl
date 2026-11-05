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

The built-in macros are those of the compiler: ``__LINE__`` is the line where the outermost invocation that
produced it ends, as in Clang and GCC; the C and C++ dialects add Clang's ``__COUNTER__``, ``__INCLUDE_LEVEL__``,
``__BASE_FILE__`` and (Clang 9 and later) ``__FILE_NAME__``, and refuse ``__DATE__``, ``__TIME__`` and
``__TIMESTAMP__``, which would make the translation depend on the build time. A definition in the source takes
precedence over a built-in one, as it does in Clang after its redefinition warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from crosstl.translator.preprocessor.engine import BUILTIN_CODE, PreprocessorError
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
# Clang's further built-in macros in C and C++, and the ones it would expand to the build time.
CLANG_BUILTIN_MACROS = ("__COUNTER__", "__INCLUDE_LEVEL__", "__BASE_FILE__")
NONDETERMINISTIC_MACROS = ("__DATE__", "__TIME__", "__TIMESTAMP__")
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
    and later, GCC 12 and later, in every mode). ``clang_builtins`` adds ``__COUNTER__``, ``__INCLUDE_LEVEL__``
    and ``__BASE_FILE__`` and refuses the build-time macros; ``file_name_macro`` is ``__FILE_NAME__`` (Clang 9).
    """

    mode: LexMode = CPP_MODE
    va_opt: bool = True
    gnu_comma: bool = True
    int_width: int = 64
    operator_names: bool = True
    elifdef: bool = True
    clang_builtins: bool = True
    file_name_macro: bool = True

    @property
    def builtins(self) -> frozenset[str]:
        """Names of the built-in macros, for ``defined`` and ``#ifdef``."""
        names = set(BUILTIN_MACROS)
        if self.clang_builtins:
            names.update(CLANG_BUILTIN_MACROS + NONDETERMINISTIC_MACROS)
        if self.file_name_macro:
            names.add("__FILE_NAME__")
        return frozenset(names)


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
    file_name_macro=False,
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
    clang_builtins=False,
    file_name_macro=False,
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
    """The shared expander's view of a text macro table, with the dialect's built-in macros for one line."""

    def __init__(
        self,
        preprocessor: HLSLPreprocessor,
        cache: dict[str, tuple[tuple, SharedMacro]],
        line_num: int,
        file_path: str | None,
    ):
        self._preprocessor = preprocessor
        self._macros = preprocessor.macros
        self._cache = cache
        self._dialect: MacroDialect = preprocessor.macro_dialect
        self._builtins = self._dialect.builtins
        self._line_num = line_num
        self._file_path = file_path

    def get(self, name: str) -> SharedMacro | None:
        macro = self._macros.get(name)
        if macro is None:
            if name in self._builtins:
                return SharedMacro(name, builtin=self._builtin)
            return None
        params = tuple(macro.params) if macro.params is not None else None
        key = (params, macro.replacement, macro.is_variadic)
        cached = self._cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]
        converted = SharedMacro(
            name,
            body=lex_text(macro.replacement or "", mode=self._dialect.mode),
            params=list(params) if params is not None else None,
            variadic=bool(macro.is_variadic and params),
        )
        self._cache[name] = (key, converted)
        return converted

    def _builtin(self, token: Token) -> list[Token]:
        name = token.text
        preprocessor = self._preprocessor
        if name == "__LINE__":
            return [self._literal(NUMBER, str(self._line(token)), token)]
        if name == "__FILE__":
            text = preprocessor._stringize(self._file_path or "")
            return [self._literal(STRING, text, token)]
        if name == "__COUNTER__":
            value = getattr(preprocessor, "_macro_counter", 0)
            preprocessor._macro_counter = value + 1
            return [self._literal(NUMBER, str(value), token)]
        if name == "__INCLUDE_LEVEL__":
            level = getattr(preprocessor, "_include_depth", 0)
            return [self._literal(NUMBER, str(level), token)]
        if name == "__BASE_FILE__":
            base = getattr(preprocessor, "_base_file", None)
            return [self._literal(STRING, _quote(base or "<stdin>"), token)]
        if name == "__FILE_NAME__":
            leaf = re.split(r"[\\/]", self._file_path or "<stdin>")[-1]
            return [self._literal(STRING, _quote(leaf), token)]
        raise PreprocessorError(
            f"{name} makes the output depend on the build time and is not supported",
            span=token.span(),
            code=BUILTIN_CODE,
        )

    def _line(self, token: Token) -> int:
        """The physical line where the outermost invocation containing ``token`` ends."""
        source, offset = token.end_location()
        if source is None:
            return self._line_num
        segment = source.position(offset)[0] - 1
        joined = getattr(self._preprocessor, "_joined_line_numbers", None)
        if joined and segment < len(joined):
            return joined[segment]
        return self._line_num + segment

    @staticmethod
    def _literal(kind: str, text: str, origin: Token) -> Token:
        return Token(kind, text, origin.source, origin.offset, origin.space)


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def code_extent(text: str) -> int:
    """Where a block comment that ``text`` opens but does not close begins, or ``len(text)``."""
    if "/*" not in text:
        return len(text)
    for match in _SKIPPED_RE.finditer(text):
        if match.group("open") is not None:
            return match.start()
    return len(text)


def _names_a_macro(
    text: str, macros: dict[str, Macro], builtins: frozenset[str]
) -> bool:
    return any(
        name in macros or name in builtins for name in _IDENTIFIER_RE.findall(text)
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
    """Replace the macros of one logical source line.

    Sets ``preprocessor._invocation_continues`` when the replaced line ends in the name of a function-like macro
    that may still be invoked by a ``(`` on a following line ([cpp.rescan]/1).
    """
    macros = preprocessor.macros
    preprocessor._invocation_continues = False
    dialect: MacroDialect = preprocessor.macro_dialect
    builtins = dialect.builtins
    if not _names_a_macro(text, macros, builtins):
        return text
    extent = code_extent(text)
    code, tail = text[:extent], text[extent:]
    tokens = _lex_joined(code, dialect.mode)
    first = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.kind == IDENT and (token.text in macros or token.text in builtins)
        ),
        None,
    )
    if first is None:
        return text
    hide = frozenset(disabled_macros or ())
    pending = [token.copy(hide=hide) if hide else token for token in tokens[first:]]
    expanded = _expander(preprocessor, line_num, file_path).expand(pending)
    if expanded and not tail:
        last = expanded[-1]
        macro = macros.get(last.text) if last.kind == IDENT else None
        preprocessor._invocation_continues = bool(
            macro is not None and macro.is_function_like() and not last.noexpand
        )
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
        for token in lex_text(text[: code_extent(text)], mode=dialect.mode)
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
            name.text in macros or name.text in dialect.builtins
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
    table = _SharedTable(preprocessor, cache, line_num, file_path)
    return Expander(
        table,  # type: ignore[arg-type]  # the expander only calls get()
        cpp=dialect.mode.cpp,
        gnu_comma=dialect.gnu_comma,
        va_opt=dialect.va_opt,
        mode=dialect.mode,
    )
