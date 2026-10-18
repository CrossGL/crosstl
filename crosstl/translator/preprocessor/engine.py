"""Directive processing for the shared preprocessor: C++ [cpp], C 6.10 and GLSL section 3.3.

The engine walks each file's logical lines. Text lines of active groups are buffered and macro-replaced as one
segment when a directive or the end of the file is reached, so a macro invocation may span lines; one that would
span a directive is refused, as is every other construct the engine cannot reproduce exactly. Includes are
resolved like Clang: a quoted name first in the including file's directory, then the user paths, then the system
paths; ``#pragma once`` and ``#include_next`` are honoured, and the include closure is recorded in order.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..diagnostics import Diagnostic, TranslationDiagnostic
from .dialects import CLANG_FEATURE_TESTS, Dialect, HeaderMacros
from .expression import OPERATOR_NAMES, evaluate
from .lexer import (
    IDENT,
    NEWLINE,
    NUMBER,
    PUNCT,
    STRING,
    UNTERMINATED,
    SourceFile,
    Token,
    lex,
    lex_text,
    spell,
)
from .macros import DEFINITION_CODE, Expander, Macro, UnterminatedInvocation
from .output import Writer

CONDITIONAL_CODE = "project.translate.preprocessor-unbalanced-conditional"
DIRECTIVE_CODE = "project.translate.preprocessor-invalid-directive"
ERROR_CODE = "project.translate.preprocessor-error-directive"
INCLUDE_CODE = "project.translate.preprocessor-include-not-found"
INCLUDE_DEPTH_CODE = "project.translate.preprocessor-include-depth"
EMBEDDED_CODE = "project.translate.preprocessor-directive-in-macro-arguments"
BUILTIN_CODE = "project.translate.preprocessor-unsupported-builtin"
INVALID_TOKEN_CODE = "project.translate.preprocessor-invalid-token"
HEADER_CODE = "project.translate.preprocessor-header-unavailable"
HEADER_CONFIGURATION_CODE = "project.translate.preprocessor-header-configuration"

MAX_INCLUDE_DEPTH = 200
_CONDITIONALS = {
    "if",
    "ifdef",
    "ifndef",
    "elif",
    "elifdef",
    "elifndef",
    "else",
    "endif",
}
_NONDETERMINISTIC = ("__DATE__", "__TIME__", "__TIMESTAMP__")


class PreprocessorError(TranslationDiagnostic):
    code = DIRECTIVE_CODE


class FileSystem:
    """Where included files come from; the default reads the host file system as UTF-8."""

    def is_file(self, path: str) -> bool:
        return os.path.isfile(path)

    def read(self, path: str) -> str:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8", errors="replace")

    def identity(self, path: str) -> str:
        return os.path.realpath(path)


class MemoryFileSystem(FileSystem):
    """Files held in a mapping from normalized path to text, for in-memory sources and tests."""

    def __init__(self, files: dict[str, str]):
        self.files = {os.path.normpath(path): text for path, text in files.items()}

    def is_file(self, path: str) -> bool:
        return os.path.normpath(path) in self.files

    def read(self, path: str) -> str:
        return self.files[os.path.normpath(path)]

    def identity(self, path: str) -> str:
        return os.path.normpath(path)


@dataclass
class _Conditional:
    directive: Token
    parent_active: bool
    active: bool
    taken: bool
    seen_else: bool = False


@dataclass
class _Context:
    source: SourceFile
    directory: str
    found_in: int | None
    presumed: str | None
    line_delta: int = 0


@dataclass
class PreprocessResult:
    """Preprocessed text; ``line_map[i]`` is the (file, line) that output line ``i + 1`` came from."""

    text: str
    line_map: list[tuple[str | None, int]]
    includes: list[str] = field(default_factory=list)
    retained_includes: list[str] = field(default_factory=list)
    macros: dict[str, Macro] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def origin(self, line: int) -> tuple[str | None, int]:
        """Source (file, line) of 1-based output ``line``."""
        return (
            self.line_map[line - 1] if 0 < line <= len(self.line_map) else (None, line)
        )


class Preprocessor:
    def __init__(
        self,
        dialect: Dialect,
        *,
        include_paths: Sequence[str] = (),
        system_paths: Sequence[str] = (),
        defines: dict[str, str | None] | None = None,
        undefines: Sequence[str] = (),
        file_system: FileSystem | None = None,
    ):
        self.dialect = dialect
        self.search = [str(path) for path in include_paths] + [
            str(path) for path in system_paths
        ]
        self.quote_only = 0
        self.fs = file_system or FileSystem()
        self.macros: dict[str, Macro] = {}
        self.mode = dialect.lex_mode
        self.expander = Expander(
            self.macros,
            cpp=dialect.cpp,
            gnu_comma=dialect.gnu_comma,
            va_opt=dialect.va_opt,
            mode=self.mode,
        )
        self.diagnostics: list[Diagnostic] = []
        self.contexts: list[_Context] = []
        self.once: set = set()
        self.includes: list[str] = []
        self.retained: list[str] = []
        self.counter = 0
        self.enabled_extensions: set = set()
        self.main_path: str | None = None
        self._install_builtins()
        for definition in dialect.predefined_definitions:
            self.define_definition(definition)
        for name, value in dialect.predefined.items():
            self.define_text(name, value)
        # The macro table the compiler's own headers were recorded against: the predefined macros, plus what
        # each compiler header entered so far defined. A header whose conditions see a different value is refused.
        self.pristine: dict[str, Macro] = dict(self.macros)
        self.entered: set = set(dialect.implicit_headers)
        self.header_defined: set = set()
        for name, value in (defines or {}).items():
            self.define_text(name, "1" if value is None else str(value))
        for name in undefines:
            self.macros.pop(name, None)

    # Public entry point ------------------------------------------------------------------------------
    def run(self, text: str, path: str | None = None) -> PreprocessResult:
        self.main_path = path
        self._check_conditions(
            self.dialect.implicit_conditions, "the implicitly included headers", None
        )
        writer = Writer(self.mode)
        directory = os.path.dirname(path) if path else "."
        source = SourceFile(
            path, text, splice_whitespace=self.dialect.splice_whitespace
        )
        self._process(source, directory, None, writer)
        return PreprocessResult(
            writer.text(),
            writer.line_map,
            list(self.includes),
            list(self.retained),
            dict(self.macros),
            list(self.diagnostics),
        )

    def define_text(self, name: str, value: str) -> None:
        """Define ``name`` as if by ``-D name=value``: no reserved-name check, and a later definition wins."""
        tokens = lex_text(f"{name} {value}", mode=self.mode)
        self.macros.pop(name, None)
        self._define(tokens, Token(PUNCT, "#"), checked=False)

    def define_definition(self, definition: str) -> None:
        """Define a macro from its ``#define`` text (``NAME body`` or ``NAME(params) body``), as a header would."""
        self._define(
            lex_text(definition, mode=self.mode), Token(PUNCT, "#"), checked=False
        )

    # Files and lines ---------------------------------------------------------------------------------
    def _process(
        self, source: SourceFile, directory: str, found_in, writer: Writer
    ) -> None:
        if len(self.contexts) >= MAX_INCLUDE_DEPTH:
            raise PreprocessorError(
                f"#include nested deeper than {MAX_INCLUDE_DEPTH} levels",
                span=source.span(0),
                code=INCLUDE_DEPTH_CODE,
            )
        self.contexts.append(_Context(source, directory, found_in, source.path))
        tokens = lex(source, mode=self.mode)
        stack: list[_Conditional] = []
        segment: list[Token] = []
        line: list[Token] = []
        for token in tokens:
            if token.kind != NEWLINE:
                line.append(token)
                continue
            if line and line[0].kind == PUNCT and line[0].text in ("#", "%:"):
                interrupted = self._flush(segment, writer, directive=line[0])
                segment = []
                if interrupted is not None:
                    raise interrupted
                self._directive(line, stack, writer)
            elif not stack or stack[-1].active:
                segment.extend(line)
                segment.append(token)
            line = []
        failure = self._flush(segment, writer, directive=None)
        if failure is not None:
            raise failure
        if stack:
            raise PreprocessorError(
                f"unterminated #{stack[-1].directive.text}",
                span=stack[-1].directive.span(),
                code=CONDITIONAL_CODE,
            )
        self.contexts.pop()

    def _flush(self, segment: list[Token], writer: Writer, *, directive: Token | None):
        """Macro-replace and write a text segment; returns the error to raise when a directive interrupted a
        macro invocation, so the caller raises it with the directive's location."""
        if not segment:
            return None
        try:
            expanded = self.expander.expand(segment)
        except UnterminatedInvocation:
            if directive is None:
                raise
            return PreprocessorError(
                "a preprocessing directive inside the arguments of a macro invocation is not supported",
                span=directive.span(),
                code=EMBEDDED_CODE,
            )
        for token in expanded:
            if token.kind == UNTERMINATED:
                raise PreprocessorError(
                    f"missing terminating {token.text} character",
                    span=token.span(),
                    code=INVALID_TOKEN_CODE,
                )
        writer.write(
            self._pragma_operators(expanded)
            if self.dialect.pragma_operator
            else expanded
        )
        return None

    def _pragma_operators(self, tokens: list[Token]) -> list[Token]:
        """[cpp.pragma.op]: ``_Pragma("x")`` becomes a ``#pragma x`` line."""
        out: list[Token] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.kind == IDENT and token.text == "_Pragma":
                window = [t for t in tokens[index + 1 : index + 4] if t.kind != NEWLINE]
                if (
                    len(window) < 3
                    or window[0].text != "("
                    or window[1].kind != STRING
                    or window[2].text != ")"
                ):
                    raise PreprocessorError(
                        "_Pragma takes a parenthesized string literal",
                        span=token.span(),
                    )
                literal = window[1].text
                body = (
                    literal[literal.index('"') + 1 : -1]
                    .replace('\\"', '"')
                    .replace("\\\\", "\\")
                )
                index = tokens.index(window[2], index) + 1
                pragma = lex_text(body, mode=self.mode)
                if self.dialect.cpp and [t.text for t in pragma] == ["once"]:
                    self._pragma(
                        pragma, token
                    )  # applied like '#pragma once', which is not kept either
                    continue
                out.append(Token(NEWLINE, "\n", token.source, token.offset))
                out.extend(
                    t.copy(source=token.source, offset=token.offset)
                    for t in lex_text("#pragma " + body, mode=self.mode)
                )
                out.append(Token(NEWLINE, "\n", token.source, token.offset))
                continue
            out.append(token)
            index += 1
        return out

    # Directives --------------------------------------------------------------------------------------
    def _directive(
        self, line: list[Token], stack: list[_Conditional], writer: Writer
    ) -> None:
        hash_token = line[0]
        if len(line) == 1:
            return  # the null directive
        name_token = line[1]
        name = name_token.text if name_token.kind == IDENT else None
        args = line[2:]
        active = not stack or stack[-1].active
        if (
            name in ("elifdef", "elifndef")
            and self.dialect.cpp
            and not self.dialect.elifdef
        ):
            name = None  # before C++23 these are not directives: skipped in a skipped group, else an error
        if name in _CONDITIONALS:
            self._conditional(name, name_token, args, stack)
            return
        if not active:
            return  # [cpp.cond]: non-conditional directives of a skipped group are not processed
        if name == "define":
            self._define(args, name_token)
        elif name == "undef":
            self._undef(args, name_token)
        elif name in ("include", "include_next") and (
            self.dialect.include or self._include_enabled()
        ):
            if name == "include_next" and not self.dialect.cpp:
                raise PreprocessorError(
                    "#include_next is not GLSL", span=name_token.span()
                )
            self._include(args, name_token, writer, after=name == "include_next")
        elif name == "line":
            self._line(args, name_token)
        elif name == "error":
            raise PreprocessorError(
                f"#error {spell(args)}".rstrip(),
                span=hash_token.span(),
                code=ERROR_CODE,
            )
        elif name == "warning" and self.dialect.cpp:
            self._warn(
                "project.translate.preprocessor-warning-directive",
                f"#warning {spell(args)}",
                hash_token,
            )
        elif name == "pragma":
            if self._pragma(args, hash_token):
                writer.directive(line)
        elif name in ("ident", "sccs") and self.dialect.ident_directive:
            if (
                len(args) != 1
                or args[0].kind != STRING
                or not args[0].text.startswith('"')
            ):
                raise PreprocessorError(
                    f"invalid #{name} directive", span=name_token.span()
                )
            writer.directive([hash_token, name_token.copy(text="ident"), args[0]])
        elif name in self.dialect.retained_directives:
            if name == "extension" and args and args[0].kind == IDENT:
                self._extension(args)
            if name == "version":
                self._version(args)
            writer.directive(line)
        else:
            raise PreprocessorError(
                f"invalid preprocessing directive #{name_token.text}",
                span=name_token.span(),
            )

    def _conditional(
        self, name: str, name_token: Token, args: list[Token], stack
    ) -> None:
        if name in ("if", "ifdef", "ifndef"):
            parent = not stack or stack[-1].active
            truth = parent and self._condition(name, name_token, args)
            stack.append(_Conditional(name_token, parent, truth, truth))
            return
        if not stack:
            raise PreprocessorError(
                f"#{name} without #if", span=name_token.span(), code=CONDITIONAL_CODE
            )
        top = stack[-1]
        if name == "endif":
            stack.pop()
            return
        if top.seen_else:
            raise PreprocessorError(
                f"#{name} after #else", span=name_token.span(), code=CONDITIONAL_CODE
            )
        if name == "else":
            top.seen_else = True
            top.active = top.parent_active and not top.taken
            top.taken = top.taken or top.active
            return
        # #elif, #elifdef, #elifndef: not evaluated once a group was taken ([cpp.cond]/13, C23 6.10.2).
        if not self.dialect.cpp and name != "elif":
            raise PreprocessorError(f"#{name} is not GLSL", span=name_token.span())
        if top.taken or not top.parent_active:
            top.active = False
            return
        top.active = self._condition(
            name[2:] if name != "elif" else "if", name_token, args
        )
        top.taken = top.active

    def _condition(self, kind: str, name_token: Token, args: list[Token]) -> bool:
        if kind == "if":
            expanded = self.expander.expand(args, on_identifier=self._if_identifier)
            return evaluate(
                expanded,
                width=self.dialect.int_width,
                cpp=self.dialect.cpp,
                directive=name_token,
                operator_names=self.dialect.operator_names,
            )
        if not args or args[0].kind != IDENT:
            raise PreprocessorError(
                f"macro name missing after #{name_token.text}", span=name_token.span()
            )
        defined = args[0].text in self.macros
        return defined if kind == "ifdef" else not defined

    def _if_identifier(self, token: Token, stack: list[Token]):
        """Claims ``defined``, ``__has_include`` and Clang's feature tests while an ``#if`` is expanded."""
        if token.text == "defined":
            if stack and stack[-1].text == "(":
                stack.pop()
                name = stack.pop() if stack else None
                closing = stack.pop() if stack else None
                if (
                    name is None
                    or name.kind != IDENT
                    or closing is None
                    or closing.text != ")"
                ):
                    raise PreprocessorError(
                        "'defined' takes one identifier", span=token.span()
                    )
            else:
                name = stack.pop() if stack else None
                if name is None or name.kind != IDENT:
                    raise PreprocessorError(
                        "'defined' takes one identifier", span=token.span()
                    )
            return [
                Token(
                    NUMBER,
                    "1" if name.text in self.macros else "0",
                    token.source,
                    token.offset,
                    token.space,
                )
            ]
        if not self.dialect.cpp:
            return None
        if token.text in ("__has_include", "__has_include_next"):
            header, quoted = self._header_name(self._parenthesized(token, stack), token)
            found = self._resolve(header, quoted, after=token.text.endswith("next"))
            info = self.dialect.headers.get(header)
            known = found is not None or (
                not quoted
                and header in self.dialect.retained_includes
                and (info is None or info.available)
            )
            return [
                Token(
                    NUMBER,
                    "1" if known else "0",
                    token.source,
                    token.offset,
                    token.space,
                )
            ]
        if token.text in CLANG_FEATURE_TESTS:
            argument = spell(self._parenthesized(token, stack))
            table = self.dialect.feature_tests.get(token.text)
            if table is None or argument not in table:
                raise PreprocessorError(
                    f"{token.text}({argument}) depends on the target compiler and is not known for "
                    f"{self.dialect.name}",
                    span=token.span(),
                    code=BUILTIN_CODE,
                )
            return [
                Token(
                    NUMBER,
                    str(table[argument]),
                    token.source,
                    token.offset,
                    token.space,
                )
            ]
        return None

    def _parenthesized(self, token: Token, stack: list[Token]) -> list[Token]:
        if not stack or stack[-1].text != "(":
            raise PreprocessorError(
                f"'{token.text}' must be followed by '('", span=token.span()
            )
        stack.pop()
        inner, depth = [], 0
        while stack:
            item = stack.pop()
            if item.text == "(":
                depth += 1
            elif item.text == ")":
                if depth == 0:
                    return inner
                depth -= 1
            inner.append(item)
        raise PreprocessorError(f"missing ')' after '{token.text}'", span=token.span())

    # #define and #undef ------------------------------------------------------------------------------
    def _define(
        self, args: list[Token], directive: Token, *, checked: bool = True
    ) -> None:
        if not args or args[0].kind != IDENT:
            raise PreprocessorError(
                "macro name must be an identifier",
                span=directive.span(),
                code=DEFINITION_CODE,
            )
        name = args[0]
        if checked:
            self._check_name(name, "define")
        params, variadic, body_start = None, False, 1
        if len(args) > 1 and args[1].text == "(" and not args[1].space:
            params, variadic, body_start = self._parameters(args, name)
        body = [token.copy() for token in args[body_start:]]
        if body:
            body[0].space = False
        for index, token in enumerate(body):
            if token.text in ("##", "%:%:") and (index == 0 or index == len(body) - 1):
                raise PreprocessorError(
                    "'##' cannot appear at either end of a macro expansion",
                    span=token.span(),
                    code=DEFINITION_CODE,
                )
            if params is not None and token.text in ("#", "%:"):
                following = body[index + 1] if index + 1 < len(body) else None
                if following is None or (
                    following.text not in params
                    and not (variadic and following.text == "__VA_OPT__")
                ):
                    raise PreprocessorError(
                        "'#' is not followed by a macro parameter",
                        span=token.span(),
                        code=DEFINITION_CODE,
                    )
        macro = Macro(name.text, body, params, variadic, name.span())
        previous = self.macros.get(name.text)
        if (
            previous is not None
            and previous.builtin is None
            and not previous.same_as(macro)
        ):
            message = f"macro '{name.text}' redefined with a different replacement"
            if self.dialect.redefinition_is_error:
                raise PreprocessorError(message, span=name.span(), code=DEFINITION_CODE)
            self._warn("project.translate.preprocessor-macro-redefined", message, name)
        self.macros[name.text] = macro

    def _parameters(self, args: list[Token], name: Token):
        params: list[str] = []
        index, variadic = 2, False
        while True:
            token = args[index] if index < len(args) else None
            if token is None:
                raise PreprocessorError(
                    "missing ')' in macro parameter list",
                    span=name.span(),
                    code=DEFINITION_CODE,
                )
            if token.text == ")" and not params and not variadic:
                return params, False, index + 1
            if token.text == "..." and self.dialect.variadic:
                params.append("__VA_ARGS__")
                variadic = True
            elif (
                token.kind == IDENT
                and token.text not in params
                and token.text != "__VA_ARGS__"
            ):
                params.append(token.text)
                following = args[index + 1] if index + 1 < len(args) else None
                if (
                    following is not None
                    and following.text == "..."
                    and self.dialect.variadic
                    and self.dialect.cpp
                ):
                    variadic = True  # GNU named variable argument
                    index += 1
            else:
                raise PreprocessorError(
                    f"invalid macro parameter '{token.text}'",
                    span=token.span(),
                    code=DEFINITION_CODE,
                )
            index += 1
            separator = args[index] if index < len(args) else None
            if separator is not None and separator.text == ")":
                return params, variadic, index + 1
            if separator is None or separator.text != "," or variadic:
                raise PreprocessorError(
                    "expected ',' or ')' in macro parameter list",
                    span=(separator or token).span(),
                    code=DEFINITION_CODE,
                )
            index += 1

    def _undef(self, args: list[Token], directive: Token) -> None:
        if not args or args[0].kind != IDENT:
            raise PreprocessorError(
                "macro name must be an identifier",
                span=directive.span(),
                code=DEFINITION_CODE,
            )
        self._check_name(args[0], "undef")
        self.macros.pop(args[0].text, None)

    def _check_name(self, name: Token, action: str) -> None:
        reserved = (
            ("__VA_ARGS__", "__VA_OPT__") if self.dialect.va_opt else ("__VA_ARGS__",)
        )
        if name.text == "defined" or (self.dialect.cpp and name.text in reserved):
            raise PreprocessorError(
                f"'{name.text}' cannot be a macro name",
                span=name.span(),
                code=DEFINITION_CODE,
            )
        if self.dialect.operator_names and name.text in OPERATOR_NAMES:
            raise PreprocessorError(
                f"C++ operator '{name.text}' cannot be a macro name",
                span=name.span(),
                code=DEFINITION_CODE,
            )
        for prefix in self.dialect.reserved_prefixes:
            if name.text.startswith(prefix):
                raise PreprocessorError(
                    f"cannot {action} reserved macro name '{name.text}'",
                    span=name.span(),
                    code=DEFINITION_CODE,
                )

    # #include ----------------------------------------------------------------------------------------
    def _include_enabled(self) -> bool:
        return bool(self.enabled_extensions & self.dialect.include_extensions)

    def _header_name(self, tokens: list[Token], anchor: Token) -> tuple[str, bool]:
        if (
            len(tokens) == 1
            and tokens[0].kind == STRING
            and tokens[0].text.startswith('"')
        ):
            return tokens[0].text[1:-1], True
        if tokens and tokens[0].text == "<":
            closing = next((i for i, t in enumerate(tokens) if t.text == ">"), None)
            if closing is not None:
                first, last = tokens[0], tokens[closing]
                if (
                    first.source is not None
                    and first.source is last.source
                    and not first.hide
                    and not last.hide
                ):
                    return first.source.text[first.offset + 1 : last.offset], False
                return "".join(t.text for t in tokens[1:closing]), False
        raise PreprocessorError('expected "FILENAME" or <FILENAME>', span=anchor.span())

    def _resolve(
        self, header: str, quoted: bool, *, after: bool = False
    ) -> tuple[str, int | None] | None:
        if os.path.isabs(header):
            return (header, None) if self.fs.is_file(header) else None
        context = self.contexts[-1] if self.contexts else None
        start = 0
        if after and context is not None and context.found_in is not None:
            start = context.found_in + 1
        elif quoted and context is not None and not after:
            candidate = os.path.normpath(os.path.join(context.directory, header))
            if self.fs.is_file(candidate):
                return candidate, None
        for index in range(start, len(self.search)):
            candidate = os.path.normpath(os.path.join(self.search[index], header))
            if self.fs.is_file(candidate):
                return candidate, index
        return None

    def _include(
        self, args: list[Token], directive: Token, writer: Writer, *, after: bool
    ) -> None:
        tokens = args
        if not (args and (args[0].kind == STRING or args[0].text == "<")):
            tokens = self.expander.expand(args)
        header, quoted = self._header_name(tokens, directive)
        found = self._resolve(header, quoted, after=after)
        if found is None:
            if header in self.dialect.retained_includes:
                info = self.dialect.headers.get(header)
                if info is not None:
                    self._enter_header(header, info, directive)
                if header not in self.retained:
                    self.retained.append(header)
                writer.directive(
                    [Token(PUNCT, "#", directive.source, directive.offset), directive]
                    + lex_text(
                        f"<{header}>" if not quoted else f'"{header}"', mode=self.mode
                    )
                )
                return
            raise PreprocessorError(
                f"'{header}' file not found", span=directive.span(), code=INCLUDE_CODE
            )
        path, index = found
        identity = self.fs.identity(path)
        if identity in self.once:
            return
        if path not in self.includes:
            self.includes.append(path)
        source = SourceFile(
            path, self.fs.read(path), splice_whitespace=self.dialect.splice_whitespace
        )
        self._process(source, os.path.dirname(path), index, writer)

    def _enter_header(self, header: str, info: HeaderMacros, directive: Token) -> None:
        """Apply what including a compiler header does to the macro table, as recorded from the compiler."""
        if info.refusal is not None:
            raise PreprocessorError(
                info.refusal, span=directive.span(), code=HEADER_CODE
            )
        if header in self.entered:
            return  # its include guard makes a second inclusion empty
        self._check_conditions(info.conditions, f"<{header}>", directive)
        for definition in info.defines:
            tokens = lex_text(definition, mode=self.mode)
            if tokens[0].text in self.header_defined:
                continue  # defined by a header already entered, whose guard keeps it from being redefined
            self._define(tokens, Token(PUNCT, "#"), checked=False)
            self.pristine[tokens[0].text] = self.macros[tokens[0].text]
            self.header_defined.add(tokens[0].text)
        self.entered.update(info.closure)
        self.entered.add(header)

    def _check_conditions(self, names, what: str, directive: Token | None) -> None:
        """Refuse when the source changed a macro that a compiler header's conditional directives test."""
        for name in sorted(names):
            current, expected = self.macros.get(name), self.pristine.get(name)
            if current is not None and current.builtin is not None:
                continue
            if current is None and expected is None:
                continue
            if (
                current is not None
                and expected is not None
                and current.same_as(expected)
            ):
                continue
            change = (
                "undefined"
                if current is None
                else "defined" if expected is None else "redefined"
            )
            raise PreprocessorError(
                f"{what} test macro '{name}', which the source has {change}; the compiler header's expansion "
                f"under that change is not recorded",
                span=directive.span() if directive is not None else None,
                code=HEADER_CONFIGURATION_CODE,
            )

    # #line, #pragma, #extension, #version ------------------------------------------------------------
    def _line(self, args: list[Token], directive: Token) -> None:
        tokens = self.expander.expand(args)
        if not tokens or tokens[0].kind != NUMBER or not tokens[0].text.isdigit():
            raise PreprocessorError(
                "#line requires a positive integer", span=directive.span()
            )
        context = self.contexts[-1]
        last = args[-1]
        line, _ = (
            last.source.position(last.offset) if last.source is not None else (0, 0)
        )
        context.line_delta = int(tokens[0].text) - (line + 1)
        if len(tokens) > 1:
            if self.dialect.cpp and tokens[1].kind == STRING:
                context.presumed = tokens[1].text[1:-1]
            elif not self.dialect.cpp and tokens[1].kind == NUMBER:
                context.presumed = tokens[1].text
            else:
                raise PreprocessorError(
                    "invalid filename in #line", span=tokens[1].span()
                )

    def _pragma(self, args: list[Token], directive: Token) -> bool:
        """Applies ``#pragma once``; every other pragma is kept for the front end."""
        if args and args[0].text == "once" and len(args) == 1 and self.dialect.cpp:
            self.once.add(self.fs.identity(self.contexts[-1].source.path or "<main>"))
            return False
        return True

    def _extension(self, args: list[Token]) -> None:
        behaviour = args[2].text if len(args) >= 3 and args[1].text == ":" else None
        if behaviour in ("enable", "require", "warn"):
            self.enabled_extensions.add(args[0].text)
        elif behaviour == "disable":
            self.enabled_extensions.discard(args[0].text)

    def _version(self, args: list[Token]) -> None:
        if not args or args[0].kind != NUMBER:
            return
        self.define_text("__VERSION__", args[0].text)
        profile = args[1].text if len(args) > 1 else None
        if profile == "es":
            self.define_text("GL_ES", "1")
        elif int(args[0].text) >= 150 and profile != "compatibility":
            self.define_text("GL_core_profile", "1")
        elif profile == "compatibility":
            self.define_text("GL_compatibility_profile", "1")

    # Built-in macros ---------------------------------------------------------------------------------
    def _install_builtins(self) -> None:
        def line(token: Token) -> list[Token]:
            source, offset = token.end_location()
            context = self._context_of(source)
            number = (
                source.position(offset)[0] + (context.line_delta if context else 0)
                if source
                else 0
            )
            return [Token(NUMBER, str(number), token.source, token.offset, token.space)]

        def file(token: Token) -> list[Token]:
            context = self._context_of(token.source)
            presumed = context.presumed if context else None
            if not self.dialect.cpp:
                return [
                    Token(
                        NUMBER, presumed or "0", token.source, token.offset, token.space
                    )
                ]
            text = (
                '"'
                + (presumed or "<stdin>").replace("\\", "\\\\").replace('"', '\\"')
                + '"'
            )
            return [Token(STRING, text, token.source, token.offset, token.space)]

        def file_name(token: Token) -> list[Token]:
            context = self._context_of(token.source)
            presumed = (context.presumed if context else None) or "<stdin>"
            text = '"' + re.split(r"[\\/]", presumed)[-1].replace('"', '\\"') + '"'
            return [Token(STRING, text, token.source, token.offset, token.space)]

        builtins: dict[str, Callable[[Token], list[Token]]] = {
            "__LINE__": line,
            "__FILE__": file,
        }
        if self.dialect.file_name_macro:
            builtins["__FILE_NAME__"] = file_name
        if self.dialect.cpp:

            def counter(token: Token) -> list[Token]:
                value = self.counter
                self.counter += 1
                return [
                    Token(NUMBER, str(value), token.source, token.offset, token.space)
                ]

            def level(token: Token) -> list[Token]:
                return [
                    Token(
                        NUMBER,
                        str(max(0, len(self.contexts) - 1)),
                        token.source,
                        token.offset,
                        token.space,
                    )
                ]

            def base(token: Token) -> list[Token]:
                text = (
                    '"'
                    + (self.main_path or "<stdin>")
                    .replace("\\", "\\\\")
                    .replace('"', '\\"')
                    + '"'
                )
                return [Token(STRING, text, token.source, token.offset, token.space)]

            def refuse(token: Token) -> list[Token]:
                raise PreprocessorError(
                    f"{token.text} makes the output depend on the build time and is not supported",
                    span=token.span(),
                    code=BUILTIN_CODE,
                )

            def outside_if(token: Token) -> list[Token]:
                raise PreprocessorError(
                    f"{token.text} can only be used in #if and #elif",
                    span=token.span(),
                    code=BUILTIN_CODE,
                )

            builtins.update(
                {
                    "__COUNTER__": counter,
                    "__INCLUDE_LEVEL__": level,
                    "__BASE_FILE__": base,
                }
            )
            builtins.update({name: refuse for name in _NONDETERMINISTIC})
            for name in ("__has_include", "__has_include_next", *CLANG_FEATURE_TESTS):
                builtins[name] = outside_if
        for name, function in builtins.items():
            self.macros[name] = Macro(name, builtin=function)

    def _context_of(self, source) -> _Context | None:
        for context in reversed(self.contexts):
            if context.source is source:
                return context
        return None

    def _warn(self, code: str, message: str, token: Token) -> None:
        self.diagnostics.append(Diagnostic("warning", code, message, token.span()))
