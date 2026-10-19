"""Macro definitions and macro replacement: C++ [cpp.replace], C 6.10.3 and GLSL section 3.3.

Replacement follows the hide-set formulation of the standard's rescanning rules (D. F. Prosser, as used by GCC):
every token carries the set of macro names it may not invoke. An object-like expansion adds the macro name to the
hide set of every token it produces; a function-like expansion uses the intersection of the name's and the closing
parenthesis's hide sets plus the name. A token whose hide set names its own macro is never expanded again, which
is the standard's "painted blue". Arguments are fully expanded before substitution unless they are operands of
``#`` or ``##``; ``##`` works on placemarkers for empty arguments, and ``__VA_OPT__`` follows C++20 [cpp.subst].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..diagnostics import SourceSpan, TranslationDiagnostic
from .lexer import (
    CHAR,
    IDENT,
    NEWLINE,
    PUNCT,
    STRING,
    LexMode,
    Token,
    lex_text,
    resolve_mode,
)

ARGUMENTS_CODE = "project.translate.preprocessor-macro-arguments-invalid"
PASTE_CODE = "project.translate.preprocessor-invalid-paste"
STRINGIZE_CODE = "project.translate.preprocessor-invalid-stringize"
DEFINITION_CODE = "project.translate.preprocessor-invalid-macro-definition"

PLACEMARKER = "placemarker"
PASTE = "paste-operator"  # a '##' of the replacement list, as opposed to a '##' token passed as an argument


class MacroError(TranslationDiagnostic):
    code = ARGUMENTS_CODE


class UnterminatedInvocation(MacroError):
    """The input ended inside the argument list of a function-like macro invocation."""


@dataclass
class Macro:
    """A macro. ``params`` is ``None`` for an object-like macro; a variadic macro's last parameter is its
    variable argument (``__VA_ARGS__`` or a GNU named one). ``builtin`` computes the replacement of a
    dynamic macro such as ``__LINE__`` from the invoking token."""

    name: str
    body: list[Token] = field(default_factory=list)
    params: list[str] | None = None
    variadic: bool = False
    span: SourceSpan | None = None
    builtin: Callable[[Token], list[Token]] | None = None

    @property
    def function_like(self) -> bool:
        return self.params is not None

    def same_as(self, other: Macro) -> bool:
        """[cpp.replace]/2: identical parameters and replacement lists, whitespace compared as present or not."""
        if (self.params, self.variadic) != (other.params, other.variadic):
            return False
        if len(self.body) != len(other.body):
            return False
        return all(
            a.text == b.text and (index == 0 or a.space == b.space)
            for index, (a, b) in enumerate(zip(self.body, other.body))
        )


def stringize(tokens: Sequence[Token], anchor: Token) -> Token:
    """[cpp.stringize]: spell the argument with single spaces, escaping quotes and backslashes in literals."""
    parts = []
    for index, token in enumerate(tokens):
        if token.kind == PLACEMARKER:
            continue
        if parts and token.space:
            parts.append(" ")
        if token.kind in (STRING, CHAR):
            parts.append(token.text.replace("\\", "\\\\").replace('"', '\\"'))
        else:
            parts.append(token.text)
    text = '"' + "".join(parts) + '"'
    lexed = lex_text(text)
    if len(lexed) != 1 or lexed[0].kind != STRING:
        raise MacroError(
            f"stringizing produced an invalid string literal {text}",
            span=anchor.span(),
            code=STRINGIZE_CODE,
        )
    return Token(STRING, text, anchor.source, anchor.offset, anchor.space)


class Expander:
    """Expands token sequences against a macro table. ``on_builtin`` hooks, the include-level state and
    ``__COUNTER__`` live in the engine, which owns this object."""

    def __init__(
        self,
        macros: dict[str, Macro],
        *,
        cpp: bool,
        gnu_comma: bool,
        va_opt: bool = True,
        mode: LexMode | None = None,
    ):
        self.macros = macros
        self.cpp = cpp
        self.gnu_comma = gnu_comma
        # Without __VA_OPT__ (C++20, and Clang before 6), the name is an ordinary identifier.
        self.va_opt = va_opt and cpp
        self.mode = resolve_mode(cpp, mode)

    # Argument collection -----------------------------------------------------------------------------
    def _collect(self, macro: Macro, name: Token, stack: list[Token]):
        """Pop the arguments of ``macro`` from ``stack`` (input reversed); ``None`` when no '(' follows."""
        probe = len(stack) - 1
        while probe >= 0 and stack[probe].kind == NEWLINE:
            probe -= 1
        if probe < 0 or stack[probe].kind != PUNCT or stack[probe].text != "(":
            return None
        del stack[probe:]
        arguments: list[list[Token]] = [[]]
        depth, pending_space = 0, False
        count = len(macro.params)
        while True:
            if not stack:
                raise UnterminatedInvocation(
                    f"unterminated argument list invoking macro '{macro.name}'",
                    span=name.span(),
                )
            token = stack.pop()
            if token.kind == NEWLINE:
                pending_space = True
                continue
            if pending_space:
                token = token.copy(space=True)
                pending_space = False
            if token.kind == PUNCT:
                if token.text == "(":
                    depth += 1
                elif token.text == ")":
                    if depth == 0:
                        return arguments, token
                    depth -= 1
                elif token.text == "," and depth == 0:
                    if not (macro.variadic and len(arguments) >= count):
                        arguments.append([])
                        continue
            arguments[-1].append(token)

    def _bind(self, macro: Macro, name: Token, arguments: list[list[Token]]):
        count = len(macro.params)
        if count == 0 and arguments == [[]]:
            return {}
        if macro.variadic and len(arguments) == count - 1:
            arguments = arguments + [None]  # the variable argument is absent, not empty
        if len(arguments) != count:
            raise MacroError(
                f"macro '{macro.name}' requires {count} argument{'s' if count != 1 else ''}, "
                f"but {len(arguments)} {'were' if len(arguments) != 1 else 'was'} given",
                span=name.span(),
            )
        bound = {}
        for param, argument in zip(macro.params, arguments):
            if argument is not None and argument:
                argument = [argument[0].copy(space=False)] + argument[1:]
            bound[param] = argument
        return bound

    # Substitution ------------------------------------------------------------------------------------
    def _substitute(
        self,
        macro: Macro,
        body: Sequence[Token],
        bound,
        origin: Token,
        *,
        keep_placemarkers=False,
    ):
        """Replacement list with parameters substituted, '#' and '##' applied; placemarkers stand for empty
        operands and are removed at the end, except inside ``__VA_OPT__``, whose result still takes part in an
        enclosing '##' ([cpp.subst]/3)."""
        out: list[Token] = []
        index = 0
        params = bound.keys()

        def raw(param):
            argument = bound[param]
            return list(argument) if argument else []

        def operand_of_paste(position):
            before = out[-1] if out else None
            after = body[position + 1] if position + 1 < len(body) else None
            return (before is not None and before.kind == PASTE) or (
                after is not None
                and after.kind == PUNCT
                and after.text in ("##", "%:%:")
            )

        while index < len(body):
            token = body[index]
            if (
                macro.function_like
                and token.kind == PUNCT
                and token.text in ("#", "%:")
            ):
                target = body[index + 1] if index + 1 < len(body) else None
                if (
                    target is not None
                    and target.text == "__VA_OPT__"
                    and macro.variadic
                    and self.va_opt
                ):
                    content, index = self._va_opt_content(macro, body, index + 1)
                    produced = self._va_opt(macro, content, bound, origin)
                    out.append(stringize(produced, token.copy(space=token.space)))
                    continue
                if target is None or target.kind != IDENT or target.text not in params:
                    raise MacroError(
                        "'#' is not followed by a macro parameter",
                        span=token.span(),
                        code=DEFINITION_CODE,
                    )
                out.append(stringize(raw(target.text), token))
                index += 2
                continue
            if (
                token.kind == IDENT
                and token.text == "__VA_OPT__"
                and macro.variadic
                and self.va_opt
            ):
                content, after = self._va_opt_content(macro, body, index)
                produced = self._va_opt(macro, content, bound, origin)
                if not produced:
                    produced = [
                        Token(PLACEMARKER, "", token.source, token.offset, token.space)
                    ]
                else:
                    produced[0] = produced[0].copy(space=token.space)
                out.extend(produced)
                index = after
                continue
            if token.kind == PUNCT and token.text in ("##", "%:%:"):
                out.append(Token(PASTE, "##", token.source, token.offset, token.space))
                index += 1
                continue
            if token.kind == IDENT and token.text in params:
                param = token.text
                if (
                    self.gnu_comma
                    and macro.variadic
                    and param == macro.params[-1]
                    and len(out) >= 2
                    and out[-1].kind == PASTE
                    and out[-2].kind == PUNCT
                    and out[-2].text == ","
                ):
                    # GNU ', ## __VA_ARGS__': drop the comma when the variable argument is empty or absent.
                    out.pop()
                    if not bound[param]:
                        out.pop()
                    else:
                        out.extend(self._spaced(raw(param), token))
                    index += 1
                    continue
                if operand_of_paste(index):
                    produced = raw(param)
                else:
                    produced = self.expand(raw(param)) if bound[param] else []
                if not produced:
                    produced = [
                        Token(PLACEMARKER, "", token.source, token.offset, token.space)
                    ]
                out.extend(self._spaced(produced, token))
                index += 1
                continue
            out.append(token)
            index += 1
        pasted = self._paste_all(out)
        return (
            pasted
            if keep_placemarkers
            else [token for token in pasted if token.kind != PLACEMARKER]
        )

    @staticmethod
    def _spaced(tokens: list[Token], where: Token) -> list[Token]:
        if tokens:
            tokens = [tokens[0].copy(space=where.space)] + tokens[1:]
        return tokens

    def _va_opt_content(self, macro: Macro, body: Sequence[Token], index: int):
        """Tokens between the parentheses of the ``__VA_OPT__`` at ``index``, and the index after them."""
        opening = body[index + 1] if index + 1 < len(body) else None
        if opening is None or opening.text != "(":
            raise MacroError(
                "'__VA_OPT__' must be followed by '('",
                span=body[index].span(),
                code=DEFINITION_CODE,
            )
        depth, position = 0, index + 1
        while position < len(body):
            text = body[position].text
            depth += text == "("
            depth -= text == ")"
            if depth == 0:
                return list(body[index + 2 : position]), position + 1
            position += 1
        raise MacroError(
            "unterminated '__VA_OPT__'", span=body[index].span(), code=DEFINITION_CODE
        )

    def _va_opt(self, macro: Macro, content, bound, origin) -> list[Token]:
        """[cpp.subst]/3: the content when the variable argument has tokens after expansion, else nothing."""
        variable = bound[macro.params[-1]]
        if not variable or not self.expand(list(variable)):
            return []
        return self._substitute(macro, content, bound, origin, keep_placemarkers=True)

    def _paste_all(self, tokens: list[Token]) -> list[Token]:
        out: list[Token] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.kind == PASTE:
                if not out or index + 1 >= len(tokens):
                    raise MacroError(
                        "'##' cannot appear at either end of a macro expansion",
                        span=token.span(),
                        code=DEFINITION_CODE,
                    )
                out[-1] = self._paste(out[-1], tokens[index + 1])
                index += 2
                continue
            out.append(token)
            index += 1
        return out

    def _paste(self, left: Token, right: Token) -> Token:
        if right.kind == PLACEMARKER:
            return left
        if left.kind == PLACEMARKER:
            return right.copy(space=left.space)
        text = left.text + right.text
        lexed = lex_text(text, mode=self.mode)
        if len(lexed) != 1 or lexed[0].text != text:
            raise MacroError(
                f"pasting '{left.text}' and '{right.text}' does not give a valid preprocessing token",
                span=left.span(),
                code=PASTE_CODE,
            )
        return Token(lexed[0].kind, text, left.source, left.offset, left.space)

    # Rescanning --------------------------------------------------------------------------------------
    def expand(self, tokens: Sequence[Token], *, on_identifier=None) -> list[Token]:
        """Fully macro-replace ``tokens``. ``on_identifier`` may claim an identifier (``defined`` in ``#if``)
        by returning the tokens to emit for it, given the remaining reversed input."""
        stack = list(reversed(tokens))
        out: list[Token] = []
        while stack:
            token = stack.pop()
            if token.kind != IDENT or token.noexpand:
                out.append(token)
                continue
            if on_identifier is not None:
                claimed = on_identifier(token, stack)
                if claimed is not None:
                    out.extend(claimed)
                    continue
            macro = self.macros.get(token.text)
            if macro is None:
                out.append(token)
                continue
            if token.text in token.hide:
                painted = token.copy()
                painted.noexpand = True
                out.append(painted)
                continue
            if macro.builtin is not None:
                out.extend(macro.builtin(token))
                continue
            if not macro.function_like:
                hide = token.hide | {macro.name}
                end = token.end_location()
                produced = self._substitute(macro, macro.body, {}, token)
            else:
                collected = self._collect(macro, token, stack)
                if collected is None:
                    out.append(token)
                    continue
                arguments, closing = collected
                bound = self._bind(macro, token, arguments)
                hide = (token.hide & closing.hide) | {macro.name}
                end = closing.end_location()
                produced = self._substitute(macro, macro.body, bound, token)
            stack.extend(reversed(self._relocate(produced, token, hide, end)))
        return out

    @staticmethod
    def _relocate(produced: list[Token], origin: Token, hide, end) -> list[Token]:
        """Give every produced token the invocation's location, its end and the expansion's hide set."""
        result = []
        for index, token in enumerate(produced):
            copy = token.copy(
                source=origin.source,
                offset=origin.offset,
                hide=token.hide | hide,
                space=origin.space if index == 0 else token.space,
                end=end,
            )
            result.append(copy)
        return result
