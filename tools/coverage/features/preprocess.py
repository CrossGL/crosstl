"""The parts of the C preprocessor a header's function definitions depend on.

Some headers define their functions in conditional blocks, for a range of target architectures or compiler versions,
and through function-like macros that stamp out one definition per type, such as Clang's ``__MAKE_SYNC_SHUFFLES``.
:func:`keep` drops the branches a compile with the given macros does not see, and :func:`expand` replaces each
function-like macro invocation with its body, so that a pattern over the result finds every definition the compile
sees. Neither handles more of the preprocessor than such headers use: object-like macros only count as defined, and
``#include`` is ignored.
"""

from __future__ import annotations

import ast
import re

_DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*?)\s*$")
_DEFINED = re.compile(r"\bdefined\s*(?:\(\s*(\w+)\s*\)|(\w+))")
_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_INTEGER = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b")
_MACRO = re.compile(r"^\s*#\s*define\s+(\w+)\(([^)]*)\)(.*)$")
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
# A token of a macro body: the paste operator, a stringified parameter, or a word.
_TOKEN = re.compile(r"(##)|#\s*(\w+)|\b(\w+)\b")
# The operators a condition may use, after its C spellings are rewritten as Python's.
_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.Constant,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


def _value(condition: str, macros: dict[str, int]) -> bool:
    """Whether an ``#if`` condition holds; an identifier that names no macro is 0, as in C."""
    text = _DEFINED.sub(
        lambda m: "1" if (m.group(1) or m.group(2)) in macros else "0", condition
    )
    text = _INTEGER.sub(lambda m: str(int(m.group(1), 0)), text)
    text = _IDENTIFIER.sub(lambda m: str(macros.get(m.group(0), 0)), text)
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    tree = ast.parse(text.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"unsupported preprocessor condition: {condition}")
    return bool(eval(compile(tree, "<condition>", "eval"), {"__builtins__": {}}))


def keep(text: str, macros: dict[str, int]) -> str:
    """``text`` without comments and without the conditional branches a compile with ``macros`` defined drops.

    ``#define`` and ``#undef`` of an object-like macro in a kept branch update the macros later conditions see.
    """
    macros = dict(macros)
    kept: list[str] = []
    # One entry per open conditional: whether its current branch is kept, and whether an earlier branch was.
    stack: list[tuple[bool, bool]] = []
    for line in _COMMENTS.sub("", text).replace("\\\n", "\x00").splitlines():
        directive = _DIRECTIVE.match(line.replace("\x00", " "))
        live = all(branch for branch, _ in stack)
        name, rest = directive.groups() if directive else ("", "")
        if name in ("if", "ifdef", "ifndef"):
            holds = (
                _value(rest, macros)
                if name == "if"
                else (rest in macros) == (name == "ifdef")
            )
            stack.append((live and holds, holds))
        elif name == "elif":
            parent = all(branch for branch, _ in stack[:-1])
            taken = stack[-1][1]
            holds = not taken and _value(rest, macros)
            stack[-1] = (parent and holds, taken or holds)
        elif name == "else":
            parent = all(branch for branch, _ in stack[:-1])
            stack[-1] = (parent and not stack[-1][1], True)
        elif name == "endif":
            stack.pop()
        elif live:
            if name == "define" and "(" not in rest.split()[0]:
                word = rest.split()
                macros[word[0]] = (
                    int(word[1], 0) if len(word) > 1 and word[1].isdigit() else 1
                )
            elif name == "undef":
                macros.pop(rest, None)
            kept.append(line)
    return "\n".join(kept).replace("\x00", "\\\n")


def _arguments(text: str, start: int) -> tuple[list[str], int]:
    """The comma-separated arguments of the invocation whose ``(`` is at ``start``, and the index after its ``)``."""
    depth, quoted, current, found = 0, "", [], []
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            quoted = "" if char == quoted and text[index - 1] != "\\" else quoted
        elif char in "\"'":
            quoted = char
        elif char == "(":
            depth += 1
            if depth == 1:
                continue
        elif char == ")":
            depth -= 1
            if depth == 0:
                found.append("".join(current).strip())
                return found, index + 1
        elif char == "," and depth == 1:
            found.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    raise ValueError("unterminated macro invocation")


def _substitute(match: re.Match[str], values: dict[str, str]) -> str:
    """One token of a macro body with the invocation's arguments in place of the parameters."""
    pasted, stringified, word = match.groups()
    if pasted:
        return pasted
    if stringified:
        return f'"{values[stringified]}"' if stringified in values else match.group(0)
    return values.get(word, word)


def expand(text: str) -> str:
    """``text`` with its function-like macros' definitions removed and every invocation of one replaced by its body.

    A parameter is substituted as a whole word, and ``##`` then pastes the tokens on either side of it. An invocation
    in a body is expanded after substitution, as the preprocessor rescans it.
    """
    macros: dict[str, tuple[list[str], str]] = {}
    lines = []
    for line in text.replace("\\\n", " ").splitlines():
        definition = _MACRO.match(line)
        if definition:
            parameters = [
                p.strip() for p in definition.group(2).split(",") if p.strip()
            ]
            macros[definition.group(1)] = (parameters, definition.group(3))
        elif not line.lstrip().startswith("#"):
            lines.append(line)
    text = "\n".join(lines)
    if not macros:
        return text
    invocation = re.compile(r"\b(" + "|".join(map(re.escape, macros)) + r")\s*(?=\()")
    for _ in range(8):
        match = invocation.search(text)
        if not match:
            return text
        parts = []
        position = 0
        while match:
            arguments, end = _arguments(text, match.end())
            parameters, body = macros[match.group(1)]
            values = dict(zip(parameters, arguments))
            body = _TOKEN.sub(lambda m: _substitute(m, values), body)
            parts.append(text[position : match.start()] + re.sub(r"\s*##\s*", "", body))
            position = end
            match = invocation.search(text, end)
        text = "".join(parts) + text[position:]
    raise ValueError("macro expansion does not terminate")
