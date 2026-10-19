"""Preprocessor translation units for crash fuzzing: macro definitions and invocations, conditionals, includes.

A unit is a main file plus up to three headers in a memory file system. Lines are drawn from a grammar of
directives (object-like, function-like and variadic macros with ``#``, ``##`` and ``__VA_OPT__``; ``#if``
arithmetic with ``defined``; ``#include`` with ``#pragma once`` or guards; ``#line``, ``#pragma``, ``_Pragma``,
``#undef`` and ``#error``) and of text lines that invoke the macros. Conditionals are usually balanced, and a
small share of units are deliberately ill-formed, because the preprocessor must refuse those with a typed
diagnostic rather than fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

DIALECTS = ("c++", "glsl", "msl", "hlsl")
MACROS = ("A", "B", "F", "G", "CAT", "STR", "V", "__LINE__", "__FILE__", "__COUNTER__")
HEADERS = ("h0.h", "h1.h", "sub/h2.h")
# fmt: off
PUNCTUATION = (
    "+", "-", "*", "/", "%", "<<", ">>", "<", ">", "<=", ">=", "==", "!=", "&", "|", "^", "&&", "||",
    "?", ":", "(", ")", ",", "!", "~", "#", "##", "...", ";", "{", "}", "[", "]", ".", "->", "<:", "%:",
)
ATOMS = (
    "0", "1", "2", "-1", "0x7fffffffffffffff", "0xffffffffffffffffu", "18446744073709551615",
    "9223372036854775807", "1'000", "07", "0b101", "'a'", "'\\n'", "u8'a'", "\"s\"", "L\"w\"", "1.5",
    "x", "y", "true", "false", "defined", "__VA_ARGS__",
)
# fmt: on


@dataclass(frozen=True)
class Unit:
    dialect: str
    main: str
    headers: dict[str, str]

    def to_json(self):
        return {
            "dialect": self.dialect,
            "main": self.main,
            "headers": dict(self.headers),
        }


def _expression(draw, depth=2):
    if depth <= 0 or draw(st.integers(0, 2)) == 0:
        return draw(
            st.one_of(
                st.sampled_from(ATOMS[:10] + ("A", "B", "x")),
                st.sampled_from(MACROS[:3]).map(lambda name: f"defined({name})"),
                st.sampled_from(MACROS[:3]).map(lambda name: f"defined {name}"),
                st.integers(-(2**64), 2**64).map(str),
            )
        )
    form = draw(st.integers(0, 3))
    left = _expression(draw, depth - 1)
    if form == 0:
        return f"({draw(st.sampled_from(('!', '~', '-', '+')))}{left})"
    if form == 1:
        right = _expression(draw, depth - 1)
        other = _expression(draw, depth - 1)
        return f"({left} ? {right} : {other})"
    op = draw(st.sampled_from(PUNCTUATION[:18]))
    return f"({left} {op} {_expression(draw, depth - 1)})"


def _text(draw):
    pieces = []
    for _ in range(draw(st.integers(0, 8))):
        form = draw(st.integers(0, 5))
        if form == 0:
            pieces.append(draw(st.sampled_from(MACROS)))
        elif form == 1:
            name = draw(st.sampled_from(("F", "G", "CAT", "STR", "V")))
            args = draw(
                st.lists(
                    st.sampled_from(ATOMS + MACROS[:3] + ("", "(a, b)")), max_size=4
                )
            )
            pieces.append(f"{name}({', '.join(args)})")
        elif form == 2:
            pieces.append(draw(st.sampled_from(PUNCTUATION)))
        elif form == 3:
            pieces.append(draw(st.sampled_from(ATOMS)))
        elif form == 4:
            pieces.append(
                draw(st.sampled_from(("/* c */", "// c", "\\\n", '_Pragma("once")')))
            )
        else:
            pieces.append(draw(st.sampled_from(("F", "G", "(", ")", "F("))))
    return " ".join(pieces)


def _define(draw):
    name = draw(st.sampled_from(MACROS[:7]))
    form = draw(st.integers(0, 4))
    body = _text(draw)
    if form == 0:
        return f"#define {name} {body}"
    if form == 1:
        return f"#define {name}(x, y) x ## y {body}"
    if form == 2:
        return f"#define {name}(x) #x {body}"
    if form == 3:
        return f"#define {name}(x, ...) x __VA_OPT__(,) __VA_ARGS__ {body}"
    params = draw(
        st.sampled_from(("()", "(x)", "(x, x)", "(...)", "(x,", "(1)", "(x...)"))
    )
    return f"#define {name}{params} {body}"


def _lines(draw, depth, headers):
    lines = []
    for _ in range(draw(st.integers(0, 6))):
        form = draw(st.integers(0, 9))
        if form <= 2:
            lines.append(_define(draw))
        elif form == 3:
            lines.append(f"#undef {draw(st.sampled_from(MACROS[:7]))}")
        elif form == 4 and depth > 0:
            opener = draw(
                st.sampled_from(
                    ("#if {}", "#ifdef A", "#ifndef B", "#if defined(F) && {}")
                )
            )
            lines.append(opener.format(_expression(draw)))
            lines += _lines(draw, depth - 1, headers)
            if draw(st.booleans()):
                lines.append(f"#elif {_expression(draw)}")
                lines += _lines(draw, depth - 1, headers)
            if draw(st.booleans()):
                lines.append("#else")
                lines += _lines(draw, depth - 1, headers)
            lines.append("#endif")
        elif form == 5 and headers:
            quoted = draw(st.booleans())
            header = draw(st.sampled_from(headers + ("missing.h",)))
            lines.append(f'#include "{header}"' if quoted else f"#include <{header}>")
        elif form == 6:
            lines.append(
                draw(
                    st.sampled_from(
                        (
                            "#line 10",
                            '#line 20 "other.h"',
                            "#line A",
                            "#pragma once",
                            "#pragma STDC FP_CONTRACT OFF",
                            '#pragma push_macro("A")',
                            '#pragma pop_macro("A")',
                            "#",
                            "#error stop",
                            "#warning careful",
                            "#elif 1",
                            "#endif",
                            "#ifdef",
                            "#unknown",
                            "#include",
                            "#include MACRO",
                        )
                    )
                )
            )
        else:
            lines.append(_text(draw))
    return lines


@st.composite
def units(draw):
    """The hypothesis strategy of preprocessor units."""
    dialect = draw(st.sampled_from(DIALECTS))
    headers = {}
    for name in HEADERS[: draw(st.integers(0, len(HEADERS)))]:
        guard = draw(st.sampled_from(("pragma", "guard", "none")))
        body = _lines(draw, 1, HEADERS)
        if guard == "pragma":
            body = ["#pragma once"] + body
        elif guard == "guard":
            macro = "GUARD_" + name.replace("/", "_").replace(".", "_").upper()
            body = [f"#ifndef {macro}", f"#define {macro}"] + body + ["#endif"]
        headers[name] = "\n".join(body) + "\n"
    prefix = ["#version 450"] if dialect == "glsl" and draw(st.booleans()) else []
    main = "\n".join(prefix + _lines(draw, 3, tuple(headers))) + "\n"
    return Unit(dialect, main, headers)
