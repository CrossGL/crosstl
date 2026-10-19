"""Constant expressions of ``#if`` and ``#elif``: C++ [cpp.cond] and GLSL section 3.3.

C and C++ evaluate in ``intmax_t`` and ``uintmax_t`` with the usual arithmetic conversions: 64 bits for Clang and
DXC, 128 bits for the Metal compiler (``__INTMAX_WIDTH__`` is 128 for AIR); GLSL evaluates in 32-bit ``int``. In
C++ the operator names ``and``, ``or``, ``not`` and the rest ([lex.digraph]) are operators, not identifiers. Signed overflow wraps, as Clang and glslang do after a
warning. Operands of ``&&``, ``||`` and the unselected arm of ``?:`` are parsed but not evaluated, so a division
by zero there is not an error. Anything the target compiler would reject is refused with a typed diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..diagnostics import TranslationDiagnostic
from .lexer import CHAR, IDENT, NUMBER, PUNCT, Token

EXPRESSION_CODE = "project.translate.preprocessor-if-expression-invalid"
DIVISION_CODE = "project.translate.preprocessor-if-division-by-zero"


@dataclass(frozen=True)
class Value:
    value: int
    unsigned: bool = False


class ExpressionError(TranslationDiagnostic):
    code = EXPRESSION_CODE


_BINARY = {
    # operator: (precedence, ...) - higher binds tighter.
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    ">": 7,
    "<=": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}
_LENGTH = r"(?:ll|LL|[lL]|[zZ])"
_CPP_INTEGER = re.compile(
    r"(0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)([uU]?"
    + _LENGTH
    + r"?|"
    + _LENGTH
    + r"[uU])"
)
_GLSL_INTEGER = re.compile(r"(0[xX][0-9a-fA-F]+|[0-9]+)([uU]?)")
_ESCAPES = {
    "n": 10,
    "t": 9,
    "v": 11,
    "b": 8,
    "r": 13,
    "f": 12,
    "a": 7,
    "\\": 92,
    "'": 39,
    '"': 34,
    "?": 63,
}


class _Evaluator:
    def __init__(
        self, tokens: Sequence[Token], width: int, cpp: bool, directive: Token
    ):
        self.tokens = list(tokens)
        self.index = 0
        self.width = width
        self.cpp = cpp
        self.directive = directive
        self.mask = (1 << width) - 1

    # Value helpers -----------------------------------------------------------------------------------
    def fail(self, message, token=None, code=EXPRESSION_CODE):
        anchor = (
            token if token is not None and token.source is not None else self.directive
        )
        raise ExpressionError(
            message, span=anchor.span() if anchor else None, code=code
        )

    def wrap(self, value: int, unsigned: bool) -> Value:
        value &= self.mask
        if not unsigned and value >> (self.width - 1):
            value -= 1 << self.width
        return Value(value, unsigned)

    def boolean(self, flag: bool) -> Value:
        return Value(1 if flag else 0)

    # Token stream ------------------------------------------------------------------------------------
    def peek(self) -> Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> Token:
        token = self.peek()
        if token is None:
            self.fail("expected a value in preprocessor expression")
        self.index += 1
        return token

    def expect(self, text: str):
        token = self.peek()
        if token is None or token.text != text:
            self.fail(f"expected '{text}' in preprocessor expression", token)
        self.index += 1

    # Grammar -----------------------------------------------------------------------------------------
    def parse(self) -> Value:
        if not self.tokens:
            self.fail("#if with no expression")
        value = self.comma(True)
        if self.peek() is not None:
            self.fail(
                f"token '{self.peek().text}' is not valid in a preprocessor expression",
                self.peek(),
            )
        return value

    def comma(self, live: bool) -> Value:
        value = self.conditional(live)
        while self.cpp and self.peek() is not None and self.peek().text == ",":
            self.index += 1
            value = self.conditional(live)
        return value

    def conditional(self, live: bool) -> Value:
        condition = self.binary(1, live)
        token = self.peek()
        if token is None or token.text != "?":
            return condition
        self.index += 1
        chosen = live and condition.value != 0
        first = self.comma(chosen)
        self.expect(":")
        second = self.conditional(live and not chosen)
        unsigned = first.unsigned or second.unsigned
        picked = first if condition.value != 0 else second
        return self.wrap(picked.value, unsigned) if live else Value(0, unsigned)

    def binary(self, minimum: int, live: bool) -> Value:
        left = self.unary(live)
        while True:
            token = self.peek()
            precedence = (
                _BINARY.get(token.text)
                if token is not None and token.kind == PUNCT
                else None
            )
            if precedence is None or precedence < minimum:
                return left
            self.index += 1
            if token.text == "&&":
                right = self.binary(precedence + 1, live and left.value != 0)
                left = self.boolean(left.value != 0 and right.value != 0)
            elif token.text == "||":
                right = self.binary(precedence + 1, live and left.value == 0)
                left = self.boolean(left.value != 0 or right.value != 0)
            else:
                right = self.binary(precedence + 1, live)
                left = self.apply(token, left, right, live)

    def apply(self, token: Token, left: Value, right: Value, live: bool) -> Value:
        operator = token.text
        if operator in ("<<", ">>"):
            if not live:
                return Value(0, left.unsigned)
            count = right.value
            if count < 0 or count >= self.width:
                self.fail(
                    f"shift count {count} is out of range in preprocessor expression",
                    token,
                )
            if operator == "<<":
                return self.wrap(left.value << count, left.unsigned)
            return self.wrap(left.value >> count, left.unsigned)
        unsigned = left.unsigned or right.unsigned
        a = left.value & self.mask if unsigned else left.value
        b = right.value & self.mask if unsigned else right.value
        if operator in ("==", "!=", "<", ">", "<=", ">="):
            return self.boolean(
                {
                    "==": a == b,
                    "!=": a != b,
                    "<": a < b,
                    ">": a > b,
                    "<=": a <= b,
                    ">=": a >= b,
                }[operator]
            )
        if not live:
            return Value(0, unsigned)
        if operator in ("/", "%"):
            if b == 0:
                self.fail(
                    "division by zero in preprocessor expression", token, DIVISION_CODE
                )
            quotient = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
            return self.wrap(
                quotient if operator == "/" else a - b * quotient, unsigned
            )
        result = {
            "+": lambda: a + b,
            "-": lambda: a - b,
            "*": lambda: a * b,
            "&": lambda: a & b,
            "|": lambda: a | b,
            "^": lambda: a ^ b,
        }[operator]()
        return self.wrap(result, unsigned)

    def unary(self, live: bool) -> Value:
        token = self.take()
        if token.kind == PUNCT and token.text in ("+", "-", "~", "!"):
            operand = self.unary(live)
            if token.text == "!":
                return self.boolean(operand.value == 0)
            if token.text == "+":
                return operand
            if token.text == "-":
                return self.wrap(-operand.value, operand.unsigned)
            return self.wrap(~operand.value, operand.unsigned)
        if token.kind == PUNCT and token.text == "(":
            value = self.comma(live)
            self.expect(")")
            return value
        if token.kind == NUMBER:
            return self.number(token)
        if token.kind == CHAR and self.cpp:
            return self.character(token)
        if token.kind == IDENT:
            if self.cpp and token.text in ("true", "false"):
                return self.boolean(token.text == "true")
            # [cpp.cond]/11: an identifier left after macro expansion is replaced by 0.
            return Value(0)
        self.fail(
            f"token '{token.text}' is not valid in a preprocessor expression", token
        )

    def number(self, token: Token) -> Value:
        text = token.text.replace("'", "") if self.cpp else token.text
        match = (_CPP_INTEGER if self.cpp else _GLSL_INTEGER).fullmatch(text)
        if match is None:
            self.fail(
                f"invalid integer literal '{token.text}' in preprocessor expression",
                token,
            )
        digits, suffix = match.group(1).lower(), match.group(2).lower()
        if digits.startswith("0x"):
            value = int(digits[2:], 16)
        elif digits.startswith("0b"):
            value = int(digits[2:], 2)
        elif digits.startswith("0") and len(digits) > 1:
            if not set(digits) <= set("01234567"):
                self.fail(f"invalid digit in octal literal '{token.text}'", token)
            value = int(digits, 8)
        else:
            value = int(digits, 10)
        unsigned = "u" in suffix
        if value > self.mask:
            self.fail(
                f"integer literal '{token.text}' is too large for the preprocessor",
                token,
            )
        if not unsigned and value >> (self.width - 1):
            if not self.cpp:
                self.fail(f"integer literal '{token.text}' does not fit in int", token)
            unsigned = True  # Clang and GCC read an over-large signed literal as unsigned, with a warning.
        return Value(value, unsigned)

    def character(self, token: Token) -> Value:
        text = token.text
        prefix = text[: text.index("'")]
        body = text[len(prefix) + 1 : -1]
        values, position = [], 0
        while position < len(body):
            char = body[position]
            if char != "\\":
                values.append(ord(char))
                position += 1
                continue
            escape = body[position + 1 : position + 2]
            if escape in _ESCAPES:
                values.append(_ESCAPES[escape])
                position += 2
            elif escape == "x":
                end = position + 2
                while end < len(body) and body[end] in "0123456789abcdefABCDEF":
                    end += 1
                if end == position + 2:
                    self.fail("\\x used with no hexadecimal digits", token)
                values.append(int(body[position + 2 : end], 16))
                position = end
            elif escape and escape in "01234567":
                end = position + 1
                while (
                    end < len(body) and end < position + 4 and body[end] in "01234567"
                ):
                    end += 1
                values.append(int(body[position + 1 : end], 8))
                position = end
            else:
                self.fail(f"unsupported escape sequence in '{text}'", token)
        if not values:
            self.fail("empty character constant", token)
        if prefix:
            if len(values) != 1:
                self.fail(
                    f"multi-character constant '{text}' with an encoding prefix", token
                )
            return Value(values[0])
        if len(values) == 1:
            value = values[0] & 0xFF
            return Value(
                value - 256 if value >= 128 else value
            )  # plain char is signed on every target here
        # Clang: a multi-character constant is an int built from its bytes, most significant first.
        value = 0
        for item in values[-4:]:
            value = (value << 8) | (item & 0xFF)
        return Value(value - (1 << 32) if value >> 31 else value)


# [lex.digraph]: the alternative spellings of operators that can occur in a preprocessor expression.
OPERATOR_NAMES = {
    "and": "&&",
    "or": "||",
    "not": "!",
    "bitand": "&",
    "bitor": "|",
    "xor": "^",
    "compl": "~",
    "not_eq": "!=",
    "and_eq": "&=",
    "or_eq": "|=",
    "xor_eq": "^=",
}


def evaluate(
    tokens: list[Token],
    *,
    width: int,
    cpp: bool,
    directive: Token,
    operator_names: bool = False,
) -> bool:
    """Truth of an already macro-expanded ``#if`` expression."""
    if operator_names:
        tokens = [
            (
                token.copy(kind=PUNCT, text=OPERATOR_NAMES[token.text])
                if token.kind == IDENT and token.text in OPERATOR_NAMES
                else token
            )
            for token in tokens
        ]
    return _Evaluator(tokens, width, cpp, directive).parse().value != 0
