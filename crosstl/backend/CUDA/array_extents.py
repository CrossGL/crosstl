"""Constant array extents in the CUDA and HIP importers.

CUDA and HIP size arrays with integral constant expressions: ``float t[N]``, ``float s[N * 2]``,
``__shared__ float tile[TILE][TILE + 1]``, where ``N`` and ``TILE`` are ``constexpr`` or ``const`` integers declared
at namespace or block scope ([dcl.array]/1: a converted constant expression). The importers wrote such an extent as
an invented identifier (``array<f32, N_mul_2>``, ``array<f32, cuda_array_extent_32_plus_1>``) that no program
declares, and a bare constant (``array<f32, N>``) as a name the CrossGL parser reads as a type, so every target
refused the array.

:class:`IntegerConstants` records the value of each integral constant as the importer visits its declaration, in the
importer's block scopes, so a later declaration of the same name (a parameter, a local variable, a template value
parameter) hides it as C++ does. :func:`crossgl_array_extent` folds an extent to its value with C++ integer
arithmetic ([expr.mul]/4: division truncates toward zero), so the array has the same length in every target. An
extent that is not a constant the importer can evaluate keeps the identifier spelling the importers wrote before.
"""

from __future__ import annotations

import re

from ..common_ast import (
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    TernaryOpNode,
    UnaryOpNode,
)

__all__ = [
    "IntegerConstants",
    "crossgl_array_extent",
    "extent_identifier",
    "fold_integer_constant",
    "integral_constant_declaration_value",
]

_INTEGRAL_WORDS = frozenset(
    {
        "char",
        "short",
        "int",
        "long",
        "signed",
        "unsigned",
        "size_t",
        "ptrdiff_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "uint",
        "auto",
    }
)
_UNSIGNED_WORDS = frozenset(
    {"unsigned", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "uint"}
)
_DECLARATION_QUALIFIERS = frozenset(
    {"const", "constexpr", "static", "inline", "volatile", "__device__", "__host__"}
)
_TOKEN = re.compile(
    r"\s*(?:(?P<number>0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+)(?P<suffix>[uUlL]*)"
    r"|(?P<name>(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)"
    r"|(?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>()?:]))"
)
_BINARY_PRECEDENCE = {
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
_CAST_CALLS = re.compile(
    r"(?:static_cast<\s*)?(?P<type>(?:unsigned\s+|signed\s+)?(?:char|short|int|long(?:\s+long)?)"
    r"|unsigned|size_t|u?int(?:8|16|32|64)_t)(?:\s*>)?"
)
# The largest extent folded: a 32-bit signed int, the widest array index every target takes.
_MAX_EXTENT = 2**31 - 1


class IntegerConstants:
    """The integral constants in scope where the importer is, by name; ``None`` marks a name that is not one."""

    def __init__(self):
        self.scopes = [{}]

    def push(self):
        self.scopes.append({})

    def pop(self):
        if len(self.scopes) > 1:
            self.scopes.pop()

    def declare(self, name, value=None):
        """Record ``name`` in the innermost scope: an ``(int, unsigned)`` value, or ``None`` when it hides one."""
        if isinstance(name, str) and name:
            self.scopes[-1][name] = value

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None


class _NotConstant(Exception):
    pass


def _literal(number, suffix):
    base = 10
    digits = number
    if number[:2] in ("0x", "0X"):
        base, digits = 16, number[2:]
    elif number[:2] in ("0b", "0B"):
        base, digits = 2, number[2:]
    elif len(number) > 1 and number.startswith("0"):
        base = 8
    try:
        value = int(digits, base)
    except ValueError as error:
        raise _NotConstant from error
    return value, "u" in suffix.lower() or value > _MAX_EXTENT


def _binary(op, left, right):
    (a, a_unsigned), (b, b_unsigned) = left, right
    unsigned = a_unsigned or b_unsigned
    if op in ("<<", ">>"):
        if b < 0 or b >= 64:
            raise _NotConstant
        unsigned = a_unsigned
        value = a << b if op == "<<" else a >> b
    elif op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
        if unsigned and (a < 0 or b < 0) and op not in ("&&", "||"):
            # The usual arithmetic conversions would wrap the negative operand first.
            raise _NotConstant
        value = int(
            {
                "==": a == b,
                "!=": a != b,
                "<": a < b,
                ">": a > b,
                "<=": a <= b,
                ">=": a >= b,
                "&&": bool(a) and bool(b),
                "||": bool(a) or bool(b),
            }[op]
        )
        return value, False
    elif op in ("/", "%"):
        if b == 0:
            raise _NotConstant
        quotient = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            quotient = -quotient
        value = quotient if op == "/" else a - quotient * b
    else:
        value = {
            "+": lambda: a + b,
            "-": lambda: a - b,
            "*": lambda: a * b,
            "&": lambda: a & b,
            "|": lambda: a | b,
            "^": lambda: a ^ b,
        }[op]()
    if unsigned and value < 0:
        raise _NotConstant
    return value, unsigned


def _unary(op, operand):
    value, unsigned = operand
    if op == "+":
        return value, unsigned
    if op == "!":
        return int(not value), False
    if unsigned:
        raise _NotConstant
    return (-value if op == "-" else ~value), False


class _TextEvaluator:
    def __init__(self, text, constants):
        self.tokens = []
        position = 0
        text = text.strip()
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None or match.end() == position:
                raise _NotConstant
            self.tokens.append(match)
            position = match.end()
        self.index = 0
        self.constants = constants

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take_op(self, op):
        token = self.peek()
        if token is not None and token.group("op") == op:
            self.index += 1
            return True
        return False

    def evaluate(self):
        value = self.conditional()
        if self.peek() is not None:
            raise _NotConstant
        return value

    def conditional(self):
        condition = self.binary(1)
        if not self.take_op("?"):
            return condition
        true_value = self.conditional()
        if not self.take_op(":"):
            raise _NotConstant
        false_value = self.conditional()
        return true_value if condition[0] else false_value

    def binary(self, minimum):
        left = self.unary()
        while True:
            token = self.peek()
            op = token.group("op") if token is not None else None
            precedence = _BINARY_PRECEDENCE.get(op)
            if precedence is None or precedence < minimum:
                return left
            self.index += 1
            right = self.binary(precedence + 1)
            left = _binary(op, left, right)

    def unary(self):
        token = self.peek()
        if token is not None and token.group("op") in ("-", "+", "~", "!"):
            self.index += 1
            return _unary(token.group("op"), self.unary())
        return self.primary()

    def primary(self):
        token = self.peek()
        if token is None:
            raise _NotConstant
        self.index += 1
        if token.group("number") is not None:
            return _literal(token.group("number"), token.group("suffix"))
        if token.group("op") == "(":
            value = self.conditional()
            if not self.take_op(")"):
                raise _NotConstant
            return value
        name = token.group("name")
        if name is None:
            raise _NotConstant
        value = self.constants.lookup(name) if self.constants is not None else None
        if value is None:
            raise _NotConstant
        return value


def _cast_unsigned(type_text):
    words = set(str(type_text).replace("<", " ").replace(">", " ").split())
    if not words or not words <= (_INTEGRAL_WORDS | {"static_cast"}) - {"auto"}:
        raise _NotConstant
    return bool(words & _UNSIGNED_WORDS)


def _evaluate_node(node, constants):
    if isinstance(node, str):
        return _TextEvaluator(node, constants).evaluate()
    if isinstance(node, BinaryOpNode):
        if node.op not in _BINARY_PRECEDENCE:
            raise _NotConstant
        return _binary(
            node.op,
            _evaluate_node(node.left, constants),
            _evaluate_node(node.right, constants),
        )
    if isinstance(node, UnaryOpNode):
        if node.op not in ("-", "+", "~", "!"):
            raise _NotConstant
        return _unary(node.op, _evaluate_node(node.operand, constants))
    if isinstance(node, TernaryOpNode):
        condition = _evaluate_node(node.condition, constants)
        branch = node.true_expr if condition[0] else node.false_expr
        return _evaluate_node(branch, constants)
    if isinstance(node, CastNode):
        value, _ = _evaluate_node(node.expression, constants)
        unsigned = _cast_unsigned(node.target_type)
        if unsigned and value < 0:
            raise _NotConstant
        return value, unsigned
    if (
        isinstance(node, FunctionCallNode)
        and isinstance(node.name, str)
        and len(node.args) == 1
    ):
        match = _CAST_CALLS.fullmatch(node.name.strip())
        if match is None:
            raise _NotConstant
        value, _ = _evaluate_node(node.args[0], constants)
        unsigned = _cast_unsigned(match.group("type"))
        if unsigned and value < 0:
            raise _NotConstant
        return value, unsigned
    raise _NotConstant


def fold_integer_constant(expression, constants):
    """The ``(value, unsigned)`` of an integral constant expression (text or CUDA/HIP AST), else ``None``."""
    try:
        return _evaluate_node(expression, constants)
    except (_NotConstant, RecursionError):
        return None


def integral_constant_declaration_value(node, constants):
    """The value of ``node`` when it declares a ``const``/``constexpr`` integral constant, else ``None``."""
    value = getattr(node, "value", None)
    if value is None:
        return None
    qualifiers = set(getattr(node, "qualifiers", None) or [])
    vtype = str(getattr(node, "vtype", "") or "")
    if "*" in vtype or "&" in vtype or "[" in vtype or "<" in vtype:
        return None
    words = set(vtype.split()) | qualifiers
    if not {"const", "constexpr"} & words:
        return None
    type_words = set(vtype.split()) - _DECLARATION_QUALIFIERS
    if not type_words or not type_words <= _INTEGRAL_WORDS:
        return None
    folded = fold_integer_constant(value, constants)
    if folded is None:
        return None
    unsigned = bool(type_words & _UNSIGNED_WORDS)
    if unsigned and folded[0] < 0:
        return None
    return folded[0], unsigned or (folded[1] and "auto" in type_words)


def extent_identifier(size, prefix):
    """An identifier spelling of an extent CrossGL cannot read (``FFT_storage_size`` for ``FFT::storage_size``).

    No program declares it, so a target that reaches the array refuses it as an undeclared type that names the
    source extent.
    """
    replacements = {
        "*": "_mul_",
        "/": "_div_",
        "%": "_mod_",
        "+": "_plus_",
        "-": "_minus_",
    }
    extent = str(size)
    for operator, word in replacements.items():
        extent = extent.replace(operator, word)
    extent = re.sub(r"[^A-Za-z0-9_]+", "_", extent).strip("_")
    extent = re.sub(r"_+", "_", extent) or "expr"
    if extent[0].isdigit():
        extent = f"{prefix}{extent}"
    return extent


def crossgl_array_extent(size, constants, other_extent):
    """The CrossGL extent for a CUDA or HIP array extent ``size``.

    A constant the importer can evaluate becomes its value, and a bare name or literal it cannot evaluate stays as
    it is. ``other_extent`` spells any other extent: the CrossGL parser reads an expression extent in some
    declarations only (``var a: array<f32, (K + 1)>`` but not ``var<workgroup>`` or a resource), so such an extent
    keeps the identifier spelling of :func:`extent_identifier`, which a target refuses as undeclared.
    """
    size = str(size).strip()
    if not size:
        return size
    folded = fold_integer_constant(size, constants)
    if folded is not None and 0 < folded[0] <= _MAX_EXTENT:
        return str(folded[0])
    if re.fullmatch(r"[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+)[uU]?", size):
        return size
    return other_extent(size)
