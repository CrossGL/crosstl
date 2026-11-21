"""GLSL's 8- and 16-bit integer scalar arithmetic spelled in CrossGL, which promotes narrow scalars to ``int``.

GL_EXT_shader_explicit_arithmetic_types gives ``int8_t``, ``uint8_t``, ``int16_t`` and ``uint16_t`` arithmetic
of their own width: ``v * v`` of two ``int8_t`` is an ``int8_t``, so ``int8_t v = 13; float(v * v)`` is ``-87.0``.
CrossGL promotes a narrow integer scalar to ``int`` before arithmetic, as C++ [conv.prom] and MSL do
(``crosstl/translator/semantics/narrow.py``), so the same CrossGL expression is ``169.0``. The importer therefore
makes the narrow result of a GLSL scalar operation explicit wherever the operation can leave the operand range, in
the forms ``crosstl/translator/semantics/narrow_arithmetic.py`` writes for promoting targets:

- ``+``, ``-``, ``*`` and ``<<`` compute on ``uint`` operands, ``T((uint(a) * uint(b)))``, so the 32-bit
  intermediate wraps instead of overflowing a signed ``int``, and the conversion to ``T`` reduces it modulo 2^N;
- signed ``/``, unary ``-`` and unsigned ``~`` become ``T((e))``;
- ``&``, ``|``, ``^``, ``>>``, ``%``, unsigned ``/`` and signed ``~`` stay in range and keep their form.

An operation stored straight into a register of its own type (initializer, ``=`` assignment, return value,
conversion ``T(e)``) keeps its form too, because the store reduces it already. GLSL converts a narrow operand to the
type of a 32-bit operand or literal (``v * 2`` is an ``int``), and CrossGL does not promote vectors, so neither
needs a conversion. An operation that mixes two different narrow types, or a narrow scalar with an operand whose
type the importer cannot determine, is refused with :class:`GlslNarrowArithmeticError` rather than guessed.
"""

from __future__ import annotations

import re

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .aggregate_equality import GlslExpressionTypes
from .import_exact_lookups import folded_operator
from .OpenglAst import (
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    FunctionNode,
    NumberNode,
    PostfixOpNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = [
    "NARROW_INTEGER_SCALARS",
    "GlslNarrowArithmeticError",
    "import_narrow_binary",
    "import_narrow_unary",
    "narrow_store_targets",
]

NARROW_INTEGER_SCALARS = frozenset({"int8_t", "uint8_t", "int16_t", "uint16_t"})
# Operators computed on ``uint`` operands so the 32-bit intermediate wraps.
_UNSIGNED_INTERMEDIATE = frozenset({"+", "-", "*", "<<"})
# Operators that can leave the range for some operand type (``_leaves_range`` decides per type).
_RANGE_CHANGING_BINARY = _UNSIGNED_INTERMEDIATE | {"/"}
_RANGE_CHANGING_UNARY = frozenset({"-", "~"})
# Operators whose result has the type of the (converted) operands.
_TYPED_BINARY = frozenset({"+", "-", "*", "/", "%", "&", "|", "^"})
_SHIFTS = frozenset({"<<", ">>"})
_TYPED_UNARY = frozenset({"-", "~", "+"})
# GL_EXT_shader_explicit_arithmetic_types 16-bit literal suffixes.
_SHORT_LITERAL = re.compile(r"^(?:0[xX][0-9a-fA-F]+|\d+)(?P<unsigned>[uU])?[sS]$")
_NOT_NARROW = ""


class GlslNarrowArithmeticError(TranslationDiagnostic):
    """A GLSL narrow integer scalar operation whose result type the importer cannot determine."""

    code = CODE_PREFIX + "glsl-narrow-arithmetic-untyped"


def _operator(op):
    return folded_operator(op) or op


def _normalized(type_name):
    if type_name is None:
        return None
    words = str(type_name).split()
    return words[-1] if words else None


def _call_name(expr):
    name = getattr(expr, "name", None)
    return name.name if isinstance(name, VariableNode) else name


def _leaves_range(operator, narrow, unary):
    """Whether ``operator`` on in-range operands of scalar type ``narrow`` can produce a value outside its range."""
    signed = not narrow.startswith("u")
    if unary:
        return operator == "-" or (operator == "~" and not signed)
    if operator in _UNSIGNED_INTERMEDIATE:
        return True
    return operator == "/" and signed


def narrow_store_targets(root):
    """Map ``id(value)`` to ``(value, store)`` for every GLSL value stored straight into a register.

    ``store`` is the declared type name, or the assignment's left operand, whose type is looked up while the
    importer renders the function (its variable scopes exist only then).
    """
    stores = {}
    seen = set()

    def record(value, store):
        if isinstance(value, (BinaryOpNode, UnaryOpNode)):
            stores[id(value)] = (value, store)

    def walk(node, return_type):
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item, return_type)
            return
        if not isinstance(node, ASTNode) or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, FunctionNode):
            return_type = getattr(node, "return_type", None)
        elif isinstance(node, VariableNode) and node.value is not None:
            record(node.value, node.vtype)
        elif isinstance(node, AssignmentNode) and getattr(node, "operator", "=") == "=":
            left = node.left
            declared = (
                getattr(left, "vtype", None) if isinstance(left, VariableNode) else None
            )
            record(node.right, declared or left)
        elif isinstance(node, ReturnNode) and return_type is not None:
            record(node.value, return_type)
        elif isinstance(node, FunctionCallNode):
            arguments = list(getattr(node, "args", None) or [])
            if _call_name(node) in NARROW_INTEGER_SCALARS and len(arguments) == 1:
                record(arguments[0], _call_name(node))
        for child in vars(node).values():
            walk(child, return_type)

    walk(root, None)
    return stores


class _NarrowTypes:
    """The narrow integer scalar type of a GLSL expression: its name, ``""`` when it has another type, or ``None``."""

    def __init__(self, importer):
        self.typer = GlslExpressionTypes(importer)

    def of(self, expr):
        if isinstance(expr, BinaryOpNode):
            return self.binary(expr)
        if isinstance(expr, UnaryOpNode):
            if _operator(expr.op) in _TYPED_UNARY:
                return self.of(expr.operand)
            return _NOT_NARROW
        if isinstance(expr, PostfixOpNode):
            return self.of(expr.operand)
        if isinstance(expr, AssignmentNode):
            return self.of(getattr(expr, "left", None))
        if isinstance(expr, TernaryOpNode):
            arms = {self.of(expr.true_expr), self.of(expr.false_expr)}
            if None in arms:
                return None
            return arms.pop() if len(arms) == 1 else _NOT_NARROW
        if isinstance(expr, NumberNode):
            short = _SHORT_LITERAL.match(str(expr.value))
            if short:
                return "uint16_t" if short.group("unsigned") else "int16_t"
        if (
            isinstance(expr, FunctionCallNode)
            and _call_name(expr) in NARROW_INTEGER_SCALARS
        ):
            return _call_name(expr)
        return self.declared(self.typer.type_of(expr))

    @staticmethod
    def declared(type_name):
        declared = _normalized(type_name)
        if declared is None:
            return None
        return declared if declared in NARROW_INTEGER_SCALARS else _NOT_NARROW

    def binary(self, expr):
        op = _operator(expr.op)
        if op not in _TYPED_BINARY and op not in _SHIFTS:
            return _NOT_NARROW
        left = self.of(expr.left)
        if op in _SHIFTS:
            # A shift has the type of its left operand (GLSL 4.60 section 5.9).
            return left
        right = self.of(expr.right)
        if left and right:
            if left != right:
                raise GlslNarrowArithmeticError(
                    f"GLSL '{op}' mixes {left} and {right}; the importer does not model the explicit arithmetic "
                    "types' implicit conversions between narrow integer types",
                    target="crossgl",
                )
            return left
        if (left and right is None) or (right and left is None):
            raise GlslNarrowArithmeticError(
                f"cannot determine an operand type of a GLSL {left or right} '{op}'; GLSL computes narrow integer "
                "arithmetic at its own width while CrossGL promotes it to int",
                target="crossgl",
            )
        if left is None or right is None:
            return None
        return _NOT_NARROW


def _converted_type(importer, node, operator, unary):
    """The narrow type to convert the result of GLSL operation ``node`` to, or ``None``."""
    if operator not in (_RANGE_CHANGING_UNARY if unary else _RANGE_CHANGING_BINARY):
        return None
    types = _NarrowTypes(importer)
    narrow = types.of(node)
    if not narrow or not _leaves_range(operator, narrow, unary):
        return None
    stored = (getattr(importer, "glsl_narrow_stores", None) or {}).get(id(node))
    if stored is not None and stored[0] is node:
        store = stored[1]
        store_type = store if isinstance(store, str) else types.typer.type_of(store)
        if _normalized(store_type) == narrow:
            return None
    return narrow


def import_narrow_binary(importer, node, left, operator, right):
    """CrossGL for GLSL ``left operator right``, converted back to its narrow scalar type where GLSL wraps."""
    narrow = _converted_type(importer, node, _operator(node.op), unary=False)
    if narrow is None:
        return f"({left} {operator} {right})"
    if _operator(node.op) in _UNSIGNED_INTERMEDIATE:
        shifted = right if _operator(node.op) in _SHIFTS else f"uint({right})"
        return f"{narrow}((uint({left}) {operator} {shifted}))"
    return f"{narrow}(({left} {operator} {right}))"


def import_narrow_unary(importer, node, operator, operand):
    """CrossGL for GLSL ``operator operand``, converted back to its narrow scalar type where GLSL wraps."""
    narrow = _converted_type(importer, node, _operator(node.op), unary=True)
    rendered = f"({operator}{operand})"
    return f"{narrow}({rendered})" if narrow else rendered
