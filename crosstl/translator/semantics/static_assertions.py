"""``static_assert`` declarations: checked once at translation time, then removed.

A static assertion (C++17 [dcl.dcl]/6, MSL 4.1 section 1.4, HLSL 2021) declares nothing and has no run-time effect:
it makes the program ill-formed when its constant condition is false. CrossGL keeps the declaration, at shader, stage
and structure scope in ``ShaderNode.static_assertions`` and in a block as a statement, so the Metal importer's
assertions are not lost on the way to a target. GLSL, WGSL and SPIR-V have no such declaration, and the other
generators do not print one. The parser used to skip a shader-scope assertion token by token, and printed a
block-scope one as a call of an undeclared ``static_assert`` function.

:func:`resolve_static_assertions` evaluates each condition, for every target, and removes the assertion when it
holds. A false condition raises :class:`StaticAssertionFailedError`. A condition the translator cannot evaluate raises
:class:`StaticAssertionUnresolvedError`: dropping it would translate a program its source compiler might reject.

The evaluator covers integral constant expressions over integer and boolean literals, the program's integer
``const`` declarations and enumerators, ``sizeof`` already folded by :mod:`.sizeof_values`, the arithmetic, bitwise,
shift, comparison and logical operators, ``!`` and ``?:``, and conversions to integer types and ``bool``. Every value
it computes, constants included, must lie in ``[0, 2**31 - 1]``: there a value means the same in every signed and
unsigned integer type of 32 bits or more, so the result does not depend on the usual arithmetic conversions
(C++17 [expr]/11), on wrap-around or on the sign of a quotient. Anything else, a negative or larger value included,
is unresolved. A block-scope condition that names a parameter or local of some function is unresolved too, since that
name may shadow the program's constant.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .binding_constants import _AMBIGUOUS, INTEGER_TYPES, _ConstantTable, _type_name
from .integral_types import convert_to_integer_type
from .storage_blocks import _declared_names, _walk

__all__ = [
    "StaticAssertionFailedError",
    "StaticAssertionUnresolvedError",
    "resolve_static_assertions",
]

_LARGEST = 2**31 - 1
_COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}
_ARITHMETIC = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
}


class StaticAssertionFailedError(TranslationDiagnostic):
    """A ``static_assert`` whose condition evaluates to false."""

    missing_capabilities = ()

    def __init__(self, target, condition, message):
        detail = f": {message}" if message else ""
        super().__init__(
            f"static assertion failed{detail} (condition {condition})",
            target=target,
            code=f"{CODE_PREFIX}static-assertion-failed",
        )


class StaticAssertionUnresolvedError(TranslationDiagnostic):
    """A ``static_assert`` whose condition the translator cannot evaluate."""

    def __init__(self, target, condition, reason):
        self.missing_capabilities = ("crossgl.static-assertion-evaluation",)
        super().__init__(
            f"static assertion {condition} cannot be checked at translation time: {reason}",
            target=target,
            code=f"{CODE_PREFIX}static-assertion-unresolved",
        )


class _Unresolved(Exception):
    pass


def _checked(value):
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int) or value < 0 or value > _LARGEST:
        raise _Unresolved(
            f"the value {value} lies outside [0, {_LARGEST}], where integer types disagree"
        )
    return value


class _Evaluator:
    """Integral constant expressions of one program, every value bounded as the module docstring describes."""

    def __init__(self, program):
        self.table = _ConstantTable(program)
        self.values = {}
        self.pending = set()

    def lookup(self, name):
        if name in self.values:
            return self.values[name]
        entry = self.table.entries.get(name)
        if entry is None:
            raise _Unresolved(f"{name} is not an integer constant or enumerator")
        if entry is _AMBIGUOUS:
            raise _Unresolved(f"{name} names several constants")
        if entry[0] == "unresolved":
            raise _Unresolved(entry[1])
        if id(entry) in self.pending:
            raise _Unresolved(f"{name} is defined in terms of itself")
        self.pending.add(id(entry))
        try:
            if entry[0] == "const":
                value = _checked(
                    convert_to_integer_type(self.evaluate(entry[2]), entry[3])
                )
            else:
                value = self._enumerator(entry)
        finally:
            self.pending.discard(id(entry))
        self.values[name] = value
        return value

    def _enumerator(self, entry):
        _, enum, variant, previous = entry
        if getattr(variant, "value", None) is not None:
            return self.evaluate(variant.value)
        if previous is None:
            return 0
        return _checked(self.lookup(f"{enum.name}::{previous[2].name}") + 1)

    def evaluate(self, node):
        if isinstance(node, A.LiteralNode):
            return self._literal(node)
        if isinstance(node, A.IdentifierNode):
            if node.name in {"true", "false"}:
                return int(node.name == "true")
            return self.lookup(node.name)
        if isinstance(node, A.MemberAccessNode):
            owner = getattr(getattr(node, "object_expr", None), "name", None)
            member = getattr(node, "member", None)
            if isinstance(owner, str) and isinstance(member, str):
                return self.lookup(f"{owner}::{member}")
            raise _Unresolved("a member access is not a constant")
        if isinstance(node, A.BinaryOpNode):
            return self._binary(str(node.op), node.left, node.right)
        if isinstance(node, A.UnaryOpNode) and not getattr(node, "is_postfix", False):
            return self._unary(str(node.op), node.operand)
        if isinstance(node, A.TernaryOpNode):
            condition = self.evaluate(node.condition)
            true_value = self.evaluate(node.true_expr)
            false_value = self.evaluate(node.false_expr)
            return true_value if condition else false_value
        if isinstance(node, A.CastNode):
            return self._conversion(_type_name(node.target_type), [node.expression])
        if isinstance(node, A.FunctionCallNode):
            callee = _type_name(getattr(node, "function", None)) or _type_name(
                getattr(node, "name", None)
            )
            return self._conversion(
                callee, list(getattr(node, "arguments", None) or [])
            )
        raise _Unresolved(f"{type(node).__name__} is not an integral constant")

    def _literal(self, node):
        value = node.value
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return _checked(value)
        raise _Unresolved(f"the literal {value!r} is not an integer or boolean")

    def _conversion(self, type_name, arguments):
        type_name = self.table.aliases.resolve(type_name)
        if len(arguments) != 1:
            raise _Unresolved(f"a call of {type_name} is not a constant")
        if type_name == "bool":
            return int(self.evaluate(arguments[0]) != 0)
        if type_name in INTEGER_TYPES:
            return _checked(
                convert_to_integer_type(self.evaluate(arguments[0]), type_name)
            )
        raise _Unresolved(f"a call of {type_name} is not a constant")

    def _unary(self, operator, operand):
        value = self.evaluate(operand)
        if operator == "!":
            return int(value == 0)
        if operator == "+":
            return value
        if operator == "-":
            return _checked(-value)
        raise _Unresolved(f"operator {operator} is not evaluated")

    def _binary(self, operator, left_node, right_node):
        left = self.evaluate(left_node)
        right = self.evaluate(right_node)
        if operator == "&&":
            return int(bool(left) and bool(right))
        if operator == "||":
            return int(bool(left) or bool(right))
        if operator in _COMPARISONS:
            return int(_COMPARISONS[operator](left, right))
        if operator in _ARITHMETIC:
            return _checked(_ARITHMETIC[operator](left, right))
        if operator in {"/", "%"}:
            if right == 0:
                raise _Unresolved("the condition divides by zero")
            return left // right if operator == "/" else left % right
        if operator in {"<<", ">>"}:
            if right > 31:
                raise _Unresolved("the condition shifts by more than 31 bits")
            return _checked(left << right) if operator == "<<" else left >> right
        raise _Unresolved(f"operator {operator} is not evaluated")


def _condition_text(node):
    """A compact rendering of a condition for diagnostics."""
    if isinstance(node, A.LiteralNode):
        return (
            str(node.value).lower() if isinstance(node.value, bool) else str(node.value)
        )
    if isinstance(node, A.IdentifierNode):
        return str(node.name)
    if isinstance(node, A.BinaryOpNode):
        return f"({_condition_text(node.left)} {node.op} {_condition_text(node.right)})"
    if isinstance(node, A.UnaryOpNode):
        return f"{node.op}{_condition_text(node.operand)}"
    if isinstance(node, A.MemberAccessNode):
        return f"{_condition_text(node.object_expr)}.{node.member}"
    if isinstance(node, A.FunctionCallNode):
        callee = _type_name(getattr(node, "function", None)) or "call"
        arguments = ", ".join(_condition_text(a) for a in node.arguments or [])
        return f"{callee}({arguments})"
    if isinstance(node, A.TernaryOpNode):
        return (
            f"({_condition_text(node.condition)} ? {_condition_text(node.true_expr)}"
            f" : {_condition_text(node.false_expr)})"
        )
    return type(node).__name__


def _block_assertions(program):
    """Remove every block-scope assertion from its statement list; return them in source order."""
    found = []
    for node in list(_walk(program)):
        for field, value in list(vars(node).items()):
            if field == "parent" or not isinstance(value, list):
                continue
            if not any(isinstance(item, A.StaticAssertNode) for item in value):
                continue
            found.extend(item for item in value if isinstance(item, A.StaticAssertNode))
            value[:] = [
                item for item in value if not isinstance(item, A.StaticAssertNode)
            ]
    return found


def resolve_static_assertions(program, target) -> int:
    """Check and remove every ``static_assert`` of ``program``; returns how many held.

    Raises :class:`StaticAssertionFailedError` or :class:`StaticAssertionUnresolvedError` for the first one that is
    false or cannot be evaluated.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    declared = list(getattr(program, "static_assertions", None) or [])
    program.static_assertions = []
    in_blocks = _block_assertions(program)
    if not declared and not in_blocks:
        return 0
    evaluator = _Evaluator(program)
    shadowed = _declared_names(program) if in_blocks else set()
    target = str(target or "")
    block_ids = {id(assertion) for assertion in in_blocks}
    for assertion in [*declared, *in_blocks]:
        condition = assertion.condition
        text = _condition_text(condition)
        try:
            if id(assertion) in block_ids:
                names = {
                    node.name
                    for node in _walk(condition)
                    if isinstance(node, A.IdentifierNode)
                }
                if names & shadowed:
                    raise _Unresolved(
                        f"{', '.join(sorted(names & shadowed))} may name a parameter or local"
                    )
            value = evaluator.evaluate(condition)
        except _Unresolved as error:
            raise StaticAssertionUnresolvedError(target, text, str(error)) from None
        if not value:
            raise StaticAssertionFailedError(target, text, assertion.message)
    return len(declared) + len(in_blocks)
