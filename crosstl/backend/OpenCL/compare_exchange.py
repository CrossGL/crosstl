"""OpenCL C ``atomic_compare_exchange_strong`` and ``_weak`` (OpenCL C 3.0 section 6.15.12.7), lowered to CrossGL.

``bool atomic_compare_exchange_strong(volatile A *object, C *expected, C desired)`` compares ``*object`` with
``*expected``; if they are equal it stores ``desired`` in ``*object``, otherwise it stores the value of ``*object``
in ``*expected``, and it returns whether the exchange happened. CrossGL's ``atomicCompareExchange(object, compare,
value)`` returns the value the object held before, so the importer reads the expected value once, exchanges, and
writes the original value back only when the exchange failed::

    var expected_0: i32 = e;
    var original_0: i32 = atomicCompareExchange(object, expected_0, desired);
    if (original_0 != expected_0) {
        e = original_0;
    }

and the call's value is ``(original_0 == expected_0)``. The weak form may fail spuriously, so the strong lowering is
a valid weak one. The non-explicit forms take the relaxed device-scope order of every CrossGL atomic, as the other
OpenCL C atomics do; an ``_explicit`` form with another order or a wider scope is refused by the converter.

These statements must run where the call is evaluated, exactly once. The importer emits them before a statement
whose evaluation reaches the call unconditionally (an expression statement, a declaration, a ``return``, or an
``if`` or ``switch`` head) and refuses a call anywhere else, such as a loop condition or the right operand of ``&&``,
with :class:`OpenCLCompareExchangeUnsupportedError`. So is an expected pointer whose address has side effects, which
the lowering would evaluate twice, and a floating-point object: OpenCL compares such objects bit by bit, which ``!=``
does not.
"""

from __future__ import annotations

from crosstl.backend.HIP.HipAst import (
    ArrayAccessNode,
    AssignmentNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "OpenCLCompareExchangeMixin",
    "OpenCLCompareExchangeUnsupportedError",
    "is_compare_exchange",
]

COMPARE_EXCHANGE_FUNCTIONS = frozenset(
    {
        "atomic_compare_exchange_strong",
        "atomic_compare_exchange_strong_explicit",
        "atomic_compare_exchange_weak",
        "atomic_compare_exchange_weak_explicit",
    }
)
_INTEGER_TYPES = frozenset({"i32", "u32", "i64", "u64"})
_PURE_UNARY_OPERATORS = frozenset({"-", "+", "!", "~", "&", "*"})
_SIDE_EFFECT_BINARY_OPERATORS = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", ","}
)


class OpenCLCompareExchangeUnsupportedError(TranslationDiagnostic):
    """An OpenCL C ``atomic_compare_exchange_*`` call CrossGL cannot evaluate exactly once in its place."""

    code = CODE_PREFIX + "opencl-compare-exchange-unsupported"


def is_compare_exchange(node) -> bool:
    return (
        isinstance(node, AtomicOperationNode)
        and node.operation in COMPARE_EXCHANGE_FUNCTIONS
    )


def _unconditional_operands(node):
    """The operands evaluating ``node`` always evaluates, in evaluation order where C fixes one."""
    if isinstance(node, TernaryOpNode):
        return [node.condition]
    if isinstance(node, BinaryOpNode):
        return [node.left] if node.op in ("&&", "||") else [node.left, node.right]
    if isinstance(node, AssignmentNode):
        return [node.right, node.left]
    if isinstance(node, UnaryOpNode):
        return [node.operand]
    if isinstance(node, FunctionCallNode):
        # AtomicOperationNode is a FunctionCallNode whose name is the operation.
        name = node.name
        return [*([] if isinstance(name, str) else [name]), *(node.args or [])]
    if isinstance(node, ArrayAccessNode):
        return [node.array, node.index]
    if isinstance(node, MemberAccessNode):
        return [node.object]
    if isinstance(node, CastNode):
        return [node.expression]
    if isinstance(node, InitializerListNode):
        return list(node.elements or [])
    return []


def _statement_heads(stmt):
    """The expressions a statement evaluates once, before anything else it does."""
    if isinstance(stmt, VariableNode):
        return [stmt.value] if getattr(stmt, "value", None) is not None else []
    if isinstance(stmt, ReturnNode):
        return [stmt.value] if stmt.value is not None else []
    if isinstance(stmt, IfNode):
        return [stmt.condition]
    if isinstance(stmt, SwitchNode):
        return [stmt.expression]
    if isinstance(stmt, (AssignmentNode, BinaryOpNode, UnaryOpNode)):
        return [stmt]
    if isinstance(stmt, (FunctionCallNode, TernaryOpNode, CastNode)):
        return [stmt]
    return []


def _collect_compare_exchanges(expression, found):
    if expression is None or isinstance(expression, (str, int, float, bool)):
        return
    for operand in _unconditional_operands(expression):
        _collect_compare_exchanges(operand, found)
    if is_compare_exchange(expression):
        found.append(expression)


def _is_pure(node) -> bool:
    """Whether evaluating ``node`` twice reads the same location and changes nothing."""
    if node is None or isinstance(node, (str, int, float, bool)):
        return True
    if isinstance(node, UnaryOpNode):
        return node.op in _PURE_UNARY_OPERATORS and _is_pure(node.operand)
    if isinstance(node, BinaryOpNode):
        return (
            node.op not in _SIDE_EFFECT_BINARY_OPERATORS
            and _is_pure(node.left)
            and _is_pure(node.right)
        )
    if isinstance(node, ArrayAccessNode):
        return _is_pure(node.array) and _is_pure(node.index)
    if isinstance(node, MemberAccessNode):
        return _is_pure(node.object)
    if isinstance(node, CastNode):
        return _is_pure(node.expression)
    return False


class OpenCLCompareExchangeMixin:
    """Lower OpenCL C compare-and-exchange calls to CrossGL ``atomicCompareExchange`` statements."""

    def reset_opencl_compare_exchanges(self):
        # id(call) -> the CrossGL boolean expression that stands for the call's value.
        self.opencl_compare_exchange_results = {}
        self.opencl_compare_exchange_count = 0

    def emit_opencl_compare_exchanges(self, stmt):
        """Emit the lowering of every compare-exchange ``stmt`` evaluates first; True when nothing else is left."""
        found = []
        for head in _statement_heads(stmt):
            _collect_compare_exchanges(head, found)
        for call in found:
            self.opencl_compare_exchange_results[id(call)] = (
                self.emit_opencl_compare_exchange(call)
            )
        return bool(found) and is_compare_exchange(stmt)

    def emit_opencl_compare_exchange(self, call):
        args = list(call.args)
        if call.operation.endswith("_explicit"):
            args = self.opencl_relaxed_atomic_arguments(call.operation, args)
        if len(args) != 3:
            raise OpenCLCompareExchangeUnsupportedError(
                f"{call.operation} takes 3 arguments besides its memory orders and scope, got {len(args)}"
            )
        target, expected_pointer, desired = args
        if not _is_pure(expected_pointer):
            raise OpenCLCompareExchangeUnsupportedError(
                f"{call.operation}: the expected-value pointer has side effects, and the lowering reads and "
                "writes the value it addresses"
            )
        value_type = self.opencl_pointer_result_type(call.operation, expected_pointer)
        if value_type not in _INTEGER_TYPES:
            raise OpenCLCompareExchangeUnsupportedError(
                f"{call.operation} on {value_type}: OpenCL compares the object bit by bit, which CrossGL's "
                "atomicCompareExchange keeps only for 32- and 64-bit integers"
            )
        expected = self.visit(
            self.opencl_pointed_value(call.operation, expected_pointer)
        )
        index = self.opencl_compare_exchange_count
        self.opencl_compare_exchange_count += 1
        saved = self.opencl_unused_helper_name(f"cl_expected_{index}")
        original = self.opencl_unused_helper_name(f"cl_original_{index}")
        target_text = self.visit(self.opencl_pointed_value(call.operation, target))
        desired_text = self.visit(desired)
        self.emit(f"var {saved}: {value_type} = {expected};")
        self.emit(
            f"var {original}: {value_type} = atomicCompareExchange({target_text}, {saved}, {desired_text});"
        )
        self.emit(f"if ({original} != {saved}) {{")
        self.emit(f"    {expected} = {original};")
        self.emit("}")
        return f"({original} == {saved})"

    def format_opencl_compare_exchange_value(self, node):
        """The CrossGL value of a compare-exchange call whose statements were emitted, refusing any other."""
        result = self.opencl_compare_exchange_results.get(id(node))
        if result is not None:
            return result
        raise OpenCLCompareExchangeUnsupportedError(
            f"{node.operation} in a position CrossGL evaluates conditionally or more than once, such as a loop "
            "condition or the right operand of && or ||; the importer lowers it only where its statement "
            "evaluates it exactly once"
        )
