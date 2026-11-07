"""MSL's ordered, unordered and normal-value tests, spelled for targets that lack them.

MSL defines three relational functions that no other shading language has (MSL 3.2 section 6.4, "Relational
Functions"), each component by component for vector operands, with a ``bool`` or ``boolN`` result:

* ``isordered(x, y)`` is ``(x == x) && (y == y)``: neither operand is a NaN;
* ``isunordered(x, y)`` is true when either operand is a NaN;
* ``isnormal(x)`` is true when ``x`` is neither zero, subnormal, infinite nor a NaN (C99 7.12.3.5).

CrossGL imported from Metal keeps these calls, so the GLSL, HLSL, WGSL and other generators emitted them verbatim (an
undeclared function) and the SPIR-V generator refused them. :func:`lower_float_relations` rewrites them for every
target except Metal with the classification functions every generator lowers exactly, ``isnan`` and ``isfinite`` (the
SPIR-V generator tests their bits, so driver fast-math cannot fold them):

* ``isunordered(x, y)`` becomes ``isnan(x) || isnan(y)``. CrossGL's ``||`` on boolean vectors combines them component
  by component and evaluates both operands. On scalars it short-circuits, which skips the second operand when the
  first is a NaN, while a call evaluates both; so an operand with side effects goes first, and when both have side
  effects the scalar form is ``bool(uint(isnan(x)) | uint(isnan(y)))``.
* ``isordered(x, y)`` is the negation of that form.
* ``isnormal(x)`` becomes ``isfinite(x) && abs(x) >= m``, where ``m`` is the smallest normal value of ``x``'s format
  (``2^-14`` for binary16, ``2^-126`` for binary32), exact in binary32 so that ``abs(x)`` compares exactly. A
  subnormal ``x`` that the target flushes to zero still compares below ``m``. A vector applies it to each component.
  The lowering reads ``x`` more than once, so an operand with side effects is refused with
  :class:`FloatRelationUnsupportedError`, as are operands whose format the pass cannot infer or that are binary64.

A program that defines a function with one of these names keeps its calls.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from .hub_types import call_name, iter_functions
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child
from .scoped_walk import ScopedWalker

__all__ = ["FloatRelationUnsupportedError", "lower_float_relations"]

_ARITY = {"isordered": 2, "isunordered": 2, "isnormal": 1}
_CARRYING = frozenset({"metal", "cgl", "crossgl", ""})
# The smallest positive normal value of each binary format the targets compute, by width in bits.
_SMALLEST_NORMAL = {16: 2.0**-14, 32: 2.0**-126}
# Calls that read memory without writing it, so skipping one changes nothing.
_READ_ONLY_CALLS = frozenset({"buffer_load"})
_COMPONENTS = "xyzw"


class FloatRelationUnsupportedError(TranslationDiagnostic):
    """An MSL ordered, unordered or normal-value test whose operand the lowering cannot compute exactly."""

    def __init__(self, target, name, reason, span=None):
        self.missing_capabilities = (f"{target}.metal-relational-{name}",)
        super().__init__(
            reason,
            span=span,
            target=target,
            code=f"{CODE_PREFIX}{target}-metal-relational-function-unsupported",
        )


def lower_float_relations(program, target) -> int:
    """Rewrite or refuse ``program``'s MSL-only relational calls for ``target``; returns how many were rewritten."""
    target = str(target or "").lower()
    if target in _CARRYING or not isinstance(program, A.ShaderNode):
        return 0
    defined = {getattr(function, "name", None) for function in iter_functions(program)}
    names = frozenset(_ARITY) - defined
    if not names:
        return 0
    walker = _FloatRelations(program, target, names)
    walker.run_program()
    return walker.rewritten


def _effect_free(expr) -> bool:
    """True when skipping the evaluation of ``expr`` changes nothing but its value."""
    if is_pure(expr):
        return True
    if isinstance(expr, A.FunctionCallNode):
        return call_name(expr) in _READ_ONLY_CALLS and all(
            _effect_free(argument) for argument in expr.arguments or []
        )
    if isinstance(expr, A.ArrayAccessNode):
        return _effect_free(expr.array_expr) and _effect_free(expr.index_expr)
    if isinstance(expr, A.MemberAccessNode):
        return _effect_free(expr.object_expr)
    if isinstance(expr, A.SwizzleNode):
        return _effect_free(expr.vector_expr)
    return False


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        line, column, file = (location.get(key) for key in ("line", "column", "file"))
    else:
        line, column, file = (
            getattr(location, key, None) for key in ("line", "column", "file")
        )
    if not isinstance(line, int):
        return None
    return SourceSpan(
        file=file, line=line, column=column if isinstance(column, int) else None
    )


class _FloatRelations(ScopedWalker):
    def __init__(self, program, target, names):
        super().__init__(program)
        self.target = target
        self.names = names
        self.rewritten = 0

    def _walk(self, node):
        if isinstance(node, A.VariableNode):
            # The walker declares a variable instead of visiting it; its initializer is rewritten first.
            for field, key in list(child_slots(node)):
                self._walk(get_child(node, field, key))
            self.visit(node)
            self.scope.declare(node.name, node.var_type)
            return
        super()._walk(node)

    def visit(self, node):
        # Children are visited first, so a nested call is rewritten before the expression that uses it.
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self._replacement(child)
            if replacement is not None:
                set_child(node, field, key, replacement)
                self.rewritten += 1

    def _replacement(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return None
        name = call_name(node)
        arguments = list(node.arguments or [])
        if name not in self.names or len(arguments) != _ARITY[name]:
            return None
        if name == "isnormal":
            return self._normal(node, arguments[0])
        unordered = self._unordered(node, name, *arguments)
        return unordered if name == "isunordered" else A.UnaryOpNode("!", unordered)

    def _refuse(self, node, name, reason):
        return FloatRelationUnsupportedError(
            self.target,
            name,
            reason,
            span=_span(getattr(node, "source_location", None)),
        )

    def _unordered(self, node, name, left, right):
        if _effect_free(left) and not _effect_free(right):
            # The short-circuit operator always evaluates its first operand.
            left, right = right, left
        tests = [call("isnan", [left]), call("isnan", [right])]
        if _effect_free(right):
            return A.BinaryOpNode(tests[0], "||", tests[1])
        value = self.scope.infer(left)
        if value is None:
            raise self._refuse(
                node,
                name,
                f"{name}'s operands both have side effects and their shape cannot be inferred, so CrossTL cannot "
                "evaluate both of them exactly once",
            )
        if value.lanes > 1:
            # On boolean vectors ``||`` evaluates both operands.
            return A.BinaryOpNode(tests[0], "||", tests[1])
        lanes = [call("uint", [test]) for test in tests]
        return call("bool", [A.BinaryOpNode(lanes[0], "|", lanes[1])])

    def _normal(self, node, operand):
        if not is_pure(operand):
            raise self._refuse(
                node,
                "isnormal",
                "isnormal reads its operand more than once in its lowering, and this operand has side effects",
            )
        value = self.scope.infer(operand)
        if value is None or value.kind != "f":
            raise self._refuse(
                node,
                "isnormal",
                "isnormal needs a floating-point operand whose format CrossTL can infer",
            )
        smallest = _SMALLEST_NORMAL.get(value.bits)
        if smallest is None:
            raise self._refuse(
                node,
                "isnormal",
                f"isnormal has no lowering for {value.bits}-bit floating-point operands",
            )
        if value.lanes == 1:
            return self._normal_scalar(operand, smallest)
        components = [
            self._normal_scalar(
                A.MemberAccessNode(clone(operand), _COMPONENTS[lane]), smallest
            )
            for lane in range(value.lanes)
        ]
        return call(f"bvec{value.lanes}", components)

    @staticmethod
    def _normal_scalar(operand, smallest):
        # Each use gets its own copy, so a later in-place rewrite of one leaves the other intact.
        magnitude = call("abs", [clone(operand)])
        bound = A.LiteralNode(smallest, A.PrimitiveType("float"))
        return A.BinaryOpNode(
            call("isfinite", [operand]), "&&", A.BinaryOpNode(magnitude, ">=", bound)
        )
