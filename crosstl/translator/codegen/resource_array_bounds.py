"""Sizes of unsized resource arrays that are indexed with values computed at run time.

An HLSL ``Texture2D t[]`` or ``RWStructuredBuffer<T> b[]`` (and their CrossGL forms) declares as many resources as the
host binds. Metal's ``array<T, N>`` arguments and OpenGL's resource arrays indexed with a non-constant value need a
size. The shared size inference (:func:`resource_arrays.collect_resource_array_size_hints`) follows compile-time
values through branches, loops and calls precisely, but it skipped any other index: ``b[2]`` and ``b[i % 5]`` gave
``b`` three elements, so ``b[4]`` read past the array, and ``b[material]`` gave it one.

:class:`DynamicIndexSizes` separates the indexes the shared inference can evaluate from those it cannot. An index is
*static* when every value it can take derives from compile-time constants, whichever path the program takes (a
flow-insensitive taint analysis: a run-time input, a resource read or a loop with a run-time trip count makes the
names it reaches dynamic). A static index the shared inference evaluated in every context is left to it. Every other
index needs a bound this module proves; the array then gets one element more than the largest value. An index is
bounded when it is

* a compile-time integer ``c`` (it needs ``c + 1`` elements);
* ``x % n`` or ``x & m`` with constant ``n`` or ``m`` (``n`` and ``m + 1``);
* ``min(x, c)`` or ``clamp(x, lo, c)`` with constant ``c``;
* the induction variable of ``for (k = a; k < b; k++)`` (also ``<=``) with constant ``b``, which the loop body does not
  change;
* a local declared once with a bounded initializer and never changed, or a parameter every caller passes a bounded
  value for;
* a sum, difference, product, quotient or shift of bounded values, a conversion of one, or ``NonUniformResourceIndex``
  of one. Products, quotients and shifts need operands that are known not to be negative.

An index with neither property, into an array whose size has to be inferred, is refused with
:class:`ResourceArrayIndexUnboundedError`. A valid source program indexes only the resources the host bound, so a
negative or larger index is out of bounds there too; the bound covers every index such a program can use.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    RangeNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.rewrite import child_slots, get_child
from .array_utils import evaluate_literal_int_expression

__all__ = ["DynamicIndexSizes", "ResourceArrayIndexUnboundedError"]

# Calls that return their (first) argument's value, or a conversion of it.
_TRANSPARENT_CALLS = frozenset(
    {
        "NonUniformResourceIndex",
        "nonuniformEXT",
        "uint",
        "int",
        "uint32_t",
        "int32_t",
        "asuint",
        "asint",
    }
)
# Built-in calls whose result depends only on their arguments and that write none of them.
_PURE_CALLS = _TRANSPARENT_CALLS | frozenset({"min", "max", "clamp", "abs"})
_CONSTRUCTOR = re.compile(
    r"^(?:u?int|bool|float|half|double|[iubd]?vec|u?short|u?long)\d*(?:x\d+)?(?:_t)?$"
)
_LOOPS = (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)
_WRITING_QUALIFIERS = frozenset({"out", "inout"})


class ResourceArrayIndexUnboundedError(TranslationDiagnostic):
    """An unsized resource array indexed with a value whose range gives the array no size."""

    code = f"{CODE_PREFIX}resource-array-index-unbounded"


def _call_name(node):
    function = getattr(node, "function", None)
    if function is None:
        function = getattr(node, "name", None)
    if isinstance(function, str):
        return function
    return getattr(function, "name", None)


def _arguments(node):
    return list(getattr(node, "arguments", None) or [])


def _walk(node):
    """Every AST node under ``node``, ``node`` first."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
            continue
        if current is None or not hasattr(current, "__dict__"):
            continue
        yield current
        children = [get_child(current, slot, key) for slot, key in child_slots(current)]
        stack.extend(reversed(children))


def _root_name(target):
    """The variable an assignment target writes: ``v`` for ``v``, ``v.x`` and ``v[i].y``."""
    while isinstance(target, (MemberAccessNode, ArrayAccessNode)):
        target = (
            target.object_expr
            if isinstance(target, MemberAccessNode)
            else target.array_expr
        )
    return target.name if isinstance(target, IdentifierNode) else None


def _parameter_names(function):
    return [
        getattr(parameter, "name", None)
        for parameter in getattr(function, "parameters", None) or []
    ]


def _pattern_name(node):
    pattern = getattr(node, "pattern", None)
    if isinstance(pattern, IdentifierNode):
        return pattern.name
    return pattern if isinstance(pattern, str) else None


def _declared_names(function):
    """Each local's number of declarations, counting the variables of ``for ... in`` loops."""
    counts = {}
    for node in _walk(getattr(function, "body", None)):
        name = (
            node.name
            if isinstance(node, VariableNode)
            else _pattern_name(node) if isinstance(node, ForInNode) else None
        )
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _identifiers(node):
    return {child.name for child in _walk(node) if isinstance(child, IdentifierNode)}


def _join(intervals, pick_low, pick_high):
    lows = [interval[0] for interval in intervals]
    highs = [interval[1] for interval in intervals]
    return (None if None in lows else pick_low(lows), pick_high(highs))


def _constant(node, constants):
    try:
        return evaluate_literal_int_expression(node, constants)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _strip_conversions(node):
    """The index without the conversions around it: a converted negative or wrapped index is out of bounds in the
    source as well, so only the values below the bound matter there."""
    while True:
        if isinstance(node, CastNode):
            node = node.expression
        elif (
            isinstance(node, FunctionCallNode)
            and _call_name(node) in _TRANSPARENT_CALLS
            and len(_arguments(node)) == 1
        ):
            node = node.arguments[0]
        else:
            return node


def _increments(update, name):
    if isinstance(update, UnaryOpNode) and update.operator == "++":
        return (
            isinstance(update.operand, IdentifierNode) and update.operand.name == name
        )
    if isinstance(update, AssignmentNode) and update.operator == "+=":
        step = _constant(update.value, {})
        return (
            isinstance(update.target, IdentifierNode)
            and update.target.name == name
            and step is not None
            and step >= 0
        )
    return False


class _Known:
    """An operand whose range is already known."""

    def __init__(self, bound):
        self.range = bound


_STEP = LiteralNode(1, None)


class _Operands:
    def __init__(self, left, right):
        self.left = left
        self.right = right


class _Bounds:
    """Value ranges of integer index expressions in one function.

    A range is ``(low, high)``: every value the expression takes lies in ``[low, high]``. ``low`` is ``None`` when the
    value may be negative, or may have wrapped through an unsigned conversion; multiplication, division and shifts
    need a non-negative ``low``, because a wrapped value would not stay below the bound they compute. ``None`` for
    the whole range means no bound is known.
    """

    def __init__(self, constants, stable, changed_in, names=None, call_range=None):
        self.constants = constants
        self.stable = stable  # locals no statement changes after their only declaration
        self.changed_in = changed_in  # statement -> names it may change
        self.names = dict(
            names or {}
        )  # bounded parameters, induction variables and locals -> range
        self.call_range = call_range  # (user function, argument ranges) -> range of its results, or None

    def constant(self, node):
        if isinstance(node, _Known):
            low, high = node.range or (None, None)
            return low if low is not None and low == high else None
        return _constant(node, self.constants)

    def of(self, node):
        if isinstance(node, _Known):
            return node.range
        value = self.constant(node)
        if value is not None:
            return (value, value)
        if isinstance(node, IdentifierNode):
            return self.names.get(node.name)
        if isinstance(node, MemberAccessNode):
            # A component or swizzle of a vector lies in the range of the whole vector.
            return self.of(node.object_expr)
        if isinstance(node, RangeNode):
            start, end = self.of(node.start), self.of(node.end)
            if start is None or end is None:
                return None
            return (start[0], end[1] - (0 if node.inclusive else 1))
        if isinstance(node, CastNode):
            return self.of(node.expression)
        if isinstance(node, TernaryOpNode):
            branches = (self.of(node.true_expr), self.of(node.false_expr))
            return None if None in branches else _join(branches, min, max)
        if isinstance(node, FunctionCallNode):
            return self.call(node)
        if isinstance(node, BinaryOpNode):
            return self.binary(node)
        return None

    def call(self, node):
        name = _call_name(node)
        arguments = [self.of(argument) for argument in _arguments(node)]
        if self.call_range is not None:
            result = self.call_range(name, arguments)
            if result is not False:
                return result
        if name in _TRANSPARENT_CALLS and len(arguments) == 1:
            return arguments[0]
        if name and _CONSTRUCTOR.match(name) and arguments:
            # A vector's range covers all its components.
            return None if None in arguments else _join(arguments, min, max)
        if name == "min" and len(arguments) == 2:
            known = [argument for argument in arguments if argument is not None]
            if not known:
                return None
            low = None if None in arguments else min(argument[0] for argument in known)
            return (low, min(argument[1] for argument in known))
        if name == "max" and len(arguments) == 2:
            if None in arguments:
                return None
            lows = [argument[0] for argument in arguments if argument[0] is not None]
            return (
                max(lows) if lows else None,
                max(argument[1] for argument in arguments),
            )
        if name == "clamp" and len(arguments) == 3:
            # clamp(x, lo, hi) is min(max(x, lo), hi).
            lower, upper = arguments[1], arguments[2]
            if upper is None:
                return None
            low = (
                None
                if lower is None or None in (lower[0], upper[0])
                else min(lower[0], upper[0])
            )
            return (low, upper[1])
        return None

    def binary(self, node):
        return self.apply(node.operator, node.left, node.right)

    def apply(self, operator, left_node, right_node):
        node = _Operands(left_node, right_node)
        if operator == "%":
            divisor = self.constant(node.right)
            if divisor is None or divisor <= 0:
                return None
            dividend = self.of(node.left)
            if dividend is not None and dividend[0] is not None and dividend[0] >= 0:
                return (0, min(dividend[1], divisor - 1))
            return (1 - divisor, divisor - 1)
        if operator == "&":
            masks = [
                value
                for value in (self.constant(node.left), self.constant(node.right))
                if value is not None and value >= 0
            ]
            return (0, min(masks)) if masks else None
        left, right = self.of(node.left), self.of(node.right)
        if left is None or right is None:
            return None
        if operator == "+":
            low = None if None in (left[0], right[0]) else left[0] + right[0]
            return (low, left[1] + right[1])
        if operator == "-":
            if right[0] is None:
                return None
            low = None if left[0] is None else left[0] - right[1]
            return (low, left[1] - right[0])
        if left[0] is None or left[0] < 0 or right[0] is None or right[0] < 0:
            return None
        if operator == "*":
            return (left[0] * right[0], left[1] * right[1])
        amount = self.constant(node.right)
        if amount is None or (operator == "/" and amount == 0):
            return None
        if operator == "/":
            return (left[0] // amount, left[1] // amount)
        if operator == ">>":
            return (left[0] >> amount, left[1] >> amount)
        if operator == "<<":
            return (left[0] << amount, left[1] << amount)
        return None

    def declare(self, node):
        """Bound a local that is declared once, with an initializer, and never changed, by its initializer where it
        is declared: inside a loop that can be tighter than its range over the whole function.
        """
        initial = getattr(node, "initial_value", None)
        if node.name not in self.stable or initial is None:
            return
        bound = self.of(initial)
        if bound is not None:
            self.names[node.name] = bound

    def enter_loop(self, node):
        """The induction variable a canonical ``for`` loop bounds, with its range, or ``None``."""
        init, condition = node.init, node.condition
        if isinstance(init, VariableNode):
            name, start = init.name, getattr(init, "initial_value", None)
        elif (
            isinstance(init, AssignmentNode)
            and isinstance(init.target, IdentifierNode)
            and init.operator == "="
        ):
            name, start = init.target.name, init.value
        else:
            return None
        start = self.of(start)
        if start is None or not isinstance(condition, BinaryOpNode):
            return None
        if not (
            isinstance(condition.left, IdentifierNode) and condition.left.name == name
        ):
            return None
        limit = self.constant(condition.right)
        if limit is None or condition.operator not in ("<", "<="):
            return None
        if name in self.changed_in(node.body):
            return None
        # The condition holds whenever the body runs, and the body does not change the variable. It only grows
        # from its start when the update only increments it.
        low = start[0] if _increments(node.update, name) else None
        return name, (low, limit - 1 if condition.operator == "<" else limit)


class _Taint:
    """Which values of one function may not derive from compile-time constants alone.

    Flow-insensitive: a name is dynamic when any value stored to it is, when a call may write it, or when a loop
    whose trip count is dynamic stores to it. A branch only selects between the values its arms store, so its
    condition does not make them dynamic.
    """

    def __init__(self, program, function, dynamic_parameters):
        self.program = program
        self.constants = program.constants(function)
        self.parameters = set(_parameter_names(function))
        self.locals = set(_declared_names(function))
        self.dynamic = set(dynamic_parameters)
        self.solve(function)

    def solve(self, function):
        sources = []
        loops = []
        for node in _walk(function.body):
            if isinstance(node, VariableNode) and node.initial_value is not None:
                sources.append((node.name, node.initial_value))
            elif isinstance(node, AssignmentNode):
                name = _root_name(node.target)
                if name is not None:
                    sources.append((name, node.value))
            elif isinstance(node, FunctionCallNode):
                self.dynamic.update(self.program.written_arguments(node))
            elif isinstance(node, ForInNode):
                pattern = _pattern_name(node)
                if pattern is not None:
                    sources.append((pattern, node.iterable))
                loops.append((node.iterable, self.program.changed_in(node.body)))
            if isinstance(node, (ForNode, WhileNode, DoWhileNode, LoopNode)):
                changed = self.program.changed_in(node.body)
                if isinstance(node, ForNode):
                    changed = changed | self.program.changed_in(node.update)
                loops.append((getattr(node, "condition", None), changed))
            if isinstance(node, _LOOPS) and id(node) in self.program.approximate_loops:
                # The shared inference followed one iteration of this loop, so what the loop stores is not exact.
                changed = self.program.changed_in(node.body)
                if isinstance(node, ForNode):
                    changed = changed | self.program.changed_in(node.update)
                if isinstance(node, ForInNode) and _pattern_name(node) is not None:
                    changed = changed | {_pattern_name(node)}
                self.dynamic.update(changed)
        changed = True
        while changed:
            changed = False
            for name, value in sources:
                if name not in self.dynamic and self.of(value):
                    self.dynamic.add(name)
                    changed = True
            for condition, names in loops:
                if (
                    condition is None or self.of(condition)
                ) and not names <= self.dynamic:
                    self.dynamic.update(names)
                    changed = True

    def of(self, node):
        """Whether ``node`` may take a value that does not derive from compile-time constants."""
        if node is None or isinstance(node, LiteralNode):
            return False
        if _constant(node, self.constants) is not None:
            return False
        if isinstance(node, IdentifierNode):
            if node.name in self.dynamic:
                return True
            return not (
                node.name in self.locals
                or node.name in self.parameters
                or node.name in self.constants
            )
        if isinstance(node, MemberAccessNode):
            return self.of(node.object_expr)
        if isinstance(node, (BinaryOpNode, RangeNode)):
            left = getattr(node, "left", getattr(node, "start", None))
            right = getattr(node, "right", getattr(node, "end", None))
            return self.of(left) or self.of(right)
        if isinstance(node, UnaryOpNode):
            return self.of(node.operand)
        if isinstance(node, CastNode):
            return self.of(node.expression)
        if isinstance(node, TernaryOpNode):
            return self.of(node.true_expr) or self.of(node.false_expr)
        if isinstance(node, FunctionCallNode):
            return self.call(node)
        return True

    def call(self, node):
        name = _call_name(node)
        arguments = _arguments(node)
        if name in self.program.by_name:
            summary = self.program.summaries.get(name)
            if summary is None:
                return True
            intrinsic, depends = summary
            return intrinsic or any(
                self.of(argument)
                for position, argument in enumerate(arguments)
                if position in depends
            )
        if name in _PURE_CALLS or (name and _CONSTRUCTOR.match(name)):
            return any(self.of(argument) for argument in arguments)
        return True


class _Program:
    """The user functions of a shader, their call graph, and what the analyses learn about them."""

    def __init__(
        self, functions, global_arrays, function_arrays, constants, approximate_loops=()
    ):
        self.functions = [
            function
            for function in functions
            if getattr(function, "body", None) is not None
            and getattr(function, "name", None)
        ]
        self.by_name = {}
        for function in self.functions:
            self.by_name.setdefault(function.name, []).append(function)
        self.global_arrays = set(global_arrays)
        self.function_arrays = {
            name: set(parameters or ())
            for name, parameters in (function_arrays or {}).items()
        }
        self.constants = constants
        self.approximate_loops = frozenset(approximate_loops)
        self.summaries = (
            {}
        )  # function name -> (result dynamic by itself, parameter positions it depends on)
        self.call_ranges = (
            {}
        )  # (function name, argument ranges) -> range of its results
        self.arguments = (
            {}
        )  # (function name, parameter position) -> range, or None when unbounded
        self.dynamic_arguments = (
            set()
        )  # (function name, parameter position) some caller passes a dynamic value for
        self.called = set()
        self.cyclic = set()
        self.accesses = {}  # array key -> required size, or None when unbounded
        self.where = {}  # array key -> function that indexes it without a bound
        self.bindings = (
            {}
        )  # ("param", function name, parameter) -> caller array keys passed to it

    def order(self):
        """Function names with every caller before its callees; functions on a call cycle come last."""
        callees = {name: set() for name in self.by_name}
        for function in self.functions:
            for node in _walk(function.body):
                if (
                    isinstance(node, FunctionCallNode)
                    and _call_name(node) in self.by_name
                ):
                    callees[function.name].add(_call_name(node))
        incoming = {name: 0 for name in callees}
        for name in sorted(callees):
            for callee in callees[name] - {name}:
                incoming[callee] += 1
        ready = sorted(name for name, count in incoming.items() if count == 0)
        ordered = []
        while ready:
            name = ready.pop(0)
            ordered.append(name)
            for callee in sorted(callees[name] - {name}):
                incoming[callee] -= 1
                if incoming[callee] == 0:
                    ready.append(callee)
        # Functions on a call cycle, and their callees, may be analyzed before all their callers are.
        self.cyclic = set(callees) - set(ordered)
        return ordered + sorted(self.cyclic)

    def written_arguments(self, node):
        """The names a call may write: bare arguments for ``out``/``inout`` parameters of user functions, and every
        bare argument of a built-in that is not known to be pure."""
        name = _call_name(node)
        arguments = _arguments(node)
        if name in _PURE_CALLS or (name and _CONSTRUCTOR.match(name)):
            return set()
        if name in self.by_name:
            positions = set()
            for function in self.by_name[name]:
                for position, parameter in enumerate(
                    getattr(function, "parameters", None) or []
                ):
                    qualifiers = {
                        str(qualifier)
                        for qualifier in getattr(parameter, "qualifiers", None) or []
                    }
                    if qualifiers & _WRITING_QUALIFIERS:
                        positions.add(position)
        else:
            positions = set(range(len(arguments)))
        return {
            argument.name
            for position, argument in enumerate(arguments)
            if position in positions and isinstance(argument, IdentifierNode)
        }

    def changed_in(self, node):
        """Names a statement may change: assignment and increment targets, and arguments calls may write."""
        names = set()
        for child in _walk(node):
            if isinstance(child, AssignmentNode):
                name = _root_name(child.target)
                if name is not None:
                    names.add(name)
            elif isinstance(child, UnaryOpNode) and child.operator in ("++", "--"):
                name = _root_name(child.operand)
                if name is not None:
                    names.add(name)
            elif isinstance(child, FunctionCallNode):
                names.update(self.written_arguments(child))
        return names

    def stable_locals(self, function):
        """Locals declared exactly once in ``function`` that no statement changes afterwards."""
        parameters = set(_parameter_names(function))
        changed = self.changed_in(function.body)
        return {
            name
            for name, count in _declared_names(function).items()
            if count == 1 and name not in parameters and name not in changed
        }

    def stores(self, function):
        """Every value stored to each local or parameter: ``(operator, operand)``, ``operator`` ``None`` for a plain
        store, and ``(None, None)`` for a write the analysis cannot follow."""
        stores = {}
        for node in _walk(function.body):
            if isinstance(node, VariableNode):
                if node.initial_value is not None:
                    stores.setdefault(node.name, []).append((None, node.initial_value))
            elif isinstance(node, ForInNode):
                pattern = _pattern_name(node)
                if pattern is not None:
                    stores.setdefault(pattern, []).append((None, node.iterable))
            elif isinstance(node, AssignmentNode):
                name = _root_name(node.target)
                if name is None:
                    continue
                operator = node.operator[:-1] if node.operator != "=" else None
                if not isinstance(node.target, IdentifierNode) and operator is not None:
                    stores.setdefault(name, []).append((None, None))
                else:
                    stores.setdefault(name, []).append((operator, node.value))
            elif isinstance(node, UnaryOpNode) and node.operator in ("++", "--"):
                name = _root_name(node.operand)
                if name is not None:
                    operator = "+" if node.operator == "++" else "-"
                    if isinstance(node.operand, IdentifierNode):
                        stores.setdefault(name, []).append((operator, _STEP))
                    else:
                        stores.setdefault(name, []).append((None, None))
            elif isinstance(node, FunctionCallNode):
                for name in self.written_arguments(node):
                    stores.setdefault(name, []).append((None, None))
        return stores

    def local_ranges(self, function, parameters, call_range):
        """Flow-insensitive ranges of a function's locals and parameters: the union of every value stored to each.

        ``parameters`` holds the ranges callers pass. A name with a store whose range grows on every round (a counter
        in a loop) or that the analysis cannot follow is unbounded.
        """
        stores = self.stores(function)
        bottom = object()
        state = {name: bottom for name in stores}
        for name in _parameter_names(function):
            state[name] = parameters.get(name)
        bounds = _Bounds(
            self.constants(function), set(), self.changed_in, {}, call_range
        )

        def evaluate(operator, operand, current):
            if operand is None:
                return None
            if operator is None:
                return bounds.of(operand)
            if current is None:
                return None
            return bounds.apply(operator, _Known(current), operand)

        # Ranges only grow, and after the widening round only to unbounded, so this settles.
        for round_number in range(8 + 2 * len(state)):
            bounds.names = {
                name: bound
                for name, bound in state.items()
                if bound is not bottom and bound is not None
            }
            pending = {name for name, bound in state.items() if bound is bottom}
            changed = False
            for name in sorted(stores):
                if state[name] is None:
                    continue
                contributions = []
                for operator, operand in stores[name]:
                    if operand is not None and _identifiers(operand) & pending:
                        continue  # depends on a name with no value yet
                    current = state[name] if state[name] is not bottom else None
                    if operator is not None and state[name] is bottom:
                        continue
                    contributions.append(evaluate(operator, operand, current))
                if not contributions:
                    continue
                joined = (
                    None if None in contributions else _join(contributions, min, max)
                )
                if (
                    state[name] is not bottom
                    and state[name] is not None
                    and joined is not None
                ):
                    joined = _join((state[name], joined), min, max)
                if joined != state[name]:
                    # Widen a range that still grows after several rounds.
                    state[name] = joined if round_number < 6 else None
                    changed = True
            if not changed:
                break
        return {
            name: bound
            for name, bound in state.items()
            if bound is not bottom and bound is not None
        }

    def array_key(self, function, name):
        if name in self.function_arrays.get(
            function.name, ()
        ) and name in _parameter_names(function):
            return ("param", function.name, name)
        shadowing = set(_parameter_names(function)) | set(_declared_names(function))
        if name in self.global_arrays and name not in shadowing:
            return ("global", name)
        return None

    def known_parameters(self, function):
        return function.name in self.called and function.name not in self.cyclic

    def summarize(self, ordered):
        """Callees first: whether each function's result is dynamic, by itself or through which parameters."""
        for name in reversed(ordered):
            if name in self.cyclic:
                continue
            intrinsic, depends = False, set()
            for function in self.by_name[name]:
                returns = [
                    node.value
                    for node in _walk(function.body)
                    if isinstance(node, ReturnNode)
                    and getattr(node, "value", None) is not None
                ]
                base = _Taint(self, function, ())
                intrinsic = intrinsic or any(base.of(value) for value in returns)
                for position, parameter in enumerate(_parameter_names(function)):
                    tainted = _Taint(self, function, {parameter})
                    if any(tainted.of(value) for value in returns):
                        depends.add(position)
            self.summaries[name] = (intrinsic, depends)

    def call_range(self, name, ranges, stack):
        """The range of what user function ``name`` returns for arguments in ``ranges``; ``False`` for a call that
        is not to a user function, ``None`` when unbounded."""
        if name not in self.by_name:
            return False
        if name in stack or name in self.cyclic:
            return None
        key = (name, tuple(ranges))
        if key not in self.call_ranges:
            results = []
            for function in self.by_name[name]:
                parameters = _parameter_names(function)
                if len(parameters) != len(ranges):
                    continue
                names = {
                    parameter: bound
                    for parameter, bound in zip(parameters, ranges)
                    if bound is not None
                }
                results.extend(
                    self.analyze(function, False, names=names, stack=stack + (name,))
                )
            self.call_ranges[key] = (
                _join(results, min, max) if results and None not in results else None
            )
        return self.call_ranges[key]

    def parameter_ranges(self, function):
        if not self.known_parameters(function):
            return {}
        ranges = {}
        for position, name in enumerate(_parameter_names(function)):
            bound = self.arguments.get((function.name, position))
            if bound is not None:
                ranges[name] = bound
        return ranges

    def dynamic_parameters(self, function):
        names = _parameter_names(function)
        if not self.known_parameters(function):
            return set(names)
        return {
            name
            for position, name in enumerate(names)
            if (function.name, position) in self.dynamic_arguments
        }

    def require(self, key, bound, function):
        if bound is None:
            self.accesses[key] = None
            self.where.setdefault(key, function.name)
        elif key not in self.accesses or self.accesses[key] is not None:
            self.accesses[key] = max(self.accesses.get(key) or 0, bound[1] + 1, 1)

    def record_call(self, function, node, bounds, taint):
        callee = _call_name(node)
        arguments = _arguments(node)
        self.called.add(callee)
        for target in self.by_name[callee]:
            parameters = _parameter_names(target)
            if len(parameters) != len(arguments):
                continue
            for position, argument in enumerate(arguments):
                key = (callee, position)
                bound = bounds.of(argument)
                if key in self.arguments:
                    previous = self.arguments[key]
                    bound = (
                        None
                        if previous is None or bound is None
                        else _join((previous, bound), min, max)
                    )
                self.arguments[key] = bound
                if taint.of(argument):
                    self.dynamic_arguments.add(key)
                if parameters[position] in self.function_arrays.get(
                    callee, ()
                ) and isinstance(argument, IdentifierNode):
                    caller_key = self.array_key(function, argument.name)
                    if caller_key is not None:
                        self.bindings.setdefault(
                            ("param", callee, parameters[position]), set()
                        ).add(caller_key)

    def analyze(
        self,
        function,
        record,
        resolved=frozenset(),
        shared_keys=None,
        names=None,
        stack=(),
    ):
        """Walk ``function`` and return the ranges of its results. With ``record`` set, note what its array accesses
        need and what its calls pass; otherwise ``names`` holds its parameters' ranges.
        """

        def call_range(callee, ranges):
            return self.call_range(callee, ranges, stack + (function.name,))

        parameters = self.parameter_ranges(function) if record else (names or {})
        bounds = _Bounds(
            self.constants(function),
            self.stable_locals(function),
            self.changed_in,
            self.local_ranges(function, parameters, call_range),
            call_range,
        )
        taint = (
            _Taint(self, function, self.dynamic_parameters(function))
            if record
            else None
        )
        results = []

        def check(node):
            key = self.array_key(function, node.array_expr.name)
            if key is None:
                return
            index = node.index_expr
            if (
                id(node) in resolved
                and key in (shared_keys or ())
                and not taint.of(index)
            ):
                return  # the shared inference evaluated every value this index can take
            self.require(key, bounds.of(_strip_conversions(index)), function)

        def visit(node):
            if isinstance(node, list):
                for item in node:
                    visit(item)
                return
            if node is None or not hasattr(node, "__dict__"):
                return
            if isinstance(node, ForNode):
                saved = dict(bounds.names)
                visit(node.init)
                loop = bounds.enter_loop(node)
                visit(node.condition)
                visit(node.update)
                if loop is not None:
                    bounds.names[loop[0]] = loop[1]
                visit(node.body)
                bounds.names = saved
                return
            if isinstance(node, VariableNode):
                visit(getattr(node, "initial_value", None))
                bounds.declare(node)
                return
            if (
                isinstance(node, ReturnNode)
                and getattr(node, "value", None) is not None
            ):
                results.append(bounds.of(node.value))
            if (
                record
                and isinstance(node, FunctionCallNode)
                and _call_name(node) in self.by_name
            ):
                self.record_call(function, node, bounds, taint)
            if (
                record
                and isinstance(node, ArrayAccessNode)
                and isinstance(node.array_expr, IdentifierNode)
                and bounds.constant(node.index_expr) is None
            ):
                check(node)
            for slot, key in child_slots(node):
                visit(get_child(node, slot, key))

        visit(function.body)
        return results

    def propagate(self, ordered):
        """Carry what a callee needs of an array parameter to the arrays its callers pass."""
        for name in reversed(ordered):
            for key in sorted(
                (key for key in self.accesses if key[0] == "param" and key[1] == name),
                key=repr,
            ):
                size = self.accesses[key]
                for caller_key in sorted(self.bindings.get(key, ()), key=repr):
                    if size is None:
                        self.accesses[caller_key] = None
                        self.where.setdefault(caller_key, self.where.get(key, name))
                    elif (
                        caller_key not in self.accesses
                        or self.accesses[caller_key] is not None
                    ):
                        self.accesses[caller_key] = max(
                            self.accesses.get(caller_key) or 0, size
                        )


class DynamicIndexSizes:
    """The sizes unsized global resource arrays need for the indexes the shared size inference cannot evaluate.

    ``global_arrays`` names the unsized global arrays and ``function_arrays`` maps a function name to its unsized
    array parameters; ``constants(function)`` returns the integer constants visible in a function.
    ``shared_globals`` names the global arrays the shared inference sizes (default: all of ``global_arrays``).

    Calling the object with ``resolved``, the ids of the access nodes the shared inference evaluated in every context,
    and ``approximate_loops``, the loops it did not unroll everywhere
    (:func:`resource_arrays.collect_resource_array_size_hints` does), returns ``{name: size}`` and keeps it in
    :attr:`sizes`. An index into a parameter counts against every array the callers pass for it. Raises
    :class:`ResourceArrayIndexUnboundedError` when an index into an array that reaches an unsized global has no bound.
    """

    def __init__(
        self,
        *,
        global_arrays,
        function_arrays,
        functions,
        constants,
        target,
        shared_globals=None,
    ):
        self.global_arrays = set(global_arrays)
        self.shared_globals = (
            self.global_arrays if shared_globals is None else set(shared_globals)
        )
        self.function_arrays = function_arrays
        self.functions = functions
        self.constants = constants
        self.target = target
        self.sizes = None

    def __call__(self, resolved=frozenset(), approximate_loops=frozenset()):
        program = _Program(
            self.functions,
            self.global_arrays,
            self.function_arrays,
            self.constants,
            approximate_loops,
        )
        self.sizes = {}
        if not program.global_arrays:
            return self.sizes
        shared_keys = {("global", name) for name in self.shared_globals} | {
            ("param", function, parameter)
            for function, parameters in program.function_arrays.items()
            for parameter in parameters
        }
        ordered = program.order()
        program.summarize(ordered)
        for name in ordered:
            for function in program.by_name[name]:
                program.analyze(function, True, resolved, shared_keys)
        program.propagate(ordered)
        for key in sorted(
            (key for key in program.accesses if key[0] == "global"), key=repr
        ):
            size = program.accesses[key]
            if size is None:
                raise ResourceArrayIndexUnboundedError(
                    f"the unsized resource array {key[1]} is indexed in {program.where[key]} with a value whose "
                    "range gives the array no size; declare the array with a size",
                    target=self.target,
                )
            self.sizes[key[1]] = size
        return self.sizes

    def result(self):
        """:attr:`sizes`, computed without shared resolutions when the shared inference never asked for them."""
        return self.sizes if self.sizes is not None else self()
