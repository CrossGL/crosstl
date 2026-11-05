"""Guarded subscripts through rebased storage pointers whose base may be negative.

A block load that walks a line from its end addresses the block as
``in + size - offset - N``. For the last blocks that base lies before the
buffer, and every read ``input[i]`` is guarded by ``start + N - i - 1 <
total``, where the helper's ``start`` and ``total`` carry ``offset`` and
``size``. Intervals alone cannot bound ``base + i`` from below; the relation
between the base and the helper's scalar parameters can.

A call site records the base as an affine function of the helper's by-value
integer parameters plus a bounded residual. Inside the helper, a subscript's
complete offset is bounded by the comparisons that dominate it, each taken
with weight zero or one, and by interval facts for the values that do not
cancel. Pointer arithmetic moves by each delta's exact value, so the
composed base is the exact sum of its deltas.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from ..arithmetic_conversions import ArithmeticScalarKind, target_arithmetic_type
from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    WhileNode,
)
from .index_range_contracts import INDEX_RANGE_STATIC, IntegerRange

INT_MIN = -(1 << 31)
INT_MAX = (1 << 31) - 1
# Bounds the weight search: single facts and pairs of at most this many facts.
MAX_GUARD_FACTS = 16
COMPARISONS = {"<", "<=", ">", ">=", "=="}
NEGATED = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


def composition_terms(offset: Any) -> list[tuple[int, Any]] | None:
    """Return the signed deltas of a composed pointer offset, or ``None``."""
    if not getattr(offset, "_glsl_pointer_offset_composition", False):
        return None
    terms: list[tuple[int, Any]] = []

    def flatten(node: Any, sign: int) -> bool:
        if getattr(node, "_glsl_pointer_offset_composition", False):
            if isinstance(node, BinaryOpNode):
                return flatten(node.left, sign) and flatten(node.right, sign)
            if isinstance(node, UnaryOpNode):
                return flatten(node.operand, -sign)
            return False
        terms.append((sign, node))
        return True

    return terms if flatten(offset, 1) else None


def _signed_range(codegen: Any, sign: int, term: Any) -> tuple[int, int] | None:
    if isinstance(term, str):
        return None
    value = codegen.glsl_index_value_range(term)
    if value is None:
        return None
    if sign > 0:
        return value.minimum, value.maximum
    return -value.maximum, -value.minimum


def composed_offset_range(codegen: Any, offset: Any) -> IntegerRange | None:
    """Return the exact range of a composed offset from its deltas' ranges.

    Each delta keeps the facts of its own source location, so a base offset
    lowered from an earlier rebase is not evaluated under a later binding of
    the same name.
    """
    terms = composition_terms(offset)
    if terms is None:
        return None
    low = high = 0
    for sign, term in terms:
        bounds = _signed_range(codegen, sign, term)
        if bounds is None:
            return None
        low, high = low + bounds[0], high + bounds[1]
    return IntegerRange(low, high, INDEX_RANGE_STATIC, "exact pointer offset sum")


def _scalar_domain(codegen: Any, type_name: Any) -> tuple[int, int] | None:
    info = codegen.glsl_value_type_info(type_name)
    if info is None or info["width"] != 1 or info["family"] not in {"int", "uint"}:
        return None
    bits = info["narrow"]["bits"] if info["narrow"] is not None else info["bits"]
    if info["family"] == "int":
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _within(bounds: Any, domain: Any) -> bool:
    return (
        bounds is not None
        and domain is not None
        and domain[0] <= bounds[0] <= bounds[1] <= domain[1]
    )


def _root_identifier(node: Any) -> Any:
    while True:
        if isinstance(node, MemberAccessNode):
            node = node.object
        elif isinstance(node, ArrayAccessNode):
            node = node.array
        elif isinstance(node, UnaryOpNode) and node.op in {"&", "*"}:
            node = node.operand
        else:
            return node if isinstance(node, IdentifierNode) else None


def _children(node: Any):
    if isinstance(node, (list, tuple)):
        yield from node
    elif hasattr(node, "child_nodes"):
        yield from node.child_nodes()


def _walk(node: Any):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))
        if not isinstance(current, (list, tuple)):
            yield current
        stack.extend(_children(current))


def _mutated_roots(codegen: Any, region: Any) -> set[str]:
    """Names any write in ``region`` may change, including by out arguments."""
    names: set[str] = set()

    def add(node: Any) -> None:
        root = _root_identifier(node)
        if root is not None:
            names.add(root.name)

    for node in _walk(region):
        if isinstance(node, AssignmentNode):
            add(getattr(node, "target", getattr(node, "left", None)))
        elif isinstance(node, UnaryOpNode) and (
            codegen.map_operator(node.op) in {"++", "--", "&"}
            or node.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }
        ):
            add(node.operand)
        elif isinstance(node, FunctionCallNode):
            if codegen.glsl_integer_scalar_cast_call(node) is not None:
                continue
            for argument in node.arguments or []:
                add(argument)
    return names


def _rebound_names(codegen: Any, region: Any) -> set[str]:
    """Names ``region`` may rebind; writes through a pointer keep its value."""
    names: set[str] = set()
    for node in _walk(region):
        if isinstance(node, AssignmentNode):
            target = getattr(node, "target", getattr(node, "left", None))
        elif isinstance(node, UnaryOpNode) and (
            codegen.map_operator(node.op) in {"++", "--", "&"}
            or node.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }
        ):
            target = node.operand
        elif isinstance(node, FunctionCallNode):
            for argument in node.arguments or []:
                if isinstance(argument, IdentifierNode):
                    names.add(argument.name)
            continue
        else:
            continue
        if isinstance(target, IdentifierNode):
            names.add(target.name)
    return names


def _affine(codegen: Any, node: Any, atom_key: Any, intervals: Any, constants: Any):
    """Return ``(coefficients, low, high)`` for an integer value, or ``None``.

    Arithmetic nodes must already be known not to wrap; integer conversions
    keep their operand only when its range fits the destination. Values that
    are not affine contribute their checked interval.
    """
    literal = codegen.literal_int_value(node, constants)
    if isinstance(literal, int) and not isinstance(literal, bool):
        return {}, literal, literal
    key = atom_key(node)
    if key is not None:
        return {key: 1}, 0, 0
    if isinstance(node, UnaryOpNode) and codegen.map_operator(node.op) in {"-", "+"}:
        inner = _affine(codegen, node.operand, atom_key, intervals, constants)
        if inner is None or codegen.map_operator(node.op) == "+":
            return inner
        return {k: -v for k, v in inner[0].items()}, -inner[2], -inner[1]
    if isinstance(node, BinaryOpNode) and codegen.map_operator(node.op) in {
        "+",
        "-",
        "*",
    }:
        operator = codegen.map_operator(node.op)
        left = _affine(codegen, node.left, atom_key, intervals, constants)
        right = _affine(codegen, node.right, atom_key, intervals, constants)
        if left is not None and right is not None and operator in {"+", "-"}:
            scale = 1 if operator == "+" else -1
            coefficients = dict(left[0])
            for k, v in right[0].items():
                coefficients[k] = coefficients.get(k, 0) + scale * v
            low, high = (right[1], right[2]) if scale > 0 else (-right[2], -right[1])
            return coefficients, left[1] + low, left[2] + high
        if left is not None and right is not None and operator == "*":
            for factor, other in ((left, right), (right, left)):
                if not factor[0] and factor[1] == factor[2]:
                    c = factor[1]
                    low, high = sorted((other[1] * c, other[2] * c))
                    return {k: v * c for k, v in other[0].items()}, low, high
    cast = codegen.glsl_integer_scalar_cast_call(node)
    if cast is not None:
        destination, (argument,) = cast
        bounds = codegen.glsl_checked_private_pointer_interval(
            argument, intervals, constants
        )
        if _within(bounds, _scalar_domain(codegen, destination)):
            return _affine(codegen, argument, atom_key, intervals, constants)
    bounds = codegen.glsl_checked_private_pointer_interval(node, intervals, constants)
    if bounds is None:
        return None
    return {}, bounds[0], bounds[1]


def _parameter_type(codegen: Any, parameter: Any) -> str:
    raw = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
    return codegen.type_name_string(raw) or ""


def _has_side_effects(codegen: Any, expression: Any) -> bool:
    """Whether evaluating ``expression`` may write; value constructors do not."""
    for node in _walk(expression):
        if isinstance(node, AssignmentNode):
            return True
        if isinstance(node, UnaryOpNode) and (
            codegen.map_operator(node.op) in {"++", "--"}
            or node.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }
        ):
            return True
        if isinstance(node, FunctionCallNode):
            name = codegen.function_call_name(node)
            if (
                name in getattr(codegen, "function_return_types", {})
                or codegen.glsl_value_type_info(name) is None
            ):
                return True
    return False


def call_offset_relation(
    codegen, callee, params, args, pointer_index, binding, local_names=None
):
    """Relate a composed pointer argument's offset to the callee's scalars.

    Returns ``(coefficients, residual, total)``: the offset equals
    ``sum(c * parameter) + r`` with ``r`` in ``residual``, and lies in
    ``total``, a range within ``int``. Only deltas written in this call's
    pointer argument are matched to parameters, and only to arguments of the
    same call, so each match names one value. No argument may write a name
    those arguments read; where one calls a function, they may read only the
    caller's locals (``local_names``), which a callee cannot reach.
    """
    terms = composition_terms(binding.get("offset"))
    if (
        terms is None
        or binding.get("pointer_reinterpretation") is not None
        or str(binding.get("byte_offset") or "0") not in {"0", "0u"}
        or binding.get("offset_direction")
        or len(params) != len(args)
    ):
        return None
    pointer_name = getattr(params[pointer_index], "name", None)
    body = getattr(callee, "body", None)
    written = _mutated_roots(codegen, body)
    if not pointer_name or pointer_name in _rebound_names(codegen, body):
        return None
    constants = codegen.glsl_active_literal_int_constants()

    def atom_name(node):
        if isinstance(node, IdentifierNode):
            return ("source", node.name)
        path = _member_path(node) if isinstance(node, MemberAccessNode) else None
        return ("source", path) if path else None

    substitutions = {}
    for index, (parameter, argument) in enumerate(zip(params, args)):
        name = getattr(parameter, "name", None)
        type_name = _parameter_type(codegen, parameter)
        if (
            index == pointer_index
            or not name
            or name in written
            or "&" in type_name
            or "*" in type_name
            or {"out", "inout"} & set(codegen.glsl_parameter_qualifiers(parameter))
        ):
            continue
        intervals = getattr(argument, "_glsl_control_flow_intervals", {})
        form = _affine(codegen, argument, atom_name, intervals, constants)
        bounds = codegen.glsl_checked_private_pointer_interval(
            argument, intervals, constants
        )
        if (
            form is None
            or len(form[0]) != 1
            or form[1] != form[2]
            or not _within(bounds, _scalar_domain(codegen, type_name))
        ):
            continue
        ((atom, sign),) = form[0].items()
        if sign in {1, -1}:
            # argument = sign * atom + c, so atom = sign * (parameter - c).
            substitutions.setdefault(atom, (index, name, sign, form[1]))

    argument_nodes = {id(node) for node in _walk(args[pointer_index])}
    coefficients = {}
    residual = [0, 0]
    total = [0, 0]
    for sign, term in terms:
        bounds = _signed_range(codegen, sign, term)
        if bounds is None:
            return None
        total = [total[0] + bounds[0], total[1] + bounds[1]]
        atom = atom_name(term) if id(term) in argument_nodes else None
        if atom in substitutions:
            index, name, atom_sign, constant = substitutions[atom]
            weight = sign * atom_sign
            coefficients[(index, name)] = coefficients.get((index, name), 0) + weight
            residual = [
                residual[0] - weight * constant,
                residual[1] - weight * constant,
            ]
        else:
            residual = [residual[0] + bounds[0], residual[1] + bounds[1]]
    coefficients = {key: value for key, value in coefficients.items() if value}
    if not coefficients or not INT_MIN <= total[0] <= total[1] <= INT_MAX:
        return None
    read = {
        node.name
        for index in (pointer_index, *(index for index, _ in coefficients))
        for node in _walk(args[index])
        if isinstance(node, IdentifierNode)
    }
    if read & _mutated_roots(codegen, list(args)):
        return None
    if any(_has_side_effects(codegen, argument) for argument in args) and not (
        read <= set(local_names or ())
    ):
        return None
    return (
        tuple((name, value) for (_, name), value in sorted(coefficients.items())),
        tuple(residual),
        tuple(total),
    )


def _member_path(node: Any) -> str | None:
    parts = []
    while isinstance(node, MemberAccessNode):
        parts.append(str(node.member))
        node = node.object
    if not isinstance(node, IdentifierNode):
        return None
    return ".".join([node.name, *reversed(parts)])


def _guard_index(codegen: Any, function: Any) -> dict[int, tuple]:
    """Map each subscript index node to the conditions that dominate it.

    Each entry is ``(condition, expected, region)``: ``condition`` evaluated
    to ``expected`` before ``region`` runs, and the subscript lies in it.
    """
    index: dict[int, tuple] = {}
    seen: set[int] = set()

    def visit(node: Any, guards: tuple) -> None:
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child, guards)
            return
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, IfNode):
            condition = getattr(node, "if_condition", node.condition)
            visit(condition, guards)
            visit(node.if_body, guards + ((condition, True, node.if_body),))
            rest = (
                list(node.else_if_conditions or []),
                list(node.else_if_bodies or []),
                node.else_body,
            )
            negated = guards + ((condition, False, rest),)
            for else_condition, else_body in zip(
                node.else_if_conditions or [], node.else_if_bodies or []
            ):
                visit(else_condition, negated)
                visit(else_body, negated + ((else_condition, True, else_body),))
                negated = negated + ((else_condition, False, rest),)
            visit(node.else_body, negated)
            return
        if isinstance(node, TernaryOpNode):
            visit(node.condition, guards)
            visit(node.true_expr, guards + ((node.condition, True, node.true_expr),))
            visit(node.false_expr, guards + ((node.condition, False, node.false_expr),))
            return
        if isinstance(node, BinaryOpNode) and codegen.map_operator(node.op) in {
            "&&",
            "||",
        }:
            visit(node.left, guards)
            expected = codegen.map_operator(node.op) == "&&"
            visit(node.right, guards + ((node.left, expected, node.right),))
            return
        if isinstance(node, (ForNode, WhileNode)):
            visit(getattr(node, "init", None), guards)
            visit(node.condition, guards)
            if node.condition is not None:
                visit(node.body, guards + ((node.condition, True, node.body),))
            else:
                visit(node.body, guards)
            visit(getattr(node, "update", None), guards)
            return
        if isinstance(node, ArrayAccessNode):
            index[id(node.index)] = guards
        for child in _children(node):
            visit(child, guards)

    visit(getattr(function, "body", None), ())
    return index


def _function_facts(codegen: Any, function: Any) -> dict:
    cache = codegen.__dict__.setdefault("_glsl_guarded_offset_facts", {})
    cached = cache.get(id(function))
    if cached is not None and cached["function"] is function:
        return cached
    lexical = codegen.glsl_scalar_array_view_lexical_bindings(function)
    declarations = lexical["declarations"]
    counts: dict[str, int] = {}
    for declaration in declarations.values():
        counts[declaration["name"]] = counts.get(declaration["name"], 0) + 1
    cached = {
        "function": function,
        "lexical": lexical,
        "name_counts": counts,
        "guards": _guard_index(codegen, function),
        "written": _mutated_roots(codegen, getattr(function, "body", None)),
    }
    cache[id(function)] = cached
    return cached


def _comparison_facts(codegen, condition, expected, atom_key, constants):
    """Nonnegative affine forms that hold when ``condition`` is ``expected``."""
    if isinstance(condition, UnaryOpNode) and codegen.map_operator(condition.op) == "!":
        return _comparison_facts(
            codegen, condition.operand, not expected, atom_key, constants
        )
    if not isinstance(condition, BinaryOpNode):
        return []
    operator = codegen.map_operator(condition.op)
    if operator in {"&&", "||"}:
        if expected != (operator == "&&"):
            return []
        return _comparison_facts(
            codegen, condition.left, expected, atom_key, constants
        ) + _comparison_facts(codegen, condition.right, expected, atom_key, constants)
    if not expected:
        operator = NEGATED.get(operator)
    if operator not in COMPARISONS:
        return []
    intervals = getattr(condition, "_glsl_control_flow_intervals", None) or {}
    plan = codegen.glsl_arithmetic_conversion_plan(
        codegen.glsl_source_expression_type(condition.left),
        codegen.glsl_source_expression_type(condition.right),
        codegen.map_operator(condition.op),
    )
    if plan is None:
        return []
    forms = []
    for operand, promoted in (
        (condition.left, plan.left_target_type),
        (condition.right, plan.right_target_type),
    ):
        # The comparison orders mathematical values only when neither operand
        # wraps and its promotion preserves it.
        bounds = codegen.glsl_checked_private_pointer_interval(
            operand, intervals, constants, require_arithmetic_types=True
        )
        info = target_arithmetic_type(promoted)
        if (
            bounds is None
            or info is None
            or info.lanes != 1
            or info.kind not in {
                ArithmeticScalarKind.SIGNED_INTEGER,
                ArithmeticScalarKind.UNSIGNED_INTEGER,
            }
        ):
            return []
        signed = info.kind == ArithmeticScalarKind.SIGNED_INTEGER
        low = -(1 << (info.bits - 1)) if signed else 0
        if not low <= bounds[0] <= bounds[1] <= (1 << (info.bits - int(signed))) - 1:
            return []
        form = _affine(codegen, operand, atom_key, intervals, constants)
        if form is None:
            return []
        forms.append(form)
    left, right = forms
    if operator in {">", ">="}:
        left, right = right, left
    # left < right  =>  right - left - 1 >= 0;  left <= right  =>  right - left >= 0.
    gap = 1 if operator in {"<", ">"} else 0
    facts = [_difference(right, left, gap)]
    if operator == "==":
        facts.append(_difference(left, right, 0))
    return facts


def _difference(left, right, constant=0):
    coefficients = dict(left[0])
    for key, value in right[0].items():
        coefficients[key] = coefficients.get(key, 0) - value
    return coefficients, left[1] - right[2] - constant, left[2] - right[1] - constant


def _proves_nonnegative(form, facts, interval_of) -> bool:
    candidates = [()]
    candidates += [(fact,) for fact in facts]
    candidates += list(combinations(facts, 2))
    for chosen in candidates:
        remainder = form
        for fact in chosen:
            remainder = _difference(remainder, fact)
        lower = remainder[1]
        for key, coefficient in remainder[0].items():
            if coefficient == 0:
                continue
            bounds = interval_of(key)
            if bounds is None:
                break
            lower += coefficient * (bounds[0] if coefficient > 0 else bounds[1])
        else:
            if lower >= 0:
                return True
    return False


def guarded_rebased_subscript(codegen, function, index_expr, indexed_expr, binding):
    """Prove ``0 <= offset + index <= INT_MAX`` from the subscript's guards."""
    relation = binding.get("guarded_offset_relation")
    if (
        relation is None
        or function is None
        or not isinstance(indexed_expr, IdentifierNode)
    ):
        return False
    alias = (getattr(function, "_glsl_storage_pointer_aliases", None) or {}).get(
        indexed_expr.name
    )
    offset = binding.get("offset")
    offset_name = getattr(offset, "name", offset)
    if (
        alias is None
        or alias.get("guarded_offset_relation") != relation
        or not isinstance(offset_name, str)
        or offset_name != alias.get("offset")
    ):
        return False
    facts = _function_facts(codegen, function)
    guards = facts["guards"].get(id(index_expr))
    if guards is None:
        return False
    lexical = facts["lexical"]
    declarations = lexical["declarations"]
    references = lexical["reference_bindings"]
    counts = facts["name_counts"]
    parameters = {
        declaration["name"]: key
        for key, declaration in declarations.items()
        if declaration["kind"] == "parameter"
    }
    coefficients, residual, _total = relation
    target = {}
    for name, coefficient in coefficients:
        key = parameters.get(name)
        if key is None or counts.get(name) != 1 or name in facts["written"]:
            return False
        target[("d", key)] = coefficient

    def atom_key(node):
        if isinstance(node, IdentifierNode):
            key = references.get(id(node))
            return None if key is None else ("d", key)
        return None

    constants = codegen.glsl_active_literal_int_constants()
    source_types = {
        declaration["name"]: declaration["source_type"]
        for declaration in declarations.values()
        if declaration.get("source_type") is not None
        and counts.get(declaration["name"]) == 1
    }
    index_intervals = getattr(index_expr, "_glsl_control_flow_intervals", None) or {}
    with codegen.glsl_lexical_source_type_scope(source_types):
        index_bounds = codegen.glsl_checked_private_pointer_interval(
            index_expr, index_intervals, constants, require_arithmetic_types=True
        )
        index_form = _affine(codegen, index_expr, atom_key, index_intervals, constants)
        if not _within(index_bounds, (INT_MIN, INT_MAX)) or index_form is None:
            return False
        for key, value in index_form[0].items():
            target[key] = target.get(key, 0) + value
        form = (target, residual[0] + index_form[1], residual[1] + index_form[2])
        guard_facts = []
        for condition, expected, region in guards:
            if _has_side_effects(codegen, condition):
                continue
            names = {
                node.name
                for node in _walk(condition)
                if isinstance(node, IdentifierNode)
            }
            if names & _mutated_roots(codegen, region):
                continue
            guard_facts += _comparison_facts(
                codegen, condition, expected, atom_key, constants
            )
    guard_facts = guard_facts[:MAX_GUARD_FACTS]
    index_atoms = {
        references.get(id(node))
        for node in _walk(index_expr)
        if isinstance(node, IdentifierNode)
    }

    def interval_of(key):
        declaration = declarations.get(key[1])
        if declaration is None:
            return None
        name = declaration["name"]
        # Name-keyed facts describe the declaration visible at the subscript.
        if counts.get(name) != 1 and key[1] not in index_atoms:
            return None
        return index_intervals.get(name)

    upper = (
        {key: -value for key, value in form[0].items()},
        INT_MAX - form[2],
        INT_MAX - form[1],
    )
    return _proves_nonnegative(form, guard_facts, interval_of) and _proves_nonnegative(
        upper, guard_facts, interval_of
    )
