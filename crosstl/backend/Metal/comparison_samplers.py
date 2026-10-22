"""MSL samplers that only feed depth comparisons, typed as CrossGL ``comparison_sampler``.

MSL has one ``sampler`` type; whether a sampler compares is sampler state (``compare_func`` on a ``constexpr
sampler``, ``MTLSamplerDescriptor.compareFunction`` for an argument), and ``sample_compare`` and ``gather_compare``
use it (MSL 3.2 sections 2.10 and 6.12.4). CrossGL types a comparison sampler as ``comparison_sampler``, so HLSL
declares a ``SamplerComparisonState``, WGSL a ``sampler_comparison`` and Vulkan GLSL a ``samplerShadow`` for it.

:func:`collect_comparison_sampler_ids` returns the declarations (globals and function parameters) of the samplers
that are used, directly or through the arguments they are passed to, and only as the sampler of a comparison method.
A sampler that a call binds to a parameter shares that parameter's type, so both are decided together. A sampler
that is also used to sample, gather, or anything else stays a plain ``sampler``.
"""

from __future__ import annotations

from crosstl.backend.common_ast import (
    ArrayAccessNode,
    FunctionCallNode,
    FunctionNode,
    MethodCallNode,
    VariableNode,
)
from crosstl.backend.usage_slots import UsageSlots

__all__ = ["collect_comparison_sampler_ids"]

_COMPARE_METHODS = frozenset({"sample_compare", "gather_compare"})
_SAMPLER_METHODS = frozenset({"sample", "gather"}) | _COMPARE_METHODS


def _is_sampler_type(vtype):
    if not isinstance(vtype, str):
        return False
    text = vtype.replace(" ", "")
    for prefix in ("constexpr", "const", "metal::"):
        text = text.removeprefix(prefix)
    return text == "sampler" or (
        text.startswith(("array<sampler,", "array<metal::sampler,"))
        and text.endswith(">")
    )


def _root_variable(expression):
    while isinstance(expression, ArrayAccessNode):
        expression = expression.array
    if isinstance(expression, VariableNode) and isinstance(expression.name, str):
        return expression
    return None


def _call_name(node):
    name = getattr(node, "name", None)
    if isinstance(name, VariableNode):
        name = name.name
    return name if isinstance(name, str) else None


def _walk(node, seen):
    """Yield every AST object under ``node`` once, parents before children."""

    stack = [node]
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if isinstance(current, (list, tuple, set)):
            stack.extend(reversed(list(current)))
            continue
        if isinstance(current, dict):
            stack.extend(reversed(list(current.values())))
            continue
        if id(current) in seen or not hasattr(current, "__dict__"):
            continue
        seen.add(id(current))
        yield current
        stack.extend(reversed(list(vars(current).values())))


def _global_samplers(shader):
    found = {}
    for attribute in ("global_variables", "global_vars", "uniforms"):
        for variable in getattr(shader, attribute, None) or []:
            if isinstance(variable, VariableNode) and _is_sampler_type(variable.vtype):
                found.setdefault(variable.name, variable)
    return found


def collect_comparison_sampler_ids(shader) -> set[int]:
    """``id`` of every sampler declaration in ``shader`` that is only ever used to compare."""

    functions = [
        function
        for function in getattr(shader, "functions", None) or []
        if isinstance(function, FunctionNode)
    ]
    globals_by_name = _global_samplers(shader)
    params = {
        (id(function), index): param
        for function in functions
        for index, param in enumerate(function.params or [])
        if _is_sampler_type(getattr(param, "vtype", None))
    }
    if not globals_by_name and not params:
        return set()

    slots = UsageSlots()
    for function in functions:
        local_slots = {
            param.name: (id(function), index)
            for index, param in enumerate(function.params or [])
            if (id(function), index) in params
        }
        shadowed = {
            node.name
            for node in _walk(function.body, set())
            if isinstance(node, VariableNode) and getattr(node, "vtype", None)
        }

        def slot_of(expression):
            variable = _root_variable(expression)
            if variable is None:
                return None, None
            if variable.name in local_slots:
                return local_slots[variable.name], variable
            if variable.name in globals_by_name and variable.name not in shadowed:
                return ("global", variable.name), variable
            return None, None

        accounted = set()
        references = []
        for node in _walk(function.body, set()):
            if isinstance(node, MethodCallNode) and node.method in _SAMPLER_METHODS:
                slot, variable = slot_of((node.args or [None])[0])
                if slot is not None:
                    accounted.add(id(variable))
                    slots.use(
                        slot, "compare" if node.method in _COMPARE_METHODS else "plain"
                    )
            elif isinstance(node, FunctionCallNode):
                name = _call_name(node)
                args = node.args or []
                for callee in functions:
                    if callee.name != name or len(callee.params or []) != len(args):
                        continue
                    for index, arg in enumerate(args):
                        slot, variable = slot_of(arg)
                        if slot is not None and (id(callee), index) in params:
                            accounted.add(id(variable))
                            slots.join(slot, (id(callee), index))
            elif isinstance(node, VariableNode) and not getattr(node, "vtype", None):
                references.append(node)
        # Any other reference (a copy, a member of a returned struct, a call to a built-in) keeps the sampler plain.
        for reference in references:
            if id(reference) in accounted:
                continue
            slot, _ = slot_of(reference)
            if slot is not None:
                slots.use(slot, "plain")

    comparison_ids = {
        id(variable)
        for name, variable in globals_by_name.items()
        if slots.only(("global", name), "compare")
    }
    comparison_ids.update(
        id(param) for key, param in params.items() if slots.only(key, "compare")
    )
    return comparison_ids
