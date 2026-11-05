"""Lower DirectX typed-buffer and group-shared atomics that appear in control-flow conditions.

HLSL ``Interlocked*`` intrinsics return their original value through an ``out`` argument, so an atomic that
CrossGL writes as an expression has to run as a statement first. Expression contexts (initialisers, returns, call
arguments, ternaries) already hoist that statement; this module does the same for the conditions of ``if``,
``else if``, ``while``, ``for``, ``do``/``while`` and ``switch`` while keeping each condition's evaluation order
and count: an ``else if`` condition runs only after the earlier ones failed, and a loop condition runs once per
iteration, including the iterations that a ``continue`` starts.
"""

from __future__ import annotations

__all__ = [
    "condition_has_hlsl_atomic",
    "generate_hlsl_atomic_condition_do_while",
    "generate_hlsl_atomic_condition_if",
    "generate_hlsl_atomic_condition_switch",
    "generate_hlsl_atomic_condition_while",
    "hlsl_atomic_condition_loop_guard",
]

_PAD = "    "


def condition_has_hlsl_atomic(gen, condition) -> bool:
    """Whether ``condition`` holds an atomic that DirectX can only express as a statement."""
    if condition is None or isinstance(condition, str):
        return False
    return bool(gen.hlsl_expression_contains_typed_buffer_atomic(condition))


def _render(gen, condition, expected_type, indent):
    prelude, rendered = gen.render_hlsl_typed_buffer_atomic_value_expression(
        condition, expected_type, indent
    )
    return prelude, gen.hlsl_strip_wrapping_parentheses(rendered)


def hlsl_atomic_condition_loop_guard(gen, condition, indent) -> str:
    """Statements that evaluate a loop ``condition`` and leave the loop when it is false."""
    prelude, rendered = _render(gen, condition, "bool", indent)
    return (
        prelude
        + f"{_PAD * indent}if (!({rendered})) {{\n{_PAD * (indent + 1)}break;\n{_PAD * indent}}}\n"
    )


def generate_hlsl_atomic_condition_if(
    gen, node, condition, then_body, else_body, indent
):
    """Emit an ``if``/``else if`` chain whose conditions hold atomics, or ``None`` when none do.

    A condition with an atomic opens a nested ``else`` block so that its statements run only when every earlier
    condition was false.
    """
    else_if_conditions = list(getattr(node, "else_if_conditions", None) or [])
    else_if_bodies = list(getattr(node, "else_if_bodies", None) or [])
    conditions = [condition, *else_if_conditions]
    if not any(condition_has_hlsl_atomic(gen, item) for item in conditions):
        return None
    bodies = [then_body, *else_if_bodies]

    code = ""
    level = indent
    closers = []
    for position, (branch_condition, body) in enumerate(zip(conditions, bodies)):
        if condition_has_hlsl_atomic(gen, branch_condition):
            if position:
                code += " else {\n"
                closers.append(level)
                level += 1
            prelude, rendered = _render(gen, branch_condition, "bool", level)
            code += prelude + f"{_PAD * level}if ({rendered}) {{\n"
        else:
            rendered = gen.hlsl_condition_expression(branch_condition)
            opener = f"{_PAD * level}if (" if position == 0 else " else if ("
            code += f"{opener}{rendered}) {{\n"
        code += gen.generate_scoped_statement_body(body, level + 1)
        code += f"{_PAD * level}}}"
    if else_body:
        code += " else {\n"
        code += gen.generate_scoped_statement_body(else_body, level + 1)
        code += f"{_PAD * level}}}"
    code += "\n"
    for closer in reversed(closers):
        code += f"{_PAD * closer}}}\n"
    return code


def generate_hlsl_atomic_condition_while(gen, node, indent):
    """``while (c) body`` as ``while (true) { if (!c) break; body }``, or ``None`` without an atomic."""
    condition = getattr(node, "condition", None)
    if not condition_has_hlsl_atomic(gen, condition):
        return None
    pad = _PAD * indent
    code = f"{pad}while (true) {{\n"
    code += hlsl_atomic_condition_loop_guard(gen, condition, indent + 1)
    code += gen.generate_scoped_statement_body(getattr(node, "body", []), indent + 1)
    return code + f"{pad}}}\n"


def generate_hlsl_atomic_condition_do_while(gen, node, indent):
    """``do body while (c)`` with a first-iteration flag, or ``None`` without an atomic.

    The condition is tested at the top of every iteration after the first, so a ``continue`` in the body still
    reaches it, as it does in the original loop.
    """
    condition = getattr(node, "condition", None)
    if not condition_has_hlsl_atomic(gen, condition):
        return None
    pad = _PAD * indent
    inner = _PAD * (indent + 1)
    first = gen.next_hlsl_temp_variable("do_while_first")
    code = f"{pad}bool {first} = true;\n"
    code += f"{pad}while (true) {{\n"
    code += f"{inner}if (!{first}) {{\n"
    code += hlsl_atomic_condition_loop_guard(gen, condition, indent + 2)
    code += f"{inner}}}\n"
    code += f"{inner}{first} = false;\n"
    code += gen.generate_scoped_statement_body(getattr(node, "body", []), indent + 1)
    return code + f"{pad}}}\n"


def generate_hlsl_atomic_condition_switch(gen, node, indent):
    """Evaluate an atomic ``switch`` selector before the ``switch``, or return ``None`` without one."""
    selector = getattr(node, "expression", None)
    if not condition_has_hlsl_atomic(gen, selector):
        return None
    selector_type = gen.expression_result_type(selector)
    prelude, rendered = _render(gen, selector, selector_type, indent)
    return prelude, rendered
