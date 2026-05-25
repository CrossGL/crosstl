"""Helpers for lowering CrossGL match statements to C-style shader languages."""

from ..ast import LiteralPatternNode, WildcardPatternNode


def is_switch_lowerable_match(node):
    """Return true when a match can be emitted as a plain switch statement."""
    return all(
        getattr(arm, "guard", None) is None
        and isinstance(
            getattr(arm, "pattern", None), (LiteralPatternNode, WildcardPatternNode)
        )
        for arm in getattr(node, "arms", []) or []
    )


def generate_switch_match(generator, node, indent):
    """Lower an unguarded literal/wildcard match to a switch statement."""
    indent_str = "    " * indent
    expression = generator.generate_expression(getattr(node, "expression", ""))

    code = f"{indent_str}switch ({expression}) {{\n"
    for arm in getattr(node, "arms", []) or []:
        pattern = getattr(arm, "pattern", None)
        if isinstance(pattern, WildcardPatternNode):
            label = "default"
        else:
            label = f"case {generator.generate_expression(pattern.literal)}"
        body = getattr(arm, "body", [])
        code += generator.generate_switch_case(label, body, indent + 1, auto_break=True)

    code += f"{indent_str}}}\n"
    return code


def generate_ordered_conditional_match(generator, node, indent, target_name):
    """Lower a guarded literal/wildcard match to an ordered if/else chain."""
    indent_str = "    " * indent
    expression = generator.generate_expression(getattr(node, "expression", ""))

    code = ""
    emitted_arm = False
    for arm in getattr(node, "arms", []) or []:
        condition = match_arm_condition(generator, expression, arm, target_name)
        body = getattr(arm, "body", [])

        if condition is None:
            if emitted_arm:
                code += f"{indent_str}else {{\n"
            else:
                code += f"{indent_str}{{\n"
            code += generator.generate_scoped_statement_body(body, indent + 1)
            code += f"{indent_str}}}\n"
            emitted_arm = True
            continue

        prefix = "if" if not emitted_arm else "else if"
        code += f"{indent_str}{prefix} ({condition}) {{\n"
        code += generator.generate_scoped_statement_body(body, indent + 1)
        code += f"{indent_str}}}\n"
        emitted_arm = True

    return code


def match_arm_condition(generator, expression, arm, target_name):
    pattern = getattr(arm, "pattern", None)
    guard = getattr(arm, "guard", None)

    if isinstance(pattern, LiteralPatternNode):
        literal = generator.generate_expression(pattern.literal)
        pattern_condition = f"({expression} == {literal})"
    elif isinstance(pattern, WildcardPatternNode):
        pattern_condition = None
    else:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; only literal "
            "and wildcard patterns are supported"
        )

    guard_condition = (
        generator.generate_expression(guard) if guard is not None else None
    )
    if pattern_condition and guard_condition:
        return f"({pattern_condition} && ({guard_condition}))"
    return pattern_condition or guard_condition
