"""Helpers for lowering CrossGL match statements to C-style shader languages."""

from ..ast import (
    IdentifierPatternNode,
    LiteralPatternNode,
    StructPatternNode,
    WildcardPatternNode,
)
from .array_utils import format_c_style_array_declaration


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
    """Lower a match to an ordered if/else chain."""
    indent_str = "    " * indent
    expression = generator.generate_expression(getattr(node, "expression", ""))
    expression_type = expression_result_type(generator, getattr(node, "expression", ""))

    code = ""
    emitted_arm = False
    for arm in getattr(node, "arms", []) or []:
        condition, bindings, binding_types = match_arm_condition_and_bindings(
            generator,
            expression,
            expression_type,
            arm,
            target_name,
        )
        body = getattr(arm, "body", [])

        if bindings and getattr(arm, "guard", None) is not None:
            raise ValueError(
                f"Unsupported match arm for {target_name} codegen; guarded "
                "binding patterns are not supported"
            )

        if condition is None:
            if emitted_arm:
                code += f"{indent_str}else {{\n"
            else:
                code += f"{indent_str}{{\n"
            code += generate_bound_match_body(
                generator, bindings, binding_types, body, indent + 1
            )
            code += f"{indent_str}}}\n"
            break

        prefix = "if" if not emitted_arm else "else if"
        code += f"{indent_str}{prefix} ({condition}) {{\n"
        code += generate_bound_match_body(
            generator, bindings, binding_types, body, indent + 1
        )
        code += f"{indent_str}}}\n"
        emitted_arm = True

    return code


def match_arm_condition_and_bindings(
    generator,
    expression,
    expression_type,
    arm,
    target_name,
):
    pattern = getattr(arm, "pattern", None)
    guard = getattr(arm, "guard", None)

    pattern_condition, bindings, binding_types = lower_match_pattern(
        generator,
        pattern,
        expression,
        expression_type,
        target_name,
    )

    guard_condition = generator.generate_expression(guard) if guard is not None else None
    if pattern_condition and guard_condition:
        return f"({pattern_condition} && ({guard_condition}))", bindings, binding_types
    return pattern_condition or guard_condition, bindings, binding_types


def lower_match_pattern(generator, pattern, expression, expression_type, target_name):
    if isinstance(pattern, LiteralPatternNode):
        literal = generator.generate_expression(pattern.literal)
        return f"({expression} == {literal})", [], {}

    if isinstance(pattern, WildcardPatternNode):
        return None, [], {}

    if isinstance(pattern, IdentifierPatternNode):
        return lower_identifier_pattern(
            generator,
            pattern,
            expression,
            expression_type,
            target_name,
        )

    if isinstance(pattern, StructPatternNode):
        return lower_struct_pattern(
            generator,
            pattern,
            expression,
            expression_type,
            target_name,
        )

    raise ValueError(
        f"Unsupported match arm for {target_name} codegen; only literal, "
        "wildcard, identifier binding, and plain struct patterns are supported"
    )


def lower_identifier_pattern(
    generator,
    pattern,
    expression,
    expression_type,
    target_name,
):
    name = getattr(pattern, "name", "")
    if name == ".." or "::" in name:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum path "
            "patterns are not supported"
        )

    binding_type = require_pattern_binding_type(
        generator,
        expression_type,
        name,
        target_name,
    )
    declaration = binding_declaration(generator, binding_type, name, expression)
    return None, [declaration], {name: type_name_string(generator, binding_type)}


def lower_struct_pattern(generator, pattern, expression, expression_type, target_name):
    pattern_type = getattr(pattern, "type_name", "")
    if "::" in pattern_type:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum struct "
            "patterns are not supported"
        )

    subject_type = type_name_string(generator, expression_type) or pattern_type
    if subject_type and base_type_name(subject_type) != base_type_name(pattern_type):
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; struct pattern "
            f"{pattern_type} cannot match expression type {subject_type}"
        )

    conditions = []
    bindings = []
    binding_types = {}
    for field_name, field_pattern in getattr(pattern, "field_patterns", {}).items():
        field_type = struct_field_type(generator, pattern_type, field_name)
        field_expr = f"{expression}.{field_name}"
        condition, field_bindings, field_binding_types = lower_struct_field_pattern(
            generator,
            field_pattern,
            field_expr,
            field_type,
            target_name,
        )
        if condition:
            conditions.append(condition)
        bindings.extend(field_bindings)
        binding_types.update(field_binding_types)

    condition = " && ".join(f"({part})" for part in conditions) or None
    return condition, bindings, binding_types


def lower_struct_field_pattern(
    generator,
    pattern,
    expression,
    expression_type,
    target_name,
):
    if isinstance(pattern, IdentifierPatternNode):
        return lower_identifier_pattern(
            generator,
            pattern,
            expression,
            expression_type,
            target_name,
        )
    if isinstance(pattern, (LiteralPatternNode, WildcardPatternNode, StructPatternNode)):
        return lower_match_pattern(
            generator,
            pattern,
            expression,
            expression_type,
            target_name,
        )
    raise ValueError(
        f"Unsupported match arm for {target_name} codegen; struct fields may "
        "only use literal, wildcard, identifier binding, or nested struct patterns"
    )


def generate_bound_match_body(generator, bindings, binding_types, body, indent):
    indent_str = "    " * indent
    saved_types = getattr(generator, "local_variable_types", None)
    if saved_types is not None:
        saved_types = dict(saved_types)
        generator.local_variable_types.update(binding_types)
    try:
        code = "".join(f"{indent_str}{binding}\n" for binding in bindings)
        code += generator.generate_scoped_statement_body(body, indent)
        return code
    finally:
        if saved_types is not None:
            generator.local_variable_types = saved_types


def binding_declaration(generator, binding_type, name, expression):
    declaration = format_c_style_array_declaration(
        generator.map_type(binding_type), name
    )
    return f"{declaration} = {expression};"


def require_pattern_binding_type(generator, binding_type, name, target_name):
    binding_type = type_name_string(generator, binding_type)
    if not binding_type:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; cannot infer "
            f"type for binding pattern '{name}'"
        )
    return binding_type


def expression_result_type(generator, expression):
    result_type = getattr(generator, "expression_result_type", None)
    if result_type is None:
        return None
    return result_type(expression)


def struct_field_type(generator, struct_name, field_name):
    member_types = getattr(generator, "struct_member_types", {})
    return member_types.get(base_type_name(struct_name), {}).get(field_name)


def type_name_string(generator, value):
    type_name = getattr(generator, "type_name_string", None)
    if type_name is None:
        return None if value is None else str(value)
    return type_name(value)


def base_type_name(type_name):
    type_name = str(type_name or "")
    return type_name.split("<", 1)[0].strip()
