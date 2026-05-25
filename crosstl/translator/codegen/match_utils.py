"""Helpers for lowering CrossGL match statements to C-style shader languages."""

from ..ast import (
    ConstructorPatternNode,
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
    return generate_ordered_conditional_match_arms(
        generator,
        list(getattr(node, "arms", []) or []),
        getattr(node, "expression", ""),
        indent,
        target_name,
    )


def generate_ordered_conditional_match_arms(
    generator,
    arms,
    expression_node,
    indent,
    target_name,
):
    indent_str = "    " * indent
    expression = generator.generate_expression(expression_node)
    expression_type = expression_result_type(generator, expression_node)

    code = ""
    emitted_arm = False
    for index, arm in enumerate(arms):
        pattern_condition, bindings, binding_types = match_arm_pattern_lowering(
            generator,
            expression,
            expression_type,
            arm,
            target_name,
        )
        guard = getattr(arm, "guard", None)
        guard_condition = (
            None
            if guard is None
            else generate_match_guard_condition(generator, guard, binding_types)
        )
        body = getattr(arm, "body", [])

        if bindings and guard is not None:
            code += generate_guarded_bound_match_arm(
                generator,
                arms[index + 1 :],
                expression_node,
                pattern_condition,
                bindings,
                binding_types,
                guard,
                body,
                indent,
                emitted_arm,
                target_name,
            )
            break

        condition = combine_match_conditions(pattern_condition, guard_condition)
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


def match_arm_pattern_lowering(
    generator,
    expression,
    expression_type,
    arm,
    target_name,
):
    pattern = getattr(arm, "pattern", None)
    return lower_match_pattern(
        generator,
        pattern,
        expression,
        expression_type,
        target_name,
    )


def combine_match_conditions(pattern_condition, guard_condition):
    if pattern_condition and guard_condition:
        return f"({pattern_condition} && ({guard_condition}))"
    return pattern_condition or guard_condition


def generate_match_guard_condition(generator, guard, binding_types):
    saved_types = getattr(generator, "local_variable_types", None)
    if saved_types is not None:
        saved_types = dict(saved_types)
        generator.local_variable_types.update(binding_types)
    try:
        return generator.generate_expression(guard)
    finally:
        if saved_types is not None:
            generator.local_variable_types = saved_types


def generate_guarded_bound_match_arm(
    generator,
    rest_arms,
    expression_node,
    pattern_condition,
    bindings,
    binding_types,
    guard,
    body,
    indent,
    emitted_arm,
    target_name,
):
    indent_str = "    " * indent

    if pattern_condition is None:
        code = f"{indent_str}else {{\n" if emitted_arm else f"{indent_str}{{\n"
        code += generate_guarded_bound_match_body(
            generator,
            rest_arms,
            expression_node,
            bindings,
            binding_types,
            guard,
            body,
            indent + 1,
            target_name,
        )
        code += f"{indent_str}}}\n"
        return code

    prefix = "if" if not emitted_arm else "else if"
    code = f"{indent_str}{prefix} ({pattern_condition}) {{\n"
    code += generate_guarded_bound_match_body(
        generator,
        rest_arms,
        expression_node,
        bindings,
        binding_types,
        guard,
        body,
        indent + 1,
        target_name,
    )
    code += f"{indent_str}}}\n"
    if rest_arms:
        code += f"{indent_str}else {{\n"
        code += generate_ordered_conditional_match_arms(
            generator,
            rest_arms,
            expression_node,
            indent + 1,
            target_name,
        )
        code += f"{indent_str}}}\n"
    return code


def generate_guarded_bound_match_body(
    generator,
    rest_arms,
    expression_node,
    bindings,
    binding_types,
    guard,
    body,
    indent,
    target_name,
):
    indent_str = "    " * indent
    guard_condition = generate_match_guard_condition(generator, guard, binding_types)

    code = "".join(f"{indent_str}{binding}\n" for binding in bindings)
    code += f"{indent_str}if ({guard_condition}) {{\n"
    code += generate_body_with_binding_types(generator, binding_types, body, indent + 1)
    code += f"{indent_str}}}"
    if rest_arms:
        code += " else {\n"
        code += generate_ordered_conditional_match_arms(
            generator,
            rest_arms,
            expression_node,
            indent + 1,
            target_name,
        )
        code += f"{indent_str}}}"
    code += "\n"
    return code


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

    if isinstance(pattern, ConstructorPatternNode):
        return lower_constructor_pattern(pattern, target_name)

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
    if "::" in name:
        enum_value = enum_variant_reference(
            generator,
            name,
            expression_type,
            target_name,
        )
        return f"({expression} == {enum_value})", [], {}

    if name == "..":
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; rest patterns "
            "are not supported"
        )

    binding_type = require_pattern_binding_type(
        generator,
        expression_type,
        name,
        target_name,
    )
    declaration = binding_declaration(generator, binding_type, name, expression)
    return None, [declaration], {name: type_name_string(generator, binding_type)}


def enum_variant_reference(generator, path, expression_type, target_name):
    subject_type = base_type_name(type_name_string(generator, expression_type))
    enum_name, _variant_name = split_enum_path(path)
    if subject_type and enum_name != subject_type:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum path "
            f"{path} cannot match expression type {subject_type}"
        )

    constants = getattr(generator, "enum_variant_constants", {})
    if path in constants:
        return constants[path]

    raise ValueError(
        f"Unsupported match arm for {target_name} codegen; enum path "
        "patterns are not supported"
    )


def split_enum_path(path):
    enum_name, variant_name = str(path).rsplit("::", 1)
    return enum_name, variant_name


def lower_constructor_pattern(pattern, target_name):
    pattern_type = getattr(pattern, "type_name", "")
    if "::" in pattern_type:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum constructor "
            "patterns are not supported"
        )

    raise ValueError(
        f"Unsupported match arm for {target_name} codegen; constructor patterns "
        "are not supported"
    )


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
    if isinstance(
        pattern, (LiteralPatternNode, WildcardPatternNode, StructPatternNode)
    ):
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
    code = "".join(f"{indent_str}{binding}\n" for binding in bindings)
    code += generate_body_with_binding_types(generator, binding_types, body, indent)
    return code


def generate_body_with_binding_types(generator, binding_types, body, indent):
    saved_types = getattr(generator, "local_variable_types", None)
    if saved_types is not None:
        saved_types = dict(saved_types)
        generator.local_variable_types.update(binding_types)
    try:
        return generator.generate_scoped_statement_body(body, indent)
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
