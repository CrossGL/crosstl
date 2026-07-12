"""Helpers for lowering CrossGL match statements to C-style shader languages."""

from ..ast import (
    ConstructorPatternNode,
    IdentifierPatternNode,
    LiteralPatternNode,
    StructPatternNode,
    WildcardPatternNode,
)
from .array_utils import format_c_style_array_declaration
from .enum_utils import enum_variant_fields_for_type


def is_switch_lowerable_match(node):
    """Return true when a match can be emitted as a plain switch statement."""
    arms = list(getattr(node, "arms", []) or [])
    wildcard_index = None
    for index, arm in enumerate(arms):
        if getattr(arm, "guard", None) is not None:
            return False

        pattern = getattr(arm, "pattern", None)
        if isinstance(pattern, WildcardPatternNode):
            if wildcard_index is not None or index != len(arms) - 1:
                return False
            wildcard_index = index
            continue

        if not isinstance(pattern, LiteralPatternNode):
            return False

    return True


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


def generate_match_expression_assignment(
    generator,
    node,
    target_variable,
    target_type,
    indent,
    target_name,
):
    """Lower a value-position match into assignments to an existing local."""
    return generate_match_expression_assignment_arms(
        generator,
        list(getattr(node, "arms", []) or []),
        getattr(node, "expression", ""),
        target_variable,
        target_type,
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
    if match_subject_requires_temp(expression_node, expression_type):
        temp_name = next_match_temp_variable(generator)
        declaration = format_c_style_array_declaration(
            generator.map_type(expression_type),
            temp_name,
        )
        code += f"{indent_str}{declaration} = {expression};\n"
        expression = temp_name

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
        code += (
            f"{indent_str}{prefix} "
            f"{format_match_condition(generator, condition)} {{\n"
        )
        code += generate_bound_match_body(
            generator, bindings, binding_types, body, indent + 1
        )
        code += f"{indent_str}}}\n"
        emitted_arm = True

    return code


def generate_match_expression_assignment_arms(
    generator,
    arms,
    expression_node,
    target_variable,
    target_type,
    indent,
    target_name,
):
    indent_str = "    " * indent
    expression = generator.generate_expression(expression_node)
    expression_type = expression_result_type(generator, expression_node)

    code = ""
    if match_subject_requires_temp(expression_node, expression_type):
        temp_name = next_match_temp_variable(generator)
        declaration = format_c_style_array_declaration(
            generator.map_type(expression_type),
            temp_name,
        )
        code += f"{indent_str}{declaration} = {expression};\n"
        expression = temp_name

    emitted_arm = False
    handled_unconditional = False
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

        if bindings and guard is not None:
            code += generate_guarded_bound_match_assignment_arm(
                generator,
                arms[index + 1 :],
                expression_node,
                pattern_condition,
                bindings,
                binding_types,
                guard,
                arm,
                target_variable,
                target_type,
                indent,
                emitted_arm,
                target_name,
            )
            handled_unconditional = True
            break

        condition = combine_match_conditions(pattern_condition, guard_condition)
        if condition is None:
            if emitted_arm:
                code += f"{indent_str}else {{\n"
            else:
                code += f"{indent_str}{{\n"
            code += generate_match_expression_assignment_body(
                generator,
                bindings,
                binding_types,
                arm,
                target_variable,
                target_type,
                indent + 1,
                target_name,
            )
            code += f"{indent_str}}}\n"
            handled_unconditional = True
            break

        prefix = "if" if not emitted_arm else "else if"
        code += (
            f"{indent_str}{prefix} "
            f"{format_match_condition(generator, condition)} {{\n"
        )
        code += generate_match_expression_assignment_body(
            generator,
            bindings,
            binding_types,
            arm,
            target_variable,
            target_type,
            indent + 1,
            target_name,
        )
        code += f"{indent_str}}}\n"
        emitted_arm = True

    if not handled_unconditional and (
        match_expression_requires_fallback_assignment(generator)
        or not match_arms_exhaust_subject(generator, arms, expression_type)
    ):
        fallback = match_expression_unmatched_assignment(
            generator,
            target_variable,
            target_type,
            indent + (1 if emitted_arm else 0),
            target_name,
        )
        if fallback:
            if emitted_arm:
                code += f"{indent_str}else {{\n"
                code += fallback
                code += f"{indent_str}}}\n"
            else:
                code += fallback

    return code


def match_arms_exhaust_subject(generator, arms, expression_type):
    subject_type = base_type_name(type_name_string(generator, expression_type))
    if not subject_type:
        return False

    if subject_type == "bool":
        return match_arms_cover_bool_literals(arms)

    expected_variants = enum_variants_for_subject(generator, subject_type)
    if not expected_variants:
        return False

    covered_variants = set()
    for arm in arms:
        if getattr(arm, "guard", None) is not None:
            continue
        pattern = getattr(arm, "pattern", None)
        if isinstance(pattern, WildcardPatternNode):
            return True
        variant = unconditionally_covered_enum_variant(pattern)
        if variant is not None:
            enum_name, variant_name = split_enum_path(variant)
            if enum_name == subject_type:
                covered_variants.add(variant_name)

    return expected_variants <= covered_variants


def enum_variants_for_subject(generator, subject_type):
    constants = getattr(generator, "enum_variant_constants", {})
    variants = set()
    for path in constants:
        if "::" not in path:
            continue
        enum_name, variant_name = split_enum_path(path)
        if enum_name == subject_type:
            variants.add(variant_name)
    return variants


def unconditionally_covered_enum_variant(pattern):
    if isinstance(pattern, IdentifierPatternNode):
        name = getattr(pattern, "name", "")
        return name if "::" in name else None

    if isinstance(pattern, ConstructorPatternNode):
        pattern_type = getattr(pattern, "type_name", "")
        if "::" not in pattern_type:
            return None
        if all(
            match_field_pattern_is_unconditional(argument)
            for argument in getattr(pattern, "arguments", []) or []
        ):
            return pattern_type
        return None

    if isinstance(pattern, StructPatternNode):
        pattern_type = getattr(pattern, "type_name", "")
        if "::" not in pattern_type:
            return None
        if all(
            match_field_pattern_is_unconditional(field_pattern)
            for field_pattern in getattr(pattern, "field_patterns", {}).values()
        ):
            return pattern_type
        return None

    return None


def match_field_pattern_is_unconditional(pattern):
    return isinstance(pattern, (IdentifierPatternNode, WildcardPatternNode))


def match_arms_cover_bool_literals(arms):
    covered = set()
    for arm in arms:
        if getattr(arm, "guard", None) is not None:
            continue
        pattern = getattr(arm, "pattern", None)
        if isinstance(pattern, WildcardPatternNode):
            return True
        if not isinstance(pattern, LiteralPatternNode):
            continue
        literal = getattr(pattern, "literal", None)
        value = getattr(literal, "value", literal)
        if isinstance(value, bool):
            covered.add(value)
    return covered == {False, True}


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


def format_match_condition(generator, condition):
    formatter = getattr(generator, "format_match_condition", None)
    if callable(formatter):
        return formatter(condition)
    return f"({condition})"


def apply_match_binding_types(generator, binding_types):
    saved_type_maps = {}
    for attribute in ("local_variable_types", "local_variable_source_types"):
        type_map = getattr(generator, attribute, None)
        if type_map is None:
            continue
        saved_type_maps[attribute] = dict(type_map)
        type_map.update(binding_types)
    return saved_type_maps


def restore_match_binding_types(generator, saved_type_maps):
    for attribute, type_map in saved_type_maps.items():
        setattr(generator, attribute, type_map)


def generate_match_guard_condition(generator, guard, binding_types):
    saved_type_maps = apply_match_binding_types(generator, binding_types)
    try:
        return generator.generate_expression(guard)
    finally:
        restore_match_binding_types(generator, saved_type_maps)


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
    code = (
        f"{indent_str}{prefix} "
        f"{format_match_condition(generator, pattern_condition)} {{\n"
    )
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
    code += f"{indent_str}if {format_match_condition(generator, guard_condition)} {{\n"
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


def generate_guarded_bound_match_assignment_arm(
    generator,
    rest_arms,
    expression_node,
    pattern_condition,
    bindings,
    binding_types,
    guard,
    arm,
    target_variable,
    target_type,
    indent,
    emitted_arm,
    target_name,
):
    indent_str = "    " * indent

    if pattern_condition is None:
        code = f"{indent_str}else {{\n" if emitted_arm else f"{indent_str}{{\n"
        code += generate_guarded_bound_match_assignment_body(
            generator,
            rest_arms,
            expression_node,
            bindings,
            binding_types,
            guard,
            arm,
            target_variable,
            target_type,
            indent + 1,
            target_name,
        )
        code += f"{indent_str}}}\n"
        return code

    prefix = "if" if not emitted_arm else "else if"
    code = (
        f"{indent_str}{prefix} "
        f"{format_match_condition(generator, pattern_condition)} {{\n"
    )
    code += generate_guarded_bound_match_assignment_body(
        generator,
        rest_arms,
        expression_node,
        bindings,
        binding_types,
        guard,
        arm,
        target_variable,
        target_type,
        indent + 1,
        target_name,
    )
    code += f"{indent_str}}}\n"
    if rest_arms:
        code += f"{indent_str}else {{\n"
        code += generate_match_expression_assignment_arms(
            generator,
            rest_arms,
            expression_node,
            target_variable,
            target_type,
            indent + 1,
            target_name,
        )
        code += f"{indent_str}}}\n"
    return code


def generate_guarded_bound_match_assignment_body(
    generator,
    rest_arms,
    expression_node,
    bindings,
    binding_types,
    guard,
    arm,
    target_variable,
    target_type,
    indent,
    target_name,
):
    indent_str = "    " * indent
    guard_condition = generate_match_guard_condition(generator, guard, binding_types)

    code = "".join(f"{indent_str}{binding}\n" for binding in bindings)
    code += f"{indent_str}if {format_match_condition(generator, guard_condition)} {{\n"
    code += generate_match_expression_assignment_body(
        generator,
        [],
        binding_types,
        arm,
        target_variable,
        target_type,
        indent + 1,
        target_name,
    )
    code += f"{indent_str}}}"
    if rest_arms:
        code += " else {\n"
        code += generate_match_expression_assignment_arms(
            generator,
            rest_arms,
            expression_node,
            target_variable,
            target_type,
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
        return lower_constructor_pattern(
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
    if "::" in name:
        condition = enum_variant_condition(
            generator,
            name,
            expression,
            expression_type,
            target_name,
        )
        return condition, [], {}

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


def enum_variant_condition(generator, path, expression, expression_type, target_name):
    enum_value = enum_variant_reference(generator, path, expression_type, target_name)
    enum_name, _variant_name = split_enum_path(path)
    tag_expression = enum_subject_tag_expression(
        generator, expression, expression_type, enum_name
    )
    return f"({tag_expression} == {enum_value})"


def enum_subject_tag_expression(generator, expression, expression_type, enum_name=None):
    subject_type = base_type_name(type_name_string(generator, expression_type))
    struct_enum_types = getattr(generator, "enum_struct_type_names", set())
    if subject_type in struct_enum_types or enum_name in struct_enum_types:
        return f"{expression}.variant"
    return expression


def split_enum_path(path):
    enum_name, variant_name = str(path).rsplit("::", 1)
    return enum_name, variant_name


def lower_constructor_pattern(
    generator,
    pattern,
    expression,
    expression_type,
    target_name,
):
    pattern_type = getattr(pattern, "type_name", "")
    if "::" in pattern_type:
        return lower_enum_constructor_pattern(
            generator,
            pattern,
            expression,
            expression_type,
            target_name,
        )

    raise ValueError(
        f"Unsupported match arm for {target_name} codegen; constructor patterns "
        "are not supported"
    )


def lower_struct_pattern(generator, pattern, expression, expression_type, target_name):
    pattern_type = getattr(pattern, "type_name", "")
    if "::" in pattern_type:
        return lower_enum_struct_pattern(
            generator,
            pattern,
            expression,
            expression_type,
            target_name,
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


def lower_enum_struct_pattern(
    generator,
    pattern,
    expression,
    expression_type,
    target_name,
):
    pattern_type = getattr(pattern, "type_name", "")
    enum_name, variant_name = split_enum_path(pattern_type)
    subject_type = base_type_name(type_name_string(generator, expression_type))
    if subject_type and enum_name != subject_type:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum struct "
            f"pattern {pattern_type} cannot match expression type {subject_type}"
        )

    variant_fields = enum_variant_fields_for_type(
        generator,
        enum_name,
        variant_name,
        expression_type,
    )
    if variant_fields is None:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum struct "
            "patterns are not supported"
        )

    condition = enum_variant_condition(
        generator,
        pattern_type,
        expression,
        expression_type,
        target_name,
    )
    bindings = []
    binding_types = {}
    for field_name, field_pattern in getattr(pattern, "field_patterns", {}).items():
        field_type = variant_fields.get(field_name)
        if field_type is None:
            raise ValueError(
                f"Unsupported match arm for {target_name} codegen; enum struct "
                f"pattern {pattern_type} has no field '{field_name}'"
            )
        field_expr = f"{expression}.{field_name}"
        field_condition, field_bindings, field_binding_types = (
            lower_struct_field_pattern(
                generator,
                field_pattern,
                field_expr,
                field_type,
                target_name,
            )
        )
        if field_condition:
            condition = f"({condition}) && ({field_condition})"
        bindings.extend(field_bindings)
        binding_types.update(field_binding_types)

    return condition, bindings, binding_types


def lower_enum_constructor_pattern(
    generator,
    pattern,
    expression,
    expression_type,
    target_name,
):
    pattern_type = getattr(pattern, "type_name", "")
    enum_name, variant_name = split_enum_path(pattern_type)
    subject_type = base_type_name(type_name_string(generator, expression_type))
    if subject_type and enum_name != subject_type:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum constructor "
            f"pattern {pattern_type} cannot match expression type {subject_type}"
        )

    variant_fields = enum_variant_fields_for_type(
        generator,
        enum_name,
        variant_name,
        expression_type,
    )
    if variant_fields is None:
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum constructor "
            "patterns are not supported"
        )

    field_items = list(variant_fields.items())
    arguments = getattr(pattern, "arguments", []) or []
    if len(arguments) != len(field_items):
        raise ValueError(
            f"Unsupported match arm for {target_name} codegen; enum constructor "
            f"pattern {pattern_type} expects {len(field_items)} arguments"
        )

    condition = enum_variant_condition(
        generator,
        pattern_type,
        expression,
        expression_type,
        target_name,
    )
    bindings = []
    binding_types = {}
    for field_pattern, (field_name, field_type) in zip(arguments, field_items):
        field_expr = f"{expression}.{field_name}"
        field_condition, field_bindings, field_binding_types = (
            lower_struct_field_pattern(
                generator,
                field_pattern,
                field_expr,
                field_type,
                target_name,
            )
        )
        if field_condition:
            condition = f"({condition}) && ({field_condition})"
        bindings.extend(field_bindings)
        binding_types.update(field_binding_types)

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
        pattern,
        (
            LiteralPatternNode,
            WildcardPatternNode,
            StructPatternNode,
            ConstructorPatternNode,
        ),
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


def generate_match_expression_assignment_body(
    generator,
    bindings,
    binding_types,
    arm,
    target_variable,
    target_type,
    indent,
    target_name,
):
    indent_str = "    " * indent
    prefix, value = match_arm_value_parts(getattr(arm, "body", None))

    code = "".join(f"{indent_str}{binding}\n" for binding in bindings)
    saved_type_maps = apply_match_binding_types(generator, binding_types)
    try:
        if prefix:
            code += generator.generate_scoped_statement_body(prefix, indent)

        if value is None:
            code += match_expression_missing_value_assignment(
                generator,
                target_variable,
                target_type,
                indent,
                target_name,
            )
        elif getattr(value, "__class__", None).__name__ == "MatchNode":
            code += generate_match_expression_assignment(
                generator,
                value,
                target_variable,
                target_type,
                indent,
                target_name,
            )
        else:
            rhs = generator.generate_expression_with_expected(value, target_type)
            code += f"{indent_str}{target_variable} = {rhs};\n"
    finally:
        restore_match_binding_types(generator, saved_type_maps)
    return code


def generate_body_with_binding_types(generator, binding_types, body, indent):
    saved_type_maps = apply_match_binding_types(generator, binding_types)
    try:
        return generator.generate_scoped_statement_body(body, indent)
    finally:
        restore_match_binding_types(generator, saved_type_maps)


def match_subject_requires_temp(expression_node, expression_type):
    if expression_type is None:
        return False
    class_name = expression_node.__class__.__name__
    return "FunctionCall" in class_name


def next_match_temp_variable(generator):
    if hasattr(generator, "next_hlsl_temp_variable"):
        return generator.next_hlsl_temp_variable("match_subject")
    if hasattr(generator, "next_metal_temp_variable"):
        return generator.next_metal_temp_variable("match_subject")

    index = getattr(generator, "match_temp_variable_index", 0)
    generator.match_temp_variable_index = index + 1
    return f"cgl_match_subject_{index}"


def binding_declaration(generator, binding_type, name, expression):
    binding_name = name
    binding_name_hook = getattr(generator, "match_binding_name", None)
    if callable(binding_name_hook):
        binding_name = binding_name_hook(name)
    declaration = format_c_style_array_declaration(
        generator.map_type(binding_type), binding_name
    )
    formatter = getattr(generator, "format_match_binding_declaration", None)
    if callable(formatter):
        declaration = formatter(declaration, binding_name)
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


def infer_match_expression_result_type(generator, node):
    expression_node = getattr(node, "expression", None)
    expression_type = expression_result_type(generator, expression_node)
    result_type = None

    for arm in getattr(node, "arms", []) or []:
        _condition, _bindings, binding_types = match_arm_pattern_lowering(
            generator,
            "cgl_match_subject",
            expression_type,
            arm,
            "match expression",
        )
        value = match_arm_value_expression(getattr(arm, "body", None))
        if value is None:
            continue

        saved_type_maps = apply_match_binding_types(generator, binding_types)
        try:
            arm_type = expression_result_type(generator, value)
        finally:
            restore_match_binding_types(generator, saved_type_maps)

        if arm_type is None:
            continue
        if result_type is None:
            result_type = arm_type
            continue
        result_type = compatible_match_result_type(generator, result_type, arm_type)

    return result_type


def compatible_match_result_type(generator, left_type, right_type):
    left_name = type_name_string(generator, left_type)
    right_name = type_name_string(generator, right_type)
    if left_name == right_name:
        return left_name
    if getattr(generator, "is_vector_value_type", lambda _type: False)(left_name):
        return left_name
    if getattr(generator, "is_vector_value_type", lambda _type: False)(right_name):
        return right_name
    if "float" in {left_name, right_name}:
        return "float"
    if "double" in {left_name, right_name}:
        return "double"
    left_mapped = generator.map_type(left_name) if left_name else None
    right_mapped = generator.map_type(right_name) if right_name else None
    if left_mapped == right_mapped:
        return left_name
    return left_name or right_name


def match_arm_value_expression(body):
    _prefix, value = match_arm_value_parts(body)
    return value


def match_arm_value_parts(body):
    statements = statement_list(body)
    if not statements:
        return [], None

    tail = statements[-1]
    if hasattr(tail, "expression") and getattr(tail, "is_tail_expression", False):
        return statements[:-1], tail.expression
    if getattr(tail, "__class__", None).__name__ == "MatchNode":
        return statements[:-1], tail
    return statements, None


def match_expression_missing_value_assignment(
    generator,
    target_variable,
    target_type,
    indent,
    target_name,
):
    hook = getattr(generator, "generate_match_expression_diagnostic_assignment", None)
    if hook is not None:
        return hook(
            target_variable,
            target_type,
            "arm does not produce a value",
            indent,
            target_name,
        )
    raise ValueError(
        f"Unsupported match expression for {target_name} codegen; arms must "
        "end in a value expression"
    )


def match_expression_unmatched_assignment(
    generator,
    target_variable,
    target_type,
    indent,
    target_name,
):
    hook = getattr(generator, "generate_match_expression_diagnostic_assignment", None)
    if hook is None:
        return ""
    return hook(
        target_variable,
        target_type,
        "no wildcard arm handles remaining cases",
        indent,
        target_name,
    )


def match_expression_requires_fallback_assignment(generator):
    hook = getattr(generator, "match_expression_requires_fallback_assignment", None)
    if callable(hook):
        return bool(hook())
    return False


def statement_list(body):
    if hasattr(body, "statements"):
        return list(getattr(body, "statements", []) or [])
    if isinstance(body, list):
        return body
    if body is None:
        return []
    return [body]


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
    if type_name == "None":
        return ""
    return type_name.split("<", 1)[0].strip()
