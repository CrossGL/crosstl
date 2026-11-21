"""Lowering of CrossGL match statements and expressions to Mojo conditionals, with pattern exhaustiveness checks."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AtomicOpNode,
    BinaryOpNode,
    BufferOpNode,
    BuiltinVariableNode,
    CastNode,
    ConstructorNode,
    ConstructorPatternNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    LiteralPatternNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    StructPatternNode,
    SwizzleNode,
    TernaryOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WildcardPatternNode,
)


class MojoMatchMixin:
    """Match members of ``MojoCodeGen``."""

    def generate_match(self, node, indent):
        indent_str = "    " * indent
        subject_expr = getattr(node, "expression", "")
        subject_type = self.expression_result_type(subject_expr)
        expression = self.generate_expression(subject_expr)
        arms = getattr(node, "arms", []) or []
        self.validate_match_arms(arms, subject_type)
        code = ""
        subject_temp_state = None
        if self.match_subject_needs_temp(subject_expr) and subject_type is not None:
            subject_name = self.next_match_subject_temp_name()
            code += (
                f"{indent_str}var {subject_name}: {self.map_type(subject_type)} = "
                f"{expression}\n"
            )
            expression = subject_name
            subject_temp_state = self.register_match_subject_temp_metadata(
                subject_name, subject_type, subject_expr
            )
        emitted_condition = False

        try:
            for arm_index, arm in enumerate(arms):
                pattern = getattr(arm, "pattern", None)
                guard = getattr(arm, "guard", None)
                body = getattr(arm, "body", [])
                replacements = self.match_arm_identifier_replacements(
                    pattern, expression, guard, subject_type
                )
                if self.is_unconditional_final_match_arm(
                    arms, arm_index, pattern, guard, subject_type
                ):
                    if emitted_condition:
                        code += f"{indent_str}else:\n"
                        code += self.generate_match_arm_body(
                            body, indent + 1, replacements
                        )
                    else:
                        code += self.generate_match_arm_body(body, indent, replacements)
                    continue

                keyword = "if" if not emitted_condition else "elif"
                condition = self.match_arm_condition(
                    pattern, expression, guard, replacements, subject_type
                )
                code += f"{indent_str}{keyword} {condition}:\n"
                code += self.generate_match_arm_body(body, indent + 1, replacements)
                emitted_condition = True
        finally:
            if subject_temp_state is not None:
                self.restore_match_subject_temp_metadata(
                    subject_name, subject_temp_state
                )

        if not code and not emitted_condition:
            code += f"{indent_str}pass\n"

        return code

    def next_match_expression_temp_name(self):
        name = f"__cgl_match_value_{self.match_expression_counter}"
        self.match_expression_counter += 1
        return name

    def next_match_subject_temp_name(self):
        name = f"__cgl_match_subject_{self.match_subject_counter}"
        self.match_subject_counter += 1
        return name

    def register_match_subject_temp_metadata(self, name, subject_type, subject_expr):
        path_prefix = f"{name}."
        state = {
            "had_type": name in self.variable_types,
            "type": self.variable_types.get(name),
            "had_aliases": name in self.resource_access_qualifier_aliases,
            "aliases": list(self.resource_access_qualifier_aliases.get(name, [])),
            "path_aliases": {
                path: list(candidates)
                for path, candidates in (
                    self.resource_access_path_qualifier_aliases.items()
                )
                if path == name or path.startswith(path_prefix)
            },
        }
        self.register_variable_type(name, subject_type)
        self.register_resource_access_alias_metadata(name, subject_type, subject_expr)
        return state

    def restore_match_subject_temp_metadata(self, name, state):
        if state["had_type"]:
            self.variable_types[name] = state["type"]
        else:
            self.variable_types.pop(name, None)

        self.clear_resource_access_path_aliases(name)
        self.resource_access_path_qualifier_aliases.update(state["path_aliases"])
        if state["had_aliases"]:
            self.resource_access_qualifier_aliases[name] = state["aliases"]
        else:
            self.resource_access_qualifier_aliases.pop(name, None)

    def match_subject_needs_temp(self, subject_expr):
        if isinstance(subject_expr, (str, VariableNode, IdentifierNode)):
            return False
        if isinstance(subject_expr, MemberAccessNode):
            return self.match_subject_needs_temp(subject_expr.object)
        return True

    def match_arm_error(self, reason=None):
        detail = reason or "only unguarded literal and wildcard patterns are supported"
        return ValueError(f"Unsupported match arm for Mojo codegen; {detail}")

    def match_expression_error(self, reason):
        return ValueError(f"Unsupported match expression for Mojo codegen; {reason}")

    def match_arm_value_expression(self, arm):
        body = getattr(arm, "body", [])
        if isinstance(body, ExpressionStatementNode):
            if getattr(body, "is_tail_expression", False):
                return body.expression, []
            raise self.match_expression_error(
                "every arm must end with a value expression"
            )
        if self.is_match_expression_value_node(body):
            return body, []

        statements = self.statement_list(body)
        if (
            statements
            and isinstance(statements[-1], ExpressionStatementNode)
            and getattr(statements[-1], "is_tail_expression", False)
        ):
            return statements[-1].expression, statements[:-1]
        if statements and self.is_match_expression_value_node(statements[-1]):
            return statements[-1], statements[:-1]

        raise self.match_expression_error("every arm must end with a value expression")

    def is_match_expression_value_node(self, node):
        return isinstance(
            node,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                AtomicOpNode,
                BinaryOpNode,
                BufferOpNode,
                BuiltinVariableNode,
                CastNode,
                ConstructorNode,
                FunctionCallNode,
                MatchNode,
                MemberAccessNode,
                MeshOpNode,
                RayQueryOpNode,
                RayTracingOpNode,
                SwizzleNode,
                TernaryOpNode,
                TextureNode,
                TextureOpNode,
                UnaryOpNode,
                VariableNode,
                WaveOpNode,
            ),
        ) or (
            isinstance(node, (str, int, float, bool))
            or (hasattr(node, "__class__") and "Identifier" in str(node.__class__))
            or (hasattr(node, "__class__") and "Literal" in str(node.__class__))
        )

    def generate_match_value_expression(self, node, target_type, context):
        arms = getattr(node, "arms", []) or []
        subject_expr = getattr(node, "expression", "")
        subject_type = self.expression_result_type(subject_expr)
        self.validate_match_expression_arms(arms, subject_type)
        match_type = self.type_name(target_type or self.expression_result_type(node))
        if match_type is None:
            raise self.match_expression_error("target type is required")
        if not self.expression_prelude_stack:
            raise self.match_expression_error("expression prelude context is required")

        prelude_context = self.expression_prelude_stack[-1]
        indent = prelude_context["indent"]
        indent_str = "    " * indent
        branch_indent = indent + 1
        temp_name = self.next_match_expression_temp_name()
        expression = self.generate_expression(subject_expr)
        lines = []
        subject_temp_state = None
        if self.match_subject_needs_temp(subject_expr) and subject_type is not None:
            subject_name = self.next_match_subject_temp_name()
            lines.append(
                f"{indent_str}var {subject_name}: {self.map_type(subject_type)} = "
                f"{expression}\n"
            )
            expression = subject_name
            subject_temp_state = self.register_match_subject_temp_metadata(
                subject_name, subject_type, subject_expr
            )
        lines.append(
            f"{indent_str}var {temp_name}: {self.map_type(match_type)} = "
            f"{self.zero_value_for_type(match_type)}\n"
        )

        try:
            emitted_condition = False
            for arm_index, arm in enumerate(arms, start=1):
                pattern = getattr(arm, "pattern", None)
                guard = getattr(arm, "guard", None)
                replacements = self.match_arm_identifier_replacements(
                    pattern, expression, guard, subject_type
                )
                if self.is_unconditional_final_match_arm(
                    arms, arm_index - 1, pattern, guard, subject_type
                ):
                    if emitted_condition:
                        lines.append(f"{indent_str}else:\n")
                        assignment_indent = branch_indent
                    else:
                        assignment_indent = indent
                    lines.append(
                        self.generate_match_value_arm_assignment(
                            arm,
                            match_type,
                            f"{context} match arm {arm_index}",
                            temp_name,
                            assignment_indent,
                            replacements,
                        )
                    )
                    continue

                keyword = "if" if not emitted_condition else "elif"
                condition = self.match_arm_condition(
                    pattern, expression, guard, replacements, subject_type
                )
                lines.append(f"{indent_str}{keyword} {condition}:\n")
                lines.append(
                    self.generate_match_value_arm_assignment(
                        arm,
                        match_type,
                        f"{context} match arm {arm_index}",
                        temp_name,
                        branch_indent,
                        replacements,
                    )
                )
                emitted_condition = True
        finally:
            if subject_temp_state is not None:
                self.restore_match_subject_temp_metadata(
                    subject_name, subject_temp_state
                )

        prelude_context["lines"].extend(lines)
        return temp_name

    def generate_match_value_arm_assignment(
        self, arm, target_type, context, temp_name, indent, replacements=None
    ):
        def render_assignment():
            value_expr, prelude_statements = self.match_arm_value_expression(arm)
            code = ""
            for stmt in prelude_statements:
                code += self.generate_statement(stmt, indent)

            self.validate_expression_target_shape(value_expr, target_type, context)
            value = self.generate_expression(value_expr, target_type, context)
            return f"{code}{'    ' * indent}{temp_name} = {value}\n"

        return self.generate_with_identifier_replacements(
            replacements, render_assignment
        )

    def validate_match_arms(self, arms, subject_type=None):
        rejection_reason = self.match_arm_rejection_reason(arms, subject_type)
        if rejection_reason is not None:
            raise self.match_arm_error(rejection_reason)

    def validate_match_expression_arms(self, arms, subject_type=None):
        self.validate_match_arms(arms, subject_type)
        if not arms:
            raise self.match_expression_error("at least one arm is required")
        final_pattern = getattr(arms[-1], "pattern", None)
        final_guard = getattr(arms[-1], "guard", None)
        if not (
            isinstance(final_pattern, WildcardPatternNode)
            or self.is_irrefutable_match_pattern(
                final_pattern, final_guard, subject_type
            )
            or self.match_arms_cover_enum_variants(arms, subject_type)
        ):
            raise self.match_expression_error(
                "match expressions must include a final wildcard arm"
            )
        for arm in arms:
            self.match_arm_value_expression(arm)

    def match_arm_identifier_replacements(
        self, pattern, expression, guard=None, subject_type=None
    ):
        replacements = {}
        if (
            guard is not None
            and isinstance(pattern, IdentifierPatternNode)
            and not self.is_enum_identifier_pattern(pattern, subject_type)
        ):
            replacements[pattern.name] = expression
            return replacements

        self.collect_match_pattern_bindings(
            pattern,
            expression,
            replacements,
            top_level=True,
            subject_type=subject_type,
        )
        return replacements

    def generate_match_arm_body(self, body, indent, replacements=None):
        return self.generate_with_identifier_replacements(
            replacements, lambda: self.generate_switch_case_body(body, indent)
        )

    def match_arm_condition(
        self,
        pattern,
        expression,
        guard=None,
        replacements=None,
        subject_type=None,
    ):
        replacements = replacements or {}
        condition = None

        if isinstance(pattern, LiteralPatternNode):
            condition = f"{expression} == {self.generate_expression(pattern.literal)}"
        elif isinstance(pattern, IdentifierPatternNode):
            if self.is_enum_identifier_pattern(pattern, subject_type):
                condition = self.enum_identifier_pattern_condition(
                    pattern, expression, subject_type
                )
        elif isinstance(pattern, ConstructorPatternNode):
            condition = self.constructor_pattern_condition(
                pattern, expression, replacements, subject_type
            )
        elif isinstance(pattern, StructPatternNode):
            condition = self.struct_pattern_condition(
                pattern, expression, replacements, subject_type
            )
        elif isinstance(pattern, WildcardPatternNode):
            condition = "True"

        if guard is not None:
            guard_expression = self.generate_expression_with_identifier_replacements(
                guard, replacements
            )
            if condition is None:
                return guard_expression
            if condition == "True":
                return guard_expression
            return f"({condition}) and ({guard_expression})"

        if condition is None:
            raise self.match_arm_error()
        return condition

    def is_enum_identifier_pattern(self, pattern, subject_type=None):
        name = getattr(pattern, "name", None)
        if not isinstance(name, str):
            return False
        return (
            self.resolve_match_enum_variant_path(name, subject_type)
            in self.enum_variant_aliases
        )

    def match_subject_uses_payload_struct(self, enum_type, subject_type=None):
        return (
            enum_type in self.struct_types
            or self.generic_enum_specialization_for_type(
                subject_type, expected_base=enum_type
            )
            or self.generic_enum_pattern_specialization_for_type(
                subject_type, expected_base=enum_type
            )
        )

    def enum_identifier_pattern_condition(self, pattern, expression, subject_type=None):
        name = self.resolve_match_enum_variant_path(pattern.name, subject_type)
        alias = self.enum_variant_aliases[name]
        enum_type = self.enum_variant_result_types.get(name)
        if self.match_subject_uses_payload_struct(enum_type, subject_type):
            return f"{expression}.variant == {alias}"
        return f"{expression} == {alias}"

    def constructor_pattern_condition(
        self, pattern, expression, replacements, subject_type=None
    ):
        variant_path = self.resolve_match_enum_variant_path(
            getattr(pattern, "type_name", None), subject_type
        )
        variant_fields = self.enum_pattern_variant_fields(variant_path, subject_type)
        if variant_fields is None:
            raise self.match_arm_error("constructor patterns cannot be lowered")

        arguments = getattr(pattern, "arguments", []) or []
        if len(arguments) != len(variant_fields):
            raise self.match_arm_error(
                f"constructor pattern {variant_path} expects "
                f"{len(variant_fields)} fields, got {len(arguments)}"
            )

        conditions = [
            self.enum_variant_tag_condition(variant_path, expression, subject_type)
        ]
        for argument, (field_name, field_type) in zip(arguments, variant_fields):
            field_expression = self.match_pattern_field_expression(
                expression, field_name
            )
            conditions.extend(
                self.match_pattern_field_conditions(
                    argument,
                    field_expression,
                    replacements,
                    top_level=False,
                    subject_type=field_type,
                )
            )
        return (
            " and ".join(condition for condition in conditions if condition) or "True"
        )

    def struct_pattern_condition(
        self, pattern, expression, replacements, subject_type=None
    ):
        pattern_type = self.resolve_match_enum_variant_path(
            getattr(pattern, "type_name", None), subject_type
        )
        variant_fields = self.enum_pattern_variant_fields(pattern_type, subject_type)
        if variant_fields is not None:
            field_types = dict(variant_fields)
            conditions = [
                self.enum_variant_tag_condition(pattern_type, expression, subject_type)
            ]
        elif (
            self.struct_pattern_type_name(pattern_type, subject_type)
            in self.struct_types
        ):
            field_types = self.struct_types[
                self.struct_pattern_type_name(pattern_type, subject_type)
            ]
            conditions = []
        else:
            raise self.match_arm_error(
                "struct destructuring patterns cannot be lowered"
            )

        field_patterns = getattr(pattern, "field_patterns", {}) or {}
        for field_name, field_pattern in field_patterns.items():
            if field_name not in field_types:
                raise self.match_arm_error(
                    f"struct pattern field {field_name} cannot be lowered"
                )
            field_expression = self.match_pattern_field_expression(
                expression, field_name
            )
            conditions.extend(
                self.match_pattern_field_conditions(
                    field_pattern,
                    field_expression,
                    replacements,
                    top_level=False,
                    subject_type=field_types[field_name],
                )
            )
        return (
            " and ".join(condition for condition in conditions if condition) or "True"
        )

    def enum_pattern_variant_fields(self, variant_path, subject_type=None):
        variant_path = self.resolve_match_enum_variant_path(variant_path, subject_type)
        variant_fields = self.enum_variant_constructor_fields.get(variant_path)
        if variant_fields is not None:
            return variant_fields
        return self.generic_enum_variant_fields_for_type(variant_path, subject_type)

    def enum_variant_tag_condition(self, variant_path, expression, subject_type=None):
        variant_path = self.resolve_match_enum_variant_path(variant_path, subject_type)
        alias = self.enum_variant_aliases[variant_path]
        enum_type = self.enum_variant_result_types.get(variant_path)
        if self.match_subject_uses_payload_struct(enum_type, subject_type):
            return f"{expression}.variant == {alias}"
        return f"{expression} == {alias}"

    def resolve_match_enum_variant_path(self, variant_path, subject_type=None):
        if not isinstance(variant_path, str):
            return variant_path

        normalized = variant_path.replace(".", "::")
        if normalized in self.enum_variant_aliases:
            return normalized
        if "::" in normalized:
            return normalized

        enum_name = self.match_subject_enum_type_name(subject_type)
        if enum_name is None:
            return normalized
        if normalized not in self.enum_variant_names.get(enum_name, []):
            return normalized
        return f"{enum_name}::{normalized}"

    def match_subject_enum_type_name(self, subject_type):
        specialization = self.generic_enum_specialization_for_type(subject_type)
        if specialization is None:
            specialization = self.generic_enum_pattern_specialization_for_type(
                subject_type
            )
        if specialization is not None:
            return specialization["base_name"]

        type_name = self.type_name(subject_type)
        if type_name in self.enum_variant_names:
            return type_name
        return None

    def struct_pattern_type_name(self, pattern_type, subject_type=None):
        if not isinstance(pattern_type, str):
            return pattern_type
        generic_enum_type = self.generic_enum_mapped_type(subject_type)
        if generic_enum_type is not None and pattern_type == self.type_name(
            subject_type
        ):
            return generic_enum_type
        return self.type_name(pattern_type)

    def match_pattern_field_expression(self, expression, field_name):
        if re.match(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$", expression):
            return f"{expression}.{field_name}"
        return f"({expression}).{field_name}"

    def collect_match_pattern_bindings(
        self,
        pattern,
        expression,
        replacements,
        top_level=False,
        subject_type=None,
    ):
        self.match_pattern_field_conditions(
            pattern,
            expression,
            replacements,
            top_level=top_level,
            collect_conditions=False,
            subject_type=subject_type,
        )

    def match_pattern_field_conditions(
        self,
        pattern,
        expression,
        replacements,
        top_level=False,
        collect_conditions=True,
        subject_type=None,
    ):
        conditions = []
        if isinstance(pattern, WildcardPatternNode):
            return conditions
        if isinstance(pattern, LiteralPatternNode):
            binding_name = self.literal_pattern_binding_name(pattern)
            if binding_name is not None:
                if not top_level:
                    replacements[binding_name] = expression
                return conditions
            if collect_conditions:
                conditions.append(
                    f"{expression} == {self.generate_expression(pattern.literal)}"
                )
            return conditions
        if isinstance(pattern, IdentifierPatternNode):
            if self.is_enum_identifier_pattern(pattern, subject_type):
                if collect_conditions:
                    conditions.append(
                        self.enum_identifier_pattern_condition(
                            pattern, expression, subject_type
                        )
                    )
                return conditions
            if not top_level:
                replacements[pattern.name] = expression
            return conditions
        if isinstance(pattern, ConstructorPatternNode):
            variant_path = self.resolve_match_enum_variant_path(
                getattr(pattern, "type_name", None), subject_type
            )
            variant_fields = self.enum_pattern_variant_fields(
                variant_path, subject_type
            )
            if variant_fields is None:
                return conditions
            if collect_conditions:
                conditions.append(
                    self.enum_variant_tag_condition(
                        variant_path, expression, subject_type
                    )
                )
            for argument, (field_name, field_type) in zip(
                getattr(pattern, "arguments", []) or [], variant_fields
            ):
                conditions.extend(
                    self.match_pattern_field_conditions(
                        argument,
                        self.match_pattern_field_expression(expression, field_name),
                        replacements,
                        top_level=False,
                        collect_conditions=collect_conditions,
                        subject_type=field_type,
                    )
                )
            return conditions
        if isinstance(pattern, StructPatternNode):
            pattern_type = self.resolve_match_enum_variant_path(
                getattr(pattern, "type_name", None), subject_type
            )
            variant_fields = self.enum_pattern_variant_fields(
                pattern_type, subject_type
            )
            if variant_fields is not None:
                if collect_conditions:
                    conditions.append(
                        self.enum_variant_tag_condition(
                            pattern_type, expression, subject_type
                        )
                    )
            field_types = None
            if variant_fields is not None:
                field_types = dict(variant_fields)
            elif (
                self.struct_pattern_type_name(pattern_type, subject_type)
                in self.struct_types
            ):
                field_types = self.struct_types[
                    self.struct_pattern_type_name(pattern_type, subject_type)
                ]
            for field_name, field_pattern in (
                getattr(pattern, "field_patterns", {}) or {}
            ).items():
                conditions.extend(
                    self.match_pattern_field_conditions(
                        field_pattern,
                        self.match_pattern_field_expression(expression, field_name),
                        replacements,
                        top_level=False,
                        collect_conditions=collect_conditions,
                        subject_type=(
                            field_types.get(field_name)
                            if field_types is not None
                            else subject_type
                        ),
                    )
                )
        return conditions

    def literal_pattern_binding_name(self, pattern):
        literal = getattr(pattern, "literal", None)
        value = getattr(literal, "value", None)
        literal_type = getattr(getattr(literal, "literal_type", None), "name", None)
        if (
            literal_type == "unknown"
            and isinstance(value, str)
            and re.fullmatch(r"[A-Za-z_]\w*", value)
        ):
            return value
        return None

    def match_arm_rejection_reason(self, arms, subject_type=None):
        wildcard_index = None
        irrefutable_index = None
        for index, arm in enumerate(arms):
            guard = getattr(arm, "guard", None)

            pattern = getattr(arm, "pattern", None)
            if self.is_range_style_match_pattern(pattern):
                return "range-style patterns cannot be lowered"

            if isinstance(pattern, WildcardPatternNode):
                if guard is not None:
                    return "guarded wildcard arms cannot be lowered"
                if wildcard_index is not None:
                    return "multiple wildcard arms cannot be lowered"
                wildcard_index = index
                continue

            if isinstance(pattern, LiteralPatternNode):
                continue

            if isinstance(pattern, IdentifierPatternNode):
                if self.is_enum_identifier_pattern(pattern, subject_type):
                    continue
                if guard is not None:
                    continue
                return "identifier binding patterns cannot be lowered"

            if isinstance(pattern, ConstructorPatternNode):
                reason = self.constructor_pattern_rejection_reason(
                    pattern, subject_type
                )
                if reason is not None:
                    return reason
                continue

            if isinstance(pattern, StructPatternNode):
                reason = self.struct_pattern_rejection_reason(pattern, subject_type)
                if reason is not None:
                    return reason
                if self.is_irrefutable_match_pattern(pattern, guard, subject_type):
                    if irrefutable_index is not None:
                        return "multiple irrefutable patterns cannot be lowered"
                    irrefutable_index = index
                continue

            return "only unguarded literal patterns and a final wildcard can be lowered"

        if wildcard_index is not None and wildcard_index != len(arms) - 1:
            return "wildcard arm must be final"
        if irrefutable_index is not None and irrefutable_index != len(arms) - 1:
            return "irrefutable pattern must be final"
        return None

    def is_irrefutable_match_pattern(self, pattern, guard=None, subject_type=None):
        if guard is not None:
            return False
        if isinstance(pattern, WildcardPatternNode):
            return True
        if isinstance(pattern, StructPatternNode):
            pattern_type = getattr(pattern, "type_name", None)
            if self.enum_pattern_variant_fields(pattern_type, subject_type) is not None:
                return False
            return (
                self.struct_pattern_type_name(pattern_type, subject_type)
                in self.struct_types
            )
        return False

    def match_arms_cover_enum_variants(self, arms, subject_type=None):
        enum_type = None
        patterns_by_variant = {}
        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            pattern = getattr(arm, "pattern", None)
            variant_path = self.match_pattern_enum_variant_path(pattern, subject_type)
            if variant_path is None:
                continue
            arm_enum_type = self.enum_variant_result_types.get(variant_path)
            if arm_enum_type is None:
                continue
            if enum_type is None:
                enum_type = arm_enum_type
            elif enum_type != arm_enum_type:
                return False
            variant_name = variant_path.split("::", 1)[1]
            patterns_by_variant.setdefault(variant_name, []).append(pattern)

        expected_variants = set(self.enum_variant_names.get(enum_type, []))
        covered_variants = {
            variant_name
            for variant_name, patterns in patterns_by_variant.items()
            if self.match_patterns_cover_enum_variant_group(patterns, subject_type)
        }
        return bool(expected_variants) and expected_variants.issubset(covered_variants)

    def match_patterns_cover_enum_variant_group(self, patterns, subject_type=None):
        if any(
            self.match_pattern_covers_entire_enum_variant(pattern, subject_type)
            for pattern in patterns
        ):
            return True

        variant_path = self.match_pattern_enum_variant_path(patterns[0], subject_type)
        variant_fields = self.enum_pattern_variant_fields(variant_path, subject_type)
        if not variant_fields:
            return False

        fields_by_name = dict(variant_fields)
        constrained_fields = {}
        for pattern in patterns:
            field_patterns = self.match_pattern_field_pattern_map(pattern, subject_type)
            if field_patterns is None:
                return False
            for field_name in fields_by_name:
                field_pattern = field_patterns.get(field_name)
                if field_pattern is None:
                    continue
                if self.is_unconstrained_match_pattern(
                    field_pattern, fields_by_name[field_name]
                ):
                    continue
                constrained_fields.setdefault(field_name, []).append(field_pattern)

        if len(constrained_fields) != 1:
            return False
        field_name, nested_patterns = next(iter(constrained_fields.items()))
        return self.match_patterns_cover_enum_type(
            nested_patterns,
            fields_by_name[field_name],
        )

    def match_pattern_field_pattern_map(self, pattern, subject_type=None):
        if isinstance(pattern, StructPatternNode):
            return dict(getattr(pattern, "field_patterns", {}) or {})
        if isinstance(pattern, ConstructorPatternNode):
            variant_path = getattr(pattern, "type_name", None)
            variant_fields = self.enum_pattern_variant_fields(
                variant_path, subject_type
            )
            if variant_fields is None:
                return None
            return {
                field_name: argument
                for argument, (field_name, _field_type) in zip(
                    getattr(pattern, "arguments", []) or [], variant_fields
                )
            }
        return None

    def match_patterns_cover_enum_type(self, patterns, subject_type):
        enum_type = None
        covered_variants = set()
        for pattern in patterns:
            if not self.match_pattern_covers_entire_enum_variant(pattern, subject_type):
                return False
            variant_path = self.match_pattern_enum_variant_path(pattern, subject_type)
            if variant_path is None:
                return False
            pattern_enum_type = self.enum_variant_result_types.get(variant_path)
            if pattern_enum_type is None:
                return False
            if enum_type is None:
                enum_type = pattern_enum_type
            elif enum_type != pattern_enum_type:
                return False
            covered_variants.add(variant_path.split("::", 1)[1])

        expected_variants = set(self.enum_variant_names.get(enum_type, []))
        return bool(expected_variants) and expected_variants.issubset(covered_variants)

    def is_unconditional_final_match_arm(
        self, arms, index, pattern, guard, subject_type=None
    ):
        if guard is not None or index != len(arms) - 1:
            return False
        if isinstance(pattern, WildcardPatternNode):
            return True
        if self.is_irrefutable_match_pattern(pattern, guard, subject_type):
            return True
        return self.match_arms_cover_enum_variants(
            arms, subject_type
        ) and self.match_pattern_covers_entire_enum_variant(pattern, subject_type)

    def match_pattern_covers_entire_enum_variant(self, pattern, subject_type=None):
        variant_path = self.match_pattern_enum_variant_path(pattern, subject_type)
        if variant_path is None:
            return False
        variant_fields = self.enum_pattern_variant_fields(variant_path, subject_type)
        if variant_fields is None:
            return False
        if isinstance(pattern, IdentifierPatternNode):
            return not variant_fields
        if isinstance(pattern, ConstructorPatternNode):
            arguments = getattr(pattern, "arguments", []) or []
            return len(arguments) == len(variant_fields) and all(
                self.is_unconstrained_match_pattern(argument, field_type)
                for argument, (_field_name, field_type) in zip(
                    arguments, variant_fields
                )
            )
        if isinstance(pattern, StructPatternNode):
            field_types = dict(variant_fields)
            return all(
                self.is_unconstrained_match_pattern(
                    field_pattern, field_types.get(field_name)
                )
                for field_name, field_pattern in (
                    getattr(pattern, "field_patterns", {}) or {}
                ).items()
            )
        return False

    def is_unconstrained_match_pattern(self, pattern, subject_type=None):
        if isinstance(pattern, WildcardPatternNode):
            return True
        if isinstance(pattern, LiteralPatternNode):
            return self.literal_pattern_binding_name(pattern) is not None
        if isinstance(pattern, IdentifierPatternNode):
            return not self.is_enum_identifier_pattern(pattern, subject_type)
        if isinstance(pattern, StructPatternNode):
            pattern_type = self.resolve_match_enum_variant_path(
                getattr(pattern, "type_name", None), subject_type
            )
            if self.enum_pattern_variant_fields(pattern_type, subject_type) is not None:
                return False
            struct_type = self.struct_pattern_type_name(pattern_type, subject_type)
            field_types = self.struct_types.get(struct_type)
            if field_types is None:
                return False
            for field_name, field_pattern in (
                getattr(pattern, "field_patterns", {}) or {}
            ).items():
                if field_name not in field_types:
                    return False
                if not self.is_unconstrained_match_pattern(
                    field_pattern, field_types[field_name]
                ):
                    return False
            return True
        return False

    def match_pattern_enum_variant_path(self, pattern, subject_type=None):
        if isinstance(
            pattern, IdentifierPatternNode
        ) and self.is_enum_identifier_pattern(pattern, subject_type):
            return self.resolve_match_enum_variant_path(pattern.name, subject_type)
        if isinstance(pattern, ConstructorPatternNode):
            variant_path = self.resolve_match_enum_variant_path(
                getattr(pattern, "type_name", None), subject_type
            )
            if (
                variant_path in self.enum_variant_result_types
                or self.generic_enum_variant_fields_for_type(variant_path, subject_type)
                is not None
            ):
                return variant_path.replace(".", "::")
        if isinstance(pattern, StructPatternNode):
            variant_path = self.resolve_match_enum_variant_path(
                getattr(pattern, "type_name", None), subject_type
            )
            if (
                variant_path in self.enum_variant_result_types
                or self.generic_enum_variant_fields_for_type(variant_path, subject_type)
                is not None
            ):
                return variant_path.replace(".", "::")
        return None

    def constructor_pattern_rejection_reason(self, pattern, subject_type=None):
        variant_path = self.resolve_match_enum_variant_path(
            getattr(pattern, "type_name", None), subject_type
        )
        variant_fields = self.enum_pattern_variant_fields(variant_path, subject_type)
        if variant_fields is None:
            return "constructor patterns cannot be lowered"
        arguments = getattr(pattern, "arguments", []) or []
        if len(arguments) != len(variant_fields):
            return (
                f"constructor pattern {variant_path} expects "
                f"{len(variant_fields)} fields, got {len(arguments)}"
            )
        for argument, (_field_name, field_type) in zip(arguments, variant_fields):
            reason = self.nested_pattern_rejection_reason([argument], field_type)
            if reason is not None:
                return reason
        return None

    def struct_pattern_rejection_reason(self, pattern, subject_type=None):
        pattern_type = self.resolve_match_enum_variant_path(
            getattr(pattern, "type_name", None), subject_type
        )
        variant_fields = self.enum_pattern_variant_fields(pattern_type, subject_type)
        if variant_fields is not None:
            field_types = dict(variant_fields)
        elif (
            self.struct_pattern_type_name(pattern_type, subject_type)
            in self.struct_types
        ):
            field_types = self.struct_types[
                self.struct_pattern_type_name(pattern_type, subject_type)
            ]
        else:
            return "struct destructuring patterns cannot be lowered"

        field_patterns = getattr(pattern, "field_patterns", {}) or {}
        for field_name, field_pattern in field_patterns.items():
            if field_name not in field_types:
                return f"struct pattern field {field_name} cannot be lowered"
            reason = self.nested_pattern_rejection_reason(
                [field_pattern], field_types[field_name]
            )
            if reason is not None:
                return reason
        return None

    def nested_pattern_rejection_reason(self, patterns, subject_type=None):
        for nested_pattern in patterns:
            if isinstance(nested_pattern, (WildcardPatternNode, LiteralPatternNode)):
                continue
            if isinstance(nested_pattern, IdentifierPatternNode):
                if self.is_enum_identifier_pattern(nested_pattern, subject_type):
                    continue
                continue
            if isinstance(nested_pattern, ConstructorPatternNode):
                reason = self.constructor_pattern_rejection_reason(
                    nested_pattern, subject_type
                )
                if reason is not None:
                    return reason
                continue
            if isinstance(nested_pattern, StructPatternNode):
                reason = self.struct_pattern_rejection_reason(
                    nested_pattern, subject_type
                )
                if reason is not None:
                    return reason
                continue
            if self.is_range_style_match_pattern(nested_pattern):
                return "range-style patterns cannot be lowered"
            return "nested pattern cannot be lowered"
        return None

    def is_range_style_match_pattern(self, pattern):
        if isinstance(pattern, RangeNode):
            return True
        return isinstance(pattern, IdentifierPatternNode) and pattern.name == ".."
