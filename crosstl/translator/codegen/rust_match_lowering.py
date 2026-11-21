"""CrossGL switch and match statements lowered to Rust match expressions, with exhaustiveness and binding checks."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ConstructorPatternNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    LiteralNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    RangeNode,
    ReturnNode,
    StructPatternNode,
    SwitchNode,
    TernaryOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
    WildcardPatternNode,
)
from ..refusals import RustRefusal


class RustMatchMixin:
    """Switch and match members of ``RustCodeGen``."""

    def generate_match(
        self,
        node,
        indent,
        expression_target_type=None,
        return_context=False,
    ):
        indent_str = "    " * indent
        arm_indent = "    " * (indent + 1)
        subject_expr = getattr(node, "expression", "")
        subject_type = self.expression_result_type(subject_expr)
        expression = None
        if return_context or self.match_statement_terminates_all_arms(node):
            expression = self.generate_direct_return_move_expression(subject_expr)
        if expression is None:
            expression = self.generate_expression(subject_expr)

        code = f"{indent_str}match {expression} {{\n"
        has_unconditional_wildcard = False
        for arm in getattr(node, "arms", []) or []:
            if not self.is_supported_match_arm(arm):
                raise RustRefusal("Unsupported match arm for Rust codegen")

            pattern = getattr(arm, "pattern", None)
            guard = getattr(arm, "guard", None)
            body = getattr(arm, "body", [])
            unused_bindings = self.unused_match_pattern_binding_names(
                pattern,
                guard,
                body,
            )
            mutable_bindings = self.mutable_match_pattern_binding_names(pattern, body)
            arm_pattern = self.generate_match_pattern(
                pattern,
                unused_bindings,
                mutable_bindings,
            )
            saved_variable_types = self.variable_types.copy()
            saved_local_variable_names = self.local_variable_names.copy()
            try:
                self.register_match_pattern_bindings(pattern, subject_type)

                has_unconditional_wildcard = (
                    has_unconditional_wildcard
                    or guard is None
                    and isinstance(pattern, WildcardPatternNode)
                )
                if guard is not None:
                    arm_pattern = f"{arm_pattern} if {self.generate_expression(guard)}"

                code += f"{arm_indent}{arm_pattern} => {{\n"
                code += self.generate_match_arm_body(
                    body,
                    indent + 2,
                    expression_target_type=expression_target_type,
                    return_context=return_context,
                )
                code += f"{arm_indent}}},\n"
            finally:
                self.variable_types = saved_variable_types
                self.local_variable_names = saved_local_variable_names

        if not has_unconditional_wildcard and not self.match_arms_are_exhaustive(
            node, subject_type
        ):
            code += f"{arm_indent}_ => unreachable!(),\n"

        code += f"{indent_str}}}\n"
        return code

    def match_statement_terminates_all_arms(self, node):
        arms = getattr(node, "arms", []) or []
        return bool(arms) and all(
            self.match_arm_body_terminates(getattr(arm, "body", None)) for arm in arms
        )

    def match_arm_body_terminates(self, body):
        return self.statement_body_terminates(body)

    def statement_body_terminates(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False

        tail = statements[-1]
        return self.statement_terminates(tail)

    def statement_terminates(self, tail):
        if isinstance(tail, ReturnNode):
            return True
        if isinstance(tail, BlockNode):
            return self.statement_body_terminates(tail)
        if isinstance(tail, LoopNode):
            return self.loop_statement_terminates(tail)
        if isinstance(tail, MatchNode):
            return self.match_statement_terminates_all_arms(tail)
        if isinstance(tail, IfNode):
            return self.if_statement_terminates_all_paths(tail)
        return False

    def loop_statement_terminates(self, node):
        body = getattr(node, "body", None)
        if self.loop_body_contains_control_flow(body):
            return False
        return self.statement_body_terminates(body)

    def loop_body_contains_control_flow(self, body):
        for statement in self.statement_list(body):
            if isinstance(statement, (BreakNode, ContinueNode)):
                return True
            if isinstance(statement, IfNode):
                if self.loop_body_contains_control_flow(
                    getattr(
                        statement, "then_branch", getattr(statement, "if_body", None)
                    )
                ):
                    return True
                if self.loop_body_contains_control_flow(
                    getattr(
                        statement, "else_branch", getattr(statement, "else_body", None)
                    )
                ):
                    return True
            elif isinstance(statement, MatchNode):
                for arm in getattr(statement, "arms", []) or []:
                    if self.loop_body_contains_control_flow(getattr(arm, "body", None)):
                        return True
            elif isinstance(statement, BlockNode):
                if self.loop_body_contains_control_flow(statement):
                    return True
        return False

    def if_statement_terminates_all_paths(self, node):
        then_branch = getattr(node, "then_branch", getattr(node, "if_body", None))
        else_branch = getattr(node, "else_branch", getattr(node, "else_body", None))
        if else_branch is None:
            return False
        return self.statement_body_terminates(
            then_branch
        ) and self.statement_body_terminates(else_branch)

    def generate_match_expression(
        self,
        node,
        indent=0,
        target_type=None,
        return_context=False,
    ):
        """Render a match node where Rust expects an expression value."""
        code = self.generate_match(
            node,
            indent,
            expression_target_type=target_type,
            return_context=return_context,
        ).rstrip("\n")
        indent_str = "    " * indent
        if indent_str and code.startswith(indent_str):
            code = code[len(indent_str) :]
        return code

    def register_match_pattern_bindings(self, pattern, subject_type):
        """Track types introduced by a match pattern while rendering its arm."""
        if isinstance(pattern, IdentifierPatternNode):
            if pattern.name.isidentifier() and subject_type is not None:
                self.register_variable_type(
                    pattern.name,
                    subject_type,
                    scope="local",
                )
            return

        if isinstance(pattern, StructPatternNode):
            pattern_type = self.specialize_pattern_type(
                pattern.type_name,
                subject_type,
            )
            variant_name = self.match_pattern_variant_name_for_subject(
                pattern,
                subject_type,
            )
            variant_field_types = (
                self.resolve_enum_variant_field_types(subject_type, variant_name)
                if variant_name is not None
                else {}
            )
            for field_name, field_pattern in pattern.field_patterns.items():
                field_type = variant_field_types.get(field_name)
                if field_type is None:
                    field_type = self.resolve_struct_member_type(
                        pattern_type,
                        field_name,
                    )
                if field_type is None:
                    field_type = self.resolve_struct_member_type(
                        pattern.type_name,
                        field_name,
                    )
                self.register_match_pattern_bindings(field_pattern, field_type)
            return

        if isinstance(pattern, ConstructorPatternNode):
            for index, arg_pattern in enumerate(pattern.arguments):
                arg_type = self.constructor_pattern_argument_type(
                    pattern.type_name,
                    subject_type,
                    index,
                )
                self.register_match_pattern_bindings(arg_pattern, arg_type)

    def specialize_pattern_type(self, pattern_type, subject_type):
        """Prefer the scrutinee's generic arguments for matching struct patterns."""
        if subject_type is None:
            return pattern_type
        subject_base, _ = self.generic_type_parts(subject_type)
        pattern_base, _ = self.generic_type_parts(pattern_type)
        if subject_base == pattern_base:
            return subject_type
        return pattern_type

    def constructor_pattern_argument_type(self, pattern_type, subject_type, index):
        """Infer payload types for common generic enum-like constructor patterns."""
        if subject_type is None:
            return None
        base, args = self.generic_type_parts(subject_type)
        variant_base, variant = self.split_variant_path(
            self.normalize_turbofish_type_name(pattern_type)
        )
        if variant_base is not None:
            variant_base, _variant_args = self.generic_type_parts(variant_base)
            if variant_base != base:
                return None
        payload_types = self.resolve_enum_variant_payload_types(subject_type, variant)
        if index < len(payload_types):
            return payload_types[index]
        if base == "Option" and variant == "Some" and index == 0 and args:
            return args[0]
        if base == "Result" and args:
            if variant == "Ok" and index == 0:
                return args[0]
            if variant == "Err" and index == 0 and len(args) > 1:
                return args[1]
        return None

    def is_supported_match_arm(self, arm):
        pattern = getattr(arm, "pattern", None)
        return isinstance(
            pattern,
            (
                LiteralPatternNode,
                WildcardPatternNode,
                IdentifierPatternNode,
                ConstructorPatternNode,
                StructPatternNode,
            ),
        )

    def match_arms_are_exhaustive(self, node, subject_type):
        """Return whether known bool/enum match arms already cover all cases."""
        arms = getattr(node, "arms", []) or []
        if self.match_arms_cover_bool(arms, subject_type):
            return True
        if self.match_arms_cover_struct(arms, subject_type):
            return True

        variants = self.enum_variants_for_type(subject_type)
        if not variants:
            return False

        covered_variants = set()
        conditional_variant_patterns = {variant: [] for variant in variants}
        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            pattern = getattr(arm, "pattern", None)
            variant_name = self.match_pattern_variant_name_for_subject(
                pattern,
                subject_type,
            )
            if variant_name is None:
                continue
            if self.is_unconditional_variant_pattern(pattern):
                covered_variants.add(variant_name)
            else:
                conditional_variant_patterns[variant_name].append(pattern)

        for variant in variants:
            if variant in covered_variants:
                continue
            if self.variant_patterns_cover_nested_enum_field(
                conditional_variant_patterns.get(variant, []),
                subject_type,
                variant,
            ):
                continue
            return False

        return True

    def match_arms_cover_bool(self, arms, subject_type):
        if self.normalize_scalar_type(subject_type) != "bool":
            return False

        covered = set()
        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            value = self.bool_match_pattern_value(getattr(arm, "pattern", None))
            if value is not None:
                covered.add(value)

        return covered == {True, False}

    def match_arms_cover_struct(self, arms, subject_type):
        subject_base, _generic_args = self.generic_type_parts(subject_type)
        member_types = self.struct_member_types.get(subject_base)
        if not member_types:
            return False

        for arm in arms:
            if getattr(arm, "guard", None) is not None:
                continue
            pattern = getattr(arm, "pattern", None)
            if not isinstance(pattern, StructPatternNode):
                continue
            pattern_base, _pattern_args = self.generic_type_parts(pattern.type_name)
            if pattern_base != subject_base:
                continue
            if not self.is_unconditional_variant_pattern(pattern):
                continue
            if getattr(pattern, "has_rest", False) or set(
                pattern.field_patterns
            ) == set(member_types):
                return True

        return False

    def bool_match_pattern_value(self, pattern):
        if isinstance(pattern, LiteralPatternNode):
            value = getattr(pattern.literal, "value", None)
            if isinstance(value, bool):
                return value
        if isinstance(pattern, IdentifierPatternNode):
            if pattern.name == "true":
                return True
            if pattern.name == "false":
                return False
        return None

    def enum_variants_for_type(self, subject_type):
        if subject_type is None:
            return []
        base_type, _generic_args = self.generic_type_parts(subject_type)
        return self.enum_variant_names.get(base_type, [])

    def match_pattern_variant_name_for_subject(self, pattern, subject_type):
        variant_path = self.match_pattern_variant_path(pattern)
        if variant_path is None:
            return None

        subject_base, _generic_args = self.generic_type_parts(subject_type)
        variant_base, variant_name = self.split_variant_path(variant_path)
        if variant_base is not None and variant_base != subject_base:
            return None

        variants = self.enum_variants_for_type(subject_type)
        if variant_name not in variants:
            return None

        return variant_name

    def unconditional_match_variant_name(self, pattern, subject_type):
        variant_name = self.match_pattern_variant_name_for_subject(
            pattern,
            subject_type,
        )
        if variant_name is None:
            return None
        if not self.is_unconditional_variant_pattern(pattern):
            return None

        return variant_name

    def variant_patterns_cover_nested_enum_field(
        self,
        patterns,
        subject_type,
        variant_name,
    ):
        if not patterns:
            return False

        subject_base, _generic_args = self.generic_type_parts(subject_type)
        variant_fields = self.enum_variant_field_types.get(subject_base, {}).get(
            variant_name,
            {},
        )
        if not variant_fields:
            return False

        for field_name, field_type in variant_fields.items():
            nested_variants = self.enum_variants_for_type(field_type)
            if not nested_variants:
                continue

            covered_nested_variants = set()
            for pattern in patterns:
                if not isinstance(pattern, StructPatternNode):
                    continue
                if not getattr(pattern, "has_rest", False) and set(
                    pattern.field_patterns
                ) != set(variant_fields):
                    continue
                if not self.struct_pattern_other_fields_are_unconditional(
                    pattern,
                    field_name,
                ):
                    continue
                field_pattern = pattern.field_patterns.get(field_name)
                if field_pattern is None:
                    if getattr(pattern, "has_rest", False):
                        return True
                    continue
                if self.is_unconditional_payload_pattern(field_pattern):
                    return True
                nested_variant = self.unconditional_match_variant_name(
                    field_pattern,
                    field_type,
                )
                if nested_variant is not None:
                    covered_nested_variants.add(nested_variant)

            if set(nested_variants).issubset(covered_nested_variants):
                return True

        return False

    def struct_pattern_other_fields_are_unconditional(self, pattern, skipped_field):
        for field_name, field_pattern in pattern.field_patterns.items():
            if field_name == skipped_field:
                continue
            if not self.is_unconditional_payload_pattern(field_pattern):
                return False
        return True

    def match_pattern_variant_path(self, pattern):
        if isinstance(pattern, ConstructorPatternNode):
            return pattern.type_name
        if isinstance(pattern, StructPatternNode):
            return pattern.type_name
        if isinstance(pattern, IdentifierPatternNode):
            return pattern.name
        return None

    def split_variant_path(self, path):
        path = str(path)
        if "::" not in path:
            return None, path
        base, variant_name = path.rsplit("::", 1)
        return base, variant_name

    def is_unconditional_variant_pattern(self, pattern):
        if isinstance(pattern, IdentifierPatternNode):
            return "::" in pattern.name
        if isinstance(pattern, ConstructorPatternNode):
            return all(
                self.is_unconditional_payload_pattern(argument)
                for argument in getattr(pattern, "arguments", []) or []
            )
        if isinstance(pattern, StructPatternNode):
            return all(
                self.is_unconditional_payload_pattern(field_pattern)
                for field_pattern in getattr(pattern, "field_patterns", {}).values()
            )
        return False

    def is_unconditional_payload_pattern(self, pattern):
        if isinstance(pattern, WildcardPatternNode):
            return True
        if isinstance(pattern, IdentifierPatternNode):
            return "::" not in pattern.name
        return False

    def unused_match_pattern_binding_names(self, pattern, guard, body):
        """Return pattern bindings that are not referenced by the arm."""
        binding_names = self.match_pattern_binding_names(pattern)
        if not binding_names:
            return set()

        used_names = self.collect_referenced_names([guard, body])
        return binding_names - used_names

    def mutable_match_pattern_binding_names(self, pattern, body):
        """Return pattern bindings that are reassigned inside the arm body."""
        binding_names = self.match_pattern_binding_names(pattern)
        if not binding_names:
            return set()

        return binding_names & self.collect_mutated_binding_names(body)

    def match_pattern_binding_names(self, pattern):
        """Collect identifiers introduced by a match pattern."""
        if isinstance(pattern, IdentifierPatternNode):
            if self.is_match_binding_identifier(pattern.name):
                return {pattern.name}
            return set()
        if isinstance(pattern, ConstructorPatternNode):
            names = set()
            for argument in getattr(pattern, "arguments", []) or []:
                names.update(self.match_pattern_binding_names(argument))
            return names
        if isinstance(pattern, StructPatternNode):
            names = set()
            for field_pattern in getattr(pattern, "field_patterns", {}).values():
                names.update(self.match_pattern_binding_names(field_pattern))
            return names
        return set()

    def is_match_binding_identifier(self, name):
        """Return whether a pattern identifier introduces a Rust binding."""
        return (
            isinstance(name, str)
            and name.isidentifier()
            and name not in {"true", "false"}
        )

    def collect_referenced_names(self, node):
        """Collect identifier references from statements and expressions."""
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if hasattr(current, "statements"):
                collect(current.statements)
                return

            if isinstance(current, IdentifierNode):
                names.add(current.name)
                return

            if isinstance(current, str):
                if current.isidentifier():
                    names.add(current)
                return

            if isinstance(current, VariableNode):
                collect(getattr(current, "initial_value", None))
                return

            if isinstance(current, AssignmentNode):
                collect(current.target)
                collect(current.value)
                return

            if isinstance(current, ExpressionStatementNode):
                collect(current.expression)
                return

            if isinstance(current, ReturnNode):
                collect(current.value)
                return

            if isinstance(current, IfNode):
                collect(current.condition)
                collect(current.then_branch)
                collect(current.else_branch)
                return

            if isinstance(current, ForNode):
                collect(current.init)
                collect(current.condition)
                collect(current.update)
                collect(current.body)
                return

            if isinstance(current, ForInNode):
                collect(current.iterable)
                collect(current.body)
                return

            if isinstance(current, (WhileNode, DoWhileNode)):
                collect(current.condition)
                collect(current.body)
                return

            if isinstance(current, LoopNode):
                collect(current.body)
                return

            if isinstance(current, MatchNode):
                collect(current.expression)
                for arm in getattr(current, "arms", []) or []:
                    collect(getattr(arm, "guard", None))
                    collect(getattr(arm, "body", None))
                return

            if isinstance(current, SwitchNode):
                collect(getattr(current, "expression", None))
                for case in getattr(current, "cases", []) or []:
                    collect(getattr(case, "value", None))
                    collect(getattr(case, "statements", []))
                collect(getattr(current, "default_case", None))
                return

            if isinstance(current, UnaryOpNode):
                collect(current.operand)
                return

            if isinstance(current, BinaryOpNode):
                collect(current.left)
                collect(current.right)
                return

            if isinstance(current, TernaryOpNode):
                collect(current.condition)
                collect(current.true_expr)
                collect(current.false_expr)
                return

            if isinstance(current, FunctionCallNode):
                collect(current.function)
                collect(getattr(current, "arguments", getattr(current, "args", [])))
                return

            if isinstance(current, TextureNode):
                collect(getattr(current, "texture_expr", None))
                collect(getattr(current, "sampler_expr", None))
                collect(getattr(current, "coordinates", None))
                collect(getattr(current, "level", None))
                collect(getattr(current, "offset", None))
                return

            if isinstance(current, TextureOpNode):
                collect(getattr(current, "texture_expr", None))
                collect(getattr(current, "sampler_expr", None))
                collect(getattr(current, "arguments", []))
                return

            if isinstance(current, MemberAccessNode):
                collect(current.object_expr)
                return

            if isinstance(current, PointerAccessNode):
                collect(current.pointer_expr)
                return

            if isinstance(current, ArrayAccessNode):
                collect(current.array_expr)
                collect(current.index_expr)
                return

            if isinstance(current, RangeNode):
                collect(current.start)
                collect(current.end)
                return

            if isinstance(current, ConstructorNode):
                collect(getattr(current, "arguments", []))
                collect(list((getattr(current, "named_arguments", {}) or {}).values()))
                return

            if isinstance(current, ArrayLiteralNode):
                collect(getattr(current, "elements", []))

        collect(node)
        return names

    def format_match_binding_identifier(
        self,
        name,
        unused_bindings,
        mutable_bindings=None,
    ):
        """Prefix unused pattern bindings to suppress Rust warnings."""
        mutable_bindings = mutable_bindings or set()
        binding_name = name
        if name in unused_bindings and not name.startswith("_"):
            binding_name = f"_{name}"
        else:
            binding_name = self.rust_identifier(binding_name)
        if name in mutable_bindings:
            return f"mut {binding_name}"
        return binding_name

    def generate_match_pattern(
        self,
        pattern,
        unused_bindings=None,
        mutable_bindings=None,
    ):
        unused_bindings = unused_bindings or set()
        mutable_bindings = mutable_bindings or set()
        if isinstance(pattern, WildcardPatternNode):
            return "_"
        if isinstance(pattern, LiteralPatternNode):
            return self.generate_expression(pattern.literal)
        if isinstance(pattern, IdentifierPatternNode):
            if self.is_match_binding_identifier(pattern.name):
                return self.format_match_binding_identifier(
                    pattern.name,
                    unused_bindings,
                    mutable_bindings,
                )
            return self.rust_identifier(pattern.name)
        if isinstance(pattern, ConstructorPatternNode):
            if not pattern.arguments:
                return pattern.type_name
            args = ", ".join(
                self.generate_match_pattern(arg, unused_bindings, mutable_bindings)
                for arg in pattern.arguments
            )
            return f"{pattern.type_name}({args})"
        if isinstance(pattern, StructPatternNode):
            fields = []
            for field_name, field_pattern in pattern.field_patterns.items():
                field_pattern_code = self.generate_match_pattern(
                    field_pattern,
                    unused_bindings,
                    mutable_bindings,
                )
                field_ident = self.rust_identifier(field_name)
                if (
                    isinstance(field_pattern, IdentifierPatternNode)
                    and field_pattern.name == field_name
                    and field_pattern_code == field_ident
                ):
                    fields.append(field_ident)
                else:
                    fields.append(f"{field_ident}: {field_pattern_code}")
            if getattr(pattern, "has_rest", False):
                fields.append("..")
            return f"{pattern.type_name} {{ {', '.join(fields)} }}"
        raise RustRefusal(f"Unsupported match pattern for Rust codegen: {pattern!r}")

    def generate_match_arm_body(
        self,
        body,
        indent,
        expression_target_type=None,
        return_context=False,
    ):
        statements = self.statement_list(body)
        expression_context = return_context or expression_target_type is not None
        if not expression_context or not statements:
            return "".join(self.generate_statement(stmt, indent) for stmt in statements)

        code = "".join(
            self.generate_statement(stmt, indent) for stmt in statements[:-1]
        )
        tail = statements[-1]
        tail_expression = None
        if isinstance(tail, ExpressionStatementNode) and getattr(
            tail,
            "is_tail_expression",
            False,
        ):
            tail_expression = tail.expression
        elif isinstance(
            tail,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                BinaryOpNode,
                BlockNode,
                ConstructorNode,
                FunctionCallNode,
                IdentifierNode,
                LiteralNode,
                MatchNode,
                MemberAccessNode,
                TernaryOpNode,
                UnaryOpNode,
            ),
        ) or isinstance(tail, str):
            tail_expression = tail

        if tail_expression is None:
            return code + self.generate_statement(tail, indent)

        if return_context:
            tail_value = self.generate_return_branch_expression_with_type(
                tail_expression,
                expression_target_type,
            )
        else:
            tail_value = self.generate_expression_with_type(
                tail_expression,
                expression_target_type,
            )
        tail_value = self.normalize_assignment_rhs(
            expression_target_type,
            tail_expression,
            tail_value,
            "=",
        )
        return code + f"{'    ' * indent}{tail_value}\n"
