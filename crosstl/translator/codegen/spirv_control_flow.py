"""Control flow in the SPIR-V generator: returns, if statements, for-in loops, match selections and results, constructor patterns, enum struct values, match conditions and arm validation, switch case literals, loops, increments, inputs and output semantic aliases."""

from typing import Optional

from ..ast import (
    ArrayLiteralNode,
    BreakNode,
    ConstructorPatternNode,
    ContinueNode,
    CooperativeMatrixOpNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    LiteralPatternNode,
    LoopNode,
    MatchNode,
    RangeNode,
    ReturnNode,
    StructPatternNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
    WildcardPatternNode,
)
from ..refusals import VulkanRefusal
from .array_utils import evaluate_literal_int_expression
from .pointer_reinterpret import PointerReinterpretationError
from .spirv_extension_builtins import extension_any_stage_builtin_info
from .spirv_shading_rate import shading_rate_semantic_builtin
from .spirv_short_circuit import evaluate_loop_condition
from .spirv_values import (
    SpirvId,
    SpirvType,
)
from .spirv_wave_extended import SPIRV_SUBGROUP_BUILTINS


class SpirvControlFlowMixin:
    """Control flow members of ``VulkanSPIRVCodeGen``."""

    def process_return(self, node: ReturnNode):
        """Process a CrossGL return statement."""
        if self.loop_merge_labels or self.cooperative_matrix_selection_depth > 0:
            self.cooperative_matrix_control_flow_diverged = True
        if self.inline_return_context_stack:
            self.process_inline_return(node)
            return
        if hasattr(node, "value") and node.value:
            if isinstance(node.value, list) and node.value:
                return_value = self.process_expression_with_expected_type(
                    node.value[0], self.current_return_type_source
                )
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
            else:
                if isinstance(node.value, ArrayLiteralNode):
                    return_value = self.process_array_literal(
                        node.value, self.current_return_type
                    )
                elif isinstance(node.value, MatchNode):
                    return_value = self.process_match_expression_return(node.value)
                else:
                    return_value = self.process_expression_with_expected_type(
                        node.value, self.current_return_type_source
                    )
                if return_value:
                    self.create_return_value(return_value)
                else:
                    self.create_return()
        else:
            self.create_return()

    def process_inline_return(self, node: ReturnNode):
        """Exit the active inline invocation without returning from its caller."""
        context = self.inline_return_context_stack[-1]
        value_expression = getattr(node, "value", None)
        if isinstance(value_expression, list):
            value_expression = value_expression[0] if value_expression else None

        if context.result_pointer is not None:
            if value_expression is None:
                raise VulkanRefusal("non-void inline return requires a value")
            if isinstance(value_expression, ArrayLiteralNode):
                return_value = self.process_array_literal(
                    value_expression, context.return_type
                )
            elif isinstance(value_expression, MatchNode):
                return_value = self.process_match_expression_return(value_expression)
            else:
                expected_type = (
                    self.current_return_type_source
                    or context.return_type.type.base_type
                )
                return_value = self.process_expression_with_expected_type(
                    value_expression, expected_type
                )
            if return_value is None:
                raise VulkanRefusal("non-void inline return value could not be lowered")
            self.store_to_variable(
                context.result_pointer,
                self.convert_value_to_type(return_value, context.return_type),
            )
        elif isinstance(value_expression, MatchNode):
            self.process_match(value_expression)
        elif value_expression is not None:
            self.process_expression(value_expression)

        bool_type = self.register_primitive_type("bool")
        self.store_to_variable(
            context.returned_pointer,
            self.register_constant(True, bool_type),
        )
        if len(self.loop_merge_labels) > context.merge_stack_depth:
            self.create_branch(self.loop_merge_labels[-1])

    def process_match_expression_return(self, node: MatchNode) -> Optional[SpirvId]:
        """Lower return-position matches through a temporary selected value."""
        return_type = self.current_return_type
        if return_type is None or return_type.type.base_type == "void":
            self.process_match(node)
            return None

        result_pointer = self.create_variable(return_type, "Function", "__match_return")
        default_value = self.default_value_for_type(return_type)
        if default_value is not None:
            self.store_to_variable(result_pointer, default_value)
        self.process_match_expression_assignment(node, result_pointer, return_type)
        return self.load_from_variable(result_pointer, return_type)

    def process_if(self, node: IfNode):
        """Process a CrossGL if statement."""
        literal_condition = evaluate_literal_int_expression(
            node.if_condition,
            self.active_literal_int_constants(),
        )
        if literal_condition is not None:
            reachable_body = node.if_body if literal_condition != 0 else node.else_body
            if reachable_body:
                self.process_statements(reachable_body)
            return

        condition = self.process_expression(node.if_condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])
        condition = self.ensure_bool_value(condition)

        constant_condition = self.constant_values_by_id.get(condition.id)
        if isinstance(constant_condition, bool):
            reachable_body = node.if_body if constant_condition else node.else_body
            if reachable_body:
                self.process_statements(reachable_body)
            return

        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        then_label = SpirvId(self.get_id(), SpirvType("label"))
        else_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, then_label, else_label)

        self.cooperative_matrix_selection_depth += 1
        try:
            self.emit(f"%{then_label.id} = OpLabel")
            self.current_label = then_label.id
            self.process_statements(node.if_body)
            then_terminated = self.current_block_has_terminator()
            if not then_terminated:
                self.create_branch(merge_label)

            self.emit(f"%{else_label.id} = OpLabel")
            self.current_label = else_label.id
            if node.else_body:
                self.process_statements(node.else_body)
            else_terminated = self.current_block_has_terminator()
            if not else_terminated:
                self.create_branch(merge_label)
        finally:
            self.cooperative_matrix_selection_depth -= 1

        if then_terminated or else_terminated:
            self.cooperative_matrix_control_flow_diverged = True

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def branch_after_loop_body(
        self, body, merge_label: SpirvId, continue_label: SpirvId
    ):
        """Exit an enclosing loop when a nested inline call path returned."""
        if not self.inline_return_context_stack or not self.statement_contains_return(
            body
        ):
            self.create_branch(continue_label)
            return

        context = self.inline_return_context_stack[-1]
        bool_type = self.register_primitive_type("bool")
        returned = self.load_from_variable(context.returned_pointer, bool_type)
        fallthrough_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_selection_merge(fallthrough_label)
        self.create_conditional_branch(returned, merge_label, fallthrough_label)

        self.emit(f"%{fallthrough_label.id} = OpLabel")
        self.current_label = fallthrough_label.id
        self.create_branch(continue_label)

    def process_for(self, node: ForNode):
        """Process a CrossGL for loop."""
        # ``for (int a = 0, b = n; ...)`` keeps its declarations as a list, run in order.
        for init in node.init if isinstance(node.init, list) else [node.init]:
            if init:
                self.process_statement(init)

        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        condition = evaluate_loop_condition(
            self, node.condition, header_label, merge_label, continue_label
        )
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.branch_after_loop_body(node.body, merge_label, continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        if node.update:
            self.process_loop_control_statement(node.update)
        if not self.current_block_has_terminator():
            self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def for_in_sequence_element_info(self, iterable):
        """Return fixed-size sequence element info for array/vector for-in loops."""
        iterable_pointer = self.variable_pointer_from_expression(iterable)
        if iterable_pointer is not None:
            iterable_type = self.variable_value_types.get(iterable_pointer.id)
            array_info = self.array_type_info_from_type(iterable_type)
            if array_info is not None:
                return array_info

            vector_info = self.vector_type_info_from_type(iterable_type)
            if vector_info is not None:
                return vector_info

        iterable_value = self.process_expression(iterable)
        if iterable_value is None:
            return None

        iterable_type = self.value_types.get(
            iterable_value.id
        ) or self.find_registered_type_by_base(iterable_value.type.base_type)
        array_info = self.array_type_info_from_type(iterable_type)
        if array_info is not None:
            return array_info

        return self.vector_type_info_from_type(iterable_type)

    def for_in_sequence_size(self, size) -> Optional[int]:
        """Return the literal sequence length when it is known at compile time."""
        if isinstance(size, int):
            return size
        if isinstance(size, str):
            try:
                return int(size, 0)
            except ValueError:
                return None
        literal_size = evaluate_literal_int_expression(
            size, self.active_literal_int_constants()
        )
        return int(literal_size) if literal_size is not None else None

    def process_sequence_for_in(self, node: ForInNode, pattern: str, iterable) -> bool:
        """Process array/vector CrossGL for-in loops as element iteration."""
        sequence_info = self.for_in_sequence_element_info(iterable)
        if sequence_info is None:
            return False

        element_type, sequence_size = sequence_info
        sequence_size = self.for_in_sequence_size(sequence_size)
        if sequence_size is None:
            self.refuse_lowering(
                "for-in over dynamically sized sequence requires "
                "a compile-time length"
            )
            return True

        int_type = self.register_primitive_type("int")
        start = self.register_constant(0, int_type)
        end_value = self.register_constant(sequence_size, int_type)

        previous_variable = self.local_variables.get(pattern)
        previous_local_type = self.local_variable_types.get(pattern)
        previous_was_alias = pattern in self.resource_alias_variables

        loop_index = self.create_variable(int_type, "Function")
        pattern_variable = self.create_variable(element_type, "Function", pattern)
        self.local_variables[pattern] = pattern_variable
        self.local_variable_types.pop(pattern, None)
        self.resource_alias_variables.discard(pattern)
        self.store_to_variable(loop_index, start)

        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        loop_value = self.get_variable_value(loop_index)
        condition = self.binary_operation("<", int_type, loop_value, end_value)

        self.create_loop_merge(merge_label, continue_label)
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        loop_value = self.get_variable_value(loop_index)
        element_pointer, current_element_type = self.create_array_element_access(
            iterable, loop_value, iterable
        )
        if element_pointer is not None:
            element_value = self.get_variable_value(element_pointer)
            target_type = current_element_type or element_type
            element_value = self.convert_value_to_type(element_value, target_type)
            self.store_to_variable(pattern_variable, element_value)

        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.branch_after_loop_body(node.body, merge_label, continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        current_value = self.get_variable_value(loop_index)
        one = self.register_constant(1, int_type)
        next_value = self.binary_operation("+", int_type, current_value, one)
        self.store_to_variable(loop_index, next_value)
        if not self.current_block_has_terminator():
            self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

        if previous_variable is None:
            self.local_variables.pop(pattern, None)
        else:
            self.local_variables[pattern] = previous_variable

        if previous_local_type is None:
            self.local_variable_types.pop(pattern, None)
        else:
            self.local_variable_types[pattern] = previous_local_type

        if previous_was_alias:
            self.resource_alias_variables.add(pattern)
        else:
            self.resource_alias_variables.discard(pattern)

        return True

    def process_for_in(self, node: ForInNode):
        """Process CrossGL for-in loops as structured counted loops."""
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)
        int_type = self.register_primitive_type("int")

        if isinstance(iterable, RangeNode):
            start = self.process_loop_control_expression(iterable.start)
            end_expr = iterable.end
            comparator = "<=" if iterable.inclusive else "<"
        elif self.process_sequence_for_in(node, pattern, iterable):
            return
        else:
            start = self.register_constant(0, int_type)
            end_expr = iterable
            comparator = "<"

        if start is None:
            if isinstance(iterable, RangeNode) and iterable.start is not None:
                self.refuse_lowering(
                    "for-in range start has no value: "
                    f"{self.diagnostic_expression(iterable.start)}"
                )
            start = self.register_constant(0, int_type)

        previous_variable = self.local_variables.get(pattern)
        loop_variable = self.create_variable(int_type, "Function", pattern)
        self.local_variables[pattern] = loop_variable
        self.store_to_variable(loop_variable, start)

        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        loop_value = self.get_variable_value(loop_variable)
        end_value = self.process_loop_control_expression(end_expr)
        if end_value is None:
            self.refuse_lowering(
                "for-in bound has no value: " f"{self.diagnostic_expression(end_expr)}"
            )
            end_value = self.register_constant(0, int_type)
        condition = self.binary_operation(comparator, int_type, loop_value, end_value)

        self.create_loop_merge(merge_label, continue_label)
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.branch_after_loop_body(node.body, merge_label, continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        current_value = self.get_variable_value(loop_variable)
        one = self.register_constant(1, int_type)
        next_value = self.binary_operation("+", int_type, current_value, one)
        self.store_to_variable(loop_variable, next_value)
        if not self.current_block_has_terminator():
            self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

        if previous_variable is None:
            self.local_variables.pop(pattern, None)
        else:
            self.local_variables[pattern] = previous_variable

    def process_match(self, node: MatchNode):
        """Process CrossGL match statements as an ordered SPIR-V selection chain."""
        self.process_match_selection(
            node,
            lambda arm: self.process_match_statement_body(getattr(arm, "body", [])),
        )

    def process_match_expression_assignment(
        self, node: MatchNode, target_pointer: SpirvId, target_type: SpirvId
    ):
        """Lower a value-position match into stores to an existing local."""
        self.process_match_selection(
            node,
            lambda arm: self.process_match_expression_arm_body(
                getattr(arm, "body", []), target_pointer, target_type
            ),
        )

    def process_match_selection(self, node: MatchNode, process_arm_body):
        arms = getattr(node, "arms", []) or []
        expression = self.process_expression(getattr(node, "expression", None))
        if expression is None:
            self.refuse_lowering(
                "match scrutinee has no value: "
                f"{self.diagnostic_expression(getattr(node, 'expression', None))}"
            )
            expression = self.register_constant(0, self.register_primitive_type("int"))

        if not arms:
            return

        bool_type = self.register_primitive_type("bool")
        matched_variable = self.create_variable(
            bool_type, "Function", "__match_matched"
        )
        self.store_to_variable(
            matched_variable, self.register_constant(False, bool_type)
        )
        matched_true = self.register_constant(True, bool_type)

        for arm in arms:
            condition, bindings = self.lower_match_pattern_condition(
                getattr(arm, "pattern", None), expression
            )
            restore_bindings = self.apply_match_bindings(bindings)
            try:
                guard = getattr(arm, "guard", None)
                if guard is not None:
                    guard_condition = self.process_expression(guard)
                    condition = self.combine_match_conditions(
                        condition, guard_condition
                    )

                already_matched = self.load_from_variable(matched_variable, bool_type)
                not_matched = self.unary_operation("!", bool_type, already_matched)
                condition = self.combine_match_conditions(not_matched, condition)

                body_label = SpirvId(self.get_id(), SpirvType("label"))
                next_label = SpirvId(self.get_id(), SpirvType("label"))

                self.create_selection_merge(next_label)
                self.create_conditional_branch(condition, body_label, next_label)

                self.emit(f"%{body_label.id} = OpLabel")
                self.current_label = body_label.id
                self.cooperative_matrix_selection_depth += 1
                try:
                    process_arm_body(arm)
                finally:
                    self.cooperative_matrix_selection_depth -= 1
                arm_terminated = self.current_block_has_terminator()
                if arm_terminated:
                    self.cooperative_matrix_control_flow_diverged = True
                else:
                    self.store_to_variable(matched_variable, matched_true)
                    self.create_branch(next_label)

                self.emit(f"%{next_label.id} = OpLabel")
                self.current_label = next_label.id
            finally:
                restore_bindings()

    def process_match_statement_body(self, body):
        self.process_statements(body)

    def process_match_expression_arm_body(
        self, body, target_pointer: SpirvId, target_type: SpirvId
    ):
        statements = getattr(body, "statements", None)
        if statements is not None:
            statements = list(statements)
            if statements and getattr(statements[-1], "is_tail_expression", False):
                self.process_statements(statements[:-1])
                self.store_match_expression_result(
                    getattr(statements[-1], "expression", None),
                    target_pointer,
                    target_type,
                )
                return
            self.process_statements(statements)
            return

        if hasattr(body, "expression"):
            self.store_match_expression_result(
                getattr(body, "expression", None), target_pointer, target_type
            )
            return

        self.process_statement(body)

    def store_match_expression_result(
        self, expression, target_pointer: SpirvId, target_type: SpirvId
    ):
        value = self.process_expression_with_expected_type(
            expression, target_type.type.base_type
        )
        if value is None:
            return
        self.store_to_variable(
            target_pointer, self.convert_value_to_type(value, target_type)
        )

    def lower_match_pattern_condition(self, pattern, expression: SpirvId):
        if isinstance(pattern, LiteralPatternNode):
            pattern_value = self.process_expression(pattern.literal)
            if pattern_value is None:
                pattern_value = self.register_constant(
                    0, self.register_primitive_type("int")
                )
            return self.compare_match_values(expression, pattern_value), []

        if isinstance(pattern, WildcardPatternNode):
            return None, []

        if isinstance(pattern, IdentifierPatternNode):
            return self.lower_identifier_match_pattern(pattern, expression)

        if isinstance(pattern, StructPatternNode):
            return self.lower_struct_match_pattern(pattern, expression)

        if isinstance(pattern, ConstructorPatternNode):
            return self.lower_constructor_match_pattern(pattern, expression)

        self.raise_match_pattern_gap(
            f"{type(pattern).__name__} patterns are not lowerable"
        )

    def lower_identifier_match_pattern(
        self, pattern: IdentifierPatternNode, expression: SpirvId
    ):
        name = pattern.name
        if name in {"_", ".."}:
            return None, []

        enum_constant = self.enum_variant_constant(name)
        if enum_constant is not None:
            return self.enum_variant_match_condition(expression, name), []

        if "::" in name:
            self.raise_match_pattern_gap(
                "enum path patterns without a registered plain enum "
                f"discriminant are not lowerable: {name}"
            )

        return None, [(name, expression)]

    def lower_constructor_match_pattern(
        self, pattern: ConstructorPatternNode, expression: SpirvId
    ):
        if pattern.type_name in self.enum_variant_values:
            condition = self.enum_variant_match_condition(expression, pattern.type_name)
            field_condition, bindings = self.lower_enum_payload_pattern_bindings(
                expression,
                pattern.type_name,
                list(getattr(pattern, "arguments", []) or []),
            )
            return self.combine_match_conditions(condition, field_condition), bindings

        self.raise_match_pattern_gap(
            "payload enum constructor patterns require tagged enum struct lowering"
        )

    def lower_struct_match_pattern(
        self, pattern: StructPatternNode, expression: SpirvId
    ):
        if "::" in pattern.type_name:
            condition = self.enum_variant_match_condition(expression, pattern.type_name)
            field_condition, bindings = self.lower_enum_struct_pattern_bindings(
                expression,
                pattern.type_name,
                getattr(pattern, "field_patterns", {}) or {},
            )
            return self.combine_match_conditions(condition, field_condition), bindings

        struct_type_name = expression.type.base_type
        if pattern.type_name != struct_type_name:
            self.raise_match_pattern_gap(
                "struct pattern " f"{pattern.type_name} cannot match {struct_type_name}"
            )

        condition = None
        bindings = []
        for field_name, field_pattern in (pattern.field_patterns or {}).items():
            member_info = self.struct_member_info(struct_type_name, field_name)
            if member_info is None:
                self.raise_match_pattern_gap(
                    "struct pattern "
                    f"field {field_name} does not exist on {struct_type_name}"
                )
            member_index, member_type = member_info
            field_value = self.composite_extract(expression, member_type, member_index)
            field_condition, field_bindings = self.lower_match_pattern_condition(
                field_pattern, field_value
            )
            condition = self.combine_match_conditions(condition, field_condition)
            bindings.extend(field_bindings)

        return condition, bindings

    def enum_variant_match_condition(self, expression: SpirvId, path: str) -> SpirvId:
        enum_constant = self.enum_variant_constant(path)
        if enum_constant is None:
            self.raise_match_pattern_gap(f"unknown enum variant {path}")

        parts = self.enum_path_parts(path)
        enum_name = parts[0] if parts else None
        if enum_name in self.enum_struct_type_names:
            variant_value = self.enum_struct_variant_value(expression, enum_name)
            return self.compare_match_values(variant_value, enum_constant)

        return self.compare_match_values(expression, enum_constant)

    def enum_struct_variant_value(self, expression: SpirvId, enum_name: str) -> SpirvId:
        struct_name = expression.type.base_type
        if struct_name != enum_name:
            specialization = self.generic_enum_specialization_for_struct_name(
                struct_name
            )
            if specialization is None or specialization["base_name"] != enum_name:
                self.raise_match_pattern_gap(
                    f"enum pattern {enum_name} cannot match {struct_name}"
                )
        member_info = self.struct_member_info(struct_name, "variant")
        if member_info is None:
            self.raise_match_pattern_gap(f"enum {struct_name} has no variant tag")
        member_index, member_type = member_info
        return self.composite_extract(expression, member_type, member_index)

    def lower_enum_payload_pattern_bindings(
        self, expression: SpirvId, path: str, argument_patterns
    ):
        variant_fields = self.enum_variant_fields_for_path(path, expression)
        if variant_fields is None:
            return []
        if len(argument_patterns) != len(variant_fields):
            raise VulkanRefusal(
                f"Enum pattern {path} expects {len(variant_fields)} arguments, "
                f"got {len(argument_patterns)}"
            )

        condition = None
        bindings = []
        for pattern, (field_name, _field_type) in zip(
            argument_patterns, variant_fields
        ):
            field_value = self.enum_struct_field_value(expression, field_name)
            field_condition, field_bindings = self.lower_match_pattern_condition(
                pattern, field_value
            )
            condition = self.combine_match_conditions(condition, field_condition)
            bindings.extend(field_bindings)
        return condition, bindings

    def lower_enum_struct_pattern_bindings(
        self, expression: SpirvId, path: str, field_patterns
    ):
        variant_fields = dict(self.enum_variant_fields_for_path(path, expression) or [])
        condition = None
        bindings = []
        for field_name, field_pattern in field_patterns.items():
            if field_name not in variant_fields:
                self.raise_match_pattern_gap(
                    f"enum pattern {path} has no payload field {field_name}"
                )
            field_value = self.enum_struct_field_value(expression, field_name)
            field_condition, field_bindings = self.lower_match_pattern_condition(
                field_pattern, field_value
            )
            condition = self.combine_match_conditions(condition, field_condition)
            bindings.extend(field_bindings)
        return condition, bindings

    def enum_struct_field_value(self, expression: SpirvId, field_name: str) -> SpirvId:
        member_info = self.struct_member_info(expression.type.base_type, field_name)
        if member_info is None:
            self.raise_match_pattern_gap(
                f"enum payload field {field_name} does not exist on "
                f"{expression.type.base_type}"
            )
        member_index, member_type = member_info
        return self.composite_extract(expression, member_type, member_index)

    def raise_match_pattern_gap(self, detail: str):
        raise VulkanRefusal(f"SPIR-V match pattern unsupported: {detail}")

    def compare_match_values(self, expression: SpirvId, pattern_value: SpirvId):
        return self.binary_operation(
            "==",
            self.ensure_registered_type(expression.type),
            expression,
            pattern_value,
        )

    def combine_match_conditions(
        self, left: Optional[SpirvId], right: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if left is None:
            return right
        if right is None:
            return left
        return self.binary_operation(
            "&&", self.register_primitive_type("bool"), left, right
        )

    def apply_match_bindings(self, bindings):
        missing = object()
        previous_values = []
        for name, value in bindings:
            if name in {"_", ".."}:
                continue
            previous_values.append((name, self.local_variables.get(name, missing)))
            value_type = self.ensure_registered_type(value.type)
            variable = self.create_variable(value_type, "Function", name)
            self.local_variables[name] = variable
            self.store_to_variable(variable, value)

        def restore():
            for name, previous in reversed(previous_values):
                if previous is missing:
                    self.local_variables.pop(name, None)
                else:
                    self.local_variables[name] = previous

        return restore

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def validate_match_arms(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if not self.is_supported_match_arm(arm):
                return False
            if isinstance(getattr(arm, "pattern", None), WildcardPatternNode):
                if wildcard_index is not None:
                    return False
                wildcard_index = index
        return wildcard_index is None or wildcard_index == len(arms) - 1

    def switch_case_statements(self, case):
        """Return the body statements for a switch case/default node."""
        if case is None:
            return []
        if hasattr(case, "statements"):
            return getattr(case, "statements") or []
        if hasattr(case, "body"):
            return getattr(case, "body") or []
        if isinstance(case, list):
            return case
        return [case]

    def process_switch(self, node: SwitchNode):
        """Process CrossGL switch statements as a structured OpSwitch."""
        cases = getattr(node, "cases", []) or []
        expression = self.process_expression(getattr(node, "expression", None))
        if expression is None:
            self.refuse_lowering(
                "switch selector has no value: "
                f"{self.diagnostic_expression(getattr(node, 'expression', None))}"
            )
            expression = self.register_constant(0, self.register_primitive_type("int"))

        explicit_cases = [
            case
            for case in cases
            if hasattr(case, "value") and getattr(case, "value", None) is not None
        ]
        default_case = next(
            (
                case
                for case in cases
                if hasattr(case, "value") and getattr(case, "value", None) is None
            ),
            None,
        )
        if default_case is None:
            default_case = getattr(node, "default_case", None)

        if not explicit_cases and default_case is None:
            return

        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        case_labels = []
        for case in explicit_cases:
            literal_value = self.switch_case_literal_value(getattr(case, "value", None))
            if literal_value is None:
                raise VulkanRefusal(
                    "SPIR-V switch case values must be scalar integer literals"
                )
            case_labels.append(
                (literal_value, SpirvId(self.get_id(), SpirvType("label")), case)
            )

        default_label = (
            SpirvId(self.get_id(), SpirvType("label"))
            if default_case is not None
            else merge_label
        )

        switch_operands = " ".join(
            f"{literal_value} %{label.id}"
            for literal_value, label, _case in case_labels
        )
        self.create_selection_merge(merge_label)
        if switch_operands:
            self.emit(
                f"OpSwitch %{expression.id} %{default_label.id} {switch_operands}"
            )
        else:
            self.emit(f"OpSwitch %{expression.id} %{default_label.id}")

        # Cases run in source order: one whose statements do not end in a jump falls through into the next, which
        # SPIR-V expresses as a branch to that case's block (SPIR-V 1.6 section 2.11.1).
        labels_by_case = {id(case): label for _literal, label, case in case_labels}
        sequence = []
        for case in cases:
            if id(case) in labels_by_case:
                sequence.append((labels_by_case[id(case)], case))
            elif default_case is not None and case is default_case:
                sequence.append((default_label, case))
        if default_case is not None and all(
            case is not default_case for _label, case in sequence
        ):
            sequence.append((default_label, default_case))

        for index, (label, case) in enumerate(sequence):
            following = (
                sequence[index + 1][0] if index + 1 < len(sequence) else merge_label
            )
            self.emit(f"%{label.id} = OpLabel")
            self.current_label = label.id
            self.loop_merge_labels.append(merge_label)
            try:
                self.process_statements(self.switch_case_statements(case))
                if not self.current_block_has_terminator():
                    self.create_branch(following)
            finally:
                self.loop_merge_labels.pop()

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def switch_case_literal_value(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        literal_value = getattr(value, "value", None)
        if isinstance(literal_value, bool):
            return int(literal_value)
        if isinstance(literal_value, int):
            return literal_value
        name = getattr(value, "name", None)
        if name in self.enum_variant_values:
            return self.enum_variant_values[name]
        # A case label is an integral constant expression (GLSL 4.60 6.2, HLSL and
        # MSL follow C++), so named constants and their arithmetic fold to literals.
        folded = evaluate_literal_int_expression(
            value, self.active_literal_int_constants()
        )
        return None if folded is None else int(folded)

    def process_while(self, node: WhileNode):
        """Process a CrossGL while loop."""
        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id

        condition = evaluate_loop_condition(
            self, node.condition, header_label, merge_label, continue_label
        )
        self.create_conditional_branch(condition, body_label, merge_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.branch_after_loop_body(node.body, merge_label, continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_do_while(self, node: DoWhileNode):
        """Process a CrossGL do-while loop."""
        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id
        self.create_loop_merge(merge_label, continue_label)
        self.create_branch(body_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.branch_after_loop_body(node.body, merge_label, continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id

        condition = self.process_loop_control_expression(node.condition)
        if condition is None:
            condition = self.register_constant(True, self.primitive_types["bool"])
        condition = self.ensure_bool_value(condition)

        self.create_conditional_branch(condition, header_label, merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_loop(self, node: LoopNode):
        """Process a CrossGL unconditional loop."""
        header_label = SpirvId(self.get_id(), SpirvType("label"))
        body_label = SpirvId(self.get_id(), SpirvType("label"))
        continue_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_branch(header_label)

        self.emit(f"%{header_label.id} = OpLabel")
        self.current_label = header_label.id
        self.create_loop_merge(merge_label, continue_label)
        self.create_branch(body_label)

        self.emit(f"%{body_label.id} = OpLabel")
        self.current_label = body_label.id
        self.loop_merge_labels.append(merge_label)
        self.loop_continue_labels.append(continue_label)
        try:
            if node.body:
                self.process_statements(node.body)
            if not self.current_block_has_terminator():
                self.branch_after_loop_body(node.body, merge_label, continue_label)
        finally:
            self.loop_continue_labels.pop()
            self.loop_merge_labels.pop()

        self.emit(f"%{continue_label.id} = OpLabel")
        self.current_label = continue_label.id
        self.create_branch(header_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def process_break(self, node: BreakNode):
        """Process a CrossGL break statement."""
        if self.loop_merge_labels:
            self.create_branch(self.loop_merge_labels[-1])
            return
        self.refuse_lowering("break used outside a loop or switch")

    def process_continue(self, node: ContinueNode):
        """Process a CrossGL continue statement."""
        if not self.loop_continue_labels:
            self.refuse_lowering("continue used outside a loop")
            return
        self.create_branch(self.loop_continue_labels[-1])

    def process_increment_expression(self, node: UnaryOpNode) -> SpirvId:
        """Process prefix/postfix ++ and -- as load/update/store operations."""
        if (
            isinstance(node.operand, CooperativeMatrixOpNode)
            and node.operand.operation == "element"
        ):
            if not self.cooperative_matrix_khr_enabled:
                self.process_expression(node.operand)
            variable_id, _ = self.cooperative_matrix_element_pointer(node.operand)
        else:
            variable_id = self.storage_reference_pointer(node.operand)
        if variable_id is None:
            self.refuse_lowering("increment target is not assignable")
            int_type = self.register_primitive_type("int")
            return self.register_constant(0, int_type)

        if self.is_storage_buffer_pointer_arithmetic_base(variable_id):
            is_postfix = getattr(
                node,
                "is_postfix",
                getattr(node, "postfix", False),
            )
            old_pointer = (
                self.copy_storage_buffer_pointer_alias(variable_id, None)
                if is_postfix
                else None
            )
            offset_alias = self.storage_buffer_pointer_offset_aliases.get(
                self.storage_buffer_pointer_alias_key(variable_id)
            )
            if offset_alias is None or offset_alias.offset_variable is None:
                offset_type = self.register_primitive_type("int")
            else:
                offset_type = self.variable_value_types[offset_alias.offset_variable.id]
            step_value = self.register_constant(1, offset_type)
            operator = "+" if node.op == "++" else "-"
            self.record_storage_buffer_pointer_offset_alias(
                variable_id,
                variable_id,
                step_value,
                operator,
            )
            return old_pointer or variable_id

        value_type = self.variable_value_types.get(variable_id.id)
        if value_type is None:
            value_type = self.find_registered_type_by_base(
                variable_id.type.base_type.replace("ptr_", "", 1)
            )
        if value_type is None:
            value_type = self.register_primitive_type("int")

        old_value = self.load_from_variable(variable_id, value_type)
        step_value = (
            self.register_constant(1.0, value_type)
            if value_type.type.base_type == "float"
            else self.register_constant(1, value_type)
        )
        operator = "+" if node.op == "++" else "-"
        new_value = self.binary_operation(operator, value_type, old_value, step_value)
        self.store_to_variable(variable_id, new_value)

        if getattr(node, "is_postfix", getattr(node, "postfix", False)):
            return old_value
        return new_value

    def process_non_uniform_function_call(self, function_name, args) -> SpirvId:
        if len(args) != 1:
            self.refuse_lowering(f"{function_name} requires exactly one operand")
            return self.register_constant(0, self.register_primitive_type("int"))

        value = self.process_expression(args[0])
        if value is None:
            self.refuse_lowering(f"{function_name} operand could not be evaluated")
            return self.register_constant(0, self.register_primitive_type("int"))

        result_type = self.value_types.get(value.id) or self.ensure_registered_type(
            value.type
        )
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpCopyObject %{result_type.id} %{value.id}")
        copied = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.mark_non_uniform_result(copied)
        return copied

    def process_storage_pointer_reinterpret_load(self, array_expr, index):
        if isinstance(array_expr, str):
            name = array_expr
        elif isinstance(array_expr, (IdentifierNode, VariableNode)):
            name = array_expr.name
        else:
            return None
        pointer = self.local_variables.get(name)
        if pointer is None:
            return None
        reinterpretation = self.storage_pointer_reinterpretations.get(pointer)
        if reinterpretation is None:
            return None

        uint_type = self.register_primitive_type("uint")
        index = self.convert_value_to_type(index, uint_type)
        root_pointer, element_index, _ = self.storage_buffer_pointer_alias_access(
            pointer, index
        )
        element_index = self.convert_value_to_type(element_index, uint_type)
        target_layout = reinterpretation["target_layout"]
        source_layout = reinterpretation["source_layout"]

        byte_index = element_index
        if target_layout.byte_width != 1:
            target_bytes = self.register_constant(target_layout.byte_width, uint_type)
            byte_index = self.binary_operation(
                "*", uint_type, element_index, target_bytes
            )
        byte_offset = reinterpretation.get("byte_offset")
        if byte_offset is not None:
            byte_index = self.binary_operation("+", uint_type, byte_offset, byte_index)

        source_bytes = self.register_constant(source_layout.byte_width, uint_type)
        source_index = self.binary_operation("/", uint_type, byte_index, source_bytes)
        source_pointer = self.structured_buffer_element_pointer(
            root_pointer, source_index, apply_pointer_offset=False
        )
        if source_pointer is None:
            raise PointerReinterpretationError(
                "SPIR-V pointer reinterpretation lost its storage-buffer backing "
                "object before the read",
                source_type=reinterpretation["source_type"],
                target_type=reinterpretation["target_type"],
                address_space="storage",
                alignment=target_layout.byte_width,
                access="read",
                reason="backing-provenance-unavailable",
            )

        source_type = self.variable_value_types[source_pointer.id]
        source_value = self.load_from_variable(source_pointer, source_type)
        source_component = self.normalize_primitive_name(source_type.type.base_type)
        source_bits = (
            source_value
            if source_component == "uint"
            else self.bitcast_wave_value(source_value, uint_type)
        )

        target_type = reinterpretation["target_type_id"]
        if target_layout.bit_width == 32:
            target_component = self.normalize_primitive_name(target_type.type.base_type)
            if target_component == "uint":
                return source_bits
            return self.bitcast_wave_value(source_bits, target_type)

        remainder = self.binary_operation("%", uint_type, byte_index, source_bytes)
        eight = self.register_constant(8, uint_type)
        bit_offset = self.binary_operation("*", uint_type, remainder, eight)
        shifted = self.binary_operation(">>", uint_type, source_bits, bit_offset)
        mask = self.register_constant((1 << target_layout.bit_width) - 1, uint_type)
        extracted = self.binary_operation("&", uint_type, shifted, mask)
        if not target_layout.signed:
            return self.convert_value_to_type(extracted, target_type)

        left_shift_count = self.register_constant(
            32 - target_layout.bit_width, uint_type
        )
        sign_positioned = self.binary_operation(
            "<<", uint_type, extracted, left_shift_count
        )
        int_type = self.register_primitive_type("int")
        signed_value = self.bitcast_wave_value(sign_positioned, int_type)
        result_id = self.get_id()
        self.emit(
            f"%{result_id} = OpShiftRightArithmetic %{int_type.id} "
            f"%{signed_value.id} %{left_shift_count.id}"
        )
        result = SpirvId(result_id, int_type.type)
        self.value_types[result_id] = int_type
        return self.convert_value_to_type(result, target_type)

    def register_input(
        self,
        name: str,
        type_id: SpirvId,
        location: int,
        binding: int,
        source_node=None,
    ) -> SpirvId:
        """Register an input variable with location decoration."""
        self.validate_user_defined_interface_type(type_id, "Input", name, source_node)
        ptr_type = self.register_pointer_type(type_id, "Input")

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Input")

        self.decorations.append(f"OpDecorate %{id_value} Location {location}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.inputs.append(spirv_id)
        return spirv_id

    def register_output(
        self,
        name: str,
        type_id: SpirvId,
        location: int,
        binding: int,
        source_node=None,
    ) -> SpirvId:
        """Register an output variable with location decoration."""
        self.validate_user_defined_interface_type(type_id, "Output", name, source_node)
        ptr_type = self.register_pointer_type(type_id, "Output")

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpVariable %{ptr_type.id} Output")

        self.decorations.append(f"OpDecorate %{id_value} Location {location}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, ptr_type.type, name)
        self.variable_value_types[id_value] = type_id
        self.outputs.append(spirv_id)
        return spirv_id

    def compute_builtin_info(self, name: str):
        name = self.spirv_builtin_alias(name)
        builtins = {
            "gl_GlobalInvocationID": ("uvec3", "GlobalInvocationId", "Input"),
            "gl_LocalInvocationID": ("uvec3", "LocalInvocationId", "Input"),
            "gl_WorkGroupID": ("uvec3", "WorkgroupId", "Input"),
            "gl_NumWorkGroups": ("uvec3", "NumWorkgroups", "Input"),
            "gl_LocalInvocationIndex": ("uint", "LocalInvocationIndex", "Input"),
            "gl_WorkGroupSize": ("uvec3", "WorkgroupSize", "Constant"),
            "gl_SubgroupSize": ("uint", "SubgroupSize", "Input"),
            "gl_SubgroupInvocationID": (
                "uint",
                "SubgroupLocalInvocationId",
                "Input",
            ),
            **SPIRV_SUBGROUP_BUILTINS,
        }
        return builtins.get(name) or extension_any_stage_builtin_info(name)

    def spirv_output_semantic_alias(self, name: str) -> str:
        aliases = {
            "SV_POSITION": "gl_Position",
            "SV_DEPTH": "gl_FragDepth",
            "SV_COVERAGE": "gl_SampleMask",
            "SV_STENCILREF": "gl_FragStencilRefEXT",
            "SV_SHADINGRATE": "gl_PrimitiveShadingRateEXT",
        }
        return aliases.get(str(name).upper(), name)

    def spirv_interface_semantic_alias(
        self,
        semantic: str,
        execution_model: Optional[str],
        storage_class: str,
    ) -> str:
        upper_name = str(semantic).upper()
        if semantic == "gl_Position":
            return (
                "gl_FragCoord"
                if execution_model == "Fragment" and storage_class == "Input"
                else semantic
            )
        if upper_name == "SV_POSITION":
            return (
                "gl_FragCoord"
                if execution_model == "Fragment" and storage_class == "Input"
                else "gl_Position"
            )
        if upper_name == "SV_COVERAGE":
            return (
                "gl_SampleMaskIn"
                if execution_model == "Fragment" and storage_class == "Input"
                else "gl_SampleMask"
            )
        if upper_name == "SV_BARYCENTRICS":
            return "gl_BaryCoordEXT"
        if upper_name == "SV_SAMPLEINDEX":
            return "gl_SampleID"
        if upper_name == "SV_STENCILREF":
            return "gl_FragStencilRefEXT"
        if upper_name == "SV_DEPTH":
            return "gl_FragDepth"
        shading_rate = shading_rate_semantic_builtin(
            semantic, execution_model, storage_class
        )
        if shading_rate is not None:
            return shading_rate
        return semantic

    def spirv_builtin_cache_name(self, name: str) -> str:
        return self.spirv_builtin_alias(name)
