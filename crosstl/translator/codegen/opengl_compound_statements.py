"""Compound statements in the GLSL generator: boolean and arithmetic compound assignments, narrow updates, if, while, array and match statements."""

from ..ast import (
    ArrayNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
    RangeNode,
    ReferenceType,
    VariableNode,
)
from ..validation import expression_debug_name
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
)
from .for_in_utils import generate_typed_array_for_in
from .glsl_stage_output_values import builtin_output_value
from .glsl_word_views import generate_glsl_word_view_assignment
from .match_utils import (
    generate_ordered_conditional_match,
    generate_switch_match,
    is_switch_lowerable_match,
)
from .opengl_values import (
    OpenGLCompoundAssignmentError,
    OpenGLForInIterableError,
    OpenGLPrivatePointerParameterError,
)


class OpenGLCompoundStatementMixin:
    """Compound statement members of ``GLSLCodeGen``."""

    def generate_glsl_boolean_arithmetic_compound_assignment(
        self,
        node,
        target,
        value,
        operator,
        target_type,
        *,
        statement_context,
    ):
        if self.GLSL_TARGET_DISPLAY_NAME != "OpenGL":
            return None
        binary_operator = (
            self.BOOLEAN_ARITHMETIC_COMPOUND_ASSIGNMENT_BINARY_OPERATORS.get(operator)
        )
        target_info = self.glsl_value_type_info(target_type)
        if (
            binary_operator is None
            or target_info is None
            or target_info["family"] != "bool"
        ):
            return None

        right_type = self.glsl_source_expression_type(value)

        def reject(reason, detail, operation_type=None):
            self.glsl_boolean_compound_assignment_error(
                node,
                target,
                operator,
                target_type,
                right_type,
                reason=reason,
                detail=detail,
                operation_type=operation_type,
            )

        try:
            plan = self.glsl_arithmetic_conversion_plan(
                target_type,
                right_type,
                binary_operator,
                source_node=node,
                fail_closed=True,
            )
        except self.GLSL_ARITHMETIC_CONVERSION_ERROR as error:
            reject(
                error.reason,
                (
                    "source operand promotion has no faithful target form "
                    f"({error.reason.replace('-', ' ')})"
                ),
                error.attempted_common_type,
            )
        if plan is None:
            reject(
                "unresolved-promoted-operand-type",
                (
                    "the right operand must resolve to a scalar or vector arithmetic "
                    "type before source promotion"
                ),
            )

        result_info = self.glsl_value_type_info(plan.result_type)
        if result_info is None or result_info["width"] != target_info["width"]:
            reject(
                "target-shape-unrepresentable",
                (
                    "the promoted result shape cannot be converted back to the "
                    "boolean lvalue shape"
                ),
                plan.result_type,
            )
        if not statement_context:
            reject(
                "statement-context-required",
                (
                    "the promoted lvalue read, ordered right operand, and boolean "
                    "writeback require statement temporaries"
                ),
                plan.result_type,
            )

        prefix, stable_target = self.glsl_stabilize_boolean_compound_lvalue(
            node,
            target,
            operator,
            target_type,
            right_type,
        )
        left = self.generate_glsl_buffer_block_mutation_target(stable_target)
        left_value = self.generate_expression_with_expected(
            stable_target, plan.left_target_type
        )
        right_value = self.generate_expression_with_expected(
            value, plan.right_target_type
        )
        left_name = self.glsl_boolean_compound_temporary(
            "cgl_bool_compound_lhs", plan.left_target_type
        )
        right_name = self.glsl_boolean_compound_temporary(
            "cgl_bool_compound_rhs", plan.right_target_type
        )
        operation = f"({left_name} {binary_operator} {right_name})"
        writeback = self.glsl_bool_conversion_expression(
            operation, result_info, target_info, node
        )
        return "\n".join(
            [
                *prefix,
                f"{plan.left_target_type} {left_name} = {left_value}",
                f"{plan.right_target_type} {right_name} = {right_value}",
                f"{left} = {writeback}",
            ]
        )

    def glsl_narrow_update_value(
        self, target, value, binary_operator, expected_type, source_node
    ):
        if not self.glsl_stable_update_target(target):
            self.glsl_scalar_conversion_error(
                source_node,
                expected_type,
                expected_type,
                "narrow-update-lvalue-side-effects",
            )
        operation = BinaryOpNode(
            target,
            binary_operator,
            value,
            source_location=getattr(source_node, "source_location", None),
        )
        return self.generate_expression_with_expected(operation, expected_type)

    def glsl_boolean_compound_assignment_value(
        self, target, value, binary_operator, expected_type, source_node
    ):
        expected = self.glsl_value_type_info(expected_type)
        if (
            binary_operator is None
            or (
                self.GLSL_TARGET_DISPLAY_NAME == "OpenGL"
                and binary_operator not in {"&", "|", "^"}
            )
            or expected is None
            or expected["family"] != "bool"
        ):
            return None
        if not self.glsl_stable_update_target(target):
            target_name = self.expression_name(target)
            raise self.GLSL_BOOLEAN_COMPOUND_ASSIGNMENT_ERROR(
                f"{self.GLSL_TARGET_DISPLAY_NAME} cannot safely lower boolean "
                "compound assignment "
                f"'{binary_operator}=' for a side-effecting lvalue",
                operator=f"{binary_operator}=",
                target=target_name,
                target_type=expected["source"],
                reason="lvalue-side-effects",
                source_location=getattr(source_node, "source_location", None),
            )
        operation = BinaryOpNode(
            target,
            binary_operator,
            value,
            source_location=getattr(source_node, "source_location", None),
        )
        return self.generate_expression_with_expected(operation, expected_type)

    def glsl_bfloat_integer_compound_assignment_value(
        self, target, value, binary_operator, expected_type, source_node
    ):
        if binary_operator not in {"+", "-", "*", "/"}:
            return None
        value_type = self.glsl_source_expression_type(value)
        target_info = self.glsl_value_type_info(expected_type)
        if (
            self.glsl_bfloat_width(value_type) != 1
            or target_info is None
            or target_info["family"] not in {"int", "uint"}
            or target_info["width"] != 1
        ):
            return None
        reason = None
        if not self.glsl_stable_update_target(target):
            reason = "lvalue-side-effects"
        elif not self.glsl_side_effect_free_expression(value):
            reason = "rhs-may-change-assignment-target"
        if reason is not None:
            raise OpenGLCompoundAssignmentError(
                "OpenGL bfloat compound assignment requires a stable integer "
                "target and a right operand without observable side effects; "
                "materialize side effects before the assignment",
                operator=f"{binary_operator}=",
                target=expression_debug_name(target),
                target_type=self.type_name_string(expected_type),
                value_type=self.type_name_string(value_type),
                common_type="bfloat",
                reason=reason,
                source_location=getattr(source_node, "source_location", None),
            )
        # The integer is converted before arithmetic, and the bfloat result is
        # rounded before conversion back to the destination's integer type.
        operation = BinaryOpNode(
            FunctionCallNode("bfloat", [target]),
            binary_operator,
            value,
            source_location=getattr(source_node, "source_location", None),
        )
        return self.generate_expression_with_expected(operation, expected_type)

    def glsl_converted_compound_assignment_value(
        self, target, value, binary_operator, expected_type, source_node
    ):
        if binary_operator is None or expected_type is None:
            return None
        value_type = self.glsl_source_expression_type(value)
        conversion_types = self.glsl_binary_operand_conversion_types(
            expected_type,
            value_type,
            binary_operator,
            source_node=source_node,
            fail_closed=True,
        )
        if conversion_types is None:
            return None
        target_operand_type, value_operand_type = conversion_types
        signed_remainder = (
            binary_operator == "%"
            and self.glsl_signed_remainder_type(expected_type, value_type) is not None
        )
        if (
            not signed_remainder
            and self.map_type(target_operand_type) == self.map_type(expected_type)
            and self.map_type(value_operand_type) == self.map_type(value_type)
        ):
            return None

        common_type = self.glsl_common_arithmetic_type(
            expected_type, value_type, binary_operator
        )
        if (
            signed_remainder
            and not self.glsl_side_effect_free_expression(value)
            and not isinstance(target, (IdentifierNode, VariableNode))
        ):
            raise OpenGLCompoundAssignmentError(
                "OpenGL cannot safely lower signed remainder assignment when the "
                "right operand may change the assignment target",
                operator="%=",
                target=expression_debug_name(target),
                target_type=self.type_name_string(expected_type),
                value_type=self.type_name_string(value_type),
                common_type=self.type_name_string(common_type),
                reason="rhs-may-change-assignment-target",
                source_location=getattr(source_node, "source_location", None),
            )
        if not self.glsl_stable_update_target(target):
            target_name = expression_debug_name(target)
            raise OpenGLCompoundAssignmentError(
                "OpenGL cannot safely lower mixed-type compound assignment "
                f"'{binary_operator}=' for side-effecting lvalue '{target_name}'",
                operator=f"{binary_operator}=",
                target=target_name,
                target_type=self.type_name_string(expected_type),
                value_type=self.type_name_string(value_type),
                common_type=self.type_name_string(common_type),
                reason="lvalue-side-effects",
                source_location=getattr(source_node, "source_location", None),
            )

        operation = BinaryOpNode(
            target,
            binary_operator,
            value,
            source_location=getattr(source_node, "source_location", None),
        )
        return self.generate_expression_with_expected(operation, expected_type)

    def generate_glsl_narrow_unary_update(self, expression, operator):
        expected_type = self.glsl_source_expression_type(expression.operand)
        if (
            self.glsl_narrow_integer_contract(expected_type) is None
            and self.glsl_narrow_float_width(expected_type) is None
        ):
            return None
        is_postfix = bool(
            getattr(expression, "is_postfix", False)
        ) or expression.op in {
            "POST_INCREMENT",
            "POST_DECREMENT",
        }
        if is_postfix and self.current_discarded_expression is not expression:
            narrow = self.glsl_narrow_integer_contract(expected_type)
            owner = expression.operand
            while isinstance(owner, MemberAccessNode):
                owner = owner.object_expr
            if (
                narrow is None
                or narrow["bits"] != 8
                or not isinstance(owner, (IdentifierNode, VariableNode))
            ):
                self.glsl_scalar_conversion_error(
                    expression,
                    expected_type,
                    expected_type,
                    "narrow-postfix-result",
                )
        one = LiteralNode(1, PrimitiveType("int"))
        value = self.glsl_narrow_update_value(
            expression.operand,
            one,
            "+" if operator == "++" else "-",
            expected_type,
            expression,
        )
        target = self.generate_glsl_buffer_block_mutation_target(expression.operand)
        if is_postfix and self.current_discarded_expression is not expression:
            inverse = "-" if operator == "++" else "+"
            one = "1u" if narrow["mapped"].startswith("u") else "1"
            previous = self.glsl_apply_narrow_integer_contract(
                f"({target} {inverse} {one})", expected_type
            )
            return f"(({target} = {value}), {previous})"
        return f"({target} = {value})"

    def generate_assignment(self, node, is_main=False, *, statement_context=False):
        rebased_parameter = self.glsl_private_pointer_base_rebinding_parameters.get(
            id(node)
        )
        if rebased_parameter is not None:
            if not statement_context:
                display_name = self.glsl_private_pointer_display_name(
                    self.current_glsl_private_pointer_function_name
                )
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL cannot preserve the value of a private pointer "
                    "parameter rebase",
                    function_name=display_name,
                    parameter_name=rebased_parameter,
                    reason="pointer-parameter-rebinding",
                    source_location=getattr(node, "source_location", None),
                )
            operator = self.map_operator(
                getattr(node, "operator", getattr(node, "op", "="))
            )
            if operator not in {"+=", "-="}:
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL cannot emit an unsupported private pointer base assignment",
                    function_name=self.glsl_private_pointer_display_name(
                        self.current_glsl_private_pointer_function_name
                    ),
                    parameter_name=rebased_parameter,
                    reason="pointer-parameter-rebinding",
                    source_location=getattr(node, "source_location", None),
                )
            base_name = self.current_glsl_private_pointer_base_names.get(
                rebased_parameter
            )
            if base_name is None:
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL cannot emit a private pointer rebase without a "
                    "synthetic array base",
                    function_name=self.glsl_private_pointer_display_name(
                        self.current_glsl_private_pointer_function_name
                    ),
                    parameter_name=rebased_parameter,
                    reason="pointer-parameter-rebinding",
                    source_location=getattr(node, "source_location", None),
                )
            value = getattr(node, "value", getattr(node, "right", None))
            return f"{base_name} {operator} int({self.generate_expression(value)})"
        if id(node) in self.glsl_private_pointer_noop_rebinding_ids:
            if not statement_context:
                target = getattr(node, "target", getattr(node, "left", None))
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL can only eliminate a proven-zero private pointer "
                    "rebase when its result is discarded",
                    function_name=self.current_glsl_private_pointer_function_name,
                    parameter_name=self.expression_name(target),
                    reason="pointer-parameter-rebinding",
                    source_location=getattr(node, "source_location", None),
                )
            return ""
        left_node = getattr(node, "target", getattr(node, "left", None))
        right_node = getattr(node, "value", getattr(node, "right", None))
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "=")))
        emulated_matrix_assignment = (
            self.generate_glsl_emulated_matrix_compound_assignment(
                left_node, right_node, op
            )
        )
        if emulated_matrix_assignment is not None:
            return emulated_matrix_assignment
        structured_assignment = self.generate_glsl_hlsl_structured_assignment(
            left_node, right_node, op, statement_context=statement_context
        )
        if structured_assignment is not None:
            return structured_assignment
        union = self.union_storage.assignment(
            left_node, right_node, op, statement_context
        )
        if union is not None:
            return union
        storage_to_workgroup_copy = self.glsl_storage_to_workgroup_byte_array_copy(
            left_node,
            right_node,
            op,
            statement_context=statement_context,
        )
        if storage_to_workgroup_copy is not None:
            return storage_to_workgroup_copy
        self.glsl_storage_struct_reinterpret_mutation(left_node)
        destination_type = self.glsl_source_expression_type(left_node)
        storage_aggregate = self.glsl_storage_struct_reinterpret_materialization(
            right_node,
            destination_type,
        )
        if storage_aggregate is not None:
            if op != "=":
                contract = self.glsl_storage_struct_reinterpret_contract(
                    right_node,
                    destination_type,
                )
                self.glsl_storage_struct_reinterpret_error(
                    contract["reinterpretation"],
                    source_type=contract["source_type"],
                    target_type=contract["target_type"],
                    reason="storage-aggregate-assignment-operator-unsupported",
                    detail=f"assignment operator '{op}' is not supported",
                    alignment=contract["member_layout"].byte_width,
                )
            left = self.generate_glsl_buffer_block_mutation_target(left_node)
            return f"{left} = {storage_aggregate}"
        self.validate_glsl_private_pointer_byte_view_mutation(left_node)
        word_store = generate_glsl_word_view_assignment(
            self, node, left_node, right_node, op
        )
        if word_store is not None:
            return word_store
        self.validate_glsl_storage_pointer_mutation_target(
            left_node, "write" if op == "=" else "read_write"
        )
        self.validate_glsl_buffer_block_assignment_target(left_node, op)
        storage_assignment = self.float_atomic_storage.assignment(
            node, left_node, right_node, op, statement_context
        )
        if storage_assignment is not None:
            return storage_assignment
        expected_type = self.glsl_tessellation_factor_assignment_expected_type(
            left_node
        )
        if expected_type is not None:
            self.validate_glsl_tessellation_factor_assignment_value(
                left_node, right_node
            )
        else:
            expected_type = destination_type
        binary_operator = self.COMPOUND_ASSIGNMENT_BINARY_OPERATORS.get(op)
        boolean_arithmetic_assignment = (
            self.generate_glsl_boolean_arithmetic_compound_assignment(
                node,
                left_node,
                right_node,
                op,
                expected_type,
                statement_context=statement_context,
            )
        )
        if boolean_arithmetic_assignment is not None:
            return boolean_arithmetic_assignment
        left = self.generate_glsl_buffer_block_mutation_target(left_node)
        bfloat_integer_assignment = self.glsl_bfloat_integer_compound_assignment_value(
            left_node, right_node, binary_operator, expected_type, node
        )
        if bfloat_integer_assignment is not None:
            return f"{left} = {bfloat_integer_assignment}"
        if binary_operator is not None and (
            self.glsl_narrow_integer_contract(expected_type) is not None
            or self.glsl_narrow_float_width(expected_type) is not None
        ):
            right = self.glsl_narrow_update_value(
                left_node,
                right_node,
                binary_operator,
                expected_type,
                node,
            )
            return f"{left} = {right}"
        float_remainder = self.glsl_float_remainder_assignment_value(
            left_node, right_node, op, expected_type
        )
        if float_remainder is not None:
            return f"{left} = {float_remainder}"
        boolean_assignment = self.glsl_boolean_compound_assignment_value(
            left_node,
            right_node,
            binary_operator,
            expected_type,
            node,
        )
        if boolean_assignment is not None:
            return f"{left} = {boolean_assignment}"
        complex_assignment = self.glsl_complex64_compound_assignment_expression(
            node,
            left,
            right_node,
            op,
            expected_type,
        )
        if complex_assignment is not None:
            return complex_assignment
        if binary_operator in {"<<", ">>"}:
            operand_types = self.glsl_binary_operand_conversion_types(
                expected_type,
                self.glsl_source_expression_type(right_node),
                binary_operator,
                source_node=node,
                fail_closed=True,
            )
            if operand_types is not None and self.map_type(
                operand_types[0]
            ) == self.map_type(expected_type):
                right = self.generate_expression_with_expected(
                    right_node, operand_types[1]
                )
                return f"{left} {op} {right}"
        converted_assignment = self.glsl_converted_compound_assignment_value(
            left_node,
            right_node,
            binary_operator,
            expected_type,
            node,
        )
        if converted_assignment is not None:
            return f"{left} = {converted_assignment}"
        right = self.generate_expression_with_expected(
            right_node, expected_type if op == "=" else None
        )
        precise_compound = self.glsl_precise_compound_store(
            left_node, left, op, right, destination_type
        )
        if precise_compound is not None:
            return precise_compound
        right = self.glsl_precise_stored_value(
            left_node,
            right_node,
            right,
            (
                expected_type
                if op == "="
                else self.glsl_source_expression_type(right_node)
            ),
        )
        right = builtin_output_value(
            self, left, right, self.glsl_source_expression_type(right_node)
        )
        return f"{left} {op} {right}"

    def generate_if(self, node, indent, is_main=False):
        indent_str = "    " * indent
        static_branch_known, static_body = getattr(
            node, "_glsl_checked_static_if", (False, None)
        )
        if not static_branch_known:
            static_branch_known, static_body = self.glsl_static_if_body(node)
        if static_branch_known:
            if not self.statement_body_has_statements(static_body):
                return ""
            code = f"{indent_str}{{\n"
            code += self.generate_scoped_statement_body(static_body, indent + 1)
            code += f"{indent_str}}}\n"
            return code

        if id(node) in self.glsl_software_subgroup_masked_if_plans:
            return self.generate_glsl_software_subgroup_masked_if(node, indent)

        if id(node) not in self.glsl_software_subgroup_uniform_if_node_ids:
            plan = self.glsl_software_subgroup_on_demand_masked_if_plan(node)
            if plan is not None:
                self.glsl_software_subgroup_masked_if_plans[id(node)] = plan
                return self.generate_glsl_software_subgroup_masked_if(node, indent)
            self.reject_glsl_software_subgroup_control_flow(node)
        condition_node = (
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        condition = self.glsl_precise_condition(
            condition_node, self.generate_glsl_boolean_context(condition_node)
        )
        code = f"{indent_str}if ({condition}) {{\n"

        if_body = node.if_body
        code += self.generate_scoped_statement_body(if_body, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                condition = self.glsl_precise_condition(
                    else_if_condition,
                    self.generate_glsl_boolean_context(else_if_condition),
                )
                code += f" else if ({condition}) {{\n"
                code += self.generate_scoped_statement_body(else_if_body, indent + 1)
                code += f"{indent_str}}}"

        if hasattr(node, "else_body") and node.else_body:
            code += " else {\n"
            else_body = node.else_body
            code += self.generate_scoped_statement_body(else_body, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def glsl_static_if_body(self, node):
        conditions = [
            getattr(node, "condition", getattr(node, "if_condition", None)),
            *(getattr(node, "else_if_conditions", []) or []),
        ]
        bodies = [node.if_body, *(getattr(node, "else_if_bodies", []) or [])]
        fixed = getattr(self, "glsl_software_subgroup_constant_guards", {}).get(
            id(node)
        )
        if fixed is not None:
            return True, node.if_body if fixed else getattr(node, "else_body", None)
        for condition, body in zip(conditions, bodies):
            literal = evaluate_literal_int_expression(
                condition,
                self.glsl_active_literal_int_constants(),
            )
            if literal is None:
                return False, None
            if literal:
                return True, body
        return True, getattr(node, "else_body", None)

    def generate_glsl_software_subgroup_strided_for(self, node, indent, plan):
        initializer = node.init
        self.generate_for_initializer(initializer)
        loop_name = self.glsl_local_identifier_name(initializer.name)
        loop_type = plan["loopType"]
        base_name = self.glsl_synthetic_local_identifier("crossglSoftwareSubgroupBase")
        active_name = self.glsl_synthetic_local_identifier(
            "crossglSoftwareSubgroupLoopActive"
        )
        initial_value = self.generate_expression_with_expected(
            initializer.initial_value,
            getattr(initializer, "var_type", None),
        )
        bound = self.generate_expression_with_expected(
            node.condition.right,
            getattr(initializer, "var_type", None),
        )
        step = plan["step"]
        zero = "0u" if loop_type == "uint" else "0"
        step_literal = f"{step}u" if loop_type == "uint" else str(step)
        indent_str = "    " * indent
        body_indent = "    " * (indent + 1)
        code = (
            f"{indent_str}for ({loop_type} {base_name} = {zero}; "
            f"({base_name} < {bound}); {base_name} += {step_literal}) {{\n"
            f"{body_indent}{loop_type} {loop_name} = "
            f"({base_name} + {initial_value});\n"
        )

        statements = self.glsl_software_subgroup_statement_list(node.body)
        operation_if_index = plan["operationIfIndex"]
        prefix = list(statements[:operation_if_index])
        suffix = list(statements[operation_if_index + 1 :])
        hoisted = set(plan["hoistedPrefixDeclarationIndices"])
        for index in sorted(hoisted):
            code += self.generate_statement(prefix[index], indent + 1)
        guarded_prefix = [
            statement for index, statement in enumerate(prefix) if index not in hoisted
        ]

        condition = self.generate_glsl_boolean_context(node.condition)
        code += f"{body_indent}bool {active_name} = {condition};\n"
        if guarded_prefix:
            code += f"{body_indent}if ({active_name}) {{\n"
            code += self.generate_scoped_statement_body(guarded_prefix, indent + 2)
            code += f"{body_indent}}}\n"

        operation_if = statements[operation_if_index]
        code += self.generate_glsl_software_subgroup_masked_if(
            operation_if,
            indent + 1,
            additional_condition=active_name,
        )
        if suffix:
            code += f"{body_indent}if ({active_name}) {{\n"
            code += self.generate_scoped_statement_body(suffix, indent + 2)
            code += f"{body_indent}}}\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_for(self, node, indent, is_main=False):
        if getattr(node, "_glsl_checked_zero_trip", False):
            return ""
        if id(node) in getattr(
            self, "glsl_software_subgroup_zero_trip_for_node_ids", set()
        ):
            return ""
        indent_str = "    " * indent
        strided_plan = getattr(
            self,
            "glsl_software_subgroup_strided_for_plans",
            {},
        ).get(id(node))
        if (
            strided_plan is None
            and id(node) not in self.glsl_software_subgroup_uniform_for_node_ids
            and not self.glsl_software_subgroup_uniform_for(node)
        ):
            self.reject_glsl_software_subgroup_control_flow(node)
        initializer_node = getattr(node, "init", None)
        initializer_declared_names = set()
        if isinstance(initializer_node, (VariableNode, ArrayNode)):
            initializer_declared_names.add(getattr(initializer_node, "name", None))
        previous_local_variable_types = dict(self.local_variable_types)
        previous_local_variable_source_types = dict(self.local_variable_source_types)
        previous_workgroup_pointer_aliases = dict(
            self.current_glsl_workgroup_pointer_aliases
        )
        previous_storage_pointer_aliases = dict(
            self.current_glsl_storage_pointer_aliases
        )
        previous_scalar_array_views = dict(self.current_glsl_scalar_array_views)
        previous_compile_time_constant_state = (
            self.glsl_compile_time_constant_scope_state()
        )
        post_initializer_workgroup_aliases = None
        post_initializer_storage_aliases = None

        try:
            if strided_plan is not None:
                return self.generate_glsl_software_subgroup_strided_for(
                    node,
                    indent,
                    strided_plan,
                )
            init = self.generate_for_initializer(getattr(node, "init", None))
            post_initializer_workgroup_aliases = dict(
                self.current_glsl_workgroup_pointer_aliases
            )
            post_initializer_storage_aliases = dict(
                self.current_glsl_storage_pointer_aliases
            )
            condition = (
                self.glsl_precise_condition(
                    node.condition,
                    self.generate_glsl_boolean_context(node.condition),
                )
                if getattr(node, "condition", None)
                else ""
            )
            if isinstance(getattr(node, "update", None), list):
                update = ", ".join(
                    self.generate_discarded_expression(expression).strip().rstrip(";")
                    for expression in node.update
                )
            else:
                update = (
                    self.generate_discarded_expression(node.update)
                    if getattr(node, "update", None)
                    else ""
                )

            code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"

            body = node.body
            code += self.generate_scoped_statement_body(body, indent + 1)

            code += f"{indent_str}}}\n"

            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.local_variable_source_types = previous_local_variable_source_types
            restored_workgroup_aliases = dict(previous_workgroup_pointer_aliases)
            restored_storage_aliases = dict(previous_storage_pointer_aliases)
            if post_initializer_workgroup_aliases is not None:
                for name in previous_workgroup_pointer_aliases:
                    if (
                        name not in initializer_declared_names
                        and name in post_initializer_workgroup_aliases
                    ):
                        restored_workgroup_aliases[name] = (
                            post_initializer_workgroup_aliases[name]
                        )
            if post_initializer_storage_aliases is not None:
                for name in previous_storage_pointer_aliases:
                    if (
                        name not in initializer_declared_names
                        and name in post_initializer_storage_aliases
                    ):
                        restored_storage_aliases[name] = (
                            post_initializer_storage_aliases[name]
                        )
            self.current_glsl_workgroup_pointer_aliases = restored_workgroup_aliases
            self.current_glsl_storage_pointer_aliases = restored_storage_aliases
            self.current_glsl_scalar_array_views = previous_scalar_array_views
            self.restore_glsl_compile_time_constant_scope_state(
                previous_compile_time_constant_state
            )

    def generate_for_in(self, node, indent):
        self.reject_glsl_software_subgroup_control_flow(node)
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)
        previous_local_variable_source_types = dict(self.local_variable_source_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        previous_scalar_array_views = dict(self.current_glsl_scalar_array_views)
        missing_pattern_alias = object()
        previous_resource_pattern_alias = self.current_resource_aliases.get(
            pattern, missing_pattern_alias
        )
        previous_workgroup_pattern_alias = (
            self.current_glsl_workgroup_pointer_aliases.get(
                pattern, missing_pattern_alias
            )
        )
        previous_storage_pattern_alias = self.current_glsl_storage_pointer_aliases.get(
            pattern, missing_pattern_alias
        )
        previous_stage_pattern_alias = self.current_stage_parameter_aliases.get(
            pattern, missing_pattern_alias
        )
        previous_stage_input_pattern_alias = self.current_stage_inputs.get(
            pattern, missing_pattern_alias
        )
        previous_stage_output_pattern_alias = self.current_stage_outputs.get(
            pattern, missing_pattern_alias
        )
        previous_compile_time_constant_state = (
            self.glsl_compile_time_constant_scope_state()
        )

        try:
            if getattr(node, "binding_type", None) is not None:
                self.glsl_for_in_fixed_array_type(node, pattern, iterable_node)
                return generate_typed_array_for_in(self, node, indent, target="opengl")
            binding_qualifier = self.validate_glsl_for_in_binding(node, pattern)
            pattern_view = self.current_glsl_scalar_array_view_plans.get(id(node))

            def begin_pattern_scope():
                # Resolve and render the iterable in the outer lexical
                # environment first.  This includes compile-time constants
                # used by fixed-array extents.  Only once the pattern binding
                # enters scope may it shadow every same-named namespace.
                self.current_identifier_aliases.pop(pattern, None)
                self.current_resource_aliases.pop(pattern, None)
                self.current_glsl_workgroup_pointer_aliases.pop(pattern, None)
                self.current_glsl_storage_pointer_aliases.pop(pattern, None)
                self.current_stage_parameter_aliases.pop(pattern, None)
                self.current_stage_inputs.pop(pattern, None)
                self.current_stage_outputs.pop(pattern, None)
                self.current_compile_time_int_constants.pop(pattern, None)
                self.current_compile_time_int_vector_constants.pop(pattern, None)
                self.current_compile_time_int_vector_array_constants.pop(pattern, None)
                return self.glsl_local_identifier_name(pattern)

            iterable_dependencies = self.glsl_interval_expression_keys(iterable_node)
            pattern_captures_iterable = bool(pattern) and (
                self.glsl_interval_key_sets_overlap(
                    {pattern},
                    iterable_dependencies,
                )
            )
            pattern_initializer = None
            pattern_requires_body_declaration = False

            if isinstance(iterable_node, RangeNode):
                start = self.generate_expression(iterable_node.start)
                end = self.generate_expression(iterable_node.end)
                pattern_name = begin_pattern_scope()
                comparator = "<=" if iterable_node.inclusive else "<"
                pattern_type = "int"
                if pattern_view is None and not pattern_captures_iterable:
                    code = (
                        f"{indent_str}for (int {pattern_name} = {start}; "
                        f"{pattern_name} {comparator} {end}; ++{pattern_name}) {{\n"
                    )
                else:
                    index_name = self.glsl_synthetic_local_identifier(
                        f"{pattern}_crossgl_index"
                    )
                    self.local_variable_types[index_name] = "int"
                    self.local_variable_source_types[index_name] = "int"
                    code = (
                        f"{indent_str}for (int {index_name} = {start}; "
                        f"{index_name} {comparator} {end}; ++{index_name}) {{\n"
                    )
                    pattern_initializer = index_name
                    pattern_requires_body_declaration = pattern_view is None
            elif self.is_scalar_integer_type(
                self.glsl_source_expression_type(iterable_node)
            ):
                iterable = self.generate_expression(iterable_node)
                pattern_name = begin_pattern_scope()
                pattern_type = "int"
                if pattern_view is None and not pattern_captures_iterable:
                    code = (
                        f"{indent_str}for (int {pattern_name} = 0; "
                        f"{pattern_name} < {iterable}; ++{pattern_name}) {{\n"
                    )
                else:
                    index_name = self.glsl_synthetic_local_identifier(
                        f"{pattern}_crossgl_index"
                    )
                    self.local_variable_types[index_name] = "int"
                    self.local_variable_source_types[index_name] = "int"
                    code = (
                        f"{indent_str}for (int {index_name} = 0; "
                        f"{index_name} < {iterable}; ++{index_name}) {{\n"
                    )
                    pattern_initializer = index_name
                    pattern_requires_body_declaration = pattern_view is None
            else:
                iterable_type = self.glsl_for_in_fixed_array_type(
                    node, pattern, iterable_node
                )
                extent, element_type = self.glsl_for_in_array_shape(
                    node, pattern, iterable_type
                )
                if extent == 0:
                    if not self.glsl_side_effect_free_expression(iterable_node):
                        self.raise_glsl_for_in_iterable_error(
                            node,
                            pattern,
                            iterable_type,
                            reason="side-effecting-empty-array",
                        )
                    return ""

                mapped_iterable_type = self.map_type(iterable_type)
                mapped_element_type = self.map_type(element_type)
                if self.type_contains_opaque_resource(element_type):
                    self.raise_glsl_for_in_iterable_error(
                        node,
                        pattern,
                        iterable_type,
                        binding_type=element_type,
                        reason="unsupported-element-type",
                    )

                container_name = self.glsl_synthetic_local_identifier(
                    f"{pattern}_crossgl_iterable"
                )
                index_name = self.glsl_synthetic_local_identifier(
                    f"{pattern}_crossgl_index"
                )
                iterable = self.generate_expression_with_expected(
                    iterable_node, iterable_type
                )
                pattern_name = begin_pattern_scope()
                container_declaration = format_c_style_array_declaration(
                    mapped_iterable_type, container_name
                )
                self.local_variable_types[container_name] = iterable_type
                self.local_variable_source_types[container_name] = iterable_type
                self.local_variable_types[index_name] = "int"
                self.local_variable_source_types[index_name] = "int"
                pattern_type = element_type
                pattern_initializer = f"{container_name}[{index_name}]"
                code = (
                    f"{indent_str}{container_declaration} = {iterable};\n"
                    f"{indent_str}for (int {index_name} = 0; "
                    f"{index_name} < {extent}; ++{index_name}) {{\n"
                )
                if pattern_view is None:
                    pattern_declaration = format_c_style_array_declaration(
                        mapped_element_type, pattern_name
                    )
                    pattern_declaration = f"{binding_qualifier}{pattern_declaration}"
                    code += (
                        f"{indent_str}    {pattern_declaration} = "
                        f"{pattern_initializer};\n"
                    )

            # A for-in pattern is a declaration whose lexical lifetime begins
            # inside the loop body.  It must shadow an active same-named outer
            # scalar view even when the pattern itself is not address-taken.
            self.current_glsl_scalar_array_views.pop(pattern, None)
            self.local_variable_types[pattern] = pattern_type
            self.local_variable_source_types[pattern] = pattern_type
            if pattern_view is not None:
                if self.map_type(pattern_type) != self.map_type(
                    pattern_view["element_type"]
                ):
                    raise OpenGLPrivatePointerParameterError(
                        "OpenGL for-in scalar address view changes element type for "
                        f"'{pattern}'",
                        function_name=getattr(
                            self, "current_glsl_source_function_name", None
                        ),
                        parameter_name=pattern,
                        reason="scalar-array-view-element-type-mismatch",
                        source_location=getattr(node, "source_location", None),
                    )
                self.current_glsl_scalar_array_views[pattern] = pattern_view
                mapped_pattern_type = self.map_type(pattern_type)
                declaration = (
                    f"{binding_qualifier}{mapped_pattern_type} {pattern_name}[1]"
                )
                code += (
                    f"{indent_str}    {declaration} = {mapped_pattern_type}[1]("
                    f"{pattern_initializer});\n"
                )
            elif pattern_requires_body_declaration:
                declaration = format_c_style_array_declaration(
                    self.map_type(pattern_type),
                    pattern_name,
                )
                code += (
                    f"{indent_str}    {binding_qualifier}{declaration} = "
                    f"{pattern_initializer};\n"
                )

            code += self.generate_scoped_statement_body(
                getattr(node, "body", []), indent + 1
            )
            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.local_variable_source_types = previous_local_variable_source_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_glsl_scalar_array_views = previous_scalar_array_views
            for aliases, previous in (
                (self.current_resource_aliases, previous_resource_pattern_alias),
                (
                    self.current_glsl_workgroup_pointer_aliases,
                    previous_workgroup_pattern_alias,
                ),
                (
                    self.current_glsl_storage_pointer_aliases,
                    previous_storage_pattern_alias,
                ),
                (self.current_stage_parameter_aliases, previous_stage_pattern_alias),
                (self.current_stage_inputs, previous_stage_input_pattern_alias),
                (self.current_stage_outputs, previous_stage_output_pattern_alias),
            ):
                if previous is missing_pattern_alias:
                    aliases.pop(pattern, None)
                else:
                    aliases[pattern] = previous
            self.restore_glsl_compile_time_constant_scope_state(
                previous_compile_time_constant_state
            )

    def validate_glsl_for_in_binding(self, node, pattern):
        binding_type = getattr(
            node, "binding_type", getattr(node, "pattern_type", None)
        )
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "binding_qualifiers", []) or []
        }
        mutable_reference = bool(
            getattr(node, "is_mutable_reference", False)
            or getattr(node, "mutable_reference", False)
            or (
                isinstance(binding_type, ReferenceType)
                and getattr(binding_type, "is_mutable", False)
            )
            or (
                not isinstance(binding_type, ReferenceType)
                and self.type_name_string(binding_type or "").rstrip().endswith("&")
                and "const" not in qualifiers
                and "readonly" not in qualifiers
            )
        )
        if not mutable_reference:
            immutable_reference = isinstance(
                binding_type, ReferenceType
            ) and not getattr(binding_type, "is_mutable", False)
            return (
                "const "
                if immutable_reference
                or qualifiers
                & {
                    "const",
                    "constant",
                    "readonly",
                }
                else ""
            )
        self.raise_glsl_for_in_iterable_error(
            node,
            pattern,
            self.glsl_source_expression_type(getattr(node, "iterable", None)),
            binding_type=self.type_name_string(binding_type),
            reason="mutable-reference-binding",
        )

    def glsl_for_in_fixed_array_type(self, node, pattern, iterable_node):
        iterable_type = self.type_name_string(
            self.glsl_source_expression_type(iterable_node)
        )
        if not iterable_type:
            self.raise_glsl_for_in_iterable_error(
                node, pattern, None, reason="unresolved-iterable-type"
            )
        if self.glsl_outer_array_type(iterable_type) is None:
            self.raise_glsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="not-fixed-array"
            )
        return iterable_type

    def glsl_for_in_array_shape(self, node, pattern, iterable_type):
        extent_text, element_type = self.glsl_outer_array_type(iterable_type)
        constants = {
            **self.literal_int_constants,
            **self.current_compile_time_int_constants,
        }
        extent = evaluate_literal_int_expression(extent_text, constants)
        if extent is None:
            self.raise_glsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="unknown-extent"
            )
        if extent < 0:
            self.raise_glsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="invalid-extent"
            )
        return extent, element_type

    def raise_glsl_for_in_iterable_error(
        self,
        node,
        pattern,
        iterable_type,
        *,
        binding_type=None,
        reason,
    ):
        descriptions = {
            "binding-address-view-unsupported": (
                "requires an addressable loop binding that cannot retain its source identity"
            ),
            "reference-helper-transport-unsupported": (
                "passes a reference binding to a helper whose parameter copies cannot preserve aliasing"
            ),
            "reference-container-shadowed": (
                "has a reference whose binding or container is shadowed in the loop body"
            ),
            "reference-element-type-mismatch": (
                "requires a reference conversion that cannot retain the source element"
            ),
            "unsupported-container-identity": (
                "requires an array container identity that cannot be captured"
            ),
            "unresolved-selector-type": (
                "has an array selector without a resolved integer type"
            ),
            "invalid-extent": "has a negative fixed-array extent",
            "mutable-reference-binding": (
                "requires mutable reference binding, which GLSL cannot preserve"
            ),
            "not-fixed-array": "is not a fixed array",
            "side-effecting-empty-array": (
                "is an empty array whose container expression has side effects"
            ),
            "unknown-extent": "does not have a compile-time fixed-array extent",
            "unresolved-iterable-type": "has an unresolved iterable type",
            "unsupported-element-type": "contains an unsupported opaque element type",
        }
        detail = descriptions.get(reason, "has no faithful GLSL iterable contract")
        type_label = iterable_type or "<unresolved>"
        raise OpenGLForInIterableError(
            f"OpenGL for-in binding '{pattern}' over '{type_label}' {detail}",
            pattern=pattern,
            iterable_type=iterable_type,
            binding_type=binding_type,
            reason=reason,
            source_location=getattr(node, "source_location", None)
            or getattr(getattr(node, "iterable", None), "source_location", None),
        )

    def generate_while(self, node, indent):
        self.reject_glsl_software_subgroup_control_flow(node)
        indent_str = "    " * indent
        condition_node = getattr(node, "condition", None)
        condition = self.glsl_precise_condition(
            condition_node, self.generate_glsl_boolean_context(condition_node)
        )

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        self.reject_glsl_software_subgroup_control_flow(node)
        indent_str = "    " * indent
        condition_node = getattr(node, "condition", None)
        condition = self.glsl_precise_condition(
            condition_node, self.generate_glsl_boolean_context(condition_node)
        )

        code = f"{indent_str}do {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}} while ({condition});\n"
        return code

    def generate_loop(self, node, indent):
        self.reject_glsl_software_subgroup_control_flow(node)
        indent_str = "    " * indent

        code = f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_switch(self, node, indent):
        self.reject_glsl_software_subgroup_control_flow(node)
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))

        code = f"{indent_str}switch ({expression}) {{\n"
        for case in getattr(node, "cases", []) or []:
            value = getattr(case, "value", None)
            if value is None:
                label = "default"
            else:
                label = f"case {self.generate_expression(value)}"
            code += self.generate_switch_case(
                label, getattr(case, "statements", []), indent + 1
            )

        default_case = getattr(node, "default_case", None)
        if default_case is not None:
            code += self.generate_switch_case(
                "default",
                getattr(default_case, "statements", default_case),
                indent + 1,
            )

        code += f"{indent_str}}}\n"
        return code

    def generate_match(self, node, indent):
        self.reject_glsl_software_subgroup_control_flow(node)
        expression_type = self.map_type(
            self.expression_result_type(getattr(node, "expression", None))
        )
        if is_switch_lowerable_match(node) and expression_type != "bool":
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "GLSL")
