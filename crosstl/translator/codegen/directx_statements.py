"""Statement bodies in the HLSL generator: built-in options, assignments, resource none values, conditions, iterables, while, switch and match statements, and termination."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    IdentifierNode,
    IfNode,
    RangeNode,
    ReferenceType,
    ReturnNode,
    VariableNode,
)
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .directx_values import (
    DirectXCooperativeMatrixUnsupportedError,
    DirectXForInIterableError,
    DirectXPrivatePointerParameterError,
    _directx_cooperative_matrix_fragment_contract,
)
from .enum_utils import generate_enum_constructor_call
from .for_in_utils import generate_typed_array_for_in
from .hlsl_array_list_assignment import generate_hlsl_array_list_assignment
from .hlsl_atomic_conditions import (
    condition_has_hlsl_atomic,
    generate_hlsl_atomic_condition_do_while,
    generate_hlsl_atomic_condition_if,
    generate_hlsl_atomic_condition_switch,
    generate_hlsl_atomic_condition_while,
    hlsl_atomic_condition_loop_guard,
)
from .hlsl_struct_list_assignment import generate_hlsl_struct_list_assignment
from .hlsl_word_views import generate_hlsl_word_view_assignment
from .match_utils import (
    generate_ordered_conditional_match,
    generate_switch_match,
    is_switch_lowerable_match,
)
from .matrix_compound_assignment import matrix_product_assignment


class DirectXStatementMixin:
    """Statement members of ``HLSLCodeGen``."""

    def generate_assignment(self, node, *, statement_context=False):
        if hasattr(node, "target") and hasattr(node, "value"):
            target = node.target
            value = node.value
            op = getattr(node, "operator", "=")
        else:
            target = node.left
            value = node.right
            op = getattr(node, "operator", "=")
        self.hlsl_tall_matrix_rows.mark_chain(target)

        # HLSL *= on matrices is component-wise; CrossGL's is the product.
        matrix_product = matrix_product_assignment(
            target,
            value,
            op,
            type_of=self.expression_result_type,
            target_type_of=self.hlsl_assignment_target_type,
            is_matrix=self.hlsl_numeric_matrix_type,
            is_vector=self.is_vector_value_type,
            backend="directx",
        )
        if matrix_product is not None:
            return self.generate_assignment(
                matrix_product, statement_context=statement_context
            )

        private_rebase = self.generate_hlsl_private_pointer_rebase_assignment(
            node, target, value, op, statement_context=statement_context
        )
        if private_rebase is not None:
            return private_rebase
        self.hlsl_private_pointer_word_view_write_error(target)
        self.hlsl_storage_struct_view_write_error(target)
        record_transfer = self.generate_hlsl_workgroup_record_transfer(
            target, value, op, statement_context=statement_context
        )
        if record_transfer is not None:
            return record_transfer
        storage_assignment = self.hlsl_half_storage.assignment(
            node, target, value, op, statement_context=statement_context
        )
        if storage_assignment is not None:
            return storage_assignment
        column_assignment = self.hlsl_matrix_orientation.column_assignment(
            target, value, op, statement_context
        )
        if column_assignment is not None:
            return column_assignment
        target_type = self.hlsl_assignment_target_type(target)
        target_name = (
            target
            if isinstance(target, str)
            else (
                getattr(target, "name", None)
                if isinstance(target, (IdentifierNode, VariableNode))
                else None
            )
        )
        if target_name is not None and op == "=":
            storage_struct_assignment = self.generate_hlsl_storage_struct_view_copy(
                self.hlsl_identifier_name(target_name),
                target_type,
                value,
                0,
            )
            if storage_struct_assignment is not None:
                return storage_struct_assignment.rstrip()
        boolean_compound_assignment = self.generate_hlsl_boolean_compound_assignment(
            node,
            target,
            value,
            op,
            target_type=target_type,
            statement_context=statement_context,
        )
        if boolean_compound_assignment is not None:
            return boolean_compound_assignment

        union_assignment = self.generate_hlsl_union_assignment(target, value, op)
        if union_assignment is not None:
            return union_assignment

        if op != "=":
            byte_update = self.hlsl_byte_update_expression(node, target, op, value)
            if byte_update is not None:
                return byte_update

        struct_list = generate_hlsl_struct_list_assignment(
            self, target, value, op, target_type, statement_context
        )
        if struct_list is not None:
            return struct_list

        if (
            isinstance(value, ArrayLiteralNode)
            and self.hlsl_outer_array_type(target_type)
            and not self.hlsl_glsl_buffer_whole_array_target(target)
        ):
            if op != "=" or not self.hlsl_repeatable_array_assignment_target(target):
                self.hlsl_fixed_array_aggregate_initializer(value, target_type)
                self.hlsl_aggregate_error(
                    value,
                    expected_type=self.map_type(target_type),
                    reason="array-assignment-unsupported",
                    detail=(
                        "this fixed-array assignment target cannot be expanded "
                        "without changing evaluation order"
                    ),
                )
            list_assignment = generate_hlsl_array_list_assignment(
                self, target, value, target_type
            )
            if list_assignment is not None:
                return list_assignment
            return "\n".join(
                self.hlsl_aggregate_assignment_statements(
                    self.generate_expression(target),
                    value,
                    target_type,
                    value,
                )
            )

        word_store = generate_hlsl_word_view_assignment(self, node, target, value, op)
        if word_store is not None:
            return word_store

        pointer_store = self.generate_hlsl_resource_pointer_dereference_assignment(
            target, value, op
        )
        if pointer_store is not None:
            return pointer_store

        resource_offset = self.generate_hlsl_resource_pointer_offset_assignment(
            target,
            value,
            op,
            statement_context=statement_context,
        )
        if resource_offset is not None:
            return resource_offset

        if isinstance(
            target, ArrayAccessNode
        ) and self.hlsl_resource_pointer_alias_expression(target.array):
            lhs = self.generate_hlsl_resource_pointer_access(
                target.array, target.index, require_write=True
            )
            binding = self.hlsl_resource_pointer_binding(target.array)
            compound_assignment = self.generate_hlsl_bfloat16_compound_assignment(
                node,
                target,
                value,
                op,
                lhs=lhs,
                target_type=binding.get("element_type") if binding else None,
            )
            if compound_assignment is not None:
                return compound_assignment
            compound_assignment = self.generate_hlsl_native_16_bit_compound_assignment(
                node,
                target,
                value,
                op,
                lhs=lhs,
                target_type=binding.get("element_type") if binding else None,
            )
            if compound_assignment is not None:
                return compound_assignment
            compound_assignment = (
                self.generate_hlsl_minimum_precision_integer_compound_assignment(
                    node,
                    target,
                    value,
                    op,
                    lhs=lhs,
                    target_type=binding.get("element_type") if binding else None,
                )
            )
            if compound_assignment is not None:
                return compound_assignment
            target_type = self.hlsl_resource_pointer_assignment_type(binding)
            compound_assignment = self.generate_hlsl_wide_integer_compound_assignment(
                node, target, value, op, lhs=lhs, target_type=target_type
            )
            if compound_assignment is not None:
                return compound_assignment
            rhs = self.generate_expression_with_expected(
                value,
                target_type,
                source_location=getattr(node, "source_location", None),
            )
            rhs = self.hlsl_bfloat16_storage_assignment_expression(
                target,
                rhs,
                target_type,
                source_type=self.hlsl_source_expression_type(value),
            )
            rhs = self.hlsl_precise_stored_value(target, value, rhs, target_type, op)
            return f"{lhs} {op} {rhs}"

        block_store = self.generate_glsl_buffer_block_store(target, value, op)
        if block_store is not None:
            return block_store
        unsupported_store = self.unsupported_glsl_buffer_block_assignment_diagnostic(
            target
        )
        if unsupported_store is not None:
            return unsupported_store

        atomic_assignment = self.generate_hlsl_typed_buffer_atomic_statement(
            value, target, target_type
        )
        if atomic_assignment is not None and op == "=":
            return atomic_assignment

        lifted_value = self.hlsl_typed_buffer_atomic_lifted_expression(
            value, target_type
        )
        if lifted_value is not None:
            lift_statements, rhs = lifted_value
            lhs = self.generate_expression(target)
            return "\n".join([*lift_statements, f"{lhs} {op} {rhs}"])

        compound_assignment = self.generate_hlsl_complex64_compound_assignment(
            node,
            target,
            value,
            op,
            target_type=target_type,
        )
        if compound_assignment is not None:
            return compound_assignment

        compound_assignment = self.generate_hlsl_bfloat16_compound_assignment(
            node,
            target,
            value,
            op,
            target_type=target_type,
        )
        if compound_assignment is not None:
            return compound_assignment

        compound_assignment = self.generate_hlsl_native_16_bit_compound_assignment(
            node,
            target,
            value,
            op,
            target_type=target_type,
        )
        if compound_assignment is not None:
            return compound_assignment

        compound_assignment = (
            self.generate_hlsl_minimum_precision_integer_compound_assignment(
                node,
                target,
                value,
                op,
                target_type=target_type,
            )
        )
        if compound_assignment is not None:
            return compound_assignment

        compound_assignment = self.generate_hlsl_wide_integer_compound_assignment(
            node, target, value, op, target_type=target_type
        )
        if compound_assignment is not None:
            return compound_assignment
        lhs = self.generate_expression(target)
        rhs = self.generate_expression_with_expected(
            value,
            target_type,
            source_location=getattr(node, "source_location", None),
        )
        rhs = self.hlsl_bfloat16_storage_assignment_expression(
            target,
            rhs,
            target_type,
            source_type=self.hlsl_source_expression_type(value),
        )
        rhs = self.hlsl_precise_stored_value(target, value, rhs, target_type, op)
        return f"{lhs} {op} {rhs}"

    def hlsl_resource_pointer_delta_type(self, value):
        source_type = self.hlsl_source_expression_type(value)
        mapped_type = self.map_type(source_type) if source_type is not None else ""
        if mapped_type in self.HLSL_RESOURCE_POINTER_UNSIGNED_DELTA_TYPES:
            return "uint"
        if mapped_type == "int64_t":
            return "int64_t"
        return "int"

    def generate_hlsl_private_pointer_rebase_assignment(
        self, node, target, value, op, *, statement_context=False
    ):
        """Lower ``p += n`` on a private pointer parameter to ``p_base += int(n)``.

        Private pointer parameters are emitted as ``array, int base`` pairs, so
        moving the pointer moves only the by-value base index.  The private
        pointer analysis must have proven every reachable view offset of this
        rebase non-negative and within the backing arrays it validated.
        """

        name = self.hlsl_private_pointer_rebase_target(
            node, set(self.current_hlsl_private_pointer_base_names)
        )
        if name is None:
            return None
        function_name = self.current_function_name

        def reject(detail):
            raise DirectXPrivatePointerParameterError(
                "DirectX cannot rebase private pointer parameter "
                f"'{function_name}.{name}' because {detail}",
                function_name=function_name,
                parameter_name=name,
                reason="pointer-parameter-rebinding",
                source_location=getattr(node, "source_location", None),
            )

        rejected = self.hlsl_private_pointer_rejected_rebases.get(id(node))
        if rejected is not None:
            reject(rejected)
        if id(node) not in self.hlsl_private_pointer_rebase_ids:
            reject("no bounded view offset was proven for it")
        if not statement_context:
            reject("the rebased pointer value is used")
        if id(node) not in self.hlsl_private_pointer_nonzero_rebase_ids:
            return ""
        scalar_parameters = self.function_private_pointer_scalar_parameters.get(
            function_name, set()
        )
        variant = self.current_hlsl_private_pointer_variant.get(name)
        if name in scalar_parameters:
            reject("it is bound to a scalar")
        if self.hlsl_private_pointer_word_view_for_variant(variant) is not None:
            reject("it is bound to a reinterpreted scalar view")
        operator = self.map_operator(op)
        rendered_delta = self.generate_expression(value)
        base_name = self.current_hlsl_private_pointer_base_names[name]
        return f"{base_name} {operator} int({rendered_delta})"

    def generate_hlsl_resource_pointer_offset_assignment(
        self, target, value, op, *, statement_context=False
    ):
        alias_assignment = self.generate_hlsl_resource_pointer_alias_assignment(
            target,
            value,
            op,
            statement_context=statement_context,
        )
        if alias_assignment is not None:
            return alias_assignment
        if op not in {"+=", "-="}:
            return None
        if not (
            hasattr(target, "__class__")
            and (
                "Identifier" in str(target.__class__)
                or target.__class__.__name__ == "VariableNode"
            )
        ):
            return None

        original_name = getattr(target, "name", None)
        if original_name is None:
            return None

        binding = self.hlsl_resource_pointer_binding(target)
        if binding is not None:
            used_names = set(self.current_identifier_reserved_names)
            used_names.update(self.local_variable_types)
            used_names.update(self.current_identifier_aliases.values())
            offset_name = self.hlsl_unique_local_identifier(
                f"{original_name}_offset", used_names
            )
            self.current_identifier_reserved_names.add(offset_name)
            initial_offset = binding["offset"]
            binding["offset"] = offset_name
            self.current_hlsl_resource_pointer_aliases[original_name] = binding
            self.local_variable_types[offset_name] = "int64_t"
            rhs = self.generate_expression(value)
            return (
                f"int64_t {offset_name} = int64_t({initial_offset})\n"
                f"{offset_name} {op} {self.hlsl_resource_pointer_delta_type(value)}({rhs})"
            )

        if op != "+=":
            return None
        rendered_name = self.generate_expression(target)
        if not self.hlsl_writable_resource_pointer_name(rendered_name, original_name):
            return None

        rhs = self.generate_expression(value)
        previous_offset = self.current_hlsl_resource_pointer_offsets.get(rendered_name)
        if previous_offset:
            self.current_hlsl_resource_pointer_offsets[rendered_name] = (
                f"({previous_offset} + {rhs})"
            )
        else:
            self.current_hlsl_resource_pointer_offsets[rendered_name] = rhs
        return ""

    def hlsl_writable_resource_pointer_name(self, name, original_name=None):
        candidate_names = [name]
        if original_name is not None and original_name not in candidate_names:
            candidate_names.append(original_name)
        aliased_name = self.current_identifier_aliases.get(original_name)
        if aliased_name is not None and aliased_name not in candidate_names:
            candidate_names.append(aliased_name)

        for candidate in candidate_names:
            resource_type = (
                self.local_variable_types.get(candidate)
                or self.hlsl_struct_buffer_resource_types.get(candidate)
                or self.global_variable_types.get(candidate)
            )
            if isinstance(resource_type, str) and resource_type.startswith(
                ("RWStructuredBuffer<", "RWByteAddressBuffer")
            ):
                return True
        return False

    def hlsl_strip_wrapping_parentheses(self, expression):
        expression = str(expression).strip()
        while expression.startswith("(") and expression.endswith(")"):
            inner = expression[1:-1].strip()
            if not inner:
                break
            depth = 0
            wraps = True
            for index, char in enumerate(expression):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(expression) - 1:
                        wraps = False
                        break
                if depth < 0:
                    wraps = False
                    break
            if not wraps or depth != 0:
                break
            expression = inner
        return expression

    def hlsl_condition_expression(self, condition):
        return self.hlsl_precise_condition(
            condition,
            self.hlsl_strip_wrapping_parentheses(
                self.generate_expression_with_expected(condition, "bool")
            ),
        )

    def generate_if(self, node, indent):
        indent_str = "    " * indent

        condition = getattr(node, "condition", getattr(node, "if_condition", None))
        then_body = getattr(node, "then_branch", getattr(node, "if_body", []))
        else_body = getattr(node, "else_branch", getattr(node, "else_body", []))

        literal_condition = self.literal_int_value(condition, {})
        guards, short_circuited = getattr(
            self, "current_hlsl_constant_branch_guards", ({}, set())
        )
        fixed = None if literal_condition is not None else guards.get(id(node))
        if fixed is not None:
            if id(node) in short_circuited:
                # The undecided operands may not type-check in HLSL.
                literal_condition = int(fixed)
            elif fixed:
                else_body = []
            else:
                then_body = []
            if literal_condition is None and not (then_body or else_body):
                return ""
        if literal_condition is not None and (
            literal_condition != 0 or not getattr(node, "else_if_conditions", None)
        ):
            selected_body = then_body if literal_condition != 0 else else_body
            if not selected_body:
                return ""
            if isinstance(selected_body, IfNode):
                return self.generate_if(selected_body, indent)
            if isinstance(selected_body, BlockNode):
                return self.generate_block(selected_body, indent)
            code = f"{indent_str}{{\n"
            code += self.generate_scoped_statement_body(selected_body, indent + 1)
            return code + f"{indent_str}}}\n"

        if id(node) in getattr(self, "hlsl_software_subgroup_masked_if_plans", {}):
            return self.generate_hlsl_software_subgroup_masked_if(node, indent)

        atomic_chain = generate_hlsl_atomic_condition_if(
            self, node, condition, then_body, else_body, indent
        )
        if atomic_chain is not None:
            return atomic_chain

        code = f"{indent_str}if ({self.hlsl_condition_expression(condition)}) {{\n"

        code += self.generate_scoped_statement_body(then_body, indent + 1)

        code += f"{indent_str}}}"

        if hasattr(node, "else_if_conditions") and hasattr(node, "else_if_bodies"):
            for else_if_condition, else_if_body in zip(
                node.else_if_conditions, node.else_if_bodies
            ):
                code += (
                    f" else if ({self.hlsl_condition_expression(else_if_condition)}) "
                    "{\n"
                )
                code += self.generate_scoped_statement_body(else_if_body, indent + 1)
                code += f"{indent_str}}}"

        if else_body:
            code += " else {\n"
            code += self.generate_scoped_statement_body(else_body, indent + 1)
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent
        previous_local_variable_types = dict(self.local_variable_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        previous_resource_pointer_aliases = dict(
            self.current_hlsl_resource_pointer_aliases
        )
        previous_bounded_loop_indices = dict(self.current_hlsl_bounded_loop_indices)
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        previous_visible_int_constants = (
            None
            if self.current_hlsl_visible_int_constants is None
            else dict(self.current_hlsl_visible_int_constants)
        )
        previous_texture_offset_constants = (
            None
            if self.current_hlsl_texture_offset_constants is None
            else dict(self.current_hlsl_texture_offset_constants)
        )

        try:
            loop_range = self.hlsl_for_index_range(node)
            if loop_range is not None:
                loop_name, loop_start, loop_stop = loop_range
                self.current_hlsl_bounded_loop_indices[loop_name] = (
                    loop_start,
                    loop_stop,
                )
            strided_plan = getattr(
                self, "hlsl_software_subgroup_strided_for_plans", {}
            ).get(id(node))
            if strided_plan is not None:
                return self.generate_hlsl_software_subgroup_strided_for(
                    node, indent, strided_plan
                )
            init = ""
            condition = ""
            update = ""
            condition_guard = ""

            if hasattr(node, "init") and node.init:
                if isinstance(node.init, str):
                    init = node.init
                else:
                    init = self.generate_for_initializer(node.init)

            if hasattr(node, "condition") and node.condition:
                if isinstance(node.condition, str):
                    condition = node.condition
                elif condition_has_hlsl_atomic(self, node.condition):
                    condition_guard = hlsl_atomic_condition_loop_guard(
                        self, node.condition, indent + 1
                    )
                else:
                    # A proven software-subgroup counter loop may test an
                    # equivalent condition that cannot overflow.
                    source_condition = getattr(
                        self, "hlsl_software_subgroup_rebased_conditions", {}
                    ).get(id(node), node.condition)
                    condition = self.hlsl_precise_condition(
                        source_condition,
                        self.generate_expression(source_condition).strip().rstrip(";"),
                    )

            if hasattr(node, "update") and node.update:
                if isinstance(node.update, str):
                    update = node.update
                elif isinstance(node.update, list):
                    update = ", ".join(
                        self.generate_hlsl_discarded_update(expr)
                        for expr in node.update
                    )
                else:
                    update = self.generate_hlsl_discarded_update(node.update)

            code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
            code += condition_guard

            body = getattr(node, "body", [])
            code += self.generate_scoped_statement_body(body, indent + 1)

            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_hlsl_resource_pointer_aliases = (
                previous_resource_pointer_aliases
            )
            self.current_hlsl_bounded_loop_indices = previous_bounded_loop_indices
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )
            self.current_hlsl_visible_int_constants = previous_visible_int_constants
            self.current_hlsl_texture_offset_constants = (
                previous_texture_offset_constants
            )

    def generate_block(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}{{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "statements", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_scoped_statement_body(self, body, indent):
        previous_local_variable_types = dict(self.local_variable_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        previous_resource_pointer_aliases = dict(
            self.current_hlsl_resource_pointer_aliases
        )
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        previous_visible_int_constants = (
            None
            if self.current_hlsl_visible_int_constants is None
            else dict(self.current_hlsl_visible_int_constants)
        )
        previous_texture_offset_constants = (
            None
            if self.current_hlsl_texture_offset_constants is None
            else dict(self.current_hlsl_texture_offset_constants)
        )
        try:
            return self.generate_statement_body(body, indent)
        finally:
            self.local_variable_types = previous_local_variable_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_hlsl_resource_pointer_aliases = (
                previous_resource_pointer_aliases
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )
            self.current_hlsl_visible_int_constants = previous_visible_int_constants
            self.current_hlsl_texture_offset_constants = (
                previous_texture_offset_constants
            )

    def generate_hlsl_discarded_update(self, expression):
        """A for-loop update, whose value the loop discards."""
        step = self.generate_hlsl_resource_pointer_step(
            expression, statement_context=True
        )
        if step is not None:
            return step
        return self.generate_discarded_expression(expression).strip().rstrip(";")

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable_node = getattr(node, "iterable", "")
        previous_local_variable_types = dict(self.local_variable_types)
        previous_local_variable_source_types = dict(self.local_variable_source_types)
        previous_identifier_aliases = dict(self.current_identifier_aliases)
        previous_resource_pointer_aliases = dict(
            self.current_hlsl_resource_pointer_aliases
        )
        previous_unsupported_locals = set(
            self.current_unsupported_glsl_buffer_block_local_variables
        )
        previous_visible_int_constants = (
            None
            if self.current_hlsl_visible_int_constants is None
            else dict(self.current_hlsl_visible_int_constants)
        )
        previous_texture_offset_constants = (
            None
            if self.current_hlsl_texture_offset_constants is None
            else dict(self.current_hlsl_texture_offset_constants)
        )

        try:
            if getattr(node, "binding_type", None) is not None:
                return generate_typed_array_for_in(self, node, indent, target="directx")
            pattern_name = self.hlsl_declaration_identifier_name(pattern)
            self.current_unsupported_glsl_buffer_block_local_variables.discard(pattern)

            if isinstance(iterable_node, RangeNode):
                self.local_variable_types[pattern] = "int"
                self.local_variable_source_types[pattern] = "int"
                start = self.generate_expression(iterable_node.start)
                end = self.generate_expression(iterable_node.end)
                comparator = "<=" if iterable_node.inclusive else "<"
                code = (
                    f"{indent_str}for (int {pattern_name} = {start}; "
                    f"{pattern_name} {comparator} {end}; ++{pattern_name}) {{\n"
                )
                code += self.generate_scoped_statement_body(
                    getattr(node, "body", []), indent + 1
                )
                code += f"{indent_str}}}\n"
                return code

            binding_qualifier = self.validate_hlsl_for_in_binding(node, pattern)
            iterable_type = self.hlsl_for_in_iterable_type(node, pattern, iterable_node)

            # Preserve the established shorthand where an integer expression is
            # the exclusive upper bound of a counted loop.
            if self.is_scalar_integer_type(iterable_type):
                self.local_variable_types[pattern] = "int"
                self.local_variable_source_types[pattern] = "int"
                iterable = self.generate_expression(iterable_node)
                code = (
                    f"{indent_str}for (int {pattern_name} = 0; "
                    f"{pattern_name} < {iterable}; ++{pattern_name}) {{\n"
                )
                code += self.generate_scoped_statement_body(
                    getattr(node, "body", []), indent + 1
                )
                code += f"{indent_str}}}\n"
                return code

            extent, element_type = self.hlsl_for_in_array_shape(
                node, pattern, iterable_type
            )
            if extent == 0:
                if not self.hlsl_expression_is_repeatable(iterable_node):
                    self.raise_hlsl_for_in_iterable_error(
                        node,
                        pattern,
                        iterable_type,
                        reason="side-effecting-empty-array",
                    )
                return ""

            mapped_iterable_type = self.map_type(iterable_type)
            mapped_element_type = self.map_type(element_type)
            element_base, _element_suffix = split_array_type_suffix(mapped_element_type)
            if self.is_resource_parameter_type(
                element_base
            ) or self.is_hlsl_global_resource_type(element_base):
                self.raise_hlsl_for_in_iterable_error(
                    node,
                    pattern,
                    iterable_type,
                    binding_type=element_type,
                    reason="unsupported-element-type",
                )

            used_names = set(self.local_variable_types)
            used_names.update(self.global_variable_types)
            used_names.update(self.current_identifier_aliases.values())
            used_names.update(self.current_identifier_reserved_names)
            container_name = self.hlsl_unique_local_identifier(
                f"{pattern}_crossgl_iterable", used_names
            )
            used_names.add(container_name)
            index_name = self.hlsl_unique_local_identifier(
                f"{pattern}_crossgl_index", used_names
            )
            iterable = self.generate_expression_with_expected(
                iterable_node, iterable_type
            )
            container_declaration = format_c_style_array_declaration(
                mapped_iterable_type, container_name
            )
            pattern_declaration = format_c_style_array_declaration(
                mapped_element_type, pattern_name
            )
            pattern_declaration = f"{binding_qualifier}{pattern_declaration}"

            self.local_variable_types[container_name] = iterable_type
            self.local_variable_source_types[container_name] = iterable_type
            self.local_variable_types[index_name] = "int"
            self.local_variable_source_types[index_name] = "int"
            self.local_variable_types[pattern] = element_type
            self.local_variable_source_types[pattern] = element_type
            code = (
                f"{indent_str}{{\n"
                f"{indent_str}    {container_declaration} = {iterable};\n"
                f"{indent_str}    for (int {index_name} = 0; "
                f"{index_name} < {extent}; ++{index_name}) {{\n"
                f"{indent_str}        {pattern_declaration} = "
                f"{container_name}[{index_name}];\n"
            )
            code += self.generate_scoped_statement_body(
                getattr(node, "body", []), indent + 2
            )
            code += f"{indent_str}    }}\n"
            code += f"{indent_str}}}\n"
            return code
        finally:
            self.local_variable_types = previous_local_variable_types
            self.local_variable_source_types = previous_local_variable_source_types
            self.current_identifier_aliases = previous_identifier_aliases
            self.current_hlsl_resource_pointer_aliases = (
                previous_resource_pointer_aliases
            )
            self.current_unsupported_glsl_buffer_block_local_variables = (
                previous_unsupported_locals
            )
            self.current_hlsl_visible_int_constants = previous_visible_int_constants
            self.current_hlsl_texture_offset_constants = (
                previous_texture_offset_constants
            )

    def validate_hlsl_for_in_binding(self, node, pattern):
        binding_type = getattr(
            node, "binding_type", getattr(node, "pattern_type", None)
        )
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "binding_qualifiers", []) or []
        }
        binding_type_name = self.type_name_string(binding_type or "")
        if isinstance(binding_type, ReferenceType):
            binding_type_name = f"{binding_type_name}&"
        readonly = bool(
            qualifiers & {"const", "constant", "in", "readonly"}
            or "const" in binding_type_name.lower().split()
            or "readonly" in binding_type_name.lower().split()
        )
        mutable_reference = bool(
            getattr(node, "is_mutable_reference", False)
            or getattr(node, "mutable_reference", False)
            or "mut" in qualifiers
            or (
                isinstance(binding_type, ReferenceType)
                and getattr(binding_type, "is_mutable", False)
            )
            or (
                not isinstance(binding_type, ReferenceType)
                and binding_type_name.rstrip().endswith("&")
                and not readonly
            )
        )
        if mutable_reference:
            self.raise_hlsl_for_in_iterable_error(
                node,
                pattern,
                self.type_name_string(
                    self.hlsl_source_expression_type(getattr(node, "iterable", None))
                ),
                binding_type=binding_type_name or None,
                reason="mutable-reference-binding",
            )

        modifiers = []
        if "precise" in qualifiers:
            modifiers.append("precise")
        if readonly or (
            isinstance(binding_type, ReferenceType)
            and not getattr(binding_type, "is_mutable", False)
        ):
            modifiers.append("const")
        return f"{' '.join(modifiers)} " if modifiers else ""

    def hlsl_for_in_iterable_type(self, node, pattern, iterable_node):
        iterable_type = self.type_name_string(
            self.hlsl_source_expression_type(iterable_node)
        )
        if not iterable_type:
            self.raise_hlsl_for_in_iterable_error(
                node, pattern, None, reason="unresolved-iterable-type"
            )
        return iterable_type

    def hlsl_for_in_outer_array_type(self, iterable_type):
        base_type, array_suffix = split_array_type_suffix(str(iterable_type or ""))
        if not array_suffix:
            return None
        match = re.fullmatch(r"\[([^\]]*)\](.*)", array_suffix)
        if match is None:
            return None
        extent_text = match.group(1).strip()
        remaining_suffix = match.group(2)
        return extent_text, f"{base_type}{remaining_suffix}"

    def hlsl_for_in_array_shape(self, node, pattern, iterable_type):
        array_shape = self.hlsl_for_in_outer_array_type(iterable_type)
        if array_shape is None:
            self.raise_hlsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="not-fixed-array"
            )
        extent_text, element_type = array_shape
        if not extent_text:
            self.raise_hlsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="unknown-extent"
            )
        constants = dict(self.literal_int_constants)
        constants.update(self.current_hlsl_visible_int_constants or {})
        extent = evaluate_literal_int_expression(extent_text, constants)
        if extent is None:
            self.raise_hlsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="unknown-extent"
            )
        if extent < 0:
            self.raise_hlsl_for_in_iterable_error(
                node, pattern, iterable_type, reason="invalid-extent"
            )
        return extent, element_type

    def raise_hlsl_for_in_iterable_error(
        self,
        node,
        pattern,
        iterable_type,
        *,
        binding_type=None,
        reason,
    ):
        descriptions = {
            "reference-container-shadowed": (
                "has a reference whose binding or container is shadowed in the loop body"
            ),
            "reference-helper-transport-unsupported": (
                "passes a reference binding to a helper whose parameter copies cannot preserve aliasing"
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
                "requires mutable reference binding, which HLSL cannot preserve"
            ),
            "not-fixed-array": "is neither a fixed array nor an integer loop bound",
            "side-effecting-empty-array": (
                "is an empty array whose container expression has side effects"
            ),
            "unknown-extent": "does not have a compile-time fixed-array extent",
            "unresolved-iterable-type": "has an unresolved iterable type",
            "unsupported-element-type": "contains an unsupported resource element type",
        }
        detail = descriptions.get(reason, "has no faithful HLSL iterable contract")
        type_label = iterable_type or "<unresolved>"
        raise DirectXForInIterableError(
            f"DirectX for-in binding '{pattern}' over '{type_label}' {detail}",
            pattern=pattern,
            iterable_type=iterable_type,
            binding_type=binding_type,
            reason=reason,
            source_location=getattr(node, "source_location", None)
            or getattr(getattr(node, "iterable", None), "source_location", None),
        )

    def generate_while(self, node, indent):
        atomic_loop = generate_hlsl_atomic_condition_while(self, node, indent)
        if atomic_loop is not None:
            return atomic_loop
        indent_str = "    " * indent
        condition = self.hlsl_precise_condition(
            getattr(node, "condition", None),
            self.generate_expression(getattr(node, "condition", "")),
        )

        code = f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        atomic_loop = generate_hlsl_atomic_condition_do_while(self, node, indent)
        if atomic_loop is not None:
            return atomic_loop
        indent_str = "    " * indent
        condition = self.hlsl_precise_condition(
            getattr(node, "condition", None),
            self.generate_expression(getattr(node, "condition", "")),
        )

        code = f"{indent_str}do {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}} while ({condition});\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent

        code = f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_statement_body(
            getattr(node, "body", []), indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        code = ""
        atomic_selector = generate_hlsl_atomic_condition_switch(self, node, indent)
        if atomic_selector is not None:
            code, expression = atomic_selector
        else:
            expression = self.generate_expression(getattr(node, "expression", ""))

        code += f"{indent_str}switch ({expression}) {{\n"
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

    def format_match_condition(self, condition):
        return f"({self.hlsl_strip_wrapping_parentheses(condition)})"

    def generate_match(self, node, indent):
        subject_type = self.type_name_string(
            self.expression_result_type(getattr(node, "expression", None))
        )
        if is_switch_lowerable_match(node) and self.map_type(subject_type) != "bool":
            return generate_switch_match(self, node, indent)
        return generate_ordered_conditional_match(self, node, indent, "HLSL")

    def hlsl_statement_terminates(self, statement):
        if isinstance(statement, (BreakNode, ContinueNode, ReturnNode)):
            return True
        if isinstance(statement, BlockNode):
            return self.statement_body_terminates(statement)
        if not isinstance(statement, IfNode):
            return False

        condition = getattr(
            statement, "condition", getattr(statement, "if_condition", None)
        )
        then_body = getattr(
            statement, "then_branch", getattr(statement, "if_body", None)
        )
        else_body = getattr(
            statement, "else_branch", getattr(statement, "else_body", None)
        )
        literal_condition = self.literal_int_value(condition, {})
        if literal_condition is not None:
            selected_body = then_body if literal_condition != 0 else else_body
            return self.statement_body_terminates(selected_body)

        if not self.statement_body_terminates(then_body):
            return False
        if any(
            not self.statement_body_terminates(branch)
            for branch in getattr(statement, "else_if_bodies", []) or []
        ):
            return False
        return else_body is not None and self.statement_body_terminates(else_body)

    def statement_body_terminates(self, body):
        if hasattr(body, "statements"):
            statements = body.statements
        elif isinstance(body, list):
            statements = body
        elif body is None:
            statements = []
        else:
            statements = [body]

        return bool(statements) and self.hlsl_statement_terminates(statements[-1])

    def statement_body_has_statements(self, body):
        if hasattr(body, "statements"):
            return bool(body.statements)
        if isinstance(body, list):
            return bool(body)
        return body is not None

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def generate_statement_body(self, body, indent):
        code = ""
        if hasattr(body, "statements"):
            statements = body.statements
        elif isinstance(body, list):
            statements = body
        elif body is not None:
            statements = [body]
        else:
            statements = []

        for index, stmt in enumerate(statements):
            if index == len(statements) - 2:
                selection_return = self.hlsl_half_selection_tail_return(
                    stmt, statements[index + 1]
                )
                if selection_return is not None:
                    code += self.generate_statement(selection_return, indent)
                    break
            if isinstance(stmt, VariableNode):
                following_statements = statements[index + 1 :]
                self.prepare_hlsl_resource_pointer_array(stmt, following_statements)
                self.activate_hlsl_hoisted_groupshared_declaration(stmt)
            code += self.generate_statement(stmt, indent)
        return code

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, str):
            return init
        if isinstance(init, list):
            declaration_list = all(isinstance(item, VariableNode) for item in init)
            return ", ".join(
                (
                    self.generate_for_variable_initializer(
                        item, include_type=not declaration_list or index == 0
                    )
                    if isinstance(item, VariableNode)
                    else self.generate_expression(item).strip().rstrip(";")
                )
                for index, item in enumerate(init)
            )
        if isinstance(init, VariableNode):
            return self.generate_for_variable_initializer(init)
        if hasattr(init, "__class__") and "ExpressionStatement" in str(init.__class__):
            return self.generate_statement(init, 0).strip().rstrip(";")
        return self.generate_expression(init).strip().rstrip(";")

    def generate_for_variable_initializer(self, node, include_type=True):
        source_vtype = self.hlsl_source_variable_declared_type(node)
        vtype = self.local_variable_declared_type(node)
        self.local_variable_types[node.name] = self.type_name_string(vtype)
        self.local_variable_source_types[node.name] = self.type_name_string(
            source_vtype
        )
        name = self.hlsl_declaration_identifier_name(node.name)
        mapped_type = self.map_type(vtype)
        declaration = format_c_style_array_declaration(mapped_type, name)
        if include_type:
            declaration = f"{self.local_variable_qualifier(node)}{declaration}"
        else:
            prefix = f"{mapped_type} "
            if declaration.startswith(prefix):
                declaration = declaration[len(prefix) :]

        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            value = self.generate_expression_with_expected(initial_value, vtype)
            declaration = f"{declaration} = {value}"
        return declaration

    def is_hlsl_builtin_option_none_expression(self, expr):
        if not getattr(self, "hlsl_builtin_option_available", False):
            return False
        if isinstance(expr, str):
            return expr.rsplit("::", 1)[-1] == "None"
        name = getattr(expr, "name", None)
        return isinstance(name, str) and name.rsplit("::", 1)[-1] == "None"

    def hlsl_builtin_option_constructor_name(self, func_name):
        if not getattr(self, "hlsl_builtin_option_available", False):
            return func_name
        base_name = str(func_name).rsplit("::", 1)[-1]
        if base_name == "Some":
            return "Option::Some"
        if base_name == "None":
            return "Option::Absent"
        return func_name

    def generate_hlsl_builtin_option_none_value(self):
        if not getattr(self, "hlsl_builtin_option_available", False):
            return None
        return generate_enum_constructor_call(self, "Option::Absent", [])

    def generate_hlsl_builtin_option_none_comparison(self, left, op, right):
        if op not in {"==", "!="}:
            return None
        left_is_none = self.is_hlsl_builtin_option_none_expression(left)
        right_is_none = self.is_hlsl_builtin_option_none_expression(right)
        if left_is_none == right_is_none:
            return None

        subject = right if left_is_none else left
        subject_expr = self.generate_expression(subject)
        operator = "==" if op == "==" else "!="
        return f"({subject_expr}.variant {operator} Option_Absent)"

    def directx_cooperative_matrix_unsupported(
        self,
        message,
        *,
        operation,
        matrix_type,
        reason,
        source_location=None,
    ):
        fragment_contract = _directx_cooperative_matrix_fragment_contract(matrix_type)
        return DirectXCooperativeMatrixUnsupportedError(
            message,
            operation=operation,
            matrix_type=matrix_type,
            **fragment_contract,
            reason=reason,
            source_location=(
                source_location
                if source_location is not None
                else getattr(matrix_type, "source_location", None)
            ),
        )

    @classmethod
    def directx_cooperative_matrix_contract_signature(cls, lowering):
        return tuple(
            lowering[field] for field in cls.DIRECTX_COOPERATIVE_MATRIX_CONTRACT_FIELDS
        )

    @classmethod
    def directx_cooperative_matrix_contract_differences(cls, left, right):
        return [
            field
            for field in cls.DIRECTX_COOPERATIVE_MATRIX_CONTRACT_FIELDS
            if left[field] != right[field]
        ]
