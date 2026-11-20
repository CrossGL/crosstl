"""Loops, conditionals, switch, assignments, compound assignments, initializers and lambda expressions."""

import re

from ....translator.integer_literals import integer_literal_parts
from ...switch_clauses import ordered_switch_clauses
from ..aggregate_scalar_conversions import render_aggregate_scalar_conversion
from ..atomic_result_conversions import (
    convert_atomic_result,
    is_atomic_result_call,
)
from ..implicit_constructor_conversions import converted_assignment_value
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    DesignatedInitializerNode,
    FunctionCallNode,
    InitializerListNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)
from .support import (
    MetalCallableLoweringError,
    MetalConstexprBranchLoweringError,
    MetalConstructorContractError,
    MetalIndexedComponentTypeResolutionError,
    MetalSourceOverloadResolutionError,
    MetalWideVectorLoweringError,
)


class MetalStatementMixin:
    """Members of ``MetalToCrossGLConverter``: loops, conditionals, switch, assignments, compound assignments, initializers and lambda expressions."""

    def generate_for_loop(self, node, indent, is_main):
        previous_variable_types = self.current_variable_types
        previous_variable_type_qualifiers = self.current_variable_type_qualifiers
        self.current_variable_types = dict(previous_variable_types)
        self.current_variable_type_qualifiers = dict(previous_variable_type_qualifiers)
        self.template_binding_shadow_scopes.append(set())
        try:
            for declaration in self.for_initializer_declarations(node.init):
                self.current_variable_types[declaration.name] = declaration.vtype
                self.current_variable_type_qualifiers[declaration.name] = (
                    self.metal_declaration_type_qualifiers(declaration)
                )

            init = self.generate_for_clause(node.init, is_main)
            condition = self.generate_for_clause(node.condition, is_main)
            update = self.generate_for_clause(node.update, is_main)

            code = f"for ({init}; {condition}; {update}) {{\n"
            code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
            code += "    " * indent + "}\n"
            return code
        finally:
            self.template_binding_shadow_scopes.pop()
            self.current_variable_types = previous_variable_types
            self.current_variable_type_qualifiers = previous_variable_type_qualifiers

    def for_initializer_declarations(self, initializer):
        if isinstance(initializer, (list, tuple)):
            return [
                declaration
                for item in initializer
                for declaration in self.for_initializer_declarations(item)
            ]
        declaration = (
            initializer.left if isinstance(initializer, AssignmentNode) else initializer
        )
        if isinstance(declaration, VariableNode) and declaration.vtype:
            return [declaration]
        return []

    def generate_for_clause(self, expr, is_main):
        if isinstance(expr, list):
            return ", ".join(self.generate_for_clause(item, is_main) for item in expr)
        return self.generate_expression(expr, is_main)

    def generate_range_for_loop(self, node, indent, is_main):
        iterable = self.generate_expression(node.iterable, is_main)
        iterable_type = self.expression_metal_type(node.iterable)
        element_type = self.split_outer_metal_declarator_array_type(iterable_type)
        standard_array = self.metal_array_type_parts(iterable_type)
        if standard_array is not None:
            element_type = standard_array[0]
        qualifiers = self.metal_declaration_type_qualifiers(node)
        binding_type = str(node.vtype).strip()
        reference = binding_type.endswith("&")
        value_type = binding_type.rstrip("&").strip()
        if value_type == "auto" and element_type is not None:
            value_type = element_type
        mapped_type = self.map_type(value_type)
        if reference:
            mapped_type += "&" if "const" in qualifiers else "& mut"
        qualifier_text = " ".join(qualifiers)
        declaration = f"{qualifier_text} {mapped_type}".strip()
        previous_variable_types = dict(self.current_variable_types)
        previous_variable_qualifiers = dict(self.current_variable_type_qualifiers)
        self.template_binding_shadow_scopes.append(set())
        try:
            self.current_variable_types[node.name] = value_type
            self.current_variable_type_qualifiers[node.name] = qualifiers
            if any(node.name in bindings for bindings in self.template_value_bindings):
                self.template_binding_shadow_scopes[-1].add(node.name)
            code = f"for {node.name}: {declaration} in {iterable} {{\n"
            code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
            code += "    " * indent + "}\n"
            return code
        finally:
            self.template_binding_shadow_scopes.pop()
            self.current_variable_types = previous_variable_types
            self.current_variable_type_qualifiers = previous_variable_qualifiers

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)
        code = f"while ({condition}) {{\n"
        code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)
        code = "do {\n"
        code += self.generate_scoped_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_if_statement(self, node, indent, is_main):
        if any(getattr(node, "if_constexpr", ())) or any(
            getattr(node, "else_if_constexpr", ())
        ):
            return self.generate_constexpr_if_statement(node, indent, is_main)
        code = ""
        if node.if_chain:
            for condition, body in node.if_chain:
                code += f"if ({self.generate_expression(condition, is_main)}) {{\n"
                code += self.generate_scoped_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"
        if node.else_if_chain:
            for condition, body in node.else_if_chain:
                code += (
                    f" else if ({self.generate_expression(condition, is_main)}) {{\n"
                )
                code += self.generate_scoped_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"

        if node.else_body:
            code += " else {\n"
            code += self.generate_scoped_function_body(
                node.else_body, indent + 1, is_main
            )
            code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_constexpr_if_statement(self, node, indent, is_main):
        code = ""
        selected = False
        branches = []
        for chain, flags in (
            (node.if_chain, node.if_constexpr),
            (node.else_if_chain, node.else_if_constexpr),
        ):
            branches.extend(
                (condition, body, flags[index] if index < len(flags) else False)
                for index, (condition, body) in enumerate(chain)
            )
        for condition, body, is_constexpr in branches:
            rendered = self.generate_expression(condition, is_main)
            if is_constexpr:
                value = self.evaluate_value_template_constant_expression(
                    self.substitute_local_integral_constant_text(rendered),
                    constants={"true": 1, "false": 0},
                )
                if value is None:
                    raise MetalConstexprBranchLoweringError(
                        rendered, getattr(node, "source_location", None)
                    )
                if not value:
                    continue
                # Retain a lexical block without emitting or inspecting the
                # discarded branch. Earlier runtime conditions still govern it.
                prefix = " else" if code else "if (true)"
                selected = True
            else:
                prefix = f"{' else if' if code else 'if'} ({rendered})"
            code += prefix + " {\n"
            code += self.generate_scoped_function_body(body, indent + 1, is_main)
            code += "    " * indent + "}"
            if selected:
                break
        if not selected and node.else_body:
            code += (" else" if code else "if (true)") + " {\n"
            code += self.generate_scoped_function_body(
                node.else_body, indent + 1, is_main
            )
            code += "    " * indent + "}"
        return code + "\n"

    def generate_small_vector_component_read(self, expression, info, is_main=False):
        vector = self.generate_postfix_operand(expression.array, is_main)
        index = self.generate_expression(expression.index, is_main)
        helper = self.small_vector_index_helper_name(info, "get")
        return f"{helper}({vector}, {index})"

    def generate_small_vector_component_assignment(
        self,
        target,
        info,
        rendered_value,
        operator,
        *,
        right_type,
        computation_type=None,
        is_main=False,
    ):
        operation = (
            "set"
            if operator == "="
            else self.small_vector_index_operation_name(operator)
        )
        if operation is None:
            raise MetalIndexedComponentTypeResolutionError(
                info["vector_type"],
                "vector-component-assignment",
                f"operator '{operator}' has no scalar component lowering",
                getattr(target, "source_location", None),
                self.metal_index_expression_text(target.index),
            )

        resource_operation = self.generate_small_vector_resource_component_operation(
            target,
            info,
            operation,
            rendered_value,
            right_type=right_type,
            computation_type=computation_type,
            is_main=is_main,
        )
        if resource_operation is not None:
            return resource_operation

        vector = self.generate_postfix_operand(target.array, is_main)
        index = self.generate_expression(target.index, is_main)
        helper = self.small_vector_index_helper_name(
            info,
            operation,
            right_type=right_type,
            computation_type=computation_type,
        )
        return f"{helper}({vector}, {index}, {rendered_value})"

    def generate_small_vector_component_update(
        self, target, operator, *, postfix, is_main=False
    ):
        if self.metal_cooperative_matrix_element_access(target) is not None:
            return None
        info = self.small_vector_index_info(target, require_type=True)
        if info is None:
            return None
        update = self.small_vector_index_operation_name(operator)
        if update is None:
            return None
        operation = f"{'post' if postfix else 'pre'}_{update}"
        computation_type = self.small_vector_update_computation_type(info, target)
        resource_operation = self.generate_small_vector_resource_component_operation(
            target,
            info,
            operation,
            computation_type=computation_type,
            is_main=is_main,
        )
        if resource_operation is not None:
            return resource_operation
        vector = self.generate_postfix_operand(target.array, is_main)
        index = self.generate_expression(target.index, is_main)
        helper = self.small_vector_index_helper_name(
            info, operation, computation_type=computation_type
        )
        return f"{helper}({vector}, {index})"

    METAL_COMPOUND_ASSIGNMENT_OPERATORS = frozenset({"+=", "-=", "*=", "/=", "%="})

    def metal_side_effect_free_lvalue_part(self, expr):
        """Whether evaluating ``expr`` again reads the same value and writes nothing."""
        if expr is None or isinstance(expr, (bool, int, float, str)):
            return True
        if isinstance(expr, VariableNode):
            return not getattr(expr, "vtype", None)
        if isinstance(expr, (MemberAccessNode,)):
            return self.metal_side_effect_free_lvalue_part(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.metal_side_effect_free_lvalue_part(
                expr.array
            ) and self.metal_side_effect_free_lvalue_part(expr.index)
        if isinstance(expr, UnaryOpNode):
            return expr.op not in {
                "++",
                "--",
                "&",
                "*",
            } and self.metal_side_effect_free_lvalue_part(expr.operand)
        if isinstance(expr, BinaryOpNode):
            return all(
                self.metal_side_effect_free_lvalue_part(part)
                for part in (expr.left, expr.right)
            )
        if isinstance(expr, CastNode):
            return self.metal_side_effect_free_lvalue_part(expr.expression)
        return False

    def metal_private_compound_lvalue(self, expr):
        """Whether ``expr`` is a repeatable lvalue in thread storage.

        Only member and element selections of a local value qualify, with
        indices free of calls, assignments and increments. Pointer selections,
        references and threadgroup, device or constant storage do not.
        """
        while isinstance(expr, (MemberAccessNode, ArrayAccessNode)):
            if isinstance(expr, MemberAccessNode):
                if getattr(expr, "is_pointer", False):
                    return False
                expr = expr.object
                continue
            array_type = self.expression_metal_type(expr.array)
            if (
                array_type is None
                or "*" in str(array_type)
                or self.metal_pointer_pointee_type_once(array_type) is not None
                or not self.metal_side_effect_free_lvalue_part(expr.index)
            ):
                return False
            expr = expr.array
        if isinstance(expr, VariableNode) and not getattr(expr, "vtype", None):
            name = expr.name
        elif isinstance(expr, str) and expr.isidentifier():
            name = expr
        else:
            return False
        variable_type = self.current_variable_types.get(name)
        if variable_type is None or any(
            token in str(variable_type) for token in ("*", "&")
        ):
            return False
        qualifiers = set(self.current_variable_type_qualifiers.get(name, ()))
        qualifiers.update(str(variable_type).split())
        return not qualifiers & {"threadgroup", "device", "constant", "volatile"}

    def generate_metal_compound_operator_assignment(self, node, is_main):
        """Lower ``x op= y`` on an aggregate through its source ``operator op=``.

        The preprocessor materializes a thread ``operator op=`` as a helper
        that applies the source body to a copy of ``x`` and returns it, so
        ``x = helper(x, y)`` stores the value the source operator stores in
        ``x``. ``x`` must be a repeatable thread-storage lvalue; anything
        else keeps the source assignment.
        """
        operator = getattr(node, "operator", None)
        if operator not in self.METAL_COMPOUND_ASSIGNMENT_OPERATORS:
            return None
        target_type = self.expression_metal_type(node.left)
        if (
            target_type is None
            or self.normalized_metal_type(self.resolve_type_alias(target_type))
            not in self.struct_member_types
            or not self.metal_free_operator_candidates(operator, 2)
            or not self.metal_private_compound_lvalue(node.left)
        ):
            return None
        try:
            call = self.generate_metal_free_operator_call(
                operator,
                (node.left, node.right),
                is_main,
                getattr(node, "source_location", None),
            )
        except MetalSourceOverloadResolutionError:
            return None
        if call is None:
            return None
        return f"{self.generate_expression(node.left, is_main)} = {call}"

    def structured_buffer_lvalue_root(self, expr):
        """The structured-buffer element a member, swizzle or index chain writes through, if any."""
        node = expr
        while isinstance(node, (MemberAccessNode, ArrayAccessNode)):
            if isinstance(node, ArrayAccessNode) and (
                self.is_structured_buffer_element_access(node)
            ):
                return node
            node = node.object if isinstance(node, MemberAccessNode) else node.array
        return None

    def generate_with_structured_buffer_lvalue(self, target, generate):
        """Run ``generate`` with ``target``'s structured-buffer element spelled as ``buffer[index]``.

        ``buffer_load(b, i)`` is a value; ``buffer_load(b, i).m = v`` would assign to a copy, so the element an
        assignment or increment writes through keeps CrossGL's indexing form, which names the storage.
        """
        root = self.structured_buffer_lvalue_root(target)
        if root is None:
            return generate()
        roots = self.structured_buffer_lvalue_roots
        roots.add(id(root))
        try:
            return generate()
        finally:
            roots.discard(id(root))

    def structured_buffer_parameter_layout(self, func, parameter):
        """The ``@buffer_layout`` a buffer or device pointer parameter needs to keep its Metal element layout."""
        if self.structured_buffer_pointer_type(parameter):
            element_type = self.structured_buffer_element_source_type(parameter)
            kind = "buffer parameter"
        else:
            element_type = self.metal_device_pointer_element_type(parameter)
            kind = "pointer parameter"
        if element_type is None:
            return None
        return self.metal_buffer_element_layout(
            element_type,
            f"{kind} {parameter.name} of {getattr(func, 'name', 'a function')}",
        )

    def binds_reference_argument(self, function_name, expression, index):
        """Whether argument ``index`` of a call binds a reference parameter of a source function."""
        candidates = self.metal_user_function_overloads(
            function_name, source_offset=self.alias_source_offset(expression)
        )
        return any(
            index < len(getattr(candidate, "params", None) or [])
            and self.reference_parameter(candidate.params[index])
            for candidate in candidates
        )

    def generate_assignment(self, node, is_main):
        return self.generate_with_structured_buffer_lvalue(
            node.left, lambda: self.generate_assignment_statement(node, is_main)
        )

    def generate_assignment_statement(self, node, is_main):
        contracted_assignment = self.generate_metal_contracted_assignment(node, is_main)
        if contracted_assignment is not None:
            return contracted_assignment
        compound_assignment = self.generate_metal_compound_operator_assignment(
            node, is_main
        )
        if compound_assignment is not None:
            return compound_assignment
        if (
            node.operator in {"/=", "+=", "-=", "*="}
            and self.metal_arithmetic_profile(node.operator[:-1])[0] is not None
        ):
            profiled_assignment = self.generate_profiled_metal_arithmetic_assignment(
                node, is_main
            )
            if profiled_assignment is not None:
                return profiled_assignment
        fragment_assignment = self.generate_cooperative_matrix_fragment_assignment(
            node, is_main
        )
        if fragment_assignment is not None:
            return fragment_assignment
        component_info = (
            None
            if self.metal_cooperative_matrix_element_access(node.left) is not None
            else self.small_vector_index_info(node.left, require_type=True)
        )
        if component_info is None and self.is_structured_buffer_element_access(
            node.left
        ):
            structured_store = self.generate_structured_buffer_store(
                node.left, node.right, node.operator, is_main
            )
            if structured_store is not None:
                return structured_store
        lhs_info = self.wide_vector_expression_info(node.left)
        lhs = (
            None
            if component_info is not None
            else self.generate_expression(node.left, is_main)
        )
        pushed_context = bool(
            self.is_supported_constexpr_local_declaration(node)
            and self.push_materialized_constexpr_expression_context(
                node.right,
                required=bool(self.current_function_materialization_bindings),
                allow_concrete_without_bindings=True,
            )
        )
        initializer_type = None
        if component_info is not None:
            initializer_type = component_info["element_type"]
        elif isinstance(node.left, VariableNode):
            initializer_type = getattr(node.left, "vtype", None)
            if not initializer_type and isinstance(node.right, InitializerListNode):
                initializer_type = self.expression_metal_type(node.left)
        elif isinstance(node.right, InitializerListNode):
            initializer_type = self.expression_metal_type(node.left)
        if self.binary16_widening_profile is not None and not initializer_type:
            initializer_type = self.expression_metal_type(node.left)
        if (
            not initializer_type
            and node.operator == "="
            and is_atomic_result_call(self, node.right)
        ):
            initializer_type = self.expression_metal_type(node.left)
        converted_rhs = None
        if component_info is None and lhs_info is None:
            converted_rhs = render_aggregate_scalar_conversion(
                self,
                (
                    getattr(node.left, "vtype", None)
                    if isinstance(node.left, VariableNode)
                    else None
                )
                or self.expression_metal_type(node.left),
                node.right,
                node.operator,
                is_main,
            )
        if converted_rhs is None and component_info is None and lhs_info is None:
            converted_rhs = converted_assignment_value(self, node, is_main)
        try:
            # `device T& r = buffer[i]` binds the element; buffer_load would
            # bind a copy and drop every write through `r`.
            if converted_rhs is not None:
                rhs = converted_rhs
            elif self.binds_reference_to_structured_buffer_element(node):
                rhs = self.generate_structured_buffer_element_lvalue(
                    node.right, is_main
                )
            else:
                rhs = self.generate_initializer_value(
                    self.array_initializer_without_designators(node.left, node.right),
                    is_main,
                    initializer_type,
                    (
                        self.variable_has_array_initializer_shape(node.left)
                        if isinstance(node.left, VariableNode)
                        else False
                    ),
                    (
                        self.declaration_constructor_receiver_address_space(node.left)
                        if isinstance(node.left, VariableNode)
                        else None
                    ),
                    copy_initialize_lvalue=(
                        isinstance(node.left, VariableNode)
                        and bool(getattr(node.left, "vtype", None))
                    ),
                )
        finally:
            if pushed_context:
                self.materialized_constexpr_expression_contexts.pop()
        op = node.operator
        if op != "=" and op.endswith("="):
            bfloat_plan = self.metal_bfloat_arithmetic_plan(
                op[:-1],
                self.expression_metal_type(node.left),
                self.expression_metal_type(node.right),
                getattr(node, "source_location", None),
            )
            if bfloat_plan is not None and bfloat_plan[1][1] is not None:
                rhs = f"{self.map_type(bfloat_plan[1][1])}({rhs})"
            conversion = self.metal_vector_scalar_conversion(
                op[:-1],
                self.expression_metal_type(node.left),
                self.expression_metal_type(node.right),
            )
            if conversion is not None and conversion[0] == 1:
                rhs = self.metal_vector_scalar_operand(
                    rhs, self.expression_metal_type(node.right), conversion[1]
                )
        if component_info is not None:
            if op == "=":
                right_type = component_info["element_type"]
                computation_type = None
            else:
                right_type, computation_type = (
                    self.small_vector_compound_operation_types(
                        component_info, op, node.right, node.left
                    )
                )
            return self.generate_small_vector_component_assignment(
                node.left,
                component_info,
                rhs,
                op,
                right_type=right_type,
                computation_type=computation_type,
                is_main=is_main,
            )
        if lhs_info is not None and op != "=":
            binary_operator = op[:-1] if op.endswith("=") else None
            if self.wide_vector_binary_operation_name(binary_operator) is None:
                raise MetalWideVectorLoweringError(
                    lhs_info["source_type"],
                    "the compound operator has no semantics-preserving "
                    "aggregate lowering",
                    getattr(node, "source_location", None),
                    operation=op,
                )
            right_info = self.wide_vector_expression_info(node.right)
            if right_info is not None and right_info["key"] != lhs_info["key"]:
                raise MetalWideVectorLoweringError(
                    lhs_info["source_type"],
                    "compound-assignment operands have different element types "
                    "or widths",
                    getattr(node, "source_location", None),
                    operation=op,
                )
            if right_info is None and not (
                self.wide_vector_constructor_argument_is_scalar(node.right)
            ):
                raise MetalWideVectorLoweringError(
                    lhs_info["source_type"],
                    "the compound-assignment right operand is not scalar",
                    getattr(node, "source_location", None),
                    operation=op,
                )
            right_kind = "vector" if right_info is not None else "scalar"
            self.wide_vector_compound_helpers.add(
                (lhs_info["key"], binary_operator, right_kind)
            )
            helper_name = self.wide_vector_compound_helper_name(
                lhs_info, binary_operator, right_kind
            )
            return f"{helper_name}({lhs}, {rhs})"
        if (
            op == "="
            and isinstance(node.left, VariableNode)
            and getattr(node.left, "vtype", None)
        ):
            self.register_local_integral_constant(node.left, rhs)
        return f"{lhs} {op} {rhs}"

    def generate_initializer_value(
        self,
        expr,
        is_main=False,
        expected_type=None,
        expected_array=False,
        receiver_address_space=None,
        copy_initialize_lvalue=False,
    ):
        wrapper_read = (
            self.generate_storage_wrapper_value_read(expr, is_main, expected_type)
            if not expected_array
            else None
        )
        if wrapper_read is not None:
            return wrapper_read
        if (
            expected_type
            and not self.is_plain_metal_auto_type(expected_type)
            and isinstance(expr, ArrayAccessNode)
            and self.metal_cooperative_matrix_element_access(expr) is None
        ):
            self.expression_metal_type(expr)
        constructor_contract = (
            self.metal_constructor_contract(expected_type) if expected_type else None
        )
        if constructor_contract and expected_array:
            raise MetalConstructorContractError(
                expected_type,
                (),
                [
                    self.constructor_candidate_signature(constructor)
                    for constructor in constructor_contract[1]
                ],
                "array element construction cannot be represented by one "
                "aggregate assignment",
                getattr(expr, "source_location", None),
            )
        if constructor_contract:
            expected_owner = self.normalized_metal_type(
                self.resolve_type_alias(expected_type)
            )
            expression_owner = None
            arguments = None
            if isinstance(expr, VectorConstructorNode):
                expression_owner = expr.type_name
                arguments = list(expr.args)
            elif isinstance(expr, FunctionCallNode):
                expression_owner = expr.name
                arguments = list(expr.args)
                if (
                    getattr(expr, "is_braced_constructor", False)
                    and len(arguments) == 1
                    and isinstance(arguments[0], InitializerListNode)
                ):
                    arguments = list(arguments[0].elements)
            elif isinstance(expr, CastNode):
                expression_owner = expr.target_type
                arguments = [expr.expression]
            if (
                expression_owner is not None
                and self.normalized_metal_type(
                    self.resolve_type_alias(expression_owner)
                )
                == expected_owner
            ):
                return self.generate_explicit_constructor_call(
                    expected_type,
                    arguments,
                    is_main,
                    getattr(expr, "source_location", None),
                    receiver_address_space,
                )
            copy_source = isinstance(
                expr, (VariableNode, MemberAccessNode, ArrayAccessNode)
            ) or (isinstance(expr, UnaryOpNode) and expr.op == "*")
            if copy_initialize_lvalue and copy_source:
                expression_type = self.metal_source_overload_value_type(
                    self.expression_metal_type(expr)
                )
                if self.metal_source_overload_type_identity(
                    expression_type
                ) == self.metal_source_overload_type_identity(expected_owner):
                    return self.generate_explicit_constructor_call(
                        expected_type,
                        [expr],
                        is_main,
                        getattr(expr, "source_location", None),
                        receiver_address_space,
                    )
            if not isinstance(expr, InitializerListNode):
                expression_type = self.metal_source_overload_value_type(
                    self.expression_metal_type(expr)
                )
                if expression_type is not None:
                    if self.metal_source_overload_type_identity(
                        expression_type
                    ) == self.metal_source_overload_type_identity(expected_owner):
                        return self.generate_expression(expr, is_main)
                    return self.generate_explicit_constructor_call(
                        expected_type,
                        [expr],
                        is_main,
                        getattr(expr, "source_location", None),
                        receiver_address_space,
                        allow_explicit=False,
                    )
        if isinstance(expr, InitializerListNode):
            if constructor_contract:
                designated = [
                    element
                    for element in expr.elements
                    if isinstance(element, DesignatedInitializerNode)
                ]
                if designated:
                    raise MetalConstructorContractError(
                        expected_type,
                        (),
                        [
                            self.constructor_candidate_signature(constructor)
                            for constructor in constructor_contract[1]
                        ],
                        "designated fields cannot select an explicit constructor",
                        getattr(expr, "source_location", None),
                    )
                return self.generate_explicit_constructor_call(
                    expected_type,
                    list(expr.elements),
                    is_main,
                    getattr(expr, "source_location", None),
                    receiver_address_space,
                )
            return self.generate_initializer_list(
                expr, is_main, expected_type, expected_array
            )
        value = self.generate_expression(expr, is_main)
        if not expected_array:
            value = convert_atomic_result(self, expr, value, expected_type)
        if self.binary16_widening_profile is not None and not expected_array:
            value = self.render_metal_half_widening(
                value,
                self.expression_metal_type(expr),
                expected_type,
                getattr(expr, "source_location", None),
            )
        return value

    def generate_initializer_list(
        self, node, is_main=False, expected_type=None, expected_array=False
    ):
        wide_vector = self.wide_vector_type_info(
            expected_type, getattr(node, "source_location", None)
        )
        if wide_vector is not None:
            return self.generate_wide_vector_constructor(
                expected_type,
                node.elements,
                is_main,
                getattr(node, "source_location", None),
                braced=True,
            )
        mapped_type = self.map_type(expected_type) if expected_type else None
        member_types = {}
        member_names = {}
        if expected_type:
            resolved_type = self.resolve_type_alias(expected_type)
            member_types = self.struct_member_types.get(
                self.normalized_metal_type(resolved_type), {}
            )
            member_names = self.struct_member_name_maps.get(
                self.map_type(resolved_type), {}
            )

        named_elements = []
        positional_elements = []
        for index, element in enumerate(node.elements):
            if isinstance(element, DesignatedInitializerNode):
                named_elements.append(
                    self.generate_designated_initializer(
                        element,
                        is_main,
                        member_types,
                        member_names,
                    )
                )
            else:
                element_type = None
                if self.binary16_widening_profile is not None:
                    vector = self.metal_small_vector_type_parts(expected_type)
                    if expected_array:
                        element_type = expected_type
                    elif vector is not None:
                        element_type = vector[0]
                    elif index < len(member_types):
                        element_type = list(member_types.values())[index]
                    else:
                        element_type = expected_type
                positional_elements.append(
                    self.generate_initializer_value(element, is_main, element_type)
                )

        elements = named_elements + positional_elements
        if expected_array:
            return "{" + ", ".join(elements) + "}"
        if self.metal_small_vector_type_parts(expected_type) is not None:
            # Keep list initialization distinct from scalar splat construction.
            # Parentheses prevent identifiers from becoming field shorthands.
            values = ", ".join(f"({element})" for element in elements)
            return f"{mapped_type}{{{values}{',' if elements else ''}}}"
        if mapped_type and mapped_type.startswith(("vec", "ivec", "uvec", "bvec")):
            return f"{mapped_type}({', '.join(elements)})"
        if mapped_type and named_elements:
            return f"{mapped_type}{{{', '.join(elements)}}}"
        if mapped_type and positional_elements and not named_elements:
            return f"{mapped_type}({', '.join(elements)})"
        return "{" + ", ".join(elements) + "}"

    def generate_designated_initializer(
        self,
        node,
        is_main=False,
        member_types=None,
        member_names=None,
    ):
        member_types = member_types or {}
        member_names = member_names or {}
        if len(node.designators) == 1 and node.designators[0][0] == "field":
            field_name = node.designators[0][1]
            value = self.generate_initializer_value(
                node.value, is_main, member_types.get(field_name)
            )
            emitted_name = member_names.get(
                field_name, self.sanitize_identifier(field_name)
            )
            return f"{emitted_name}: {value}"

        # CrossGL has field designators only; array designators are resolved to positions before generation
        # (array_designators), so one left here sits where its array cannot be known.
        return self.refuse_import(
            "Metal designated initializer "
            + "".join(
                f".{designator[1]}" if designator[0] == "field" else "[...]"
                for designator in node.designators
            )
            + " = ... outside an array declaration CrossGL can resolve to positions",
            "0",
        )

    def normalize_literal_string(self, value):
        if "'" in value and re.match(r"^(?:0[xX][0-9a-fA-F]|0[bB][01]|\d|\.\d)", value):
            value = value.replace("'", "")
        if re.fullmatch(
            r"(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?"
            r"|0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)[pP][+-]?\d+)[hH]",
            value,
        ):
            # A half literal keeps its type: `h * 0.5h` is half arithmetic, `h * 0.5` float (MSL section 2.1).
            return f"float16({value[:-1]})"
        if re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)[uUlL]*", value):
            digits, type_name = integer_literal_parts(value, legacy_octal=True)
            suffix = {"int": "", "uint": "u", "int64_t": "l", "uint64_t": "ul"}[
                type_name
            ]
            return digits + suffix
        return value

    def generate_lambda_expression(self, expr, is_main=False):
        del is_main
        raise self.unsupported_callback_error("callback expression", expr)

    def unsupported_callback_error(self, helper, callback):
        return self.callback_lowering_error(
            helper,
            "the callback helper has no semantics-preserving CrossGL lowering",
            callback=callback,
        )

    def callback_lowering_error(
        self, helper, reason, callback=None, source_location=None
    ):
        return MetalCallableLoweringError(
            str(helper),
            reason,
            getattr(callback, "source_location", None) or source_location,
            capture=getattr(callback, "capture", None),
            enclosing_function=self.current_function_name,
            suggested_action=(
                "add a helper-specific lowering that preserves callback invocation "
                "count, compile-time arguments, and capture mutation semantics"
            ),
        )

    def generate_switch_statement(self, node, indent, is_main):
        expression = self.generate_expression(node.expression, is_main)
        previous_variable_types = self.current_variable_types
        previous_variable_type_qualifiers = self.current_variable_type_qualifiers
        previous_storage_texture_names = self.current_storage_texture_names
        previous_structured_buffer_names = self.current_structured_buffer_names
        self.current_variable_types = dict(previous_variable_types)
        self.current_variable_type_qualifiers = dict(previous_variable_type_qualifiers)
        self.current_storage_texture_names = set(previous_storage_texture_names)
        self.current_structured_buffer_names = set(previous_structured_buffer_names)
        previous_type_aliases = dict(self.type_aliases)
        previous_type_alias_qualifiers = dict(self.type_alias_qualifiers)
        previous_type_alias_pointee_qualifiers = dict(
            self.type_alias_pointee_qualifiers
        )
        previous_local_type_alias_names = set(self.local_type_alias_names)
        previous_unresolved_local_type_aliases = dict(
            self.unresolved_local_type_aliases
        )
        previous_local_struct_type_aliases = dict(self.local_struct_type_aliases)
        previous_local_integral_constant_bindings = dict(
            self.local_integral_constant_bindings
        )
        code = f"switch ({expression}) {{\n"
        try:
            # The parser keeps each clause's own jumps, so a clause without one
            # still falls through into the next.
            for value, statements in ordered_switch_clauses(node):
                label = (
                    "default"
                    if value is None
                    else f"case {self.generate_expression(value, is_main)}"
                )
                code += "    " * (indent + 1) + f"{label}:\n"
                code += self.generate_function_body(statements, indent + 2, is_main)

            code += "    " * indent + "}\n"
            return code
        finally:
            self.current_variable_types = previous_variable_types
            self.current_variable_type_qualifiers = previous_variable_type_qualifiers
            self.current_storage_texture_names = previous_storage_texture_names
            self.current_structured_buffer_names = previous_structured_buffer_names
            self.type_aliases = previous_type_aliases
            self.type_alias_qualifiers = previous_type_alias_qualifiers
            self.type_alias_pointee_qualifiers = previous_type_alias_pointee_qualifiers
            self.local_type_alias_names = previous_local_type_alias_names
            self.unresolved_local_type_aliases = previous_unresolved_local_type_aliases
            self.local_struct_type_aliases = previous_local_struct_type_aliases
            self.local_integral_constant_bindings = (
                previous_local_integral_constant_bindings
            )
