"""Statements of the WGSL generator: blocks, if, for and switch statements, assignments, matrix compound assignments, stage interface array and matrix stores, and swizzle assignments."""

import re

from ..ast import (
    AssignmentNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ContinueNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    VariableNode,
    WhileNode,
)
from ..refusals import WGSLRefusal
from .wgsl_boolean_xor import (
    wgsl_boolean_xor_assignment,
    wgsl_boolean_xor_assignment_statement,
)
from .wgsl_buffer_atomics import WGSLCompareExchangeLoop
from .wgsl_integer_operands import (
    BITWISE_OPERATORS,
    MIXED_INTEGER_OPERATORS,
    SHIFT_OPERATORS,
    wgsl_mixed_integer_operands,
    wgsl_shift_amount,
)
from .wgsl_loop_continuing import WGSLContinuingLoopNode
from .wgsl_must_use import wgsl_call_statement
from .wgsl_primitive_index import refuse_primitive_index_write
from .wgsl_switch_clauses import (
    is_discard_statement,
    wgsl_switch_clauses,
)
from .wgsl_value_side_effects import is_increment


class WGSLStatementMixin:
    """Statement members of ``WGSLCodeGen``."""

    def generate_statement(self, stmt, indent=0):
        pad = "    " * indent
        if isinstance(stmt, WGSLCompareExchangeLoop):
            return self.generate_compare_exchange_loop(stmt, indent)
        lowered = self.lower_value_side_effects(stmt)
        if lowered is not None:
            return "\n".join(
                self.generate_statement(statement, indent) for statement in lowered
            )
        lowered = self.lower_effectful_ternary(stmt)
        if lowered is not None:
            return "\n".join(
                self.generate_statement(statement, indent) for statement in lowered
            )
        lowered = self.lower_reference_argument_copies(stmt)
        if lowered is not None:
            return "\n".join(
                self.generate_statement(statement, indent) for statement in lowered
            )
        lowered = self.lower_composite_ternary(stmt)
        if lowered is not None:
            return "\n".join(
                self.generate_statement(statement, indent) for statement in lowered
            )
        if isinstance(stmt, BlockNode):
            return self.generate_block(stmt, indent)
        if isinstance(stmt, VariableNode):
            mutable_keyword = "var" if stmt.is_mutable else "let"
            var_type = self.local_variable_declaration_type(stmt)
            initializer = ""
            if stmt.initial_value is not None:
                initializer = " = " + self.generate_expression_for_target(
                    stmt.initial_value, var_type
                )
            self.refuse_atomic_local(stmt)
            variable_name = self.declare_local_identifier(stmt.name)
            line = (
                f"{pad}{mutable_keyword} {variable_name}: "
                f"{self.type_name_string(var_type)}{initializer};"
            )
            self.register_local_identifier(stmt.name)
            self.register_value_type(stmt.name, var_type)
            self.register_resource_aliases(
                stmt.name, var_type, initializer=stmt.initial_value
            )
            return line
        if isinstance(stmt, ExpressionStatementNode):
            if isinstance(stmt.expression, AssignmentNode):
                return self.generate_assignment_statement(stmt.expression, indent)
            if is_discard_statement(stmt):
                return f"{pad}discard;"
            expression = self.generate_statement_expression(stmt.expression)
            if self.is_multi_barrier_memory_lowering(stmt.expression):
                return "\n".join(f"{pad}{call};" for call in expression.split("; "))
            user_functions = set(self._function_identifier_names.values())
            user_functions.update(self._function_signature_identifier_names.values())
            user_functions.update(self._function_overloads_by_name)
            struct_types = set(self._type_identifier_names.values())
            if not isinstance(
                stmt.expression, (FunctionCallNode, ConstructorNode)
            ) and not is_increment(stmt.expression):
                # WGSL statements are calls, updates and assignments only (WGSL section 9); any other
                # expression is evaluated and its value discarded with a phony assignment (section 9.2.1).
                return f"{pad}_ = {expression};"
            statement = wgsl_call_statement(expression, user_functions, struct_types)
            return f"{pad}{statement};"
        if isinstance(stmt, AssignmentNode):
            return self.generate_assignment_statement(stmt, indent)
        if isinstance(stmt, ReturnNode):
            if stmt.value is None:
                return f"{pad}return;"
            value = self.generate_expression_for_target(
                stmt.value, self._current_function_return_type
            )
            return f"{pad}return {value};"
        if isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        if isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        if isinstance(stmt, WhileNode):
            return f"{pad}while ({self.generate_expression(stmt.condition)}) {self.generate_block(stmt.body, indent)}"
        if isinstance(stmt, LoopNode):
            return f"{pad}loop {self.generate_block(stmt.body, indent)}"
        if isinstance(stmt, BreakNode):
            return f"{pad}break;"
        if isinstance(stmt, ContinueNode):
            return f"{pad}continue;"
        if isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        if isinstance(stmt, WGSLContinuingLoopNode):
            return self.generate_continuing_loop(stmt, indent)
        if isinstance(stmt, ForInNode):
            raise WGSLRefusal("WGSL target does not support for-in statements")
        if isinstance(stmt, MatchNode):
            raise WGSLRefusal("WGSL target does not support match statements")
        raise WGSLRefusal(
            f"WGSL target does not support statement {type(stmt).__name__}"
        )

    def generate_block(self, block, indent=0):
        statements = self.block_statements(block)
        if not statements:
            return "{}"
        pad = "    " * indent
        lines = [f"{pad}{{"]
        self.push_identifier_scope()
        try:
            for stmt in statements:
                lines.append(self.generate_statement(stmt, indent + 1))
        finally:
            self.pop_identifier_scope()
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_if(self, node, indent):
        pad = "    " * indent
        code = (
            f"{pad}if ({self.generate_expression(node.condition)}) "
            f"{self.generate_block(node.then_branch, indent)}"
        )
        if isinstance(node.else_branch, IfNode):
            code += " else " + self.generate_if(node.else_branch, indent).lstrip()
        elif node.else_branch is not None:
            code += f" else {self.generate_block(node.else_branch, indent)}"
        return code

    def generate_for(self, node, indent):
        pad = "    " * indent
        self.push_identifier_scope()
        try:
            init = self.generate_for_initializer(node.init)
            condition = (
                self.generate_expression(node.condition) if node.condition else ""
            )
            update = (
                self.generate_statement_expression(node.update) if node.update else ""
            )
            return (
                f"{pad}for ({init}; {condition}; {update}) "
                f"{self.generate_block(node.body, indent)}"
            )
        finally:
            self.pop_identifier_scope()

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, VariableNode):
            var_type = self.local_variable_declaration_type(init)
            initializer = ""
            if init.initial_value is not None:
                initializer = " = " + self.generate_expression_for_target(
                    init.initial_value, var_type
                )
            variable_name = self.declare_local_identifier(init.name)
            line = (
                f"var {variable_name}: {self.type_name_string(var_type)}"
                f"{initializer}"
            )
            self.register_local_identifier(init.name)
            self.register_value_type(init.name, var_type)
            self.register_resource_aliases(
                init.name, var_type, initializer=init.initial_value
            )
            return line
        if isinstance(init, ExpressionStatementNode):
            return self.generate_statement_expression(init.expression)
        return self.generate_statement_expression(init)

    def generate_statement_expression(self, expr):
        """An expression in statement position, where WGSL allows updates."""
        if isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        if is_increment(expr):
            refuse_primitive_index_write(self, expr.operand)
            operator = "+=" if expr.operator == "++" else "-="
            return f"{self.generate_expression(expr.operand)} {operator} 1"
        return self.generate_expression(expr)

    def generate_switch(self, node, indent):
        pad = "    " * indent
        lines = [f"{pad}switch ({self.generate_expression(node.expression)}) {{"]
        for clause in wgsl_switch_clauses(node):
            selectors = ", ".join(
                "default" if value is None else self.generate_expression(value)
                for value in clause.selectors
            )
            label = "default" if selectors == "default" else f"case {selectors}"
            lines.append(f"{pad}    {label}: {{")
            # Each WGSL clause is its own compound statement.
            self.push_identifier_scope()
            try:
                for stmt in clause.statements:
                    lines.append(self.generate_statement(stmt, indent + 2))
            finally:
                self.pop_identifier_scope()
            lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_assignment(self, node):
        self.validate_storage_assignment_target(node.target)
        refuse_primitive_index_write(self, node.target)
        boolean_xor = wgsl_boolean_xor_assignment(self, node)
        if boolean_xor is not None:
            return boolean_xor
        matrix_scalar = self.generate_matrix_scalar_compound_assignment(node)
        if matrix_scalar is not None:
            return matrix_scalar
        target_type = self.expression_type(node.target)
        stage_matrix = self.stage_interface_matrix_access(node.target)
        if stage_matrix is not None:
            # A for-loop update or statement expression reaches here instead
            # of the statement path, so it needs the same repeatability check.
            self.require_repeatable_matrix_operands(
                "assignment to a split stage-interface member",
                (stage_matrix["object_expr"],),
            )
        target_shape = self.vector_shape(target_type)
        value_type = self.expression_type(node.value)
        value_shape = self.vector_shape(value_type)
        if (
            node.operator[:-1] in MIXED_INTEGER_OPERATORS | BITWISE_OPERATORS
            and target_shape is not None
            and target_shape[0] == "u32"
            and value_shape == ("i32", target_shape[1])
        ):
            left, right = wgsl_mixed_integer_operands(self, node.target, node.value)
            return f"{left} {node.operator} {right}"
        if node.operator[:-1] in SHIFT_OPERATORS:
            value = wgsl_shift_amount(self, node.value, target_type)
        else:
            value = self.generate_expression_for_target(node.value, target_type)
        return f"{self.generate_expression(node.target)} {node.operator} {value}"

    def generate_assignment_statement(self, node, indent):
        stage_array_assignment = (
            self.generate_stage_interface_array_assignment_statement(node, indent)
        )
        if stage_array_assignment is not None:
            return stage_array_assignment
        stage_matrix_assignment = (
            self.generate_stage_interface_matrix_assignment_statement(node, indent)
        )
        if stage_matrix_assignment is not None:
            return stage_matrix_assignment
        swizzle_assignment = self.generate_swizzle_assignment_statement(node, indent)
        if swizzle_assignment is not None:
            return swizzle_assignment
        boolean_xor = wgsl_boolean_xor_assignment_statement(self, node, indent)
        if boolean_xor is not None:
            return boolean_xor
        pad = "    " * indent
        return f"{pad}{self.generate_assignment(node)};"

    def generate_matrix_scalar_compound_assignment(self, node):
        """``m = matCxR<T>(m[0] + s, ...)`` for ``m += s``, ``-=`` or ``/=``, else ``None``.

        generate_assignment calls it after the target's write checks, so a statement
        and a for-loop update are lowered alike.
        """
        operator = node.operator
        if operator not in {"+=", "-=", "/="}:
            return None
        binary_operator = operator[:-1]
        matrix_shape = self.matrix_shape(self.expression_type(node.target))
        if matrix_shape is None:
            return None
        element, columns, rows = matrix_shape
        if self.matrix_shape(self.expression_type(node.value)) is not None:
            # WGSL applies `+=` and `-=` to two matrices component by component,
            # but has no `/` on matrices at all.
            if binary_operator == "/":
                self.refuse_matrix_scalar_binary(node)
            return None
        if not self.is_matrix_scalar_operand(node.value, element):
            # WGSL has no `/` on matrices, and a scalar of another type does not
            # convert to the matrix element type.
            if binary_operator == "/" or self.scalar_type_name(
                self.expression_type(node.value)
            ):
                self.refuse_matrix_scalar_binary(node)
            return None
        self.require_repeatable_matrix_operands(
            f"matrix-scalar {operator}", (node.target, node.value)
        )
        target = self.generate_expression(node.target)
        scalar = self.generate_expression(node.value)
        parts = [
            f"{self.matrix_column_expression(target, index)} {binary_operator} {scalar}"
            for index in range(columns)
        ]
        return f"{target} = mat{columns}x{rows}<{element}>({', '.join(parts)})"

    def generate_stage_interface_array_assignment_statement(self, node, indent):
        target = getattr(node, "target", None)
        access = self.stage_interface_array_access(target)
        if access is None:
            return None

        target_type = self.expression_type(target)
        value = self.generate_expression_for_target(node.value, target_type)
        index = self.resolve_integer_expression(target.index_expr)
        pad = "    " * indent
        if index is not None:
            if index < 0 or index >= access["size"]:
                raise WGSLRefusal(
                    "WGSL target stage interface array "
                    f"{access['member_name']} index {index} is outside the lowered "
                    f"field range 0..{access['size'] - 1}"
                )
            field = self.stage_interface_array_access_field(access, index)
            return f"{pad}{field} {node.operator} {value};"

        index_expr = self.generate_expression(target.index_expr)
        lines = [f"{pad}switch ({index_expr}) {{"]
        for field_index in range(access["size"]):
            field = self.stage_interface_array_access_field(access, field_index)
            lines.append(f"{pad}    case {field_index}: {{")
            lines.append(f"{pad}        {field} {node.operator} {value};")
            lines.append(f"{pad}        break;")
            lines.append(f"{pad}    }}")
        lines.append(f"{pad}    default: {{")
        lines.append(f"{pad}        break;")
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_stage_interface_matrix_assignment_statement(self, node, indent):
        target = getattr(node, "target", None)
        access = self.stage_interface_matrix_access(target)
        if access is None:
            return None

        target_type = self.expression_type(target)
        pad = "    " * indent
        if node.operator != "=":
            compound = self.generate_stage_interface_matrix_compound_assignment(
                node, access, target_type, pad
            )
            if compound is not None:
                return compound
        # A plain assignment still writes one column at a time, so its
        # object must be repeatable too; the compound operators are refused
        # above when their object is not.
        self.require_repeatable_matrix_operands(
            "assignment to a split stage-interface member",
            (access["object_expr"],),
        )
        value = self.generate_expression_for_target(node.value, target_type)
        temp_name = self.declare_synthetic_local_identifier(
            self.swizzle_base_name(access["member_name"])
        )
        lines = [
            f"{pad}let {temp_name}: {self.type_name_string(target_type)} = {value};"
        ]
        for index in range(access["columns"]):
            field = self.stage_interface_matrix_access_field(access, index)
            lines.append(f"{pad}{field} {node.operator} {temp_name}[{index}];")
        return "\n".join(lines)

    def generate_stage_interface_matrix_compound_assignment(
        self, node, access, target_type, pad
    ):
        """Column lowering for ``o.m op= v`` on a split stage-output matrix.

        Returns None when the existing column-wise lowering stays correct:
        ``+=``/``-=`` with a matrix value, an operator other than the four
        arithmetic compound assignments, or a value that is neither a matrix
        nor a scalar. The column-wise lowering is wrong for the other cases:
        a scalar value must not be declared as a matrix temporary (naga
        rejects it), ``*=`` with a matrix is the matrix product rather than
        a component-wise product, and WGSL has no division on matrices.
        """
        operator = node.operator
        if operator not in {"+=", "-=", "*=", "/="}:
            return None
        target_shape = self.matrix_shape(target_type)
        if target_shape is None:
            return None
        # Each column reads and writes the target's object again, so, as for
        # other matrix compound assignments, it must have no side effects.
        self.require_repeatable_matrix_operands(
            f"matrix {operator} on a split stage-interface member",
            (access["object_expr"],),
        )
        element = target_shape[0]
        value_type = self.expression_type(node.value)
        if self.matrix_shape(value_type) is not None:
            if operator == "/=":
                # WGSL has no division on matrices, as for other targets.
                self.refuse_matrix_scalar_binary(node)
            if operator != "*=":
                # WGSL applies `+=` and `-=` to two matrices component by
                # component, so the column-wise lowering stays correct.
                return None
            value = self.generate_expression_for_target(node.value, target_type)
            matrix_type = access["matrix_type_name"]
            columns = ", ".join(
                self.stage_interface_matrix_access_field(access, index)
                for index in range(access["columns"])
            )
            temp_name = self.declare_synthetic_local_identifier(
                self.swizzle_base_name(access["member_name"])
            )
            lines = [
                f"{pad}let {temp_name}: {matrix_type} = "
                f"{matrix_type}({columns}) * {value};"
            ]
            for index in range(access["columns"]):
                field = self.stage_interface_matrix_access_field(access, index)
                lines.append(f"{pad}{field} = {temp_name}[{index}];")
            return "\n".join(lines)
        if self.scalar_type_name(value_type) is None:
            return None
        if not self.is_matrix_scalar_operand(node.value, element):
            self.refuse_matrix_scalar_binary(node)
        # Apply the scalar to each column directly: WGSL has vector-scalar
        # forms of all four operators, and the value is evaluated once into
        # a scalar temporary instead of a matrix temporary.
        scalar = self.generate_expression(node.value)
        temp_name = self.declare_synthetic_local_identifier(
            self.swizzle_base_name(access["member_name"])
        )
        lines = [f"{pad}let {temp_name}: {element} = {scalar};"]
        for index in range(access["columns"]):
            field = self.stage_interface_matrix_access_field(access, index)
            lines.append(f"{pad}{field} {operator} {temp_name};")
        return "\n".join(lines)

    def generate_swizzle_assignment_statement(self, node, indent):
        target = getattr(node, "target", None)
        target_info = self.swizzle_assignment_target_info(target)
        if target_info is None:
            return None
        base, components = target_info
        if not self.assignable_swizzle_components(components):
            return None

        pad = "    " * indent
        target_type = self.expression_type(target)
        # A compound assignment applies a scalar value to each component in turn, so it stays a scalar.
        value = self.generate_expression_for_target(
            node.value, target_type, splat_scalars=node.operator == "="
        )
        value_type = self.expression_type(node.value)

        if node.operator != "=":
            lines = []
            for index, component in enumerate(components):
                rhs = self.swizzle_assignment_value_component(
                    value, value_type, index, len(components)
                )
                lines.append(f"{pad}{base}.{component} {node.operator} {rhs};")
            return "\n".join(lines)

        temp_name = self.declare_synthetic_local_identifier(
            f"{self.swizzle_base_name(base)}_{''.join(components)}"
        )
        lines = [
            f"{pad}let {temp_name}: {self.type_name_string(target_type)} = {value};"
        ]
        for index, component in enumerate(components):
            rhs = self.swizzle_assignment_value_component(
                temp_name, target_type, index, len(components)
            )
            lines.append(f"{pad}{base}.{component} = {rhs};")
        return "\n".join(lines)

    def swizzle_assignment_target_info(self, target):
        if isinstance(target, SwizzleNode) and len(target.components) > 1:
            return self.generate_expression(target.vector_expr), list(target.components)
        if isinstance(target, MemberAccessNode) and len(target.member) > 1:
            object_type = self.expression_type(target.object_expr)
            if self.vector_component_type(object_type) is None:
                return None
            if not self.is_vector_member(target.member):
                return None
            return self.generate_expression(target.object_expr), list(target.member)
        return None

    def assignable_swizzle_components(self, components):
        if len(set(components)) != len(components):
            raise WGSLRefusal(
                "WGSL target cannot lower assignment to swizzle with duplicate "
                f"components {components}"
            )
        return all(component in "xyzw" for component in components)

    def swizzle_assignment_value_component(
        self, value_expression, value_type, component_index, component_count
    ):
        value_component_count = self.vector_component_count(value_type)
        if (
            value_component_count is not None
            and value_component_count >= component_count
        ):
            return f"({value_expression}).{'xyzw'[component_index]}"
        return value_expression

    def swizzle_base_name(self, base):
        return re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "swizzle"

    def declare_synthetic_local_identifier(self, base):
        used_names = set()
        for scope in self._identifier_scopes:
            used_names.update(scope)
        for aliases in self._identifier_alias_scopes:
            used_names.update(aliases.values())
        emitted_name = self.unique_wgsl_identifier(
            self.safe_wgsl_identifier(base), used_names
        )
        if self._identifier_scopes:
            self._identifier_scopes[-1].add(emitted_name)
        return emitted_name
