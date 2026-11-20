"""Statements in the Mojo generator: assignments, dimension queries, increments, switches, branches and loops."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    RangeNode,
    ReturnNode,
    SwitchNode,
    SyncNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import MojoRefusal


class MojoStatementMixin:
    """Statement members of ``MojoCodeGen``."""

    def generate_expression_with_prelude(
        self, expr, indent, target_type=None, target_context=None
    ):
        self.expression_prelude_stack.append({"indent": indent, "lines": []})
        try:
            expression = self.generate_expression(expr, target_type, target_context)
            prelude = "".join(self.expression_prelude_stack[-1]["lines"])
            return prelude, expression
        finally:
            self.expression_prelude_stack.pop()

    def generate_assignment_statement(self, node, indent):
        swizzle_assignment = self.generate_swizzle_assignment_statement(node, indent)
        if swizzle_assignment is not None:
            return swizzle_assignment

        indent_str = "    " * indent
        self.validate_resource_write_access(node.left, "assignment")
        left = self.generate_expression(node.left)
        left_type = self.expression_result_type(node.left)
        self.validate_expression_target_shape(
            node.right, left_type, "assignment target"
        )
        prelude, right = self.generate_expression_with_prelude(
            node.right, indent, left_type, "assignment target"
        )
        self.validate_scalar_target_conversion(
            node.right, left_type, "assignment target"
        )
        op = self.map_operator(node.operator)
        self.register_resource_assignment_alias_metadata(node.left, node.right, op)
        return f"{prelude}{indent_str}{left} {op} {right}\n"

    def generate_swizzle_assignment_statement(self, node, indent):
        target = getattr(node, "left", getattr(node, "target", None))
        if not isinstance(target, MemberAccessNode):
            return None
        swizzle_indices = self.get_swizzle_indices(getattr(target, "member", ""))
        if swizzle_indices is None or len(swizzle_indices) <= 1:
            return None

        object_type = self.expression_result_type(target.object)
        if self.vector_type_info(object_type) is None:
            return None

        indent_str = "    " * indent
        self.validate_resource_write_access(target, "assignment")
        left_type = self.expression_result_type(target)
        self.validate_expression_target_shape(
            node.right, left_type, "assignment target"
        )
        prelude, right = self.generate_expression_with_prelude(
            node.right, indent, left_type, "assignment target"
        )
        op = self.map_operator(node.operator)
        object_expr = self.generate_expression(target.object)
        temp_name = self.next_swizzle_assignment_temp_name()
        code = f"{prelude}{indent_str}var {temp_name}: {self.map_type(left_type)} = {right}\n"
        for component_index, source_index in enumerate(swizzle_indices):
            code += (
                f"{indent_str}{object_expr}[{source_index}] {op} "
                f"{temp_name}[{component_index}]\n"
            )
        return code

    def generate_get_dimensions_statement(self, stmt, indent):
        call = self.get_expression_statement_call(stmt)
        if call is None:
            return None

        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if not isinstance(func_expr, MemberAccessNode):
            return None
        if getattr(func_expr, "member", None) != "GetDimensions":
            return None

        args = list(getattr(call, "args", []) or [])
        if not args:
            return None

        resource_expr = getattr(
            func_expr, "object", getattr(func_expr, "object_expr", None)
        )
        raw_resource_type = self.expression_result_type(resource_expr)
        resource_type = self.map_type(raw_resource_type)
        buffer_info = self.buffer_resource_info(raw_resource_type)

        if buffer_info is not None:
            return self.generate_buffer_get_dimensions_statement(
                resource_expr, args, buffer_info, indent
            )
        if self.is_texture_resource_type(resource_type):
            return self.generate_texture_get_dimensions_statement(
                resource_expr, args, resource_type, indent
            )
        if self.is_image_resource_type(resource_type):
            return self.generate_image_get_dimensions_statement(
                resource_expr, args, resource_type, indent
            )
        return None

    def get_get_dimensions_out_arguments(self, call):
        args = list(getattr(call, "args", []) or [])
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if not isinstance(func_expr, MemberAccessNode):
            return args
        if getattr(func_expr, "member", None) != "GetDimensions":
            return args

        resource_expr = getattr(
            func_expr, "object", getattr(func_expr, "object_expr", None)
        )
        raw_resource_type = self.expression_result_type(resource_expr)
        resource_type = self.map_type(raw_resource_type)

        if self.buffer_resource_info(raw_resource_type) is not None:
            return args
        if self.is_image_resource_type(resource_type):
            return args
        if not self.is_texture_resource_type(resource_type):
            return args

        dimension_count = self.resource_dimension_component_count(resource_type)
        if dimension_count is None:
            return args
        if self.is_multisample_resource_type(resource_type):
            return args
        if len(args) == dimension_count + 2:
            return args[1:]
        return args

    def get_expression_statement_call(self, stmt):
        if isinstance(stmt, FunctionCallNode):
            return stmt
        if (
            hasattr(stmt, "__class__")
            and "ExpressionStatement" in str(stmt.__class__)
            and isinstance(getattr(stmt, "expression", None), FunctionCallNode)
        ):
            return stmt.expression
        return None

    def generate_buffer_get_dimensions_statement(
        self, resource_expr, args, buffer_info, indent
    ):
        if len(args) not in {1, 2}:
            buffer_type, _ = buffer_info
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                f"{buffer_type} expected 1 or 2 out parameter(s), got {len(args)}"
            )

        buffer_type, _ = buffer_info
        self.register_buffer_resource_type(buffer_type)
        self.required_buffer_dimensions_helpers.add(buffer_info)

        obj = self.generate_expression(resource_expr)
        indent_str = "    " * indent
        size_temp = self.next_dimension_query_temp_name()
        code = f"{indent_str}var {size_temp} = buffer_dimensions({obj})\n"
        for index, target in enumerate(args):
            value = f"{size_temp}[{index}]"
            code += self.generate_dimension_target_assignment(target, value, indent)
        return code

    def generate_texture_get_dimensions_statement(
        self, resource_expr, args, resource_type, indent
    ):
        dimension_count = self.resource_dimension_component_count(resource_type)
        if dimension_count is None:
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                f"sizable texture resource required: {resource_type}"
            )

        if self.is_multisample_resource_type(resource_type):
            if len(args) not in {dimension_count, dimension_count + 1}:
                raise MojoRefusal(
                    "Unsupported GetDimensions overload for Mojo codegen; "
                    f"{resource_type} expected {dimension_count} dimension "
                    f"out parameter(s) plus optional sample count, got {len(args)}"
                )
            dimension_targets = args[:dimension_count]
            sample_target = (
                args[dimension_count] if len(args) > dimension_count else None
            )
            return self.generate_resource_dimension_assignments(
                resource_expr,
                resource_type,
                "texture_size",
                dimension_targets,
                indent,
                sample_target=sample_target,
                sample_helper="texture_samples",
            )

        if len(args) == dimension_count:
            lod_arg = None
            dimension_targets = args
            levels_target = None
        elif len(args) == dimension_count + 1:
            lod_arg = None
            dimension_targets = args[:dimension_count]
            levels_target = args[dimension_count]
        elif len(args) == dimension_count + 2:
            lod_arg = args[0]
            dimension_targets = args[1 : 1 + dimension_count]
            levels_target = args[-1]
        else:
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                f"{resource_type} expected {dimension_count}, "
                f"{dimension_count + 1}, or {dimension_count + 2} argument(s), "
                f"got {len(args)}"
            )

        return self.generate_resource_dimension_assignments(
            resource_expr,
            resource_type,
            "texture_size",
            dimension_targets,
            indent,
            lod_arg=lod_arg,
            levels_target=levels_target,
        )

    def generate_image_get_dimensions_statement(
        self, resource_expr, args, resource_type, indent
    ):
        dimension_count = self.resource_dimension_component_count(resource_type)
        if dimension_count is None:
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                f"sizable image resource required: {resource_type}"
            )

        if self.is_multisample_resource_type(resource_type):
            if len(args) not in {dimension_count, dimension_count + 1}:
                raise MojoRefusal(
                    "Unsupported GetDimensions overload for Mojo codegen; "
                    f"{resource_type} expected {dimension_count} dimension "
                    f"out parameter(s) plus optional sample count, got {len(args)}"
                )
            dimension_targets = args[:dimension_count]
            sample_target = (
                args[dimension_count] if len(args) > dimension_count else None
            )
        else:
            if len(args) != dimension_count:
                raise MojoRefusal(
                    "Unsupported GetDimensions overload for Mojo codegen; "
                    f"{resource_type} expected {dimension_count} dimension "
                    f"out parameter(s), got {len(args)}"
                )
            dimension_targets = args
            sample_target = None

        return self.generate_resource_dimension_assignments(
            resource_expr,
            resource_type,
            "image_size",
            dimension_targets,
            indent,
            sample_target=sample_target,
            sample_helper="image_samples",
        )

    def resource_dimension_component_count(self, resource_type):
        base_type = self.image_resource_base_type(resource_type)
        if not isinstance(base_type, str):
            return None
        if "1DArray" in base_type:
            return 2
        if "1D" in base_type:
            return 1
        if "2DMSArray" in base_type:
            return 3
        if "2DArray" in base_type:
            return 3
        if "2D" in base_type:
            return 2
        if "3D" in base_type:
            return 3
        if "CubeArray" in base_type:
            return 3
        if "Cube" in base_type:
            return 2
        return None

    def generate_resource_dimension_assignments(
        self,
        resource_expr,
        resource_type,
        size_helper,
        dimension_targets,
        indent,
        lod_arg=None,
        levels_target=None,
        sample_target=None,
        sample_helper=None,
    ):
        size_type = self.resource_size_return_type(resource_type)
        if size_type is None:
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                f"sizable resource required: {resource_type}"
            )

        size_expr = self.resource_dimension_size_expression(
            resource_expr, resource_type, size_helper, lod_arg
        )
        indent_str = "    " * indent
        code = ""

        if len(dimension_targets) == 1 and not levels_target and not sample_target:
            component_values = [size_expr]
        else:
            size_temp = self.next_dimension_query_temp_name()
            code += f"{indent_str}var {size_temp} = {size_expr}\n"
            component_values = [
                self.dimension_component_expression(size_temp, size_type, index)
                for index, _ in enumerate(dimension_targets)
            ]

        for target, value in zip(dimension_targets, component_values):
            code += self.generate_dimension_target_assignment(target, value, indent)

        if levels_target is not None:
            levels_expr = self.generate_resource_query_levels_call([resource_expr])
            code += self.generate_dimension_target_assignment(
                levels_target, levels_expr, indent
            )

        if sample_target is not None:
            samples_expr = self.generate_resource_sample_count_call(
                [resource_expr], sample_helper
            )
            code += self.generate_dimension_target_assignment(
                sample_target, samples_expr, indent
            )

        return code

    def resource_dimension_size_expression(
        self, resource_expr, resource_type, helper_name, lod_arg=None
    ):
        self.required_resource_size_types.add(resource_type)
        self.required_resource_size_helpers.add((helper_name, resource_type))
        resource = self.generate_expression(resource_expr)
        if lod_arg is None:
            return f"{helper_name}({resource})"
        lod = self.dimension_lod_expression(lod_arg, resource_type)
        return f"{helper_name}({resource}, {lod})"

    def dimension_lod_expression(self, lod_arg, resource_type):
        lod_type = self.expression_result_type(lod_arg)
        if not self.is_scalar_integer_type(lod_type):
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                f"lod argument for {resource_type} must be an integer scalar"
            )
        lod = self.generate_expression(lod_arg)
        if self.map_type(lod_type) == "Int32":
            return lod
        return f"Int32({lod})"

    def dimension_component_expression(self, size_expr, size_type, index):
        if size_type == "Int32":
            if index == 0:
                return size_expr
            raise MojoRefusal(
                "Unsupported GetDimensions overload for Mojo codegen; "
                "dimension component index exceeds scalar size result"
            )
        return f"{size_expr}[{index}]"

    def generate_dimension_target_assignment(self, target, value, indent):
        self.validate_resource_write_access(target, "GetDimensions")
        target_expr = self.generate_expression(target)
        assignment_value = self.cast_dimension_assignment_value(target, value)
        return f"{'    ' * indent}{target_expr} = {assignment_value}\n"

    def cast_dimension_assignment_value(self, target, value):
        target_type = self.expression_result_type(target)
        if target_type is None:
            return value
        if not self.is_scalar_integer_type(target_type):
            raise MojoRefusal(
                "Unsupported GetDimensions out parameter for Mojo codegen; "
                f"integer scalar target required: {target_type}"
            )
        mapped_type = self.map_type(target_type)
        if mapped_type == "Int32":
            return value
        return f"{mapped_type}({value})"

    def next_dimension_query_temp_name(self):
        name = f"__cgl_dimensions_{self.dimension_query_counter}"
        self.dimension_query_counter += 1
        return name

    def next_scalar_atomic_temp_name(self):
        name = f"__cgl_atomic_{self.scalar_atomic_counter}"
        self.scalar_atomic_counter += 1
        return name

    def next_swizzle_assignment_temp_name(self):
        name = f"__cgl_swizzle_{self.swizzle_assignment_counter}"
        self.swizzle_assignment_counter += 1
        return name

    def next_increment_expression_temp_name(self):
        name = f"__cgl_increment_{self.increment_expression_counter}"
        self.increment_expression_counter += 1
        return name

    def is_increment_or_decrement_node(self, node):
        return isinstance(node, UnaryOpNode) and self.map_operator(
            getattr(node, "operator", getattr(node, "op", ""))
        ) in {"++", "--"}

    def increment_assignment_operator(self, node):
        op = self.map_operator(getattr(node, "operator", getattr(node, "op", "")))
        return "+=" if op == "++" else "-="

    def generate_increment_statement(self, node, indent):
        indent_str = "    " * indent
        operand = getattr(node, "operand", "")
        self.validate_resource_write_access(operand, "increment/decrement")
        target = self.generate_expression(operand)
        return f"{indent_str}{target} {self.increment_assignment_operator(node)} 1\n"

    def generate_increment_expression(self, expr):
        if not self.expression_prelude_stack:
            raise MojoRefusal(
                "Unsupported increment/decrement expression for Mojo codegen; "
                "statement prelude context is required"
            )

        operand = getattr(expr, "operand", "")
        self.validate_resource_write_access(operand, "increment/decrement")
        target = self.generate_expression(operand)
        assignment_op = self.increment_assignment_operator(expr)
        context = self.expression_prelude_stack[-1]
        indent_str = "    " * context["indent"]
        temp_name = self.next_increment_expression_temp_name()
        target_type = self.expression_result_type(operand) or "int"

        if getattr(expr, "is_postfix", False):
            context["lines"].append(
                f"{indent_str}var {temp_name}: {self.map_type(target_type)} = "
                f"{target}\n"
            )
            context["lines"].append(f"{indent_str}{target} {assignment_op} 1\n")
            return temp_name

        context["lines"].append(f"{indent_str}{target} {assignment_op} 1\n")
        context["lines"].append(
            f"{indent_str}var {temp_name}: {self.map_type(target_type)} = {target}\n"
        )
        return temp_name

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL statement as Mojo code."""
        indent_str = "    " * indent

        lowered_get_dimensions = self.generate_get_dimensions_statement(stmt, indent)
        if lowered_get_dimensions is not None:
            return lowered_get_dimensions

        if isinstance(stmt, VariableNode):
            var_type = self.variable_declared_type(stmt)
            if getattr(stmt, "vtype", None) and var_type is not None:
                # Old AST structure - check if this is actually an array declaration disguised as a variable
                vtype_str = str(stmt.vtype)
                if (
                    "ArrayAccessNode" in vtype_str
                    and "array=" in vtype_str
                    and "index=" in vtype_str
                ):
                    # This is likely an array declaration
                    array_match = re.search(r"array=(\w+).*?index=(\w+)", vtype_str)
                    if array_match:
                        array_match.group(1)
                        size = array_match.group(2)
                        base_type = "Float32"  # Default, could be improved
                        return (
                            f"{indent_str}var {self.mojo_identifier(stmt.name)} = "
                            f"InlineArray[{base_type}, {size}]"
                            "(unsafe_uninitialized=True)\n"
                        )

            if hasattr(stmt, "initial_value") and stmt.initial_value is not None:
                effective_type = var_type or self.expression_result_type(
                    stmt.initial_value
                )
                self.register_variable_type(
                    stmt.name,
                    effective_type,
                )
                self.register_resource_access_metadata(stmt, var_type)
                self.register_resource_access_alias_metadata(
                    stmt.name, effective_type, stmt.initial_value
                )
                increment_init = self.generate_increment_initializer_declaration(
                    stmt,
                    stmt.initial_value,
                    var_type,
                    indent,
                )
                if increment_init is not None:
                    return increment_init
                self.validate_expression_target_shape(
                    stmt.initial_value, var_type, f"declaration {stmt.name}"
                )
                prelude, init_expr = self.generate_expression_with_prelude(
                    stmt.initial_value,
                    indent,
                    var_type,
                    f"declaration {stmt.name}",
                )
                self.validate_scalar_target_conversion(
                    stmt.initial_value, var_type, f"declaration {stmt.name}"
                )
                var_name = self.mojo_identifier(stmt.name)
                if var_type is None:
                    return f"{prelude}{indent_str}var {var_name} = {init_expr}\n"
                return (
                    f"{prelude}{indent_str}var {var_name}: "
                    f"{self.map_type(var_type)} = {init_expr}\n"
                )

            var_type = var_type or "float"
            self.register_variable_type(stmt.name, var_type)
            self.register_resource_access_metadata(stmt, var_type)
            var_name = self.mojo_identifier(stmt.name)
            if self.is_array_type_name(var_type):
                return (
                    f"{indent_str}var {var_name} = "
                    f"{self.array_initial_value_for_type(var_type)}\n"
                )
            elif self.is_struct_type_name(var_type):
                return (
                    f"{indent_str}var {var_name} = "
                    f"{self.zero_value_for_type(var_type)}\n"
                )
            else:
                return f"{indent_str}var {var_name}: {self.map_type(var_type)}\n"
        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)
        elif isinstance(stmt, AssignmentNode):
            return self.generate_assignment_statement(stmt, indent)
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)
        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)
        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)
        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)
        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)
        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            if isinstance(stmt.value, list):
                # Multiple return values
                values = ", ".join(self.generate_expression(val) for val in stmt.value)
                return f"{indent_str}return {values}\n"
            else:
                self.validate_expression_target_shape(
                    stmt.value, self.current_return_type, "return value"
                )
                prelude, return_value = self.generate_expression_with_prelude(
                    stmt.value,
                    indent,
                    self.current_return_type,
                    "return value",
                )
                self.validate_scalar_target_conversion(
                    stmt.value, self.current_return_type, "return value"
                )
                return f"{prelude}{indent_str}return {return_value}\n"
        elif isinstance(stmt, BreakNode):
            self.record_loop_exit_alias_state()
            context = self.active_do_while_context()
            if context:
                break_flag = context["break_flag"]
                return f"{indent_str}{break_flag} = True\n{indent_str}break\n"
            return f"{indent_str}break\n"
        elif isinstance(stmt, ContinueNode):
            if self.active_do_while_context():
                return f"{indent_str}break\n"
            context = self.active_for_context()
            if context:
                update = self.generate_for_update_statement(context["update"], indent)
                return f"{update}{indent_str}continue\n"
            return f"{indent_str}continue\n"
        elif isinstance(stmt, SyncNode):
            return self.generate_sync_node(stmt, indent)
        elif self.is_increment_or_decrement_node(stmt):
            return self.generate_increment_statement(stmt, indent)
        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode should not appear as a statement by itself - it's likely a misclassified array declaration
            # Try to handle it gracefully
            return f"{indent_str}# Unhandled ArrayAccessNode: {stmt}\n"
        elif isinstance(stmt, ExpressionStatementNode):
            expr = getattr(stmt, "expression", None)
            if isinstance(expr, AssignmentNode):
                return self.generate_assignment_statement(expr, indent)
            if self.is_increment_or_decrement_node(expr):
                return self.generate_increment_statement(expr, indent)

            prelude, expr_result = self.generate_expression_with_prelude(expr, indent)
            if expr_result.strip():
                return f"{prelude}{indent_str}{expr_result}\n"
            return f"{indent_str}# Unhandled statement: {type(stmt).__name__}\n"
        else:
            # Handle expressions that may be used as statements
            prelude, expr_result = self.generate_expression_with_prelude(stmt, indent)
            if expr_result.strip():
                return f"{prelude}{indent_str}{expr_result}\n"
            else:
                return f"{indent_str}# Unhandled statement: {type(stmt).__name__}\n"

    def generate_increment_initializer_declaration(
        self,
        stmt,
        initial_value,
        var_type,
        indent,
    ):
        if not isinstance(initial_value, UnaryOpNode):
            return None

        op = self.map_operator(
            getattr(initial_value, "operator", getattr(initial_value, "op", ""))
        )
        if op not in {"++", "--"}:
            return None

        operand = self.generate_expression(getattr(initial_value, "operand", ""))
        assignment_op = "+=" if op == "++" else "-="
        indent_str = "    " * indent
        update = f"{indent_str}{operand} {assignment_op} 1\n"
        var_name = self.mojo_identifier(stmt.name)
        if var_type is None:
            declaration = f"{indent_str}var {var_name} = {operand}\n"
        else:
            declaration = (
                f"{indent_str}var {var_name}: {self.map_type(var_type)} = {operand}\n"
            )
        is_postfix = getattr(
            initial_value,
            "is_postfix",
            getattr(initial_value, "postfix", False),
        )
        if is_postfix:
            return declaration + update
        return update + declaration

    def generate_switch(self, node, indent):
        indent_str = "    " * indent
        expression = self.generate_expression(getattr(node, "expression", ""))
        code = ""
        emitted_condition = False
        default_body = None

        for case in getattr(node, "cases", []) or []:
            if not isinstance(case, CaseNode):
                continue
            value = getattr(case, "value", None)
            if value is None:
                default_body = getattr(case, "statements", [])
                continue

            keyword = "if" if not emitted_condition else "elif"
            condition = f"{expression} == {self.generate_expression(value)}"
            code += f"{indent_str}{keyword} {condition}:\n"
            code += self.generate_switch_case_body(
                getattr(case, "statements", []), indent + 1
            )
            emitted_condition = True

        explicit_default = getattr(node, "default_case", None)
        if explicit_default is not None:
            default_body = explicit_default

        if default_body is not None:
            code += f"{indent_str}else:\n"
            code += self.generate_switch_case_body(default_body, indent + 1)
        elif not emitted_condition:
            code += f"{indent_str}pass\n"

        return code

    def generate_switch_case_body(self, body, indent):
        statements = self.statement_list(body)
        code = ""
        for stmt in statements:
            if isinstance(stmt, BreakNode):
                continue
            code += self.generate_statement(stmt, indent)
        if not code:
            code = f"{'    ' * indent}pass\n"
        return code

    def statement_list(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def active_do_while_context(self):
        if not self.do_while_contexts:
            return None
        context = self.do_while_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def active_for_context(self):
        if not self.for_contexts:
            return None
        context = self.for_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def push_loop_exit_alias_state_scope(self):
        self.loop_exit_alias_state_stack.append(
            {"loop_depth": self.loop_depth, "states": []}
        )

    def pop_loop_exit_alias_state_scope(self):
        if not self.loop_exit_alias_state_stack:
            return []
        return self.loop_exit_alias_state_stack.pop()["states"]

    def record_loop_exit_alias_state(self):
        if not self.loop_exit_alias_state_stack:
            return
        scope = self.loop_exit_alias_state_stack[-1]
        if scope["loop_depth"] != self.loop_depth:
            return
        scope["states"].append(self.resource_access_alias_state())

    def statement_body_terminates_inner_loop(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        variable_type = self.array_declared_type(node)
        self.register_variable_type(node.name, variable_type)
        resource_comment = self.generate_resource_metadata_comment(node, variable_type)
        return (
            f"{resource_comment}{indent_str}var {self.mojo_identifier(node.name)} = "
            f"{self.array_initial_value_for_type(variable_type)}\n"
        )

    def generate_assignment(self, node):
        self.validate_resource_write_access(node.left, "assignment")
        left = self.generate_expression(node.left)
        left_type = self.expression_result_type(node.left)
        self.validate_expression_target_shape(
            node.right, left_type, "assignment target"
        )
        right = self.generate_expression(node.right, left_type, "assignment target")
        self.validate_scalar_target_conversion(
            node.right, left_type, "assignment target"
        )
        op = self.map_operator(node.operator)
        return f"{left} {op} {right}"

    def generate_branch_statements(self, branch, indent):
        code = ""
        if hasattr(branch, "statements"):
            for stmt in branch.statements:
                code += self.generate_statement(stmt, indent)
        elif isinstance(branch, list):
            for stmt in branch:
                code += self.generate_statement(stmt, indent)
        elif branch is not None:
            code += self.generate_statement(branch, indent)
        return code

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition_expr = (
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        condition = self.generate_expression(condition_expr)
        self.validate_condition_scalar(condition_expr, "if condition")
        entry_alias_state = self.resource_access_alias_state()
        code = f"{indent_str}if {condition}:\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        self.restore_resource_access_alias_state(entry_alias_state)
        code += self.generate_branch_statements(if_body, indent + 1)
        then_alias_state = self.resource_access_alias_state()

        else_alias_state = entry_alias_state
        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            self.restore_resource_access_alias_state(entry_alias_state)
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                remaining_code = self.generate_if(else_branch, indent)
                remaining_lines = remaining_code.splitlines(keepends=True)
                if remaining_lines and remaining_lines[0].strip().startswith("if "):
                    remaining_lines[0] = remaining_lines[0].replace("if ", "elif ", 1)
                code += "".join(remaining_lines)
            else:
                code += f"{indent_str}else:\n"
                code += self.generate_branch_statements(else_branch, indent + 1)
            else_alias_state = self.resource_access_alias_state()

        self.restore_resource_access_alias_state(
            self.merge_resource_access_alias_states(
                then_alias_state,
                else_alias_state,
            )
        )
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        if isinstance(node.init, list):
            # ``int i = 0, j = 10`` and ``k = 0, m = 3`` run in source order before the loop.
            code = "".join(
                f"{indent_str}{self.generate_statement(item, 0).strip()}\n"
                for item in node.init
                if item is not None
            )
        else:
            init = self.generate_statement(node.init, 0).strip()
            code = f"{indent_str}{init}\n"
        condition = self.generate_expression(node.condition)
        self.validate_condition_scalar(node.condition, "for condition")
        entry_alias_state = self.resource_access_alias_state()

        code += f"{indent_str}while {condition}:\n"

        self.loop_depth += 1
        self.for_contexts.append({"loop_depth": self.loop_depth, "update": node.update})
        self.push_loop_exit_alias_state_scope()
        loop_exit_alias_states = []
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
        finally:
            loop_exit_alias_states = self.pop_loop_exit_alias_state_scope()
            self.for_contexts.pop()
            self.loop_depth -= 1

        code += self.generate_for_update_statement(node.update, indent + 1)
        update_alias_state = self.resource_access_alias_state()
        self.restore_resource_access_alias_state(
            self.merge_resource_access_alias_states(
                entry_alias_state,
                update_alias_state,
                *loop_exit_alias_states,
            )
        )

        return code

    def generate_for_update_statement(self, update, indent):
        if update is None:
            return ""

        if isinstance(update, list):
            # ``i++, j--``: the updates run left to right.
            return "".join(
                self.generate_for_update_statement(item, indent) for item in update
            )

        if isinstance(update, ExpressionStatementNode):
            update = update.expression

        if isinstance(update, AssignmentNode):
            return self.generate_assignment_statement(update, indent)

        if self.is_increment_or_decrement_node(update):
            return self.generate_increment_statement(update, indent)

        indent_str = "    " * indent
        prelude, expression = self.generate_expression_with_prelude(update, indent)
        if expression.strip():
            return f"{prelude}{indent_str}{expression}\n"
        return prelude

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        iterable = self.generate_for_in_iterable(getattr(node, "iterable", None))
        entry_alias_state = self.resource_access_alias_state()

        code = f"{indent_str}for {self.mojo_identifier(pattern)} in {iterable}:\n"

        self.loop_depth += 1
        self.push_loop_exit_alias_state_scope()
        body_alias_state = entry_alias_state
        loop_exit_alias_states = []
        try:
            body_code = ""
            for stmt in self.statement_list(getattr(node, "body", [])):
                body_code += self.generate_statement(stmt, indent + 1)
            body_alias_state = self.resource_access_alias_state()
        finally:
            loop_exit_alias_states = self.pop_loop_exit_alias_state_scope()
            self.loop_depth -= 1

        if body_code:
            code += body_code
        else:
            code += f"{indent_str}    pass\n"

        self.restore_resource_access_alias_state(
            self.merge_resource_access_alias_states(
                entry_alias_state,
                body_alias_state,
                *loop_exit_alias_states,
            )
        )

        return code

    def generate_for_in_iterable(self, iterable_node):
        if isinstance(iterable_node, RangeNode):
            start = self.generate_expression(iterable_node.start)
            end = self.generate_expression(iterable_node.end)
            if iterable_node.inclusive:
                end = f"({end} + 1)"
            return f"range({start}, {end})"

        iterable = self.generate_expression(iterable_node)
        return f"range({iterable})"

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_expression(node.condition)
        self.validate_condition_scalar(node.condition, "while condition")
        entry_alias_state = self.resource_access_alias_state()

        code = f"{indent_str}while {condition}:\n"

        self.loop_depth += 1
        self.push_loop_exit_alias_state_scope()
        body_alias_state = entry_alias_state
        loop_exit_alias_states = []
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
            body_alias_state = self.resource_access_alias_state()
        finally:
            loop_exit_alias_states = self.pop_loop_exit_alias_state_scope()
            self.loop_depth -= 1

        self.restore_resource_access_alias_state(
            self.merge_resource_access_alias_states(
                entry_alias_state,
                body_alias_state,
                *loop_exit_alias_states,
            )
        )

        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}while True:\n"

        self.loop_depth += 1
        self.push_loop_exit_alias_state_scope()
        body_alias_state = self.resource_access_alias_state()
        loop_exit_alias_states = []
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 1)
            body_alias_state = self.resource_access_alias_state()
        finally:
            loop_exit_alias_states = self.pop_loop_exit_alias_state_scope()
            self.loop_depth -= 1

        self.restore_resource_access_alias_state(
            self.merge_resource_access_alias_states(
                *(loop_exit_alias_states or [body_alias_state])
            )
        )

        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        break_flag = f"__cgl_do_break_{self.do_while_counter}"
        self.do_while_counter += 1
        condition = self.generate_expression(node.condition)
        self.validate_condition_scalar(node.condition, "do-while condition")

        code = f"{indent_str}var {break_flag}: Bool = False\n"
        code += f"{indent_str}while True:\n"
        code += f"{indent_str}    while True:\n"

        self.loop_depth += 1
        self.do_while_contexts.append(
            {"loop_depth": self.loop_depth, "break_flag": break_flag}
        )
        self.push_loop_exit_alias_state_scope()
        body_alias_state = self.resource_access_alias_state()
        loop_exit_alias_states = []
        try:
            for stmt in self.statement_list(node.body):
                code += self.generate_statement(stmt, indent + 2)
            body_alias_state = self.resource_access_alias_state()
        finally:
            loop_exit_alias_states = self.pop_loop_exit_alias_state_scope()
            self.do_while_contexts.pop()
            self.loop_depth -= 1

        if not self.statement_body_terminates_inner_loop(node.body):
            code += f"{indent_str}        break\n"
        code += f"{indent_str}    if {break_flag}:\n"
        code += f"{indent_str}        break\n"
        code += f"{indent_str}    if not {condition}:\n"
        code += f"{indent_str}        break\n"

        self.restore_resource_access_alias_state(
            self.merge_resource_access_alias_states(
                body_alias_state,
                *loop_exit_alias_states,
            )
        )

        return code
