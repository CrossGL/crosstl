"""Resource and private pointer arrays in the HLSL generator: sampler dimensions, unsized sizes, texture intervals, fixed bindings, coordinates, loop rebasing and arguments."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    PointerReinterpretNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from . import hlsl_private_pointer_call_intervals
from .array_utils import (
    parse_array_type,
    split_array_type_suffix,
)
from .ast_walk import walk_ast_values
from .directx_values import (
    HLSL_PRIVATE_POINTER_BASE_MAX,
    HLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS,
    HLSL_PRIVATE_POINTER_OFFSET_KEY,
    DirectXPrivatePointerParameterError,
)
from .generic_function_utils import generic_function_call_name
from .hlsl_private_member_view_types import hlsl_typed_member_view_root
from .hlsl_texel_buffers import texel_buffer_base_type
from .image_access_contracts import (
    is_image_format_attribute,
    is_integer_coordinate_type_name,
    is_resource_access_attribute,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_offset_operation,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
)
from .index_range_contracts import truncating_division_interval
from .private_array_rows import (
    array_row_shape,
    array_row_view,
)


class DirectXResourceArrayMixin:
    """Resource array members of ``HLSLCodeGen``."""

    def hlsl_private_pointer_condition_value(self, expression, intervals, constants):
        if self.hlsl_private_pointer_expression_has_side_effects(expression):
            return None

        value = self.literal_int_value(expression, constants)
        if value is not None:
            return value != 0
        if self.hlsl_private_pointer_condition_uses_unsafe_intervals(
            expression, intervals, constants
        ):
            return None

        if isinstance(expression, UnaryOpNode) and expression.op == "!":
            operand = self.hlsl_private_pointer_condition_value(
                expression.operand, intervals, constants
            )
            return None if operand is None else not operand

        if isinstance(expression, BinaryOpNode):
            operator = expression.op
            if operator in {"&&", "and", "||", "or"}:
                left = self.hlsl_private_pointer_condition_value(
                    expression.left, intervals, constants
                )
                if operator in {"&&", "and"} and left is False:
                    return False
                if operator in {"||", "or"} and left is True:
                    return True
                right = self.hlsl_private_pointer_condition_value(
                    expression.right, intervals, constants
                )
                if left is None or right is None:
                    return None
                return left and right if operator in {"&&", "and"} else left or right

            if operator in {"==", "!=", "<", "<=", ">", ">="}:
                left = self.hlsl_private_pointer_interval(
                    expression.left, intervals, constants
                )
                right = self.hlsl_private_pointer_interval(
                    expression.right, intervals, constants
                )
                if left is None or right is None:
                    return None
                if operator == "==":
                    if left[1] < right[0] or right[1] < left[0]:
                        return False
                    if left[0] == left[1] == right[0] == right[1]:
                        return True
                elif operator == "!=":
                    if left[1] < right[0] or right[1] < left[0]:
                        return True
                    if left[0] == left[1] == right[0] == right[1]:
                        return False
                elif operator == "<":
                    if left[1] < right[0]:
                        return True
                    if left[0] >= right[1]:
                        return False
                elif operator == "<=":
                    if left[1] <= right[0]:
                        return True
                    if left[0] > right[1]:
                        return False
                elif operator == ">":
                    if left[0] > right[1]:
                        return True
                    if left[1] <= right[0]:
                        return False
                elif operator == ">=":
                    if left[0] >= right[1]:
                        return True
                    if left[1] < right[0]:
                        return False
                return None

        interval = self.hlsl_private_pointer_interval(expression, intervals, constants)
        if interval is not None and interval[0] == interval[1]:
            return interval[0] != 0
        return None

    def hlsl_private_pointer_interval(self, expression, intervals, constants):
        value = self.literal_int_value(expression, constants)
        if value is not None:
            return value, value
        if expression is None:
            return None
        name = self.expression_name(expression)
        if name in intervals:
            return intervals[name]
        if isinstance(expression, UnaryOpNode):
            if expression.op in {"++", "--"}:
                return None
            operand = self.hlsl_private_pointer_interval(
                expression.operand, intervals, constants
            )
            if operand is None:
                return None
            if expression.op == "+":
                return operand
            if expression.op == "-":
                return -operand[1], -operand[0]
            return None
        if isinstance(expression, BinaryOpNode):
            left = self.hlsl_private_pointer_interval(
                expression.left, intervals, constants
            )
            right = self.hlsl_private_pointer_interval(
                expression.right, intervals, constants
            )
            if left is None or right is None:
                return None
            if expression.op == "+":
                return left[0] + right[0], left[1] + right[1]
            if expression.op == "-":
                return left[0] - right[1], left[1] - right[0]
            if expression.op == "*":
                products = (
                    left[0] * right[0],
                    left[0] * right[1],
                    left[1] * right[0],
                    left[1] * right[1],
                )
                return min(products), max(products)
            if expression.op == "/" and right[0] == right[1] and right[0] > 0:
                return truncating_division_interval(left, right)
            return None
        if isinstance(expression, TernaryOpNode):
            condition = self.hlsl_private_pointer_condition_value(
                expression.condition, intervals, constants
            )
            if condition is not None:
                selected = expression.true_expr if condition else expression.false_expr
                return self.hlsl_private_pointer_interval(
                    selected, intervals, constants
                )
            if self.hlsl_private_pointer_expression_has_side_effects(expression):
                return None
            true_interval = self.hlsl_private_pointer_interval(
                expression.true_expr, intervals, constants
            )
            false_interval = self.hlsl_private_pointer_interval(
                expression.false_expr, intervals, constants
            )
            if true_interval is None or false_interval is None:
                return None
            return (
                min(true_interval[0], false_interval[0]),
                max(true_interval[1], false_interval[1]),
            )
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            arguments = list(getattr(expression, "arguments", []) or [])
            if (
                function_name in {
                    "int",
                    "uint",
                    "short",
                    "ushort",
                    "int16",
                    "uint16",
                    "int32_t",
                    "uint32_t",
                    "size_t",
                    "ptrdiff_t",
                }
                and len(arguments) == 1
            ):
                return self.hlsl_private_pointer_interval(
                    arguments[0], intervals, constants
                )
            return hlsl_private_pointer_call_intervals.builtin_interval(
                function_name,
                arguments,
                lambda argument: self.hlsl_private_pointer_interval(
                    argument, intervals, constants
                ),
                lambda argument: (
                    (
                        self.function_call_name(argument),
                        list(getattr(argument, "arguments", []) or []),
                    )
                    if isinstance(argument, FunctionCallNode)
                    else None
                ),
            )
        return None

    def hlsl_private_pointer_array_row(
        self, expression, parameter_names, local_extents, local_types, constants
    ):
        """The row of a fixed multidimensional local array ``expression`` selects, or None."""

        def shape_of(name):
            if name in parameter_names or name not in local_extents:
                return None
            return array_row_shape(
                (local_types or {}).get(name),
                lambda size: self.literal_int_value(size, constants),
            )

        return array_row_view(expression, shape_of)

    def hlsl_private_pointer_static_binding(
        self,
        expression,
        parameter_names,
        local_extents,
        intervals,
        constants,
        local_types=None,
    ):
        local_types = local_types or {}
        if isinstance(expression, PointerReinterpretNode):
            address_space = str(
                getattr(expression.target_type, "address_space", None) or ""
            ).lower()
            if address_space and address_space not in {
                "thread",
                "private",
                "function",
            }:
                return None
            operand = getattr(expression, "expression", None)
            backing = getattr(operand, "operand", None)
            backing_name = getattr(backing, "name", None)
            source_type = local_types.get(backing_name)
            if (
                not isinstance(operand, UnaryOpNode)
                or operand.op != "&"
                or source_type is None
            ):
                return None
            layout = self.hlsl_private_pointer_word_view_layout(expression, source_type)
            return {
                "root_kind": "local_word_view",
                "root": backing_name,
                "member": layout["member"],
                "extent": layout["target_extent"],
                "offset": (0, 0),
                "side_effecting": False,
                "word_view": layout["variant"],
            }
        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            name = (
                expression
                if isinstance(expression, str)
                else getattr(expression, "name", None)
            )
            if name in parameter_names:
                return {
                    "root_kind": "parameter",
                    "root": name,
                    "offset": (
                        (intervals or {}).get(
                            (HLSL_PRIVATE_POINTER_OFFSET_KEY, name), (0, 0)
                        )
                    ),
                    "side_effecting": False,
                }
            if name in local_extents:
                return {
                    "root_kind": "local",
                    "root": name,
                    "extent": local_extents[name],
                    "offset": (0, 0),
                    "side_effecting": False,
                }
            return None
        if isinstance(expression, ArrayAccessNode):
            row = self.hlsl_private_pointer_array_row(
                expression, parameter_names, local_extents, local_types, constants
            )
            if row is None:
                return None
            root_name, extent, _element_type, indices = row
            return {
                "root_kind": "local",
                "root": root_name,
                "extent": extent,
                "offset": (0, 0),
                "side_effecting": any(
                    self.hlsl_private_pointer_expression_has_side_effects(index)
                    for index in indices
                ),
                "row_view": True,
            }
        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            operand = expression.operand
            if isinstance(operand, (IdentifierNode, VariableNode)):
                name = getattr(operand, "name", None)
                if name not in parameter_names and name not in local_extents:
                    return {
                        "root_kind": "scalar",
                        "root": name,
                        "extent": 1,
                        "offset": (0, 0),
                        "side_effecting": False,
                    }
            if isinstance(operand, MemberAccessNode):
                root = hlsl_typed_member_view_root(
                    self,
                    operand,
                    set(parameter_names) | set(local_extents),
                    local_types,
                )
                if root is None:
                    return None
                return {
                    "root_kind": "scalar",
                    "root": root,
                    "extent": 1,
                    "offset": (0, 0),
                    "side_effecting": False,
                    "member_view": True,
                }
            if not isinstance(operand, ArrayAccessNode):
                return None
            binding = self.hlsl_private_pointer_static_binding(
                operand.array,
                parameter_names,
                local_extents,
                intervals,
                constants,
                local_types,
            )
            if binding is None:
                return None
            delta_expression = operand.index
        elif isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            binding = self.hlsl_private_pointer_static_binding(
                expression.left,
                parameter_names,
                local_extents,
                intervals,
                constants,
                local_types,
            )
            delta_expression = expression.right
            if binding is None and expression.op == "+":
                binding = self.hlsl_private_pointer_static_binding(
                    expression.right,
                    parameter_names,
                    local_extents,
                    intervals,
                    constants,
                    local_types,
                )
                delta_expression = expression.left
            if binding is None:
                return None
        else:
            return None
        delta = self.hlsl_private_pointer_interval(
            delta_expression, intervals, constants
        )
        if delta is not None and isinstance(expression, BinaryOpNode):
            if expression.op == "-":
                delta = -delta[1], -delta[0]
        current = binding.get("offset")
        binding["offset"] = (
            None
            if current is None or delta is None
            else (current[0] + delta[0], current[1] + delta[1])
        )
        binding["side_effecting"] = binding.get(
            "side_effecting", False
        ) or self.hlsl_private_pointer_expression_has_side_effects(delta_expression)
        return binding

    def hlsl_private_pointer_loop_interval(self, node, intervals, constants):
        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None)
        if not loop_name or loop_name not in intervals:
            return None, None
        start = intervals[loop_name]
        if start[0] != start[1]:
            return loop_name, None
        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return loop_name, None
        left_name = self.expression_name(condition.left)
        right_name = self.expression_name(condition.right)
        operator = condition.op
        if left_name == loop_name:
            bound_expression = condition.right
        elif right_name == loop_name:
            bound_expression = condition.left
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(operator)
        else:
            return loop_name, None
        bound = self.hlsl_private_pointer_interval(
            bound_expression, intervals, constants
        )
        update = getattr(node, "update", None)
        step = None
        if isinstance(update, UnaryOpNode):
            update_name = self.expression_name(update.operand)
            if update_name == loop_name and update.op in {"++", "--"}:
                step = 1 if update.op == "++" else -1
        elif isinstance(update, AssignmentNode):
            update_name = self.expression_name(
                getattr(update, "target", getattr(update, "left", None))
            )
            update_operator = getattr(update, "operator", "=")
            update_value = getattr(update, "value", getattr(update, "right", None))
            update_interval = self.hlsl_private_pointer_interval(
                update_value, intervals, constants
            )
            if (
                update_name == loop_name
                and update_operator in {"+=", "-="}
                and update_interval is not None
                and update_interval[0] == update_interval[1]
                and update_interval[0] != 0
            ):
                step = update_interval[0]
                if update_operator == "-=":
                    step = -step
        if bound is None or step is None:
            return loop_name, None
        if step > 0 and operator in {"<", "<="}:
            maximum = bound[1] - (1 if operator == "<" else 0)
            if maximum <= start[0]:
                return loop_name, (start[0], start[0])
            reachable_maximum = start[0] + ((maximum - start[0]) // step) * step
            return loop_name, (start[0], reachable_maximum)
        if step < 0 and operator in {">", ">="}:
            minimum = bound[0] + (1 if operator == ">" else 0)
            if minimum >= start[0]:
                return loop_name, (start[0], start[0])
            reachable_minimum = start[0] - ((start[0] - minimum) // abs(step)) * abs(
                step
            )
            return loop_name, (reachable_minimum, start[0])
        return loop_name, None

    def hlsl_private_pointer_rebase_target(self, node, parameter_names):
        """Return the private pointer parameter moved by ``p += n`` or ``p -= n``."""

        if not isinstance(node, AssignmentNode):
            return None
        if self.map_operator(getattr(node, "operator", "=")) not in {"+=", "-="}:
            return None
        target = getattr(node, "target", getattr(node, "left", None))
        if not isinstance(target, (str, IdentifierNode, VariableNode)):
            return None
        name = self.expression_name(target)
        return name if name in parameter_names else None

    def hlsl_private_pointer_statement_assignment_ids(self, function):
        """Return ids of assignments whose value is discarded as a statement."""

        assignment_ids = set()

        def visit_statement(statement):
            if statement is None:
                return
            if isinstance(statement, (list, tuple)):
                for child in statement:
                    visit_statement(child)
                return
            if isinstance(statement, AssignmentNode):
                assignment_ids.add(id(statement))
                return
            if isinstance(statement, ExpressionStatementNode):
                expression = getattr(statement, "expression", None)
                if isinstance(expression, AssignmentNode) and not getattr(
                    statement, "is_tail_expression", False
                ):
                    assignment_ids.add(id(expression))
                return
            if isinstance(statement, BlockNode):
                visit_statement(getattr(statement, "statements", None))
                return
            if isinstance(statement, IfNode):
                visit_statement(getattr(statement, "then_branch", None))
                visit_statement(getattr(statement, "else_branch", None))
                for branch in getattr(statement, "else_if_bodies", []) or []:
                    visit_statement(branch)
                return
            if isinstance(
                statement, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)
            ):
                visit_statement(getattr(statement, "body", None))
                return
            if isinstance(statement, MatchNode):
                for arm in getattr(statement, "arms", []) or []:
                    visit_statement(getattr(arm, "body", None))
                return
            if isinstance(statement, SwitchNode):
                for case in getattr(statement, "cases", []) or []:
                    visit_statement(getattr(case, "statements", None))
                visit_statement(getattr(statement, "default_case", None))

        visit_statement(getattr(function, "body", None))
        return assignment_ids

    def hlsl_private_pointer_mutation_root_names(self, target):
        if isinstance(target, (str, IdentifierNode, VariableNode)):
            name = self.expression_name(target)
            return {name} if name else set()
        if isinstance(target, ArrayAccessNode):
            return self.hlsl_private_pointer_mutation_root_names(target.array)
        if isinstance(target, MemberAccessNode):
            return self.hlsl_private_pointer_mutation_root_names(target.object)
        if isinstance(target, SwizzleNode):
            return self.hlsl_private_pointer_mutation_root_names(
                getattr(target, "expression", None)
            )
        return set()

    HLSL_PRIVATE_POINTER_OUTPUT_BUILTINS = frozenset(
        {"modf", "frexp", "sincos", "remquo"}
    )

    def hlsl_private_pointer_call_writes_argument(
        self, call, argument_index, functions_by_name
    ):
        name = self.function_call_name(call)
        functions_by_name = functions_by_name or {}
        function = functions_by_name.get(name)
        if function is None and isinstance(name, str):
            function = functions_by_name.get(self.hlsl_materialized_function_name(name))
        if function is None:
            return name in self.HLSL_PRIVATE_POINTER_OUTPUT_BUILTINS or str(
                name
            ).startswith(("atomic", "Interlocked"))
        parameters = list(getattr(function, "parameters", []) or [])
        if argument_index >= len(parameters):
            return True
        parameter = parameters[argument_index]
        if self.hlsl_parameter_direction_qualifiers(parameter) & {"out", "inout"}:
            return True
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        return isinstance(parameter_type, (ReferenceType, PointerType)) or any(
            hasattr(parameter_type, attribute)
            for attribute in ("referenced_type", "pointee_type")
        )

    def hlsl_private_pointer_mutated_names(self, root, functions_by_name=None):
        """Return names a subtree may write.

        Identifiers passed to ``out``/``inout``/reference/pointer parameters of
        user functions, or to output builtins, count as written.
        """

        mutated = set()
        for node in self.walk_ast(root):
            if isinstance(node, AssignmentNode):
                mutated.update(
                    self.hlsl_private_pointer_mutation_root_names(
                        getattr(node, "target", getattr(node, "left", None))
                    )
                )
            elif isinstance(node, UnaryOpNode) and (
                self.map_operator(node.op) in {"++", "--", "&"}
                or node.op in {
                    "PRE_INCREMENT",
                    "PRE_DECREMENT",
                    "POST_INCREMENT",
                    "POST_DECREMENT",
                }
            ):
                mutated.update(
                    self.hlsl_private_pointer_mutation_root_names(node.operand)
                )
            elif isinstance(node, FunctionCallNode):
                for index, argument in enumerate(getattr(node, "arguments", []) or []):
                    if self.hlsl_private_pointer_call_writes_argument(
                        node, index, functions_by_name
                    ):
                        mutated.update(
                            self.hlsl_private_pointer_mutation_root_names(argument)
                        )
        return mutated

    def hlsl_private_pointer_rebase_loop_iteration_values(
        self, node, intervals, constants, functions_by_name=None
    ):
        """Return the finite iteration values of a loop that rebases a view.

        The loop must count with a constant step from an exact start towards a
        bound the interval analysis can prove, and its body must not leave the
        loop early or write the counter or the bound.  Otherwise ``None``.
        """

        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None)
        start = intervals.get(loop_name) if loop_name else None
        if start is None or start[0] != start[1]:
            return None
        condition = getattr(node, "condition", None)
        if not isinstance(
            condition, BinaryOpNode
        ) or self.hlsl_private_pointer_expression_has_side_effects(condition):
            return None

        update = getattr(node, "update", None)
        step = None
        if isinstance(update, UnaryOpNode):
            update_operator = self.map_operator(update.op)
            if self.expression_name(update.operand) == loop_name:
                if update_operator == "++" or update.op in {
                    "PRE_INCREMENT",
                    "POST_INCREMENT",
                }:
                    step = 1
                elif update_operator == "--" or update.op in {
                    "PRE_DECREMENT",
                    "POST_DECREMENT",
                }:
                    step = -1
        elif isinstance(update, AssignmentNode):
            update_target = getattr(update, "target", getattr(update, "left", None))
            update_value = getattr(update, "value", getattr(update, "right", None))
            update_operator = self.map_operator(getattr(update, "operator", "="))
            update_interval = self.hlsl_private_pointer_interval(
                update_value, intervals, constants
            )
            if (
                self.expression_name(update_target) == loop_name
                and update_operator in {"+=", "-="}
                and update_interval is not None
                and update_interval[0] == update_interval[1]
            ):
                step = update_interval[0]
                if update_operator == "-=":
                    step = -step
        if step in {None, 0}:
            return None

        body = getattr(node, "body", None)
        for child in self.walk_ast(body):
            if isinstance(child, (BreakNode, ContinueNode, ReturnNode)):
                return None
            if isinstance(child, (VariableNode, ArrayNode)) and (
                getattr(child, "name", None) == loop_name
            ):
                return None
        mutated = self.hlsl_private_pointer_mutated_names(body, functions_by_name)
        if loop_name in mutated:
            return None
        dependencies = {
            getattr(child, "name", None)
            for child in self.walk_ast(condition)
            if isinstance(child, (IdentifierNode, VariableNode))
        }
        dependencies.discard(loop_name)
        if dependencies & mutated:
            return None

        proven_name, loop_interval = self.hlsl_private_pointer_loop_interval(
            node, intervals, constants
        )
        if proven_name != loop_name or loop_interval is None:
            return None
        first = start[0]
        if step > 0:
            if loop_interval[0] != first or loop_interval[1] < first:
                return None
            iteration_count = ((loop_interval[1] - first) // step) + 1
            if iteration_count > HLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS:
                return None
            values = tuple(range(first, loop_interval[1] + 1, step))
        else:
            if loop_interval[1] != first or loop_interval[0] > first:
                return None
            iteration_count = ((first - loop_interval[0]) // (-step)) + 1
            if iteration_count > HLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS:
                return None
            values = tuple(range(first, loop_interval[0] - 1, step))
        if not values or (min(values), max(values)) != loop_interval:
            return None
        terminal_value = values[-1] + step
        if any(
            value < -(1 << 31) or value > HLSL_PRIVATE_POINTER_BASE_MAX
            for value in (*values, terminal_value)
        ):
            return None
        return values

    def hlsl_private_pointer_call_callee(self, call):
        """The function a call reaches, as the private pointer analysis names it."""
        callee_name = self.function_call_name(call)
        arguments = list(getattr(call, "arguments", []) or [])
        specialized_name = generic_function_call_name(self, callee_name, arguments)
        if specialized_name in self.function_private_pointer_parameters:
            return specialized_name
        return self.hlsl_materialized_function_name(callee_name)

    def hlsl_private_pointer_signed_slots(self, function, parameter_indices):
        if function is None:
            return ()
        return hlsl_private_pointer_call_intervals.signed_parameter_slots(
            function,
            set(parameter_indices),
            lambda parameter: self.map_type(
                self.type_name_string(
                    getattr(
                        parameter,
                        "param_type",
                        getattr(parameter, "vtype", None),
                    )
                )
                or ""
            ),
        )

    def hlsl_analyze_private_pointer_functions(
        self, functions_by_name, pointer_parameters
    ):
        """Analyze every function, seeding helper parameters from their calls.

        Returns the direct spans and the recorded view calls; see
        hlsl_private_pointer_call_intervals.
        """
        # Helpers that move a private pointer parameter, with every call of
        # each in the program.
        call_sites = {}
        for function_name, parameters in pointer_parameters.items():
            names = {parameter.name for parameter in parameters}
            function = functions_by_name.get(function_name)
            if function is not None and any(
                self.hlsl_private_pointer_rebase_target(node, names) is not None
                for node in self.walk_ast(getattr(function, "body", []))
            ):
                call_sites[function_name] = set()
        for function in functions_by_name.values() if call_sites else ():
            for node in self.walk_ast(getattr(function, "body", [])):
                if isinstance(node, FunctionCallNode):
                    callee = self.hlsl_private_pointer_call_callee(node)
                    if callee in call_sites:
                        call_sites[callee].add(id(node))

        def run_pass(seeds):
            self.hlsl_private_pointer_parameter_seeds = seeds
            direct_spans = {
                function_name: {parameter.name: 0 for parameter in parameters}
                for function_name, parameters in pointer_parameters.items()
            }
            self.function_private_pointer_full_span_parameters = {
                function_name: set() for function_name in pointer_parameters
            }
            view_calls = {function_name: [] for function_name in functions_by_name}
            try:
                for function_name, function in functions_by_name.items():
                    self.hlsl_analyze_private_pointer_function(
                        function,
                        functions_by_name,
                        direct_spans.get(function_name, {}),
                        view_calls[function_name],
                    )
            finally:
                self.hlsl_private_pointer_parameter_seeds = {}
            return direct_spans, view_calls

        names = (
            "hlsl_private_pointer_zero_index_access_ids",
            "hlsl_private_pointer_rebase_ids",
            "hlsl_private_pointer_nonzero_rebase_ids",
            "hlsl_private_pointer_rejected_rebases",
        )

        class State:
            @staticmethod
            def snapshot():
                return {
                    name: type(getattr(self, name))(getattr(self, name))
                    for name in names
                }

            @staticmethod
            def restore(snapshot):
                for name, value in snapshot.items():
                    setattr(self, name, type(value)(value))

        return hlsl_private_pointer_call_intervals.analyze_with_call_seeds(
            run_pass,
            lambda result: hlsl_private_pointer_call_intervals.joined_parameter_seeds(
                result[1], call_sites
            ),
            State,
            len(functions_by_name) + 1,
            (DirectXPrivatePointerParameterError,),
        )

    def hlsl_private_pointer_binding_interval(
        self, call, parameter_name, binding, required_span=None
    ):
        if binding.get("side_effecting"):
            reason = "side-effecting-view-offset"
        elif binding.get("offset") is None:
            caller_name = call.get("caller")
            root_name = binding.get("root")
            full_span_parameters = (
                self.function_private_pointer_full_span_parameters.get(
                    caller_name, set()
                )
            )
            caller_span = self.function_private_pointer_required_spans.get(
                caller_name, {}
            ).get(root_name)
            if (
                binding.get("root_kind") == "parameter"
                and root_name in full_span_parameters
                and required_span is not None
                and caller_span is not None
                and caller_span >= required_span
            ):
                return 0, caller_span - required_span
            reason = "unprovable-view-offset"
        else:
            offset = binding["offset"]
            if offset[0] >= 0:
                return offset
            reason = "negative-view-offset"
        backing_name = binding.get("root") or "unknown"
        raise DirectXPrivatePointerParameterError(
            "DirectX cannot prove the private pointer view offset for "
            f"'{call['callee']}.{parameter_name}' into backing array "
            f"'{backing_name}'",
            function_name=call["callee"],
            parameter_name=parameter_name,
            reason=reason,
            source_location=getattr(call.get("node"), "source_location", None),
        )

    def hlsl_private_pointer_backing_error(self, call, parameter_name, binding):
        backing_name = binding.get("root") or "unknown"
        raise DirectXPrivatePointerParameterError(
            "DirectX private pointer view "
            f"'{call['callee']}.{parameter_name}' has no concrete fixed backing "
            f"array for '{backing_name}'",
            function_name=call["callee"],
            parameter_name=parameter_name,
            reason="missing-view-backing",
            source_location=getattr(call.get("node"), "source_location", None),
        )

    def hlsl_validate_private_pointer_call_binding(
        self,
        caller_name,
        caller_variant,
        call,
        parameter_name,
        binding,
        extent,
        caller_parameter_positions,
    ):
        if binding.get("root_kind") not in {
            "local",
            "local_word_view",
            "parameter",
            "scalar",
        }:
            self.hlsl_private_pointer_backing_error(call, parameter_name, binding)
        required_span = self.function_private_pointer_required_spans[call["callee"]][
            parameter_name
        ]
        if parameter_name in self.function_private_pointer_full_span_parameters.get(
            call["callee"], set()
        ):
            required_span = extent
        offset = self.hlsl_private_pointer_binding_interval(
            call,
            parameter_name,
            binding,
            required_span=required_span,
        )
        upper = offset[1] + required_span - 1
        if binding["root_kind"] in {"local", "local_word_view", "scalar"}:
            available = extent
        else:
            root_name = binding["root"]
            available = self.function_private_pointer_required_spans[caller_name][
                root_name
            ]
            variant_extent = self.hlsl_private_pointer_variant_extent(
                caller_variant[caller_parameter_positions[root_name]]
            )
            if available > variant_extent:
                available = variant_extent
        if upper < available:
            return
        backing_name = binding.get("root") or "unknown"
        raise DirectXPrivatePointerParameterError(
            "DirectX private pointer view "
            f"'{call['callee']}.{parameter_name}' requires elements "
            f"{offset[0]} through {upper}, but backing array '{backing_name}' "
            f"has extent {available}",
            function_name=call["callee"],
            parameter_name=parameter_name,
            reason="view-out-of-bounds",
            source_location=getattr(call.get("node"), "source_location", None),
        )

    def hlsl_resource_array_size_expression(self, node, vtype=None):
        if node is None:
            return None

        if vtype is None:
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if vtype is None:
                vtype = getattr(node, "param_type", None)

        if vtype is None or (
            hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1
        ):
            return None

        base_type = (
            self.convert_type_node_to_string(vtype)
            if hasattr(vtype, "name") or hasattr(vtype, "element_type")
            else split_array_type_suffix(str(vtype))[0]
        )
        if not self.is_resource_array_hint_type(base_type):
            return None

        for attr in getattr(node, "attributes", []) or []:
            if not self.is_hlsl_resource_array_size_attribute(attr):
                continue
            return getattr(attr, "name", None)
        return None

    def is_hlsl_resource_array_size_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name or getattr(attr, "arguments", None):
            return False
        # Vulkan block kinds mark how a ConstantBuffer<T> is fed; they never name an array size.
        if str(attr_name).lower() in {"push_constant", "shader_record"}:
            return False
        if (
            is_image_format_attribute(attr)
            or self.is_resource_binding_attribute(attr)
            or is_resource_access_attribute(attr)
            or self.is_resource_memory_attribute(attr)
            or self.is_rasterizer_ordered_attribute(attr)
            or self.is_glsl_buffer_block_attribute(attr)
            or self.hlsl_mesh_parameter_role_attribute_name(attr)
            or self.hlsl_mesh_payload_parameter_attribute_name(attr)
            or self.hlsl_stage_attribute_name(attr)
            or self.is_hlsl_declaration_metadata_attribute(attr)
        ):
            return False
        return True

    def is_hlsl_resource_array_size_marker(self, node, attr):
        array_size = self.hlsl_resource_array_size_expression(node)
        return array_size is not None and getattr(attr, "name", None) == array_size

    def collect_unsized_resource_globals(self, ast):
        globals_by_name = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if (
                name
                and self.is_unsized_resource_array_type(vtype)
                and not self.is_hlsl_buffer_resource_array_type(vtype)
            ):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_fixed_resource_global_sizes(self, ast):
        global_arrays = {}
        for node in getattr(ast, "global_variables", []) or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            size = self.fixed_resource_array_size(vtype)
            if size is None:
                size_expr = self.hlsl_resource_array_size_expression(node, vtype)
                size = self.literal_int_value(size_expr, self.literal_int_constants)
            if name and size is not None:
                global_arrays[name] = size
        return global_arrays

    def collect_unsized_resource_parameters(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_resource_array_type(
                    vtype
                ) or self.is_unsized_storage_array_parameter(param):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                size = self.fixed_resource_array_size(vtype)
                if size is None:
                    size_expr = self.hlsl_resource_array_size_expression(param, vtype)
                    size = self.literal_int_value(
                        size_expr, self.initial_literal_int_constants(func)
                    )
                if size is not None:
                    function_arrays.setdefault(func_name, {})[param.name] = size
        return function_arrays

    def fixed_resource_array_size(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is None:
                return None
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_array_hint_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_resource_array_hint_type(base_type):
            return None
        return max(size, 1)

    def is_unsized_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_resource_array_hint_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_resource_array_hint_type(base_type)

    def is_unsized_storage_array_parameter(self, parameter):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "resource_qualifiers", []) or []
        }
        if "storage" not in qualifiers:
            return False

        vtype = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return vtype.size is None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        _, size = parse_array_type(type_string)
        return size is None

    def is_resource_array_hint_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            # ConstantBuffer<T> name[] is a runtime-sized array of constant-buffer descriptors (shader model 5.1).
            or self.is_hlsl_constant_buffer_type(vtype)
            or texel_buffer_base_type(vtype) is not None
            or self.is_hlsl_readonly_buffer_type(vtype)
            or self.is_hlsl_uav_buffer_type(vtype)
            or str(self.resource_base_type(vtype))
            in self.glsl_buffer_block_struct_names
        )

    def is_hlsl_buffer_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
        elif hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        else:
            type_string = str(vtype)
            if "[" not in type_string or "]" not in type_string:
                return False
            base_type, _ = parse_array_type(type_string)
        return self.is_hlsl_readonly_buffer_type(
            base_type
        ) or self.is_hlsl_uav_buffer_type(base_type)

    def walk_ast(self, root):
        return walk_ast_values(root)

    def expression_name(self, expr):
        if isinstance(expr, str):
            return expr
        if hasattr(expr, "name") and isinstance(expr.name, str):
            return expr.name
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.expression_name(array_expr)
        return None

    def is_shadow_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler1DShadow",
            "sampler1DArrayShadow",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def is_multisample_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def is_sampler_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return mapped_type in {"SamplerState", "SamplerComparisonState"}

    def is_texture_type(self, vtype):
        return self.map_type(self.resource_base_type(vtype)).startswith(
            ("Texture", "FeedbackTexture")
        )

    def is_image_type(self, vtype):
        return self.map_type(self.resource_base_type(vtype)).startswith(
            ("RWTexture", "RasterizerOrderedTexture")
        )

    def is_texture_or_image_type(self, vtype):
        return self.is_texture_type(vtype) or self.is_image_type(vtype)

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return is_integer_coordinate_type_name(base_type, mapped_type)

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampled_shape_type = self.sampled_texture_shape_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = self.is_multisample_texture_resource_type(texture_type)
        is_storage_image = self.is_storage_image_resource_type(texture_type)
        shape_type = sampled_shape_type if sampled_shape_type else texture_type

        coordinate_dimension = None
        if texture_type and "Cube" not in shape_type:
            if texture_type.startswith(
                ("RWTexture1DArray<", "RasterizerOrderedTexture1DArray<")
            ):
                coordinate_dimension = 2
            elif texture_type.startswith(
                ("RWTexture1D<", "RasterizerOrderedTexture1D<")
            ):
                coordinate_dimension = 1
            elif texture_type.startswith(
                ("RWTexture2DArray<", "RasterizerOrderedTexture2DArray<")
            ):
                coordinate_dimension = 3
            elif texture_type.startswith("RWTexture2DMSArray<"):
                coordinate_dimension = 3
            elif texture_type.startswith(
                ("RWTexture3D<", "RasterizerOrderedTexture3D<")
            ):
                coordinate_dimension = 3
            elif texture_type.startswith("RWTexture2DMS<"):
                coordinate_dimension = 2
            elif texture_type.startswith(
                ("RWTexture2D<", "RasterizerOrderedTexture2D<")
            ):
                coordinate_dimension = 2
            elif texture_type.startswith("Texture2DMSArray"):
                coordinate_dimension = 3
            elif texture_type.startswith("Texture2DMS"):
                coordinate_dimension = 2
            else:
                coordinate_dimension = {
                    "Texture1D": 1,
                    "Texture1DArray": 2,
                    "Texture2D": 2,
                    "Texture2DArray": 3,
                    "Texture3D": 3,
                }.get(shape_type)

        offset_dimension = None
        if texture_type and "Cube" not in shape_type:
            offset_dimension = {
                "Texture1D": 1,
                "Texture1DArray": 1,
                "Texture2D": 2,
                "Texture2DArray": 2,
                "Texture3D": 3,
            }.get(shape_type)

        gradient_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            gradient_dimension = {
                "Texture1D": 1,
                "Texture1DArray": 1,
                "Texture2D": 2,
                "Texture2DArray": 2,
                "Texture3D": 3,
                "TextureCube": 3,
                "TextureCubeArray": 3,
            }.get(shape_type)

        query_lod_coordinate_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            query_lod_coordinate_dimension = {
                "Texture1D": 1,
                "Texture1DArray": 2,
                "Texture2D": 2,
                "Texture2DArray": 3,
                "Texture3D": 3,
                "TextureCube": 3,
                "TextureCubeArray": 4,
            }.get(shape_type)

        descriptor = texture_resource_dimension_descriptor(
            texture_type,
            sampling,
            coordinate_dimension=coordinate_dimension,
            offset_dimension=offset_dimension,
            gradient_dimension=gradient_dimension,
            query_lod_coordinate_dimension=query_lod_coordinate_dimension,
            is_multisample=is_multisample,
        )
        if sampling.get("compare_offset") and offset_dimension is not None:
            descriptor["compare_offset_dimension"] = offset_dimension
            descriptor["compare_lod_offset_dimension"] = offset_dimension
            descriptor["compare_grad_offset_dimension"] = offset_dimension
        return descriptor

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        if (
            is_texture_compare_offset_operation(func_name)
            or is_texture_compare_lod_offset_operation(func_name)
            or is_texture_compare_grad_offset_operation(func_name)
        ):
            if descriptor["compare_offset_dimension"] is None:
                return None
            return (
                descriptor["offset_dimension"]
                if descriptor["offset_dimension"] is not None
                else descriptor["compare_offset_dimension"]
            )
        key = texture_resource_offset_dimension_key(
            func_name, collapse_compare_offsets=True
        )
        return descriptor[key]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

    def is_resource_parameter_type(self, vtype):
        return (
            self.is_texture_type(vtype)
            or self.is_sampler_type(vtype)
            or self.is_image_type(vtype)
            or self.is_hlsl_acceleration_structure_type(vtype)
        )

    def is_hlsl_acceleration_structure_type(self, vtype):
        return (
            self.map_type(self.resource_base_type(vtype))
            == "RaytracingAccelerationStructure"
        )

    def is_explicit_sampler_argument(self, args):
        if len(args) < 3:
            return False
        return self.texture_call_uses_explicit_sampler(args)

    def texture_call_uses_explicit_sampler(self, args):
        if len(args) < 2:
            return False
        sampler_name = self.expression_name(args[1]) or self.generate_expression(
            args[1]
        )
        if sampler_name in self.local_variable_types:
            sampler_type = self.local_variable_types.get(sampler_name)
            return sampler_type is not None and self.is_sampler_type(sampler_type)
        if (
            sampler_name in self.sampler_variables
            or sampler_name in self.current_sampler_parameters
        ):
            return True
        arg_type = self.expression_result_type(args[1])
        return arg_type is not None and self.is_sampler_type(arg_type)

    def inner_resource_array_count(self, element_type):
        """Registers one element of a resource array of arrays takes: the product of its own fixed extents."""
        count = 1
        while isinstance(element_type, ArrayType):
            count *= self.resource_array_count(element_type.size)
            element_type = element_type.element_type
        return count

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def global_resource_shape(self, node):
        resource_count = 1
        if hasattr(node, "var_type"):
            if (
                hasattr(node.var_type, "name")
                or hasattr(node.var_type, "element_type")
                or isinstance(node.var_type, PointerType)
            ):
                if (
                    hasattr(node.var_type, "element_type")
                    and str(type(node.var_type)).find("ArrayType") != -1
                ):
                    vtype = split_array_type_suffix(
                        self.convert_type_node_to_string(node.var_type.element_type)
                    )[0]
                    array_size = (
                        self.generate_expression(node.var_type.size)
                        if node.var_type.size
                        else self.resource_array_size_hints.get(node.name, "")
                    )
                    resource_count = self.resource_array_count(
                        node.var_type.size if node.var_type.size else array_size
                    ) * self.inner_resource_array_count(node.var_type.element_type)
                else:
                    vtype = self.convert_type_node_to_string(node.var_type)
            else:
                vtype, array_suffix = split_array_type_suffix(str(node.var_type))
                if array_suffix:
                    resource_count = self.resource_array_count(
                        array_suffix[1 : array_suffix.index("]")]
                    )
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            vtype = "float"
        attribute_array_size = self.hlsl_resource_array_size_expression(node, vtype)
        if attribute_array_size is not None:
            resource_count = self.resource_array_count(attribute_array_size)
        return vtype, resource_count
