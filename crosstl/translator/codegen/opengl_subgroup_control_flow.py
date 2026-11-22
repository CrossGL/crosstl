"""Software subgroup control flow in the GLSL generator: lane-local matrices, uniform and masked execution, loop plans, constant guards and the first subgroup operation."""

from ..arithmetic_conversions import target_arithmetic_type
from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ContinueNode,
    CooperativeMatrixOpNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    PrimitiveType,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from .array_utils import evaluate_literal_int_expression
from .glsl_constant_guards import glsl_constant_guards
from .opengl_values import (
    OpenGLMappedOverloadError,
    OpenGLSoftwareSubgroupError,
)
from .pointer_reinterpret import scalar_storage_layout
from .stage_utils import (
    collect_stage_local_cbuffers,
    function_stage_name,
    normalize_stage_name,
)
from .subgroup_loop_carried import loop_carried_candidates
from .subgroup_reassignments import (
    followed_locals,
    followed_offsets,
)


class OpenGLSubgroupControlFlowMixin:
    """Subgroup control flow members of ``GLSLCodeGen``."""

    def glsl_software_subgroup_lane_local_matrix_nodes(self, root):
        """Matrix fills and transfers, which software lowering keeps per lane."""
        if not self.cooperative_matrix_software_lowering:
            return []
        return [
            node
            for node in self.walk_ast(root)
            if isinstance(node, CooperativeMatrixOpNode)
            and node.operation in {"fill", "diagonal", "load", "store"}
        ]

    def glsl_software_subgroup_stage_entries(self, ast, target_stage=None):
        entries = []
        seen = set()
        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if target_stage is not None and stage_name != target_stage:
                continue
            function = getattr(stage, "entry_point", None)
            if function is None or id(function) in seen:
                continue
            seen.add(id(function))
            entries.append(
                (stage_name, function, getattr(stage, "execution_config", None))
            )
        for function in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(function)
            if stage_name is None:
                continue
            if target_stage is not None and stage_name != target_stage:
                continue
            if id(function) in seen:
                continue
            seen.add(id(function))
            entries.append(
                (
                    stage_name,
                    function,
                    getattr(function, "execution_config", None),
                )
            )
        return entries

    def glsl_software_subgroup_error(
        self,
        message,
        *,
        workgroup_size=None,
        operation=None,
        reason,
        source_location=None,
    ):
        return OpenGLSoftwareSubgroupError(
            message,
            software_subgroup_width=self.software_subgroup_width,
            workgroup_size=workgroup_size,
            operation=operation,
            reason=reason,
            source_location=source_location,
        )

    def glsl_software_subgroup_functions(self, ast):
        functions = []
        seen = set()
        for function in self.collect_functions(ast):
            if id(function) in seen:
                continue
            seen.add(id(function))
            functions.append(function)
        return functions

    @staticmethod
    def glsl_software_subgroup_statement_list(body):
        if hasattr(body, "statements"):
            return list(body.statements or [])
        if isinstance(body, list):
            return list(body)
        if body is None:
            return []
        return [body]

    def glsl_software_subgroup_top_level_call(self, statement):
        expression = statement
        if isinstance(statement, ExpressionStatementNode):
            expression = getattr(statement, "expression", None)
        if isinstance(expression, VariableNode):
            expression = getattr(expression, "initial_value", None)
        elif isinstance(expression, AssignmentNode):
            expression = getattr(
                expression,
                "value",
                getattr(expression, "right", None),
            )
        if isinstance(expression, FunctionCallNode):
            # Canonical stores eagerly evaluate their value once, just like
            # assignment RHSs. Do not extend this rule to source-owned calls.
            if (
                isinstance(expression.function, (str, IdentifierNode))
                and self.function_call_name(expression) == "buffer_store"
                and "buffer_store" not in self.function_return_types
                and len(expression.arguments) == 3
                and isinstance(expression.arguments[0], IdentifierNode)
                and isinstance(expression.arguments[2], FunctionCallNode)
                and all(
                    isinstance(
                        node,
                        (
                            IdentifierNode,
                            LiteralNode,
                            BinaryOpNode,
                            MemberAccessNode,
                            SwizzleNode,
                            PrimitiveType,
                        ),
                    )
                    or isinstance(node, UnaryOpNode)
                    and node.operator in {"+", "-", "!", "~"}
                    for node in self.walk_ast(expression.arguments[1])
                )
            ):
                return expression.arguments[2]
            return expression
        return None

    def glsl_software_subgroup_direct_operation(self, expression):
        if isinstance(expression, WaveOpNode):
            operation = expression.operation
            arguments = list(expression.arguments or [])
        elif isinstance(expression, FunctionCallNode):
            operation = self.glsl_wave_operation_name(
                self.function_call_name(expression)
            )
            arguments = list(
                getattr(expression, "arguments", getattr(expression, "args", [])) or []
            )
        else:
            return None
        if operation not in self.GLSL_WAVE_INTRINSIC_ARITIES:
            return None
        return operation, arguments, expression

    def glsl_software_subgroup_speculatively_safe_expression(self, expression):
        if not self.glsl_side_effect_free_expression(expression):
            return False
        for child in self.walk_ast(expression):
            if isinstance(child, ArrayAccessNode):
                return False
            if (
                isinstance(child, UnaryOpNode)
                and self.map_operator(
                    getattr(child, "op", getattr(child, "operator", None))
                )
                == "*"
            ):
                return False
            if isinstance(child, BinaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"/", "%", "<<", ">>"}:
                return False
        return True

    def glsl_software_subgroup_masked_reduction(self, statement, operation_node):
        """Return ``(wrapped,)`` when ``statement`` is ``x = op(v)``.

        ``op`` must be ``operation_node`` itself, optionally under scalar
        conversions, and ``x`` a side-effect-free name. Return ``None`` when
        the statement has another shape and ``False`` when its shape cannot be
        predicated.
        """
        candidate = statement
        if isinstance(candidate, ExpressionStatementNode):
            candidate = getattr(candidate, "expression", None)
        if not isinstance(candidate, AssignmentNode):
            return None
        operator = self.map_operator(
            getattr(candidate, "operator", getattr(candidate, "op", "="))
        )
        if operator != "=":
            return None
        value = getattr(candidate, "value", getattr(candidate, "right", None))
        wrappers, inner = self.glsl_software_subgroup_conversion_chain(value)
        direct = self.glsl_software_subgroup_direct_operation(inner)
        if direct is None or id(direct[2]) != id(operation_node) or len(direct[1]) != 1:
            return None
        # Converted results reuse the source assignment, whose reduction is
        # replaced by the predicated result; that substitution needs the
        # canonical wave node.
        if wrappers and not isinstance(direct[2], WaveOpNode):
            return False
        target = getattr(candidate, "target", getattr(candidate, "left", None))
        target_name = self.expression_name(target)
        if not target_name or not self.glsl_side_effect_free_expression(target):
            return False
        return (bool(wrappers),)

    def glsl_software_subgroup_masked_if_plan(self, node, *, allow_multiple=False):
        """Plan ``if (c) { prefix; x = subgroupAdd(v); suffix }`` as masked.

        With ``allow_multiple``, every top-level statement of the body that
        holds a collective must be such a reduction, as in MLX's RMS VJP
        ``if (c) { a = simd_sum(u); b = simd_sum(v); }``. The reductions then
        run in source order in every invocation, and each statement between
        them still runs only where ``c`` held, after the previous assignment.
        """
        if (
            getattr(node, "else_if_conditions", None)
            or getattr(node, "else_if_bodies", None)
            or getattr(node, "else_body", None) is not None
        ):
            return None
        condition = getattr(node, "condition", getattr(node, "if_condition", None))
        if not self.glsl_side_effect_free_expression(condition):
            return None
        records = self.glsl_software_subgroup_operation_records(node)
        if (
            len(records) > 1
            and allow_multiple
            and all(
                operation in {"WaveActiveSum", "WaveActiveMin", "WaveActiveMax"}
                for operation, _node in records
            )
        ):
            return self.glsl_software_subgroup_masked_if_sequence_plan(
                node, condition, [record_node for _operation, record_node in records]
            )
        if len(records) != 1 or records[0][0] not in {
            "WaveActiveSum",
            "WaveActiveMin",
            "WaveActiveMax",
        }:
            return None
        statements = self.glsl_software_subgroup_statement_list(node.if_body)
        operation_node = records[0][1]
        for index, statement in enumerate(statements):
            candidate = statement
            if isinstance(candidate, ExpressionStatementNode):
                candidate = getattr(candidate, "expression", None)
            if not isinstance(candidate, AssignmentNode):
                continue
            operator = self.map_operator(
                getattr(candidate, "operator", getattr(candidate, "op", "="))
            )
            if operator != "=":
                continue
            value = getattr(candidate, "value", getattr(candidate, "right", None))
            wrappers, inner = self.glsl_software_subgroup_conversion_chain(value)
            direct = self.glsl_software_subgroup_direct_operation(inner)
            if (
                direct is None
                or id(direct[2]) != id(operation_node)
                or len(direct[1]) != 1
            ):
                continue
            # Converted results reuse the source assignment, whose reduction is
            # replaced by the predicated result; that substitution needs the
            # canonical wave node.
            if wrappers and not isinstance(direct[2], WaveOpNode):
                return None
            target = getattr(candidate, "target", getattr(candidate, "left", None))
            target_name = self.expression_name(target)
            if not target_name or not self.glsl_side_effect_free_expression(target):
                return None
            prefix = statements[:index]
            hoisted_target_declaration_index = None
            for prefix_index, prefix_statement in enumerate(prefix):
                if isinstance(prefix_statement, ArrayNode):
                    return None
                if isinstance(prefix_statement, VariableNode):
                    if (
                        prefix_statement.name != target_name
                        or hoisted_target_declaration_index is not None
                        or getattr(prefix_statement, "initial_value", None) is None
                        or not self.glsl_software_subgroup_speculatively_safe_expression(
                            prefix_statement.initial_value
                        )
                    ):
                        return None
                    hoisted_target_declaration_index = prefix_index
                if self.glsl_software_subgroup_operation_records(prefix_statement):
                    return None
                if any(
                    isinstance(child, (BreakNode, ContinueNode, ReturnNode))
                    for child in self.walk_ast(prefix_statement)
                ):
                    return None
            if any(
                isinstance(child, (BreakNode, ContinueNode, ReturnNode))
                for child in self.walk_ast(node.if_body)
            ):
                return None
            return {
                "statementIndex": index,
                "hoistedTargetDeclarationIndex": hoisted_target_declaration_index,
                "wrapped": bool(wrappers),
                "subgroupUniform": (
                    self.glsl_software_subgroup_guard_is_subgroup_uniform(condition)
                ),
            }
        return None

    def glsl_software_subgroup_masked_if_sequence_plan(
        self, node, condition, operation_nodes
    ):
        statements = self.glsl_software_subgroup_statement_list(node.if_body)
        if any(
            isinstance(child, (BreakNode, ContinueNode, ReturnNode))
            for child in self.walk_ast(node.if_body)
        ):
            return None
        pending = list(operation_nodes)
        indices = []
        wrapped = []
        for index, statement in enumerate(statements):
            if not self.glsl_software_subgroup_operation_records(statement):
                continue
            reduction = (
                self.glsl_software_subgroup_masked_reduction(statement, pending[0])
                if pending
                else None
            )
            if not reduction:
                return None
            pending.pop(0)
            indices.append(index)
            wrapped.append(reduction[0])
        if pending or not indices:
            return None
        # Each later block starts a new scope, so declarations before the last
        # reduction would fall out of scope.
        if any(
            isinstance(statement, (VariableNode, ArrayNode))
            for statement in statements[: indices[-1]]
        ):
            return None
        return {
            "statementIndex": indices[0],
            "hoistedTargetDeclarationIndex": None,
            "wrapped": wrapped[0],
            "subgroupUniform": self.glsl_software_subgroup_guard_is_subgroup_uniform(
                condition
            ),
            "statementIndices": tuple(indices),
            "wrappedReductions": tuple(wrapped),
        }

    def glsl_software_subgroup_conversion_chain(self, expression):
        """Split ``T(U(op(x)))`` into its scalar conversions and ``op(x)``.

        MLX's bfloat16 ``simd_max`` and ``simd_sum`` reduce a float conversion
        and convert the result back, so their collective sits under
        single-argument scalar conversions.
        """
        wrappers = []
        current = expression
        while True:
            if isinstance(current, CastNode):
                target = self.type_name_string(current.target_type)
                if scalar_storage_layout(target) is None:
                    break
                wrappers.append(current)
                current = current.expression
                continue
            if isinstance(current, FunctionCallNode):
                arguments = list(
                    getattr(current, "arguments", getattr(current, "args", [])) or []
                )
                name = self.function_call_name(current)
                if (
                    len(arguments) != 1
                    or not isinstance(name, str)
                    or name in self.function_return_types
                    or scalar_storage_layout(name) is None
                ):
                    break
                wrappers.append(current)
                current = arguments[0]
                continue
            break
        return wrappers, current

    def glsl_software_subgroup_guard_is_subgroup_uniform(self, condition):
        """Whether every invocation of one software subgroup agrees on a guard.

        The guard may combine the entry's subgroup index, which the entry never
        writes, with literals and compile-time constants. Each subgroup then
        takes part in a predicated reduction completely or not at all.
        """
        subgroup_names = getattr(self, "glsl_software_subgroup_guard_names", set())
        constant_names = getattr(
            self, "glsl_software_subgroup_guard_constant_names", set()
        )

        def uniform(expression):
            if isinstance(expression, LiteralNode):
                return True
            if isinstance(expression, (IdentifierNode, VariableNode)):
                name = getattr(expression, "name", None)
                return name in subgroup_names or name in constant_names
            if isinstance(expression, CastNode):
                return uniform(expression.expression)
            if isinstance(expression, BinaryOpNode):
                operator = self.map_operator(
                    getattr(expression, "op", getattr(expression, "operator", None))
                )
                return (
                    operator in self.GLSL_SOFTWARE_SUBGROUP_UNIFORM_GUARD_OPERATORS
                    and uniform(expression.left)
                    and uniform(expression.right)
                )
            if isinstance(expression, UnaryOpNode):
                operator = self.map_operator(
                    getattr(expression, "op", getattr(expression, "operator", None))
                )
                return operator in {"!", "-", "~"} and uniform(expression.operand)
            if isinstance(expression, FunctionCallNode):
                wrappers, inner = self.glsl_software_subgroup_conversion_chain(
                    expression
                )
                return bool(wrappers) and uniform(inner)
            return False

        return bool(subgroup_names) and uniform(condition)

    def glsl_software_subgroup_entry_unwritten_names(self, entry_function, names):
        """Return the ``names`` that the entry never rebinds or writes."""
        names = set(names)
        for node in self.walk_ast(getattr(entry_function, "body", None)):
            if isinstance(node, VariableNode):
                names.discard(node.name)
            elif isinstance(node, AssignmentNode):
                names.discard(self.glsl_software_subgroup_assignment_target_name(node))
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--", "&"}:
                names.discard(self.expression_name(getattr(node, "operand", None)))
            elif isinstance(node, FunctionCallNode):
                for argument in (
                    getattr(node, "arguments", getattr(node, "args", [])) or []
                ):
                    if isinstance(argument, (IdentifierNode, VariableNode)) and (
                        self.function_call_name(node) in self.function_return_types
                    ):
                        # A helper could receive the name by reference.
                        names.discard(argument.name)
        return names

    def glsl_software_subgroup_subgroup_id_expression(self, expression):
        current = expression
        while True:
            if isinstance(current, CastNode):
                current = current.expression
                continue
            if isinstance(current, ConstructorNode):
                arguments = list(getattr(current, "arguments", None) or [])
                if len(arguments) != 1:
                    return False
                current = arguments[0]
                continue
            if isinstance(current, FunctionCallNode):
                arguments = list(
                    getattr(current, "arguments", getattr(current, "args", [])) or []
                )
                if len(arguments) != 1 or self.function_call_name(current) not in {
                    "int",
                    "uint",
                    "int32_t",
                    "uint32_t",
                }:
                    return False
                current = arguments[0]
                continue
            break
        if not isinstance(current, IdentifierNode):
            return False
        semantic = self.semantic_from_node(current)
        mapped_semantic = self.map_semantic(semantic) if semantic else None
        return (
            mapped_semantic == "gl_SubgroupID"
            or current.name == "gl_SubgroupID"
            or current.name in getattr(
                self, "glsl_software_subgroup_id_parameter_names", set()
            )
        )

    def glsl_software_subgroup_strided_for_plan(
        self,
        node,
        uniform_names,
        constants,
    ):
        initializer = getattr(node, "init", None)
        if not isinstance(initializer, VariableNode):
            return None
        loop_name = getattr(initializer, "name", None)
        loop_type = self.map_type(getattr(initializer, "var_type", None))
        initial_value = getattr(initializer, "initial_value", None)
        if (
            not loop_name
            or loop_type not in {"int", "uint"}
            or not self.glsl_software_subgroup_subgroup_id_expression(initial_value)
        ):
            return None

        condition = getattr(node, "condition", None)
        if (
            not isinstance(condition, BinaryOpNode)
            or self.map_operator(
                getattr(condition, "op", getattr(condition, "operator", None))
            )
            != "<"
            or self.expression_name(condition.left) != loop_name
            or self.expression_name(condition.right) == loop_name
            or not self.glsl_software_subgroup_uniform_expression(
                condition.right, uniform_names
            )
        ):
            return None

        update = getattr(node, "update", None)
        if (
            not isinstance(update, AssignmentNode)
            or self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            != "+="
            or self.glsl_software_subgroup_assignment_target_name(update) != loop_name
        ):
            return None
        step_expression = getattr(update, "value", getattr(update, "right", None))
        step = evaluate_literal_int_expression(step_expression, constants)
        if step != self.glsl_software_subgroup_count:
            return None

        body_statements = self.glsl_software_subgroup_statement_list(
            getattr(node, "body", None)
        )
        records = self.glsl_software_subgroup_operation_records(node.body)
        if len(records) != 1 or records[0][0] not in {
            "WaveActiveSum",
            "WaveActiveMin",
            "WaveActiveMax",
        }:
            return None
        operation_if_indices = [
            index
            for index, statement in enumerate(body_statements)
            if isinstance(statement, IfNode)
            and self.glsl_software_subgroup_operation_records(statement)
        ]
        if len(operation_if_indices) != 1:
            return None
        operation_if_index = operation_if_indices[0]
        operation_if = body_statements[operation_if_index]
        masked_if_plan = self.glsl_software_subgroup_masked_if_plan(operation_if)
        if masked_if_plan is None:
            return None

        if any(
            isinstance(child, (BreakNode, ContinueNode, ReturnNode))
            for child in self.walk_ast(node.body)
        ):
            return None
        bound_names = self.glsl_software_subgroup_expression_identifier_names(
            condition.right
        )
        for child in self.walk_ast(node.body):
            if isinstance(child, AssignmentNode):
                target_name = self.glsl_software_subgroup_assignment_target_name(child)
                if target_name == loop_name or target_name in bound_names:
                    return None
            elif isinstance(child, UnaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"++", "--"}:
                target_name = self.expression_name(getattr(child, "operand", None))
                if target_name == loop_name or target_name in bound_names:
                    return None
            elif isinstance(child, FunctionCallNode):
                observed = self.glsl_software_subgroup_expression_identifier_names(
                    child
                )
                if observed & bound_names:
                    return None

        prefix = body_statements[:operation_if_index]
        hoisted_prefix_declaration_indices = []
        for index, statement in enumerate(prefix):
            if isinstance(statement, ArrayNode):
                return None
            if not isinstance(statement, VariableNode):
                continue
            initial = getattr(statement, "initial_value", None)
            if initial is None or not self.glsl_software_subgroup_uniform_expression(
                initial, uniform_names
            ):
                return None
            hoisted_prefix_declaration_indices.append(index)

        return {
            "operationIfIndex": operation_if_index,
            "maskedIfPlan": masked_if_plan,
            "hoistedPrefixDeclarationIndices": tuple(
                hoisted_prefix_declaration_indices
            ),
            "loopType": loop_type,
            "step": step,
        }

    def prepare_glsl_software_subgroup_masked_if_plans(self, entry_function):
        self.glsl_software_subgroup_masked_if_plans = {}
        for statement in self.glsl_software_subgroup_statement_list(
            getattr(entry_function, "body", None)
        ):
            if not isinstance(statement, IfNode):
                continue
            if not self.glsl_software_subgroup_operation_records(statement):
                continue
            plan = self.glsl_software_subgroup_masked_if_plan(statement)
            if plan is not None:
                self.glsl_software_subgroup_masked_if_plans[id(statement)] = plan

    def glsl_software_subgroup_on_demand_masked_if_plan(self, node):
        """Plan an entry if whose guard the uniformity analysis cannot prove.

        Such an if may sit in a uniform loop, as MLX's RMS VJP row loop, or
        guard several reductions. Enclosing loops and branches are still
        proven uniform where they are generated, before their bodies.
        """
        if (
            self.software_subgroup_width is None
            or self.current_glsl_source_function_id
            != self.glsl_software_subgroup_entry_function_id
            or not self.glsl_software_subgroup_operation_records(node)
        ):
            return None
        return self.glsl_software_subgroup_masked_if_plan(node, allow_multiple=True)

    def prepare_glsl_software_subgroup_strided_for_plans(
        self,
        entry_function,
        uniform_names,
    ):
        self.glsl_software_subgroup_strided_for_plans = {}
        constants = self.initial_literal_int_constants(entry_function)
        for statement in self.glsl_software_subgroup_statement_list(
            getattr(entry_function, "body", None)
        ):
            if isinstance(statement, VariableNode):
                name = getattr(statement, "name", None)
                if name:
                    constants.pop(name, None)
                    if "const" in (getattr(statement, "qualifiers", []) or []):
                        value = evaluate_literal_int_expression(
                            getattr(statement, "initial_value", None),
                            constants,
                        )
                        if value is not None:
                            constants[name] = value
                continue
            if not isinstance(statement, ForNode):
                continue
            if not self.glsl_software_subgroup_operation_records(statement):
                continue
            plan = self.glsl_software_subgroup_strided_for_plan(
                statement,
                uniform_names,
                constants,
            )
            if plan is None:
                continue
            self.glsl_software_subgroup_strided_for_plans[id(statement)] = plan
            operation_if = self.glsl_software_subgroup_statement_list(statement.body)[
                plan["operationIfIndex"]
            ]
            self.glsl_software_subgroup_masked_if_plans[id(operation_if)] = plan[
                "maskedIfPlan"
            ]

    def prepare_glsl_software_subgroup_constant_guards(self, ast):
        (
            self.glsl_software_subgroup_constant_guards,
            self.glsl_software_subgroup_dead_node_ids,
        ) = glsl_constant_guards(self, self.glsl_software_subgroup_functions(ast))

    def glsl_software_subgroup_immutable_entry_names(self, entry_function, seeds):
        # Resolve call effects before the final convergence checks. Earlier
        # planning passes remain conservative while the call table is empty.
        immutable_names = set(seeds)
        uniform_builtins = set(self.GLSL_SOFTWARE_SUBGROUP_WORKGROUP_UNIFORM_BUILTINS)
        if self.glsl_software_subgroup_count == 1:
            uniform_builtins.add("gl_SubgroupID")
        for parameter in getattr(entry_function, "parameters", []) or []:
            if self.glsl_software_subgroup_constant_resource(parameter):
                continue
            semantic = self.semantic_from_node(parameter)
            mapped_semantic = self.map_semantic(semantic) if semantic else None
            qualifiers = {
                str(qualifier).strip().lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
            }
            if mapped_semantic is not None:
                if mapped_semantic not in uniform_builtins:
                    immutable_names.discard(getattr(parameter, "name", None))
            elif not qualifiers & {"constant", "uniform"}:
                immutable_names.discard(getattr(parameter, "name", None))
        for node in self.glsl_software_subgroup_reachable_nodes(
            getattr(entry_function, "body", None)
        ):
            invalidated = None
            if isinstance(node, VariableNode):
                immutable_names.discard(getattr(node, "name", None))
                if "&" in (
                    self.type_name_string(getattr(node, "var_type", None)) or ""
                ):
                    invalidated = getattr(node, "initial_value", None)
            elif isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                written = self.glsl_mutation_target_names(target)
                if written:
                    immutable_names.difference_update(written)
                    continue
                invalidated = target
            elif isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                "++",
                "--",
                "&",
            }:
                invalidated = node.operand
            elif isinstance(
                node, FunctionCallNode
            ) and not self.glsl_software_subgroup_exit_constructor(node):
                mutations = self.glsl_software_subgroup_call_mutations.get(id(node))
                if mutations is not None:
                    immutable_names.difference_update(mutations)
                    continue
                invalidated = getattr(node, "arguments", [])
            if invalidated is not None:
                immutable_names.difference_update(
                    self.glsl_software_subgroup_expression_identifier_names(invalidated)
                )
        return immutable_names

    def glsl_software_subgroup_exit_constructor(self, node):
        if not isinstance(node, FunctionCallNode):
            return False
        name = self.function_call_name(node)
        if name in self.function_definitions:
            return False
        constructor = target_arithmetic_type(self.glsl_constructor_type(name))
        return constructor is not None and (
            1 <= len(node.arguments) <= constructor.lanes
        )

    def glsl_software_subgroup_exit_facts(self, function, seeds, entry):
        names = set(self.GLSL_SOFTWARE_SUBGROUP_WORKGROUP_UNIFORM_BUILTINS) | set(seeds)
        components = {
            builtin: {
                axis
                for axis, size in enumerate(self.glsl_software_subgroup_workgroup_size)
                if size == 1
            }
            for builtin in ("gl_GlobalInvocationID", "gl_LocalInvocationID")
        }
        mutable = set()
        declarations = {}
        for parameter in function.parameters or []:
            name = parameter.name
            declarations[name] = declarations.get(name, 0) + 1
            if name not in seeds:
                names.discard(name)
            components.pop(name, None)
            semantic = self.map_semantic(self.semantic_from_node(parameter))
            if entry and semantic in {"gl_GlobalInvocationID", "gl_LocalInvocationID"}:
                components[name] = {
                    axis
                    for axis, size in enumerate(
                        self.glsl_software_subgroup_workgroup_size
                    )
                    if size == 1
                }
        for node in self.walk_ast(function.body):
            invalidated = None
            if isinstance(node, VariableNode):
                declarations[node.name] = declarations.get(node.name, 0) + 1
                if "&" in (self.type_name_string(node.var_type) or ""):
                    invalidated = node.initial_value
            elif isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                written = self.glsl_mutation_target_names(target)
                if written:
                    mutable.update(written)
                    continue
                invalidated = target
            elif isinstance(node, UnaryOpNode) and self.map_operator(node.operator) in {
                "++",
                "--",
                "&",
            }:
                invalidated = node.operand
            elif isinstance(
                node, FunctionCallNode
            ) and not self.glsl_software_subgroup_exit_constructor(node):
                mutations = self.glsl_software_subgroup_call_mutations.get(id(node))
                if mutations is not None:
                    mutable.update(mutations)
                    continue
                invalidated = node.arguments
            if invalidated is not None:
                mutable.update(
                    self.glsl_software_subgroup_expression_identifier_names(invalidated)
                )
        # Locals declared in several scopes but never written hold one value per
        # declaration; the flows scope each declaration's fact.
        parameters = {parameter.name for parameter in function.parameters or []}
        redeclared = {
            name
            for name, count in declarations.items()
            if count > 1 and name not in mutable and name not in parameters
        }
        mutable.update(name for name, count in declarations.items() if count > 1)
        # Immutable entry storage moved only by uniform-trackable offsets stays a
        # resource; the exit and argument flows follow its view.
        offsets = (
            followed_offsets(
                function,
                {
                    parameter.name
                    for parameter in function.parameters or []
                    if declarations.get(parameter.name) == 1
                    and self.glsl_software_subgroup_constant_resource(parameter)
                },
                self.glsl_software_subgroup_call_mutations,
                walk=self.walk_ast,
                map_operator=self.map_operator,
                identifier_names=self.glsl_software_subgroup_expression_identifier_names,
                write_roots=self.glsl_mutation_target_names,
            )
            if entry
            else set()
        )
        names.update(offsets)
        self.glsl_software_subgroup_resource_names = {
            declaration.name
            for declaration in [
                *(function.parameters or []),
                *(
                    node
                    for node in self.walk_ast(function.body)
                    if isinstance(node, VariableNode)
                ),
            ]
            if declarations[declaration.name] == 1
            and (declaration.name not in mutable or declaration.name in offsets)
            and self.glsl_software_subgroup_uniform_returns.resource_type(
                getattr(
                    declaration, "param_type", getattr(declaration, "var_type", None)
                ),
                declaration,
            )
        }
        names.difference_update(mutable - offsets)
        if (
            self.glsl_software_subgroup_count == 1
            and "gl_SubgroupID" not in mutable
            and "gl_SubgroupID" not in declarations
        ):
            names.add("gl_SubgroupID")
        components = {
            name: axes for name, axes in components.items() if name not in mutable
        }
        analysis = self.glsl_software_subgroup_uniform_returns
        # Locals clamped outside loops, as a causal attention block bound, stay
        # uniform where every write before the use is uniform.
        followed = followed_locals(
            function,
            mutable,
            self.glsl_software_subgroup_call_mutations,
            walk=self.walk_ast,
            map_operator=self.map_operator,
            identifier_names=self.glsl_software_subgroup_expression_identifier_names,
            write_roots=self.glsl_mutation_target_names,
            value_type=analysis.value_type,
            volatile=lambda node: analysis._qualified(node, {"volatile"}),
        )
        if entry:
            # A counter only its initializer-free loops write is uniform at
            # each loop when the flow proves it so; the loop proof does the rest.
            followed = set(followed) | {
                record[0]
                for record in getattr(
                    self, "glsl_software_subgroup_counter_loops", {}
                ).values()
            }
        # Locals a uniform loop advances on every trip, as a block-mask offset.
        carried = loop_carried_candidates(
            function,
            mutable,
            self.glsl_software_subgroup_call_mutations,
            walk=self.walk_ast,
            map_operator=self.map_operator,
            identifier_names=self.glsl_software_subgroup_expression_identifier_names,
            value_type=analysis.value_type,
            volatile=lambda node: analysis._qualified(node, {"volatile"}),
            write_roots=self.glsl_mutation_target_names,
            excluded=set(followed) | set(offsets),
        )
        return {
            "uniform_names": names,
            "uniform_components": components,
            "mutable_names": mutable,
            "offset_names": offsets,
            "followed_names": set(followed) | carried,
            "carried_names": carried,
            "redeclared_names": redeclared,
        }

    def glsl_software_subgroup_uniform_seed_names(
        self,
        ast,
        entry_function,
        target_stage=None,
    ):
        uniform_builtins = set(self.GLSL_SOFTWARE_SUBGROUP_WORKGROUP_UNIFORM_BUILTINS)
        if self.glsl_software_subgroup_count == 1:
            uniform_builtins.add("gl_SubgroupID")
        names = set(uniform_builtins)
        for node in (
            *(getattr(ast, "constants", []) or []),
            *(getattr(ast, "global_variables", []) or []),
        ):
            name = getattr(node, "name", None)
            qualifiers = {
                str(qualifier).strip().lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if name and (
                node in (getattr(ast, "constants", []) or [])
                or qualifiers & {"const", "constant", "uniform"}
            ):
                names.add(name)

        cbuffers = list(getattr(ast, "cbuffers", []) or [])
        cbuffers.extend(collect_stage_local_cbuffers(ast, target_stage))
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                name = getattr(member, "name", None)
                if name:
                    names.add(name)

        for parameter in getattr(entry_function, "parameters", []) or []:
            name = getattr(parameter, "name", None)
            if not name:
                continue
            qualifiers = {
                str(qualifier).strip().lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
            }
            semantic = self.semantic_from_node(parameter)
            mapped_semantic = self.map_semantic(semantic) if semantic else None
            if (
                mapped_semantic in uniform_builtins
                or (mapped_semantic is None and qualifiers & {"constant", "uniform"})
                or (self.glsl_software_subgroup_constant_resource(parameter))
            ):
                names.add(name)
                if mapped_semantic:
                    names.add(mapped_semantic)
            else:
                names.discard(name)
        return names

    def glsl_software_subgroup_constant_resource(self, parameter):
        """Whether an entry parameter is read-only storage nothing writes.

        Constant storage is immutable by type. Read-only storage is immutable
        only when a checked contract names the parameter, because another
        binding of the dispatch could otherwise alias it and write.
        """
        analysis = self.glsl_software_subgroup_uniform_returns
        return analysis.resource_type(parameter.param_type, parameter) and (
            analysis._qualified(parameter, {"constant"})
            or getattr(parameter.param_type, "address_space", None) == "constant"
            or getattr(parameter, "name", None) in self.glsl_immutable_storage_names
        )

    def glsl_software_subgroup_uniform_expression(self, expression, uniform_names):
        if expression is None:
            return False
        constants = self.glsl_active_literal_int_constants()
        if evaluate_literal_int_expression(expression, constants) is not None:
            return True
        return self.glsl_software_subgroup_exit_uniform_expression(
            expression, uniform_names, {}
        )

    def glsl_software_subgroup_assignment_target_name(self, node):
        target = getattr(node, "target", getattr(node, "left", None))
        return self.expression_name(target)

    def glsl_software_subgroup_assigned_names(self, root):
        names = set()
        for node in self.glsl_software_subgroup_reachable_nodes(root):
            if isinstance(node, AssignmentNode):
                name = self.glsl_software_subgroup_assignment_target_name(node)
                if name:
                    names.add(name)
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--"}:
                name = self.expression_name(getattr(node, "operand", None))
                if name:
                    names.add(name)
        return names

    def glsl_software_subgroup_expression_identifier_names(self, expression):
        return {
            node.name
            for node in self.walk_ast(expression)
            if isinstance(node, IdentifierNode) and getattr(node, "name", None)
        }

    def glsl_software_subgroup_dependency_names(self, expression):
        """Identifiers whose values or storage ``expression`` can observe.

        A call whose effect-free arguments have a dependency summary observes
        only the arguments that summary reads, and storage only through
        immutable resources. Arguments it merely forwards, such as resource
        identities passed to an accessor helper, cannot change its value.
        Every other node observes every identifier beneath it.
        """
        analysis = self.glsl_software_subgroup_uniform_returns
        resources = getattr(self, "glsl_software_subgroup_resource_names", set())
        names = set()
        visited = set()
        pending = [expression]
        while pending:
            value = pending.pop()
            if value is None or isinstance(value, (str, int, float, bool)):
                continue
            if isinstance(value, dict):
                pending.extend(value.values())
                continue
            if isinstance(value, (list, tuple, set)):
                pending.extend(value)
                continue
            if id(value) in visited:
                continue
            visited.add(id(value))
            if isinstance(value, FunctionCallNode):
                required = analysis.uniform_arguments(value, resources)
                if required is not None:
                    pending.extend(required)
                    continue
            if isinstance(value, IdentifierNode) and getattr(value, "name", None):
                names.add(value.name)
            if hasattr(value, "__dict__"):
                pending.extend(vars(value).values())
        return names

    def glsl_software_subgroup_analyze_uniform_if(self, node, uniform_names):
        fixed = getattr(self, "glsl_software_subgroup_constant_guards", {}).get(
            id(node)
        )
        if fixed is not None:
            # Every invocation takes the branch the constant selects.
            self.glsl_software_subgroup_uniform_if_node_ids.add(id(node))
            return self.glsl_software_subgroup_analyze_uniform_statements(
                node.if_body if fixed else getattr(node, "else_body", None),
                set(uniform_names),
            )
        conditions = [
            getattr(node, "condition", getattr(node, "if_condition", None)),
            *(getattr(node, "else_if_conditions", []) or []),
        ]
        bodies = [node.if_body, *(getattr(node, "else_if_bodies", []) or [])]
        else_body = getattr(node, "else_body", None)
        if not all(
            self.glsl_software_subgroup_uniform_expression(condition, uniform_names)
            for condition in conditions
        ):
            assigned = set()
            for body in [*bodies, else_body]:
                assigned.update(self.glsl_software_subgroup_assigned_names(body))
            return set(uniform_names) - assigned

        if all(
            self.glsl_software_subgroup_uniform_expression(
                condition, self.glsl_software_subgroup_immutable_uniform_names
            )
            for condition in conditions
        ):
            self.glsl_software_subgroup_uniform_if_node_ids.add(id(node))
        branch_results = [
            self.glsl_software_subgroup_analyze_uniform_statements(
                body,
                set(uniform_names),
            )
            for body in bodies
        ]
        branch_results.append(
            self.glsl_software_subgroup_analyze_uniform_statements(
                else_body,
                set(uniform_names),
            )
            if else_body is not None
            else set(uniform_names)
        )
        return (
            set.intersection(*branch_results) if branch_results else set(uniform_names)
        )

    def glsl_software_subgroup_analyze_uniform_statement(
        self,
        statement,
        uniform_names,
    ):
        names = set(uniform_names)
        if statement is None:
            return names
        if isinstance(statement, BlockNode) or hasattr(statement, "statements"):
            return self.glsl_software_subgroup_analyze_uniform_statements(
                statement,
                names,
            )
        if isinstance(statement, ExpressionStatementNode):
            return self.glsl_software_subgroup_analyze_uniform_statement(
                getattr(statement, "expression", None),
                names,
            )
        if isinstance(statement, VariableNode):
            name = getattr(statement, "name", None)
            if name:
                if self.glsl_software_subgroup_uniform_expression(
                    getattr(statement, "initial_value", None),
                    names,
                ):
                    names.add(name)
                else:
                    names.discard(name)
            return names
        if isinstance(statement, AssignmentNode):
            name = self.glsl_software_subgroup_assignment_target_name(statement)
            if not name:
                return names
            operator = self.map_operator(
                getattr(statement, "op", getattr(statement, "operator", "="))
            )
            value = getattr(statement, "value", getattr(statement, "right", None))
            value_is_uniform = self.glsl_software_subgroup_uniform_expression(
                value,
                names,
            )
            if value_is_uniform and (operator == "=" or name in names):
                names.add(name)
            else:
                names.discard(name)
            return names
        if isinstance(statement, UnaryOpNode):
            operator = self.map_operator(
                getattr(statement, "op", getattr(statement, "operator", None))
            )
            if operator in {"++", "--"}:
                name = self.expression_name(getattr(statement, "operand", None))
                if name not in names:
                    names.discard(name)
            return names
        if isinstance(statement, IfNode):
            return self.glsl_software_subgroup_analyze_uniform_if(statement, names)
        if isinstance(statement, ForNode):
            if id(statement) in self.glsl_software_subgroup_zero_trip_for_node_ids:
                return names
            is_uniform = self.glsl_software_subgroup_uniform_for(statement, names)
            # Only the entry is analyzed here, so its subgroup index is known.
            trip_uniform = (
                not is_uniform
                and id(statement) not in self.glsl_software_subgroup_strided_for_plans
                and self.glsl_software_subgroup_trip_uniform_for(statement, names)
            )
            contains_subgroup_work = bool(
                self.glsl_software_subgroup_first_operation(statement)
            ) or any(
                isinstance(child, FunctionCallNode)
                and self.glsl_software_subgroup_candidate_helper_call(child)
                for child in self.walk_ast(getattr(statement, "body", None))
            )
            if (is_uniform or trip_uniform) and contains_subgroup_work:
                self.glsl_software_subgroup_uniform_for_node_ids.add(id(statement))
            initializer = getattr(statement, "init", None)
            loop_name = getattr(initializer, "name", None)
            nested_names = set(names)
            if is_uniform and loop_name:
                nested_names.add(loop_name)
            elif trip_uniform:
                nested_names.discard(loop_name)
            counter = self.glsl_software_subgroup_counter_loops.get(id(statement))
            if (
                is_uniform
                and counter is not None
                and id(statement)
                in getattr(self, "glsl_software_subgroup_fresh_counter_loops", ())
            ):
                # Written only by this uniform loop's update from its literal
                # initializer, the counter is uniform before, inside and after
                # the loop.
                self.glsl_software_subgroup_immutable_uniform_names.add(counter[0])
            self.glsl_software_subgroup_analyze_uniform_statements(
                getattr(statement, "body", None),
                nested_names,
            )
            written = self.glsl_software_subgroup_assigned_names(
                getattr(statement, "body", None)
            )
            if not (is_uniform and counter is not None):
                # An update or condition may write a counter declared outside
                # the loop, as an initializer-free for does.
                for part in ("init", "condition", "update"):
                    written |= self.glsl_software_subgroup_assigned_names(
                        getattr(statement, part, None)
                    )
                written.discard(loop_name)
            return names - (written & names)
        if isinstance(
            statement, (WhileNode, DoWhileNode, LoopNode, SwitchNode, MatchNode)
        ):
            return names - self.glsl_software_subgroup_assigned_names(statement)
        if isinstance(statement, FunctionCallNode):
            # CrossGL parameters may be references. Without a resolved pure-call
            # contract, a call that observes a proven-uniform name may mutate it.
            observed = self.glsl_software_subgroup_expression_identifier_names(
                statement
            )
            return names - observed
        return names

    def glsl_software_subgroup_candidate_helper_call(self, call):
        targets = getattr(self, "glsl_software_subgroup_resolved_call_targets", {})
        if id(call) in targets:
            return (
                targets[id(call)]
                in self.glsl_software_subgroup_candidate_helper_function_ids
            )
        name = self.function_call_name(call)
        if name not in self.glsl_function_overloads_by_name:
            return False
        arguments = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        try:
            target = self.resolve_glsl_function_overload(
                name,
                arguments,
                call_node=call,
            )
        except OpenGLMappedOverloadError:
            return False
        return target is not None and id(target) in getattr(
            self,
            "glsl_software_subgroup_candidate_helper_function_ids",
            set(),
        )

    def glsl_software_subgroup_analyze_uniform_statements(
        self,
        body,
        uniform_names,
    ):
        names = set(uniform_names)
        for statement in self.glsl_software_subgroup_statement_list(body):
            names = self.glsl_software_subgroup_analyze_uniform_statement(
                statement,
                names,
            )
        return names

    def glsl_software_subgroup_first_operation(self, root):
        records = self.glsl_software_subgroup_operation_records(root)
        return records[0] if records else None
