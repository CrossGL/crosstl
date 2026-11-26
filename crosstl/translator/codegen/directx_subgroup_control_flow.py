"""Software subgroup control flow in the HLSL generator: uniform and masked execution, loop guards and trip counts, strided plans, static counters, and aliased and preserved values."""

from ..arithmetic_conversions import narrow_integer_shape
from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    ReturnNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
    WhileNode,
)
from .array_utils import (
    _UnsignedLiteralInt,
    evaluate_literal_int_expression,
)
from .counter_loops import (
    any_start_counter_loop_terminates,
    counter_update,
    wrapping_counter_loop_uniform,
)
from .loop_induction import (
    checked_induction_value,
    metadata_bound_key,
    offset_counter_loop_terminates,
    static_loop_trip_count,
    subgroup_strided_trips,
    walk_control_scopes,
)
from .pointer_reinterpret import scalar_storage_layout
from .subgroup_reassignments import member_write_root


class DirectXSubgroupControlFlowMixin:
    """Subgroup control flow members of ``HLSLCodeGen``."""

    def hlsl_software_subgroup_uniform_for(
        self,
        node,
        uniform_names,
        call_mutations,
        uniform_components,
        loop_constants=None,
        metadata_bounds=None,
    ):
        initializer = getattr(node, "init", None)
        if initializer is None:
            return self.hlsl_software_subgroup_counter_loop_uniform(
                node,
                uniform_names,
                call_mutations,
                uniform_components,
                loop_constants,
                metadata_bounds,
            )
        if not isinstance(initializer, VariableNode):
            return False
        loop_name = getattr(initializer, "name", None)
        loop_type = self.map_type(getattr(initializer, "var_type", None))
        if not loop_name or loop_type not in {
            "int",
            "uint",
            "int16_t",
            "uint16_t",
            "min16int",
            "min16uint",
        }:
            return False
        constants = {
            name: value
            for name, value in (
                self.literal_int_constants if loop_constants is None else loop_constants
            ).items()
            if name in uniform_names and name != loop_name
        }
        if self.hlsl_software_subgroup_static_for(node, constants, call_mutations):
            return True
        initial_expression = getattr(initializer, "initial_value", None)
        uniform_components = {
            name: components
            for name, components in uniform_components.items()
            if name != loop_name
        }
        initial_value = evaluate_literal_int_expression(initial_expression, constants)
        if (
            initial_value is None
            and not self.hlsl_software_subgroup_uniform_expression(
                initial_expression, set(uniform_names) - {loop_name}, uniform_components
            )
        ):
            return False

        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return False
        operator = self.map_operator(
            getattr(condition, "op", getattr(condition, "operator", None))
        )
        if operator not in {"<", "<=", ">", ">="}:
            return False
        left_name = self.expression_name(condition.left)
        right_name = self.expression_name(condition.right)
        if left_name == loop_name and right_name != loop_name:
            bound = condition.right
            loop_on_left = True
        elif right_name == loop_name and left_name != loop_name:
            bound = condition.left
            loop_on_left = False
        else:
            return False
        literal_bound = evaluate_literal_int_expression(bound, constants)
        if (
            literal_bound is None
            and not self.hlsl_software_subgroup_uniform_expression(
                bound, set(uniform_names) | {loop_name}, uniform_components
            )
        ):
            return False

        update = getattr(node, "update", None)
        halving_update = False
        additive_update = False
        geometric_update = False
        modular_update = False
        if isinstance(update, UnaryOpNode):
            update_operator = self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            if update_operator not in {"++", "--"}:
                return False
            if self.expression_name(getattr(update, "operand", None)) != loop_name:
                return False
            increasing = update_operator == "++"
        elif isinstance(update, AssignmentNode):
            update_operator = self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            if (
                update_operator not in {"/=", ">>=", "+=", "-=", "*=", "<<="}
                or self.hlsl_software_subgroup_assignment_target_name(update)
                != loop_name
            ):
                return False
            update_value = getattr(update, "value", getattr(update, "right", None))
            concrete_update_value = evaluate_literal_int_expression(
                update_value, constants
            )
            if concrete_update_value is None:
                # A modular update visits the same values in every invocation
                # for any uniform operand, so it needs no operand range.
                if (
                    update_operator != "+="
                    or id(node)
                    not in getattr(
                        self, "hlsl_software_subgroup_modular_stride_loops", ()
                    )
                    or not self.hlsl_software_subgroup_uniform_expression(
                        update_value,
                        set(uniform_names) - {loop_name},
                        uniform_components,
                    )
                ):
                    return False
                increasing = True
                modular_update = True
            elif update_operator in {"+=", "-="}:
                step = concrete_update_value * (1 if update_operator == "+=" else -1)
                increasing = step > 0
                additive_update = True
            elif update_operator in {"*=", "<<="}:
                increasing = True
                geometric_update = True
            else:
                if (
                    update_operator == "/="
                    and concrete_update_value < 2
                    or update_operator == ">>="
                    and concrete_update_value < 1
                ):
                    return False
                increasing = False
                halving_update = True
        else:
            return False

        normalized_operator = (
            operator
            if loop_on_left
            else {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[operator]
        )
        if halving_update:
            canonical_positive_bound = (
                normalized_operator == ">" and literal_bound == 0
            ) or (normalized_operator == ">=" and literal_bound == 1)
            if not canonical_positive_bound:
                return False
        elif increasing != (normalized_operator in {"<", "<="}):
            return False
        checked_stride = not modular_update and (
            additive_update
            or geometric_update
            or metadata_bound_key(bound) in (metadata_bounds or {})
            or (
                isinstance(update, UnaryOpNode)
                and narrow_integer_shape(self.type_name_string(initializer.var_type))
                is not None
            )
        )
        if checked_stride and not self.hlsl_software_subgroup_fixed_stride(
            initializer,
            bound,
            update,
            constants,
            normalized_operator,
            metadata_bounds,
            any_start=initial_value is None,
        ):
            return False

        bound_names = self.hlsl_software_subgroup_expression_identifier_names(bound)
        stride_names = (
            self.hlsl_software_subgroup_expression_identifier_names(update.value)
            if additive_update or geometric_update or modular_update
            else set()
        )
        return self.hlsl_software_subgroup_loop_controls_preserved(
            node,
            bound_names | stride_names | {loop_name},
            call_mutations,
            uniform_exits=True,
        )

    def hlsl_software_subgroup_static_for(self, node, constants, call_mutations):
        """Accept a loop whose controls read only literal constants.

        MLX's ``for (int step = 1; (1 << step) < C; step++)`` runs the same
        checked trips in every invocation when its body leaves the counter
        and the constants alone.
        """
        initializer = getattr(node, "init", None)
        analysis = self.hlsl_software_subgroup_uniform_returns
        if (
            not isinstance(initializer, VariableNode)
            or analysis._qualified(initializer, {"volatile"})
            or analysis._qualified(initializer.var_type, {"volatile"})
        ):
            return False
        trips = static_loop_trip_count(
            self.type_name_string(initializer.var_type),
            initializer.initial_value,
            getattr(node, "condition", None),
            getattr(node, "update", None),
            initializer.name,
            constants,
        )
        if trips is None:
            return False
        controls = {initializer.name}
        for part in (node.condition, node.update):
            controls |= self.hlsl_software_subgroup_expression_identifier_names(part)
        return self.hlsl_software_subgroup_loop_controls_preserved(
            node, controls, call_mutations, uniform_exits=True
        )

    def hlsl_software_subgroup_counter_loop_uniform(
        self,
        node,
        uniform_names,
        call_mutations,
        uniform_components,
        loop_constants=None,
        metadata_bounds=None,
    ):
        """Accept MLX's ``int t = 0; for (; t + C <= T; t += C)`` chunk loop.

        ``init_less_counter_loops`` proves the counter is written only by this
        loop's update and reached with its initializer's value. With a uniform
        bound, a literal offset and stride, and a body that leaves them alone,
        every invocation runs the same trips, so the counter stays uniform
        after the loop too. The checked termination proof uses the bound's
        literal value or metadata range; a 32-bit unsigned counter, which
        wraps alike on every target, needs none. ``any_start_counter_loops`` adds
        loops, such as ``for (; k > 0; k--)`` in each branch of an ``if``,
        that the counter may reach with any value; the flow analysis proves
        that value uniform and termination must hold from every start.
        """
        record = getattr(self, "hlsl_software_subgroup_counter_loops", {}).get(id(node))
        if record is None:
            return False
        name, declaration, _loop = record
        analysis = self.hlsl_software_subgroup_uniform_returns
        if (
            name not in uniform_names
            or analysis._qualified(declaration, {"volatile"})
            or analysis._qualified(declaration.var_type, {"volatile"})
        ):
            return False
        constants = {
            key: value
            for key, value in (
                self.literal_int_constants if loop_constants is None else loop_constants
            ).items()
            if key in uniform_names and key != name
        }
        # The initializer runs at the declaration, where a later local may
        # not yet shadow a constant it names; only literals are proven. A loop
        # the counter may reach with another value proves every start.
        start = (
            checked_induction_value(declaration.initial_value, {})
            if id(node)
            in getattr(self, "hlsl_software_subgroup_fresh_counter_loops", ())
            else None
        )
        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return False
        operator = self.map_operator(condition.operator)
        left = condition.left
        offset = 0
        offset_node = None
        offset_names = set()
        if isinstance(left, BinaryOpNode) and self.map_operator(left.operator) == "+":
            if self.expression_name(left.left) == name:
                offset_node = left.right
            elif self.expression_name(left.right) == name:
                offset_node = left.left
            else:
                return False
            offset = checked_induction_value(offset_node, constants)
            offset_names = self.hlsl_software_subgroup_expression_identifier_names(
                offset_node
            )
            if offset is None or name in offset_names:
                return False
        elif self.expression_name(left) != name:
            return False
        bound = condition.right
        bound_names = self.hlsl_software_subgroup_expression_identifier_names(bound)
        if name in bound_names:
            return False
        literal_bound = checked_induction_value(bound, constants)
        if literal_bound is not None:
            stops = [literal_bound]
        else:
            if not self.hlsl_software_subgroup_uniform_expression(
                bound, uniform_names, uniform_components
            ):
                return False
            record = (metadata_bounds or {}).get(metadata_bound_key(bound))
            stops = []
            if record is not None:
                convert = (
                    _UnsignedLiteralInt if record["dtype"].startswith("u") else int
                )
                stops = [convert(record["minimum"]), convert(record["maximum"])]
        update = counter_update(node.update)
        if update is None or update[0] != name:
            return False
        _counter, sign, stride = update
        stride_names = set()
        step = sign
        if stride is not None:
            magnitude = checked_induction_value(stride, constants)
            step = None if magnitude is None else sign * magnitude
            stride_names = self.hlsl_software_subgroup_expression_identifier_names(
                stride
            )
        type_name = self.type_name_string(declaration.var_type)
        terminates = bool(stops) and (
            start is not None
            and offset_counter_loop_terminates(
                type_name, start, offset, stops, step, operator
            )
            or any_start_counter_loop_terminates(
                type_name, offset, stops, step, operator
            )
        )
        rebased = None
        if (
            step is not None
            and stops
            and not terminates
            and offset_node is not None
            and self.hlsl_rebased_offset_counter_loop_terminates(
                type_name, offset, stops, step, operator
            )
        ):
            # Emitted as ``c OP stop - offset``, which cannot overflow.
            rebased = self.hlsl_rebased_offset_condition(condition, offset_node, bound)
            terminates = True
        if step is None or not (
            terminates
            or wrapping_counter_loop_uniform(type_name, offset, step, operator)
        ):
            return False
        if not self.hlsl_software_subgroup_loop_controls_preserved(
            node, {name} | bound_names | offset_names | stride_names, call_mutations
        ):
            return False
        self.hlsl_software_subgroup_proven_counter_loops.add(id(node))
        if rebased is not None:
            self.hlsl_software_subgroup_rebased_conditions[id(node)] = rebased
        return True

    def hlsl_software_subgroup_loop_controls_preserved(
        self, node, control_names, call_mutations, *, uniform_exits=False
    ):
        """Whether ``node``'s body leaves its counter, bound and stride alone.

        The body may not exit the loop early, redeclare a control in its
        outermost scope, write one, take its address or pass it to a
        possibly mutating call. With ``uniform_exits``, an exit every
        invocation takes together keeps the trips of all invocations equal.
        """
        for child, visible in walk_control_scopes(
            getattr(node, "body", None), control_names
        ):
            if isinstance(child, (BreakNode, ContinueNode, ReturnNode)):
                if uniform_exits and self.hlsl_software_subgroup_uniform_exit(child):
                    continue
                return False
            if isinstance(child, VariableNode) and child.name in visible:
                return False
            if (
                isinstance(child, VariableNode)
                and isinstance(child.var_type, (PointerType, ReferenceType))
                and call_mutations.get(
                    id(child),
                    self.hlsl_software_subgroup_aliased_names(child.initial_value),
                )
                & visible
            ):
                return False
            if (
                isinstance(child, UnaryOpNode)
                and self.map_operator(child.operator) == "&"
                and self.hlsl_software_subgroup_aliased_names(child) & visible
            ):
                return False
            if call_mutations.get(id(child), set()) & visible:
                return False
            if isinstance(child, AssignmentNode):
                target_name = self.hlsl_software_subgroup_assignment_target_name(child)
                if target_name in visible:
                    return False
            if isinstance(child, UnaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"++", "--"}:
                target_name = self.expression_name(getattr(child, "operand", None))
                if target_name in visible:
                    return False
        return True

    def hlsl_software_subgroup_aliased_names(self, expression):
        """Names whose storage an address expression may alias.

        ``&vals[i * 2]`` and ``&s.rows[i]`` designate an element of ``vals``
        or ``s``: the index is only read, so it is never aliased. Any other
        expression conservatively aliases every name it reads.
        """
        node = None
        if isinstance(expression, UnaryOpNode) and (
            self.map_operator(getattr(expression, "operator", None)) == "&"
        ):
            node = expression.operand
        while isinstance(node, (ArrayAccessNode, MemberAccessNode, SwizzleNode)):
            if isinstance(node, ArrayAccessNode):
                node = node.array_expr
            elif isinstance(node, MemberAccessNode):
                node = node.object_expr
            else:
                node = node.vector_expr
        if isinstance(node, IdentifierNode):
            return {node.name}
        return self.hlsl_software_subgroup_expression_identifier_names(expression)

    def hlsl_software_subgroup_trip_uniform_for(
        self, node, uniform_names, call_mutations, constants
    ):
        """Accept ``for (i = simd_gid; i < B; i += S)`` whose trips agree.

        Each subgroup starts at its own index, so the counter is not uniform
        and the caller keeps it out of the body's uniform names. When the
        literal bound and stride give every subgroup the same trip count, as
        in MLX's Winograd ``for (c = simd_group_id; c < 32; c += 4)`` over four
        SIMD-groups, every invocation still reaches the body's collectives
        equally often and in the same order.
        """
        invocation_count = self.hlsl_software_subgroup_invocation_count
        initializer = getattr(node, "init", None)
        if (
            self.software_subgroup_width is None
            or not invocation_count
            or not isinstance(initializer, VariableNode)
            or not initializer.name
            or self.hlsl_software_subgroup_uniform_returns._qualified(
                initializer, {"volatile"}
            )
            or self.hlsl_software_subgroup_uniform_returns._qualified(
                initializer.var_type, {"volatile"}
            )
            or not self.hlsl_software_subgroup_subgroup_id_expression(
                getattr(initializer, "initial_value", None)
            )
        ):
            return False
        loop_name = initializer.name
        condition = getattr(node, "condition", None)
        if (
            not isinstance(condition, BinaryOpNode)
            or self.expression_name(condition.left) != loop_name
            or loop_name
            in self.hlsl_software_subgroup_expression_identifier_names(condition.right)
        ):
            return False
        operator = self.map_operator(
            getattr(condition, "op", getattr(condition, "operator", None))
        )
        constants = {
            name: value
            for name, value in constants.items()
            if name in uniform_names and name != loop_name
        }
        update = getattr(node, "update", None)
        stride_names = set()
        if isinstance(update, UnaryOpNode):
            if (
                self.map_operator(
                    getattr(update, "op", getattr(update, "operator", None))
                )
                != "++"
                or self.expression_name(getattr(update, "operand", None)) != loop_name
            ):
                return False
            step = 1
        elif (
            isinstance(update, AssignmentNode)
            and self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            == "+="
            and self.hlsl_software_subgroup_assignment_target_name(update) == loop_name
        ):
            value = getattr(update, "value", getattr(update, "right", None))
            step = evaluate_literal_int_expression(value, constants)
            stride_names = self.hlsl_software_subgroup_expression_identifier_names(
                value
            )
        else:
            return False
        trips = subgroup_strided_trips(
            self.type_name_string(initializer.var_type),
            evaluate_literal_int_expression(condition.right, constants),
            step,
            operator,
            -(-invocation_count // self.software_subgroup_width),
        )
        return (
            trips is not None
            and self.hlsl_software_subgroup_loop_controls_preserved(
                node,
                {loop_name}
                | stride_names
                | self.hlsl_software_subgroup_expression_identifier_names(
                    condition.right
                ),
                call_mutations,
            )
        )

    def hlsl_software_subgroup_contains_work(self, root, dependent_calls):
        if self.hlsl_software_subgroup_operation_records(root):
            return True
        return any(
            isinstance(node, FunctionCallNode) and id(node) in dependent_calls
            for node in self.walk_ast(root)
        )

    def hlsl_software_subgroup_written_names(self, root):
        """Return the names that assignments and increments in ``root`` write."""
        names = set()
        for node in self.walk_ast(root):
            if isinstance(node, AssignmentNode):
                names.add(self.hlsl_software_subgroup_assignment_target_name(node))
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--"}:
                names.add(self.expression_name(getattr(node, "operand", None)))
        names.discard(None)
        return names

    def hlsl_software_subgroup_reassigned_candidates(
        self, function, mutable_names, call_mutations
    ):
        """Return written locals whose uniformity can follow their writes.

        Each is a value declared once and written only by plain
        ``name = value;`` statements outside loops, as MLX's RMS VJP clamps
        ``row_end`` to ``n_rows``, or by member writes such as a resource
        view's ``view.offset += stride;`` outside loops. A local declared
        without an initializer is not uniform until a write makes it so, as
        ``uint k_start; if (c) { k_start = a; } else { k_start = b; }``.
        Address-taking, increments, other writes, reference views and possibly
        mutating calls disqualify a name.
        """
        body = getattr(function, "body", None)
        parameters = {
            parameter.name
            for parameter in getattr(function, "parameters", []) or []
            if getattr(parameter, "name", None)
        }
        counts = {}
        declarations = {}
        for node in self.walk_ast(body):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                counts[node.name] = counts.get(node.name, 0) + 1
                declarations[node.name] = node
        analysis = self.hlsl_software_subgroup_uniform_returns
        candidates = {
            name
            for name, declaration in declarations.items()
            if counts[name] == 1
            and name not in parameters
            and name in mutable_names
            and analysis.value_type(declaration.var_type)
            and not analysis._qualified(declaration, {"volatile"})
            and not analysis._qualified(declaration.var_type, {"volatile"})
        }
        if not candidates:
            return set()
        followed_writes = set()

        def visit(statements, in_loop):
            for statement in self.hlsl_software_subgroup_statement_list(statements):
                node = statement
                if isinstance(node, ExpressionStatementNode):
                    node = getattr(node, "expression", None)
                if isinstance(node, AssignmentNode):
                    if not in_loop and (
                        (
                            isinstance(
                                getattr(node, "target", getattr(node, "left", None)),
                                IdentifierNode,
                            )
                            and self.map_operator(
                                getattr(node, "operator", getattr(node, "op", "="))
                            )
                            == "="
                        )
                        or member_write_root(node, self.map_operator)
                    ):
                        followed_writes.add(id(node))
                elif isinstance(node, IfNode):
                    visit(node.then_branch, in_loop)
                    for branch in getattr(node, "else_if_bodies", []) or []:
                        visit(branch, in_loop)
                    visit(node.else_branch, in_loop)
                elif isinstance(
                    node, (ForNode, WhileNode, DoWhileNode, LoopNode, ForInNode)
                ):
                    visit(getattr(node, "body", None), True)
                elif isinstance(node, BlockNode):
                    visit(node, in_loop)

        visit(body, False)
        for node in self.walk_ast(body):
            if isinstance(node, AssignmentNode):
                if id(node) not in followed_writes:
                    candidates.discard(
                        self.hlsl_software_subgroup_assignment_target_name(node)
                    )
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--", "&"}:
                candidates.difference_update(
                    self.hlsl_software_subgroup_expression_identifier_names(
                        node.operand
                    )
                )
            elif isinstance(node, FunctionCallNode):
                candidates.difference_update(call_mutations.get(id(node), set()))
            elif isinstance(node, VariableNode) and isinstance(
                getattr(node, "var_type", None), (PointerType, ReferenceType)
            ):
                candidates.difference_update(
                    self.hlsl_software_subgroup_expression_identifier_names(
                        node.initial_value
                    )
                )
        return candidates

    # Identity of each predicated reduction. Signed zero is the exact float
    # sum identity, so inactive invocations cannot change a -0.0 result, and
    # the infinities bound every non-NaN float for min and max. Binary16 sums
    # round every pairwise partial, so a masked binary16 sum additionally
    # requires a guard that every invocation of a subgroup evaluates alike.
    HLSL_SOFTWARE_SUBGROUP_MASKED_IDENTITIES = {
        "WaveActiveSum": {
            "float": "asfloat(0x80000000u)",
            "float16_t": "asfloat16(uint16_t(0x8000u))",
            "int": "0",
            "uint": "0u",
        },
        "WaveActiveMin": {
            "float": "asfloat(0x7f800000u)",
            "float16_t": "asfloat16(uint16_t(0x7c00u))",
            "int": "2147483647",
            "uint": "0xffffffffu",
        },
        "WaveActiveMax": {
            "float": "asfloat(0xff800000u)",
            "float16_t": "asfloat16(uint16_t(0xfc00u))",
            "int": "(-2147483647 - 1)",
            "uint": "0u",
        },
    }
    HLSL_SOFTWARE_SUBGROUP_UNIFORM_GUARD_OPERATORS = {
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "&&",
        "||",
        "!",
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "<<",
        ">>",
    }

    def hlsl_software_subgroup_conversion_chain(self, expression):
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

    def hlsl_software_subgroup_guard_is_subgroup_uniform(
        self, condition, subgroup_names, constant_names
    ):
        """Whether every invocation of one software subgroup agrees on a guard.

        The guard may combine the entry's subgroup index, which the entry never
        writes, with literals and compile-time constants. Each subgroup then
        takes part in a predicated reduction completely or not at all.
        """

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
                    operator in self.HLSL_SOFTWARE_SUBGROUP_UNIFORM_GUARD_OPERATORS
                    and uniform(expression.left)
                    and uniform(expression.right)
                )
            if isinstance(expression, UnaryOpNode):
                operator = self.map_operator(
                    getattr(expression, "op", getattr(expression, "operator", None))
                )
                return operator in {"!", "-", "~"} and uniform(expression.operand)
            if isinstance(expression, FunctionCallNode):
                wrappers, inner = self.hlsl_software_subgroup_conversion_chain(
                    expression
                )
                return bool(wrappers) and uniform(inner)
            return False

        return bool(subgroup_names) and uniform(condition)

    def hlsl_software_subgroup_entry_unwritten_names(self, entry_function, names):
        """Return the ``names`` that the entry never rebinds or writes."""
        names = set(names)
        for node in self.walk_ast(getattr(entry_function, "body", None)):
            if isinstance(node, VariableNode):
                names.discard(node.name)
            elif isinstance(node, AssignmentNode):
                names.discard(self.hlsl_software_subgroup_assignment_target_name(node))
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

    def hlsl_software_subgroup_entry_subgroup_names(self, entry_function):
        """Return subgroup-index parameters the entry never rebinds or writes."""
        return self.hlsl_software_subgroup_entry_unwritten_names(
            entry_function,
            {
                parameter.name
                for parameter in getattr(entry_function, "parameters", []) or []
                if getattr(parameter, "name", None)
                and "gl_SubgroupID"
                in {
                    self.semantic_from_node(parameter),
                    self.hlsl_canonical_semantic(self.semantic_from_node(parameter)),
                }
            },
        )

    def hlsl_software_subgroup_direct_operation(self, expression):
        records = self.hlsl_software_subgroup_operation_records(expression)
        if len(records) != 1 or records[0][1] is not expression:
            return None
        operation, node = records[0]
        arguments = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
        return operation, arguments, node

    def hlsl_software_subgroup_masked_reduction(self, statement, record_node):
        """Return ``(operation, wrapped)`` when ``statement`` is ``x = op(v)``.

        ``op`` must be ``record_node`` itself, optionally under scalar
        conversions, and ``x`` a plain name. Return ``None`` when the statement
        has another shape and ``False`` when its shape cannot be predicated.
        """
        candidate = statement
        if isinstance(candidate, ExpressionStatementNode):
            candidate = getattr(candidate, "expression", None)
        if not isinstance(candidate, AssignmentNode):
            return None
        if (
            self.map_operator(
                getattr(candidate, "operator", getattr(candidate, "op", "="))
            )
            != "="
        ):
            return None
        wrappers, inner = self.hlsl_software_subgroup_conversion_chain(
            getattr(candidate, "value", getattr(candidate, "right", None))
        )
        direct = self.hlsl_software_subgroup_direct_operation(inner)
        if direct is None or direct[2] is not record_node or len(direct[1]) != 1:
            return None
        # Converted results reuse the source assignment, whose reduction is
        # replaced by the predicated result; that substitution needs the
        # canonical wave node.
        if wrappers and not isinstance(direct[2], WaveOpNode):
            return False
        target = getattr(candidate, "target", getattr(candidate, "left", None))
        if not isinstance(target, (IdentifierNode, VariableNode)) or not getattr(
            target, "name", None
        ):
            return False
        return direct[0], bool(wrappers)

    def hlsl_software_subgroup_masked_if_plan(
        self,
        node,
        subgroup_names=frozenset(),
        constant_names=frozenset(),
        *,
        allow_multiple=False,
        allow_hoisted_target=False,
    ):
        """Plan ``if (c) { prefix; x = simd_sum(v); suffix }`` as a masked sum.

        Every invocation then runs the reduction with the operation's identity
        when ``c`` is false, which matches a reduction over the active lanes.
        The reduction may sit under scalar conversions, as in
        ``x = bfloat16(simd_max(float(v)))``.

        With ``allow_multiple``, every top-level statement of the body that
        holds a collective must be such a reduction, as in MLX's RMS VJP
        ``if (c) { a = simd_sum(u); b = simd_sum(v); }``. The reductions then
        run in source order in every invocation, and each statement between
        them still runs only where ``c`` held, after the previous assignment.

        With ``allow_hoisted_target``, the prefix of a single reduction may
        declare its target once with a speculatively safe initializer, as in
        MLX's ``U score = 0; ...; score = simd_sum(score);``. The declaration
        then precedes the guard, so the target stays in scope for the suffix.
        """

        if (
            getattr(node, "else_if_conditions", None)
            or getattr(node, "else_if_bodies", None)
            or getattr(node, "else_branch", getattr(node, "else_body", None))
        ):
            return None
        condition = getattr(node, "condition", getattr(node, "if_condition", None))
        body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if (
            condition is None
            or self.hlsl_expression_has_observable_side_effects(condition)
            or self.hlsl_software_subgroup_operation_records(condition)
        ):
            return None
        records = self.hlsl_software_subgroup_operation_records(body)
        if (
            not records
            or len(records) != 1
            and not allow_multiple
            or any(
                operation not in self.HLSL_SOFTWARE_SUBGROUP_MASKED_IDENTITIES
                for operation, _node in records
            )
        ):
            return None
        if any(
            isinstance(child, (BreakNode, ContinueNode, ReturnNode))
            for child in self.walk_ast(body)
        ):
            return None
        statements = self.hlsl_software_subgroup_statement_list(body)
        indices = []
        operations = []
        wrapped = []
        if len(records) == 1:
            for index, statement in enumerate(statements):
                reduction = self.hlsl_software_subgroup_masked_reduction(
                    statement, records[0][1]
                )
                if reduction is None:
                    continue
                if reduction is False:
                    return None
                indices.append(index)
                operations.append(reduction[0])
                wrapped.append(reduction[1])
                break
        else:
            pending = [node for _operation, node in records]
            for index, statement in enumerate(statements):
                if not self.hlsl_software_subgroup_operation_records(statement):
                    continue
                reduction = (
                    self.hlsl_software_subgroup_masked_reduction(statement, pending[0])
                    if pending
                    else None
                )
                if not reduction:
                    return None
                pending.pop(0)
                indices.append(index)
                operations.append(reduction[0])
                wrapped.append(reduction[1])
            if pending:
                return None
        if not indices:
            return None
        hoisted_index = None
        if allow_hoisted_target and len(indices) == 1:
            assignment = statements[indices[0]]
            if isinstance(assignment, ExpressionStatementNode):
                assignment = assignment.expression
            target_name = self.expression_name(
                getattr(assignment, "target", getattr(assignment, "left", None))
            )
            declarations = [
                index
                for index, prefix in enumerate(statements[: indices[0]])
                if isinstance(prefix, VariableNode) and prefix.name == target_name
            ]
            if len(declarations) == 1:
                declaration = statements[declarations[0]]
                if self.hlsl_software_subgroup_speculatively_safe_expression(
                    getattr(declaration, "initial_value", None)
                ):
                    hoisted_index = declarations[0]
        # Each later block starts a new scope, so declarations before the last
        # reduction would fall out of scope.
        if any(
            isinstance(prefix, (VariableNode, ArrayNode))
            for index, prefix in enumerate(statements[: indices[-1]])
            if index != hoisted_index
        ):
            return None
        plan = {
            "statementIndex": indices[0],
            "operation": operations[0],
            "wrapped": wrapped[0],
            "subgroupUniform": self.hlsl_software_subgroup_guard_is_subgroup_uniform(
                condition, subgroup_names, constant_names
            ),
        }
        if hoisted_index is not None:
            plan["hoistedTargetDeclarationIndex"] = hoisted_index
        if len(indices) > 1:
            plan.update(
                statementIndices=tuple(indices),
                operations=tuple(operations),
                wrappedReductions=tuple(wrapped),
            )
        return plan

    def prepare_hlsl_software_subgroup_masked_if_plans(self, entry_function):
        self.hlsl_software_subgroup_masked_if_plans = {}
        subgroup_names = self.hlsl_software_subgroup_entry_subgroup_names(
            entry_function
        )
        constant_names = self.hlsl_software_subgroup_entry_unwritten_names(
            entry_function, self.initial_literal_int_constants(entry_function)
        )
        # Validation plans guarded ifs that it cannot prove uniform with the
        # same facts.
        self.hlsl_software_subgroup_guard_names = subgroup_names
        self.hlsl_software_subgroup_guard_constant_names = constant_names
        for statement in self.hlsl_software_subgroup_statement_list(
            getattr(entry_function, "body", None)
        ):
            if not isinstance(
                statement, IfNode
            ) or not self.hlsl_software_subgroup_operation_records(statement):
                continue
            plan = self.hlsl_software_subgroup_masked_if_plan(
                statement, subgroup_names, constant_names
            )
            if plan is not None:
                self.hlsl_software_subgroup_masked_if_plans[id(statement)] = plan

    def hlsl_software_subgroup_speculatively_safe_expression(self, expression):
        """Whether ``expression`` may also run where its guard is false.

        A hoisted initializer runs in every invocation, so it must not write,
        read memory, dereference, divide or shift.
        """
        if expression is None or self.hlsl_expression_has_observable_side_effects(
            expression, allow_numeric_constructors=True
        ):
            return False
        for child in self.walk_ast(expression):
            if isinstance(child, (ArrayAccessNode, MemberAccessNode)):
                return False
            if isinstance(child, UnaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"*", "&"}:
                return False
            if isinstance(child, BinaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"/", "%", "<<", ">>"}:
                return False
            if isinstance(child, (IdentifierNode, VariableNode)) and (
                getattr(child, "name", None)
                in self.hlsl_software_subgroup_resource_names
            ):
                return False
        return True

    def hlsl_software_subgroup_subgroup_id_expression(self, expression):
        """Whether ``expression`` is the entry's never-written subgroup index.

        Integer conversions of the index qualify.
        """
        current = expression
        while True:
            if isinstance(current, CastNode):
                current = current.expression
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
        return isinstance(current, (IdentifierNode, VariableNode)) and (
            getattr(current, "name", None)
            in getattr(self, "hlsl_software_subgroup_guard_names", frozenset())
        )

    def hlsl_software_subgroup_hoisted_target_is_private(
        self, statements, operation_if, plan
    ):
        """Whether a masked reduction's target may move before its guard.

        The declaration then shares the scope of ``statements``, so no other
        statement there may declare or name it.
        """
        operation_body = self.hlsl_software_subgroup_statement_list(
            getattr(operation_if, "then_branch", getattr(operation_if, "if_body", None))
        )
        hoisted_name = operation_body[plan["hoistedTargetDeclarationIndex"]].name
        for statement in statements:
            if statement is operation_if:
                continue
            if hoisted_name in self.hlsl_software_subgroup_expression_identifier_names(
                statement
            ) or any(
                isinstance(child, (VariableNode, ArrayNode))
                and getattr(child, "name", None) == hoisted_name
                for child in self.walk_ast(statement)
            ):
                return False
        return True

    def hlsl_software_subgroup_strided_for_plan(
        self, node, uniform_names, uniform_components, constants
    ):
        """Plan ``for (i = simd_gid; i < n; i += simd_groups)`` in the entry.

        Each subgroup takes every ``simd_groups``-th iteration, so subgroups can
        disagree on the trip count. Every invocation instead runs the rounds of
        the whole workgroup from zero while ``base < n`` and predicates its
        iteration ``i = base + simd_gid`` on the source condition. The body must
        hold one collective, a reduction under a guard that a masked-if plan
        predicates; declarations before that guard must be uniform and
        speculatively safe, as they then run in every invocation.
        """
        invocation_count = self.hlsl_software_subgroup_invocation_count
        if self.software_subgroup_width is None or not invocation_count:
            return None
        initializer = getattr(node, "init", None)
        if not isinstance(initializer, VariableNode) or not initializer.name:
            return None
        loop_name = initializer.name
        loop_type = self.map_type(getattr(initializer, "var_type", None))
        if loop_type not in {"int", "uint"}:
            return None
        names = set(uniform_names) - {loop_name}
        components = {
            name: value
            for name, value in uniform_components.items()
            if name != loop_name
        }
        start = self.hlsl_software_subgroup_strided_start(
            getattr(initializer, "initial_value", None), names, components, loop_name
        )
        if start is None:
            return None
        condition = getattr(node, "condition", None)
        if (
            not isinstance(condition, BinaryOpNode)
            or self.map_operator(
                getattr(condition, "op", getattr(condition, "operator", None))
            )
            != "<"
            or self.expression_name(condition.left) != loop_name
            or loop_name
            in self.hlsl_software_subgroup_expression_identifier_names(condition.right)
            or not self.hlsl_software_subgroup_uniform_expression(
                condition.right, names, components
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
            or self.hlsl_software_subgroup_assignment_target_name(update) != loop_name
        ):
            return None
        step = evaluate_literal_int_expression(
            getattr(update, "value", getattr(update, "right", None)),
            {
                name: value
                for name, value in constants.items()
                if name in names and name != loop_name
            },
        )
        subgroup_count = -(-invocation_count // self.software_subgroup_width)
        if step != subgroup_count:
            return None
        body = getattr(node, "body", None)
        records = self.hlsl_software_subgroup_operation_records(body)
        if len(records) != 1 or (
            records[0][0] not in self.HLSL_SOFTWARE_SUBGROUP_MASKED_IDENTITIES
        ):
            return None
        if any(
            isinstance(child, (BreakNode, ContinueNode, ReturnNode))
            for child in self.walk_ast(body)
        ):
            return None
        statements = self.hlsl_software_subgroup_statement_list(body)
        operation_if_indices = [
            index
            for index, statement in enumerate(statements)
            if isinstance(statement, IfNode)
            and self.hlsl_software_subgroup_operation_records(statement)
        ]
        if len(operation_if_indices) != 1:
            return None
        operation_if_index = operation_if_indices[0]
        masked_if_plan = self.hlsl_software_subgroup_masked_if_plan(
            statements[operation_if_index],
            getattr(self, "hlsl_software_subgroup_guard_names", frozenset()),
            getattr(self, "hlsl_software_subgroup_guard_constant_names", frozenset()),
            allow_hoisted_target=True,
        )
        if masked_if_plan is None:
            return None
        hoisted_target = masked_if_plan.get("hoistedTargetDeclarationIndex")
        if hoisted_target is not None and not (
            self.hlsl_software_subgroup_hoisted_target_is_private(
                statements, statements[operation_if_index], masked_if_plan
            )
        ):
            return None
        control_names = {loop_name} | (
            self.hlsl_software_subgroup_expression_identifier_names(condition.right)
        )
        for child in self.walk_ast(body):
            if isinstance(child, AssignmentNode) and (
                self.hlsl_software_subgroup_assignment_target_name(child)
                in control_names
            ):
                return None
            if isinstance(child, UnaryOpNode) and self.map_operator(
                getattr(child, "op", getattr(child, "operator", None))
            ) in {"++", "--", "&"}:
                if (
                    self.hlsl_software_subgroup_expression_identifier_names(
                        getattr(child, "operand", None)
                    )
                    & control_names
                ):
                    return None
            if (
                isinstance(child, VariableNode)
                and getattr(child, "name", None) in control_names
            ):
                return None
            if (
                isinstance(child, FunctionCallNode)
                and self.function_call_name(child) in self.function_return_types
                and self.hlsl_software_subgroup_expression_identifier_names(child)
                & control_names
            ):
                # A helper could receive the counter or bound by reference.
                return None
        hoisted = []
        for index, statement in enumerate(statements[:operation_if_index]):
            if isinstance(statement, ArrayNode):
                return None
            if not isinstance(statement, VariableNode):
                continue
            initial = getattr(statement, "initial_value", None)
            if not (
                self.hlsl_software_subgroup_uniform_expression(
                    initial, names, components
                )
                and self.hlsl_software_subgroup_speculatively_safe_expression(initial)
            ):
                return None
            hoisted.append(index)
        return {
            "operationIfIndex": operation_if_index,
            "maskedIfPlan": masked_if_plan,
            "hoistedPrefixDeclarationIndices": tuple(hoisted),
            "loopType": loop_type,
            "step": step,
            # The workgroup-uniform part of the start, or None, and the
            # subgroup index the start adds to it.
            "start": start,
        }

    def generate_hlsl_software_subgroup_strided_for(self, node, indent, plan):
        """Emit a subgroup-strided loop as uniform rounds of the workgroup."""
        initializer = node.init
        declaration = self.generate_for_initializer(initializer)
        target, separator, _initial_value = declaration.partition(" = ")
        if not separator:
            raise self.hlsl_software_subgroup_error(
                "DirectX subgroup-strided loops require an initialized counter",
                workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                reason="potentially-divergent-control-flow",
                source_location=getattr(node, "source_location", None),
            )
        loop_type = plan["loopType"]
        used_names = set(self.local_variable_types)
        used_names.update(self.global_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        base_name = self.hlsl_unique_local_identifier(
            "crossglSoftwareSubgroupBase", used_names
        )
        used_names.add(base_name)
        active_name = self.hlsl_unique_local_identifier(
            "crossglSoftwareSubgroupLoopActive", used_names
        )
        self.local_variable_types[base_name] = loop_type
        self.local_variable_types[active_name] = "bool"
        bound = self.generate_expression_with_expected(
            node.condition.right, getattr(initializer, "var_type", None)
        )
        header, counter = self.hlsl_software_subgroup_strided_header(
            node, plan, base_name, bound
        )
        indent_str = "    " * indent
        body_indent = "    " * (indent + 1)
        code = f"{indent_str}{header}\n" f"{body_indent}{target} = {counter};\n"
        statements = self.hlsl_software_subgroup_statement_list(node.body)
        operation_if_index = plan["operationIfIndex"]
        prefix = list(statements[:operation_if_index])
        suffix = list(statements[operation_if_index + 1 :])
        hoisted = set(plan["hoistedPrefixDeclarationIndices"])
        for index in sorted(hoisted):
            code += self.generate_statement(prefix[index], indent + 1)
        guarded_prefix = [
            statement for index, statement in enumerate(prefix) if index not in hoisted
        ]
        condition = self.hlsl_condition_expression(node.condition)
        code += f"{body_indent}bool {active_name} = {condition};\n"
        if guarded_prefix:
            code += f"{body_indent}if ({active_name}) {{\n"
            code += self.generate_scoped_statement_body(guarded_prefix, indent + 2)
            code += f"{body_indent}}}\n"
        code += self.generate_hlsl_software_subgroup_masked_if(
            statements[operation_if_index],
            indent + 1,
            additional_condition=active_name,
        )
        if suffix:
            code += f"{body_indent}if ({active_name}) {{\n"
            code += self.generate_scoped_statement_body(suffix, indent + 2)
            code += f"{body_indent}}}\n"
        code += f"{indent_str}}}\n"
        return code
