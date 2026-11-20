"""Integer interval facts of a GLSL function, propagated through its control flow for ``GLSLCodeGen.annotate_glsl_control_flow_intervals``."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
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
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..validation import expression_debug_name
from .index_range_contracts import IntegerRange
from .opengl_control_flow_interval_statements import (
    OpenGLControlFlowIntervalStatementMixin,
)
from .opengl_loop_trip_bounds import OpenGLLoopTripBoundMixin
from .scalar_return_bounds import aggregate_call_intervals


class GLSLControlFlowIntervalAnalysis(
    OpenGLControlFlowIntervalStatementMixin, OpenGLLoopTripBoundMixin
):
    """Integer interval facts of one GLSL function, propagated through its statements and loops."""

    def __init__(
        self,
        *,
        function,
        generator,
        constants,
        lexical,
        replay_depth,
        replay_budget,
        replay_intervals,
        pointer_read_intervals,
        reached_statements,
        unreachable_tails,
        escaped_names,
        local_storage_pointer_names,
        bounded_zero_trip,
        possibly_entered_loops,
    ):
        self.function = function
        self.generator = generator
        self.constants = constants
        self.lexical = lexical
        self.replay_depth = replay_depth
        self.replay_budget = replay_budget
        self.replay_intervals = replay_intervals
        self.pointer_read_intervals = pointer_read_intervals
        self.reached_statements = reached_statements
        self.unreachable_tails = unreachable_tails
        self.escaped_names = escaped_names
        self.local_storage_pointer_names = local_storage_pointer_names
        self.bounded_zero_trip = bounded_zero_trip
        self.possibly_entered_loops = possibly_entered_loops

    def pointer_array_extent(self, expression):
        if not isinstance(expression, IdentifierNode):
            return None
        declaration = self.lexical["declarations"].get(
            self.lexical["reference_bindings"].get(id(expression))
        )
        if declaration is not None:
            contract = self.generator.glsl_storage_pointer_array_contract(
                declaration["node"]
            )
        else:
            contract = getattr(self.function, "_glsl_storage_pointer_aliases", {}).get(
                expression.name
            )
            if contract is not None and contract.get("kind") != "storage-pointer-array":
                contract = None
        return contract.get("extent") if contract is not None else None

    def lexical_source_types(self, expression):
        source_types = {}
        for node in self.generator.walk_ast(expression):
            if not isinstance(node, IdentifierNode):
                continue
            binding = self.lexical["declarations"].get(
                self.lexical["reference_bindings"].get(id(node))
            )
            if binding is not None:
                if binding["source_type"] is None:
                    return None
                source_types[node.name] = binding["source_type"]
        return source_types

    def checked_interval(self, expression, intervals, destination_type=None):
        source_types = self.lexical_source_types(expression)
        if source_types is None:
            return None
        with self.generator.glsl_lexical_source_type_scope(source_types):
            interval = self.generator.glsl_checked_private_pointer_interval(
                expression, intervals, self.constants
            )
            if interval is None and destination_type is not None:
                # Unsigned wrap is exact modulo the destination width.
                interval = self.generator.glsl_congruent_storage_interval(
                    expression, intervals, self.constants, destination_type
                )
            return interval

    def checked_components(self, expression, vector_type, intervals):
        source_types = self.lexical_source_types(expression)
        if source_types is None:
            return None
        with self.generator.glsl_lexical_source_type_scope(source_types):
            components = self.generator.glsl_static_integer_vector_component_intervals(
                expression, vector_type, intervals, self.constants
            )
            if components is None:
                return None
            return tuple(
                self.generator.glsl_integer_storage_interval(
                    bounds, vector_type, component=True
                )
                for bounds in components
            )

    def expression_source_type(self, expression):
        source_types = self.lexical_source_types(expression)
        if source_types is None:
            return None
        with self.generator.glsl_lexical_source_type_scope(source_types):
            return self.generator.glsl_source_expression_type(expression)

    def aggregate_updates(self, expression, intervals, *, additive=False):
        source_types = self.lexical_source_types(expression)
        if source_types is None:
            return None
        has_aggregate = False
        for argument in expression.arguments or ():
            source_type = self.expression_source_type(argument)
            if not self.generator.glsl_struct_member_types(
                self.generator.type_name_string(source_type)
            ):
                continue
            has_aggregate = True
            if (
                not isinstance(argument, IdentifierNode)
                or self.stable_lexical_type(argument) is None
                or argument.name in self.escaped_names
            ):
                return None
        if not has_aggregate:
            return None
        with self.generator.glsl_lexical_source_type_scope(source_types):
            return aggregate_call_intervals(
                self.generator, expression, intervals, self.constants, additive=additive
            )

    def stable_lexical_type(self, expression):
        owner = (
            expression.object
            if isinstance(expression, MemberAccessNode)
            else expression
        )
        if not isinstance(owner, IdentifierNode):
            return None
        binding = self.lexical["declarations"].get(
            self.lexical["reference_bindings"].get(id(owner))
        )
        if binding is None or binding["source_type"] is None:
            return None
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(binding["node"], "qualifiers", ()) or ()
        }
        source_type = binding["source_type"]
        metadata = getattr(self.generator, "glsl_metadata_entry_bounds", {}).get(
            id(self.function), {}
        )
        if (
            owner.name in metadata
            and any(
                binding["node"] is parameter for parameter in self.function.parameters
            )
            and getattr(source_type, "referenced_type", None) is not None
        ):
            # These source-owned immutable scalar references have already
            # passed metadata validation; their loaded values are integers.
            source_type = source_type.referenced_type
        type_name = self.generator.type_name_string(source_type) or ""
        if (
            "&" in type_name
            or "volatile" in type_name.replace("*", " ").split()
            or qualifiers.intersection({"volatile", "out", "inout"})
        ):
            return None
        if isinstance(expression, MemberAccessNode):
            field_type = self.generator.glsl_struct_member_types(type_name).get(
                str(expression.member)
            )
            if field_type is not None:
                return field_type
            component = self.generator.glsl_index_component_index(expression.member)
            info = self.generator.glsl_value_type_info(source_type)
            if (
                info is None
                or component is None
                or not 2 <= info["width"] <= 4
                or component >= info["width"]
            ):
                return None
            return self.expression_source_type(expression)
        return source_type

    def lexical_integer_domain(self, expression):
        source_type = self.stable_lexical_type(expression)
        if source_type is None or "*" in (
            self.generator.type_name_string(source_type) or ""
        ):
            return None
        owner = (
            expression.object
            if isinstance(expression, MemberAccessNode)
            else expression
        )
        with self.generator.glsl_lexical_source_type_scope(
            {owner.name: self.stable_lexical_type(owner)}
        ):
            return self.generator.glsl_integer_type_domain(expression, allow_wide=True)

    def unit_increment(self, operator):
        operator = self.generator.map_operator(operator)
        if operator in {"++", "PRE_INCREMENT", "POST_INCREMENT"}:
            return 1
        if operator in {"--", "PRE_DECREMENT", "POST_DECREMENT"}:
            return -1
        return None

    def checked_integer_interval(self, expression, intervals):
        source_types = {}
        for child in self.generator.walk_ast(expression):
            if not isinstance(child, IdentifierNode):
                continue
            binding = self.lexical["declarations"].get(
                self.lexical["reference_bindings"].get(id(child))
            )
            if binding is not None:
                source_type = self.stable_lexical_type(child)
                if source_type is None or "*" in (
                    self.generator.type_name_string(source_type) or ""
                ):
                    return None
                source_types[child.name] = source_type
        with self.generator.glsl_lexical_source_type_scope(source_types):
            bounds = self.generator.glsl_checked_private_pointer_interval(
                expression, intervals, self.constants, require_arithmetic_types=True
            )
            domain = self.generator.glsl_integer_type_domain(
                expression, allow_wide=True
            )
        if bounds is None or domain is None:
            return None
        return (
            bounds
            if domain.minimum <= bounds[0] <= bounds[1] <= domain.maximum
            else None
        )

    def refine_guard(self, condition, intervals, expected):
        source_types = {}
        expressions = []
        cast_names = {
            id(node.function)
            for node in self.generator.walk_ast(condition)
            if isinstance(node, FunctionCallNode)
            and self.generator.glsl_integer_scalar_cast_call(node) is not None
        }
        for node in self.generator.walk_ast(condition):
            if isinstance(node, MemberAccessNode):
                expressions.append(node)
                continue
            if not isinstance(node, IdentifierNode) or id(node) in cast_names:
                continue
            binding = self.lexical["declarations"].get(
                self.lexical["reference_bindings"].get(id(node))
            )
            if binding is None or binding["source_type"] is None:
                return dict(intervals)
            source_types[node.name] = binding["source_type"]
            expressions.append(node)
        with self.generator.glsl_lexical_source_type_scope(source_types):
            guarded = dict(intervals)
            seeded_domains = {}
            for node in expressions:
                owner = node
                while isinstance(owner, MemberAccessNode):
                    if (
                        self.generator.glsl_declared_struct_member_interval_key(owner)
                        is None
                    ):
                        break
                    owner = owner.object
                if not isinstance(owner, IdentifierNode):
                    continue
                if owner.name in self.escaped_names:
                    continue
                if isinstance(node, MemberAccessNode):
                    binding = self.lexical["declarations"][
                        self.lexical["reference_bindings"][id(owner)]
                    ]
                    owner_type = (
                        self.generator.type_name_string(binding["source_type"]) or ""
                    )
                    if (
                        "*" in owner_type
                        or "&" in owner_type
                        or set(
                            self.generator.glsl_parameter_qualifiers(binding["node"])
                        ).intersection({"out", "inout"})
                    ):
                        continue
                    name = self.generator.glsl_declared_struct_member_interval_key(node)
                else:
                    name = node.name
                domain = self.generator.glsl_integer_type_domain(node, allow_wide=True)
                if domain is not None:
                    current = guarded.get(name)
                    if current is None or not (
                        domain.minimum <= current[0] <= current[1] <= domain.maximum
                    ):
                        guarded[name] = (domain.minimum, domain.maximum)
                        seeded_domains[name] = guarded[name]
            refined = self.generator.glsl_private_pointer_refined_condition_intervals(
                condition,
                guarded,
                self.constants,
                expected,
                excluded_names=self.escaped_names,
                check_source_types=True,
            )
            if refined is not None:
                for name, domain in seeded_domains.items():
                    if refined.get(name) == domain:
                        refined.pop(name, None)
            return refined

    def merge_intervals(self, left, right):
        return {
            name: (
                min(left[name][0], right[name][0]),
                max(left[name][1], right[name][1]),
            )
            for name in left.keys() & right.keys()
        }

    def annotate_expression(self, expression, intervals):
        if (
            isinstance(expression, ArrayAccessNode)
            and self.pointer_array_extent(expression.array) is not None
        ):
            # A recurrence can read a different initialized slot on every
            # visit. Join the values read, not the intersection of slots.
            current = self.checked_interval(expression, intervals)
            if id(expression) in self.pointer_read_intervals:
                previous = self.pointer_read_intervals[id(expression)]
                current = (
                    (min(previous[0], current[0]), max(previous[1], current[1]))
                    if previous is not None and current is not None
                    else None
                )
            self.pointer_read_intervals[id(expression)] = current
        if hasattr(expression, "__dict__"):
            bounds = dict(intervals)
            if self.replay_depth:
                previous = self.replay_intervals.get(id(expression))
                if previous is not None:
                    bounds = self.merge_intervals(previous, bounds)
                self.replay_intervals[id(expression)] = bounds
            expression._glsl_control_flow_intervals = bounds

    def target_interval_keys(self, target):
        if (
            isinstance(target, UnaryOpNode)
            and self.generator.map_operator(target.op) == "&"
        ):
            return self.target_interval_keys(target.operand)
        field_key = self.generator.glsl_declared_struct_member_interval_key(target)
        if field_key is not None:
            return {field_key}
        if isinstance(target, ArrayAccessNode):
            component_index = self.generator.literal_int_value(
                target.index, self.constants
            )
            array_name = self.generator.expression_name(target.array)
            if self.pointer_array_extent(target.array) is not None:
                array_key = self.generator.glsl_pointer_array_interval_key(
                    array_name, component_index
                )
                return {array_key} if array_key is not None else {array_name}
            if (
                isinstance(component_index, int)
                and not isinstance(component_index, bool)
                and component_index in range(4)
            ):
                key = self.generator.glsl_index_component_interval_key(
                    target.array,
                    component_index,
                )
                return {key} if key is not None else set()
            return {
                key
                for component_index in range(4)
                if (
                    key := self.generator.glsl_index_component_interval_key(
                        target.array,
                        component_index,
                    )
                )
                is not None
            }
        if isinstance(target, MemberAccessNode):
            component_indices = tuple(
                self.generator.glsl_index_component_index(component)
                for component in str(target.member)
            )
            if component_indices and all(
                component_index is not None for component_index in component_indices
            ):
                return {
                    key
                    for component_index in component_indices
                    if (
                        key := self.generator.glsl_index_component_interval_key(
                            target.object,
                            component_index,
                        )
                    )
                    is not None
                }
            object_name = expression_debug_name(target.object)
            return {f"{object_name}.{target.member}"} if object_name else set()
        name = self.generator.expression_name(target)
        if not name:
            return set()
        keys = {name}
        keys.update(
            self.generator.glsl_index_component_interval_key(name, component_index)
            for component_index in range(4)
        )
        keys.discard(None)
        return keys

    def clear_interval_targets(self, active_intervals, target_keys):
        if not target_keys:
            return
        target_keys = set(target_keys)
        for name, binding in getattr(
            self.function, "_glsl_storage_pointer_aliases", {}
        ).items():
            if name in target_keys and isinstance(binding.get("offset"), str):
                target_keys.add(binding["offset"])
        for interval_key in list(active_intervals):
            if self.generator.glsl_interval_key_sets_overlap(
                {interval_key},
                target_keys,
            ):
                active_intervals.pop(interval_key, None)

    def call_argument_mutation_keys(
        self,
        function_name,
        arguments,
        argument_index,
        argument,
    ):
        candidates = list(
            self.generator.glsl_function_overloads_by_name.get(function_name, ())
        )
        fallback = self.generator.function_definitions.get(function_name)
        if fallback is not None and fallback not in candidates:
            candidates.append(fallback)
        argument_name = expression_debug_name(argument)
        result = set()
        for candidate in candidates:
            parameters = list(
                getattr(
                    candidate,
                    "parameters",
                    getattr(candidate, "params", []),
                )
                or []
            )
            if argument_index >= len(parameters):
                continue
            parameter = parameters[argument_index]
            if (
                self.generator.glsl_storage_pointer_array_contract(parameter)
                is not None
            ):
                if getattr(candidate, "body", None) is None or (
                    self.generator.glsl_storage_pointer_array_parameter_is_mutated(
                        candidate, getattr(parameter, "name", None)
                    )
                ):
                    result.update(self.target_interval_keys(argument))
                continue
            if not set(
                self.generator.glsl_parameter_qualifiers(parameter)
            ).intersection({"out", "inout"}):
                continue
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name or getattr(candidate, "body", None) is None:
                return self.target_interval_keys(argument)
            mutations = self.generator.glsl_function_mutated_interval_keys(candidate)
            relevant = {
                key
                for key in mutations
                if key == parameter_name or key.startswith(f"{parameter_name}.")
            }
            if not argument_name and relevant:
                return self.target_interval_keys(argument)
            for key in relevant:
                result.add(f"{argument_name}{key[len(parameter_name) :]}")
        return result

    def mutation_interval_keys(self, value, declaration=None):
        mutated = set()

        def relevant(target):
            if declaration is None:
                return True
            name = self.lexical["declarations"][declaration]["node"].name
            return any(
                isinstance(child, IdentifierNode)
                and child.name == name
                and self.lexical["reference_bindings"].get(id(child))
                in {None, declaration}
                for child in self.generator.walk_ast(target)
            )

        def collect(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, dict):
                for child in node.values():
                    collect(child)
                return
            if isinstance(node, (list, tuple, set)):
                for child in node:
                    collect(child)
                return
            if isinstance(node, AssignmentNode):
                target = getattr(
                    node,
                    "target",
                    getattr(node, "left", None),
                )
                if relevant(target):
                    mutated.update(self.target_interval_keys(target))
            elif isinstance(node, UnaryOpNode) and (
                self.generator.map_operator(node.op) in {"++", "--"}
                or node.op in {
                    "PRE_INCREMENT",
                    "PRE_DECREMENT",
                    "POST_INCREMENT",
                    "POST_DECREMENT",
                }
            ):
                if relevant(node.operand):
                    mutated.update(self.target_interval_keys(node.operand))
            elif isinstance(node, FunctionCallNode):
                function_name = self.generator.function_call_name(node)
                candidates = list(
                    self.generator.glsl_function_overloads_by_name.get(
                        function_name, ()
                    )
                )
                fallback = self.generator.function_definitions.get(function_name)
                if fallback is not None and fallback not in candidates:
                    candidates.append(fallback)
                for argument_index, argument in enumerate(node.arguments or []):
                    if not relevant(argument):
                        continue
                    mutated.update(
                        self.call_argument_mutation_keys(
                            function_name,
                            list(node.arguments or []),
                            argument_index,
                            argument,
                        )
                    )
            if hasattr(node, "child_nodes"):
                for child in node.child_nodes():
                    collect(child)
                return
            if hasattr(node, "__dict__"):
                for field, child in vars(node).items():
                    if field not in {
                        "annotations",
                        "parent",
                        "source_location",
                        "_glsl_control_flow_intervals",
                    }:
                        collect(child)

        collect(value)
        return mutated

    def visit_sequence(self, value, intervals):
        if isinstance(value, BlockNode):
            return self.visit(value, intervals)
        statements = getattr(value, "statements", value)
        if not isinstance(statements, (list, tuple)):
            statements = [statements]
        for index, statement in enumerate(statements):
            if self.visit(statement, intervals) is False:
                self.unreachable_tails.update(
                    id(item) for item in statements[index + 1 :]
                )
                return False
        return True

    def refuted_branch(self, body):
        # The checked intervals refute this branch's guard, so its
        # statements are unreachable unless another replay reaches them.
        statements = getattr(body, "statements", body)
        if statements is None:
            return
        if not isinstance(statements, (list, tuple)):
            statements = [statements]
        self.unreachable_tails.update(id(item) for item in statements)

    def bounded_loop_bounds(self, node, loop_intervals, mutated):
        init, condition, update = node.init, node.condition, node.update
        if not isinstance(init, VariableNode):
            return None

        def counter_operand(expression):
            if isinstance(expression, IdentifierNode) and expression.name == init.name:
                return expression
            if (
                isinstance(expression, BinaryOpNode)
                and self.generator.map_operator(expression.op) == "+"
            ):
                for counter, offset in (
                    (expression.left, expression.right),
                    (expression.right, expression.left),
                ):
                    if (
                        isinstance(counter, IdentifierNode)
                        and counter.name == init.name
                        and init.name
                        not in self.generator.glsl_interval_dependency_names(offset)
                    ):
                        return counter
            return None

        conditions = [condition]
        for candidate in conditions:
            if (
                isinstance(candidate, BinaryOpNode)
                and self.generator.map_operator(candidate.op) == "&&"
            ):
                conditions.extend((candidate.left, candidate.right))
        condition = next(
            (
                candidate
                for candidate in conditions
                if isinstance(candidate, BinaryOpNode)
                and isinstance(candidate.left, IdentifierNode)
                and candidate.left.name == init.name
                and self.generator.map_operator(candidate.op) in {"<", "<="}
            ),
            None,
        )
        if condition is None:
            condition = next(
                (
                    candidate
                    for candidate in conditions
                    if isinstance(candidate, BinaryOpNode)
                    and self.generator.map_operator(candidate.op) in {"<", "<="}
                    and counter_operand(candidate.left) is not None
                ),
                None,
            )
        counter = counter_operand(condition.left) if condition is not None else None
        increment = None
        if (
            isinstance(update, UnaryOpNode)
            and isinstance(update.operand, IdentifierNode)
            and update.operand.name == init.name
            and self.unit_increment(update.op) == 1
        ):
            increment = LiteralNode(1, PrimitiveType("int"))
        elif (
            isinstance(update, AssignmentNode)
            and isinstance(update.target, IdentifierNode)
            and update.target.name == init.name
            and self.generator.map_operator(update.operator) == "+="
        ):
            increment = update.value
        if (
            not isinstance(init, VariableNode)
            or getattr(init, "qualifiers", None)
            or not isinstance(condition, BinaryOpNode)
            or counter is None
            or self.generator.map_operator(condition.op) not in {"<", "<="}
            or increment is None
            or not self.generator.glsl_side_effect_free_expression(increment)
            or self.mutation_interval_keys(increment)
            or init.name in self.escaped_names
            or not self.generator.glsl_side_effect_free_expression(init.initial_value)
            or self.mutation_interval_keys(init.initial_value)
        ):
            return None
        domain = self.lexical_integer_domain(counter)
        start = loop_intervals.get(init.name)
        dependencies = self.generator.glsl_interval_dependency_names(condition)
        dependencies.discard(init.name)
        dependencies.update(self.generator.glsl_interval_dependency_names(increment))
        for child in self.generator.walk_ast(node.condition):
            if isinstance(child, AssignmentNode) or (
                isinstance(child, UnaryOpNode)
                and self.unit_increment(child.op) is not None
            ):
                return None
            if isinstance(child, FunctionCallNode) and (
                self.generator.glsl_private_pointer_expression_has_side_effects(child)
            ):
                # A return summary proves effects only for the analyzed
                # arguments. They must remain invariant on later calls.
                if (
                    self.generator.glsl_interval_key_sets_overlap(
                        self.generator.glsl_interval_dependency_names(child),
                        mutated | self.escaped_names | {init.name},
                    )
                    or self.checked_integer_interval(child, loop_intervals) is None
                ):
                    return None
        if (
            domain is None
            or start is None
            or not 0 <= start[0] <= start[1] <= domain.maximum
            or self.checked_integer_interval(init.initial_value, loop_intervals)
            != start
            or self.generator.glsl_interval_key_sets_overlap(
                dependencies, mutated | self.escaped_names | {init.name}
            )
            or init.name in self.mutation_interval_keys(node.body, id(init))
            or init.name in self.generator.glsl_interval_dependency_names(
                condition.right
            )
            or self.mutation_interval_keys(node.condition)
        ):
            return None
        return self.bounded_loop_trip_bounds(
            node,
            loop_intervals,
            condition,
            counter,
            dependencies,
            domain,
            increment,
            init,
            start,
        )

    def additive_loop_intervals(self, node, incoming, mutated, count):
        """Bound one unconditional, invariant additive update per iteration."""
        statements = list(getattr(node.body, "statements", node.body) or [])
        pointer_names = (
            set(
                getattr(self.generator, "glsl_entry_pointer_parameters", {}).get(
                    id(self.function), ()
                )
            )
            | set(getattr(self.function, "_glsl_storage_pointer_aliases", {}))
            | self.local_storage_pointer_names
        )
        declared = {
            child.name
            for child in self.generator.walk_ast(node.body)
            if isinstance(child, (VariableNode, ArrayNode))
        }
        entries, exits = {}, {}
        for index, statement in enumerate(statements):
            update = (
                statement.expression
                if isinstance(statement, ExpressionStatementNode)
                else statement
            )
            call_step = None
            if isinstance(update, FunctionCallNode):
                deltas = self.aggregate_updates(update, incoming, additive=True)
                if not deltas or len(deltas) != 1:
                    continue
                name, call_step = next(iter(deltas.items()))
                owner_name, _, member = name.rpartition(".")
                owners = [
                    argument
                    for argument in update.arguments
                    if isinstance(argument, IdentifierNode)
                    and argument.name == owner_name
                ]
                if len(owners) != 1:
                    continue
                target = MemberAccessNode(owners[0], member)
                assigned = [
                    argument
                    for argument in update.arguments
                    if argument is not owners[0]
                ]
                operator = "+="
            elif isinstance(update, AssignmentNode):
                target = getattr(update, "target", getattr(update, "left", None))
                assigned = getattr(update, "value", getattr(update, "right", None))
                operator = self.generator.map_operator(update.operator)
            else:
                continue
            if not isinstance(
                target, (IdentifierNode, MemberAccessNode)
            ) or operator not in {
                "+=",
                "-=",
            }:
                continue
            if isinstance(target, MemberAccessNode):
                if self.stable_lexical_type(target) is None:
                    continue
                with self.generator.glsl_lexical_source_type_scope(
                    {target.object.name: self.stable_lexical_type(target.object)}
                ):
                    name = self.generator.glsl_declared_struct_member_interval_key(
                        target
                    ) or self.generator.glsl_index_component_interval_key(
                        target.object,
                        self.generator.glsl_index_component_index(target.member),
                    )
            else:
                name = target.name
            initial = incoming.get(name)
            dependencies = self.generator.glsl_interval_dependency_names(assigned)
            other_statements = statements[:index] + statements[index + 1 :]
            if (
                initial is None
                or self.generator.glsl_interval_key_sets_overlap(
                    {name}, self.escaped_names | declared
                )
                or self.generator.glsl_interval_key_sets_overlap(
                    dependencies, mutated | self.escaped_names | declared
                )
                or self.generator.glsl_interval_key_sets_overlap(
                    {name}, self.mutation_interval_keys(other_statements)
                )
                or any(
                    self.generator.glsl_private_pointer_expression_has_side_effects(
                        item
                    )
                    for item in (assigned if call_step is not None else [assigned])
                )
            ):
                continue
            domain = self.lexical_integer_domain(target)
            source_type = self.stable_lexical_type(target)
            if (
                domain is None
                and name in pointer_names
                and source_type is not None
                and (
                    "*" in self.generator.type_name_string(source_type)
                    or self.generator.is_structured_buffer_type(source_type)
                )
            ):
                domain = IntegerRange(0, (1 << 31) - 1)
            step = (
                call_step
                if call_step is not None
                else self.checked_integer_interval(assigned, incoming)
            )
            if domain is None or step is None:
                continue
            if operator == "-=":
                step = (-step[1], -step[0])
            exit_interval = (
                initial[0] + min(0, count * step[0]),
                initial[1] + max(0, count * step[1]),
            )
            if (
                not domain.minimum
                <= exit_interval[0]
                <= exit_interval[1]
                <= domain.maximum
            ):
                continue
            entry_interval = (
                initial[0] + min(0, (count - 1) * step[0]),
                initial[1] + max(0, (count - 1) * step[1]),
            )
            if call_step is not None:
                # Recheck promoted arithmetic over every possible entry,
                # not just the zero-based probe used to obtain the step.
                after = self.aggregate_updates(
                    update, {**incoming, name: entry_interval}
                )
                bounds = after.get(name) if after is not None else None
                if (
                    bounds is None
                    or not exit_interval[0]
                    <= bounds[0]
                    <= bounds[1]
                    <= exit_interval[1]
                ):
                    continue
            entries[name] = entry_interval
            exits[name] = exit_interval
        return entries, exits

    def visit(self, value, intervals):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for child in value.values():
                self.visit(child, intervals)
            return
        if isinstance(value, (list, tuple, set)):
            return self.visit_sequence(list(value), intervals)

        self.reached_statements.add(id(value))
        self.annotate_expression(value, intervals)
        if isinstance(value, ReturnNode):
            self.visit(value.value, intervals)
            return False
        if isinstance(value, (VariableNode, ArrayNode)):
            return self.visit_declaration(value, intervals)
        if isinstance(value, AssignmentNode):
            return self.visit_assignment(value, intervals)
        if isinstance(value, TernaryOpNode):
            self.visit(value.condition, intervals)
            true_intervals = dict(intervals)
            false_intervals = dict(intervals)
            self.visit(value.true_expr, true_intervals)
            self.visit(value.false_expr, false_intervals)
            intervals.clear()
            intervals.update(self.merge_intervals(true_intervals, false_intervals))
            return
        if isinstance(value, BinaryOpNode) and self.generator.map_operator(
            value.op
        ) in {
            "&&",
            "||",
        }:
            self.visit(value.left, intervals)
            left_intervals = dict(intervals)
            right_intervals = dict(left_intervals)
            self.visit(value.right, right_intervals)
            intervals.clear()
            intervals.update(self.merge_intervals(left_intervals, right_intervals))
            return
        if isinstance(value, FunctionCallNode):
            return self.visit_call(value, intervals)
        if (
            isinstance(value, UnaryOpNode)
            and self.generator.map_operator(value.op) == "&"
        ):
            self.visit(value.operand, intervals)
            self.clear_interval_targets(
                intervals,
                self.target_interval_keys(value.operand),
            )
            return
        if isinstance(value, UnaryOpNode) and (
            self.generator.map_operator(value.op) in {"++", "--"}
            or value.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }
        ):
            return self.visit_increment(value, intervals)
        if isinstance(value, BlockNode):
            return self.visit_block(value, intervals)
        if isinstance(value, IfNode):
            return self.visit_if(value, intervals)
        if isinstance(value, (SwitchNode, MatchNode)):
            return self.visit_switch(value, intervals)
        if isinstance(value, ForNode):
            return self.visit_for(value, intervals)
        if isinstance(value, ForInNode):
            return self.visit_for_in(value, intervals)
        if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
            return self.visit_while(value, intervals)
        if hasattr(value, "child_nodes"):
            for child in value.child_nodes():
                self.visit(child, intervals)
            return
        if hasattr(value, "__dict__"):
            for field, child in vars(value).items():
                if field not in {
                    "annotations",
                    "parent",
                    "source_location",
                    "_glsl_control_flow_intervals",
                }:
                    self.visit(child, intervals)


def annotate_glsl_control_flow_intervals(generator, function):
    """Record on ``function``'s nodes the integer intervals that hold where they run."""
    constants = generator.initial_literal_int_constants(function)
    lexical = generator.glsl_scalar_array_view_lexical_bindings(function)
    replay_depth = 0
    replay_budget = 4096
    replay_intervals = {}
    pointer_read_intervals = {}
    reached_statements = set()
    unreachable_tails = set()

    for node in generator.walk_ast(getattr(function, "body", [])):
        if isinstance(node, ArrayAccessNode):
            node._glsl_storage_pointer_read_interval = None

    escaped_names = set()
    local_storage_pointer_names = set()
    for node in generator.walk_ast(getattr(function, "body", [])):
        if (
            isinstance(node, VariableNode)
            and generator.glsl_storage_pointer_element_type(node) is not None
        ):
            local_storage_pointer_names.add(node.name)
        if isinstance(node, UnaryOpNode) and generator.map_operator(node.op) == "&":
            escaped_names.update(generator.glsl_interval_dependency_names(node.operand))
        elif isinstance(node, VariableNode) and (
            generator.type_name_string(getattr(node, "var_type", None)) or ""
        ).endswith("&"):
            escaped_names.update(
                generator.glsl_interval_dependency_names(node.initial_value)
            )

    bounded_zero_trip = set()
    possibly_entered_loops = set()

    analysis = GLSLControlFlowIntervalAnalysis(
        function=function,
        generator=generator,
        constants=constants,
        lexical=lexical,
        replay_depth=replay_depth,
        replay_budget=replay_budget,
        replay_intervals=replay_intervals,
        pointer_read_intervals=pointer_read_intervals,
        reached_statements=reached_statements,
        unreachable_tails=unreachable_tails,
        escaped_names=escaped_names,
        local_storage_pointer_names=local_storage_pointer_names,
        bounded_zero_trip=bounded_zero_trip,
        possibly_entered_loops=possibly_entered_loops,
    )
    visit_sequence = analysis.visit_sequence
    visit_sequence(
        getattr(function, "body", []),
        generator.glsl_workgroup_proof_initial_intervals(function),
    )
    for node in generator.walk_ast(getattr(function, "body", [])):
        if isinstance(node, ArrayAccessNode):
            node._glsl_storage_pointer_read_interval = pointer_read_intervals.get(
                id(node)
            )
        # Replayed nodes are unreachable only if no iteration reaches them.
        if id(node) in unreachable_tails or hasattr(node, "_glsl_checked_unreachable"):
            node._glsl_checked_unreachable = (
                id(node) in unreachable_tails and id(node) not in reached_statements
            )
        if isinstance(node, ForNode):
            node._glsl_checked_zero_trip = (
                id(node) in bounded_zero_trip and id(node) not in possibly_entered_loops
            ) or generator.glsl_control_flow_zero_trip_for(node, lexical)
        elif isinstance(node, IfNode):
            node._glsl_checked_static_if = generator.glsl_control_flow_static_if_body(
                node, lexical
            )
