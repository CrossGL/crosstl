"""Private pointer bindings in the GLSL generator: loop intervals, static and view bindings, unsized arrays and fixed resource parameter sizes."""

from types import SimpleNamespace

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
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
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import OpenGLRefusal
from .array_utils import split_array_type_suffix
from .enum_utils import sanitize_type_name
from .generic_function_utils import generic_function_call_name
from .opengl_values import OpenGLPrivatePointerParameterError
from .pointer_reinterpret import (
    PointerReinterpretationError,
    scalar_storage_layout,
)
from .private_member_views import (
    glsl_member_view_root,
    is_single_object_type,
    member_view_chain,
)
from .stage_utils import collect_stage_local_variables


class OpenGLPrivatePointerBindingMixin:
    """Private pointer binding members of ``GLSLCodeGen``."""

    def glsl_private_pointer_loop_interval(self, node, intervals, constants):
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
        bound = self.glsl_private_pointer_interval(
            bound_expression, intervals, constants
        )
        update = getattr(node, "update", None)
        step = None
        if isinstance(update, UnaryOpNode):
            update_name = self.expression_name(update.operand)
            update_operator = self.map_operator(update.op)
            if update_name == loop_name and update_operator in {"++", "--"}:
                step = 1 if update_operator == "++" else -1
        elif isinstance(update, AssignmentNode):
            update_target = getattr(
                update,
                "target",
                getattr(update, "left", None),
            )
            update_value = getattr(
                update,
                "value",
                getattr(update, "right", None),
            )
            update_operator = self.map_operator(
                getattr(update, "operator", getattr(update, "op", "="))
            )
            update_interval = self.glsl_private_pointer_interval(
                update_value,
                intervals,
                constants,
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
        if bound is None or step in {None, 0}:
            return loop_name, None
        if step > 0 and operator in {"<", "<="}:
            maximum = bound[1] - (1 if operator == "<" else 0)
            if maximum < start[0]:
                return loop_name, None
            reachable_maximum = start[0] + ((maximum - start[0]) // step) * step
            return loop_name, (start[0], reachable_maximum)
        if step < 0 and operator in {">", ">="}:
            minimum = bound[0] + (1 if operator == ">" else 0)
            if minimum > start[0]:
                return loop_name, None
            stride = -step
            reachable_minimum = start[0] - ((start[0] - minimum) // stride) * stride
            return loop_name, (reachable_minimum, start[0])
        return loop_name, None

    def glsl_private_pointer_rebase_loop_iteration_values(
        self,
        node,
        intervals,
        constants,
    ):
        """Return a conservative finite iteration sequence for pointer rebases."""

        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None)
        start = intervals.get(loop_name)
        if not loop_name or start is None or start[0] != start[1]:
            return None
        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode) or (
            self.glsl_private_pointer_expression_has_side_effects(condition)
        ):
            return None

        update = getattr(node, "update", None)
        step = None
        if isinstance(update, UnaryOpNode):
            update_name = self.expression_name(update.operand)
            update_operator = self.map_operator(update.op)
            if update_name == loop_name and update_operator in {"++", "--"}:
                step = 1 if update_operator == "++" else -1
        elif isinstance(update, AssignmentNode):
            update_target = getattr(
                update,
                "target",
                getattr(update, "left", None),
            )
            update_value = getattr(
                update,
                "value",
                getattr(update, "right", None),
            )
            update_operator = self.map_operator(
                getattr(update, "operator", getattr(update, "op", "="))
            )
            update_interval = self.glsl_private_pointer_interval(
                update_value,
                intervals,
                constants,
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
        body_mutations = self.glsl_function_mutated_interval_names(
            SimpleNamespace(body=body)
        )
        if loop_name in body_mutations:
            return None
        condition_dependencies = self.glsl_interval_dependency_names(condition)
        condition_dependencies.discard(loop_name)
        if condition_dependencies & body_mutations:
            return None

        proven_name, loop_interval = self.glsl_private_pointer_loop_interval(
            node,
            intervals,
            constants,
        )
        if proven_name != loop_name or loop_interval is None:
            return None
        first = start[0]
        if step > 0:
            if loop_interval[0] != first or loop_interval[1] < first:
                return None
            iteration_count = ((loop_interval[1] - first) // step) + 1
            if iteration_count > self.GLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS:
                return None
            values = tuple(range(first, loop_interval[1] + 1, step))
        else:
            if loop_interval[1] != first or loop_interval[0] > first:
                return None
            iteration_count = ((first - loop_interval[0]) // (-step)) + 1
            if iteration_count > self.GLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS:
                return None
            values = tuple(range(first, loop_interval[0] - 1, step))
        if not values or (min(values), max(values)) != loop_interval:
            return None
        terminal_value = values[-1] + step
        if any(
            value < -(1 << 31) or value > self.GLSL_PRIVATE_POINTER_BASE_MAX
            for value in (*values, terminal_value)
        ):
            return None
        return values

    @staticmethod
    def glsl_private_pointer_static_binding_interval(binding):
        interval = binding.get("offset_interval")
        exact = binding.get("offset")
        if interval is None and isinstance(exact, int):
            return exact, exact
        if (
            isinstance(interval, tuple)
            and len(interval) == 2
            and all(
                isinstance(bound, int) and not isinstance(bound, bool)
                for bound in interval
            )
            and interval[0] <= interval[1]
        ):
            return interval
        return None

    def glsl_private_pointer_rebased_static_binding(
        self,
        binding,
        delta_interval,
        operator,
    ):
        current_interval = self.glsl_private_pointer_static_binding_interval(binding)
        if current_interval is None or not (
            isinstance(delta_interval, tuple)
            and len(delta_interval) == 2
            and all(
                isinstance(bound, int) and not isinstance(bound, bool)
                for bound in delta_interval
            )
            and delta_interval[0] <= delta_interval[1]
        ):
            return None
        if operator == "-=":
            delta_interval = (-delta_interval[1], -delta_interval[0])
        elif operator != "+=":
            return None
        replacement_interval = (
            current_interval[0] + delta_interval[0],
            current_interval[1] + delta_interval[1],
        )
        if (
            replacement_interval[0] < 0
            or replacement_interval[1] > self.GLSL_PRIVATE_POINTER_BASE_MAX
        ):
            return None
        current_exact = binding.get("offset")
        delta_exact = (
            delta_interval[0] if delta_interval[0] == delta_interval[1] else None
        )
        replacement = dict(binding)
        replacement["offset"] = (
            current_exact + delta_exact
            if isinstance(current_exact, int) and delta_exact is not None
            else None
        )
        replacement["offset_interval"] = replacement_interval
        replacement["side_effecting"] = False
        return replacement

    def glsl_merge_private_pointer_static_bindings(self, bindings):
        bindings = [binding for binding in bindings if binding is not None]
        if not bindings:
            return None
        signature_fields = (
            "root_kind",
            "root",
            "extent",
            "element_type",
            "byte_view",
        )
        signatures = {
            tuple(binding.get(field) for field in signature_fields)
            for binding in bindings
        }
        intervals = [
            self.glsl_private_pointer_static_binding_interval(binding)
            for binding in bindings
        ]
        merged = dict(bindings[0])
        if len(signatures) != 1 or any(interval is None for interval in intervals):
            merged["offset"] = None
            merged["offset_interval"] = None
            merged["side_effecting"] = any(
                binding.get("side_effecting", False) for binding in bindings
            )
            return merged
        exact_offsets = {binding.get("offset") for binding in bindings}
        merged["offset"] = (
            next(iter(exact_offsets))
            if len(exact_offsets) == 1 and isinstance(next(iter(exact_offsets)), int)
            else None
        )
        merged["offset_interval"] = (
            min(interval[0] for interval in intervals),
            max(interval[1] for interval in intervals),
        )
        merged["side_effecting"] = any(
            binding.get("side_effecting", False) for binding in bindings
        )
        return merged

    @staticmethod
    def glsl_private_pointer_discarded_assignment_ids(function):
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
                statement,
                (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode),
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

    def glsl_private_pointer_call_target_key(self, call, source_candidates):
        function_name = self.function_call_name(call)
        arguments = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        normalized_name = self.glsl_private_pointer_materialized_function_name(
            function_name
        )
        for candidate_name in (function_name, normalized_name):
            specialized_name = generic_function_call_name(
                self, candidate_name, arguments
            )
            if specialized_name in self.function_private_pointer_parameters:
                return specialized_name
            candidates = source_candidates.get(candidate_name, ())
            private_candidates = [
                candidate
                for candidate in candidates
                if candidate in self.function_private_pointer_parameters
            ]
            if (
                candidate_name in self.function_private_pointer_parameters
                and candidate_name not in private_candidates
            ):
                private_candidates.append(candidate_name)
            if not private_candidates:
                continue
            resolved = self.resolve_glsl_function_overload(
                candidate_name,
                arguments,
                call_node=call,
            )
            if resolved is not None:
                resolved_key = self.glsl_function_declaration_name(resolved)
                if resolved_key in self.function_private_pointer_parameters:
                    return resolved_key
            if len(private_candidates) == 1:
                return private_candidates[0]
        return None

    def glsl_private_pointer_materialized_function_name(self, name):
        if not isinstance(name, str) or "_u" not in name:
            return name
        decoded = name
        for encoded, character in (
            ("_u3c", "<"),
            ("_u2c", ","),
            ("_u3e", ">"),
            ("_u20", " "),
        ):
            decoded = decoded.replace(encoded, character)
        candidate = sanitize_type_name(decoded)
        return (
            candidate if candidate in self.function_private_pointer_parameters else name
        )

    def glsl_private_pointer_display_name(self, function_name):
        """The source name a diagnostic shows for ``function_name``: its original's, for a specialized clone."""
        source_name = self.function_private_pointer_source_names.get(function_name)
        return function_name if source_name is None else source_name

    def glsl_private_pointer_binding_offset_interval(
        self, call, parameter_name, binding, *, require_exact=False
    ):
        exact_offset = binding.get("offset")
        interval = binding.get("offset_interval")
        if interval is None and isinstance(exact_offset, int):
            interval = (exact_offset, exact_offset)
        if binding.get("side_effecting"):
            reason = "side-effecting-view-offset"
        elif require_exact and not isinstance(exact_offset, int):
            reason = "unprovable-view-offset"
        elif binding.get("correlated_access_contract") is not None:
            return None
        elif not (
            isinstance(interval, tuple)
            and len(interval) == 2
            and all(isinstance(bound, int) for bound in interval)
            and interval[0] <= interval[1]
        ):
            reason = "unprovable-view-offset"
        elif interval[0] < 0:
            reason = "negative-view-offset"
        else:
            return interval
        display_name = self.glsl_private_pointer_display_name(call["callee"])
        backing_name = binding.get("root") or "unknown"
        raise OpenGLPrivatePointerParameterError(
            "OpenGL cannot prove the private pointer view offset for "
            f"'{display_name}.{parameter_name}' into backing array "
            f"'{backing_name}'",
            function_name=display_name,
            parameter_name=parameter_name,
            reason=reason,
            source_location=getattr(call.get("node"), "source_location", None),
        )

    def glsl_private_pointer_backing_error(self, call, parameter_name, binding):
        display_name = self.glsl_private_pointer_display_name(call["callee"])
        backing_name = binding.get("root") or "unknown"
        if binding.get("ambiguous_shape"):
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot select a scalar or array private pointer form for "
                f"'{display_name}.{parameter_name}' from an ambiguous backing "
                "expression",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="ambiguous-scalar-array-backing",
                source_location=getattr(call.get("node"), "source_location", None),
            )
        raise OpenGLPrivatePointerParameterError(
            "OpenGL private pointer view "
            f"'{display_name}.{parameter_name}' has no concrete fixed backing "
            f"array for '{backing_name}'",
            function_name=display_name,
            parameter_name=parameter_name,
            reason="missing-view-backing",
            source_location=getattr(call.get("node"), "source_location", None),
        )

    def glsl_private_pointer_robust_zero_read_eligible(
        self,
        callee,
        parameter_name,
        binding,
        extent,
    ):
        if self.private_pointer_out_of_bounds_read != "zero":
            return False
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            return False
        if (
            parameter_name
            not in self.function_private_pointer_read_only_parameters.get(callee, set())
        ):
            return False
        if binding.get("root_kind") != "local":
            return False
        if binding.get("offset") != 0 or binding.get("offset_interval") != (0, 0):
            return False
        return binding.get("extent") == extent

    def glsl_validate_private_pointer_call_binding(
        self,
        call,
        parameter_name,
        binding,
        extent,
    ):
        required_span = self.function_private_pointer_required_spans[call["callee"]][
            parameter_name
        ]
        if extent == 0:
            offset, _upper_offset = self.glsl_private_pointer_binding_offset_interval(
                call, parameter_name, binding, require_exact=True
            )
            display_name = self.glsl_private_pointer_display_name(call["callee"])
            backing_name = binding.get("root") or "unknown"
            if offset != 0:
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL scalar private pointer view "
                    f"'{display_name}.{parameter_name}' cannot use nonzero "
                    f"offset {offset} from backing scalar '{backing_name}'",
                    function_name=display_name,
                    parameter_name=parameter_name,
                    reason="nonzero-scalar-view-offset",
                    source_location=getattr(call.get("node"), "source_location", None),
                )
            if parameter_name in (
                self.function_private_pointer_scalar_access_violations.get(
                    call["callee"], set()
                )
            ):
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL scalar private pointer view "
                    f"'{display_name}.{parameter_name}' uses a nonzero pointer "
                    "offset or element index",
                    function_name=display_name,
                    parameter_name=parameter_name,
                    reason="nonzero-scalar-index",
                    source_location=getattr(call.get("node"), "source_location", None),
                )
            if parameter_name in (
                self.function_private_pointer_full_span_parameters.get(
                    call["callee"], set()
                )
            ):
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL cannot prove scalar private pointer index zero for "
                    f"'{display_name}.{parameter_name}'",
                    function_name=display_name,
                    parameter_name=parameter_name,
                    reason="unprovable-scalar-index",
                    source_location=getattr(call.get("node"), "source_location", None),
                )
            if required_span != 1:
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL scalar private pointer view "
                    f"'{display_name}.{parameter_name}' requires {required_span} "
                    "elements but a scalar backing permits only index zero",
                    function_name=display_name,
                    parameter_name=parameter_name,
                    reason="nonzero-scalar-index",
                    source_location=getattr(call.get("node"), "source_location", None),
                )
            return
        correlated_contract = binding.get("correlated_access_contract")
        if correlated_contract is not None:
            covered_span = correlated_contract["span"]
            if (
                parameter_name
                not in self.function_private_pointer_full_span_parameters.get(
                    call["callee"], set()
                )
                and required_span <= covered_span
                and required_span <= extent
            ):
                return
            if required_span > extent:
                display_name = self.glsl_private_pointer_display_name(call["callee"])
                backing_name = binding.get("root") or "unknown"
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL private pointer view "
                    f"'{display_name}.{parameter_name}' requires at least "
                    f"{required_span} elements, but backing array "
                    f"'{backing_name}' has extent {extent}",
                    function_name=display_name,
                    parameter_name=parameter_name,
                    reason="view-out-of-bounds",
                    source_location=getattr(call.get("node"), "source_location", None),
                )
            binding = dict(binding)
            binding.pop("correlated_access_contract", None)
        lower_offset, upper_offset = self.glsl_private_pointer_binding_offset_interval(
            call, parameter_name, binding
        )
        if parameter_name in self.function_private_pointer_full_span_parameters.get(
            call["callee"], set()
        ):
            if (lower_offset, upper_offset) != (0, 0):
                display_name = self.glsl_private_pointer_display_name(call["callee"])
                backing_name = binding.get("root") or "unknown"
                offset_label = (
                    f"offset {lower_offset}"
                    if lower_offset == upper_offset
                    else f"offset range {lower_offset} through {upper_offset}"
                )
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL private pointer view "
                    f"'{display_name}.{parameter_name}' requires the complete "
                    f"backing array '{backing_name}' and cannot use {offset_label}",
                    function_name=display_name,
                    parameter_name=parameter_name,
                    reason="full-span-view-offset",
                    source_location=getattr(call.get("node"), "source_location", None),
                )
            required_span = extent
        upper = upper_offset + required_span - 1
        if upper < extent:
            return
        if self.glsl_private_pointer_robust_zero_read_eligible(
            call["callee"],
            parameter_name,
            binding,
            extent,
        ):
            self.glsl_private_pointer_robust_zero_read_variants[
                (call["callee"], parameter_name, extent)
            ] = {
                "backing_extent": extent,
                "required_span": required_span,
            }
            return
        display_name = self.glsl_private_pointer_display_name(call["callee"])
        backing_name = binding.get("root") or "unknown"
        required_range = (
            f"requires at least {upper + 1} elements"
            if lower_offset == 0
            else f"requires elements {lower_offset} through {upper}"
        )
        raise OpenGLPrivatePointerParameterError(
            "OpenGL private pointer view "
            f"'{display_name}.{parameter_name}' {required_range}, but backing "
            f"array '{backing_name}' has extent {extent}",
            function_name=display_name,
            parameter_name=parameter_name,
            reason="view-out-of-bounds",
            source_location=getattr(call.get("node"), "source_location", None),
        )

    def glsl_private_pointer_contract_function_name(self, function_name):
        if function_name in self.function_private_pointer_parameters:
            return function_name
        candidates = [
            key
            for key, source_name in self.function_private_pointer_source_names.items()
            if source_name == function_name
        ]
        return candidates[0] if len(candidates) == 1 else None

    def glsl_private_pointer_offset_sum(self, base, delta):
        if base in {None, "", "0", "0u"}:
            return delta
        if delta in {None, "", "0", "0u", "int(0)", "int(0u)"}:
            return base
        return f"({base} + {delta})"

    def glsl_private_pointer_member_view_binding(self, operand):
        """Bind ``&root.member`` to the member lvalue of a private value."""
        root, _members = member_view_chain(operand)
        source_type = self.local_variable_source_types.get(root)
        if source_type is None:
            source_type = self.local_variable_types.get(root)
        if not is_single_object_type(self.type_name_string(source_type)):
            return None
        binding = glsl_member_view_root(
            self, operand, {"root_kind": "scalar_value", "root": root}
        )
        if binding is None:
            return None
        return {
            "backing": self.generate_expression(operand),
            "offset": "0",
            "root_kind": "scalar",
            "source_type": binding["source_type"],
            "member_view": True,
        }

    def glsl_private_pointer_view_binding(self, expression):
        function_name = self.current_glsl_private_pointer_function_name
        local_arrays = self.function_private_pointer_local_arrays.get(function_name, {})
        if isinstance(expression, PointerReinterpretNode):
            binding = self.glsl_private_pointer_view_binding(expression.expression)
            if binding is None:
                return None
            return self.glsl_private_pointer_reinterpret_binding(
                expression,
                binding,
                self.glsl_active_literal_int_constants(),
            )

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            raw_name = (
                expression
                if isinstance(expression, str)
                else getattr(expression, "name", None)
            )
            active_source_type = self.local_variable_source_types.get(raw_name)
            _active_base_type, active_array_suffix = split_array_type_suffix(
                str(active_source_type or "")
            )
            if raw_name in local_arrays and active_array_suffix:
                return {
                    "backing": self.generate_glsl_identifier_expression(raw_name),
                    "offset": "0",
                    "root_kind": "local",
                }
            if raw_name in self.current_glsl_private_pointer_base_names:
                variant = self.current_glsl_private_pointer_variant.get(raw_name)
                if self.glsl_private_pointer_byte_view_variant(variant):
                    return {
                        "backing": self.generate_glsl_identifier_expression(raw_name),
                        "offset": self.current_glsl_private_pointer_base_names[
                            raw_name
                        ],
                        "root_kind": "byte_view_parameter",
                        "byte_view": variant,
                        "element_type": variant[7],
                    }
                if active_source_type and not str(active_source_type).rstrip().endswith(
                    "*"
                ):
                    return None
                if self.current_glsl_private_pointer_variant.get(raw_name) in {
                    0,
                    "0",
                }:
                    return {
                        "backing": self.generate_glsl_identifier_expression(raw_name),
                        "offset": "0",
                        "root_kind": "scalar_parameter",
                    }
                return {
                    "backing": self.generate_glsl_identifier_expression(raw_name),
                    "offset": self.current_glsl_private_pointer_base_names[raw_name],
                    "root_kind": "parameter",
                    "parameter_name": raw_name,
                    "extent": self.current_glsl_private_pointer_variant.get(raw_name),
                }
            if raw_name in local_arrays:
                return {
                    "backing": self.generate_glsl_identifier_expression(raw_name),
                    "offset": "0",
                    "root_kind": "local",
                }
            return None

        if isinstance(expression, ArrayAccessNode):
            row = self.glsl_private_pointer_array_row(
                expression,
                local_arrays,
                set(self.current_glsl_private_pointer_base_names),
            )
            if row is None:
                return None
            if any(
                self.glsl_private_pointer_expression_has_side_effects(index)
                for index in row[3]
            ):
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL cannot emit a side-effecting private array row index",
                    function_name=self.glsl_private_pointer_display_name(function_name),
                    reason="side-effecting-view-offset",
                    source_location=getattr(expression, "source_location", None),
                )
            return {
                "backing": self.generate_expression(expression),
                "offset": "0",
                "root_kind": "local",
            }

        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            operand = expression.operand
            if isinstance(operand, (str, IdentifierNode, VariableNode)):
                raw_name = (
                    operand
                    if isinstance(operand, str)
                    else getattr(operand, "name", None)
                )
                source_type = self.local_variable_source_types.get(raw_name)
                if source_type is None:
                    source_type = self.local_variable_types.get(raw_name)
                _base_type, array_suffix = split_array_type_suffix(
                    str(source_type or "")
                )
                if (
                    source_type
                    and not array_suffix
                    and not str(source_type).rstrip().endswith(("*", "&"))
                ):
                    return {
                        "backing": self.generate_glsl_identifier_expression(raw_name),
                        "offset": "0",
                        "root_kind": "scalar",
                        "source_type": source_type,
                    }
            if isinstance(operand, MemberAccessNode):
                return self.glsl_private_pointer_member_view_binding(operand)
            if not isinstance(operand, ArrayAccessNode):
                return None
            binding = self.glsl_private_pointer_view_binding(operand.array)
            if binding is None:
                return None
            delta_expression = operand.index
            operator = "+"
        elif isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            binding = self.glsl_private_pointer_view_binding(expression.left)
            delta_expression = expression.right
            operator = expression.op
            if binding is None and expression.op == "+":
                binding = self.glsl_private_pointer_view_binding(expression.right)
                delta_expression = expression.left
            if binding is None:
                return None
        else:
            return None

        if self.glsl_private_pointer_expression_has_side_effects(delta_expression):
            display_name = self.glsl_private_pointer_display_name(function_name)
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot emit a side-effecting private pointer view offset",
                function_name=display_name,
                reason="side-effecting-view-offset",
                source_location=getattr(expression, "source_location", None),
            )
        rendered_delta = self.generate_expression(delta_expression)
        if operator == "-":
            rendered_delta = f"-({rendered_delta})"
        rendered_delta = f"int({rendered_delta})"
        binding["offset"] = self.glsl_private_pointer_offset_sum(
            binding["offset"], rendered_delta
        )
        return binding

    def glsl_private_pointer_byte_view_read_expression(
        self,
        binding,
        index_expression,
    ):
        variant = binding.get("byte_view")
        if not self.glsl_private_pointer_byte_view_variant(variant):
            return None
        if self.glsl_private_pointer_expression_has_side_effects(index_expression):
            display_name = self.glsl_private_pointer_display_name(
                self.current_glsl_private_pointer_function_name
            )
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot lower a side-effecting private byte-view index in "
                f"'{display_name}'",
                function_name=display_name,
                reason="side-effecting-byte-view-index",
                source_location=getattr(index_expression, "source_location", None),
            )
        rendered_index = (
            str(index_expression)
            if isinstance(index_expression, (int, float))
            else self.generate_expression(index_expression)
        )
        element_index = self.glsl_private_pointer_offset_sum(
            binding.get("offset", "0"),
            f"int({rendered_index})",
        )
        reinterpret_binding = {
            "root": f"{binding['backing']}.{variant[3]}",
            "byte_offset": "0",
            "pointer_reinterpretation": {
                "source_layout": scalar_storage_layout(variant[6]),
                "target_layout": scalar_storage_layout(variant[7]),
            },
        }
        return self.glsl_pointer_reinterpret_read_expression(
            reinterpret_binding,
            element_index,
        )

    def validate_glsl_private_pointer_byte_view_mutation(self, expression):
        scalar_struct = self.glsl_private_scalar_struct_reinterpret_contract(expression)
        if scalar_struct is not None:
            self.glsl_private_scalar_struct_reinterpret_error(
                scalar_struct["reinterpretation"],
                source_type=scalar_struct["source_type"],
                target_type=scalar_struct["target_type"],
                reason="private-scalar-struct-view-write-unsupported",
                detail="the scalar-to-struct reinterpretation is read-only",
                alignment=scalar_struct["member_layout"].byte_width,
                access="write",
            )
        if isinstance(expression, ArrayAccessNode):
            binding = self.glsl_private_pointer_view_binding(expression.array)
        elif (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "*"
        ):
            binding = self.glsl_private_pointer_view_binding(expression.operand)
        else:
            binding = None
        variant = binding.get("byte_view") if binding else None
        if not self.glsl_private_pointer_byte_view_variant(variant):
            return
        target_layout = scalar_storage_layout(variant[7])
        raise PointerReinterpretationError(
            "OpenGL private pointer reinterpretation is read-only for local "
            "struct word-array backing",
            source_type=variant[1],
            target_type=variant[7],
            address_space="thread",
            alignment=target_layout.byte_width,
            access="write",
            target_backend="opengl",
            reason="private-byte-view-write-unsupported",
            source_location=getattr(expression, "source_location", None),
        )

    def generate_glsl_private_pointer_view_access(
        self,
        pointer_expression,
        index_expression=0,
        *,
        access_expression=None,
    ):
        binding = self.glsl_private_pointer_view_binding(pointer_expression)
        byte_view_read = self.glsl_private_pointer_byte_view_read_expression(
            binding or {},
            index_expression,
        )
        if byte_view_read is not None:
            return byte_view_read
        if binding is None or binding.get("root_kind") not in {
            "parameter",
            "scalar_parameter",
        }:
            return None
        if binding["root_kind"] == "scalar_parameter":
            if self.glsl_private_pointer_expression_has_side_effects(index_expression):
                display_name = self.glsl_private_pointer_display_name(
                    self.current_glsl_private_pointer_function_name
                )
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL scalar private pointer index has side effects in "
                    f"'{display_name}'",
                    function_name=display_name,
                    reason="side-effecting-scalar-index",
                    source_location=getattr(index_expression, "source_location", None),
                )
            proven_zero_access = (
                access_expression is not None
                and id(access_expression)
                in self.glsl_private_pointer_zero_index_access_ids
            )
            literal_index = self.literal_int_value(
                index_expression, self.literal_int_constants
            )
            if not proven_zero_access and not (
                binding.get("offset") in {None, "", "0", "0u", "int(0)", "int(0u)"}
                and literal_index == 0
            ):
                display_name = self.glsl_private_pointer_display_name(
                    self.current_glsl_private_pointer_function_name
                )
                raise OpenGLPrivatePointerParameterError(
                    "OpenGL scalar private pointer views support only a proven "
                    f"zero index in '{display_name}'",
                    function_name=display_name,
                    reason="unprovable-scalar-index",
                    source_location=getattr(index_expression, "source_location", None),
                )
            return binding["backing"]
        rendered_index = (
            str(index_expression)
            if isinstance(index_expression, (int, float))
            else self.generate_expression(index_expression)
        )
        element_index = f"({binding['offset']} + int({rendered_index}))"
        parameter_name = binding.get("parameter_name")
        extent = binding.get("extent")
        recovery_key = (
            self.current_glsl_private_pointer_function_name,
            parameter_name,
            extent,
        )
        if recovery_key not in self.glsl_private_pointer_robust_zero_read_variants:
            return f"{binding['backing']}[{element_index}]"
        if self.glsl_private_pointer_expression_has_side_effects(index_expression):
            display_name = self.glsl_private_pointer_display_name(
                self.current_glsl_private_pointer_function_name
            )
            raise OpenGLPrivatePointerParameterError(
                "OpenGL robust private-pointer read index has side effects in "
                f"'{display_name}.{parameter_name}'",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="side-effecting-robust-read-index",
                source_location=getattr(index_expression, "source_location", None),
            )
        element_type = self.function_private_pointer_element_types.get(
            self.current_glsl_private_pointer_function_name, {}
        ).get(parameter_name)
        zero_value = self.default_value_expression_for_type(
            element_type
        ) or self.primitive_default_value_expression(element_type)
        return (
            f"(({element_index} >= 0 && {element_index} < {extent}) ? "
            f"{binding['backing']}[{element_index}] : {zero_value})"
        )

    def glsl_private_pointer_call_argument_binding(
        self,
        function_name,
        parameter_name,
        argument,
    ):
        contract_name = self.glsl_private_pointer_contract_function_name(function_name)
        if contract_name is None:
            return None
        parameter_names = {
            parameter.name
            for parameter in self.function_private_pointer_parameters[contract_name]
        }
        if parameter_name not in parameter_names:
            return None
        binding = self.glsl_private_pointer_view_binding(argument)
        if binding is None:
            display_name = self.glsl_private_pointer_display_name(contract_name)
            raise OpenGLPrivatePointerParameterError(
                "OpenGL private pointer argument for "
                f"'{display_name}.{parameter_name}' has no concrete backing array",
                function_name=display_name,
                parameter_name=parameter_name,
                reason="missing-view-backing",
                source_location=getattr(argument, "source_location", None),
            )
        return binding

    def glsl_private_pointer_member_expression(
        self,
        pointer_expression,
        member,
    ):
        pointee = self.generate_glsl_private_pointer_view_access(pointer_expression)
        if pointee is None:
            return None
        return f"{pointee}.{member}"

    def collect_unsized_sampled_texture_globals(self, ast):
        globals_by_name = {}
        for node in [
            *(getattr(ast, "global_variables", []) or []),
            *collect_stage_local_variables(ast),
        ]:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_unsized_sampled_texture_array_type(vtype):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_structured_buffer_global_array_size_hints(self, ast, function_hints):
        structured_globals = self.collect_unsized_structured_buffer_globals(ast)
        if not structured_globals:
            return {}

        required_sizes = {name: 0 for name in structured_globals}
        fixed_requirements = {}
        functions_by_name = {
            getattr(func, "name", None): func for func in self.collect_functions(ast)
        }
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }

        def raise_fixed_size_conflict(name, left, right):
            raise OpenGLRefusal(
                "Conflicting fixed resource array sizes for "
                f"'{name}': {left} and {right}"
            )

        def record_required_size(name, size):
            size = self.positive_int_value(size)
            if name not in required_sizes or size is None:
                return
            fixed_size = fixed_requirements.get(name)
            if fixed_size is not None and size > fixed_size:
                raise_fixed_size_conflict(name, size, fixed_size)
            required_sizes[name] = max(required_sizes[name], size)

        def record_fixed_requirement(name, size):
            size = self.positive_int_value(size)
            if name not in required_sizes or size is None:
                return
            fixed_size = fixed_requirements.get(name)
            if fixed_size is not None and fixed_size != size:
                raise_fixed_size_conflict(name, fixed_size, size)
            current_size = required_sizes.get(name, 0)
            if current_size and current_size > size:
                raise_fixed_size_conflict(name, current_size, size)
            fixed_requirements[name] = size
            required_sizes[name] = max(current_size, size)

        for node in self.walk_ast(ast):
            if isinstance(node, ArrayAccessNode):
                array_expr = getattr(node, "array", getattr(node, "array_expr", None))
                array_name = self.expression_name(array_expr)
                if array_name not in structured_globals:
                    continue
                index_expr = getattr(node, "index", getattr(node, "index_expr", None))
                index = self.literal_int_value(index_expr, self.literal_int_constants)
                if index is not None and index >= 0:
                    record_required_size(array_name, index + 1)
                continue

            if not isinstance(node, FunctionCallNode):
                continue
            callee_name = self.function_call_name(node)
            callee = functions_by_name.get(callee_name)
            if callee is None:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
            callee_hints = function_hints.get(callee_name, {})
            for index, arg in enumerate(args):
                if index >= len(params):
                    break
                arg_name = self.expression_name(arg)
                if arg_name not in structured_globals:
                    continue
                param = params[index]
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                base_type, _ = self.structured_buffer_array_base_type(param_type)
                if base_type is None or not self.is_structured_buffer_type(base_type):
                    continue
                param_name = getattr(param, "name", None)
                fixed_size = self.fixed_resource_array_size(param_type)
                if fixed_size is not None:
                    record_fixed_requirement(arg_name, fixed_size)
                else:
                    record_required_size(arg_name, callee_hints.get(param_name))

        def structured_buffer_param(params, index):
            if index >= len(params):
                return None, None, None
            param = params[index]
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            base_type, _ = self.structured_buffer_array_base_type(param_type)
            if base_type is None or not self.is_structured_buffer_type(base_type):
                return None, None, None
            return param, param_type, getattr(param, "name", None)

        def function_hint_size(func_name, param_name):
            return self.positive_int_value(
                function_hints.get(func_name, {}).get(param_name)
            )

        def record_function_required_size(func_name, param_name, size):
            size = self.positive_int_value(size)
            if not func_name or not param_name or size is None:
                return False
            hints = function_hints.setdefault(func_name, {})
            current_size = self.positive_int_value(hints.get(param_name))
            if current_size is not None and current_size >= size:
                return False
            hints[param_name] = str(size)
            return True

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions_by_name.items():
                caller_hints = function_hints.get(caller_name, {})
                for call in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue
                    params = list(
                        getattr(callee, "parameters", getattr(callee, "params", []))
                    )
                    args = list(
                        getattr(call, "arguments", getattr(call, "args", [])) or []
                    )
                    for index, arg in enumerate(args):
                        _, param_type, param_name = structured_buffer_param(
                            params, index
                        )
                        if (
                            param_type is None
                            or param_name is None
                            or self.fixed_resource_array_size(param_type) is not None
                        ):
                            continue

                        arg_name = self.expression_name(arg)
                        if not arg_name:
                            continue

                        callee_size = function_hint_size(callee_name, param_name)
                        if arg_name in required_sizes:
                            if callee_size is not None:
                                current_size = required_sizes.get(arg_name, 0)
                                record_required_size(arg_name, callee_size)
                                changed = (
                                    required_sizes.get(arg_name, 0) != current_size
                                    or changed
                                )
                            changed = (
                                record_function_required_size(
                                    callee_name,
                                    param_name,
                                    required_sizes.get(arg_name),
                                )
                                or changed
                            )
                            continue

                        caller_size = self.positive_int_value(
                            caller_hints.get(arg_name)
                        )
                        if caller_size is not None:
                            changed = (
                                record_function_required_size(
                                    callee_name, param_name, caller_size
                                )
                                or changed
                            )

        return {
            name: str(size)
            for name, size in required_sizes.items()
            if size is not None and size > 1
        }

    def collect_unsized_structured_buffer_globals(self, ast):
        globals_by_name = {}
        for node in [
            *(getattr(ast, "global_variables", []) or []),
            *collect_stage_local_variables(ast),
        ]:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            base_type, declared_size = self.structured_buffer_array_base_type(vtype)
            if (
                name
                and base_type is not None
                and declared_size is None
                and self.is_structured_buffer_type(base_type)
            ):
                globals_by_name[name] = vtype
        return globals_by_name

    def collect_fixed_resource_global_sizes(self, ast):
        global_arrays = {}
        for node in [
            *(getattr(ast, "global_variables", []) or []),
            *collect_stage_local_variables(ast),
        ]:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            size = self.fixed_resource_array_size(vtype)
            if name and size is not None:
                global_arrays[name] = size
        return global_arrays

    def collect_unsized_sampled_texture_parameters(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                vtype = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_unsized_resource_array_type(vtype):
                    function_arrays.setdefault(func_name, {})[param.name] = vtype
        return function_arrays

    def collect_fixed_resource_parameter_sizes(self, ast):
        function_arrays = {}
        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                size = self.fixed_resource_array_size(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                if size is not None:
                    function_arrays.setdefault(func_name, {})[param.name] = size
        return function_arrays
