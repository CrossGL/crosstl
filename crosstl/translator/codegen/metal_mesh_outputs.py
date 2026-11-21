"""Mesh outputs and function parameters in the Metal generator: mesh output flow validation, output counts, dispatch contexts and payload parameters, local identifier remaps, mesh topology, parameter declarations and collection, structured buffer length dependencies and global resource names."""

from ...glsl_builtins import GLSL_BUILTIN_INT_LIMITS
from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MemberAccessNode,
    MeshOpNode,
    ReturnNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)
from ..refusals import MetalRefusal
from .array_utils import split_array_type_suffix
from .metal_exact_lookups import metal_unrenamed_identifier
from .metal_varyings import is_metal_varying_builtin
from .stage_utils import normalize_stage_name


class MetalMeshOutputMixin:
    """Mesh output members of ``MetalCodeGen``."""

    def merge_metal_mesh_output_flow_states(self, states, fallthrough_terminators=()):
        fallthrough_states = []
        propagated_exits = []
        for state in states or []:
            propagated_exits.extend(state.get("exits", []))
            if (
                state.get("falls_through", True)
                or state.get("terminator") in fallthrough_terminators
            ):
                fallthrough_states.append(
                    self.copy_metal_mesh_output_flow_state(
                        state,
                        falls_through=True,
                        exits=[],
                    )
                )
            elif state.get("terminator") is not None:
                propagated_exits.append(self.metal_mesh_output_exit_state(state))
        if not fallthrough_states:
            return self.metal_mesh_output_flow_state(
                falls_through=False,
                exits=propagated_exits,
            )

        counts_seen = all(
            state.get("counts_seen", False) for state in fallthrough_states
        )
        counts = None
        if counts_seen:
            counts = {}
            for role in ("vertices", "primitives"):
                role_values = []
                role_has_unknown = False
                for state in fallthrough_states:
                    state_counts = state.get("counts")
                    if state_counts is None or state_counts.get(role) is None:
                        role_has_unknown = True
                        break
                    role_values.append(state_counts[role])
                counts[role] = (
                    None if role_has_unknown or not role_values else min(role_values)
                )

        return self.metal_mesh_output_flow_state(
            counts_seen=counts_seen,
            counts=counts,
            falls_through=True,
            exits=propagated_exits,
        )

    def metal_mesh_output_states_for_terminator(self, state, terminator):
        states = [
            exit_state
            for exit_state in state.get("exits", [])
            if exit_state.get("terminator") == terminator
        ]
        if (
            not state.get("falls_through", True)
            and state.get("terminator") == terminator
        ):
            states.append(self.metal_mesh_output_exit_state(state, terminator))
        return states

    def metal_mesh_statement_list(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        return [body]

    def validate_metal_mesh_output_statement_sequence(self, statements, state):
        current = self.copy_metal_mesh_output_flow_state(state)
        for stmt in statements or []:
            if not current.get("falls_through", True):
                break
            current = self.validate_metal_mesh_output_statement_flow(stmt, current)
        return current

    def validate_metal_mesh_output_statement_flow(self, stmt, state):
        current = self.copy_metal_mesh_output_flow_state(state)
        if isinstance(stmt, IfNode):
            return self.validate_metal_mesh_if_output_flow(stmt, current)
        if isinstance(stmt, BlockNode):
            return self.validate_metal_mesh_output_statement_sequence(
                self.metal_mesh_statement_list(stmt),
                current,
            )
        if isinstance(stmt, SwitchNode):
            return self.validate_metal_mesh_switch_output_flow(stmt, current)
        if isinstance(stmt, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)):
            return self.validate_metal_mesh_loop_output_flow(stmt, current)
        if isinstance(stmt, ReturnNode):
            return self.copy_metal_mesh_output_flow_state(
                current, falls_through=False, terminator="return"
            )
        if isinstance(stmt, BreakNode):
            return self.copy_metal_mesh_output_flow_state(
                current, falls_through=False, terminator="break"
            )
        if isinstance(stmt, ContinueNode):
            return self.copy_metal_mesh_output_flow_state(
                current, falls_through=False, terminator="continue"
            )

        expr = getattr(stmt, "expression", stmt)
        if isinstance(expr, IfNode):
            return self.validate_metal_mesh_if_output_flow(expr, current)
        if isinstance(expr, SwitchNode):
            return self.validate_metal_mesh_switch_output_flow(expr, current)

        mesh_counts = self.metal_mesh_set_output_counts(expr)
        if mesh_counts is not None:
            return self.metal_mesh_output_flow_state(
                counts_seen=True,
                counts=mesh_counts,
                falls_through=current.get("falls_through", True),
                exits=current.get("exits", []),
            )

        if isinstance(expr, AssignmentNode):
            self.validate_metal_mesh_output_assignment_usage(
                expr,
                current.get("counts_seen", False),
                current.get("counts"),
            )
            return current

        if isinstance(expr, FunctionCallNode):
            self.validate_metal_mesh_output_helper_call_usage(
                expr,
                current.get("counts_seen", False),
                current.get("counts"),
            )
            return current

        nested_statements = self.metal_mesh_nested_statements(stmt)
        for nested in nested_statements:
            self.validate_metal_mesh_output_statement_sequence(nested, current)

        return current

    def validate_metal_mesh_if_output_flow(self, stmt, state):
        then_state = self.validate_metal_mesh_output_statement_sequence(
            self.metal_mesh_statement_list(getattr(stmt, "then_branch", None)),
            state,
        )
        else_branch = getattr(stmt, "else_branch", None)
        if else_branch is None:
            else_state = self.copy_metal_mesh_output_flow_state(state)
        else:
            else_state = self.validate_metal_mesh_output_statement_sequence(
                self.metal_mesh_statement_list(else_branch),
                state,
            )
        return self.merge_metal_mesh_output_flow_states([then_state, else_state])

    def validate_metal_mesh_switch_output_flow(self, stmt, state):
        branch_states = []
        has_default = False
        for case in getattr(stmt, "cases", []) or []:
            if getattr(case, "value", None) is None:
                has_default = True
            branch_states.append(
                self.validate_metal_mesh_output_statement_sequence(
                    self.metal_mesh_statement_list(getattr(case, "statements", [])),
                    state,
                )
            )

        default_case = getattr(stmt, "default_case", None)
        if default_case is not None:
            has_default = True
            branch_states.append(
                self.validate_metal_mesh_output_statement_sequence(
                    self.metal_mesh_statement_list(default_case),
                    state,
                )
            )

        if not has_default:
            branch_states.append(self.copy_metal_mesh_output_flow_state(state))
        return self.merge_metal_mesh_output_flow_states(
            branch_states,
            fallthrough_terminators={"break"},
        )

    def validate_metal_mesh_loop_output_flow(self, stmt, state):
        body_state = self.validate_metal_mesh_output_statement_sequence(
            self.metal_mesh_statement_list(getattr(stmt, "body", [])),
            state,
        )
        break_states = self.metal_mesh_output_states_for_terminator(body_state, "break")
        continue_states = self.metal_mesh_output_states_for_terminator(
            body_state, "continue"
        )
        return_states = self.metal_mesh_output_states_for_terminator(
            body_state, "return"
        )

        loop_exits = list(return_states)
        if isinstance(stmt, LoopNode) or self.metal_mesh_loop_condition_is_true(stmt):
            fallthrough_states = break_states
        elif isinstance(stmt, DoWhileNode):
            fallthrough_states = list(break_states)
            if body_state.get("falls_through", True):
                fallthrough_states.append(
                    self.copy_metal_mesh_output_flow_state(body_state, exits=[])
                )
            fallthrough_states.extend(continue_states)
        else:
            fallthrough_states = [self.copy_metal_mesh_output_flow_state(state)]
            if body_state.get("falls_through", True):
                fallthrough_states.append(
                    self.copy_metal_mesh_output_flow_state(body_state, exits=[])
                )
            fallthrough_states.extend(break_states)
            fallthrough_states.extend(continue_states)

        if not fallthrough_states:
            return self.metal_mesh_output_flow_state(
                falls_through=False,
                exits=loop_exits,
            )

        merged = self.merge_metal_mesh_output_flow_states(
            fallthrough_states,
            fallthrough_terminators={"break", "continue"},
        )
        return self.copy_metal_mesh_output_flow_state(
            merged,
            exits=loop_exits + merged.get("exits", []),
        )

    def metal_mesh_loop_condition_is_true(self, stmt):
        if isinstance(stmt, LoopNode):
            return True
        if isinstance(stmt, ForNode) and getattr(stmt, "condition", None) is None:
            return True

        condition = getattr(stmt, "condition", None)
        literal_value = self.metal_literal_bool_value(condition)
        return literal_value is True

    def metal_literal_bool_value(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return None
        if hasattr(value, "value"):
            return self.metal_literal_bool_value(getattr(value, "value"))
        return None

    def metal_mesh_set_output_counts(self, expr):
        if not isinstance(expr, MeshOpNode) or expr.operation != "SetMeshOutputCounts":
            return None
        if len(expr.arguments) != 2:
            raise MetalRefusal(
                "Metal mesh SetMeshOutputCounts requires exactly 2 arguments"
            )

        vertex_count = self.literal_int_value(
            expr.arguments[0], self.literal_int_constants
        )
        primitive_count = self.literal_int_value(
            expr.arguments[1], self.literal_int_constants
        )
        self.validate_metal_mesh_output_count_bound(
            "numVertices",
            vertex_count,
            self.current_metal_mesh_output_config["max_vertices"],
        )
        self.validate_metal_mesh_output_count_bound(
            "numPrimitives",
            primitive_count,
            self.current_metal_mesh_output_config["max_primitives"],
        )
        return {
            "vertices": vertex_count,
            "primitives": primitive_count,
        }

    def validate_metal_mesh_output_count_bound(self, label, value, declared_bound):
        if value is None:
            return
        declared = self.metal_literal_int_text_value(declared_bound)
        if declared is None:
            return
        if value > declared:
            raise MetalRefusal(
                f"Metal mesh SetMeshOutputCounts {label} argument "
                f"({value}) cannot exceed declared output count ({declared})"
            )

    def validate_metal_mesh_output_assignment_usage(
        self, assignment, set_counts_seen, set_counts
    ):
        target = getattr(assignment, "target", getattr(assignment, "left", None))
        target_info = self.metal_mesh_output_assignment_target(target)
        if target_info is None:
            return

        role = target_info["role"]
        name = target_info["name"]
        if not set_counts_seen:
            raise MetalRefusal(
                f"Metal mesh output array '{name}' must be written only after "
                "SetMeshOutputCounts"
            )

        index_value = self.literal_int_value(
            target_info.get("index"), self.literal_int_constants
        )
        if index_value is None:
            return

        declared_bound = self.metal_literal_int_text_value(
            target_info["output"]["count"]
        )
        if declared_bound is not None and index_value >= declared_bound:
            raise MetalRefusal(
                f"Metal mesh {role} output array '{name}' index ({index_value}) "
                f"must be less than declared array size ({declared_bound})"
            )

        active_bound = None
        if set_counts is not None:
            active_bound = (
                set_counts.get("vertices")
                if role == "vertices"
                else set_counts.get("primitives")
            )
        if active_bound is not None and index_value >= active_bound:
            raise MetalRefusal(
                f"Metal mesh {role} output array '{name}' index ({index_value}) "
                f"must be less than SetMeshOutputCounts "
                f"{'numVertices' if role == 'vertices' else 'numPrimitives'} "
                f"({active_bound})"
            )

    def validate_metal_mesh_output_helper_call_usage(
        self, call, set_counts_seen, set_counts
    ):
        func_name = self.function_call_name(call)
        if func_name == "SetIndex":
            self.validate_metal_mesh_set_index_helper_call_usage(
                call, set_counts_seen, set_counts
            )
            return

        helper_roles = {
            "SetVertex": ("vertices", "max_vertices", "numVertices"),
            "SetPrimitive": ("primitives", "max_primitives", "numPrimitives"),
        }
        role_info = helper_roles.get(func_name)
        if role_info is None:
            return

        args = getattr(call, "arguments", getattr(call, "args", []))
        if len(args) != 2:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' requires exactly 2 arguments"
            )

        if not set_counts_seen:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' must be called only after "
                "SetMeshOutputCounts"
            )

        index_value = self.literal_int_value(args[0], self.literal_int_constants)
        if index_value is None:
            return

        role, declared_key, active_label = role_info
        mesh_output = self.current_metal_mesh_output_config or {}
        declared_bound = self.metal_literal_int_text_value(
            mesh_output.get(declared_key)
        )
        if declared_bound is not None and index_value >= declared_bound:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' index ({index_value}) "
                f"must be less than declared output count ({declared_bound})"
            )

        active_bound = None
        if set_counts is not None:
            active_bound = set_counts.get(role)
        if active_bound is not None and index_value >= active_bound:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' index ({index_value}) "
                f"must be less than SetMeshOutputCounts {active_label} "
                f"({active_bound})"
            )

    def validate_metal_mesh_set_index_helper_call_usage(
        self, call, set_counts_seen, set_counts
    ):
        func_name = self.function_call_name(call)
        args = getattr(call, "arguments", getattr(call, "args", []))
        if len(args) != 2:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' requires exactly 2 arguments"
            )

        if not set_counts_seen:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' must be called only after "
                "SetMeshOutputCounts"
            )

        index_value = self.literal_int_value(args[0], self.literal_int_constants)
        if index_value is None:
            return

        write_width = self.metal_mesh_index_vector_width(args[1]) or 1
        mesh_output = self.current_metal_mesh_output_config or {}
        topology_width = self.metal_mesh_topology_index_width(
            mesh_output.get("topology")
        )
        if topology_width is None:
            return

        declared_primitives = self.metal_literal_int_text_value(
            mesh_output.get("max_primitives")
        )
        if declared_primitives is not None:
            declared_index_count = declared_primitives * topology_width
            self.validate_metal_mesh_set_index_span(
                func_name,
                index_value,
                write_width,
                declared_index_count,
                "declared flattened index count",
            )

        active_primitives = None
        if set_counts is not None:
            active_primitives = set_counts.get("primitives")
        if active_primitives is not None:
            active_index_count = active_primitives * topology_width
            self.validate_metal_mesh_set_index_span(
                func_name,
                index_value,
                write_width,
                active_index_count,
                "SetMeshOutputCounts numPrimitives flattened index count",
            )

    def validate_metal_mesh_set_index_span(
        self, func_name, index_value, write_width, flattened_bound, bound_label
    ):
        last_index = index_value + write_width - 1
        if index_value < 0:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' index ({index_value}) "
                "must be non-negative"
            )
        if last_index >= flattened_bound:
            raise MetalRefusal(
                f"Metal mesh output helper '{func_name}' index range "
                f"[{index_value}, {last_index}] must be less than "
                f"{bound_label} ({flattened_bound})"
            )

    def metal_mesh_nested_statements(self, stmt):
        nested = []
        for attribute in (
            "body",
            "then_branch",
            "if_body",
            "else_branch",
            "else_body",
        ):
            value = getattr(stmt, attribute, None)
            if value is None:
                continue
            nested.append(
                getattr(value, "statements", value if isinstance(value, list) else [])
            )
        return [statements for statements in nested if statements]

    def metal_literal_int_text_value(self, value):
        if value is None:
            return None
        try:
            return int(str(value), 0)
        except (TypeError, ValueError):
            return None

    def collect_function_metal_mesh_dispatch_contexts(self, functions):
        functions = list(functions)
        functions_by_name = {
            func.name: func for func in functions if getattr(func, "name", None)
        }
        contexts_by_id = {
            id(func): self.direct_metal_mesh_dispatch_context(func)
            for func in functions
        }

        changed = True
        while changed:
            changed = False
            for func in functions:
                name = getattr(func, "name", None)
                if not name:
                    continue
                context = contexts_by_id.setdefault(id(func), {})
                for called_name in self.called_user_function_names(func):
                    called_func = functions_by_name.get(called_name)
                    called_context = (
                        contexts_by_id.get(id(called_func))
                        if called_func is not None
                        else None
                    )
                    if not called_context:
                        continue
                    if self.merge_metal_mesh_dispatch_context(
                        context, called_context, name, called_name
                    ):
                        changed = True

        self.function_metal_mesh_dispatch_contexts_by_id = {
            func_id: context for func_id, context in contexts_by_id.items() if context
        }
        return {
            getattr(func, "name", None): context
            for func in functions
            for context in [contexts_by_id.get(id(func), {})]
            if getattr(func, "name", None) and context
        }

    def direct_metal_mesh_dispatch_context(self, func):
        context = {}
        if self.function_contains_mesh_op(
            func, "DispatchMesh", argument_counts={1, 3, 4}
        ):
            context["grid"] = True
        if not self.function_contains_mesh_op(
            func, "DispatchMesh", argument_counts={4}
        ):
            return context

        dispatch_payload_type = self.metal_dispatch_mesh_payload_argument_type(func)
        expected_payload_type = self.metal_program_mesh_payload_type
        if (
            dispatch_payload_type is not None
            and expected_payload_type is not None
            and dispatch_payload_type != expected_payload_type
        ):
            raise MetalRefusal(
                "Metal DispatchMesh payload type "
                f"'{dispatch_payload_type}' must match mesh payload type "
                f"'{expected_payload_type}'"
            )

        payload_type = expected_payload_type or dispatch_payload_type
        if payload_type is None:
            raise MetalRefusal("Metal DispatchMesh payload argument type is unknown")
        context["payload"] = True
        context["payload_type"] = payload_type
        return context

    def merge_metal_mesh_dispatch_context(
        self, context, called_context, func_name, called_name
    ):
        changed = False
        if called_context.get("grid") and not context.get("grid"):
            context["grid"] = True
            changed = True
        if called_context.get("payload"):
            if not context.get("payload"):
                context["payload"] = True
                changed = True
            payload_type = context.get("payload_type")
            called_payload_type = called_context.get("payload_type")
            if (
                payload_type
                and called_payload_type
                and payload_type != called_payload_type
            ):
                raise MetalRefusal(
                    "Metal mesh dispatch helper payload type mismatch: "
                    f"'{func_name}' calls '{called_name}' requiring "
                    f"'{called_payload_type}' but already requires '{payload_type}'"
                )
            if called_payload_type and payload_type != called_payload_type:
                context["payload_type"] = called_payload_type
                changed = True
        return changed

    def metal_mesh_dispatch_context_for_function(self, func_or_name):
        if not isinstance(func_or_name, str):
            context = self.function_metal_mesh_dispatch_contexts_by_id.get(
                id(func_or_name)
            )
            if context is not None:
                return context
        name = (
            getattr(func_or_name, "name", None)
            if not isinstance(func_or_name, str)
            else func_or_name
        )
        if not name:
            return {}
        return self.function_metal_mesh_dispatch_contexts.get(name, {})

    def append_metal_mesh_dispatch_context_parameters(
        self, params_str, func, reserved_parameter_names
    ):
        context = self.metal_mesh_dispatch_context_for_function(func)
        if not context:
            return params_str

        reserved_names = set(reserved_parameter_names or ())
        reserved_names.update(self.metal_function_local_variable_names(func))
        if context.get("payload"):
            payload_type = (
                context.get("payload_type") or self.metal_program_mesh_payload_type
            )
            payload_name = self.unique_metal_generated_name(
                "_crossglMeshPayload", reserved_names
            )
            reserved_names.add(payload_name)
            params_str = self.append_parameter_declaration(
                params_str,
                f"object_data {payload_type}& {payload_name}",
            )
            self.current_metal_mesh_payload_parameter = payload_name
            self.current_metal_mesh_payload_type = payload_type
            self.current_address_space_variables[payload_name] = "object_data"

        if context.get("grid"):
            grid_name = self.unique_metal_generated_name(
                "_crossglMeshGrid", reserved_names
            )
            params_str = self.append_parameter_declaration(
                params_str,
                f"mesh_grid_properties {grid_name}",
            )
            self.current_metal_mesh_grid_properties_parameter = grid_name
        return params_str

    def required_metal_mesh_dispatch_context_arguments(self, func_name):
        context = self.metal_mesh_dispatch_context_for_function(func_name)
        if not context:
            return []
        args = []
        if context.get("payload"):
            args.append(self.current_metal_mesh_payload_parameter)
        if context.get("grid"):
            args.append(self.current_metal_mesh_grid_properties_parameter)
        return [arg for arg in args if arg]

    def metal_mesh_dispatch_context_call_diagnostic(self, func_name):
        context = self.metal_mesh_dispatch_context_for_function(func_name)
        if not context:
            return None
        if context.get("payload") and not self.current_metal_mesh_payload_parameter:
            return self.refuse_lowering(
                f"mesh dispatch helper call: function '{func_name}' requires an "
                "object_data payload context",
                "(void)0",
            )
        if (
            context.get("grid")
            and not self.current_metal_mesh_grid_properties_parameter
        ):
            return self.refuse_lowering(
                f"mesh dispatch helper call: function '{func_name}' requires "
                "mesh_grid_properties context",
                "(void)0",
            )
        return None

    def metal_mesh_payload_type_for_program(self, ast):
        payload_types = set()
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            if normalize_stage_name(stage_type) != "mesh":
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            for parameter in getattr(entry_point, "parameters", []) or []:
                if not self.is_metal_mesh_payload_parameter("mesh", parameter):
                    continue
                payload_types.add(self.metal_parameter_mapped_type(parameter))

        if len(payload_types) > 1:
            expected = ", ".join(sorted(payload_types))
            raise MetalRefusal(
                "Metal mesh stages in one pipeline must use one mesh payload type; "
                f"got {expected}"
            )
        return next(iter(payload_types), None)

    def validate_metal_mesh_payload_parameter_placement(self, ast, functions):
        mesh_payload_entry_stages_by_id = {}
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in {"object", "task", "amplification", "mesh"}:
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                mesh_payload_entry_stages_by_id[id(entry_point)] = stage_name
                for parameter in getattr(entry_point, "parameters", []) or []:
                    if self.is_metal_mesh_payload_parameter(stage_name, parameter):
                        self.validate_metal_mesh_payload_parameter_address_space(
                            stage_name, parameter
                        )

        for func in functions:
            if id(func) in mesh_payload_entry_stages_by_id:
                continue
            for parameter in getattr(func, "parameters", []) or []:
                if not self.is_explicit_metal_mesh_payload_semantic(
                    self.semantic_from_node(parameter)
                ):
                    continue
                func_name = getattr(func, "name", "<anonymous>")
                param_name = getattr(parameter, "name", "<anonymous>")
                raise MetalRefusal(
                    "Metal mesh payload parameters are only supported on "
                    "object, task, amplification, or mesh stage entry points; "
                    f"function '{func_name}' parameter '{param_name}' declares "
                    "mesh payload semantics"
                )

    def validate_metal_mesh_payload_parameter_address_space(
        self, stage_name, parameter
    ):
        address_space = self.normalized_address_space(
            self.parameter_address_space(parameter)
        )
        if address_space in {None, "object_data"}:
            return

        param_name = getattr(parameter, "name", "<anonymous>")
        raise MetalRefusal(
            f"Metal mesh payload parameter '{param_name}' on {stage_name} stage "
            f"uses {address_space} address space; mesh payload parameters "
            "require object_data"
        )

    def generated_metal_mesh_payload_parameter(
        self, shader_type, func, reserved_parameter_names
    ):
        if shader_type not in {"object", "task", "amplification"}:
            return None
        if self.current_metal_mesh_payload_parameter is not None:
            return None
        context = self.metal_mesh_dispatch_context_for_function(func)
        if not context.get("payload"):
            return None

        expected_payload_type = self.metal_program_mesh_payload_type
        dispatch_payload_type = context.get("payload_type")
        if (
            dispatch_payload_type is not None
            and expected_payload_type is not None
            and dispatch_payload_type != expected_payload_type
        ):
            raise MetalRefusal(
                "Metal DispatchMesh payload type "
                f"'{dispatch_payload_type}' must match mesh payload type "
                f"'{expected_payload_type}'"
            )

        payload_type = expected_payload_type or dispatch_payload_type
        if payload_type is None:
            raise MetalRefusal("Metal DispatchMesh payload argument type is unknown")

        reserved_names = set(reserved_parameter_names or ())
        reserved_names.update(self.metal_function_local_variable_names(func))
        payload_name = self.unique_metal_generated_name(
            "_crossglMeshPayload", reserved_names
        )
        return payload_type, payload_name

    def metal_dispatch_mesh_payload_argument_type(self, func):
        declared_types = self.metal_function_declared_value_types(func)
        payload_types = set()
        for node in self.metal_render_walks.body_nodes_of_type(func, MeshOpNode):
            arguments = self.normalized_metal_intrinsic_args(node.arguments)
            if node.operation != "DispatchMesh" or len(arguments) != 4:
                continue
            payload_type = self.metal_dispatch_mesh_payload_type(
                arguments[3], declared_types
            )
            if payload_type is not None:
                payload_types.add(payload_type)

        if len(payload_types) > 1:
            expected = ", ".join(sorted(payload_types))
            raise MetalRefusal(
                "Metal DispatchMesh payload arguments must use one type; "
                f"got {expected}"
            )
        return next(iter(payload_types), None)

    def metal_dispatch_mesh_payload_type(self, expr, declared_types):
        if isinstance(expr, ArrayAccessNode):
            payload_type = self.metal_dispatch_mesh_payload_type(
                getattr(expr, "array", getattr(expr, "array_expr", None)),
                declared_types,
            )
            return self.metal_mesh_payload_value_type(payload_type)
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            object_type = self.metal_dispatch_mesh_payload_type(
                object_expr, declared_types
            )
            object_type = self.metal_mesh_payload_value_type(object_type)
            member_type = self.struct_member_types.get(object_type, {}).get(expr.member)
            if member_type is not None:
                return self.metal_mesh_payload_value_type(member_type)
        expr_name = self.expression_name(expr)
        if expr_name in declared_types:
            return self.metal_mesh_payload_value_type(declared_types[expr_name])
        payload_type = self.expression_result_type(expr)
        return self.metal_mesh_payload_value_type(payload_type)

    def metal_mesh_payload_value_type(self, payload_type):
        if not payload_type:
            return None
        payload_type = self.type_name_string(payload_type).strip()
        while payload_type.endswith(("&", "*")):
            payload_type = payload_type[:-1].strip()
        if "[" in payload_type and "]" in payload_type:
            payload_type, _ = split_array_type_suffix(payload_type)
        return self.map_type(payload_type)

    def metal_function_declared_value_types(self, func):
        declared_types = {}
        for parameter in getattr(func, "parameters", []) or []:
            declared_types[parameter.name] = self.metal_parameter_raw_type(parameter)
        # Program-scope groupshared values lowered into this entry are its locals.
        for (
            node
        ) in self.metal_lowered_program_scope_groupshared_globals_by_function.get(
            id(func), []
        ):
            raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
            name = self.metal_variable_name(node)
            if name and raw_type is not None:
                declared_types[name] = self.type_name_string(raw_type)
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            name = getattr(node, "name", None)
            if not name:
                continue
            raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if raw_type is not None:
                declared_types[name] = self.type_name_string(raw_type)
        return declared_types

    def metal_function_local_variable_names(self, func):
        names = set()
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            if getattr(node, "name", None):
                names.add(node.name)
        return names

    def collect_metal_local_identifier_remaps(
        self, func, parameters, stage_parameters, stage_local_variables=None
    ):
        reserved_names = []
        for parameter in parameters or []:
            name = getattr(parameter, "name", None)
            if name:
                reserved_names.append(name)

        ordered_names = []
        for parameter in stage_parameters or []:
            name = parameter.get("name") if isinstance(parameter, dict) else None
            if name:
                ordered_names.append(name)
        for variable in stage_local_variables or []:
            name = getattr(variable, "name", None)
            if name:
                ordered_names.append(name)
        for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
            if getattr(node, "name", None):
                ordered_names.append(node.name)

        used_names = set(reserved_names) | set(ordered_names)
        remaps = {}
        for name in ordered_names:
            if name in remaps or name not in self.METAL_RESERVED_LOCAL_IDENTIFIERS:
                continue
            base_name = f"{name}_"
            candidate = base_name
            suffix = 2
            while candidate in used_names:
                candidate = f"{base_name}{suffix}"
                suffix += 1
            remaps[name] = candidate
            used_names.add(candidate)
        return remaps

    def metal_local_identifier_name(self, name):
        """The Metal spelling of identifier ``name``: its local rename, or the name itself if Metal can spell it."""
        renamed = self.current_local_identifier_remaps.get(name)
        if renamed is not None:
            return renamed
        return metal_unrenamed_identifier(self, name)

    def metal_parameter_mapped_type(self, parameter):
        return self.map_type(self.metal_parameter_raw_type(parameter))

    def metal_parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return self.type_name_string(parameter.param_type)
        if hasattr(parameter, "vtype"):
            return self.type_name_string(parameter.vtype)
        return "float"

    def metal_mesh_grid_properties_parameter(
        self, shader_type, func, reserved_parameter_names
    ):
        if shader_type not in {"object", "task", "amplification"}:
            return None
        context = self.metal_mesh_dispatch_context_for_function(func)
        if not context.get("grid"):
            return None
        return self.unique_metal_generated_name(
            "_crossglMeshGrid", reserved_parameter_names
        )

    def function_contains_mesh_op(self, func, operation, argument_counts=None):
        for node in self.metal_render_walks.body_nodes_of_type(func, MeshOpNode):
            if node.operation != operation:
                continue
            arguments = self.normalized_metal_intrinsic_args(node.arguments)
            if argument_counts is not None and len(arguments) not in argument_counts:
                continue
            return True
        return False

    def unique_metal_generated_name(self, base_name, reserved_names):
        reserved_names = set(reserved_names or ())
        if base_name not in reserved_names:
            return base_name
        index = 1
        while f"{base_name}_{index}" in reserved_names:
            index += 1
        return f"{base_name}_{index}"

    def metal_single_stage_attribute_argument(self, func, attribute_name):
        arguments = self.metal_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise MetalRefusal(
                f"Metal stage attribute {attribute_name} requires exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def metal_mesh_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise MetalRefusal(
                f"Metal mesh output topology cannot be lowered: {topology}"
            )
        return mapped

    def metal_mesh_topology_index_width(self, topology):
        return {"point": 1, "line": 2, "triangle": 3}.get(topology)

    def generate_metal_mesh_vertex_output_struct(self, mesh_output):
        return (
            f"struct {mesh_output['vertex_type']} {{\n"
            "    float4 position [[position]];\n"
            "};\n"
        )

    def metal_mesh_stage_output_parameter_declaration(self, mesh_output):
        return (
            f"mesh<{mesh_output['vertex_type']}, {mesh_output['primitive_type']}, "
            f"{mesh_output['max_vertices']}, {mesh_output['max_primitives']}, "
            f"topology::{mesh_output['topology']}> {mesh_output['parameter_name']}"
        )

    def append_parameter_declaration(self, params_str, declaration):
        if not params_str:
            return declaration
        return f"{params_str}, {declaration}"

    def should_wrap_metal_vertex_stage_input_parameter(
        self, raw_param_type, shader_type, parameter
    ):
        if shader_type != "vertex":
            return False
        if self.is_metal_resource_or_buffer_parameter(
            raw_param_type, parameter, shader_type
        ):
            return False
        if self.type_name_string(raw_param_type) in self.structs_by_name:
            return False
        if self.is_metal_ray_stage(shader_type):
            return False
        if self.is_graphics_stage_output_parameter(parameter, shader_type):
            return False
        # Built-in inputs such as [[vertex_id]] are parameters; a [[stage_in]] struct holds only attributes.
        semantic = self.semantic_from_node(parameter)
        if semantic is not None and is_metal_varying_builtin(
            self.canonical_metal_semantic(semantic)
        ):
            return False
        return True

    def metal_vertex_input_takes_next_attribute(
        self, raw_param_type, shader_type, parameter, semantic
    ):
        """Whether a vertex input with a user semantic gets its attribute from the ``[[stage_in]]`` structure."""
        return self.should_wrap_metal_vertex_stage_input_parameter(
            raw_param_type, shader_type, parameter
        ) and self.is_metal_user_stage_io_semantic(semantic)

    def is_metal_resource_or_buffer_parameter(
        self, raw_param_type, parameter, shader_type=None
    ):
        if self.is_resource_parameter_type(raw_param_type):
            return True
        if self.is_raw_buffer_parameter_type(raw_param_type, parameter):
            return True
        if self.is_bound_scalar_value_parameter(raw_param_type, parameter, shader_type):
            return True
        return False

    def all_functions(self, ast):
        functions = list(getattr(ast, "functions", []) or [])
        for stage in getattr(ast, "stages", {}).values():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
            functions.extend(getattr(stage, "local_functions", []) or [])
        return functions

    def generate_metal_builtin_limit_fallbacks(self, ast, functions):
        name = "gl_MaxImageUnits"
        declared_names = {
            getattr(node, "name", None)
            for node in [
                *(getattr(ast, "constants", []) or []),
                *(getattr(ast, "global_variables", []) or []),
            ]
        }
        if name in declared_names or not self.functions_use_identifier(functions, name):
            return ""
        default = GLSL_BUILTIN_INT_LIMITS[name]
        return (
            "/* CrossGL fallback: GLSL builtin limit specialization "
            "gl_MaxImageUnits is not available in Metal; using the OpenGL "
            "minimum value. */\n"
            f"{self.metal_unused_declaration_qualifier('constant')} "
            f"int gl_MaxImageUnits = {default};\n"
        )

    def functions_use_identifier(self, functions, name):
        return any(
            self.function_body_uses_identifier(getattr(func, "body", []), name)
            for func in functions or []
        )

    def collect_function_parameters(self, functions):
        parameters = []
        for func in functions or []:
            parameters.extend(getattr(func, "parameters", getattr(func, "params", [])))
        return parameters

    def collect_function_parameter_infos(self, functions):
        parameter_infos = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            infos = []
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                infos.append(
                    (getattr(param, "name", None), self.type_name_string(raw_type))
                )
            parameter_infos[func_name] = infos
        return parameter_infos

    def collect_function_parameter_nodes(self, functions):
        parameter_nodes = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_nodes[func_name] = list(
                getattr(func, "parameters", getattr(func, "params", [])) or []
            )
        return parameter_nodes

    def collect_unused_array_parameter_indices(self, functions):
        """Map names to the array parameter indices every overload leaves unused.

        Call sites only know the callee's name, so an index is dropped only when
        each signature declared under that name ignores it.  A prototype has no
        body to inspect: its definition decides, and a prototype without a
        definition keeps every parameter.
        """
        indices_by_signature = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            params = list(
                getattr(func, "parameters", getattr(func, "params", [])) or []
            )
            signature = tuple(
                self.type_name_string(self.parameter_raw_type(param))
                for param in params
            )
            signatures = indices_by_signature.setdefault(func_name, {})
            body = getattr(func, "body", None)
            if body is None:
                signatures.setdefault(signature, set())
                continue
            indices = set()
            for index, param in enumerate(params):
                param_name = getattr(param, "name", None)
                if not param_name:
                    continue
                raw_type = self.parameter_raw_type(param)
                if not self.is_array_type_node(raw_type):
                    continue
                if self.is_resource_parameter_type(raw_type):
                    continue
                if not self.function_body_uses_identifier(body, param_name):
                    indices.add(index)
            signatures[signature] = indices
        skipped = {}
        for func_name, signatures in indices_by_signature.items():
            indices = set.intersection(*signatures.values())
            if indices:
                skipped[func_name] = indices
        return skipped

    def skipped_function_parameter_indices(self, func_name):
        return self.metal_skipped_function_parameter_indices.get(func_name, set())

    def function_body_uses_identifier(self, body, name):
        for node in self.iter_ast_nodes(body):
            if getattr(node, "name", None) == name:
                return True
        return False

    def structured_buffer_parameter_type_map(self, func):
        parameter_types = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])):
            name = getattr(param, "name", None)
            raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
            type_name = self.type_name_string(raw_type)
            if name and self.is_structured_buffer_type(type_name):
                parameter_types[name] = type_name
        return parameter_types

    def function_call_arguments(self, call):
        return getattr(call, "arguments", getattr(call, "args", []))

    def function_call_matches_known_signature(self, func_name, args):
        if func_name not in self.user_function_names:
            return False
        parameter_nodes = self.function_parameter_nodes.get(func_name)
        if parameter_nodes is None:
            return False
        return len(args or []) == len(parameter_nodes)

    def collect_function_structured_buffer_length_dependencies(self, functions):
        parameter_types = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if func_name:
                # Overloads share a name; each keeps only its own buffer
                # parameters when the declaration is rendered.
                parameter_types.setdefault(func_name, {}).update(
                    self.structured_buffer_parameter_type_map(func)
                )
        dependencies = {func_name: set() for func_name in parameter_types}

        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            for node in self.metal_render_walks.body_nodes_of_type(
                func, FunctionCallNode
            ):
                if self.function_call_name(node) != "buffer_dimensions":
                    continue
                args = self.function_call_arguments(node)
                if not args:
                    continue
                buffer_name = self.expression_name(args[0])
                if buffer_name in parameter_types.get(func_name, {}):
                    dependencies[func_name].add(buffer_name)

        changed = True
        while changed:
            changed = False
            for func in functions or []:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                current_dependencies = dependencies.setdefault(func_name, set())
                before = set(current_dependencies)
                for node in self.metal_render_walks.body_nodes_of_type(
                    func, FunctionCallNode
                ):
                    called_name = self.function_call_name(node)
                    if called_name not in self.user_function_names:
                        continue
                    called_dependencies = dependencies.get(called_name, set())
                    if not called_dependencies:
                        continue
                    called_parameters = self.function_parameter_infos.get(
                        called_name, []
                    )
                    for index, arg in enumerate(self.function_call_arguments(node)):
                        if index >= len(called_parameters):
                            continue
                        called_param_name, _called_param_type = called_parameters[index]
                        if called_param_name not in called_dependencies:
                            continue
                        arg_name = self.expression_name(arg)
                        if arg_name in parameter_types.get(func_name, {}):
                            current_dependencies.add(arg_name)
                if current_dependencies != before:
                    changed = True
        return dependencies

    def collect_global_structured_buffer_length_dependencies(
        self, functions, global_vars
    ):
        global_buffer_names = set()
        for node in global_vars or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if name and self.is_structured_buffer_type(raw_type):
                global_buffer_names.add(name)

        dependencies = set()
        for func in functions or []:
            local_names = {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if getattr(param, "name", None)
            }
            for node in self.metal_render_walks.body_nodes_of_type(func, VariableNode):
                if getattr(node, "name", None):
                    local_names.add(node.name)

            for node in self.metal_render_walks.body_nodes_of_type(
                func, FunctionCallNode
            ):
                func_name = self.function_call_name(node)
                args = self.function_call_arguments(node)
                if func_name == "buffer_dimensions" and args:
                    buffer_name = self.expression_name(args[0])
                    if (
                        buffer_name in global_buffer_names
                        and buffer_name not in local_names
                    ):
                        dependencies.add(buffer_name)
                    continue
                if func_name not in self.user_function_names:
                    continue
                called_dependencies = (
                    self.function_structured_buffer_length_dependencies.get(
                        func_name, set()
                    )
                )
                if not called_dependencies:
                    continue
                called_parameters = self.function_parameter_infos.get(func_name, [])
                for index, arg in enumerate(args):
                    if index >= len(called_parameters):
                        continue
                    called_param_name, _called_param_type = called_parameters[index]
                    if called_param_name not in called_dependencies:
                        continue
                    arg_name = self.expression_name(arg)
                    if arg_name in global_buffer_names and arg_name not in local_names:
                        dependencies.add(arg_name)
        return dependencies

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in getattr(root, "global_variables", []) or []:
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and (
                self.is_resource_parameter_type(var_type)
                or self.glsl_buffer_block_attribute(node) is not None
            ):
                resource_names.add(var_name)
        return resource_names
