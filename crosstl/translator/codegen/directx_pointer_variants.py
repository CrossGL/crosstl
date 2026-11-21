"""Workgroup and private pointer variants in the HLSL generator: compute workgroup counts, resource variants, side effects, static offsets and intervals, and resolved local calls."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    AtomicOpNode,
    BinaryOpNode,
    ForInNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    PointerReinterpretNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from . import hlsl_private_pointer_variants
from .array_utils import (
    _UnsignedLiteralInt,
    get_array_size_from_node,
)
from .directx_values import (
    HLSL_NULL_WORKGROUP_POINTER_CONTRACT,
    DirectXPrivatePointerParameterError,
    DirectXWorkgroupPointerError,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_parameters,
)
from .hlsl_host_workgroup_size import hlsl_entry_proof_local_size
from .resource_arrays import (
    collect_resource_array_size_hints,
    is_private_pointer_parameter,
)
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class DirectXPointerVariantMixin:
    """Pointer variant members of ``HLSLCodeGen``."""

    def hlsl_compute_num_workgroups_uniform_names(
        self, ast, target_stage, cbuffers, global_vars, functions
    ):
        if not self.hlsl_compute_target_uses_num_workgroups(
            ast, target_stage, functions
        ):
            return None

        used_names = set(self.global_variable_types)
        used_names.update(self.collect_hlsl_program_identifier_names(ast))
        used_names.update(
            getattr(cbuffer, "name", None)
            for cbuffer in cbuffers or []
            if getattr(cbuffer, "name", None)
        )
        used_names.update(
            getattr(node, "name", getattr(node, "variable_name", None))
            for node in global_vars or []
            if getattr(node, "name", getattr(node, "variable_name", None))
        )

        buffer_name = self.hlsl_unique_local_identifier(
            "CrossGLDispatchInfo", used_names
        )
        used_names.add(buffer_name)
        member_name = self.hlsl_unique_local_identifier(
            "crossglNumWorkGroups", used_names
        )
        self.global_variable_types[member_name] = "uint3"
        return {"buffer_name": buffer_name, "member_name": member_name}

    def hlsl_compute_target_uses_num_workgroups(self, ast, target_stage, functions):
        def function_uses_num_workgroups(function):
            used_builtins = self.used_hlsl_compute_builtin_names(
                getattr(function, "body", [])
            )
            if used_builtins.intersection({"gl_NumWorkGroups", "threads_per_grid"}):
                return True
            for parameter in getattr(
                function, "parameters", getattr(function, "params", [])
            ):
                semantic = self.semantic_from_node(parameter)
                if semantic not in {"gl_NumWorkGroups", "threads_per_grid"}:
                    continue
                parameter_name = getattr(parameter, "name", None)
                if parameter_name and not self.hlsl_compute_parameter_unused(
                    function, parameter_name
                ):
                    return True
            return False

        for func in functions or []:
            if function_stage_name(func) != "compute":
                continue
            if not should_emit_qualified_function(target_stage, "compute"):
                continue
            if function_uses_num_workgroups(func):
                return True

        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name != "compute" or not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            if function_uses_num_workgroups(entry_point):
                return True

        return False

    def collect_hlsl_program_identifier_names(self, ast):
        names = set()
        for node in self.walk_ast(ast):
            name = None
            if isinstance(node, (FunctionNode, StructNode, VariableNode)):
                name = getattr(node, "name", None)
            elif isinstance(node, ForInNode):
                name = getattr(node, "pattern", None)
            elif hasattr(node, "__class__") and "Identifier" in str(node.__class__):
                name = getattr(node, "name", None)
            if name:
                names.add(name)
            if isinstance(node, StructNode):
                for member in getattr(node, "members", []) or []:
                    member_name = getattr(member, "name", None)
                    if member_name:
                        names.add(member_name)
        return names

    def generate_hlsl_compute_num_workgroups_cbuffer(
        self, used_resource_registers, cbuffer_registers
    ):
        uniform = self.hlsl_compute_num_workgroups_uniform
        if not uniform:
            return ""

        space = None
        binding = self.next_available_resource_register(
            used_resource_registers,
            "b",
            cbuffer_registers,
            space,
            1,
        )
        self.reserve_resource_register_range(
            used_resource_registers,
            "b",
            binding,
            1,
            uniform["buffer_name"],
            space,
        )
        self.advance_resource_register(cbuffer_registers, space, binding, 1)
        register = self.resource_register_suffix_for_space("b", binding, space)
        return (
            f"cbuffer {uniform['buffer_name']}{register} {{\n"
            f"    uint3 {uniform['member_name']};\n"
            "};\n"
        )

    def collect_resource_array_size_hints(self, ast):
        return collect_resource_array_size_hints(
            global_arrays=self.collect_unsized_resource_globals(ast),
            function_arrays=self.collect_unsized_resource_parameters(ast),
            fixed_global_array_sizes=self.collect_fixed_resource_global_sizes(ast),
            fixed_function_array_sizes=self.collect_fixed_resource_parameter_sizes(ast),
            functions=self.collect_functions(ast),
            walk_nodes=self.walk_ast,
            expression_name=self.expression_name,
            literal_int_value=self.literal_int_value,
            visible_literal_int_constants=self.visible_literal_int_constants,
            function_call_name=self.function_call_name,
            initial_size=0,
            format_size=lambda size: str(size) if size > 1 else "",
            initial_literal_int_constants=self.initial_literal_int_constants,
        )

    def hlsl_static_workgroup_pointer_binding(
        self, expression, aliases, constants=None
    ):
        if expression is None:
            return None

        if isinstance(expression, ArrayAccessNode):
            binding = self.hlsl_static_workgroup_pointer_binding(
                expression.array, aliases, constants
            )
            if binding is None:
                return None
            binding["offset"] = self.hlsl_static_workgroup_pointer_offset_sum(
                binding.get("offset"), expression.index
            )
            return binding

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            raw_name = (
                expression
                if isinstance(expression, str)
                else getattr(expression, "name", None)
            )
            binding = aliases.get(raw_name)
            if binding is not None and binding.get("kind") == "workgroup-pointer":
                return dict(binding)
            return None

        if isinstance(expression, TernaryOpNode):
            condition = self.literal_int_value(expression.condition, constants or {})
            if condition is None:
                return None
            selected = expression.true_expr if condition else expression.false_expr
            binding = self.hlsl_static_workgroup_pointer_binding(
                selected, aliases, constants
            )
            if binding is not None:
                return binding
            if self.hlsl_null_pointer_expression(selected):
                alternate = expression.false_expr if condition else expression.true_expr
                binding = self.hlsl_static_workgroup_pointer_binding(
                    alternate, aliases, constants
                )
                if binding is not None:
                    binding["selected_null"] = True
                    return binding
            return None

        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            operand = expression.operand
            if isinstance(operand, ArrayAccessNode):
                return self.hlsl_static_workgroup_pointer_binding(
                    operand, aliases, constants
                )
            return None

        if isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
            binding = self.hlsl_static_workgroup_pointer_binding(
                expression.left, aliases, constants
            )
            delta = expression.right
            if binding is None and expression.op == "+":
                binding = self.hlsl_static_workgroup_pointer_binding(
                    expression.right, aliases, constants
                )
                delta = expression.left
            if binding is None:
                return None
            if expression.op == "-":
                delta = UnaryOpNode("-", delta)
            binding["offset"] = self.hlsl_static_workgroup_pointer_offset_sum(
                binding.get("offset"), delta
            )
            return binding

        return None

    def hlsl_static_workgroup_pointer_offset_sum(self, base, delta):
        if base in {None, 0, "0", "0u"}:
            return delta
        if delta in {None, 0, "0", "0u"}:
            return base
        return BinaryOpNode(base, "+", delta)

    def hlsl_workgroup_pointer_entry_intervals(self, ast):
        intervals = {}
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            if normalize_stage_name(stage_type) != "compute":
                continue
            entry_point = getattr(stage, "entry_point", None)
            function_name = getattr(entry_point, "name", None)
            if not function_name:
                continue
            execution_config = getattr(stage, "execution_config", {}) or {}
            if hlsl_entry_proof_local_size(execution_config) is None:
                continue
            sizes = []
            for name in ("local_size_x", "local_size_y", "local_size_z"):
                size = self.literal_int_value(execution_config.get(name, 1), {})
                if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                    sizes = []
                    break
                sizes.append(size)
            if not sizes:
                continue
            invocation_count = sizes[0] * sizes[1] * sizes[2]
            function_intervals = intervals.setdefault(function_name, {})
            for parameter in getattr(entry_point, "parameters", []) or []:
                semantic = self.semantic_from_node(parameter)
                if semantic in {
                    "gl_LocalInvocationIndex",
                    "thread_index_in_threadgroup",
                    "thread_index_in_threadgroup_flat",
                }:
                    function_intervals[parameter.name] = (0, invocation_count - 1)
        return intervals

    def hlsl_workgroup_pointer_access_error(
        self,
        *,
        function_name,
        parameter_name,
        binding,
        access_range,
        node,
    ):
        extent = binding.get("array_size")
        root = binding.get("root")
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            return
        if access_range is None:
            raise self.hlsl_workgroup_pointer_error(
                "DirectX cannot prove the composed workgroup pointer offset for "
                f"'{function_name}.{parameter_name}' into '{root}'",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="unprovable-view-offset",
                node=node,
            )
        lower, upper = access_range
        if lower >= 0 and upper < extent:
            return
        raise self.hlsl_workgroup_pointer_error(
            "DirectX workgroup pointer view "
            f"'{function_name}.{parameter_name}' requires elements {lower} through "
            f"{upper}, but backing array '{root}' has extent {extent}",
            function_name=function_name,
            parameter_name=parameter_name,
            reason="view-out-of-bounds",
            node=node,
        )

    def hlsl_workgroup_pointer_null_observation_error(
        self, *, function_name, parameter_name, node
    ):
        raise self.hlsl_workgroup_pointer_error(
            "DirectX cannot lower statically selected null workgroup pointer "
            f"argument for '{function_name}.{parameter_name}' because the "
            "parameter is observed on the reachable path",
            function_name=function_name,
            parameter_name=parameter_name,
            reason="null-pointer-observed",
            node=node,
        )

    def hlsl_validate_workgroup_pointer_access_ranges(
        self, calls_by_caller, direct_ranges, direct_observations
    ):
        required_ranges = {
            function_name: dict(ranges)
            for function_name, ranges in direct_ranges.items()
        }
        required_observations = {
            function_name: set(parameters)
            for function_name, parameters in direct_observations.items()
        }
        unprovable = {function_name: set() for function_name in direct_ranges}

        def merge_range(function_name, parameter_name, access_range):
            ranges = required_ranges.setdefault(function_name, {})
            current = ranges.get(parameter_name)
            merged = (
                access_range
                if current is None
                else (
                    min(current[0], access_range[0]),
                    max(current[1], access_range[1]),
                )
            )
            if merged == current:
                return False
            ranges[parameter_name] = merged
            return True

        changed = True
        propagation_round = 0
        propagation_limit = max(1, len(calls_by_caller) + 1)
        while changed:
            propagation_round += 1
            if propagation_round > propagation_limit:
                recursive_call = next(
                    (
                        call
                        for calls in calls_by_caller.values()
                        for call in calls
                        if call.get("bindings")
                    ),
                    None,
                )
                parameter_name = (
                    next(iter(recursive_call["bindings"]))
                    if recursive_call is not None
                    else None
                )
                raise self.hlsl_workgroup_pointer_error(
                    "DirectX cannot prove bounded workgroup pointer offsets across "
                    "a recursive helper call graph",
                    function_name=(
                        recursive_call.get("callee") if recursive_call else None
                    ),
                    parameter_name=parameter_name,
                    reason="unprovable-view-offset",
                    node=recursive_call.get("node") if recursive_call else None,
                )
            changed = False
            for caller_name, calls in calls_by_caller.items():
                for call in calls:
                    callee_name = call["callee"]
                    for parameter_name, binding in call["bindings"].items():
                        callee_range = required_ranges.get(callee_name, {}).get(
                            parameter_name
                        )
                        callee_unprovable = parameter_name in unprovable.get(
                            callee_name, set()
                        )
                        callee_observed = parameter_name in required_observations.get(
                            callee_name, set()
                        )
                        if (
                            callee_range is None
                            and not callee_unprovable
                            and not callee_observed
                        ):
                            continue

                        if binding.get("root_kind") == "null":
                            # A null contract has no backing storage: the callee
                            # must not observe it on any reachable path.
                            if (
                                callee_observed
                                or callee_range is not None
                                or callee_unprovable
                            ):
                                self.hlsl_workgroup_pointer_null_observation_error(
                                    function_name=callee_name,
                                    parameter_name=parameter_name,
                                    node=call["node"],
                                )
                            continue

                        composed_range = None
                        if callee_range is not None:
                            offset_range = self.hlsl_private_pointer_interval(
                                binding.get("offset", 0),
                                call.get("intervals", {}),
                                call.get("constants", {}),
                            )
                            if not callee_unprovable and offset_range is not None:
                                composed_range = (
                                    offset_range[0] + callee_range[0],
                                    offset_range[1] + callee_range[1],
                                )

                        if binding.get("root_kind") == "parameter":
                            root = binding.get("root")
                            if callee_observed:
                                caller_observations = required_observations.setdefault(
                                    caller_name, set()
                                )
                                if root not in caller_observations:
                                    caller_observations.add(root)
                                    changed = True
                            if callee_range is None and not callee_unprovable:
                                continue
                            if composed_range is None:
                                function_unprovable = unprovable.setdefault(
                                    caller_name, set()
                                )
                                if root not in function_unprovable:
                                    function_unprovable.add(root)
                                    changed = True
                            elif merge_range(caller_name, root, composed_range):
                                changed = True
                            continue

                        if binding.get("root_kind") == "concrete":
                            if binding.get("selected_null") and callee_observed:
                                self.hlsl_workgroup_pointer_null_observation_error(
                                    function_name=callee_name,
                                    parameter_name=parameter_name,
                                    node=call["node"],
                                )
                            if callee_range is None and not callee_unprovable:
                                continue
                            self.hlsl_workgroup_pointer_access_error(
                                function_name=callee_name,
                                parameter_name=parameter_name,
                                binding=binding,
                                access_range=composed_range,
                                node=call["node"],
                            )

        return required_ranges

    def hlsl_workgroup_pointer_error(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        reason=None,
        node=None,
    ):
        return DirectXWorkgroupPointerError(
            message,
            function_name=function_name,
            parameter_name=parameter_name,
            reason=reason,
            source_location=getattr(node, "source_location", None),
        )

    def hlsl_resolved_function_call_name(self, node, available_names):
        if not isinstance(node, FunctionCallNode):
            return None
        name = self.function_call_name(node)
        if not name:
            return None
        arguments = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
        specialized_name = generic_function_call_name(self, name, arguments)
        candidates = [
            specialized_name,
            self.hlsl_materialized_function_name(specialized_name),
            self.hlsl_materialized_function_name(name),
            name,
        ]
        return next(
            (candidate for candidate in candidates if candidate in available_names),
            None,
        )

    def hlsl_resolved_called_function_names(self, function, available_names):
        return {
            called_name
            for node in self.walk_ast(getattr(function, "body", []))
            if (
                called_name := self.hlsl_resolved_function_call_name(
                    node, available_names
                )
            )
            is not None
        }

    def hlsl_reachable_function_names(self, ast, functions_by_name, target_stage=None):
        all_entry_functions = []
        selected_entry_functions = []
        stage_entry_types = self.stage_entry_types()

        for function in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(function)
            if stage_name not in stage_entry_types:
                continue
            all_entry_functions.append(function)
            if stage_matches(target_stage, stage_name):
                selected_entry_functions.append(function)

        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            all_entry_functions.append(entry_point)
            if stage_matches(target_stage, normalize_stage_name(stage_type)):
                selected_entry_functions.append(entry_point)

        if not all_entry_functions:
            return set(functions_by_name)

        available_names = set(functions_by_name)
        reachable_names = set()
        visited_ids = set()
        pending = list(selected_entry_functions)
        while pending:
            function = pending.pop()
            if function is None or id(function) in visited_ids:
                continue
            visited_ids.add(id(function))
            function_name = getattr(function, "name", None)
            if function_name:
                reachable_names.add(function_name)
            for called_name in self.hlsl_resolved_called_function_names(
                function, available_names
            ):
                reachable_names.add(called_name)
                overloads = self.hlsl_function_overloads_by_name.get(called_name)
                if overloads:
                    pending.extend(overloads)
                    continue
                callee = functions_by_name.get(called_name)
                if callee is not None:
                    pending.append(callee)

        return reachable_names

    def hlsl_function_has_workgroup_pointer_declaration(self, function):
        parameters = (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        if any(
            self.hlsl_workgroup_pointer_declaration(parameter)
            for parameter in parameters
        ):
            return True
        return any(
            isinstance(node, VariableNode)
            and self.hlsl_workgroup_pointer_declaration(node)
            for node in self.walk_ast(getattr(function, "body", []))
        )

    def hlsl_unreachable_workgroup_pointer_function_names(
        self, functions_by_name, reachable_names
    ):
        available_names = set(functions_by_name)
        calls_by_caller = {
            function_name: self.hlsl_resolved_called_function_names(
                function, available_names
            )
            for function_name, function in functions_by_name.items()
        }
        pointer_parameter_function_names = {
            function_name
            for function_name, function in functions_by_name.items()
            if any(
                self.hlsl_workgroup_pointer_declaration(parameter)
                for parameter in (
                    getattr(
                        function,
                        "parameters",
                        getattr(function, "params", []),
                    )
                    or []
                )
            )
        }
        pointer_sensitive_names = {
            function_name
            for function_name, function in functions_by_name.items()
            if self.hlsl_function_has_workgroup_pointer_declaration(function)
            or calls_by_caller[function_name] & pointer_parameter_function_names
        }
        omitted_names = pointer_sensitive_names - reachable_names

        changed = True
        while changed:
            changed = False
            for function_name, called_names in calls_by_caller.items():
                if function_name in reachable_names or function_name in omitted_names:
                    continue
                if called_names & omitted_names:
                    omitted_names.add(function_name)
                    changed = True

        return omitted_names

    def collect_hlsl_workgroup_pointer_variants(self, ast, target_stage=None):
        functions = list(self.collect_functions(ast))
        functions.extend(getattr(self, "generic_function_specializations", {}).values())
        all_functions_by_name = {}
        all_function_overloads_by_name = {}
        for function in functions:
            function_name = getattr(function, "name", None)
            if not function_name or generic_function_parameters(function):
                continue
            all_functions_by_name[function_name] = function
            all_function_overloads_by_name.setdefault(function_name, []).append(
                function
            )

        reachable_names = self.hlsl_reachable_function_names(
            ast, all_functions_by_name, target_stage
        )
        self.hlsl_omitted_function_names = (
            self.hlsl_unreachable_workgroup_pointer_function_names(
                all_functions_by_name, reachable_names
            )
        )
        functions_by_name = {
            function_name: function
            for function_name, function in all_functions_by_name.items()
            if function_name in reachable_names
        }

        def workgroup_pointer_signature(function):
            parameters = list(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            return tuple(
                (
                    index,
                    parameter.name,
                    self.hlsl_workgroup_pointer_element_type(parameter),
                )
                for index, parameter in enumerate(parameters)
                if self.hlsl_workgroup_pointer_declaration(parameter)
            )

        compatible_overloads_by_name = {}
        pointer_parameters = {}
        pointer_parameter_indices = {}
        pointer_base_names = {}
        for function_name, function in functions_by_name.items():
            signature = workgroup_pointer_signature(function)
            compatible_overloads = [
                overload
                for overload in all_function_overloads_by_name[function_name]
                if workgroup_pointer_signature(overload) == signature
            ]
            compatible_overloads_by_name[function_name] = compatible_overloads
            parameters = list(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            workgroup_parameters = [
                parameter
                for parameter in parameters
                if self.hlsl_workgroup_pointer_declaration(parameter)
            ]
            if not workgroup_parameters:
                continue
            pointer_parameters[function_name] = workgroup_parameters
            pointer_parameter_indices[function_name] = {
                index: parameter.name
                for index, parameter in enumerate(parameters)
                if self.hlsl_workgroup_pointer_declaration(parameter)
            }
            used_names = {getattr(parameter, "name", None) for parameter in parameters}
            for overload in compatible_overloads:
                used_names.update(self.collect_hlsl_function_identifier_names(overload))
            base_names = {}
            for parameter in workgroup_parameters:
                base_name = self.hlsl_unique_local_identifier(
                    f"{parameter.name}_offset", used_names
                )
                used_names.add(base_name)
                base_names[parameter.name] = base_name
            pointer_base_names[function_name] = base_names

        self.function_hlsl_workgroup_pointer_parameters = pointer_parameters
        self.function_hlsl_workgroup_pointer_parameter_indices = (
            pointer_parameter_indices
        )
        self.function_hlsl_workgroup_pointer_base_names = pointer_base_names
        if not pointer_parameters and not any(
            isinstance(node, AssignmentNode)
            and self.hlsl_workgroup_record_transfer(node.target, node.value) is not None
            for function in functions_by_name.values()
            for node in self.walk_ast(getattr(function, "body", []))
        ):
            return

        calls_by_caller = {function_name: [] for function_name in functions_by_name}
        entry_intervals = self.hlsl_workgroup_pointer_entry_intervals(ast)
        direct_ranges = {}
        direct_observations = {}
        for function_name, overloads in compatible_overloads_by_name.items():
            merged_ranges = {}
            merged_observations = set()
            for function in overloads:
                overload_ranges, overload_observations = (
                    self.hlsl_analyze_workgroup_pointer_function(
                        function,
                        pointer_parameters,
                        pointer_parameter_indices,
                        calls_by_caller[function_name],
                        entry_intervals.get(function_name, {}),
                    )
                )
                for parameter_name, access_range in overload_ranges.items():
                    current = merged_ranges.get(parameter_name)
                    merged_ranges[parameter_name] = (
                        access_range
                        if current is None
                        else (
                            min(current[0], access_range[0]),
                            max(current[1], access_range[1]),
                        )
                    )
                merged_observations.update(overload_observations)
            direct_ranges[function_name] = merged_ranges
            direct_observations[function_name] = merged_observations

        self.hlsl_validate_workgroup_pointer_access_ranges(
            calls_by_caller, direct_ranges, direct_observations
        )

        variants = {function_name: set() for function_name in pointer_parameters}
        parameter_positions = {
            function_name: {
                parameter.name: index for index, parameter in enumerate(parameters)
            }
            for function_name, parameters in pointer_parameters.items()
        }
        changed = True
        while changed:
            changed = False
            for caller_name, calls in calls_by_caller.items():
                caller_parameters = pointer_parameters.get(caller_name, [])
                caller_variants = variants.get(caller_name, {()})
                if caller_parameters and not caller_variants:
                    continue
                for caller_variant in tuple(caller_variants):
                    for call in calls:
                        callee_variant = []
                        for parameter in pointer_parameters[call["callee"]]:
                            binding = call["bindings"][parameter.name]
                            if binding.get("root_kind") == "null":
                                root, extent = HLSL_NULL_WORKGROUP_POINTER_CONTRACT
                            elif binding.get("root_kind") == "concrete":
                                root = binding.get("root")
                                extent = binding.get("array_size")
                            elif binding.get("root_kind") == "parameter":
                                position = parameter_positions.get(caller_name, {}).get(
                                    binding.get("root")
                                )
                                contract = (
                                    caller_variant[position]
                                    if position is not None
                                    and position < len(caller_variant)
                                    else None
                                )
                                if contract is None:
                                    root = None
                                    extent = None
                                else:
                                    root, extent = contract
                            else:
                                root = None
                                extent = None
                            if (
                                root,
                                extent,
                            ) != HLSL_NULL_WORKGROUP_POINTER_CONTRACT and (
                                not isinstance(root, str)
                                or not root
                                or not isinstance(extent, int)
                                or extent <= 0
                            ):
                                raise self.hlsl_workgroup_pointer_error(
                                    "DirectX workgroup pointer argument for "
                                    f"'{call['callee']}.{parameter.name}' has no "
                                    "concrete shared-array extent",
                                    function_name=call["callee"],
                                    parameter_name=parameter.name,
                                    reason="missing-concrete-size",
                                    node=call["node"],
                                )
                            callee_variant.append((root, extent))
                        callee_variant = tuple(callee_variant)
                        if callee_variant not in variants[call["callee"]]:
                            variants[call["callee"]].add(callee_variant)
                            if len(variants[call["callee"]]) > 64:
                                raise self.hlsl_workgroup_pointer_error(
                                    "DirectX workgroup pointer specialization "
                                    f"for '{call['callee']}' exceeds 64 concrete "
                                    "array contracts",
                                    function_name=call["callee"],
                                    reason="variant-limit-exceeded",
                                    node=call["node"],
                                )
                            changed = True

        for function_name, parameters in pointer_parameters.items():
            if variants[function_name]:
                continue
            parameter = parameters[0]
            raise self.hlsl_workgroup_pointer_error(
                "DirectX workgroup pointer parameter "
                f"'{function_name}.{parameter.name}' has no call with a concrete "
                "shared-array backing object",
                function_name=function_name,
                parameter_name=parameter.name,
                reason="missing-concrete-size",
                node=parameter,
            )

        self.function_hlsl_workgroup_pointer_backing_variants = {
            function_name: tuple(sorted(function_variants))
            for function_name, function_variants in variants.items()
        }
        self.register_hlsl_mapped_workgroup_pointer_variants(functions)

    def register_hlsl_mapped_workgroup_pointer_variants(self, functions):
        """Expose analyzed pointer contracts under mapped overload names."""

        for function in functions:
            source_name = getattr(function, "name", None)
            target_name = self.hlsl_function_declaration_name(function)
            if not source_name or not target_name or source_name == target_name:
                continue
            variants = self.function_hlsl_workgroup_pointer_backing_variants.get(
                source_name
            )
            if variants is None:
                continue

            parameters = list(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            workgroup_parameters = [
                parameter
                for parameter in parameters
                if self.hlsl_workgroup_pointer_declaration(parameter)
            ]
            if not workgroup_parameters:
                continue

            self.function_hlsl_workgroup_pointer_parameters[target_name] = (
                workgroup_parameters
            )
            self.function_hlsl_workgroup_pointer_parameter_indices[target_name] = {
                index: parameter.name
                for index, parameter in enumerate(parameters)
                if self.hlsl_workgroup_pointer_declaration(parameter)
            }
            self.function_hlsl_workgroup_pointer_backing_variants[target_name] = (
                variants
            )

    def collect_hlsl_private_pointer_fixed_array_parameters(self, functions_by_name):
        candidates = {}
        pointer_callees = {}
        selected = set()
        for function_name, function in functions_by_name.items():
            indexed = {}
            pointers = {}
            for index, parameter in enumerate(
                getattr(function, "parameters", []) or []
            ):
                if is_private_pointer_parameter(parameter):
                    pointers[index] = parameter
                if not self.hlsl_private_pointer_fixed_array_candidate(parameter):
                    continue
                indexed[index] = parameter
            if indexed:
                candidates[function_name] = indexed
            if pointers:
                pointer_callees[function_name] = pointers

        def resolve_callee(call, arguments, callees=candidates):
            raw_name = self.function_call_name(call)
            specialized_name = generic_function_call_name(self, raw_name, arguments)
            materialized_name = self.hlsl_materialized_function_name(raw_name)
            for candidate in (specialized_name, materialized_name, raw_name):
                if candidate in callees:
                    return candidate
            return None

        def candidate_root(expression, caller_parameters):
            name = self.expression_name(expression)
            if name in caller_parameters:
                return name
            if isinstance(expression, UnaryOpNode) and expression.op == "&":
                return candidate_root(expression.operand, caller_parameters)
            if isinstance(expression, ArrayAccessNode):
                return candidate_root(expression.array, caller_parameters)
            if isinstance(expression, PointerReinterpretNode):
                return candidate_root(
                    getattr(expression, "expression", None), caller_parameters
                )
            if isinstance(expression, BinaryOpNode) and expression.op in {"+", "-"}:
                return candidate_root(
                    expression.left, caller_parameters
                ) or candidate_root(expression.right, caller_parameters)
            return None

        def argument_requires_view(expression):
            return (
                isinstance(expression, UnaryOpNode)
                and expression.op == "&"
                or isinstance(expression, PointerReinterpretNode)
                or isinstance(expression, BinaryOpNode)
                and expression.op in {"+", "-"}
            )

        def direct_parameter_name(expression, parameter_names):
            if not isinstance(expression, (str, IdentifierNode, VariableNode)):
                return None
            name = self.expression_name(expression)
            return name if name in parameter_names else None

        def bare_parameter_error(function_name, parameter_name, node):
            raise DirectXPrivatePointerParameterError(
                "DirectX cannot emit fixed private array parameter "
                "decay as a first-class pointer expression",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="bare-pointer-expression",
                source_location=getattr(node, "source_location", None),
            )

        calls = []
        for caller_name, function in functions_by_name.items():
            caller_parameters = {
                parameter.name for parameter in candidates.get(caller_name, {}).values()
            }
            for node in self.walk_ast(getattr(function, "body", [])):
                if isinstance(node, UnaryOpNode):
                    name = direct_parameter_name(node.operand, caller_parameters)
                    if name is not None:
                        if node.op in {"++", "--"}:
                            raise DirectXPrivatePointerParameterError(
                                "DirectX cannot lower fixed private array parameter "
                                f"pointer operator '{node.op}' for "
                                f"'{caller_name}.{name}'",
                                function_name=caller_name,
                                parameter_name=name,
                                reason="assignment-operator-unsupported",
                                source_location=getattr(node, "source_location", None),
                            )
                        if node.op == "*":
                            selected.add((caller_name, name))
                        else:
                            bare_parameter_error(caller_name, name, node)
                elif isinstance(node, AssignmentNode):
                    operator = getattr(node, "operator", "=")
                    target = getattr(node, "target", None) or getattr(
                        node, "left", None
                    )
                    assigned = getattr(node, "value", None) or getattr(
                        node, "right", None
                    )
                    target_name = direct_parameter_name(target, caller_parameters)
                    assigned_name = direct_parameter_name(assigned, caller_parameters)
                    if target_name is not None and operator in {"=", "+=", "-="}:
                        raise DirectXPrivatePointerParameterError(
                            "DirectX cannot lower fixed private array parameter "
                            f"pointer assignment for '{caller_name}.{target_name}'",
                            function_name=caller_name,
                            parameter_name=target_name,
                            reason="assignment-operator-unsupported",
                            source_location=getattr(node, "source_location", None),
                        )
                    if assigned_name is not None:
                        bare_parameter_error(caller_name, assigned_name, node)
                elif isinstance(node, VariableNode):
                    initial_value = getattr(
                        node, "initial_value", getattr(node, "value", None)
                    )
                    name = direct_parameter_name(initial_value, caller_parameters)
                    if name is not None:
                        bare_parameter_error(caller_name, name, node)
                elif isinstance(node, BinaryOpNode):
                    direct_names = [
                        direct_parameter_name(operand, caller_parameters)
                        for operand in (node.left, node.right)
                    ]
                    if node.op in {"+", "-"}:
                        for name in direct_names:
                            if name is not None:
                                selected.add((caller_name, name))
                    else:
                        for name in direct_names:
                            if name is not None:
                                bare_parameter_error(caller_name, name, node)
                elif isinstance(node, ReturnNode):
                    name = direct_parameter_name(
                        getattr(node, "value", None), caller_parameters
                    )
                    if name is not None:
                        bare_parameter_error(caller_name, name, node)
                elif isinstance(node, TernaryOpNode):
                    for expression in (
                        node.condition,
                        node.true_expr,
                        node.false_expr,
                    ):
                        name = direct_parameter_name(expression, caller_parameters)
                        if name is not None:
                            bare_parameter_error(caller_name, name, node)
                if not isinstance(node, FunctionCallNode):
                    continue
                arguments = list(getattr(node, "arguments", []) or [])
                # A fixed array parameter that roots a view passed to a private
                # pointer parameter backs that view, so it is lowered as one.
                pointer_callee = resolve_callee(node, arguments, pointer_callees)
                if caller_parameters and pointer_callee is not None:
                    for index in pointer_callees[pointer_callee]:
                        if index >= len(arguments):
                            continue
                        root_name = candidate_root(arguments[index], caller_parameters)
                        if root_name is not None:
                            selected.add((caller_name, root_name))
                callee_name = resolve_callee(node, arguments)
                if callee_name is None:
                    continue
                for index, parameter in candidates[callee_name].items():
                    if index >= len(arguments):
                        continue
                    argument = arguments[index]
                    root_name = candidate_root(argument, caller_parameters)
                    calls.append((caller_name, root_name, callee_name, parameter.name))
                    if argument_requires_view(argument):
                        selected.add((callee_name, parameter.name))
                        if root_name is not None:
                            selected.add((caller_name, root_name))

        changed = True
        while changed:
            changed = False
            for caller_name, root_name, callee_name, parameter_name in calls:
                if root_name is None:
                    continue
                caller_key = (caller_name, root_name)
                callee_key = (callee_name, parameter_name)
                if callee_key in selected and caller_key not in selected:
                    selected.add(caller_key)
                    changed = True
                if caller_key in selected and callee_key not in selected:
                    selected.add(callee_key)
                    changed = True
        return selected

    def hlsl_private_pointer_callee_variant(
        self, caller_variant, call, pointer_parameters, scalar_parameters
    ):
        """The backing variant ``call`` binds from one variant of its caller."""

        caller_positions = {
            parameter.name: index
            for index, parameter in enumerate(
                pointer_parameters.get(call.get("caller"), ())
            )
        }
        values = []
        for parameter in pointer_parameters[call["callee"]]:
            binding = call["bindings"][parameter.name]
            root_kind = binding.get("root_kind")
            position = caller_positions.get(binding.get("root"))
            if root_kind == "local":
                values.append(binding["extent"])
            elif root_kind == "local_word_view":
                values.append(binding["word_view"])
            elif root_kind == "scalar" and parameter.name in scalar_parameters.get(
                call["callee"], set()
            ):
                values.append(1)
            elif (
                root_kind == "parameter"
                and position is not None
                and position < len(caller_variant)
            ):
                values.append(caller_variant[position])
            else:
                return None
        return tuple(values)

    def hlsl_private_pointer_unrecorded_callees(self, functions_by_name, view_calls):
        """Helpers with a call the private pointer analysis did not record."""

        recorded = {id(call["node"]) for calls in view_calls.values() for call in calls}
        callees = set()
        for function in functions_by_name.values():
            for node in self.walk_ast(getattr(function, "body", [])):
                if not isinstance(node, FunctionCallNode) or id(node) in recorded:
                    continue
                callee_name = self.hlsl_private_pointer_call_callee(node)
                if callee_name in self.function_private_pointer_parameters:
                    callees.add(callee_name)
        return sorted(callees)

    def hlsl_private_pointer_emitted_variants(
        self,
        functions_by_name,
        pointer_parameters,
        scalar_parameters,
        required_spans,
        view_calls,
        variants,
    ):
        """The backing variants to emit, refusing indistinguishable overloads."""

        reached = hlsl_private_pointer_variants.reached_variants(
            variants,
            view_calls,
            pointer_parameters,
            lambda caller_variant, call: self.hlsl_private_pointer_callee_variant(
                caller_variant, call, pointer_parameters, scalar_parameters
            ),
        )
        for function_name in self.hlsl_private_pointer_unrecorded_callees(
            functions_by_name, view_calls
        ):
            default = tuple(
                required_spans[function_name][parameter.name]
                for parameter in pointer_parameters[function_name]
            )
            variants[function_name].add(default)
            reached[function_name].add(default)

        previous_variant = self.current_hlsl_private_pointer_variant
        emitted = {}
        try:
            for function_name in sorted(variants):
                parameters = pointer_parameters[function_name]

                def signature(variant):
                    self.current_hlsl_private_pointer_variant = {
                        parameter.name: variant[index]
                        for index, parameter in enumerate(parameters)
                    }
                    return tuple(
                        self.hlsl_private_pointer_parameter_array_type(
                            parameter, function_name
                        )
                        for parameter in parameters
                    )

                if len(variants[function_name]) < 2:
                    emitted[function_name] = set(variants[function_name])
                    continue
                emitted[function_name], collisions = (
                    hlsl_private_pointer_variants.emitted_variants(
                        variants[function_name], reached[function_name], signature
                    )
                )
                for first, second in collisions:
                    index = next(
                        index
                        for index in range(len(parameters))
                        if first[index] != second[index]
                    )
                    parameter = parameters[index]
                    raise DirectXPrivatePointerParameterError(
                        "DirectX private pointer parameter "
                        f"'{function_name}.{parameter.name}' is bound to backings "
                        f"{first[index]!r} and {second[index]!r}, which both "
                        f"become '{signature(first)[index]}' and cannot be "
                        "distinct HLSL overloads",
                        function_name=function_name,
                        parameter_name=parameter.name,
                        reason="indistinguishable-backing-variants",
                        source_location=getattr(
                            functions_by_name.get(function_name),
                            "source_location",
                            None,
                        ),
                    )
        finally:
            self.current_hlsl_private_pointer_variant = previous_variant
        return emitted

    def hlsl_private_pointer_local_array_extents(self, function):
        constants = self.visible_literal_int_constants(function)
        extents = {}
        ambiguous = set()
        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, (VariableNode, ArrayNode)):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            node_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if isinstance(node, ArrayNode):
                size_expression = get_array_size_from_node(node)
            elif "ArrayType" in node_type.__class__.__name__:
                size_expression = getattr(node_type, "size", None)
            else:
                continue
            extent = self.literal_int_value(size_expression, constants)
            if extent is None or extent <= 0:
                ambiguous.add(name)
                continue
            if name in extents and extents[name] != extent:
                ambiguous.add(name)
                continue
            extents[name] = extent
        for name in ambiguous:
            extents.pop(name, None)
        return extents

    def hlsl_private_pointer_local_backing_types(self, function):
        local_types = {}
        ambiguous = set()
        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, (VariableNode, ArrayNode)):
                continue
            name = getattr(node, "name", None)
            node_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if not name or node_type is None or name in ambiguous:
                continue
            existing = local_types.get(name)
            if existing is not None and self.type_name_string(
                existing
            ) != self.type_name_string(node_type):
                local_types.pop(name, None)
                ambiguous.add(name)
                continue
            local_types[name] = node_type
        return local_types

    # Integer conversions a private pointer offset may apply.
    HLSL_PRIVATE_POINTER_CAST_CALLS = frozenset(
        {
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
    )
    # Integer builtins a storage pointer array offset may also call: each is
    # pure, so the offset still only snapshots its operands.
    HLSL_RESOURCE_POINTER_ARRAY_PURE_CALLS = frozenset({"min", "max", "clamp", "abs"})
    HLSL_SIDE_EFFECTING_UNARY_OPERATORS = frozenset(
        {
            "++",
            "--",
            "PRE_INCREMENT",
            "PRE_DECREMENT",
            "POST_INCREMENT",
            "POST_DECREMENT",
        }
    )

    def hlsl_resource_pointer_array_offset_has_side_effects(self, expression):
        """Return whether a storage pointer array offset may have side effects.

        Beyond the private pointer predicate's integer conversions, the pure
        integer builtins ``min``, ``max``, ``clamp`` and ``abs`` are allowed
        unless a user function shadows them.  Call arguments are inspected
        too, and atomics, assignments, increments and any other call count as
        side effects.
        """

        if expression is None or isinstance(expression, (str, int, float, bool)):
            return False
        if isinstance(expression, (list, tuple)):
            return any(
                self.hlsl_resource_pointer_array_offset_has_side_effects(item)
                for item in expression
            )
        if isinstance(expression, (AssignmentNode, AtomicOpNode)):
            return True
        if isinstance(expression, UnaryOpNode) and (
            expression.op in self.HLSL_SIDE_EFFECTING_UNARY_OPERATORS
            or self.map_operator(expression.op)
            in self.HLSL_SIDE_EFFECTING_UNARY_OPERATORS
        ):
            return True
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            pure = (
                function_name in self.HLSL_RESOURCE_POINTER_ARRAY_PURE_CALLS
                and function_name not in self.function_return_types
            )
            if not pure and function_name not in self.HLSL_PRIVATE_POINTER_CAST_CALLS:
                return True
            return self.hlsl_resource_pointer_array_offset_has_side_effects(
                getattr(expression, "arguments", None)
            )
        if not hasattr(expression, "__dict__"):
            return False
        return any(
            self.hlsl_resource_pointer_array_offset_has_side_effects(child)
            for key, child in vars(expression).items()
            if key not in {"parent", "annotations", "array", "index", "args", "name"}
        )

    def hlsl_private_pointer_expression_has_side_effects(self, expression):
        if expression is None or isinstance(expression, (str, int, float, bool)):
            return False
        if isinstance(expression, AssignmentNode):
            return True
        if isinstance(expression, UnaryOpNode) and expression.op in {"++", "--"}:
            return True
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            if function_name not in self.HLSL_PRIVATE_POINTER_CAST_CALLS:
                return True
        if not hasattr(expression, "__dict__"):
            return False
        return any(
            self.hlsl_private_pointer_expression_has_side_effects(child)
            for key, child in vars(expression).items()
            if key not in {"parent", "annotations", "array", "index", "args", "name"}
        )

    def hlsl_private_pointer_condition_uses_unsafe_intervals(
        self, expression, intervals, constants
    ):
        safe_names = self.current_hlsl_private_pointer_condition_interval_names
        for node in self.walk_ast(expression):
            value = self.literal_int_value(node, constants)
            if isinstance(value, _UnsignedLiteralInt):
                return True
            if isinstance(node, (IdentifierNode, VariableNode)):
                name = getattr(node, "name", None)
                if name in intervals and name not in safe_names:
                    return True
        return False
