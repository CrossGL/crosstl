"""Function intervals in the GLSL generator: top-level helpers that depend on stage locals, fragment-only functions, index and component intervals, and deferred storage pointer aliases."""

from types import SimpleNamespace

from ..ast import (
    ArrayNode,
    AssignmentNode,
    BlockNode,
    ConstructorNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MemberAccessNode,
    RangeNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)
from ..dispatch_region_contract import codegen_dispatch_region
from ..metadata_preconditions import validate_codegen_metadata_preconditions
from ..refusals import OpenGLRefusal
from ..validation import expression_debug_name
from .array_utils import evaluate_literal_int_expression
from .generic_function_utils import generic_function_parameters
from .glsl_exact_lookups import glsl_call_names_resolving_to
from .glsl_host_workgroup_size import glsl_workgroup_proof_local_size
from .image_access_contracts import collect_function_parameter_names
from .loop_induction import (
    metadata_loop_bindings,
    metadata_parameter_bindings,
)
from .opengl_values import OpenGLStoragePointerError
from .scalar_return_bounds import vector_return_intervals
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
)


class OpenGLFunctionIntervalMixin:
    """Function interval members of ``GLSLCodeGen``."""

    def top_level_helpers_depending_on_stage_locals(self, functions, local_functions):
        required_names = {
            getattr(func, "name", None)
            for func in local_functions
            if getattr(func, "name", None)
        }
        if not required_names:
            return []

        deferred_ids = set()
        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name or id(func) in deferred_ids:
                    continue
                if function_stage_name(func):
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    if self.function_call_name(node) not in required_names:
                        continue
                    deferred_ids.add(id(func))
                    required_names.add(func_name)
                    changed = True
                    break

        return [func for func in functions if id(func) in deferred_ids]

    def collect_fragment_only_function_requirements(self, functions):
        """Mark helpers whose call graph requires fragment-stage GLSL builtins."""
        functions_by_name = {}
        duplicate_names = set()
        for func in functions:
            name = getattr(func, "name", None)
            if not name:
                continue
            if name in functions_by_name:
                duplicate_names.add(name)
                continue
            functions_by_name[name] = func
        for name in duplicate_names:
            functions_by_name.pop(name, None)

        requirements = {
            name: self.function_body_uses_fragment_only_call(func)
            for name, func in functions_by_name.items()
        }

        changed = True
        while changed:
            changed = False
            for name, func in functions_by_name.items():
                if requirements.get(name):
                    continue
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(node)
                    if requirements.get(callee_name):
                        requirements[name] = True
                        changed = True
                        break

        return requirements

    def function_body_uses_fragment_only_call(self, func):
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            if self.is_fragment_only_call_name(self.function_call_name(node)):
                return True
        return False

    def is_fragment_only_call_name(self, func_name):
        if not func_name:
            return False
        if func_name == "clip" and func_name not in self.function_return_types:
            return True
        return func_name in glsl_call_names_resolving_to(
            self,
            "fragment-only",
            self.GLSL_DERIVATIVE_FUNCTIONS | self.GLSL_INTERPOLATION_FUNCTIONS,
        )

    def validate_fragment_only_helper_call(self, func_name):
        if (
            not func_name
            or not self.function_fragment_only_requirements.get(func_name)
            or self.current_stage_entry_type in (None, "fragment")
        ):
            return
        raise OpenGLRefusal(
            f"OpenGL helper function '{func_name}' uses fragment-only operations "
            "and is only valid in fragment stages"
        )

    def glsl_workgroup_proof_initial_intervals(self, function):
        intervals = dict(
            getattr(function, "_glsl_workgroup_proof_parameter_intervals", {}) or {}
        )
        for name in getattr(self, "glsl_entry_pointer_parameters", {}).get(
            id(function), ()
        ):
            intervals[name] = (0, 0)
        for name, binding in getattr(
            function, "_glsl_storage_pointer_aliases", {}
        ).items():
            if binding.get("kind") == "storage-pointer-array":
                for index, bounds in enumerate(
                    binding.get("source_offset_intervals", ())
                ):
                    if bounds is not None:
                        intervals[self.glsl_pointer_array_interval_key(name, index)] = (
                            bounds
                        )
                continue
            offset = binding.get("offset")
            if isinstance(offset, str) and offset in intervals:
                intervals[name] = intervals[offset]
        for name, record in (
            getattr(self, "glsl_metadata_entry_bounds", {})
            .get(id(function), {})
            .items()
        ):
            intervals[name] = (record["minimum"], record["maximum"])
        for name, record in getattr(self, "glsl_metadata_element_entry_bounds", {}).get(
            id(function), ()
        ):
            intervals[self.glsl_metadata_element_key(name, record["element"])] = (
                record["minimum"],
                record["maximum"],
            )
        constants = self.initial_literal_int_constants(function)
        for name, value in constants.items():
            literal = self.literal_int_value(value, constants)
            if literal is not None:
                intervals[name] = (literal, literal)
        parameter_names = {
            getattr(parameter, "name", None)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        }
        for name, values in self.literal_int_vector_constants.items():
            if name in parameter_names:
                continue
            for component_index, value in enumerate(values):
                key = self.glsl_index_component_interval_key(name, component_index)
                if key is not None:
                    intervals[key] = (value, value)

        builtin_intervals = {}
        subgroup_width = getattr(
            function,
            "_glsl_workgroup_proof_subgroup_width",
            None,
        )
        if (
            isinstance(subgroup_width, int)
            and not isinstance(subgroup_width, bool)
            and subgroup_width > 0
        ):
            builtin_intervals.update(
                {
                    "gl_SubgroupInvocationID": (0, subgroup_width - 1),
                    "gl_SubgroupSize": (subgroup_width, subgroup_width),
                }
            )

        local_size = getattr(function, "_glsl_workgroup_proof_local_size", None)
        invocation_count = None
        if local_size is not None:
            resolved_local_size = tuple(
                evaluate_literal_int_expression(component, constants)
                for component in local_size
            )
            if all(
                isinstance(component, int)
                and not isinstance(component, bool)
                and component > 0
                for component in resolved_local_size
            ):
                invocation_count = (
                    resolved_local_size[0]
                    * resolved_local_size[1]
                    * resolved_local_size[2]
                )
                builtin_intervals["gl_LocalInvocationIndex"] = (
                    0,
                    invocation_count - 1,
                )
                quadgroup_count = (invocation_count + 3) // 4
                builtin_intervals.update(
                    {
                        "gl_QuadGroupID": (0, quadgroup_count - 1),
                        "gl_QuadGroupInvocationID": (
                            0,
                            min(3, invocation_count - 1),
                        ),
                    }
                )
                for component_index, component_size in enumerate(resolved_local_size):
                    invocation_key = self.glsl_index_component_interval_key(
                        "gl_LocalInvocationID",
                        component_index,
                    )
                    workgroup_size_key = self.glsl_index_component_interval_key(
                        "gl_WorkGroupSize",
                        component_index,
                    )
                    if invocation_key is not None:
                        builtin_intervals[invocation_key] = (
                            0,
                            component_size - 1,
                        )
                    if workgroup_size_key is not None:
                        builtin_intervals[workgroup_size_key] = (
                            component_size,
                            component_size,
                        )
                if subgroup_width is not None:
                    subgroup_count = (
                        invocation_count + subgroup_width - 1
                    ) // subgroup_width
                    builtin_intervals.update(
                        {
                            "gl_SubgroupID": (0, subgroup_count - 1),
                            "gl_NumSubgroups": (subgroup_count, subgroup_count),
                        }
                    )
                elif invocation_count == 1:
                    # A one-invocation workgroup necessarily occupies exactly
                    # one subgroup regardless of the hardware subgroup width.
                    builtin_intervals.update(
                        {
                            "gl_SubgroupID": (0, 0),
                            "gl_NumSubgroups": (1, 1),
                        }
                    )

        region = getattr(self, "glsl_dispatch_entry_regions", {}).get(id(function))
        if region is not None:
            for component, (count, width) in enumerate(
                zip(region.workgroup_count, region.workgroup_size)
            ):
                for builtin, interval in (
                    ("gl_WorkGroupID", (0, count - 1)),
                    ("gl_NumWorkGroups", (count, count)),
                    ("gl_GlobalInvocationID", (0, count * width - 1)),
                ):
                    key = self.glsl_index_component_interval_key(builtin, component)
                    builtin_intervals[key] = interval
        intervals.update(builtin_intervals)
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            semantic = self.semantic_from_node(parameter)
            if semantic is None:
                continue
            mapped_semantic = self.map_semantic(semantic)
            interval = builtin_intervals.get(mapped_semantic)
            if interval is not None:
                intervals[parameter.name] = interval
            for component_index in range(4):
                source_key = self.glsl_index_component_interval_key(
                    mapped_semantic,
                    component_index,
                )
                target_key = self.glsl_index_component_interval_key(
                    parameter.name,
                    component_index,
                )
                component_interval = builtin_intervals.get(source_key)
                if target_key is not None and component_interval is not None:
                    intervals[target_key] = component_interval
        return intervals

    def glsl_declared_struct_member_interval_key(self, expression):
        if not isinstance(expression, MemberAccessNode):
            return None
        owner_type = self.type_name_string(
            self.glsl_source_expression_type(expression.object)
        )
        if str(expression.member) in self.glsl_struct_member_types(owner_type):
            return expression_debug_name(expression)
        return None

    def glsl_index_component_index(self, member):
        return {
            "x": 0,
            "r": 0,
            "s": 0,
            "y": 1,
            "g": 1,
            "t": 1,
            "z": 2,
            "b": 2,
            "p": 2,
            "w": 3,
            "a": 3,
            "q": 3,
        }.get(str(member))

    def glsl_index_component_interval_key(self, expression, component_index):
        name = (
            expression
            if isinstance(expression, str)
            else expression_debug_name(expression)
        )
        if not name or component_index not in range(4):
            return None
        return f"{name}.{'xyzw'[component_index]}"

    def glsl_clear_index_component_intervals(self, intervals, expression):
        for component_index in range(4):
            key = self.glsl_index_component_interval_key(
                expression,
                component_index,
            )
            if key is not None:
                intervals.pop(key, None)

    def glsl_static_integer_vector_component_intervals(
        self,
        expression,
        vector_type,
        intervals,
        constants,
    ):
        info = self.glsl_value_type_info(vector_type)
        if info is None or info["family"] not in {"int", "uint"} or info["width"] <= 1:
            return None
        width = info["width"]

        returned = vector_return_intervals(
            self, expression, vector_type, intervals, constants
        )
        if returned is not None:
            return returned

        literal = self.glsl_integer_vector_literal_value(
            expression,
            int_constants=constants,
            vector_constants={},
            vector_array_constants={},
        )
        if literal is not None and len(literal) == width:
            return tuple((value, value) for value in literal)

        source_name = expression_debug_name(expression)
        inherited = tuple(
            intervals.get(self.glsl_index_component_interval_key(source_name, index))
            for index in range(width)
        )
        if all(interval is not None for interval in inherited):
            return inherited

        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            arguments = list(expression.arguments or [])
        elif isinstance(expression, ConstructorNode):
            function_name = self.type_name_string(expression.constructor_type)
            arguments = list(expression.arguments or [])
        else:
            return None
        constructor_info = self.glsl_value_type_info(function_name)
        if constructor_info is None or constructor_info["width"] != width:
            return None

        if len(arguments) == 1:
            scalar = self.glsl_private_pointer_interval(
                arguments[0], intervals, constants
            )
            if scalar is not None:
                return (scalar,) * width
            source_name = expression_debug_name(arguments[0])
            inherited = tuple(
                intervals.get(
                    self.glsl_index_component_interval_key(source_name, index)
                )
                for index in range(width)
            )
            return inherited if all(item is not None for item in inherited) else None

        if len(arguments) != width:
            return None
        components = tuple(
            self.glsl_private_pointer_interval(argument, intervals, constants)
            for argument in arguments
        )
        return components if all(item is not None for item in components) else None

    def glsl_for_in_pattern_interval(
        self,
        node,
        intervals,
        constants,
        interval_evaluator=None,
    ):
        iterable = getattr(node, "iterable", None)
        evaluate = interval_evaluator or self.glsl_private_pointer_interval
        repeated_bound = None
        if isinstance(iterable, RangeNode):
            repeated_bound = getattr(iterable, "end", None)
        else:
            iterable_type = self.glsl_source_expression_type(iterable)
            if iterable_type is None or self.is_scalar_integer_type(iterable_type):
                # Pre-emission analysis can lose a same-named outer scalar's
                # type when the for-in pattern introduces a new lexical
                # binding.  An unresolved type remains safe here: only the
                # integer interval evaluator below can admit it as a count.
                repeated_bound = iterable

        # Range ends and scalar counts are emitted directly in the GLSL loop
        # condition, so source semantics re-evaluate them before every
        # iteration. An interval captured before the loop is therefore sound
        # only while the body cannot mutate any dependency. The pattern is
        # the generated loop controller for these forms and must likewise
        # remain owned by the loop regardless of where a mutation appears in
        # the body. Treat address-taking and out/inout forwarding as mutation
        # so proof never depends on statement order or a stale initial value.
        mutation_keys = self.glsl_function_mutated_interval_keys(
            SimpleNamespace(
                body=[
                    repeated_bound,
                    getattr(node, "body", None),
                ]
            )
        )
        pattern = getattr(node, "pattern", None)
        bound_dependencies = self.glsl_interval_expression_keys(repeated_bound)
        if self.glsl_interval_key_sets_overlap(
            {pattern} if pattern else set(),
            mutation_keys,
        ) or self.glsl_interval_key_sets_overlap(
            bound_dependencies,
            mutation_keys,
        ):
            return None

        if isinstance(iterable, RangeNode):
            start = evaluate(getattr(iterable, "start", None), intervals, constants)
            end = evaluate(getattr(iterable, "end", None), intervals, constants)
            if start and end:
                maximum = end[1] - (0 if getattr(iterable, "inclusive", False) else 1)
                if start[0] <= maximum:
                    return (start[0], maximum)
            return None

        if repeated_bound is iterable:
            count = evaluate(iterable, intervals, constants)
            if count and count[1] > 0:
                return (0, count[1] - 1)
        return None

    def glsl_control_flow_signed_constants(self, expressions, intervals, lexical):
        """Use exact flow facts only for visible, non-volatile signed scalars."""
        cast_names = {
            id(node.function)
            for node in self.walk_ast(expressions)
            if isinstance(node, FunctionCallNode)
            and self.glsl_software_subgroup_proof_int_type(
                self.function_call_name(node)
            )
        }
        constants = {}
        for node in self.walk_ast(expressions):
            if not isinstance(node, IdentifierNode) or id(node) in cast_names:
                continue
            binding = lexical["declarations"].get(
                lexical["reference_bindings"].get(id(node))
            )
            if binding is None or not self.glsl_software_subgroup_proof_int_type(
                binding["source_type"]
            ):
                return None
            if "volatile" in {
                str(qualifier).lower()
                for qualifier in getattr(binding["node"], "qualifiers", ())
            }:
                return None
            interval = intervals.get(node.name)
            if interval is not None and interval[0] == interval[1]:
                constants[node.name] = interval[0]
        return constants

    def glsl_control_flow_zero_trip_for(self, node, lexical):
        initializer = getattr(node, "init", None)
        if not self.glsl_software_subgroup_integral_variable(initializer):
            return False
        if getattr(initializer, "qualifiers", None):
            return False
        constants = self.glsl_control_flow_signed_constants(
            [initializer.initial_value, node.condition],
            getattr(node, "_glsl_control_flow_intervals", {}),
            lexical,
        )
        return (
            constants is not None
            and self.glsl_software_subgroup_proven_zero_trip_for(node, constants)
        )

    def glsl_control_flow_static_if_body(self, node, lexical):
        conditions = [node.condition, *(node.else_if_conditions or [])]
        bodies = [node.if_body, *(node.else_if_bodies or [])]
        for condition, body in zip(conditions, bodies):
            constants = self.glsl_control_flow_signed_constants(
                condition, getattr(node, "_glsl_control_flow_intervals", {}), lexical
            )
            value = (
                None
                if constants is None
                else self.glsl_software_subgroup_safe_int_value(condition, constants)
            )
            if value is None:
                return False, None
            if value:
                return True, body
        return True, node.else_body

    def prepare_glsl_resource_function_specializations(self, ast, target_stage=None):
        """Create GLSL helper variants for concrete storage image arguments."""
        self.glsl_metadata_entry_bounds = {}
        self.glsl_metadata_element_entry_bounds = {}
        entries = self.glsl_software_subgroup_stage_entries(ast, target_stage)
        region = codegen_dispatch_region(ast, entries)
        self.glsl_dispatch_entry_regions = (
            {id(entries[0][1]): region} if region is not None else {}
        )
        self.glsl_entry_pointer_parameters = {
            id(function): {
                parameter.name
                for parameter in (
                    getattr(function, "parameters", getattr(function, "params", []))
                    or []
                )
                if self.glsl_stage_entry_storage_pointer_binding(
                    parameter, parameter.name
                )
                is not None
            }
            for _stage, function, _execution in entries
        }
        if self.metadata_preconditions:
            records = validate_codegen_metadata_preconditions(
                entries, self.metadata_preconditions, getattr(ast, "structs", ())
            )
            function = entries[0][1]
            self.glsl_metadata_entry_bounds[id(function)] = metadata_loop_bindings(
                function,
                records,
                self.glsl_function_mutated_interval_names(function),
                self.walk_ast,
            )
            self.glsl_metadata_element_entry_bounds[id(function)] = [
                (name, record)
                for name, record in metadata_parameter_bindings(
                    function,
                    records,
                    self.glsl_function_mutated_interval_names(function),
                    self.walk_ast,
                )
                if "element" in record
            ]
        self.glsl_resource_function_specializations = {}
        self.glsl_resource_function_call_names = {}
        self.glsl_resource_specialized_source_names = set()
        self.glsl_resource_function_call_plans = {}
        self.glsl_resource_function_semantic_call_plans = {}
        pending = []

        for stage_name, stage in getattr(ast, "stages", {}).items():
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                entry_point._glsl_workgroup_proof_local_size = (
                    glsl_workgroup_proof_local_size(
                        getattr(stage, "execution_config", None)
                    )
                )
                entry_point._glsl_workgroup_proof_subgroup_width = (
                    self.software_subgroup_width
                    if self.software_subgroup_width is not None
                    else self.glsl_exact_subgroup_width(
                        entry_point,
                        normalize_stage_name(stage_name),
                    )
                )

        def scan_function(func, aliases):
            # A resource-specialized caller may have inherited plans when its
            # source function was deep-copied.  Only plans proved while scanning
            # this exact clone may authorize emission-time cache reuse.
            for node in self.walk_ast(getattr(func, "body", [])):
                if isinstance(node, FunctionCallNode) and hasattr(
                    node, "_glsl_resource_specialization_plan"
                ):
                    delattr(node, "_glsl_resource_specialization_plan")
            # Discover addressed local scalars before interval annotation so a
            # mutating one-element helper is already modeled as ``inout`` at
            # every call site, including serial-loop proof analysis.
            lexical_bindings = self.glsl_scalar_array_view_lexical_bindings(func)
            self.glsl_scalar_array_view_locals(func, lexical=lexical_bindings)
            self.annotate_glsl_control_flow_intervals(func)
            aliases = self.glsl_function_resource_pointer_aliases(func, aliases)
            self.prepare_glsl_storage_pointer_array_aliases(func, aliases)
            self.prepare_glsl_deferred_storage_pointer_aliases(func, aliases)
            specialized_workgroup_parameters = set(
                (getattr(func, "_glsl_workgroup_pointer_aliases", {}) or {}).keys()
            )
            has_unbound_workgroup_parameter = any(
                self.glsl_workgroup_pointer_element_type(parameter) is not None
                and getattr(parameter, "name", None)
                not in specialized_workgroup_parameters
                for parameter in (
                    getattr(func, "parameters", getattr(func, "params", [])) or []
                )
            )
            if has_unbound_workgroup_parameter:
                # A source helper with an abstract workgroup pointer cannot yet
                # choose concrete nested resource specializations or prove their
                # view ranges.  Its concrete clone is queued from the caller and
                # scanned below with the backing and scalar proof intervals bound.
                return
            previous_source_types = getattr(
                self,
                "glsl_resource_specialization_source_types",
                None,
            )
            self.glsl_resource_specialization_source_types = (
                self.glsl_storage_pointer_array_mutation_source_types(func)
            )
            try:
                before = set(self.glsl_resource_function_specializations)
                self.collect_glsl_resource_specializations_from_body(
                    getattr(func, "body", []),
                    aliases,
                    lexical_call_source_types=lexical_bindings["call_source_types"],
                    semantic_call_locations=(
                        self.glsl_resource_function_call_semantic_locations(func)
                    ),
                )
                # The specialization mapping preserves lexical discovery order.  A
                # set difference here made nested overloads claim unsuffixed and
                # numbered target names according to hash iteration order, so the
                # same source emitted different artifacts across Python processes.
                pending.extend(
                    key
                    for key in self.glsl_resource_function_specializations
                    if key not in before
                )
            finally:
                self.glsl_resource_specialization_source_types = previous_source_types

        global_pointer_aliases = {
            **self.glsl_global_workgroup_pointer_aliases,
            **self.glsl_global_storage_pointer_aliases,
        }
        for func in self.collect_functions(ast):
            if generic_function_parameters(func):
                continue
            scan_function(func, global_pointer_aliases)
        for func in self.generic_function_specializations.values():
            scan_function(func, global_pointer_aliases)

        seen = set()
        while pending:
            key = pending.pop(0)
            if key in seen:
                continue
            seen.add(key)
            specialized_func = self.glsl_resource_function_specializations.get(key)
            if specialized_func is None:
                continue
            specialization_aliases = {
                **global_pointer_aliases,
                **(getattr(specialized_func, "_glsl_resource_aliases", {}) or {}),
                **(
                    getattr(
                        specialized_func,
                        "_glsl_workgroup_pointer_aliases",
                        {},
                    )
                    or {}
                ),
                **(
                    getattr(
                        specialized_func,
                        "_glsl_storage_pointer_aliases",
                        {},
                    )
                    or {}
                ),
            }
            scan_function(
                specialized_func,
                specialization_aliases,
            )

        if not self.glsl_resource_function_specializations:
            return

        specialized_functions = list(
            self.glsl_resource_function_specializations.values()
        )
        self.function_definitions.update(
            {
                func.name: func
                for func in specialized_functions
                if getattr(func, "name", None)
            }
        )
        self.function_return_types.update(
            {
                func.name: self.type_name_string(getattr(func, "return_type", "void"))
                for func in specialized_functions
                if getattr(func, "name", None)
            }
        )
        self.function_parameter_names.update(
            collect_function_parameter_names(specialized_functions)
        )
        self.function_parameter_types.update(
            self.collect_function_parameter_types(specialized_functions)
        )
        self.function_parameter_infos.update(
            self.collect_function_parameter_infos(specialized_functions)
        )
        self.function_sampler_parameter_indices.update(
            self.collect_function_sampler_parameter_indices(specialized_functions)
        )

    def glsl_function_workgroup_pointer_aliases(self, function, aliases):
        visible_aliases = {name: dict(binding) for name, binding in aliases.items()}
        specialized_aliases = set(
            (getattr(function, "_glsl_workgroup_pointer_aliases", {}) or {}).keys()
        )
        parameters = getattr(
            function,
            "parameters",
            getattr(function, "params", []),
        )
        for parameter in parameters or []:
            name = getattr(parameter, "name", None)
            if name and name not in specialized_aliases:
                visible_aliases.pop(name, None)
        return visible_aliases

    def glsl_function_resource_pointer_aliases(self, function, aliases):
        visible_aliases = self.glsl_function_workgroup_pointer_aliases(
            function, aliases
        )
        visible_aliases.update(
            {
                name: dict(binding)
                for name, binding in self.stage_entry_storage_pointer_bindings.get(
                    id(function), {}
                ).items()
            }
        )
        return visible_aliases

    @staticmethod
    def glsl_deferred_storage_pointer_binding_signature(binding):
        return (
            binding.get("root"),
            binding.get("element_type"),
            binding.get("source_element_type"),
            binding.get("view_element_type"),
            binding.get("access"),
        )

    def validate_glsl_deferred_storage_pointer_assignment(
        self,
        function,
        declarations,
    ):
        """Require every deferred storage-pointer use to be definitely assigned."""

        deferred_names = frozenset(declarations)
        function_name = getattr(function, "name", None)

        def fail(name, node):
            raise OpenGLStoragePointerError(
                "OpenGL deferred storage pointer "
                f"'{name}' may be used before every control-flow path assigns "
                "a concrete same-backing view",
                function_name=function_name,
                parameter_name=name,
                reason="deferred-use-before-assignment",
                source_location=getattr(node, "source_location", None),
            )

        def require_expression(value, assigned):
            if value is None:
                return
            if isinstance(value, str):
                if value in deferred_names and value not in assigned:
                    fail(value, value)
                return
            for node in self.walk_ast(value):
                if not isinstance(node, (IdentifierNode, VariableNode)):
                    continue
                name = getattr(node, "name", None)
                if name in deferred_names and name not in assigned:
                    fail(name, node)

        def visit_sequence(value, assigned):
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            for statement in statements:
                visit(statement, assigned)
            return assigned

        def visit(value, assigned):
            if value is None or isinstance(value, (int, float, bool)):
                return assigned
            if isinstance(value, (list, tuple)):
                return visit_sequence(value, assigned)
            if isinstance(value, BlockNode):
                return visit_sequence(value, assigned)
            if isinstance(value, (VariableNode, ArrayNode)):
                name = getattr(value, "name", None)
                if name in deferred_names:
                    assigned.discard(name)
                    return assigned
                require_expression(getattr(value, "initial_value", None), assigned)
                return assigned
            if isinstance(value, ExpressionStatementNode):
                return visit(getattr(value, "expression", None), assigned)
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                right = getattr(value, "value", getattr(value, "right", None))
                operator = self.map_operator(
                    getattr(value, "operator", getattr(value, "op", "="))
                )
                target_name = (
                    self.expression_name(target)
                    if isinstance(target, (str, IdentifierNode, VariableNode))
                    else None
                )
                require_expression(right, assigned)
                if target_name in deferred_names:
                    if operator != "=" and target_name not in assigned:
                        fail(target_name, target)
                    if operator == "=":
                        assigned.add(target_name)
                    return assigned
                require_expression(target, assigned)
                return assigned
            if isinstance(value, IfNode):
                condition = getattr(
                    value,
                    "condition",
                    getattr(value, "if_condition", None),
                )
                require_expression(condition, assigned)
                static_known, static_body = self.glsl_static_if_body(value)
                if static_known:
                    return visit_sequence(static_body, assigned)

                branch_states = []
                then_state = set(assigned)
                visit_sequence(value.if_body, then_state)
                branch_states.append(then_state)
                for else_if_condition, else_if_body in zip(
                    getattr(value, "else_if_conditions", []) or [],
                    getattr(value, "else_if_bodies", []) or [],
                ):
                    require_expression(else_if_condition, assigned)
                    branch_state = set(assigned)
                    visit_sequence(else_if_body, branch_state)
                    branch_states.append(branch_state)
                else_body = getattr(value, "else_body", None)
                if else_body is None:
                    branch_states.append(set(assigned))
                else:
                    else_state = set(assigned)
                    visit_sequence(else_body, else_state)
                    branch_states.append(else_state)
                merged = set.intersection(*branch_states) if branch_states else set()
                assigned.clear()
                assigned.update(merged)
                return assigned
            if isinstance(value, ForNode):
                visit(getattr(value, "init", None), assigned)
                require_expression(getattr(value, "condition", None), assigned)
                loop_state = set(assigned)
                visit_sequence(getattr(value, "body", None), loop_state)
                visit(getattr(value, "update", None), loop_state)
                return assigned
            if isinstance(value, ForInNode):
                require_expression(getattr(value, "iterable", None), assigned)
                visit_sequence(getattr(value, "body", None), set(assigned))
                return assigned
            if isinstance(value, DoWhileNode):
                body_state = set(assigned)
                visit_sequence(getattr(value, "body", None), body_state)
                require_expression(getattr(value, "condition", None), body_state)
                assigned.intersection_update(body_state)
                return assigned
            if isinstance(value, (WhileNode, LoopNode)):
                require_expression(getattr(value, "condition", None), assigned)
                visit_sequence(getattr(value, "body", None), set(assigned))
                return assigned
            if isinstance(value, SwitchNode):
                require_expression(getattr(value, "expression", None), assigned)
                for case in getattr(value, "cases", []) or []:
                    require_expression(getattr(case, "value", None), assigned)
                    visit_sequence(
                        getattr(case, "statements", None),
                        set(assigned),
                    )
                default_case = getattr(value, "default_case", None)
                visit_sequence(
                    getattr(default_case, "statements", default_case),
                    set(assigned),
                )
                return assigned

            require_expression(value, assigned)
            return assigned

        visit_sequence(getattr(function, "body", []), set())

    def prepare_glsl_deferred_storage_pointer_aliases(self, function, aliases):
        """Bind uninitialized local storage pointers when every assignment agrees."""

        body = getattr(function, "body", [])
        nodes = list(self.walk_ast(body))
        declaration_counts = {}
        candidates = {}
        for node in nodes:
            if not isinstance(node, (VariableNode, ArrayNode)):
                continue
            name = getattr(node, "name", None)
            if name:
                declaration_counts[name] = declaration_counts.get(name, 0) + 1
            if hasattr(node, "_glsl_deferred_storage_pointer_binding"):
                delattr(node, "_glsl_deferred_storage_pointer_binding")
            if (
                name
                and getattr(node, "initial_value", None) is None
                and self.glsl_storage_pointer_pointee_type(node) is not None
            ):
                candidates[name] = node

        candidates = {
            name: declaration
            for name, declaration in candidates.items()
            if declaration_counts.get(name) == 1
        }
        if not candidates:
            return

        assignments = {name: [] for name in candidates}
        unresolved = {}
        for node in nodes:
            if not isinstance(node, AssignmentNode):
                continue
            target = getattr(node, "target", getattr(node, "left", None))
            if not isinstance(target, (str, IdentifierNode, VariableNode)):
                continue
            name = self.expression_name(target)
            if name not in candidates:
                continue
            operator = self.map_operator(
                getattr(node, "operator", getattr(node, "op", "="))
            )
            if operator != "=":
                continue
            right = getattr(node, "value", getattr(node, "right", None))
            binding = self.glsl_storage_pointer_binding(right, aliases)
            if binding is None:
                unresolved[name] = node
            else:
                assignments[name].append((node, binding))

        prepared = {}
        for name, declaration in candidates.items():
            concrete = assignments[name]
            if not concrete:
                continue
            if name in unresolved:
                node = unresolved[name]
                raise OpenGLStoragePointerError(
                    "OpenGL deferred storage pointer "
                    f"'{name}' has an assignment without a concrete "
                    "storage-buffer backing object",
                    function_name=getattr(function, "name", None),
                    parameter_name=name,
                    reason="deferred-assignment-backing-unresolved",
                    source_location=getattr(node, "source_location", None),
                )
            first = concrete[0][1]
            first_signature = self.glsl_deferred_storage_pointer_binding_signature(
                first
            )
            for node, binding in concrete:
                if binding.get("pointer_reinterpretation") is not None:
                    raise OpenGLStoragePointerError(
                        "OpenGL deferred storage pointer "
                        f"'{name}' cannot change typed storage views before "
                        "initialization",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="deferred-reinterpretation-unsupported",
                        source_location=getattr(node, "source_location", None),
                    )
                if (
                    self.glsl_deferred_storage_pointer_binding_signature(binding)
                    != first_signature
                ):
                    raise OpenGLStoragePointerError(
                        "OpenGL deferred storage pointer "
                        f"'{name}' does not resolve to one concrete same-backing "
                        "storage view",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="deferred-backing-mismatch",
                        source_location=getattr(node, "source_location", None),
                    )

            declared_element_type = self.glsl_storage_pointer_element_type(declaration)
            if (
                declared_element_type is not None
                and first.get("element_type") is not None
                and declared_element_type != first.get("element_type")
            ):
                raise OpenGLStoragePointerError(
                    "OpenGL deferred storage pointer "
                    f"'{name}' changes element type from "
                    f"{first.get('element_type')} to {declared_element_type}",
                    function_name=getattr(function, "name", None),
                    parameter_name=name,
                    reason="element-type-mismatch",
                    source_location=getattr(declaration, "source_location", None),
                )
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(declaration, "qualifiers", []) or []
            }
            required_access = (
                "read"
                if qualifiers.intersection({"const", "constant", "readonly", "in"})
                else "read_write"
            )
            self.validate_glsl_storage_pointer_binding_access(
                first,
                required_access,
                declaration,
            )
            deferred_binding = {
                **first,
                "offset": 0,
                "byte_offset": 0,
                "element_type": declared_element_type or first.get("element_type"),
                "access": required_access,
                "resource_root": False,
                "deferred_initialization": True,
            }
            declaration._glsl_deferred_storage_pointer_binding = deferred_binding
            prepared[name] = declaration

        if prepared:
            self.validate_glsl_deferred_storage_pointer_assignment(
                function,
                prepared,
            )
