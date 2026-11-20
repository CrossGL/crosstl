"""Inline bindings and pointer parameters in the SPIR-V generator: pointer-preserving inline bindings, inline return values, unused and skipped parameters, called functions, stage object arguments, storage image and resource array parameters, uniform block members, storage buffer pointer metadata and offset aliases, pointer dereferences, block member lengths, and structured buffer defaults and dimensions."""

from typing import (
    Optional,
    Tuple,
)

from ..ast import (
    ArrayLiteralNode,
    ConstantNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    ReturnNode,
    VariableNode,
)
from ..refusals import VulkanRefusal
from .array_utils import evaluate_literal_int_expression
from .buffer_length import BufferLengthUnsupportedError
from .spirv_swizzle_reference_arguments import prepare_swizzle_reference_binding
from .spirv_value_result_arguments import (
    prepare_value_result_binding,
    value_result_qualifier,
)
from .spirv_values import (
    InlineReturnContext,
    PointerOffsetState,
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)


class SpirvPointerParameterMixin:
    """Pointer parameter members of ``VulkanSPIRVCodeGen``."""

    def prepare_pointer_preserving_inline_bindings(
        self,
        function_node,
        call_args,
        generic_type_substitutions,
    ):
        """Evaluate actual arguments under caller scope before binding parameters."""
        func_name = getattr(function_node, "name", "unknown")
        prepared = []
        for index, param, arg in self.storage_buffer_effective_call_bindings(
            function_node, call_args
        ):
            param_name = getattr(param, "name", f"param{index}")
            if (
                not self.function_parameter_has_reachable_use(function_node, param_name)
                and value_result_qualifier(self, param) != "inout"
            ):
                if not self.side_effect_free_call_argument(arg):
                    self.process_call_argument(func_name, arg, index)
                continue

            storage_buffer_type_name = self.storage_buffer_parameter_type_name(param)
            if storage_buffer_type_name is not None:
                storage_buffer_type_name = self.substitute_generic_signature_type(
                    storage_buffer_type_name,
                    generic_type_substitutions,
                )
                if self.expression_contains_empty_array_literal(arg):
                    raise self.unsupported_call_initializer_binding_error(
                        func_name, index, arg
                    )
                if self.pointer_ternary_statically_selects_null(arg):
                    raise self.unsupported_null_pointer_argument_error(arg)
                pointer_arg = self.variable_pointer_from_expression(arg)
                if pointer_arg is None:
                    self.refuse_lowering(
                        f"function call '{func_name}' requires a "
                        f"storage buffer argument for parameter {param_name}"
                    )
                    return None
                actual_type_name = self.storage_buffer_expression_type_name(arg)
                if (
                    actual_type_name is not None
                    and not self.storage_buffer_parameter_type_is_compatible(
                        storage_buffer_type_name, actual_type_name
                    )
                ):
                    self.refuse_lowering(
                        f"function call '{func_name}' requires "
                        f"{storage_buffer_type_name} storage buffer type for "
                        "argument "
                        f"{self.expression_debug_name(arg)} "
                        f"passed to parameter {param_name}: got {actual_type_name}"
                    )
                    return None
                prepared.append(
                    {
                        "kind": "storage_buffer",
                        "name": param_name,
                        "value": pointer_arg,
                        "copy_pointer": not self.function_parameter_is_reference_like(
                            param
                        ),
                    }
                )
                continue

            param_type_name = self.substitute_generic_signature_type(
                self.function_parameter_type_name(param),
                generic_type_substitutions,
            )
            if self.function_parameter_requires_pointer_alias(param):
                if self.expression_contains_empty_array_literal(arg):
                    raise self.unsupported_call_initializer_binding_error(
                        func_name, index, arg
                    )
                param_value_type = (
                    self.map_crossgl_type(param_type_name)
                    if param_type_name is not None
                    else None
                )
                swizzle_binding = prepare_swizzle_reference_binding(
                    self, func_name, index, param, param_name, arg, param_value_type
                )
                if swizzle_binding is not None:
                    prepared.append(swizzle_binding)
                    continue
                pointer_arg = self.assignable_pointer_from_expression(arg)
                source_value_type = (
                    self.pointer_pointee_type(pointer_arg)
                    if pointer_arg is not None
                    else None
                )
                if pointer_arg is None or source_value_type is None:
                    raise self.unsupported_reference_argument_error(
                        func_name,
                        index,
                        param_name,
                        arg,
                        feature="non-addressable-reference-argument",
                        reason="the argument is not addressable",
                    )
                if param_value_type is not None and (
                    source_value_type.id != param_value_type.id
                    and not self.aggregate_types_are_layout_compatible(
                        source_value_type, param_value_type
                    )
                ):
                    raise self.unsupported_reference_argument_error(
                        func_name,
                        index,
                        param_name,
                        arg,
                        feature="incompatible-reference-argument",
                        reason=(
                            f"argument type {source_value_type.type.base_type} "
                            f"is incompatible with {param_type_name}"
                        ),
                    )
                prepared.append(
                    prepare_value_result_binding(
                        self,
                        param,
                        param_name,
                        pointer_arg,
                        source_value_type,
                        param_value_type,
                    )
                    or {
                        "kind": "reference",
                        "name": param_name,
                        "value": pointer_arg,
                        "copy_pointer": False,
                    }
                )
                continue

            expected_type = (
                self.map_crossgl_type(param_type_name)
                if param_type_name is not None
                else None
            )
            arg_value = self.process_call_argument(
                func_name,
                arg,
                index,
                expected_type=expected_type,
            )
            if arg_value is None:
                self.refuse_lowering(f"Failed to evaluate argument for {func_name}")
                return None
            if expected_type is not None:
                arg_value = self.convert_value_to_type(arg_value, expected_type)
            prepared.append(
                {
                    "kind": "value",
                    "name": param_name,
                    "value": arg_value,
                    "copy_pointer": False,
                }
            )
        return prepared

    def inline_function_body(self, function_node) -> Optional[SpirvId]:
        body = getattr(function_node, "body", [])
        statements = (
            body.statements
            if hasattr(body, "statements")
            else body if isinstance(body, list) else [body]
        )

        has_tail_expression = any(
            getattr(node, "is_tail_expression", False)
            for node in self.walk_ast_nodes(statements)
        )
        has_top_level_value_return = any(
            isinstance(statement, ReturnNode)
            and getattr(statement, "value", None) is not None
            for statement in statements
        )
        non_void_fallthrough = (
            self.current_return_type is not None
            and self.current_return_type.type.base_type != "void"
            and not has_top_level_value_return
        )
        if (
            self.nested_inline_return_node(function_node) is not None
            or has_tail_expression
            or non_void_fallthrough
        ):
            return self.inline_function_body_with_return_state(statements)

        for stmt in statements:
            if isinstance(stmt, ReturnNode):
                if getattr(stmt, "value", None) is None:
                    return None
                if isinstance(stmt.value, ArrayLiteralNode):
                    return self.process_array_literal(
                        stmt.value, self.current_return_type
                    )
                if isinstance(stmt.value, MatchNode):
                    return self.process_match_expression_return(stmt.value)
                return self.process_expression(stmt.value)

            self.process_statement(stmt)

        return None

    def inline_function_body_with_return_state(self, statements) -> Optional[SpirvId]:
        """Lower inline returns through invocation-local state."""
        return_type = self.current_return_type
        bool_type = self.register_primitive_type("bool")
        returned_pointer = self.create_variable(
            bool_type, "Function", "__inline_returned"
        )
        self.store_to_variable(
            returned_pointer, self.register_constant(False, bool_type)
        )
        result_pointer = None

        if return_type is not None and return_type.type.base_type != "void":
            result_pointer = self.create_variable(
                return_type, "Function", "__inline_return_value"
            )

        context = InlineReturnContext(
            return_type,
            returned_pointer,
            result_pointer,
            len(self.loop_merge_labels),
        )
        self.inline_return_context_stack.append(context)
        try:
            self.process_statements(statements)
        finally:
            self.inline_return_context_stack.pop()

        if result_pointer is None:
            return None
        self.require_inline_return_value(context)
        return self.load_from_variable(result_pointer, return_type)

    def require_inline_return_value(self, context: InlineReturnContext):
        """Terminate a non-void inline fallthrough path as unreachable."""
        bool_type = self.register_primitive_type("bool")
        returned = self.load_from_variable(context.returned_pointer, bool_type)
        value_label = SpirvId(self.get_id(), SpirvType("label"))
        unreachable_label = SpirvId(self.get_id(), SpirvType("label"))
        merge_label = SpirvId(self.get_id(), SpirvType("label"))

        self.create_selection_merge(merge_label)
        self.create_conditional_branch(returned, value_label, unreachable_label)

        self.emit(f"%{value_label.id} = OpLabel")
        self.current_label = value_label.id
        self.create_branch(merge_label)

        self.emit(f"%{unreachable_label.id} = OpLabel")
        self.current_label = unreachable_label.id
        self.create_unreachable()

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id

    def collect_function_execution_models(self, ast, all_functions=None):
        ast_functions = self.collect_ast_functions(ast)
        functions = {id(func): func for func in ast_functions}

        stage_scopes = {}
        stage_function_ids = set()
        stage_entries = []
        for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
            scope = list(getattr(stage, "local_functions", []) or [])
            entry_function = getattr(stage, "entry_point", None)
            if entry_function is not None:
                scope.append(entry_function)
                stage_entries.append((stage_type, entry_function))
            scope = list({id(func): func for func in scope}.values())
            for func in scope:
                functions[id(func)] = func
                stage_function_ids.add(id(func))
                stage_scopes.setdefault(id(func), []).append(scope)

        global_functions = list(getattr(ast, "functions", []) or [])
        global_function_ids = {id(func) for func in global_functions}
        global_functions.extend(
            func
            for func in ast_functions
            if id(func) not in global_function_ids
            and id(func) not in stage_function_ids
        )
        global_functions = list({id(func): func for func in global_functions}.values())

        def functions_named(candidates, name):
            return [
                candidate
                for candidate in candidates
                if getattr(candidate, "name", None) == name
            ]

        def accepts_arity(candidate, argument_count):
            parameters = self.function_parameters(candidate)
            required_count = sum(
                getattr(param, "default_value", None) is None for param in parameters
            )
            return required_count <= argument_count <= len(parameters)

        def function_value_types(function_node):
            value_types = {
                param.name: self.function_parameter_type_name(param)
                for param in self.function_parameters(function_node)
                if getattr(param, "name", None)
            }
            for node in self.walk_ast_nodes(getattr(function_node, "body", [])):
                if isinstance(node, VariableNode):
                    value_types[node.name] = self.type_name_from_value(
                        getattr(node, "var_type", getattr(node, "vtype", None))
                    )
                elif isinstance(node, ConstantNode):
                    value_types[node.name] = self.type_name_from_value(
                        getattr(node, "const_type", None)
                    )
            return value_types

        def expression_type_name(expression, value_types):
            expression_type = getattr(expression, "expression_type", None)
            if expression_type is not None:
                return self.normalize_signature_type_name(expression_type)
            if isinstance(expression, LiteralNode):
                return self.normalize_signature_type_name(expression.literal_type)
            if isinstance(expression, IdentifierNode):
                return self.normalize_signature_type_name(
                    value_types.get(expression.name)
                )
            if isinstance(expression, ConstructorNode):
                return self.normalize_signature_type_name(expression.constructor_type)
            return None

        def matching_overloads(candidates, call, value_types):
            arguments = list(
                getattr(call, "arguments", getattr(call, "args", [])) or []
            )
            candidates = [
                candidate
                for candidate in candidates
                if accepts_arity(candidate, len(arguments))
            ]
            if len(candidates) <= 1:
                return candidates

            argument_types = [
                expression_type_name(argument, value_types) for argument in arguments
            ]
            scores = {}
            for candidate in candidates:
                parameter_types = [
                    self.normalize_signature_type_name(
                        self.function_parameter_type_name(param)
                    )
                    for param in self.function_parameters(candidate)
                ]
                scores[id(candidate)] = sum(
                    argument_type is not None
                    and parameter_type is not None
                    and argument_type == parameter_type
                    for argument_type, parameter_type in zip(
                        argument_types, parameter_types
                    )
                )
            best_score = max(scores.values())
            if best_score == 0:
                return candidates
            return [
                candidate
                for candidate in candidates
                if scores[id(candidate)] == best_score
            ]

        def call_targets(function_node, call, value_types):
            call_name = self.function_call_name(call)
            if call_name is None:
                return []

            scopes = stage_scopes.get(id(function_node))
            candidate_groups = []
            if scopes:
                candidate_groups.extend(scopes)
            candidate_groups.append(global_functions)
            for candidates in candidate_groups:
                matches = matching_overloads(
                    functions_named(candidates, call_name), call, value_types
                )
                if matches:
                    return matches
            return []

        calls_by_function = {}
        for function_id, func in functions.items():
            value_types = function_value_types(func)
            callees = set()
            for call in self.walk_ast_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                callees.update(
                    id(target) for target in call_targets(func, call, value_types)
                )
            calls_by_function[function_id] = callees

        execution_models = {function_id: set() for function_id in functions}

        def mark_callgraph(entry_function, execution_model: str):
            pending = [id(entry_function)]
            visited = set()
            while pending:
                function_id = pending.pop()
                if function_id in visited or function_id not in functions:
                    continue
                visited.add(function_id)
                execution_models[function_id].add(execution_model)
                pending.extend(calls_by_function.get(function_id, ()))

        stage_qualifiers = {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "object",
            "amplification",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_miss",
            "ray_any_hit",
            "ray_callable",
        }
        for func in getattr(ast, "functions", []):
            qualifier = self.get_function_qualifier(func)
            if func.name == "main" or qualifier in stage_qualifiers:
                mark_callgraph(func, self.spirv_execution_model(qualifier))

        for stage_type, entry_function in stage_entries:
            mark_callgraph(
                entry_function,
                self.spirv_execution_model(self.stage_key(stage_type)),
            )

        for function_node in all_functions or ():
            source_function_id = getattr(
                function_node, "_generic_source_function_id", None
            )
            if source_function_id not in execution_models:
                continue
            execution_models[id(function_node)] = set(
                execution_models[source_function_id]
            )

        return {
            function_id: models
            for function_id, models in execution_models.items()
            if models
        }

    def unused_array_parameter_indices(self, func):
        indices = set()
        body = getattr(func, "body", [])
        for index, param in enumerate(
            getattr(func, "parameters", getattr(func, "params", [])) or []
        ):
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            type_name = str(self.type_name_from_value(param_type) or "")
            if "[" not in type_name:
                continue
            if self.is_resource_type_name(self.array_base_type_name(type_name)):
                continue
            if not self.function_body_uses_identifier(body, param_name):
                indices.add(index)
        return indices

    def collect_unused_array_parameter_index_maps(self, functions):
        by_name_values = {}
        by_id = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            indices = self.unused_array_parameter_indices(func)
            by_id[id(func)] = indices
            by_name_values.setdefault(func_name, []).append(indices)

        by_name = {}
        for func_name, index_sets in by_name_values.items():
            if not index_sets:
                continue
            first_indices = index_sets[0]
            if not first_indices:
                continue
            if all(indices == first_indices for indices in index_sets[1:]):
                by_name[func_name] = set(first_indices)
        return by_name, by_id

    def function_body_uses_identifier(self, body, name):
        for node in self.walk_ast_nodes(body):
            if getattr(node, "name", None) == name:
                return True
        return False

    def skipped_function_parameter_indices(self, func_name):
        return self.spirv_skipped_function_parameter_indices.get(func_name, set())

    def skipped_function_parameter_indices_for_node(self, function_node):
        function_key = id(function_node)
        if function_key in self.spirv_skipped_function_parameter_indices_by_id:
            return self.spirv_skipped_function_parameter_indices_by_id[function_key]
        return self.skipped_function_parameter_indices(
            getattr(function_node, "name", None)
        )

    def collect_function_stage_object_dependencies(
        self, ast, target_stage, object_name
    ):
        direct_dependencies = {}
        function_calls = {}
        object_types = {}

        for func in self.collect_ast_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, None)
                function_calls.setdefault(
                    func_name, self.called_user_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = self.stage_key(stage_type)
            if target_stage is not None and stage_name != self.stage_key(target_stage):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if object_name == "output":
                object_type = getattr(entry_point, "return_type", None)
                if (
                    object_type is None
                    or self.type_name_from_value(object_type) == "void"
                ):
                    continue
            else:
                object_type = None
                for param in (
                    getattr(
                        entry_point, "parameters", getattr(entry_point, "params", [])
                    )
                    or []
                ):
                    if getattr(param, "name", None) == object_name:
                        object_type = getattr(
                            param, "param_type", getattr(param, "vtype", None)
                        )
                        break
                if object_type is None:
                    continue

            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                object_types[func_name] = object_type
                if self.direct_stage_object_dependency(func, object_name):
                    direct_dependencies[func_name] = object_type
                function_calls[func_name] = self.called_user_function_names(func)

        dependencies = dict(direct_dependencies)
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                if dependencies.get(func_name) is not None:
                    continue
                for called_name in calls:
                    if dependencies.get(called_name) is not None:
                        dependencies[func_name] = object_types.get(
                            func_name
                        ) or dependencies.get(called_name)
                        changed = True
                        break

        return {
            func_name: object_type
            for func_name, object_type in dependencies.items()
            if object_type is not None
        }

    def direct_stage_object_dependency(self, func, object_name):
        parameter_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
        }
        if object_name in parameter_names:
            return False
        local_names = set(parameter_names)
        for node in self.walk_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)
        if object_name in local_names:
            return False
        for node in self.walk_ast_nodes(getattr(func, "body", [])):
            if isinstance(node, MemberAccessNode):
                if self.expression_name(getattr(node, "object", None)) == object_name:
                    return True
            if getattr(node, "name", None) == object_name:
                return True
        return False

    def called_user_function_names(self, func):
        names = set()
        for node in self.walk_ast_nodes(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            name = self.function_call_name(node)
            if name:
                names.add(name)
        return names

    def required_function_stage_input_type(self, func_name):
        return self.function_stage_input_dependencies.get(func_name)

    def required_function_stage_output_type(self, func_name):
        return self.function_stage_output_dependencies.get(func_name)

    def required_function_stage_object_argument_names(self, func_name):
        names = []
        if self.required_function_stage_input_type(func_name) is not None:
            names.append("input")
        if self.required_function_stage_output_type(func_name) is not None:
            names.append("output")
        return names

    def stage_object_pointer_parameter(self, name, type_source):
        parameter = VariableNode(name, type_source)
        parameter._spirv_stage_object_pointer = True
        return parameter

    def collect_storage_image_pointer_parameters_for_functions(self, function_nodes):
        functions = {getattr(func, "name", None): func for func in function_nodes}
        functions = {name: func for name, func in functions.items() if name}
        parameter_names = {
            func_name: [
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            ]
            for func_name, func in functions.items()
        }
        parameter_name_sets = {
            func_name: {name for name in names if name}
            for func_name, names in parameter_names.items()
        }

        pointer_params = {func_name: set() for func_name in functions}
        for func_name, func in functions.items():
            for call in self.walk_ast_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                if (
                    self.function_call_name(call)
                    not in self.image_atomic_function_names()
                ):
                    continue
                args = getattr(call, "arguments", getattr(call, "args", []))
                if not args:
                    continue
                arg_name = self.expression_name(args[0])
                if arg_name in parameter_name_sets[func_name]:
                    pointer_params[func_name].add(arg_name)

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions.items():
                caller_params = parameter_name_sets[caller_name]
                for call in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    if callee_name not in functions:
                        continue
                    callee_pointer_params = pointer_params.get(callee_name, set())
                    if not callee_pointer_params:
                        continue

                    args = getattr(call, "arguments", getattr(call, "args", []))
                    callee_params = parameter_names.get(callee_name, [])
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            continue
                        if callee_params[index] not in callee_pointer_params:
                            continue
                        arg_name = self.expression_name(arg)
                        if arg_name not in caller_params:
                            continue
                        if arg_name not in pointer_params[caller_name]:
                            pointer_params[caller_name].add(arg_name)
                            changed = True

        return {
            func_name: params for func_name, params in pointer_params.items() if params
        }

    def collect_storage_image_pointer_parameters(self, ast):
        return self.collect_storage_image_pointer_parameters_for_functions(
            self.collect_ast_functions(ast)
        )

    def collect_stage_local_storage_image_pointer_metadata(self, ast):
        self.stage_local_function_storage_image_pointer_params = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            pointer_params = (
                self.collect_storage_image_pointer_parameters_for_functions(
                    visible_functions
                )
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_storage_image_pointer_params[key] = (
                    pointer_params.get(function_name, set())
                )

    def collect_resource_array_parameter_type_hints_for_functions(
        self, function_nodes, global_nodes
    ):
        functions = {getattr(func, "name", None): func for func in function_nodes}
        functions = {name: func for name, func in functions.items() if name}

        global_types = {}
        for node in self.walk_ast_nodes(global_nodes):
            if isinstance(node, VariableNode):
                global_types[node.name] = self.type_name_from_value(
                    getattr(node, "var_type", getattr(node, "vtype", "float"))
                )

        declared_param_types = {}
        for func_name, func in functions.items():
            declared_param_types[func_name] = {}
            for param in getattr(func, "parameters", getattr(func, "params", [])):
                param_name = getattr(param, "name", None)
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                if param_name and param_type is not None:
                    declared_param_types[func_name][param_name] = (
                        self.type_name_from_value(param_type)
                    )

        hints = {func_name: {} for func_name in functions}

        def visible_types(func_name):
            visible = dict(global_types)
            for param_name, param_type in declared_param_types.get(
                func_name, {}
            ).items():
                visible[param_name] = hints.get(func_name, {}).get(
                    param_name, param_type
                )
            return visible

        changed = True
        while changed:
            changed = False
            for caller_name, func in functions.items():
                caller_visible_types = visible_types(caller_name)
                for call in self.walk_ast_nodes(getattr(func, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(call)
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, arg in enumerate(args):
                        if index >= len(callee_params):
                            continue

                        param = callee_params[index]
                        param_name = getattr(param, "name", None)
                        declared_type = declared_param_types.get(callee_name, {}).get(
                            param_name
                        )
                        if not param_name or declared_type is None:
                            continue

                        arg_name = self.expression_name(arg)
                        arg_type = caller_visible_types.get(arg_name)
                        if arg_type is None:
                            continue

                        if not self.resource_array_param_type_is_compatible(
                            declared_type, arg_type
                        ):
                            raise VulkanRefusal(
                                "Incompatible SPIR-V resource array parameter shape "
                                f"for '{param_name}': expected {declared_type}, "
                                f"got {arg_type}"
                            )

                        fixed_type = self.fixed_type_for_unsized_resource_param(
                            declared_type, arg_type
                        )
                        if fixed_type is None:
                            continue

                        existing = hints.setdefault(callee_name, {}).get(param_name)
                        if existing is not None and existing != fixed_type:
                            raise VulkanRefusal(
                                "Conflicting SPIR-V resource array parameter sizes for "
                                f"'{param_name}': {existing} and {fixed_type}"
                            )
                        if existing != fixed_type:
                            hints[callee_name][param_name] = fixed_type
                            changed = True

        return {
            func_name: param_hints
            for func_name, param_hints in hints.items()
            if param_hints
        }

    def collect_resource_array_parameter_type_hints(self, ast):
        global_nodes = list(getattr(ast, "global_variables", []) or [])
        for stage in (getattr(ast, "stages", None) or {}).values():
            global_nodes.extend(getattr(stage, "local_variables", []) or [])

        return self.collect_resource_array_parameter_type_hints_for_functions(
            self.collect_ast_functions(ast),
            global_nodes,
        )

    def collect_stage_local_resource_array_parameter_type_hints(self, ast):
        self.stage_local_function_resource_array_type_hints = {}
        global_functions = list(getattr(ast, "functions", []) or [])

        for stage in (getattr(ast, "stages", None) or {}).values():
            local_functions = list(getattr(stage, "local_functions", []) or [])
            if not local_functions:
                continue

            local_names = {
                getattr(func, "name", None)
                for func in local_functions
                if getattr(func, "name", None)
            }
            visible_functions = [
                func
                for func in global_functions
                if getattr(func, "name", None) not in local_names
            ] + local_functions
            entry_function = getattr(stage, "entry_point", None)
            if entry_function is not None:
                visible_functions.append(entry_function)

            global_nodes = list(getattr(ast, "global_variables", []) or [])
            global_nodes.extend(getattr(stage, "local_variables", []) or [])
            hints = self.collect_resource_array_parameter_type_hints_for_functions(
                visible_functions,
                global_nodes,
            )

            for function_node in local_functions:
                function_name = getattr(function_node, "name", None)
                if not function_name:
                    continue
                key = self.stage_local_function_key(stage, function_name)
                if key is None:
                    continue
                self.stage_local_function_resource_array_type_hints[key] = hints.get(
                    function_name, {}
                )

    def split_outer_array_type(self, type_name: str):
        if not type_name or "[" not in type_name or not type_name.endswith("]"):
            return None

        open_bracket = type_name.find("[")
        close_bracket = type_name.find("]", open_bracket)
        if close_bracket == -1:
            return None

        base_type = type_name[:open_bracket]
        remaining_suffix = type_name[close_bracket + 1 :]
        element_type = (
            f"{base_type}{remaining_suffix}" if remaining_suffix else base_type
        )
        size_text = type_name[open_bracket + 1 : close_bracket].strip()
        if not size_text:
            return element_type, None
        if self.named_specialization_array_length(size_text) is not None:
            return element_type, size_text

        literal_size = evaluate_literal_int_expression(
            size_text, self.active_literal_int_constants()
        )
        return element_type, literal_size

    def array_base_type_name(self, type_name: str):
        if not type_name or "[" not in type_name:
            return type_name
        return type_name[: type_name.find("[")]

    def get_variable_value(self, variable_id: SpirvId) -> SpirvId:
        self.mark_interface_variable_if_needed(variable_id)
        wrapped_pointer = self.uniform_block_wrapped_member_pointer(variable_id)
        if wrapped_pointer is not None:
            return self.get_variable_value(wrapped_pointer)

        value_type = self.variable_value_types.get(variable_id.id)
        if value_type:
            return self.load_from_variable(variable_id, value_type)

        if variable_id.type.storage_class:
            base_type = variable_id.type.base_type.replace("ptr_", "", 1)
            var_type = self.find_registered_type_by_base(base_type)
            if var_type:
                return self.load_from_variable(variable_id, var_type)
        return variable_id

    def cbuffer_member_pointer(self, name: str) -> Optional[SpirvId]:
        member_info = self.cbuffer_members.get(name)
        if member_info is None:
            return None

        cbuffer_var, member_type, member_index = member_info
        int_type = self.primitive_types["int"]
        index = self.register_constant(member_index, int_type)
        ptr_type = self.register_pointer_type(
            member_type, cbuffer_var.type.storage_class or "Uniform"
        )
        access = self.access_chain(cbuffer_var, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        return access

    def uniform_block_wrapped_member_type(
        self, variable_id: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if variable_id is None:
            return None
        metadata = self.uniform_block_wrapped_variables.get(variable_id.id)
        if metadata is None:
            return None
        return metadata["member_type"]

    def uniform_block_wrapped_member_pointer(
        self, variable_id: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        metadata = (
            self.uniform_block_wrapped_variables.get(variable_id.id)
            if variable_id is not None
            else None
        )
        if metadata is None:
            return None

        int_type = self.primitive_types["int"]
        index = self.register_constant(metadata["member_index"], int_type)
        member_type = metadata["member_type"]
        ptr_type = self.register_pointer_type(member_type, "Uniform")
        access = self.access_chain(variable_id, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        return access

    def structured_buffer_metadata_for_pointer(self, pointer_id: SpirvId):
        metadata = self.structured_buffer_metadata.get(pointer_id.id)
        if metadata is not None:
            return metadata

        pointee_type = self.variable_value_types.get(pointer_id.id)
        if pointee_type is not None:
            return self.structured_buffer_metadata.get(pointee_type.id)

        return None

    def storage_buffer_access_metadata_for_pointer(self, pointer_id: SpirvId):
        if pointer_id is None:
            return None

        metadata = self.storage_buffer_access_metadata.get(pointer_id.id)
        if metadata is not None:
            return metadata

        metadata = self.structured_buffer_metadata_for_pointer(pointer_id)
        if metadata is not None:
            return metadata

        pointee_type = self.variable_value_types.get(pointer_id.id)
        if pointee_type is not None:
            return self.storage_buffer_access_metadata.get(pointee_type.id)

        return None

    def propagate_storage_buffer_access_metadata(
        self, source_pointer: SpirvId, target_pointer: SpirvId
    ):
        metadata = self.storage_buffer_access_metadata_for_pointer(source_pointer)
        if metadata is not None:
            self.storage_buffer_access_metadata[target_pointer.id] = metadata

    def propagate_structured_buffer_descriptor_access_metadata(
        self, source_pointer: SpirvId, target_pointer: SpirvId, index: SpirvId
    ):
        metadata = self.structured_buffer_metadata_for_pointer(source_pointer)
        if metadata is None or not metadata.get("descriptor_array"):
            return

        source_type = self.variable_value_types.get(source_pointer.id)
        block_type = metadata.get("block_type")
        if not self.array_type_contains_element_type(source_type, block_type):
            return

        descriptor_indices = list(metadata.get("_descriptor_indices", []))
        descriptor_indices.append(index)
        access_metadata = {**metadata, "_descriptor_indices": descriptor_indices}
        self.structured_buffer_metadata[target_pointer.id] = access_metadata
        self.storage_buffer_access_metadata[target_pointer.id] = access_metadata

    def storage_buffer_pointer_alias_index(
        self, base_index: SpirvId, index: SpirvId, operator: str = "+"
    ):
        index_type = self.value_types.get(base_index.id) or self.ensure_registered_type(
            base_index.type
        )
        converted_index = self.convert_value_to_type(index, index_type)
        return self.binary_operation(
            operator,
            index_type,
            base_index,
            converted_index,
        )

    def storage_buffer_pointer_alias_key(self, pointer: SpirvId) -> SpirvId:
        """Use generator object identity so copied pointer aliases remain distinct."""
        return pointer

    def is_storage_buffer_pointer_arithmetic_base(self, pointer: SpirvId) -> bool:
        if (
            self.storage_buffer_pointer_alias_key(pointer)
            in self.storage_buffer_pointer_offset_aliases
        ):
            return True

        metadata = self.structured_buffer_metadata_for_pointer(pointer)
        if metadata is None:
            return False
        if metadata.get("_access_path") in {"element", "member"}:
            return False

        type_info = self.storage_buffer_resource_type_info(
            metadata.get("declared_type_name")
        )
        if type_info is not None and type_info.get("kind") == "storage_buffer_pointer":
            return True

        return metadata.get("kind") == "structured_buffer" and metadata.get(
            "buffer_kind"
        ) in {"StructuredBuffer", "RWStructuredBuffer", "StorageBufferPointer"}

    def record_storage_buffer_pointer_offset_alias(
        self,
        source_pointer: SpirvId,
        alias_pointer: SpirvId,
        index: SpirvId,
        operator: str = "+",
    ):
        same_pointer = alias_pointer is source_pointer
        existing_alias = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(source_pointer)
        )
        if existing_alias is None:
            root_pointer = source_pointer
            domain = self.storage_buffer_pointer_alias_domain(root_pointer)
            index_type = self.value_types.get(index.id) or self.ensure_registered_type(
                index.type
            )
            zero = self.register_constant(0, index_type)
            if same_pointer:
                offset_variable = self.create_variable(
                    index_type,
                    "Function",
                    initializer=zero,
                )
                root_index = self.load_from_variable(offset_variable, index_type)
            else:
                offset_variable = None
                root_index = zero
        else:
            index_type = self.value_types.get(index.id) or self.ensure_registered_type(
                index.type
            )
            existing_alias = self.materialize_storage_buffer_pointer_offset_alias(
                source_pointer,
                existing_alias,
                index_type,
            )
            root_pointer = existing_alias.root_pointer
            source_offset_variable = existing_alias.offset_variable
            domain = existing_alias.domain
            source_offset_type = self.variable_value_types[source_offset_variable.id]
            root_index = self.load_from_variable(
                source_offset_variable,
                source_offset_type,
            )
            if same_pointer:
                offset_variable = source_offset_variable
            else:
                zero = self.register_constant(0, source_offset_type)
                offset_variable = self.create_variable(
                    source_offset_type,
                    "Function",
                    initializer=zero,
                )

        if offset_variable is None:
            offset_type = self.value_types.get(
                root_index.id
            ) or self.ensure_registered_type(root_index.type)
            zero = self.register_constant(0, offset_type)
            offset_variable = self.create_variable(
                offset_type,
                "Function",
                initializer=zero,
            )

        combined_index = self.storage_buffer_pointer_alias_index(
            root_index,
            index,
            operator,
        )
        self.store_to_variable(offset_variable, combined_index)
        self.storage_buffer_pointer_offset_aliases[
            self.storage_buffer_pointer_alias_key(alias_pointer)
        ] = PointerOffsetState(root_pointer, offset_variable, domain)
        reinterpretation = self.storage_pointer_reinterpretations.get(source_pointer)
        if reinterpretation is not None:
            self.storage_pointer_reinterpretations[alias_pointer] = dict(
                reinterpretation
            )

    def storage_buffer_pointer_alias_access(
        self, pointer: SpirvId, index: SpirvId
    ) -> Tuple[SpirvId, SpirvId, bool]:
        """Resolve a resource pointer and index through its mutable offset state."""
        offset_alias = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(pointer)
        )
        if offset_alias is None:
            return pointer, index, False

        root_pointer = offset_alias.root_pointer
        offset_variable = offset_alias.offset_variable
        if offset_variable is None:
            return root_pointer, index, True
        offset_type = self.variable_value_types[offset_variable.id]
        base_index = self.load_from_variable(offset_variable, offset_type)
        combined_index = self.storage_buffer_pointer_alias_index(base_index, index)
        return root_pointer, combined_index, True

    def storage_buffer_pointer_alias_root(self, pointer: SpirvId) -> SpirvId:
        """Resolve descriptor-level operations without applying element offsets."""
        offset_alias = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(pointer)
        )
        return offset_alias.root_pointer if offset_alias is not None else pointer

    def dereference_pointer_from_expression(self, expression) -> Optional[SpirvId]:
        pointer = self.variable_pointer_from_expression(expression)
        if pointer is None or not self.is_storage_buffer_pointer_arithmetic_base(
            pointer
        ):
            return None

        offset_alias = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(pointer)
        )
        if offset_alias is not None:
            if offset_alias.offset_variable is not None:
                index_type = self.variable_value_types[offset_alias.offset_variable.id]
            else:
                index_type = self.register_primitive_type("uint")
            zero = self.register_constant(0, index_type)
            root_pointer, effective_index, _ = self.storage_buffer_pointer_alias_access(
                pointer, zero
            )
            structured_access = self.structured_buffer_element_pointer(
                root_pointer,
                effective_index,
                apply_pointer_offset=False,
            )
            if structured_access is not None:
                return structured_access
            local_access = self.local_array_offset_element_pointer(
                root_pointer, effective_index
            )
            if local_access is not None:
                return local_access

        zero = self.register_constant(0, self.register_primitive_type("uint"))
        structured_access = self.structured_buffer_element_pointer(pointer, zero)
        if structured_access is not None:
            return structured_access

        return None

    def unsupported_pointer_dereference_error(self, expression):
        return UnsupportedSPIRVFeatureError(
            "pointer dereference",
            "SPIR-V pointer dereference requires a pointer with preserved storage "
            "provenance",
            missing_capabilities=("spirv.pointer_dereference",),
            source_location=getattr(expression, "source_location", None),
        )

    def structured_buffer_element_pointer(
        self,
        buffer_pointer: SpirvId,
        index_id: SpirvId,
        *,
        apply_pointer_offset: bool = True,
    ) -> Optional[SpirvId]:
        if apply_pointer_offset:
            buffer_pointer, index_id, _ = self.storage_buffer_pointer_alias_access(
                buffer_pointer,
                index_id,
            )
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            return None
        # A pointer to one element (or a member of one) is no longer the buffer: a further
        # subscript selects a component, column or array element of that value.
        if metadata.get("_access_path") in {"element", "member"}:
            return None

        pointee_type = self.variable_value_types.get(buffer_pointer.id)
        descriptor_array = self.array_type_info_from_type(pointee_type)
        block_type = metadata.get("block_type")
        if (
            descriptor_array is not None
            and block_type is not None
            and self.array_type_contains_element_type(pointee_type, block_type)
        ):
            return None

        element_type = metadata["element_type"]
        int_type = self.primitive_types["int"]
        member_index = self.register_constant(metadata.get("member_index", 0), int_type)
        ptr_type = self.register_pointer_type(
            element_type, buffer_pointer.type.storage_class or "Uniform"
        )
        access = self.access_chain(buffer_pointer, [member_index, index_id], ptr_type)
        self.variable_value_types[access.id] = element_type
        access_metadata = {**metadata, "_access_path": "element"}
        self.structured_buffer_metadata[access.id] = access_metadata
        return access

    def process_block_member_length_call(self, expr):
        """``block.member.length()`` (GLSL 4.60 section 4.1.9) as an int.

        A runtime array's length is ``OpArrayLength`` on the block variable; a sized array's is its size.
        """
        callee = getattr(expr, "function", None)
        if (
            isinstance(callee, MemberAccessNode)
            and callee.member == "length"
            and not (getattr(expr, "args", None) or [])
            and isinstance(callee.object_expr, (IdentifierNode, VariableNode))
        ):
            handled = self.process_runtime_view_length_call(callee.object_expr)
            if handled is not None:
                return True, handled
        if not (
            isinstance(callee, MemberAccessNode)
            and callee.member == "length"
            and not (getattr(expr, "args", None) or [])
            and isinstance(callee.object_expr, MemberAccessNode)
        ):
            return False, None
        array_expr = callee.object_expr
        block_pointer = self.variable_pointer_from_expression(array_expr.object_expr)
        if block_pointer is None:
            return False, None
        member_info = self.struct_member_info(
            self.struct_type_name_from_pointer(block_pointer), array_expr.member
        )
        array_info = (
            self.array_type_info_from_type(member_info[1]) if member_info else None
        )
        if array_info is None:
            return False, None
        int_type = self.register_primitive_type("int")
        if array_info[1] is not None:
            return True, self.register_constant(int(array_info[1]), int_type)
        uint_type = self.register_primitive_type("uint")
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpArrayLength %{uint_type.id} %{block_pointer.id} "
            f"{member_info[0]}"
        )
        self.value_types[id_value] = uint_type
        length = SpirvId(id_value, uint_type.type)
        return True, self.convert_value_to_type(length, int_type)

    def process_runtime_view_length_call(self, object_expr):
        """``name.length()`` for a ``buffer T* name`` view: ``OpArrayLength`` of its block's runtime array."""
        pointer = self.variable_pointer_from_expression(object_expr)
        if pointer is None:
            return None
        if pointer in self.storage_pointer_reinterpretations:
            raise BufferLengthUnsupportedError(
                "spirv",
                f"cannot determine length() of reinterpreted buffer view '{getattr(object_expr, 'name', object_expr)}'",
            )
        offset_alias = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(pointer)
        )
        base_pointer = (
            offset_alias.root_pointer if offset_alias is not None else pointer
        )
        metadata = self.structured_buffer_metadata_for_pointer(base_pointer)
        if metadata is None or metadata.get("runtime_array_type") is None:
            return None
        if metadata.get("_access_path") in {"element", "member"}:
            raise BufferLengthUnsupportedError(
                "spirv",
                f"cannot recover base storage block for '{getattr(object_expr, 'name', object_expr)}.length()'",
            )
        member_index = metadata.get("member_index", 0)
        uint_type = self.register_primitive_type("uint")
        int_type = self.register_primitive_type("int")
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpArrayLength %{uint_type.id} %{base_pointer.id} {member_index}"
        )
        self.value_types[id_value] = uint_type
        length = SpirvId(id_value, uint_type.type)
        length_int = self.convert_value_to_type(length, int_type)
        if offset_alias is not None and offset_alias.offset_variable is not None:
            offset_type = self.variable_value_types[offset_alias.offset_variable.id]
            offset_val = self.load_from_variable(
                offset_alias.offset_variable, offset_type
            )
            offset_int = self.convert_value_to_type(offset_val, int_type)
            return self.binary_operation("-", int_type, length_int, offset_int)
        return length_int

    def single_struct_buffer_zero_index_alias(
        self, buffer_pointer: SpirvId, index_expr
    ) -> Optional[SpirvId]:
        """Treat a single buffer block indexed as [0] as an alias for the block."""
        if self.literal_int_argument(index_expr) != 0:
            return None

        if self.structured_buffer_metadata_for_pointer(buffer_pointer) is not None:
            return None

        metadata = self.storage_buffer_access_metadata_for_pointer(buffer_pointer)
        if metadata is None or metadata.get("kind") != "glsl_buffer_block":
            return None

        value_type = self.variable_value_types.get(buffer_pointer.id)
        if value_type is None:
            return None
        if self.array_type_info_from_type(value_type) is not None:
            return None
        if value_type.type.base_type not in self.current_struct_members:
            return None

        return buffer_pointer

    def structured_buffer_default_value(self, metadata) -> SpirvId:
        element_type = metadata.get("element_type") if metadata is not None else None
        if element_type is not None:
            return self.default_value_for_type(element_type)
        return self.register_constant(0.0, self.register_primitive_type("float"))

    def structured_buffer_counter_default_value(self) -> SpirvId:
        return self.register_constant(0, self.register_primitive_type("uint"))

    def structured_buffer_dimensions_default_value(self) -> SpirvId:
        return self.register_constant(0, self.register_primitive_type("uint"))

    def emit_structured_buffer_dimensions(
        self, buffer_pointer: SpirvId, metadata, diagnostic_name: str
    ) -> SpirvId:
        buffer_pointer = self.storage_buffer_pointer_alias_root(buffer_pointer)
        metadata = (
            self.structured_buffer_metadata_for_pointer(buffer_pointer) or metadata
        )
        block_type = metadata.get("block_type") if metadata is not None else None
        if block_type is None:
            self.refuse_lowering(f"{diagnostic_name} requires a buffer block operand")
            return self.structured_buffer_dimensions_default_value()

        uint_type = self.register_primitive_type("uint")
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpArrayLength %{uint_type.id} %{buffer_pointer.id} "
            f"{metadata.get('member_index', 0)}"
        )
        self.value_types[id_value] = uint_type
        length = SpirvId(id_value, uint_type.type)
        if not metadata.get("byte_address"):
            return length

        byte_stride = self.register_constant(4, uint_type)
        return self.binary_operation("*", uint_type, length, byte_stride)
