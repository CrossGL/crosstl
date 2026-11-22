"""Resource specializations in the GLSL generator: resource specializations of function bodies, dynamic resource arguments, fixed array parameters and forwarding."""

import re
from contextlib import contextmanager
from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    AssignmentNode,
    BlockNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MemberAccessNode,
    PointerAccessNode,
    PointerType,
    ReferenceType,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_emission_list,
    generic_function_parameters,
    generic_function_value_arguments,
)
from .glsl_exact_lookups import glsl_call_names_resolving_to
from .opengl_values import OpenGLPrivatePointerParameterError


class OpenGLResourceSpecializationMixin:
    """Resource specialization members of ``GLSLCodeGen``."""

    def collect_glsl_resource_specializations_from_body(
        self,
        body,
        aliases,
        *,
        lexical_call_source_types=None,
        semantic_call_locations=None,
    ):
        lexical_call_source_types = lexical_call_source_types or {}
        semantic_call_locations = semantic_call_locations or {}

        def visit_sequence(value, visible_aliases):
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            for statement in statements:
                visit(statement, visible_aliases)

        def visit(node, visible_aliases):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, dict):
                for child in node.values():
                    visit(child, visible_aliases)
                return
            if isinstance(node, (list, tuple, set)):
                visit_sequence(list(node), visible_aliases)
                return
            # The emitter drops statements the interval proof refutes, so the
            # calls in them need no helper specialization.
            if getattr(node, "_glsl_checked_unreachable", False):
                return

            if isinstance(node, VariableNode):
                if self.is_glsl_type_alias_declaration(node):
                    return
                visible_aliases.pop(node.name, None)
                visit(getattr(node, "initial_value", None), visible_aliases)
                shared_binding = self.glsl_hoisted_shared_binding_for_declaration(node)
                if shared_binding is not None:
                    visible_aliases[node.name] = dict(shared_binding)
                    return
                storage_pointer_array_binding = getattr(
                    node,
                    "_glsl_storage_pointer_array_binding",
                    None,
                )
                if storage_pointer_array_binding is not None:
                    visible_aliases[node.name] = dict(storage_pointer_array_binding)
                    return
                if self.glsl_workgroup_pointer_declaration(node, visible_aliases):
                    pointer_binding = self.glsl_workgroup_pointer_binding(
                        getattr(node, "initial_value", None), visible_aliases
                    )
                    if pointer_binding is not None:
                        visible_aliases[node.name] = pointer_binding
                    else:
                        visible_aliases.pop(node.name, None)
                elif self.glsl_storage_pointer_declaration(node, visible_aliases):
                    pointer_binding = self.glsl_storage_pointer_binding(
                        getattr(node, "initial_value", None), visible_aliases
                    )
                    if pointer_binding is None:
                        deferred_binding = getattr(
                            node,
                            "_glsl_deferred_storage_pointer_binding",
                            None,
                        )
                        if deferred_binding is not None:
                            pointer_binding = dict(deferred_binding)
                    if pointer_binding is not None:
                        visible_aliases[node.name] = pointer_binding
                    else:
                        visible_aliases.pop(node.name, None)
                return

            if isinstance(node, AssignmentNode):
                right = getattr(node, "value", getattr(node, "right", None))
                visit(right, visible_aliases)
                target = getattr(node, "target", getattr(node, "left", None))
                if not isinstance(target, (str, IdentifierNode, VariableNode)):
                    visit(target, visible_aliases)
                    return
                target_name = self.expression_name(target)
                current = visible_aliases.get(target_name)
                if current is not None and current.get("kind") == "workgroup-pointer":
                    operator = self.map_operator(
                        getattr(node, "operator", getattr(node, "op", "="))
                    )
                    if operator == "=":
                        replacement = self.glsl_workgroup_pointer_binding(
                            right, visible_aliases
                        )
                    elif operator in {"+=", "-="}:
                        delta = (
                            right
                            if operator == "+="
                            else self.glsl_pointer_offset_negation(right)
                        )
                        replacement = dict(current)
                        replacement["offset"] = self.glsl_pointer_offset_sum(
                            replacement.get("offset"), delta
                        )
                    else:
                        replacement = None
                    if replacement is not None:
                        visible_aliases[target_name] = replacement
                elif current is not None and current.get("kind") == "storage-pointer":
                    operator = self.map_operator(
                        getattr(node, "operator", getattr(node, "op", "="))
                    )
                    if operator == "=":
                        replacement = self.glsl_storage_pointer_binding(
                            right, visible_aliases
                        )
                    elif operator in {"+=", "-="}:
                        delta = (
                            right
                            if operator == "+="
                            else self.glsl_pointer_offset_negation(right)
                        )
                        replacement = dict(current)
                        replacement["offset"] = self.glsl_pointer_offset_sum(
                            replacement.get("offset"), delta
                        )
                    else:
                        replacement = None
                    if replacement is not None:
                        visible_aliases[target_name] = replacement
                return

            if isinstance(node, FunctionCallNode):
                args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
                for argument in args:
                    visit(argument, visible_aliases)
                func_name = self.function_call_name(node)
                if not func_name:
                    return
                specialized_func_name = generic_function_call_name(
                    self, func_name, args
                )
                if specialized_func_name is not None:
                    args = generic_function_value_arguments(self, func_name, args)
                    func_name = specialized_func_name
                lexical_source_types = lexical_call_source_types.get(id(node), {})
                specialization = self.ensure_glsl_resource_function_specialization(
                    func_name,
                    args,
                    visible_aliases,
                    lexical_source_types=lexical_source_types,
                )
                if specialization is not None:
                    specialization_key = getattr(
                        specialization,
                        "_glsl_resource_specialization_key",
                        None,
                    )
                    if specialization_key is not None:
                        plan = (func_name, len(args), specialization_key)
                        self.record_glsl_prepared_resource_function_call_plan(
                            func_name,
                            args,
                            node,
                            plan,
                            semantic_call_locations=semantic_call_locations,
                        )
                self.ensure_glsl_dynamic_resource_function_specializations(
                    func_name,
                    args,
                    visible_aliases,
                    lexical_source_types=lexical_source_types,
                )
                return

            if isinstance(node, BlockNode):
                visit_sequence(node, dict(visible_aliases))
                return

            if isinstance(node, IfNode):
                condition = getattr(
                    node, "condition", getattr(node, "if_condition", None)
                )
                visit(condition, visible_aliases)
                static_known, static_body = self.glsl_static_if_body(node)
                if static_known:
                    visit_sequence(static_body, dict(visible_aliases))
                    return
                visit_sequence(node.if_body, dict(visible_aliases))
                for else_if_condition, else_if_body in zip(
                    getattr(node, "else_if_conditions", []) or [],
                    getattr(node, "else_if_bodies", []) or [],
                ):
                    visit(else_if_condition, visible_aliases)
                    visit_sequence(else_if_body, dict(visible_aliases))
                visit_sequence(getattr(node, "else_body", None), dict(visible_aliases))
                return

            if isinstance(node, ForNode):
                loop_aliases = dict(visible_aliases)
                visit(getattr(node, "init", None), loop_aliases)
                visit(getattr(node, "condition", None), loop_aliases)
                visit_sequence(getattr(node, "body", None), dict(loop_aliases))
                visit(getattr(node, "update", None), loop_aliases)
                return

            if isinstance(node, ForInNode):
                visit(getattr(node, "iterable", None), visible_aliases)
                loop_aliases = dict(visible_aliases)
                pattern = getattr(node, "pattern", None)
                if pattern:
                    loop_aliases.pop(pattern, None)
                visit_sequence(getattr(node, "body", None), loop_aliases)
                return

            if isinstance(node, (WhileNode, DoWhileNode, LoopNode)):
                visit(getattr(node, "condition", None), visible_aliases)
                visit_sequence(getattr(node, "body", None), dict(visible_aliases))
                return

            if isinstance(node, SwitchNode):
                visit(getattr(node, "expression", None), visible_aliases)
                for case in getattr(node, "cases", []) or []:
                    visit(getattr(case, "value", None), visible_aliases)
                    visit_sequence(
                        getattr(case, "statements", None), dict(visible_aliases)
                    )
                default_case = getattr(node, "default_case", None)
                visit_sequence(
                    getattr(default_case, "statements", default_case),
                    dict(visible_aliases),
                )
                return

            if hasattr(node, "child_nodes"):
                for child in node.child_nodes():
                    visit(child, visible_aliases)
                return
            if hasattr(node, "__dict__"):
                for field, child in vars(node).items():
                    if field not in {"annotations", "parent", "source_location"}:
                        visit(child, visible_aliases)

        visit_sequence(body, dict(aliases))

    def glsl_resource_specializations_replace(self, function):
        """Whether resource specializations stand in for ``function`` entirely.

        GLSL cannot pass buffer blocks to functions, so a helper declared with
        StructuredBuffer parameters (Metal helpers whose pointer parameters
        carry ``[[buffer(n)]]``) only has a valid GLSL form once a call binds
        those parameters to concrete blocks.  The unspecialized original would
        declare unsized array parameters, which GLSL rejects.
        """
        if getattr(function, "_glsl_resource_source_name", None) is not None:
            return False
        name = getattr(function, "name", None)
        if not name or not self.glsl_function_structured_buffer_parameter_aliases(
            function
        ):
            return False
        target_name = self.glsl_module_identifier_name(name)
        return any(
            self.glsl_module_identifier_name(
                getattr(specialization, "_glsl_resource_source_name", None) or ""
            )
            == target_name
            for specialization in self.glsl_resource_function_specializations.values()
        )

    def glsl_resource_function_emission_list(self, source_name):
        if not source_name:
            return []
        functions = [
            func
            for func in self.glsl_resource_function_specializations.values()
            if getattr(func, "_glsl_resource_source_name", None) == source_name
        ]
        functions.sort(key=lambda func: getattr(func, "name", ""))
        result = []
        for func in functions:
            name = self.glsl_function_declaration_name(func)
            if name in self.emitted_glsl_resource_specialization_names:
                continue
            self.emitted_glsl_resource_specialization_names.add(name)
            result.append(func)
        return result

    def generate_glsl_resource_function_declarations(self):
        specializations = sorted(
            (
                function
                for function in self.glsl_resource_function_specializations.values()
                if not self.should_elide_glsl_function(function)
            ),
            key=lambda func: self.glsl_function_declaration_name(func),
        )
        return self.generate_glsl_function_declarations(specializations)

    def glsl_concrete_helper_functions(self, functions):
        concrete = []
        for function in functions:
            if generic_function_parameters(function):
                candidates = generic_function_emission_list(self, function)
            else:
                candidates = [function]
            concrete.extend(
                candidate
                for candidate in candidates
                if not self.should_elide_glsl_function(candidate)
                and not self.glsl_function_has_target_pointer_parameter(candidate)
            )
        return concrete

    def generate_glsl_function_declarations(self, functions, *, stage_context=None):
        declarations = ""
        seen_signatures = set()
        for function in functions:
            target_name = self.glsl_function_declaration_name(function)
            self.register_glsl_function_graph_scope(
                function,
                stage_context=stage_context,
            )
            signature = (
                target_name,
                self.glsl_mapped_function_signature_key(function),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            prototype = deepcopy(function)
            prototype.name = target_name
            prototype.body = None
            declarations += self.generate_function(
                prototype,
                stage_context=stage_context,
            )
        return declarations

    def glsl_resource_binding_info(self, arg, aliases):
        arg_name = self.expression_name(arg)
        is_array_access = isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        )
        alias_binding = aliases.get(arg_name)
        if alias_binding is not None and not is_array_access:
            return alias_binding

        resource_type = self.texture_argument_resource_type(arg)
        if not self.is_storage_image_type(resource_type) and alias_binding is not None:
            resource_type = alias_binding.get("type")
        if not self.is_storage_image_type(resource_type):
            return None

        expression = self.glsl_resource_argument_expression(arg, aliases)
        if expression is None:
            return None

        dynamic_parameters = self.glsl_resource_argument_dynamic_parameters(arg)
        specializable = self.glsl_resource_argument_is_specializable(arg)
        if alias_binding is not None:
            specializable = specializable and alias_binding.get("specializable", True)
            dynamic_parameters = [
                *(alias_binding.get("dynamic_parameters") or []),
                *(dynamic_parameters or []),
            ]

        if alias_binding is not None:
            return {
                **alias_binding,
                "expression": expression,
                "type": resource_type,
                "specializable": specializable,
                "dynamic_parameters": dynamic_parameters or [],
            }

        binding = {
            "expression": expression,
            "type": resource_type,
            "format": self.image_resource_format(arg),
            "access": self.image_resource_access(arg),
            "specializable": specializable,
            "dynamic_parameters": dynamic_parameters or [],
        }
        array_size = self.resource_variable_array_sizes.get(arg_name)
        if array_size is not None:
            binding["array_size"] = array_size
        return binding

    def glsl_resource_argument_expression(self, arg, aliases):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_code = self.glsl_resource_argument_expression(array_expr, aliases)
            if array_code is None:
                return None
            index_code = self.glsl_index_expression(index_expr, array_expr)
            return f"{array_code}[{index_code}]"
        arg_name = self.expression_name(arg)
        if arg_name in aliases:
            return aliases[arg_name]["expression"]
        if isinstance(arg, str):
            return arg
        if hasattr(arg, "name") and isinstance(arg.name, str):
            return arg.name
        return None

    def glsl_resource_argument_dynamic_parameters(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            dynamic_parameters = self.glsl_resource_argument_dynamic_parameters(
                array_expr
            )
            if dynamic_parameters is None:
                return None
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            if isinstance(literal_index, int) and not isinstance(literal_index, bool):
                return dynamic_parameters
            index_name = self.expression_name(index_expr)
            if not index_name:
                return None
            index_expression = self.generate_expression(index_expr)
            dynamic_parameters.append(
                {
                    "name": index_name,
                    "type": self.dynamic_resource_index_parameter_type(index_expr),
                    "expression": index_expression,
                }
            )
            return dynamic_parameters
        return []

    def glsl_resource_argument_is_specializable(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            literal_index = self.literal_int_value(
                index_expr, self.literal_int_constants
            )
            return isinstance(literal_index, int) and not isinstance(
                literal_index, bool
            )
        return True

    def glsl_resource_array_size_value(self, size):
        literal_size = self.literal_int_value(size, self.literal_int_constants)
        if isinstance(literal_size, int) and not isinstance(literal_size, bool):
            return literal_size if literal_size > 0 else None
        return None

    def glsl_resource_parameter_array_size(self, func_name, param_name, param_type):
        fixed_size = self.fixed_resource_array_size(param_type)
        if fixed_size is not None:
            return fixed_size
        hinted_size = self.function_resource_array_size_hints.get(func_name, {}).get(
            param_name
        )
        return self.glsl_resource_array_size_value(hinted_size)

    def glsl_dynamic_resource_array_access_info(self, arg, aliases):
        if not (
            isinstance(arg, ArrayAccessNode)
            or (hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__))
        ):
            return None
        if self.glsl_resource_argument_is_specializable(arg):
            return None

        array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
        index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
        array_name = self.expression_name(array_expr)
        if not array_name:
            return None

        alias_binding = aliases.get(array_name)
        if alias_binding is not None:
            array_size = self.glsl_resource_array_size_value(
                alias_binding.get("array_size")
            )
        else:
            array_size = self.resource_variable_array_sizes.get(array_name)
        if array_size is None:
            return None

        return {
            "array_expr": array_expr,
            "index_expr": index_expr,
            "array_size": array_size,
        }

    def glsl_static_array_access_argument(self, dynamic_info, index):
        return ArrayAccessNode(dynamic_info["array_expr"], index)

    def glsl_dynamic_resource_call_info(
        self,
        func_name,
        args,
        aliases,
        *,
        lexical_source_types=None,
    ):
        callee = self.glsl_resource_function_source(
            func_name,
            args,
            aliases,
            lexical_source_types=lexical_source_types,
        )
        if callee is None:
            return None
        if generic_function_parameters(callee):
            return None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        dynamic_arg = None
        for index, (param, arg) in enumerate(zip(params, args or [])):
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if not self.is_storage_image_type(param_type):
                continue
            binding = self.glsl_resource_binding_info(arg, aliases)
            if binding is None or binding.get("specializable", True):
                continue
            dynamic_info = self.glsl_dynamic_resource_array_access_info(arg, aliases)
            if dynamic_info is None:
                continue
            if dynamic_arg is not None:
                return None
            dynamic_arg = {
                "arg_index": index,
                **dynamic_info,
            }

        return dynamic_arg

    def ensure_glsl_dynamic_resource_function_specializations(
        self,
        func_name,
        args,
        aliases,
        *,
        lexical_source_types=None,
    ):
        dynamic_info = self.glsl_dynamic_resource_call_info(
            func_name,
            args,
            aliases,
            lexical_source_types=lexical_source_types,
        )
        if dynamic_info is None:
            return []

        specializations = []
        for index in range(dynamic_info["array_size"]):
            static_args = list(args or [])
            static_args[dynamic_info["arg_index"]] = (
                self.glsl_static_array_access_argument(dynamic_info, index)
            )
            specialized = self.ensure_glsl_resource_function_specialization(
                func_name,
                static_args,
                aliases,
                lexical_source_types=lexical_source_types,
            )
            if specialized is None:
                return []
            specializations.append((index, specialized, static_args))
        return specializations

    @contextmanager
    def glsl_lexical_source_type_scope(self, lexical_source_types):
        """Temporarily make one call site's exact lexical types authoritative."""

        previous_source_types = self.local_variable_source_types
        previous_types = self.local_variable_types
        source_types = lexical_source_types or {}
        if source_types:
            self.local_variable_source_types = {
                **previous_source_types,
                **source_types,
            }
            self.local_variable_types = {
                **previous_types,
                **source_types,
            }
        try:
            yield
        finally:
            self.local_variable_source_types = previous_source_types
            self.local_variable_types = previous_types

    def dynamic_resource_index_parameter_type(self, index_expr):
        index_name = self.expression_name(index_expr)
        if index_name:
            return self.local_variable_types.get(index_name, "int")
        return "int"

    def glsl_function_has_fixed_array_parameter(self, function):
        return any(
            self.glsl_fixed_array_parameter_shape(parameter) is not None
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        )

    def glsl_resource_function_source(
        self,
        func_name,
        args,
        aliases=None,
        *,
        lexical_source_types=None,
    ):
        candidates = [
            function
            for function in self.glsl_function_overloads_by_name.get(func_name, ())
            if len(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            == len(args or [])
        ]
        if len(candidates) == 1:
            return candidates[0]
        fallback = self.function_definitions.get(func_name)
        if not candidates:
            if fallback is None:
                return None
            parameters = (
                getattr(
                    fallback,
                    "parameters",
                    getattr(fallback, "params", []),
                )
                or []
            )
            return fallback if len(parameters) == len(args or []) else None
        if not any(
            self.glsl_function_has_resource_contract(function)
            or self.glsl_function_has_workgroup_pointer_parameter(function)
            or self.glsl_function_has_storage_pointer_parameter(function)
            or self.glsl_function_has_fixed_array_parameter(function)
            for function in candidates
        ):
            return fallback
        previous_storage_pointer_aliases = self.current_glsl_storage_pointer_aliases
        previous_source_types = self.local_variable_source_types
        previous_types = self.local_variable_types
        if aliases is not None:
            self.current_glsl_storage_pointer_aliases = {
                name: dict(binding)
                for name, binding in aliases.items()
                if binding.get("kind") in {"storage-pointer", "storage-pointer-array"}
            }
        specialization_source_types = getattr(
            self,
            "glsl_resource_specialization_source_types",
            None,
        )
        source_types = {
            **(specialization_source_types or {}),
            **(lexical_source_types or {}),
        }
        if source_types:
            self.local_variable_source_types = {
                **previous_source_types,
                **source_types,
            }
            self.local_variable_types = {
                **previous_types,
                **source_types,
            }
        try:
            return self.resolve_glsl_function_overload(func_name, args)
        finally:
            self.current_glsl_storage_pointer_aliases = previous_storage_pointer_aliases
            self.local_variable_source_types = previous_source_types
            self.local_variable_types = previous_types

    def glsl_fixed_array_parameter_shape(self, parameter):
        raw_type = getattr(
            parameter,
            "param_type",
            getattr(parameter, "vtype", None),
        )
        extent_expression = None
        element_type = None
        if isinstance(raw_type, ArrayType):
            if raw_type.size is None:
                return None
            extent_expression = raw_type.size
            element_type = raw_type.element_type
            nested_array = isinstance(element_type, ArrayType)
        else:
            type_text = self.type_name_string(raw_type)
            base_type, array_suffix = split_array_type_suffix(type_text)
            if not array_suffix or array_suffix == "[]":
                return None
            match = re.fullmatch(r"\[([^\[\]]+)\]", array_suffix.strip())
            extent_expression = match.group(1).strip() if match else None
            element_type = base_type
            nested_array = match is None

        extent = evaluate_literal_int_expression(
            extent_expression,
            self.literal_int_constants,
        )
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            extent = None
        element_type_name = self.type_name_string(element_type)
        if self.is_structured_buffer_type(
            element_type_name
        ) or self.canonical_resource_type(element_type_name):
            return None
        mapped_element_type = None if nested_array else self.map_type(element_type_name)
        extent_text = (
            self.expression_to_string(extent_expression)
            if extent_expression is not None and not isinstance(extent_expression, str)
            else str(extent_expression or "")
        )
        return {
            "extent": extent,
            "extent_text": extent_text,
            "element_type": mapped_element_type,
            "source_element_type": element_type_name,
        }

    def glsl_fixed_array_parameter_contract(self, function, parameter):
        shape = self.glsl_fixed_array_parameter_shape(parameter)
        if shape is None:
            return None
        return {
            **shape,
            "required_access": self.glsl_fixed_array_parameter_required_access(
                function, parameter
            ),
        }

    def validate_glsl_fixed_array_parameter_pointer_escape(
        self,
        function,
        parameter,
    ):
        """Reject local pointer aliases that GLSL array parameters cannot model."""

        parameter_name = getattr(parameter, "name", None)
        shape = self.glsl_fixed_array_parameter_shape(parameter)
        if not parameter_name or shape is None:
            return
        parameter_element_types = {
            parameter_name: (
                shape.get("element_type")
                or self.map_type(shape.get("source_element_type"))
            )
        }
        constants = self.initial_literal_int_constants(function)
        initial_bindings = {
            parameter_name: {
                "root_kind": "parameter",
                "root": parameter_name,
                "offset": 0,
                "offset_interval": (0, 0),
                "side_effecting": False,
                "element_type": parameter_element_types[parameter_name],
            }
        }

        def parameter_binding(
            value,
            lexical_bindings,
            *,
            include_elements=False,
        ):
            if isinstance(value, TernaryOpNode):
                for branch in (value.true_expr, value.false_expr):
                    resolved = parameter_binding(
                        branch,
                        lexical_bindings,
                        include_elements=include_elements,
                    )
                    if resolved is not None:
                        return resolved
                return None
            binding_value = (
                getattr(value, "array", getattr(value, "array_expr", None))
                if include_elements and isinstance(value, ArrayAccessNode)
                else value
            )
            resolved = self.glsl_private_pointer_static_binding(
                binding_value,
                parameter_element_types,
                {},
                constants,
                lexical_bindings,
            )
            if (
                resolved is not None
                and resolved.get("root_kind") == "parameter"
                and resolved.get("root") == parameter_name
            ):
                return resolved
            return None

        def shadow_binding(node):
            name = getattr(node, "name", None)
            array_info = self.glsl_private_pointer_local_array_node_info(
                node,
                constants,
            )
            if array_info is not None:
                return {
                    "root_kind": "local",
                    "root": name,
                    "extent": array_info["extent"],
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": array_info["element_type"],
                }
            scalar_info = self.glsl_private_pointer_local_scalar_node_info(node)
            if scalar_info is not None:
                return {
                    "root_kind": "scalar_value",
                    "root": name,
                    "extent": 1,
                    "offset": 0,
                    "offset_interval": (0, 0),
                    "side_effecting": False,
                    "element_type": scalar_info["element_type"],
                    "source_type": scalar_info["source_type"],
                }
            return {
                "root_kind": None,
                "root": name,
                "offset": None,
                "offset_interval": None,
                "side_effecting": False,
                "element_type": None,
            }

        def reject(alias_name, alias_kind, source):
            function_name = getattr(function, "name", None)
            raise OpenGLPrivatePointerParameterError(
                "OpenGL fixed-array parameter "
                f"'{function_name}.{parameter_name}' escapes through local "
                f"private {alias_kind} '{alias_name}'",
                function_name=function_name,
                parameter_name=parameter_name,
                reason="pointer-escape",
                source_location=getattr(source, "source_location", None),
            )

        def visit(value, lexical_bindings, pointer_names):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child, lexical_bindings, pointer_names)
                return
            if isinstance(value, (list, tuple, set)):
                for child in value:
                    visit(child, lexical_bindings, pointer_names)
                return
            if isinstance(value, BlockNode):
                block_bindings = dict(lexical_bindings)
                block_pointer_names = dict(pointer_names)
                visit(
                    list(getattr(value, "statements", []) or []),
                    block_bindings,
                    block_pointer_names,
                )
                return
            if isinstance(value, (VariableNode, ArrayNode)):
                name = getattr(value, "name", None)
                initial_value = getattr(value, "initial_value", None)
                node_type = getattr(
                    value,
                    "var_type",
                    getattr(value, "vtype", None),
                )
                if isinstance(node_type, (PointerType, ReferenceType)):
                    alias_kind = (
                        "pointer" if isinstance(node_type, PointerType) else "reference"
                    )
                    if (
                        parameter_binding(
                            initial_value,
                            lexical_bindings,
                            include_elements=isinstance(node_type, ReferenceType),
                        )
                        is not None
                    ):
                        reject(name, alias_kind, value)
                visit(initial_value, lexical_bindings, pointer_names)
                if name:
                    lexical_bindings[name] = shadow_binding(value)
                    pointer_names[name] = isinstance(node_type, PointerType)
                return
            if isinstance(value, AssignmentNode):
                target = getattr(value, "target", getattr(value, "left", None))
                assigned = getattr(value, "value", getattr(value, "right", None))
                target_name = (
                    self.expression_name(target)
                    if isinstance(target, (str, IdentifierNode, VariableNode))
                    else None
                )
                if pointer_names.get(target_name) and (
                    parameter_binding(assigned, lexical_bindings) is not None
                ):
                    reject(target_name, "pointer", value)
                visit(target, lexical_bindings, pointer_names)
                visit(assigned, lexical_bindings, pointer_names)
                return
            if isinstance(value, IfNode):
                visit(
                    getattr(value, "condition", getattr(value, "if_condition", None)),
                    lexical_bindings,
                    pointer_names,
                )
                visit(
                    getattr(value, "if_body", None),
                    dict(lexical_bindings),
                    dict(pointer_names),
                )
                for condition, body in zip(
                    getattr(value, "else_if_conditions", []) or [],
                    getattr(value, "else_if_bodies", []) or [],
                ):
                    branch_bindings = dict(lexical_bindings)
                    branch_pointer_names = dict(pointer_names)
                    visit(condition, branch_bindings, branch_pointer_names)
                    visit(body, branch_bindings, branch_pointer_names)
                visit(
                    getattr(value, "else_body", None),
                    dict(lexical_bindings),
                    dict(pointer_names),
                )
                return
            if isinstance(value, ForNode):
                loop_bindings = dict(lexical_bindings)
                loop_pointer_names = dict(pointer_names)
                visit(getattr(value, "init", None), loop_bindings, loop_pointer_names)
                visit(
                    getattr(value, "condition", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                visit(
                    getattr(value, "body", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                visit(
                    getattr(value, "update", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                return
            if isinstance(value, ForInNode):
                loop_bindings = dict(lexical_bindings)
                loop_pointer_names = dict(pointer_names)
                visit(
                    getattr(value, "iterable", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                visit(
                    getattr(value, "body", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                return
            if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
                loop_bindings = dict(lexical_bindings)
                loop_pointer_names = dict(pointer_names)
                visit(
                    getattr(value, "condition", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                visit(
                    getattr(value, "body", None),
                    loop_bindings,
                    loop_pointer_names,
                )
                return
            if isinstance(value, SwitchNode):
                visit(
                    getattr(value, "expression", None),
                    lexical_bindings,
                    pointer_names,
                )
                for case in getattr(value, "cases", []) or []:
                    visit(
                        getattr(case, "value", None),
                        lexical_bindings,
                        pointer_names,
                    )
                    visit(
                        getattr(case, "statements", None),
                        dict(lexical_bindings),
                        dict(pointer_names),
                    )
                default_case = getattr(value, "default_case", None)
                visit(
                    getattr(default_case, "statements", default_case),
                    dict(lexical_bindings),
                    dict(pointer_names),
                )
                return
            if hasattr(value, "child_nodes"):
                for child in value.child_nodes():
                    visit(child, lexical_bindings, pointer_names)
                return
            if hasattr(value, "__dict__"):
                for field, child in vars(value).items():
                    if field not in {"annotations", "parent", "source_location"}:
                        visit(child, lexical_bindings, pointer_names)

        visit(getattr(function, "body", []), initial_bindings, {})

    def glsl_fixed_array_parameter_required_access(self, function, parameter):
        self.validate_glsl_fixed_array_parameter_pointer_escape(function, parameter)
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if "inout" in qualifiers:
            return "read_write"
        if "out" in qualifiers or "writeonly" in qualifiers:
            return "write"
        if qualifiers.intersection({"const", "constant", "readonly", "in"}):
            return "read"

        parameter_name = getattr(parameter, "name", None)
        if parameter_name and self.glsl_fixed_array_parameter_is_mutated(
            function, parameter_name
        ):
            return "read_write"
        return "read"

    def glsl_fixed_array_mutation_analysis_state(self):
        owner = self.glsl_function_overloads_by_name
        if getattr(self, "glsl_fixed_array_mutation_owner", None) is not owner:
            self.glsl_fixed_array_mutation_owner = owner
            self.glsl_fixed_array_mutation_cache = {}
            self.glsl_fixed_array_mutation_active = set()
        return (
            self.glsl_fixed_array_mutation_cache,
            self.glsl_fixed_array_mutation_active,
        )

    def glsl_fixed_array_readonly_intrinsic(self, function_name):
        if not function_name:
            return False
        if (
            function_name not in self.function_map
            and self.glsl_constructor_type(function_name) is not None
        ):
            return True
        if function_name in glsl_call_names_resolving_to(
            self, "texture-resource-operations", self.texture_resource_operation_names()
        ):
            return True
        readonly_names = (
            set(self.GLSL_COMPONENTWISE_UNARY_FUNCTIONS)
            | set(self.GLSL_COMPONENTWISE_BOOLEAN_FUNCTIONS)
            | set(self.GLSL_BOOLEAN_REDUCTION_FUNCTIONS)
            | set(self.GLSL_DERIVATIVE_FUNCTIONS)
            | set(self.GLSL_INTERPOLATION_FUNCTIONS)
            | set(self.GLSL_BITCAST_FUNCTIONS)
            | set(self.GLSL_COPYSIGN_NAMES)
            | set(self.GLSL_ISFINITE_NAMES)
            | set(self.GLSL_SIGNBIT_NAMES)
            | {
                "atan2",
                "clamp",
                "cross",
                "distance",
                "dot",
                "fmod",
                "length",
                "lerp",
                "log10",
                "mad",
                "max",
                "min",
                "mix",
                "mod",
                "mul",
                "normalize",
                "pow",
                "rcp",
                "reflect",
                "refract",
                "saturate",
                "smoothstep",
                "step",
            }
        )
        return function_name in readonly_names or function_name in (
            glsl_call_names_resolving_to(self, "fixed-array-readonly", readonly_names)
        )

    def glsl_fixed_array_mutation_callee(
        self,
        function,
        call,
        parameter_name,
        *,
        lexical_source_types=None,
    ):
        function_name = self.function_call_name(call)
        arguments = list(getattr(call, "arguments", getattr(call, "args", [])) or [])
        candidates = [
            candidate
            for candidate in self.glsl_function_overloads_by_name.get(function_name, ())
            if len(
                getattr(
                    candidate,
                    "parameters",
                    getattr(candidate, "params", []),
                )
                or []
            )
            == len(arguments)
        ]
        fallback = self.function_definitions.get(function_name)
        if (
            fallback is not None
            and fallback not in candidates
            and len(
                getattr(fallback, "parameters", getattr(fallback, "params", [])) or []
            )
            == len(arguments)
        ):
            candidates.append(fallback)
        if not candidates:
            if self.glsl_fixed_array_readonly_intrinsic(function_name):
                return None
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot prove fixed-array writeback through unresolved "
                f"call '{function_name or '<unknown>'}' from "
                f"'{getattr(function, 'name', None)}'",
                function_name=getattr(function, "name", None),
                parameter_name=parameter_name,
                reason="fixed-array-mutation-call-unresolved",
                source_location=getattr(call, "source_location", None),
            )

        if len(candidates) == 1:
            callee = candidates[0]
        else:
            callee = self.glsl_resolve_function_overload_in_scope(
                function,
                function_name,
                arguments,
                call_node=call,
                lexical_source_types=lexical_source_types,
            )
        if callee is None or getattr(callee, "body", None) is None:
            raise OpenGLPrivatePointerParameterError(
                "OpenGL cannot prove fixed-array writeback through call "
                f"'{function_name or '<unknown>'}' without one resolved helper "
                "definition",
                function_name=getattr(function, "name", None),
                parameter_name=parameter_name,
                reason="fixed-array-mutation-call-unresolved",
                source_location=getattr(call, "source_location", None),
            )
        return callee

    def glsl_fixed_array_forwarding_kind(self, argument, parameter_name):
        addressed = False
        value = argument
        if isinstance(value, UnaryOpNode) and self.map_operator(value.op) == "&":
            addressed = True
            value = value.operand
        if self.glsl_fixed_array_target_root_name(value) != parameter_name:
            return None
        if isinstance(value, (str, IdentifierNode, VariableNode)):
            return "addressed-array" if addressed else "array"
        if isinstance(value, (ArrayAccessNode, MemberAccessNode, PointerAccessNode)):
            return "addressed-element" if addressed else "element"
        return "addressed-value" if addressed else "value"
