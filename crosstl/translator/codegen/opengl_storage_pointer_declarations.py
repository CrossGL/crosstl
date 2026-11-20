"""Storage pointer declarations in the GLSL generator: global declaration names, array elements, workgroup functions, mutation and pointee types."""

import re
from copy import deepcopy
from types import SimpleNamespace

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
    PointerType,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import (
    evaluate_literal_int_expression,
    split_array_type_suffix,
)
from .opengl_values import OpenGLStoragePointerError


class OpenGLStoragePointerDeclarationMixin:
    """Storage pointer declaration members of ``GLSLCodeGen``."""

    def glsl_global_declaration_identifier_name(self, node, name):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", ()) or ()
        }
        attributes = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", ()) or ()
        }
        preserves_fragment_depth = (
            name == "gl_FragDepth"
            and "out" in qualifiers
            and bool(
                attributes
                & {"depth_any", "depth_greater", "depth_less", "depth_unchanged"}
            )
        )
        if preserves_fragment_depth:
            self.current_identifier_aliases.pop(name, None)
            return name
        return self.glsl_global_identifier_name(name)

    def glsl_declaration_has_shared_qualifier(self, node):
        """Return whether a declaration carries a workgroup-shared qualifier."""
        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        return bool(qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"})

    def glsl_workgroup_pointer_pointee_type(self, node):
        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        qualifiers = {
            str(value).lower() for value in getattr(node, "qualifiers", []) or []
        }
        for attribute in getattr(node, "attributes", []) or []:
            name = getattr(attribute, "name", None)
            if name:
                qualifiers.add(str(name).lower())
        if not qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"}:
            return None
        if isinstance(raw_type, PointerType):
            return raw_type.pointee_type
        raw_type_name = self.type_name_string(raw_type)
        if isinstance(raw_type_name, str) and raw_type_name.rstrip().endswith("*"):
            return raw_type_name.rstrip()[:-1].strip()
        return None

    def glsl_workgroup_pointer_element_type(self, node):
        pointee_type = self.glsl_workgroup_pointer_pointee_type(node)
        if pointee_type is None:
            return None
        return self.map_type(self.type_name_string(pointee_type))

    def glsl_workgroup_pointer_declaration(self, node, aliases=None):
        if self.glsl_workgroup_pointer_pointee_type(node) is not None:
            return True
        if aliases is None:
            return False

        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        raw_type_name = self.type_name_string(raw_type)
        if not isinstance(raw_type_name, str) or raw_type_name.strip() != "auto":
            return False
        initial_value = getattr(node, "initial_value", None)
        if isinstance(initial_value, ArrayAccessNode):
            return False
        return self.glsl_workgroup_pointer_binding(initial_value, aliases) is not None

    def glsl_function_has_workgroup_pointer_parameter(self, function):
        return any(
            self.glsl_workgroup_pointer_declaration(parameter)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        )

    def glsl_storage_pointer_pointee_type(self, node):
        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        qualifiers = {
            str(value).lower() for value in getattr(node, "qualifiers", []) or []
        }
        for attribute in getattr(node, "attributes", []) or []:
            name = getattr(attribute, "name", None)
            if name:
                qualifiers.add(str(name).lower())
        if qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"}:
            return None
        # CrossGL spells a storage-buffer pointer `buffer T* p`.
        if isinstance(raw_type, PointerType) and "buffer" in qualifiers:
            return raw_type.pointee_type
        if not qualifiers & {"constant", "device", "storage"}:
            return None
        if isinstance(raw_type, PointerType):
            return raw_type.pointee_type
        raw_type_name = self.type_name_string(raw_type)
        if isinstance(raw_type_name, str) and raw_type_name.rstrip().endswith("*"):
            return raw_type_name.rstrip()[:-1].strip()
        return None

    def glsl_storage_pointer_element_type(self, node):
        pointee_type = self.glsl_storage_pointer_pointee_type(node)
        if pointee_type is None:
            return None
        pointee_name = self.strip_metal_parameter_indirection(
            self.type_name_string(pointee_type)
        )
        if pointee_name == "auto":
            return None
        return self.map_type(pointee_name)

    def glsl_storage_pointer_array_contract(self, node):
        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        pointer_type = None
        extent_expression = None
        if isinstance(raw_type, ArrayType):
            pointer_type = raw_type.element_type
            extent_expression = raw_type.size
        else:
            type_name = self.type_name_string(raw_type)
            base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
            match = re.fullmatch(r"\[([^\[\]]+)\]", array_suffix.strip())
            if match is not None and base_type.rstrip().endswith("*"):
                pointer_type = base_type
                extent_expression = match.group(1).strip()
        if pointer_type is None:
            return None

        qualifiers = {
            str(value).lower() for value in getattr(node, "qualifiers", []) or []
        }
        for attribute in getattr(node, "attributes", []) or []:
            name = getattr(attribute, "name", None)
            if name:
                qualifiers.add(str(name).lower())
        if qualifiers & {"groupshared", "shared", "threadgroup", "workgroup"}:
            return None

        if isinstance(pointer_type, PointerType):
            address_space = str(pointer_type.address_space or "").lower()
            pointee_type = pointer_type.pointee_type
        else:
            address_space = ""
            pointer_name = self.type_name_string(pointer_type).rstrip()
            pointee_type = (
                pointer_name[:-1].strip() if pointer_name.endswith("*") else None
            )
        if pointee_type is None:
            return None
        if address_space not in {"constant", "device", "storage", "global"} and not (
            qualifiers & {"constant", "device", "storage", "global"}
        ):
            return None

        extent = evaluate_literal_int_expression(
            extent_expression,
            self.glsl_active_literal_int_constants(),
        )
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            extent = None
        pointee_name = self.strip_metal_parameter_indirection(
            self.type_name_string(pointee_type)
        )
        if not pointee_name or pointee_name == "auto":
            return None
        required_access = (
            "read"
            if qualifiers.intersection({"const", "constant", "readonly", "in"})
            else "read_write"
        )
        return {
            "extent": extent,
            "extent_expression": extent_expression,
            "element_type": self.map_type(pointee_name),
            "source_element_type": pointee_name,
            "required_access": required_access,
        }

    def glsl_storage_pointer_array_mutation_analysis_state(self):
        """Return mutation caches bound to the current prepared overload graph."""

        owner = self.glsl_function_overloads_by_name
        if (
            getattr(self, "glsl_storage_pointer_array_mutation_owner", None)
            is not owner
        ):
            self.glsl_storage_pointer_array_mutation_owner = owner
            self.glsl_storage_pointer_array_mutation_cache = {}
            self.glsl_storage_pointer_array_mutation_active = set()
        return (
            self.glsl_storage_pointer_array_mutation_cache,
            self.glsl_storage_pointer_array_mutation_active,
        )

    def glsl_function_overload_source_types(self, function):
        """Collect unambiguous function-scope hints for pre-emission binding."""

        source_types = {}
        signatures = {}
        ambiguous = set()

        def register(node):
            name = getattr(node, "name", None)
            raw_type = getattr(
                node,
                "param_type",
                getattr(node, "var_type", getattr(node, "vtype", None)),
            )
            type_name = self.type_name_string(raw_type)
            if not name or not type_name or type_name == "auto":
                return
            signature = self.glsl_source_type_signature(raw_type)
            previous = signatures.get(name)
            if previous is not None and previous != signature:
                ambiguous.add(name)
                source_types.pop(name, None)
                return
            if name not in ambiguous:
                signatures[name] = signature
                source_types[name] = raw_type

        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            register(parameter)
        for node in self.walk_ast(getattr(function, "body", [])):
            if isinstance(node, (VariableNode, ArrayNode)):
                register(node)
        return source_types

    def glsl_storage_pointer_array_mutation_source_types(self, function):
        """Compatibility wrapper for existing specialization preparation."""

        return self.glsl_function_overload_source_types(function)

    def glsl_resolve_function_overload_in_scope(
        self,
        function,
        function_name,
        arguments,
        *,
        call_node=None,
        lexical_source_types=None,
    ):
        """Resolve a call with conservative pre-emission function type hints."""

        previous_source_types = self.local_variable_source_types
        previous_types = self.local_variable_types
        source_types = {
            **self.glsl_function_overload_source_types(function),
            **previous_source_types,
            **(lexical_source_types or {}),
        }
        self.local_variable_source_types = source_types
        self.local_variable_types = {
            **source_types,
            **previous_types,
            **(lexical_source_types or {}),
        }
        try:
            return self.resolve_glsl_function_overload(
                function_name,
                arguments,
                call_node=call_node,
            )
        finally:
            self.local_variable_source_types = previous_source_types
            self.local_variable_types = previous_types

    def glsl_resource_function_call_specialization_in_scope(
        self,
        function,
        function_name,
        arguments,
        *,
        lexical_source_types=None,
        call_node=None,
    ):
        """Look up a prepared specialization before local declarations emit."""

        previous_source_types = self.local_variable_source_types
        previous_types = self.local_variable_types
        source_types = self.glsl_function_overload_source_types(function)
        self.local_variable_source_types = {
            **source_types,
            **previous_source_types,
            **(lexical_source_types or {}),
        }
        self.local_variable_types = {
            **source_types,
            **previous_types,
            **(lexical_source_types or {}),
        }
        try:
            return self.glsl_resource_function_call_specialization(
                function_name,
                arguments,
                lexical_source_types=lexical_source_types,
                call_node=call_node,
                caller_function=function,
            )
        finally:
            self.local_variable_source_types = previous_source_types
            self.local_variable_types = previous_types

    def glsl_storage_pointer_array_mutation_callee(
        self,
        function,
        call,
        *,
        lexical_source_types=None,
    ):
        """Resolve one helper call before following pointer-array mutation."""

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
            raise OpenGLStoragePointerError(
                "OpenGL cannot prove whether storage pointer-array offsets are "
                f"mutated through unresolved call '{function_name or '<unknown>'}' "
                f"from '{getattr(function, 'name', None)}'",
                function_name=getattr(function, "name", None),
                reason="pointer-array-mutation-call-unresolved",
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
            raise OpenGLStoragePointerError(
                "OpenGL cannot prove whether storage pointer-array offsets are "
                f"mutated through call '{function_name or '<unknown>'}' without "
                "one resolved helper definition",
                function_name=getattr(function, "name", None),
                reason="pointer-array-mutation-call-unresolved",
                source_location=getattr(call, "source_location", None),
            )
        return callee

    def glsl_storage_pointer_array_parameter_is_mutated(self, function, parameter_name):
        """Prove direct and transitively forwarded logical-offset mutation."""

        cache, active = self.glsl_storage_pointer_array_mutation_analysis_state()
        cache_key = (id(function), parameter_name)
        if cache_key in cache:
            return cache[cache_key]
        if cache_key in active:
            raise OpenGLStoragePointerError(
                "OpenGL cannot prove storage pointer-array offset mutation "
                f"through recursive helper '{getattr(function, 'name', None)}'",
                function_name=getattr(function, "name", None),
                parameter_name=parameter_name,
                reason="pointer-array-mutation-cycle",
                source_location=getattr(function, "source_location", None),
            )

        parameters = list(
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        parameter = next(
            (
                candidate
                for candidate in parameters
                if getattr(candidate, "name", None) == parameter_name
            ),
            None,
        )
        source_contract = (
            self.glsl_storage_pointer_array_contract(parameter)
            if parameter is not None
            else None
        )
        if source_contract is None:
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer-array mutation analysis requires a "
                f"resolved contract for '{getattr(function, 'name', None)}."
                f"{parameter_name}'",
                function_name=getattr(function, "name", None),
                parameter_name=parameter_name,
                reason="pointer-array-mutation-contract-unresolved",
                source_location=getattr(parameter, "source_location", None),
            )

        nodes = list(self.walk_ast(getattr(function, "body", [])))
        lexical_call_source_types = self.glsl_scalar_array_view_lexical_bindings(
            function
        )["call_source_types"]
        shadowing_declaration = next(
            (
                node
                for node in nodes
                if isinstance(node, (VariableNode, ArrayNode))
                and getattr(node, "name", None) == parameter_name
            ),
            None,
        )
        if shadowing_declaration is not None:
            raise OpenGLStoragePointerError(
                "OpenGL cannot prove storage pointer-array offset mutation "
                f"for '{getattr(function, 'name', None)}.{parameter_name}' "
                "across a shadowing local declaration",
                function_name=getattr(function, "name", None),
                parameter_name=parameter_name,
                reason="pointer-array-mutation-shadowed-binding",
                source_location=getattr(
                    shadowing_declaration,
                    "source_location",
                    None,
                ),
            )

        active.add(cache_key)
        try:
            for node in nodes:
                target = None
                if isinstance(node, AssignmentNode):
                    target = getattr(node, "target", getattr(node, "left", None))
                elif isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                    "++",
                    "--",
                }:
                    target = node.operand
                if not isinstance(target, ArrayAccessNode):
                    continue
                array = getattr(target, "array", getattr(target, "array_expr", None))
                if isinstance(array, (str, IdentifierNode, VariableNode)) and (
                    self.expression_name(array) == parameter_name
                ):
                    cache[cache_key] = True
                    return True

            for node in nodes:
                if not isinstance(node, FunctionCallNode):
                    continue
                arguments = list(
                    getattr(node, "arguments", getattr(node, "args", [])) or []
                )
                forwarded_arguments = []
                for index, argument in enumerate(arguments):
                    if isinstance(argument, (str, IdentifierNode, VariableNode)) and (
                        self.expression_name(argument) == parameter_name
                    ):
                        forwarded_arguments.append((index, "array"))
                        continue

                    element = None
                    forwarding_kind = None
                    if isinstance(argument, ArrayAccessNode):
                        element = argument
                        forwarding_kind = "element"
                    elif (
                        isinstance(argument, UnaryOpNode)
                        and self.map_operator(argument.op) == "&"
                        and isinstance(argument.operand, ArrayAccessNode)
                    ):
                        element = argument.operand
                        forwarding_kind = "element-array"
                    if element is None:
                        continue
                    array = getattr(
                        element,
                        "array",
                        getattr(element, "array_expr", None),
                    )
                    if isinstance(array, (str, IdentifierNode, VariableNode)) and (
                        self.expression_name(array) == parameter_name
                    ):
                        forwarded_arguments.append((index, forwarding_kind))

                if not forwarded_arguments:
                    continue

                callee = self.glsl_storage_pointer_array_mutation_callee(
                    function,
                    node,
                    lexical_source_types=lexical_call_source_types.get(id(node), {}),
                )
                callee_parameters = list(
                    getattr(
                        callee,
                        "parameters",
                        getattr(callee, "params", []),
                    )
                    or []
                )
                for argument_index, forwarding_kind in forwarded_arguments:
                    if argument_index >= len(callee_parameters):
                        raise OpenGLStoragePointerError(
                            "OpenGL storage pointer-array mutation analysis "
                            "resolved a helper with an incompatible arity",
                            function_name=getattr(function, "name", None),
                            parameter_name=parameter_name,
                            reason="pointer-array-mutation-call-unresolved",
                            source_location=getattr(node, "source_location", None),
                        )
                    callee_parameter = callee_parameters[argument_index]
                    if forwarding_kind == "element":
                        if not self.glsl_storage_pointer_declaration(callee_parameter):
                            raise OpenGLStoragePointerError(
                                "OpenGL cannot prove storage pointer-array element "
                                "offset mutation through a helper parameter without "
                                "a scalar storage-pointer contract",
                                function_name=getattr(function, "name", None),
                                parameter_name=parameter_name,
                                reason="pointer-array-mutation-contract-unresolved",
                                source_location=getattr(node, "source_location", None),
                            )
                        if self.glsl_storage_pointer_element_type(
                            callee_parameter
                        ) != source_contract.get("element_type"):
                            raise OpenGLStoragePointerError(
                                "OpenGL cannot preserve storage pointer-array "
                                "element mutation across incompatible helper "
                                "contracts",
                                function_name=getattr(function, "name", None),
                                parameter_name=parameter_name,
                                reason="pointer-array-mutation-contract-mismatch",
                                source_location=getattr(node, "source_location", None),
                            )
                        if (
                            self.glsl_storage_pointer_offset_parameter_direction(
                                callee_parameter
                            )
                            is not None
                        ):
                            cache[cache_key] = True
                            return True
                        continue

                    callee_contract = self.glsl_storage_pointer_array_contract(
                        callee_parameter
                    )
                    if callee_contract is None:
                        raise OpenGLStoragePointerError(
                            "OpenGL cannot preserve storage pointer-array offset "
                            f"mutation through '{getattr(callee, 'name', None)}."
                            f"{getattr(callee_parameter, 'name', None)}' without "
                            "a matching fixed pointer-array contract",
                            function_name=getattr(function, "name", None),
                            parameter_name=parameter_name,
                            reason="pointer-array-mutation-contract-unresolved",
                            source_location=getattr(node, "source_location", None),
                        )
                    expected_extent = (
                        source_contract.get("extent")
                        if forwarding_kind == "array"
                        else 1
                    )
                    if callee_contract.get("extent") != expected_extent or (
                        callee_contract.get("element_type")
                        != source_contract.get("element_type")
                    ):
                        raise OpenGLStoragePointerError(
                            "OpenGL cannot preserve storage pointer-array offset "
                            "mutation across incompatible helper contracts",
                            function_name=getattr(function, "name", None),
                            parameter_name=parameter_name,
                            reason="pointer-array-mutation-contract-mismatch",
                            source_location=getattr(node, "source_location", None),
                        )
                    if self.glsl_storage_pointer_array_parameter_is_mutated(
                        callee,
                        getattr(callee_parameter, "name", None),
                    ):
                        cache[cache_key] = True
                        return True

            cache[cache_key] = False
            return False
        finally:
            active.discard(cache_key)

    def glsl_storage_pointer_array_index_interval(
        self,
        binding,
        index,
        expression,
        *,
        interval_overrides=None,
    ):
        intervals = dict(
            getattr(index, "_glsl_control_flow_intervals", None)
            or getattr(expression, "_glsl_control_flow_intervals", None)
            or {}
        )
        if interval_overrides:
            intervals.update(interval_overrides)
        interval = self.glsl_private_pointer_interval(
            index,
            intervals,
            self.glsl_active_literal_int_constants(),
        )
        extent = binding.get("extent")
        name = binding.get("source_name") or binding.get("parameter_name")
        if interval is None:
            raise OpenGLStoragePointerError(
                "OpenGL cannot prove a fixed storage-pointer-array index range "
                f"for '{name}'",
                parameter_name=name,
                reason="pointer-array-index-unproven",
                source_location=getattr(expression, "source_location", None),
            )
        if not isinstance(extent, int) or interval[0] < 0 or interval[1] >= extent:
            raise OpenGLStoragePointerError(
                "OpenGL fixed storage-pointer-array access for "
                f"'{name}' requires elements {interval[0]} through {interval[1]}, "
                f"but the proven extent is {extent}",
                parameter_name=name,
                reason="pointer-array-index-out-of-range",
                source_location=getattr(expression, "source_location", None),
            )
        return interval

    def glsl_storage_pointer_array_selected_offset(self, binding, index, expression):
        interval = self.glsl_storage_pointer_array_index_interval(
            binding, index, expression
        )
        rendered_index = self.glsl_index_expression(index, binding.get("source_name"))
        offset_array = binding.get("offset_array")
        if offset_array:
            return f"{offset_array}[{rendered_index}]"
        offsets = list(binding.get("offset_expressions") or [])
        extent = binding.get("extent")
        if len(offsets) != extent:
            return None
        if interval[0] == interval[1]:
            return offsets[interval[0]]
        selected = str(offsets[interval[1]])
        for element_index in range(interval[1] - 1, interval[0] - 1, -1):
            selected = (
                f"(({rendered_index} == {element_index}) ? "
                f"{offsets[element_index]} : {selected})"
            )
        return selected

    def glsl_storage_pointer_array_element_binding(self, expression, aliases):
        if not isinstance(expression, ArrayAccessNode):
            return None
        array = getattr(expression, "array", getattr(expression, "array_expr", None))
        if not isinstance(array, (str, IdentifierNode, VariableNode)):
            return None
        name = self.expression_name(array)
        binding = aliases.get(name)
        if binding is None or binding.get("kind") != "storage-pointer-array":
            return None
        index = getattr(expression, "index", getattr(expression, "index_expr", None))
        offset = self.glsl_storage_pointer_array_selected_offset(
            binding, index, expression
        )
        if offset is None:
            return None
        bounds = getattr(expression, "_glsl_storage_pointer_read_interval", None)
        if (
            binding.get("offset_array")
            and bounds is not None
            and 0 <= bounds[0] <= bounds[1] < (1 << 31)
            and "pointer_reinterpretation" not in binding
            and "byte_offset" not in binding
        ):
            # Retain a typed subscript and its source-owned occurrence proof.
            # Never attach this fact to the mutable alias table.
            offset = ArrayAccessNode(
                IdentifierNode(binding["offset_array"]), deepcopy(index)
            )
            offset._glsl_lowered_pointer_offset_interval = bounds
            offset.source_location = getattr(expression, "source_location", None)
        return {
            **binding,
            "kind": "storage-pointer",
            "offset": offset,
            "resource_root": False,
        }

    def glsl_storage_pointer_array_binding(self, expression, aliases, contract=None):
        name = self.expression_name(expression)
        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            binding = aliases.get(name)
            if binding is not None and binding.get("kind") == "storage-pointer-array":
                return dict(binding)

        if (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "&"
        ):
            scalar_binding = self.glsl_storage_pointer_binding(
                expression.operand, aliases
            )
            if scalar_binding is None:
                return None
            expected_extent = contract.get("extent") if contract is not None else 1
            if expected_extent != 1:
                raise OpenGLStoragePointerError(
                    "OpenGL cannot bind the address of one storage pointer to a "
                    f"fixed pointer array of extent {expected_extent}",
                    parameter_name=(contract or {}).get("parameter_name"),
                    reason="pointer-array-extent-mismatch",
                    source_location=getattr(expression, "source_location", None),
                )
            return {
                **scalar_binding,
                "kind": "storage-pointer-array",
                "extent": 1,
                # Discovery has lexical bindings, not an emission context.
                # Keep helper calls and source range facts until call emission.
                "offset_expressions": [scalar_binding.get("offset", "0")],
                "offset_array": None,
                "resource_root": False,
            }
        return None

    def glsl_storage_pointer_array_loop_iteration_values(
        self, node, *, counter_mutated=None
    ):
        """Return exact values for a bounded side-effect-free unit-step loop."""

        init = getattr(node, "init", None)
        loop_name = getattr(init, "name", None)
        initial_value = getattr(init, "initial_value", None)
        if not loop_name or initial_value is None or counter_mutated:
            return None
        constants = self.glsl_active_literal_int_constants()
        initial_intervals = dict(
            getattr(init, "_glsl_control_flow_intervals", {}) or {}
        )
        start = self.glsl_private_pointer_interval(
            initial_value,
            initial_intervals,
            constants,
        )
        if start is None or start[0] != start[1]:
            return None
        initial_intervals[loop_name] = start
        proven_name, loop_interval = self.glsl_private_pointer_loop_interval(
            node,
            initial_intervals,
            constants,
        )
        if proven_name != loop_name or loop_interval is None:
            return None

        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return None
        operator = self.map_operator(condition.op)
        left_name = self.expression_name(condition.left)
        right_name = self.expression_name(condition.right)
        if left_name == loop_name:
            bound_expression = condition.right
        elif right_name == loop_name:
            bound_expression = condition.left
            operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(operator)
        else:
            return None
        condition_intervals = getattr(
            condition,
            "_glsl_control_flow_intervals",
            {},
        )
        bound = self.glsl_private_pointer_interval(
            bound_expression,
            condition_intervals,
            constants,
        )
        if bound is None or bound[0] != bound[1]:
            return None

        update = getattr(node, "update", None)
        if not isinstance(update, UnaryOpNode):
            return None
        update_name = self.expression_name(update.operand)
        update_operator = self.map_operator(update.op)
        if update_name != loop_name or update_operator not in {"++", "--"}:
            return None
        for child in self.walk_ast(getattr(node, "body", None)):
            if isinstance(child, (BreakNode, ContinueNode, ReturnNode)):
                return None
            if (
                counter_mutated is None
                and isinstance(child, (VariableNode, ArrayNode))
                and (getattr(child, "name", None) == loop_name)
            ):
                return None
            if counter_mutated is None and isinstance(child, AssignmentNode):
                target = getattr(child, "target", getattr(child, "left", None))
                if self.expression_name(target) == loop_name:
                    return None
            if (
                counter_mutated is None
                and isinstance(child, UnaryOpNode)
                and self.map_operator(child.op)
                in {
                    "++",
                    "--",
                }
            ):
                if self.expression_name(child.operand) == loop_name:
                    return None

        predicates = {
            "<": lambda value, limit: value < limit,
            "<=": lambda value, limit: value <= limit,
            ">": lambda value, limit: value > limit,
            ">=": lambda value, limit: value >= limit,
        }
        predicate = predicates.get(operator)
        if predicate is None:
            return None
        body = getattr(node, "body", None)
        body_mutations = (
            self.glsl_function_mutated_interval_names(SimpleNamespace(body=body))
            if counter_mutated is None
            else set()
        )
        if loop_name in body_mutations:
            # Exact iteration replay is only sound while the loop controller is
            # owned by the loop itself. Taking its address, passing it to an
            # out/inout helper, or mutating it directly invalidates the proof.
            return None

        step = 1 if update_operator == "++" else -1
        current = start[0]
        limit = bound[0]
        values = []
        while predicate(current, limit):
            if len(values) >= 4096:
                return None
            values.append(current)
            current += step
        if values and (min(values), max(values)) != loop_interval:
            return None
        return tuple(values)

    def validate_glsl_storage_pointer_array_initialization(
        self,
        function,
        declaration,
        binding,
    ):
        """Require every reachable pointer-array use to be definitely initialized."""

        name = getattr(declaration, "name", None)
        extent = binding.get("extent")
        expected = set(range(extent))
        function_name = getattr(function, "name", None)
        interval_overrides = {}

        def fail(missing, node):
            raise OpenGLStoragePointerError(
                "OpenGL fixed storage-pointer array "
                f"'{name}' may use uninitialized elements {sorted(missing)}",
                function_name=function_name,
                parameter_name=name,
                reason="pointer-array-use-before-initialization",
                source_location=getattr(node, "source_location", None),
            )

        def direct_access(value):
            if not isinstance(value, ArrayAccessNode):
                return None
            array = getattr(value, "array", getattr(value, "array_expr", None))
            if not isinstance(array, (str, IdentifierNode, VariableNode)):
                return None
            if self.expression_name(array) != name:
                return None
            index = getattr(value, "index", getattr(value, "index_expr", None))
            interval = self.glsl_storage_pointer_array_index_interval(
                binding,
                index,
                value,
                interval_overrides=interval_overrides,
            )
            return array, set(range(interval[0], interval[1] + 1))

        def require_expression(value, assigned):
            if value is None:
                return
            if isinstance(value, str):
                if value == name and assigned != expected:
                    fail(expected - assigned, value)
                return
            if isinstance(value, (int, float, bool)):
                return
            nodes = list(self.walk_ast(value))
            suppressed_identifiers = set()
            for child in nodes:
                access = direct_access(child)
                if access is None:
                    continue
                array, indices = access
                if isinstance(array, (IdentifierNode, VariableNode)):
                    suppressed_identifiers.add(id(array))
                missing = indices - assigned
                if missing:
                    fail(missing, child)
            for child in nodes:
                if not isinstance(child, (IdentifierNode, VariableNode)):
                    continue
                if id(child) in suppressed_identifiers:
                    continue
                if getattr(child, "name", None) == name and assigned != expected:
                    fail(expected - assigned, child)

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
                if id(value) == id(declaration):
                    assigned.clear()
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
                access = direct_access(target)
                require_expression(right, assigned)
                if access is not None:
                    _array, indices = access
                    index = getattr(
                        target,
                        "index",
                        getattr(target, "index_expr", None),
                    )
                    require_expression(index, assigned)
                    if operator == "=":
                        # A non-singleton interval proves only that this
                        # invocation writes one of the possible elements. It
                        # does not definitely initialize every element in the
                        # interval. Exact serial-loop iterations are narrowed
                        # to singleton intervals by ``interval_overrides``.
                        if len(indices) == 1:
                            assigned.update(indices)
                    else:
                        missing = indices - assigned
                        if missing:
                            fail(missing, target)
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
                iteration_values = (
                    self.glsl_storage_pointer_array_loop_iteration_values(value)
                )
                loop_state = set(assigned)
                if iteration_values is None:
                    visit_sequence(getattr(value, "body", None), loop_state)
                    visit(getattr(value, "update", None), loop_state)
                    return assigned

                loop_name = getattr(getattr(value, "init", None), "name", None)
                saved_overrides = dict(interval_overrides)
                try:
                    for iteration_value in iteration_values:
                        interval_overrides[loop_name] = (
                            iteration_value,
                            iteration_value,
                        )
                        visit_sequence(getattr(value, "body", None), loop_state)
                        visit(getattr(value, "update", None), loop_state)
                finally:
                    interval_overrides.clear()
                    interval_overrides.update(saved_overrides)
                assigned.clear()
                assigned.update(loop_state)
                return assigned
            if isinstance(value, ForInNode):
                require_expression(getattr(value, "iterable", None), assigned)
                visit_sequence(getattr(value, "body", None), set(assigned))
                return assigned
            if isinstance(value, DoWhileNode):
                body_state = set(assigned)
                visit_sequence(getattr(value, "body", None), body_state)
                require_expression(getattr(value, "condition", None), body_state)
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
            if isinstance(value, ReturnNode):
                require_expression(
                    getattr(value, "value", getattr(value, "expression", None)),
                    assigned,
                )
                return assigned

            require_expression(value, assigned)
            return assigned

        visit_sequence(getattr(function, "body", []), set())

    def prepare_glsl_storage_pointer_array_aliases(self, function, aliases):
        """Prove local fixed pointer arrays have one backing and full initialization."""

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
            if hasattr(node, "_glsl_storage_pointer_array_binding"):
                delattr(node, "_glsl_storage_pointer_array_binding")
            contract = self.glsl_storage_pointer_array_contract(node)
            if name and contract is not None:
                candidates[name] = (node, contract)

        for name, (declaration, contract) in candidates.items():
            if getattr(declaration, "initial_value", None) is not None:
                raise OpenGLStoragePointerError(
                    "OpenGL local storage pointer-array aggregate initializers are "
                    "not supported",
                    function_name=getattr(function, "name", None),
                    parameter_name=name,
                    reason="pointer-array-aggregate-initializer-unsupported",
                    source_location=getattr(declaration, "source_location", None),
                )
            if declaration_counts.get(name) != 1:
                continue
            extent = contract.get("extent")
            if not isinstance(extent, int):
                raise OpenGLStoragePointerError(
                    "OpenGL fixed storage-pointer-array lowering requires a "
                    f"positive compile-time extent for '{name}'",
                    function_name=getattr(function, "name", None),
                    parameter_name=name,
                    reason="pointer-array-extent-unresolved",
                    source_location=getattr(declaration, "source_location", None),
                )
            assignments = []
            for node in nodes:
                if not isinstance(node, AssignmentNode):
                    continue
                target = getattr(node, "target", getattr(node, "left", None))
                if not isinstance(target, ArrayAccessNode):
                    continue
                array = getattr(target, "array", getattr(target, "array_expr", None))
                if not isinstance(array, (str, IdentifierNode, VariableNode)):
                    continue
                if self.expression_name(array) == name:
                    assignments.append(node)
            if not assignments:
                continue

            visible_aliases = dict(aliases)
            prepared_binding = None
            covered = set()
            unresolved = []
            for assignment in assignments:
                operator = self.map_operator(
                    getattr(assignment, "operator", getattr(assignment, "op", "="))
                )
                if operator != "=":
                    unresolved.append(assignment)
                    continue
                target = getattr(
                    assignment, "target", getattr(assignment, "left", None)
                )
                index = getattr(target, "index", getattr(target, "index_expr", None))
                right = getattr(assignment, "value", getattr(assignment, "right", None))
                replacement = self.glsl_storage_pointer_binding(right, visible_aliases)
                if replacement is None:
                    unresolved.append(assignment)
                    continue
                if prepared_binding is None:
                    prepared_binding = {
                        **replacement,
                        **contract,
                        "kind": "storage-pointer-array",
                        "source_name": name,
                        "parameter_name": name,
                        "offset_array": f"__crossgl_{name}_offsets",
                        "offset_expressions": [],
                        "resource_root": False,
                    }
                    visible_aliases[name] = prepared_binding
                if replacement.get("root") != prepared_binding.get(
                    "root"
                ) or replacement.get("element_type") != contract.get("element_type"):
                    raise OpenGLStoragePointerError(
                        "OpenGL fixed storage-pointer array "
                        f"'{name}' does not resolve to one same-typed backing buffer",
                        function_name=getattr(function, "name", None),
                        parameter_name=name,
                        reason="pointer-array-backing-mismatch",
                        source_location=getattr(assignment, "source_location", None),
                    )
                self.validate_glsl_storage_pointer_binding_access(
                    replacement,
                    contract.get("required_access") or "read_write",
                    assignment,
                )
                interval = self.glsl_storage_pointer_array_index_interval(
                    prepared_binding, index, target
                )
                indices = set(range(interval[0], interval[1] + 1))
                covered.update(indices)

            if prepared_binding is None:
                # The function still has an unbound pointer parameter. A later
                # concrete resource specialization will repeat this proof.
                continue
            if unresolved:
                first = unresolved[0]
                raise OpenGLStoragePointerError(
                    "OpenGL fixed storage-pointer array "
                    f"'{name}' has an assignment without a proven same-backing "
                    "storage view",
                    function_name=getattr(function, "name", None),
                    parameter_name=name,
                    reason="pointer-array-assignment-unresolved",
                    source_location=getattr(first, "source_location", None),
                )
            expected = set(range(extent))
            if covered != expected:
                missing = sorted(expected - covered)
                raise OpenGLStoragePointerError(
                    "OpenGL fixed storage-pointer array "
                    f"'{name}' is not definitely initialized for elements {missing}",
                    function_name=getattr(function, "name", None),
                    parameter_name=name,
                    reason="pointer-array-initialization-unproven",
                    source_location=getattr(declaration, "source_location", None),
                )

            self.validate_glsl_storage_pointer_array_initialization(
                function,
                declaration,
                prepared_binding,
            )
            declaration._glsl_storage_pointer_array_binding = prepared_binding

    def glsl_storage_pointer_declaration(self, node, aliases=None):
        if self.glsl_storage_pointer_pointee_type(node) is not None:
            return True
        if aliases is None:
            return False

        raw_type = getattr(
            node,
            "param_type",
            getattr(node, "var_type", getattr(node, "vtype", None)),
        )
        raw_type_name = self.type_name_string(raw_type)
        if not isinstance(raw_type_name, str) or raw_type_name.strip() != "auto":
            return False
        return (
            self.glsl_storage_pointer_binding(
                getattr(node, "initial_value", None), aliases
            )
            is not None
        )
