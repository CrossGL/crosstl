"""Storage pointer bindings in the GLSL generator: storage pointer parameters, binding offsets, workgroup arrays and struct word array layouts."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    ForInNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    MemberAccessNode,
    PointerAccessNode,
    PointerReinterpretNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import (
    evaluate_literal_int_expression,
    parse_array_type,
    split_array_type_suffix,
)
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_value_arguments,
)
from .glsl_guarded_offsets import composed_offset_range
from .index_range_contracts import IntegerRange
from .opengl_values import OpenGLWorkgroupPointerError
from .pointer_reinterpret import (
    PointerReinterpretationError,
    same_scalar_lane_view,
    scalar_storage_layout,
)
from .storage_pointer_local_aliases import bind_local_storage_pointers
from .storage_word_views import reinterpret_source_layout

_WORKGROUP_VIEW_SPACES = (None, "threadgroup", "workgroup", "shared", "groupshared")
_VIEW_TYPE_QUALIFIERS = {"const", "volatile", "threadgroup", "restrict"}


def _same_view_element_type(source_type, view_type):
    """Whether a pointer view of ``view_type`` keeps the ``source_type`` elements unchanged."""
    source_layout = scalar_storage_layout(source_type)
    view_layout = scalar_storage_layout(view_type)
    if source_layout is not None or view_layout is not None:
        return source_layout == view_layout

    def unqualified(type_name):
        words = str(type_name or "").split()
        return " ".join(word for word in words if word not in _VIEW_TYPE_QUALIFIERS)

    return bool(unqualified(source_type)) and unqualified(source_type) == unqualified(
        view_type
    )


class OpenGLStoragePointerBindingMixin:
    """Storage pointer binding members of ``GLSLCodeGen``."""

    def glsl_function_has_storage_pointer_parameter(self, function):
        return any(
            self.glsl_storage_pointer_declaration(parameter)
            or self.glsl_storage_pointer_array_contract(parameter) is not None
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        )

    def glsl_function_forwards_structured_buffer_to_storage_pointer(self, function):
        aliases = self.glsl_function_structured_buffer_parameter_aliases(function)
        if not aliases:
            return False

        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            callee = self.function_definitions.get(self.function_call_name(node))
            if callee is None:
                continue
            callee_parameters = list(
                getattr(callee, "parameters", getattr(callee, "params", [])) or []
            )
            arguments = list(
                getattr(node, "arguments", getattr(node, "args", [])) or []
            )
            for callee_parameter, argument in zip(callee_parameters, arguments):
                if not self.glsl_storage_pointer_declaration(callee_parameter):
                    continue
                if self.glsl_storage_pointer_binding(argument, aliases) is not None:
                    return True
        return False

    def glsl_function_uses_structured_buffer_pointer_alias(self, function):
        aliases = self.glsl_function_structured_buffer_parameter_aliases(function)
        if not aliases:
            return False

        for node in self.walk_ast(getattr(function, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            if self.glsl_storage_pointer_declaration(node, aliases):
                return True
        return False

    def glsl_function_structured_buffer_parameter_aliases(self, function):
        aliases = {}
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            param_type = self.type_name_string(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            if self.structured_buffer_type_name(param_type) not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
            }:
                continue
            name = getattr(parameter, "name", None)
            if not name:
                continue
            access = self.structured_buffer_effective_access(param_type, parameter)
            source_element_type = self.structured_buffer_source_element_type(param_type)
            aliases[name] = {
                "kind": "storage-pointer",
                "root": name,
                "offset": 0,
                "element_type": self.map_type(
                    self.structured_buffer_element_type(param_type)
                ),
                "source_element_type": source_element_type,
                "access": access or "read_write",
            }
        return aliases

    def glsl_storage_pointer_mutation_binding(self, expression, aliases):
        if isinstance(expression, ArrayAccessNode):
            binding = self.glsl_storage_pointer_binding(expression.array, aliases)
            if binding is not None:
                return binding
            return self.glsl_storage_pointer_mutation_binding(expression.array, aliases)
        if (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "*"
        ):
            return self.glsl_storage_pointer_binding(expression.operand, aliases)
        if isinstance(expression, PointerAccessNode):
            return self.glsl_storage_pointer_binding(expression.pointer_expr, aliases)
        if isinstance(expression, MemberAccessNode):
            binding = self.glsl_storage_pointer_binding(expression.object, aliases)
            if binding is not None:
                return binding
            return self.glsl_storage_pointer_mutation_binding(
                expression.object, aliases
            )
        return None

    def glsl_storage_pointer_parameter_required_access(self, function, parameter):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if qualifiers.intersection({"const", "constant", "readonly", "in"}):
            return "read"

        parameter_name = getattr(parameter, "name", None)
        if not parameter_name:
            return "read_write"
        cache_key = (id(function), parameter_name)
        cached = self.glsl_storage_pointer_access_requirement_cache.get(cache_key)
        if cached is not None:
            return cached
        if cache_key in self.glsl_storage_pointer_access_requirement_active:
            return "read_write"

        self.glsl_storage_pointer_access_requirement_active.add(cache_key)
        try:
            element_type = self.glsl_storage_pointer_element_type(parameter)
            aliases = {
                parameter_name: {
                    "kind": "storage-pointer",
                    "root": parameter_name,
                    "offset": 0,
                    "element_type": element_type,
                    "access": "read_write",
                }
            }
            nodes = list(self.walk_ast(getattr(function, "body", [])))
            if not bind_local_storage_pointers(
                [node for node in nodes if isinstance(node, VariableNode)],
                aliases,
                self.glsl_storage_pointer_declaration,
                self.glsl_storage_pointer_binding,
                preferred_root=parameter_name,
            ):
                self.glsl_storage_pointer_access_requirement_cache[cache_key] = (
                    "read_write"
                )
                return "read_write"

            for node in nodes:
                if isinstance(node, AssignmentNode):
                    target = getattr(node, "target", getattr(node, "left", None))
                    binding = self.glsl_storage_pointer_mutation_binding(
                        target, aliases
                    )
                    if binding is not None and binding.get("root") == parameter_name:
                        self.glsl_storage_pointer_access_requirement_cache[
                            cache_key
                        ] = "read_write"
                        return "read_write"
                if isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                    "++",
                    "--",
                }:
                    binding = self.glsl_storage_pointer_mutation_binding(
                        node.operand, aliases
                    )
                    if binding is not None and binding.get("root") == parameter_name:
                        self.glsl_storage_pointer_access_requirement_cache[
                            cache_key
                        ] = "read_write"
                        return "read_write"
                if not isinstance(node, FunctionCallNode):
                    continue
                callee = self.function_definitions.get(self.function_call_name(node))
                if callee is None:
                    continue
                callee_parameters = list(
                    getattr(callee, "parameters", getattr(callee, "params", [])) or []
                )
                arguments = list(
                    getattr(node, "arguments", getattr(node, "args", [])) or []
                )
                for callee_parameter, argument in zip(callee_parameters, arguments):
                    binding = self.glsl_storage_pointer_binding(argument, aliases)
                    if binding is None or binding.get("root") != parameter_name:
                        continue
                    if not self.glsl_storage_pointer_declaration(callee_parameter):
                        continue
                    required_access = (
                        self.glsl_storage_pointer_parameter_required_access(
                            callee, callee_parameter
                        )
                    )
                    if required_access != "read":
                        self.glsl_storage_pointer_access_requirement_cache[
                            cache_key
                        ] = "read_write"
                        return "read_write"

            self.glsl_storage_pointer_access_requirement_cache[cache_key] = "read"
            return "read"
        finally:
            self.glsl_storage_pointer_access_requirement_active.discard(cache_key)

    @staticmethod
    def glsl_storage_pointer_offset_parameter_direction(parameter):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if "inout" in qualifiers:
            return "inout"
        if "out" in qualifiers:
            return "out"
        return None

    def glsl_function_has_target_pointer_parameter(self, function):
        return (
            self.glsl_function_has_workgroup_pointer_parameter(function)
            or self.glsl_function_has_storage_pointer_parameter(function)
            or self.glsl_function_forwards_structured_buffer_to_storage_pointer(
                function
            )
            or self.glsl_function_uses_structured_buffer_pointer_alias(function)
        )

    @staticmethod
    def glsl_array_suffix_rank(array_suffix):
        """Count the top-level ``[...]`` groups of an array type suffix."""
        rank = depth = 0
        for character in array_suffix:
            if character == "[":
                rank += depth == 0
                depth += 1
            elif character == "]":
                depth -= 1
        return rank

    def glsl_shared_storage_binding(self, node, root_name, *, entry_owner="module"):
        raw_type = getattr(node, "var_type", getattr(node, "vtype", None))
        declared_type = self.local_variable_declared_type(node)
        element_type, array_suffix = split_array_type_suffix(str(declared_type or ""))
        if not array_suffix:
            return None
        # A workgroup pointer addresses one flat element sequence, so its
        # subscripts add up as offsets. An array of arrays such as MLX's
        # Winograd ``threadgroup T Ws[BO][R][R][BC]`` keeps GLSL's own
        # arrays-of-arrays subscripts: adding them would alias distinct rows.
        if (
            isinstance(raw_type, ArrayType)
            and isinstance(raw_type.element_type, ArrayType)
        ) or self.glsl_array_suffix_rank(array_suffix) > 1:
            return None

        size_expression = getattr(raw_type, "size", None)
        constants = {
            **getattr(self, "literal_int_constants", {}),
            **getattr(self, "glsl_shared_dimension_constants", {}),
        }
        array_size = evaluate_literal_int_expression(size_expression, constants)
        if array_size is None:
            resolved_type = self.resolve_glsl_shared_array_dimensions(declared_type)
            _resolved_element, array_size = parse_array_type(resolved_type)
        if not isinstance(array_size, int) or isinstance(array_size, bool):
            array_size = None
        elif array_size <= 0:
            array_size = None

        source_element_type = element_type
        if isinstance(raw_type, ArrayType):
            source_element_type = self.type_name_string(raw_type.element_type)

        return {
            "kind": "workgroup-pointer",
            "root": root_name,
            "offset": 0,
            "offset_interval": (0, 0),
            "array_size": array_size,
            "element_type": self.map_type(element_type),
            "source_element_type": source_element_type,
            "entry_owner": entry_owner,
            "backing_declaration": getattr(node, "name", None) or root_name,
            "backing_location": getattr(node, "source_location", None),
        }

    def collect_glsl_global_shared_bindings(self, nodes):
        bindings = {}
        for node in nodes or []:
            if not self.glsl_declaration_has_shared_qualifier(node):
                continue
            if self.glsl_workgroup_pointer_declaration(node):
                raise OpenGLWorkgroupPointerError(
                    "OpenGL cannot declare a module-scope workgroup pointer",
                    parameter_name=getattr(node, "name", None),
                    reason="pointer-storage-declaration",
                    source_location=getattr(node, "source_location", None),
                )
            source_name = self.resource_node_name(node)
            if not source_name:
                continue
            emitted_name = self.glsl_global_identifier_name(source_name)
            binding = self.glsl_shared_storage_binding(node, emitted_name)
            if binding is None:
                continue
            bindings[source_name] = dict(binding)
            bindings[emitted_name] = dict(binding)
        return bindings

    def collect_glsl_function_local_identifier_names(self, function):
        names = {
            getattr(parameter, "name", None)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        }
        for node in self.walk_ast(getattr(function, "body", [])):
            if isinstance(node, (VariableNode, ArrayNode)):
                names.add(getattr(node, "name", None))
            elif isinstance(node, ForInNode):
                pattern = getattr(node, "pattern", None)
                if isinstance(pattern, str):
                    names.add(pattern)
            elif isinstance(node, IdentifierPatternNode):
                name = getattr(node, "name", None)
                if name not in {None, ".."} and "::" not in name:
                    names.add(name)
        return {name for name in names if name}

    def generate_glsl_mutable_storage_pointer_root_declarations(self, function, indent):
        body = getattr(function, "body", [])
        lexical_call_source_types = self.glsl_scalar_array_view_lexical_bindings(
            function
        )["call_source_types"]
        mutated_names = set()
        for node in self.walk_glsl_reachable_ast(body):
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
                if isinstance(target, (str, IdentifierNode, VariableNode)):
                    name = getattr(target, "name", None) or self.expression_name(target)
                    if name in self.current_glsl_storage_pointer_aliases:
                        mutated_names.add(name)
            elif isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                "++",
                "--",
            }:
                operand = node.operand
                if isinstance(operand, (str, IdentifierNode, VariableNode)):
                    name = getattr(operand, "name", None) or self.expression_name(
                        operand
                    )
                    if name in self.current_glsl_storage_pointer_aliases:
                        mutated_names.add(name)
            elif isinstance(node, FunctionCallNode):
                arguments = list(
                    getattr(node, "arguments", getattr(node, "args", [])) or []
                )
                function_name = self.function_call_name(node)
                if not function_name:
                    continue
                specialized_name = generic_function_call_name(
                    self, function_name, arguments
                )
                if specialized_name is not None:
                    arguments = generic_function_value_arguments(
                        self, function_name, arguments
                    )
                    function_name = specialized_name
                if function_name not in self.glsl_resource_specialized_source_names:
                    continue
                specialization = (
                    self.glsl_resource_function_call_specialization_in_scope(
                        function,
                        function_name,
                        arguments,
                        lexical_source_types=lexical_call_source_types.get(
                            id(node), {}
                        ),
                        call_node=node,
                    )
                )
                if specialization is None:
                    continue
                offset_directions = getattr(
                    specialization,
                    "_glsl_storage_pointer_offset_directions",
                    {},
                )
                for index, direction in offset_directions.items():
                    if direction not in {"out", "inout"} or index >= len(arguments):
                        continue
                    argument = arguments[index]
                    if not isinstance(argument, (str, IdentifierNode, VariableNode)):
                        continue
                    raw_name = getattr(argument, "name", argument)
                    rendered_name = self.expression_name(argument)
                    for name in (raw_name, rendered_name):
                        if name in self.current_glsl_storage_pointer_aliases:
                            mutated_names.add(name)
                            break

        code = ""
        for name in sorted(mutated_names):
            binding = self.current_glsl_storage_pointer_aliases.get(name)
            if binding is None:
                continue
            rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
            if self.glsl_storage_pointer_offset_lvalue(rendered_offset):
                continue
            offset_name = self.glsl_synthetic_local_identifier(f"{name}_offset")
            lowered_binding = {
                **binding,
                "offset": offset_name,
                "resource_root": False,
            }
            for alias_name, alias_binding in list(
                self.current_glsl_storage_pointer_aliases.items()
            ):
                if alias_binding.get("kind") != "storage-pointer":
                    continue
                if alias_binding.get("root") != binding.get("root"):
                    continue
                alias_offset = self.glsl_workgroup_pointer_offset_expression(
                    alias_binding
                )
                if alias_offset != rendered_offset:
                    continue
                self.current_glsl_storage_pointer_aliases[alias_name] = {
                    **alias_binding,
                    "offset": offset_name,
                    "resource_root": False,
                }
            self.current_glsl_storage_pointer_aliases[name] = lowered_binding
            self.local_variable_types[offset_name] = "int"
            self.local_variable_source_types[offset_name] = "int"
            code += f"{'    ' * indent}int {offset_name} = int({rendered_offset});\n"
        return code

    @staticmethod
    def glsl_storage_pointer_offset_lvalue(offset):
        text = str(offset or "").strip()
        return re.fullmatch(r"[A-Za-z_]\w*(?:\s*\[[^\]]+\])?", text) is not None

    def glsl_synthetic_local_identifier(self, base_name, used_names=None):
        used = set(used_names or self.current_glsl_synthetic_local_names)
        used.update(self.local_variable_types)
        used.update(self.current_identifier_aliases.values())
        used.update(self.current_glsl_local_identifier_names.values())
        used.update(self.current_stage_declared_names())
        used.update(self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS)
        candidate = self.glsl_unique_identifier(
            self.glsl_sanitized_identifier_base(
                base_name,
                fallback="workgroup_offset",
                extra_reserved=self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS,
            ),
            used,
            extra_reserved=self.GLSL_ALIAS_TARGET_LOCAL_IDENTIFIERS,
        )
        if used_names is None:
            self.current_glsl_synthetic_local_names.add(candidate)
        else:
            used_names.add(candidate)
        return candidate

    def next_glsl_temp_variable(self, base_name):
        index = self.match_temp_variable_index
        self.match_temp_variable_index += 1
        return self.glsl_synthetic_local_identifier(f"cgl_{base_name}_{index}")

    def glsl_pointer_offset_sum(self, base, delta):
        if base in {None, 0, "0"}:
            return delta
        if delta in {None, 0, "0"}:
            return base
        node = BinaryOpNode(base, "+", delta)
        # Pointer arithmetic adds each delta's value exactly; the sum itself is
        # not a source-typed integer expression that could wrap.
        node._glsl_pointer_offset_composition = True
        return node

    def glsl_pointer_offset_negation(self, delta):
        node = UnaryOpNode("-", delta)
        node._glsl_pointer_offset_composition = True
        return node

    def glsl_workgroup_pointer_offset_interval(self, binding, intervals=None):
        interval = binding.get("offset_interval")
        if interval is not None:
            return interval
        return self.glsl_private_pointer_interval(
            binding.get("offset", 0),
            intervals or {},
            self.glsl_active_literal_int_constants(),
        )

    def glsl_workgroup_pointer_binding(self, expression, aliases, intervals=None):
        if expression is None:
            return None
        if intervals is None:
            intervals = getattr(expression, "_glsl_control_flow_intervals", None)

        if isinstance(expression, PointerReinterpretNode):
            return self.glsl_workgroup_pointer_view_binding(
                expression, aliases, intervals
            )

        if isinstance(expression, ArrayAccessNode):
            binding = self.glsl_workgroup_pointer_binding(
                expression.array, aliases, intervals
            )
            if binding is None:
                return None
            base_interval = self.glsl_workgroup_pointer_offset_interval(
                binding, intervals
            )
            delta_interval = self.glsl_private_pointer_interval(
                expression.index,
                intervals or {},
                self.glsl_active_literal_int_constants(),
            )
            binding["offset"] = self.glsl_pointer_offset_sum(
                binding.get("offset"), expression.index
            )
            binding["offset_interval"] = (
                None
                if base_interval is None or delta_interval is None
                else (
                    base_interval[0] + delta_interval[0],
                    base_interval[1] + delta_interval[1],
                )
            )
            return binding

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            raw_name = getattr(expression, "name", expression)
            name = self.expression_name(expression)
            for candidate in (raw_name, name):
                if candidate and candidate in aliases:
                    binding = aliases[candidate]
                    if binding.get("kind") == "workgroup-pointer":
                        return dict(binding)
            if raw_name == "nullptr" or name == "nullptr":
                return None

        if isinstance(expression, TernaryOpNode):
            condition = evaluate_literal_int_expression(
                expression.condition,
                self.glsl_active_literal_int_constants(),
            )
            if condition is None:
                return None
            selected = expression.true_expr if condition else expression.false_expr
            return self.glsl_workgroup_pointer_binding(selected, aliases, intervals)

        if (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "&"
        ):
            operand = expression.operand
            if isinstance(operand, ArrayAccessNode):
                return self.glsl_workgroup_pointer_binding(operand, aliases, intervals)
            if (
                self.glsl_workgroup_pointer_binding(operand, aliases, intervals)
                is not None
            ):
                raise OpenGLWorkgroupPointerError(
                    "OpenGL cannot lower the address of a workgroup pointer",
                    parameter_name=getattr(operand, "name", None),
                    reason="address-of-pointer",
                    source_location=getattr(expression, "source_location", None),
                )
            return None

        if isinstance(expression, BinaryOpNode) and self.map_operator(
            expression.op
        ) in {
            "+",
            "-",
        }:
            operator = self.map_operator(expression.op)
            binding = self.glsl_workgroup_pointer_binding(
                expression.left, aliases, intervals
            )
            delta = expression.right
            if binding is None and operator == "+":
                binding = self.glsl_workgroup_pointer_binding(
                    expression.right, aliases, intervals
                )
                delta = expression.left
            if binding is None:
                return None
            base_interval = self.glsl_workgroup_pointer_offset_interval(
                binding, intervals
            )
            delta_interval = self.glsl_private_pointer_interval(
                delta,
                intervals or {},
                self.glsl_active_literal_int_constants(),
            )
            if operator == "-":
                delta = self.glsl_pointer_offset_negation(delta)
                if delta_interval is not None:
                    delta_interval = (-delta_interval[1], -delta_interval[0])
            binding["offset"] = self.glsl_pointer_offset_sum(
                binding.get("offset"), delta
            )
            binding["offset_interval"] = (
                None
                if base_interval is None or delta_interval is None
                else (
                    base_interval[0] + delta_interval[0],
                    base_interval[1] + delta_interval[1],
                )
            )
            return binding

        return None

    def glsl_workgroup_pointer_view_binding(self, expression, aliases, intervals):
        """The binding of ``(threadgroup T*)tile`` when ``T`` is the element type of ``tile``.

        Such a view is the shared array itself. GLSL cannot alias a ``shared`` array, so a view that changes the
        element type is refused instead of reading the array's values as another type.
        """
        target_type = getattr(expression, "target_type", None)
        if getattr(target_type, "address_space", None) not in _WORKGROUP_VIEW_SPACES:
            return None
        binding = self.glsl_workgroup_pointer_binding(
            expression.expression, aliases, intervals
        )
        if binding is None:
            return None
        view_type = self.type_name_string(getattr(target_type, "pointee_type", None))
        source_type = binding.get("source_element_type") or binding.get("element_type")
        if _same_view_element_type(source_type, view_type):
            return binding
        raise PointerReinterpretationError(
            f"OpenGL cannot view the {source_type} workgroup array "
            f"'{binding.get('root')}' through a {view_type} pointer: a GLSL "
            "shared array has one element type and cannot be aliased",
            source_type=source_type,
            target_type=view_type,
            address_space="workgroup",
            target_backend="opengl",
            reason="workgroup-view-type-change",
            source_location=getattr(expression, "source_location", None),
        )

    def glsl_storage_pointer_binding(self, expression, aliases):
        if expression is None:
            return None

        pointer_array_element = self.glsl_storage_pointer_array_element_binding(
            expression, aliases
        )
        if pointer_array_element is not None:
            return pointer_array_element

        if isinstance(expression, PointerReinterpretNode):
            binding = self.glsl_storage_pointer_binding(expression.expression, aliases)
            if binding is None:
                return None
            return self.glsl_pointer_reinterpret_binding(expression, binding)

        if isinstance(expression, (str, IdentifierNode, VariableNode)):
            raw_name = getattr(expression, "name", expression)
            name = self.expression_name(expression)
            for candidate in (raw_name, name):
                if candidate and candidate in aliases:
                    binding = aliases[candidate]
                    if binding.get("kind") == "storage-pointer":
                        binding = dict(binding)
                        offset = binding.get("offset")
                        bounds = getattr(
                            expression, "_glsl_control_flow_intervals", {}
                        ).get(raw_name)
                        if (
                            isinstance(expression, IdentifierNode)
                            and isinstance(offset, str)
                            and offset.isidentifier()
                            and bounds is not None
                            and -(1 << 31) <= bounds[0] <= bounds[1] < (1 << 31)
                            and "pointer_reinterpretation" not in binding
                            and "byte_offset" not in binding
                        ):
                            # Keep source facts on this use, never on the alias
                            # table: later mutations require a fresh flow proof.
                            lowered = IdentifierNode(offset)
                            lowered._glsl_control_flow_intervals = {offset: bounds}
                            lowered.source_location = getattr(
                                expression, "source_location", None
                            )
                            binding["offset"] = lowered
                        return binding
            return None

        if isinstance(expression, TernaryOpNode):
            condition = evaluate_literal_int_expression(
                expression.condition,
                self.glsl_active_literal_int_constants(),
            )
            if condition is None:
                return None
            selected = expression.true_expr if condition else expression.false_expr
            return self.glsl_storage_pointer_binding(selected, aliases)

        if (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "&"
        ):
            operand = expression.operand
            if isinstance(operand, ArrayAccessNode):
                binding = self.glsl_storage_pointer_binding(operand.array, aliases)
                if binding is None:
                    return None
                binding["offset"] = self.glsl_pointer_offset_sum(
                    binding.get("offset"), operand.index
                )
                info = self.glsl_value_type_info(
                    self.glsl_source_expression_type(operand.index)
                )
                if info is not None and info["width"] == 1 and info["bits"] > 32:
                    binding["wide_offset"] = True
                return binding
            return None

        if isinstance(expression, BinaryOpNode) and self.map_operator(
            expression.op
        ) in {"+", "-"}:
            operator = self.map_operator(expression.op)
            binding = self.glsl_storage_pointer_binding(expression.left, aliases)
            delta = expression.right
            if binding is None and operator == "+":
                binding = self.glsl_storage_pointer_binding(expression.right, aliases)
                delta = expression.left
            if binding is None:
                return None
            info = self.glsl_value_type_info(self.glsl_source_expression_type(delta))
            if info is not None and info["width"] == 1 and info["bits"] > 32:
                binding["wide_offset"] = True
            if operator == "-":
                delta = self.glsl_pointer_offset_negation(delta)
            binding["offset"] = self.glsl_pointer_offset_sum(
                binding.get("offset"), delta
            )
            # Preserve the source pointer-arithmetic expression as the explicit
            # range-contract key. The lowered offset can contain synthetic
            # aliases or specialization names that project authors cannot
            # express against the source program.
            binding["index_assertion_expression"] = expression
            return binding

        return None

    def glsl_pointer_reinterpret_binding(self, expression, binding):
        target_type = getattr(expression.target_type, "pointee_type", None)
        target_type_name = self.strip_metal_parameter_indirection(
            self.type_name_string(target_type)
        )
        mapped_target_type = self.map_type(target_type_name)
        view_type_name = (
            binding.get("view_element_type")
            or binding.get("source_element_type")
            or binding.get("element_type")
        )
        source_type_name = binding.get("source_element_type") or view_type_name
        # A one-scalar wrapper record backs the view with its member's words.
        view_layout, _ = reinterpret_source_layout(view_type_name, self.structs_by_name)
        source_layout, source_member = reinterpret_source_layout(
            source_type_name, self.structs_by_name
        )
        target_layout = scalar_storage_layout(target_type_name)
        lane_gather = self.glsl_same_scalar_lane_gather_binding(
            binding,
            target_type,
            source_type_name=source_type_name,
            source_layout=source_layout,
            view_layout=view_layout,
        )
        if lane_gather is not None:
            return lane_gather
        target_width = 1
        if (
            target_layout is None
            and source_layout is not None
            and source_layout.bit_width == 32
            and self.is_vector_value_type(mapped_target_type)
        ):
            target_layout = scalar_storage_layout(
                self.vector_component_type(mapped_target_type)
            )
            target_width = int(mapped_target_type[-1])
        if (
            view_layout is None
            or source_layout is None
            or target_layout is None
            or source_layout.bit_width != 32
            or target_layout.bit_width not in {8, 16, 32}
            or target_width > 1
            and target_layout.bit_width != 32
            or target_layout.kind == "floating"
            and target_layout.bit_width != 32
        ):
            raise PointerReinterpretationError(
                "OpenGL storage pointer reinterpretation requires a 32-bit "
                "scalar backing element and either an 8-, 16-, or 32-bit scalar "
                "view or a 2- to 4-lane 32-bit vector view",
                source_type=source_type_name,
                target_type=target_type_name,
                address_space="storage",
                alignment=target_layout.byte_width if target_layout else None,
                access=binding.get("access"),
                reason="unsupported-scalar-layout",
                source_location=getattr(expression, "source_location", None),
            )

        if target_width == 1 and target_layout == view_layout and not source_member:
            return {
                **binding,
                "element_type": self.map_type(target_type_name),
                "view_element_type": target_type_name,
            }

        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        scaled_offset = rendered_offset
        if view_layout.byte_width != 1 and rendered_offset not in {"0", "0u"}:
            scaled_offset = f"({rendered_offset} * {view_layout.byte_width})"
        existing_byte_offset = str(binding.get("byte_offset", "0"))
        if existing_byte_offset in {"0", "0u"}:
            byte_offset = scaled_offset
        elif scaled_offset in {"0", "0u"}:
            byte_offset = existing_byte_offset
        else:
            byte_offset = f"({existing_byte_offset} + {scaled_offset})"

        return {
            **binding,
            "offset": 0,
            "byte_offset": byte_offset,
            "element_type": self.map_type(target_type_name),
            "view_element_type": target_type_name,
            "source_element_type": source_type_name,
            "pointer_reinterpretation": {
                "source_layout": source_layout,
                "target_layout": target_layout,
                "target_width": target_width,
                "source_member": source_member,
            },
        }

    def glsl_same_scalar_lane_gather_binding(
        self,
        binding,
        target_type,
        *,
        source_type_name,
        source_layout,
        view_layout,
    ):
        """Gather a ``vec<T, N>`` view of a widened 8- or 16-bit buffer.

        OpenGL stores each 8- or 16-bit storage element in one 32-bit array
        element, so a vector view of a ``T`` buffer is the N consecutive array
        elements. Bit-level views of those widened words stay rejected.
        """

        if source_layout is None or source_layout.bit_width not in {8, 16}:
            return None
        lane_view = same_scalar_lane_view(target_type, source_layout)
        if (
            lane_view is None
            or view_layout != source_layout
            or binding.get("pointer_reinterpretation") is not None
            or str(binding.get("byte_offset", "0")) not in {"0", "0u"}
            or binding.get("element_type") != self.map_type(source_type_name)
        ):
            return None
        lanes, component = lane_view
        vector_type = f"vec<{component}, {lanes}>"
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        byte_offset = rendered_offset
        if source_layout.byte_width != 1 and rendered_offset not in {"0", "0u"}:
            byte_offset = f"({rendered_offset} * {source_layout.byte_width})"
        return {
            **binding,
            "offset": 0,
            "byte_offset": byte_offset,
            "element_type": self.map_type(vector_type),
            "view_element_type": vector_type,
            "source_element_type": source_type_name,
            "pointer_reinterpretation": {
                "source_layout": source_layout,
                "target_layout": source_layout,
                "target_width": lanes,
                "lane_gather": True,
            },
        }

    def glsl_storage_struct_reinterpret_node(self, expression):
        if not (
            isinstance(expression, UnaryOpNode)
            and self.map_operator(expression.op) == "*"
            and isinstance(expression.operand, PointerReinterpretNode)
        ):
            return None
        return expression.operand

    def glsl_expression_references_storage_pointer(self, expression, aliases):
        for node in self.walk_ast(expression):
            if not isinstance(node, (IdentifierNode, VariableNode)):
                continue
            binding = aliases.get(getattr(node, "name", None))
            if binding is not None and binding.get("kind") == "storage-pointer":
                return True
        return False

    def glsl_storage_struct_reinterpret_error(
        self,
        expression,
        *,
        source_type,
        target_type,
        reason,
        detail,
        access="read",
        alignment=4,
    ):
        raise PointerReinterpretationError(
            "OpenGL cannot preserve a storage-buffer aggregate reinterpretation "
            f"from '{source_type or 'unknown'}' to "
            f"'{target_type or 'unknown'}': {detail}",
            source_type=source_type,
            target_type=target_type,
            address_space="storage",
            alignment=alignment,
            access=access,
            target_backend="opengl",
            reason=reason,
            source_location=getattr(expression, "source_location", None),
        )

    def glsl_storage_struct_byte_array_layout(
        self,
        target_type,
        expression,
        source_type,
        *,
        diagnose=True,
    ):
        def reject(reason, detail):
            if diagnose:
                self.glsl_storage_struct_reinterpret_error(
                    expression,
                    source_type=source_type,
                    target_type=target_type,
                    reason=reason,
                    detail=detail,
                )
            return None

        mapped_target_type = self.map_type(target_type)
        struct = self.glsl_struct_node(mapped_target_type)
        if struct is None:
            return reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "the aggregate target is not a concrete struct",
            )
        members = [
            member
            for member in getattr(struct, "members", []) or []
            if not self.glsl_static_struct_member(member)
        ]
        if (
            len(members) != 1
            or getattr(struct, "generic_params", None)
            or getattr(struct, "inheritance", None)
        ):
            return reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "only a concrete struct with one fixed byte-array member is supported",
            )

        alignment = 1
        attributes = list(getattr(struct, "attributes", None) or [])
        if attributes:
            attribute = attributes[0] if len(attributes) == 1 else None
            arguments = (
                list(getattr(attribute, "arguments", None) or []) if attribute else []
            )
            if (
                attribute is None
                or getattr(attribute, "name", None) != "metal_alignas"
                or len(arguments) != 1
            ):
                return reject(
                    "unsupported-storage-workgroup-aggregate-layout",
                    "the byte-array struct has unsupported layout attributes",
                )
            alignment = self.literal_int_value(
                arguments[0],
                self.glsl_active_literal_int_constants(),
            )
            if (
                not isinstance(alignment, int)
                or isinstance(alignment, bool)
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                return reject(
                    "storage-workgroup-aggregate-layout-unresolved",
                    "the byte-array struct alignment is not a fixed positive power of two",
                )

        member = members[0]
        if getattr(member, "attributes", None) or getattr(
            member, "resource_qualifiers", None
        ):
            return reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "the byte-array member has unsupported attributes or qualifiers",
            )
        member_type = getattr(member, "member_type", None)
        if not isinstance(member_type, ArrayType):
            return reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "the aggregate member is not a fixed byte array",
            )
        extent = self.literal_int_value(
            member_type.size,
            self.glsl_active_literal_int_constants(),
        )
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            return reject(
                "storage-workgroup-aggregate-layout-unresolved",
                "the byte-array extent is not a fixed positive integer",
            )
        member_element_type = self.glsl_normalized_source_type(member_type.element_type)
        member_layout = scalar_storage_layout(member_element_type)
        if (
            member_layout is None
            or member_layout.kind != "integer"
            or member_layout.bit_width != 8
        ):
            return reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "the aggregate member element does not have an exact 8-bit integer layout",
            )
        byte_extent = extent * member_layout.byte_width
        if byte_extent % alignment != 0:
            return reject(
                "unsupported-storage-workgroup-aggregate-layout",
                "the explicit struct alignment would add unsupported tail padding",
            )
        return {
            "mapped_type": mapped_target_type,
            "member_layout": member_layout,
            "extent": extent,
            "byte_extent": byte_extent,
            "alignment": alignment,
        }

    def glsl_workgroup_byte_array_copy_word_count(
        self,
        reinterpretation,
        aliases,
        intervals=None,
    ):
        pointer_type = getattr(reinterpretation, "target_type", None)
        address_space = str(getattr(pointer_type, "address_space", None) or "").lower()
        if address_space not in {"threadgroup", "workgroup"}:
            return None
        target_type = self.glsl_normalized_source_type(
            getattr(pointer_type, "pointee_type", None)
        )
        if not target_type:
            return None
        binding = self.glsl_workgroup_pointer_binding(
            reinterpretation.expression,
            aliases,
            intervals,
        )
        if binding is None:
            return None
        source_type = self.strip_metal_parameter_indirection(
            self.type_name_string(
                binding.get("source_element_type") or binding.get("element_type")
            )
        )
        source_layout = scalar_storage_layout(source_type)
        layout = self.glsl_storage_struct_byte_array_layout(
            target_type,
            reinterpretation,
            source_type,
            diagnose=False,
        )
        if (
            layout is None
            or source_layout is None
            or source_layout.bit_width not in {16, 32}
            or layout["alignment"] > source_layout.byte_width
            or layout["byte_extent"] % source_layout.byte_width != 0
        ):
            return None
        return layout["byte_extent"] // source_layout.byte_width

    def glsl_storage_pointer_offset_range(self, binding, expression):
        if isinstance(expression, ArrayAccessNode) and (
            "offset_expressions" in binding
            and "pointer_reinterpretation" not in binding
            and "byte_offset" not in binding
        ):
            bounds = self.glsl_checked_private_pointer_interval(
                expression,
                getattr(expression, "_glsl_control_flow_intervals", {}),
                self.glsl_active_literal_int_constants(),
            )
            if bounds is not None:
                return IntegerRange(*bounds)
        assertion = binding.get("index_assertion_expression")
        if assertion is not None:
            asserted = self.glsl_index_asserted_range(assertion)
            if asserted is not None:
                return asserted
        offset = binding.get("offset", 0)
        intervals = getattr(expression, "_glsl_control_flow_intervals", {})
        offset_range = self.glsl_index_value_range(offset, intervals=intervals)
        if offset_range is None:
            # A composed offset is the exact sum of its deltas, also where a
            # source-typed negation of one delta would wrap.
            offset_range = composed_offset_range(self, offset)
        return offset_range

    @staticmethod
    def glsl_storage_to_workgroup_copy_index(offset, index):
        if offset in {"0", "0u"}:
            return str(index)
        if index == 0:
            return f"int({offset})"
        return f"(int({offset}) + {index})"

    def glsl_storage_to_workgroup_byte_array_copy(
        self,
        left_expression,
        right_expression,
        operator,
        *,
        statement_context,
    ):
        contract = self.glsl_storage_to_workgroup_byte_array_copy_contract(
            left_expression,
            right_expression,
            operator,
            statement_context=statement_context,
        )
        if contract is None:
            return None
        destination_root = contract["destination_binding"]["root"]
        source_root = contract["source_binding"]["root"]
        assignments = []
        for index in range(contract["word_count"]):
            destination_index = self.glsl_storage_to_workgroup_copy_index(
                contract["destination_offset"],
                index,
            )
            source_index = self.glsl_storage_to_workgroup_copy_index(
                contract["source_offset"],
                index,
            )
            assignments.append(
                f"{destination_root}[{destination_index}] = "
                f"{source_root}[{source_index}]"
            )
        return "\n".join(assignments)

    def glsl_storage_struct_word_array_layout(
        self,
        target_type,
        expression,
        source_type,
    ):
        mapped_target_type = self.map_type(target_type)
        struct = self.glsl_struct_node(mapped_target_type)
        if struct is None:
            return None
        members = [
            member
            for member in getattr(struct, "members", []) or []
            if not self.glsl_static_struct_member(member)
        ]
        if (
            len(members) != 1
            or getattr(struct, "generic_params", None)
            or getattr(struct, "inheritance", None)
        ):
            self.glsl_storage_struct_reinterpret_error(
                expression,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-layout",
                detail=(
                    "only a padding-free struct with one homogeneous fixed "
                    "scalar-array member is supported"
                ),
            )
        alignment = 1
        attributes = list(getattr(struct, "attributes", None) or [])
        if attributes:
            attribute = attributes[0] if len(attributes) == 1 else None
            arguments = (
                list(getattr(attribute, "arguments", None) or [])
                if attribute is not None
                else []
            )
            if (
                attribute is None
                or getattr(attribute, "name", None) != "metal_alignas"
                or len(arguments) != 1
            ):
                self.glsl_storage_struct_reinterpret_error(
                    expression,
                    source_type=source_type,
                    target_type=target_type,
                    reason="unsupported-storage-aggregate-layout",
                    detail="the scalar-array struct has unsupported layout attributes",
                )
            alignment = self.literal_int_value(
                arguments[0], self.glsl_active_literal_int_constants()
            )
            if (
                not isinstance(alignment, int)
                or isinstance(alignment, bool)
                or alignment <= 0
                or alignment & (alignment - 1)
            ):
                self.glsl_storage_struct_reinterpret_error(
                    expression,
                    source_type=source_type,
                    target_type=target_type,
                    reason="storage-aggregate-layout-unresolved",
                    detail="the struct alignment is not a fixed positive power of two",
                )
        member = members[0]
        if getattr(member, "attributes", None) or getattr(
            member, "resource_qualifiers", None
        ):
            self.glsl_storage_struct_reinterpret_error(
                expression,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-layout",
                detail="the scalar-array member has unsupported attributes or qualifiers",
            )
        member_type = getattr(member, "member_type", None)
        if not isinstance(member_type, ArrayType):
            self.glsl_storage_struct_reinterpret_error(
                expression,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-layout",
                detail="the destination struct member is not a fixed scalar array",
            )
        extent = self.literal_int_value(
            member_type.size,
            self.glsl_active_literal_int_constants(),
        )
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            self.glsl_storage_struct_reinterpret_error(
                expression,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-layout-unresolved",
                detail="the destination word-array extent is not a fixed positive integer",
            )
        member_element_type = self.glsl_normalized_source_type(member_type.element_type)
        member_layout = scalar_storage_layout(member_element_type)
        if (
            member_layout is None
            or member_layout.bit_width not in {8, 16, 32}
            or (member_layout.kind == "floating" and member_layout.bit_width != 32)
        ):
            self.glsl_storage_struct_reinterpret_error(
                expression,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-layout",
                detail=(
                    "the destination array element does not have a bit-preserving "
                    "8-, 16-, or 32-bit scalar layout"
                ),
            )
        byte_extent = extent * member_layout.byte_width
        if byte_extent % alignment != 0:
            self.glsl_storage_struct_reinterpret_error(
                expression,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-layout",
                detail="the explicit struct alignment would add unsupported tail padding",
            )
        return {
            "mapped_type": mapped_target_type,
            "member_type": self.map_type(member_type),
            "member_layout": member_layout,
            "extent": extent,
            "alignment": alignment,
        }
