"""Storage reinterpretation in the GLSL generator: struct reinterpret contracts, null workgroup pointers, parameter aliases and storage pointer access expressions."""

import re

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import evaluate_literal_int_expression
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_value_arguments,
)
from .image_access_contracts import image_access_satisfies_requirement
from .index_range_contracts import IntegerRange
from .opengl_values import (
    OpenGLMappedOverloadError,
    OpenGLStoragePointerError,
    OpenGLTemplateTypeError,
)
from .pointer_reinterpret import scalar_storage_layout


class OpenGLStorageReinterpretationMixin:
    """Storage reinterpretation members of ``GLSLCodeGen``."""

    def glsl_storage_struct_reinterpret_contract(
        self,
        expression,
        destination_type=None,
    ):
        reinterpretation = self.glsl_storage_struct_reinterpret_node(expression)
        if reinterpretation is None:
            return None
        pointer_type = getattr(reinterpretation, "target_type", None)
        target_type_node = getattr(pointer_type, "pointee_type", None)
        target_type = self.glsl_normalized_source_type(target_type_node)
        if (
            not target_type
            or scalar_storage_layout(target_type) is not None
            or self.is_vector_value_type(target_type)
        ):
            return None

        aliases = self.glsl_storage_pointer_aliases()
        source_expression = reinterpretation.expression
        binding = self.glsl_storage_pointer_binding(source_expression, aliases)
        references_storage = self.glsl_expression_references_storage_pointer(
            source_expression,
            aliases,
        )
        address_space = str(getattr(pointer_type, "address_space", None) or "").lower()
        storage_address_spaces = {"device", "constant", "storage", "global"}
        private_address_spaces = {"thread", "private", "threadgroup", "workgroup"}
        if binding is None and not (
            references_storage or address_space in storage_address_spaces
        ):
            return None

        source_type = (
            binding.get("source_element_type") if binding is not None else None
        )
        if source_type is None and binding is not None:
            source_type = binding.get("element_type")
        source_type = self.glsl_normalized_source_type(source_type)
        if binding is None:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-backing-unresolved",
                detail="the source pointer has no single proven storage-buffer backing",
            )
        if address_space in private_address_spaces:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-address-space-mismatch",
                detail=(
                    f"address space '{address_space}' does not describe the "
                    "resolved storage-buffer backing"
                ),
            )
        pointer_reinterpretation = binding.get("pointer_reinterpretation")
        if pointer_reinterpretation is None and str(
            binding.get("byte_offset", "0")
        ) not in {"0", "0u"}:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-source-layout",
                detail="the source pointer has an untyped byte offset",
            )

        source_layout = scalar_storage_layout(source_type)
        layout = self.glsl_storage_struct_word_array_layout(
            target_type,
            reinterpretation,
            source_type,
        )
        if layout is None:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-layout",
                detail="the destination is not a concrete struct",
            )
        if source_layout is None or source_layout.bit_width != 32:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="unsupported-storage-aggregate-source-layout",
                detail="the storage-buffer backing element is not a 32-bit scalar word",
                alignment=layout["member_layout"].byte_width,
            )
        if pointer_reinterpretation is None:
            if layout["member_layout"].bit_width != 32:
                self.glsl_storage_struct_reinterpret_error(
                    reinterpretation,
                    source_type=source_type,
                    target_type=target_type,
                    reason="unsupported-storage-aggregate-layout",
                    detail=(
                        "a direct storage aggregate load requires 32-bit "
                        "destination array elements"
                    ),
                    alignment=layout["member_layout"].byte_width,
                )
        else:
            view_source_layout = pointer_reinterpretation.get("source_layout")
            view_target_layout = pointer_reinterpretation.get("target_layout")
            if (
                view_source_layout != source_layout
                or pointer_reinterpretation.get("target_width", 1) != 1
                or view_target_layout != layout["member_layout"]
            ):
                self.glsl_storage_struct_reinterpret_error(
                    reinterpretation,
                    source_type=source_type,
                    target_type=target_type,
                    reason="unsupported-storage-aggregate-source-layout",
                    detail=(
                        "the proven storage scalar view does not exactly match "
                        "the destination scalar-array layout"
                    ),
                    alignment=layout["member_layout"].byte_width,
                )

        normalized_destination = self.glsl_normalized_source_type(destination_type)
        if normalized_destination is not None and self.map_type(
            normalized_destination
        ) != self.map_type(target_type):
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-destination-mismatch",
                detail=(
                    f"the destination has type '{normalized_destination}', not "
                    f"'{target_type}'"
                ),
                alignment=layout["member_layout"].byte_width,
            )
        if not image_access_satisfies_requirement("read", binding.get("access")):
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-source-not-readable",
                detail="the storage-buffer backing is not readable",
                alignment=layout["member_layout"].byte_width,
            )

        offset = binding.get("offset", 0)
        if offset is None:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-offset-unresolved",
                detail="the source element offset cannot be proven",
                alignment=layout["member_layout"].byte_width,
            )
        if self.glsl_private_pointer_expression_has_side_effects(offset):
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-offset-side-effecting",
                detail="the source element offset has side effects",
                alignment=layout["member_layout"].byte_width,
            )
        literal_offset = self.literal_int_value(
            offset,
            self.glsl_active_literal_int_constants(),
        )
        if literal_offset is not None and literal_offset < 0:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-offset-out-of-range",
                detail="the source element offset is negative",
                alignment=layout["member_layout"].byte_width,
            )
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        if not rendered_offset:
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-offset-unresolved",
                detail="the source element offset cannot be rendered",
                alignment=layout["member_layout"].byte_width,
            )
        return {
            **layout,
            "binding": binding,
            "source_layout": source_layout,
            "source_type": source_type,
            "target_type": target_type,
            "offset": rendered_offset,
            "reinterpretation": reinterpretation,
        }

    def glsl_storage_word_conversion(
        self,
        source_value,
        source_layout,
        target_layout,
    ):
        if source_layout.kind == "floating":
            source_bits = f"floatBitsToUint({source_value})"
        elif source_layout.signed:
            source_bits = f"uint({source_value})"
        else:
            source_bits = source_value
        if target_layout.kind == "floating":
            return f"uintBitsToFloat({source_bits})"
        if target_layout.signed:
            return f"int({source_bits})"
        return source_bits

    def glsl_storage_struct_reinterpret_materialization(
        self,
        expression,
        destination_type=None,
    ):
        contract = self.glsl_storage_struct_reinterpret_contract(
            expression,
            destination_type,
        )
        if contract is None:
            return None
        root = contract["binding"]["root"]
        offset = contract["offset"]
        values = []
        for index in range(contract["extent"]):
            if offset in {"0", "0u"}:
                source_index = str(index)
            elif index == 0:
                source_index = f"int({offset})"
            else:
                source_index = f"(int({offset}) + {index})"
            if contract["binding"].get("pointer_reinterpretation") is not None:
                source_value = self.glsl_pointer_reinterpret_read_expression(
                    contract["binding"],
                    source_index,
                )
                if source_value is None:
                    self.glsl_storage_struct_reinterpret_error(
                        contract["reinterpretation"],
                        source_type=contract["source_type"],
                        target_type=contract["target_type"],
                        reason="unsupported-storage-aggregate-source-layout",
                        detail="the proven storage scalar view cannot be rendered",
                        alignment=contract["member_layout"].byte_width,
                    )
                values.append(source_value)
            else:
                source_value = f"{root}[{source_index}]"
                values.append(
                    self.glsl_storage_word_conversion(
                        source_value,
                        contract["source_layout"],
                        contract["member_layout"],
                    )
                )
        array_value = f"{contract['member_type']}({', '.join(values)})"
        return f"{contract['mapped_type']}({array_value})"

    def glsl_storage_struct_reinterpret_mutation(self, expression):
        candidates = [expression]
        for node in self.walk_ast(expression):
            if node is not expression:
                candidates.append(node)
        for candidate in candidates:
            reinterpretation = self.glsl_storage_struct_reinterpret_node(candidate)
            if reinterpretation is None:
                continue
            pointer_type = getattr(reinterpretation, "target_type", None)
            target_type = self.glsl_normalized_source_type(
                getattr(pointer_type, "pointee_type", None)
            )
            if (
                not target_type
                or self.glsl_struct_node(self.map_type(target_type)) is None
            ):
                continue
            aliases = self.glsl_storage_pointer_aliases()
            binding = self.glsl_storage_pointer_binding(
                reinterpretation.expression,
                aliases,
            )
            if binding is None:
                continue
            source_type = self.glsl_normalized_source_type(
                binding.get("source_element_type") or binding.get("element_type")
            )
            self.glsl_storage_struct_reinterpret_error(
                reinterpretation,
                source_type=source_type,
                target_type=target_type,
                reason="storage-aggregate-write-unsupported",
                detail="aggregate storage reinterpretation is read-only",
                access="write",
            )
        return None

    def glsl_storage_pointer_aliases(self, aliases=None):
        return {
            **getattr(self, "glsl_global_storage_pointer_aliases", {}),
            **(aliases or {}),
            **getattr(self, "current_glsl_storage_pointer_aliases", {}),
        }

    def glsl_workgroup_pointer_null_expression(self, expression):
        if isinstance(expression, str):
            return expression == "nullptr"
        if isinstance(expression, (IdentifierNode, VariableNode)):
            return getattr(expression, "name", None) == "nullptr"
        if not isinstance(expression, TernaryOpNode):
            return False
        condition = evaluate_literal_int_expression(
            expression.condition,
            self.glsl_active_literal_int_constants(),
        )
        if condition is None:
            return False
        selected = expression.true_expr if condition else expression.false_expr
        return self.glsl_workgroup_pointer_null_expression(selected)

    def glsl_active_literal_int_constants(self):
        current_constants = getattr(self, "current_compile_time_int_constants", None)
        constants = dict(
            self.literal_int_constants
            if current_constants is None
            else current_constants
        )
        for name in getattr(self, "glsl_specialization_constant_names", set()):
            constants.pop(name, None)
        if self.software_subgroup_width is not None:
            for name, alias in self.current_stage_parameter_aliases.items():
                match = re.fullmatch(r"([0-9]+)u?", str(alias).strip())
                if match is not None:
                    constants[name] = int(match.group(1))
        return constants

    def glsl_compile_time_constant_scope_state(self):
        return (
            dict(self.current_compile_time_int_constants),
            dict(self.current_compile_time_int_vector_constants),
            dict(self.current_compile_time_int_vector_array_constants),
        )

    def restore_glsl_compile_time_constant_scope_state(self, state):
        (
            self.current_compile_time_int_constants,
            self.current_compile_time_int_vector_constants,
            self.current_compile_time_int_vector_array_constants,
        ) = state

    def glsl_workgroup_pointer_parameter_has_reachable_use(
        self,
        function,
        parameter_name,
        active=None,
        *,
        declaration=None,
    ):
        """Prove whether a workgroup pointer is used along a reachable path.

        References bind to declarations rather than names so lexical shadows do
        not keep an otherwise unused null pointer alive. Direct forwarding is
        followed through the exact call-site overload; unresolved forwarding
        and recursive cycles remain conservatively reachable. ``declaration``
        selects a local pointer declaration instead of the named parameter.
        """

        active = set() if active is None else active
        active_key = (
            id(function),
            parameter_name,
            None if declaration is None else id(declaration),
        )
        if active_key in active:
            return True
        active.add(active_key)

        lexical_bindings = self.glsl_scalar_array_view_lexical_bindings(function)
        lexical_call_source_types = lexical_bindings["call_source_types"]
        reference_bindings = lexical_bindings["reference_bindings"]
        declarations = lexical_bindings["declarations"]
        parameter = (
            declaration
            if declaration is not None
            else next(
                (
                    candidate
                    for candidate in (
                        getattr(function, "parameters", getattr(function, "params", []))
                        or []
                    )
                    if getattr(candidate, "name", None) == parameter_name
                ),
                None,
            )
        )
        parameter_binding_id = id(parameter) if parameter is not None else None

        def is_parameter_reference(value):
            if isinstance(value, str):
                # Raw string expressions carry no declaration identity. Preserve
                # the conservative name-based fallback for those legacy forms.
                return value == parameter_name
            if not isinstance(value, (IdentifierNode, VariableNode)) or (
                getattr(value, "name", None) != parameter_name
            ):
                return False
            binding_id = reference_bindings.get(id(value))
            if binding_id is None:
                declaration = declarations.get(id(value))
                if declaration is not None:
                    binding_id = declaration["id"]
            # An unresolved same-named reference remains fail-closed. Exact
            # lexical shadows carry a different declaration identity.
            return binding_id is None or binding_id == parameter_binding_id

        def contains_parameter(value, visited=None):
            visited = set() if visited is None else visited
            if value is None or isinstance(value, (int, float, bool)):
                return False
            if isinstance(value, str):
                return value == parameter_name
            if isinstance(value, dict):
                return any(
                    contains_parameter(child, visited) for child in value.values()
                )
            if isinstance(value, (list, tuple, set)):
                return any(contains_parameter(child, visited) for child in value)

            value_id = id(value)
            if value_id in visited:
                return False
            visited.add(value_id)
            if is_parameter_reference(value):
                return True
            if hasattr(value, "child_nodes"):
                return any(
                    contains_parameter(child, visited) for child in value.child_nodes()
                )
            if not hasattr(value, "__dict__"):
                return False
            return any(
                contains_parameter(child, visited)
                for field, child in vars(value).items()
                if field not in {"annotations", "parent", "source_location"}
            )

        def is_direct_forward(value):
            if isinstance(value, str):
                return value == parameter_name
            if isinstance(value, (IdentifierNode, VariableNode)):
                return is_parameter_reference(value)
            if not isinstance(value, TernaryOpNode):
                return False
            condition = evaluate_literal_int_expression(
                value.condition,
                self.glsl_active_literal_int_constants(),
            )
            if condition is None:
                return False
            selected = value.true_expr if condition else value.false_expr
            return is_direct_forward(selected)

        def forwarded_call_uses_parameter(call):
            arguments = list(
                getattr(call, "arguments", getattr(call, "args", [])) or []
            )
            function_name = self.function_call_name(call)
            if not function_name:
                return any(contains_parameter(argument) for argument in arguments)
            specialized_name = generic_function_call_name(
                self,
                function_name,
                arguments,
            )
            if specialized_name is not None:
                arguments = generic_function_value_arguments(
                    self,
                    function_name,
                    arguments,
                )
                function_name = specialized_name
            try:
                callee = self.glsl_resource_function_source(
                    function_name,
                    arguments,
                    lexical_source_types=lexical_call_source_types.get(id(call), {}),
                )
            except (OpenGLMappedOverloadError, OpenGLTemplateTypeError):
                callee = None
            parameters = list(
                getattr(callee, "parameters", getattr(callee, "params", [])) or []
            )
            for index, argument in enumerate(arguments):
                if not contains_parameter(argument):
                    continue
                if not is_direct_forward(argument) or index >= len(parameters):
                    return True
                callee_parameter = parameters[index]
                if self.glsl_workgroup_pointer_element_type(callee_parameter) is None:
                    return True
                callee_parameter_name = getattr(callee_parameter, "name", None)
                if (
                    not callee_parameter_name
                    or self.glsl_workgroup_pointer_parameter_has_reachable_use(
                        callee,
                        callee_parameter_name,
                        active,
                    )
                ):
                    return True
            return False

        visited = set()

        def references(value):
            if value is None or isinstance(value, (int, float, bool)):
                return False
            if isinstance(value, str):
                return value == parameter_name
            if isinstance(value, dict):
                return any(references(child) for child in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(references(child) for child in value)

            value_id = id(value)
            if value_id in visited:
                return False
            visited.add(value_id)
            if declaration is not None and value is declaration:
                # The tracked local's own declaration is not a use of it.
                return references(getattr(value, "initial_value", None))
            if is_parameter_reference(value):
                return True
            if isinstance(value, FunctionCallNode):
                return forwarded_call_uses_parameter(value)
            if isinstance(value, TernaryOpNode):
                # A statically selected conditional evaluates only one arm, so
                # `ARG_SORT ? view + offset : nullptr` never reads `view` when
                # the condition folds to false.
                literal_condition = evaluate_literal_int_expression(
                    value.condition,
                    self.glsl_active_literal_int_constants(),
                )
                if literal_condition is not None:
                    return references(value.condition) or references(
                        value.true_expr if literal_condition else value.false_expr
                    )
            if isinstance(value, IfNode):
                condition = getattr(
                    value, "condition", getattr(value, "if_condition", None)
                )
                if references(condition):
                    return True
                literal_condition = evaluate_literal_int_expression(
                    condition,
                    self.glsl_active_literal_int_constants(),
                )
                if literal_condition is None:
                    if references(value.if_body):
                        return True
                    return any(
                        references(else_if_condition) or references(else_if_body)
                        for else_if_condition, else_if_body in zip(
                            getattr(value, "else_if_conditions", []) or [],
                            getattr(value, "else_if_bodies", []) or [],
                        )
                    ) or references(getattr(value, "else_body", None))
                if literal_condition:
                    return references(value.if_body)
                else_if_conditions = getattr(value, "else_if_conditions", []) or []
                else_if_bodies = getattr(value, "else_if_bodies", []) or []
                for branch_index, (else_if_condition, else_if_body) in enumerate(
                    zip(else_if_conditions, else_if_bodies)
                ):
                    if references(else_if_condition):
                        return True
                    else_if_literal = evaluate_literal_int_expression(
                        else_if_condition,
                        self.glsl_active_literal_int_constants(),
                    )
                    if else_if_literal is None:
                        if references(else_if_body):
                            return True
                        return any(
                            references(later_condition) or references(later_body)
                            for later_condition, later_body in zip(
                                else_if_conditions[branch_index + 1 :],
                                else_if_bodies[branch_index + 1 :],
                            )
                        ) or references(getattr(value, "else_body", None))
                    if else_if_literal:
                        return references(else_if_body)
                return references(getattr(value, "else_body", None))

            if hasattr(value, "child_nodes"):
                return any(references(child) for child in value.child_nodes())
            if not hasattr(value, "__dict__"):
                return False
            return any(
                references(child)
                for field, child in vars(value).items()
                if field not in {"annotations", "parent", "source_location"}
            )

        try:
            return references(getattr(function, "body", []))
        finally:
            active.remove(active_key)

    def glsl_storage_pointer_parameter_has_reachable_use(
        self,
        function,
        parameter_name,
        active=None,
    ):
        active = set() if active is None else active
        active_key = (id(function), parameter_name)
        if active_key in active:
            return True
        active.add(active_key)

        lexical_bindings = self.glsl_scalar_array_view_lexical_bindings(function)
        lexical_call_source_types = lexical_bindings["call_source_types"]
        reference_bindings = lexical_bindings["reference_bindings"]
        declarations = lexical_bindings["declarations"]
        parameter = next(
            (
                candidate
                for candidate in (
                    getattr(function, "parameters", getattr(function, "params", []))
                    or []
                )
                if getattr(candidate, "name", None) == parameter_name
            ),
            None,
        )
        parameter_binding_id = id(parameter) if parameter is not None else None

        def is_parameter_reference(value):
            if isinstance(value, str):
                # Raw string expressions carry no declaration identity. Preserve
                # the conservative name-based fallback for those legacy forms.
                return value == parameter_name
            if not isinstance(value, (IdentifierNode, VariableNode)) or (
                getattr(value, "name", None) != parameter_name
            ):
                return False
            binding_id = reference_bindings.get(id(value))
            if binding_id is None:
                declaration = declarations.get(id(value))
                if declaration is not None:
                    binding_id = declaration["id"]
            # An unresolved same-named reference remains fail-closed. Exact
            # lexical shadows carry a different declaration identity.
            return binding_id is None or binding_id == parameter_binding_id

        def contains_parameter(value, visited=None):
            visited = set() if visited is None else visited
            if value is None or isinstance(value, (int, float, bool)):
                return False
            if isinstance(value, str):
                return value == parameter_name
            if isinstance(value, dict):
                return any(
                    contains_parameter(child, visited) for child in value.values()
                )
            if isinstance(value, (list, tuple, set)):
                return any(contains_parameter(child, visited) for child in value)

            value_id = id(value)
            if value_id in visited:
                return False
            visited.add(value_id)
            if is_parameter_reference(value):
                return True
            if hasattr(value, "child_nodes"):
                return any(
                    contains_parameter(child, visited) for child in value.child_nodes()
                )
            if not hasattr(value, "__dict__"):
                return False
            return any(
                contains_parameter(child, visited)
                for field, child in vars(value).items()
                if field not in {"annotations", "parent", "source_location"}
            )

        def is_direct_forward(value):
            if isinstance(value, str):
                return value == parameter_name
            if isinstance(value, (IdentifierNode, VariableNode)):
                return is_parameter_reference(value)
            if not isinstance(value, TernaryOpNode):
                return False
            condition = evaluate_literal_int_expression(
                value.condition,
                self.glsl_active_literal_int_constants(),
            )
            if condition is None:
                return False
            selected = value.true_expr if condition else value.false_expr
            return is_direct_forward(selected)

        def forwarded_call_uses_parameter(call):
            arguments = list(
                getattr(call, "arguments", getattr(call, "args", [])) or []
            )
            function_name = self.function_call_name(call)
            if not function_name:
                return any(contains_parameter(argument) for argument in arguments)
            specialized_name = generic_function_call_name(
                self,
                function_name,
                arguments,
            )
            if specialized_name is not None:
                arguments = generic_function_value_arguments(
                    self,
                    function_name,
                    arguments,
                )
                function_name = specialized_name
            try:
                callee = self.glsl_resource_function_source(
                    function_name,
                    arguments,
                    lexical_source_types=lexical_call_source_types.get(id(call), {}),
                )
            except (OpenGLMappedOverloadError, OpenGLTemplateTypeError):
                callee = None
            parameters = list(
                getattr(callee, "parameters", getattr(callee, "params", [])) or []
            )
            for index, argument in enumerate(arguments):
                if not contains_parameter(argument):
                    continue
                if not is_direct_forward(argument) or index >= len(parameters):
                    return True
                callee_parameter = parameters[index]
                if self.glsl_storage_pointer_element_type(callee_parameter) is None:
                    return True
                callee_parameter_name = getattr(callee_parameter, "name", None)
                if (
                    not callee_parameter_name
                    or self.glsl_storage_pointer_parameter_has_reachable_use(
                        callee,
                        callee_parameter_name,
                        active,
                    )
                ):
                    return True
            return False

        visited = set()

        def references(value):
            if value is None or isinstance(value, (int, float, bool)):
                return False
            if isinstance(value, str):
                return value == parameter_name
            if isinstance(value, dict):
                return any(references(child) for child in value.values())
            if isinstance(value, (list, tuple, set)):
                return any(references(child) for child in value)

            value_id = id(value)
            if value_id in visited:
                return False
            visited.add(value_id)
            if is_parameter_reference(value):
                return True
            if isinstance(value, FunctionCallNode):
                return forwarded_call_uses_parameter(value)
            if isinstance(value, IfNode):
                condition = getattr(
                    value, "condition", getattr(value, "if_condition", None)
                )
                if references(condition):
                    return True
                literal_condition = evaluate_literal_int_expression(
                    condition,
                    self.glsl_active_literal_int_constants(),
                )
                if literal_condition is None:
                    if references(value.if_body):
                        return True
                    return any(
                        references(else_if_condition) or references(else_if_body)
                        for else_if_condition, else_if_body in zip(
                            getattr(value, "else_if_conditions", []) or [],
                            getattr(value, "else_if_bodies", []) or [],
                        )
                    ) or references(getattr(value, "else_body", None))
                if literal_condition:
                    return references(value.if_body)
                else_if_conditions = getattr(value, "else_if_conditions", []) or []
                else_if_bodies = getattr(value, "else_if_bodies", []) or []
                for branch_index, (else_if_condition, else_if_body) in enumerate(
                    zip(else_if_conditions, else_if_bodies)
                ):
                    if references(else_if_condition):
                        return True
                    else_if_literal = evaluate_literal_int_expression(
                        else_if_condition,
                        self.glsl_active_literal_int_constants(),
                    )
                    if else_if_literal is None:
                        if references(else_if_body):
                            return True
                        return any(
                            references(later_condition) or references(later_body)
                            for later_condition, later_body in zip(
                                else_if_conditions[branch_index + 1 :],
                                else_if_bodies[branch_index + 1 :],
                            )
                        ) or references(getattr(value, "else_body", None))
                    if else_if_literal:
                        return references(else_if_body)
                return references(getattr(value, "else_body", None))

            if hasattr(value, "child_nodes"):
                return any(references(child) for child in value.child_nodes())
            if not hasattr(value, "__dict__"):
                return False
            return any(
                references(child)
                for field, child in vars(value).items()
                if field not in {"annotations", "parent", "source_location"}
            )

        try:
            return references(getattr(function, "body", []))
        finally:
            active.remove(active_key)

    def glsl_fold_null_workgroup_pointer_locals(self, function):
        """Elide workgroup pointer locals that are statically ``nullptr``.

        A specialization can select ``nullptr`` for a view such as
        ``const threadgroup uint* idx = ARG_SORT ? tile + offset : nullptr``.
        Such a local has no shared backing, so no ``int`` offset can represent
        it. When it is never reassigned or address-taken and every reachable
        use forwards it to a helper parameter that is itself proven unused,
        every reference bound to it becomes ``nullptr`` and its declaration
        emits nothing: call sites then elide the callee parameter, and
        references in statically dead branches are dropped with them. Any other
        reachable use keeps the declaration, which fails closed.
        """

        body = getattr(function, "body", None)
        candidates = {
            id(node): node
            for node in self.walk_ast(body)
            if isinstance(node, VariableNode)
            and self.glsl_workgroup_pointer_element_type(node) is not None
            and self.glsl_workgroup_pointer_null_expression(
                getattr(node, "initial_value", None)
            )
        }
        if not candidates:
            return
        reference_bindings = self.glsl_scalar_array_view_lexical_bindings(function)[
            "reference_bindings"
        ]
        for node in self.walk_ast(body):
            target = None
            if isinstance(node, AssignmentNode):
                target = getattr(node, "target", getattr(node, "left", None))
            elif isinstance(node, UnaryOpNode) and self.map_operator(node.op) in {
                "++",
                "--",
                "&",
            }:
                target = node.operand
            if isinstance(target, (IdentifierNode, VariableNode)):
                candidates.pop(reference_bindings.get(id(target)), None)
        candidates = {
            key: declaration
            for key, declaration in candidates.items()
            if not self.glsl_workgroup_pointer_parameter_has_reachable_use(
                function,
                declaration.name,
                declaration=declaration,
            )
        }
        if not candidates:
            return
        rewritten = set()
        for node in self.walk_ast(body):
            if (
                isinstance(node, IdentifierNode)
                and reference_bindings.get(id(node)) in candidates
            ):
                node.name = "nullptr"
                node.identifier = "nullptr"
                rewritten.add(id(node))
        for declaration in candidates.values():
            declaration._glsl_null_workgroup_pointer_local = True
        if rewritten:
            self.glsl_fold_null_literal_conditionals(function, rewritten)

    def glsl_replace_null_pointer_parameter_references(
        self,
        function,
        parameter_names,
        *,
        error_type=OpenGLStoragePointerError,
        pointer_kind="storage",
    ):
        lexical_bindings = self.glsl_scalar_array_view_lexical_bindings(function)
        reference_bindings = lexical_bindings["reference_bindings"]
        target_binding_ids = {
            id(parameter)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            if getattr(parameter, "name", None) in parameter_names
        }
        if len(target_binding_ids) != len(parameter_names):
            raise error_type(
                "OpenGL cannot bind every elided null "
                f"{pointer_kind}-pointer parameter",
                function_name=getattr(function, "name", None),
                reason="null-pointer-reference-binding-unresolved",
                source_location=getattr(function, "source_location", None),
            )

        visited = set()

        def replace(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, dict):
                for child in node.values():
                    replace(child)
                return
            if isinstance(node, (list, tuple, set)):
                for child in node:
                    replace(child)
                return

            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            if (
                isinstance(node, IdentifierNode)
                and getattr(node, "name", None) in parameter_names
            ):
                binding_id = reference_bindings.get(node_id)
                if binding_id is None:
                    raise error_type(
                        "OpenGL cannot resolve the lexical binding for an "
                        f"elided null {pointer_kind}-pointer reference",
                        function_name=getattr(function, "name", None),
                        parameter_name=getattr(node, "name", None),
                        reason="null-pointer-reference-binding-unresolved",
                        source_location=getattr(node, "source_location", None),
                    )
                if binding_id in target_binding_ids:
                    node.name = "nullptr"
                    node.identifier = "nullptr"
                    replaced.add(node_id)
                return
            if hasattr(node, "child_nodes"):
                for child in node.child_nodes():
                    replace(child)
                return
            if hasattr(node, "__dict__"):
                for field, child in vars(node).items():
                    if field not in {"annotations", "parent", "source_location"}:
                        replace(child)

        replaced = set()
        replace(getattr(function, "body", None))
        if replaced:
            self.glsl_fold_null_literal_conditionals(function, replaced)

    def glsl_fold_null_literal_conditionals(self, function, replaced_ids):
        """Keep only the live arm of literal conditionals naming elided nulls.

        Use analysis ignores the unselected arm of a statically selected
        conditional such as ``false ? float(view[0]) : 0.5``. Once ``view`` is
        an elided null parameter, that dead arm cannot be emitted, so the
        conditional is replaced by its selected arm, which is the only arm it
        ever evaluates. Conditionals without such a reference are unchanged.
        """

        constants = self.glsl_active_literal_int_constants()

        def live(node):
            while isinstance(node, TernaryOpNode):
                condition = evaluate_literal_int_expression(node.condition, constants)
                if condition is None:
                    break
                dead = node.false_expr if condition else node.true_expr
                if not any(id(item) in replaced_ids for item in self.walk_ast(dead)):
                    break
                node = node.true_expr if condition else node.false_expr
            return node

        visited = set()

        def rewrite(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if id(node) in visited:
                return
            visited.add(id(node))
            if isinstance(node, list):
                for index, child in enumerate(node):
                    selected = live(child)
                    if selected is not child:
                        node[index] = selected
                    rewrite(selected)
                return
            if isinstance(node, (tuple, set, dict)):
                for child in node.values() if isinstance(node, dict) else node:
                    rewrite(child)
                return
            if not hasattr(node, "__dict__"):
                return
            for field, child in list(vars(node).items()):
                if field in {"annotations", "parent", "source_location"}:
                    continue
                selected = live(child)
                if selected is not child:
                    setattr(node, field, selected)
                rewrite(selected)

        rewrite(getattr(function, "body", None))

    def glsl_workgroup_pointer_aliases(self, aliases=None):
        return {
            **(aliases or {}),
            **getattr(self, "current_glsl_workgroup_pointer_aliases", {}),
        }

    def glsl_workgroup_pointer_offset_expression(self, binding):
        offset = binding.get("offset", 0)
        offset_info = self.glsl_value_type_info(
            self.glsl_source_expression_type(offset)
        )
        # Logical pointer offsets are stored in GLSL int locals and helper
        # parameters. Wide unsigned source expressions must therefore prove
        # the signed-int range instead of narrowing to uint and overflowing at
        # the subsequent int conversion.
        target_signed = (
            True
            if offset_info is not None
            and offset_info["width"] == 1
            and offset_info["bits"] > 32
            else None
        )
        try:
            rendered = self.glsl_index_expression(
                offset,
                binding.get("root"),
                target_signed=target_signed,
                assertion_expression=binding.get("index_assertion_expression"),
            )
        except self.GLSL_INDEX_RANGE_ERROR:
            rendered = self.glsl_signed_logical_offset_expression(offset)
            if rendered is None:
                raise
        return rendered or "0"

    def glsl_signed_logical_offset_expression(self, offset):
        """Render a composed pointer offset exactly in GLSL ``int`` arithmetic.

        Pointer arithmetic moves by the exact value of each delta, so a
        logical offset such as ``in + axis_size - offset - 4`` may be negative
        before guarded accesses add their element index (MLX's reverse scans
        form such pointers). The offset is accepted only when every delta's
        proved value and every left-to-right partial sum fits ``int``; then
        the termwise ``int`` sum cannot overflow and equals the exact offset.
        The subscripts through it still need their own complete-offset proof.
        """
        terms = []

        def flatten(node, sign):
            if getattr(node, "_glsl_pointer_offset_composition", False):
                if isinstance(node, BinaryOpNode):
                    return flatten(node.left, sign) and flatten(node.right, sign)
                if isinstance(node, UnaryOpNode):
                    return flatten(node.operand, -sign)
                return False
            terms.append((sign, node))
            return True

        if not flatten(offset, 1) or len(terms) < 2:
            return None
        signed_int = IntegerRange(-(1 << 31), (1 << 31) - 1)
        total = None
        rendered = None
        for sign, term in terms:
            if isinstance(term, str):
                return None
            value = self.glsl_index_value_range(term)
            if value is None:
                return None
            low, high = (
                (value.minimum, value.maximum)
                if sign > 0
                else (-value.maximum, -value.minimum)
            )
            if not (
                value.is_within(signed_int)
                and signed_int.minimum <= low <= high <= signed_int.maximum
            ):
                return None
            total = (low, high) if total is None else (total[0] + low, total[1] + high)
            if not signed_int.minimum <= total[0] <= total[1] <= signed_int.maximum:
                return None
            info = self.glsl_value_type_info(self.glsl_source_expression_type(term))
            if (
                info is None
                or info["width"] != 1
                or info["family"]
                not in {
                    "int",
                    "uint",
                }
            ):
                return None
            literal = self.literal_int_value(
                term, self.glsl_active_literal_int_constants()
            )
            if isinstance(literal, int) and not isinstance(literal, bool):
                text = self.glsl_target_index_literal(literal, "int")
            else:
                text = self.generate_expression_with_expected(term, None)
                bits = (
                    info["narrow"]["bits"]
                    if info["narrow"] is not None
                    else info["bits"]
                )
                if info["family"] != "int" or bits > 32:
                    text = f"int({text})"
            if rendered is None:
                rendered = text if sign > 0 else f"(-{text})"
            else:
                rendered = f"({rendered} {'+' if sign > 0 else '-'} {text})"
        return rendered

    def glsl_workgroup_pointer_access_expression(self, node):
        array = getattr(node, "array", getattr(node, "array_expr", None))
        aliases = self.glsl_workgroup_pointer_aliases()
        if isinstance(array, ArrayAccessNode) and self.glsl_workgroup_vector_element(
            self.glsl_workgroup_pointer_binding(
                getattr(array, "array", getattr(array, "array_expr", None)), aliases
            )
        ):
            # ``p[a][b]`` of vector or matrix elements: ``p[a]`` is one element and ``b`` selects a component
            # or column of it. ``b`` does not move the pointer, so it is not added to the element offset.
            return None
        binding = self.glsl_workgroup_pointer_binding(array, aliases)
        if binding is None:
            return None
        index = getattr(node, "index", getattr(node, "index_expr", None))
        rendered_index = self.glsl_index_expression(index, array)
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        if rendered_offset in {"0", "0u"}:
            composed_index = rendered_index
        elif rendered_index in {"0", "0u"}:
            composed_index = rendered_offset
        else:
            index_type = self.map_type(self.expression_result_type(index))
            if index_type in {"uint", "uint64_t"}:
                rendered_index = f"int({rendered_index})"
            composed_index = f"({rendered_offset} + {rendered_index})"
        return f"{binding['root']}[{composed_index}]"

    def glsl_workgroup_vector_element(self, binding):
        """Whether the elements a workgroup pointer ``binding`` addresses are vectors or matrices."""
        if binding is None:
            return False
        element = str(binding.get("element_type") or "")
        if re.fullmatch(r"(?:f16|f64)?d?mat[234](?:x[234])?", element):
            return True
        info = self.glsl_value_type_info(element)
        return info is not None and info["width"] > 1

    def glsl_workgroup_pointer_element_expression(self, expression, index=0):
        binding = self.glsl_workgroup_pointer_binding(
            expression,
            self.glsl_workgroup_pointer_aliases(),
        )
        if binding is None:
            return None
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        rendered_index = self.glsl_index_expression(index, expression)
        if rendered_offset in {"0", "0u"}:
            composed_index = rendered_index
        elif rendered_index in {"0", "0u"}:
            composed_index = rendered_offset
        else:
            composed_index = f"({rendered_offset} + int({rendered_index}))"
        return f"{binding['root']}[{composed_index}]"

    def glsl_storage_pointer_access_expression(self, node):
        array = getattr(node, "array", getattr(node, "array_expr", None))
        binding = self.glsl_storage_pointer_binding(
            array,
            self.glsl_storage_pointer_aliases(),
        )
        if binding is None:
            return None
        index = getattr(node, "index", getattr(node, "index_expr", None))
        rendered_index = self.glsl_index_expression(index, array)
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        if rendered_offset in {"0", "0u"}:
            composed_index = rendered_index
        elif rendered_index in {"0", "0u"}:
            composed_index = rendered_offset
        else:
            index_type = self.map_type(self.expression_result_type(index))
            if index_type in {"uint", "uint64_t"}:
                rendered_index = f"int({rendered_index})"
            composed_index = f"({rendered_offset} + {rendered_index})"
        reinterpret_read = self.glsl_pointer_reinterpret_read_expression(
            binding, composed_index
        )
        if reinterpret_read is not None:
            return reinterpret_read
        return f"{binding['root']}[{composed_index}]"
