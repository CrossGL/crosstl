"""Workgroup pointers in the GLSL generator: element expressions, shared declarations and aliases, hoisted shared declarations, bindings and compile-time bitcasts."""

import re

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
)
from ..validation import expression_debug_name
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .glsl_exact_lookups import GLSL_BITCAST_CANONICAL_NAMES
from .glsl_shared_array_values import is_shared_array_value
from .image_access_contracts import image_access_satisfies_requirement
from .index_range_contracts import IntegerRange
from .opengl_values import (
    OpenGLCompileTimeGlobalError,
    OpenGLStoragePointerError,
    OpenGLTemplateTypeError,
    OpenGLWorkgroupPointerError,
)
from .pointer_reinterpret import PointerReinterpretationError
from .stage_utils import (
    collect_stage_local_variables,
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .task_payload_variables import dispatch_mesh_payload_names


class OpenGLWorkgroupPointerMixin:
    """Workgroup pointer members of ``GLSLCodeGen``."""

    def glsl_storage_pointer_element_expression(self, expression, index=0):
        pointer_update = None
        pointer_expression = expression
        if isinstance(expression, UnaryOpNode):
            operator = self.map_operator(expression.op)
            if operator in {"++", "--"} or expression.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }:
                pointer_update = operator
                pointer_expression = expression.operand
        binding = self.glsl_storage_pointer_binding(
            pointer_expression,
            self.glsl_storage_pointer_aliases(),
        )
        if binding is None:
            return None
        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        if pointer_update is not None:
            if not self.glsl_storage_pointer_offset_lvalue(rendered_offset):
                name = self.expression_name(pointer_expression) or binding.get("root")
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer update for "
                    f"'{name}' has no assignable logical offset",
                    parameter_name=name,
                    reason="pointer-update-offset-lvalue-unresolved",
                    source_location=getattr(expression, "source_location", None),
                )
            is_postfix = bool(getattr(expression, "is_postfix", False)) or (
                expression.op in {"POST_INCREMENT", "POST_DECREMENT"}
            )
            rendered_offset = (
                f"({rendered_offset}{pointer_update})"
                if is_postfix
                else f"({pointer_update}{rendered_offset})"
            )
        rendered_index = self.glsl_index_expression(index, pointer_expression)
        if rendered_offset in {"0", "0u"}:
            composed_index = rendered_index
        elif rendered_index in {"0", "0u"}:
            composed_index = rendered_offset
        else:
            composed_index = f"({rendered_offset} + int({rendered_index}))"
        reinterpret_read = self.glsl_pointer_reinterpret_read_expression(
            binding, composed_index
        )
        if reinterpret_read is not None:
            return reinterpret_read
        return f"{binding['root']}[{composed_index}]"

    def glsl_pointer_reinterpret_read_expression(self, binding, element_index):
        reinterpretation = binding.get("pointer_reinterpretation")
        if reinterpretation is None:
            return None

        source_layout = reinterpretation["source_layout"]
        target_layout = reinterpretation["target_layout"]
        target_width = reinterpretation.get("target_width", 1)
        source_member = reinterpretation.get("source_member")
        member_suffix = f".{source_member}" if source_member else ""
        byte_offset = str(binding.get("byte_offset", "0"))
        element_bytes = str(element_index)
        target_byte_width = target_layout.byte_width * target_width
        if target_byte_width != 1:
            element_bytes = f"({element_bytes} * {target_byte_width})"
        if byte_offset in {"0", "0u"}:
            byte_index = element_bytes
        elif element_bytes in {"0", "0u"}:
            byte_index = byte_offset
        else:
            byte_index = f"({byte_offset} + {element_bytes})"

        if target_width > 1:
            lane_gather = bool(reinterpretation.get("lane_gather"))
            components = []
            for lane in range(target_width):
                lane_byte_index = byte_index
                if lane:
                    lane_byte_index = (
                        f"({byte_index} + {lane * target_layout.byte_width})"
                    )
                source_index = f"int(({lane_byte_index}) / {source_layout.byte_width})"
                source_value = f"{binding['root']}[{source_index}]{member_suffix}"
                if lane_gather:
                    # Each widened array element holds one logical lane.
                    components.append(source_value)
                    continue
                if source_layout.kind == "floating":
                    source_bits = f"floatBitsToUint({source_value})"
                elif source_layout.signed:
                    source_bits = f"uint({source_value})"
                else:
                    source_bits = source_value
                if target_layout.kind == "floating":
                    components.append(f"uintBitsToFloat({source_bits})")
                elif target_layout.signed:
                    components.append(f"int({source_bits})")
                else:
                    components.append(source_bits)
            constructor = self.map_type(
                binding.get("view_element_type") or binding.get("element_type")
            )
            return f"{constructor}({', '.join(components)})"

        source_index = f"int(({byte_index}) / {source_layout.byte_width})"
        source_value = f"{binding['root']}[{source_index}]{member_suffix}"
        if source_layout.kind == "floating":
            source_bits = f"floatBitsToUint({source_value})"
        elif source_layout.signed:
            source_bits = f"uint({source_value})"
        else:
            source_bits = source_value

        if target_layout.bit_width == 32:
            if target_layout.kind == "floating":
                return f"uintBitsToFloat({source_bits})"
            if target_layout.signed:
                return f"int({source_bits})"
            return source_bits

        bit_offset = f"int((({byte_index}) % {source_layout.byte_width}) * 8)"
        extraction_source = (
            f"int({source_bits})" if target_layout.signed else source_bits
        )
        return (
            f"bitfieldExtract({extraction_source}, {bit_offset}, "
            f"{target_layout.bit_width})"
        )

    def glsl_workgroup_pointer_member_expression(
        self, pointer_expression, member, source_location=None
    ):
        pointee = self.glsl_workgroup_pointer_element_expression(pointer_expression)
        if pointee is None:
            return None
        binding = self.glsl_workgroup_pointer_binding(
            pointer_expression,
            self.glsl_workgroup_pointer_aliases(),
        )
        element_type = binding.get("element_type") if binding else None
        member_name = (
            self.glsl_struct_member_name(element_type, str(member))
            if self.glsl_struct_node(element_type) is not None
            else str(member)
        )
        return f"{pointee}.{member_name}"

    def glsl_storage_pointer_member_expression(
        self, pointer_expression, member, source_location=None
    ):
        pointee = self.glsl_storage_pointer_element_expression(pointer_expression)
        if pointee is None:
            return None
        binding = self.glsl_storage_pointer_binding(
            pointer_expression,
            self.glsl_storage_pointer_aliases(),
        )
        element_type = binding.get("element_type") if binding else None
        member_name = (
            self.glsl_struct_member_name(element_type, str(member))
            if self.glsl_struct_node(element_type) is not None
            else str(member)
        )
        return f"{pointee}.{member_name}"

    def glsl_workgroup_pointer_bare_expression_error(self, expression):
        name = getattr(expression, "name", None) or self.expression_name(expression)
        binding = self.current_glsl_workgroup_pointer_aliases.get(name)
        if binding is None or binding.get("kind") != "workgroup-pointer":
            return
        if is_shared_array_value(self, name):
            return
        raise OpenGLWorkgroupPointerError(
            f"OpenGL cannot emit a workgroup pointer as a first-class value: {name}",
            parameter_name=name,
            reason="bare-pointer-expression",
            source_location=getattr(expression, "source_location", None),
        )

    def glsl_storage_pointer_bare_expression_error(self, expression):
        name = getattr(expression, "name", None) or self.expression_name(expression)
        binding = self.current_glsl_storage_pointer_aliases.get(name)
        if binding is None or binding.get("kind") not in {
            "storage-pointer",
            "storage-pointer-array",
        }:
            return
        if binding.get("resource_root"):
            return
        pointer_description = (
            "storage pointer array"
            if binding.get("kind") == "storage-pointer-array"
            else "storage pointer"
        )
        raise OpenGLStoragePointerError(
            f"OpenGL cannot emit a {pointer_description} as a first-class value: "
            f"{name}",
            parameter_name=name,
            reason="bare-pointer-expression",
            source_location=getattr(expression, "source_location", None),
        )

    def validate_glsl_storage_pointer_binding_access(
        self, binding, required_access, expression
    ):
        if binding is None or image_access_satisfies_requirement(
            required_access, binding.get("access")
        ):
            return
        name = self.expression_name(expression) or binding.get("root")
        raise OpenGLStoragePointerError(
            "OpenGL storage pointer access for "
            f"'{name}' requires {required_access} storage, but the backing buffer "
            f"is {binding.get('access') or 'unspecified'}",
            parameter_name=name,
            reason="access-mismatch",
            source_location=getattr(expression, "source_location", None),
        )

    def validate_glsl_storage_pointer_mutation_target(
        self, expression, required_access="read_write"
    ):
        """Refuse a mutation through a storage view whose backing buffer lacks ``required_access``.

        A plain store needs ``write`` access, so a ``writeonly`` buffer accepts it; a compound assignment, an
        increment or an atomic reads the value as well and needs ``read_write``.
        """
        self.glsl_storage_struct_reinterpret_mutation(expression)
        binding = self.glsl_storage_pointer_mutation_binding(
            expression, self.glsl_storage_pointer_aliases()
        )
        reinterpretation = binding.get("pointer_reinterpretation") if binding else None
        if reinterpretation is not None:
            raise PointerReinterpretationError(
                "OpenGL cannot preserve writes through a reinterpreted typed "
                "storage-buffer view",
                source_type=reinterpretation["source_layout"].name,
                target_type=reinterpretation["target_layout"].name,
                address_space="storage",
                alignment=reinterpretation["target_layout"].byte_width,
                access="write",
                reason="write-requires-byte-address-storage",
                source_location=getattr(expression, "source_location", None),
            )
        self.validate_glsl_storage_pointer_binding_access(
            binding, required_access, expression
        )

    def generate_glsl_workgroup_pointer_alias_declaration(self, node, indent):
        if getattr(node, "_glsl_null_workgroup_pointer_local", False):
            # Folded to nullptr: references were rewritten, nothing is stored.
            return ""
        initial_value = getattr(node, "initial_value", None)
        binding = self.glsl_workgroup_pointer_binding(
            initial_value,
            self.glsl_workgroup_pointer_aliases(),
        )
        if binding is None:
            raise OpenGLWorkgroupPointerError(
                "OpenGL cannot lower workgroup pointer alias "
                f"'{node.name}' without a concrete shared-array backing object",
                parameter_name=node.name,
                reason="backing-unresolved",
                source_location=getattr(node, "source_location", None),
            )
        if binding.get("array_size") is None:
            raise OpenGLWorkgroupPointerError(
                "OpenGL cannot lower workgroup pointer alias "
                f"'{node.name}' because the backing array extent is unresolved",
                parameter_name=node.name,
                reason="missing-concrete-size",
                source_location=getattr(node, "source_location", None),
            )

        declared_element_type = self.glsl_workgroup_pointer_element_type(node)
        backing_element_type = binding.get("element_type")
        if (
            declared_element_type is not None
            and backing_element_type is not None
            and declared_element_type != backing_element_type
        ):
            raise OpenGLWorkgroupPointerError(
                "OpenGL workgroup pointer alias "
                f"'{node.name}' changes element type from {backing_element_type} "
                f"to {declared_element_type}",
                parameter_name=node.name,
                reason="element-type-mismatch",
                source_location=getattr(node, "source_location", None),
            )

        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        offset_name = self.glsl_synthetic_local_identifier(f"{node.name}_offset")
        lowered_binding = {
            **binding,
            "offset": offset_name,
            "element_type": declared_element_type or backing_element_type,
        }
        self.current_glsl_workgroup_pointer_aliases[node.name] = lowered_binding
        source_type = self.local_variable_declared_type(node)
        self.local_variable_types[node.name] = source_type
        self.local_variable_source_types[node.name] = self.type_name_string(source_type)
        self.local_variable_types[offset_name] = "int"
        self.local_variable_source_types[offset_name] = "int"
        return f"{'    ' * indent}int {offset_name} = int({rendered_offset});\n"

    def generate_glsl_workgroup_pointer_assignment(self, node, indent):
        target = getattr(node, "target", getattr(node, "left", None))
        if not isinstance(target, (str, IdentifierNode, VariableNode)):
            return None
        target_name = getattr(target, "name", None) or self.expression_name(target)
        current = self.current_glsl_workgroup_pointer_aliases.get(target_name)
        if current is None or current.get("kind") != "workgroup-pointer":
            return None

        operator = self.map_operator(
            getattr(node, "operator", getattr(node, "op", "="))
        )
        right = getattr(node, "value", getattr(node, "right", None))
        offset_name = current.get("offset")
        if not isinstance(offset_name, str) or not offset_name:
            raise OpenGLWorkgroupPointerError(
                f"OpenGL cannot reassign workgroup pointer '{target_name}'",
                parameter_name=target_name,
                reason="assignment-storage-unavailable",
                source_location=getattr(node, "source_location", None),
            )

        indent_str = "    " * indent
        if operator in {"+=", "-="}:
            rendered_delta = self.generate_expression(right)
            return f"{indent_str}{offset_name} {operator} int({rendered_delta});\n"
        if operator != "=":
            raise OpenGLWorkgroupPointerError(
                f"OpenGL cannot lower workgroup pointer operator '{operator}'",
                parameter_name=target_name,
                reason="assignment-operator-unsupported",
                source_location=getattr(node, "source_location", None),
            )

        replacement = self.glsl_workgroup_pointer_binding(
            right, self.glsl_workgroup_pointer_aliases()
        )
        if replacement is None:
            raise OpenGLWorkgroupPointerError(
                "OpenGL workgroup pointer assignment has no concrete shared-array "
                f"backing object: {target_name}",
                parameter_name=target_name,
                reason="assignment-backing-unresolved",
                source_location=getattr(node, "source_location", None),
            )
        if replacement.get("array_size") is None:
            raise OpenGLWorkgroupPointerError(
                "OpenGL workgroup pointer assignment has an unresolved backing "
                f"extent: {target_name}",
                parameter_name=target_name,
                reason="missing-concrete-size",
                source_location=getattr(node, "source_location", None),
            )
        if replacement.get("element_type") != current.get("element_type"):
            raise OpenGLWorkgroupPointerError(
                "OpenGL workgroup pointer assignment changes element type for "
                f"'{target_name}'",
                parameter_name=target_name,
                reason="element-type-mismatch",
                source_location=getattr(node, "source_location", None),
            )
        if (
            replacement.get("root") != current.get("root")
            and self.glsl_workgroup_pointer_scope_depth > 0
        ):
            raise OpenGLWorkgroupPointerError(
                "OpenGL cannot preserve a workgroup pointer backing-object change "
                f"for '{target_name}' across nested control flow",
                parameter_name=target_name,
                reason="dynamic-backing-reassignment",
                source_location=getattr(node, "source_location", None),
            )

        rendered_offset = self.glsl_workgroup_pointer_offset_expression(replacement)
        self.current_glsl_workgroup_pointer_aliases[target_name] = {
            **replacement,
            "offset": offset_name,
        }
        return f"{indent_str}{offset_name} = int({rendered_offset});\n"

    def generate_glsl_storage_pointer_array_alias_declaration(self, node, indent):
        binding = getattr(node, "_glsl_storage_pointer_array_binding", None)
        if not isinstance(binding, dict):
            if self.glsl_storage_pointer_array_contract(node) is not None:
                raise OpenGLStoragePointerError(
                    "OpenGL cannot emit a storage pointer array without a proven "
                    f"backing buffer: {getattr(node, 'name', '<anonymous>')}",
                    parameter_name=getattr(node, "name", None),
                    reason="pointer-array-backing-unresolved",
                    source_location=getattr(node, "source_location", None),
                )
            return None
        extent = binding.get("extent")
        if not isinstance(extent, int) or extent <= 0:
            raise OpenGLStoragePointerError(
                "OpenGL local storage pointer array has an invalid fixed-offset "
                f"contract: {getattr(node, 'name', '<anonymous>')}",
                parameter_name=getattr(node, "name", None),
                reason="pointer-array-declaration-contract-invalid",
                source_location=getattr(node, "source_location", None),
            )
        offset_array = self.glsl_synthetic_local_identifier(f"{node.name}_offsets")
        offsets = [f"{offset_array}[{index}]" for index in range(extent)]
        if getattr(node, "initial_value", None) is not None:
            raise OpenGLStoragePointerError(
                "OpenGL local storage pointer-array aggregate initializers are "
                "not supported",
                parameter_name=getattr(node, "name", None),
                reason="pointer-array-aggregate-initializer-unsupported",
                source_location=getattr(node, "source_location", None),
            )

        lowered_binding = {
            **binding,
            "offset_array": offset_array,
            "offset_expressions": offsets,
        }
        self.current_glsl_storage_pointer_aliases[node.name] = lowered_binding
        source_type = self.local_variable_declared_type(node)
        self.local_variable_types[node.name] = source_type
        self.local_variable_source_types[node.name] = self.type_name_string(source_type)
        self.local_variable_types[offset_array] = f"int[{extent}]"
        self.local_variable_source_types[offset_array] = f"int[{extent}]"
        return f"{'    ' * indent}int {offset_array}[{extent}];\n"

    def generate_glsl_storage_pointer_array_assignment(self, node, indent):
        target = getattr(node, "target", getattr(node, "left", None))
        if not isinstance(target, ArrayAccessNode):
            return None
        base_expression, index_expression = target.array, target.index
        base_name = self.expression_name(base_expression)
        current = self.current_glsl_storage_pointer_aliases.get(base_name)
        if current is None or current.get("kind") != "storage-pointer-array":
            return None

        extent = current.get("extent")
        offsets = list(current.get("offset_expressions") or [])
        if not isinstance(extent, int) or extent <= 0 or len(offsets) != extent:
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer-array assignment has an invalid fixed "
                "offset contract",
                parameter_name=base_name,
                reason="pointer-array-assignment-contract-invalid",
                source_location=getattr(node, "source_location", None),
            )
        lower, upper = self.glsl_storage_pointer_array_index_interval(
            current,
            index_expression,
            node,
        )
        for element_index in range(lower, upper + 1):
            if not self.glsl_storage_pointer_offset_lvalue(offsets[element_index]):
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array assignment has no assignable "
                    f"logical offset for '{base_name}[{element_index}]'",
                    parameter_name=base_name,
                    reason="pointer-array-assignment-offset-lvalue-unresolved",
                    source_location=getattr(node, "source_location", None),
                )

        operator = self.map_operator(
            getattr(node, "operator", getattr(node, "op", "="))
        )
        right = getattr(node, "value", getattr(node, "right", None))
        if operator in {"+=", "-="}:
            rendered_value = self.generate_expression(right)
            assignment_for = lambda offset: (
                f"{offset} {operator} int({rendered_value});"
            )
        elif operator == "=":
            replacement = self.glsl_storage_pointer_binding(
                right,
                self.glsl_storage_pointer_aliases(),
            )
            if replacement is None:
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array assignment has no concrete "
                    f"storage-buffer backing object: {base_name}",
                    parameter_name=base_name,
                    reason="pointer-array-assignment-backing-unresolved",
                    source_location=getattr(node, "source_location", None),
                )
            if replacement.get("root") != current.get("root"):
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array assignment changes backing "
                    f"object for '{base_name}'",
                    parameter_name=base_name,
                    reason="pointer-array-assignment-backing-mismatch",
                    source_location=getattr(node, "source_location", None),
                )
            if replacement.get("element_type") != current.get("element_type"):
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array assignment changes element "
                    f"type for '{base_name}'",
                    parameter_name=base_name,
                    reason="pointer-array-assignment-element-type-mismatch",
                    source_location=getattr(node, "source_location", None),
                )
            self.validate_glsl_storage_pointer_binding_access(
                replacement,
                current.get("access") or "read",
                target,
            )
            rendered_value = self.glsl_workgroup_pointer_offset_expression(replacement)
            assignment_for = lambda offset: f"{offset} = int({rendered_value});"
        else:
            raise OpenGLStoragePointerError(
                f"OpenGL cannot lower storage pointer-array operator '{operator}'",
                parameter_name=base_name,
                reason="pointer-array-assignment-operator-unsupported",
                source_location=getattr(node, "source_location", None),
            )

        indent_str = "    " * indent
        if lower == upper:
            return f"{indent_str}{assignment_for(offsets[lower])}\n"

        selector = self.generate_expression(index_expression)
        code = ""
        for branch_index, element_index in enumerate(range(lower, upper + 1)):
            if branch_index == 0:
                code += f"{indent_str}if (int({selector}) == {element_index}) {{\n"
            else:
                code += f" else if (int({selector}) == {element_index}) {{\n"
            code += (
                f"{indent_str}    {assignment_for(offsets[element_index])}\n"
                f"{indent_str}}}"
            )
        return code + "\n"

    def generate_glsl_storage_pointer_alias_declaration(self, node, indent):
        initial_value = getattr(node, "initial_value", None)
        binding = self.glsl_storage_pointer_binding(
            initial_value,
            self.glsl_storage_pointer_aliases(),
        )
        deferred_binding = getattr(
            node,
            "_glsl_deferred_storage_pointer_binding",
            None,
        )
        if binding is None and deferred_binding is not None:
            binding = dict(deferred_binding)
        if binding is None:
            raise OpenGLStoragePointerError(
                "OpenGL cannot lower storage pointer alias "
                f"'{node.name}' without a concrete storage-buffer backing object",
                parameter_name=node.name,
                reason="backing-unresolved",
                source_location=getattr(node, "source_location", None),
            )

        declared_element_type = self.glsl_storage_pointer_element_type(node)
        backing_element_type = binding.get("element_type")
        if (
            declared_element_type is not None
            and backing_element_type is not None
            and declared_element_type != backing_element_type
        ):
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer alias "
                f"'{node.name}' changes element type from {backing_element_type} "
                f"to {declared_element_type}",
                parameter_name=node.name,
                reason="element-type-mismatch",
                source_location=getattr(node, "source_location", None),
            )

        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        required_access = (
            "read"
            if qualifiers.intersection({"const", "constant", "readonly", "in"})
            else "read_write"
        )
        self.validate_glsl_storage_pointer_binding_access(
            binding, required_access, node
        )

        rendered_offset = self.glsl_workgroup_pointer_offset_expression(binding)
        offset_name = self.glsl_synthetic_local_identifier(f"{node.name}_offset")
        lowered_binding = {
            **binding,
            "offset": offset_name,
            "element_type": declared_element_type or backing_element_type,
            "access": required_access,
            "resource_root": False,
        }
        self.current_glsl_storage_pointer_aliases[node.name] = lowered_binding
        source_type = self.local_variable_declared_type(node)
        self.local_variable_types[node.name] = source_type
        self.local_variable_source_types[node.name] = self.type_name_string(source_type)
        self.local_variable_types[offset_name] = "int"
        self.local_variable_source_types[offset_name] = "int"
        if initial_value is None and deferred_binding is not None:
            return f"{'    ' * indent}int {offset_name};\n"
        return f"{'    ' * indent}int {offset_name} = int({rendered_offset});\n"

    def glsl_asserted_complete_pointer_update(
        self, target, operator, delta, proved_delta
    ):
        """Accept ``target += delta`` from an asserted result and bounded delta.

        Inside a loop, or after an unproved rebase, the current offset has no
        proved interval, so a complete-pointer assertion cannot be turned into
        a delta range. It still bounds the updated offset itself. The emitted
        ``offset += int(delta)`` is then exact when the wide delta is also
        proved or separately asserted to fit ``int``: every value stored in the
        ``int`` offset local is proved or asserted, so the ``int`` addition
        cannot overflow. Either assertion alone is still insufficient.
        """
        delta_range = proved_delta
        if delta_range is None:
            asserted = self.glsl_index_asserted_range(delta)
            if asserted is None:
                return False
            delta_range = (asserted.minimum, asserted.maximum)
        signed_int = IntegerRange(-(1 << 31), (1 << 31) - 1)
        if not IntegerRange(*delta_range).is_within(signed_int):
            return False
        target_name = expression_debug_name(target)
        delta_name = expression_debug_name(delta)
        if not target_name or not delta_name:
            return False
        symbol = "+" if operator == "+=" else "-"
        complete = self.glsl_index_asserted_range(
            f"{target_name} {symbol} {delta_name}"
        )
        return complete is not None and complete.is_within(
            IntegerRange(0, (1 << 31) - 1)
        )

    def glsl_asserted_pointer_update_delta(self, target, operator, delta, base):
        """Return an explicitly asserted wide delta for ``target += delta``.

        Callers may assert either the delta itself or the complete updated
        pointer ``target + delta`` (``target - delta`` for ``-=``), matching the
        source pointer-arithmetic keys used for pointer-valued expressions. The
        result is caller-trusted: deriving a delta from the complete key needs a
        proved base (``glsl_asserted_complete_pointer_update`` covers updates
        without one), and the subsequent subscripts keep their own
        complete-offset obligations.
        """
        asserted = self.glsl_index_asserted_range(delta)
        if asserted is not None:
            return asserted.minimum, asserted.maximum
        target_name = expression_debug_name(target)
        delta_name = expression_debug_name(delta)
        if not target_name or not delta_name:
            return None
        symbol = "+" if operator == "+=" else "-"
        complete = self.glsl_index_asserted_range(
            f"{target_name} {symbol} {delta_name}"
        )
        if complete is None:
            return None
        if operator == "+=":
            return complete.minimum - base[1], complete.maximum - base[0]
        return base[0] - complete.maximum, base[1] - complete.minimum

    def generate_glsl_storage_pointer_step(self, expression, operator):
        """``p++``, ``++p``, ``p--`` or ``--p`` of a storage pointer whose value
        is discarded.

        A storage pointer lowers to its backing buffer and an element offset,
        so the step moves the offset by one element, as ``p += 1`` does. A
        step whose value is used needs the pointer as a first-class value and
        stays refused.
        """
        if expression is not self.current_discarded_expression:
            return None
        operand = expression.operand
        if not isinstance(operand, (str, IdentifierNode, VariableNode)):
            return None
        name = getattr(operand, "name", None) or self.expression_name(operand)
        binding = self.current_glsl_storage_pointer_aliases.get(name)
        if binding is None or binding.get("kind") != "storage-pointer":
            return None
        offset = binding.get("offset")
        if binding.get("resource_root") or not self.glsl_storage_pointer_offset_lvalue(
            offset
        ):
            raise OpenGLStoragePointerError(
                f"OpenGL cannot reassign storage pointer '{name}'",
                parameter_name=name,
                reason="assignment-storage-unavailable",
                source_location=getattr(expression, "source_location", None),
            )
        increment = operator == "++" or expression.op in {
            "PRE_INCREMENT",
            "POST_INCREMENT",
        }
        return f"({offset} {'+=' if increment else '-='} 1)"

    def generate_glsl_storage_pointer_assignment(self, node, indent):
        target = getattr(node, "target", getattr(node, "left", None))
        if not isinstance(target, (str, IdentifierNode, VariableNode)):
            return None
        target_name = getattr(target, "name", None) or self.expression_name(target)
        current = self.current_glsl_storage_pointer_aliases.get(target_name)
        if current is None or current.get("kind") != "storage-pointer":
            return None

        operator = self.map_operator(
            getattr(node, "operator", getattr(node, "op", "="))
        )
        right = getattr(node, "value", getattr(node, "right", None))
        offset_name = current.get("offset")
        if not isinstance(offset_name, str) or not offset_name:
            raise OpenGLStoragePointerError(
                f"OpenGL cannot reassign storage pointer '{target_name}'",
                parameter_name=target_name,
                reason="assignment-storage-unavailable",
                source_location=getattr(node, "source_location", None),
            )

        indent_str = "    " * indent
        if operator in {"+=", "-="}:
            rendered_delta = self.generate_expression(right)
            info = self.glsl_value_type_info(self.glsl_source_expression_type(right))
            if info is not None and info["width"] == 1 and info["bits"] > 32:
                current["wide_offset"] = True
                intervals = getattr(node, "_glsl_control_flow_intervals", {})
                base = intervals.get(target_name)
                delta = self.glsl_checked_private_pointer_interval(
                    right,
                    intervals,
                    self.glsl_active_literal_int_constants(),
                    require_arithmetic_types=True,
                )
                proved_delta = delta
                if delta is None and base is not None:
                    delta = self.glsl_asserted_pointer_update_delta(
                        target, operator, right, base
                    )
                combined = None
                accepted = IntegerRange(-(1 << 31), (1 << 31) - 1)
                delta_fits = delta is not None and IntegerRange(*delta).is_within(
                    accepted
                )
                if base is not None and delta is not None:
                    if operator == "-=":
                        delta = (-delta[1], -delta[0])
                    combined = IntegerRange(base[0] + delta[0], base[1] + delta[1])
                if (
                    not delta_fits
                    or combined is None
                    or not combined.is_within(accepted)
                ) and not self.glsl_asserted_complete_pointer_update(
                    target, operator, right, proved_delta
                ):
                    self.glsl_index_type_error(
                        right,
                        target,
                        self.glsl_source_expression_type(right),
                        "int",
                        "pointer-offset-range-unproven",
                        range_status="unproven",
                        source_range=combined,
                        accepted_range=accepted,
                    )
            return f"{indent_str}{offset_name} {operator} int({rendered_delta});\n"
        if operator != "=":
            raise OpenGLStoragePointerError(
                f"OpenGL cannot lower storage pointer operator '{operator}'",
                parameter_name=target_name,
                reason="assignment-operator-unsupported",
                source_location=getattr(node, "source_location", None),
            )

        replacement = self.glsl_storage_pointer_binding(
            right, self.glsl_storage_pointer_aliases()
        )
        if replacement is None:
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer assignment has no concrete storage-buffer "
                f"backing object: {target_name}",
                parameter_name=target_name,
                reason="assignment-backing-unresolved",
                source_location=getattr(node, "source_location", None),
            )
        if replacement.get("element_type") != current.get("element_type"):
            raise OpenGLStoragePointerError(
                "OpenGL storage pointer assignment changes element type for "
                f"'{target_name}'",
                parameter_name=target_name,
                reason="element-type-mismatch",
                source_location=getattr(node, "source_location", None),
            )
        self.validate_glsl_storage_pointer_binding_access(
            replacement, current.get("access") or "read_write", target
        )
        if (
            replacement.get("root") != current.get("root")
            and self.glsl_workgroup_pointer_scope_depth > 0
        ):
            raise OpenGLStoragePointerError(
                "OpenGL cannot preserve a storage pointer backing-object change "
                f"for '{target_name}' across nested control flow",
                parameter_name=target_name,
                reason="dynamic-backing-reassignment",
                source_location=getattr(node, "source_location", None),
            )

        rendered_offset = self.glsl_workgroup_pointer_offset_expression(replacement)
        self.current_glsl_storage_pointer_aliases[target_name] = {
            **replacement,
            "offset": offset_name,
            "access": current.get("access"),
            "resource_root": False,
        }
        return f"{indent_str}{offset_name} = int({rendered_offset});\n"

    def glsl_emitted_functions_for_shared_lowering(self, ast, target_stage=None):
        """Return functions whose bodies are emitted for the requested stage."""
        functions = []
        seen = set()

        def add(func):
            if func is None or id(func) in seen:
                return
            seen.add(id(func))
            functions.append(func)

        for func in getattr(ast, "functions", []) or []:
            qualifier_name = function_stage_name(func)
            if should_emit_qualified_function(target_stage, qualifier_name):
                add(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            for func in getattr(stage, "local_functions", []) or []:
                add(func)
            add(getattr(stage, "entry_point", None))

        return functions

    def glsl_unique_shared_identifier(self, base_name, used_names):
        """Return a unique global identifier for a hoisted shared variable."""
        reserved_names = set(used_names)
        reserved_names.update(self.glsl_module_used_identifier_names)
        return self.glsl_unique_identifier(base_name, reserved_names)

    def collect_glsl_function_local_shared_declarations(self, ast, target_stage=None):
        """Hoist function-local workgroup-shared declarations to global scope.

        GLSL requires ``shared`` variables to live at module scope, so every
        threadgroup/groupshared/workgroup-qualified local declaration is given a
        unique global alias.  References are rewritten to the alias and the
        original local declaration is suppressed (mirrors the DirectX backend's
        ``groupshared`` hoisting).
        """
        used_names = set(self.global_variable_types)
        used_names.update(
            getattr(node, "name", None)
            for node in collect_stage_local_variables(
                ast, target_stage, self.glsl_declaration_has_shared_qualifier
            )
            if getattr(node, "name", None)
        )
        declarations = []
        aliases_by_declaration = {}
        owners_by_declaration = {}
        declaration_ids = set()

        for func in self.glsl_emitted_functions_for_shared_lowering(ast, target_stage):
            func_name = getattr(func, "name", None) or "shader"
            payload_names = dispatch_mesh_payload_names(getattr(func, "body", None))
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, VariableNode):
                    continue
                if not self.glsl_declaration_has_shared_qualifier(node):
                    continue
                if self.glsl_workgroup_pointer_declaration(node):
                    continue

                name = getattr(node, "name", None)
                if not name:
                    continue

                alias = self.glsl_unique_shared_identifier(
                    f"{func_name}_{name}", used_names
                )
                used_names.add(alias)
                self.glsl_module_used_identifier_names.add(alias)
                declared_type = self.local_variable_declared_type(node)
                self.global_variable_types[alias] = declared_type
                declarations.append((node, alias))
                aliases_by_declaration[id(node)] = alias
                owners_by_declaration[id(node)] = f"{func_name}:{alias}"
                declaration_ids.add(id(node))
                if name in payload_names:
                    # task_payload_variables.py: the DispatchMesh payload is taskPayloadSharedEXT storage.
                    node._glsl_task_payload_shared = True
                    self.task_payload_shared_variables.append(
                        {
                            "name": alias,
                            "type": self.map_type(declared_type),
                            "raw_type": self.type_name_string(declared_type),
                            "array_suffix": "",
                        }
                    )

        self.glsl_hoisted_shared_declarations = declarations
        self.glsl_hoisted_shared_aliases_by_declaration = aliases_by_declaration
        self.glsl_hoisted_shared_declaration_ids = declaration_ids
        self.glsl_hoisted_shared_alias_names = {alias for _, alias in declarations}
        self.glsl_shared_dimension_constants = self.collect_glsl_local_int_constants(
            self.glsl_emitted_functions_for_shared_lowering(ast, target_stage)
        )
        self.glsl_hoisted_shared_bindings_by_declaration = {
            id(node): self.glsl_shared_storage_binding(
                node,
                alias,
                entry_owner=owners_by_declaration[id(node)],
            )
            for node, alias in declarations
        }
        for node, alias in declarations:
            node._glsl_hoisted_shared_alias = alias
            node._glsl_hoisted_shared_binding = (
                self.glsl_hoisted_shared_bindings_by_declaration.get(id(node))
            )

    def collect_glsl_local_int_constants(self, functions):
        """Collect function-local compile-time integer constants by name.

        Threadgroup arrays are frequently sized by a function-local
        ``constexpr int`` (e.g. MLX's ``constexpr int SIMD_SIZE = 32;``). When
        such an array is hoisted to module scope its size expression must not
        reference the (now out-of-scope) local constant, so we resolve those
        names to their integer values and substitute them into the hoisted
        array dimension.
        """
        constants: dict[str, int] = {}
        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, VariableNode):
                    continue
                name = getattr(node, "name", None)
                if not name or name in constants:
                    continue
                qualifiers = {
                    str(q).lower() for q in getattr(node, "qualifiers", []) or []
                }
                if not qualifiers & {"const", "constexpr", "constant"}:
                    continue
                declared = self.type_name_string(
                    getattr(node, "var_type", getattr(node, "vtype", None))
                )
                if declared not in {"int", "uint", "short", "ushort", "long", "ulong"}:
                    continue
                value = evaluate_literal_int_expression(
                    getattr(node, "initial_value", None), constants
                )
                if value is not None:
                    constants[name] = value
        return constants

    def resolve_glsl_shared_array_dimensions(self, type_string):
        """Substitute known local int constants into an array-type suffix.

        Turns e.g. ``float[SIMD_SIZE]`` / ``float[(3 * SIMD_SIZE)]`` into
        ``float[32]`` / ``float[(3 * 32)]`` so the hoisted module-scope shared
        declaration does not reference an out-of-scope local constant.
        """
        constants = getattr(self, "glsl_shared_dimension_constants", None)
        if not constants or not type_string or "[" not in type_string:
            return type_string
        base, suffix = split_array_type_suffix(str(type_string))

        def _replace(match):
            token = match.group(0)
            if token in constants:
                return str(constants[token])
            return token

        resolved_suffix = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _replace, suffix)
        return f"{base}{resolved_suffix}"

    def generate_glsl_hoisted_shared_declaration(self, node, alias):
        """Render a hoisted workgroup-shared variable as a global declaration."""
        if self.glsl_workgroup_pointer_declaration(node):
            raise OpenGLWorkgroupPointerError(
                "OpenGL workgroup pointer aliases cannot be emitted as shared "
                f"declarations: {getattr(node, 'name', '<unnamed>')}",
                parameter_name=getattr(node, "name", None),
                reason="pointer-storage-declaration",
                source_location=getattr(node, "source_location", None),
            )
        vtype = self.local_variable_declared_type(node)
        vtype = self.resolve_glsl_shared_array_dimensions(vtype)
        storage_type = self.float_atomic_storage.declaration(
            self.map_type(vtype), alias
        )
        declaration = format_c_style_array_declaration(storage_type, alias)
        qualifier = (
            "taskPayloadSharedEXT "
            if getattr(node, "_glsl_task_payload_shared", False)
            else self.global_variable_qualifier(node)
        )
        return (
            self.float_atomic_storage.take_definitions()
            + f"{qualifier}{declaration};\n"
        )

    def activate_glsl_hoisted_shared_declaration(self, node):
        """Alias a hoisted shared local to its global name for the current scope."""
        alias = self.glsl_hoisted_shared_alias_for_declaration(node)
        if alias is None:
            return False

        name = getattr(node, "name", None)
        if name:
            self.current_identifier_aliases[name] = alias
            declared_type = self.local_variable_declared_type(node)
            if declared_type:
                self.local_variable_types[name] = declared_type
                self.local_variable_types[alias] = declared_type
            binding = self.glsl_hoisted_shared_binding_for_declaration(node)
            if binding is not None:
                self.current_glsl_workgroup_pointer_aliases[name] = dict(binding)
        return True

    def glsl_hoisted_shared_alias_for_declaration(self, node):
        alias = self.glsl_hoisted_shared_aliases_by_declaration.get(id(node))
        if alias is not None:
            return alias
        alias = getattr(node, "_glsl_hoisted_shared_alias", None)
        if alias in self.glsl_hoisted_shared_alias_names:
            return alias
        return None

    def glsl_hoisted_shared_binding_for_declaration(self, node):
        binding = self.glsl_hoisted_shared_bindings_by_declaration.get(id(node))
        if binding is not None:
            return binding
        if self.glsl_hoisted_shared_alias_for_declaration(node) is None:
            return None
        return getattr(node, "_glsl_hoisted_shared_binding", None)

    def match_binding_name(self, name):
        return self.glsl_local_identifier_name(name)

    def current_stage_declared_names(self):
        names = set(self.current_stage_parameter_aliases.values())
        for member_map in self.current_stage_inputs.values():
            names.update(member_map.values())
        for member_map in self.current_stage_outputs.values():
            names.update(member_map.values())
        names.update(self.current_stage_output_member_map.values())
        if self.current_stage_output is not None:
            names.add(self.current_stage_output["name"])
        return names

    def local_variable_qualifier(self, node):
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.add(str(attr_name).lower())

        parts = []
        if "const" in qualifiers:
            parts.append("const")
        if (
            "precise" in qualifiers
            or (
                id(node) in getattr(self, "current_glsl_exactness_test_local_ids", ())
                and self.glsl_exactness_test_value_has_float_sum(
                    self.current_glsl_exactness_test_local_ids[id(node)]
                )
            )
            or ("const" not in qualifiers and self.glsl_precise_function_local(node))
        ):
            parts.append("precise")
        parts.extend(
            qualifier
            for qualifier in ("lowp", "mediump", "highp")
            if qualifier in qualifiers
        )
        return f"{' '.join(parts)} " if parts else ""

    def glsl_precise_value_type(self, vtype):
        """Return whether a value of ``vtype`` can carry the precise qualifier."""
        if vtype is None or isinstance(vtype, (PointerType, ReferenceType)):
            return False
        try:
            mapped = self.map_type(vtype)
        except (OpenGLTemplateTypeError, ValueError):
            return False
        if not mapped or mapped == "void" or self.is_opaque_resource_type(mapped):
            return False
        if self.is_structured_buffer_type(vtype):
            return False
        return not self.type_contains_opaque_resource(vtype)

    @staticmethod
    def glsl_has_precise_qualifier(node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "precise" in qualifiers:
            return True
        return any(
            str(getattr(attribute, "name", "")).lower() == "precise"
            for attribute in getattr(node, "attributes", []) or []
        )

    def glsl_precise_return_local_type(self, node, declaration_type):
        """Return the type materialized for a portable precise function result."""
        if declaration_type == "void" or not self.glsl_has_precise_qualifier(node):
            return None
        return declaration_type

    def opengl_compile_time_global_error(
        self,
        node,
        expression,
        *,
        reason,
        detail=None,
        operation=None,
        operand_type=None,
        result_type=None,
    ):
        name = self.resource_node_name(node, "<unnamed>")
        description = reason.replace("-", " ")
        if detail:
            description += f" '{detail}'"
        raise OpenGLCompileTimeGlobalError(
            f"OpenGL compile-time global '{name}' cannot use {description}",
            variable_name=name,
            expression_kind=(
                expression.__class__.__name__ if expression is not None else None
            ),
            detail=detail,
            operation=operation,
            operand_type=operand_type,
            result_type=result_type,
            reason=reason,
            source_location=getattr(expression, "source_location", None)
            or getattr(node, "source_location", None),
        )

    def glsl_compile_time_bitcast_contract(self, node, expression):
        function_name = self.function_call_name(expression)
        if function_name in self.function_return_types:
            return None

        canonical_name = GLSL_BITCAST_CANONICAL_NAMES.get(function_name)
        explicit_target = self.metal_as_type_target(function_name)
        if canonical_name is None and explicit_target is None:
            return None

        arguments = list(
            getattr(expression, "arguments", getattr(expression, "args", [])) or []
        )
        if len(arguments) != 1:
            self.opengl_compile_time_global_error(
                node,
                expression,
                reason="bitcast-arity",
                detail=f"{function_name} expects exactly one operand",
                operation=function_name,
            )

        operand = arguments[0]
        operand_type = self.map_type(self.expression_result_type(operand))
        if explicit_target is None:
            result_type = self.glsl_bitcast_result_type(canonical_name, operand)
            native_target = self.glsl_bitcast_target(canonical_name, operand)
            if native_target is None:
                result_type = self.map_type(result_type)
                self.opengl_compile_time_global_error(
                    node,
                    expression,
                    reason="bitcast-type-unresolved",
                    detail=(
                        f"{function_name} cannot resolve an exact bitcast from "
                        f"'{operand_type}' to '{result_type}'"
                    ),
                    operation=function_name,
                    operand_type=operand_type,
                    result_type=result_type,
                )
        else:
            result_type = self.map_type(explicit_target)

        operand_info = self.glsl_numeric_bitcast_shape(operand_type)
        result_info = self.glsl_numeric_bitcast_shape(result_type)
        if operand_info is None or result_info is None:
            self.opengl_compile_time_global_error(
                node,
                expression,
                reason="bitcast-type-unresolved",
                detail=(
                    f"{function_name} cannot resolve an exact bitcast from "
                    f"'{operand_type}' to '{result_type}'"
                ),
                operation=function_name,
                operand_type=operand_type,
                result_type=result_type,
            )

        try:
            self.generate_glsl_explicit_bitcast(
                operand_info["type"], result_info["type"], "value"
            )
        except ValueError as exc:
            self.opengl_compile_time_global_error(
                node,
                expression,
                reason="bitcast-shape-unsupported",
                detail=str(exc),
                operation=function_name,
                operand_type=operand_info["type"],
                result_type=result_info["type"],
            )
        return {
            "operation": function_name,
            "operand": operand,
            "operand_type": operand_info["type"],
            "result_type": result_info["type"],
        }
