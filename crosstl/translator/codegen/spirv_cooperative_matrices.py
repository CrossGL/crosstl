"""Entry point returns and cooperative matrices in the SPIR-V generator: interface locations, bool interfaces, entry point return values, resource types with formats, buffer array pointers, storage buffer resource classes, array sizes, registered types, and cooperative matrix types, components, roles and element assignments."""

import re
from typing import (
    Optional,
    Union,
)

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BlockNode,
    CooperativeMatrixOpNode,
    CooperativeMatrixType,
    ForNode,
    IdentifierNode,
    LiteralNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import evaluate_literal_int_expression
from .enum_utils import generic_type_parts
from .generic_struct_utils import generic_struct_specialization_name
from .spirv_generic_structs import materialize_generic_struct_instance
from .spirv_image_texels import atomic_storage_image_format
from .spirv_stage_distances import store_entry_point_distance
from .spirv_struct_varyings import store_flattened_struct_varying
from .spirv_type_queries import type_query_pointer
from .spirv_values import (
    SpirvId,
    SpirvType,
    UnsupportedSPIRVFeatureError,
)


class SpirvCooperativeMatrixMixin:
    """Cooperative matrix members of ``VulkanSPIRVCodeGen``."""

    def entry_point_interface_location(
        self,
        semantic,
        storage_class: str,
        node,
        member_index: Optional[int],
        execution_model: Optional[str] = None,
    ) -> Optional[int]:
        if semantic is None:
            return None
        # COLORn names a colour attachment only for a fragment output; elsewhere it is a user varying.
        if storage_class == "Output" and execution_model in (None, "Fragment"):
            color_location = self.spirv_color_semantic_location(semantic, node)
            if color_location is not None:
                return color_location
        semantic_location = self.mesh_output_semantic_location(semantic)
        if semantic_location is not None:
            return semantic_location
        return member_index

    def interface_type_requires_flat(self, type_id: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(type_id.type)
        return self.normalize_primitive_name(component_type) in {"int", "uint", "bool"}

    def lowered_user_defined_interface_type(self, type_id: SpirvId) -> SpirvId:
        type_id = self.ensure_registered_type(type_id)
        type_name = type_id.type.base_type
        if self.normalize_primitive_name(type_name) == "bool":
            return self.register_primitive_type("uint")

        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is None:
            return type_id

        component_type, component_count = vector_info
        if self.normalize_primitive_name(component_type) != "bool":
            return type_id

        uint_type = self.register_primitive_type("uint")
        return self.register_vector_type(uint_type, component_count)

    def interface_type_contains_bool(
        self, type_id: Optional[SpirvId], seen=None
    ) -> bool:
        if type_id is None:
            return False
        if seen is None:
            seen = set()
        if type_id.id in seen:
            return False
        seen.add(type_id.id)

        type_name = type_id.type.base_type
        if self.normalize_primitive_name(type_name) == "bool":
            return True

        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            component_type, _ = vector_info
            return self.normalize_primitive_name(component_type) == "bool"

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, _ = array_info
            return self.interface_type_contains_bool(element_type, seen)

        members = self.current_struct_members.get(type_name)
        if members is not None:
            return any(
                self.interface_type_contains_bool(member_type, seen)
                for member_type, _ in members
            )

        return False

    def interface_source_declaration_text(
        self,
        source_node,
        fallback_name: Optional[str],
        fallback_type: Optional[SpirvId],
    ) -> Optional[str]:
        if isinstance(source_node, AssignmentNode):
            source_node = getattr(source_node, "left", None)

        declaration_name = getattr(source_node, "name", None) or fallback_name
        type_value = (
            getattr(source_node, "var_type", None)
            or getattr(source_node, "param_type", None)
            or getattr(source_node, "const_type", None)
        )
        type_name = self.type_name_from_value(type_value)
        if type_name is None and fallback_type is not None:
            type_name = fallback_type.type.base_type

        qualifiers = [
            str(qualifier)
            for qualifier in getattr(source_node, "qualifiers", None) or []
            if qualifier
        ]
        parts = [*qualifiers]
        if type_name:
            parts.append(str(type_name))
        if declaration_name:
            parts.append(str(declaration_name))
        declaration = " ".join(parts).strip()
        return declaration or None

    def validate_user_defined_interface_type(
        self,
        type_id: SpirvId,
        storage_class: str,
        name: Optional[str],
        source_node=None,
    ):
        if storage_class not in {"Input", "Output"}:
            return
        if not self.interface_type_contains_bool(type_id):
            return

        direction = storage_class.lower()
        variable_name = f" '{name}'" if name else ""
        declaration = self.interface_source_declaration_text(source_node, name, type_id)
        declaration_suffix = (
            f" from source declaration '{declaration}'" if declaration else ""
        )
        source_location = getattr(source_node, "source_location", None)
        location_label = self.format_source_location(source_location)
        location_suffix = f" at {location_label}" if location_label else ""
        raise UnsupportedSPIRVFeatureError(
            "spirv.bool_interface",
            f"Unsupported user-defined SPIR-V {direction} variable{variable_name}"
            f"{declaration_suffix}{location_suffix}; Vulkan requires "
            "Input/Output OpTypeBool interfaces to use BuiltIn. Lower boolean "
            "input-like values as int or uint uniforms or specialization "
            "constants before SPIR-V generation.",
            missing_capabilities=("spirv.bool_interface_lowering",),
            source_location=source_location,
        )

    def store_entry_point_return_value(self, value: SpirvId) -> bool:
        outputs = self.current_entry_point_return_outputs
        if outputs is None:
            return False
        if outputs["kind"] == "value":
            target = outputs["variable"]
            target_type = self.pointer_pointee_type(target) or outputs["type"]
            self.store_to_variable(
                target, self.convert_value_to_type(value, target_type)
            )
            self.create_return()
            return True

        for member_index, target, member_type in outputs["outputs"]:
            member_value = self.composite_extract(value, member_type, member_index)
            self.store_to_variable(target, member_value)
        for member_index, target, member_type, slot in outputs.get("distances", ()):
            member_value = self.composite_extract(value, member_type, member_index)
            store_entry_point_distance(self, target, slot, member_value, member_type)
        for member_index, member_type, fields in outputs.get("struct_fields", ()):
            member_value = self.composite_extract(value, member_type, member_index)
            store_flattened_struct_varying(self, member_value, fields)
        self.create_return()
        return True

    def max_optional_int(self, first: Optional[int], second: Optional[int]):
        if first is None:
            return second
        if second is None:
            return first
        return max(first, second)

    def explicit_image_format(self, node) -> Optional[str]:
        if not hasattr(node, "attributes"):
            return None

        supported_formats = self.spirv_image_format_map()
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue

            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                return attr_name

            if attr_name != "format":
                continue

            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue

            format_name = self.attribute_value_to_string(arguments[0])
            if format_name is None:
                continue

            format_name = str(format_name).lower()
            if format_name in supported_formats:
                return format_name

        return None

    def map_resource_type_with_format(self, type_name, node=None) -> SpirvId:
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            type_str = "None"

        type_str = self.normalize_generic_vector_type(type_str)
        type_str = self.normalize_hlsl_matrix_type(type_str)
        explicit_format = self.explicit_image_format(node) if node is not None else None

        array_type = self.split_outer_array_type(type_str)
        if array_type is not None:
            base_type = self.array_base_type_name(type_str)
            element_type_name, size = array_type
            if self.is_resource_type_name(base_type):
                element_type = self.map_resource_type_with_format(
                    element_type_name, node
                )
                return self.register_array_type(element_type, size)

        if self.is_resource_type_name(type_str):
            if explicit_format is None and node is not None:
                explicit_format = atomic_storage_image_format(self, node, type_str)
            return self.register_resource_type(type_str, explicit_format)

        return self.map_crossgl_type(type_str)

    def spirv_struct_member_storage_type_name(
        self, type_name, allow_runtime_array: bool = False
    ) -> str:
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return type_str

        type_str = self.normalize_generic_vector_type(type_str)
        base_type_name = self.array_base_type_name(type_str)
        if (
            self.is_resource_type_name(base_type_name)
            or self.is_acceleration_structure_type_name(base_type_name)
            or self.is_ray_query_type_name(base_type_name)
        ):
            return f"uint{type_str[len(base_type_name):]}"
        if "[]" in type_str and not allow_runtime_array:
            return type_str.replace("[]", "[1]")

        return type_str

    def storage_buffer_parameter_type_name(self, param) -> Optional[str]:
        param_type = getattr(param, "param_type", getattr(param, "vtype", None))
        if param_type is None:
            return None

        type_name = self.storage_buffer_resource_type_name(param_type)
        if type_name is not None:
            return type_name
        if self.has_attribute(param, "glsl_buffer_block"):
            return self.type_name_from_value(param_type)
        return self.buffer_array_parameter_pointer_type_name(param, param_type)

    def buffer_array_parameter_pointer_type_name(self, param, param_type):
        """An array parameter in device or constant memory is a pointer to its element.

        C++ adjusts a parameter of type "array of T" to "pointer to T" ([dcl.fct]/5), and MSL
        keeps the address space on the pointer (MSL section 4.2), so ``constant int s[3]`` binds
        the caller's buffer by reference.
        """
        if param_type.__class__.__name__ != "ArrayType":
            return None
        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", None) or []}
        if not qualifiers & {"constant", "device"}:
            return None
        element_type = getattr(param_type, "element_type", None)
        if element_type is None or element_type.__class__.__name__ == "ArrayType":
            return None
        element_name = self.type_name_from_value(element_type)
        if not element_name:
            return None
        return self.storage_buffer_resource_type_name(f"{element_name}*")

    def storage_buffer_expression_type_name(self, expr) -> Optional[str]:
        if isinstance(expr, TernaryOpNode):
            true_is_null = self.is_null_pointer_expression(expr.true_expr)
            false_is_null = self.is_null_pointer_expression(expr.false_expr)
            if true_is_null != false_is_null:
                pointer_expr = expr.false_expr if true_is_null else expr.true_expr
                return self.storage_buffer_expression_type_name(pointer_expr)
        if (
            isinstance(expr, UnaryOpNode)
            and self.is_address_of_unary(expr)
            and isinstance(expr.operand, ArrayAccessNode)
        ):
            array_expr = getattr(
                expr.operand, "array", getattr(expr.operand, "array_expr", None)
            )
            base_type = self.storage_buffer_expression_type_name(array_expr)
            type_info = self.storage_buffer_resource_type_info(base_type)
            if type_info is not None:
                element_type = type_info.get("element_type_name")
                if element_type is not None:
                    return f"{element_type}*"
        if isinstance(expr, ArrayAccessNode):
            base_type = self.storage_buffer_expression_type_name(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
            if base_type is not None:
                array_type = self.split_outer_array_type(base_type)
                if array_type is not None:
                    return array_type[0]

        source_type_name = self.source_storage_buffer_expression_type_name(expr)
        if source_type_name is not None:
            return source_type_name
        if getattr(self, "collecting_generic_function_specializations", False):
            # Collection runs before any variable is declared: resolving a pointer would lower the expression.
            return None

        pointer = type_query_pointer(self, expr)
        if pointer is None:
            return None
        metadata = self.structured_buffer_metadata_for_pointer(pointer)
        if metadata is None:
            return None
        declared_type_name = self.storage_buffer_resource_type_name(
            metadata.get("declared_type_name")
        )
        return declared_type_name or metadata.get("declared_type_name")

    def source_storage_buffer_expression_type_name(self, expr) -> Optional[str]:
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            name = expr if isinstance(expr, str) else getattr(expr, "name", None)
            if not name:
                return None
            type_name = self.local_variable_types.get(
                name
            ) or self.source_variable_type_names.get(name)
            if type_name is None:
                return None
            if self.storage_buffer_type_is_inferred_pointer(type_name):
                return None
            type_name = self.storage_buffer_resource_type_name(type_name)
            if type_name is not None:
                return type_name
        return None

    def storage_buffer_resource_type_name(self, type_name) -> Optional[str]:
        type_name = self.type_name_from_value(type_name)
        if type_name is None:
            return None

        type_name = self.normalize_reference_type_name(type_name)
        type_name = re.sub(r"\s+", "", str(type_name))
        if not type_name:
            return None

        base_type = self.array_base_type_name(type_name)
        if self.pointer_pointee_type_name_from_string(base_type) is not None:
            return type_name
        if self.is_structured_buffer_declared_type_name(type_name):
            return type_name
        return None

    def storage_buffer_resource_type_info(self, type_name):
        type_name = self.storage_buffer_resource_type_name(type_name)
        if type_name is None:
            return None

        dimensions = self.array_dimensions(type_name) or []
        base_type = self.array_base_type_name(type_name)
        structured_info = self.structured_buffer_type_info(base_type)
        if structured_info is not None:
            return {
                **structured_info,
                "declared_type_name": type_name,
                "resource_class": (
                    structured_info.get("buffer_kind") or "storage buffer"
                ),
                "array_dimensions": dimensions,
            }

        pointee_type = self.pointer_pointee_type_name_from_string(base_type)
        if pointee_type is None:
            return None

        return {
            "kind": "storage_buffer_pointer",
            "buffer_kind": "StorageBufferPointer",
            "element_type_name": self.normalize_signature_type_name(pointee_type),
            "declared_type_name": type_name,
            "resource_class": "storage buffer pointer",
            "array_dimensions": dimensions,
        }

    def is_storage_buffer_resource_type_name(self, type_name) -> bool:
        return self.storage_buffer_resource_type_info(type_name) is not None

    def storage_buffer_resource_class_label(self, type_name) -> str:
        info = self.storage_buffer_resource_type_info(type_name)
        if info is None:
            return "storage buffer"
        if info.get("byte_address"):
            return "byte-address buffer"
        return info.get("resource_class") or "storage buffer"

    def storage_buffer_parameter_type_is_compatible(
        self, declared_type: str, actual_type: str
    ) -> bool:
        declared_type = self.storage_buffer_resource_type_name(declared_type)
        actual_type = self.storage_buffer_resource_type_name(actual_type)
        declared_info = self.storage_buffer_resource_type_info(declared_type)
        actual_info = self.storage_buffer_resource_type_info(actual_type)
        if declared_info is None or actual_info is None:
            return declared_info is None and actual_info is None

        declared_dimensions = declared_info.get("array_dimensions", [])
        actual_dimensions = actual_info.get("array_dimensions", [])
        if len(declared_dimensions) != len(actual_dimensions):
            return False
        for declared_dimension, actual_dimension in zip(
            declared_dimensions, actual_dimensions
        ):
            if declared_dimension and declared_dimension != actual_dimension:
                return False

        if bool(declared_info.get("byte_address")) != bool(
            actual_info.get("byte_address")
        ):
            return False
        if declared_info.get("byte_address"):
            return True

        declared_element_type = self.normalize_signature_type_name(
            declared_info.get("element_type_name")
        )
        actual_element_type = self.normalize_signature_type_name(
            actual_info.get("element_type_name")
        )
        if declared_element_type != actual_element_type and not (
            self.same_storage_element_type(declared_element_type, actual_element_type)
        ):
            return False

        if (
            declared_info.get("kind") == "storage_buffer_pointer"
            or actual_info.get("kind") == "storage_buffer_pointer"
        ):
            return True

        declared_kind = declared_info.get("buffer_kind")
        actual_kind = actual_info.get("buffer_kind")
        if declared_kind == "StructuredBuffer":
            return actual_kind in {"StructuredBuffer", "RWStructuredBuffer"}
        if declared_kind == "RWStructuredBuffer":
            return actual_kind in {"StructuredBuffer", "RWStructuredBuffer"}
        return declared_kind == actual_kind

    def same_storage_element_type(self, first, second) -> bool:
        """Whether two spellings of a buffer element (``f32`` and ``float``, ``vec4<f32>`` and ``float4``) agree."""
        if not first or not second:
            return False
        try:
            first_type, second_type = (
                self.map_crossgl_type(first),
                self.map_crossgl_type(second),
            )
        except (TypeError, ValueError, KeyError):
            return False
        return getattr(first_type, "id", None) is not None and getattr(
            first_type, "id", None
        ) == getattr(second_type, "id", None)

    def function_storage_buffer_parameters(self, function_node) -> set:
        return {
            getattr(param, "name", None)
            for param in getattr(
                function_node, "parameters", getattr(function_node, "params", [])
            )
            if self.storage_buffer_parameter_type_name(param) is not None
        } - {None}

    def function_has_storage_buffer_parameters(self, function_node) -> bool:
        return bool(self.function_storage_buffer_parameters(function_node))

    def function_has_aliasing_parameters(self, function_node) -> bool:
        return any(
            self.function_parameter_requires_pointer_alias(param)
            for param in self.function_parameters(function_node)
        )

    def function_requires_pointer_preserving_inlining(self, function_node) -> bool:
        return self.function_has_storage_buffer_parameters(
            function_node
        ) or self.function_has_aliasing_parameters(function_node)

    def format_array_size(self, size, constants=None):
        if size is None:
            return None
        size_text = self.struct_array_extent_expression_text(size)
        if self.named_specialization_array_length(size_text) is not None:
            return size_text
        literal_size = evaluate_literal_int_expression(
            size,
            self.active_literal_int_constants() if constants is None else constants,
        )
        if literal_size is not None:
            return literal_size
        if hasattr(size, "value"):
            return size.value
        return size_text

    def active_literal_int_constants(self) -> dict:
        """Return visible compile-time integer values for the current function."""
        resolved = dict(self.literal_int_constants)
        for name in self.current_function_literal_int_constant_shadows:
            resolved.pop(name, None)
        resolved.update(self.current_generic_literal_int_constants())
        resolved.update(self.current_function_literal_int_constants)
        return resolved

    def current_generic_literal_int_constants(self) -> dict:
        """Return concrete integer value bindings for the current specialization."""
        resolved = {}
        for name, value in (self.current_generic_function_substitutions or {}).items():
            literal_value = evaluate_literal_int_expression(
                value,
                {**self.literal_int_constants, **resolved},
            )
            if literal_value is not None:
                resolved[name] = literal_value
        return resolved

    def unresolved_fixed_array_extent(self, type_value) -> Optional[str]:
        """Return the first non-empty array extent that is not compile-time known."""
        if type_value is None:
            return None

        if type_value.__class__.__name__ == "ArrayType":
            size = getattr(type_value, "size", None)
            size_text = self.struct_array_extent_expression_text(size)
            if (
                size is not None
                and self.named_specialization_array_length(size_text) is None
                and evaluate_literal_int_expression(
                    size, self.active_literal_int_constants()
                )
                is None
            ):
                return size_text
            return self.unresolved_fixed_array_extent(
                getattr(type_value, "element_type", None)
            )

        type_name = self.type_name_from_value(type_value)
        if type_name is None or "[" not in type_name:
            return None
        for extent in re.findall(r"\[([^\[\]]*)\]", type_name):
            extent = extent.strip()
            if not extent:
                continue
            if self.named_specialization_array_length(extent) is not None:
                continue
            if (
                evaluate_literal_int_expression(
                    extent, self.active_literal_int_constants()
                )
                is None
            ):
                return extent
        return None

    def find_registered_type_by_base(self, base_type: str) -> Optional[SpirvId]:
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
            self.cooperative_matrix_types,
            self.struct_types,
            self.array_types,
            self.layout_struct_types,
            self.layout_array_types,
            self.resource_types,
            self.resource_image_types,
            self.ray_query_types,
        ]:
            for type_id in type_dict.values():
                if type_id.type.base_type == base_type:
                    return type_id
        return None

    def find_registered_type_by_id(self, id_value: int) -> Optional[SpirvId]:
        for type_dict in [
            self.primitive_types,
            self.vector_types,
            self.matrix_types,
            self.cooperative_matrix_types,
            self.struct_types,
            self.array_types,
            self.layout_struct_types,
            self.layout_array_types,
            self.resource_types,
            self.resource_image_types,
            self.ray_query_types,
        ]:
            for type_id in type_dict.values():
                if type_id.id == id_value:
                    return type_id
        return None

    def ensure_registered_type(self, type_ref: Union[SpirvId, SpirvType]) -> SpirvId:
        if isinstance(type_ref, SpirvId):
            return type_ref

        registered_type = self.find_registered_type_by_base(type_ref.base_type)
        if registered_type is not None:
            return registered_type

        return self.map_crossgl_type(type_ref.base_type)

    def scalar_or_vector_component_type(self, spirv_type: SpirvType) -> str:
        vector_info = self.vector_component_type_and_count(spirv_type.base_type)
        if vector_info is not None:
            return vector_info[0]
        return spirv_type.base_type

    def map_crossgl_type(self, type_name) -> SpirvId:
        """Map a CrossGL type name to a SPIR-V type ID."""
        if isinstance(type_name, CooperativeMatrixType):
            type_str = self.type_name_from_value(type_name)
            if self.cooperative_matrix_khr_enabled:
                return self.register_cooperative_matrix_type(type_name, type_str)
            raise self.unsupported_cooperative_matrix_type_error(type_name, type_str)

        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            type_str = "None"

        type_str = self.normalize_reference_type_name(type_str)
        type_str = self.normalize_generic_vector_type(type_str)
        type_str = self.normalize_hlsl_matrix_type(type_str)
        if type_str.startswith("&"):
            return self.map_crossgl_type(type_str[1:].strip())
        if self.is_cooperative_matrix_type_name(type_str):
            if self.cooperative_matrix_khr_enabled:
                return self.register_cooperative_matrix_type(type_name, type_str)
            raise self.unsupported_cooperative_matrix_type_error(type_name, type_str)
        if self.is_metal_simdgroup_matrix_type(type_str):
            raise self.unsupported_cooperative_matrix_type_error(type_name, type_str)

        array_type = self.split_outer_array_type(type_str)
        if array_type is not None:
            element_type_name, size = array_type
            element_type = self.map_crossgl_type(element_type_name)
            return self.register_array_type(element_type, size)

        primitive_type = self.normalize_primitive_name(type_str)
        if primitive_type in {
            "float",
            "double",
            "int",
            "uint",
            "i64",
            "u64",
            "bool",
            "void",
            self.BFLOAT16_TYPE_NAME,
        }:
            return self.register_primitive_type(primitive_type)

        vector_info = self.vector_component_type_and_count(type_str)
        if vector_info:
            component_type, size = vector_info
            component_type_id = self.register_primitive_type(component_type)
            return self.register_vector_type(component_type_id, size)

        matrix_match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", type_str)
        if matrix_match:
            is_double, cols, rows = matrix_match.groups()
            component_type = self.register_primitive_type(
                "double" if is_double else "float"
            )
            row_count = int(rows or cols)
            col_count = int(cols)
            col_type = self.register_vector_type(component_type, row_count)
            return self.register_matrix_type(col_type, col_count)

        patch_info = self.patch_type_info_from_name(type_str)
        if patch_info is not None:
            element_type = self.map_crossgl_type(patch_info["element_type_name"])
            return self.register_array_type(element_type, patch_info["control_points"])

        if type_str == "str":
            return self.register_primitive_type("int")

        atomic_element_type_name = self.atomic_element_type_name(type_str)
        if atomic_element_type_name is not None:
            return self.map_crossgl_type(atomic_element_type_name)

        generic_enum_type = self.generic_enum_specialization_for_type(type_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_base_name, generic_args = generic_type_parts(type_str)
        if generic_args:
            # ``Box<float>`` of ``generic<T> struct Box`` is declared as the concrete ``Box_float`` on first use.
            materialize_generic_struct_instance(
                self, type_str, self.program_generic_parameter_names
            )
            specialized_struct_type = self.ensure_declared_struct_type(
                generic_struct_specialization_name(type_str)
            )
            if specialized_struct_type is not None:
                return specialized_struct_type
            declared_generic_struct_type = self.ensure_declared_struct_type(
                generic_base_name
            )
            if declared_generic_struct_type is not None:
                return declared_generic_struct_type
        if generic_args and generic_base_name in self.struct_types:
            return self.struct_types[generic_base_name]

        declared_struct_type = self.ensure_declared_struct_type(type_str)
        if declared_struct_type is not None:
            return declared_struct_type

        if type_str in self.enum_struct_type_names:
            enum_struct_type = self.ensure_enum_struct_type_registered(type_str)
            if enum_struct_type is not None:
                return enum_struct_type

        if type_str in self.enum_type_names:
            return self.register_primitive_type("int")

        option_payload = self.lowerable_option_payload_type_name(type_str)
        if option_payload is not None:
            return self.map_crossgl_type(option_payload)

        registered_type = self.find_registered_type_by_base(type_str)
        if registered_type:
            return registered_type

        if self.is_ray_query_type_name(type_str):
            return self.register_ray_query_type(type_str)

        if self.is_resource_type_name(type_str):
            return self.register_resource_type(type_str)

        if type_str in self.struct_types:
            return self.struct_types[type_str]
        elif type_str == "RayDesc":
            # DXR's built-in ray description, which TraceRayInline unpacks.
            float_type = self.register_primitive_type("float")
            vec3_type = self.register_vector_type(float_type, 3)
            return self.register_struct_type(
                "RayDesc",
                [
                    (vec3_type, "Origin"),
                    (float_type, "TMin"),
                    (vec3_type, "Direction"),
                    (float_type, "TMax"),
                ],
            )
        else:
            self.refuse_lowering(f"Unknown type {type_str}, using float as default")
            return self.register_primitive_type("float")

    def unsupported_cooperative_matrix_type_error(
        self, type_source, type_name: str
    ) -> UnsupportedSPIRVFeatureError:
        compact_type_name = re.sub(r"\s+", "", str(type_name))
        return UnsupportedSPIRVFeatureError(
            "cooperative-matrix-lowering",
            "SPIR-V cooperative-matrix lowering is not available for "
            f"'{compact_type_name}'; scalar substitution would change "
            "subgroup-distributed matrix semantics",
            missing_capabilities=("spirv.cooperative_matrix.khr",),
            source_location=getattr(type_source, "source_location", None),
        )

    def cooperative_matrix_contract(self, type_source, type_name):
        if isinstance(type_source, CooperativeMatrixType):
            return (
                type_source.element_type,
                type_source.rows,
                type_source.cols,
                type_source.scope,
                type_source.use,
                type_source.layout,
            )

        compact_type_name = re.sub(r"\s+", "", str(type_name))
        generic_base, generic_args = generic_type_parts(compact_type_name)
        if generic_base.rsplit("::", 1)[-1] != "CooperativeMatrix" or not (
            3 <= len(generic_args) <= 6
        ):
            return None
        defaults = ("subgroup", "unspecified", "unspecified")
        generic_args = [*generic_args, *defaults[len(generic_args) - 3 :]]
        return tuple(generic_args)

    def cooperative_matrix_literal_dimension(self, value):
        if isinstance(value, int) and not isinstance(value, bool):
            return value if value > 0 else None
        if isinstance(value, LiteralNode):
            return self.cooperative_matrix_literal_dimension(value.value)
        text = self.type_name_from_value(value)
        if text is not None and re.fullmatch(r"[1-9]\d*", text.strip()):
            return int(text)
        return None

    def cooperative_matrix_component_name(self, component_type):
        raw_name = self.type_name_from_value(component_type)
        if raw_name is None:
            return None
        compact_name = re.sub(r"\s+", "", raw_name).lower()
        if compact_name in {"float", "f32", "float32", "float32_t"}:
            return "float"
        if compact_name in {"int", "i32", "int32", "int32_t"}:
            return "int"
        if compact_name in {"uint", "u32", "uint32", "uint32_t"}:
            return "uint"
        return None

    def cooperative_matrix_contract_error(
        self, type_source, feature, message, capability
    ):
        return UnsupportedSPIRVFeatureError(
            feature,
            message,
            missing_capabilities=(capability,),
            source_location=getattr(type_source, "source_location", None),
        )

    def register_cooperative_matrix_type(self, type_source, type_name) -> SpirvId:
        contract = self.cooperative_matrix_contract(type_source, type_name)
        if contract is None:
            raise self.unsupported_cooperative_matrix_type_error(type_source, type_name)

        component_source, rows_source, cols_source, scope, use, layout = contract
        component_name = self.cooperative_matrix_component_name(component_source)
        if component_name is None:
            component_label = self.type_name_from_value(component_source)
            raise self.cooperative_matrix_contract_error(
                type_source,
                "cooperative-matrix-component-type",
                "SPIR-V KHR cooperative matrices currently require a 32-bit "
                f"float, signed integer, or unsigned integer component; got "
                f"'{component_label}'",
                "spirv.cooperative_matrix.component-type",
            )

        rows = self.cooperative_matrix_literal_dimension(rows_source)
        cols = self.cooperative_matrix_literal_dimension(cols_source)
        if rows is None or cols is None:
            raise self.cooperative_matrix_contract_error(
                type_source,
                "cooperative-matrix-shape",
                "SPIR-V KHR cooperative-matrix type registration currently "
                "requires positive literal row and column dimensions",
                "spirv.cooperative_matrix.static-shape",
            )

        scope_name = str(scope).strip().lower().replace("-", "_")
        scope_value = self.COOPERATIVE_MATRIX_SCOPE_VALUES.get(scope_name)
        if scope_value is None:
            raise self.cooperative_matrix_contract_error(
                type_source,
                "cooperative-matrix-scope",
                "The Vulkan cooperative-matrix target profile currently supports "
                f"only subgroup scope; got '{scope}'",
                "spirv.cooperative_matrix.subgroup-scope",
            )

        use_name = str(use).strip().lower().replace("-", "_")
        use_value = self.COOPERATIVE_MATRIX_USE_VALUES.get(use_name)
        if use_value is None:
            raise self.cooperative_matrix_contract_error(
                type_source,
                "cooperative-matrix-use-role",
                "SPIR-V KHR cooperative-matrix types require an explicit "
                "matrix_a, matrix_b, or accumulator use role; "
                f"got '{use}'",
                "spirv.cooperative_matrix.use-role",
            )

        layout_name = str(layout or "unspecified").strip().lower().replace("-", "_")
        component_type = self.register_primitive_type(component_name)
        key = (component_type.id, scope_value, rows, cols, use_value, layout_name)
        existing = self.cooperative_matrix_types.get(key)
        if existing is not None:
            return existing

        self.require_capability("CooperativeMatrixKHR")
        self.require_capability("VulkanMemoryModel")
        self.require_extension("SPV_KHR_cooperative_matrix")
        self.require_extension("SPV_KHR_vulkan_memory_model")
        if self.uses_device_scope_memory_operation:
            self.require_capability("VulkanMemoryModelDeviceScopeKHR")

        uint_type = self.register_primitive_type("uint")
        scope_id = self.register_constant(scope_value, uint_type)
        rows_id = self.register_constant(rows, uint_type)
        cols_id = self.register_constant(cols, uint_type)
        use_id = self.register_constant(use_value, uint_type)
        type_id = self.get_id()
        self.emit(
            f"%{type_id} = OpTypeCooperativeMatrixKHR %{component_type.id} "
            f"%{scope_id.id} %{rows_id.id} %{cols_id.id} %{use_id.id}"
        )
        canonical_name = re.sub(r"\s+", "", str(type_name))
        matrix_id = SpirvId(type_id, SpirvType(canonical_name), canonical_name)
        self.cooperative_matrix_types[key] = matrix_id
        self.cooperative_matrix_type_metadata[matrix_id.id] = {
            "component_type": component_type,
            "component_name": component_name,
            "scope": scope_name,
            "scope_value": scope_value,
            "rows": rows,
            "cols": cols,
            "use": self.COOPERATIVE_MATRIX_USE_NAMES[use_value],
            "use_value": use_value,
            "layout": layout_name,
        }
        self.cooperative_matrix_containing_type_ids.add(matrix_id.id)
        return matrix_id

    def cooperative_matrix_type_info(self, type_id: SpirvId):
        """Return the registered KHR cooperative-matrix contract for a type."""
        if type_id is None:
            return None
        registered_type = self.ensure_registered_type(type_id)
        return self.cooperative_matrix_type_metadata.get(registered_type.id)

    def cooperative_matrix_operation_error(
        self, expr, feature: str, message: str, capability: str
    ):
        return UnsupportedSPIRVFeatureError(
            feature,
            message,
            missing_capabilities=(capability,),
            source_location=getattr(expr, "source_location", None),
        )

    def cooperative_matrix_role_specialization_error(
        self, node, reason: str, message: str
    ):
        return UnsupportedSPIRVFeatureError(
            f"cooperative-matrix-role-specialization-{reason}",
            message,
            missing_capabilities=(
                f"spirv.cooperative_matrix.role-specialization-{reason}",
            ),
            source_location=getattr(node, "source_location", None),
        )

    def neutral_cooperative_matrix_type(self, type_source):
        if not isinstance(type_source, CooperativeMatrixType):
            return None
        use = str(getattr(type_source, "use", "unspecified"))
        normalized_use = use.strip().lower().replace("-", "_")
        return type_source if normalized_use == "unspecified" else None

    def cooperative_matrix_type_with_use(self, type_source, use: str):
        return CooperativeMatrixType(
            type_source.element_type,
            type_source.rows,
            type_source.cols,
            scope=type_source.scope,
            use=use,
            layout=type_source.layout,
            source_location=getattr(type_source, "source_location", None),
            annotations=dict(getattr(type_source, "annotations", {}) or {}),
        )

    def cooperative_matrix_direct_identifier_name(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return expr.name
        if isinstance(expr, str):
            return expr
        return None

    def cooperative_matrix_role_binding(self, expr, scopes):
        name = self.cooperative_matrix_direct_identifier_name(expr)
        if name is None:
            return None
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None

    def record_cooperative_matrix_role_demand(
        self, binding, role: str, source, analysis
    ):
        if binding is None or binding[0] != "neutral":
            return
        declaration = binding[1]
        analysis["demands"][id(declaration)].setdefault(role, []).append(source)

    def scan_cooperative_matrix_direct_role_operand(
        self, value, scopes, analysis, role: str
    ):
        if self.cooperative_matrix_direct_identifier_name(value) is not None:
            binding = self.cooperative_matrix_role_binding(value, scopes)
            if role != "element":
                self.record_cooperative_matrix_role_demand(
                    binding, role, value, analysis
                )
            return
        self.scan_cooperative_matrix_role_expression(value, scopes, analysis)

    def scan_cooperative_matrix_role_expression(self, value, scopes, analysis):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for item in value.values():
                self.scan_cooperative_matrix_role_expression(item, scopes, analysis)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self.scan_cooperative_matrix_role_expression(item, scopes, analysis)
            return
        if isinstance(value, BlockNode):
            self.scan_cooperative_matrix_role_block(value, scopes, analysis)
            return
        if isinstance(value, VariableNode):
            self.scan_cooperative_matrix_role_declaration(value, scopes, analysis)
            return
        if isinstance(value, AssignmentNode):
            self.scan_cooperative_matrix_role_assignment(value, scopes, analysis)
            return
        if isinstance(value, IdentifierNode):
            binding = self.cooperative_matrix_role_binding(value, scopes)
            if binding is None or binding[0] != "neutral":
                return
            declaration = binding[1]
            raise self.cooperative_matrix_role_specialization_error(
                declaration,
                "escape",
                "SPIR-V cooperative-matrix role specialization cannot prove the "
                f"role of local '{declaration.name}' because it escapes direct "
                "matrix multiply and lane-element operations",
            )
        if isinstance(value, CooperativeMatrixOpNode):
            if value.operation == "multiply":
                for index, role in ((0, "matrix_a"), (1, "matrix_b")):
                    if index >= len(value.arguments):
                        continue
                    self.scan_cooperative_matrix_direct_role_operand(
                        value.arguments[index], scopes, analysis, role
                    )
                for argument in value.arguments[2:]:
                    self.scan_cooperative_matrix_role_expression(
                        argument, scopes, analysis
                    )
                return
            if value.operation == "element":
                if value.arguments:
                    self.scan_cooperative_matrix_direct_role_operand(
                        value.arguments[0], scopes, analysis, "element"
                    )
                for argument in value.arguments[1:]:
                    self.scan_cooperative_matrix_role_expression(
                        argument, scopes, analysis
                    )
                return
            if value.operation == "load":
                if value.arguments:
                    self.scan_cooperative_matrix_direct_role_operand(
                        value.arguments[0], scopes, analysis, "element"
                    )
                for argument in value.arguments[1:]:
                    self.scan_cooperative_matrix_role_expression(
                        argument, scopes, analysis
                    )
                return
            if value.operation == "store":
                if value.arguments:
                    self.scan_cooperative_matrix_role_expression(
                        value.arguments[0], scopes, analysis
                    )
                if len(value.arguments) >= 2:
                    self.scan_cooperative_matrix_direct_role_operand(
                        value.arguments[1], scopes, analysis, "element"
                    )
                for argument in value.arguments[2:]:
                    self.scan_cooperative_matrix_role_expression(
                        argument, scopes, analysis
                    )
                return

        if isinstance(value, ForNode):
            scopes.append({})
            try:
                self.scan_cooperative_matrix_role_expression(
                    value.init, scopes, analysis
                )
                self.scan_cooperative_matrix_role_expression(
                    value.condition, scopes, analysis
                )
                self.scan_cooperative_matrix_role_expression(
                    value.update, scopes, analysis
                )
                self.scan_cooperative_matrix_role_expression(
                    value.body, scopes, analysis
                )
            finally:
                scopes.pop()
            return

        if hasattr(value, "child_nodes"):
            visited_children = set()
            for child in value.child_nodes():
                child_id = id(child)
                if child_id in visited_children:
                    continue
                visited_children.add(child_id)
                self.scan_cooperative_matrix_role_expression(child, scopes, analysis)

    def scan_cooperative_matrix_role_declaration(self, node, scopes, analysis):
        source_type = getattr(node, "var_type", getattr(node, "vtype", None))
        neutral_type = self.neutral_cooperative_matrix_type(source_type)
        existing_binding = self.cooperative_matrix_role_binding(node.name, scopes)
        if existing_binding is not None and (
            neutral_type is not None or existing_binding[0] == "neutral"
        ):
            raise self.cooperative_matrix_role_specialization_error(
                node,
                "shadowing",
                "SPIR-V cooperative-matrix role specialization requires a "
                f"unique unshadowed local declaration for '{node.name}'",
            )
        if neutral_type is not None and getattr(node, "is_type_alias", False):
            raise self.cooperative_matrix_role_specialization_error(
                node,
                "alias",
                "SPIR-V cooperative-matrix role specialization does not support "
                f"role-neutral type alias '{node.name}'",
            )

        binding_kind = "neutral" if neutral_type is not None else "other"
        binding = (binding_kind, node)
        scopes[-1][node.name] = binding
        if neutral_type is not None:
            analysis["declarations"][id(node)] = node
            analysis["demands"][id(node)] = {}

        initial_value = getattr(node, "initial_value", None)
        if (
            neutral_type is not None
            and initial_value is not None
            and not (
                isinstance(initial_value, CooperativeMatrixOpNode)
                and initial_value.operation == "multiply"
            )
        ):
            raise self.cooperative_matrix_role_specialization_error(
                node,
                "alias",
                "SPIR-V cooperative-matrix role specialization does not support "
                f"value aliases or non-multiply initializers for local '{node.name}'",
            )
        if (
            neutral_type is not None
            and isinstance(initial_value, CooperativeMatrixOpNode)
            and initial_value.operation == "multiply"
        ):
            self.record_cooperative_matrix_role_demand(
                binding, "accumulator", initial_value, analysis
            )
        self.scan_cooperative_matrix_role_expression(initial_value, scopes, analysis)

    def scan_cooperative_matrix_role_assignment(self, node, scopes, analysis):
        value = getattr(node, "value", None)
        target = getattr(node, "target", None)
        if (
            getattr(node, "operator", "=") == "="
            and isinstance(value, CooperativeMatrixOpNode)
            and value.operation == "multiply"
        ):
            binding = self.cooperative_matrix_role_binding(target, scopes)
            if binding is not None and binding[0] == "neutral":
                self.record_cooperative_matrix_role_demand(
                    binding, "accumulator", value, analysis
                )
            else:
                self.scan_cooperative_matrix_role_expression(target, scopes, analysis)
            self.scan_cooperative_matrix_role_expression(value, scopes, analysis)
            return

        self.scan_cooperative_matrix_role_expression(target, scopes, analysis)
        self.scan_cooperative_matrix_role_expression(value, scopes, analysis)

    def scan_cooperative_matrix_role_block(self, block, scopes, analysis):
        scopes.append({})
        try:
            statements = getattr(block, "statements", block)
            for statement in statements or []:
                self.scan_cooperative_matrix_role_expression(
                    statement, scopes, analysis
                )
        finally:
            scopes.pop()

    def cooperative_matrix_role_overrides_for_function(self, function_node):
        if not self.cooperative_matrix_khr_enabled:
            return {}

        body = getattr(function_node, "body", None)
        if body is None:
            return {}
        neutral_parameters = [
            parameter
            for parameter in self.function_parameters(function_node)
            if self.neutral_cooperative_matrix_type(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            is not None
        ]
        if neutral_parameters:
            parameter = neutral_parameters[0]
            raise self.cooperative_matrix_role_specialization_error(
                parameter,
                "parameter",
                "SPIR-V cooperative-matrix role specialization currently "
                "supports only local declarations; role-neutral parameter "
                f"'{parameter.name}' requires an explicit role",
            )
        parameter_scope = {
            getattr(parameter, "name", f"param{index}"): ("parameter", parameter)
            for index, parameter in enumerate(self.function_parameters(function_node))
        }
        analysis = {"declarations": {}, "demands": {}}
        self.scan_cooperative_matrix_role_block(body, [parameter_scope], analysis)

        overrides = {}
        for declaration_id, declaration in analysis["declarations"].items():
            roles = analysis["demands"][declaration_id]
            if not roles:
                raise self.cooperative_matrix_role_specialization_error(
                    declaration,
                    "unresolved",
                    "SPIR-V cooperative-matrix role specialization could not "
                    "infer a MatrixA, MatrixB, or accumulator role for local "
                    f"'{declaration.name}'",
                )
            if len(roles) != 1:
                role_names = ", ".join(sorted(roles))
                raise self.cooperative_matrix_role_specialization_error(
                    declaration,
                    "conflict",
                    "SPIR-V cooperative-matrix role specialization found "
                    f"conflicting roles for local '{declaration.name}': {role_names}",
                )
            role = next(iter(roles))
            source_type = getattr(
                declaration, "var_type", getattr(declaration, "vtype", None)
            )
            overrides[declaration_id] = self.cooperative_matrix_type_with_use(
                source_type, role
            )
        return overrides

    def cooperative_matrix_operation_value(self, expr, operand, label: str):
        value = self.process_expression(operand)
        if value is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-operand-type",
                f"SPIR-V cooperative-matrix {expr.operation} could not evaluate "
                f"the {label} operand",
                "spirv.cooperative_matrix.operand-type",
            )
        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        contract = self.cooperative_matrix_type_info(value_type)
        if contract is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-operand-type",
                f"SPIR-V cooperative-matrix {expr.operation} requires a "
                f"cooperative-matrix {label} operand",
                "spirv.cooperative_matrix.operand-type",
            )
        return value, contract

    def cooperative_matrix_element_contract(self, expr):
        if len(expr.arguments) != 2:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-arity",
                "SPIR-V cooperative-matrix element access requires exactly "
                f"two operands; got {len(expr.arguments)}",
                "spirv.cooperative_matrix.element-arity",
            )

        matrix_expr, _ = expr.arguments
        matrix_pointer = self.variable_pointer_from_expression(matrix_expr)
        if matrix_pointer is None or not matrix_pointer.type.storage_class:
            matrix_value = (
                matrix_pointer
                if matrix_pointer is not None
                else self.process_expression(matrix_expr)
            )
            matrix_type = (
                self.registered_value_type(matrix_value)
                if matrix_value is not None
                else None
            )
            if self.cooperative_matrix_type_info(matrix_type) is None:
                raise self.cooperative_matrix_operation_error(
                    expr,
                    "cooperative-matrix-element-operand-type",
                    "SPIR-V cooperative-matrix element access requires an "
                    "addressable cooperative-matrix operand",
                    "spirv.cooperative_matrix.element-operand-type",
                )
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-addressability",
                "SPIR-V cooperative-matrix element access requires Function "
                "or Private storage; temporary matrix values are not yet "
                "addressable",
                "spirv.cooperative_matrix.element-addressability",
            )

        matrix_type = self.pointer_pointee_type(matrix_pointer)
        contract = self.cooperative_matrix_type_info(matrix_type)
        if contract is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-operand-type",
                "SPIR-V cooperative-matrix element access requires an "
                "addressable cooperative-matrix operand",
                "spirv.cooperative_matrix.element-operand-type",
            )

        storage_class = matrix_pointer.type.storage_class
        if storage_class not in {"Function", "Private"}:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-storage-class",
                "SPIR-V KHR cooperative-matrix element access requires "
                "Function or Private storage; "
                f"got {storage_class or 'unknown'}",
                "spirv.cooperative_matrix.storage-class",
            )

        return matrix_pointer, contract

    def cooperative_matrix_element_pointer(self, expr):
        matrix_pointer, contract = self.cooperative_matrix_element_contract(expr)
        index_expr = expr.arguments[1]

        index = self.process_expression(index_expr)
        index_type = self.registered_value_type(index) if index is not None else None
        index_type_name = (
            self.normalize_primitive_name(index_type.type.base_type)
            if index_type is not None
            else None
        )
        if index_type_name not in self.INTEGER_TYPE_NAMES:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-index-type",
                "SPIR-V cooperative-matrix element access requires a scalar "
                f"integer index; got {index_type_name or 'unknown'}",
                "spirv.cooperative_matrix.element-index-type",
            )

        # The lane-local upper bound is implementation-defined and must not be
        # inferred from the logical row and column counts.
        literal_index = evaluate_literal_int_expression(
            index_expr, self.active_literal_int_constants()
        )
        if literal_index is not None and literal_index < 0:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-index-range",
                "SPIR-V cooperative-matrix lane-local element indices cannot "
                f"be negative; got {literal_index}",
                "spirv.cooperative_matrix.element-index-range",
            )

        component_type = contract["component_type"]
        storage_class = matrix_pointer.type.storage_class
        pointer_type = self.register_pointer_type(component_type, storage_class)
        element_pointer = self.access_chain(matrix_pointer, [index], pointer_type)
        self.variable_value_types[element_pointer.id] = component_type
        return element_pointer, component_type

    def process_cooperative_matrix_element(self, expr):
        element_pointer, component_type = self.cooperative_matrix_element_pointer(expr)
        return self.load_from_variable(element_pointer, component_type)

    def process_cooperative_matrix_element_assignment(
        self, expr, value_expr, target_is_precise: bool
    ):
        value = self.process_expression_with_precision(value_expr, target_is_precise)
        if value is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-value-type",
                "SPIR-V cooperative-matrix element assignment could not "
                "evaluate a scalar component value",
                "spirv.cooperative_matrix.element-value-type",
            )

        element_pointer, component_type = self.cooperative_matrix_element_pointer(expr)
        value_type = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        value_type_name = self.normalize_primitive_name(value_type.type.base_type)
        scalar_types = {"bool", "float", "double"} | self.INTEGER_TYPE_NAMES
        if value_type_name not in scalar_types:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-value-type",
                "SPIR-V cooperative-matrix element assignment requires a scalar "
                f"component value; got {value_type_name}",
                "spirv.cooperative_matrix.element-value-type",
            )

        value = self.convert_value_to_type(value, component_type)
        converted_type = self.registered_value_type(value)
        if converted_type is None or converted_type.id != component_type.id:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-element-value-type",
                "SPIR-V cooperative-matrix element assignment could not convert "
                f"{value_type_name} to {component_type.type.base_type}",
                "spirv.cooperative_matrix.element-value-type",
            )
        self.store_to_variable(element_pointer, value)

    def cooperative_matrix_operation_result_type(self, expr):
        result_source = getattr(expr, "result_type", None)
        if (
            self.neutral_cooperative_matrix_type(result_source) is not None
            and self.current_expression_expected_type is not None
        ):
            result_source = self.current_expression_expected_type
        if result_source is None:
            result_source = self.current_expression_expected_type
        if (
            result_source is None
            or self.neutral_cooperative_matrix_type(result_source) is not None
        ):
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-result-contract",
                "SPIR-V cooperative-matrix operations require an explicit result "
                "type with an accumulator use role",
                "spirv.cooperative_matrix.result-contract",
            )
        result_type = self.map_crossgl_type(result_source)
        contract = self.cooperative_matrix_type_info(result_type)
        if contract is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-result-contract",
                "SPIR-V cooperative-matrix operations require a cooperative-matrix "
                "result type",
                "spirv.cooperative_matrix.result-contract",
            )
        return result_type, contract
