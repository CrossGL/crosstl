"""Storage copies, access chains and function references of the SPIR-V generator: array and structured buffer copies into function storage, sample mask stores, layout-compatible structs, numeric promotion, ternary values, access chain indices, composite construction, vector members and swizzles, function creation, global variables, and the resolution of function references, resource array parameters and image access requirements."""

from typing import (
    List,
    Optional,
    Tuple,
)

from .generic_function_utils import generic_function_parameters
from .spirv_specialization_conversions import specialization_constituent
from .spirv_uniform_block_arrays import require_non_uniform_block_array_indexing
from .spirv_values import SpirvId


class SpirvAccessChainMixin:
    """Access chain members of ``VulkanSPIRVCodeGen``."""

    def copy_array_pointer_to_function_storage(
        self,
        source_pointer: SpirvId,
        target_array_type: SpirvId,
        name: Optional[str] = None,
    ) -> Optional[SpirvId]:
        target_array_info = self.array_type_info_from_type(target_array_type)
        if target_array_info is None:
            return None

        target_element_type, target_size = target_array_info
        if target_size is None:
            return None

        source_array_type = self.pointer_pointee_type(source_pointer)
        source_array_info = self.array_type_info_from_type(source_array_type)
        if source_array_info is None:
            return self.copy_structured_buffer_to_function_storage(
                source_pointer,
                target_array_type,
                target_element_type,
                int(target_size),
                name,
            )

        source_element_type, source_size = source_array_info
        if source_size is not None and int(source_size) != int(target_size):
            return None

        scratch_array = self.create_variable(target_array_type, "Function", name)
        index_type = self.register_primitive_type("int")
        source_storage_class = source_pointer.type.storage_class or "Function"
        source_element_ptr_type = self.register_pointer_type(
            source_element_type, source_storage_class
        )
        target_element_ptr_type = self.register_pointer_type(
            target_element_type, "Function"
        )

        for index_value in range(int(target_size)):
            index = self.register_constant(index_value, index_type)
            source_element_pointer = self.access_chain(
                source_pointer, [index], source_element_ptr_type
            )
            self.variable_value_types[source_element_pointer.id] = source_element_type
            self.propagate_storage_buffer_access_metadata(
                source_pointer, source_element_pointer
            )
            self.propagate_structured_buffer_descriptor_access_metadata(
                source_pointer, source_element_pointer, index
            )
            self.propagate_resource_access_metadata(
                source_pointer, source_element_pointer, source_element_type
            )
            self.propagate_readonly_builtin_pointer_name(
                source_pointer, source_element_pointer
            )
            self.propagate_readonly_pointer_name(source_pointer, source_element_pointer)

            target_element_pointer = self.access_chain(
                scratch_array, [index], target_element_ptr_type
            )
            self.variable_value_types[target_element_pointer.id] = target_element_type

            value = self.load_from_variable(source_element_pointer, source_element_type)
            value = self.convert_value_to_type(value, target_element_type)
            self.store_to_variable(target_element_pointer, value)

        return scratch_array

    def copy_structured_buffer_to_function_storage(
        self,
        source_pointer: SpirvId,
        target_array_type: SpirvId,
        target_element_type: SpirvId,
        target_size: int,
        name: Optional[str] = None,
    ) -> Optional[SpirvId]:
        """Materialize a Function-local array from a structured/SSBO buffer source.

        Some helpers take a fixed-size array parameter (e.g. ``int64_t strides[2]``)
        whose SPIR-V parameter type is ``OpTypePointer Function (array<T, N>)``. When
        the matching caller argument is a buffer pointer (``constant const T*``), it
        lowers to a Uniform block struct wrapping a runtime array rather than a
        directly-registered array type, so the real-array copy path cannot see through
        it. Copy the first ``target_size`` elements out of the buffer into a
        Function-local scratch array whose pointer type matches the parameter type.

        Returns ``None`` when the source is not an indexable structured buffer, in
        which case the caller preserves its existing behaviour unchanged.
        """
        metadata = self.structured_buffer_metadata_for_pointer(source_pointer)
        if metadata is None or metadata.get("_access_path") == "member":
            return None

        scratch_array = self.create_variable(target_array_type, "Function", name)
        index_type = self.register_primitive_type("int")
        target_element_ptr_type = self.register_pointer_type(
            target_element_type, "Function"
        )

        for index_value in range(target_size):
            index = self.register_constant(index_value, index_type)
            source_element_pointer = self.structured_buffer_element_pointer(
                source_pointer, index
            )
            if source_element_pointer is None:
                return None

            source_element_type = (
                self.variable_value_types.get(source_element_pointer.id)
                or metadata.get("element_type")
                or target_element_type
            )

            target_element_pointer = self.access_chain(
                scratch_array, [index], target_element_ptr_type
            )
            self.variable_value_types[target_element_pointer.id] = target_element_type

            value = self.load_from_variable(source_element_pointer, source_element_type)
            value = self.convert_value_to_type(value, target_element_type)
            self.store_to_variable(target_element_pointer, value)

        return scratch_array

    def prepare_function_call_argument(
        self, arg: SpirvId, expected_type: SpirvId
    ) -> SpirvId:
        if expected_type.type.storage_class is None:
            return self.convert_value_to_type(arg, expected_type)

        expected_pointee_type = self.pointer_type_pointee_type(expected_type)
        if expected_pointee_type is None:
            return arg

        if arg.type.storage_class == expected_type.type.storage_class:
            return arg

        if expected_type.type.storage_class == "Function":
            expected_array_info = self.array_type_info_from_type(expected_pointee_type)
            if expected_array_info is not None:
                copied_array = self.copy_array_pointer_to_function_storage(
                    arg, expected_pointee_type
                )
                if copied_array is not None:
                    return copied_array

        return arg

    def is_sample_mask_builtin_variable(self, variable_id: SpirvId) -> bool:
        return self.builtin_names_by_variable_id.get(variable_id.id) == "SampleMask"

    def convert_sample_mask_store_value(
        self, variable_id: SpirvId, value_id: SpirvId
    ) -> Optional[SpirvId]:
        if not self.is_sample_mask_builtin_variable(variable_id):
            return None

        target_type = self.pointer_pointee_type(variable_id)
        array_info = self.array_type_info_from_type(target_type)
        if array_info is None:
            return None

        element_type, size = array_info
        if size is None:
            return None

        source_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if source_type is not None:
            source_array_info = self.array_type_info_from_type(source_type)
            if source_array_info is not None:
                return self.convert_value_to_type(value_id, target_type)

        converted = self.convert_value_to_type(value_id, element_type)
        if not self.value_has_type(converted, element_type):
            return value_id

        components = [converted]
        for _ in range(1, self.array_length_default_value(size)):
            components.append(self.default_value_for_type(element_type))
        return self.composite_construct(target_type, components)

    def load_entry_point_interface_value(
        self, variable_id: SpirvId, result_type: SpirvId
    ) -> SpirvId:
        if variable_id.type.storage_class is None:
            # Constant builtins such as gl_WorkGroupSize are materialized as an
            # OpConstantComposite value rather than a pointer variable, so the
            # value is used directly; emitting OpLoad against a non-pointer would
            # be invalid SPIR-V. A scalar parameter (e.g. threads_per_threadgroup)
            # takes the leading component, mirroring the other compute builtins, and a narrower vector
            # (``ushort2 [[threads_per_threadgroup]]``) the leading components.
            return self.convert_value_to_type(
                variable_id,
                result_type,
                allow_vector_to_scalar=True,
                allow_vector_resize=True,
            )

        if not self.is_sample_mask_builtin_variable(variable_id):
            variable_type = self.pointer_pointee_type(variable_id)
            if (
                variable_type is not None
                and variable_type.type.base_type != result_type.type.base_type
            ):
                loaded = self.load_from_variable(variable_id, variable_type)
                return self.convert_value_to_type(
                    loaded, result_type, allow_vector_resize=True
                )
            return self.load_from_variable(variable_id, result_type)

        variable_type = self.pointer_pointee_type(variable_id)
        array_info = self.array_type_info_from_type(variable_type)
        result_array = self.array_type_info_from_type(result_type)
        if array_info is None or result_array is not None:
            return self.load_from_variable(variable_id, result_type)

        element_type, _ = array_info
        loaded = self.load_from_variable(variable_id, variable_type)
        scalar = self.composite_extract(loaded, element_type, 0)
        return self.convert_value_to_type(scalar, result_type)

    def convert_value_to_type(
        self,
        value_id: SpirvId,
        target_type: SpirvId,
        *,
        allow_vector_to_scalar: bool = False,
        allow_vector_resize: bool = False,
    ) -> SpirvId:
        """Convert scalar values to a compatible scalar or vector target type."""
        target_type = self.ensure_registered_type(target_type)
        source_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if (
            source_type is not None
            and source_type.type.base_type == target_type.type.base_type
        ):
            return value_id

        if source_type is not None:
            aggregate_value = self.convert_aggregate_value_to_type(
                value_id, source_type, target_type
            )
            if aggregate_value is not None:
                return aggregate_value

        target_vector = self.vector_component_type_and_count(target_type.type.base_type)
        source_base_type = (
            source_type.type.base_type
            if source_type is not None
            else value_id.type.base_type
        )
        source_vector = self.vector_component_type_and_count(source_base_type)
        if target_vector is None:
            if source_vector is not None:
                if not allow_vector_to_scalar:
                    return value_id
                target_scalar_type = self.normalize_primitive_name(
                    target_type.type.base_type
                )
                scalar_types = {"bool", "float", "double"} | self.INTEGER_TYPE_NAMES
                if target_scalar_type not in scalar_types:
                    return value_id

                component_type = self.register_primitive_type(source_vector[0])
                scalar_value = self.composite_extract(value_id, component_type, 0)
                return self.convert_value_to_type(scalar_value, target_type)
            converted = self.convert_scalar_to_type(value_id, target_type)
            if self.normalize_primitive_name(
                converted.type.base_type
            ) == self.normalize_primitive_name(target_type.type.base_type):
                return converted
            return value_id

        if source_vector is not None:
            if source_vector[1] != target_vector[1] and not allow_vector_resize:
                return value_id
            converted = self.convert_vector_to_type(
                value_id, source_vector, target_type, target_vector
            )
            return converted or value_id

        component_type = self.register_primitive_type(target_vector[0])
        converted = self.convert_scalar_to_type(value_id, component_type)
        if self.normalize_primitive_name(converted.type.base_type) != target_vector[0]:
            return value_id
        return self.splat_scalar_to_vector(converted, target_type)

    def convert_vector_to_type(
        self,
        value_id: SpirvId,
        source_vector: Tuple[str, int],
        target_type: SpirvId,
        target_vector: Tuple[str, int],
    ) -> Optional[SpirvId]:
        target_component_type = self.register_primitive_type(target_vector[0])
        source_component_type = self.register_primitive_type(source_vector[0])
        components = []
        copied_components = min(source_vector[1], target_vector[1])
        for index in range(copied_components):
            component = self.composite_extract(value_id, source_component_type, index)
            converted = self.convert_scalar_to_type(component, target_component_type)
            if (
                self.normalize_primitive_name(converted.type.base_type)
                != target_vector[0]
            ):
                return None
            components.append(converted)

        while len(components) < target_vector[1]:
            components.append(self.default_value_for_type(target_component_type))

        return self.composite_construct(target_type, components)

    def value_has_type(self, value_id: SpirvId, target_type: SpirvId) -> bool:
        value_type = self.value_types.get(
            value_id.id
        ) or self.find_registered_type_by_base(value_id.type.base_type)
        if value_type is None:
            return value_id.type.base_type == target_type.type.base_type
        return (
            value_type.id == target_type.id
            or value_type.type.base_type == target_type.type.base_type
        )

    def aggregate_canonical_key(self, type_id: SpirvId):
        source_type = self.layout_struct_source_types.get(type_id.id)
        if source_type is not None:
            return ("struct", source_type.id)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return ("array", self.aggregate_canonical_key(element_type), size)

        if type_id.type.base_type in self.current_struct_members:
            return ("struct", type_id.id)

        return ("type", type_id.id)

    def aggregate_types_are_layout_compatible(
        self, source_type: SpirvId, target_type: SpirvId
    ) -> bool:
        source_is_struct = source_type.type.base_type in self.current_struct_members
        target_is_struct = target_type.type.base_type in self.current_struct_members
        source_is_array = self.array_type_info_from_type(source_type) is not None
        target_is_array = self.array_type_info_from_type(target_type) is not None
        if not (
            (source_is_struct and target_is_struct)
            or (source_is_array and target_is_array)
        ):
            return False
        return self.aggregate_canonical_key(
            source_type
        ) == self.aggregate_canonical_key(target_type)

    def is_complex_struct_type(
        self, type_id: SpirvId, members: Optional[List[Tuple[SpirvId, str]]] = None
    ) -> bool:
        if members is None:
            members = self.current_struct_members.get(type_id.type.base_type)
        if members is None or len(members) != 2:
            return False

        member_names = [name for _, name in members]
        type_name = str(type_id.type.base_type)
        if not (type_name.startswith("complex") or member_names == ["real", "imag"]):
            return False

        return all(
            self.normalize_primitive_name(member_type.type.base_type)
            in {"float", "double"}
            for member_type, _ in members
        )

    def complex_struct_types_are_layout_compatible(
        self,
        source_type: SpirvId,
        target_type: SpirvId,
        source_members: List[Tuple[SpirvId, str]],
        target_members: List[Tuple[SpirvId, str]],
    ) -> bool:
        if len(source_members) != len(target_members):
            return False
        if not (
            self.is_complex_struct_type(source_type, source_members)
            or self.is_complex_struct_type(target_type, target_members)
        ):
            return False

        for source_member_type, target_member_type in zip(
            (member_type for member_type, _ in source_members),
            (member_type for member_type, _ in target_members),
        ):
            source_type_name = self.normalize_primitive_name(
                source_member_type.type.base_type
            )
            target_type_name = self.normalize_primitive_name(
                target_member_type.type.base_type
            )
            if source_type_name not in {"float", "double"}:
                return False
            if target_type_name not in {"float", "double"}:
                return False

        return True

    def scalar_type_can_initialize_complex_struct(self, type_id: SpirvId) -> bool:
        type_name = self.normalize_primitive_name(type_id.type.base_type)
        return type_name in {"float", "double"} | self.INTEGER_TYPE_NAMES

    def convert_scalar_value_to_complex_struct_type(
        self,
        value_id: SpirvId,
        source_type: SpirvId,
        target_type: SpirvId,
        target_members: List[Tuple[SpirvId, str]],
    ) -> Optional[SpirvId]:
        if not self.is_complex_struct_type(target_type, target_members):
            return None
        if not self.scalar_type_can_initialize_complex_struct(source_type):
            return None

        values = []
        for index, (member_type, _) in enumerate(target_members):
            if index == 0:
                converted = self.convert_value_to_type(value_id, member_type)
                if not self.value_has_type(converted, member_type):
                    return None
                values.append(converted)
            else:
                values.append(self.default_value_for_type(member_type))

        return self.composite_construct(target_type, values)

    def convert_aggregate_value_to_type(
        self, value_id: SpirvId, source_type: SpirvId, target_type: SpirvId
    ) -> Optional[SpirvId]:
        if self.type_contains_runtime_array(
            source_type
        ) or self.type_contains_runtime_array(target_type):
            return None

        source_vector = self.vector_type_info_from_type(source_type)
        target_vector = self.vector_type_info_from_type(target_type)
        source_members = self.current_struct_members.get(source_type.type.base_type)
        target_members = self.current_struct_members.get(target_type.type.base_type)

        if source_members is None and target_members is not None:
            scalar_complex = self.convert_scalar_value_to_complex_struct_type(
                value_id, source_type, target_type, target_members
            )
            if scalar_complex is not None:
                return scalar_complex

        if source_vector is not None and target_members is not None:
            source_component_type, source_count = source_vector
            if len(target_members) != source_count:
                return None

            values = []
            for index, (target_member_type, _) in enumerate(target_members):
                source_member = self.composite_extract(
                    value_id, source_component_type, index
                )
                target_member = self.convert_value_to_type(
                    source_member, target_member_type
                )
                if not self.value_has_type(target_member, target_member_type):
                    return None
                values.append(target_member)
            return self.composite_construct(target_type, values)

        if source_members is not None and target_vector is not None:
            target_component_type, target_count = target_vector
            if len(source_members) != target_count:
                return None

            values = []
            for index, (source_member_type, _) in enumerate(source_members):
                source_member = self.composite_extract(
                    value_id, source_member_type, index
                )
                target_member = self.convert_value_to_type(
                    source_member, target_component_type
                )
                if not self.value_has_type(target_member, target_component_type):
                    return None
                values.append(target_member)
            return self.composite_construct(target_type, values)

        if source_members is None or target_members is None:
            if not self.aggregate_types_are_layout_compatible(source_type, target_type):
                return None

            source_array = self.array_type_info_from_type(source_type)
            target_array = self.array_type_info_from_type(target_type)
            if source_array is not None or target_array is not None:
                if source_array is None or target_array is None:
                    return None
                source_element_type, source_size = source_array
                target_element_type, target_size = target_array
                if (
                    source_size is None
                    or target_size is None
                    or source_size != target_size
                ):
                    return None

                elements = []
                for index in range(int(target_size)):
                    source_element = self.composite_extract(
                        value_id, source_element_type, index
                    )
                    target_element = self.convert_value_to_type(
                        source_element, target_element_type
                    )
                    if not self.value_has_type(target_element, target_element_type):
                        return None
                    elements.append(target_element)
                return self.composite_construct(target_type, elements)

            return None

        if len(source_members) != len(target_members):
            return None

        allow_layout_only_members = self.complex_struct_types_are_layout_compatible(
            source_type, target_type, source_members, target_members
        )
        values = []
        for index, (
            (source_member_type, source_name),
            (target_member_type, target_name),
        ) in enumerate(zip(source_members, target_members)):
            if source_name != target_name and not allow_layout_only_members:
                return None
            source_member = self.composite_extract(value_id, source_member_type, index)
            target_member = self.convert_value_to_type(
                source_member, target_member_type
            )
            if not self.value_has_type(target_member, target_member_type):
                return None
            values.append(target_member)

        return self.composite_construct(target_type, values)

    def promoted_numeric_type_name(
        self, type_names: List[str], *, arithmetic: bool = False
    ) -> Optional[str]:
        """The common numeric type of ``type_names``, or ``None`` when one is not numeric.

        bfloat16 operands are compared and selected as binary32, which is exact. An ``arithmetic`` result of
        bfloat16 operands, with no binary32 or binary64 operand, is bfloat16 itself: MSL's ``bfloat`` operators
        and CUDA's ``__nv_bfloat16`` operators round each result to bfloat16, so ``binary_operation`` computes
        it in binary32 and rounds it back instead of keeping the binary32 result.
        """
        numeric_types = {
            "float",
            "double",
            self.BFLOAT16_TYPE_NAME,
        } | self.INTEGER_TYPE_NAMES
        normalized = [self.normalize_primitive_name(name) for name in type_names]
        # C++ integral promotion: bool operands promote to int before the usual
        # arithmetic conversions. Without this, an arithmetic expression whose
        # operands are both bool (e.g. MLX's `Sign` functor
        # `(x > 0) - (x < 0)`) would infer a bool result type, causing the
        # subtraction to be computed in uint and coerced back to bool -- which
        # both loses the sign and produces a bool that cannot be returned from a
        # numeric (float/int) function. Treating bool as int keeps the result
        # numeric so it converts to the actual result/return type on use.
        normalized = ["int" if name == "bool" else name for name in normalized]
        if any(name not in numeric_types for name in normalized):
            return None
        if "double" in normalized:
            return "double"
        if "float" in normalized:
            return "float"
        if self.BFLOAT16_TYPE_NAME in normalized:
            return self.BFLOAT16_TYPE_NAME if arithmetic else "float"
        return self.usual_arithmetic_integer_type_name(normalized)

    def usual_arithmetic_integer_type_name(self, normalized: List[str]) -> str:
        """Apply the C/C++ usual arithmetic conversions to 32/64-bit integer types.

        Two operands of the same signedness convert to the wider type. Otherwise
        the unsigned operand wins unless the signed operand is strictly wider, so
        ``int < uint`` compares as unsigned (C++ [expr.arith.conv], HLSL and
        GLSL 4.00 implicit int-to-uint conversion) while ``long < uint`` stays
        signed 64-bit.
        """
        result = normalized[0] if normalized else "int"
        for name in normalized[1:]:
            if name == result:
                continue
            result_width = self.integer_type_width(result) or 32
            name_width = self.integer_type_width(name) or 32
            result_unsigned = result in self.UNSIGNED_INTEGER_TYPES
            if result_unsigned == (name in self.UNSIGNED_INTEGER_TYPES):
                result = name if name_width > result_width else result
                continue
            unsigned, signed = (result, name) if result_unsigned else (name, result)
            unsigned_width = self.integer_type_width(unsigned) or 32
            signed_width = self.integer_type_width(signed) or 32
            result = unsigned if unsigned_width >= signed_width else signed
        return result

    def promoted_bitwise_integer_type_name(
        self, type_names: List[str]
    ) -> Optional[str]:
        normalized = [self.normalize_primitive_name(name) for name in type_names]
        if any(name not in self.INTEGER_TYPE_NAMES for name in normalized):
            return None
        return self.usual_arithmetic_integer_type_name(normalized)

    def ternary_result_type(self, true_value: SpirvId, false_value: SpirvId) -> SpirvId:
        true_type = self.value_types.get(true_value.id) or self.ensure_registered_type(
            true_value.type
        )
        false_type = self.value_types.get(
            false_value.id
        ) or self.ensure_registered_type(false_value.type)
        if true_type.type.base_type == false_type.type.base_type:
            return true_type

        true_vector = self.vector_component_type_and_count(true_type.type.base_type)
        false_vector = self.vector_component_type_and_count(false_type.type.base_type)
        if true_vector is not None and false_vector is not None:
            if true_vector[1] != false_vector[1]:
                return true_type
            component_type_name = self.promoted_numeric_type_name(
                [true_vector[0], false_vector[0]]
            )
            if component_type_name is None:
                return true_type
            component_type = self.register_primitive_type(component_type_name)
            return self.register_vector_type(component_type, true_vector[1])

        if true_vector is not None:
            component_type_name = self.promoted_numeric_type_name(
                [true_vector[0], false_type.type.base_type]
            )
            if component_type_name is None:
                return true_type
            component_type = self.register_primitive_type(component_type_name)
            return self.register_vector_type(component_type, true_vector[1])

        if false_vector is not None:
            component_type_name = self.promoted_numeric_type_name(
                [true_type.type.base_type, false_vector[0]]
            )
            if component_type_name is None:
                return false_type
            component_type = self.register_primitive_type(component_type_name)
            return self.register_vector_type(component_type, false_vector[1])

        scalar_type_name = self.promoted_numeric_type_name(
            [true_type.type.base_type, false_type.type.base_type]
        )
        if scalar_type_name is None:
            return true_type
        return self.register_primitive_type(scalar_type_name)

    def normalize_ternary_values(
        self, true_value: SpirvId, false_value: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        result_type = self.ternary_result_type(true_value, false_value)
        true_value = self.convert_value_to_type(true_value, result_type)
        false_value = self.convert_value_to_type(false_value, result_type)
        if (
            self.value_types.get(true_value.id, true_value).type.base_type
            != result_type.type.base_type
            or self.value_types.get(false_value.id, false_value).type.base_type
            != result_type.type.base_type
        ):
            return self.ensure_registered_type(true_value.type), true_value, false_value
        return result_type, true_value, false_value

    def access_chain(
        self, base_id: SpirvId, indices: List[SpirvId], result_type: SpirvId
    ) -> SpirvId:
        """Create an access chain to a struct or array member."""
        id_value = self.get_id()

        indices = [self.access_chain_index(index) for index in indices]
        result_type, pointee_type = self.access_chain_result_pointer_type(
            base_id, indices, result_type
        )
        index_list = " ".join([f"%{index.id}" for index in indices])
        self.emit(
            f"%{id_value} = OpAccessChain %{result_type.id} %{base_id.id} {index_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.access_chain_ids.add(id_value)
        if pointee_type is not None:
            self.variable_value_types[id_value] = pointee_type
        if self.is_non_uniform_value(base_id) or any(
            self.is_non_uniform_value(index) for index in indices
        ):
            self.mark_non_uniform_result(spirv_id)
            require_non_uniform_block_array_indexing(self, base_id, indices)
        return spirv_id

    def access_chain_result_pointer_type(
        self, base_id: SpirvId, indices: List[SpirvId], fallback_type: SpirvId
    ) -> Tuple[SpirvId, Optional[SpirvId]]:
        pointee_type = self.access_chain_result_pointee_type(base_id, indices)
        if pointee_type is None:
            return fallback_type, self.pointer_type_pointee_type(fallback_type)

        storage_class = base_id.type.storage_class or fallback_type.type.storage_class
        if storage_class is None:
            return fallback_type, pointee_type

        return self.register_pointer_type(pointee_type, storage_class), pointee_type

    def access_chain_result_pointee_type(
        self, base_id: SpirvId, indices: List[SpirvId]
    ) -> Optional[SpirvId]:
        current_type = self.pointer_pointee_type(base_id)
        if current_type is None and base_id.type.storage_class is not None:
            current_type = self.find_registered_type_by_base(
                base_id.type.base_type.replace("ptr_", "", 1)
            )
        if current_type is None:
            return None

        for index in indices:
            array_info = self.array_type_info_from_type(current_type)
            if array_info is not None:
                current_type = array_info[0]
                continue

            matrix_info = self.matrix_type_info_from_type(current_type)
            if matrix_info is not None:
                current_type = matrix_info[0]
                continue

            vector_info = self.vector_type_info_from_type(current_type)
            if vector_info is not None:
                current_type = vector_info[0]
                continue

            members = self.current_struct_members.get(current_type.type.base_type)
            if members is None:
                return None

            member_index = self.constant_values_by_id.get(index.id)
            if member_index is None:
                return None
            try:
                member_index = int(member_index)
            except (TypeError, ValueError):
                return None
            if not 0 <= member_index < len(members):
                return None
            current_type = members[member_index][0]

        return current_type

    def access_chain_index(self, index: SpirvId) -> SpirvId:
        """Return an integer-typed value suitable for OpAccessChain indexes."""
        index_type = self.registered_value_type(
            index
        ) or self.find_registered_type_by_base(index.type.base_type)
        index_type_name = (
            self.normalize_primitive_name(index_type.type.base_type)
            if index_type is not None
            else self.normalize_primitive_name(index.type.base_type)
        )
        if index_type_name in self.INTEGER_TYPE_NAMES:
            return index

        int_type = self.register_primitive_type("int")
        coerced = self.coerce_scalar_constant_to_type(index, int_type)
        if coerced.id != index.id:
            return coerced

        converted = self.convert_scalar_to_type(index, int_type)
        converted_type = self.registered_value_type(
            converted
        ) or self.find_registered_type_by_base(converted.type.base_type)
        converted_type_name = (
            self.normalize_primitive_name(converted_type.type.base_type)
            if converted_type is not None
            else self.normalize_primitive_name(converted.type.base_type)
        )
        if converted_type_name in self.INTEGER_TYPE_NAMES:
            return converted

        self.refuse_lowering("OpAccessChain index could not be converted to an integer")
        return self.register_constant(0, int_type)

    def composite_extract(
        self, composite: SpirvId, member_type: SpirvId, member_index: int
    ) -> SpirvId:
        """Extract a member from a composite value."""
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpCompositeExtract %{member_type.id} "
            f"%{composite.id} {member_index}"
        )

        spirv_id = SpirvId(id_value, member_type.type)
        self.value_types[id_value] = member_type
        return spirv_id

    def composite_insert(
        self,
        composite: SpirvId,
        value: SpirvId,
        composite_type: SpirvId,
        member_index: int,
    ) -> SpirvId:
        """Insert a member value into a composite value."""
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpCompositeInsert %{composite_type.id} "
            f"%{value.id} %{composite.id} {member_index}"
        )

        spirv_id = SpirvId(id_value, composite_type.type)
        self.value_types[id_value] = composite_type
        return spirv_id

    def vector_shuffle(
        self, vector: SpirvId, result_type: SpirvId, component_indices: List[int]
    ) -> SpirvId:
        """Create a vector by selecting components from an existing vector."""
        id_value = self.get_id()
        components = " ".join(str(index) for index in component_indices)
        self.emit(
            f"%{id_value} = OpVectorShuffle %{result_type.id} "
            f"%{vector.id} %{vector.id} {components}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def composite_construct(
        self, result_type: SpirvId, components: List[SpirvId]
    ) -> SpirvId:
        """Construct a composite value from component values."""
        result_type = self.ensure_registered_type(result_type)
        components = self.coerce_composite_components(
            result_type, components, constant=False
        )
        id_value = self.get_id()
        component_list = " ".join(f"%{component.id}" for component in components)
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{result_type.id} {component_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def composite_member_types(
        self, composite_type: SpirvId
    ) -> Optional[List[SpirvId]]:
        """Return the ordered constituent types for a composite result type.

        Handles vectors, matrices, arrays, and structs. Returns ``None`` when
        the constituent layout cannot be determined (for example a
        runtime-sized array), signalling callers to leave the components as-is.
        """
        vector_info = self.vector_type_info_from_type(composite_type)
        if vector_info is not None:
            component_type, count = vector_info
            return [component_type] * count

        matrix_info = self.matrix_type_info_from_type(composite_type)
        if matrix_info is not None:
            column_type, count = matrix_info
            return [column_type] * count

        array_info = self.array_type_info_from_type(composite_type)
        if array_info is not None:
            element_type, size = array_info
            if size is None:
                return None
            return [element_type] * size

        members = self.current_struct_members.get(composite_type.type.base_type)
        if members is not None:
            return [member_type for member_type, _ in members]

        return None

    def coerce_composite_components(
        self,
        composite_type: SpirvId,
        components: List[SpirvId],
        *,
        constant: bool,
    ) -> List[SpirvId]:
        """Coerce every constituent to its declared member/element type.

        ``OpCompositeConstruct`` and ``OpConstantComposite`` require each
        constituent type to match the corresponding member type of the result
        exactly. Callers do not always coerce (for example a brace initializer
        ``{x, 0}`` for a ``struct { float a; float b; }`` supplies an integer
        literal for a float member), so the coercion is centralised here to
        cover every construction site.

        Vector constructions may legitimately pass fewer constituents than the
        component count (for example a ``vec4`` assembled from two ``vec2``
        values), so coercion is only applied when the constituent count matches
        the member layout; otherwise the components are returned unchanged.
        """
        member_types = self.composite_member_types(composite_type)
        if member_types is None or len(member_types) != len(components):
            return components

        coerced = []
        changed = False
        for component, member_type in zip(components, member_types):
            new_component = self.coerce_composite_component(
                component, member_type, constant=constant
            )
            changed = changed or new_component.id != component.id
            coerced.append(new_component)
        return coerced if changed else components

    def coerce_composite_component(
        self, component: SpirvId, member_type: SpirvId, *, constant: bool
    ) -> SpirvId:
        """Coerce a single constituent to ``member_type``.

        Constant scalar literals are folded to a member-typed constant so the
        constituent matches exactly (an integer ``0`` used for a ``float``
        member becomes a ``float`` constant). Runtime constituents fall back to
        the general value conversion. Constant composites keep any non-foldable
        constituent unchanged because runtime conversions are not permitted in a
        constant context.
        """
        member_type = self.ensure_registered_type(member_type)
        folded = self.coerce_scalar_constant_to_type(component, member_type)
        if folded.id != component.id:
            return folded
        if constant:
            return specialization_constituent(self, component, member_type)
        return self.convert_value_to_type(component, member_type)

    def vector_member_info(self, vector_type: str, member_name: str):
        vector_info = self.vector_component_type_and_count(vector_type)
        if vector_info is None:
            return None

        component_type, component_count = vector_info
        component_indices = {
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
        }
        member_index = component_indices.get(member_name)
        if member_index is None or member_index >= component_count:
            return None

        return member_index, self.register_primitive_type(component_type)

    def vector_swizzle_indices(self, vector_type: str, member_name: str):
        vector_info = self.vector_component_type_and_count(vector_type)
        if vector_info is None or not 2 <= len(member_name) <= 4:
            return None

        component_indices = (
            {"x": 0, "y": 1, "z": 2, "w": 3},
            {"r": 0, "g": 1, "b": 2, "a": 3},
            {"s": 0, "t": 1, "p": 2, "q": 3},
        )
        family = next(
            (
                indices
                for indices in component_indices
                if all(component in indices for component in member_name)
            ),
            None,
        )
        if family is None:
            return None

        _, component_count = vector_info
        indices = [family[component] for component in member_name]
        if any(index >= component_count for index in indices):
            return None

        return indices

    def vector_swizzle_info(self, vector_type: str, member_name: str):
        vector_info = self.vector_component_type_and_count(vector_type)
        if vector_info is None:
            return None

        indices = self.vector_swizzle_indices(vector_type, member_name)
        if indices is None:
            return None

        component_type_name, _ = vector_info
        component_type = self.register_primitive_type(component_type_name)
        result_type = self.register_vector_type(component_type, len(indices))
        return indices, component_type, result_type

    SCALAR_SWIZZLE_COMPONENT_TYPES = frozenset(
        {"float", "double", "int", "uint", "bool", "i64", "u64"}
    )

    def scalar_swizzle(self, base: SpirvId, member_name) -> Optional[SpirvId]:
        """A component selection on a scalar value (HLSL, GLSL 4.20 section 5.5): ``s.x`` is ``s``, ``s.xxx`` a splat.

        The scalar has one component, so only the first component name (``x``, ``r`` or ``s``) may be selected.
        """
        if not isinstance(member_name, str) or not 1 <= len(member_name) <= 4:
            return None
        if len(set(member_name)) != 1 or member_name[0] not in "xrs":
            return None
        scalar_name = self.normalize_primitive_name(str(base.type.base_type))
        if scalar_name not in self.SCALAR_SWIZZLE_COMPONENT_TYPES:
            return None
        if len(member_name) == 1:
            return base
        scalar_type = self.register_primitive_type(scalar_name)
        vector_type = self.register_vector_type(scalar_type, len(member_name))
        return self.splat_scalar_to_vector(base, vector_type)

    def struct_member_info(self, struct_type: str, member_name: str):
        members = self.current_struct_members.get(struct_type)
        if not members:
            return None
        for index, (member_type, name) in enumerate(members):
            if name == member_name:
                return index, member_type
        return None

    def struct_type_name_from_pointer(self, pointer: SpirvId):
        struct_type_id = self.variable_value_types.get(pointer.id)
        return struct_type_id.type.base_type if struct_type_id else None

    def create_member_access_pointer(
        self, base_pointer: SpirvId, member_name: str
    ) -> Optional[SpirvId]:
        struct_type = self.struct_type_name_from_pointer(base_pointer)
        member_info = self.struct_member_info(struct_type, member_name)
        if member_info is None:
            return None

        member_index, member_type = member_info
        int_type = self.primitive_types["int"]
        index = self.register_constant(member_index, int_type)
        storage_class = base_pointer.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(member_type, storage_class)
        access = self.access_chain(base_pointer, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        metadata = self.structured_buffer_metadata_for_pointer(base_pointer)
        if metadata is not None and metadata.get("_access_path") in {
            "element",
            "member",
        }:
            self.structured_buffer_metadata[access.id] = {
                **metadata,
                "_access_path": "member",
            }
        self.propagate_storage_buffer_access_metadata(base_pointer, access)
        self.propagate_readonly_builtin_pointer_name(base_pointer, access)
        self.propagate_readonly_pointer_name(base_pointer, access)
        return access

    def create_function(
        self,
        name: str,
        return_type: SpirvId,
        param_types: List[SpirvId],
        *,
        register_global: bool = True,
    ) -> SpirvId:
        """Create a function declaration."""
        function_type = self.register_function_type(return_type, param_types)

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpFunction %{return_type.id} None %{function_type.id}"
        )

        spirv_id = SpirvId(id_value, return_type.type, name)
        if register_global:
            self.functions[name] = spirv_id
            self.function_signatures[name] = (return_type, param_types)
            self.function_overloads.setdefault(name, []).append(
                (spirv_id, (return_type, param_types))
            )

        return spirv_id

    def stage_local_function_key(self, stage, name: str):
        if stage is None:
            return None
        return (id(stage), name)

    def stage_global_variable_key(self, stage, name: str):
        if stage is None:
            return None
        return (id(stage), name)

    def current_stage_global_variable(self, name: str) -> Optional[SpirvId]:
        key = self.stage_global_variable_key(self.current_stage, name)
        if key is None:
            return None
        return self.stage_global_variables.get(key)

    def resolve_global_variable(self, name: str) -> Optional[SpirvId]:
        return self.current_stage_global_variable(name) or self.global_variables.get(
            name
        )

    def global_variable_exists_for_current_stage(self, name: str) -> bool:
        if self.current_stage_global_variable(name) is not None:
            return True
        return name in self.global_variables

    def register_global_variable_name(
        self, name: str, variable: SpirvId, storage_class: str
    ):
        if self.current_stage is not None and storage_class in {"Input", "Output"}:
            key = self.stage_global_variable_key(self.current_stage, name)
            self.stage_global_variables[key] = variable
            return
        self.global_variables[name] = variable

    def register_stage_local_function(
        self,
        stage,
        name: str,
        function_id: SpirvId,
        return_type: SpirvId,
        param_types: List[SpirvId],
    ):
        key = self.stage_local_function_key(stage, name)
        if key is None:
            return
        self.stage_local_functions[key] = function_id
        self.stage_local_function_signatures[key] = (return_type, param_types)
        self.stage_local_function_overloads.setdefault(key, []).append(
            (function_id, (return_type, param_types))
        )

    def function_reference_candidates(self, function_name: str):
        candidates = []
        seen = set()
        if self.current_stage is not None:
            key = self.stage_local_function_key(self.current_stage, function_name)
            for candidate in self.stage_local_function_overloads.get(key, []):
                function_id, _signature = candidate
                if function_id.id in seen:
                    continue
                candidates.append(candidate)
                seen.add(function_id.id)

        for candidate in self.function_overloads.get(function_name, []):
            function_id, _signature = candidate
            if function_id.id in seen:
                continue
            candidates.append(candidate)
            seen.add(function_id.id)
        return candidates

    def function_argument_type_match_score(
        self, expected_type: SpirvId, actual_type: Optional[SpirvId]
    ) -> Optional[int]:
        if expected_type is None:
            return 0 if actual_type is None else None
        if actual_type is None:
            return 0
        if actual_type.id == expected_type.id:
            return 16
        if actual_type.type.base_type == expected_type.type.base_type:
            return 14

        expected_storage = expected_type.type.storage_class
        actual_storage = actual_type.type.storage_class
        if expected_storage is not None or actual_storage is not None:
            if expected_storage != actual_storage:
                return None
            expected_pointee = self.pointer_type_pointee_type(expected_type)
            actual_pointee = self.pointer_type_pointee_type(actual_type)
            pointee_score = self.function_argument_type_match_score(
                expected_pointee, actual_pointee
            )
            if pointee_score is None:
                return None
            return 8 + pointee_score

        expected_name = self.normalize_signature_type_name(expected_type.type.base_type)
        actual_name = self.normalize_signature_type_name(actual_type.type.base_type)
        if expected_name is None or actual_name is None:
            return 0
        if expected_name == actual_name:
            return 12

        numeric_types = {"bool", "float", "double"} | self.INTEGER_TYPE_NAMES
        expected_vector = self.vector_component_type_and_count(expected_name)
        actual_vector = self.vector_component_type_and_count(actual_name)
        if expected_vector is not None or actual_vector is not None:
            if expected_vector is None:
                return None
            if actual_vector is None:
                if expected_vector[0] in numeric_types and actual_name in numeric_types:
                    return 2
                return None
            if expected_vector[1] != actual_vector[1]:
                return None
            if (
                expected_vector[0] in numeric_types
                and actual_vector[0] in numeric_types
            ):
                return 7
            return None

        if expected_name in numeric_types and actual_name in numeric_types:
            return 6
        if self.scalar_or_vector_type_compatible(expected_name, actual_name):
            return 1
        return None

    def function_call_match_score(
        self, param_types: List[SpirvId], args: List[SpirvId]
    ) -> Optional[int]:
        if len(param_types) != len(args):
            return None
        score = 0
        for expected_type, arg in zip(param_types, args):
            actual_type = self.registered_value_type(arg)
            arg_score = self.function_argument_type_match_score(
                expected_type, actual_type
            )
            if arg_score is None:
                return None
            score += arg_score
        return score

    def select_function_reference_for_args(
        self, function_name: str, args: List[SpirvId]
    ):
        candidates = self.function_reference_candidates(function_name)
        if not candidates:
            return None

        scored_candidates = []
        for candidate in candidates:
            _function_id, (_return_type, param_types) = candidate
            score = self.function_call_match_score(param_types, args)
            if score is not None:
                scored_candidates.append((score, candidate))

        if scored_candidates:
            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            return scored_candidates[0][1]
        return candidates[-1]

    def resolve_function_reference(
        self, function_name: str, args: Optional[List] = None
    ):
        if args is not None:
            function_reference = self.select_function_reference_for_args(
                function_name, args
            )
            if function_reference is not None:
                return function_reference

        if self.current_stage is not None:
            key = self.stage_local_function_key(self.current_stage, function_name)
            if key in self.stage_local_functions:
                return (
                    self.stage_local_functions[key],
                    self.stage_local_function_signatures[key],
                )
        if function_name in self.functions:
            return (
                self.functions[function_name],
                self.function_signatures[function_name],
            )
        return None

    def resolve_function_signature(
        self, function_name: str, args: Optional[List] = None
    ):
        function_reference = self.resolve_function_reference(function_name, args)
        if function_reference is None:
            return None
        return function_reference[1]

    def has_function_reference(self, function_name: str) -> bool:
        return self.resolve_function_reference(function_name) is not None

    def stage_local_metadata(self, mapping, function_name: str):
        if self.current_stage is None:
            return None
        key = self.stage_local_function_key(self.current_stage, function_name)
        return mapping.get(key)

    def resolve_function_resource_array_params(self, function_name: str):
        params = self.stage_local_metadata(
            self.stage_local_function_resource_array_params, function_name
        )
        if params is not None:
            return params
        return self.function_resource_array_params.get(function_name, set())

    def resolve_function_resource_array_type_hints(self, function_name: str):
        hints = self.stage_local_metadata(
            self.stage_local_function_resource_array_type_hints, function_name
        )
        if hints is not None:
            return hints
        return self.function_resource_array_type_hints.get(function_name, {})

    def resolve_function_mesh_output_parameter_indices(self, function_name: str):
        indices = self.stage_local_metadata(
            self.stage_local_function_mesh_output_parameter_indices, function_name
        )
        if indices is not None:
            return indices
        return self.function_mesh_output_parameter_indices.get(function_name, set())

    def resolve_function_parameter_names(self, function_name: str):
        names = self.stage_local_metadata(
            self.stage_local_function_parameter_names, function_name
        )
        if names is not None:
            return names
        return self.function_parameter_names.get(function_name, [])

    def resolve_function_image_access_requirements(self, function_name: str):
        requirements = self.stage_local_metadata(
            self.stage_local_function_image_access_requirements, function_name
        )
        if requirements is not None:
            return requirements
        return self.function_image_access_requirements.get(function_name)

    def resolve_function_storage_buffer_access_requirements(self, function_name: str):
        requirements = self.stage_local_metadata(
            self.stage_local_function_storage_buffer_access_requirements,
            function_name,
        )
        if requirements is not None:
            return requirements
        return self.function_storage_buffer_access_requirements.get(function_name)

    def resolve_function_storage_image_pointer_params(self, function_name: str):
        params = self.stage_local_metadata(
            self.stage_local_function_storage_image_pointer_params,
            function_name,
        )
        if params is not None:
            return params
        return self.function_storage_image_pointer_params.get(function_name, set())

    def function_parameter_requires_storage_image_pointer(
        self, function_name: str, arg_index: int
    ) -> bool:
        pointer_param_names = self.resolve_function_storage_image_pointer_params(
            function_name
        )
        if not pointer_param_names:
            return False

        parameter_names = self.resolve_function_parameter_names(function_name)
        return (
            arg_index < len(parameter_names)
            and parameter_names[arg_index] in pointer_param_names
        )

    def resolve_inline_storage_buffer_function(
        self, function_name: str, call_args, call_node=None
    ):
        candidates = []
        seen_candidates = set()
        stage_candidates = self.stage_local_metadata(
            self.stage_local_inline_storage_buffer_functions, function_name
        )
        if stage_candidates is not None:
            for candidate in stage_candidates:
                candidates.append(candidate)
                seen_candidates.add(id(candidate))

        for candidate in self.inline_storage_buffer_functions.get(function_name, []):
            candidate_key = id(candidate)
            if candidate_key in seen_candidates:
                continue
            candidates.append(candidate)
            seen_candidates.add(candidate_key)

        if not candidates:
            return None

        scored_matches = []
        rejection_reasons = {}
        for candidate in candidates:
            match, rejection_reason = self.storage_buffer_candidate_call_match(
                candidate, call_args
            )
            if match is None:
                rejection_reasons[id(candidate)] = rejection_reason
                continue
            specificity_score = 0 if generic_function_parameters(candidate) else 1
            scored_matches.append(
                (match["rank_score"], match["score"], specificity_score, candidate)
            )

        if not scored_matches:
            if len(candidates) == 1:
                return candidates[0]
            if self.has_matching_non_storage_function_candidate(
                function_name, call_args
            ):
                return None
            raise self.storage_buffer_overload_diagnostic(
                function_name,
                call_args,
                candidates,
                "no candidate has compatible argument alignment",
                call_node=call_node,
                rejection_reasons=rejection_reasons,
            )

        scored_matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best_rank_score, best_score, best_specificity_score, best_candidate = (
            scored_matches[0]
        )
        tied_candidates = [
            candidate
            for rank_score, score, specificity_score, candidate in scored_matches
            if (
                rank_score == best_rank_score
                and score == best_score
                and specificity_score == best_specificity_score
            )
        ]
        if len(tied_candidates) > 1:
            raise self.storage_buffer_overload_diagnostic(
                function_name,
                call_args,
                tied_candidates,
                "multiple candidates have compatible argument types",
                call_node=call_node,
                rejection_reasons=rejection_reasons,
            )
        return best_candidate
