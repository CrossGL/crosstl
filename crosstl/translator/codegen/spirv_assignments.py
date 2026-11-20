"""Array literals and assignments in the SPIR-V generator: array literals, typed constants, vector list constructors, assignable pointers, resource arrays, tessellation indices, readonly pointers, assignments, vector component updates and pointers, storage references, swizzle stores and compound assignment operators."""

from typing import (
    List,
    Optional,
    Tuple,
)

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    ConstructorNode,
    CooperativeMatrixOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from .pointer_reinterpret import PointerReinterpretationError
from .spirv_constant_folding import spirv_folded_constant
from .spirv_constant_matrices import constant_matrix_constructor
from .spirv_single_evaluation import evaluate_subscript_index
from .spirv_specialization_conversions import specialization_constituent
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)


class SpirvAssignmentMixin:
    """Assignment members of ``VulkanSPIRVCodeGen``."""

    def process_array_literal(
        self,
        expr: ArrayLiteralNode,
        target_type: Optional[SpirvId] = None,
        constant: bool = False,
    ) -> Optional[SpirvId]:
        if not expr.elements:
            if target_type is None:
                raise UnsupportedSPIRVFeatureError(
                    "untyped empty initializer",
                    "SPIR-V cannot infer a value type from an empty initializer; "
                    "preserve its declared or parameter type before code generation.",
                    missing_capabilities=("spirv.empty_initializer_type_inference",),
                    source_location=getattr(expr, "source_location", None),
                )
            target_type = self.ensure_registered_type(target_type)
            if self.type_contains_runtime_array(target_type):
                raise UnsupportedSPIRVFeatureError(
                    "runtime-sized empty initializer",
                    "SPIR-V cannot materialize an empty aggregate value for a "
                    "type containing a runtime-sized array.",
                    missing_capabilities=("spirv.runtime_array_aggregate_initializer",),
                    source_location=getattr(expr, "source_location", None),
                )
            if self.type_contains_specialization_sized_array(target_type):
                raise UnsupportedSPIRVFeatureError(
                    "specialization-sized array initializer",
                    "SPIR-V aggregate initializers require a fixed constituent "
                    "count; the destination contains a specialization-sized array.",
                    missing_capabilities=("spirv.specialization_array_initializer",),
                    source_location=getattr(expr, "source_location", None),
                )
            return self.default_value_for_type(target_type)

        array_type = target_type
        element_type = None
        target_size = None
        constituent_types = None

        if array_type is not None:
            array_type = self.ensure_registered_type(array_type)
            array_info = self.array_type_info_from_type(array_type)
            if array_info is not None:
                element_type, target_size = array_info
                if self.named_specialization_array_length(target_size) is not None:
                    raise UnsupportedSPIRVFeatureError(
                        "specialization-sized array initializer",
                        "SPIR-V aggregate initializers require a fixed constituent "
                        f"count; array extent '{target_size}' is specialization-dependent",
                        missing_capabilities=(
                            "spirv.specialization_array_initializer",
                        ),
                    )
            constituent_types = self.composite_member_types(array_type)

        values = []
        for element_index, element in enumerate(expr.elements):
            constituent_type = element_type
            if constituent_types is not None and element_index < len(constituent_types):
                constituent_type = constituent_types[element_index]
            value = self.process_array_literal_element(
                element, constituent_type, constant
            )
            if value is None:
                return None
            values.append(value)

        if array_type is None:
            if values:
                element_type = self.value_types.get(
                    values[0].id
                ) or self.find_registered_type_by_base(values[0].type.base_type)
            if element_type is None:
                element_type = self.register_primitive_type("float")
            target_size = len(values)
            array_type = self.register_array_type(element_type, target_size)
            values = [
                self.convert_value_to_type(value, element_type) for value in values
            ]

        if constituent_types is not None:
            values = values[: len(constituent_types)]
            while len(values) < len(constituent_types):
                values.append(
                    self.default_value_for_type(constituent_types[len(values)])
                )
        elif target_size is not None:
            values = values[:target_size]
            if element_type is not None:
                while len(values) < target_size:
                    values.append(self.default_value_for_type(element_type))

        if constant:
            if not all(self.is_constant_instruction(value) for value in values):
                return None
            return self.register_composite_constant(array_type, values)

        # Route through composite_construct so every constituent is coerced to
        # its member/element type. Brace initializers targeting a struct (for
        # example ``return {x, 0};``) reach this path with an array-literal
        # node, and their constituents must match the struct member types.
        return self.composite_construct(array_type, values)

    def process_array_literal_element(
        self,
        element,
        target_type: Optional[SpirvId],
        constant: bool,
    ) -> Optional[SpirvId]:
        if isinstance(element, ArrayLiteralNode):
            return self.process_array_literal(element, target_type, constant)

        if constant:
            return self.process_constant_expression(element, target_type)

        value = self.process_expression(element)
        if value is not None and target_type is not None:
            value = self.convert_value_to_type(value, target_type)
        return value

    def constant_literal_for_type(
        self, expr, target_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        if target_type is None:
            return None

        target_type = self.ensure_registered_type(target_type)
        if self.vector_component_type_and_count(target_type.type.base_type) is not None:
            return None

        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        value = self.constant_scalar_literal_value(expr)
        if value is None:
            return None

        if target_type_name == "bool":
            if isinstance(value, bool):
                return self.register_constant(value, target_type)
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        if target_type_name in {"float", "double"}:
            return self.register_constant(float(value), target_type)
        if target_type_name == self.BFLOAT16_TYPE_NAME:
            return self.register_constant(
                self.bfloat16_bits_for_float(float(value)), target_type
            )
        if target_type_name in self.INTEGER_TYPE_NAMES:
            if target_type_name in self.UNSIGNED_INTEGER_TYPES and value < 0:
                return None
            return self.register_constant(int(value), target_type)
        return None

    def constant_scalar_literal_value(self, expr):
        if isinstance(expr, UnaryOpNode):
            value = self.constant_scalar_literal_value(expr.operand)
            if value is None or isinstance(value, bool):
                return None
            operator = getattr(expr, "op", getattr(expr, "operator", None))
            if operator == "-":
                return -value
            if operator == "+":
                return value
            return None

        if isinstance(expr, LiteralNode):
            return expr.value

        if isinstance(expr, (bool, int, float)):
            return expr

        return None

    def process_vector_list_constructor(self, expr, *, constant=False):
        if not isinstance(expr, ConstructorNode) or not getattr(
            expr, "is_braced_constructor", False
        ):
            return None
        type_name = self.convert_type_node_to_string(expr.constructor_type)
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is None:
            return None
        if expr.named_arguments or len(expr.arguments) > vector_info[1]:
            raise UnsupportedSPIRVFeatureError(
                "vector-list-initialization",
                f"Vector initializer for {type_name} requires at most "
                f"{vector_info[1]} positional components",
                source_location=getattr(expr, "source_location", None),
            )
        return self.process_array_literal(
            ArrayLiteralNode(list(expr.arguments)),
            self.map_crossgl_type(expr.constructor_type),
            constant=constant,
        )

    def process_constant_expression(
        self,
        expr,
        target_type: Optional[SpirvId] = None,
    ) -> Optional[SpirvId]:
        if isinstance(expr, ArrayLiteralNode):
            return self.process_array_literal(expr, target_type, constant=True)

        vector_list = self.process_vector_list_constructor(expr, constant=True)
        if vector_list is not None:
            return vector_list

        converted_literal = self.constant_literal_for_type(expr, target_type)
        if converted_literal is not None:
            return converted_literal

        if isinstance(expr, LiteralNode):
            return self.process_expression(expr)

        folded = spirv_folded_constant(self, expr, target_type)
        if folded is not None:
            return folded

        if isinstance(expr, FunctionCallNode):
            callee_expr = getattr(expr, "function", getattr(expr, "name", None))
            callee_name = getattr(callee_expr, "name", callee_expr)
            if target_type is not None:
                struct_constant = self.process_constant_struct_constructor(
                    callee_name, expr, target_type
                )
                if struct_constant is not None:
                    return struct_constant

            matrix_constant = constant_matrix_constructor(
                self, expr, callee_name, target_type
            )
            if matrix_constant is not None:
                return matrix_constant

            vector_type = target_type
            if vector_type is None and isinstance(callee_name, str):
                if self.vector_component_type_and_count(callee_name) is not None:
                    vector_type = self.map_crossgl_type(callee_name)

            if vector_type is not None:
                vector_info = self.vector_type_info_from_type(vector_type)
                if vector_info is not None:
                    component_type, component_count = vector_info
                    components = []
                    for arg in getattr(expr, "args", []):
                        value = self.process_constant_expression(arg, component_type)
                        if value is None:
                            return None
                        components.append(
                            specialization_constituent(self, value, component_type)
                        )

                    if len(components) == 1 and component_count > 1:
                        components *= component_count

                    components = components[:component_count]
                    while len(components) < component_count:
                        components.append(self.default_value_for_type(component_type))

                    return self.register_vector_constant(vector_type, components)

        specialization_constant_op = self.process_specialization_constant_operation(
            expr, target_type
        )
        if specialization_constant_op is not None:
            return specialization_constant_op

        value = self.process_expression(expr)
        if value is not None and self.is_constant_instruction(value):
            return value
        return None

    def coerce_scalar_constant_to_type(
        self, value: SpirvId, target_type: SpirvId
    ) -> SpirvId:
        source_type = self.value_types.get(value.id)
        if source_type is not None and source_type.id == target_type.id:
            return value

        literal_value = self.constant_values_by_id.get(value.id)
        if literal_value is None or isinstance(literal_value, bool):
            return value

        target_type = self.ensure_registered_type(target_type)
        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        if target_type_name in {"float", "double"} and isinstance(
            literal_value, (int, float)
        ):
            return self.register_constant(float(literal_value), target_type)
        if (
            target_type_name in self.INTEGER_TYPE_NAMES
            and isinstance(literal_value, (int, float))
            and float(literal_value).is_integer()
        ):
            int_value = int(literal_value)
            if target_type_name in self.UNSIGNED_INTEGER_TYPES and int_value < 0:
                return value
            return self.register_constant(int_value, target_type)
        return value

    def process_constant_struct_constructor(
        self, callee_name, expr, target_type: SpirvId
    ) -> Optional[SpirvId]:
        struct_name = target_type.type.base_type
        if callee_name != struct_name:
            return None

        members = self.current_struct_members.get(struct_name)
        if members is None:
            return None

        args = getattr(expr, "args", [])
        if len(args) == 1 and self.expression_result_type(args[0]) == struct_name:
            copy_value = self.process_constant_expression(args[0], target_type)
            if copy_value is not None and self.value_has_type(copy_value, target_type):
                return copy_value
        values = []
        for index, (member_type, _) in enumerate(members):
            if index < len(args):
                value = self.process_constant_expression(args[index], member_type)
                if value is None:
                    return None
                values.append(value)
            else:
                values.append(self.default_value_for_type(member_type))

        if len(args) < len(members):
            self.refuse_lowering(
                f"Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}"
            )
        elif len(args) > len(members):
            self.refuse_lowering(
                f"Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}"
            )

        return self.register_composite_constant(target_type, values)

    def ensure_assignable_pointer_for_name(self, name: str) -> Optional[SpirvId]:
        var_id = self.local_variables.get(name)
        if var_id is not None:
            if var_id.type.storage_class:
                return var_id

            value_type = self.value_types.get(var_id.id)
            if value_type is None:
                return None

            mutable_id = self.create_variable(value_type, "Function", name)
            self.store_to_variable(mutable_id, var_id)
            self.local_variables[name] = mutable_id
            return mutable_id

        var_id = (
            self.resolve_global_variable(name)
            or self.ensure_tessellation_patch_constant_input(name)
            or self.ensure_stage_builtin(name)
        )
        self.mark_interface_variable_if_needed(var_id)
        wrapped_pointer = self.uniform_block_wrapped_member_pointer(var_id)
        if wrapped_pointer is not None:
            return wrapped_pointer
        return var_id

    def assignable_pointer_from_expression(self, expr) -> Optional[SpirvId]:
        if (
            isinstance(expr, UnaryOpNode)
            and not getattr(expr, "is_postfix", False)
            and expr.op == "*"
        ):
            return self.dereference_pointer_from_expression(expr.operand)
        if isinstance(expr, IdentifierNode):
            return self.ensure_assignable_pointer_for_name(expr.name)
        if isinstance(expr, VariableNode):
            return self.ensure_assignable_pointer_for_name(expr.name)
        if isinstance(expr, str):
            return self.ensure_assignable_pointer_for_name(expr)
        if isinstance(expr, MemberAccessNode):
            base_pointer = self.assignable_pointer_from_expression(expr.object)
            if base_pointer is None:
                return None
            access = self.create_member_access_pointer(base_pointer, expr.member)
            if access is None:
                access = self.vector_component_reference_pointer(
                    base_pointer, expr.member
                )
            return access
        if isinstance(expr, ArrayAccessNode):
            index = evaluate_subscript_index(self, expr.index)
            if index is None:
                return None
            if not self.validate_tessellation_patch_builtin_index(
                expr.array, index, expr.index
            ):
                return None

            array_variable = self.assignable_pointer_from_expression(expr.array)
            if array_variable is None:
                return None
            reinterpretation = self.storage_pointer_reinterpretations.get(
                array_variable
            )
            if reinterpretation is not None:
                raise PointerReinterpretationError(
                    "SPIR-V cannot preserve writes through a reinterpreted typed "
                    "storage-buffer view",
                    source_type=reinterpretation["source_type"],
                    target_type=reinterpretation["target_type"],
                    address_space="storage",
                    alignment=reinterpretation["target_layout"].byte_width,
                    access="write",
                    reason="write-requires-byte-address-storage",
                    source_location=getattr(expr, "source_location", None),
                )
            if not self.validate_tessellation_patch_parameter_index(
                array_variable, index, expr.index
            ):
                return None

            # An offset-aliased pointer (e.g. `p = &shared[base]`) resolves
            # `p[j]` to its root array at `base + j` rather than indexing the
            # element pointer itself.
            offset_alias = self.storage_buffer_pointer_offset_aliases.get(
                self.storage_buffer_pointer_alias_key(array_variable)
            )
            if offset_alias is not None:
                root_pointer, combined_index, _ = (
                    self.storage_buffer_pointer_alias_access(array_variable, index)
                )
                structured_access = self.structured_buffer_element_pointer(
                    root_pointer,
                    combined_index,
                    apply_pointer_offset=False,
                )
                if structured_access is not None:
                    return structured_access
                generic_access = self.local_array_offset_element_pointer(
                    root_pointer, combined_index
                )
                if generic_access is not None:
                    return generic_access

            structured_access = self.structured_buffer_element_pointer(
                array_variable, index
            )
            if structured_access is not None:
                return structured_access

            array_type = self.variable_value_types.get(array_variable.id)
            element_type = self.array_element_type_from_type(array_type)
            if element_type is None:
                element_type = self.determine_array_element_type(array_variable)
            if element_type is None:
                return None

            storage_class = array_variable.type.storage_class or "Function"
            ptr_type = self.register_pointer_type(element_type, storage_class)
            access = self.access_chain(array_variable, [index], ptr_type)
            self.variable_value_types[access.id] = element_type
            self.propagate_storage_buffer_access_metadata(array_variable, access)
            self.propagate_structured_buffer_descriptor_access_metadata(
                array_variable, access, index
            )
            self.propagate_resource_access_metadata(
                array_variable, access, element_type
            )
            self.propagate_readonly_builtin_pointer_name(array_variable, access)
            self.propagate_readonly_pointer_name(array_variable, access)
            return access
        return None

    def is_resource_array_type(self, array_type: Optional[SpirvId]) -> bool:
        element_type = self.array_element_type_from_type(array_type)
        while element_type is not None:
            if element_type.id in self.resource_type_metadata:
                return True
            element_type = self.array_element_type_from_type(element_type)
        return False

    def tessellation_patch_builtin_size(self, name: Optional[str]) -> Optional[int]:
        return {"gl_TessLevelOuter": 4, "gl_TessLevelInner": 2}.get(name or "")

    def validate_tessellation_integer_index(
        self,
        diagnostic_subject: str,
        size: int,
        index: SpirvId,
        index_expr=None,
    ) -> bool:
        index_type = self.value_types.get(
            index.id
        ) or self.find_registered_type_by_base(index.type.base_type)
        index_type_name = (
            self.normalize_primitive_name(index_type.type.base_type)
            if index_type is not None
            else index.type.base_type
        )
        if index_type_name not in {"int", "uint"}:
            self.refuse_lowering(
                f"{diagnostic_subject} "
                f"index requires a scalar integer value, got {index.type.base_type}"
            )
            return False

        literal_index = self.literal_int_argument(index_expr)
        if literal_index is not None and not 0 <= literal_index < size:
            self.refuse_lowering(
                f"{diagnostic_subject} "
                f"index {literal_index} out of range; valid range is 0..{size - 1}"
            )
            return False

        return True

    def validate_tessellation_patch_builtin_index(
        self, array_expr, index: SpirvId, index_expr=None
    ) -> bool:
        name = self.direct_expression_name(array_expr)
        size = self.tessellation_patch_builtin_size(name)
        if size is None:
            return True

        return self.validate_tessellation_integer_index(
            f"SPIR-V tessellation patch builtin {name} component",
            size,
            index,
            index_expr,
        )

    def validate_tessellation_patch_parameter_index(
        self, array_variable: SpirvId, index: SpirvId, index_expr=None
    ) -> bool:
        metadata = self.patch_parameter_metadata.get(array_variable.id)
        if metadata is None:
            return True

        return self.validate_tessellation_integer_index(
            f"SPIR-V tessellation patch parameter {metadata['name']} " "control-point",
            metadata["control_points"],
            index,
            index_expr,
        )

    def propagate_readonly_builtin_pointer_name(
        self, source_pointer: SpirvId, target_pointer: SpirvId
    ):
        builtin_name = self.readonly_builtin_pointer_names.get(source_pointer.id)
        if builtin_name is not None:
            self.readonly_builtin_pointer_names[target_pointer.id] = builtin_name

    def propagate_readonly_pointer_name(
        self, source_pointer: SpirvId, target_pointer: SpirvId
    ):
        pointer_name = self.readonly_pointer_names.get(source_pointer.id)
        if pointer_name is not None:
            self.readonly_pointer_names[target_pointer.id] = pointer_name

    def create_array_element_access(self, array_expr, index: SpirvId, index_expr=None):
        if not self.validate_tessellation_patch_builtin_index(
            array_expr, index, index_expr
        ):
            return None, None

        array_variable = self.variable_pointer_from_expression(array_expr)
        if array_variable is None or not array_variable.type.storage_class:
            addressable_array = self.assignable_pointer_from_expression(array_expr)
            if addressable_array is not None:
                array_variable = addressable_array

        if (
            array_variable is None
            and self.tessellation_patch_builtin_size(
                self.direct_expression_name(array_expr)
            )
            is not None
        ):
            return None, None

        if array_variable is not None:
            if not self.validate_tessellation_patch_parameter_index(
                array_variable, index, index_expr
            ):
                return None, None

            block_alias = self.single_struct_buffer_zero_index_alias(
                array_variable, index_expr
            )
            if block_alias is not None:
                return block_alias, self.variable_value_types.get(block_alias.id)

            offset_alias = self.storage_buffer_pointer_offset_aliases.get(
                self.storage_buffer_pointer_alias_key(array_variable)
            )
            if offset_alias is not None:
                root_pointer, combined_index, _ = (
                    self.storage_buffer_pointer_alias_access(array_variable, index)
                )
                structured_access = self.structured_buffer_element_pointer(
                    root_pointer,
                    combined_index,
                    apply_pointer_offset=False,
                )
                if structured_access is not None:
                    return structured_access, self.variable_value_types.get(
                        structured_access.id
                    )
                # A plain local-array root (threadgroup/function `shared[...]`):
                # index it directly at the combined offset.
                generic_access = self.local_array_offset_element_pointer(
                    root_pointer, combined_index
                )
                if generic_access is not None:
                    return generic_access, self.variable_value_types.get(
                        generic_access.id
                    )

            structured_access = self.structured_buffer_element_pointer(
                array_variable, index
            )
            if structured_access is not None:
                return structured_access, self.variable_value_types.get(
                    structured_access.id
                )

            array_type = self.variable_value_types.get(array_variable.id)
            element_type = self.array_element_type_from_type(array_type)
            vector_info = (
                self.vector_component_type_and_count(array_type.type.base_type)
                if element_type is None and array_type is not None
                else None
            )
            if vector_info is not None:
                # A subscript of a vector, such as a matrix column, selects one component.
                element_type = self.register_primitive_type(vector_info[0])
            if (
                element_type is None
                and array_type is not None
                and array_type.type.base_type not in self.current_struct_members
            ):
                return array_variable, array_type
            if element_type is None:
                element_type = self.determine_array_element_type(array_variable)
            if element_type is None:
                return None, None

            storage_class = array_variable.type.storage_class or "Function"
            ptr_type = self.register_pointer_type(element_type, storage_class)
            access = self.access_chain(array_variable, [index], ptr_type)
            self.variable_value_types[access.id] = element_type
            self.propagate_storage_buffer_access_metadata(array_variable, access)
            self.propagate_structured_buffer_descriptor_access_metadata(
                array_variable, access, index
            )
            self.propagate_resource_access_metadata(
                array_variable, access, element_type
            )
            self.propagate_readonly_builtin_pointer_name(array_variable, access)
            self.propagate_readonly_pointer_name(array_variable, access)
            return access, element_type

        array = self.process_expression(array_expr)
        if array is None:
            return None, None

        element_type = self.determine_array_element_type(array)
        if element_type is None:
            return None, None

        array_type = self.value_types.get(
            array.id
        ) or self.find_registered_type_by_base(array.type.base_type)
        if array_type is not None and not array.type.storage_class:
            array_variable = self.create_variable(array_type, "Function")
            self.store_to_variable(array_variable, array)
            storage_class = array_variable.type.storage_class or "Function"
            ptr_type = self.register_pointer_type(element_type, storage_class)
            access = self.access_chain(array_variable, [index], ptr_type)
            self.variable_value_types[access.id] = element_type
            self.propagate_storage_buffer_access_metadata(array_variable, access)
            self.propagate_structured_buffer_descriptor_access_metadata(
                array_variable, access, index
            )
            self.propagate_resource_access_metadata(
                array_variable, access, element_type
            )
            self.propagate_readonly_builtin_pointer_name(array_variable, access)
            self.propagate_readonly_pointer_name(array_variable, access)
            return access, element_type

        storage_class = array.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(element_type, storage_class)
        access = self.access_chain(array, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.propagate_storage_buffer_access_metadata(array, access)
        self.propagate_structured_buffer_descriptor_access_metadata(
            array, access, index
        )
        self.propagate_resource_access_metadata(array, access, element_type)
        self.propagate_readonly_builtin_pointer_name(array, access)
        self.propagate_readonly_pointer_name(array, access)
        return access, element_type

    def process_assignment(self, node: AssignmentNode):
        """Process a CrossGL assignment statement."""
        target = getattr(
            node, "name", getattr(node, "target", getattr(node, "left", None))
        )
        target_is_precise = self.assignment_target_is_precise(target)
        operator = getattr(node, "operator", "=")

        if (
            isinstance(target, CooperativeMatrixOpNode)
            and target.operation == "element"
        ):
            if not self.cooperative_matrix_khr_enabled:
                self.process_expression(target)
            if operator != "=":
                raise self.cooperative_matrix_operation_error(
                    target,
                    "cooperative-matrix-element-compound-assignment",
                    "SPIR-V cooperative-matrix element compound assignment is "
                    f"not implemented for operator '{operator}'",
                    "spirv.cooperative_matrix.element-compound-assignment",
                )
            self.process_cooperative_matrix_element_assignment(
                target, node.value, target_is_precise
            )
            return

        if operator != "=":
            self.process_compound_assignment(node, target, operator, target_is_precise)
            return

        if self.process_mesh_output_assignment(target, node.value):
            return

        if isinstance(node.value, MatchNode):
            target_pointer = self.assignable_pointer_from_expression(target)
            target_type = (
                self.variable_value_types.get(target_pointer.id)
                if target_pointer is not None
                else None
            )
            if target_pointer is None or target_type is None:
                self.refuse_lowering("Could not determine match assignment target type")
                return
            self.process_match_expression_assignment(
                node.value, target_pointer, target_type
            )
            return

        if isinstance(node.value, ArrayLiteralNode):
            target_pointer = self.assignable_pointer_from_expression(target)
            target_type = (
                self.variable_value_types.get(target_pointer.id)
                if target_pointer is not None
                else None
            )
            rhs_value = self.process_array_literal(node.value, target_type)
        else:
            target_type = None
            if isinstance(node.value, CooperativeMatrixOpNode) and isinstance(
                target, (IdentifierNode, VariableNode, str)
            ):
                target_name = target if isinstance(target, str) else target.name
                target_pointer = self.ensure_assignable_pointer_for_name(target_name)
                if target_pointer is not None:
                    target_type = self.variable_value_types.get(target_pointer.id)
            if target_type is not None:
                rhs_value = self.process_expression_with_expected_type_and_precision(
                    node.value, target_type, target_is_precise
                )
            else:
                rhs_value = self.process_expression_with_precision(
                    node.value, target_is_precise
                )

        if rhs_value is None:
            return

        if isinstance(target, IdentifierNode):
            target = target.name

        if isinstance(target, str):
            if target in self.resource_alias_variables:
                self.refuse_lowering(
                    f"cannot assign to SPIR-V descriptor resource alias " f"{target}"
                )
                return

            var_id = self.ensure_assignable_pointer_for_name(target)
            if var_id is None and target in self.cbuffer_members:
                # GLSL 4.60 section 4.3.5 and HLSL constant buffers: uniforms are read-only.
                self.refuse_lowering(f"cannot assign to read-only uniform {target}")
                return
            if var_id is None:
                self.refuse_lowering(f"cannot assign to undeclared variable {target}")
                return

            self.store_to_variable(var_id, rhs_value)

        elif isinstance(target, MemberAccessNode):
            base_pointer = self.assignable_pointer_from_expression(target.object)
            if base_pointer is None:
                # A member of a value with no storage (a call result, a temporary) is not an lvalue.
                self.refuse_lowering(
                    f"Cannot assign to member {target.member} of "
                    f"{self.diagnostic_expression(target.object)}: it does not name storage"
                )
                return

            member_name = target.member
            access = self.create_member_access_pointer(base_pointer, member_name)
            if access is not None:
                self.store_to_variable(access, rhs_value)
                return
            if self.store_to_vector_swizzle(base_pointer, member_name, rhs_value):
                return
            if self.store_to_vector_component(base_pointer, member_name, rhs_value):
                return
            if self.emit_invalid_vector_swizzle_warning(base_pointer, member_name):
                return

            struct_type = self.struct_type_name_from_pointer(base_pointer)
            self.refuse_lowering(
                f"Could not find member {member_name} in {struct_type}"
            )

        elif (
            isinstance(target, UnaryOpNode)
            and not getattr(target, "is_postfix", False)
            and target.op == "*"
        ):
            access = self.assignable_pointer_from_expression(target)
            element_type = (
                self.variable_value_types.get(access.id) if access is not None else None
            )
            if access is None or element_type is None:
                raise self.unsupported_pointer_dereference_error(target)
            self.store_to_variable(access, rhs_value)
        elif isinstance(target, ArrayAccessNode):
            access = self.assignable_pointer_from_expression(target)
            element_type = (
                self.variable_value_types.get(access.id) if access is not None else None
            )

            if access is None or element_type is None:
                self.refuse_lowering(
                    "Could not determine array element type for "
                    f"{self.diagnostic_expression(target.array)}"
                )
                return

            self.store_to_variable(access, rhs_value)
        else:
            self.refuse_lowering(
                f"Unsupported LHS type in assignment: {type(target).__name__}"
            )

    def process_compound_assignment(
        self, node: AssignmentNode, target, operator: str, target_is_precise: bool
    ):
        spv_operator = self.compound_assignment_operator(operator)
        if spv_operator is None:
            self.refuse_lowering(f"Unsupported compound assignment operator {operator}")
            return
        if isinstance(node.value, ArrayLiteralNode):
            self.refuse_lowering("Compound array literal assignment is unsupported")
            return

        if self.process_mesh_output_compound_assignment(
            target, node.value, spv_operator, target_is_precise
        ):
            return

        target_pointer = None
        if not isinstance(target, MemberAccessNode):
            target_pointer = self.assignable_pointer_from_expression(target)
            if (
                spv_operator in {"+", "-"}
                and target_pointer is not None
                and self.is_storage_buffer_pointer_arithmetic_base(target_pointer)
            ):
                offset = self.process_expression(node.value)
                if offset is not None:
                    self.record_storage_buffer_pointer_offset_alias(
                        target_pointer,
                        target_pointer,
                        offset,
                        spv_operator,
                    )
                return

        if isinstance(target, MemberAccessNode):
            base_pointer = self.assignable_pointer_from_expression(target.object)
            if (
                base_pointer is not None
                and self.process_vector_swizzle_compound_assignment(
                    base_pointer,
                    target.member,
                    node.value,
                    spv_operator,
                    target_is_precise,
                )
            ):
                return
            if (
                base_pointer is not None
                and self.process_vector_component_compound_assignment(
                    base_pointer,
                    target.member,
                    node.value,
                    spv_operator,
                    target_is_precise,
                )
            ):
                return
            if base_pointer is not None and self.emit_invalid_vector_swizzle_warning(
                base_pointer, target.member
            ):
                return

        if target_pointer is None:
            target_pointer = self.assignable_pointer_from_expression(target)

        if target_pointer is None:
            self.refuse_lowering(
                f"Unsupported LHS type in assignment: {type(target).__name__}"
            )
            return

        target_type = self.variable_value_types.get(target_pointer.id)
        if target_type is None:
            target_type = self.find_registered_type_by_base(
                target_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if target_type is None:
            self.refuse_lowering("Could not determine compound assignment type")
            return

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            current_value = self.load_from_variable(target_pointer, target_type)
            rhs_value = self.process_expression(node.value)
            if rhs_value is None:
                return
            result = self.binary_operation(
                spv_operator, target_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        self.store_to_variable(target_pointer, result)

    def vector_component_update_info(self, base_pointer: SpirvId, member_name: str):
        vector_type = self.variable_value_types.get(base_pointer.id)
        if vector_type is None:
            vector_type = self.find_registered_type_by_base(
                base_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if vector_type is None:
            return None

        member_info = self.vector_member_info(vector_type.type.base_type, member_name)
        if member_info is None:
            return None

        member_index, member_type = member_info
        return vector_type, member_index, member_type

    def store_to_vector_component(
        self, base_pointer: SpirvId, member_name: str, value: SpirvId
    ) -> bool:
        update_info = self.vector_component_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_index, member_type = update_info
        component_value = self.convert_vector_component_value(
            value, member_type, member_name
        )
        if component_value is None:
            return True

        self.store_vector_components(
            base_pointer,
            vector_type,
            None,
            [(member_index, component_value)],
            member_type,
        )
        return True

    # Storage other invocations can access while this one runs: buffers and workgroup memory (Vulkan 1.3
    # appendix B, "Memory Model"). A write to part of a vector there must not store the other components.
    SHARED_VECTOR_STORAGE_CLASSES = frozenset(
        {
            "StorageBuffer",
            "Uniform",
            "Workgroup",
            "PhysicalStorageBuffer",
            "CrossWorkgroup",
            "TaskPayloadWorkgroupEXT",
        }
    )

    def shared_vector_component_pointer(
        self, base_pointer: SpirvId, member_index: int, member_type: SpirvId
    ) -> Optional[SpirvId]:
        """A pointer to one component of a vector in shared storage, or None for invocation-private storage.

        A store through the vector pointer writes every component, so another invocation's write to a
        different component of the same vector could be lost: the source wrote only some components.
        glslang and DXC store each component through its own ``OpAccessChain`` (SPIR-V 1.6, ``OpAccessChain``
        indexes into a vector).
        """
        if base_pointer.type.storage_class not in self.SHARED_VECTOR_STORAGE_CLASSES:
            return None
        return self.vector_component_pointer(base_pointer, member_index, member_type)

    def vector_component_pointer(
        self, base_pointer: SpirvId, member_index: int, member_type: SpirvId
    ) -> SpirvId:
        storage_class = base_pointer.type.storage_class
        index = self.register_constant(member_index, self.primitive_types["int"])
        ptr_type = self.register_pointer_type(member_type, storage_class)
        access = self.access_chain(base_pointer, [index], ptr_type)
        self.variable_value_types[access.id] = member_type
        self.propagate_storage_buffer_access_metadata(base_pointer, access)
        self.propagate_readonly_builtin_pointer_name(base_pointer, access)
        self.propagate_readonly_pointer_name(base_pointer, access)
        return access

    def vector_component_reference_pointer(
        self, base_pointer: SpirvId, member_name: str
    ) -> Optional[SpirvId]:
        """``v.y`` as storage, for an atomic's memory, an increment or an ``inout`` argument.

        A single component of a vector in memory is addressable (SPIR-V 1.6, ``OpAccessChain``; GLSL 4.60
        section 5.8 and HLSL allow a one-component swizzle as an l-value), so it gets its own pointer.
        """
        if not base_pointer.type.storage_class:
            return None
        update_info = self.vector_component_update_info(base_pointer, member_name)
        if update_info is None:
            return None
        _, member_index, member_type = update_info
        return self.vector_component_pointer(base_pointer, member_index, member_type)

    def storage_reference_pointer(self, expr) -> Optional[SpirvId]:
        """A pointer to the memory ``expr`` names, where that may be one component of a vector."""
        if not isinstance(expr, MemberAccessNode):
            return self.variable_pointer_from_expression(expr)
        base_pointer = self.variable_pointer_from_expression(expr.object)
        if base_pointer is None:
            return None
        access = self.create_member_access_pointer(base_pointer, expr.member)
        if access is None:
            access = self.vector_component_reference_pointer(base_pointer, expr.member)
        return access

    def load_vector_components(
        self,
        base_pointer: SpirvId,
        member_indices: List[int],
        member_type: SpirvId,
    ) -> Optional[List[SpirvId]]:
        """The selected components of a vector in shared storage, each read through its own pointer.

        None for invocation-private storage, where the caller loads the whole vector.
        """
        if base_pointer.type.storage_class not in self.SHARED_VECTOR_STORAGE_CLASSES:
            return None
        return [
            self.load_from_variable(
                self.shared_vector_component_pointer(
                    base_pointer, member_index, member_type
                ),
                member_type,
            )
            for member_index in member_indices
        ]

    def store_vector_components(
        self,
        base_pointer: SpirvId,
        vector_type: SpirvId,
        vector_value: Optional[SpirvId],
        updates: List[Tuple[int, SpirvId]],
        member_type: SpirvId,
    ) -> None:
        """Store ``(component index, value)`` updates into the vector ``base_pointer`` points to.

        Shared storage gets one store per component. Private storage gets the vector with the components
        inserted, starting from ``vector_value`` or, when that is None, a fresh load.
        """
        if base_pointer.type.storage_class in self.SHARED_VECTOR_STORAGE_CLASSES:
            for member_index, component_value in updates:
                component_pointer = self.shared_vector_component_pointer(
                    base_pointer, member_index, member_type
                )
                self.store_to_variable(component_pointer, component_value)
            return
        if vector_value is None:
            vector_value = self.load_from_variable(base_pointer, vector_type)
        for member_index, component_value in updates:
            vector_value = self.composite_insert(
                vector_value, component_value, vector_type, member_index
            )
        self.store_to_variable(base_pointer, vector_value)

    def vector_swizzle_update_info(self, base_pointer: SpirvId, member_name: str):
        vector_type = self.variable_value_types.get(base_pointer.id)
        if vector_type is None:
            vector_type = self.find_registered_type_by_base(
                base_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if vector_type is None:
            return None

        swizzle_info = self.vector_swizzle_info(vector_type.type.base_type, member_name)
        if swizzle_info is None:
            return None

        indices, member_type, swizzle_type = swizzle_info
        return vector_type, indices, member_type, swizzle_type

    def emit_invalid_vector_swizzle_warning(
        self, base_pointer: SpirvId, member_name: str
    ) -> bool:
        if len(member_name) <= 1:
            return False

        vector_type = self.variable_value_types.get(base_pointer.id)
        if vector_type is None:
            vector_type = self.find_registered_type_by_base(
                base_pointer.type.base_type.replace("ptr_", "", 1)
            )
        if vector_type is None:
            return False

        vector_type_name = vector_type.type.base_type
        if self.vector_component_type_and_count(vector_type_name) is None:
            return False
        if self.vector_swizzle_info(vector_type_name, member_name) is not None:
            return False

        self.refuse_lowering(
            f"Invalid vector swizzle {member_name} for {vector_type_name}"
        )
        return True

    def store_to_vector_swizzle(
        self, base_pointer: SpirvId, member_name: str, value: SpirvId
    ) -> bool:
        update_info = self.vector_swizzle_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_indices, member_type, _ = update_info
        if len(set(member_indices)) != len(member_indices):
            self.refuse_lowering(
                f"Cannot assign to vector swizzle {member_name} "
                "with duplicate components"
            )
            return True

        component_values = self.convert_vector_swizzle_assignment_components(
            value, member_type, len(member_indices), member_name
        )
        if component_values is None:
            return True

        self.store_vector_components(
            base_pointer,
            vector_type,
            None,
            list(zip(member_indices, component_values)),
            member_type,
        )
        return True

    def convert_vector_swizzle_assignment_components(
        self,
        value: SpirvId,
        member_type: SpirvId,
        component_count: int,
        member_name: str,
    ) -> Optional[List[SpirvId]]:
        source_vector_info = self.vector_component_type_and_count(value.type.base_type)
        if source_vector_info is None:
            self.refuse_lowering(
                f"Cannot assign scalar value to vector swizzle {member_name}"
            )
            return None

        source_component_type_name, source_component_count = source_vector_info
        if source_component_count != component_count:
            self.refuse_lowering(
                f"Cannot assign {source_component_count}-component vector "
                f"to {component_count}-component swizzle {member_name}"
            )
            return None

        source_component_type = self.register_primitive_type(source_component_type_name)
        target_type_name = self.normalize_primitive_name(member_type.type.base_type)
        component_values = []
        for source_index in range(component_count):
            source_value = self.composite_extract(
                value, source_component_type, source_index
            )
            component_value = self.convert_scalar_to_type(source_value, member_type)
            converted_type_name = self.normalize_primitive_name(
                component_value.type.base_type
            )
            if converted_type_name != target_type_name:
                self.refuse_lowering(
                    f"Cannot assign {converted_type_name} value to "
                    f"vector swizzle {member_name} of type {target_type_name}"
                )
                return None
            component_values.append(component_value)

        return component_values

    def process_vector_swizzle_compound_assignment(
        self,
        base_pointer: SpirvId,
        member_name: str,
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        update_info = self.vector_swizzle_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_indices, member_type, swizzle_type = update_info
        if len(set(member_indices)) != len(member_indices):
            self.refuse_lowering(
                f"Cannot assign to vector swizzle {member_name} "
                "with duplicate components"
            )
            return True

        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            if self.is_scalar_numeric_value(rhs_value):
                # `v.xy op= s` applies the scalar to every selected component
                # (GLSL 4.60 section 5.9; HLSL scalar promotion).
                scalar = self.convert_scalar_to_type(rhs_value, member_type)
                rhs_components = [scalar] * len(member_indices)
            else:
                rhs_components = self.convert_vector_swizzle_assignment_components(
                    rhs_value, member_type, len(member_indices), member_name
                )
            if rhs_components is None:
                return True

            rhs_vector = self.composite_construct(swizzle_type, rhs_components)
            shared_components = self.load_vector_components(
                base_pointer, member_indices, member_type
            )
            if shared_components is not None:
                vector_value = None
                current_value = self.composite_construct(
                    swizzle_type, shared_components
                )
            else:
                vector_value = self.load_from_variable(base_pointer, vector_type)
                current_value = self.vector_shuffle(
                    vector_value, swizzle_type, member_indices
                )
            result = self.binary_operation(
                spv_operator, swizzle_type, current_value, rhs_vector
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        if vector_value is None:
            self.store_vector_components(
                base_pointer,
                vector_type,
                None,
                [
                    (
                        member_index,
                        self.composite_extract(result, member_type, position),
                    )
                    for position, member_index in enumerate(member_indices)
                ],
                member_type,
            )
            return True
        updated_value = vector_value
        for result_index, member_index in enumerate(member_indices):
            component_value = self.composite_extract(result, member_type, result_index)
            updated_value = self.composite_insert(
                updated_value, component_value, vector_type, member_index
            )
        self.store_to_variable(base_pointer, updated_value)
        return True

    def convert_vector_component_value(
        self, value: SpirvId, member_type: SpirvId, member_name: str
    ) -> Optional[SpirvId]:
        if self.vector_component_type_and_count(value.type.base_type) is not None:
            self.refuse_lowering(
                f"Cannot assign composite value to vector component {member_name}"
            )
            return None

        component_value = self.convert_scalar_to_type(value, member_type)
        source_type = self.normalize_primitive_name(component_value.type.base_type)
        target_type = self.normalize_primitive_name(member_type.type.base_type)
        if source_type != target_type:
            self.refuse_lowering(
                f"Cannot assign {source_type} value to vector component "
                f"{member_name} of type {target_type}"
            )
            return None

        return component_value

    def process_vector_component_compound_assignment(
        self,
        base_pointer: SpirvId,
        member_name: str,
        value_expr,
        spv_operator: str,
        target_is_precise: bool,
    ) -> bool:
        update_info = self.vector_component_update_info(base_pointer, member_name)
        if update_info is None:
            return False

        vector_type, member_index, member_type = update_info
        if target_is_precise:
            self.precise_expression_depth += 1

        try:
            shared_components = self.load_vector_components(
                base_pointer, [member_index], member_type
            )
            if shared_components is not None:
                vector_value = None
                current_value = shared_components[0]
            else:
                vector_value = self.load_from_variable(base_pointer, vector_type)
                current_value = self.composite_extract(
                    vector_value, member_type, member_index
                )
            rhs_value = self.process_expression(value_expr)
            if rhs_value is None:
                return True
            rhs_value = self.convert_vector_component_value(
                rhs_value, member_type, member_name
            )
            if rhs_value is None:
                return True

            result = self.binary_operation(
                spv_operator, member_type, current_value, rhs_value
            )
        finally:
            if target_is_precise:
                self.precise_expression_depth -= 1

        if vector_value is None:
            self.store_vector_components(
                base_pointer, vector_type, None, [(member_index, result)], member_type
            )
            return True
        updated_value = self.composite_insert(
            vector_value, result, vector_type, member_index
        )
        self.store_to_variable(base_pointer, updated_value)
        return True

    def compound_assignment_operator(self, operator: str) -> Optional[str]:
        return {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }.get(operator)

    def assignment_target_is_precise(self, target) -> bool:
        if isinstance(target, IdentifierNode):
            return self.variable_name_is_precise(target.name)
        if isinstance(target, str):
            return self.variable_name_is_precise(target)
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_is_precise(target.object)
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_is_precise(target.array)
        return False

    def variable_name_is_precise(self, name: str) -> bool:
        if name in self.precise_flow_names:
            return True
        if name in self.local_variables:
            return name in self.precise_local_variables
        return name in self.precise_global_variables
