"""Cooperative matrix operations, enums and structs in the SPIR-V generator: cooperative matrix operations, loads and memory operands, loop controls, enum structs and metadata, generic enum specializations, enum variants, struct constructors, generic struct materialization and static struct members."""

import re
from typing import Optional

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    BinaryOpNode,
    ConstructorNode,
    CooperativeMatrixOpNode,
    EnumNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    NamedType,
    StructNode,
    UnaryOpNode,
)
from ..matrix_shape import matrix_type_spelling
from ..refusals import VulkanRefusal
from .array_utils import (
    evaluate_literal_int_expression,
    prepend_array_dimension,
)
from .enum_utils import (
    enum_struct_fields,
    enum_variant_payload_fields,
    generic_enum_specialized_fields,
    generic_enum_specialized_variant_fields,
    generic_type_parts,
    resolve_generic_enum_specialization,
)
from .spirv_generic_structs import materialize_generic_struct_instance
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)


class SpirvEnumStructMixin:
    """Enum struct members of ``VulkanSPIRVCodeGen``."""

    def process_cooperative_matrix_operation(self, expr):
        if expr.operation == "element":
            return self.process_cooperative_matrix_element(expr)
        if expr.operation == "load":
            return self.process_cooperative_matrix_load(expr)
        if expr.operation == "store":
            return self.process_cooperative_matrix_store(expr)
        if expr.operation != "multiply":
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-operation-lowering",
                "SPIR-V KHR cooperative-matrix type support is enabled, but "
                f"operation '{expr.operation}' is not implemented",
                "spirv.cooperative_matrix.operation-lowering",
            )
        return self.process_cooperative_matrix_multiply(expr)

    def cooperative_matrix_memory_control_flow(self, expr, operation: str):
        if self.current_function_node is not None and not self.function_is_entry_point(
            self.current_function_node, self.current_stage
        ):
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-control-flow",
                "SPIR-V cooperative-matrix memory lowering currently requires "
                f"{operation} directly in an entry point so subgroup participation "
                "can be proven",
                f"spirv.cooperative_matrix.{operation}-uniform-control-flow",
            )
        if (
            not self.loop_merge_labels
            and self.cooperative_matrix_selection_depth == 0
            and self.cooperative_matrix_loop_control_depth == 0
            and not self.cooperative_matrix_control_flow_diverged
        ):
            return
        raise self.cooperative_matrix_operation_error(
            expr,
            f"cooperative-matrix-{operation}-control-flow",
            "SPIR-V cooperative-matrix memory lowering currently requires "
            f"statically unconditional subgroup participation for {operation}",
            f"spirv.cooperative_matrix.{operation}-uniform-control-flow",
        )

    def process_loop_control_expression(self, expr):
        self.cooperative_matrix_loop_control_depth += 1
        try:
            return self.process_expression(expr)
        finally:
            self.cooperative_matrix_loop_control_depth -= 1

    def process_loop_control_statement(self, statement):
        self.cooperative_matrix_loop_control_depth += 1
        try:
            return self.process_statement(statement)
        finally:
            self.cooperative_matrix_loop_control_depth -= 1

    def cooperative_matrix_memory_layout(self, expr, contract, operation: str):
        layout_name = str(contract.get("layout") or "unspecified")
        layout_name = layout_name.strip().lower().replace("-", "_")
        if layout_name not in {"row_major", "rowmajor", "unspecified"}:
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-layout",
                "SPIR-V cooperative-matrix memory lowering currently supports "
                f"only default row-major layout; got '{layout_name}'",
                f"spirv.cooperative_matrix.{operation}-row-major-layout",
            )
        uint_type = self.register_primitive_type("uint")
        return self.register_constant(0, uint_type)

    def cooperative_matrix_memory_alignment(self, contract) -> int:
        component_size = self.uniform_scalar_size(contract["component_type"])
        return min(16, contract["cols"] * component_size)

    def cooperative_matrix_memory_operand_alignment(
        self, required_alignment: int
    ) -> int:
        return required_alignment & -required_alignment

    def cooperative_matrix_memory_pointee_size(self, pointee) -> Optional[int]:
        vector_info = self.vector_type_info_from_type(pointee)
        if vector_info is not None:
            component_type, component_count = vector_info
        else:
            component_type, component_count = pointee, 1
        component_name = self.normalize_primitive_name(component_type.type.base_type)
        if component_name not in {"float", "int", "uint"}:
            return None
        return self.uniform_scalar_size(component_type) * component_count

    def cooperative_matrix_memory_base_has_static_offset(self, base_source) -> bool:
        base_pointer = self.variable_pointer_from_expression(base_source)
        if base_pointer is None:
            return False
        offset_state = self.storage_buffer_pointer_offset_aliases.get(
            self.storage_buffer_pointer_alias_key(base_pointer)
        )
        return offset_state is None

    def cooperative_matrix_memory_static_offset(self, pointer_source) -> Optional[int]:
        if self.cooperative_matrix_direct_identifier_name(pointer_source) is not None:
            if self.cooperative_matrix_memory_base_has_static_offset(pointer_source):
                return 0
            return None
        if isinstance(pointer_source, ArrayAccessNode):
            if self.cooperative_matrix_direct_identifier_name(
                pointer_source.array
            ) is None or not self.cooperative_matrix_memory_base_has_static_offset(
                pointer_source.array
            ):
                return None
            offset = evaluate_literal_int_expression(
                pointer_source.index, self.active_literal_int_constants()
            )
            return offset if offset is not None and offset >= 0 else None
        if isinstance(pointer_source, BinaryOpNode) and pointer_source.op == "+":
            for base, offset_source in (
                (pointer_source.left, pointer_source.right),
                (pointer_source.right, pointer_source.left),
            ):
                if self.cooperative_matrix_direct_identifier_name(
                    base
                ) is None or not self.cooperative_matrix_memory_base_has_static_offset(
                    base
                ):
                    continue
                offset = evaluate_literal_int_expression(
                    offset_source, self.active_literal_int_constants()
                )
                if offset is not None and offset >= 0:
                    return offset
        return None

    def cooperative_matrix_memory_stride(
        self,
        expr,
        stride_source,
        operation: str,
        pointee_size: int,
        required_alignment: int,
    ):
        stride = evaluate_literal_int_expression(
            stride_source, self.active_literal_int_constants()
        )
        minimum_stride = 0 if operation == "load" else 1
        if (
            stride is None
            or isinstance(stride, bool)
            or stride < minimum_stride
            or stride > self.UNSIGNED_INT32_MAX
        ):
            requirement = "non-negative" if operation == "load" else "positive"
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-stride",
                "SPIR-V cooperative-matrix memory lowering currently requires "
                f"a {requirement}, compile-time 32-bit integer stride for "
                f"{operation}",
                f"spirv.cooperative_matrix.{operation}-stride",
            )
        if (stride * pointee_size) % required_alignment != 0:
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-alignment",
                "SPIR-V cooperative-matrix memory lowering requires the byte "
                f"stride for {operation} to be aligned to {required_alignment} "
                f"bytes; got {stride * pointee_size} bytes",
                f"spirv.cooperative_matrix.{operation}-alignment",
            )
        uint_type = self.register_primitive_type("uint")
        return self.register_constant(stride, uint_type)

    def cooperative_matrix_memory_pointer(
        self, expr, pointer_source, contract, operation: str
    ):
        static_offset = self.cooperative_matrix_memory_static_offset(pointer_source)
        pointer = self.variable_pointer_from_expression(pointer_source)
        if pointer is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-pointer",
                "SPIR-V cooperative-matrix memory lowering requires an "
                f"addressable scalar or vector pointer for {operation}",
                f"spirv.cooperative_matrix.{operation}-pointer",
            )

        root_pointee = self.pointer_pointee_type(pointer)
        pointee = root_pointee
        if (
            pointee is None
            or self.cooperative_matrix_memory_pointee_size(pointee) is None
        ):
            int_type = self.register_primitive_type("int")
            zero = self.register_constant(0, int_type)
            pointer, pointee = self.create_array_element_access(pointer_source, zero, 0)
        pointee_size = (
            self.cooperative_matrix_memory_pointee_size(pointee)
            if pointee is not None
            else None
        )
        if pointer is None or pointee is None or pointee_size is None:
            pointee_name = None if pointee is None else pointee.type.base_type
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-pointee-type",
                "SPIR-V cooperative-matrix memory lowering currently requires a "
                "32-bit scalar or vector pointer element; got "
                f"'{pointee_name}'",
                f"spirv.cooperative_matrix.{operation}-pointee-type",
            )

        storage_class = pointer.type.storage_class
        access_metadata = self.storage_buffer_access_metadata_for_pointer(pointer)
        if storage_class == "StorageBuffer":
            legal_storage = (
                access_metadata is not None
                and access_metadata.get("_access_path") == "element"
                and not access_metadata.get("descriptor_array")
            )
        elif storage_class == "Workgroup":
            workgroup_array_type = root_pointee
            if isinstance(pointer_source, ArrayAccessNode):
                workgroup_array = self.variable_pointer_from_expression(
                    pointer_source.array
                )
                workgroup_array_type = (
                    self.pointer_pointee_type(workgroup_array)
                    if workgroup_array is not None
                    else None
                )
            legal_storage = (
                workgroup_array_type is not None
                and self.array_type_info_from_type(workgroup_array_type) is not None
            )
        else:
            legal_storage = False
        if not legal_storage:
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-storage-class",
                "SPIR-V cooperative-matrix memory lowering currently supports "
                "only storage-buffer and workgroup array elements; got storage "
                f"class '{storage_class}'",
                f"spirv.cooperative_matrix.{operation}-storage-class",
            )

        if access_metadata is not None:
            if operation == "load" and access_metadata.get("writeonly"):
                access_error = "a readable source buffer"
            elif operation == "store" and access_metadata.get("readonly"):
                access_error = "a writable destination buffer"
            else:
                access_error = None
            if access_error is not None:
                raise self.cooperative_matrix_operation_error(
                    expr,
                    f"cooperative-matrix-{operation}-buffer-access",
                    "SPIR-V cooperative-matrix memory lowering requires "
                    f"{access_error}",
                    f"spirv.cooperative_matrix.{operation}-buffer-access",
                )
            if access_metadata.get("volatile") or access_metadata.get("coherent"):
                raise self.cooperative_matrix_operation_error(
                    expr,
                    f"cooperative-matrix-{operation}-memory-operands",
                    "SPIR-V cooperative-matrix memory lowering does not yet "
                    "preserve volatile or coherent memory operands",
                    f"spirv.cooperative_matrix.{operation}-memory-operands",
                )

        if static_offset is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-uniformity",
                "SPIR-V cooperative-matrix memory pointer operands must have a "
                "statically provable subgroup-uniform offset",
                f"spirv.cooperative_matrix.{operation}-uniformity",
            )
        if self.is_non_uniform_value(pointer):
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-uniformity",
                "SPIR-V cooperative-matrix memory pointer operands must be "
                "dynamically uniform within the subgroup",
                f"spirv.cooperative_matrix.{operation}-uniformity",
            )
        required_alignment = self.cooperative_matrix_memory_alignment(contract)
        pointer_byte_offset = static_offset * pointee_size
        if pointer_byte_offset % required_alignment != 0:
            raise self.cooperative_matrix_operation_error(
                expr,
                f"cooperative-matrix-{operation}-alignment",
                "SPIR-V cooperative-matrix memory lowering requires the pointer "
                f"offset for {operation} to be aligned to {required_alignment} "
                f"bytes; got {pointer_byte_offset} bytes",
                f"spirv.cooperative_matrix.{operation}-alignment",
            )
        memory_operand_alignment = self.cooperative_matrix_memory_operand_alignment(
            required_alignment
        )
        return pointer, pointee_size, required_alignment, memory_operand_alignment

    def process_cooperative_matrix_load(self, expr):
        self.cooperative_matrix_memory_control_flow(expr, "load")
        if len(expr.arguments) != 3:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-load-memory-contract",
                "SPIR-V cooperative-matrix load currently supports the default "
                "row-major form with destination, pointer, and stride operands; "
                f"got {len(expr.arguments)} operands",
                "spirv.cooperative_matrix.load-memory-contract",
            )

        destination_source, pointer_source, stride_source = expr.arguments
        if self.cooperative_matrix_direct_identifier_name(destination_source) is None:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-load-destination",
                "SPIR-V cooperative-matrix load requires a direct local matrix "
                "destination",
                "spirv.cooperative_matrix.load-destination",
            )
        destination = self.variable_pointer_from_expression(destination_source)
        destination_type = (
            self.pointer_pointee_type(destination) if destination is not None else None
        )
        contract = self.cooperative_matrix_type_info(destination_type)
        if (
            destination is None
            or destination.type.storage_class != "Function"
            or contract is None
        ):
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-load-destination",
                "SPIR-V cooperative-matrix load requires a function-local "
                "cooperative-matrix destination",
                "spirv.cooperative_matrix.load-destination",
            )

        layout = self.cooperative_matrix_memory_layout(expr, contract, "load")
        pointer, pointee_size, required_alignment, memory_operand_alignment = (
            self.cooperative_matrix_memory_pointer(
                expr, pointer_source, contract, "load"
            )
        )
        stride = self.cooperative_matrix_memory_stride(
            expr,
            stride_source,
            "load",
            pointee_size,
            required_alignment,
        )
        result_id = self.get_id()
        self.emit(
            f"%{result_id} = OpCooperativeMatrixLoadKHR %{destination_type.id} "
            f"%{pointer.id} %{layout.id} %{stride.id} Aligned "
            f"{memory_operand_alignment}"
        )
        result = SpirvId(result_id, destination_type.type)
        self.value_types[result_id] = destination_type
        self.store_to_variable(destination, result)
        return result

    def process_cooperative_matrix_store(self, expr):
        self.cooperative_matrix_memory_control_flow(expr, "store")
        if len(expr.arguments) != 3:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-store-memory-contract",
                "SPIR-V cooperative-matrix store currently supports the default "
                "row-major form with pointer, matrix, and stride operands; got "
                f"{len(expr.arguments)} operands",
                "spirv.cooperative_matrix.store-memory-contract",
            )

        pointer_source, matrix_source, stride_source = expr.arguments
        matrix, contract = self.cooperative_matrix_operation_value(
            expr, matrix_source, "stored"
        )
        layout = self.cooperative_matrix_memory_layout(expr, contract, "store")
        pointer, pointee_size, required_alignment, memory_operand_alignment = (
            self.cooperative_matrix_memory_pointer(
                expr, pointer_source, contract, "store"
            )
        )
        stride = self.cooperative_matrix_memory_stride(
            expr,
            stride_source,
            "store",
            pointee_size,
            required_alignment,
        )
        self.emit(
            f"OpCooperativeMatrixStoreKHR %{pointer.id} %{matrix.id} "
            f"%{layout.id} %{stride.id} Aligned {memory_operand_alignment}"
        )
        return None

    def process_cooperative_matrix_multiply(self, expr):
        if len(expr.arguments) != 2:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-multiply-arity",
                "SPIR-V cooperative-matrix multiply requires exactly two operands; "
                f"got {len(expr.arguments)}",
                "spirv.cooperative_matrix.multiply-arity",
            )

        for label, operand, required_use in (
            ("left", expr.arguments[0], "matrix_a"),
            ("right", expr.arguments[1], "matrix_b"),
        ):
            if (
                isinstance(operand, CooperativeMatrixOpNode)
                and operand.operation == "multiply"
            ):
                raise self.cooperative_matrix_operation_error(
                    expr,
                    "cooperative-matrix-role-conversion",
                    "SPIR-V cooperative-matrix multiply produces an accumulator "
                    f"result, which cannot be consumed as the outer {label} "
                    f"{required_use} operand without a semantics-preserving role "
                    "conversion",
                    "spirv.cooperative_matrix.role-conversion",
                )

        left, left_contract = self.cooperative_matrix_operation_value(
            expr, expr.arguments[0], "left"
        )
        right, right_contract = self.cooperative_matrix_operation_value(
            expr, expr.arguments[1], "right"
        )
        result_type, result_contract = self.cooperative_matrix_operation_result_type(
            expr
        )

        required_uses = (
            ("left", left_contract, "matrix_a"),
            ("right", right_contract, "matrix_b"),
            ("result", result_contract, "accumulator"),
        )
        for label, contract, required_use in required_uses:
            if contract["use"] != required_use:
                raise self.cooperative_matrix_operation_error(
                    expr,
                    "cooperative-matrix-multiply-use-role",
                    "SPIR-V cooperative-matrix multiply requires the "
                    f"{label} value to use role {required_use}; got "
                    f"{contract['use']}",
                    "spirv.cooperative_matrix.multiply-use-role",
                )

        contracts = (left_contract, right_contract, result_contract)
        if len({contract["scope_value"] for contract in contracts}) != 1:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-multiply-scope",
                "SPIR-V cooperative-matrix multiply requires all operands and the "
                "result to use the same execution scope",
                "spirv.cooperative_matrix.multiply-scope",
            )

        expected_shape = (left_contract["rows"], right_contract["cols"])
        if (
            left_contract["cols"] != right_contract["rows"]
            or (result_contract["rows"], result_contract["cols"]) != expected_shape
        ):
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-multiply-shape",
                "SPIR-V cooperative-matrix multiply requires compatible MxK and "
                "KxN operands and an MxN result; got "
                f"{left_contract['rows']}x{left_contract['cols']}, "
                f"{right_contract['rows']}x{right_contract['cols']}, and "
                f"{result_contract['rows']}x{result_contract['cols']}",
                "spirv.cooperative_matrix.multiply-shape",
            )

        component_names = {contract["component_name"] for contract in contracts}
        if component_names != {"float"}:
            raise self.cooperative_matrix_operation_error(
                expr,
                "cooperative-matrix-multiply-component-type",
                "SPIR-V cooperative-matrix multiply currently requires matching "
                "32-bit floating-point A, B, accumulator, and result components",
                "spirv.cooperative_matrix.multiply-component-type",
            )

        zero_scalar = self.register_constant(0.0, result_contract["component_type"])
        zero_accumulator = self.register_composite_constant(result_type, [zero_scalar])
        result_id = self.get_id()
        self.emit(
            f"%{result_id} = OpCooperativeMatrixMulAddKHR %{result_type.id} "
            f"%{left.id} %{right.id} %{zero_accumulator.id}"
        )
        result = SpirvId(result_id, result_type.type)
        self.value_types[result_id] = result_type
        return result

    def is_cooperative_matrix_type_name(self, type_name: str) -> bool:
        """Return whether a type uses the canonical cooperative-matrix spelling."""
        compact = re.sub(r"\s+", "", str(type_name))
        generic_base, generic_args = generic_type_parts(compact)
        return bool(
            generic_args and generic_base.rsplit("::", 1)[-1] == "CooperativeMatrix"
        )

    def is_metal_simdgroup_matrix_type(self, type_name: str) -> bool:
        """Return whether a type names a Metal subgroup-distributed matrix."""
        compact = re.sub(r"\s+", "", str(type_name))
        generic_base, generic_args = generic_type_parts(compact)
        if generic_args and generic_base.rsplit("::", 1)[-1] == "simdgroup_matrix":
            return True
        return bool(
            re.fullmatch(
                r"(?:metal::)?simdgroup_(?:float|half|bfloat)\d+x\d+",
                compact,
            )
        )

    def ensure_declared_struct_type(self, type_name: str) -> Optional[SpirvId]:
        if type_name in self.struct_types:
            return self.struct_types[type_name]

        struct_node = self.struct_declarations.get(type_name)
        if struct_node is None:
            return None
        if type_name in self.struct_registration_stack:
            return None

        self.struct_registration_stack.add(type_name)
        try:
            return self.process_crossgl_struct(struct_node)
        finally:
            self.struct_registration_stack.remove(type_name)

    def ensure_enum_struct_type_registered(self, enum_name: str) -> Optional[SpirvId]:
        if enum_name in self.struct_types:
            return self.struct_types[enum_name]
        if enum_name not in self.enum_struct_type_names:
            return None

        if (
            enum_name not in self.enum_declarations
            or enum_name in self.enum_struct_registration_stack
        ):
            return None

        self.enum_struct_registration_stack.add(enum_name)
        try:
            int_type = self.register_primitive_type("int")
            members = [(int_type, "variant")]
            for field_name, field_type in self.enum_struct_fields.get(enum_name, []):
                members.append((self.map_crossgl_type(field_type), field_name))
            return self.register_struct_type(enum_name, members)
        finally:
            self.enum_struct_registration_stack.remove(enum_name)

    def generic_argument_text(self, argument) -> str:
        """A generic argument as source text; flag expressions such as ``A | B`` too."""
        if isinstance(argument, BinaryOpNode):
            operator = getattr(argument, "operator", getattr(argument, "op", ""))
            left = self.generic_argument_text(argument.left)
            right = self.generic_argument_text(argument.right)
            return f"({left} {operator} {right})"
        if isinstance(argument, IdentifierNode):
            return argument.name
        if isinstance(argument, LiteralNode):
            return str(argument.value)
        return self.convert_type_node_to_string(argument)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if isinstance(type_node, FunctionCallNode):
            callee = self.convert_type_node_to_string(type_node.function)
            args = ", ".join(
                self.convert_type_node_to_string(arg)
                for arg in getattr(type_node, "arguments", []) or []
            )
            return f"{callee}({args})"
        if isinstance(type_node, IdentifierNode):
            return type_node.name
        if isinstance(type_node, LiteralNode):
            return str(type_node.value)
        if type_node.__class__.__name__ == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return prepend_array_dimension(element_type, size)
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            return f"{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            return f"&{referenced_type}"
        if hasattr(type_node, "return_type") and hasattr(type_node, "param_types"):
            return_type = self.convert_type_node_to_string(type_node.return_type)
            param_types = ", ".join(
                self.convert_type_node_to_string(param_type)
                for param_type in getattr(type_node, "param_types", []) or []
            )
            return f"fn({param_types}) -> {return_type}"
        if hasattr(type_node, "name") and type_node.name is not None:
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.generic_argument_text(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            return matrix_type_spelling(type_node, element_type)
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type in {"float", "f32"}:
                return f"vec{size}"
            elif element_type in {"int", "i32"}:
                return f"ivec{size}"
            elif element_type in {"uint", "u32"}:
                return f"uvec{size}"
            elif element_type in {"double", "f64"}:
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            else:
                return f"{element_type}{size}"
        else:
            return str(type_node)

    def pointer_pointee_type_name_from_string(self, type_name) -> Optional[str]:
        """Return the pointee type for CrossGL pointer spelling such as T*."""
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return None

        type_str = type_str.strip()
        if not type_str.endswith("*"):
            return None

        pointee = type_str[:-1].strip()
        return pointee or None

    def storage_buffer_type_is_inferred_pointer(self, type_name) -> bool:
        """Return whether a storage-buffer pointer spelling still needs inference."""
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return False

        type_str = self.normalize_reference_type_name(type_str)
        if type_str is None:
            return False

        type_str = str(type_str).strip()
        if not type_str.endswith("*"):
            return False

        pointee = type_str[:-1].strip()
        pointee = re.sub(
            r"\b(?:const|volatile|device|constant|thread|threadgroup|restrict)\b",
            "",
            pointee,
        )
        pointee = re.sub(r"\s+", "", pointee)
        for qualifier in (
            "threadgroup",
            "constant",
            "volatile",
            "restrict",
            "device",
            "thread",
            "const",
        ):
            while pointee.startswith(qualifier) and len(pointee) > len(qualifier):
                pointee = pointee[len(qualifier) :]
        return pointee in {"auto", "decltype(auto)"}

    def normalize_reference_type_name(self, type_name) -> Optional[str]:
        """Return the value type for CrossGL reference spelling such as &T or T&."""
        if type_name is None:
            return None

        type_str = str(type_name).strip()
        while type_str.startswith("&"):
            type_str = type_str[1:].strip()
        while type_str.endswith("&"):
            type_str = type_str[:-1].strip()
        return type_str or None

    def atomic_element_type_name(self, type_name) -> Optional[str]:
        """Return the scalar payload type for atomic<T> aliases."""
        type_str = self.type_name_from_value(type_name)
        if type_str is None:
            return None

        compact = re.sub(r"\s+", "", str(type_str))
        atomic_match = re.fullmatch(r"atomic<(.+)>", compact)
        if atomic_match:
            return atomic_match.group(1)

        return {
            "atomic_int": "int",
            "atomic_uint": "uint",
        }.get(compact)

    def storage_buffer_element_type_name(self, type_name) -> str:
        atomic_type_name = self.atomic_element_type_name(type_name)
        return atomic_type_name or self.type_name_from_value(type_name)

    def collect_enum_metadata(self, nodes):
        """Collect SPIR-V-lowerable enum types and variant discriminants."""
        for node in nodes or []:
            if isinstance(node, EnumNode):
                self.register_enum_metadata(node)
            elif isinstance(node, StructNode):
                if node.name in self.generic_enum_struct_definitions:
                    continue
                self.collect_enum_metadata(getattr(node, "members", []) or [])

    def register_enum_metadata(self, enum_node: EnumNode):
        variants = getattr(enum_node, "variants", []) or []
        has_payload = any(
            self.enum_variant_has_payload(variant) for variant in variants
        )
        if has_payload:
            fields = enum_struct_fields(enum_node)
            if fields is not None:
                self.enum_struct_type_names.add(enum_node.name)
                self.enum_struct_fields[enum_node.name] = fields
                self.enum_struct_variant_fields[enum_node.name] = {
                    variant.name: enum_variant_payload_fields(variant) or []
                    for variant in variants
                }
        else:
            self.enum_type_names.add(enum_node.name)

        next_value = 0
        for variant in variants:
            explicit_value = self.literal_int_argument(getattr(variant, "value", None))
            if explicit_value is not None:
                next_value = explicit_value
            self.enum_variant_values[f"{enum_node.name}::{variant.name}"] = next_value
            next_value += 1

    def register_generic_enum_metadata(self):
        """Collect generic enum wrapper variant tags shared by all specializations."""
        for name, definition in self.generic_enum_struct_definitions.items():
            self.enum_struct_type_names.add(name)
            next_value = 0
            for variant in definition["enum"].variants or []:
                explicit_value = self.literal_int_argument(
                    getattr(variant, "value", None)
                )
                if explicit_value is not None:
                    next_value = explicit_value
                self.enum_variant_values[f"{name}::{variant.name}"] = next_value
                next_value += 1

    def generic_enum_specialization_for_type(self, type_value) -> Optional[SpirvId]:
        specialization = resolve_generic_enum_specialization(self, type_value)
        if specialization is None:
            return None
        return self.ensure_generic_enum_specialization_registered(specialization)

    def generic_enum_specialization_for_struct_name(self, struct_name: str):
        for specialization in self.generic_enum_specializations.values():
            if specialization["struct_name"] == struct_name:
                return specialization
        return None

    def generic_enum_constructor_specialization(self, enum_name: str):
        expected_type = self.current_expression_expected_type
        specialization = resolve_generic_enum_specialization(
            self, expected_type, expected_base=enum_name
        )
        if specialization is not None:
            self.ensure_generic_enum_specialization_registered(specialization)
            return specialization

        specialization = self.generic_enum_specialization_for_struct_name(
            str(expected_type)
        )
        if specialization is not None and specialization["base_name"] == enum_name:
            return specialization

        candidates = [
            specialization
            for specialization in self.generic_enum_specializations.values()
            if specialization["base_name"] == enum_name
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def ensure_generic_enum_specialization_registered(self, specialization) -> SpirvId:
        type_name = specialization["type_name"]
        struct_name = specialization["struct_name"]
        self.generic_enum_specializations.setdefault(type_name, specialization)
        self.enum_struct_type_names.add(struct_name)

        if struct_name not in self.enum_struct_fields:
            self.enum_struct_fields[struct_name] = generic_enum_specialized_fields(
                self, specialization
            )
            self.enum_struct_variant_fields[struct_name] = {
                variant.name: (
                    generic_enum_specialized_variant_fields(
                        self, specialization, variant.name
                    )
                    or []
                )
                for variant in specialization["definition"]["enum"].variants or []
            }

        if struct_name not in self.struct_types:
            int_type = self.register_primitive_type("int")
            members = [(int_type, "variant")]
            for field_name, field_type in self.enum_struct_fields.get(struct_name, []):
                members.append((self.map_crossgl_type(field_type), field_name))
            self.register_struct_type(struct_name, members)

        return self.struct_types[struct_name]

    def enum_variant_has_payload(self, variant) -> bool:
        return bool(getattr(variant, "data", None) or getattr(variant, "fields", None))

    def enum_variant_constant(self, name: str) -> Optional[SpirvId]:
        value = self.enum_variant_values.get(name)
        if value is None:
            return None
        return self.register_constant(value, self.register_primitive_type("int"))

    def process_enum_structs(self, nodes):
        """Register tagged-struct representations for payload enums."""
        for node in nodes or []:
            if isinstance(node, EnumNode) and node.name in self.enum_struct_type_names:
                self.ensure_enum_struct_type_registered(node.name)
            elif isinstance(node, StructNode):
                self.process_enum_structs(getattr(node, "members", []) or [])

    def enum_path_parts(self, path: str):
        if "::" not in str(path):
            return None
        enum_name, variant_name = str(path).split("::", 1)
        return enum_name, variant_name

    def enum_variant_is_payload_path(self, path: str) -> bool:
        parts = self.enum_path_parts(path)
        if parts is None:
            return False
        enum_name, _variant_name = parts
        return enum_name in self.enum_struct_type_names

    def enum_variant_fields_for_path(
        self, path: str, expression: Optional[SpirvId] = None
    ):
        parts = self.enum_path_parts(path)
        if parts is None:
            return None
        enum_name, variant_name = parts
        if enum_name in self.generic_enum_struct_definitions:
            specialization = None
            if expression is not None:
                specialization = self.generic_enum_specialization_for_struct_name(
                    expression.type.base_type
                )
            if specialization is None:
                specialization = self.generic_enum_constructor_specialization(enum_name)
            if specialization is None:
                return None
            return generic_enum_specialized_variant_fields(
                self, specialization, variant_name
            )
        return self.enum_struct_variant_fields.get(enum_name, {}).get(variant_name)

    def process_enum_variant_constructor(
        self,
        path: str,
        positional_args,
        named_args=None,
    ) -> Optional[SpirvId]:
        parts = self.enum_path_parts(path)
        if parts is None:
            return None
        enum_name, _variant_name = parts
        if enum_name in self.generic_enum_struct_definitions:
            return self.process_generic_enum_variant_constructor(
                path, positional_args, named_args
            )

        if enum_name not in self.enum_struct_type_names:
            if positional_args or named_args:
                return None
            return self.enum_variant_constant(path)

        enum_type = self.struct_types.get(enum_name)
        if enum_type is None:
            return None

        variant_fields = self.enum_variant_fields_for_path(path)
        if variant_fields is None:
            return None

        named_args = dict(named_args or {})
        if len(positional_args) > len(variant_fields):
            raise VulkanRefusal(
                f"Enum constructor {path} expects at most {len(variant_fields)} "
                f"arguments, got {len(positional_args)}"
            )

        unknown_names = sorted(set(named_args) - {name for name, _ in variant_fields})
        if unknown_names:
            raise VulkanRefusal(
                f"Enum constructor {path} has no field {', '.join(unknown_names)}"
            )

        active_values = {}
        missing_names = []
        for index, (field_name, field_type) in enumerate(variant_fields):
            if index < len(positional_args):
                value_expr = positional_args[index]
            elif field_name in named_args:
                value_expr = named_args[field_name]
            else:
                missing_names.append(field_name)
                continue

            value = self.process_expression(value_expr)
            if value is None:
                return None
            active_values[field_name] = self.convert_value_to_type(
                value, self.map_crossgl_type(field_type)
            )

        if missing_names:
            raise VulkanRefusal(
                f"Enum constructor {path} is missing field {', '.join(missing_names)}"
            )

        components = [self.enum_variant_constant(path)]
        for field_name, field_type in self.enum_struct_fields.get(enum_name, []):
            field_type_id = self.map_crossgl_type(field_type)
            components.append(
                active_values.get(field_name)
                or self.default_value_for_type(field_type_id)
            )

        return self.composite_construct(enum_type, components)

    def process_generic_enum_variant_constructor(
        self,
        path: str,
        positional_args,
        named_args=None,
    ) -> Optional[SpirvId]:
        enum_name, variant_name = self.enum_path_parts(path)
        specialization = self.generic_enum_constructor_specialization(enum_name)
        if specialization is None:
            return None

        enum_type = self.ensure_generic_enum_specialization_registered(specialization)
        variant_fields = generic_enum_specialized_variant_fields(
            self, specialization, variant_name
        )
        if variant_fields is None:
            return None

        named_args = dict(named_args or {})
        if len(positional_args) > len(variant_fields):
            raise VulkanRefusal(
                f"Enum constructor {path} expects at most {len(variant_fields)} "
                f"arguments, got {len(positional_args)}"
            )

        unknown_names = sorted(set(named_args) - {name for name, _ in variant_fields})
        if unknown_names:
            raise VulkanRefusal(
                f"Enum constructor {path} has no field {', '.join(unknown_names)}"
            )

        active_values = {}
        missing_names = []
        for index, (field_name, field_type) in enumerate(variant_fields):
            if index < len(positional_args):
                value_expr = positional_args[index]
            elif field_name in named_args:
                value_expr = named_args[field_name]
            else:
                missing_names.append(field_name)
                continue

            value = self.process_expression_with_expected_type(value_expr, field_type)
            if value is None:
                return None
            active_values[field_name] = self.convert_value_to_type(
                value, self.map_crossgl_type(field_type)
            )

        if missing_names:
            raise VulkanRefusal(
                f"Enum constructor {path} is missing field {', '.join(missing_names)}"
            )

        components = [self.enum_variant_constant(path)]
        for field_name, field_type in self.enum_struct_fields.get(
            specialization["struct_name"], []
        ):
            field_type_id = self.map_crossgl_type(field_type)
            components.append(
                active_values.get(field_name)
                or self.default_value_for_type(field_type_id)
            )

        return self.composite_construct(enum_type, components)

    def process_generic_struct_constructor_call(
        self, callee_name, expr
    ) -> Optional[SpirvId]:
        """Lower ``Pair<vec2, float>(a, b)``, a constructor of a generic structure's instance, else ``None``."""
        if not isinstance(callee_name, str) or "<" not in callee_name:
            return None
        struct_name = materialize_generic_struct_instance(
            self, callee_name, self.program_generic_parameter_names
        )
        if struct_name is None or self.ensure_declared_struct_type(struct_name) is None:
            return None
        return self.process_struct_constructor_node(
            ConstructorNode(
                NamedType(struct_name),
                list(getattr(expr, "arguments", getattr(expr, "args", [])) or []),
            )
        )

    def process_struct_constructor_node(
        self, expr: ConstructorNode
    ) -> Optional[SpirvId]:
        type_name = self.convert_type_node_to_string(expr.constructor_type)
        if "::" in type_name:
            return self.process_enum_variant_constructor(
                type_name,
                list(getattr(expr, "arguments", []) or []),
                getattr(expr, "named_arguments", {}) or {},
            )

        struct_type = self.struct_types.get(type_name)
        if struct_type is None:
            return None

        members = self.current_struct_members.get(type_name, [])
        positional_args = list(getattr(expr, "arguments", []) or [])
        named_args = dict(getattr(expr, "named_arguments", {}) or {})
        member_names = [member_name for _member_type, member_name in members]
        unknown_names = sorted(set(named_args) - set(member_names))
        if unknown_names:
            raise VulkanRefusal(
                f"Struct constructor {type_name} has no field "
                f"{', '.join(unknown_names)}"
            )

        if (
            len(positional_args) == 1
            and not named_args
            and self.expression_result_type(positional_args[0]) == type_name
        ):
            copy_value = self.process_expression_with_expected_type(
                positional_args[0],
                type_name,
            )
            if copy_value is not None and self.value_has_type(copy_value, struct_type):
                return copy_value

        components = []
        for index, (member_type, member_name) in enumerate(members):
            if index < len(positional_args):
                value_expr = positional_args[index]
            elif member_name in named_args:
                value_expr = named_args[member_name]
            else:
                components.append(self.default_value_for_type(member_type))
                continue

            value = self.process_expression_with_expected_type(
                value_expr, member_type.type.base_type
            )
            if value is None:
                return None
            components.append(self.convert_value_to_type(value, member_type))

        if len(positional_args) > len(members):
            self.refuse_lowering(
                f"Constructor {type_name} expected {len(members)} "
                f"members but got {len(positional_args)}"
            )

        return self.composite_construct(struct_type, components)

    def struct_member_unlowered_pointer_info(
        self, struct_node: StructNode, member
    ) -> Optional[dict]:
        raw_type = (
            getattr(member, "element_type", None)
            if isinstance(member, ArrayNode)
            else getattr(
                member,
                "member_type",
                getattr(member, "var_type", getattr(member, "vtype", None)),
            )
        )
        type_text = (
            self.convert_type_node_to_string(raw_type)
            if raw_type is not None and not isinstance(raw_type, str)
            else str(raw_type or "")
        ).strip()
        if "*" not in type_text:
            return None
        qualifiers = [
            str(q) for q in getattr(member, "qualifiers", None) or () if q
        ] + [
            str(q)
            for q in (
                getattr(getattr(member, "attributes", None), "qualifiers", None) or ()
            )
            if q
        ]
        address_space = getattr(raw_type, "address_space", None)
        if not address_space:
            for candidate in (
                "device",
                "constant",
                "threadgroup",
                "thread",
                "ray_data",
            ):
                if candidate in qualifiers or re.search(rf"\b{candidate}\b", type_text):
                    address_space = candidate
                    break
        if address_space is None:
            for attr in getattr(member, "attributes", None) or ():
                attr_name = getattr(attr, "name", None)
                attr_args = getattr(attr, "args", None) or ()
                if attr_name in {"address_space", "resource_space"} and attr_args:
                    address_space = str(attr_args[0]).strip()
                    break
        entry_point = None
        stages = getattr(getattr(self, "spirv_program_ast", None), "stages", None) or {}
        for stage in stages.values():
            entry = getattr(stage, "entry_point", None)
            if getattr(entry, "name", None):
                entry_point = str(entry.name)
                break
        return {
            "struct_name": getattr(struct_node, "name", None),
            "member_name": getattr(member, "name", None),
            "type_name": type_text,
            "address_space": address_space or "unspecified",
            "entry_point": entry_point,
        }

    def process_crossgl_struct(self, struct_node: StructNode) -> Optional[SpirvId]:
        """Process a CrossGL struct definition."""
        if struct_node.name in self.generic_enum_struct_definitions:
            return self.struct_types.get(struct_node.name)
        if getattr(
            struct_node, "generic_params", None
        ) and self.generic_struct_requires_materialization(struct_node):
            return None

        members = []
        member_metadata = {}
        allow_runtime_array = struct_node.name in self.glsl_buffer_block_type_names
        struct_constants = self.struct_compile_time_integer_constants(struct_node)

        for member in struct_node.members:
            if self.is_static_struct_member(member):
                continue
            member_type = None
            member_name = member.name
            pointer_info = self.struct_member_unlowered_pointer_info(
                struct_node, member
            )
            if pointer_info is not None:
                entry_suffix = (
                    f" in entry point '{pointer_info['entry_point']}'"
                    if pointer_info.get("entry_point")
                    else ""
                )
                self.refuse_lowering(
                    f"Pointer-bearing aggregate member '{struct_node.name}.{member_name}' "
                    f"({pointer_info['type_name']}, address space '{pointer_info['address_space']}') "
                    f"was not lowered before SPIR-V struct emission{entry_suffix}"
                )
                continue

            if isinstance(member, ArrayNode):
                element_type = member.element_type
                if hasattr(element_type, "name") or hasattr(
                    element_type, "element_type"
                ):
                    element_type = self.convert_type_node_to_string(element_type)
                size = self.format_array_size(member.size, struct_constants)
                member_type_name = (
                    f"{element_type}[{size}]"
                    if size is not None
                    else f"{element_type}[]"
                )
                member_type_name = self.resolve_struct_member_array_extents(
                    member_type_name,
                    struct_constants,
                    struct_name=struct_node.name,
                    member_name=member_name,
                )
                member_type = self.map_crossgl_type(
                    self.spirv_struct_member_storage_type_name(
                        member_type_name, allow_runtime_array=allow_runtime_array
                    )
                )
            else:
                member_type_source = getattr(
                    member,
                    "member_type",
                    getattr(member, "var_type", getattr(member, "vtype", None)),
                )
                if member_type_source is not None:
                    member_type_source = self.resolve_struct_member_array_extents(
                        member_type_source,
                        struct_constants,
                        struct_name=struct_node.name,
                        member_name=member_name,
                    )
                    member_type = self.map_crossgl_type(
                        self.spirv_struct_member_storage_type_name(
                            member_type_source,
                            allow_runtime_array=allow_runtime_array,
                        )
                    )

            if member_type:
                members.append((member_type, member_name))
                member_metadata[member_name] = {
                    "node": member,
                    "semantic": self.semantic_from_node(member),
                    "type": member_type,
                }

        if any(
            self.type_contains_runtime_array(member_type)
            for member_type, _member_name in members
        ):
            members = [
                (self.storage_layout_type(member_type, "std430"), member_name)
                for member_type, member_name in members
            ]
            self.std430_runtime_array_struct_types.add(struct_node.name)
            for member_type, member_name in members:
                if member_name in member_metadata:
                    member_metadata[member_name]["type"] = member_type

        self.struct_member_metadata[struct_node.name] = member_metadata
        return self.register_struct_type(struct_node.name, members)

    def generic_struct_requires_materialization(self, struct_node) -> bool:
        """Return whether a generic struct has no legal placeholder layout."""
        type_parameters = {
            str(getattr(parameter, "name", parameter))
            for parameter in getattr(struct_node, "generic_params", None) or []
        }
        for member in getattr(struct_node, "members", []) or []:
            if isinstance(member, ArrayNode):
                if (
                    member.size is None
                    or evaluate_literal_int_expression(
                        member.size, self.literal_int_constants
                    )
                    is None
                ):
                    return True
                member_type = member.element_type
            else:
                member_type = getattr(
                    member,
                    "member_type",
                    getattr(member, "var_type", getattr(member, "vtype", None)),
                )
            if self.generic_struct_member_type_requires_materialization(member_type):
                return True
            # A stored member typed by the struct's own parameter has no layout until instantiation.
            if not self.is_static_struct_member(
                member
            ) and self.type_mentions_type_parameter(member_type, type_parameters):
                return True
        return False

    def type_mentions_type_parameter(self, type_value, type_parameters) -> bool:
        """Return whether a member type names one of a generic struct's type parameters."""
        if type_value is None or not type_parameters:
            return False
        if not isinstance(type_value, str):
            type_value = self.convert_type_node_to_string(type_value)
        return any(
            name in type_parameters
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(type_value))
        )

    def generic_struct_member_type_requires_materialization(self, type_value) -> bool:
        if type_value is None:
            return False
        if type_value.__class__.__name__ == "ArrayType":
            if (
                type_value.size is None
                or evaluate_literal_int_expression(
                    type_value.size, self.literal_int_constants
                )
                is None
            ):
                return True
            return self.generic_struct_member_type_requires_materialization(
                type_value.element_type
            )
        if hasattr(type_value, "pointee_type"):
            return True
        if isinstance(type_value, str):
            if "*" in type_value:
                return True
            for extent in re.findall(r"\[([^\[\]]*)\]", type_value):
                if (
                    not extent.strip()
                    or evaluate_literal_int_expression(
                        extent, self.literal_int_constants
                    )
                    is None
                ):
                    return True
        return False

    def is_static_struct_member(self, member) -> bool:
        attributes = {
            str(getattr(attribute, "name", attribute)).lower()
            for attribute in getattr(member, "attributes", []) or []
        }
        return "static" in attributes

    def struct_compile_time_integer_constants(self, struct_node) -> dict:
        constants = dict(self.literal_int_constants)
        for member in getattr(struct_node, "members", []) or []:
            if not self.is_static_struct_member(member):
                continue
            default_value = getattr(member, "default_value", None)
            if default_value is None:
                continue
            value = evaluate_literal_int_expression(default_value, constants)
            if value is not None:
                constants[member.name] = value
        return constants

    def resolve_struct_member_array_extents(
        self,
        type_name,
        constants: dict,
        *,
        struct_name: str,
        member_name: str,
    ) -> str:
        if type_name.__class__.__name__ == "ArrayType":
            element_type = self.resolve_struct_member_array_extents(
                type_name.element_type,
                constants,
                struct_name=struct_name,
                member_name=member_name,
            )
            if type_name.size is None:
                return prepend_array_dimension(element_type)
            value = evaluate_literal_int_expression(type_name.size, constants)
            if value is None:
                extent = self.struct_array_extent_expression_text(type_name.size)
                if self.named_specialization_array_length(extent) is not None:
                    return prepend_array_dimension(element_type, extent)
                raise UnsupportedSPIRVFeatureError(
                    "unresolved fixed array extent",
                    "SPIR-V requires a concrete fixed-array extent for "
                    f"{struct_name}.{member_name}; could not resolve '{extent}'",
                )
            return prepend_array_dimension(element_type, value)

        type_text = self.type_name_from_value(type_name)
        if type_text is None or "[" not in type_text:
            return type_text

        def resolve_extent(match):
            extent = match.group(1).strip()
            if not extent:
                return "[]"
            value = evaluate_literal_int_expression(extent, constants)
            if value is None:
                if self.named_specialization_array_length(extent) is not None:
                    return f"[{extent}]"
                raise UnsupportedSPIRVFeatureError(
                    "unresolved fixed array extent",
                    "SPIR-V requires a concrete fixed-array extent for "
                    f"{struct_name}.{member_name}; could not resolve '{extent}'",
                )
            return f"[{value}]"

        return re.sub(r"\[([^\[\]]*)\]", resolve_extent, type_text)

    def struct_array_extent_expression_text(self, expression) -> str:
        if isinstance(expression, IdentifierNode):
            return expression.name
        if isinstance(expression, LiteralNode):
            return str(expression.value)
        if isinstance(expression, BinaryOpNode):
            left = self.struct_array_extent_expression_text(expression.left)
            right = self.struct_array_extent_expression_text(expression.right)
            if isinstance(expression.left, BinaryOpNode):
                left = f"({left})"
            if isinstance(expression.right, BinaryOpNode):
                right = f"({right})"
            return f"{left} {expression.op} {right}"
        if isinstance(expression, UnaryOpNode):
            operand = self.struct_array_extent_expression_text(expression.operand)
            if isinstance(expression.operand, BinaryOpNode):
                operand = f"({operand})"
            return f"{expression.op}{operand}"
        return str(expression)
