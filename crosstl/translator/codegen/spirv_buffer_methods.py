"""Structured and byte address buffer methods in the SPIR-V generator: structured buffer dimensions, counters, appends and method calls, byte address store widths, interlocked atomics, component indices, load helpers and method calls, array element pointer aliases, pointer ternaries, side-effect-free arguments, and array and vector type info."""

from typing import (
    List,
    Optional,
    Tuple,
)

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    BinaryOpNode,
    ConstantNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerReinterpretNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import evaluate_literal_int_expression
from .spirv_byte_address_atomics import process_byte_address_member_min_max
from .spirv_single_evaluation import evaluate_subscript_index
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)


class SpirvBufferMethodMixin:
    """Buffer method members of ``VulkanSPIRVCodeGen``."""

    def store_structured_buffer_dimensions_result(
        self,
        target_expr,
        value: SpirvId,
        diagnostic_name: str,
    ) -> bool:
        target_pointer = self.assignable_pointer_from_expression(target_expr)
        if target_pointer is None:
            self.refuse_lowering(
                f"{diagnostic_name} output operand must be an assignable "
                "integer target"
            )
            return False

        target_type = self.pointer_pointee_type(target_pointer)
        if target_type is None:
            self.refuse_lowering(
                f"{diagnostic_name} output operand type could not be " "determined"
            )
            return False

        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        if target_type_name not in {"int", "uint"}:
            self.refuse_lowering(
                f"{diagnostic_name} output operand must be an integer " "target"
            )
            return False

        self.store_to_variable(
            target_pointer, self.convert_value_to_type(value, target_type)
        )
        return True

    def process_buffer_dimensions_function_call(
        self, expr: FunctionCallNode
    ) -> SpirvId:
        diagnostic_name = "buffer_dimensions"
        args = list(getattr(expr, "args", []) or [])
        if not args:
            self.refuse_lowering("buffer_dimensions requires a buffer operand")
            return self.structured_buffer_dimensions_default_value()
        if len(args) > 2:
            self.refuse_lowering(
                "buffer_dimensions accepts only buffer and optional " "output operands"
            )
            return self.structured_buffer_dimensions_default_value()

        buffer_pointer = self.variable_pointer_from_expression(args[0])
        if buffer_pointer is None:
            self.refuse_lowering(
                "buffer_dimensions requires a structured or byte-address "
                "buffer operand"
            )
            return self.structured_buffer_dimensions_default_value()

        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            self.refuse_lowering(
                "buffer_dimensions requires a structured or byte-address "
                "buffer operand"
            )
            return self.structured_buffer_dimensions_default_value()

        length = self.emit_structured_buffer_dimensions(
            buffer_pointer, metadata, diagnostic_name
        )
        if len(args) == 2:
            self.store_structured_buffer_dimensions_result(
                args[1], length, diagnostic_name
            )
        return length

    def process_structured_buffer_dimensions_method_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
    ) -> SpirvId:
        if len(args) > 1:
            self.refuse_lowering(
                f"{diagnostic_name} accepts only an optional output " "operand"
            )
            return self.structured_buffer_dimensions_default_value()

        length = self.emit_structured_buffer_dimensions(
            buffer_pointer, metadata, diagnostic_name
        )
        if args:
            self.store_structured_buffer_dimensions_result(
                args[0], length, diagnostic_name
            )
        return length

    def ensure_structured_buffer_counter_metadata(self, metadata) -> bool:
        if metadata.get("counter_variable") is not None:
            return True
        if metadata.get("buffer_kind") != "RWStructuredBuffer":
            return False

        root_pointer = metadata.get("buffer_variable")
        root_metadata = (
            self.structured_buffer_metadata.get(root_pointer.id)
            if root_pointer is not None
            else None
        )
        if root_metadata is not None and root_metadata.get("counter_variable"):
            counter_metadata = {
                key: value
                for key, value in root_metadata.items()
                if key.startswith("counter_")
            }
            counter_metadata["counter_variable"] = root_metadata["counter_variable"]
            metadata.update(counter_metadata)
            return True

        declaration_node = metadata.get("declaration_node")
        declared_type_name = metadata.get("declared_type_name")
        if declaration_node is None or declared_type_name is None:
            return False

        counter_metadata = self.process_structured_buffer_counter_declaration(
            declaration_node, declared_type_name
        )
        metadata.update(counter_metadata)
        if root_metadata is not None:
            root_metadata.update(counter_metadata)
            self.structured_buffer_metadata[root_pointer.id] = root_metadata
            block_type = root_metadata.get("block_type")
            if block_type is not None:
                self.structured_buffer_metadata[block_type.id] = root_metadata
        return True

    def structured_buffer_counter_pointer(
        self, buffer_pointer: SpirvId
    ) -> Optional[SpirvId]:
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None:
            return None

        counter_pointer = metadata.get("counter_variable")
        counter_block_type = metadata.get("counter_block_type")
        if counter_pointer is None or counter_block_type is None:
            return None
        if self.include_resource_interface_variables:
            self.mark_function_interface_variable(counter_pointer)

        counter_value_type = self.variable_value_types.get(counter_pointer.id)
        for index in metadata.get("_descriptor_indices", []):
            counter_element_type = self.array_element_type_from_type(counter_value_type)
            if counter_element_type is None:
                return None

            ptr_type = self.register_pointer_type(
                counter_element_type, self.storage_buffer_storage_class()
            )
            counter_pointer = self.access_chain(counter_pointer, [index], ptr_type)
            self.variable_value_types[counter_pointer.id] = counter_element_type
            counter_value_type = counter_element_type

        if counter_value_type is None or counter_value_type.id != counter_block_type.id:
            return None

        uint_type = self.register_primitive_type("uint")
        member_index = self.register_constant(
            metadata.get("counter_member_index", 0), self.primitive_types["int"]
        )
        ptr_type = self.register_pointer_type(
            uint_type, self.storage_buffer_storage_class()
        )
        access = self.access_chain(counter_pointer, [member_index], ptr_type)
        self.variable_value_types[access.id] = uint_type
        self.storage_buffer_access_metadata[access.id] = {
            "kind": "structured_buffer_counter",
            "block_type": counter_block_type,
            "readonly": False,
            "writeonly": False,
        }
        return access

    def emit_structured_buffer_counter_atomic(
        self, opcode: str, counter_pointer: SpirvId, value: SpirvId
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        value = self.convert_value_to_type(value, uint_type)
        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = {opcode} %{uint_type.id} %{counter_pointer.id} "
            f"%{scope.id} %{semantics.id} %{value.id}"
        )
        self.value_types[id_value] = uint_type
        return SpirvId(id_value, uint_type.type)

    def process_structured_buffer_append_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
    ) -> Tuple[bool, Optional[SpirvId]]:
        if len(args) < 1:
            self.refuse_lowering(f"{diagnostic_name} requires a value operand")
            return True, None
        if len(args) > 1:
            self.refuse_lowering(f"{diagnostic_name} accepts only a value operand")
            return True, None
        if metadata.get("buffer_kind") != "AppendStructuredBuffer":
            self.refuse_lowering(
                f"{diagnostic_name} requires an AppendStructuredBuffer"
            )
            return True, None
        if metadata.get("readonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a writable buffer")
            return True, None

        value = (
            args[0]
            if isinstance(args[0], SpirvId)
            else self.process_expression(args[0])
        )
        if value is None:
            self.refuse_lowering(f"{diagnostic_name} value could not be evaluated")
            return True, None
        value = self.convert_value_to_type(value, metadata["element_type"])

        counter_pointer = self.structured_buffer_counter_pointer(buffer_pointer)
        if counter_pointer is None:
            self.refuse_lowering(f"{diagnostic_name} requires a counter buffer")
            return True, None

        one = self.register_constant(1, self.register_primitive_type("uint"))
        index = self.emit_structured_buffer_counter_atomic(
            "OpAtomicIAdd", counter_pointer, one
        )
        element_pointer = self.structured_buffer_element_pointer(buffer_pointer, index)
        if element_pointer is None:
            self.refuse_lowering(
                f"{diagnostic_name} requires an AppendStructuredBuffer " "element"
            )
            return True, None

        self.store_to_variable(element_pointer, value)
        return True, None

    def process_structured_buffer_consume_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
    ) -> Tuple[bool, Optional[SpirvId]]:
        if args:
            self.refuse_lowering(f"{diagnostic_name} accepts no operands")
            return True, self.structured_buffer_default_value(metadata)
        if metadata.get("buffer_kind") != "ConsumeStructuredBuffer":
            self.refuse_lowering(
                f"{diagnostic_name} requires a ConsumeStructuredBuffer"
            )
            return True, self.structured_buffer_default_value(metadata)
        if metadata.get("writeonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a readable buffer")
            return True, self.structured_buffer_default_value(metadata)

        counter_pointer = self.structured_buffer_counter_pointer(buffer_pointer)
        if counter_pointer is None:
            self.refuse_lowering(f"{diagnostic_name} requires a counter buffer")
            return True, self.structured_buffer_default_value(metadata)

        uint_type = self.register_primitive_type("uint")
        one = self.register_constant(1, uint_type)
        old_count = self.emit_structured_buffer_counter_atomic(
            "OpAtomicISub", counter_pointer, one
        )
        index = self.binary_operation("-", uint_type, old_count, one)
        element_pointer = self.structured_buffer_element_pointer(buffer_pointer, index)
        if element_pointer is None:
            self.refuse_lowering(
                f"{diagnostic_name} requires a ConsumeStructuredBuffer " "element"
            )
            return True, self.structured_buffer_default_value(metadata)

        element_type = self.variable_value_types[element_pointer.id]
        return True, self.load_from_variable(element_pointer, element_type)

    def process_structured_buffer_counter_method_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        args,
        diagnostic_name: str,
        increment: bool,
    ) -> Tuple[bool, SpirvId]:
        if args:
            self.refuse_lowering(f"{diagnostic_name} accepts no operands")
            return True, self.structured_buffer_counter_default_value()
        if metadata.get("buffer_kind") != "RWStructuredBuffer":
            self.refuse_lowering(f"{diagnostic_name} requires an RWStructuredBuffer")
            return True, self.structured_buffer_counter_default_value()
        if metadata.get("readonly") or metadata.get("writeonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a read-write buffer")
            return True, self.structured_buffer_counter_default_value()
        if not self.ensure_structured_buffer_counter_metadata(metadata):
            self.refuse_lowering(f"{diagnostic_name} requires a counter buffer")
            return True, self.structured_buffer_counter_default_value()

        counter_pointer = self.structured_buffer_counter_pointer(buffer_pointer)
        if counter_pointer is None:
            self.refuse_lowering(f"{diagnostic_name} requires a counter buffer")
            return True, self.structured_buffer_counter_default_value()

        uint_type = self.register_primitive_type("uint")
        one = self.register_constant(1, uint_type)
        if increment:
            counter_value = self.emit_structured_buffer_counter_atomic(
                "OpAtomicIAdd", counter_pointer, one
            )
            return True, counter_value

        old_count = self.emit_structured_buffer_counter_atomic(
            "OpAtomicISub", counter_pointer, one
        )
        return True, self.binary_operation("-", uint_type, old_count, one)

    def process_structured_buffer_method_call(
        self, expr: FunctionCallNode
    ) -> Tuple[bool, Optional[SpirvId]]:
        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(callee_expr, MemberAccessNode):
            return False, None

        method_name = getattr(callee_expr, "member", None)
        if method_name not in {
            "Load",
            "Store",
            "GetDimensions",
            "Append",
            "Consume",
            "IncrementCounter",
            "DecrementCounter",
        }:
            return False, None

        buffer_pointer = self.variable_pointer_from_expression(callee_expr.object)
        if buffer_pointer is None:
            return False, None

        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None or metadata.get("byte_address"):
            return False, None

        args = list(getattr(expr, "args", []) or [])
        if method_name == "GetDimensions":
            return True, self.process_structured_buffer_dimensions_method_call(
                buffer_pointer,
                metadata,
                args,
                "StructuredBuffer.GetDimensions",
            )
        if method_name == "Append":
            return self.process_structured_buffer_append_call(
                buffer_pointer,
                metadata,
                args,
                "AppendStructuredBuffer.Append",
            )
        if method_name == "Consume":
            return self.process_structured_buffer_consume_call(
                buffer_pointer,
                metadata,
                args,
                "ConsumeStructuredBuffer.Consume",
            )
        if method_name in {"IncrementCounter", "DecrementCounter"}:
            return self.process_structured_buffer_counter_method_call(
                buffer_pointer,
                metadata,
                args,
                f"RWStructuredBuffer.{method_name}",
                method_name == "IncrementCounter",
            )

        if method_name == "Load":
            diagnostic_name = "StructuredBuffer.Load"
            if len(args) < 1:
                self.refuse_lowering(f"{diagnostic_name} requires an index operand")
                return True, self.structured_buffer_default_value(metadata)
            if len(args) > 1:
                self.refuse_lowering(f"{diagnostic_name} accepts only an index operand")
                return True, self.structured_buffer_default_value(metadata)
            if metadata.get("writeonly"):
                self.refuse_lowering(f"{diagnostic_name} requires a readable buffer")
                return True, self.structured_buffer_default_value(metadata)
            if metadata.get("buffer_kind") not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
            }:
                self.refuse_lowering(
                    f"{diagnostic_name} requires a StructuredBuffer " "element"
                )
                return True, self.structured_buffer_default_value(metadata)

            index = self.process_expression(args[0])
            if index is None:
                self.refuse_lowering(f"{diagnostic_name} index could not be evaluated")
                return True, self.structured_buffer_default_value(metadata)
            element_pointer = self.structured_buffer_element_pointer(
                buffer_pointer, index
            )
            if element_pointer is None:
                self.refuse_lowering(
                    f"{diagnostic_name} requires a StructuredBuffer " "element"
                )
                return True, self.structured_buffer_default_value(metadata)

            element_type = self.variable_value_types[element_pointer.id]
            return True, self.load_from_variable(element_pointer, element_type)

        diagnostic_name = "RWStructuredBuffer.Store"
        if len(args) < 2:
            self.refuse_lowering(f"{diagnostic_name} requires index and value operands")
            return True, None
        if len(args) > 2:
            self.refuse_lowering(
                f"{diagnostic_name} accepts only index and value operands"
            )
            return True, None
        if metadata.get("readonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a writable buffer")
            return True, None
        if metadata.get("buffer_kind") != "RWStructuredBuffer":
            self.refuse_lowering(
                f"{diagnostic_name} requires an RWStructuredBuffer element"
            )
            return True, None

        index = self.process_expression(args[0])
        if index is None:
            self.refuse_lowering(f"{diagnostic_name} index could not be evaluated")
            return True, None
        element_pointer = self.structured_buffer_element_pointer(buffer_pointer, index)
        if element_pointer is None:
            self.refuse_lowering(
                f"{diagnostic_name} requires an RWStructuredBuffer element"
            )
            return True, None

        value = self.process_expression(args[1])
        if value is None:
            self.refuse_lowering(f"{diagnostic_name} value could not be evaluated")
            return True, None

        self.store_to_variable(element_pointer, value)
        return True, None

    def array_type_contains_element_type(
        self, array_type: Optional[SpirvId], target_type: Optional[SpirvId]
    ) -> bool:
        if array_type is None or target_type is None:
            return False

        element_type = self.array_element_type_from_type(array_type)
        while element_type is not None:
            if element_type.id == target_type.id:
                return True
            element_type = self.array_element_type_from_type(element_type)
        return False

    def byte_address_method_load_width(self, method_name: str) -> Optional[int]:
        return {"Load": 1, "Load2": 2, "Load3": 3, "Load4": 4}.get(method_name)

    def byte_address_method_store_width(self, method_name: str) -> Optional[int]:
        return {"Store": 1, "Store2": 2, "Store3": 3, "Store4": 4}.get(method_name)

    def byte_address_method_interlocked_info(self, method_name: str):
        return {
            "InterlockedAdd": {
                "opcode": "OpAtomicIAdd",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedAnd": {
                "opcode": "OpAtomicAnd",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedOr": {
                "opcode": "OpAtomicOr",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedXor": {
                "opcode": "OpAtomicXor",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedMin": {
                "opcode": "OpAtomicUMin",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedMax": {
                "opcode": "OpAtomicUMax",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedExchange": {
                "opcode": "OpAtomicExchange",
                "value_roles": ("value",),
                "min_args": 2,
                "max_args": 3,
                "required": "byte offset and value operands",
                "accepted": "byte offset, value, and optional original operands",
                "original_index": 2,
            },
            "InterlockedCompareExchange": {
                "opcode": "OpAtomicCompareExchange",
                "value_roles": ("compare", "value"),
                "min_args": 4,
                "max_args": 4,
                "required": "byte offset, compare, value, and original operands",
                "accepted": "byte offset, compare, value, and original operands",
                "original_index": 3,
            },
            "InterlockedCompareStore": {
                "opcode": "OpAtomicCompareExchange",
                "value_roles": ("compare", "value"),
                "min_args": 3,
                "max_args": 3,
                "required": "byte offset, compare, and value operands",
                "accepted": "byte offset, compare, and value operands",
                "original_index": None,
            },
        }.get(method_name)

    def byte_address_helper_load_width(self, function_name: str) -> Optional[int]:
        return {"buffer_load2": 2, "buffer_load3": 3, "buffer_load4": 4}.get(
            function_name
        )

    def byte_address_helper_store_width(self, function_name: str) -> Optional[int]:
        return {"buffer_store2": 2, "buffer_store3": 3, "buffer_store4": 4}.get(
            function_name
        )

    def byte_address_value_type(self, component_count: int) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        if component_count == 1:
            return uint_type
        return self.register_vector_type(uint_type, component_count)

    def byte_address_default_value(self, component_count: int) -> SpirvId:
        return self.default_value_for_type(
            self.byte_address_value_type(component_count)
        )

    def byte_address_interlocked_default_value(self) -> SpirvId:
        return self.default_value_for_type(self.register_primitive_type("uint"))

    def byte_address_interlocked_uint_operand(
        self, expr, diagnostic_name: str, role: str
    ) -> Optional[SpirvId]:
        value = self.process_expression(expr)
        if value is None:
            self.refuse_lowering(
                f"{diagnostic_name} {role} operand could not be evaluated"
            )
            return None

        value_type = self.value_types.get(
            value.id
        ) or self.find_registered_type_by_base(value.type.base_type)
        if value_type is None:
            self.refuse_lowering(
                f"{diagnostic_name} {role} operand type could not be " "determined"
            )
            return None

        if self.vector_type_info_from_type(value_type) is not None:
            self.refuse_lowering(
                f"{diagnostic_name} {role} operand must be a scalar integer"
            )
            return None
        if self.matrix_type_info_from_type(value_type) is not None:
            self.refuse_lowering(
                f"{diagnostic_name} {role} operand must be a scalar integer"
            )
            return None

        type_name = self.normalize_primitive_name(value_type.type.base_type)
        if type_name not in {"int", "uint"}:
            self.refuse_lowering(f"{diagnostic_name} {role} operand must be an integer")
            return None

        return self.convert_value_to_type(value, self.register_primitive_type("uint"))

    def byte_address_interlocked_original_pointer(
        self, expr, diagnostic_name: str
    ) -> Optional[SpirvId]:
        original_pointer = self.assignable_pointer_from_expression(expr)
        if original_pointer is None:
            self.refuse_lowering(
                f"{diagnostic_name} original operand must be an "
                "assignable scalar uint target"
            )
            return None

        original_type = self.pointer_pointee_type(original_pointer)
        if original_type is None:
            self.refuse_lowering(
                f"{diagnostic_name} original operand type could not be " "determined"
            )
            return None

        if self.vector_type_info_from_type(original_type) is not None:
            self.refuse_lowering(
                f"{diagnostic_name} original operand must be scalar uint"
            )
            return None
        if self.matrix_type_info_from_type(original_type) is not None:
            self.refuse_lowering(
                f"{diagnostic_name} original operand must be scalar uint"
            )
            return None

        type_name = self.normalize_primitive_name(original_type.type.base_type)
        if type_name != "uint":
            self.refuse_lowering(
                f"{diagnostic_name} original operand must be scalar uint"
            )
            return None

        return original_pointer

    def emit_byte_address_interlocked_atomic(
        self,
        opcode: str,
        target_pointer: SpirvId,
        value_operands: List[SpirvId],
    ) -> SpirvId:
        uint_type = self.register_primitive_type("uint")
        scope = self.spirv_scope_constant("Device")
        semantics = self.spirv_memory_semantics_constant()
        id_value = self.get_id()

        if opcode == "OpAtomicCompareExchange":
            compare_id, value_id = value_operands
            self.emit(
                f"%{id_value} = OpAtomicCompareExchange %{uint_type.id} "
                f"%{target_pointer.id} %{scope.id} %{semantics.id} %{semantics.id} "
                f"%{value_id.id} %{compare_id.id}"
            )
        else:
            value_id = value_operands[0]
            self.emit(
                f"%{id_value} = {opcode} %{uint_type.id} %{target_pointer.id} "
                f"%{scope.id} %{semantics.id} %{value_id.id}"
            )

        self.value_types[id_value] = uint_type
        return SpirvId(id_value, uint_type.type)

    def byte_address_buffer_element_index_from_value(
        self, byte_offset: SpirvId
    ) -> Optional[SpirvId]:
        offset_type_name = self.normalize_primitive_name(byte_offset.type.base_type)
        if offset_type_name not in {"int", "uint"}:
            self.refuse_lowering("ByteAddressBuffer byte offset must be an integer")
            return None

        uint_type = self.register_primitive_type("uint")
        byte_offset = self.convert_value_to_type(byte_offset, uint_type)
        element_size = self.register_constant(4, uint_type)
        return self.binary_operation("/", uint_type, byte_offset, element_size)

    def byte_address_buffer_element_index(self, offset_expr) -> Optional[SpirvId]:
        byte_offset = self.process_expression(offset_expr)
        if byte_offset is None:
            self.refuse_lowering("ByteAddressBuffer byte offset could not be evaluated")
            return None

        return self.byte_address_buffer_element_index_from_value(byte_offset)

    def byte_address_component_index(
        self, element_index: SpirvId, component_index: int
    ) -> SpirvId:
        if component_index == 0:
            return element_index

        uint_type = self.register_primitive_type("uint")
        component_offset = self.register_constant(component_index, uint_type)
        return self.binary_operation("+", uint_type, element_index, component_offset)

    def load_byte_address_buffer_value(
        self,
        buffer_pointer: SpirvId,
        element_index: SpirvId,
        component_count: int,
        diagnostic_name: str,
    ) -> SpirvId:
        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is not None and metadata.get("writeonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a readable buffer")
            return self.byte_address_default_value(component_count)

        uint_type = self.register_primitive_type("uint")
        components = []
        for component_index in range(component_count):
            index = self.byte_address_component_index(element_index, component_index)
            component = self.call_resource_function(
                "buffer_load", [buffer_pointer, index]
            )
            if component is None:
                component = self.default_value_for_type(uint_type)
            components.append(self.convert_value_to_type(component, uint_type))

        if component_count == 1:
            return components[0]

        return self.composite_construct(
            self.byte_address_value_type(component_count), components
        )

    def store_byte_address_buffer_value(
        self,
        buffer_pointer: SpirvId,
        element_index: SpirvId,
        value: SpirvId,
        component_count: int,
        diagnostic_name: str,
    ) -> None:
        uint_type = self.register_primitive_type("uint")
        if component_count == 1:
            value = self.convert_value_to_type(value, uint_type)
            self.call_resource_function(
                "buffer_store", [buffer_pointer, element_index, value]
            )
            return

        value_type = self.value_types.get(
            value.id
        ) or self.find_registered_type_by_base(value.type.base_type)
        value_type_name = (
            value_type.type.base_type
            if value_type is not None
            else value.type.base_type
        )
        vector_info = self.vector_component_type_and_count(value_type_name)
        if vector_info is None or vector_info[1] != component_count:
            self.refuse_lowering(
                f"{diagnostic_name} requires a uvec{component_count} value"
            )
            return
        if vector_info[0] not in {"int", "uint"}:
            self.refuse_lowering(f"{diagnostic_name} requires an integer vector value")
            return

        vector_type = self.byte_address_value_type(component_count)
        value = self.convert_value_to_type(value, vector_type)
        if not self.value_has_type(value, vector_type):
            self.refuse_lowering(
                f"{diagnostic_name} value could not be converted to "
                f"uvec{component_count}"
            )
            return

        for component_index in range(component_count):
            component = self.composite_extract(value, uint_type, component_index)
            index = self.byte_address_component_index(element_index, component_index)
            self.call_resource_function(
                "buffer_store", [buffer_pointer, index, component]
            )

    def call_byte_address_buffer_load_helper(
        self, function_name: str, args: List[SpirvId], component_count: int
    ) -> SpirvId:
        if len(args) < 2:
            self.refuse_lowering(
                f"{function_name} requires buffer and byte offset operands"
            )
            return self.byte_address_default_value(component_count)
        if len(args) > 2:
            self.refuse_lowering(
                f"{function_name} accepts only buffer and byte offset " "operands"
            )
            return self.byte_address_default_value(component_count)

        metadata = self.structured_buffer_metadata_for_pointer(args[0])
        if metadata is None or not metadata.get("byte_address"):
            self.refuse_lowering(
                f"{function_name} requires a ByteAddressBuffer operand"
            )
            return self.byte_address_default_value(component_count)

        element_index = self.byte_address_buffer_element_index_from_value(args[1])
        if element_index is None:
            return self.byte_address_default_value(component_count)

        return self.load_byte_address_buffer_value(
            args[0], element_index, component_count, function_name
        )

    def call_byte_address_buffer_store_helper(
        self, function_name: str, args: List[SpirvId], component_count: int
    ) -> None:
        if len(args) < 3:
            self.refuse_lowering(
                f"{function_name} requires buffer, byte offset, and " "value operands"
            )
            return None
        if len(args) > 3:
            self.refuse_lowering(
                f"{function_name} accepts only buffer, byte offset, and "
                "value operands"
            )
            return None

        metadata = self.structured_buffer_metadata_for_pointer(args[0])
        if metadata is None or not metadata.get("byte_address"):
            self.refuse_lowering(
                f"{function_name} requires an RWByteAddressBuffer operand"
            )
            return None
        if metadata.get("readonly"):
            self.refuse_lowering(f"{function_name} requires a writable buffer")
            return None

        element_index = self.byte_address_buffer_element_index_from_value(args[1])
        if element_index is None:
            return None

        self.store_byte_address_buffer_value(
            args[0], element_index, args[2], component_count, function_name
        )
        return None

    def process_byte_address_buffer_interlocked_call(
        self,
        buffer_pointer: SpirvId,
        metadata,
        method_name: str,
        args,
    ) -> SpirvId:
        diagnostic_name = f"RWByteAddressBuffer.{method_name}"
        info = self.byte_address_method_interlocked_info(method_name)
        if info is None:
            return self.byte_address_interlocked_default_value()

        if len(args) < info["min_args"]:
            self.refuse_lowering(f"{diagnostic_name} requires {info['required']}")
            return self.byte_address_interlocked_default_value()
        if len(args) > info["max_args"]:
            self.refuse_lowering(f"{diagnostic_name} accepts only {info['accepted']}")
            return self.byte_address_interlocked_default_value()
        if metadata.get("readonly") or metadata.get("writeonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a read-write buffer")
            return self.byte_address_interlocked_default_value()

        element_index = self.byte_address_buffer_element_index(args[0])
        if element_index is None:
            return self.byte_address_interlocked_default_value()

        value_operands = []
        for role_index, role in enumerate(info["value_roles"], start=1):
            value_operand = self.byte_address_interlocked_uint_operand(
                args[role_index], diagnostic_name, role
            )
            if value_operand is None:
                return self.byte_address_interlocked_default_value()
            value_operands.append(value_operand)

        original_pointer = None
        original_index = info["original_index"]
        if original_index is not None and len(args) > original_index:
            original_pointer = self.byte_address_interlocked_original_pointer(
                args[original_index], diagnostic_name
            )
            if original_pointer is None:
                return self.byte_address_interlocked_default_value()

        target_pointer = self.structured_buffer_element_pointer(
            buffer_pointer, element_index
        )
        if target_pointer is None:
            self.refuse_lowering(
                f"{diagnostic_name} requires a byte-address buffer element"
            )
            return self.byte_address_interlocked_default_value()

        atomic_result = self.emit_byte_address_interlocked_atomic(
            info["opcode"], target_pointer, value_operands
        )

        if original_pointer is not None:
            self.store_to_variable(original_pointer, atomic_result)

        return atomic_result

    def process_byte_address_buffer_method_call(
        self, expr: FunctionCallNode
    ) -> Tuple[bool, Optional[SpirvId]]:
        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(callee_expr, MemberAccessNode):
            return False, None

        method_name = getattr(callee_expr, "member", None)
        load_width = self.byte_address_method_load_width(method_name)
        store_width = self.byte_address_method_store_width(method_name)
        interlocked_info = self.byte_address_method_interlocked_info(method_name)
        if (
            load_width is None
            and store_width is None
            and interlocked_info is None
            and method_name != "GetDimensions"
        ):
            return False, None

        buffer_pointer = self.variable_pointer_from_expression(callee_expr.object)
        if buffer_pointer is None:
            return False, None

        metadata = self.structured_buffer_metadata_for_pointer(buffer_pointer)
        if metadata is None or not metadata.get("byte_address"):
            return False, None

        args = list(getattr(expr, "args", []) or [])
        if method_name == "GetDimensions":
            return True, self.process_structured_buffer_dimensions_method_call(
                buffer_pointer,
                metadata,
                args,
                "ByteAddressBuffer.GetDimensions",
            )

        if interlocked_info is not None:
            handled, result = process_byte_address_member_min_max(
                self, buffer_pointer, metadata, method_name, args
            )
            if handled:
                return True, result
            return True, self.process_byte_address_buffer_interlocked_call(
                buffer_pointer, metadata, method_name, args
            )

        if load_width is not None:
            diagnostic_name = f"ByteAddressBuffer.{method_name}"
            if len(args) < 1:
                self.refuse_lowering(f"{diagnostic_name} requires a byte offset")
                return True, self.byte_address_default_value(load_width)
            if len(args) > 1:
                self.refuse_lowering(f"{diagnostic_name} accepts only a byte offset")
                return True, self.byte_address_default_value(load_width)

            element_index = self.byte_address_buffer_element_index(args[0])
            if element_index is None:
                return True, self.byte_address_default_value(load_width)
            return True, self.load_byte_address_buffer_value(
                buffer_pointer, element_index, load_width, diagnostic_name
            )

        diagnostic_name = f"RWByteAddressBuffer.{method_name}"
        if len(args) < 2:
            self.refuse_lowering(
                f"{diagnostic_name} requires byte offset and value operands"
            )
            return True, None
        if len(args) > 2:
            self.refuse_lowering(
                f"{diagnostic_name} accepts only byte offset and value " "operands"
            )
            return True, None
        if metadata.get("readonly"):
            self.refuse_lowering(f"{diagnostic_name} requires a writable buffer")
            return True, None

        element_index = self.byte_address_buffer_element_index(args[0])
        if element_index is None:
            return True, None

        value = self.process_expression(args[1])
        if value is None:
            self.refuse_lowering(f"{diagnostic_name} value could not be evaluated")
            return True, None

        self.store_byte_address_buffer_value(
            buffer_pointer, element_index, value, store_width, diagnostic_name
        )
        return True, None

    def is_address_of_unary(self, expr) -> bool:
        # A prefix address-of (`&x`) unary operator.
        if getattr(expr, "is_postfix", False):
            return False
        op = getattr(expr, "op", getattr(expr, "operator", None))
        return op == "&"

    def pointer_points_to_indexable_array(self, pointer: SpirvId) -> bool:
        # True when the pointer's pointee is an array type we can index (a plain
        # local `shared[N]` / stack array), so `&arr[i]` can be offset-aliased.
        if pointer is None:
            return False
        pointee = self.variable_value_types.get(pointer.id)
        if pointee is None:
            return False
        return self.array_element_type_from_type(pointee) is not None

    def array_element_pointer_alias(self, array_expr, index_expr) -> Optional[SpirvId]:
        # Produce a pointer to `array_expr[index_expr]` and, for a plain local
        # array base, record an offset alias `(array_root, index)` on it so a
        # pointer aliasing this element resolves `[j]` / `+ j` to
        # `array_root[index + j]`.
        index = self.process_expression(index_expr)
        if index is None:
            return None
        access, _ = self.create_array_element_access(array_expr, index, index_expr)
        if access is None:
            return None
        if (
            self.storage_buffer_pointer_alias_key(access)
            not in self.storage_buffer_pointer_offset_aliases
        ):
            array_root = self.variable_pointer_from_expression(array_expr)
            if self.pointer_points_to_indexable_array(
                array_root
            ) or self.is_storage_buffer_pointer_arithmetic_base(array_root):
                self.record_storage_buffer_pointer_offset_alias(
                    array_root, access, index
                )
        return access

    def local_array_offset_element_pointer(
        self, root_pointer: SpirvId, index: SpirvId
    ) -> Optional[SpirvId]:
        # Access `root_pointer[index]` for a plain local array root (the generic
        # counterpart of structured_buffer_element_pointer, used to resolve an
        # offset alias whose root is a threadgroup/function array rather than a
        # storage buffer).
        pointee = self.variable_value_types.get(root_pointer.id)
        element_type = (
            self.array_element_type_from_type(pointee) if pointee is not None else None
        )
        if element_type is None:
            return None
        storage_class = root_pointer.type.storage_class or "Function"
        ptr_type = self.register_pointer_type(element_type, storage_class)
        access = self.access_chain(root_pointer, [index], ptr_type)
        self.variable_value_types[access.id] = element_type
        self.propagate_storage_buffer_access_metadata(root_pointer, access)
        return access

    def is_null_pointer_expression(self, expr) -> bool:
        if isinstance(expr, str):
            return expr == "nullptr"
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return getattr(expr, "name", None) == "nullptr"
        return False

    def unsupported_pointer_ternary_error(
        self,
        expr: TernaryOpNode,
        pointer: SpirvId,
    ) -> UnsupportedSPIRVFeatureError:
        if pointer.type.storage_class == "Workgroup":
            pointer_domain = "Workgroup"
            missing_capabilities = ("spirv.variable_pointers",)
        elif self.storage_buffer_access_metadata_for_pointer(pointer) is not None:
            pointer_domain = "storage-buffer"
            missing_capabilities = ("spirv.variable_pointers_storage_buffer",)
        else:
            pointer_domain = pointer.type.storage_class or "general"
            missing_capabilities = ("spirv.variable_pointers",)
        return UnsupportedSPIRVFeatureError(
            "variable-pointer-ternary",
            "SPIR-V pointer selection between nullptr and a "
            f"{pointer_domain} pointer requires target support for variable "
            "pointers; the active target profile does not enable that feature",
            missing_capabilities=missing_capabilities,
            source_location=getattr(expr, "source_location", None),
        )

    def unsupported_null_pointer_argument_error(
        self,
        expr: TernaryOpNode,
    ) -> UnsupportedSPIRVFeatureError:
        return UnsupportedSPIRVFeatureError(
            "null-pointer-argument",
            "SPIR-V helper inlining cannot bind a statically selected nullptr "
            "argument without proving that the pointer parameter is unreachable",
            missing_capabilities=("spirv.null_pointer_argument",),
            source_location=getattr(expr, "source_location", None),
        )

    def pointer_ternary_statically_selects_null(self, expr) -> bool:
        if not isinstance(expr, TernaryOpNode):
            return False
        literal_condition = evaluate_literal_int_expression(
            expr.condition,
            self.active_literal_int_constants(),
        )
        if literal_condition is None:
            return False
        selected = expr.true_expr if literal_condition != 0 else expr.false_expr
        return self.is_null_pointer_expression(selected)

    def side_effect_free_call_argument(self, expression) -> bool:
        if expression is None or isinstance(
            expression, (bool, int, float, str, ConstantNode, LiteralNode)
        ):
            return True
        if isinstance(expression, (IdentifierNode, VariableNode)):
            return True
        if isinstance(expression, MemberAccessNode):
            return self.side_effect_free_call_argument(expression.object)
        if isinstance(expression, ArrayAccessNode):
            return self.side_effect_free_call_argument(
                expression.array
            ) and self.side_effect_free_call_argument(expression.index)
        if isinstance(expression, BinaryOpNode):
            return self.side_effect_free_call_argument(
                expression.left
            ) and self.side_effect_free_call_argument(expression.right)
        if isinstance(expression, UnaryOpNode):
            return expression.op not in {"++", "--"} and (
                self.side_effect_free_call_argument(expression.operand)
            )
        if isinstance(expression, TernaryOpNode):
            return all(
                self.side_effect_free_call_argument(part)
                for part in (
                    expression.condition,
                    expression.true_expr,
                    expression.false_expr,
                )
            )
        if isinstance(expression, ArrayLiteralNode):
            return all(
                self.side_effect_free_call_argument(element)
                for element in expression.elements
            )
        if isinstance(expression, ConstructorNode):
            return all(
                self.side_effect_free_call_argument(argument)
                for argument in list(expression.arguments) + list(
                    expression.named_arguments.values()
                )
            )
        if isinstance(expression, FunctionCallNode):
            callee_name = self.function_call_name(expression)
            if not isinstance(callee_name, str) or self.has_function_reference(
                callee_name
            ):
                return False
            normalized_name = self.normalize_primitive_name(callee_name)
            scalar_constructor = normalized_name in {
                "bool",
                "float",
                "double",
                self.BFLOAT16_TYPE_NAME,
                *self.INTEGER_TYPE_NAMES,
            }
            vector_constructor = (
                self.vector_component_type_and_count(callee_name) is not None
            )
            if not scalar_constructor and not vector_constructor:
                return False
            return all(
                self.side_effect_free_call_argument(argument)
                for argument in list(getattr(expression, "args", []) or [])
            )
        return False

    def variable_pointer_from_expression(self, expr) -> Optional[SpirvId]:
        if isinstance(expr, PointerReinterpretNode):
            return self.variable_pointer_from_expression(expr.expression)
        if isinstance(expr, IdentifierNode):
            name = expr.name
        elif isinstance(expr, VariableNode):
            name = expr.name
        elif isinstance(expr, str):
            name = expr
        elif isinstance(expr, ArrayAccessNode):
            index = evaluate_subscript_index(self, expr.index)
            if index is None:
                return None
            access, _ = self.create_array_element_access(expr.array, index, expr.index)
            return access
        elif (
            isinstance(expr, UnaryOpNode)
            and self.is_address_of_unary(expr)
            and isinstance(expr.operand, ArrayAccessNode)
        ):
            # `&arr[i]` yields a POINTER to the element. For a plain local array
            # (threadgroup/function `shared[...]`) we record an offset alias on
            # the element pointer so a pointer bound to it resolves `ptr[j]` /
            # `ptr + j` to `arr[i + j]` instead of indexing a squashed scalar.
            # Only the array-element form is handled here; `&scalar` and other
            # address-of forms keep their prior behavior (fall through to None)
            # so thread-local pointer arguments are unaffected.
            return self.array_element_pointer_alias(
                expr.operand.array, expr.operand.index
            )
        elif isinstance(expr, MemberAccessNode):
            base_pointer = self.variable_pointer_from_expression(expr.object)
            if base_pointer is None:
                return None

            return self.create_member_access_pointer(base_pointer, expr.member)
        elif isinstance(expr, BinaryOpNode) and expr.op == "+":
            left_pointer = self.variable_pointer_from_expression(expr.left)
            if (
                left_pointer is not None
                and self.is_storage_buffer_pointer_arithmetic_base(left_pointer)
            ):
                index = self.process_expression(expr.right)
                if index is None:
                    return None
                access, _ = self.create_array_element_access(
                    expr.left, index, expr.right
                )
                if access is not None:
                    self.record_storage_buffer_pointer_offset_alias(
                        left_pointer,
                        access,
                        index,
                    )
                return access

            right_pointer = self.variable_pointer_from_expression(expr.right)
            if (
                right_pointer is not None
                and self.is_storage_buffer_pointer_arithmetic_base(right_pointer)
            ):
                index = self.process_expression(expr.left)
                if index is None:
                    return None
                access, _ = self.create_array_element_access(
                    expr.right, index, expr.left
                )
                if access is not None:
                    self.record_storage_buffer_pointer_offset_alias(
                        right_pointer, access, index
                    )
                return access
            return None
        elif isinstance(expr, TernaryOpNode):
            literal_condition = evaluate_literal_int_expression(
                expr.condition,
                self.active_literal_int_constants(),
            )
            if literal_condition is not None:
                selected = expr.true_expr if literal_condition != 0 else expr.false_expr
                if self.is_null_pointer_expression(selected):
                    raise self.unsupported_null_pointer_argument_error(expr)
                return self.variable_pointer_from_expression(selected)

            true_is_null = self.is_null_pointer_expression(expr.true_expr)
            false_is_null = self.is_null_pointer_expression(expr.false_expr)
            if true_is_null == false_is_null:
                return None
            pointer_expr = expr.false_expr if true_is_null else expr.true_expr
            pointer = self.variable_pointer_from_expression(pointer_expr)
            if pointer is None:
                return None
            raise self.unsupported_pointer_ternary_error(expr, pointer)
        else:
            return None

        pointer = (
            self.local_variables.get(name)
            or self.resolve_global_variable(name)
            or self.cbuffer_member_pointer(name)
            or self.ensure_tessellation_patch_constant_input(name)
            or self.ensure_compute_builtin(name)
            or self.ensure_stage_builtin(name)
        )
        self.mark_interface_variable_if_needed(pointer)
        wrapped_pointer = self.uniform_block_wrapped_member_pointer(pointer)
        if wrapped_pointer is not None:
            return wrapped_pointer
        return pointer

    def array_element_type_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        array_info = self.array_type_info_from_type(array_type)
        if array_info is not None:
            return array_info[0]

        vector_info = self.vector_type_info_from_type(array_type)
        if vector_info is not None:
            return vector_info[0]

        matrix_info = self.matrix_type_info_from_type(array_type)
        if matrix_info is not None:
            return matrix_info[0]

        return None

    def array_type_info_from_type(self, array_type: Optional[SpirvId]):
        if array_type is None:
            return None

        for (element_type_id, size), arr_type_id in self.array_types.items():
            if arr_type_id.id == array_type.id:
                return self.find_registered_type_by_id(element_type_id), size
        for (element_type_id, size, _), arr_type_id in self.layout_array_types.items():
            if arr_type_id.id == array_type.id:
                return self.find_registered_type_by_id(element_type_id), size

        return None

    def type_contains_runtime_array(
        self, type_id: Optional[SpirvId], seen: Optional[set] = None
    ) -> bool:
        if type_id is None:
            return False
        if seen is None:
            seen = set()
        if type_id.id in seen:
            return False
        seen.add(type_id.id)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            return size is None or self.type_contains_runtime_array(element_type, seen)

        struct_members = self.current_struct_members.get(type_id.type.base_type)
        if struct_members is not None:
            return any(
                self.type_contains_runtime_array(member_type, seen)
                for member_type, _ in struct_members
            )

        return False

    def runtime_array_aggregate_fallback(self, reason: str) -> SpirvId:
        self.refuse_lowering(f"{reason}")
        return self.register_constant(0.0, self.register_primitive_type("float"))

    def vector_type_info_from_type(self, vector_type: Optional[SpirvId]):
        if vector_type is None:
            return None

        for (component_type_id, count), vec_type_id in self.vector_types.items():
            if vec_type_id.id == vector_type.id:
                return self.find_registered_type_by_id(component_type_id), count

        return None

    def matrix_type_info_from_type(self, matrix_type: Optional[SpirvId]):
        if matrix_type is None:
            return None

        for (column_type_id, count), mat_type_id in self.matrix_types.items():
            if mat_type_id.id == matrix_type.id:
                return self.find_registered_type_by_id(column_type_id), count

        return None

    def default_value_for_type(self, type_id: SpirvId) -> SpirvId:
        primitive_name = self.normalize_primitive_name(type_id.type.base_type)
        if primitive_name in {"float", "double"}:
            return self.register_constant(0.0, type_id)
        if primitive_name == self.BFLOAT16_TYPE_NAME:
            return self.register_constant(0, type_id)
        if primitive_name in self.INTEGER_TYPE_NAMES:
            return self.register_constant(0, type_id)
        if primitive_name == "bool":
            return self.register_constant(False, type_id)
        if self.type_contains_runtime_array(type_id):
            return self.runtime_array_aggregate_fallback(
                "runtime-array aggregate default values cannot be materialized "
                "in SPIR-V"
            )
        if self.type_contains_specialization_sized_array(type_id):
            return self.register_undefined_value(type_id)

        vector_info = self.vector_type_info_from_type(type_id)
        if vector_info is not None:
            component_type, count = vector_info
            components = [
                self.default_value_for_type(component_type) for _ in range(count)
            ]
            return self.register_vector_constant(type_id, components)

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is not None:
            column_type, count = matrix_info
            columns = [self.default_value_for_type(column_type) for _ in range(count)]
            return self.register_composite_constant(type_id, columns)

        array_info = self.array_type_info_from_type(type_id)
        if array_info is not None:
            element_type, size = array_info
            elements = [
                self.default_value_for_type(element_type)
                for _ in range(self.array_length_default_value(size, fallback=0))
            ]
            return self.register_composite_constant(type_id, elements)

        members = self.current_struct_members.get(type_id.type.base_type)
        if members is not None:
            values = [
                self.default_value_for_type(member_type) for member_type, _ in members
            ]
            return self.register_composite_constant(type_id, values)

        return self.register_constant(0.0, self.register_primitive_type("float"))
