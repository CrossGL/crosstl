"""Structured and typed buffers in the Slang generator: resource predicates, atomics, loads, stores, counters and member calls."""

from ..ast import (
    ArrayAccessNode,
    MemberAccessNode,
)
from .array_utils import split_array_type_suffix
from .slang_structured_atomics import structured_buffer_atomic_location


class SlangStructuredBuffersMixin:
    """Structured-buffer members of ``SlangCodeGen``."""

    def structured_buffer_resource_type(self, buffer_arg):
        buffer_name = self.get_expression_name(buffer_arg)
        if buffer_name:
            tracked_type = self.buffer_resource_types.get(buffer_name)
            if tracked_type is not None:
                return tracked_type

        buffer_type = self.get_expression_type(buffer_arg)
        if buffer_type is None:
            buffer_type = self.expression_result_type(buffer_arg)
        mapped_type = self.map_resource_type_with_format(buffer_type)
        return self.resource_base_type(mapped_type)

    def buffer_resource_access(self, buffer_arg):
        buffer_name = self.get_expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.buffer_resource_accesses.get(buffer_name)

    def is_buffer_resource_type(self, buffer_type):
        return (
            self.is_byte_address_buffer_resource_type(buffer_type)
            or self.is_structured_buffer_resource_type(buffer_type)
            or self.is_append_structured_buffer_resource_type(buffer_type)
            or self.is_consume_structured_buffer_resource_type(buffer_type)
        )

    def is_byte_address_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return buffer_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}

    def is_writable_byte_address_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return buffer_type == "RWByteAddressBuffer"

    def is_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            ("StructuredBuffer<", "RWStructuredBuffer<")
        )

    def is_writable_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            "RWStructuredBuffer<"
        )

    def is_append_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            "AppendStructuredBuffer<"
        )

    def is_consume_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            "ConsumeStructuredBuffer<"
        )

    def structured_buffer_element_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        if (
            not isinstance(buffer_type, str)
            or "<" not in buffer_type
            or not buffer_type.endswith(">")
        ):
            return None
        element_type = buffer_type[buffer_type.find("<") + 1 : -1].strip()
        if not element_type:
            return None
        return self.convert_type(element_type)

    def structured_buffer_atomic_operations(self):
        return {
            "atomicAdd": ("InterlockedAdd", 1),
            "atomicMin": ("InterlockedMin", 1),
            "atomicMax": ("InterlockedMax", 1),
            "atomicAnd": ("InterlockedAnd", 1),
            "atomicOr": ("InterlockedOr", 1),
            "atomicXor": ("InterlockedXor", 1),
            "atomicExchange": ("InterlockedExchange", 1),
            "atomicCompSwap": ("InterlockedCompareExchange", 2),
            "atomicCompareExchange": ("InterlockedCompareExchange", 2),
        }

    def structured_buffer_atomic_target_resource_type(self, target):
        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            array_type = self.structured_buffer_resource_type(array_expr)
            if self.is_structured_buffer_resource_type(array_type):
                return array_type
            return self.structured_buffer_atomic_target_resource_type(array_expr)
        if isinstance(target, MemberAccessNode):
            object_expr = getattr(
                target, "object", getattr(target, "object_expr", None)
            )
            return self.structured_buffer_atomic_target_resource_type(object_expr)
        return self.structured_buffer_resource_type(target)

    def unique_expression_temp_name(self, base_name):
        candidate = base_name
        suffix = 1
        used_names = set(self.user_symbol_names) | set(self.expression_temp_names)
        used_names.update(name for name in self.variable_types if name)
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.expression_temp_names.add(candidate)
        return candidate

    def structured_buffer_atomic_value_expression(
        self, func_name, intrinsic, target, value_args, element_type
    ):
        if not self.expression_prelude_active():
            if self.atomic_value_context_stack:
                context = self.atomic_value_context_stack[-1]
                return self.unsupported_structured_buffer_call(
                    func_name,
                    "expression-valued atomic cannot be used in "
                    f"{context}; use explicit original output argument",
                    element_type,
                )
            return self.unsupported_structured_buffer_call(
                func_name,
                "requires target, value, and original output outside "
                "statement expression context",
                element_type,
            )

        original_name = self.unique_expression_temp_name(f"cgl_{func_name}_original")
        target_expr = self.generate_expression(target)
        value_exprs = [
            self.generate_expression_with_expected(value_arg, element_type)
            for value_arg in value_args
        ]
        self.add_expression_prelude(
            [
                f"{element_type} {original_name};",
                f"{intrinsic}({', '.join([target_expr, *value_exprs, original_name])});",
            ],
            result_name=original_name,
        )
        return original_name

    def structured_buffer_atomic_expression(
        self, func_name, args, statement_context=False
    ):
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not args:
            return None

        diagnostic = self.unsupported_structured_buffer_call
        target = args[0]
        buffer_type = self.structured_buffer_atomic_target_resource_type(target)
        if not self.is_structured_buffer_resource_type(buffer_type):
            return None

        buffer_name, element_type = structured_buffer_atomic_location(
            self, target, buffer_type
        )
        element_type = element_type or "uint"
        result_type = element_type if element_type in {"int", "uint"} else "uint"
        if self.buffer_resource_accesses.get(buffer_name) == "readonly":
            return diagnostic(
                func_name,
                "requires writable structured buffer resource",
                result_type,
            )
        if not self.is_writable_structured_buffer_resource_type(buffer_type):
            return diagnostic(
                func_name,
                "requires RWStructuredBuffer resource",
                result_type,
            )
        if element_type not in {"int", "uint"}:
            return diagnostic(
                func_name,
                "requires scalar int or uint RWStructuredBuffer element",
                result_type,
            )

        intrinsic, value_arg_count = operation
        expression_args = 1 + value_arg_count
        explicit_result_args = expression_args + 1
        if len(args) == expression_args:
            target_reason = self.atomic_result_expected_type_unsupported_reason(
                func_name, result_type
            )
            if target_reason:
                return diagnostic(
                    func_name,
                    target_reason,
                    self.atomic_result_diagnostic_fallback_type(result_type),
                )
            return self.structured_buffer_atomic_value_expression(
                func_name, intrinsic, target, args[1:], element_type
            )
        if len(args) != explicit_result_args:
            required_shape = (
                "target, compare, value, and original output"
                if value_arg_count == 2
                else "target, value, and original output"
            )
            return diagnostic(
                func_name,
                f"requires {required_shape}",
                result_type,
            )

        if not statement_context:
            return diagnostic(
                func_name,
                "explicit original output atomic cannot be used as a value expression",
                self.atomic_result_diagnostic_fallback_type(result_type),
            )

        target_expr = self.generate_expression(target)
        value_exprs = [
            self.generate_expression_with_expected(value_arg, element_type)
            for value_arg in args[1 : 1 + value_arg_count]
        ]
        original_expr = self.generate_expression(args[-1])
        return f"{intrinsic}({', '.join([target_expr, *value_exprs, original_expr])})"

    def unsupported_structured_buffer_call(self, operation, reason, result_type=None):
        placeholder = (
            "" if result_type is None else self.zero_value_for_type(result_type)
        )
        return self.refuse_lowering(
            f"Slang structured buffer: {operation} {reason}", placeholder
        )

    def unsupported_byte_address_buffer_call(self, operation, reason, result_type=None):
        placeholder = (
            "" if result_type is None else self.zero_value_for_type(result_type)
        )
        return self.refuse_lowering(
            f"Slang byte-address buffer: {operation} {reason}", placeholder
        )

    def atomic_result_expected_type(self):
        expected_type = self.convert_type(self.current_expression_expected_type)
        if not expected_type or expected_type == "auto":
            return None
        if self.is_scalar_value_type(expected_type) or self.is_vector_value_type(
            expected_type
        ):
            return expected_type
        return None

    def atomic_result_expected_type_unsupported_reason(self, operation, result_type):
        expected_type = self.atomic_result_expected_type()
        result_type = self.convert_type(result_type)
        if not expected_type or not result_type:
            return None
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def atomic_result_diagnostic_fallback_type(self, result_type):
        return self.atomic_result_expected_type() or result_type

    def byte_address_load_expected_type(self):
        expected_type = self.convert_type(self.current_expression_expected_type)
        if not expected_type or expected_type in {"auto", "void"}:
            return None
        return expected_type

    def byte_address_load_expected_type_unsupported_reason(
        self, operation, result_type
    ):
        expected_type = self.byte_address_load_expected_type()
        result_type = self.convert_type(result_type)
        if not expected_type or not result_type:
            return None
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def byte_address_load_diagnostic_fallback_type(self, result_type):
        return self.byte_address_load_expected_type() or result_type

    def zero_value_for_type(self, type_name):
        type_name = self.convert_type(type_name)
        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            return self.zero_array_value(base_type, array_suffix)
        if type_name == "bool":
            return "false"
        if type_name == "uint":
            return "0u"
        if type_name in {"int", "float", "double"}:
            return "0"
        if self.is_vector_value_type(type_name):
            component_zero = self.vector_zero_value(type_name)
            return f"{type_name}({component_zero})"
        if self.is_matrix_value_type(type_name):
            return f"{type_name}(0.0)"
        if type_name in self.user_struct_names:
            return f"{type_name}()"
        return "0"

    def zero_array_value(self, base_type, array_suffix):
        dimensions = self.array_suffix_literal_dimensions(array_suffix)
        if not dimensions:
            return "0"
        return self.zero_array_literal(base_type, dimensions)

    def array_suffix_literal_dimensions(self, array_suffix):
        dimensions = []
        remaining = array_suffix
        while remaining.startswith("["):
            closing_bracket = remaining.find("]")
            if closing_bracket < 0:
                return []
            size = remaining[1:closing_bracket].strip()
            if not size.isdigit():
                return []
            dimensions.append(int(size))
            remaining = remaining[closing_bracket + 1 :]
        return dimensions if not remaining else []

    def zero_array_literal(self, base_type, dimensions):
        if not dimensions:
            return self.zero_value_for_type(base_type)
        element = self.zero_array_literal(base_type, dimensions[1:])
        return "{" + ", ".join(element for _ in range(dimensions[0])) + "}"

    def buffer_load_expression(self, args):
        if len(args) < 2:
            return self.unsupported_structured_buffer_call(
                "buffer_load", "requires buffer and index arguments", "uint"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        element_type = self.structured_buffer_element_type(buffer_type) or "uint"
        if self.buffer_resource_access(args[0]) == "writeonly":
            if self.is_byte_address_buffer_resource_type(buffer_type):
                return self.unsupported_byte_address_buffer_call(
                    "buffer_load",
                    "requires readable byte-address buffer resource",
                    "uint",
                )
            return self.unsupported_structured_buffer_call(
                "buffer_load",
                "requires readable structured buffer resource",
                element_type,
            )
        if self.is_byte_address_buffer_resource_type(buffer_type):
            target_reason = self.byte_address_load_expected_type_unsupported_reason(
                "buffer_load", "uint"
            )
            if target_reason is not None:
                return self.unsupported_byte_address_buffer_call(
                    "buffer_load",
                    target_reason,
                    self.byte_address_load_diagnostic_fallback_type("uint"),
                )
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            return f"{buffer}.Load({index})"

        if not self.is_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_load",
                "requires StructuredBuffer or RWStructuredBuffer resource",
                element_type,
            )

        buffer = self.generate_expression(args[0])
        index = self.generate_expression(args[1])
        return f"{buffer}.Load({index})"

    def buffer_store_expression(self, args, statement_context=False):
        if len(args) < 3:
            return self.unsupported_structured_buffer_call(
                "buffer_store", "requires buffer, index, and value arguments"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        if self.is_byte_address_buffer_resource_type(buffer_type):
            if self.buffer_resource_access(args[0]) == "readonly":
                return self.unsupported_byte_address_buffer_call(
                    "buffer_store", "requires writable byte-address buffer resource"
                )
            if not self.is_writable_byte_address_buffer_resource_type(buffer_type):
                return self.unsupported_byte_address_buffer_call(
                    "buffer_store", "requires RWByteAddressBuffer resource"
                )
            if not statement_context:
                return self.unsupported_byte_address_buffer_call(
                    "buffer_store",
                    "cannot be used as a value expression",
                    self.atomic_result_diagnostic_fallback_type("uint"),
                )
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            return f"{buffer}.Store({index}, {value})"

        if self.buffer_resource_access(args[0]) == "readonly":
            return self.unsupported_structured_buffer_call(
                "buffer_store", "requires writable structured buffer resource"
            )

        if not self.is_writable_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_store", "requires RWStructuredBuffer resource"
            )

        # Slang's RWStructuredBuffer has no Store method (slangc E30027): its elements are written by subscript.
        buffer = self.generate_expression(args[0])
        index = self.generate_expression(args[1])
        value = self.generate_expression(args[2])
        store = f"{buffer}[{index}] = {value}"
        return store if statement_context else f"({store})"

    def buffer_append_expression(self, args):
        if len(args) < 2:
            return self.unsupported_structured_buffer_call(
                "buffer_append", "requires buffer and value arguments"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        if self.buffer_resource_access(args[0]) == "readonly":
            return self.unsupported_structured_buffer_call(
                "buffer_append", "requires writable structured buffer resource"
            )
        if not self.is_append_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_append", "requires AppendStructuredBuffer resource"
            )

        buffer = self.generate_expression(args[0])
        value = self.generate_expression(args[1])
        return f"{buffer}.Append({value})"

    def buffer_consume_expression(self, args):
        if not args:
            return self.unsupported_structured_buffer_call(
                "buffer_consume", "requires a buffer argument", "uint"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        element_type = self.structured_buffer_element_type(buffer_type) or "uint"
        if self.buffer_resource_access(args[0]) == "writeonly":
            return self.unsupported_structured_buffer_call(
                "buffer_consume",
                "requires readable structured buffer resource",
                element_type,
            )
        if not self.is_consume_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_consume",
                "requires ConsumeStructuredBuffer resource",
                element_type,
            )

        buffer = self.generate_expression(args[0])
        return f"{buffer}.Consume()"

    def buffer_counter_expression(self, func_name, args):
        """``IncrementCounter()`` / ``DecrementCounter()``, which Slang keeps on an ``RWStructuredBuffer``."""
        if len(args) != 1:
            return self.unsupported_structured_buffer_call(
                func_name, "takes exactly one buffer operand", "uint"
            )
        buffer_type = self.resource_base_type(
            self.structured_buffer_resource_type(args[0])
        )
        if not (
            isinstance(buffer_type, str)
            and buffer_type.startswith("RWStructuredBuffer<")
        ) or self.buffer_resource_access(args[0]) in {"readonly", "writeonly"}:
            return self.unsupported_structured_buffer_call(
                func_name, "requires a read-write RWStructuredBuffer resource", "uint"
            )
        method = (
            "IncrementCounter"
            if func_name == "buffer_increment_counter"
            else "DecrementCounter"
        )
        return f"{self.generate_expression(args[0])}.{method}()"

    def buffer_dimensions_expression(self, args):
        if not args:
            return self.unsupported_structured_buffer_call(
                "buffer_dimensions", "requires a buffer argument", "uint"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        if self.is_byte_address_buffer_resource_type(buffer_type):
            buffer = self.generate_expression(args[0])
            if len(args) > 2:
                return self.unsupported_byte_address_buffer_call(
                    "buffer_dimensions",
                    "writes only the byte size",
                    "uint",
                )
            if len(args) == 2:
                return f"{buffer}.GetDimensions({self.generate_expression(args[1])})"

            helper_name = self.buffer_dimensions_helper_name(buffer_type)
            return f"{helper_name}({buffer})"

        if not self.is_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_dimensions",
                "requires StructuredBuffer or RWStructuredBuffer resource",
                "uint",
            )

        buffer = self.generate_expression(args[0])
        if len(args) > 3:
            return self.unsupported_structured_buffer_call(
                "buffer_dimensions",
                "writes only the element count and stride of a structured buffer",
                "uint",
            )
        if len(args) == 3:
            dimensions = ", ".join(self.generate_expression(arg) for arg in args[1:])
            return f"{buffer}.GetDimensions({dimensions})"

        # A structured buffer's GetDimensions always writes the stride as well, so a lone output takes the count.
        helper_name = self.buffer_dimensions_helper_name(buffer_type)
        if len(args) == 2:
            return f"{self.generate_expression(args[1])} = {helper_name}({buffer})"
        return f"{helper_name}({buffer})"

    def buffer_dimensions_helper_name(self, buffer_type):
        helper_name = self.helper_function_name(
            f"cgl_bufferDimensions_{self.resource_helper_type_suffix(buffer_type)}"
        )
        self.register_helper_function(
            helper_name,
            self.build_buffer_dimensions_helper(helper_name, buffer_type),
        )
        return helper_name

    def build_buffer_dimensions_helper(self, helper_name, buffer_type):
        # A byte-address buffer reports its size in one argument; a structured buffer reports its element count
        # and stride, and Slang has no overload that omits the stride.
        if self.is_byte_address_buffer_resource_type(buffer_type):
            query = "    uint count;\n    buffer.GetDimensions(count);\n"
        else:
            query = (
                "    uint count;\n"
                "    uint stride;\n"
                "    buffer.GetDimensions(count, stride);\n"
            )
        return (
            f"uint {helper_name}({buffer_type} buffer)\n"
            "{\n"
            f"{query}"
            "    return count;\n"
            "}"
        )

    def byte_address_member_call_result_type(self, func_expr):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        if not self.is_byte_address_buffer_resource_type(
            self.structured_buffer_resource_type(receiver)
        ):
            return None
        member = str(getattr(func_expr, "member", ""))
        return {
            "Load": "uint",
            "Load2": "uint2",
            "Load3": "uint3",
            "Load4": "uint4",
        }.get(member)

    def byte_address_interlocked_members(self):
        return {
            "InterlockedAdd",
            "InterlockedAnd",
            "InterlockedCompareExchange",
            "InterlockedCompareStore",
            "InterlockedExchange",
            "InterlockedMax",
            "InterlockedMin",
            "InterlockedOr",
            "InterlockedXor",
        }

    def structured_buffer_member_call_result_type(self, func_expr):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        member = str(getattr(func_expr, "member", ""))
        if member != "Consume":
            return None
        receiver_type = self.structured_buffer_resource_type(receiver)
        if not (
            self.is_consume_structured_buffer_resource_type(receiver_type)
            or self.is_structured_buffer_resource_type(receiver_type)
            or self.is_append_structured_buffer_resource_type(receiver_type)
        ):
            return None
        return self.structured_buffer_element_type(receiver_type) or "uint"

    def generate_resource_member_call(self, func_expr, args, statement_context=False):
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        receiver_type = self.structured_buffer_resource_type(receiver)

        member = str(getattr(func_expr, "member", ""))
        if self.is_byte_address_buffer_resource_type(receiver_type) and member in {
            "Load",
            "Load2",
            "Load3",
            "Load4",
        }:
            if self.buffer_resource_access(receiver) == "writeonly":
                result_type = self.byte_address_member_call_result_type(func_expr)
                return self.unsupported_byte_address_buffer_call(
                    member,
                    "requires readable byte-address buffer receiver",
                    result_type or "uint",
                )
            result_type = self.byte_address_member_call_result_type(func_expr) or "uint"
            target_reason = self.byte_address_load_expected_type_unsupported_reason(
                member, result_type
            )
            if target_reason is not None:
                return self.unsupported_byte_address_buffer_call(
                    member,
                    target_reason,
                    self.byte_address_load_diagnostic_fallback_type(result_type),
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            if member == "Load3" and len(args) == 1:
                return self.slang_three_component_access.load(receiver_expr, args_expr)
            return f"{receiver_expr}.{member}({args_expr})"

        if self.is_byte_address_buffer_resource_type(receiver_type) and member in {
            "Store",
            "Store2",
            "Store3",
            "Store4",
        }:
            if self.buffer_resource_access(receiver) == "readonly":
                return self.unsupported_byte_address_buffer_call(
                    member, "requires writable byte-address buffer receiver"
                )
            if not self.is_writable_byte_address_buffer_resource_type(receiver_type):
                return self.unsupported_byte_address_buffer_call(
                    member, "requires RWByteAddressBuffer receiver"
                )
            if not statement_context:
                return self.unsupported_byte_address_buffer_call(
                    member,
                    "cannot be used as a value expression",
                    self.atomic_result_diagnostic_fallback_type("uint"),
                )
            receiver_expr = self.generate_expression(receiver)
            if member == "Store3" and len(args) == 2:
                offset, value = (self.generate_expression(arg) for arg in args)
                return self.slang_three_component_access.store(
                    receiver_expr, offset, value
                )
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if (
            self.is_byte_address_buffer_resource_type(receiver_type)
            and member in self.byte_address_interlocked_members()
        ):
            if self.buffer_resource_access(receiver) == "readonly":
                return self.unsupported_byte_address_buffer_call(
                    member, "requires writable byte-address buffer receiver"
                )
            if not self.is_writable_byte_address_buffer_resource_type(receiver_type):
                return self.unsupported_byte_address_buffer_call(
                    member, "requires RWByteAddressBuffer receiver"
                )
            if not statement_context:
                return self.unsupported_byte_address_buffer_call(
                    member,
                    "cannot be used as a value expression",
                    self.atomic_result_diagnostic_fallback_type("uint"),
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if member in {"Append", "Consume"} and not (
            self.is_structured_buffer_resource_type(receiver_type)
            or self.is_append_structured_buffer_resource_type(receiver_type)
            or self.is_consume_structured_buffer_resource_type(receiver_type)
        ):
            return None

        if member == "Append":
            if len(args) < 1:
                return self.unsupported_structured_buffer_call(
                    "Append", "requires a value argument"
                )
            if self.buffer_resource_access(receiver) == "readonly":
                return self.unsupported_structured_buffer_call(
                    "Append", "requires writable structured buffer receiver"
                )
            if not self.is_append_structured_buffer_resource_type(receiver_type):
                return self.unsupported_structured_buffer_call(
                    "Append", "requires AppendStructuredBuffer receiver"
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if member == "Consume":
            result_type = self.structured_buffer_member_call_result_type(func_expr)
            if args:
                return self.unsupported_structured_buffer_call(
                    "Consume", "requires no arguments", result_type or "uint"
                )
            if self.buffer_resource_access(receiver) == "writeonly":
                return self.unsupported_structured_buffer_call(
                    "Consume",
                    "requires readable structured buffer receiver",
                    result_type or "uint",
                )
            if not self.is_consume_structured_buffer_resource_type(receiver_type):
                return self.unsupported_structured_buffer_call(
                    "Consume",
                    "requires ConsumeStructuredBuffer receiver",
                    result_type or "uint",
                )
            receiver_expr = self.generate_expression(receiver)
            return f"{receiver_expr}.{member}()"

        return None
