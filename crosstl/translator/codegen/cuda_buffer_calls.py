"""Structured-buffer, byte-address-buffer and atomic call lowering of the CUDA generator."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    UnaryOpNode,
)


class CudaBufferCallMixin:
    """Buffer call members of ``CudaCodeGen``."""

    def struct_constructor_arguments(self, struct_name, raw_args, args):
        """Expand struct constructors with resource-member metadata sidecars."""
        metadata_members = self.struct_query_metadata_members.get(struct_name, set())
        if not metadata_members:
            return args

        expanded_args = []
        member_types = list(self.struct_member_types.get(struct_name, {}).items())
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(raw_args) or index >= len(member_types):
                continue
            member_name, member_type = member_types[index]
            if member_name not in metadata_members:
                continue
            metadata_arg = self.query_metadata_expression(raw_args[index])
            if metadata_arg is None:
                metadata_arg = self.unavailable_query_metadata_argument(
                    member_name,
                    member_type,
                )
            expanded_args.append(metadata_arg)
        return expanded_args

    def generate_buffer_call(self, function_expr, func_name, raw_args, args):
        """Lower structured-buffer loads and stores to CUDA pointer indexing."""
        byte_address_call = self.generate_byte_address_buffer_call(
            function_expr, func_name, raw_args, args
        )
        if byte_address_call is not None:
            return byte_address_call

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "Append" and args:
                return self.generate_structured_buffer_append(
                    buffer_expr, buffer_type, args[0], operation
                )
            if operation == "Consume":
                return self.generate_structured_buffer_consume(
                    buffer_expr, buffer_type, operation
                )
            if operation == "GetDimensions":
                return self.generate_structured_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )
            access = self.format_structured_buffer_access(buffer_expr, raw_args, args)
            if access is None:
                return None
            if operation == "Load":
                diagnostic = self.structured_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "read",
                    self.diagnostic_zero_value_for_type(
                        self.array_access_element_type(buffer_type)
                    ),
                )
                if diagnostic is not None:
                    return diagnostic
                return access
            if operation == "Store":
                diagnostic = self.structured_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "write",
                    "((void)0)",
                )
                if diagnostic is not None:
                    return diagnostic
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.unsupported_structured_buffer_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            diagnostic = self.structured_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "read",
                self.diagnostic_zero_value_for_type(
                    self.array_access_element_type(buffer_type)
                ),
            )
            if diagnostic is not None:
                return diagnostic
            return f"{args[0]}[{args[1]}]"

        if func_name == "buffer_append" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_append(
                raw_args[0], buffer_type, args[1], func_name
            )

        if func_name == "buffer_consume" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_consume(
                raw_args[0], buffer_type, func_name
            )

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            diagnostic = self.structured_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "write",
                "((void)0)",
            )
            if diagnostic is not None:
                return diagnostic
            if self.structured_buffer_is_writable(buffer_type):
                return f"{args[0]}[{args[1]}] = {args[2]}"
            return self.unsupported_structured_buffer_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_structured_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower structured-buffer dimensions through an explicit length sidecar."""
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* CUDA structured buffer dimensions "
                "requires explicit length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def generate_structured_buffer_append(
        self, buffer_expr, buffer_type, value, operation
    ):
        """Lower AppendStructuredBuffer writes through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        diagnostic = self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            buffer_expr,
            "write",
            "((void)0)",
        )
        if diagnostic is not None:
            return diagnostic
        if parts is None or parts[0] != "AppendStructuredBuffer":
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, "((void)0)"
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, "((void)0)"
            )

        helper_name = self.require_structured_buffer_append_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter}, {value})"

    def generate_structured_buffer_consume(self, buffer_expr, buffer_type, operation):
        """Lower ConsumeStructuredBuffer reads through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        fallback = self.diagnostic_zero_value_for_type(
            parts[1] if parts is not None else None
        )
        diagnostic = self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            buffer_expr,
            "readwrite",
            fallback,
        )
        if diagnostic is not None:
            return diagnostic
        if parts is None or parts[0] != "ConsumeStructuredBuffer":
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, fallback
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.unsupported_structured_buffer_call(
                operation,
                buffer_type,
                self.diagnostic_zero_value_for_type(parts[1]),
            )

        helper_name = self.require_structured_buffer_consume_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter})"

    def require_structured_buffer_append_helper(self):
        """Register the CUDA helper for AppendStructuredBuffer operations."""
        helper_name = "cgl_append_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline void "
            f"{helper_name}(T* buffer, uint* counter, const T& value)\n"
            "{\n"
            "    uint index = atomicAdd(counter, 1u);\n"
            "    buffer[index] = value;\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_structured_buffer_consume_helper(self):
        """Register the CUDA helper for ConsumeStructuredBuffer operations."""
        helper_name = "cgl_consume_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline T "
            f"{helper_name}(const T* buffer, uint* counter)\n"
            "{\n"
            "    uint index = atomicSub(counter, 1u) - 1u;\n"
            "    return buffer[index];\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def structured_buffer_atomic_operations(self):
        """Return generic atomic mappings for RWStructuredBuffer element targets."""
        integer_kinds = {"int", "uint"}
        add_kinds = {*integer_kinds, "uint64", "float"}
        add_exchange_kinds = {*integer_kinds, "float"}
        operations = {
            "atomicAdd": ("atomicAdd", 2, add_kinds, "int/uint/u64/float"),
            "atomicSub": ("atomicSub", 2, integer_kinds, "int/uint"),
            "atomicMin": ("atomicMin", 2, integer_kinds, "int/uint"),
            "atomicMax": ("atomicMax", 2, integer_kinds, "int/uint"),
            "atomicAnd": ("atomicAnd", 2, integer_kinds, "int/uint"),
            "atomicOr": ("atomicOr", 2, integer_kinds, "int/uint"),
            "atomicXor": ("atomicXor", 2, integer_kinds, "int/uint"),
            "atomicInc": ("atomicInc", 2, {"uint"}, "uint"),
            "atomicDec": ("atomicDec", 2, {"uint"}, "uint"),
            "atomicExchange": ("atomicExch", 2, add_exchange_kinds, "int/uint/float"),
            "atomicCompareExchange": ("atomicCAS", 3, integer_kinds, "int/uint"),
            "atomicCompSwap": ("atomicCAS", 3, integer_kinds, "int/uint"),
        }
        # CUDA's block- and system-scoped atomics (atomicAdd_block,
        # atomicCAS_system, ...) take the same operands as the device-wide
        # intrinsic they are named after.
        for intrinsic, arg_count, kinds, kinds_label in list(operations.values()):
            for scope in ("_block", "_system"):
                operations.setdefault(
                    f"{intrinsic}{scope}",
                    (f"{intrinsic}{scope}", arg_count, kinds, kinds_label),
                )
        return operations

    def generate_structured_buffer_atomic_call(self, func_name, raw_args, args):
        """Lower atomics on RWStructuredBuffer element lvalues to CUDA atomics."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not raw_args:
            return None

        target = self.structured_buffer_atomic_target(raw_args[0])
        if target is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        target_type = target["target_type"]
        fallback = self.diagnostic_zero_value_for_type(target_type)

        if len(args) != required_arg_count:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        access_diagnostic = self.structured_buffer_access_diagnostic(
            func_name,
            target["buffer_type"],
            target["target_expr"],
            "readwrite",
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic

        buffer_base_type, _ = self.structured_buffer_type_parts(target["buffer_type"])
        if buffer_base_type != "RWStructuredBuffer":
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                "requires RWStructuredBuffer target",
                fallback,
            )

        scalar_kind = self.cuda_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires supported scalar {supported_kinds_label} target",
                fallback,
            )

        target_expr = self.visit(target["target_expr"])
        target_pointer = self.cuda_atomic_target_pointer(
            f"&{target_expr}", target_type, scalar_kind
        )
        value_args = ", ".join(args[1:])
        return f"{intrinsic}({target_pointer}, {value_args})"

    def generate_plain_atomic_call(self, func_name, raw_args, args):
        """Lower CUDA atomics on ordinary scalar lvalues to pointer operands."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        raw_target = raw_args[0] if raw_args else None
        target_expr = (
            self.strip_address_of_expression(raw_target) if raw_target else None
        )
        address_taken = target_expr is not raw_target
        target_type = self.expression_result_type(target_expr) if target_expr else None
        indirect_info = self.cuda_indirect_type_info(target_type)
        pointee_type = (
            indirect_info["pointee_type"] if indirect_info is not None else None
        )
        atomic_type = pointee_type or target_type
        fallback = self.diagnostic_zero_value_for_type(atomic_type)

        if len(args) != required_arg_count:
            return self.unsupported_plain_atomic_call(
                func_name,
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        if target_expr is None or (
            pointee_type is None and not self.is_plain_atomic_lvalue(target_expr)
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                "requires assignable scalar target",
                fallback,
            )

        if (
            address_taken
            and indirect_info is not None
            and indirect_info["kind"] == "pointer"
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {indirect_info['type_label']} requires pointer target, "
                "not address of pointer",
                fallback,
            )

        readonly_reason = self.plain_atomic_readonly_target_reason(
            target_expr, indirect_info
        )
        if readonly_reason is not None:
            return self.unsupported_plain_atomic_call(
                func_name,
                readonly_reason,
                fallback,
            )
        access_diagnostic = self.glsl_buffer_block_read_write_diagnostic(
            target_expr, func_name, fallback
        )
        if access_diagnostic is not None:
            return access_diagnostic

        scalar_kind = self.cuda_atomic_scalar_kind(atomic_type)
        if scalar_kind not in supported_kinds:
            type_label = (
                indirect_info["type_label"]
                if indirect_info is not None
                else self.type_name_string(atomic_type) or "unknown target"
            )
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {type_label} requires supported scalar "
                f"{supported_kinds_label} target",
                fallback,
            )

        target_code = self.visit(target_expr)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            target_pointer = target_code
        else:
            target_pointer = f"&{target_code}"
        target_pointer = self.cuda_atomic_target_pointer(
            target_pointer, atomic_type, scalar_kind
        )
        value_args = ", ".join(args[1:])
        return f"{intrinsic}({target_pointer}, {value_args})"

    def cuda_atomic_target_pointer(self, target_pointer, atomic_type, scalar_kind):
        """CUDA declares its 64-bit atomics only for ``unsigned long long*``, which
        ``uint64_t`` is not on 64-bit Linux (``unsigned long``), so cast the target."""
        if scalar_kind != "uint64" or self.convert_crossgl_type_to_cuda(
            self.type_name_string(atomic_type)
        ) in {"unsigned long long", "unsigned long long int"}:
            return target_pointer
        return f"reinterpret_cast<unsigned long long*>({target_pointer})"

    def is_plain_atomic_lvalue(self, expr):
        """Return whether an expression can be addressed for a CUDA atomic."""
        return isinstance(
            expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode, PointerAccessNode)
        )

    def cuda_atomic_pointer_pointee_type(self, type_name):
        """Return a scalar pointee type for pointer-typed CUDA atomic operands."""
        info = self.cuda_indirect_type_info(type_name)
        return info["pointee_type"] if info is not None else None

    def cuda_indirect_type_info(self, type_name):
        """Return pointer/reference metadata for CUDA indirect type spellings."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        if type_name.startswith("ptr<") and type_name.endswith(">"):
            pointee = type_name[4:-1].strip()
            return {
                "kind": "pointer",
                "pointee_type": pointee,
                "readonly": False,
                "type_label": f"{pointee}*",
            }

        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        for candidate in (type_name, mapped_type):
            info = self.cuda_pointer_or_reference_type_info(candidate)
            if info is not None:
                return info
        return None

    def cuda_pointer_or_reference_type_info(self, type_name):
        """Parse a CUDA pointer/reference spelling into target metadata."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        stripped = type_name.strip()
        for suffix, kind in (("*", "pointer"), ("&", "reference")):
            if not stripped.endswith(suffix):
                continue
            pointee = stripped[: -len(suffix)].strip()
            readonly = pointee.startswith("const ")
            if readonly:
                pointee = pointee[len("const ") :].strip()
            return {
                "kind": kind,
                "pointee_type": pointee,
                "readonly": readonly,
                "type_label": stripped,
            }
        return None

    def plain_atomic_readonly_target_reason(self, target_expr, indirect_info):
        """Return a diagnostic reason for readonly plain atomic targets."""
        if indirect_info is not None and indirect_info["readonly"]:
            if indirect_info["kind"] == "reference":
                return (
                    f"on {indirect_info['type_label']} requires mutable "
                    "reference target"
                )
            return (
                f"on {indirect_info['type_label']} requires writable pointer " "target"
            )

        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            array_type = self.expression_result_type(array_expr)
            array_info = self.cuda_indirect_type_info(array_type)
            if array_info is not None and array_info["readonly"]:
                if array_info["kind"] == "reference":
                    return (
                        f"on {array_info['type_label']} requires mutable "
                        "reference target"
                    )
                return (
                    f"on {array_info['type_label']} requires writable pointer " "target"
                )
        return None

    def unsupported_plain_atomic_call(self, operation, reason, fallback):
        """Refuse unsupported plain CUDA atomics; the returned value stands in until the refusal is raised."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} atomic call: {operation} {reason}",
            fallback,
        )

    def structured_buffer_atomic_target(self, target_expr):
        """Return RWStructuredBuffer target metadata for an atomic lvalue."""
        target_expr = self.strip_address_of_expression(target_expr)
        element_access = self.structured_buffer_element_access(target_expr)
        if element_access is None:
            return None

        array_expr = getattr(
            element_access, "array_expr", getattr(element_access, "array", None)
        )
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        target_type = self.expression_result_type(target_expr) or parts[1]
        return {
            "buffer_type": buffer_type,
            "target_expr": target_expr,
            "target_type": target_type,
        }

    def strip_address_of_expression(self, expr):
        """Return the lvalue inside an address-of expression."""
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "&":
                return getattr(expr, "operand", expr)
        return expr

    def structured_buffer_element_access(self, target_expr):
        """Return the structured-buffer element access inside an atomic lvalue."""
        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            buffer_type = self.expression_result_type(array_expr)
            if self.structured_buffer_type_parts(buffer_type) is not None:
                return target_expr

        if isinstance(target_expr, MemberAccessNode):
            object_expr = getattr(
                target_expr, "object_expr", getattr(target_expr, "object", None)
            )
            return self.structured_buffer_element_access(object_expr)

        return None

    def cuda_atomic_scalar_kind(self, type_name):
        """Return the CUDA atomic scalar kind supported for structured buffers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        if type_name in {"uint", "u32"} or mapped_type in {"uint", "unsigned int"}:
            return "uint"
        if type_name in {"u64", "uint64", "uint64_t"} or mapped_type in {
            "unsigned long long",
            "unsigned long long int",
        }:
            return "uint64"
        if type_name in {"int", "i32"} or mapped_type == "int":
            return "int"
        if type_name in {"float", "f32"} or mapped_type == "float":
            return "float"
        return None

    def unsupported_structured_buffer_atomic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Refuse unsupported structured-buffer atomics; the returned value stands in until the refusal is raised."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} structured buffer atomic: {operation} on {buffer_type} {reason}",
            fallback,
        )

    def generate_byte_address_buffer_call(
        self, function_expr, func_name, raw_args, args
    ):
        """Lower byte-address buffer methods to typed CUDA byte-pointer helpers."""
        member_call = self.byte_address_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "GetDimensions":
                return self.generate_byte_address_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )

            if operation in self.byte_address_buffer_atomic_operations():
                return self.generate_byte_address_buffer_atomic(
                    buffer_expr, buffer_type, operation, args
                )

            component_count = self.byte_address_buffer_component_count(operation)
            if component_count is None:
                return None

            buffer_name = self.visit(buffer_expr)
            if operation.startswith("Load") and len(args) >= 1:
                diagnostic = self.byte_address_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "read",
                    self.diagnostic_zero_value_for_type(
                        self.byte_address_buffer_value_type(component_count)
                    ),
                )
                if diagnostic is not None:
                    return diagnostic
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
                diagnostic = self.byte_address_buffer_access_diagnostic(
                    operation,
                    buffer_type,
                    buffer_expr,
                    "write",
                    "((void)0)",
                )
                if diagnostic is not None:
                    return diagnostic
                if self.byte_address_buffer_is_writable(buffer_type):
                    helper_name = self.require_byte_address_store_helper(
                        component_count
                    )
                    return f"{helper_name}({buffer_name}, {args[0]}, {args[1]})"
                return self.unsupported_byte_address_buffer_call(
                    operation, buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "read",
                "0u",
            )
            if diagnostic is not None:
                return diagnostic
            helper_name = self.require_byte_address_load_helper(1)
            return f"{helper_name}({args[0]}, {args[1]})"

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            return self.generate_byte_address_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_access_diagnostic(
                func_name,
                buffer_type,
                raw_args[0],
                "write",
                "((void)0)",
            )
            if diagnostic is not None:
                return diagnostic
            if self.byte_address_buffer_is_writable(buffer_type):
                helper_name = self.require_byte_address_store_helper(1)
                return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"
            return self.unsupported_byte_address_buffer_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_byte_address_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower byte-address buffer dimensions through a byte-length sidecar."""
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* CUDA byte-address buffer dimensions "
                "requires explicit byte-length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def byte_address_buffer_atomic_operations(self):
        """Return supported RWByteAddressBuffer atomic method mappings."""
        return {
            "InterlockedAdd": ("add", "atomicAdd", 2),
            "InterlockedMin": ("min", "atomicMin", 2),
            "InterlockedMax": ("max", "atomicMax", 2),
            "InterlockedAnd": ("and", "atomicAnd", 2),
            "InterlockedOr": ("or", "atomicOr", 2),
            "InterlockedXor": ("xor", "atomicXor", 2),
            "InterlockedExchange": ("exchange", "atomicExch", 2),
            "InterlockedCompareExchange": ("compare_exchange", "atomicCAS", 3),
        }

    def generate_byte_address_buffer_atomic(
        self, buffer_expr, buffer_type, operation, args
    ):
        """Lower RWByteAddressBuffer Interlocked* methods to CUDA atomics."""
        operation_info = self.byte_address_buffer_atomic_operations().get(operation)
        if operation_info is None:
            return None

        operation_name, intrinsic, required_args = operation_info
        has_out_arg = len(args) == required_args + 1
        fallback = "((void)0)" if has_out_arg else "0u"
        if len(args) < required_args or len(args) > required_args + 1:
            return self.unsupported_byte_address_buffer_call(
                operation, buffer_type, fallback
            )
        access_diagnostic = self.byte_address_buffer_access_diagnostic(
            operation,
            buffer_type,
            buffer_expr,
            "readwrite",
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic
        if not self.byte_address_buffer_is_writable(buffer_type):
            return self.unsupported_byte_address_buffer_call(
                operation, buffer_type, fallback
            )

        helper_name = self.require_byte_address_atomic_helper(operation_name, intrinsic)
        buffer_name = self.visit(buffer_expr)
        helper_args = [buffer_name, *args[:required_args]]
        call = f"{helper_name}({', '.join(helper_args)})"
        if has_out_arg:
            return f"{args[required_args]} = {call}"
        return call

    def byte_address_atomic_helper_name(self, operation):
        """Return a stable CUDA helper name for a byte-address atomic operation."""
        return f"cgl_byte_address_atomic_{operation}_uint"

    def require_byte_address_atomic_helper(self, operation, intrinsic):
        """Register a CUDA byte-address atomic helper and return its name."""
        helper_name = self.byte_address_atomic_helper_name(operation)
        if helper_name in self.helper_functions:
            return helper_name

        pointer_expr = "reinterpret_cast<unsigned int*>(buffer + offset)"
        if operation == "compare_exchange":
            helper = (
                "__device__ inline uint "
                f"{helper_name}(unsigned char* buffer, uint offset, "
                "uint compare_value, uint value)\n"
                "{\n"
                f"    return {intrinsic}({pointer_expr}, compare_value, value);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        helper = (
            "__device__ inline uint "
            f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
            "{\n"
            f"    return {intrinsic}({pointer_expr}, value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_member_call(self, function_expr):
        """Return byte-address buffer member call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def byte_address_buffer_component_count(self, operation):
        """Return the uint lane count for Load/Store byte-address operations."""
        operation_counts = {
            "Load": 1,
            "Load2": 2,
            "Load3": 3,
            "Load4": 4,
            "Store": 1,
            "Store2": 2,
            "Store3": 3,
            "Store4": 4,
        }
        return operation_counts.get(operation)

    def byte_address_buffer_value_type(self, component_count):
        """Return the CUDA uint vector type for a byte-address operation."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def byte_address_helper_suffix(self, component_count):
        """Return a stable helper-name suffix for byte-address operations."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def require_byte_address_load_helper(self, component_count):
        """Register a CUDA byte-address load helper and return its name."""
        helper_name = (
            f"cgl_byte_address_load_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        if component_count == 1:
            helper = (
                "__device__ inline uint "
                f"{helper_name}(const unsigned char* buffer, uint offset)\n"
                "{\n"
                "    return *reinterpret_cast<const uint*>(buffer + offset);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_load_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        args = [
            f"{scalar_helper}(buffer, offset + {index * 4}u)"
            for index, _ in enumerate(components)
        ]
        value_type = self.byte_address_buffer_value_type(component_count)
        constructor = self.convert_builtin_function(value_type)
        helper = (
            f"__device__ inline {value_type} "
            f"{helper_name}(const unsigned char* buffer, uint offset)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_byte_address_store_helper(self, component_count):
        """Register a CUDA byte-address store helper and return its name."""
        helper_name = (
            f"cgl_byte_address_store_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        value_type = self.byte_address_buffer_value_type(component_count)
        if component_count == 1:
            helper = (
                "__device__ inline void "
                f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
                "{\n"
                "    *reinterpret_cast<uint*>(buffer + offset) = value;\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_store_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        lines = [
            f"    {scalar_helper}(buffer, offset + {index * 4}u, value.{component});"
            for index, component in enumerate(components)
        ]
        helper = (
            f"__device__ inline void {helper_name}"
            f"(unsigned char* buffer, uint offset, {value_type} value)\n"
            "{\n" + "\n".join(lines) + "\n}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def unsupported_byte_address_buffer_call(self, operation, buffer_type, fallback):
        """Refuse unsupported byte-address buffer operations; the returned value stands in until the refusal is raised."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} byte-address buffer call: {operation} on {buffer_type}",
            fallback,
        )

    def structured_buffer_member_call(self, function_expr):
        """Return structured-buffer member call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def format_structured_buffer_access(self, buffer_expr, raw_args, args):
        """Format one CUDA pointer access for a structured-buffer operation."""
        if not raw_args or not args:
            return None
        buffer_name = self.visit(buffer_expr)
        return f"{buffer_name}[{args[0]}]"

    def unsupported_structured_buffer_call(self, operation, buffer_type, fallback):
        """Refuse unsupported structured-buffer operations; the returned value stands in until the refusal is raised."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} structured buffer call: {operation} on {buffer_type}",
            fallback,
        )
