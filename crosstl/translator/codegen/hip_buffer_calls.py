"""Buffer, structured-buffer and atomic call lowering of the HIP generator, with their access diagnostics."""

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    UnaryOpNode,
)

# The system-scoped atomics HIP does not declare, and the helper that performs each.
HIP_SYSTEM_WRAPPING_ATOMICS = {
    "atomicInc_system": "cgl_hip_atomicInc_system",
    "atomicDec_system": "cgl_hip_atomicDec_system",
}
# The value each helper stores in place of ``assumed``.
HIP_SYSTEM_WRAPPING_ATOMIC_UPDATES = {
    "cgl_hip_atomicInc_system": "(assumed >= limit) ? 0u : assumed + 1u",
    "cgl_hip_atomicDec_system": (
        "((assumed == 0u) || (assumed > limit)) ? limit : assumed - 1u"
    ),
}


class HipBufferCallMixin:
    """Buffer call members of ``HipCodeGen``."""

    def generate_buffer_call(self, function_expr, func_name, raw_args, args):
        """Lower structured and byte-address buffer calls to HIP pointer forms."""
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
                parts = self.structured_buffer_type_parts(buffer_type)
                fallback = self.diagnostic_zero_value_for_type(
                    parts[1] if parts is not None else None
                )
                diagnostic = self.structured_buffer_read_diagnostic(
                    buffer_expr, buffer_type, operation, fallback
                )
                if diagnostic is not None:
                    return diagnostic
                return access
            if operation == "Store":
                diagnostic = self.structured_buffer_write_diagnostic(
                    buffer_expr, buffer_type, operation
                )
                if diagnostic is not None:
                    return diagnostic
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.structured_buffer_diagnostic_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is None:
                return None
            fallback = self.diagnostic_zero_value_for_type(parts[1])
            diagnostic = self.structured_buffer_read_diagnostic(
                raw_args[0], buffer_type, func_name, fallback
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
            diagnostic = self.structured_buffer_write_diagnostic(
                raw_args[0], buffer_type, func_name
            )
            if diagnostic is not None:
                return diagnostic
            if self.structured_buffer_is_writable(buffer_type):
                return f"{args[0]}[{args[1]}] = {args[2]}"
            return self.structured_buffer_diagnostic_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_byte_address_buffer_call(
        self, function_expr, func_name, raw_args, args
    ):
        """Lower byte-address buffer loads and stores to typed HIP helpers."""
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
                fallback = self.diagnostic_zero_value_for_type(
                    self.byte_address_buffer_value_type(component_count)
                )
                diagnostic = self.byte_address_buffer_read_diagnostic(
                    buffer_expr, buffer_type, operation, fallback
                )
                if diagnostic is not None:
                    return diagnostic
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
                diagnostic = self.byte_address_buffer_write_diagnostic(
                    buffer_expr, buffer_type, operation
                )
                if diagnostic is not None:
                    return diagnostic
                if self.byte_address_buffer_is_writable(buffer_type):
                    helper_name = self.require_byte_address_store_helper(
                        component_count
                    )
                    return f"{helper_name}({buffer_name}, {args[0]}, {args[1]})"
                return self.byte_address_buffer_diagnostic_call(
                    operation, buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_read_diagnostic(
                raw_args[0], buffer_type, func_name, "0u"
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
            diagnostic = self.byte_address_buffer_write_diagnostic(
                raw_args[0], buffer_type, func_name
            )
            if diagnostic is not None:
                return diagnostic
            if self.byte_address_buffer_is_writable(buffer_type):
                helper_name = self.require_byte_address_store_helper(1)
                return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"
            return self.byte_address_buffer_diagnostic_call(
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
                "0 /* HIP structured buffer dimensions "
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
        if parts is None or parts[0] != "AppendStructuredBuffer":
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, "((void)0)"
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, "((void)0)"
            )

        helper_name = self.require_structured_buffer_append_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter}, {value})"

    def generate_structured_buffer_consume(self, buffer_expr, buffer_type, operation):
        """Lower ConsumeStructuredBuffer reads through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "ConsumeStructuredBuffer":
            fallback = self.diagnostic_zero_value_for_type(
                parts[1] if parts is not None else None
            )
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.structured_buffer_diagnostic_call(
                operation,
                buffer_type,
                self.diagnostic_zero_value_for_type(parts[1]),
            )

        helper_name = self.require_structured_buffer_consume_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter})"

    def require_structured_buffer_append_helper(self):
        """Register the HIP helper for AppendStructuredBuffer operations."""
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
        """Register the HIP helper for ConsumeStructuredBuffer operations."""
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
        add_exchange_kinds = {*integer_kinds, "float"}
        operations = {
            "atomicAdd": ("atomicAdd", 2, add_exchange_kinds, "int/uint/float"),
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
        # HIP's system-scoped atomics (atomicAdd_system, atomicCAS_system, ...)
        # take the same operands as the device-wide intrinsic they are named
        # after. HIP declares no atomicInc_system or atomicDec_system: a
        # compare-and-swap loop on atomicCAS_system performs them
        # (HIP_SYSTEM_WRAPPING_ATOMICS).
        for intrinsic, arg_count, kinds, kinds_label in list(operations.values()):
            system_name = f"{intrinsic}_system"
            system_intrinsic = (
                HIP_SYSTEM_WRAPPING_ATOMICS[system_name]
                if system_name in HIP_SYSTEM_WRAPPING_ATOMICS
                else system_name
            )
            operations.setdefault(
                f"{intrinsic}_system",
                (system_intrinsic, arg_count, kinds, kinds_label),
            )
        # HIP's atomicAddNoRet adds to a float and returns nothing; it has no
        # system-scoped form.
        operations["atomicAddNoRet"] = ("atomicAddNoRet", 2, {"float"}, "float")
        return operations

    def generate_structured_buffer_atomic_call(self, func_name, raw_args, args):
        """Lower atomics on RWStructuredBuffer element lvalues to HIP atomics."""
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

        buffer_base_type, _ = self.structured_buffer_type_parts(target["buffer_type"])
        access_diagnostic = self.structured_buffer_read_write_diagnostic(
            target["buffer_expr"],
            target["buffer_type"],
            func_name,
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic
        if buffer_base_type != "RWStructuredBuffer":
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                "requires RWStructuredBuffer target",
                fallback,
            )

        scalar_kind = self.hip_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires supported scalar {supported_kinds_label} target",
                fallback,
            )

        target_expr = args[0]
        value_args = ", ".join(args[1:])
        self.require_hip_system_wrapping_atomic_helper(intrinsic)
        return f"{intrinsic}(&{target_expr}, {value_args})"

    def generate_plain_atomic_call(self, func_name, raw_args, args):
        """Lower HIP atomics on ordinary scalar lvalues to pointer operands."""
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
        indirect_info = self.hip_indirect_type_info(target_type)
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

        scalar_kind = self.hip_atomic_scalar_kind(atomic_type)
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
        value_args = ", ".join(args[1:])
        self.require_hip_system_wrapping_atomic_helper(intrinsic)
        return f"{intrinsic}({target_pointer}, {value_args})"

    def require_hip_system_wrapping_atomic_helper(self, intrinsic):
        """Register the compare-and-swap loop that performs a system-scoped ``atomicInc``/``atomicDec``.

        CUDA's ``atomicInc_system(address, limit)`` stores ``(old >= limit) ? 0 : old + 1`` and
        ``atomicDec_system(address, limit)`` stores ``((old == 0) || (old > limit)) ? limit : old - 1``, atomically
        for the host and peer devices too, and both return ``old`` (CUDA C++ Programming Guide, "Atomic
        Functions"). The loop retries ``atomicCAS_system`` until no other thread changed the word in between, the
        construction the guide gives for atomics the hardware lacks.
        """
        update = HIP_SYSTEM_WRAPPING_ATOMIC_UPDATES.get(intrinsic)
        if update is None or intrinsic in self.helper_functions:
            return
        self.helper_functions[intrinsic] = (
            f"__device__ inline unsigned int {intrinsic}(unsigned int* address, unsigned int limit)\n"
            "{\n"
            "    unsigned int old = *address;\n"
            "    unsigned int assumed;\n"
            "    do {\n"
            "        assumed = old;\n"
            f"        old = atomicCAS_system(address, assumed, {update});\n"
            "    } while (old != assumed);\n"
            "    return old;\n"
            "}"
        )

    def is_plain_atomic_lvalue(self, expr):
        """Return whether an expression can be addressed for a HIP atomic."""
        return isinstance(
            expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode, PointerAccessNode)
        )

    def hip_indirect_type_info(self, type_name):
        """Return pointer/reference metadata for HIP indirect type spellings."""
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

        mapped_type = self.map_type(type_name)
        for candidate in (type_name, mapped_type):
            info = self.hip_pointer_or_reference_type_info(candidate)
            if info is not None:
                return info
        return None

    def hip_pointer_or_reference_type_info(self, type_name):
        """Parse a HIP pointer/reference spelling into target metadata."""
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
            return f"on {indirect_info['type_label']} requires writable pointer target"

        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            array_type = self.expression_result_type(array_expr)
            array_info = self.hip_indirect_type_info(array_type)
            if array_info is not None and array_info["readonly"]:
                if array_info["kind"] == "reference":
                    return (
                        f"on {array_info['type_label']} requires mutable "
                        "reference target"
                    )
                return f"on {array_info['type_label']} requires writable pointer target"
        return None

    def strip_address_of_expression(self, expr):
        """Return the lvalue inside an address-of expression."""
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "&":
                return getattr(expr, "operand", expr)
        return expr

    def unsupported_plain_atomic_call(self, operation, reason, fallback):
        """Refuse unsupported ordinary HIP atomics; the returned value stands in until the refusal is raised."""
        return self.refuse_lowering(
            f"{self.resource_backend_name()} atomic: {operation} {reason}", fallback
        )

    def structured_buffer_atomic_target(self, target_expr):
        """Return RWStructuredBuffer target metadata for an atomic lvalue."""
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
            "buffer_expr": array_expr,
            "buffer_type": buffer_type,
            "target_type": target_type,
        }

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

    def structured_buffer_data_access_info(self, expr):
        """Return direct structured-buffer element access metadata, if any."""
        if not isinstance(expr, ArrayAccessNode):
            return None

        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        result_type = self.expression_result_type(expr)
        if self.structured_buffer_type_parts(result_type) is not None:
            return None

        return {
            "buffer_expr": array_expr,
            "buffer_type": buffer_type,
            "element_type": result_type or parts[1],
        }

    def structured_buffer_element_read_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(info["element_type"])
        return self.structured_buffer_read_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation, fallback
        )

    def structured_buffer_element_write_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        return self.structured_buffer_write_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation
        )

    def structured_buffer_element_read_write_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(info["element_type"])
        return self.structured_buffer_read_write_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation, fallback
        )

    def hip_atomic_scalar_kind(self, type_name):
        """Return the HIP atomic scalar kind supported for structured buffers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        mapped_type = self.map_type(type_name)
        if type_name in {"uint", "u32"} or mapped_type in {"uint", "unsigned int"}:
            return "uint"
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

    def generate_byte_address_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower byte-address buffer dimensions through a byte-length sidecar."""
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* HIP byte-address buffer dimensions "
                "requires explicit byte-length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def structured_buffer_member_call(self, function_expr):
        """Return structured-buffer member-call pieces, if applicable."""
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
        """Format one HIP pointer access for a structured-buffer operation."""
        if not raw_args or not args:
            return None
        buffer_name = self.visit(buffer_expr)
        return f"{buffer_name}[{args[0]}]"

    def structured_buffer_diagnostic_call(self, operation, buffer_type, fallback):
        """Refuse unsupported structured-buffer operations; the returned value stands in until the refusal is raised."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} structured buffer call: {operation} on {buffer_type}",
            fallback,
        )

    def structured_buffer_access_diagnostic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Refuse unsupported structured-buffer access metadata; the returned value stands in until the refusal is raised."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} structured buffer access: {operation} {reason} on {buffer_type}",
            fallback,
        )

    def structured_buffer_read_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access == "writeonly":
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readable buffer access",
                fallback,
            )
        return None

    def structured_buffer_write_diagnostic(self, buffer_expr, buffer_type, operation):
        access = self.buffer_resource_access(buffer_expr)
        if access == "readonly":
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires writable buffer access",
                "((void)0)",
            )
        return None

    def structured_buffer_read_write_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access in {"readonly", "writeonly"}:
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readwrite buffer access",
                fallback,
            )
        return None

    def diagnostic_zero_value_for_type(self, type_name):
        """Return a HIP fallback expression for rejected value-producing calls."""
        if type_name is None:
            return "0"

        mapped_type = self.map_type(type_name)
        vector_info = self.vector_type_info(mapped_type) or self.vector_type_info(
            self.type_name_string(type_name)
        )
        if vector_info is not None:
            component_type = vector_info["component_type"]
            if component_type == "uint":
                zero = "0u"
            elif component_type == "double":
                zero = "0.0"
            elif component_type == "float":
                zero = "0.0f"
            elif component_type == "bool":
                zero = "false"
            else:
                zero = "0"
            values = ", ".join([zero] * len(vector_info["components"]))
            return f"{vector_info['constructor']}({values})"

        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "half"}:
            return "0.0f"
        if mapped_type == "double":
            return "0.0"
        if mapped_type in {"uint", "unsigned int", "u32"}:
            return "0u"
        if mapped_type in {
            "int",
            "short",
            "char",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned long long",
        }:
            return "0"
        return f"{mapped_type}{{}}"
