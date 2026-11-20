"""Atomic fences, barriers and atomic functions."""

from ..barrier_import import metal_barrier_statements
from ..bit_scan_import import lower_scan as lower_bit_scan
from ..bit_scan_import import scan_function as bit_scan_function
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .support import (
    MetalAtomicCompareExchangeLoweringError,
    MetalAtomicFenceLoweringError,
    MetalAtomicFetchSubLoweringError,
    MetalAtomicLoadLoweringError,
    MetalAtomicStoreLoweringError,
)


class MetalSynchronizationMixin:
    """Members of ``MetalToCrossGLConverter``: atomic fences, barriers and atomic functions."""

    def is_metal_atomic_fence_call(self, name):
        function_name = str(name).lstrip(":")
        if function_name == "metal::atomic_thread_fence":
            return True
        return (
            function_name == "atomic_thread_fence"
            and function_name not in self.user_function_names
        )

    @staticmethod
    def metal_atomic_fence_operand_identifier(expr):
        if isinstance(expr, VariableNode):
            name = getattr(expr, "name", None)
        elif isinstance(expr, str):
            name = expr
        else:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def metal_atomic_fence_operand_text(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.metal_atomic_fence_operand_text(expr.left)} | "
                f"{self.metal_atomic_fence_operand_text(expr.right)}"
            )
        name = self.metal_atomic_fence_operand_identifier(expr)
        return name if name is not None else type(expr).__name__

    def collect_metal_atomic_fence_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_metal_atomic_fence_flags(expr.left)
            right = self.collect_metal_atomic_fence_flags(expr.right)
            if left is None or right is None:
                return None
            return left + right
        name = self.metal_atomic_fence_operand_identifier(expr)
        if name in self.metal_atomic_fence_memory_flags:
            return (name,)
        return None

    def metal_atomic_thread_fence_call(self, args, source_location=None):
        if len(args) not in {2, 3}:
            raise MetalAtomicFenceLoweringError(
                "invalid-argument-count",
                memory_flags=(
                    self.metal_atomic_fence_operand_text(args[0]) if args else None
                ),
                memory_order=(
                    self.metal_atomic_fence_operand_text(args[1])
                    if len(args) > 1
                    else None
                ),
                thread_scope=(
                    self.metal_atomic_fence_operand_text(args[2])
                    if len(args) > 2
                    else None
                ),
                source_location=source_location,
            )

        flags = self.collect_metal_atomic_fence_flags(args[0])
        order = self.metal_atomic_fence_operand_identifier(args[1])
        scope = (
            self.metal_atomic_fence_operand_identifier(args[2])
            if len(args) == 3
            else "thread_scope_device"
        )
        contract = {
            "memory_flags": self.metal_atomic_fence_operand_text(args[0]),
            "memory_order": order,
            "thread_scope": scope,
            "source_location": source_location,
        }
        if flags is None:
            raise MetalAtomicFenceLoweringError("unsupported-memory-flags", **contract)
        if order not in self.metal_atomic_fence_memory_orders:
            raise MetalAtomicFenceLoweringError("unsupported-memory-order", **contract)
        if scope not in self.metal_atomic_fence_thread_scopes:
            raise MetalAtomicFenceLoweringError("unsupported-thread-scope", **contract)

        rendered_flags = " | ".join(flags)
        return f"atomicThreadFence({rendered_flags}, {order}, {scope})"

    def metal_synchronization_function_call(self, name, args, source_location=None):
        unscoped_name = str(name).split("::")[-1]

        if self.is_metal_atomic_fence_call(name):
            return self.metal_atomic_thread_fence_call(args, source_location)

        if unscoped_name in {"threadgroup_barrier", "simdgroup_barrier"}:
            if self.metal_user_function_overloads(name):
                return None
            lowered = metal_barrier_statements(
                unscoped_name, self.metal_mem_flag_names(args)
            )
            if isinstance(lowered, str):
                return self.refuse_import(lowered, "workgroupBarrier()")
            if len(lowered) != 1:
                return self.refuse_import(
                    f"{unscoped_name} with device or texture memory flags becomes a sequence of CrossGL "
                    "barriers, which needs a statement of its own",
                    "workgroupBarrier()",
                )
            return f"{lowered[0]}()"

        return None

    def generate_metal_barrier_statement(self, stmt, indent):
        """A Metal barrier statement as its CrossGL barrier sequence, or ``None`` for other statements."""
        name = str(stmt.name).lstrip(":")
        unscoped_name = name.split("::")[-1]
        if unscoped_name not in {"threadgroup_barrier", "simdgroup_barrier"}:
            return None
        if "::" in name and not name.startswith("metal::"):
            return None
        if self.metal_user_function_overloads(name):
            return None
        lowered = metal_barrier_statements(
            unscoped_name, self.metal_mem_flag_names(stmt.args)
        )
        if isinstance(lowered, str) or len(lowered) == 1:
            return None
        separator = ";\n" + "    " * indent
        return separator.join(f"{call}()" for call in lowered) + ";\n"

    def metal_bit_scan_expression(self, name, args, is_main):
        """MSL ``clz``/``ctz`` as a CrossGL ``findMSB``/``findLSB`` expression, or ``None`` for other calls."""
        function = bit_scan_function(name, self.user_function_names)
        if function is None or len(args) != 1:
            return None
        metal_type = self.resolve_type_alias(self.expression_metal_type(args[0]))
        return lower_bit_scan(
            function,
            self.generate_expression(args[0], is_main),
            metal_type,
            self.map_type(metal_type) if metal_type else None,
            self.metal_small_vector_type_parts,
            self.metal_scalar_arithmetic_type_info,
        )

    def metal_uniform_value_expression(self, name, args, is_main):
        """Lower Metal's uniformity assertion while preserving its value."""
        function_name = str(name).lstrip(":")
        explicitly_metal = function_name.startswith("metal::")
        unscoped_name = function_name.split("::")[-1]
        if unscoped_name != "make_uniform":
            return None
        if "::" in function_name and not explicitly_metal:
            return None
        if not explicitly_metal and unscoped_name in self.user_function_names:
            return None
        if len(args) != 1:
            return None
        rendered = self.generate_expression(args[0], is_main)
        if isinstance(args[0], (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            return f"({rendered})"
        return rendered

    def metal_atomic_function_call(self, name, args, is_main, source_location=None):
        function_name = str(name).lstrip(":")
        if function_name in {
            "atomic_compare_exchange_weak_explicit",
            "metal::atomic_compare_exchange_weak_explicit",
        }:
            if self.metal_user_function_overloads(name):
                return None
            error = MetalAtomicCompareExchangeLoweringError
            if len(args) != 5:
                raise error(
                    "requires target, expected pointer, desired value and two memory orders",
                    source_location,
                )
            for argument in args[3:]:
                order = str(getattr(argument, "name", argument)).lstrip(":")
                shadowed = order == "memory_order_relaxed" and (
                    order in self.current_variable_types
                    or order in self.global_variable_types
                )
                if shadowed or order not in {
                    "memory_order_relaxed",
                    "metal::memory_order_relaxed",
                }:
                    raise error(
                        "requires memory_order_relaxed for success and failure",
                        source_location,
                    )
            target_qualifiers = set(self.expression_metal_type_qualifiers(args[0]))
            if "const" in target_qualifiers or not target_qualifiers.intersection(
                {"device", "threadgroup"}
            ):
                raise error(
                    "requires writable device or threadgroup storage", source_location
                )
            expected_qualifiers = set(self.expression_metal_type_qualifiers(args[1]))
            if "const" in expected_qualifiers or "thread" not in expected_qualifiers:
                raise error(
                    "requires writable thread storage for expected", source_location
                )
            target_type = self.metal_pointer_pointee_type_once(
                self.expression_metal_type(args[0])
            )
            expected_type = self.metal_pointer_pointee_type_once(
                self.expression_metal_type(args[1])
            )
            if (
                target_type is None
                or expected_type is None
                or not self.is_portable_atomic_scalar_type(self.map_type(target_type))
                or self.map_type(expected_type) != self.map_type(target_type)
            ):
                raise error(
                    "requires matching scalar int, uint or float target and expected storage",
                    source_location,
                )
            if "atomicCompareExchangeWeak" in self.user_function_names:
                raise error(
                    "canonical atomicCompareExchangeWeak conflicts with a source function",
                    source_location,
                )
            target = self.generate_metal_atomic_target(args[0], is_main)
            expected = self.generate_metal_atomic_target(args[1], is_main)
            desired = self.generate_expression(args[2], is_main)
            return f"atomicCompareExchangeWeak({target}, {expected}, {desired})"
        if function_name in {"atomic_load_explicit", "metal::atomic_load_explicit"}:
            if self.metal_user_function_overloads(name):
                return None
            if len(args) != 2:
                raise MetalAtomicLoadLoweringError(
                    "requires a target and explicit memory order", source_location
                )
            order = str(getattr(args[1], "name", args[1])).lstrip(":")
            shadowed_order = order == "memory_order_relaxed" and (
                order in self.current_variable_types
                or order in self.global_variable_types
            )
            if shadowed_order or order not in {
                "memory_order_relaxed",
                "metal::memory_order_relaxed",
            }:
                raise MetalAtomicLoadLoweringError(
                    "only memory_order_relaxed has a portable load contract",
                    source_location,
                )
            qualifiers = set(self.expression_metal_type_qualifiers(args[0]))
            if not qualifiers.intersection({"device", "threadgroup"}):
                raise MetalAtomicLoadLoweringError(
                    "requires tracked device or threadgroup storage", source_location
                )
            pointee = self.metal_pointer_pointee_type_once(
                self.expression_metal_type(args[0])
            )
            if pointee is None or not self.is_portable_atomic_scalar_type(
                self.map_type(pointee)
            ):
                raise MetalAtomicLoadLoweringError(
                    "requires a scalar int, uint or float atomic target",
                    source_location,
                )
            if "atomicLoad" in self.user_function_names:
                raise MetalAtomicLoadLoweringError(
                    "canonical atomicLoad conflicts with a source function",
                    source_location,
                )
            target = self.generate_metal_atomic_target(args[0], is_main)
            return f"atomicLoad({target})"
        if function_name in {"atomic_store_explicit", "metal::atomic_store_explicit"}:
            if self.metal_user_function_overloads(name):
                return None
            if len(args) != 3:
                raise MetalAtomicStoreLoweringError(
                    "requires a target, value and explicit memory order",
                    source_location,
                )
            order = str(getattr(args[2], "name", args[2])).lstrip(":")
            shadowed_order = order == "memory_order_relaxed" and (
                order in self.current_variable_types
                or order in self.global_variable_types
            )
            if shadowed_order or order not in {
                "memory_order_relaxed",
                "metal::memory_order_relaxed",
            }:
                raise MetalAtomicStoreLoweringError(
                    "only memory_order_relaxed has a portable store contract",
                    source_location,
                )
            target = self.generate_metal_atomic_target(args[0], is_main)
            value = self.generate_expression(args[1], is_main)
            return f"atomicStore({target}, {value})"
        unscoped_name = str(name).split("::")[-1]
        if (
            unscoped_name == "atomic_fetch_sub_explicit"
            and unscoped_name not in self.user_function_names
            and not self.metal_user_function_overloads(name)
            and len(args) >= 2
        ):
            return self.metal_atomic_fetch_sub(args, is_main, source_location)
        mapped = self.metal_atomic_intrinsics.get(unscoped_name)
        if mapped is None or unscoped_name in self.user_function_names:
            return None
        if len(args) < 2:
            return None
        # Drop Metal's trailing memory_order argument; keep the atomic target and
        # the operand value.
        target = self.generate_metal_atomic_target(args[0], is_main)
        value = self.generate_expression(args[1], is_main)
        return f"{mapped}({target}, {value})"

    def metal_atomic_fetch_sub(self, args, is_main, source_location=None):
        """Spell ``atomic_fetch_sub_explicit(object, operand, order)`` as ``atomicAdd`` of the negated operand.

        CrossGL, GLSL and HLSL have no atomic subtraction. Metal's integer atomics wrap modulo 2^N (MSL 3.2
        section 6.15.4.5), so adding the two's-complement negation stores and returns the same values; negating a
        float operand is exact.
        """
        element = self.atomic_element_type(
            self.normalized_metal_type(
                self.resolve_type_alias(
                    self.metal_pointer_pointee_type_once(
                        self.expression_metal_type(args[0])
                    )
                    or ""
                )
            )
        )
        element = self.map_type(element) if element else None
        if element not in {"int", "uint", "float"}:
            raise MetalAtomicFetchSubLoweringError(
                "requires an atomic_int, atomic_uint or atomic_float object",
                source_location,
            )
        target = self.generate_metal_atomic_target(args[0], is_main)
        value = self.generate_expression(args[1], is_main)
        operand_type = self.expression_metal_type(args[1])
        if not operand_type or self.map_type(operand_type) != element:
            value = f"{element}({value})"
        if element == "uint":
            return f"atomicAdd({target}, 0u - {value})"
        return f"atomicAdd({target}, -({value}))"

    def generate_metal_atomic_target(self, expr, is_main):
        # Metal addresses an atomic as a pointer (`&buffer[i]` or a pointer var);
        # the CrossGL atomic intrinsics and the GLSL/DirectX/SPIR-V backends
        # address the lvalue. Peel a leading address-of, and emit a structured
        # buffer element as a plain subscript `buffer[index]` (not buffer_load,
        # which the DirectX typed-buffer-atomic lowering does not recognise).
        if isinstance(expr, UnaryOpNode) and getattr(expr, "op", None) == "&":
            expr = expr.operand
        elif self.metal_pointer_pointee_type_once(self.expression_metal_type(expr)):
            expr = ArrayAccessNode(expr, 0)
        if self.is_structured_buffer_element_access(expr):
            buffer = self.generate_without_structured_buffer_index_lowering(
                expr.array, is_main
            )
            index = self.generate_expression(expr.index, is_main)
            return f"{buffer}[{index}]"
        return self.generate_expression(expr, is_main)

    def metal_mem_flag_names(self, args):
        if len(args) != 1:
            return None
        flags = self.collect_metal_mem_flags(args[0])
        return flags or None

    def collect_metal_mem_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_metal_mem_flags(expr.left)
            right = self.collect_metal_mem_flags(expr.right)
            if left is None or right is None:
                return None
            return left | right
        if isinstance(expr, VariableNode):
            name = str(expr.name).split("::")[-1]
            if name.startswith("mem_"):
                return {name}
        return None
