"""Operators, fences and wave operations of the Metal generator: bitcasts and integer bit calls, operator precedence and operands, atomic fences and synchronization, wave operations with their validation and diagnostics, and ray query calls and helpers."""

import re

from ..ast import (
    BinaryOpNode,
    IdentifierNode,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import MetalRefusal
from ..validation import expression_debug_name
from ..wave_operations import (
    WAVE_EXTENDED_OPERATIONS,
    wave_extended_result_type,
)
from .array_utils import split_array_type_suffix
from .metal_wave_extended import metal_wave_extension_call


class MetalWaveOperationMixin:
    """Wave operation members of ``MetalCodeGen``."""

    def metal_native_narrow_bitcast_storage_type(self, value_type):
        raw_type = self.resolve_metal_type_alias(self.type_name_string(value_type))
        if not raw_type:
            return None
        scalar_types = {
            "char": "char",
            "uchar": "uchar",
            "short": "short",
            "ushort": "ushort",
            "int8": "char",
            "int8_t": "char",
            "i8": "char",
            "uint8": "uchar",
            "uint8_t": "uchar",
            "u8": "uchar",
            "int16": "short",
            "int16_t": "short",
            "i16": "short",
            "uint16": "ushort",
            "uint16_t": "ushort",
            "u16": "ushort",
        }
        if raw_type in scalar_types:
            return scalar_types[raw_type]
        vector_match = re.fullmatch(
            r"(?:(?P<prefix>[iu])(?P<bits>8|16)vec|"
            r"(?P<word>char|uchar|short|ushort|u?int(?:8|16)(?:_t)?))"
            r"(?P<width>[234])",
            raw_type,
        )
        if vector_match is None:
            generic_match = re.fullmatch(
                r"vec(?P<width>[234])<(?P<component>[iu](?:8|16))>",
                raw_type,
            )
            if generic_match is None:
                return None
            component = scalar_types.get(generic_match.group("component"))
            return (
                f"{component}{generic_match.group('width')}"
                if component is not None
                else None
            )
        component_name = vector_match.group("word")
        if component_name is None:
            component_name = (
                "uint" if vector_match.group("prefix") == "u" else "int"
            ) + vector_match.group("bits")
        component = scalar_types.get(component_name)
        return (
            f"{component}{vector_match.group('width')}"
            if component is not None
            else None
        )

    def generate_metal_explicit_as_type_call(self, func_name, args):
        target_type = self.metal_explicit_as_type_target(func_name, args)
        if target_type is None:
            return None
        argument = args[0]
        source_type = self.expression_result_type(argument)
        source_value_type = (
            self.reference_referent_type_name(source_type) or source_type
        )
        packed_target = self.metal_explicit_packed_vector_type(target_type)
        packed_source = self.metal_explicit_packed_vector_type(source_value_type)
        if packed_target is not None or packed_source is not None:
            packed_roles = []
            if packed_target is not None:
                packed_roles.append(f"target {packed_target}")
            if packed_source is not None:
                packed_roles.append(f"source {packed_source}")
            raise MetalRefusal(
                f"Metal as_type<{target_type}> cannot preserve packed vector "
                f"storage for {', '.join(packed_roles)}"
            )
        target_info = self.metal_explicit_bitcast_type_info(target_type)
        source_info = self.metal_explicit_bitcast_type_info(source_value_type)
        if target_info is None or source_info is None:
            source_display = self.type_name_string(source_type) or "<unknown>"
            raise MetalRefusal(
                f"Metal as_type<{target_type}> requires statically known numeric "
                f"scalar or vector storage types; inferred source type is "
                f"{source_display}"
            )
        if target_info["totalWidth"] != source_info["totalWidth"]:
            raise MetalRefusal(
                f"Metal as_type<{target_info['type']}> cannot bitcast from "
                f"{source_info['type']}: total widths are "
                f"{target_info['totalWidth']} and {source_info['totalWidth']} bits"
            )
        # A bit cast's operand has its own type; the expected type of the result
        # must not convert the operand's arithmetic to the result's signedness.
        rendered = self.generate_expression_with_expected(argument, None)
        storage_type = self.metal_native_narrow_bitcast_storage_type(source_value_type)
        if storage_type is not None:
            rendered = f"{storage_type}({rendered})"
        bitcast = f"as_type<{target_type}>({rendered})"
        mapped_result_type = self.map_type(target_type)
        if mapped_result_type != target_type:
            return f"{mapped_result_type}({bitcast})"
        return bitcast

    def metal_bitcast_result_type(self, func_name, args):
        if (
            func_name not in self.METAL_BITCAST_FUNCTION_TARGETS
            or func_name in self.user_function_names
            or len(args or []) != 1
        ):
            return None

        argument_type = self.expression_result_type(args[0])
        mapped_argument_type = self.map_type(argument_type)
        target_component = self.METAL_BITCAST_FUNCTION_TARGETS[func_name]
        narrow_match = re.fullmatch(r"(?:half|bfloat)([234])?", mapped_argument_type)
        if narrow_match is not None and target_component in {"int", "uint"}:
            width = narrow_match.group(1) or ""
            narrow_component = "short" if target_component == "int" else "ushort"
            return f"{narrow_component}{width}"
        match = re.fullmatch(r"(?:float|int|uint)([234])?", mapped_argument_type)
        if match is None:
            return target_component

        width = match.group(1) or ""
        return f"{target_component}{width}"

    def generate_metal_bitcast_call(self, func_name, args):
        target_type = self.metal_bitcast_result_type(func_name, args)
        if target_type is None:
            return None

        # The operand keeps its own type (see generate_metal_explicit_as_type_call).
        arg = self.generate_expression_with_expected(args[0], None)
        return f"as_type<{target_type}>({arg})"

    def metal_integer_bit_result_type(self, func_name, args):
        if (
            func_name not in self.METAL_INTEGER_BIT_FUNCTION_TARGETS
            or func_name in self.user_function_names
            or len(args or []) != 1
        ):
            return None

        argument_type = self.expression_result_type(args[0])
        mapped_argument_type = self.map_type(argument_type)
        component_type = self.vector_component_type(mapped_argument_type)
        if component_type not in {"int", "uint"}:
            return None

        return mapped_argument_type

    def generate_metal_integer_bit_call(self, func_name, args):
        if self.metal_integer_bit_result_type(func_name, args) is None:
            return None

        intrinsic_name = self.METAL_INTEGER_BIT_FUNCTION_TARGETS[func_name]
        call_name = (
            f"metal::{intrinsic_name}"
            if self.metal_function_name_is_shadowed(intrinsic_name)
            else intrinsic_name
        )
        arg = self.generate_expression(args[0])
        return f"{call_name}({arg})"

    def binary_precedence(self, operator):
        return self.BINARY_PRECEDENCE.get(operator, 0)

    def binary_child_needs_parentheses(
        self, parent_operator, child, is_right_child=False
    ):
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_operator)
        child_operator = self.map_operator(getattr(child, "op", ""))
        child_precedence = self.binary_precedence(child_operator)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            # Clang's -Wparentheses (part of -Wall) warns on these children even
            # though C++ precedence is unambiguous.  Retain explicit grouping so
            # warning-fatal native Metal builds stay portable.
            return child_operator in self.CLANG_PARENTHESIZED_BINARY_CHILDREN.get(
                parent_operator, ()
            )
        return is_right_child and (
            parent_operator not in self.ASSOCIATIVE_BINARY_OPS
            or child_operator != parent_operator
        )

    def generate_binary_operand(self, expr, parent_operator=None, is_right_child=False):
        rendered = self.generate_expression(expr)
        if (
            parent_operator is not None
            and self.binary_child_needs_parentheses(
                parent_operator, expr, is_right_child
            )
        ) or isinstance(expr, TernaryOpNode):
            return f"({rendered})"
        return rendered

    def generate_unary_operand(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, (BinaryOpNode, TernaryOpNode)):
            return f"({rendered})"
        return rendered

    def atomic_fence_operand_identifier(self, expr):
        name = self.expression_name(expr)
        if name is None:
            return None
        return str(name).lstrip(":").rsplit("::", 1)[-1]

    def atomic_fence_operand_text(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            return (
                f"{self.atomic_fence_operand_text(expr.left)} | "
                f"{self.atomic_fence_operand_text(expr.right)}"
            )
        name = self.atomic_fence_operand_identifier(expr)
        return name if name is not None else expression_debug_name(expr)

    def collect_atomic_fence_memory_flags(self, expr):
        if isinstance(expr, BinaryOpNode) and expr.op == "|":
            left = self.collect_atomic_fence_memory_flags(expr.left)
            right = self.collect_atomic_fence_memory_flags(expr.right)
            if left is None or right is None:
                return None
            return left + right
        name = self.atomic_fence_operand_identifier(expr)
        if name in self.METAL_ATOMIC_FENCE_MEMORY_FLAGS:
            return (name,)
        return None

    def generate_metal_atomic_thread_fence(self, args, *, source_location=None):
        if len(args) != 3:
            raise self.metal_atomic_fence_contract_error(
                "invalid-argument-count",
                args,
                source_location=source_location,
            )

        flags = self.collect_atomic_fence_memory_flags(args[0])
        order = self.atomic_fence_operand_identifier(args[1])
        scope = self.atomic_fence_operand_identifier(args[2])
        if flags is None:
            raise self.metal_atomic_fence_contract_error(
                "unsupported-memory-flags",
                args,
                source_location=source_location,
            )
        if order not in self.METAL_ATOMIC_FENCE_MEMORY_ORDERS:
            raise self.metal_atomic_fence_contract_error(
                "unsupported-memory-order",
                args,
                source_location=source_location,
            )
        if scope not in self.METAL_ATOMIC_FENCE_THREAD_SCOPES:
            raise self.metal_atomic_fence_contract_error(
                "unsupported-thread-scope",
                args,
                source_location=source_location,
            )

        rendered_flags = " | ".join(f"metal::mem_flags::{flag}" for flag in flags)
        return (
            f"metal::atomic_thread_fence({rendered_flags}, "
            f"metal::{order}, metal::{scope})"
        )

    def synchronization_function_call(self, func_name, args, *, source_location=None):
        if func_name in self.user_function_names:
            return None
        if func_name == "atomicThreadFence":
            return self.generate_metal_atomic_thread_fence(
                args, source_location=source_location
            )
        if args:
            return None
        return {
            "barrier": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "workgroupBarrier": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "workgroupExecutionBarrier": "threadgroup_barrier(mem_flags::mem_none)",
            "subgroupExecutionBarrier": "simdgroup_barrier(mem_flags::mem_none)",
            # The GL_KHR_shader_subgroup barriers order buffer, image and shared memory, or one of them, for the
            # subgroup; Metal spells each as a SIMD-group barrier over the same memory, as it spells workgroup
            # fences as threadgroup barriers.
            "subgroupBarrier": (
                "simdgroup_barrier(mem_flags::mem_device | "
                "mem_flags::mem_threadgroup | mem_flags::mem_texture)"
            ),
            "subgroupMemoryBarrier": (
                "simdgroup_barrier(mem_flags::mem_device | "
                "mem_flags::mem_threadgroup | mem_flags::mem_texture)"
            ),
            "subgroupMemoryBarrierBuffer": "simdgroup_barrier(mem_flags::mem_device)",
            "subgroupMemoryBarrierShared": (
                "simdgroup_barrier(mem_flags::mem_threadgroup)"
            ),
            "subgroupMemoryBarrierImage": "simdgroup_barrier(mem_flags::mem_texture)",
            # groupMemoryBarrier and memoryBarrier order buffer, image and shared memory (GLSL 4.60 section
            # 8.16); deviceMemoryBarrier orders buffers and textures (HLSL DeviceMemoryBarrier).
            "groupMemoryBarrier": (
                "threadgroup_barrier(mem_flags::mem_device | "
                "mem_flags::mem_threadgroup | mem_flags::mem_texture)"
            ),
            "memoryBarrierShared": "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "memoryBarrierBuffer": "threadgroup_barrier(mem_flags::mem_device)",
            "deviceMemoryBarrier": (
                "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_texture)"
            ),
            "memoryBarrierImage": "threadgroup_barrier(mem_flags::mem_texture)",
            "memoryBarrier": (
                "threadgroup_barrier(mem_flags::mem_device | "
                "mem_flags::mem_threadgroup | mem_flags::mem_texture)"
            ),
            "allMemoryBarrier": (
                "threadgroup_barrier(mem_flags::mem_device | "
                "mem_flags::mem_threadgroup | mem_flags::mem_texture)"
            ),
        }.get(func_name)

    def generate_metal_wave_op_expression(self, node):
        result = self.generate_metal_wave_operation(node.operation, node.arguments)
        if result is None:
            return self.metal_wave_diagnostic_expression(
                node.operation, node.arguments, "is not recognized by the Metal backend"
            )
        return result

    def generate_metal_wave_operation(self, operation, arguments):
        extension = metal_wave_extension_call(self, operation, arguments)
        if extension is not None:
            return extension
        expected_arity = self.METAL_WAVE_INTRINSIC_ARITIES.get(operation)
        if expected_arity is None:
            return None

        actual_arity = len(arguments)
        accepted_arities = (
            (3, 4) if operation == "WaveShuffleAndFillUp" else (expected_arity,)
        )
        if actual_arity not in accepted_arities:
            expected = " or ".join(str(arity) for arity in accepted_arities)
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                f"expects {expected} arguments, got {actual_arity}",
            )

        unsupported_reason = self.METAL_WAVE_UNSUPPORTED_OPERATIONS.get(operation)
        if unsupported_reason is not None:
            return self.metal_wave_diagnostic_expression(
                operation, arguments, unsupported_reason
            )

        bfloat_lane_move = self.metal_wave_bfloat_lane_move(operation, arguments)
        if bfloat_lane_move is not None:
            return bfloat_lane_move

        type_diagnostic = self.metal_wave_type_diagnostic(operation, arguments)
        if type_diagnostic is not None:
            return type_diagnostic

        if operation == "WaveGetLaneIndex":
            lane_index_parameter = self.current_metal_wave_lane_index_parameter
            if lane_index_parameter is not None:
                return lane_index_parameter
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                "requires a compute-stage thread_index_in_simdgroup value",
            )
        if operation == "WaveGetLaneCount":
            lane_count_parameter = self.current_metal_wave_lane_count_parameter
            if lane_count_parameter is not None:
                return lane_count_parameter
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                "requires a compute-stage threads_per_simdgroup value",
            )
        if operation == "WaveIsFirstLane":
            return "simd_is_first()"
        if operation == "WaveActiveCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"uint(popcount(simd_vote::vote_t(simd_ballot({predicate}))))"
        if operation == "WavePrefixCountBits":
            predicate = self.generate_expression(arguments[0])
            return f"simd_prefix_exclusive_sum(({predicate}) ? 1u : 0u)"
        if operation == "WaveActiveBallot":
            self.required_metal_wave_ballot_helper = True
            predicate = self.generate_expression(arguments[0])
            return f"__crossgl_metal_wave_ballot({predicate})"
        if operation == "WaveActiveAllEqual":
            value = self.generate_expression(arguments[0])
            mapped_type, component_type, _array_suffix = (
                self.metal_wave_argument_mapped_type(arguments[0])
            )
            equality = f"{value} == simd_broadcast_first({value})"
            if (
                mapped_type is not None
                and component_type is not None
                and mapped_type != component_type
            ):
                equality = f"all({equality})"
            return f"simd_all({equality})"
        if operation == "WaveMatch":
            lane_count_parameter = self.current_metal_wave_lane_count_parameter
            if lane_count_parameter is None:
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    "requires a compute-stage threads_per_simdgroup value",
                )
            self.required_metal_wave_match_helper = True
            value = self.generate_expression(arguments[0])
            return f"__crossgl_metal_wave_match({value}, {lane_count_parameter})"
        if operation in self.METAL_WAVE_MULTI_PREFIX_INTRINSICS:
            lane_index_parameter = self.current_metal_wave_lane_index_parameter
            lane_count_parameter = self.current_metal_wave_lane_count_parameter
            if lane_index_parameter is None or lane_count_parameter is None:
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    (
                        "requires compute-stage thread_index_in_simdgroup "
                        "and threads_per_simdgroup values"
                    ),
                )
            self.required_metal_wave_ballot_helper = True
            self.required_metal_wave_mask_contains_helper = True
            self.required_metal_wave_multi_prefix_helpers.add(operation)
            value = self.generate_expression(arguments[0])
            mask = self.generate_expression(arguments[1])
            helper = self.METAL_WAVE_MULTI_PREFIX_HELPERS[operation]
            return (
                f"{helper}({value}, {mask}, {lane_index_parameter}, "
                f"{lane_count_parameter})"
            )
        if operation == "WaveReadLaneAt":
            value = self.generate_expression(arguments[0])
            lane = self.generate_expression(arguments[1])
            # Broadcast requires a subgroup-uniform index; arbitrary lane reads do not.
            intrinsic = (
                "simd_broadcast"
                if self.literal_int_value(arguments[1]) is not None
                else "simd_shuffle"
            )
            return f"{intrinsic}({value}, ushort({lane}))"
        if operation in self.METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS:
            value = self.generate_expression(arguments[0])
            fill = self.generate_expression(arguments[1])
            delta = self.generate_expression(arguments[2])
            modulo = (
                f", ushort({self.generate_expression(arguments[3])})"
                if len(arguments) == 4
                else ""
            )
            mapped_type, component_type, _array_suffix = (
                self.metal_wave_argument_mapped_type(arguments[0])
            )
            boolean_payload = (
                mapped_type == "bool"
                or component_type == "bool"
                or all(
                    self.metal_wave_expression_is_boolean(argument)
                    for argument in arguments[:2]
                )
            )
            if boolean_payload:
                component_count = self.value_component_count(mapped_type) or 1
                payload_type = (
                    "uint" if component_count == 1 else f"uint{component_count}"
                )
                zero = "0u" if component_count == 1 else f"{payload_type}(0u)"
                return (
                    f"(simd_shuffle_and_fill_up({payload_type}({value}), "
                    f"{payload_type}({fill}), ushort({delta}){modulo}) != {zero})"
                )
            result = (
                f"simd_shuffle_and_fill_up({value}, {fill}, ushort({delta}){modulo})"
            )
            expected_type = self.map_type(self.current_expression_expected_type)
            if expected_type == "bool":
                return (
                    f"({result} != "
                    f"{self.diagnostic_zero_value_for_type(mapped_type)})"
                )
            return result
        if operation == "QuadReadLaneAt":
            value = self.generate_expression(arguments[0])
            lane = self.generate_expression(arguments[1])
            return f"quad_broadcast({value}, ushort({lane}))"
        if operation == "QuadReadAcrossX":
            value = self.generate_expression(arguments[0])
            return f"quad_shuffle_xor({value}, ushort(1))"
        if operation == "QuadReadAcrossY":
            value = self.generate_expression(arguments[0])
            return f"quad_shuffle_xor({value}, ushort(2))"
        if operation == "QuadReadAcrossDiagonal":
            value = self.generate_expression(arguments[0])
            return f"quad_shuffle_xor({value}, ushort(3))"

        mapped = self.METAL_WAVE_DIRECT_MAPPINGS.get(operation)
        if mapped is None:
            return self.metal_wave_diagnostic_expression(
                operation, arguments, "is not recognized by the Metal backend"
            )
        args = ", ".join(self.generate_expression(arg) for arg in arguments)
        call_name = (
            f"metal::{mapped}"
            if self.metal_function_name_is_shadowed(mapped)
            else mapped
        )
        return f"{call_name}({args})"

    METAL_WAVE_BFLOAT_LANE_MOVES = {
        "WaveReadLaneAt",
        "WaveReadLaneFirst",
        "WaveShuffleDown",
        "WaveShuffleUp",
        "WaveShuffleXor",
        "WaveShuffleAndFillUp",
        "QuadReadAcrossX",
        "QuadReadAcrossY",
        "QuadReadAcrossDiagonal",
        "QuadReadLaneAt",
    }

    def metal_wave_bfloat_lane_move(self, operation, arguments):
        """Move bfloat lanes as their exact bits, as MLX's bf16 overloads do.

        Metal's SIMD-group shuffles do not accept ``bfloat`` operands; moving the
        ``ushort`` representation is lossless and keeps NaN payloads intact.
        """
        if operation not in self.METAL_WAVE_BFLOAT_LANE_MOVES or not arguments:
            return None
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(arguments[0])
        )
        if mapped_type is None or array_suffix:
            return None
        if self.is_scalar_value_type(mapped_type):
            component_type = mapped_type
        if component_type != "bfloat" or self.metal_wave_mapped_type_is_matrix(
            mapped_type
        ):
            return None
        count = self.value_component_count(mapped_type) or 1
        bits_type = "ushort" if count == 1 else f"ushort{count}"

        def bits(argument):
            return f"as_type<{bits_type}>({self.generate_expression(argument)})"

        if operation == "WaveShuffleAndFillUp":
            fill_type, _fill_component, _fill_suffix = (
                self.metal_wave_argument_mapped_type(arguments[1])
            )
            if fill_type is not None and fill_type != mapped_type:
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    "value and fill arguments must have matching types, got "
                    f"{mapped_type} and {fill_type}",
                )
            for argument, role in zip(arguments[2:], ("delta", "modulo")):
                diagnostic = self.metal_wave_validate_lane_argument(
                    operation, argument, role
                )
                if diagnostic is not None:
                    return diagnostic
            delta = self.generate_expression(arguments[2])
            modulo = (
                f", ushort({self.generate_expression(arguments[3])})"
                if len(arguments) == 4
                else ""
            )
            inner = (
                f"simd_shuffle_and_fill_up({bits(arguments[0])}, "
                f"{bits(arguments[1])}, ushort({delta}){modulo})"
            )
        elif operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            diagnostic = self.metal_wave_validate_lane_argument(
                operation, arguments[1], "lane index"
            )
            if diagnostic is None and operation == "QuadReadLaneAt":
                diagnostic = self.metal_wave_validate_quad_lane_range(
                    operation, arguments[1]
                )
            if diagnostic is not None:
                return diagnostic
            lane = self.generate_expression(arguments[1])
            if operation == "QuadReadLaneAt":
                intrinsic = "quad_broadcast"
            else:
                intrinsic = (
                    "simd_broadcast"
                    if self.literal_int_value(arguments[1]) is not None
                    else "simd_shuffle"
                )
            inner = f"{intrinsic}({bits(arguments[0])}, ushort({lane}))"
        elif operation.startswith("QuadReadAcross"):
            mask = {"X": 1, "Y": 2, "Diagonal": 3}[operation[len("QuadReadAcross") :]]
            inner = f"quad_shuffle_xor({bits(arguments[0])}, ushort({mask}))"
        elif operation == "WaveReadLaneFirst":
            inner = f"simd_broadcast_first({bits(arguments[0])})"
        else:
            diagnostic = self.metal_wave_validate_lane_argument(
                operation, arguments[1], "delta"
            )
            if diagnostic is not None:
                return diagnostic
            intrinsic = self.METAL_WAVE_DIRECT_MAPPINGS[operation]
            inner = (
                f"{intrinsic}({bits(arguments[0])}, "
                f"ushort({self.generate_expression(arguments[1])}))"
            )
        return f"as_type<{mapped_type}>({inner})"

    def metal_wave_result_type(self, operation, arguments):
        if operation in WAVE_EXTENDED_OPERATIONS:
            return wave_extended_result_type(
                operation,
                self.expression_result_type(arguments[0]) if arguments else None,
            )
        if operation in self.METAL_WAVE_UINT_RESULT_INTRINSICS:
            return "uint"
        if operation in self.METAL_WAVE_BOOL_RESULT_INTRINSICS:
            return "bool"
        if operation in self.METAL_WAVE_UINT4_RESULT_INTRINSICS:
            return "uint4"
        if operation in self.METAL_WAVE_VALUE_RESULT_INTRINSICS and arguments:
            return self.expression_result_type(arguments[0])
        return None

    def metal_wave_argument_mapped_type(self, argument):
        literal_type = getattr(getattr(argument, "literal_type", None), "name", None)
        argument_type = literal_type or self.expression_result_type(argument)
        if argument_type is None:
            return None, None, None
        mapped_type = self.map_type(argument_type)
        component_type = self.vector_component_type(mapped_type)
        return mapped_type, component_type, split_array_type_suffix(mapped_type)[1]

    def metal_wave_validate_predicate_argument(self, operation, argument):
        mapped_type, _component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if array_suffix or mapped_type != "bool":
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                ("predicate argument must be scalar bool, got " f"{mapped_type}"),
            )
        return None

    def metal_wave_validate_value_argument(
        self, operation, argument, allowed_components, description
    ):
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if array_suffix or component_type not in allowed_components:
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"value argument must be {description}, got {mapped_type}",
            )
        return None

    def metal_wave_validate_lane_argument(self, operation, argument, role):
        mapped_type, _component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if array_suffix or mapped_type not in {"int", "uint"}:
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"{role} must be scalar int or uint, got {mapped_type}",
            )
        return None

    def metal_wave_validate_match_argument(self, operation, argument):
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if (
            array_suffix
            or mapped_type != component_type
            or component_type not in self.METAL_WAVE_NUMERIC_COMPONENT_TYPES
        ):
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"value argument must be numeric scalar, got {mapped_type}",
            )
        return None

    def metal_wave_mapped_type_is_matrix(self, mapped_type):
        return self.is_matrix_value_type(mapped_type)

    def metal_wave_validate_non_matrix_value_argument(
        self, operation, argument, allowed_components, description
    ):
        mapped_type, component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        if (
            array_suffix
            or self.metal_wave_mapped_type_is_matrix(mapped_type)
            or component_type not in allowed_components
        ):
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"value argument must be {description}, got {mapped_type}",
            )
        return None

    def metal_wave_validate_multi_prefix_mask_argument(
        self, operation, argument, value_argument=None
    ):
        mapped_type, _component_type, array_suffix = (
            self.metal_wave_argument_mapped_type(argument)
        )
        if mapped_type is None:
            return None
        diagnostic_arguments = [value_argument] if value_argument is not None else []
        diagnostic_arguments.append(argument)
        if array_suffix or mapped_type != "uint4":
            return self.metal_wave_diagnostic_expression(
                operation,
                diagnostic_arguments,
                f"mask argument must be uint4, got {mapped_type}",
            )
        return None

    def metal_wave_validate_quad_lane_range(self, operation, argument):
        lane_index = self.literal_int_value(argument, self.literal_int_constants)
        if lane_index is None:
            return None
        if not 0 <= lane_index <= 3:
            return self.metal_wave_diagnostic_expression(
                operation,
                [argument],
                f"quad lane index must be in the range 0 to 3, got {lane_index}",
            )
        return None

    def metal_wave_validate_shuffle_and_fill_arguments(self, operation, arguments):
        allowed_components = self.METAL_WAVE_NUMERIC_COMPONENT_TYPES | {"bool"}
        mapped_types = []
        for argument, role in ((arguments[0], "value"), (arguments[1], "fill")):
            mapped_type, component_type, array_suffix = (
                self.metal_wave_argument_mapped_type(argument)
            )
            mapped_types.append(mapped_type)
            if mapped_type is None:
                continue
            if self.is_scalar_value_type(mapped_type):
                component_type = mapped_type
            if (
                array_suffix
                or self.metal_wave_mapped_type_is_matrix(mapped_type)
                or component_type not in allowed_components
            ):
                return self.metal_wave_diagnostic_expression(
                    operation,
                    arguments,
                    f"{role} argument must be a basic scalar or vector, got "
                    f"{mapped_type}",
                )

        value_type, fill_type = mapped_types
        if value_type is not None and fill_type is not None and value_type != fill_type:
            return self.metal_wave_diagnostic_expression(
                operation,
                arguments,
                "value and fill arguments must have matching types, got "
                f"{value_type} and {fill_type}",
            )
        for argument, role in zip(arguments[2:], ("delta", "modulo")):
            diagnostic = self.metal_wave_validate_lane_argument(
                operation, argument, role
            )
            if diagnostic is not None:
                return diagnostic
        return None

    def metal_wave_expression_is_boolean(self, expression):
        expression_type = self.expression_result_type(expression)
        mapped_type = self.map_type(expression_type) if expression_type else None
        if mapped_type == "bool" or self.vector_component_type(mapped_type) == "bool":
            return True
        literal_type = getattr(getattr(expression, "literal_type", None), "name", None)
        return self.map_type(literal_type) == "bool" if literal_type else False

    def metal_wave_type_diagnostic(self, operation, arguments):
        if operation in self.METAL_WAVE_SHUFFLE_AND_FILL_INTRINSICS:
            return self.metal_wave_validate_shuffle_and_fill_arguments(
                operation, arguments
            )
        if operation == "WaveMultiPrefixCountBits":
            diagnostic = self.metal_wave_validate_predicate_argument(
                operation, arguments[0]
            )
            if diagnostic is not None:
                return diagnostic
            return self.metal_wave_validate_multi_prefix_mask_argument(
                operation, arguments[1], arguments[0]
            )
        if operation in self.METAL_WAVE_MULTI_PREFIX_NUMERIC_INTRINSICS:
            diagnostic = self.metal_wave_validate_non_matrix_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
            if diagnostic is not None:
                return diagnostic
            return self.metal_wave_validate_multi_prefix_mask_argument(
                operation, arguments[1], arguments[0]
            )
        if operation in self.METAL_WAVE_MULTI_PREFIX_INTEGER_INTRINSICS:
            diagnostic = self.metal_wave_validate_non_matrix_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_INTEGER_COMPONENT_TYPES,
                "integer scalar or vector",
            )
            if diagnostic is not None:
                return diagnostic
            return self.metal_wave_validate_multi_prefix_mask_argument(
                operation, arguments[1], arguments[0]
            )
        if operation in self.METAL_WAVE_BOOL_ARGUMENT_INTRINSICS:
            return self.metal_wave_validate_predicate_argument(operation, arguments[0])
        if operation in self.METAL_WAVE_NUMERIC_VALUE_INTRINSICS:
            return self.metal_wave_validate_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
        if operation in self.METAL_WAVE_INTEGER_VALUE_INTRINSICS:
            return self.metal_wave_validate_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_INTEGER_COMPONENT_TYPES,
                "integer scalar or vector",
            )
        if operation == "WaveActiveAllEqual":
            return self.metal_wave_validate_non_matrix_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric scalar or vector",
            )
        if operation in self.METAL_WAVE_SIMDGROUP_VALUE_INTRINSICS:
            diagnostic = self.metal_wave_validate_value_argument(
                operation,
                arguments[0],
                self.METAL_WAVE_NUMERIC_COMPONENT_TYPES,
                "numeric or integer scalar or vector",
            )
            if diagnostic is not None:
                return diagnostic
        if operation == "WaveMatch":
            diagnostic = self.metal_wave_validate_match_argument(
                operation, arguments[0]
            )
            if diagnostic is not None:
                return diagnostic

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            diagnostic = self.metal_wave_validate_lane_argument(
                operation, arguments[1], "lane index"
            )
            if diagnostic is not None:
                return diagnostic
            if operation == "QuadReadLaneAt":
                return self.metal_wave_validate_quad_lane_range(operation, arguments[1])
        return None

    def metal_wave_default_value(self, operation, arguments):
        result_type = (
            self.current_expression_expected_type
            or self.metal_wave_result_type(operation, arguments)
        )
        if result_type:
            return self.diagnostic_zero_value_for_type(result_type)
        if operation in self.METAL_WAVE_BOOL_RESULT_INTRINSICS:
            return "false"
        if operation in self.METAL_WAVE_UINT4_RESULT_INTRINSICS:
            return "uint4(0)"
        return "0u"

    def metal_wave_diagnostic_expression(self, operation, arguments, reason):
        return self.refuse_lowering(
            f"wave intrinsic: {operation} {reason}",
            self.metal_wave_default_value(operation, arguments),
        )

    def generate_metal_ray_query_call(self, operation, query_expr, arguments):
        self.require_metal_ray_query_runtime()
        self.refuse_metal_ray_query(operation)
        self.required_metal_ray_query_helpers.add(str(operation))
        helper_name = self.metal_ray_query_helper_name(operation)
        args = [self.generate_expression(query_expr)]
        args.extend(self.generate_expression(arg) for arg in arguments or [])
        return f"{helper_name}({', '.join(args)})"

    def refuse_metal_ray_query(self, operation=None):
        """Refuse inline ray queries: the ``cgl_ray_query_*`` helpers trace nothing and return default values."""
        construct = f"RayQuery.{operation}" if operation else "RayQuery"
        self.refuse_lowering(
            f"ray query: {construct} has no Metal intersection_query lowering"
        )

    def metal_ray_query_default_value(self, return_type):
        if return_type == "bool":
            return "false"
        if return_type == "float":
            return "0.0"
        if return_type == "float2":
            return "float2(0.0)"
        if return_type == "float3":
            return "float3(0.0)"
        if return_type == "float3x4":
            return "float3x4(0.0)"
        return "0u"

    def generate_metal_ray_query_helpers(self):
        if not (
            self.required_metal_ray_query_runtime
            or self.required_metal_ray_desc_runtime
            or self.required_metal_ray_query_helpers
        ):
            return ""

        code = ""
        if self.required_metal_ray_desc_runtime:
            code += (
                "struct CglRayDesc {\n"
                "    float3 origin;\n"
                "    float t_min;\n"
                "    float3 direction;\n"
                "    float t_max;\n"
                "\n"
                "    CglRayDesc() {}\n"
                "\n"
                "    CglRayDesc(float3 origin_value, float t_min_value, "
                "float3 direction_value, float t_max_value)\n"
                "        : origin(origin_value),\n"
                "          t_min(t_min_value),\n"
                "          direction(direction_value),\n"
                "          t_max(t_max_value) {}\n"
                "};\n\n"
            )
        if (
            self.required_metal_ray_query_runtime
            or self.required_metal_ray_query_helpers
        ):
            code += "struct CglRayQuery {\n" "    uint state;\n" "};\n\n"

        for operation in sorted(self.required_metal_ray_query_helpers):
            return_type = self.metal_ray_query_method_return_type(operation)
            if return_type is None:
                continue
            helper_name = self.metal_ray_query_helper_name(operation)
            if operation == "TraceRayInline":
                code += (
                    "template <typename Accel, typename Flags, typename Mask, "
                    "typename Ray>\n"
                    f"inline void {helper_name}(\n"
                    "    CglRayQuery query,\n"
                    "    Accel acceleration_structure,\n"
                    "    Flags ray_flags,\n"
                    "    Mask instance_mask,\n"
                    "    Ray ray) {\n"
                    "    (void)query;\n"
                    "    (void)acceleration_structure;\n"
                    "    (void)ray_flags;\n"
                    "    (void)instance_mask;\n"
                    "    (void)ray;\n"
                    "}\n\n"
                    "template <typename Accel, typename Flags, typename Mask, "
                    "typename Origin, typename MinDistance, typename Direction, "
                    "typename MaxDistance>\n"
                    f"inline void {helper_name}(\n"
                    "    CglRayQuery query,\n"
                    "    Accel acceleration_structure,\n"
                    "    Flags ray_flags,\n"
                    "    Mask instance_mask,\n"
                    "    Origin origin,\n"
                    "    MinDistance min_distance,\n"
                    "    Direction direction,\n"
                    "    MaxDistance max_distance) {\n"
                    "    (void)query;\n"
                    "    (void)acceleration_structure;\n"
                    "    (void)ray_flags;\n"
                    "    (void)instance_mask;\n"
                    "    (void)origin;\n"
                    "    (void)min_distance;\n"
                    "    (void)direction;\n"
                    "    (void)max_distance;\n"
                    "}\n\n"
                )
                continue
            if operation in {
                "CommitProceduralPrimitiveHit",
                "CandidateTriangleVertexPositions",
                "CommittedTriangleVertexPositions",
            }:
                code += (
                    "template <typename Arg>\n"
                    f"inline void {helper_name}(CglRayQuery query, "
                    "const Arg& arg) {\n"
                    "    (void)query;\n"
                    "    (void)arg;\n"
                    "}\n\n"
                )
                continue
            if return_type == "void":
                code += (
                    f"inline void {helper_name}(CglRayQuery query) {{\n"
                    "    (void)query;\n"
                    "}\n\n"
                )
                continue
            code += (
                f"inline {return_type} {helper_name}(CglRayQuery query) {{\n"
                "    (void)query;\n"
                f"    return {self.metal_ray_query_default_value(return_type)};\n"
                "}\n\n"
            )
        return code

    def generate_metal_wave_helpers(self):
        code = ""
        if self.required_metal_wave_ballot_helper:
            code += (
                "static uint4 __crossgl_metal_wave_ballot(bool predicate) {\n"
                "    simd_vote::vote_t mask = simd_vote::vote_t(simd_ballot(predicate));\n"
                "    return uint4(\n"
                "        uint(mask & simd_vote::vote_t(0xffffffffu)),\n"
                "        uint((mask >> simd_vote::vote_t(32u)) & "
                "simd_vote::vote_t(0xffffffffu)),\n"
                "        0u,\n"
                "        0u);\n"
                "}\n\n"
            )
        if self.required_metal_wave_match_helper:
            code += (
                "template <typename T>\n"
                "static uint4 __crossgl_metal_wave_match(T value, uint laneCount) {\n"
                "    uint4 mask = uint4(0u);\n"
                "    for (uint lane = 0u; lane < laneCount; ++lane) {\n"
                "        if (simd_broadcast(value, ushort(lane)) == value) {\n"
                "            if (lane < 32u) {\n"
                "                mask.x |= (1u << lane);\n"
                "            } else if (lane < 64u) {\n"
                "                mask.y |= (1u << (lane - 32u));\n"
                "            } else if (lane < 96u) {\n"
                "                mask.z |= (1u << (lane - 64u));\n"
                "            } else {\n"
                "                mask.w |= (1u << (lane - 96u));\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    return mask;\n"
                "}\n\n"
            )
        if self.required_metal_wave_mask_contains_helper:
            code += (
                "static bool __crossgl_metal_wave_mask_contains(uint4 mask, uint lane) {\n"
                "    if (lane < 32u) {\n"
                "        return (mask.x & (1u << lane)) != 0u;\n"
                "    }\n"
                "    if (lane < 64u) {\n"
                "        return (mask.y & (1u << (lane - 32u))) != 0u;\n"
                "    }\n"
                "    if (lane < 96u) {\n"
                "        return (mask.z & (1u << (lane - 64u))) != 0u;\n"
                "    }\n"
                "    return (mask.w & (1u << (lane - 96u))) != 0u;\n"
                "}\n\n"
            )
        for operation in self.METAL_WAVE_MULTI_PREFIX_HELPERS:
            if operation not in self.required_metal_wave_multi_prefix_helpers:
                continue
            helper_name = self.METAL_WAVE_MULTI_PREFIX_HELPERS[operation]
            if operation == "WaveMultiPrefixCountBits":
                code += (
                    f"static uint {helper_name}(bool value, uint4 mask, uint laneIndex, "
                    "uint laneCount) {\n"
                    "    uint laneValue = value ? 1u : 0u;\n"
                    "    uint result = 0u;\n"
                    "    uint4 activeMask = __crossgl_metal_wave_ballot(true);\n"
                    "    uint limit = min(laneIndex, laneCount);\n"
                    "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                    "        if (__crossgl_metal_wave_mask_contains(mask, lane) && "
                    "__crossgl_metal_wave_mask_contains(activeMask, lane)) {\n"
                    "            result += simd_broadcast(laneValue, ushort(lane));\n"
                    "        }\n"
                    "    }\n"
                    "    return result;\n"
                    "}\n\n"
                )
                continue
            if operation == "WaveMultiPrefixSum":
                identity = "T(0)"
                assignment = "+="
            elif operation == "WaveMultiPrefixProduct":
                identity = "T(1)"
                assignment = "*="
            elif operation == "WaveMultiPrefixBitAnd":
                identity = "~T(0)"
                assignment = "&="
            elif operation == "WaveMultiPrefixBitOr":
                identity = "T(0)"
                assignment = "|="
            else:
                identity = "T(0)"
                assignment = "^="
            code += (
                "template <typename T>\n"
                f"static T {helper_name}(T value, uint4 mask, uint laneIndex, "
                "uint laneCount) {\n"
                f"    T result = {identity};\n"
                "    uint4 activeMask = __crossgl_metal_wave_ballot(true);\n"
                "    uint limit = min(laneIndex, laneCount);\n"
                "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                "        if (__crossgl_metal_wave_mask_contains(mask, lane) && "
                "__crossgl_metal_wave_mask_contains(activeMask, lane)) {\n"
                f"            result {assignment} simd_broadcast(value, ushort(lane));\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def generate_metal_inverse_helpers(self):
        code = ""
        required = getattr(self, "required_metal_inverse_helpers", set())
        if "float2x2" in required:
            code += (
                "static inline float2x2 __crossgl_inverse_float2x2(float2x2 m) {\n"
                "    float det = m[0][0] * m[1][1] - m[1][0] * m[0][1];\n"
                "    if (abs(det) <= 1.0e-8) {\n"
                "        return float2x2(1.0);\n"
                "    }\n"
                "    return float2x2(\n"
                "        float2(m[1][1], -m[0][1]),\n"
                "        float2(-m[1][0], m[0][0])) / det;\n"
                "}\n\n"
            )
        if "float3x3" in required:
            code += (
                "static inline float3x3 __crossgl_inverse_float3x3(float3x3 m) {\n"
                "    float3 c0 = m[0];\n"
                "    float3 c1 = m[1];\n"
                "    float3 c2 = m[2];\n"
                "    float3 r0 = cross(c1, c2);\n"
                "    float3 r1 = cross(c2, c0);\n"
                "    float3 r2 = cross(c0, c1);\n"
                "    float det = dot(c0, r0);\n"
                "    if (abs(det) <= 1.0e-8) {\n"
                "        return float3x3(1.0);\n"
                "    }\n"
                "    return transpose(float3x3(r0, r1, r2)) / det;\n"
                "}\n\n"
            )
        if "float4x4" in required:
            code += (
                "static inline float __crossgl_det3_float4x4(\n"
                "    float a00, float a01, float a02,\n"
                "    float a10, float a11, float a12,\n"
                "    float a20, float a21, float a22) {\n"
                "    return a00 * (a11 * a22 - a12 * a21)\n"
                "         - a01 * (a10 * a22 - a12 * a20)\n"
                "         + a02 * (a10 * a21 - a11 * a20);\n"
                "}\n\n"
                "static inline float __crossgl_cofactor_float4x4(\n"
                "    float4x4 m, int column, int row) {\n"
                "    float values[9];\n"
                "    int index = 0;\n"
                "    for (int c = 0; c < 4; ++c) {\n"
                "        if (c == column) {\n"
                "            continue;\n"
                "        }\n"
                "        for (int r = 0; r < 4; ++r) {\n"
                "            if (r == row) {\n"
                "                continue;\n"
                "            }\n"
                "            values[index++] = m[c][r];\n"
                "        }\n"
                "    }\n"
                "    float minor_det = __crossgl_det3_float4x4(\n"
                "        values[0], values[1], values[2],\n"
                "        values[3], values[4], values[5],\n"
                "        values[6], values[7], values[8]);\n"
                "    return ((column + row) & 1) ? -minor_det : minor_det;\n"
                "}\n\n"
                "static inline float4x4 __crossgl_inverse_float4x4(float4x4 m) {\n"
                "    float det = determinant(m);\n"
                "    if (abs(det) <= 1.0e-8) {\n"
                "        return float4x4(1.0);\n"
                "    }\n"
                "    float4x4 result;\n"
                "    for (int c = 0; c < 4; ++c) {\n"
                "        for (int r = 0; r < 4; ++r) {\n"
                "            result[c][r] = __crossgl_cofactor_float4x4(m, r, c) / det;\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def generate_ray_tracing_op_expression(self, expr):
        raw_args = self.normalized_metal_intrinsic_args(expr.arguments)
        rendered_args = [self.generate_expression(arg) for arg in raw_args]
        if expr.operation == "TraceRay":
            trace_ray = self.generate_metal_trace_ray(raw_args, rendered_args)
            if trace_ray is not None:
                return trace_ray
            return self.unsupported_metal_ray_tracing_intrinsic(
                "TraceRay",
                "expected acceleration structure, flags, mask, SBT offsets, "
                "origin, tmin, direction, tmax, and payload location",
            )

        if expr.operation == "CallShader":
            call_shader = self.generate_metal_call_shader(raw_args, rendered_args)
            if call_shader is not None:
                return call_shader
            return self.unsupported_metal_ray_tracing_intrinsic(
                "CallShader",
                "requires one visible_function_table resource, or an explicit "
                "visible_function_table argument",
            )

        unsupported_reasons = {
            "ReportHit": (
                "intersection acceptance is expressed through the Metal intersection return value"
            ),
            "AcceptHitAndEndSearch": (
                "Metal hit acceptance is controlled by intersection results"
            ),
            "IgnoreHit": "Metal does not expose a direct ignore-intersection intrinsic",
        }
        reason = unsupported_reasons.get(expr.operation)
        if reason is not None:
            return self.unsupported_metal_ray_tracing_intrinsic(expr.operation, reason)

        return f"{expr.operation}({', '.join(rendered_args)})"

    def normalized_metal_intrinsic_args(self, args):
        normalized = []
        index = 0
        while index < len(args):
            arg = args[index]
            if self.is_unary_deref_marker(arg) and index + 1 < len(args):
                normalized.append(UnaryOpNode("*", args[index + 1]))
                index += 2
                continue
            normalized.append(arg)
            index += 1
        return normalized

    def is_unary_deref_marker(self, arg):
        return isinstance(arg, IdentifierNode) and getattr(arg, "name", None) == "*"
