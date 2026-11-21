"""Function parameters and binary operations of the SPIR-V generator: parameter creation and function ends, bool arithmetic, bitwise and arithmetic operand alignment, matrix operations and shape checks, no-contraction decorations, precision, bool values, complex struct arithmetic, bfloat16 conversions, integer type queries and select operations."""

import struct
from typing import (
    Optional,
    Tuple,
)

from .spirv_values import (
    SpirvId,
    SpirvType,
)


class SpirvBinaryOperationMixin:
    """Binary operation members of ``VulkanSPIRVCodeGen``."""

    def create_function_parameter(
        self, param_type: SpirvId, name: Optional[str] = None
    ) -> SpirvId:
        """Create a function parameter."""
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpFunctionParameter %{param_type.id}")

        if name:
            self.emit(f'OpName %{id_value} "{name}"')

        spirv_id = SpirvId(id_value, param_type.type, name)
        self.value_types[id_value] = param_type
        return spirv_id

    def begin_block(self) -> SpirvId:
        """Begin a new basic block."""
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpLabel")

        self.current_label = id_value
        return SpirvId(id_value, SpirvType("label"))

    def end_function(self):
        """End the current function."""
        self.emit("OpFunctionEnd")
        self.current_label = None

    def binary_operation(
        self, op: str, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> SpirvId:
        """Create a binary operation."""
        matrix_result = self.matrix_binary_operation(op, result_type, left, right)
        if matrix_result is not None:
            return matrix_result

        complex_result = self.complex_struct_arithmetic_operation(op, left, right)
        if complex_result is not None:
            return complex_result

        result_type = self.ensure_registered_type(result_type)
        result_component_type = self.scalar_or_vector_component_type(result_type.type)
        if (
            op in {"+", "-", "*", "MULTIPLY", "/", "%", self.FLOORED_MOD_OPERATOR}
            and result_component_type == self.BFLOAT16_TYPE_NAME
        ):
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            float_type = self.register_primitive_type("float")
            arithmetic_type = (
                self.register_vector_type(float_type, result_vector[1])
                if result_vector is not None
                else float_type
            )
            promoted_left = self.convert_value_to_type(left, arithmetic_type)
            promoted_right = self.convert_value_to_type(right, arithmetic_type)
            promoted_result = self.binary_operation(
                op, arithmetic_type, promoted_left, promoted_right
            )
            return self.convert_value_to_type(promoted_result, result_type)

        arithmetic_ops = {
            "+": ("OpFAdd", "OpIAdd", "OpIAdd"),
            "-": ("OpFSub", "OpISub", "OpISub"),
            "*": ("OpFMul", "OpIMul", "OpIMul"),
            "MULTIPLY": ("OpFMul", "OpIMul", "OpIMul"),
            "/": ("OpFDiv", "OpSDiv", "OpUDiv"),
            # C, C++, MSL and HLSL `%` (and C `fmod`) truncate the quotient, so the
            # remainder takes the sign of the dividend: OpSRem/OpFRem. OpSMod and
            # OpFMod take the sign of the divisor and only implement GLSL's
            # floored `mod`, which callers request with FLOORED_MOD_OPERATOR.
            "%": ("OpFRem", "OpSRem", "OpUMod"),
            self.FLOORED_MOD_OPERATOR: ("OpFMod", "OpSMod", "OpUMod"),
        }
        comparison_ops = {
            "==": ("OpFOrdEqual", "OpIEqual", "OpIEqual"),
            # `!=` is true when either operand is NaN in GLSL (glslang emits the unordered
            # form), HLSL (DXIL fcmp une) and MSL (C++), so `x != x` stays a NaN test.
            "!=": ("OpFUnordNotEqual", "OpINotEqual", "OpINotEqual"),
            "<": ("OpFOrdLessThan", "OpSLessThan", "OpULessThan"),
            ">": ("OpFOrdGreaterThan", "OpSGreaterThan", "OpUGreaterThan"),
            "<=": ("OpFOrdLessThanEqual", "OpSLessThanEqual", "OpULessThanEqual"),
            ">=": (
                "OpFOrdGreaterThanEqual",
                "OpSGreaterThanEqual",
                "OpUGreaterThanEqual",
            ),
        }

        if op in {"&&", "||"}:
            result_type, left, right = self.logical_result_type_and_operands(
                left, right
            )
            spv_op = "OpLogicalAnd" if op == "&&" else "OpLogicalOr"
        elif op in comparison_ops:
            complex_result = self.complex_struct_comparison_operation(op, left, right)
            if complex_result is not None:
                return complex_result
            result_type, left, right = self.comparison_result_type_and_operands(
                left, right
            )
            float_op, signed_op, unsigned_op = comparison_ops[op]
            component_type = self.scalar_or_vector_component_type(left.type)
            if component_type == "bool" and op in {"==", "!="}:
                spv_op = "OpLogicalEqual" if op == "==" else "OpLogicalNotEqual"
            elif component_type == "bool":
                # SPIR-V has no ordered/integer comparison for bool operands, so an
                # ordered comparison (<, >, <=, >=) of booleans -- e.g. ArgMin/ArgMax
                # over a bool tensor comparing best.val > current.val -- must promote
                # the operands to uint (false -> 0, true -> 1) like C++ bool
                # arithmetic and use the unsigned comparison opcode.
                vector_info = self.vector_component_type_and_count(left.type.base_type)
                uint_type = self.register_primitive_type("uint")
                operand_type = (
                    self.register_vector_type(uint_type, vector_info[1])
                    if vector_info is not None
                    else uint_type
                )
                left = self.convert_value_to_type(left, operand_type)
                right = self.convert_value_to_type(right, operand_type)
                spv_op = unsigned_op
            else:
                spv_op = (
                    unsigned_op
                    if component_type in self.UNSIGNED_INTEGER_TYPES
                    else (
                        signed_op
                        if component_type in self.SIGNED_INTEGER_TYPES
                        else float_op
                    )
                )
        elif op in arithmetic_ops:
            bool_arithmetic_type = self.bool_arithmetic_integer_type(result_type)
            if bool_arithmetic_type is not None:
                left = self.convert_value_to_type(left, bool_arithmetic_type)
                right = self.convert_value_to_type(right, bool_arithmetic_type)
                _float_op, _signed_op, spv_op = arithmetic_ops[op]
                id_value = self.get_id()
                self.emit(
                    f"%{id_value} = {spv_op} %{bool_arithmetic_type.id} "
                    f"%{left.id} %{right.id}"
                )
                self.decorate_no_contraction_result(
                    id_value,
                    spv_op,
                    bool_arithmetic_type,
                )
                arithmetic_result = SpirvId(id_value, bool_arithmetic_type.type)
                self.value_types[id_value] = bool_arithmetic_type
                return self.convert_value_to_type(arithmetic_result, result_type)
            result_type, left, right = self.align_binary_arithmetic_operands(
                result_type, left, right
            )
            float_op, signed_op, unsigned_op = arithmetic_ops[op]
            component_type = self.scalar_or_vector_component_type(left.type)
            spv_op = (
                unsigned_op
                if component_type in self.UNSIGNED_INTEGER_TYPES
                else (
                    signed_op
                    if component_type in self.SIGNED_INTEGER_TYPES
                    else float_op
                )
            )
        elif op in {"&", "|", "^"}:
            bitwise_op = {
                "&": "OpBitwiseAnd",
                "|": "OpBitwiseOr",
                "^": "OpBitwiseXor",
            }[op]
            # SPIR-V's OpBitwiseAnd/Or/Xor require integer operands, so a bitwise
            # op whose result type is bool -- e.g. MLX's integer-power `exp & 1`
            # and the BitwiseAnd/Or/Xor bool specializations of binary.metal --
            # must be evaluated on a promoted integer type and converted back to
            # bool (SPIR-V rejects `OpBitwise* %bool`).
            bool_integer_type = self.bool_arithmetic_integer_type(result_type)
            if bool_integer_type is not None:
                return self.emit_bool_result_integer_binary_operation(
                    bitwise_op, bool_integer_type, result_type, left, right
                )
            result_type, left, right = self.align_binary_bitwise_operands(
                result_type, left, right
            )
            spv_op = bitwise_op
        elif op in {"<<", ">>"}:
            shift_op = {
                "<<": "OpShiftLeftLogical",
                ">>": "OpShiftRightLogical",
            }[op]
            # OpShiftLeftLogical/OpShiftRightLogical likewise reject a bool result
            # type -- e.g. the `exp >>= 1` step of the same integer-power bool
            # specialization -- so promote to an integer type and convert back.
            bool_integer_type = self.bool_arithmetic_integer_type(result_type)
            if bool_integer_type is not None:
                return self.emit_bool_result_integer_binary_operation(
                    shift_op, bool_integer_type, result_type, left, right
                )
            result_type, left, right = self.align_binary_shift_operands(
                result_type, left, right
            )
            spv_op = shift_op
            if (
                op == ">>"
                and self.scalar_or_vector_component_type(result_type.type)
                in self.SIGNED_INTEGER_TYPES
            ):
                # C++20/MSL, HLSL and GLSL shift a signed left operand right
                # arithmetically (sign-extending); OpShiftRightLogical would
                # zero-fill and turn -8 >> 1 into a large positive value.
                spv_op = "OpShiftRightArithmetic"
        else:
            spv_op = f"Op{op}"

        if spv_op == "OpFRem":
            return self.emit_truncated_float_remainder(result_type, left, right)

        id_value = self.get_id()
        self.emit(f"%{id_value} = {spv_op} %{result_type.id} %{left.id} %{right.id}")
        self.decorate_no_contraction_result(id_value, spv_op, result_type)

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def bool_arithmetic_integer_type(self, result_type: SpirvId) -> Optional[SpirvId]:
        result_type = self.ensure_registered_type(result_type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type != "bool":
            return None
        uint_type = self.register_primitive_type("uint")
        vector_info = self.vector_component_type_and_count(result_type.type.base_type)
        if vector_info is None:
            return uint_type
        return self.register_vector_type(uint_type, vector_info[1])

    def emit_bool_result_integer_binary_operation(
        self,
        spv_op: str,
        integer_type: SpirvId,
        result_type: SpirvId,
        left: SpirvId,
        right: SpirvId,
    ) -> SpirvId:
        """Evaluate an integer-only opcode whose source result type is bool.

        SPIR-V's integer opcodes (OpBitwiseAnd/Or/Xor, OpShiftLeftLogical/
        OpShiftRightLogical) reject bool result types. C++ promotes bool operands
        to int for these operators, so the operands are converted to
        ``integer_type`` (false -> 0, true -> 1), the opcode is emitted on that
        integer type, and the result is converted back to the bool result type.
        Mirrors the bool arithmetic handling in ``binary_operation``.
        """
        result_type = self.ensure_registered_type(result_type)
        left = self.convert_value_to_type(left, integer_type)
        right = self.convert_value_to_type(right, integer_type)
        id_value = self.get_id()
        self.emit(f"%{id_value} = {spv_op} %{integer_type.id} %{left.id} %{right.id}")
        self.value_types[id_value] = integer_type
        integer_result = SpirvId(id_value, integer_type.type)
        return self.convert_value_to_type(integer_result, result_type)

    def align_binary_bitwise_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        """Convert bitwise operands to the result integer type before emission."""
        result_type = self.ensure_registered_type(result_type)
        left = self.convert_value_to_type(left, result_type)
        right = self.convert_value_to_type(right, result_type)
        return result_type, left, right

    def align_binary_shift_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        """Match SPIR-V shift count shape to the left/result operand shape."""
        result_type = self.ensure_registered_type(result_type)
        result_vector = self.vector_component_type_and_count(result_type.type.base_type)

        left = self.convert_value_to_type(left, result_type)
        if result_vector is None:
            right = self.convert_value_to_type(right, result_type)
            return result_type, left, right

        component_type_name, component_count = result_vector
        component_type = self.register_primitive_type(component_type_name)
        right_vector = self.vector_component_type_and_count(right.type.base_type)

        if right_vector is None:
            right = self.convert_scalar_to_type(right, component_type)
            if self.scalar_or_vector_component_type(right.type) == component_type_name:
                right = self.splat_scalar_to_vector(right, result_type)
        elif right_vector[1] == component_count:
            right = self.convert_value_to_type(right, result_type)

        return result_type, left, right

    def matrix_binary_operation(
        self, op: str, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Optional[SpirvId]:
        """Lower matrix arithmetic to SPIR-V matrix or column-vector operations."""
        left_type = self.registered_value_type(left)
        right_type = self.registered_value_type(right)
        left_matrix = self.matrix_operand_info(left_type)
        right_matrix = self.matrix_operand_info(right_type)
        if left_matrix is None and right_matrix is None:
            return None

        if op not in {"+", "-", "*", "/"}:
            return self.unsupported_matrix_operation(op, result_type)

        if left_matrix is not None and right_matrix is not None:
            if op in {"+", "-", "/"}:
                return self.componentwise_matrix_operation(
                    op, left, left_type, left_matrix, right, right_type, right_matrix
                )
            if op == "*":
                return self.matrix_times_matrix(
                    left, left_matrix, right, right_matrix, result_type
                )
            return self.unsupported_matrix_operation(op, result_type)

        if left_matrix is not None:
            right_vector = self.vector_type_info_from_type(right_type)
            if op == "*" and right_vector is not None:
                return self.matrix_times_vector(left, left_matrix, right, right_vector)
            if right_vector is None:
                if op == "*":
                    return self.matrix_times_scalar(left, left_type, left_matrix, right)
                return self.matrix_scalar_componentwise(
                    op, left, left_type, left_matrix, right, scalar_first=False
                )
            return self.unsupported_matrix_operation(op, result_type)

        left_vector = self.vector_type_info_from_type(left_type)
        if op == "*" and left_vector is not None and right_matrix is not None:
            return self.vector_times_matrix(left, left_vector, right, right_matrix)
        if op == "*" and left_vector is None and right_matrix is not None:
            return self.matrix_times_scalar(right, right_type, right_matrix, left)
        if left_vector is None and right_matrix is not None:
            return self.matrix_scalar_componentwise(
                op, right, right_type, right_matrix, left, scalar_first=True
            )

        return self.unsupported_matrix_operation(op, result_type)

    def matrix_operand_info(self, type_id: SpirvId):
        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is None:
            return None

        column_type, column_count = matrix_info
        column_info = self.vector_type_info_from_type(column_type)
        if column_info is None:
            return None

        component_type, row_count = column_info
        return {
            "type": type_id,
            "column_type": column_type,
            "column_count": column_count,
            "component_type": component_type,
            "row_count": row_count,
        }

    def matrix_shapes_match(self, left_info, right_info) -> bool:
        return (
            left_info["column_count"] == right_info["column_count"]
            and left_info["row_count"] == right_info["row_count"]
            and left_info["component_type"].id == right_info["component_type"].id
        )

    def matrix_scalar_operand(self, value: SpirvId, component_type: SpirvId):
        value_type = self.registered_value_type(value)
        if value_type is not None:
            if self.vector_type_info_from_type(value_type) is not None:
                return None
            if self.matrix_type_info_from_type(value_type) is not None:
                return None

        converted = self.convert_scalar_to_type(value, component_type)
        if self.normalize_primitive_name(
            converted.type.base_type
        ) != self.normalize_primitive_name(component_type.type.base_type):
            return None
        return converted

    def unsupported_matrix_operation(self, op: str, fallback_type: SpirvId) -> SpirvId:
        self.refuse_lowering(f"matrix operation '{op}' is not supported for operands")
        return self.default_value_for_type(self.ensure_registered_type(fallback_type))

    def componentwise_matrix_operation(
        self,
        op: str,
        left: SpirvId,
        left_type: SpirvId,
        left_info,
        right: SpirvId,
        right_type: SpirvId,
        right_info,
    ) -> SpirvId:
        if not self.matrix_shapes_match(left_info, right_info):
            return self.unsupported_matrix_operation(op, left_type)

        columns = []
        for index in range(left_info["column_count"]):
            left_column = self.composite_extract(left, left_info["column_type"], index)
            right_column = self.composite_extract(
                right, right_info["column_type"], index
            )
            columns.append(
                self.binary_operation(
                    op, left_info["column_type"], left_column, right_column
                )
            )
        return self.composite_construct(left_type, columns)

    def matrix_times_scalar(
        self, matrix: SpirvId, matrix_type: SpirvId, matrix_info, scalar: SpirvId
    ) -> SpirvId:
        scalar = self.matrix_scalar_operand(scalar, matrix_info["component_type"])
        if scalar is None:
            return self.unsupported_matrix_operation("*", matrix_type)

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpMatrixTimesScalar %{matrix_type.id} "
            f"%{matrix.id} %{scalar.id}"
        )

        spirv_id = SpirvId(id_value, matrix_type.type)
        self.value_types[id_value] = matrix_type
        self.decorate_no_contraction_result(
            id_value, "OpMatrixTimesScalar", matrix_type
        )
        return spirv_id

    def matrix_scalar_componentwise(
        self,
        op: str,
        matrix: SpirvId,
        matrix_type: SpirvId,
        matrix_info,
        scalar: SpirvId,
        *,
        scalar_first: bool,
    ) -> SpirvId:
        """``m op s`` or ``s op m`` for ``+``, ``-`` or ``/``, one column at a time (GLSL 4.60 section 5.9)."""
        scalar = self.matrix_scalar_operand(scalar, matrix_info["component_type"])
        if scalar is None:
            return self.unsupported_matrix_operation(op, matrix_type)

        columns = []
        for index in range(matrix_info["column_count"]):
            column = self.composite_extract(matrix, matrix_info["column_type"], index)
            operands = (scalar, column) if scalar_first else (column, scalar)
            columns.append(
                self.binary_operation(op, matrix_info["column_type"], *operands)
            )
        return self.composite_construct(matrix_type, columns)

    def matrix_times_vector(
        self, matrix: SpirvId, matrix_info, vector: SpirvId, vector_info
    ) -> SpirvId:
        if (
            vector_info[0].id != matrix_info["component_type"].id
            or vector_info[1] != matrix_info["column_count"]
        ):
            return self.unsupported_matrix_operation("*", matrix_info["column_type"])

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpMatrixTimesVector %{matrix_info['column_type'].id} "
            f"%{matrix.id} %{vector.id}"
        )

        spirv_id = SpirvId(id_value, matrix_info["column_type"].type)
        self.value_types[id_value] = matrix_info["column_type"]
        self.decorate_no_contraction_result(
            id_value, "OpMatrixTimesVector", matrix_info["column_type"]
        )
        return spirv_id

    def vector_times_matrix(
        self, vector: SpirvId, vector_info, matrix: SpirvId, matrix_info
    ) -> SpirvId:
        if (
            vector_info[0].id != matrix_info["component_type"].id
            or vector_info[1] != matrix_info["row_count"]
        ):
            return self.unsupported_matrix_operation("*", matrix_info["column_type"])

        result_type = self.register_vector_type(
            matrix_info["component_type"], matrix_info["column_count"]
        )
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpVectorTimesMatrix %{result_type.id} "
            f"%{vector.id} %{matrix.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.decorate_no_contraction_result(
            id_value, "OpVectorTimesMatrix", result_type
        )
        return spirv_id

    def matrix_times_matrix(
        self,
        left: SpirvId,
        left_info,
        right: SpirvId,
        right_info,
        fallback_type: SpirvId,
    ) -> SpirvId:
        if (
            left_info["component_type"].id != right_info["component_type"].id
            or left_info["column_count"] != right_info["row_count"]
        ):
            return self.unsupported_matrix_operation("*", fallback_type)

        result_column_type = self.register_vector_type(
            left_info["component_type"], left_info["row_count"]
        )
        result_type = self.register_matrix_type(
            result_column_type, right_info["column_count"]
        )
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpMatrixTimesMatrix %{result_type.id} "
            f"%{left.id} %{right.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        self.decorate_no_contraction_result(
            id_value, "OpMatrixTimesMatrix", result_type
        )
        return spirv_id

    def decorate_no_contraction_result(
        self, id_value: int, opcode: str, result_type: SpirvId
    ):
        if self.precise_expression_depth <= 0:
            return
        if opcode not in {
            "OpFAdd",
            "OpFSub",
            "OpFMul",
            "OpFDiv",
            "OpFMod",
            "OpFRem",
            "OpFNegate",
            "OpDot",
            "OpMatrixTimesScalar",
            "OpMatrixTimesVector",
            "OpVectorTimesMatrix",
            "OpMatrixTimesMatrix",
        }:
            return
        if not self.is_float_like_type_id(result_type):
            return
        self.decorate_no_contraction(id_value)

    def decorate_no_contraction(self, id_value):
        id_number = id_value.id if isinstance(id_value, SpirvId) else id_value
        if id_number in self.no_contraction_ids:
            return
        self.no_contraction_ids.add(id_number)
        self.decorations.append(f"OpDecorate %{id_number} NoContraction")

    def is_float_like_type_id(self, type_id: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(type_id.type)
        if component_type in {"float", "double"}:
            return True

        matrix_info = self.matrix_type_info_from_type(type_id)
        if matrix_info is None:
            return False

        column_type, _ = matrix_info
        return self.is_float_like_type_id(column_type)

    def process_expression_with_precision(
        self, expr, precise: bool
    ) -> Optional[SpirvId]:
        if not precise:
            return self.process_expression(expr)

        self.precise_expression_depth += 1
        try:
            return self.process_expression(expr)
        finally:
            self.precise_expression_depth -= 1

    def logical_result_type_and_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        bool_type = self.register_primitive_type("bool")
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        if left_vector is None and right_vector is None:
            return (
                bool_type,
                self.ensure_bool_value(left),
                self.ensure_bool_value(right),
            )

        component_type, component_count = left_vector or right_vector
        if component_type != "bool":
            return bool_type, left, right

        vector_operand_type = self.ensure_registered_type(
            left.type if left_vector is not None else right.type
        )

        if left_vector is not None and right_vector is None:
            if self.scalar_or_vector_component_type(right.type) == "bool":
                right = self.splat_scalar_to_vector(right, vector_operand_type)
        elif right_vector is not None and left_vector is None:
            if self.scalar_or_vector_component_type(left.type) == "bool":
                left = self.splat_scalar_to_vector(left, vector_operand_type)

        return self.register_vector_type(bool_type, component_count), left, right

    def ensure_bool_value(self, value: SpirvId) -> SpirvId:
        bool_type = self.register_primitive_type("bool")
        if value.type.base_type == "bool":
            return value
        return self.convert_value_to_type(value, bool_type)

    def complex_struct_comparison_operation(
        self, op: str, left: SpirvId, right: SpirvId
    ) -> Optional[SpirvId]:
        left_type = self.registered_value_type(left) or self.ensure_registered_type(
            left.type
        )
        right_type = self.registered_value_type(right) or self.ensure_registered_type(
            right.type
        )
        left_members = self.current_struct_members.get(left_type.type.base_type)
        right_members = self.current_struct_members.get(right_type.type.base_type)
        if left_members is None or right_members is None:
            return None
        if not self.complex_struct_types_are_layout_compatible(
            left_type, right_type, left_members, right_members
        ):
            return None

        left_real = self.composite_extract(left, left_members[0][0], 0)
        left_imag = self.composite_extract(left, left_members[1][0], 1)
        right_real = self.composite_extract(right, right_members[0][0], 0)
        right_imag = self.composite_extract(right, right_members[1][0], 1)
        bool_type = self.register_primitive_type("bool")

        if op == "==":
            real_equal = self.binary_operation("==", bool_type, left_real, right_real)
            imag_equal = self.binary_operation("==", bool_type, left_imag, right_imag)
            return self.binary_operation("&&", bool_type, real_equal, imag_equal)
        if op == "!=":
            real_not_equal = self.binary_operation(
                "!=", bool_type, left_real, right_real
            )
            imag_not_equal = self.binary_operation(
                "!=", bool_type, left_imag, right_imag
            )
            return self.binary_operation(
                "||", bool_type, real_not_equal, imag_not_equal
            )

        primary_op = ">" if op in {">", ">="} else "<"
        secondary_op = ">=" if op == ">=" else "<=" if op == "<=" else primary_op
        primary = self.binary_operation(primary_op, bool_type, left_real, right_real)
        real_equal = self.binary_operation("==", bool_type, left_real, right_real)
        secondary = self.binary_operation(
            secondary_op, bool_type, left_imag, right_imag
        )
        tied_secondary = self.binary_operation("&&", bool_type, real_equal, secondary)
        return self.binary_operation("||", bool_type, primary, tied_secondary)

    def complex_struct_operand_info(self, value: SpirvId):
        type_id = self.registered_value_type(value) or self.ensure_registered_type(
            value.type
        )
        members = self.current_struct_members.get(type_id.type.base_type)
        if members is None or not self.is_complex_struct_type(type_id, members):
            return None
        return type_id, members

    def complex_struct_negate_operation(self, operand: SpirvId) -> Optional[SpirvId]:
        """Negate a complex-number struct componentwise.

        SPIR-V has no negation opcode for aggregate (struct) types, so unary
        minus on a complex value negates its real and imaginary members and
        reconstructs the struct: ``-(a + b*i) == (-a) + (-b)*i``. Returns None
        when the operand is not a complex struct so scalar/vector negation keeps
        using the direct OpFNegate/OpSNegate path.
        """
        operand_info = self.complex_struct_operand_info(operand)
        if operand_info is None:
            return None

        result_type, members = operand_info
        real_type = members[0][0]
        imag_type = members[1][0]
        real = self.composite_extract(operand, real_type, 0)
        imag = self.composite_extract(operand, imag_type, 1)
        negated_real = self.convert_value_to_type(
            self.unary_operation("-", real_type, real), real_type
        )
        negated_imag = self.convert_value_to_type(
            self.unary_operation("-", imag_type, imag), imag_type
        )
        if not self.value_has_type(negated_real, real_type) or not self.value_has_type(
            negated_imag, imag_type
        ):
            return None
        return self.composite_construct(result_type, [negated_real, negated_imag])

    def complex_struct_arithmetic_operation(
        self, op: str, left: SpirvId, right: SpirvId
    ) -> Optional[SpirvId]:
        if op == "MULTIPLY":
            op = "*"
        if op not in {"+", "-", "*", "/", "%"}:
            return None

        left_info = self.complex_struct_operand_info(left)
        right_info = self.complex_struct_operand_info(right)
        if left_info is None and right_info is None:
            return None

        result_type, result_members = left_info or right_info
        if left_info is None:
            left = self.convert_value_to_type(left, result_type)
            left_info = self.complex_struct_operand_info(left)
        if right_info is None:
            right = self.convert_value_to_type(right, result_type)
            right_info = self.complex_struct_operand_info(right)
        if left_info is None or right_info is None:
            return None

        left_type, left_members = left_info
        right_type, right_members = right_info
        if not self.complex_struct_types_are_layout_compatible(
            left_type, right_type, left_members, right_members
        ):
            return None

        left_real = self.composite_extract(left, left_members[0][0], 0)
        left_imag = self.composite_extract(left, left_members[1][0], 1)
        right_real = self.composite_extract(right, right_members[0][0], 0)
        right_imag = self.composite_extract(right, right_members[1][0], 1)
        real_type = result_members[0][0]
        imag_type = result_members[1][0]

        if op == "+":
            real = self.binary_operation("+", real_type, left_real, right_real)
            imag = self.binary_operation("+", imag_type, left_imag, right_imag)
        elif op == "-":
            real = self.binary_operation("-", real_type, left_real, right_real)
            imag = self.binary_operation("-", imag_type, left_imag, right_imag)
        elif op == "*":
            ac = self.binary_operation("*", real_type, left_real, right_real)
            bd = self.binary_operation("*", real_type, left_imag, right_imag)
            ad = self.binary_operation("*", imag_type, left_real, right_imag)
            bc = self.binary_operation("*", imag_type, left_imag, right_real)
            real = self.binary_operation("-", real_type, ac, bd)
            imag = self.binary_operation("+", imag_type, ad, bc)
        elif op == "/":
            c2 = self.binary_operation("*", real_type, right_real, right_real)
            d2 = self.binary_operation("*", real_type, right_imag, right_imag)
            denom = self.binary_operation("+", real_type, c2, d2)
            ac = self.binary_operation("*", real_type, left_real, right_real)
            bd = self.binary_operation("*", real_type, left_imag, right_imag)
            bc = self.binary_operation("*", imag_type, left_imag, right_real)
            ad = self.binary_operation("*", imag_type, left_real, right_imag)
            x = self.binary_operation("+", real_type, ac, bd)
            y = self.binary_operation("-", imag_type, bc, ad)
            real = self.binary_operation("/", real_type, x, denom)
            imag = self.binary_operation("/", imag_type, y, denom)
        else:
            real = self.binary_operation("%", real_type, left_real, right_real)
            imag = self.binary_operation("%", imag_type, left_imag, right_imag)

        real = self.convert_value_to_type(real, real_type)
        imag = self.convert_value_to_type(imag, imag_type)
        if not self.value_has_type(real, real_type) or not self.value_has_type(
            imag, imag_type
        ):
            return None
        return self.composite_construct(result_type, [real, imag])

    def comparison_result_type_and_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        bool_type = self.register_primitive_type("bool")
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        if left_vector is None and right_vector is None:
            component_type_name = self.promoted_numeric_type_name(
                [left.type.base_type, right.type.base_type]
            )
            if component_type_name is not None:
                operand_type = self.register_primitive_type(component_type_name)
                left = self.convert_value_to_type(left, operand_type)
                right = self.convert_value_to_type(right, operand_type)
            return bool_type, left, right

        component_type, component_count = left_vector or right_vector
        promoted_component_type = component_type
        if left_vector is not None and right_vector is not None:
            if left_vector[1] != right_vector[1]:
                return (
                    self.register_vector_type(bool_type, component_count),
                    left,
                    right,
                )
            promoted_component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_vector[0]]
            )
            if promoted_component_type is None:
                promoted_component_type = (
                    left_vector[0]
                    if left_vector[0] == right_vector[0]
                    else component_type
                )
        elif left_vector is not None:
            right_component = self.scalar_or_vector_component_type(right.type)
            promoted_component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_component]
            )
            if promoted_component_type is None:
                promoted_component_type = (
                    left_vector[0]
                    if right_component == left_vector[0]
                    else component_type
                )
        elif right_vector is not None:
            left_component = self.scalar_or_vector_component_type(left.type)
            promoted_component_type = self.promoted_numeric_type_name(
                [left_component, right_vector[0]]
            )
            if promoted_component_type is None:
                promoted_component_type = (
                    right_vector[0]
                    if left_component == right_vector[0]
                    else component_type
                )

        vector_operand_type = self.register_vector_type(
            self.register_primitive_type(promoted_component_type),
            component_count,
        )

        if left_vector is not None and right_vector is None:
            left = self.convert_value_to_type(left, vector_operand_type)
            component = self.register_primitive_type(promoted_component_type)
            right = self.convert_scalar_to_type(right, component)
            if (
                self.scalar_or_vector_component_type(right.type)
                == promoted_component_type
            ):
                right = self.splat_scalar_to_vector(right, vector_operand_type)
        elif right_vector is not None and left_vector is None:
            right = self.convert_value_to_type(right, vector_operand_type)
            component = self.register_primitive_type(promoted_component_type)
            left = self.convert_scalar_to_type(left, component)
            if (
                self.scalar_or_vector_component_type(left.type)
                == promoted_component_type
            ):
                left = self.splat_scalar_to_vector(left, vector_operand_type)
        elif left_vector is not None and right_vector is not None:
            left = self.convert_value_to_type(left, vector_operand_type)
            right = self.convert_value_to_type(right, vector_operand_type)

        return self.register_vector_type(bool_type, component_count), left, right

    def align_binary_arithmetic_operands(
        self, result_type: SpirvId, left: SpirvId, right: SpirvId
    ) -> Tuple[SpirvId, SpirvId, SpirvId]:
        result_type = self.ensure_registered_type(result_type)
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)

        if left_vector is not None and right_vector is None:
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            vector_type = (
                result_type
                if result_vector is not None and result_vector[1] == left_vector[1]
                else self.ensure_registered_type(left.type)
            )
            component_type_name = self.vector_component_type_and_count(
                vector_type.type.base_type
            )[0]
            component_type = self.register_primitive_type(component_type_name)
            left = self.convert_value_to_type(left, vector_type)
            right = self.convert_scalar_to_type(right, component_type)
            if self.scalar_or_vector_component_type(right.type) == component_type_name:
                return (
                    vector_type,
                    left,
                    self.splat_scalar_to_vector(right, vector_type),
                )
        if right_vector is not None and left_vector is None:
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            vector_type = (
                result_type
                if result_vector is not None and result_vector[1] == right_vector[1]
                else self.ensure_registered_type(right.type)
            )
            component_type_name = self.vector_component_type_and_count(
                vector_type.type.base_type
            )[0]
            component_type = self.register_primitive_type(component_type_name)
            right = self.convert_value_to_type(right, vector_type)
            left = self.convert_scalar_to_type(left, component_type)
            if self.scalar_or_vector_component_type(left.type) == component_type_name:
                return (
                    vector_type,
                    self.splat_scalar_to_vector(left, vector_type),
                    right,
                )
        if left_vector is not None and right_vector is not None:
            result_vector = self.vector_component_type_and_count(
                result_type.type.base_type
            )
            if (
                result_vector is not None
                and left_vector[1] == result_vector[1]
                and right_vector[1] == result_vector[1]
            ):
                left = self.convert_value_to_type(left, result_type)
                right = self.convert_value_to_type(right, result_type)
                if self.value_has_type(left, result_type) and self.value_has_type(
                    right, result_type
                ):
                    return result_type, left, right

        if left_vector is None and right_vector is None:
            left = self.convert_value_to_type(left, result_type)
            right = self.convert_value_to_type(right, result_type)

        return result_type, left, right

    def binary_expression_result_type(
        self, op: str, left_type: Optional[SpirvId], right_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        """Infer the SPIR-V result type for a binary expression."""
        if left_type is None or right_type is None:
            return left_type or right_type

        left_type = self.ensure_registered_type(left_type)
        right_type = self.ensure_registered_type(right_type)
        left_vector = self.vector_component_type_and_count(left_type.type.base_type)
        right_vector = self.vector_component_type_and_count(right_type.type.base_type)

        if op in {"&&", "||"}:
            if left_vector is not None or right_vector is not None:
                component_count = (left_vector or right_vector)[1]
                return self.register_vector_type(
                    self.register_primitive_type("bool"), component_count
                )
            return self.register_primitive_type("bool")

        if op in {"==", "!=", "<", ">", "<=", ">="}:
            if left_vector is not None or right_vector is not None:
                component_count = (left_vector or right_vector)[1]
                return self.register_vector_type(
                    self.register_primitive_type("bool"), component_count
                )
            return self.register_primitive_type("bool")

        if op in {"&", "|", "^"}:
            if left_vector is not None or right_vector is not None:
                component_count = (left_vector or right_vector)[1]
                component_type = self.promoted_bitwise_integer_type_name(
                    [
                        (
                            left_vector[0]
                            if left_vector is not None
                            else left_type.type.base_type
                        ),
                        (
                            right_vector[0]
                            if right_vector is not None
                            else right_type.type.base_type
                        ),
                    ]
                )
                if component_type is None:
                    return left_type
                return self.register_vector_type(
                    self.register_primitive_type(component_type), component_count
                )

            scalar_type = self.promoted_bitwise_integer_type_name(
                [left_type.type.base_type, right_type.type.base_type]
            )
            if scalar_type is None:
                return left_type
            return self.register_primitive_type(scalar_type)

        if op not in {"+", "-", "*", "MULTIPLY", "/", "%"}:
            return left_type

        if left_vector is not None and right_vector is not None:
            if left_vector[1] != right_vector[1]:
                return left_type
            component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_vector[0]], arithmetic=True
            )
            if component_type is None:
                return left_type
            return self.register_vector_type(
                self.register_primitive_type(component_type), left_vector[1]
            )

        if left_vector is not None:
            component_type = self.promoted_numeric_type_name(
                [left_vector[0], right_type.type.base_type], arithmetic=True
            )
            if component_type is None:
                return left_type
            return self.register_vector_type(
                self.register_primitive_type(component_type), left_vector[1]
            )

        if right_vector is not None:
            component_type = self.promoted_numeric_type_name(
                [left_type.type.base_type, right_vector[0]], arithmetic=True
            )
            if component_type is None:
                return right_type
            return self.register_vector_type(
                self.register_primitive_type(component_type), right_vector[1]
            )

        scalar_type = self.promoted_numeric_type_name(
            [left_type.type.base_type, right_type.type.base_type],
            arithmetic=True,
        )
        if scalar_type is None:
            return left_type
        return self.register_primitive_type(scalar_type)

    def bitwise_expression_operand_type(
        self, left_type: Optional[SpirvId], right_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        """Return a shared integer type for bitwise operands when it is knowable."""
        if left_type is None and right_type is None:
            return None

        left_type = self.ensure_registered_type(left_type) if left_type else None
        right_type = self.ensure_registered_type(right_type) if right_type else None
        left_vector = (
            self.vector_component_type_and_count(left_type.type.base_type)
            if left_type is not None
            else None
        )
        right_vector = (
            self.vector_component_type_and_count(right_type.type.base_type)
            if right_type is not None
            else None
        )

        if left_vector is not None or right_vector is not None:
            if (
                left_vector is not None
                and right_vector is not None
                and left_vector[1] != right_vector[1]
            ):
                return None
            component_count = (left_vector or right_vector)[1]
            component_names = [
                vector[0]
                for vector in (left_vector, right_vector)
                if vector is not None
            ]
            for scalar_type in (left_type, right_type):
                if scalar_type is not None:
                    scalar_component = self.normalize_primitive_name(
                        scalar_type.type.base_type
                    )
                    if scalar_component in {"int", "uint"}:
                        component_names.append(scalar_component)
            if not component_names or any(
                component not in {"int", "uint"} for component in component_names
            ):
                return None
            component_type = "uint" if "uint" in component_names else "int"
            return self.register_vector_type(
                self.register_primitive_type(component_type), component_count
            )

        type_names = [
            self.normalize_primitive_name(value.type.base_type)
            for value in (left_type, right_type)
            if value is not None
        ]
        if not type_names or any(
            type_name not in {"int", "uint"} for type_name in type_names
        ):
            return None
        return self.register_primitive_type("uint" if "uint" in type_names else "int")

    def splat_scalar_to_vector(
        self, scalar_id: SpirvId, vector_type: SpirvId
    ) -> SpirvId:
        vector_info = self.vector_component_type_and_count(vector_type.type.base_type)
        if vector_info is None:
            return scalar_id

        _, component_count = vector_info
        id_value = self.get_id()
        component_list = " ".join(f"%{scalar_id.id}" for _ in range(component_count))
        self.emit(
            f"%{id_value} = OpCompositeConstruct %{vector_type.id} {component_list}"
        )
        self.value_types[id_value] = vector_type
        return SpirvId(id_value, vector_type.type)

    @staticmethod
    def bfloat16_bits_for_float(value: float) -> int:
        """Round a binary32 value to bfloat16, preserving a quiet NaN payload."""
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        absolute_bits = bits & 0x7FFFFFFF
        if absolute_bits > 0x7F800000:
            return ((bits >> 16) | 0x40) & 0xFFFF
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        return ((bits + rounding_bias) >> 16) & 0xFFFF

    def convert_bfloat16_to_float(self, value: SpirvId) -> SpirvId:
        """Expand bfloat16 storage bits into an IEEE binary32 value."""
        uint_type = self.register_primitive_type("uint")
        float_type = self.register_primitive_type("float")

        widened_id = self.get_id()
        self.emit(f"%{widened_id} = OpUConvert %{uint_type.id} %{value.id}")
        self.value_types[widened_id] = uint_type
        widened = SpirvId(widened_id, uint_type.type)

        shift = self.register_constant(16, uint_type)
        shifted = self.binary_operation("<<", uint_type, widened, shift)
        result_id = self.get_id()
        self.emit(f"%{result_id} = OpBitcast %{float_type.id} %{shifted.id}")
        self.value_types[result_id] = float_type
        result = SpirvId(result_id, float_type.type)
        if self.is_non_uniform_value(value):
            self.mark_non_uniform_result(result)
        return result

    def convert_float_to_bfloat16(
        self, value: SpirvId, target_type: SpirvId
    ) -> SpirvId:
        """Round binary32 to bfloat16 storage bits using round-to-nearest-even."""
        uint_type = self.register_primitive_type("uint")
        bool_type = self.register_primitive_type("bool")

        bits_id = self.get_id()
        self.emit(f"%{bits_id} = OpBitcast %{uint_type.id} %{value.id}")
        self.value_types[bits_id] = uint_type
        bits = SpirvId(bits_id, uint_type.type)

        absolute_mask = self.register_constant(0x7FFFFFFF, uint_type)
        positive_infinity = self.register_constant(0x7F800000, uint_type)
        absolute_bits = self.binary_operation("&", uint_type, bits, absolute_mask)
        is_nan = self.binary_operation(">", bool_type, absolute_bits, positive_infinity)

        shift = self.register_constant(16, uint_type)
        one = self.register_constant(1, uint_type)
        upper_bits = self.binary_operation(">>", uint_type, bits, shift)
        least_significant_bit = self.binary_operation("&", uint_type, upper_bits, one)
        rounding_bias = self.binary_operation(
            "+",
            uint_type,
            self.register_constant(0x7FFF, uint_type),
            least_significant_bit,
        )
        rounded = self.binary_operation("+", uint_type, bits, rounding_bias)
        rounded_upper_bits = self.binary_operation(">>", uint_type, rounded, shift)

        quiet_nan_bits = self.binary_operation(
            "|",
            uint_type,
            upper_bits,
            self.register_constant(0x40, uint_type),
        )
        selected_bits = self.select_operation(
            uint_type, is_nan, quiet_nan_bits, rounded_upper_bits
        )

        result_id = self.get_id()
        self.emit(f"%{result_id} = OpUConvert %{target_type.id} %{selected_bits.id}")
        self.value_types[result_id] = target_type
        result = SpirvId(result_id, target_type.type)
        if self.is_non_uniform_value(value):
            self.mark_non_uniform_result(result)
        return result

    def convert_scalar_to_type(self, value: SpirvId, target_type: SpirvId) -> SpirvId:
        """Convert a scalar value to a compatible scalar target type."""
        source_type_name = self.normalize_primitive_name(value.type.base_type)
        target_type_name = self.normalize_primitive_name(target_type.type.base_type)
        if source_type_name == target_type_name:
            return value

        if source_type_name == self.BFLOAT16_TYPE_NAME:
            decoded = self.convert_bfloat16_to_float(value)
            return self.convert_scalar_to_type(decoded, target_type)

        if target_type_name == self.BFLOAT16_TYPE_NAME:
            float_type = self.register_primitive_type("float")
            if source_type_name == "bool":
                float_value = self.select_operation(
                    float_type,
                    value,
                    self.register_constant(1.0, float_type),
                    self.register_constant(0.0, float_type),
                )
            else:
                float_value = self.convert_scalar_to_type(value, float_type)
            if self.normalize_primitive_name(float_value.type.base_type) != "float":
                return value
            return self.convert_float_to_bfloat16(float_value, target_type)

        float_types = {"float", "double"}
        integer_types = self.INTEGER_TYPE_NAMES

        if source_type_name == "bool" and target_type_name in self.INTEGER_TYPE_NAMES:
            true_value = self.register_constant(1, target_type)
            false_value = self.register_constant(0, target_type)
            return self.select_operation(target_type, value, true_value, false_value)

        if source_type_name == "bool" and target_type_name in float_types:
            # true converts to 1.0 and false to 0.0 (GLSL 4.60 section 5.4.1).
            true_value = self.register_constant(1.0, target_type)
            false_value = self.register_constant(0.0, target_type)
            return self.select_operation(target_type, value, true_value, false_value)

        if source_type_name in self.INTEGER_TYPE_NAMES and target_type_name == "bool":
            source_type = self.ensure_registered_type(value.type)
            zero_value = self.register_constant(0, source_type)
            bool_type = self.register_primitive_type("bool")
            return self.binary_operation("!=", bool_type, value, zero_value)

        if source_type_name in float_types and target_type_name == "bool":
            source_type = self.ensure_registered_type(value.type)
            zero_value = self.register_constant(0.0, source_type)
            bool_type = self.register_primitive_type("bool")
            return self.binary_operation("!=", bool_type, value, zero_value)

        scalar_types = float_types | integer_types
        if source_type_name not in scalar_types or target_type_name not in scalar_types:
            return value

        if source_type_name in float_types and target_type_name in float_types:
            opcode = "OpFConvert"
        elif source_type_name in integer_types and target_type_name in float_types:
            opcode = (
                "OpConvertUToF"
                if source_type_name in self.UNSIGNED_INTEGER_TYPES
                else "OpConvertSToF"
            )
        elif source_type_name in float_types and target_type_name in integer_types:
            opcode = (
                "OpConvertFToU"
                if target_type_name in self.UNSIGNED_INTEGER_TYPES
                else "OpConvertFToS"
            )
        elif source_type_name in integer_types and target_type_name in integer_types:
            source_width = self.integer_type_width(source_type_name)
            target_width = self.integer_type_width(target_type_name)
            if source_width == target_width:
                opcode = "OpBitcast"
            elif source_type_name in self.UNSIGNED_INTEGER_TYPES:
                if target_type_name in self.UNSIGNED_INTEGER_TYPES:
                    opcode = "OpUConvert"
                elif source_width < target_width:
                    # An unsigned source widens by zero extension even into a
                    # signed type; OpSConvert would sign-extend values with the
                    # high bit set, and OpUConvert needs an unsigned result.
                    widened = self.convert_scalar_to_type(
                        value, self.register_primitive_type("u64")
                    )
                    return self.convert_scalar_to_type(widened, target_type)
                else:
                    opcode = "OpSConvert"
            else:
                opcode = "OpSConvert"
        else:
            return value

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{target_type.id} %{value.id}")
        self.value_types[id_value] = target_type
        return SpirvId(id_value, target_type.type)

    def is_integer_type(self, spirv_type: SpirvType) -> bool:
        return (
            self.normalize_primitive_name(spirv_type.base_type)
            in self.INTEGER_TYPE_NAMES
        )

    def is_unsigned_type(self, spirv_type: SpirvType) -> bool:
        return (
            self.normalize_primitive_name(spirv_type.base_type)
            in self.UNSIGNED_INTEGER_TYPES
        )

    def integer_type_width(self, type_name: str) -> Optional[int]:
        return self.INTEGER_TYPE_WIDTHS.get(self.normalize_primitive_name(type_name))

    def select_operation(
        self,
        result_type: SpirvId,
        condition: SpirvId,
        true_value: SpirvId,
        false_value: SpirvId,
    ) -> SpirvId:
        """Create a SPIR-V select operation for ternary expressions."""
        true_value = self.convert_value_to_type(true_value, result_type)
        false_value = self.convert_value_to_type(false_value, result_type)
        if self.vector_component_type_and_count(condition.type.base_type) is None:
            condition = self.ensure_bool_value(condition)
        if not self.can_use_select_operation(result_type, condition):
            return self.select_composite_operation(
                result_type, condition, true_value, false_value
            )

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpSelect %{result_type.id} %{condition.id} "
            f"%{true_value.id} %{false_value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def any_bool_vector_operation(self, vector: SpirvId) -> SpirvId:
        """Reduce a SPIR-V bool vector to a scalar bool using OpAny."""
        bool_type = self.register_primitive_type("bool")
        id_value = self.get_id()
        self.emit(f"%{id_value} = OpAny %{bool_type.id} %{vector.id}")
        spirv_id = SpirvId(id_value, bool_type.type)
        self.value_types[id_value] = bool_type
        return spirv_id

    def is_select_result_type(self, result_type: SpirvId) -> bool:
        """Return whether OpSelect can directly produce this result type."""
        base_type = self.normalize_primitive_name(result_type.type.base_type)
        if base_type in {"bool", "float", "double"} | self.INTEGER_TYPE_NAMES:
            return True
        return any(
            vector_type.type.base_type == base_type
            for vector_type in self.vector_types.values()
        )

    def can_use_select_operation(
        self, result_type: SpirvId, condition: SpirvId
    ) -> bool:
        if not self.is_select_result_type(result_type):
            return False

        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        condition_vector = self.vector_component_type_and_count(
            condition.type.base_type
        )
        if result_vector is not None:
            return (
                condition_vector is not None
                and condition_vector[0] == "bool"
                and condition_vector[1] == result_vector[1]
            )

        return condition.type.base_type == "bool"

    def select_composite_operation(
        self,
        result_type: SpirvId,
        condition: SpirvId,
        true_value: SpirvId,
        false_value: SpirvId,
    ) -> SpirvId:
        """Select composite values through control flow instead of OpSelect."""
        result_variable = self.create_variable(result_type, "Function")
        merge_label = SpirvId(self.get_id(), SpirvType("label"))
        then_label = SpirvId(self.get_id(), SpirvType("label"))
        else_label = SpirvId(self.get_id(), SpirvType("label"))
        condition = self.ensure_bool_value(condition)

        self.create_selection_merge(merge_label)
        self.create_conditional_branch(condition, then_label, else_label)

        self.emit(f"%{then_label.id} = OpLabel")
        self.current_label = then_label.id
        self.store_to_variable(result_variable, true_value)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{else_label.id} = OpLabel")
        self.current_label = else_label.id
        self.store_to_variable(result_variable, false_value)
        if not self.current_block_has_terminator():
            self.create_branch(merge_label)

        self.emit(f"%{merge_label.id} = OpLabel")
        self.current_label = merge_label.id
        return self.load_from_variable(result_variable, result_type)
