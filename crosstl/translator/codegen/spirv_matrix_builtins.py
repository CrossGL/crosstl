"""SPIR-V lowerings for the GLSL matrix functions of GLSL 4.60 section 8.6.

``transpose`` and ``outerProduct`` are core instructions (``OpTranspose``, ``OpOuterProduct``), ``determinant``
and ``inverse`` are ``GLSL.std.450`` ``Determinant`` and ``MatrixInverse``, and ``matrixCompMult`` multiplies the
columns component-wise, which is how glslang lowers it. Before these existed the emitter refused every call, so a
shader that used any of them had no SPIR-V output at all.

The core instructions and ``matrixCompMult`` take 32- and 64-bit float matrices of any shape; ``Determinant`` and
``MatrixInverse`` take square ones only (GLSL.std.450 specification, instructions 33 and 34). An operand of any
other type is refused with the host's typed builtin-lowering error rather than lowered to something else.

The mixin relies on the host class's ``find_registered_type_by_base``, ``matrix_type_info_from_type``,
``vector_type_info_from_type``, ``register_vector_type``, ``register_matrix_type``, ``composite_extract``,
``composite_construct``, ``binary_operation``, ``emit_glsl_std450_instruction``, ``get_id``, ``emit``,
``value_types`` and ``builtin_lowering_error``.
"""

from __future__ import annotations

__all__ = ["SpirvMatrixBuiltinMixin", "MATRIX_BUILTINS"]

MATRIX_BUILTINS = frozenset(
    {"transpose", "determinant", "inverse", "outerProduct", "matrixCompMult"}
)
_FLOAT_COMPONENTS = ("float", "double")


class SpirvMatrixBuiltinMixin:
    """Lower ``transpose``, ``determinant``, ``inverse``, ``outerProduct`` and ``matrixCompMult``."""

    def call_matrix_builtin_function(self, function_name: str, args: list):
        """Lower a GLSL matrix function, or return None when ``function_name`` is not one."""
        if function_name not in MATRIX_BUILTINS:
            return None
        expected = 2 if function_name in {"outerProduct", "matrixCompMult"} else 1
        if len(args) != expected:
            operand_label = "operand" if expected == 1 else "operands"
            raise self.builtin_lowering_error(
                function_name, f"{function_name} requires {expected} {operand_label}"
            )
        if function_name == "outerProduct":
            return self._lower_outer_product(*args)
        if function_name == "matrixCompMult":
            return self._lower_matrix_comp_mult(*args)

        shape = self._float_matrix_shape(args[0])
        if shape is None:
            raise self.builtin_lowering_error(
                function_name, f"{function_name} requires a float matrix operand"
            )
        matrix_type, column_type, component_type, cols, rows = shape
        if function_name == "transpose":
            result_type = self.register_matrix_type(
                self.register_vector_type(component_type, cols), rows
            )
            return self._emit_matrix_instruction("OpTranspose", result_type, [args[0]])
        if cols != rows:
            raise self.builtin_lowering_error(
                function_name,
                f"{function_name} requires a square matrix, got {cols} columns of {rows} rows",
            )
        if function_name == "determinant":
            return self.emit_glsl_std450_instruction(
                "Determinant", component_type, [args[0]]
            )
        return self.emit_glsl_std450_instruction(
            "MatrixInverse", matrix_type, [args[0]]
        )

    def _float_matrix_shape(self, value):
        """``(matrix type, column type, component type, columns, rows)`` of a float matrix value, else None."""
        matrix_type = self.find_registered_type_by_base(value.type.base_type)
        matrix_info = self.matrix_type_info_from_type(matrix_type)
        if matrix_info is None:
            return None
        column_type, cols = matrix_info
        column_info = self.vector_type_info_from_type(column_type)
        if column_info is None:
            return None
        component_type, rows = column_info
        if component_type is None or component_type.type.base_type not in (
            _FLOAT_COMPONENTS
        ):
            return None
        return matrix_type, column_type, component_type, cols, rows

    def _float_vector_shape(self, value):
        """``(vector type, component type, count)`` of a float vector value, else None."""
        vector_type = self.find_registered_type_by_base(value.type.base_type)
        vector_info = self.vector_type_info_from_type(vector_type)
        if vector_info is None:
            return None
        component_type, count = vector_info
        if component_type is None or component_type.type.base_type not in (
            _FLOAT_COMPONENTS
        ):
            return None
        return vector_type, component_type, count

    def _lower_outer_product(self, column, row):
        column_shape = self._float_vector_shape(column)
        row_shape = self._float_vector_shape(row)
        if (
            column_shape is None
            or row_shape is None
            or column_shape[1].id != row_shape[1].id
        ):
            raise self.builtin_lowering_error(
                "outerProduct",
                "outerProduct requires two float vectors of the same component type",
            )
        column_type, _component_type, _rows = column_shape
        result_type = self.register_matrix_type(column_type, row_shape[2])
        return self._emit_matrix_instruction(
            "OpOuterProduct", result_type, [column, row]
        )

    def _lower_matrix_comp_mult(self, left, right):
        left_shape = self._float_matrix_shape(left)
        right_shape = self._float_matrix_shape(right)
        if (
            left_shape is None
            or right_shape is None
            or left_shape[0].id != right_shape[0].id
        ):
            raise self.builtin_lowering_error(
                "matrixCompMult",
                "matrixCompMult requires two float matrices of the same type",
            )
        matrix_type, column_type, _component_type, cols, _rows = left_shape
        columns = [
            self.binary_operation(
                "*",
                column_type,
                self.composite_extract(left, column_type, index),
                self.composite_extract(right, column_type, index),
            )
            for index in range(cols)
        ]
        return self.composite_construct(matrix_type, columns)

    def _emit_matrix_instruction(self, opcode, result_type, operands):
        id_value = self.get_id()
        operand_list = " ".join(f"%{operand.id}" for operand in operands)
        self.emit(f"%{id_value} = {opcode} %{result_type.id} {operand_list}")
        self.value_types[id_value] = result_type
        # The host's SpirvId class; importing it here would be circular.
        return type(result_type)(id_value, result_type.type)
