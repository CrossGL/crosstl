"""Vector and matrix operations in the Mojo generator: Boolean selects, matrix constructor helpers, comparisons and binary operators."""

from ..refusals import MojoRefusal
from .mojo_tables import (
    MOJO_DTYPE_INFO,
    MOJO_DTYPE_SUFFIX,
    MOJO_INTEGER_DTYPES,
    MOJO_INTEGER_INDEX_TYPES,
    MOJO_MAPPED_INTEGER_TYPES,
    MOJO_SCALAR_DTYPES,
    MOJO_VALUE_STRUCT_TRAITS,
    MOJO_VECTOR_ARITHMETIC_OPS,
    MOJO_VECTOR_COMPARISON_METHODS,
)


class MojoVectorOpMixin:
    """Vector-operation members of ``MojoCodeGen``."""

    def generate_bool_vector_select_expression(
        self, condition_expr, true_expr, false_expr
    ):
        condition_info = self.vector_type_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info[0] != "DType.bool":
            return None

        true_info = self.vector_type_info(self.expression_result_type(true_expr))
        false_info = self.vector_type_info(self.expression_result_type(false_expr))
        if (
            true_info is None
            or false_info is None
            or true_info[:3] != false_info[:3]
            or condition_info[1] != true_info[1]
        ):
            return None

        dtype, source_width, storage_width, _ = true_info
        helper_name = self.select_helper_name(dtype, source_width, storage_width)
        self.required_select_helpers.add((dtype, source_width, storage_width))
        condition = self.generate_expression(condition_expr)
        true_value = self.generate_expression(true_expr)
        false_value = self.generate_expression(false_expr)
        return f"{helper_name}({condition}, {true_value}, {false_value})"

    def is_scalar_integer_type(self, type_name):
        if type_name is None or self.vector_type_info(type_name) is not None:
            return False
        type_name = self.type_name(type_name)
        if type_name in MOJO_INTEGER_INDEX_TYPES:
            return True
        if MOJO_SCALAR_DTYPES.get(type_name) in MOJO_INTEGER_DTYPES:
            return True
        return (
            type_name in MOJO_MAPPED_INTEGER_TYPES
            or self.type_mapping.get(type_name) in MOJO_MAPPED_INTEGER_TYPES
        )

    def generate_matrix_constructor_helper_call(self, dtype, columns, rows, args):
        component_count = columns * rows
        pieces = []

        for arg in args:
            piece = self.constructor_piece_for_expression(arg, dtype)
            if piece is None:
                return None
            pieces.append(piece)

        if len(args) == 1 and pieces and pieces[0]["kind"] == "scalar":
            return None

        pieces = self.select_constructor_pieces(pieces, component_count)
        if pieces is None:
            return None

        has_duplicate_sensitive_vector = any(
            piece["kind"] == "vector" and piece["duplicate_sensitive"]
            for piece in pieces
        )
        if not has_duplicate_sensitive_vector:
            return None

        key = self.matrix_constructor_helper_key(dtype, columns, rows, pieces)
        helper_name = self.matrix_constructor_helper_name(key)
        self.required_matrix_constructor_helpers[key] = {
            "key": key,
            "dtype": dtype,
            "columns": columns,
            "rows": rows,
            "pieces": pieces,
        }

        call_args = [
            self.generate_constructor_helper_argument(
                piece, dtype, f"matrix constructor helper argument {index}"
            )
            for index, piece in enumerate(pieces, start=1)
        ]
        return f"{helper_name}({', '.join(call_args)})"

    def matrix_constructor_helper_key(self, dtype, columns, rows, pieces):
        signature = self.constructor_helper_key(dtype, columns * rows, columns, pieces)
        return (dtype, columns, rows, signature[3])

    def matrix_constructor_helper_name(self, key):
        dtype, columns, rows, signature = key
        vector_key = (dtype, columns * rows, columns, signature)
        suffix = self.constructor_helper_name(vector_key).split("_", 4)[4]
        return (
            f"_crossgl_construct_matrix_{MOJO_DTYPE_SUFFIX[dtype]}_"
            f"c{columns}_r{rows}_{suffix}"
        )

    def matrix_diagonal_helper_name(self, dtype, columns, rows):
        return f"_crossgl_matrix_diagonal_{MOJO_DTYPE_SUFFIX[dtype]}_c{columns}_r{rows}"

    def matrix_type_name(self, dtype, columns, rows):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype].upper()
        return f"CrossGLMatrix{dtype_suffix}C{columns}R{rows}"

    def matrix_storage_rows(self, rows):
        return 4 if rows == 3 else rows

    def matrix_zero_literal(self, dtype):
        return MOJO_DTYPE_INFO[dtype][2]

    def generate_matrix_type(self, key):
        dtype, columns, rows = key
        name = self.matrix_type_name(dtype, columns, rows)
        storage_rows = self.matrix_storage_rows(rows)
        column_type = f"SIMD[{dtype}, {storage_rows}]"
        code = f"@fieldwise_init\nstruct {name}({MOJO_VALUE_STRUCT_TRAITS}):\n"
        for column in range(columns):
            code += f"    var c{column}: {column_type}\n"
        code += "\n"
        for index_type in ("Int", "Int32", "UInt32"):
            code += (
                f"    def __getitem__(self, index: {index_type}) -> {column_type}:\n"
            )
            for column in range(columns - 1):
                code += f"        if index == {column}:\n"
                code += f"            return self.c{column}\n"
            code += f"        return self.c{columns - 1}\n\n"

            code += (
                f"    def __setitem__(mut self, index: {index_type}, "
                f"value: {column_type}):\n"
            )
            for column in range(columns - 1):
                code += f"        if index == {column}:\n"
                code += f"            self.c{column} = value\n"
                code += "            return\n"
            code += f"        self.c{columns - 1} = value\n\n"
        if columns == rows:
            vector_terms = [
                f"self.c{column} * value[{column}]" for column in range(columns)
            ]
            code += f"    def __mul__(self, value: {column_type}) -> {column_type}:\n"
            code += f"        return {' + '.join(vector_terms)}\n\n"
            matrix_args = ", ".join(
                f"self * other.c{column}" for column in range(columns)
            )
            code += f"    def __mul__(self, other: {name}) -> {name}:\n"
            code += f"        return {name}({matrix_args})\n\n"

            transpose_args = []
            zero = self.matrix_zero_literal(dtype)
            for column in range(columns):
                components = [f"m.c{row}[{column}]" for row in range(rows)]
                if rows == 3:
                    components.append(zero)
                transpose_args.append(f"{column_type}({', '.join(components)})")
            code += f"def transpose(m: {name}) -> {name}:\n"
            code += f"    return {name}({', '.join(transpose_args)})\n\n"
            code += f"def inverse(m: {name}) -> {name}:\n"
            code += "    return m\n\n"
        return code + "\n"

    def generate_matrix_constructor_helper(self, helper):
        dtype = helper["dtype"]
        columns = helper["columns"]
        rows = helper["rows"]
        matrix_type = self.matrix_type_name(dtype, columns, rows)
        scalar_type, _, _ = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        params = []
        components = []
        prelude = []

        for index, piece in enumerate(helper["pieces"]):
            if piece["kind"] == "vector":
                param_name = f"v{index}"
                vector_type = f"SIMD[{piece['dtype']}, {piece['storage_width']}]"
                params.append(f"{param_name}: {vector_type}")
                vector_expr = param_name
                if piece["dtype"] != dtype:
                    vector_expr = f"{param_name}_cast"
                    prelude.append(
                        f"    var {vector_expr} = {param_name}.cast[{dtype}]()\n"
                    )
                components.extend(
                    f"{vector_expr}[{component_index}]"
                    for component_index in piece["indices"]
                )
            else:
                param_name = f"s{index}"
                piece_dtype = piece.get("dtype")
                param_scalar_type = mojo_scalar_type
                if piece_dtype is not None and piece_dtype != dtype:
                    scalar_type = MOJO_DTYPE_INFO[piece_dtype][0]
                    param_scalar_type = self.map_type(scalar_type)
                params.append(f"{param_name}: {param_scalar_type}")
                components.append(
                    self.cast_scalar_text(param_name, piece_dtype, dtype)
                    if piece_dtype is not None
                    else param_name
                )

        column_args = []
        storage_rows = self.matrix_storage_rows(rows)
        pad_literal = self.matrix_zero_literal(dtype)
        for column in range(columns):
            start = column * rows
            column_components = components[start : start + rows]
            if rows == 3:
                column_components.append(pad_literal)
            column_type = f"SIMD[{dtype}, {storage_rows}]"
            column_args.append(f"{column_type}({', '.join(column_components)})")

        helper_name = self.matrix_constructor_helper_name(helper["key"])
        code = f"def {helper_name}({', '.join(params)}) -> {matrix_type}:\n"
        code += "".join(prelude)
        code += f"    return {matrix_type}({', '.join(column_args)})\n\n"
        return code

    def generate_matrix_diagonal_helper(self, dtype, columns, rows):
        scalar_type, _, _ = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        matrix_type = self.matrix_type_name(dtype, columns, rows)
        storage_rows = self.matrix_storage_rows(rows)
        column_type = f"SIMD[{dtype}, {storage_rows}]"
        zero = self.matrix_zero_literal(dtype)
        column_args = []

        for column in range(columns):
            components = ["s" if column == row else zero for row in range(rows)]
            if rows == 3:
                components.append(zero)
            column_args.append(f"{column_type}({', '.join(components)})")

        helper_name = self.matrix_diagonal_helper_name(dtype, columns, rows)
        code = f"def {helper_name}(s: {mojo_scalar_type}) -> {matrix_type}:\n"
        code += f"    return {matrix_type}({', '.join(column_args)})\n\n"
        return code

    def vector_comparison_info(self, op, left_type, right_type):
        if self.map_operator(op) not in MOJO_VECTOR_COMPARISON_METHODS:
            return None
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        if left_info is not None and right_info is not None and left_info != right_info:
            raise MojoRefusal(
                "Mojo vector comparison requires matching element types and widths"
            )
        return left_info or right_info

    def generate_vector_comparison(self, expr):
        info = self.vector_comparison_info(
            expr.op,
            self.expression_result_type(expr.left),
            self.expression_result_type(expr.right),
        )
        if info is None:
            return None
        left = self.generate_numeric_vector_binary_operand(
            expr.left, info[0], "vector comparison"
        )
        right = self.generate_numeric_vector_binary_operand(
            expr.right, info[0], "vector comparison"
        )
        if self.vector_type_info(self.expression_result_type(expr.left)) is None:
            left = self.scalar_splat_expression(info[0], info[2], left)
        if (
            info[0] == "DType.bool"
            and self.vector_type_info(self.expression_result_type(expr.right)) is None
        ):
            # Mojo does not convert a Bool argument to a mask either.
            right = self.scalar_splat_expression(info[0], info[2], right)
        method = MOJO_VECTOR_COMPARISON_METHODS[self.map_operator(expr.op)]
        return f"({left}).{method}({right})"

    def generate_boolean_vector_logical_op(self, expr):
        """Combine boolean vector masks per component.

        Mojo's ``and`` and ``or`` return one whole operand, chosen by whether
        any lane of the left mask is true, so ``&&`` and ``||`` on boolean
        vectors are written with the lane-wise ``&`` and ``|``. A scalar
        ``bool`` operand of a mask is splatted to the mask's lanes. Two scalar
        ``bool`` operands keep ``and`` and ``or``, which short-circuit.
        """
        op = self.map_operator(expr.op)
        if op not in {"and", "or"}:
            return None
        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        left_mask = left_info is not None and left_info[0] == "DType.bool"
        right_mask = right_info is not None and right_info[0] == "DType.bool"
        if left_mask and right_mask:
            if left_info != right_info:
                raise MojoRefusal(
                    "Mojo boolean vector && and || require masks of the same width"
                )
        elif not (
            (left_mask and MOJO_SCALAR_DTYPES.get(right_type) == "DType.bool")
            or (right_mask and MOJO_SCALAR_DTYPES.get(left_type) == "DType.bool")
        ):
            return None
        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        # Mojo does not convert a Bool to a mask, so splat it explicitly.
        if not left_mask:
            left = f"SIMD[DType.bool, {right_info[2]}](fill={left})"
        if not right_mask:
            right = f"SIMD[DType.bool, {left_info[2]}](fill={right})"
        lane_op = "&" if op == "and" else "|"
        return f"({left} {lane_op} {right})"

    def generate_boolean_vector_reduction(self, name, args):
        if name not in {"any", "all"} or len(args) != 1:
            return None
        info = self.vector_type_info(self.expression_result_type(args[0]))
        if info is None or info[0] != "DType.bool":
            return None
        mask = self.generate_expression(args[0])
        if info[1] != info[2]:
            # Three-lane vectors use four storage lanes. Ignore padding without
            # evaluating a side-effecting argument more than once.
            used, padding, op = (
                ("True", "False", "&") if name == "any" else ("False", "True", "|")
            )
            lanes = ", ".join([used] * info[1] + [padding] * (info[2] - info[1]))
            mask = f"({mask} {op} SIMD[DType.bool, {info[2]}]({lanes}))"
        reduction = "reduce_or" if name == "any" else "reduce_and"
        return f"({mask}).{reduction}()"

    def generate_vector_binary_op(self, expr):
        op = self.map_operator(expr.op)
        if op not in MOJO_VECTOR_ARITHMETIC_OPS:
            return None

        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        left_is_vec3 = left_info is not None and left_info[1] == 3
        right_is_vec3 = right_info is not None and right_info[1] == 3

        if not left_is_vec3 and not right_is_vec3:
            return None

        if left_is_vec3 and right_is_vec3:
            if left_info[0] != right_info[0]:
                return None
            dtype = left_info[0]
            helper_kind = "vv"
        elif left_is_vec3:
            if right_info is not None or self.matrix_type_info(right_type) is not None:
                return None
            dtype = left_info[0]
            helper_kind = "vs"
        else:
            if left_info is not None or self.matrix_type_info(left_type) is not None:
                return None
            dtype = right_info[0]
            helper_kind = "sv"

        if dtype == "DType.bool" or dtype not in MOJO_DTYPE_SUFFIX:
            return None

        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        helper_name = self.vector_binary_helper_name(dtype, op, helper_kind)
        self.required_helpers.add((dtype, op, helper_kind))
        return f"{helper_name}({left}, {right})"
