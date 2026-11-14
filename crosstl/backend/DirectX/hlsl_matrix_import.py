"""Import HLSL matrices with CrossGL's column-by-row meaning (ADR 0004).

HLSL ``floatRxC`` has R rows and C columns: ``m[i]`` is row ``i``, constructors and initializer lists fill rows,
``m._mRC`` (zero-based) and ``m._RC`` (one-based) name element (R, C), ``*`` between two matrices is componentwise
and ``mul`` is the linear-algebra product. CrossGL follows GLSL, so the same matrix is ``matCxR``: C columns of
R components, ``m[c][r]`` is element (r, c), constructors fill columns and ``*`` is the linear-algebra product.

HLSL's default ``column_major`` packing stores the C columns of a ``floatRxC`` the way std140 and std430 store the
columns of a ``matCxR``, and ``row_major`` stores the R rows as GLSL's ``row_major`` does, so the layout qualifiers
carry over unchanged.
"""

from __future__ import annotations

import re

from ..common_ast import InitializerListNode, ReturnNode
from .DirectxAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)

# HLSL scalar spelling -> CrossGL matrix prefix.
_MATRIX_PREFIX = {
    "float": "mat",
    "fixed": "mat",
    "half": "f16mat",
    "min16float": "f16mat",
    "min10float": "f16mat",
    "float16_t": "f16mat",
    "double": "dmat",
}

_HLSL_MATRIX = re.compile(
    r"(float|half|fixed|double|min16float|min10float|float16_t|int|uint|bool|"
    r"min16int|min12int|int16_t|min16uint|uint16_t|int64_t|uint64_t)([1-4])x([1-4])"
)
_ZERO_BASED_ELEMENTS = re.compile(r"(?:_m[0-3][0-3])+")
_ONE_BASED_ELEMENTS = re.compile(r"(?:_[1-4][1-4])+")
_ELEMENTWISE_INTRINSICS = frozenset(
    {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "clamp",
        "cos",
        "cosh",
        "degrees",
        "exp",
        "exp2",
        "floor",
        "fmod",
        "frac",
        "lerp",
        "log",
        "log10",
        "log2",
        "mad",
        "max",
        "min",
        "pow",
        "radians",
        "round",
        "rsqrt",
        "saturate",
        "sign",
        "sin",
        "sinh",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "tanh",
        "trunc",
    }
)
# Always a scalar, whatever the operands.
_SCALAR_RESULT_INTRINSICS = frozenset({"determinant", "distance", "dot", "length"})
# Vector or matrix results whatever the operands; these are never splatted.
_AGGREGATE_RESULT_INTRINSICS = frozenset({"transpose"})
_RAYTRACING_MATRIX_INTRINSICS = {
    "ObjectToWorld": "float3x4",
    "ObjectToWorld3x4": "float3x4",
    "ObjectToWorld4x3": "float4x3",
    "WorldToObject": "float3x4",
    "WorldToObject3x4": "float3x4",
    "WorldToObject4x3": "float4x3",
}
_HIT_OBJECT_MATRIX_METHODS = {
    "GetObjectToWorld3x4": "float3x4",
    "GetObjectToWorld4x3": "float4x3",
    "GetWorldToObject3x4": "float3x4",
    "GetWorldToObject4x3": "float4x3",
}
_RAY_QUERY_MATRIX_METHODS = {
    "CandidateObjectToWorld3x4": "float3x4",
    "CandidateObjectToWorld4x3": "float4x3",
    "CandidateWorldToObject3x4": "float3x4",
    "CandidateWorldToObject4x3": "float4x3",
    "CommittedObjectToWorld3x4": "float3x4",
    "CommittedObjectToWorld4x3": "float4x3",
    "CommittedWorldToObject3x4": "float3x4",
    "CommittedWorldToObject4x3": "float4x3",
}
_HIT_OBJECT_STATIC_ARITIES = {
    "FromRayQuery": frozenset({1, 3}),
    "MakeMiss": frozenset({3}),
    "MakeNop": frozenset({0}),
    "TraceRay": frozenset({8}),
}
# Scalar for scalar operands, the operands' shape otherwise.
_OPERAND_SHAPED_INTRINSICS = frozenset({"and", "mul", "or", "select"})
_FLOAT_TYPES = frozenset(
    {"double", "fixed", "float", "float16_t", "half", "min10float", "min16float"}
)
_FAMILY_RANK = {"bool": 0, "int": 1, "float": 2}
# Keep the shape of the operand, only changing the element type.
_REINTERPRET_INTRINSICS = frozenset({"asfloat", "asint", "asuint"})
_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
# Componentwise on integer matrices, keeping the shape.
_BITWISE_OPERATORS = frozenset({"<<", ">>", "&", "|", "^"})
_COMPARISON_OPERATORS = frozenset({"<", ">", "<=", ">=", "==", "!="})
_SCALAR_TYPES = frozenset(
    {
        "bool",
        "double",
        "dword",
        "fixed",
        "float",
        "float16_t",
        "half",
        "int",
        "int16_t",
        "int64_t",
        "min10float",
        "min12int",
        "min16float",
        "min16int",
        "min16uint",
        "uint",
        "uint16_t",
        "uint64_t",
    }
)
# Intrinsics whose vector arguments DXC truncates to a common (narrowest) width, with the argument positions that
# take part (``None``: every argument). ``refract``'s index of refraction and ``ldexp``-style exponents stay out.
_TRUNCATING_INTRINSICS = {
    "dot": None,
    "distance": None,
    "reflect": None,
    "refract": (0, 1),
    "faceforward": None,
    "min": None,
    "max": None,
    "clamp": None,
    "lerp": None,
    "step": None,
    "smoothstep": None,
    "pow": None,
    "atan2": None,
    "fmod": None,
}


# HLSL integer and boolean matrix scalars -> the scalar of their CrossGL spelling.
_INTEGER_MATRIX_SCALAR = {"int": "int", "uint": "uint", "bool": "bool"}


def crossgl_matrix_type_name(scalar, rows, cols):
    """CrossGL spelling of the HLSL ``scalar`` matrix with ``rows`` rows and ``cols`` columns.

    Floating-point matrices are ``matCxR``. Integer and boolean matrices keep HLSL's row-first spelling
    ``intRxC`` with CrossGL's column indexing, so ``int3x2`` is the integer counterpart of ``mat2x3``.
    """
    if not (2 <= rows <= 4 and 2 <= cols <= 4):
        return None
    integer_scalar = _INTEGER_MATRIX_SCALAR.get(scalar)
    if integer_scalar is not None:
        return f"{integer_scalar}{rows}x{cols}"
    prefix = _MATRIX_PREFIX.get(scalar)
    if prefix is None:
        return None
    return f"{prefix}{cols}" if rows == cols else f"{prefix}{cols}x{rows}"


def hlsl_matrix_type_map():
    """``floatRxC`` -> ``matCxR`` for every HLSL floating-point matrix spelling."""
    return {
        f"{scalar}{rows}x{cols}": crossgl_matrix_type_name(scalar, rows, cols)
        for scalar in _MATRIX_PREFIX
        for rows in range(2, 5)
        for cols in range(2, 5)
    }


def _operator_text(op):
    return getattr(op, "value", op)


class HLSLMatrixImportMixin:
    """Matrix lowerings of :class:`HLSLToCrossGLConverter`."""

    # ----------------------------------------------------------------- shapes

    def hlsl_matrix_shape(self, raw_type):
        """``(scalar, rows, cols)`` of an HLSL matrix type with two to four rows and columns, else ``None``."""
        if not raw_type:
            return None
        text = self.resolve_integer_alias(str(raw_type).strip())
        if text == "matrix":
            return "float", 4, 4
        if text.startswith("matrix<") and text.endswith(">"):
            args = self.split_generic_arguments(text[len("matrix<") : -1])
            scalar = self.canonical_composite_type(args[0]) if args else "float"
            rows = self.parse_template_dimension(args[1]) if len(args) > 1 else 4
            cols = self.parse_template_dimension(args[2]) if len(args) > 2 else 4
        else:
            match = _HLSL_MATRIX.fullmatch(text)
            if match is None:
                return None
            scalar, rows, cols = (
                match.group(1),
                int(match.group(2)),
                int(match.group(3)),
            )
        if rows is None or cols is None or not (2 <= rows <= 4 and 2 <= cols <= 4):
            return None
        return scalar, rows, cols

    def matrix_row_raw_type(self, raw_type):
        """HLSL type of ``m[i]`` for a matrix ``m``: a vector with one component per column."""
        shape = self.hlsl_matrix_shape(raw_type)
        if shape is None:
            return None
        return f"{shape[0]}{shape[2]}"

    def matrix_expression_shape(self, expr):
        """Matrix shape of the value of ``expr``; ``None`` for non-matrices, arrays and unknown types."""
        cache = self.__dict__.setdefault("_matrix_shape_cache", {})
        key = (id(expr), id(self.current_variable_types))
        if key not in cache:
            cache[key] = (expr, self._matrix_expression_shape(expr))
        return cache[key][1]

    def _matrix_expression_shape(self, expr):
        if isinstance(expr, BinaryOpNode):
            operator = _operator_text(expr.op)
            if operator in _COMPARISON_OPERATORS:
                shape = self.matrix_expression_shape(
                    expr.left
                ) or self.matrix_expression_shape(expr.right)
                return None if shape is None else ("bool", shape[1], shape[2])
            if operator not in _ARITHMETIC_OPERATORS | _BITWISE_OPERATORS:
                return None
            return self.matrix_expression_shape(
                expr.left
            ) or self.matrix_expression_shape(expr.right)
        if isinstance(expr, UnaryOpNode):
            return self.matrix_expression_shape(expr.operand)
        if isinstance(expr, TernaryOpNode):
            return self.matrix_expression_shape(
                expr.true_expr
            ) or self.matrix_expression_shape(expr.false_expr)
        if isinstance(expr, AssignmentNode):
            return self.matrix_expression_shape(expr.left)
        if isinstance(expr, CastNode):
            return self.hlsl_matrix_shape(expr.target_type)
        if isinstance(expr, VectorConstructorNode):
            return self.hlsl_matrix_shape(expr.type_name)
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            return self._matrix_call_shape(expr)
        if self.matrix_operand_is_array(expr):
            return None
        try:
            raw_type = self.expression_value_raw_type(expr)
        except ValueError:
            return None
        return self.hlsl_matrix_shape(raw_type)

    def _matrix_call_shape(self, expr):
        name = self.normalize_hlsl_intrinsic_name(expr.name)
        args = list(expr.args or [])
        if name in self.source_function_names:
            try:
                return self.hlsl_matrix_shape(self.function_call_value_raw_type(expr))
            except ValueError:
                return None
        constructed = self.hlsl_matrix_shape(name)
        if constructed is not None:
            return constructed
        if name == "mul" and len(args) == 2:
            left = self.matrix_expression_shape(args[0])
            right = self.matrix_expression_shape(args[1])
            if left is not None and right is not None:
                return left[0], left[1], right[2]
            return None
        if name == "transpose" and len(args) == 1:
            shape = self.matrix_expression_shape(args[0])
            return None if shape is None else (shape[0], shape[2], shape[1])
        if name in _ELEMENTWISE_INTRINSICS:
            for arg in args:
                shape = self.matrix_expression_shape(arg)
                if shape is not None:
                    return shape
        return self.hlsl_matrix_shape(self.raytracing_matrix_call_raw_type(expr))

    def is_builtin_hit_object_type(self, raw_type):
        if not raw_type:
            return False
        base = self.raw_type_base(self.resolve_integer_alias(raw_type))
        if base == "dx::HitObject":
            return "dx::HitObject" not in self.struct_type_names
        return base == "HitObject" and "HitObject" not in self.struct_type_names

    def is_builtin_ray_query_type(self, raw_type):
        if not raw_type:
            return False
        base = self.raw_type_base(self.resolve_integer_alias(raw_type))
        return base == "RayQuery" and "RayQuery" not in self.struct_type_names

    def hit_object_static_call_raw_type(self, expr):
        if not (isinstance(expr, FunctionCallNode) and isinstance(expr.name, str)):
            return None
        if expr.name in self.source_function_names:
            return None
        scoped = expr.name.removeprefix("::")
        if "::" not in scoped:
            return None
        owner, member = scoped.rsplit("::", 1)
        arities = _HIT_OBJECT_STATIC_ARITIES.get(member)
        if arities is None or len(expr.args or []) not in arities:
            return None
        if owner in getattr(self, "struct_methods", {}):
            return None
        return "dx::HitObject" if self.is_builtin_hit_object_type(owner) else None

    def raytracing_matrix_call_raw_type(self, expr):
        if not isinstance(expr, FunctionCallNode) or expr.args:
            return None
        if isinstance(expr.name, str):
            if expr.name in self.source_function_names:
                return None
            name = self.normalize_hlsl_intrinsic_name(expr.name)
            if name in self.source_function_names:
                return None
            return _RAYTRACING_MATRIX_INTRINSICS.get(name)
        if isinstance(expr.name, MemberAccessNode):
            member = expr.name.member
            if not isinstance(member, str):
                return None
            if (
                member not in _HIT_OBJECT_MATRIX_METHODS
                and member not in _RAY_QUERY_MATRIX_METHODS
            ):
                return None
            if self.matrix_operand_is_array(expr.name.object):
                return None
            try:
                receiver_type = self.struct_method_receiver_type(expr.name.object)
            except ValueError:
                return None
            if self.is_builtin_hit_object_type(receiver_type):
                return _HIT_OBJECT_MATRIX_METHODS.get(member)
            if self.is_builtin_ray_query_type(receiver_type):
                return _RAY_QUERY_MATRIX_METHODS.get(member)
        return None

    def matrix_operand_is_array(self, expr):
        """Whether ``expr`` names an array (of matrices) rather than one element of it."""
        depth = 0
        base = expr
        while isinstance(base, ArrayAccessNode):
            depth += 1
            base = base.array
        if isinstance(base, MemberAccessNode):
            dims = self.expression_struct_member_array_dims(base) or 0
        elif isinstance(base, (str, VariableNode)):
            name = base if isinstance(base, str) else base.name
            dims = self.current_resource_array_dims.get(
                name, self.global_resource_array_dims.get(name, 0)
            )
        else:
            return False
        return depth < (dims or 0)

    # ------------------------------------------------------------- rendering

    def hlsl_matrix_rows_are_columns(self):
        """Whether CrossGL holds each HLSL row as a column.

        With row-major packing (DXC ``-Zpr``) the importer represents HLSL ``floatRxC`` by its transpose, the
        CrossGL ``matRxC``, so the CrossGL column-major storage has the bytes of the HLSL rows. Rows are then
        CrossGL columns: ``m[i]``, ``m[i][j]``, ``m._mRC``, constructors and initializers keep their order, and
        :meth:`generate_mul_expression` swaps the operands of ``mul``.
        """
        return getattr(self, "matrix_packing", None) == "row_major"

    def matrix_crossgl_type(self, shape):
        scalar, rows, cols = shape
        if self.hlsl_matrix_rows_are_columns():
            return crossgl_matrix_type_name(scalar, cols, rows)
        return crossgl_matrix_type_name(scalar, rows, cols)

    def matrix_transposed_crossgl_type(self, shape):
        """CrossGL type whose columns are the HLSL rows: ``matRxC`` for HLSL ``floatRxC``."""
        scalar, rows, cols = shape
        return crossgl_matrix_type_name(scalar, cols, rows)

    def matrix_vector_type(self, scalar, width):
        return self.map_type(f"{scalar}{width}" if width > 1 else scalar)

    def postfix_operand(self, expr, rendered):
        if isinstance(expr, (BinaryOpNode, TernaryOpNode, AssignmentNode, UnaryOpNode)):
            return f"({rendered})"
        return rendered

    def matrix_row_access(self, expr):
        """``(matrix, row)`` when ``expr`` is ``m[row]`` on a matrix value ``m``."""
        if not isinstance(expr, ArrayAccessNode):
            return None
        if self.matrix_expression_shape(expr.array) is None:
            return None
        return expr.array, expr.index

    def matrix_index_expression(self, expr, is_main):
        """Render ``m[i]`` (an HLSL row) or ``m[i][j]`` (element (i, j)) with CrossGL's column-first indexing."""
        if self.hlsl_matrix_rows_are_columns():
            return None
        inner = self.matrix_row_access(getattr(expr, "array", None))
        if inner is not None:
            matrix, row = inner
            rendered = self.postfix_operand(
                matrix, self.generate_expression(matrix, is_main)
            )
            row_text = self.generate_expression(row, is_main)
            column_text = self.generate_expression(expr.index, is_main)
            return f"{rendered}[{column_text}][{row_text}]"
        access = self.matrix_row_access(expr)
        if access is None:
            return None
        matrix, row = access
        rendered = self.generate_expression(matrix, is_main)
        return f"transpose({rendered})[{self.generate_expression(row, is_main)}]"

    def matrix_element_members(self, member):
        """``[(row, col), ...]`` for an HLSL ``_mRC`` or ``_RC`` matrix swizzle, else ``None``."""
        if not isinstance(member, str):
            return None
        if _ZERO_BASED_ELEMENTS.fullmatch(member):
            return [
                (int(member[i + 2]), int(member[i + 3]))
                for i in range(0, len(member), 4)
            ]
        if _ONE_BASED_ELEMENTS.fullmatch(member):
            return [
                (int(member[i + 1]) - 1, int(member[i + 2]) - 1)
                for i in range(0, len(member), 3)
            ]
        return None

    def matrix_member_expression(self, expr, is_main):
        """Render ``m._mRC...`` element swizzles and single-component swizzles of an HLSL row."""
        member = getattr(expr, "member", None)
        shape = self.matrix_expression_shape(expr.object)
        elements = self.matrix_element_members(member) if shape else None
        if elements:
            rendered = self.postfix_operand(
                expr.object, self.generate_expression(expr.object, is_main)
            )
            if any(row >= shape[1] or col >= shape[2] for row, col in elements):
                return self.refuse_import(
                    f"HLSL matrix swizzle .{member} is outside {shape[0]}{shape[1]}x{shape[2]}",
                    rendered,
                )
            if self.hlsl_matrix_rows_are_columns():
                parts = [f"{rendered}[{row}][{col}]" for row, col in elements]
            else:
                parts = [f"{rendered}[{col}][{row}]" for row, col in elements]
            if len(parts) == 1:
                return parts[0]
            vector_type = self.matrix_vector_type(shape[0], len(parts))
            return f"{vector_type}({', '.join(parts)})"
        access = self.matrix_row_access(expr.object)
        if (
            access is None
            or not isinstance(member, str)
            or len(member) != 1
            or self.hlsl_matrix_rows_are_columns()
        ):
            return None
        column = "xyzw".find(member) if member in "xyzw" else "rgba".find(member)
        if column < 0:
            return None
        matrix, row = access
        rendered = self.postfix_operand(
            matrix, self.generate_expression(matrix, is_main)
        )
        return f"{rendered}[{column}][{self.generate_expression(row, is_main)}]"

    def matrix_constructor_expression(self, type_name, args, is_main):
        """``floatRxC(...)`` fills rows; the CrossGL ``matRxC`` with the same arguments, transposed, is equal."""
        if not isinstance(type_name, str) or type_name in self.source_function_names:
            return None
        shape = self.hlsl_matrix_shape(type_name)
        if shape is None:
            return None
        args = list(args or [])
        if len(args) == 1:
            return self.matrix_conversion_expression(shape, args[0], is_main)
        reordered = self.column_filled_matrix_expression(shape, args, is_main)
        if reordered is not None:
            return reordered
        rendered = ", ".join(self.generate_expression(arg, is_main) for arg in args)
        return self.row_filled_matrix_expression(shape, rendered)

    def row_filled_matrix_expression(self, shape, rendered):
        """CrossGL value of the HLSL matrix whose rows the rendered argument list fills."""
        if self.hlsl_matrix_rows_are_columns():
            return f"{self.matrix_crossgl_type(shape)}({rendered})"
        return f"transpose({self.matrix_transposed_crossgl_type(shape)}({rendered}))"

    def column_filled_matrix_expression(self, shape, leaves, is_main):
        """Row-major scalar elements of side-effect-free leaves, re-listed column by column.

        The result stays a constant expression when the elements are (``static const`` matrices), which
        ``transpose(...)`` is not in GLSL ES 3.00 or glslang's global initializers.
        """
        _, rows, cols = shape
        if len(leaves) != rows * cols:
            return None
        if any(
            self.initializer_component_count(leaf) != 1
            or not self.matrix_operand_is_pure(leaf)
            for leaf in leaves
        ):
            return None
        rendered = [self.generate_expression(leaf, is_main) for leaf in leaves]
        if self.hlsl_matrix_rows_are_columns():
            ordered = rendered
        else:
            ordered = [
                rendered[row * cols + col] for col in range(cols) for row in range(rows)
            ]
        return f"{self.matrix_crossgl_type(shape)}({', '.join(ordered)})"

    def matrix_conversion_expression(self, shape, value, is_main):
        """``(floatRxC)value`` and one-argument constructors: truncate a matrix, splat a scalar, reshape a vector."""
        rendered = self.generate_expression(value, is_main)
        target = self.matrix_crossgl_type(shape)
        if self.matrix_expression_shape(value) is not None:
            # GLSL keeps the upper-left columns and rows, the elements HLSL truncation keeps.
            return f"{target}({rendered})"
        try:
            raw_type = self.resolve_integer_alias(self.expression_value_raw_type(value))
        except ValueError:
            raw_type = None
        if raw_type and self.vector_element_raw_type(raw_type):
            # A vector lists the elements row by row.
            return self.row_filled_matrix_expression(shape, rendered)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        ):
            return f"{target}({rendered})"
        if (
            raw_type not in _SCALAR_TYPES
            and not isinstance(value, (int, float, bool))
            and self.hlsl_value_kind(value) != "scalar"
        ):
            return self.refuse_import(
                "HLSL matrix conversion from a value whose type the importer cannot determine",
                f"{target}({rendered})",
            )
        if not self.matrix_operand_is_pure(value):
            return self.refuse_import(
                "HLSL scalar-to-matrix conversion of an expression with side effects",
                f"{target}({rendered})",
            )
        column_length, column_count = shape[1], shape[2]
        if self.hlsl_matrix_rows_are_columns():
            column_length, column_count = column_count, column_length
        column_type = self.matrix_vector_type(shape[0], column_length)
        element = self.splat_scalar_text(shape, value, rendered)
        columns = ", ".join(f"{column_type}({element})" for _ in range(column_count))
        return f"{target}({columns})"

    def scalar_matrix_splat_expression(self, shape, value, is_main):
        """The splat of a scalar ``value`` stored to a matrix of ``shape``, or ``None`` for any other value.

        HLSL repeats a scalar into every component of the matrix it is stored to, so ``float3x3 m = 1.0;`` is the
        all-ones matrix. CrossGL reads ``mat3 m = 1.0;`` as a type error and ``mat3(1.0)`` as the identity, so the
        importer writes the same splat the cast ``(float3x3)v`` gets.
        """
        if shape is None or isinstance(value, InitializerListNode):
            return None
        if self.matrix_expression_shape(value) is not None:
            return None
        if self.hlsl_value_kind(value) == "other":
            return None
        # A value whose type cannot be worked out is refused by the conversion.
        return self.matrix_conversion_expression(shape, value, is_main)

    def hlsl_value_kind(self, expr):
        """``"scalar"`` or ``"other"`` for a value of known type, ``None`` when the type cannot be told.

        ``expression_value_raw_type`` only types integer arithmetic, so float expressions such as ``x * 2.0 + 1.0``,
        ``c ? x : 1.0`` and ``sin(x)`` are typed here from their operands.
        """
        if isinstance(expr, (bool, int, float)):
            return "scalar"
        if isinstance(expr, CastNode):
            target = self.resolve_integer_alias(str(expr.target_type).strip())
            if target in _SCALAR_TYPES:
                return "scalar"
        elif isinstance(expr, UnaryOpNode):
            if _operator_text(expr.op) not in {"++", "--"}:
                return self.hlsl_value_kind(expr.operand)
        elif isinstance(expr, BinaryOpNode):
            if _operator_text(
                expr.op
            ) in _ARITHMETIC_OPERATORS | _BITWISE_OPERATORS | _COMPARISON_OPERATORS | {
                "&&",
                "||",
            }:
                return self._combined_value_kind([expr.left, expr.right])
        elif isinstance(expr, TernaryOpNode):
            return self._combined_value_kind(
                [expr.condition, expr.true_expr, expr.false_expr]
            )
        elif isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            name = self.normalize_hlsl_intrinsic_name(expr.name)
            if name not in self.source_function_names:
                if name in _SCALAR_TYPES:
                    return "scalar"
                if name in _SCALAR_RESULT_INTRINSICS:
                    return "scalar"
                if name in _AGGREGATE_RESULT_INTRINSICS:
                    return "other"
                if (
                    name
                    in _ELEMENTWISE_INTRINSICS
                    | _REINTERPRET_INTRINSICS
                    | _OPERAND_SHAPED_INTRINSICS
                ):
                    return self._combined_value_kind(list(expr.args or []))
        try:
            raw_type = self.expression_value_raw_type(expr)
            raw_type = self.resolve_integer_alias(raw_type) if raw_type else None
        except ValueError:
            return None
        if not raw_type:
            return None
        return "scalar" if raw_type in _SCALAR_TYPES else "other"

    def hlsl_scalar_family(self, expr):
        """``"bool"``, ``"int"`` or ``"float"`` for the scalar type of ``expr``.

        ``"literal"`` is an integer literal, which takes the type it is stored as, and ``None`` is a value whose
        type the importer cannot tell. A vector reports the family of its elements.
        """
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, int):
            return "literal"
        if isinstance(expr, CastNode):
            return self._type_family(str(expr.target_type).strip())
        if isinstance(expr, UnaryOpNode):
            if _operator_text(expr.op) == "!":
                return "bool"
            return self.hlsl_scalar_family(expr.operand)
        if isinstance(expr, BinaryOpNode):
            operator = _operator_text(expr.op)
            if operator in _COMPARISON_OPERATORS | {"&&", "||"}:
                return "bool"
            if operator in _ARITHMETIC_OPERATORS | _BITWISE_OPERATORS:
                family = self._promoted_family([expr.left, expr.right])
                return "int" if family == "bool" else family
        elif isinstance(expr, TernaryOpNode):
            return self._promoted_family([expr.true_expr, expr.false_expr])
        elif isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            name = self.normalize_hlsl_intrinsic_name(expr.name)
            args = list(expr.args or [])
            if name not in self.source_function_names:
                if name in _SCALAR_TYPES:
                    return self._type_family(name)
                if name in {"length", "distance", "determinant", "asfloat"}:
                    return "float"
                if name in {"asint", "asuint"}:
                    return "int"
                if name in {"and", "or"}:
                    return "bool"
                if name == "select":
                    return self._promoted_family(args[1:])
                if name in {"dot", "mul"} | _ELEMENTWISE_INTRINSICS:
                    return self._promoted_family(args)
        try:
            raw_type = self.expression_value_raw_type(expr)
        except ValueError:
            return None
        if not raw_type:
            return None
        return self._type_family(self.vector_element_raw_type(raw_type) or raw_type)

    def _type_family(self, type_name):
        type_name = self.resolve_integer_alias(type_name)
        if type_name == "bool":
            return "bool"
        if type_name in _FLOAT_TYPES:
            return "float"
        return "int" if type_name in _SCALAR_TYPES else None

    def _promoted_family(self, operands):
        families = [self.hlsl_scalar_family(operand) for operand in operands]
        if not families or None in families:
            return None
        typed = [family for family in families if family != "literal"]
        if not typed:
            return "literal"
        return max(typed, key=_FAMILY_RANK.__getitem__)

    def splat_scalar_text(self, shape, value, rendered):
        """``rendered`` converted to the matrix's element type when ``value`` is a scalar of another one.

        WGSL's ``vec3<f32>(true)`` is a type error and ``dot`` of integer vectors is an integer, so a bool or
        integer value is written ``float(...)`` before it is repeated into the columns.
        """
        family = self.hlsl_scalar_family(value)
        if family in (None, "literal"):
            return rendered
        if family == self._type_family(shape[0]):
            return rendered
        return f"{self.map_type(shape[0])}({rendered})"

    def _combined_value_kind(self, operands):
        kinds = [self.hlsl_value_kind(operand) for operand in operands]
        if "other" in kinds:
            return "other"
        if None in kinds:
            return None
        return "scalar"

    def scalar_matrix_assignment_value(self, node, is_main):
        """The splat for ``m = scalar;``, or ``None`` when ``node`` is anything else."""
        if node.operator != "=":
            return None
        target = node.left
        if isinstance(target, MemberAccessNode) and self.matrix_element_members(
            target.member
        ):
            # ``m._m01 = s`` writes one element; the matrix_assignment path owns it.
            return None
        shape = self.matrix_expression_shape(target)
        return self.scalar_matrix_splat_expression(shape, node.right, is_main)

    def source_parameter_types(self, name, args):
        """Declared parameter types of the one source function ``name(args)`` calls.

        Every entry is ``None`` when the call is not to a source function or the
        overload set is ambiguous, so callers keep the argument as written.
        """
        unknown = [None] * len(args)
        candidates = [
            f for f in self.source_functions.get(name, []) if len(f.params) == len(args)
        ]
        if not candidates:
            return unknown
        signatures = {tuple(p.vtype for p in f.params) for f in candidates}
        if len(signatures) != 1:
            return unknown
        return list(signatures.pop())

    def matrix_initializer_expression(self, expected_type, initializer, is_main):
        """A brace initializer of an HLSL matrix lists its rows; render the equal CrossGL constructor."""
        shape = self.hlsl_matrix_shape(expected_type)
        if shape is None or not isinstance(initializer, InitializerListNode):
            return None
        size = shape[1] * shape[2]
        leaves = self.flatten_initializer(initializer)
        counts = [self.initializer_component_count(leaf) for leaf in leaves]
        if None not in counts and sum(counts) == size:
            reordered = self.column_filled_matrix_expression(shape, leaves, is_main)
            if reordered is not None:
                return reordered
            rendered = ", ".join(
                self.generate_expression(leaf, is_main) for leaf in leaves
            )
            return self.row_filled_matrix_expression(shape, rendered)
        elements = list(initializer.elements or [])
        matrices = [
            self.array_element_matrix_expression(shape, element, is_main)
            for element in elements
        ]
        if elements and None not in matrices:
            return f"{{{', '.join(matrices)}}}"
        return self.refuse_import(
            f"HLSL initializer list for {shape[0]}{shape[1]}x{shape[2]} whose component count "
            "the importer cannot match to the matrix",
            "{}",
        )

    def array_element_matrix_expression(self, shape, element, is_main):
        """One element of an array-of-matrices initializer, or ``None`` when its shape differs."""
        if not isinstance(element, InitializerListNode):
            element_shape = self.matrix_expression_shape(element)
            if element_shape is not None:
                if element_shape != shape:
                    return None
                return self.generate_expression(element, is_main)
        group = self.flatten_initializer(element)
        counts = [self.initializer_component_count(leaf) for leaf in group]
        if None in counts or sum(counts) != shape[1] * shape[2]:
            return None
        return self.column_filled_matrix_expression(
            shape, group, is_main
        ) or self.row_filled_matrix_expression(
            shape,
            ", ".join(self.generate_expression(leaf, is_main) for leaf in group),
        )

    def flatten_initializer(self, node):
        if isinstance(node, InitializerListNode):
            leaves = []
            for element in node.elements or []:
                leaves.extend(self.flatten_initializer(element))
            return leaves
        return [node]

    def initializer_component_count(self, leaf):
        if isinstance(leaf, (bool, int, float)):
            return 1
        shape = self.matrix_expression_shape(leaf)
        if shape is not None:
            return None
        try:
            raw_type = self.resolve_integer_alias(self.expression_value_raw_type(leaf))
        except ValueError:
            return None
        if raw_type and self.vector_element_raw_type(raw_type):
            return int(raw_type[-1])
        return 1 if raw_type in _SCALAR_TYPES else None

    def value_vector_width(self, expr):
        """Component count of a vector-valued ``expr``, ``None`` for scalars, matrices and unknown types."""
        if self.matrix_expression_shape(expr) is not None:
            return None
        try:
            raw_type = self.resolve_integer_alias(self.expression_value_raw_type(expr))
        except ValueError:
            return None
        if (
            raw_type
            and self.vector_element_raw_type(raw_type)
            and raw_type[-1].isdigit()
        ):
            return int(raw_type[-1])
        return None

    def mul_call_expression(self, args, rendered):
        """HLSL ``mul``: the linear-algebra product, with DXC's implicit vector truncation.

        ``mul(M, v)`` takes the first C components of ``v`` for a ``floatRxC`` ``M`` and ``mul(v, M)`` the first R;
        ``mul`` of two vectors is their inner product. Operand order and precise
        evaluation follow :meth:`generate_mul_expression`.
        """
        left_shape = self.matrix_expression_shape(args[0])
        right_shape = self.matrix_expression_shape(args[1])
        left, right = rendered
        if left_shape is not None and right_shape is None:
            right = self._truncated_mul_vector(args[1], right, left_shape[2])
        elif right_shape is not None and left_shape is None:
            left = self._truncated_mul_vector(args[0], left, right_shape[1])
        elif left_shape is None and right_shape is None:
            left_width = self.value_vector_width(args[0])
            right_width = self.value_vector_width(args[1])
            if left_width and right_width and left_width > 1 and right_width > 1:
                width = min(left_width, right_width)
                left = self._truncated_mul_vector(args[0], left, width)
                right = self._truncated_mul_vector(args[1], right, width)
                precise_dot = self.precise_intrinsic_call("dot", args, [left, right])
                return precise_dot or f"dot({left}, {right})"
        return self.generate_mul_expression(args, [left, right])

    def _truncated_mul_vector(self, expr, rendered, width):
        current = self.value_vector_width(expr)
        if current is None or current <= width:
            return rendered
        return self.swizzle_rendered_expression(rendered, "xyzw"[:width])

    def truncate_intrinsic_vector_arguments(self, name, args, rendered):
        """DXC resolves componentwise intrinsics over mixed vector widths by truncating to the narrowest vector."""
        if name not in _TRUNCATING_INTRINSICS or len(args) < 2:
            return rendered
        positions = _TRUNCATING_INTRINSICS[name] or range(len(args))
        widths = {
            index: self.value_vector_width(args[index])
            for index in positions
            if index < len(args)
        }
        known = [width for width in widths.values() if width and width > 1]
        if len(set(known)) < 2:
            return rendered
        narrowest = min(known)
        result = list(rendered)
        for index, width in widths.items():
            if width and width > narrowest:
                result[index] = self.swizzle_rendered_expression(
                    result[index], "xyzw"[:narrowest]
                )
        return result

    def matrix_product_expression(self, expr, left, right):
        """HLSL ``*`` of two matrices is componentwise."""
        if _operator_text(expr.op) != "*":
            return None
        if (
            self.matrix_expression_shape(expr.left) is None
            or self.matrix_expression_shape(expr.right) is None
        ):
            return None
        return f"matrixCompMult({left}, {right})"

    # ------------------------------------------------------------ assignment

    def matrix_assignment_expression(self, node, is_main):
        """Lower assignments HLSL spells through rows: ``m[i] = v``, ``m[i].xy = v``, ``m *= n``."""
        operator = _operator_text(getattr(node, "operator", "="))
        target = node.left
        if operator == "*=" and self.matrix_expression_shape(target) is not None:
            if self.matrix_expression_shape(node.right) is not None:
                if not self.matrix_operand_is_pure(target):
                    return self.refuse_import(
                        "HLSL componentwise matrix *= on a target with side effects", ""
                    )
                lhs = self.generate_expression(target, is_main)
                rhs = self.generate_expression(node.right, is_main)
                return f"{lhs} = matrixCompMult({lhs}, {rhs})"
            return None
        if self.hlsl_matrix_rows_are_columns():
            return None
        if isinstance(target, MemberAccessNode):
            elements = self.matrix_element_members(target.member)
            row_swizzle = self.matrix_row_access(target.object)
            if (
                elements
                and len(elements) > 1
                and self.matrix_expression_shape(target.object)
            ) or (row_swizzle is not None and len(str(target.member)) > 1):
                return self.refuse_import(
                    f"HLSL assignment through the multi-component matrix swizzle .{target.member}",
                    "",
                )
            return None
        access = self.matrix_row_access(target)
        if access is None:
            return None
        matrix, row = access
        if not (
            self.matrix_operand_is_pure(matrix) and self.matrix_operand_is_pure(row)
        ):
            return self.refuse_import(
                "HLSL matrix row assignment whose matrix or row index has side effects",
                "",
            )
        shape = self.matrix_expression_shape(matrix)
        matrix_text = self.generate_expression(matrix, is_main)
        row_text = self.generate_expression(row, is_main)
        value = self.generate_expression(node.right, is_main)
        if operator != "=":
            binary = operator[:-1]
            if binary not in _ARITHMETIC_OPERATORS:
                return self.refuse_import(
                    f"HLSL matrix row compound assignment {operator}", ""
                )
            value = f"transpose({matrix_text})[{row_text}] {binary} ({value})"
        helper = self.matrix_row_setter_name(shape)
        row_type = self.matrix_vector_type(shape[0], shape[2])
        try:
            value_type = self.resolve_integer_alias(
                self.expression_value_raw_type(node.right)
            )
        except ValueError:
            value_type = None
        if operator != "=" or value_type != f"{shape[0]}{shape[2]}":
            value = f"{row_type}({value})"
        return f"{helper}({matrix_text}, int({row_text}), {value})"

    def matrix_operand_is_pure(self, expr):
        """Whether ``expr`` can be evaluated twice: names, members, literals and side-effect-free indexing."""
        if isinstance(expr, (bool, int, float, str, VariableNode)):
            return True
        if isinstance(expr, MemberAccessNode):
            return self.matrix_operand_is_pure(expr.object)
        if isinstance(expr, ArrayAccessNode):
            return self.matrix_operand_is_pure(
                expr.array
            ) and self.matrix_operand_is_pure(expr.index)
        if isinstance(expr, BinaryOpNode):
            return self.matrix_operand_is_pure(
                expr.left
            ) and self.matrix_operand_is_pure(expr.right)
        if isinstance(expr, UnaryOpNode):
            return _operator_text(expr.op) not in {
                "++",
                "--",
            } and self.matrix_operand_is_pure(expr.operand)
        if isinstance(expr, CastNode):
            return self.matrix_operand_is_pure(expr.expression)
        if isinstance(expr, TernaryOpNode):
            return all(
                self.matrix_operand_is_pure(part)
                for part in (expr.condition, expr.true_expr, expr.false_expr)
            )
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            if self.raytracing_matrix_call_raw_type(expr) is not None:
                return True
            name = self.normalize_hlsl_intrinsic_name(expr.name)
            return (
                name
                in _ELEMENTWISE_INTRINSICS
                | _SCALAR_RESULT_INTRINSICS
                | _AGGREGATE_RESULT_INTRINSICS
                | _REINTERPRET_INTRINSICS
                | _OPERAND_SHAPED_INTRINSICS
                and name not in self.source_function_names
                and all(self.matrix_operand_is_pure(arg) for arg in expr.args or [])
            )
        if isinstance(expr, FunctionCallNode) and isinstance(
            expr.name, MemberAccessNode
        ):
            if self.raytracing_matrix_call_raw_type(expr) is not None:
                return self.matrix_operand_is_pure(expr.name.object)
            method = self.source_struct_method(expr.name)
            return (
                method is not None
                and self.struct_method_is_pure(method)
                and self.matrix_operand_is_pure(expr.name.object)
                and all(self.matrix_operand_is_pure(arg) for arg in expr.args or [])
            )
        return False

    def source_struct_method(self, member):
        """The source struct method that ``object.member(...)`` calls, or ``None``."""
        owner = self.raw_type_base(self.expression_raw_type(member.object))
        method = self.struct_methods.get(owner, {}).get(
            self.templated_method_base(member.member)
        )
        if method is None or method.definition is None:
            return None
        return method

    def struct_method_is_pure(self, method):
        """Whether calling ``method`` twice does the same as calling it once.

        Only a body of declarations and a ``return`` over reads, intrinsics and arithmetic qualifies, and the
        method must neither modify its receiver nor take an ``out`` or ``inout`` parameter.
        """
        definition = method.definition
        if method.mutates_receiver or getattr(definition, "template_parameters", None):
            return False
        for param in getattr(definition, "params", []) or []:
            qualifiers = {
                str(q).lower() for q in getattr(param, "qualifiers", []) or []
            }
            if qualifiers & {"out", "inout"}:
                return False
        body = getattr(definition, "body", None)
        if not isinstance(body, list):
            return False
        for statement in body:
            if isinstance(statement, ReturnNode):
                value = statement.value
            elif isinstance(statement, VariableNode):
                value = statement.value
            else:
                return False
            if value is not None and not self.matrix_operand_is_pure(value):
                return False
        return True

    def matrix_row_setter_name(self, shape):
        helpers = self.__dict__.setdefault("matrix_row_setters", {})
        key = self.matrix_crossgl_type(shape)
        if key not in helpers:
            name = f"cgl_hlsl_set_row_{key}"
            while (
                name in self.integer_helper_reserved_names or name in helpers.values()
            ):
                name += "_"
            helpers[key] = (name, shape)
        return helpers[key][0]

    def generate_matrix_row_setters(self):
        code = ""
        for matrix_type, (name, shape) in sorted(
            self.__dict__.get("matrix_row_setters", {}).items()
        ):
            row_type = self.matrix_vector_type(shape[0], shape[2])
            code += f"    {row_type} {name}(inout {matrix_type} target, int row, {row_type} value) {{\n"
            for column in range(shape[2]):
                code += f"        target[{column}][row] = value.{'xyzw'[column]};\n"
            code += "        return value;\n    }\n"
        return code

    def reset_matrix_import_state(self):
        self.__dict__["matrix_row_setters"] = {}
        self.__dict__["_matrix_shape_cache"] = {}
