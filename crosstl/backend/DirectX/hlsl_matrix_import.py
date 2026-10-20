"""Import HLSL matrices with CrossGL's column-by-row meaning (ADR 0003).

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

from ..common_ast import InitializerListNode
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

# HLSL scalar spelling -> CrossGL matrix prefix. Integer and boolean matrices have no CrossGL spelling.
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
_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
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


def crossgl_matrix_type_name(scalar, rows, cols):
    """CrossGL spelling of the HLSL ``scalar`` matrix with ``rows`` rows and ``cols`` columns."""
    prefix = _MATRIX_PREFIX.get(scalar)
    if prefix is None or not (2 <= rows <= 4 and 2 <= cols <= 4):
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
            if _operator_text(expr.op) not in _ARITHMETIC_OPERATORS:
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
        return None

    def matrix_operand_is_array(self, expr):
        """Whether ``expr`` names an array (of matrices) rather than one element of it."""
        depth = 0
        base = expr
        while isinstance(base, ArrayAccessNode):
            depth += 1
            base = base.array
        if isinstance(base, MemberAccessNode):
            owner = self.raw_type_base(self.expression_raw_type(base.object))
            dims = self.struct_member_array_dims.get(owner, {}).get(base.member) or 0
        elif isinstance(base, (str, VariableNode)):
            name = base if isinstance(base, str) else base.name
            dims = self.current_resource_array_dims.get(
                name, self.global_resource_array_dims.get(name, 0)
            )
        else:
            return False
        return depth < (dims or 0)

    # ------------------------------------------------------------- rendering

    def matrix_crossgl_type(self, shape):
        return crossgl_matrix_type_name(*shape)

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
            parts = [f"{rendered}[{col}][{row}]" for row, col in elements]
            if len(parts) == 1:
                return parts[0]
            vector_type = self.matrix_vector_type(shape[0], len(parts))
            return f"{vector_type}({', '.join(parts)})"
        access = self.matrix_row_access(expr.object)
        if access is None or not isinstance(member, str) or len(member) != 1:
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
        rendered = ", ".join(self.generate_expression(arg, is_main) for arg in args)
        return f"transpose({self.matrix_transposed_crossgl_type(shape)}({rendered}))"

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
            return (
                f"transpose({self.matrix_transposed_crossgl_type(shape)}({rendered}))"
            )
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        ):
            return f"{target}({rendered})"
        if raw_type not in _SCALAR_TYPES and not isinstance(value, (int, float, bool)):
            return self.refuse_import(
                "HLSL matrix conversion from a value whose type the importer cannot determine",
                f"{target}({rendered})",
            )
        if not self.matrix_operand_is_pure(value):
            return self.refuse_import(
                "HLSL scalar-to-matrix conversion of an expression with side effects",
                f"{target}({rendered})",
            )
        column_type = self.matrix_vector_type(shape[0], shape[1])
        columns = ", ".join(f"{column_type}({rendered})" for _ in range(shape[2]))
        return f"{target}({columns})"

    def matrix_initializer_expression(self, expected_type, initializer, is_main):
        """A brace initializer of an HLSL matrix lists its rows; render the equal CrossGL constructor."""
        shape = self.hlsl_matrix_shape(expected_type)
        if shape is None or not isinstance(initializer, InitializerListNode):
            return None
        size = shape[1] * shape[2]
        leaves = self.flatten_initializer(initializer)
        counts = [self.initializer_component_count(leaf) for leaf in leaves]
        if None not in counts and sum(counts) == size:
            rendered = ", ".join(
                self.generate_expression(leaf, is_main) for leaf in leaves
            )
            return (
                f"transpose({self.matrix_transposed_crossgl_type(shape)}({rendered}))"
            )
        elements = list(initializer.elements or [])
        groups = [self.flatten_initializer(element) for element in elements]
        if elements and all(
            None not in (
                c := [self.initializer_component_count(leaf) for leaf in group]
            )
            and sum(c) == size
            for group in groups
        ):
            matrices = ", ".join(
                f"transpose({self.matrix_transposed_crossgl_type(shape)}("
                + ", ".join(self.generate_expression(leaf, is_main) for leaf in group)
                + "))"
                for group in groups
            )
            return f"{{{matrices}}}"
        return self.refuse_import(
            f"HLSL initializer list for {shape[0]}{shape[1]}x{shape[2]} whose component count "
            "the importer cannot match to the matrix",
            "{}",
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
        return False

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
