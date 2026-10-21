"""CrossGL emission of HLSL value shapes and constant expressions.

HLSL infers component widths, matrix shapes and scalar results from operand
types, swizzles matrices, folds constant initializers and zero-initializes
values. These members compute those shapes so the emitted CrossGL states them
explicitly: component widths, ``mul`` results, matrix swizzles, shape-preserving
type mapping, overload return types and zero values.
"""

import re

from ..common_ast import (
    ArrayAccessNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)


class HLSLValueShapeMixin:
    """HLSL value-shape and constant-expression members of ``HLSLToCrossGLConverter``."""

    # Intrinsics DXC evaluates while folding constant operands.
    HLSL_FOLDABLE_INTRINSICS = frozenset(
        {
            "abs",
            "ceil",
            "clamp",
            "cos",
            "degrees",
            "exp",
            "exp2",
            "floor",
            "frac",
            "lerp",
            "log",
            "log2",
            "max",
            "min",
            "pow",
            "radians",
            "round",
            "rsqrt",
            "saturate",
            "sign",
            "sin",
            "sqrt",
            "tan",
            "trunc",
        }
    )

    def zero_value_expression(self, hlsl_type, array_sizes=None, depth=0):
        """CrossGL spelling of HLSL's zero conversion ``(T)0`` for a type."""
        if depth > 16 or array_sizes:
            return None
        base = self.raw_type_base(hlsl_type)
        struct = getattr(self, "structs_by_name", {}).get(base)
        if struct is not None:
            values = []
            for member in getattr(struct, "members", []) or []:
                if not isinstance(member, VariableNode):
                    return None
                value = self.zero_value_expression(
                    getattr(member, "vtype", None),
                    getattr(member, "array_sizes", None),
                    depth + 1,
                )
                if value is None:
                    return None
                values.append(value)
            return f"{self.render_type_identifier(base)}({', '.join(values)})"
        mapped = str(self.map_type(hlsl_type) or "")
        family = self.numeric_type_family(self.canonical_composite_type(str(hlsl_type)))
        if self.canonical_composite_type(str(hlsl_type)).startswith("bool"):
            scalar = "false"
        elif family == "uint":
            scalar = "0u"
        elif family == "int":
            scalar = "0"
        elif family == "float" or mapped.startswith(("double", "dvec", "dmat")):
            scalar = "0.0"
        else:
            return None
        if mapped in {"bool", "int", "uint", "float", "double", "float16"}:
            return scalar
        return f"{mapped}({scalar})"

    def is_implicit_value_array_parameter(self, parameter):
        """Return whether a helper array parameter has HLSL copy-in semantics.

        HLSL passes every parameter without ``out`` or ``inout`` by value, so a
        helper that writes to such an array changes only its own copy. CrossGL
        leaves an unqualified array parameter's mutation mode open because
        pointer-like sources (Metal) alias the caller's array, so the importer
        states the HLSL contract with an explicit ``in`` qualifier. A
        ``groupshared`` parameter refers to the caller's workgroup array and is
        not a copy.
        """
        if not getattr(parameter, "array_sizes", None):
            return False
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        return not qualifiers & {
            "in",
            "out",
            "inout",
            "const",
            "uniform",
            "groupshared",
            "shared",
        }

    def matrix_row_raw_type(self, type_name):
        """HLSL ``floatRxC[i]`` is row ``i``: a ``floatC`` vector."""
        match = re.fullmatch(
            r"(min16float|min10float|min16uint|min16int|min12int|float16_t|"
            r"uint16_t|int16_t|double|float|half|fixed|uint|int|bool)([1-4])x([1-4])",
            self.canonical_composite_type(str(type_name or "").strip()),
        )
        if not match:
            return None
        scalar, _rows, cols = match.groups()
        return scalar if cols == "1" else f"{scalar}{cols}"

    def overload_return_raw_type(self, expr):
        """Return type of the one overload whose parameter widths match."""
        overloads = getattr(self, "function_overloads", {}).get(expr.name)
        if not overloads or len(overloads) < 2:
            return None
        widths = [self.hlsl_component_width(arg) for arg in expr.args]
        if None in widths:
            return None
        matches = {
            return_type
            for parameter_types, return_type in overloads
            if [self.raw_type_component_width(item) for item in parameter_types]
            == widths
        }
        return next(iter(matches)) if len(matches) == 1 else None

    def collect_static_const_initializers(self, ast):
        """Initializers of ``static const`` globals, which DXC folds."""
        initializers = {}
        for node in getattr(ast, "global_variables", []) or []:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            is_const = getattr(node, "is_const", False) or "const" in qualifiers
            name = getattr(node, "name", None)
            value = getattr(node, "value", None)
            if is_const and isinstance(name, str) and value is not None:
                initializers[name] = value
        return initializers

    HLSL_NUMERIC_LITERAL = re.compile(
        r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFhHlLuU]*"
    )

    def is_hlsl_constant_expression(self, expr, visiting=frozenset()):
        """Whether DXC evaluates ``expr`` at compile time.

        Literals, ``static const`` values with constant initializers and
        operators, casts, constructors and foldable intrinsics over them.
        """
        if isinstance(expr, (bool, int, float)):
            return True
        name = None
        if isinstance(expr, str):
            if self.HLSL_NUMERIC_LITERAL.fullmatch(expr.strip()):
                return True
            name = expr
        elif isinstance(expr, VariableNode) and not getattr(expr, "vtype", None):
            name = getattr(expr, "name", None)
        if name is not None:
            if name in visiting:
                return False
            # Function scopes start from the global types; a different type
            # means a local declaration shadows the global constant.
            if self.current_variable_types.get(name) != self.global_variable_types.get(
                name
            ):
                return False
            initializer = self.hlsl_static_const_initializers.get(name)
            return initializer is not None and self.is_hlsl_constant_expression(
                initializer, visiting | {name}
            )
        if isinstance(expr, UnaryOpNode):
            return self.is_hlsl_constant_expression(expr.operand, visiting)
        if isinstance(expr, BinaryOpNode):
            return self.is_hlsl_constant_expression(
                expr.left, visiting
            ) and self.is_hlsl_constant_expression(expr.right, visiting)
        if isinstance(expr, TernaryOpNode):
            return all(
                self.is_hlsl_constant_expression(part, visiting)
                for part in (expr.condition, expr.true_expr, expr.false_expr)
            )
        if isinstance(expr, CastNode):
            return self.is_hlsl_constant_expression(expr.expression, visiting)
        if isinstance(expr, MemberAccessNode):
            return self.is_hlsl_constant_expression(expr.object, visiting)
        if isinstance(expr, ArrayAccessNode):
            return self.is_hlsl_constant_expression(
                expr.array, visiting
            ) and self.is_hlsl_constant_expression(expr.index, visiting)
        if isinstance(expr, VectorConstructorNode):
            return all(
                self.is_hlsl_constant_expression(arg, visiting) for arg in expr.args
            )
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            callee = self.normalize_hlsl_intrinsic_name(expr.name)
            if callee in self.source_function_names:
                return False
            mapped = self.map_type(callee)
            is_constructor = isinstance(mapped, str) and re.fullmatch(
                r"(?:float|int|uint|bool|half|double|"
                r"(?:[ibud]|f16)?vec[2-4]|(?:d|f16)?mat[2-4](?:x[2-4])?)",
                mapped,
            )
            if is_constructor or callee in self.HLSL_FOLDABLE_INTRINSICS:
                return all(
                    self.is_hlsl_constant_expression(arg, visiting) for arg in expr.args
                )
        return False

    # Intrinsics whose result is one component.
    HLSL_SCALAR_RESULT_INTRINSICS = frozenset(
        {"determinant", "distance", "dot", "length", "noise"}
    )

    # Intrinsics whose result has the shape of their widest operand.
    HLSL_COMPONENTWISE_INTRINSICS = frozenset(
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
            "ddx",
            "ddx_coarse",
            "ddx_fine",
            "ddy",
            "ddy_coarse",
            "ddy_fine",
            "degrees",
            "exp",
            "exp2",
            "faceforward",
            "floor",
            "fmod",
            "frac",
            "fwidth",
            "ldexp",
            "lerp",
            "log",
            "log10",
            "log2",
            "mad",
            "max",
            "min",
            "normalize",
            "pow",
            "radians",
            "rcp",
            "reflect",
            "refract",
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

    def raw_type_component_width(self, raw_type):
        """Component count of a scalar or vector type name, otherwise ``None``."""
        if not raw_type:
            return None
        mapped = self.map_type(self.canonical_composite_type(str(raw_type)))
        if not isinstance(mapped, str):
            return None
        if mapped in {"float", "int", "uint", "bool", "half", "double"}:
            return 1
        vector = re.fullmatch(r"(?:[ibud]|f16)?vec([2-4])", mapped)
        return int(vector.group(1)) if vector else None

    def hlsl_component_width(self, expr):
        """Component count of a scalar or vector HLSL expression, or ``None``."""
        width = self.raw_type_component_width(self.expression_value_raw_type(expr))
        if width is not None:
            return width
        if isinstance(expr, (bool, int, float)):
            return 1
        if isinstance(expr, str) and self.HLSL_NUMERIC_LITERAL.fullmatch(expr.strip()):
            return 1
        if isinstance(expr, UnaryOpNode):
            return self.hlsl_component_width(expr.operand)
        if isinstance(expr, BinaryOpNode):
            widths = [self.hlsl_component_width(expr.left)]
            widths.append(self.hlsl_component_width(expr.right))
            return None if None in widths else max(widths)
        if isinstance(expr, TernaryOpNode):
            widths = [
                self.hlsl_component_width(expr.true_expr),
                self.hlsl_component_width(expr.false_expr),
            ]
            return None if None in widths else max(widths)
        if (
            isinstance(expr, MemberAccessNode)
            and isinstance(expr.member, str)
            and re.fullmatch(r"[xyzw]{1,4}|[rgba]{1,4}", expr.member)
            and self.is_swizzle_object(expr.object)
        ):
            return len(expr.member)
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            callee = self.normalize_hlsl_intrinsic_name(expr.name)
            if callee in self.source_function_names:
                return None
            if callee == "mul" and len(expr.args) == 2:
                return self.mul_result_width(expr.args)
            if callee in self.HLSL_SCALAR_RESULT_INTRINSICS:
                return 1
            if callee == "cross":
                return 3
            if callee in self.HLSL_COMPONENTWISE_INTRINSICS and expr.args:
                widths = [self.hlsl_component_width(arg) for arg in expr.args]
                return None if None in widths else max(widths)
        return None

    def is_swizzle_object(self, expr):
        """Whether a ``.xyzw``-style member of ``expr`` must be a swizzle.

        Scalars, vectors, operator results and intrinsic results cannot be
        structs, so the member selects components.
        """
        if self.hlsl_component_width(expr) is not None:
            return True
        if isinstance(expr, (BinaryOpNode, UnaryOpNode, TernaryOpNode)):
            return True
        return (
            isinstance(expr, FunctionCallNode)
            and isinstance(expr.name, str)
            and self.normalize_hlsl_intrinsic_name(expr.name)
            not in self.source_function_names
            and self.raw_type_base(self.normalize_hlsl_intrinsic_name(expr.name))
            not in getattr(self, "structs_by_name", {})
        )

    def mul_result_width(self, args):
        """Component count of an HLSL ``mul`` result, or ``None``."""
        shapes = [self.raw_value_shape(arg) for arg in args]
        if shapes == ["vector", "vector"]:
            return 1
        if "matrix" in shapes:
            matrix_index = shapes.index("matrix")
            if shapes[1 - matrix_index] != "vector":
                return None
            raw_type = self.canonical_composite_type(
                str(self.expression_value_raw_type(args[matrix_index]) or "")
            )
            dimensions = re.fullmatch(r"[a-z_0-9]*?([1-4])x([1-4])", raw_type)
            if dimensions is None:
                return None
            rows, cols = (int(value) for value in dimensions.groups())
            # mul(v, M) is a row vector times M; mul(M, v) a column vector.
            return cols if matrix_index == 1 else rows
        if "scalar" in shapes:
            return self.hlsl_component_width(
                args[1] if shapes[0] == "scalar" else args[0]
            )
        return None

    SCALAR_RESULT_INTRINSICS = frozenset(
        {"dot", "length", "distance", "determinant", "any", "all"}
    )

    def raw_value_shape(self, expr):
        """``"matrix"``, ``"vector"``, ``"scalar"`` or ``None`` for a value."""
        if isinstance(expr, BinaryOpNode):
            op = self.binary_operator_text(expr.op)
            left = self.raw_value_shape(expr.left)
            right = self.raw_value_shape(expr.right)
            if op in {"&&", "||"} and left == right == "scalar":
                return "scalar"
            shapes = {left, right}
            if None in shapes:
                return None
            for shape in ("matrix", "vector", "scalar"):
                if shape in shapes:
                    return shape
        if isinstance(expr, UnaryOpNode):
            return self.raw_value_shape(expr.operand)
        if isinstance(expr, TernaryOpNode):
            return self.raw_value_shape(expr.true_expr) or self.raw_value_shape(
                expr.false_expr
            )
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            name = self.normalize_hlsl_intrinsic_name(expr.name)
            if (
                name not in self.source_function_names
                and self.function_call_value_raw_type(expr) is None
            ):
                if name in self.SCALAR_RESULT_INTRINSICS:
                    return "scalar"
                if name == "mul" and len(expr.args) == 2:
                    shapes = {self.raw_value_shape(arg) for arg in expr.args}
                    if shapes == {"vector"}:
                        return "scalar"
                    if shapes == {"matrix"}:
                        return "matrix"
                    if "matrix" in shapes and "vector" in shapes:
                        return "vector"
                    return None
                if expr.args:
                    return self.raw_value_shape(expr.args[0])
        raw_type = self.expression_value_raw_type(expr)
        if raw_type is None:
            return None
        base = self.canonical_composite_type(str(raw_type).strip())
        scalar = (
            r"(?:min16float|min10float|min16uint|min16int|min12int|float16_t|"
            r"uint16_t|int16_t|double|float|half|fixed|uint|int|bool|dword)"
        )
        match = re.fullmatch(scalar + r"([1-4])x([1-4])", base)
        if match:
            rows, cols = (int(value) for value in match.groups())
            if rows > 1 and cols > 1:
                return "matrix"
            return "vector" if rows * cols > 1 else "scalar"
        if re.fullmatch(scalar + r"[2-4]", base):
            return "vector"
        if re.fullmatch(scalar + r"1?", base):
            return "scalar"
        if self.raw_type_base(base) in {"vector", "matrix"}:
            return self.raw_type_base(base)
        return None

    def generate_matrix_swizzle(self, expr, is_main=False):
        """Render ``m._m01`` / ``m._12_21`` as indexed matrix elements.

        HLSL ``_mRC`` is zero-based and ``_RC`` one-based. CrossGL indexes
        columns first, so element (R, C) is ``m[C][R]``; with row-major
        packing CrossGL holds HLSL row ``R`` as column ``R`` and the element
        is ``m[R][C]``.
        """
        member = expr.member
        if not isinstance(member, str) or not member.startswith("_"):
            return None
        zero_based = re.fullmatch(r"(?:_m[0-3][0-3])+", member)
        one_based = re.fullmatch(r"(?:_[1-4][1-4])+", member)
        if not zero_based and not one_based:
            return None
        if zero_based:
            elements = [
                (int(row), int(col)) for row, col in re.findall(r"_m(\d)(\d)", member)
            ]
        else:
            elements = [
                (int(row) - 1, int(col) - 1)
                for row, col in re.findall(r"_(\d)(\d)", member)
            ]
        if len(elements) > 4:
            return None
        vector_swizzle = self.generate_vector_matrix_swizzle(expr, elements, is_main)
        if vector_swizzle is not None:
            return vector_swizzle
        if self.raw_value_shape(expr.object) != "matrix":
            return None
        obj = self.generate_expression(expr.object, is_main)
        obj = self.maybe_parenthesize(expr.object, obj)
        if isinstance(expr.object, (UnaryOpNode, CastNode)):
            obj = f"({obj})"
        if self.hlsl_matrix_rows_are_columns():
            accesses = [f"{obj}[{row}][{col}]" for row, col in elements]
        else:
            accesses = [f"{obj}[{col}][{row}]" for row, col in elements]
        if len(accesses) == 1:
            return accesses[0]
        matrix_type = self.canonical_composite_type(
            str(self.expression_value_raw_type(expr.object) or "")
        )
        scalar = re.match(r"[a-z_0-9]*?(?=\d+x\d+$)", matrix_type)
        vector_type = self.map_type(
            f"{scalar.group(0) if scalar else 'float'}{len(accesses)}"
        )
        return f"{vector_type}({', '.join(accesses)})"

    def generate_vector_matrix_swizzle(self, expr, elements, is_main=False):
        """``_mRC`` on a ``1xN`` or ``Nx1`` matrix, which CrossGL holds as a vector."""
        raw_type = self.canonical_composite_type(
            str(self.expression_value_raw_type(expr.object) or "")
        )
        shape = re.fullmatch(r"[a-z_0-9]*?([1-4])x([1-4])", raw_type)
        if shape is None:
            return None
        rows, columns = int(shape.group(1)), int(shape.group(2))
        if 1 not in (rows, columns) or rows == columns:
            return None
        indices = []
        for row, column in elements:
            if row >= rows or column >= columns:
                return None
            indices.append(column if rows == 1 else row)
        obj = self.generate_expression(expr.object, is_main)
        obj = self.maybe_parenthesize(expr.object, obj)
        if isinstance(expr.object, (UnaryOpNode, CastNode)):
            obj = f"({obj})"
        return f"{obj}.{''.join('xyzw'[index] for index in indices)}"

    def generate_mul_expression(self, args, rendered_args):
        """Lower HLSL ``mul``.

        Two vectors give their dot product. With row-major packing CrossGL
        holds HLSL rows as columns, which turns ``mul(a, b)`` into ``b * a``.
        """
        left_shape = self.raw_value_shape(args[0])
        right_shape = self.raw_value_shape(args[1])
        if left_shape == "vector" and right_shape == "vector":
            precise_dot = self.precise_intrinsic_call("dot", args, rendered_args)
            if precise_dot is not None:
                return precise_dot
            return f"dot({rendered_args[0]}, {rendered_args[1]})"
        left = self.maybe_parenthesize(args[0], rendered_args[0])
        right = self.maybe_parenthesize(args[1], rendered_args[1])
        if self.matrix_packing == "row_major" and "matrix" in (left_shape, right_shape):
            precise_product = self.precise_matrix_product(args, rendered_args, 1, 0)
            if precise_product is not None:
                return precise_product
            return f"({right} * {left})"
        if self.matrix_packing == "row_major" and None in (left_shape, right_shape):
            raise ValueError(
                "Cannot order HLSL mul operands with row-major matrix packing: "
                f"unknown operand type in mul({rendered_args[0]}, {rendered_args[1]})"
            )
        if "matrix" in (left_shape, right_shape):
            precise_product = self.precise_matrix_product(args, rendered_args, 0, 1)
            if precise_product is not None:
                return precise_product
        return f"({left} * {right})"

    def map_type_preserving_matrix_shape(self, hlsl_type):
        """Map an HLSL type name to CrossGL keeping matrix rows and columns."""
        if not hlsl_type:
            return hlsl_type
        type_name = self.canonical_composite_type(str(hlsl_type))
        wide_vector = re.fullmatch(r"(u?int)64_t([234])", type_name)
        if wide_vector is not None:
            prefix = "u64vec" if wide_vector.group(1) == "uint" else "i64vec"
            return prefix + wide_vector.group(2)
        matrix_alias_type = self.map_hlsl_matrix_alias_type(type_name)
        if matrix_alias_type:
            return matrix_alias_type
        if "<" in type_name and type_name.endswith(">"):
            base, generic_args = type_name.split("<", 1)
            base = base.strip()
            generic_type = generic_args[:-1].strip()
            vector_or_matrix_type = self.map_template_vector_or_matrix_type(
                base, generic_type
            )
            if vector_or_matrix_type:
                return vector_or_matrix_type
            rasterizer_buffer_type = self.map_rasterizer_ordered_buffer_type(
                base, generic_type
            )
            if rasterizer_buffer_type:
                return rasterizer_buffer_type
            feedback_texture_type = self.map_feedback_texture_type(base, generic_type)
            if feedback_texture_type:
                return feedback_texture_type
            if base in self.structured_buffer_types:
                mapped_args = ", ".join(
                    self.sanitize_type_name(self.canonical_composite_type(arg))
                    for arg in self.split_generic_arguments(generic_type)
                )
                return f"{self.sanitize_type_name(base)}<{mapped_args}>"
            storage_image_type = self.map_rw_texture_type(base, generic_type)
            if storage_image_type:
                return storage_image_type
            sampled_texture_type = self.map_sampled_texture_type(base, generic_type)
            if sampled_texture_type:
                return sampled_texture_type
            if base in self.generic_passthrough_types:
                mapped_args = ", ".join(
                    self.sanitize_type_name(self.canonical_composite_type(arg))
                    for arg in self.split_generic_arguments(generic_type)
                )
                return f"{self.sanitize_type_name(base)}<{mapped_args}>"
            type_name = base
        default_template_type = {
            "vector": "vec4",
            "matrix": "mat4",
        }.get(type_name)
        if default_template_type:
            return default_template_type
        return self.type_map.get(type_name, self.sanitize_type_name(type_name))
