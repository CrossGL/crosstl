"""Contraction-free HLSL arithmetic for CrossGL targets.

DXC lowers the HLSL intrinsics handled here to sequences of DXIL
multiplications and additions: ``dot`` sums the component products from the
first component on, matrix products accumulate their inner products in the
same order, ``lerp(x, y, s)`` is ``x + s * (y - x)`` and so on. A Direct3D
implementation that executes DXIL without contraction, such as WARP, rounds
every product and every sum. CrossGL targets implement the same intrinsics
with fused multiply-adds: Metal fuses inside ``dot``, ``mix``, ``length``,
``normalize``, ``smoothstep``, ``cross``, ``reflect`` and matrix products
whatever the contraction setting of the calling code.

With ``floating_point_contraction="off"`` the HLSL frontend marks every
function ``precise`` and replaces these intrinsics with precise CrossGL
helpers that evaluate the DXC lowering one operation at a time.
"""

import re

from .DirectxAst import (
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)

HELPER_PREFIX = "crosstl_hlsl"
COMPONENTS = "xyzw"
VECTOR_TYPE = re.compile(r"(f16|[ibud])?vec([2-4])")
MATRIX_TYPE = re.compile(r"(d)?mat([2-4])(?:x([2-4]))?")
VECTOR_FAMILIES = {
    None: "float",
    "d": "double",
    "f16": "half",
    "i": "int",
    "u": "uint",
    "b": "bool",
}
SCALAR_FAMILIES = {
    "float": "float",
    "double": "double",
    "float16": "half",
    "half": "half",
    "int": "int",
    "uint": "uint",
    "bool": "bool",
}
INTEGER_FAMILIES = frozenset({"int", "uint", "bool"})
# Intrinsics that also take integer operands.
INTEGER_INTRINSICS = frozenset({"dot"})
COMPARISON_OPERATORS = frozenset({"==", "!=", "<", "<=", ">", ">=", "&&", "||"})
# Intrinsics whose result is floating-point whatever their operands.
FLOAT_RESULT_INTRINSICS = frozenset(
    {
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "cosh",
        "cross",
        "degrees",
        "distance",
        "exp",
        "exp2",
        "floor",
        "fmod",
        "frac",
        "length",
        "lerp",
        "log",
        "log10",
        "log2",
        "normalize",
        "pow",
        "radians",
        "rcp",
        "reflect",
        "round",
        "rsqrt",
        "saturate",
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
# Intrinsics whose result has the family of their operands.
FAMILY_PRESERVING_INTRINSICS = frozenset({"abs", "clamp", "max", "min", "mad"})
# Intrinsics that only involve dot products of their own operands.
DOT_PRODUCT_USERS = frozenset(
    {"length", "distance", "normalize", "reflect", "faceforward"}
)


class HLSLPreciseArithmeticError(ValueError):
    """An intrinsic cannot be evaluated without contraction."""


class HLSLPreciseArithmeticMixin:
    """Expand HLSL arithmetic intrinsics for ``floating_point_contraction="off"``."""

    FLOATING_POINT_CONTRACTIONS = ("allowed", "off")
    # name -> number of arguments
    PRECISE_ARITHMETIC_INTRINSICS = {
        "cross": 2,
        "distance": 2,
        "dot": 2,
        "faceforward": 3,
        "length": 1,
        "lerp": 3,
        "normalize": 1,
        "reflect": 2,
        "smoothstep": 3,
    }

    def precise_arithmetic_enabled(self):
        return getattr(self, "floating_point_contraction", "allowed") == "off"

    def reset_precise_arithmetic(self):
        self.precise_arithmetic_uses = set()

    # Operand typing -----------------------------------------------------

    def precise_operand_type(self, expr):
        """``(family, width)`` of a scalar or vector operand.

        ``family`` is ``"float"``, ``"double"``, ``"half"``, ``"int"``,
        ``"uint"``, ``"bool"`` or ``None`` when unknown; ``width`` is the
        component count or ``None``.
        """
        return self.precise_operand_family(expr), self.hlsl_component_width(expr)

    def precise_operand_family(self, expr):
        raw_type = self.expression_value_raw_type(expr)
        mapped = None
        if raw_type:
            mapped = self.map_type(self.canonical_composite_type(str(raw_type)))
        if isinstance(mapped, str):
            vector = VECTOR_TYPE.fullmatch(mapped)
            if vector:
                return VECTOR_FAMILIES[vector.group(1)]
            if mapped in SCALAR_FAMILIES:
                return SCALAR_FAMILIES[mapped]
        if isinstance(expr, bool):
            return "bool"
        if isinstance(expr, int):
            return "int"
        if isinstance(expr, float):
            return "float"
        if isinstance(expr, str) and self.HLSL_NUMERIC_LITERAL.fullmatch(expr.strip()):
            text = expr.strip().lower()
            if text.startswith("0x"):
                return "int"
            return "float" if any(mark in text for mark in (".", "e", "f")) else "int"
        if isinstance(expr, UnaryOpNode):
            if str(getattr(expr, "op", "")) == "!":
                return "bool"
            return self.precise_operand_family(expr.operand)
        if isinstance(expr, BinaryOpNode):
            if self.binary_operator_text(expr.op) in COMPARISON_OPERATORS:
                return "bool"
            return self.combined_family([expr.left, expr.right])
        if isinstance(expr, TernaryOpNode):
            return self.combined_family([expr.true_expr, expr.false_expr])
        if (
            isinstance(expr, MemberAccessNode)
            and isinstance(expr.member, str)
            and re.fullmatch(r"[xyzw]{1,4}|[rgba]{1,4}", expr.member)
            and self.is_swizzle_object(expr.object)
        ):
            return self.precise_operand_family(expr.object)
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            callee = self.normalize_hlsl_intrinsic_name(expr.name)
            if callee in self.source_function_names:
                return None
            if callee in FLOAT_RESULT_INTRINSICS:
                families = [self.precise_operand_family(arg) for arg in expr.args]
                return "double" if "double" in families else "float"
            if callee in FAMILY_PRESERVING_INTRINSICS and expr.args:
                return self.combined_family(expr.args)
        return None

    def combined_family(self, operands):
        """Family of an arithmetic result after the usual conversions."""
        families = [self.precise_operand_family(operand) for operand in operands]
        for family in ("double", "float", "half"):
            if family in families:
                return family
        if None in families:
            return None
        for family in ("uint", "int", "bool"):
            if family in families:
                return family
        return None

    def precise_value_type(self, name, args, rendered_args, require_vector=False):
        """Common CrossGL value type of floating-point intrinsic operands.

        Returns ``None`` when every operand is an integer or boolean, whose
        arithmetic is exact and keeps the target intrinsic.
        """
        typed = [self.precise_operand_type(arg) for arg in args]
        families = [family for family, _ in typed]
        widths = [width for _, width in typed]
        call = f"{name}({', '.join(rendered_args)})"
        if "double" in families:
            family = "double"
        elif "float" in families:
            family = "float"
        elif "half" in families:
            family = "half"
        elif name in INTEGER_INTRINSICS and families:
            # Only dot products also take integer operands, whose arithmetic
            # is exact.
            if all(family in INTEGER_FAMILIES for family in families):
                return None
            if any(family in INTEGER_FAMILIES for family in families):
                raise HLSLPreciseArithmeticError(
                    f"Cannot evaluate {call} without contraction: an operand "
                    "type is unknown, so the product may be integer or "
                    "floating-point"
                )
            family = "float"
        else:
            # The other intrinsics convert integer operands to float.
            family = "float"
        if family == "half":
            raise HLSLPreciseArithmeticError(
                f"Cannot evaluate {call} without contraction: half-precision "
                "operands are not supported"
            )
        if None in widths:
            raise HLSLPreciseArithmeticError(
                f"Cannot evaluate {call} without contraction: an operand is not "
                "a known scalar or vector"
            )
        vector_widths = {width for width in widths if width > 1}
        if len(vector_widths) > 1:
            raise HLSLPreciseArithmeticError(
                f"Cannot evaluate {call} without contraction: the operands are "
                "vectors of different sizes"
            )
        width = max(widths)
        if require_vector and width == 1:
            return None
        if width == 1:
            return family
        return f"{'d' if family == 'double' else ''}vec{width}"

    def precise_argument(self, arg, rendered, value_type):
        """Render ``arg`` converted (or splatted) to ``value_type``."""
        family, width = self.precise_operand_type(arg)
        target_width = self.precise_type_width(value_type)
        target_family = "double" if value_type.startswith("d") else "float"
        if width == target_width and family in (target_family, None):
            return rendered
        return f"{value_type}({rendered})"

    @staticmethod
    def precise_type_width(value_type):
        vector = VECTOR_TYPE.fullmatch(value_type)
        return int(vector.group(2)) if vector else 1

    @staticmethod
    def precise_scalar_type(value_type):
        return "double" if value_type.startswith("d") else "float"

    # Intrinsic expansion ------------------------------------------------

    def precise_intrinsic_call(self, func_name, args, rendered_args):
        """Precise expansion of an HLSL intrinsic call, or ``None``."""
        if not self.precise_arithmetic_enabled():
            return None
        arity = self.PRECISE_ARITHMETIC_INTRINSICS.get(func_name)
        if arity is None or arity != len(args):
            return None
        if func_name in self.source_function_names:
            return None
        if self.constant_initializer_depth or all(
            self.is_hlsl_constant_expression(arg) for arg in args
        ):
            # DXC evaluates constant operands at compile time.
            return None
        if func_name == "cross":
            value_type = self.precise_value_type(func_name, args, rendered_args)
            if value_type is None:
                return None
            if self.precise_type_width(value_type) != 3:
                raise HLSLPreciseArithmeticError(
                    f"Cannot evaluate cross({', '.join(rendered_args)}) without "
                    "contraction: cross takes three-component vectors"
                )
        else:
            value_type = self.precise_value_type(
                func_name,
                args,
                rendered_args,
                require_vector=func_name in {"normalize", "reflect", "faceforward"},
            )
            if value_type is None:
                return None
        width = self.precise_type_width(value_type)
        operands = [
            self.precise_argument(arg, rendered, value_type)
            for arg, rendered in zip(args, rendered_args)
        ]
        if width == 1:
            scalar = self.precise_scalar_form(func_name, operands)
            if scalar is not None:
                return scalar
        self.precise_arithmetic_uses.add((func_name, value_type))
        return f"{HELPER_PREFIX}_{func_name}_{value_type}({', '.join(operands)})"

    @staticmethod
    def precise_scalar_form(func_name, operands):
        """Scalar forms that need a single operation or none."""
        if func_name == "dot":
            return f"({operands[0]} * {operands[1]})"
        if func_name == "length":
            return f"abs({operands[0]})"
        if func_name == "distance":
            return f"abs(({operands[0]} - {operands[1]}))"
        return None

    def precise_matrix_product(self, args, rendered_args, left_index, right_index):
        """Precise expansion of the CrossGL product ``args[left] * args[right]``.

        ``left_index`` and ``right_index`` select the CrossGL operands of the
        product, which row-major packing swaps relative to HLSL ``mul``.
        Returns ``None`` when the product needs no expansion.
        """
        if not self.precise_arithmetic_enabled():
            return None
        if self.constant_initializer_depth or all(
            self.is_hlsl_constant_expression(arg) for arg in args
        ):
            return None
        operand_types = []
        for index in (left_index, right_index):
            raw_type = self.expression_value_raw_type(args[index])
            mapped = None
            if raw_type:
                mapped = self.map_type(self.canonical_composite_type(str(raw_type)))
            operand_types.append(mapped if isinstance(mapped, str) else None)
        left_type, right_type = operand_types
        call = f"mul({', '.join(rendered_args)})"
        shapes = [self.precise_product_shape(value) for value in operand_types]
        if any(shape is None for shape in shapes):
            if any(
                isinstance(value, str) and value.startswith(("int", "uint", "bool"))
                for value in operand_types
            ):
                # Integer and boolean products are exact.
                return None
            raise HLSLPreciseArithmeticError(
                f"Cannot evaluate {call} without contraction: an operand is not a "
                "known floating-point matrix or vector"
            )
        left_kind, left_rows, left_cols, left_family = shapes[0]
        right_kind, right_rows, right_cols, right_family = shapes[1]
        if "scalar" in (left_kind, right_kind):
            return None
        if left_family != right_family:
            raise HLSLPreciseArithmeticError(
                f"Cannot evaluate {call} without contraction: the operands have "
                "different element types"
            )
        if left_kind == "matrix" and right_kind == "vector":
            inner = left_cols == right_rows
        elif left_kind == "vector" and right_kind == "matrix":
            inner = left_rows == right_rows
        elif left_kind == "matrix" and right_kind == "matrix":
            inner = left_cols == right_rows
        else:
            return None
        if not inner:
            raise HLSLPreciseArithmeticError(
                f"Cannot evaluate {call} without contraction: the operand "
                "dimensions do not match"
            )
        self.precise_arithmetic_uses.add(("mul", left_type, right_type))
        left = rendered_args[left_index]
        right = rendered_args[right_index]
        return f"{HELPER_PREFIX}_mul_{left_type}_{right_type}({left}, {right})"

    @staticmethod
    def precise_product_shape(value_type):
        """``(kind, rows, cols, family)`` of a CrossGL product operand.

        CrossGL ``matRxC`` has R rows and C columns; ``m[k]`` is column k. A
        vector is reported with its length as ``rows``.
        """
        if value_type is None:
            return None
        if value_type in ("float", "double"):
            return ("scalar", 1, 1, value_type)
        vector = VECTOR_TYPE.fullmatch(value_type)
        if vector:
            if vector.group(1) not in (None, "d"):
                return None
            family = VECTOR_FAMILIES[vector.group(1)]
            return ("vector", int(vector.group(2)), 1, family)
        matrix = MATRIX_TYPE.fullmatch(value_type)
        if matrix:
            rows = int(matrix.group(2))
            cols = int(matrix.group(3) or rows)
            family = "double" if matrix.group(1) else "float"
            return ("matrix", rows, cols, family)
        return None

    # Helper definitions -------------------------------------------------

    def generate_precise_arithmetic_helpers(self):
        uses = set(getattr(self, "precise_arithmetic_uses", ()) or ())
        if not uses:
            return ""
        for use in list(uses):
            if use[0] in DOT_PRODUCT_USERS:
                uses.add(("dot", use[1]))
        order = {"dot": 0}
        code = ""
        for use in sorted(uses, key=lambda item: (order.get(item[0], 1), item)):
            if use[0] == "mul":
                code += self.precise_matrix_product_helper(use[1], use[2])
            else:
                code += self.precise_intrinsic_helper(use[0], use[1])
        return code

    @staticmethod
    def precise_function(return_type, name, params, body_lines):
        body = "".join(f"        {line}\n" for line in body_lines)
        return (
            "    @ precise\n"
            f"    {return_type} {HELPER_PREFIX}_{name}({params}) {{\n"
            f"{body}"
            "    }\n\n"
        )

    @staticmethod
    def precise_sum(terms):
        total = terms[0]
        for term in terms[1:]:
            total = f"({total} + {term})"
        return total

    def precise_intrinsic_helper(self, name, value_type):
        scalar = self.precise_scalar_type(value_type)
        width = self.precise_type_width(value_type)
        components = COMPONENTS[:width]
        suffix = f"{name}_{value_type}"
        rsqrt = self.function_map.get("rsqrt", "inverseSqrt")
        dot = f"{HELPER_PREFIX}_dot_{value_type}"
        T = value_type
        if name == "dot":
            terms = [f"(a.{c} * b.{c})" for c in components]
            return self.precise_function(
                scalar,
                suffix,
                f"{T} a, {T} b",
                [f"return {self.precise_sum(terms)};"],
            )
        if name == "length":
            return self.precise_function(
                scalar, suffix, f"{T} v", [f"return sqrt({dot}(v, v));"]
            )
        if name == "distance":
            return self.precise_function(
                scalar,
                suffix,
                f"{T} a, {T} b",
                [
                    f"precise {T} difference = (a - b);",
                    f"return sqrt({dot}(difference, difference));",
                ],
            )
        if name == "normalize":
            return self.precise_function(
                T, suffix, f"{T} v", [f"return (v * {rsqrt}({dot}(v, v)));"]
            )
        if name == "lerp":
            return self.precise_function(
                T, suffix, f"{T} a, {T} b, {T} t", ["return (a + (t * (b - a)));"]
            )
        if name == "smoothstep":
            return self.precise_function(
                T,
                suffix,
                f"{T} low, {T} high, {T} x",
                [
                    f"precise {T} t = clamp(((x - low) / (high - low)), 0.0, 1.0);",
                    "return ((t * t) * (3.0 - (2.0 * t)));",
                ],
            )
        if name == "cross":
            return self.precise_function(
                T,
                suffix,
                f"{T} a, {T} b",
                [
                    f"return {T}(((a.y * b.z) - (a.z * b.y)), "
                    "((a.z * b.x) - (a.x * b.z)), ((a.x * b.y) - (a.y * b.x)));"
                ],
            )
        if name == "reflect":
            return self.precise_function(
                T,
                suffix,
                f"{T} i, {T} n",
                [f"return (i - (n * (2.0 * {dot}(i, n))));"],
            )
        if name == "faceforward":
            return self.precise_function(
                T,
                suffix,
                f"{T} n, {T} i, {T} ng",
                [f"return (({dot}(i, ng) < 0.0) ? n : (-n));"],
            )
        raise AssertionError(f"unknown precise intrinsic {name}")

    def precise_matrix_product_helper(self, left_type, right_type):
        left_kind, left_rows, left_cols, family = self.precise_product_shape(left_type)
        right_kind, right_rows, right_cols, _ = self.precise_product_shape(right_type)
        vector_prefix = "dvec" if family == "double" else "vec"
        name = f"mul_{left_type}_{right_type}"
        params = f"{left_type} left, {right_type} right"

        def vector(width, items):
            return f"{vector_prefix}{width}({', '.join(items)})"

        if left_kind == "matrix" and right_kind == "vector":
            # (M * v)[i] = sum_k M[k][i] * v[k]
            items = [
                self.precise_sum(
                    [
                        f"(left[{k}][{i}] * right.{COMPONENTS[k]})"
                        for k in range(left_cols)
                    ]
                )
                for i in range(left_rows)
            ]
            return self.precise_function(
                f"{vector_prefix}{left_rows}",
                name,
                params,
                [f"return {vector(left_rows, items)};"],
            )
        if left_kind == "vector" and right_kind == "matrix":
            # (v * M)[j] = sum_k v[k] * M[j][k]
            items = [
                self.precise_sum(
                    [
                        f"(left.{COMPONENTS[k]} * right[{j}][{k}])"
                        for k in range(left_rows)
                    ]
                )
                for j in range(right_cols)
            ]
            return self.precise_function(
                f"{vector_prefix}{right_cols}",
                name,
                params,
                [f"return {vector(right_cols, items)};"],
            )
        # (A * B)[j][i] = sum_k A[k][i] * B[j][k]
        columns = []
        for j in range(right_cols):
            items = [
                self.precise_sum(
                    [f"(left[{k}][{i}] * right[{j}][{k}])" for k in range(left_cols)]
                )
                for i in range(left_rows)
            ]
            columns.append(vector(left_rows, items))
        prefix = "dmat" if family == "double" else "mat"
        if left_rows == right_cols:
            result_type = f"{prefix}{left_rows}"
        else:
            result_type = f"{prefix}{left_rows}x{right_cols}"
        return self.precise_function(
            result_type,
            name,
            params,
            [f"return {result_type}({', '.join(columns)});"],
        )
