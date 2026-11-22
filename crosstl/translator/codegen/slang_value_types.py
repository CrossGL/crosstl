"""Value shapes in the Slang generator: scalar, vector and matrix types, matrix and select helpers, and operator result types."""

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .image_access_contracts import numeric_expression_component_count
from .operator_spellings import c_family_operator_spelling


class SlangValueTypesMixin:
    """Value-type members of ``SlangCodeGen``."""

    def type_name_string(self, type_name):
        if type_name is None:
            return None
        if not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def is_scalar_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return self.convert_type(type_name) in {
            "half",
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }

    def is_vector_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return self.convert_type(type_name) in {
            "half2",
            "half3",
            "half4",
            "float2",
            "float3",
            "float4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def value_component_count(self, vtype):
        info = self.vector_value_info(vtype)
        if info is not None:
            return info["size"]
        if self.is_matrix_value_type(vtype):
            dims = self.hlsl_matrix_dimensions(vtype)
            if dims is not None:
                return dims[1] * dims[2]
        mapped = self.type_name_string(self.map_type(vtype))
        if mapped in {"float", "double", "half", "int", "uint", "bool"}:
            return 1
        return None

    def expression_component_count(self, expr):
        return numeric_expression_component_count(
            expr,
            self.expression_result_type,
            self.type_name_string,
            self.map_type,
            self.vector_component_type,
            scalar_types={
                "float",
                "double",
                "half",
                "int",
                "uint",
                "bool",
            },
            excluded_type_markers=("x",),
        )

    def hlsl_expression_is_repeatable(self, expr):
        if expr is None:
            return False
        if isinstance(expr, (int, float, str)):
            return True
        if isinstance(expr, LiteralNode):
            return True
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return True
        if isinstance(expr, MemberAccessNode):
            return self.hlsl_expression_is_repeatable(
                getattr(expr, "object", getattr(expr, "object_expr", None))
            )
        if isinstance(expr, ArrayAccessNode):
            index = getattr(expr, "index", None)
            indices = getattr(expr, "indices", None)
            if indices is None and index is not None:
                indices = [index]
            return self.hlsl_expression_is_repeatable(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            ) and all(
                self.hlsl_expression_is_repeatable(item) for item in indices or []
            )
        # Operators, conditionals and scalar conversions of repeatable operands write nothing, as in the HLSL
        # emitter; an increment or decrement does.
        if isinstance(expr, UnaryOpNode):
            if c_family_operator_spelling(getattr(expr, "op", None)) in {"++", "--"}:
                return False
            return self.hlsl_expression_is_repeatable(getattr(expr, "operand", None))
        if isinstance(expr, BinaryOpNode):
            spelling = c_family_operator_spelling(getattr(expr, "op", None))
            if spelling is None or (
                spelling.endswith("=") and spelling not in {"==", "!=", "<=", ">="}
            ):
                return False
            return self.hlsl_expression_is_repeatable(
                getattr(expr, "left", None)
            ) and self.hlsl_expression_is_repeatable(getattr(expr, "right", None))
        if isinstance(expr, TernaryOpNode):
            return all(
                self.hlsl_expression_is_repeatable(getattr(expr, name, None))
                for name in ("condition", "true_expr", "false_expr")
            )
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", getattr(expr, "name", None))
            func_name = getattr(func_expr, "name", func_expr)
            args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            return self.is_scalar_value_type(func_name) and all(
                self.hlsl_expression_is_repeatable(arg) for arg in args
            )
        return False

    def hlsl_matrix_shape(self, vtype):
        mapped_type = self.map_type(vtype)
        if not isinstance(mapped_type, str):
            return None
        for component_type in ("double", "float", "half", "uint", "int", "bool"):
            if not mapped_type.startswith(component_type):
                continue
            suffix = mapped_type[len(component_type) :]
            if (
                len(suffix) == 3
                and suffix[0] in {"2", "3", "4"}
                and suffix[1] == "x"
                and suffix[2] in {"2", "3", "4"}
            ):
                return component_type, suffix[0], suffix[2]
        return None

    def hlsl_matrix_dimensions(self, vtype):
        matrix_shape = self.hlsl_matrix_shape(vtype)
        if matrix_shape is None:
            return None
        component_type, rows, columns = matrix_shape
        return component_type, int(rows), int(columns)

    def is_matrix_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        type_name = self.convert_type(type_name)
        if not isinstance(type_name, str):
            return False
        for prefix in ("float", "double"):
            if not type_name.startswith(prefix):
                continue
            suffix = type_name[len(prefix) :]
            return (
                len(suffix) == 3
                and suffix[0] in "234"
                and suffix[1] == "x"
                and suffix[2] in "234"
            )
        return False

    def matrix_value_info(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        mapped_type = self.convert_type(type_name)
        if not isinstance(mapped_type, str):
            return None
        for component_type in ("double", "float"):
            if not mapped_type.startswith(component_type):
                continue
            suffix = mapped_type[len(component_type) :]
            if (
                len(suffix) == 3
                and suffix[0] in "234"
                and suffix[1] == "x"
                and suffix[2] in "234"
            ):
                return {
                    "type": mapped_type,
                    "component_type": component_type,
                    "rows": int(suffix[0]),
                    "columns": int(suffix[2]),
                }
        return None

    def binary_requires_slang_mul(self, left_type, right_type):
        left_matrix = self.matrix_value_info(left_type)
        right_matrix = self.matrix_value_info(right_type)
        if left_matrix is not None and right_matrix is not None:
            return True
        if left_matrix is not None and self.vector_value_info(right_type) is not None:
            return True
        if right_matrix is not None and self.vector_value_info(left_type) is not None:
            return True
        return False

    def vector_component_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if not isinstance(mapped_type, str):
            return None
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("half"):
            return "half"
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def vector_value_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.convert_type(type_name)
        for component_type in ("double", "half", "float", "uint", "int", "bool"):
            if not mapped_type.startswith(component_type):
                continue
            suffix = mapped_type[len(component_type) :]
            if suffix in {"2", "3", "4"}:
                size = int(suffix)
                return {
                    "type": mapped_type,
                    "component_type": component_type,
                    "size": size,
                    "components": ("x", "y", "z", "w")[:size],
                }
        return None

    def generate_bool_vector_select_expression(
        self, condition_expr, true_expr, false_expr
    ):
        condition_info = self.vector_value_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        true_info = self.vector_value_info(self.expression_result_type(true_expr))
        false_info = self.vector_value_info(self.expression_result_type(false_expr))
        if (
            true_info is None
            or false_info is None
            or true_info["type"] != false_info["type"]
            or condition_info["size"] != true_info["size"]
        ):
            return None

        helper_name = self.require_vector_select_helper(true_info, condition_info)
        condition = self.generate_expression(condition_expr)
        true_value = self.generate_expression(true_expr)
        false_value = self.generate_expression(false_expr)
        return f"{helper_name}({condition}, {true_value}, {false_value})"

    def generate_slang_matrix_inverse_call(self, arg):
        arg_type = self.expression_result_type(arg)
        matrix_info = self.matrix_value_info(arg_type)
        arg_expr = self.generate_expression(arg)
        if matrix_info is None:
            fallback_type = self.current_expression_expected_type or arg_type or "float"
            return self.refuse_lowering(
                "Slang inverse: requires a matrix argument",
                self.zero_value_for_type(fallback_type),
            )
        if matrix_info["rows"] != matrix_info["columns"]:
            return self.refuse_lowering(
                "Slang inverse: requires a square matrix",
                self.zero_value_for_type(matrix_info["type"]),
            )
        helper_name = self.require_slang_matrix_inverse_helper(matrix_info)
        return f"{helper_name}({arg_expr})"

    def generate_slang_matrix_builtin_call(self, func_name, args):
        """``matrixCompMult`` or ``outerProduct``, which Slang does not declare (GLSL 4.60 section 8.6)."""
        fallback_type = self.current_expression_expected_type or "float"
        if func_name == "outerProduct":
            # slangc rejects the DirectX form mul(float3x1(c), float1x2(r)), so a helper builds the rows.
            result_type = self.hlsl_matrix_orientation.call_type(
                func_name, [self.expression_result_type(arg) for arg in args]
            )
            matrix_info = self.matrix_value_info(result_type)
            if matrix_info is None:
                return self.refuse_lowering(
                    "Slang outerProduct: requires two float or double vectors",
                    self.zero_value_for_type(fallback_type),
                )
            helper_name = self.require_slang_outer_product_helper(matrix_info)
            column, row = (self.generate_expression_without_expected(a) for a in args)
            return f"{helper_name}({column}, {row})"
        matrix_builtin = self.hlsl_matrix_orientation.builtin_call(func_name, args)
        if matrix_builtin is not None:
            return matrix_builtin
        return self.refuse_lowering(
            "Slang matrixCompMult: requires a matrix of a known shape",
            self.zero_value_for_type(fallback_type),
        )

    def require_slang_outer_product_helper(self, matrix_info):
        matrix_type = matrix_info["type"]
        helper_name = self.helper_function_name(f"_crossgl_outer_product_{matrix_type}")
        if helper_name in self.helper_functions:
            return helper_name

        component_type = matrix_info["component_type"]
        # Row i of the result is c[i] * r, so CrossGL column j is c * r[j].
        rows = ", ".join(f"c[{row}] * r" for row in range(matrix_info["rows"]))
        helper = (
            f"{matrix_type} {helper_name}("
            f"{component_type}{matrix_info['rows']} c, "
            f"{component_type}{matrix_info['columns']} r)\n"
            "{\n"
            f"    return {matrix_type}({rows});\n"
            "}"
        )
        self.register_helper_function(helper_name, helper)
        return helper_name

    def require_slang_matrix_inverse_helper(self, matrix_info):
        matrix_type = matrix_info["type"]
        helper_name = self.helper_function_name(f"_crossgl_inverse_{matrix_type}")
        if helper_name in self.helper_functions:
            return helper_name

        size = matrix_info["rows"]
        component_type = matrix_info["component_type"]
        vector_type = f"{component_type}{size}"
        helper = (
            f"{matrix_type} {helper_name}({matrix_type} m)\n"
            "{\n"
            f"    {matrix_type} a = m;\n"
            f"    {matrix_type} inv = {matrix_type}(1.0);\n"
            f"    for (int i = 0; i < {size}; i++)\n"
            "    {\n"
            "        int pivotRow = i;\n"
            f"        {component_type} pivotAbs = abs(a[i][i]);\n"
            f"        for (int row = i + 1; row < {size}; row++)\n"
            "        {\n"
            f"            {component_type} candidateAbs = abs(a[row][i]);\n"
            "            if (candidateAbs > pivotAbs)\n"
            "            {\n"
            "                pivotAbs = candidateAbs;\n"
            "                pivotRow = row;\n"
            "            }\n"
            "        }\n"
            "        if (pivotRow != i)\n"
            "        {\n"
            f"            {vector_type} tmpA = a[i];\n"
            "            a[i] = a[pivotRow];\n"
            "            a[pivotRow] = tmpA;\n"
            f"            {vector_type} tmpInv = inv[i];\n"
            "            inv[i] = inv[pivotRow];\n"
            "            inv[pivotRow] = tmpInv;\n"
            "        }\n"
            f"        {component_type} invPivot = 1.0 / a[i][i];\n"
            "        a[i] = a[i] * invPivot;\n"
            "        inv[i] = inv[i] * invPivot;\n"
            f"        for (int row = 0; row < {size}; row++)\n"
            "        {\n"
            "            if (row != i)\n"
            "            {\n"
            f"                {component_type} factor = a[row][i];\n"
            "                a[row] = a[row] - factor * a[i];\n"
            "                inv[row] = inv[row] - factor * inv[i];\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return inv;\n"
            "}"
        )
        self.register_helper_function(helper_name, helper)
        return helper_name

    def require_vector_select_helper(self, result_info, condition_info):
        result_type = result_info["type"]
        condition_type = condition_info["type"]
        base_name = f"_crossgl_select_{condition_type}_{result_type}"
        helper_name = self.helper_function_name(base_name)
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            f"(mask.{component} ? trueValue.{component} : falseValue.{component})"
            for component in result_info["components"]
        ]
        helper = (
            f"{result_type} {helper_name}("
            f"{condition_type} mask, {result_type} trueValue, "
            f"{result_type} falseValue)\n"
            "{\n"
            f"    return {result_type}({', '.join(components)});\n"
            "}"
        )
        self.register_helper_function(helper_name, helper)
        return helper_name

    def generate_slang_mod_expression(self, left, right):
        return f"(({left}) - (floor(({left}) / ({right})) * ({right})))"

    def binary_expression_result_type(self, expr):
        left_type, right_type = self.slang_promoted_operand_types(
            expr,
            self.expression_result_type(expr.left),
            self.expression_result_type(expr.right),
        )
        if expr.op in {"==", "!=", "<", ">", "<=", ">="}:
            return self.comparison_expression_result_type(left_type, right_type)
        if expr.op in {"&&", "||"}:
            return self.logical_expression_result_type(left_type, right_type)
        if expr.op == "*":
            product_type = self.hlsl_matrix_orientation.product_type(
                left_type, right_type
            )
            if product_type is not None:
                return product_type
        matrix_type = self.hlsl_matrix_orientation.arithmetic_type(
            expr.op, left_type, right_type
        )
        if matrix_type is not None:
            return matrix_type
        if self.is_vector_value_type(left_type):
            return left_type
        if self.is_vector_value_type(right_type):
            return right_type
        if left_type == "float" or right_type == "float":
            return "float"
        return left_type or right_type

    def comparison_expression_result_type(self, left_type, right_type):
        left_info = self.vector_value_info(left_type)
        right_info = self.vector_value_info(right_type)
        if left_info is not None and right_info is not None:
            if left_info["size"] == right_info["size"]:
                return f"bool{left_info['size']}"
            return None
        if left_info is not None and self.is_scalar_value_type(right_type):
            return f"bool{left_info['size']}"
        if right_info is not None and self.is_scalar_value_type(left_type):
            return f"bool{right_info['size']}"
        if left_type is not None or right_type is not None:
            return "bool"
        return None

    def logical_expression_result_type(self, left_type, right_type):
        left_info = self.vector_value_info(left_type)
        right_info = self.vector_value_info(right_type)
        if (
            left_info is not None
            and left_info["component_type"] == "bool"
            and right_type == "bool"
        ):
            return f"bool{left_info['size']}"
        if (
            right_info is not None
            and right_info["component_type"] == "bool"
            and left_type == "bool"
        ):
            return f"bool{right_info['size']}"
        if (
            left_info is not None
            and right_info is not None
            and left_info["component_type"] == "bool"
            and right_info["component_type"] == "bool"
            and left_info["size"] == right_info["size"]
        ):
            return f"bool{left_info['size']}"
        if left_type == "bool" and right_type == "bool":
            return "bool"
        return None
