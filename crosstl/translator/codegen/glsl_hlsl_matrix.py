"""OpenGL emulation of the integer and boolean matrices GLSL has no type for.

GLSL only provides floating-point matrices; HLSL also has ``int``, ``uint`` and
``bool`` matrices, which CrossGL keeps under their HLSL names (``int3x2`` has
three rows and two columns, the integer counterpart of CrossGL ``mat2x3``). Each one becomes a
struct of column vectors named like the GLSL matrix it stands in for: CrossGL
``int3x2`` is ``crossgl_imat2x3 { ivec3 c[2]; }`` just as ``mat3x2`` is GLSL
``mat2x3``. Indexing selects a column (``m[i]`` is ``m.c[i]``), constructors
fill whole columns, ``*`` with a vector or matrix is the linear-algebra product
and every other operator, comparison and supported intrinsic is element-wise.
GLSL has no operator overloading, so operations become generated helper
functions; constructors from scalars stay constant expressions so emulated
matrices can initialize ``const`` globals.
"""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)

EMULATED_MATRIX_PATTERN = re.compile(
    r"(?P<scalar>int|uint|bool)(?P<rows>[2-4])x(?P<columns>[2-4])"
)
NATIVE_MATRIX_PATTERN = re.compile(r"(?P<prefix>d?)mat(?P<a>[2-4])(?:x(?P<b>[2-4]))?")
GLSL_SCALAR_PREFIX = {"int": "i", "uint": "u", "bool": "b", "float": "", "double": "d"}
SCALAR_RANK = {"bool": 0, "int": 1, "uint": 2, "float": 3, "double": 4}
TYPE_ATTRIBUTES = (
    "vtype",
    "var_type",
    "return_type",
    "param_type",
    "member_type",
    "const_type",
    "element_type",
)

ELEMENTWISE_OPERATORS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
    "&": "and",
    "|": "or",
    "^": "xor",
    "<<": "shl",
    ">>": "shr",
}
COMPARISON_FUNCTIONS = {
    "==": ("eq", "equal"),
    "!=": ("ne", "notEqual"),
    "<": ("lt", "lessThan"),
    "<=": ("le", "lessThanEqual"),
    ">": ("gt", "greaterThan"),
    ">=": ("ge", "greaterThanEqual"),
}
COMPONENTWISE_BUILTINS = {"abs": 1, "sign": 1, "min": 2, "max": 2, "clamp": 3}


class OpenGLEmulatedMatrixError(ValueError):
    """An integer or boolean matrix operation OpenGL cannot reproduce."""

    def __init__(self, message, *, operation=None, operand_types=None, reason=None):
        super().__init__(message)
        self.operation = operation
        self.operand_types = tuple(operand_types or ())
        self.reason = reason


def emulated_matrix_shape(type_name):
    """``(scalar, rows, columns)`` of a CrossGL matrix GLSL has to emulate."""
    if not isinstance(type_name, str):
        return None
    match = EMULATED_MATRIX_PATTERN.fullmatch(type_name.strip())
    if match is None:
        return None
    return (
        match.group("scalar"),
        int(match.group("rows")),
        int(match.group("columns")),
    )


def emulated_matrix_struct_name(shape):
    scalar, rows, columns = shape
    return f"crossgl_{GLSL_SCALAR_PREFIX[scalar]}mat{columns}x{rows}"


def emulated_matrix_source_name(shape):
    scalar, rows, columns = shape
    return f"{scalar}{rows}x{columns}"


def vector_type_name(scalar, width):
    if width == 1:
        return scalar
    return f"{GLSL_SCALAR_PREFIX[scalar]}vec{width}"


class GLSLEmulatedMatrices:
    """GLSL code generator support for HLSL integer and boolean matrices."""

    def reset_glsl_emulated_matrices(self, ast=None):
        self.glsl_emulated_matrix_shapes = set()
        self.glsl_emulated_matrix_helpers = {}
        self.glsl_program_uses_emulated_matrices = (
            ast is not None and self.glsl_ast_uses_emulated_matrices(ast)
        )

    def glsl_ast_uses_emulated_matrices(self, ast):
        for node in self.walk_ast(ast):
            name = getattr(node, "name", None)
            if isinstance(name, str) and EMULATED_MATRIX_PATTERN.fullmatch(name):
                return True
            for attribute in TYPE_ATTRIBUTES:
                value = getattr(node, attribute, None)
                if isinstance(value, str) and EMULATED_MATRIX_PATTERN.search(value):
                    return True
        return False

    # Types -----------------------------------------------------------------

    def glsl_emulated_matrix_mapped_type(self, type_name):
        """The GLSL struct for an emulated CrossGL matrix type, or ``None``."""
        shape = emulated_matrix_shape(type_name)
        if shape is None:
            return None
        if not hasattr(self, "glsl_emulated_matrix_shapes"):
            self.reset_glsl_emulated_matrices()
        self.glsl_emulated_matrix_shapes.add(shape)
        return emulated_matrix_struct_name(shape)

    def glsl_emulated_shape_of_type(self, type_value):
        if type_value is None:
            return None
        type_name = (
            type_value
            if isinstance(type_value, str)
            else self.type_name_string(type_value)
        )
        return emulated_matrix_shape(type_name)

    def glsl_emulated_matrix_expression_type(self, expr):
        """CrossGL type of an expression that involves an emulated matrix."""
        if not getattr(self, "glsl_program_uses_emulated_matrices", False):
            return None
        if isinstance(expr, ArrayAccessNode):
            if not self.glsl_may_be_emulated_matrix(expr.array):
                return None
            shape = self.glsl_emulated_shape_of_type(
                self.glsl_source_expression_type(expr.array)
            )
            if shape is None:
                return None
            return vector_type_name(shape[0], shape[1])
        if not isinstance(expr, (BinaryOpNode, UnaryOpNode, FunctionCallNode)):
            return None
        if not self.glsl_may_be_emulated_matrix(expr):
            return None
        if isinstance(expr, BinaryOpNode):
            left = self.glsl_emulated_operand(expr.left)
            right = self.glsl_emulated_operand(expr.right)
            if "matrix" not in (left["kind"], right["kind"]):
                return None
            op = self.map_operator(expr.op)
            return self.glsl_emulated_binary_result_type(op, left, right)
        if isinstance(expr, UnaryOpNode):
            operand = self.glsl_emulated_operand(expr.operand)
            if operand["kind"] != "matrix":
                return None
            if self.map_operator(expr.op) == "!":
                return emulated_matrix_source_name(("bool",) + operand["shape"][1:])
            return operand["type"]
        name = self.function_call_name(expr)
        if emulated_matrix_shape(name) is not None:
            return name
        args = list(getattr(expr, "args", getattr(expr, "arguments", [])) or [])
        operands = [self.glsl_emulated_operand(arg) for arg in args]
        matrices = [operand for operand in operands if operand["kind"] == "matrix"]
        if not matrices or name in getattr(self, "function_return_types", {}):
            return None
        shape = matrices[0]["shape"]
        if name == "select" and len(operands) == 3:
            scalar = self.glsl_emulated_common_scalar(
                operands[:2], "select", comparison=True
            )
            return emulated_matrix_source_name((scalar,) + shape[1:])
        if name in {"and", "or"}:
            return emulated_matrix_source_name(("bool",) + shape[1:])
        if name in {"all", "any"}:
            return "bool"
        if name == "transpose":
            return emulated_matrix_source_name((shape[0], shape[2], shape[1]))
        if name in COMPONENTWISE_BUILTINS:
            return matrices[0]["type"]
        return None

    def glsl_may_be_emulated_matrix(self, expr):
        """Cheap test whether any value in ``expr`` can be an emulated matrix."""
        for node in self.walk_ast(expr):
            if isinstance(node, (VariableNode, IdentifierNode)):
                name = getattr(node, "name", None)
                if self.glsl_identifier_has_emulated_type(name):
                    return True
            elif isinstance(node, FunctionCallNode):
                name = self.function_call_name(node)
                if emulated_matrix_shape(name) is not None:
                    return True
                return_type = getattr(self, "function_return_types", {}).get(name)
                if isinstance(return_type, str) and EMULATED_MATRIX_PATTERN.search(
                    return_type
                ):
                    return True
            elif isinstance(node, MemberAccessNode):
                member = str(getattr(node, "member", ""))
                for members in getattr(self, "struct_member_types", {}).values():
                    member_type = self.type_name_string(members.get(member))
                    if member_type and EMULATED_MATRIX_PATTERN.search(member_type):
                        return True
        return False

    def glsl_identifier_has_emulated_type(self, name):
        if not name:
            return False
        for table in (
            getattr(self, "local_variable_source_types", {}),
            getattr(self, "local_variable_types", {}),
            getattr(self, "glsl_constant_types", {}),
            getattr(self, "global_variable_types", {}),
        ):
            type_name = self.type_name_string(table.get(name))
            if type_name and EMULATED_MATRIX_PATTERN.search(type_name):
                return True
        return False

    def glsl_emulated_operand(self, expr):
        """Classify an operand: emulated matrix, native matrix, scalar or vector."""
        source_type = self.glsl_source_expression_type(expr)
        type_name = self.type_name_string(source_type) if source_type else None
        shape = emulated_matrix_shape(type_name)
        if shape is not None:
            return {"kind": "matrix", "shape": shape, "type": type_name, "expr": expr}
        native = self.glsl_native_matrix_dimensions(expr, type_name)
        if native is not None:
            return {"kind": "native", "dims": native, "type": type_name, "expr": expr}
        info = self.glsl_value_type_info(type_name) if type_name else None
        if info is None and isinstance(expr, (bool, int, float)):
            info = self.glsl_value_type_info(
                "bool" if isinstance(expr, bool) else type(expr).__name__
            )
        if info is not None and info["family"] in SCALAR_RANK:
            kind = "scalar" if info["width"] == 1 else "vector"
            return {
                "kind": kind,
                "scalar": info["family"] if info["bits"] != 64 else "double",
                "width": info["width"],
                "glsl": info["mapped"],
                "type": type_name,
                "expr": expr,
            }
        return {"kind": "unknown", "type": type_name, "expr": expr}

    def glsl_native_matrix_dimensions(self, expr, type_name):
        """``(scalar, rows, columns)`` of a native floating-point matrix value.

        CrossGL and GLSL both spell a matrix of C columns and R rows ``matCxR``.
        """
        if not isinstance(type_name, str):
            return None
        match = NATIVE_MATRIX_PATTERN.fullmatch(type_name.strip())
        if match is None:
            return None
        scalar = "double" if match.group("prefix") else "float"
        columns = int(match.group("a"))
        rows = int(match.group("b") or columns)
        return (scalar, rows, columns)

    def glsl_emulated_common_scalar(self, operands, operation, *, comparison=False):
        scalars = []
        for operand in operands:
            if operand["kind"] == "matrix":
                scalars.append(operand["shape"][0])
            elif operand["kind"] == "scalar":
                scalars.append(operand["scalar"])
            else:
                self.glsl_emulated_matrix_error(operation, operands, "operand-shape")
        if comparison and set(scalars) == {"bool"}:
            return "bool"
        promoted = ["int" if scalar == "bool" else scalar for scalar in scalars]
        common = max(promoted, key=SCALAR_RANK.__getitem__)
        if common in {"float", "double"}:
            self.glsl_emulated_matrix_error(operation, operands, "floating-point-mix")
        return common

    def glsl_emulated_binary_result_type(self, op, left, right):
        operands = (left, right)
        if op == "*" and self.glsl_is_linear_algebra_product(left, right):
            plan = self.glsl_emulated_product_plan(left, right)
            return plan["source_type"]
        if op in COMPARISON_FUNCTIONS:
            shape = self.glsl_emulated_binary_shape(op, left, right)
            return emulated_matrix_source_name(("bool",) + shape[1:])
        if op in ELEMENTWISE_OPERATORS:
            shape = self.glsl_emulated_binary_shape(op, left, right)
            if op in {"<<", ">>"}:
                scalar = left["shape"][0] if left["kind"] == "matrix" else None
                if scalar is None:
                    self.glsl_emulated_matrix_error(op, operands, "shift-operand")
                scalar = "int" if scalar == "bool" else scalar
            else:
                scalar = self.glsl_emulated_common_scalar(operands, op)
            return emulated_matrix_source_name((scalar,) + shape[1:])
        self.glsl_emulated_matrix_error(op, operands, "unsupported-operator")

    def glsl_emulated_binary_shape(self, op, left, right):
        shapes = [o["shape"] for o in (left, right) if o["kind"] == "matrix"]
        for operand in (left, right):
            if operand["kind"] not in {"matrix", "scalar"}:
                self.glsl_emulated_matrix_error(op, (left, right), "operand-shape")
        if len({shape[1:] for shape in shapes}) != 1:
            self.glsl_emulated_matrix_error(op, (left, right), "dimension-mismatch")
        return shapes[0]

    def glsl_is_linear_algebra_product(self, left, right):
        kinds = {left["kind"], right["kind"]}
        return "matrix" in kinds and kinds <= {"matrix", "vector"}

    # Operations --------------------------------------------------------------

    def glsl_emulated_matrix_error(self, operation, operands, reason):
        described = tuple(
            operand.get("type") or operand["kind"] for operand in operands
        )
        raise OpenGLEmulatedMatrixError(
            "OpenGL cannot preserve integer or boolean matrix operation "
            f"'{operation}' for operand types {described}: {reason}",
            operation=operation,
            operand_types=described,
            reason=reason,
        )

    def glsl_emulated_helper(self, name, text):
        if not hasattr(self, "glsl_emulated_matrix_helpers"):
            self.reset_glsl_emulated_matrices()
        self.glsl_emulated_matrix_helpers.setdefault(name, text)
        return name

    def glsl_emulated_struct(self, shape):
        return self.glsl_emulated_matrix_mapped_type(emulated_matrix_source_name(shape))

    def glsl_emulated_tag(self, operand):
        if operand["kind"] == "matrix":
            return self.glsl_emulated_struct(operand["shape"])[len("crossgl_") :]
        return operand["glsl"]

    def glsl_emulated_parameter(self, operand, name):
        if operand["kind"] == "matrix":
            return f"{self.glsl_emulated_struct(operand['shape'])} {name}"
        return f"{operand['glsl']} {name}"

    def glsl_emulated_columns(self, shape, columns):
        """A constant-expression constructor from column expressions."""
        struct = self.glsl_emulated_struct(shape)
        column_type = vector_type_name(shape[0], shape[1])
        return f"{struct}({column_type}[{shape[2]}]({', '.join(columns)}))"

    def glsl_emulated_column(self, operand, name, index, scalar, rows):
        """Column ``index`` of an operand converted to ``scalar`` components."""
        column_type = vector_type_name(scalar, rows)
        if operand["kind"] == "matrix":
            column = f"{name}.c[{index}]"
            if operand["shape"][0] != scalar:
                column = f"{column_type}({column})"
            return column
        value = name if operand["scalar"] == scalar else f"{scalar}({name})"
        return value

    def generate_glsl_emulated_matrix_binary(self, expr, op):
        if not getattr(self, "glsl_program_uses_emulated_matrices", False):
            return None
        if not self.glsl_may_be_emulated_matrix(expr):
            return None
        left = self.glsl_emulated_operand(expr.left)
        right = self.glsl_emulated_operand(expr.right)
        if "matrix" not in (left["kind"], right["kind"]):
            return None
        if op == "*" and self.glsl_is_linear_algebra_product(left, right):
            return self.generate_glsl_emulated_product(left, right)
        result_type = self.glsl_emulated_binary_result_type(op, left, right)
        result = emulated_matrix_shape(result_type)
        if op in COMPARISON_FUNCTIONS:
            operand_scalar = self.glsl_emulated_common_scalar(
                (left, right), op, comparison=True
            )
            suffix, function = COMPARISON_FUNCTIONS[op]
        else:
            operand_scalar = result[0]
            suffix, function = ELEMENTWISE_OPERATORS[op], None
        name = (
            f"crossgl_{suffix}_{self.glsl_emulated_tag(left)}_"
            f"{self.glsl_emulated_tag(right)}"
        )
        rows, columns = result[1], result[2]
        column_type = vector_type_name(operand_scalar, rows)
        parts = []
        for index in range(columns):
            if op in {"<<", ">>"}:
                a = self.glsl_emulated_column(left, "a", index, operand_scalar, rows)
                shift_scalar = (
                    right["shape"][0] if right["kind"] == "matrix" else right["scalar"]
                )
                shift_scalar = "int" if shift_scalar == "bool" else shift_scalar
                b = self.glsl_emulated_column(right, "b", index, shift_scalar, rows)
                parts.append(f"{a} {op} {b}")
                continue
            a = self.glsl_emulated_column(left, "a", index, operand_scalar, rows)
            b = self.glsl_emulated_column(right, "b", index, operand_scalar, rows)
            if function is None:
                parts.append(f"{a} {op} {b}")
                continue
            if left["kind"] != "matrix":
                a = f"{column_type}({a})"
            if right["kind"] != "matrix":
                b = f"{column_type}({b})"
            parts.append(f"{function}({a}, {b})")
        struct = self.glsl_emulated_struct(result)
        body = self.glsl_emulated_columns(result, parts)
        self.glsl_emulated_helper(
            name,
            f"{struct} {name}({self.glsl_emulated_parameter(left, 'a')}, "
            f"{self.glsl_emulated_parameter(right, 'b')}) {{\n"
            f"    return {body};\n}}\n",
        )
        return (
            f"{name}({self.generate_expression(expr.left)}, "
            f"{self.generate_expression(expr.right)})"
        )

    def glsl_emulated_product_plan(self, left, right):
        operands = (left, right)
        scalar = self.glsl_emulated_common_scalar(
            [o if o["kind"] != "vector" else {**o, "kind": "scalar"} for o in operands],
            "*",
        )
        if left["kind"] == "matrix" and right["kind"] == "matrix":
            _s, rows, inner = left["shape"]
            _t, inner_right, columns = right["shape"]
            if inner != inner_right:
                self.glsl_emulated_matrix_error("*", operands, "dimension-mismatch")
            shape = (scalar, rows, columns)
            return {
                "source_type": emulated_matrix_source_name(shape),
                "shape": shape,
                "scalar": scalar,
            }
        if left["kind"] == "matrix":
            _s, rows, inner = left["shape"]
            if right["width"] != inner:
                self.glsl_emulated_matrix_error("*", operands, "dimension-mismatch")
            return {
                "source_type": vector_type_name(scalar, rows),
                "width": rows,
                "scalar": scalar,
            }
        _s, rows, columns = right["shape"]
        if left["width"] != rows:
            self.glsl_emulated_matrix_error("*", operands, "dimension-mismatch")
        return {
            "source_type": vector_type_name(scalar, columns),
            "width": columns,
            "scalar": scalar,
        }

    def generate_glsl_emulated_product(self, left, right):
        plan = self.glsl_emulated_product_plan(left, right)
        scalar = plan["scalar"]
        if scalar == "bool":
            self.glsl_emulated_matrix_error("*", (left, right), "boolean-product")
        name = (
            f"crossgl_matmul_{self.glsl_emulated_tag(left)}_"
            f"{self.glsl_emulated_tag(right)}"
        )

        def component(operand, var, column, row):
            if operand["kind"] == "matrix":
                value = f"{var}.c[{column}][{row}]"
                same = operand["shape"][0] == scalar
            else:
                value = f"{var}[{row}]"
                same = operand["scalar"] == scalar
            return value if same else f"{scalar}({value})"

        if left["kind"] == "matrix" and right["kind"] == "matrix":
            _s, rows, inner = left["shape"]
            columns = right["shape"][2]
            result_shape = (scalar, rows, columns)
            parts = []
            for j in range(columns):
                terms = [
                    f"{self.glsl_emulated_column(left, 'a', k, scalar, rows)} * "
                    f"{component(right, 'b', j, k)}"
                    for k in range(inner)
                ]
                parts.append(" + ".join(terms))
            result_type = self.glsl_emulated_struct(result_shape)
            body = self.glsl_emulated_columns(result_shape, parts)
        elif left["kind"] == "matrix":
            _s, rows, inner = left["shape"]
            result_type = vector_type_name(scalar, rows)
            terms = [
                f"{self.glsl_emulated_column(left, 'a', k, scalar, rows)} * "
                f"{component(right, 'b', None, k)}"
                for k in range(inner)
            ]
            body = " + ".join(terms)
        else:
            _s, rows, columns = right["shape"]
            result_type = vector_type_name(scalar, columns)
            values = [
                " + ".join(
                    f"{component(left, 'a', None, r)} * {component(right, 'b', j, r)}"
                    for r in range(rows)
                )
                for j in range(columns)
            ]
            body = f"{result_type}({', '.join(values)})"
        self.glsl_emulated_helper(
            name,
            f"{result_type} {name}({self.glsl_emulated_parameter(left, 'a')}, "
            f"{self.glsl_emulated_parameter(right, 'b')}) {{\n"
            f"    return {body};\n}}\n",
        )
        return (
            f"{name}({self.generate_expression(left['expr'])}, "
            f"{self.generate_expression(right['expr'])})"
        )

    def generate_glsl_emulated_matrix_unary(self, expr, op):
        if not getattr(self, "glsl_program_uses_emulated_matrices", False):
            return None
        if not self.glsl_may_be_emulated_matrix(expr.operand):
            return None
        operand = self.glsl_emulated_operand(expr.operand)
        if operand["kind"] != "matrix":
            return None
        if op == "+":
            return self.generate_expression(expr.operand)
        scalar, rows, columns = operand["shape"]
        if op == "!":
            result = ("bool", rows, columns)
            parts = [
                f"not({self.glsl_emulated_column(operand, 'a', i, 'bool', rows)})"
                for i in range(columns)
            ]
            suffix = "not"
        elif op in {"-", "~"} and scalar != "bool":
            result = operand["shape"]
            parts = [f"{op}a.c[{i}]" for i in range(columns)]
            suffix = "neg" if op == "-" else "inv"
        else:
            self.glsl_emulated_matrix_error(op, (operand,), "unsupported-operator")
        name = f"crossgl_{suffix}_{self.glsl_emulated_tag(operand)}"
        self.glsl_emulated_helper(
            name,
            f"{self.glsl_emulated_struct(result)} {name}"
            f"({self.glsl_emulated_parameter(operand, 'a')}) {{\n"
            f"    return {self.glsl_emulated_columns(result, parts)};\n}}\n",
        )
        return f"{name}({self.generate_expression(expr.operand)})"

    def generate_glsl_emulated_matrix_index(self, expr):
        if not getattr(self, "glsl_program_uses_emulated_matrices", False):
            return None
        if not self.glsl_may_be_emulated_matrix(expr.array):
            return None
        shape = self.glsl_emulated_shape_of_type(
            self.glsl_source_expression_type(expr.array)
        )
        if shape is None:
            return None
        array = self.generate_expression(expr.array)
        if not re.fullmatch(r"[A-Za-z_][\w.]*(?:\[[^\[\]]*\])*", array):
            array = f"({array})"
        return f"{array}.c[{self.generate_expression(expr.index)}]"

    def generate_glsl_emulated_matrix_call(self, expr, func_name, args):
        """Constructors, casts and intrinsics with emulated matrix operands."""
        if not getattr(self, "glsl_program_uses_emulated_matrices", False):
            return None
        shape = emulated_matrix_shape(func_name)
        if shape is not None:
            return self.generate_glsl_emulated_constructor(shape, args)
        if not any(self.glsl_may_be_emulated_matrix(arg) for arg in args):
            return None
        if func_name in getattr(self, "function_return_types", {}):
            return None
        operands = [self.glsl_emulated_operand(arg) for arg in args]
        matrices = [operand for operand in operands if operand["kind"] == "matrix"]
        if not matrices:
            return None
        native = NATIVE_MATRIX_PATTERN.fullmatch(func_name or "")
        if native is not None and len(operands) == 1:
            return self.generate_glsl_emulated_to_native(func_name, operands[0])
        if func_name == "select" and len(operands) == 3:
            return self.generate_glsl_emulated_select(operands)
        if func_name in {"and", "or"} and len(operands) == 2:
            return self.generate_glsl_emulated_logic(func_name, operands)
        if func_name in {"all", "any"} and len(operands) == 1:
            return self.generate_glsl_emulated_reduction(func_name, operands[0])
        if func_name == "transpose" and len(operands) == 1:
            return self.generate_glsl_emulated_transpose(operands[0])
        if COMPONENTWISE_BUILTINS.get(func_name) == len(operands):
            return self.generate_glsl_emulated_componentwise(func_name, operands)
        self.glsl_emulated_matrix_error(func_name, operands, "unsupported-intrinsic")

    def generate_glsl_emulated_constructor(self, shape, args):
        scalar, rows, columns = shape
        struct = self.glsl_emulated_struct(shape)
        column_type = vector_type_name(scalar, rows)
        operands = [self.glsl_emulated_operand(arg) for arg in args]
        if len(operands) == 1 and operands[0]["kind"] in {"matrix", "native"}:
            source = operands[0]
            source_shape = (
                source["shape"] if source["kind"] == "matrix" else source["dims"]
            )
            if source_shape[1:] != shape[1:]:
                self.glsl_emulated_matrix_error(
                    emulated_matrix_source_name(shape), operands, "dimension-mismatch"
                )
            if source["kind"] == "matrix" and source_shape[0] == scalar:
                return self.generate_expression(args[0])
            name = f"crossgl_cast_{struct[len('crossgl_'):]}_{self.glsl_emulated_cast_tag(source)}"
            access = "m.c[{i}]" if source["kind"] == "matrix" else "m[{i}]"
            parts = [f"{column_type}({access.format(i=i)})" for i in range(columns)]
            parameter = (
                self.glsl_emulated_parameter(source, "m")
                if source["kind"] == "matrix"
                else f"{self.glsl_native_matrix_name(source['dims'])} m"
            )
            self.glsl_emulated_helper(
                name,
                f"{struct} {name}({parameter}) {{\n"
                f"    return {self.glsl_emulated_columns(shape, parts)};\n}}\n",
            )
            return f"{name}({self.generate_expression(args[0])})"
        if len(operands) == 1 and operands[0]["kind"] == "scalar":
            name = f"crossgl_splat_{struct[len('crossgl_'):]}"
            parts = [f"{column_type}(s)"] * columns
            self.glsl_emulated_helper(
                name,
                f"{struct} {name}({scalar} s) {{\n"
                f"    return {self.glsl_emulated_columns(shape, parts)};\n}}\n",
            )
            value = self.generate_expression(args[0])
            if operands[0]["scalar"] != scalar:
                value = f"{scalar}({value})"
            return f"{name}({value})"
        widths = []
        for operand in operands:
            if operand["kind"] not in {"scalar", "vector"}:
                self.glsl_emulated_matrix_error(
                    emulated_matrix_source_name(shape), operands, "constructor-operand"
                )
            widths.append(operand["width"])
        if sum(widths) != rows * columns:
            self.glsl_emulated_matrix_error(
                emulated_matrix_source_name(shape), operands, "component-count"
            )
        rendered = [self.generate_expression(arg) for arg in args]
        boundaries = set(range(0, rows * columns + 1, rows))
        offsets = [sum(widths[:index]) for index in range(len(widths) + 1)]
        aligned = all(
            any(start < boundary < end for boundary in boundaries) is False
            for start, end in zip(offsets, offsets[1:])
        )
        if aligned:
            parts, current, filled = [], [], 0
            for value, width in zip(rendered, widths):
                current.append(value)
                filled += width
                if filled == rows:
                    parts.append(f"{column_type}({', '.join(current)})")
                    current, filled = [], 0
            return self.glsl_emulated_columns(shape, parts)
        # A vector operand spans two columns: split it inside a helper so every
        # operand is evaluated once.
        parameters = [
            f"{operand['glsl']} v{index}" for index, operand in enumerate(operands)
        ]
        components = []
        for index, operand in enumerate(operands):
            if operand["width"] == 1:
                components.append(f"v{index}")
            else:
                components.extend(
                    f"v{index}.{'xyzw'[lane]}" for lane in range(operand["width"])
                )
        parts = [
            f"{column_type}({', '.join(components[i * rows:(i + 1) * rows])})"
            for i in range(columns)
        ]
        signature = "_".join(operand["glsl"] for operand in operands)
        name = f"crossgl_make_{struct[len('crossgl_'):]}_{signature}"
        self.glsl_emulated_helper(
            name,
            f"{struct} {name}({', '.join(parameters)}) {{\n"
            f"    return {self.glsl_emulated_columns(shape, parts)};\n}}\n",
        )
        return f"{name}({', '.join(rendered)})"

    def glsl_native_matrix_name(self, dims):
        scalar, rows, columns = dims
        prefix = "d" if scalar == "double" else ""
        return f"{prefix}mat{columns}x{rows}"

    def glsl_emulated_cast_tag(self, operand):
        if operand["kind"] == "matrix":
            return self.glsl_emulated_tag(operand)
        return self.glsl_native_matrix_name(operand["dims"])

    def generate_glsl_emulated_to_native(self, func_name, operand):
        match = NATIVE_MATRIX_PATTERN.fullmatch(func_name)
        scalar = "double" if match.group("prefix") else "float"
        columns = int(match.group("a"))
        rows = int(match.group("b") or columns)
        if operand["kind"] != "matrix" or operand["shape"][1:] != (rows, columns):
            self.glsl_emulated_matrix_error(func_name, (operand,), "dimension-mismatch")
        native = self.glsl_native_matrix_name((scalar, rows, columns))
        column_type = vector_type_name(scalar, rows)
        name = f"crossgl_cast_{native}_{self.glsl_emulated_tag(operand)}"
        parts = [f"{column_type}(m.c[{i}])" for i in range(columns)]
        self.glsl_emulated_helper(
            name,
            f"{native} {name}({self.glsl_emulated_parameter(operand, 'm')}) {{\n"
            f"    return {native}({', '.join(parts)});\n}}\n",
        )
        return f"{name}({self.generate_expression(operand['expr'])})"

    def glsl_emulated_bool_column(self, operand, name, index, rows):
        column = f"{name}.c[{index}]"
        if operand["shape"][0] == "bool":
            return column
        return f"{vector_type_name('bool', rows)}({column})"

    def generate_glsl_emulated_select(self, operands):
        """CrossGL ``select(a, b, c)`` is ``c ? b : a`` for each element."""
        false_value, true_value, condition = operands
        if condition["kind"] != "matrix":
            self.glsl_emulated_matrix_error("select", operands, "condition-shape")
        shape = condition["shape"]
        for operand in (false_value, true_value):
            if operand["kind"] == "matrix" and operand["shape"][1:] != shape[1:]:
                self.glsl_emulated_matrix_error(
                    "select", operands, "dimension-mismatch"
                )
        if {false_value["kind"], true_value["kind"]} - {"matrix", "scalar"}:
            self.glsl_emulated_matrix_error("select", operands, "operand-shape")
        scalar = self.glsl_emulated_common_scalar(
            (false_value, true_value), "select", comparison=True
        )
        result = (scalar,) + shape[1:]
        rows, columns = shape[1], shape[2]
        column_type = vector_type_name(scalar, rows)
        parts = []
        for index in range(columns):
            values = []
            for operand, var in ((false_value, "a"), (true_value, "b")):
                value = self.glsl_emulated_column(operand, var, index, scalar, rows)
                if operand["kind"] != "matrix":
                    value = f"{column_type}({value})"
                values.append(value)
            mask = self.glsl_emulated_bool_column(condition, "c", index, rows)
            parts.append(f"mix({values[0]}, {values[1]}, {mask})")
        name = "crossgl_select_" + "_".join(
            self.glsl_emulated_tag(operand) for operand in operands
        )
        self.glsl_emulated_helper(
            name,
            f"{self.glsl_emulated_struct(result)} {name}("
            f"{self.glsl_emulated_parameter(false_value, 'a')}, "
            f"{self.glsl_emulated_parameter(true_value, 'b')}, "
            f"{self.glsl_emulated_parameter(condition, 'c')}) {{\n"
            f"    return {self.glsl_emulated_columns(result, parts)};\n}}\n",
        )
        rendered = ", ".join(
            self.generate_expression(operand["expr"]) for operand in operands
        )
        return f"{name}({rendered})"

    def generate_glsl_emulated_logic(self, func_name, operands):
        left, right = operands
        if left["kind"] != "matrix" or right["kind"] != "matrix":
            self.glsl_emulated_matrix_error(func_name, operands, "operand-shape")
        if left["shape"][1:] != right["shape"][1:]:
            self.glsl_emulated_matrix_error(func_name, operands, "dimension-mismatch")
        _scalar, rows, columns = left["shape"]
        result = ("bool", rows, columns)
        bool_type = vector_type_name("bool", rows)
        uint_type = vector_type_name("uint", rows)
        op = "&" if func_name == "and" else "|"
        parts = [
            f"{bool_type}({uint_type}({self.glsl_emulated_bool_column(left, 'a', i, rows)})"
            f" {op} {uint_type}({self.glsl_emulated_bool_column(right, 'b', i, rows)}))"
            for i in range(columns)
        ]
        name = (
            f"crossgl_{func_name}_{self.glsl_emulated_tag(left)}_"
            f"{self.glsl_emulated_tag(right)}"
        )
        self.glsl_emulated_helper(
            name,
            f"{self.glsl_emulated_struct(result)} {name}("
            f"{self.glsl_emulated_parameter(left, 'a')}, "
            f"{self.glsl_emulated_parameter(right, 'b')}) {{\n"
            f"    return {self.glsl_emulated_columns(result, parts)};\n}}\n",
        )
        return (
            f"{name}({self.generate_expression(left['expr'])}, "
            f"{self.generate_expression(right['expr'])})"
        )

    def generate_glsl_emulated_reduction(self, func_name, operand):
        _scalar, rows, columns = operand["shape"]
        joiner = " && " if func_name == "all" else " || "
        terms = [
            f"{func_name}({self.glsl_emulated_bool_column(operand, 'a', i, rows)})"
            for i in range(columns)
        ]
        name = f"crossgl_{func_name}_{self.glsl_emulated_tag(operand)}"
        self.glsl_emulated_helper(
            name,
            f"bool {name}({self.glsl_emulated_parameter(operand, 'a')}) {{\n"
            f"    return {joiner.join(terms)};\n}}\n",
        )
        return f"{name}({self.generate_expression(operand['expr'])})"

    def generate_glsl_emulated_transpose(self, operand):
        scalar, rows, columns = operand["shape"]
        result = (scalar, columns, rows)
        column_type = vector_type_name(scalar, columns)
        parts = [
            f"{column_type}({', '.join(f'a.c[{c}][{r}]' for c in range(columns))})"
            for r in range(rows)
        ]
        name = f"crossgl_transpose_{self.glsl_emulated_tag(operand)}"
        self.glsl_emulated_helper(
            name,
            f"{self.glsl_emulated_struct(result)} {name}"
            f"({self.glsl_emulated_parameter(operand, 'a')}) {{\n"
            f"    return {self.glsl_emulated_columns(result, parts)};\n}}\n",
        )
        return f"{name}({self.generate_expression(operand['expr'])})"

    def generate_glsl_emulated_componentwise(self, func_name, operands):
        matrices = [operand for operand in operands if operand["kind"] == "matrix"]
        shape = matrices[0]["shape"]
        for operand in operands:
            if operand["kind"] == "matrix" and operand["shape"] != shape:
                self.glsl_emulated_matrix_error(func_name, operands, "operand-type")
            if operand["kind"] == "scalar" and operand["scalar"] != shape[0]:
                self.glsl_emulated_matrix_error(func_name, operands, "operand-type")
            if operand["kind"] not in {"matrix", "scalar"}:
                self.glsl_emulated_matrix_error(func_name, operands, "operand-shape")
        if shape[0] == "bool" or (func_name == "sign" and shape[0] == "uint"):
            self.glsl_emulated_matrix_error(func_name, operands, "operand-type")
        variables = [f"v{index}" for index in range(len(operands))]
        parts = []
        for column in range(shape[2]):
            values = [
                f"{var}.c[{column}]" if operand["kind"] == "matrix" else var
                for operand, var in zip(operands, variables)
            ]
            parts.append(f"{func_name}({', '.join(values)})")
        name = f"crossgl_{func_name}_" + "_".join(
            self.glsl_emulated_tag(operand) for operand in operands
        )
        parameters = ", ".join(
            self.glsl_emulated_parameter(operand, var)
            for operand, var in zip(operands, variables)
        )
        self.glsl_emulated_helper(
            name,
            f"{self.glsl_emulated_struct(shape)} {name}({parameters}) {{\n"
            f"    return {self.glsl_emulated_columns(shape, parts)};\n}}\n",
        )
        rendered = ", ".join(
            self.generate_expression(operand["expr"]) for operand in operands
        )
        return f"{name}({rendered})"

    def generate_glsl_emulated_matrix_compound_assignment(self, target, value, op):
        """``m op= x`` on an emulated matrix as ``m = m op x``."""
        if not getattr(self, "glsl_program_uses_emulated_matrices", False):
            return None
        binary = self.COMPOUND_ASSIGNMENT_BINARY_OPERATORS.get(op)
        if binary is None or not self.glsl_may_be_emulated_matrix(target):
            return None
        if self.glsl_emulated_operand(target)["kind"] != "matrix":
            return None
        for node in self.walk_ast(target):
            if isinstance(node, (FunctionCallNode, UnaryOpNode)):
                self.glsl_emulated_matrix_error(
                    op,
                    (self.glsl_emulated_operand(target),),
                    "compound-target-side-effects",
                )
        rendered = self.generate_glsl_emulated_matrix_binary(
            BinaryOpNode(target, binary, value), binary
        )
        return f"{self.generate_expression(target)} = {rendered}"

    def generate_glsl_hlsl_logical_intrinsic(self, func_name, args):
        """HLSL ``and``/``or`` on scalars and vectors (matrices are emulated).

        GLSL ``&&``/``||`` only accept scalars, so vectors combine as ``0``/``1``
        unsigned lanes.
        """
        if func_name not in {"and", "or"} or len(args) != 2:
            return None
        if func_name in getattr(self, "function_return_types", {}):
            return None
        infos = [
            self.glsl_value_type_info(self.glsl_source_expression_type(arg))
            for arg in args
        ]
        if any(info is None for info in infos):
            return None
        width = max(info["width"] for info in infos)
        rendered = [self.generate_expression(arg) for arg in args]
        if width == 1:
            operator = "&&" if func_name == "and" else "||"
            return f"(bool({rendered[0]}) {operator} bool({rendered[1]}))"
        bool_type = vector_type_name("bool", width)
        uint_type = vector_type_name("uint", width)
        lanes = []
        for value, info in zip(rendered, infos):
            if info["family"] != "bool" or info["width"] != width:
                value = f"{bool_type}({value})"
            lanes.append(f"{uint_type}({value})")
        operator = "&" if func_name == "and" else "|"
        return f"{bool_type}({lanes[0]} {operator} {lanes[1]})"

    def generate_glsl_emulated_matrix_support(self):
        """Struct definitions and helpers, emitted ahead of their first use."""
        shapes = sorted(getattr(self, "glsl_emulated_matrix_shapes", ()) or ())
        if not shapes:
            return ""
        code = ""
        for shape in shapes:
            code += (
                f"struct {emulated_matrix_struct_name(shape)} {{\n"
                f"    {vector_type_name(shape[0], shape[1])} c[{shape[2]}];\n"
                "};\n"
            )
        code += "\n"
        for text in self.glsl_emulated_matrix_helpers.values():
            code += text + "\n"
        return code
