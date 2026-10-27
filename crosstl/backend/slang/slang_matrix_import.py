"""Import Slang matrices with CrossGL's column-by-row meaning (ADR 0004).

Slang matrices follow HLSL: ``floatRxC`` has R rows and C columns, ``m[i]`` is row ``i``, constructors and
initializer lists fill rows, ``m._mRC`` (zero-based) and ``m._RC`` (one-based) name element (R, C), ``*`` between
two matrices is componentwise and ``mul`` is the linear-algebra product. CrossGL follows GLSL, so the same matrix
is ``matCxR``: C columns of R components, ``m[c][r]`` is element (r, c), constructors fill columns and ``*`` is the
linear-algebra product.

The importer keeps the HLSL importer's convention (:mod:`crosstl.backend.DirectX.hlsl_matrix_import`): the CrossGL
value is the Slang matrix itself. Constructors and initializer lists are re-listed column by column, ``m[i][j]``
becomes ``m[j][i]``, a row read becomes ``transpose(m)[i]``, a row write calls a generated setter and ``*`` of two
matrices becomes ``matrixCompMult``, so ``mul(a, b)`` is the CrossGL ``a * b``. Slang's default ``column_major``
layout stores the C columns of a ``floatRxC`` as CrossGL stores those of ``matCxR``; ``row_major`` and
``column_major`` declarations are carried as ``@ row_major`` and ``@ column_major``.

Expressions whose matrix shape the importer cannot determine keep their spelling. Matrix constructs it cannot write
faithfully (a conversion from a value of unknown type, a row write through a multi-component swizzle, a struct
initializer list with a matrix member) raise :class:`SlangImportRefused`.
"""

from __future__ import annotations

import re

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    InitializerListNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)
from ..DirectX.hlsl_matrix_import import crossgl_matrix_type_name
from ..import_refusals import SlangImportRefused

_MATRIX_SCALARS = ("float", "double", "int", "uint", "bool")
_MATRIX_TYPE = re.compile(r"(float|double|int|uint|bool)([1-4])x([1-4])")
_VECTOR_TYPE = re.compile(r"(float|half|double|int|uint|bool)([1-4])")
_ZERO_BASED_ELEMENTS = re.compile(r"(?:_m[0-3][0-3])+")
_ONE_BASED_ELEMENTS = re.compile(r"(?:_[1-4][1-4])+")
_BOOL_LITERALS = frozenset({"true", "false"})
_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
_COMPARISON_OPERATORS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_LOGICAL_OPERATORS = frozenset({"&&", "||"})
_BUFFER_ELEMENT_RESOURCES = frozenset({"StructuredBuffer", "RWStructuredBuffer"})
# Componentwise intrinsics: the result has the shape of their matrix (or widest) argument.
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
_SCALAR_RESULT_INTRINSICS = frozenset({"dot", "length", "distance", "determinant"})


def _refuse(reason):
    raise SlangImportRefused([reason])


def _literal_scalar_type(text):
    """Scalar type of a Slang literal token, ``None`` when ``text`` is not a literal."""
    if text in _BOOL_LITERALS:
        return "bool"
    if not text or not (text[0].isdigit() or text[0] == "."):
        return None
    lowered = text.lower()
    if lowered.startswith("0x"):
        return "uint" if lowered.endswith("u") else "int"
    if any(marker in lowered for marker in (".", "e")) or lowered.endswith(
        ("f", "h", "l")
    ):
        return "double" if lowered.endswith("l") else "float"
    return "uint" if lowered.endswith("u") else "int"


class SlangMatrixImportMixin:
    """Matrix lowerings of :class:`SlangToCrossGLConverter`."""

    # ------------------------------------------------------------- state

    def reset_matrix_import_state(self, ast, structs):
        """Collect the array ranks and return types the shape analysis needs; forget earlier setters."""
        self.matrix_row_setters = {}
        self.matrix_struct_member_ranks = {}
        for struct in structs:
            self._collect_matrix_member_ranks(struct)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self._collect_matrix_member_ranks(cbuffer)
        global_ranks = {}
        self.matrix_cbuffer_member_types = {}
        for node in getattr(ast, "global_vars", []) or []:
            declaration = node.left if isinstance(node, AssignmentNode) else node
            name = getattr(declaration, "name", None)
            if name is not None:
                global_ranks[name] = len(
                    getattr(declaration, "array_sizes", None) or []
                )
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            for instance in getattr(cbuffer, "instances", []) or []:
                name = getattr(instance, "name", None)
                if name is not None:
                    global_ranks[name] = len(
                        getattr(instance, "array_sizes", None) or []
                    )
            if getattr(cbuffer, "instances", None):
                continue
            for member in getattr(cbuffer, "members", []) or []:
                name = getattr(member, "name", None)
                if name is not None and getattr(member, "vtype", None):
                    self.matrix_cbuffer_member_types[name] = (
                        member.vtype,
                        len(getattr(member, "array_sizes", None) or []),
                    )
        self.matrix_rank_scopes = [global_ranks]
        returns = {}
        for function in getattr(ast, "functions", []) or []:
            name = getattr(function, "name", None)
            if name is not None:
                returns.setdefault(name, set()).add(
                    str(getattr(function, "return_type", None) or "")
                )
        self.matrix_function_return_types = returns

    def _collect_matrix_member_ranks(self, struct):
        name = getattr(struct, "name", None)
        if name is None:
            return
        ranks = self.matrix_struct_member_ranks.setdefault(name, {})
        for member in getattr(struct, "members", []) or []:
            member_name = getattr(member, "name", None)
            if member_name is not None:
                ranks[member_name] = len(getattr(member, "array_sizes", None) or [])
        for nested in getattr(struct, "structs", []) or []:
            self._collect_matrix_member_ranks(nested)

    def push_matrix_rank_scope(self, params):
        scope = {}
        for param in params or []:
            name = getattr(param, "name", None)
            if name is not None:
                scope[name] = len(getattr(param, "array_sizes", None) or [])
        self.matrix_rank_scopes.append(scope)

    def pop_matrix_rank_scope(self):
        if len(self.matrix_rank_scopes) > 1:
            self.matrix_rank_scopes.pop()

    def record_matrix_rank(self, node):
        name = getattr(node, "name", None)
        if name is not None:
            self.matrix_rank_scopes[-1][name] = len(
                getattr(node, "array_sizes", None) or []
            )

    def record_inferred_matrix_type(self, node):
        """Give ``let m = <matrix>`` and ``var m = <matrix>`` the matrix type of their initializer."""
        target = node.left
        if not (
            isinstance(target, VariableNode)
            and target.vtype in {"let", "var"}
            and target.name is not None
        ):
            return
        typed = self.matrix_value_type(node.right)
        if typed is not None and not typed[1] and self.slang_matrix_shape(typed[0]):
            self.variable_type_scopes[-1][target.name] = typed[0]

    def _variable_rank(self, name):
        for scope in reversed(self.matrix_rank_scopes):
            if name in scope:
                return scope[name]
        return 0

    # ------------------------------------------------------------ shapes

    def slang_matrix_shape(self, type_name):
        """``(scalar, rows, cols)`` of a Slang matrix type with two to four rows and columns, else ``None``."""
        if not type_name:
            return None
        text = str(type_name).strip()
        if text == "matrix":
            return "float", 4, 4
        if text.startswith("matrix<") and text.endswith(">"):
            args = [str(arg).strip() for arg in self.generic_type_arguments(text)]
            if len(args) != 3 or args[1] not in "1234" or args[2] not in "1234":
                return None
            scalar, rows, cols = args[0], int(args[1]), int(args[2])
        else:
            match = _MATRIX_TYPE.fullmatch(text)
            if match is None:
                return None
            scalar, rows, cols = (
                match.group(1),
                int(match.group(2)),
                int(match.group(3)),
            )
        if scalar not in _MATRIX_SCALARS or not (2 <= rows <= 4 and 2 <= cols <= 4):
            return None
        return scalar, rows, cols

    def map_slang_matrix_type(self, type_name):
        """CrossGL spelling of a Slang matrix type, ``None`` for other types."""
        shape = self.slang_matrix_shape(type_name)
        return None if shape is None else crossgl_matrix_type_name(*shape)

    def matrix_shape(self, expr):
        """Matrix shape of the value of ``expr``; ``None`` for non-matrices, arrays and unknown types."""
        typed = self.matrix_value_type(expr)
        if typed is None or typed[1]:
            return None
        return self.slang_matrix_shape(typed[0])

    def matrix_value_type(self, expr):
        """``(Slang type, remaining array rank)`` of the value of ``expr``, ``None`` when unknown."""
        if isinstance(expr, str):
            scalar = _literal_scalar_type(expr)
            return None if scalar is None else (scalar, 0)
        if isinstance(expr, VariableNode):
            if expr.vtype:
                return expr.vtype, len(getattr(expr, "array_sizes", None) or [])
            vtype = self.lookup_variable_type(expr.name)
            if vtype:
                return vtype, self._variable_rank(expr.name)
            return self.matrix_cbuffer_member_types.get(expr.name)
        if isinstance(expr, MemberAccessNode):
            return self._member_value_type(expr)
        if isinstance(expr, ArrayAccessNode):
            return self._element_value_type(expr)
        if isinstance(expr, VectorConstructorNode):
            return expr.type_name, 0
        if isinstance(expr, CastNode):
            return expr.target_type, 0
        if isinstance(expr, FunctionCallNode) and isinstance(expr.name, str):
            return self._call_value_type(expr)
        if isinstance(expr, BinaryOpNode):
            return self._binary_value_type(expr)
        if isinstance(expr, UnaryOpNode):
            if expr.op == "!":
                return "bool", 0
            return self.matrix_value_type(expr.operand)
        if isinstance(expr, TernaryOpNode):
            return self.matrix_value_type(expr.true_expr) or self.matrix_value_type(
                expr.false_expr
            )
        if isinstance(expr, AssignmentNode):
            return self.matrix_value_type(expr.left)
        return None

    def _member_value_type(self, expr):
        qualified_name = self.member_access_qualified_name(expr)
        if qualified_name:
            qualified_type = self.lookup_variable_type(qualified_name)
            if qualified_type:
                return qualified_type, 0
        owner = self.matrix_value_type(expr.object)
        if owner is None or owner[1]:
            return None
        member = expr.member
        shape = self.slang_matrix_shape(owner[0])
        if shape is not None:
            elements = self.matrix_element_members(member)
            if not elements:
                return None
            return self._slang_vector_type(shape[0], len(elements)), 0
        struct_type = self.unwrap_resource_container_type(owner[0])
        members = self.struct_member_types.get(struct_type)
        if members is not None and member in members:
            rank = self.matrix_struct_member_ranks.get(struct_type, {}).get(member, 0)
            return members[member], rank
        swizzled = self.vector_swizzle_type(owner[0], member)
        return None if swizzled is None else (swizzled, 0)

    def _element_value_type(self, expr):
        owner = self.matrix_value_type(expr.array)
        if owner is None:
            return None
        vtype, rank = owner
        if rank:
            return vtype, rank - 1
        base = str(vtype).split("<", 1)[0].strip()
        if base in _BUFFER_ELEMENT_RESOURCES:
            args = self.generic_type_arguments(vtype)
            return (str(args[0]).strip(), 0) if args else None
        shape = self.slang_matrix_shape(vtype)
        if shape is not None:
            return self._slang_vector_type(shape[0], shape[2]), 0
        vector = _VECTOR_TYPE.fullmatch(str(vtype).strip())
        if vector is not None:
            return vector.group(1), 0
        return None

    def _call_value_type(self, expr):
        name, _ = self.canonical_function_call_name(expr.name)
        if expr.name in self.user_function_names:
            returns = self.matrix_function_return_types.get(expr.name, set())
            return (next(iter(returns)), 0) if len(returns) == 1 else None
        if self.slang_matrix_shape(name) is not None:
            return name, 0
        if name in self.struct_member_types:
            return name, 0
        args = list(expr.args or [])
        if name == "mul" and len(args) == 2:
            left = self.matrix_shape(args[0])
            right = self.matrix_shape(args[1])
            if left is not None and right is not None:
                return f"{left[0]}{left[1]}x{right[2]}", 0
            if left is not None:
                return self._slang_vector_type(left[0], left[1]), 0
            if right is not None:
                return self._slang_vector_type(right[0], right[2]), 0
            return None
        if name == "transpose" and len(args) == 1:
            shape = self.matrix_shape(args[0])
            return None if shape is None else (f"{shape[0]}{shape[2]}x{shape[1]}", 0)
        if name in _ELEMENTWISE_INTRINSICS:
            typed = [self.matrix_value_type(arg) for arg in args]
            for candidate in typed:
                if candidate and self.slang_matrix_shape(candidate[0]):
                    return candidate
            for candidate in typed:
                if candidate and _VECTOR_TYPE.fullmatch(str(candidate[0]).strip()):
                    return candidate
            return typed[0] if typed and typed[0] else None
        if name in _SCALAR_RESULT_INTRINSICS and args:
            typed = self.matrix_value_type(args[0])
            return None if typed is None else (self._slang_scalar_type(typed[0]), 0)
        return None

    def _binary_value_type(self, expr):
        if expr.op in _LOGICAL_OPERATORS:
            return "bool", 0
        left = self.matrix_value_type(expr.left)
        right = self.matrix_value_type(expr.right)
        if expr.op in _COMPARISON_OPERATORS:
            for side in (left, right):
                shape = self.slang_matrix_shape(side[0]) if side else None
                if shape is not None:
                    return f"bool{shape[1]}x{shape[2]}", 0
            for side in (left, right):
                vector = _VECTOR_TYPE.fullmatch(str(side[0]).strip()) if side else None
                if vector is not None:
                    return f"bool{vector.group(2)}", 0
            return "bool", 0
        for side in (left, right):
            if side and not side[1] and self.slang_matrix_shape(side[0]):
                return side
        for side in (left, right):
            if side and not side[1] and _VECTOR_TYPE.fullmatch(str(side[0]).strip()):
                return side
        return left or right

    def _slang_vector_type(self, scalar, width):
        return scalar if width == 1 else f"{scalar}{width}"

    def _slang_scalar_type(self, vtype):
        text = str(vtype).strip()
        shape = self.slang_matrix_shape(text)
        if shape is not None:
            return shape[0]
        vector = _VECTOR_TYPE.fullmatch(text)
        return vector.group(1) if vector is not None else text

    def matrix_component_count(self, expr):
        """Scalar component count of a scalar or vector ``expr``; ``None`` for matrices and unknown types."""
        typed = self.matrix_value_type(expr)
        if typed is None or typed[1]:
            return None
        text = str(typed[0]).strip()
        if text in _MATRIX_SCALARS or text == "half":
            return 1
        vector = _VECTOR_TYPE.fullmatch(text)
        return int(vector.group(2)) if vector is not None else None

    def matrix_operand_is_pure(self, expr):
        """Whether ``expr`` can be evaluated twice: names, members, literals and side-effect-free indexing."""
        if isinstance(expr, (str, VariableNode)):
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
            return expr.op not in {
                "++",
                "--",
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            } and self.matrix_operand_is_pure(expr.operand)
        return False

    # --------------------------------------------------------- rendering

    def _crossgl_vector_type(self, scalar, width):
        if width == 1:
            return self.map_type(scalar)
        return self.map_generic_vector_type(f"vector<{scalar}, {width}>") or (
            f"{scalar}{width}"
        )

    def _postfix_operand(self, expr, is_main):
        rendered = self.generate_expression(expr, is_main)
        if isinstance(expr, (BinaryOpNode, TernaryOpNode, AssignmentNode, UnaryOpNode)):
            return f"({rendered})"
        return rendered

    def matrix_row_access(self, expr):
        """``(matrix, row)`` when ``expr`` is ``m[row]`` on a matrix value ``m``."""
        if not isinstance(expr, ArrayAccessNode):
            return None
        if self.matrix_shape(expr.array) is None:
            return None
        return expr.array, expr.index

    def matrix_index_expression(self, expr, is_main):
        """``m[i][j]`` -> ``m[j][i]`` and the row read ``m[i]`` -> ``transpose(m)[i]``; ``None`` otherwise."""
        inner = self.matrix_row_access(expr.array)
        if inner is not None:
            matrix, row = inner
            rendered = self._postfix_operand(matrix, is_main)
            row_text = self.generate_subscript_index(row, is_main)
            column_text = self.generate_subscript_index(expr.index, is_main)
            return f"{rendered}[{column_text}][{row_text}]"
        access = self.matrix_row_access(expr)
        if access is None:
            return None
        matrix, row = access
        rendered = self.generate_expression(matrix, is_main)
        return f"transpose({rendered})[{self.generate_subscript_index(row, is_main)}]"

    def matrix_element_members(self, member):
        """``[(row, col), ...]`` for a ``_mRC`` or ``_RC`` matrix swizzle, else ``None``."""
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
        """Element swizzles ``m._mRC...`` and single-component swizzles ``m[i].x`` of a row."""
        member = expr.member
        shape = self.matrix_shape(expr.object)
        elements = self.matrix_element_members(member) if shape else None
        if elements:
            if any(row >= shape[1] or col >= shape[2] for row, col in elements):
                _refuse(
                    f"Slang matrix swizzle .{member} is outside "
                    f"{shape[0]}{shape[1]}x{shape[2]}"
                )
            rendered = self._postfix_operand(expr.object, is_main)
            parts = [f"{rendered}[{col}][{row}]" for row, col in elements]
            if len(parts) == 1:
                return parts[0]
            vector_type = self._crossgl_vector_type(shape[0], len(parts))
            return f"{vector_type}({', '.join(parts)})"
        access = self.matrix_row_access(expr.object)
        if access is None or not isinstance(member, str) or len(member) != 1:
            return None
        column = "xyzw".find(member) if member in "xyzw" else "rgba".find(member)
        if column < 0:
            return None
        matrix, row = access
        rendered = self._postfix_operand(matrix, is_main)
        return f"{rendered}[{column}][{self.generate_subscript_index(row, is_main)}]"

    def matrix_constructor_expression(self, type_name, args, is_main):
        """``floatRxC(...)`` fills rows; render the equal CrossGL ``matCxR`` value, ``None`` for other types."""
        if not isinstance(type_name, str) or type_name in self.user_function_names:
            return None
        shape = self.slang_matrix_shape(type_name)
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
        """CrossGL value of the Slang matrix whose rows the rendered argument list fills."""
        scalar, rows, cols = shape
        return f"transpose({crossgl_matrix_type_name(scalar, cols, rows)}({rendered}))"

    def column_filled_matrix_expression(self, shape, leaves, is_main):
        """Row-major scalar elements of side-effect-free leaves, re-listed column by column.

        The result stays a constant expression when the elements are, which ``transpose(...)`` is not in GLSL ES
        3.00 or in glslang's global initializers.
        """
        scalar, rows, cols = shape
        if len(leaves) != rows * cols:
            return None
        if any(
            self.matrix_component_count(leaf) != 1
            or not self.matrix_operand_is_pure(leaf)
            for leaf in leaves
        ):
            return None
        rendered = [self.generate_expression(leaf, is_main) for leaf in leaves]
        ordered = [
            rendered[row * cols + col] for col in range(cols) for row in range(rows)
        ]
        return f"{crossgl_matrix_type_name(*shape)}({', '.join(ordered)})"

    def matrix_conversion_expression(self, shape, value, is_main):
        """``(floatRxC)value`` and one-argument constructors: truncate a matrix, splat a scalar, reshape a vector."""
        rendered = self.generate_expression(value, is_main)
        target = crossgl_matrix_type_name(*shape)
        if self.matrix_shape(value) is not None:
            # GLSL keeps the upper-left columns and rows, the elements Slang's truncation keeps.
            return f"{target}({rendered})"
        count = self.matrix_component_count(value)
        if count is not None and count > 1:
            # A vector lists the elements row by row.
            return self.row_filled_matrix_expression(shape, rendered)
        if count is None:
            _refuse(
                f"Slang conversion to {shape[0]}{shape[1]}x{shape[2]} from a value whose "
                "type the importer cannot determine"
            )
        if not self.matrix_operand_is_pure(value):
            _refuse(
                f"Slang scalar-to-{shape[0]}{shape[1]}x{shape[2]} conversion of an "
                "expression with side effects"
            )
        column_type = self._crossgl_vector_type(shape[0], shape[1])
        columns = ", ".join(f"{column_type}({rendered})" for _ in range(shape[2]))
        return f"{target}({columns})"

    def matrix_cast_expression(self, expr, is_main):
        shape = self.slang_matrix_shape(expr.target_type)
        if shape is None:
            return None
        return self.matrix_conversion_expression(shape, expr.expression, is_main)

    def matrix_initializer_expression(self, declared_type, rank, initializer, is_main):
        """A brace initializer of a Slang matrix (or array of matrices) lists rows; render CrossGL constructors."""
        if not isinstance(initializer, InitializerListNode):
            return None
        shape = self.slang_matrix_shape(declared_type)
        if shape is None:
            self._refuse_struct_matrix_initializer(declared_type)
            return None
        if rank > 1:
            _refuse(
                f"Slang initializer list for a {rank}-dimensional array of "
                f"{shape[0]}{shape[1]}x{shape[2]}"
            )
        if rank == 0:
            return self._matrix_from_leaves(
                shape, self._flatten_initializer(initializer), is_main
            )
        size = shape[1] * shape[2]
        elements = list(initializer.elements or [])
        if elements and all(
            isinstance(element, InitializerListNode)
            or self.matrix_shape(element) is not None
            for element in elements
        ):
            matrices = [
                (
                    self._matrix_from_leaves(
                        shape, self._flatten_initializer(element), is_main
                    )
                    if isinstance(element, InitializerListNode)
                    else self.generate_expression(element, is_main)
                )
                for element in elements
            ]
            return f"{{{', '.join(matrices)}}}"
        leaves = self._flatten_initializer(initializer)
        counts = [self.matrix_component_count(leaf) for leaf in leaves]
        if None in counts or any(count != 1 for count in counts) or len(leaves) % size:
            _refuse(
                f"Slang initializer list for an array of {shape[0]}{shape[1]}x{shape[2]} "
                "whose elements the importer cannot match to the matrices"
            )
        matrices = [
            self._matrix_from_leaves(shape, leaves[start : start + size], is_main)
            for start in range(0, len(leaves), size)
        ]
        return f"{{{', '.join(matrices)}}}"

    def _matrix_from_leaves(self, shape, leaves, is_main):
        counts = [self.matrix_component_count(leaf) for leaf in leaves]
        if None in counts or sum(counts) != shape[1] * shape[2]:
            _refuse(
                f"Slang initializer list for {shape[0]}{shape[1]}x{shape[2]} whose "
                "component count the importer cannot match to the matrix"
            )
        reordered = self.column_filled_matrix_expression(shape, leaves, is_main)
        if reordered is not None:
            return reordered
        rendered = ", ".join(self.generate_expression(leaf, is_main) for leaf in leaves)
        return self.row_filled_matrix_expression(shape, rendered)

    def _flatten_initializer(self, node):
        if isinstance(node, InitializerListNode):
            leaves = []
            for element in node.elements or []:
                leaves.extend(self._flatten_initializer(element))
            return leaves
        return [node]

    def _refuse_struct_matrix_initializer(self, declared_type, seen=None):
        """Refuse a brace initializer of a struct that holds a matrix, whose rows it would list unconverted."""
        struct_type = self.unwrap_resource_container_type(declared_type)
        members = self.struct_member_types.get(struct_type)
        if not members:
            return
        seen = set() if seen is None else seen
        if struct_type in seen:
            return
        seen.add(struct_type)
        for member_type in members.values():
            if self.slang_matrix_shape(member_type) is not None:
                _refuse(
                    f"Slang initializer list for {struct_type}, which has a matrix member"
                )
            self._refuse_struct_matrix_initializer(member_type, seen)

    def matrix_product_expression(self, expr, left, right):
        """Slang ``*`` of two matrices is componentwise."""
        if expr.op != "*":
            return None
        if (
            self.matrix_shape(expr.left) is None
            or self.matrix_shape(expr.right) is None
        ):
            return None
        return f"matrixCompMult({left}, {right})"

    # ------------------------------------------------------------ assignment

    def matrix_assignment_expression(self, node, is_main):
        """Lower assignments Slang spells through rows or componentwise: ``m[i] = v``, ``m *= n``."""
        target = node.left
        operator = node.operator
        if isinstance(target, VariableNode) and target.vtype:
            initializer = self.matrix_initializer_expression(
                target.vtype,
                len(getattr(target, "array_sizes", None) or []),
                node.right,
                is_main,
            )
            if initializer is None:
                return None
            return f"{self.generate_expression(target, is_main)} = {initializer}"
        if operator == "*=" and self.matrix_shape(target) is not None:
            if self.matrix_shape(node.right) is None:
                return None
            if not self.matrix_operand_is_pure(target):
                _refuse("Slang componentwise matrix *= on a target with side effects")
            lhs = self.generate_expression(target, is_main)
            rhs = self.generate_expression(node.right, is_main)
            return f"{lhs} = matrixCompMult({lhs}, {rhs})"
        if isinstance(target, MemberAccessNode):
            elements = self.matrix_element_members(target.member)
            if (
                elements
                and len(elements) > 1
                and self.matrix_shape(target.object) is not None
            ) or (
                self.matrix_row_access(target.object) is not None
                and len(str(target.member)) > 1
            ):
                _refuse(
                    "Slang assignment through the multi-component matrix swizzle "
                    f".{target.member}"
                )
            return None
        access = self.matrix_row_access(target)
        if access is None:
            return None
        matrix, row = access
        if not (
            self.matrix_operand_is_pure(matrix) and self.matrix_operand_is_pure(row)
        ):
            _refuse(
                "Slang matrix row assignment whose matrix or row index has side effects"
            )
        shape = self.matrix_shape(matrix)
        matrix_text = self.generate_expression(matrix, is_main)
        row_text = self.generate_subscript_index(row, is_main)
        value = self.generate_expression(node.right, is_main)
        row_type = self._crossgl_vector_type(shape[0], shape[2])
        if operator != "=":
            binary = operator[:-1]
            if binary not in _ARITHMETIC_OPERATORS:
                _refuse(f"Slang matrix row compound assignment {operator}")
            value = f"transpose({matrix_text})[{row_text}] {binary} ({value})"
        if operator != "=" or self.matrix_value_type(node.right) != (
            self._slang_vector_type(shape[0], shape[2]),
            0,
        ):
            value = f"{row_type}({value})"
        helper = self.matrix_row_setter_name(shape)
        return f"{helper}({matrix_text}, int({row_text}), {value})"

    def refuse_matrix_row_update(self, expr):
        """``++``/``--`` of a matrix row has no CrossGL lvalue."""
        if expr.op in {
            "PRE_INCREMENT",
            "PRE_DECREMENT",
            "POST_INCREMENT",
            "POST_DECREMENT",
        } and (
            self.matrix_row_access(expr.operand) is not None
            or (
                isinstance(expr.operand, MemberAccessNode)
                and self.matrix_row_access(expr.operand.object) is not None
                and len(str(expr.operand.member)) > 1
            )
        ):
            _refuse("Slang increment or decrement of a matrix row")

    def matrix_row_setter_name(self, shape):
        key = crossgl_matrix_type_name(*shape)
        if key not in self.matrix_row_setters:
            name = f"cgl_slang_set_row_{key}"
            taken = {entry[0] for entry in self.matrix_row_setters.values()}
            while name in self.user_function_names or name in taken:
                name += "_"
            self.matrix_row_setters[key] = (name, shape)
        return self.matrix_row_setters[key][0]

    def generate_matrix_row_setters(self):
        code = ""
        for matrix_type, (name, shape) in sorted(self.matrix_row_setters.items()):
            row_type = self._crossgl_vector_type(shape[0], shape[2])
            code += (
                f"    {row_type} {name}(inout {matrix_type} target, int row, "
                f"{row_type} value) {{\n"
            )
            for column in range(shape[2]):
                code += f"        target[{column}][row] = value.{'xyzw'[column]};\n"
            code += "        return value;\n    }\n\n"
        return code

    # ---------------------------------------------------------------- layout

    def matrix_layout_attributes(self, node, indent):
        """``@ row_major`` / ``@ column_major`` lines for a declaration that carries the qualifier."""
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        return "".join(
            "    " * indent + f"@ {qualifier}\n"
            for qualifier in ("row_major", "column_major")
            if qualifier in qualifiers
        )
