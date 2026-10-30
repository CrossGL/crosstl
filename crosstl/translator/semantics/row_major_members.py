"""Row-major struct members on targets whose structs cannot carry a matrix layout.

A ``@ row_major`` struct member stores its matrix row by row: HLSL and Slang ``row_major``, GLSL ``layout(row_major)``
on a buffer block member, SPIR-V ``RowMajor``. GLSL allows no layout qualifier on a structure member (GLSL 4.60
section 4.1.8), and MSL, WGSL, CUDA, HIP, Rust and Mojo have no row-major matrices, so their generators declared the
member column-major and every read, every write and the offset of every member after it changed meaning.

A row-major ``matCxR`` (``C`` columns of ``R`` components) is laid out as ``R`` row vectors of ``C`` components. That
is the layout of the column-major ``matRxC`` which is its transpose, in every buffer layout CrossGL reproduces (std140,
std430, scalar, HLSL constant and structured packing, MSL) and in stage interface slots. For those targets
:func:`store_row_major_members_transposed` declares such a member as its transpose and rewrites every use, so the
struct keeps its bytes:

* a read of the matrix ``s.m`` becomes ``transpose(s.m)``, and of a column ``s.m[i]`` ``transpose(s.m)[i]``;
* an element ``s.m[i][j]``, read or written, becomes ``s.m[j][i]`` when neither index has side effects;
* ``s.m = v`` becomes ``s.m = transpose(v)``, and ``s.m op= v`` ``s.m = transpose(transpose(s.m) op v)``; ``++`` and
  ``--`` of the whole matrix are componentwise and stay as they are;
* a struct constructor ``S(...)`` or brace initializer passes ``transpose(v)`` for the member.

Uses the rewrite cannot keep exact raise :class:`RowMajorMemberError`: a write to a column or to components of one,
a matrix or column passed to an ``out``, ``inout`` or reference parameter, an array member used without selecting
one matrix, and a write through an object or index with side effects. Constant buffer members keep the row-major
handling of each generator and are left alone, except on Metal, whose unmarked constant blocks
:mod:`.row_major_constant_members` stores the same way. HLSL and Slang declare ``row_major`` themselves, so for them only the
types of GLSL buffer blocks, which their generators lower to byte-address buffers read column by column, change.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..codegen.buffer_member_layout import member_is_row_major
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, element_type_node, iter_functions
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child
from .scoped_walk import ScopedWalker, program_stages

__all__ = [
    "BLOCK_ONLY_TARGETS",
    "RowMajorMemberError",
    "TARGETS",
    "store_row_major_members_transposed",
]

TARGETS = frozenset(
    {
        "opengl",
        "glsl",
        "webgl",
        "gles",
        "essl",
        "vulkan",
        "spirv",
        "metal",
        "wgsl",
        "cuda",
        "hip",
        "rust",
        "mojo",
    }
)

# HLSL and Slang declare row_major themselves, but lower a GLSL buffer block to a byte-address buffer whose loads and
# stores read every matrix column by column; only block types are transposed for them.
BLOCK_ONLY_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "slang"}
)

_SWIZZLE = re.compile(r"[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4}")
_LAYOUT_ATTRIBUTES = frozenset({"row_major", "column_major"})


class RowMajorMemberError(TranslationDiagnostic):
    """A use of a row-major struct member that its transposed storage cannot reproduce."""

    code = f"{CODE_PREFIX}row-major-member-unsupported"


class _Member:
    """A row-major member: its struct, name and the number of array dimensions around the matrix."""

    def __init__(self, struct, member, dims):
        self.struct = struct
        self.name = member.name
        self.dims = dims


def _matrix_dims(member_type):
    """``(matrix, dims)`` for a matrix or an array of matrices, else ``(None, 0)``."""
    dims = 0
    node = member_type
    while isinstance(node, A.ArrayType):
        node = node.element_type
        dims += 1
    if isinstance(node, A.MatrixType):
        return node, dims
    return None, 0


def _transposed(member_type):
    if isinstance(member_type, A.ArrayType):
        duplicate = clone(member_type)
        duplicate.element_type = _transposed(member_type.element_type)
        return duplicate
    return A.MatrixType(
        member_type.element_type, rows=member_type.cols, cols=member_type.rows
    )


def _program_structs(program):
    structs = list(getattr(program, "structs", None) or [])
    for stage in program_stages(program):
        structs.extend(getattr(stage, "local_structs", None) or [])
    return [
        struct
        for struct in structs
        if isinstance(struct, A.StructNode) and not getattr(struct, "is_cbuffer", False)
    ]


def _block_type_names(program):
    """Names of the structs that GLSL buffer blocks (``@ glsl_buffer_block``) are declared with."""
    names = set()
    declarations = list(getattr(program, "global_variables", None) or [])
    for stage in program_stages(program):
        declarations.extend(getattr(stage, "local_variables", None) or [])
    for declaration in declarations:
        attributes = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(declaration, "attributes", None) or []
        }
        if "glsl_buffer_block" not in attributes:
            continue
        var_type = getattr(declaration, "var_type", None)
        while isinstance(var_type, A.ArrayType):
            var_type = var_type.element_type
        name = _type_name(var_type)
        if name:
            names.add(name)
    return names


def _collect_members(program, only=None):
    """Declare every row-major matrix member transposed; returns ``{struct: {member: _Member}}``.

    ``only``, when given, limits the structs to those names.
    """
    found = {}
    for struct in _program_structs(program):
        if only is not None and struct.name not in only:
            continue
        for member in getattr(struct, "members", None) or []:
            if not isinstance(member, A.StructMemberNode) or not member_is_row_major(
                member
            ):
                continue
            matrix, dims = _matrix_dims(member.member_type)
            if matrix is None:
                continue
            member.member_type = _transposed(member.member_type)
            member.attributes = [
                attribute
                for attribute in member.attributes
                if str(getattr(attribute, "name", "")).lower() not in _LAYOUT_ATTRIBUTES
            ]
            found.setdefault(struct.name, {})[member.name] = _Member(
                struct.name, member, dims
            )
    return found


def _type_name(node):
    if isinstance(node, A.ReferenceType):
        return _type_name(node.referenced_type)
    if isinstance(node, (A.NamedType, A.PrimitiveType)):
        return node.name
    return None


class _Sites(ScopedWalker):
    """Finds the member accesses that name a row-major member, with the declarations in scope."""

    def __init__(self, program, members, target):
        super().__init__(program)
        self.target = target
        self.members = members
        self.names = {name for table in members.values() for name in table}
        self.sites = {}
        self.literal_types = {}

    def _object_type(self, expr):
        node = self.scope.node_type(expr)
        if node is None and isinstance(expr, A.FunctionCallNode):
            node = self.scope.functions.get(call_name(expr))
        if node is None and isinstance(expr, A.TernaryOpNode):
            node = self._object_type(expr.true_expr) or self._object_type(
                expr.false_expr
            )
        return node

    def visit(self, node):
        if isinstance(node, A.AssignmentNode) and isinstance(
            node.value, A.ArrayLiteralNode
        ):
            self.literal_types[id(node.value)] = self.scope.node_type(node.target)
            return
        if isinstance(node, A.PointerAccessNode):
            pointee = element_type_node(self.scope.node_type(node.pointer_expr))
            self._record(node, _type_name(pointee), node.member)
            return
        if not isinstance(node, A.MemberAccessNode) or node.member not in self.names:
            return
        object_type = self._object_type(node.object_expr)
        if object_type is None:
            inferred = self.scope.infer(node.object_expr)
            if inferred is not None and _SWIZZLE.fullmatch(node.member):
                return
            raise RowMajorMemberError(
                f"cannot tell whether '.{node.member}' names a row-major matrix member, whose storage this "
                "target transposes",
                target=self.target,
            )
        self._record(node, _type_name(object_type), node.member)

    def _record(self, node, struct_name, member):
        info = self.members.get(struct_name, {}).get(member)
        if info is not None:
            self.sites[id(node)] = info


class _Rewriter:
    """Rewrites every use of a transposed member, top down, knowing whether each expression is read or written."""

    def __init__(self, sites, members, structs, functions, literal_types, target):
        self.sites = sites
        self.members = members
        self.struct_nodes = {struct.name: struct for struct in structs}
        self.functions = functions
        self.literal_types = literal_types
        self.target = target

    def refuse(self, what):
        raise RowMajorMemberError(
            f"{what} of a row-major matrix member cannot be expressed on {self.target}, whose structs store the "
            "matrix transposed",
            target=self.target,
        )

    # Matching -----------------------------------------------------------------------------------------------------
    def matrix(self, node):
        """The member access when ``node`` evaluates to one stored (transposed) matrix, else ``None``."""
        indices = 0
        base = node
        while isinstance(base, A.ArrayAccessNode):
            base = base.array_expr
            indices += 1
        info = self.sites.get(id(base))
        if info is None:
            return None
        if indices < info.dims:
            self.refuse(f"using the array '{info.name}' without selecting one matrix")
        return base if indices == info.dims else None

    def shape(self, node):
        """``("matrix", m)``, ``("column", m, i)`` or ``("element", m, i, j)`` when ``node`` reads into a member."""
        if self.matrix(node) is not None:
            return ("matrix", node)
        if isinstance(node, A.ArrayAccessNode):
            inner = node.array_expr
            if self.matrix(inner) is not None:
                return ("column", inner, node.index_expr)
            if isinstance(inner, A.ArrayAccessNode) and self.matrix(inner.array_expr):
                return ("element", inner.array_expr, inner.index_expr, node.index_expr)
        return None

    # Traversal ----------------------------------------------------------------------------------------------------
    def children(self, node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self.expression(child)
            if replacement is not child:
                set_child(node, field, key, replacement)

    def matrix_parts(self, matrix):
        """Rewrite the object and the array indices of a member access, which are reads."""
        node = matrix
        while isinstance(node, A.ArrayAccessNode):
            node.index_expr = node.index = self.expression(node.index_expr)
            node = node.array_expr
        if isinstance(node, A.PointerAccessNode):
            node.pointer_expr = self.expression(node.pointer_expr)
        elif isinstance(node, A.MemberAccessNode):
            node.object_expr = node.object = self.expression(node.object_expr)

    def expression(self, node):
        """Return the rewritten ``node``, read as a value."""
        if not isinstance(node, A.ASTNode):
            return node
        if isinstance(node, A.AssignmentNode):
            return self.assignment(node)
        if isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
            node.operand = self.target_expression(
                node.operand, "an increment or decrement"
            )
            return node
        if isinstance(node, A.FunctionCallNode):
            return self.call(node)
        if isinstance(node, A.VariableNode):
            self.declaration(node)
            return node
        shape = self.shape(node)
        if shape is None:
            self.children(node)
            return node
        kind, matrix = shape[0], shape[1]
        self.matrix_parts(matrix)
        if kind == "matrix":
            return call("transpose", [matrix])
        if kind == "column":
            index = self.expression(shape[2])
            return A.ArrayAccessNode(call("transpose", [matrix]), index)
        row, column = self.expression(shape[2]), self.expression(shape[3])
        if is_pure(row) and is_pure(column):
            return A.ArrayAccessNode(A.ArrayAccessNode(matrix, column), row)
        return A.ArrayAccessNode(
            A.ArrayAccessNode(call("transpose", [matrix]), row), column
        )

    def target_expression(self, node, what):
        """Rewrite ``node`` where it is written (assignment target, ``++``/``--`` operand, ``out`` argument)."""
        shape = self.shape(node)
        if shape is None:
            if isinstance(node, (A.MemberAccessNode, A.SwizzleNode)):
                inner = getattr(node, "object_expr", None) or getattr(
                    node, "vector_expr", None
                )
                inner_shape = self.shape(inner) if inner is not None else None
                if inner_shape is not None and inner_shape[0] == "column":
                    self.refuse(f"{what} of components of a column")
            self.children(node)
            return node
        kind, matrix = shape[0], shape[1]
        if kind == "column":
            self.refuse(f"{what} of a column")
        if kind == "matrix":
            if what != "an increment or decrement":
                self.refuse(f"{what} of a whole matrix")
            self.matrix_parts(matrix)
            return node
        self.matrix_parts(matrix)
        row, column = self.expression(shape[2]), self.expression(shape[3])
        if not (is_pure(row) and is_pure(column)):
            self.refuse(f"{what} through indices with side effects")
        return A.ArrayAccessNode(A.ArrayAccessNode(matrix, column), row)

    def assignment(self, node):
        value = self.expression(node.value)
        shape = self.shape(node.target)
        if shape is not None and shape[0] == "matrix":
            matrix = shape[1]
            self.matrix_parts(matrix)
            if node.operator == "=":
                value = call("transpose", [value])
            else:
                if not is_pure(matrix):
                    self.refuse(
                        "a compound assignment through an object with side effects"
                    )
                operator = node.operator[:-1]
                combined = A.BinaryOpNode(
                    call("transpose", [clone(matrix)]), operator, value
                )
                value = call("transpose", [combined])
                node.operator = "="
        else:
            target = self.target_expression(node.target, "an assignment")
            if target is not node.target:
                node.target = node.left = target
            if id(node.value) in self.literal_types:
                self.literal(value, self.literal_types[id(node.value)])
        node.value = node.right = value
        return node

    # Calls and initializers ---------------------------------------------------------------------------------------
    def written_parameters(self, name):
        function = self.functions.get(name)
        if function is None:
            return set()
        written = set()
        for index, parameter in enumerate(getattr(function, "parameters", None) or []):
            qualifiers = {
                str(q).lower() for q in getattr(parameter, "qualifiers", []) or []
            }
            if qualifiers & {"out", "inout"} or isinstance(
                getattr(parameter, "param_type", None), A.ReferenceType
            ):
                written.add(index)
        return written

    def call(self, node):
        name = call_name(node)
        written = self.written_parameters(name)
        arguments = node.arguments or []
        for index, argument in enumerate(arguments):
            if index in written:
                arguments[index] = self.target_expression(
                    argument, "passing to an out or inout parameter"
                )
            else:
                arguments[index] = self.expression(argument)
        if not isinstance(node.function, A.IdentifierNode):
            node.function = self.expression(node.function)
        if name in self.members and name not in self.functions:
            self.struct_arguments(name, arguments)
        node.name = node.function
        node.args = node.arguments
        return node

    def struct_arguments(self, struct_name, arguments):
        struct = self.struct_nodes[struct_name]
        members = [m for m in struct.members if isinstance(m, A.StructMemberNode)]
        if len(arguments) != len(members):
            return
        for index, member in enumerate(members):
            info = self.members[struct_name].get(member.name)
            if info is None:
                continue
            arguments[index] = self.transposed_value(arguments[index], info.dims)

    def transposed_value(self, value, dims):
        if dims == 0:
            return call("transpose", [value])
        if isinstance(value, A.ArrayLiteralNode):
            value.elements = [
                self.transposed_value(element, dims - 1) for element in value.elements
            ]
            return value
        self.refuse("initializing an array of matrices from an array value")

    def literal(self, value, type_node):
        """Transpose the members of a brace initializer of ``type_node``."""
        if not isinstance(value, A.ArrayLiteralNode):
            return
        if isinstance(type_node, A.ArrayType):
            for element in value.elements:
                self.literal(element, type_node.element_type)
            return
        struct_name = _type_name(type_node)
        if struct_name not in self.struct_nodes:
            return
        members = [
            m
            for m in self.struct_nodes[struct_name].members
            if isinstance(m, A.StructMemberNode)
        ]
        if len(value.elements) != len(members):
            if struct_name in self.members:
                self.refuse(
                    "a brace initializer whose elements do not match the members"
                )
            return
        for index, member in enumerate(members):
            info = self.members.get(struct_name, {}).get(member.name)
            if info is not None:
                value.elements[index] = self.transposed_value(
                    value.elements[index], info.dims
                )
            else:
                self.literal(value.elements[index], member.member_type)

    def declaration(self, node):
        if node.initial_value is None:
            return
        node.initial_value = self.expression(node.initial_value)
        self.literal(node.initial_value, node.var_type)


def store_row_major_members_transposed(program, target) -> int:
    """Store row-major struct members transposed for ``target``; returns how many members changed."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode):
        return 0
    if target in TARGETS:
        only = None
    elif target in BLOCK_ONLY_TARGETS:
        only = _block_type_names(program)
        if not only:
            return 0
    else:
        return 0
    structs = _program_structs(program)
    members = _collect_members(program, only)
    if not members:
        return 0
    walker = _Sites(program, members, target)
    walker.run_program()
    functions = {}
    for function in iter_functions(program):
        functions.setdefault(function.name, function)
    rewriter = _Rewriter(
        walker.sites, members, structs, functions, walker.literal_types, target
    )
    for function in iter_functions(program):
        if getattr(function, "body", None) is not None:
            rewriter.children(function.body)
    for declaration in _module_declarations(program):
        rewriter.declaration(declaration)
    return sum(len(table) for table in members.values())


def _module_declarations(program):
    for name in ("global_variables", "constants"):
        for node in getattr(program, name, None) or []:
            if isinstance(node, A.VariableNode):
                yield node
    for stage in program_stages(program):
        for node in getattr(stage, "local_variables", None) or []:
            if isinstance(node, A.VariableNode):
                yield node
