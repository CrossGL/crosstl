"""HLSL typed buffers written in CrossGL, as Metal texture buffers.

``Buffer<T>``, ``RWBuffer<T>`` and ``RasterizerOrderedBuffer<T>`` are HLSL's typed buffers: one-dimensional arrays of
formatted texels whose view format converts each element to ``T``, a scalar or vector of one to four components
(HLSL reference, "Buffer<T>" and "RWBuffer<T>"). They are the same resources as CrossGL's texel buffers,
``samplerBuffer`` and ``imageBuffer``, which the HLSL importer writes for them, but a CrossGL program may also name
them with the HLSL spelling, which the HLSL generator keeps. Metal output wrote such a declaration as a program-scope
``constant Buffer<float> values = Buffer<float>(0);``, which the Metal compiler rejects (``no template named
'Buffer'``, issue #3642).

:func:`lower_metal_typed_buffers` rewrites them into the texel buffers that the Metal generator already writes as
``texture_buffer<T, access>`` (Metal Shading Language Specification, "Texture buffers"):

- ``Buffer<T>`` becomes ``samplerBuffer``, and ``RWBuffer<T>`` and ``RasterizerOrderedBuffer<T>`` become
  ``imageBuffer``, with the ``i``/``u`` prefix of ``T``'s scalar family; a rasterizer-ordered buffer also gets
  ``@rasterizer_ordered``, which places it in raster order group 0;
- an element read ``b[i]`` becomes ``texelFetch(b, i)`` or ``imageLoad(b, i)`` narrowed to ``T``'s components, as
  HLSL narrows the four-component texel, and converted to ``T`` when ``T`` has 16-bit components;
- a store ``b[i] = v`` becomes ``imageStore(b, i, v')``, with ``v`` converted to ``T`` and widened to four
  components with zeros; a component store, compound assignment or increment of an element reads the element
  into a local of type ``T``, applies the statement to it and stores it back;
- an atomic on an element, ``atomicAdd(b[i], v)`` and the other value-returning atomics, becomes the image atomic
  ``imageAtomicAdd(b, i, v)``;
- ``buffer_dimensions(b)`` becomes the element count ``uint(textureSize(b))`` (``imageSize`` for a writable buffer),
  and the statement ``buffer_dimensions(b, n)`` assigns it to ``n``.

The texel-buffer calls (``texelFetch``, ``imageStore`` and the others) and program functions that take a typed buffer
keep it as their argument. Everything else is reported through ``refuse(reason)``: a typed buffer of 64-bit,
boolean or aggregate elements (Metal texture buffers hold 16- and 32-bit integer and floating-point texels), a store
to a ``Buffer<T>``, a read-modify-write whose buffer or index has side effects, a store or increment inside a larger
expression, an element passed to an ``out``/``inout`` parameter, a typed buffer in a structure or a return type, and
any other use of the buffer.

GLSL output applies the same rewrite (``target="GLSL"``): its ``samplerBuffer`` and ``imageBuffer`` texel buffers
hold the same texels and take the same ``texelFetch``, ``imageLoad``, ``imageStore`` and image atomic calls.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..semantics.hub_types import TypeScope, type_from_node
from ..semantics.rewrite import child_slots, clone, get_child, is_pure, set_child

__all__ = ["TYPED_BUFFER_KINDS", "lower_metal_typed_buffers"]

# HLSL typed buffer -> (CrossGL texel buffer stem, writable, rasterizer ordered).
TYPED_BUFFER_KINDS = {
    "Buffer": ("samplerBuffer", False, False),
    "RWBuffer": ("imageBuffer", True, False),
    "RasterizerOrderedBuffer": ("imageBuffer", True, True),
}
_FAMILY_PREFIX = {"f": "", "i": "i", "u": "u"}
_SCALAR_NAMES = {"f": "float", "i": "int", "u": "uint"}
_VECTOR_PREFIX = {"f": "", "i": "i", "u": "u"}
_ATOMICS = {
    "atomicAdd": "imageAtomicAdd",
    "atomicMin": "imageAtomicMin",
    "atomicMax": "imageAtomicMax",
    "atomicAnd": "imageAtomicAnd",
    "atomicOr": "imageAtomicOr",
    "atomicXor": "imageAtomicXor",
    "atomicExchange": "imageAtomicExchange",
    "atomicCompSwap": "imageAtomicCompSwap",
}
_TEXEL_CALLS = frozenset(
    {
        "texelFetch",
        "textureSize",
        "imageLoad",
        "imageStore",
        "imageSize",
        *_ATOMICS.values(),
    }
)
_SWIZZLES = {1: "x", 2: "xy", 3: "xyz"}
# What each output language calls the texel buffers a typed buffer becomes; ``texel buffers`` otherwise.
_TEXEL_BUFFER_NOUNS = {"Metal": "texture buffers"}
_SCOPES = (A.BlockNode, A.ForNode, A.ForInNode, A.FunctionNode)
_OUT_QUALIFIERS = frozenset({"out", "inout"})


def lower_metal_typed_buffers(program, refuse, target="Metal"):
    """A copy of ``program`` whose typed buffers are texel buffers; ``program`` itself when it declares none.

    ``refuse(reason)`` receives one reason per construct that has no faithful lowering. ``target`` names the output
    language in those reasons: GLSL output applies the same rewrite to reach its ``samplerBuffer`` and
    ``imageBuffer`` texel buffers, which hold the same 16- and 32-bit texels.
    """
    if not isinstance(program, A.ShaderNode) or not _has_typed_buffers(program):
        return program
    lowered = copy.deepcopy(program)
    return _Lowerer(lowered, refuse or (lambda _reason: None), target).run()


def _typed_buffer(type_node):
    """``(kind, element)`` of the typed buffer a type names, or ``None``; arrays of typed buffers are not unwrapped."""
    if (
        isinstance(type_node, A.NamedType)
        and type_node.name in TYPED_BUFFER_KINDS
        and len(type_node.generic_args or []) == 1
    ):
        return type_node.name, type_node.generic_args[0]
    return None


def _element_type(type_node):
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    return type_node


def _texel_type(element):
    """The ``HubType`` of a typed-buffer element Metal can hold as a texel, or ``None``."""
    value = type_from_node(element)
    if value is None or value.kind not in _FAMILY_PREFIX:
        return None
    if value.bits not in (16, 32) or not 1 <= value.lanes <= 4:
        return None
    return value


def _vector_name(kind, lanes):
    if lanes == 1:
        return _SCALAR_NAMES[kind]
    return f"{_VECTOR_PREFIX[kind]}vec{lanes}"


def _zero(kind):
    if kind == "f":
        return A.LiteralNode(0.0, A.PrimitiveType("float"))
    return A.LiteralNode(0, A.PrimitiveType(_SCALAR_NAMES[kind]))


def _one(kind):
    if kind == "f":
        return A.LiteralNode(1.0, A.PrimitiveType("float"))
    return A.LiteralNode(1, A.PrimitiveType(_SCALAR_NAMES[kind]))


def _call(name, arguments):
    return A.FunctionCallNode(A.IdentifierNode(name), list(arguments))


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, A.IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


def _declared_type(node):
    for attribute in ("var_type", "param_type", "member_type", "vtype"):
        value = getattr(node, attribute, None)
        if value is not None:
            return value
    return None


def _program_functions(program):
    functions = list(getattr(program, "functions", None) or [])
    for stage in (getattr(program, "stages", None) or {}).values():
        functions.extend(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
    return functions


def _declarations(program):
    """Every variable, parameter and structure member of the program."""
    stack, seen = [program], set()
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if isinstance(node, A.StageMap):
            stack.extend(node.values())
            continue
        if not isinstance(node, A.ASTNode) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (A.VariableNode, A.ParameterNode, A.StructMemberNode)):
            yield node
        for name, value in vars(node).items():
            if name in ("parent", "annotations", "source_location"):
                continue
            if isinstance(value, (A.ASTNode, list, A.StageMap)):
                stack.append(value)


def _has_typed_buffers(program):
    """Whether a global, stage variable, parameter or structure member is (an array of) a typed buffer."""
    candidates = list(getattr(program, "global_variables", None) or [])
    for struct in getattr(program, "structs", None) or []:
        candidates.extend(getattr(struct, "members", None) or [])
    for stage in (getattr(program, "stages", None) or {}).values():
        candidates.extend(getattr(stage, "local_variables", None) or [])
    for function in _program_functions(program):
        candidates.extend(getattr(function, "parameters", None) or [])
    return any(
        _typed_buffer(_element_type(_declared_type(node))) is not None
        for node in candidates
    )


def _texel_buffer_type(type_node, texel):
    """``type_node`` with its typed buffer replaced by the CrossGL texel buffer of the same family."""
    if isinstance(type_node, A.ArrayType):
        arrayed = copy.copy(type_node)
        arrayed.element_type = _texel_buffer_type(type_node.element_type, texel)
        return arrayed
    stem = TYPED_BUFFER_KINDS[type_node.name][0]
    return A.NamedType(f"{_FAMILY_PREFIX[texel.kind]}{stem}")


class _Lowerer:
    """One walk over the program with lexical scopes, rewriting each typed-buffer access in place."""

    def __init__(self, program, refuse, target="Metal"):
        self.program = program
        self.refuse = refuse
        self.target = target
        self.scope = TypeScope(program)
        self.functions = {
            getattr(function, "name", None): function
            for function in _program_functions(program)
        }
        self.used_names = {
            node.name
            for node in _declarations(program)
            if isinstance(getattr(node, "name", None), str)
        }

    def fresh_name(self, stem):
        name, suffix = stem, 0
        while name in self.used_names:
            suffix += 1
            name = f"{stem}_{suffix}"
        self.used_names.add(name)
        return name

    # Types ------------------------------------------------------------------------------------------------------
    def buffer(self, expr):
        """``(kind, element, texel)`` for an expression that denotes a typed buffer (a name or an indexed name)."""
        if not isinstance(expr, (A.IdentifierNode, A.ArrayAccessNode)):
            return None
        typed = _typed_buffer(self.scope.node_type(expr))
        if typed is None:
            return None
        return (*typed, _texel_type(typed[1]))

    def element_access(self, expr):
        """``(buffer expression, index, kind, element, texel)`` for an element access ``b[i]`` of a typed buffer."""
        if not isinstance(expr, A.ArrayAccessNode):
            return None
        typed = self.buffer(expr.array_expr)
        if typed is None:
            return None
        return (expr.array_expr, expr.index_expr, *typed)

    def usable(self, access, writes=False):
        """Whether ``access`` can be lowered, refusing it otherwise."""
        buffer, _index, kind, element, texel = access
        if texel is None:
            # Its declaration is refused; the access stays as it is.
            return False
        if writes and not TYPED_BUFFER_KINDS[kind][1]:
            self.refuse(
                f"store to the read-only {kind}<{_type_text(element)}> '{_buffer_name(buffer)}'"
            )
            return False
        return True

    def converted(self, value, kind, lanes, exact=False):
        """``value`` as a 32-bit ``kind`` vector of ``lanes`` components, the texel a store writes.

        ``exact`` says that ``value`` already has that type.
        """
        inferred = self.scope.infer(value)
        if exact or (
            inferred is not None
            and inferred.kind == kind
            and inferred.bits == 32
            and inferred.lanes == lanes
        ):
            return value
        return _call(_vector_name(kind, lanes), [value])

    # Accesses ---------------------------------------------------------------------------------------------------
    def load(self, access):
        buffer, index, kind, element, texel = access
        read = "imageLoad" if TYPED_BUFFER_KINDS[kind][1] else "texelFetch"
        value = _call(read, [clone(buffer), index])
        if texel.lanes in _SWIZZLES:
            value = A.MemberAccessNode(value, _SWIZZLES[texel.lanes])
        if texel.bits != 32:
            value = _call(_type_text(element), [value])
        return value

    def store(self, access, value, exact=False):
        buffer, index, _kind, _element, texel = access
        texel_value = self.converted(value, texel.kind, texel.lanes, exact)
        if texel.lanes < 4:
            zeros = [_zero(texel.kind) for _ in range(4 - texel.lanes)]
            texel_value = _call(_vector_name(texel.kind, 4), [texel_value, *zeros])
        return A.ExpressionStatementNode(
            _call("imageStore", [clone(buffer), clone(index), texel_value])
        )

    def size(self, buffer, typed):
        kind = typed[0]
        query = "imageSize" if TYPED_BUFFER_KINDS[kind][1] else "textureSize"
        return _call("uint", [_call(query, [clone(buffer)])])

    def modified_access(self, statement):
        """The typed-buffer element a statement modifies, with the slot of the modified expression, or ``None``."""
        expression = statement.expression
        if isinstance(expression, A.AssignmentNode):
            holder, field = expression, "target"
        elif isinstance(expression, A.UnaryOpNode) and expression.op in ("++", "--"):
            holder, field = expression, "operand"
        else:
            return None
        target = getattr(holder, field)
        while True:
            access = self.element_access(target)
            if access is not None:
                return access, holder, field
            if isinstance(target, A.MemberAccessNode):
                holder, field, target = target, "object_expr", target.object_expr
            elif isinstance(target, A.SwizzleNode):
                holder, field, target = target, "vector_expr", target.vector_expr
            else:
                return None

    def statement_replacement(self, node):
        """The lowered statement for a store, read-modify-write or size query used as a statement, else ``None``.

        ``False`` marks a statement the pass refused.
        """
        if not isinstance(node, A.ExpressionStatementNode):
            return None
        expression = node.expression
        if (
            isinstance(expression, A.FunctionCallNode)
            and _call_name(expression) == "buffer_dimensions"
            and len(expression.arguments or []) == 2
        ):
            typed = self.buffer(expression.arguments[0])
            if typed is None:
                return None
            receiver = expression.arguments[1]
            return A.ExpressionStatementNode(
                A.AssignmentNode(receiver, self.size(expression.arguments[0], typed))
            )
        modified = self.modified_access(node)
        if modified is None:
            return None
        access, holder, field = modified
        if not self.usable(access, writes=True):
            return False
        if (
            isinstance(expression, A.AssignmentNode)
            and holder is expression
            and expression.operator == "="
        ):
            return self.store(access, expression.value)
        buffer, index, _kind, element, _texel = access
        if not (is_pure(buffer) and is_pure(index)):
            self.refuse(
                f"read-modify-write of an element of '{_buffer_name(buffer)}' whose buffer or index has side effects"
            )
            return False
        name = self.fresh_name("crosstl_texel")
        local = A.VariableNode(name, copy.deepcopy(element), self.load(access))
        set_child(holder, field, None, A.IdentifierNode(name))
        store = self.store(
            (clone(buffer), clone(index), *access[2:]),
            A.IdentifierNode(name),
            exact=access[4].bits == 32,
        )
        return A.BlockNode([local, node, store])

    def expression_replacement(self, node):
        """The lowered expression for an element read, element atomic or size query, else ``None``."""
        if isinstance(node, A.ArrayAccessNode):
            access = self.element_access(node)
            if access is None:
                return None
            return self.load(access) if self.usable(access) else False
        if not isinstance(node, A.FunctionCallNode):
            return None
        name = _call_name(node)
        args = list(node.arguments or [])
        if name in _ATOMICS and args:
            access = self.element_access(args[0])
            if access is None:
                return None
            if not self.usable(access, writes=True):
                return False
            buffer, index = access[0], access[1]
            return _call(_ATOMICS[name], [clone(buffer), index, *args[1:]])
        if name == "buffer_dimensions" and len(args) == 1:
            typed = self.buffer(args[0])
            return None if typed is None else self.size(args[0], typed)
        return None

    def check_modification(self, node):
        """Refuse a store or increment of a typed-buffer element that is not a statement of its own."""
        if isinstance(node, A.AssignmentNode):
            target = node.target
        elif isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
            target = node.operand
        else:
            return
        while isinstance(target, (A.MemberAccessNode, A.SwizzleNode)):
            target = getattr(target, "object_expr", None) or target.vector_expr
        access = self.element_access(target)
        if access is not None:
            self.refuse(
                f"store to an element of '{_buffer_name(access[0])}' inside a larger expression"
            )

    def check_call_arguments(self, node):
        """Typed buffers go to texel-buffer calls and program functions; elements go to no ``out`` parameter."""
        name = _call_name(node)
        function = self.functions.get(name)
        parameters = list(getattr(function, "parameters", None) or [])
        for position, argument in enumerate(node.arguments or []):
            if self.buffer(argument) is not None:
                if name not in _TEXEL_CALLS and function is None:
                    self.refuse(
                        f"typed buffer '{_buffer_name(argument)}' passed to '{name}', which has no texel-buffer "
                        f"form in {self.target}"
                    )
                continue
            if self.element_access(argument) is None or position >= len(parameters):
                continue
            qualifiers = set(getattr(parameters[position], "qualifiers", None) or [])
            if qualifiers & _OUT_QUALIFIERS:
                self.refuse(
                    f"element of '{_buffer_name(argument.array_expr)}' passed to the {'/'.join(sorted(qualifiers & _OUT_QUALIFIERS))} "
                    f"parameter '{parameters[position].name}' of '{name}'"
                )

    # Traversal --------------------------------------------------------------------------------------------------
    def visit(self, node):
        scoped = isinstance(node, _SCOPES)
        if scoped:
            self.scope.push()
        if isinstance(node, A.FunctionNode):
            for parameter in node.parameters or []:
                self.scope.declare(parameter.name, parameter.param_type)
        if isinstance(node, A.FunctionCallNode):
            self.check_call_arguments(node)
        self.check_modification(node)
        if isinstance(node, A.IdentifierNode) and self.buffer(node) is not None:
            self.refuse(
                f"typed buffer '{node.name}' used other than through an element access, a texel-buffer call, "
                "buffer_dimensions or a program function argument"
            )
        self.visit_children(node)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        if scoped:
            self.scope.pop()

    def visit_children(self, node, fields=None):
        buffer_arguments = set()
        if isinstance(node, A.FunctionCallNode):
            buffer_arguments = {
                id(argument)
                for argument in node.arguments or []
                if self.buffer(argument) is not None
            }
        if isinstance(node, A.ArrayAccessNode) and self.buffer(node.array_expr):
            buffer_arguments.add(id(node.array_expr))
        for field, key in list(child_slots(node)):
            if fields is not None and field not in fields:
                continue
            if field in ("parameters", "return_type") and isinstance(
                node, A.FunctionNode
            ):
                continue
            if isinstance(node, A.FunctionCallNode) and field in ("function", "name"):
                continue
            child = get_child(node, field, key)
            replacement = self.statement_replacement(child)
            if replacement is None:
                replacement = self.expression_replacement(child)
            if replacement is not None and replacement is not False:
                set_child(node, field, key, replacement)
                child = replacement
            if id(child) in buffer_arguments:
                self.visit_buffer_reference(child)
                continue
            if replacement is False:
                # Refused once; its operands are not reported again as bare uses of the buffer.
                continue
            self.visit(child)

    def visit_buffer_reference(self, expr):
        """Visit the indices of a buffer reference such as ``buffers[k]`` without treating the name as a use."""
        if isinstance(expr, A.ArrayAccessNode):
            self.visit_buffer_reference(expr.array_expr)
            self.visit_children(expr, fields=("index_expr",))

    # Program ----------------------------------------------------------------------------------------------------
    def run(self):
        program = self.program
        for constant in getattr(program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
        for cbuffer in getattr(program, "cbuffers", None) or []:
            self.scope.declare_cbuffer(cbuffer)
        for variable in getattr(program, "global_variables", None) or []:
            self.scope.declare(variable.name, variable.var_type)
        for function in getattr(program, "functions", None) or []:
            self.visit(function)
        for stage in (getattr(program, "stages", None) or {}).values():
            self.scope.push()
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                self.scope.declare_cbuffer(cbuffer)
            for variable in getattr(stage, "local_variables", None) or []:
                self.scope.declare(variable.name, variable.var_type)
            for function in getattr(stage, "local_functions", None) or []:
                self.visit(function)
            if getattr(stage, "entry_point", None) is not None:
                self.visit(stage.entry_point)
            self.scope.pop()
        self.rewrite_declarations()
        return program

    def rewrite_declarations(self):
        for node in list(_declarations(self.program)):
            declared = _declared_type(node)
            typed = _typed_buffer(_element_type(declared))
            if typed is None:
                continue
            kind, element = typed
            name = getattr(node, "name", None)
            if isinstance(node, A.StructMemberNode):
                self.refuse(f"{kind} '{name}' as a structure member")
                continue
            texel = _texel_type(element)
            if texel is None:
                self.refuse(
                    f"{kind}<{_type_text(element)}> '{name}': {self.target} {_TEXEL_BUFFER_NOUNS.get(self.target, 'texel buffers')} "
                    "hold texels of one to four 16- or 32-bit integer or floating-point components"
                )
                continue
            replacement = _texel_buffer_type(declared, texel)
            for attribute in ("var_type", "param_type", "vtype"):
                if getattr(node, attribute, None) is declared:
                    setattr(node, attribute, replacement)
            if TYPED_BUFFER_KINDS[kind][2]:
                node.attributes = [
                    *(getattr(node, "attributes", None) or []),
                    A.AttributeNode("rasterizer_ordered"),
                ]
        for function in _program_functions(self.program):
            returned = _typed_buffer(
                _element_type(getattr(function, "return_type", None))
            )
            if returned is not None:
                self.refuse(
                    f"function '{function.name}' returns a {returned[0]}, which {self.target} cannot return"
                )


def _buffer_name(expr):
    while isinstance(expr, A.ArrayAccessNode):
        expr = expr.array_expr
    return getattr(expr, "name", "?")


def _type_text(type_node):
    if isinstance(type_node, A.VectorType):
        return f"{_type_text(type_node.element_type)}{type_node.size}"
    return str(getattr(type_node, "name", type_node))
