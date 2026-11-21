"""Union members that word storage carries element by element: structures, narrow vectors and partial members.

HLSL, GLSL and SPIR-V have no unions. Their generators keep a source union that carries ``@union_layout`` as one
array of 32-bit words and read each member through a *word view* (:mod:`..codegen.union_layout`): a 32-bit scalar
or vector that covers every word, a ``u8vec4``, or a fixed array of 8- or 16-bit lanes that covers every byte.
Other members are refused, although Metal, CUDA and HIP sources write them routinely:

* a member of structure type, ``union { struct { half d; half m; }; half2 dm; }`` (``block_q4_1`` in ggml);
* a vector of 8- or 16-bit elements, ``half2 dm`` in a four-byte union;
* a vector or scalar that covers part of the storage: ``float3 center`` beside ``float4 data`` (the fourth word
  is the vector's padding in Metal), or ``ushort lo`` beside ``uint word``.

:func:`split_union_word_views` rewrites each such member as one lane-array view per element type, each covering
the whole storage, and rewrites every access to name the element it reads or writes: ``u.dm.x`` is
``u.dm[0]`` on the view ``float16 dm[2]``, ``u.s.m`` is ``u.s[1]``, and a whole read ``u.dm`` is
``f16vec2(u.dm[0], u.dm[1])``. A whole write in statement position becomes one write per element, through a
temporary when the value could read the storage it overwrites. The element at byte offset ``o`` of type ``T`` is
lane ``o / sizeof(T)`` of the view, which is the little-endian byte order ``@union_layout`` requires, so every
access reads and writes exactly the bytes it does in the source.

A structure member is split only when its fields are 8-, 16- or 32-bit scalars whose natural layout has the size
and alignment ``@union_member_layout`` records; any other member keeps its type and the generator's refusal. An
access the rewrite cannot express exactly (a pointer to the member, an increment, a whole write inside an
expression, or a member whose union cannot be determined) raises :class:`UnionWordViewError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import _struct_name, element_type_node, type_from_name
from .rewrite import child_slots, clone, get_child, is_pure, set_child
from .scoped_walk import ScopedWalker

__all__ = ["UnionWordViewError", "split_union_word_views"]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
# The generators that carry a union as canonical 32-bit words.
_TARGETS = (
    frozenset({"opengl", "glsl", "webgl", "gles", "essl", "vulkan", "spirv"})
    | _HLSL_TARGETS
)

# (kind, bits) -> the CrossGL element spelling of a lane view and of a vector of it.
_ELEMENTS = {
    ("f", 16): ("float16", "f16vec"),
    ("i", 16): ("int16", "i16vec"),
    ("u", 16): ("uint16", "u16vec"),
    ("i", 8): ("int8", "i8vec"),
    ("u", 8): ("uint8", "u8vec"),
    ("f", 32): ("float", "vec"),
    ("i", 32): ("int", "ivec"),
    ("u", 32): ("uint", "uvec"),
}
_NARROW_VECTOR = re.compile(r"([iu])(8|16)vec([234])")
_SWIZZLES = ("xyzw", "rgba", "stpq")
_SWIZZLE = re.compile(r"[xyzw]{1,4}|[rgba]{1,4}|[stpq]{1,4}")
_BUFFER_READS = frozenset({"buffer_load"})


class UnionWordViewError(TranslationDiagnostic):
    """An access to a union member that word storage cannot carry exactly."""

    def __init__(self, target, union, member, detail):
        self.union_name = union
        self.member_name = member
        super().__init__(
            f"union '{union}' member '{member}' cannot be carried in word storage: {detail}",
            target=target,
            code=f"{CODE_PREFIX}union-word-view-unsupported",
        )


@dataclass(frozen=True)
class _Element:
    view: str
    index: int | None  # None: a scalar view that is the whole single-word storage
    spelling: str


@dataclass
class _Plan:
    """How one decomposed member maps onto views: ``elements`` in constructor order, keyed by component or field."""

    union: str
    member: str
    shape: str  # "vector", "struct" or "scalar"
    constructor: str
    vector_prefix: str = ""
    elements: dict = field(default_factory=dict)
    first: bool = False


def _int(node):
    value = getattr(node, "value", node)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).rstrip("uU")
    return int(text) if text.isdigit() else None


def _attribute(node, name):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() == name:
            return attribute
    return None


def _layout(node, name, count):
    attribute = _attribute(node, name)
    arguments = list(getattr(attribute, "arguments", None) or [])
    if attribute is None or len(arguments) < count:
        return None
    return [_int(argument) for argument in arguments]


def _type_name(type_node):
    if isinstance(type_node, (A.NamedType, A.PrimitiveType)) and not getattr(
        type_node, "generic_args", None
    ):
        return str(type_node.name)
    if isinstance(type_node, A.VectorType):
        element = _type_name(type_node.element_type)
        return f"{element}{type_node.size}" if element else None
    return None


def _value_shape(type_node):
    """``(kind, bits, lanes)`` of an 8-, 16- or 32-bit integer or float scalar or vector type, else ``None``."""
    name = _type_name(type_node)
    if name is None:
        return None
    if name == "bfloat16":
        return None
    match = _NARROW_VECTOR.fullmatch(name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    hub = type_from_name(name)
    if hub is None or (hub.kind, hub.bits) not in _ELEMENTS:
        return None
    return hub.kind, hub.bits, hub.lanes


class _Unions:
    """The decomposition plans of the program's ``@union_layout`` structures."""

    def __init__(self, program):
        self.structs = {
            node.name: node
            for node in getattr(program, "structs", None) or []
            if isinstance(node, A.StructNode) and node.name
        }
        self.owners = {}
        for struct in self.structs.values():
            for member in struct.members or []:
                if getattr(member, "name", None):
                    self.owners.setdefault(member.name, set()).add(struct.name)
        self.plans = {}
        for struct in self.structs.values():
            self._plan_union(struct)

    def _plan_union(self, struct):
        layout = _layout(struct, "union_layout", 4)
        if layout is None or layout[0] not in (4, 8, 16) or layout[1] != layout[0]:
            return
        size = layout[0]
        names = {member.name for member in struct.members if member.name}
        members = []
        changed = False
        for position, member in enumerate(struct.members):
            plan = self._plan_member(struct.name, member, size, names)
            if plan is None:
                members.append(member)
                continue
            changed = True
            plan.first = position == 0
            self.plans[(struct.name, member.name)] = plan
            members.extend(self._views(plan, member, size))
        if changed:
            struct.members = members

    def _plan_member(self, union, member, size, names):
        if not isinstance(member, A.StructMemberNode) or not member.name:
            return None
        layout = _layout(member, "union_member_layout", 3)
        if layout is None or None in layout[:3]:
            return None
        offset, member_size, alignment = layout[:3]
        if offset < 0 or member_size <= 0 or offset + member_size > size:
            return None
        nested = self.structs.get(_type_name(member.member_type) or "")
        if nested is not None:
            return self._plan_struct(union, member, nested, offset, member_size, alignment, size, names)
        shape = _value_shape(member.member_type)
        if shape is None:
            return None
        kind, bits, lanes = shape
        width = bits // 8
        whole = offset == 0 and member_size == size
        if bits == 32 and whole and lanes == size // 4:
            return None  # a 32-bit scalar or vector word view
        if (kind, bits, lanes) == ("u", 8, 4) and whole and size == 4:
            return None  # a u8vec4 byte view
        if offset % width or lanes * width > member_size:
            return None
        spelling, prefix = _ELEMENTS[(kind, bits)]
        if lanes == 1:
            plan = _Plan(union, member.name, "scalar", spelling)
            plan.elements[0] = (spelling, offset)
            return self._name_views(plan, size, names)
        plan = _Plan(union, member.name, "vector", f"{prefix}{lanes}", prefix)
        for lane in range(lanes):
            plan.elements[lane] = (spelling, offset + lane * width)
        return self._name_views(plan, size, names)

    def _plan_struct(self, union, member, nested, offset, member_size, alignment, size, names):
        if _attribute(nested, "union_layout") is not None:
            return None
        plan = _Plan(union, member.name, "struct", nested.name)
        cursor, widest = 0, 1
        for nested_member in nested.members or []:
            shape = _value_shape(getattr(nested_member, "member_type", None))
            if not isinstance(nested_member, A.StructMemberNode) or shape is None:
                return None
            kind, bits, lanes = shape
            if lanes != 1 or not nested_member.name:
                return None
            width = bits // 8
            cursor = -(-cursor // width) * width
            plan.elements[nested_member.name] = (_ELEMENTS[(kind, bits)][0], offset + cursor)
            cursor += width
            widest = max(widest, width)
        total = -(-cursor // widest) * widest
        if not plan.elements or total != member_size or widest != alignment or offset % widest:
            return None
        return self._name_views(plan, size, names)

    @staticmethod
    def _name_views(plan, size, names):
        spellings = []
        for spelling, _offset in plan.elements.values():
            if spelling not in spellings:
                spellings.append(spelling)
        views = {}
        for spelling in spellings:
            if len(spellings) == 1:
                views[spelling] = plan.member
                continue
            candidate, suffix = f"{plan.member}_{spelling}", 2
            while candidate in names:
                candidate, suffix = f"{plan.member}_{spelling}{suffix}", suffix + 1
            names.add(candidate)
            views[spelling] = candidate
        for key, (spelling, offset) in list(plan.elements.items()):
            width = 4 if spelling in ("float", "int", "uint") else int(re.sub(r"\D", "", spelling)) // 8
            index = None if width == 4 and size == 4 else offset // width
            plan.elements[key] = _Element(views[spelling], index, spelling)
        return plan

    @staticmethod
    def _views(plan, member, size):
        views = []
        seen = set()
        for element in plan.elements.values():
            if element.view in seen:
                continue
            seen.add(element.view)
            width = 4 if element.spelling in ("float", "int", "uint") else int(
                re.sub(r"\D", "", element.spelling)
            ) // 8
            view_type = A.NamedType(element.spelling)
            if element.index is not None:
                view_type = A.ArrayType(view_type, _literal(size // width))
            views.append(
                A.StructMemberNode(
                    element.view,
                    view_type,
                    attributes=[
                        A.AttributeNode(
                            "union_member_layout",
                            [_literal(0), _literal(size), _literal(width)],
                        )
                    ],
                    source_location=getattr(member, "source_location", None),
                )
            )
        return views


def _literal(value):
    return A.LiteralNode(value, A.PrimitiveType("int"))


def _side_effect_free(expr):
    """True when evaluating ``expr`` twice reads the same values and writes nothing (memory reads allowed)."""
    if is_pure(expr):
        return True
    if isinstance(expr, A.UnaryOpNode):
        return expr.op not in ("++", "--") and _side_effect_free(expr.operand)
    if isinstance(expr, A.MemberAccessNode):
        return _side_effect_free(expr.object_expr)
    if isinstance(expr, A.ArrayAccessNode):
        return _side_effect_free(expr.array_expr) and _side_effect_free(expr.index_expr)
    if isinstance(expr, A.FunctionCallNode):
        name = getattr(expr.function, "name", expr.function)
        return name in _BUFFER_READS and all(
            _side_effect_free(argument) for argument in expr.arguments or []
        )
    return False


class _Rewriter(ScopedWalker):
    def __init__(self, program, unions, target):
        super().__init__(program)
        self.unions = unions
        self.target = target
        self.generated = set()
        self.temporaries = 0
        self.changed = 0

    # Resolution ----------------------------------------------------------------------------------------------
    def _struct_of(self, expr):
        """The structure name of ``expr``'s value, following pointers, buffers and buffer reads; ``None`` if unknown."""
        type_node = self._type_of(expr)
        name = _struct_name(type_node) if type_node is not None else None
        return name if name in self.unions.structs else None

    def _type_of(self, expr):
        if isinstance(expr, A.IdentifierNode):
            return self.scope.lookup(expr.name)
        if isinstance(expr, A.UnaryOpNode) and expr.op == "*":
            base = self._type_of(expr.operand)
            return element_type_node(base) if base is not None else None
        if isinstance(expr, A.ArrayAccessNode):
            base = self._type_of(expr.array_expr)
            return element_type_node(base) if base is not None else None
        if isinstance(expr, (A.MemberAccessNode, A.PointerAccessNode)):
            inner = expr.object_expr if isinstance(expr, A.MemberAccessNode) else None
            if inner is None:
                pointer = self._type_of(expr.pointer_expr)
                base = element_type_node(pointer) if pointer is not None else None
            else:
                base = self._type_of(inner)
            struct = self.unions.structs.get(_struct_name(base) or "")
            for member in getattr(struct, "members", None) or []:
                if getattr(member, "name", None) == expr.member:
                    return member.member_type
            return None
        if isinstance(expr, A.FunctionCallNode):
            name = getattr(expr.function, "name", expr.function)
            if name in _BUFFER_READS and expr.arguments:
                base = self._type_of(expr.arguments[0])
                return element_type_node(base) if base is not None else None
        return None

    def _plan(self, expr):
        """The plan of a decomposed member ``expr`` reads (``object.member``), or ``None``."""
        if not isinstance(expr, A.MemberAccessNode) or id(expr) in self.generated:
            return None
        candidates = {
            union
            for union in self.unions.owners.get(expr.member, ())
            if (union, expr.member) in self.unions.plans
        }
        if not candidates:
            return None
        union = self._struct_of(expr.object_expr)
        if union is not None:
            return self.unions.plans.get((union, expr.member))
        owners = self.unions.owners[expr.member]
        if len(owners) == 1 and not _SWIZZLE.fullmatch(expr.member):
            return self.unions.plans[(next(iter(owners)), expr.member)]
        union = sorted(candidates)[0]
        raise UnionWordViewError(
            self.target,
            union,
            expr.member,
            "the union of this access cannot be determined",
        )

    def _refuse(self, plan, detail):
        raise UnionWordViewError(self.target, plan.union, plan.member, detail)

    # Construction --------------------------------------------------------------------------------------------
    def _element(self, base, element):
        view = A.MemberAccessNode(clone(base), element.view)
        self.generated.add(id(view))
        if element.index is None:
            return view
        return A.ArrayAccessNode(view, _literal(element.index))

    def _construct(self, plan, base, spelling, elements):
        if len(elements) == 1:
            return self._element(base, elements[0])
        if not _side_effect_free(base):
            self._refuse(plan, "a whole read evaluates the containing expression more than once")
        return A.FunctionCallNode(
            A.IdentifierNode(spelling), [self._element(base, element) for element in elements]
        )

    def _components(self, plan, text):
        for letters in _SWIZZLES:
            if all(letter in letters for letter in text):
                indices = [letters.index(letter) for letter in text]
                if all(index in plan.elements for index in indices):
                    return indices
        return None

    # Rewriting -----------------------------------------------------------------------------------------------
    def _access(self, expr, role):
        """The replacement of an access through a decomposed member, after rewriting its operands; or ``None``."""
        outer, inner, plan = expr, None, None
        if isinstance(expr, (A.MemberAccessNode, A.SwizzleNode, A.ArrayAccessNode)):
            operand = (
                expr.object_expr
                if isinstance(expr, A.MemberAccessNode)
                else expr.vector_expr if isinstance(expr, A.SwizzleNode) else expr.array_expr
            )
            plan = self._plan(operand)
            if plan is not None:
                inner = operand
        if plan is None:
            plan = self._plan(expr)
            if plan is None:
                return None
            inner, outer = expr, None
        if isinstance(outer, A.ArrayAccessNode) and plan.shape != "vector":
            return None
        self._walk_slot(inner, "object_expr", None)
        base = inner.object_expr
        if outer is None:
            if role != "value":
                self._refuse(plan, f"a whole-member {role} needs one write per element in statement position")
            self.changed += 1
            return self._construct(plan, base, plan.constructor, list(plan.elements.values()))
        if isinstance(outer, A.ArrayAccessNode):
            self._walk_slot(outer, "index_expr", None)
            first = plan.elements[0]
            index = outer.index_expr
            if first.index:
                index = A.BinaryOpNode(index, "+", _literal(first.index))
            view = A.MemberAccessNode(clone(base), first.view)
            self.generated.add(id(view))
            self.changed += 1
            return self._checked(plan, role, A.ArrayAccessNode(view, index))
        text = outer.member if isinstance(outer, A.MemberAccessNode) else outer.components
        if plan.shape == "struct":
            element = plan.elements.get(text)
            if element is None:
                self._refuse(plan, f"field '{text}' is not a field of '{plan.constructor}'")
            self.changed += 1
            return self._checked(plan, role, self._element(base, element))
        indices = self._components(plan, text) if plan.shape == "vector" else None
        if indices is None:
            self._refuse(plan, f"'.{text}' does not select components of '{plan.constructor}'")
        if len(indices) > 1 and role != "value":
            self._refuse(plan, f"a {role} of the swizzle '.{text}' is not a single element")
        self.changed += 1
        elements = [plan.elements[index] for index in indices]
        spelling = f"{plan.vector_prefix}{len(elements)}"
        return self._checked(plan, role, self._construct(plan, base, spelling, elements))

    def _checked(self, plan, role, replacement):
        if role in ("address", "update"):
            self._refuse(plan, f"an element of the member has no {role} in word storage")
        return replacement

    @staticmethod
    def _role(parent, field_name):
        if isinstance(parent, A.AssignmentNode) and field_name in ("target", "left"):
            return "write"
        if isinstance(parent, A.UnaryOpNode):
            if parent.op == "&":
                return "address"
            if parent.op in ("++", "--"):
                return "update"
        return "value"

    def _whole_write(self, statement):
        """One write per element for ``object.member = value;``, in a block; ``None`` for other statements."""
        assignment = statement
        if isinstance(statement, A.ExpressionStatementNode):
            assignment = statement.expression
        if not isinstance(assignment, A.AssignmentNode) or assignment.operator != "=":
            return None
        plan = self._plan(assignment.target)
        if plan is None:
            return None
        inner = assignment.target
        self._walk_slot(inner, "object_expr", None)
        self._walk_slot(assignment, "value", None)
        base, value = inner.object_expr, assignment.value
        if not _side_effect_free(base):
            self._refuse(plan, "a whole write evaluates the containing expression more than once")
        statements = []
        if not isinstance(value, (A.IdentifierNode, A.LiteralNode)):
            self.temporaries += 1
            name = f"crossgl_union_value{self.temporaries}"
            while self.scope.is_declared(name):
                self.temporaries += 1
                name = f"crossgl_union_value{self.temporaries}"
            statements.append(
                A.VariableNode(name, A.NamedType(plan.constructor), initial_value=value)
            )
            value = A.IdentifierNode(name)
        for key, element in plan.elements.items():
            if plan.shape == "scalar":
                source = clone(value)
            else:
                component = key if plan.shape == "struct" else "xyzw"[key]
                source = A.MemberAccessNode(clone(value), component)
            target = self._element(base, element)
            statements.append(A.ExpressionStatementNode(A.AssignmentNode(target, source, "=")))
        self.changed += 1
        return A.BlockNode(statements)

    def _refuse_union_initializers(self, node):
        if isinstance(node, (A.FunctionCallNode, A.ConstructorNode)):
            callee = (
                getattr(node.function, "name", node.function)
                if isinstance(node, A.FunctionCallNode)
                else _type_name(node.constructor_type)
            )
            if node.arguments:
                for (union, _member), plan in self.unions.plans.items():
                    if union == callee and plan.first:
                        self._refuse(plan, "a union initializer initializes the first member, which is split")

    def _walk_slot(self, parent, field_name, key):
        child = get_child(parent, field_name, key)
        if child is None or id(child) in self.generated:
            return
        replacement = self._access(child, self._role(parent, field_name))
        if replacement is not None:
            set_child(parent, field_name, key, replacement)
            return
        self._walk(child)

    def _walk(self, node):
        if node is None or isinstance(node, (A.FunctionNode, A.StructNode)):
            return
        scoped = isinstance(node, (A.BlockNode, A.ForNode))
        if scoped:
            self.scope.push()
        self._refuse_union_initializers(node)
        for field_name, key in list(child_slots(node)):
            if isinstance(node, A.BlockNode) and field_name == "statements":
                split = self._whole_write(get_child(node, field_name, key))
                if split is not None:
                    set_child(node, field_name, key, split)
                    continue
            self._walk_slot(node, field_name, key)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        if scoped:
            self.scope.pop()


def split_union_word_views(program, target) -> int:
    """Split the union members word storage cannot carry into lane views for ``target``; returns the rewrites."""
    if str(target or "").lower() not in _TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    if not any(
        _attribute(node, "union_layout") is not None
        for node in getattr(program, "structs", None) or []
    ):
        return 0
    unions = _Unions(program)
    if not unions.plans:
        return 0
    rewriter = _Rewriter(program, unions, str(target))
    rewriter.run_program()
    return rewriter.changed + len(unions.plans)
