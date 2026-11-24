"""Overlapping union storage for the SPIR-V, WGSL and Slang generators.

A union reaches CrossGL as a structure marked ``@union_layout(size, alignment, byte_order, abi)`` whose members carry
``@union_member_layout(offset, size, alignment)``; the CUDA and Metal importers write it. SPIR-V, WGSL and Slang
structures have no overlapping members, so a plain structure gives each member storage of its own, and a read of
one member after a write of another returns a value the source never stored. :func:`lower_union_storage` rewrites
a copy of the program before generation, as the GLSL and HLSL generators lower the same contract
(:mod:`.glsl_union_storage`):

- the structure keeps one member, ``CrossGLUnionStorage``, a ``uint``, ``uvec2`` or ``uvec4`` that the
  ``@union_layout`` size selects (4, 8 or 16 bytes, aligned to its size);
- a read of a member becomes the storage's bits as the member's type (``uintBitsToFloat``, ``int``/``ivecN``, or
  the words themselves), and a write stores the value's bits; a compound write reads, combines and stores;
- an element of a word-array view (``float a[2]`` in eight bytes) reads and writes one component of the storage;
- a ``u8vec4`` byte view (``uchar4 bytes[2]``) reads each vector as the four bytes of one word, and a ``u16vec2``
  16-bit lane view (``ushort2 halves[2]``) reads each vector as the two 16-bit lanes of one word; a write of the
  vector or of one lane stores into that word (:mod:`.spirv_union_bytes`);
- a constructor initializes the first member, as a C union initializer does, and a brace initializer is one.

:class:`.union_layout.UnionLayoutCollector` checks the contract for every generator. A view this pass cannot express
exactly raises :class:`UnionLayoutError`, which names the union, the member and the reason: a sub-word lane view or a
signed or Boolean byte view in use, a byte or 16-bit lane write that cannot be packed in one evaluation, a
whole-array view used as a value, the address of a view or a view passed by reference, a reinterpreting write whose
value is used, a union in a stage input or output, and a member access whose object type is unknown. A union in a
buffer or uniform block keeps the source layout, because its storage has the union's size and alignment.
"""

from __future__ import annotations

import copy

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    PointerAccessNode,
    PointerType,
    PrimitiveType,
    ReferenceType,
    StructMemberNode,
    StructNode,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
)
from ..semantics.hub_types import (
    call_name,
    element_type_node,
    iter_functions,
    iter_structs,
    type_from_name,
    type_from_node,
)
from ..semantics.narrow import NARROW_TARGETS
from ..semantics.rewrite import child_slots, clone, get_child, is_pure, set_child
from ..semantics.scoped_walk import ScopedWalker, program_stages
from .array_utils import collect_literal_int_constants, evaluate_literal_int_expression
from .spirv_union_bytes import (
    LANE_KINDS,
    SUBWORD_KINDS,
    SUBWORD_LANES,
    is_subword_lane_helper,
    lane_of,
    packable,
    subword_decode,
    subword_encode,
    subword_lane_write,
)
from .union_layout import UnionLayoutCollector

__all__ = ["UnionLayoutError", "lower_union_storage"]

TARGET_LABELS = {"vulkan": "SPIR-V", "wgsl": "WGSL", "slang": "Slang"}

# Member views this pass lowers: 32-bit words read as uint, int or float, alone or as a word array.
WORD_KINDS = frozenset({"uint32", "int32", "float32"})

_SCALAR_SPELLINGS = {"f": "float", "i": "int", "u": "uint", "b": "bool"}
_VECTOR_SPELLINGS = {"f": "vec", "i": "ivec", "u": "uvec", "b": "bvec"}
_NARROW_SPELLINGS = {
    ("u", 8): "uint8",
    ("i", 8): "int8",
    ("u", 16): "uint16",
    ("i", 16): "int16",
    ("f", 16): "float16",
}
# Stage inputs and outputs: another stage interpolates and matches them by location, member by member.
_INTERFACE_QUALIFIERS = frozenset({"in", "out", "inout", "attribute", "varying"})
_INTERFACE_ATTRIBUTES = frozenset({"location", "builtin"})
_REFERENCE_QUALIFIERS = frozenset({"out", "inout"})
# CrossGL buffer calls whose value is an element of the buffer they take first.
_ELEMENT_CALLS = frozenset({"buffer_load", "buffer_consume"})


class UnionLayoutError(ValueError):
    """Raised when a union's overlapping storage cannot be represented exactly in the target."""

    def __init__(
        self,
        message,
        *,
        target,
        union_name=None,
        member_name=None,
        member_type=None,
        size=None,
        alignment=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.project_diagnostic_code = (
            f"project.translate.{target}-union-layout-unsupported"
        )
        self.missing_capabilities = (f"{target}.union-storage-aliasing",)
        self.target = target
        self.union_name = union_name
        self.member_name = member_name
        self.member_type = member_type
        self.size = size
        self.alignment = alignment
        self.reason = reason
        self.source_location = source_location


def type_spelling(type_node, constants):
    """A CrossGL type as the GLSL-style spelling :class:`UnionLayoutCollector` reads (``float``, ``uvec2``)."""
    if type_node is None:
        return ""
    if isinstance(type_node, ArrayType):
        element = type_spelling(type_node.element_type, constants)
        size = type_node.size
        extent = size if isinstance(size, int) else None
        if extent is None and size is not None:
            extent = evaluate_literal_int_expression(size, constants)
        return f"{element}[{extent if extent is not None else '?'}]"
    if isinstance(type_node, (PrimitiveType, VectorType)):
        value = type_from_node(type_node)
        if value is not None:
            return value_spelling(value) or str(getattr(type_node, "name", type_node))
    if isinstance(type_node, (PrimitiveType, NamedType)):
        return str(type_node.name)
    return str(type_node)


def value_spelling(value):
    if value is None:
        return None
    if value.bits not in {1, 32}:
        # The collector's names for the sub-word views it recognizes (union_layout.LANE_ELEMENTS, u8vec4, u16vec2).
        if value.lanes == 1:
            return _NARROW_SPELLINGS.get((value.kind, value.bits))
        if (value.kind, value.bits, value.lanes) == ("u", 8, 4):
            return "u8vec4"
        if (value.kind, value.bits, value.lanes) == ("u", 16, 2):
            return "u16vec2"
        return None
    if value.lanes == 1:
        return _SCALAR_SPELLINGS.get(value.kind)
    prefix = _VECTOR_SPELLINGS.get(value.kind)
    return f"{prefix}{value.lanes}" if prefix else None


class ContractTypes:
    """The generator interface :class:`UnionLayoutCollector` reads, over the CrossGL program's own spellings."""

    # This lowering lowers 16-bit lane views of a union's storage words; other generators that share the
    # collector refuse them, because their lowering has no u16vec2 form.
    union_u16_lane_views = True

    def __init__(self, program):
        constants = list(getattr(program, "constants", None) or [])
        self.literal_int_constants = collect_literal_int_constants(constants)
        self.struct_member_types = {}
        for struct in iter_structs(program):
            if not isinstance(struct, StructNode):
                continue
            self.struct_member_types[struct.name] = {
                member.name: type_spelling(
                    getattr(member, "member_type", None), self.literal_int_constants
                )
                for member in getattr(struct, "members", None) or []
                if getattr(member, "name", None)
            }

    @staticmethod
    def map_type(name):
        return value_spelling(type_from_name(name)) or str(name)

    @staticmethod
    def attribute_value_to_string(value):
        if isinstance(value, IdentifierNode):
            return value.name
        if isinstance(value, LiteralNode):
            return str(value.value)
        return None if value is None else str(value)


def is_static_member(member):
    qualifiers = {str(q).lower() for q in getattr(member, "qualifiers", None) or []}
    attributes = {
        str(getattr(a, "name", a)).lower()
        for a in getattr(member, "attributes", None) or []
    }
    return "static" in qualifiers or "static" in attributes


def struct_name(type_node):
    while isinstance(type_node, ReferenceType):
        type_node = type_node.referenced_type
    if isinstance(type_node, (NamedType, PrimitiveType)):
        return type_node.name
    return None


def view_kind(member):
    """The value kind of one access of ``member``: an array view's element is read one word at a time."""
    kind = member["kind"]
    return kind[: -len("_array")] if kind.endswith("_array") else kind


def call(name, arguments):
    return FunctionCallNode(IdentifierNode(name), list(arguments))


def uint_literal(value):
    return LiteralNode(value, PrimitiveType("uint"))


class UnionViewFinder(ScopedWalker):
    """Finds every member view of a union in the function bodies, with the declarations in scope.

    Reads, writes and constructor calls are recorded for :class:`UnionStorageRewriter`; a view the rewrite cannot
    express raises through ``fail`` while the program is unchanged.
    """

    def __init__(self, program, layouts, fail, label, narrow_registers=False):
        super().__init__(program)
        self.layouts = layouts
        self.fail = fail
        self.label = label
        # Narrow-integer legalization ran: a u8vec4 value outside the union is an ivec4 register.
        self.narrow_registers = narrow_registers
        self.views = {}
        self.array_views = {}
        self.indexed = set()
        self.elements = {}
        self.writes = {}
        self.lane_writes = {}
        self.statements = set()
        self.constructors = {}
        self.member_names = {
            name for layout in layouts.values() for name in layout["members"]
        }
        self.functions = {}
        for function in iter_functions(program):
            self.functions.setdefault(function.name, []).append(function)

    def object_type(self, expr):
        type_node = self.scope.node_type(expr)
        if type_node is not None:
            return type_node
        if isinstance(expr, MemberAccessNode):
            members = self.scope.structs.get(
                struct_name(self.object_type(expr.object_expr))
            )
            return members.get(expr.member) if members else None
        if isinstance(expr, ArrayAccessNode):
            return element_type_node(self.object_type(expr.array_expr))
        if isinstance(expr, FunctionCallNode):
            name = call_name(expr)
            if name in self.layouts and name not in self.functions:
                return NamedType(name)
            arguments = expr.arguments or []
            if name in _ELEMENT_CALLS and arguments and name not in self.functions:
                return element_type_node(self.object_type(arguments[0]))
            return self.scope.functions.get(name)
        if isinstance(expr, ConstructorNode):
            return expr.constructor_type
        if isinstance(expr, UnaryOpNode) and expr.operator == "*":
            return element_type_node(self.object_type(expr.operand))
        if isinstance(expr, PointerAccessNode):
            pointee = element_type_node(self.object_type(expr.pointer_expr))
            members = self.scope.structs.get(struct_name(pointee))
            return members.get(expr.member) if members else None
        return None

    def union_of(self, expr):
        return self.layouts.get(struct_name(self.object_type(expr)))

    def view_of(self, expr):
        """``(layout, member, node)`` when ``expr`` is a view or one element of an array view."""
        if id(expr) in self.views:
            return self.views[id(expr)]
        if id(expr) in self.elements:
            return self.views[self.elements[id(expr)]]
        return None

    def view_in(self, expr):
        """The view ``expr`` is or is part of (``b.v.x`` is part of ``b.v``), or ``None``."""
        while expr is not None:
            view = self.view_of(expr)
            if view is not None:
                return view
            if isinstance(expr, MemberAccessNode):
                expr = expr.object_expr
            elif isinstance(expr, ArrayAccessNode):
                expr = expr.array_expr
            elif isinstance(expr, SwizzleNode):
                expr = expr.vector_expr
            else:
                return None
        return None

    def visit(self, node):
        if isinstance(node, (MemberAccessNode, PointerAccessNode)):
            self.visit_member(node)
        elif isinstance(node, ArrayAccessNode):
            view = self.array_views.get(id(node.array_expr))
            if view is not None:
                self.indexed.add(id(node.array_expr))
                self.elements[id(node)] = id(node.array_expr)
        elif isinstance(node, AssignmentNode):
            self.visit_assignment(node)
        elif isinstance(node, UnaryOpNode):
            self.visit_unary(node)
        elif isinstance(node, FunctionCallNode):
            self.visit_call(node)
        elif isinstance(node, ExpressionStatementNode):
            if isinstance(node.expression, (AssignmentNode, FunctionCallNode)):
                self.statements.add(id(node.expression))
        elif isinstance(node, ForNode):
            for part in (node.init, node.update):
                for item in part if isinstance(part, list) else [part]:
                    if isinstance(item, AssignmentNode):
                        self.statements.add(id(item))

    def visit_member(self, node):
        if isinstance(node, PointerAccessNode):
            layout = self.layouts.get(
                struct_name(element_type_node(self.object_type(node.pointer_expr)))
            )
            object_expr = node.pointer_expr
        else:
            layout = self.union_of(node.object_expr)
            object_expr = node.object_expr
        if layout is not None:
            member = layout["members"].get(node.member)
            if member is None:
                return
            self.require_lowered_view(layout, member)
            view = (layout, member, node)
            self.views[id(node)] = view
            if member["kind"].endswith("_array"):
                self.array_views[id(node)] = view
            return
        if (
            node.member in self.member_names
            and self.object_type(object_expr) is None
            and self.scope.infer(object_expr) is None
        ):
            layout = next(
                layout
                for layout in self.layouts.values()
                if node.member in layout["members"]
            )
            self.fail(
                layout,
                layout["members"][node.member],
                "view-object-type-unresolved",
                f"the type of the object of '.{node.member}' is not known, so it may be a view of the union",
            )

    def subword_view(self, expr):
        """The view when ``expr`` is one element of a sub-word view (an element of an array view), else ``None``."""
        view = self.view_of(expr)
        if view is None or view_kind(view[1]) not in SUBWORD_KINDS:
            return None
        if id(expr) in self.elements or not view[1]["kind"].endswith("_array"):
            return view
        return None

    def record_lane_write(self, node, vector, lane, value, operator):
        view = self.subword_view(vector)
        if view is None:
            return False
        layout, member, _ = view
        kind = view_kind(member)
        if kind in LANE_KINDS:
            reason = "lane-write-unsupported"
            detail = (
                "a 16-bit lane write reads and rewrites its storage word, so it must be a simple assignment "
                "whose object and lane index have no side effects"
            )
        else:
            reason = "byte-lane-write-unsupported"
            detail = (
                "a byte write reads and rewrites its storage word, so it must be a simple assignment whose "
                "object and lane index have no side effects"
            )
        if operator != "=" or not is_pure(lane) or not is_pure(vector):
            self.fail(layout, member, reason, detail)
        self.lane_writes[id(node)] = (view, vector, lane, value)
        return True

    def visit_assignment(self, node):
        target = node.target
        lane = lane_of(target)
        if lane is not None and self.record_lane_write(
            node, lane[0], lane[1], node.value, node.operator
        ):
            return
        view = self.view_of(target)
        if view is not None and (
            id(target) in self.elements or not view[1]["kind"].endswith("_array")
        ):
            layout, member, _ = view
            kind = view_kind(member)
            if kind in SUBWORD_KINDS and (
                node.operator != "=" or not packable(node.value, SUBWORD_LANES[kind])
            ):
                if kind in LANE_KINDS:
                    reason = "lane-view-write-unsupported"
                    detail = (
                        "a write of a 16-bit lane vector packs its two lanes into one word, so it must be a "
                        "simple assignment of a two-component constructor or of a value without side effects"
                    )
                else:
                    reason = "byte-view-write-unsupported"
                    detail = (
                        "a write of a byte vector packs its four lanes into one word, so it must be a simple "
                        "assignment of a four-component constructor or of a value without side effects"
                    )
                self.fail(layout, member, reason, detail)
            if view_kind(member) != "uint32" and node.operator != "=":
                if not is_pure(target):
                    self.fail(
                        layout,
                        member,
                        "interpreted-compound-write-unsupported",
                        "a compound write of a reinterpreted view evaluates its object twice",
                    )
            self.writes[id(node)] = view
            return
        view = self.view_in(target)
        if view is not None and view_kind(view[1]) != "uint32":
            self.fail(
                view[0],
                view[1],
                "interpreted-subcomponent-write-unsupported",
                "this subcomponent is not a word of the union's storage",
            )

    def visit_unary(self, node):
        operator = node.operator
        view = self.view_in(node.operand)
        if view is None:
            return
        if operator == "&":
            self.fail(
                view[0],
                view[1],
                "view-address-unsupported",
                "the address of a view does not alias the other views",
            )
        if operator in {"++", "--"} and view_kind(view[1]) != "uint32":
            self.fail(
                view[0],
                view[1],
                "interpreted-update-unsupported",
                "this view is not a word of the union's storage",
            )

    def visit_call(self, node):
        name = call_name(node)
        if name in self.layouts and name not in self.functions:
            self.visit_constructor(node, self.layouts[name])
            return
        arguments = list(node.arguments or [])
        skipped = None
        if is_subword_lane_helper(name, self.functions) and len(arguments) == 3:
            if self.record_lane_write(
                node, arguments[0], arguments[1], arguments[2], "="
            ):
                skipped = 0
        for index, argument in enumerate(arguments):
            if index == skipped:
                continue
            view = self.view_in(argument)
            if view is not None and self.reference_parameter(name, index):
                self.fail(
                    view[0],
                    view[1],
                    "view-reference-argument-unsupported",
                    f"'{name}' takes argument {index + 1} by reference, which would not alias the other views",
                )

    def reference_parameter(self, name, index):
        for function in self.functions.get(name, ()):
            parameters = list(getattr(function, "parameters", None) or [])
            if index >= len(parameters):
                continue
            parameter = parameters[index]
            qualifiers = {str(q).lower() for q in parameter.qualifiers or []}
            if qualifiers & _REFERENCE_QUALIFIERS or isinstance(
                parameter.param_type, (ReferenceType, PointerType)
            ):
                return True
        return False

    def visit_constructor(self, node, layout):
        arguments = list(node.arguments or [])
        if not arguments:
            self.constructors[id(node)] = (layout, "zero")
            return
        if len(arguments) == 1 and self.union_of(arguments[0]) is layout:
            self.constructors[id(node)] = (layout, "copy")
            return
        member = next(iter(layout["members"].values()))
        if len(arguments) != 1 or member["kind"].endswith("_array"):
            self.fail(
                layout,
                member,
                "union-initializer-shape-unsupported",
                "a union initializer must initialize its first value member",
            )
        self.require_word_view(layout, member)
        self.constructors[id(node)] = (layout, "first")

    def require_lowered_view(self, layout, member):
        """Refuse a view other than a 32-bit word view or a lowered sub-word view."""
        if view_kind(member) not in SUBWORD_KINDS:
            self.require_word_view(layout, member)

    def require_word_view(self, layout, member):
        """Refuse a sub-word lane or byte view; a union may declare one that the program never uses."""
        if view_kind(member) not in WORD_KINDS:
            self.fail(
                layout,
                member,
                "member-view-unsupported",
                f"{self.label} union storage lowers 32-bit word views only, and '{member['type']}' is a "
                "sub-word lane or byte view",
            )

    def finish(self):
        for view_id, (layout, member, _node) in self.array_views.items():
            if view_id not in self.indexed:
                self.fail(
                    layout,
                    member,
                    "whole-array-view-unsupported",
                    "a union array view cannot be used as a whole array",
                )
        for assignment_id, (layout, member, _node) in self.writes.items():
            if view_kind(member) != "uint32" and assignment_id not in self.statements:
                self.fail(
                    layout,
                    member,
                    "interpreted-assignment-value-unsupported",
                    "the value of a reinterpreting write is the stored word, not the assigned value",
                )
        for write_id, (view, _vector, _lane, _value) in self.lane_writes.items():
            if write_id not in self.statements:
                self.fail(
                    view[0],
                    view[1],
                    "interpreted-assignment-value-unsupported",
                    "the value of a sub-word lane write is the stored word, not the assigned value",
                )


class UnionStorageRewriter:
    """Rewrites the views :class:`UnionViewFinder` recorded into reads and writes of the union's storage."""

    def __init__(self, finder):
        self.finder = finder

    def rewrite(self, node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self.replace(child)
            if replacement is not child:
                set_child(node, field, key, replacement)

    def replace(self, node):
        finder = self.finder
        if id(node) in finder.lane_writes:
            return self.lane_write(*finder.lane_writes[id(node)])
        if id(node) in finder.writes:
            return self.write(node, finder.writes[id(node)])
        view = finder.view_of(node)
        if view is not None and (
            id(node) in finder.elements or not view[1]["kind"].endswith("_array")
        ):
            return self.decode(view, self.storage(node))
        if id(node) in finder.constructors:
            return self.construct(node, *finder.constructors[id(node)])
        self.rewrite(node)
        return node

    def storage(self, node):
        """The storage word or words that a view ``node`` (or an element of an array view) reads."""
        if id(node) in self.finder.elements:
            words = self.storage(node.array_expr)
            return ArrayAccessNode(words, self.replace(node.index_expr))
        field = self.finder.views[id(node)][0]["storage_field"]
        if isinstance(node, PointerAccessNode):
            return PointerAccessNode(self.replace(node.pointer_expr), field)
        return MemberAccessNode(self.replace(node.object_expr), field)

    @staticmethod
    def width(view, storage):
        return 1 if isinstance(storage, ArrayAccessNode) else view[0]["word_count"]

    def lane_write(self, view, vector, lane, value):
        storage = self.storage(vector)
        merged = subword_lane_write(
            view_kind(view[1]), storage, self.replace(lane), self.replace(value)
        )
        return AssignmentNode(storage, merged, "=")

    def decode(self, view, storage):
        kind = view_kind(view[1])
        if kind in SUBWORD_KINDS:
            return subword_decode(kind, storage, self.finder.narrow_registers)
        width = self.width(view, storage)
        if kind == "float32":
            return call("uintBitsToFloat", [storage])
        if kind == "int32":
            return call("int" if width == 1 else f"ivec{width}", [storage])
        return storage

    def encode(self, view, storage, value):
        kind = view_kind(view[1])
        if kind in SUBWORD_KINDS:
            return subword_encode(kind, value)
        width = self.width(view, storage)
        if kind == "float32":
            return call("floatBitsToUint", [value])
        if kind == "int32":
            return call("uint" if width == 1 else f"uvec{width}", [value])
        return value

    def write(self, node, view):
        storage = self.storage(node.target)
        value = self.replace(node.value)
        if view_kind(view[1]) == "uint32":
            return AssignmentNode(storage, value, node.operator)
        if node.operator != "=":
            current = self.decode(view, clone(storage))
            value = BinaryOpNode(current, node.operator[:-1], value)
        return AssignmentNode(storage, self.encode(view, storage, value), "=")

    def construct(self, node, layout, form):
        if form == "copy":
            return self.replace(node.arguments[0])
        words = layout["word_count"]
        if form == "zero":
            zero = uint_literal(0)
            storage = zero if words == 1 else call(f"uvec{words}", [zero])
            return call(layout["name"], [storage])
        member = next(iter(layout["members"].values()))
        view = (layout, member, None)
        value = self.encode(view, None, self.replace(node.arguments[0]))
        return call(layout["name"], [value])


def contains_union(type_node, layouts, structs, seen=None):
    """Whether a declaration of ``type_node`` holds a union, directly or in an array, pointer or structure."""
    seen = set() if seen is None else seen
    while True:
        element = element_type_node(type_node)
        if element is None:
            break
        type_node = element
    if isinstance(type_node, NamedType) and type_node.generic_args:
        return any(
            contains_union(argument, layouts, structs, seen)
            for argument in type_node.generic_args
        )
    name = struct_name(type_node)
    if name in layouts:
        return True
    if name is None or name in seen or name not in structs:
        return False
    seen.add(name)
    return any(
        contains_union(member_type, layouts, structs, seen)
        for member_type in structs[name].values()
    )


def interface_declaration(declaration):
    qualifiers = {
        str(q).lower() for q in getattr(declaration, "qualifiers", None) or []
    }
    attributes = {
        str(getattr(a, "name", a)).lower()
        for a in getattr(declaration, "attributes", None) or []
    }
    return bool(
        qualifiers & _INTERFACE_QUALIFIERS
        or attributes & _INTERFACE_ATTRIBUTES
        or getattr(declaration, "semantic", None)
    )


def interface_declarations(program):
    """Every stage input and output declaration: a module or stage variable, or an entry point parameter.

    A buffer or uniform block keeps the source's layout, because the storage words have the union's size and
    alignment; a stage interface would carry the storage instead of the source's member to the other stage.
    """
    variables = list(getattr(program, "global_variables", None) or [])
    for stage in program_stages(program):
        variables.extend(getattr(stage, "local_variables", None) or [])
        entry = getattr(stage, "entry_point", None)
        for parameter in getattr(entry, "parameters", None) or []:
            if interface_declaration(parameter):
                yield parameter, parameter.param_type
    for variable in variables:
        if getattr(variable, "name", None) and interface_declaration(variable):
            yield variable, variable.var_type


def brace_initializers(program, layouts):
    """Turn a brace initializer of a union (``B b = {x};``) into the constructor call it denotes."""

    def lower(type_node, value):
        if not isinstance(value, ArrayLiteralNode):
            return value
        if isinstance(type_node, ArrayType):
            value.elements = [lower(type_node.element_type, e) for e in value.elements]
            return value
        name = struct_name(type_node)
        if name in layouts:
            return call(name, value.elements)
        return value

    def walk(node):
        for field, key in list(child_slots(node)):
            walk(get_child(node, field, key))
        if isinstance(node, VariableNode) and node.initial_value is not None:
            node.initial_value = lower(node.var_type, node.initial_value)

    walk(program)


def lower_union_storage(program, target):
    """Return ``program`` with every ``@union_layout`` structure lowered to one storage member, or raise.

    A program without a union is returned unchanged; otherwise the rewrite applies to a copy, so the caller's
    program stays as parsed.
    """
    label = TARGET_LABELS[target]

    def error(node, *, reason, detail, member=None, **details):
        name = getattr(node, "name", None)
        return UnionLayoutError(
            f"{label} cannot preserve union '{name}': {detail}",
            target=target,
            union_name=name,
            member_name=getattr(member, "name", None),
            member_type=details.get("member_type"),
            size=details.get("size"),
            alignment=details.get("alignment"),
            reason=reason,
            source_location=getattr(member, "source_location", None)
            or getattr(node, "source_location", None),
        )

    def fail(layout, member, reason, detail):
        raise error(
            layout["node"],
            reason=reason,
            detail=detail,
            member=member["node"] if member else None,
            member_type=member["type"] if member else None,
            size=layout["size"],
            alignment=layout["alignment"],
        )

    if not any(
        UnionLayoutCollector.attribute(struct, "union_layout") is not None
        for struct in iter_structs(program)
        if isinstance(struct, StructNode)
    ):
        return program
    program = copy.deepcopy(program)
    types = ContractTypes(program)
    structs = [s for s in iter_structs(program) if isinstance(s, StructNode)]
    layouts = UnionLayoutCollector(types, error, is_static_member).collect(structs)
    member_types = {
        struct.name: {
            member.name: getattr(member, "member_type", None)
            for member in struct.members or []
            if getattr(member, "name", None)
        }
        for struct in structs
    }
    for declaration, type_node in interface_declarations(program):
        if contains_union(type_node, layouts, member_types):
            layout = next(iter(layouts.values()))
            if struct_name(type_node) in layouts:
                layout = layouts[struct_name(type_node)]
            raise error(
                layout["node"],
                reason="interface-union-layout-unsupported",
                detail=(
                    f"stage interface '{getattr(declaration, 'name', None)}' would pass the storage words "
                    "to the other stage instead of the source's members"
                ),
                size=layout["size"],
                alignment=layout["alignment"],
            )

    brace_initializers(program, layouts)
    finder = UnionViewFinder(
        program, layouts, fail, label, narrow_registers=target in NARROW_TARGETS
    )
    finder.run_program()
    finder.finish()
    UnionStorageRewriter(finder).rewrite(program)
    for layout in layouts.values():
        node = layout["node"]
        storage = PrimitiveType("uint")
        if layout["word_count"] > 1:
            storage = VectorType(PrimitiveType("uint"), layout["word_count"])
        kept = [member for member in node.members or [] if is_static_member(member)]
        node.members = [StructMemberNode(layout["storage_field"], storage), *kept]
        node.attributes = [
            attribute
            for attribute in node.attributes or []
            if str(getattr(attribute, "name", "")).lower() != "union_layout"
        ]
    return program
