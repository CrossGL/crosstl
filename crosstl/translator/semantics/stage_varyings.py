"""Shared pieces of the passes that rewrite an HLSL-style stage interface into GLSL's stage variables.

HLSL passes a geometry or tessellation stage its vertices as entry-point arrays of structures (``triangle V v[3]``,
``InputPatch<V, N>``, ``OutputPatch<V, N>``) and links stages by semantic. GLSL and SPIR-V declare one arrayed stage
variable per structure member and link stages by location. :mod:`.geometry_streams` and
:mod:`.tessellation_patches` both need to

* read the semantic, built-in and explicit location a structure member carries;
* give a user varying the location the neighbouring stage gives the same member (:func:`neighbour_location`):
  its explicit ``location``, else for GLSL the location of its semantic, else for SPIR-V its semantic's fixed
  location or its running position in the structure (:mod:`..codegen.stage_io_semantic_locations`);
* declare that varying (:func:`varying_declaration`) under a name no other declaration uses (:class:`FreshNames`);
* rewrite ``v[i].member`` reads of an entry-point array into reads of those variables, or build a local copy of the
  array when it is used otherwise (:func:`rewrite_array_reads`).
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.stage_io_semantic_locations import (
    glsl_semantic_location,
    spirv_semantic_location,
)
from .rewrite import child_slots, get_child, set_child

__all__ = [
    "GLSL_TARGETS",
    "INTERPOLATION",
    "SPIRV_TARGETS",
    "FreshNames",
    "builtin",
    "call_statement",
    "explicit_location",
    "function_local_names",
    "indexed",
    "int_literal",
    "is_system_value",
    "is_written",
    "literal_int",
    "location_span",
    "member_type",
    "neighbour_location",
    "per_vertex_builtin",
    "replace_statement",
    "rewrite_array_reads",
    "semantic",
    "slots",
    "stage_name",
    "stages",
    "type_name",
    "varying_declaration",
]

GLSL_TARGETS = frozenset({"opengl", "glsl"})
SPIRV_TARGETS = frozenset({"vulkan", "spirv"})
INTERPOLATION = frozenset({"flat", "noperspective", "centroid", "sample", "smooth"})
_LAYOUT_ATTRIBUTES = frozenset({"location", "component"})


def stages(program):
    found = getattr(program, "stages", None) or {}
    return list(found.values()) if isinstance(found, dict) else list(found)


def stage_name(stage):
    value = getattr(stage, "stage", None)
    return str(getattr(value, "value", value) or "").lower()


def type_name(type_node):
    if isinstance(type_node, str):
        return type_node
    return getattr(type_node, "name", None)


def member_type(member):
    return getattr(member, "member_type", None) or getattr(member, "param_type", None)


def int_literal(value):
    return A.LiteralNode(int(value), A.PrimitiveType("int"))


def literal_int(node):
    if isinstance(node, int) and not isinstance(node, bool):
        return node
    value = getattr(node, "value", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def semantic(node):
    """The semantic attribute of a member or parameter: its first attribute that is no interpolation or layout."""
    for attr in getattr(node, "attributes", None) or []:
        name = str(getattr(attr, "name", ""))
        if name.lower() in INTERPOLATION or name.lower() in _LAYOUT_ATTRIBUTES:
            continue
        return name
    return None


def builtin(node):
    found = semantic(node)
    return found if found is not None and found.startswith("gl_") else None


def is_system_value(node):
    found = semantic(node)
    return found is not None and found.upper().startswith("SV_")


def explicit_location(node):
    for attr in getattr(node, "attributes", None) or []:
        if str(getattr(attr, "name", "")).lower() == "location":
            return literal_int((attr.arguments or [None])[0])
    return None


def location_span(type_node):
    """Locations a varying of ``type_node`` occupies, counted as the SPIR-V emitter counts them."""
    if isinstance(type_node, A.ArrayType):
        return max(literal_int(type_node.size) or 1, 1) * location_span(
            type_node.element_type
        )
    if isinstance(type_node, A.MatrixType):
        return int(type_node.cols)
    return 1


def neighbour_location(target, struct_name, member, running, refuse):
    """The location of a user varying the neighbouring stage declares from the same structure member.

    ``refuse(detail, reason)`` raises the caller's diagnostic when GLSL would link the member by name.
    """
    explicit = explicit_location(member)
    if explicit is not None:
        return explicit
    found = semantic(member)
    if target in GLSL_TARGETS:
        location = glsl_semantic_location(found)
        if location is None:
            refuse(
                f"member '{struct_name}.{member.name}' (semantic {found or 'none'}) has no GLSL location, "
                "so it would link by name, and the lowered stage cannot declare an input and an output of "
                "the same name",
                "unlocated-varying",
            )
        return location
    location = spirv_semantic_location(found)
    return running if location is None else location


def varying_declaration(name, member, var_type, location, qualifiers):
    """A located stage variable for ``member``, keeping its semantic and interpolation attributes."""
    attributes = [A.AttributeNode("location", [int_literal(location)])]
    found = semantic(member)
    if found is not None:
        # Kept so that cross-stage validation matches the varying with the neighbouring stage's member.
        attributes.append(A.AttributeNode(found))
    attributes += [
        A.AttributeNode(attr.name)
        for attr in member.attributes or []
        if str(attr.name).lower() in INTERPOLATION
    ]
    return A.VariableNode(
        name=name, var_type=var_type, attributes=attributes, qualifiers=list(qualifiers)
    )


class FreshNames:
    """Allocates names no declaration of a program uses yet."""

    def __init__(self, program):
        self.names = _declared_names(program)

    def __call__(self, preferred):
        name, index = preferred, 0
        while name in self.names:
            index += 1
            name = f"{preferred}{index}"
        self.names.add(name)
        return name


def rewrite_array_reads(function, name, element_type, count, sources, refuse):
    """Rewrite ``name[i].member`` reads in ``function`` into ``sources[member](i)``.

    ``sources`` maps each member name (``None`` for an array of plain values) to a function from an index
    expression to the expression that reads it. When ``name`` is used other than through such reads (as a whole,
    written, or indexed without a member), the reads are left alone and the statements that build a local copy
    ``element_type name[count]`` are returned for the caller to prepend; otherwise the result is empty.
    """
    has_members = None not in sources
    if name in function_local_names(function):
        refuse(
            f"'{function.name}' declares a local '{name}' that shadows its vertex array",
            "shadowed-array",
        )
    uses, reads = 0, []
    for parent, field, key, node in slots(function.body):
        if isinstance(node, A.IdentifierNode) and node.name == name:
            uses += 1
        access = _array_read(node, name, has_members)
        if access is not None and not is_written(parent, field, node):
            reads.append((parent, field, key, access))
    if uses != len(reads):
        statements = [
            A.VariableNode(
                name=name, var_type=A.ArrayType(element_type, int_literal(count))
            )
        ]
        for index in range(count):
            for member, source in sources.items():
                target = A.ArrayAccessNode(A.IdentifierNode(name), int_literal(index))
                if member is not None:
                    target = A.MemberAccessNode(target, member)
                statements.append(A.AssignmentNode(target, source(int_literal(index))))
        return statements
    for parent, field, key, (index, member) in reads:
        if member not in sources:
            refuse(f"vertex array '{name}' has no member '{member}'", "array-member")
        set_child(parent, field, key, sources[member](index))
    return []


def per_vertex_builtin(array, name):
    """``index -> array[index].name``, a read or write of a ``gl_in[]`` or ``gl_out[]`` built-in."""

    def source(index):
        return A.MemberAccessNode(
            A.ArrayAccessNode(A.IdentifierNode(array), index), name
        )

    return source


def indexed(name):
    """``index -> name[index]``, a read or write of an arrayed stage variable."""

    def source(index):
        return A.ArrayAccessNode(A.IdentifierNode(name), index)

    return source


def call_statement(name, arguments=()):
    return A.ExpressionStatementNode(
        A.FunctionCallNode(A.IdentifierNode(name), list(arguments))
    )


def slots(root):
    """``(parent, field, key, child)`` for every child under ``root``, parents before children, each once."""
    stack, seen = [root], set()
    while stack:
        parent = stack.pop()
        if id(parent) in seen or not isinstance(parent, A.ASTNode):
            continue
        seen.add(id(parent))
        children = []
        for field, key in child_slots(parent):
            child = get_child(parent, field, key)
            yield parent, field, key, child
            children.append(child)
        stack.extend(reversed(children))


def is_written(parent, field, node):
    if isinstance(parent, A.AssignmentNode) and parent.target is node:
        return True
    return isinstance(parent, A.UnaryOpNode) and parent.operator in ("++", "--")


def replace_statement(body, statement, replacement):
    """Replace ``statement`` under ``body`` by the statement list ``replacement``."""
    for parent, field, key, node in slots(body):
        if node is not statement:
            continue
        container = getattr(parent, field)
        if (
            key is not None
            and isinstance(container, list)
            and isinstance(parent, A.BlockNode)
        ):
            container[key : key + 1] = replacement
        else:
            set_child(parent, field, key, A.BlockNode(replacement))
        return
    raise AssertionError("statement not found in its function body")


def function_local_names(function):
    names = set()
    for _, _, _, node in slots(function.body):
        if isinstance(node, A.VariableNode):
            names.add(node.name)
    return names


def _array_read(node, name, has_members):
    """``(index, member)`` when ``node`` reads ``name[index].member`` (or ``name[index]`` for plain values)."""
    if has_members:
        if not isinstance(node, A.MemberAccessNode):
            return None
        array = node.object_expr
        member = node.member
    else:
        array, member = node, None
    if (
        isinstance(array, A.ArrayAccessNode)
        and isinstance(array.array_expr, A.IdentifierNode)
        and array.array_expr.name == name
    ):
        return array.index_expr, member
    return None


def _declared_names(program):
    names = set()
    stack, seen = [program], set()
    while stack:
        node = stack.pop()
        if id(node) in seen or not isinstance(node, A.ASTNode):
            continue
        seen.add(id(node))
        name = getattr(node, "name", None)
        if isinstance(name, str):
            names.add(name)
        stack.extend(get_child(node, f, k) for f, k in child_slots(node))
    return names
