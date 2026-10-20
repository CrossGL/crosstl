"""Structures built from one scalar, written member by member.

HLSL converts a scalar to a structure by converting it to every scalar component of the structure: ``(VSOutput)0``
sets each member, element and nested member to zero of its own type (HLSL specification, "Conversions", scalar to
aggregate). The HLSL importer carries the cast as ``VSOutput(0)``, a structure constructor with one argument. The
HLSL and Slang generators read that call as a splat; GLSL, SPIR-V and Metal take a structure constructor's
arguments as its members in order, so they refused the call or lost the members after the first.

:func:`expand_struct_scalar_splats` rewrites ``S(x)``, for a structure ``S`` of the program whose members are not
all filled by ``x`` alone, into ``S(m1(x), m2(x), ...)``: a scalar or vector member converts ``x`` to its type, an
array member repeats its element's value for every element, and a nested structure is expanded the same way. The
scalar is evaluated once per component, so the rewrite only applies when it is pure. A matrix member takes the
splat only from a literal zero, because a one-argument matrix constructor fills the diagonal alone. Calls the
rewrite cannot write member by member are left as they are, and the generators keep refusing them.

The rewrite carries an HLSL and Slang rule, so it applies to programs imported from those languages only: Metal's
``S{x}`` imports as the same one-argument call but initializes the first member alone. Structures with a
registered scalar conversion (:mod:`crosstl.translator.structure_conversions`) keep their own contract and are not
rewritten.
"""

from __future__ import annotations

from .. import ast as A
from ..structure_conversions import registered_structure_conversion_for_identity
from .hub_types import call_name, type_from_node
from .rewrite import call, clone, is_pure
from .scoped_walk import ScopedWalker, program_stages

__all__ = ["expand_struct_scalar_splats", "struct_member_types"]

_VECTOR_PREFIXES = {
    "float": "vec",
    "int": "ivec",
    "uint": "uvec",
    "double": "dvec",
    "bool": "bvec",
}
_MATRIX_PREFIXES = {"float": "mat", "double": "dmat"}
# Source languages whose one-scalar structure conversion is a splat.
_SPLATTING_SOURCES = frozenset({"directx", "hlsl", "slang"})
# Targets whose generators give a one-scalar structure constructor HLSL's splat meaning.
_SPLATTING_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "slang"}
)


def expand_struct_scalar_splats(program, source, target) -> int:
    """Rewrite one-scalar structure constructors member by member in place; returns how many were rewritten.

    Only a program imported from HLSL or Slang carries the splat; a Metal ``S{x}`` imports as the same call but
    sets the first member alone. The HLSL and Slang generators read the call as a splat themselves.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    if str(source or "").lower() not in _SPLATTING_SOURCES:
        return 0
    if str(target or "").lower() in _SPLATTING_TARGETS:
        return 0
    structs = struct_member_types(program)
    if not structs:
        return 0
    walker = _SplatWalker(program, structs)
    walker.run_program()
    return walker.rewritten


def struct_member_types(program):
    """``{struct name: [(member name, member type), ...]}`` for the program's structures, in declaration order."""
    structs = {}
    declared = list(getattr(program, "structs", None) or [])
    for stage in program_stages(program):
        declared.extend(getattr(stage, "local_structs", None) or [])
    for struct in declared:
        if not isinstance(struct, A.StructNode) or not getattr(struct, "name", None):
            continue
        if getattr(struct, "is_forward_declaration", False):
            continue
        members = []
        for member in getattr(struct, "members", None) or []:
            member_type = getattr(member, "member_type", None)
            if member_type is None:
                member_type = getattr(member, "var_type", None)
            members.append((getattr(member, "name", None), member_type))
        structs[struct.name] = members
    return structs


class _SplatWalker(ScopedWalker):
    def __init__(self, program, structs):
        super().__init__(program)
        self.structs = structs
        self.rewritten = 0

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        name = call_name(node)
        members = self.structs.get(name)
        arguments = node.arguments
        if not members or not isinstance(arguments, list) or len(arguments) != 1:
            return
        if registered_structure_conversion_for_identity(name) is not None:
            return
        source = arguments[0]
        source_type = self.scope.infer(source)
        if source_type is None or source_type.lanes != 1 or not is_pure(source):
            return
        if len(members) == 1 and type_from_node(members[0][1]) is not None:
            # One scalar or vector member: the call is an ordinary constructor already.
            return
        expanded = _member_values(members, source, self.structs, (name,))
        if expanded is None:
            return
        arguments[:] = expanded
        self.rewritten += 1


def _member_values(members, source, structs, open_structs):
    values = []
    for member_name, member_type in members:
        if member_name is None:
            return None
        value = _splat_value(member_type, source, structs, open_structs)
        if value is None:
            return None
        values.append(value)
    return values


def _splat_value(type_node, source, structs, open_structs):
    if isinstance(type_node, A.ArrayType):
        size = _literal_size(type_node.size)
        element_type = type_node.element_type
        if size is None or isinstance(element_type, A.ArrayType):
            return None
        element_name = _spelling(element_type)
        if element_name is None:
            return None
        elements = []
        for _ in range(size):
            element = _splat_value(element_type, source, structs, open_structs)
            if element is None:
                return None
            elements.append(element)
        return call(f"{element_name}[{size}]", elements)
    if isinstance(type_node, A.MatrixType):
        name = _spelling(type_node)
        if name is None or not _is_literal_zero(source):
            return None
        return call(name, [clone(source)])
    struct_name = _struct_name(type_node)
    if struct_name in structs:
        if struct_name in open_structs:
            return None
        nested = _member_values(
            structs[struct_name], source, structs, open_structs + (struct_name,)
        )
        return call(struct_name, nested) if nested is not None else None
    if type_from_node(type_node) is None:
        return None
    name = _spelling(type_node)
    return call(name, [clone(source)]) if name is not None else None


def _spelling(type_node):
    """The CrossGL constructor name of a scalar, vector, matrix or structure type, or ``None``."""
    if isinstance(type_node, A.PrimitiveType):
        return type_node.name
    if isinstance(type_node, A.NamedType):
        return None if type_node.generic_args else type_node.name
    if isinstance(type_node, A.VectorType):
        element = _spelling(type_node.element_type)
        prefix = _VECTOR_PREFIXES.get(element)
        size = type_node.size
        return f"{prefix}{size}" if prefix and str(size).isdigit() else None
    if isinstance(type_node, A.MatrixType):
        prefix = _MATRIX_PREFIXES.get(_spelling(type_node.element_type))
        rows, cols = type_node.rows, type_node.cols
        if prefix is None or not str(rows).isdigit() or not str(cols).isdigit():
            return None
        return f"{prefix}{rows}" if rows == cols else f"{prefix}{rows}x{cols}"
    return None


def _struct_name(type_node):
    if isinstance(type_node, (A.NamedType, A.PrimitiveType)):
        return type_node.name
    return None


def _literal_size(size):
    if isinstance(size, int) and not isinstance(size, bool):
        return size if size > 0 else None
    if isinstance(size, A.LiteralNode):
        return _literal_size(size.value)
    if isinstance(size, str) and size.isdigit():
        return int(size) or None
    return None


def _is_literal_zero(expr):
    if not isinstance(expr, A.LiteralNode):
        return False
    value = expr.value
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    text = str(value).rstrip("uUfFhHlL")
    try:
        return float(text) == 0
    except ValueError:
        return text == "false"
