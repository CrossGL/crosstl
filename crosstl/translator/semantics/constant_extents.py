"""Array extents named by integer constants.

C-family sources size arrays with integral constant expressions of named constants: an MSL header shared with the
host declares ``static constant constexpr size_t N = X * Y * Z;`` and then ``Data items[N];`` (MSL 4.1 section 2.12
and C++17 [dcl.array]: the bound is a converted constant expression). HLSL does the same with ``static const uint``
and GLSL with ``const int`` (GLSL 4.60 section 4.1.9: the size is a constant integral expression). The CrossGL
generators print an extent as written and declare structures before constants. So a structure member whose extent
names a constant reached HLSL, GLSL and MSL before the constant's declaration, which DXC, glslang and the Metal
compiler reject ("use of undeclared identifier"). SPIR-V, which needs the value, refused it.

:func:`fold_constant_extents` replaces each extent of a structure or constant-block member that is an integer
expression of the program's integer constants and enumerators (combined with C's integer operators, as
:mod:`.binding_constants` evaluates binding indices) by its value, for every target. Other declarations follow the
constants in every generator's output and keep their spelling. An extent that names anything else is left to the
generators: a specialization constant, which GLSL and SPIR-V keep symbolic and the others refuse, a name the program
does not declare, or a name that a parameter or local declaration could shadow.

:func:`fold_stage_layout_counts` does the same for the element counts a stage's ``layout(...) in|out`` qualifiers
and entry-point attributes give (``max_vertices``, ``max_primitives``, ``vertices``, ``invocations`` and their
HLSL spellings). GLSL 4.60 section 4.4 makes each a constant integral expression, and ``GL_EXT_mesh_shader`` samples
write ``max_vertices = 4 * groupSizeX * groupSizeY`` over ``const uint`` declarations. The count sizes the stage's
implicitly sized output arrays (:mod:`.stage_io_arrays`) and the emitters' stage declarations, which accepted only a
literal or a single constant name, so those shaders were refused although their counts are known.
"""

from __future__ import annotations

from .. import ast as A
from .binding_constants import _ConstantTable, _refers_to_names, _Unresolved
from .declared_types import _nodes
from .hub_types import iter_structs
from .storage_blocks import _declared_names, _walk

__all__ = ["fold_constant_extents", "fold_stage_layout_counts"]

_CARRYING = frozenset({"cgl", "crossgl"})
# Stage layout entries and entry-point attributes whose single argument is an element count.
_STAGE_COUNTS = frozenset(
    {
        "max_vertices",
        "max_primitives",
        "maxvertexcount",
        "maxprimitivecount",
        "vertices",
        "outputcontrolpoints",
        "invocations",
        "instance",
    }
)
# The largest count folded: a larger value would depend on the width of the constants' unsigned type.
_LARGEST_COUNT = 2**31 - 1


def fold_constant_extents(program, target) -> int:
    """Replace the member array extents of ``program`` that name integer constants by their values; returns how many."""
    if str(target or "").lower() in _CARRYING or not isinstance(program, A.ShaderNode):
        return 0
    table = shadowed = None
    changed = 0
    # Type nodes included: the extents sit in declared types, which the rewriting walk skips.
    member_types = [
        node
        for container in iter_structs(program)
        for member in getattr(container, "members", None) or []
        for node in _nodes(member)
    ]
    for node in member_types:
        if not isinstance(node, A.ArrayType):
            continue
        size = getattr(node, "size", None)
        if not isinstance(size, A.ASTNode) or not _refers_to_names(size):
            continue
        if table is None:
            table, shadowed = _ConstantTable(program), _declared_names(program)
        names = {
            item.name for item in _walk(size) if isinstance(item, A.IdentifierNode)
        }
        if names & shadowed:
            continue
        try:
            value = table.evaluate(size)
        except _Unresolved:
            continue
        if value <= 0:
            continue
        node.size = A.LiteralNode(value, A.PrimitiveType("int"))
        changed += 1
    return changed


def _stage_count_arguments(stage):
    """The count arguments of ``stage``'s layout qualifiers and entry-point attributes, as ``(owner, index)``."""
    entries = [
        entry
        for qualifier in getattr(stage, "layout_qualifiers", None) or []
        for entry in getattr(qualifier, "entries", None) or []
    ]
    entry_point = getattr(stage, "entry_point", None)
    entries += list(getattr(entry_point, "attributes", None) or [])
    return [
        entry
        for entry in entries
        if str(getattr(entry, "name", "")).lower() in _STAGE_COUNTS
        and len(getattr(entry, "arguments", None) or []) == 1
    ]


def _stage_scope_names(stage):
    """Names a stage declares at its own scope, which would shadow a program constant in its layout."""
    names = set()
    for field in (
        "local_variables",
        "local_functions",
        "local_structs",
        "local_cbuffers",
    ):
        names.update(
            getattr(node, "name", None) for node in getattr(stage, field, None) or []
        )
    return names


def fold_stage_layout_counts(program, target) -> int:
    """Replace stage count arguments that are integer expressions of the program's constants by their values."""
    if str(target or "").lower() in _CARRYING or not isinstance(program, A.ShaderNode):
        return 0
    stages = getattr(program, "stages", None) or {}
    table = None
    changed = 0
    for stage in stages.values() if isinstance(stages, dict) else stages:
        local_names = None
        for entry in _stage_count_arguments(stage):
            argument = entry.arguments[0]
            if not isinstance(argument, A.ASTNode) or not _refers_to_names(argument):
                continue
            if local_names is None:
                local_names = _stage_scope_names(stage)
            names = {
                item.name
                for item in _walk(argument)
                if isinstance(item, A.IdentifierNode)
            }
            if names & local_names:
                continue
            if table is None:
                table = _ConstantTable(program)
            try:
                value = table.evaluate(argument)
            except _Unresolved:
                continue
            if not 0 < value <= _LARGEST_COUNT:
                continue
            entry.arguments[0] = A.LiteralNode(value, A.PrimitiveType("int"))
            changed += 1
    return changed
