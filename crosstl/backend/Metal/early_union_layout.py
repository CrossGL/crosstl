"""Fail fast on Metal unions that word-based targets cannot store.

The HLSL, GLSL, and SPIR-V generators validate retained source unions late,
after the whole file has been reverse-translated and lowered
(:class:`crosstl.translator.codegen.union_layout.UnionLayoutCollector` for
HLSL, :class:`crosstl.translator.codegen.glsl_union_storage.UnionStorage` for
GLSL/SPIR-V).  A union whose storage is not a 4-, 8-, or 16-byte word is
refused there, but by then a large Metal kernel has already spent minutes in
materialization, parsing, reverse code generation, and the semantics passes.
The corpus worker gives each file 120 seconds, so such files time out instead
of being refused quickly.

This module runs the same check right after the Metal parse, on the Metal
AST, so the refusal lands while the pipeline is still cheap (materialize +
parse).  It only refuses when the union's size and alignment are *proven*
from the source, using the same layout rules as
``MetalCrossGLCodeGen.metal_concrete_type_layout`` /
``metal_struct_member_layout``.  Anything it cannot prove -- a member type it
cannot lay out, a non-literal array extent, an ``alignas`` it cannot
evaluate, an ambiguous typedef -- is left alone (fail open): the late check
still refuses those files if it must.  The error raised is the exact typed
error the late check raises, so the corpus outcome (``refused``) and the
diagnostic code are unchanged.

The check deliberately does *not* run for ``vulkan``: the SPIR-V generator
has no union validation of its own, so a 2-byte union is not, by itself, a
SPIR-V refusal.  It also does not run for ``metal``, ``slang``, ``wgsl``,
``cuda``, or ``cgl`` targets, which keep unions natively.
"""

from __future__ import annotations

# Targets whose generators refuse non-word unions (see module docstring).
_WORD_TARGETS = frozenset({"directx", "opengl", "webgl"})

_SUPPORTED_UNION_SIZES = frozenset({4, 8, 16})

_MAX_LAYOUT_SIZE = (1 << 63) - 1


def refuse_unsupported_metal_union_layouts(program, target):
    """Raise the late union-layout error early, or return doing nothing.

    ``program`` is the parsed Metal AST; ``target`` is the normalized backend
    name.  Returns ``None`` when the target has no union-layout restriction
    or when no union in the program is provably unrepresentable.
    """
    target = str(target or "").strip().lower()
    if target not in _WORD_TARGETS:
        return
    structs, aliases = _collect_program_tables(program)
    for union in structs.values():
        if getattr(union, "aggregate_kind", None) != "union":
            continue
        layout = _aggregate_layout(union, aliases, structs, frozenset())
        if layout is None:
            # Unprovable layout: fail open and let the late check decide.
            continue
        size, alignment = layout
        if size not in _SUPPORTED_UNION_SIZES:
            raise _union_error(
                target,
                union,
                member=None,
                reason="storage-size-unsupported",
                detail=(
                    f"{size}-byte storage is outside the supported "
                    "4/8/16-byte value contract"
                ),
                size=size,
                alignment=alignment,
            )
        if alignment != size:
            raise _union_error(
                target,
                union,
                member=None,
                reason="aggregate-alignment-unsupported",
                detail=(
                    f"{size}-byte storage with {alignment}-byte alignment "
                    "may change enclosing aggregate layout"
                ),
                size=size,
                alignment=alignment,
            )
        for member, member_size, member_alignment in _union_member_layouts(
            union, aliases, structs
        ):
            if member_size != size:
                raise _union_error(
                    target,
                    union,
                    member=member,
                    reason="member-size-mismatch",
                    detail=(
                        f"member size {member_size!r} does not cover the "
                        f"full {size}-byte canonical storage"
                    ),
                    size=size,
                    alignment=alignment,
                )
            if (
                not member_alignment
                or member_alignment > alignment
                or alignment % member_alignment != 0
            ):
                raise _union_error(
                    target,
                    union,
                    member=member,
                    reason="member-alignment-unsupported",
                    detail=(
                        f"member alignment {member_alignment!r} is "
                        f"incompatible with aggregate alignment {alignment}"
                    ),
                    size=size,
                    alignment=alignment,
                )


def _union_error(target, union, *, member, reason, detail, size, alignment):
    """Build the same typed error the late union check raises."""
    union_name = getattr(union, "name", None) or "<anonymous>"
    member_name = getattr(member, "name", None) if member is not None else None
    member_type = getattr(member, "vtype", None) if member is not None else None
    source_location = getattr(member, "source_location", None) or getattr(
        union, "source_location", None
    )
    if target == "directx":
        from ...translator.codegen.directx_codegen import DirectXUnionLayoutError

        member_detail = f" member '{member_name}'" if member_name else ""
        return DirectXUnionLayoutError(
            f"DirectX cannot lower source union '{union_name}'{member_detail} "
            f"to canonical HLSL storage: {detail}",
            union_name=union_name,
            member_name=member_name,
            member_type=member_type,
            size=size,
            alignment=alignment,
            reason=reason,
            source_location=source_location,
        )
    from ...translator.codegen.glsl_union_storage import OpenGLUnionLayoutError

    return OpenGLUnionLayoutError(
        f"OpenGL cannot preserve union '{union_name}': {detail}",
        union_name=union_name,
        member_name=member_name,
        member_type=member_type,
        size=size,
        alignment=alignment,
        reason=reason,
        source_location=source_location,
    )


def _collect_program_tables(program):
    """Return ``(structs_by_name, aliases)`` for the Metal AST.

    ``structs_by_name`` maps every struct/union name to its node.  A name
    declared twice is dropped (ambiguous: fail open for that name).
    ``aliases`` maps every uniquely-declared typedef name to its alias text.
    """
    from ..common_ast import StructNode, TypeAliasNode

    structs = {}
    ambiguous_structs = set()
    aliases = {}
    ambiguous_aliases = set()
    for node in _iter_ast_nodes(program):
        if isinstance(node, StructNode):
            name = getattr(node, "name", None)
            if not name:
                continue
            if name in structs and structs[name] is not node:
                ambiguous_structs.add(name)
            else:
                structs[name] = node
        elif isinstance(node, TypeAliasNode):
            name = getattr(node, "name", None)
            alias_type = getattr(node, "alias_type", None)
            if not name or not isinstance(alias_type, str):
                continue
            if name in aliases and aliases[name] != alias_type:
                ambiguous_aliases.add(name)
            else:
                aliases[name] = alias_type
    for name in ambiguous_structs:
        structs.pop(name, None)
    for name in ambiguous_aliases:
        aliases.pop(name, None)
    return structs, aliases


def _iter_ast_nodes(root):
    """Yield every AST node reachable from ``root`` (visited-set walk)."""
    from ..common_ast import ASTNode

    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, ASTNode):
            if id(value) in seen:
                continue
            seen.add(id(value))
            yield value
            children = (
                child
                for key, child in vars(value).items()
                if key not in {"parent", "annotations"}
            )
        elif isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = value
        else:
            continue
        stack.extend(children)


def _resolve_alias(type_text, aliases):
    """Resolve ``type_text`` through ``aliases``; ``None`` on a cycle."""
    seen = set()
    current = type_text
    while current in aliases:
        if current in seen:
            return None
        seen.add(current)
        current = aliases[current]
    return current


def _type_layout(type_text, aliases, structs, resolving):
    """Return ``(size, alignment)`` for a Metal type, or ``None``.

    Mirrors the layout rules of ``MetalCrossGLCodeGen.metal_concrete_type_layout``
    for the cases this check can prove: concrete types via
    :func:`crosstl.backend.Metal.type_layout.metal_type_layout`, and user
    aggregates via :func:`_aggregate_layout`.  Pointers, references, and
    unknown types return ``None`` (fail open).
    """
    from .type_layout import metal_type_layout

    resolved = _resolve_alias(type_text, aliases)
    if resolved is None:
        return None
    layout = metal_type_layout(resolved)
    if layout is not None:
        return layout
    struct = structs.get(resolved)
    if struct is None:
        return None
    return _aggregate_layout(struct, aliases, structs, resolving)


def _aggregate_layout(struct, aliases, structs, resolving):
    """Return ``(size, alignment)`` for a struct/union node, or ``None``.

    Mirrors ``MetalCrossGLCodeGen.metal_concrete_type_layout``: a union takes
    the largest member size rounded up to the largest member alignment; a
    struct lays members out sequentially with alignment padding.  Returns
    ``None`` whenever the layout cannot be proven (attributes, ``alignas``,
    template/generic/bases, unprovable member, recursion).
    """
    from ..common_ast import (
        EnumNode,
        FunctionNode,
        StaticAssertNode,
        StructNode,
        TypeAliasNode,
        VariableNode,
    )

    name = getattr(struct, "name", None)
    if name in resolving:
        return None
    if (
        getattr(struct, "alignas", None)
        or getattr(struct, "attributes", None)
        or getattr(struct, "template_parameters", None)
        or getattr(struct, "generics", None)
        or getattr(struct, "bases", None)
        or getattr(struct, "base_classes", None)
        or getattr(struct, "base_types", None)
    ):
        return None
    resolving = resolving | ({name} if name else set())
    member_layouts = []
    for member in getattr(struct, "members", []) or []:
        if isinstance(
            member,
            (EnumNode, FunctionNode, StaticAssertNode, StructNode, TypeAliasNode),
        ):
            continue
        if not isinstance(member, VariableNode):
            return None
        if "static" in {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }:
            continue
        layout = _member_layout(member, aliases, structs, resolving)
        if layout is None:
            return None
        member_layouts.append(layout)
    if getattr(struct, "aggregate_kind", None) == "union":
        size = max((layout[0] for layout in member_layouts), default=0)
        alignment = max((layout[1] for layout in member_layouts), default=1)
        size = _align_up(size, alignment)
        size = max(size, 1)
    else:
        offset = 0
        alignment = 1
        for member_size, member_alignment in member_layouts:
            offset = _align_up(offset, member_alignment)
            offset += member_size
            alignment = max(alignment, member_alignment)
        size = _align_up(offset, alignment)
        size = max(size, 1)
    if size > _MAX_LAYOUT_SIZE:
        return None
    return size, alignment


def _member_layout(member, aliases, structs, resolving):
    """Return ``(size, alignment)`` for one struct/union member, or ``None``.

    Mirrors ``MetalCrossGLCodeGen.metal_struct_member_layout``: the member
    type's layout, widened by any fixed array extents (each extent must be a
    provable positive integer literal, otherwise ``None``).
    """
    # Mirror metal_struct_member_layout: ``alignas`` is truthy-tested (an
    # empty specifier list means "no alignas"), ``bitfield_width`` is None-tested.
    if (
        getattr(member, "alignas", None)
        or getattr(member, "bitfield_width", None) is not None
    ):
        return None
    member_type = getattr(member, "vtype", None)
    if not isinstance(member_type, str) or not member_type:
        return None
    if "*" in member_type or "&" in member_type:
        return None
    suffix = getattr(member, "declarator_type_suffix", "") or ""
    if "*" in suffix or "&" in suffix:
        return None
    layout = _type_layout(member_type, aliases, structs, resolving)
    if layout is None:
        return None
    member_size, member_alignment = layout
    for extent_expression in getattr(member, "array_sizes", None) or []:
        extent = _constant_int(extent_expression)
        if extent is None or isinstance(extent, bool) or extent <= 0:
            return None
        stride = _align_up(member_size, member_alignment)
        if stride > _MAX_LAYOUT_SIZE // extent:
            return None
        member_size = stride * extent
    return member_size, member_alignment


def _union_member_layouts(union, aliases, structs):
    """Return ``[(member, size, alignment)]`` for a union's instance members.

    Returns an empty list when any member layout is unprovable (fail open);
    the aggregate-size check has already run by then, so only the
    member-level rules are skipped.
    """
    from ..common_ast import VariableNode

    layouts = []
    for member in getattr(union, "members", []) or []:
        if not isinstance(member, VariableNode):
            continue
        if "static" in {
            str(qualifier).lower()
            for qualifier in getattr(member, "qualifiers", []) or []
        }:
            continue
        if getattr(member, "name", None) is None:
            # Anonymous member: the late check has its own rule for this.
            return []
        layout = _member_layout(member, aliases, structs, frozenset())
        if layout is None:
            return []
        layouts.append((member, layout[0], layout[1]))
    return layouts


def _constant_int(expression):
    """Evaluate a literal integer array extent, or return ``None``."""
    from ..common_ast import BinaryOpNode

    if isinstance(expression, bool):
        return None
    if isinstance(expression, int):
        return expression
    if isinstance(expression, str):
        text = expression.strip()
        if text.lstrip("-").isdigit():
            return int(text)
        return None
    if isinstance(expression, BinaryOpNode):
        left = _constant_int(getattr(expression, "left", None))
        right = _constant_int(getattr(expression, "right", None))
        if left is None or right is None:
            return None
        operator = getattr(expression, "op", None)
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/" and right != 0:
            return left // right
        return None
    return None


def _align_up(value, alignment):
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment
