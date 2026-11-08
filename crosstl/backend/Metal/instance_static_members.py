"""Static data members read through an object expression.

``t.value`` where ``value`` is a ``static constexpr`` member of ``t``'s struct
names the same constant as ``T::value`` ([class.static]/1). The importer used
to emit it as an ordinary field read, but the lowered struct keeps no storage
for static members (an empty one becomes ``int _crossgl_empty``), so the
generated HLSL read a field that does not exist. The access is rendered as the
member's constant instead.

C++ still evaluates the object expression of ``f().value``, so only plain
objects (names, member chains, constant-indexed elements and ``p->value``)
are folded; any other object expression keeps its existing lowering.
"""

from __future__ import annotations

import re
from typing import Any

from ..common_ast import ArrayAccessNode, MemberAccessNode, VariableNode

_QUALIFIER_RE = re.compile(
    r"\b(?:const|volatile|thread|device|constant|threadgroup|threadgroup_imageblock"
    r"|ray_data|object_data)\b"
)


def _is_plain_object(node: Any) -> bool:
    if isinstance(node, VariableNode):
        return isinstance(getattr(node, "name", None), str)
    if isinstance(node, MemberAccessNode):
        return _is_plain_object(node.object)
    if isinstance(node, ArrayAccessNode):
        index = node.index
        return _is_plain_object(node.array) and (
            _is_plain_object(index)
            or (isinstance(index, (int, str)) and re.fullmatch(r"\d+[uU]?", str(index)))
        )
    return False


def _owner_type_name(text: str, through_pointer: bool) -> str | None:
    owner = _QUALIFIER_RE.sub(" ", text)
    owner = re.sub(r"\s*&+\s*$", "", owner).strip()
    if through_pointer and owner.endswith("*"):
        # `p->member`: the inferred type is the pointer or, for locals, already its pointee.
        owner = owner[:-1].strip()
    if not owner or re.search(r"[*&\[\]]", owner):
        return None
    return re.sub(r"\s+", " ", owner)


def render_instance_static_struct_member(codegen: Any, expr: Any) -> str | None:
    """The constant ``expr`` (``obj.member``) names, or ``None`` to keep the field read."""

    member = str(getattr(expr, "member", ""))
    obj = getattr(expr, "object", None)
    if not member or obj is None:
        return None
    if isinstance(obj, VariableNode) and obj.name == "this":
        return None
    inferred = codegen.expression_metal_type(obj)
    if not isinstance(inferred, str) or not inferred:
        return None
    owner = _owner_type_name(
        codegen.normalized_metal_type(codegen.resolve_type_alias(inferred)),
        bool(getattr(expr, "is_pointer", False)),
    )
    if owner is None:
        return None
    resolved = codegen.static_struct_member_owner(f"{owner}::{member}")
    if resolved is None or resolved[0] not in codegen.struct_name_map:
        return None
    key = (codegen.map_struct_name(resolved[0]), member)
    if key not in codegen.struct_static_constant_members:
        return None
    if not _is_plain_object(obj):
        # The object is evaluated for its side effects; the project pipeline lowers that form itself.
        return None
    rendered = codegen.render_static_struct_member_identifier(
        f"{owner}::{member}", require_constant=True
    )
    if rendered is not None:
        codegen.record_elided_parameter_uses(obj)
    return rendered
