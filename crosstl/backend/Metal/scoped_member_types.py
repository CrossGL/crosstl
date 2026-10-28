"""Resolve ``Owner::member`` type spellings whose owner is a concrete struct.

A qualified name whose nested-name-specifier denotes a class names the member
of that class ([basic.lookup.qual], [class.qual]).  When the owner is a struct
the module emits, or a function-local alias of one, a member type alias such
as ``Frag_float_8_8::frag_type`` therefore denotes the type that alias names
(``metal::vec<float, 2>``), not an unqualified ``frag_type``.
"""

from __future__ import annotations

import re
from typing import Any

_SCOPED_MEMBER_RE = re.compile(
    r"(?:typename\s+)?(?P<owner>(?:::)?[A-Za-z_]\w*)"
    r"(?P<members>(?:\s*::\s*[A-Za-z_]\w*)+)"
)


def resolve_concrete_scoped_member_type(converter: Any, text: Any) -> str | None:
    """Return the type ``text`` names through a concrete struct, or ``None``.

    ``converter`` is the Metal-to-CrossGL code generator.  ``None`` means that
    ``text`` is not ``Owner::member`` spelling, that the owner is not a struct
    the converter knows, or that a member is missing or ambiguous; callers keep
    their own spelling in those cases.
    """
    match = _SCOPED_MEMBER_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    owner = match.group("owner").lstrip(":")
    if owner == "metal":
        return None
    local_owner = getattr(converter, "local_struct_type_aliases", {}).get(owner)
    candidates = [owner] if local_owner is None else [local_owner, owner]
    struct_names = getattr(converter, "struct_name_map", {})
    for candidate in candidates:
        resolved_owner = converter.normalized_metal_type(
            converter.resolve_type_alias(candidate)
        )
        if resolved_owner not in struct_names:
            continue
        members = "".join(match.group("members").split())
        return converter.resolve_concrete_struct_member_aliases(resolved_owner, members)
    return None
