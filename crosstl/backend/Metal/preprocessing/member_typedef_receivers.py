"""Receivers declared through a member type alias of a concrete struct, as in ``Frag::mat_type m;``.

A qualified name whose nested-name-specifier denotes a class names a member of that class ([class.qual]), so
``Frag_float_8_8::mat_type`` denotes whatever ``typedef metal::simdgroup_matrix<float, 8, 8> mat_type;`` inside
``Frag_float_8_8`` names. When that chain of struct-scoped aliases ends at a Metal library template, the receiver
is a library object: its member calls (``m.thread_elements()``, ``tex.sample(...)``) are library operations the
frontend lowers, not struct methods. A chain that is missing, ambiguous, cyclic or too deep resolves to nothing, so
method lowering keeps its fail-closed diagnostic.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = ["member_typedef_target", "names_library_template"]

_QUALIFIED = re.compile(r"(?:::)?(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_MAX_ALIAS_STEPS = 16


def _owner_struct(owner: str, structs_by_name: Mapping[str, Any]) -> Any | None:
    """The struct ``owner`` (``Frag``, ``ns::Frag`` or ``::ns::Frag``) names, or ``None`` when none or several do."""
    owner = owner.lstrip(":")
    if "::" not in owner:
        return structs_by_name.get(owner)
    matches = [
        struct
        for struct in structs_by_name.values()
        if (getattr(struct, "qualified_name", "") or "").lstrip(":") == owner
    ]
    return matches[0] if len(matches) == 1 else None


def member_typedef_target(
    declared_type: str, structs_by_name: Mapping[str, Any]
) -> str | None:
    """The type ``Owner::member`` names through struct-scoped aliases, or ``None``.

    ``structs_by_name`` maps each concrete struct's name to its definition, whose ``type_aliases`` maps a member
    alias to the type text it names. The walk follows qualified targets (``typename Frag::mat_type``) and the
    owner's own aliases (``typedef mat_t mat_type;``) until the text is no longer a member alias.
    """
    spelled = re.sub(r"\s+", "", declared_type or "")
    if _QUALIFIED.fullmatch(spelled) is None:
        return None
    owner_text, member = spelled.rsplit("::", 1)
    owner = _owner_struct(owner_text, structs_by_name)
    seen: set[tuple[int, str]] = set()
    for _step in range(_MAX_ALIAS_STEPS):
        if owner is None:
            return None
        aliases = getattr(owner, "type_aliases", None) or {}
        if member not in aliases or (id(owner), member) in seen:
            return None
        seen.add((id(owner), member))
        target = re.sub(r"^typename\s+", "", str(aliases[member]).strip()).strip()
        compact = re.sub(r"\s+", "", target)
        if _QUALIFIED.fullmatch(compact) is not None:
            owner_text, member = compact.rsplit("::", 1)
            next_owner = _owner_struct(owner_text, structs_by_name)
            if next_owner is None:
                return target
            owner = next_owner
            continue
        if _IDENTIFIER.fullmatch(compact) is not None and compact in aliases:
            member = compact
            continue
        return target
    return None


def names_library_template(type_text: str, template_re: re.Pattern[str]) -> bool:
    """Whether ``type_text`` is exactly one Metal library template-id such as ``metal::simdgroup_matrix<float, 8, 8>``."""
    text = (type_text or "").strip()
    match = template_re.match(text)
    if match is None:
        return False
    depth = 0
    for index in range(match.end() - 1, len(text)):
        if text[index] == "<":
            depth += 1
        elif text[index] == ">":
            depth -= 1
            if depth == 0:
                return text[index + 1 :].strip() == ""
    return False
