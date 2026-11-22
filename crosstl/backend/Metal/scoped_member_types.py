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


class MetalUndeclaredMemberTypeError(ValueError):
    """``Owner::member`` names a member type that the concrete struct ``Owner`` does not declare."""

    project_diagnostic_code = "project.translate.metal-undeclared-member-type"

    def __init__(self, spelling, struct_name, member, function_name=None, owner=None):
        self.spelling = spelling
        self.struct_name = struct_name
        self.member = member
        self.function_name = function_name
        self.owner = owner
        where = f" in function '{function_name}'" if function_name else ""
        if struct_name is None:
            # The owner is a block-local alias used after the block that declares it ended.
            reason = (
                f"'{owner}' names no type there: the alias declaration of '{owner}' is out of scope "
                "([basic.scope.block])"
            )
        else:
            reason = f"struct '{struct_name}' declares no member type '{member}'"
        super().__init__(
            f"Metal input names the member type '{spelling}'{where}, but {reason}, so the program is "
            "ill-formed ([class.qual], [temp.res])"
        )


_QUALIFIER_KEYWORDS_RE = re.compile(r"\b(?:typename|template)\s+")
_DECLARATION_QUALIFIERS_RE = re.compile(
    r"^(?:(?:const|volatile|device|thread|threadgroup|constant|ray_data|object_data)\s+)+"
)


def refuse_undeclared_scoped_member_type(converter: Any, type_name: Any) -> None:
    """Raise when ``type_name`` is ``Owner::member`` and the concrete struct ``Owner`` lacks that member type.

    ``type_name`` is the type of a declaration. The importer used to pass such a spelling,
    ``Plain::missing_t<float>``, to every target, which none compiles, or to drop the owner of a name without
    template arguments. Only an owner that resolves to a struct the module
    declares is checked: a template parameter, a namespace or a library type has members the importer does not
    record. An owner with a base class the importer cannot follow is not checked either.
    """
    text = _QUALIFIER_KEYWORDS_RE.sub("", str(type_name or "")).strip().rstrip("*& ")
    text = _DECLARATION_QUALIFIERS_RE.sub("", text).strip()
    base_name, _arguments = converter.generic_type_parts(text)
    name = base_name or text
    if "<" in name or "::" not in name:
        return
    owner_text, member = (part.strip() for part in name.rsplit("::", 1))
    owner_text = owner_text.lstrip(":").strip()
    owner = _concrete_owner(converter, owner_text)
    function_name = getattr(getattr(converter, "current_function", None), "name", None)
    if owner is None:
        if _out_of_scope_local_alias(converter, owner_text):
            raise MetalUndeclaredMemberTypeError(
                f"{owner_text}::{member}", None, member, function_name, owner_text
            )
        return
    if _declares_member_type(converter, owner, member, set()) is not False:
        return
    raise MetalUndeclaredMemberTypeError(
        f"{owner_text}::{member}",
        getattr(owner, "name", owner_text),
        member,
        function_name,
        owner_text,
    )


def record_block_local_struct_alias(converter: Any, name: Any) -> None:
    """Remember that ``name`` was declared as a block-local alias of a struct.

    The importer forgets the alias when its block ends; this record outlives the block, so a later
    ``name::member`` can be told apart from an owner the importer never saw (a template parameter, a namespace).
    """
    if name:
        converter.__dict__.setdefault("block_local_struct_alias_names", set()).add(
            str(name)
        )


def _out_of_scope_local_alias(converter: Any, owner_text: str) -> bool:
    """Whether ``owner_text`` is a block-local struct alias whose block has ended and that names nothing else."""
    if owner_text not in getattr(converter, "block_local_struct_alias_names", ()):
        return False
    if (
        owner_text in (getattr(converter, "local_struct_type_aliases", None) or {})
        or owner_text in (getattr(converter, "type_aliases", None) or {})
        or owner_text in (getattr(converter, "struct_declarations", None) or {})
    ):
        return False
    from .generic_local_aliases import function_type_parameter_names

    struct_parameters = getattr(converter, "struct_template_parameters", None) or {}
    if owner_text in function_type_parameter_names(
        getattr(converter, "current_function", None)
    ) or any(owner_text in (names or ()) for names in struct_parameters.values()):
        return False
    prefix = f"{owner_text}::"
    declared = (
        *(getattr(converter, "struct_declarations", None) or {}),
        *(getattr(converter, "type_aliases", None) or {}),
    )
    # A namespace of that name declares the owner's members at namespace scope.
    return not any(str(name).startswith(prefix) for name in declared)


def _concrete_owner(converter: Any, owner_text: str) -> Any:
    """The struct declaration ``owner_text`` names, or ``None`` when it names no struct the module declares."""
    if not owner_text or owner_text.split("::", 1)[0] == "metal" or "<" in owner_text:
        return None
    declarations = getattr(converter, "struct_declarations", None) or {}
    ambiguous = getattr(converter, "ambiguous_struct_names", None) or set()
    local_owner = getattr(converter, "local_struct_type_aliases", {}).get(owner_text)
    candidates = [owner_text] if local_owner is None else [local_owner, owner_text]
    for candidate in candidates:
        resolved = converter.normalized_metal_type(
            converter.resolve_type_alias(candidate)
        )
        if resolved in ambiguous:
            return None
        if resolved in declarations:
            return declarations[resolved]
    return None


def _declares_member_type(
    converter: Any, struct_node: Any, member: str, seen: set
) -> bool | None:
    """Whether ``struct_node`` or a base declares member type ``member``; ``None`` when that is unknown."""
    name = getattr(struct_node, "name", None)
    if name in seen:
        return None
    seen.add(name)
    declared = {
        getattr(alias, "name", None)
        for alias in getattr(struct_node, "type_aliases", None) or ()
    }
    declared.update(
        getattr(nested, "name", None)
        for nested in getattr(struct_node, "nested_structs", None) or ()
    )
    declarations = getattr(converter, "struct_declarations", None) or {}
    if member in declared or f"{name}::{member}" in declarations:
        return True
    if any(
        not getattr(field, "name", None)
        for field in getattr(struct_node, "members", None) or ()
    ):
        # A member the parser read without a name may be a declaration it does not record, such as a member
        # template.
        return None
    for base in getattr(struct_node, "base_types", None) or ():
        base_node = _concrete_owner(converter, str(base).strip())
        if base_node is None:
            return None
        inherited = _declares_member_type(converter, base_node, member, seen)
        if inherited is not False:
            return inherited
    return False


def inherited_scoped_member_type(converter: Any, type_name: Any) -> str | None:
    """The type ``Owner::member`` names through a base or a nested struct of concrete ``Owner``, or ``None``.

    Qualified lookup of ``member`` in ``Owner`` also searches its base classes ([class.member.lookup]), and
    ``Owner::Inner::t`` names the member ``t`` of the nested struct ``Owner::Inner``. The importer resolved only an
    alias that ``Owner`` itself declares, and wrote any other such name without its owner (``scalar`` for
    ``Derived::scalar``), which names nothing in CrossGL. ``None`` leaves the spelling to the caller.
    """
    text = _QUALIFIER_KEYWORDS_RE.sub("", str(type_name or "")).strip()
    if "<" in text or "::" not in text:
        return None
    owner_text, member = (part.strip() for part in text.rsplit("::", 1))
    owner = _concrete_owner(converter, owner_text.lstrip(":").strip())
    if owner is None:
        return None
    return _member_type_target(converter, owner, member, set())


def _member_type_target(
    converter: Any, struct_node: Any, member: str, seen: set
) -> str | None:
    name = getattr(struct_node, "name", None)
    if name in seen:
        return None
    seen.add(name)
    aliases = [
        alias
        for alias in getattr(struct_node, "type_aliases", None) or ()
        if getattr(alias, "name", None) == member
    ]
    if aliases:
        if len(aliases) != 1 or getattr(aliases[0], "template_parameters", None):
            return None
        return str(aliases[0].alias_type).strip()
    nested = f"{name}::{member}"
    if nested in (getattr(converter, "struct_declarations", None) or {}):
        return nested
    for base in getattr(struct_node, "base_types", None) or ():
        base_node = _concrete_owner(converter, str(base).strip())
        if base_node is None:
            return None
        target = _member_type_target(converter, base_node, member, seen)
        if target is not None:
            return target
    return None
