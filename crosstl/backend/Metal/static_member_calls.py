"""Refuse a call to an undeclared static member of a concrete struct.

``Owner::name(args)`` whose owner names a struct calls that struct's static member function or constructs
its member type ``name`` ([class.qual], [expr.call]).  Member lowering rewrites every static member function
it materializes to a free helper, and the converter binds those helpers before it maps a call name.  A
qualified call that is still unbound therefore names a member that the struct does not declare, or one that
was never materialized.  Mapping such a name as a namespace-qualified function would drop ``Owner::`` and
emit a call to an undeclared ``name``, so the converter raises instead.

A qualified call whose qualifier names a template parameter of the generic function being converted, such as
``F::add(d, a)`` or ``Tile<T>::Frag_type::add(d, a)``, is a dependent call: its owner is known only once the
template is instantiated ([temp.dep.type], [temp.res]/6). CrossGL generic functions have no form for it, and
mapping the name would drop the qualifier and bind ``add`` to whatever unqualified function of that name the
program declares. The converter refuses such a call; instantiating the function template first (project
translation does, as does a call with explicit template arguments) gives the qualifier a concrete owner.
"""

from __future__ import annotations

import re
from typing import Any

from crosstl.translator.diagnostics import CODE_PREFIX
from crosstl.translator.refusals import MetalSourceRefusal

from .scoped_member_types import resolve_concrete_scoped_member_type


class MetalDependentMemberCallRefusal(MetalSourceRefusal):
    """A call through a qualifier that depends on a template parameter of the generic function being converted."""

    code = f"{CODE_PREFIX}metal-dependent-member-call-unsupported"
    project_diagnostic_code = "project.translate.metal-dependent-member-call"
    missing_capabilities = ("metal.dependent-member-call",)


_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def dependent_member_call_qualifier(converter: Any, name: Any) -> str | None:
    """The qualifier of the call ``name`` when it names a template parameter of the current generic function."""
    qualifier, separator, _member = str(name or "").lstrip(":").rpartition("::")
    if not separator or not qualifier:
        return None
    function = getattr(converter, "current_function", None)
    parameters = {
        entry[1]
        for entry in getattr(function, "template_parameters", None) or ()
        if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1]
    }
    if parameters & set(_IDENTIFIER.findall(qualifier)):
        return qualifier
    return None


def refuse_dependent_member_call(
    converter: Any, name: Any, source_location=None
) -> None:
    """Raise :class:`MetalDependentMemberCallRefusal` for a dependent qualified call."""
    qualifier = dependent_member_call_qualifier(converter, name)
    if qualifier is None:
        return
    function = getattr(converter, "current_function", None)
    member = str(name).lstrip(":").rpartition("::")[2]
    location = ""
    if isinstance(source_location, dict) and source_location.get("line"):
        location = f" at line {source_location['line']}"
    raise MetalDependentMemberCallRefusal(
        f"Metal input cannot carry the call '{qualifier}::{member}'{location} in generic function "
        f"'{getattr(function, 'name', '')}': its qualifier depends on a template parameter, so its owner is known "
        "only after instantiation, and a CrossGL generic function has no dependent member call "
        "([temp.dep.type]); instantiate the function first (project translation, or explicit template arguments)"
    )


def unresolved_static_member_call_owner(converter: Any, name: Any) -> tuple | None:
    """Return ``(owner, member)`` when ``name`` calls an unbound member of a struct.

    ``None`` means that ``name`` is unqualified, that its owner is not a struct the converter emits (a
    namespace such as ``metal::``) or has several declarations, or that ``Owner::name`` names a member
    type, which is a constructor.
    """
    text = str(name or "").lstrip(":")
    if "::" not in text or "<" in text:
        return None
    resolved = converter.static_struct_member_owner(text)
    if resolved is None:
        return None
    owner, member = resolved
    if (
        owner not in converter.struct_name_map
        or owner in getattr(converter, "ambiguous_struct_names", ())
        or text in converter.struct_name_map
        or text in getattr(converter, "type_aliases", {})
        or resolve_concrete_scoped_member_type(converter, text) is not None
    ):
        return None
    declaration = getattr(converter, "struct_declarations", {}).get(owner)
    if declaration is None:
        return None
    # A member type, alias or nested struct makes the call a construction.
    if any(
        getattr(alias, "name", None) == member
        for alias in getattr(declaration, "type_aliases", None) or ()
    ):
        return None
    return owner, member
