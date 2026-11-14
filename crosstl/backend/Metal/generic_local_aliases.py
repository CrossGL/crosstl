"""Body-local type aliases that stay dependent inside a generic Metal function.

A function template body may name a dependent member alias such as
``using U = typename elem<device T*>::type;``.  When the member alias resolves
through the struct template (or one of its partial specializations) to a type
that is spelled only with the function's own type parameters, the alias is a
plain synonym of that generic type and every later use can name it directly.

When it cannot be resolved, the alias must not be dropped silently: later uses
would otherwise reach CrossGL as an undeclared name.  Such aliases are
recorded so that the first use raises :class:`MetalLocalTypeAliasResolutionError`.
"""

from __future__ import annotations

import re
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CV_QUALIFIERS = frozenset({"const", "volatile"})
_INTEGER_ARGUMENT = re.compile(r"[0-9]+[uU]?")


class MetalLocalTypeAliasResolutionError(ValueError):
    """A body-local alias of a dependent type is used but has no CrossGL type."""

    project_diagnostic_code = "project.translate.metal-local-type-alias-unresolved"

    def __init__(self, alias_name, alias_type, reason, source_location=None):
        self.alias_name = alias_name
        self.alias_type = alias_type
        self.reason = reason
        self.source_location = source_location
        location = f" at {source_location}" if source_location else ""
        super().__init__(
            f"Cannot resolve Metal body-local type alias '{alias_name}' = "
            f"'{alias_type}'{location}: {reason}"
        )


def generic_alias_constructor(converter: Any, name: str, mapped: Any) -> bool:
    """Whether the constructor ``name(...)`` names a generic body-local alias.

    ``mapped`` is the alias's CrossGL spelling; it is a generic constructor when
    it mentions one of the current function's type parameters.
    """
    if name not in converter.local_type_alias_names or not mapped:
        return False
    type_parameters = function_type_parameter_names(
        getattr(converter, "current_function", None)
    )
    return bool(set(_IDENTIFIER.findall(str(mapped))) & type_parameters)


def function_type_parameter_names(function: Any) -> frozenset[str]:
    """Return the type template parameter names of ``function``."""
    parameters = getattr(function, "template_parameters", None) or ()
    return frozenset(
        name
        for kind, name in parameters
        if name and str(kind).strip() in {"typename", "class"}
    )


def _strip_cv(text: str) -> str:
    words = text.split()
    while words and words[0] in _CV_QUALIFIERS:
        words.pop(0)
    while words and words[-1] in _CV_QUALIFIERS:
        words.pop()
    return " ".join(words)


def generic_local_alias_target(converter: Any, alias_type: Any) -> str | None:
    """Return the generic type a dependent body-local alias names, if any.

    The result is accepted only when every identifier it contains is a type
    parameter of the current function, or when the whole type maps to a
    CrossGL primitive, vector or matrix type (possibly over those parameters,
    as in ``vec<T, 4>``), or to a user struct template instance over such
    arguments.  Anything else, such as a remaining ``Owner::member``, a pointer
    or reference, or an unknown template, yields ``None``.
    """
    type_parameters = function_type_parameter_names(
        getattr(converter, "current_function", None)
    )
    if not type_parameters:
        return None
    original = str(alias_type or "").strip()
    if not original:
        return None
    resolved = converter.materialize_alias_template_type(original, required=False)
    resolved = _strip_cv(str(resolved or "").strip())
    if not _generic_spelling(converter, resolved, type_parameters):
        return None
    return resolved


def _generic_spelling(converter: Any, text: str, type_parameters: frozenset) -> bool:
    """Whether CrossGL can spell ``text`` inside the generic function as is."""
    text = _strip_cv(text.strip())
    if not text or any(token in text for token in ("::", "*", "&", "[", "(")):
        return False
    if text in type_parameters:
        return True
    if converter.crossgl_typedef_source_type(converter.map_type(text)) is not None:
        # Primitives, and vector or matrix templates such as ``vec<T, 4>``.
        return True
    base, arguments = converter.generic_type_parts(text)
    if not base or not arguments or base not in converter.struct_name_map:
        return False
    if not converter.struct_template_parameters.get(converter.map_struct_name(base)):
        return False
    # A user struct template over spellable arguments, emitted as a CrossGL generic struct.
    return all(
        _INTEGER_ARGUMENT.fullmatch(argument.strip())
        or _generic_spelling(converter, argument, type_parameters)
        for argument in arguments
    )


def bind_generic_local_alias(
    converter: Any, alias: Any, alias_type: Any, qualifiers: list
) -> bool:
    """Bind ``alias`` to its generic target, or record it as unresolved.

    Returns ``True`` when the alias was bound.  Aliases that do not mention a
    type parameter of the current function are left to the caller's existing
    handling, which materializes concrete user templates later.
    """
    name = getattr(alias, "name", None)
    if not name:
        return False
    target = generic_local_alias_target(converter, alias_type)
    if target is not None:
        converter.type_aliases[name] = target
        converter.type_alias_qualifiers[name] = list(qualifiers)
        converter.local_type_alias_names.add(name)
        return True
    type_parameters = function_type_parameter_names(
        getattr(converter, "current_function", None)
    )
    if set(_IDENTIFIER.findall(str(alias_type))) & type_parameters:
        converter.unresolved_local_type_aliases[name] = (
            str(alias_type),
            getattr(alias, "source_location", None),
        )
    return False


_UNQUALIFIED_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_:.])[A-Za-z_][A-Za-z0-9_]*")


def refuse_unresolved_local_alias_type(converter: Any, metal_type: Any) -> None:
    """Raise when ``metal_type`` names a recorded, unresolved body-local alias."""
    unresolved = getattr(converter, "unresolved_local_type_aliases", None)
    if not unresolved:
        return
    for name in _UNQUALIFIED_IDENTIFIER.findall(str(metal_type or "")):
        entry = unresolved.get(name)
        if entry is None:
            continue
        alias_type, source_location = entry
        raise MetalLocalTypeAliasResolutionError(
            name,
            alias_type,
            "the dependent alias names no type that CrossGL can spell inside "
            "the generic function",
            source_location=source_location,
        )
