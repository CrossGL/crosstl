"""Argument types as template argument deduction compares them.

Deduction matches a parameter type against the type of the call argument
([temp.deduct.call]).  Two spellings of the same type must therefore compare
equal: a struct-scoped member alias such as ``Frag_float_8_8::frag_type`` denotes
the type it names, and a Metal vector shorthand such as ``half2`` is
``metal::vec<half, 2>`` (Metal Shading Language Specification, section 2.2,
Vector Data Types).
"""

from __future__ import annotations

import re
from typing import Any

_VECTOR_SHORTHAND_RE = re.compile(
    r"(bool|char|uchar|short|ushort|int|uint|long|ulong|half|float|bfloat)([234])"
)
_VECTOR_TEMPLATE_RE = re.compile(r"((?:::)?(?:metal::)?)vec<.*>")


def deduction_argument_type(
    preprocessor: Any,
    parameter_type: str,
    argument_type: str,
    structs_by_name: dict[str, Any] | None,
) -> str:
    """Return ``argument_type`` spelled so it can be matched to ``parameter_type``."""
    resolved = canonicalize_qualified_argument_type(
        preprocessor, argument_type, structs_by_name
    )
    return vector_argument_for_parameter(parameter_type, resolved)


def vector_argument_for_parameter(parameter_type: str, argument_type: str) -> str:
    """Spell a vector shorthand argument as the ``vec<T, N>`` form a parameter uses.

    ``thread vec<U, 2>&`` deduces ``U = half`` from a ``half2`` argument.  The
    expansion uses the parameter's own qualification, and any other argument is
    returned unchanged.
    """
    shorthand = _VECTOR_SHORTHAND_RE.fullmatch(argument_type.strip())
    template = _VECTOR_TEMPLATE_RE.fullmatch(parameter_type.strip())
    if shorthand is None or template is None:
        return argument_type
    return f"{template.group(1)}vec<{shorthand.group(1)},{shorthand.group(2)}>"


_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")


def canonicalize_qualified_argument_type(
    preprocessor: Any,
    type_text: str,
    structs_by_name: dict[str, Any] | None,
) -> str:
    """Return ``type_text`` with its struct-scoped member type names resolved.

    Template argument deduction compares a parameter type with the type of the
    argument ([temp.deduct.call]), and a name such as ``Frag_float_8_8::frag_type``
    denotes the type its member alias names.  An argument typed through the
    return type of another struct's method keeps such a qualified name, so it is
    resolved here through the structs it names before it is matched.  Only the
    structs whose names occur in ``type_text`` take part, and a name that does
    not resolve is returned unchanged for the existing diagnostics.
    """
    if "::" not in type_text or not structs_by_name:
        return type_text
    resolved = type_text
    # Each pass resolves the owners named so far; a member alias whose target
    # names a further struct is resolved by the next pass.
    for _ in range(8):
        if "::" not in resolved:
            break
        named = _structs_named_in(preprocessor, resolved, structs_by_name)
        if not named:
            break
        updated = preprocessor._canonicalize_qualified_struct_type_aliases(
            resolved, named
        )
        updated = preprocessor._normalize_template_argument_text(updated)
        if not updated or updated == resolved:
            break
        resolved = updated
    return resolved


def _structs_named_in(
    preprocessor: Any, type_text: str, structs_by_name: dict[str, Any]
) -> list[Any]:
    identifiers = set(_IDENTIFIER_RE.findall(type_text))
    named: list[Any] = []
    for name in sorted(structs_by_name):
        struct = structs_by_name[name]
        owners = {struct.name}
        specialization = preprocessor._materialized_struct_specializations.get(
            struct.name
        )
        if specialization is not None:
            owners.add(specialization[0])
        if owners & identifiers:
            named.append(struct)
    return named
