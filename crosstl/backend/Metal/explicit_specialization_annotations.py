"""``@explicit_specialization`` annotations for imported Metal explicit specializations.

``template <> float twice<float>(float x)`` replaces the body of the primary ``twice`` for ``T = float``
([temp.expl.spec]/1). For the CrossGL and Metal targets the importer keeps the primary generic and emits the
specialization as the concrete function ``twice_u3cfloat_u3e``. The annotation names the primary and the
template arguments in CrossGL spelling, so generators instantiating ``twice`` for ``float`` (also from inside other
generic bodies) call the specialization instead of a clone of the primary's body.
"""

from __future__ import annotations

import re

from .conditional_types import split_template_arguments

__all__ = ["explicit_specialization_annotation"]

_TEMPLATE_NAME = re.compile(r"^(?P<base>[A-Za-z_]\w*)\s*<(?P<arguments>.*)>$")
_SPELLING = re.compile(r"[A-Za-z_]\w*|-?\d+")


def _generic_primaries(codegen, base):
    return [
        function
        for function in codegen.user_function_overloads_by_name.get(base, [])
        if getattr(function, "generics", None)
        or getattr(function, "template_parameters", None)
    ]


def explicit_specialization_annotation(codegen, func):
    """``@explicit_specialization(primary, args...)`` for an explicit specialization ``func``, or ``None``."""
    if getattr(func, "generics", None) or getattr(func, "template_parameters", None):
        return None
    match = _TEMPLATE_NAME.match(str(getattr(func, "name", "") or "").strip())
    if match is None:
        return None
    primaries = _generic_primaries(codegen, match.group("base"))
    if len(primaries) != 1:
        return None
    arguments = split_template_arguments(match.group("arguments"))
    if not arguments:
        return None
    spellings = []
    for argument in arguments:
        argument = argument.strip()
        if re.fullmatch(r"-?\d+", argument):
            spellings.append(argument)
            continue
        mapped = str(codegen.map_type(argument)).strip()
        if not _SPELLING.fullmatch(mapped):
            return None
        spellings.append(mapped)
    primary_name = codegen.sanitize_identifier(
        codegen.function_output_name(primaries[0])
    )
    return f"@explicit_specialization({primary_name}, {', '.join(spellings)})"
