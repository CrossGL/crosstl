"""Select the function-template overload named by an explicit instantiation.

Metal (C++) allows several function templates to share one name when their
template parameter lists differ.  An explicit instantiation such as
``template [[host_name("k")]] [[kernel]] void f<float, mask_t, 3>(...);``
names exactly one of them: the overload whose parameter count and parameter
kinds (type versus non-type) accept the spelled arguments.  Picking the wrong
overload binds a type to a value parameter (or the reverse) and produces
meaningless code, so selection here only proves incompatibility from facts
that are certain and never guesses between overloads that both still fit.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

_VALUE_LITERAL_RE = re.compile(
    r"[-+]?(?:0[xX][0-9A-Fa-f']+|0[bB][01']+|\d[\d']*(?:\.\d*)?(?:[eE][-+]?\d+)?"
    r"|\.\d+(?:[eE][-+]?\d+)?)[uUlLfFhH]*"
)
_DECLARED_TYPE_RE = re.compile(
    r"\b(?:struct|class|union|enum(?:\s+(?:class|struct))?)\s+([A-Za-z_]\w*)"
    r"|\busing\s+([A-Za-z_]\w*)\s*="
)
_TYPEDEF_RE = re.compile(r"\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)\s*;")
_BUILTIN_TYPE_NAMES = frozenset({"void", "bool", "size_t", "ptrdiff_t"})


def declared_type_names(code: str) -> frozenset[str]:
    """Return the struct, class, union, enum, alias and typedef names in ``code``."""
    names = {
        match.group(1) or match.group(2) for match in _DECLARED_TYPE_RE.finditer(code)
    }
    names.update(match.group(1) for match in _TYPEDEF_RE.finditer(code))
    return frozenset(name for name in names if name)


def _argument_kind(
    preprocessor: Any, argument: str, type_names: frozenset[str]
) -> str | None:
    """Return ``"type"`` or ``"value"`` when the argument kind is certain."""
    text = argument.strip()
    if not text:
        return None
    if text in {"true", "false"} or _VALUE_LITERAL_RE.fullmatch(text):
        return "value"
    unqualified = re.sub(r"^(?:::)?(?:metal::)?", "", text)
    if (
        unqualified in _BUILTIN_TYPE_NAMES
        or unqualified in type_names
        or preprocessor._is_metal_scalar_or_vector_type(text)
    ):
        return "type"
    if re.fullmatch(r"(?:const\s+)?[A-Za-z_][\w:]*(?:\s*<.*>)?\s*[*&]+", text):
        return "type"
    return None


def _accepts(
    preprocessor: Any,
    template: Any,
    arguments: Sequence[str],
    type_names: frozenset[str],
) -> bool:
    parameters = list(template.template_parameters)
    variadic = set(template.variadic_template_parameters)
    if len(arguments) > len(parameters) and not (
        parameters and parameters[-1] in variadic
    ):
        return False
    defaults = template.template_parameter_defaults
    for parameter in parameters[len(arguments) :]:
        if parameter not in defaults and parameter not in variadic:
            return False
    value_parameters = template.template_parameter_types
    for index, argument in enumerate(arguments):
        parameter = parameters[min(index, len(parameters) - 1)]
        kind = _argument_kind(preprocessor, argument, type_names)
        if kind is None:
            continue
        expected = "value" if parameter in value_parameters else "type"
        if kind != expected:
            return False
    return True


def select_instantiated_overload(
    preprocessor: Any,
    type_names: frozenset[str],
    candidates: Sequence[Any],
    arguments: Sequence[str],
) -> Any | None:
    """Return the overload an explicit instantiation names, or ``None``.

    ``type_names`` comes from :func:`declared_type_names` for the source that
    holds the templates.  A single candidate is returned unchanged so the existing materialization
    diagnostics still own malformed instantiations.  With several overloads,
    the ones whose arity or certain argument kinds cannot accept ``arguments``
    are discarded.  ``None`` means no overload accepts the arguments.  When more
    than one overload still fits, the last declaration is kept, which matches
    the single-name lookup this selection replaces.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    accepted = [
        candidate
        for candidate in candidates
        if _accepts(preprocessor, candidate, arguments, type_names)
    ]
    if not accepted:
        return None
    return accepted[-1]
