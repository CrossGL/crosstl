"""Prove that an exact-match constrained template outranks a non-template overload that needs a conversion.

Constrained overload resolution refuses a call when another declaration of the callee is visible, because it
does not rank the two. One case can be ranked without the full rules ([over.match.best]/2.1,
[over.ics.rank]/3.2):

* every parameter of the enabled template candidate is taken by value and, after substitution, has exactly the
  type of its argument, so each implicit conversion sequence is the identity, which no other sequence beats;
* the competitor takes at least one argument by value as a different built-in scalar or vector type, so that
  argument needs a promotion or a conversion (or the competitor is not viable).

Then the template candidate is the better function, whatever the competitor's other parameters are. The
non-template tie-breaker ([over.match.best]/2.4) applies only when no argument tells the two apart, which the
second condition rules out. Other shapes stay unproven, so the call is still refused.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["exact_template_outranks"]

# Spellings of one type. Equal entries must be the same type; a pair that is
# merged here wrongly can only make the proof fail.
_CANONICAL_SCALARS = {
    "int32_t": "int",
    "uint32_t": "uint",
    "int16_t": "short",
    "uint16_t": "ushort",
    "int8_t": "char",
    "uint8_t": "uchar",
    "int64_t": "long",
    "uint64_t": "ulong",
    "float16_t": "half",
    "bfloat16_t": "bfloat",
}
_KNOWN_SCALARS = frozenset(
    {
        "bool",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "half",
        "bfloat",
        "float",
    }
)


def _builtin_value_type(preprocessor, type_text: str | None) -> str | None:
    """``base`` or ``baseN`` for a by-value built-in scalar or vector type, else None."""
    if not type_text or "&" in type_text or "*" in type_text or "[" in type_text:
        return None
    decomposed = preprocessor._scalar_and_width(type_text)
    if decomposed is None:
        return None
    base, width = decomposed
    if base in _CANONICAL_SCALARS:
        base = _CANONICAL_SCALARS[base]
    elif base not in _KNOWN_SCALARS:
        return None
    return base if width == 1 else f"{base}{width}"


def _parameter_value_type(preprocessor, parameter: str) -> str | None:
    type_text = preprocessor._normalize_function_parameter_type_text(parameter)
    if not type_text or "&" in type_text or "*" in type_text:
        return None
    return _builtin_value_type(
        preprocessor, preprocessor._normalize_inferred_type(type_text)
    )


def exact_template_outranks(
    preprocessor,
    template_parameters: str | None,
    bindings,
    argument_types: Sequence[str | None],
    competitor_parameters: Sequence[str],
) -> bool:
    """Whether the enabled template candidate is better than the competing declaration."""
    if template_parameters is None:
        return False
    parameters = [
        parameter
        for parameter in preprocessor._split_top_level_commas(template_parameters)
        if parameter.strip()
    ]
    if len(parameters) != len(argument_types) or len(competitor_parameters) < len(
        argument_types
    ):
        return False
    substitutions = {name: str(value) for name, value in dict(bindings).items()}
    arguments = [
        _builtin_value_type(preprocessor, argument_type)
        for argument_type in argument_types
    ]
    if any(argument is None for argument in arguments):
        return False
    for parameter, argument in zip(parameters, arguments):
        substituted = preprocessor._replace_identifiers(parameter, substitutions)
        if _parameter_value_type(preprocessor, substituted) != argument:
            return False
    for parameter, argument in zip(competitor_parameters, arguments):
        competitor = _parameter_value_type(preprocessor, parameter)
        if competitor is not None and competitor != argument:
            return True
    return False
