"""Template argument deduction failures that rule out a member overload.

Deduction compares each function parameter type ``P`` that names a template
parameter with the argument type ``A``. When ``P`` is a template-id such as
``metal::vec<T, 8>`` or ``metal::bool_constant<b>``, ``A`` must be a
specialization of the same template, otherwise deduction fails and the
function template is not a candidate ([temp.deduct.call]/4,
[temp.deduct.type]/8-9). Two overloads that differ only in where a tag
parameter sits, such as ``f(vec<T, 8>, bool_constant<b>, vec<U, 8>)`` and
``f(vec<T, 8>, vec<T, 8>, bool_constant<b>)``, are told apart this way.

The check is conservative: it only reports a mismatch when both outer
template names are known, after mapping Metal's vector shorthands
(``float4`` is ``vec<float, 4>``) and ``bool_constant`` (an alias of
``integral_constant<bool, b>``) to their primary templates.
"""

from __future__ import annotations

import re
from typing import AbstractSet

_OUTER_TEMPLATE_RE = re.compile(
    r"\s*(?:typename\s+)?(?:(?:const|volatile|thread|threadgroup|device|constant)\s+)*"
    r"(?P<name>(?:::)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)\s*<"
)
_VECTOR_SHORTHAND_RE = re.compile(
    r"\s*(?:metal\s*::\s*)?"
    r"(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|float|bfloat)[234]\s*"
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_PRIMARY_TEMPLATES = {
    "bool_constant": "integral_constant",
}


def _primary_template_name(type_text: str) -> str | None:
    if _VECTOR_SHORTHAND_RE.fullmatch(type_text):
        return "vec"
    match = _OUTER_TEMPLATE_RE.match(type_text)
    if match is None:
        return None
    name = re.sub(r"\s+", "", match.group("name"))
    name = re.sub(r"^(?:::)?(?:metal::|std::)", "", name)
    # The result is only compared with another primary-template name and is
    # never emitted. An alias template maps to the template it names; any
    # other template-id is a specialization of its own primary template.
    if name in _PRIMARY_TEMPLATES:
        return _PRIMARY_TEMPLATES[name]
    return name


def deduction_pattern_mismatch(
    declared_type: str,
    argument_type: str,
    deducible_parameters: AbstractSet[str],
) -> bool:
    """Whether deducing ``declared_type`` from ``argument_type`` must fail.

    ``deducible_parameters`` are the template parameters that the call has not
    supplied explicitly. Only a template-id parameter type that names one of
    them can fail this way.
    """
    if not deducible_parameters.intersection(_IDENTIFIER_RE.findall(declared_type)):
        return False
    declared_template = _primary_template_name(declared_type)
    if declared_template is None:
        return False
    argument_template = _primary_template_name(argument_type)
    if argument_template is None:
        # A plain scalar or struct argument is not a specialization of any
        # template; only scalars are known to be non-template types here.
        return (
            re.fullmatch(
                r"\s*(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|"
                r"float|bfloat|int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|"
                r"int64_t|uint64_t|size_t)\s*",
                argument_type,
            )
            is not None
        )
    return argument_template != declared_template
