"""Deduce the Metal function template arguments an explicit call leaves out.

An explicit template-argument list may name only the leading template parameters.
C++ deduces every parameter it does not supply from the types of the call
arguments ([temp.deduct.call]), and a declaration default applies only to a
parameter that deduction leaves unbound ([temp.deduct]/5).  A call such as
``offset<true>(ptr, n)`` against

.. code-block:: cpp

    template <bool scaled, typename T>
    void offset(const device T*& p, int n);

therefore binds ``T`` from the type of ``ptr``.

The comparison is textual over normalized Metal spellings.  A parameter type
whose shape does not match the argument type contributes nothing, and two
arguments that deduce different types for one parameter make the call
ill-formed, which the caller reports.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from crosstl.translator.refusals import TranslationRefusal

from .deduction_arguments import vector_argument_for_parameter

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")

# Qualifiers that do not take part in matching the parameter's type shape.  A
# reference parameter binds the referred-to type, top-level cv-qualifiers are
# ignored for deduction ([temp.deduct.call]/2-3), and Metal address spaces are
# carried separately from the declared type.
_IGNORED_QUALIFIERS = frozenset(
    {
        "const",
        "volatile",
        "device",
        "constant",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
        "ray_data",
        "object_data",
        "struct",
        "class",
        "typename",
    }
)


def _normalized_type(type_text: object) -> str:
    text = str(type_text or "")
    text = re.sub(r"\[\[[^\]]*\]\]", " ", text)
    text = re.sub(r"\bmetal::", "", text)
    words = [
        word
        for word in re.split(r"(\W)", text)
        if word.strip() and word not in _IGNORED_QUALIFIERS
    ]
    text = " ".join(words)
    text = re.sub(r"\s*([<>,*&\[\]()])\s*", r"\1", text)
    text = re.sub(r"\s*::\s*", "::", text)
    return text.rstrip("&").strip()


def _match_parameter_type(
    parameter_type: str,
    argument_type: str,
    pending: Iterable[str],
) -> dict[str, str] | None:
    """Return the bindings that make ``parameter_type`` spell ``argument_type``."""

    pending = set(pending)
    captures: list[str] = []
    pattern: list[str] = []
    position = 0
    for match in _IDENTIFIER_RE.finditer(parameter_type):
        pattern.append(re.escape(parameter_type[position : match.start()]))
        identifier = match.group(0)
        if identifier in pending:
            pattern.append(r"(.+?)")
            captures.append(identifier)
        else:
            pattern.append(re.escape(identifier))
        position = match.end()
    pattern.append(re.escape(parameter_type[position:]))
    if not captures:
        return None
    matched = re.fullmatch("".join(pattern), argument_type)
    if matched is None:
        return None
    bindings: dict[str, str] = {}
    for index, name in enumerate(captures, start=1):
        value = matched.group(index).strip()
        if not value or bindings.get(name, value) != value:
            return None
        bindings[name] = value
    return bindings


class TemplateDeductionConflict(TranslationRefusal):
    """Two call arguments deduce different arguments for one template parameter."""

    code = "project.translate.metal-template-deduction-conflict"

    def __init__(self, parameter_name: str, values: Sequence[str]):
        super().__init__(
            f"deduces conflicting arguments {', '.join(repr(v) for v in values)}"
        )
        self.parameter_name = parameter_name
        self.values = tuple(values)


def deduce_omitted_template_arguments(
    template_parameters: Sequence[tuple[str, str]],
    explicit_count: int,
    parameter_types: Sequence[object],
    argument_types: Sequence[str | None],
) -> dict[str, str]:
    """Deduce the template parameters after the first ``explicit_count``.

    ``parameter_types`` are the declared function parameter types and
    ``argument_types`` the Metal types of the call arguments, position by
    position.  A ``None`` argument type deduces nothing.  Raises
    :class:`TemplateDeductionConflict` when two arguments disagree.
    """

    pending = [name for _kind, name in template_parameters[explicit_count:] if name]
    if not pending:
        return {}
    bindings: dict[str, str] = {}
    for parameter_type, argument_type in zip(parameter_types, argument_types):
        if argument_type is None:
            continue
        pattern = _normalized_type(parameter_type)
        concrete = vector_argument_for_parameter(
            pattern, _normalized_type(argument_type)
        )
        matched = _match_parameter_type(pattern, concrete, pending)
        for name, value in sorted((matched or {}).items()):
            existing = bindings.get(name)
            if existing is not None and existing != value:
                raise TemplateDeductionConflict(name, (existing, value))
            bindings[name] = value
    return bindings
