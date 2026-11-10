"""Function templates that name a member type which an explicit struct specialization redefines.

A header may pick a type per template argument through a trait struct and its explicit specializations::

    template <typename T> struct OpMathType { using type = T; };
    template <> struct OpMathType<half> { using type = float; };
    template <typename T> using opmath_t = typename OpMathType<T>::type;

    template <typename T>
    T scale(T x, opmath_t<T> factor) {
      using op_T = opmath_t<T>;
      return static_cast<T>(static_cast<op_T>(x) * factor);
    }

``opmath_t<T>`` is ``T`` only for the arguments no explicit specialization names; ``scale<half>`` computes in
``float`` ([temp.expl.spec]/1). A CrossGL generic function names one type for every instantiation, so it cannot
carry such a member type: written as ``T`` it would compute ``scale<half>`` in ``half``. The constrained call pass
therefore materializes such a template as a carrier (constrained_call_carriers.py), as it does for a result type
that is a ``decltype`` of the parameters (dependent_result_templates.py). Each reachable call gets a concrete
instantiation whose member types name the concrete arguments, so member lookup selects the specialization they
match.

The scan is textual over the source with comments and literals masked:

* a *specialized struct* is a struct or class template that ``template <> struct S<...>`` specializes explicitly;
  its *member types* are the names any of its declarations introduces with ``using name =`` or ``typedef``;
* a *specialized alias* is an alias template whose target names a member type of a specialized struct instance
  (``S<...>::type``) or another specialized alias;
* a template is a carrier when its declaration or body names such a member type or such an alias with an argument
  list that mentions one of its template parameters.

A static data member such as ``Limits<T>::max`` is no member type and makes no carrier. A partial specialization
makes none either: the generic path matches it against the dependent arguments and refuses a member it cannot
decide (partial_struct_specializations.py).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .alias_template_types import find_alias_templates

__all__ = ["specialized_member_type_template_names"]

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_EXPLICIT_SPECIALIZATION_RE = re.compile(
    r"\btemplate\s*<\s*>\s*(?:struct|class)\s+([A-Za-z_]\w*)\s*<"
)
_TEMPLATE_ID_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*<")
_MEMBER_RE = re.compile(r"::\s*(?:template\s+)?([A-Za-z_]\w*)")
_USING_RE = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=")
_TYPEDEF_RE = re.compile(r"\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)\s*;")


def _member_type_names(preprocessor: Any, masked: str, name: str) -> set[str]:
    """Member type names that the declarations of struct ``name`` in ``masked`` introduce."""

    members: set[str] = set()
    for match in re.finditer(rf"\b(?:struct|class)\s+{re.escape(name)}\b", masked):
        brace = masked.find("{", match.end())
        stop = masked.find(";", match.end())
        if brace < 0 or 0 <= stop < brace:
            continue
        close = preprocessor._find_matching_delimiter(masked, brace, "{", "}")
        if close is None or close < 0:
            continue
        body = masked[brace + 1 : close]
        members.update(_USING_RE.findall(body))
        members.update(_TYPEDEF_RE.findall(body))
    return members


def _template_ids(preprocessor: Any, text: str) -> Iterable[tuple[str, str, str]]:
    """Yield ``(name, arguments, following)`` for each ``name<arguments>`` in ``text``."""

    for match in _TEMPLATE_ID_RE.finditer(text):
        angle_start = match.end() - 1
        angle_end = preprocessor._find_matching_angle(text, angle_start)
        if angle_end is None:
            continue
        yield (
            match.group(1),
            text[angle_start + 1 : angle_end],
            text[angle_end + 1 :].lstrip(),
        )


def _names_specialized_type(
    preprocessor: Any,
    text: str,
    member_types: Mapping[str, set[str]],
    aliases: set[str],
    parameters: Sequence[str],
) -> bool:
    """Whether ``text`` names a specialized member type or alias over one of ``parameters``."""

    for name, arguments, following in _template_ids(preprocessor, text):
        if name in aliases:
            pass
        elif name in member_types:
            member = _MEMBER_RE.match(following)
            if member is None or member.group(1) not in member_types[name]:
                continue
        else:
            continue
        if set(_IDENTIFIER_RE.findall(arguments)) & set(parameters):
            return True
    return False


def _specialized_alias_names(
    preprocessor: Any, code: str, member_types: Mapping[str, set[str]]
) -> set[str]:
    declarations = [
        alias
        for declared in find_alias_templates(preprocessor, code).values()
        for alias in declared
    ]
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for alias in declarations:
            if alias.name in names:
                continue
            if _names_specialized_type(
                preprocessor, alias.target, member_types, names, alias.parameters
            ):
                names.add(alias.name)
                changed = True
    return names


def specialized_member_type_template_names(
    preprocessor: Any, code: str, templates: Sequence[Any]
) -> set[str]:
    """Names of the templates in ``templates`` that name a member type an explicit specialization may redefine."""

    if not templates or "::" not in code:
        return set()
    masked = preprocessor._mask_comments_and_literals(code)
    specialized = sorted(set(_EXPLICIT_SPECIALIZATION_RE.findall(masked)))
    member_types = {
        name: members
        for name in specialized
        if (members := _member_type_names(preprocessor, masked, name))
    }
    if not member_types:
        return set()
    aliases = _specialized_alias_names(preprocessor, code, member_types)
    names: set[str] = set()
    for template in templates:
        declaration_start = template.span[1] - len(template.source)
        text = masked[declaration_start : template.span[1]]
        if _names_specialized_type(
            preprocessor, text, member_types, aliases, template.template_parameters
        ):
            names.add(template.name)
    return names
