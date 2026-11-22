"""Function templates that use a struct template with member functions at their own template arguments.

A functor written as a struct template is usually instantiated inside another template::

    template <typename T> struct Less {
      bool operator()(T a, T b) const thread { return a < b; }
    };

    template <typename T>
    bool less(T a, T b) { Less<T> op; return op(a, b); }

CrossGL structs carry data only. Member lowering (preprocessing/struct_member_lowering.py) turns the members of each
concrete struct, including the ones the struct-template pass materializes (``Less_float``), into free functions, but
``Less<T>`` in a generic body names no concrete struct: the CrossGL generic path would keep ``generic<T> struct
Less {}`` without its operator and a call ``op(a, b)`` that nothing defines. The constrained call pass therefore
materializes such a template as a carrier (constrained_call_carriers.py), as it does for a template whose result
type is a ``decltype`` of its parameters (dependent_result_templates.py). Each reachable call gets a concrete
instantiation, whose ``Less<float>`` the struct-template pass materializes and member lowering lowers.

The scan is textual over the source with comments and literals masked. A struct template has member functions when
any of its declarations, primary or specialization, declares one. A function template is a carrier when its
declaration or body names such a struct template with an argument list that mentions one of its template parameters.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .member_lowering_layers import lower_members_with_late_specializations
from .preprocessing.support import MetalStructMethodError

_STRUCT_KEYWORD_RE = re.compile(r"\b(?:struct|class)\b")


def _declares_member_functions(preprocessor: Any, template_struct: Any) -> bool:
    source = template_struct.source
    keyword = _STRUCT_KEYWORD_RE.search(source)
    if keyword is None:
        return False
    for struct in preprocessor._find_concrete_struct_definitions(
        source[keyword.start() :]
    ):
        if struct.methods or struct.template_methods:
            return True
    return False


def method_struct_template_names(
    preprocessor: Any, code: str, templates: Sequence[Any]
) -> set[str]:
    """Names of the templates in ``templates`` that name a struct template with member functions dependently."""

    return set(method_struct_template_uses(preprocessor, code, templates))


def method_struct_template_uses(
    preprocessor: Any, code: str, templates: Sequence[Any]
) -> dict[str, set[str]]:
    """Map each template in ``templates`` that names a struct template with member functions dependently to those
    struct templates."""

    if not templates or "template" not in code:
        return {}
    method_structs = sorted(
        {
            template_struct.name
            for template_struct in preprocessor._find_template_structs(code)
            if not template_struct.is_forward_declaration
            and _declares_member_functions(preprocessor, template_struct)
        }
    )
    if not method_structs:
        return {}
    masked = preprocessor._mask_comments_and_literals(code)
    use_re = re.compile(
        r"\b(" + "|".join(re.escape(name) for name in method_structs) + r")\s*<"
    )
    uses: dict[str, set[str]] = {}
    for template in templates:
        parameters = [name for name in template.template_parameters if name]
        if not parameters:
            continue
        parameter_re = re.compile(
            r"\b(?:" + "|".join(re.escape(name) for name in parameters) + r")\b"
        )
        declaration_start = template.span[1] - len(template.source)
        text = masked[declaration_start : template.span[1]]
        for use in use_re.finditer(text):
            opening = use.end() - 1
            closing = preprocessor._find_matching_angle(text, opening)
            if closing is None:
                continue
            if parameter_re.search(text, opening + 1, closing):
                uses.setdefault(template.name, set()).add(use.group(1))
    return uses


def lower_late_member_functions(preprocessor: Any, code: str) -> str:
    """Lower the member functions of concrete structs that a late materialization produced."""

    if not any(
        struct.methods or struct.template_methods
        for struct in preprocessor._find_concrete_struct_definitions(code)
    ):
        return code
    materialize = preprocessor._materialize_explicit_template_struct_instantiations
    return lower_members_with_late_specializations(
        preprocessor,
        code,
        lambda source: materialize(preprocessor._lower_struct_member_functions(source)),
        materialize,
        MetalStructMethodError,
    )
