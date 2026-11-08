"""Constrained function templates that no code outside unused templates names.

A Metal library header often declares a family of overloads selected by ``enable_if``::

    template <typename T, enable_if_t<is_metal_atomic<T>, bool> = true>
    T load(device holder<T>* object, size_t offset) { ... }

    template <typename T, enable_if_t<!is_metal_atomic<T>, bool> = true>
    T load(device holder<T>* object, size_t offset) { ... }

A function template that nothing names is never instantiated, so it contributes no code to the program
([temp.inst]/4). The constrained call pass binds every call of such a template in concrete code to the
instantiation that overload resolution selects, and removes the templates it consulted. The overloads that no
concrete code calls stay. A CrossGL generic function has no constraint, so the importer would turn the family into
several unconstrained generics with one signature, an ambiguous and unfaithful program, or refuse the source when a
body needs a choice that only the constraint decides, such as the partial specialization ``holder<T>`` names.

:func:`prune_unreferenced_constrained_templates` removes those overloads before the importer reads the source:

* The *live* names are the function names that code outside function templates and their explicit
  specializations spells, and, transitively, the names that templates and specializations of a live name spell.
  A name is matched textually, so any mention, a call, prototype, address or same-named member, keeps it live.
* Of the names that are not live, it removes those with a constrained template, and the names whose templates or
  specializations spell a removed name, since their generic bodies would call an overload whose contract the
  importer cannot express. Explicit specializations of a removed name go with it ([temp.expl.spec]/1).

Unconstrained templates that nothing names stay, as before, because CrossGL can express them as generics. Member
templates of aggregates are not considered; their bodies count as code outside function templates.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Tuple

__all__ = ["prune_unreferenced_constrained_templates"]

Span = Tuple[int, int]


def _free_templates(preprocessor: Any, code: str) -> list[Any]:
    templates = preprocessor._find_template_functions(code)
    if not templates:
        return []
    aggregate_spans = sorted(
        [struct.span for struct in preprocessor._find_concrete_struct_definitions(code)]
        + [struct.span for struct in preprocessor._find_template_structs(code)]
    )
    return [
        template
        for template in templates
        if preprocessor._containing_span(template.span[0], aggregate_spans) is None
    ]


def _referenced_names(
    masked: str, spans: Iterable[Span], pattern: re.Pattern, names: set[str]
) -> set[str]:
    found: set[str] = set()
    for start, end in spans:
        for match in pattern.finditer(masked, start, end):
            if match.group(0) in names:
                found.add(match.group(0))
    return found


def _closure(seeds: set[str], edges: dict[str, set[str]]) -> set[str]:
    reached = set(seeds)
    pending = sorted(seeds)
    while pending:
        name = pending.pop()
        for target in sorted(edges.get(name, ())):
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return reached


def prune_unreferenced_constrained_templates(preprocessor: Any, code: str) -> str:
    """Return ``code`` without the constrained free function templates that only unused code names."""

    templates = _free_templates(preprocessor, code)
    constrained_names = {
        template.name for template in templates if template.template_constraints
    }
    if not constrained_names:
        return code
    spans_by_name: dict[str, list[Span]] = {}
    for template in templates:
        spans_by_name.setdefault(template.name, []).append(tuple(template.span))
    specializations = preprocessor._find_explicit_template_function_specializations(
        code, include_namespaced=True
    )
    for specialization in specializations.values():
        spans_by_name.setdefault(str(specialization["name"]), []).append(
            tuple(specialization["span"])
        )
    names = set(spans_by_name)
    all_spans = sorted({span for spans in spans_by_name.values() for span in spans})
    masked = preprocessor._mask_comments_and_literals(code)
    pattern = re.compile(r"(?<![\w$])[A-Za-z_]\w*")

    outside: set[str] = set()
    position = 0
    for start, end in all_spans:
        if start > position:
            outside |= _referenced_names(masked, [(position, start)], pattern, names)
        position = max(position, end)
    outside |= _referenced_names(masked, [(position, len(masked))], pattern, names)

    edges = {
        name: _referenced_names(masked, spans, pattern, names) - {name}
        for name, spans in spans_by_name.items()
    }
    live = _closure(outside, edges)
    unused = names - live
    removed = constrained_names & unused
    changed = bool(removed)
    while changed:
        changed = False
        for name in sorted(unused - removed):
            if edges[name] & removed:
                removed.add(name)
                changed = True
    if not removed:
        return code
    # A declaration nested in another one goes with its enclosing declaration or stays with it.
    removals = sorted(
        {
            (start, end, "")
            for name in removed
            for start, end in spans_by_name[name]
            if not any(
                (outer_start, outer_end) != (start, end)
                and outer_start <= start
                and end <= outer_end
                for outer_start, outer_end in all_spans
            )
        }
    )
    return preprocessor._apply_text_replacements(code, removals)
