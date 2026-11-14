"""Unconstrained templates the constrained call pass must instantiate itself.

The constrained call pass binds calls in reachable concrete bodies to the ``enable_if``-constrained template that
overload resolution selects, and then removes the templates it consulted. A call in the body of an unconstrained
template, such as ``max(a, b)`` in::

    template <typename T> T wrap(T a, T b) { return max(a, b); }

has dependent argument types, so the pass cannot bind it where it stands. Left to the CrossGL generic path, an
instantiation ``wrap<float>`` would call whichever ``max`` the target resolves, typically the Metal standard
library function of that name, whose semantics differ from the source's constrained template.

Such a template is a carrier: the pass materializes its reachable calls like those of a constrained template whose
every constraint holds, and binds the calls in each materialized body. A template whose body calls a carrier is a
carrier too.

A template whose result type the generic path cannot carry is a carrier as well, whatever its body calls
(dependent_result_templates.py).

Only an unambiguous template can be a carrier: one free function template of its name, no constrained template of
the same name and no concrete overload. Anything else stays with the generic path, as before.
"""

from typing import AbstractSet, Any, Iterable, List, Sequence, Set

__all__ = ["constrained_call_carriers"]


def constrained_call_carriers(
    templates: Sequence[Any],
    constrained_names: Set[str],
    overloaded_names: Set[str],
    references: Iterable[Any],
    required_names: AbstractSet[str] = frozenset(),
) -> List[Any]:
    """The carrier templates among ``templates``, in source order.

    ``templates`` are the free function templates outside aggregates. ``overloaded_names`` are names that also
    declare a concrete free function. ``references`` are the call references of every candidate and constrained
    name, outside member calls. ``required_names`` name templates that are carriers by themselves.
    """
    counts: dict = {}
    for template in templates:
        counts[template.name] = counts.get(template.name, 0) + 1
    candidates = [
        template
        for template in templates
        if not template.template_constraints
        and counts[template.name] == 1
        and template.name not in constrained_names
        and template.name not in overloaded_names
    ]
    positions_by_name: dict = {}
    for reference in references:
        positions_by_name.setdefault(reference.name, []).append(reference.span[0])

    carriers: List[Any] = [
        template for template in candidates if template.name in required_names
    ]
    needed = set(constrained_names) | {template.name for template in carriers}
    changed = True
    while changed:
        changed = False
        for template in candidates:
            if template in carriers:
                continue
            if any(
                template.body_start <= position < template.span[1]
                for name in needed
                for position in positions_by_name.get(name, ())
            ):
                carriers.append(template)
                needed.add(template.name)
                changed = True
    return sorted(carriers, key=lambda template: template.span)
