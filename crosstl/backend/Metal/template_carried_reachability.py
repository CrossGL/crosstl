"""Concrete functions that entry points reach through the bodies of function templates.

The constrained call pass binds calls only in the bodies of concrete functions that an entry point reaches. Plain
reachability follows calls between concrete functions and stops at a template, because a template's body is
instantiated later. A concrete function called only from a template body, such as ``h`` in::

    template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true> T max(T a, T b);
    int h(int a) { return max(a, 4); }
    template <typename T> T foo(T a) { return T(h(int(a))); }
    kernel void k(...) { ... foo(x) ... }

is reached all the same once ``foo`` is instantiated, and its calls have concrete argument types that the pass can
bind where they stand. Left unreached, they would be bound by whatever the target resolves after the pass has
removed the constrained templates.

A template counts as reached when a reached body names it in a call, including through explicit template arguments
(``foo<float>(x)``). The bodies of reached templates are scanned the same way, until nothing new is reached. Names are
matched without overload resolution, so the result over-approximates which functions run, as plain reachability does.
"""

import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

__all__ = ["template_carried_reachable_spans"]

Span = Tuple[int, int]

_CALL_NAME_RE = re.compile(r"(?<![\w.])(?<!->)([A-Za-z_]\w*)\s*(?=[(<])")


def _called_names(masked_code: str, start: int, end: int, names: Set[str]) -> Set[str]:
    """The names in ``names`` that ``masked_code[start:end]`` calls, outside member calls."""
    found: Set[str] = set()
    for match in _CALL_NAME_RE.finditer(masked_code, start, end):
        name = match.group(1)
        if name in names:
            found.add(name)
    return found


def template_carried_reachable_spans(
    masked_code: str,
    reachable_spans: Iterable[Span],
    functions: Sequence[Any],
    templates: Sequence[Any],
) -> List[Span]:
    """The spans of ``functions`` reached from ``reachable_spans`` directly or through template bodies, in source order.

    ``masked_code`` is the source with comments and literals masked. ``functions`` are the concrete function
    definitions, with ``name``, ``span`` and ``body_span``. ``templates`` are the free function templates, with
    ``name``, ``span`` and ``body_start``.
    """
    functions_by_name: Dict[str, List[Any]] = {}
    for function in functions:
        functions_by_name.setdefault(function.name, []).append(function)
    templates_by_name: Dict[str, List[Any]] = {}
    for template in templates:
        templates_by_name.setdefault(template.name, []).append(template)
    names = set(functions_by_name) | set(templates_by_name)

    reached: Set[Span] = set(reachable_spans)
    walked_templates: Set[Span] = set()
    pending: List[Span] = [
        function.body_span for function in functions if function.span in reached
    ]
    while pending:
        start, end = pending.pop()
        for name in sorted(_called_names(masked_code, start, end, names)):
            for template in templates_by_name.get(name, ()):
                if template.span not in walked_templates:
                    walked_templates.add(template.span)
                    pending.append((template.body_start, template.span[1]))
            for function in functions_by_name.get(name, ()):
                if function.span not in reached:
                    reached.add(function.span)
                    pending.append(function.body_span)
    return [function.span for function in functions if function.span in reached]
