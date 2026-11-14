"""Explicit function template specializations whose primary template the materializer removed.

An explicit specialization ``template <> void f<float>(...) { ... }`` is not a function of its own. It replaces the
body of its primary template for those arguments ([temp.expl.spec]/1) and is only reached through calls of the
primary. Project materialization for template-hostile targets turns every reachable call of the primary into a call
of a concrete copy, which already takes the specialization's body, and then removes the primary templates. A
specialization left behind keeps its source spelling: its body still calls the removed generic helpers by their
template names, and its parameter types are not canonicalized like those of the materialized copies. Generators
then emit that dead copy, and target compilers reject it.

:func:`prune_orphaned_explicit_specializations` removes such a specialization once nothing in the source names its
function any more: no remaining primary template, overload, prototype or call outside the specializations of that
name. A specialization whose name is still referenced is kept, because a later pass may still select it.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["prune_orphaned_explicit_specializations"]


def prune_orphaned_explicit_specializations(preprocessor: Any, code: str) -> str:
    """Return ``code`` without the explicit function specializations nothing else names."""
    specializations = preprocessor._find_explicit_template_function_specializations(
        code, include_namespaced=True
    )
    if not specializations:
        return code
    masked = preprocessor._mask_comments_and_literals(code)
    spans_by_name: dict[str, list[tuple[int, int]]] = {}
    for specialization in specializations.values():
        spans_by_name.setdefault(str(specialization["name"]), []).append(
            tuple(specialization["span"])
        )
    removals = []
    for name in sorted(spans_by_name):
        spans = spans_by_name[name]
        referenced = any(
            not any(start <= match.start() < end for start, end in spans)
            for match in re.finditer(rf"(?<!\w){re.escape(name)}\b", masked)
        )
        if referenced:
            continue
        removals.extend((start, end, "") for start, end in spans)
    if not removals:
        return code
    return preprocessor._apply_text_replacements(code, sorted(removals))
