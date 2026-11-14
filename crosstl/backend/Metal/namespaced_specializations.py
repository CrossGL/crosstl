"""Namespace placement of materialized Metal function template specializations.

The Metal importer turns each function template a program calls with concrete arguments into an ordinary function,
appends that function after the translation unit's last declaration and renames the calls. A template declared in
a namespace looks up its names in that namespace (C++ [basic.lookup.unqual], [temp.res]): a type alias, alias
template or constant declared beside it is visible in its body and signature without qualification. Appended at
global scope, the specialization loses that scope, so ``using real_t = float;`` in ``namespace c10`` leaves
``real_t`` undeclared in the specialization.

The specialization is therefore declared in reopened definitions of the template's namespace, which restores the
same lookup scope, and every renamed call names it by its fully qualified name, which resolves from any scope. A
template that is a member of a structure keeps its existing placement.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

__all__ = [
    "place_in_namespace",
    "specialization_namespaces",
    "specialization_reference",
]


def _components(namespace: str) -> tuple[str, ...]:
    return tuple(
        part.strip() for part in str(namespace or "").split("::") if part.strip()
    )


def place_in_namespace(namespace: str, declaration: str) -> tuple[str, int]:
    """``declaration`` inside reopened definitions of ``namespace``.

    Returns the text and the offset of ``declaration`` within it. The global namespace leaves the declaration
    unchanged at offset 0.
    """

    components = _components(namespace)
    if not components:
        return declaration, 0
    prefix = "".join(f"namespace {component} {{\n" for component in components)
    suffix = "\n" + "}\n" * len(components)
    return prefix + declaration.rstrip() + suffix, len(prefix)


def specialization_reference(namespace: str, name: str) -> str:
    """The name that calls ``name`` declared in ``namespace`` from any scope."""

    components = _components(namespace)
    if not components:
        return name
    return "::" + "::".join((*components, name))


def specialization_namespaces(
    templates: Iterable[object],
    struct_spans: Sequence[tuple[int, int]],
) -> Mapping[tuple[int, int], str]:
    """The namespace that receives each template's specializations, keyed by template span.

    Structure members map to the global namespace, their existing placement; free templates map to the namespace
    that declares them.
    """

    placements = {}
    for template in templates:
        span = tuple(template.span)
        namespace = str(getattr(template, "namespace", "") or "")
        if namespace and any(start <= span[0] < end for start, end in struct_spans):
            namespace = ""
        placements[span] = namespace
    return placements
