"""Free functions of the same name declared in several namespaces of a Metal unit.

CrossGL has one global scope, so the importer emits every namespace member at program scope under its unqualified
name. Functions ``a::f``, ``b::f`` and ``::f`` with the same parameter list were emitted as three definitions of
``f`` (which no target accepts). With different parameter lists they became one overload set, so a call inside
``namespace a`` that C++ binds to ``a::f`` because it hides ``::f`` ([basic.lookup.unqual]) could silently bind to
``::f`` instead.

The importer already keeps source identity for user functions that share a name with a Metal library function: their
overloads are grouped per qualified name, each group gets its own program-scope name, and a call is bound through
namespace lookup. A name declared in more than one namespace takes the same path
(:func:`declared_in_several_namespaces`), so each declaration keeps a distinct program-scope name and every call binds
to the declaration lookup selects.

:func:`nominated_namespaces_in_scope` places the namespaces a using-directive nominates where [namespace.udir] puts
them: during unqualified lookup their members appear as if declared in the nearest enclosing namespace that contains
both the using-directive and the nominated namespace, and the using-directives of a nominated namespace are followed
transitively. ``namespace b { using namespace a; }`` therefore makes ``a::f`` a candidate next to ``::f`` in the
global scope of a lookup from ``b``, not ahead of it.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

__all__ = [
    "METAL_LIBRARY_MATH_FUNCTIONS",
    "declared_in_several_namespaces",
    "nominated_namespaces_in_scope",
]

# Math functions of the Metal standard library (Metal Shading Language Specification, section 6.5, Math Functions)
# that the importer passes through by name instead of lowering them as intrinsics. A user overload of one of them,
# such as ``metal::ldexp(bfloat, int)`` next to ``metal::fast::ldexp(bfloat, int)``, extends the library overload set
# like an overload of an intrinsic does, so ``ldexp(float, int)`` still binds to the library function.
METAL_LIBRARY_MATH_FUNCTIONS = frozenset(
    {
        "fdim",
        "fmax3",
        "fmedian3",
        "fmin3",
        "frexp",
        "ilogb",
        "ldexp",
        "max3",
        "median3",
        "min3",
        "modf",
        "nextafter",
        "pown",
    }
)


def declared_in_several_namespaces(overloads: Iterable[object]) -> bool:
    """Whether the free functions ``overloads`` (one unqualified name) are declared in more than one namespace."""
    namespaces = {
        str(getattr(function, "namespace", "") or "") for function in overloads
    }
    return len(namespaces) > 1


def _components(namespace: str) -> list[str]:
    return [part for part in str(namespace or "").split("::") if part]


def _nearest_common_namespace(first: str, second: str) -> str:
    common = []
    for left, right in zip(_components(first), _components(second)):
        if left != right:
            break
        common.append(left)
    return "::".join(common)


def nominated_namespaces_in_scope(
    lookup_namespace: str,
    scope: str,
    nominated_by: Callable[[str], Sequence[str]],
) -> list[str]:
    """The namespaces whose members unqualified lookup from ``lookup_namespace`` finds in the enclosing ``scope``.

    ``scope`` is ``lookup_namespace`` or one of its enclosing namespaces (``""`` is the global namespace).
    ``nominated_by(namespace)`` gives the namespaces nominated by the using-directives visible in ``namespace``.
    """
    scope_depth = len(_components(scope))
    origins = _components(lookup_namespace)
    found: list[str] = []
    for depth in range(len(origins), scope_depth - 1, -1):
        directive_scope = "::".join(origins[:depth])
        pending = list(nominated_by(directive_scope))
        followed = {directive_scope}
        while pending:
            target = pending.pop(0)
            if target in followed:
                continue
            followed.add(target)
            if (
                _nearest_common_namespace(directive_scope, target) == scope
                and target not in found
            ):
                found.append(target)
            pending.extend(nominated_by(target))
    return found
