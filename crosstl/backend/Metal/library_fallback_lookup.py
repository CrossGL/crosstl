"""Whether name lookup at a call of a constrained template also finds the Metal library function of that name.

The constrained call pass leaves a call whose constrained candidates are all disabled spelled as written, so that the
Metal standard library overload of the same name (``metal::max``, ``metal::isnan``) resolves it once the disabled
templates are removed. That is correct only when name lookup at the call would have found the library function in the
first place. Where it does, the library's overloads also compete with an enabled template
(``library_overloads.library_overload_precedence``). C++ lookup stops at the first scope that declares the name, and
SFINAE removes a candidate only after lookup has chosen the scope:

* An unqualified call inside namespace ``ns`` that finds ``ns::max`` never reaches the global scope, where
  ``using namespace metal;`` makes the library names appear ([namespace.udir]: they appear in the nearest enclosing
  namespace that contains both the directive and ``metal``, which is always the global namespace). With ``ns::max``
  disabled the call has no viable function.
* A qualified call such as ``::max(a, b)`` or ``ns::max(a, b)`` follows the using-directives of the named namespace
  only when that namespace declares no ``max`` itself ([namespace.qual]).
* An unqualified call that reaches the global scope finds the library function only through an active
  ``using namespace metal;``.

Argument-dependent lookup adds nothing for scalar and vector arguments, which have no associated namespaces, so the
caller applies these rules only to such calls. Templates declared in namespace ``metal`` itself sit beside the library
functions, and the caller counts a ``using metal::max;`` declaration anywhere in the source as making the library
visible.

Removing disabled templates and keeping the call would make such an ill-formed program bind the library function, so
the pass refuses it instead.
"""

import re
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = ["library_fallback_obstacle", "library_using_declaration_present"]

_LIBRARY_NAMESPACE = "metal"


def _common_namespace(first: str, second: str) -> str:
    """The innermost namespace that encloses both ``first`` and ``second``."""
    shared = []
    for left, right in zip(
        [part for part in first.split("::") if part],
        [part for part in second.split("::") if part],
    ):
        if left != right:
            break
        shared.append(left)
    return "::".join(shared)


def _enclosing_namespaces(namespace: str) -> Sequence[str]:
    components = [part for part in namespace.split("::") if part]
    return ["::".join(components[:length]) for length in range(len(components), -1, -1)]


_LIBRARY_USING_DECLARATION = re.compile(
    r"(?<![\w:])using\s+(?:::\s*)?" + _LIBRARY_NAMESPACE + r"\s*::\s*(\w+)\s*;"
)


@lru_cache(maxsize=8)
def _library_using_declarations(masked_code: str) -> frozenset:
    """The names of every ``using metal::<name>;`` in ``masked_code``, found in one scan.

    The constrained call pass asks once per call; scanning the whole unit for each name took minutes on large units.
    """
    if "using" not in masked_code:
        return frozenset()
    return frozenset(
        match.group(1) for match in _LIBRARY_USING_DECLARATION.finditer(masked_code)
    )


def library_using_declaration_present(masked_code: str, name: str) -> bool:
    """Whether ``masked_code`` declares ``using metal::<name>;`` (comments and literals masked)."""
    return name in _library_using_declarations(masked_code)


def _active_directives(directives: Iterable[Any], position: int) -> list:
    return [
        directive
        for directive in directives
        if directive.declaration_position <= position
        and directive.scope_start <= position < directive.scope_end
    ]


def library_fallback_obstacle(
    *,
    name: str,
    reference_namespace: Optional[str],
    call_namespace: str,
    position: int,
    candidate_namespaces: Iterable[str],
    directives: Iterable[Any],
    directive_targets: Callable[[Any], Sequence[str]],
) -> Optional[str]:
    """Why lookup at the call does not find ``metal::<name>``, or ``None`` when it finds it beside the templates.

    ``reference_namespace`` is the call's qualifier (``None`` when unqualified, ``"::"`` for a global qualifier).
    ``candidate_namespaces`` are the namespaces of the constrained templates that lookup found. ``directives`` are
    the using-directives of the source and ``directive_targets`` lists the namespaces a directive may nominate.
    """
    namespaces = sorted({namespace.lstrip(":") for namespace in candidate_namespaces})
    if not namespaces:
        return None
    if reference_namespace is not None:
        # The qualifier resolved to the namespace that declares the templates.
        if all(namespace == _LIBRARY_NAMESPACE for namespace in namespaces):
            return None
        scope = reference_namespace.lstrip(":")
        where = f"namespace '{scope}'" if scope else "the global namespace"
        return (
            f"qualified lookup in {where} finds only constrained '{name}' templates, "
            f"so it does not consider '{_LIBRARY_NAMESPACE}::{name}'"
        )

    active = _active_directives(directives, position)
    enclosing = set(_enclosing_namespaces(call_namespace))
    levels = set()
    for namespace in namespaces:
        if namespace in enclosing:
            levels.add(namespace)
        else:
            # Found through a using-directive: the names appear in the innermost namespace that encloses both the
            # directive and the namespace it nominates.
            levels.update(
                _common_namespace(namespace, directive.declaration_namespace)
                for directive in active
                if namespace in directive_targets(directive)
            )
    # Templates declared in namespace metal itself sit beside the library functions.
    hiding = [level for level in levels if level and level != _LIBRARY_NAMESPACE]
    if hiding:
        innermost = max(hiding, key=lambda level: (level.count("::"), level))
        return (
            f"unqualified lookup stops at namespace '{innermost}', whose constrained '{name}' templates hide "
            f"'{_LIBRARY_NAMESPACE}::{name}'"
        )
    if _LIBRARY_NAMESPACE in levels:
        return None

    if not any(
        _LIBRARY_NAMESPACE in directive_targets(directive) for directive in active
    ):
        return (
            f"no using-directive makes '{_LIBRARY_NAMESPACE}::{name}' visible where the constrained '{name}' "
            "templates are found"
        )
    return None
