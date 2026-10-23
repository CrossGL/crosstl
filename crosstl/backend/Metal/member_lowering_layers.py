"""Lower member functions of structs that only an instantiated member body names.

A member function template body is instantiated only when a call site selects
it ([temp.inst]).  When that body names another class template specialization
that nothing else names, for example ``Frag<T>::template f<Op>(...)`` inside
``Tile<T, R>::g<Op>``, the specialization ``Frag<float>`` is first spelled by
the free function that member lowering emits.  A struct materializer that
runs after member lowering produces ``Frag_float``, but by then member lowering
has finished: the qualified call in the emitted body is never lowered, and the
template members of other structs that the new body calls (``Op::apply``) have
already been dropped because nothing called them during lowering.

:func:`lower_members_with_late_specializations` therefore detects concrete
structs with member functions that appear only after lowering, and lowers the
original source again with those specializations named up front.  Each retry
names at least one new specialization and the number of retries is bounded, so
a source that keeps exposing new specializations fails closed.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Materialization state that member lowering and struct materialization grow.
# A retry restores it so that the retry names and emits exactly what one pass
# over the hinted source would.
_RESTORED_STATE = (
    "_materialized_struct_specializations",
    "_materialized_struct_specialization_namespaces",
    "_materialized_struct_primary_templates",
    "_known_member_function_return_types",
    "_instantiated_template_member_calls",
    "_materialized_function_names",
    "_materialized_function_templates",
)
MAX_LATE_SPECIALIZATION_RETRIES = 8
HINT_PREFIX = "crosstl_member_layer_specialization_"
_HINT_DECLARATION_RE = re.compile(
    rf"\btypedef\s+[^;{{}}]*?\b{HINT_PREFIX}\d+\s*;[ \t]*\n?"
)


def _late_specializations(
    preprocessor: Any, code: str, known: set[str]
) -> list[tuple[str, str]]:
    """Return ``(concrete, spelling)`` for new specializations with members."""
    pending = set(preprocessor._materialized_struct_specializations) - known
    if not pending:
        return []
    late = []
    for struct in preprocessor._find_concrete_struct_definitions(code):
        if struct.name not in pending or not (
            struct.methods or struct.template_methods
        ):
            continue
        template_name, arguments = preprocessor._materialized_struct_specializations[
            struct.name
        ]
        spelling = preprocessor._template_specialization_signature(
            template_name.lstrip(":"), list(arguments)
        )
        namespace = preprocessor._materialized_struct_specialization_namespaces.get(
            struct.name
        )
        if namespace:
            spelling = f"{namespace}::{spelling}"
        late.append((struct.name, spelling))
    return sorted(set(late))


def _copy_container(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: list(item) if isinstance(item, list) else item
            for key, item in value.items()
        }
    if isinstance(value, (set, list)):
        return type(value)(value)
    return value


def _snapshot(preprocessor: Any) -> dict[str, Any]:
    return {
        name: _copy_container(getattr(preprocessor, name))
        for name in _RESTORED_STATE
        if hasattr(preprocessor, name)
    }


def _restore(preprocessor: Any, snapshot: dict[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(preprocessor, name, _copy_container(value))


def _with_hints(code: str, spellings: list[str]) -> str:
    if not spellings:
        return code
    declarations = "".join(
        f"typedef {spelling} {HINT_PREFIX}{index};\n"
        for index, spelling in enumerate(spellings)
    )
    return code + ("" if code.endswith("\n") else "\n") + declarations


def _without_hints(code: str, error_type: type) -> str:
    stripped = _HINT_DECLARATION_RE.sub("", code)
    if HINT_PREFIX in stripped:
        raise error_type(
            "Metal member lowering could not remove a late struct specialization "
            "request from the lowered source."
        )
    return stripped


def lower_members_with_late_specializations(
    preprocessor: Any,
    code: str,
    lower: Callable[[str], str],
    materialize: Callable[[str], str],
    error_type: type,
) -> str:
    """Run ``lower`` and retry it while it leaves members on new specializations.

    ``lower`` performs member lowering and whatever struct materialization the
    caller runs after it.  ``materialize`` is the struct materializer.  When
    ``lower`` produces a new concrete struct that still declares member
    functions, ``code`` is lowered again with a typedef that names that
    specialization, so the specialization exists, with its members, while
    member lowering runs.  ``error_type`` is raised when a retry makes no
    progress or the retries run out.
    """
    spellings: list[str] = []
    snapshot = _snapshot(preprocessor)
    for _attempt in range(MAX_LATE_SPECIALIZATION_RETRIES + 1):
        source = _with_hints(code, spellings)
        if spellings:
            _restore(preprocessor, snapshot)
            source = materialize(source)
        known = set(preprocessor._materialized_struct_specializations)
        result = lower(source)
        late = _late_specializations(preprocessor, result, known)
        if not late:
            return _without_hints(result, error_type) if spellings else result
        new = [spelling for _name, spelling in late if spelling not in spellings]
        if not new:
            names = ", ".join(name for name, _spelling in late)
            raise error_type(
                "Metal member lowering left member functions on struct "
                f"specializations that only lowered bodies name: {names}."
            )
        spellings.extend(new)
    raise error_type(
        "Metal member lowering kept exposing new struct specializations with "
        f"member functions after {MAX_LATE_SPECIALIZATION_RETRIES} retries."
    )
