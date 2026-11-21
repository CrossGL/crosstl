"""Lower the struct members that helpers instantiated after member lowering call.

A function template is instantiated where a call selects it ([temp.inst]).  When
the only call to ``tile_mad<T, N>`` sits in a member function body, the project
pipeline finds that call only after member lowering has turned the body into a
free function, so ``tile_mad_int_4`` is materialized after the members of
``Tile_int_4`` were lowered and dropped.  Its body still calls ``d.at(i)`` and
``b.template scaled<3>(i)``, which ``Tile_int_4`` no longer declares: Metal
output kept those member calls verbatim, and the other targets refused the
member template that was never instantiated.

:func:`lower_members_before_post_member_helpers` therefore detects helpers whose
member calls name a struct that member lowering has already lowered, and lowers
the original source again with every helper of that round defined up front, so
member lowering sees their call sites with the members still declared.  A
helper whose receiver struct is first materialized after lowering still finds
that struct's members, so it needs no retry.  The retries are bounded, and a
source that keeps exposing such helpers fails closed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Mapping

from crosstl.backend.Metal.member_lowering_layers import (
    _RESTORED_STATE,
    _restore,
    _snapshot,
)

MAX_POST_MEMBER_HELPER_RETRIES = 4

# ``receiver.member(`` or ``receiver.template member<...>(``, optionally through
# one subscript (``tiles[i].member(``).  A receiver that is itself a member
# (``self.tile.member(``) or a qualified name is not a local declaration.
_MEMBER_CALL_RE = re.compile(
    r"(?<![\w.:>])([A-Za-z_]\w*)\s*(?:\[[^\[\]{};]*\]\s*)?\.\s*"
    r"(?:template\s+)?([A-Za-z_]\w*)\s*[<(]"
)

HelperResult = tuple[str, list[dict[str, Any]], set[str], dict[Any, str]]


def _struct_members(preprocessor: Any, code: str) -> dict[str, frozenset[str]]:
    """Member function names of every concrete struct that still declares some."""
    return {
        struct.name: frozenset(
            method.name for method in (*struct.methods, *struct.template_methods)
        )
        for struct in preprocessor._find_concrete_struct_definitions(code)
        if struct.methods or struct.template_methods
    }


def _function_definitions(preprocessor: Any, code: str) -> list[Any]:
    spans = list(preprocessor._find_template_declaration_spans(code))
    return preprocessor._find_non_template_function_definitions(code, spans)


def _header(code: str, function: Any) -> str:
    return " ".join(code[function.span[0] : function.body_span[0]].split())


def _new_function_definitions(preprocessor: Any, before: str, after: str) -> list[Any]:
    """Concrete functions of ``after`` whose headers ``before`` does not define."""
    remaining = Counter(
        _header(before, function)
        for function in _function_definitions(preprocessor, before)
    )
    added = []
    for function in _function_definitions(preprocessor, after):
        header = _header(after, function)
        if remaining[header]:
            remaining[header] -= 1
        else:
            added.append(function)
    return added


def _member_calls(masked: str, function: Any):
    start, end = function.body_span
    for match in _MEMBER_CALL_RE.finditer(masked, start, end):
        yield match.start(1), match.group(1), match.group(2)


def _helpers_calling_lowered_members(
    preprocessor: Any,
    source_members: Mapping[str, frozenset[str]],
    text: str,
    helpers: list[Any],
    materialize: Callable[[str], str],
) -> list[str]:
    """Names of ``helpers`` that call a member of a struct lowered before them."""
    member_names = frozenset().union(*source_members.values())
    masked = preprocessor._mask_comments_and_literals(text)
    candidates = {
        helper.name
        for helper in helpers
        if any(
            member in member_names
            for _position, _receiver, member in _member_calls(masked, helper)
        )
    }
    if not candidates:
        return []
    # Helper parameters still spell template-ids such as ``Tile<int, 4>``; the
    # struct materializer resolves them to the concrete structs they name.
    # Probe without keeping the state it records.
    state = _snapshot(preprocessor)
    try:
        probe = materialize(text)
    finally:
        _restore(preprocessor, state)
    still_declared = _struct_members(preprocessor, probe)
    lowered = {
        name: members
        for name, members in source_members.items()
        if name not in still_declared
    }
    if not lowered:
        return []
    probe_masked = preprocessor._mask_comments_and_literals(probe)
    late: set[str] = set()
    for function in _function_definitions(preprocessor, probe):
        if function.name not in candidates or function.name in late:
            continue
        for position, receiver, member in _member_calls(probe_masked, function):
            declaration = preprocessor._visible_receiver_declaration_at(
                probe, receiver, position
            )
            if declaration is not None and member in lowered.get(
                declaration.declared_type, ()
            ):
                late.add(function.name)
                break
    return sorted(late)


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for name in _RESTORED_STATE:
        if name not in after:
            continue
        old, new = before.get(name), after[name]
        if isinstance(new, dict):
            old = old or {}
            delta[name] = {
                key: value
                for key, value in new.items()
                if key not in old or old[key] != value
            }
        elif isinstance(new, set):
            delta[name] = set(new) - set(old or ())
    return delta


def _apply_state(preprocessor: Any, delta: dict[str, Any]) -> None:
    for name, values in delta.items():
        current = getattr(preprocessor, name, None)
        if current is not None:
            current.update(values)


def _merge_state(merged: dict[str, Any], delta: dict[str, Any]) -> None:
    for name, values in delta.items():
        merged.setdefault(name, type(values)()).update(values)


def lower_members_before_post_member_helpers(
    preprocessor: Any,
    source: str,
    *,
    lower: Callable[[str], str],
    materialize_helpers: Callable[[str, Mapping[Any, str]], HelperResult],
    materialize: Callable[[str], str],
    known: Mapping[Any, str],
    error_type: type,
) -> HelperResult:
    """Lower members, then materialize helpers, so no helper calls a lowered member.

    ``lower`` performs member lowering on ``source``.  ``materialize_helpers``
    materializes the template helpers that the lowered text calls, skipping the
    keys it is given, and returns the text, the specialization records, the
    materialized template names and the key-to-name map.  ``materialize`` is the
    struct materializer.  When a helper that ``materialize_helpers`` emits calls
    a member of a struct that ``lower`` already lowered, ``source`` is lowered
    again with that round's helpers appended, their keys known and the state
    they recorded restored, so the helpers are lowered with the structs.
    """
    source_members = _struct_members(preprocessor, source)
    snapshot = _snapshot(preprocessor)
    definitions: list[str] = []
    defined_names: set[str] = set()
    helper_state: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    retry_known: Mapping[Any, str] = known
    late: list[str] = []
    for _attempt in range(MAX_POST_MEMBER_HELPER_RETRIES + 1):
        hinted = source
        roots = preprocessor._additional_reachability_roots
        if definitions:
            _restore(preprocessor, snapshot)
            _apply_state(preprocessor, helper_state)
            # Only lowered bodies call the helpers, so no entry point reaches
            # them yet; member lowering skips the call sites of dead functions.
            preprocessor._additional_reachability_roots = roots | defined_names
        try:
            if definitions:
                hinted = materialize(
                    source.rstrip() + "\n\n" + "\n\n".join(definitions) + "\n"
                )
            lowered = lower(hinted)
        finally:
            preprocessor._additional_reachability_roots = roots
        before = _snapshot(preprocessor)
        text, new_records, template_names, materialized_names = materialize_helpers(
            lowered, retry_known
        )
        helpers = _new_function_definitions(preprocessor, lowered, text)
        late = (
            _helpers_calling_lowered_members(
                preprocessor, source_members, text, helpers, materialize
            )
            if source_members and helpers
            else []
        )
        if not late:
            return text, records + new_records, template_names, materialized_names
        if defined_names.intersection(late):
            break
        _merge_state(helper_state, _state_delta(before, _snapshot(preprocessor)))
        for helper in helpers:
            definitions.append(text[helper.span[0] : helper.span[1]].strip())
            defined_names.add(helper.name)
        records.extend(new_records)
        retry_known = dict(materialized_names)
    raise error_type(
        "Metal member lowering left helpers calling members of structs it had "
        f"already lowered: {', '.join(late)}."
    )
