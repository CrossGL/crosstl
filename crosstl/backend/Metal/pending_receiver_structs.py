"""Concrete structs whose member functions a function template still to be instantiated calls.

Member lowering (preprocessing/struct_member_lowering.py) moves the member functions of each concrete struct out of
the struct and rewrites every member call it finds. A member call in a function template body is dependent::

    template <typename T, int M, int N>
    void tile_add(thread Tile<T, M, N>& D, thread Tile<T, M, N>& A) { D.frag_at(0, 0) += A.frag_at(0, 0); }

``D.frag_at`` names a member of whichever specialization a call binds ``D`` to, so it is lowered in an instantiation
([temp.dep]). The passes that instantiate deduced calls run after member lowering for calls in member function bodies
(``tile_add(P, Q)`` in ``Block::run``) and for carrier templates (method_struct_carriers.py). Lowering every struct
first leaves the instantiation ``tile_add_float_1_1`` calling ``frag_at`` on a ``Tile_float_1_1`` without members.

:func:`pending_receiver_struct_names` names the concrete structs member lowering keeps whole instead: a struct with
member functions that a pending free function template names, either by its own name or, for a specialization the
struct-template pass materialized, by its primary template with an argument list that mentions the template's own
parameters. A template is pending when concrete code calls it, directly or through other pending templates, and its
body makes a member call. Once the instantiation round has replaced those calls, the next member-lowering pass lowers
these structs with the instantiated bodies. :func:`final_member_lowering` turns the deferral off for the last pass, so
a struct never reaches CrossGL with its members because a call to a template stayed generic.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .method_struct_carriers import method_struct_template_uses

_DEFERRAL_ATTRIBUTE = "_defer_pending_receiver_structs"
_DEFERRED_ATTRIBUTE = "_deferred_receiver_structs"
_MEMBER_CALL_RE = re.compile(
    r"(?:\.|->)\s*(?:template\s+)?[A-Za-z_]\w*\s*(?:<[^;{}()]*>\s*)?\("
)
# `Op op;` or `apply_helper<T, Op> helper;`: a local whose call `op(a, b)` is a call of its operator().
_LOCAL_DECLARATION_RE = re.compile(
    r"(?:^|[;{}])\s*(?:thread\s+)?[A-Za-z_][\w:]*\s*(?:<[^;{}()]*>)?\s+([A-Za-z_]\w*)\s*;"
)


def _unqualified(name: str) -> str:
    return name.rsplit("::", 1)[-1]


def _called_names(
    masked: str, call_re: re.Pattern[str], start: int, end: int
) -> list[tuple[int, str]]:
    return [
        (match.start(1), match.group(1))
        for match in call_re.finditer(masked, start, end)
    ]


def _makes_receiver_call(masked: str, start: int, end: int) -> bool:
    """Whether ``masked[start:end]`` calls a member, or calls a local declared there as a functor (``op(a, b)``)."""

    if _MEMBER_CALL_RE.search(masked, start, end):
        return True
    body = masked[start:end]
    for match in _LOCAL_DECLARATION_RE.finditer(body):
        if re.search(
            rf"(?<![\w.:]){re.escape(match.group(1))}\s*\(", body[match.end() :]
        ):
            return True
    return False


def _template_argument_names(masked: str, open_angle: int) -> set[str]:
    """Identifiers in the template argument list that starts at ``masked[open_angle] == '<'``."""

    depth = 0
    for index in range(open_angle, len(masked)):
        character = masked[index]
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
            if depth == 0:
                return set(re.findall(r"[A-Za-z_]\w*", masked[open_angle + 1 : index]))
        elif character in ";{}":
            break
    return set()


def pending_receiver_struct_names(
    preprocessor: Any,
    code: str,
    structs: Sequence[Any],
    template_spans: Sequence[tuple[int, int]],
) -> set[str]:
    """Names of the structs in ``structs`` whose members a pending function template call needs."""

    if not getattr(preprocessor, _DEFERRAL_ATTRIBUTE, True) or "template" not in code:
        return set()
    method_structs = [
        struct for struct in structs if struct.methods or struct.template_methods
    ]
    if not method_structs:
        return set()
    aggregate_spans = sorted(
        [struct.span for struct in preprocessor._find_concrete_struct_definitions(code)]
        + [struct.span for struct in preprocessor._find_template_structs(code)]
    )
    templates = [
        template
        for template in preprocessor._find_template_functions(code)
        if preprocessor._containing_span(template.span[0], aggregate_spans) is None
    ]
    if not templates:
        return set()
    masked = preprocessor._mask_comments_and_literals(code)
    primaries = method_struct_template_uses(preprocessor, code, templates)
    concrete_names = sorted({struct.name for struct in method_structs})
    concrete_re = re.compile(
        r"(?<![\w:])(" + "|".join(re.escape(name) for name in concrete_names) + r")\b"
    )
    uses: dict[str, set[str]] = {}
    for template in templates:
        if not _makes_receiver_call(masked, template.body_start, template.span[1]):
            continue
        named = set(primaries.get(template.name, ()))
        named.update(
            match.group(1)
            for match in concrete_re.finditer(
                masked, template.span[0], template.span[1]
            )
        )
        if named:
            uses.setdefault(template.name, set()).update(named)
    if not uses:
        return set()

    call_re = re.compile(
        r"(?<![\w.:])(?<!->)("
        + "|".join(re.escape(name) for name in sorted({t.name for t in templates}))
        + r")\s*[<(]"
    )
    # A member function of a concrete struct, a member template included, is lowered and instantiated in this pass,
    # so the template calls in its body are pending like calls in concrete functions.
    member_bodies = sorted(
        span for struct in method_structs for span in struct.member_function_body_spans
    )
    pending: set[str] = set()
    call_arguments: dict[str, set[str]] = {}
    for position, name in _called_names(masked, call_re, 0, len(masked)):
        if (
            preprocessor._containing_span(position, list(template_spans)) is None
            or preprocessor._containing_span(position, member_bodies) is not None
        ):
            pending.add(name)
            # `combine<T, Add<T>>(a, b)`: the instantiation calls members of the structs its arguments name.
            open_angle = masked.find("<", position + len(name))
            if (
                open_angle != -1
                and not masked[position + len(name) : open_angle].strip()
            ):
                call_arguments.setdefault(name, set()).update(
                    _template_argument_names(masked, open_angle)
                )
    worklist = sorted(pending)
    while worklist:
        name = worklist.pop()
        for template in templates:
            if template.name != name:
                continue
            for _position, callee in _called_names(
                masked, call_re, template.body_start, template.span[1]
            ):
                if callee not in pending:
                    pending.add(callee)
                    worklist.append(callee)
    needed = {
        _unqualified(struct_name)
        for name in pending
        for struct_name in uses.get(name, ())
    }
    for name, arguments in sorted(call_arguments.items()):
        if _reaches_receiver_call(name, templates, masked, call_re, uses):
            needed |= {_unqualified(argument) for argument in arguments}
    if not needed:
        return set()
    specializations = preprocessor._materialized_struct_specializations
    deferred = {
        struct.name
        for struct in method_structs
        if struct.name in needed
        or (
            struct.name in specializations
            and _unqualified(str(specializations[struct.name][0]).lstrip(":")) in needed
        )
    }
    # A kept struct's member bodies are lowered in the later pass, so the structs they name (`Op op; op(a, b)` with
    # `Op = Add<uint>`) keep their members for that pass too.
    by_name = {struct.name: struct for struct in method_structs}
    worklist = sorted(deferred)
    while worklist:
        start, end = by_name[worklist.pop()].span
        for match in concrete_re.finditer(masked, start, end):
            if match.group(1) not in deferred:
                deferred.add(match.group(1))
                worklist.append(match.group(1))
    if deferred:
        recorded = getattr(preprocessor, _DEFERRED_ATTRIBUTE, None)
        if recorded is None:
            recorded = set()
            setattr(preprocessor, _DEFERRED_ATTRIBUTE, recorded)
        recorded.update(deferred)
    return deferred


def _reaches_receiver_call(
    name: str,
    templates: Sequence[Any],
    masked: str,
    call_re: re.Pattern[str],
    uses: dict[str, set[str]],
) -> bool:
    """Whether template ``name``, or a template it calls, calls a member of a struct it names."""

    seen = {name}
    worklist = [name]
    while worklist:
        current = worklist.pop()
        if current in uses:
            return True
        for template in templates:
            if template.name != current:
                continue
            for _position, callee in _called_names(
                masked, call_re, template.body_start, template.span[1]
            ):
                if callee not in seen:
                    seen.add(callee)
                    worklist.append(callee)
    return False


def deferred_receiver_structs(preprocessor: Any) -> set[str]:
    """The structs member lowering has kept whole since the last :func:`reset_deferred_receiver_structs`."""

    return set(getattr(preprocessor, _DEFERRED_ATTRIBUTE, ()))


def reset_deferred_receiver_structs(preprocessor: Any) -> None:
    setattr(preprocessor, _DEFERRAL_ATTRIBUTE, True)
    setattr(preprocessor, _DEFERRED_ATTRIBUTE, set())


@contextmanager
def final_member_lowering(preprocessor: Any) -> Iterator[None]:
    """Lower every struct, including the ones a pending template call kept whole."""

    previous = getattr(preprocessor, _DEFERRAL_ATTRIBUTE, True)
    setattr(preprocessor, _DEFERRAL_ATTRIBUTE, False)
    try:
        yield
    finally:
        setattr(preprocessor, _DEFERRAL_ATTRIBUTE, previous)
