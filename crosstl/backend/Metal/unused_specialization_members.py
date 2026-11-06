"""Drop the lowered members of class template specializations that nothing calls.

Implicitly instantiating a class template specialization instantiates the declarations of its member functions but
not their definitions; a member definition is instantiated only when it is used ([temp.inst]/3-4). A member body may
therefore be ill-formed for some template arguments, for example ``simd_prefix_inclusive_sum(x)`` with ``U = cplx``,
as long as the program never calls that member of that specialization.

Member lowering emits every non-template member of a materialized specialization such as ``Sum<cplx>`` as a free
helper ``Sum_cplx__scan_impl``. An unreferenced helper is code the source program never instantiates, and keeping it
asks every target to translate a body the source compiler never checked.
:func:`prune_unreferenced_specialization_members` removes those helpers once lowering has finished. A helper stays
whenever any text outside the removed helpers names it, including a prototype or a qualified ``Sum_cplx::member``
spelling. Operator members stay as well, because code generation binds them to expressions without naming them. A
source without an entry point is a library: nothing in it instantiates anything, and its helpers are left as lowered.
"""

import re
from typing import Any, Dict, List, Set, Tuple

_LOWERED_HEADER_RE = re.compile(
    r"(?m)^[ \t]*static[ \t]+inline\b[^;{}()]*?\b(?P<name>[A-Za-z_]\w*)\s*\("
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
_QUALIFIED_MEMBER_RE = re.compile(r"\b(?P<owner>[A-Za-z_]\w*)\s*::\s*(?P<member>\w+)")
_ENTRY_RE = re.compile(
    r"\b(?:kernel|vertex|fragment)\s+[A-Za-z_]|\[\[\s*(?:kernel|vertex|fragment|mesh|object)\s*\]\]"
)


def _specialization_member(name: str, specializations: List[str]) -> Tuple[str, str]:
    """The (specialization, member) a lowered helper name belongs to, or empty strings."""
    for specialization in specializations:
        prefix = f"{specialization}__"
        if name.startswith(prefix) and len(name) > len(prefix):
            member = name[len(prefix) :].split("__", 1)[0]
            if member and not member.startswith("operator"):
                return specialization, member
    return "", ""


def _lowered_definitions(
    preprocessor: Any, masked: str, specializations: List[str]
) -> Dict[str, List[Tuple[int, int, str, str]]]:
    """Helper name -> [(start, end, specialization, member)] for each lowered member definition."""
    definitions: Dict[str, List[Tuple[int, int, str, str]]] = {}
    for match in _LOWERED_HEADER_RE.finditer(masked):
        name = match.group("name")
        specialization, member = _specialization_member(name, specializations)
        if not specialization:
            continue
        close = preprocessor._find_matching_delimiter(masked, match.end() - 1, "(", ")")
        if close is None:
            continue
        brace = re.match(r"[^;{}]*", masked[close + 1 :])
        brace_at = close + 1 + brace.end()
        if brace_at >= len(masked) or masked[brace_at] != "{":
            # A prototype or an unrecognized declarator is not a definition this pass owns.
            continue
        end = preprocessor._find_matching_brace(masked, brace_at)
        if end is None:
            continue
        start = match.start()
        while start > 0 and masked[start - 1] in " \t":
            start -= 1
        definitions.setdefault(name, []).append((start, end, specialization, member))
    return definitions


def _referencing_definitions(
    masked: str,
    definitions: Dict[str, List[Tuple[int, int, str, str]]],
    spans: List[Tuple[int, int, str]],
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """For each helper, the helpers its body names, and the helpers named anywhere else.

    A helper's own name in its header is its declaration, not a use of it.
    """
    callees: Dict[str, Set[str]] = {name: set() for name in definitions}
    external: Set[str] = set()
    span_index = 0
    for match in _IDENTIFIER_RE.finditer(masked):
        name = match.group()
        if name not in definitions:
            continue
        position = match.start()
        while span_index < len(spans) and spans[span_index][1] <= position:
            span_index += 1
        owner = ""
        if span_index < len(spans) and spans[span_index][0] <= position:
            owner = spans[span_index][2]
        if owner == name:
            continue
        if owner:
            callees[owner].add(name)
        else:
            external.add(name)
    by_member: Dict[Tuple[str, str], List[str]] = {}
    for name, entries in definitions.items():
        for _start, _end, specialization, member in entries:
            by_member.setdefault((specialization, member), []).append(name)
    for match in _QUALIFIED_MEMBER_RE.finditer(masked):
        external.update(
            by_member.get((match.group("owner"), match.group("member")), ())
        )
    return callees, external


def prune_unreferenced_specialization_members(preprocessor: Any, code: str) -> str:
    """Remove lowered specialization members that the translated program never uses."""
    specializations = sorted(
        preprocessor._materialized_struct_specializations,
        key=lambda name: (-len(name), name),
    )
    if not specializations:
        return code
    masked = preprocessor._mask_comments_and_literals(code)
    if _ENTRY_RE.search(masked) is None:
        return code
    definitions = _lowered_definitions(preprocessor, masked, specializations)
    if not definitions:
        return code
    spans = sorted(
        (start, end, name)
        for name, entries in definitions.items()
        for start, end, _specialization, _member in entries
    )
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        return code
    callees, external = _referencing_definitions(masked, definitions, spans)
    # A helper is live when code outside the helpers names it, or when a live helper does.
    live: Set[str] = set()
    pending = sorted(external)
    while pending:
        name = pending.pop()
        if name in live:
            continue
        live.add(name)
        pending.extend(sorted(callees[name] - live))
    replacements = [
        (start, end, re.sub(r"[^\r\n]", "", code[start:end]))
        for start, end, name in spans
        if name not in live
    ]
    if not replacements:
        return code
    return preprocessor._apply_text_replacements(code, replacements)
