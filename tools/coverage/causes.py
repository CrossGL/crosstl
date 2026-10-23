"""Group the wrong results of a coverage row by cause.

A cause is a wrong result's diagnostic with what is particular to its probe taken out, so one translator bug that
breaks many probes is one cause, and :mod:`.issues` files one bug issue for it.
"""

from __future__ import annotations

import re

from . import spellings

# Numbers on their own, such as line and column numbers, become N. The digits of a name such as ``float3x4`` stay,
# and so do a template's arguments, which spell a type such as ``matrix<float,3,4>`` (group 1).
_DIGITS = re.compile(r"(\w+<[^<>'\"`]*>)|(?<!\w)\d+(?!\w)")
# The buffers, locals and entry points of the generated probes (:mod:`.probes`), quoted in a diagnostic.
_PLUMBING = re.compile(
    r"""(['"`])(?:fin|iin|uin|rin|outp|i|r|a\d+|tile|probe|main|compute_main)\1"""
)
# A cause writes the feature's own name as FEATURE. An input row's diagnostics start with _ROUND_TRIP, whose words
# are never the name of a feature such as ``round``.
FEATURE = "<feature>"
_ROUND_TRIP = "after the round trip: "
# The file and position a diagnostic starts with: ``probe.cu(N): error:``, ``probe.hlsl:N:N: error:``, ``ERROR: N:N:``
# and glslang's ``ERROR: probe.frag:N:``.
_LOCATION = re.compile(
    r"^(?:after the round trip: )?(?:(?:ERROR: )?\S*probe\.\w+[(:][N:)]*:?\s*|ERROR: N:N:\s*)?(?:error:\s*)?"
)

# A row's wrong results grouped by cause, as (cause, features) pairs.
Causes = list[tuple[str, list[dict]]]


def diagnostic(feature: dict) -> str:
    """The first line of a wrong result's diagnostic without its line and column numbers and the names every probe
    uses."""
    detail = (feature.get("detail") or "no diagnostic was recorded").splitlines()[0]
    detail = _PLUMBING.sub(lambda m: m.group(1) + "<probe>" + m.group(1), detail)
    return _DIGITS.sub(lambda m: m.group(1) or "N", detail).strip()


def headline(text: str) -> str:
    """``text``, a :func:`diagnostic` or a cause, without the file and position it starts with."""
    return _LOCATION.sub("", text, count=1).strip()


def cause(feature: dict, language: str = "") -> str:
    """The diagnostic of a wrong result with what is particular to its probe removed: line and column numbers, the
    names every probe uses, and the feature's own name or CrossGL spelling, including the name CrossGL gives a
    ``language`` built-in (:data:`.spellings.RENAMES`). Wrong results with one cause are one bug, so a generator that
    emits twenty builtins unchanged has one issue rather than twenty.
    """
    detail = diagnostic(feature)
    prefix = _ROUND_TRIP if detail.startswith(_ROUND_TRIP) else ""
    rest = detail[len(prefix) :]
    # An operator spelling such as spellings.SUBSCRIPT names no function a diagnostic could mention.
    names = {feature.get("name"), feature.get("crossgl")} - {
        None,
        "",
        spellings.SUBSCRIPT,
        spellings.ELEMENT,
    }
    renames = spellings.RENAMES.get(language, {})
    names |= {renames[name] for name in names if name in renames}
    for name in sorted(names, key=lambda n: (-len(n), n)):
        rest = re.sub(rf"(?<![\w.:]){re.escape(name)}(?!\w|::)", FEATURE, rest)
    return prefix + rest


def causes_of(row: dict) -> dict[str, str]:
    """The cause of each wrong result of ``row`` by feature name.

    A diagnostic that names one feature but is shared by another that it does not name is the cause of both as it
    stands, so the ``geometry`` stage that fails with "Rust geometry stage requires ..." stays with the functions that
    fail that way. Any other wrong result has its :func:`cause`.
    """
    wrong = [f for f in row["features"] if f["outcome"] == "wrong"]
    named = {f["name"]: cause(f, row["language"]) for f in wrong}
    literal = {diagnostic(f) for f in wrong if named[f["name"]] == diagnostic(f)}
    return {
        f["name"]: diagnostic(f) if diagnostic(f) in literal else named[f["name"]]
        for f in wrong
    }


def grouped(row: dict) -> Causes:
    """The row's wrong results grouped by :func:`causes_of`, the cause shared by most features first."""
    causes = causes_of(row)
    groups: dict[str, list[dict]] = {}
    for feature in sorted(row["features"], key=lambda f: f["name"]):
        if feature["outcome"] == "wrong":
            groups.setdefault(causes[feature["name"]], []).append(feature)
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
