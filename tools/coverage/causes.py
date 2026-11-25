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
    r"""(['"`])(?:fin|iin|uin|outp|i|r|a\d+|tile|probe|main|compute_main)\1"""
)
# nvcc names the intermediate files of each compilation tmpxft_<process>_<counter>-<n>_<input stem> in its temporary
# directory, and ptxas quotes that path (``ptxas /tmp/tmpxft_0000cd1c_00000000-6_probe.ptx, line 9; fatal : ...``),
# so the path becomes the input's own name, ``probe.ptx``.
_NVCC_TEMPORARY = re.compile(
    r"""(?:[^\s'"`]*[/\\])?tmpxft_[0-9a-fA-F]+_[0-9a-fA-F]+-\d+_(?=\w)"""
)
# A cause writes the feature's own name as FEATURE. An input row's diagnostics start with _ROUND_TRIP, whose words
# are never the name of a feature such as ``round``.
FEATURE = "<feature>"
# CrossGL spells its built-in variables as GLSL does, and GLSL reserves the ``gl_`` prefix for them (GLSL 4.60,
# section 3.7), so a name with that prefix in a diagnostic is a built-in the probe uses. A construct such as
# ``subgroup-size`` records no CrossGL spelling, so this is how its ``gl_SubgroupSize`` becomes FEATURE.
_BUILTIN = re.compile(r"(?<![\w.:])gl_[A-Za-z]\w*")
_ROUND_TRIP = "after the round trip: "
# The file and position a diagnostic starts with: ``probe.cu(N): error:``, ``probe.hlsl:N:N: error:``, ``ERROR: N:N:``
# and glslang's ``ERROR: probe.frag:N:``.
_LOCATION = re.compile(
    r"^(?:after the round trip: )?(?:(?:ERROR: )?\S*probe\.\w+[(:][N:)]*:?\s*|ERROR: N:N:\s*)?(?:error:\s*)?"
)

# A row's wrong results grouped by cause, as (cause, features) pairs.
Causes = list[tuple[str, list[dict]]]


def diagnostic(feature: dict) -> str:
    """The first line of a wrong result's diagnostic without its line and column numbers, the names every probe
    uses and the temporary files a compiler names after the probe."""
    detail = (feature.get("detail") or "no diagnostic was recorded").splitlines()[0]
    detail = _NVCC_TEMPORARY.sub("", detail)
    detail = _PLUMBING.sub(lambda m: m.group(1) + "<probe>" + m.group(1), detail)
    return _DIGITS.sub(lambda m: m.group(1) or "N", detail).strip()


def headline(text: str) -> str:
    """``text``, a :func:`diagnostic` or a cause, without the file and position it starts with."""
    return _LOCATION.sub("", text, count=1).strip()


def cause(feature: dict, language: str = "") -> str:
    """The diagnostic of a wrong result with what is particular to its probe removed: line and column numbers, the
    names every probe uses, and the feature's own name or CrossGL spelling, including the name CrossGL gives a
    ``language`` built-in (:data:`.spellings.RENAMES`, :data:`.spellings.WAVES` and :data:`.spellings.OTHER_NAMES`),
    and any ``gl_`` built-in variable. Wrong results with one cause are one bug, so a generator that emits twenty
    builtins unchanged has one issue rather than twenty.
    """
    detail = diagnostic(feature)
    prefix = _ROUND_TRIP if detail.startswith(_ROUND_TRIP) else ""
    rest = detail[len(prefix) :]
    names = {feature.get("name"), feature.get("crossgl")}
    # A CrossGL IR feature carries the diagnostic of one of its members as ``<language>: <diagnostic>``, and that
    # diagnostic names the member (``cuda/atomicAdd_block``) by its name in that language.
    member_language = rest.partition(": ")[0]
    members = [m.split("/", 1) for m in feature.get("members", ())]
    member_names = {name for lang, name in members if lang == member_language}
    if member_names:
        language = member_language
        names |= member_names
    # An operator spelling such as spellings.SUBSCRIPT names no function a diagnostic could mention.
    names -= {None, "", spellings.SUBSCRIPT, spellings.ELEMENT}
    renames = (
        spellings.RENAMES.get(language, {})
        | spellings.WAVES.get(language, {})
        | spellings.OTHER_NAMES.get(language, {})
    )
    names |= {renames[name] for name in names if name in renames}
    # An expression spelling such as ``clockRealtime2x32EXT().x`` is named in a diagnostic by what it calls.
    names |= {
        n
        for name in names
        if spellings.is_expression(name)
        for n in spellings.named(name)
    }
    for name in sorted(names, key=lambda n: (-len(n), n)):
        rest = re.sub(rf"(?<![\w.:]){re.escape(name)}(?!\w|::)", FEATURE, rest)
    return prefix + _BUILTIN.sub(FEATURE, rest)


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
