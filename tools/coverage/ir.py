"""The CrossGL IR row: which semantic features of any backend CrossGL can express.

Its feature universe is the union of every backend's output features, grouped by semantic id (:mod:`.semantic`); a
function with no id takes the id of the GLSL function that SPIRV-Cross writes it for (:data:`.semantic.CROSS_COMPILED`)
or of the functions that CrossGL spells the same way, and functions that still have none are one semantic feature
per CrossGL spelling, named ``crossgl/<function>`` when several languages have it and ``<language>/<function>``
otherwise. Any other feature without an id is a semantic feature of its own. A semantic
feature is expressible, and counts as ``supported``, when the CrossGL probe of at least one member parses,
type-checks and translates correctly to that member's backend, or translates to valid SPIR-V whose call has the
member's semantic id (``expressed``, :func:`.runner.expressed`). Otherwise the row reports a ``wrong`` result before
a ``refused`` one, and ``untested`` only when no member was measured. The row is computed from the report alone, so
reports from several runners can be merged first.
"""

from __future__ import annotations

from . import spellings
from .model import Score
from .semantic import BORROWED, CROSS_COMPILED

LANGUAGE = "crossgl"
DIRECTION = "ir"
_PRECEDENCE = ("supported", "wrong", "refused", "untested")


def spelling(feature: dict) -> str:
    """The CrossGL name of a report ``feature``, which keys the features without a semantic id."""
    # A call that spells its generic arguments, as Slang's ``bit_cast<int>``, is keyed by the function's name, and so
    # is an operation CrossGL writes as an expression, as Slang's ``WorkgroupSize`` (``ivec3(gl_WorkGroupSize)``).
    spelled = feature.get("crossgl") or feature["name"]
    if spellings.is_expression(spelled):
        return feature["name"]
    return spelled.split("<", 1)[0]


def groups(backends: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """The ``(language, feature)`` members of each semantic feature of the offered output rows, by its name."""
    outputs = [b for b in backends if b["direction"] == "output" and b.get("offered")]
    ids = {
        (b["language"], f["name"]): f["semantic"]
        for b in outputs
        for f in b["features"]
        if f.get("semantic")
    }
    # The ids of the functions of every language that CrossGL spells alike. A function without an id of its own or
    # its lender's takes the id when there is only one: CUDA ``cospi``, which no compiler here lowers, and Slang
    # ``cospi`` are one feature. A method's name says nothing without its object's type, so methods take none.
    shared: dict[str, set[str]] = {}
    for backend in outputs:
        for feature in backend["features"]:
            if feature.get("semantic") and feature["category"] == "function":
                shared.setdefault(spelling(feature), set()).add(feature["semantic"])
    alike = {
        spelled: found.pop() for spelled, found in shared.items() if len(found) == 1
    }
    keyed: list[tuple[str, dict, str]] = []
    for backend in outputs:
        language = backend["language"]
        lender = BORROWED.get(language, "")
        for feature in backend["features"]:
            function = feature["category"] == "function"
            spelled = spelling(feature)
            twin = CROSS_COMPILED.get(language, {}).get(feature["name"], "")
            key = (
                feature.get("semantic")
                or ids.get(("glsl", twin))
                or ids.get((lender, spelled))
                or (alike.get(spelled, "") if function else "")
            )
            keyed.append((language, feature, key))
    # Without an id, functions CrossGL spells alike are one feature, because their CrossGL probes call one CrossGL
    # function. Within a language that is CUDA ``erff`` and ``erf``. Across languages, as CUDA and HIP ``__clz`` or
    # CUDA and Mojo ``erf``, the feature is named after the CrossGL function rather than after one of the languages.
    spread: dict[str, set[str]] = {}
    for language, feature, key in keyed:
        if not key and feature["category"] == "function":
            spread.setdefault(spelling(feature), set()).add(language)
    groups: dict[str, list[tuple[str, dict]]] = {}
    for language, feature, key in keyed:
        if not key:
            spelled = spelling(feature)
            alone = feature["category"] != "function" or len(spread[spelled]) == 1
            key = f"{language if alone else LANGUAGE}/{spelled}"
        groups.setdefault(key, []).append((language, feature))
    return groups


def _outcome(feature: dict) -> str:
    # A member its own generator fails still counts when CrossTL's SPIR-V translation of its probe expresses it.
    return "supported" if feature.get("expressed") else feature["outcome"]


def _entry(key: str, members: list[tuple[str, dict]]) -> dict:
    outcome = min((_outcome(feature) for _, feature in members), key=_PRECEDENCE.index)
    languages = sorted({language for language, _ in members})
    entry = {
        "name": key,
        "category": members[0][1]["category"],
        "group": "shared" if len(languages) > 1 else languages[0],
        "outcome": outcome,
        "members": sorted(f"{language}/{f['name']}" for language, f in members),
    }
    if outcome == "supported":
        entry["via"] = sorted(
            {
                f.get("expressed") or language
                for language, f in members
                if _outcome(f) == "supported"
            }
        )
        return entry
    for language, feature in sorted(members, key=lambda m: m[0]):
        if feature["outcome"] == outcome and feature.get("detail"):
            entry["detail"] = f"{language}: {feature['detail']}"
            break
    return entry


def build(backends: list[dict]) -> dict | None:
    """The IR row for the report's backend rows, or ``None`` when no output direction was measured."""
    grouped = groups(backends)
    if not grouped:
        return None
    features = [_entry(key, members) for key, members in sorted(grouped.items())]
    score = Score()
    for feature in features:
        score.add(feature["outcome"])
    return {
        "language": LANGUAGE,
        "direction": DIRECTION,
        "offered": True,
        "score": None if score.percent is None else round(score.percent, 1),
        "counts": score.counts,
        "features": features,
    }
