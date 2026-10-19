"""The CrossGL IR row: which semantic features of any backend CrossGL can express.

Its feature universe is the union of every backend's output features, grouped by semantic id (:mod:`.semantic`); a
feature with no id is a semantic feature of its own. A semantic feature is expressible, and counts as ``supported``,
when the CrossGL probe of at least one member parses, type-checks and translates correctly to that member's backend.
Otherwise the row reports a ``wrong`` result before a ``refused`` one, and ``untested`` only when no member was
measured. The row is computed from the report alone, so reports from several runners can be merged first.
"""

from __future__ import annotations

from .model import Score

LANGUAGE = "crossgl"
DIRECTION = "ir"
_PRECEDENCE = ("supported", "wrong", "refused", "untested")


def _members(backends: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    groups: dict[str, list[tuple[str, dict]]] = {}
    for backend in backends:
        if backend["direction"] != "output" or not backend.get("offered"):
            continue
        language = backend["language"]
        for feature in backend["features"]:
            key = feature.get("semantic") or f"{language}/{feature['name']}"
            groups.setdefault(key, []).append((language, feature))
    return groups


def _entry(key: str, members: list[tuple[str, dict]]) -> dict:
    outcome = min((feature["outcome"] for _, feature in members), key=_PRECEDENCE.index)
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
            {language for language, f in members if f["outcome"] == "supported"}
        )
        return entry
    for language, feature in sorted(members, key=lambda m: m[0]):
        if feature["outcome"] == outcome and feature.get("detail"):
            entry["detail"] = f"{language}: {feature['detail']}"
            break
    return entry


def build(backends: list[dict]) -> dict | None:
    """The IR row for the report's backend rows, or ``None`` when no output direction was measured."""
    groups = _members(backends)
    if not groups:
        return None
    features = [_entry(key, members) for key, members in sorted(groups.items())]
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
