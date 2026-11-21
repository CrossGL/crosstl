"""The translation matrix: how CrossTL translates each offered source language straight to each other target.

A cell ``A -> B`` holds the features of A's input row that B can express too: those whose semantic feature, the key
the CrossGL IR row groups output features by (:func:`.ir.groups`), is also the key of a feature of B's output row. A
feature of A without such a partner is outside the cell, as an :data:`.templates.ABSENT` construct is outside a row,
and so is one whose probe runs in a stage B lacks (:func:`outside`). The cells ``A -> A`` and OpenCL C to HLSL are
the input rows themselves (:data:`.model.ROUND_TRIPS`), so the matrix measures only the other cells.

A cell's feature is measured as a user translates a file: CrossTL translates A's native probe to B in one call,
``crosstl.translate(probe, backend=B)``, which imports it to CrossGL and generates B. The translation is judged by
the rules of B's output row (:func:`.runner.run_output`): a typed diagnostic is a refusal; B's validator rejecting
it, a placeholder comment, a lost anchor or a crash is wrong. Only B's tool is needed, because A's probe is text;
whether A's own compiler accepts the probe is recorded in A's input row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import (
    constructs,
    decorations,
    execution,
    ir,
    placeholders,
    probes,
    runner,
    spirv_input,
    validators,
)
from .model import ROUND_TRIPS, Feature, Result

DIRECTION = "pair"
# The stages a target cannot hold a program of, and why. Every WebGL probe is a fragment shader because GLSL ES 3.00
# has no compute stage (:class:`.languages.WebGL`), so another language's compute probe has no translation to it.
_MISSING_STAGES = {"webgl": {"compute": "GLSL ES 3.00 has no compute stage"}}


@dataclass(frozen=True, order=True)
class Member:
    """A feature of a cell's source language, its semantic key and the target's features with that key, each as
    ``(category, name)``."""

    category: str
    name: str
    key: str
    partners: tuple[tuple[str, str], ...]


def _offered(backends: list[dict], direction: str) -> list[dict]:
    return [
        b
        for b in backends
        if b["direction"] == direction and b.get("offered") and b.get("features")
    ]


def is_row(source: str, target: str) -> bool:
    """Whether the cell ``source -> target`` is ``source``'s input row rather than a cell of its own."""
    return target == ROUND_TRIPS.get(source, source)


def keys(backends: list[dict]) -> dict[tuple[str, str, str], str]:
    """The semantic key of each feature of the offered rows, by ``(language, category, name)``.

    An output row's features take the keys :func:`.ir.groups` gives them, and the language's input row lists the
    same features. A language with only an input row, as OpenCL C, has no semantic ids: its constructs take the
    construct's own id (:func:`.constructs.semantic_id`), and a function takes the key of the output functions that
    CrossGL spells the same way when they are one semantic feature. Its other features are keys of their own.
    """
    keyed: dict[tuple[str, str, str], str] = {}
    alike: dict[str, set[str]] = {}
    for key, members in ir.groups(backends).items():
        for language, feature in members:
            keyed[language, feature["category"], feature["name"]] = key
            if feature["category"] == "function":
                alike.setdefault(ir.spelling(feature), set()).add(key)
    outputs = {language for language, _, _ in keyed}
    for backend in _offered(backends, "input"):
        language = backend["language"]
        if language in outputs:
            continue
        for feature in backend["features"]:
            category, name = feature["category"], feature["name"]
            spelled = ir.spelling(feature)
            shared = alike.get(spelled, set()) if category == "function" else set()
            probe = Feature(language, category, name, None)
            if constructs.is_construct(probe):
                key = constructs.semantic_id(probe)
            elif len(shared) == 1:
                (key,) = shared
            else:
                key = f"{language}/{spelled}"
            keyed[language, category, name] = key
    return keyed


def cells(backends: list[dict]) -> dict[tuple[str, str], list[Member]]:
    """The features of every cell of the matrix, by ``(source, target)``, for each offered input row and each
    offered output row that is not the input row itself; a cell whose languages share no feature is empty.
    """
    keyed = keys(backends)
    targets: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for backend in _offered(backends, "output"):
        by_key = targets.setdefault(backend["language"], {})
        for feature in backend["features"]:
            entry = (feature["category"], feature["name"])
            by_key.setdefault(keyed[(backend["language"], *entry)], []).append(entry)
    matrix: dict[tuple[str, str], list[Member]] = {}
    for backend in sorted(_offered(backends, "input"), key=lambda b: b["language"]):
        source = backend["language"]
        for target, by_key in sorted(targets.items()):
            if is_row(source, target):
                continue
            members = []
            for feature in backend["features"]:
                key = keyed.get((source, feature["category"], feature["name"]), "")
                if key in by_key:
                    partners = tuple(sorted(by_key[key]))
                    members.append(
                        Member(feature["category"], feature["name"], key, partners)
                    )
            matrix[source, target] = sorted(members)
    return matrix


def choose_partner(candidates: list[Feature]) -> Feature:
    """The feature of the target whose checks a cell's translation gets: one probed for the target language's usual
    compiler target when there is one, so the translation is validated as a user's would be.
    """
    return min(candidates, key=lambda f: (f.target != "", f.category, f.name))


def outside(feature: Feature, target: str) -> str:
    """Why ``target`` cannot hold the program of ``feature``'s native probe at all; empty when it can."""
    return _MISSING_STAGES.get(target, {}).get(feature.stage, "")


def run_pair(feature: Feature, source: str, target: str, partner: Feature) -> Result:
    """Translate ``feature``'s native probe in ``source`` straight to ``target`` and classify the translation.

    ``partner`` is the feature of ``target``'s row with the same semantic key (:func:`choose_partner`): the translation is
    validated for its compiler target and must keep its anchors. A supported translation that ``target`` can run
    carries the native probe and the translation in ``sources``, for :mod:`.execution` to compare.
    """

    def result(outcome: str, detail: str = "") -> Result:
        tool = validators.TOOLS[target] if outcome in ("supported", "wrong") else ""
        return Result(feature, DIRECTION, outcome, detail, tool)

    try:
        if source == "spirv":
            probe = spirv_input.build(feature)
        else:
            probe = probes.build(feature, source)
    except probes.ProbeError as error:
        return result("untested", str(error))
    folder = f"to-{target}"
    runner._keep(feature, folder, source, "probe" + probe.suffix, probe.source)
    try:
        translated = runner.translate(
            probe.source,
            probe.suffix,
            runner.OUTPUT_BACKENDS[target],
            runner.INPUT_BACKENDS[source],
        )
    except runner.Refused as error:
        return result("refused", str(error))
    except runner.Broken as error:
        return result("wrong", str(error))
    stage = feature.stage
    runner._keep(
        feature,
        folder,
        source,
        "translated" + validators.suffix(target, stage),
        translated,
    )
    if not validators.available(target, partner.target):
        tool = validators.missing(target, partner.target) or validators.TOOLS[target]
        return result("untested", f"{tool} is not installed")
    verdict = runner._validated(target, translated, stage, "", partner.target)
    if not verdict.accepted:
        return result("wrong", verdict.log)
    comment = placeholders.placeholder(translated)
    if comment:
        return result("wrong", placeholders.detail(comment))
    lost = decorations.lost(partner, target, translated, imported=source)
    if lost:
        return result("wrong", lost)
    if target == "spirv":
        ignored = spirv_input.ignores_nan(partner, translated)
        if ignored:
            return result("wrong", ignored)
    if stage == "compute" and execution.runs(target):
        return replace(result("supported"), sources=(probe.source, translated))
    return result("supported")
