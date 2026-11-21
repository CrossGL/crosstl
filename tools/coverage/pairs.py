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
whether A's own compiler accepts the probe is recorded in A's input row, so a feature that row leaves untested is
untested in A's cells too (:func:`plan`). Where both ends lower to SPIR-V, :mod:`.execution` runs the native probe and
the translation on the same runtime, and a translation that computes something else is wrong.

``python -m tools.coverage run --pairs REPORT --backend B`` measures the cells into B that a merged report of every
row defines, and ``merge`` adds them to the rows as the report's ``pairs``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

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
from .model import PAIR, ROUND_TRIPS, Feature, Result

DIRECTION = PAIR
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


@dataclass
class Plan:
    """What measuring the cells into some targets takes: the runner jobs, the results that need no job, and every cell
    measured, an empty one included."""

    jobs: list[tuple] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    cells: list[tuple[str, str]] = field(default_factory=list)


def plan(
    backends: list[dict],
    targets: Iterable[str],
    catalog: Callable[[str], list[Feature]],
    names: Iterable[str] = (),
) -> Plan:
    """How to measure every cell of the rows ``backends`` whose target is one of ``targets``.

    ``catalog`` gives a language's features (:func:`.features.load`), and ``names``, when given, keeps only the
    features with those names. The rows define the cells (:func:`cells`); a feature the catalog no longer lists, as
    one a report from another commit names, is left out, and so is a feature whose probe runs in a stage the target
    lacks (:func:`outside`). A feature its source's input row leaves untested is untested in each of its cells too:
    the source's compiler rejects the probe, or the row could not check it, so translating the probe would show
    nothing about CrossTL.
    """
    wanted, only = set(targets), set(names)
    listed: dict[str, dict[tuple[str, str], Feature]] = {}

    def lookup(language: str, entry: tuple[str, str]) -> Feature | None:
        if language not in listed:
            listed[language] = {(f.category, f.name): f for f in catalog(language)}
        return listed[language].get(entry)

    rows = {
        b["language"]: {(f["category"], f["name"]): f for f in b["features"]}
        for b in _offered(backends, "input")
    }
    measured = Plan()
    for (source, target), members in cells(backends).items():
        if target not in wanted:
            continue
        measured.cells.append((source, target))
        for member in members:
            entry = (member.category, member.name)
            feature = lookup(source, entry)
            if feature is None or (only and member.name not in only):
                continue
            if outside(feature, target):
                continue
            row = rows[source][entry]
            if row["outcome"] == "untested":
                reason = row.get("detail", "")
                detail = f"untested in the {source} input row" + (
                    f": {reason}" if reason else ""
                )
                measured.results.append(
                    Result(feature, DIRECTION, "untested", detail, into=target)
                )
                continue
            candidates = [
                partner
                for partner in (lookup(target, named) for named in member.partners)
                if partner is not None
            ]
            if candidates:
                measured.jobs.append(
                    (feature, DIRECTION, target, choose_partner(candidates))
                )
    return measured


def _partner_rejection(
    feature: Feature, source: str, target: str, partner: Feature
) -> tuple[Feature, str]:
    """``partner`` as the target probes it, and why the target's compiler rejects its own probe of ``partner``: an
    empty string when it accepts the probe or the check cannot run. As in the target's output row, a rejected probe
    for the usual compiler target is retried for the fallback one (:func:`.runner._native_verdict`).
    """
    if target not in probes.DIALECTS or not validators.available(
        target, partner.target
    ):
        return partner, ""
    try:
        native = probes.build(partner, target)
    except probes.ProbeError:
        return partner, ""
    runner._keep(
        feature, f"to-{target}", source, "partner" + native.suffix, native.source
    )
    partner, verdict = runner._native_verdict(
        partner, target, native.source, native.entry
    )
    return partner, "" if verdict.accepted else verdict.log


def run_pair(feature: Feature, source: str, target: str, partner: Feature) -> Result:
    """Translate ``feature``'s native probe in ``source`` straight to ``target`` and classify the translation.

    ``partner`` is the feature of ``target``'s row with the same semantic key (:func:`choose_partner`): the translation is
    validated for its compiler target and must keep its anchors. When ``target``'s compiler rejects its own probe of
    ``partner``, the target lacks the feature and the translation is untested, as the partner is in the target's
    output row. A supported translation that :func:`.execution.runs_pair` can run with its native probe carries both
    in ``sources``, for :mod:`.execution` to compare.
    """

    def result(outcome: str, detail: str = "") -> Result:
        tool = validators.TOOLS[target] if outcome in ("supported", "wrong") else ""
        return Result(feature, DIRECTION, outcome, detail, tool, into=target)

    try:
        if source == "spirv":
            probe = spirv_input.build(feature)
        else:
            probe = probes.build(feature, source)
    except probes.ProbeError as error:
        return result("untested", str(error))
    folder = f"to-{target}"
    runner._keep(feature, folder, source, "probe" + probe.suffix, probe.source)
    partner, rejected = _partner_rejection(feature, source, target, partner)
    if rejected:
        return result("untested", f"the target's own probe is rejected: {rejected}")
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
    if stage == "compute" and execution.runs_pair(source, target):
        return replace(result("supported"), sources=(probe.source, translated))
    return result("supported")
