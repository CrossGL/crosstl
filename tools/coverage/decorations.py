"""Whether a translated graphics probe keeps the decorations, built-ins, capabilities and execution modes its feature
is anchored to.

A graphics probe is validated but never run, so a translation that drops an interpolation qualifier or a built-in
still validates. Where the target language compiles to SPIR-V, the translation is lowered with the language's
reference compiler and must keep every ``Decoration``, ``BuiltIn``, ``Capability`` and ``ExecutionMode`` anchor of
its feature, as each native probe does. GLSL ES 3.00 is lowered as GLSL ES 3.10, the first version glslang compiles
for Vulkan. Metal compiles to AIR instead, where a translation must keep the attribute of each anchor (:data:`AIR`)
or the call that stands for it (:data:`AIR_CALLS`). A translation that its validator accepts but its compiler does
not lower here is only validated.
"""

from __future__ import annotations

import re

from . import validators
from .model import Feature

_PATTERNS = {
    "Decoration": r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? {name}\b",
    "BuiltIn": r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? BuiltIn {name}\b",
    "Capability": r"^\s*OpCapability {name}\b",
    "ExecutionMode": r"^\s*OpExecutionMode %\w+ {name}\b",
}
# The instructions that stand for an anchor as well: whether an invocation is a helper is the ``HelperInvocation``
# built-in or, with SPV_EXT_demote_to_helper_invocation, ``OpIsHelperInvocationEXT``, which DXC and slangc emit for
# ``IsHelperLane()``.
_INSTRUCTIONS = {"BuiltIn.HelperInvocation": r"^\s*%\w+ = OpIsHelperInvocationEXT\b"}

# The metadata the Metal compiler writes in AIR for the attribute that carries each anchor: ``[[flat]]`` becomes
# ``!"air.flat"`` and ``[[vertex_id]]`` ``!"air.vertex_id"``. A fragment function's ``[[position]]`` input is the
# fragment coordinate, and ``[[color(n)]]`` is a render target at a location. A capability has no attribute, so a
# Metal translation is not checked for one.
AIR = {
    "BuiltIn.FragCoord": "air.position",
    "BuiltIn.FrontFacing": "air.front_facing",
    "BuiltIn.FragDepth": "air.depth",
    "BuiltIn.VertexIndex": "air.vertex_id",
    "BuiltIn.InstanceIndex": "air.instance_id",
    "BuiltIn.PointSize": "air.point_size",
    "BuiltIn.ClipDistance": "air.clip_distance",
    "BuiltIn.PrimitiveId": "air.primitive_id",
    "BuiltIn.SampleId": "air.sample_id",
    "BuiltIn.SampleMask": "air.sample_mask",
    "BuiltIn.Layer": "air.render_target_array_index",
    "BuiltIn.ViewportIndex": "air.viewport_array_index",
    "BuiltIn.PointCoord": "air.point_coord",
    "Decoration.Flat": "air.flat",
    "Decoration.NoPerspective": "air.no_perspective",
    "Decoration.Centroid": "air.centroid",
    "Decoration.Sample": "air.sample",
    "Decoration.Invariant": "air.invariant",
    "Decoration.Location": "air.render_target",
    "ExecutionMode.EarlyFragmentTests": "early_fragment_tests",
    "ExecutionMode.DepthGreater": "air.greater",
}
# The attributes that AIR names apart when a fragment function reads them: a ``[[sample_mask]]`` input is
# ``!"air.sample_mask_in"``. A feature whose name ends in ``-input`` reads its built-in.
AIR_INPUTS = {"BuiltIn.SampleMask": "air.sample_mask_in"}
# The anchors that AIR carries as a call rather than an attribute: ``simd_is_helper_thread()`` calls
# ``@air.simd_is_helper_thread``.
AIR_CALLS = {"BuiltIn.HelperInvocation": "air.simd_is_helper_thread"}

_ESSL_300 = re.compile(r"\A(\s*#version\s+)300(\s+es\b)")


def _wanted(feature: Feature, language: str) -> list[tuple[str, str]]:
    """The anchors of a graphics ``feature`` that the compiled ``language`` shows, as ``(anchor, pattern)`` pairs."""
    if feature.category != "graphics":
        return []
    wanted = []
    for anchor in feature.group.split():
        kind, _, name = anchor.partition(".")
        if language == "metal":
            air = AIR_INPUTS if feature.name.endswith("-input") else {}
            attribute = air.get(anchor, AIR.get(anchor))
            if attribute:
                wanted.append((anchor, rf'!"{re.escape(attribute)}"'))
            elif anchor in AIR_CALLS:
                wanted.append((anchor, rf"@{re.escape(AIR_CALLS[anchor])}\("))
        elif kind in _PATTERNS:
            pattern = _PATTERNS[kind].format(name=re.escape(name))
            if anchor in _INSTRUCTIONS:
                pattern = f"{pattern}|{_INSTRUCTIONS[anchor]}"
            wanted.append((anchor, pattern))
    return wanted


def _compiled(feature: Feature, language: str, source: str) -> str:
    """``source`` as SPIR-V assembly, or as AIR for Metal; empty when it cannot be compiled here."""
    if language == "spirv":
        return source
    if language == "metal":
        return validators.air(source)
    if language == "webgl":
        language, source = "glsl", _ESSL_300.sub(r"\g<1>310\2", source, count=1)
    if not validators.can_lower(language):
        return ""
    entry = (
        validators.hlsl_entry(source, feature.stage)
        if language in ("hlsl", "slang")
        else "main"
    )
    assembly, _ = validators.lower(
        language, source, entry, feature.stage, raw_ids=False, auto_locations=True
    )
    return assembly


def lost(feature: Feature, language: str, source: str) -> str:
    """The anchors of ``feature`` that ``source``, a validated translation into ``language``, no longer has.

    Empty when it keeps them all, when ``feature`` has none to check or when ``source`` cannot be compiled here.
    """
    wanted = _wanted(feature, language)
    if not wanted:
        return ""
    compiled = _compiled(feature, language, source)
    if not compiled:
        return ""
    missing = [
        anchor for anchor, pattern in wanted if not re.search(pattern, compiled, re.M)
    ]
    return f"the translation loses {', '.join(missing)}" if missing else ""
