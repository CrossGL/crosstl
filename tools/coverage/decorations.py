"""Whether a translated graphics probe keeps the decorations, built-ins and capabilities its feature is anchored to.

A graphics probe is validated but never run, so a translation that drops an interpolation qualifier or a built-in
still validates. Where the target language compiles to SPIR-V, the translation is lowered with the language's
reference compiler and must keep every ``Decoration``, ``BuiltIn`` and ``Capability`` anchor of its feature, as each
native probe does. Metal and GLSL ES 3.00 do not compile to SPIR-V, so their translations are only validated, as is
a translation that its validator accepts but the compiler does not lower for Vulkan.
"""

from __future__ import annotations

import re

from . import validators
from .model import Feature

_PATTERNS = {
    "Decoration": r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? {name}\b",
    "BuiltIn": r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? BuiltIn {name}\b",
    "Capability": r"^\s*OpCapability {name}\b",
}


def _wanted(feature: Feature) -> list[tuple[str, str]]:
    """The anchors of a graphics ``feature`` that a SPIR-V module shows, as ``(anchor, pattern)`` pairs."""
    if feature.category != "graphics":
        return []
    wanted = []
    for anchor in feature.group.split():
        kind, _, name = anchor.partition(".")
        if kind in _PATTERNS:
            wanted.append((anchor, _PATTERNS[kind].format(name=re.escape(name))))
    return wanted


def lost(feature: Feature, language: str, source: str) -> str:
    """The anchors of ``feature`` that ``source``, a validated translation into ``language``, no longer has.

    Empty when it keeps them all, when ``feature`` has none to check or when ``source`` cannot be lowered here.
    """
    wanted = _wanted(feature)
    if not wanted:
        return ""
    if language == "spirv":
        assembly = source
    elif validators.can_lower(language):
        entry = (
            validators.hlsl_entry(source, feature.stage)
            if language in ("hlsl", "slang")
            else "main"
        )
        assembly, _ = validators.lower(
            language, source, entry, feature.stage, raw_ids=False, auto_locations=True
        )
    else:
        return ""
    if not assembly:
        return ""
    missing = [
        anchor for anchor, pattern in wanted if not re.search(pattern, assembly, re.M)
    ]
    return f"the translation loses {', '.join(missing)}" if missing else ""
