"""Feature lists generated from pinned upstream sources."""

from __future__ import annotations

from ..model import Feature
from ..upstream import fetch
from . import glsl, spirv


def _glsl(offline: bool) -> list[Feature]:
    return glsl.parse(fetch("glsl", offline=offline))


def _spirv(offline: bool) -> list[Feature]:
    return spirv.parse(fetch("spirv", offline=offline), fetch("glsl", offline=offline))


PARSERS = {"glsl": _glsl, "spirv": _spirv}


def load(language: str, *, offline: bool = False) -> list[Feature]:
    """The feature list of one language, parsed from its pinned upstream sources."""
    return PARSERS[language](offline)
