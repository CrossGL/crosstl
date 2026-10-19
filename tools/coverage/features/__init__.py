"""Feature lists generated from pinned upstream sources."""

from __future__ import annotations

from ..model import Feature
from ..upstream import fetch
from . import glsl, hlsl, spirv


def _glsl_names(offline: bool) -> set[str]:
    return set(glsl.overloads(fetch("glsl", offline=offline)))


def _glsl(offline: bool) -> list[Feature]:
    return glsl.parse(fetch("glsl", offline=offline))


def _hlsl(offline: bool) -> list[Feature]:
    return hlsl.parse(fetch("hlsl", offline=offline), _glsl_names(offline))


def _spirv(offline: bool) -> list[Feature]:
    return spirv.parse(fetch("spirv", offline=offline), fetch("glsl", offline=offline))


PARSERS = {"glsl": _glsl, "hlsl": _hlsl, "spirv": _spirv}


def load(language: str, *, offline: bool = False) -> list[Feature]:
    """The feature list of one language, parsed from its pinned upstream sources."""
    return PARSERS[language](offline)
