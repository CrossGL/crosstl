"""Feature lists generated from pinned upstream sources, and for Metal from its toolchain headers."""

from __future__ import annotations

from .. import constructs
from ..model import Feature
from ..upstream import fetch
from . import cuda, glsl, hlsl, metal, spirv, wgsl


def _glsl_names(offline: bool) -> set[str]:
    return set(glsl.overloads(fetch("glsl", offline=offline)))


def _cuda(offline: bool) -> list[Feature]:
    return cuda.parse(
        fetch("cuda", offline=offline),
        fetch("cuda-device", offline=offline),
        _glsl_names(offline),
    )


def _glsl(offline: bool) -> list[Feature]:
    return glsl.parse(fetch("glsl", offline=offline))


def _hlsl(offline: bool) -> list[Feature]:
    return hlsl.parse(fetch("hlsl", offline=offline), _glsl_names(offline))


def _metal(offline: bool) -> list[Feature]:
    return metal.parse(metal.LIST.read_text(), _glsl_names(offline))


def _spirv(offline: bool) -> list[Feature]:
    return spirv.parse(fetch("spirv", offline=offline), fetch("glsl", offline=offline))


def _wgsl(offline: bool) -> list[Feature]:
    return wgsl.parse(fetch("wgsl", offline=offline), _glsl_names(offline))


PARSERS = {
    "cuda": _cuda,
    "glsl": _glsl,
    "hlsl": _hlsl,
    "metal": _metal,
    "spirv": _spirv,
    "wgsl": _wgsl,
}


def load(language: str, *, offline: bool = False) -> list[Feature]:
    """The feature list of one language: its built-in functions, parsed from its pinned upstream sources, then its
    language-level features (:mod:`..constructs`), checked against the pinned SPIR-V core grammar.
    """
    grammar = fetch("spirv-core", offline=offline)
    return PARSERS[language](offline) + constructs.features(language, grammar)
