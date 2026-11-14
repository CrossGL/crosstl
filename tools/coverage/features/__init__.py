"""Feature lists generated from pinned upstream sources, and for Metal from its toolchain headers."""

from __future__ import annotations

from .. import constructs
from ..model import Feature
from ..upstream import fetch
from . import (
    cuda,
    glam,
    glsl,
    glsl_subgroup,
    hip,
    hlsl,
    metal,
    mojo,
    opencl,
    rust,
    slang,
    spirv,
    spirv_std,
    webgl,
    wgsl,
)


def _glsl_names(offline: bool) -> set[str]:
    return set(glsl.overloads(fetch("glsl", offline=offline)))


def _cuda(offline: bool) -> list[Feature]:
    return cuda.parse(
        fetch("cuda", offline=offline),
        fetch("cuda-device", offline=offline),
        _glsl_names(offline),
        fetch("cuda-intrinsics", offline=offline),
    )


def _glsl(offline: bool) -> list[Feature]:
    return glsl.parse(fetch("glsl", offline=offline)) + glsl_subgroup.parse(
        fetch("glsl-subgroup", offline=offline)
    )


def _hip(offline: bool) -> list[Feature]:
    return hip.parse(
        fetch("hip", offline=offline),
        fetch("hip-atomic", offline=offline),
        fetch("hip-device", offline=offline),
        _glsl_names(offline),
        fetch("hip-warp", offline=offline),
    )


def _hlsl(offline: bool) -> list[Feature]:
    return hlsl.parse(fetch("hlsl", offline=offline), _glsl_names(offline))


def _metal(offline: bool) -> list[Feature]:
    return metal.parse(metal.LIST.read_text(), _glsl_names(offline))


def _mojo(offline: bool) -> list[Feature]:
    return mojo.parse(
        fetch("mojo-math-exports", offline=offline),
        fetch("mojo-math", offline=offline),
        fetch("mojo-numerics", offline=offline),
        fetch("mojo-bit-exports", offline=offline),
        fetch("mojo-bit", offline=offline),
        _glsl_names(offline),
        warp=fetch("mojo-gpu-warp", offline=offline),
        block=fetch("mojo-gpu-block", offline=offline),
        sync_init=fetch("mojo-gpu-sync-exports", offline=offline),
        sync=fetch("mojo-gpu-sync", offline=offline),
        ids=fetch("mojo-gpu-id", offline=offline),
        lanes=fetch("mojo-gpu-lanes", offline=offline),
        atomic_init=fetch("mojo-atomic-exports", offline=offline),
        atomic=fetch("mojo-atomic", offline=offline),
    )


def _opencl(offline: bool) -> list[Feature]:
    declared = opencl.parse(fetch("opencl", offline=offline))
    compiler = opencl.compiler(
        fetch("opencl-base", offline=offline),
        fetch("opencl-builtins", offline=offline),
        fetch("opencl-keywords", offline=offline),
    )
    return sorted(declared + compiler, key=lambda f: (f.group, f.name))


# The modules of spirv_std::arch, each a source of :data:`..upstream.SOURCES` named ``rust-<module>``.
RUST_ARCH = (
    "arch",
    "atomics",
    "barrier",
    "demote_to_helper_invocation_ext",
    "derivative",
    "mesh_shading",
    "primitive",
    "ray_tracing",
    "subgroup",
)


def _rust(offline: bool) -> list[Feature]:
    glsl_names = _glsl_names(offline)
    features = rust.parse(
        {module: fetch(f"rust-{module}", offline=offline) for module in RUST_ARCH},
        fetch("rust-float", offline=offline),
        fetch("rust-vec3", offline=offline),
        glsl_names,
    )
    features += rust.packing(fetch("rust-packing", offline=offline), glsl_names)
    features += spirv_std.images(fetch("rust-image", offline=offline))
    features += glam.parse(
        {name: fetch(key, offline=offline) for name, key in glam.SOURCES.items()},
        glsl_names,
    )
    return sorted(features, key=lambda f: (f.group, f.name))


def _slang(offline: bool) -> list[Feature]:
    return slang.parse(
        fetch("slang-core", offline=offline),
        fetch("slang", offline=offline),
        _glsl_names(offline),
    )


def _spirv(offline: bool) -> list[Feature]:
    glsl_text = fetch("glsl", offline=offline)
    return spirv.parse(fetch("spirv", offline=offline), glsl_text) + spirv.core(
        fetch("spirv-core", offline=offline),
        glsl_text,
        fetch("glsl-subgroup", offline=offline),
    )


def _webgl(offline: bool) -> list[Feature]:
    return webgl.parse(fetch("webgl", offline=offline))


def _wgsl(offline: bool) -> list[Feature]:
    return wgsl.parse(fetch("wgsl", offline=offline), _glsl_names(offline))


PARSERS = {
    "cuda": _cuda,
    "glsl": _glsl,
    "hip": _hip,
    "hlsl": _hlsl,
    "metal": _metal,
    "mojo": _mojo,
    "opencl": _opencl,
    "rust": _rust,
    "slang": _slang,
    "spirv": _spirv,
    "webgl": _webgl,
    "wgsl": _wgsl,
}


def load(language: str, *, offline: bool = False) -> list[Feature]:
    """The feature list of one language: its built-in functions, parsed from its pinned upstream sources, then its
    language-level features (:mod:`..constructs`), checked against the pinned SPIR-V core grammar.
    """
    grammar = fetch("spirv-core", offline=offline)
    return PARSERS[language](offline) + constructs.features(language, grammar)
