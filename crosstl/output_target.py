"""The target language a single-file translation's output file name asks for.

``crosstl shader.metal -o shader.hlsl`` writes HLSL: without ``--backend``, the command line takes the target from
the output file's extension, as it takes the source language from the input file's. A file name the code generators
claim (``.hlsl``, ``.webgl.glsl``) names that target. Otherwise an extension the source frontends read names their
language when CrossTL can also write it, as ``.frag`` names GLSL and ``.cgl`` CrossGL. Any other output file name is
refused, because writing CrossGL into a file named ``shader.spv`` or ``shader.cl`` would hide the mistake.
"""

from __future__ import annotations

import os
from pathlib import PurePath

from .translator.codegen import backend_names
from .translator.codegen.registry import BACKEND_REGISTRY
from .translator.plugin_loader import discover_backend_plugins
from .translator.refusals import RegistryRefusal
from .translator.source_registry import SOURCE_REGISTRY, register_default_sources

CROSSGL = "cgl"


def target_for_output(output_path: str | os.PathLike[str]) -> str:
    """The backend name, or ``cgl`` for CrossGL, that ``output_path``'s extension names.

    Raises :class:`RegistryRefusal` when the extension names no language CrossTL writes.
    """
    register_default_sources()
    discover_backend_plugins()
    name = PurePath(os.fspath(output_path)).name
    suffixes = [suffix.lower() for suffix in PurePath(name).suffixes]
    targets = BACKEND_REGISTRY.extensions()
    # The longest compound extension a generator claims wins, so shader.webgl.glsl is WebGL rather than GLSL.
    for index in range(len(suffixes)):
        backend = targets.get("".join(suffixes[index:]))
        if backend:
            return backend
    try:
        source = SOURCE_REGISTRY.get_by_extension(name) if suffixes else None
    except RegistryRefusal:
        source = None
    if source and (source.name == CROSSGL or source.name in backend_names()):
        return source.name
    known = ", ".join(sorted({*targets, ".cgl"}))
    if source:
        raise RegistryRefusal(
            f"Output file {name!r}: CrossTL reads {source.name} sources but has no {source.name} code generator. "
            f"Pass --backend to choose a target, or name the file with one of {known}."
        )
    raise RegistryRefusal(
        f"Cannot tell the target language from the output file name {name!r}. "
        f"Name it with one of {known}, or pass --backend ({', '.join([*backend_names(), CROSSGL])})."
    )
