"""Every artifact of one translation, with the source imported once.

A GLSL compilation unit holds one entry point per stage, ``main`` (GLSL 4.60 and GLSL ES 3.20 section 6.1), so the
GLSL generators refuse a whole program with two entries of one stage
(:mod:`crosstl.translator.codegen.glsl_stage_entries`), and such a source is translated one entry per artifact with
:func:`crosstl.translate_entries`. Translating the whole file first only to learn from that refusal which entries it
holds imports the source twice: the front end, the Metal template materialization and the rule-carrying passes run
once for the refused whole program and again for its entries, and for a large source they cost most of the
translation.

:func:`translate_artifacts` decides before importing. When the target holds one entry per stage and the source's
entry discovery (the same discovery the project pipeline splits units by, :mod:`crosstl.project.glsl_entry_split`)
finds two entries of one stage, it translates the discovered entries with one import. Otherwise it translates the
whole file, and still splits it on the generator's refusal when discovery could not see the entries. Each artifact
is exactly what :func:`crosstl.translate` returns for the whole file or for that ``entry_point``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .translator.codegen.registry import BACKEND_REGISTRY, normalize_backend_name
from .translator.entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from .translator.source_registry import SOURCE_REGISTRY, register_default_sources
from .translator.stage_feasibility import refuse_infeasible_entry_stages
from .translator.stage_utils import normalize_stage_name

__all__ = ["TranslationArtifact", "holds_one_entry_per_stage", "translate_artifacts"]

# Why a discovered entry could not be selected: discovery and the imported program disagree about its name, and the
# whole-file translation is the one that reports the program's own entries.
_UNSELECTABLE_ENTRY_REASONS = frozenset({"not-found", "ambiguous"})


@dataclass(frozen=True)
class TranslationArtifact:
    """One generated artifact: the whole file (``entry_point`` is ``None``) or one entry of ``stage``."""

    code: str
    entry_point: str | None = None
    stage: str | None = None


def holds_one_entry_per_stage(target: str) -> bool:
    """Whether ``target``'s generator refuses a program with two entries of one stage."""
    spec = BACKEND_REGISTRY.get(normalize_backend_name(target) or target)
    return spec is not None and bool(
        getattr(spec.codegen_class, "one_entry_per_stage", False)
    )


def translate_artifacts(
    file_path: str,
    backend: str,
    format_output: bool = True,
    source_backend: str | None = None,
    *,
    include_paths: Sequence[str] | None = None,
    defines: Mapping[str, str] | None = None,
    source_options: Mapping[str, Any] | None = None,
) -> tuple[TranslationArtifact, ...]:
    """Translate ``file_path`` to ``backend`` as the artifacts the target needs, importing the source once.

    Returns one whole-file artifact, or one artifact per entry, in source order, when the target's compilation unit
    holds one entry per stage and the source has two entries of a stage. A refusal of the whole file, or of any
    entry, is raised as :func:`crosstl.translate` raises it.
    """
    from . import _crosstl

    options = {
        "include_paths": include_paths,
        "defines": defines,
        "source_options": source_options,
    }
    stage_entries = discovered_repeated_stage_entries(
        file_path, backend, source_backend, **options
    )
    if stage_entries:
        try:
            return _entry_artifacts(
                file_path,
                backend,
                stage_entries,
                format_output,
                source_backend,
                options,
            )
        except ValueError as error:
            if getattr(error, "reason", None) not in _UNSELECTABLE_ENTRY_REASONS:
                raise
    try:
        code = _crosstl.translate(
            file_path,
            backend=backend,
            format_output=format_output,
            source_backend=source_backend,
            **options,
        )
    except Exception as error:
        stage_entries = tuple(getattr(error, "stage_entries", None) or ())
        if not stage_entries:
            raise
        return _entry_artifacts(
            file_path, backend, stage_entries, format_output, source_backend, options
        )
    return (TranslationArtifact(code),)


def discovered_repeated_stage_entries(
    file_path: str,
    backend: str,
    source_backend: str | None = None,
    *,
    include_paths: Sequence[str] | None = None,
    defines: Mapping[str, str] | None = None,
    source_options: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """The ``(stage, entry)`` pairs to translate one per artifact, or ``()`` to translate the whole file.

    They are the source's discovered entries when ``backend`` holds one entry per stage and two of them share a
    stage. A source whose entries no stage of ``backend`` supports raises the whole file's stage refusal first, as
    :func:`crosstl.translate` does before importing it.
    """
    from ._crosstl import _read_shader_source

    target = normalize_backend_name((backend or "").strip().lower())
    if not target or not holds_one_entry_per_stage(target):
        return ()
    register_default_sources()
    spec = (
        SOURCE_REGISTRY.get(source_backend)
        if source_backend
        else SOURCE_REGISTRY.get_by_extension(str(file_path))
    )
    if spec is None or getattr(spec, "entry_point_discoverer", None) is None:
        return ()
    code = _read_shader_source(str(file_path), spec.name)
    try:
        discovery = spec.discover_entry_points(
            code,
            str(file_path),
            include_paths=include_paths,
            defines=defines,
            source_options=source_options,
        )
    except Exception:  # noqa: BLE001
        # The whole-file translation reports the source's own error.
        return ()
    if discovery.status != ENTRY_DISCOVERY_AVAILABLE:
        return ()
    stage_entries = tuple(
        (normalize_stage_name(entry.stage), entry.name) for entry in discovery.entries
    )
    names = [name for _stage, name in stage_entries]
    # An entry is selected by name, so a name discovery gives two entries leaves the choice to the generator.
    if any(stage is None or not name for stage, name in stage_entries) or len(
        set(names)
    ) != len(names):
        return ()
    if max(Counter(stage for stage, _ in stage_entries).values(), default=0) < 2:
        return ()
    refuse_infeasible_entry_stages(
        spec,
        code,
        target,
        file_path=str(file_path),
        include_paths=include_paths,
        defines=defines,
        source_options=source_options,
    )
    return stage_entries


def _entry_artifacts(
    file_path, backend, stage_entries, format_output, source_backend, options
):
    from ._crosstl import translate_entries

    names = [name for _stage, name in stage_entries]
    artifacts = translate_entries(
        file_path,
        backend,
        names,
        format_output,
        source_backend,
        **options,
    )
    return tuple(
        TranslationArtifact(artifacts[name], entry_point=name, stage=stage)
        for stage, name in stage_entries
    )
