"""Stage refusals decided from a source's entry points, before the source is imported.

WebGL 2 (GLSL ES 3.00) has only vertex and fragment shaders, so the WebGL generator refuses a program whose stages
are all others (:meth:`WebGLCodeGen.validate_webgl_stage_support`). For a Metal source the generator only sees the
program after the Metal importer has materialized every template, which for a compute library such as a matrix
multiplication kernel set takes minutes, and the refusal was known from the start: entry discovery reads the
``kernel``, ``vertex`` and ``fragment`` qualifiers in well under a second.

:func:`refuse_infeasible_entry_stages` raises the generator's own :class:`StageUnsupportedError` when discovery
succeeds, finds entry points, and finds none in a stage the target can declare. Anything else (no discoverer, a
failed or empty discovery, a selected entry point, a target without such a rule) falls through to the full
translation, which decides as before.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .codegen.stage_refusals import StageUnsupportedError
from .entry_discovery import ENTRY_DISCOVERY_AVAILABLE
from .stage_utils import normalize_stage_name

__all__ = ["refuse_infeasible_entry_stages"]


def _webgl_stage_rule():
    from .codegen.webgl_codegen import WebGLCodeGen

    return (
        "WebGL",
        WebGLCodeGen.SUPPORTED_STAGE_NAMES,
        WebGLCodeGen.UNSUPPORTED_STAGE_NAMES,
        "WebGL 2 (GLSL ES 3.00) has only vertex and fragment shaders",
    )


# Target -> loader of (label, stages it declares, stages it refuses, reason), matching the generator's refusal.
_STAGE_RULES = {"webgl": _webgl_stage_rule}


def refuse_infeasible_entry_stages(
    source_spec: Any,
    code: str,
    target: str,
    *,
    file_path: str | None = None,
    include_paths: Sequence[str] | None = None,
    defines: Mapping[str, str] | None = None,
    source_options: Mapping[str, Any] | None = None,
    entry_point: str | None = None,
) -> None:
    """Raise ``target``'s stage refusal when every entry point ``source_spec`` discovers in ``code`` is refused."""
    rule = _STAGE_RULES.get(target)
    if (
        rule is None
        or entry_point is not None
        or getattr(source_spec, "entry_point_discoverer", None) is None
    ):
        return
    try:
        discovery = source_spec.discover_entry_points(
            code,
            file_path,
            include_paths=include_paths,
            defines=defines,
            source_options=source_options,
        )
    except (
        Exception
    ):  # noqa: BLE001 - the full translation reports the source's own error
        return
    if discovery.status != ENTRY_DISCOVERY_AVAILABLE or not discovery.entries:
        return
    label, supported, unsupported, reason = rule()
    stages = {normalize_stage_name(entry.stage) for entry in discovery.entries}
    if None in stages or stages & supported or not stages <= unsupported:
        return
    raise StageUnsupportedError(target, label, sorted(stages), reason)
