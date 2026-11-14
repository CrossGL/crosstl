"""Shared shader stage normalization helpers.

:func:`normalize_stage_name` answers "which shader stage does this name spell?": a :class:`ShaderStage` value, a
CrossGL stage name or one of the backend aliases in :data:`STAGE_NAME_ALIASES`. Any other name, such as an attribute
or a qualifier, spells no stage and gives ``None``; it is never returned as if it were a stage name.
:func:`requested_stage_name` is the same lookup for a stage the caller asked for, where a name that spells no stage is
refused with :class:`UnknownShaderStageError`.
"""

from .ast import ShaderStage
from .diagnostics import CODE_PREFIX, TranslationDiagnostic

STAGE_NAME_ALIASES = {
    "amplification_shader": "amplification",
    "anyhit": "ray_any_hit",
    "anyhit_shader": "ray_any_hit",
    "any_hit": "ray_any_hit",
    "any_hit_shader": "ray_any_hit",
    "as": "amplification",
    "callable": "ray_callable",
    "callable_shader": "ray_callable",
    "closest_hit": "ray_closest_hit",
    "closest_hit_shader": "ray_closest_hit",
    "closesthit": "ray_closest_hit",
    "closesthit_shader": "ray_closest_hit",
    "compute_shader": "compute",
    "cs": "compute",
    "domain": "tessellation_evaluation",
    "domain_shader": "tessellation_evaluation",
    "ds": "tessellation_evaluation",
    "frag": "fragment",
    "fragment_shader": "fragment",
    "fs": "fragment",
    "geometry_shader": "geometry",
    "glcompute": "compute",
    "gs": "geometry",
    "hull": "tessellation_control",
    "hull_shader": "tessellation_control",
    "hs": "tessellation_control",
    "intersection": "ray_intersection",
    "intersection_shader": "ray_intersection",
    "kernel": "compute",
    "miss": "ray_miss",
    "miss_shader": "ray_miss",
    "mesh_shader": "mesh",
    "ms": "mesh",
    "pixel": "fragment",
    "pixel_shader": "fragment",
    "ps": "fragment",
    "rahit": "ray_any_hit",
    "rcall": "ray_callable",
    "rchit": "ray_closest_hit",
    "rgen": "ray_generation",
    "rint": "ray_intersection",
    "rmiss": "ray_miss",
    "ray_generation_shader": "ray_generation",
    "raygen": "ray_generation",
    "raygeneration": "ray_generation",
    "raygeneration_shader": "ray_generation",
    "task_shader": "task",
    "tesscontrol": "tessellation_control",
    "tesseval": "tessellation_evaluation",
    "vertex_shader": "vertex",
    "vs": "vertex",
}

SHADER_STAGE_NAMES = frozenset(stage.value for stage in ShaderStage)
SHADER_STAGE_BY_NAME = {stage.value: stage for stage in ShaderStage}
STAGE_QUALIFIER_NAMES = SHADER_STAGE_NAMES


class UnknownShaderStageError(TranslationDiagnostic):
    """A requested shader stage that is neither a CrossGL stage name nor a known alias of one."""

    def __init__(self, stage):
        self.stage = stage
        self.missing_capabilities = ("crossgl.shader-stage",)
        known = ", ".join(sorted(SHADER_STAGE_NAMES))
        super().__init__(
            f"{str(stage)!r} names no shader stage; expected one of {known} or a backend alias of one",
            code=f"{CODE_PREFIX}unknown-shader-stage",
        )


def _stage_spelling(stage):
    if hasattr(stage, "value") and not isinstance(stage, str):
        stage = getattr(stage, "value")
    name = str(stage).split(".")[-1].strip().strip("\"'").lower()
    return name.replace("-", "_").replace(" ", "_")


def normalize_stage_name(stage):
    """The canonical name of the shader stage ``stage`` spells, or ``None`` when it spells none."""
    if stage is None:
        return None
    name = _stage_spelling(stage)
    if name in SHADER_STAGE_NAMES:
        return name
    if name in STAGE_NAME_ALIASES:
        return STAGE_NAME_ALIASES[name]
    return None


def requested_stage_name(stage):
    """The canonical name of a requested stage; ``None`` requests every stage.

    Raises :class:`UnknownShaderStageError` when ``stage`` spells no shader stage, so that a misspelt request cannot
    select no stage, or every stage, without a word.
    """
    if stage is None:
        return None
    name = normalize_stage_name(stage)
    if name is None:
        raise UnknownShaderStageError(stage)
    return name


def shader_stage_from_name(name):
    """Return the canonical shader stage enum for a CrossGL or backend alias."""
    return SHADER_STAGE_BY_NAME.get(normalize_stage_name(name))
