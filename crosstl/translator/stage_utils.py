"""Shared shader stage normalization helpers."""

from .ast import ShaderStage

STAGE_NAME_ALIASES = {
    "anyhit": "ray_any_hit",
    "callable": "ray_callable",
    "closesthit": "ray_closest_hit",
    "domain": "tessellation_evaluation",
    "hull": "tessellation_control",
    "intersection": "ray_intersection",
    "miss": "ray_miss",
    "pixel": "fragment",
    "pixel_shader": "fragment",
    "raygen": "ray_generation",
    "raygeneration": "ray_generation",
    "tesscontrol": "tessellation_control",
    "tesseval": "tessellation_evaluation",
}

SHADER_STAGE_NAMES = frozenset(stage.value for stage in ShaderStage)
SHADER_STAGE_BY_NAME = {stage.value: stage for stage in ShaderStage}
STAGE_QUALIFIER_NAMES = SHADER_STAGE_NAMES


def normalize_stage_name(stage):
    """Normalize a shader stage enum or string into a canonical stage name."""
    if stage is None:
        return None
    if hasattr(stage, "value") and not isinstance(stage, str):
        stage = getattr(stage, "value")
    name = str(stage).split(".")[-1].strip().strip("\"'").lower()
    name = name.replace("-", "_").replace(" ", "_")
    return STAGE_NAME_ALIASES.get(name, name)


def shader_stage_from_name(name):
    """Return the canonical shader stage enum for a CrossGL or backend alias."""
    return SHADER_STAGE_BY_NAME.get(normalize_stage_name(name))
