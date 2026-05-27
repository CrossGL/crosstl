"""Shared shader stage normalization helpers."""

from .ast import ShaderStage

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
