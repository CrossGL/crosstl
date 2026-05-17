STAGE_QUALIFIER_NAMES = frozenset(
    {
        "vertex",
        "fragment",
        "compute",
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "amplification",
        "object",
        "ray_generation",
        "ray_intersection",
        "ray_any_hit",
        "ray_closest_hit",
        "ray_miss",
        "ray_callable",
        "intersection",
        "anyhit",
        "closesthit",
        "miss",
        "callable",
    }
)


def normalize_stage_name(stage):
    if stage is None:
        return None
    return str(stage).split(".")[-1].lower()


def stage_matches(target_stage, stage):
    target_stage = normalize_stage_name(target_stage)
    stage = normalize_stage_name(stage)
    return target_stage is None or stage == target_stage


def should_emit_qualified_function(target_stage, qualifier):
    target_stage = normalize_stage_name(target_stage)
    qualifier = normalize_stage_name(qualifier)
    return not (
        target_stage is not None
        and qualifier in STAGE_QUALIFIER_NAMES
        and qualifier != target_stage
    )


def compute_local_size(execution_config=None):
    config = execution_config or {}
    for key in ("local_size", "workgroup_size", "numthreads"):
        value = config.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return tuple(compute_local_size_value(item) for item in value[:3])

    return (
        compute_local_size_value(config.get("local_size_x", 1)),
        compute_local_size_value(config.get("local_size_y", 1)),
        compute_local_size_value(config.get("local_size_z", 1)),
    )


def compute_local_size_value(value):
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
