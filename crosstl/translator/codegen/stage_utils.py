"""Utilities for normalizing and matching shader stage qualifiers."""

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
    """Normalize a shader stage enum or string into a lowercase name."""
    if stage is None:
        return None
    return str(stage).split(".")[-1].lower()


def stage_matches(target_stage, stage):
    """Return whether a stage should be emitted for the target stage."""
    target_stage = normalize_stage_name(target_stage)
    stage = normalize_stage_name(stage)
    return target_stage is None or stage == target_stage


def should_emit_qualified_function(target_stage, qualifier):
    """Return whether a qualified function belongs in the target output."""
    target_stage = normalize_stage_name(target_stage)
    qualifier = normalize_stage_name(qualifier)
    return not (
        target_stage is not None
        and qualifier in STAGE_QUALIFIER_NAMES
        and qualifier != target_stage
    )


def function_stage_name(func):
    qualifiers = getattr(func, "qualifiers", []) or []
    qualifier = qualifiers[0] if qualifiers else getattr(func, "qualifier", None)
    return normalize_stage_name(qualifier)


def collect_stage_entry_records(
    ast,
    target_stage,
    stage_entry_types,
    include_qualified_function=should_emit_qualified_function,
    include_stage=stage_matches,
):
    entries = []

    for func in getattr(ast, "functions", []) or []:
        stage_name = function_stage_name(func)
        if stage_name in stage_entry_types and include_qualified_function(
            target_stage, stage_name
        ):
            entries.append((id(func), stage_name, func))

    for stage_type, stage in getattr(ast, "stages", {}).items():
        stage_name = normalize_stage_name(stage_type)
        if not include_stage(target_stage, stage_name):
            continue

        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            entries.append((id(entry_point), stage_name, entry_point))

    return entries


def collect_stage_entry_reserved_function_names(
    ast,
    target_stage,
    stage_entry_types,
    include_qualified_function=should_emit_qualified_function,
    include_stage=stage_matches,
):
    names = set()

    for func in getattr(ast, "functions", []) or []:
        stage_name = function_stage_name(func)
        is_entry = stage_name in stage_entry_types and include_qualified_function(
            target_stage, stage_name
        )
        if is_entry:
            continue

        name = getattr(func, "name", None)
        if name:
            names.add(name)

    for stage_type, stage in getattr(ast, "stages", {}).items():
        stage_name = normalize_stage_name(stage_type)
        if not include_stage(target_stage, stage_name):
            continue

        for func in getattr(stage, "local_functions", []) or []:
            name = getattr(func, "name", None)
            if name:
                names.add(name)

    return names


def assign_stage_entry_names(
    entries,
    reserved_names,
    base_name_for_entry,
    single_entry_default=None,
):
    if (
        single_entry_default is not None
        and len(entries) <= 1
        and single_entry_default not in reserved_names
    ):
        return {}

    names = {}
    used_names = set(reserved_names)
    for func_id, stage_name, func in entries:
        base_name = base_name_for_entry(stage_name, func)
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        names[func_id] = candidate
        used_names.add(candidate)

    return names


def order_functions_by_dependencies(
    functions,
    walk_nodes,
    function_call_name,
    function_call_node_type,
):
    ordered = []
    visiting = set()
    visited = set()
    function_list = list(functions)
    function_names = [getattr(func, "name", None) for func in function_list]
    unique_names = {
        name for name in function_names if name and function_names.count(name) == 1
    }
    functions_by_name = {
        func.name: func
        for func in function_list
        if getattr(func, "name", None) in unique_names
    }

    def visit(func):
        func_id = id(func)
        if func_id in visited or func_id in visiting:
            return

        visiting.add(func_id)
        for node in walk_nodes(getattr(func, "body", [])):
            if not isinstance(node, function_call_node_type):
                continue

            dependency = functions_by_name.get(function_call_name(node))
            if dependency is not None and dependency is not func:
                visit(dependency)

        visiting.remove(func_id)
        visited.add(func_id)
        ordered.append(func)

    for func in function_list:
        visit(func)

    return ordered


def compute_local_size(execution_config=None):
    """Return a three-component workgroup size from execution metadata."""
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
    """Return a string representation for a local-size dimension value."""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
