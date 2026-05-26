"""Utilities for normalizing and matching shader stage qualifiers."""

from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name


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


def collect_stage_local_variables(ast, target_stage=None, predicate=None):
    """Return stage-local variable declarations for matching stages."""
    variables = []
    for stage_type, stage in getattr(ast, "stages", {}).items():
        stage_name = normalize_stage_name(stage_type)
        if not stage_matches(target_stage, stage_name):
            continue
        for variable in getattr(stage, "local_variables", []) or []:
            if predicate is not None and not predicate(variable):
                continue
            variables.append(variable)
    return variables


def collect_stage_local_structs(ast, target_stage=None):
    """Return stage-local struct declarations for matching stages."""
    structs = []
    for stage_type, stage in getattr(ast, "stages", {}).items():
        stage_name = normalize_stage_name(stage_type)
        if not stage_matches(target_stage, stage_name):
            continue
        structs.extend(getattr(stage, "local_structs", []) or [])
    return structs


def collect_stage_local_cbuffers(ast, target_stage=None):
    """Return stage-local cbuffer declarations for matching stages."""
    cbuffers = []
    for stage_type, stage in getattr(ast, "stages", {}).items():
        stage_name = normalize_stage_name(stage_type)
        if not stage_matches(target_stage, stage_name):
            continue
        cbuffers.extend(getattr(stage, "local_cbuffers", []) or [])
    return cbuffers


def stage_layout_qualifiers(stage, direction=None):
    """Return stage-level layout qualifiers, optionally filtered by direction."""
    layouts = list(getattr(stage, "layout_qualifiers", []) or [])
    if direction is None:
        return layouts

    direction = normalize_layout_direction(direction)
    return [
        layout
        for layout in layouts
        if normalize_layout_direction(getattr(layout, "direction", None)) == direction
    ]


def stage_layout_entries(stage, direction=None):
    """Return all entries from matching stage-level layout qualifiers."""
    entries = []
    for layout in stage_layout_qualifiers(stage, direction):
        entries.extend(getattr(layout, "entries", []) or [])
    return entries


def stage_layout_entry(stage, name, direction=None):
    """Return the first matching stage layout entry by name."""
    target_name = normalize_layout_entry_name(name)
    for entry in stage_layout_entries(stage, direction):
        if normalize_layout_entry_name(getattr(entry, "name", None)) == target_name:
            return entry
    return None


def stage_layout_entry_arguments(stage, name, direction=None):
    """Return arguments for the first matching stage layout entry."""
    entry = stage_layout_entry(stage, name, direction)
    return list(getattr(entry, "arguments", []) or []) if entry is not None else []


def stage_layout_entry_value(stage, name, direction=None, default=None):
    """Return the first argument value for a named stage layout entry."""
    arguments = stage_layout_entry_arguments(stage, name, direction)
    if not arguments:
        return default
    return layout_argument_value(arguments[0])


def normalize_layout_direction(direction):
    """Normalize a layout direction token."""
    if direction is None:
        return None
    return str(direction).split(".")[-1].lower()


def normalize_layout_entry_name(name):
    """Normalize a layout entry name for lookup."""
    if name is None:
        return None
    return str(name).lower()


def layout_argument_value(argument):
    """Return a simple value from a layout argument expression."""
    op = getattr(argument, "op", None)
    if op is not None and hasattr(argument, "left") and hasattr(argument, "right"):
        left = layout_argument_value(argument.left)
        right = layout_argument_value(argument.right)
        return f"{left} {op} {right}"
    if op is not None and hasattr(argument, "operand"):
        operand = layout_argument_value(argument.operand)
        if getattr(argument, "is_postfix", False):
            return f"{operand}{op}"
        return f"{op}{operand}"
    value = getattr(argument, "value", None)
    if value is not None:
        return str(value)
    name = getattr(argument, "name", None)
    if name is not None:
        return str(name)
    return str(argument)


def deduplicate_named_declarations(nodes, kind):
    """Deduplicate named declarations and reject conflicting same-name entries."""
    declarations = []
    declarations_by_name = {}

    for node in nodes:
        name = getattr(node, "name", getattr(node, "variable_name", None))
        if not name:
            declarations.append(node)
            continue

        signature = declaration_signature(node)
        existing = declarations_by_name.get(name)
        if existing is None:
            declarations_by_name[name] = signature
            declarations.append(node)
            continue

        if existing != signature:
            raise ValueError(f"Conflicting {kind} declaration for '{name}'")

    return declarations


def declaration_signature(node):
    """Build a small structural signature for named declaration deduplication."""
    node_type = node.__class__.__name__
    declared_type = getattr(
        node,
        "var_type",
        getattr(node, "vtype", getattr(node, "param_type", None)),
    )
    members = getattr(node, "members", None)
    attributes = getattr(node, "attributes", None)

    if members is not None:
        member_signature = tuple(
            (
                getattr(member, "name", None),
                repr(
                    getattr(
                        member,
                        "member_type",
                        getattr(
                            member,
                            "vtype",
                            getattr(member, "element_type", None),
                        ),
                    )
                ),
                repr(getattr(member, "attributes", None)),
                repr(getattr(member, "semantic", None)),
            )
            for member in members
        )
    else:
        member_signature = None

    return (
        node_type,
        repr(declared_type),
        repr(attributes),
        member_signature,
    )


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
