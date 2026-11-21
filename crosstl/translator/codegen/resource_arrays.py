"""Helpers for collecting resource-array size hints from CrossGL AST nodes."""

from ..ast import PointerType
from .resource_array_size_inference import ResourceArraySizeInference


def split_array_suffix(array_suffix):
    """Split a C-style array suffix into dimension strings."""
    dimensions = []
    offset = 0
    while offset < len(array_suffix):
        if array_suffix[offset] != "[":
            return None
        end = array_suffix.find("]", offset + 1)
        if end == -1:
            return None
        dimensions.append(array_suffix[offset + 1 : end])
        offset = end + 1
    return dimensions


def format_array_declarator(
    mapped_base,
    name,
    array_suffix,
    dynamic_array_as_pointer=True,
):
    """Format a typed C-style array declarator for generated code."""
    if not dynamic_array_as_pointer or "[]" not in array_suffix:
        return f"{mapped_base} {name}{array_suffix}"

    dimensions = split_array_suffix(array_suffix)
    if not dimensions:
        return f"{mapped_base} {name}{array_suffix}"

    if dimensions[0] == "":
        trailing_dimensions = dimensions[1:]
        if not trailing_dimensions:
            return f"{mapped_base}* {name}"
        if all(dimension != "" for dimension in trailing_dimensions):
            trailing_suffix = "".join(
                f"[{dimension}]" for dimension in trailing_dimensions
            )
            return f"{mapped_base} (*{name}){trailing_suffix}"

    collapsed_suffix = "".join(
        f"[{dimension}]" for dimension in dimensions if dimension != ""
    )
    return f"{mapped_base}* {name}{collapsed_suffix}"


NON_PRIVATE_POINTER_QUALIFIERS = frozenset(
    {
        "buffer",
        "constant",
        "device",
        "global",
        "groupshared",
        "local",
        "shared",
        "storage",
        "threadgroup",
        "threadgroup_imageblock",
        "uniform",
        "workgroup",
    }
)


def is_private_pointer_parameter(parameter):
    """Return whether a parameter denotes a function-local pointer."""
    vtype = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
    if not isinstance(vtype, PointerType):
        return False
    qualifiers = {
        str(qualifier).lower()
        for qualifier in getattr(parameter, "qualifiers", []) or []
    }
    return not qualifiers.intersection(NON_PRIVATE_POINTER_QUALIFIERS)


def collect_private_pointer_parameters(functions):
    """Collect private pointer parameters that require target array lowering."""
    parameters = {}
    for function in functions:
        function_name = getattr(function, "name", None)
        if not function_name:
            continue
        for parameter in getattr(
            function, "parameters", getattr(function, "params", [])
        ):
            if is_private_pointer_parameter(parameter):
                parameters.setdefault(function_name, {})[
                    parameter.name
                ] = parameter.param_type
    return parameters


def collect_resource_array_size_hints(
    *,
    global_arrays,
    function_arrays,
    fixed_global_array_sizes=None,
    fixed_function_array_sizes=None,
    functions,
    walk_nodes,
    expression_name,
    literal_int_value,
    visible_literal_int_constants,
    function_call_name,
    initial_size,
    format_size,
    initial_literal_int_constants=None,
    strict_fixed_local_array_sizes=False,
    propagate_argument_sizes_to_callee=True,
    validate_fixed_argument_sizes=True,
    required_sizes=None,
):
    """Infer resource-array sizes from literal accesses and call propagation.

    ``required_sizes`` adds sizes unsized global arrays need for other reasons, such as their non-constant indexes. It
    is a mapping, or a callable that receives ``resolved``, the ids of the array-access nodes whose index this
    inference evaluated in every context it scanned them in, and ``approximate_loops``, the ids of the loops it did not
    unroll in some scan (it follows one iteration of those), and returns one
    (:class:`resource_array_bounds.DynamicIndexSizes`). The sizes propagate like those of literal accesses.
    """
    return ResourceArraySizeInference(
        global_arrays=global_arrays,
        function_arrays=function_arrays,
        fixed_global_array_sizes=fixed_global_array_sizes,
        fixed_function_array_sizes=fixed_function_array_sizes,
        functions=functions,
        walk_nodes=walk_nodes,
        expression_name=expression_name,
        literal_int_value=literal_int_value,
        visible_literal_int_constants=visible_literal_int_constants,
        function_call_name=function_call_name,
        initial_size=initial_size,
        format_size=format_size,
        initial_literal_int_constants=initial_literal_int_constants,
        strict_fixed_local_array_sizes=strict_fixed_local_array_sizes,
        propagate_argument_sizes_to_callee=propagate_argument_sizes_to_callee,
        validate_fixed_argument_sizes=validate_fixed_argument_sizes,
        required_sizes=required_sizes,
    ).run()


def collect_private_pointer_array_size_hints(
    *,
    functions,
    walk_nodes,
    expression_name,
    literal_int_value,
    visible_literal_int_constants,
    function_call_name,
    initial_literal_int_constants=None,
    strict_fixed_local_array_sizes=True,
    propagate_argument_sizes_to_callee=True,
    validate_fixed_argument_sizes=True,
):
    """Infer exact target array extents for private pointer parameters."""
    function_arrays = collect_private_pointer_parameters(functions)
    if not function_arrays:
        return {}
    _global_hints, function_hints = collect_resource_array_size_hints(
        global_arrays={},
        function_arrays=function_arrays,
        functions=functions,
        walk_nodes=walk_nodes,
        expression_name=expression_name,
        literal_int_value=literal_int_value,
        visible_literal_int_constants=visible_literal_int_constants,
        function_call_name=function_call_name,
        initial_size=0,
        format_size=lambda size: str(size) if size > 0 else "",
        initial_literal_int_constants=initial_literal_int_constants,
        strict_fixed_local_array_sizes=strict_fixed_local_array_sizes,
        propagate_argument_sizes_to_callee=propagate_argument_sizes_to_callee,
        validate_fixed_argument_sizes=validate_fixed_argument_sizes,
    )
    return function_hints
