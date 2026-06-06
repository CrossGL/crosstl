"""Shared image access analysis and metadata helpers for code generators."""

from ..ast import FunctionCallNode
from ..validation import (
    floating_coordinate_dimension,
    integer_coordinate_dimension,
    is_floating_scalar_type,
    is_integer_scalar_type,
    is_numeric_scalar_type,
    texture_intrinsic_allowed_argument_counts,
    texture_intrinsic_max_argument_count,
    texture_intrinsic_min_argument_count,
)

IMAGE_ATOMIC_INTRINSIC_NAMES = {
    "imageAtomicAdd",
    "imageAtomicMin",
    "imageAtomicMax",
    "imageAtomicAnd",
    "imageAtomicOr",
    "imageAtomicXor",
    "imageAtomicExchange",
    "imageAtomicCompSwap",
}

IMAGE_ATOMIC_VALUE_INTRINSIC_NAMES = IMAGE_ATOMIC_INTRINSIC_NAMES - {
    "imageAtomicCompSwap"
}

IMAGE_ATOMIC_INTEGER_FORMATS = ("r32i", "r32ui")
IMAGE_ATOMIC_EXCHANGE_FORMATS = ("r32i", "r32ui", "r32f")


def is_image_atomic_operation(operation):
    """Return whether an intrinsic is a storage image atomic operation."""
    return operation in IMAGE_ATOMIC_INTRINSIC_NAMES


SUPPORTED_IMAGE_FORMATS = frozenset(
    {
        "r8",
        "r8_snorm",
        "r8i",
        "r8ui",
        "r16",
        "r16_snorm",
        "r16f",
        "r16i",
        "r16ui",
        "r32f",
        "r32i",
        "r32ui",
        "rg8",
        "rg8_snorm",
        "rg8i",
        "rg8ui",
        "rg16",
        "rg16_snorm",
        "rg16f",
        "rg16i",
        "rg16ui",
        "rg32f",
        "rg32i",
        "rg32ui",
        "rgba8",
        "rgba8_snorm",
        "rgba8i",
        "rgba8ui",
        "rgba16",
        "rgba16_snorm",
        "rgba16f",
        "rgba16i",
        "rgba16ui",
        "rgba32f",
        "rgba32i",
        "rgba32ui",
    }
)

IMAGE_FORMAT_CHANNEL_COUNTS = {
    image_format: (
        4
        if image_format.startswith("rgba")
        else 2 if image_format.startswith("rg") else 1
    )
    for image_format in SUPPORTED_IMAGE_FORMATS
}

IMAGE_FORMAT_COMPONENT_KINDS = {
    image_format: (
        "uint"
        if image_format.endswith("ui")
        else "int" if image_format.endswith("i") else "float"
    )
    for image_format in SUPPORTED_IMAGE_FORMATS
}

NUMERIC_COMPONENT_KINDS = frozenset({"float", "int", "uint"})

GLSL_INTEGER_IMAGE_TYPES = frozenset(
    {
        "iimage1D",
        "iimage1DArray",
        "iimage2D",
        "iimage3D",
        "iimage2DArray",
        "iimage2DMS",
        "iimage2DMSArray",
        "uimage1D",
        "uimage1DArray",
        "uimage2D",
        "uimage3D",
        "uimage2DArray",
        "uimage2DMS",
        "uimage2DMSArray",
    }
)

GLSL_FLOAT_IMAGE_RESOURCE_TYPES = frozenset(
    {
        "image1D",
        "image1DArray",
        "image2D",
        "image3D",
        "image2DArray",
        "image2DMS",
        "image2DMSArray",
    }
)

METAL_STORAGE_IMAGE_PREFIXES = (
    "texture1d<",
    "texture1d_array<",
    "texture2d<",
    "texture3d<",
    "texture2d_array<",
    "texture2d_ms<",
    "texture2d_ms_array<",
)

METAL_STORAGE_IMAGE_ACCESS_TOKENS = (
    "access::read>",
    "access::write>",
    "access::read_write>",
)


def collect_function_parameter_names(functions):
    """Return function name -> ordered parameter names."""
    parameter_names = {}
    for func in functions:
        func_name = getattr(func, "name", None)
        if not func_name:
            continue
        parameter_names[func_name] = [
            param.name
            for param in getattr(func, "parameters", getattr(func, "params", []))
        ]
    return parameter_names


def image_operation_access_requirement(func_name):
    """Return the storage image access mode required by an image operation."""
    if func_name == "imageLoad":
        return "read"
    if func_name == "imageStore":
        return "write"
    if func_name in IMAGE_ATOMIC_INTRINSIC_NAMES:
        return "read_write"
    return None


def merge_image_access_requirement(current, incoming):
    """Merge two inferred image access requirements."""
    if incoming is None:
        return current
    if current is None or current == incoming:
        return incoming
    return "read_write"


def collect_function_image_access_requirements(
    functions,
    function_parameter_names,
    iter_nodes,
    function_call_name,
    expression_name,
):
    """Infer function parameter image access contracts from direct and nested calls."""
    functions = list(functions)
    requirements = {
        getattr(func, "name", None): {}
        for func in functions
        if getattr(func, "name", None)
    }
    parameter_names = {
        func_name: list(names) for func_name, names in function_parameter_names.items()
    }
    parameter_sets = {
        func_name: set(names) for func_name, names in parameter_names.items()
    }

    for func in functions:
        func_name = getattr(func, "name", None)
        if not func_name:
            continue
        parameter_set = parameter_sets.get(func_name, set())
        for node in iter_nodes(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            operation = function_call_name(node)
            required_access = image_operation_access_requirement(operation)
            if required_access is None:
                continue
            args = getattr(node, "arguments", getattr(node, "args", []))
            if not args:
                continue
            target_name = expression_name(args[0])
            if target_name not in parameter_set:
                continue
            current = requirements[func_name].get(target_name)
            requirements[func_name][target_name] = merge_image_access_requirement(
                current, required_access
            )

    changed = True
    while changed:
        changed = False
        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_set = parameter_sets.get(func_name, set())
            if not parameter_set:
                continue
            for node in iter_nodes(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                callee_name = function_call_name(node)
                callee_requirements = requirements.get(callee_name)
                if not callee_requirements:
                    continue
                callee_parameters = parameter_names.get(callee_name, [])
                args = getattr(node, "arguments", getattr(node, "args", []))
                for callee_param, required_access in callee_requirements.items():
                    try:
                        index = callee_parameters.index(callee_param)
                    except ValueError:
                        continue
                    if index >= len(args):
                        continue
                    target_name = expression_name(args[index])
                    if target_name not in parameter_set:
                        continue
                    current = requirements[func_name].get(target_name)
                    merged = merge_image_access_requirement(current, required_access)
                    if merged != current:
                        requirements[func_name][target_name] = merged
                        changed = True

    return {name: reqs for name, reqs in requirements.items() if reqs}


def image_access_satisfies_requirement(required_access, actual_access):
    """Return true if an actual storage image access mode satisfies a requirement."""
    if required_access is None:
        return True
    if actual_access is None or actual_access == "read_write":
        return True
    if required_access == "read":
        return actual_access == "read"
    if required_access == "write":
        return actual_access == "write"
    return False


def image_access_requirement_label(required_access, read_write_label="read-write"):
    """Return a diagnostic label for a required image access mode."""
    return {
        "read": "read-capable",
        "write": "write-capable",
        "read_write": read_write_label,
    }.get(required_access, str(required_access))


def image_access_diagnostic_name(access, read_write_label="readwrite"):
    """Return a source-level diagnostic name for an actual image access mode."""
    return {
        "read": "readonly",
        "write": "writeonly",
        "read_write": read_write_label,
    }.get(access, str(access))


def operation_argument_type_error(
    backend_name,
    operation_kind,
    func_name,
    requirement,
    argument_label,
    argument_name,
    argument_type,
):
    """Return a backend resource/texture argument type diagnostic."""
    return (
        f"{backend_name} {operation_kind} operation '{func_name}' requires "
        f"{requirement} {argument_label} argument: {argument_name} has type "
        f"{argument_type}"
    )


def operation_dimension_argument_error(
    backend_name,
    operation_kind,
    func_name,
    expected_dimension,
    value_kind,
    argument_label,
    resource_type,
    argument_name,
    argument_type,
):
    """Return a backend resource/texture argument dimension diagnostic."""
    return (
        f"{backend_name} {operation_kind} operation '{func_name}' requires a "
        f"{expected_dimension}D {value_kind} {argument_label} for "
        f"{resource_type}: {argument_name} has type {argument_type}"
    )


def texture_argument_diagnostic_type(
    arg,
    texture_resource_type,
    expression_name,
    expression_result_type,
    sampler_names,
):
    """Return the diagnostic type label for a texture/resource call argument."""
    texture_type = texture_resource_type(arg)
    if texture_type is not None:
        return texture_type
    if expression_name(arg) in sampler_names:
        return "sampler"
    return expression_result_type(arg)


def validate_texture_operation_arity(
    backend_name,
    operation,
    args,
    texture_resource_operation_names,
    texture_call_uses_explicit_sampler,
):
    """Validate shared texture/image operation arity for a backend."""
    if operation not in texture_resource_operation_names:
        return
    has_explicit_sampler = texture_call_uses_explicit_sampler(args)
    argument_count = len(args)
    min_count = texture_intrinsic_min_argument_count(
        operation,
        has_explicit_sampler,
    )
    if min_count is not None and argument_count < min_count:
        raise ValueError(
            f"{backend_name} texture operation '{operation}' requires at least "
            f"{min_count} argument(s), got {argument_count}"
        )
    allowed_counts = texture_intrinsic_allowed_argument_counts(
        operation,
        has_explicit_sampler,
    )
    if allowed_counts is not None and argument_count not in allowed_counts:
        counts = ", ".join(str(count) for count in allowed_counts)
        raise ValueError(
            f"{backend_name} texture operation '{operation}' accepts "
            f"{counts} argument(s), got {argument_count}"
        )
    max_count = texture_intrinsic_max_argument_count(
        operation,
        has_explicit_sampler,
    )
    if max_count is None or argument_count <= max_count:
        return
    raise ValueError(
        f"{backend_name} texture operation '{operation}' accepts at most "
        f"{max_count} argument(s), got {argument_count}"
    )


INTEGER_COORDINATE_TYPE_NAMES = frozenset(
    {
        "int",
        "uint",
        "ivec2",
        "ivec3",
        "ivec4",
        "uvec2",
        "uvec3",
        "uvec4",
        "int2",
        "int3",
        "int4",
        "uint2",
        "uint3",
        "uint4",
    }
)


def is_integer_coordinate_type_name(type_name, mapped_type=None):
    """Return whether a source or backend type name can address integer texels."""
    return (
        type_name in INTEGER_COORDINATE_TYPE_NAMES
        or mapped_type in INTEGER_COORDINATE_TYPE_NAMES
    )


def integer_coordinate_dimension_from_type_name(type_name, map_type):
    """Return integer coordinate width for a source or backend type name."""
    if not type_name:
        return None
    mapped_type = map_type(type_name)
    return integer_coordinate_dimension(type_name) or integer_coordinate_dimension(
        mapped_type
    )


def _is_mapped_scalar_type_name(type_name, map_type, scalar_type_predicate):
    if not type_name or "[" in str(type_name):
        return False
    mapped_type = map_type(type_name)
    return scalar_type_predicate(mapped_type) or scalar_type_predicate(type_name)


def is_floating_scalar_type_name(type_name, map_type):
    """Return whether a source or backend type name is scalar floating-point."""
    return _is_mapped_scalar_type_name(type_name, map_type, is_floating_scalar_type)


def is_numeric_scalar_type_name(type_name, map_type):
    """Return whether a source or backend type name is scalar numeric."""
    return _is_mapped_scalar_type_name(type_name, map_type, is_numeric_scalar_type)


def is_integer_scalar_type_name(type_name, map_type):
    """Return whether a source or backend type name is scalar integer."""
    return _is_mapped_scalar_type_name(type_name, map_type, is_integer_scalar_type)


def floating_coordinate_dimension_from_type_name(type_name, map_type):
    """Return floating coordinate width for a source or backend type name."""
    if not type_name:
        return None
    mapped_type = map_type(type_name)
    return floating_coordinate_dimension(mapped_type) or floating_coordinate_dimension(
        type_name
    )


def _offset_dimension_from_capability(
    sampling, capability_name, default_dimension, fallback_dimension=None
):
    if capability_name in sampling:
        return default_dimension if sampling[capability_name] else None
    return fallback_dimension


def texture_resource_dimension_descriptor(
    texture_type,
    sampling,
    coordinate_dimension=None,
    offset_dimension=None,
    sample_offset_dimension=None,
    texel_fetch_offset_dimension=None,
    gradient_dimension=None,
    query_lod_coordinate_dimension=None,
    is_multisample=False,
):
    """Build the shared texture/image dimension descriptor shape."""
    sample_offset_dimension = (
        sample_offset_dimension
        if sample_offset_dimension is not None
        else offset_dimension
    )
    if not sampling.get("sample_offset"):
        sample_offset_dimension = None

    if is_multisample:
        texel_fetch_offset_dimension = None
    elif texel_fetch_offset_dimension is None:
        texel_fetch_offset_dimension = offset_dimension

    gather_offset_dimension = _offset_dimension_from_capability(
        sampling, "gather_offset", 2
    )
    compare_offset_dimension = _offset_dimension_from_capability(
        sampling, "compare_offset", 2
    )
    compare_lod_offset_dimension = _offset_dimension_from_capability(
        sampling,
        "compare_lod_offset",
        2,
        fallback_dimension=compare_offset_dimension,
    )
    compare_grad_offset_dimension = _offset_dimension_from_capability(
        sampling,
        "compare_grad_offset",
        2,
        fallback_dimension=compare_offset_dimension,
    )
    gather_compare_offset_dimension = _offset_dimension_from_capability(
        sampling, "gather_compare_offset", 2
    )

    return {
        "texture_type": texture_type,
        "coordinate_dimension": coordinate_dimension,
        "offset_dimension": offset_dimension,
        "sample_offset_dimension": sample_offset_dimension,
        "texel_fetch_offset_dimension": texel_fetch_offset_dimension,
        "gather_offset_dimension": gather_offset_dimension,
        "compare_offset_dimension": compare_offset_dimension,
        "compare_lod_offset_dimension": compare_lod_offset_dimension,
        "compare_grad_offset_dimension": compare_grad_offset_dimension,
        "gather_compare_offset_dimension": gather_compare_offset_dimension,
        "gradient_dimension": gradient_dimension,
        "query_lod_coordinate_dimension": query_lod_coordinate_dimension,
    }


def requires_integer_coordinate(operation, integer_coordinate_intrinsic_names):
    """Return whether an operation validates argument 1 as integer coordinates."""
    return operation in integer_coordinate_intrinsic_names


def image_atomic_format_allowed_names(func_name):
    """Return explicit image formats accepted by an image atomic operation."""
    if func_name == "imageAtomicExchange":
        return IMAGE_ATOMIC_EXCHANGE_FORMATS
    return IMAGE_ATOMIC_INTEGER_FORMATS


def image_atomic_format_requirement(func_name):
    """Return a readable explicit-format requirement for image atomics."""
    allowed = image_atomic_format_allowed_names(func_name)
    if len(allowed) == 2:
        return "r32i or r32ui"
    return "r32i, r32ui, or r32f"


def image_atomic_explicit_format_component_kind(func_name, image_format):
    """Return the component kind when an explicit image atomic format is valid."""
    if image_format not in image_atomic_format_allowed_names(func_name):
        return None
    return image_format_component_kind(image_format)


def image_atomic_value_argument_indices(func_name, has_sample):
    """Return the argument slice containing data operands for an image atomic."""
    if func_name == "imageAtomicCompSwap":
        first_value_index = 3 if has_sample else 2
        return first_value_index, first_value_index + 2
    value_index = 3 if has_sample else 2
    return value_index, value_index + 1


def image_atomic_value_arguments(func_name, args, has_sample):
    """Return data operands for an image atomic call."""
    start, end = image_atomic_value_argument_indices(func_name, has_sample)
    return args[start:end]


def should_validate_image_atomic_component_kind(func_name, component_kind):
    """Return true when an image atomic's data/result kind should be checked."""
    if component_kind not in NUMERIC_COMPONENT_KINDS:
        return False
    return component_kind != "float" or func_name == "imageAtomicExchange"


def image_atomic_value_kind_mismatch(
    func_name, value_args, component_kind, expression_kind
):
    """Return the first image atomic data operand with an incompatible kind."""
    if not should_validate_image_atomic_component_kind(func_name, component_kind):
        return None
    for value_arg in value_args:
        value_kind = expression_kind(value_arg)
        if value_kind is None or value_kind == component_kind:
            continue
        return value_arg, value_kind
    return None


def image_atomic_result_kind_mismatch(expected_kind, component_kind):
    """Return the expected result kind when it conflicts with an atomic kind."""
    if expected_kind is None or component_kind not in NUMERIC_COMPONENT_KINDS:
        return None
    if expected_kind == component_kind:
        return None
    return expected_kind


def image_atomic_storage_component_kind(func_name, component_type):
    """Return component kind for storage image atomics without explicit formats."""
    if component_type in {"int", "uint"}:
        return component_type
    if func_name == "imageAtomicExchange" and component_type == "float":
        return "float"
    return None


def image_atomic_format_error(backend_name, func_name, image_format):
    """Return an explicit image atomic format diagnostic."""
    return (
        f"{backend_name} image atomic operation '{func_name}' requires "
        f"{image_atomic_format_requirement(func_name)} image format, got "
        f"{image_format}"
    )


def image_atomic_value_kind_error(
    backend_name, func_name, format_label, component_kind, value_name, value_kind
):
    """Return an image atomic data argument kind diagnostic."""
    return (
        f"{backend_name} image atomic operation '{func_name}' requires "
        f"{component_kind} data argument for {format_label} images: "
        f"{value_name} has type {value_kind}"
    )


def image_atomic_result_kind_error(
    backend_name, func_name, format_label, component_kind, expected_kind
):
    """Return an image atomic result context kind diagnostic."""
    return (
        f"{backend_name} image atomic operation '{func_name}' requires "
        f"{component_kind} result context for {format_label} images: "
        f"expected {expected_kind}"
    )


def image_atomic_resource_type_error(backend_name, func_name, image_type):
    """Return a scalar storage image requirement diagnostic for image atomics."""
    return (
        f"{backend_name} image atomic operation '{func_name}' requires a scalar "
        f"r32i or r32ui integer storage image, got {image_type}"
    )


def resolve_image_atomic_component_kind(
    func_name, image_format, component_type, backend_name, resource_type
):
    """Return the image atomic component kind or raise a backend diagnostic."""
    if image_format is not None:
        component_kind = image_atomic_explicit_format_component_kind(
            func_name, image_format
        )
        if component_kind is None:
            raise ValueError(
                image_atomic_format_error(backend_name, func_name, image_format)
            )
        return component_kind

    component_kind = image_atomic_storage_component_kind(func_name, component_type)
    if component_kind is not None:
        return component_kind
    raise ValueError(
        image_atomic_resource_type_error(backend_name, func_name, resource_type)
    )


def storage_image_atomic_zero_value(component_type):
    """Return a typed zero literal for unsupported storage image atomic fallbacks."""
    if component_type == "uint":
        return "0u"
    if component_type == "float":
        return "0.0"
    return "0"


def unsupported_image_atomic_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported storage image atomic fallback expression."""
    return (
        f"/* unsupported {backend_name} image atomic resource call: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_image_atomic_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample image atomic fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample image atomic: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_image_store_expression(backend_name, resource_type):
    """Return an unsupported multisample image store fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample image store: "
        f"imageStore on {resource_type} */ ((void)0)"
    )


def image_atomic_helper_descriptor_fields(
    operation, component_type, suffix_family, coord_type
):
    """Return shared descriptor fields for backend image atomic helpers."""
    if component_type not in {"int", "uint"} or not suffix_family or not coord_type:
        return None
    return {
        "helper_name": (
            f"{operation}_{'i' if component_type == 'int' else 'u'}" f"{suffix_family}"
        ),
        "return_type": component_type,
        "coord_type": coord_type,
    }


def image_atomic_helper_resource_metadata(
    texture_family,
    suffix_by_family,
    coord_type_by_family,
    sample_families=None,
    extra_fields_by_family=None,
):
    """Return shared image atomic metadata for a backend resource family."""
    suffix_family = suffix_by_family.get(texture_family)
    coord_type = coord_type_by_family.get(texture_family)
    if not suffix_family or not coord_type:
        return None

    metadata = {
        "suffix_family": suffix_family,
        "coord_type": coord_type,
    }
    if sample_families and texture_family in sample_families:
        metadata["has_sample"] = True
    if extra_fields_by_family is not None:
        extra_fields = extra_fields_by_family.get(texture_family)
        if extra_fields is None:
            return None
        metadata.update(extra_fields)
    return metadata


def resource_query_get_dimensions_descriptor(
    size_return_type,
    dimensions,
    size_return_expr,
    function_params="",
    get_dimensions_args=None,
):
    """Return shared metadata for resource dimension query helpers."""
    dimensions = tuple(dimensions)
    return {
        "size_return_type": size_return_type,
        "function_params": function_params,
        "dimensions": dimensions,
        "get_dimensions_args": (
            tuple(get_dimensions_args)
            if get_dimensions_args is not None
            else dimensions
        ),
        "size_return_expr": size_return_expr,
    }


def resource_query_size_components_descriptor(
    size_return_type,
    size_components,
    tail_dimensions=(),
    function_params="",
    get_dimensions_prefix=(),
):
    """Return GetDimensions metadata from the dimensions that form the size."""
    size_components = tuple(size_components)
    tail_dimensions = tuple(tail_dimensions)
    dimensions = size_components + tail_dimensions
    if size_return_type == "int":
        size_return_expr = f"int({size_components[0]})"
    else:
        size_return_expr = f"{size_return_type}({', '.join(size_components)})"
    get_dimensions_prefix = tuple(get_dimensions_prefix)
    get_dimensions_args = None
    if get_dimensions_prefix:
        get_dimensions_args = get_dimensions_prefix + dimensions
    return resource_query_get_dimensions_descriptor(
        size_return_type,
        dimensions,
        size_return_expr,
        function_params=function_params,
        get_dimensions_args=get_dimensions_args,
    )


def resource_query_size_helper_descriptor(
    query_descriptor, include_function_fields=True
):
    """Return size-helper metadata from a resource dimension query descriptor."""
    if query_descriptor is None:
        return None

    descriptor = {
        "return_type": query_descriptor["size_return_type"],
        "dimensions": query_descriptor["dimensions"],
        "return_expr": query_descriptor["size_return_expr"],
    }
    if include_function_fields:
        descriptor.update(
            {
                "function_params": query_descriptor["function_params"],
                "get_dimensions_args": query_descriptor["get_dimensions_args"],
            }
        )
    return descriptor


def resource_query_scalar_helper_descriptor(
    query_descriptor, return_expr, return_type="int"
):
    """Return scalar-helper metadata from a resource dimension query descriptor."""
    if query_descriptor is None:
        return None

    return {
        "return_type": return_type,
        "function_params": "",
        "dimensions": query_descriptor["dimensions"],
        "get_dimensions_args": query_descriptor["get_dimensions_args"],
        "return_expr": return_expr,
    }


def resource_query_scalar_constant_helper_descriptor(return_expr, return_type="int"):
    """Return scalar-helper metadata for queries with a constant result."""
    return {
        "return_type": return_type,
        "function_params": "",
        "dimensions": (),
        "get_dimensions_args": (),
        "return_expr": return_expr,
    }


def resource_query_method_size_descriptor(return_type, dimensions):
    """Return size-query metadata for APIs that expose per-dimension methods."""
    return {
        "return_type": return_type,
        "dimensions": tuple(dimensions),
    }


def unsupported_texture_query_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported texture query fallback expression."""
    return (
        f"/* unsupported {backend_name} texture query: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_texture_query_lod_expression(backend_name, resource_type):
    """Return an unsupported textureQueryLod fallback expression."""
    return unsupported_texture_query_expression(
        backend_name,
        "textureQueryLod",
        resource_type,
        texture_query_lod_zero_value(backend_name),
    )


def unsupported_texture_query_levels_expression(backend_name, resource_type):
    """Return an unsupported textureQueryLevels fallback expression."""
    return unsupported_texture_query_expression(
        backend_name,
        "textureQueryLevels",
        resource_type,
        texture_query_levels_zero_value(),
    )


def unsupported_multisample_texture_query_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture query fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture query: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_query_lod_expression(backend_name, resource_type):
    """Return an unsupported multisample textureQueryLod fallback expression."""
    return unsupported_multisample_texture_query_expression(
        backend_name,
        "textureQueryLod",
        resource_type,
        texture_query_lod_zero_value(backend_name),
    )


def texture_query_levels_multisample_expression():
    """Return the mip-level count for multisample textures."""
    return "1"


def texture_query_levels_zero_value():
    """Return the fallback mip-level count for unsupported texture queries."""
    return "0"


def texture_samples_query_expression(backend_name, texture_name):
    """Return the supported multisample texture sample-count expression."""
    return {
        "GLSL": f"textureSamples({texture_name})",
        "DirectX": f"textureSamples({texture_name})",
        "Metal": f"int({texture_name}.get_num_samples())",
    }[backend_name]


def texture_samples_query_requirement_name(backend_name):
    """Return the backend term used in unsupported sample-query diagnostics."""
    return {
        "GLSL": "sampler",
        "DirectX": "texture",
        "Metal": "texture",
    }[backend_name]


def unsupported_texture_samples_query_expression(
    backend_name, multisample_resource_name
):
    """Return an unsupported texture samples fallback expression."""
    return (
        f"/* unsupported {backend_name} texture samples query: "
        f"requires multisample {multisample_resource_name} */ 0"
    )


def unsupported_texture_samples_query_call_expression(backend_name):
    """Return an unsupported textureSamples fallback expression."""
    return unsupported_texture_samples_query_expression(
        backend_name, texture_samples_query_requirement_name(backend_name)
    )


def unsupported_multisample_texture_call_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture-call fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture call: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_call_vector_expression(
    backend_name, operation, resource_type
):
    """Return an unsupported multisample texture-call fallback with vector zero."""
    return unsupported_multisample_texture_call_expression(
        backend_name,
        operation,
        resource_type,
        texture_vector_zero_value(backend_name),
    )


def unsupported_multisample_texture_compare_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture comparison fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture comparison: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_compare_scalar_expression(
    backend_name, operation, resource_type
):
    """Return an unsupported multisample texture comparison with scalar zero."""
    return unsupported_multisample_texture_compare_expression(
        backend_name,
        operation,
        resource_type,
        texture_scalar_zero_value(backend_name),
    )


def unsupported_multisample_texture_gather_compare_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture gather-compare fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture gather comparison: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_gather_compare_vector_expression(
    backend_name, operation, resource_type
):
    """Return an unsupported multisample gather-compare fallback with vector zero."""
    return unsupported_multisample_texture_gather_compare_expression(
        backend_name,
        operation,
        resource_type,
        texture_vector_zero_value(backend_name),
    )


def unsupported_storage_image_texture_comparison_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported storage-image texture comparison fallback."""
    return (
        f"/* unsupported {backend_name} storage image texture comparison: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_storage_image_texture_comparison_scalar_expression(
    backend_name, operation, resource_type
):
    """Return an unsupported storage-image texture comparison with scalar zero."""
    return unsupported_storage_image_texture_comparison_expression(
        backend_name,
        operation,
        resource_type,
        texture_scalar_zero_value(backend_name),
    )


def unsupported_storage_image_texture_operation_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported storage-image texture operation fallback."""
    return (
        f"/* unsupported {backend_name} storage image texture operation: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_storage_image_texture_operation_vector_expression(
    backend_name, operation, resource_type
):
    """Return an unsupported storage-image texture operation with vector zero."""
    return unsupported_storage_image_texture_operation_expression(
        backend_name,
        operation,
        resource_type,
        texture_vector_zero_value(backend_name),
    )


PROJECTED_TEXTURE_BASIC_INTRINSIC_NAMES = frozenset(
    {
        "textureProj",
    }
)


PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureProjOffset",
    }
)


PROJECTED_TEXTURE_LOD_INTRINSIC_NAMES = frozenset(
    {
        "textureProjLod",
    }
)


PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureProjLodOffset",
    }
)


PROJECTED_TEXTURE_GRAD_INTRINSIC_NAMES = frozenset(
    {
        "textureProjGrad",
    }
)


PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureProjGradOffset",
    }
)


PROJECTED_TEXTURE_INTRINSIC_NAMES = (
    PROJECTED_TEXTURE_BASIC_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_LOD_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_GRAD_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES
)


PROJECTED_TEXTURE_OFFSET_INTRINSIC_NAMES = (
    PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES
)


PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES = frozenset(
    {
        "textureCompareProj",
        "textureCompareProjOffset",
        "textureCompareProjLod",
        "textureCompareProjLodOffset",
        "textureCompareProjGrad",
        "textureCompareProjGradOffset",
    }
)


TEXTURE_COMPARE_BASIC_INTRINSIC_NAMES = frozenset(
    {
        "textureCompare",
        "textureCompareProj",
    }
)


TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureCompareOffset",
        "textureCompareProjOffset",
    }
)


TEXTURE_COMPARE_LOD_INTRINSIC_NAMES = frozenset(
    {
        "textureCompareLod",
        "textureCompareProjLod",
    }
)


TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureCompareLodOffset",
        "textureCompareProjLodOffset",
    }
)


TEXTURE_COMPARE_GRAD_INTRINSIC_NAMES = frozenset(
    {
        "textureCompareGrad",
        "textureCompareProjGrad",
    }
)


TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureCompareGradOffset",
        "textureCompareProjGradOffset",
    }
)


TEXTURE_COMPARE_OFFSET_OPERATION_NAMES = (
    TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES
    | TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES
    | TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES
)


TEXTURE_COMPARE_INTRINSIC_NAMES = (
    TEXTURE_COMPARE_BASIC_INTRINSIC_NAMES
    | TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES
    | TEXTURE_COMPARE_LOD_INTRINSIC_NAMES
    | TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES
    | TEXTURE_COMPARE_GRAD_INTRINSIC_NAMES
    | TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES
)


TEXTURE_COMPARE_NON_PROJECTED_INTRINSIC_NAMES = (
    TEXTURE_COMPARE_INTRINSIC_NAMES - PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES
)


TEXTURE_COMPARE_NON_PROJECTED_OFFSET_OPERATION_NAMES = (
    TEXTURE_COMPARE_OFFSET_OPERATION_NAMES - PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES
)


TEXTURE_SAMPLE_BASIC_INTRINSIC_NAMES = frozenset(
    {
        "texture",
    }
)


TEXTURE_SAMPLE_LOD_INTRINSIC_NAMES = frozenset(
    {
        "textureLod",
    }
)


TEXTURE_SAMPLE_GRAD_INTRINSIC_NAMES = frozenset(
    {
        "textureGrad",
    }
)


TEXTURE_SAMPLE_INTRINSIC_NAMES = (
    TEXTURE_SAMPLE_BASIC_INTRINSIC_NAMES
    | TEXTURE_SAMPLE_LOD_INTRINSIC_NAMES
    | TEXTURE_SAMPLE_GRAD_INTRINSIC_NAMES
)


TEXTURE_SAMPLE_BASIC_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureOffset",
    }
)


TEXTURE_SAMPLE_LOD_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureLodOffset",
    }
)


TEXTURE_SAMPLE_GRAD_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureGradOffset",
    }
)


TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES = (
    TEXTURE_SAMPLE_BASIC_OFFSET_INTRINSIC_NAMES
    | TEXTURE_SAMPLE_LOD_OFFSET_INTRINSIC_NAMES
    | TEXTURE_SAMPLE_GRAD_OFFSET_INTRINSIC_NAMES
)


TEXTURE_SAMPLING_OFFSET_INTRINSIC_NAMES = (
    TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES | PROJECTED_TEXTURE_OFFSET_INTRINSIC_NAMES
)


TEXTURE_GATHER_BASIC_INTRINSIC_NAMES = frozenset(
    {
        "textureGather",
    }
)


TEXTURE_GATHER_SINGLE_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureGatherOffset",
    }
)


TEXTURE_GATHER_MULTI_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureGatherOffsets",
    }
)


TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES = (
    TEXTURE_GATHER_SINGLE_OFFSET_INTRINSIC_NAMES
    | TEXTURE_GATHER_MULTI_OFFSET_INTRINSIC_NAMES
)


TEXTURE_GATHER_INTRINSIC_NAMES = (
    TEXTURE_GATHER_BASIC_INTRINSIC_NAMES | TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES
)


TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES = frozenset(
    {
        "textureGatherCompare",
        "textureGatherCompareOffset",
    }
)


TEXTURE_GATHER_COMPARE_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "textureGatherCompareOffset",
    }
)


TEXTURE_QUERY_LOD_INTRINSIC_NAMES = frozenset(
    {
        "textureQueryLod",
    }
)


TEXTURE_QUERY_LEVELS_INTRINSIC_NAMES = frozenset(
    {
        "textureQueryLevels",
    }
)


TEXTURE_QUERY_SIZE_INTRINSIC_NAMES = frozenset(
    {
        "textureSize",
    }
)


TEXTURE_QUERY_SAMPLES_INTRINSIC_NAMES = frozenset(
    {
        "textureSamples",
    }
)


TEXTURE_QUERY_INTRINSIC_NAMES = (
    TEXTURE_QUERY_LOD_INTRINSIC_NAMES
    | TEXTURE_QUERY_LEVELS_INTRINSIC_NAMES
    | TEXTURE_QUERY_SIZE_INTRINSIC_NAMES
    | TEXTURE_QUERY_SAMPLES_INTRINSIC_NAMES
)


RESOURCE_QUERY_SIZE_INTRINSIC_NAMES = frozenset(
    {
        "textureSize",
        "imageSize",
    }
)


RESOURCE_QUERY_SAMPLES_INTRINSIC_NAMES = frozenset(
    {
        "textureSamples",
        "imageSamples",
    }
)


TEXTURE_TEXEL_FETCH_BASIC_INTRINSIC_NAMES = frozenset(
    {
        "texelFetch",
    }
)


TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES = frozenset(
    {
        "texelFetchOffset",
    }
)


TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES = (
    TEXTURE_TEXEL_FETCH_BASIC_INTRINSIC_NAMES
    | TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES
)


TEXTURE_OFFSET_INTRINSIC_NAMES = (
    TEXTURE_SAMPLING_OFFSET_INTRINSIC_NAMES
    | TEXTURE_COMPARE_OFFSET_OPERATION_NAMES
    | TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES
    | TEXTURE_GATHER_COMPARE_OFFSET_INTRINSIC_NAMES
    | TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES
)


TEXTURE_SAMPLING_INTRINSIC_NAMES = (
    TEXTURE_SAMPLE_INTRINSIC_NAMES
    | TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES
    | PROJECTED_TEXTURE_INTRINSIC_NAMES
)


TEXTURE_RESOURCE_INTRINSIC_NAMES = (
    TEXTURE_SAMPLING_INTRINSIC_NAMES
    | TEXTURE_COMPARE_INTRINSIC_NAMES
    | TEXTURE_GATHER_INTRINSIC_NAMES
    | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
    | TEXTURE_QUERY_INTRINSIC_NAMES
    | TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES
)


TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES = (
    TEXTURE_SAMPLING_INTRINSIC_NAMES
    | TEXTURE_COMPARE_INTRINSIC_NAMES
    | TEXTURE_GATHER_INTRINSIC_NAMES
    | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
    | TEXTURE_QUERY_LOD_INTRINSIC_NAMES
)


STORAGE_IMAGE_TEXTURE_COMPARISON_OPERATIONS = TEXTURE_COMPARE_INTRINSIC_NAMES


STORAGE_IMAGE_TEXTURE_OPERATIONS = (
    TEXTURE_SAMPLING_INTRINSIC_NAMES
    | TEXTURE_GATHER_INTRINSIC_NAMES
    | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES
    | TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES
)


def is_storage_image_texture_comparison_operation(operation):
    """Return whether a texture intrinsic is an unsupported storage-image comparison."""
    return operation in STORAGE_IMAGE_TEXTURE_COMPARISON_OPERATIONS


def is_storage_image_texture_operation(operation):
    """Return whether a texture intrinsic is unsupported for storage images."""
    return operation in STORAGE_IMAGE_TEXTURE_OPERATIONS


def is_projected_texture_operation(operation):
    """Return whether a texture intrinsic uses projected coordinates."""
    return operation in PROJECTED_TEXTURE_INTRINSIC_NAMES


def is_projected_texture_basic_operation(operation):
    """Return whether a projected texture intrinsic uses implicit derivatives."""
    return operation in PROJECTED_TEXTURE_BASIC_INTRINSIC_NAMES


def is_projected_texture_basic_offset_operation(operation):
    """Return whether a projected texture intrinsic uses offset and implicit derivatives."""
    return operation in PROJECTED_TEXTURE_BASIC_OFFSET_INTRINSIC_NAMES


def is_projected_texture_lod_operation(operation):
    """Return whether a projected texture intrinsic uses explicit LOD."""
    return operation in PROJECTED_TEXTURE_LOD_INTRINSIC_NAMES


def is_projected_texture_lod_offset_operation(operation):
    """Return whether a projected texture intrinsic uses explicit LOD and offset."""
    return operation in PROJECTED_TEXTURE_LOD_OFFSET_INTRINSIC_NAMES


def is_projected_texture_grad_operation(operation):
    """Return whether a projected texture intrinsic uses explicit gradients."""
    return operation in PROJECTED_TEXTURE_GRAD_INTRINSIC_NAMES


def is_projected_texture_grad_offset_operation(operation):
    """Return whether a projected texture intrinsic uses gradients and offset."""
    return operation in PROJECTED_TEXTURE_GRAD_OFFSET_INTRINSIC_NAMES


def is_projected_texture_compare_operation(operation):
    """Return whether a texture compare intrinsic uses projected coordinates."""
    return operation in PROJECTED_TEXTURE_COMPARE_INTRINSIC_NAMES


def is_texture_compare_operation(operation):
    """Return whether a texture intrinsic is a depth-compare operation."""
    return operation in TEXTURE_COMPARE_INTRINSIC_NAMES


def is_texture_compare_basic_operation(operation):
    """Return whether a texture compare intrinsic uses base compare sampling."""
    return operation in TEXTURE_COMPARE_BASIC_INTRINSIC_NAMES


def is_texture_compare_lod_operation(operation):
    """Return whether a texture compare intrinsic uses explicit LOD."""
    return operation in TEXTURE_COMPARE_LOD_INTRINSIC_NAMES


def is_texture_compare_grad_operation(operation):
    """Return whether a texture compare intrinsic uses explicit gradients."""
    return operation in TEXTURE_COMPARE_GRAD_INTRINSIC_NAMES


def is_texture_compare_non_projected_operation(operation):
    """Return whether a texture compare intrinsic is not projected."""
    return operation in TEXTURE_COMPARE_NON_PROJECTED_INTRINSIC_NAMES


def is_texture_sample_operation(operation):
    """Return whether a texture intrinsic is a basic sample operation."""
    return operation in TEXTURE_SAMPLE_INTRINSIC_NAMES


def is_texture_sampling_operation(operation):
    """Return whether a texture intrinsic performs non-compare sampling."""
    return operation in TEXTURE_SAMPLING_INTRINSIC_NAMES


def is_texture_sample_basic_operation(operation):
    """Return whether a texture sample intrinsic uses implicit derivatives."""
    return operation in TEXTURE_SAMPLE_BASIC_INTRINSIC_NAMES


def is_texture_sample_lod_operation(operation):
    """Return whether a texture sample intrinsic uses explicit LOD."""
    return operation in TEXTURE_SAMPLE_LOD_INTRINSIC_NAMES


def is_texture_sample_grad_operation(operation):
    """Return whether a texture sample intrinsic uses explicit gradients."""
    return operation in TEXTURE_SAMPLE_GRAD_INTRINSIC_NAMES


def is_texture_sample_offset_operation(operation):
    """Return whether a texture intrinsic is a sample-offset operation."""
    return operation in TEXTURE_SAMPLE_OFFSET_INTRINSIC_NAMES


def is_texture_sample_basic_offset_operation(operation):
    """Return whether a texture sample intrinsic uses offset with implicit derivatives."""
    return operation in TEXTURE_SAMPLE_BASIC_OFFSET_INTRINSIC_NAMES


def is_texture_sample_lod_offset_operation(operation):
    """Return whether a texture sample intrinsic uses explicit LOD and offset."""
    return operation in TEXTURE_SAMPLE_LOD_OFFSET_INTRINSIC_NAMES


def is_texture_sample_grad_offset_operation(operation):
    """Return whether a texture sample intrinsic uses gradients and offset."""
    return operation in TEXTURE_SAMPLE_GRAD_OFFSET_INTRINSIC_NAMES


def is_texture_sampling_offset_operation(operation):
    """Return whether a texture sampling intrinsic uses an offset argument."""
    return operation in TEXTURE_SAMPLING_OFFSET_INTRINSIC_NAMES


def is_texture_gather_operation(operation):
    """Return whether a texture intrinsic is a gather operation."""
    return operation in TEXTURE_GATHER_INTRINSIC_NAMES


def is_texture_gather_basic_operation(operation):
    """Return whether a texture gather intrinsic uses no offset arguments."""
    return operation in TEXTURE_GATHER_BASIC_INTRINSIC_NAMES


def is_texture_gather_single_offset_operation(operation):
    """Return whether a texture gather intrinsic uses one offset argument."""
    return operation in TEXTURE_GATHER_SINGLE_OFFSET_INTRINSIC_NAMES


def is_texture_gather_multi_offset_operation(operation):
    """Return whether a texture gather intrinsic uses four offset arguments."""
    return operation in TEXTURE_GATHER_MULTI_OFFSET_INTRINSIC_NAMES


def is_texture_gather_offset_operation(operation):
    """Return whether a texture gather intrinsic uses offset argument(s)."""
    return operation in TEXTURE_GATHER_OFFSET_INTRINSIC_NAMES


def is_texture_gather_compare_operation(operation):
    """Return whether a texture intrinsic is a gather-compare operation."""
    return operation in TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES


def is_texture_gather_compare_offset_operation(operation):
    """Return whether a texture gather-compare intrinsic uses an offset."""
    return operation in TEXTURE_GATHER_COMPARE_OFFSET_INTRINSIC_NAMES


def is_texture_query_operation(operation):
    """Return whether a texture intrinsic is a query operation."""
    return operation in TEXTURE_QUERY_INTRINSIC_NAMES


def is_texture_query_lod_operation(operation):
    """Return whether a texture query intrinsic reads LOD."""
    return operation in TEXTURE_QUERY_LOD_INTRINSIC_NAMES


def is_texture_query_levels_operation(operation):
    """Return whether a texture query intrinsic reads mip level count."""
    return operation in TEXTURE_QUERY_LEVELS_INTRINSIC_NAMES


def is_texture_size_query_operation(operation):
    """Return whether a texture query intrinsic reads texture dimensions."""
    return operation in TEXTURE_QUERY_SIZE_INTRINSIC_NAMES


def is_texture_samples_query_operation(operation):
    """Return whether a texture query intrinsic reads sample count."""
    return operation in TEXTURE_QUERY_SAMPLES_INTRINSIC_NAMES


def is_resource_size_query_operation(operation):
    """Return whether a resource query intrinsic reads resource dimensions."""
    return operation in RESOURCE_QUERY_SIZE_INTRINSIC_NAMES


def is_resource_samples_query_operation(operation):
    """Return whether a resource query intrinsic reads sample count."""
    return operation in RESOURCE_QUERY_SAMPLES_INTRINSIC_NAMES


def texture_query_lod_coordinate_type_error(
    backend_name, operation, coordinate_name, coordinate_type
):
    """Return the diagnostic for non-floating textureQueryLod coordinates."""
    return operation_argument_type_error(
        backend_name,
        "texture query",
        operation,
        "a floating",
        "coordinate",
        coordinate_name,
        coordinate_type,
    )


def texture_query_lod_coordinate_dimension_error(
    backend_name,
    operation,
    expected_dimension,
    resource_type,
    coordinate_name,
    coordinate_type,
):
    """Return the diagnostic for wrong-dimensional textureQueryLod coordinates."""
    return operation_dimension_argument_error(
        backend_name,
        "texture query",
        operation,
        expected_dimension,
        "floating",
        "coordinate",
        resource_type,
        coordinate_name,
        coordinate_type,
    )


def is_texture_compare_offset_operation(operation):
    """Return whether a texture compare intrinsic uses a direct offset."""
    return operation in TEXTURE_COMPARE_OFFSET_INTRINSIC_NAMES


def is_texture_compare_lod_offset_operation(operation):
    """Return whether a texture compare intrinsic uses lod and offset arguments."""
    return operation in TEXTURE_COMPARE_LOD_OFFSET_INTRINSIC_NAMES


def is_texture_compare_grad_offset_operation(operation):
    """Return whether a texture compare intrinsic uses gradients and an offset."""
    return operation in TEXTURE_COMPARE_GRAD_OFFSET_INTRINSIC_NAMES


def is_texture_compare_any_offset_operation(operation):
    """Return whether any texture compare intrinsic uses an offset."""
    return operation in TEXTURE_COMPARE_OFFSET_OPERATION_NAMES


def texture_compare_projected_coordinate_error(backend_name):
    """Return the backend-specific projected compare coordinate diagnostic."""
    return {
        "GLSL": (
            "requires sampler2DShadow vec3/vec4 or sampler2DArrayShadow "
            "vec4 projection coordinates"
        ),
        "DirectX": (
            "requires Texture2D vec3/vec4 or Texture2DArray "
            "vec4 projection coordinates"
        ),
        "Metal": (
            "requires depth2d vec3/vec4 or depth2d_array " "vec4 projection coordinates"
        ),
    }[backend_name]


def texture_compare_coordinate_error():
    """Return the diagnostic for unsupported non-projected compare coordinates."""
    return "requires supported shadow texture coordinates"


def texture_coordinate_arguments_error():
    """Return the diagnostic for missing texture/coordinate arguments."""
    return "requires texture and coordinate arguments"


def texture_compare_argument_error():
    """Return the diagnostic for missing texture compare arguments."""
    return "requires a compare argument"


def texture_gather_capability_error():
    """Return the diagnostic for unsupported texture gather resources."""
    return "requires 2D, 2D-array, cube, or cube-array textures"


def texture_gather_offset_capability_error():
    """Return the diagnostic for unsupported texture gather offsets."""
    return "offsets require 2D or 2D-array textures"


def texture_gather_component_count_error():
    """Return the diagnostic for too many basic gather component arguments."""
    return "accepts at most one component argument"


def texture_gather_offset_argument_count_error():
    """Return the diagnostic for invalid single-offset gather argument counts."""
    return "requires offset and optional component arguments"


def texture_gather_offsets_argument_count_error():
    """Return the diagnostic for invalid multi-offset gather argument counts."""
    return "requires a typed offsets array or four offset arguments"


def texture_gather_operation_error():
    """Return the diagnostic for unhandled gather operation shapes."""
    return "requires a gather operation"


def texture_gather_component_literal_error():
    """Return the diagnostic for invalid gather component literals."""
    return "component literal must be 0, 1, 2, or 3"


def texture_sample_offset_capability_error(backend_name):
    """Return the backend-specific unsupported sample-offset diagnostic."""
    return {
        "GLSL": "offsets require 1D, 2D, 2D-array, 3D, " "or planar shadow samplers",
        "DirectX": "offsets require 1D, 2D, 2D-array, or 3D textures",
        "Metal": "offsets require 2D, 2D-array, or 3D textures",
    }[backend_name]


def projected_texture_offset_capability_error():
    """Return the diagnostic for unsupported projected texture offsets."""
    return "offsets require 2D, 2D-array, or 3D textures"


def texture_sample_offset_extra_argument_count_error(operation, argument_count):
    """Return the shape-specific sample-offset argument-count diagnostic reason."""
    if is_texture_sample_basic_offset_operation(operation):
        if argument_count in {1, 2}:
            return None
        return "requires offset and optional bias arguments"
    if is_texture_sample_lod_offset_operation(operation):
        if argument_count == 2:
            return None
        return "requires lod and offset arguments"
    if is_texture_sample_grad_offset_operation(operation):
        if argument_count == 3:
            return None
        return "requires gradient x, gradient y, and offset arguments"
    return None


def projected_texture_extra_argument_count_error(operation, argument_count):
    """Return the shape-specific projected texture argument-count diagnostic reason."""
    if is_projected_texture_basic_operation(operation):
        if argument_count in {0, 1}:
            return None
        return "accepts at most one bias argument"
    if is_projected_texture_basic_offset_operation(operation):
        if argument_count in {1, 2}:
            return None
        return "requires offset and optional bias arguments"
    if is_projected_texture_lod_operation(operation):
        if argument_count == 1:
            return None
        return "requires one lod argument"
    if is_projected_texture_lod_offset_operation(operation):
        if argument_count == 2:
            return None
        return "requires lod and offset arguments"
    if is_projected_texture_grad_operation(operation):
        if argument_count == 2:
            return None
        return "requires gradient x and gradient y arguments"
    if is_projected_texture_grad_offset_operation(operation):
        if argument_count == 3:
            return None
        return "requires gradient x, gradient y, and offset arguments"
    return None


def unsupported_texture_offset_operation_error():
    """Return the diagnostic for unhandled texture offset operation shapes."""
    return "is not a supported texture offset operation"


def unsupported_projected_texture_operation_error():
    """Return the diagnostic for unhandled projected texture operation shapes."""
    return "unsupported projected texture operation"


def texture_compare_projected_lod_array_error():
    """Return the diagnostic for unsupported projected array compare LOD."""
    return "projected explicit LOD is not supported for sampler2DArrayShadow"


def unsupported_texture_compare_operation_error(projected=False):
    """Return the diagnostic for unhandled compare operation shapes."""
    if projected:
        return "is not a supported projected shadow compare operation"
    return "is not a supported shadow compare operation"


def texture_compare_offset_capability_error(backend_name):
    """Return the backend-specific unsupported compare-offset diagnostic."""
    return {
        "GLSL": "offsets require 2D or 2D-array shadow samplers",
        "DirectX": "offsets require 1D, 1D-array, 2D, or 2D-array textures",
        "Metal": "offsets require 2D or 2D-array depth textures",
    }[backend_name]


def texture_compare_extra_argument_count_error(operation, argument_count):
    """Return the shape-specific compare argument-count diagnostic reason."""
    if is_texture_compare_basic_operation(operation):
        expected_count = 1
        reason = "accepts no extra arguments"
    elif is_texture_compare_offset_operation(operation):
        expected_count = 2
        reason = "requires compare and offset arguments"
    elif is_texture_compare_lod_operation(operation):
        expected_count = 2
        reason = "requires compare and lod arguments"
    elif is_texture_compare_lod_offset_operation(operation):
        expected_count = 3
        reason = "requires compare, lod, and offset arguments"
    elif is_texture_compare_grad_operation(operation):
        expected_count = 3
        reason = "requires compare, gradient x, and gradient y arguments"
    elif is_texture_compare_grad_offset_operation(operation):
        expected_count = 4
        reason = "requires compare, gradient x, gradient y, and offset arguments"
    else:
        return None
    if argument_count == expected_count:
        return None
    return reason


def texture_gather_compare_extra_argument_count_error(operation, argument_count):
    """Return the shape-specific gather-compare argument-count diagnostic reason."""
    if is_texture_gather_compare_offset_operation(operation):
        expected_count = 2
        reason = "requires compare and offset arguments"
    elif is_texture_gather_compare_operation(operation):
        expected_count = 1
        reason = "accepts no extra arguments"
    else:
        return None
    if argument_count == expected_count:
        return None
    return reason


def is_texture_compare_non_projected_offset_operation(operation):
    """Return whether a non-projected texture compare intrinsic uses an offset."""
    return operation in TEXTURE_COMPARE_NON_PROJECTED_OFFSET_OPERATION_NAMES


def is_texel_fetch_offset_operation(operation):
    """Return whether a texel-fetch intrinsic uses an offset."""
    return operation in TEXTURE_TEXEL_FETCH_OFFSET_INTRINSIC_NAMES


def is_texel_fetch_operation(operation):
    """Return whether a texture intrinsic fetches an explicit texel."""
    return operation in TEXTURE_TEXEL_FETCH_INTRINSIC_NAMES


def is_texel_fetch_basic_operation(operation):
    """Return whether a texel-fetch intrinsic has no offset argument."""
    return operation in TEXTURE_TEXEL_FETCH_BASIC_INTRINSIC_NAMES


def is_texture_offset_operation(operation):
    """Return whether a texture intrinsic has offset argument semantics."""
    return operation in TEXTURE_OFFSET_INTRINSIC_NAMES


def texture_resource_offset_dimension_key(operation, collapse_compare_offsets=False):
    """Return the descriptor key used to validate an operation's offset argument."""
    if is_texture_gather_offset_operation(operation):
        return "gather_offset_dimension"
    if is_texture_gather_compare_offset_operation(operation):
        return "gather_compare_offset_dimension"
    if collapse_compare_offsets:
        if is_texture_compare_any_offset_operation(operation):
            return "compare_offset_dimension"
    else:
        if is_texture_compare_offset_operation(operation):
            return "compare_offset_dimension"
        if is_texture_compare_lod_offset_operation(operation):
            return "compare_lod_offset_dimension"
        if is_texture_compare_grad_offset_operation(operation):
            return "compare_grad_offset_dimension"
    if is_texel_fetch_offset_operation(operation):
        return "texel_fetch_offset_dimension"
    if is_texture_sampling_offset_operation(operation):
        return "sample_offset_dimension"
    return "offset_dimension"


def is_texture_resource_operation(operation):
    """Return whether an intrinsic operates on texture resources."""
    return operation in TEXTURE_RESOURCE_INTRINSIC_NAMES


def texture_image_resource_operation_names(image_resource_intrinsic_names):
    """Return texture and image resource operation names for resource validation."""
    return (
        TEXTURE_RESOURCE_INTRINSIC_NAMES
        | frozenset(image_resource_intrinsic_names)
        | {"imageSamples"}
    )


def is_image_resource_operation(operation, image_resource_intrinsic_names):
    """Return whether an intrinsic operates on storage image resources."""
    return operation in image_resource_intrinsic_names


def is_texture_implicit_sampler_operation(operation):
    """Return whether a texture intrinsic may require an implicit sampler."""
    return operation in TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES


def unsupported_texel_fetch_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported texel fetch fallback expression."""
    return (
        f"/* unsupported {backend_name} texel fetch: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def texture_vector_zero_value(backend_name):
    """Return the backend vector fallback value for texture diagnostics."""
    if backend_name == "GLSL":
        return "vec4(0.0)"
    if backend_name == "DirectX":
        return "float4(0.0, 0.0, 0.0, 0.0)"
    return "float4(0.0)"


def texture_scalar_zero_value(backend_name):
    """Return the backend scalar fallback value for texture diagnostics."""
    return "0.0"


def texture_query_lod_zero_value(backend_name):
    """Return the backend vector fallback value for textureQueryLod diagnostics."""
    if backend_name == "GLSL":
        return "vec2(0.0)"
    if backend_name == "DirectX":
        return "float2(0.0, 0.0)"
    return "float2(0.0)"


def texture_query_lod_coordinate_swizzle(backend_name, texture_type):
    """Return the non-layer coordinate swizzle for textureQueryLod resources."""
    texture_type = str(texture_type or "")
    if backend_name == "GLSL":
        if texture_type in {
            "sampler1DArray",
            "isampler1DArray",
            "usampler1DArray",
        }:
            return "x"
        if texture_type in {
            "sampler2DArray",
            "sampler2DArrayShadow",
            "isampler2DArray",
            "usampler2DArray",
        }:
            return "xy"
        if texture_type in {
            "samplerCubeArray",
            "samplerCubeArrayShadow",
            "isamplerCubeArray",
            "usamplerCubeArray",
        }:
            return "xyz"
    elif backend_name == "DirectX":
        if texture_type == "Texture1DArray":
            return "x"
        if texture_type == "Texture2DArray":
            return "xy"
        if texture_type == "TextureCubeArray":
            return "xyz"
    elif backend_name == "Metal":
        if texture_type.startswith("texture1d_array<"):
            return "x"
        if texture_type.startswith(("texture2d_array<", "depth2d_array<")):
            return "xy"
        if texture_type.startswith(("texturecube_array<", "depthcube_array<")):
            return "xyz"
    return None


def texel_fetch_zero_value(backend_name):
    """Return the backend fallback value for unsupported texel fetches."""
    return texture_vector_zero_value(backend_name)


def unsupported_cube_texel_fetch_expression(backend_name, operation, resource_type):
    """Return an unsupported cube texel fetch fallback expression."""
    return unsupported_texel_fetch_expression(
        backend_name, operation, resource_type, texel_fetch_zero_value(backend_name)
    )


def unsupported_texel_fetch_offset_expression(backend_name, reason, zero_value):
    """Return an unsupported texel-fetch offset fallback expression."""
    return (
        f"/* unsupported {backend_name} texel fetch offset: "
        f"{reason} */ {zero_value}"
    )


def texel_fetch_offset_multisample_reason(backend_name, texture_type=None):
    """Return the diagnostic reason for multisample texelFetchOffset."""
    if backend_name == "GLSL":
        return f"multisample texture {texture_type} does not support offsets"
    return "multisample textures do not support offsets"


def unsupported_multisample_texel_fetch_offset_expression(
    backend_name, texture_type=None
):
    """Return an unsupported multisample texelFetchOffset fallback expression."""
    return unsupported_texel_fetch_offset_expression(
        backend_name,
        texel_fetch_offset_multisample_reason(backend_name, texture_type),
        texel_fetch_zero_value(backend_name),
    )


def unsupported_texture_offset_expression(backend_name, operation, reason, zero_value):
    """Return an unsupported texture offset fallback expression."""
    return (
        f"/* unsupported {backend_name} texture offset: "
        f"{operation} {reason} */ {zero_value}"
    )


def unsupported_texture_offset_call_expression(backend_name, operation, reason):
    """Return an unsupported texture offset fallback with backend zero value."""
    return unsupported_texture_offset_expression(
        backend_name, operation, reason, texture_vector_zero_value(backend_name)
    )


def unsupported_projected_texture_expression(
    backend_name, operation, reason, zero_value
):
    """Return an unsupported projected texture fallback expression."""
    return (
        f"/* unsupported {backend_name} projected texture: "
        f"{operation} {reason} */ {zero_value}"
    )


def unsupported_projected_texture_call_expression(backend_name, operation, reason):
    """Return an unsupported projected texture fallback with backend zero value."""
    return unsupported_projected_texture_expression(
        backend_name, operation, reason, texture_vector_zero_value(backend_name)
    )


def unsupported_texture_compare_expression(backend_name, operation, reason, zero_value):
    """Return an unsupported texture compare fallback expression."""
    return (
        f"/* unsupported {backend_name} texture compare: "
        f"{operation} {reason} */ {zero_value}"
    )


def unsupported_texture_compare_scalar_expression(backend_name, operation, reason):
    """Return an unsupported texture compare fallback with scalar zero."""
    return unsupported_texture_compare_expression(
        backend_name, operation, reason, texture_scalar_zero_value(backend_name)
    )


def unsupported_texture_gather_expression(backend_name, operation, reason, zero_value):
    """Return an unsupported texture gather fallback expression."""
    return (
        f"/* unsupported {backend_name} texture gather: "
        f"{operation} {reason} */ {zero_value}"
    )


def unsupported_texture_gather_call_expression(backend_name, operation, reason):
    """Return an unsupported texture gather fallback with backend zero value."""
    return unsupported_texture_gather_expression(
        backend_name, operation, reason, texture_vector_zero_value(backend_name)
    )


def unsupported_texture_gather_compare_expression(
    backend_name, operation, reason, zero_value
):
    """Return an unsupported texture gather-compare fallback expression."""
    return (
        f"/* unsupported {backend_name} texture gather compare: "
        f"{operation} {reason} */ {zero_value}"
    )


def unsupported_texture_gather_compare_call_expression(backend_name, operation, reason):
    """Return an unsupported texture gather-compare fallback with backend zero value."""
    return unsupported_texture_gather_compare_expression(
        backend_name, operation, reason, texture_vector_zero_value(backend_name)
    )


def component_count_mismatch(expected_count, actual_count, allow_scalar=True):
    """Return the actual component count when it does not satisfy a shape."""
    if expected_count is None or actual_count is None:
        return None
    if allow_scalar and actual_count == 1:
        return None
    if actual_count == expected_count:
        return None
    return actual_count


def component_kind_mismatch(expected_kind, actual_kind):
    """Return the actual component kind when it conflicts with an expected kind."""
    if expected_kind is None or actual_kind is None:
        return None
    if expected_kind == actual_kind:
        return None
    return actual_kind


def image_load_result_kind_mismatch(expected_kind, component_kind):
    """Return expected result kind when an image load result kind conflicts."""
    if expected_kind is None or component_kind not in NUMERIC_COMPONENT_KINDS:
        return None
    if expected_kind == component_kind:
        return None
    return expected_kind


def image_load_result_shape_mismatch(loaded_count, expected_count):
    """Return the expected result component count when shape validation fails."""
    return component_count_mismatch(loaded_count, expected_count)


def image_load_result_kind_error(
    backend_name, format_label, component_kind, expected_kind
):
    """Return a backend imageLoad result kind diagnostic."""
    return (
        f"{backend_name} image load operation 'imageLoad' requires "
        f"{component_kind} result context for {format_label} images: "
        f"expected {expected_kind}"
    )


def image_load_result_shape_error(
    backend_name, format_label, loaded_count, expected_count
):
    """Return a backend imageLoad result shape diagnostic."""
    expected_shape = component_shape_requirement(loaded_count, "result context")
    return (
        f"{backend_name} image load operation 'imageLoad' requires "
        f"{expected_shape} for {format_label} images: "
        f"expected {expected_count}-component"
    )


def image_store_value_shape_mismatch(expected_count, actual_count):
    """Return the actual store value component count when shape validation fails."""
    return component_count_mismatch(expected_count, actual_count)


def image_store_value_kind_mismatch(expected_kind, actual_kind):
    """Return the actual store value kind when kind validation fails."""
    return component_kind_mismatch(expected_kind, actual_kind)


def image_store_value_shape_error(
    backend_name, format_label, value_name, expected_count, actual_count
):
    """Return a backend imageStore value shape diagnostic."""
    expected_shape = component_shape_requirement(expected_count, "value")
    return (
        f"{backend_name} image store operation 'imageStore' requires "
        f"{expected_shape} for {format_label} images: "
        f"{value_name} has {actual_count} components"
    )


def image_store_value_kind_error(
    backend_name, format_label, value_name, expected_kind, actual_kind
):
    """Return a backend imageStore value kind diagnostic."""
    return (
        f"{backend_name} image store operation 'imageStore' requires "
        f"{expected_kind} value for {format_label} images: "
        f"{value_name} has type {actual_kind}"
    )


def should_validate_image_load_result_shape(expected_kind, component_kind):
    """Return true when image load result shape validation has enough type signal."""
    return expected_kind is not None and component_kind in NUMERIC_COMPONENT_KINDS


def image_format_or_default_channel_count(image_format, default_channel_count):
    """Return explicit image format channel count or a backend default count."""
    if image_format:
        return image_format_channel_count(image_format)
    return default_channel_count


def default_storage_image_channel_count(component_kind):
    """Return the default channel count for scalar storage image component kinds."""
    if component_kind in NUMERIC_COMPONENT_KINDS:
        return 4
    return None


def numeric_scalar_type_kind(vtype, type_name_string, map_type):
    """Return the numeric scalar kind for a backend-mapped type."""
    type_name = type_name_string(vtype)
    if not type_name:
        return None
    mapped_type = map_type(type_name)
    if mapped_type in NUMERIC_COMPONENT_KINDS:
        return mapped_type
    return None


def numeric_scalar_expression_kind(
    expr, expression_result_type, type_name_string, map_type
):
    """Return a literal or inferred numeric scalar kind for an expression."""
    literal_kind = literal_numeric_component_kind(expr)
    if literal_kind is not None:
        return literal_kind
    return numeric_scalar_type_kind(
        expression_result_type(expr), type_name_string, map_type
    )


def numeric_component_kind_from_type(
    vtype, type_name_string, map_type, vector_component_type
):
    """Return numeric component kind for a backend-mapped scalar or vector type."""
    type_name = type_name_string(vtype)
    if not type_name:
        return None
    mapped_type = map_type(type_name)
    return numeric_type_component_kind(mapped_type, vector_component_type(type_name))


def numeric_expression_component_kind(
    expr, expression_result_type, type_name_string, map_type, vector_component_type
):
    """Return a literal or inferred numeric component kind for an expression."""
    literal_kind = literal_numeric_component_kind(expr)
    if literal_kind is not None:
        return literal_kind
    return numeric_component_kind_from_type(
        expression_result_type(expr),
        type_name_string,
        map_type,
        vector_component_type,
    )


def numeric_component_count_from_type(
    vtype,
    type_name_string,
    map_type,
    vector_component_type,
    scalar_types=None,
    excluded_type_markers=(),
):
    """Return numeric component count for a backend-mapped scalar or vector type."""
    type_name = type_name_string(vtype)
    if not type_name:
        return None
    mapped_type = map_type(type_name)
    return numeric_type_component_count(
        mapped_type,
        vector_component_type(type_name),
        scalar_types=scalar_types,
        excluded_type_markers=excluded_type_markers,
    )


def numeric_expression_component_count(
    expr,
    expression_result_type,
    type_name_string,
    map_type,
    vector_component_type,
    scalar_types=None,
    excluded_type_markers=(),
):
    """Return a literal or inferred numeric component count for an expression."""
    literal_count = literal_numeric_component_count(expr)
    if literal_count is not None:
        return literal_count
    expr_type = expression_result_type(expr)
    if expr_type is None:
        return None
    return numeric_component_count_from_type(
        expr_type,
        type_name_string,
        map_type,
        vector_component_type,
        scalar_types=scalar_types,
        excluded_type_markers=excluded_type_markers,
    )


def normalized_image_access(value):
    """Normalize source-level image access spelling to read/write/read_write."""
    if value is None:
        return None
    value = str(value).strip().lower().replace("-", "_")
    if value.startswith("access::"):
        value = value.split("::", 1)[1]
    access_names = {
        "read": "read",
        "readonly": "read",
        "read_only": "read",
        "write": "write",
        "writeonly": "write",
        "write_only": "write",
        "readwrite": "read_write",
        "read_write": "read_write",
        "readwriteonly": "read_write",
    }
    return access_names.get(value)


def is_resource_access_attribute(attr):
    """Return true for attributes that describe storage image access mode."""
    attr_name = getattr(attr, "name", None)
    if not attr_name:
        return False
    attr_name = str(attr_name).lower()
    return attr_name == "access" or normalized_image_access(attr_name) is not None


def explicit_image_access(node, attribute_value_to_string):
    """Extract explicit image access metadata from qualifiers or attributes."""
    if node is None:
        return None

    for qualifier in getattr(node, "qualifiers", []) or []:
        access = normalized_image_access(qualifier)
        if access is not None:
            return access

    for attr in getattr(node, "attributes", []) or []:
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            continue
        attr_name = str(attr_name).lower()
        if attr_name == "access":
            arguments = getattr(attr, "arguments", []) or []
            if arguments:
                access = normalized_image_access(
                    attribute_value_to_string(arguments[0])
                )
                if access is not None:
                    return access
            continue
        access = normalized_image_access(attr_name)
        if access is not None:
            return access
    return None


def record_explicit_image_metadata(
    resource_name,
    node,
    attribute_value_to_string,
    image_formats=None,
    image_accesses=None,
):
    """Record explicit image format/access metadata for a named resource."""
    if not resource_name:
        return None, None

    image_format = explicit_image_format(node, attribute_value_to_string)
    if image_format is not None and image_formats is not None:
        image_formats[resource_name] = image_format

    image_access = explicit_image_access(node, attribute_value_to_string)
    if image_access is not None and image_accesses is not None:
        image_accesses[resource_name] = image_access

    return image_format, image_access


def image_resource_metadata(
    texture_arg, expression_name, parameter_metadata, variable_metadata
):
    """Return image metadata by preferring current parameters over globals."""
    texture_name = expression_name(texture_arg)
    if not texture_name:
        return None
    return parameter_metadata.get(texture_name, variable_metadata.get(texture_name))


def supported_image_formats():
    """Return the shared set of supported storage image format names."""
    return set(SUPPORTED_IMAGE_FORMATS)


def image_format_channel_count(image_format):
    """Return the number of components in a supported image format."""
    if image_format is None:
        return None
    return IMAGE_FORMAT_CHANNEL_COUNTS.get(str(image_format).lower())


def image_format_component_kind(image_format):
    """Return float/int/uint for a supported image format."""
    if image_format is None:
        return None
    return IMAGE_FORMAT_COMPONENT_KINDS.get(str(image_format).lower())


def numeric_type_component_kind(type_name, vector_component_kind=None):
    """Return float/int/uint for scalar or vector type metadata."""
    if type_name in NUMERIC_COMPONENT_KINDS:
        return type_name
    if vector_component_kind in NUMERIC_COMPONENT_KINDS:
        return vector_component_kind
    return None


def literal_numeric_component_kind(expr):
    """Return float/int/uint for typed numeric literal expressions."""
    literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
    if literal_type in NUMERIC_COMPONENT_KINDS:
        return literal_type
    return None


def numeric_type_component_count(
    type_name,
    vector_component_kind=None,
    scalar_types=None,
    excluded_type_markers=(),
):
    """Return component count for scalar/vector type metadata."""
    if type_name is None:
        return None
    type_name = str(type_name)
    if not type_name:
        return None
    scalar_types = set(scalar_types or NUMERIC_COMPONENT_KINDS)
    if type_name in scalar_types:
        return 1
    if any(marker in type_name for marker in excluded_type_markers):
        return None
    if vector_component_kind is None:
        return None
    suffix = type_name[-1]
    return int(suffix) if suffix in {"2", "3", "4"} else None


def literal_numeric_component_count(expr):
    """Return component count for typed numeric literal expressions."""
    return 1 if literal_numeric_component_kind(expr) is not None else None


def component_shape_requirement(channel_count, noun):
    """Return a diagnostic phrase for scalar or exact-width vector contexts."""
    if channel_count == 1:
        return f"scalar {noun}"
    return f"scalar or {channel_count}-component {noun}"


def image_multisample_sample_argument_index(
    func_name, argument_count, is_multisample_image, backend_name
):
    """Validate image arity around multisample sample indexes.

    Returns the sample-index argument position when the call has one.
    """
    if (
        func_name
        not in {
            "imageLoad",
            "imageStore",
            "imageAtomicCompSwap",
        }
        and func_name not in IMAGE_ATOMIC_VALUE_INTRINSIC_NAMES
    ):
        return None

    if func_name == "imageLoad":
        if is_multisample_image:
            if argument_count != 3:
                raise ValueError(
                    f"{backend_name} multisample image operation 'imageLoad' "
                    "requires image, coordinate, and sample index arguments, "
                    f"got {argument_count}"
                )
            return 2
        if argument_count > 2:
            raise ValueError(
                f"{backend_name} texture operation 'imageLoad' accepts at most "
                f"2 argument(s), got {argument_count}"
            )
        return None

    if func_name == "imageStore":
        if is_multisample_image:
            if argument_count != 4:
                raise ValueError(
                    f"{backend_name} multisample image operation 'imageStore' "
                    "requires image, coordinate, sample index, and value "
                    f"arguments, got {argument_count}"
                )
            return 2
        if argument_count > 3:
            raise ValueError(
                f"{backend_name} texture operation 'imageStore' accepts at most "
                f"3 argument(s), got {argument_count}"
            )
        return None

    if func_name in IMAGE_ATOMIC_VALUE_INTRINSIC_NAMES:
        if is_multisample_image:
            if argument_count != 4:
                raise ValueError(
                    f"{backend_name} multisample image atomic operation "
                    f"'{func_name}' requires image, coordinate, sample index, "
                    f"and value arguments, got {argument_count}"
                )
            return 2
        if argument_count > 3:
            raise ValueError(
                f"{backend_name} texture operation '{func_name}' accepts at "
                f"most 3 argument(s), got {argument_count}"
            )
        return None

    if is_multisample_image:
        if argument_count != 5:
            raise ValueError(
                f"{backend_name} multisample image atomic operation "
                "'imageAtomicCompSwap' requires image, coordinate, sample "
                f"index, compare, and value arguments, got {argument_count}"
            )
        return 2
    if argument_count > 4:
        raise ValueError(
            f"{backend_name} texture operation 'imageAtomicCompSwap' accepts at "
            f"most 4 argument(s), got {argument_count}"
        )
    return None


def image_multisample_sample_type_mismatch(sample_type, is_scalar_integer_type):
    """Return the sample type when a multisample image sample index is invalid."""
    if sample_type is None or is_scalar_integer_type(sample_type):
        return None
    return sample_type


def image_multisample_sample_type_error(
    backend_name, func_name, sample_name, sample_type_name
):
    """Return a backend multisample image sample-index diagnostic."""
    return (
        f"{backend_name} multisample image operation '{func_name}' requires a "
        f"scalar integer sample index argument: {sample_name} has type "
        f"{sample_type_name}"
    )


def texture_multisample_sample_type_error(
    backend_name, func_name, sample_name, sample_type_name
):
    """Return a backend multisample texel-fetch sample-index diagnostic."""
    return operation_argument_type_error(
        backend_name,
        "multisample texel fetch",
        func_name,
        "a scalar integer",
        "sample index",
        sample_name,
        sample_type_name,
    )


def image_format_component_type(
    image_format,
    float_type="float",
    int_type="int",
    uint_type="uint",
):
    """Return the backend scalar component type for a supported image format."""
    return {
        "float": float_type,
        "int": int_type,
        "uint": uint_type,
    }.get(image_format_component_kind(image_format))


def image_format_vector_type(
    image_format,
    float_type="float",
    int_type="int",
    uint_type="uint",
):
    """Return the backend vector component type for multi-channel image formats."""
    component_type = image_format_component_type(
        image_format,
        float_type=float_type,
        int_type=int_type,
        uint_type=uint_type,
    )
    channel_count = image_format_channel_count(image_format)
    if component_type is None or channel_count is None:
        return None
    if channel_count == 1:
        return component_type
    return f"{component_type}{channel_count}"


def image_format_result_type(image_format, scalar_types=None, vector_prefixes=None):
    """Return the backend result type for an explicit-format image load."""
    component_kind = image_format_component_kind(image_format)
    channel_count = image_format_channel_count(image_format)
    if component_kind is None or channel_count is None:
        return None

    scalar_types = scalar_types or {
        "float": "float",
        "int": "int",
        "uint": "uint",
    }
    if channel_count == 1:
        return scalar_types.get(component_kind)

    vector_prefixes = vector_prefixes or scalar_types
    vector_prefix = vector_prefixes.get(component_kind)
    if vector_prefix is None:
        return None
    return f"{vector_prefix}{channel_count}"


def is_scalar_image_format(image_format):
    """Return true for single-component supported image formats."""
    return image_format_channel_count(image_format) == 1


def is_two_component_image_format(image_format):
    """Return true for two-component supported image formats."""
    return image_format_channel_count(image_format) == 2


def storage_image_load_component_suffix(
    image_format,
    expected_scalar,
    scalar_integer_resource=False,
    float_resource=False,
):
    """Return component suffix needed after image loads for packed values."""
    if expected_scalar and (image_format_channel_count(image_format) or 0) > 1:
        return ".x"
    if scalar_integer_resource:
        return ".x"
    if float_resource and expected_scalar:
        return ".x"
    if is_two_component_image_format(image_format):
        return ".x" if expected_scalar else ".xy"
    return ""


def storage_image_format_store_constructor(image_format, constructors_by_kind):
    """Return constructor for scalar image format stores."""
    if not is_scalar_image_format(image_format):
        return None
    return constructors_by_kind.get(image_format_component_kind(image_format))


def storage_image_store_constructors(
    float_constructor, int_constructor, uint_constructor
):
    """Return storage image value constructors keyed by component kind."""
    return {
        "float": float_constructor,
        "int": int_constructor,
        "uint": uint_constructor,
    }


def storage_image_zero_values(float_zero="0.0", int_zero="0", uint_zero="0u"):
    """Return storage image padding values keyed by component kind."""
    return {
        "float": float_zero,
        "int": int_zero,
        "uint": uint_zero,
    }


def storage_image_store_vector_constructor(
    component_type, channel_count, component_kind, zero_values_by_kind=None
):
    """Return vector constructor metadata for storage image component types."""
    if component_kind not in NUMERIC_COMPONENT_KINDS:
        return None
    if numeric_type_component_count(component_type, component_kind) != channel_count:
        return None
    if zero_values_by_kind is None:
        return component_type
    zero_value = zero_values_by_kind.get(component_kind)
    if zero_value is None:
        return None
    return component_type, zero_value


def storage_image_two_component_store_expression(
    image_format,
    value,
    value_is_scalar,
    constructors_by_kind,
    zero_values_by_kind,
):
    """Return padded constructor expression for two-component image stores."""
    if not is_two_component_image_format(image_format):
        return None
    kind = image_format_component_kind(image_format)
    constructor = constructors_by_kind.get(kind)
    zero_value = zero_values_by_kind.get(kind)
    if constructor is None or zero_value is None:
        return None
    if value_is_scalar:
        return f"{constructor}({value}, {zero_value}, {zero_value}, {zero_value})"
    return f"{constructor}({value}, {zero_value}, {zero_value})"


def storage_image_store_value_expression(
    image_format,
    value,
    value_is_scalar,
    scalar_integer_resource=False,
    float_resource=False,
    integer_constructor=None,
    float_constructor=None,
    constructors_by_kind=None,
    zero_values_by_kind=None,
):
    """Return a storage image store value expression with required packing."""
    constructors_by_kind = constructors_by_kind or {}
    zero_values_by_kind = zero_values_by_kind or {}
    two_component_value = storage_image_two_component_store_expression(
        image_format,
        value,
        value_is_scalar,
        constructors_by_kind,
        zero_values_by_kind,
    )
    if two_component_value is not None:
        return two_component_value

    constructor = None
    if value_is_scalar and (image_format_channel_count(image_format) or 0) > 1:
        constructor = constructors_by_kind.get(
            image_format_component_kind(image_format)
        )
    if scalar_integer_resource:
        constructor = integer_constructor or storage_image_format_store_constructor(
            image_format, constructors_by_kind
        )
    elif constructor is None and float_resource and value_is_scalar:
        constructor = float_constructor
    if constructor:
        return f"{constructor}({value})"
    return value


def is_glsl_storage_image_type(type_name):
    """Return true for GLSL image, iimage, and uimage resource spellings."""
    return str(type_name).startswith(("image", "iimage", "uimage"))


def is_glsl_integer_image_type(type_name):
    """Return true for GLSL signed or unsigned integer image resource types."""
    return str(type_name) in GLSL_INTEGER_IMAGE_TYPES


def is_glsl_float_image_resource(type_name):
    """Return true for GLSL floating-point image resources with vector load values."""
    return str(type_name) in GLSL_FLOAT_IMAGE_RESOURCE_TYPES


def metal_storage_image_access_agnostic_type(type_name):
    """Normalize Metal storage texture access so type predicates ignore access mode."""
    if type_name is None:
        return ""
    return (
        str(type_name)
        .replace("access::read>", "access::read_write>")
        .replace("access::write>", "access::read_write>")
    )


def is_metal_storage_image_resource(type_name):
    """Return true for Metal storage textures with an explicit access mode."""
    if type_name is None:
        return False
    type_name = str(type_name)
    return type_name.startswith(METAL_STORAGE_IMAGE_PREFIXES) and any(
        access in type_name for access in METAL_STORAGE_IMAGE_ACCESS_TOKENS
    )


def metal_storage_image_component_type(type_name):
    """Return float/int/uint for a Metal storage texture type."""
    type_name = metal_storage_image_access_agnostic_type(type_name)
    if not is_metal_storage_image_resource(type_name) or "<" not in type_name:
        return None
    component_type = type_name.split("<", 1)[1].split(",", 1)[0].strip()
    if component_type in {"float", "int", "uint"}:
        return component_type
    return None


def is_metal_integer_image_type(type_name):
    """Return true for Metal signed or unsigned integer storage textures."""
    return metal_storage_image_component_type(type_name) in {"int", "uint"}


def is_metal_float_image_resource(type_name):
    """Return true for Metal floating-point storage textures."""
    return metal_storage_image_component_type(type_name) == "float"


def explicit_image_format(
    node,
    attribute_value_to_string,
    supported_formats=SUPPORTED_IMAGE_FORMATS,
):
    """Extract explicit storage image format metadata from attributes."""
    if not hasattr(node, "attributes"):
        return None
    for attr in node.attributes:
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            continue
        attr_name = str(attr_name).lower()
        if attr_name in supported_formats:
            return attr_name
        if attr_name == "format":
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            format_name = attribute_value_to_string(arguments[0])
            if format_name is None:
                continue
            format_name = str(format_name).lower()
            if format_name in supported_formats:
                return format_name
    return None


def is_image_format_attribute(attr, supported_formats=SUPPORTED_IMAGE_FORMATS):
    """Return true for attributes that describe storage image format metadata."""
    attr_name = getattr(attr, "name", None)
    if not attr_name:
        return False
    attr_name = str(attr_name).lower()
    return attr_name == "format" or attr_name in supported_formats
