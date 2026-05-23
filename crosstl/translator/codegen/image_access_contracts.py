"""Shared image access analysis and metadata helpers for code generators."""

from ..ast import FunctionCallNode

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


def unsupported_texture_query_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported texture query fallback expression."""
    return (
        f"/* unsupported {backend_name} texture query: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_query_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture query fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture query: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_texture_samples_query_expression(
    backend_name, multisample_resource_name
):
    """Return an unsupported texture samples fallback expression."""
    return (
        f"/* unsupported {backend_name} texture samples query: "
        f"requires multisample {multisample_resource_name} */ 0"
    )


def unsupported_multisample_texture_call_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture-call fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture call: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_compare_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture comparison fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture comparison: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_multisample_texture_gather_compare_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported multisample texture gather-compare fallback expression."""
    return (
        f"/* unsupported {backend_name} multisample texture gather comparison: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_storage_image_texture_comparison_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported storage-image texture comparison fallback."""
    return (
        f"/* unsupported {backend_name} storage image texture comparison: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


def unsupported_storage_image_texture_operation_expression(
    backend_name, operation, resource_type, zero_value
):
    """Return an unsupported storage-image texture operation fallback."""
    return (
        f"/* unsupported {backend_name} storage image texture operation: "
        f"{operation} on {resource_type} */ {zero_value}"
    )


STORAGE_IMAGE_TEXTURE_COMPARISON_OPERATIONS = frozenset(
    {
        "textureCompare",
        "textureCompareOffset",
        "textureCompareLod",
        "textureCompareLodOffset",
        "textureCompareGrad",
        "textureCompareGradOffset",
        "textureCompareProj",
        "textureCompareProjOffset",
        "textureCompareProjLod",
        "textureCompareProjLodOffset",
        "textureCompareProjGrad",
        "textureCompareProjGradOffset",
    }
)


STORAGE_IMAGE_TEXTURE_OPERATIONS = frozenset(
    {
        "texture",
        "textureLod",
        "textureGrad",
        "textureOffset",
        "textureLodOffset",
        "textureGradOffset",
        "textureProj",
        "textureProjOffset",
        "textureProjLod",
        "textureProjLodOffset",
        "textureProjGrad",
        "textureProjGradOffset",
        "textureGather",
        "textureGatherOffset",
        "textureGatherOffsets",
        "textureGatherCompare",
        "textureGatherCompareOffset",
        "texelFetch",
        "texelFetchOffset",
    }
)


def is_storage_image_texture_comparison_operation(operation):
    """Return whether a texture intrinsic is an unsupported storage-image comparison."""
    return operation in STORAGE_IMAGE_TEXTURE_COMPARISON_OPERATIONS


def is_storage_image_texture_operation(operation):
    """Return whether a texture intrinsic is unsupported for storage images."""
    return operation in STORAGE_IMAGE_TEXTURE_OPERATIONS


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
