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
    image_format: 4
    if image_format.startswith("rgba")
    else 2
    if image_format.startswith("rg")
    else 1
    for image_format in SUPPORTED_IMAGE_FORMATS
}

IMAGE_FORMAT_COMPONENT_KINDS = {
    image_format: "uint"
    if image_format.endswith("ui")
    else "int"
    if image_format.endswith("i")
    else "float"
    for image_format in SUPPORTED_IMAGE_FORMATS
}

GLSL_INTEGER_IMAGE_TYPES = frozenset(
    {
        "iimage1D",
        "iimage1DArray",
        "iimage2D",
        "iimage3D",
        "iimage2DArray",
        "uimage1D",
        "uimage1DArray",
        "uimage2D",
        "uimage3D",
        "uimage2DArray",
    }
)

GLSL_FLOAT_IMAGE_RESOURCE_TYPES = frozenset(
    {
        "image1D",
        "image1DArray",
        "image2D",
        "image3D",
        "image2DArray",
    }
)

METAL_STORAGE_IMAGE_PREFIXES = (
    "texture1d<",
    "texture1d_array<",
    "texture2d<",
    "texture3d<",
    "texture2d_array<",
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
    if scalar_integer_resource:
        constructor = integer_constructor or storage_image_format_store_constructor(
            image_format, constructors_by_kind
        )
    elif float_resource and value_is_scalar:
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
