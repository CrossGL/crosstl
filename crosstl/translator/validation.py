"""Shared AST validation helpers for the CrossGL translator."""

TEXTURE_INTRINSIC_MIN_ARGUMENTS = {
    "texture": 2,
    "textureLod": 3,
    "textureGrad": 4,
    "textureOffset": 3,
    "textureLodOffset": 4,
    "textureGradOffset": 5,
    "textureProj": 2,
    "textureProjOffset": 3,
    "textureProjLod": 3,
    "textureProjLodOffset": 4,
    "textureProjGrad": 4,
    "textureProjGradOffset": 5,
    "textureCompare": 3,
    "textureCompareOffset": 4,
    "textureCompareLod": 4,
    "textureCompareLodOffset": 5,
    "textureCompareGrad": 5,
    "textureCompareGradOffset": 6,
    "textureCompareProj": 3,
    "textureCompareProjOffset": 4,
    "textureCompareProjLod": 4,
    "textureCompareProjLodOffset": 5,
    "textureCompareProjGrad": 5,
    "textureCompareProjGradOffset": 6,
    "textureGather": 2,
    "textureGatherOffset": 3,
    "textureGatherOffsets": 3,
    "textureGatherCompare": 3,
    "textureGatherCompareOffset": 4,
    "textureQueryLod": 2,
    "textureQueryLevels": 1,
    "textureSize": 1,
    "textureSamples": 1,
    "texelFetch": 3,
    "texelFetchOffset": 4,
    "imageLoad": 2,
    "imageStore": 3,
    "imageSize": 1,
    "imageSamples": 1,
    "imageAtomicAdd": 3,
    "imageAtomicMin": 3,
    "imageAtomicMax": 3,
    "imageAtomicAnd": 3,
    "imageAtomicOr": 3,
    "imageAtomicXor": 3,
    "imageAtomicExchange": 3,
    "imageAtomicCompSwap": 4,
}

TEXTURE_INTRINSICS_WITH_EXPLICIT_SAMPLERS = {
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
    "textureGather",
    "textureGatherOffset",
    "textureGatherOffsets",
    "textureGatherCompare",
    "textureGatherCompareOffset",
    "textureQueryLod",
}

TEXTURE_INTRINSIC_MAX_ARGUMENTS = {
    "texture": 3,
    "textureLod": 3,
    "textureGrad": 4,
    "textureOffset": 4,
    "textureLodOffset": 4,
    "textureGradOffset": 5,
    "textureProj": 3,
    "textureProjOffset": 4,
    "textureProjLod": 3,
    "textureProjLodOffset": 4,
    "textureProjGrad": 4,
    "textureProjGradOffset": 5,
    "textureCompare": 3,
    "textureCompareOffset": 4,
    "textureCompareLod": 4,
    "textureCompareLodOffset": 5,
    "textureCompareGrad": 5,
    "textureCompareGradOffset": 6,
    "textureCompareProj": 3,
    "textureCompareProjOffset": 4,
    "textureCompareProjLod": 4,
    "textureCompareProjLodOffset": 5,
    "textureCompareProjGrad": 5,
    "textureCompareProjGradOffset": 6,
    "textureGather": 3,
    "textureGatherOffset": 4,
    "textureGatherOffsets": 7,
    "textureGatherCompare": 3,
    "textureGatherCompareOffset": 4,
    "textureQueryLod": 2,
    "textureQueryLevels": 1,
    "textureSize": 2,
    "textureSamples": 1,
    "texelFetch": 3,
    "texelFetchOffset": 4,
    "imageLoad": 3,
    "imageStore": 4,
    "imageSize": 1,
    "imageSamples": 1,
    "imageAtomicAdd": 4,
    "imageAtomicMin": 4,
    "imageAtomicMax": 4,
    "imageAtomicAnd": 4,
    "imageAtomicOr": 4,
    "imageAtomicXor": 4,
    "imageAtomicExchange": 4,
    "imageAtomicCompSwap": 5,
}

TEXTURE_INTRINSIC_ALLOWED_ARGUMENT_COUNTS = {
    "textureGatherOffsets": (3, 4, 6, 7),
}

METADATA_CONFLICT_GROUPS = (
    (frozenset({"flat", "noperspective"}), "@flat and @noperspective"),
    (frozenset({"centroid", "sample"}), "@centroid and @sample"),
)

INTERPOLATION_MODE_METADATA_NAMES = {
    "flat": "flat",
    "nointerpolation": "flat",
    "smooth": "smooth",
    "linear": "smooth",
    "linear_centroid": "smooth",
    "linear_sample": "smooth",
    "noperspective": "noperspective",
    "linear_noperspective": "noperspective",
    "linear_noperspective_centroid": "noperspective",
}

INTERPOLATION_SAMPLING_METADATA_NAMES = {
    "centroid": "centroid",
    "linear_centroid": "centroid",
    "linear_noperspective_centroid": "centroid",
    "sample": "sample",
    "linear_sample": "sample",
}

SINGLE_VALUE_METADATA_NAMES = frozenset(
    {
        "access",
        "binding",
        "buffer",
        "component",
        "format",
        "index",
        "location",
        "set",
        "texture",
        "uav",
    }
)

RESOURCE_ACCESS_METADATA_NAMES = {
    "read": "readonly",
    "readonly": "readonly",
    "write": "writeonly",
    "writeonly": "writeonly",
    "read_write": "readwrite",
    "readwrite": "readwrite",
    "access::read": "readonly",
    "access::write": "writeonly",
    "access::read_write": "readwrite",
}

ADDRESS_SPACE_METADATA_NAMES = {
    "device": "device",
    "global": "device",
    "constant": "constant",
    "thread": "thread",
    "local": "thread",
    "private": "thread",
    "function": "thread",
    "threadgroup": "workgroup",
    "workgroup": "workgroup",
    "shared": "workgroup",
    "groupshared": "workgroup",
    "storage": "storage",
}

DECLARATION_ROLE_METADATA_NAMES = {
    "payload": "payload",
    "ray_payload": "payload",
    "raypayload": "payload",
    "hit_attribute": "hit_attribute",
    "hitattribute": "hit_attribute",
    "callable_data": "callable_data",
    "callabledata": "callable_data",
    "vertices": "mesh_vertices",
    "indices": "mesh_indices",
    "primitives": "mesh_primitives",
}

FUNCTION_STAGE_ATTRIBUTE_NAMES = {
    "domain": "domain",
    "local_size": "threadgroup_size",
    "max_total_threads_per_threadgroup": "max_total_threads_per_threadgroup",
    "maxprimitivecount": "maxprimitivecount",
    "maxvertexcount": "maxvertexcount",
    "numthreads": "threadgroup_size",
    "outputcontrolpoints": "outputcontrolpoints",
    "outputtopology": "outputtopology",
    "partitioning": "partitioning",
    "patchconstantfunc": "patchconstantfunc",
    "shader": "shader",
    "workgroup_size": "threadgroup_size",
}

STAGE_LAYOUT_DIRECTION_REQUIREMENTS = {
    "local_size_x": "in",
    "local_size_y": "in",
    "local_size_z": "in",
    "invocations": "in",
    "max_vertices": "out",
    "maxvertexcount": "out",
    "max_primitives": "out",
    "maxprimitivecount": "out",
    "vertices": "out",
    "outputcontrolpoints": "out",
}

STAGE_LAYOUT_EXCLUSIVE_ENTRY_GROUPS = (
    frozenset(
        {
            "points",
            "lines",
            "line_strip",
            "triangles",
            "triangle_strip",
            "lines_adjacency",
            "triangles_adjacency",
            "lineadj",
            "triangleadj",
            "quads",
            "isolines",
        }
    ),
    frozenset(
        {
            "equal_spacing",
            "fractional_even_spacing",
            "fractional_odd_spacing",
        }
    ),
    frozenset({"cw", "ccw"}),
)

IMAGE_RESOURCE_INTRINSIC_NAMES = {
    "imageLoad",
    "imageStore",
    "imageSize",
    "imageAtomicAdd",
    "imageAtomicMin",
    "imageAtomicMax",
    "imageAtomicAnd",
    "imageAtomicOr",
    "imageAtomicXor",
    "imageAtomicExchange",
    "imageAtomicCompSwap",
}

INTEGER_COORDINATE_INTRINSIC_NAMES = {
    "texelFetch",
    "texelFetchOffset",
} | (IMAGE_RESOURCE_INTRINSIC_NAMES - {"imageSize"})

OFFSET_DIMENSION_INTRINSIC_NAMES = {
    "texelFetchOffset",
    "textureOffset",
    "textureLodOffset",
    "textureGradOffset",
    "textureProjOffset",
    "textureProjLodOffset",
    "textureProjGradOffset",
    "textureCompareOffset",
    "textureCompareLodOffset",
    "textureCompareGradOffset",
    "textureCompareProjOffset",
    "textureCompareProjLodOffset",
    "textureCompareProjGradOffset",
    "textureGatherOffset",
    "textureGatherOffsets",
    "textureGatherCompareOffset",
}

OFFSET_ARGUMENT_INDEX_OFFSETS = {
    "textureOffset": 1,
    "textureLodOffset": 2,
    "textureGradOffset": 3,
    "textureProjOffset": 1,
    "textureProjLodOffset": 2,
    "textureProjGradOffset": 3,
    "textureCompareOffset": 2,
    "textureCompareLodOffset": 3,
    "textureCompareGradOffset": 4,
    "textureCompareProjOffset": 2,
    "textureCompareProjLodOffset": 3,
    "textureCompareProjGradOffset": 4,
    "textureGatherOffset": 1,
    "textureGatherCompareOffset": 2,
}

GRADIENT_DIMENSION_INTRINSIC_NAMES = {
    "textureGrad",
    "textureGradOffset",
    "textureProjGrad",
    "textureProjGradOffset",
    "textureCompareGrad",
    "textureCompareGradOffset",
    "textureCompareProjGrad",
    "textureCompareProjGradOffset",
}

GRADIENT_ARGUMENT_INDEX_OFFSETS = {
    "textureGrad": (1, 2),
    "textureGradOffset": (1, 2),
    "textureProjGrad": (1, 2),
    "textureProjGradOffset": (1, 2),
    "textureCompareGrad": (2, 3),
    "textureCompareGradOffset": (2, 3),
    "textureCompareProjGrad": (2, 3),
    "textureCompareProjGradOffset": (2, 3),
}

COMPARE_INTRINSIC_NAMES = {
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
    "textureGatherCompare",
    "textureGatherCompareOffset",
}

LOD_ARGUMENT_INDEX_OFFSETS = {
    "textureLod": 1,
    "textureLodOffset": 1,
    "textureProjLod": 1,
    "textureProjLodOffset": 1,
    "textureCompareLod": 2,
    "textureCompareLodOffset": 2,
    "textureCompareProjLod": 2,
    "textureCompareProjLodOffset": 2,
}

BIAS_ARGUMENT_INDEX_OFFSETS = {
    "texture": 1,
    "textureOffset": 2,
    "textureProj": 1,
    "textureProjOffset": 2,
}

MIP_LEVEL_ARGUMENT_INDICES = {
    "textureSize": 1,
    "texelFetch": 2,
    "texelFetchOffset": 2,
}

GATHER_COMPONENT_INTRINSIC_NAMES = {
    "textureGather",
    "textureGatherOffset",
    "textureGatherOffsets",
}


def texture_intrinsic_min_argument_count(func_name, has_explicit_sampler=False):
    """Return the minimum CrossGL argument count for a texture intrinsic."""
    min_count = TEXTURE_INTRINSIC_MIN_ARGUMENTS.get(func_name)
    if min_count is None:
        return None
    if has_explicit_sampler and func_name in TEXTURE_INTRINSICS_WITH_EXPLICIT_SAMPLERS:
        return min_count + 1
    return min_count


def texture_intrinsic_max_argument_count(func_name, has_explicit_sampler=False):
    """Return the maximum CrossGL argument count for a bounded intrinsic."""
    max_count = TEXTURE_INTRINSIC_MAX_ARGUMENTS.get(func_name)
    if max_count is None:
        return None
    if has_explicit_sampler and func_name in TEXTURE_INTRINSICS_WITH_EXPLICIT_SAMPLERS:
        return max_count + 1
    return max_count


def texture_intrinsic_allowed_argument_counts(func_name, has_explicit_sampler=False):
    """Return exact accepted CrossGL argument counts for non-contiguous forms."""
    counts = TEXTURE_INTRINSIC_ALLOWED_ARGUMENT_COUNTS.get(func_name)
    if counts is None:
        return None
    if has_explicit_sampler and func_name in TEXTURE_INTRINSICS_WITH_EXPLICIT_SAMPLERS:
        return tuple(count + 1 for count in counts)
    return counts


def texture_offset_argument_indices(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return argument indices that should be integer texel offsets."""
    if func_name == "texelFetchOffset":
        return [3] if argument_count is None or argument_count > 3 else []

    if func_name not in OFFSET_DIMENSION_INTRINSIC_NAMES:
        return []

    coord_index = 2 if has_explicit_sampler else 1
    if func_name == "textureGatherOffsets":
        first_offset_index = coord_index + 1
        if argument_count is not None and argument_count <= first_offset_index:
            return []
        if argument_count is not None and argument_count >= first_offset_index + 4:
            return list(range(first_offset_index, first_offset_index + 4))
        return [first_offset_index]

    offset_from_coord = OFFSET_ARGUMENT_INDEX_OFFSETS.get(func_name)
    if offset_from_coord is None:
        return []
    index = coord_index + offset_from_coord
    if argument_count is not None and argument_count <= index:
        return []
    return [index]


def texture_gradient_argument_indices(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return argument indices that should be texture gradient derivatives."""
    if func_name not in GRADIENT_DIMENSION_INTRINSIC_NAMES:
        return []
    coord_index = 2 if has_explicit_sampler else 1
    offsets = GRADIENT_ARGUMENT_INDEX_OFFSETS.get(func_name)
    if offsets is None:
        return []
    indices = [coord_index + offset for offset in offsets]
    if argument_count is None:
        return indices
    return [index for index in indices if argument_count > index]


def texture_compare_argument_index(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return the compare/depth argument index for texture compare operations."""
    if func_name not in COMPARE_INTRINSIC_NAMES:
        return None
    coord_index = 2 if has_explicit_sampler else 1
    index = coord_index + 1
    if argument_count is not None and argument_count <= index:
        return None
    return index


def texture_lod_argument_index(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return the explicit LOD argument index for texture LOD operations."""
    offset_from_coord = LOD_ARGUMENT_INDEX_OFFSETS.get(func_name)
    if offset_from_coord is None:
        return None
    coord_index = 2 if has_explicit_sampler else 1
    index = coord_index + offset_from_coord
    if argument_count is not None and argument_count <= index:
        return None
    return index


def texture_bias_argument_index(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return the optional sample bias argument index for texture operations."""
    offset_from_coord = BIAS_ARGUMENT_INDEX_OFFSETS.get(func_name)
    if offset_from_coord is None:
        return None
    coord_index = 2 if has_explicit_sampler else 1
    index = coord_index + offset_from_coord
    if argument_count is not None and argument_count <= index:
        return None
    return index


def texture_query_lod_coordinate_argument_index(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return the coordinate argument index for textureQueryLod operations."""
    if func_name != "textureQueryLod":
        return None
    index = 2 if has_explicit_sampler else 1
    if argument_count is not None and argument_count <= index:
        return None
    return index


def texture_mip_level_argument_index(func_name, argument_count=None):
    """Return the integer mip/sample level argument index for texture operations."""
    index = MIP_LEVEL_ARGUMENT_INDICES.get(func_name)
    if index is None:
        return None
    if argument_count is not None and argument_count <= index:
        return None
    return index


def texture_sample_index_argument_index(func_name, argument_count=None):
    """Return the sample-index argument index for multisample texture fetches."""
    if func_name != "texelFetch":
        return None
    index = 2
    if argument_count is not None and argument_count <= index:
        return None
    return index


def texture_gather_component_argument_index(
    func_name, has_explicit_sampler=False, argument_count=None
):
    """Return the optional component selector index for texture gather operations."""
    if func_name not in GATHER_COMPONENT_INTRINSIC_NAMES:
        return None
    if argument_count is None:
        return None

    coord_index = 2 if has_explicit_sampler else 1
    extra_count = argument_count - coord_index - 1

    if func_name == "textureGather":
        return coord_index + 1 if extra_count >= 1 else None
    if func_name == "textureGatherOffset":
        return coord_index + 2 if extra_count >= 2 else None
    if extra_count == 2:
        return coord_index + 2
    if extra_count == 5:
        return coord_index + 5
    return None


def integer_coordinate_dimension(type_name):
    """Return the vector width of a known integer coordinate type."""
    if type_name is None:
        return None
    type_name = str(type_name)
    if "[" in type_name and "]" in type_name:
        type_name = type_name.split("[", 1)[0]
    dimensions = {
        "int": 1,
        "uint": 1,
        "ivec2": 2,
        "ivec3": 3,
        "ivec4": 4,
        "uvec2": 2,
        "uvec3": 3,
        "uvec4": 4,
        "int2": 2,
        "int3": 3,
        "int4": 4,
        "uint2": 2,
        "uint3": 3,
        "uint4": 4,
    }
    return dimensions.get(type_name)


def floating_coordinate_dimension(type_name):
    """Return the vector width of a known floating-point coordinate type."""
    if type_name is None:
        return None
    type_name = str(type_name)
    if "[" in type_name and "]" in type_name:
        type_name = type_name.split("[", 1)[0]
    dimensions = {
        "float": 1,
        "half": 1,
        "double": 1,
        "vec2": 2,
        "vec3": 3,
        "vec4": 4,
        "dvec2": 2,
        "dvec3": 3,
        "dvec4": 4,
        "float2": 2,
        "float3": 3,
        "float4": 4,
        "half2": 2,
        "half3": 3,
        "half4": 4,
        "double2": 2,
        "double3": 3,
        "double4": 4,
    }
    return dimensions.get(type_name)


def is_floating_scalar_type(type_name):
    """Return true for known scalar floating-point types."""
    return str(type_name) in {"float", "half", "double"}


def is_integer_scalar_type(type_name):
    """Return true for known scalar integer types."""
    return str(type_name) in {"int", "uint"}


def is_numeric_scalar_type(type_name):
    """Return true for known scalar numeric types."""
    return str(type_name) in {"float", "half", "double", "int", "uint"}


def expression_debug_name(expr):
    """Return a stable source-like expression name for diagnostics."""
    if expr is None:
        return ""
    if isinstance(expr, str):
        return expr
    op = getattr(expr, "op", None)
    if op is not None and hasattr(expr, "left") and hasattr(expr, "right"):
        left = expression_debug_name(expr.left)
        right = expression_debug_name(expr.right)
        return f"{left} {op} {right}"
    if op is not None and hasattr(expr, "operand"):
        operand = expression_debug_name(expr.operand)
        if getattr(expr, "is_postfix", False):
            return f"{operand}{op}"
        return f"{op}{operand}"
    name = getattr(expr, "name", None)
    if isinstance(name, str):
        return name
    if hasattr(expr, "member") and (
        hasattr(expr, "object") or hasattr(expr, "object_expr")
    ):
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        object_name = expression_debug_name(object_expr)
        member_name = str(getattr(expr, "member", ""))
        return f"{object_name}.{member_name}" if object_name else member_name
    if hasattr(expr, "array") or hasattr(expr, "array_expr"):
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
        array_name = expression_debug_name(array_expr)
        index_name = expression_debug_name(index_expr)
        if array_name and index_name:
            return f"{array_name}[{index_name}]"
        return array_name or str(expr)
    if hasattr(expr, "value"):
        return str(getattr(expr, "value"))
    return str(expr)


def collect_duplicate_cbuffer_member_names(cbuffers):
    """Return cbuffer member names that appear more than once."""
    seen_members = set()
    duplicate_members = set()
    for cbuffer in cbuffers or []:
        for member in getattr(cbuffer, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            if member_name in seen_members:
                duplicate_members.add(member_name)
            else:
                seen_members.add(member_name)
    return duplicate_members


def collect_duplicate_cbuffer_names(cbuffers):
    """Return cbuffer block names that appear more than once."""
    seen_names = set()
    duplicate_names = set()
    for cbuffer in cbuffers or []:
        name = getattr(cbuffer, "name", None)
        if not name:
            continue
        if name in seen_names:
            duplicate_names.add(name)
        else:
            seen_names.add(name)
    return duplicate_names


def collect_cbuffer_declaration_name_conflicts(shader):
    """Return cbuffer names that collide with other top-level declarations."""
    cbuffers = getattr(shader, "cbuffers", []) or []
    cbuffer_names = {
        getattr(cbuffer, "name", None)
        for cbuffer in cbuffers
        if getattr(cbuffer, "name", None)
    }
    declaration_names = _named_node_set(getattr(shader, "structs", []) or [])
    declaration_names.update(
        _named_node_set(getattr(shader, "global_variables", []) or [])
    )
    declaration_names.update(_named_node_set(getattr(shader, "constants", []) or []))
    return cbuffer_names & declaration_names


def collect_cbuffer_member_global_conflicts(shader):
    """Return cbuffer member names that collide with global declarations."""
    global_names = _named_node_set(getattr(shader, "global_variables", []) or [])
    global_names.update(_named_node_set(getattr(shader, "constants", []) or []))
    if not global_names:
        return set()

    member_names = set()
    for cbuffer in getattr(shader, "cbuffers", []) or []:
        for member in getattr(cbuffer, "members", []) or []:
            member_name = getattr(member, "name", None)
            if member_name:
                member_names.add(member_name)
    return member_names & global_names


def collect_non_resource_global_resource_shadows(
    shader, global_resource_names, is_resource_type
):
    """Return non-resource locals/parameters that shadow global resources."""
    if not global_resource_names:
        return set()

    conflicts = set()
    for func in _shader_functions(shader):
        params = getattr(func, "parameters", getattr(func, "params", [])) or []
        for param in params:
            name = getattr(param, "name", None)
            if not name or name not in global_resource_names:
                continue
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            if not is_resource_type(param_type):
                conflicts.add(name)

        for node in _walk_ast(getattr(func, "body", [])):
            if not _is_variable_node(node):
                continue
            name = getattr(node, "name", None)
            if not name or name not in global_resource_names:
                continue
            var_type = getattr(node, "var_type", getattr(node, "vtype", None))
            if not is_resource_type(var_type):
                conflicts.add(name)

    return conflicts


def validate_shader_metadata(shader):
    """Validate backend-neutral declaration metadata for contradictions."""
    for function, context in _shader_function_contexts(shader):
        validate_function_metadata(function, context)
    for node, context in _shader_metadata_nodes(shader):
        validate_node_metadata(node, context)
    for stage in getattr(shader, "stages", {}).values():
        validate_stage_layout_metadata(stage)
    return shader


def validate_function_metadata(function, context):
    """Validate backend-neutral function/stage metadata for contradictions."""
    values_by_name = {}
    display_names = {}

    for attr in getattr(function, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        canonical_name = FUNCTION_STAGE_ATTRIBUTE_NAMES.get(attr_name)
        if canonical_name is None:
            continue

        attr_value = _attribute_metadata_values(attr)
        previous_value = values_by_name.setdefault(canonical_name, attr_value)
        display_names.setdefault(canonical_name, attr_name)
        if previous_value != attr_value:
            raise ValueError(
                f"Conflicting {display_names[canonical_name]} metadata on {context}: "
                f"{_metadata_value_phrase(previous_value)} vs "
                f"{_metadata_value_phrase(attr_value)}"
            )


def validate_node_metadata(node, context):
    """Validate qualifier and attribute metadata on a declaration node."""
    names = _node_metadata_names(node)
    for group, description in METADATA_CONFLICT_GROUPS:
        if group <= names:
            raise ValueError(f"Conflicting metadata on {context}: {description}")

    interpolation_modes = _normalized_metadata_names(
        names, INTERPOLATION_MODE_METADATA_NAMES
    )
    if len(set(interpolation_modes.values())) > 1:
        raise ValueError(
            f"Conflicting interpolation mode metadata on {context}: "
            f"{_metadata_name_phrase(interpolation_modes)}"
        )

    interpolation_sampling = _normalized_metadata_names(
        names, INTERPOLATION_SAMPLING_METADATA_NAMES
    )
    if len(set(interpolation_sampling.values())) > 1:
        raise ValueError(
            f"Conflicting interpolation sampling metadata on {context}: "
            f"{_metadata_name_phrase(interpolation_sampling)}"
        )

    address_spaces = _normalized_metadata_names(names, ADDRESS_SPACE_METADATA_NAMES)
    if len(set(address_spaces.values())) > 1:
        raise ValueError(
            f"Conflicting address space metadata on {context}: "
            f"{_metadata_name_phrase(address_spaces)}"
        )

    declaration_roles = _normalized_metadata_names(
        names, DECLARATION_ROLE_METADATA_NAMES
    )
    if len(set(declaration_roles.values())) > 1:
        raise ValueError(
            f"Conflicting declaration role metadata on {context}: "
            f"{_metadata_name_phrase(declaration_roles)}"
        )

    access_names = _node_resource_access_names(node)
    if len(access_names) > 1:
        access_list = ", ".join(sorted(f"@{name}" for name in access_names))
        raise ValueError(
            f"Conflicting resource access metadata on {context}: {access_list}"
        )

    values_by_name = {}
    for attr in getattr(node, "attributes", []) or []:
        attr_name = str(getattr(attr, "name", "")).lower()
        if attr_name not in SINGLE_VALUE_METADATA_NAMES:
            continue
        attr_value = _attribute_metadata_value(attr)
        if attr_value is None:
            continue
        previous_value = values_by_name.setdefault(attr_name, attr_value)
        if previous_value != attr_value:
            raise ValueError(
                f"Conflicting {attr_name} metadata on {context}: "
                f"{previous_value} vs {attr_value}"
            )


def validate_stage_layout_metadata(stage):
    """Validate stage-level layout qualifiers for contradictions."""
    entries_by_direction = {}
    for layout in getattr(stage, "layout_qualifiers", []) or []:
        direction = _normalized_layout_direction(getattr(layout, "direction", None))
        context = _stage_layout_context(stage, direction)
        for entry in getattr(layout, "entries", []) or []:
            entry_name = _normalized_metadata_name(getattr(entry, "name", None))
            if not entry_name:
                continue

            required_direction = STAGE_LAYOUT_DIRECTION_REQUIREMENTS.get(entry_name)
            if required_direction and direction != required_direction:
                raise ValueError(
                    f"Stage layout '{entry_name}' on {context} must use "
                    f"'{required_direction}' direction"
                )

            entry_value = _stage_layout_entry_value(entry)
            key = (direction, entry_name)
            previous_value = entries_by_direction.setdefault(key, entry_value)
            if previous_value != entry_value:
                raise ValueError(
                    f"Conflicting stage layout '{entry_name}' metadata on {context}: "
                    f"{previous_value} vs {entry_value}"
                )

    for direction in {
        _normalized_layout_direction(getattr(layout, "direction", None))
        for layout in getattr(stage, "layout_qualifiers", []) or []
    }:
        layout_names = {
            _normalized_metadata_name(getattr(entry, "name", None))
            for layout in getattr(stage, "layout_qualifiers", []) or []
            if _normalized_layout_direction(getattr(layout, "direction", None))
            == direction
            for entry in getattr(layout, "entries", []) or []
        }
        layout_names.discard(None)
        for group in STAGE_LAYOUT_EXCLUSIVE_ENTRY_GROUPS:
            conflicts = sorted(layout_names & group)
            if len(conflicts) > 1:
                context = _stage_layout_context(stage, direction)
                raise ValueError(
                    f"Conflicting stage layout entries on {context}: "
                    f"{', '.join(conflicts)}"
                )

    layout_threadgroup_size = _stage_layout_threadgroup_size(stage)
    function_threadgroup_size = _function_threadgroup_size(
        getattr(stage, "entry_point", None)
    )
    if layout_threadgroup_size and function_threadgroup_size:
        for index, name in enumerate(("local_size_x", "local_size_y", "local_size_z")):
            layout_value = layout_threadgroup_size.get(name)
            if layout_value is None:
                continue
            function_value = function_threadgroup_size[index]
            if layout_value != function_value:
                context = _stage_layout_context(stage, "in")
                raise ValueError(
                    f"Conflicting stage threadgroup size metadata on {context}: "
                    f"{name}={layout_value} vs numthreads[{index}]={function_value}"
                )


def _node_metadata_names(node):
    names = {
        str(qualifier).lower()
        for qualifier in getattr(node, "qualifiers", []) or []
        if qualifier is not None
    }
    names.update(
        str(getattr(attr, "name", "")).lower()
        for attr in getattr(node, "attributes", []) or []
        if getattr(attr, "name", None)
    )
    return names


def _node_resource_access_names(node):
    access_names = set()
    for qualifier in getattr(node, "qualifiers", []) or []:
        access_name = RESOURCE_ACCESS_METADATA_NAMES.get(str(qualifier).lower())
        if access_name:
            access_names.add(access_name)

    for attr in getattr(node, "attributes", []) or []:
        attr_name = str(getattr(attr, "name", "")).lower()
        if attr_name == "access":
            access_value = _attribute_metadata_value(attr)
            access_name = RESOURCE_ACCESS_METADATA_NAMES.get(str(access_value).lower())
        else:
            access_name = RESOURCE_ACCESS_METADATA_NAMES.get(attr_name)
        if access_name:
            access_names.add(access_name)

    return access_names


def _normalized_metadata_names(names, canonical_names):
    return {name: canonical_names[name] for name in names if name in canonical_names}


def _metadata_name_phrase(names):
    formatted_names = sorted(f"@{name}" for name in names)
    if len(formatted_names) == 2:
        return f"{formatted_names[0]} and {formatted_names[1]}"
    return ", ".join(formatted_names)


def _normalized_metadata_name(name):
    if name is None:
        return None
    return str(name).split(".")[-1].lower()


def _normalized_layout_direction(direction):
    return _normalized_metadata_name(direction)


def _stage_layout_entry_value(entry):
    arguments = getattr(entry, "arguments", []) or []
    if not arguments:
        return None
    return expression_debug_name(arguments[0])


def _stage_layout_context(stage, direction):
    stage_name = _normalized_metadata_name(getattr(stage, "stage", None)) or "<unknown>"
    if direction is None:
        return f"{stage_name} stage layout"
    return f"{stage_name} stage '{direction}' layout"


def _attribute_metadata_value(attr):
    arguments = getattr(attr, "arguments", []) or []
    if not arguments:
        return None
    return expression_debug_name(arguments[0])


def _attribute_metadata_values(attr):
    return tuple(
        expression_debug_name(argument)
        for argument in getattr(attr, "arguments", []) or []
    )


def _metadata_value_phrase(value):
    if isinstance(value, tuple):
        return f"({', '.join(value)})"
    return str(value)


def _stage_layout_threadgroup_size(stage):
    values = {}
    for layout in getattr(stage, "layout_qualifiers", []) or []:
        direction = _normalized_layout_direction(getattr(layout, "direction", None))
        if direction != "in":
            continue
        for entry in getattr(layout, "entries", []) or []:
            entry_name = _normalized_metadata_name(getattr(entry, "name", None))
            if entry_name in {"local_size_x", "local_size_y", "local_size_z"}:
                values[entry_name] = _stage_layout_entry_value(entry)
    return values


def _function_threadgroup_size(function):
    for attr in getattr(function, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if FUNCTION_STAGE_ATTRIBUTE_NAMES.get(attr_name) != "threadgroup_size":
            continue

        values = _attribute_metadata_values(attr)
        if len(values) == 3:
            return values
    return None


def _shader_function_contexts(shader):
    for function in getattr(shader, "functions", []) or []:
        yield function, f"function '{getattr(function, 'name', '<anonymous>')}'"

    for stage in getattr(shader, "stages", {}).values():
        stage_name = _normalized_metadata_name(getattr(stage, "stage", None))
        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            yield entry_point, (
                f"{stage_name or '<unknown>'} stage entry function "
                f"'{getattr(entry_point, 'name', '<anonymous>')}'"
            )
        for function in getattr(stage, "local_functions", []) or []:
            yield function, (
                f"{stage_name or '<unknown>'} stage local function "
                f"'{getattr(function, 'name', '<anonymous>')}'"
            )


def _shader_metadata_nodes(shader):
    for variable in getattr(shader, "global_variables", []) or []:
        yield variable, _node_context("global variable", variable)

    for struct in getattr(shader, "structs", []) or []:
        yield from _struct_member_metadata_nodes(struct, "struct")

    for cbuffer in getattr(shader, "cbuffers", []) or []:
        yield from _struct_member_metadata_nodes(cbuffer, "cbuffer")

    for function in getattr(shader, "functions", []) or []:
        yield from _function_metadata_nodes(function)

    for stage in getattr(shader, "stages", {}).values():
        for variable in getattr(stage, "local_variables", []) or []:
            yield variable, _node_context("stage variable", variable)
        for struct in getattr(stage, "local_structs", []) or []:
            yield from _struct_member_metadata_nodes(struct, "stage struct")
        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            yield from _function_metadata_nodes(entry_point)
        for function in getattr(stage, "local_functions", []) or []:
            yield from _function_metadata_nodes(function)


def _function_metadata_nodes(function):
    function_name = getattr(function, "name", "<anonymous>")
    for parameter in getattr(function, "parameters", []) or []:
        yield parameter, (
            f"parameter '{getattr(parameter, 'name', '<anonymous>')}' "
            f"of function '{function_name}'"
        )
    for node in _walk_ast(getattr(function, "body", [])):
        if _is_variable_node(node):
            yield node, _node_context(f"local variable in '{function_name}'", node)


def _struct_member_metadata_nodes(struct, kind):
    struct_name = getattr(struct, "name", "<anonymous>")
    for member in getattr(struct, "members", []) or []:
        yield member, (
            f"{kind} '{struct_name}' member "
            f"'{getattr(member, 'name', '<anonymous>')}'"
        )


def _node_context(kind, node):
    return f"{kind} '{getattr(node, 'name', '<anonymous>')}'"


def _named_node_set(nodes):
    return {
        getattr(node, "name", None) for node in nodes if getattr(node, "name", None)
    }


def _shader_functions(shader):
    functions = list(getattr(shader, "functions", []) or [])
    for stage in getattr(shader, "stages", {}).values():
        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            functions.append(entry_point)
        functions.extend(getattr(stage, "local_functions", []) or [])
    return functions


def _walk_ast(root):
    visited = set()

    def walk(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                yield from walk(item)
            return

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        yield value

        if hasattr(value, "__dict__"):
            for child in vars(value).values():
                yield from walk(child)

    yield from walk(root)


def _is_variable_node(node):
    return hasattr(node, "__class__") and "Variable" in node.__class__.__name__


def validate_shader_cbuffers(shader):
    """Validate source-level cbuffer constraints shared by all backends."""
    duplicate_names = collect_duplicate_cbuffer_names(
        getattr(shader, "cbuffers", []) or []
    )
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        raise SyntaxError(f"Duplicate cbuffer name(s): {names}")

    duplicate_members = collect_duplicate_cbuffer_member_names(
        getattr(shader, "cbuffers", []) or []
    )
    if duplicate_members:
        names = ", ".join(sorted(duplicate_members))
        raise SyntaxError(f"Ambiguous cbuffer member name(s): {names}")

    declaration_conflicts = collect_cbuffer_declaration_name_conflicts(shader)
    if declaration_conflicts:
        names = ", ".join(sorted(declaration_conflicts))
        raise SyntaxError(
            f"Cbuffer name(s) conflict with existing declaration(s): {names}"
        )

    global_member_conflicts = collect_cbuffer_member_global_conflicts(shader)
    if global_member_conflicts:
        names = ", ".join(sorted(global_member_conflicts))
        raise SyntaxError(
            f"Cbuffer member name(s) conflict with global declaration(s): {names}"
        )
    return validate_shader_metadata(shader)
