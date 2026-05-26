"""Shared AST validation helpers for the CrossGL translator."""

from .stage_utils import (
    SHADER_STAGE_NAMES,
    STAGE_NAME_ALIASES,
    normalize_stage_name,
)

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
    (frozenset({"row_major", "column_major"}), "@row_major and @column_major"),
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
        "builtin",
        "constant_id",
        "component",
        "format",
        "function_constant",
        "index",
        "location",
        "align",
        "offset",
        "packoffset",
        "sampler",
        "set",
        "texture",
        "uav",
        "user",
    }
)

SINGLE_VALUE_METADATA_ALIASES = {
    "group": "set",
}

MULTI_VALUE_METADATA_NAMES = frozenset({"register"})

HLSL_SEMANTIC_METADATA_BASE_NAMES = frozenset(
    {
        "binormal",
        "blendindices",
        "blendweight",
        "clipdistance",
        "color",
        "culldistance",
        "depth",
        "fog",
        "normal",
        "position",
        "positiont",
        "psize",
        "tangent",
        "tessfactor",
        "texcoord",
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

IMAGE_FORMAT_METADATA_NAMES = frozenset(
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

STORAGE_IMAGE_TYPE_NAMES = frozenset(
    {
        "image1d",
        "image1darray",
        "image2d",
        "image3d",
        "imagecube",
        "image2darray",
        "image2dms",
        "image2dmsarray",
        "imagebuffer",
        "iimage1d",
        "iimage1darray",
        "iimage2d",
        "iimage3d",
        "iimagecube",
        "iimage2darray",
        "iimage2dms",
        "iimage2dmsarray",
        "iimagebuffer",
        "uimage1d",
        "uimage1darray",
        "uimage2d",
        "uimage3d",
        "uimagecube",
        "uimage2darray",
        "uimage2dms",
        "uimage2dmsarray",
        "uimagebuffer",
    }
)

RESOURCE_BUFFER_TYPE_NAMES = frozenset(
    {
        "appendstructuredbuffer",
        "buffer",
        "byteaddressbuffer",
        "consumestructuredbuffer",
        "rasterizerorderedbuffer",
        "rasterizerorderedbyteaddressbuffer",
        "rasterizerorderedstructuredbuffer",
        "rwbuffer",
        "rwbyteaddressbuffer",
        "rwstructuredbuffer",
        "structuredbuffer",
    }
)

UAV_RESOURCE_BUFFER_TYPE_NAMES = frozenset(
    {
        "appendstructuredbuffer",
        "consumestructuredbuffer",
        "rasterizerorderedbuffer",
        "rasterizerorderedbyteaddressbuffer",
        "rasterizerorderedstructuredbuffer",
        "rwbuffer",
        "rwbyteaddressbuffer",
        "rwstructuredbuffer",
    }
)

SAMPLER_STATE_TYPE_NAMES = frozenset(
    {
        "sampler",
        "samplerstate",
        "samplercomparisonstate",
    }
)

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

MEMORY_LAYOUT_METADATA_NAMES = {
    "std140": "std140",
    "std430": "std430",
    "scalar": "scalar",
}

DECLARATION_ROLE_METADATA_NAMES = {
    "payload": "payload",
    "ray_payload": "payload",
    "raypayload": "payload",
    "hit_attribute": "hit_attribute",
    "hitattribute": "hit_attribute",
    "callable_data": "callable_data",
    "callabledata": "callable_data",
    "mesh_payload": "mesh_payload",
    "meshpayload": "mesh_payload",
    "vertices": "mesh_vertices",
    "indices": "mesh_indices",
    "primitives": "mesh_primitives",
}

STRUCT_DECLARATION_ROLE_NAMES = frozenset(
    {
        "callable_data",
        "hit_attribute",
        "mesh_payload",
        "payload",
    }
)

PARAMETER_DECLARATION_ROLE_NAMES = frozenset(
    {
        "callable_data",
        "hit_attribute",
        "mesh_indices",
        "mesh_payload",
        "mesh_primitives",
        "mesh_vertices",
        "payload",
    }
)

BUILTIN_SEMANTIC_METADATA_NAMES = {
    "position": "position",
    "gl_position": "position",
    "sv_position": "position",
    "point_size": "point_size",
    "gl_pointsize": "point_size",
    "vertex_index": "vertex_index",
    "vertexid": "vertex_index",
    "gl_vertexid": "vertex_index",
    "sv_vertexid": "vertex_index",
    "instance_index": "instance_index",
    "instanceid": "instance_index",
    "gl_instanceid": "instance_index",
    "sv_instanceid": "instance_index",
    "front_facing": "front_facing",
    "gl_frontfacing": "front_facing",
    "sv_isfrontface": "front_facing",
    "frag_coord": "frag_coord",
    "gl_fragcoord": "frag_coord",
    "point_coord": "point_coord",
    "gl_pointcoord": "point_coord",
    "frag_depth": "frag_depth",
    "gl_fragdepth": "frag_depth",
    "sv_depth": "frag_depth",
    "primitive_id": "primitive_id",
    "primitiveid": "primitive_id",
    "gl_primitiveid": "primitive_id",
    "sv_primitiveid": "primitive_id",
    "global_invocation_id": "global_invocation_id",
    "gl_globalinvocationid": "global_invocation_id",
    "sv_dispatchthreadid": "global_invocation_id",
    "local_invocation_id": "local_invocation_id",
    "gl_localinvocationid": "local_invocation_id",
    "sv_groupthreadid": "local_invocation_id",
    "workgroup_id": "workgroup_id",
    "gl_workgroupid": "workgroup_id",
    "sv_groupid": "workgroup_id",
    "local_invocation_index": "local_invocation_index",
    "gl_localinvocationindex": "local_invocation_index",
    "sv_groupindex": "local_invocation_index",
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

SHADER_STAGE_ATTRIBUTE_ALIASES = STAGE_NAME_ALIASES
SHADER_STAGE_ATTRIBUTE_NAMES = SHADER_STAGE_NAMES

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

FUNCTION_STAGE_LAYOUT_FLAG_ALIASES = {
    "domain": {
        "tri": "triangles",
        "triangle": "triangles",
        "triangles": "triangles",
        "quad": "quads",
        "quads": "quads",
        "isoline": "isolines",
        "isolines": "isolines",
    },
    "partitioning": {
        "integer": "equal_spacing",
        "equal": "equal_spacing",
        "equal_spacing": "equal_spacing",
        "fractional_even": "fractional_even_spacing",
        "fractional_even_spacing": "fractional_even_spacing",
        "fractional_odd": "fractional_odd_spacing",
        "fractional_odd_spacing": "fractional_odd_spacing",
    },
}

FUNCTION_STAGE_OUTPUT_TOPOLOGY_ALIASES = {
    "point": ("points",),
    "points": ("points",),
    "line": ("lines",),
    "lines": ("lines",),
    "line_strip": ("line_strip",),
    "triangle": ("triangles",),
    "triangles": ("triangles",),
    "triangle_cw": ("triangles", "cw"),
    "triangle_ccw": ("triangles", "ccw"),
}

FUNCTION_STAGE_LAYOUT_VALUE_ENTRIES = {
    "outputcontrolpoints": ("out", ("vertices", "outputcontrolpoints")),
    "maxvertexcount": ("out", ("max_vertices", "maxvertexcount")),
    "maxprimitivecount": ("out", ("max_primitives", "maxprimitivecount")),
}

TESSELLATION_CONTROL_STAGE_LAYOUT_ENTRIES = frozenset(
    {
        "outputcontrolpoints",
        "vertices",
    }
)

TESSELLATION_EVALUATION_STAGE_LAYOUT_ENTRIES = frozenset(
    {
        "ccw",
        "cw",
        "equal_spacing",
        "fractional_even_spacing",
        "fractional_odd_spacing",
        "isolines",
        "point_mode",
        "quads",
    }
)

TESSELLATION_CONTROL_FUNCTION_ATTRIBUTE_NAMES = frozenset(
    {
        "outputcontrolpoints",
        "patchconstantfunc",
    }
)

TESSELLATION_STAGE_FUNCTION_ATTRIBUTE_NAMES = frozenset(
    {
        "domain",
        "partitioning",
    }
)

TESSELLATION_EVALUATION_FUNCTION_FLAG_NAMES = frozenset(
    {
        "ccw",
        "cw",
        "point_mode",
    }
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
        validate_stage_shader_attribute_metadata(stage)
        validate_stage_layout_metadata(stage)
    validate_tessellation_patch_cross_stage_metadata(shader)
    validate_cross_stage_interfaces(shader)
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

    validate_hlsl_semantic_metadata(function, context)


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

    memory_layouts = _node_memory_layout_names(node)
    if len(set(memory_layouts.values())) > 1:
        raise ValueError(
            f"Conflicting memory layout metadata on {context}: "
            f"{_metadata_name_phrase(memory_layouts)}"
        )

    declaration_roles = _normalized_metadata_names(
        names, DECLARATION_ROLE_METADATA_NAMES
    )
    if len(set(declaration_roles.values())) > 1:
        raise ValueError(
            f"Conflicting declaration role metadata on {context}: "
            f"{_metadata_name_phrase(declaration_roles)}"
        )
    validate_declaration_role_metadata(node, context, declaration_roles)

    builtin_semantics = _node_builtin_semantic_metadata(node)
    if len(set(builtin_semantics.values())) > 1:
        raise ValueError(
            f"Conflicting builtin metadata on {context}: "
            f"{_metadata_name_phrase(builtin_semantics)}"
        )

    access_names = _node_resource_access_names(node)
    if len(access_names) > 1:
        access_list = ", ".join(sorted(f"@{name}" for name in access_names))
        raise ValueError(
            f"Conflicting resource access metadata on {context}: {access_list}"
        )
    if access_names and not _node_allows_resource_access_metadata(node):
        access_list = ", ".join(sorted(f"@{name}" for name in access_names))
        type_name = _type_debug_name(_node_declared_type(node)) or "<unknown>"
        raise ValueError(
            f"Resource access metadata on {context} requires resource type: "
            f"{access_list} on {type_name}"
        )

    validate_image_format_metadata(node, context)
    validate_descriptor_index_metadata(node, context)
    validate_hlsl_semantic_metadata(node, context)

    values_by_name = {}
    for attr in getattr(node, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        metadata_name = SINGLE_VALUE_METADATA_ALIASES.get(attr_name, attr_name)
        if metadata_name not in SINGLE_VALUE_METADATA_NAMES:
            continue
        attr_values = _attribute_metadata_values(attr)
        if len(attr_values) > 1:
            raise ValueError(
                f"Metadata '@{attr_name}' on {context} accepts at most one value: "
                f"{_metadata_value_phrase(attr_values)}"
            )
        attr_value = attr_values[0] if attr_values else None
        if attr_value is None:
            continue
        previous_value = values_by_name.setdefault(metadata_name, attr_value)
        if previous_value != attr_value:
            raise ValueError(
                f"Conflicting {metadata_name} metadata on {context}: "
                f"{previous_value} vs {attr_value}"
            )

    multi_values_by_name = {}
    for attr in getattr(node, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if attr_name not in MULTI_VALUE_METADATA_NAMES:
            continue
        attr_value = _attribute_metadata_values(attr)
        previous_value = multi_values_by_name.setdefault(attr_name, attr_value)
        if previous_value != attr_value:
            raise ValueError(
                f"Conflicting {attr_name} metadata on {context}: "
                f"{_metadata_value_phrase(previous_value)} vs "
                f"{_metadata_value_phrase(attr_value)}"
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
            validate_stage_layout_entry_placement(stage, direction, entry_name)

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

    validate_function_stage_layout_metadata(stage)
    validate_tessellation_patch_parameter_metadata(stage)


def validate_cross_stage_interfaces(shader):
    """Validate explicit producer/consumer graphics-stage interfaces."""
    stages = _shader_stages_by_name(shader)

    graphics_chain = [
        "vertex",
        "tessellation_control",
        "tessellation_evaluation",
        "geometry",
        "fragment",
    ]
    present_chain = [
        stage_name for stage_name in graphics_chain if stage_name in stages
    ]
    for producer_name, consumer_name in zip(present_chain, present_chain[1:]):
        _validate_stage_interface_pair(
            shader,
            producer_name,
            stages[producer_name],
            consumer_name,
            stages[consumer_name],
        )

    if "mesh" in stages and "fragment" in stages:
        _validate_stage_interface_pair(
            shader, "mesh", stages["mesh"], "fragment", stages["fragment"]
        )


def _validate_stage_interface_pair(
    shader, producer_name, producer_stage, consumer_name, consumer_stage
):
    producer_outputs = _stage_interface_entries(
        shader, producer_stage, producer_name, "out"
    )
    consumer_inputs = _stage_interface_entries(
        shader, consumer_stage, consumer_name, "in"
    )

    output_by_key = _stage_interface_entries_by_key(
        producer_outputs, producer_name, "output"
    )
    _stage_interface_entries_by_key(consumer_inputs, consumer_name, "input")

    if not consumer_inputs:
        return

    for consumer_entry in consumer_inputs:
        if not consumer_entry["keys"]:
            continue

        producer_entry = _matching_stage_interface_entry(
            output_by_key, consumer_entry["keys"]
        )
        if producer_entry is None:
            raise ValueError(
                f"Missing cross-stage interface output from {producer_name} to "
                f"{consumer_name} for {_stage_interface_key_phrase(consumer_entry)} "
                f"consumed by {consumer_entry['context']}"
            )

        _validate_stage_interface_key_consistency(
            producer_name, consumer_name, producer_entry, consumer_entry
        )

        producer_type = producer_entry.get("type")
        consumer_type = consumer_entry.get("type")
        if (
            producer_type
            and consumer_type
            and _canonical_stage_interface_type(producer_type)
            != _canonical_stage_interface_type(consumer_type)
        ):
            raise ValueError(
                f"Conflicting cross-stage interface type from {producer_name} to "
                f"{consumer_name} for {_stage_interface_key_phrase(consumer_entry)}: "
                f"{producer_entry['context']} is {producer_type}, "
                f"{consumer_entry['context']} is {consumer_type}"
            )

        _validate_stage_interface_interpolation(
            producer_name, consumer_name, producer_entry, consumer_entry
        )


def _stage_interface_entries_by_key(entries, stage_name, direction):
    entries_by_key = {}
    for entry in entries:
        for key in entry["keys"]:
            previous = entries_by_key.setdefault(key, entry)
            if previous is not entry:
                raise ValueError(
                    f"Duplicate {direction} stage interface metadata on "
                    f"{stage_name} stage for {_format_stage_interface_key(key)}: "
                    f"{previous['context']} and {entry['context']}"
                )
    return entries_by_key


def _matching_stage_interface_entry(entries_by_key, keys):
    for key in keys:
        entry = entries_by_key.get(key)
        if entry is not None:
            return entry
    return None


def _validate_stage_interface_key_consistency(
    producer_name, consumer_name, producer_entry, consumer_entry
):
    for kind in ("location", "semantic"):
        producer_keys = {
            key for key in producer_entry.get("keys", ()) if key[0] == kind
        }
        consumer_keys = {
            key for key in consumer_entry.get("keys", ()) if key[0] == kind
        }
        if producer_keys and consumer_keys and producer_keys.isdisjoint(consumer_keys):
            raise ValueError(
                f"Conflicting cross-stage interface {kind} metadata from "
                f"{producer_name} to {consumer_name}: "
                f"{producer_entry['context']} has "
                f"{_format_stage_interface_key_set(producer_keys)}, "
                f"{consumer_entry['context']} has "
                f"{_format_stage_interface_key_set(consumer_keys)}"
            )


def _validate_stage_interface_interpolation(
    producer_name, consumer_name, producer_entry, consumer_entry
):
    for kind, description in (
        ("interpolation_mode", "interpolation mode"),
        ("interpolation_sampling", "interpolation sampling"),
    ):
        producer_value = producer_entry.get(kind)
        consumer_value = consumer_entry.get(kind)
        if producer_value and consumer_value and producer_value[1] != consumer_value[1]:
            raise ValueError(
                f"Conflicting cross-stage interface {description} from "
                f"{producer_name} to {consumer_name} for "
                f"{_stage_interface_key_phrase(consumer_entry)}: "
                f"{producer_value[0]} vs {consumer_value[0]}"
            )


def _stage_interface_entries(shader, stage, stage_name, direction):
    struct_map = _stage_struct_map(shader, stage)
    entries = []

    for variable in getattr(stage, "local_variables", []) or []:
        if _stage_interface_direction(variable) == direction:
            entries.append(
                _stage_interface_entry(
                    variable,
                    getattr(variable, "var_type", getattr(variable, "vtype", None)),
                    f"{stage_name} stage {direction} variable "
                    f"'{getattr(variable, 'name', '<anonymous>')}'",
                )
            )

    entry_point = getattr(stage, "entry_point", None)
    if entry_point is None:
        return [entry for entry in entries if entry["keys"]]

    if direction == "out":
        entries.extend(
            _function_return_stage_interface_entries(
                entry_point, struct_map, stage_name
            )
        )
    else:
        entries.extend(
            _function_parameter_stage_interface_entries(
                entry_point, struct_map, stage_name
            )
        )

    return [entry for entry in entries if entry["keys"]]


def _function_return_stage_interface_entries(function, struct_map, stage_name):
    return_type = getattr(function, "return_type", None)
    return_struct = _resolve_stage_interface_struct(return_type, struct_map)
    if return_struct is not None:
        return [
            _stage_interface_entry(
                member,
                getattr(member, "member_type", None),
                f"{stage_name} stage return member "
                f"'{return_struct.name}.{getattr(member, 'name', '<anonymous>')}'",
            )
            for member in getattr(return_struct, "members", []) or []
        ]

    return [
        _stage_interface_entry(
            function,
            return_type,
            f"{stage_name} stage return value of "
            f"'{getattr(function, 'name', '<anonymous>')}'",
        )
    ]


def _function_parameter_stage_interface_entries(function, struct_map, stage_name):
    entries = []
    for parameter in getattr(function, "parameters", []) or []:
        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        parameter_struct = _resolve_stage_interface_struct(parameter_type, struct_map)
        if parameter_struct is not None:
            entries.extend(
                _stage_interface_entry(
                    member,
                    getattr(member, "member_type", None),
                    f"{stage_name} stage parameter "
                    f"'{getattr(parameter, 'name', '<anonymous>')}."
                    f"{getattr(member, 'name', '<anonymous>')}'",
                )
                for member in getattr(parameter_struct, "members", []) or []
            )
        else:
            entries.append(
                _stage_interface_entry(
                    parameter,
                    parameter_type,
                    f"{stage_name} stage parameter "
                    f"'{getattr(parameter, 'name', '<anonymous>')}'",
                )
            )
    return entries


def _stage_interface_entry(node, type_node, context):
    return {
        "node": node,
        "type": _stage_interface_type_name(type_node),
        "context": context,
        "keys": _stage_interface_keys(node),
        "interpolation_mode": _stage_interface_interpolation_mode(node),
        "interpolation_sampling": _stage_interface_interpolation_sampling(node),
    }


def _stage_interface_direction(node):
    names = _node_metadata_names(node)
    if "out" in names or "output" in names:
        return "out"
    if "in" in names or "input" in names:
        return "in"
    return None


def _stage_interface_keys(node):
    attributes = getattr(node, "attributes", []) or []
    location_value = _attribute_value_by_name(attributes, "location")
    component_value = _attribute_value_by_name(attributes, "component")
    keys = []
    if location_value is not None:
        keys.append(("location", location_value, component_value or "0"))

    for attr in attributes:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if _is_cross_stage_semantic_name(attr_name):
            keys.append(("semantic", attr_name))
    return tuple(keys)


def _is_cross_stage_semantic_name(name):
    if not name:
        return False
    if name.startswith("gl_") or name.startswith("sv_"):
        return False
    if name in {"builtin", "position"}:
        return False
    return _is_hlsl_semantic_metadata_name(name)


def _attribute_value_by_name(attributes, name):
    for attr in attributes:
        if _normalized_metadata_name(getattr(attr, "name", None)) == name:
            return _attribute_metadata_value(attr)
    return None


def _stage_interface_interpolation_mode(node):
    names = _node_metadata_names(node)
    matches = _normalized_metadata_names(names, INTERPOLATION_MODE_METADATA_NAMES)
    if not matches:
        return None
    attr_name = sorted(matches)[0]
    return f"@{attr_name}", matches[attr_name]


def _stage_interface_interpolation_sampling(node):
    names = _node_metadata_names(node)
    matches = _normalized_metadata_names(names, INTERPOLATION_SAMPLING_METADATA_NAMES)
    if not matches:
        return None
    attr_name = sorted(matches)[0]
    return f"@{attr_name}", matches[attr_name]


def _stage_interface_key_phrase(entry):
    if not entry.get("keys"):
        return "unkeyed interface"
    return " / ".join(_format_stage_interface_key(key) for key in entry["keys"])


def _format_stage_interface_key(key):
    if key[0] == "location":
        return f"location {key[1]} component {key[2]}"
    if key[0] == "semantic":
        return f"semantic {key[1]}"
    return " ".join(str(part) for part in key)


def _format_stage_interface_key_set(keys):
    return " / ".join(_format_stage_interface_key(key) for key in sorted(keys))


def _shader_stages_by_name(shader):
    stages = {}
    for stage_key, stage in getattr(shader, "stages", {}).items():
        stage_name = _normalized_metadata_name(stage_key)
        if stage_name:
            stages[stage_name] = stage
    return stages


def _stage_struct_map(shader, stage):
    structs = {}
    for struct in getattr(shader, "structs", []) or []:
        name = getattr(struct, "name", None)
        if name:
            structs[name] = struct
    for struct in getattr(stage, "local_structs", []) or []:
        name = getattr(struct, "name", None)
        if name:
            structs[name] = struct
    return structs


def _resolve_stage_interface_struct(type_node, struct_map):
    if type_node is None:
        return None
    while hasattr(type_node, "element_type"):
        type_node = type_node.element_type
    type_name = getattr(type_node, "name", None)
    if not type_name:
        return None
    return struct_map.get(type_name)


def _stage_interface_type_name(type_node):
    if type_node is None:
        return None
    if isinstance(type_node, str):
        return type_node
    if hasattr(type_node, "value"):
        return str(getattr(type_node, "value"))
    if type_node.__class__.__name__ == "ArrayType":
        size = (
            expression_debug_name(type_node.size) if type_node.size is not None else ""
        )
        return f"{_stage_interface_type_name(type_node.element_type)}[{size}]"
    if hasattr(type_node, "pointee_type"):
        return f"{_stage_interface_type_name(type_node.pointee_type)}*"
    if hasattr(type_node, "referenced_type"):
        return f"{_stage_interface_type_name(type_node.referenced_type)}&"
    if (
        hasattr(type_node, "element_type")
        and hasattr(type_node, "rows")
        and hasattr(type_node, "cols")
    ):
        element_type = _stage_interface_type_name(type_node.element_type)
        prefix = "dmat" if element_type == "double" else "mat"
        if type_node.rows == type_node.cols:
            return f"{prefix}{type_node.rows}"
        return f"{prefix}{type_node.rows}x{type_node.cols}"
    if type_node.__class__.__name__ == "VectorType":
        element_type = _stage_interface_type_name(type_node.element_type)
        vector_prefixes = {
            "float": "vec",
            "int": "ivec",
            "uint": "uvec",
            "double": "dvec",
            "bool": "bvec",
        }
        return f"{vector_prefixes.get(element_type, element_type)}{type_node.size}"
    if hasattr(type_node, "name"):
        generic_args = getattr(type_node, "generic_args", []) or []
        if generic_args:
            args = ", ".join(_stage_interface_type_name(arg) for arg in generic_args)
            return f"{type_node.name}<{args}>"
        return str(type_node.name)
    return str(type_node)


def _canonical_stage_interface_type(type_name):
    normalized = str(type_name)
    aliases = {
        "float2": "vec2",
        "float3": "vec3",
        "float4": "vec4",
        "int2": "ivec2",
        "int3": "ivec3",
        "int4": "ivec4",
        "uint2": "uvec2",
        "uint3": "uvec3",
        "uint4": "uvec4",
        "double2": "dvec2",
        "double3": "dvec3",
        "double4": "dvec4",
    }
    return aliases.get(normalized, normalized)


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


def _node_allows_resource_access_metadata(node):
    names = _node_metadata_names(node)
    node_type = _node_declared_type(node)
    return (
        _is_storage_image_type(node_type)
        or _is_resource_buffer_type(node_type)
        or _type_has_access_qualifier(node_type)
        or "buffer" in names
        or "glsl_buffer_block" in names
        or bool(_normalized_metadata_names(names, ADDRESS_SPACE_METADATA_NAMES))
        or any(name in {"buffer", "texture", "sampler", "uav"} for name in names)
    )


def validate_declaration_role_metadata(node, context, declaration_roles):
    """Validate declaration role metadata is attached to supported owners."""
    if not declaration_roles:
        return

    role_values = set(declaration_roles.values())
    role_names = sorted(declaration_roles)

    if _is_parameter_node(node):
        unsupported = role_values - PARAMETER_DECLARATION_ROLE_NAMES
        if not unsupported:
            return

    if _is_struct_node(node):
        unsupported = role_values - STRUCT_DECLARATION_ROLE_NAMES
        if not unsupported:
            return

    role_list = ", ".join(f"@{name}" for name in role_names)
    if role_values <= STRUCT_DECLARATION_ROLE_NAMES:
        requirement = "function parameter or role struct"
    else:
        requirement = "function parameter"
    raise ValueError(
        f"Declaration role metadata on {context} {role_list} " f"requires {requirement}"
    )


def validate_image_format_metadata(node, context):
    """Validate storage-image format metadata appears on storage images only."""
    format_metadata = _node_image_format_metadata(node)
    if not format_metadata:
        return

    node_type = _node_declared_type(node)
    if _is_storage_image_type(node_type):
        return

    format_list = ", ".join(sorted(f"@{name}" for name in format_metadata))
    type_name = _type_debug_name(node_type) or "<unknown>"
    raise ValueError(
        f"Image format metadata on {context} requires storage image type: "
        f"{format_list} on {type_name}"
    )


def validate_descriptor_index_metadata(node, context):
    """Validate descriptor role attributes match the declared resource family."""
    attributes = getattr(node, "attributes", []) or []
    node_type = _node_declared_type(node)

    for attr in attributes:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if attr_name == "texture" and not _is_texture_descriptor_type(node_type):
            _raise_descriptor_role_error("texture", "texture resource", node, context)
        if attr_name == "sampler" and not _is_sampler_descriptor_type(node_type):
            _raise_descriptor_role_error("sampler", "sampler resource", node, context)
        if attr_name == "uav" and not _is_uav_descriptor_type(node_type):
            _raise_descriptor_role_error("uav", "storage resource", node, context)


def _raise_descriptor_role_error(role, required_role, node, context):
    type_name = _type_debug_name(_node_declared_type(node)) or "<unknown>"
    raise ValueError(
        f"{role} metadata on {context} requires {required_role}: "
        f"@{role} on {type_name}"
    )


def _node_image_format_metadata(node):
    format_metadata = set()
    for attr in getattr(node, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if attr_name in IMAGE_FORMAT_METADATA_NAMES:
            format_metadata.add(attr_name)
            continue
        if attr_name != "format":
            continue
        format_name = _normalized_metadata_name(_attribute_metadata_value(attr))
        if format_name:
            format_metadata.add(f"format({format_name})")
        else:
            format_metadata.add("format")
    return format_metadata


def _node_declared_type(node):
    return getattr(
        node,
        "var_type",
        getattr(node, "param_type", getattr(node, "member_type", None)),
    )


def _is_parameter_node(node):
    return node.__class__.__name__ == "ParameterNode"


def _is_struct_node(node):
    return node.__class__.__name__ == "StructNode"


def _is_storage_image_type(type_node):
    base_name = _type_base_name(type_node)
    if not base_name:
        return False

    normalized_name = base_name.lower()
    if normalized_name in STORAGE_IMAGE_TYPE_NAMES:
        return True
    if normalized_name.startswith(("rwtexture", "rasterizerorderedtexture")):
        return True
    if normalized_name.startswith("texture_storage"):
        return True
    if normalized_name.startswith("texture") and _type_has_access_qualifier(type_node):
        return True
    return False


def _is_resource_buffer_type(type_node):
    base_name = _type_base_name(type_node)
    return bool(base_name and base_name.lower() in RESOURCE_BUFFER_TYPE_NAMES)


def _is_texture_descriptor_type(type_node):
    if _is_storage_image_type(type_node):
        return True

    base_name = _type_base_name(type_node)
    if not base_name:
        return False

    normalized_name = base_name.lower()
    if normalized_name in SAMPLER_STATE_TYPE_NAMES:
        return False
    return normalized_name.startswith(
        (
            "isampler",
            "itexture",
            "sampler1d",
            "sampler2d",
            "sampler3d",
            "samplercube",
            "texture",
            "usampler",
            "utexture",
        )
    )


def _is_sampler_descriptor_type(type_node):
    base_name = _type_base_name(type_node)
    return bool(base_name and base_name.lower() in SAMPLER_STATE_TYPE_NAMES)


def _is_uav_descriptor_type(type_node):
    if _is_storage_image_type(type_node):
        return True

    base_name = _type_base_name(type_node)
    return bool(base_name and base_name.lower() in UAV_RESOURCE_BUFFER_TYPE_NAMES)


def _type_has_access_qualifier(type_node):
    generic_args = getattr(type_node, "generic_args", None)
    if generic_args is None:
        nested_type = _nested_resource_type(type_node)
        generic_args = getattr(nested_type, "generic_args", None)

    for argument in generic_args or []:
        argument_name = _type_debug_name(argument)
        if argument_name is None:
            continue
        if RESOURCE_ACCESS_METADATA_NAMES.get(argument_name.lower()):
            return True
    return False


def _type_base_name(type_node):
    type_node = _nested_resource_type(type_node)
    name = getattr(type_node, "name", None)
    if name is not None:
        return str(name).split("<", 1)[0]

    if type_node is None:
        return None

    return str(type_node).split("<", 1)[0].strip()


def _nested_resource_type(type_node):
    current = type_node
    while current is not None:
        nested = None
        for attribute_name in ("element_type", "pointee_type", "referenced_type"):
            candidate = getattr(current, attribute_name, None)
            if candidate is not None:
                nested = candidate
                break
        if nested is None:
            return current
        current = nested
    return None


def _type_debug_name(type_node):
    if type_node is None:
        return None
    name = getattr(type_node, "name", None)
    if name is not None:
        return str(name)
    return str(type_node)


def _node_memory_layout_names(node):
    memory_layouts = _normalized_metadata_names(
        _node_metadata_names(node), MEMORY_LAYOUT_METADATA_NAMES
    )
    for attr in getattr(node, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if attr_name != "glsl_buffer_block":
            continue
        block_layout = _normalized_metadata_name(_attribute_metadata_value(attr))
        if block_layout in MEMORY_LAYOUT_METADATA_NAMES:
            memory_layouts[f"glsl_buffer_block({block_layout})"] = (
                MEMORY_LAYOUT_METADATA_NAMES[block_layout]
            )
    return memory_layouts


def _node_builtin_semantic_metadata(node):
    metadata = {}
    for attr in getattr(node, "attributes", []) or []:
        raw_name = getattr(attr, "name", None)
        attr_name = _normalized_metadata_name(raw_name)
        if attr_name == "builtin":
            builtin_name = _normalized_metadata_name(_attribute_metadata_value(attr))
            if not builtin_name:
                continue
            metadata[f"builtin({builtin_name})"] = BUILTIN_SEMANTIC_METADATA_NAMES.get(
                builtin_name, builtin_name
            )
            continue

        builtin_name = _builtin_semantic_attribute_name(raw_name)
        if builtin_name is not None:
            metadata[str(raw_name)] = builtin_name
    return metadata


def _builtin_semantic_attribute_name(name):
    attr_name = _normalized_metadata_name(name)
    if attr_name not in BUILTIN_SEMANTIC_METADATA_NAMES:
        return None

    raw_name = str(name)
    if raw_name == attr_name or attr_name.startswith(("gl_", "sv_")):
        return BUILTIN_SEMANTIC_METADATA_NAMES[attr_name]
    return None


def validate_hlsl_semantic_metadata(node, context):
    semantic_metadata = _node_hlsl_semantic_metadata(node)
    if len(semantic_metadata) > 1:
        raise ValueError(
            f"Conflicting semantic metadata on {context}: "
            f"{_metadata_attribute_phrase(semantic_metadata)}"
        )


def _node_hlsl_semantic_metadata(node):
    metadata = set()
    for attr in getattr(node, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if _is_hlsl_semantic_metadata_name(attr_name):
            metadata.add((attr_name, _attribute_metadata_values(attr)))
    return metadata


def _is_hlsl_semantic_metadata_name(name):
    if not name:
        return False
    if name.startswith("sv_"):
        return True
    base_name = name.rstrip("0123456789")
    return base_name in HLSL_SEMANTIC_METADATA_BASE_NAMES


def _normalized_metadata_names(names, canonical_names):
    return {name: canonical_names[name] for name in names if name in canonical_names}


def _metadata_name_phrase(names):
    formatted_names = sorted(f"@{name}" for name in names)
    if len(formatted_names) == 2:
        return f"{formatted_names[0]} and {formatted_names[1]}"
    return ", ".join(formatted_names)


def _metadata_attribute_phrase(metadata):
    formatted_metadata = []
    for name, value in metadata:
        if value:
            formatted_metadata.append(f"@{name}{_metadata_value_phrase(value)}")
        else:
            formatted_metadata.append(f"@{name}")
    formatted_metadata = sorted(formatted_metadata)
    if len(formatted_metadata) == 2:
        return f"{formatted_metadata[0]} and {formatted_metadata[1]}"
    return ", ".join(formatted_metadata)


def validate_function_stage_layout_metadata(stage):
    """Validate equivalent function and stage layout metadata agree."""
    function = getattr(stage, "entry_point", None)
    if function is None:
        return

    layout_entries = _stage_layout_entries_by_direction(stage)
    layout_names = _stage_layout_names_by_direction(stage)

    for attr in getattr(function, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        validate_stage_function_attribute_placement(stage, function, attr_name)

        attr_value = _first_attribute_metadata_value(attr)
        if attr_value is None:
            continue

        if attr_name in FUNCTION_STAGE_LAYOUT_FLAG_ALIASES:
            canonical_entry = FUNCTION_STAGE_LAYOUT_FLAG_ALIASES[attr_name].get(
                _normalized_metadata_name(attr_value)
            )
            if canonical_entry is not None:
                _validate_function_stage_layout_flag(
                    layout_names,
                    "in",
                    canonical_entry,
                    attr_name,
                    attr_value,
                    stage,
                )
            continue

        if attr_name == "outputtopology":
            canonical_entries = FUNCTION_STAGE_OUTPUT_TOPOLOGY_ALIASES.get(
                _normalized_metadata_name(attr_value), ()
            )
            for canonical_entry in canonical_entries:
                _validate_function_stage_layout_flag(
                    layout_names,
                    "in",
                    canonical_entry,
                    attr_name,
                    attr_value,
                    stage,
                )
            continue

        value_entry = FUNCTION_STAGE_LAYOUT_VALUE_ENTRIES.get(attr_name)
        if value_entry is None:
            continue

        direction, entry_names = value_entry
        function_value = attr_value
        for entry_name in entry_names:
            key = (direction, entry_name)
            if key not in layout_entries:
                continue
            layout_value = layout_entries[key]
            if layout_value != function_value:
                context = _stage_layout_context(stage, direction)
                raise ValueError(
                    f"Conflicting stage/function metadata on {context}: "
                    f"layout {entry_name}={layout_value} vs "
                    f"{attr_name}={function_value}"
                )


def validate_stage_layout_entry_placement(stage, direction, entry_name):
    """Validate stage layout entries appear on stages that can consume them."""
    stage_name = _normalized_metadata_name(getattr(stage, "stage", None))
    context = _stage_layout_context(stage, direction)

    if entry_name in TESSELLATION_CONTROL_STAGE_LAYOUT_ENTRIES and (
        stage_name != "tessellation_control" or direction != "out"
    ):
        raise ValueError(
            f"Stage layout '{entry_name}' on {context} requires "
            "tessellation_control stage 'out' layout"
        )

    if entry_name in TESSELLATION_EVALUATION_STAGE_LAYOUT_ENTRIES and (
        stage_name != "tessellation_evaluation" or direction != "in"
    ):
        raise ValueError(
            f"Stage layout '{entry_name}' on {context} requires "
            "tessellation_evaluation stage 'in' layout"
        )


def validate_stage_function_attribute_placement(stage, function, attr_name):
    """Validate function metadata that is meaningful only on specific stages."""
    if not attr_name:
        return

    stage_name = _normalized_metadata_name(getattr(stage, "stage", None))
    context = _stage_entry_function_context(stage, function)

    if (
        attr_name in TESSELLATION_CONTROL_FUNCTION_ATTRIBUTE_NAMES
        and stage_name != "tessellation_control"
    ):
        raise ValueError(
            f"Function metadata '@{attr_name}' on {context} requires "
            "tessellation_control stage"
        )

    if attr_name in TESSELLATION_STAGE_FUNCTION_ATTRIBUTE_NAMES and stage_name not in {
        "tessellation_control",
        "tessellation_evaluation",
    }:
        raise ValueError(
            f"Function metadata '@{attr_name}' on {context} requires "
            "tessellation stage"
        )

    if (
        attr_name in TESSELLATION_EVALUATION_FUNCTION_FLAG_NAMES
        and stage_name != "tessellation_evaluation"
    ):
        raise ValueError(
            f"Function metadata '@{attr_name}' on {context} requires "
            "tessellation_evaluation stage"
        )


def validate_stage_shader_attribute_metadata(stage):
    """Validate function shader-stage attributes inside an explicit stage block."""
    stage_name = _canonical_shader_stage_attribute_name(getattr(stage, "stage", None))
    if stage_name is None:
        return

    functions = []
    entry_point = getattr(stage, "entry_point", None)
    if entry_point is not None:
        functions.append(entry_point)
    functions.extend(getattr(stage, "local_functions", []) or [])

    for function in functions:
        for attr in getattr(function, "attributes", []) or []:
            attr_name = _normalized_metadata_name(getattr(attr, "name", None))
            if attr_name != "shader":
                continue

            context = _stage_function_context(stage, function)
            values = _attribute_metadata_values(attr)
            attr_display = _shader_attribute_display(attr)
            if len(values) != 1:
                raise ValueError(
                    f"Function metadata '{attr_display}' on {context} requires "
                    "exactly one shader stage value"
                )

            requested_stage = _canonical_shader_stage_attribute_name(values[0])
            if requested_stage not in SHADER_STAGE_ATTRIBUTE_NAMES:
                valid_stages = ", ".join(sorted(SHADER_STAGE_ATTRIBUTE_NAMES))
                raise ValueError(
                    f"Function metadata '{attr_display}' on {context} uses "
                    f"unknown shader stage '{values[0]}'; expected one of: "
                    f"{valid_stages}"
                )

            if requested_stage != stage_name:
                raise ValueError(
                    f"Function metadata '{attr_display}' on {context} conflicts "
                    f"with {stage_name} stage"
                )


def validate_tessellation_patch_parameter_metadata(stage):
    """Validate patch parameter placement and control-point counts."""
    stage_name = _normalized_metadata_name(getattr(stage, "stage", None))
    function = getattr(stage, "entry_point", None)
    if function is None:
        return

    output_control_points = _stage_output_control_point_count(stage)
    for parameter, patch_type, control_points in _patch_parameters(function):
        context = _stage_entry_function_context(stage, function)
        parameter_name = getattr(parameter, "name", "<anonymous>")
        display_type = _type_debug_name(_node_declared_type(parameter)) or patch_type
        patch_display_type = (
            _stage_interface_type_name(_node_declared_type(parameter)) or display_type
        )

        if patch_type == "inputpatch" and stage_name not in {
            "tessellation_control",
            "tessellation_evaluation",
        }:
            raise ValueError(
                f"Patch type '{display_type}' parameter '{parameter_name}' on "
                f"{context} requires tessellation_control or "
                "tessellation_evaluation stage"
            )

        if patch_type == "outputpatch" and stage_name not in {
            "tessellation_control",
            "tessellation_evaluation",
        }:
            raise ValueError(
                f"Patch type '{display_type}' parameter '{parameter_name}' on "
                f"{context} requires tessellation_control or "
                "tessellation_evaluation stage"
            )

        if (
            patch_type == "outputpatch"
            and stage_name == "tessellation_control"
            and output_control_points is not None
            and control_points is not None
            and output_control_points != control_points
        ):
            raise ValueError(
                "Conflicting tessellation output control-point metadata on "
                f"{context}: {patch_display_type} must match "
                f"outputcontrolpoints({output_control_points}); OutputPatch "
                f"control points={control_points} on parameter '{parameter_name}'"
            )


def validate_tessellation_patch_cross_stage_metadata(shader):
    """Validate tessellation patch sizes agree across control/evaluation stages."""
    stages = _shader_stages_by_name(shader)
    control_stage = stages.get("tessellation_control")
    evaluation_stage = stages.get("tessellation_evaluation")
    if control_stage is None or evaluation_stage is None:
        return

    control_points = _tessellation_control_output_count(control_stage)
    if control_points is None:
        return

    evaluation_function = getattr(evaluation_stage, "entry_point", None)
    for parameter, patch_type, evaluation_points in _patch_parameters(
        evaluation_function
    ):
        if patch_type != "outputpatch" or evaluation_points is None:
            continue
        if control_points == evaluation_points:
            continue
        parameter_name = getattr(parameter, "name", "<anonymous>")
        patch_display_type = _stage_interface_type_name(_node_declared_type(parameter))
        control_patch_type = _tessellation_control_output_patch_type(
            control_stage, control_points
        )
        raise ValueError(
            "Conflicting tessellation patch control-point metadata from "
            "tessellation_control to tessellation_evaluation: "
            f"{patch_display_type} must match tessellation_control output "
            f"{control_patch_type} from outputcontrolpoints({control_points}); "
            f"OutputPatch control points={evaluation_points} on parameter "
            f"'{parameter_name}'"
        )


def _stage_entry_function_context(stage, function):
    stage_name = _normalized_metadata_name(getattr(stage, "stage", None))
    return (
        f"{stage_name or '<unknown>'} stage entry function "
        f"'{getattr(function, 'name', '<anonymous>')}'"
    )


def _stage_function_context(stage, function):
    stage_name = _normalized_metadata_name(getattr(stage, "stage", None))
    function_name = getattr(function, "name", "<anonymous>")
    if function is getattr(stage, "entry_point", None):
        return f"{stage_name or '<unknown>'} stage entry function '{function_name}'"
    return f"{stage_name or '<unknown>'} stage local function '{function_name}'"


def _canonical_shader_stage_attribute_name(name):
    return normalize_stage_name(name)


def _shader_attribute_display(attr):
    attr_name = _normalized_metadata_name(getattr(attr, "name", None)) or "shader"
    values = _attribute_metadata_values(attr)
    if not values:
        return f"@{attr_name}"
    return f"@{attr_name}{_metadata_value_phrase(values)}"


def _validate_function_stage_layout_flag(
    layout_names, direction, canonical_entry, attr_name, attr_value, stage
):
    direction_layout_names = layout_names.get(direction, set())
    for group in STAGE_LAYOUT_EXCLUSIVE_ENTRY_GROUPS:
        if canonical_entry not in group:
            continue
        conflicts = sorted((direction_layout_names & group) - {canonical_entry})
        if conflicts:
            context = _stage_layout_context(stage, direction)
            raise ValueError(
                f"Conflicting stage/function metadata on {context}: "
                f"layout {', '.join(conflicts)} vs {attr_name}({attr_value})"
            )


def _stage_layout_entries_by_direction(stage):
    entries = {}
    for layout in getattr(stage, "layout_qualifiers", []) or []:
        direction = _normalized_layout_direction(getattr(layout, "direction", None))
        for entry in getattr(layout, "entries", []) or []:
            entry_name = _normalized_metadata_name(getattr(entry, "name", None))
            if entry_name:
                entries[(direction, entry_name)] = _stage_layout_entry_value(entry)
    return entries


def _stage_layout_names_by_direction(stage):
    names = {}
    for layout in getattr(stage, "layout_qualifiers", []) or []:
        direction = _normalized_layout_direction(getattr(layout, "direction", None))
        direction_names = names.setdefault(direction, set())
        for entry in getattr(layout, "entries", []) or []:
            entry_name = _normalized_metadata_name(getattr(entry, "name", None))
            if entry_name:
                direction_names.add(entry_name)
    return names


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


def _first_attribute_metadata_value(attr):
    value = _attribute_metadata_value(attr)
    if value == "":
        return None
    return value


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


def _stage_output_control_point_count(stage):
    function = getattr(stage, "entry_point", None)
    for attr in getattr(function, "attributes", []) or []:
        attr_name = _normalized_metadata_name(getattr(attr, "name", None))
        if attr_name != "outputcontrolpoints":
            continue
        attr_value = _attribute_metadata_value(attr)
        if attr_value is not None:
            return attr_value

    layout_entries = _stage_layout_entries_by_direction(stage)
    for entry_name in ("vertices", "outputcontrolpoints"):
        entry_value = layout_entries.get(("out", entry_name))
        if entry_value is not None:
            return entry_value
    return None


def _tessellation_control_output_count(stage):
    control_points = _stage_output_control_point_count(stage)
    if control_points is not None:
        return control_points

    function = getattr(stage, "entry_point", None)
    for _parameter, patch_type, patch_points in _patch_parameters(function):
        if patch_type == "outputpatch" and patch_points is not None:
            return patch_points
    return None


def _tessellation_control_output_patch_type(stage, control_points):
    function = getattr(stage, "entry_point", None)
    for parameter, patch_type, patch_points in _patch_parameters(function):
        if patch_type != "outputpatch":
            continue
        if patch_points is not None and patch_points != control_points:
            continue
        return _stage_interface_type_name(_node_declared_type(parameter))

    return_type = getattr(function, "return_type", None)
    return_type_name = _stage_interface_type_name(return_type)
    if return_type_name and return_type_name != "void":
        return f"OutputPatch<{return_type_name}, {control_points}>"
    return f"OutputPatch<unknown, {control_points}>"


def _patch_parameters(function):
    if function is None:
        return

    for parameter in getattr(function, "parameters", []) or []:
        parameter_type = _node_declared_type(parameter)
        base_name = _type_base_name(parameter_type)
        if base_name is None:
            continue
        patch_type = base_name.lower()
        if patch_type not in {"inputpatch", "outputpatch"}:
            continue
        yield parameter, patch_type, _patch_control_point_count(parameter_type)


def _patch_control_point_count(type_node):
    type_node = _nested_resource_type(type_node)
    generic_args = getattr(type_node, "generic_args", []) or []
    if len(generic_args) < 2:
        return None
    return expression_debug_name(generic_args[1])


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
        yield struct, _node_context("struct", struct)
        yield from _struct_member_metadata_nodes(struct, "struct")

    for cbuffer in getattr(shader, "cbuffers", []) or []:
        yield cbuffer, _node_context("cbuffer", cbuffer)
        yield from _struct_member_metadata_nodes(cbuffer, "cbuffer")

    for function in getattr(shader, "functions", []) or []:
        yield from _function_metadata_nodes(function)

    for stage in getattr(shader, "stages", {}).values():
        for variable in getattr(stage, "local_variables", []) or []:
            yield variable, _node_context("stage variable", variable)
        for struct in getattr(stage, "local_structs", []) or []:
            yield struct, _node_context("stage struct", struct)
            yield from _struct_member_metadata_nodes(struct, "stage struct")
        for cbuffer in getattr(stage, "local_cbuffers", []) or []:
            yield cbuffer, _node_context("stage cbuffer", cbuffer)
            yield from _struct_member_metadata_nodes(cbuffer, "stage cbuffer")
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
