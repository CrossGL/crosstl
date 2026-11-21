"""Token and name tables of the CrossGL parser."""

from ..wave_operations import (
    WAVE_EXTENDED_OPERATIONS,
    WAVE_PARTITIONED_OPERATIONS,
    WAVE_ROTATE_OPERATIONS,
)

WAVE_INTRINSICS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveIsFirstLane",
    "WaveActiveSum",
    "WaveActiveProduct",
    "WaveActiveBitAnd",
    "WaveActiveBitOr",
    "WaveActiveBitXor",
    "WaveActiveMin",
    "WaveActiveMax",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveAllEqual",
    "WaveActiveBallot",
    "WaveActiveCountBits",
    "WaveReadLaneAt",
    "WaveReadLaneFirst",
    "WaveShuffleDown",
    "WaveShuffleUp",
    "WaveShuffleAndFillUp",
    "WaveShuffleXor",
    "WavePrefixSum",
    "WavePrefixProduct",
    "WavePrefixInclusiveSum",
    "WavePrefixInclusiveProduct",
    "WavePrefixCountBits",
    "QuadActiveSum",
    "QuadReadAcrossX",
    "QuadReadAcrossY",
    "QuadReadAcrossDiagonal",
    "QuadReadLaneAt",
    "QuadAny",
    "QuadAll",
    "WaveMatch",
    "WaveMultiPrefixSum",
    "WaveMultiPrefixCountBits",
    "WaveMultiPrefixProduct",
    "WaveMultiPrefixBitAnd",
    "WaveMultiPrefixBitOr",
    "WaveMultiPrefixBitXor",
    *sorted(WAVE_EXTENDED_OPERATIONS),
    *sorted(WAVE_PARTITIONED_OPERATIONS),
    *sorted(WAVE_ROTATE_OPERATIONS),
}
COOPERATIVE_MATRIX_INTRINSICS = {
    "cooperative_matrix_element": "element",
    "cooperative_matrix_fill": "fill",
    "cooperative_matrix_diagonal": "diagonal",
    "cooperative_matrix_load": "load",
    "cooperative_matrix_store": "store",
    "cooperative_matrix_multiply": "multiply",
    "cooperative_matrix_multiply_accumulate": "multiply_accumulate",
    "cooperative_matrix_add": "elementwise_add",
    "cooperative_matrix_subtract": "elementwise_subtract",
    "cooperative_matrix_elementwise_multiply": "elementwise_multiply",
    "cooperative_matrix_negate": "negate",
}
RAYTRACING_INTRINSICS = {
    "TraceRay",
    "ReportHit",
    "CallShader",
    "AcceptHitAndEndSearch",
    "IgnoreHit",
}
MESH_INTRINSICS = {
    "SetMeshOutputCounts",
    "DispatchMesh",
}
# Generic vector types whose braced initializer lists their components, as a constructor call does.
_GENERIC_VECTOR_TYPE_NAMES = frozenset({"vec", "vector"})
VARIABLE_QUALIFIER_TOKEN_TYPES = frozenset(
    {
        "CONST",
        "STATIC",
        "MUT",
        "SHARED",
        "UNIFORM",
        "BUFFER",
        "IN",
        "PRIVATE",
        "GLOBAL",
        "LOCAL",
        "THREADGROUP_IMAGEBLOCK",
        "THREADGROUP",
        "WORKGROUP",
    }
)
PARAMETER_QUALIFIER_TOKEN_TYPES = VARIABLE_QUALIFIER_TOKEN_TYPES - {"MUT"}
RESOURCE_MEMORY_QUALIFIER_NAMES = frozenset({"coherent", "volatile"})
RESOURCE_ADDRESS_SPACE_NAMES = frozenset(
    {
        "constant",
        "device",
        "global",
        "local",
        "private",
        "storage",
        "thread",
        "threadgroup",
        "threadgroup_imageblock",
        "workgroup",
    }
)
RESOURCE_ACCESS_MODE_NAMES = frozenset(
    {
        "read",
        "readonly",
        "read_write",
        "readwrite",
        "write",
        "writeonly",
    }
)
VARIABLE_QUALIFIER_NAMES = frozenset(
    {
        "out",
        "input",
        "output",
        "inout",
        "patch",
        "flat",
        "smooth",
        "noperspective",
        "centroid",
        "sample",
        "pervertex",
        "perprimitive",
        "perview",
        "readonly",
        "writeonly",
        "readwrite",
        "coherent",
        "volatile",
        "restrict",
        "device",
        "constant",
        "thread",
        "threadgroup_imageblock",
        "threadgroup",
        "workgroup",
        "storage",
        "private",
        "function",
        "groupshared",
        "row_major",
        "column_major",
        "precise",
        "nointerpolation",
        "linear",
        "linear_centroid",
        "linear_noperspective",
        "linear_noperspective_centroid",
        "linear_sample",
    }
)
PARAMETER_PRIMITIVE_QUALIFIER_NAMES = frozenset(
    {
        "point",
        "line",
        "triangle",
        "lineadj",
        "triangleadj",
    }
)
TEXTURE_TYPE_NAMES = {
    "TEXTURE1D": "texture1D",
    "TEXTURE2D": "texture2D",
    "TEXTURE3D": "texture3D",
    "TEXTURECUBE": "textureCube",
    "TEXTURE2DARRAY": "texture2DArray",
}
RAYQUERY_METHODS = {
    "Proceed",
    "Abort",
    "CandidateType",
    "CommittedStatus",
    "CandidatePrimitiveIndex",
    "CommittedPrimitiveIndex",
    "CandidateInstanceID",
    "CommittedInstanceID",
    "CandidateGeometryIndex",
    "CommittedGeometryIndex",
    "CandidateObjectRayOrigin",
    "CandidateObjectRayDirection",
    "CommittedObjectRayOrigin",
    "CommittedObjectRayDirection",
    "CommittedRayT",
    "CandidateTriangleRayT",
    "TraceRayInline",
}
SHADER_STAGE_TOKEN_TYPES = frozenset(
    {
        "VERTEX",
        "FRAGMENT",
        "COMPUTE",
        "GEOMETRY",
        "TESSELLATION_CONTROL",
        "TESSELLATION_EVALUATION",
        "TASK",
        "MESH",
        "OBJECT",
        "AMPLIFICATION",
        "RAY_GENERATION",
        "RAY_INTERSECTION",
        "RAY_CLOSEST_HIT",
        "RAY_MISS",
        "RAY_ANY_HIT",
        "RAY_CALLABLE",
    }
)

# How a cooperative-matrix contract mismatch names each field it compares.
COOPERATIVE_MATRIX_FIELD_LABELS = {
    "element_type": "component type",
    "rows": "row count",
    "cols": "column count",
    "scope": "scope",
    "use": "use",
    "layout": "layout",
    "subgroup_size": "subgroup size",
    "elements_per_lane": "elements-per-lane count",
    "fragment_layout": "fragment layout",
    "fragment_provenance": "fragment provenance",
    "fragment_mapping": "fragment mapping",
    "fragment_mapping_provenance": "fragment mapping provenance",
}
# Scalar spellings a vector generic argument may use for the CrossGL scalars ``float``, ``double``, ``int``, ``uint``.
WGSL_STYLE_SCALAR_SPELLINGS = {
    "f32": "float",
    "f64": "double",
    "i32": "int",
    "u32": "uint",
}
