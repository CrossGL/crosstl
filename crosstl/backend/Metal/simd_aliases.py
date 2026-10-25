"""The ``<simd/simd.h>`` aliases of Metal's vector and matrix types.

Apple's SIMD library names the Metal vector and matrix types with C aliases that both Metal and host code share:
``simd_float2`` and ``vector_float2`` are ``float2``, and ``simd_float4x3`` and ``matrix_float4x3`` are
``float4x3`` (four columns of three rows, as in Metal); its C++ interface spells them ``simd::float2`` and
``simd::float4x3``. Metal shaders use them as types and as constructors (``vector_float2(viewport)``). :data:`SIMD_TYPE_ALIASES` maps each alias to its Metal type, so the importer maps it
like that type wherever it appears.
"""

_VECTOR_ELEMENTS = ("float", "half", "int", "uint", "short", "ushort", "char", "uchar")
_MATRIX_ELEMENTS = ("float", "half")
_SIZES = (2, 3, 4)

SIMD_TYPE_ALIASES = {
    **{
        f"{prefix}_{element}{size}": f"{element}{size}"
        for prefix in ("simd", "vector")
        for element in _VECTOR_ELEMENTS
        for size in _SIZES
    },
    **{
        f"{prefix}_{element}{columns}x{rows}": f"{element}{columns}x{rows}"
        for prefix in ("simd", "matrix")
        for element in _MATRIX_ELEMENTS
        for columns in _SIZES
        for rows in _SIZES
    },
    **{
        f"simd::{element}{size}": f"{element}{size}"
        for element in _VECTOR_ELEMENTS
        for size in _SIZES
    },
    **{
        f"simd::{element}{columns}x{rows}": f"{element}{columns}x{rows}"
        for element in _MATRIX_ELEMENTS
        for columns in _SIZES
        for rows in _SIZES
    },
}
