"""The vector and matrix typedefs of ``<simd/simd.h>``, which Metal shaders share with their host code.

A header that a Metal shader and its Objective-C, Swift or C++ host both include declares its structures with the
simd module's typedefs, ``vector_floatN``, ``vector_intN``, ``vector_uintN`` and ``matrix_floatNxM``, because the
host compiler does not know MSL's ``floatN``. In MSL each one names the same type as the MSL spelling
(``vector_float4`` is ``float4``, and ``matrix_float4x4`` is ``float4x4``, N columns of M rows). The shader uses
them both as declaration types and as constructors (``vector_float4(0.0, 0.0, 0.0, 1.0)``).
"""

from __future__ import annotations

__all__ = ["SIMD_HEADER_TYPES"]

_VECTORS = {"float": "vec", "int": "ivec", "uint": "uvec"}

SIMD_HEADER_TYPES = {
    **{
        f"vector_{scalar}{size}": f"{prefix}{size}"
        for scalar, prefix in _VECTORS.items()
        for size in (2, 3, 4)
    },
    # matrix_floatCxR has C columns of R rows, as CrossGL's matCxR does.
    **{
        f"matrix_float{columns}x{rows}": (
            f"mat{columns}" if columns == rows else f"mat{columns}x{rows}"
        )
        for columns in (2, 3, 4)
        for rows in (2, 3, 4)
    },
}
