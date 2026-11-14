"""Metal ``constant floatCxR &m [[buffer(n)]]`` entry parameters as HLSL constant buffers.

A Metal entry parameter that references one constant value becomes an HLSL ``cbuffer`` holding that value. Matrices
were missing from that rule, so ``constant float4x4 &transform`` became a per-vertex input (``float4x4 transform :
TEXCOORD0``), which reads a vertex attribute instead of the bound matrix.

HLSL packs a matrix in a constant buffer column by column (``column_major`` is the default), starting every column
on a new 16-byte register. MSL stores the columns of a ``floatCx3`` or ``floatCx4`` matrix 16 bytes apart too, so
those keep their layout. A ``floatCx2`` matrix has 8-byte columns in MSL, and a ``half`` matrix 4- or 8-byte
columns, so the HLSL constant buffer would read every column after the first from the wrong offset; those are
refused with :class:`DirectXConstantMatrixLayoutError`.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["DirectXConstantMatrixLayoutError", "hlsl_constant_matrix_member_type"]

_COLUMN_STRIDE_COMPONENTS = frozenset({"float"})
_REGISTER_ROWS = frozenset({3, 4})


class DirectXConstantMatrixLayoutError(TranslationDiagnostic):
    """A constant matrix parameter whose MSL column stride differs from the HLSL constant buffer's 16 bytes."""

    def __init__(self, parameter, hlsl_type, reason):
        self.missing_capabilities = ("directx.constant-matrix-column-stride",)
        super().__init__(
            f"DirectX cannot bind constant matrix parameter '{parameter}' ({hlsl_type}) as a constant buffer: "
            f"{reason}, but HLSL starts every matrix column on a new 16-byte register",
            target="directx",
            code=f"{CODE_PREFIX}directx-constant-matrix-layout-unsupported",
        )


def hlsl_constant_matrix_member_type(codegen, element_type, parameter_name):
    """The HLSL member type of a constant matrix parameter, ``None`` for a non-matrix, or a typed refusal."""
    dimensions = codegen.hlsl_matrix_dimensions(element_type)
    if dimensions is None:
        return None
    component, rows, _columns = dimensions
    hlsl_type = codegen.map_type(element_type)
    if component not in _COLUMN_STRIDE_COMPONENTS:
        raise DirectXConstantMatrixLayoutError(
            parameter_name,
            hlsl_type,
            f"MSL packs the {rows}-row columns of a {component} matrix "
            f"{2 * (4 if rows > 2 else 2)} bytes apart",
        )
    if rows not in _REGISTER_ROWS:
        raise DirectXConstantMatrixLayoutError(
            parameter_name, hlsl_type, "MSL packs 2-row float columns 8 bytes apart"
        )
    return hlsl_type
