"""glam's spellings of the generator's vector and matrix types.

The generator names a vector ``Vec3<f32>`` and a matrix ``Mat4<f32>`` while it works out types. rust-gpu shaders
use glam (re-exported as ``spirv_std::glam``), which has one type per element type and size: ``Vec3``, ``DVec3``,
``IVec3``, ``UVec3`` and ``BVec3`` for vectors and the square ``Mat2``-``Mat4`` and ``DMat2``-``DMat4`` for matrices.
spirv-std 0.11 enables glam's ``f64``, ``i32`` and ``u32`` features only, so a vector of any other element type, and
a matrix that is not square or not of floats, has no glam type and is refused.
"""

from __future__ import annotations

import re

from ..refusals import RustRefusal

# glam names a matrix's columns x_axis, y_axis, z_axis and w_axis.
GLAM_AXES = ("x_axis", "y_axis", "z_axis", "w_axis")
_VECTOR_PREFIXES = {"f32": "", "f64": "D", "i32": "I", "u32": "U", "bool": "B"}
_MATRIX_PREFIXES = {"f32": "", "f64": "D"}
# ``Vec3<f32>`` and ``Vec3::<f32>::new``, the type argument a scalar, after a character that cannot continue a path
# segment, so ``MyVec3<f32>`` and ``glam::Vec3`` paths are left alone.
_GENERIC = re.compile(
    r"(?<![\w:])(?P<kind>Vec|Mat)(?P<shape>[234](?:x[234])?)(?P<turbofish>::)?<(?P<scalar>\w+)>"
)


def glam_spelling(code: str) -> str:
    """``code`` with every generator vector and matrix type written as glam's type; refuses one glam lacks."""

    def replace(match: re.Match[str]) -> str:
        kind, shape, scalar = match["kind"], match["shape"], match["scalar"]
        if kind == "Vec":
            prefix = _VECTOR_PREFIXES.get(scalar)
            if prefix is None or "x" in shape:
                raise RustRefusal(
                    f"type: spirv-std's glam has no vector of {scalar} (Vec{shape}<{scalar}>); rust-gpu shaders "
                    "have f32, f64, i32, u32 and bool vectors"
                )
        else:
            prefix = _MATRIX_PREFIXES.get(scalar)
            if prefix is None or "x" in shape:
                raise RustRefusal(
                    f"type: glam has no {shape} matrix of {scalar} (Mat{shape}<{scalar}>); rust-gpu shaders have "
                    "square f32 and f64 matrices"
                )
        return f"{prefix}{kind}{shape}"

    return _GENERIC.sub(replace, code)
