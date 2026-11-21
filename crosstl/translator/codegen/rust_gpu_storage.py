"""Storage-buffer slices of 3-vectors, which rust-gpu stores as 4-vectors.

A CrossGL storage buffer of 3-vectors (``buffer vec3* data;``) has the std430 array stride of 16 bytes, and Vulkan's
storage buffer layout rules require it. glam's ``Vec3`` is 12 bytes, so a ``&mut [Vec3]`` slice would have a 12-byte
stride, which spirv-val rejects. The slice is written as ``&mut [Vec4]`` (``IVec4``, ``UVec4``, ``DVec4``) instead:
an element read is the 4-vector's ``truncate()``, a store writes ``extend(0)`` into the padding lane, and a component
or swizzle of an element addresses the 4-vector directly.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, IdentifierNode

# The 3-vector element types of a padded slice, with the 4-vector type and the padding lane's zero.
PADDED_ELEMENTS = {
    "vec3": ("Vec4", "0.0"),
    "ivec3": ("IVec4", "0"),
    "uvec3": ("UVec4", "0"),
    "dvec3": ("DVec4", "0.0"),
}
_COMPOUND = {"+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%"}


def padded_slice_element(type_name):
    """The padded element of a storage slice type ``&mut vec3[]``, else ``None``."""
    text = str(type_name or "").strip()
    for prefix in ("&mut ", "&"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    else:
        return None
    if not text.endswith("[]"):
        return None
    return PADDED_ELEMENTS.get(text[:-2].strip())


class RustGpuPaddedSliceMixin:
    """Reads and stores of padded storage slices for ``RustCodeGen``."""

    def rust_gpu_padded_element(self, array_expr):
        if not isinstance(array_expr, IdentifierNode):
            return None
        return padded_slice_element(self.variable_types.get(array_expr.name))

    def rust_gpu_padded_slice_read(self, array_expr, array, index):
        """``array[index]`` as the 3-vector it holds, outside an assignment target."""
        if self.assignment_lhs_depth > 0 or self.rust_gpu_padded_element(array_expr) is None:
            return None
        return f"{array}[{index}].truncate()"

    def generate_rust_gpu_padded_slice_assignment(self, target, value, operator):
        """A store of a whole element of a padded slice, or ``None`` for any other assignment."""
        if not isinstance(target, ArrayAccessNode):
            return None
        array_expr = getattr(target, "array_expr", None)
        padded = self.rust_gpu_padded_element(array_expr)
        if padded is None:
            return None
        _vector, zero = padded
        array = self.rust_identifier(array_expr.name)
        index = self.generate_array_index_expression(getattr(target, "index_expr", None))
        element_type = self.expression_result_type(target)
        rhs = self.generate_expression_with_type(value, element_type)
        if operator == "=":
            return f"{array}[{index}] = ({rhs}).extend({zero})"
        binary = _COMPOUND.get(operator)
        if binary is None:
            return self.refuse_lowering(
                f"assignment: {operator} on an element of the 3-vector storage buffer '{array_expr.name}'"
            )
        return (
            f"{{ let __index = {index}; {array}[__index] = "
            f"({array}[__index].truncate() {binary} ({rhs})).extend({zero}); }}"
        )
