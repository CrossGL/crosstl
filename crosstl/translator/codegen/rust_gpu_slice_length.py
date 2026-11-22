"""``length()`` of a runtime-sized storage buffer as the length of its rust-gpu slice.

A runtime-sized CrossGL buffer (``buffer float* data;``, ``RWStructuredBuffer<float>``) is an entry-point slice
parameter in rust-gpu (``&mut [f32]``), and ``data.length()`` is its element count, an ``int`` as in GLSL (GLSL 4.60
section 5.9). rust-gpu lowers ``slice.len()`` to ``OpArrayLength``, so the call is ``(data.len() as i32)``. Sized
arrays, vectors and matrices never reach this: :mod:`..semantics.array_length` folds their ``length()`` to a
constant first.
"""

from __future__ import annotations

from ..ast import MemberAccessNode


def _slice_length_receiver(func_expr, args):
    if not isinstance(func_expr, MemberAccessNode) or func_expr.member != "length":
        return None
    if args:
        return None
    return getattr(func_expr, "object_expr", getattr(func_expr, "object", None))


class RustGpuSliceLengthMixin:
    """Write and type ``length()`` calls on rust-gpu storage slices."""

    def is_rust_gpu_slice_type(self, type_name):
        return isinstance(type_name, str) and type_name.replace(" ", "").endswith("[]")

    def rust_gpu_slice_length_result_type(self, func_expr, args):
        receiver = _slice_length_receiver(func_expr, args)
        if receiver is None:
            return None
        if not self.is_rust_gpu_slice_type(self.expression_result_type(receiver)):
            return None
        return "int"

    def generate_rust_gpu_slice_length(self, func_expr, args):
        receiver = _slice_length_receiver(func_expr, args)
        if receiver is None:
            return None
        if not self.is_rust_gpu_slice_type(self.expression_result_type(receiver)):
            return None
        # As the object of a member access the slice is borrowed, not cloned.
        self.member_object_depth += 1
        try:
            slice_expr = self.generate_expression(receiver)
        finally:
            self.member_object_depth -= 1
        return f"({slice_expr}.len() as i32)"
