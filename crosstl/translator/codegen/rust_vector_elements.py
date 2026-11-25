"""Vector components and matrix columns selected by an index, without glam's ``Index`` and ``IndexMut``.

glam's ``Index``/``IndexMut`` for ``Vec2``-``Vec4`` and the ``col_mut`` of ``Mat2``-``Mat4`` select a reference to
one component with a ``match``. rust-gpu writes that selection as an ``OpPhi`` of ``Function``-storage pointers, which
spirv-val rejects for Vulkan: variable pointers cover only the ``StorageBuffer`` and ``Workgroup`` storage classes.
So the generator never indexes a glam vector or a matrix's columns:

- a constant index names the component or column, ``v.y`` or ``m.z_axis``, which is a place, so stores work too;
- a vector read through a dynamic index is spirv-std's ``vector_extract_dynamic`` (``OpVectorExtractDynamic``), a
  bool vector's is glam's ``BVecN::test`` and a matrix column's is ``Mat::col``; all three are values;
- a store, compound assignment or increment through a dynamic index is a ``match`` on the index with one arm per
  component, each writing through the constant-index place. An index past the last component writes nothing: the
  write is undefined behaviour in CrossGL's source languages (GLSL 4.60 section 5.5, HLSL and MSL alike).
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, LiteralNode, MemberAccessNode
from .rust_gpu_spelling import GLAM_AXES

VECTOR_COMPONENTS = ("x", "y", "z", "w")


def constant_component_index(index_expr):
    """The non-negative integer ``index_expr`` writes literally, else ``None``."""
    if isinstance(index_expr, LiteralNode):
        value = index_expr.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            text = value.strip().rstrip("uU")
            return int(text) if text.isdigit() else None
        return None
    if isinstance(index_expr, bool):
        return None
    if isinstance(index_expr, int):
        return index_expr if index_expr >= 0 else None
    if isinstance(index_expr, str) and index_expr.isdigit():
        return int(index_expr)
    return None


class RustVectorElementMixin:
    """Index vectors and matrix columns through component names, spirv-std extraction and ``match`` stores."""

    def component_index_overrides(self):
        """The constant each enclosing ``match`` arm gives a dynamically indexed access, by the access's id."""
        return self.__dict__.setdefault("rust_component_index_overrides", {})

    def indexed_component_count(self, array_expr):
        """``(kind, count)`` for an index into a vector (``"vector"``/``"bool"``) or a matrix's columns, else
        ``None``."""
        array_type = self.expression_result_type(array_expr)
        matrix = self.matrix_type_info(array_type)
        if matrix is not None:
            return "matrix", matrix["columns"]
        vector = self.vector_type_info(array_type)
        if vector is not None:
            kind = "bool" if vector["component_type"] == "bool" else "vector"
            return kind, vector["size"]
        return None

    def component_access_index(self, expr):
        """The constant index ``expr`` selects, through its literal or its enclosing ``match`` arm, else ``None``."""
        overrides = self.component_index_overrides()
        if id(expr) in overrides:
            return overrides[id(expr)]
        index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
        return constant_component_index(index_expr)

    def generate_component_access(self, expr, array_expr, array):
        """Rust for ``array[index]`` when ``array`` is a vector or a matrix, else ``None``."""
        shape = self.indexed_component_count(array_expr)
        if shape is None:
            return None
        kind, count = shape
        names = GLAM_AXES if kind == "matrix" else VECTOR_COMPONENTS
        index = self.component_access_index(expr)
        if index is not None:
            if index >= count:
                return self.refuse_lowering(
                    f"index {index} is past the last of the {count} components of "
                    f"{self.map_type(self.expression_result_type(array_expr))}",
                    f"{array}.{names[0]}",
                )
            return f"{array}.{names[index]}"
        index_expr = getattr(expr, "index_expr", getattr(expr, "index", None))
        dynamic = self.generate_array_index_expression(index_expr)
        if self.assignment_lhs_depth > 0:
            # Assignments, compound assignments and increments expand the index into a match before they reach
            # here; any other place (an inout argument) would need a reference selected at run time.
            return self.refuse_lowering(
                "a vector component or matrix column selected by a run-time index as a reference: rust-gpu "
                "selects it with an OpPhi of Function pointers, which Vulkan SPIR-V does not allow",
                f"{array}.{names[0]}",
            )
        if kind == "matrix":
            return f"{array}.col({dynamic})"
        if kind == "bool":
            return f"{array}.test({dynamic})"
        return (
            f"unsafe {{ spirv_std::arch::vector_extract_dynamic({array}, {dynamic}) }}"
        )

    def dynamic_component_access(self, target):
        """The access nearest the root of the place ``target`` that indexes a vector or matrix columns with an index
        that is neither a literal nor fixed by an enclosing ``match`` arm, else ``None``.
        """
        found = None
        node = target
        while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
            if isinstance(node, ArrayAccessNode):
                array_expr = getattr(node, "array_expr", getattr(node, "array", None))
                if (
                    self.component_access_index(node) is None
                    and self.indexed_component_count(array_expr) is not None
                ):
                    found = node
                node = array_expr
            else:
                node = getattr(node, "object_expr", getattr(node, "object", None))
        return found

    def expand_dynamic_component_write(self, target, write):
        """A ``match`` on the first run-time component index of the place ``target`` whose arms are ``write()``, the
        Rust of the store, with that index fixed; ``None`` when the place has no such index.

        ``write`` re-enters this expansion, so each further run-time index of the place nests one more ``match``.
        """
        access = self.dynamic_component_access(target)
        if access is None:
            return None
        array_expr = getattr(access, "array_expr", getattr(access, "array", None))
        _, count = self.indexed_component_count(array_expr)
        index_expr = getattr(access, "index_expr", getattr(access, "index", None))
        index = self.generate_array_index_expression(index_expr)
        overrides = self.component_index_overrides()
        arms = []
        for component in range(count):
            overrides[id(access)] = component
            try:
                arms.append(f"{component} => {{ {write()}; }}")
            finally:
                del overrides[id(access)]
        return f"match {index} {{ {' '.join(arms)} _ => {{}} }}"
