"""Workgroup variables that are not arrays, in Rust output.

rust-gpu declares workgroup memory as an entry-point parameter ``#[spirv(workgroup)] name: &mut T``
(:mod:`.rust_gpu_interface`), and the emitter threads the same reference to every helper that uses the variable
(:func:`.rust_resource_bindings.resource_parameters`). Indexing a workgroup array dereferences implicitly
(``name[i]``), but any other use of a workgroup scalar, vector, matrix or structure needs an explicit dereference:
``s.v + a0`` with ``a0: &mut f32`` has no ``Add<&mut f32>`` implementation, and ``a0 = 1.0`` rebinds the parameter
instead of writing the shared memory every invocation of the workgroup reads.

Inside a function that holds such a reference the emitter therefore types the variable as ``T``, writes it as the
place ``(*name)`` in every value and assignment position, and passes the reference itself (``name``) to the
helpers it is threaded to, which Rust reborrows.
"""

from __future__ import annotations

from .rust_gpu_geometry import STAGE_OUTPUT_KIND
from .rust_gpu_interface import WORKGROUP_KIND

__all__ = ["RustWorkgroupReferenceMixin", "workgroup_reference_pointee"]

_REFERENCE_PREFIX = "&mut "
# Workgroup variables and geometry outputs (:mod:`.rust_gpu_geometry`) are both ``&mut T`` entry parameters.
_REFERENCE_KINDS = frozenset({WORKGROUP_KIND, STAGE_OUTPUT_KIND})


def workgroup_reference_pointee(type_name):
    """The CrossGL type a non-array workgroup reference type ``&mut T`` holds, else ``None``."""
    text = str(type_name or "").strip()
    if not text.startswith(_REFERENCE_PREFIX):
        return None
    pointee = text[len(_REFERENCE_PREFIX) :].strip()
    if not pointee or pointee.endswith("]"):
        return None
    return pointee


class RustWorkgroupReferenceMixin:
    """Dereferences workgroup scalars, vectors, matrices and structures that a function holds by reference."""

    def bind_rust_workgroup_references(self, names):
        """Treat the workgroup references among the parameter ``names`` as places in the function being generated.

        Each one leaves :attr:`current_mutated_names`, since the function writes through the reference and never
        rebinds it; :meth:`retype_rust_workgroup_references` types it as the value it refers to once the
        parameters are written. Returns the previous bindings, which the caller restores with
        :meth:`restore_rust_workgroup_references`.
        """
        previous = dict(getattr(self, "current_rust_workgroup_references", {}) or {})
        infos = getattr(self, "rust_entry_resource_infos", None) or {}
        references = {}
        for name in names:
            info = infos.get(name)
            if not info or info.get("kind") not in _REFERENCE_KINDS:
                continue
            pointee = workgroup_reference_pointee(self.variable_types.get(name))
            if pointee is None:
                continue
            references[name] = pointee
        self.current_rust_workgroup_references = references
        mutated = getattr(self, "current_mutated_names", None)
        if references and mutated:
            self.current_mutated_names = set(mutated) - set(references)
        return previous

    def retype_rust_workgroup_references(self):
        for name, pointee in sorted(self.current_rust_workgroup_references.items()):
            self.variable_types[name] = pointee

    def restore_rust_workgroup_references(self, previous):
        self.current_rust_workgroup_references = previous

    def is_rust_workgroup_reference(self, name):
        """Whether ``name`` is a workgroup reference of the function being generated, not shadowed by a local."""
        references = getattr(self, "current_rust_workgroup_references", None) or {}
        pointee = references.get(name)
        return pointee is not None and self.variable_types.get(name) == pointee

    def rust_workgroup_reference_place(self, name):
        """``(*name)`` for a workgroup reference of the function being generated, else ``None``."""
        if not self.is_rust_workgroup_reference(name):
            return None
        return f"(*{self.rust_identifier(name)})"
