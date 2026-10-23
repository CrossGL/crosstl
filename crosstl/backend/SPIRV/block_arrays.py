"""Arrays of SPIR-V storage blocks: the declaration and the accesses the importer writes for them.

GLSL declares an array of storage blocks as ``buffer B { ... } bufs[N]`` (GLSL 4.60 section 4.3.9). In SPIR-V the
variable points to an ``OpTypeArray`` of the ``Block``-decorated struct, or to an ``OpTypeRuntimeArray`` of it with
descriptor indexing, and every access chain into it starts with the element index (SPIR-V 1.6 section 2.16.2; Vulkan
1.3 section 15.5.3, "DescriptorSet and Binding Assignment"). The importer looked for the struct directly behind the
pointer, so it dropped the declaration of such an array and wrote ``bufs[k].data[i]`` against an undeclared name.

The importer now declares the array as ``RWStructuredBuffer<B> bufs[N]`` (``StructuredBuffer`` when it is read-only)
with the variable's set and binding. An access is ``bufs[k][0].m...``, the single-block form ``v[0].m...`` applied to
element ``k``, and ``OpArrayLength`` on element ``k`` is ``spirvArrayLength(bufs[k], m)``. The SPIR-V storage block
pass (:mod:`crosstl.translator.semantics.storage_blocks`) rewrites both into the forms the GLSL importer produces.

An array of uniform blocks is refused with :class:`SpirvBlockArrayUnsupportedError`. CrossGL keeps a uniform block's
layout rule on a ``cbuffer``, which has no array form, and a plain struct array would take each target's default
layout instead of the module's ``Offset`` and ``ArrayStride`` decorations.
"""

from __future__ import annotations

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "SpirvBlockArrayUnsupportedError",
    "block_array_element",
]


class SpirvBlockArrayUnsupportedError(TranslationDiagnostic):
    """An array of SPIR-V blocks that CrossGL has no declaration for."""

    def __init__(self, variable, storage_class):
        self.missing_capabilities = ("spirv.import.block_array",)
        super().__init__(
            f"SPIR-V variable '{variable}' is an array of {storage_class} blocks; CrossGL declares arrays of "
            "storage blocks only (StorageBuffer, or Uniform with BufferBlock), since a uniform block keeps its "
            "layout only on a single cbuffer",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-block-array-unsupported",
        )


def block_array_element(type_id, types):
    """``(struct type id, length id)`` when ``type_id`` is an array of structs, else ``None``.

    The length id is ``None`` for a runtime array.
    """
    info = types.get(type_id) or {}
    kind = info.get("kind")
    if kind not in ("array", "runtime_array"):
        return None
    element = info.get("element_type")
    if (types.get(element) or {}).get("kind") != "struct":
        return None
    return element, info.get("length_id") if kind == "array" else None
