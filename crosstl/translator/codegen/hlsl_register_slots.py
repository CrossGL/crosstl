"""One slot sequence for the implicit registers of HLSL buffers, textures and constant buffers.

HLSL numbers registers per class: ``t`` for shader resource views (read-only buffers and textures), ``u`` for
unordered access views (writable buffers and images), ``b`` for constant buffers and ``s`` for samplers. A resource
whose source gives no binding takes the next free register. Counting each class from 0 gave ``readonly buffer fin;
buffer outp;`` the registers ``t0`` and ``u0``.

Every other CrossTL target numbers those resources in one sequence, in declaration order: SPIR-V and Vulkan GLSL
bind ``fin`` at 0 and ``outp`` at 1, Metal at ``[[buffer(0)]]`` and ``[[buffer(1)]]``, and an HLSL source imported
from Metal, CUDA or GLSL keeps its explicit slots (``t0``, ``u1``). DXC's SPIR-V back end maps register ``tN``,
``uN`` and ``bN`` of a space to binding N of the matching descriptor set, so ``t0`` and ``u0`` become two
descriptors at binding 0 and the host's binding order is lost.

The implicit ``t``, ``u`` and ``b`` registers therefore come from one cursor (:class:`SharedResourceSlots`), and
the slot an implicit resource takes is free in all three classes, also of the registers a source gives explicitly.
Samplers keep their own sequence: SPIR-V and GLSL combine a texture with its sampler in one binding, which HLSL
splits into a ``Texture`` and a ``SamplerState``.
"""

from __future__ import annotations

__all__ = ["SHARED_SLOT_CLASSES", "SharedResourceSlots", "next_shared_resource_slot"]

SHARED_SLOT_CLASSES = ("t", "u", "b")


class SharedResourceSlots(dict):
    """The implicit-register cursors, by space, that the ``t``, ``u`` and ``b`` classes share."""


def next_shared_resource_slot(allocator, register_prefix, cursors, space, start, count):
    """The first register at or after ``start`` with ``count`` free slots.

    When ``cursors`` is shared and ``register_prefix`` is one of :data:`SHARED_SLOT_CLASSES`, the slots must be free
    in each of those classes; otherwise only in ``register_prefix``.
    """
    if not isinstance(cursors, SharedResourceSlots) or register_prefix not in SHARED_SLOT_CLASSES:
        return allocator.next_available((register_prefix, space), start, count)
    binding = start
    while True:
        candidate = binding
        for prefix in SHARED_SLOT_CLASSES:
            candidate = allocator.next_available((prefix, space), candidate, count)
        if candidate == binding:
            return binding
        binding = candidate
