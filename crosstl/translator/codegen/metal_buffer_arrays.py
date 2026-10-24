"""Arrays of buffers as Metal entry-point arguments.

HLSL ``StructuredBuffer<T> b[N] : register(tK)``, a GLSL storage-block array ``buffer B { ... } b[N]`` and their
CrossGL forms bind ``N`` buffers at ``N`` consecutive slots. A Metal entry point cannot declare them as one
argument: the ``buffer`` attribute takes a pointer or reference to device or constant memory, and an array of
buffers exists only inside an argument buffer, so ``array<device T*, N> b [[buffer(K)]]`` is rejected ("type
'array<device T *, N>' is not valid for attribute 'buffer'").

:func:`expand_entry_buffer_array` declares one pointer argument per element instead, at slots ``K`` to
``K + N - 1``, which are the slots the emitter already reserves for the array. The entry point then rebuilds the
array as a local ``array<device T*, N>`` named like the resource, so indexing, buffer lengths and helper-function
calls keep their form. The length and counter side buffers of such an array are expanded in the same way, one
side buffer per element.

An array declared without a size takes the extent the generator infers from its constant indices, as the
other resource arrays do. An array whose size is not a compile-time constant has no fixed slot range; Metal could
only bind it through an argument buffer that the host encodes, so it is refused.
"""

from __future__ import annotations

import re

__all__ = ["expand_entry_buffer_array"]

_ARRAY_DECLARATION = re.compile(
    r"array<(?P<element>[^<>]+\*), (?P<size>[^<>,]+)> (?P<name>[A-Za-z_]\w*)"
)


def expand_entry_buffer_array(codegen, declaration, slot, prologue):
    """The ``[[buffer]]`` parameter declarations for one entry-point buffer argument at ``slot``.

    ``declaration`` is the argument as the emitter spells it. A pointer argument is returned unchanged with its
    attribute. An ``array<P, N> name`` argument becomes ``N`` arguments ``P name_k [[buffer(slot + k)]]``, and the
    statement that rebuilds ``name`` from them is appended to ``prologue``.
    """
    match = _ARRAY_DECLARATION.fullmatch(declaration)
    if match is None:
        return [f"{declaration} [[buffer({slot})]]"]
    element, name = match["element"], match["name"]
    size_text = match["size"].strip()
    count = codegen.literal_int_value(size_text, codegen.literal_int_constants)
    if count is None or count < 1:
        codegen.refuse_lowering(
            f"array of buffers '{name}' has size '{size_text}', which is not a positive constant: a Metal "
            "entry point binds a fixed number of buffers, and a variable number needs an argument buffer "
            "encoded by the host"
        )
        return [f"{element} {name} [[buffer({slot})]]"]
    taken = getattr(codegen, "metal_entry_buffer_array_names", None)
    if taken is None:
        taken = codegen.metal_entry_buffer_array_names = set()
    names = []
    for index in range(count):
        candidate = codegen.metal_generated_companion_name(f"{name}_{index}")
        while candidate in taken:
            candidate = f"{candidate}_"
        taken.add(candidate)
        names.append(candidate)
    prologue.append(f"array<{element}, {count}> {name} = {{{', '.join(names)}}};")
    return [
        f"{element} {element_name} [[buffer({slot + index})]]"
        for index, element_name in enumerate(names)
    ]
