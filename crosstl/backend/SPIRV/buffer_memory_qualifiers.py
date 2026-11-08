"""Memory qualifiers of a SPIR-V storage block, as the CrossGL attributes of the imported buffer.

GLSL's memory qualifiers on a storage block (``coherent``, ``volatile``, ``restrict``, ``readonly`` and
``writeonly``; GLSL 4.60 section 4.10) become the SPIR-V decorations ``Coherent``, ``Volatile``, ``Restrict``,
``NonWritable`` and ``NonReadable``. A decoration on the variable applies to all of the block's memory; one on a member
(``OpMemberDecorate``) applies to that member's memory (SPIR-V 1.6 section 3.20). glslang decorates both; other
producers, CrossTL's own SPIR-V output among them, decorate only the members. The importer wrote every storage block
as ``RWStructuredBuffer<B>`` and only turned ``NonWritable`` on the variable into ``StructuredBuffer``, so a round trip
lost ``Coherent``, ``Volatile``, ``Restrict`` and ``NonReadable``, and a block whose members alone were non-writable
was imported writable.

:func:`storage_block_memory_qualifiers` gives the qualifiers the whole imported buffer keeps, in a fixed order. CrossGL
carries memory qualifiers per buffer, so a member decoration is lifted to the buffer only in the direction that keeps
every access correct: ``Coherent`` and ``Volatile`` on any member make the buffer coherent or volatile (a stronger
visibility guarantee for the other members), while ``Restrict``, ``NonWritable`` and ``NonReadable`` hold for the
buffer only when the variable, the struct or every member carries them (claiming them for an undecorated member
would assert a no-aliasing or access promise the module never made).
"""

from __future__ import annotations

__all__ = ["storage_block_memory_qualifiers", "storage_buffer_memory_attribute_suffix"]

# SPIR-V decoration -> CrossGL qualifier, in emission order.
_QUALIFIERS = (
    ("NonWritable", "readonly"),
    ("NonReadable", "writeonly"),
    ("Coherent", "coherent"),
    ("Volatile", "volatile"),
    ("Restrict", "restrict"),
)
# Decorations a single member's decoration lifts to the whole buffer.
_ANY_MEMBER = frozenset({"Coherent", "Volatile"})


def storage_block_memory_qualifiers(
    block_decorations, member_decorations, member_count
):
    """The CrossGL memory qualifiers of a storage block.

    ``block_decorations`` are the ``(decoration, operands)`` pairs of the variable and of its struct type,
    ``member_decorations`` the ``(member, decoration, operands)`` triples of the struct, with the member index as a
    string, and ``member_count`` the struct's member count.
    """
    whole = {decoration for decoration, _operands in block_decorations}
    per_member = {}
    for member, decoration, _operands in member_decorations:
        per_member.setdefault(decoration, set()).add(str(member))
    every_member = {str(index) for index in range(member_count)}
    qualifiers = []
    for decoration, qualifier in _QUALIFIERS:
        members = per_member.get(decoration, set())
        if decoration in whole:
            qualifiers.append(qualifier)
        elif decoration in _ANY_MEMBER and members:
            qualifiers.append(qualifier)
        elif member_count and every_member <= members:
            qualifiers.append(qualifier)
    return qualifiers


def storage_buffer_memory_attribute_suffix(node):
    """`` @writeonly @coherent ...``: the CrossGL attributes of an imported storage buffer's memory qualifiers.

    ``readonly`` is not among them: the importer spells it as ``StructuredBuffer``.
    """
    qualifiers = set(getattr(node, "declaration_qualifiers", None) or [])
    attributes = [
        f"@{qualifier}"
        for _decoration, qualifier in _QUALIFIERS
        if qualifier != "readonly" and qualifier in qualifiers
    ]
    return "".join(f" {attribute}" for attribute in attributes)
