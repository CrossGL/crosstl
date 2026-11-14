"""References to the enumerators of unscoped Metal enums.

MSL follows C++: the enumerators of an unscoped enum (``enum E { A, B }`` or the C ``typedef enum { ... } E``) are
names of the enclosing scope, so shaders write ``A`` (C++17 section 10.2). CrossGL enumerators are scoped and every
generator spells ``E::A`` as its own constant (``E_A`` in HLSL and GLSL), so a bare ``A`` named nothing in the
output: ``out[i] = AAPLNumMeshTextures;`` failed to compile on HLSL and GLSL and was refused on SPIR-V.
:func:`unscoped_enumerator_owners` maps each enumerator of an unscoped enum to its enum, so the importer writes
``E::A`` for it.
"""

from __future__ import annotations

__all__ = ["unscoped_enumerator_owners"]


def unscoped_enumerator_owners(enums_with_names) -> dict:
    """``{enumerator: enum name}`` for the unscoped enums of ``enums_with_names`` (``(enum, emitted name)`` pairs).

    An enumerator two unscoped enums both declare is left out: C++ rejects that redeclaration, and its references
    cannot be attributed.
    """
    owners, repeated = {}, set()
    for enum, name in enums_with_names:
        if getattr(enum, "is_scoped", False) or not name:
            continue
        for member, _value in getattr(enum, "members", None) or ():
            if member in owners and owners[member] != name:
                repeated.add(member)
            owners.setdefault(member, name)
    for member in repeated:
        owners.pop(member, None)
    return owners
