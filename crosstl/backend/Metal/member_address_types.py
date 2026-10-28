"""Pointer types of addressed members of private Metal values.

``&value.member`` names one member of a local struct or union value. Its type
is a pointer to the member's declared field type in the address space of the
local value itself (``thread`` unless the local was declared ``threadgroup``),
with the local's ``const`` qualification. Template argument deduction needs that
type to pick a constrained overload such as
``f(device T* object, thread uint* expected, ...)``.

Only one member level is typed, and only when the root's declaration retained
storage provenance and the field is a non-array, non-pointer value. Anything
else stays un-inferable so overload resolution fails closed.
"""

import re

_MEMBER_RE = re.compile(
    r"(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?P<member>[A-Za-z_][A-Za-z0-9_]*)"
)
_ADDRESS_SPACES = ("thread", "threadgroup")


def addressed_member_pointer_type(
    preprocessor, operand, local_variable_types, struct_field_types
):
    """Pointer type of ``&operand`` for ``operand`` = ``local.member``, or None."""
    match = _MEMBER_RE.fullmatch(operand)
    if match is None:
        return None
    root_pointer = getattr(
        local_variable_types.get(match.group("obj")), "pointer_type", None
    )
    if not root_pointer:
        return None
    root_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", root_pointer))
    spaces = [space for space in _ADDRESS_SPACES if space in root_tokens]
    if len(spaces) != 1:
        return None
    field_type = preprocessor._struct_member_field_type(operand, struct_field_types)
    if not field_type:
        return None
    field_type = field_type.strip()
    if "[" in field_type or "*" in field_type or "&" in field_type:
        return None
    if not preprocessor._is_metal_scalar_or_vector_type(
        preprocessor._normalize_inferred_type(field_type)
    ):
        return None
    qualifier = "const " if "const" in root_tokens else ""
    return preprocessor._normalize_known_address_space_pointer_type(
        f"{spaces[0]} {qualifier}{field_type}*"
    )


def merge_receiver_variable_types(code, local_variable_types, receiver_variable_types):
    """Merge struct-receiver declarations without hiding storage provenance.

    The local scanner records a declaration at the start of its statement with
    the local's address-space provenance; the receiver scanner records the same
    declaration at its type token without it. Keeping both would let the later,
    provenance-free entry win the nearest-preceding lookup, so a receiver entry
    that only restates an already recorded declaration is dropped.
    """
    for name, entries in receiver_variable_types.items():
        existing = local_variable_types.setdefault(name, [])
        for entry in entries:
            if entry in existing or _restates_declaration(code, existing, entry):
                continue
            existing.append(entry)
        existing.sort(key=lambda item: item[0])


def _restates_declaration(code, existing, entry):
    position, declared = entry
    for other_position, other in existing:
        if (
            getattr(other, "pointer_type", None)
            and str(other) == str(declared)
            and other_position <= position
            and not code[other_position:position].strip()
        ):
            return True
    return False
