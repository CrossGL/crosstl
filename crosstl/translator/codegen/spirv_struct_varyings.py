"""Stage varyings of structure type, flattened to one SPIR-V interface variable per field.

A stage-structure member of structure type with a location (the GLSL import produces one for an instance interface
block, ``out Data {...} data;``; CrossGL can declare one directly) occupies consecutive locations from that location,
as many per field as the field's type needs. One ``Output`` or ``Input`` variable of that non-``Block`` structure
type keeps the locations, but it cannot keep the fields' own decorations: Vulkan allows ``Location`` and
``Component`` on members of ``Block`` structures only (Vulkan 1.3, section 15.1.5, "Location Assignment"), and an
interpolation decoration on a member of an ordinary structure says nothing about the interface. The emitter had
therefore dropped ``flat`` from a field, so an integer fragment input was interpolated, which
``VUID-StandaloneSpirv-Flat-04744`` forbids, and dropped a field's explicit location.

Each field becomes its own interface variable instead, as glslang lays out a block's members: the field's own
location when it has one, otherwise the location after the previous field's, and the field's interpolation and
``component`` decorations together with those of the enclosing member. Nested structures are flattened the same
way. A vertex output and a fragment input flattened alike occupy the same locations, so the two stages still match,
and a variable per location also matches a GLSL ``Block`` or an HLSL structure semantic in the other stage.
"""

from __future__ import annotations

import copy
from typing import NamedTuple

from ..ast import AttributeNode

__all__ = [
    "FlattenedField",
    "flatten_struct_varying",
    "load_flattened_struct_varying",
    "store_flattened_struct_varying",
]

# Varyings that are one value per invocation; arrayed stage interfaces (geometry, tessellation, mesh) keep their form.
_FLATTENED_INTERFACES = {("Vertex", "Output"), ("Fragment", "Input")}
_INTERPOLATION = frozenset({"flat", "noperspective", "smooth"})
_QUALIFIER_GROUPS = (
    _INTERPOLATION,
    frozenset({"centroid", "sample"}),
    frozenset({"invariant"}),
)
_INHERITED_ATTRIBUTES = frozenset().union(*_QUALIFIER_GROUPS)


class FlattenedField(NamedTuple):
    """One leaf field: its index path in the member value, the types along that path, and its variable."""

    path: tuple[int, ...]
    types: tuple[object, ...]
    variable: object


def _attribute_names(node) -> set:
    return {
        str(getattr(attr, "name", "")).lower()
        for attr in getattr(node, "attributes", []) or []
    }


def _struct_fields(codegen, type_id):
    """``(field_type, field_name, field_node)`` for a user structure type, else None."""
    if codegen.array_type_info_from_type(type_id) is not None:
        return None
    type_name = type_id.type.base_type
    members = codegen.current_struct_members.get(type_name)
    if not members:
        return None
    metadata = codegen.struct_member_metadata.get(type_name, {})
    return [
        (field_type, field_name, metadata.get(field_name, {}).get("node"))
        for field_type, field_name in members
    ]


def _contains_struct_array(codegen, type_id) -> bool:
    array_info = codegen.array_type_info_from_type(type_id)
    if array_info is None:
        return False
    element_type, _ = array_info
    return _struct_fields(codegen, element_type) is not None or _contains_struct_array(
        codegen, element_type
    )


def _field_node(codegen, field_node, field_type, name, location, inherited):
    """A copy of ``field_node`` at ``location`` that also carries the enclosing member's interpolation."""
    if field_node is None:
        from ..ast import StructMemberNode

        field_node = StructMemberNode(name, field_type.type.base_type)
    node = copy.copy(field_node)
    attributes = [
        attr
        for attr in getattr(field_node, "attributes", []) or []
        if str(getattr(attr, "name", "")).lower() != "location"
    ]
    present = _attribute_names(field_node)
    for attr in inherited:
        # A field's own interpolation or auxiliary qualifier overrides the enclosing member's.
        group = next(group for group in _QUALIFIER_GROUPS if attr.name.lower() in group)
        if group is _INTERPOLATION and codegen.interface_type_requires_flat(
            codegen.lowered_user_defined_interface_type(field_type)
        ):
            # An integer field is always flat; the enclosing member's interpolation applies to its other fields.
            continue
        if not present & group:
            attributes.append(attr)
            present |= {attr.name.lower()}
    attributes.append(AttributeNode("location", [str(location)]))
    node.attributes = attributes
    return node


def _leaves(codegen, type_id, location, inherited, path, types, names):
    """Yield ``(path, types, names, field_node, location)`` for each leaf field, in declaration order."""
    fields = _struct_fields(codegen, type_id)
    for index, (field_type, field_name, field_node) in enumerate(fields):
        own = codegen.explicit_location_attribute(field_node) if field_node else None
        if own is not None:
            location = own
        field_path = path + (index,)
        field_types = types + (field_type,)
        field_names = names + (str(field_name),)
        if _struct_fields(codegen, field_type) is not None:
            # The nearer qualifier comes first, so it wins over the enclosing member's.
            nested_inherited = [
                attr
                for attr in getattr(field_node, "attributes", []) or []
                if str(getattr(attr, "name", "")).lower() in _INHERITED_ATTRIBUTES
            ] + inherited
            for leaf in _leaves(
                codegen,
                field_type,
                location,
                nested_inherited,
                field_path,
                field_types,
                field_names,
            ):
                yield leaf
                location = leaf[4] + codegen.struct_member_interface_location_span(
                    leaf[3]
                )
            continue
        leaf_node = _field_node(
            codegen, field_node, field_type, field_name, location, inherited
        )
        yield field_path, field_types, field_names, leaf_node, location
        location += codegen.struct_member_interface_location_span(leaf_node)


def flatten_struct_varying(
    codegen,
    execution_model: str | None,
    storage_class: str,
    name: str,
    member_type,
    member_node,
) -> list[FlattenedField] | None:
    """Register one interface variable per leaf field of a located structure-typed varying.

    Returns the fields in declaration order, or None when ``member_node`` is not such a varying: not a vertex output
    or fragment input, not of structure type, without an explicit location, or holding an array of structures,
    which keeps its single variable.
    """
    if (execution_model, storage_class) not in _FLATTENED_INTERFACES:
        return None
    if member_node is None or _struct_fields(codegen, member_type) is None:
        return None
    location = codegen.explicit_location_attribute(member_node)
    if location is None:
        return None
    if _has_struct_array(codegen, member_type):
        return None

    inherited = [
        attr
        for attr in getattr(member_node, "attributes", []) or []
        if str(getattr(attr, "name", "")).lower() in _INHERITED_ATTRIBUTES
    ]
    flattened = []
    for path, types, names, field_node, _ in list(
        _leaves(codegen, member_type, location, inherited, (), (), ())
    ):
        variable = codegen.register_entry_point_interface_variable(
            execution_model,
            storage_class,
            "_".join((name,) + names),
            types[-1],
            field_node,
            member_name=names[-1],
        )
        flattened.append(FlattenedField(path, types, variable))
    return flattened


def _has_struct_array(codegen, type_id) -> bool:
    for field_type, _, _ in _struct_fields(codegen, type_id):
        if _contains_struct_array(codegen, field_type):
            return True
        if _struct_fields(codegen, field_type) is not None and _has_struct_array(
            codegen, field_type
        ):
            return True
    return False


def load_flattened_struct_varying(codegen, member_type, fields: list[FlattenedField]):
    """The ``member_type`` value assembled from the flattened fields' input variables."""

    def build(type_id, depth, group):
        components = []
        index = 0
        while index < len(group):
            field = group[index]
            if len(field.path) == depth + 1:
                components.append(
                    codegen.load_entry_point_interface_value(
                        field.variable, field.types[depth]
                    )
                )
                index += 1
                continue
            nested = [
                item
                for item in group[index:]
                if item.path[: depth + 1] == field.path[: depth + 1]
            ]
            components.append(build(field.types[depth], depth + 1, nested))
            index += len(nested)
        return codegen.composite_construct(type_id, components)

    return build(member_type, 0, fields)


def store_flattened_struct_varying(codegen, member_value, fields: list[FlattenedField]):
    """Store each leaf of ``member_value`` to its flattened output variable."""
    for field in fields:
        value = member_value
        for index, type_id in zip(field.path, field.types):
            value = codegen.composite_extract(value, type_id, index)
        codegen.store_to_variable(field.variable, value)
