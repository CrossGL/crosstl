"""Intersection tags of a Metal intersection function.

Metal writes an intersection function as ``[[intersection(primitive_type, tags...)]]`` (Metal Shading Language
Specification, section 5.1.6). A tag states which intersector may call the function and which inputs the function may
take. Apple's compiler rejects ``[[instance_id]]`` and ``[[user_instance_id]]`` without ``instancing``;
``[[world_space_origin]]``, ``[[world_space_direction]]``, ``[[object_to_world_transform]]`` and
``[[world_to_object_transform]]`` without ``world_space_data``; ``[[barycentric_coord]]`` and ``[[front_facing]]``
without ``triangle_data``; ``[[time]]`` without ``primitive_motion``; and ``world_space_data``, ``instance_motion``
and ``max_levels<N>`` without ``instancing``. It also rejects a scalar ``[[instance_id]]`` in a
function tagged ``max_levels<N>``, which the generator refuses.

CrossGL states a tag as an entry attribute next to the primitive type: ``@instancing``, ``@world_space_data``,
``@max_levels(2)`` and so on, as the Metal importer writes them. The generator writes the tags the entry states plus
the tags its inputs require, in one fixed order, so a function that reads ``[[instance_id]]`` compiles whether or not
its source named ``instancing``.
"""

from __future__ import annotations

from ..refusals import MetalRefusal

__all__ = [
    "METAL_INTERSECTION_INPUTS",
    "metal_intersection_tags",
]

# Emission order of the tags without an argument; ``max_levels<N>`` follows them.
_TAG_ORDER = (
    "triangle_data",
    "instancing",
    "world_space_data",
    "primitive_motion",
    "instance_motion",
    "extended_limits",
    "curve_data",
)

# Each input attribute that needs a tag, and the tag (section 5.2.3.7).
_INPUT_TAGS = {
    "barycentric_coord": "triangle_data",
    "front_facing": "triangle_data",
    "instance_id": "instancing",
    "time": "primitive_motion",
    "user_instance_id": "instancing",
    "world_space_origin": "world_space_data",
    "world_space_direction": "world_space_data",
    "object_to_world_transform": "world_space_data",
    "world_to_object_transform": "world_space_data",
}

# Tags the compiler accepts only together with ``instancing``.
_NEED_INSTANCING = frozenset({"instance_motion", "world_space_data"})

# Input attributes an intersection function may take, by primitive type (section 5.2.3.7).
METAL_INTERSECTION_INPUTS = {
    "bounding_box": frozenset(
        {
            "direction",
            "geometry_id",
            "instance_id",
            "max_distance",
            "min_distance",
            "object_to_world_transform",
            "origin",
            "primitive_id",
            "time",
            "user_instance_id",
            "world_space_direction",
            "world_space_origin",
            "world_to_object_transform",
        }
    ),
}
METAL_INTERSECTION_INPUTS["triangle"] = METAL_INTERSECTION_INPUTS["bounding_box"] | {
    "barycentric_coord",
    "distance",
    "front_facing",
}


def metal_intersection_tags(generator, func, primitive_type):
    """The tags after ``primitive_type`` in the entry's ``[[intersection(...)]]`` attribute, in emission order."""
    tags = set()
    max_levels = None
    for attribute in getattr(func, "attributes", None) or []:
        name = str(getattr(attribute, "name", "") or "").strip()
        if name in _TAG_ORDER:
            tags.add(name)
        elif name == "max_levels":
            max_levels = _max_levels(generator, func, attribute)
            tags.add("instancing")
    inputs = _input_attributes(generator, func)
    tags.update(_INPUT_TAGS[name] for name in inputs if name in _INPUT_TAGS)
    if tags & _NEED_INSTANCING:
        tags.add("instancing")
    if max_levels is not None and inputs & {"instance_id", "user_instance_id"}:
        raise MetalRefusal(
            f"Metal intersection function '{func.name}' reads [[instance_id]] or [[user_instance_id]] with "
            "max_levels<N>; Metal rejects a scalar instance id in a multi-level instancing function"
        )
    ordered = [tag for tag in _TAG_ORDER if tag in tags]
    if max_levels is not None:
        ordered.append(f"max_levels<{max_levels}>")
    return ordered


def _input_attributes(generator, func):
    names = set()
    for parameter in getattr(func, "parameters", getattr(func, "params", [])) or []:
        key = generator.canonical_metal_semantic(
            generator.semantic_from_node(parameter)
        )
        if key is not None:
            names.add(str(key).strip().lower())
    return names


def _max_levels(generator, func, attribute):
    arguments = (
        getattr(attribute, "arguments", None) or getattr(attribute, "args", None) or []
    )
    text = (
        generator.attribute_value_to_string(arguments[0])
        if len(arguments) == 1
        else None
    )
    if text is None or not str(text).strip().isdigit() or int(str(text).strip()) < 1:
        raise MetalRefusal(
            f"Metal intersection function '{func.name}' needs one positive integer in @max_levels(N)"
        )
    return int(str(text).strip())
