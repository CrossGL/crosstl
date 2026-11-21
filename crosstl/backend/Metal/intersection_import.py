"""Metal intersection functions imported as CrossGL intersection stages.

A Metal intersection function is marked ``[[intersection(primitive_type, tags...)]]`` (Metal Shading Language
Specification, section 5.1.6). It receives the ray and the candidate primitive through attributed parameters
(``[[origin]]``, ``[[min_distance]]``, ``[[primitive_id]]``, ...; section 5.2.3.7), may read and write the ray's
``ray_data`` payload, and returns its verdict: ``bool`` or a structure whose members are marked
``[[accept_intersection]]``, ``[[continue_search]]`` or ``[[distance]]`` (section 5.2.3.8).

CrossGL spells the same function as its ``ray_intersection`` stage with the primitive type as an entry attribute,
``@bounding_box`` or ``@triangle``. The intersection tags after the primitive type state which intersector may call
the function and which inputs it may take (the compiler rejects ``[[instance_id]]`` without ``instancing``); each
becomes an entry attribute of the same name (``@instancing``, ``@world_space_data``, ``@max_levels(2)`` for
``max_levels<2>``), which the Metal generator writes back. Parameter and member attributes keep their Metal names as
CrossGL semantics, including ``[[primitive_id]]``, ``[[instance_id]]``, ``[[front_facing]]`` and
``[[barycentric_coord]]``, which describe the candidate primitive and instance here rather than the rasterized
primitive and the drawn instance. A primitive type other than ``bounding_box`` and ``triangle`` (``curve``) and an
unknown tag are refused.
"""

from __future__ import annotations

import re

from ..common_ast import AttributeNode

__all__ = ["MetalIntersectionImport"]

_PRIMITIVE_TYPES = frozenset({"bounding_box", "triangle"})

# Intersection tags with no argument (section 5.1.6, table 5.2).
_TAGS = frozenset(
    {
        "curve_data",
        "extended_limits",
        "instance_motion",
        "instancing",
        "primitive_motion",
        "triangle_data",
        "world_space_data",
    }
)
_MAX_LEVELS = re.compile(r"max_levels\s*<\s*(\d+)\s*>")

# Ray-input attributes the stage keeps under their Metal names instead of the rasterizer built-ins they share a
# spelling with in other stages.
_RAY_INPUT_SEMANTICS = {
    name: name
    for name in ("barycentric_coord", "front_facing", "instance_id", "primitive_id")
}


class MetalIntersectionImport:
    """Import members of ``MetalToCrossGLConverter`` for ``[[intersection(...)]]`` functions."""

    def generate_metal_intersection_stage(self, function, **options):
        """A ``ray_intersection { ... }`` block for one intersection function, or ``""`` when it is refused."""
        attributes = list(getattr(function, "attributes", None) or [])
        attribute = next(
            (
                item
                for item in attributes
                if getattr(item, "name", None) == "intersection"
            ),
            None,
        )
        arguments = [
            str(argument).strip() for argument in getattr(attribute, "args", None) or []
        ]
        if not arguments or arguments[0] not in _PRIMITIVE_TYPES:
            primitive = arguments[0] if arguments else "<none>"
            self.refuse_import(
                f"Metal intersection function '{function.name}' has primitive type '{primitive}'; CrossGL's "
                "intersection stage imports bounding_box and triangle functions"
            )
            return ""
        tags = []
        for argument in arguments[1:]:
            tag = _intersection_tag(argument)
            if tag is None:
                self.refuse_import(
                    f"Metal intersection function '{function.name}' names intersection tag '{argument}'; CrossGL "
                    f"spells the tags {', '.join(sorted(_TAGS))} and max_levels<N>"
                )
                return ""
            tags.append(tag)
        function.attributes = [item for item in attributes if item is not attribute] + [
            AttributeNode(arguments[0]),
            *tags,
        ]
        semantics = self.map_semantics
        self.map_semantics = {**semantics, **_RAY_INPUT_SEMANTICS}
        try:
            text = self.generate_function(
                function, stage_entry=function.name != "main", **options
            )
        finally:
            function.attributes = attributes
            self.map_semantics = semantics
        return (
            "    // Intersection Shader\n    ray_intersection {\n" + text + "    }\n\n"
        )


def _intersection_tag(argument):
    """The CrossGL entry attribute for one Metal intersection tag, or None for a tag CrossGL does not spell."""
    if argument in _TAGS:
        return AttributeNode(argument)
    match = _MAX_LEVELS.fullmatch(argument)
    if match:
        return AttributeNode("max_levels", [match.group(1)])
    return None
