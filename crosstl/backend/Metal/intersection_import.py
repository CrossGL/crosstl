"""Metal intersection functions imported as CrossGL intersection stages.

A Metal intersection function is marked ``[[intersection(primitive_type, tags...)]]`` (Metal Shading Language
Specification, section 5.1.6). It receives the ray and the candidate primitive through attributed parameters
(``[[origin]]``, ``[[min_distance]]``, ``[[primitive_id]]``, ...; section 5.2.3.7), may read and write the ray's
``ray_data`` payload, and returns its verdict: ``bool`` or a structure whose members are marked
``[[accept_intersection]]``, ``[[continue_search]]`` or ``[[distance]]`` (section 5.2.3.8). The importer refused
every such function.

CrossGL spells the same function as its ``ray_intersection`` stage with the primitive type as an entry attribute,
``@bounding_box`` or ``@triangle``. Parameter and member attributes keep their Metal names as CrossGL semantics, as
the Metal generator reads them back. Intersection tags beyond the primitive type (``instancing``,
``triangle_data``, ``world_space_data``, ...) state which intersector may call the function and change the
parameters it may take; CrossGL has no spelling for them yet, so a function that names one is refused, as is a
primitive type other than ``bounding_box`` and ``triangle`` (``curve``).
"""

from __future__ import annotations

from ..common_ast import AttributeNode

__all__ = ["MetalIntersectionImport"]

_PRIMITIVE_TYPES = frozenset({"bounding_box", "triangle"})


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
        if len(arguments) > 1:
            self.refuse_import(
                f"Metal intersection function '{function.name}' names intersection tags "
                f"{', '.join(arguments[1:])}; CrossGL has no spelling for intersection tags"
            )
            return ""
        function.attributes = [item for item in attributes if item is not attribute] + [
            AttributeNode(arguments[0])
        ]
        try:
            text = self.generate_function(
                function, stage_entry=function.name != "main", **options
            )
        finally:
            function.attributes = attributes
        return (
            "    // Intersection Shader\n    ray_intersection {\n" + text + "    }\n\n"
        )
