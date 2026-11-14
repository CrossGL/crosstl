"""The stage roles of Slang entry-point parameters and attributes, in the CrossGL spelling the HLSL importer writes.

Slang declares mesh shaders as HLSL does: ``out vertices V verts[N]``, ``out indices uint3 tris[M]``, ``out primitives
P prims[M]`` and ``in payload T payload`` parameters, with ``[numthreads(...)]`` and ``[outputtopology(...)]`` on the
entry point. Ray tracing stages take their payload, hit attributes and callable data as plain parameters (Slang
language reference, "Ray tracing"). The Slang importer dropped every one of these roles: the mesh parameters became
ordinary ``out`` arrays, ``in payload T payload`` became ``in payload payload T`` (which the CrossGL parser rejects),
the mesh stage lost its thread-group size and topology, and a miss or hit stage's payload was an ordinary ``inout``
parameter. The HLSL importer writes ``@ vertices out``, ``@ indices out``, ``@ primitives out``,
``@ mesh_payload in``, ``@ numthreads(...)``, ``@ outputtopology(...)`` and ``@ payload``, ``@ hit_attribute`` and
``@ callable_data``; these helpers give the Slang importer the same spellings.
"""

from __future__ import annotations

__all__ = [
    "slang_mesh_parameter_role",
    "slang_mesh_stage_attributes",
    "slang_ray_parameter_role",
]

_MESH_ROLES = {
    "vertices": "vertices",
    "indices": "indices",
    "primitives": "primitives",
    "payload": "mesh_payload",
}
_HIT_STAGES = frozenset({"ray_closest_hit", "ray_any_hit"})
_HIT_ATTRIBUTE_TYPE = "BuiltInTriangleIntersectionAttributes"


def _qualifier_names(param):
    return [str(qualifier) for qualifier in getattr(param, "qualifiers", None) or []]


def slang_mesh_parameter_role(param, function_qualifier):
    """``vertices``, ``indices``, ``primitives`` or ``mesh_payload`` for a mesh entry parameter, else ``None``."""
    if function_qualifier != "mesh":
        return None
    for qualifier in _qualifier_names(param):
        role = _MESH_ROLES.get(qualifier)
        if role is not None:
            return role
    return None


def slang_ray_parameter_role(param, function_qualifier, index):
    """``payload``, ``hit_attribute`` or ``callable_data`` for a ray tracing entry parameter, else ``None``.

    The roles follow the parameters' positions, as the HLSL importer infers them: a miss stage's first parameter is
    its payload, a hit stage's first its payload and its second (or any of the built-in triangle attribute type) its
    hit attributes, and a callable stage's first its callable data.
    """
    if index is None:
        return None
    if function_qualifier == "ray_miss" and index == 0:
        return "payload"
    if function_qualifier in _HIT_STAGES:
        if str(getattr(param, "vtype", "")) == _HIT_ATTRIBUTE_TYPE or index == 1:
            return "hit_attribute"
        if index == 0:
            return "payload"
    if function_qualifier == "ray_callable" and index == 0:
        return "callable_data"
    return None


def _attribute(func, name):
    for attribute in getattr(func, "attributes", None) or []:
        if str(attribute.get("name", "")).lower() == name:
            return attribute
    return None


def slang_mesh_stage_attributes(func, function_qualifier):
    """The ``@ numthreads(...)`` and ``@ outputtopology(...)`` lines of a mesh or task entry point."""
    if function_qualifier not in {"mesh", "task", "amplification"}:
        return ""
    lines = []
    numthreads = getattr(func, "numthreads", None)
    if numthreads:
        lines.append(
            f"    @ numthreads({', '.join(str(size) for size in numthreads)})\n"
        )
    topology = _attribute(func, "outputtopology")
    if function_qualifier == "mesh" and topology and topology.get("arguments"):
        lines.append(f"    @ outputtopology({topology['arguments'][0]})\n")
    return "".join(lines)
