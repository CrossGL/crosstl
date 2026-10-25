"""Plain-data structures that the Metal standard library declares in namespace ``metal``.

``<metal_stdlib>`` declares the per-patch tessellation factors a compute kernel writes for a tessellated draw
(Metal Shading Language Specification, section 2.18 "Per-Patch Tessellation Factors Structure",
``metal_tessellation``) and the indirect draw and dispatch argument records (``metal_graphics``,
``metal_compute``). Their layout is fixed by the Metal API. The Metal importer declares the ones a unit uses as
CrossGL structures with these members; the Metal generator emits no declaration for a CrossGL structure that
matches one exactly (:func:`is_metal_library_struct`), because Metal code that declares the name again makes
every unqualified use ambiguous under ``using namespace metal``.
"""

# name -> ((element type, member name, array length or None), ...), in declaration order, as the Metal
# standard library headers declare them.
METAL_LIBRARY_STRUCTS = {
    "MTLQuadTessellationFactorsHalf": (
        ("half", "edgeTessellationFactor", 4),
        ("half", "insideTessellationFactor", 2),
    ),
    "MTLTriangleTessellationFactorsHalf": (
        ("half", "edgeTessellationFactor", 3),
        ("half", "insideTessellationFactor", None),
    ),
    "MTLDrawPatchIndirectArguments": (
        ("uint", "patchCount", None),
        ("uint", "instanceCount", None),
        ("uint", "patchStart", None),
        ("uint", "baseInstance", None),
    ),
    "MTLDrawPrimitivesIndirectArguments": (
        ("uint", "vertexCount", None),
        ("uint", "instanceCount", None),
        ("uint", "vertexStart", None),
        ("uint", "baseInstance", None),
    ),
    "MTLDrawIndexedPrimitivesIndirectArguments": (
        ("uint", "indexCount", None),
        ("uint", "instanceCount", None),
        ("uint", "indexStart", None),
        ("uint", "baseVertex", None),
        ("uint", "baseInstance", None),
    ),
    "MTLDispatchThreadgroupsIndirectArguments": (("uint", "threadgroupsPerGrid", 3),),
    "MTLDispatchThreadsIndirectArguments": (
        ("uint", "threadsPerGrid", 3),
        ("uint", "threadsPerThreadgroup", 3),
    ),
}

# CrossGL spellings of each element type.
_CROSSGL_ELEMENT_SPELLINGS = {
    "half": frozenset({"half", "float16", "float16_t"}),
    "uint": frozenset({"uint", "uint32", "uint32_t"}),
}


def _type_name(type_node):
    if isinstance(type_node, str):
        return type_node.strip()
    name = getattr(type_node, "name", None)
    return name.strip() if isinstance(name, str) else None


def _array_length(type_node):
    size = getattr(type_node, "size", None)
    value = getattr(size, "value", size)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _member_matches(member, expected):
    element_type, member_name, length = expected
    if getattr(member, "name", None) != member_name:
        return False
    if getattr(member, "attributes", None) or getattr(member, "default_value", None):
        return False
    member_type = getattr(member, "member_type", None) or getattr(member, "vtype", None)
    if length is None:
        element = member_type
    else:
        element = getattr(member_type, "element_type", None)
        if element is None or _array_length(member_type) != length:
            return False
    return _type_name(element) in _CROSSGL_ELEMENT_SPELLINGS[element_type]


def is_metal_library_struct(node):
    """Return whether CrossGL structure ``node`` is the Metal library structure of its name."""
    expected = METAL_LIBRARY_STRUCTS.get(getattr(node, "name", None))
    if expected is None:
        return False
    members = list(getattr(node, "members", None) or [])
    return len(members) == len(expected) and all(
        _member_matches(member, wanted) for member, wanted in zip(members, expected)
    )
