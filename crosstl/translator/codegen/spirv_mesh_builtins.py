"""GLSL mesh-stage built-in output arrays for SPIR-V (GL_EXT_mesh_shader section 7.1.6).

A GLSL mesh shader writes its per-vertex and per-primitive built-ins through the implicitly declared arrays
``gl_MeshVerticesEXT[]`` (block ``gl_MeshPerVertexEXT``) and ``gl_MeshPrimitivesEXT[]`` (block
``gl_MeshPerPrimitiveEXT``), sized by ``max_vertices`` and ``max_primitives``. CrossGL keeps those spellings when
it imports GLSL or SPIR-V. The SPIR-V generator already lowers HLSL-style ``out vertices``/``out primitives``
parameters member by member; :func:`mesh_builtin_output_info` describes the two GLSL arrays in the same form, so
``gl_MeshVerticesEXT[i].gl_Position = p`` stores to the ``Position`` output at index ``i``. The primitive index
arrays (``gl_PrimitiveTriangleIndicesEXT[]`` and its point and line forms) lower like an ``out indices`` parameter.

Only the scalar and vector members are described. ``gl_ClipDistance`` and ``gl_CullDistance`` are arrays inside
the block, which the member lowering does not express, so writing them is refused as an unknown member.
"""

from __future__ import annotations

__all__ = [
    "MESH_BUILTIN_ARRAYS",
    "MESH_INDEX_ARRAYS",
    "mesh_builtin_output_info",
    "mesh_builtin_member_builtin",
    "mesh_index_builtin_variable",
]

# Name -> (lowering role, block name, ((member, CrossGL type), ...)), in block member order.
MESH_BUILTIN_ARRAYS = {
    "gl_MeshVerticesEXT": (
        "vertices",
        "gl_MeshPerVertexEXT",
        (("gl_Position", "vec4"), ("gl_PointSize", "float")),
    ),
    "gl_MeshPrimitivesEXT": (
        "primitives",
        "gl_MeshPerPrimitiveEXT",
        (
            ("gl_PrimitiveID", "int"),
            ("gl_Layer", "int"),
            ("gl_ViewportIndex", "int"),
            ("gl_CullPrimitiveEXT", "bool"),
            ("gl_PrimitiveShadingRateEXT", "int"),
        ),
    ),
}

# GL_EXT_mesh_shader section 7.1.6: the primitive index array the output topology selects.
MESH_INDEX_ARRAYS = {
    "gl_PrimitivePointIndicesEXT": "PrimitivePointIndicesEXT",
    "gl_PrimitiveLineIndicesEXT": "PrimitiveLineIndicesEXT",
    "gl_PrimitiveTriangleIndicesEXT": "PrimitiveTriangleIndicesEXT",
}

# Members the HLSL-style member lowering does not map to a BuiltIn decoration by semantic.
_EXTRA_MEMBER_BUILTINS = {
    ("vertices", "gl_PointSize"): "PointSize",
    ("primitives", "gl_PrimitiveShadingRateEXT"): "PrimitiveShadingRateKHR",
}


def mesh_builtin_output_info(codegen, name):
    """Mesh output description of ``gl_MeshVerticesEXT``/``gl_MeshPrimitivesEXT`` in a mesh stage, else ``None``.

    The description has the keys of ``VulkanSPIRVCodeGen.mesh_output_parameter_info``. ``element_type`` is
    ``None``: GLSL has no whole-element assignment to a built-in block, so only member stores are lowered.
    """
    if codegen.current_execution_model != "MeshEXT":
        return None
    name = str(name)
    if name not in MESH_BUILTIN_ARRAYS and name not in MESH_INDEX_ARRAYS:
        return None
    if codegen.resolve_global_variable(name) is not None:
        # A program that declares its own variable of that name means that variable.
        return None
    if name in MESH_INDEX_ARRAYS:
        return _index_array_info(codegen, name)
    role, block_name, members = MESH_BUILTIN_ARRAYS[name]
    return {
        "name": str(name),
        "role": role,
        "element_type_name": block_name,
        "element_type": None,
        "count": None,
        "members": {
            member: {
                "index": index,
                "type": codegen.map_crossgl_type(type_name),
                "semantic": member,
            }
            for index, (member, type_name) in enumerate(members)
        },
    }


def _index_array_info(codegen, name):
    """Mesh output description of a ``gl_Primitive*IndicesEXT`` array; refuse one the topology does not declare."""
    topology = codegen.mesh_primitive_index_builtin_info()
    if topology is None or topology[0] != MESH_INDEX_ARRAYS[name]:
        # GL_EXT_mesh_shader section 7.1.6: only the array of the declared output primitive type exists.
        codegen.refuse_lowering(
            f"{name} does not match the mesh stage's output primitive topology"
        )
    return {
        "name": name,
        "role": "indices",
        "element_type_name": None,
        "element_type": None if topology is None else topology[1],
        "count": None,
        "members": {},
    }


def mesh_builtin_member_builtin(role, semantic):
    """BuiltIn decoration of a mesh block member the semantic table does not cover, or ``None``."""
    return _EXTRA_MEMBER_BUILTINS.get((role, str(semantic)))


def mesh_index_builtin_variable(codegen, semantic, storage_class):
    """Primitive index output a declared ``gl_Primitive*IndicesEXT`` array names, or ``None`` for other semantics.

    A SPIR-V import declares the index array it read (``uint3 gl_PrimitiveTriangleIndicesEXT[N] @output
    @gl_PrimitiveTriangleIndicesEXT``); it is the topology's ``Primitive*IndicesEXT`` built-in, never a location.
    """
    name = str(semantic)
    if (
        storage_class != "Output"
        or codegen.current_execution_model != "MeshEXT"
        or name not in MESH_INDEX_ARRAYS
    ):
        return None
    topology = codegen.mesh_primitive_index_builtin_info()
    if topology is None or topology[0] != MESH_INDEX_ARRAYS[name]:
        codegen.refuse_lowering(
            f"{name} does not match the mesh stage's output primitive topology"
        )
        return None
    variable, _element_type = codegen.ensure_mesh_primitive_index_output()
    return variable
