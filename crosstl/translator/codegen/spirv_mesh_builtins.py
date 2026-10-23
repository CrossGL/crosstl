"""GLSL ``GL_EXT_mesh_shader`` output built-ins as SPIR-V mesh output arrays.

GLSL mesh shaders write their per-vertex and per-primitive built-ins through the
implicitly declared arrays ``gl_MeshVerticesEXT[]`` and ``gl_MeshPrimitivesEXT[]``
and the topology's index array ``gl_Primitive{Point,Line,Triangle}IndicesEXT[]``
(GLSL_EXT_mesh_shader, "Built-in Variables"). The SPIR-V generator lowers HLSL
``out vertices``/``out primitives``/``out indices`` parameters to the same
``BuiltIn``-decorated ``Output`` arrays, so the GLSL arrays are described with
the parameter metadata those lowerings consume.

Only built-ins with a single SPIR-V decoration are described. Writes to other
members, such as ``gl_ClipDistance``, find no member and are refused.

A SPIR-V import spells the index array as a stage output carrying its built-in
name as the semantic. That declaration names the same built-in, so it is not
emitted as a ``Location`` output; the array is lowered like the implicit one.
"""

__all__ = [
    "glsl_mesh_builtin_output_infos",
    "is_glsl_mesh_builtin_output_declaration",
]

# (member, CrossGL type, semantic understood by ``mesh_output_member_builtin``)
_VERTEX_MEMBERS = (
    ("gl_Position", "vec4", "gl_Position"),
    ("gl_PointSize", "float", "gl_PointSize"),
)
_PRIMITIVE_MEMBERS = (
    ("gl_PrimitiveID", "int", "gl_PrimitiveID"),
    ("gl_Layer", "int", "gl_Layer"),
    ("gl_ViewportIndex", "int", "gl_ViewportIndex"),
    ("gl_CullPrimitiveEXT", "bool", "gl_CullPrimitiveEXT"),
)
_INDEX_ARRAYS = {
    "PrimitivePointIndicesEXT": "gl_PrimitivePointIndicesEXT",
    "PrimitiveLineIndicesEXT": "gl_PrimitiveLineIndicesEXT",
    "PrimitiveTriangleIndicesEXT": "gl_PrimitiveTriangleIndicesEXT",
}


def _members(codegen, members):
    return {
        name: {
            "index": index,
            "type": codegen.map_crossgl_type(type_name),
            "semantic": semantic,
        }
        for index, (name, type_name, semantic) in enumerate(members)
    }


def _info(name, role, element_type, members):
    return {
        "name": name,
        "role": role,
        "element_type_name": None,
        "element_type": element_type,
        "count": None,
        "members": members,
    }


_BUILTIN_ARRAYS = frozenset(
    {"gl_MeshVerticesEXT", "gl_MeshPrimitivesEXT", *_INDEX_ARRAYS.values()}
)


def is_glsl_mesh_builtin_output_declaration(codegen, node) -> bool:
    """Whether a mesh-stage output declaration names a GLSL mesh built-in array."""
    if codegen.current_execution_model != "MeshEXT":
        return False
    name = getattr(node, "name", None)
    if name not in _BUILTIN_ARRAYS:
        return False
    return str(codegen.semantic_from_node(node) or "") == name


def glsl_mesh_builtin_output_infos(codegen) -> dict:
    """Mesh output metadata for the GLSL built-in arrays the module does not redeclare."""
    declared = {
        name
        for name in _BUILTIN_ARRAYS
        if codegen.global_variable_exists_for_current_stage(name)
    }
    infos = {}
    if "gl_MeshVerticesEXT" not in declared:
        infos["gl_MeshVerticesEXT"] = _info(
            "gl_MeshVerticesEXT", "vertices", None, _members(codegen, _VERTEX_MEMBERS)
        )
    if "gl_MeshPrimitivesEXT" not in declared:
        infos["gl_MeshPrimitivesEXT"] = _info(
            "gl_MeshPrimitivesEXT",
            "primitives",
            None,
            _members(codegen, _PRIMITIVE_MEMBERS),
        )
    index_builtin = codegen.mesh_primitive_index_builtin_info()
    if index_builtin is not None:
        builtin_name, element_type = index_builtin
        name = _INDEX_ARRAYS[builtin_name]
        if name not in declared:
            infos[name] = _info(name, "indices", element_type, {})
    return infos
