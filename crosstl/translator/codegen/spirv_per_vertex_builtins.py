"""SPIR-V lowering of GLSL's per-vertex built-in arrays ``gl_in`` and ``gl_out``.

A GLSL geometry, tessellation control or tessellation evaluation stage reads the built-ins of each input vertex
through the ``gl_PerVertex`` block array ``gl_in`` (``gl_in[i].gl_Position``), and a tessellation control stage
writes those of its output control points through ``gl_out`` (``gl_out[gl_InvocationID].gl_Position``). SPIR-V has
no implicit built-ins. Vulkan accepts each built-in as its own arrayed interface variable, matched across stages
by its ``BuiltIn`` decoration rather than by block layout (Vulkan 1.3 section 15.1.4), and DXC declares them that
way. :func:`lower_per_vertex_builtin_arrays` therefore rewrites ``gl_in[i].m`` into an index of an arrayed built-in
named by :func:`per_vertex_builtin_name`, which :func:`per_vertex_builtin_info` sizes as glslang does: an input of a
tessellation stage has ``gl_MaxPatchVertices`` (32) elements, a geometry input one per vertex of its input
primitive, and ``gl_out`` one per output control point.

``gl_in.length()`` and ``gl_out.length()`` fold to those lengths. A use the arrays cannot express (a whole
``gl_in[i]`` value, ``gl_out`` outside a tessellation control stage, or a block member other than ``gl_Position``
and ``gl_PointSize``) is left in place, so the emitter refuses it as an unknown variable.
"""

from __future__ import annotations

from .. import ast as A
from ..semantics.rewrite import child_slots, get_child, set_child

__all__ = [
    "MAX_PATCH_VERTICES",
    "lower_per_vertex_builtin_arrays",
    "per_vertex_builtin_info",
    "per_vertex_builtin_name",
]

# gl_MaxPatchVertices, the size glslang gives unsized tessellation inputs (GLSL 4.60 section 7.3).
MAX_PATCH_VERTICES = 32

_MEMBERS = {
    "gl_Position": ("vec4", "Position"),
    "gl_PointSize": ("float", "PointSize"),
}
_BLOCKS = {"gl_in": "Input", "gl_out": "Output"}
_PREFIX = "gl_PerVertex"
_INPUT_STAGES = {"Geometry", "TessellationControl", "TessellationEvaluation"}
_GEOMETRY_INPUT_VERTICES = {
    "InputPoints": 1,
    "InputLines": 2,
    "InputLinesAdjacency": 4,
    "Triangles": 3,
    "InputTrianglesAdjacency": 6,
}


def per_vertex_builtin_name(block, member):
    """Name of the arrayed built-in that holds ``member`` of every element of ``block`` (``gl_in``/``gl_out``)."""
    storage = "In" if _BLOCKS[block] == "Input" else "Out"
    return f"{_PREFIX}{storage}_{member}"


def _parse_name(name):
    for block, storage in _BLOCKS.items():
        storage_tag = "In" if storage == "Input" else "Out"
        prefix = f"{_PREFIX}{storage_tag}_"
        if name.startswith(prefix) and name[len(prefix) :] in _MEMBERS:
            return storage, name[len(prefix) :]
    return None


def lower_per_vertex_builtin_arrays(program, codegen):
    """Rewrite ``gl_in[i].m``, ``gl_out[i].m`` and ``gl_in.length()`` in ``program``, in place; return the count.

    ``length()`` folds to the array length of the stage the call is in. A function outside every stage is folded
    only when the program has a single stage.
    """
    if program is None:
        return 0
    stages = list((getattr(program, "stages", None) or {}).items())
    count = 0
    for stage_type, stage in stages:
        model = codegen.spirv_execution_model(codegen.stage_key(stage_type))
        lengths = _block_lengths(codegen, model, stage)
        lowering = _PerVertexLowering(lengths)
        for function in [
            getattr(stage, "entry_point", None),
            *(getattr(stage, "local_functions", None) or ()),
        ]:
            if function is not None:
                count += lowering.visit(function)
    shared = _block_lengths(codegen, *_sole_stage(codegen, stages))
    lowering = _PerVertexLowering(shared)
    for function in getattr(program, "functions", None) or ():
        count += lowering.visit(function)
    return count


def _sole_stage(codegen, stages):
    if len(stages) != 1:
        return None, None
    stage_type, stage = stages[0]
    return codegen.spirv_execution_model(codegen.stage_key(stage_type)), stage


def _block_lengths(codegen, model, stage):
    """Element counts of ``gl_in`` and ``gl_out`` in a stage of execution model ``model``."""
    lengths = {}
    if model in {"TessellationControl", "TessellationEvaluation"}:
        lengths["gl_in"] = MAX_PATCH_VERTICES
    elif model == "Geometry":
        lengths["gl_in"] = _GEOMETRY_INPUT_VERTICES.get(
            codegen.geometry_input_mode(stage)
        )
    if model == "TessellationControl":
        lengths["gl_out"] = codegen.tessellation_control_output_vertices(stage)
    return {block: length for block, length in lengths.items() if length}


def _block_name(node):
    if isinstance(node, (A.IdentifierNode, A.VariableNode)) and node.name in _BLOCKS:
        return node.name
    return None


class _PerVertexLowering:
    def __init__(self, lengths):
        self.lengths = lengths
        self.visited = set()

    def visit(self, node):
        count = 0
        if id(node) in self.visited:
            return count
        self.visited.add(id(node))
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self.rewrite(child)
            if replacement is not child:
                set_child(node, field, key, replacement)
                count += 1
                child = replacement
            count += self.visit(child)
        return count

    def rewrite(self, node):
        if isinstance(node, A.FunctionCallNode):
            return self.rewrite_length(node)
        if not isinstance(node, A.MemberAccessNode):
            return node
        element = node.object
        if not isinstance(element, A.ArrayAccessNode):
            return node
        block = _block_name(element.array)
        if block is None or node.member not in _MEMBERS:
            return node
        array = A.IdentifierNode(per_vertex_builtin_name(block, node.member))
        return A.ArrayAccessNode(array, element.index)

    def rewrite_length(self, node):
        callee = node.function
        if node.arguments or not isinstance(callee, A.MemberAccessNode):
            return node
        if callee.member != "length":
            return node
        length = self.lengths.get(_block_name(callee.object))
        if length is None:
            return node
        return A.LiteralNode(length, A.PrimitiveType("int"))


# Geometry-stage built-ins (GLSL 4.60 section 7.1.4): a geometry stage writes the layer, viewport and primitive id of
# the primitive it emits and reads the incoming primitive id as gl_PrimitiveIDIn; a fragment stage reads all three.
_PRIMITIVE_BUILTINS = {
    "gl_Layer": (
        "int",
        "Layer",
        {"Geometry": "Output", "Fragment": "Input"},
        {"Geometry", "Fragment"},
    ),
    "gl_ViewportIndex": (
        "int",
        "ViewportIndex",
        {"Geometry": "Output", "Fragment": "Input"},
        {"Geometry", "Fragment"},
    ),
    "gl_PrimitiveIDIn": ("int", "PrimitiveId", "Input", {"Geometry"}),
    "gl_PrimitiveID": (
        "int",
        "PrimitiveId",
        {
            "Fragment": "Input",
            "Geometry": "Output",
            "TessellationControl": "Input",
            "TessellationEvaluation": "Input",
        },
        {"Fragment", "Geometry", "TessellationControl", "TessellationEvaluation"},
    ),
}


def per_vertex_builtin_info(codegen, name):
    """``stage_builtin_info`` entry for an arrayed per-vertex or a primitive built-in, or ``None`` for other names.

    The array length depends on the stage being emitted, so the entry is built when the built-in is first used.
    """
    if str(name) in _PRIMITIVE_BUILTINS:
        return _PRIMITIVE_BUILTINS[str(name)]
    parsed = _parse_name(str(name))
    if parsed is None:
        return None
    storage, member = parsed
    element_type, builtin = _MEMBERS[member]
    model = codegen.current_stage_builtin_execution_model()
    if storage == "Output":
        models = {"TessellationControl"}
        length = (
            codegen.tessellation_control_output_vertices(codegen.current_stage)
            if model == "TessellationControl"
            else 1
        )
    else:
        models = _INPUT_STAGES
        if model == "Geometry":
            length = _GEOMETRY_INPUT_VERTICES.get(
                codegen.geometry_input_mode(codegen.current_stage), 1
            )
        else:
            length = MAX_PATCH_VERTICES
    return f"{element_type}[{length}]", builtin, storage, models


def _per_vertex_stage(codegen):
    """Execution model and stage node a module-scope interface declaration belongs to, or ``(None, None)``."""
    model = codegen.current_execution_model
    if model is not None:
        return model, codegen.current_stage
    stages = list(getattr(codegen, "per_vertex_program_stages", None) or ())
    if len(stages) != 1:
        return None, None
    stage_type, stage = stages[0]
    return codegen.spirv_execution_model(codegen.stage_key(stage_type)), stage


def sized_per_vertex_interface_type(codegen, var_type, storage_class):
    """Size an unsized per-vertex interface array (``in vec3 normal[];``) as glslang does.

    A tessellation input gets ``gl_MaxPatchVertices`` elements, a geometry input one per vertex of the input
    primitive and a tessellation control output one per output control point. Any other type is returned as is.
    """
    array_info = codegen.array_type_info_from_type(var_type)
    if array_info is None or array_info[1] is not None or array_info[0] is None:
        return var_type
    model, stage = _per_vertex_stage(codegen)
    length = None
    if storage_class == "Input" and model in {
        "TessellationControl",
        "TessellationEvaluation",
    }:
        length = MAX_PATCH_VERTICES
    elif storage_class == "Input" and model == "Geometry":
        length = _GEOMETRY_INPUT_VERTICES.get(codegen.geometry_input_mode(stage))
    elif storage_class == "Output" and model == "TessellationControl":
        length = codegen.tessellation_control_output_vertices(stage)
    if length is None:
        return var_type
    return codegen.register_array_type(array_info[0], length)
