"""Stage interface arrays that GLSL declares without an element count: give them the count the stage implies.

GLSL lets some ``in`` and ``out`` arrays omit their size because the pipeline fixes it (GLSL 4.60 sections 4.3.4,
4.3.6 and 7.1; ``GL_EXT_mesh_shader``; ``GL_EXT_fragment_shader_barycentric``):

* a geometry shader input has one element per vertex of the input primitive in its ``layout(...) in`` qualifier:
  1 for ``points``, 2 for ``lines``, 4 for ``lines_adjacency``, 3 for ``triangles`` and 6 for
  ``triangles_adjacency``;
* a tessellation control output has one element per output patch vertex, ``layout(vertices = N) out``;
* a mesh shader output has ``max_vertices`` elements, or ``max_primitives`` when it is ``perprimitive``;
* a ``pervertex`` fragment input has one element per vertex of the triangle, 3;
* ``gl_ClipDistance`` and ``gl_CullDistance`` are sized by the largest constant index the shader writes them with.

:func:`size_stage_io_arrays` writes those counts into the declarations, so that every target declares the same
array (HLSL ``triangle T input[3]``, Metal and SPIR-V arrays need a constant count).

A tessellation control or evaluation shader input that is not ``patch`` is the one stage array whose count only the
pipeline knows: GLSL sizes it ``gl_MaxPatchVertices`` and only ``gl_PatchVerticesIn`` elements are valid. It stays
unsized, and :func:`size_stage_io_arrays` returns it so that :mod:`.unsized_arrays` does not refuse it; an emitter
that cannot express a pipeline-sized patch refuses it itself. A stage array whose count cannot be found (no
primitive layout, a clip distance indexed by a non-constant, a declaration several stages could own) is left
unsized, and :mod:`.unsized_arrays` refuses it as before.

:func:`refuse_per_vertex_inputs` refuses ``pervertex`` fragment inputs for targets whose emitters do not lower them:
GLSL and SPIR-V carry them (``pervertexEXT``, ``PerVertexKHR``), but the other emitters dropped the qualifier and
declared one interpolated input per element, a different value from the one the source reads. HLSL can express them
with ``nointerpolation`` inputs read through ``GetAttributeAtVertex`` (shader model 6.1); that lowering is not
implemented yet.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions, iter_structs
from .storage_blocks import _walk

__all__ = [
    "PerVertexInputUnsupportedError",
    "refuse_per_vertex_inputs",
    "size_stage_io_arrays",
]

_PRIMITIVE_VERTICES = {
    "points": 1,
    "lines": 2,
    "lines_adjacency": 4,
    "triangles": 3,
    "triangles_adjacency": 6,
}
_DISTANCE_ARRAYS = frozenset({"gl_ClipDistance", "gl_CullDistance"})
_PER_VERTEX = frozenset({"pervertex", "pervertexext", "pervertexnv"})
_PER_PRIMITIVE = frozenset({"perprimitive", "perprimitiveext", "perprimitivenv"})
_PIPELINE_SIZED = object()

S = A.ShaderStage
# Targets whose emitters declare per-vertex fragment inputs.
_PER_VERTEX_TARGETS = frozenset(
    {"opengl", "glsl", "webgl", "vulkan", "spirv", "cgl", "crossgl"}
)


class PerVertexInputUnsupportedError(TranslationDiagnostic):
    """A ``pervertex`` fragment input the target's emitter cannot declare."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.per-vertex-fragment-input",)
        super().__init__(
            f"per-vertex fragment input '{name}' (GL_EXT_fragment_shader_barycentric pervertexEXT) has no "
            f"lowering for {target}: each element is one vertex's value before interpolation, which an "
            "interpolated input array does not carry",
            target=target,
            code=f"{CODE_PREFIX}{target}-per-vertex-input-unsupported",
        )


def _names(node):
    """Lower-case qualifier and attribute names of a declaration."""
    names = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    names.update(
        str(getattr(a, "name", "")).lower()
        for a in getattr(node, "attributes", None) or []
    )
    return names


def _unsized(type_node):
    return isinstance(type_node, A.ArrayType) and type_node.size is None


def _int_value(node):
    if isinstance(node, A.LiteralNode):
        value = node.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip().rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


def _set_size(declaration, count):
    sized = A.ArrayType(
        declaration_type(declaration).element_type,
        A.LiteralNode(count, A.PrimitiveType("int")),
    )
    if isinstance(declaration, A.StructMemberNode):
        declaration.member_type = sized
    else:
        declaration.var_type = sized
        if hasattr(declaration, "vtype"):
            declaration.vtype = sized


def declaration_type(declaration):
    if isinstance(declaration, A.StructMemberNode):
        return declaration.member_type
    return getattr(declaration, "var_type", None)


def _stage_kind(key, stage):
    kind = getattr(stage, "stage", None) or key
    if isinstance(kind, S):
        return kind
    try:
        return S(str(kind).lower())
    except ValueError:
        return None


def _layout(stage, direction):
    """``{entry name: first integer argument or None}`` of a stage's ``layout(...) in|out`` qualifiers."""
    entries = {}
    for qualifier in getattr(stage, "layout_qualifiers", None) or []:
        if getattr(qualifier, "direction", None) != direction:
            continue
        for entry in qualifier.entries or []:
            arguments = getattr(entry, "arguments", None) or []
            entries[str(entry.name).lower()] = (
                _int_value(arguments[0]) if arguments else None
            )
    return entries


def _stage_count(kind, stage, direction, names):
    """The element count a stage gives its unsized ``direction`` arrays, ``_PIPELINE_SIZED`` or ``None``."""
    if "patch" in names:
        return None
    if direction == "in" and kind is S.GEOMETRY:
        primitives = [
            _PRIMITIVE_VERTICES[name]
            for name in _layout(stage, "in")
            if name in _PRIMITIVE_VERTICES
        ]
        return primitives[0] if len(primitives) == 1 else None
    if direction == "in" and kind in (
        S.TESSELLATION_CONTROL,
        S.TESSELLATION_EVALUATION,
    ):
        return _PIPELINE_SIZED
    if direction == "out" and kind is S.TESSELLATION_CONTROL:
        return _layout(stage, "out").get("vertices")
    if direction == "out" and kind is S.MESH:
        key = "max_primitives" if names & _PER_PRIMITIVE else "max_vertices"
        return _layout(stage, "out").get(key)
    return None


class _Sizer:
    def __init__(self, program):
        self.program = program
        stages = getattr(program, "stages", None) or {}
        items = stages.items() if isinstance(stages, dict) else enumerate(stages)
        self.stages = [
            (kind, stage)
            for kind, stage in ((_stage_kind(k, s), s) for k, s in items)
            if kind is not None
        ]
        self.pipeline_sized = set()

    def run(self):
        for variable in getattr(self.program, "global_variables", None) or []:
            self.size_interface(variable, self.stages)
        for kind, stage in self.stages:
            for variable in getattr(stage, "local_variables", None) or []:
                self.size_interface(variable, [(kind, stage)])
        self.size_distances()
        return self.pipeline_sized

    def size_interface(self, variable, owners):
        if not isinstance(variable, A.VariableNode) or not _unsized(
            declaration_type(variable)
        ):
            return
        names = _names(variable)
        if names & _PER_VERTEX:
            _set_size(variable, 3)
            return
        direction = "in" if "in" in names else "out" if "out" in names else None
        if direction is None:
            return
        counts = {
            _stage_count(kind, stage, direction, names) for kind, stage in owners
        } - {None}
        if len(counts) != 1:
            return
        (count,) = counts
        if count is _PIPELINE_SIZED:
            self.pipeline_sized.add(id(variable))
        elif count > 0:
            _set_size(variable, count)

    def size_distances(self):
        members = [
            member
            for struct in iter_structs(self.program)
            if isinstance(struct, A.StructNode)
            for member in struct.members or []
            if isinstance(member, A.StructMemberNode)
        ]
        for member in members:
            if _unsized(member.member_type) and _names(member) & _PER_VERTEX:
                _set_size(member, 3)
        declarations = [
            member
            for member in members
            if _unsized(member.member_type) and self.distance_name(member)
        ] + [
            variable
            for variable in getattr(self.program, "global_variables", None) or []
            if isinstance(variable, A.VariableNode)
            and _unsized(declaration_type(variable))
            and variable.name in _DISTANCE_ARRAYS
        ]
        if not declarations:
            return
        largest = self.distance_indices()
        for declaration in declarations:
            count = largest.get(self.distance_name(declaration) or declaration.name)
            if count:
                _set_size(declaration, count)

    @staticmethod
    def distance_name(declaration):
        if declaration.name in _DISTANCE_ARRAYS:
            return declaration.name
        for attribute in getattr(declaration, "attributes", None) or []:
            if getattr(attribute, "name", None) in _DISTANCE_ARRAYS:
                return attribute.name
        return None

    def distance_indices(self):
        """``{gl_ClipDistance|gl_CullDistance: largest constant index + 1}``; ``0`` once any index is not constant."""
        member_names = {}
        for struct in iter_structs(self.program):
            for member in getattr(struct, "members", None) or []:
                name = (
                    self.distance_name(member)
                    if isinstance(member, A.StructMemberNode)
                    else None
                )
                if name:
                    member_names[member.name] = name
        largest = {}
        for function in iter_functions(self.program):
            for node in _walk(function.body):
                if not isinstance(node, A.ArrayAccessNode):
                    continue
                array = node.array_expr
                if isinstance(array, A.MemberAccessNode):
                    name = member_names.get(array.member)
                elif isinstance(array, A.IdentifierNode):
                    name = array.name if array.name in _DISTANCE_ARRAYS else None
                else:
                    name = None
                if name is None:
                    continue
                index = _int_value(node.index_expr)
                if index is None or largest.get(name) == 0:
                    largest[name] = 0
                else:
                    largest[name] = max(largest.get(name, 0), index + 1)
        return largest


def size_stage_io_arrays(program) -> set:
    """Size the unsized stage interface arrays of ``program`` from their stage, in place.

    Returns the ``id()`` of every tessellation input left unsized because the pipeline sizes it.
    """
    if not isinstance(program, A.ShaderNode):
        return set()
    return _Sizer(program).run()


def refuse_per_vertex_inputs(program, target) -> None:
    """Raise :class:`PerVertexInputUnsupportedError` for a ``pervertex`` input when ``target`` cannot declare it."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target in _PER_VERTEX_TARGETS:
        return
    declarations = [
        member
        for struct in iter_structs(program)
        for member in getattr(struct, "members", None) or []
    ]
    declarations += list(getattr(program, "global_variables", None) or [])
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        declarations += list(getattr(stage, "local_variables", None) or [])
    for declaration in declarations:
        if _names(declaration) & _PER_VERTEX:
            raise PerVertexInputUnsupportedError(
                target,
                getattr(declaration, "name", "?"),
                source_location=getattr(declaration, "source_location", None),
            )
