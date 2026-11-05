"""Direct3D's vertex ID, which counts from the draw's base vertex, on targets whose vertex index includes the base.

Direct3D's ``SV_VertexID`` excludes the draw's base vertex: a non-indexed draw numbers its vertices from 0 whatever
its ``StartVertexLocation``, and an indexed draw delivers the index buffer value without ``BaseVertexLocation``.
OpenGL's ``gl_VertexID`` (GLSL 4.60 section 7.1.1), Vulkan's ``VertexIndex`` and MSL's ``[[vertex_id]]`` include
it: they start at the first vertex of a non-indexed draw and add the base vertex to the index of an indexed one.
DXC (``-fvk-support-nonzero-base-vertex``), slangc, vkd3d-shader and DXVK therefore lower ``SV_VertexID`` to
``VertexIndex - BaseVertex``. The HLSL and Slang importers read ``SV_VertexID`` as ``gl_VertexID``, so every other
target read the vertex ``BaseVertex`` positions further on whenever the draw had a base vertex.

CrossGL spells the Direct3D value ``gl_RelativeVertexID``, an ``int`` like ``gl_VertexID``. :func:`lower_relative_vertex_ids`
spells it for the target, in place:

* Direct3D and Slang targets read it as ``gl_VertexID``, which their generators write as ``SV_VertexID``;
* every other target reads the base-inclusive index ``gl_VertexIndex`` and the base vertex ``gl_BaseVertex``
  (GLSL 4.60 section 7.1.1; SPIR-V ``BuiltIn BaseVertex``, capability ``DrawParameters``; MSL ``[[base_vertex]]``)
  and takes their difference. A vertex entry point's parameter ``T v @ gl_RelativeVertexID`` becomes a
  ``gl_VertexIndex`` parameter under a fresh name, the entry point takes the base vertex (reusing a parameter that
  already carries it) and declares ``T v = T(index) - T(base);`` first. A stage input structure member carrying the
  value carries ``gl_VertexIndex`` instead, and the entry point copies the structure parameter into a local of the
  original name whose member it rebases first. A read of the built-in by name becomes
  ``gl_VertexIndex - gl_BaseVertex``.

Vulkan's ``BaseVertex`` and MSL's ``[[base_vertex]]`` hold the first vertex of a non-indexed draw, so the difference is
Direct3D's value for every draw. OpenGL's ``gl_BaseVertex`` is zero for a command without a ``basevertex`` parameter
(GLSL 4.60 section 7.1.1), so OpenGL output matches Direct3D for indexed draws and for non-indexed draws from vertex
0; ``glDrawArrays`` from vertex ``first`` still reads ``first`` more.

Refused with :class:`RelativeVertexIdUnsupportedError`: the value on WebGL and WGSL, which cannot read the base vertex
(GLSL ES 3.00 declares no ``gl_BaseVertex``, WGSL has no built-in value for it), and a member of a nested structure
carrying it. A shader whose draws all have a zero base vertex can say so to the HLSL and Slang importers
(``source_options={"zero_base_vertex": True}``); they then read ``SV_VertexID`` as ``gl_VertexID``, which every
target expresses.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .builtin_values import glsl_builtin_name
from .hub_types import iter_functions, iter_structs
from .rewrite import child_slots, get_child, set_child

__all__ = [
    "RELATIVE_VERTEX_ID",
    "RelativeVertexIdUnsupportedError",
    "lower_relative_vertex_ids",
]

RELATIVE_VERTEX_ID = "gl_RelativeVertexID"
_VERTEX_INDEX = "gl_VertexIndex"
_BASE_VERTEX = "gl_BaseVertex"
_DIRECT3D_SPELLING = "gl_VertexID"
_DIRECT3D_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "slang"}
)
_CROSSGL_TARGETS = frozenset({"cgl", "crossgl"})
# Targets that cannot read the base vertex: GLSL ES 3.00 declares no gl_BaseVertex, and WGSL has no built-in
# value for it.
_NO_BASE_VERTEX = {
    "webgl": "GLSL ES 3.00 has no base vertex",
    "wgsl": "WGSL has no base vertex built-in value",
}


class RelativeVertexIdUnsupportedError(TranslationDiagnostic):
    """A Direct3D vertex ID in a position the base-vertex rewrite cannot reach."""

    def __init__(self, detail, target):
        super().__init__(
            f"{RELATIVE_VERTEX_ID} (Direct3D SV_VertexID): {detail}",
            target=target,
            code=f"{CODE_PREFIX}relative-vertex-id-unsupported",
        )


def _carries(node, name):
    return any(
        glsl_builtin_name(attribute) == name
        for attribute in getattr(node, "attributes", None) or []
    )


def _respell(node, name, spelling):
    for attribute in getattr(node, "attributes", None) or []:
        if glsl_builtin_name(attribute) == name:
            attribute.name = spelling


def _type_name(type_node):
    return str(getattr(type_node, "name", type_node) or "")


def _names(function):
    names = {
        getattr(parameter, "name", None) for parameter in function.parameters or []
    }
    stack = [getattr(function, "body", None)]
    while stack:
        node = stack.pop()
        if isinstance(node, A.VariableNode):
            names.add(node.name)
        if isinstance(node, A.ASTNode):
            stack.extend(
                get_child(node, field, key) for field, key in child_slots(node)
            )
    return names


def _fresh(taken, stem):
    name, index = stem, 1
    while name in taken:
        index += 1
        name = f"{stem}{index}"
    taken.add(name)
    return name


def _convert(type_node, expression):
    return A.FunctionCallNode(A.IdentifierNode(_type_name(type_node)), [expression])


def _rebased(type_node, index, base):
    """``index - T(base)`` for an ``index`` of type ``T``."""
    return A.BinaryOpNode(index, "-", _convert(type_node, A.IdentifierNode(base)))


def _respell_reads(node, spelling):
    """Replace every read of the built-in by name below ``node`` with ``spelling(node)``."""
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        if isinstance(child, A.IdentifierNode) and child.name == RELATIVE_VERTEX_ID:
            set_child(node, field, key, spelling())
        elif (
            isinstance(child, A.BuiltinVariableNode)
            and child.builtin_name == RELATIVE_VERTEX_ID
        ):
            set_child(node, field, key, spelling())
        elif isinstance(child, A.ASTNode):
            _respell_reads(child, spelling)


class _Lowering:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.structs = {struct.name: struct for struct in iter_structs(program)}

    def refuse_missing_base(self, where):
        reason = _NO_BASE_VERTEX.get(self.target)
        if reason is not None:
            raise RelativeVertexIdUnsupportedError(
                f"{where} reads it, which excludes the draw's base vertex while the target's vertex index "
                f"includes it, and {reason} to subtract; declare source_options={{'zero_base_vertex': True}} if "
                "every draw has a zero base vertex",
                self.target,
            )

    def member(self, struct):
        """The member of ``struct`` carrying the value, refusing one in a nested structure."""
        found = None
        if struct is None:
            return None
        for member in getattr(struct, "members", None) or []:
            if _carries(member, RELATIVE_VERTEX_ID):
                found = member
                continue
            nested = self.structs.get(_type_name(getattr(member, "member_type", None)))
            if nested is not None and nested is not struct and self.member(nested):
                raise RelativeVertexIdUnsupportedError(
                    f"structure '{struct.name}' carries it in its nested member '{member.name}'; pass it as an "
                    "entry-point parameter or a member of the stage input structure itself",
                    self.target,
                )
        return found

    def lower_entry(self, entry):
        parameters = list(entry.parameters or [])
        direct, wrapped = [], []
        for parameter in parameters:
            if _carries(parameter, RELATIVE_VERTEX_ID):
                direct.append(parameter)
                continue
            struct = self.structs.get(_type_name(parameter.param_type))
            member = self.member(struct)
            if member is not None:
                wrapped.append((parameter, member))
        if not direct and not wrapped:
            return
        self.refuse_missing_base(f"vertex entry point '{entry.name}'")
        taken = _names(entry)
        base = next((p.name for p in parameters if _carries(p, _BASE_VERTEX)), None)
        if base is None:
            base = _fresh(taken, "cgl_base_vertex")
            if entry.parameters is None:
                entry.parameters = []
            entry.parameters.append(
                A.ParameterNode(
                    base,
                    A.PrimitiveType("int"),
                    attributes=[A.AttributeNode(_BASE_VERTEX)],
                )
            )
        prologue = []
        for parameter in direct:
            index = _fresh(taken, f"{parameter.name}_index")
            prologue.append(
                A.VariableNode(
                    parameter.name,
                    copy.deepcopy(parameter.param_type),
                    initial_value=_rebased(
                        parameter.param_type, A.IdentifierNode(index), base
                    ),
                )
            )
            parameter.name = index
            _respell(parameter, RELATIVE_VERTEX_ID, _VERTEX_INDEX)
        for parameter, member in wrapped:
            value = _fresh(taken, f"cgl_{parameter.name}")
            prologue.append(
                A.VariableNode(
                    parameter.name,
                    copy.deepcopy(parameter.param_type),
                    initial_value=A.IdentifierNode(value),
                )
            )
            field = A.MemberAccessNode(A.IdentifierNode(parameter.name), member.name)
            prologue.append(
                A.AssignmentNode(
                    field,
                    _rebased(member.member_type, copy.deepcopy(field), base),
                )
            )
            parameter.name = value
        entry.body.statements[0:0] = prologue

    def run(self):
        for stage in _stages(self.program):
            entry = getattr(stage, "entry_point", None)
            if entry is not None and isinstance(
                getattr(entry, "body", None), A.BlockNode
            ):
                self.lower_entry(entry)
        for struct in iter_structs(self.program):
            for member in getattr(struct, "members", None) or []:
                _respell(member, RELATIVE_VERTEX_ID, _VERTEX_INDEX)
        for function in iter_functions(self.program):
            _respell_reads(function, lambda: self.read_by_name(function))

    def read_by_name(self, function):
        self.refuse_missing_base(f"function '{function.name}'")
        return A.BinaryOpNode(
            A.IdentifierNode(_VERTEX_INDEX), "-", A.IdentifierNode(_BASE_VERTEX)
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _respell_direct3d(program):
    for function in iter_functions(program):
        for parameter in function.parameters or []:
            _respell(parameter, RELATIVE_VERTEX_ID, _DIRECT3D_SPELLING)
        _respell_reads(function, lambda: A.IdentifierNode(_DIRECT3D_SPELLING))
    for struct in iter_structs(program):
        for member in getattr(struct, "members", None) or []:
            _respell(member, RELATIVE_VERTEX_ID, _DIRECT3D_SPELLING)


def lower_relative_vertex_ids(program, target) -> None:
    """Spell ``program``'s ``gl_RelativeVertexID`` for ``target``, in place."""
    target = str(target or "").lower()
    if target in _CROSSGL_TARGETS or not isinstance(program, A.ShaderNode):
        return
    if target in _DIRECT3D_TARGETS:
        _respell_direct3d(program)
        return
    _Lowering(program, target).run()
