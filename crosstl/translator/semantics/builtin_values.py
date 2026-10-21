"""Built-in value attributes spelled the way the target's generator reads them.

CrossGL accepts three spellings for a built-in value on a parameter, structure member, stage variable or entry-point
return: the GLSL variable name (``@gl_GlobalInvocationID``), the HLSL system-value semantic
(``@SV_DispatchThreadID``) and WGSL's ``@builtin(global_invocation_id)`` (WGSL section 13.3.1.1, "Built-in Values").
The validator treats the three as one built-in (``validation.BUILTIN_SEMANTIC_METADATA_NAMES``), but each generator
read its own subset. None read the WGSL form: HLSL emitted ``: builtin`` semantics, Metal ``[[builtin]]`` attributes
or user varyings with a zero position, and GLSL undeclared names or user varyings. GLSL also read ``@SV_Position``
as a user varying, so the vertex stage never wrote ``gl_Position``, and HLSL emitted ``@gl_SampleID`` verbatim.

:func:`canonicalize_builtin_values` rewrites each such attribute to the spelling the target's generator reads:

* for HLSL targets, the system-value semantic (HLSL "Semantics", System-Value Semantics); an ``SV_`` spelling is
  kept as written, because HLSL semantics are case-insensitive;
* for GLSL, SPIR-V and Metal, the GLSL built-in variable (GLSL 4.60 section 7.1, "Built-In Language Variables");
  a ``gl_`` spelling is kept as written. GLSL names the sample mask ``gl_SampleMaskIn`` as a fragment input and
  ``gl_SampleMask`` as an output, so a sample mask takes the name of the direction it is declared in.

A WGSL built-in that CrossGL does not name, one the target has no built-in for (Direct3D 10 and later have no
point sprites, so HLSL has no point-sprite coordinate), and a sample mask on a structure used both as a stage input
and as a stage output are refused with :class:`BuiltinValueUnsupportedError`. HLSL has no system value for the
dispatch's workgroup count either; that one keeps its GLSL name, which the HLSL generator reads from a constant
buffer the host fills. Other targets keep their
spellings; their generators read the WGSL form or refuse it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions, iter_structs

__all__ = [
    "BuiltinValueUnsupportedError",
    "canonicalize_builtin_values",
    "glsl_builtin_name",
]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_GLSL_NAME_TARGETS = frozenset(
    {"opengl", "glsl", "webgl", "gles", "essl", "vulkan", "spirv", "metal"}
)


class BuiltinValueUnsupportedError(TranslationDiagnostic):
    """A built-in value attribute the target cannot express, or whose name CrossGL does not define."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}builtin-value-unsupported",
        )


@dataclass(frozen=True)
class _Builtin:
    glsl: str
    hlsl: str | None
    glsl_input: str | None = (
        None  # the GLSL name of a fragment input that differs from the output's
    )


# HLSL spellings are the ones the HLSL generator writes for the GLSL names, so a program spelled with GLSL names
# keeps its HLSL output.
_BUILTINS = {
    "position": _Builtin("gl_Position", "SV_POSITION"),
    "frag_coord": _Builtin("gl_FragCoord", "SV_Position"),
    "point_size": _Builtin("gl_PointSize", "PSIZE"),
    "vertex_index": _Builtin("gl_VertexID", "SV_VertexID"),
    "instance_index": _Builtin("gl_InstanceID", "SV_InstanceID"),
    "front_facing": _Builtin("gl_FrontFacing", "SV_IsFrontFace"),
    "frag_depth": _Builtin("gl_FragDepth", "SV_DEPTH"),
    "sample_index": _Builtin("gl_SampleID", "SV_SampleIndex"),
    "sample_mask": _Builtin("gl_SampleMask", "SV_Coverage", "gl_SampleMaskIn"),
    "primitive_id": _Builtin("gl_PrimitiveID", "SV_PrimitiveID"),
    "point_coord": _Builtin("gl_PointCoord", None),
    "clip_distances": _Builtin("gl_ClipDistance", "SV_ClipDistance"),
    "global_invocation_id": _Builtin("gl_GlobalInvocationID", "SV_DispatchThreadID"),
    "local_invocation_id": _Builtin("gl_LocalInvocationID", "SV_GroupThreadID"),
    "local_invocation_index": _Builtin("gl_LocalInvocationIndex", "SV_GroupIndex"),
    "workgroup_id": _Builtin("gl_WorkGroupID", "SV_GroupID"),
    # HLSL has no system value for the dispatch's group count; the HLSL generator reads gl_NumWorkGroups from a
    # dispatch-information constant buffer the host fills.
    "num_workgroups": _Builtin("gl_NumWorkGroups", "gl_NumWorkGroups"),
}

# ``@builtin(name)`` names: WGSL's built-in values and the CrossGL names the validator accepts.
_WGSL_NAMES = {name: name for name in _BUILTINS}
_WGSL_NAMES.update(
    {
        "vertex_id": "vertex_index",
        "vertexid": "vertex_index",
        "instance_id": "instance_index",
        "instanceid": "instance_index",
        "primitive_index": "primitive_id",
        "primitiveid": "primitive_id",
        "sample_id": "sample_index",
    }
)

# Lower-cased GLSL and HLSL spellings.
_SPELLINGS = {
    "gl_position": "position",
    "gl_fragcoord": "frag_coord",
    "gl_pointsize": "point_size",
    "gl_vertexid": "vertex_index",
    "gl_instanceid": "instance_index",
    "gl_frontfacing": "front_facing",
    "gl_fragdepth": "frag_depth",
    "gl_sampleid": "sample_index",
    "gl_samplemask": "sample_mask",
    "gl_samplemaskin": "sample_mask",
    "gl_primitiveid": "primitive_id",
    "gl_pointcoord": "point_coord",
    "gl_clipdistance": "clip_distances",
    "gl_globalinvocationid": "global_invocation_id",
    "gl_localinvocationid": "local_invocation_id",
    "gl_localinvocationindex": "local_invocation_index",
    "gl_workgroupid": "workgroup_id",
    "gl_numworkgroups": "num_workgroups",
    # SV_Position names the clip-space position a vertex stage writes and the window position a fragment stage
    # reads; every GLSL-name generator reads gl_Position on a fragment input as gl_FragCoord.
    "sv_position": "position",
    "sv_vertexid": "vertex_index",
    "sv_instanceid": "instance_index",
    "sv_isfrontface": "front_facing",
    "sv_depth": "frag_depth",
    "sv_sampleindex": "sample_index",
    "sv_coverage": "sample_mask",
    "sv_primitiveid": "primitive_id",
    "sv_clipdistance": "clip_distances",
    "sv_dispatchthreadid": "global_invocation_id",
    "sv_groupthreadid": "local_invocation_id",
    "sv_groupindex": "local_invocation_index",
    "sv_groupid": "workgroup_id",
}

_INPUT, _OUTPUT, _BOTH = "input", "output", "input and output"


def canonicalize_builtin_values(program, target) -> int:
    """Rewrite built-in value attributes for ``target`` in place; returns how many were rewritten.

    Raises :class:`BuiltinValueUnsupportedError` for a built-in the target cannot express or CrossGL does not
    name.
    """
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or (
        target not in _HLSL_TARGETS and target not in _GLSL_NAME_TARGETS
    ):
        return 0
    rewriter = _Rewriter(target, _struct_directions(program))
    for function in iter_functions(program):
        name = getattr(function, "name", "function")
        rewriter.rewrite(function, f"the return value of '{name}'", _OUTPUT)
        for parameter in getattr(function, "parameters", None) or []:
            rewriter.rewrite(
                parameter,
                f"parameter '{getattr(parameter, 'name', '')}' of '{name}'",
                _parameter_direction(parameter),
            )
    for struct in iter_structs(program):
        direction = rewriter.directions.get(getattr(struct, "name", None))
        for member in getattr(struct, "members", None) or []:
            rewriter.rewrite(
                member,
                f"member '{getattr(member, 'name', '')}' of '{getattr(struct, 'name', '')}'",
                direction,
            )
    for variable in _variables(program):
        rewriter.rewrite(
            variable,
            f"variable '{getattr(variable, 'name', '')}'",
            _variable_direction(variable),
        )
    return rewriter.rewritten


class _Rewriter:
    def __init__(self, target, directions):
        self.target = target
        self.hlsl = target in _HLSL_TARGETS
        self.directions = directions
        self.rewritten = 0

    def rewrite(self, node, context, direction):
        attributes = getattr(node, "attributes", None)
        if not attributes:
            return
        for index, attribute in enumerate(list(attributes)):
            spelling = self._spelling(attribute, context, direction)
            if spelling is not None and spelling != getattr(attribute, "name", None):
                attributes[index] = A.AttributeNode(spelling, [])
                self.rewritten += 1

    def _spelling(self, attribute, context, direction):
        """The attribute name ``attribute`` takes for the target, or ``None`` to keep it."""
        raw = str(getattr(attribute, "name", "") or "")
        lowered = raw.lower()
        if lowered == "builtin":
            written = _argument_name(attribute)
            canonical = _WGSL_NAMES.get(str(written or "").lower())
            if canonical is None:
                raise BuiltinValueUnsupportedError(
                    f"Unknown built-in value @builtin({written}) on {context}: CrossGL names "
                    f"{', '.join(sorted(_BUILTINS))}",
                    self.target,
                )
            return self._target_spelling(
                canonical, f"@builtin({written})", context, direction
            )
        canonical = _SPELLINGS.get(lowered)
        if canonical is None:
            return None
        if self.hlsl and lowered.startswith("sv_"):
            return None
        if not self.hlsl and lowered.startswith("gl_"):
            return None
        return self._target_spelling(canonical, f"@{raw}", context, direction)

    def _target_spelling(self, canonical, written, context, direction):
        builtin = _BUILTINS[canonical]
        if self.hlsl:
            if builtin.hlsl is None:
                raise BuiltinValueUnsupportedError(
                    f"HLSL has no system value for the built-in {written} on {context}",
                    self.target,
                )
            return builtin.hlsl
        if builtin.glsl_input is None:
            return builtin.glsl
        if direction == _INPUT:
            return builtin.glsl_input
        if direction == _OUTPUT:
            return builtin.glsl
        raise BuiltinValueUnsupportedError(
            f"The built-in {written} on {context} is a fragment input in GLSL "
            f"({builtin.glsl_input}) and an output ({builtin.glsl}), and its structure is used as "
            f"{direction or 'neither'}",
            self.target,
        )


def glsl_builtin_name(attribute):
    """The GLSL built-in variable ``attribute`` names in any CrossGL spelling, or ``None`` for a user attribute.

    A ``gl_`` spelling is returned as written. A sample mask is named by its output, ``gl_SampleMask``.
    """
    raw = str(getattr(attribute, "name", "") or "")
    if raw.startswith("gl_"):
        return raw
    lowered = raw.lower()
    if lowered == "builtin":
        canonical = _WGSL_NAMES.get(str(_argument_name(attribute) or "").lower())
    else:
        canonical = _SPELLINGS.get(lowered)
    return None if canonical is None else _BUILTINS[canonical].glsl


def _argument_name(attribute):
    arguments = getattr(attribute, "arguments", None) or []
    if not arguments:
        return None
    argument = arguments[0]
    for field in ("name", "value"):
        value = getattr(argument, field, None)
        if value is not None:
            return value
    return argument if isinstance(argument, str) else None


def _type_name(type_node):
    name = getattr(type_node, "name", None)
    return name if isinstance(name, str) else str(type_node or "")


def _struct_directions(program):
    """Each structure a stage's entry point takes (input) or returns (output), by name."""
    struct_names = {getattr(struct, "name", None) for struct in iter_structs(program)}
    directions = {}

    def note(type_node, direction):
        name = _type_name(type_node)
        if name not in struct_names:
            return
        previous = directions.get(name)
        directions[name] = direction if previous in (None, direction) else _BOTH

    for stage in _stages(program):
        function = getattr(stage, "entry_point", None)
        if function is None:
            continue
        note(getattr(function, "return_type", None), _OUTPUT)
        for parameter in getattr(function, "parameters", None) or []:
            note(
                getattr(parameter, "param_type", None), _parameter_direction(parameter)
            )
    return directions


def _qualifiers(node):
    names = {
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", None) or []
    }
    names.update(
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", None) or []
    )
    return names


def _parameter_direction(parameter):
    qualifiers = _qualifiers(parameter)
    if "inout" in qualifiers:
        return _BOTH
    if "out" in qualifiers:
        return _OUTPUT
    return _INPUT


def _variable_direction(variable):
    qualifiers = _qualifiers(variable)
    if "out" in qualifiers or "output" in qualifiers:
        return _OUTPUT
    if "in" in qualifiers or "input" in qualifiers:
        return _INPUT
    return None


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _variables(program):
    yield from getattr(program, "global_variables", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_variables", None) or []
