"""Ray payload structures declared with the access qualifiers HLSL shader model 6.7 requires.

From shader model 6.7 a structure used as a ray payload must be declared ``struct [raypayload] Name`` and each of
its fields qualified with the stages that read and write it, ``float4 color : read(caller) : write(closesthit)``
(DirectX Raytracing specification, "Payload access qualifiers"); DXC rejects an unannotated payload type for
``lib_6_7`` and later. Older shader models ignore the qualifiers, with a warning.

:func:`annotate_hlsl_ray_payloads` marks every structure the program uses as a payload with ``@raypayload``: the
type of a closest-hit, any-hit or miss entry's ``@payload`` parameter and the type of ``TraceRay``'s payload
argument. CrossGL carries the qualifiers as ``@ray_payload_read(stages)`` and ``@ray_payload_write(stages)``
member attributes, which the HLSL and Slang importers write; a field that carries them keeps them as written. A
field with none, as every field from a source language without qualifiers, gets ``read`` and ``write`` for all
four payload stages: those qualifiers promise nothing, so every stage still sees the field as it did before shader
model 6.7.

A ``TraceRay`` payload whose structure cannot be resolved, or that is not a structure, is refused with
:class:`HlslRayPayloadUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions, iter_structs
from .rewrite import child_slots, get_child

__all__ = [
    "HlslRayPayloadUnsupportedError",
    "PAYLOAD_ACCESS_ATTRIBUTES",
    "PAYLOAD_STAGES",
    "RAY_PAYLOAD_ATTRIBUTE",
    "annotate_hlsl_ray_payloads",
]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
S = A.ShaderStage
# Stages whose entry point receives the incoming payload as its ``@payload`` parameter.
_PAYLOAD_ENTRY_STAGES = frozenset({S.RAY_CLOSEST_HIT, S.RAY_ANY_HIT, S.RAY_MISS})
# The payload access qualifier attributes, by the HLSL keyword each one is written as.
PAYLOAD_ACCESS_ATTRIBUTES = {"ray_payload_read": "read", "ray_payload_write": "write"}
# Every stage a payload access qualifier can name.
PAYLOAD_STAGES = ("caller", "closesthit", "miss", "anyhit")
RAY_PAYLOAD_ATTRIBUTE = "raypayload"


class HlslRayPayloadUnsupportedError(TranslationDiagnostic):
    """A ray payload whose structure the HLSL generator cannot declare with payload access qualifiers."""

    def __init__(self, detail):
        self.missing_capabilities = ("directx.ray_payload_access_qualifiers",)
        super().__init__(
            f"DirectX ray payload: {detail}",
            target="directx",
            code=f"{CODE_PREFIX}directx-ray-payload-unsupported",
        )


def annotate_hlsl_ray_payloads(program, target) -> int:
    """Mark the structures ``program`` uses as ray payloads for an HLSL ``target``; return how many."""
    if (
        not isinstance(program, A.ShaderNode)
        or str(target or "").lower() not in _HLSL_TARGETS
    ):
        return 0
    structs = {}
    for struct in iter_structs(program):
        if isinstance(struct, A.StructNode):
            structs.setdefault(struct.name, struct)
    names = []
    for stage, function in _stage_functions(program):
        if function is getattr(stage, "entry_point", None) and (
            getattr(stage, "stage", None) in _PAYLOAD_ENTRY_STAGES
        ):
            for parameter in getattr(function, "parameters", None) or []:
                if _has_attribute(parameter, "payload"):
                    names.append(_struct_name(_declared_type(parameter)))
        for call in _trace_ray_calls(function):
            names.append(_payload_struct_name(program, stage, function, call))
    marked = 0
    for name in dict.fromkeys(name for name in names if name is not None):
        struct = structs.get(name)
        if struct is None:
            raise HlslRayPayloadUnsupportedError(
                f"the payload type '{name}' is not a structure; DXR requires a structure"
            )
        if _mark(struct):
            marked += 1
    return marked


def _stage_functions(program):
    """``(stage, function)`` for every function, ``stage`` being ``None`` for a module-scope function."""
    stages = getattr(program, "stages", None) or {}
    stages = list(stages.values()) if isinstance(stages, dict) else list(stages)
    owned = set()
    for stage in stages:
        for function in getattr(stage, "local_functions", None) or []:
            owned.add(id(function))
            yield stage, function
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            owned.add(id(entry))
            yield stage, entry
    for function in iter_functions(program):
        if id(function) not in owned:
            owned.add(id(function))
            yield None, function


def _walk(node):
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key))


def _trace_ray_calls(function):
    for node in _walk(function):
        if isinstance(node, A.RayTracingOpNode) and node.operation == "TraceRay":
            yield node
        elif (
            isinstance(node, A.FunctionCallNode)
            and _callee_name(node) == "TraceRay"
            and len(getattr(node, "arguments", None) or []) == 8
        ):
            yield node


def _callee_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, A.IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


def _payload_struct_name(program, stage, function, call):
    arguments = list(getattr(call, "arguments", None) or [])
    if len(arguments) != 8:
        # The GLSL form names the payload by location; lower_glsl_ray_interfaces rewrote every such call earlier.
        raise HlslRayPayloadUnsupportedError(
            f"TraceRay takes 8 arguments in DXR, this call has {len(arguments)}"
        )
    payload = arguments[7]
    base, element = payload, False
    while isinstance(base, A.ArrayAccessNode):
        base, element = base.array, True
    if isinstance(base, A.IdentifierNode):
        declared = _variable_type(program, stage, function, base.name)
        if declared is not None:
            return _struct_name(declared, element=element)
    raise HlslRayPayloadUnsupportedError(
        "the payload argument of TraceRay must name a variable or an element of an array variable"
    )


def _variable_type(program, stage, function, name):
    """The declared type of the variable ``name`` visible in ``function``, innermost scope first."""
    for parameter in getattr(function, "parameters", None) or []:
        if getattr(parameter, "name", None) == name:
            return _declared_type(parameter)
    for node in _walk(function):
        if isinstance(node, A.VariableNode) and node.name == name:
            declared = _declared_type(node)
            if declared is not None:
                return declared
    scopes = []
    if stage is not None:
        scopes.append(getattr(stage, "local_variables", None) or [])
    scopes.append(getattr(program, "global_variables", None) or [])
    for scope in scopes:
        for variable in scope:
            if getattr(variable, "name", None) == name:
                return _declared_type(variable)
    return None


def _declared_type(node):
    for field in ("vtype", "var_type", "param_type", "member_type"):
        value = getattr(node, field, None)
        if value is not None:
            return value
    return None


def _struct_name(type_node, element=False):
    """The structure name of ``type_node``, or of its element type when ``element`` is set."""
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, A.NamedType):
        return type_node.name
    if isinstance(type_node, str):
        text = type_node.strip()
        return text.split("[", 1)[0].strip() if element else text
    return None if type_node is None else str(type_node)


def _has_attribute(node, name):
    return any(
        str(getattr(attribute, "name", "")).lower() == name
        for attribute in getattr(node, "attributes", None) or []
    )


def _mark(struct):
    """Mark ``struct`` as a payload and qualify each unqualified field for every stage; return whether it changed.

    DXC requires every field of a payload structure to carry access qualifiers. A field without any is read and
    written by every stage, as every field was before shader model 6.7.
    """
    changed = False
    if not _has_attribute(struct, RAY_PAYLOAD_ATTRIBUTE):
        struct.attributes = list(getattr(struct, "attributes", None) or []) + [
            A.AttributeNode(RAY_PAYLOAD_ATTRIBUTE, [])
        ]
        changed = True
    for member in getattr(struct, "members", None) or []:
        if any(_has_attribute(member, name) for name in PAYLOAD_ACCESS_ATTRIBUTES):
            continue
        member.attributes = list(getattr(member, "attributes", None) or []) + [
            A.AttributeNode(name, [A.IdentifierNode(stage) for stage in PAYLOAD_STAGES])
            for name in PAYLOAD_ACCESS_ATTRIBUTES
        ]
        changed = True
    return changed
