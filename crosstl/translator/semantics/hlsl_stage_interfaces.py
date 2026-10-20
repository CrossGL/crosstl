"""Stage interfaces that HLSL declares differently from GLSL and SPIR-V.

GLSL (``GL_EXT_mesh_shader``, ``GL_EXT_ray_tracing``) and SPIR-V pass a mesh or ray stage's interface through
module-scope variables; HLSL passes it through the entry point and through intrinsic arguments (HLSL shader model
6.5 mesh and amplification shaders, DXR 1.0 shader signatures). :func:`legalize_hlsl_stage_interfaces` rewrites
the one case whose HLSL form is fixed, and refuses the others before the DirectX emitter sees them:

* A task shader that launches mesh groups without a payload (``EmitMeshTasksEXT(x, y, z)`` and no
  ``taskPayloadSharedEXT`` variable) becomes ``DispatchMesh(x, y, z, payload)`` with a 4-byte ``groupshared``
  struct no stage reads: HLSL's ``DispatchMesh`` always takes a payload, and DXC rejects a zero-sized one.
* A ``taskPayloadSharedEXT`` variable is refused: HLSL needs it passed to ``DispatchMesh`` and received as the mesh
  entry's ``in payload`` parameter, which is not implemented yet.
* A mesh entry point without ``out vertices`` / ``out indices`` parameters, whose outputs are GLSL-style
  module-scope arrays (``gl_MeshVerticesEXT``, ``gl_PrimitiveTriangleIndicesEXT``, ``out`` blocks), is refused:
  lowering them to entry parameters is not implemented yet.
* Ray payloads, hit attributes and callable data declared as variables (``rayPayloadEXT``, ``rayPayloadInEXT``,
  ``hitAttributeEXT``, ``callableDataEXT``, ``callableDataInEXT``) are refused: DXR takes them as entry parameters
  and ``TraceRay`` / ``CallShader`` arguments of struct type, and the lowering is not implemented yet.

The refusals are typed (:class:`HlslStageInterfaceUnsupportedError`); before them the emitter failed with an
untyped error or printed HLSL that DXC rejects.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hlsl_ray_interfaces import lower_glsl_ray_interfaces
from .hub_types import iter_functions, iter_structs
from .storage_blocks import _walk

__all__ = ["HlslStageInterfaceUnsupportedError", "legalize_hlsl_stage_interfaces"]

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_TASK_STAGES = frozenset(
    {A.ShaderStage.TASK, A.ShaderStage.AMPLIFICATION, A.ShaderStage.OBJECT}
)
_RAY_INTERFACE_ATTRIBUTES = {
    "raypayloadext": "rayPayloadEXT",
    "raypayloadinext": "rayPayloadInEXT",
    "raypayloadnv": "rayPayloadEXT",
    "raypayloadinnv": "rayPayloadInEXT",
    "hitattributeext": "hitAttributeEXT",
    "hitattributenv": "hitAttributeEXT",
    "callabledataext": "callableDataEXT",
    "callabledatainext": "callableDataInEXT",
    "callabledatanv": "callableDataEXT",
    "callabledatainnv": "callableDataInEXT",
}
_TASK_PAYLOAD_ATTRIBUTES = frozenset({"taskpayloadsharedext", "taskpayloadsharednv"})
_MESH_OUTPUT_ROLES = frozenset({"vertices", "indices", "primitives"})
_PAYLOAD_STRUCT = "CrossGLMeshPayload"
_PAYLOAD_VARIABLE = "crossglMeshPayload"


class HlslStageInterfaceUnsupportedError(TranslationDiagnostic):
    """A GLSL-style mesh or ray stage interface the DirectX emitter cannot declare yet."""

    def __init__(self, stage, capability, detail):
        self.stage = stage
        self.missing_capabilities = (f"directx.{capability}",)
        super().__init__(
            f"DirectX {stage} stage: {detail}",
            target="directx",
            code=f"{CODE_PREFIX}directx-stage-interface-unsupported",
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _names(node):
    """Lower-case names of a declaration's attributes and qualifiers."""
    names = {
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", None) or []
    }
    names.update(
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", None) or []
    )
    return names


def _variables(program, stage=None):
    """Variables a stage can see: the program's globals and the stage's own."""
    yield from getattr(program, "global_variables", None) or []
    if stage is not None:
        yield from getattr(stage, "local_variables", None) or []


def _stage_name(stage):
    kind = getattr(stage, "stage", None)
    return getattr(kind, "value", str(kind))


def _refuse_ray_interfaces(program):
    for stage in _stages(program):
        if not _stage_name(stage).startswith("ray_"):
            continue
        for variable in _variables(program, stage):
            kinds = sorted(
                {
                    _RAY_INTERFACE_ATTRIBUTES[name]
                    for name in _names(variable)
                    if name in _RAY_INTERFACE_ATTRIBUTES
                }
            )
            if kinds:
                raise HlslStageInterfaceUnsupportedError(
                    _stage_name(stage),
                    "ray-interface-variable",
                    f"'{variable.name}' is a {kinds[0]} variable; DXR passes payloads, hit attributes and callable "
                    "data as entry parameters and TraceRay/CallShader arguments of struct type, and lowering "
                    "module-scope ray interface variables to them is not implemented yet",
                )


def _refuse_global_mesh_outputs(program):
    for stage in _stages(program):
        if getattr(stage, "stage", None) is not A.ShaderStage.MESH:
            continue
        entry = getattr(stage, "entry_point", None)
        if entry is None:
            continue
        if any(
            _names(parameter) & _MESH_OUTPUT_ROLES
            for parameter in getattr(entry, "parameters", None) or []
        ):
            continue
        raise HlslStageInterfaceUnsupportedError(
            "mesh",
            "mesh-output-variables",
            "the entry declares no out vertices/indices parameters; its outputs are module-scope variables "
            "(GLSL gl_MeshVerticesEXT, gl_Primitive*IndicesEXT and out arrays), and lowering them to HLSL "
            "mesh output parameters is not implemented yet",
        )


def _task_payload_variable(program, stage):
    for variable in _variables(program, stage):
        if _names(variable) & _TASK_PAYLOAD_ATTRIBUTES:
            return variable
    return None


def _dispatch_mesh_calls(nodes):
    return [
        node
        for node in _walk(list(nodes))
        if isinstance(node, A.MeshOpNode) and node.operation == "DispatchMesh"
    ]


def _unused_name(base, taken):
    name, suffix = base, 1
    while name in taken:
        name, suffix = f"{base}{suffix}", suffix + 1
    return name


def _supply_dispatch_mesh_payloads(program):
    shared = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        if getattr(stage, "stage", None) not in _TASK_STAGES:
            continue
        payload = _task_payload_variable(program, stage)
        if payload is not None:
            raise HlslStageInterfaceUnsupportedError(
                _stage_name(stage),
                "task-payload-variable",
                f"'{payload.name}' is a taskPayloadSharedEXT variable; HLSL passes the payload to DispatchMesh and "
                "the mesh entry receives it as an in payload parameter, which is not implemented yet",
            )
        functions = [*(getattr(stage, "local_functions", None) or []), *shared]
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
        calls = [
            call for call in _dispatch_mesh_calls(functions) if len(call.arguments) == 3
        ]
        if not calls:
            continue
        taken = {struct.name for struct in iter_structs(program)}
        taken.update(function.name for function in iter_functions(program))
        taken.update(
            variable.name
            for variable in _walk([program])
            if isinstance(variable, A.VariableNode)
        )
        struct_name = _unused_name(_PAYLOAD_STRUCT, taken)
        variable_name = _unused_name(_PAYLOAD_VARIABLE, taken | {struct_name})
        program.structs.append(
            A.StructNode(
                struct_name, [A.StructMemberNode("unused", A.PrimitiveType("uint"))]
            )
        )
        program.global_variables.append(
            A.VariableNode(
                variable_name, A.NamedType(struct_name), qualifiers=["groupshared"]
            )
        )
        for call in calls:
            call.arguments.append(A.IdentifierNode(variable_name))


def legalize_hlsl_stage_interfaces(program, target):
    """Rewrite or refuse GLSL-style mesh and ray stage interfaces when ``target`` is HLSL."""
    if (
        not isinstance(program, A.ShaderNode)
        or str(target or "").lower() not in _HLSL_TARGETS
    ):
        return
    lower_glsl_ray_interfaces(program)
    _refuse_ray_interfaces(program)
    _refuse_global_mesh_outputs(program)
    _supply_dispatch_mesh_payloads(program)
