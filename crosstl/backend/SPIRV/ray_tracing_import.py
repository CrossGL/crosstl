"""SPIR-V ray tracing and mesh shading instructions, imported as CrossGL intrinsic calls.

SPIR-V names a ray payload or callable data variable directly in ``OpTraceRayKHR`` and ``OpExecuteCallableKHR``;
CrossGL, like GL_EXT_ray_tracing, names it by its ``location``. glslang does not decorate these variables with a
``Location``, so :func:`ray_variable_locations` gives each undecorated one the lowest location its family leaves
free, in module order: outgoing and incoming payloads share one family, callable data the other.

Every result-less instruction the importer does not translate must stop the import:
:class:`SpirvInstructionUnsupportedError` replaces the old behaviour of dropping it, which silently removed effects
such as tracing a ray.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .VulkanAst import FunctionCallNode

PAYLOAD_STORAGE_CLASSES = frozenset(
    {"RayPayloadKHR", "IncomingRayPayloadKHR", "RayPayloadNV", "IncomingRayPayloadNV"}
)
CALLABLE_STORAGE_CLASSES = frozenset(
    {
        "CallableDataKHR",
        "IncomingCallableDataKHR",
        "CallableDataNV",
        "IncomingCallableDataNV",
    }
)
# SPIR-V ``BuiltIn`` decorations of ray tracing, mesh shading and subgroup variables, and the GLSL built-in each is
# in GL_EXT_ray_tracing, GL_EXT_mesh_shader and GL_KHR_shader_subgroup: the names the GLSL importer gives CrossGL,
# so both importers reach the emitters' built-in mappings. ``RayTmaxKHR`` is also what glslang emits for
# ``gl_HitTEXT``, which GL_EXT_ray_tracing defines as an alias of ``gl_RayTmaxEXT`` in hit shaders.
BUILTIN_VARIABLE_NAMES = {
    "CullMaskKHR": "gl_CullMaskEXT",
    "CullPrimitiveEXT": "gl_CullPrimitiveEXT",
    "DeviceIndex": "gl_DeviceIndex",
    "DrawIndex": "gl_DrawID",
    "HitKindKHR": "gl_HitKindEXT",
    "HitTNV": "gl_HitTEXT",
    "HitTriangleVertexPositionsKHR": "gl_HitTriangleVertexPositionsEXT",
    "IncomingRayFlagsKHR": "gl_IncomingRayFlagsEXT",
    "InstanceCustomIndexKHR": "gl_InstanceCustomIndexEXT",
    "LaunchIdKHR": "gl_LaunchIDEXT",
    "LaunchSizeKHR": "gl_LaunchSizeEXT",
    "ObjectRayDirectionKHR": "gl_ObjectRayDirectionEXT",
    "ObjectRayOriginKHR": "gl_ObjectRayOriginEXT",
    "ObjectToWorldKHR": "gl_ObjectToWorldEXT",
    "PrimitiveLineIndicesEXT": "gl_PrimitiveLineIndicesEXT",
    "PrimitivePointIndicesEXT": "gl_PrimitivePointIndicesEXT",
    "PrimitiveShadingRateKHR": "gl_PrimitiveShadingRateEXT",
    "PrimitiveTriangleIndicesEXT": "gl_PrimitiveTriangleIndicesEXT",
    "RayGeometryIndexKHR": "gl_GeometryIndexEXT",
    "RayTmaxKHR": "gl_RayTmaxEXT",
    "RayTminKHR": "gl_RayTminEXT",
    "ShadingRateKHR": "gl_ShadingRateEXT",
    "SubgroupEqMask": "gl_SubgroupEqMask",
    "SubgroupGeMask": "gl_SubgroupGeMask",
    "SubgroupGtMask": "gl_SubgroupGtMask",
    "SubgroupLeMask": "gl_SubgroupLeMask",
    "SubgroupLtMask": "gl_SubgroupLtMask",
    "WorldRayDirectionKHR": "gl_WorldRayDirectionEXT",
    "WorldRayOriginKHR": "gl_WorldRayOriginEXT",
    "WorldToObjectKHR": "gl_WorldToObjectEXT",
}
# Result-less instructions with no effect of their own: debug records, structured control flow markers that the
# importer rebuilds from the CFG, and terminators that the block structure already carries.
STRUCTURAL_INSTRUCTIONS = frozenset(
    {
        "OpBranch",
        "OpBranchConditional",
        "OpFunctionEnd",
        "OpLabel",
        "OpLine",
        "OpLoopMerge",
        "OpNoLine",
        "OpNop",
        "OpReturn",
        "OpSelectionMerge",
        "OpSwitch",
        "OpUnreachable",
    }
)
# Statement instructions and the CrossGL intrinsic each becomes; the payload operand index, if any, is replaced by
# the variable's location.
_STATEMENTS = {
    "OpTraceRayKHR": ("TraceRay", 11, 10),
    "OpExecuteCallableKHR": ("CallShader", 2, 1),
    "OpIgnoreIntersectionKHR": ("IgnoreHit", 0, None),
    "OpTerminateRayKHR": ("AcceptHitAndEndSearch", 0, None),
    "OpSetMeshOutputsEXT": ("SetMeshOutputCounts", 2, None),
    # The optional payload operand is the stage's taskPayloadSharedEXT variable, which CrossGL declares instead.
    "OpEmitMeshTasksEXT": ("DispatchMesh", 3, None),
}


class SpirvInstructionUnsupportedError(TranslationDiagnostic):
    """A SPIR-V instruction the importer cannot translate, refused rather than dropped."""

    def __init__(self, opcode, detail="has no CrossGL translation"):
        self.opcode = opcode
        self.missing_capabilities = (f"spirv.instruction.{opcode}",)
        super().__init__(
            f"SPIR-V instruction {opcode} {detail}",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-instruction-unsupported",
        )


def _decorated_location(decorations, variable_id):
    for decoration, operands in decorations.get(variable_id, []) or []:
        if decoration == "Location" and operands:
            try:
                return int(operands[0])
            except (TypeError, ValueError):
                return None
    return None


def ray_variable_locations(variables, decorations):
    """``{variable id: location}`` for every ray payload and callable data variable, decorated or assigned."""
    locations, undecorated = {}, []
    used = {"payload": set(), "callable": set()}
    for variable in variables:
        storage_class = variable.get("storage_class")
        family = (
            "payload"
            if storage_class in PAYLOAD_STORAGE_CLASSES
            else "callable" if storage_class in CALLABLE_STORAGE_CLASSES else None
        )
        if family is None:
            continue
        location = _decorated_location(decorations, variable["id"])
        if location is None:
            undecorated.append((variable["id"], family))
        else:
            locations[variable["id"]] = location
            used[family].add(location)
    for variable_id, family in undecorated:
        location = 0
        while location in used[family]:
            location += 1
        used[family].add(location)
        locations[variable_id] = location
    return locations


def ray_statement(opcode, operands, operand_expression, locations):
    """The CrossGL call for a ray tracing or mesh statement instruction, or ``None`` for other opcodes."""
    spec = _STATEMENTS.get(opcode)
    if spec is None:
        return None
    name, arity, payload_index = spec
    if len(operands) < arity:
        raise SpirvInstructionUnsupportedError(
            opcode, f"needs {arity} operands, got {len(operands)}"
        )
    arguments = []
    for index, operand in enumerate(operands[:arity]):
        if index == payload_index:
            location = locations.get(operand)
            if location is None:
                raise SpirvInstructionUnsupportedError(
                    opcode,
                    f"names {operand}, which is not a ray payload or callable data variable",
                )
            arguments.append(location)
        else:
            arguments.append(operand_expression(operand))
    return FunctionCallNode(name, arguments)


def refuse_untranslated_statement(result_id, opcode):
    """Raise for a result-less instruction that reached the end of the statement builder untranslated."""
    if result_id or opcode in STRUCTURAL_INSTRUCTIONS:
        return
    raise SpirvInstructionUnsupportedError(opcode)
