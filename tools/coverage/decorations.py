"""Whether a translated graphics, compute or texture probe keeps the decorations, built-ins, capabilities, execution
modes and image shapes its feature is anchored to.

A graphics probe is validated but never run, so a translation that drops an interpolation qualifier or a built-in
still validates. A compute probe runs with one invocation per workgroup, so a translation that writes a constant in
place of a built-in such as the subgroup invocation index computes what the native probe computes. A translation that
samples a 2D texture where its probe samples a cube or an array texture can still validate, since HLSL truncates the
extra coordinate. Where the target language compiles to SPIR-V, the translation is lowered with the language's
reference compiler and must keep every ``Decoration``, ``BuiltIn``, ``Capability`` and ``ExecutionMode`` anchor of its
feature, the task-payload storage a mesh stage reads, and the storage and calls through which a ray-tracing stage
hands data to another, as each native probe does, except the anchors in
:data:`FOLDED`. A resource feature keeps only its image's
``Dim`` anchor, arrayed when the feature's name ends in ``-array``, the capability that shape needs, its push-constant
storage and a descriptor set other than the first. GLSL ES 3.00
is lowered as GLSL ES 3.10, the first version glslang compiles for Vulkan. Metal compiles to AIR instead, where a
translation must keep the attribute of each anchor (:data:`AIR`), the call, metadata or qualifier that stands for it
(:data:`AIR_PATTERNS`), or the texture type of its shape (:data:`AIR_TEXTURES`). An anchor the compiler resolves
before it writes SPIR-V is checked in the translation's own declarations instead (:data:`SOURCE`). A translation that
its validator accepts but its compiler does not lower here is only validated.
"""

from __future__ import annotations

import re

from . import validators
from .model import Feature
from .templates import ABSENT, TEMPLATES

_PATTERNS = {
    "Decoration": r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? {name}\b",
    "BuiltIn": r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? BuiltIn {name}\b",
    "Capability": r"^\s*OpCapability {name}\b",
    "ExecutionMode": r"^\s*OpExecutionMode %\w+ {name}\b",
    # An image type's dimension and, after its depth operand, whether it is arrayed.
    "Dim": r"^\s*%\w+ = OpTypeImage %\w+ {name} \d+ {arrayed}\b",
}
_IMAGE_KINDS = frozenset({"Dim", "Capability"})
# The instructions that stand for an anchor as well: whether an invocation is a helper is the ``HelperInvocation``
# built-in or, with SPV_EXT_demote_to_helper_invocation, ``OpIsHelperInvocationEXT``, which DXC and slangc emit for
# ``IsHelperLane()``.
_INSTRUCTIONS = {"BuiltIn.HelperInvocation": r"^\s*%\w+ = OpIsHelperInvocationEXT\b"}
# The anchors whose operand must differ from the default as well: glslang, DXC and slangc write ``Invocations 1``
# for every geometry stage, so a translation keeps an instanced stage only with more than one invocation. Every
# tessellation stage decorates its levels ``Patch``, so a translation keeps a per-patch value only when a variable
# with a ``Location`` is ``Patch`` too.
_OPERANDS = {
    "ExecutionMode.Invocations": r"^\s*OpExecutionMode %\w+ Invocations (?!1\b)\d+\b",
    # A mesh stage that reads its task payload declares a variable in task-payload storage.
    "StorageClass.TaskPayloadWorkgroupEXT": (
        r"^\s*%\w+ = OpVariable %\w+ TaskPayloadWorkgroupEXT\b"
    ),
    "Decoration.Patch": (
        r"^\s*OpDecorate (%\w+) Patch\b[\s\S]*^\s*OpDecorate \1 Location\b"
        r"|^\s*OpDecorate (%\w+) Location\b[\s\S]*^\s*OpDecorate \2 Patch\b"
    ),
    # A ray-tracing stage that hands a payload or callable data to another stage, or reads its shader record,
    # declares a variable in that storage class, and calls the instruction that hands the data over. SPIR-V gives
    # these storage classes and OpReportIntersectionKHR the values of their SPV_NV_ray_tracing forerunners, whose
    # names a disassembler built from an older grammar prints.
    **{
        f"StorageClass.{name}KHR": rf"^\s*%\w+ = OpVariable %\w+ {name}(?:KHR|NV)\b"
        for name in ("RayPayload", "CallableData", "ShaderRecordBuffer")
    },
    "OpTraceRayKHR": r"^\s*OpTraceRayKHR\b",
    "OpExecuteCallableKHR": r"^\s*OpExecuteCallableKHR\b",
    "OpReportIntersectionKHR": r"^\s*%\w+ = OpReportIntersection(?:KHR|NV)\b",
    # Attributes an intersection stage reports, or a hit stage reads in place of the triangle's barycentrics, are a
    # structure in hit-attribute storage.
    "StorageClass.HitAttributeKHR": (
        r"^\s*(%\w+) = OpTypeStruct\b[\s\S]*"
        r"^\s*%\w+ = OpTypePointer HitAttribute(?:KHR|NV) \1\s*$"
    ),
    # Push constants are a variable in push-constant storage, and a block bound past the first descriptor set is
    # decorated with that set.
    "StorageClass.PushConstant": r"^\s*%\w+ = OpVariable %\w+ PushConstant\b",
    "Decoration.DescriptorSet": r"^\s*OpDecorate %\w+ DescriptorSet (?!0\b)\d+\b",
}

# The enumerants SPIR-V shares with an earlier NV extension under one value, which a disassembler built from an older
# grammar names by their NV spelling: ``OutputLinesEXT`` is also ``OutputLinesNV``.
_ALIASES = {"OutputLinesEXT": "OutputLinesNV", "PerPrimitiveEXT": "PerPrimitiveNV"}

# The metadata the Metal compiler writes in AIR for the attribute that carries each anchor: ``[[flat]]`` becomes
# ``!"air.flat"`` and ``[[vertex_id]]`` ``!"air.vertex_id"``. A fragment function's ``[[position]]`` input is the
# fragment coordinate, and ``[[color(n)]]`` is a render target at a location. A capability has no attribute, so a
# Metal translation is not checked for one.
AIR = {
    "BuiltIn.FragCoord": "air.position",
    "BuiltIn.FrontFacing": "air.front_facing",
    "BuiltIn.FragDepth": "air.depth",
    "BuiltIn.VertexIndex": "air.vertex_id",
    "BuiltIn.InstanceIndex": "air.instance_id",
    "BuiltIn.PointSize": "air.point_size",
    "BuiltIn.ClipDistance": "air.clip_distance",
    "BuiltIn.PrimitiveId": "air.primitive_id",
    "BuiltIn.SampleId": "air.sample_id",
    "BuiltIn.SampleMask": "air.sample_mask",
    "BuiltIn.Layer": "air.render_target_array_index",
    "BuiltIn.ViewportIndex": "air.viewport_array_index",
    "BuiltIn.PointCoord": "air.point_coord",
    "Decoration.Flat": "air.flat",
    "Decoration.NoPerspective": "air.no_perspective",
    "Decoration.Centroid": "air.centroid",
    "Decoration.Sample": "air.sample",
    "Decoration.Invariant": "air.invariant",
    "Decoration.Location": "air.render_target",
    "ExecutionMode.EarlyFragmentTests": "early_fragment_tests",
    "ExecutionMode.DepthGreater": "air.greater",
    "BuiltIn.BaseVertex": "air.base_vertex",
    "BuiltIn.BaseInstance": "air.base_instance",
    "BuiltIn.GlobalInvocationId": "air.thread_position_in_grid",
    "BuiltIn.LocalInvocationId": "air.thread_position_in_threadgroup",
    "BuiltIn.WorkgroupId": "air.threadgroup_position_in_grid",
    "BuiltIn.LocalInvocationIndex": "air.thread_index_in_threadgroup",
    "BuiltIn.NumWorkgroups": "air.threadgroups_per_grid",
    "BuiltIn.WorkgroupSize": "air.threads_per_threadgroup",
    "BuiltIn.SubgroupSize": "air.threads_per_simdgroup",
    "BuiltIn.SubgroupLocalInvocationId": "air.thread_index_in_simdgroup",
    "BuiltIn.NumSubgroups": "air.simdgroups_per_threadgroup",
    "BuiltIn.SubgroupId": "air.simdgroup_index_in_threadgroup",
    "BuiltIn.BaryCoordKHR": "air.barycentric_coord",
    "BuiltIn.ViewIndex": "air.amplification_id",
    "BuiltIn.FragStencilRefEXT": "air.stencil",
    "Decoration.Patch": "air.patch_input",
    "Decoration.PerPrimitiveEXT": "air.mesh_primitive_data",
    "BuiltIn.CullPrimitiveEXT": "air.primitive_culled",
    "StorageClass.TaskPayloadWorkgroupEXT": "air.payload",
}
# The attributes that AIR names apart when a fragment function reads them: a ``[[sample_mask]]`` input is
# ``!"air.sample_mask_in"``. A feature whose name ends in ``-input`` reads its built-in.
AIR_INPUTS = {"BuiltIn.SampleMask": "air.sample_mask_in"}
# The anchors that AIR carries otherwise than as an attribute, as patterns of its text: ``simd_is_helper_thread()``
# calls ``@air.simd_is_helper_thread``, each function constant is an entry of ``!air.function_constants``, a
# ``__restrict`` pointer is a ``noalias`` kernel argument, a ``volatile`` one is read and written with volatile loads
# and stores, a ``coherent(device)`` one through the ``device_coherent`` intrinsics, a quad patch is named in the
# ``!"air.patch"`` metadata of its post-tessellation vertex function, and a mesh function's topology in its
# ``!"air.mesh_type_info"``.
AIR_PATTERNS = {
    "BuiltIn.HelperInvocation": r"@air\.simd_is_helper_thread\(",
    "Decoration.SpecId": r"^!air\.function_constants = ",
    "Decoration.Restrict": r"^define [^\n]*\bnoalias\b",
    "Decoration.Volatile": r"\b(?:load|store) volatile\b",
    "Decoration.Coherent": r"@air\.\w+\.device_coherent\.",
    "ExecutionMode.Quads": r'!"air\.patch", !"quad"',
    "ExecutionMode.OutputPoints": r'!"air\.mesh_type_info",[^\n]*!"air\.point"\}',
    "ExecutionMode.OutputLinesEXT": r'!"air\.mesh_type_info",[^\n]*!"air\.line"\}',
}
# The AIR type of a texture of each dimension: a ``texturecube`` argument is a ``%struct._texture_cube_t``, and an
# arrayed one a ``%struct._texture_cube_array_t``.
AIR_TEXTURES = {
    "Dim.1D": "texture_1d",
    "Dim.2D": "texture_2d",
    "Dim.3D": "texture_3d",
    "Dim.Cube": "texture_cube",
    "Dim.Buffer": "texture_buffer_1d",
}
# DXC writes every output of a geometry stage on stream 0 when it compiles HLSL to SPIR-V, and slangc cannot compile
# a second stream to SPIR-V at all, so an HLSL or Slang translation keeps its streams with a second stream parameter.
_STREAMS = r"\binout\s+PointStream\s*<[^>]*>\s*\w+\s*,[^)]*\binout\s+PointStream\s*<"
# slangc does not decorate a member of a patch-constant structure ``Patch`` unless it is a tessellation level, so a
# Slang translation keeps a per-patch value when the structure that holds the levels has a member with a user
# semantic.
_PATCH_VALUE = (
    r"\bstruct\s+\w+\s*\{(?=[^}]*\bSV_TessFactor\b)[^}]*:\s*(?!SV_)[A-Za-z_]\w*\s*;"
)
# The anchors a language's compiler resolves before it writes SPIR-V, with the declaration that keeps each one in the
# translation itself: naga replaces every pipeline-overridable constant with its value, so a WGSL translation keeps a
# specialization constant when it declares an ``override``.
_STREAM_ANCHORS = ("Decoration.Stream", "Capability.GeometryStreams")
SOURCE = {
    "wgsl": {"Decoration.SpecId": r"^\s*override\s+\w+"},
    "hlsl": dict.fromkeys(_STREAM_ANCHORS, _STREAMS),
    "slang": {
        **dict.fromkeys(_STREAM_ANCHORS, _STREAMS),
        "Decoration.Patch": _PATCH_VALUE,
    },
}
# The built-ins that are constants: GLSL defines ``gl_WorkGroupSize`` as the local size, so a translation may write
# the value itself, as slangc does for ``WorkgroupSize()``.
FOLDED = frozenset({"BuiltIn.WorkgroupSize"})

_ESSL_300 = re.compile(r"\A(\s*#version\s+)300(\s+es\b)")


def _wanted(feature: Feature, language: str) -> list[tuple[str, str]]:
    """The anchors of a graphics or compute ``feature`` that the compiled ``language`` shows, as ``(anchor,
    pattern)`` pairs."""
    if feature.category not in ("graphics", "compute", "resource"):
        return []
    # A language that cannot spell the feature has no native probe to match, as HLSL has no workgroup count for the
    # OpenCL C translations it carries, so its translation is only validated.
    if TEMPLATES.get(language, {}).get(feature.name) is ABSENT:
        return []
    # A resource keeps the shape of its image: the dimension, whether it is arrayed, and the capability the shape
    # needs. Its other anchors are the layouts and access of buffers, which running the probe compares.
    kinds = _IMAGE_KINDS if feature.category == "resource" else _PATTERNS
    arrayed = int(feature.name.endswith("-array"))
    wanted = []
    for anchor in feature.group.split():
        kind, _, name = anchor.partition(".")
        if anchor in FOLDED or (kind not in kinds and anchor not in _OPERANDS):
            continue
        if language == "metal":
            air = AIR_INPUTS if feature.name.endswith("-input") else {}
            attribute = air.get(anchor, AIR.get(anchor))
            if attribute:
                wanted.append((anchor, rf'!"{re.escape(attribute)}"'))
            elif anchor in AIR_PATTERNS:
                wanted.append((anchor, AIR_PATTERNS[anchor]))
            elif anchor in AIR_TEXTURES:
                texture = AIR_TEXTURES[anchor] + ("_array" if arrayed else "")
                wanted.append((anchor, rf"%struct\._{texture}_t\b"))
        else:
            spelled = re.escape(name)
            if name in _ALIASES:
                spelled = f"(?:{spelled}|{_ALIASES[name]})"
            pattern = _OPERANDS.get(anchor) or _PATTERNS[kind].format(
                name=spelled, arrayed=arrayed
            )
            if anchor in _INSTRUCTIONS:
                pattern = f"{pattern}|{_INSTRUCTIONS[anchor]}"
            wanted.append((anchor, pattern))
    return wanted


def _compiled(feature: Feature, language: str, source: str) -> str:
    """``source`` as SPIR-V assembly, or as AIR for Metal; empty when it cannot be compiled here."""
    if language == "spirv":
        return source
    if language == "metal":
        return validators.air(source)
    if language == "webgl":
        language, source = "glsl", _ESSL_300.sub(r"\g<1>310\2", source, count=1)
    if not validators.can_lower(language):
        return ""
    entry = (
        validators.hlsl_entry(source, feature.stage)
        if language in ("hlsl", "slang")
        else "main"
    )
    assembly, _ = validators.lower(
        language, source, entry, feature.stage, raw_ids=False, auto_locations=True
    )
    return assembly


def lost(feature: Feature, language: str, source: str) -> str:
    """The anchors of ``feature`` that ``source``, a validated translation into ``language``, no longer has.

    Empty when it keeps them all, when ``feature`` has none to check or when ``source`` cannot be compiled here.
    """
    wanted = _wanted(feature, language)
    declared = SOURCE.get(language, {})
    missing = [
        anchor
        for anchor, _ in wanted
        if anchor in declared and not re.search(declared[anchor], source, re.M)
    ]
    wanted = [(anchor, pattern) for anchor, pattern in wanted if anchor not in declared]
    compiled = _compiled(feature, language, source) if wanted else ""
    if compiled:
        missing += [
            anchor
            for anchor, pattern in wanted
            if not re.search(pattern, compiled, re.M)
        ]
    return f"the translation loses {', '.join(missing)}" if missing else ""
