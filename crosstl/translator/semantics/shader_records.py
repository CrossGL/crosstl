"""Shader record buffers: the ``shaderRecordEXT`` buffer block of a ray tracing stage, carried or refused per target.

A ray generation, hit, miss or callable shader reads the data its shader binding table record carries through a
buffer block qualified ``shaderRecordEXT`` (GL_EXT_ray_tracing, "Shader Record Buffers"); such a block takes no
``set`` or ``binding``. SPIR-V declares it as a ``Block``-decorated struct in the ``ShaderRecordBufferKHR`` storage
class with no descriptor decorations (SPV_KHR_ray_tracing). CrossGL spells it as a buffer block whose layout names
``shaderRecordEXT``::

    SBT sbt @glsl_buffer_block(shaderRecordEXT, std430);

The GLSL block syntax ``layout(shaderRecordEXT, std430) buffer SBT { ... } sbt;`` parses to the same declaration with
``shaderRecordEXT`` as an attribute of its own; :func:`is_shader_record_buffer` accepts both spellings.

Before this module the qualifier was dropped on the way out: SPIR-V bound the block as an ordinary storage buffer
at set 0, binding 0, and GLSL redeclared it with ``binding = 0``, so the stage read a descriptor the host never
binds instead of its record. HLSL declares the block as a ``[[vk::shader_record_ext]]`` constant buffer that a
local root signature feeds on Direct3D 12 (:mod:`crosstl.translator.codegen.hlsl_shader_records`), and Slang as a
``[[vk::shader_record]]`` constant buffer (:mod:`crosstl.translator.codegen.slang_shader_records`).
:func:`refuse_shader_record_buffers` refuses the targets that have no shader binding table: Metal, WGSL and the
compute languages. On SPIR-V, GLSL and Slang it also refuses an entry point of a stage other than a ray tracing stage
that reads a record: only ray generation, intersection, any-hit, closest-hit, miss and callable shaders may use the
``ShaderRecordBufferKHR`` storage class (VUID-StandaloneSpirv-ShaderRecordBufferKHR-07119), and glslang refuses
``shaderRecordEXT`` in other stages. HLSL output checks the same in :mod:`crosstl.translator.codegen.hlsl_shader_records`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import normalize_stage_name
from .rewrite import child_slots, get_child

__all__ = [
    "SHADER_RECORD_TARGETS",
    "ShaderRecordBufferUnsupportedError",
    "ShaderRecordStageError",
    "is_shader_record_buffer",
    "refuse_shader_record_buffers",
    "shader_record_memory_layout",
]

# Targets whose emitters declare a shader record buffer: GLSL (the Vulkan dialect for ray stages), SPIR-V, HLSL and
# Slang.
SHADER_RECORD_TARGETS = frozenset(
    {"opengl", "glsl", "vulkan", "spirv", "directx", "hlsl", "slang", "cgl", "crossgl"}
)
# Targets on which this module refuses a record that a stage without a shader binding table reads.
_STAGE_CHECKED_TARGETS = frozenset({"opengl", "glsl", "vulkan", "spirv", "slang"})
RAY_TRACING_STAGES = frozenset(
    {
        "ray_generation",
        "ray_intersection",
        "ray_any_hit",
        "ray_closest_hit",
        "ray_miss",
        "ray_callable",
    }
)
_MEMORY_LAYOUTS = ("std430", "std140", "scalar")


def _normalized(text):
    return str(text).strip().lower().replace("_", "")


def _argument_text(argument):
    for field in ("name", "value"):
        value = getattr(argument, field, None)
        if value is not None:
            return str(value)
    return str(argument)


def _buffer_block_arguments(node):
    for attribute in getattr(node, "attributes", None) or []:
        if _normalized(getattr(attribute, "name", "")) == "glslbufferblock":
            arguments = getattr(attribute, "arguments", None)
            if arguments is None:
                arguments = getattr(attribute, "args", None) or []
            return [_argument_text(argument) for argument in arguments]
    return []


def is_shader_record_buffer(node) -> bool:
    """Whether a declaration is a ``shaderRecordEXT`` (or NV ``shaderRecordNV``) buffer block."""
    names = {
        _normalized(getattr(a, "name", ""))
        for a in getattr(node, "attributes", None) or []
    }
    names.update(_normalized(q) for q in getattr(node, "qualifiers", None) or [])
    names.update(_normalized(argument) for argument in _buffer_block_arguments(node))
    return bool(names & {"shaderrecordext", "shaderrecordnv"})


def shader_record_memory_layout(node, default="std430") -> str:
    """The memory layout a shader record block declares, ``std430`` when it names none."""
    for argument in _buffer_block_arguments(node):
        if _normalized(argument) in _MEMORY_LAYOUTS:
            return _normalized(argument)
    for attribute in getattr(node, "attributes", None) or []:
        if _normalized(getattr(attribute, "name", "")) in _MEMORY_LAYOUTS:
            return _normalized(attribute.name)
    return default


class ShaderRecordBufferUnsupportedError(TranslationDiagnostic):
    """A shader record buffer the target cannot read."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.shader-record-buffer",)
        super().__init__(
            f"shader record buffer '{name}' (GL_EXT_ray_tracing shaderRecordEXT) cannot be read on {target}: "
            "the target has no shader binding table",
            target=target,
            code=f"{CODE_PREFIX}{target}-shader-record-buffer-unsupported",
        )


class ShaderRecordStageError(TranslationDiagnostic):
    """A shader record that an entry point of a stage without a shader binding table reads."""

    def __init__(self, target, name, stage, entry, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.shader-record-buffer.{stage}",)
        super().__init__(
            f"shader record buffer '{name}' (GL_EXT_ray_tracing shaderRecordEXT) is read by the {stage} entry point "
            f"'{entry}', and only ray tracing stages have a shader binding table record "
            "(VUID-StandaloneSpirv-ShaderRecordBufferKHR-07119)",
            target=target,
            code=f"{CODE_PREFIX}{target}-shader-record-stage-unsupported",
        )


def _stage_items(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.items()) if isinstance(stages, dict) else []


def _record_declarations(program):
    declarations = list(getattr(program, "global_variables", None) or [])
    for _stage_type, stage in _stage_items(program):
        declarations += list(getattr(stage, "local_variables", None) or [])
    return [node for node in declarations if is_shader_record_buffer(node)]


def refuse_shader_record_buffers(program, target) -> None:
    """Raise for a shader record buffer ``target`` cannot declare, or that a stage without a record reads.

    :class:`ShaderRecordBufferUnsupportedError` names a record on a target without a shader binding table;
    :class:`ShaderRecordStageError` names a non-ray-tracing entry point that reads one on SPIR-V, GLSL or Slang.
    """
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode):
        return
    records = _record_declarations(program)
    if not records:
        return
    if target not in SHADER_RECORD_TARGETS:
        declaration = records[0]
        raise ShaderRecordBufferUnsupportedError(
            target,
            getattr(declaration, "name", "?"),
            source_location=getattr(declaration, "source_location", None),
        )
    if target in _STAGE_CHECKED_TARGETS:
        _refuse_non_ray_stage_reads(program, target, records)


def _entry_points(program):
    """``(stage name, entry function)`` for each stage block's entry point and each stage-qualified function."""
    from ..codegen.stage_utils import function_stage_name

    entries = []
    for function in getattr(program, "functions", None) or []:
        stage = function_stage_name(function)
        if stage:
            entries.append((normalize_stage_name(stage) or stage, function))
    for stage_type, stage in _stage_items(program):
        entry = getattr(stage, "entry_point", None)
        name = normalize_stage_name(stage_type)
        if isinstance(entry, A.FunctionNode) and name:
            entries.append((name, entry))
    return entries


def _declared_names(node, names):
    if isinstance(node, A.VariableNode):
        names.add(node.name)
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        if child is not None and not isinstance(child, (A.FunctionNode, A.StructNode)):
            _declared_names(child, names)
    return names


def _read_names(node, record_names, found):
    if isinstance(node, A.IdentifierNode) and node.name in record_names:
        found.add(node.name)
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        if child is not None and not isinstance(child, (A.FunctionNode, A.StructNode)):
            _read_names(child, record_names, found)
    return found


def _function_reads(function, record_names):
    """The record names ``function``'s body reads, less those a parameter or local of the same name hides."""
    body = getattr(function, "body", None)
    if body is None:
        return set()
    hidden = {
        getattr(parameter, "name", None) for parameter in function.parameters or []
    }
    hidden = _declared_names(body, hidden)
    return _read_names(body, record_names - hidden, set())


def _refuse_non_ray_stage_reads(program, target, records):
    from ..codegen.call_graph import CallGraph

    record_names = {str(node.name) for node in records}
    graph = CallGraph(program)
    index_of = {id(function): index for index, function in enumerate(graph.functions)}
    for stage, entry in _entry_points(program):
        if stage in RAY_TRACING_STAGES or id(entry) not in index_of:
            continue
        reached, pending = set(), [index_of[id(entry)]]
        while pending:
            index = pending.pop()
            if index not in reached:
                reached.add(index)
                pending.extend(sorted(graph.edges[index]))
        read = set()
        for index in sorted(reached):
            read |= _function_reads(graph.functions[index], record_names)
        if read:
            name = sorted(read)[0]
            declaration = next(node for node in records if str(node.name) == name)
            raise ShaderRecordStageError(
                target,
                name,
                stage,
                entry.name,
                source_location=getattr(declaration, "source_location", None),
            )
