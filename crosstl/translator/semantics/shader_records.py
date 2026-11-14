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
binds instead of its record. :func:`refuse_shader_record_buffers` refuses targets that cannot read a record:

* Direct3D 12 passes record data as local root arguments, which need a local root signature (DirectX Raytracing
  functional spec, "Local root signatures vs global root signatures"); CrossTL does not emit root signatures.
* Metal, WGSL and the compute languages have no shader binding table.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "SHADER_RECORD_TARGETS",
    "ShaderRecordBufferUnsupportedError",
    "is_shader_record_buffer",
    "refuse_shader_record_buffers",
    "shader_record_memory_layout",
]

# Targets whose emitters declare a shader record buffer: GLSL (the Vulkan dialect for ray stages) and SPIR-V.
SHADER_RECORD_TARGETS = frozenset(
    {"opengl", "glsl", "vulkan", "spirv", "cgl", "crossgl"}
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
        reason = (
            "Direct3D 12 passes shader record data as local root arguments, which need a local root signature, "
            "and CrossTL does not emit root signatures"
            if target in {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"}
            else "the target has no shader binding table"
        )
        super().__init__(
            f"shader record buffer '{name}' (GL_EXT_ray_tracing shaderRecordEXT) cannot be read on {target}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-shader-record-buffer-unsupported",
        )


def refuse_shader_record_buffers(program, target) -> None:
    """Raise :class:`ShaderRecordBufferUnsupportedError` for a shader record buffer ``target`` cannot declare."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target in SHADER_RECORD_TARGETS:
        return
    declarations = list(getattr(program, "global_variables", None) or [])
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        declarations += list(getattr(stage, "local_variables", None) or [])
    for declaration in declarations:
        if is_shader_record_buffer(declaration):
            raise ShaderRecordBufferUnsupportedError(
                target,
                getattr(declaration, "name", "?"),
                source_location=getattr(declaration, "source_location", None),
            )
