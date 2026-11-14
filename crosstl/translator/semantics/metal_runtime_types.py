"""Metal runtime object types that no other target declares.

MSL gives shaders types for objects of the Metal runtime:

- indirect command buffers, ``command_buffer``, ``render_command`` and ``compute_command`` (MSL 4.1 section 6.17,
  "Encoding Commands for Indirect Command Buffers"), through which a kernel encodes draw and dispatch commands;
- variable rasterization rate maps, ``rasterization_rate_map_data`` and ``rasterization_rate_map_decoder`` (MSL 4.1
  section 6.11), which map between the screen and physical coordinates of a rate map the host configures.

HLSL, GLSL and SPIR-V have no shader-side counterpart. Direct3D 12 ``ExecuteIndirect`` and Vulkan indirect draws read
argument buffers that the host binds, with the commands recorded on the host. The fragment shading rates of Direct3D
12 and Vulkan expose no coordinate mapping. The type names are CrossGL built-ins so that the Metal generator can emit
them, but the other generators would copy them verbatim. :func:`refuse_metal_runtime_types` raises
:class:`MetalRuntimeTypeUnsupportedError` for a use reachable from a stage entry (directly, or through a structure
that holds one) for every target other than Metal.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .declared_types import _dead_functions, _nodes, _short

__all__ = [
    "METAL_RUNTIME_TYPES",
    "MetalRuntimeTypeUnsupportedError",
    "metal_runtime_type_uses",
    "refuse_metal_runtime_types",
]

_COMMANDS = "its indirect commands are recorded on the host"
_RATE_MAP = "its fragment shading rates expose no mapping between screen and physical coordinates"
# Type name -> (what it is, why the other targets cannot declare it).
METAL_RUNTIME_TYPES = {
    "command_buffer": ("an indirect command buffer", _COMMANDS),
    "render_command": ("an indirect render command", _COMMANDS),
    "compute_command": ("an indirect compute command", _COMMANDS),
    "rasterization_rate_map_data": ("variable rasterization rate map data", _RATE_MAP),
    "rasterization_rate_map_decoder": (
        "a variable rasterization rate map decoder",
        _RATE_MAP,
    ),
}
_METAL_TARGETS = frozenset({"metal", "cgl", "crossgl"})


class MetalRuntimeTypeUnsupportedError(TranslationDiagnostic):
    """A Metal runtime object type used for a target that has no shader-side counterpart."""

    def __init__(self, target, name, *, source_location=None):
        self.type_name = name
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.metal-runtime-type-{name}",)
        what, why = METAL_RUNTIME_TYPES[name]
        super().__init__(
            f"'{name}' is {what}, a Metal runtime object that shaders of this target cannot declare: {why}",
            target=target,
            code=f"{CODE_PREFIX}{target}-metal-runtime-type-unsupported",
        )


def _base_name(node):
    name = getattr(node, "name", None)
    if not isinstance(name, str):
        return None
    return _short(name.split("<", 1)[0].strip())


def _holding_structs(program):
    """Structure name -> the Metal runtime type it holds, directly or through another structure."""
    structs = [node for node in _nodes(program) if isinstance(node, A.StructNode)]
    holding = {}
    changed = True
    while changed:
        changed = False
        for struct in structs:
            if struct.name in holding:
                continue
            for node in _nodes(struct):
                if not isinstance(node, (A.NamedType, A.PrimitiveType)):
                    continue
                base = _base_name(node)
                held = base if base in METAL_RUNTIME_TYPES else holding.get(base)
                if held is not None:
                    holding[struct.name] = held
                    changed = True
                    break
    return structs, holding


def metal_runtime_type_uses(program):
    """``(type name, node)`` for each use of a Metal runtime type in code a stage entry can reach."""
    structs, holding = _holding_structs(program)
    skip = _dead_functions(program) | {id(struct) for struct in structs}
    uses = []
    for node in _nodes(program, skip):
        if not isinstance(node, (A.NamedType, A.PrimitiveType)):
            continue
        base = _base_name(node)
        held = base if base in METAL_RUNTIME_TYPES else holding.get(base)
        if held is not None:
            uses.append((held, node))
    return uses


def refuse_metal_runtime_types(program, target):
    """Raise :class:`MetalRuntimeTypeUnsupportedError` for the first reachable use, unless ``target`` is Metal."""
    target = str(target or "").lower()
    if target in _METAL_TARGETS:
        return
    for name, node in metal_runtime_type_uses(program):
        raise MetalRuntimeTypeUnsupportedError(
            target, name, source_location=getattr(node, "source_location", None)
        )
