"""Debug printing: CrossGL ``printf`` refused for the targets that cannot print from a shader.

CrossGL ``printf(format, values...)`` is the shader debug print of HLSL compiled to SPIR-V, Slang and GLSL's
GL_EXT_debug_printf ``debugPrintfEXT``, with the Vulkan debug printf format. SPIR-V carries it as
``NonSemantic.DebugPrintf``, GLSL as ``debugPrintfEXT`` and Metal 3.2 as ``os_log_default.log``.
:func:`refuse_debug_printf` refuses the targets that have no shader print:

* DXIL has no print operation; DXC accepts ``printf`` only when it compiles HLSL to SPIR-V (``-spirv``) and
  reports "use of unsupported identifier 'printf'" otherwise, so the Direct3D targets cannot carry it.
* WebGL 2 (GLSL ES 3.00) and WGSL define no shader print and no extension that adds one.

A program that declares its own ``printf`` function calls that function, which every target can.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions

__all__ = [
    "DEBUG_PRINTF_FUNCTIONS",
    "DebugPrintfUnsupportedError",
    "refuse_debug_printf",
]

DEBUG_PRINTF_FUNCTIONS = frozenset({"printf", "debugPrintfEXT"})

_HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_REASONS = {
    **dict.fromkeys(
        _HLSL_TARGETS,
        "DXIL has no print operation; DXC accepts printf only when it compiles HLSL to SPIR-V",
    ),
    "webgl": "WebGL 2 (GLSL ES 3.00) has no shader print",
    "wgsl": "WGSL has no shader print",
    "webgpu": "WGSL has no shader print",
}


class DebugPrintfUnsupportedError(TranslationDiagnostic):
    """A debug print for a target that cannot print from a shader."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.debug-printf",)
        super().__init__(
            f"debug print '{name}' cannot be carried to {target}: {_REASONS[target]}",
            target=target,
            code=f"{CODE_PREFIX}{target}-debug-printf-unsupported",
        )


def _call_name(node):
    function = getattr(node, "function", None) or getattr(node, "name", None)
    return getattr(function, "name", function)


def _nodes(node, seen):
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _nodes(item, seen)
        return
    if isinstance(node, dict):
        for item in node.values():
            yield from _nodes(item, seen)
        return
    if not isinstance(node, A.ASTNode):
        return
    yield node
    for key, value in vars(node).items():
        if key not in {"parent", "source_location"}:
            yield from _nodes(value, seen)


def refuse_debug_printf(program, target) -> None:
    """Raise :class:`DebugPrintfUnsupportedError` for a debug print in ``program`` that ``target`` cannot carry."""
    target = str(target or "").lower()
    if target not in _REASONS or not isinstance(program, A.ShaderNode):
        return
    declared = {getattr(function, "name", None) for function in iter_functions(program)}
    for node in _nodes(program, set()):
        name = _call_name(node) if isinstance(node, A.FunctionCallNode) else None
        if name in DEBUG_PRINTF_FUNCTIONS and name not in declared:
            raise DebugPrintfUnsupportedError(
                target, name, source_location=getattr(node, "source_location", None)
            )
