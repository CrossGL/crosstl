"""Shader execution reordering: HLSL's ``dx::MaybeReorderThread``, kept for Direct3D and refused for other targets.

A ray generation shader asks the scheduler to reorder the threads of a wave around a coherence hint with
``dx::MaybeReorderThread(CoherenceHint, NumCoherenceHintBitsFromLSB)``, the overload that takes no hit object (HLSL
Shader Model 6.9; DirectX Raytracing 1.2, "Shader Execution Reordering"). CrossGL spells the call the way HLSL does,
``dx::MaybeReorderThread(hint, bits)``, and the DirectX generator writes it back unchanged, which DXC compiles.

Vulkan has the operation through the ``shader_invocation_reorder`` extensions (``reorderThreadEXT`` in GLSL,
``OpReorderThreadWithHintEXT`` in SPIR-V) and Slang spells it ``ReorderThread``, but the GLSL, SPIR-V and Slang
generators do not lower the call to those forms. Metal Shading Language, GLSL ES, WGSL, CUDA C++, HIP, rust-gpu and
Mojo have no such intrinsic. The generators copied the HLSL spelling into their output anyway, which the target
compilers reject: the Metal compiler reports ``error: use of undeclared identifier 'dx'``, and ``glslangValidator``
reports ``'::' : not supported`` on the GLSL. :func:`refuse_shader_execution_reordering` refuses every target but
Direct3D with :class:`ShaderExecutionReorderingUnsupportedError` before generation, so a program that reorders its
threads never reaches a target as a call its compiler cannot resolve.

The overloads that take a hit object share the name and are refused by this rule as well. The ``dx::HitObject``
type they use and the ``dx::HitObject::`` static methods are refused by :mod:`.hit_objects`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions

__all__ = [
    "REORDER_THREAD_CALLS",
    "REORDERING_TARGETS",
    "ShaderExecutionReorderingUnsupportedError",
    "refuse_shader_execution_reordering",
]

# The HLSL spelling of the Shader Model 6.9 reorder intrinsic.
REORDER_THREAD_CALLS = frozenset({"dx::MaybeReorderThread"})
# Targets that keep the HLSL spelling: the Direct3D generators and CrossGL itself.
REORDERING_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "cgl", "crossgl"}
)
_REASONS = {
    "vulkan": (
        "the SPIR-V generator does not lower it to OpReorderThreadWithHintEXT "
        "(SPV_EXT_shader_invocation_reorder)"
    ),
    "metal": "Metal Shading Language has no shader execution reordering intrinsic",
    "opengl": (
        "the GLSL generator does not lower it to reorderThreadEXT "
        "(GL_EXT_shader_invocation_reorder)"
    ),
    "webgl": "GLSL ES 3.00 has no shader execution reordering intrinsic",
    "wgsl": "WGSL has no shader execution reordering intrinsic",
    "cuda": "CUDA C++ has no shader execution reordering intrinsic",
    "hip": "HIP C++ has no shader execution reordering intrinsic",
    "slang": "the Slang generator does not lower it to Slang's ReorderThread",
    "rust": "rust-gpu has no shader execution reordering intrinsic",
    "mojo": "Mojo has no shader execution reordering intrinsic",
}


class ShaderExecutionReorderingUnsupportedError(TranslationDiagnostic):
    """A shader execution reordering call for a target whose generator does not carry it."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.shader-execution-reordering",)
        reason = _REASONS.get(
            target, f"{target} has no shader execution reordering operation"
        )
        super().__init__(
            f"{name} (HLSL Shader Model 6.9 shader execution reordering) cannot be carried to {target}: "
            f"{reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-shader-execution-reordering-unsupported",
        )


def refuse_shader_execution_reordering(program, target) -> None:
    """Raise :class:`ShaderExecutionReorderingUnsupportedError` for the first reorder call ``target`` cannot carry."""
    target = str(target or "").lower()
    if target in REORDERING_TARGETS or not isinstance(program, A.ShaderNode):
        return
    # A program that declares a function of this name calls its own function, which every target can carry.
    declared = {getattr(function, "name", None) for function in iter_functions(program)}
    for node in program.walk():
        if not isinstance(node, A.FunctionCallNode):
            continue
        name = call_name(node)
        if name in REORDER_THREAD_CALLS and name not in declared:
            raise ShaderExecutionReorderingUnsupportedError(
                target, name, source_location=getattr(node, "source_location", None)
            )
