"""Refusal of GLSL-style tessellation stage interfaces in HLSL output.

An HLSL hull or domain shader receives its control points as an ``InputPatch``/``OutputPatch`` parameter, its
domain location and tessellation factors as ``SV_DomainLocation``/``SV_TessFactor`` parameters or patch-constant
members, and returns its outputs. A GLSL-style CrossGL tessellation stage, as the GLSL and SPIR-V importers write
it, instead reads and writes stage-scope variables (``@input``/``@output``) and the GLSL built-in arrays
``gl_in``, ``gl_out``, ``gl_TessCoord``, ``gl_TessLevelOuter`` and ``gl_TessLevelInner``. The HLSL emitter has
no lowering from one interface to the other for these stages, and writing such an entry unchanged produces a
shader whose inputs and outputs are undeclared identifiers. It is refused instead.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

TESSELLATION_STAGES = frozenset({"tessellation_control", "tessellation_evaluation"})

GLSL_TESSELLATION_BUILTINS = frozenset(
    {
        "gl_in",
        "gl_out",
        "gl_InvocationID",
        "gl_PatchVerticesIn",
        "gl_TessCoord",
        "gl_TessLevelInner",
        "gl_TessLevelOuter",
        "gl_Position",
        "gl_PointSize",
        "gl_ClipDistance",
        "gl_CullDistance",
    }
)


class DirectXGLSLTessellationInterfaceError(TranslationDiagnostic):
    """A tessellation entry that uses GLSL-style stage variables, which an HLSL hull or domain shader cannot."""

    code = f"{CODE_PREFIX}directx-glsl-tessellation-interface"

    def __init__(self, stage, entry_name, names):
        self.stage = stage
        self.entry_name = entry_name
        self.names = tuple(names)
        self.missing_capabilities = ("directx.glsl-tessellation-interface",)
        super().__init__(
            f"DirectX {stage} entry '{entry_name}' uses GLSL-style stage variables "
            f"({', '.join(self.names)}); an HLSL hull or domain shader takes its control points, domain "
            "location and tessellation factors as patch parameters and returns its outputs, and HLSL output "
            "has no lowering from GLSL-style tessellation stage variables",
            target="directx",
        )


def _parameter_names(func):
    return {
        getattr(param, "name", None)
        for param in (
            getattr(func, "parameters", None) or getattr(func, "params", None) or []
        )
    }


def glsl_style_tessellation_names(codegen, func, stage_node):
    """Return the GLSL-style stage variables a tessellation entry reads or writes, sorted."""
    referenced = codegen.hlsl_referenced_identifier_names(func) - _parameter_names(func)
    names = set(referenced & GLSL_TESSELLATION_BUILTINS)
    for variable in getattr(stage_node, "local_variables", None) or []:
        name = getattr(variable, "name", None)
        if name in referenced and codegen.hlsl_stage_local_variable_direction(variable):
            names.add(name)
    return sorted(names)


def refuse_glsl_style_tessellation_interface(codegen, func, shader_type, stage_node):
    """Raise :class:`DirectXGLSLTessellationInterfaceError` for a GLSL-style tessellation entry."""
    if shader_type not in TESSELLATION_STAGES:
        return
    names = glsl_style_tessellation_names(codegen, func, stage_node)
    if names:
        raise DirectXGLSLTessellationInterfaceError(
            shader_type, getattr(func, "name", "<unnamed>"), names
        )
