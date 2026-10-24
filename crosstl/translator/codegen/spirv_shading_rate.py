"""SPIR-V shading-rate built-ins (SPV_KHR_fragment_shading_rate, GL_EXT_fragment_shading_rate, HLSL VRS).

A fragment stage reads the rate it was shaded at through ``gl_ShadingRateEXT`` (HLSL ``SV_ShadingRate`` on a
pixel-shader input); the last pre-rasterization stage writes the rate of the primitive it emits through
``gl_PrimitiveShadingRateEXT`` (``SV_ShadingRate`` on a vertex or geometry output). Both are the SPIR-V
``ShadingRateKHR`` and ``PrimitiveShadingRateKHR`` built-ins, 32-bit integer scalars that need the
``FragmentShadingRateKHR`` capability (SPV_KHR_fragment_shading_rate, BuiltIn section).

The value encodes log2 of the fragment width in bits 2-3 and log2 of the height in bits 0-1, the
``gl_ShadingRateFlag*EXT`` constants of GL_EXT_fragment_shading_rate section 7.3. D3D12's ``D3D12_SHADING_RATE``
uses the same encoding (``D3D12_SHADING_RATE_2X1`` is 0x4), so ``SV_ShadingRate`` maps to the built-in unchanged.

Mesh stages write the primitive rate per primitive through ``gl_MeshPrimitivesEXT``; that path is the mesh
lowering's and is not covered here.
"""

from __future__ import annotations

__all__ = [
    "SHADING_RATE_BUILTINS",
    "SHADING_RATE_FLAGS",
    "require_shading_rate_capability",
    "shading_rate_builtin_info",
    "shading_rate_flag_value",
    "shading_rate_semantic_builtin",
]

# GL_EXT_fragment_shading_rate section 7.1: ``flat in int gl_ShadingRateEXT`` in the fragment language and
# ``out int gl_PrimitiveShadingRateEXT`` in the vertex and geometry languages.
SHADING_RATE_BUILTINS = {
    "gl_ShadingRateEXT": ("int", "ShadingRateKHR", "Input", {"Fragment"}),
    "gl_PrimitiveShadingRateEXT": (
        "int",
        "PrimitiveShadingRateKHR",
        "Output",
        {"Vertex", "Geometry"},
    ),
}

# GL_EXT_fragment_shading_rate section 7.3.
SHADING_RATE_FLAGS = {
    "gl_ShadingRateFlag2VerticalPixelsEXT": 1,
    "gl_ShadingRateFlag4VerticalPixelsEXT": 2,
    "gl_ShadingRateFlag2HorizontalPixelsEXT": 4,
    "gl_ShadingRateFlag4HorizontalPixelsEXT": 8,
}


def shading_rate_builtin_info(name):
    """``stage_builtin_info`` entry for a shading-rate built-in, or ``None`` for other names."""
    return SHADING_RATE_BUILTINS.get(str(name))


def shading_rate_flag_value(name):
    """Value of a ``gl_ShadingRateFlag*EXT`` constant, or ``None`` for other names."""
    return SHADING_RATE_FLAGS.get(str(name))


def shading_rate_semantic_builtin(semantic, execution_model, storage_class):
    """GLSL built-in an HLSL ``SV_ShadingRate`` interface value names, or ``None`` for other semantics.

    A pixel-shader input reads the fragment's rate; a vertex or geometry output writes the primitive's rate.
    """
    if str(semantic).upper() != "SV_SHADINGRATE":
        return None
    if execution_model == "Fragment" and storage_class == "Input":
        return "gl_ShadingRateEXT"
    return "gl_PrimitiveShadingRateEXT"


def require_shading_rate_capability(codegen, builtin_name):
    """Declare the capability and extension a shading-rate built-in needs; return whether it was one."""
    if builtin_name not in {"ShadingRateKHR", "PrimitiveShadingRateKHR"}:
        return False
    codegen.require_capability("FragmentShadingRateKHR")
    codegen.require_extension("SPV_KHR_fragment_shading_rate")
    return True
