"""GLSL built-in values HLSL reads through another system value, through an intrinsic, or not at all.

The HLSL generator spells a built-in value attribute with the system-value semantic its ``semantic_map`` lists and
keeps any other name as written. DXC accepts any identifier as a user semantic, so a ``gl_`` built-in that table did
not list became a varying no stage writes (``float3 b : gl_BaryCoordNV``), and a body read of ``gl_SubgroupSize``
outside a compute stage, or of ``gl_NumSamples`` anywhere, became an undeclared name.

This module maps the built-ins HLSL can express:

* ``gl_SampleID`` takes ``SV_SampleIndex`` in the generator itself, not only after ``translate``'s canonicalization;
* ``gl_BaryCoordNV`` and ``gl_BaryCoordNoPerspNV`` take ``SV_Barycentrics`` (Shader Model 6.1), as their EXT
  equivalents do (GL_EXT_fragment_shader_barycentric defines the same values as the NV extension);
* ``gl_ShadingRateEXT`` (a fragment input) and ``gl_PrimitiveShadingRateEXT`` (a vertex, geometry or mesh output)
  take ``SV_ShadingRate`` (Shader Model 6.4). GL_EXT_fragment_shading_rate's flags and ``D3D12_SHADING_RATE`` encode
  a rate the same way, log2 of the width in bits 2 and 3 and log2 of the height in bits 0 and 1, so the value passes
  unchanged. DXIL requires ``uint`` ("SV_ShadingRate must be uint"), so an ``int`` input parameter is read through a
  ``uint`` one and a structure member is declared ``uint``;
* on an entry parameter, the inputs HLSL reads with an intrinsic become locals initialised from it:
  ``gl_HelperInvocation`` (``IsHelperLane()``), ``gl_SubgroupSize`` and ``gl_SubgroupInvocationID`` outside compute
  stages (``WaveGetLaneCount()``, ``WaveGetLaneIndex()``; the compute path has its own), the lane masks
  ``gl_SubgroupEqMask`` to ``gl_SubgroupLtMask`` (built from ``WaveGetLaneIndex()``) and ``gl_NumSamples`` in a pixel
  shader (``GetRenderTargetSampleCount()``, which DXIL allows only there);
* ``gl_FragSizeEXT`` is derived from an ``SV_ShadingRate`` input, and ``gl_FullyCoveredEXT`` reads
  ``SV_InnerCoverage`` (conservative rasterization tier 3);
* body reads of ``gl_SubgroupSize``, ``gl_SubgroupInvocationID`` and ``gl_NumSamples`` take the same intrinsics.

:func:`refuse_unmapped_hlsl_builtin` refuses every other CrossGL ``gl_`` built-in that reaches a semantic. Direct3D
has no counterpart for the NV per-view, secondary-position and viewport-mask outputs, ``gl_DeviceIndex``,
``gl_FragInvocationCountEXT``, the NV SM built-ins or Metal's quad-group indices, and its three-weight
``SV_Barycentrics`` does not order its weights the way GL_AMD_shader_explicit_vertex_parameter's two-weight
barycentrics do.
"""

from __future__ import annotations

from ..ast import IdentifierNode, VariableNode
from ..semantics.builtin_names import is_crossgl_gl_builtin
from ..stage_utils import normalize_stage_name
from ..wave_operations import SUBGROUP_LANE_MASK_BUILTINS
from .array_utils import split_array_type_suffix
from .hlsl_wave_extended import hlsl_subgroup_lane_mask_expression

__all__ = [
    "HLSL_GLSL_SYSTEM_VALUES",
    "HLSL_NOPERSPECTIVE_BARYCENTRICS",
    "hlsl_builtin_input_parameter",
    "hlsl_stage_builtin_read",
    "refuse_unmapped_hlsl_builtin",
]

# GLSL built-ins the generator's semantic table lacked. ``translate`` rewrites ``gl_SampleID`` to ``SV_SampleIndex``
# (:mod:`..semantics.builtin_values`), but a program the HLSL importer wrote reached the generator with the GLSL name.
HLSL_GLSL_SYSTEM_VALUES = {
    "gl_SampleID": "SV_SampleIndex",
    "gl_BaryCoordNV": "SV_Barycentrics",
    "gl_BaryCoordNoPerspNV": "SV_Barycentrics",
    "gl_ShadingRateEXT": "SV_ShadingRate",
    "gl_PrimitiveShadingRateEXT": "SV_ShadingRate",
}

HLSL_NOPERSPECTIVE_BARYCENTRICS = frozenset(
    {"gl_BaryCoordNoPerspEXT", "gl_BaryCoordNoPerspNV"}
)

_WAVE_VALUES = {
    "gl_SubgroupSize": "WaveGetLaneCount()",
    "gl_SubgroupInvocationID": "WaveGetLaneIndex()",
}

_SCALARS = frozenset({"bool", "int", "uint"})

_COLOUR_PREFIXES = ("gl_fragcolor", "gl_fragdata")


def _shadowed(codegen, name):
    for table in ("local_variable_types", "global_variable_types"):
        if name in (getattr(codegen, table, None) or {}):
            return True
    return name in (getattr(codegen, "current_identifier_aliases", None) or {})


def _sample_count(codegen, stage):
    """``GetRenderTargetSampleCount()`` as GLSL's ``int``, refused where DXIL does not allow it."""
    if stage in codegen.stage_entry_types() and stage != "fragment":
        return codegen.refuse_lowering(
            "gl_NumSamples: HLSL reads the sample count with GetRenderTargetSampleCount(), "
            f"which DXIL allows only in pixel shaders, not in a {stage} stage",
            "1",
        )
    if codegen.target_profile == "dx10":
        return codegen.refuse_lowering(
            "gl_NumSamples: GetRenderTargetSampleCount() requires Shader Model 4.1, "
            "which DirectX profile dx10 does not provide",
            "1",
        )
    return "int(GetRenderTargetSampleCount())"


def hlsl_stage_builtin_read(codegen, expr):
    """HLSL for a body read of a built-in HLSL reads with an intrinsic, or ``None`` for other expressions."""
    if not isinstance(expr, (IdentifierNode, VariableNode)):
        return None
    name = getattr(expr, "name", None)
    if name != "gl_NumSamples" and name not in _WAVE_VALUES:
        return None
    if _shadowed(codegen, name):
        return None
    stage = normalize_stage_name(getattr(codegen, "current_hlsl_semantic_stage", None))
    if name == "gl_NumSamples":
        return _sample_count(codegen, stage)
    # Compute entries read these through their own parameters, including software subgroups.
    if stage == "compute" or codegen.software_subgroup_width is not None:
        return None
    codegen.validate_directx_wave_intrinsic_profile(name)
    return _WAVE_VALUES[name]


def _input_expression(codegen, name, stage):
    """``(expression, its HLSL type)`` for an input HLSL reads with an intrinsic, or ``None``."""
    if name == "gl_HelperInvocation" and stage == "fragment":
        return codegen.hlsl_helper_invocation_expression(IdentifierNode(name)), "bool"
    if name in _WAVE_VALUES and stage != "compute":
        if codegen.software_subgroup_width is not None:
            return None
        codegen.validate_directx_wave_intrinsic_profile(name)
        return _WAVE_VALUES[name], "uint"
    if name in SUBGROUP_LANE_MASK_BUILTINS:
        return hlsl_subgroup_lane_mask_expression(codegen, name), "uint4"
    if name == "gl_NumSamples":
        return _sample_count(codegen, stage), "int"
    return None


def _initialiser(codegen, name, base_type, expression, expression_type):
    if base_type == expression_type:
        return expression
    if base_type in _SCALARS and expression_type in _SCALARS:
        return f"{base_type}({expression})"
    if base_type in {"int2", "uint2"} and expression_type in {"int2", "uint2"}:
        return f"{base_type}({expression})"
    return codegen.refuse_lowering(
        f"@{name} is an HLSL {expression_type} value; a {base_type} parameter cannot hold it",
        expression,
    )


def _local(codegen, parameter, base_type, initialiser):
    local_name = codegen.hlsl_declaration_identifier_name(parameter.name)
    codegen.local_variable_types[local_name] = base_type
    return f"{base_type} {local_name} = {initialiser};"


def _bridge_parameter(codegen, parameter, suffix, semantic, emitted_param_names):
    """Declare a ``uint`` entry parameter for ``semantic`` and return ``(declaration, its name)``."""
    used_names = set(emitted_param_names)
    used_names.update(codegen.local_variable_types)
    used_names.update(codegen.global_variable_types)
    used_names.update(codegen.current_identifier_aliases.values())
    local_name = codegen.hlsl_declaration_identifier_name(parameter.name)
    bridge_name = codegen.hlsl_unique_local_identifier(
        f"{local_name}_{suffix}", used_names
    )
    emitted_param_names.add(bridge_name)
    codegen.current_identifier_reserved_names.add(bridge_name)
    codegen.local_variable_types[bridge_name] = "uint"
    return f"uint {bridge_name} : {semantic}", bridge_name


def _converted_input(codegen, parameter, name, base_type, stage, emitted_param_names):
    """``(declaration, prologue)`` for an input read through another ``uint`` system value, or ``None``."""
    if stage != "fragment":
        return None
    if name == "gl_ShadingRateEXT":
        if base_type != "int":
            return None  # a uint input takes SV_ShadingRate directly
        declaration, bridge = _bridge_parameter(
            codegen, parameter, "shadingRate", "SV_ShadingRate", emitted_param_names
        )
        return declaration, _local(codegen, parameter, base_type, f"int({bridge})")
    if name == "gl_FragSizeEXT":
        declaration, bridge = _bridge_parameter(
            codegen, parameter, "shadingRate", "SV_ShadingRate", emitted_param_names
        )
        codegen.required_hlsl_fragment_shading_rate_helper = True
        size = f"CrossGLFragmentSizeFromShadingRate({bridge})"
        return declaration, _local(
            codegen,
            parameter,
            base_type,
            _initialiser(codegen, name, base_type, size, "uint2"),
        )
    if name == "gl_FullyCoveredEXT":
        declaration, bridge = _bridge_parameter(
            codegen, parameter, "innerCoverage", "SV_InnerCoverage", emitted_param_names
        )
        return declaration, _local(
            codegen,
            parameter,
            base_type,
            _initialiser(codegen, name, base_type, f"({bridge} != 0u)", "bool"),
        )
    return None


def hlsl_builtin_input_parameter(
    codegen, parameter, param_type, semantic, stage, emitted_param_names
):
    """``(declaration or None, prologue)`` for an entry parameter whose built-in HLSL reads another way, or ``None``.

    A ``None`` declaration drops the parameter from the signature; the prologue declares the local the body reads.
    """
    name = str(semantic or "").strip()
    stage = normalize_stage_name(stage)
    if not name.startswith("gl_") or stage not in codegen.stage_entry_types():
        return None
    parameter_name = getattr(parameter, "name", None)
    if not isinstance(parameter_name, str) or not parameter_name:
        return None
    base_type, array_suffix = split_array_type_suffix(str(param_type))
    if array_suffix:
        return None
    read = _input_expression(codegen, name, stage)
    if read is not None:
        expression, expression_type = read
        initialiser = _initialiser(
            codegen, name, base_type, expression, expression_type
        )
        return None, _local(codegen, parameter, base_type, initialiser)
    return _converted_input(
        codegen, parameter, name, base_type, stage, emitted_param_names
    )


def refuse_unmapped_hlsl_builtin(codegen, semantic, mapped):
    """``mapped``, after refusing a CrossGL ``gl_`` built-in the HLSL semantic table left as a user semantic."""
    text = str(semantic or "").strip()
    if not str(mapped).lower().startswith("gl_") or not is_crossgl_gl_builtin(text):
        return mapped
    if text.lower().startswith(_COLOUR_PREFIXES):
        return mapped
    return codegen.refuse_lowering(
        f"HLSL has no system value for the built-in @{text}; "
        "it would become a user semantic no stage writes",
        mapped,
    )
