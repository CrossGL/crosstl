"""SPIR-V built-ins of GLSL extensions the stage tables do not list, and the refusal of every other ``gl_`` name.

The SPIR-V emitter decorates a stage-interface value with ``BuiltIn`` when its ``gl_`` attribute names an entry of
``compute_builtin_info`` or ``stage_builtin_info``, and gives any other value a user ``Location``. A built-in those
tables did not list, such as a fragment input ``@gl_HelperInvocation``, therefore became a varying no stage writes.

:func:`extension_builtin_info` adds the built-ins whose SPIR-V form is one decorated variable:

* ``gl_HelperInvocation`` (``HelperInvocation``, SPIR-V 1.0 core) and ``gl_FullyCoveredEXT``
  (``SPV_EXT_fragment_fully_covered``) in a fragment stage;
* the AMD barycentric coordinates of ``GL_AMD_shader_explicit_vertex_parameter``
  (``SPV_AMD_shader_explicit_vertex_parameter``) in a fragment stage;
* ``gl_DeviceIndex`` (``SPV_KHR_device_group``) and the ``GL_NV_shader_sm_builtins`` values
  (``SPV_NV_shader_sm_builtins``), which every stage reads.

``gl_BaryCoordNV`` and ``gl_BaryCoordNoPerspNV`` take the ``SPV_KHR_fragment_shader_barycentric`` built-ins: the
NV extension's ``BaryCoordNV`` and ``BaryCoordNoPerspNV`` are the same enumerants (5286 and 5287) with the same
meaning (SPIR-V 1.6 section 3.21, ``BuiltIn``).

:func:`unmapped_builtin_reason` names a ``gl_`` built-in that reached the user-location path anyway, which the
emitter refuses: CrossGL defines it (:mod:`..semantics.builtin_names`) but SPIR-V has no built-in for that stage
and direction here, as for ``gl_NumSamples`` (a GLSL uniform Vulkan does not provide) or Metal's quad-group
indices. Colour outputs (``gl_FragColor``, ``gl_FragData``) are attachment locations, not built-ins.
"""

from __future__ import annotations

from ..semantics.builtin_names import is_crossgl_gl_builtin

__all__ = [
    "BUILTIN_ALIASES",
    "extension_any_stage_builtin_info",
    "extension_builtin_info",
    "require_extension_builtin_capability",
    "unmapped_builtin_reason",
]

# ``compute_builtin_info`` entries: inputs of every stage.
_ANY_STAGE_INPUTS = {
    # GL_EXT_device_group: ``in highp int gl_DeviceIndex``.
    "gl_DeviceIndex": ("int", "DeviceIndex", "Input"),
    # GL_NV_shader_sm_builtins: ``in highp uint`` each.
    "gl_SMCountNV": ("uint", "SMCountNV", "Input"),
    "gl_SMIDNV": ("uint", "SMIDNV", "Input"),
    "gl_WarpIDNV": ("uint", "WarpIDNV", "Input"),
    "gl_WarpsPerSMNV": ("uint", "WarpsPerSMNV", "Input"),
}

_FRAGMENT = {"Fragment"}

# ``stage_builtin_info`` entries.
_STAGE_BUILTINS = {
    "gl_HelperInvocation": ("bool", "HelperInvocation", "Input", _FRAGMENT),
    "gl_FullyCoveredEXT": ("bool", "FullyCoveredEXT", "Input", _FRAGMENT),
    # GL_AMD_shader_explicit_vertex_parameter: ``in vec2`` each, the pull model ``in vec3``.
    "gl_BaryCoordNoPerspAMD": ("vec2", "BaryCoordNoPerspAMD", "Input", _FRAGMENT),
    "gl_BaryCoordNoPerspCentroidAMD": (
        "vec2",
        "BaryCoordNoPerspCentroidAMD",
        "Input",
        _FRAGMENT,
    ),
    "gl_BaryCoordNoPerspSampleAMD": (
        "vec2",
        "BaryCoordNoPerspSampleAMD",
        "Input",
        _FRAGMENT,
    ),
    "gl_BaryCoordSmoothAMD": ("vec2", "BaryCoordSmoothAMD", "Input", _FRAGMENT),
    "gl_BaryCoordSmoothCentroidAMD": (
        "vec2",
        "BaryCoordSmoothCentroidAMD",
        "Input",
        _FRAGMENT,
    ),
    "gl_BaryCoordSmoothSampleAMD": (
        "vec2",
        "BaryCoordSmoothSampleAMD",
        "Input",
        _FRAGMENT,
    ),
    "gl_BaryCoordPullModelAMD": ("vec3", "BaryCoordPullModelAMD", "Input", _FRAGMENT),
}

# Upper-case spellings, as ``spirv_builtin_alias`` looks names up.
BUILTIN_ALIASES = {
    "GL_BARYCOORDNV": "gl_BaryCoordEXT",
    "GL_BARYCOORDNOPERSPNV": "gl_BaryCoordNoPerspEXT",
}

# BuiltIn -> (capability or None, extension).
_REQUIREMENTS = {
    "DeviceIndex": ("DeviceGroup", "SPV_KHR_device_group"),
    "FullyCoveredEXT": ("FragmentFullyCoveredEXT", "SPV_EXT_fragment_fully_covered"),
    **{
        builtin: ("ShaderSMBuiltinsNV", "SPV_NV_shader_sm_builtins")
        for builtin in ("SMCountNV", "SMIDNV", "WarpIDNV", "WarpsPerSMNV")
    },
    **{
        info[1]: (None, "SPV_AMD_shader_explicit_vertex_parameter")
        for name, info in _STAGE_BUILTINS.items()
        if name.endswith("AMD")
    },
}

_COLOUR_PREFIXES = ("gl_fragcolor", "gl_fragdata")


def extension_any_stage_builtin_info(name):
    """``compute_builtin_info`` entry for an extension built-in every stage reads, or ``None``."""
    return _ANY_STAGE_INPUTS.get(str(name))


def extension_builtin_info(name):
    """``stage_builtin_info`` entry for a stage-specific extension built-in, or ``None``."""
    return _STAGE_BUILTINS.get(str(name))


def require_extension_builtin_capability(codegen, builtin_name):
    """Declare the capability and extension an extension built-in needs; return whether it was one."""
    requirement = _REQUIREMENTS.get(str(builtin_name))
    if requirement is None:
        return False
    capability, extension = requirement
    if capability is not None:
        codegen.require_capability(capability)
    codegen.require_extension(extension)
    return True


def unmapped_builtin_reason(semantic, storage_class, execution_model):
    """Refusal text for a CrossGL ``gl_`` built-in about to take a user location, or ``None`` for other values."""
    text = str(semantic or "")
    if not text.lower().startswith("gl_") or not is_crossgl_gl_builtin(text):
        return None
    if text.lower().startswith(_COLOUR_PREFIXES):
        return None
    direction = "input" if storage_class == "Input" else "output"
    stage = execution_model or "this"
    return (
        f"SPIR-V has no {direction} built-in for @{text} in a {stage} stage; "
        "it would become a user location no stage matches"
    )
