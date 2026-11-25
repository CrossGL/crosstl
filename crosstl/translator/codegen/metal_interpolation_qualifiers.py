"""The one Metal interpolation attribute of a stage value whose CrossGL attributes name its mode and sampling.

CrossGL spells interpolation as GLSL and HLSL do, with up to two attributes: a mode (``@ flat``, ``@ smooth``,
``@ noperspective``) and a sampling location (``@ centroid``, ``@ sample``), so ``noperspective centroid float2 uv``
imports as ``vec2 uv @ noperspective @ centroid``. Metal names the pair with a single attribute from its sampling
and interpolation attributes: ``center_perspective`` (the default), ``center_no_perspective``,
``centroid_perspective``, ``centroid_no_perspective``, ``sample_perspective``, ``sample_no_perspective`` and
``flat``. The generator used to take the first attribute it recognised, so ``@ noperspective @ centroid`` lost its
centroid sampling. A combined Metal spelling (``@ centroid_no_perspective``) is still accepted as both halves.
"""

from __future__ import annotations

from ..refusals import MetalRefusal

# Interpolation modes, and the Metal suffix each one takes after the sampling location.
_MODES = {
    "flat": "flat",
    "nointerpolation": "flat",
    "smooth": "perspective",
    "linear": "perspective",
    "perspective": "perspective",
    "noperspective": "no_perspective",
    "no_perspective": "no_perspective",
}

_SAMPLINGS = ("center", "centroid", "sample")

# Spellings that state both halves: Metal's own names and the HLSL-style names some importers write.
_COMBINED = {
    "center_perspective": ("perspective", "center"),
    "center_no_perspective": ("no_perspective", "center"),
    "centroid_perspective": ("perspective", "centroid"),
    "centroid_no_perspective": ("no_perspective", "centroid"),
    "sample_perspective": ("perspective", "sample"),
    "sample_no_perspective": ("no_perspective", "sample"),
    "linear_centroid": ("perspective", "centroid"),
    "linear_sample": ("perspective", "sample"),
    "linear_noperspective": ("no_perspective", None),
    "linear_noperspective_centroid": ("no_perspective", "centroid"),
    "linear_noperspective_sample": ("no_perspective", "sample"),
}


def _split(name):
    """The (mode, sampling) halves that one CrossGL attribute name states, or None for any other attribute."""
    normalized = str(name or "").lower()
    if normalized.startswith(("metal_", "msl_")):
        normalized = normalized.split("_", 1)[1]
    if normalized in _MODES:
        return _MODES[normalized], None
    if normalized in _SAMPLINGS:
        return None, normalized
    return _COMBINED.get(normalized)


def metal_interpolation_qualifier(node):
    """The Metal interpolation attribute name for ``node``'s CrossGL metadata, or None when it states none.

    Raises MetalRefusal when the attributes state two different modes or two different sampling locations, which
    no single Metal attribute expresses.
    """
    modes = []
    samplings = []
    # A struct member carries its qualifiers as attributes; a parameter keeps ``nointerpolation vec2 uv`` as a
    # qualifier, the form the Slang importer writes.
    names = [
        getattr(attribute, "name", None)
        for attribute in getattr(node, "attributes", None) or []
    ]
    names += [str(qualifier) for qualifier in getattr(node, "qualifiers", None) or []]
    for attribute_name in names:
        halves = _split(attribute_name)
        if halves is None:
            continue
        mode, sampling = halves
        if mode is not None and mode not in modes:
            modes.append(mode)
        if sampling is not None and sampling not in samplings:
            samplings.append(sampling)
    if not modes and not samplings:
        return None
    name = getattr(node, "name", None) or "?"
    if len(modes) > 1 or len(samplings) > 1:
        stated = ", ".join(modes + samplings)
        raise MetalRefusal(
            f"Metal stage value '{name}' states conflicting interpolation ({stated});"
            " one Metal interpolation attribute names one mode and one sampling location"
        )
    mode = modes[0] if modes else "perspective"
    if mode == "flat":
        # Flat values do not interpolate, so where they are sampled does not matter.
        return "flat"
    if mode == "perspective" and not samplings:
        # ``@ smooth`` alone is Metal's default interpolation, which needs no attribute.
        return None
    return f"{samplings[0]}_{mode}" if samplings else f"center_{mode}"
