"""GLSL's bias form of ``texture()`` on WGSL output.

``texture(gsampler, P, bias)`` adds ``bias`` to the implicit level of detail (GLSL 4.60 section 8.9.2). The WGSL
emitter writes it as ``textureSampleBias(t, s, coords, [array_index,] bias)`` with the texture's split sampler. WGSL
defines ``textureSampleBias`` only for ``texture_2d``, ``texture_2d_array``, ``texture_3d``, ``texture_cube`` and
``texture_cube_array`` (WGSL section 17.7.7), so a bias on a 1D texture is refused with
:class:`WGSLTextureBiasUnsupportedError`, as is a bias on an element of a sampled texture array selected at run time,
which the emitter reads through a dispatch helper that takes no bias.

``textureSampleBias`` needs implicit derivatives, which WGSL has only in fragment shaders. ``legalize_implicit_lod``
refuses a bias reached from a vertex stage and leaves compute stages to the emitters; the WGSL emitter refuses a bias
in a compute entry point with ``ImplicitLodStageUnsupportedError``. Shadow samplers do not reach the emitter with a
bias, because ``legalize_shadow_samples`` refuses them (WGSL has no biased comparison), and integer samplers are
refused by :mod:`.wgsl_integer_textures`.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["WGSLTextureBiasUnsupportedError", "has_texture_sample_bias"]

# The WGSL sampled texture types that have a textureSampleBias overload (WGSL section 17.7.7).
TEXTURE_SAMPLE_BIAS_TEXTURES = frozenset(
    {
        "texture_2d",
        "texture_2d_array",
        "texture_3d",
        "texture_cube",
        "texture_cube_array",
    }
)


class WGSLTextureBiasUnsupportedError(TranslationDiagnostic):
    """A ``texture()`` bias that no ``textureSampleBias`` overload takes."""

    def __init__(self, function_name, texture, reason):
        self.function_name = function_name
        self.texture = texture
        self.missing_capabilities = ("wgsl.texture-sample-bias",)
        super().__init__(
            f"WGSL target cannot lower {function_name}() with a bias on {texture}: {reason}",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-texture-bias-unsupported",
        )


def has_texture_sample_bias(texture_type):
    """Whether ``texture_type``, a WGSL sampled texture type such as ``texture_2d<f32>``, has a bias overload."""
    return re.sub(r"<[^>]*>$", "", texture_type or "") in TEXTURE_SAMPLE_BIAS_TEXTURES
