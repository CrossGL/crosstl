"""Filtered texture lookups through a comparison sampler, which WGSL output refuses.

WGSL defines ``textureSample``, ``textureSampleLevel``, ``textureSampleGrad`` and ``textureSampleBias`` only with a
filtering ``sampler`` (WGSL section 17.7). A ``sampler_comparison`` is accepted only by the ``textureSampleCompare``
and ``textureSampleCompareLevel`` built-ins on depth textures. When a combined ``sampler2D`` is paired with an explicit
``comparison_sampler``, standard sampling functions cannot be lowered. The WGSL emitter refuses these lookups with
:class:`WGSLComparisonSamplerUnsupportedError` rather than write a call that naga rejects.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["WGSLComparisonSamplerUnsupportedError"]


class WGSLComparisonSamplerUnsupportedError(TranslationDiagnostic):
    """A regular texture lookup paired with a comparison sampler, which no WGSL sample built-in accepts."""

    def __init__(self, function_name, texture, builtin_name, sampler_name):
        self.function_name = function_name
        self.texture = texture
        self.builtin_name = builtin_name
        self.sampler_name = sampler_name
        self.missing_capabilities = ("wgsl.comparison-sampler",)
        super().__init__(
            f"WGSL target cannot lower {function_name}() on {texture}: "
            f"{builtin_name} takes a sampler, but {sampler_name} is a comparison sampler",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-comparison-sampler-unsupported",
        )
