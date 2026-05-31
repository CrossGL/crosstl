"""Preprocessor support for Vulkan GLSL/SPIR-V source imports."""

from crosstl.backend.GLSL.preprocessor import GLSLPreprocessor


class VulkanPreprocessor(GLSLPreprocessor):
    """GLSL-compatible preprocessor used before Vulkan source lexing."""

