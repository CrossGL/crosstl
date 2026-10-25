"""Subpass-input reads imported from SPIR-V.

``OpImageRead`` of an image whose ``Dim`` is ``SubpassData`` reads the input attachment at the current fragment's
framebuffer position plus ``Coordinate`` (SPIR-V 1.6 section 3.49.10, ``OpImageRead``). CrossGL spells that read
``subpassLoad(image[, sample])``, which has no offset, exactly as GLSL does (GL_KHR_vulkan_glsl). glslang and DXC
emit the constant ``(0, 0)`` coordinate, so a read with that coordinate is imported as ``subpassLoad``; any other
coordinate is refused rather than dropped.
"""

from .ray_tracing_import import SpirvInstructionUnsupportedError
from .VulkanAst import FunctionCallNode


def is_subpass_image(image_operand, expression_type_ids, types):
    """Whether ``image_operand`` is a value of an ``OpTypeImage`` with ``Dim`` ``SubpassData``."""
    image_type = types.get(expression_type_ids.get(image_operand), {})
    return image_type.get("kind") == "image" and image_type.get("dim") == "SubpassData"


def _is_zero(value):
    text = str(getattr(value, "value", value)).strip()
    try:
        return int(text, 0) == 0
    except ValueError:
        return False


def _is_zero_offset(coordinate):
    """Whether a constant coordinate is the all-zero integer vector ``(0, 0)``."""
    if isinstance(coordinate, FunctionCallNode):
        args = list(coordinate.args or [])
        return bool(args) and all(_is_zero(arg) for arg in args)
    return False


def subpass_load_expression(image, coordinate_operand, constants, sample=None):
    """``subpassLoad(image[, sample])`` for a subpass read at offset ``(0, 0)``; refuse any other offset."""
    if not _is_zero_offset(constants.get(coordinate_operand)):
        raise SpirvInstructionUnsupportedError(
            "OpImageRead",
            "reads a subpass input at a coordinate offset that is not the constant (0, 0); CrossGL subpassLoad "
            "reads the current fragment's attachment value and has no offset",
        )
    args = [image] if sample is None else [image, sample]
    return FunctionCallNode("subpassLoad", args)
