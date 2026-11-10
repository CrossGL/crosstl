"""CrossGL names of SPIR-V Image Format operands (SPIR-V 1.6 section 3.11).

Most operands spell the GLSL format qualifier in camel case (``Rgba16Snorm`` is
``rgba16_snorm``). The packed formats do not: ``Rgb10A2``, ``Rgb10a2ui`` and
``R11fG11fB10f`` are GLSL's ``rgb10_a2``, ``rgb10_a2ui`` and ``r11f_g11f_b10f``.

An operand CrossGL has no format for, such as ``R64ui`` (``Int64ImageEXT``), is
refused: dropping it would turn the image into one whose format the runtime
chooses, which reads and writes texels differently.
"""

from ...translator.codegen.image_access_contracts import SUPPORTED_IMAGE_FORMATS
from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

PACKED_SPIRV_IMAGE_FORMATS = {
    "Rgb10A2": "rgb10_a2",
    "Rgb10a2ui": "rgb10_a2ui",
    "R11fG11fB10f": "r11f_g11f_b10f",
}


class SpirvImageFormatUnsupportedError(TranslationDiagnostic):
    """A SPIR-V image format operand that CrossGL has no storage image format for."""

    def __init__(self, image_format):
        self.image_format = image_format
        self.missing_capabilities = (f"spirv.ImageFormat.{image_format}",)
        super().__init__(
            f"SPIR-V image format {image_format} has no CrossGL storage image format",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-image-format-unsupported",
        )


def crossgl_image_format(image_format):
    """The CrossGL format of a SPIR-V Image Format operand, or ``None`` for ``Unknown``."""
    if not image_format or image_format == "Unknown":
        return None
    text = str(image_format)
    name = PACKED_SPIRV_IMAGE_FORMATS.get(text, text.replace("Snorm", "_snorm").lower())
    if name not in SUPPORTED_IMAGE_FORMATS:
        raise SpirvImageFormatUnsupportedError(text)
    return name
