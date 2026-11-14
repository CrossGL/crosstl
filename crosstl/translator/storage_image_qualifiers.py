"""Bare attributes a CrossGL storage image declaration may carry.

A storage image names its texel format with a bare attribute, as GLSL's format
layout qualifier does: ``layout(rgba8) uniform image2D img`` is ``@rgba8``
(GLSL 4.60 section 4.4.7). The other bare attributes a storage image takes are
memory qualifiers (GLSL 4.60 section 4.10, GL_KHR_memory_scope_semantics, HLSL
``globallycoherent`` and ``reordercoherent``), access qualifiers, precision
qualifiers (GLSL ES 3.10 section 4.7) and the MSL and HLSL declaration flags
below.

Any other bare name is refused. It is a format CrossGL does not have, such as
``rgb9_e5``, or a misspelt one, and without the refusal targets would write
another format in its place or, in HLSL, read the name as an array size.
"""

from .diagnostics import CODE_PREFIX, TranslationDiagnostic

STORAGE_IMAGE_QUALIFIER_NAMES = frozenset(
    {
        # Memory qualifiers.
        "coherent",
        "devicecoherent",
        "queuefamilycoherent",
        "workgroupcoherent",
        "subgroupcoherent",
        "shadercallcoherent",
        "nonprivate",
        "nontemporal",
        "globallycoherent",
        "reordercoherent",
        "volatile",
        "restrict",
        # A bare ``@access`` lacks its operand; the access validators report it.
        "access",
        # Precision qualifiers.
        "lowp",
        "mediump",
        "highp",
        # Declaration flags.
        "rasterizer_ordered",
        "metal_texture_element_half",
        "precise",
        "invariant",
        "maybe_unused",
    }
)


class UnsupportedImageFormatError(TranslationDiagnostic):
    """A storage image declaration that names a format CrossGL does not have."""

    def __init__(self, name, context):
        self.image_format = name
        super().__init__(
            f"Unsupported storage image format on {context}: @{name} is not a "
            "CrossGL storage image format or qualifier",
            code=f"{CODE_PREFIX}image-format-unsupported",
        )


def unknown_storage_image_attribute(attributes, known_names, normalize):
    """The first bare attribute name outside ``known_names``, or ``None``."""
    for attribute in attributes or []:
        name = getattr(attribute, "name", None)
        if not name or getattr(attribute, "arguments", None):
            continue
        if normalize(name) not in known_names:
            return str(name)
    return None
