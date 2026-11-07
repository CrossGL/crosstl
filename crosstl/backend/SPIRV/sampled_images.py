"""``OpSampledImage`` of a separate image and sampler, imported as CrossGL's separate-sampler texture calls.

SPIR-V samples through a sampled image. A module with separate images and samplers (every module DXC writes, and
GLSL that uses ``texture2D`` with ``sampler``) forms one with ``OpSampledImage`` right before the lookup (SPIR-V
3.37.10). The importer wrote that as GLSL's constructor, ``texture(sampler2D(image, sampler), uv)``, which no emitter
but GLSL for Vulkan reads: the others take a declared texture as the first argument. CrossGL's own spelling for a
lookup through a separate sampler names both resources, ``texture(image, sampler, uv)``, as the HLSL importer writes
it, and every emitter reads that.

:func:`sampled_image_call` builds the constructor and remembers its operands. :func:`sampled_image_arguments` gives
a lookup's leading arguments, the image and the sampler for such a constructor, and :func:`sampled_image_image` the
image ``OpImage`` takes back out of it.
"""

from __future__ import annotations

__all__ = ["sampled_image_arguments", "sampled_image_call", "sampled_image_image"]

_OPERANDS = "spirv_sampled_image_operands"


def sampled_image_call(call_type, type_name, image, sampler):
    """``type_name(image, sampler)`` as a ``call_type`` node that records it combines ``image`` and ``sampler``."""
    call = call_type(type_name, [image, sampler])
    setattr(call, _OPERANDS, (image, sampler))
    return call


def sampled_image_arguments(expression):
    """The leading arguments of a lookup through ``expression``: its image and sampler, or ``expression`` itself."""
    operands = getattr(expression, _OPERANDS, None)
    if operands is None:
        return [expression]
    return list(operands)


def sampled_image_image(expression):
    """The image ``OpImage`` takes out of ``expression``: its image when it combines one, else ``expression``."""
    operands = getattr(expression, _OPERANDS, None)
    return expression if operands is None else operands[0]
