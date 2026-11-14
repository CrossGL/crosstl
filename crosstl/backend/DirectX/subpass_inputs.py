"""HLSL subpass inputs (``SubpassInput<T>``, ``SubpassInputMS<T>``) as CrossGL ``subpassInput`` types.

DXC's SPIR-V back end reads ``[[vk::input_attachment_index(i)]] SubpassInput<T> x`` as a Vulkan input attachment:
an ``OpTypeImage`` with ``Dim SubpassData`` whose sampled type is ``T``'s scalar (``float4`` when ``T`` is omitted),
multisampled for ``SubpassInputMS``. ``x.SubpassLoad()`` reads it at the fragment, and ``x.SubpassLoad(s)`` reads
sample ``s`` of the multisampled form. CrossGL spells these GLSL's way (GL_KHR_vulkan_glsl): ``subpassInput``,
``isubpassInput`` and ``usubpassInput``, with an ``MS`` suffix for the multisampled forms, read by
``subpassLoad(x)`` and ``subpassLoad(x, s)``.
"""

from __future__ import annotations

__all__ = ["SUBPASS_INPUT_TYPES", "subpass_input_crossgl_type"]

SUBPASS_INPUT_TYPES = frozenset({"SubpassInput", "SubpassInputMS"})


def subpass_input_crossgl_type(base, element_type=None):
    """The CrossGL type of HLSL ``base<element_type>``, or ``None`` when ``base`` is no subpass input."""
    if base not in SUBPASS_INPUT_TYPES:
        return None
    element = str(element_type or "float4").strip()
    prefix = (
        "u" if element.startswith("uint") else "i" if element.startswith("int") else ""
    )
    suffix = "MS" if base == "SubpassInputMS" else ""
    return f"{prefix}subpassInput{suffix}"
