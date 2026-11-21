"""HLSL output for a stage variable that has both an HLSL semantic and ``@location(N)``.

The HLSL importer writes ``[[vk::location(N)]] float2 uv : TEXCOORD0`` as ``vec2 uv @ TexCoord0 @location(N)``:
the semantic is the Direct3D linkage and the location the Vulkan one (``crosstl/backend/DirectX/vk_locations.py``).
HLSL output used to replace such a semantic with ``TEXCOORD<N>``, which renamed a vertex input from ``POSITION`` to
``TEXCOORD0`` (the host's input layout binds by name) and dropped the Vulkan location. It now keeps the semantic and
writes ``[[vk::location(N)]]`` in front of the declaration, which DXC's SPIR-V output places at ``Location N``. DXC's
DXIL output ignores the attribute with a ``-Wignored-attributes`` warning, so the attribute sits inside
``#ifdef __spirv__``, the macro DXC defines when it compiles for SPIR-V, as ``vk::image_format`` does.

A semantic counts as declared when it is an HLSL semantic name: one of the base names Direct3D defines
(``POSITION``, ``TEXCOORD``, ``COLOR``, ...) with an optional index, in any case, or an upper-case user semantic
such as ``VECTOR0``. The direction and placement markers other importers write before ``@location`` (``@input``,
``@output``, ``@patch``, ``@component``, ...) are lower case and keep the old ``TEXCOORD<N>`` placement.
"""

import re

from ..validation import HLSL_SEMANTIC_METADATA_BASE_NAMES

_UPPER_CASE_USER_SEMANTIC = re.compile(r"[A-Z_][A-Z0-9_]*")


def is_hlsl_semantic_name(semantic):
    """Whether ``semantic`` (CrossGL attribute text) names an HLSL semantic rather than a placement marker."""
    text = str(semantic or "").strip()
    if not text or text.lower().startswith(("sv_", "gl_")):
        return False
    if _UPPER_CASE_USER_SEMANTIC.fullmatch(text):
        return True
    base = re.sub(r"\d+$", "", text).lower()
    return base in HLSL_SEMANTIC_METADATA_BASE_NAMES


def hlsl_location_with_declared_semantic(codegen, node):
    """``N`` when ``node`` has ``@location(N)`` and a declared HLSL semantic, else ``None``."""
    index = codegen.hlsl_location_attribute_index(node)
    if index is None:
        return None
    if not is_hlsl_semantic_name(codegen.semantic_from_node(node)):
        return None
    return index


def hlsl_location_replaces_semantic(codegen, node):
    """Whether ``node``'s ``@location(N)`` decides its HLSL semantic (``TEXCOORD<N>``)."""
    return (
        codegen.hlsl_location_attribute_index(node) is not None
        and hlsl_location_with_declared_semantic(codegen, node) is None
    )


def hlsl_vk_location_prefix(codegen, node, indent=""):
    """The guarded ``[[vk::location(N)]]`` lines for a member or parameter that keeps its declared semantic.

    The lines end with a newline, so the declaration follows on its own line; ``indent`` indents the attribute.
    Returns ``""`` when ``node`` needs no location.
    """
    index = hlsl_location_with_declared_semantic(codegen, node)
    if index is None:
        return ""
    return f"#ifdef __spirv__\n{indent}[[vk::location({index})]]\n#endif\n"
