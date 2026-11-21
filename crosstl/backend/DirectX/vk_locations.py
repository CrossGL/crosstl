"""DXC's ``[[vk::location(N)]]`` on HLSL stage inputs and outputs, imported as CrossGL ``@location(N)``.

DXC's SPIR-V back end places a stage variable that carries ``[[vk::location(N)]]`` at ``Location N``; its DXIL back
end ignores the attribute (``-Wignored-attributes``) and links the stages by semantic. The importer used to drop
the attribute, so CrossGL placed the variable by its semantic alone, and every target that numbers its varyings
(GLSL, SPIR-V, Metal) moved it, for example ``[[vk::location(2)]] float2 uv : TEXCOORD0`` to location 5 where
the host's vertex input or the other stage expects 2. A member or parameter now keeps its semantic and gains
``@location(N)``: the semantic is the Direct3D linkage and the location the Vulkan one.

DXC also ignores the attribute on a system value that becomes a SPIR-V built-in, such as ``SV_Position``, so the
importer drops it there. ``SV_Target<i>`` is the exception: DXC's SPIR-V output puts render target ``i`` at
``Location N``, so with ``N != i`` Direct3D and Vulkan write different attachments. CrossGL holds one render target
per output, so the importer refuses that case; with ``N == i`` the attribute changes nothing and is dropped.
"""

import re

VK_LOCATION_ATTRIBUTE = "vk::location"

_RENDER_TARGET = re.compile(r"sv_target(\d*)", re.IGNORECASE)


def _attribute_arguments(attr):
    arguments = getattr(attr, "args", None)
    if arguments is None:
        arguments = getattr(attr, "arguments", None)
    return list(arguments or [])


def _vk_location_attribute(node):
    for attr in getattr(node, "attributes", None) or []:
        if str(getattr(attr, "name", "")) == VK_LOCATION_ATTRIBUTE:
            return attr
    return None


def _literal_index(argument):
    value = getattr(argument, "value", argument)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def vk_location_index(codegen, node, owner):
    """The ``N`` of ``node``'s ``[[vk::location(N)]]``, or ``None`` when it has none.

    DXC accepts only a non-negative integer literal; anything else is refused, naming ``owner``.
    """
    attr = _vk_location_attribute(node)
    if attr is None:
        return None
    arguments = _attribute_arguments(attr)
    index = _literal_index(arguments[0]) if len(arguments) == 1 else None
    if index is None or index < 0:
        codegen.refuse_import(
            f"[[vk::location]] on {owner} needs one non-negative integer literal"
        )
        return None
    return index


def _render_target_conflict(index, semantic, owner):
    """The refusal reason when ``SV_Target<i>`` sits at ``Location index`` with ``i != index``, else ``None``."""
    match = _RENDER_TARGET.fullmatch(str(semantic or "").strip())
    if match is None:
        return None
    target = int(match.group(1) or 0)
    if target == index:
        return None
    return (
        f"[[vk::location({index})]] on {owner} with semantic {semantic}: DXC's SPIR-V output "
        f"writes render target {target} at Location {index}, so Direct3D and Vulkan write "
        "different attachments, and a CrossGL output names one render target"
    )


def vk_location_suffix(codegen, node, semantic, owner):
    """`` @location(N)`` for a member or parameter with ``[[vk::location(N)]]`` and a user semantic, else ``""``.

    ``semantic`` is the HLSL semantic text of ``node``. A system value keeps no location (see the module
    docstring), and an ``SV_Target<i>`` whose ``i`` differs from ``N`` is refused.
    """
    index = vk_location_index(codegen, node, owner)
    if index is None:
        return ""
    conflict = _render_target_conflict(index, semantic, owner)
    if conflict is not None:
        codegen.refuse_import(conflict)
        return ""
    if str(semantic or "").strip().lower().startswith("sv_"):
        return ""
    return f" @location({index})"


def check_function_return_vk_location(codegen, func):
    """Refuse a function's ``[[vk::location(N)]]`` unless its return semantic makes it redundant.

    On ``SV_Target<N>`` and on any other system value the attribute changes nothing CrossGL holds, so the caller
    drops it. A user semantic would need both the semantic and the location on the return value, which a CrossGL
    function return cannot carry.
    """
    owner = f"the return value of '{getattr(func, 'name', '<function>')}'"
    index = vk_location_index(codegen, func, owner)
    if index is None:
        return
    semantic = getattr(func, "semantic", None)
    conflict = _render_target_conflict(index, semantic, owner)
    if conflict is not None:
        codegen.refuse_import(conflict)
        return
    if str(semantic or "").strip().lower().startswith("sv_"):
        return
    codegen.refuse_import(
        f"[[vk::location({index})]] on {owner} with semantic {semantic or '(none)'}: a CrossGL "
        "function return carries one semantic, so return a structure whose member has both"
    )
