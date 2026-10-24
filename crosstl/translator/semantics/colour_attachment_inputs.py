"""Fragment inputs that read colour attachments (MSL programmable blending) on targets without them.

A Metal fragment function can take a colour attachment as an input, as Apple's single-pass deferred renderers do:
``GBufferData gbuffer`` whose members are ``[[color(n)]]`` reads the G-buffer the earlier draws wrote to the
same pixel (MSL 5.2.3.4, "programmable blending"). CrossGL keeps those members as ``@gl_FragColor<n>``. The other
targets have no such read in the shader: Direct3D has no framebuffer fetch, Vulkan reads earlier subpasses through
input attachments that the render pass must declare, and desktop GLSL has none (``GL_EXT_shader_framebuffer_fetch``
is a GLSL ES extension). The generators emitted them anyway: HLSL as ``SV_Target`` inputs, which DXC rejects, and
GLSL and SPIR-V as interpolated inputs at a location, which read a varying instead of the attachment.
:func:`refuse_colour_attachment_inputs` refuses them with :class:`ColourAttachmentInputUnsupportedError`.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["ColourAttachmentInputUnsupportedError", "refuse_colour_attachment_inputs"]

_CARRYING = frozenset({"metal", "cgl", "crossgl", ""})
_COLOUR = re.compile(r"gl_FragColor\d*|gl_FragData(?:\[\d+\])?")
_OUTPUT_QUALIFIERS = frozenset({"out", "inout"})
_REASONS = {
    "directx": (
        "Direct3D pixel shaders cannot read a render target (there is no framebuffer fetch)"
    ),
    "vulkan": (
        "Vulkan reads earlier subpasses only through input attachments its render pass declares"
    ),
    "opengl": "desktop GLSL has no framebuffer fetch",
    "webgl": "WebGL has no framebuffer fetch",
}


class ColourAttachmentInputUnsupportedError(TranslationDiagnostic):
    """A fragment input that reads a colour attachment, which the target cannot read in the shader."""

    def __init__(self, target, function, name, semantic):
        family = _family(target)
        self.missing_capabilities = (f"{family}.colour-attachment-input",)
        reason = _REASONS.get(family, f"{target} has no colour attachment reads")
        super().__init__(
            f"fragment function '{function}' reads colour attachment {semantic} through input '{name}' "
            f"(MSL programmable blending), but {reason}",
            target=target,
            code=f"{CODE_PREFIX}{family}-colour-attachment-input-unsupported",
        )


def _family(target):
    if target in {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"}:
        return "directx"
    if target in {"vulkan", "spirv"}:
        return "vulkan"
    if target in {"webgl", "essl"}:
        return "webgl"
    if target in {"opengl", "glsl"}:
        return "opengl"
    return target


def _colour_semantic(node):
    for attribute in getattr(node, "attributes", None) or []:
        name = str(getattr(attribute, "name", ""))
        if _COLOUR.fullmatch(name):
            return name
    semantic = getattr(node, "semantic", None)
    if semantic and _COLOUR.fullmatch(str(semantic)):
        return str(semantic)
    return None


def _struct_members(program):
    structs = {}
    for node in getattr(program, "structs", None) or []:
        if isinstance(node, A.StructNode):
            structs[node.name] = node.members
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        for node in getattr(stage, "local_structs", None) or []:
            if isinstance(node, A.StructNode):
                structs[node.name] = node.members
    return structs


def _fragment_entries(program):
    stages = getattr(program, "stages", None) or {}
    for key, stage in stages.items() if isinstance(stages, dict) else ():
        if "fragment" not in str(getattr(key, "value", key)).lower():
            continue
        if getattr(stage, "entry_point", None) is not None:
            yield stage.entry_point
        for function in getattr(stage, "local_functions", None) or []:
            if any(
                str(getattr(attribute, "name", "")).lower() == "stage_entry"
                for attribute in getattr(function, "attributes", None) or []
            ):
                yield function


def refuse_colour_attachment_inputs(program, target) -> None:
    """Raise :class:`ColourAttachmentInputUnsupportedError` for a fragment input that reads a colour attachment."""
    target = str(target or "").lower()
    if target in _CARRYING or not isinstance(program, A.ShaderNode):
        return
    structs = _struct_members(program)
    for function in _fragment_entries(program):
        for parameter in getattr(function, "parameters", None) or []:
            qualifiers = {
                str(q).lower() for q in getattr(parameter, "qualifiers", None) or []
            }
            if qualifiers & _OUTPUT_QUALIFIERS:
                continue
            semantic = _colour_semantic(parameter)
            if semantic is not None:
                raise ColourAttachmentInputUnsupportedError(
                    target, function.name, parameter.name, semantic
                )
            type_name = getattr(getattr(parameter, "param_type", None), "name", None)
            for member in structs.get(type_name, ()):
                semantic = _colour_semantic(member)
                if semantic is not None:
                    raise ColourAttachmentInputUnsupportedError(
                        target,
                        function.name,
                        f"{parameter.name}.{member.name}",
                        semantic,
                    )
