"""HLSL for CrossGL's texture residency queries: the ``out uint Status`` overloads and ``CheckAccessFullyMapped``.

Shader Model 5 gives every texture read a tiled-resource overload whose last operand, ``out uint Status``, receives
the residency of the texels the read touched; ``CheckAccessFullyMapped(Status)`` tests it. Status follows the
optional operands, so a read that omits them gets their neutral values first: a zero ``Offset`` (cube textures take
none) and a ``Clamp`` of ``0.0f``, which excludes no mip level. A CrossGL query::

    int code = textureResidency(texture(t, s, uv), texel);

becomes, with a status local the semantic pass declares at the top of the function
(:func:`crosstl.translator.semantics.texture_residency.declare_hlsl_residency_status`)::

    int code = ((texel = t.Sample(s_sampler, uv, int2(0, 0), 0.0f, status), CheckAccessFullyMapped(status)) ? 0 : 1);

The comma operator sequences the read, and so the copy-out of ``status``, before ``CheckAccessFullyMapped`` reads
it. ``textureResident(code)`` is ``(code == 0)``, as in MSL. A read whose HLSL form has no Status overload is refused
with :class:`~crosstl.translator.codegen.texture_residency.TextureResidencyUnsupportedError`.
"""

from __future__ import annotations

import re

from .texture_lod_clamp import join_call_text, split_call_text
from .texture_residency import (
    _SIMPLE_OPERAND,
    HLSL_RESIDENCY_STATUS_ATTRIBUTE,
    _call_name,
    _refuse,
    _render_read,
    _resident_code,
    split_texture_residency_call,
)

__all__ = ["hlsl_texture_residency_call", "hlsl_texture_resident_call"]

_GATHER_CHANNELS = ("Red", "Green", "Blue", "Alpha")
# HLSL texture methods with a Status overload: the operands before the optional ones, and the optional operands
# Status follows, in order (HLSL texture object methods, Shader Model 5 tiled-resource overloads).
_STATUS_FORMS = {
    "Sample": (2, ("offset", "clamp")),
    "SampleBias": (3, ("offset", "clamp")),
    "SampleGrad": (4, ("offset", "clamp")),
    "SampleCmp": (3, ("offset", "clamp")),
    "SampleCmpBias": (4, ("offset", "clamp")),
    "SampleCmpGrad": (5, ("offset", "clamp")),
    "SampleLevel": (3, ("offset",)),
    "SampleCmpLevel": (4, ("offset",)),
    "SampleCmpLevelZero": (3, ("offset",)),
    "Load": (1, ("offset",)),
    "Gather": (2, ("offset",)),
    "GatherCmp": (3, ("offset",)),
    **{f"Gather{channel}": (2, ("offset",)) for channel in _GATHER_CHANNELS},
    **{f"GatherCmp{channel}": (3, ("offset",)) for channel in _GATHER_CHANNELS},
}
# Per-channel gathers also take four offsets, one per texel, before Status.
_FOUR_OFFSET_GATHERS = frozenset(
    name
    for name in _STATUS_FORMS
    if name.startswith("Gather") and name not in ("Gather", "GatherCmp")
)
_CUBE_SHAPES = frozenset({"TextureCube", "TextureCubeArray"})
_MULTISAMPLE_SHAPES = frozenset({"Texture2DMS", "Texture2DMSArray"})
_NEUTRAL_CLAMP = "0.0f"
_INDEXED_LOAD = re.compile(r"^(?P<image>[A-Za-z_][A-Za-z0-9_.]*)\[(?P<index>.+)\]$")


def _read_call_parts(base_text, read_name):
    """``(callee, arguments)`` of the HLSL read; an image load written ``image[coord]`` becomes ``image.Load``."""
    parts = split_call_text(base_text)
    if parts is not None:
        return parts
    if read_name == "imageLoad":
        indexed = _INDEXED_LOAD.match(base_text.strip())
        if indexed is not None and split_call_text(f"f({indexed.group('index')})"):
            return f"{indexed.group('image')}.Load", [indexed.group("index")]
    return None


def _shape(codegen, read):
    arguments = list(getattr(read, "arguments", None) or [])
    texture_type = codegen.texture_resource_type(arguments[0]) if arguments else None
    return texture_type, codegen.sampled_texture_shape_type(texture_type) or ""


def _zero_offset(codegen, method, texture_type, shape):
    if shape in _MULTISAMPLE_SHAPES:
        return "int2(0, 0)"
    probe = "texelFetchOffset" if method == "Load" else "textureOffset"
    if "Cmp" in method:
        probe = "textureCompareOffset"
    dimension = codegen.resource_offset_dimension(probe, texture_type)
    return (
        codegen.hlsl_render_texture_offset_value((0,) * dimension)
        if dimension
        else None
    )


def _status_arguments(codegen, read, method, arguments, status):
    """The read's operands with the neutral optional operands and Status appended, or a refusal reason."""
    mandatory, optional = _STATUS_FORMS[method]
    texture_type, shape = _shape(codegen, read)
    if method == "Load":
        if shape in _MULTISAMPLE_SHAPES:
            mandatory = 2
        elif not shape.startswith("Texture"):
            # Buffer<T>.Load and RWTexture*.Load take only the location before Status.
            optional = ()
    if shape in _CUBE_SHAPES:
        optional = tuple(operand for operand in optional if operand != "offset")
    supplied = len(arguments) - mandatory
    if method in _FOUR_OFFSET_GATHERS and supplied == 4 and shape not in _CUBE_SHAPES:
        return [*arguments, status]
    if supplied < 0 or supplied > len(optional):
        return f"its operands do not match any {method} overload with a Status operand"
    completed = list(arguments)
    for operand in optional[supplied:]:
        if operand == "clamp":
            completed.append(_NEUTRAL_CLAMP)
            continue
        offset = _zero_offset(codegen, method, texture_type, shape)
        if offset is None:
            return f"HLSL takes Status after an offset, and {shape or 'the resource'} has no offset dimension"
        completed.append(offset)
    completed.append(status)
    return completed


def hlsl_texture_residency_call(codegen, args, call_node=None):
    """HLSL for ``textureResidency``: the read's Status overload, tested with ``CheckAccessFullyMapped``."""
    read, texel = split_texture_residency_call("directx", args, call_node)
    status = getattr(call_node, HLSL_RESIDENCY_STATUS_ATTRIBUTE, None)
    base_text = _render_read(codegen, read)
    if not status:
        return _refuse(
            codegen,
            "directx",
            base_text,
            "no status local was declared for the read (the query is outside a function body)",
            call_node,
        )
    parts = _read_call_parts(base_text, _call_name(read))
    callee = parts[0] if parts else ""
    receiver, _, method = callee.rpartition(".")
    if not receiver or method not in _STATUS_FORMS:
        return _refuse(
            codegen,
            "directx",
            base_text,
            f"the read lowered to '{base_text}', not to a texture method with a Status overload "
            f"({', '.join(sorted(_STATUS_FORMS))})",
            call_node,
        )
    arguments = _status_arguments(codegen, read, method, list(parts[1]), status)
    if isinstance(arguments, str):
        return _refuse(
            codegen,
            "directx",
            base_text,
            f"the read lowered to '{base_text}': {arguments}",
            call_node,
        )
    texel_text = codegen.generate_expression(texel)
    read_text = join_call_text(callee, arguments)
    # The comma sits in the condition of ?:, where DXC does not mistake it for a constructor list (-Wcomma-in-init).
    return f"(({texel_text} = {read_text}, CheckAccessFullyMapped({status})) ? 0 : 1)"


def hlsl_texture_resident_call(codegen, args, call_node=None):
    """HLSL for ``textureResident``: the code is 0 when every texel was resident."""
    code = codegen.generate_expression(_resident_code("directx", args, call_node))
    if not _SIMPLE_OPERAND.fullmatch(code):
        code = f"({code})"
    return f"({code} == 0)"
