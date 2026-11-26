"""Import HLSL's tiled-resource Status operands and ``CheckAccessFullyMapped`` as CrossGL texture residency.

Shader Model 5 gives the texture, storage-texture and typed-buffer reads an overload whose last operand,
``out uint Status``, receives the residency of the texels the read touched, and ``CheckAccessFullyMapped(Status)``
tests it (DXC ``utils/hct/gen_intrin_main.txt``, the ``*_s`` overloads). CrossGL wraps an ordinary read instead
(:mod:`crosstl.translator.codegen.texture_residency`): ``textureResidency(read, texel)`` performs ``read``, stores
its texel in ``texel`` and returns an opaque residency code, which only ``textureResident(code)`` interprets. So::

    float4 c = t.SampleLevel(s, uv, 0.0f, int2(0, 0), status);   // HLSL

becomes a comma expression around a texel local declared before the statement::

    vec4 crosstl_status_texel_0;
    vec4 c = (status = uint(textureResidency(textureLodOffset(t, s, uv, 0.0, ivec2(0, 0)), crosstl_status_texel_0)),
              crosstl_status_texel_0);

The comma operator keeps the read, and the store of its Status, where HLSL evaluates them, even inside a ternary
arm, a short-circuit operand or a loop condition; only the declaration of the texel local moves. A read whose texel
HLSL narrows (``Texture2D<float>`` returns ``float``) narrows the local the same way. ``CheckAccessFullyMapped(x)``
becomes ``textureResident(int(x))``: Status holds the code CrossGL's read returned, so the HLSL generator writes it
back as the read's own Status and ``CheckAccessFullyMapped``, and every other target interprets its own code.

The byte-address and structured-buffer Status overloads stay refused: CrossGL's residency query wraps texture
reads only, and a buffer load has no residency form in GLSL, MSL or SPIR-V. A Status read outside a function body,
whose read imports as anything but one direct CrossGL texture read, is refused as well.
"""

from __future__ import annotations

import copy
import re

from ...translator.codegen.texture_lod_clamp import split_call_text
from ...translator.codegen.texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENCY_READS,
    TEXTURE_RESIDENT_OPERATION,
)
from .DirectxAst import ArrayAccessNode, FunctionCallNode, MemberAccessNode

__all__ = [
    "STATUS_TEXEL_PREFIX",
    "begin_status_texel_function",
    "check_access_fully_mapped_expression",
    "end_status_texel_function",
    "open_status_texel_scope",
    "status_operand_count",
    "status_read_expression",
    "take_status_texel_declarations",
]

STATUS_TEXEL_PREFIX = "crosstl_status_texel_"

_GATHER_CHANNELS = ("Red", "Green", "Blue", "Alpha")
# Operand count of each method's Status overload, Status included, on a texture with offsets and on a cube texture,
# which takes none (DXC gen_intrin_main.txt, Texture*Methods and TextureCUBE*Methods).
_SAMPLED_STATUS_COUNTS = {
    "Sample": (5, 4),
    "SampleBias": (6, 5),
    "SampleGrad": (7, 6),
    "SampleLevel": (5, 4),
    "SampleCmp": (6, 5),
    "SampleCmpBias": (7, 6),
    "SampleCmpGrad": (8, 7),
    "SampleCmpLevel": (6, 5),
    "SampleCmpLevelZero": (5, 4),
    "Gather": (4, 3),
    "GatherRaw": (4, None),
    "GatherCmp": (5, 4),
    **{f"Gather{channel}": (4, 3) for channel in _GATHER_CHANNELS},
    **{f"GatherCmp{channel}": (5, 4) for channel in _GATHER_CHANNELS},
}
# The per-channel gathers also take one offset per texel before Status.
_FOUR_OFFSET_STATUS_COUNTS = {
    **{f"Gather{channel}": 7 for channel in _GATHER_CHANNELS},
    **{f"GatherCmp{channel}": 8 for channel in _GATHER_CHANNELS},
}
_SAMPLED_TEXTURES = frozenset(
    {
        "Texture1D",
        "Texture1DArray",
        "Texture2D",
        "Texture2DArray",
        "Texture3D",
        "TextureCube",
        "TextureCubeArray",
    }
)
_CUBE_TEXTURES = frozenset({"TextureCube", "TextureCubeArray"})
# Load(location[, sample], offset, Status) on sampled textures; Load(location[, sample], Status) on storage textures
# and typed buffers, which take no offset.
_LOAD_STATUS_COUNTS = {
    "Texture1D": 3,
    "Texture1DArray": 3,
    "Texture2D": 3,
    "Texture2DArray": 3,
    "Texture3D": 3,
    "Texture2DMS": 4,
    "Texture2DMSArray": 4,
    "RWTexture1D": 2,
    "RWTexture1DArray": 2,
    "RWTexture2D": 2,
    "RWTexture2DArray": 2,
    "RWTexture3D": 2,
    "RWTexture2DMS": 3,
    "RWTexture2DMSArray": 3,
    "Buffer": 2,
    "RWBuffer": 2,
}
# Every resource spelling with Status overloads, mapped to its canonical name; any other resource has none.
_STATUS_RESOURCE_NAMES = {
    **{name: name for name in sorted(_SAMPLED_TEXTURES | _LOAD_STATUS_COUNTS.keys())},
    "TextureCUBE": "TextureCube",
    "TextureCUBEArray": "TextureCubeArray",
}
_SWIZZLE = re.compile(r"\.[xyzw]{1,4}$")
_SIGNED_KIND = re.compile(r"^(?P<kind>[ui])(?=texture|image|sampler)")


def _resource_base(codegen, resource_type):
    """Canonical name of a resource with Status overloads, or None for any other resource."""
    return _STATUS_RESOURCE_NAMES.get(codegen.raw_type_base(resource_type))


def status_operand_count(member, resource_base):
    """Operand counts of ``member``'s Status overloads on ``resource_base``, Status included; empty when it has none."""
    if member == "Load":
        count = _LOAD_STATUS_COUNTS.get(resource_base)
        return frozenset({count}) if count else frozenset()
    if resource_base not in _SAMPLED_TEXTURES or member not in _SAMPLED_STATUS_COUNTS:
        return frozenset()
    if resource_base == "Texture3D" and member.startswith("Gather"):
        return frozenset()
    flat, cube = _SAMPLED_STATUS_COUNTS[member]
    if resource_base in _CUBE_TEXTURES:
        return frozenset({cube}) if cube else frozenset()
    if resource_base.startswith("Texture1D") and member.startswith("Gather"):
        return frozenset()
    counts = {flat}
    if member in _FOUR_OFFSET_STATUS_COUNTS:
        counts.add(_FOUR_OFFSET_STATUS_COUNTS[member])
    return frozenset(counts)


def _is_lvalue(expr):
    if isinstance(expr, str):
        return True
    if isinstance(expr, (MemberAccessNode, ArrayAccessNode)):
        inner = expr.object if isinstance(expr, MemberAccessNode) else expr.array
        return _is_lvalue(inner)
    return False


def _texel_type(codegen, callee, resource_type):
    """The CrossGL type of ``callee``'s whole texel: ``float`` for a depth compare, a four-vector otherwise."""
    if callee.startswith("textureCompare"):
        return "float"
    mapped = str(codegen.map_type(resource_type) or "")
    kind = _SIGNED_KIND.match(mapped)
    return f"{kind.group('kind')}vec4" if kind else "vec4"


def _scope(codegen):
    return getattr(codegen, "_status_texel_scopes", None)


def status_read_expression(
    codegen, expr, resource_type, is_main=False, *, raw_texel=False
):
    """CrossGL for an HLSL read that writes a Status operand, or ``None`` when ``expr`` is no such read.

    ``raw_texel`` asks for the read's whole texel, as the caller asked of the read itself.
    """
    if not isinstance(expr, FunctionCallNode) or not isinstance(
        expr.name, MemberAccessNode
    ):
        return None
    member = codegen.templated_method_base(expr.name.member)
    resource_base = _resource_base(codegen, resource_type)
    arguments = list(expr.args or [])
    if len(arguments) not in status_operand_count(member, resource_base):
        return None
    status = arguments[-1]
    resource = resource_base or "resource"
    read = copy.copy(expr)
    read.args = arguments[:-1]
    codegen.read_raw_texel_method_result = raw_texel
    text = codegen.generate_expression(read, is_main)
    # A refused read keeps the read without its Status in its place, so generation reaches the other refusals.
    if not _is_lvalue(status):
        return codegen.refuse_import(
            f"DirectX tiled-resource status for {resource}.{member}: the Status operand is not an lvalue",
            text,
        )
    scopes = _scope(codegen)
    if not scopes:
        return codegen.refuse_import(
            f"DirectX tiled-resource status for {resource}.{member}: a read outside a function body has no "
            "statement to declare its texel before",
            text,
        )
    swizzle = _SWIZZLE.search(text)
    call_text = text[: swizzle.start()] if swizzle else text
    parts = split_call_text(call_text)
    callee = parts[0] if parts else ""
    if callee not in TEXTURE_RESIDENCY_READS:
        return codegen.refuse_import(
            f"DirectX tiled-resource status for {resource}.{member}: the read imports as '{text}', not as one "
            "CrossGL texture read that textureResidency can wrap",
            text,
        )
    texel = f"{STATUS_TEXEL_PREFIX}{codegen._status_texel_count}"
    codegen._status_texel_count += 1
    scopes[-1].append(f"{_texel_type(codegen, callee, resource_type)} {texel};")
    status_text = codegen.generate_expression(status, is_main)
    value = f"{texel}{swizzle.group(0)}" if swizzle else texel
    return (
        f"({status_text} = uint({TEXTURE_RESIDENCY_OPERATION}({call_text}, {texel})), "
        f"{value})"
    )


def check_access_fully_mapped_expression(codegen, args, is_main=False):
    """``CheckAccessFullyMapped(status)`` as ``textureResident(int(status))``."""
    args = list(args or [])
    if len(args) != 1:
        return codegen.refuse_import(
            f"DirectX CheckAccessFullyMapped takes one Status operand, got {len(args)}"
        )
    status = codegen.generate_expression(args[0], is_main)
    return f"{TEXTURE_RESIDENT_OPERATION}(int({status}))"


def begin_status_texel_function(codegen):
    """Number a function's texel locals from 0; returns the state :func:`end_status_texel_function` restores."""
    previous = (
        getattr(codegen, "_status_texel_scopes", None),
        getattr(codegen, "_status_texel_count", 0),
    )
    codegen._status_texel_scopes = []
    codegen._status_texel_count = 0
    return previous


def end_status_texel_function(codegen, previous):
    codegen._status_texel_scopes, codegen._status_texel_count = previous


def open_status_texel_scope(codegen):
    """Start collecting the texel declarations of one statement."""
    if getattr(codegen, "_status_texel_scopes", None) is None:
        codegen._status_texel_scopes = []
        codegen._status_texel_count = 0
    codegen._status_texel_scopes.append([])


def take_status_texel_declarations(codegen):
    """End the statement's scope and return the texel declarations its Status reads need."""
    return codegen._status_texel_scopes.pop()
