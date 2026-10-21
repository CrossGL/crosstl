"""Layout qualifiers that a constant-buffer member carries itself.

A member of an HLSL constant buffer or a GLSL uniform block may fix its own
byte offset and its matrix orientation, and the host writes the member at the
bytes those qualifiers name:

* HLSL ``packoffset(c<N>.<component>)`` places a member at byte
  ``16 * N + 4 * component`` of the constant buffer; GLSL
  ``layout(offset = N)`` places a block member at byte ``N`` (GLSL 4.60
  section 4.4.5). CrossGL keeps them as ``@ packoffset(c1.y)`` and
  ``@ offset(16)``.
* ``row_major`` (HLSL, and GLSL ``layout(row_major)``) stores a matrix row by
  row; ``column_major`` is the default. CrossGL keeps them as ``@ row_major``
  and ``@ column_major``.

Backends that reproduce the layout of a source buffer read these qualifiers
from here, so that every target places a member at the same bytes.
"""

import re

from ..ast import IdentifierNode, LiteralNode, MemberAccessNode
from .hlsl_buffer_layout import CBUFFER_REGISTER_SIZE, HlslLayoutError

PACKOFFSET_COMPONENTS = {"x": 0, "y": 1, "z": 2, "w": 3, "r": 0, "g": 1, "b": 2, "a": 3}
PACKOFFSET_PATTERN = re.compile(
    r"\s*c(?P<register>\d+)\s*(?:\.\s*(?P<component>[xyzwrgba]))?\s*"
)


def member_attribute_names(member):
    """Lower-case names of the attributes and qualifiers of a member."""
    names = {
        str(qualifier).lower()
        for qualifier in getattr(member, "qualifiers", None) or []
    }
    names.update(
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(member, "attributes", None) or []
    )
    return names


def member_is_row_major(member):
    """Whether a member is declared ``row_major`` (and not also ``column_major``)."""
    names = member_attribute_names(member)
    return "row_major" in names and "column_major" not in names


def member_explicit_offset(member):
    """The byte offset a member's ``packoffset`` or ``offset`` qualifier names, or ``None``."""
    for attribute in getattr(member, "attributes", None) or []:
        name = str(getattr(attribute, "name", "")).lower()
        if name not in ("packoffset", "offset"):
            continue
        arguments = list(
            getattr(attribute, "arguments", None)
            or getattr(attribute, "args", None)
            or []
        )
        if len(arguments) != 1:
            raise HlslLayoutError(
                f"member '{getattr(member, 'name', '')}' has a malformed {name} qualifier"
            )
        if name == "packoffset":
            return packoffset_bytes(arguments[0], getattr(member, "name", ""))
        return offset_bytes(arguments[0], getattr(member, "name", ""))
    return None


def packoffset_bytes(argument, member_name=""):
    """Byte offset of an HLSL ``packoffset`` argument such as ``c1.y`` (an expression or its text)."""
    if isinstance(argument, MemberAccessNode):
        register = getattr(argument, "object", None)
        text = (
            f"{getattr(register, 'name', register)}.{getattr(argument, 'member', '')}"
        )
    elif isinstance(argument, IdentifierNode):
        text = argument.name
    else:
        text = str(getattr(argument, "value", argument)).strip().strip('"')
    match = PACKOFFSET_PATTERN.fullmatch(text)
    if match is None:
        raise HlslLayoutError(
            f"member '{member_name}' has an unsupported packoffset '{text}'"
        )
    component = PACKOFFSET_COMPONENTS[match.group("component") or "x"]
    return CBUFFER_REGISTER_SIZE * int(match.group("register")) + 4 * component


def offset_bytes(argument, member_name=""):
    """Byte offset of a GLSL ``offset`` qualifier argument."""
    value = (
        getattr(argument, "value", argument)
        if isinstance(argument, LiteralNode)
        else argument
    )
    value = getattr(value, "name", value)
    text = str(value).strip()
    try:
        offset = int(text, 0)
    except ValueError:
        raise HlslLayoutError(
            f"member '{member_name}' has a non-constant offset '{text}'"
        ) from None
    if offset < 0:
        raise HlslLayoutError(f"member '{member_name}' has a negative offset {offset}")
    return offset


def packoffset_text(offset):
    """The HLSL ``packoffset`` argument for a 4-byte-aligned byte offset."""
    if offset % 4:
        raise HlslLayoutError(
            f"byte offset {offset} cannot be expressed with packoffset"
        )
    register, within = divmod(offset, CBUFFER_REGISTER_SIZE)
    if within == 0:
        return f"c{register}"
    return f"c{register}.{'xyzw'[within // 4]}"
