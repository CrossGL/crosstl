"""Locations of the members of an anonymous GLSL ``in``/``out`` interface block.

GLSL 4.60, section 4.4.1: when a block has a location, its first member without one takes the block's location,
and every later member without one takes the next location after the locations the previous member consumes.
A member with its own location uses it, and counting continues from there.
"""

import re

from .OpenglAst import NumberNode

_STAGE_IO_QUALIFIERS = {"in", "out"}
_VECTOR = re.compile(r"^(?:[biud]?vec|f16vec|i16vec|u16vec|i8vec|u8vec)([234])$")
_MATRIX = re.compile(r"^(d?)mat([234])(?:x([234]))?$")
_SCALARS = {
    "bool",
    "int",
    "uint",
    "float",
    "double",
    "half",
    "float16_t",
    "int16_t",
    "uint16_t",
    "int8_t",
    "uint8_t",
    "int64_t",
    "uint64_t",
}


def _literal_int(value):
    if isinstance(value, NumberNode):
        value = value.value
    text = str(value).strip() if value is not None else ""
    if re.fullmatch(r"\d+[uU]?", text):
        return int(text.rstrip("uU"))
    return None


def _type_span(type_name):
    """Locations one value of ``type_name`` consumes, or None for a type this module cannot count."""
    text = str(type_name or "").strip()
    suffixes = re.findall(r"\[([^\]]*)\]", text)
    base = text.split("[", 1)[0].strip()
    count = 1
    for size in suffixes:
        size = _literal_int(size)
        if size is None:
            return None
        count *= size
    if base in _SCALARS:
        return count * (2 if base in {"double", "int64_t", "uint64_t"} else 1)
    vector = _VECTOR.match(base)
    if vector:
        wide = base.startswith(("dvec", "i64vec", "u64vec"))
        return count * (2 if wide and int(vector.group(1)) > 2 else 1)
    matrix = _MATRIX.match(base)
    if matrix:
        columns = int(matrix.group(2))
        rows = int(matrix.group(3) or matrix.group(2))
        return count * columns * (2 if matrix.group(1) and rows > 2 else 1)
    return None


def member_location_span(member):
    """Locations a block member declaration consumes, counting its declarator array sizes."""
    span = _type_span(getattr(member, "vtype", None))
    if span is None:
        return None
    sizes = list(getattr(member, "array_sizes", None) or [])
    if not sizes and getattr(member, "array_size", None) is not None:
        sizes = [member.array_size]
    for size in sizes:
        size = _literal_int(size)
        if size is None:
            return None
        span *= size
    return span


def anonymous_block_member_layouts(block_name, members, qualifiers, layout):
    """The layout of each member of an anonymous block, with consecutive locations for ``in``/``out`` blocks.

    Returns a list parallel to ``members``. Blocks that are not stage inputs or outputs, or that have no
    location, keep the block layout for every member.
    """
    lowered = {str(qualifier).lower() for qualifier in qualifiers or []}
    block_location = _literal_int((layout or {}).get("location"))
    if not lowered & _STAGE_IO_QUALIFIERS or block_location is None:
        return [layout for _ in members]

    layouts = []
    running = block_location
    unknown = None
    for member in members:
        own = dict(getattr(member, "interface_member_layout", None) or {})
        location = _literal_int(own.get("location"))
        if location is None:
            if unknown is not None:
                from .OpenglParser import UnsupportedGLSLFeatureError

                raise UnsupportedGLSLFeatureError(
                    "anonymous-interface-block-location",
                    f"GLSL block '{block_name}' member '{member.name}' takes its location after member "
                    f"'{unknown}', whose type '{_member_type(members, unknown)}' has no countable location span; "
                    "give the member an explicit layout(location)",
                )
            location = running
        merged = dict(layout or {})
        merged.update(own)
        merged["location"] = str(location)
        layouts.append(merged)
        span = member_location_span(member)
        if span is None:
            unknown = member.name
        else:
            unknown = None
            running = location + span
    return layouts


def _member_type(members, name):
    for member in members:
        if member.name == name:
            return getattr(member, "vtype", None)
    return None
