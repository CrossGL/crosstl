"""Shared classification for ordered boolean intrinsics."""

import re


def ordered_boolean_minmax_width(operation, argument_types):
    """Return the common boolean vector width for a builtin min/max call."""
    if operation not in {"min", "max"} or len(argument_types) != 2:
        return None

    widths = [boolean_type_width(type_name) for type_name in argument_types]
    if None in widths or widths[0] != widths[1]:
        return None
    return widths[0]


def boolean_type_width(type_name):
    normalized = re.sub(r"\s+", "", str(type_name or ""))
    if normalized == "bool":
        return 1
    match = re.fullmatch(r"(?:bool|bvec)([234])", normalized)
    if match is not None:
        return int(match.group(1))
    match = re.fullmatch(r"vec([234])<bool>", normalized)
    if match is not None:
        return int(match.group(1))
    return None


def is_boolean_type(type_name):
    normalized = re.sub(r"\s+", "", str(type_name or ""))
    return boolean_type_width(normalized) is not None or bool(
        re.fullmatch(r"(?:bool|bvec)[0-9]+", normalized)
        or re.fullmatch(r"vec[0-9]+<bool>", normalized)
    )
