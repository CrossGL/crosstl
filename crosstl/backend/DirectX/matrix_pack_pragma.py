"""``#pragma pack_matrix`` as a member orientation.

``#pragma pack_matrix(row_major)`` and ``#pragma pack_matrix(column_major)``
set how the matrices declared after them are stored in constant buffers and
structs until the next such pragma. A ``row_major`` or ``column_major``
qualifier on a declaration wins over it. The pragma therefore says the same
as an explicit qualifier on every matrix declared after it, and that is how
the HLSL importer records it: CrossGL keeps the orientation on each member
(``@ row_major``), where every target reads it, and drops the pragma.
"""

import re

from .DirectxAst import PragmaNode

MATRIX_ORIENTATIONS = ("row_major", "column_major")
PACK_MATRIX_PATTERN = re.compile(r"\(\s*(row_major|column_major)\s*\)")
HLSL_MATRIX_TYPE_PATTERN = re.compile(
    r"\s*(?:(?:const|uniform|static|precise|volatile|snorm|unorm)\s+)*"
    r"(?:(?:min16|min10|min12)?(?:float|half|double|int|uint|bool|dword|float16_t|float32_t|float64_t"
    r"|int16_t|int32_t|int64_t|uint16_t|uint32_t|uint64_t)[1-4]x[1-4]|matrix\s*(?:<[^>]*>)?)\s*(?:\[.*\])?\s*"
)


def pack_matrix_orientation(directive):
    """The orientation a ``#pragma pack_matrix(...)`` sets, or ``None`` for any other directive.

    A ``pack_matrix`` pragma with an argument other than ``row_major`` or ``column_major`` raises
    :class:`SyntaxError`, as DXC rejects it.
    """
    if not isinstance(directive, PragmaNode):
        return None
    # ``#pragma pack_matrix(row_major)`` has no space before its argument.
    text = f"{directive.directive or ''} {directive.value or ''}".strip()
    if not re.match(r"pack_matrix\b", text):
        return None
    match = PACK_MATRIX_PATTERN.fullmatch(text[len("pack_matrix") :].strip())
    if match is None:
        raise SyntaxError(f"unsupported #pragma {text}")
    return match.group(1)


def is_hlsl_matrix_type(type_name):
    return HLSL_MATRIX_TYPE_PATTERN.fullmatch(str(type_name or "")) is not None


def apply_matrix_orientation(declaration, orientation):
    """Give ``declaration`` (a variable, a struct or a constant buffer) the pragma's ``orientation``.

    Every matrix member, and a matrix variable itself, that has no orientation qualifier of its own gets one.
    """
    if orientation is None:
        return
    members = getattr(declaration, "members", None)
    targets = list(members) if members is not None else [declaration]
    for target in targets:
        type_name = getattr(target, "vtype", getattr(target, "var_type", None))
        if not is_hlsl_matrix_type(type_name):
            continue
        qualifiers = getattr(target, "qualifiers", None)
        if qualifiers is None:
            qualifiers = []
            target.qualifiers = qualifiers
        if not any(
            str(qualifier).lower() in MATRIX_ORIENTATIONS for qualifier in qualifiers
        ):
            qualifiers.append(orientation)
