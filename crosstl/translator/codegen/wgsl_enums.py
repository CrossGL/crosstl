"""Plain enumerations in WGSL output.

WGSL has no enumerations. A plain enumeration, one whose variants carry no data, is lowered to one ``i32``
constant per variant, ``const Mode_First: i32 = 0;``, as GLSL and Metal output write ``int`` constants, and the
enumeration's name stands for ``i32``. An enumeration whose variants carry data is refused with
:class:`WGSLEnumUnsupportedError`.
"""

from __future__ import annotations

from ..ast import EnumNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .enum_utils import enum_constant_name, enum_variant_has_payload, literal_int_value


class WGSLEnumUnsupportedError(TranslationDiagnostic):
    """An enumeration WGSL output cannot lower to integer constants."""

    def __init__(self, enum_name):
        self.enum_name = enum_name
        self.missing_capabilities = ("wgsl.enum-payload",)
        super().__init__(
            f"WGSL target cannot lower enumeration '{enum_name}': WGSL has no enumerations, and only "
            "enumerations whose variants carry no data are lowered to i32 constants",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-enum-unsupported",
        )


def split_enums(nodes):
    """``(structs, enums)``: ``nodes`` without its enumerations, and the plain enumerations it has.

    Raises :class:`WGSLEnumUnsupportedError` for a generic enumeration or one whose variants carry data.
    """
    structs, enums = [], []
    for node in nodes:
        if not isinstance(node, EnumNode):
            structs.append(node)
            continue
        if getattr(node, "generic_params", None) or any(
            enum_variant_has_payload(variant) for variant in node.variants or []
        ):
            raise WGSLEnumUnsupportedError(node.name)
        enums.append(node)
    return structs, enums


def enum_variant_constants(enums):
    """``{"Mode::First": "Mode_First", ...}`` for the variants of ``enums``."""
    return {
        f"{enum.name}::{variant.name}": enum_constant_name(enum.name, variant.name)
        for enum in enums
        for variant in enum.variants or []
    }


def generate_enum_constants(generator, enums):
    """``const Mode_First: i32 = 0;`` for each variant; a variant without a value follows the previous one."""
    lines = []
    for enum in enums:
        next_value = 0
        for variant in enum.variants or []:
            if getattr(variant, "value", None) is None:
                value = str(next_value)
                next_value += 1
            else:
                value = generator.generate_expression(variant.value)
                literal_value = literal_int_value(variant.value)
                next_value = literal_value + 1 if literal_value is not None else 0
            name = enum_constant_name(enum.name, variant.name)
            lines.append(f"const {name}: i32 = {value};")
    return "\n".join(lines)
