"""Plain enumerations in WGSL output.

WGSL has no enumerations. A plain enumeration, one whose variants carry no data, is lowered to one ``i32``
constant per variant, ``const Mode_First: i32 = 0;``, as GLSL and Metal output write ``int`` constants, and the
enumeration's name stands for ``i32``. An enumeration whose variants carry data is refused with
:class:`WGSLEnumUnsupportedError`.
"""

from __future__ import annotations

from ..ast import EnumNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .enum_utils import enum_constant_name, enum_variant_has_payload


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


def generate_enum_constants(generator, enums, values):
    """``const Mode_First: i32 = 0;`` for each variant of ``enums``.

    A variant without a value is one more than the previous one, as in C. ``values`` holds the integer of each
    variant whose value CrossTL can compute, ``{"Mode::First": 0, ...}``
    (:func:`~crosstl.translator.semantics.enum_values.enum_values`); a variant after a value it cannot compute, such
    as a shader constant, is written as the previous constant plus one.
    """
    lines = []
    for enum in enums:
        previous = None
        for variant in enum.variants or []:
            name = enum_constant_name(enum.name, variant.name)
            path = f"{enum.name}::{variant.name}"
            if getattr(variant, "value", None) is not None:
                value = generator.generate_expression(variant.value)
            elif path in values:
                value = str(values[path])
            else:
                value = "0" if previous is None else f"{previous} + 1"
            lines.append(f"const {name}: i32 = {value};")
            previous = name
    return "\n".join(lines)
