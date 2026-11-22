"""Rust field-less enums as CrossGL enums.

A Rust enum whose variants carry no data (``enum Mode { Off, Low = 2, High }``) is a set of named integer
constants, which CrossGL declares the same way: ``enum Mode : int { Off, Low = 2, High };``. Explicit
discriminants keep their values and the others count up from the previous one, as in both languages, and
``#[repr(T)]`` names the underlying integer type (``isize`` without one, which kernels treat as ``i32``).
Enums whose variants carry tuple or struct fields have no CrossGL counterpart and are not declared here.
"""

from __future__ import annotations

# The CrossGL spelling of each integer type ``#[repr(...)]`` can name.
_REPR_TYPES = {
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint",
    "u64": "uint64_t",
}


class RustEnumMixin:
    """Write the field-less enums of a Rust module as CrossGL enum declarations."""

    def generate_unit_enums(self, ast):
        code = ""
        for enum in getattr(ast, "enums", None) or []:
            code += self.generate_unit_enum(enum) or ""
        return code

    def generate_unit_enum(self, enum):
        variants = list(getattr(enum, "variants", None) or [])
        if not variants or getattr(enum, "generics", None):
            return None
        if any(getattr(variant, "kind", "unit") != "unit" for variant in variants):
            return None
        underlying = self.unit_enum_underlying_type(enum)
        lines = []
        for variant in variants:
            value = getattr(variant, "value", None)
            if value is None:
                lines.append(f"        {variant.name}")
            else:
                lines.append(
                    f"        {variant.name} = {self.generate_expression(value)}"
                )
        body = ",\n".join(lines)
        return f"    enum {enum.name} : {underlying} {{\n{body}\n    }};\n\n"

    @staticmethod
    def unit_enum_underlying_type(enum):
        for attribute in getattr(enum, "attributes", None) or []:
            if getattr(attribute, "name", None) != "repr":
                continue
            for argument in getattr(attribute, "args", None) or []:
                mapped = _REPR_TYPES.get(str(argument).strip())
                if mapped is not None:
                    return mapped
        return "int"
