"""Helpers for lowering CrossGL enums to C-style shader languages."""

from ..ast import EnumNode


def collect_plain_enums(nodes):
    """Return enums whose variants do not carry payload data."""
    enums = []
    for node in nodes or []:
        if not isinstance(node, EnumNode):
            continue
        if any(enum_variant_has_payload(variant) for variant in node.variants or []):
            continue
        enums.append(node)
    return enums


def enum_variant_has_payload(variant):
    return bool(getattr(variant, "data", None) or getattr(variant, "fields", None))


def collect_enum_type_names(enums):
    return {enum.name for enum in enums or []}


def collect_enum_variant_constants(enums):
    constants = {}
    for enum in enums or []:
        for variant in enum.variants or []:
            constants[f"{enum.name}::{variant.name}"] = enum_constant_name(
                enum.name, variant.name
            )
    return constants


def enum_constant_name(enum_name, variant_name):
    return f"{enum_name}_{variant_name}"


def generate_enum_constants(generator, enums, qualifier="static const"):
    code = ""
    for enum in enums or []:
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
            code += f"{qualifier} int {name} = {value};\n"
    if code:
        code += "\n"
    return code


def literal_int_value(expr):
    value = getattr(expr, "value", expr)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None
