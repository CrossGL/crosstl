"""Helpers for lowering CrossGL enums to C-style shader languages."""

from ..ast import EnumNode
from .array_utils import format_c_style_array_declaration


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


def collect_struct_payload_enums(nodes):
    """Return non-generic payload enums lowerable to tagged structs."""
    enums = []
    for node in nodes or []:
        if not isinstance(node, EnumNode):
            continue
        if getattr(node, "generic_params", None):
            continue
        if not any(enum_variant_has_payload(variant) for variant in node.variants):
            continue
        if enum_struct_fields(node) is None:
            continue
        enums.append(node)
    return enums


def enum_variant_has_payload(variant):
    return bool(getattr(variant, "data", None) or getattr(variant, "fields", None))


def collect_enum_type_names(enums):
    return {enum.name for enum in enums or []}


def collect_enum_struct_variant_fields(enums):
    fields_by_enum = {}
    for enum in enums or []:
        variants = {}
        for variant in enum.variants or []:
            variants[variant.name] = dict(enum_variant_payload_fields(variant) or [])
        fields_by_enum[enum.name] = variants
    return fields_by_enum


def collect_enum_variant_constructor_fields(enums):
    fields_by_path = {}
    for enum in enums or []:
        for variant in enum.variants or []:
            fields_by_path[f"{enum.name}::{variant.name}"] = (
                enum_variant_payload_fields(variant) or []
            )
    return fields_by_path


def collect_enum_variant_constructors(enums):
    constructors = {}
    for enum in enums or []:
        for variant in enum.variants or []:
            constructors[f"{enum.name}::{variant.name}"] = (
                enum_variant_constructor_name(enum.name, variant.name)
            )
    return constructors


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


def enum_variant_constructor_name(enum_name, variant_name):
    return f"{enum_name}_{variant_name}_make"


def enum_value_expression(generator, path):
    constructor = getattr(generator, "enum_variant_constructors", {}).get(path)
    fields = getattr(generator, "enum_variant_constructor_fields", {}).get(path)
    expected_type = getattr(generator, "current_expression_expected_type", None)
    enum_name = str(path).split("::", 1)[0]
    if constructor and fields == [] and expected_type == enum_name:
        return f"{constructor}()"
    return getattr(generator, "enum_variant_constants", {}).get(path, path)


def generate_enum_constructor_call(generator, path, args):
    constructor = getattr(generator, "enum_variant_constructors", {}).get(path)
    if constructor is None:
        return None

    fields = getattr(generator, "enum_variant_constructor_fields", {}).get(path, [])
    if len(args) != len(fields):
        raise ValueError(
            f"Enum constructor {path} expects {len(fields)} arguments, got {len(args)}"
        )

    rendered_args = []
    for index, arg in enumerate(args):
        expected_type = fields[index][1] if index < len(fields) else None
        rendered_args.append(
            generator.generate_expression_with_expected(arg, expected_type)
        )
    return f"{constructor}({', '.join(rendered_args)})"


def generate_enum_structs(generator, enums):
    code = ""
    for enum in enums or []:
        fields = enum_struct_fields(enum) or []
        code += f"struct {enum.name} {{\n"
        code += "    int variant;\n"
        for field_name, field_type in fields:
            declaration = format_c_style_array_declaration(
                generator.map_type(field_type), field_name
            )
            code += f"    {declaration};\n"
        code += "};\n\n"
    return code


def generate_enum_constructor_functions(generator, enums):
    code = ""
    for enum in enums or []:
        all_fields = enum_struct_fields(enum) or []
        for variant in enum.variants or []:
            variant_fields = enum_variant_payload_fields(variant) or []
            params = ", ".join(
                f"{generator.map_type(field_type)} payload{index}"
                for index, (_field_name, field_type) in enumerate(variant_fields)
            )
            code += (
                f"{enum.name} "
                f"{enum_variant_constructor_name(enum.name, variant.name)}({params}) {{\n"
            )
            code += f"    {enum.name} result;\n"
            code += (
                f"    result.variant = {enum_constant_name(enum.name, variant.name)};\n"
            )
            active_fields = {field_name for field_name, _field_type in variant_fields}
            for field_name, field_type in all_fields:
                if field_name in active_fields:
                    continue
                code += (
                    f"    result.{field_name} = "
                    f"{default_value_expression(generator, field_type)};\n"
                )
            for index, (field_name, _field_type) in enumerate(variant_fields):
                code += f"    result.{field_name} = payload{index};\n"
            code += "    return result;\n"
            code += "}\n\n"
    return code


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


def enum_struct_fields(enum):
    fields = []
    fields_by_name = {}
    for variant in enum.variants or []:
        variant_fields = enum_variant_payload_fields(variant)
        if variant_fields is None:
            return None
        for field_name, field_type in variant_fields:
            previous_type = fields_by_name.get(field_name)
            if previous_type is not None:
                if str(previous_type) != str(field_type):
                    return None
                continue
            fields_by_name[field_name] = field_type
            fields.append((field_name, field_type))
    return fields


def enum_variant_payload_fields(variant):
    if getattr(variant, "fields", None):
        return enum_tuple_payload_fields(variant.name, variant.fields)

    data = getattr(variant, "data", None)
    if not data:
        return []

    if all(isinstance(item, tuple) and len(item) == 2 for item in data):
        fields = []
        for item in data:
            field_name, field_type = item
            if not isinstance(field_name, str):
                return None
            fields.append((field_name, field_type))
        return fields

    if any(isinstance(item, tuple) for item in data):
        return None

    return enum_tuple_payload_fields(variant.name, data)


def enum_tuple_payload_fields(variant_name, fields):
    return [
        (f"{variant_name}_{index}", field_type)
        for index, field_type in enumerate(fields or [])
    ]


def default_value_expression(generator, field_type):
    mapped_type = generator.map_type(field_type)
    if mapped_type == "bool":
        return "false"
    if mapped_type.startswith("bool") or mapped_type.startswith("bvec"):
        return f"{mapped_type}(false)"
    return f"{mapped_type}(0)"


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
