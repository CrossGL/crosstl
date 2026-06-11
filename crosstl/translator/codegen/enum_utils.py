"""Helpers for lowering CrossGL enums to C-style shader languages."""

import re

from ..ast import EnumNode, StructNode
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


def collect_generic_enum_struct_definitions(nodes):
    """Return generic struct wrappers that behave like enum payload containers."""
    definitions = {}
    for node in nodes or []:
        definition = generic_enum_struct_definition(node)
        if definition is not None:
            definitions[definition["name"]] = definition
    return definitions


def generic_enum_struct_definition(node):
    if not isinstance(node, StructNode):
        return None

    generic_params = [param.name for param in getattr(node, "generic_params", [])]
    if not generic_params:
        return None

    enum_members = [
        member
        for member in getattr(node, "members", [])
        if isinstance(member, EnumNode)
    ]
    if len(enum_members) != 1:
        return None

    enum = enum_members[0]
    if not any(enum_variant_has_payload(variant) for variant in enum.variants or []):
        return None
    if enum_struct_fields(enum) is None:
        return None

    has_variant_member = any(
        getattr(member, "name", None) == "variant"
        for member in getattr(node, "members", [])
    )
    if not has_variant_member:
        return None

    return {
        "name": node.name,
        "generic_params": generic_params,
        "enum": enum,
    }


def collect_generic_enum_specializations(nodes, definitions, type_name_string):
    """Collect concrete generic enum wrapper instantiations used by the AST."""
    if not definitions:
        return {}

    generic_names = collect_generic_parameter_names(nodes)
    specializations = {}
    visited = set()

    def add_type_text(type_text):
        base_name, generic_args = generic_type_parts(str(type_text or ""))
        definition = definitions.get(base_name)
        if definition is None:
            return None
        if len(generic_args) != len(definition["generic_params"]):
            return None
        if any(
            type_text_contains_generic_parameter(arg, generic_names)
            for arg in generic_args
        ):
            return None
        if type_text in specializations:
            return None

        specialization = build_generic_enum_specialization(type_text, definition)
        specializations[type_text] = specialization
        return specialization

    def visit(value):
        if value is None:
            return
        if isinstance(value, str):
            add_type_text(value)
            return
        if isinstance(value, (int, float, bool)):
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if not hasattr(value, "__dict__"):
            return

        name = getattr(value, "name", None)
        generic_args = getattr(value, "generic_args", None)
        if name in definitions and generic_args:
            add_type_text(type_name_string(value))

        for child in vars(value).values():
            visit(child)

    visit(nodes)
    return dict(
        sorted(
            specializations.items(),
            key=lambda item: (
                generic_type_depth(item[0]),
                item[1]["struct_name"],
            ),
        )
    )


def collect_generic_parameter_names(value):
    names = {"Self"}
    visited = set()

    def visit(current):
        if current is None:
            return
        if isinstance(current, (str, int, float, bool)):
            return
        if isinstance(current, (list, tuple, set)):
            for item in current:
                visit(item)
            return
        if isinstance(current, dict):
            for item in current.values():
                visit(item)
            return

        current_id = id(current)
        if current_id in visited:
            return
        visited.add(current_id)
        if not hasattr(current, "__dict__"):
            return

        for param in getattr(current, "generic_params", []) or []:
            name = getattr(param, "name", None)
            if name:
                names.add(name)

        for child in vars(current).values():
            visit(child)

    visit(value)
    return names


def type_node_contains_generic_parameter(value, generic_names):
    if value is None:
        return False
    if isinstance(value, str):
        return type_text_contains_generic_parameter(value, generic_names)
    name = getattr(value, "name", None)
    generic_args = getattr(value, "generic_args", None)
    if name in generic_names and not generic_args:
        return True
    for arg in generic_args or []:
        if type_node_contains_generic_parameter(arg, generic_names):
            return True
    element_type = getattr(value, "element_type", None)
    return type_node_contains_generic_parameter(element_type, generic_names)


def type_text_contains_generic_parameter(type_text, generic_names):
    type_text = str(type_text or "").strip()
    if not type_text:
        return False
    base_name, generic_args = generic_type_parts(type_text)
    if not generic_args:
        return base_name in generic_names
    return any(
        type_text_contains_generic_parameter(arg, generic_names) for arg in generic_args
    )


def build_generic_enum_specialization(type_text, definition):
    base_name, generic_args = generic_type_parts(type_text)
    substitutions = dict(zip(definition["generic_params"], generic_args))
    return {
        "type_name": type_text,
        "base_name": base_name,
        "struct_name": generic_enum_specialization_name(type_text),
        "definition": definition,
        "substitutions": substitutions,
    }


def collect_generic_enum_variant_constants(definitions):
    constants = {}
    for name, definition in definitions.items():
        for variant in definition["enum"].variants or []:
            constants[f"{name}::{variant.name}"] = enum_constant_name(
                name, variant.name
            )
    return constants


def generic_enum_specialized_type_name(generator, vtype):
    specialization = resolve_generic_enum_specialization(generator, vtype)
    if specialization is None:
        return None
    return specialization["struct_name"]


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

    generic_value = generic_enum_value_expression(generator, path)
    if generic_value is not None:
        return generic_value

    return getattr(generator, "enum_variant_constants", {}).get(path, path)


def generate_enum_constructor_call(generator, path, args):
    constructor = getattr(generator, "enum_variant_constructors", {}).get(path)
    if constructor is None:
        return generate_generic_enum_constructor_call(generator, path, args)

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


def generate_enum_constructor_expression(generator, expr):
    constructor_type = getattr(expr, "constructor_type", None)
    path = generator_type_name_string(generator, constructor_type)
    if "::" not in str(path):
        return None

    constructor = getattr(generator, "enum_variant_constructors", {}).get(path)
    fields = getattr(generator, "enum_variant_constructor_fields", {}).get(path)
    if constructor is not None and fields is not None:
        rendered_args = render_enum_constructor_arguments(
            generator,
            path,
            expr,
            fields,
        )
        return f"{constructor}({', '.join(rendered_args)})"

    return generate_generic_enum_constructor_expression(generator, path, expr)


def generate_generic_enum_constructor_expression(generator, path, expr):
    enum_name, variant_name = str(path).split("::", 1)
    expected_type = getattr(generator, "current_expression_expected_type", None)
    specialization = resolve_generic_enum_specialization(
        generator,
        expected_type,
        expected_base=enum_name,
    )
    if specialization is None:
        return None

    fields = generic_enum_specialized_variant_fields(
        generator,
        specialization,
        variant_name,
    )
    if fields is None:
        return None

    rendered_args = render_enum_constructor_arguments(
        generator,
        path,
        expr,
        fields,
    )
    constructor = enum_variant_constructor_name(
        specialization["struct_name"],
        variant_name,
    )
    return f"{constructor}({', '.join(rendered_args)})"


def infer_enum_constructor_type(generator, expr):
    constructor_type = getattr(expr, "constructor_type", None)
    path = generator_type_name_string(generator, constructor_type)
    if "::" not in str(path):
        return None

    enum_name, variant_name = str(path).split("::", 1)
    if path in getattr(generator, "enum_variant_constructor_fields", {}):
        return enum_name

    expected_type = getattr(generator, "current_expression_expected_type", None)
    specialization = resolve_generic_enum_specialization(
        generator,
        expected_type,
        expected_base=enum_name,
    )
    if specialization is None:
        return None
    fields = generic_enum_specialized_variant_fields(
        generator,
        specialization,
        variant_name,
    )
    if fields is None:
        return None
    return specialization["type_name"]


def render_enum_constructor_arguments(generator, path, expr, fields):
    positional_args = list(getattr(expr, "arguments", []) or [])
    named_args = dict(getattr(expr, "named_arguments", {}) or {})
    field_names = [field_name for field_name, _field_type in fields]

    if len(positional_args) > len(fields):
        raise ValueError(
            f"Enum constructor {path} expects at most {len(fields)} arguments, "
            f"got {len(positional_args)}"
        )
    unknown_names = sorted(set(named_args) - set(field_names))
    if unknown_names:
        raise ValueError(
            f"Enum constructor {path} has no field {', '.join(unknown_names)}"
        )

    rendered_args = []
    missing_names = []
    for index, (field_name, field_type) in enumerate(fields):
        if index < len(positional_args):
            rendered_args.append(
                generator.generate_expression_with_expected(
                    positional_args[index],
                    field_type,
                )
            )
            continue
        if field_name in named_args:
            rendered_args.append(
                generator.generate_expression_with_expected(
                    named_args[field_name],
                    field_type,
                )
            )
            continue
        missing_names.append(field_name)

    if missing_names:
        raise ValueError(
            f"Enum constructor {path} is missing field {', '.join(missing_names)}"
        )
    return rendered_args


def generate_generic_enum_constructor_call(generator, path, args):
    if "::" not in str(path):
        return None

    enum_name, variant_name = str(path).split("::", 1)
    expected_type = getattr(generator, "current_expression_expected_type", None)
    specialization = resolve_generic_enum_specialization(
        generator,
        expected_type,
        expected_base=enum_name,
    )
    if specialization is None:
        return None

    fields = generic_enum_specialized_variant_fields(
        generator,
        specialization,
        variant_name,
    )
    if fields is None:
        return None
    if len(args) != len(fields):
        raise ValueError(
            f"Enum constructor {path} expects {len(fields)} arguments, got {len(args)}"
        )

    rendered_args = []
    for index, arg in enumerate(args):
        rendered_args.append(
            generator.generate_expression_with_expected(arg, fields[index][1])
        )
    constructor = enum_variant_constructor_name(
        specialization["struct_name"],
        variant_name,
    )
    return f"{constructor}({', '.join(rendered_args)})"


def enum_variant_fields_for_type(generator, enum_name, variant_name, subject_type):
    specialization = resolve_generic_enum_specialization(
        generator,
        subject_type,
        expected_base=enum_name,
    )
    if specialization is not None:
        fields = generic_enum_specialized_variant_fields(
            generator,
            specialization,
            variant_name,
        )
        if fields is not None:
            return dict(fields)

    fields_by_variant = getattr(generator, "enum_struct_variant_fields", {})
    return fields_by_variant.get(enum_name, {}).get(variant_name)


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


def generate_generic_enum_constants(generator, definitions, qualifier="static const"):
    code = ""
    for name, definition in sorted((definitions or {}).items()):
        next_value = 0
        for variant in definition["enum"].variants or []:
            if getattr(variant, "value", None) is None:
                value = str(next_value)
                next_value += 1
            else:
                value = generator.generate_expression(variant.value)
                literal_value = literal_int_value(variant.value)
                next_value = literal_value + 1 if literal_value is not None else 0
            code += (
                f"{qualifier} int {enum_constant_name(name, variant.name)} = {value};\n"
            )
    if code:
        code += "\n"
    return code


def generate_generic_enum_structs(generator, specializations):
    code = ""
    for specialization in (specializations or {}).values():
        code += f"struct {specialization['struct_name']} {{\n"
        code += "    int variant;\n"
        for field_name, field_type in generic_enum_specialized_fields(
            generator,
            specialization,
        ):
            declaration = format_c_style_array_declaration(
                generator.map_type(field_type),
                field_name,
            )
            code += f"    {declaration};\n"
        code += "};\n\n"
    return code


def generate_generic_enum_constructor_functions(generator, specializations):
    code = ""
    for specialization in (specializations or {}).values():
        all_fields = generic_enum_specialized_fields(generator, specialization)
        definition = specialization["definition"]
        for variant in definition["enum"].variants or []:
            variant_fields = generic_enum_specialized_variant_fields(
                generator,
                specialization,
                variant.name,
            )
            params = ", ".join(
                f"{generator.map_type(field_type)} payload{index}"
                for index, (_field_name, field_type) in enumerate(variant_fields)
            )
            code += (
                f"{specialization['struct_name']} "
                f"{enum_variant_constructor_name(specialization['struct_name'], variant.name)}({params}) {{\n"
            )
            code += f"    {specialization['struct_name']} result;\n"
            code += (
                f"    result.variant = "
                f"{enum_constant_name(definition['name'], variant.name)};\n"
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


def collect_generic_enum_specialization_member_types(generator, specializations):
    member_types = {}
    for type_text, specialization in (specializations or {}).items():
        fields = {"variant": "int"}
        for field_name, field_type in generic_enum_specialized_fields(
            generator,
            specialization,
        ):
            fields[field_name] = field_type
        member_types[type_text] = dict(fields)
        member_types[specialization["struct_name"]] = dict(fields)
    return member_types


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
    custom_default = getattr(generator, "default_value_expression_for_type", None)
    if callable(custom_default):
        default_value = custom_default(field_type)
        if default_value is not None:
            return default_value

    mapped_type = generator.map_type(field_type)
    if mapped_type == "bool":
        return "false"
    if mapped_type.startswith("bool") or mapped_type.startswith("bvec"):
        return f"{mapped_type}(false)"
    return f"{mapped_type}(0)"


def generic_enum_value_expression(generator, path):
    if "::" not in str(path):
        return None

    enum_name, variant_name = str(path).split("::", 1)
    expected_type = getattr(generator, "current_expression_expected_type", None)
    specialization = resolve_generic_enum_specialization(
        generator,
        expected_type,
        expected_base=enum_name,
    )
    if specialization is None:
        return None

    fields = generic_enum_specialized_variant_fields(
        generator,
        specialization,
        variant_name,
    )
    if fields == []:
        constructor = enum_variant_constructor_name(
            specialization["struct_name"],
            variant_name,
        )
        return f"{constructor}()"
    return None


def resolve_generic_enum_specialization(generator, type_value, expected_base=None):
    type_text = generator_type_name_string(generator, type_value)
    if not type_text:
        return None

    base_name, generic_args = generic_type_parts(type_text)
    if expected_base is not None and base_name != expected_base:
        return None

    definitions = getattr(generator, "generic_enum_struct_definitions", {})
    definition = definitions.get(base_name)
    if definition is None:
        return None
    if len(generic_args) != len(definition["generic_params"]):
        return None

    specializations = getattr(generator, "generic_enum_specializations", {})
    return specializations.get(type_text) or build_generic_enum_specialization(
        type_text,
        definition,
    )


def generic_enum_specialized_fields(generator, specialization):
    fields = []
    for variant in specialization["definition"]["enum"].variants or []:
        fields.extend(
            generic_enum_specialized_variant_fields(
                generator,
                specialization,
                variant.name,
            )
            or []
        )
    deduped = []
    seen = {}
    for field_name, field_type in fields:
        previous_type = seen.get(field_name)
        if previous_type is not None:
            if previous_type == field_type:
                continue
            return []
        seen[field_name] = field_type
        deduped.append((field_name, field_type))
    return deduped


def generic_enum_specialized_variant_fields(generator, specialization, variant_name):
    variant = generic_enum_variant_by_name(specialization["definition"], variant_name)
    if variant is None:
        return None

    fields = enum_variant_payload_fields(variant)
    if fields is None:
        return None

    specialized_fields = []
    for field_name, field_type in fields:
        field_type_text = generator_type_name_string(generator, field_type)
        specialized_fields.append(
            (
                field_name,
                substitute_generic_type_name(
                    field_type_text,
                    specialization["substitutions"],
                ),
            )
        )
    return specialized_fields


def generic_enum_variant_by_name(definition, variant_name):
    for variant in definition["enum"].variants or []:
        if variant.name == variant_name:
            return variant
    return None


def generator_type_name_string(generator, type_value):
    if type_value is None:
        return None
    type_name = getattr(generator, "type_name_string", None)
    if type_name is not None:
        return type_name(type_value)
    return str(type_value)


def substitute_generic_type_name(type_name, substitutions):
    type_name = str(type_name or "")
    if type_name in substitutions:
        return substitutions[type_name]

    stripped_reference = _strip_reference_wrappers(type_name)
    if stripped_reference != type_name:
        substituted = substitute_generic_type_name(stripped_reference, substitutions)
        if type_name.lstrip().startswith("&"):
            return f"&{substituted}"
        return f"{substituted}&"

    if type_name.strip().endswith("*"):
        pointee_type = type_name.strip()[:-1].strip()
        if pointee_type:
            return f"{substitute_generic_type_name(pointee_type, substitutions)}*"

    base_name, generic_args = generic_type_parts(type_name)
    if not generic_args:
        return substitutions.get(base_name, type_name)

    return (
        f"{base_name}<"
        + ", ".join(
            substitute_generic_type_name(arg, substitutions) for arg in generic_args
        )
        + ">"
    )


def generic_enum_specialization_name(type_name):
    base_name, generic_args = generic_type_parts(type_name)
    suffix = "_".join(sanitize_type_name(arg) for arg in generic_args)
    return f"{sanitize_type_name(base_name)}_{suffix}" if suffix else base_name


def sanitize_type_name(type_name):
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_]", "_", str(type_name))).strip("_")


def _strip_reference_wrappers(type_name):
    type_name = str(type_name or "").strip()
    while type_name.startswith("&"):
        type_name = type_name[1:].strip()
    while type_name.endswith("&"):
        type_name = type_name[:-1].strip()
    return type_name


def generic_type_parts(type_name):
    type_name = str(type_name or "").strip()
    if "<" not in type_name or not type_name.endswith(">"):
        return type_name, []

    base_name, rest = type_name.split("<", 1)
    return base_name.strip(), split_top_level_generic_args(rest[:-1])


def split_top_level_generic_args(args_text):
    args = []
    depth = 0
    current = []
    for char in str(args_text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current or args_text:
        args.append("".join(current).strip())
    return [arg for arg in args if arg]


def generic_type_depth(type_name):
    _base_name, generic_args = generic_type_parts(type_name)
    if not generic_args:
        return 0
    return 1 + max(generic_type_depth(arg) for arg in generic_args)


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
