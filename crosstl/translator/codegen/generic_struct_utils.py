"""Helpers for lowering CrossGL generic structs to C-style shader languages."""

from ..ast import ArrayNode, EnumNode, FunctionNode, StructNode
from .array_utils import format_c_style_array_declaration, split_array_type_suffix
from .enum_utils import (
    collect_generic_parameter_names,
    default_value_expression,
    generator_type_name_string,
    generic_type_depth,
    generic_type_parts,
    sanitize_type_name,
    substitute_generic_type_name,
    type_node_contains_generic_parameter,
)


def collect_generic_struct_definitions(nodes, excluded_names=None):
    """Return generic data structs that can be concretely specialized."""
    excluded_names = set(excluded_names or set())
    definitions = {}
    for node in nodes or []:
        if not isinstance(node, StructNode):
            continue
        if node.name in excluded_names:
            continue
        generic_params = [
            getattr(param, "name", None)
            for param in getattr(node, "generic_params", [])
        ]
        generic_params = [name for name in generic_params if name]
        if not generic_params:
            continue
        members = generic_struct_data_members(node)
        if not members:
            continue
        definitions[node.name] = {
            "name": node.name,
            "generic_params": generic_params,
            "members": members,
        }
    return definitions


def generic_struct_data_members(node):
    members = []
    for member in getattr(node, "members", []) or []:
        if isinstance(member, (EnumNode, FunctionNode, StructNode)):
            return None
        if not getattr(member, "name", None):
            return None
        if (
            isinstance(member, ArrayNode)
            or hasattr(member, "member_type")
            or hasattr(member, "vtype")
        ):
            members.append(member)
            continue
        return None
    return members


def collect_generic_struct_specializations(nodes, definitions, type_name_string):
    """Collect concrete generic struct instantiations used by the AST."""
    if not definitions:
        return {}

    generic_names = collect_generic_parameter_names(nodes)
    specializations = {}
    visited = set()

    def add_type_text(type_text):
        type_text = normalize_specialization_type_text(type_text)
        if not type_text:
            return None

        base_name, generic_args = generic_type_parts(type_text)
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

        specialization = build_generic_struct_specialization(type_text, definition)
        specializations[type_text] = specialization
        return specialization

    def visit(value):
        if value is None:
            return
        if isinstance(value, (str, int, float, bool)):
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
            definition = definitions[name]
            if len(generic_args) == len(definition["generic_params"]) and not any(
                type_node_contains_generic_parameter(arg, generic_names)
                for arg in generic_args
            ):
                add_type_text(type_name_string(value))

        for child in vars(value).values():
            visit(child)

    visit(nodes)

    changed = True
    while changed:
        changed = False
        for specialization in list(specializations.values()):
            for _field_name, field_type in generic_struct_specialized_fields(
                type_name_string,
                specialization,
            ):
                if add_type_text(field_type) is not None:
                    changed = True

    return dict(
        sorted(
            specializations.items(),
            key=lambda item: (
                generic_type_depth(item[0]),
                item[1]["struct_name"],
            ),
        )
    )


def normalize_specialization_type_text(type_text):
    type_text = str(type_text or "").strip()
    if not type_text:
        return None
    base_type, _array_suffix = split_array_type_suffix(type_text)
    return base_type.strip()


def type_text_contains_generic_parameter(type_text, generic_names):
    type_text = normalize_specialization_type_text(type_text)
    if not type_text:
        return False
    base_name, generic_args = generic_type_parts(type_text)
    if not generic_args:
        return base_name in generic_names
    return any(
        type_text_contains_generic_parameter(arg, generic_names) for arg in generic_args
    )


def build_generic_struct_specialization(type_text, definition):
    base_name, generic_args = generic_type_parts(type_text)
    substitutions = dict(zip(definition["generic_params"], generic_args))
    return {
        "type_name": type_text,
        "base_name": base_name,
        "struct_name": generic_struct_specialization_name(type_text),
        "definition": definition,
        "substitutions": substitutions,
    }


def generic_struct_specialized_type_name(generator, vtype):
    specialization = resolve_generic_struct_specialization(generator, vtype)
    if specialization is None:
        return None
    return specialization["struct_name"]


def resolve_generic_struct_specialization(generator, type_value, expected_base=None):
    type_text = normalize_specialization_type_text(
        generator_type_name_string(generator, type_value)
    )
    if not type_text:
        return None

    base_name, generic_args = generic_type_parts(type_text)
    if expected_base is not None and base_name != expected_base:
        return None

    definitions = getattr(generator, "generic_struct_definitions", {})
    definition = definitions.get(base_name)
    if definition is None:
        return None
    if len(generic_args) != len(definition["generic_params"]):
        return None

    specializations = getattr(generator, "generic_struct_specializations", {})
    return specializations.get(type_text) or build_generic_struct_specialization(
        type_text,
        definition,
    )


def generate_generic_structs(generator, specializations):
    code = ""
    for specialization in (specializations or {}).values():
        code += f"struct {specialization['struct_name']} {{\n"
        for field_name, field_type in generic_struct_specialized_fields(
            generator.type_name_string,
            specialization,
        ):
            declaration = format_c_style_array_declaration(
                generator.map_type(field_type),
                field_name,
            )
            code += f"    {declaration};\n"
        code += "};\n\n"
    return code


def collect_generic_struct_specialization_member_types(generator, specializations):
    member_types = {}
    for type_text, specialization in (specializations or {}).items():
        fields = {
            field_name: field_type
            for field_name, field_type in generic_struct_specialized_fields(
                generator.type_name_string,
                specialization,
            )
        }
        member_types[type_text] = dict(fields)
        member_types[specialization["struct_name"]] = dict(fields)
    return member_types


def infer_struct_constructor_type(generator, expr):
    constructor_type = getattr(expr, "constructor_type", None)
    constructor_type_text = generator_type_name_string(generator, constructor_type)
    constructor_base, constructor_args = generic_type_parts(constructor_type_text)
    if not constructor_base or "::" in constructor_base:
        return None

    expected_type = getattr(generator, "current_expression_expected_type", None)
    expected_base, _expected_args = generic_type_parts(expected_type)
    if expected_base == constructor_base:
        return normalize_specialization_type_text(expected_type)

    if constructor_args:
        return normalize_specialization_type_text(constructor_type_text)

    inferred_generic = infer_generic_struct_constructor_type(generator, expr)
    if inferred_generic is not None:
        return inferred_generic

    if constructor_base in getattr(generator, "structs_by_name", {}):
        return constructor_base
    return None


def generate_struct_constructor_expression(generator, expr):
    constructor = generate_generic_struct_constructor_expression(generator, expr)
    if constructor is not None:
        return constructor

    constructor_type = getattr(expr, "constructor_type", None)
    constructor_type_text = generator_type_name_string(generator, constructor_type)
    constructor_base, constructor_args = generic_type_parts(constructor_type_text)
    if not constructor_base or constructor_args or "::" in constructor_base:
        return None
    if constructor_base not in getattr(generator, "structs_by_name", {}):
        return None

    fields = struct_constructor_fields(generator, constructor_base)
    rendered_args = render_constructor_arguments(
        generator,
        constructor_base,
        expr,
        fields,
    )
    return format_struct_constructor_expression(
        generator,
        generator.map_type(constructor_base),
        rendered_args,
    )


def generate_generic_struct_constructor_expression(generator, expr):
    constructor_type = getattr(expr, "constructor_type", None)
    constructor_type_text = generator_type_name_string(generator, constructor_type)
    constructor_base, _constructor_args = generic_type_parts(constructor_type_text)
    if not constructor_base:
        return None

    specialization = resolve_generic_struct_constructor_specialization(generator, expr)
    if specialization is None:
        return None

    fields = generic_struct_specialized_fields(
        generator.type_name_string,
        specialization,
    )
    rendered_args = render_constructor_arguments(
        generator,
        constructor_base,
        expr,
        fields,
    )
    return format_struct_constructor_expression(
        generator,
        specialization["struct_name"],
        rendered_args,
    )


def resolve_generic_struct_constructor_specialization(generator, expr):
    constructor_type = getattr(expr, "constructor_type", None)
    constructor_type_text = generator_type_name_string(generator, constructor_type)
    constructor_base, _constructor_args = generic_type_parts(constructor_type_text)
    if not constructor_base:
        return None

    expected_type = getattr(generator, "current_expression_expected_type", None)
    specialization = resolve_generic_struct_specialization(
        generator,
        expected_type,
        expected_base=constructor_base,
    )
    if specialization is not None:
        return specialization

    specialization = resolve_generic_struct_specialization(
        generator,
        constructor_type,
        expected_base=constructor_base,
    )
    if specialization is not None:
        return specialization

    inferred_type = infer_generic_struct_constructor_type(generator, expr)
    if inferred_type is None:
        return None

    return resolve_generic_struct_specialization(
        generator,
        inferred_type,
        expected_base=constructor_base,
    )


def infer_generic_struct_constructor_type(generator, expr):
    constructor_type = getattr(expr, "constructor_type", None)
    constructor_type_text = generator_type_name_string(generator, constructor_type)
    constructor_base, _constructor_args = generic_type_parts(constructor_type_text)
    definition = getattr(generator, "generic_struct_definitions", {}).get(
        constructor_base
    )
    if definition is None:
        return None

    fields = [
        (
            member.name,
            generic_struct_member_type_name(member, generator.type_name_string),
        )
        for member in definition["members"]
    ]
    argument_expressions = constructor_argument_expressions(expr, fields)
    if argument_expressions is None:
        return None

    substitutions = {}
    for field_name, field_type in fields:
        argument = argument_expressions.get(field_name)
        if argument is None:
            continue
        argument_type = expression_result_type(generator, argument)
        if not argument_type:
            continue
        if not infer_generic_type_substitutions(
            field_type,
            argument_type,
            set(definition["generic_params"]),
            substitutions,
        ):
            return None

    if any(param not in substitutions for param in definition["generic_params"]):
        return None

    return (
        f"{constructor_base}<"
        + ", ".join(substitutions[param] for param in definition["generic_params"])
        + ">"
    )


def render_constructor_arguments(generator, constructor_name, expr, fields):
    positional_args = list(getattr(expr, "arguments", []) or [])
    named_args = dict(getattr(expr, "named_arguments", {}) or {})
    field_names = [field_name for field_name, _field_type in fields]

    if len(positional_args) > len(fields):
        raise ValueError(
            f"Struct constructor {constructor_name} expects at most {len(fields)} "
            f"arguments, got {len(positional_args)}"
        )
    unknown_names = sorted(set(named_args) - set(field_names))
    if unknown_names:
        raise ValueError(
            f"Struct constructor {constructor_name} has no field "
            f"{', '.join(unknown_names)}"
        )

    rendered_args = []
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
        rendered_args.append(default_value_expression(generator, field_type))

    return rendered_args


def constructor_argument_expressions(expr, fields):
    positional_args = list(getattr(expr, "arguments", []) or [])
    named_args = dict(getattr(expr, "named_arguments", {}) or {})
    field_names = [field_name for field_name, _field_type in fields]
    if len(positional_args) > len(fields):
        return None
    if set(named_args) - set(field_names):
        return None

    arguments = {}
    for index, (field_name, _field_type) in enumerate(fields):
        if index < len(positional_args):
            arguments[field_name] = positional_args[index]
        elif field_name in named_args:
            arguments[field_name] = named_args[field_name]
    return arguments


def infer_generic_type_substitutions(
    pattern_type,
    concrete_type,
    generic_params,
    substitutions,
):
    pattern_base, pattern_array = split_array_type_suffix(str(pattern_type or ""))
    concrete_base, concrete_array = split_array_type_suffix(str(concrete_type or ""))
    if pattern_array != concrete_array:
        return True

    pattern_name, pattern_args = generic_type_parts(pattern_base)
    concrete_name, concrete_args = generic_type_parts(concrete_base)
    if pattern_name in generic_params and not pattern_args:
        previous = substitutions.get(pattern_name)
        if previous is None:
            substitutions[pattern_name] = concrete_base
            return True
        return previous == concrete_base

    if len(pattern_args) != len(concrete_args):
        return True

    return all(
        infer_generic_type_substitutions(
            pattern_arg,
            concrete_arg,
            generic_params,
            substitutions,
        )
        for pattern_arg, concrete_arg in zip(pattern_args, concrete_args)
    )


def struct_constructor_fields(generator, struct_name):
    struct = getattr(generator, "structs_by_name", {}).get(struct_name)
    members = generic_struct_data_members(struct) if struct is not None else None
    if members is None:
        member_types = getattr(generator, "struct_member_types", {}).get(
            struct_name,
            {},
        )
        return list(member_types.items())
    return [
        (
            member.name,
            generic_struct_member_type_name(member, generator.type_name_string),
        )
        for member in members
    ]


def expression_result_type(generator, expr):
    result_type = getattr(generator, "expression_result_type", None)
    if result_type is None:
        return None
    return generator.type_name_string(result_type(expr))


def format_struct_constructor_expression(generator, type_name, rendered_args):
    args = ", ".join(rendered_args)
    if getattr(generator, "struct_constructor_uses_braces", False):
        return f"{type_name}{{{args}}}"
    return f"{type_name}({args})"


def generic_struct_specialized_fields(type_name_string, specialization):
    fields = []
    for member in specialization["definition"]["members"]:
        field_name = member.name
        field_type = generic_struct_member_type_name(member, type_name_string)
        fields.append(
            (
                field_name,
                substitute_generic_type_name_with_arrays(
                    field_type,
                    specialization["substitutions"],
                ),
            )
        )
    return fields


def generic_struct_member_type_name(member, type_name_string):
    if isinstance(member, ArrayNode):
        element_type = getattr(
            member,
            "element_type",
            getattr(member, "vtype", "float"),
        )
        base_type = type_name_string(element_type)
        size = getattr(member, "size", None)
        return f"{base_type}[{size}]" if size else f"{base_type}[]"
    if hasattr(member, "member_type"):
        return type_name_string(member.member_type)
    return type_name_string(getattr(member, "vtype", "float"))


def substitute_generic_type_name_with_arrays(type_name, substitutions):
    base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
    return substitute_generic_type_name(base_type, substitutions) + array_suffix


def generic_struct_specialization_name(type_name):
    base_name, generic_args = generic_type_parts(type_name)
    suffix = "_".join(sanitize_type_name(arg) for arg in generic_args)
    return f"{sanitize_type_name(base_name)}_{suffix}" if suffix else base_name
