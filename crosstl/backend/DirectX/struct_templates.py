"""HLSL 2021 struct templates import as CrossGL generic structures.

The importer dropped a struct's ``template <typename T>`` list and every use's arguments, so ``template <typename T>
struct Box { T value; };`` became ``struct Box { T value; }`` with an undeclared ``T``, ``Box<float> a;`` and
``Box<int> b;`` both became ``Box``, and a member function was lowered for the bare ``Box``.

The parser keeps the parameters on the struct (``template_parameters``, as for function templates in
:mod:`.function_templates`), and the importer emits ``generic<T> struct Box``, keeps each use's arguments in CrossGL
spellings (``Box<vec2>``), and lowers a member function to a generic function over its receiver (``generic<T> T
Box_twice(in Box<T> __this)``), whose calls deduce ``T`` from the receiver. An explicit specialization (``template <>
struct Box<int>``) becomes the concrete ``Box_int_specialization`` that uses with those arguments name. A value
parameter (``template <typename T, uint N>``) is a generic parameter that each use binds (``Ring<float, 4u>``), with
defaults spelled out at the use. Partial specializations have no CrossGL form yet: such a template is left
undeclared, and a program that names it is refused.
"""

from __future__ import annotations

import re

from .function_templates import TemplateParameter, template_argument
from .namespaces import source_name

__all__ = [
    "explicit_struct_specialization_arguments",
    "name_explicit_struct_specializations",
    "refuse_struct_template_references",
    "struct_template_parameters",
    "struct_template_prefix",
    "struct_template_owner",
    "struct_template_receiver_type",
    "struct_template_type_names",
    "templated_struct_type",
    "unsupported_struct_templates",
]

_INTEGER_ARGUMENT = re.compile(r"^[+-]?(?:0[xX][0-9A-Fa-f]+|\d+)[uUlL]*$")


def struct_template_parameters(struct):
    """The template parameters of a struct template, ``[]`` for an explicit specialization, else ``None``."""
    return getattr(struct, "template_parameters", None)


def struct_template_type_names(struct):
    """The names of a struct template's type parameters, for the types its members and methods spell."""
    return frozenset(
        parameter.name
        for parameter in struct_template_parameters(struct) or []
        if isinstance(parameter, TemplateParameter)
        and parameter.is_type
        and parameter.name
    )


def _argument(codegen, text):
    text = str(text).strip()
    if _INTEGER_ARGUMENT.match(text):
        return text
    return str(codegen.map_type(text)).strip()


def explicit_struct_specialization_arguments(struct):
    """The arguments of ``template <> struct Box<float>`` (``["float"]``), else ``None``."""
    arguments = getattr(struct, "specialization_arguments", None)
    if struct_template_parameters(struct) == [] and isinstance(arguments, list):
        return arguments
    return None


def _unsupported_reason(struct):
    """Why CrossGL cannot declare ``struct`` as a generic structure, else ``None``."""
    parameters = struct_template_parameters(struct)
    if getattr(struct, "specialization_arguments", None) is True:
        return (
            f"HLSL struct '{source_name(struct)}' is a partial specialization, which CrossGL generic structures "
            "cannot express"
        )
    if parameters == [] and explicit_struct_specialization_arguments(struct) is None:
        return f"HLSL struct '{source_name(struct)}' is an explicit specialization without arguments"
    for parameter in parameters or []:
        if not parameter.name:
            return f"HLSL struct template '{source_name(struct)}' declares an unnamed type parameter"
    return None


def unsupported_struct_templates(structs):
    """``{name: reason}`` for the struct templates CrossGL cannot declare, with every struct sharing such a name.

    A partial specialization changes what its primary means, so the primary is not declared either; so does an
    explicit specialization of a primary that is itself not declared, or that no primary template declares.
    """
    reasons = {}
    for struct in structs or []:
        reason = _unsupported_reason(struct)
        if reason is not None and getattr(struct, "name", None):
            reasons.setdefault(struct.name, reason)
    primaries = {
        struct.name for struct in structs or [] if struct_template_parameters(struct)
    }
    for struct in structs or []:
        if (
            explicit_struct_specialization_arguments(struct) is not None
            and struct.name not in primaries
        ):
            reasons.setdefault(
                struct.name,
                f"HLSL struct '{source_name(struct)}' is an explicit specialization of no struct template",
            )
    return reasons


def name_explicit_struct_specializations(codegen, structs):
    """Rename each explicit specialization to ``<base>_<args>_specialization``; returns ``{(base, args): name}``.

    The CrossGL arguments in the key are those a use spells (``Box<vec2>`` for ``Box<float2>``), so the use reads
    the specialization's own members, as C++ requires (C++17 17.7.3).
    """
    names = {}
    reasons = getattr(codegen, "unsupported_struct_templates", {})
    for struct in structs or []:
        arguments = explicit_struct_specialization_arguments(struct)
        if arguments is None or struct.name in reasons:
            continue
        template = getattr(codegen, "struct_templates", {}).get(struct.name, ())
        mapped = _template_arguments(codegen, template, arguments)
        suffix = re.sub(r"\W+", "_", "_".join(mapped)).strip("_")
        name = f"{struct.name}_{suffix}_specialization"
        names[(struct.name, mapped)] = name
        struct.hlsl_specialized_template = struct.name
        struct.name = name
    return names


def refuse_struct_template_references(codegen, ast, reasons):
    """Refuse each unsupported struct template that the rest of the program names.

    The importer leaves such a template undeclared; a program that never names it (a library of type traits, say)
    loses nothing, and any other is refused rather than translated with an undeclared name.
    """
    if not reasons:
        return
    pattern = re.compile(
        r"(?<![\w.])(?:"
        + "|".join(re.escape(name) for name in sorted(reasons))
        + r")\b"
    )
    found = set()

    def visit(value, seen):
        if isinstance(value, str):
            found.update(match.group(0) for match in pattern.finditer(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, seen)
        elif hasattr(value, "__dict__") and id(value) not in seen:
            seen.add(id(value))
            for field, item in vars(value).items():
                if field not in ("parent", "source_location"):
                    visit(item, seen)

    seen = set()
    for struct in getattr(ast, "structs", None) or []:
        if getattr(struct, "name", None) in reasons:
            seen.add(id(struct))
    for collection in (
        "functions",
        "structs",
        "global_variables",
        "cbuffers",
        "typedefs",
    ):
        visit(getattr(ast, collection, None) or [], seen)
    for name in sorted(found):
        codegen.refuse_import(reasons[name])


def struct_template_prefix(codegen, struct):
    """``generic<T, U = vec2> `` for a struct template, ``""`` for any other struct."""
    parameters = struct_template_parameters(struct)
    if not parameters:
        return ""
    rendered = []
    for parameter in parameters:
        text = parameter.name
        # Every use spells its arguments out (``_with_defaults``), so a value default needs no CrossGL form.
        if parameter.default is not None and parameter.is_type:
            text += f" = {_argument(codegen, parameter.default)}"
        rendered.append(text)
    return f"generic<{', '.join(rendered)}> "


def _with_defaults(template, arguments):
    """``arguments`` completed with the template's default arguments, which may name earlier parameters."""
    arguments = [str(argument).strip() for argument in arguments]
    for parameter in template[len(arguments) :]:
        if parameter.default is None:
            break
        bound = dict(zip((item.name for item in template), arguments))
        # Only the names of earlier parameters are replaced; every other word of the default is already HLSL.
        earlier = r"\b(?:" + "|".join(re.escape(name) for name in bound) + r")\b"
        arguments.append(
            re.sub(earlier, lambda match: bound[match.group(0)], parameter.default)
            if bound
            else parameter.default
        )
    return arguments


def _template_arguments(codegen, template, arguments):
    """The CrossGL arguments of a use of ``template``, defaults spelled out, values typed by their parameters."""
    template = list(template or [])
    return tuple(
        template_argument(
            codegen, template[index] if index < len(template) else None, argument
        )
        for index, argument in enumerate(_with_defaults(template, arguments))
    )


def templated_struct_type(codegen, base, arguments):
    """``Box<vec2>`` for the use ``Box<float2>`` of a struct template, its explicit specialization's name when one
    matches the arguments, else ``None``. Default arguments are spelled out: ``Pair<vec2>`` of ``template
    <typename T, typename U = float2>`` is ``Pair<vec2, vec2>``."""
    base = str(base).strip()
    template = getattr(codegen, "struct_templates", {}).get(base)
    if template is None:
        return None
    mapped = _template_arguments(codegen, template, arguments)
    specialized = getattr(codegen, "explicit_struct_specializations", {}).get(
        (base, mapped)
    )
    if specialized is not None:
        return specialized
    return f"{codegen.sanitize_type_name(base)}<{', '.join(mapped)}>"


def struct_template_owner(codegen, raw_type):
    """The struct whose member functions a receiver of HLSL type ``raw_type`` calls.

    ``Box<float>`` calls those of the explicit specialization ``Box<float>`` when the program declares one, and
    those of the template ``Box`` otherwise; any other type gives ``None``.
    """
    text = str(raw_type or "").strip()
    if "<" not in text or not text.endswith(">"):
        return None
    base, arguments = text.split("<", 1)
    base = base.strip()
    template = getattr(codegen, "struct_templates", {}).get(base)
    if template is None:
        return None
    mapped = _template_arguments(
        codegen, template, codegen.split_generic_arguments(arguments[:-1])
    )
    return getattr(codegen, "explicit_struct_specializations", {}).get(
        (base, mapped), base
    )


def struct_template_receiver_type(codegen, owner):
    """The receiver type of a member function of ``owner``: ``Box<T>`` for a struct template, else ``owner``."""
    template = getattr(codegen, "struct_templates", {}).get(owner)
    if not template:
        return owner
    names = [parameter.name for parameter in template]
    return f"{owner}<{', '.join(names)}>"
