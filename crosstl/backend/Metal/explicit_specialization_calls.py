"""Deduced calls that select an explicit function template specialization.

``template <> float twice<float>(float x)`` replaces the body of the primary ``template <typename T> T twice(T)``
for ``T = float`` ([temp.expl.spec]/1). A call reaches it whenever the primary's template arguments, explicit or
deduced from the call's arguments ([temp.deduct.call]), are exactly ``float``: ``twice(f)``, ``twice<float>(f)`` and,
for ``template <typename U, typename T> U cast_to(T)``, ``cast_to<bool>(f)``.

The importer keeps primaries generic for the CrossGL and Metal targets and emits each explicit specialization as a
concrete function. Calls that spell every template argument already name that function. A call that deduces some
of them used to name the generic primary, so the generators instantiated the primary's body instead of the
specialization's. :func:`explicit_specialization_call` rebinds such a call when every argument it deduces is a
concrete type; a call whose deduced arguments depend on the enclosing template's parameters is left to the
generators' instantiation.
"""

from __future__ import annotations

import copy
import re

from .conditional_types import split_template_arguments

__all__ = ["explicit_specialization_call"]

_QUALIFIERS = re.compile(
    r"\b(?:const|volatile|thread|device|threadgroup|constant|ray_data|object_data)\b"
)
_TEMPLATE_NAME = re.compile(
    r"^(?P<base>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*<(?P<arguments>.*)>$"
)


def _split_name(name):
    match = _TEMPLATE_NAME.match(str(name).strip())
    if match is None:
        return str(name).strip(), []
    arguments = split_template_arguments(match.group("arguments"))
    if arguments is None:
        return None, None
    return match.group("base"), [argument.strip() for argument in arguments]


def _specializations(codegen, base):
    """``(template arguments, name)`` of each concrete explicit specialization of ``base``."""
    overloads = codegen.user_function_overloads_by_name
    key = (id(overloads), len(overloads))
    cached = getattr(codegen, "_explicit_specializations_by_base", None)
    index = cached[1] if cached is not None and cached[0] == key else None
    if index is None:
        index = {}
        for name, functions in overloads.items():
            if "<" not in str(name):
                continue
            specialization_base, arguments = _split_name(name)
            if not specialization_base or not arguments:
                continue
            if any(_template_parameters(function) for function in functions):
                continue
            index.setdefault(specialization_base, []).append((tuple(arguments), name))
        for entries in index.values():
            entries.sort(key=lambda entry: entry[1])
        codegen._explicit_specializations_by_base = (key, index)
    return index.get(base, [])


def _template_parameters(function):
    generics = getattr(function, "generics", None) or []
    if generics:
        return [str(name) for name in generics]
    return [
        str(parameter[-1])
        for parameter in getattr(function, "template_parameters", None) or []
        if isinstance(parameter, (tuple, list)) and parameter
    ]


def _normalized_type(text):
    text = _QUALIFIERS.sub(" ", str(text))
    text = text.replace("&", " ").replace("metal::", "")
    return re.sub(r"\s+", "", text)


def _deduce(parameter_type, argument_type, parameters, bindings):
    """Bind the template parameters ``parameter_type`` spells by matching it against ``argument_type``."""
    pattern = _normalized_type(parameter_type)
    names = [
        name for name in parameters if re.search(rf"\b{re.escape(name)}\b", pattern)
    ]
    if not names:
        return True
    regex = ""
    for token in re.split(r"(\b[A-Za-z_]\w*\b)", pattern):
        if token in names:
            group = f"p{names.index(token)}"
            regex += (
                f"(?P={group})" if f"(?P<{group}>" in regex else f"(?P<{group}>.+?)"
            )
        else:
            regex += re.escape(token)
    match = re.fullmatch(regex, _normalized_type(argument_type))
    if match is None:
        return False
    for position, name in enumerate(names):
        value = match.group(f"p{position}")
        if bindings.setdefault(name, value) != value:
            return False
    return True


def _enclosing_template_parameters(codegen):
    function = getattr(codegen, "current_function", None)
    return set(_template_parameters(function)) if function is not None else set()


def explicit_specialization_call(codegen, expr):
    """A copy of call ``expr`` naming the explicit specialization it selects, or ``None``."""
    base, explicit = _split_name(getattr(expr, "name", ""))
    if not base:
        return None
    specializations = _specializations(codegen, base)
    if not specializations:
        return None
    args = list(getattr(expr, "args", None) or [])
    primaries = [
        function
        for function in codegen.user_function_overloads_by_name.get(base, [])
        if _template_parameters(function)
        and len(getattr(function, "params", None) or []) == len(args)
    ]
    if len(primaries) != 1:
        return None
    primary = primaries[0]
    parameters = _template_parameters(primary)
    if len(explicit) >= len(parameters):
        return None
    bindings = dict(zip(parameters, explicit))
    for parameter, argument in zip(primary.params, args):
        parameter_type = codegen.metal_declaration_expression_type(parameter)
        argument_type = codegen.metal_source_overload_value_type(
            codegen.expression_metal_type(argument)
        )
        if parameter_type is None or argument_type is None:
            return None
        if not _deduce(parameter_type, argument_type, parameters, bindings):
            return None
    if any(name not in bindings for name in parameters):
        return None
    dependent = _enclosing_template_parameters(codegen)
    ordered = [bindings[name] for name in parameters]
    if any(
        re.search(rf"\b{re.escape(name)}\b", value)
        for value in ordered
        for name in dependent
    ):
        return None
    identity = codegen.metal_source_overload_type_identity
    wanted = tuple(identity(value) for value in ordered)
    if None in wanted:
        return None
    for arguments, name in specializations:
        if (
            len(arguments) == len(wanted)
            and tuple(identity(a) for a in arguments) == wanted
        ):
            rebound = copy.copy(expr)
            rebound.name = name
            return rebound
    return None
