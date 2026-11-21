"""Default arguments of CUDA and HIP functions, carried onto CrossGL parameters.

A C++ function declaration may give its trailing parameters default arguments, and a call that omits them receives
those values (C++17 [dcl.fct.default]). CrossGL keeps a parameter's default (``f32 b = 2.0``), and the translator
fills every omitted argument for each target, so the importers write each default on its CrossGL parameter.

A default can sit on an earlier declaration instead of the definition, as headers write it: each parameter takes
its default from exactly one declaration in the scope, so :func:`merge_declared_defaults` copies the defaults of
body-less declarations onto the definition they declare. A default the parser cannot read as an expression is kept
as :class:`UnparsedParameterDefault`, and writing it refuses the import: dropping it would leave every call that
omits the argument one argument short.
"""

from __future__ import annotations


class UnparsedParameterDefault:
    """A default argument the parser could not read as an expression, with its source text for the diagnostic."""

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"UnparsedParameterDefault({self.text!r})"


def parameter_default(param):
    """The default argument recorded on a parsed parameter (a CUDA ``VariableNode`` or a HIP parameter dict)."""
    if isinstance(param, dict):
        return param.get("default")
    return getattr(param, "default_value", None)


def _set_parameter_default(param, value):
    if isinstance(param, dict):
        param["default"] = value
    else:
        param.default_value = value


def _parameter_type(param):
    raw = (
        param.get("type", "")
        if isinstance(param, dict)
        else getattr(param, "vtype", "")
    )
    return " ".join(str(raw or "").split())


def _parameters(function):
    return list(getattr(function, "params", None) or [])


def merge_declared_defaults(functions):
    """Copy the defaults of body-less declarations onto the definition each one declares.

    A declaration declares a definition with the same name and parameter types; when no definition matches the
    types exactly, the only definition with that name and parameter count is taken. A definition keeps the
    defaults it writes itself.
    """
    functions = list(functions or [])
    definitions = [
        function
        for function in functions
        if getattr(function, "body", None) is not None
    ]
    for declaration in functions:
        if getattr(declaration, "body", None) is not None:
            continue
        declared = _parameters(declaration)
        if not any(parameter_default(param) is not None for param in declared):
            continue
        definition = _declared_definition(declaration, declared, definitions)
        if definition is None:
            continue
        for param, value in zip(_parameters(definition), declared):
            default = parameter_default(value)
            if default is not None and parameter_default(param) is None:
                _set_parameter_default(param, default)


def _declared_definition(declaration, declared, definitions):
    name = getattr(declaration, "name", None)
    same_arity = [
        function
        for function in definitions
        if getattr(function, "name", None) == name
        and len(_parameters(function)) == len(declared)
    ]
    types = [_parameter_type(param) for param in declared]
    for function in same_arity:
        if [_parameter_type(param) for param in _parameters(function)] == types:
            return function
    return same_arity[0] if len(same_arity) == 1 else None


def parameter_default_suffix(converter, function_name, param_name, param, language):
    """The ``" = value"`` a CrossGL parameter declaration ends with, or ``""`` when the parameter has no default.

    ``converter`` renders the default expression and records refusals; ``language`` names the source language in
    the diagnostic.
    """
    default = parameter_default(param)
    if default is None:
        return ""
    if isinstance(default, UnparsedParameterDefault):
        converter.refuse_import(
            f"{language} default argument '{default.text}' of parameter '{param_name}' of '{function_name}': "
            "CrossTL cannot read it as an expression, and a call that omits the argument needs its value"
        )
        return ""
    return f" = {converter.visit(default)}"
