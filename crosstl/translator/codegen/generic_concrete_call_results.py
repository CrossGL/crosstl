"""Declared result types of concrete helper calls, for generic-call deduction.

In ``twice(first(v))`` the generic call deduces ``T`` from the result of the concrete helper ``first``.
Generators that register function signatures only while they emit code (SPIR-V) cannot type that call
while specializations are collected, so the generic call stayed unresolved and was refused even though
the helper's declared return type fixes ``T``.

A call to a concrete, non-generic function of the program is typed here by the declared return type of
the overload it selects. Overloads are narrowed by arity (trailing parameters with defaults may be
omitted) and then by the argument types the caller can infer. When the remaining overloads disagree on
their return type the call stays untyped, so deduction fails closed instead of guessing.
"""

from __future__ import annotations

__all__ = [
    "concrete_call_overload",
    "concrete_call_result_type_name",
    "concrete_overload_results_differ",
    "record_concrete_function_overloads",
]


def _parameters(func):
    return list(getattr(func, "parameters", getattr(func, "params", [])) or [])


def _call_name(call):
    callee = getattr(call, "function", getattr(call, "name", None))
    name = getattr(callee, "name", callee)
    return name if isinstance(name, str) else None


def _call_arguments(call):
    return list(getattr(call, "arguments", getattr(call, "args", [])) or [])


def record_concrete_function_overloads(generator, functions, is_generic):
    """Remember the concrete functions of the program by name, in source order."""
    overloads = {}
    for func in functions or []:
        name = getattr(func, "name", None)
        if not isinstance(name, str) or not name or is_generic(func):
            continue
        overloads.setdefault(name, []).append(func)
    generator.concrete_function_overloads = overloads


def _accepts_arity(func, count):
    parameters = _parameters(func)
    if count > len(parameters):
        return False
    return all(
        getattr(param, "default_value", None) is not None
        for param in parameters[count:]
    )


def _parameter_type(param):
    return getattr(param, "param_type", getattr(param, "vtype", None))


def _matches_known_arguments(generator, func, arg_types, normalize):
    for param, arg_type in zip(_parameters(func), arg_types):
        if not arg_type:
            continue
        expected = normalize(
            generator, generator.type_name_string(_parameter_type(param))
        )
        if expected and expected != normalize(generator, arg_type):
            return False
    return True


def _result_type_name(generator, func, normalize):
    return_type = generator.type_name_string(getattr(func, "return_type", None))
    if not return_type or normalize(generator, return_type) == "void":
        return None
    return return_type


def concrete_call_overload(generator, call, argument_type, normalize):
    """The concrete function ``call`` selects, when its declared result type is unambiguous.

    ``argument_type(generator, arg)`` infers an argument's type name (or ``None``) and
    ``normalize(generator, type_name)`` spells a type name for comparison. When several
    overloads remain they must agree on their result type; the first of them is returned.
    """
    name = _call_name(call)
    if not name:
        return None
    if name in (getattr(generator, "generic_function_definitions", None) or {}):
        # Generic overloads of the name are deduced by the generic-call machinery.
        return None
    overloads = (getattr(generator, "concrete_function_overloads", None) or {}).get(
        name
    ) or []
    args = _call_arguments(call)
    candidates = [func for func in overloads if _accepts_arity(func, len(args))]
    if len(candidates) > 1:
        arg_types = [argument_type(generator, arg) for arg in args]
        candidates = [
            func
            for func in candidates
            if _matches_known_arguments(generator, func, arg_types, normalize)
        ]
    result_types = set()
    for func in candidates:
        result_type = _result_type_name(generator, func, normalize)
        if result_type is None:
            return None
        result_types.add(normalize(generator, result_type))
    if len(result_types) != 1:
        return None
    return candidates[0]


def concrete_call_result_type_name(generator, call, argument_type, normalize):
    """The declared result type of ``call`` when it names a concrete helper, else ``None``."""
    func = concrete_call_overload(generator, call, argument_type, normalize)
    if func is None:
        return None
    return _result_type_name(generator, func, normalize)


def concrete_overload_results_differ(generator, name, normalize):
    """Whether concrete overloads of ``name`` declare different result types.

    A generator that types calls by function name alone must not pick one of them.
    """
    overloads = (getattr(generator, "concrete_function_overloads", None) or {}).get(
        name
    ) or []
    spellings = {
        normalize(generator, generator.type_name_string(func.return_type))
        for func in overloads
    }
    return len(spellings) > 1
