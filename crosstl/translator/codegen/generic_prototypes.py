"""Generic function prototypes: forward declarations and declaration-only primaries.

``generic<T> T f(T x);`` declares a generic function without defining it. When the program also defines ``f``
with the same number of parameters, the prototype is a forward declaration of that definition, as a C++ function
template declaration is ([temp.over.link]/1). Generic collection treated it as a second overload instead, so calls
bound to a body-less clone (``float f_float_overload0(float x);``) that nothing defined, and an explicit
specialization of ``f`` was ignored because overloaded generics do not select one.

Without a definition the prototype is a declaration-only primary: explicit specializations supply the bodies for
their arguments ([temp.expl.spec]/1) and any other instantiation has no body to clone. Such an instantiation is
refused with :class:`GenericDefinitionMissingError` instead of emitting a prototype that nothing defines.
"""

from __future__ import annotations

__all__ = [
    "GenericDefinitionMissingError",
    "declaration_only_generic_primaries",
    "reject_undefined_generic_instantiation",
    "without_forward_generic_prototypes",
]


class GenericDefinitionMissingError(ValueError):
    """A generic function declared without a body was instantiated for arguments no specialization covers."""

    project_diagnostic_code = "project.translate.generic-definition-missing"
    missing_capabilities = ("generic.definition",)

    def __init__(self, function_name, type_arguments):
        arguments = ", ".join(str(argument) for argument in type_arguments)
        super().__init__(
            f"generic function '{function_name}' is declared without a definition and "
            f"has no explicit specialization for ({arguments}); there is no body to "
            "instantiate"
        )
        self.function_name = function_name
        self.type_arguments = tuple(type_arguments)
        self.suggested_action = (
            "define the generic function or add an explicit specialization for "
            "these arguments"
        )


def _parameter_count(func):
    return len(getattr(func, "parameters", getattr(func, "params", None)) or [])


def _defined_generic_arities(functions, is_generic):
    arities = set()
    for func in functions or []:
        if getattr(func, "body", None) is not None and is_generic(func):
            arities.add((getattr(func, "name", None), _parameter_count(func)))
    return arities


def without_forward_generic_prototypes(functions, is_generic):
    """``functions`` without the generic prototypes that a generic definition of theirs completes."""
    functions = list(functions or [])
    defined = _defined_generic_arities(functions, is_generic)
    return [
        func
        for func in functions
        if getattr(func, "body", None) is not None
        or not is_generic(func)
        or (getattr(func, "name", None), _parameter_count(func)) not in defined
    ]


def declaration_only_generic_primaries(functions, is_generic):
    """The body-less generic functions of ``functions`` that no generic definition completes."""
    return [
        func
        for func in without_forward_generic_prototypes(functions, is_generic)
        if getattr(func, "body", None) is None and is_generic(func)
    ]


def reject_undefined_generic_instantiation(func, key):
    """Raise when instantiating ``func`` for ``key`` would clone a missing body."""
    if getattr(func, "body", None) is None:
        raise GenericDefinitionMissingError(key[0], key[1])
