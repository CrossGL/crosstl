"""Explicit specializations of CrossGL generic functions.

A concrete function marked ``@explicit_specialization(twice, float)`` replaces the body of the generic ``twice``
for ``T = float``, as a C++ explicit specialization ``template <> float twice<float>(float)`` does
([temp.expl.spec]/1). Front ends that import such sources keep the generic primary and emit each specialization as
a concrete function carrying the attribute; the attribute's arguments are the primary's name and its template
arguments in order, spelled as CrossGL types.

Generators that monomorphize generics instantiate the primary for each concrete argument tuple a call reaches,
including calls inside other generic bodies that only become concrete when those are instantiated. When an
instantiation's arguments equal a specialization's, the call names the specialization instead of a clone of the
primary's body. Generators that emit native generics cannot select a body per argument tuple and refuse the
program (:func:`explicit_specialization_refusals`).
"""

from __future__ import annotations

from .ast import FunctionNode, IdentifierNode, LiteralNode
from .refusals import CrossGLSyntaxError

__all__ = [
    "EXPLICIT_SPECIALIZATION_ATTRIBUTE",
    "explicit_specialization_annotation",
    "explicit_specialization_for",
    "explicit_specialization_index",
    "explicit_specialization_refusals",
    "explicit_specializations",
]

EXPLICIT_SPECIALIZATION_ATTRIBUTE = "explicit_specialization"


def _argument_text(argument):
    if isinstance(argument, IdentifierNode):
        return str(argument.name)
    if isinstance(argument, LiteralNode):
        return str(argument.value)
    name = getattr(argument, "name", None)
    return str(name) if name is not None else None


def explicit_specialization_annotation(attribute):
    """``(primary name, template argument spellings)`` of an ``@explicit_specialization`` attribute.

    The parser keeps the pair in the function's ``annotations`` so generators never print the attribute.
    """
    texts = [
        _argument_text(argument)
        for argument in getattr(attribute, "arguments", None) or []
    ]
    if len(texts) < 2 or None in texts:
        raise CrossGLSyntaxError(
            "@explicit_specialization requires the generic function name and its "
            "template arguments"
        )
    return texts[0], tuple(texts[1:])


def explicit_specializations(functions):
    """``(function, primary name, template argument spellings)`` for each marked concrete function."""
    found = []
    for function in functions or []:
        if not isinstance(function, FunctionNode):
            continue
        if getattr(function, "generic_params", None):
            continue
        annotations = getattr(function, "annotations", None) or {}
        marked = annotations.get(EXPLICIT_SPECIALIZATION_ATTRIBUTE)
        if marked is not None:
            found.append((function, marked[0], tuple(marked[1])))
    return found


def _canonical_vector_spelling(type_name):
    """One spelling of a vector type for every generator.

    Instantiation keys carry the types a generator renders for deduced arguments (``float4`` for ``vec4`` on Metal,
    ``v4float`` on SPIR-V) but CrossGL spellings for explicit ones, and the attribute spells CrossGL types. Vectors
    are compared as ``vecN<element>``. Scalars keep their own spelling: a generator's type mapping may widen them
    (Metal maps ``uint8`` to ``uint``), which would merge distinct specializations.
    """
    from .codegen.generic_vector_deduction import vector_element_and_size

    parts = vector_element_and_size(type_name)
    if parts is None:
        return type_name
    element, size = parts
    return f"vec{size}<{element}>"


def _normalized(generator, type_name):
    normalize = getattr(generator, "normalize_signature_type_name", None)
    if normalize is not None:
        return normalize(type_name)
    try:
        rendered = generator.type_name_string(type_name)
    except (TypeError, ValueError):
        return None
    if rendered is None:
        return None
    return _canonical_vector_spelling(str(rendered).replace(" ", ""))


def explicit_specialization_index(generator, functions):
    """Map ``(primary name, normalized template arguments)`` to the specialization that replaces it."""
    index = {}
    for function, primary, arguments in explicit_specializations(functions):
        normalized = tuple(_normalized(generator, argument) for argument in arguments)
        if None in normalized:
            continue
        index[(primary, normalized)] = function
    return index


def explicit_specialization_for(generator, key):
    """The specialization an instantiation ``key`` of the monomorphizer selects, or ``None``.

    A key with an overload ordinal names one of several generic primaries of the same name; the attribute does
    not say which, so such instantiations keep the primary's body.
    """
    index = getattr(generator, "explicit_generic_specializations", None)
    if not index or len(key) != 2:
        return None
    normalized = tuple(_normalized(generator, argument) for argument in key[1])
    if None in normalized:
        return None
    return index.get((key[0], normalized))


def explicit_specialization_refusals(generator, ast, label):
    """Refuse each specialization of a generic primary on a target that emits native generics."""
    functions = list(getattr(ast, "functions", None) or [])
    generic_names = {
        function.name
        for function in functions
        if isinstance(function, FunctionNode)
        and getattr(function, "generic_params", None)
    }
    for function, primary, arguments in explicit_specializations(functions):
        if primary not in generic_names:
            continue
        generator.refuse_lowering(
            f"explicit specialization {function.name} of generic function {primary}<{', '.join(arguments)}>: "
            f"{label} generics cannot select a specialized body per template argument"
        )
