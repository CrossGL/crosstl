"""Prove that a template taking a class argument exactly outranks a non-template overload that converts it.

Constrained overload resolution refuses a call when another visible declaration of the callee could compete with
the enabled template and ``overload_precedence.exact_template_outranks`` cannot rank the two. That proof needs every
argument to be a built-in scalar or vector. A template that overloads a library function for an aggregate, such as
``Pair<U> simd_shuffle_down(Pair<U> data, ushort delta)`` beside ``complex64_t simd_shuffle_down(complex64_t,
ushort)``, is ranked by its class argument instead ([over.match.best]/2.1, [over.ics.rank]/2.1):

* the template takes some argument of class type ``S`` by value as ``S`` itself, which is the identity conversion
  ([over.best.ics]/6), and ``S`` has no base class;
* the competitor takes that argument by value as another type: a built-in type or another concrete class. Without a
  base class, ``S`` reaches such a type only through a user-defined conversion, if at all ([over.best.ics]/6), and a
  standard conversion sequence is better than any user-defined one. A competitor with no conversion is not viable;
* every other argument is taken as the same type by both functions, so its conversion sequences are identical, or
  is taken exactly by the template, so the template's is the identity, which no sequence beats.

Then the template is the better function, so the non-template tie-breaker ([over.match.best]/2.4) never applies.
Types are compared after resolving struct aliases and materialized template-ids to the concrete struct they name.
Distinct concrete struct definitions are distinct types. Every other shape stays unproven, so the call is still
refused.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Tuple

from .overload_precedence import _builtin_value_type, _parameter_value_type
from .static_member_constants import materialized_struct_for_template_id

__all__ = ["class_identity_template_outranks"]

_BASE_CLAUSE_RE = re.compile(r"(?<!:):(?!:)")
_IGNORED_WORDS = frozenset({"const", "thread"})

TypeKey = Optional[Tuple[str, str]]


def _class_key(preprocessor: Any, context: Any, text: str, position: int) -> TypeKey:
    """The concrete struct a by-value type spelling names, through aliases and template-ids."""
    text = preprocessor._normalize_inferred_type(text)
    if "<" in text:
        concrete = materialized_struct_for_template_id(preprocessor, text)
        if concrete is not None and concrete in context.structs_by_name:
            return ("class", concrete)
        return None
    words = [word for word in text.split() if word not in _IGNORED_WORDS]
    if len(words) != 1:
        return None
    name = words[0]
    if name not in context.structs_by_name:
        name = preprocessor._resolve_struct_type_alias_at(
            context.struct_type_aliases, name, position
        )
    if name is None or name not in context.structs_by_name:
        return None
    return ("class", name)


def _parameter_key(
    preprocessor: Any, context: Any, parameter: str, position: int
) -> TypeKey:
    builtin = _parameter_value_type(preprocessor, parameter)
    if builtin is not None:
        return ("builtin", builtin)
    type_text = preprocessor._normalize_function_parameter_type_text(parameter)
    if not type_text or "&" in type_text or "*" in type_text or "[" in type_text:
        return None
    return _class_key(preprocessor, context, type_text, position)


def _argument_key(
    preprocessor: Any, context: Any, argument_type: str | None, position: int
) -> TypeKey:
    if not argument_type or "*" in argument_type or "[" in argument_type:
        return None
    builtin = _builtin_value_type(preprocessor, argument_type)
    if builtin is not None:
        return ("builtin", builtin)
    return _class_key(preprocessor, context, argument_type.replace("&", " "), position)


def _has_base_clause(preprocessor: Any, code: str, struct: Any) -> bool:
    header = preprocessor._mask_comments_and_literals(
        code[struct.span[0] : struct.body_span[0]]
    )
    return _BASE_CLAUSE_RE.search(header) is not None


def class_identity_template_outranks(
    preprocessor: Any,
    code: str,
    context: Any,
    template: Any,
    bindings: Any,
    argument_types: Sequence[str | None],
    competitor: Any,
    call_position: int,
) -> bool:
    """Whether the enabled ``template`` is better than the non-template ``competitor`` declaration for the call."""
    template_parameters = preprocessor._template_function_parameter_text(template)
    if template_parameters is None:
        return False
    parameters = [
        parameter
        for parameter in preprocessor._split_top_level_commas(template_parameters)
        if parameter.strip()
    ]
    competitor_parameters = list(competitor.arguments)
    if not (len(parameters) == len(competitor_parameters) == len(argument_types)):
        return False
    substitutions = {name: str(value) for name, value in dict(bindings).items()}
    template_keys = [
        _parameter_key(
            preprocessor,
            context,
            preprocessor._replace_identifiers(parameter, substitutions),
            template.span[0],
        )
        for parameter in parameters
    ]
    competitor_keys = [
        _parameter_key(preprocessor, context, parameter, competitor.span[0])
        for parameter in competitor_parameters
    ]
    argument_keys = [
        _argument_key(preprocessor, context, argument_type, call_position)
        for argument_type in argument_types
    ]
    decisive = [
        index
        for index, (argument, mine, theirs) in enumerate(
            zip(argument_keys, template_keys, competitor_keys)
        )
        if argument is not None
        and argument[0] == "class"
        and mine == argument
        and theirs is not None
        and theirs != argument
        and not _has_base_clause(
            preprocessor, code, context.structs_by_name[argument[1]]
        )
    ]
    if not decisive:
        return False
    return all(
        index in decisive or (mine is not None and (mine == theirs or mine == argument))
        for index, (argument, mine, theirs) in enumerate(
            zip(argument_keys, template_keys, competitor_keys)
        )
    )
