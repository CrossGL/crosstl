"""Rank a non-template member function against competing member templates.

[over.match.best]/2.4: when no argument converts better for either candidate,
a non-template function is better than a function template specialization.
An argument whose parameter has the same non-dependent declared type in the
non-template and in every competing template converts identically for all of
them, so that conversion cannot distinguish the candidates. Only the remaining
arguments need an exact match before the non-template is preferred.
"""

import re


def shared_template_parameter_positions(
    preprocessor, method, template_overloads, argument_count
):
    """Return argument positions whose conversion every candidate shares."""

    if not template_overloads or method.template_parameters:
        return None
    concrete = _parameter_types(preprocessor, method, argument_count)
    if concrete is None:
        return None
    shared = set(range(argument_count))
    for template in template_overloads:
        types = _parameter_types(preprocessor, template, argument_count)
        if types is None:
            return None
        names = set(template.template_parameters) | set(
            template.variadic_template_parameters
        )
        for position in sorted(shared):
            text = types[position]
            dependent = any(
                re.search(rf"\b{re.escape(name)}\b", text) for name in names
            )
            if dependent or text != concrete[position]:
                shared.discard(position)
    return shared or None


def _parameter_types(preprocessor, method, count):
    parameters = [
        parameter
        for parameter in preprocessor._split_top_level_commas(method.parameters)
        if parameter.strip() and parameter.strip() != "void"
    ]
    fixed = [parameter for parameter in parameters if "..." not in parameter]
    if len(fixed) < count:
        return None
    return [
        preprocessor._normalize_inferred_type(
            preprocessor._normalize_function_parameter_type_text(parameter)
        )
        for parameter in fixed[:count]
    ]
