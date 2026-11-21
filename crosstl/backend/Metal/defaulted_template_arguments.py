"""Canonical argument lists for class-template specializations with defaulted parameters.

A template-id names the specialization of its argument list completed with the defaulted template arguments
([temp.arg.general]/9, [temp.type]/1), so ``Cell<float>`` and ``Cell<float, void>`` denote one type when the second
parameter defaults to ``void``. The importer names and materializes one record per argument list, so it must reduce
every spelling of a specialization to one canonical list first. Otherwise the two spellings become two records and
pointers that the source treats as one type no longer convert.

The canonical list is the shortest prefix of the written list whose completion equals the completion of the written
list. A spelling that omits its defaults is already canonical, so ordinary uses keep their names.

Partial specializations are selected on the completed list and already use it as their key, so this module covers
only the primary template.
"""

from __future__ import annotations

from typing import Any


def primary_template_structs_by_name(preprocessor: Any, source: str) -> dict[str, Any]:
    """The primary struct templates of ``source`` by qualified and, where unique, unqualified name.

    Template argument deduction reads a template's defaults from this table when an argument type spells a
    defaulted argument that the parameter type omits: ``Tile<T, M, N>`` deduces from
    ``Tile<float, 1, 1, Frag<float, 8, 8>>`` when the fourth argument equals the default ``Frag<T, 8, 8>`` completed
    with the deduced ``T`` ([temp.deduct.type]/9, [temp.arg.general]/9). Of two declarations of one name, the one
    that declares more defaults wins, as in the explicit-call deduction table.
    """
    by_name: dict[str, Any] = {}
    for declaration in preprocessor._find_template_structs(source):
        if (
            preprocessor._template_struct_specialization_arguments(declaration)
            is not None
        ):
            continue
        name = (
            f"{declaration.namespace}::{declaration.name}"
            if declaration.namespace
            else declaration.name
        )
        current = by_name.get(name)
        if current is None or (
            len(declaration.template_parameter_defaults),
            len(declaration.template_parameters),
        ) > (
            len(current.template_parameter_defaults),
            len(current.template_parameters),
        ):
            by_name[name] = declaration
    unqualified: dict[str, list[Any]] = {}
    for declaration in by_name.values():
        unqualified.setdefault(declaration.name, []).append(declaration)
    for name, declarations in sorted(unqualified.items()):
        if len(declarations) == 1:
            by_name.setdefault(name, declarations[0])
    return by_name


def canonical_struct_template_arguments(
    preprocessor: Any, template: Any, template_arguments: list[str]
) -> list[str]:
    """The canonical argument list of the primary-template specialization ``template<template_arguments>``.

    Returns ``template_arguments`` unchanged when its completion cannot be resolved, so an unresolvable default never
    merges two lists.
    """
    complete = preprocessor._template_arguments_with_resolved_defaults
    completed = complete(template, list(template_arguments))
    if completed is None:
        return list(template_arguments)
    canonical = list(template_arguments)
    while canonical:
        shorter = canonical[:-1]
        if complete(template, shorter) != completed:
            break
        canonical = shorter
    return canonical
