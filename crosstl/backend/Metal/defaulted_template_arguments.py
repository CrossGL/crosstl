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
