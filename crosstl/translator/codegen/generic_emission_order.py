"""Refuse a generic specialization that is created only after its generic function was emitted.

Generators collect the specializations of every generic function before they emit any code, but a call whose
argument types are only known while it is emitted (an entry-point resource, for instance) still creates its
specialization on demand. When that happens after the generator has already emitted the generic's specializations,
the call names a function that is never defined, so the output fails to compile.

A generator calls :func:`mark_generic_function_emitted` when it emits the specializations of a generic function, and
:func:`.generic_function_utils._ensure_generic_function_specialization` calls
:func:`reject_late_generic_specialization` before it creates one, which raises
:class:`GenericSpecializationOrderError` for a generic that was already emitted.
"""

from __future__ import annotations

__all__ = [
    "GenericSpecializationOrderError",
    "mark_generic_function_emitted",
    "reject_late_generic_specialization",
    "reset_generic_function_emission",
]


class GenericSpecializationOrderError(ValueError):
    """A generic call needed a specialization that was not collected before its generic was emitted."""

    project_diagnostic_code = "project.translate.generic-specialization-uncollected"
    missing_capabilities = ("generic.specialization-collection",)

    def __init__(self, function_name, type_arguments):
        arguments = ", ".join(str(argument) for argument in type_arguments)
        super().__init__(
            f"generic function '{function_name}' needs a specialization for ({arguments}) "
            "that was only resolved after its specializations were emitted, because the "
            "call's argument types are not known when specializations are collected"
        )
        self.function_name = function_name
        self.type_arguments = tuple(type_arguments)
        self.suggested_action = "call the generic function with explicit type arguments"


def reset_generic_function_emission(generator):
    """Forget which generic functions ``generator`` has emitted, at the start of a program."""
    generator.generic_function_emitted_source_ids = set()


def mark_generic_function_emitted(generator, func):
    """Record that ``generator`` has emitted the specializations of the generic ``func``."""
    emitted = getattr(generator, "generic_function_emitted_source_ids", None)
    if emitted is None:
        emitted = generator.generic_function_emitted_source_ids = set()
    emitted.add(id(func))


def reject_late_generic_specialization(generator, func, key):
    """Raise when the specialization ``key`` of ``func`` would be created after ``func`` was emitted."""
    emitted = getattr(generator, "generic_function_emitted_source_ids", None) or ()
    if id(func) in emitted:
        raise GenericSpecializationOrderError(key[0], key[1])
