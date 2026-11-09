"""The specializations each function calls, recorded while generic specializations are collected.

A specialization's body still spells its calls by the generic name (``twice(y)`` in the ``quad<float>`` clone of
``generic<T> T quad(T x) { return twice(twice(x)); }``); a generator resolves them to ``twice_float`` only while it
emits the body, with the call's argument types in scope. A generator that orders functions so callees come first,
as the SPIR-V generator does because it lowers a call only to a function it has already emitted, therefore saw no
dependency of ``quad_float`` on ``twice_float`` and refused the call when ``twice`` was declared after ``quad``.

:func:`record_generic_callee` notes every specialization a call resolves to while
:func:`.generic_function_utils.prepare_generic_function_specializations` analyzes a function, and
:func:`generic_callee_names` returns the names of those specializations for the function, in call order.
"""

from __future__ import annotations

__all__ = [
    "begin_generic_callee_recording",
    "end_generic_callee_recording",
    "generic_callee_names",
    "record_generic_callee",
    "reset_generic_callees",
]


def reset_generic_callees(generator):
    """Forget the recorded callees, at the start of a program's collection."""
    generator.generic_function_callee_keys = {}
    generator.generic_collection_callees = None


def begin_generic_callee_recording(generator):
    """Start recording the callees of one function; returns the state :func:`end_generic_callee_recording` restores."""
    previous = getattr(generator, "generic_collection_callees", None)
    generator.generic_collection_callees = []
    return previous


def end_generic_callee_recording(generator, func, previous):
    """Store the callees recorded for ``func`` and restore the enclosing recording."""
    callees = getattr(generator, "generic_collection_callees", None) or []
    recorded = getattr(generator, "generic_function_callee_keys", None)
    if recorded is None:
        recorded = generator.generic_function_callee_keys = {}
    keys = recorded.setdefault(id(func), [])
    for key in callees:
        if key not in keys:
            keys.append(key)
    generator.generic_collection_callees = previous


def record_generic_callee(generator, key):
    """Note that the function being analyzed calls the specialization ``key``."""
    callees = getattr(generator, "generic_collection_callees", None)
    if callees is not None:
        callees.append(key)


def generic_callee_names(generator, func):
    """The names of the specializations ``func`` calls, in the order its calls were analyzed."""
    keys = (getattr(generator, "generic_function_callee_keys", None) or {}).get(
        id(func), ()
    )
    names = getattr(generator, "generic_function_specialized_names", None) or {}
    return [names[key] for key in keys if key in names]
