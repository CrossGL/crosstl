"""Interface locations of each SPIR-V entry point.

Every ``OpEntryPoint`` lists its own ``Input`` and ``Output`` variables, so two entry points of one stage may both
put a variable at ``Location 0``; only the variables of one interface must not overlap (SPIR-V 1.6 section 2.16.2,
Vulkan "Location Assignment"). A Metal library often has several fragment functions that return the same colour
struct. The generator tracked locations per execution model across the module, so the second such function was
refused with "Duplicate SPIR-V output location 0". :func:`interface_location_state` records the locations taken
before an entry point's own interface variables are declared, and :func:`restore_interface_location_state` returns
to them after it, so each entry point starts from the variables the stage shares.
"""

from __future__ import annotations

__all__ = ["interface_location_state", "restore_interface_location_state"]


def interface_location_state(codegen):
    """The locations and location counters taken so far."""
    return (
        set(codegen.used_input_locations),
        set(codegen.used_output_locations),
        dict(codegen.interface_location_counters),
    )


def restore_interface_location_state(codegen, state):
    """Return to the locations an entry point started from."""
    inputs, outputs, counters = state
    codegen.used_input_locations = set(inputs)
    codegen.used_output_locations = set(outputs)
    codegen.interface_location_counters = dict(counters)
