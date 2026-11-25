"""Running state while ``MetalCodeGen.render_rewritten_program`` declares the program-scope variables."""

from dataclasses import dataclass


@dataclass
class MetalGlobalEmission:
    """The declarations written so far and the bindings shared by all program-scope variables.

    ``used_resource_bindings`` and ``source_resource_bindings`` are the maps that
    ``reserve_or_remap_resource_binding`` reads and updates. A variable without a binding takes the lowest free
    slots of its Metal binding space (``lowest_free_resource_binding``).
    """

    code: str
    used_resource_bindings: dict
    source_resource_bindings: dict
