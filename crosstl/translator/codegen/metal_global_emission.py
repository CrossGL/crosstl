"""Running state while ``MetalCodeGen.render_rewritten_program`` declares the program-scope variables."""

from dataclasses import dataclass


@dataclass
class MetalGlobalEmission:
    """The declarations written so far and the binding cursors shared by all program-scope variables.

    ``texture_register``, ``sampler_register`` and ``buffer_register`` are one past the highest binding index used
    so far in each Metal binding space. ``used_resource_bindings`` and ``source_resource_bindings`` are the maps
    that ``reserve_or_remap_resource_binding`` reads and updates.
    """

    code: str
    texture_register: int
    sampler_register: int
    buffer_register: int
    used_resource_bindings: dict
    source_resource_bindings: dict
