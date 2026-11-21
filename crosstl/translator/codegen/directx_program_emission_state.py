"""Running state of the global declaration loop in ``HLSLCodeGen._generate_program_attempt``."""

from dataclasses import dataclass


@dataclass
class DirectXGlobalEmission:
    """The text written so far while the global variables are declared, and the facts every global reads.

    ``texture_registers``, ``sampler_registers``, ``uav_registers`` and ``cbuffer_registers`` map binding spaces
    to the next free register, and ``used_resource_registers`` is the allocator that records explicit and
    assigned registers. The sampler name sets say which globals need a comparison, regular, implicit or
    ``CalculateLevelOfDetail`` sampler.
    """

    code: str
    comparison_sampler_names: set
    comparison_texture_names: set
    explicit_sampler_names: set
    global_implicit_sampler_texture_names: set
    global_implicit_regular_sampler_texture_names: set
    global_implicit_query_lod_texture_names: set
    texture_registers: dict
    sampler_registers: dict
    uav_registers: dict
    cbuffer_registers: dict
    used_resource_registers: object
