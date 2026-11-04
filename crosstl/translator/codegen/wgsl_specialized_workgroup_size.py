"""WGSL ``override`` constants for the workgroup dimensions a compute stage specializes.

A CrossGL ``layout(local_size_x_id = N) in;`` lets the pipeline set the x dimension through the specialization
constant ``N``. WGSL expresses that with a pipeline-overridable constant, ``@id(N) override name: u32 = default;``,
named in ``@workgroup_size`` (WGSL section 9.3.6, ``workgroup_size``; section 7.2.3, override declarations), which the
WebGPU pipeline sets from ``GPUProgrammableStage.constants``. ``gl_WorkGroupSize`` reads the same constants.

WGSL pipeline constant ids are below 65536 (WGSL section 12.3.1.3, ``id``), so a larger id is refused.
"""

from __future__ import annotations

import re

from .specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
    workgroup_size_specialization_ids,
)

__all__ = ["specialized_workgroup_size"]

_MAX_WGSL_ID = 65535


def _override_name(axis: str) -> str:
    return f"crossgl_workgroup_size_{axis}"


def _u32_default(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+", text):
        return f"{text}u"
    return f"u32({text})"


def specialized_workgroup_size(config, values):
    """The ``@workgroup_size`` arguments and the override declarations that specialize them.

    ``values`` are the three default dimensions; a dimension the stage names an id for becomes its override.
    """
    ids = workgroup_size_specialization_ids(config, "WGSL")
    arguments = []
    declarations = []
    for axis, value, spec_id in zip("xyz", values, ids):
        if spec_id is None:
            arguments.append(value)
            continue
        if spec_id > _MAX_WGSL_ID:
            raise SpecializedWorkgroupSizeError(
                f"WGSL pipeline constant ids are below 65536; local_size_{axis}_id = {spec_id} is not",
                target="WGSL",
                specialization_ids=(spec_id,),
                reason="specialization-id-out-of-range",
            )
        name = _override_name(axis)
        arguments.append(name)
        declarations.append(
            f"@id({spec_id})\noverride {name}: u32 = {_u32_default(value)};"
        )
    return tuple(arguments), declarations
