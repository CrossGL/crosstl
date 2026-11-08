"""WGSL ``override`` constants for the workgroup dimensions a compute stage specializes.

A CrossGL ``layout(local_size_x_id = N) in;`` lets the pipeline set the x dimension through the specialization
constant ``N``. WGSL expresses that with a pipeline-overridable constant, ``@id(N) override name: u32 = default;``,
named in ``@workgroup_size`` (WGSL section 9.3.6, ``workgroup_size``; section 7.2.3, override declarations), which the
WebGPU pipeline sets from ``GPUProgrammableStage.constants``. ``gl_WorkGroupSize`` reads the same constants.

A Metal kernel whose size the source's host chooses at dispatch time (:mod:`..host_workgroup_size`) is specialized
through the ids of :mod:`.host_workgroup_size_ids`, the ones the SPIR-V and GLSL generators give the host, by
overrides without an initializer: WebGPU then refuses to create a pipeline whose ``constants`` leave one unset
(WebGPU section 10.1.2, ``GPUProgrammableStage``), instead of running groups of a size nobody chose.

The overrides are module-scope declarations, so the entries of one module share them: two entries that specialize a
dimension through different ids, or one id with different defaults, are refused, as are two dimensions of one id.
WGSL pipeline constant ids are below 65536 (WGSL section 12.3.1.3, ``id``), so a larger id is refused.
"""

from __future__ import annotations

import re

from .host_workgroup_size_ids import (
    host_sized_workgroup_config,
    host_workgroup_size_ids,
    is_host_sized_workgroup,
)
from .specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
    workgroup_size_specialization_ids,
)

__all__ = ["WorkgroupSizeOverrides", "specialized_workgroup_size"]

_MAX_WGSL_ID = 65535


def _override_name(axis: str) -> str:
    return f"crossgl_workgroup_size_{axis}"


def _u32_default(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+", text):
        return f"{text}u"
    return f"u32({text})"


def _declaration(name: str, spec_id: int, value) -> str:
    if value is None:
        return f"@id({spec_id})\noverride {name}: u32;"
    return f"@id({spec_id})\noverride {name}: u32 = {_u32_default(value)};"


def specialized_workgroup_size(config, values):
    """The ``@workgroup_size`` arguments and the override declarations that specialize them.

    ``values`` are the three default dimensions; a dimension the stage names an id for becomes its override, and a
    default of ``None`` leaves the override without an initializer.
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
        declarations.append(_declaration(name, spec_id, value))
    return tuple(arguments), declarations


class WorkgroupSizeOverrides:
    """The workgroup-size overrides of one WGSL module, declared once and shared by its entries."""

    def __init__(self, program):
        self._program = program
        self._host_ids = None
        self._declared: dict[str, str] = {}
        self._names_by_id: dict[int, str] = {}

    def effective_config(self, config):
        """``config``, specialized through the program's host-size ids when the host chooses the size."""
        config = config or {}
        if not is_host_sized_workgroup(config):
            return config
        if self._host_ids is None:
            self._host_ids = host_workgroup_size_ids(self._program)
        return host_sized_workgroup_config(config, self._host_ids)

    def new_declarations(self, config, values):
        """The declarations of ``config``'s overrides that the module does not declare yet."""
        _, declarations = specialized_workgroup_size(config, values)
        specialized = [
            (_override_name(axis), spec_id)
            for axis, spec_id in zip(
                "xyz", workgroup_size_specialization_ids(config, "WGSL")
            )
            if spec_id is not None
        ]
        fresh = []
        for (name, spec_id), declaration in zip(specialized, declarations):
            owner = self._names_by_id.get(spec_id)
            previous = self._declared.get(name) or self._declared.get(owner)
            if previous == declaration and owner == name:
                continue
            if previous is not None:
                raise SpecializedWorkgroupSizeError(
                    "Entries of one WGSL module specialize their workgroup sizes through module-scope overrides "
                    f"that disagree: '{_one_line(previous)}' and '{_one_line(declaration)}'",
                    target="WGSL",
                    specialization_ids=(spec_id,),
                    reason="conflicting-workgroup-size-override",
                )
            self._declared[name] = declaration
            self._names_by_id[spec_id] = name
            fresh.append(declaration)
        return fresh


def _one_line(declaration: str) -> str:
    return declaration.replace("\n", " ")
