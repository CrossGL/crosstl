"""Slang ``[numthreads]`` of compute, mesh and amplification entries, including sizes set after generation.

Slang fixes a workgroup's size in the shader with ``[numthreads(X, Y, Z)]``. An operand may name a
``[vk::constant_id(N)]`` constant instead of a literal, which slangc lowers to SPIR-V ``LocalSizeId`` and to Metal
function constants, so the pipeline sets that dimension. Targets without specialization constants, such as HLSL,
refuse such an operand (slangc error E55205), so a specialized size fails closed there.

The constants are declared once, at module scope, in front of every entry that names them. Two entries that
specialize a dimension the same way share its constant; a different id or default gets a constant of its own, so
no name is declared twice.

A Metal kernel whose result depends on its threadgroup size, but whose source gives none, is sized by the host
when it dispatches it (:mod:`crosstl.translator.host_workgroup_size`). Its ``numthreads`` reads the
:data:`~.host_workgroup_size_ids.HOST_WORKGROUP_SIZE_MACROS` when the host defines them on the slangc command line
(``-D CROSSGL_WORKGROUP_SIZE_X=64``), as the generated HLSL and GLSL do, for any Slang target. Otherwise it names
three specialization constants whose ids are :func:`~.host_workgroup_size_ids.host_workgroup_size_ids` of the
program, the ids the generated GLSL and SPIR-V give the host, so a Vulkan or Metal pipeline specializes the size
and an HLSL compile without the macros stops at slangc's refusal. The constants are only declared when the macros
are not all defined, so an HLSL compile with the macros declares no implicit global shader parameters.
"""

from __future__ import annotations

from typing import Any

from .host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_CONSTANT_NAMES,
    HOST_WORKGROUP_SIZE_MACROS,
    host_workgroup_size_ids,
    is_host_sized_workgroup,
)
from .specialized_workgroup_size import workgroup_size_specialization_ids
from .stage_utils import compute_local_size

__all__ = ["SlangWorkgroupSizeConstants", "slang_host_size_macros_defined"]

_AXES = "xyz"


def slang_host_size_macros_defined() -> str:
    """The preprocessor condition under which the host has defined every size macro."""
    return " && ".join(f"defined({macro})" for macro in HOST_WORKGROUP_SIZE_MACROS)


class SlangWorkgroupSizeConstants:
    """The ``numthreads`` of each entry of one Slang module, and the specialization constants they name."""

    def __init__(self, program: Any):
        self.program = program
        # name -> (spec_id, default, host-sized); insertion order is generation order, so output is deterministic.
        self.constants: dict[str, tuple[int, str, bool]] = {}
        self._host_ids: tuple[int, int, int] | None = None

    def numthreads(self, execution_config: Any = None) -> str:
        """The ``[numthreads]`` attribute of an entry, with any preprocessor lines it needs."""
        if is_host_sized_workgroup(execution_config):
            return self._host_numthreads()
        sizes = list(compute_local_size(execution_config))
        ids = workgroup_size_specialization_ids(execution_config, "Slang")
        for index, spec_id in enumerate(ids):
            if spec_id is not None:
                sizes[index] = self._constant(
                    f"crossgl_workgroup_size_{_AXES[index]}",
                    spec_id,
                    sizes[index],
                    host=False,
                )
        return f"[numthreads({', '.join(map(str, sizes))})]\n"

    def _host_numthreads(self) -> str:
        if self._host_ids is None:
            self._host_ids = host_workgroup_size_ids(self.program)
        names = [
            self._constant(name, spec_id, "1", host=True)
            for name, spec_id in zip(HOST_WORKGROUP_SIZE_CONSTANT_NAMES, self._host_ids)
        ]
        return (
            f"#if {slang_host_size_macros_defined()}\n"
            f"[numthreads({', '.join(HOST_WORKGROUP_SIZE_MACROS)})]\n"
            "#else\n"
            f"[numthreads({', '.join(names)})]\n"
            "#endif\n"
        )

    def _constant(self, base: str, spec_id: int, default: Any, host: bool) -> str:
        """The name of the constant with this id and default, declaring it on first use."""
        key = (int(spec_id), str(default), host)
        for name, declared in self.constants.items():
            if declared == key:
                return name
        name, suffix = base, 1
        while name in self.constants:
            name, suffix = f"{base}_{suffix}", suffix + 1
        self.constants[name] = key
        return name

    def declarations(self) -> str:
        """Module-scope declarations of every constant an entry named, host-sized ones behind the macro check.

        Each is an ``int``: ``WorkgroupSize()`` is an ``int3``, and a ``uint`` constant makes slangc build an
        invalid composite.
        """
        specialized, host = [], []
        for name, (spec_id, default, host_sized) in self.constants.items():
            line = f"[vk::constant_id({spec_id})] const int {name} = {default};\n"
            (host if host_sized else specialized).append(line)
        code = "".join(specialized)
        if host:
            code += (
                f"#if !({slang_host_size_macros_defined()})\n"
                + "".join(host)
                + "#endif\n"
            )
        return code + "\n" if code else ""
