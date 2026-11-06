"""The specialization constants through which a pipeline sets a workgroup size that the source's host chooses.

A Metal kernel whose result depends on its threadgroup size, but whose source gives none, is sized by the host at
dispatch time (:mod:`crosstl.translator.host_workgroup_size`). Vulkan and OpenGL with SPIR-V shaders set the size
when they create the pipeline instead, through three specialization constants: GLSL ``local_size_{x,y,z}_id``
(GLSL 4.60 section 4.4.1.4) and SPIR-V ``LocalSizeId`` or a ``WorkgroupSize`` composite of specialization constants
(SPIR-V 1.6 section 3.21). :func:`host_workgroup_size_ids` allocates their ids from the CrossGL program, so the SPIR-V
generator and the GLSL generator, whose output glslang compiles to SPIR-V, give the host the same ids: the three
smallest that no specialization constant and no specialized workgroup dimension of the program uses.

A driver that compiles GLSL source has no specialization step. The generated source then reads the size from the
:data:`HOST_WORKGROUP_SIZE_MACROS`, which the host defines after ``#version``, the way OpenGL programs select shader
variants, and stops at an ``#error`` without them.
"""

from __future__ import annotations

from typing import Any

from ..host_workgroup_size import (
    HOST_WORKGROUP_SIZE_KEY,
    function_entry_execution_config,
    has_explicit_workgroup_size,
)
from .specialization_constants import walk_nodes
from .specialized_workgroup_size import (
    SPECIALIZED_WORKGROUP_SIZE_KEYS,
    has_specialized_workgroup_size,
)

__all__ = [
    "HOST_WORKGROUP_SIZE_CONSTANT_NAMES",
    "HOST_WORKGROUP_SIZE_MACROS",
    "allocate_host_workgroup_size_ids",
    "host_sized_workgroup_config",
    "host_workgroup_size_ids",
    "is_host_sized_workgroup",
    "program_specialization_ids",
]

HOST_WORKGROUP_SIZE_MACROS = (
    "CROSSGL_WORKGROUP_SIZE_X",
    "CROSSGL_WORKGROUP_SIZE_Y",
    "CROSSGL_WORKGROUP_SIZE_Z",
)

# The names of the specialization constants of the x, y and z dimensions, which reflection recognizes them by.
HOST_WORKGROUP_SIZE_CONSTANT_NAMES = (
    "crossgl_workgroup_size_x",
    "crossgl_workgroup_size_y",
    "crossgl_workgroup_size_z",
)

# Every attribute spelling of a specialization id that a CrossTL generator reads, so that the host's ids collide
# with none of them on any target.
_SPECIALIZATION_ID_ATTRIBUTES = frozenset(
    {"constant_id", "spec_id", "specialization_constant", "function_constant"}
)
_ATTRIBUTE_PREFIXES = ("glsl_", "metal_", "msl_", "spirv_", "vulkan_", "vk_")


def is_host_sized_workgroup(config: Any) -> bool:
    """Whether execution metadata leaves an entry's workgroup size to the host, with no size or specialization."""
    config = config or {}
    return bool(config.get(HOST_WORKGROUP_SIZE_KEY)) and not (
        has_explicit_workgroup_size(config) or has_specialized_workgroup_size(config)
    )


def _attribute_name(attribute: Any) -> str:
    name = str(getattr(attribute, "name", attribute) or "").lower().replace("-", "_")
    for prefix in _ATTRIBUTE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _integer(value: Any) -> int | None:
    raw = getattr(value, "value", value)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, str):
        text = raw.strip().rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


def _declared_ids(node: Any) -> set[int]:
    ids = set()
    for attribute in getattr(node, "attributes", None) or []:
        if _attribute_name(attribute) not in _SPECIALIZATION_ID_ATTRIBUTES:
            continue
        arguments = getattr(attribute, "arguments", getattr(attribute, "args", None))
        for argument in arguments or []:
            spec_id = _integer(argument)
            if spec_id is not None:
                ids.add(spec_id)
    return ids


def _entry_configs(shader: Any) -> list:
    configs = [
        getattr(stage, "execution_config", None)
        for stage in (getattr(shader, "stages", None) or {}).values()
    ]
    configs += [
        function_entry_execution_config(function)
        for function in getattr(shader, "functions", None) or []
    ]
    return [config for config in configs if config]


def program_specialization_ids(shader: Any) -> set[int]:
    """Every specialization id that a declaration or a specialized workgroup dimension of ``shader`` names.

    A malformed id is left out; the generator that reads its declaration refuses it.
    """
    used: set[int] = set()
    for node in walk_nodes(shader):
        if getattr(node, "attributes", None):
            used |= _declared_ids(node)
    for config in _entry_configs(shader):
        for key in SPECIALIZED_WORKGROUP_SIZE_KEYS:
            spec_id = _integer(config.get(key))
            if spec_id is not None:
                used.add(spec_id)
    return used


def allocate_host_workgroup_size_ids(used) -> tuple[int, int, int]:
    """The three smallest non-negative ids not in ``used``, for the x, y and z dimensions."""
    used = set(used)
    allocated: list[int] = []
    candidate = 0
    while len(allocated) < 3:
        if candidate not in used:
            allocated.append(candidate)
        candidate += 1
    return allocated[0], allocated[1], allocated[2]


def host_workgroup_size_ids(shader: Any) -> tuple[int, int, int]:
    """The specialization ids of the x, y and z dimensions of every host-sized entry of ``shader``."""
    return allocate_host_workgroup_size_ids(program_specialization_ids(shader))


def host_sized_workgroup_config(config: Any, ids) -> dict:
    """``config`` with the workgroup size specialized through ``ids`` when the host chooses it; else ``config``."""
    config = dict(config or {})
    if not is_host_sized_workgroup(config):
        return config
    return {**config, **dict(zip(SPECIALIZED_WORKGROUP_SIZE_KEYS, ids))}
