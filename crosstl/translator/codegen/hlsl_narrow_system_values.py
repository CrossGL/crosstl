"""16-bit unsigned compute thread-index parameters on HLSL entry points.

Metal lets a kernel declare its thread-index attributes with 16-bit types: ``[[thread_position_in_grid]]``,
``[[thread_position_in_threadgroup]]``, ``[[threadgroup_position_in_grid]]`` and ``[[threads_per_threadgroup]]``
take ``ushort``, ``ushort2`` or ``ushort3`` as well as the ``uint`` forms, and ``[[thread_index_in_threadgroup]]``
takes ``ushort`` or ``uint`` (Metal Shading Language Specification, section 5.2.3.6, "Kernel Function Input
Attributes"). CrossGL keeps the declared width (``u16vec3 tpitg @gl_LocalInvocationID``), which the HLSL emitter maps
to ``min16uint3``.

HLSL's compute system values ``SV_GroupID``, ``SV_GroupThreadID`` and ``SV_DispatchThreadID`` are ``uint3`` (or a
narrower ``uint`` vector), and ``SV_GroupIndex`` is ``uint``; DXC rejects a 16-bit type on them. The entry therefore
receives the 32-bit system value under a fresh name, and a body-local of the declared 16-bit type is initialised from
it, so every use in the body keeps the source's type. The conversion keeps the low 16 bits, which is what the Metal
kernel observes for a ``ushort`` attribute.
"""

from __future__ import annotations

import re

from .array_utils import format_c_style_array_declaration, split_array_type_suffix

__all__ = [
    "HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES",
    "hlsl_narrow_compute_system_value_bridge",
]

# Thread-index system values and the 32-bit type each is declared with on the entry.
_SYSTEM_VALUE_TYPES = {
    "SV_GroupID": "uint3",
    "SV_GroupThreadID": "uint3",
    "SV_DispatchThreadID": "uint3",
    "SV_GroupIndex": "uint",
}

_NARROW_UNSIGNED = re.compile(r"(?P<scalar>min16uint|uint16_t)(?P<width>[123]?)")

HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES = frozenset(
    f"{scalar}{width}"
    for scalar in ("min16uint", "uint16_t")
    for width in ("", "2", "3")
)

_COMPONENTS = {1: "x", 2: "xy", 3: "xyz"}


def hlsl_narrow_compute_system_value_bridge(
    codegen, parameter, param_type, semantic, emitted_param_names
):
    """Return ``(declaration, prologue)`` for a 16-bit thread-index entry parameter, else ``None``.

    ``declaration`` is the 32-bit entry parameter without its semantic; ``prologue`` declares the parameter's own
    name with its 16-bit type, initialised from the 32-bit value.
    """
    canonical = codegen.hlsl_canonical_semantic(semantic)
    wide_type = _SYSTEM_VALUE_TYPES.get(canonical)
    if wide_type is None:
        return None

    base_type, array_suffix = split_array_type_suffix(str(param_type))
    match = _NARROW_UNSIGNED.fullmatch(str(base_type).strip())
    if array_suffix or match is None:
        return None
    width = int(match.group("width") or "1")
    if wide_type == "uint" and width != 1:
        return None

    parameter_name = getattr(parameter, "name", None)
    if not isinstance(parameter_name, str) or not parameter_name:
        return None

    local_name = codegen.hlsl_declaration_identifier_name(parameter_name)
    used_names = set(emitted_param_names)
    used_names.update(codegen.local_variable_types)
    used_names.update(codegen.global_variable_types)
    used_names.update(codegen.current_identifier_aliases.values())
    wide_name = codegen.hlsl_unique_local_identifier(f"{local_name}_wide", used_names)
    emitted_param_names.add(wide_name)
    codegen.local_variable_types[wide_name] = wide_type
    codegen.local_variable_types[local_name] = base_type

    source = wide_name if wide_type == "uint" else f"{wide_name}.{_COMPONENTS[width]}"
    declaration = format_c_style_array_declaration(wide_type, wide_name)
    prologue = f"{base_type} {local_name} = ({base_type}){source};"
    return declaration, prologue
