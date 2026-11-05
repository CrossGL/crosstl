"""Byte-address access to GLSL buffer members whose components are 16 or 64 bits wide.

A std140 or std430 block lowered to a ``ByteAddressBuffer`` reads a 32-bit member with ``Load``/``Load2``/...,
which return ``uint`` words. Those methods cannot read an ``int16_t``, ``float16_t`` or ``int64_t`` member: a
``Load`` at the member's offset returns 32 bits whatever the member's width, and ``Load2`` of a ``u64vec2``
returns its first component split into two words. The templated forms ``Load<T>`` and ``Store<T>`` (HLSL 2018,
Shader Model 6.0; 16-bit ``T`` needs Shader Model 6.2 and ``-enable-16bit-types``) move exactly ``sizeof(T)``
bytes, so each member is read and written at its own width.

A ``float16_t`` member moves as ``uint16_t`` bits and is reinterpreted with ``asfloat16``/``asuint16``: the D3D11.3
functional specification lets every operation except a raw move change the bit pattern of a NaN (section 3.1.3.2),
which is how ``hlsl_half_storage`` keeps binary16 structured-buffer elements too.

HLSL has no 8-bit load or store, and a byte written through a 32-bit word would race with the writes other
invocations make to its neighbours, so 8-bit members stay refused. Slang's ``ByteAddressBuffer`` has the same
templated methods, so the Slang generator uses these helpers too.
"""

from __future__ import annotations

from .glsl_buffer_layout import glsl_buffer_member_leaves

__all__ = [
    "hlsl_byteaddress_component_refusal",
    "hlsl_byteaddress_is_typed_access",
    "hlsl_byteaddress_typed_load",
    "hlsl_byteaddress_typed_store",
]

# CrossGL component type -> the HLSL scalar its bytes move as.
_STORAGE_SCALARS = {
    "int16_t": "int16_t",
    "uint16_t": "uint16_t",
    "float16_t": "uint16_t",
    "int64_t": "int64_t",
    "uint64_t": "uint64_t",
}
_SIXTEEN_BIT = frozenset({"int16_t", "uint16_t", "float16_t"})


def hlsl_byteaddress_is_typed_access(access):
    """Whether ``access`` is a scalar or vector whose components move with ``Load<T>``/``Store<T>``."""
    return (
        not access.get("matrix_columns")
        and access.get("component_size", 4) != 4
        and access.get("component_type") in _STORAGE_SCALARS
    )


def hlsl_byteaddress_component_refusal(codegen, member_info):
    """Why the HLSL generator cannot read or write the components of ``member_info``, or ``None``."""
    for leaf in glsl_buffer_member_leaves(member_info):
        size = leaf.get("component_size", 4)
        component_type = leaf.get("component_type")
        if size == 4:
            continue
        if component_type not in _STORAGE_SCALARS:
            return (
                f"{size * 8}-bit {component_type} components have no ByteAddressBuffer "
                "load or store (it has no 8-bit memory access)"
            )
        if getattr(codegen, "target_profile", None) == "dx11":
            return (
                f"{size * 8}-bit {component_type} components need templated "
                "ByteAddressBuffer Load<T> and Store<T>, which Shader Model 5 lacks; "
                "use the directx-12 profile"
            )
        if component_type == "float16_t" and getattr(
            codegen, "widen_native_float16", False
        ):
            return (
                "float16_t components keep their binary16 bits through uint16_t, which "
                "widen_native_float16 does not compute in"
            )
    return None


def _typed_name(scalar, components):
    return scalar if components == 1 else f"{scalar}{components}"


def _require_native_16_bit(codegen, access):
    # DXC compiles 16-bit types only with -enable-16bit-types; slangc needs no flag.
    if access["component_type"] in _SIXTEEN_BIT and hasattr(
        codegen, "requires_hlsl_native_16_bit_types"
    ):
        codegen.requires_hlsl_native_16_bit_types = True


def hlsl_byteaddress_typed_load(codegen, buffer_name, offset, access):
    """``buffer.Load<T>(offset)`` of a 16- or 64-bit scalar or vector member."""
    _require_native_16_bit(codegen, access)
    storage = _typed_name(
        _STORAGE_SCALARS[access["component_type"]], access["components"]
    )
    load = f"{buffer_name}.Load<{storage}>({offset})"
    if access["component_type"] == "float16_t":
        return f"asfloat16({load})"
    return load


def hlsl_byteaddress_typed_store(codegen, buffer_name, offset, value, access):
    """``buffer.Store<T>(offset, value)`` of a 16- or 64-bit scalar or vector member."""
    _require_native_16_bit(codegen, access)
    components = access["components"]
    component_type = access["component_type"]
    storage = _typed_name(_STORAGE_SCALARS[component_type], components)
    value_type = _typed_name(component_type, components)
    stored = f"{value_type}({value})"
    if component_type == "float16_t":
        stored = f"asuint16({stored})"
    return f"{buffer_name}.Store<{storage}>({offset}, {stored})"
