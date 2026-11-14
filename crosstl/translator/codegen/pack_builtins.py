"""GLSL packing built-ins (GLSL 4.60 section 8.4) lowered for Metal, HLSL and SPIR-V.

CrossGL spells the packing functions as GLSL does. GLSL emits them natively and SPIR-V maps each one to its
GLSL.std.450 instruction. Metal has the normalized forms under its own names (MSL 3.2 section 6.16, "Pack and
Unpack Functions") and reaches the half forms through ``as_type``. HLSL has no packing intrinsics for these formats,
so it calls small helpers that apply the GLSL formulas:

- ``packUnorm``: ``round(clamp(c, 0, 1) * (2^n - 1))``; ``packSnorm``: ``round(clamp(c, -1, 1) * (2^(n-1) - 1))``;
- ``unpackUnorm``: ``f / (2^n - 1)``; ``unpackSnorm``: ``clamp(f / (2^(n-1) - 1), -1, 1)``;
- the first component always occupies the least significant bits.

Metal has no 64-bit floating-point type, so the double forms are refused there.
"""

from __future__ import annotations

__all__ = [
    "METAL_PACK_FUNCTIONS",
    "PACK_FUNCTIONS",
    "emit_hlsl_pack_helpers",
    "hlsl_pack_call",
    "metal_pack_call",
    "spirv_pack_call",
    "spirv_pack_result_type",
]

# CrossGL name: (operand type, result type, GLSL.std.450 instruction).
PACK_FUNCTIONS = {
    "packSnorm4x8": ("vec4", "uint", "PackSnorm4x8"),
    "packUnorm4x8": ("vec4", "uint", "PackUnorm4x8"),
    "packSnorm2x16": ("vec2", "uint", "PackSnorm2x16"),
    "packUnorm2x16": ("vec2", "uint", "PackUnorm2x16"),
    "packHalf2x16": ("vec2", "uint", "PackHalf2x16"),
    "packDouble2x32": ("uvec2", "double", "PackDouble2x32"),
    "unpackSnorm2x16": ("uint", "vec2", "UnpackSnorm2x16"),
    "unpackUnorm2x16": ("uint", "vec2", "UnpackUnorm2x16"),
    "unpackHalf2x16": ("uint", "vec2", "UnpackHalf2x16"),
    "unpackSnorm4x8": ("uint", "vec4", "UnpackSnorm4x8"),
    "unpackUnorm4x8": ("uint", "vec4", "UnpackUnorm4x8"),
    "unpackDouble2x32": ("double", "uvec2", "UnpackDouble2x32"),
}

# Metal standard-library names with exactly the GLSL meaning, and their CrossGL names.
METAL_PACK_FUNCTIONS = {
    "pack_float_to_snorm4x8": "packSnorm4x8",
    "pack_float_to_unorm4x8": "packUnorm4x8",
    "pack_float_to_snorm2x16": "packSnorm2x16",
    "pack_float_to_unorm2x16": "packUnorm2x16",
    "unpack_snorm4x8_to_float": "unpackSnorm4x8",
    "unpack_unorm4x8_to_float": "unpackUnorm4x8",
    "unpack_snorm2x16_to_float": "unpackSnorm2x16",
    "unpack_unorm2x16_to_float": "unpackUnorm2x16",
}
_METAL_NAMES = {crossgl: metal for metal, crossgl in METAL_PACK_FUNCTIONS.items()}
_METAL_OPERAND_CASTS = {"vec4": "float4", "vec2": "float2", "uint": "uint"}

_HLSL_HELPERS = {
    "packUnorm4x8": (
        "uint __crossgl_packUnorm4x8(float4 v)\n{\n"
        "    uint4 q = uint4(round(saturate(v) * 255.0));\n"
        "    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);\n}\n"
    ),
    "packSnorm4x8": (
        "uint __crossgl_packSnorm4x8(float4 v)\n{\n"
        "    uint4 q = uint4(int4(round(clamp(v, -1.0, 1.0) * 127.0))) & 0xFFu;\n"
        "    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);\n}\n"
    ),
    "packUnorm2x16": (
        "uint __crossgl_packUnorm2x16(float2 v)\n{\n"
        "    uint2 q = uint2(round(saturate(v) * 65535.0));\n"
        "    return q.x | (q.y << 16);\n}\n"
    ),
    "packSnorm2x16": (
        "uint __crossgl_packSnorm2x16(float2 v)\n{\n"
        "    uint2 q = uint2(int2(round(clamp(v, -1.0, 1.0) * 32767.0))) & 0xFFFFu;\n"
        "    return q.x | (q.y << 16);\n}\n"
    ),
    "packHalf2x16": (
        "uint __crossgl_packHalf2x16(float2 v)\n{\n"
        "    uint2 h = f32tof16(v);\n"
        "    return h.x | (h.y << 16);\n}\n"
    ),
    "unpackUnorm4x8": (
        "float4 __crossgl_unpackUnorm4x8(uint p)\n{\n"
        "    return float4((p >> uint4(0u, 8u, 16u, 24u)) & 0xFFu) / 255.0;\n}\n"
    ),
    "unpackSnorm4x8": (
        "float4 __crossgl_unpackSnorm4x8(uint p)\n{\n"
        "    int4 s = int4(p << uint4(24u, 16u, 8u, 0u)) >> 24;\n"
        "    return clamp(float4(s) / 127.0, -1.0, 1.0);\n}\n"
    ),
    "unpackUnorm2x16": (
        "float2 __crossgl_unpackUnorm2x16(uint p)\n{\n"
        "    return float2((p >> uint2(0u, 16u)) & 0xFFFFu) / 65535.0;\n}\n"
    ),
    "unpackSnorm2x16": (
        "float2 __crossgl_unpackSnorm2x16(uint p)\n{\n"
        "    int2 s = int2(p << uint2(16u, 0u)) >> 16;\n"
        "    return clamp(float2(s) / 32767.0, -1.0, 1.0);\n}\n"
    ),
    "unpackHalf2x16": (
        "float2 __crossgl_unpackHalf2x16(uint p)\n{\n"
        "    return f16tof32(uint2(p & 0xFFFFu, p >> 16));\n}\n"
    ),
    "packDouble2x32": (
        "double __crossgl_packDouble2x32(uint2 v)\n{\n"
        "    return asdouble(v.x, v.y);\n}\n"
    ),
    "unpackDouble2x32": (
        "uint2 __crossgl_unpackDouble2x32(double d)\n{\n"
        "    uint lo;\n    uint hi;\n    asuint(d, lo, hi);\n"
        "    return uint2(lo, hi);\n}\n"
    ),
}
_HLSL_OPERAND_CASTS = {
    "vec4": "float4",
    "vec2": "float2",
    "uint": "uint",
    "uvec2": "uint2",
    "double": "double",
}


def _arity_reason(name, args):
    if len(args) != 1:
        return f"{name} requires exactly one operand, got {len(args)}"
    return None


def metal_pack_call(codegen, name, rendered_args):
    """Metal for a packing built-in; ``rendered_args`` are Metal expressions."""
    reason = _arity_reason(name, rendered_args)
    operand, result, _instruction = PACK_FUNCTIONS[name]
    if reason is None and "Double" in name:
        reason = f"{name} needs a 64-bit floating-point type, which Metal does not have"
    if reason is not None:
        return codegen.refuse_lowering(reason, f"{codegen.map_type(result)}(0)")
    if name == "packHalf2x16":
        return f"as_type<uint>(half2({rendered_args[0]}))"
    value = f"{_METAL_OPERAND_CASTS[operand]}({rendered_args[0]})"
    if name == "unpackHalf2x16":
        return f"float2(as_type<half2>({value}))"
    return f"{_METAL_NAMES[name]}({value})"


def hlsl_pack_call(codegen, name, rendered_args):
    """HLSL for a packing built-in; the helper it calls is emitted by :func:`emit_hlsl_pack_helpers`."""
    reason = _arity_reason(name, rendered_args)
    operand, result, _instruction = PACK_FUNCTIONS[name]
    if reason is not None:
        return codegen.refuse_lowering(reason, f"({codegen.map_type(result)})0")
    helpers = codegen.__dict__.setdefault("hlsl_pack_helpers", {})
    helpers.setdefault(name, _HLSL_HELPERS[name])
    return f"__crossgl_{name}(({_HLSL_OPERAND_CASTS[operand]})({rendered_args[0]}))"


def emit_hlsl_pack_helpers(codegen):
    """The HLSL helpers the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("hlsl_pack_helpers", {}).values())


def _spirv_type(codegen, name):
    if name in {"uint", "double"}:
        return codegen.register_primitive_type(name)
    component = "uint" if name.startswith("u") else "float"
    return codegen.register_vector_type(
        codegen.register_primitive_type(component), int(name[-1])
    )


def spirv_pack_result_type(codegen, name):
    """The SPIR-V result type of a packing built-in, or ``None`` for another name."""
    entry = PACK_FUNCTIONS.get(name)
    return _spirv_type(codegen, entry[1]) if entry else None


def _spirv_operand(codegen, name, value, operand):
    """``value`` as the operand type, or ``None`` when it has another shape or component type."""
    expected = _spirv_type(codegen, operand)
    actual = codegen.registered_value_type(value) or codegen.ensure_registered_type(
        value.type
    )
    if actual.id == expected.id:
        return value
    if operand == "uint" and codegen.normalize_primitive_name(
        actual.type.base_type
    ) in {"int"}:
        converted = codegen.get_id()
        codegen.emit(f"%{converted} = OpBitcast %{expected.id} %{value.id}")
        codegen.value_types[converted] = expected
        return type(value)(converted, expected.type)
    return None


def spirv_pack_call(codegen, name, args):
    """The GLSL.std.450 instruction for a packing built-in; the operand must have the GLSL type exactly."""
    operand, result, instruction = PACK_FUNCTIONS[name]
    result_type = _spirv_type(codegen, result)
    reason = _arity_reason(name, args)
    value = None
    if reason is None:
        value = _spirv_operand(codegen, name, args[0], operand)
        if value is None:
            reason = f"{name} requires a {operand} operand"
    if reason is not None:
        codegen.refuse_lowering(reason)
        return codegen.default_value_for_type(result_type)
    return codegen.emit_glsl_std450_instruction(instruction, result_type, [value])
