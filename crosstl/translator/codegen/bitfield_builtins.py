"""GLSL bit-field built-ins (GLSL 4.60 section 8.8) lowered for Metal, HLSL and SPIR-V.

``bitfieldExtract(value, offset, bits)`` returns bits ``[offset, offset + bits)`` of ``value`` in the low bits of
the result, sign-extended from bit ``bits - 1`` for signed types, and 0 when ``bits`` is 0. ``bitfieldInsert(base,
insert, offset, bits)`` replaces those bits of ``base`` with the low ``bits`` bits of ``insert`` and returns
``base`` when ``bits`` is 0. Both are undefined when ``offset`` or ``bits`` is negative or ``offset + bits``
exceeds the bit width, and the lowerings may assume that never happens.

Metal has the same functions as ``extract_bits`` and ``insert_bits`` (MSL section 6.13, with ``uint`` offsets),
SPIR-V has ``OpBitFieldSExtract``, ``OpBitFieldUExtract`` and ``OpBitFieldInsert``, and HLSL calls helpers that
build the field mask. HLSL masks shift counts to five bits, so the helpers never shift by 32.
"""

from __future__ import annotations

import re

__all__ = [
    "BITFIELD_ARITIES",
    "METAL_BITFIELD_FUNCTIONS",
    "emit_hlsl_bitfield_helpers",
    "hlsl_bitfield_call",
    "metal_bitfield_call",
    "spirv_bitfield_call",
]

BITFIELD_ARITIES = {"bitfieldExtract": 3, "bitfieldInsert": 4}
METAL_BITFIELD_FUNCTIONS = {
    "extract_bits": "bitfieldExtract",
    "insert_bits": "bitfieldInsert",
}
_METAL_NAMES = {crossgl: metal for metal, crossgl in METAL_BITFIELD_FUNCTIONS.items()}
_HLSL_INTEGER = re.compile(r"(u?int)([234]?)")


def _arity_reason(name, count):
    expected = BITFIELD_ARITIES[name]
    if count != expected:
        return f"{name} requires {expected} operands, got {count}"
    return None


def metal_bitfield_call(codegen, name, rendered_args):
    """Metal for a bit-field built-in; ``rendered_args`` are Metal expressions."""
    reason = _arity_reason(name, len(rendered_args))
    if reason is not None:
        return codegen.refuse_lowering(reason, "0")
    *values, offset, bits = rendered_args
    operands = ", ".join([*values, f"uint({offset})", f"uint({bits})"])
    return f"{_METAL_NAMES[name]}({operands})"


def _hlsl_helper(name, mapped):
    scalar, _width = _HLSL_INTEGER.fullmatch(mapped).groups()
    suffix = mapped
    unsigned = "uint" + mapped[len(scalar) :]
    mask = "(bits == 0u ? 0u : (0xFFFFFFFFu >> (32u - bits)))"
    if name == "bitfieldExtract":
        if scalar == "uint":
            body = f"    return (value >> offset) & {mask};\n"
        else:
            body = (
                f"    {mapped} field = asint(asuint(value) << ((32u - offset - bits) & 31u)) >> ((32u - bits) & 31u);\n"
                f"    return bits == 0u ? ({mapped})0 : field;\n"
            )
        signature = f"{mapped} __crossgl_bitfieldExtract_{suffix}({mapped} value, uint offset, uint bits)"
    else:
        cast = "" if scalar == "uint" else "asint"
        body = (
            f"    uint mask = {mask} << offset;\n"
            f"    {unsigned} merged = (asuint(base) & ~mask) | ((asuint(insert) << offset) & mask);\n"
            f"    return {cast}(merged);\n"
        )
        signature = f"{mapped} __crossgl_bitfieldInsert_{suffix}({mapped} base, {mapped} insert, uint offset, uint bits)"
    return f"__crossgl_{name}_{suffix}", f"{signature}\n{{\n{body}}}\n"


def hlsl_bitfield_call(codegen, name, args, rendered_args):
    """HLSL for a bit-field built-in on 32-bit integers; other value types are refused."""
    reason = _arity_reason(name, len(rendered_args))
    mapped = None
    if reason is None:
        value_type = codegen.expression_result_type(args[0])
        mapped = codegen.map_type(value_type) if value_type else None
        if mapped is None or not _HLSL_INTEGER.fullmatch(mapped):
            reason = f"{name} requires a 32-bit integer scalar or vector value, got {mapped or 'an unresolved type'}"
    if reason is not None:
        return codegen.refuse_lowering(reason, "0")
    helper, code = _hlsl_helper(name, mapped)
    codegen.__dict__.setdefault("hlsl_bitfield_helpers", {}).setdefault(helper, code)
    *values, offset, bits = rendered_args
    operands = [f"({mapped})({value})" for value in values]
    operands += [f"(uint)({offset})", f"(uint)({bits})"]
    return f"{helper}({', '.join(operands)})"


def emit_hlsl_bitfield_helpers(codegen):
    """The HLSL helpers the program calls, in the order they were first needed."""
    return "".join(codegen.__dict__.get("hlsl_bitfield_helpers", {}).values())


def _spirv_integer_type(codegen, value):
    value_type = codegen.registered_value_type(value) or codegen.ensure_registered_type(
        value.type
    )
    component = codegen.normalize_primitive_name(
        codegen.scalar_or_vector_component_type(value_type.type)
    )
    return value_type, component if component in {"int", "uint"} else None


def _spirv_count(codegen, value):
    """An offset or count operand: any 32-bit integer scalar is valid as it is."""
    _type, component = _spirv_integer_type(codegen, value)
    vector = codegen.vector_component_type_and_count(_type.type.base_type)
    return value if component is not None and vector is None else None


def spirv_bitfield_call(codegen, name, args):
    """``OpBitFieldSExtract``/``OpBitFieldUExtract``/``OpBitFieldInsert`` on 32-bit integer scalars or vectors."""
    reason = _arity_reason(name, len(args))
    if reason is None:
        result_type, component = _spirv_integer_type(codegen, args[0])
        counts = [_spirv_count(codegen, value) for value in args[-2:]]
        if component is None:
            reason = f"{name} requires a 32-bit integer scalar or vector value"
        elif None in counts:
            reason = f"{name} requires integer scalar offset and bit count operands"
        elif name == "bitfieldInsert" and (
            _spirv_integer_type(codegen, args[1])[0].id != result_type.id
        ):
            reason = "bitfieldInsert requires base and insert operands of one type"
    if reason is not None:
        codegen.refuse_lowering(reason)
        fallback = codegen.register_primitive_type("int")
        return codegen.default_value_for_type(fallback)
    if name == "bitfieldExtract":
        opcode = "OpBitFieldSExtract" if component == "int" else "OpBitFieldUExtract"
    else:
        opcode = "OpBitFieldInsert"
    result_id = codegen.get_id()
    operands = " ".join(f"%{value.id}" for value in args)
    codegen.emit(f"%{result_id} = {opcode} %{result_type.id} {operands}")
    codegen.value_types[result_id] = result_type
    return type(args[0])(result_id, result_type.type)
