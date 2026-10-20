"""GLSL extended integer arithmetic (GLSL 4.60 section 8.8) lowered for HLSL, Metal and SPIR-V.

- ``uaddCarry(x, y, out carry)`` returns ``x + y`` modulo 2^32 and writes 1 to ``carry`` when the sum overflows,
  else 0.
- ``usubBorrow(x, y, out borrow)`` returns ``x - y`` modulo 2^32 and writes 1 to ``borrow`` when ``x < y``, else 0.
- ``umulExtended(x, y, out msb, out lsb)`` and ``imulExtended(x, y, out msb, out lsb)`` write the high and the low
  32 bits of the 64-bit product, unsigned and two's-complement signed respectively.

The operands are 32-bit ``uint`` scalars or vectors (``int`` for ``imulExtended``) of one type. SPIR-V has the
struct-returning ``OpIAddCarry``, ``OpISubBorrow``, ``OpUMulExtended`` and ``OpSMulExtended`` (SPIR-V 1.6 section
3.52.13), whose members are stored to the out operands. Metal has ``mulhi`` (MSL 3.2 section 6.4) but no carry
functions, and HLSL has neither, so both call helpers: the carry and borrow come from an unsigned comparison, Metal
takes the high word from ``mulhi``, and HLSL builds it from 16-bit partial products so no 64-bit integer support is
needed. Signed products wrap through ``uint`` in the helpers, since signed overflow is undefined in MSL and HLSL.
Metal binds an out operand only to a reference in one address space, so its helpers are overloaded for ``thread``,
``device`` and ``threadgroup`` operands.

A call with another operand count is refused; that includes the two-operand call a SPIR-V import leaves for the
whole struct value of one of these instructions. Operands that are not 32-bit integers of one shape are refused.
"""

from __future__ import annotations

import itertools
import re

from ..ast import AssignmentNode, IdentifierNode

__all__ = [
    "EXTENDED_ARITHMETIC_FUNCTIONS",
    "emit_hlsl_extended_arithmetic_helpers",
    "emit_metal_extended_arithmetic_helpers",
    "glsl_extended_arithmetic_call",
    "hlsl_extended_arithmetic_call",
    "metal_extended_arithmetic_call",
    "spirv_extended_arithmetic_call",
]

# CrossGL function -> (operand component, operand count, SPIR-V opcode).
EXTENDED_ARITHMETIC_FUNCTIONS = {
    "uaddCarry": ("uint", 3, "OpIAddCarry"),
    "usubBorrow": ("uint", 3, "OpISubBorrow"),
    "umulExtended": ("uint", 4, "OpUMulExtended"),
    "imulExtended": ("int", 4, "OpSMulExtended"),
}
_MULTIPLICATIONS = {"umulExtended", "imulExtended"}
_GLSL_VECTOR = re.compile(r"([iubd]?)vec([234])")
_SIZED = re.compile(r"(int|uint|float|half|double|bool)([234]?)")
_PREFIX_COMPONENT = {"": "float", "i": "int", "u": "uint", "b": "bool", "d": "double"}
_METAL_ADDRESS_SPACES = ("thread", "device", "threadgroup")
_TEMPORARY_PREFIX = "_crosstl_extended_arithmetic_"


def _shape(type_name):
    """``(component, count)`` of a scalar or vector in CrossGL, GLSL, HLSL or Metal spelling, else ``None``."""
    text = str(type_name or "").strip()
    match = _GLSL_VECTOR.fullmatch(text)
    if match:
        return _PREFIX_COMPONENT[match.group(1)], int(match.group(2))
    match = _SIZED.fullmatch(text)
    if match:
        return match.group(1), int(match.group(2) or 1)
    return None


def _sized(component, count):
    return component if count == 1 else f"{component}{count}"


def _is_user_function(codegen, name, attribute):
    return name in (getattr(codegen, attribute, None) or {})


def _checked(codegen, name, args, require_types):
    """``(component, count)`` of the operation, ``None`` when unresolved and allowed, or a refusal string."""
    component, arity, _opcode = EXTENDED_ARITHMETIC_FUNCTIONS[name]
    if len(args or ()) != arity:
        return codegen.refuse_lowering(
            f"{name} takes {arity} operands, got {len(args or ())}; a SPIR-V "
            "extended-arithmetic struct used as a whole value has no CrossGL form",
            "0",
        )
    shapes = [_shape(codegen.expression_result_type(arg)) for arg in args]
    values, outputs = shapes[:2], shapes[2:]
    if None in values:
        if not require_types:
            return None
        return codegen.refuse_lowering(
            f"{name}: the operand types could not be resolved", "0"
        )
    counts = {count for _component, count in values}
    if len(counts) != 1 or any(
        value_component not in {"int", "uint"} for value_component, _ in values
    ):
        return codegen.refuse_lowering(
            f"{name} requires two 32-bit integer operands of one shape", "0"
        )
    shape = (component, counts.pop())
    for output in outputs:
        if output is not None and output != shape:
            return codegen.refuse_lowering(
                f"{name} writes {_sized(*shape)} outputs, not {_sized(*output)}", "0"
            )
    return shape


def glsl_extended_arithmetic_call(codegen, name, args):
    """A refusal for an extended-arithmetic call GLSL cannot take, or ``None`` to emit it natively."""
    if name not in EXTENDED_ARITHMETIC_FUNCTIONS or _is_user_function(
        codegen, name, "function_return_types"
    ):
        return None
    checked = _checked(codegen, name, args, require_types=False)
    return checked if isinstance(checked, str) else None


def _hlsl_helpers(name, shape):
    unsigned = _sized("uint", shape[1])
    if name == "uaddCarry":
        return (
            f"{unsigned} __crossgl_uaddCarry_{unsigned}({unsigned} x, {unsigned} y, out {unsigned} carry)\n{{\n"
            f"    {unsigned} sum = x + y;\n"
            f"    carry = ({unsigned})(sum < x);\n"
            "    return sum;\n}\n"
        )
    if name == "usubBorrow":
        return (
            f"{unsigned} __crossgl_usubBorrow_{unsigned}({unsigned} x, {unsigned} y, out {unsigned} borrow)\n{{\n"
            f"    borrow = ({unsigned})(x < y);\n"
            "    return x - y;\n}\n"
        )
    unsigned_product = (
        f"void __crossgl_umulExtended_{unsigned}({unsigned} x, {unsigned} y, out {unsigned} msb, out {unsigned} lsb)\n{{\n"
        f"    {unsigned} xl = x & 0xFFFFu;\n"
        f"    {unsigned} xh = x >> 16;\n"
        f"    {unsigned} yl = y & 0xFFFFu;\n"
        f"    {unsigned} yh = y >> 16;\n"
        f"    {unsigned} ll = xl * yl;\n"
        f"    {unsigned} hl = xh * yl;\n"
        f"    {unsigned} middle = (ll >> 16) + (hl & 0xFFFFu) + xl * yh;\n"
        "    msb = xh * yh + (hl >> 16) + (middle >> 16);\n"
        "    lsb = x * y;\n}\n"
    )
    if name == "umulExtended":
        return unsigned_product
    signed = _sized("int", shape[1])
    return (
        f"void __crossgl_imulExtended_{signed}({signed} x, {signed} y, out {signed} msb, out {signed} lsb)\n{{\n"
        f"    {unsigned} ux = asuint(x);\n"
        f"    {unsigned} uy = asuint(y);\n"
        f"    {unsigned} high;\n"
        f"    {unsigned} low;\n"
        f"    __crossgl_umulExtended_{unsigned}(ux, uy, high, low);\n"
        "    high -= (uy & asuint(x >> 31)) + (ux & asuint(y >> 31));\n"
        "    msb = asint(high);\n"
        "    lsb = asint(low);\n}\n"
    )


def hlsl_extended_arithmetic_call(codegen, name, args):
    """The HLSL helper call for an extended-arithmetic built-in, a refusal, or ``None`` for another call."""
    if name not in EXTENDED_ARITHMETIC_FUNCTIONS or _is_user_function(
        codegen, name, "function_return_types"
    ):
        return None
    shape = _checked(codegen, name, args, require_types=True)
    if isinstance(shape, str):
        return shape
    type_name = _sized(*shape)
    helper = f"__crossgl_{name}_{type_name}"
    helpers = codegen.__dict__.setdefault("hlsl_extended_arithmetic_helpers", {})
    if name == "imulExtended":
        # The signed product corrects the unsigned one, so that helper comes first.
        unsigned = _sized("uint", shape[1])
        helpers.setdefault(
            f"__crossgl_umulExtended_{unsigned}",
            _hlsl_helpers("umulExtended", ("uint", shape[1])),
        )
    helpers.setdefault(helper, _hlsl_helpers(name, shape))
    rendered = [codegen.generate_expression(arg) for arg in args]
    operands = [f"({type_name})({value})" for value in rendered[:2]] + rendered[2:]
    return f"{helper}({', '.join(operands)})"


def emit_hlsl_extended_arithmetic_helpers(codegen):
    """The HLSL helpers the program calls, in the order they were first needed."""
    return "".join(
        codegen.__dict__.get("hlsl_extended_arithmetic_helpers", {}).values()
    )


def _metal_helper(name, type_name, unsigned):
    if name in {"uaddCarry", "usubBorrow"}:
        if name == "uaddCarry":
            body = (
                f"    {type_name} sum = x + y;\n"
                f"    carry = select({type_name}(0u), {type_name}(1u), sum < x);\n"
                "    return sum;\n"
            )
        else:
            body = (
                f"    carry = select({type_name}(0u), {type_name}(1u), x < y);\n"
                "    return x - y;\n"
            )
        return "".join(
            f"static inline {type_name} __crossgl_{name}_{type_name}({type_name} x, {type_name} y, "
            f"{space} {type_name}& carry) {{\n{body}}}\n"
            for space in _METAL_ADDRESS_SPACES
        )
    if name == "umulExtended":
        low = "x * y"
    else:
        low = f"as_type<{type_name}>(as_type<{unsigned}>(x) * as_type<{unsigned}>(y))"
    body = f"    msb = mulhi(x, y);\n    lsb = {low};\n"
    return "".join(
        f"static inline void __crossgl_{name}_{type_name}({type_name} x, {type_name} y, "
        f"{msb_space} {type_name}& msb, {lsb_space} {type_name}& lsb) {{\n{body}}}\n"
        for msb_space, lsb_space in itertools.product(_METAL_ADDRESS_SPACES, repeat=2)
    )


def metal_extended_arithmetic_call(codegen, name, args):
    """The Metal helper call for an extended-arithmetic built-in, a refusal, or ``None`` for another call."""
    if name not in EXTENDED_ARITHMETIC_FUNCTIONS or name in getattr(
        codegen, "user_function_names", ()
    ):
        return None
    shape = _checked(codegen, name, args, require_types=True)
    if isinstance(shape, str):
        return shape
    type_name = _sized(*shape)
    helper = f"__crossgl_{name}_{type_name}"
    codegen.__dict__.setdefault("metal_extended_arithmetic_helpers", {}).setdefault(
        helper, _metal_helper(name, type_name, _sized("uint", shape[1]))
    )
    rendered = [codegen.generate_expression(arg) for arg in args]
    operands = [f"{type_name}({value})" for value in rendered[:2]] + rendered[2:]
    return f"{helper}({', '.join(operands)})"


def emit_metal_extended_arithmetic_helpers(codegen):
    """The Metal helpers the program calls, in the order they were first needed."""
    return "".join(
        codegen.__dict__.get("metal_extended_arithmetic_helpers", {}).values()
    )


def _spirv_value_type(codegen, value):
    return codegen.registered_value_type(value) or codegen.ensure_registered_type(
        value.type
    )


def _spirv_shape(codegen, value_type):
    vector = codegen.vector_component_type_and_count(value_type.type.base_type)
    if vector is not None:
        return codegen.normalize_primitive_name(vector[0]), vector[1]
    return codegen.normalize_primitive_name(value_type.type.base_type), 1


def _spirv_store(codegen, target, value, value_type):
    """Store ``value`` to the CrossGL l-value ``target`` with the conversions an assignment applies."""
    codegen.extended_arithmetic_temporaries = (
        getattr(codegen, "extended_arithmetic_temporaries", 0) + 1
    )
    name = f"{_TEMPORARY_PREFIX}{codegen.extended_arithmetic_temporaries}"
    temporary = codegen.create_variable(value_type, "Function")
    codegen.store_to_variable(temporary, value)
    codegen.local_variables[name] = temporary
    try:
        codegen.process_assignment(AssignmentNode(target, IdentifierNode(name)))
    finally:
        del codegen.local_variables[name]


def spirv_extended_arithmetic_call(codegen, name, arguments):
    """Lower an extended-arithmetic call; the arguments are CrossGL expressions, not yet evaluated.

    Returns the sum or difference for ``uaddCarry``/``usubBorrow`` and ``None`` for the multiplications, which
    return nothing.
    """
    component, arity, opcode = EXTENDED_ARITHMETIC_FUNCTIONS[name]
    if len(arguments) != arity:
        raise codegen.builtin_lowering_error(
            name,
            f"it takes {arity} operands; a SPIR-V extended-arithmetic struct used as a whole value has no "
            "CrossGL form",
            reason="arity",
        )
    values = [codegen.process_expression(argument) for argument in arguments[:2]]
    if None in values:
        return None
    value_types = [_spirv_value_type(codegen, value) for value in values]
    shapes = [_spirv_shape(codegen, value_type) for value_type in value_types]
    if (
        any(shape_component not in {"int", "uint"} for shape_component, _ in shapes)
        or shapes[0][1] != shapes[1][1]
    ):
        raise codegen.builtin_lowering_error(
            name, "it requires two 32-bit integer operands of one shape"
        )
    scalar = codegen.register_primitive_type(component)
    count = shapes[0][1]
    operand_type = scalar if count == 1 else codegen.register_vector_type(scalar, count)
    values = [codegen.convert_value_to_type(value, operand_type) for value in values]
    struct_name = f"_{name}_result_{operand_type.type.base_type}"
    struct_type = codegen.struct_types.get(struct_name) or codegen.register_struct_type(
        struct_name, [(operand_type, "low"), (operand_type, "high")]
    )
    result_id = codegen.get_id()
    codegen.emit(
        f"%{result_id} = {opcode} %{struct_type.id} %{values[0].id} %{values[1].id}"
    )
    codegen.value_types[result_id] = struct_type
    result = type(values[0])(result_id, struct_type.type)
    first = codegen.composite_extract(result, operand_type, 0)
    second = codegen.composite_extract(result, operand_type, 1)
    _spirv_store(codegen, arguments[2], second, operand_type)
    if name in _MULTIPLICATIONS:
        _spirv_store(codegen, arguments[3], first, operand_type)
        return None
    return first
