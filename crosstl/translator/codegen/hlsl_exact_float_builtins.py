"""Exact ``fmod``, ``frexp`` and ``modf`` for HLSL output, and the floating-point ``%`` that means ``fmod``.

CrossGL gives these functions their C, GLSL 4.60 section 8.3 and MSL 3.2 section 6.5 meaning: ``fmod(x, y)`` is the
exact remainder ``x - n * y`` with ``n`` the quotient truncated toward zero, carrying the sign of ``x``; ``frexp``
returns a significand in [0.5, 1) with the sign of ``x``; ``modf`` returns the fraction and writes the whole part,
both with the sign of ``x``. The HLSL intrinsic documentation states the same results, but DXC's DXIL lowering
does not compute them:

- ``fmod(x, y)`` becomes ``q = x / y; r = frac(abs(q)); (q >= -q ? r : -r) * y`` with a fast division, so the
  remainder carries the quotient's rounding error (``fmod(-5.5, 2.0)`` reads ``-1.4999998`` on WARP). A floating
  ``%`` becomes ``frem fast``, and a double operand of ``%`` is an error.
- ``frexp`` masks the significand with ``0x007fffff``, so a negative value returns a positive significand, and a
  subnormal value returns zero.
- ``modf`` computes ``x - trunc(x)``, a positive zero for a negative integer and NaN for an infinity.
- A double operand of ``fmod``, ``frexp`` or ``modf`` is converted to float.

The helpers here compute each result from the bits of the operands with integer arithmetic, which DXIL executes
exactly on every device and which no fast-math flag reorders. ``fmod`` divides the integer significands as C
libraries do, so the remainder is exact for every operand, subnormal ones included; an infinite dividend, a NaN
or a zero divisor returns a quiet NaN. ``half`` operands are computed through the ``float`` helper: half values
convert to float exactly, and each result is representable in half again. Double helpers use ``uint64_t``, which
needs shader model 6.0.

An operand whose type the generator cannot resolve, and an operand that is not a floating-point or integer scalar
or vector, keeps the native spelling.
"""

from __future__ import annotations

import re

from ..ast import BinaryOpNode, FunctionCallNode, IdentifierNode, VariableNode

__all__ = ["emit_hlsl_exact_float_helpers", "hlsl_exact_float_expression"]

_FUNCTIONS = frozenset({"fmod", "frexp", "modf"})
_FLOATS = ("half", "float", "double")
_INTEGERS = frozenset({"int", "uint", "bool"})
_COMPONENTS = "xyzw"
_ATTRIBUTE = "hlsl_exact_float_helpers"

_NAMED = re.compile(
    r"(half|float|double|int|uint|bool|min16float|min10float|min16int|min16uint"
    r"|float16_t|float32_t|float64_t|int16_t|uint16_t|int32_t|uint32_t)([1-4]?)"
)
_GLSL_VECTOR = re.compile(r"(f16|f32|f64|d|i|u|b)?vec([234])")
_ALIASES = {
    **{name: name for name in ("half", "float", "double", "int", "uint", "bool")},
    "min16float": "half",
    "min10float": "half",
    "float16_t": "half",
    "float32_t": "float",
    "float64_t": "double",
    "min16int": "int",
    "int16_t": "int",
    "int32_t": "int",
    "min16uint": "uint",
    "uint16_t": "uint",
    "uint32_t": "uint",
}
_GLSL_PREFIX = {
    None: "float",
    "f32": "float",
    "f16": "half",
    "f64": "double",
    "d": "double",
    "i": "int",
    "u": "uint",
    "b": "bool",
}

# bits, fraction bits, exponent bits and exponent bias of each format the helpers decode.
_FORMATS = {"float": (32, 23, 8, 127), "double": (64, 52, 11, 1023)}


def _shape(type_name):
    """``(component, count)`` of a scalar or vector type in CrossGL or HLSL spelling, else ``None``."""
    text = str(type_name or "").strip()
    match = _GLSL_VECTOR.fullmatch(text)
    if match:
        return _GLSL_PREFIX[match.group(1)], int(match.group(2))
    match = _NAMED.fullmatch(text)
    if match:
        component = _ALIASES[match.group(1)]
        return component, int(match.group(2) or 1)
    return None


def _spell(component, count):
    return component if count == 1 else f"{component}{count}"


def _operand_shape(codegen, expr):
    try:
        return _shape(codegen.expression_result_type(expr))
    except (
        Exception
    ):  # noqa: BLE001 - an operand the generator cannot type keeps the native call
        return None


def _value_shape(shapes):
    """The floating-point shape the operands combine to, or ``None`` when the native spelling stays."""
    if any(shape is None for shape in shapes):
        return None
    floats = [shape[0] for shape in shapes if shape[0] in _FLOATS]
    if not floats or any(
        shape[0] not in _FLOATS and shape[0] not in _INTEGERS for shape in shapes
    ):
        return None
    counts = {shape[1] for shape in shapes}
    if len(counts - {1}) > 1:
        return None
    return max(floats, key=_FLOATS.index), max(counts)


def _helpers(codegen):
    helpers = codegen.__dict__.get(_ATTRIBUTE)
    if helpers is None:
        helpers = codegen.__dict__[_ATTRIBUTE] = {}
    return helpers


def _add(codegen, name, text):
    _helpers(codegen).setdefault(name, text)
    return name


class _Bits:
    """HLSL spellings of the integer word that holds a ``float`` or ``double``."""

    def __init__(self, component):
        self.bits, self.fraction, self.exponent, self.bias = _FORMATS[component]
        self.component = component
        self.word = "uint" if self.bits == 32 else "uint64_t"

    def one(self):
        return f"(({self.word})1)"

    def unpack(self, value):
        if self.bits == 32:
            return f"asuint({value})"
        return f"crosstl_double_bits({value})"

    def pack(self, value):
        if self.bits == 32:
            return f"asfloat({value})"
        return f"asdouble(uint({value}), uint(({value}) >> 32))"

    def constants(self):
        word, one = self.word, self.one()
        return (
            f"    {word} sign_mask = {one} << {self.bits - 1};\n"
            f"    {word} hidden_bit = {one} << {self.fraction};\n"
            f"    {word} fraction_mask = hidden_bit - {one};\n"
            f"    {word} infinity = (({one} << {self.exponent}) - {one}) << {self.fraction};\n"
        )


def _double_bits(codegen):
    _add(
        codegen,
        "crosstl_double_bits",
        "uint64_t crosstl_double_bits(double value) {\n"
        "    uint low_bits;\n"
        "    uint high_bits;\n"
        "    asuint(value, low_bits, high_bits);\n"
        "    return ((uint64_t)high_bits << 32) | (uint64_t)low_bits;\n"
        "}\n\n",
    )


def _fmod_scalar(codegen, component):
    """The scalar ``fmod`` helper of ``component``; integer significand division as C libraries do it."""
    name = f"crosstl_fmod_{component}"
    if name in _helpers(codegen):
        return name
    if component == "half":
        inner = _fmod_scalar(codegen, "float")
        return _add(
            codegen,
            name,
            f"half {name}(half x, half y) {{\n"
            f"    return (half){inner}((float)x, (float)y);\n"
            "}\n\n",
        )
    form = _Bits(component)
    if component == "double":
        _double_bits(codegen)
    word, zero = form.word, f"(({form.word})0)"
    text = (
        f"{component} {name}({component} x, {component} y) {{\n"
        + form.constants()
        + f"    {word} x_bits = {form.unpack('x')};\n"
        f"    {word} y_bits = {form.unpack('y')};\n"
        f"    {word} sign = x_bits & sign_mask;\n"
        f"    {word} a = x_bits & ~sign_mask;\n"
        f"    {word} b = y_bits & ~sign_mask;\n"
        f"    if (a >= infinity || b > infinity || b == {zero}) {{\n"
        f"        return {form.pack('infinity | (hidden_bit >> 1)')};\n"
        "    }\n"
        "    if (a < b) {\n"
        "        return x;\n"
        "    }\n"
        "    if (a == b) {\n"
        f"        return {form.pack('sign')};\n"
        "    }\n"
        f"    int x_exponent = int(a >> {form.fraction});\n"
        f"    int y_exponent = int(b >> {form.fraction});\n"
        f"    {word} remainder = a & fraction_mask;\n"
        f"    {word} divisor = b & fraction_mask;\n"
        "    if (x_exponent == 0) {\n"
        "        x_exponent = 1;\n"
        "        while (remainder < hidden_bit) {\n"
        "            remainder <<= 1;\n"
        "            x_exponent -= 1;\n"
        "        }\n"
        "    } else {\n"
        "        remainder |= hidden_bit;\n"
        "    }\n"
        "    if (y_exponent == 0) {\n"
        "        y_exponent = 1;\n"
        "        while (divisor < hidden_bit) {\n"
        "            divisor <<= 1;\n"
        "            y_exponent -= 1;\n"
        "        }\n"
        "    } else {\n"
        "        divisor |= hidden_bit;\n"
        "    }\n"
        "    for (int shift = x_exponent - y_exponent; shift >= 0; shift -= 1) {\n"
        "        if (remainder >= divisor) {\n"
        "            remainder -= divisor;\n"
        "        }\n"
        f"        if (remainder == {zero}) {{\n"
        f"            return {form.pack('sign')};\n"
        "        }\n"
        "        if (shift > 0) {\n"
        "            remainder <<= 1;\n"
        "        }\n"
        "    }\n"
        "    while (remainder < hidden_bit) {\n"
        "        remainder <<= 1;\n"
        "        y_exponent -= 1;\n"
        "    }\n"
        "    if (y_exponent > 0) {\n"
        f"        remainder = (({word})y_exponent << {form.fraction}) | (remainder & fraction_mask);\n"
        "    } else {\n"
        "        remainder >>= uint(1 - y_exponent);\n"
        "    }\n"
        f"    return {form.pack('remainder | sign')};\n"
        "}\n\n"
    )
    return _add(codegen, name, text)


def _frexp_scalar(codegen, component):
    """The scalar ``frexp`` helper of ``component``, writing an ``int`` exponent."""
    name = f"crosstl_frexp_{component}"
    if name in _helpers(codegen):
        return name
    if component == "half":
        inner = _frexp_scalar(codegen, "float")
        return _add(
            codegen,
            name,
            f"half {name}(half x, out int exponent) {{\n"
            f"    return (half){inner}((float)x, exponent);\n"
            "}\n\n",
        )
    form = _Bits(component)
    if component == "double":
        _double_bits(codegen)
    word, zero = form.word, f"(({form.word})0)"
    half_exponent = f"((({word}){form.bias - 1}) << {form.fraction})"
    text = (
        f"{component} {name}({component} x, out int exponent) {{\n"
        + form.constants()
        + f"    {word} bits = {form.unpack('x')};\n"
        f"    {word} magnitude = bits & ~sign_mask;\n"
        "    exponent = 0;\n"
        f"    if (magnitude == {zero} || magnitude >= infinity) {{\n"
        "        return x;\n"
        "    }\n"
        f"    int biased = int(magnitude >> {form.fraction});\n"
        f"    {word} significand = magnitude & fraction_mask;\n"
        "    if (biased == 0) {\n"
        "        biased = 1;\n"
        "        while (significand < hidden_bit) {\n"
        "            significand <<= 1;\n"
        "            biased -= 1;\n"
        "        }\n"
        "        significand &= fraction_mask;\n"
        "    }\n"
        f"    exponent = biased - {form.bias - 1};\n"
        f"    return {form.pack(f'(bits & sign_mask) | {half_exponent} | significand')};\n"
        "}\n\n"
    )
    return _add(codegen, name, text)


def _modf_scalar(codegen, component):
    """The scalar ``modf`` helper of ``component``, writing a whole part of the same type."""
    name = f"crosstl_modf_{component}"
    if name in _helpers(codegen):
        return name
    if component == "half":
        inner = _modf_scalar(codegen, "float")
        return _add(
            codegen,
            name,
            f"half {name}(half x, out half whole) {{\n"
            "    float whole_part;\n"
            f"    half fraction = (half){inner}((float)x, whole_part);\n"
            "    whole = (half)whole_part;\n"
            "    return fraction;\n"
            "}\n\n",
        )
    form = _Bits(component)
    if component == "double":
        _double_bits(codegen)
    word = form.word
    text = (
        f"{component} {name}({component} x, out {component} whole) {{\n"
        + form.constants()
        + f"    {word} bits = {form.unpack('x')};\n"
        f"    {word} sign = bits & sign_mask;\n"
        f"    {word} magnitude = bits & ~sign_mask;\n"
        f"    int unbiased = int(magnitude >> {form.fraction}) - {form.bias};\n"
        "    if (magnitude > infinity) {\n"
        "        whole = x;\n"
        "        return x;\n"
        "    }\n"
        f"    if (unbiased >= {form.fraction}) {{\n"
        "        whole = x;\n"
        f"        return {form.pack('sign')};\n"
        "    }\n"
        "    if (unbiased < 0) {\n"
        f"        whole = {form.pack('sign')};\n"
        "        return x;\n"
        "    }\n"
        f"    whole = {form.pack('bits & ~(fraction_mask >> uint(unbiased))')};\n"
        f"    {component} fraction = x - whole;\n"
        f"    return {form.pack(form.unpack('fraction') + ' | sign')};\n"
        "}\n\n"
    )
    return _add(codegen, name, text)


def _fmod(codegen, value, rendered):
    component, count = value
    scalar = _fmod_scalar(codegen, component)
    if count == 1:
        return f"{scalar}({rendered[0]}, {rendered[1]})"
    vector = _spell(component, count)
    name = f"crosstl_fmod_{vector}"
    parts = ", ".join(f"{scalar}(x.{c}, y.{c})" for c in _COMPONENTS[:count])
    _add(
        codegen,
        name,
        f"{vector} {name}({vector} x, {vector} y) {{\n"
        f"    return {vector}({parts});\n"
        "}\n\n",
    )
    return f"{name}({rendered[0]}, {rendered[1]})"


def _with_part(codegen, function, value, part, rendered):
    """A ``frexp`` or ``modf`` call whose second operand has the shape ``part``."""
    component, count = value
    scalar = (_frexp_scalar if function == "frexp" else _modf_scalar)(
        codegen, component
    )
    natural = "int" if function == "frexp" else component
    if count == 1 and part == (natural, 1):
        return f"{scalar}({rendered[0]}, {rendered[1]})"
    value_type, part_type = _spell(component, count), _spell(*part)
    name = f"crosstl_{function}_{value_type}_{part_type}"
    lines = [
        f"{value_type} {name}({value_type} x, out {part_type} part) {{\n",
        f"    {value_type} result;\n",
        f"    {_spell(natural, count)} written;\n",
        f"    {natural} component_part;\n",
    ]
    for c in (_COMPONENTS[:count] if count > 1 else ("",)):
        select = f".{c}" if c else ""
        lines.append(f"    result{select} = {scalar}(x{select}, component_part);\n")
        lines.append(f"    written{select} = component_part;\n")
    lines.append(f"    part = ({part_type})written;\n")
    lines.append("    return result;\n}\n\n")
    _add(codegen, name, "".join(lines))
    return f"{name}({rendered[0]}, {rendered[1]})"


def _call_name(expr):
    function = getattr(expr, "function", None)
    if isinstance(function, str):
        return function
    if isinstance(function, (IdentifierNode, VariableNode)):
        return getattr(function, "name", None)
    return None


def _user_function(codegen, name):
    return name in (getattr(codegen, "function_return_types", None) or {})


def _function_call(codegen, expr):
    name = _call_name(expr)
    args = list(getattr(expr, "arguments", None) or ())
    if name not in _FUNCTIONS or len(args) != 2 or _user_function(codegen, name):
        return None
    shapes = [_operand_shape(codegen, arg) for arg in args]
    if name == "fmod":
        value = _value_shape(shapes)
        if value is None:
            return None
        return _fmod(codegen, value, [codegen.generate_expression(a) for a in args])
    value = _value_shape(shapes[:1])
    part = shapes[1]
    if value is None or part is None or part[1] != value[1]:
        return None
    if part[0] not in _FLOATS and part[0] not in _INTEGERS:
        return None
    rendered = [codegen.generate_expression(a) for a in args]
    return _with_part(codegen, name, value, part, rendered)


def _remainder(codegen, expr):
    operator = str(getattr(expr, "op", "") or "")
    if operator not in {"%", "MOD"}:
        return None
    operands = (expr.left, expr.right)
    shapes = [_operand_shape(codegen, operand) for operand in operands]
    value = _value_shape(shapes)
    if value is None:
        return None
    if not _user_function(codegen, "fmod"):
        # The helper is a call, whose arguments DXC evaluates right to left; as an ``fmod`` call the operands go
        # through the call-argument ordering (:mod:`.hlsl_call_argument_order`) like any other call's.
        call = FunctionCallNode(IdentifierNode("fmod"), list(operands))
        return codegen.generate_expression(call)
    rendered = [codegen.generate_expression(operand) for operand in operands]
    return _fmod(codegen, value, rendered)


def hlsl_exact_float_expression(codegen, expr):
    """The HLSL for an ``fmod``, ``frexp`` or ``modf`` call or a floating ``%``, or ``None`` for other nodes."""
    if isinstance(expr, FunctionCallNode):
        return _function_call(codegen, expr)
    if isinstance(expr, BinaryOpNode):
        return _remainder(codegen, expr)
    return None


def emit_hlsl_exact_float_helpers(codegen):
    """The helpers the program calls, each after the helpers it calls, in the order first needed."""
    return "".join(_helpers(codegen).values())
