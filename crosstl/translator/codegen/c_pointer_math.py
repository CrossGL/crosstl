"""CUDA and HIP spelling of the C math functions that write results through pointers, and of ``ldexp``.

CrossGL takes ``frexp(x, out e)`` and ``modf(x, out i)`` from GLSL 4.60 section 8.3 and ``sincos(x, out s, out c)``
from HLSL, writes every result part as an ``out`` argument, and lets all of them act on vectors component by
component. Programs imported from HLSL also give the ``frexp`` exponent the value's type, write the ``modf`` whole
part to an integer, and pass a float or a scalar ``ldexp`` exponent next to a vector value (see
``exponent_builtins``). CrossGL programs may also call the CUDA and HIP functions of the same shape, ``remquo``,
``sincospi`` and the float spellings such as ``frexpf`` and ``__sincosf``, with the same ``out`` arguments.

CUDA and HIP declare only the scalar C functions, which write each part through a pointer: ``frexpf(float, int*)``,
``modff(float, float*)``, ``sincosf(float, float*, float*)``, ``ldexpf(float, int)`` and so on, with the ``double``
forms named without the suffix. The call written verbatim passed an ``int`` where an ``int*`` is expected and has no
overload at all for a vector, so neither compiler accepted it.

- A scalar call whose parts have the types the C function writes calls it directly, ``frexpf(x, &(e))``.
- Any other call goes through a generated ``__device__`` helper that takes the parts by reference, calls the C function
  once per component, and converts each part as an HLSL ``out`` argument of another type would.
- An ``ldexp`` floating-point exponent means ``x * 2^e`` in HLSL, which DXC lowers to ``x * exp2(e)``; that is the
  lowering here as well, so a fractional exponent keeps its meaning.

A non-floating-point or half value, a component-count mismatch, a part that is not a single assignable value, and a
part type with no conversion are refused rather than emitted as a call that does not compile.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    IdentifierNode,
    MemberAccessNode,
    VariableNode,
)

__all__ = ["POINTER_MATH_FUNCTIONS", "generate_pointer_math_call"]

# CrossGL name -> (C function stem, value operands, output kinds, output names, returns a value). An output of kind
# "int" is written as an int; one of kind "value" has the value's component type.
_PART_FUNCTIONS = {
    "frexp": ("frexp", 1, ("int",), ("part",), True),
    "modf": ("modf", 1, ("value",), ("part",), True),
    "remquo": ("remquo", 2, ("int",), ("part",), True),
    "sincos": ("sincos", 1, ("value", "value"), ("sine", "cosine"), False),
    "sincospi": ("sincospi", 1, ("value", "value"), ("sine", "cosine"), False),
}
# CUDA and HIP names of the float functions, which CrossGL programs may call directly.
_FLOAT_SPELLINGS = {
    "frexpf": "frexp",
    "modff": "modf",
    "remquof": "remquo",
    "sincosf": "sincos",
    "sincospif": "sincospi",
    "__sincosf": "sincos",
}
POINTER_MATH_FUNCTIONS = frozenset({"ldexp", *_PART_FUNCTIONS, *_FLOAT_SPELLINGS})

_FLOATS = ("float", "double")
_INTEGERS = ("int", "uint")
_SCALAR_NAMES = {
    "float": "float",
    "float32_t": "float",
    "double": "double",
    "float64_t": "double",
    "int": "int",
    "int32_t": "int",
    "uint": "uint",
    "unsigned int": "uint",
    "uint32_t": "uint",
}
_C_SCALAR = {"float": "float", "double": "double", "int": "int", "uint": "unsigned int"}
_HALF_NAMES = frozenset({"half", "__half", "float16_t", "f16", "half2", "__half2"})


def _type_name(codegen, vtype):
    if vtype is None:
        return None
    if isinstance(vtype, str):
        return vtype.strip()
    return codegen.convert_type_node_to_string(vtype)


def _shape(codegen, node):
    """``(component, lanes, C type, constructor)`` of a scalar or vector operand; ``"half"`` component for half."""
    vtype = codegen.expression_result_type(node)
    name = _type_name(codegen, vtype)
    if not name:
        return None
    vector_info = codegen.vector_type_info(vtype)
    if vector_info is not None:
        return (
            vector_info["component_type"],
            tuple(vector_info["components"]),
            vector_info["type"],
            vector_info["constructor"],
        )
    mapped = codegen.map_vector_arithmetic_type(name)
    if name in _HALF_NAMES or mapped in _HALF_NAMES:
        return ("half", (), name, None)
    component = _SCALAR_NAMES.get(name) or _SCALAR_NAMES.get(mapped)
    if component is None:
        return None
    return (component, (), _C_SCALAR[component], None)


def _c_function(name, component):
    return name if component == "double" else f"{name}f"


def _assignable(codegen, node):
    """Whether ``node`` names one assignable value a C++ reference can bind to."""
    if isinstance(node, (IdentifierNode, VariableNode, ArrayAccessNode)):
        return True
    if isinstance(node, MemberAccessNode):
        owner = getattr(node, "object", None)
        member = str(getattr(node, "member", ""))
        if owner is None or not _assignable(codegen, owner):
            return False
        # A vector swizzle of several components is a temporary in CUDA and HIP, not an lvalue.
        owner_is_vector = (
            codegen.vector_type_info(codegen.expression_result_type(owner)) is not None
        )
        return not owner_is_vector or len(member) == 1
    return False


def _convert(target_component, value):
    return f"static_cast<{_C_SCALAR[target_component]}>({value})"


def _helpers(codegen):
    return codegen.helper_functions


def _part_helper(codegen, func_name, signature, values, parts):
    """Register the helper that calls the C function once per component and return its name."""
    function, _, kinds, names, returns = signature
    component, lanes, value_type, _ = values[0]
    part_types = [part[2] for part in parts]
    helper_name = "_".join(
        [f"cgl_{func_name}", value_type, *(t.replace(" ", "_") for t in part_types)]
    )
    if helper_name in _helpers(codegen):
        return helper_name
    value_names = ("x", "y")[: len(values)]
    parameters = [f"{value_type} {name}" for name in value_names] + [
        f"{part_type}& {name}" for part_type, name in zip(part_types, names)
    ]
    lines = [
        f"__device__ inline {value_type if returns else 'void'} "
        f"{helper_name}({', '.join(parameters)})",
        "{",
    ]
    if returns:
        lines.append(f"    {value_type} result;")
    natives = [_native_component(kind, component) for kind in kinds]
    lines.extend(
        f"    {_C_SCALAR[native]} native_{name};"
        for native, name in zip(natives, names)
    )
    for lane in lanes or (None,):
        member = f".{lane}" if lane else ""
        operands = [f"{name}{member}" for name in value_names]
        operands += [f"&native_{name}" for name in names]
        call = f"{function}({', '.join(operands)})"
        lines.append(f"    result{member} = {call};" if returns else f"    {call};")
        for part, native, name in zip(parts, natives, names):
            converted = (
                f"native_{name}"
                if part[0] == native
                else _convert(part[0], f"native_{name}")
            )
            lines.append(f"    {name}{member} = {converted};")
    if returns:
        lines.append("    return result;")
    lines.append("}")
    _helpers(codegen)[helper_name] = "\n".join(lines)
    return helper_name


def _ldexp_helper(codegen, value, exponent):
    component, lanes, value_type, constructor = value
    exponent_component, exponent_lanes, exponent_type, _ = exponent
    helper_name = f"cgl_ldexp_{value_type}_{exponent_type.replace(' ', '_')}"
    if helper_name in _helpers(codegen):
        return helper_name
    components = []
    for lane in lanes:
        x = f"x.{lane}"
        e = f"e.{lane}" if exponent_lanes else "e"
        if exponent_component in _INTEGERS:
            e = e if exponent_component == "int" else _convert("int", e)
            components.append(f"{_c_function('ldexp', component)}({x}, {e})")
        else:
            e = e if exponent_component == component else _convert(component, e)
            components.append(f"{x} * {_c_function('exp2', component)}({e})")
    _helpers(codegen)[helper_name] = (
        f"__device__ inline {value_type} {helper_name}({value_type} x, {exponent_type} e)\n"
        "{\n"
        f"    return {constructor}({', '.join(components)});\n"
        "}"
    )
    return helper_name


def _ldexp_call(codegen, value, exponent, args):
    component, lanes, _, _ = value
    exponent_component = exponent[0]
    if lanes:
        return f"{_ldexp_helper(codegen, value, exponent)}({args[0]}, {args[1]})", None
    if exponent_component in _INTEGERS:
        e = args[1] if exponent_component == "int" else _convert("int", args[1])
        return f"{_c_function('ldexp', component)}({args[0]}, {e})", None
    e = args[1] if exponent_component == component else _convert(component, args[1])
    return f"({args[0]} * {_c_function('exp2', component)}({e}))", None


def _native_component(kind, component):
    return "int" if kind == "int" else component


def _signature(func_name):
    """``(C function, value operands, part kinds, part names, returns a value)`` and the required value type."""
    if func_name in _FLOAT_SPELLINGS:
        _, *shape = _PART_FUNCTIONS[_FLOAT_SPELLINGS[func_name]]
        return (func_name, *shape), "float"
    return _PART_FUNCTIONS[func_name], None


def _lower_parts(codegen, func_name, raw_args, args):
    """``(spelling, None)`` or ``(None, refusal detail)`` for a function that writes parts through pointers."""
    signature, required = _signature(func_name)
    function, value_count, kinds, _, _ = signature
    if len(raw_args) != value_count + len(kinds) or len(args) != len(raw_args):
        return None, f"it takes {value_count} value and {len(kinds)} output operands"
    shapes = [_shape(codegen, arg) for arg in raw_args]
    values, parts = shapes[:value_count], shapes[value_count:]
    value = values[0]
    if value is None or value[0] not in _FLOATS:
        kind = "a half" if value is not None and value[0] == "half" else "not a"
        return None, f"the value is {kind} float or double scalar or vector"
    if required is not None and value[0] != required:
        return None, f"{func_name} takes a {required} value"
    for other in values[1:]:
        if other is None or other[0] not in _FLOATS + _INTEGERS:
            return None, "a value operand is not a numeric scalar or vector"
        if value[1] and other[:2] != value[:2]:
            return None, "the value operands have different types"
    for part, raw in zip(parts, raw_args[value_count:]):
        if part is None or part[0] not in _FLOATS + _INTEGERS:
            return None, "the second operand is not a numeric scalar or vector"
        if len(part[1]) != len(value[1]):
            return None, (
                f"a {len(part[1]) or 1}-component second operand for a "
                f"{len(value[1]) or 1}-component value"
            )
        if not _assignable(codegen, raw):
            return None, "the output part is not a single assignable value"
    if not value[1] and all(
        part[0] == _native_component(kind, value[0]) for part, kind in zip(parts, kinds)
    ):
        if required is None:
            function = _c_function(function, value[0])
        operands = list(args[:value_count]) + [
            f"&({arg})" for arg in args[value_count:]
        ]
        return f"{function}({', '.join(operands)})", None
    if required is None:
        signature = (_c_function(function, value[0]),) + signature[1:]
    helper_name = _part_helper(codegen, func_name, signature, values, parts)
    return f"{helper_name}({', '.join(args)})", None


def _lower(codegen, func_name, raw_args, args):
    """``(spelling, None)`` or ``(None, refusal detail)``."""
    if func_name != "ldexp":
        return _lower_parts(codegen, func_name, raw_args, args)
    if len(raw_args) != 2 or len(args) != 2:
        return None, "it takes a value and a second operand"
    value = _shape(codegen, raw_args[0])
    exponent = _shape(codegen, raw_args[1])
    if value is None or value[0] not in _FLOATS:
        kind = "a half" if value is not None and value[0] == "half" else "not a"
        return None, f"the value is {kind} float or double scalar or vector"
    if exponent is None or exponent[0] not in _FLOATS + _INTEGERS:
        return None, "the exponent is neither an integer nor a float"
    if exponent[1] and len(exponent[1]) != len(value[1]):
        return None, (
            f"a {len(exponent[1])}-component second operand for a "
            f"{len(value[1]) or 1}-component value"
        )
    return _ldexp_call(codegen, value, exponent, args)


def generate_pointer_math_call(codegen, refusal, func_name, raw_args, args):
    """The CUDA or HIP spelling of a call to one of :data:`POINTER_MATH_FUNCTIONS`; ``None`` for any other call.

    ``refusal`` is the target's :class:`~.lowering_refusals.LoweringRefused` subclass, raised for a call that has no
    faithful lowering.
    """
    if func_name not in POINTER_MATH_FUNCTIONS:
        return None
    spelling, detail = _lower(codegen, func_name, raw_args, args)
    if spelling is None:
        raise refusal([f"{func_name}: {detail}"])
    return spelling
