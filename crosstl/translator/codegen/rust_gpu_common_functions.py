"""GLSL and HLSL common functions without a single Rust method, written as rust-gpu expressions.

Each lowering computes what GLSL 4.60 section 8.3 ("Common Functions") or the HLSL intrinsic reference defines, for
``f32``/``f64`` scalars and glam vectors, with every operand evaluated once:

- ``smoothstep(e0, e1, x)`` is ``t * t * (3 - 2t)`` with ``t = clamp((x - e0) / (e1 - e0), 0, 1)``;
- ``step(edge, x)`` is 0 where ``x < edge`` and 1 elsewhere (glam's ``select`` for vectors);
- ``sign`` of a float is -1, 0 or 1; Rust's ``signum`` returns 1 for zero, so floats compare instead;
- ``fract``/``frac`` is ``x - floor(x)``; Rust's ``f32::fract`` is ``x - trunc(x)``, which differs below zero;
- GLSL ``mod`` is ``x - y * floor(x / y)``; HLSL ``fmod`` takes the sign of ``x``, as Rust's ``%`` on floats does;
- ``saturate`` clamps to [0, 1]; ``mad`` is ``a * b + c``; ``atan2`` is ``atan(y, x)``; ``inverseSqrt`` and
  ``rsqrt`` are ``1 / sqrt(x)``; ``transpose`` is glam's;
- a scalar operand of ``min``, ``max`` and ``clamp`` beside a vector is splatted, as GLSL's ``genType, float``
  overloads define, and ``mix`` with a scalar or ``bool`` weight follows GLSL (``select`` picks ``y`` where the weight
  is true);
- the bit reinterpretations ``floatBitsToInt``/``floatBitsToUint``/``intBitsToFloat``/``uintBitsToFloat`` and HLSL's
  ``asint``/``asuint``/``asfloat`` are ``to_bits`` and ``from_bits`` per component, which rust-gpu writes as
  ``OpBitcast``.

Anything else returns ``None``, so the caller refuses the call.
"""

from __future__ import annotations

_FLOATS = frozenset({"f32", "f64"})
_INTEGERS = frozenset({"i32", "u32"})
_GLAM_PREFIX = {"f32": "Vec", "f64": "DVec", "i32": "IVec", "u32": "UVec", "bool": "BVec"}
_LANES = "xyzw"

COMMON_FUNCTION_NAMES = frozenset(
    {
        "smoothstep",
        "step",
        "sign",
        "isfinite",
        "fract",
        "frac",
        "saturate",
        "mod",
        "fmod",
        "atan2",
        "mad",
        "inverseSqrt",
        "rsqrt",
        "transpose",
        "min",
        "max",
        "clamp",
        "mix",
        "lerp",
        "floatBitsToInt",
        "floatBitsToUint",
        "intBitsToFloat",
        "uintBitsToFloat",
        "asint",
        "asuint",
        "asfloat",
    }
)
# Bit reinterpretations: CrossGL name -> (source scalar, result scalar). HLSL's as* accept any 32-bit source.
_BITCASTS = {
    "floatBitsToInt": ({"f32"}, "i32"),
    "floatBitsToUint": ({"f32"}, "u32"),
    "intBitsToFloat": ({"i32"}, "f32"),
    "uintBitsToFloat": ({"u32"}, "f32"),
    "asint": ({"f32", "i32", "u32"}, "i32"),
    "asuint": ({"f32", "i32", "u32"}, "u32"),
    "asfloat": ({"f32", "i32", "u32"}, "f32"),
}


def _glam(scalar, size):
    return f"{_GLAM_PREFIX[scalar]}{size}"


def _bitcast_scalar(value, source, result):
    if source == result:
        return value
    if source == "f32":
        bits = f"{value}.to_bits()"
        return bits if result == "u32" else f"({bits} as i32)"
    if result == "f32":
        return f"f32::from_bits({value})" if source == "u32" else f"f32::from_bits({value} as u32)"
    return f"({value} as {result})"


def _bitcast(name, shapes, values):
    sources, result = _BITCASTS[name]
    (scalar, size), (value,) = shapes[0], values
    if scalar not in sources:
        return None
    if size == 0:
        return f"({_bitcast_scalar(f'({value})', scalar, result)})"
    lanes = ", ".join(_bitcast_scalar(f"__v.{lane}", scalar, result) for lane in _LANES[:size])
    return f"{{ let __v = {value}; {_glam(result, size)}::new({lanes}) }}"


def _splat(value, shape, scalar, size):
    return value if shape[1] else f"{_glam(scalar, size)}::splat({value})"


def generate_common_function_call(name, shapes, values, operand_types=()):
    """The rust-gpu expression for common function ``name``, or ``None``.

    ``shapes`` holds ``(Rust scalar, vector size or 0)`` per operand, ``values`` the operands written in Rust.
    """
    if name in _BITCASTS:
        return _bitcast(name, shapes, values) if len(values) == 1 else None
    if name == "lerp":
        name = "mix"
    if name == "mix" and len(values) == 3 and shapes[2][0] == "bool":
        return _boolean_mix(shapes, values)
    scalars = {scalar for scalar, _size in shapes}
    sizes = {size for _scalar, size in shapes if size}
    if len(scalars) != 1 or len(sizes) > 1:
        return None
    (scalar,) = scalars
    size = next(iter(sizes), 0)
    if scalar in _INTEGERS:
        return _integer_call(name, shapes, values, scalar, size)
    if scalar not in _FLOATS:
        return None
    return _float_call(name, shapes, values, scalar, size)


def _boolean_mix(shapes, values):
    (x_shape, y_shape, a_shape), (x, y, a) = shapes, values
    if x_shape != y_shape or x_shape[1] != a_shape[1]:
        return None
    if a_shape[1] == 0:
        return f"(if {a} {{ {y} }} else {{ {x} }})"
    return f"{_glam(x_shape[0], x_shape[1])}::select({a}, {y}, {x})"


def _integer_call(name, shapes, values, scalar, size):
    if name == "sign" and len(values) == 1 and scalar == "i32":
        return f"({values[0]}).signum()"
    if name == "mad" and len(values) == 3:
        return f"(({values[0]}) * ({values[1]}) + ({values[2]}))"
    if size and name in {"min", "max"} and len(values) == 2:
        first, second = (_splat(v, s, scalar, size) for v, s in zip(values, shapes))
        return f"({first}).{name}({second})"
    if size and name == "clamp" and len(values) == 3:
        x, low, high = (_splat(v, s, scalar, size) for v, s in zip(values, shapes))
        return f"({x}).clamp({low}, {high})"
    return None


def _float_call(name, shapes, values, scalar, size):
    count = len(values)
    if name == "smoothstep" and count == 3:
        if size and not shapes[2][1]:
            return None
        low, high = ("0.0", "1.0") if not size else (f"{_glam(scalar, size)}::ZERO", f"{_glam(scalar, size)}::ONE")
        e0, e1, x = values
        return (
            f"{{ let __e0 = {e0}; let __e1 = {e1}; "
            f"let __t = (({x} - __e0) / (__e1 - __e0)).clamp({low}, {high}); "
            "__t * __t * (3.0 - 2.0 * __t) }"
        )
    if name == "step" and count == 2:
        edge, x = values
        if not size:
            return f"(if ({x}) < ({edge}) {{ 0.0 }} else {{ 1.0 }})"
        if not shapes[1][1]:
            return None
        vector = _glam(scalar, size)
        edge = _splat(edge, shapes[0], scalar, size)
        return f"{vector}::select(({x}).cmplt({edge}), {vector}::ZERO, {vector}::ONE)"
    if count == 1:
        return _float_unary(name, values[0], scalar, size)
    if name == "mod" and count == 2:
        return f"{{ let __x = {values[0]}; let __y = {values[1]}; __x - __y * (__x / __y).floor() }}"
    if name == "fmod" and count == 2:
        return f"(({values[0]}) % ({values[1]}))"
    if name == "atan2" and count == 2:
        if size:
            return None
        return f"({values[0]}).atan2({values[1]})"
    if name == "mad" and count == 3:
        return f"(({values[0]}) * ({values[1]}) + ({values[2]}))"
    if size and name in {"min", "max"} and count == 2:
        first, second = (_splat(v, s, scalar, size) for v, s in zip(values, shapes))
        return f"({first}).{name}({second})"
    if size and name == "clamp" and count == 3:
        x, low, high = (_splat(v, s, scalar, size) for v, s in zip(values, shapes))
        return f"({x}).clamp({low}, {high})"
    if name == "mix" and count == 3 and size and shapes[0][1] and shapes[1][1]:
        # GLSL mix(genType, genType, float): the weight is a scalar.
        x, y, a = values
        return f"{{ let __a = {a}; ({x}) * (1.0 - __a) + ({y}) * __a }}"
    return None


def _float_unary(name, value, scalar, size):
    if name == "sign":
        compare = "if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 }"
        if size:
            return f"({value}).map(|c| {compare})"
        return f"{{ let c = {value}; {compare} }}"
    if name == "isfinite":
        return f"({value}).is_finite_mask()" if size else f"({value}).is_finite()"
    if name in {"fract", "frac"}:
        return f"({value}).fract_gl()" if size else f"{{ let __x = {value}; __x - __x.floor() }}"
    if name == "saturate":
        if size:
            vector = _glam(scalar, size)
            return f"({value}).clamp({vector}::ZERO, {vector}::ONE)"
        return f"({value}).clamp(0.0, 1.0)"
    if name in {"inverseSqrt", "rsqrt"}:
        return f"({value}).map(|c| 1.0 / c.sqrt())" if size else f"(1.0 / ({value}).sqrt())"
    return None
