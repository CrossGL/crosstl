"""CrossGL built-ins outside the GLSL core set that rust-gpu computes: C and Metal math functions, the matrix,
geometric and packing functions, float constants, the subgroup barriers and the helper-invocation test.

Each lowering computes what CrossGL defines for the operand types, with every operand evaluated once:

- ``expm1``, ``log1p``, ``log10`` and ``rcp`` are num-traits' ``Float`` methods ``exp_m1``, ``ln_1p``, ``log10``
  and ``recip``, which rust-gpu computes through its libm intrinsics
  (rustc_codegen_spirv/src/builder/libm_intrinsics.rs); a vector applies them to each component with glam's
  ``map``, except ``recip``, which glam defines;
- ``cbrt`` is the real cube root, negative for a negative operand as C's is. rust-gpu writes ``Float::cbrt`` as
  GLSL.std.450 ``Pow`` with exponent 1/3, which is undefined for a negative base (GLSL.std.450 section 2.1), so the
  generator raises the magnitude and gives the result the operand's sign: ``|x|.powf(1 / 3).copysign(x)``;
- ``copysign(x, y)`` is ``Float::copysign``, and glam's ``copysign`` for vectors; ``signbit(x)`` is
  ``is_sign_negative``, a ``BVecN`` of each component's for a vector;
- ``pow`` of a float vector and a scalar exponent is glam's ``powf``;
- ``determinant`` and ``inverse`` of a square float matrix are glam's (GLSL 4.60 section 8.6);
- ``refract(I, N, eta)`` is GLSL's formula (section 8.5): with ``k = 1 - eta * eta * (1 - dot(N, I)^2)``, zero
  where ``k < 0`` and ``eta * I - (eta * dot(N, I) + sqrt(k)) * N`` elsewhere;
- ``select`` picks ``t`` where the condition holds and ``f`` elsewhere. The condition is the ``bool`` operand: first
  in ``select(c, t, f)``, HLSL's order, or last in ``select(f, t, c)``, Metal's (MSL 4.0 section 6.4); a scalar
  condition picks a whole operand and a vector one each component (glam's ``select``). Three ``bool`` operands do
  not say which order is meant, and are refused;
- ``length_squared(v)`` is ``dot(v, v)`` and ``distance_squared(a, b)`` that of ``a - b`` (MSL section 6.13);
- ``absdiff(x, y)`` of integers is ``|x - y|`` without overflow, in the unsigned type of the operands (MSL section
  6.4), which Rust's ``abs_diff`` returns;
- ``sincos(x, s, c)`` writes the sine of ``x`` to ``s`` and its cosine to ``c`` (``Float::sin_cos``);
- GLSL's packing functions (section 8.4) are ``spirv_std::float``'s, which compute the GLSL.std.450 instructions
  of the same names: ``packHalf2x16`` is ``vec2_to_f16x2``, ``unpackSnorm4x8`` is ``u8x4_to_vec4_snorm``;
- ``nan()``, ``infinity()`` and ``epsilon()`` are ``f32``'s ``NAN``, ``INFINITY`` and ``EPSILON``;
- ``subgroupBarrier``, ``subgroupMemoryBarrier`` and its ``Buffer``, ``Shared`` and ``Image`` forms are spirv_std's
  functions of the same names (crates/spirv-std/src/arch/subgroup.rs), the GL_KHR_shader_subgroup barriers;
  ``IsHelperLane()`` is ``is_helper_invocation`` (OpIsHelperInvocationEXT), in fragment shaders only.

Other operand types return ``None``, and the caller refuses the call.
"""

from __future__ import annotations

import re

__all__ = [
    "RUST_GPU_MATH_CALL_NAMES",
    "generate_rust_gpu_math_call",
    "rust_gpu_math_mutated_roots",
    "rust_gpu_math_result_type",
]

_FLOATS = frozenset({"f32", "f64"})
_SIGNED = frozenset({"i8", "i16", "i32", "i64"})
_UNSIGNED = frozenset({"u8", "u16", "u32", "u64"})
# The unsigned type of each integer type: absdiff's result.
_UNSIGNED_OF = {
    **{f"i{bits}": f"u{bits}" for bits in (8, 16, 32, 64)},
    **{f"u{bits}": f"u{bits}" for bits in (8, 16, 32, 64)},
}
_GLAM_PREFIX = {
    "f32": "Vec",
    "f64": "DVec",
    "i32": "IVec",
    "u32": "UVec",
    "bool": "BVec",
}
_LANES = "xyzw"
# A numeric literal without a type suffix, possibly negated and parenthesized: ``3``, ``(-3)``, ``2.5e3``.
_UNTYPED_LITERAL = re.compile(
    r"[(\s]*-?[(\s]*\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?[)\s]*"
)
_CROSSGL_SCALARS = {
    "f32": "float",
    "f64": "double",
    "i32": "int",
    "u32": "uint",
    "i64": "int64_t",
    "u64": "uint64_t",
}

# num-traits Float methods of one operand; glam defines the ones marked for vectors too.
_FLOAT_METHODS = {
    "expm1": "exp_m1",
    "log1p": "ln_1p",
    "log10": "log10",
    "rcp": "recip",
}
_GLAM_VECTOR_METHODS = frozenset({"recip"})
# GLSL packing functions: CrossGL name -> (spirv_std::float function, operand vector size or 0 for a uint).
_PACKING = {
    "packHalf2x16": ("vec2_to_f16x2", 2),
    "packSnorm2x16": ("vec2_to_u16x2_snorm", 2),
    "packUnorm2x16": ("vec2_to_u16x2_unorm", 2),
    "packSnorm4x8": ("vec4_to_u8x4_snorm", 4),
    "packUnorm4x8": ("vec4_to_u8x4_unorm", 4),
    "unpackHalf2x16": ("f16x2_to_vec2", 0),
    "unpackSnorm2x16": ("u16x2_to_vec2_snorm", 0),
    "unpackUnorm2x16": ("u16x2_to_vec2_unorm", 0),
    "unpackSnorm4x8": ("u8x4_to_vec4_snorm", 0),
    "unpackUnorm4x8": ("u8x4_to_vec4_unorm", 0),
}
_PACKING_RESULTS = {
    "unpackHalf2x16": "vec2",
    "unpackSnorm2x16": "vec2",
    "unpackUnorm2x16": "vec2",
    "unpackSnorm4x8": "vec4",
    "unpackUnorm4x8": "vec4",
}
_FLOAT_CONSTANTS = {"nan": "NAN", "infinity": "INFINITY", "epsilon": "EPSILON"}
SUBGROUP_BARRIERS = {
    "subgroupBarrier": "spirv_std::arch::subgroup_barrier()",
    "subgroupMemoryBarrier": "spirv_std::arch::subgroup_memory_barrier()",
    "subgroupMemoryBarrierBuffer": "spirv_std::arch::subgroup_memory_barrier_buffer()",
    "subgroupMemoryBarrierShared": "spirv_std::arch::subgroup_memory_barrier_shared()",
    "subgroupMemoryBarrierImage": "spirv_std::arch::subgroup_memory_barrier_image()",
}

RUST_GPU_MATH_CALL_NAMES = frozenset(
    set(_FLOAT_METHODS)
    | {"cbrt"}
    | set(_PACKING)
    | set(_FLOAT_CONSTANTS)
    | set(SUBGROUP_BARRIERS)
    | {
        "copysign",
        "signbit",
        "determinant",
        "inverse",
        "refract",
        "select",
        "length_squared",
        "distance_squared",
        "absdiff",
        "sincos",
        "IsHelperLane",
    }
)


def _glam(scalar, size):
    return f"{_GLAM_PREFIX[scalar]}{size}"


def _method(receiver, method, *arguments):
    return f"({receiver}).{method}({', '.join(arguments)})"


def typed_receiver(value, scalar):
    """``value`` cast to ``scalar`` when it is an unsuffixed literal, whose type Rust cannot infer for a method call
    (E0689)."""
    if _UNTYPED_LITERAL.fullmatch(value):
        return f"({value} as {scalar})"
    return value


def _shape(generator, arg):
    """``(Rust scalar, vector size or 0)`` of an operand; ``(None, 0)`` when it is neither a scalar nor a vector."""
    type_name = generator.expression_result_type(arg)
    info = generator.vector_type_info(type_name)
    if info is not None:
        scalar = generator.normalize_scalar_type(info["component_type"])
        return scalar, info["size"]
    return generator.normalize_scalar_type(type_name), 0


def _is_float_matrix(generator, arg):
    info = generator.matrix_type_info(generator.expression_result_type(arg))
    if info is None:
        return False
    rows, columns = info.get("rows"), info.get("cols", info.get("columns"))
    scalar = generator.normalize_scalar_type(info.get("component_type", "float"))
    return rows == columns and rows in (2, 3, 4) and scalar in _FLOATS


def generate_rust_gpu_math_call(generator, name, args):
    """The rust-gpu expression for the CrossGL built-in ``name(args)``, or ``None`` when it has none."""
    if name not in RUST_GPU_MATH_CALL_NAMES or generator.is_user_defined_function(name):
        return None
    args = list(args or [])
    if name in SUBGROUP_BARRIERS:
        return SUBGROUP_BARRIERS[name] if not args else None
    if name == "IsHelperLane":
        if args or not generator.rust_gpu_derivatives_allowed():
            return None
        return "spirv_std::arch::is_helper_invocation()"
    if name in _FLOAT_CONSTANTS:
        return f"f32::{_FLOAT_CONSTANTS[name]}" if not args else None
    if name in {"determinant", "inverse"}:
        if len(args) != 1 or not _is_float_matrix(generator, args[0]):
            return None
        return _method(generator.generate_expression(args[0]), name)
    shapes = [_shape(generator, arg) for arg in args]
    if not args or any(scalar is None for scalar, _size in shapes):
        return None
    values = [generator.generate_expression(arg) for arg in args]
    if name in _PACKING:
        return _packing_call(name, shapes, values)
    if name == "select":
        return _select_call(shapes, values)
    if name == "sincos":
        return _sincos_call(shapes, values)
    if name == "absdiff":
        return _absdiff_call(shapes, values)
    scalars = {scalar for scalar, _size in shapes}
    sizes = {size for _scalar, size in shapes}
    if len(scalars) != 1 or not scalars <= _FLOATS:
        return None
    (scalar,) = scalars
    if name == "refract":
        return _refract_call(shapes, values, scalar)
    if len(sizes) != 1:
        return None
    (size,) = sizes
    receiver = typed_receiver(values[0], scalar) if values else None
    if name in _FLOAT_METHODS and len(values) == 1:
        method = _FLOAT_METHODS[name]
        if size and method not in _GLAM_VECTOR_METHODS:
            return _method(values[0], "map", f"|c| c.{method}()")
        return _method(receiver, method)
    if name == "cbrt" and len(values) == 1:
        root = "c.abs().powf(1.0 / 3.0).copysign(c)"
        if size:
            return _method(values[0], "map", f"|c| {root}")
        return f"{{ let c: {scalar} = {values[0]}; {root} }}"
    if name == "copysign" and len(values) == 2:
        return _method(receiver, "copysign", values[1])
    if name == "signbit" and len(values) == 1:
        if not size:
            return _method(receiver, "is_sign_negative")
        lanes = ", ".join(f"__v.{lane}.is_sign_negative()" for lane in _LANES[:size])
        return f"{{ let __v = {values[0]}; BVec{size}::new({lanes}) }}"
    if name == "length_squared" and len(values) == 1:
        if size:
            return _method(values[0], "length_squared")
        return f"{{ let __v = {values[0]}; __v * __v }}"
    if name == "distance_squared" and len(values) == 2:
        if size:
            return _method(values[0], "distance_squared", values[1])
        return f"{{ let __d = ({values[0]}) - ({values[1]}); __d * __d }}"
    return None


def _packing_call(name, shapes, values):
    function, size = _PACKING[name]
    if len(values) != 1:
        return None
    expected = ("f32", size) if size else ("u32", 0)
    if shapes[0] != expected:
        return None
    return f"spirv_std::float::{function}({values[0]})"


def _select_call(shapes, values):
    if len(values) != 3:
        return None
    first_bool, last_bool = shapes[0][0] == "bool", shapes[2][0] == "bool"
    if first_bool and shapes[1][0] != "bool":
        condition, chosen, other = shapes[0], 1, 2
        bindings = (values[0], values[1], values[2])
    elif last_bool and not first_bool:
        condition, chosen, other = shapes[2], 1, 0
        bindings = (values[2], values[1], values[0])
    else:
        return None
    (scalar, size), (other_scalar, other_size) = shapes[chosen], shapes[other]
    if (scalar, size) != (other_scalar, other_size) or scalar == "bool":
        return None
    prefix = "let __c = {}; let __t = {}; let __f = {};".format(*bindings)
    if condition[1] == 0:
        return f"{{ {prefix} if __c {{ __t }} else {{ __f }} }}"
    if condition[1] != size or scalar not in _GLAM_PREFIX:
        return None
    return f"{{ {prefix} {_glam(scalar, size)}::select(__c, __t, __f) }}"


def _sincos_call(shapes, values):
    if len(values) != 3 or len(set(shapes)) != 1 or shapes[0][0] not in _FLOATS:
        return None
    value, sine, cosine = values
    if shapes[0][1]:
        return f"{{ let __x = {value}; {sine} = __x.map(|c| c.sin()); {cosine} = __x.map(|c| c.cos()); }}"
    return f"{{ let (__s, __c) = ({value}).sin_cos(); {sine} = __s; {cosine} = __c; }}"


def _absdiff_call(shapes, values):
    if len(values) != 2 or shapes[0] != shapes[1]:
        return None
    scalar, size = shapes[0]
    if scalar not in _SIGNED | _UNSIGNED:
        return None
    if not size:
        return _method(values[0], "abs_diff", values[1])
    if scalar not in {"i32", "u32"}:
        return None
    lanes = ", ".join(f"__a.{lane}.abs_diff(__b.{lane})" for lane in _LANES[:size])
    return (
        f"{{ let __a = {values[0]}; let __b = {values[1]}; UVec{size}::new({lanes}) }}"
    )


def _refract_call(shapes, values, scalar):
    if len(values) != 3 or shapes[0] != shapes[1] or shapes[2][1]:
        return None
    size = shapes[0][1]
    incident, normal, eta = values
    if size:
        product = "__n.dot(__i)"
        zero = f"{_glam(scalar, size)}::ZERO"
    else:
        product = "__n * __i"
        zero = "0.0"
    return (
        f"{{ let __i = {incident}; let __n = {normal}; let __e = ({eta}) as {scalar}; let __d = {product}; "
        "let __k = 1.0 - __e * __e * (1.0 - __d * __d); "
        f"if __k < 0.0 {{ {zero} }} else {{ __e * __i - (__e * __d + __k.sqrt()) * __n }} }}"
    )


def rust_gpu_math_result_type(generator, name, arguments):
    """The CrossGL type of a call this module lowers, where it is not the first operand's type; else ``None``."""
    if name not in RUST_GPU_MATH_CALL_NAMES or generator.is_user_defined_function(name):
        return None
    arguments = list(arguments or [])
    if name in _FLOAT_CONSTANTS and not arguments:
        return "float"
    if name in _PACKING_RESULTS:
        return _PACKING_RESULTS[name]
    if name in _PACKING:
        return "uint"
    if name == "IsHelperLane":
        return "bool"
    if name in SUBGROUP_BARRIERS or name == "sincos":
        return "void"
    if not arguments:
        return None
    if name in {"length_squared", "distance_squared", "determinant"}:
        scalar, _size = _shape(generator, arguments[0])
        if name == "determinant":
            info = generator.matrix_type_info(
                generator.expression_result_type(arguments[0])
            )
            scalar = generator.normalize_scalar_type(
                (info or {}).get("component_type", "float")
            )
        return _CROSSGL_SCALARS.get(scalar)
    if name == "signbit":
        _scalar, size = _shape(generator, arguments[0])
        return f"bvec{size}" if size else "bool"
    if name == "absdiff":
        scalar, size = _shape(generator, arguments[0])
        unsigned = _UNSIGNED_OF.get(scalar)
        if unsigned is None:
            return None
        if size:
            return f"uvec{size}" if unsigned == "u32" else None
        return _CROSSGL_SCALARS.get(unsigned)
    if name == "select" and len(arguments) == 3:
        # The chosen operands sit second in both of select's orders.
        return generator.expression_result_type(arguments[1])
    return generator.expression_result_type(arguments[0])


def rust_gpu_math_mutated_roots(generator, name, args):
    """The bindings a call this module lowers assigns to, which must be declared ``let mut``: ``sincos``'s outputs."""
    args = list(args or [])
    if name != "sincos" or len(args) != 3 or generator.is_user_defined_function(name):
        return set()
    roots = {generator.assignment_target_root_name(arg) for arg in args[1:]}
    return {root for root in roots if root}
