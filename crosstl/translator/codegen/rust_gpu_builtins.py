"""CrossGL built-in functions written as rust-gpu calls: ``num_traits::Float`` methods, glam methods and spirv_std
functions.

Each CrossGL built-in maps to the Rust function that computes the same thing for the operand types, the pairing the
coverage tables record (``tools/coverage/rust_spellings.py``): ``sqrt(x)`` is ``Float::sqrt``, ``log`` is
``Float::ln``, ``fract`` of a vector is glam's ``fract_gl``, ``min`` of two ``int`` values is spirv_std's
``signed_min``. A vector function glam does not define is applied to each component with glam's ``map``. A built-in
missing here, or one called with operand types it has no Rust form for, returns ``None``, and the generator refuses
the call rather than naming a function rust-gpu does not have.
"""

from __future__ import annotations

_FLOATS = frozenset({"f32", "f64"})
_SIGNED = frozenset({"i32", "i64"})
_UNSIGNED = frozenset({"u32", "u64"})
_VECTOR_SCALARS = {"float": "f32", "double": "f64", "int": "i32", "uint": "u32", "bool": "bool"}

# Float methods with one operand that CrossGL spells as the method's name or the name given here.
_FLOAT_UNARY = {
    name: name
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "exp2",
        "log2",
        "sqrt",
        "abs",
        "floor",
        "ceil",
        "trunc",
        "round",
    )
} | {
    "atan": "atan",
    "log": "ln",
    "degrees": "to_degrees",
    "radians": "to_radians",
    "isnan": "is_nan",
    "isinf": "is_infinite",
}
_FLOAT_BINARY = {"pow": "powf", "atan": "atan2", "min": "min", "max": "max"}
# glam vector methods that compute the CrossGL function component by component, with one operand.
_GLAM_UNARY = {
    "abs": "abs",
    "floor": "floor",
    "ceil": "ceil",
    "round": "round",
    "trunc": "trunc",
    "fract": "fract_gl",
    "exp": "exp",
    "exp2": "exp2",
    "log": "ln",
    "log2": "log2",
    "isnan": "is_nan_mask",
    "normalize": "normalize",
    "length": "length",
}
# glam vector methods with a second vector operand.
_GLAM_BINARY = {
    "min": "min",
    "max": "max",
    "dot": "dot",
    "cross": "cross",
    "distance": "distance",
    "reflect": "reflect",
    "equal": "cmpeq",
    "notEqual": "cmpne",
    "lessThan": "cmplt",
    "lessThanEqual": "cmple",
    "greaterThan": "cmpgt",
    "greaterThanEqual": "cmpge",
}
_INTEGER_VECTOR_METHODS = frozenset({"min", "max", "abs", "equal", "notEqual"}) | {
    "lessThan",
    "lessThanEqual",
    "greaterThan",
    "greaterThanEqual",
}
# spirv_std::arch barriers (crates/spirv-std/src/arch/barrier.rs), with CrossGL's names.
BARRIERS = {
    "barrier": "spirv_std::arch::workgroup_memory_barrier_with_group_sync()",
    "GroupMemoryBarrierWithGroupSync": (
        "spirv_std::arch::workgroup_memory_barrier_with_group_sync()"
    ),
    "memoryBarrierShared": "spirv_std::arch::workgroup_memory_barrier()",
    "GroupMemoryBarrier": "spirv_std::arch::workgroup_memory_barrier()",
    "memoryBarrierBuffer": "spirv_std::arch::device_memory_barrier()",
    "DeviceMemoryBarrier": "spirv_std::arch::device_memory_barrier()",
    "DeviceMemoryBarrierWithGroupSync": (
        "spirv_std::arch::device_memory_barrier_with_group_sync()"
    ),
    "memoryBarrier": "spirv_std::arch::all_memory_barrier()",
    "AllMemoryBarrier": "spirv_std::arch::all_memory_barrier()",
    "AllMemoryBarrierWithGroupSync": (
        "spirv_std::arch::all_memory_barrier_with_group_sync()"
    ),
    "workgroupBarrier": "spirv_std::arch::workgroup_memory_barrier_with_group_sync()",
    "groupMemoryBarrier": "spirv_std::arch::workgroup_memory_barrier()",
    "allMemoryBarrier": "spirv_std::arch::all_memory_barrier()",
    "deviceMemoryBarrier": "spirv_std::arch::device_memory_barrier()",
}

RUST_GPU_BUILTIN_NAMES = frozenset(
    set(_FLOAT_UNARY)
    | set(_FLOAT_BINARY)
    | set(_GLAM_UNARY)
    | set(_GLAM_BINARY)
    | set(BARRIERS)
    | {"fma", "clamp", "mix", "any", "all", "not", "inversesqrt"}
)


def _shape(generator, arg):
    """``(scalar, size)`` of an operand: the Rust scalar type and the vector size, 0 for a scalar."""
    type_name = generator.expression_result_type(arg)
    info = generator.vector_type_info(type_name)
    if info is not None:
        return _VECTOR_SCALARS.get(info["component_type"]), info["size"]
    return generator.normalize_scalar_type(type_name), 0


def _method(receiver: str, method: str, *arguments: str) -> str:
    return f"({receiver}).{method}({', '.join(arguments)})"


def generate_rust_gpu_builtin_call(generator, name, args):
    """The rust-gpu spelling of the CrossGL built-in call ``name(args)``, or ``None`` when it has none."""
    if not isinstance(name, str) or name not in RUST_GPU_BUILTIN_NAMES:
        return None
    if generator.is_user_defined_function(name):
        return None
    if name in BARRIERS:
        return BARRIERS[name] if not args else None
    shapes = [_shape(generator, arg) for arg in args]
    if any(scalar is None for scalar, _size in shapes):
        return None
    values = [generator.generate_expression(arg) for arg in args]
    if all(size == 0 for _scalar, size in shapes):
        return _scalar_call(name, shapes, values)
    return _vector_call(name, shapes, values)


def _scalar_call(name, shapes, values):
    scalars = {scalar for scalar, _size in shapes}
    if len(scalars) != 1:
        return None
    (scalar,) = scalars
    if scalar in _FLOATS:
        if len(values) == 1 and name in _FLOAT_UNARY:
            return _method(values[0], _FLOAT_UNARY[name])
        if len(values) == 2 and name in _FLOAT_BINARY:
            return _method(values[0], _FLOAT_BINARY[name], values[1])
        if name == "fma" and len(values) == 3:
            return _method(values[0], "mul_add", values[1], values[2])
        if name == "clamp" and len(values) == 3:
            return _method(values[0], "clamp", values[1], values[2])
        if name == "mix" and len(values) == 3:
            # GLSL defines mix as x * (1 - a) + y * a.
            x, y, a = values
            return f"(({x}) * (1.0 - ({a})) + ({y}) * ({a}))"
        if name == "inversesqrt" and len(values) == 1:
            return f"(1.0 / ({values[0]}).sqrt())"
        return None
    if scalar in _SIGNED | _UNSIGNED and name in {"min", "max"} and len(values) == 2:
        prefix = "signed" if scalar in _SIGNED else "unsigned"
        return f"spirv_std::arch::{prefix}_{name}({values[0]}, {values[1]})"
    if scalar in _SIGNED and name == "abs" and len(values) == 1:
        return _method(values[0], "wrapping_abs")
    return None


def _vector_call(name, shapes, values):
    scalars = {scalar for scalar, _size in shapes}
    sizes = {size for _scalar, size in shapes if size}
    if len(sizes) != 1:
        return None
    floating = scalars <= _FLOATS and len(scalars) == 1
    integral = len(scalars) == 1 and scalars <= _SIGNED | _UNSIGNED
    boolean = scalars == {"bool"}
    if boolean:
        if name in {"any", "all"} and len(values) == 1:
            return _method(values[0], name)
        if name == "not" and len(values) == 1:
            return f"(!({values[0]}))"
        return None
    if integral and name not in _INTEGER_VECTOR_METHODS:
        return None
    if not (floating or integral):
        return None
    every_vector = all(size for _scalar, size in shapes)
    if len(values) == 1 and name in _GLAM_UNARY:
        return _method(values[0], _GLAM_UNARY[name])
    if len(values) == 2 and name in _GLAM_BINARY and every_vector:
        return _method(values[0], _GLAM_BINARY[name], values[1])
    if not floating:
        return None
    if len(values) == 1 and name in _FLOAT_UNARY:
        return _method(values[0], "map", f"|c| c.{_FLOAT_UNARY[name]}()")
    if name == "pow" and len(values) == 2 and every_vector:
        return _componentwise(shapes[0][0], shapes[0][1], "powf", values)
    if name == "atan" and len(values) == 2 and every_vector:
        return _componentwise(shapes[0][0], shapes[0][1], "atan2", values)
    if name == "fma" and len(values) == 3 and every_vector:
        return _method(values[0], "mul_add", values[1], values[2])
    if name == "clamp" and len(values) == 3 and every_vector:
        return _method(values[0], "clamp", values[1], values[2])
    if name == "mix" and len(values) == 3 and every_vector:
        x, y, a = values
        return f"(({x}) * (1.0 - ({a})) + ({y}) * ({a}))"
    if name == "inversesqrt" and len(values) == 1:
        return _method(values[0], "map", "|c| 1.0 / c.sqrt()")
    return None


def _componentwise(scalar, size, method, values):
    """``method`` of the first operand's components with the second's, each operand evaluated once."""
    lanes = "xyzw"[:size]
    first, second = values
    parts = ", ".join(f"__a.{lane}.{method}(__b.{lane})" for lane in lanes)
    prefix = "D" if scalar == "f64" else ""
    return f"{{ let __a = {first}; let __b = {second}; {prefix}Vec{size}::new({parts}) }}"
