"""How CrossGL names and calls the built-in functions of rust-gpu's spirv-std and of glam.

These are the Rust entries of the tables in :mod:`.spellings`, which describes what each table holds: the functions
of rust-gpu's spirv-std and the methods of glam's types that CrossGL names or calls differently.
"""

from __future__ import annotations

# Rust names a function by its trait or type, as ``Float::ln`` or ``Vec3::lerp``; only the ones that compute what
# the GLSL function does are renamed.
RENAMES = {
    "Float::ln": "log",
    "Float::powf": "pow",
    "Float::mul_add": "fma",
    "Float::to_degrees": "degrees",
    "Float::to_radians": "radians",
    "Float::is_nan": "isnan",
    "Float::is_infinite": "isinf",
    "Float::atan2": "atan",
    "Vec3::ln": "log",
    "Vec3::powf": "pow",
    "Vec3::mul_add": "fma",
    "Vec3::lerp": "mix",
    "Vec3::fract_gl": "fract",
    "Vec3::is_nan_mask": "isnan",
    "Vec3::cmpeq": "equal",
    "Vec3::cmpne": "notEqual",
    "Vec3::cmplt": "lessThan",
    "Vec3::cmple": "lessThanEqual",
    "Vec3::cmpgt": "greaterThan",
    "Vec3::cmpge": "greaterThanEqual",
    # spirv_std::float's packing functions are the GLSL.std.450 instructions of GLSL's packing functions.
    "vec2_to_f16x2": "packHalf2x16",
    "f16x2_to_vec2": "unpackHalf2x16",
    "vec4_to_u8x4_snorm": "packSnorm4x8",
    "vec4_to_u8x4_unorm": "packUnorm4x8",
    "vec2_to_u16x2_snorm": "packSnorm2x16",
    "vec2_to_u16x2_unorm": "packUnorm2x16",
    "u8x4_to_vec4_snorm": "unpackSnorm4x8",
    "u8x4_to_vec4_unorm": "unpackUnorm4x8",
    "u16x2_to_vec2_snorm": "unpackSnorm2x16",
    "u16x2_to_vec2_unorm": "unpackUnorm2x16",
    "Derivative::dfdx": "dFdx",
    "Derivative::dfdy": "dFdy",
    "Derivative::fwidth": "fwidth",
    "Derivative::dfdx_fine": "dFdxFine",
    "Derivative::dfdy_fine": "dFdyFine",
    "Derivative::fwidth_fine": "fwidthFine",
    "Derivative::dfdx_coarse": "dFdxCoarse",
    "Derivative::dfdy_coarse": "dFdyCoarse",
    "Derivative::fwidth_coarse": "fwidthCoarse",
    "workgroup_memory_barrier": "memoryBarrierShared",
    "workgroup_memory_barrier_with_group_sync": "barrier",
    "device_memory_barrier": "memoryBarrierBuffer",
    "all_memory_barrier": "memoryBarrier",
    # spirv-std computes these with GLSL.std.450 SMax, SMin, UMax and UMin; CrossGL's operand types select them.
    "signed_max": "max",
    "signed_min": "min",
    "unsigned_max": "max",
    "unsigned_min": "min",
    "atomic_i_add": "atomicAdd",
    "atomic_f_add": "atomicAdd",
    "atomic_s_min": "atomicMin",
    "atomic_u_min": "atomicMin",
    "atomic_f_min": "atomicMin",
    "atomic_s_max": "atomicMax",
    "atomic_u_max": "atomicMax",
    "atomic_f_max": "atomicMax",
    "atomic_and": "atomicAnd",
    "atomic_or": "atomicOr",
    "atomic_xor": "atomicXor",
    "atomic_exchange": "atomicExchange",
    "subgroup_all": "allInvocations",
    "subgroup_any": "anyInvocation",
    "subgroup_all_equal": "allInvocationsEqual",
    "emit_vertex": "EmitVertex",
    "end_primitive": "EndPrimitive",
}

# rust-gpu names one subgroup function for each operation and operand kind, as ``subgroup_i_add`` and
# ``subgroup_f_add``; in CrossGL the operand's type selects the instruction, and ``bool`` operands make the bitwise
# operations logical. Each operator maps to its wave name and the operand kinds rust-gpu spells it for, with their
# types.
_INTEGER = {"i_": "uint", "f_": "float"}
_ORDERED = {"s_": "int", "u_": "uint", "f_": "float"}
_BITWISE = {"": "uint", "logical_": "bool"}
_SUBGROUP_OPERATORS = {
    "add": ("Sum", _INTEGER),
    "mul": ("Product", _INTEGER),
    "min": ("Min", _ORDERED),
    "max": ("Max", _ORDERED),
    "and": ("BitAnd", _BITWISE),
    "or": ("BitOr", _BITWISE),
    "xor": ("BitXor", _BITWISE),
}
WAVES = {
    f"subgroup_{scan}{kind}{operator}": prefix + wave
    for operator, (wave, kinds) in _SUBGROUP_OPERATORS.items()
    for kind in kinds
    for scan, prefix in (
        ("", "WaveActive"),
        ("inclusive_", "WavePrefixInclusive"),
        (
            "exclusive_",
            ("WavePrefix" if operator in ("add", "mul") else "WavePrefixExclusive"),
        ),
    )
} | {
    "subgroup_elect": "WaveIsFirstLane",
    "subgroup_broadcast_first": "WaveReadLaneFirst",
    "subgroup_shuffle": "WaveReadLaneAt",
    "subgroup_shuffle_xor": "WaveShuffleXor",
    "subgroup_shuffle_up": "WaveShuffleUp",
    "subgroup_shuffle_down": "WaveShuffleDown",
    "subgroup_quad_broadcast": "QuadReadLaneAt",
    "subgroup_ballot": "WaveActiveBallot",
    "subgroup_inverse_ballot": "WaveInverseBallot",
    "subgroup_ballot_bit_extract": "WaveBallotBitExtract",
    "subgroup_ballot_bit_count": "WaveBallotBitCount",
    "subgroup_ballot_inclusive_bit_count": "WaveBallotInclusiveBitCount",
    "subgroup_ballot_exclusive_bit_count": "WaveBallotExclusiveBitCount",
    "subgroup_ballot_find_lsb": "WaveBallotFindLSB",
    "subgroup_ballot_find_msb": "WaveBallotFindMSB",
    # HLSL has no subgroup barriers, so CrossGL spells them as GL_KHR_shader_subgroup does.
    "subgroup_barrier": "subgroupBarrier",
    "subgroup_memory_barrier": "subgroupMemoryBarrier",
    "subgroup_memory_barrier_buffer": "subgroupMemoryBarrierBuffer",
    "subgroup_memory_barrier_image": "subgroupMemoryBarrierImage",
    "subgroup_memory_barrier_shared": "subgroupMemoryBarrierShared",
}


# Rust returns the sine and the cosine as a tuple, where CrossGL, like HLSL, writes them to output parameters. Rust
# names a geometry stream with a constant generic parameter, which CrossGL, like GLSL, passes as an argument.
# glam's Vec3A is a 3-vector aligned to 16 bytes, as CrossGL's vec3 is in a buffer, so CrossGL converts to it
# with the vec3 constructor, which also builds a vector from an array's three elements. glam's
# extend, truncate and as_dvec3 are vector constructors in CrossGL, as in GLSL (GLSL 4.60 section 5.4.2): vec4
# appends an element, vec2 keeps the first two and dvec3 widens each one.
COUNTERPARTS = {
    "Float::sin_cos": "void sincos(float, out float, out float)",
    "Vec3::sin_cos": "void sincos(vec3, out vec3, out vec3)",
    "Vec3::to_vec3a": "vec3 vec3(vec3)",
    "Vec3::from_array": "vec3 vec3(float, float, float)",
    "Vec3::extend": "vec4 vec4(vec3, float)",
    "Vec3::truncate": "vec2 vec2(vec3)",
    "Vec3::as_dvec3": "dvec3 dvec3(vec3)",
    "emit_stream_vertex": "void EmitStreamVertex(const int)",
    "end_stream_primitive": "void EndStreamPrimitive(const int)",
}
# rust-gpu takes a clustered reduction's cluster size as a constant generic parameter, where CrossGL takes it as a
# constant argument.
COUNTERPARTS.update(
    {
        f"subgroup_clustered_{kind}{operator}": (
            f"{scalar} WaveClustered{wave}({scalar}, const uint)"
        )
        for operator, (wave, kinds) in _SUBGROUP_OPERATORS.items()
        for kind, scalar in kinds.items()
    }
)


# Built-in functions CrossGL has under a name the GLSL feature list lacks, or writes as an expression.
OTHER_NAMES = {
    # The C math functions of CUDA's device library and MSL compute what these methods of a scalar do. glam's
    # Vec3::is_finite tests every element at once, and is_finite_mask each one.
    "Float::exp_m1": "expm1",
    "Float::ln_1p": "log1p",
    # rust-gpu computes cbrt as GLSL.std.450 Pow with the exponent 1/3 (rustc_codegen_spirv's
    # builder/libm_intrinsics.rs), which is undefined for a negative base, where C's cbrt is negative.
    "Float::cbrt": "pow({0}, 1.0 / 3.0)",
    "Float::is_finite": "isfinite",
    "Float::is_normal": "isnormal",
    "Float::is_sign_negative": "signbit",
    "Vec3::is_finite_mask": "isfinite",
    # spirv-std documents these as exact implementations of the HLSL barriers.
    "all_memory_barrier_with_group_sync": "AllMemoryBarrierWithGroupSync",
    "device_memory_barrier_with_group_sync": "DeviceMemoryBarrierWithGroupSync",
    # The probe passes memory_barrier subgroup scope and acquire-release semantics on uniform memory
    # (features/rust.py), the OpMemoryBarrier that GL_KHR_shader_subgroup's subgroupMemoryBarrierBuffer is.
    "memory_barrier": "subgroupMemoryBarrierBuffer",
    # OpKill, and the ray tracing and mesh shading instructions GL_EXT_ray_tracing and GL_EXT_mesh_shader name.
    "kill": "discard",
    "report_intersection": "reportIntersectionEXT",
    "ignore_intersection": "ignoreIntersectionEXT",
    "terminate_ray": "terminateRayEXT",
    "emit_mesh_tasks_ext": "EmitMeshTasksEXT",
    "set_mesh_outputs_ext": "SetMeshOutputsEXT",
    # OpEmitMeshTasksEXT with a payload operand, which CrossGL writes as DXR's DispatchMesh with the payload last.
    "emit_mesh_tasks_ext_payload": "DispatchMesh",
    # OpExecuteCallableKHR, which CrossGL calls by DXR's name, CallShader, with the shader index first. The probe
    # passes execute_callable the index 0 as its constant generic parameter.
    "execute_callable": "CallShader(0u, {0})",
    # A lane given as a constant broadcasts, as for Slang's WaveBroadcastLaneAt. subgroup_broadcast_const takes
    # the lane as a constant generic parameter, which the probe sets to 0.
    "subgroup_broadcast": "subgroupBroadcast",
    "subgroup_broadcast_const": "subgroupBroadcast({0}, 0u)",
    # The probe swaps in direction 0, OpGroupNonUniformQuadSwap's horizontal swap, which is HLSL's
    # QuadReadAcrossX.
    "subgroup_quad_swap": "QuadReadAcrossX",
    # spirv-std passes the new value before the comparator; CrossGL's atomicCompSwap, like GLSL's, takes the
    # comparator first. Both return the original value.
    "atomic_compare_exchange": "atomicCompSwap({0}, {2}, {1})",
    # glam's x.smoothstep(edge0, edge1) interpolates its receiver (glam 0.34, src/f32/vec3.rs); CrossGL's
    # smoothstep, like GLSL's, takes the edges first.
    "Vec3::smoothstep": "smoothstep({1}, {2}, {0})",
    # OpIsHelperInvocationEXT, and OpReadClockKHR at the subgroup scope the probe passes, as a 64-bit integer or
    # as two 32-bit halves (GL_ARB_shader_clock).
    "is_helper_invocation": "IsHelperLane",
    "read_clock_khr": "clockARB",
    "read_clock_uvec2_khr": "clock2x32ARB",
    # num-traits computes recip as 1 / x and glam's Vec3::recip divides each element the same way, as HLSL's
    # rcp does. rust-gpu lowers powi to GLSL.std.450 Pow, CrossGL's pow, which takes a float exponent.
    "Float::recip": "rcp",
    "Vec3::recip": "rcp",
    "Float::powi": "pow({0}, float({1}))",
    # spirv-std converts one float to a half through the first element of PackHalf2x16, with a zero second
    # element, and back through the first element of UnpackHalf2x16 (crates/spirv-std/src/float.rs).
    "f32_to_f16": "packHalf2x16(vec2({0}, 0.0))",
    "f16_to_f32": "unpackHalf2x16({0}).x",
    # Rust's fract truncates, where CrossGL's fract floors, and num-traits' log takes a base: num-traits computes
    # them as self - trunc(self) and self.ln() / base.ln() (src/float.rs), and glam's Vec3::fract as
    # self - self.trunc() (src/f32/vec3.rs).
    "Float::fract": "({0} - trunc({0}))",
    "Vec3::fract": "({0} - trunc({0}))",
    "Float::log": "(log({0}) / log({1}))",
    # core's sign tests and constants for f32, which num-traits' Float returns: is_sign_positive is the
    # negation of is_sign_negative, a subnormal is a non-zero value smaller in magnitude than MIN_POSITIVE,
    # and signum is NaN for NaN and otherwise 1.0 with the operand's sign.
    "Float::is_sign_positive": "!signbit({0})",
    "Float::is_subnormal": "({0} != 0.0 && abs({0}) < 1.17549435e-38)",
    "Float::signum": "(isnan({0}) ? {0} : copysign(1.0, {0}))",
    "Float::max_value": "(3.40282347e38)",
    "Float::min_value": "(-3.40282347e38)",
    "Float::min_positive_value": "(1.17549435e-38)",
    "Float::neg_infinity": "(-INFINITY)",
    "Float::neg_zero": "(-0.0)",
    # glam's Vec3 methods that glam writes in Rust arithmetic (glam 0.34, src/f32/vec3.rs), written with the same
    # CrossGL operations. glam's length_squared is dot(v, v). Its max_element and min_element compare with > and
    # <, which pick the same element as max and min unless an element is NaN.
    "Vec3::element_sum": "({0}.x + {0}.y + {0}.z)",
    "Vec3::element_product": "({0}.x * {0}.y * {0}.z)",
    "Vec3::max_element": "max({0}.x, max({0}.y, {0}.z))",
    "Vec3::min_element": "min({0}.x, min({0}.y, {0}.z))",
    # max_position and min_position keep the first of equal elements: glam replaces its candidate only when a
    # later element compares greater or less.
    "Vec3::max_position": (
        "({0}.z > ({0}.y > {0}.x ? {0}.y : {0}.x) ? 2u : ({0}.y > {0}.x ? 1u : 0u))"
    ),
    "Vec3::min_position": (
        "({0}.z < ({0}.y < {0}.x ? {0}.y : {0}.x) ? 2u : ({0}.y < {0}.x ? 1u : 0u))"
    ),
    # A slice is the probe's whole input buffer, whose first three elements from_slice reads.
    "Vec3::from_slice": "vec3({0}[0], {0}[1], {0}[2])",
    "Vec3::with_x": "vec3({1}, {0}.y, {0}.z)",
    "Vec3::with_y": "vec3({0}.x, {1}, {0}.z)",
    "Vec3::with_z": "vec3({0}.x, {0}.y, {1})",
    "Vec3::midpoint": "(({0} + {1}) * 0.5)",
    "Vec3::lerp_monotonic": "({0} + ({1} - {0}) * {2})",
    "Vec3::length_recip": "(1.0 / length({0}))",
    "Vec3::dot_into_vec": "vec3(dot({0}, {1}))",
    "Vec3::to_homogeneous": "vec4({0}, 1.0)",
    "Vec3::from_homogeneous": "({0}.xyz / {0}.w)",
    "Vec3::as_ivec3": "ivec3({0})",
    "Vec3::as_uvec3": "uvec3({0})",
    "Vec3::is_finite": "all(isfinite({0}))",
    "Vec3::is_nan": "any(isnan({0}))",
    "Vec3::is_normalized": "(abs(dot({0}, {0}) - 1.0) <= 0.0002)",
    "Vec3::is_negative_mask": "bvec3(signbit({0}.x), signbit({0}.y), signbit({0}.z))",
    "Vec3::is_negative_bitmask": (
        "(uint(signbit({0}.x)) | (uint(signbit({0}.y)) << 1u) | (uint(signbit({0}.z)) << 2u))"
    ),
    "Vec3::signum": (
        "vec3(isnan({0}.x) ? {0}.x : copysign(1.0, {0}.x), isnan({0}.y) ? {0}.y : copysign(1.0, {0}.y), "
        "isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z))"
    ),
    "Vec3::abs_diff_eq": "all(lessThanEqual(abs({0} - {1}), vec3({2})))",
    "Vec3::project_onto": "({1} * (dot({0}, {1}) / dot({1}, {1})))",
    "Vec3::project_onto_normalized": "({1} * dot({0}, {1}))",
    "Vec3::reject_from": "({0} - {1} * (dot({0}, {1}) / dot({1}, {1})))",
    "Vec3::reject_from_normalized": "({0} - {1} * dot({0}, {1}))",
    "Vec3::normalize_or": (
        "(isfinite(1.0 / length({0})) && 1.0 / length({0}) > 0.0 ? {0} * (1.0 / length({0})) : {1})"
    ),
    "Vec3::normalize_or_zero": (
        "(isfinite(1.0 / length({0})) && 1.0 / length({0}) > 0.0 ? {0} * (1.0 / length({0})) : vec3(0.0))"
    ),
    "Vec3::clamp_length_max": (
        "(dot({0}, {0}) > {1} * {1} ? {1} * ({0} / sqrt(dot({0}, {0}))) : {0})"
    ),
    "Vec3::clamp_length_min": (
        "(dot({0}, {0}) < {1} * {1} ? {1} * ({0} / sqrt(dot({0}, {0}))) : {0})"
    ),
    "Vec3::clamp_length": (
        "(dot({0}, {0}) < {1} * {1} ? {1} * ({0} / sqrt(dot({0}, {0}))) "
        ": (dot({0}, {0}) > {2} * {2} ? {2} * ({0} / sqrt(dot({0}, {0}))) : {0}))"
    ),
    "Vec3::move_towards": (
        "(length({1} - {0}) <= {2} || length({1} - {0}) <= 0.0001 ? {1} "
        ": {0} + ({1} - {0}) / length({1} - {0}) * {2})"
    ),
    "Vec3::angle_to": "atan(dot(cross({0}, {1}), {2}), dot({0}, {1}))",
    "Vec3::any_orthogonal_vector": (
        "(abs({0}.x) > abs({0}.y) ? vec3(-{0}.z, 0.0, {0}.x) : vec3(0.0, {0}.z, -{0}.y))"
    ),
    # glam's no_std build, the one rust-gpu compiles, divides and takes the remainder per element with Rust's
    # truncating %, CrossGL's fmod: div_euclid corrects trunc(a / b) by one away from b when the remainder is
    # negative, and rem_euclid adds |b| to a negative remainder (glam src/f32/math.rs, libm_math).
    "Vec3::div_euclid": (
        "vec3("
        + ", ".join(
            f"(fmod({{0}}.{c}, {{1}}.{c}) < 0.0 ? trunc({{0}}.{c} / {{1}}.{c}) + ({{1}}.{c} > 0.0 ? -1.0 : 1.0)"
            f" : trunc({{0}}.{c} / {{1}}.{c}))"
            for c in "xyz"
        )
        + ")"
    ),
    "Vec3::rem_euclid": (
        "vec3("
        + ", ".join(
            f"(fmod({{0}}.{c}, {{1}}.{c}) < 0.0 ? fmod({{0}}.{c}, {{1}}.{c}) + abs({{1}}.{c})"
            f" : fmod({{0}}.{c}, {{1}}.{c}))"
            for c in "xyz"
        )
        + ")"
    ),
    # Pixar's orthonormal basis, as glam writes it: with s = signum(z) and a = -1 / (s + z), the vector is
    # (x * y * a, s + y * y * a, -y).
    "Vec3::any_orthonormal_vector": (
        "vec3({0}.x * {0}.y * (-1.0 / ((isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z)) + {0}.z)), "
        "(isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z)) + {0}.y * {0}.y "
        "* (-1.0 / ((isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z)) + {0}.z)), -{0}.y)"
    ),
    "Vec3::rotate_x": (
        "vec3({0}.x, {0}.y * cos({1}) - {0}.z * sin({1}), {0}.y * sin({1}) + {0}.z * cos({1}))"
    ),
    "Vec3::rotate_y": (
        "vec3({0}.x * cos({1}) + {0}.z * sin({1}), {0}.y, {0}.x * -sin({1}) + {0}.z * cos({1}))"
    ),
    "Vec3::rotate_z": (
        "vec3({0}.x * cos({1}) - {0}.y * sin({1}), {0}.x * sin({1}) + {0}.y * cos({1}), {0}.z)"
    ),
}
