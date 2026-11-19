"""glam's vectors, matrices and quaternions besides ``Vec3``, whose methods :mod:`.rust` reads.

GLSL's ``length`` is one feature for every vector width and scalar type, but glam gives each of its types methods of
its own. So each family of :data:`FAMILIES` counts a method once by name, probed on the first of the family's types
that has it: ``Vec2::perp`` and ``IVec3::wrapping_add`` are features, while ``Vec2::length`` is the ``Vec3::length``
that :mod:`.rust` already measures. The types are read from the glam release spirv-std's lock file resolves, with the
cargo features spirv-std enables (:data:`.rust.GLAM_FEATURES`).

CrossGL holds a quaternion as the ``vec4`` of its ``x``, ``y``, ``z`` and ``w`` components, and glam's 16-byte
aligned ``Vec3A`` and ``Mat3A`` as ``vec3`` and ``mat3``. :data:`SPELLINGS` writes each method CrossGL can compute
with the operations glam computes it with (glam 0.34.1, the ``scalar`` sources the SPIR-V target builds); any other
method has no CrossGL spelling, for the reason :data:`_UNSPELLED` gives or because no CrossGL built-in computes it.
"""

from __future__ import annotations

from dataclasses import replace

from ..model import Feature, Param, Signature, Type
from . import rust

# Each family's types in the order a method takes the first that has it.
FAMILIES: tuple[tuple[tuple[str, Type], ...], ...] = (
    (
        ("Vec3", Type("float", (3,))),
        ("Vec2", Type("float", (2,))),
        ("Vec4", Type("float", (4,))),
        ("Vec3A", Type.other("Vec3A")),
        ("IVec3", Type("int", (3,))),
        ("IVec2", Type("int", (2,))),
        ("IVec4", Type("int", (4,))),
        ("UVec3", Type("uint", (3,))),
        ("UVec2", Type("uint", (2,))),
        ("UVec4", Type("uint", (4,))),
        ("DVec3", Type("double", (3,))),
        ("DVec2", Type("double", (2,))),
        ("DVec4", Type("double", (4,))),
        ("BVec3", Type("bool", (3,))),
        ("BVec2", Type("bool", (2,))),
        ("BVec4", Type("bool", (4,))),
    ),
    (
        ("Mat4", Type("float", (4, 4))),
        ("Mat3", Type("float", (3, 3))),
        ("Mat2", Type("float", (2, 2))),
        ("Mat3A", Type.other("Mat3A")),
        ("DMat4", Type("double", (4, 4))),
        ("DMat3", Type("double", (3, 3))),
        ("DMat2", Type("double", (2, 2))),
    ),
    (
        ("Quat", Type.other("Quat")),
        ("DQuat", Type.other("DQuat")),
    ),
)
# The source of each type in :data:`..upstream.SOURCES`.
SOURCES = {name: f"rust-{name.lower()}" for family in FAMILIES for name, _ in family}
# The CrossGL type of each glam type :func:`.rust.rust_type` keeps opaque.
CROSSGL_TYPES = {
    "Vec3A": Type("float", (3,)),
    "Mat3A": Type("float", (3, 3)),
    "Quat": Type("float", (4,)),
    "DQuat": Type("double", (4,)),
}


def _floats(count: int) -> tuple[Param, ...]:
    """The CrossGL operands of a method that reads ``count`` floats from an array or a slice."""
    return tuple(Param(Type("float")) for _ in range(count))


def _listed(count: int) -> str:
    """Holes for ``count`` arguments."""
    return ", ".join(f"{{{n}}}" for n in range(count))


def _elements(count: int) -> str:
    """The first ``count`` elements of the array argument."""
    return ", ".join(f"{{0}}[{n}]" for n in range(count))


def _matrix(size: int, columns: list[list[str]]) -> str:
    """A CrossGL ``mat{size}`` of the given columns."""
    parts = [f"vec{size}({', '.join(column)})" for column in columns]
    return f"mat{size}({', '.join(parts)})"


def _each_column(size: int, template: str, joiner: str = " && ") -> str:
    """``template`` for each column index ``#`` of a matrix, joined."""
    return joiner.join(template.replace("#", str(n)) for n in range(size))


def _rotation(quat: str) -> list[str]:
    """The three columns of the rotation the quaternion ``quat`` makes, as glam's ``quat_to_axes`` computes them."""
    x, y, z, w = (f"{quat}.{c}" for c in "xyzw")
    xx, xy, xz = (f"({x} * ({c} + {c}))" for c in (x, y, z))
    yy, yz, zz = f"({y} * ({y} + {y}))", f"({y} * ({z} + {z}))", f"({z} * ({z} + {z}))"
    wx, wy, wz = (f"({w} * ({c} + {c}))" for c in (x, y, z))
    return [
        f"vec3(1.0 - ({yy} + {zz}), {xy} + {wz}, {xz} - {wy})",
        f"vec3({xy} - {wz}, 1.0 - ({xx} + {zz}), {yz} + {wx})",
        f"vec3({xz} + {wy}, {yz} - {wx}, 1.0 - ({xx} + {yy}))",
    ]


def _axis_angle(axis: str, angle: str) -> list[list[str]]:
    """The columns of glam's ``Mat4::from_axis_angle``."""
    a = {c: f"{axis}.{c}" for c in "xyz"}
    sin, cos, omc = f"sin({angle})", f"cos({angle})", f"(1.0 - cos({angle}))"
    xy, xz, yz = (f"{a[p]} * {a[q]} * {omc}" for p, q in ("xy", "xz", "yz"))
    return [
        [
            f"{a['x']} * {a['x']} * {omc} + {cos}",
            f"{xy} + {a['z']} * {sin}",
            f"{xz} - {a['y']} * {sin}",
            "0.0",
        ],
        [
            f"{xy} - {a['z']} * {sin}",
            f"{a['y']} * {a['y']} * {omc} + {cos}",
            f"{yz} + {a['x']} * {sin}",
            "0.0",
        ],
        [
            f"{xz} + {a['y']} * {sin}",
            f"{yz} - {a['x']} * {sin}",
            f"{a['z']} * {a['z']} * {omc} + {cos}",
            "0.0",
        ],
        ["0.0", "0.0", "0.0", "1.0"],
    ]


_IDENTITY_W = ["0.0", "0.0", "0.0", "1.0"]
_ROTATED = _rotation("{1}")
_ROTATED_FIRST = _rotation("{0}")
_INVERSE_FINITE = _each_column(4, "all(isfinite(inverse({0})[#]))")

# CrossGL spellings, as the operations glam computes each method with. Each value is a CrossGL name or expression, or
# the expression and the CrossGL operands it takes when they differ from the method's own, as for an array argument.
SPELLINGS: dict[str, str | tuple[str, tuple[Param, ...]]] = {
    # glam's two-dimensional rotations (src/f32/vec2.rs).
    "Vec2::from_angle": "vec2(cos({0}), sin({0}))",
    "Vec2::to_angle": "atan({0}.y, {0}.x)",
    "Vec2::perp": "vec2(-{0}.y, {0}.x)",
    "Vec2::perp_dot": "({0}.x * {1}.y - {0}.y * {1}.x)",
    "Vec2::rotate": (
        "vec2({0}.x * {1}.x - {0}.y * {1}.y, {0}.y * {1}.x + {0}.x * {1}.y)"
    ),
    "Vec2::rotate_angle": (
        "vec2({0}.x * cos({1}) - {0}.y * sin({1}), {0}.y * cos({1}) + {0}.x * sin({1}))"
    ),
    # Conversions are CrossGL's vector constructors, as in GLSL (GLSL 4.60 section 5.4.1).
    "Vec2::as_dvec2": "dvec2",
    "Vec2::as_ivec2": "ivec2",
    "Vec2::as_uvec2": "uvec2",
    "Vec4::as_dvec4": "dvec4",
    "Vec4::as_ivec4": "ivec4",
    "Vec4::as_uvec4": "uvec4",
    "IVec2::as_vec2": "vec2",
    "IVec3::as_vec3": "vec3",
    "IVec3::as_vec3a": "vec3",
    "IVec4::as_vec4": "vec4",
    "Vec3A::from_vec4": "vec3",
    "Vec3A::to_vec3": "vec3",
    # Vec4::project is Vec3::from_homogeneous.
    "Vec4::project": "({0}.xyz / {0}.w)",
    "Vec4::with_w": "vec4({0}.x, {0}.y, {0}.z, {1})",
    # Integer arithmetic wraps in CrossGL, as in GLSL (GLSL 4.60 section 5.9) and SPIR-V. Metal's addsat, subsat and
    # absdiff, which CrossGL calls by those names, are the saturating operations and the unsigned distance of Rust's
    # i32 and u32 (MSL 4.1 section 6.4).
    "IVec3::wrapping_add": "({0} + {1})",
    "IVec3::wrapping_sub": "({0} - {1})",
    "IVec3::wrapping_mul": "({0} * {1})",
    "IVec3::wrapping_div": "({0} / {1})",
    "IVec3::wrapping_add_unsigned": "({0} + ivec3({1}))",
    "IVec3::wrapping_sub_unsigned": "({0} - ivec3({1}))",
    "UVec3::wrapping_add_signed": "({0} + uvec3({1}))",
    "IVec3::saturating_add": "addsat",
    "IVec3::saturating_sub": "subsat",
    "IVec3::manhattan_distance": (
        "(absdiff({0}.x, {1}.x) + absdiff({0}.y, {1}.y) + absdiff({0}.z, {1}.z))"
    ),
    "IVec3::chebyshev_distance": (
        "max(max(absdiff({0}.x, {1}.x), absdiff({0}.y, {1}.y)), absdiff({0}.z, {1}.z))"
    ),
    # BVec3 (src/bool/bvec3.rs): bitmask puts element n in bit n.
    "BVec3::any": "any",
    "BVec3::all": "all",
    "BVec3::bitmask": "(uint({0}.x) | (uint({0}.y) << 1u) | (uint({0}.z) << 2u))",
    "BVec3::test": "{0}[{1}]",
    # glam's matrices are arrays of columns, as CrossGL's are (src/f32/scalar/mat4.rs).
    "Mat4::from_cols": "mat4",
    "Mat4::from_rows": "transpose(mat4({0}, {1}, {2}, {3}))",
    "Mat4::from_cols_array": (f"mat4({_listed(16)})", _floats(16)),
    "Mat4::from_rows_array": (f"transpose(mat4({_listed(16)}))", _floats(16)),
    # A slice is the probe's whole input buffer, whose first elements the method reads.
    "Mat4::from_cols_slice": f"mat4({_elements(16)})",
    "Mat4::from_rows_slice": f"transpose(mat4({_elements(16)}))",
    "Mat4::from_diagonal": _matrix(
        4,
        [
            ["{0}.x", "0.0", "0.0", "0.0"],
            ["0.0", "{0}.y", "0.0", "0.0"],
            ["0.0", "0.0", "{0}.z", "0.0"],
            ["0.0", "0.0", "0.0", "{0}.w"],
        ],
    ),
    "Mat4::from_scale": _matrix(
        4,
        [
            ["{0}.x", "0.0", "0.0", "0.0"],
            ["0.0", "{0}.y", "0.0", "0.0"],
            ["0.0", "0.0", "{0}.z", "0.0"],
            _IDENTITY_W,
        ],
    ),
    "Mat4::from_translation": _matrix(
        4,
        [
            ["1.0", "0.0", "0.0", "0.0"],
            ["0.0", "1.0", "0.0", "0.0"],
            ["0.0", "0.0", "1.0", "0.0"],
            ["{0}", "1.0"],
        ],
    ),
    "Mat4::from_rotation_x": _matrix(
        4,
        [
            ["1.0", "0.0", "0.0", "0.0"],
            ["0.0", "cos({0})", "sin({0})", "0.0"],
            ["0.0", "-sin({0})", "cos({0})", "0.0"],
            _IDENTITY_W,
        ],
    ),
    "Mat4::from_rotation_y": _matrix(
        4,
        [
            ["cos({0})", "0.0", "-sin({0})", "0.0"],
            ["0.0", "1.0", "0.0", "0.0"],
            ["sin({0})", "0.0", "cos({0})", "0.0"],
            _IDENTITY_W,
        ],
    ),
    "Mat4::from_rotation_z": _matrix(
        4,
        [
            ["cos({0})", "sin({0})", "0.0", "0.0"],
            ["-sin({0})", "cos({0})", "0.0", "0.0"],
            ["0.0", "0.0", "1.0", "0.0"],
            _IDENTITY_W,
        ],
    ),
    "Mat4::from_axis_angle": _matrix(4, _axis_angle("{0}", "{1}")),
    "Mat4::from_quat": _matrix(4, [[c, "0.0"] for c in _ROTATED_FIRST] + [_IDENTITY_W]),
    "Mat4::from_rotation_translation": _matrix(
        4, [[c, "0.0"] for c in _rotation("{0}")] + [["{1}", "1.0"]]
    ),
    "Mat4::from_scale_rotation_translation": (
        "mat4("
        + ", ".join(f"vec4({c}, 0.0) * {{0}}.{a}" for c, a in zip(_ROTATED, "xyz"))
        + ", vec4({2}, 1.0))"
    ),
    # GLSL's matrix constructors keep the upper-left elements of a larger matrix, and fill the rest of a larger one from
    # the identity (GLSL 4.60 section 5.4.2), as glam's from_mat3, from_mat4 and from_mat2 do.
    "Mat4::from_mat3": "mat4",
    "Mat4::from_mat3a": "mat4",
    "Mat4::from_mat3_translation": (
        "mat4(vec4({0}[0], 0.0), vec4({0}[1], 0.0), vec4({0}[2], 0.0), vec4({1}, 1.0))"
    ),
    "Mat3::from_mat4": "mat3",
    "Mat3::from_mat2": "mat3",
    "Mat4::col": "{0}[{1}]",
    "Mat4::row": "transpose({0})[{1}]",
    "Mat4::is_finite": _each_column(4, "all(isfinite({0}[#]))"),
    "Mat4::is_nan": _each_column(4, "any(isnan({0}[#]))", " || "),
    "Mat4::transpose": "transpose",
    "Mat4::diagonal": "vec4({0}[0].x, {0}[1].y, {0}[2].z, {0}[3].w)",
    "Mat4::determinant": "determinant",
    "Mat4::inverse": "inverse",
    # glam's inverse_or_zero returns zero for an inverse with an element that is not finite.
    "Mat4::inverse_or_zero": f"({_INVERSE_FINITE} ? inverse({{0}}) : mat4(0.0))",
    "Mat4::project_point3": "(({0} * vec4({1}, 1.0)).xyz / ({0} * vec4({1}, 1.0)).w)",
    "Mat4::transform_point3": "({0} * vec4({1}, 1.0)).xyz",
    "Mat4::transform_vector3": "({0} * vec4({1}, 0.0)).xyz",
    "Mat4::project_point3a": "(({0} * vec4({1}, 1.0)).xyz / ({0} * vec4({1}, 1.0)).w)",
    "Mat4::transform_point3a": "({0} * vec4({1}, 1.0)).xyz",
    "Mat4::transform_vector3a": "({0} * vec4({1}, 0.0)).xyz",
    "Mat4::mul_vec4": "({0} * {1})",
    # A vector on the left multiplies by the transpose (GLSL 4.60 section 5.10), the dot product with each column.
    "Mat4::mul_transpose_vec4": "({1} * {0})",
    "Mat4::mul_mat4": "({0} * {1})",
    "Mat4::add_mat4": "({0} + {1})",
    "Mat4::sub_mat4": "({0} - {1})",
    "Mat4::mul_scalar": "({0} * {1})",
    "Mat4::div_scalar": "({0} / {1})",
    "Mat4::mul_diagonal_scale": (
        "mat4({0}[0] * {1}.x, {0}[1] * {1}.y, {0}[2] * {1}.z, {0}[3] * {1}.w)"
    ),
    "Mat4::recip": "(1.0 / {0})",
    "Mat4::abs_diff_eq": _each_column(
        4, "all(lessThanEqual(abs({0}[#] - {1}[#]), vec4({2})))"
    ),
    "Mat4::abs": "mat4(abs({0}[0]), abs({0}[1]), abs({0}[2]), abs({0}[3]))",
    "Mat4::as_dmat4": "dmat4",
    "Mat3::from_angle": _matrix(
        3,
        [
            ["cos({0})", "sin({0})", "0.0"],
            ["-sin({0})", "cos({0})", "0.0"],
            ["0.0", "0.0", "1.0"],
        ],
    ),
    "Mat3::from_scale_angle_translation": _matrix(
        3,
        [
            ["cos({1}) * {0}.x", "sin({1}) * {0}.x", "0.0"],
            ["-sin({1}) * {0}.y", "cos({1}) * {0}.y", "0.0"],
            ["{2}.x", "{2}.y", "1.0"],
        ],
    ),
    "Mat3::transform_point2": "({0} * vec3({1}, 1.0)).xy",
    "Mat3::transform_vector2": "({0} * vec3({1}, 0.0)).xy",
    "Mat3::mul_vec3": "({0} * {1})",
    "Mat3::mul_vec3a": "({0} * {1})",
    "Mat3::mul_transpose_vec3": "({1} * {0})",
    "Mat3::mul_mat3": "({0} * {1})",
    "Mat3::add_mat3": "({0} + {1})",
    "Mat3::sub_mat3": "({0} - {1})",
    "Mat3::as_dmat3": "dmat3",
    "Mat2::from_scale_angle": _matrix(
        2,
        [
            ["cos({1}) * {0}.x", "sin({1}) * {0}.x"],
            ["-sin({1}) * {0}.y", "cos({1}) * {0}.y"],
        ],
    ),
    "Mat2::mul_vec2": "({0} * {1})",
    "Mat2::mul_transpose_vec2": "({1} * {0})",
    "Mat2::mul_mat2": "({0} * {1})",
    "Mat2::add_mat2": "({0} + {1})",
    "Mat2::sub_mat2": "({0} - {1})",
    "Mat2::as_dmat2": "dmat2",
    "Mat3A::mul_transpose_vec3a": "({1} * {0})",
    "DMat4::as_mat4": "mat4",
    "DMat3::as_mat3": "mat3",
    "DMat2::as_mat2": "mat2",
    # glam's quaternions (src/f32/scalar/quat.rs), with w last.
    "Quat::from_xyzw": "vec4",
    "Quat::from_vec4": "vec4",
    "Quat::from_array": ("vec4", _floats(4)),
    "Quat::from_slice": f"vec4({_elements(4)})",
    "Quat::from_axis_angle": "vec4({0} * sin({1} * 0.5), cos({1} * 0.5))",
    "Quat::from_scaled_axis": (
        "(length({0}) == 0.0 ? vec4(0.0, 0.0, 0.0, 1.0) "
        ": vec4({0} / length({0}) * sin(length({0}) * 0.5), cos(length({0}) * 0.5)))"
    ),
    "Quat::from_rotation_x": "vec4(sin({0} * 0.5), 0.0, 0.0, cos({0} * 0.5))",
    "Quat::from_rotation_y": "vec4(0.0, sin({0} * 0.5), 0.0, cos({0} * 0.5))",
    "Quat::from_rotation_z": "vec4(0.0, 0.0, sin({0} * 0.5), cos({0} * 0.5))",
    "Quat::xyz": "({0}.xyz)",
    # A unit quaternion's inverse is its conjugate, which is what glam's inverse returns.
    "Quat::conjugate": "vec4(-{0}.xyz, {0}.w)",
    "Quat::inverse": "vec4(-{0}.xyz, {0}.w)",
    "Quat::dot": "dot",
    "Quat::length": "length",
    "Quat::length_squared": "dot({0}, {0})",
    "Quat::length_recip": "(1.0 / length({0}))",
    "Quat::normalize": "normalize",
    "Quat::is_finite": "all(isfinite({0}))",
    "Quat::is_nan": "any(isnan({0}))",
    "Quat::is_normalized": "(abs(dot({0}, {0}) - 1.0) <= 0.0002)",
    "Quat::abs_diff_eq": "all(lessThanEqual(abs({0} - {1}), vec4({2})))",
    # lerp interpolates towards the end or its negation, whichever is nearer, and normalizes.
    "Quat::lerp": (
        "normalize({0} * (1.0 - {2}) + {1} * (dot({0}, {1}) >= 0.0 ? 1.0 : -1.0) * {2})"
    ),
    "Quat::mul_vec3": (
        "({1} * ({0}.w * {0}.w - dot({0}.xyz, {0}.xyz)) + {0}.xyz * (dot({1}, {0}.xyz) * 2.0) "
        "+ cross({0}.xyz, {1}) * ({0}.w * 2.0))"
    ),
    "Quat::mul_vec3a": (
        "({1} * ({0}.w * {0}.w - dot({0}.xyz, {0}.xyz)) + {0}.xyz * (dot({1}, {0}.xyz) * 2.0) "
        "+ cross({0}.xyz, {1}) * ({0}.w * 2.0))"
    ),
    # The Hamilton product.
    "Quat::mul_quat": (
        "vec4({0}.w * {1}.x + {0}.x * {1}.w + {0}.y * {1}.z - {0}.z * {1}.y, "
        "{0}.w * {1}.y - {0}.x * {1}.z + {0}.y * {1}.w + {0}.z * {1}.x, "
        "{0}.w * {1}.z + {0}.x * {1}.y - {0}.y * {1}.x + {0}.z * {1}.w, "
        "{0}.w * {1}.w - {0}.x * {1}.x - {0}.y * {1}.y - {0}.z * {1}.z)"
    ),
    "Quat::as_dquat": "dvec4",
    "DQuat::as_quat": "vec4",
}
# Methods with no CrossGL spelling, and why.
_OPTIONAL = "no CrossGL function returns an optional value"
_UNSPELLED = {
    "IVec3::checked_manhattan_distance": _OPTIONAL,
    "IVec3::checked_add": _OPTIONAL,
    "IVec3::checked_sub": _OPTIONAL,
    "IVec3::checked_mul": _OPTIONAL,
    "IVec3::checked_div": _OPTIONAL,
    "IVec3::checked_add_unsigned": _OPTIONAL,
    "IVec3::checked_sub_unsigned": _OPTIONAL,
    "UVec3::checked_add_signed": _OPTIONAL,
    "Mat4::try_inverse": _OPTIONAL,
    "BVec3::set": "no CrossGL function changes an element of a vector in place",
    "Mat4::set_row": "no CrossGL function changes a row of a matrix in place",
    "Mat4::col_mut": "CrossGL has no references to a matrix's columns",
    "Mat4::to_cols_array": "no CrossGL function converts a matrix to an array",
    "Mat4::to_rows_array": "no CrossGL function converts a matrix to an array",
    "Mat4::to_cols_array_2d": "no CrossGL function converts a matrix to an array",
    "Mat4::write_cols_to_slice": (
        "no CrossGL function writes a matrix's elements to an array"
    ),
    "Quat::to_array": "no CrossGL function converts a vector to an array",
    "Quat::write_to_slice": (
        "no CrossGL function writes a vector's elements to an array"
    ),
}
_NO_BUILT_IN = "no CrossGL built-in computes it"


def _crossgl(kind: Type) -> Type:
    return CROSSGL_TYPES.get(kind.opaque, kind)


def _counterpart(
    signature: Signature | None, params: tuple[Param, ...] | None = None
) -> Signature | None:
    """The CrossGL operands of a method whose own are glam types CrossGL spells otherwise, or references, or
    ``params`` when given."""
    if signature is None:
        return None
    if params is None:
        params = tuple(
            Param(_crossgl(p.type), "in" if p.direction == "address" else p.direction)
            for p in signature.params
        )
    result = _crossgl(signature.result)
    if params == signature.params and result == signature.result:
        return None
    return Signature(result, params)


def _spelled(feature: Feature) -> Feature:
    spelled = SPELLINGS.get(feature.name, "")
    operands = None
    if isinstance(spelled, tuple):
        spelled, operands = spelled
    counterpart = _counterpart(feature.signature, operands)
    note = feature.note
    if not spelled:
        note = note or _UNSPELLED.get(feature.name, _NO_BUILT_IN)
    return replace(feature, crossgl=spelled, counterpart=counterpart, note=note)


def parse(
    texts: dict[str, str], glsl_names: set[str] | frozenset[str] = frozenset()
) -> list[Feature]:
    """The methods of :data:`FAMILIES`' types other than ``Vec3``, given each type's source by its name. ``Vec3``'s
    source is read for the names its methods take, which the other vectors' methods do not repeat.
    """
    features = []
    for family in FAMILIES:
        seen: set[str] = set()
        for name, self_type in family:
            found = rust.methods(
                texts[name],
                rf"^impl {name} \{{",
                name,
                self_type,
                name.lower(),
                glsl_names,
                rust.GLAM_FEATURES,
            )
            for feature in found:
                method = feature.name.partition("::")[2]
                if method in seen:
                    continue
                seen.add(method)
                if name != "Vec3":
                    features.append(_spelled(feature))
    return features
