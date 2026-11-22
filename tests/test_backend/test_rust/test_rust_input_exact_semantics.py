"""Rust input keeps the results of Rust operations whose CrossGL namesakes round, sign or wrap differently.

Each case imports a rust-gpu kernel to CrossGL, checks the CrossGL spelling that computes Rust's result, and writes
the CrossGL back to Rust, which is compiled with rust-gpu when a build is configured (``CROSSTL_COVERAGE_RUST_GPU``).
"""

from __future__ import annotations

import pytest

from crosstl.backend.import_refusals import RustImportRefused
from tests.test_backend.test_rust.test_rust_input_round_trip import (
    PRELUDE,
    round_trip,
    translate,
)


def float_kernel(body: str, items: str = "") -> str:
    return PRELUDE + "use spirv_std::num_traits::Float;\n" + items + """
#[spirv(compute(threads(1)))]
pub fn probe(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] fin: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] iin: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] outp: &mut [f32],
) {
    let i = id.x as usize;
""" + body + "\n}\n"


def test_fract_truncates_and_fract_gl_floors(tmp_path):
    source = float_kernel("""    let a: f32 = fin[i];
    let v: Vec3 = Vec3::new(fin[i], fin[i + 1], fin[i + 2]);
    outp[i] = a.fract() + v.fract().x + v.fract_gl().y;""")
    crossgl, rust = round_trip(source, tmp_path)
    assert "(a - trunc(a))" in crossgl
    assert "(v - trunc(v)).x" in crossgl
    assert "fract(v).y" in crossgl
    assert "fract_gl" not in crossgl
    assert ".trunc()" in rust


def test_float_rem_euclid_adds_the_divisor_magnitude_to_a_negative_remainder(tmp_path):
    source = float_kernel("""    let a: f32 = fin[i];
    let b: f32 = fin[i + 1];
    let v: Vec2 = Vec2::new(fin[i], fin[i + 1]);
    outp[i] = a.rem_euclid(b) + v.rem_euclid(Vec2::splat(-1.5)).y;""")
    crossgl, _ = round_trip(source, tmp_path)
    assert "((fmod(a, b) < 0.0) ? (fmod(a, b) + abs(b)) : fmod(a, b))" in crossgl
    assert (
        "vec2(((fmod(v.x, vec2((-1.5)).x) < 0.0) ? (fmod(v.x, vec2((-1.5)).x) + abs(vec2((-1.5)).x))"
        in crossgl
    )
    assert "mod(" not in crossgl.replace("fmod(", "")


def test_integer_euclidean_division_and_remainder(tmp_path):
    source = float_kernel("""    let n: i32 = iin[i] - 8;
    let u: u32 = iin[i] as u32;
    let q = n.div_euclid(3);
    let r = n.rem_euclid(-3);
    outp[i] = q as f32 + r as f32 + u.rem_euclid(5) as f32 + u.div_euclid(2) as f32;""")
    crossgl, _ = round_trip(source, tmp_path)
    assert (
        "(((n % 3) < 0) ? ((3 > 0) ? ((n / 3) - 1) : ((n / 3) + 1)) : (n / 3))"
        in crossgl
    )
    assert "(((n % (-3)) < 0) ? ((n % (-3)) + abs((-3))) : (n % (-3)))" in crossgl
    assert "(u % 5)" in crossgl
    assert "(u / 2)" in crossgl


def test_float_div_euclid_rounds_towards_negative_infinity_for_positive_divisors(
    tmp_path,
):
    source = float_kernel("""    let a: f32 = fin[i];
    outp[i] = a.div_euclid(fin[i + 1]);""")
    crossgl, _ = round_trip(source, tmp_path)
    assert (
        "((fmod(a, fin[(i + 1)]) < 0.0) ? ((fin[(i + 1)] > 0.0) ? (trunc(a / fin[(i + 1)]) - 1.0)"
        in crossgl
    )


def test_euclidean_operand_with_side_effects_is_refused():
    source = float_kernel(
        """    let mut a: f32 = fin[i];
    outp[i] = a.rem_euclid(next(&mut a));""",
        items="fn next(x: &mut f32) -> f32 { *x += 1.0; *x }\n",
    )
    with pytest.raises(RustImportRefused, match="rem_euclid"):
        translate(source, "cgl")


def test_narrow_integer_arithmetic_wraps_in_its_type(tmp_path):
    source = float_kernel("""    let v: i8 = iin[i] as i8;
    let w: u8 = iin[i] as u8;
    let s: i16 = iin[i] as i16;
    let a = v * v + 3;
    let b = !w;
    let c = -v;
    let e = s << 3;
    outp[i] = (v * v) as f32 + a as f32 + b as f32 + c as f32 + e as f32;""")
    crossgl, rust = round_trip(source, tmp_path)
    assert "let a = int8_t((int8_t((v * v)) + 3));" in crossgl
    assert "let b = uint8_t((~w));" in crossgl
    assert "let c = int8_t((-v));" in crossgl
    assert "let e = int16_t((s << 3));" in crossgl
    assert "float(int8_t((v * v)))" in crossgl
    assert " as i8)" in rust


def test_mutable_reference_local_names_the_place_it_borrows(tmp_path):
    source = float_kernel("""    let p = &mut outp[i];
    *p = fin[i] * 2.0;
    *p += *p * 4.0 + 1.0;
    let mut a = [0.0f32; 4];
    let mut j = 1;
    let q = &mut a[j];
    j += 1;
    *q = 3.0;
    let mut v = Vec3::ZERO;
    let r = &mut v;
    r.x = 2.0;
    outp[i] += a[1] + v.x + j as f32;""")
    crossgl, rust = round_trip(source, tmp_path)
    assert "let crosstl_ref_index1 = i;" in crossgl
    assert "outp[crosstl_ref_index1] = (fin[i] * 2.0);" in crossgl
    assert (
        "outp[crosstl_ref_index1] += ((outp[crosstl_ref_index1] * 4.0) + 1.0);"
        in crossgl
    )
    # The index is read when the reference is taken, before j changes.
    assert "let crosstl_ref_index2 = j;" in crossgl
    assert "a[crosstl_ref_index2] = 3.0;" in crossgl
    assert "v.x = 2.0;" in crossgl
    assert "let p" not in crossgl and "let q" not in crossgl
    assert "outp[crosstl_ref_index1 as usize]" in rust


def test_mutable_reference_to_a_call_result_is_refused():
    source = float_kernel(
        """    let p = &mut pick(1.0);
    *p = 2.0;
    outp[i] = *p;""",
        items="fn pick(x: f32) -> f32 { x }\n",
    )
    with pytest.raises(RustImportRefused, match="CrossGL has no reference locals"):
        translate(source, "cgl")


def test_glam_smoothstep_interpolates_its_receiver(tmp_path):
    source = float_kernel(
        """    let x: Vec3 = Vec3::new(fin[i], fin[i + 1], fin[i + 2]);
    let r = x.smoothstep(Vec3::ZERO, Vec3::ONE);
    outp[i] = r.y;"""
    )
    crossgl, _ = round_trip(source, tmp_path)
    assert "smoothstep(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), x)" in crossgl
