"""Rust input writes CrossGL that keeps the meaning of Rust's value and operator semantics.

Each case is a rust-gpu kernel the importer translates to CrossGL; the CrossGL is checked for the spelling that
carries the Rust meaning, and the round trip back to Rust is compiled with rust-gpu when a build is configured
(``CROSSTL_COVERAGE_RUST_GPU``, see ``tools/coverage/rustgpu.py``), as the coverage harness measures Rust input.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import crosstl
from crosstl.backend.import_refusals import RustImportRefused
from tests.test_translator.test_codegen.test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
)

PRELUDE = """#![no_std]
use spirv_std::glam::*;
use spirv_std::spirv;
"""


def kernel(items: str, body: str) -> str:
    return PRELUDE + items + """
#[spirv(compute(threads(1)))]
pub fn probe(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] iin: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] outp: &mut [f32],
) {
    let i = id.x as usize;
""" + body + "\n}\n"


def translate(source: str, backend: str, suffix: str = ".rs") -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, f"probe{suffix}")
        path.write_text(source, encoding="utf-8")
        return crosstl.translate(str(path), backend=backend, format_output=False)


def round_trip(source: str, tmp_path) -> tuple[str, str]:
    """The CrossGL of ``source`` and the Rust written back from it, compiled when rust-gpu is configured."""
    crossgl = translate(source, "cgl")
    rust = translate(crossgl, "rust", ".cgl")
    try:
        assert_generated_rust_smoke_compiles(rust, tmp_path)
    except pytest.skip.Exception:
        pass
    return crossgl, rust


PAIR = """
#[derive(Clone, Copy, Default)]
struct Pair {
    a: f32,
    b: i32,
}

#[derive(Clone, Copy, Default)]
struct Frame {
    p: Pair,
    v: [u32; 2],
    w: Vec3,
    ok: bool,
}
"""


def test_struct_default_is_an_explicit_zero_constructor(tmp_path):
    source = kernel(
        PAIR,
        """    let mut p = Pair::default();
    let mut ps = [Pair::default(); 3];
    let f = Frame::default();
    let v = Vec4::default();
    let k = f32::default();
    p.a = iin[i] as f32;
    ps[1].b = 2;
    outp[i] = p.a + ps[1].b as f32 + f.p.a + f.v[1] as f32 + f.w.z + v.x + k;""",
    )
    crossgl, rust = round_trip(source, tmp_path)
    assert "Pair p = Pair(0.0, 0);" in crossgl
    assert "Pair ps[3] = {Pair(0.0, 0), Pair(0.0, 0), Pair(0.0, 0)};" in crossgl
    assert (
        "Frame f = Frame(Pair(0.0, 0), {0u, 0u}, vec3(0.0, 0.0, 0.0), false);"
        in crossgl
    )
    assert "let v = vec4(0.0, 0.0, 0.0, 0.0);" in crossgl
    assert "let k = 0.0;" in crossgl
    assert "::default()" not in crossgl
    assert "let mut p: Pair = Pair::new(0.0, 0);" in rust


def test_hand_written_default_keeps_its_function():
    source = kernel(
        """
#[derive(Clone, Copy)]
struct Unit {
    x: f32,
}

impl Default for Unit {
    fn default() -> Self {
        Unit { x: 1.0 }
    }
}
""",
        "    let u = Unit::default();\n    outp[i] = u.x;",
    )
    crossgl = translate(source, "cgl")
    assert "Unit(0.0)" not in crossgl
    assert "Unit_default()" in crossgl


def test_integer_not_is_bitwise_complement(tmp_path):
    source = kernel(
        "",
        """    let k = iin[i];
    let b = k > 3;
    let a = IVec2::new(k, -3);
    let e = !(!k) + -(!k);
    let c = !a;
    let g = !(!b);
    outp[i] = e as f32 + c.x as f32 + if g { 1.0 } else { 0.0 };""",
    )
    crossgl, _ = round_trip(source, tmp_path)
    assert "let e = ((~(~k)) + (-(~k)));" in crossgl
    assert "let c = (~a);" in crossgl
    assert "let g = (!(!b));" in crossgl


def test_vector_equality_compares_the_whole_vector(tmp_path):
    source = kernel(
        "",
        """    let a = Vec2::new(iin[i] as f32, 1.0);
    let b = Vec2::new(2.0, 1.0);
    let mut r = 0.0;
    if a == b {
        r += 1.0;
    }
    if a != b {
        r += 2.0;
    }
    if a.x == b.x {
        r += 4.0;
    }
    outp[i] = r;""",
    )
    crossgl, _ = round_trip(source, tmp_path)
    assert "if (all(a == b))" in crossgl
    assert "if (any(a != b))" in crossgl
    assert "if ((a.x == b.x))" in crossgl


def test_struct_and_array_locals_are_typed(tmp_path):
    source = kernel(
        """
#[derive(Clone, Copy, PartialEq)]
struct Pair {
    a: f32,
    b: f32,
}
""",
        """    let p = Pair { a: iin[i] as f32, b: 2.0 };
    let q = Pair { a: 5.0, b: p.b };
    let t = [(iin[i] & 1) as f32, 3.0];
    let mut flags = [false, false];
    flags[1] = p == q;
    outp[i] = t[0] + if flags[1] { 1.0 } else { 0.0 };""",
    )
    crossgl, rust = round_trip(source, tmp_path)
    assert "Pair p = Pair { a: float(iin[i]), b: 2.0 };" in crossgl
    assert "float t[2] = {float((iin[i] & 1)), 3.0};" in crossgl
    assert "bool flags[2] = {false, false};" in crossgl
    assert "((p.a == q.a) && (p.b == q.b))" in rust


def test_nested_array_type_keeps_the_outer_dimension_first(tmp_path):
    source = kernel(
        "",
        """    let mut a = [[0.0f32; 3]; 2];
    let b: [[u32; 4]; 2] = [[1u32; 4]; 2];
    a[1][2] = iin[i] as f32;
    outp[i] = a[1][2] + b[1][3] as f32;""",
    )
    crossgl, rust = round_trip(source, tmp_path)
    assert "float a[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};" in crossgl
    assert "uint b[2][4] = " in crossgl
    assert "let mut a: [[f32; 3]; 2]" in rust


def test_field_less_enum_is_declared(tmp_path):
    source = kernel(
        """
#[derive(Clone, Copy, PartialEq)]
#[repr(i32)]
enum Mode {
    Off,
    Low = 2,
    High,
}
""",
        """    let m = if (iin[i] & 1) != 0 { Mode::Low } else { Mode::High };
    outp[i] = m as i32 as f32 + if m == Mode::High { 8.0 } else { 0.0 };""",
    )
    crossgl, rust = round_trip(source, tmp_path)
    assert (
        "enum Mode : int {\n        Off,\n        Low = 2,\n        High\n    };"
        in crossgl
    )
    assert "let m: Mode = " in rust


def test_alias_constructor_is_the_aliased_type(tmp_path):
    source = kernel(
        "type Duo = Vec2;\n",
        "    let v = Duo::new(iin[i] as f32, 2.0);\n    outp[i] = v.x * v.y;",
    )
    crossgl, _ = round_trip(source, tmp_path)
    assert "let v = vec2(float(iin[i]), 2.0);" in crossgl
    assert "Duo::new" not in crossgl


def test_slice_len_is_the_buffer_length(tmp_path):
    source = kernel("", "    outp[i] = outp.len() as f32;")
    crossgl, rust = round_trip(source, tmp_path)
    assert "outp[i] = float(outp.length());" in crossgl
    assert "(outp.len() as i32)" in rust


def test_integer_literals_keep_their_unsigned_type(tmp_path):
    source = kernel(
        "",
        """    let top: u32 = 4294967295;
    let mask: u32 = 0x80000000;
    let one = 1f32;
    outp[i] = (top >> 28) as f32 + (mask >> 31) as f32 + (0xFFFFFFF0u32 as i32 + iin[i]) as f32 + one;""",
    )
    crossgl, _ = round_trip(source, tmp_path)
    assert "uint top = 4294967295u;" in crossgl
    assert "uint mask = 2147483648u;" in crossgl
    assert "int(4294967280u)" in crossgl
    assert "let one = 1.0;" in crossgl


def test_vector_float_predicates_reduce_like_glam(tmp_path):
    source = kernel(
        "",
        """    let v = Vec3::new(iin[i] as f32, 1.0, 2.0);
    let finite: bool = v.is_finite();
    let nan: bool = v.is_nan();
    let mask: BVec3 = v.is_nan_mask();
    outp[i] = if finite && !nan && !mask.x { 1.0 } else { 0.0 };""",
    )
    crossgl, _ = round_trip(source, tmp_path)
    assert "bool finite = all(isfinite(v));" in crossgl
    assert "bool nan = any(isnan(v));" in crossgl
    assert "bvec3 mask = isnan(v);" in crossgl


FRAGMENT = PRELUDE + """
#[spirv(fragment({modes}))]
pub fn main(uv: Vec2, color: &mut Vec4) {{
    *color = Vec4::new(uv.x, uv.y, 0.0, 1.0);
}}
"""


def test_early_fragment_tests_is_a_stage_layout():
    crossgl = translate(FRAGMENT.format(modes="early_fragment_tests"), "cgl")
    assert "layout(early_fragment_tests) in;" in crossgl
    named = translate(
        FRAGMENT.format(modes='entry_point_name = "fs", early_fragment_tests'), "cgl"
    )
    assert "fragment fs {\n        layout(early_fragment_tests) in;" in named


def test_unknown_fragment_mode_is_refused():
    with pytest.raises(RustImportRefused, match="origin_upper_left"):
        translate(FRAGMENT.format(modes="origin_upper_left"), "cgl")
