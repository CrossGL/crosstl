"""Rust ``&mut T`` parameters import as CrossGL ``inout`` parameters and round-trip with their writes."""

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser
from crosstl.translator.codegen.rust_codegen import RustCodeGen

SOURCE = """
#[derive(Clone, Copy, Default)]
pub struct Counter {
    pub total: i32,
    pub steps: i32,
}

impl Counter {
    pub fn add(&mut self, k: i32) {
        self.total = self.total + k;
        self.steps = self.steps + 1;
    }

    pub fn value(&self) -> i32 {
        self.total * 10 + self.steps
    }
}

pub fn bump(c: &mut Counter, x: &mut i32, v: &mut [i32; 2], r: &i32) -> i32 {
    c.add(*r);
    *x = *x * 2;
    v[1] = *x + *r;
    if *x > 100 {
        return 7;
    }
    3
}

pub fn calc(seed: i32) -> i32 {
    let mut c = Counter { total: seed, steps: 0 };
    let mut x: i32 = 4;
    let mut v: [i32; 2] = [0, 0];
    let r: i32 = 5;
    let k = bump(&mut c, &mut x, &mut v, &r);
    c.add(2);
    c.value() * 1000 + x * 100 + v[1] * 10 + k
}

#[compute_shader]
pub fn main() {
    let t = calc(1);
}
"""

# Counter (1, 0) -> add(5) -> (6, 1); x 8; v[1] 13; k 3; add(2) -> (8, 2), value 82.
EXPECTED = 82 * 1000 + 8 * 100 + 13 * 10 + 3


def _import(code):
    ast = RustParser(RustLexer(code).tokenize()).parse()
    return RustToCrossGLConverter().generate(ast)


def _compact(code):
    return re.sub(r"\s+", " ", code)


def test_mutable_references_import_as_inout_parameters():
    crossgl = _compact(_import(SOURCE))
    assert "int bump(inout Counter c, inout int x, inout int v[2], int r)" in crossgl
    assert "void Counter_add(inout Counter self, int k)" in crossgl
    assert "int Counter_value(Counter self)" in crossgl


def test_mutable_slices_are_not_inout_values():
    crossgl = _compact(
        _import(
            """
            pub fn fill(values: &mut [f32], x: &mut f32) {
                values[0] = *x;
            }
            """
        )
    )
    assert "inout float x" in crossgl
    assert "inout float values" not in crossgl


def _run_rust(tmp_path, name, program_text):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    program = tmp_path / f"{name}.rs"
    program.write_text(program_text, encoding="utf-8")
    binary = tmp_path / name
    compiled = subprocess.run(
        [rustc, "--edition", "2021", "-A", "warnings", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return int(ran.stdout)


def test_mutable_references_round_trip_through_crossgl(tmp_path):
    driver = '\nfn main() { println!("{}", calc(1)); }\n'
    original = SOURCE[: SOURCE.index("#[compute_shader]")] + driver
    assert _run_rust(tmp_path, "original", original) == EXPECTED

    emitted = RustCodeGen().generate(crosstl.translator.parse(_import(SOURCE)))
    start = emitted.index("#[")
    end = emitted.index("// Compute Shader")
    assert _run_rust(tmp_path, "round_trip", emitted[start:end] + driver) == EXPECTED
