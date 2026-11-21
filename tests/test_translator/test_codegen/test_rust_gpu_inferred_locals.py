"""Rust output types an untyped CrossGL ``let`` from its initializer, or refuses it.

CrossGL's ``let`` leaves the type to the generator. rust-gpu needs every binding's type, so the generator infers it
from a struct constructor, an initializer list of one element type, a unit enum variant or a buffer's ``length()``,
and refuses an initializer list whose element type it cannot infer instead of declaring the binding as a float.
Each program compiles with rust-gpu when a build is configured (``CROSSTL_COVERAGE_RUST_GPU``).
"""

from __future__ import annotations

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_rust_codegen import assert_generated_rust_smoke_compiles
from .test_rust_gpu_kernels import rust

PROGRAM = """
shader main {{
    struct Pair {{
        float a;
        int b;
    }}
    enum Mode : int {{
        Off,
        Low = 2,
        High
    }};
    compute probe {{
        void main(uvec3 id @ gl_GlobalInvocationID, RWStructuredBuffer<float> outp @ set(0) @ binding(0))
            @numthreads(1, 1, 1) {{
            uint i = id.x;
{body}
        }}
    }}
}}
"""


def program(body: str) -> str:
    return PROGRAM.format(
        body="\n".join("            " + line for line in body.splitlines())
    )


def test_untyped_locals_take_their_initializer_types(tmp_path):
    code = rust(program("""let mut p = Pair(0.5, 2);
let v = vec4(1.0, 2.0, 3.0, 4.0);
let arr = {v, v.wzyx};
let mut flags = {false, true};
let n = {1, 2, 3};
let m = (i > 2u) ? Mode::Low : Mode::High;
p.a = arr[1].x + float(n[2]);
flags[0] = m == Mode::High;
outp[i] = p.a + (flags[0] ? 1.0 : 0.0) + float(int(m));"""))
    assert "let mut p: Pair = Pair::new(0.5, 2);" in code
    assert "let arr: [Vec4; 2] = " in code
    assert "let mut flags: [bool; 2] = [false, true];" in code
    assert "let n: [i32; 3] = [1, 2, 3];" in code
    assert "let m: Mode = " in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_untyped_initializer_list_of_mixed_types_is_refused():
    with pytest.raises(RustLoweringRefused, match="untyped initializer list"):
        rust(program("let t = {1, 2.0};\noutp[i] = t[1];"))


def test_buffer_length_is_the_slice_length(tmp_path):
    code = rust(program("outp[i] = float(outp.length());"))
    assert "((outp.len() as i32) as f32)" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)
