"""CrossGL ``out``/``inout`` parameters keep their copy-back in Rust output (GLSL 4.60 section 6.1.1)."""

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

PROBE = """
shader RustOutputParameters {
    struct Pair {
        int a;
        int b;
    }

    void addTo(inout int total, int amount) {
        total = total + amount;
    }

    int clampValue(inout int value, out int steps) {
        steps = 0;
        if (value > 10) {
            value = 10;
            steps = 1;
            return 100 + value;
        }
        value = value + 1;
        steps = 2;
        return value;
    }

    void forward(inout int total) {
        addTo(total, 5);
        if (total > 100) {
            return;
        }
        addTo(total, 1);
    }

    void swapPair(inout Pair p) {
        int t = p.a;
        p.a = p.b;
        p.b = t;
    }

    int viaReturn(inout int v) {
        int s;
        return clampValue(v, s) * 2 + s;
    }

    int calc(int seed) {
        int total = seed;
        addTo(total, 3);
        forward(total);
        Pair p;
        p.a = 1;
        p.b = 2;
        swapPair(p);
        addTo(p.a, 10);
        int values[3];
        values[0] = 1;
        values[1] = 2;
        values[2] = 3;
        addTo(values[1], 40);
        int steps;
        int big = 50;
        int r1 = clampValue(big, steps);
        int small = 4;
        int steps2;
        int r2 = clampValue(small, steps2);
        int q = 20;
        int r3 = viaReturn(q);
        return total + p.a * 3 + p.b * 7 + values[1] * 11 + r1 * 13 + big * 17
            + steps * 19 + r2 * 23 + small * 29 + steps2 * 31 + r3 * 37 + q * 41;
    }

    compute {
        layout(local_size_x = 1) in;
        void main() {
            int v = calc(1);
        }
    }
}
"""

# GLSL semantics: total 10, p = (12, 1), values[1] 42, r1 110 with big 10 and steps 1,
# r2 5 with small 5 and steps2 2, r3 221 with q 10.
EXPECTED = (
    10
    + 12 * 3
    + 1 * 7
    + 42 * 11
    + 110 * 13
    + 10 * 17
    + 1 * 19
    + 5 * 23
    + 5 * 29
    + 2 * 31
    + 221 * 37
    + 10 * 41
)


def _generate(source):
    return RustCodeGen().generate(crosstl.translator.parse(source))


def _compact(code):
    return re.sub(r"\s+", " ", code)


def test_output_parameters_are_mutable_references_with_copy_back():
    code = _compact(_generate(PROBE))
    assert "pub fn addTo(total_ref: &mut i32, amount: i32)" in code
    assert "let mut total: i32 = *total_ref;" in code
    assert "total = (total + amount); *total_ref = total; }" in code
    assert "pub fn clampValue(value_ref: &mut i32, steps_ref: &mut i32)" in code
    assert (
        "let return_value = (100 + value); *value_ref = value; *steps_ref = steps; "
        "return return_value;" in code
    )
    assert "*total_ref = total; return;" in code
    assert "addTo(&mut total, 3);" in code
    assert "addTo(&mut p.a, 10);" in code
    assert "swapPair(&mut p);" in code
    assert "clampValue(&mut big, &mut steps)" in code
    assert "let mut q: i32 = 20;" in code


def test_output_parameters_match_crossgl_semantics(tmp_path):
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is not installed")
    code = _generate(PROBE)
    start = code.index("#[derive")
    end = code.index("// Compute Shader")
    program = tmp_path / "probe.rs"
    program.write_text(
        code[start:end] + '\nfn main() { println!("{}", calc(1)); }\n',
        encoding="utf-8",
    )
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [rustc, "--edition", "2021", "-A", "warnings", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert int(ran.stdout) == EXPECTED


def test_reference_names_avoid_existing_identifiers():
    source = """
    shader Names {
        void bump(inout int x) {
            int x_ref = 2;
            x = x + x_ref;
        }
        compute {
            layout(local_size_x = 1) in;
            void main() {
                int x = 1;
                bump(x);
            }
        }
    }
    """
    code = _compact(_generate(source))
    assert "pub fn bump(x_ref_1: &mut i32)" in code
    assert "let mut x: i32 = *x_ref_1;" in code
    assert "*x_ref_1 = x;" in code


def _call_source(declarations, call):
    return f"""
    shader Refused {{
        void setPair(out vec2 r) {{
            r = vec2(1.0, 2.0);
        }}
        void addTo(inout int total, int amount) {{
            total = total + amount;
        }}
        void both(inout int a, inout int b) {{
            a = b;
        }}
        compute {{
            layout(local_size_x = 1) in;
            void main() {{
                {declarations}
                {call}
            }}
        }}
    }}
    """


@pytest.mark.parametrize(
    "declarations, call, reason",
    [
        ("vec4 v;", "setPair(v.xy);", r"swizzle \.xy"),
        ("int t = 1;", "addTo(t, t);", "reads it in argument 2"),
        ("int t = 1;", "both(t, t);", "reads it in argument 2"),
        ("int t = 1;", "addTo(t + 1, 2);", "not a local variable"),
    ],
)
def test_unborrowable_output_arguments_are_refused(declarations, call, reason):
    with pytest.raises(RustLoweringRefused, match=reason):
        _generate(_call_source(declarations, call))


def test_distinct_elements_of_one_array_are_refused():
    source = _call_source("int a[2];", "both(a[0], a[1]);")
    with pytest.raises(RustLoweringRefused, match="reads it in argument 2"):
        _generate(source)


def test_shader_scope_output_arguments_are_refused():
    source = """
    shader Globals {
        const int LIMIT = 3;
        void addTo(inout int total, int amount) {
            total = total + amount;
        }
        compute {
            layout(local_size_x = 1) in;
            void main() {
                addTo(LIMIT, 1);
            }
        }
    }
    """
    with pytest.raises(RustLoweringRefused, match="not a local variable"):
        _generate(source)
