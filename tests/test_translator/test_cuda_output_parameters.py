"""CUDA and HIP output declare CrossGL ``out`` and ``inout`` parameters as C++ references (issue #3159).

Both emitters wrote ``void setx(inout float x)`` as ``void setx(float x)``, so the callee's writes never reached the
caller. They now write ``float& x``, and ``float (&v)[2]`` for an array. A reference binds the caller's object where
CrossGL copies in and out, so a call that passes one variable to an ``out``/``inout`` parameter and to another
parameter that reaches the caller is refused.
"""

import pytest

import crosstl
from crosstl.translator.codegen.c_reference_parameters import reference_declarator
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused

DIRECTIONS = """
shader main {
    struct Pair { float a; float b; }
    buffer float* results;
    void setx(inout float x) { x = x + 5.0; }
    void sety(out float y) { y = 5.0; }
    void setv(inout vec2 v) { v.y = 3.0; }
    void setp(out Pair p) { p.a = 1.0; p.b = 2.0; }
    void sets(inout float v[2]) { v[0] = 5.0; }
    float keep(float x) { x = 9.0; return x; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float a = 1.0;
        float b;
        vec2 w = vec2(0.0);
        Pair p;
        float c[2];
        c[0] = 1.0;
        c[1] = 2.0;
        setx(a);
        sety(b);
        setv(w);
        setp(p);
        sets(c);
        results[0] = a + b + w.y + p.b + c[0] + keep(a);
    }
    }
}
"""

REFUSED = {"cuda": CudaLoweringRefused, "hip": HipLoweringRefused}


def _translate(tmp_path, source, backend):
    path = tmp_path / "params.cgl"
    path.write_text(source)
    return " ".join(
        crosstl.translate(str(path), backend=backend, format_output=False).split()
    )


@pytest.mark.parametrize(
    ("declarator", "name", "expected"),
    [
        ("float x", "x", "float& x"),
        ("float v[2]", "v", "float (&v)[2]"),
        ("int g[2][3]", "g", "int (&g)[2][3]"),
        ("const Pair p", "p", "const Pair& p"),
        ("float& x", "x", "float& x"),
        ("float x", "y", "float x"),
    ],
)
def test_reference_declarator(declarator, name, expected):
    assert reference_declarator(declarator, name) == expected


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_out_and_inout_parameters_are_references(tmp_path, backend):
    code = _translate(tmp_path, DIRECTIONS, backend)
    assert "void setx(float& x)" in code
    assert "void sety(float& y)" in code
    assert "void setv(float2& v)" in code
    assert "void setp(Pair& p)" in code
    assert "void sets(float (&v)[2])" in code
    # A value parameter stays a copy.
    assert "float keep(float x)" in code
    assert "setx(a);" in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize(
    ("signature", "call"),
    [
        ("void two(inout float x, inout float y)", "two(a, a);"),
        ("void two(inout float x, out float y)", "two(v.x, v.y);"),
        ("void two(inout float x, float y[2])", "two(c[0], c);"),
    ],
)
def test_overlapping_output_arguments_are_refused(tmp_path, backend, signature, call):
    source = f"""
    shader main {{
        buffer float* results;
        {signature} {{ x = x + 1.0; }}
        compute {{
        @ numthreads(1, 1, 1)
        void main() {{
            float a = 1.0;
            vec2 v = vec2(1.0);
            float c[2];
            c[0] = 1.0;
            c[1] = 2.0;
            {call}
            results[0] = a + v.x + c[0];
        }}
        }}
    }}
    """
    with pytest.raises(REFUSED[backend]) as error:
        _translate(tmp_path, source, backend)
    assert "overlapping out/inout arguments" in str(error.value)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_an_output_and_a_value_copy_of_one_variable_translate(tmp_path, backend):
    source = """
    shader main {
        buffer float* results;
        void mix2(inout float x, float y) { x = x + y; }
        compute {
        @ numthreads(1, 1, 1)
        void main() { float a = 1.0; mix2(a, a); results[0] = a; }
        }
    }
    """
    code = _translate(tmp_path, source, backend)
    assert "void mix2(float& x, float y)" in code
    assert "mix2(a, a);" in code


GENERIC_CALLS = """
shader main {{
    buffer float* results;
    void bump<T>(inout T value, T step) {{ value = value + step; }}
    void both<T>(inout T x, inout T y) {{ x = x + y; }}
    void inc<T>(inout T value) {{ value = value + T(1.0); }}
    void inc<T>(inout T value, T step) {{ value = value + step; }}
    void twice<T>(inout T x) {{ both<T>(x, x); }}
    compute {{
    @ numthreads(1, 1, 1)
    void main() {{
        vec4 v = vec4(1.0);
        vec2 w = vec2(1.0);
        float a = 1.0;
        {call}
        results[0] = v.x + w.x + a;
    }}
    }}
}}
"""


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("bump<vec2>(v.xy, vec2(1.0));", "swizzle passed to an out/inout parameter"),
        ("both<float>(a, a);", "overlapping out/inout arguments"),
        # Deduced type arguments, an overload, and a call in a specialized body.
        ("bump(v.xy, vec2(1.0));", "bump_vec2 receives '.xy' as argument 1"),
        ("both(a, a);", "both_float receives a through more than one parameter"),
        ("inc<vec2>(v.zw, w);", "inc_vec2_overload1 receives '.zw' as argument 1"),
        ("twice<float>(a);", "both_float receives x through more than one parameter"),
    ],
)
def test_generic_callees_get_the_reference_argument_checks(
    tmp_path, backend, call, message
):
    with pytest.raises(REFUSED[backend]) as error:
        _translate(tmp_path, GENERIC_CALLS.format(call=call), backend)
    assert message in str(error.value)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_generic_callees_with_bindable_arguments_translate(tmp_path, backend):
    code = _translate(
        tmp_path,
        GENERIC_CALLS.format(call="bump<vec2>(w, vec2(1.0)); bump<float>(v.x, a);"),
        backend,
    )
    assert "bump_vec2(w, " in code
    assert "bump_float(v.x, a);" in code
