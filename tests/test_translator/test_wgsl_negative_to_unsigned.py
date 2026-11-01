"""WGSL output converts a negative integer literal to ``u32`` as an ``i32``, keeping its bits (#3686).

WGSL reads ``-1`` as an abstract integer, and ``u32(-1)`` converts its value, which ``u32`` cannot represent,
so naga rejects it. GLSL output's ``uint(-1)`` is 4294967295, as ``u32(-1i)`` is.
"""

import textwrap

import crosstl

PROGRAM = textwrap.dedent("""
    shader Probe {
        struct U { uint u; float w; };
        buffer float values[];
        compute {
            layout(local_size_x = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                float x = values[i];
                %s
            }
        }
    }
""")


def translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % body)
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


def test_negative_literal_assigned_to_uint_is_an_i32(tmp_path):
    output = translate(
        tmp_path,
        "uint u = -1; U s = U(-1, x); values[i] = float(u) + float(s.u) + s.w;",
    )
    assert "var u: u32 = u32(-1i);" in output
    assert "var s: U = U(u32(-1i), x);" in output
    assert "u32(-1)" not in output


def test_explicit_uint_conversion_of_a_negative_literal_is_an_i32(tmp_path):
    output = translate(tmp_path, "uint u = uint(-7); values[i] = float(u);")
    assert "var u: u32 = u32(-7i);" in output


def test_other_conversions_to_uint_are_unchanged(tmp_path):
    output = translate(
        tmp_path,
        "int k = -1; uint w = k; uint v = 3; values[i] = float(w) + float(v);",
    )
    assert "var w: u32 = u32(k);" in output
    assert "var v: u32 = u32(3);" in output
