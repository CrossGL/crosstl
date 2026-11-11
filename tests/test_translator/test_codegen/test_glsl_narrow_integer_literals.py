"""Integer literals spelled with the GLSL integer type they initialize.

``2654435761`` does not fit ``int``, so C and Metal sources give it a 64-bit type and CrossGL spells it
``2654435761l``. Initializing a ``uint`` converts the value, which fits. GLSL has no implicit ``int64_t`` to
``uint`` conversion, so the literal is spelled with the 32-bit type it initializes.

In the other direction, Mesa refuses ``int64_t[1](0)`` although ``GL_ARB_gpu_shader_int64`` lists the implicit
conversion, so a literal element of a 64-bit aggregate constructor carries that type's suffix. Calls and struct
constructors convert their arguments explicitly instead.
"""

import crosstl

SOURCE = """
shader S {
    compute {
        void main(RWStructuredBuffer<uint> out_ @buffer(0), uint i @gl_GlobalInvocationID) {
            const uint[3] primes = {1, 2654435761l, 805459861};
            uint single = 3674653429l;
            int small = -5l;
            int64 big = 5000000000l;
            uint64 ubig = 5000000000l;
            uint overflow_free = uint(big >> 32);
            buffer_store(out_, i, primes[i % 3] + single + uint(small) + overflow_free + uint(ubig >> 32));
        }
    }
}
"""


def _glsl(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def test_wide_literals_initializing_32_bit_integers_use_the_32_bit_type(tmp_path):
    glsl = _glsl(tmp_path)
    assert "uint[3](1, 2654435761u, 805459861)" in glsl
    assert "uint single = 3674653429u;" in glsl


def test_64_bit_objects_keep_64_bit_literals(tmp_path):
    glsl = _glsl(tmp_path)
    assert "int64_t big = 5000000000l;" in glsl
    assert "uint64_t ubig = 5000000000l;" in glsl


WIDE_SOURCE = """
shader S {
    struct Pair {
        int64 a;
        uint64 b;
    }

    int64 bump(int64 x) {
        return x + 1;
    }

    compute {
        void main(RWStructuredBuffer<int64> out_ @buffer(0), uint i @gl_GlobalInvocationID) {
            int64 totals[2] = {0, -7};
            uint64 masks[1] = {2};
            uint64 wide[1] = {5000000000l};
            Pair pair = Pair(3, 4u);
            int64 sum = 0;
            sum += bump(3) + totals[i % 2] + pair.a + int64(pair.b + masks[0] + wide[0]);
            buffer_store(out_, i, sum);
        }
    }
}
"""


def _wide_glsl(tmp_path):
    path = tmp_path / "wide.cgl"
    path.write_text(WIDE_SOURCE)
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def test_literals_in_64_bit_array_constructors_carry_the_element_type(tmp_path):
    glsl = _wide_glsl(tmp_path)
    # A negated literal is an expression; it keeps its explicit conversion.
    assert "int64_t[2](0l, int64_t((-7)))" in glsl
    assert "uint64_t[1](2ul)" in glsl
    assert "uint64_t[1](5000000000ul)" in glsl


def test_64_bit_scalars_calls_and_structs_keep_explicit_conversions(tmp_path):
    glsl = _wide_glsl(tmp_path)
    # Initializers accept the implicit conversion; calls and struct constructors convert explicitly.
    assert "int64_t sum = 0;" in glsl
    assert "bump(int64_t(3))" in glsl
    assert "Pair(int64_t(3), uint64_t(4u))" in glsl


SCALAR_VIEW_SOURCE = """
#include <metal_stdlib>
using namespace metal;
void bump(thread long t[1]) { t[0] += 1; }
void bump_unsigned(thread ulong t[1]) { t[0] += 1; }
kernel void k(const device int* in [[buffer(0)]], device long* out [[buffer(1)]],
              device ulong* out_unsigned [[buffer(2)]]) {
  int value = in[0];
  long zero = 0;
  long widened = value;
  ulong seven = 7;
  bump(&zero);
  bump(&widened);
  bump_unsigned(&seven);
  out[0] = zero + widened;
  out_unsigned[0] = seven;
}
"""


def test_literals_initializing_64_bit_scalar_array_views_carry_the_type(tmp_path):
    # A scalar passed by address to an array parameter becomes a one-element
    # array built by an array constructor, where Mesa needs the 64-bit literal.
    path = tmp_path / "views.metal"
    path.write_text(SCALAR_VIEW_SOURCE)
    glsl = crosstl.translate(str(path), backend="opengl", format_output=False)
    assert "int64_t zero[1] = int64_t[1](0l);" in glsl
    assert "uint64_t seven[1] = uint64_t[1](7ul);" in glsl
    assert "int64_t widened[1] = int64_t[1](int64_t(value));" in glsl
