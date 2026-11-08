"""64-bit integer literals that initialize 32-bit integers in GLSL.

``2654435761`` does not fit ``int``, so C and Metal sources give it a 64-bit type and CrossGL spells it
``2654435761l``. Initializing a ``uint`` converts the value, which fits. GLSL has no implicit ``int64_t`` to
``uint`` conversion, so the literal is spelled with the 32-bit type it initializes.
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
