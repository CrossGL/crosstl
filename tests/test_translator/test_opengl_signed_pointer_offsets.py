"""OpenGL logical pointer offsets that may be negative before guarded access.

MLX's reverse scans pass ``in + axis_size - offset - N_READS`` to a guarded
loader. The pointer can precede the row (and the buffer) when the block is
partial; only accesses with ``start + N - i - 1 < total`` read through it.
"""

import re

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen, OpenGLIndexTypeError
from tests.test_translator.test_metal_builtin_ownership import _compile

SHADER = """
shader ReverseGuardedLoads {
    void load_safe(inout float[4] values, const device float* input, int start, int total) {
        for (int i = 0; i < 4; i++) {
            values[4 - i - 1] = start + 4 - i - 1 < total ? input[i] : 0.0;
        }
    }

    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(
            StructuredBuffer<float> source @buffer(0),
            RWStructuredBuffer<float> result @buffer(1),
            uvec3 lid @gl_LocalInvocationID
        ) {
            uint64 total = TOTAL;
            uint offset = lid.x * 4u;
            float[4] values;
            load_safe(values, source + total - offset - 4, int(offset), int(total));
            result[lid.x] = values[0] + values[3];
        }
    }
}
"""
ACCESS = {"expression": "input + i", "minimum": 0, "maximum": 2**31 - 1}


def _generate(total="uint64(6)", assertions=()):
    return (
        GLSLCodeGen()
        .set_index_range_assertions(list(assertions))
        .generate(crosstl.translator.parse(SHADER.replace("TOTAL", total)))
    )


def test_negative_logical_offset_is_computed_exactly_in_int(tmp_path):
    generated = _generate(assertions=[ACCESS])
    # 6 - 124 - 4 < 0: each delta fits int and so does every partial sum, so the
    # termwise int arithmetic is the exact logical offset (no uint64 wrap).
    call = re.search(r"load_safe\w*\(values, [^;]*\);", generated)
    assert call is not None, generated
    assert "- int(offset)) - 4)" in call.group(0), call.group(0)
    assert "uint64_t" not in call.group(0), call.group(0)
    assert "input_offset + i" in generated
    _compile(generated, "opengl", tmp_path)


def test_guarded_accesses_through_negative_offsets_still_need_a_proof():
    # The offset alone is exact; the composed subscript is relational in the
    # guard, so without an explicit complete-offset contract it stays rejected.
    with pytest.raises(OpenGLIndexTypeError) as raised:
        _generate()
    assert raised.value.reason == "composed-pointer-index-range-unproven"
    assert raised.value.indexed_value == "input"


@pytest.mark.parametrize(
    "total",
    (
        # An unbounded wide delta cannot be shown to fit int.
        "uint64(asuint(source[0])) << 32",
        # 2^31 alone does not fit int, so int arithmetic could overflow.
        "uint64(2147483648l)",
    ),
)
def test_logical_offsets_require_every_delta_to_fit_int(total):
    with pytest.raises(OpenGLIndexTypeError):
        _generate(total=total, assertions=[ACCESS])
