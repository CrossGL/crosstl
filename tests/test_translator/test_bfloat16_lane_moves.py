"""DirectX lane moves carry exact bfloat16 values as their bits.

DirectX holds an exact bfloat16 value as its 16 bits in a ``uint``. A lane
move must hand those bits to the shuffle unchanged; converting the value to
``uint`` first moved ``uint(float(x))``, so MLX's bfloat16 maximum, minimum
and logaddexp scans, which shuffle their partial results with
``simd_shuffle_and_fill_up(x, -inf, i)``, read garbage on Direct3D 12 WARP.
"""

import re

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_votes import _codegen

SOURCE = """shader Bfloat16LaneMoves {
    StructuredBuffer<bfloat16_t> values @register(t0);
    RWStructuredBuffer<bfloat16_t> moved @register(u1);
    compute {
        @numthreads(64, 1, 1)
        void main(uint invocation @gl_LocalInvocationIndex) {
            bfloat16_t value = values[invocation];
            bfloat16_t fill = values[64];
            bfloat16_t down = WaveShuffleDown(value, 1u);
            bfloat16_t filled = WaveShuffleAndFillUp(value, fill, 2u);
            bfloat16_t read = WaveReadLaneAt(value, 5u);
            moved[invocation * 3u] = down;
            moved[invocation * 3u + 1u] = filled;
            moved[invocation * 3u + 2u] = read;
        }
    }
}"""

# A numeric conversion of a bfloat16 value into a shuffle's payload.
CONVERTED = re.compile(r"\buint\(__crossgl_bfloat16_to_float\(")


def _assignment(generated, name):
    (line,) = (
        line.strip()
        for line in generated.splitlines()
        if line.strip().startswith(f"uint {name} = ")
    )
    return line


GENERATORS = {
    "software": lambda: _codegen("directx"),
    "hardware": lambda: HLSLCodeGen(),
    "hardware-self": lambda: HLSLCodeGen(relative_wave_shuffle_out_of_range="self"),
}


@pytest.mark.parametrize("mode", GENERATORS)
def test_directx_lane_moves_keep_bfloat16_bits(tmp_path, mode):
    generated = GENERATORS[mode]().generate_stage(parse(SOURCE), "compute")
    for name in ("down", "filled", "read"):
        line = _assignment(generated, name)
        # The value and fill reach the move as the variables' own bits.
        assert not CONVERTED.search(line), line
        assert "(value," in line, line
    assert "(value, fill," in _assignment(generated, "filled")
    _compile(generated, "directx", tmp_path)


def test_opengl_lane_moves_keep_rounded_bfloat16_values(tmp_path):
    """OpenGL holds bfloat16 as rounded floats, which float moves keep."""
    generated = _codegen("opengl").generate_stage(parse(SOURCE), "compute")
    assert "crossglSoftwareSubgroupShuffleAndFillUpFloat(value, fill," in generated
    _compile(generated, "opengl", tmp_path)


def test_directx_numeric_bfloat16_conversions_stay_numeric():
    """Only moves keep bits: an explicit conversion still converts the value."""
    source = SOURCE.replace(
        "bfloat16_t down = WaveShuffleDown(value, 1u);",
        "bfloat16_t down = bfloat16_t(float(WaveShuffleDown(uint(value), 1u)));",
    )
    generated = _codegen("directx").generate_stage(parse(source), "compute")
    assert CONVERTED.search(_assignment(generated, "down").split("=", 1)[1]) or (
        "__crossgl_bfloat16_to_float(" in _assignment(generated, "down")
    )
