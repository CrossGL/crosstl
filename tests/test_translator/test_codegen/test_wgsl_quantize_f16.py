"""WGSL output of a round trip through binary16, ``f16tof32(f32tof16(v))``.

The round trip is WGSL's ``quantizeToF16``, which naga lowers to ``OpQuantizeToF16``. The generator used to write
``unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x`` per component, which means the same, but on lavapipe the
SPIR-V naga writes for that pair returned ``v`` unrounded.
"""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer vec3* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec3 a0 = vec3(fin[i + 0u], fin[i + 1u], fin[i + 2u]);
            vec3 r = f16tof32(f32tof16(a0));
            float s = f16tof32(f32tof16(fin[i]));
            vec2 p = f16tof32(f32tof16(a0.xy));
            float k = f16tof32(f32tof16(iin[i]));
            outp[i] = r + vec3(s + p.x + k);
        }
    }
}
"""


def test_round_trip_through_binary16_is_quantize_to_f16(tmp_path):
    code = WGSLCodeGen().generate(Parser(Lexer(SOURCE).get_tokens()).parse())
    assert "var r: vec3<f32> = quantizeToF16(a0);" in code
    assert "var s: f32 = quantizeToF16(fin[i]);" in code
    assert "var p: vec2<f32> = quantizeToF16(a0.xy);" in code
    # f32tof16 takes a float, so an integer operand is converted first.
    assert "var k: f32 = quantizeToF16(f32(iin[i]));" in code
    assert "pack2x16float" not in code
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "probe.wgsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run([naga, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
