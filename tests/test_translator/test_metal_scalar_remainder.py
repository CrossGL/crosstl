"""Metal lowering of scalar ``%`` whose operand types are only partly known."""

from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.test_translator.test_metal_builtin_ownership import _compile

SOURCE = """
shader RemainderShader {
    const int N_READS = 4;
    const float PERIOD = 2.5;
    compute {
        layout(std430, binding = 0) buffer Ints { int data[]; } ints;
        layout(std430, binding = 1) buffer Floats { float data[]; } floats;
        layout(local_size_x = 1) in;
        void main() {
            int row = int(gl_GlobalInvocationID.x);
            int extra = row % N_READS;
            float phase = float(row) * 0.5;
            float wrapped = phase % PERIOD;
            ints.data[0] = extra;
            floats.data[0] = wrapped;
        }
    }
}
"""


def _metal(source):
    ast = Parser(Lexer(source).get_tokens()).parse()
    return MetalCodeGen().generate(ast)


def test_integer_remainder_with_shader_constant_stays_integer(tmp_path):
    metal = _metal(SOURCE)

    assert "int extra = row % N_READS;" in metal
    assert "fmod(row" not in metal
    # A known floating operand still selects the floating remainder.
    assert "float wrapped = fmod(phase, PERIOD);" in metal
    _compile(metal, "metal", tmp_path)
