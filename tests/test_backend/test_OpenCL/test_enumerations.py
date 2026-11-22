"""OpenCL C enumerations: named ones keep their CrossGL type and their constants read ``Enum::constant``."""

import re
import shutil
import subprocess

import crosstl
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

TYPEDEF_ENUM = """
typedef enum {
    Off,
    Low = 2,
    High
} Mode;
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void probe(__global int* iin, __global float* outp) {
    uint i = get_global_id(0);
    Mode m = (iin[i] & 1) != 0 ? Low : High;
    float r = 0.0f;
    switch (m) {
    case Off:
        r = 1.0f;
        break;
    case Low:
        r = 2.0f;
        break;
    default:
        r = 4.0f;
        break;
    }
    if (m == High) {
        r += 8.0f;
    }
    outp[i] = r + (float)((int)(m)) * 64.0f + (float)((int)(Off)) * 4096.0f;
}
"""

SCOPES = """
enum Mode { Off, Low = 2, High = Low + 1 };
typedef enum Mode ModeAlias;
enum { Lone = 7 };
typedef struct { int Low; float w; } Pair;

int pick(enum Mode m, int High) {
    return m == Off ? High : (int)m;
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void probe(__global int* iin, __global float* outp) {
    uint i = get_global_id(0);
    ModeAlias a = (iin[i] & 1) ? Low : High;
    Pair p;
    p.Low = 3;
    p.w = 1.0f;
    float r = (float)pick(a, 9);
    {
        int Low = 5;
        r += (float)Low;
    }
    r += (float)p.Low + (float)Lone + (float)((int)Low);
    outp[i] = r;
}
"""


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return crossgl


def to_hlsl(tmp_path, source):
    path = tmp_path / "probe.cl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def compile_hlsl(tmp_path, hlsl):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "probe.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    entry = re.findall(r"void (\w+)\(", hlsl)[-1]
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", entry, str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def test_typedef_enumeration_constants_read_with_their_type():
    crossgl = to_crossgl(TYPEDEF_ENUM)

    assert "enum Mode {" in crossgl
    assert "var m: Mode = (((iin[i] & 1) != 0) ? Mode::Low : Mode::High);" in crossgl
    assert "case Mode::Off:" in crossgl
    assert "case Mode::Low:" in crossgl
    assert "(m == Mode::High)" in crossgl
    assert "f32(i32(Mode::Off))" in crossgl
    assert re.search(r"(?<!::)\b(Off|Low|High)\b(?!\s*[=,])", crossgl) is None


def test_typedef_enumeration_round_trips_to_hlsl(tmp_path):
    hlsl = to_hlsl(tmp_path, TYPEDEF_ENUM)

    assert "int m = (((iin[i] & 1) != 0) ? Mode_Low : Mode_High);" in hlsl
    assert "case Mode_Off:" in hlsl
    assert re.search(r"\b(Off|Low|High)\b", hlsl) is None
    compile_hlsl(tmp_path, hlsl)


def test_declarations_of_the_same_name_shadow_an_enumeration_constant():
    crossgl = to_crossgl(SCOPES)

    # A parameter and a block-scope variable named like a constant hide it (C11 6.2.1p4).
    assert "i32 pick(Mode m, i32 High) {" in crossgl
    assert "return ((m == Mode::Off) ? High : i32(m));" in crossgl
    assert "var Low: i32 = 5;\n        r += f32(Low);" in crossgl
    # The constant is visible again after the block, and a member of the same name is not a constant.
    assert "p.Low = 3;" in crossgl
    assert "f32(p.Low)" in crossgl
    assert "f32(i32(Mode::Low))" in crossgl


def test_enumeration_initializers_aliases_and_untagged_constants():
    crossgl = to_crossgl(SCOPES)

    assert "High = (Mode::Low + 1)," in crossgl
    assert "typedef Mode ModeAlias;" in crossgl
    assert "var a: Mode = ((iin[i] & 1) ? Mode::Low : Mode::High);" in crossgl
    # An enumeration with neither a tag nor a typedef name has no type to qualify with.
    assert "const i32 Lone = 7;" in crossgl
    assert "f32(Lone)" in crossgl


def test_shadowed_enumeration_constants_compile_as_hlsl(tmp_path):
    compile_hlsl(tmp_path, to_hlsl(tmp_path, SCOPES))
