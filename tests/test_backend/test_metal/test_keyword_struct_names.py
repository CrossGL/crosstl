"""Metal structures named with a word that CrossGL reads as a keyword.

Metal reserves only C++'s keywords and its own, so a Metal program may name a structure ``let``, ``var``, ``vec2`` or
``task``. The CrossGL parser reads ``let`` and ``vec2`` as keywords and refused ``struct let`` with a syntax
error. The importer now gives such a structure the suffixed name at its declaration and at every use.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import KEYWORDS as METAL_KEYWORDS
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.lexer import KEYWORDS as CROSSGL_KEYWORDS
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

SOURCE = """#include <metal_stdlib>
using namespace metal;

struct NAME {
    float a;
    int b;
};

NAME make(float x) {
    NAME t;
    t.a = x * 2.0;
    t.b = 3;
    return t;
}

float weigh(NAME t) {
    return t.a + float(t.b) * 32.0;
}

kernel void probe(device const int* iin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    NAME t = make(float(iin[i] & 7));
    t.b += iin[i] & 3;
    outp[i] = weigh(t);
}
"""

# Words a Metal program may declare that CrossGL lexes as keywords.
NAMES = ("let", "var", "vec2", "loop", "task")


def _crossgl(name):
    source = SOURCE.replace("NAME", name)
    program = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(program)
    assert CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse() is not None
    return crossgl


def test_probe_names_are_crossgl_keywords_metal_allows():
    for name in NAMES:
        assert name in CROSSGL_KEYWORDS
        assert name not in METAL_KEYWORDS


@pytest.mark.parametrize("name", NAMES)
def test_structure_named_with_a_crossgl_keyword_takes_a_suffix(name):
    crossgl = _crossgl(name)

    assert f"struct {name}_ {{" in crossgl
    assert f"{name}_ make(float x)" in crossgl
    assert f"float weigh({name}_ t)" in crossgl
    assert f"{name}_ t = make(" in crossgl
    assert f"struct {name} " not in crossgl


def test_suffixed_name_does_not_collide_with_a_structure_of_that_name():
    source = SOURCE.replace("NAME", "let") + "\nstruct let_ { float c; };\n"
    program = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(program)

    assert CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse() is not None
    assert crossgl.count("struct let_ {") == 1
    assert "struct let__2 {" in crossgl
    assert "float c;" in crossgl.split("struct let_ {")[1].split("}")[0]


def test_structure_with_an_ordinary_name_keeps_it():
    crossgl = _crossgl("Record")

    assert "struct Record {" in crossgl
    assert "Record_" not in crossgl


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_keyword_named_structure_translates_to_each_target(tmp_path, backend):
    path = tmp_path / "probe.metal"
    path.write_text(SOURCE.replace("NAME", "let"), encoding="utf-8")

    output = crosstl.translate(str(path), backend=backend)

    assert "let_" in output
    if backend == "metal" and shutil.which("xcrun"):
        target = tmp_path / "out.metal"
        target.write_text(output, encoding="utf-8")
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(target),
                "-o",
                str(tmp_path / "out.air"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
