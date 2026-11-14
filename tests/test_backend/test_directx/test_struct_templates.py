"""HLSL 2021 struct templates import as CrossGL generic structures.

The importer dropped a struct's ``template <typename T>`` list and every use's arguments: ``Box<float> a;`` and
``Box<int> b;`` both became ``Box`` with an undeclared member type ``T``, an explicit specialization became a second
``struct Box`` of the same name, and member functions were lowered for the bare ``Box``. Each case below fails on that
importer. The HLSL programs compile with DXC 1.10 (``-HV 2021``).
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.import_refusals import DirectXImportRefused
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

PROGRAM = textwrap.dedent("""
    template <typename T>
    struct Box {
        T value;
        T twice() { return value + value; }
    };
    template <typename T, typename U = float2>
    struct Pair {
        T first;
        U second;
    };
    template <typename T>
    struct TemplateValue {
        T value;
        T get() { return value; }
    };
    template <> struct TemplateValue<float> {
        float value;
        float get() { return value * 2.0; }
    };
    RWStructuredBuffer<float> f : register(u0);
    RWStructuredBuffer<int> g : register(u1);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        Box<float> a;
        a.value = f[1];
        Box<int> b;
        b.value = g[1];
        Pair<float2> p;
        p.first = float2(a.twice(), 1.0);
        p.second = p.first;
        TemplateValue<float> special;
        special.value = f[2];
        TemplateValue<int> plain;
        plain.value = g[2];
        f[0] = p.second.x + special.get();
        g[0] = b.twice() + plain.get();
    }
""").strip()


def import_hlsl(code):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)


def test_struct_templates_become_generic_structures():
    crossgl = import_hlsl(PROGRAM)

    assert "generic<T> struct Box {\n        T value;" in crossgl
    assert "generic<T, U = vec2> struct Pair {" in crossgl
    assert "Box<float> a;" in crossgl and "Box<int> b;" in crossgl
    # A default argument is spelled out at each use.
    assert "Pair<vec2, vec2> p;" in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_defaults_that_name_earlier_parameters_take_their_arguments():
    crossgl = import_hlsl(textwrap.dedent("""
        template <typename T, typename U = vector<T, 2>, typename V = T>
        struct Lane {
            T a;
            U b;
            V c;
        };
        RWStructuredBuffer<int> g : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            Lane<int> lane;
            lane.a = g[1];
            lane.b = int2(lane.a, 2);
            lane.c = lane.b.y;
            g[0] = lane.a + lane.c;
        }
    """).strip())

    assert "Lane<int, ivec2, int> lane;" in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_member_functions_are_generic_over_their_receiver():
    crossgl = import_hlsl(PROGRAM)

    assert "generic<T> T Box_twice(in Box<T> __this)" in crossgl
    assert "vec2(Box_twice(a), 1.0)" in crossgl
    assert "Box_twice(b)" in crossgl


def test_explicit_specializations_are_the_structures_their_uses_name():
    crossgl = import_hlsl(PROGRAM)

    assert "struct TemplateValue_float_specialization {" in crossgl
    assert "TemplateValue_float_specialization special;" in crossgl
    assert "TemplateValue<int> plain;" in crossgl
    # Each receiver calls the member functions of its own structure.
    assert "TemplateValue_float_specialization_get(special)" in crossgl
    assert "TemplateValue_get(plain)" in crossgl
    assert "return __this.value * 2.0;" in crossgl


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "metal":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "out.metal"
        path.write_text(output)
        command = [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"]
    else:
        tool = shutil.which("dxc")
        if tool is None:
            pytest.skip("dxc is not installed")
        path = tmp_path / "out.hlsl"
        path.write_text(output)
        command = [tool, "-HV", "2021", "-T", "cs_6_2", "-E", "CSMain", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx"])
def test_struct_templates_translate_to_valid_targets(tmp_path, backend):
    path = tmp_path / "templates.hlsl"
    path.write_text(PROGRAM)

    _validate(tmp_path, backend, crosstl.translate(str(path), backend=backend))


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (
            """
            template <typename T, typename U> struct Pair { T a; U b; };
            template <typename T> struct Pair<T, int> { T a; };
            RWStructuredBuffer<float> f : register(u0);
            [numthreads(1, 1, 1)] void main() { Pair<float, int> p; p.a = 1.0; f[0] = p.a; }
            """,
            "'Pair' is a partial specialization",
        ),
    ],
)
def test_struct_templates_without_a_crossgl_form_are_refused(code, reason):
    with pytest.raises(DirectXImportRefused, match=reason):
        import_hlsl(textwrap.dedent(code))


def test_unnamed_unsupported_struct_templates_are_left_out():
    # A trait library the program never names loses nothing.
    crossgl = import_hlsl(textwrap.dedent("""
        template <bool B, typename T> struct enable_if {};
        template <typename T> struct enable_if<true, T> { using type = T; };
        RWStructuredBuffer<float> f : register(u0);
        [numthreads(1, 1, 1)] void main() { f[0] = 1.0; }
    """))

    assert "enable_if" not in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
