"""HLSL template value parameters import as CrossGL generic parameters bound at each use.

``template <typename T, uint N> T sum(T x)`` and ``template <typename T, uint N = 4> struct Ring { T items[N]; }``
were refused ("declares the value parameter 'uint N'"). They now import as ``generic<T, N>`` declarations whose calls
and uses name the value, typed as the parameter is (``sum<uint, 5u>``, ``Ring<float, 4u>``), with defaults spelled
out at each use. Each case below fails on that importer. The HLSL program compiles with DXC 1.10 (``-HV 2021``).
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

PROGRAM = textwrap.dedent("""
    template <typename T, int N>
    T scale(T x) { return x * N; }
    template <typename T, uint N>
    T sum(T x) {
        T total = (T)0;
        for (uint i = 0; i < N; i++) { total += x; }
        return total;
    }
    template <typename T, uint N = 4>
    struct Ring {
        T items[N];
        uint head;
    };
    RWStructuredBuffer<float> f : register(u0);
    RWStructuredBuffer<uint> g : register(u1);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        float x = f[1];
        uint y = g[1];
        Ring<float> r;
        Ring<uint, 2> s;
        r.head = 0;
        r.items[0] = scale<float, 3>(x);
        s.items[1] = sum<uint, 5>(y);
        f[0] = r.items[0];
        g[0] = s.items[1] + sum<uint, 2>(y);
    }
""").strip()


def import_hlsl(code):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)


def test_value_parameters_become_generic_parameters():
    crossgl = import_hlsl(PROGRAM)

    assert "generic<T, N> T scale(T x)" in crossgl
    assert "generic<T, N> T sum(T x)" in crossgl
    assert "generic<T, N> struct Ring {\n        T items[N];" in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_uses_bind_values_typed_as_their_parameters():
    crossgl = import_hlsl(PROGRAM)

    assert "scale<float, 3>(x)" in crossgl
    assert "sum<uint, 5u>(y)" in crossgl and "sum<uint, 2u>(y)" in crossgl
    # ``Ring<float>`` takes the default ``N = 4``.
    assert "Ring<float, 4u> r;" in crossgl
    assert "Ring<uint, 2u> s;" in crossgl


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools are not installed")
        path, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        path.write_text(output)
        subprocess.run(
            [assembler, "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        command = [validator, "--target-env", "vulkan1.1", str(binary)]
    elif backend == "wgsl":
        tool = shutil.which("naga")
        if tool is None:
            pytest.skip("naga is not installed")
        path = tmp_path / "out.wgsl"
        path.write_text(output)
        command = [tool, str(path)]
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


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan", "wgsl"])
def test_value_templates_translate_to_valid_targets(tmp_path, backend):
    path = tmp_path / "values.hlsl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend=backend)

    _validate(tmp_path, backend, output)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("opengl", ("float items[4];", "uint items[2];", "(x * 3)", "(i < 5u)")),
        ("metal", ("float items[4];", "uint items[2];", "x * 3", "i < 5u")),
    ],
)
def test_specializations_read_the_bound_values(tmp_path, backend, expected):
    path = tmp_path / "values.hlsl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend=backend)

    for text in expected:
        assert text in output, text


NAMESPACED = textwrap.dedent("""
    namespace util {
        template <typename T, uint N>
        T repeat(T x) {
            T total = (T)0;
            for (uint i = 0; i < N; i++) { total += x; }
            return total;
        }
    }
    RWStructuredBuffer<uint> g : register(u0);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        g[0] = util::repeat<uint, 3>(g[1]);
    }
""").strip()


def test_namespaced_value_templates_bind_their_values():
    crossgl = import_hlsl(NAMESPACED)

    assert "<uint, 3u>(" in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan"])
def test_namespaced_value_templates_translate_to_valid_targets(tmp_path, backend):
    path = tmp_path / "namespaced.hlsl"
    path.write_text(NAMESPACED)

    output = crosstl.translate(str(path), backend=backend)

    _validate(tmp_path, backend, output)
