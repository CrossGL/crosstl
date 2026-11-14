"""HLSL 2021 function templates import as CrossGL generic functions.

The importer used to skip ``template <typename T>`` and emit ``T twice(T x)`` with an undeclared ``T``; it dropped
``template <> float twice<float>(float x)`` altogether, so instantiations for ``float`` ran the primary's body, and
kept HLSL spellings in explicit template arguments (``twice<float2>``). Each case below fails on that importer.
"""

import re
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
    template <typename T> T twice(T x) { return x + x; }
    template <> float twice<float>(float x) { return x * 3.0; }
    template <> int twice(int x) { return x * 5; }
    template <typename T, typename U> T convert(U x) { return (T)x; }
    template <typename T = float> T zero() { return (T)0; }
    template <typename T> T sum3(vector<T, 3> v) { return v.x + v.y + v.z; }
    RWStructuredBuffer<float> f : register(u0);
    RWStructuredBuffer<int> g : register(u1);
    [numthreads(64, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        float a = f[id.x];
        int b = g[id.x];
        uint c = 7;
        float2 v = float2(a, 1.0);
        int3 w = int3(b, 2, 3);
        f[id.x] = twice(a) + twice<float2>(v).y + zero();
        g[id.x] = twice<int>(b) + convert<int, float>(a) + (int)twice(c) + sum3(w);
    }
""").strip()


def import_hlsl(code):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)


def translate(tmp_path, code, backend):
    path = tmp_path / "templates.hlsl"
    path.write_text(code)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_function_templates_become_generic_functions():
    crossgl = import_hlsl(PROGRAM)

    assert "generic<T> T twice(T x)" in crossgl
    assert "generic<T, U> T convert(U x)" in crossgl
    assert "generic<T = float> T zero()" in crossgl
    assert "generic<T> T sum3(vec3<T> v)" in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_explicit_specializations_replace_the_primary_body():
    crossgl = import_hlsl(PROGRAM)

    assert (
        "float twice_float_specialization(float x) @ explicit_specialization(twice, float)"
        in crossgl
    )
    # ``template <> int twice(int)`` deduces its argument from the signature.
    assert (
        "int twice_int_specialization(int x) @ explicit_specialization(twice, int)"
        in crossgl
    )
    assert "return x * 3.0;" in crossgl and "return x * 5;" in crossgl


def test_explicit_template_arguments_use_crossgl_spellings():
    crossgl = import_hlsl(PROGRAM)

    assert "twice<vec2>(v).y" in crossgl
    assert "twice<int>(b)" in crossgl
    assert "convert<int, float>(a)" in crossgl
    assert "twice<float2>" not in crossgl


def test_template_call_results_take_their_deduced_types():
    crossgl = import_hlsl(textwrap.dedent("""
        template <typename T> T twice(T x) { return x + x; }
        RWStructuredBuffer<float2> f : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            half h = (half)1;
            f[id.x] = twice(h).xx;
        }
    """))

    # The scalar ``half`` result widens through a vector constructor, not a swizzle of a scalar.
    assert "f16vec2(twice(h))" in crossgl


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


@pytest.mark.parametrize(
    ("backend", "specialized_call", "primary_instance"),
    [
        ("directx", "twice_float_specialization(a)", "float twice_float("),
        ("metal", "twice_float_specialization(a)", "float twice_float("),
        ("opengl", "twice_float_specialization(a)", "float twice_float("),
        ("wgsl", "twice_float_specialization(a)", "fn twice_f32("),
    ],
)
def test_instantiations_run_the_specialized_bodies(
    tmp_path, backend, specialized_call, primary_instance
):
    output = translate(tmp_path, PROGRAM, backend)

    assert specialized_call in output
    assert "twice_int_specialization(b)" in output
    assert primary_instance not in output
    _validate(tmp_path, backend, output)


def _spirv_functions(output):
    """``(result type, body)`` for every SPIR-V function definition."""
    functions, current = [], None
    for line in output.splitlines():
        words = line.split()
        if "OpFunction" in words:
            current = (words[words.index("OpFunction") + 1], [])
        elif "OpFunctionEnd" in words:
            functions.append((current[0], " ".join(current[1])))
            current = None
        elif current is not None:
            current[1].append(line.strip())
    return functions


def _spirv_ids(output, pattern):
    return {
        match[1] for match in re.finditer(rf"^\s*(%\w+) = {pattern}\s*$", output, re.M)
    }


def test_spirv_instantiations_run_the_specialized_bodies(tmp_path):
    output = translate(tmp_path, PROGRAM, "vulkan")
    functions = _spirv_functions(output)
    floats = _spirv_ids(output, "OpTypeFloat 32")
    ints = _spirv_ids(output, "OpTypeInt 32 1")
    three = _spirv_ids(output, r"OpConstant %\w+ 3(?:\.0+)?")
    five = _spirv_ids(output, r"OpConstant %\w+ 5")

    def multiplies(body, opcode, constants):
        return any(
            re.search(rf"{opcode} %\w+ %\w+ {re.escape(constant)}\b", body)
            for constant in constants
        )

    assert any(
        result in floats and multiplies(body, "OpFMul", three)
        for result, body in functions
    )
    assert any(
        result in ints and multiplies(body, "OpIMul", five)
        for result, body in functions
    )
    # No float or int clone of the primary's ``x + x``.
    assert not any(
        result in floats | ints and re.search(r"Op[IF]Add %\w+ (%\w+) \1\b", body)
        for result, body in functions
    )
    _validate(tmp_path, "vulkan", output)


def test_struct_template_prefix_does_not_reach_member_functions():
    crossgl = import_hlsl(textwrap.dedent("""
        template <typename T> struct Box { float v; float get() { return v; } };
        float helper(float x) { return x; }
        float4 main() : SV_Target { return float4(helper(1.0), 0.0, 0.0, 1.0); }
    """))

    # The struct template and its member function are generic; the function after them is not.
    assert "generic<T> struct Box {" in crossgl
    assert "generic<T> float Box_get(in Box<T> __this)" in crossgl
    assert "    float helper(float x)" in crossgl
    assert "generic<T> float helper" not in crossgl


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (
            "template <typename T, int N = 2> T scale(T x) { return x * (T)N; }",
            "gives the value parameter 'int N' a default",
        ),
        (
            "namespace util { template <typename T, int N = 2> T scale(T x) { return x * (T)N; } }",
            "function template 'util::scale' gives the value parameter 'int N' a default",
        ),
        (
            "template <typename T> T twice(T x);\n"
            "template <> float twice<float>(float x) { return x * 3.0; }",
            "specializes a function template with no definition",
        ),
        (
            "template <typename T> float make(int x) { return (float)(T)x; }\n"
            "template <> float make(int x) { return 1.0; }",
            "does not determine every template argument of 'make'",
        ),
        (
            "float plain(float x) { return x; }\n"
            "template <> float plain<float>(float x) { return x; }",
            "does not name exactly one function template",
        ),
    ],
)
def test_untranslatable_templates_are_refused(code, reason):
    program = (
        code + "\nfloat4 main() : SV_Target { return float4(1.0, 0.0, 0.0, 1.0); }"
    )

    with pytest.raises(DirectXImportRefused, match=reason):
        import_hlsl(program)
