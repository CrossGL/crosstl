"""OpenCL C built-ins that store a result through a pointer, and the C11 compare-and-exchange atomics.

``fract``, ``frexp``, ``modf``, ``sincos`` and ``remquo`` pass the value their pointer addresses as a CrossGL ``out``
argument, and ``atomic_compare_exchange_*`` lowers to ``atomicCompareExchange`` statements. The importer copied
``&x`` into both, and DXC rejected the HLSL with "operator is not supported" (#2834).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.OpenCL.compare_exchange import (
    OpenCLCompareExchangeUnsupportedError,
)
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.backend.OpenCL.pointer_results import OpenCLPointerResultUnsupportedError
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

KERNEL = """
__kernel void k(__global float* a, __global float* b, __global int* e,
                __global atomic_int* counter, __global uint* flags) {{
    uint i = get_global_id(0);
{body}
}}
"""


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return crossgl


def kernel_crossgl(body):
    return to_crossgl(KERNEL.format(body=body))


def flat(code):
    return re.sub(r"\s+", " ", code)


@pytest.mark.parametrize(
    "call, expected",
    [
        ("float r = frexp(a[i], &n);", "var r: f32 = frexp(a[i], n);"),
        ("float r = frexp(a[i], e + i);", "var r: f32 = frexp(a[i], e[i]);"),
        ("float r = modf(a[i], &b[i]);", "var r: f32 = modf(a[i], b[i]);"),
        ("float r = modf(a[i], b + i + 2);", "var r: f32 = modf(a[i], b[(i + 2)]);"),
        ("float r = remquo(a[i], b[i], &n);", "var r: f32 = remquo(a[i], b[i], n);"),
    ],
)
def test_pointer_result_is_passed_as_the_value_it_addresses(call, expected):
    crossgl = kernel_crossgl(f"    int n;\n    {call}\n    b[i] = r + (float)n;")
    assert expected in flat(crossgl)
    assert "&" not in crossgl.split("// Kernel: k", 1)[1]


def test_fract_calls_a_helper_with_the_opencl_edge_cases():
    crossgl = kernel_crossgl("    float w;\n    b[i] = fract(a[i], &w) + w;")
    assert "b[i] = (cl_fract_f32(a[i], w) + w);" in crossgl
    helper = flat(crossgl.split("// Kernel: k", 1)[0])
    # OpenCL C 3.0 section 7.5.1: fract(NaN) is the NaN, fract(+-inf) a zero of its sign, fract(+-0) itself, and
    # the result never reaches 1.0.
    assert "f32 cl_fract_f32(f32 x, out f32 whole) {" in helper
    assert "whole = floor_x;" in helper
    assert "if (isnan(x)) { return x; }" in helper
    assert "if (isinf(x)) { return (x > 0.0) ? 0.0 : -0.0; }" in helper
    assert "if (x == 0.0) { return x; }" in helper
    assert "return min(x - floor_x, 0.999999940395355224609375);" in helper


def test_vector_fract_applies_the_scalar_helper_to_each_component():
    crossgl = kernel_crossgl(
        "    float3 w;\n    float3 v = (float3)(a[i], b[i], 1.0f);\n"
        "    float3 f = fract(v, &w);\n    b[i] = f.x + w.z;"
    )
    helper = flat(crossgl.split("// Kernel: k", 1)[0])
    assert "vec3<f32> cl_fract_vec3_f32(vec3<f32> x, out vec3<f32> whole) {" in helper
    for component in "xyz":
        assert f"result.{component} = cl_fract_f32(x.{component}, part);" in helper
        assert f"whole.{component} = part;" in helper
    # The scalar helper is declared before the vector helper that calls it.
    assert helper.index("f32 cl_fract_f32(") < helper.index("cl_fract_vec3_f32(")
    assert "var f: vec3<f32> = cl_fract_vec3_f32(v, w);" in crossgl


@pytest.mark.parametrize(
    "opencl_type, crossgl_type, below_one",
    [
        ("half", "f16", "0.99951171875"),
        ("double", "f64", "0.99999999999999988897769753748434595763683319091796875"),
    ],
)
def test_fract_uses_the_largest_value_below_one_of_its_type(
    opencl_type, crossgl_type, below_one
):
    crossgl = to_crossgl(
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        f"__kernel void k(__global {opencl_type}* a) {{\n"
        f"    {opencl_type} w;\n    a[0] = fract(a[1], &w);\n    a[2] = w;\n}}\n"
    )
    assert f"{crossgl_type} cl_fract_{crossgl_type}(" in crossgl
    assert f"return min(x - floor_x, {below_one});" in crossgl


def test_sincos_returns_the_sine_and_stores_the_cosine():
    crossgl = kernel_crossgl(
        "    float4 c;\n    float4 v = (float4)(a[i]);\n    float4 s = sincos(v, &c);\n"
        "    b[i] = s.x + c.y;"
    )
    helper = flat(crossgl.split("// Kernel: k", 1)[0])
    # The sine and cosine are imported the way OpenCL's sin and cos are.
    assert (
        "vec4<f32> cl_sincos_vec4_f32(vec4<f32> x, out vec4<f32> cosine) { "
        "cosine = __cgl_libm_cos(x); return __cgl_libm_sin(x); }"
    ) in helper
    assert "var s: vec4<f32> = cl_sincos_vec4_f32(v, c);" in crossgl


def test_each_helper_is_declared_once_with_a_name_the_program_does_not_use():
    crossgl = kernel_crossgl(
        "    float cl_fract_f32 = a[i];\n    float w;\n"
        "    b[i] = fract(cl_fract_f32, &w) + fract(w, &w);"
    )
    assert crossgl.count("f32 cl_fract_f32_1(f32 x, out f32 whole) {") == 1
    assert "(cl_fract_f32_1(cl_fract_f32, w) + cl_fract_f32_1(w, w))" in crossgl


@pytest.mark.parametrize(
    "call, message",
    [
        ("float8 w; float8 f = fract((float8)(1.0f), &w);", "no floating-point scalar"),
        ("float w; b[i] = fract(a[i], (float*)&b[i]);", "cast pointer"),
    ],
)
def test_pointer_results_crossgl_cannot_carry_are_refused(call, message):
    with pytest.raises(OpenCLPointerResultUnsupportedError, match=message):
        kernel_crossgl(f"    {call}")


COMPARE_EXCHANGE = """\
    int expected = 0;
    bool ok = atomic_compare_exchange_strong(&counter[0], &expected, 5);
    if (atomic_compare_exchange_weak_explicit(&counter[1], &expected, 7,
                                              memory_order_relaxed, memory_order_relaxed)) {
        flags[i] = 1u;
    } else if (atomic_compare_exchange_strong(counter + 2, &expected, 9)) {
        flags[i] = 2u;
    }
    atomic_compare_exchange_weak(&counter[3], &expected, 11);
    flags[i + 1] = ok ? (uint)expected : 0u;"""


def test_compare_exchange_writes_the_original_value_back_when_it_fails():
    crossgl = flat(kernel_crossgl(COMPARE_EXCHANGE))
    assert (
        "var cl_expected_0: i32 = expected; "
        "var cl_original_0: i32 = atomicCompareExchange(counter[0], cl_expected_0, 5); "
        "if (cl_original_0 != cl_expected_0) { expected = cl_original_0; } "
        "var ok: bool = (cl_original_0 == cl_expected_0);"
    ) in crossgl
    # The explicit form with relaxed orders drops them, and an if head is lowered before the if.
    assert (
        "var cl_original_1: i32 = atomicCompareExchange(counter[1], cl_expected_1, 7); "
        "if (cl_original_1 != cl_expected_1) { expected = cl_original_1; } "
        "if ((cl_original_1 == cl_expected_1)) {"
    ) in crossgl
    # An else-if head is evaluated only in the else branch, so its lowering is emitted there.
    assert (
        "} else { var cl_expected_2: i32 = expected; "
        "var cl_original_2: i32 = atomicCompareExchange(counter[2], cl_expected_2, 9);"
    ) in crossgl
    # A call whose value is discarded leaves no statement behind its lowering.
    assert (
        "if (cl_original_3 != cl_expected_3) { expected = cl_original_3; } "
        "flags[(i + 1)] = (ok ? u32(expected) : 0u);"
    ) in crossgl


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "    int e0 = 0;\n    while (!atomic_compare_exchange_weak(&counter[0], &e0, e0 + 1)) { }",
            "conditionally or more than once",
        ),
        (
            "    int e0 = 0;\n    bool ok = flags[i] != 0u && "
            "atomic_compare_exchange_strong(&counter[0], &e0, 1);",
            "conditionally or more than once",
        ),
        (
            "    int e0 = 0;\n    uint j = i;\n"
            "    atomic_compare_exchange_strong(&counter[0], &e[j++], 1);",
            "side effects",
        ),
        (
            "    int e0 = 0;\n    atomic_compare_exchange_strong_explicit(&counter[0], &e0, 1, "
            "memory_order_seq_cst, memory_order_relaxed);",
            None,
        ),
    ],
)
def test_compare_exchange_crossgl_cannot_evaluate_once_is_refused(body, message):
    if message is None:
        # The explicit form keeps the order checks of the other OpenCL C atomics.
        with pytest.raises(Exception, match="memory_order_relaxed"):
            kernel_crossgl(body)
        return
    with pytest.raises(OpenCLCompareExchangeUnsupportedError, match=message):
        kernel_crossgl(body)


def test_floating_point_compare_exchange_is_refused():
    source = (
        "__kernel void k(__global atomic_float* c, __global float* o) {\n"
        "    float e0 = o[0];\n    atomic_compare_exchange_strong(&c[0], &e0, 1.0f);\n    o[1] = e0;\n}\n"
    )
    with pytest.raises(OpenCLCompareExchangeUnsupportedError, match="bit by bit"):
        to_crossgl(source)


COMPILED = """\
float twice(float x) { float w; return fract(x, &w) + w; }
__kernel void k(__global float* a, __global float* b, __global int* e,
                __global atomic_int* counter, __global uint* flags) {
    uint i = get_global_id(0);
    float4 v = (float4)(a[i], -a[i], 1.5f, -0.25f);
    float4 whole;
    float4 f = fract(v, &whole);
    float4 c;
    float4 s = sincos(v, &c);
    b[i] = f.x + whole.y + s.z + c.w + twice(a[i]);
    b[i + 1] = fract(a[i], b + i + 2);
    b[i + 3] = modf(a[i], &b[i + 4]);
    int4 q;
    float4 m = frexp(v, &q);
    e[i] = q.x + (int)m.y;
%s
}
""" % COMPARE_EXCHANGE


def _run(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _translate(tmp_path, backend):
    source = tmp_path / "kernel.cl"
    source.write_text(COMPILED, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


def test_translation_compiles_as_hlsl(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    code = _translate(tmp_path, "directx")
    (tmp_path / "kernel.hlsl").write_text(code, encoding="utf-8")
    entry = re.findall(r"void\s+(\w+)\s*\(", code)[-1]
    _run(
        [
            dxc,
            "-T",
            "cs_6_6",
            "-E",
            entry,
            "-HV",
            "2021",
            "kernel.hlsl",
            "-Fo",
            "k.bin",
        ],
        tmp_path,
    )


def test_translation_compiles_as_glsl(tmp_path):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not on PATH")
    (tmp_path / "kernel.comp").write_text(
        _translate(tmp_path, "opengl"), encoding="utf-8"
    )
    _run(["glslangValidator", "kernel.comp"], tmp_path)


def test_translation_validates_as_spirv(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    (tmp_path / "kernel.spvasm").write_text(
        _translate(tmp_path, "vulkan"), encoding="utf-8"
    )
    _run(
        ["spirv-as", "--target-env", "vulkan1.1", "kernel.spvasm", "-o", "k.spv"],
        tmp_path,
    )
    _run(["spirv-val", "--target-env", "vulkan1.1", "k.spv"], tmp_path)


def test_translation_compiles_as_metal(tmp_path):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not on PATH")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    (tmp_path / "kernel.metal").write_text(
        _translate(tmp_path, "metal"), encoding="utf-8"
    )
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "kernel.metal", "-o", "k.air"],
        tmp_path,
    )
