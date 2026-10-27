"""``out`` and ``inout`` value parameters are references on CUDA and HIP (#3159).

CUDA and HIP have no ``out`` or ``inout`` qualifiers. The generators used to declare such a parameter by value, so
``void setx(inout float x) { x = 5.0; }`` became ``void setx(float x)`` and the write never reached the caller. A value
parameter written through ``out`` or ``inout`` is now declared ``T&``, and an array ``T (&v)[N]``. Pointers and
resources keep their declaration, and so do stage entry points. A swizzle of several components passed to such a
parameter is refused, because CUDA and HIP build it as a new vector that a reference cannot bind to.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.lowering_refusals import LoweringRefused

GENERATORS = {"cuda": CudaCodeGen, "hip": HipCodeGen}
BACKEND_NAMES = {"cuda": "CUDA", "hip": "HIP"}

ISSUE = """
shader main {
    buffer float* results;
    void setx(inout float x) { x = 5.0; }
    void sety(out float y) { y = 5.0; }
    compute {
    @ numthreads(1, 1, 1)
    void main() { float a = 1.0; float b; setx(a); sety(b); results[0] = a + b; }
    }
}
"""

VALUES = """
shader main {
    struct P { float a; vec2 b; };
    enum Mode { Off, On = 2 }
    buffer float* results;
    void setv(inout vec4 v, out P p, in float k, float j, float arr[2]) {
        v = vec4(k);
        p.a = j;
        arr[0] = k;
    }
    void setm(inout mat4 m, out ivec3 i, inout uint u, out bool flag) {
        m = mat4(1.0);
        i = ivec3(1, 2, 3);
        u = u + 1u;
        flag = true;
    }
    void seth(inout half h, out f16vec2 hv, out Mode mode) {
        h = h + half(1.0);
        hv = f16vec2(h);
        mode = On;
    }
    void fill(out float o[2], inout float w[2]) {
        o[0] = 0.0;
        w[1] = 7.0;
    }
    void setx(inout float x);
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        vec4 v;
        P p;
        float arr[2];
        mat4 m;
        ivec3 i;
        uint u = 1u;
        bool flag;
        half h = half(1.0);
        f16vec2 hv;
        Mode mode = Off;
        float o[2];
        setv(v, p, 1.0, 2.0, arr);
        setm(m, i, u, flag);
        seth(h, hv, mode);
        fill(o, arr);
        setx(v.x);
        results[0] = v.x + p.a + arr[0] + float(i.y) + float(u);
    }
    }
    void setx(inout float x) { x = 5.0; }
}
"""

LVALUES = """
shader main {
    struct P { float a; vec2 b; };
    buffer float* results;
    void setx(inout float x) { x = x + 5.0; }
    void set2(inout vec2 w) { w = w * 2.0; }
    float sum2(vec2 w) { return w.x + w.y; }
    void bump<T>(inout T value, T step) { value = value + step; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        vec4 v = vec4(1.0);
        P p;
        p.a = 1.0;
        p.b = vec2(1.0);
        P ps[2];
        float arr[2];
        setx(results[0]);
        setx(v.x);
        setx(v.g);
        setx(p.a);
        setx(arr[1]);
        setx(ps[1].b.y);
        set2(p.b);
        set2(ps[0].b);
        bump<float>(arr[0], 2.0);
        results[1] = sum2(v.xy) + p.a + arr[1];
    }
    }
}
"""

HLSL = """
RWStructuredBuffer<float> results : register(u0);
struct P { float a; float2 b; };
void setx(inout float x) { x = 5.0; }
void setp(out P p, inout float3 v) { p.a = 2.0; p.b = float2(1.0, 2.0); v = v * 2.0; }
[numthreads(1, 1, 1)]
void main() { float a = 1.0; P p; float3 v = float3(1.0, 1.0, 1.0); setx(a); setp(p, v); results[0] = a + p.a + v.x; }
"""


def _generate(source, backend):
    return GENERATORS[backend]().generate(crosstl.translator.parse(source))


def _signature(code, name):
    match = re.search(rf"__device__ \w+ {name}\(([^)]*)\)", code)
    assert match, name
    return match.group(1)


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_issue_example_passes_references(backend):
    code = _generate(ISSUE, backend)
    assert _signature(code, "setx") == "float& x"
    assert _signature(code, "sety") == "float& y"
    # A reference binds to the caller's variable, so the call is unchanged.
    assert "setx(a);" in code
    assert "sety(b);" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_value_types_are_references(backend):
    code = _generate(VALUES, backend)
    assert _signature(code, "setv") == "float4& v, P& p, float k, float j, float arr[2]"
    if backend == "hip":
        # HIP spells uint as unsigned int and declares enumerations as int.
        assert _signature(code, "setm") == (
            "float4x4& m, int3& i, unsigned int& u, bool& flag"
        )
        assert _signature(code, "seth") == "half& h, half2& hv, int& mode"
    else:
        assert _signature(code, "setm") == "float4x4& m, int3& i, uint& u, bool& flag"
        assert _signature(code, "seth") == "half& h, half2& hv, Mode& mode"


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_array_parameters_are_references_to_the_array(backend):
    # A reference to the array keeps its extent, and writes through it reach the caller.
    code = _generate(VALUES, backend)
    assert "void fill(float (&o)[2], float (&w)[2])" in code
    assert "fill(o, arr);" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_declaration_matches_definition(backend):
    code = _generate(VALUES, backend)
    assert re.findall(r"__device__ void setx\(([^)]*)\)", code) == [
        "float& x",
        "float& x",
    ]


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_lvalue_arguments_bind_to_references(backend):
    code = _generate(LVALUES, backend)
    assert _signature(code, "setx") == "float& x"
    assert _signature(code, "set2") == "float2& w"
    assert _signature(code, "bump_float") == "float& value, float step"
    for call in (
        "setx(results[0]);",
        "setx(v.x);",
        "setx(v.y);",
        "setx(p.a);",
        "setx(arr[1]);",
        "setx(ps[1].b.y);",
        "set2(p.b);",
        "set2(ps[0].b);",
        "bump_float(arr[0], 2.0f);",
    ):
        assert call in code, call


@pytest.mark.parametrize("backend", sorted(GENERATORS))
@pytest.mark.parametrize(
    "call, swizzle, index, function, qualifier",
    [
        ("set2(v.xy);", "xy", 1, "set2", "inout"),
        ("put2(1.0, p.b.yx);", "yx", 2, "put2", "out"),
        ("setx(v.zw.x);", "zw", 1, "setx", "inout"),
    ],
)
def test_swizzle_of_several_components_is_refused(
    backend, call, swizzle, index, function, qualifier
):
    source = f"""
    shader main {{
        struct P {{ float a; vec2 b; }};
        buffer float* results;
        void setx(inout float x) {{ x = 5.0; }}
        void set2(inout vec2 w) {{ w = w * 2.0; }}
        void put2(float k, out vec2 w) {{ w = vec2(k); }}
        void bump<T>(inout T value, T step) {{ value = value + step; }}
        compute {{
        @ numthreads(1, 1, 1)
        void main() {{
            vec4 v = vec4(1.0);
            P p;
            {call}
            results[0] = v.x;
        }}
        }}
    }}
    """
    name = BACKEND_NAMES[backend]
    with pytest.raises(LoweringRefused) as excinfo:
        _generate(source, backend)
    message = str(excinfo.value)
    assert f"{function}" in message
    assert f"'.{swizzle}' as argument {index}" in message
    assert f"{name} builds a swizzle of several components as a new vector" in message
    assert "a reference cannot bind to" in message


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_swizzle_passed_by_value_is_unchanged(backend):
    code = _generate(LVALUES, backend)
    assert _signature(code, "sum2") == "float2 w"
    assert "sum2(make_float2(v.x, v.y))" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_hlsl_out_and_inout_parameters_are_references(tmp_path, backend):
    path = tmp_path / "probe.hlsl"
    path.write_text(HLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert _signature(code, "setx") == "float& x"
    assert _signature(code, "setp") == "P& p, float3& v"


def _compile_and_run(tmp_path, source):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no C++ compiler is installed")
    program = tmp_path / "probe.cpp"
    program.write_text(source, encoding="utf-8")
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-w", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    return ran.stdout.split()


# The helpers compile as host C++ once __device__ is defined away, so the writes can be checked without a GPU.
@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_issue_example_writes_reach_the_caller(tmp_path, backend):
    code = _generate(ISSUE, backend)
    start = code.index("__device__ void setx(")
    kernel = re.compile(r'(?:extern "C"\s+)?__global__').search(code, start)
    stdout = _compile_and_run(
        tmp_path,
        "#define __device__\n"
        + code[start : kernel.start()]
        + "#include <cstdio>\nint main() { float a = 1.0f; float b = 0.0f;"
        ' setx(a); sety(b); std::printf("%g", a + b); return 0; }\n',
    )
    # The issue's shader writes 10 to results[0].
    assert stdout == ["10"]
