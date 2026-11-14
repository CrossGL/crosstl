"""Written ``in`` array parameters keep value semantics on CUDA and HIP (#2251).

HLSL passes an array parameter without ``out`` or ``inout`` by value, and the HLSL importer records it as ``in``. A
C++ array parameter decays to a pointer, so a write in the callee used to reach the caller's array. CUDA and HIP now
receive a written ``in`` array under another name and copy it into a local that keeps the source name, as Metal does.
Unwritten ``in`` arrays, ``out`` and ``inout`` arrays and arrays without a qualifier keep their declaration.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen

GENERATORS = {"cuda": CudaCodeGen, "hip": HipCodeGen}

ISSUE_HLSL = """
RWStructuredBuffer<float> results : register(u0);
float first(float v[2]) { v[0] = 3.0; return v[0] + v[1]; }
[numthreads(1,1,1)] void main() { float a[2] = { 1.0, 2.0 }; results[1] = first(a); results[2] = a[0]; }
"""

ISSUE_GLSL = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B { float results[]; };
float first(float v[2]) { v[0] = 3.0; return v[0] + v[1]; }
void main() { float a[2] = float[2](1.0, 2.0); results[0] = first(a); results[1] = a[0]; }
"""

PROBE = """
shader main {
    buffer float* results;

    float written(in float v[2]) {
        v[0] = 3.0;
        return v[0] + v[1];
    }

    float grid(in float m[2][3]) {
        m[1][2] += 4.0;
        return m[1][2] + m[0][0];
    }

    float bumped(in int c[2]) {
        c[1]++;
        return float(c[1]);
    }

    float unwritten(in float u[2]) {
        return u[0] + u[1];
    }

    void fill(inout float w[2]) {
        w[1] = 7.0;
    }

    float through_inout(in float n[2]) {
        fill(n);
        return n[1];
    }

    float through_in(in float k[2]) {
        return written(k);
    }

    void reset(out float o[2]) {
        o[0] = 0.0;
        o[1] = 0.0;
    }

    float unqualified(float p[2]) {
        p[0] = 5.0;
        return p[0];
    }

    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float a[2] = {1.0, 2.0};
        float g[2][3] = {{1.0, 0.0, 0.0}, {0.0, 0.0, 2.0}};
        int c[2] = {1, 1};
        results[0] = written(a);
        results[1] = grid(g);
        results[2] = bumped(c);
        results[3] = through_inout(a);
        results[4] = through_in(a);
        results[5] = unwritten(a);
        results[6] = a[0] + a[1] + g[1][2] + float(c[1]);
    }
    }
}
"""


def _generate(source, backend):
    return GENERATORS[backend]().generate(crosstl.translator.parse(source))


def _signature(code, name):
    # Up to the parenthesis that closes the parameter list: ``float (&w)[2]`` has its own.
    match = re.search(rf"__device__ \w+ {name}\((.*?)\)\s*[{{;]", code)
    assert match, name
    return match.group(1)


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_hlsl_array_written_by_the_callee_is_copied(tmp_path, backend):
    path = tmp_path / "probe.hlsl"
    path.write_text(ISSUE_HLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert _signature(code, "first") == "float v_crosstl_input[2]"
    assert "float v[2];" in code
    # crosstl.translate runs clang-format when it is installed (as in CI), which may
    # break the copy loop over several lines, so whitespace is not significant here.
    assert re.search(
        r"for\s*\(int crosstl_v_i0 = 0; crosstl_v_i0 < int\(2\); \+\+crosstl_v_i0\)\s*"
        r"\{\s*v\[crosstl_v_i0\] = v_crosstl_input\[crosstl_v_i0\];\s*\}",
        code,
    )


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_glsl_array_written_by_the_callee_is_copied(tmp_path, backend):
    # GLSL passes the array by value as well; the importer marks it ``in`` (#3162).
    path = tmp_path / "probe.comp"
    path.write_text(ISSUE_GLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert _signature(code, "first") == "float v_crosstl_input[2]"
    assert "float v[2];" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_written_in_arrays_are_received_under_another_name(backend):
    code = _generate(PROBE, backend)
    assert _signature(code, "written") == "float v_crosstl_input[2]"
    assert _signature(code, "grid") == "float m_crosstl_input[2][3]"
    assert _signature(code, "bumped") == "int c_crosstl_input[2]"
    # Passing the array to an inout parameter writes it.
    assert _signature(code, "through_inout") == "float n_crosstl_input[2]"
    assert "float m[2][3];" in code
    assert (
        "m[crosstl_m_i0][crosstl_m_i1] = m_crosstl_input[crosstl_m_i0][crosstl_m_i1];"
        in code
    )


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_other_array_parameters_keep_their_declaration(backend):
    code = _generate(PROBE, backend)
    assert _signature(code, "unwritten") == "float u[2]"
    # The callee copies its own in array, so passing one along is not a write.
    assert _signature(code, "through_in") == "float k[2]"
    # out and inout arrays bind the caller's array (C++ reference to array).
    assert _signature(code, "fill").endswith("(&w)[2]")
    assert _signature(code, "reset").endswith("(&o)[2]")
    assert _signature(code, "unqualified") == "float p[2]"
    assert "crosstl_input" not in "".join(
        _signature(code, name)
        for name in ("unwritten", "through_in", "fill", "reset", "unqualified")
    )


def _device_functions(code):
    """The source of every ``__device__`` helper, which is also host C++."""
    start = code.index("__device__ float written(")
    end = code.index('extern "C"') if 'extern "C"' in code else code.index("__global__")
    return code[start:end]


HOST_MAIN = """
#include <cstdio>
int main() {
    float a[2] = {1.0f, 2.0f};
    float g[2][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 2.0f}};
    int c[2] = {1, 1};
    float r[7];
    r[0] = written(a);
    r[1] = grid(g);
    r[2] = bumped(c);
    r[3] = through_inout(a);
    r[4] = through_in(a);
    r[5] = unwritten(a);
    r[6] = a[0] + a[1] + g[1][2] + float(c[1]);
    for (float value : r) std::printf("%g ", value);
    return 0;
}
"""


# The helpers compile as host C++ once __device__ is defined away, so the copies can be checked without a GPU.
@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_callee_writes_do_not_reach_the_caller(tmp_path, backend):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no C++ compiler is installed")
    code = _generate(PROBE, backend)
    program = tmp_path / "probe.cpp"
    program.write_text(
        "#define __device__\n" + _device_functions(code) + HOST_MAIN, encoding="utf-8"
    )
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-w", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    # Direct3D: no call changes a, g or c, so the last sum is 1 + 2 + 2 + 1.
    assert ran.stdout.split() == ["5", "7", "2", "7", "5", "3", "6"]


INDIRECT = """
shader main {
    buffer float* results;

    struct S {
        float a;
    };

    void store(float* p) {
        *p = 9.0;
    }

    float peek(const float* p) {
        return *p;
    }

    float through_address(in float v[2]) {
        store(&v[0]);
        return v[0];
    }

    float through_pointer(in float v[2]) {
        float* q = &v[1];
        *q = 4.0;
        return v[1];
    }

    void fill<T>(inout float w[2], T k) {
        w[1] = float(k);
    }

    float through_explicit(in float n[2]) {
        fill<int>(n, 7);
        return n[1];
    }

    float through_dereference(in float v[2]) {
        *v = 3.0;
        return v[0];
    }

    float through_decay(in float v[2]) {
        store(v);
        return v[0];
    }

    float through_offset(in float v[2]) {
        store(v + 1);
        return v[1];
    }

    float through_arrow(in S v[2]) {
        v->a = 2.0;
        return v[0].a;
    }

    float through_offset_arrow(in S v[2]) {
        (v + 1)->a = 5.0;
        return v[1].a;
    }

    float read_through_const(in float v[2]) {
        return peek(v + 1);
    }

    float pointers(in const float* w[2]) {
        w[0] = w[1];
        return w[0][0];
    }

    float volatile_pointers(in volatile float* w[2]) {
        w[0][0] = 1.0;
        return w[0][0];
    }

    // Not run: the generators drop the parentheses of (*v)++.
    float through_dereference_increment(in float v[2]) {
        (*v)++;
        return v[0];
    }

    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float a[2] = {1.0, 2.0};
        results[0] = through_address(a);
        results[1] = through_pointer(a);
        results[2] = through_explicit(a);
        results[3] = a[0] + a[1];
    }
    }
}
"""

RESOURCE_ARRAY_HLSL = """
RWStructuredBuffer<float> results : register(u0);
RWStructuredBuffer<float> bufs[2] : register(u1);
void store(RWStructuredBuffer<float> b[2], uint i) { b[0][i] = 1.0; b[1][i] = 2.0; }
[numthreads(1,1,1)] void main(uint3 id : SV_DispatchThreadID) { store(bufs, id.x); results[0] = 1.0; }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_taking_the_address_of_an_element_is_a_write(backend):
    code = _generate(INDIRECT, backend)
    assert _signature(code, "through_address") == "float v_crosstl_input[2]"
    assert _signature(code, "through_pointer") == "float v_crosstl_input[2]"


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_explicitly_specialised_generic_call_is_a_write(backend):
    code = _generate(INDIRECT, backend)
    assert _signature(code, "through_explicit") == "float n_crosstl_input[2]"


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_writes_through_dereference_arrow_and_pointer_arguments(backend):
    code = _generate(INDIRECT, backend)
    for name in (
        "through_dereference",
        "through_dereference_increment",
        "through_decay",
        "through_offset",
    ):
        assert _signature(code, name) == "float v_crosstl_input[2]", name
    assert _signature(code, "through_arrow") == "S v_crosstl_input[2]"
    assert _signature(code, "through_offset_arrow") == "S v_crosstl_input[2]"
    # A const pointer parameter cannot write the array.
    assert _signature(code, "read_through_const") == "float v[2]"


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_pointer_arrays_are_not_copied(backend):
    code = _generate(INDIRECT, backend)
    assert "w_crosstl_input" not in code
    if backend == "hip":
        # HIP keeps the qualifiers of the pointer elements.
        assert _signature(code, "pointers") == "const float* w[2]"
        assert _signature(code, "volatile_pointers") == "volatile float* w[2]"
    else:
        # CUDA drops them for every pointer array, copied or not, as before.
        assert _signature(code, "pointers") == "float* w[2]"
        assert _signature(code, "volatile_pointers") == "float* w[2]"


RESOURCE_NAMED_STRUCTS = """
shader main {
    buffer float* results;
    struct Texture2DParams { float s; };
    struct sampler_cfg { float s; };
    struct Buffer { float s; };
    float texture_params(in Texture2DParams t[2]) { t[0].s = 1.0; return t[0].s; }
    float sampler_config(in sampler_cfg t[2]) { t[0].s = 1.0; return t[0].s; }
    float buffer_struct(in Buffer t[2]) { t[0].s = 1.0; return t[0].s; }
    compute {
    @ numthreads(1, 1, 1)
    void main() { results[0] = 1.0; }
    }
}
"""

RESOURCE_NAMED_STRUCT_HLSL = """
RWStructuredBuffer<float> results : register(u0);
struct Texture2DParams { float s; };
float tp(Texture2DParams t[2]) { t[0].s = 1.0; return t[0].s; }
[numthreads(1,1,1)] void main() { Texture2DParams p[2]; p[0].s = 0; p[1].s = 0; results[0] = tp(p) + p[0].s; }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_structs_named_like_resources_are_copied(tmp_path, backend):
    code = _generate(RESOURCE_NAMED_STRUCTS, backend)
    assert _signature(code, "texture_params") == "Texture2DParams t_crosstl_input[2]"
    assert _signature(code, "sampler_config") == "sampler_cfg t_crosstl_input[2]"
    assert _signature(code, "buffer_struct") == "Buffer t_crosstl_input[2]"
    path = tmp_path / "params.hlsl"
    path.write_text(RESOURCE_NAMED_STRUCT_HLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert _signature(code, "tp") == "Texture2DParams t_crosstl_input[2]"


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_structured_buffer_arrays_are_not_copied(tmp_path, backend):
    path = tmp_path / "resources.hlsl"
    path.write_text(RESOURCE_ARRAY_HLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert re.match(r"float\* b\[2\], ", _signature(code, "store"))
    assert "crosstl_input" not in code


INDIRECT_HOST_MAIN = """
#include <cstdio>
int main() {
    float a[2] = {1.0f, 2.0f};
    S s[2] = {{1.0f}, {2.0f}};
    float r[11];
    r[0] = through_address(a);
    r[1] = through_pointer(a);
    r[2] = through_explicit(a);
    r[3] = through_dereference(a);
    r[4] = through_decay(a);
    r[5] = through_offset(a);
    r[6] = read_through_const(a);
    r[7] = a[0] + a[1];
    r[8] = through_arrow(s);
    r[9] = through_offset_arrow(s);
    r[10] = s[0].a + s[1].a;
    for (float value : r) std::printf("%g ", value);
    return 0;
}
"""


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


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_writes_through_addresses_do_not_reach_the_caller(tmp_path, backend):
    code = _generate(INDIRECT, backend)
    start = code.index("__device__ void store(")
    end = code.index("__device__ float pointers(")
    stdout = _compile_and_run(
        tmp_path,
        "#define __device__\nstruct S { float a; };\n"
        + code[start:end]
        + INDIRECT_HOST_MAIN,
    )
    assert stdout == ["9", "4", "7", "3", "9", "9", "2", "3", "2", "5", "3"]


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_issue_example_keeps_the_callers_array(tmp_path, backend):
    path = tmp_path / "probe.hlsl"
    path.write_text(ISSUE_HLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    start = code.index("__device__ float first(")
    end = code.index('extern "C"') if 'extern "C"' in code else code.index("__global__")
    stdout = _compile_and_run(
        tmp_path,
        "#define __device__\n"
        + code[start:end]
        + "#include <cstdio>\nint main() { float a[2] = {1.0f, 2.0f};"
        ' float r = first(a); std::printf("%g %g", r, a[0]); return 0; }\n',
    )
    # Direct3D writes 5 to results[1] and 1 to results[2].
    assert stdout == ["5", "1"]
