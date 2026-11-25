"""CUDA and HIP output: vector math, bit counts, sized casts, zero vectors and case scopes.

Programs imported from Metal, GLSL, HLSL, SPIR-V and Slang reached NVCC and hipcc with calls they reject:

- ``acosh``, ``fmod``, ``copysign``, ``ilogb`` and the other C math functions on vectors, which the CUDA Math API
  and HIP declare for ``float`` and ``double`` only ("no instance of overloaded function");
- ``bitCount`` of a vector as a ``uint`` vector, although it returns ``int`` components (GLSL 4.60, 8.8), so
  ``int3 r = bitCount(v)`` did not compile;
- ``int16(x)``, ``uint8(x)`` and ``float16_t(x)`` constructor casts, which name no CUDA type;
- ``vec4{}`` as ``make_float4()``, which takes four arguments;
- two ``case`` labels that declare one name, which C++ rejects as a redefinition;
- a vector initializing or assigned to a vector of another component type, which GLSL and CrossGL convert
  implicitly (GLSL 4.60, 4.1.10) and CUDA's distinct vector structures do not;
- SPIR-V's subgroup built-in inputs, declared as stage variables (``uint gl_SubgroupSize @input``), as
  uninitialized host globals.
"""

import shutil
import subprocess

import pytest

import crosstl

BACKENDS = ["cuda", "hip"]

VECTOR_MATH = """
shader main {
    compute {
        layout(local_size_x = 64) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec3 a = vec3(float(i), 2.0, 3.0);
            dvec2 d = dvec2(double(i), 0.5);
            vec3 h = acosh(a) + asinh(a) + atanh(a) + log10(a) + rint(a) + fabs(a);
            vec3 b = fmod(a, vec3(2.0)) + copysign(a, -a) + fdim(a, a) + nextafter(a, a);
            vec3 m = fmax(a, a) + fmin(a, 1.5);
            ivec3 e = ilogb(a);
            dvec2 g = fmod(d, d) + acosh(d);
            float s = acosh(a.x) + fmod(a.y, 2.0);
            outp[i] = h.x + b.y + m.z + float(e.x) + float(g.y) + s;
        }
    }
}
"""

BITS_AND_CASTS = """
shader main {
    compute {
        layout(local_size_x = 64) in;
        buffer float* outp;
        buffer int* iin;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            uvec3 u = uvec3(i, 1u, 2u);
            ivec3 c = bitCount(u);
            ivec3 n = bitCount(ivec3(c.x - 1, 2, 3)) - ivec3(1);
            int8 a = int8(iin[i]);
            uint8 b = uint8(iin[i]);
            int16 s = int16(a * b);
            uint16 w = uint16(s);
            float16_t f = float16_t(outp[i]);
            vec4 z = vec4{};
            ivec2 k = ivec2{};
            float acc = 0.0;
            switch (iin[i]) {
                case 0:
                    float t = outp[i];
                    acc = t * 2.0;
                    break;
                case 1:
                    float t = 3.0;
                    acc = t;
                    break;
                default:
                    acc = 1.0;
            }
            outp[i] = float(c.x + n.y) + float(a) + float(w) + float(f) + z.w + float(k.y) + acc;
        }
    }
}
"""

CONVERSIONS = """
shader main {
    compute {
        layout(local_size_x = 64) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            ivec3 s = ivec3(int(i), -2, 3);
            uvec3 u = s;
            vec3 f = s;
            dvec2 d = vec2(0.5, 1.5);
            uvec3 same = uvec3(1u, 2u, 3u);
            u = s * 2;
            outp[i] = float(u.x) + f.y + float(d.y) + float(same.z);
        }
    }
}
"""

SUBGROUP_INPUTS = """
shader main {
    struct B0 { float outp[]; };
    RWStructuredBuffer<B0> results @set(0) @binding(0);
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;
        uint gl_SubgroupSize @input @gl_SubgroupSize;
        uint gl_SubgroupInvocationID @input @gl_SubgroupInvocationID;
        uint gl_SubgroupID @input @gl_SubgroupID;
        uint gl_NumSubgroups @input @gl_NumSubgroups;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            results[0].outp[i] = float(gl_SubgroupSize + gl_SubgroupInvocationID + gl_SubgroupID + gl_NumSubgroups);
        }
    }
}
"""

METAL = """
#include <metal_stdlib>
using namespace metal;
kernel void probe(device const float* fin [[buffer(0)]], device const int* iin [[buffer(1)]],
                  device float* outp [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float3 a = float3(fin[i], fin[i + 1u], fin[i + 2u]);
    float3 r = acosh(a) + fmod(a, float3(2.0f)) + copysign(a, -a) + fmax(a, a);
    int3 e = ilogb(a);
    int3 p = popcount(int3(iin[i], iin[i + 1u], iin[i + 2u]));
    short v = short(iin[i]);
    outp[i] = r.x + float(e.y) + float(p.z) + float(short(v * v));
}
"""

HLSL = """
RWStructuredBuffer<float> outp : register(u0);
StructuredBuffer<float> fin : register(t0);
StructuredBuffer<int> iin : register(t1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float acc = 0.0;
    switch (iin[i]) {
        case 0:
            float t = fin[i];
            acc = t * 2.0;
            break;
        default:
            float t = 3.0;
            acc = t;
            break;
    }
    float2 m = fmod(float2(fin[i], -fin[i]), float2(4.0, -2.0));
    outp[i] = acc + m.x + m.y;
}
"""


def _generate(tmp_path, backend, source, suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_math_applies_the_scalar_function_per_component(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, VECTOR_MATH))

    assert (
        "__device__ inline float3 cgl_float3_acosh(float3 value) { return "
        "make_float3(acoshf(value.x), acoshf(value.y), acoshf(value.z)); }" in code
    )
    for name in ("asinh", "atanh", "log10", "rint", "fabs"):
        assert f"cgl_float3_{name}(a)" in code
        assert f"{name}f(value.z)" in code
    assert (
        "__device__ inline float3 cgl_float3_fmod(float3 lhs, float3 rhs) { return "
        "make_float3(fmodf(lhs.x, rhs.x), fmodf(lhs.y, rhs.y), fmodf(lhs.z, rhs.z)); }"
        in code
    )
    for name in ("copysign", "fdim", "nextafter", "fmax"):
        assert f"cgl_float3_{name}(a, " in code
    # A scalar operand applies to every component.
    assert "__device__ inline float3 cgl_float3_fmin_vs(float3 lhs, float rhs)" in code
    assert "fminf(lhs.x, rhs)" in code
    # ilogb returns int components.
    assert "__device__ inline int3 cgl_float3_ilogb(float3 value)" in code
    assert "int3 e = cgl_float3_ilogb(a);" in code
    # double vectors call the double functions.
    assert "make_double2(fmod(lhs.x, rhs.x), fmod(lhs.y, rhs.y))" in code
    assert "make_double2(acosh(value.x), acosh(value.y))" in code
    # Scalars keep the library call.
    assert "float s = (acosh(a.x) + fmod(a.y, 2.0f));" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_bit_counts_casts_zero_vectors_and_case_scopes(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, BITS_AND_CASTS))

    assert "__device__ inline int3 cgl_uint3_bitCount(uint3 value)" in code
    assert "make_int3(__popc(value.x), __popc(value.y), __popc(value.z))" in code
    assert "int3 c = cgl_uint3_bitCount(u);" in code
    assert "__device__ inline int3 cgl_int3_bitCount(int3 value)" in code
    assert "char a = static_cast<char>(iin[i]);" in code
    assert "unsigned char b = static_cast<unsigned char>(iin[i]);" in code
    assert "short s = static_cast<short>((a * b));" in code
    assert "unsigned short w = static_cast<unsigned short>(s);" in code
    assert "half f = __float2half(outp[i]);" in code
    assert "float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in code
    assert "int2 k = make_int2(0, 0);" in code
    assert "case 0: { float t = outp[i]; acc = (t * 2.0f); break; }" in code
    assert "case 1: { float t = 3.0f; acc = t; break; }" in code
    # A case without declarations keeps its statements under the label.
    assert "default: acc = 1.0f; }" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_vectors_of_another_component_type_convert_implicitly(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, CONVERSIONS))

    assert "uint3 u = cgl_convert_to_uint3(s);" in code
    assert "float3 f = cgl_convert_to_float3(s);" in code
    assert "double2 d = cgl_convert_to_double2(make_float2(0.5f, 1.5f));" in code
    assert "u = cgl_convert_to_uint3(" in code
    # A value of the target type is written as it is.
    assert "uint3 same = make_uint3(1u, 2u, 3u);" in code
    assert (
        "template <typename T> __device__ inline float3 cgl_convert_to_float3(T value)"
        in code
    )
    assert "static_cast<float>(value.x)" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_subgroup_builtin_inputs_are_expressions(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, SUBGROUP_INPUTS))

    assert "uint gl_Subgroup" not in code and "int gl_Subgroup" not in code
    assert "gl_NumSubgroups" not in code
    assert "warpSize" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_metal_vector_math_reaches_cuda_and_hip(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, METAL, ".metal"))

    assert "cgl_float3_acosh(a)" in code
    assert "cgl_float3_fmod(a, " in code
    assert "cgl_float3_copysign(a, " in code
    assert "cgl_float3_fmax(a, a)" in code
    assert "int3 e = cgl_float3_ilogb(a);" in code
    assert "__device__ inline int3 cgl_int3_bitCount(int3 value)" in code
    assert "int16(" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_hlsl_cases_that_declare_one_name_get_their_own_blocks(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, HLSL, ".hlsl"))

    assert "case 0: { float t = " in code
    assert "default: { float t = 3.0f;" in code
    assert "cgl_float2_fmod(" in code


def _compile(tmp_path, tool, args, code, suffix):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / f"componentwise{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [tool, *args, str(source), "-o", str(tmp_path / "componentwise.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


SOURCES = [
    (VECTOR_MATH, ".cgl"),
    (BITS_AND_CASTS, ".cgl"),
    (CONVERSIONS, ".cgl"),
    (SUBGROUP_INPUTS, ".cgl"),
    (METAL, ".metal"),
    (HLSL, ".hlsl"),
]
IDS = [
    "vector-math",
    "bits-and-casts",
    "conversions",
    "subgroup-inputs",
    "metal",
    "hlsl",
]


@pytest.mark.parametrize(("source", "suffix"), SOURCES, ids=IDS)
def test_nvcc_compiles(tmp_path, source, suffix):
    code = _generate(tmp_path, "cuda", source, suffix)
    _compile(tmp_path, "nvcc", ["-std=c++17", "-c"], code, ".cu")


@pytest.mark.parametrize(("source", "suffix"), SOURCES, ids=IDS)
def test_hipcc_compiles(tmp_path, source, suffix):
    code = _generate(tmp_path, "hip", source, suffix)
    _compile(
        tmp_path,
        "hipcc",
        ["-std=c++17", "-c", "--offload-arch=gfx90a"],
        code,
        ".hip",
    )
