"""Slang output of the HLSL, Metal, GLSL and CUDA constructs that the translation matrix found invalid.

Each test translates a native source straight to Slang, as a translation-matrix cell does, and compiles the Slang
for SPIR-V with slangc, with its SPIR-V validator on, when slangc is installed.
"""

import os
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.codegen.slang_value_calls import slang_empty_value_constructor


def slangc_check(tmp_path, output, entry=None, stage=None):
    slangc = shutil.which("slangc")
    if slangc is None:
        return
    path = tmp_path / "out.slang"
    path.write_text(output)
    command = [slangc, str(path), "-target", "spirv", "-o", str(tmp_path / "out.spv")]
    if entry is not None:
        command += ["-entry", entry]
    if stage is not None:
        command += ["-stage", stage]
    checked = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "SLANG_RUN_SPIRV_VALIDATION": "1"},
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def translate(tmp_path, source, suffix, **check):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    output = crosstl.translate(str(path), backend="slang")
    slangc_check(tmp_path, output, **check)
    return output


def refusal(tmp_path, source, suffix):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    with pytest.raises(SlangLoweringRefused) as raised:
        crosstl.translate(str(path), backend="slang")
    return str(raised.value)


HLSL_SWITCH = """RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<int> iin : register(u1);
RWStructuredBuffer<float> outp : register(u2);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float acc = 0.0;
    switch (iin[i] & 3) {
        case 0: { float t = fin[i]; acc = t * 2.0; break; }
        case 1: { float t = 3.0; acc = t; break; }
        default: { int t = iin[i] * 4; acc = float(t); break; }
    }
    outp[i] = acc;
}
"""


def test_switch_cases_keep_their_own_scopes(tmp_path):
    output = translate(tmp_path, HLSL_SWITCH, ".hlsl", entry="main")
    assert output.count("float t = ") == 2
    # clang-format may put a case's brace on its own line.
    flat = " ".join(output.split())
    assert "case 0: {" in flat and "default: {" in flat


def test_switch_case_that_falls_through_shares_its_local(tmp_path):
    source = HLSL_SWITCH.replace(
        "case 0: { float t = fin[i]; acc = t * 2.0; break; }\n        case 1: { float t = 3.0; acc = t; break; }",
        "case 0: float t = fin[i];\n        case 1: t = t + 1.0; acc = t; break;",
    )
    output = translate(tmp_path, source, ".hlsl", entry="main")
    assert "case 0: {" not in output


METAL_COMPUTE = """#include <metal_stdlib>
using namespace metal;
{structs}
kernel void probe(device float* fin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]{params}) {{
    uint i = id.x;
{body}
}}
"""


def metal(body, params="", structs=""):
    return METAL_COMPUTE.format(body=body, params=params, structs=structs)


def test_two_operand_sincos_returns_the_sine_and_writes_the_cosine(tmp_path):
    source = metal(
        "    float3 a0 = float3(fin[i], fin[i + 1], fin[i + 2]);\n"
        "    float3 c;\n"
        "    float3 s = sincos(a0, c);\n"
        "    outp[i] = s.x + c.y;"
    )
    output = translate(tmp_path, source, ".metal", entry="probe")
    assert "float3 cgl_sincos_float3(float3 x, out float3 cosval)" in output
    assert "sincos(x, sinval, cosval);" in output
    assert "cgl_sincos_float3(a0, c)" in output


def test_empty_initializers_are_zero_values(tmp_path):
    source = metal(
        "    float4 v = {};\n    int2 w = {};\n    float x = float();\n"
        "    outp[i] = v.x + float(w.y) + x;"
    )
    output = translate(tmp_path, source, ".metal", entry="probe")
    assert "float4 v = float4(0);" in output
    assert "int2 w = int2(0);" in output
    assert "float4()" not in output and "int2()" not in output


@pytest.mark.parametrize(
    "slang_type, expected",
    [
        ("float", "float(0)"),
        ("bool3", "bool3(false)"),
        ("half2x3", "half2x3(0)"),
        ("vector<int16_t, 3>", "vector<int16_t, 3>(0)"),
        ("Pair", None),
        ("Texture2D<float4>", None),
    ],
)
def test_empty_value_constructor_spelling(slang_type, expected):
    assert slang_empty_value_constructor(slang_type) == expected


@pytest.mark.parametrize(
    "hlsl, slang",
    [
        ("centroid", "centroid"),
        ("nointerpolation", "nointerpolation"),
        ("noperspective", "noperspective"),
        ("sample", "sample"),
    ],
)
def test_entry_parameters_keep_their_interpolation(tmp_path, hlsl, slang):
    source = (
        f"float4 main({hlsl} float2 uv : TEXCOORD0) : SV_Target {{\n"
        "    return float4(uv, 0.0, 1.0);\n}\n"
    )
    output = translate(tmp_path, source, ".hlsl", entry="main", stage="fragment")
    assert f"main({slang} float2 uv : TexCoord0)" in output


def test_barycentric_and_shading_rate_inputs_are_entry_parameters(tmp_path):
    source = """#version 460
#extension GL_EXT_fragment_shader_barycentric : require
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 color;
void main() {
    color = vec4(gl_BaryCoordEXT + gl_BaryCoordNoPerspEXT, float(gl_ShadingRateEXT));
}
"""
    output = translate(tmp_path, source, ".frag", entry="main", stage="fragment")
    assert "float3 gl_BaryCoordEXT : SV_Barycentrics" in output
    assert "noperspective float3 gl_BaryCoordNoPerspEXT : SV_Barycentrics" in output
    assert "int gl_ShadingRateEXT : SV_ShadingRate" in output


GLSL_INOUT = """#version 460
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer B0 {{ int iin[]; }};
layout(std430, binding = 1) buffer B1 {{ float outp[]; }};
void grow(inout float x, int k) {{ x = x * 2.0 + float(k); }}
void reset(out float x) {{ x = 1.0; }}
float twice(inout float x) {{ x = x * 2.0; return x; }}
void main() {{
    uint i = gl_GlobalInvocationID.x;
{body}
}}
"""


def test_buffer_elements_pass_to_inout_and_out_parameters_by_copy(tmp_path):
    source = GLSL_INOUT.format(
        body="    grow(outp[i], iin[i]);\n    reset(outp[i + 1u]);"
    )
    output = translate(tmp_path, source, ".comp", entry="main", stage="compute")
    assert "float cgl_x = asfloat(b1.Load((i * 4)));" in output
    assert "grow(cgl_x, asint(b0.Load((i * 4))));" in output
    assert "b1.Store((i * 4), asuint(cgl_x));" in output
    # An out parameter is not copied in.
    assert "float cgl_x_1;" in output
    assert "reset(cgl_x_1);" in output


def test_buffer_element_argument_in_a_larger_expression_is_refused(tmp_path):
    source = GLSL_INOUT.format(body="    outp[0] = twice(outp[i]) + 1.0;")
    assert "no l-value" in refusal(tmp_path, source, ".comp")


def test_buffer_element_argument_with_a_side_effecting_index_is_refused(tmp_path):
    source = GLSL_INOUT.format(body="    uint j = i;\n    grow(outp[j++], 1);")
    assert "side effect" in refusal(tmp_path, source, ".comp")


def test_foreign_type_names_are_refused(tmp_path):
    # The CUDA importer keeps the kernel's texture object as the raw type when a helper samples it.
    source = """__device__ float lookup(cudaTextureObject_t t, float x) {
    return tex2DLod<float4>(t, x, 0.5f, 0.0f).x;
}
extern "C" __global__ void probe(float* fin, float* outp, cudaTextureObject_t tex) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = lookup(tex, fin[i]) * 2.0f;
}
"""
    message = refusal(tmp_path, source, ".cu")
    assert "cudaTextureObject_t" in message


CUDA_DOUBLE = """extern "C" __global__ void probe(double* fin, double* outp) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = {call};
}}
"""


@pytest.mark.parametrize("call", ["acos(fin[i])", "pow(fin[i], fin[i + 1])"])
def test_binary64_math_that_slangc_writes_as_glsl_std_450_is_refused(tmp_path, call):
    name = call.split("(", 1)[0]
    message = refusal(tmp_path, CUDA_DOUBLE.format(call=call), ".cu")
    assert f"Slang {name}:" in message and "binary64" in message


def test_binary64_word_access_uses_asuint_and_asdouble(tmp_path):
    source = """shader S {
    compute {
        layout(local_size_x = 1) in;
        buffer double* d;
        buffer uint* w;
        void main() {
            uvec2 words = unpackDouble2x32(d[0]);
            w[0] = words.x;
            d[1] = packDouble2x32(uvec2(words.x, words.y + 1u));
        }
    }
}
"""
    output = translate(tmp_path, source, ".cgl", entry="main", stage="compute")
    assert "asuint(value, low, high);" in output
    assert "return asdouble(words.x, words.y);" in output
    assert "cgl_unpack_double2x32(d[0])" in output


def test_structured_buffer_counter_is_an_explicit_counter_buffer(tmp_path):
    source = """RWStructuredBuffer<uint> outp : register(u1);
RWStructuredBuffer<float> obj : register(u8);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = obj.IncrementCounter();
}
"""
    output = translate(tmp_path, source, ".hlsl", entry="main")
    assert "IncrementCounter" not in output
    assert "RWStructuredBuffer<uint> objCounter" in output
    assert "InterlockedAdd(objCounter[0u], 1u," in output


def test_compute_samples_read_the_base_level(tmp_path):
    source = metal(
        "    outp[i] = tex.sample(smp, float2(fin[i], 1)).x;",
        params=", texture1d_array<float> tex [[texture(0)]], sampler smp [[sampler(0)]]",
    )
    output = translate(tmp_path, source, ".metal", entry="probe")
    assert "tex.SampleLevel(smp, float2(fin.Load(i), 1), 0.0)" in output


def test_resource_member_of_a_buffer_element_is_refused(tmp_path):
    source = metal(
        "    uint k = uint(fin[0]) & 1u;\n"
        "    outp[i] = slots[k].tex.sample(smp, float2(fin[i], 0.5), level(0.0)).x;",
        params=", device const Slot* slots [[buffer(2)]], sampler smp [[sampler(0)]]",
        structs="struct Slot {\n    texture2d<float> tex;\n};",
    )
    assert "Slot.tex" in refusal(tmp_path, source, ".metal")
