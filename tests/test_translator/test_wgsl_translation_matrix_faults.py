"""WGSL output of the HLSL, Metal, GLSL, CUDA and SPIR-V constructs that the translation matrix found invalid.

Each test translates a native source straight to WGSL, as a translation-matrix cell does, and checks the WGSL
with naga when it is installed.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_buffer_atomics import WGSLAtomicLoweringError
from crosstl.translator.codegen.wgsl_foreign_types import WGSLLoweringRefused
from crosstl.translator.codegen.wgsl_non_finite_constants import (
    is_non_finite_f32_bits,
)
from crosstl.translator.refusals import WGSLRefusal


def translate(tmp_path, source, suffix):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


HLSL_COMPUTE = """RWStructuredBuffer<int> iin : register(u0);
RWStructuredBuffer<{out}> outp : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    uint i = id.x;
{body}
}}
"""


@pytest.mark.parametrize(
    "out, body, expected",
    [
        (
            "uint2",
            "int2 a0 = int2(iin[i], iin[i + 1u]);\n uint2 r = countbits(a0);\n outp[i] = r;",
            "var r: vec2<u32> = vec2<u32>(countOneBits(a0));",
        ),
        (
            "uint2",
            "int2 a0 = int2(iin[i], iin[i + 1u]);\n uint2 r = firstbithigh(a0);\n outp[i] = r;",
            "var r: vec2<u32> = vec2<u32>(firstLeadingBit(a0));",
        ),
        (
            "int2",
            "float2 a0 = float2(iin[i], iin[i + 1u]);\n int2 r = sign(a0);\n outp[i] = r;",
            "var r: vec2<i32> = vec2<i32>(sign(a0));",
        ),
        (
            "int",
            "float a0 = float(iin[i]);\n int r = sign(a0);\n outp[i] = r;",
            "var r: i32 = i32(sign(a0));",
        ),
    ],
)
def test_hlsl_implicit_vector_conversion_of_builtin_results(
    tmp_path, out, body, expected
):
    """``countbits`` of an ``int2`` is ``vec2<i32>`` in WGSL and ``sign`` keeps its float type."""
    output = translate(tmp_path, HLSL_COMPUTE.format(out=out, body=body), ".hlsl")
    assert expected in output


def test_hlsl_bit_cast_of_float3_named_vector(tmp_path):
    """The SPIR-V importer and HLSL name vectors ``float3``; the bit cast keeps the vector width."""
    source = """RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<int3> outp : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float3 a0 = float3(fin[i], fin[i + 1u], fin[i + 2u]);
    int3 r = asint(a0);
    outp[i] = r;
}
"""
    output = translate(tmp_path, source, ".hlsl")
    assert "bitcast<vec3<i32>>(a0)" in output
    assert "bitcast<i32>(a0)" not in output


def test_texture_1d_sample_grad_selects_the_gradient_level(tmp_path):
    """WGSL has no 1D ``textureSampleGrad``; the level is log2 of the larger scaled gradient."""
    source = """RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<float4> outp : register(u1);
Texture1D<float4> obj : register(t0);
SamplerState smp : register(s0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    outp[i] = obj.SampleGrad(smp, fin[i], fin[i + 1u], fin[i + 2u]);
}
"""
    output = translate(tmp_path, source, ".hlsl")
    assert "textureSampleGrad" not in output
    assert (
        "textureSampleLevel(obj, smp, fin[i], log2(max(max(abs(fin[(i + 1u)]), "
        "abs(fin[(i + 2u)])) * f32(textureDimensions(obj)), 1.17549435e-38)))"
    ) in output


def test_texture_1d_sample_grad_offset_is_refused(tmp_path):
    source = """shader Probe {
    sampler1D tex;
    buffer vec4 values[];
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            values[i] = textureGradOffset(tex, 0.5, 0.1, 0.1, 1);
        }
    }
}
"""
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    with pytest.raises(WGSLRefusal, match="offset on a 1D texture"):
        crosstl.translate(str(path), backend="wgsl")


def test_non_finite_bit_constant_is_made_at_run_time(tmp_path):
    """``bitcast<f32>(0x7fc00000u)`` is a const-expression NaN, which Tint rejects."""
    source = """RWStructuredBuffer<float> outp : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float n = asfloat(0x7fc00000u);
    float inf = asfloat(0x7f800000u);
    float one = asfloat(0x3f800000u);
    outp[id.x] = n + inf + one;
}
"""
    output = translate(tmp_path, source, ".hlsl")
    assert "fn cgl_f32_from_bits(bits: u32) -> f32" in output
    assert "var n: f32 = cgl_f32_from_bits(bitcast<u32>(2143289344u));" in output
    assert "var inf: f32 = cgl_f32_from_bits(bitcast<u32>(2139095040u));" in output
    assert "var one: f32 = bitcast<f32>(1065353216u);" in output


@pytest.mark.parametrize(
    "value, expected",
    [
        (0x7FC00000, True),
        (0x7F800000, True),
        (0xFF800000, True),
        (0xFFFFFFFF, True),
        (-1, True),
        (0x3F800000, False),
        (0x7F7FFFFF, False),
        (0, False),
        (None, False),
        (True, False),
    ],
)
def test_non_finite_f32_bit_patterns(value, expected):
    assert is_non_finite_f32_bits(value) is expected


def test_metal_empty_brace_initializers_are_zero_values(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
struct Pair { float a; float b; };
kernel void probe(device int* iin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    Pair p = {};
    float4 v = {};
    float a[3] = {};
    a[iin[i] % 3] = 2.0;
    outp[i] = p.a + v.x + a[0] + a[1] + a[2];
}
"""
    output = translate(tmp_path, source, ".metal")
    assert "var a: array<f32, 3> = array<f32, 3>(f32(), f32(), f32());" in output
    assert "array()" not in output


def test_metal_entry_textures_and_samplers_are_module_bindings(tmp_path):
    """MSL entry parameters ``[[texture(0)]]`` and ``[[sampler(0)]]`` become module variables with bindings."""
    source = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device float* fin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]], texture3d<float> tex [[texture(0)]],
                  sampler smp [[sampler(0)]]) {
    uint i = id.x;
    outp[i] = tex.sample(smp, float3(fin[i], 0.5, 0.5), level(0.0)).x;
}
"""
    output = translate(tmp_path, source, ".metal")
    signature = next(line for line in output.splitlines() if "fn compute_main(" in line)
    assert "texture_3d" not in signature and "sampler" not in signature
    assert "var tex: texture_3d<f32>;" in output
    assert "var smp: sampler;" in output
    bindings = [
        line for line in output.splitlines() if line.startswith("@group(0) @binding(")
    ]
    assert len(bindings) == len(set(bindings)) == 5


def test_metal_fragment_texture_parameters(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
fragment float4 fs(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                   sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv);
}
"""
    output = translate(tmp_path, source, ".metal")
    assert "fn fragment_main(in_: VOut) -> @location(0) vec4<f32>" in output
    assert "var tex: texture_2d<f32>;" in output


def test_metal_vertex_returning_float4_returns_the_position(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
struct VertexIn { float4 position [[attribute(0)]]; };
vertex float4 probe(VertexIn in [[stage_in]], uint vid [[vertex_id]]) {
    return in.position * 2.0 + float4(float(vid), 0.0, 0.0, 0.0);
}
"""
    output = translate(tmp_path, source, ".metal")
    assert "-> @builtin(position) vec4<f32>" in output


def test_metal_bool_bitwise_results_convert_back_to_bool(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device int* iin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    bool p = iin[i] > 3;
    bool q = (iin[i] & 1) != 0;
    bool a = p & q;
    bool o = p | q;
    outp[i] = float(a) + float(o) * 2.0;
}
"""
    output = translate(tmp_path, source, ".metal")
    assert "var a: bool = bool((i32(p) & i32(q)));" in output
    assert "var o: bool = bool((i32(p) | i32(q)));" in output


def test_metal_weak_compare_exchange_writes_the_found_value(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void probe(device int* iin [[buffer(0)]], device atomic_int* outp [[buffer(1)]],
                  device int* ok [[buffer(2)]], uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    int expected = iin[i];
    bool exchanged = atomic_compare_exchange_weak_explicit(&outp[i], &expected, iin[i + 1u],
        memory_order_relaxed, memory_order_relaxed);
    ok[i] = exchanged ? expected : -expected;
}
"""
    output = translate(tmp_path, source, ".metal")
    assert "var<storage, read_write> outp: array<atomic<i32>>;" in output
    assert (
        "let _crosstl_cas_0 = atomicCompareExchangeWeak(_crosstl_cas_0_ptr, "
        "_crosstl_cas_0_compare, _crosstl_cas_0_value);"
    ) in output
    assert "expected = _crosstl_cas_0.old_value;" in output
    assert "var exchanged: bool = _crosstl_cas_0_exchanged;" in output
    assert "loop {" not in output


def test_weak_compare_exchange_in_a_conditional_operand_is_refused(tmp_path):
    source = """shader Probe {
    buffer int values[];
    buffer int flags[];
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int e = 0;
            flags[i] = (flags[i] > 0 && atomicCompareExchangeWeak(values[i], e, 1)) ? 1 : 0;
        }
    }
}
"""
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    with pytest.raises(WGSLAtomicLoweringError, match="conditionally evaluated"):
        crosstl.translate(str(path), backend="wgsl")


@pytest.mark.parametrize(
    "stage, declaration, read, builtin, wgsl_read",
    [
        (
            "frag",
            "",
            "gl_FrontFacing ? 1.0 : 0.0",
            "@builtin(front_facing) front_facing: bool",
            "front_facing",
        ),
        (
            "frag",
            "",
            "float(gl_SampleID)",
            "@builtin(sample_index) sample_index: u32",
            "f32(i32(sample_index))",
        ),
        (
            "frag",
            "",
            "float(gl_PrimitiveID)",
            "@builtin(primitive_index) primitive_index: u32",
            "f32(i32(primitive_index))",
        ),
        (
            "vert",
            "",
            "float(gl_VertexID)",
            "@builtin(vertex_index) vertex_index: u32",
            "f32(i32(vertex_index))",
        ),
        (
            "vert",
            "",
            "float(gl_InstanceID)",
            "@builtin(instance_index) instance_index: u32",
            "f32(i32(instance_index))",
        ),
    ],
)
def test_glsl_builtin_inputs_read_by_name(
    tmp_path, stage, declaration, read, builtin, wgsl_read
):
    if stage == "frag":
        source = f"""#version 460
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {{
    color = vec4(uv, {read}, 1.0);
}}
"""
    else:
        source = f"""#version 460
layout(location = 0) in vec4 position;
void main() {{
    gl_Position = position + vec4({read}, 0.0, 0.0, 0.0);
}}
"""
    output = translate(tmp_path, source, f".{stage}")
    assert builtin in output
    assert wgsl_read in output
    if "primitive_index" in builtin:
        assert "enable primitive_index;" in output


FOREIGN_TYPE_PARAMETER = """shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* o;
        float pick(cudaTextureObject_t t, float x) { return x; }
        void main() {
            cudaTextureObject_t t;
            o[0] = pick(t, 1.0);
        }
    }
}
"""


def test_foreign_builtin_type_left_in_the_output_is_refused(tmp_path):
    # A CUDA built-in type name carried through the IR reaches WGSL output as a type of that name, which WGSL lacks.
    with pytest.raises(
        WGSLLoweringRefused,
        match="the built-in type 'cudaTextureObject_t' has no WGSL spelling",
    ):
        translate(tmp_path, FOREIGN_TYPE_PARAMETER, ".cgl")
