"""Metal float-to-integer conversions keep their NaN result on every target (the ``ftoi_nan`` rule family).

MSL 4.1 section 8.6 converts a NaN to 0 and rounds toward zero. crosstl/translator/semantics/nan_conversions.py carries
that rule from Metal sources; Metal discharges it natively, and every other target calls the saturating helpers that
tests/conformance/rule_table_smt.py proves against it. Narrow destinations clamp the helper's ``int`` to their range.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.registry import backend_names
from crosstl.translator.semantics.rule_table import (
    FTOI_NAN,
    SOURCE_RULES,
    TARGET_ENTRIES,
)

METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void convert(device const float* values [[buffer(0)]],
                    device const half* halves [[buffer(1)]],
                    device const uint* counts [[buffer(2)]],
                    device int* ints [[buffer(3)]],
                    device uint* uints [[buffer(4)]],
                    device uchar* bytes [[buffer(5)]],
                    device short* shorts [[buffer(6)]],
                    device int2* pairs [[buffer(7)]],
                    device uchar2* bytePairs [[buffer(8)]],
                    uint i [[thread_position_in_grid]]) {{
{body}
}}
"""


def metal(tmp_path, body, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(METAL_KERNEL.format(body=body))
    return crosstl.translate(str(path), backend=backend, format_output=False)


def compact(text):
    return re.sub(r"\s+", "", text)


def test_metal_defines_the_rule_and_only_metal_discharges_it_natively():
    assert FTOI_NAN in {rule.family for rule in SOURCE_RULES["metal"]}
    for backend in backend_names():
        kind = TARGET_ENTRIES[(FTOI_NAN, backend)].kind
        assert kind == ("native" if backend == "metal" else "polyfill"), backend


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_explicit_conversions_call_the_saturating_helpers(tmp_path, backend):
    body = """  float v = values[i];
  ints[i] = static_cast<int>(v) + int(halves[i]);
  uints[i] = (uint)v;
  pairs[i] = int2(float2(v, -v));"""
    out = compact(metal(tmp_path, body, backend))
    assert "cgl_rule_sat_ftoi(v)" in out
    # A binary16 operand widens to binary32 exactly before the helper.
    widened = "float(halves" if backend == "opengl" else "__crossgl_binary16_to_float("
    assert f"cgl_rule_sat_ftoi({widened}" in out
    assert "cgl_rule_sat_ftou(v)" in out
    assert "cgl_rule_sat_ftoi2(" in out
    # Each helper is defined once.
    assert out.count("intcgl_rule_sat_ftoi(floatx0){") == 1


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_narrow_conversions_clamp_the_saturated_int(tmp_path, backend):
    body = """  float v = values[i];
  bytes[i] = static_cast<uchar>(v);
  shorts[i] = static_cast<short>(v);
  bytePairs[i] = uchar2(float2(v, 2.0f * v));"""
    out = compact(metal(tmp_path, body, backend))
    assert "clamp(cgl_rule_sat_ftoi(v),0,255)" in out
    assert re.search(r"clamp\(cgl_rule_sat_ftoi\(v\),\(?-32768\)?,32767\)", out)
    assert re.search(
        r"clamp\(cgl_rule_sat_ftoi2\(.*?\),(?:ivec2\(0\)|int2\(0,0\))", out
    )


def test_implicit_conversions_of_stored_values_are_carried(tmp_path):
    body = """  float v = values[i];
  int d = v;
  int g;
  g = v * 2.0f;
  ints[i] = d + g;"""
    out = compact(metal(tmp_path, body, "opengl"))
    assert "intd=cgl_rule_sat_ftoi(v);" in out
    assert "g=cgl_rule_sat_ftoi((v*2.0));" in out


def test_integer_operands_carry_no_obligation(tmp_path):
    out = compact(metal(tmp_path, "  ints[i] = int(counts[i]);", "opengl"))
    assert "cgl_rule_sat" not in out


def test_metal_target_keeps_the_conversions_and_converts_narrow_ones_directly(
    tmp_path,
):
    body = """  float v = values[i];
  ints[i] = int(v);
  uints[i] = uint(v);
  bytes[i] = static_cast<uchar>(v);
  shorts[i] = static_cast<short>(halves[i]);
  bytePairs[i] = uchar2(float2(v, v));"""
    out = compact(metal(tmp_path, body, "metal"))
    assert "cgl_rule" not in out and "clamp(" not in out
    assert "ints[i]=int(v);" in out
    assert "uints[i]=uint(v);" in out
    # Through the 32-bit carrier first, uchar(uint(v)) would wrap 300.7 to 44 where uchar(v) saturates.
    assert "uchar(v)" in out and "uchar(uint(v))" not in out
    assert "short(halves[i])" in out
    assert "uchar2(float2(v,v))" in out


def test_spirv_clamps_the_helper_result_for_narrow_destinations(tmp_path):
    body = """  float v = values[i];
  bytes[i] = static_cast<uchar>(v);
  shorts[i] = static_cast<short>(v);"""
    out = metal(tmp_path, body, "vulkan")
    assert out.count("OpFunctionCall") == 2
    assert out.count("SClamp") == 2
    # The only masks are the helper's own magnitude tests: the narrow values are not reduced modulo 2^N.
    helper_only = metal(tmp_path, "  ints[i] = int(values[i]);", "vulkan")
    assert out.count("OpBitwiseAnd") == helper_only.count("OpBitwiseAnd")


def test_glsl_sources_carry_no_obligation(tmp_path):
    path = tmp_path / "kernel.comp"
    path.write_text("""#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Inp { float a[]; };
layout(std430, binding = 1) buffer Outp { int o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    o[i] = int(a[i]);
}
""")
    out = crosstl.translate(str(path), backend="directx", format_output=False)
    assert "cgl_rule_sat" not in out
    assert "asuint(int(asfloat(" in compact(out)


CGL_NARROW = """shader S {
    StructuredBuffer<float> a;
    RWStructuredBuffer<uint8> o;
    compute {
        @ numthreads(64, 1, 1)
        void main(uvec3 id @ gl_GlobalInvocationID) {
            o[id.x] = uint8(a[id.x]);
        }
    }
}
"""


@pytest.mark.parametrize("backend", ["vulkan", "wgsl"])
def test_narrow_legalization_saturates_floating_operands(tmp_path, backend):
    path = tmp_path / "kernel.cgl"
    path.write_text(CGL_NARROW)
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    if backend == "vulkan":
        assert "SClamp" in out and "OpBitwiseAnd" not in out
    else:
        assert "clamp(i32(" in compact(out) or "clamp(" in out
        assert "& 255" not in out


def test_crossgl_intrinsics_of_both_families_share_one_helper(tmp_path):
    path = tmp_path / "kernel.cgl"
    path.write_text("""shader S {
    StructuredBuffer<float> a;
    RWStructuredBuffer<int> o;
    compute {
        @ numthreads(64, 1, 1)
        void main(uvec3 id @ gl_GlobalInvocationID) {
            o[id.x] = __cgl_sat_ftoi(a[id.x]) + __cgl_nan0_ftoi(a[id.x]);
        }
    }
}
""")
    out = compact(crosstl.translate(str(path), backend="opengl", format_output=False))
    assert out.count("intcgl_rule_sat_ftoi(floatx0){") == 1
    assert "__cgl_" not in out
