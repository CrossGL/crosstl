"""Saturated conversions: OpenCL C convert_<type>[_sat][_<mode>] reaches every target as its OpenCL meaning.

The importer maps the built-ins to CrossGL casts or to the generic hub intrinsics of the ``convert_sat`` rule family
(crosstl/backend/OpenCL/conversions.py); annotation resolves those by operand type
(crosstl/translator/semantics/conversions.py) and lowering injects the helpers that tests/conformance/rule_table_smt.py
proves against OpenCL C 6.4.3.3.
"""

import re

import pytest

import crosstl
from crosstl.backend.OpenCL.conversions import (
    OpenCLConversionUnsupportedError,
    format_conversion,
)
from crosstl.translator.codegen.registry import backend_names
from crosstl.translator.semantics import RuleInexpressibleError
from crosstl.translator.semantics.helpers import helper_source
from crosstl.translator.semantics.rule_table import (
    CONVERT_SAT,
    CONVERT_SAT_INTRINSICS,
    TARGET_ENTRIES,
)

OPENCL_TYPES = {
    "int": "i32",
    "uint": "u32",
    "uchar": "u8",
    "char": "i8",
    "short": "i16",
    "ushort": "u16",
    "long": "i64",
    "ulong": "u64",
    "float": "f32",
    "uchar4": "vec4<u8>",
}


def crossgl_type(name):
    return OPENCL_TYPES[name]


@pytest.mark.parametrize(
    "call, expected",
    [
        ("convert_int", "i32(x)"),
        ("convert_int_rtz", "i32(x)"),
        ("convert_int4", "ivec4(x)"),
        ("convert_float4", "vec4(x)"),
        ("convert_float_rte", "f32(x)"),
        ("convert_uchar", "u8(x)"),
        ("convert_long", "i64(x)"),
        ("convert_int_sat", "__cgl_convert_int_sat(x)"),
        ("convert_uint2_sat_rtp", "__cgl_convert_uint_sat_rtp(x)"),
        ("convert_int_rte", "__cgl_convert_int_rte(x)"),
        ("convert_uchar_sat", "u8(clamp(__cgl_convert_int_sat(x), 0, 255))"),
        (
            "convert_short_sat_rtn",
            "i16(clamp(__cgl_convert_int_sat_rtn(x), -32768, 32767))",
        ),
        ("convert_uchar_rtn", "u8(__cgl_convert_int_rtn(x))"),
        (
            "convert_uchar4_sat",
            "vec4<u8>(clamp(__cgl_convert_int_sat(x), ivec4(0), ivec4(255)))",
        ),
        ("convert_int_sat_extra", None),
        ("convert_bool", None),
    ],
)
def test_importer_maps_conversion_builtins(call, expected):
    assert format_conversion(call, "x", crossgl_type) == expected


@pytest.mark.parametrize(
    "call, reason",
    [
        ("convert_int8", "8-lane"),
        ("convert_float16_sat", "16-lane"),
        ("convert_float_sat", "allows _sat only for integer"),
        ("convert_float_rtz", "mode _rtz"),
        ("convert_long_sat", "64-bit integer"),
        ("convert_ulong_rte", "64-bit integer"),
    ],
)
def test_importer_refuses_conversions_it_cannot_carry(call, reason):
    with pytest.raises(OpenCLConversionUnsupportedError, match=reason) as raised:
        format_conversion(call, "x", crossgl_type)
    assert raised.value.code == "project.translate.opencl-conversion-unsupported"


def test_convert_sat_has_a_polyfill_entry_for_every_backend():
    for backend in backend_names():
        entry = TARGET_ENTRIES[(CONVERT_SAT, backend)]
        assert entry.kind == "polyfill", backend
        assert entry.reason


def test_vector_helpers_lift_the_scalar_helper_per_lane():
    source = helper_source("__cgl_sat_ftou", 3)
    assert source.startswith("uvec3 cgl_rule_sat_ftou3(vec3 x0)")
    assert (
        "uvec3(cgl_rule_sat_ftou(x0.x), cgl_rule_sat_ftou(x0.y), cgl_rule_sat_ftou(x0.z))"
        in source
    )
    assert helper_source("__cgl_sat_itou", 1).startswith(
        "uint cgl_rule_sat_itou(int x0)"
    )


OPENCL_KERNEL = """__kernel void k(__global const float *f, __global const int *s, __global const uint *u,
                __global int *o, __global uint *uo, __global const float4 *v, __global int4 *vo) {{
    size_t i = get_global_id(0);
{body}
}}
"""


def opencl(tmp_path, body, backend):
    path = tmp_path / "kernel.cl"
    path.write_text(OPENCL_KERNEL.format(body=body))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan", "wgsl"])
def test_saturated_conversions_reach_every_target_as_helpers(tmp_path, backend):
    body = """    o[i] = convert_int_sat(f[i]) + convert_int_sat(u[i]) + convert_int_sat(s[i]);
    uo[i] = convert_uint_sat_rte(f[i]) + convert_uint_sat(s[i]) + convert_uint_sat(u[i]);
    vo[i] = convert_int4_sat(v[i]);"""
    out = opencl(tmp_path, body, backend)
    assert "convert_" not in out
    assert "__cgl_" not in out
    if backend == "vulkan":
        # SPIR-V keeps no helper names: each saturated conversion is a call of an injected function.
        assert out.count("OpFunctionCall") >= 7
        return
    for helper in ("ftoi", "utoi", "ftou", "itou", "ftoi4"):
        assert f"cgl_rule_sat_{helper}" in out, helper


def test_resolution_follows_the_operand_type(tmp_path):
    body = """    o[i] = convert_int_sat(s[i]);
    uo[i] = convert_uint_sat(u[i]) + convert_uint(s[i]) + convert_uint_rte(s[i]);
    o[i] += convert_int_rtp(f[i]) + convert_int_rtn(f[i]);"""
    out = opencl(tmp_path, body, "opengl")
    # Same-signedness integer operands are unchanged; a non-saturated integer conversion keeps the bits.
    assert re.search(r"o\[i\] = s\[i\];", out)
    assert "cgl_rule_sat_itou" not in out and "cgl_rule_sat_utoi" not in out
    assert "uint(s[i])" in out
    # A float operand rounds in the named mode, then saturates.
    assert "cgl_rule_sat_ftoi(ceil(f[i]))" in out
    assert "cgl_rule_sat_ftoi(floor(f[i]))" in out


def test_narrow_saturation_clamps_the_saturated_int(tmp_path):
    out = opencl(tmp_path, "    o[i] = convert_char_sat(f[i]);", "metal")
    assert "clamp(cgl_rule_sat_ftoi(f[i]), -128, 127)" in out


CGL_KERNEL = """shader S {{
    StructuredBuffer<{operand}> a;
    RWStructuredBuffer<int> o;
    compute {{
        @ numthreads(64, 1, 1)
        void main(uvec3 id @ gl_GlobalInvocationID) {{
            o[id.x] = int(__cgl_convert_uint_sat(a[id.x]));
        }}
    }}
}}
"""


def cgl(tmp_path, operand, backend):
    path = tmp_path / "kernel.cgl"
    path.write_text(CGL_KERNEL.format(operand=operand))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    "operand, expected",
    [
        ("float", "int(cgl_rule_sat_ftou(a[gl_GlobalInvocationID.x]))"),
        ("int", "int(cgl_rule_sat_itou(a[gl_GlobalInvocationID.x]))"),
        ("bool", "int(uint(a[gl_GlobalInvocationID.x]))"),
    ],
)
def test_hub_intrinsics_written_in_crossgl_resolve_by_operand(
    tmp_path, operand, expected
):
    assert expected in cgl(tmp_path, operand, "opengl")


@pytest.mark.parametrize("operand", ["double", "int64_t"])
def test_conversions_without_a_helper_are_refused(tmp_path, operand):
    with pytest.raises(RuleInexpressibleError, match="has no conversion from a 64-bit"):
        cgl(tmp_path, operand, "directx")


def test_conversion_intrinsics_are_registered():
    assert set(CONVERT_SAT_INTRINSICS) == {
        "__cgl_sat_ftoi",
        "__cgl_sat_ftou",
        "__cgl_sat_itou",
        "__cgl_sat_utoi",
    }
