"""Vector views that gather consecutive narrow storage elements as lanes."""

import pytest

import crosstl
from crosstl.translator.codegen.pointer_reinterpret import (
    PointerReinterpretationError,
    same_scalar_lane_view,
    scalar_storage_layout,
)
from tests.test_translator.test_metal_builtin_ownership import _compile

HEADER = """#include <metal_stdlib>
using namespace metal;
"""


def _gather_source(scalar, view, conversion="float4"):
    """MLX's dot-product read: a vector view of four consecutive elements."""
    return HEADER + f"""kernel void gather(const device {scalar}* values [[buffer(0)]],
                   device float* results [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {{
    float4 lanes = {conversion}(*reinterpret_cast<const device {view}*>(
        values + tid * 4u));
    results[tid] = lanes.x + lanes.y * 2.0f + lanes.z * 4.0f + lanes.w * 8.0f;
}}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "gather.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=target)


@pytest.mark.parametrize(
    "spelling,component,lanes",
    [
        ("f16vec4", "float16", 4),
        ("bfloat16vec2", "bfloat16", 2),
        ("half3", "half", 3),
        ("i8vec4", "int8", 4),
        ("u16vec2", "uint16", 2),
    ],
)
def test_lane_view_spellings_resolve_to_the_source_scalar(spelling, component, lanes):
    layout = scalar_storage_layout(component)
    assert same_scalar_lane_view(spelling, layout) == (
        lanes,
        {"half": "half"}.get(component, component),
    )


@pytest.mark.parametrize(
    "spelling,source",
    [
        ("f16vec4", "bfloat16"),
        ("vec4", "float16"),
        ("ivec4", "int8"),
        ("half8", "half"),
        ("f16vec1", "half"),
    ],
)
def test_lane_views_require_the_same_scalar(spelling, source):
    assert same_scalar_lane_view(spelling, scalar_storage_layout(source)) is None


@pytest.mark.parametrize(
    "scalar,view",
    [
        ("half", "metal::vec<half, 4>"),
        ("half", "half4"),
        ("bfloat16_t", "metal::vec<bfloat16_t, 4>"),
        ("bfloat", "bfloat4"),
    ],
)
def test_opengl_gathers_widened_narrow_lanes(tmp_path, scalar, view):
    generated = _translate(tmp_path, _gather_source(scalar, view), "opengl")
    # Widened storage holds one logical value per array element; the view must
    # not reinterpret the bits of those binary32 words.
    assert "readonly buffer valuesBuffer { float values[]; };" in generated
    assert "floatBitsToUint(values" not in generated
    assert "uintBitsToFloat" not in generated
    for lane in range(1, 4):
        assert f"+ {lane * 2})) / 2)]" in generated
    _compile(generated, "opengl", tmp_path)


def test_opengl_gathers_byte_lanes(tmp_path):
    source = HEADER + """kernel void gather(const device char* values [[buffer(0)]],
                   device int* results [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {
    int4 lanes = int4(*reinterpret_cast<const device char4*>(values + tid * 4u));
    results[tid] = lanes.x + lanes.y + lanes.z + lanes.w;
}
"""
    generated = _translate(tmp_path, source, "opengl")
    assert "bitfieldExtract" not in generated
    assert "ivec4(values[int(" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("target", ["opengl", "directx"])
def test_lane_gather_views_reject_writes(tmp_path, target):
    source = HEADER + """kernel void scatter(device half* values [[buffer(0)]],
                    uint tid [[thread_position_in_grid]]) {
    *reinterpret_cast<device half4*>(values + tid * 4u) = half4(1.0h);
}
"""
    with pytest.raises(PointerReinterpretationError, match="writes through"):
        _translate(tmp_path, source, target)


def test_directx_gathers_binary16_lanes(tmp_path):
    generated = _translate(
        tmp_path, _gather_source("half", "metal::vec<half, 4>"), "directx"
    )
    assert "StructuredBuffer<uint16_t> values" in generated
    assert generated.count("asfloat16(values[uint(") == 4
    _compile(generated, "directx", tmp_path)


def test_directx_decodes_bfloat16_lanes_exactly(tmp_path):
    generated = _translate(
        tmp_path, _gather_source("bfloat16_t", "metal::vec<bfloat16_t, 4>"), "directx"
    )
    assert "float4 __crossgl_bfloat16_to_float(uint4 value)" in generated
    assert "__crossgl_bfloat16_to_float(uint4(uint4(values[uint(" in generated
    # A numeric uint4-to-float4 conversion would read the payload as integers.
    assert "__crossgl_uint_to_float4" not in generated
    _compile(generated, "directx", tmp_path)


def test_directx_rejects_unconverted_bfloat16_lane_views(tmp_path):
    source = (
        HEADER + """kernel void gather(const device bfloat16_t* values [[buffer(0)]],
                   device float* results [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {
    vec<bfloat16_t, 4> lanes = *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
        values + tid * 4u);
    results[tid] = float(lanes.x);
}
"""
    )
    with pytest.raises(PointerReinterpretationError) as raised:
        _translate(tmp_path, source, "directx")
    assert raised.value.reason == "bfloat16-vector-view-requires-float-conversion"


def test_metal_keeps_unnamed_vector_pointee_views(tmp_path):
    generated = _translate(
        tmp_path, _gather_source("float", "metal::vec<float, 4>"), "metal"
    )
    assert "reinterpret_cast<const device float4 *>(values + tid * 4u)" in generated
    _compile(generated, "metal", tmp_path)
