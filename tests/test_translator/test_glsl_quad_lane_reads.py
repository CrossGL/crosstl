"""GLSL output reads a quad lane chosen at run time through ``crossglQuadReadLaneAt``.

``subgroupQuadBroadcast`` takes the quad index as an integral constant expression (GL_KHR_shader_subgroup), so a
``QuadReadLaneAt`` whose lane is read from memory used to be written as ``subgroupQuadBroadcast(value, lane)``,
which glslang rejects. Such a read now broadcasts each of the four lanes with a constant index and selects one;
a literal lane keeps the direct call.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import IdentifierNode, LiteralNode
from crosstl.translator.codegen.glsl_quad_lane_reads import (
    quad_lane_read_needs_helper,
)

HLSL_RUNTIME_LANE = """
RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<uint> uin : register(u1);
RWStructuredBuffer<float2> outp : register(u2);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float2 a0 = float2(fin[i + 0u], fin[i + 1u]);
    uint a1 = uin[i + 0u];
    float2 r = QuadReadLaneAt(a0, a1);
    outp[i] = r;
}
"""

CROSSGL_MIXED_LANES = """
shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* lanes;
        buffer uint* uout;
        buffer float* fout;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int lane = lanes[i];
            fout[i] = QuadReadLaneAt(fin[i], lane) + QuadReadLaneAt(fin[i], 2u);
            uout[i] = QuadReadLaneAt(i, lane & 3) + QuadReadLaneAt(i * 2u, lane);
        }
    }
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def _glslang(tmp_path, code):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / "out.comp"
    path.write_text(code, encoding="utf-8")
    for options in ((), ("--target-env", "vulkan1.1", "-o", str(tmp_path / "out.spv"))):
        result = subprocess.run(
            ["glslangValidator", *options, "-S", "comp", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_runtime_lane_reads_through_the_helper(tmp_path):
    glsl = _translate(tmp_path, HLSL_RUNTIME_LANE, "hlsl")

    assert "vec2 crossglQuadReadLaneAt(vec2 value, uint lane) {" in glsl
    for index in range(4):
        assert f"vec2 lane{index} = subgroupQuadBroadcast(value, {index}u);" in glsl
    assert "vec2 r = crossglQuadReadLaneAt(a0, a1);" in glsl
    assert "subgroupQuadBroadcast(a0, a1)" not in glsl
    _glslang(tmp_path, glsl)


def test_crossgl_lanes_declare_one_overload_per_value_type(tmp_path):
    glsl = _translate(tmp_path, CROSSGL_MIXED_LANES, "cgl")

    assert glsl.count("float crossglQuadReadLaneAt(float value, uint lane) {") == 1
    assert glsl.count("uint crossglQuadReadLaneAt(uint value, uint lane) {") == 1
    assert "crossglQuadReadLaneAt(fin[i], uint(lane))" in glsl
    assert "subgroupQuadBroadcast(fin[i], 2u)" in glsl
    assert "crossglQuadReadLaneAt(i, uint((lane & 3)))" in glsl
    # The overloads are declared in the order the program first needs them.
    assert glsl.index("float crossglQuadReadLaneAt(") < glsl.index(
        "uint crossglQuadReadLaneAt("
    )
    _glslang(tmp_path, glsl)


def test_literal_lane_keeps_the_direct_broadcast(tmp_path):
    source = HLSL_RUNTIME_LANE.replace(
        "QuadReadLaneAt(a0, a1)", "QuadReadLaneAt(a0, 3)"
    )
    glsl = _translate(tmp_path, source, "hlsl")

    assert "crossglQuadReadLaneAt" not in glsl
    assert "subgroupQuadBroadcast(a0, 3" in glsl
    _glslang(tmp_path, glsl)


@pytest.mark.parametrize(
    ("operation", "lane", "expected"),
    [
        ("QuadReadLaneAt", IdentifierNode("lane"), True),
        ("QuadReadLaneAt", LiteralNode(1, "int"), False),
        ("QuadReadLaneAt", LiteralNode(2, "uint"), False),
        ("WaveReadLaneAt", IdentifierNode("lane"), False),
    ],
)
def test_helper_is_needed_only_for_a_quad_read_with_a_non_literal_lane(
    operation, lane, expected
):
    assert (
        quad_lane_read_needs_helper(operation, [IdentifierNode("v"), lane]) is expected
    )
