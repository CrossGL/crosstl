"""Constant data larger than a Direct3D constant buffer is refused for HLSL instead of emitting an invalid cbuffer."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics import ConstantBufferTooLargeError
from crosstl.translator.semantics.constant_buffer_size import hlsl_constant_buffer_size

METAL = """#include <metal_stdlib>
using namespace metal;
struct Item {{ float4x4 transform; float4 color; }};
struct Frame {{ float4x4 view; Item items[{count}]; uint count; }};
struct VOut {{ float4 position [[position]]; float4 color; }};
vertex VOut v(constant Frame& frame [[buffer(1)]], uint vid [[vertex_id]], uint iid [[instance_id]]) {{
    VOut out;
    out.position = frame.view * frame.items[iid].transform * float4(float(vid), 0.0, 0.0, 1.0);
    out.color = frame.items[iid].color;
    return out;
}}
"""


def translate(tmp_path, source, backend, suffix=".metal"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def test_metal_constant_data_over_64_kib_is_refused_for_hlsl(tmp_path):
    # 1000 items of 80 bytes: DXC rejected the cbuffer ("CBuffer size is 80068 bytes, exceeding maximum of 65536").
    source = METAL.format(count=1000)
    with pytest.raises(ConstantBufferTooLargeError) as raised:
        translate(tmp_path, source, "directx")
    assert raised.value.buffer_name == "frame" and raised.value.size >= 80068
    assert raised.value.code == "project.translate.directx-constant-buffer-too-large"
    for backend in ("opengl", "vulkan", "metal"):
        translate(tmp_path, source, backend)


def test_constant_data_under_the_limit_stays_a_constant_buffer(tmp_path):
    hlsl = translate(tmp_path, METAL.format(count=600), "directx")
    assert "ConstantBuffer<Frame> frame : register(b1);" in hlsl
    if shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(hlsl)
        result = subprocess.run(
            ["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_large_glsl_uniform_blocks_are_refused_for_hlsl(tmp_path):
    source = (
        "#version 450\n"
        "layout(binding = 0) uniform Table { vec4 rows[5000]; } table;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = table.rows[int(gl_FragCoord.x)]; }\n"
    )
    with pytest.raises(ConstantBufferTooLargeError):
        translate(tmp_path, source, "directx", ".frag")


@pytest.mark.parametrize(
    "members, size",
    [
        ([("float4x4", None)], 64),
        ([("float3", None), ("float", None)], 16),
        ([("float", 3)], 48),
        ([("float", None), ("float4", None)], 32),
    ],
)
def test_the_size_bound_follows_the_packing_rules(members, size):
    def member_type(name, count):
        element = {
            "float4x4": A.MatrixType(A.PrimitiveType("float"), 4, 4),
            "float4": A.VectorType(A.PrimitiveType("float"), 4),
            "float3": A.VectorType(A.PrimitiveType("float"), 3),
            "float": A.PrimitiveType("float"),
        }[name]
        if count is None:
            return element
        return A.ArrayType(element, A.LiteralNode(count, A.PrimitiveType("int")))

    struct = A.StructNode(
        "S",
        [A.StructMemberNode(f"m{i}", member_type(*m)) for i, m in enumerate(members)],
    )
    assert hlsl_constant_buffer_size(A.NamedType("S"), {"S": struct}) == size
