"""Writes to components of a buffer-block vector store each component into the HLSL byte-address buffer."""

import re
import shutil
import subprocess

import crosstl

COMPONENT_WRITES = """#version 450
layout(local_size_x = 64) in;
struct Particle { vec4 pos; vec2 vel; vec2 pad; };
layout(std430, binding = 1) buffer Particles { Particle particles[]; };
layout(std430, binding = 2) buffer Single { vec4 color; uvec2 counts; } single;
void main() {
    uint index = gl_GlobalInvocationID.x;
    int k = int(index % 4u);
    particles[index].pos.x += 0.5;
    particles[index].pos.zw = vec2(1.0, 2.0);
    particles[index].vel.y = 2.0;
    particles[index].pos[k] = 3.0;
    single.color.a = 1.0;
    single.counts.yx = uvec2(index, 1u);
}
"""


def _hlsl(tmp_path):
    path = tmp_path / "shader.comp"
    path.write_text(COMPONENT_WRITES)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def test_component_writes_store_each_component(tmp_path):
    hlsl = _hlsl(tmp_path)

    buffer = re.search(r"RWByteAddressBuffer (\w+) : register\(u1\)", hlsl)
    assert buffer, hlsl
    buf = buffer.group(1)
    # Nothing assigns to a load.
    assert not re.search(r"Load\d?\([^;]*\)\)?(\.\w+|\[\w+\])\s*[-+*/]?=", hlsl), hlsl
    # particles[index].pos.x += 0.5: component 0 of the vector at index * 32.
    assert (
        f"{buf}.Store((index * 32), asuint((asfloat({buf}.Load((index * 32))) + (0.5))));"
        in hlsl
    ), hlsl
    # .zw = vec2(...): one temporary, components 2 and 3.
    temp = re.search(r"float2 (\w+) = float2\(1\.0, 2\.0\);", hlsl)
    assert temp, hlsl
    assert f"{buf}.Store((index * 32 + 8), asuint({temp.group(1)}.x));" in hlsl
    assert f"{buf}.Store((index * 32 + 12), asuint({temp.group(1)}.y));" in hlsl
    # vel.y: the vec2 at offset 16, component 1.
    assert f"{buf}.Store((index * 32 + 16 + 4), asuint((2.0)));" in hlsl
    # A run-time component index addresses 4 bytes per component.
    assert f"{buf}.Store(((index * 32) + k * 4), asuint((3.0)));" in hlsl
    # A block that is no array: color.a at 12, and counts.yx stores y at 20 and x at 16.
    assert "single.Store(12, asuint((1.0)));" in hlsl
    swizzle = re.search(r"uint2 (\w+) = uint2\(index, 1u\);", hlsl)
    assert swizzle, hlsl
    assert f"single.Store(20, {swizzle.group(1)}.x);" in hlsl
    assert f"single.Store(16, {swizzle.group(1)}.y);" in hlsl


def test_component_writes_compile_with_dxc(tmp_path):
    if shutil.which("dxc") is None:
        return
    source = tmp_path / "out.hlsl"
    source.write_text(_hlsl(tmp_path))
    result = subprocess.run(
        ["dxc", "-T", "cs_6_0", "-E", "CSMain", str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
