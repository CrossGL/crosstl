"""Metal C compound literals ``(T){...}``.

The Metal compiler accepts ``(float4x4){c0, c1, c2, c3}`` as a temporary initialized from the braced list, as the
declaration ``float4x4 m = {c0, c1, c2, c3};`` would be. The importer used to emit ``mat4({...})``, a constructor
call with an initializer-list argument that no target can lower.
"""

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

SOURCE = """
struct Pair { float a; int b; };
kernel void k(device float4x4* matrices [[buffer(0)]],
              device Pair* pairs [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    float3 f = float3(1.0f, 2.0f, 3.0f);
    float4x4 m = (float4x4) { float4(f, 0), float4(f * 2.0f, 0), float4(f * 3.0f, 0), float4(f, 1) };
    matrices[i] = m;
    pairs[i] = (Pair){ 1.5f, 2 };
    float2 v = (float2){ f.x, f.y };
    pairs[i].a += v.x;
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + SOURCE)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_compound_literals_become_braced_initializations(tmp_path):
    cgl = _translate(tmp_path, "cgl")
    assert (
        "mat4 m = mat4(vec4(f, 0), vec4(f * 2.0f, 0), vec4(f * 3.0f, 0), vec4(f, 1));"
        in cgl
    )
    assert "vec2 v = vec2{(f.x), (f.y),};" in cgl
    assert "mat4({" not in cgl
    assert "Pair({" not in cgl


def test_compound_literals_reach_directx_and_opengl(tmp_path):
    hlsl = _translate(tmp_path, "directx")
    assert (
        "float4x4 m = transpose(float4x4(float4(f, 0), float4((f * 2.0), 0), "
        "float4((f * 3.0), 0), float4(f, 1)));"
    ) in hlsl
    assert "pairs[i] = __crossgl_construct_Pair(1.5, 2);" in hlsl
    assert "float2 v = float2(f.x, f.y);" in hlsl

    glsl = _translate(tmp_path, "opengl")
    assert "pairs[i] = Pair(1.5, 2);" in glsl
    assert "vec2 v = vec2(f.x, f.y);" in glsl
