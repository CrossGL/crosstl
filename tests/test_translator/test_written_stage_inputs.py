"""Entry-point inputs that the body writes reach GLSL as local copies, since GLSL inputs are read-only."""

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """
RWStructuredBuffer<uint> outBuf : register(u0);
[numthreads(4, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint li : SV_GroupIndex, uint3 tid : SV_GroupThreadID) {
    gid.x = gid.x * 2u;
    li += 1u;
    outBuf[gid.x + tid.y] = li;
}
"""

FRAGMENT = """
struct PSIn { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
float4 main(PSIn i, float3 n : NORMAL, bool ff : SV_IsFrontFace, float w : TEXCOORD1) : SV_Target {
    i.uv = i.uv * 2.0;
    n = normalize(n);
    i.pos.x += 1.0;
    ff = !ff;
    return float4(i.uv, n.x + w, ff ? i.pos.x : 0.0);
}
"""

VERTEX = """
struct VSIn { float3 p : POSITION; float2 uv : TEXCOORD0; };
float4 main(VSIn v, float3 extra : NORMAL, uint vid : SV_VertexID) : SV_Position {
    v.p.x += 1.0;
    extra *= 2.0;
    ++vid;
    return float4(v.p + extra, float(vid));
}
"""

INOUT_AND_SHADOWING = """
void bump(inout float3 value) { value += 1.0; }
float4 main(float3 n : NORMAL, float3 m : TEXCOORD0) : SV_Target {
    bump(n);
    float3 n_copy = m;
    {
        float3 n = m * 3.0;
        n *= 2.0;
        n_copy += n;
    }
    return float4(n + n_copy, 1.0);
}
"""

METAL = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float2 uv; };
kernel void scale(device float* data [[buffer(0)]], constant uint& n [[buffer(1)]],
                  uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    for (; gid < n; gid += grid) {
        data[gid] = data[gid] * 2.0;
    }
}
"""

METAL_FRAGMENT = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float2 uv; };
fragment float4 fs(VOut in [[stage_in]]) {
    in.uv = fract(in.uv * 4.0);
    return float4(in.uv, 0.0, 1.0);
}
"""

CROSSGL = """
shader Written {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            input.uv = input.uv * 0.5;
            output.color = vec4(input.uv, 0.0, 1.0);
            return output;
        }
    }
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _body(code):
    return code[code.index("void main()") :]


def _assigned_names(body):
    """Root names of assignment and increment targets in GLSL ``body``."""
    names = set(re.findall(r"(?m)^\s*(\w+)(?:\.\w+|\[[^]]*\])*\s*[-+*/]?=[^=]", body))
    names.update(re.findall(r"(?:\+\+|--)\s*(\w+)", body))
    names.update(re.findall(r"(\w+)(?:\.\w+)*\s*(?:\+\+|--)", body))
    return names


def _glslang(tmp_path, code, stage):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    path = tmp_path / f"shader.{stage}"
    path.write_text(code)
    result = subprocess.run(
        [validator, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_written_compute_builtins_become_locals(tmp_path):
    code = _translate(tmp_path, COMPUTE, ".hlsl", "opengl")
    body = _body(code)
    assert "uvec3 gid_copy = gl_GlobalInvocationID;" in body
    assert "uint li_copy = gl_LocalInvocationIndex;" in body
    assert "gid_copy.x = (gid_copy.x * 2u);" in body
    assert "li_copy += 1u;" in body
    assert "outBuf[(gid_copy.x + gl_LocalInvocationID.y)] = li_copy;" in body
    # An input the body only reads keeps its direct spelling.
    assert "tid_copy" not in body
    assert not any(name.startswith("gl_") for name in _assigned_names(body))
    _glslang(tmp_path, code, "comp")


def test_written_fragment_inputs_keep_their_interface_names(tmp_path):
    code = _translate(tmp_path, FRAGMENT, ".hlsl", "opengl")
    body = _body(code)
    # The interface variables keep the names the vertex stage links by.
    assert re.search(r"\bin vec2 uv;", code)
    assert re.search(r"\bin vec3 n;", code)
    assert re.search(r"\bin float w;", code)
    assert "vec3 n_copy = n;" in body
    assert "bool ff_copy = gl_FrontFacing;" in body
    assert "PSIn i_copy = " in body
    assert "i_copy.uv = (i_copy.uv * 2.0);" in body
    assert "i_copy.pos.x += 1.0;" in body
    assert "n_copy = normalize(n_copy);" in body
    assert "w_copy" not in body
    assert _assigned_names(body) & {"uv", "n", "gl_FragCoord", "gl_FrontFacing"} == (
        set()
    )
    _glslang(tmp_path, code, "frag")


def test_written_vertex_inputs_become_locals(tmp_path):
    code = _translate(tmp_path, VERTEX, ".hlsl", "opengl")
    body = _body(code)
    assert re.search(r"\bin vec3 p;", code)
    assert re.search(r"\bin vec3 extra;", code)
    assert "VSIn v_copy = " in body
    assert "v_copy.p.x += 1.0;" in body
    assert "extra_copy *= 2.0;" in body
    assert "vid_copy" in body
    assert _assigned_names(body) & {"p", "extra", "gl_VertexID"} == set()
    _glslang(tmp_path, code, "vert")


def test_inout_arguments_are_writes_and_shadowing_locals_are_kept(tmp_path):
    code = _translate(tmp_path, INOUT_AND_SHADOWING, ".hlsl", "opengl")
    body = _body(code)
    # ``n_copy`` is taken by a local, so the copy takes the next free name.
    assert "vec3 n_copy_1 = n;" in body
    assert "bump(n_copy_1);" in body
    assert "vec3 n_copy = m;" in body
    # The inner block's ``n`` is its own local and keeps its name.
    assert "vec3 n = (m * 3.0);" in body
    assert "n *= 2.0;" in body
    assert "n_copy += n;" in body
    assert "fragColor = vec4((n_copy_1 + n_copy), 1.0);" in body
    assert "m_copy" not in body
    _glslang(tmp_path, code, "frag")


def test_metal_grid_stride_kernel_increments_a_copy(tmp_path):
    code = _translate(tmp_path, METAL, ".metal", "opengl")
    body = _body(code)
    assert "uint gid_copy = gid;" in body
    assert "for (; (gid_copy < n); gid_copy += grid) {" in body
    assert "data[gid_copy] = (data[gid_copy] * 2.0);" in body
    _glslang(tmp_path, code, "comp")


@pytest.mark.parametrize("backend", ["opengl", "webgl"])
def test_metal_stage_in_struct_written_through_a_copy(tmp_path, backend):
    code = _translate(tmp_path, METAL_FRAGMENT, ".metal", backend)
    body = _body(code)
    assert "VOut in_copy = in_value;" in body
    assert "in_copy.uv = fract((in_copy.uv * 4.0));" in body
    assert "uv" not in _assigned_names(body)
    _glslang(tmp_path, code, "frag")


def test_crossgl_fragment_input_struct(tmp_path):
    code = _translate(tmp_path, CROSSGL, ".cgl", "opengl")
    body = _body(code)
    assert "input_copy" in body
    assert "uv" not in _assigned_names(body)
    _glslang(tmp_path, code, "frag")


@pytest.mark.parametrize("backend", ["directx", "metal", "vulkan"])
def test_targets_with_mutable_inputs_are_unchanged(tmp_path, backend):
    code = _translate(tmp_path, FRAGMENT, ".hlsl", backend)
    assert "_copy" not in code
