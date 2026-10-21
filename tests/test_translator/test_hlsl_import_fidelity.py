"""HLSL constructs the importer and the emitters keep: scopes, buffer views, interface overlaps, tessellation factors.

* A nested compound statement ``{ ... }`` opens a scope (HLSL follows C), so two sibling blocks may declare the
  same name.
* ``ConstantBuffer<T>`` names ``T``'s members directly, so a member's matrix type is known to the importer.
* ``RaytracingAccelerationStructure`` is the CrossGL ``accelerationStructureEXT``.
* Stage interface members whose semantics overlap are refused with a typed diagnostic.
* DXIL requires ``SV_TessFactor``/``SV_InsideTessFactor`` to hold exactly the domain's factors, so longer factor
  arrays that are only indexed by constants below that count are narrowed for HLSL.
* An emitted GLSL identifier never spells a GLSL type keyword (``imageBuffer`` for a buffer named ``image``).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.interface_conflicts import (
    InterfaceLocationConflictError,
)
from crosstl.translator.semantics import TessFactorUnsupportedError


def _translate(tmp_path, source, name, backend, **options):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, **options)


def _glslang(tmp_path, code, stage, *options):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", *options, str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _squash(code):
    return re.sub(r"\s+", " ", code)


def test_sibling_blocks_keep_their_own_declarations(tmp_path):
    source = """
float4 main() : SV_Target {
    float4 c = 0;
    { float r = 1; c.x = r; }
    { const float r = 2; c.y = r; }
    return c;
}
"""
    cgl = _translate(tmp_path, source, "blocks.hlsl", "cgl")
    assert "{ float r = 1; c.x = r; } { const float r = 2; c.y = r; }" in _squash(cgl)
    code = _translate(tmp_path, source, "blocks.hlsl", "opengl")
    _glslang(tmp_path, code, "frag")


CONSTANT_BUFFER_MATRICES = """
struct Camera { float4x4 view; float4x4 model[2]; };
ConstantBuffer<Camera> camera : register(b0);
float4 main(float3 n : NORMAL, uint instance : SV_InstanceID) : SV_Position {
    float3 w = mul((float3x3) camera.view, n);
    float4 p = mul((float4x3) camera.view, n);
    return mul(camera.model[instance], p) + float4(w, 0.0);
}
"""


def test_constant_buffer_members_have_their_struct_types(tmp_path):
    code = _translate(tmp_path, CONSTANT_BUFFER_MATRICES, "camera.vert.hlsl", "opengl")
    flat = _squash(code)
    assert "mat3(camera.view)" in flat
    assert "mat3x4(camera.view)" in flat
    # An element of a matrix array is a matrix; the array itself is never converted.
    assert "camera.model[instance]" in flat
    assert "transpose" not in flat
    _glslang(tmp_path, code, "vert")


def test_acceleration_structures_import_as_the_crossgl_type(tmp_path):
    source = """
RaytracingAccelerationStructure scene : register(t0);
struct Payload { float3 color; };
[shader("raygeneration")]
void main() {
    RayDesc ray;
    ray.Origin = 0; ray.Direction = float3(0, 0, 1); ray.TMin = 0; ray.TMax = 1;
    Payload payload;
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
}
"""
    cgl = _translate(
        tmp_path, source, "scene.rgen.hlsl", "cgl", source_backend="directx"
    )
    assert "accelerationStructureEXT scene;" in cgl


def test_buffer_blocks_do_not_spell_glsl_type_keywords(tmp_path):
    source = """
RWStructuredBuffer<float4> image : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { image[id.x] = float4(1, 1, 1, 1); }
"""
    code = _translate(tmp_path, source, "image.comp.hlsl", "opengl")
    assert not re.search(r"\bimageBuffer\b", code)
    _glslang(tmp_path, code, "comp")


OVERLAPPING_SEMANTICS = """
struct VSOutput {
    float4 Pos : SV_POSITION;
    float3 LightColor[2] : TEXCOORD2;
    float3 ViewVec : TEXCOORD3;
};
VSOutput main(float3 p : POSITION) {
    VSOutput o;
    o.Pos = float4(p, 1);
    o.LightColor[0] = p; o.LightColor[1] = p;
    o.ViewVec = p;
    return o;
}
"""


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_overlapping_interface_semantics_are_refused(tmp_path, backend):
    with pytest.raises(InterfaceLocationConflictError, match="overlaps"):
        _translate(tmp_path, OVERLAPPING_SEMANTICS, "overlap.vert.hlsl", backend)


HULL = """
struct VSOutput { float4 Pos : SV_POSITION; };
struct HSOutput { float4 Pos : SV_POSITION; };
struct ConstantsHSOutput {
    float TessLevelOuter[4] : SV_TessFactor;
    float TessLevelInner[2] : SV_InsideTessFactor;
};
ConstantsHSOutput ConstantsHS(InputPatch<VSOutput, 3> patch) {
    ConstantsHSOutput output = (ConstantsHSOutput)0;
    output.TessLevelOuter[0] = 2.0;
    output.TessLevelOuter[1] = 2.0;
    output.TessLevelOuter[2] = 2.0;
    output.TessLevelInner[0] = OUTER_READ;
    return output;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("ConstantsHS")]
HSOutput main(InputPatch<VSOutput, 3> patch, uint id : SV_OutputControlPointID) {
    HSOutput output;
    output.Pos = patch[id].Pos;
    return output;
}
"""


def test_tess_factor_arrays_narrow_to_the_domain_for_hlsl(tmp_path):
    source = HULL.replace("OUTER_READ", "output.TessLevelOuter[2]")
    code = _translate(tmp_path, source, "tri.tesc.hlsl", "directx")
    flat = _squash(code)
    assert "float TessLevelOuter[3] : SV_TessFactor;" in flat
    assert "float TessLevelInner[1] : SV_InsideTessFactor;" in flat
    # GLSL keeps gl_TessLevelOuter[4] and gl_TessLevelInner[2] for every domain.
    glsl = _translate(tmp_path, source, "tri.tesc.hlsl", "opengl")
    assert "gl_TessLevelOuter[2]" in glsl


@pytest.mark.parametrize(
    "read, reason",
    [
        ("output.TessLevelOuter[id]", "indexed with a value"),
        ("output.TessLevelOuter[3]", "indexed with a value"),
    ],
)
def test_tess_factor_arrays_that_reach_past_the_domain_are_refused(
    tmp_path, read, reason
):
    source = HULL.replace("OUTER_READ", read).replace(
        "ConstantsHS(InputPatch<VSOutput, 3> patch)",
        "ConstantsHS(InputPatch<VSOutput, 3> patch, uint id : SV_PrimitiveID)",
    )
    with pytest.raises(TessFactorUnsupportedError, match=reason):
        _translate(tmp_path, source, "tri.tesc.hlsl", "directx")
