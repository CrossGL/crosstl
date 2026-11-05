"""Direct3D's ``SV_VertexID`` excludes the draw's base vertex; OpenGL, Vulkan and Metal vertex indices include it.

HLSL and Slang input read ``SV_VertexID`` as ``gl_VertexID`` (base-inclusive), so GLSL, SPIR-V and Metal output read
the vertex ``BaseVertex`` positions further on whenever the draw had a base vertex. It now imports as
``gl_RelativeVertexID``, which Direct3D and Slang output write as ``SV_VertexID`` and other targets lower to the vertex
index minus the base vertex. See :mod:`crosstl.translator.semantics.relative_vertex_ids`.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import RelativeVertexIdUnsupportedError
from tests.conformance.corpus import checks

PARAMETER = """float4 main(uint vid : SV_VertexID) : SV_Position {
    return float4(float(vid), 0.0, 0.0, 1.0);
}
"""

# The value as a stage input member, read in a helper, next to an explicit base vertex parameter.
MEMBER = """struct VSIn { float3 pos : POSITION; uint vid : SV_VertexID; };
struct VSOut { float4 pos : SV_Position; float id : TEXCOORD0; };
float scaled(uint v) { return float(v) * 0.5; }
VSOut main(VSIn input, int start : SV_StartVertexLocation) {
    VSOut o;
    o.pos = float4(input.pos, 1.0);
    o.id = scaled(input.vid) + float(start);
    return o;
}
"""

NESTED = """struct Ids { uint vid : SV_VertexID; };
struct VSIn { float3 pos : POSITION; Ids ids; };
float4 main(VSIn input) : SV_Position { return float4(input.pos, float(input.ids.vid)); }
"""

CROSSGL_READ = """shader S {
    vertex {
        float index() { return float(gl_RelativeVertexID); }
        vec4 main() @ gl_Position { return vec4(index(), 0.0, 0.0, 1.0); }
    }
}
"""

SLANG = """[shader("vertex")]
float4 main(uint vid : SV_VertexID) : SV_Position {
    return float4(float(vid), 0.0, 0.0, 1.0);
}
"""

ZERO_BASE = {"zero_base_vertex": True}


def translate(tmp_path, source, backend, name="shader.hlsl", source_options=None):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_options=source_options
    )


def _run(command, cwd):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    ("source", "name", "expected"),
    [
        (PARAMETER, "shader.hlsl", "uint vid @ gl_RelativeVertexID"),
        (MEMBER, "shader.hlsl", "uint vid @ gl_RelativeVertexID;"),
        (SLANG, "shader.slang", "uint vid @ gl_RelativeVertexID"),
    ],
)
def test_sv_vertex_id_imports_as_the_relative_vertex_id(
    tmp_path, source, name, expected
):
    assert expected in translate(tmp_path, source, "cgl", name)
    zero_base = translate(tmp_path, source, "cgl", name, ZERO_BASE)
    assert expected.replace("gl_RelativeVertexID", "gl_VertexID") in zero_base
    assert "gl_RelativeVertexID" not in zero_base


@pytest.mark.parametrize("backend", ["directx", "dx11", "slang"])
def test_direct3d_targets_keep_sv_vertex_id(tmp_path, backend):
    code = translate(tmp_path, PARAMETER, backend)
    assert re.search(r"uint vid\s*:\s*SV_VertexID", code)
    assert "StartVertexLocation" not in code
    assert "gl_RelativeVertexID" not in code


def test_hlsl_round_trip_compiles_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not available")
    for source in (PARAMETER, translate(tmp_path, PARAMETER, "cgl")):
        name = "shader.hlsl" if source is PARAMETER else "shader.cgl"
        (tmp_path / "out.hlsl").write_text(translate(tmp_path, source, "directx", name))
        result = _run([dxc, "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_subtracts_the_base_vertex(tmp_path):
    assembly = translate(tmp_path, PARAMETER, "vulkan")
    assert "OpCapability DrawParameters" in assembly
    assert re.search(r"OpDecorate %\w+ BuiltIn VertexIndex", assembly)
    assert re.search(r"OpDecorate %\w+ BuiltIn BaseVertex", assembly)
    assert "OpISub" in assembly
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        pytest.skip("spirv-as and spirv-val are required")
    (tmp_path / "out.spvasm").write_text(assembly)
    result = _run([spirv_as, "out.spvasm", "-o", "out.spv"], tmp_path)
    assert result.returncode == 0, result.stderr
    result = _run([spirv_val, "out.spv"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_glsl_subtracts_the_base_vertex(tmp_path):
    code = translate(tmp_path, PARAMETER, "opengl")
    assert "uint vid_index = uint(gl_VertexID);" in code
    assert "uint vid = (vid_index - uint(gl_BaseVertexARB));" in code
    artifact = tmp_path / "out.vert"
    artifact.write_text(code)
    verdict = checks.check(artifact, "glsl", ["vertex"])
    if verdict["verdict"] == "unavailable":
        pytest.skip("no GLSL checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


def test_metal_subtracts_the_base_vertex(tmp_path):
    code = translate(tmp_path, PARAMETER, "metal")
    assert re.search(r"uint vid_index \[\[vertex_id\]\]", code)
    assert "[[base_vertex]]" in code
    assert "uint vid = vid_index - uint(cgl_base_vertex);" in code
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    verdict = checks.check(artifact, "metal", ["vertex"])
    if verdict["verdict"] == "unavailable":
        pytest.skip("no metal checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


@pytest.mark.parametrize(
    ("backend", "checker", "suffix"),
    [("vulkan", None, None), ("opengl", "glsl", ".vert"), ("metal", "metal", ".metal")],
)
def test_input_structure_member_is_rebased_once(tmp_path, backend, checker, suffix):
    code = translate(tmp_path, MEMBER, backend)
    if backend == "vulkan":
        # The explicit SV_StartVertexLocation parameter is the base vertex subtracted: one BaseVertex input.
        assert len(re.findall(r"BuiltIn BaseVertex", code)) == 1
        return
    # The entry point rebases a copy of the stage input once, by the base vertex parameter it already has.
    assert len(re.findall(r"input_?\.vid = \(?input_?\.vid - uint\(", code)) == 1
    assert "cgl_base_vertex" not in code
    artifact = tmp_path / f"out{suffix}"
    artifact.write_text(code)
    verdict = checks.check(artifact, checker, ["vertex"])
    if verdict["verdict"] == "unavailable":
        pytest.skip(f"no {checker} checker installed")
    assert verdict["verdict"] == "accepted", verdict.get("log", "")


def test_crossgl_read_by_name(tmp_path):
    glsl = translate(tmp_path, CROSSGL_READ, "opengl", "shader.cgl")
    assert "(gl_VertexID - gl_BaseVertexARB)" in glsl
    hlsl = translate(tmp_path, CROSSGL_READ, "directx", "shader.cgl")
    assert "SV_VertexID" in hlsl and "gl_RelativeVertexID" not in hlsl


@pytest.mark.parametrize("backend", ["webgl", "wgsl"])
def test_targets_without_a_base_vertex_refuse_it(tmp_path, backend):
    with pytest.raises(RelativeVertexIdUnsupportedError) as caught:
        translate(tmp_path, PARAMETER, backend)
    assert caught.value.code == "project.translate.relative-vertex-id-unsupported"
    assert "zero_base_vertex" in caught.value.reason
    # A declared zero base vertex makes SV_VertexID the target's own vertex index.
    code = translate(tmp_path, PARAMETER, backend, source_options=ZERO_BASE)
    assert "gl_VertexID" in code or "vertex_index" in code


def test_nested_member_is_refused(tmp_path):
    with pytest.raises(RelativeVertexIdUnsupportedError, match="nested member 'ids'"):
        translate(tmp_path, NESTED, "vulkan")


# A vertex function without rasterization records the vertex ID each invocation reads.
RECORDING = """RWStructuredBuffer<uint> seen : register(u0);
void main(uint vid : SV_VertexID) {
    seen[vid % 16u] = vid + 1000u;
}
"""


def _metal_draw(source, indexed):
    """The 16 words a Metal draw of three points leaves in ``seen``: indexed from base vertex 5, or from vertex 4."""
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, str(error)
    descriptor = Metal.MTLRenderPipelineDescriptor.new()
    descriptor.setVertexFunction_(library.newFunctionWithName_("vertex_main"))
    descriptor.setRasterizationEnabled_(False)
    state, error = device.newRenderPipelineStateWithDescriptor_error_(descriptor, None)
    assert state is not None, str(error)
    shared = Metal.MTLResourceStorageModeShared
    seen = device.newBufferWithLength_options_(64, shared)
    seen.contents().as_buffer(64)[:] = bytes(64)
    indices = device.newBufferWithBytes_length_options_(
        struct.pack("<3H", 0, 1, 2), 6, shared
    )
    commands = device.newCommandQueue().commandBuffer()
    render_pass = Metal.MTLRenderPassDescriptor.renderPassDescriptor()
    render_pass.setRenderTargetWidth_(1)
    render_pass.setRenderTargetHeight_(1)
    render_pass.setDefaultRasterSampleCount_(1)
    encoder = commands.renderCommandEncoderWithDescriptor_(render_pass)
    encoder.setRenderPipelineState_(state)
    encoder.setVertexBuffer_offset_atIndex_(seen, 0, 0)
    point = Metal.MTLPrimitiveTypePoint
    if indexed:
        encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance_(  # noqa: E501
            point, 3, Metal.MTLIndexTypeUInt16, indices, 0, 1, 5, 0
        )
    else:
        encoder.drawPrimitives_vertexStart_vertexCount_(point, 4, 3)
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack("<16I", bytes(seen.contents().as_buffer(64)))


@pytest.mark.parametrize("indexed", [True, False], ids=["indexed", "arrays"])
def test_metal_draw_reads_the_direct3d_vertex_id(tmp_path, indexed):
    # Direct3D numbers the three vertices 0, 1, 2 for DrawIndexed(3, 0, 5) and for Draw(3, 4). [[vertex_id]] reads
    # 5, 6, 7 and 4, 5, 6; [[base_vertex]] is 5 and 4 (the base vertex, or the first vertex of a non-indexed draw).
    expected = (1000, 1001, 1002) + (0,) * 13
    assert _metal_draw(translate(tmp_path, RECORDING, "metal"), indexed) == expected
    zero_base = translate(tmp_path, RECORDING, "metal", source_options=ZERO_BASE)
    assert _metal_draw(zero_base, indexed) != expected
