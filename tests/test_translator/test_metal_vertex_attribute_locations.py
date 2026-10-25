"""Metal vertex attribute indices are the input locations of every target.

A Metal ``[[attribute(n)]]`` member of a ``[[stage_in]]`` structure is the vertex input the host binds at index ``n``,
the same slot as GLSL's ``layout(location = n)``, SPIR-V's ``Location n`` and the ``TEXCOORDn`` semantic CrossTL gives
location ``n`` in HLSL. The importer used to rename indices 0 to 11 to HLSL semantic names (``POSITION``,
``TEXCOORD`` for 4, ``TEXCOORD0`` for 5, ...) and keep higher ones as ``@attribute(n)``, which no generator reads:
HLSL output refused the colliding ``TEXCOORD``/``TEXCOORD0`` pair, and GLSL, SPIR-V and Metal output renumbered the
attribute sequentially. The importer now emits ``@location(n)``.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl

METAL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct VertexIn {
        float3 position [[attribute(0)]];
        float2 uv [[attribute(4)]];
        float4 tint [[attribute(5)]];
        float3 offset [[attribute(13)]];
    };

    struct VertexOut {
        float4 position [[position]];
        float2 uv;
        float4 tint;
    };

    vertex VertexOut vertexMain(VertexIn in [[stage_in]]) {
        VertexOut out;
        out.position = float4(in.position + in.offset, 1.0);
        out.uv = in.uv;
        out.tint = in.tint;
        return out;
    }
    """)

LOCATIONS = {"position": 0, "uv": 4, "tint": 5, "offset": 13}


def _translate(tmp_path, backend):
    path = tmp_path / "attributes.metal"
    path.write_text(METAL, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, *command):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_crossgl_records_the_attribute_index_as_a_location(tmp_path):
    code = _translate(tmp_path, "cgl")
    for name, location in LOCATIONS.items():
        assert re.search(rf"\b{name} @location\({location}\);", code), code
    assert not re.search(r"@(POSITION|TEXCOORD\d*|attribute)\b", code)


def test_opengl_binds_each_attribute_at_its_index(tmp_path):
    code = _translate(tmp_path, "opengl")
    for name, location in LOCATIONS.items():
        assert re.search(rf"layout\(location = {location}\) in \w+ {name};", code)
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "a.vert").write_text(code, encoding="utf-8")
    _run(tmp_path, "glslangValidator", "a.vert")


def test_directx_gives_each_attribute_its_location_semantic(tmp_path):
    code = _translate(tmp_path, "directx")
    for name, location in LOCATIONS.items():
        assert re.search(rf"\b{name}\s*:\s*TEXCOORD{location};", code), code


def test_spirv_decorates_each_attribute_with_its_location(tmp_path):
    code = _translate(tmp_path, "vulkan")
    for name, location in LOCATIONS.items():
        ids = re.findall(rf'OpName (%\S+) "CrossGL_vertex_input_\w*_{name}"', code)
        assert ids, name
        assert any(
            re.search(rf"OpDecorate {re.escape(i)} Location {location}\b", code)
            for i in ids
        ), name
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "a.spvasm").write_text(code, encoding="utf-8")
    _run(tmp_path, "spirv-as", "--target-env", "vulkan1.0", "a.spvasm", "-o", "a.spv")
    _run(tmp_path, "spirv-val", "--target-env", "vulkan1.0", "a.spv")


def test_metal_keeps_each_attribute_index(tmp_path):
    code = _translate(tmp_path, "metal")
    for name, location in LOCATIONS.items():
        assert re.search(rf"\b{name} \[\[attribute\({location}\)\]\];", code), code
    if not shutil.which("xcrun"):
        pytest.skip("the Metal toolchain is not installed")
    (tmp_path / "a.metal").write_text(code, encoding="utf-8")
    _run(tmp_path, "xcrun", "-sdk", "macosx", "metal", "-c", "a.metal", "-o", "a.air")
