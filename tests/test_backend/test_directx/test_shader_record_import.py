"""DXC shader record buffers reach CrossGL as shader record blocks with DXC's offsets.

See :mod:`crosstl.backend.DirectX.shader_record_import`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.DirectX.shader_record_import import HLSLShaderRecordError
from crosstl.translator.semantics.shader_records import (
    ShaderRecordBufferUnsupportedError,
)

MISS = """struct Payload { float3 hitValue; };
struct SBT { %s };
[[vk::%s]] ConstantBuffer<SBT> sbt;
[shader("miss")] void main(inout Payload p) { p.hitValue = %s; }
"""

PACKED = "float a; float3 b; float c[2];"
PACKED_USE = "sbt.b * sbt.a + sbt.c[1]"


def _write(tmp_path, source, name="record.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def _miss(tmp_path, members=PACKED, use=PACKED_USE, attribute="shader_record_ext"):
    return _write(tmp_path, MISS % (members, attribute, use))


def _translate(path, backend):
    return crosstl.translate(
        path, backend=backend, source_backend="directx", format_output=False
    )


def _spirv_offsets(spirv):
    """Member offsets of the one Block-decorated struct, the record."""
    (block,) = re.findall(r"OpDecorate (%\w+) Block\b", spirv)
    return [
        int(offset)
        for offset in re.findall(
            rf"OpMemberDecorate {re.escape(block)} \d+ Offset (\d+)", spirv
        )
    ]


@pytest.mark.parametrize("attribute", ["shader_record_ext", "shader_record_nv"])
def test_record_becomes_a_crossgl_shader_record_block(tmp_path, attribute):
    path = _miss(
        tmp_path, "float r; float g; float b;", "float3(sbt.r, sbt.g, sbt.b)", attribute
    )
    crossgl = _translate(path, "cgl")
    assert "SBT sbt @glsl_buffer_block(shaderRecordEXT, std430);" in crossgl
    assert "vk_shader_record" not in crossgl
    assert "ConstantBuffer<SBT>" not in crossgl


def test_record_reaches_the_spirv_shader_record_storage_class(tmp_path):
    spirv = _translate(_miss(tmp_path), "vulkan")
    variable = re.search(r"%(\w+) = OpVariable %\w+ ShaderRecordBufferKHR", spirv)
    assert variable is not None
    assert not re.search(
        rf"OpDecorate %{variable.group(1)} (?:DescriptorSet|Binding)\b", spirv
    )
    assert "Uniform" not in re.sub(r"UniformConstant", "", spirv)
    # DXC's relaxed std430 offsets: b packs after a, the array keeps a four-byte stride.
    assert _spirv_offsets(spirv) == [0, 4, 16]
    assert re.search(r"ArrayStride 4\b", spirv)
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = _write(tmp_path, spirv, "record.spvasm")
        binary = str(tmp_path / "record.spv")
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.2", assembly, "-o", binary],
            check=True,
        )
        subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", "--scalar-block-layout", binary],
            check=True,
        )


def test_straddling_vector_keeps_std430(tmp_path):
    # b would straddle the 16-byte boundary at offset 12, so DXC moves it to 16 as std430 does.
    path = _miss(tmp_path, "float a; float x; float y; float3 b;", "sbt.b * sbt.a")
    assert "@glsl_buffer_block(shaderRecordEXT, std430)" in _translate(path, "cgl")
    assert _spirv_offsets(_translate(path, "vulkan")) == [0, 4, 8, 16]


def test_record_reaches_glsl_without_a_binding(tmp_path):
    glsl = _translate(_miss(tmp_path), "opengl")
    declaration = re.search(r"layout\(([^)]*)\)\s*buffer\s+SBT", glsl)
    assert declaration is not None
    assert [part.strip() for part in declaration.group(1).split(",")] == [
        "shaderRecordEXT",
        "scalar",
    ]
    assert "#extension GL_EXT_scalar_block_layout : require" in glsl
    if shutil.which("glslangValidator"):
        path = _write(tmp_path, glsl, "record.rmiss")
        result = subprocess.run(
            [
                "glslangValidator",
                "--target-env",
                "vulkan1.2",
                "-S",
                "rmiss",
                "-o",
                str(tmp_path / "record.spv"),
                path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout


def test_record_is_refused_for_direct3d(tmp_path):
    with pytest.raises(ShaderRecordBufferUnsupportedError, match="shaderRecordEXT"):
        _translate(_miss(tmp_path), "directx")


@pytest.mark.parametrize(
    "members, use, reason",
    [
        # Relaxed: b at 4, c at 16 with stride 16. std430 puts b at 16; scalar gives c a stride of 12.
        ("float a; float3 b; float3 c[2];", "sbt.b + sbt.c[1]", "match neither"),
        ("float4x4 m;", "sbt.m[0].xyz", "float4x4"),
        ("half h;", "float3(sbt.h, 0, 0)", "half"),
        ("min16float h;", "float3(sbt.h, 0, 0)", "min16float"),
    ],
)
def test_record_without_a_crossgl_layout_is_refused(tmp_path, members, use, reason):
    with pytest.raises(HLSLShaderRecordError, match=reason) as error:
        _translate(_miss(tmp_path, members, use), "cgl")
    assert (
        error.value.project_diagnostic_code
        == "project.translate.hlsl-shader-record-unsupported"
    )


def test_cbuffer_record_is_refused(tmp_path):
    source = """struct Payload { float3 hitValue; };
[[vk::shader_record_ext]] cbuffer SBT : register(b3) { float a; float3 b; };
[shader("miss")] void main(inout Payload p) { p.hitValue = b * a; }
"""
    with pytest.raises(HLSLShaderRecordError, match="ConstantBuffer<T> form"):
        _translate(_write(tmp_path, source), "vulkan")
