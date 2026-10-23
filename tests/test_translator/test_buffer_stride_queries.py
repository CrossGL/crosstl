"""The stride output of ``StructuredBuffer::GetDimensions`` on targets without a stride query.

HLSL's ``GetDimensions(out uint count, out uint stride)`` is CrossGL's ``buffer_dimensions(b, count, stride)``.
GLSL's ``length()``, SPIR-V's ``OpArrayLength`` and the element count a Metal host passes report no stride, so the
OpenGL and Metal output assigned ``count`` and left ``stride`` unassigned, and SPIR-V output refused the query. The
stride is now the element stride of the layout the target gives the buffer: the declared HLSL structured-buffer
layout of an imported buffer, the target's own layout (std430, MSL) of an unmarked one.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.semantics import BufferStrideUnsupportedError

# Light takes 32 bytes under every rule set; Spot takes 16 under the HLSL structured-buffer rules and 32 under
# std430 and MSL, where its float3 is 16-byte aligned.
HLSL = textwrap.dedent("""
    struct Light { float4 position; float3 color; float radius; };
    struct Spot { float angle; float3 direction; };
    RWStructuredBuffer<Light> lights : register(u0);
    StructuredBuffer<Spot> spots : register(t1);
    RWStructuredBuffer<uint> results : register(u2);

    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID)
    {
        uint lightCount, lightStride, spotCount, spotStride;
        lights.GetDimensions(lightCount, lightStride);
        if (id.x == 0)
            spots.GetDimensions(spotCount, spotStride);
        results[0] = lightCount;
        results[1] = lightStride;
        results[2] = spotCount;
        results[3] = spotStride;
    }
    """)

# An unmarked buffer has the target's layout: {vec3, float} takes 16 bytes under std430 and 32 under MSL.
CGL = textwrap.dedent("""
    shader main {
        struct Packed {
            vec3 a;
            float b;
        }
        @ binding(0)
        RWStructuredBuffer<Packed> items;
        @ binding(1)
        RWStructuredBuffer<uint> results;
        compute {
            @ numthreads(1, 1, 1)
            void main(uvec3 id @ gl_GlobalInvocationID) {
                uint count;
                uint stride;
                buffer_dimensions(items, count, stride);
                results[0] = count;
                results[1] = stride;
            }
        }
    }
    """)


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"strides{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _run(tmp_path, *commands):
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr[-2000:]


def test_opengl_assigns_the_declared_hlsl_strides(tmp_path):
    code = _translate(tmp_path, HLSL, ".comp.hlsl", "opengl")
    assert re.search(r"lightCount = lights\.length\(\);\s*lightStride = 32u;", code)
    assert re.search(r"spotCount = spots\.length\(\);\s*spotStride = 16u;", code)
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not installed")
    (tmp_path / "a.comp").write_text(code, encoding="utf-8")
    _run(tmp_path, ["glslangValidator", "a.comp"])


def test_spirv_lowers_the_stride_query(tmp_path):
    code = _translate(tmp_path, HLSL, ".comp.hlsl", "vulkan")
    assert "OpArrayLength %uint %lights 0" in code
    assert re.search(r"OpConstant %uint 32\b", code)
    assert re.search(r"OpConstant %uint 16\b", code)
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "a.spvasm").write_text(code, encoding="utf-8")
    _run(
        tmp_path,
        ["spirv-as", "--target-env", "vulkan1.0", "a.spvasm", "-o", "a.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "a.spv"],
    )


def test_metal_assigns_the_declared_hlsl_strides(tmp_path):
    code = _translate(tmp_path, HLSL, ".comp.hlsl", "metal")
    assert re.search(r"lightStride = 32u;", code)
    assert re.search(r"spotStride = 16u;", code)
    assert "packed_float3 direction;" in code


def test_hlsl_keeps_get_dimensions(tmp_path):
    code = _translate(tmp_path, HLSL, ".comp.hlsl", "directx")
    assert "lights.GetDimensions(lightCount, lightStride);" in code
    assert "spots.GetDimensions(spotCount, spotStride);" in code


@pytest.mark.parametrize(
    ("backend", "stride"), [("opengl", 16), ("vulkan", 16), ("metal", 32)]
)
def test_unmarked_buffers_use_the_target_layout(tmp_path, backend, stride):
    code = _translate(tmp_path, CGL, ".cgl", backend)
    if backend == "vulkan":
        assert re.search(rf"OpConstant %uint {stride}\b", code)
        assert "OpArrayLength" in code
    else:
        assert f"stride = {stride}u;" in code


def test_stride_query_on_a_value_is_refused(tmp_path):
    source = CGL.replace(
        "buffer_dimensions(items, count, stride);",
        "results[2] = buffer_dimensions(results, count, stride);",
    )
    with pytest.raises(BufferStrideUnsupportedError, match="is a statement"):
        _translate(tmp_path, source, ".cgl", "opengl")


def test_stride_query_on_a_non_buffer_is_refused(tmp_path):
    source = CGL.replace(
        "buffer_dimensions(items, count, stride);",
        "buffer_dimensions(count, count, stride);",
    )
    with pytest.raises(BufferStrideUnsupportedError, match="structured-buffer"):
        _translate(tmp_path, source, ".cgl", "metal")
