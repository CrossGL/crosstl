"""``length()`` of runtime-sized buffer views (GLSL 4.60 section 4.1.9) on HLSL, SPIR-V, Metal and GLSL import.

A ``buffer T* name`` view indexes as its block's runtime array, so ``length()`` is that array's element count:
``GetDimensions`` on HLSL, ``OpArrayLength`` on SPIR-V. A member of an anonymous block is bound to the block's
instance by the parser. A Metal buffer argument carries no size, so Metal refuses both queries.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.buffer_length import BufferLengthUnsupportedError
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

VIEWS = """shader Views {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(std430, binding = 3) buffer Data { float data[]; };
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = data[i] + float(outp.length()) + float(data.length());
        }
    }
}"""

ANONYMOUS_GLSL = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 3) buffer Data { uint count; float data[]; };
layout(std430, binding = 4) buffer Out { float outp[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    outp[i] = float(count) + data[i] + float(data.length()) + float(outp.length());
}
"""


def _check(command, tmp_path):
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_counts_structured_and_byte_address_views(tmp_path):
    hlsl = HLSLCodeGen().generate(parse(VIEWS))

    assert "int(cgl_buffer_count(outp))" in hlsl
    assert "int((cgl_buffer_bytes(data_) - 0u) / 4u)" in hlsl
    assert ".length()" not in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "views.hlsl").write_text(hlsl, encoding="utf-8")
        _check(
            [dxc, "-T", "cs_6_0", "-E", "CSMain", "views.hlsl", "-Fo", "views.o"],
            tmp_path,
        )


def test_spirv_reads_the_runtime_array_length(tmp_path):
    spirv = VulkanSPIRVCodeGen().generate(parse(VIEWS))

    assert len(re.findall(r"OpArrayLength %\d+ %\d+ 0\b", spirv)) == 2
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "views.spvasm").write_text(spirv, encoding="utf-8")
        _check(
            ["spirv-as", "--target-env", "vulkan1.1", "views.spvasm", "-o", "v.spv"],
            tmp_path,
        )
        _check(["spirv-val", "--target-env", "vulkan1.1", "v.spv"], tmp_path)


@pytest.mark.parametrize(
    "expression, name",
    [("float(outp.length())", "'outp'"), ("float(data.length())", "'data'")],
)
def test_metal_refuses_view_lengths(expression, name):
    source = VIEWS.replace(
        "data[i] + float(outp.length()) + float(data.length())", expression
    )
    with pytest.raises(BufferLengthUnsupportedError, match=name):
        MetalCodeGen().generate(parse(source))


def test_glsl_import_binds_anonymous_member_lengths(tmp_path):
    path = tmp_path / "anon.comp"
    path.write_text(ANONYMOUS_GLSL, encoding="utf-8")

    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)

    assert "data_2.data.length()" in crossgl
    assert ".outp.length()" in crossgl
    assert "float(data.length())" not in crossgl
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert len(re.findall(r"OpArrayLength %\w+ %\w+ [01]\b", spirv)) == 2


STRUCTURED_PROBE = """shader StructuredProbe {
    StructuredBuffer<float> values;
    RWStructuredBuffer<float> outp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            outp[0] = float(values.length()) + float(outp.length());
        }
    }
}"""


@pytest.mark.parametrize(
    "expression, name",
    [("float(values.length())", "'values'"), ("float(outp.length())", "'outp'")],
)
def test_metal_refuses_structured_buffer_lengths(expression, name):
    source = STRUCTURED_PROBE.replace(
        "float(values.length()) + float(outp.length())", expression
    )
    with pytest.raises(BufferLengthUnsupportedError, match=name):
        MetalCodeGen().generate(parse(source))
