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
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
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


HELPER_VIEW = """shader HelperView {
    buffer float* data;
    RWStructuredBuffer<float> outp;
    int count(buffer float* v) {
        return v.length();
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            outp[0] = float(count(data));
        }
    }
}"""


def test_hlsl_counts_a_helper_view_parameter_from_its_offset(tmp_path):
    """A helper's ``buffer T*`` parameter is passed as its buffer and an element
    offset, so its length() is the buffer's element count less the offset."""
    hlsl = HLSLCodeGen().generate(parse(HELPER_VIEW))

    assert "(int(cgl_buffer_count(v)) - int(v_offset))" in hlsl
    assert ".length()" not in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "helper_view.hlsl").write_text(hlsl, encoding="utf-8")
        _check(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-HV",
                "2021",
                "-E",
                "CSMain",
                "helper_view.hlsl",
                "-Fo",
                "helper_view.o",
            ],
            tmp_path,
        )


LOCAL_VIEW = """shader LocalView {
    buffer float* data;
    RWStructuredBuffer<float> outp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            buffer float* p = data + 2;
            outp[0] = float(p.length());
        }
    }
}"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            HELPER_VIEW.replace("count(data)", "count(data + 2)"),
            ["count(data, int64_t(2))", "(int(cgl_buffer_count(v)) - int(v_offset))"],
        ),
        (LOCAL_VIEW, ["float((int(cgl_buffer_count(data)) - int(p_offset)))"]),
    ],
    ids=["helper-offset", "local-view"],
)
def test_hlsl_counts_offset_views_from_their_offset(source, expected, tmp_path):
    """A view passed to a helper with an offset, and a local view, count the
    elements from their offset on, as GLSL output does."""
    hlsl = HLSLCodeGen().generate(parse(source))

    for text in expected:
        assert text in hlsl
    assert ".length()" not in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        (tmp_path / "view.hlsl").write_text(hlsl, encoding="utf-8")
        _check(
            [
                dxc,
                "-T",
                "cs_6_0",
                "-HV",
                "2021",
                "-E",
                "CSMain",
                "view.hlsl",
                "-Fo",
                "view.o",
            ],
            tmp_path,
        )


def test_hlsl_does_not_count_an_element_of_a_view_as_the_view():
    """data[3] is one float of the view, whose binding is the view moved to
    element 3, so its length() must not become the view's element count."""
    hlsl = HLSLCodeGen().generate(
        parse(LOCAL_VIEW.replace("float(p.length())", "float(data[3].length())"))
    )

    assert "cgl_buffer_count" not in hlsl


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


STRUCTURED_VIEWS = """shader StructuredViews {
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
    ("expression", "name"),
    [("float(values.length())", "'values'"), ("float(outp.length())", "'outp'")],
)
def test_metal_refuses_structured_buffer_lengths(expression, name):
    """A Metal buffer argument is a pointer and carries no size, so length()
    of a StructuredBuffer is refused like length() of a buffer float* view.

    The typed resource spelling used to slip past the pointer check, and the
    call was copied unchanged onto the device pointer, which the Metal
    compiler rejects as not a structure or union.
    """
    source = STRUCTURED_VIEWS.replace(
        "float(values.length()) + float(outp.length())", expression
    )
    with pytest.raises(BufferLengthUnsupportedError, match=name):
        MetalCodeGen().generate(parse(source))


@pytest.mark.parametrize(
    ("helper", "call", "name"),
    [
        ("int count() { return values.length(); }", "float(count())", "'values'"),
        (
            "int count(StructuredBuffer<float> v) { return v.length(); }",
            "float(count(values))",
            "'v'",
        ),
    ],
)
def test_metal_refuses_structured_buffer_lengths_in_helpers(helper, call, name):
    """A helper that reads a module-scope buffer gets it as a pointer argument,
    as does a helper with a StructuredBuffer parameter, so both are refused."""
    source = STRUCTURED_VIEWS.replace(
        "    compute {", f"    {helper}\n    compute {{", 1
    ).replace("float(values.length()) + float(outp.length())", call)
    with pytest.raises(BufferLengthUnsupportedError, match=name):
        MetalCodeGen().generate(parse(source))


@pytest.mark.parametrize(
    ("declaration", "expression"),
    [
        ("AppendStructuredBuffer<float> values;", "float(values.length())"),
        ("ConsumeStructuredBuffer<float> values;", "float(values.length())"),
        ("StructuredBuffer<float> values[2];", "float(values[1].length())"),
    ],
)
def test_metal_refuses_append_consume_and_array_element_lengths(
    declaration, expression
):
    """Append and consume buffers, and an element of a structured buffer
    array, are device pointers on Metal as well, so their length() is refused.

    Their length() used to be copied unchanged onto the pointer, which the
    Metal compiler rejects as not a structure or union.
    """
    source = STRUCTURED_VIEWS.replace(
        "StructuredBuffer<float> values;", declaration
    ).replace("float(values.length()) + float(outp.length())", expression)
    with pytest.raises(BufferLengthUnsupportedError, match="'values'"):
        MetalCodeGen().generate(parse(source))


def test_structured_buffer_array_element_lengths(tmp_path):
    """An element of a structured buffer array is one block of a block array
    on GLSL, so length() reads its member, and one resource on HLSL, so it
    counts through GetDimensions. Both used to copy the call unchanged.
    """
    source = STRUCTURED_VIEWS.replace(
        "StructuredBuffer<float> values;", "StructuredBuffer<float> values[2];"
    ).replace(
        "float(values.length()) + float(outp.length())", "float(values[1].length())"
    )
    glsl = GLSLCodeGen().generate(parse(source))
    hlsl = HLSLCodeGen().generate(parse(source))

    assert "float(values[1].data.length())" in glsl
    assert "float(int(cgl_buffer_count(values[1])))" in hlsl
    if shutil.which("glslangValidator"):
        (tmp_path / "probe.comp").write_text(glsl, encoding="utf-8")
        _check(["glslangValidator", "-S", "comp", "probe.comp"], tmp_path)
    if shutil.which("dxc"):
        (tmp_path / "probe.hlsl").write_text(hlsl, encoding="utf-8")
        _check(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", "probe.hlsl", "-Fo", "probe.o"],
            tmp_path,
        )


ARRAY_ELEMENT_LENGTHS = """shader ArrayElementLengths {
    BUFFER<float> values[2];
    RWStructuredBuffer<float> outp;
    int count() { return values[1].length(); }
    compute {
        layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x % 2u;
            for (int k = 0; k < values[i].length(); k++) {
                outp[i] += float(count());
            }
        }
    }
}"""


@pytest.mark.parametrize("buffer_type", ["StructuredBuffer", "RWStructuredBuffer"])
def test_structured_buffer_array_element_lengths_in_helpers_and_loops(
    buffer_type, tmp_path
):
    """An element selected at run time, and one read in a helper, are counted
    like a constant element, for read-only and writable buffer arrays."""
    source = ARRAY_ELEMENT_LENGTHS.replace("BUFFER", buffer_type)
    glsl = GLSLCodeGen().generate(parse(source))
    hlsl = HLSLCodeGen().generate(parse(source))

    assert "return values[1].data.length();" in glsl
    assert "(k < values[i].data.length())" in glsl
    assert "return int(cgl_buffer_count(values[1]));" in hlsl
    assert "(k < int(cgl_buffer_count(values[i])))" in hlsl
    if shutil.which("glslangValidator"):
        (tmp_path / "probe.comp").write_text(glsl, encoding="utf-8")
        _check(["glslangValidator", "-S", "comp", "probe.comp"], tmp_path)
    if shutil.which("dxc"):
        (tmp_path / "probe.hlsl").write_text(hlsl, encoding="utf-8")
        _check(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", "probe.hlsl", "-Fo", "probe.o"],
            tmp_path,
        )


def test_glsl_import_binds_anonymous_member_lengths(tmp_path):
    path = tmp_path / "anon.comp"
    path.write_text(ANONYMOUS_GLSL, encoding="utf-8")

    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)

    assert "data_2.data.length()" in crossgl
    assert ".outp.length()" in crossgl
    assert "float(data.length())" not in crossgl
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert len(re.findall(r"OpArrayLength %\w+ %\w+ [01]\b", spirv)) == 2
