"""OpenGL vertex inputs read by name in HLSL, Metal and Slang output.

GLSL reads ``gl_VertexID``, ``gl_InstanceID`` and the draw parameters ``gl_BaseVertex``, ``gl_BaseInstance`` and
``gl_DrawID`` as globals (GLSL 4.60 section 7.1.1), and the CrossGL import keeps those reads. The HLSL and Metal
vertex generators bound only the Vulkan names, so an entry point's read of the OpenGL ones was copied verbatim as an
undeclared identifier; a helper's read of a draw parameter was copied on every target. Each read now arrives through
an entry parameter carrying the built-in, and the targets without a draw index refuse ``gl_DrawID``.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

GLSL = """#version 460
layout(location = 0) in vec4 position;
layout(location = 0) out float value;
void main() {
    value = float(gl_VertexID - 1) + float(gl_InstanceID) + float(gl_BaseVertex) + float(gl_BaseInstance);
    gl_Position = position;
}
"""


def _program(body, helper="", parameters=""):
    return f"""shader Probe {{
    struct VertexInput {{
        vec4 position;
    }}
    struct VertexOutput {{
        vec4 position @ gl_Position;
        float value;
    }}
    vertex {{
        {helper}
        VertexOutput main(VertexInput input{parameters}) {{
            VertexOutput output;
            output.value = {body};
            output.position = input.position;
            return output;
        }}
    }}
}}
"""


HELPER = _program(
    "vertexValue()",
    helper="float vertexValue() { return float(gl_VertexID + gl_BaseVertex + gl_BaseInstance); }",
)


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _bare_reads(code, names):
    return [name for name in names if re.search(rf"\b{name}\b", code)]


OPENGL_INPUTS = ("gl_VertexID", "gl_InstanceID", "gl_BaseVertex", "gl_BaseInstance")


def test_hlsl_output_declares_the_opengl_vertex_inputs_of_a_glsl_shader(tmp_path):
    hlsl = _translate(tmp_path, GLSL, "directx", ".vert")
    assert re.search(r"uint \w+ : SV_VertexID", hlsl)
    assert re.search(r"uint \w+ : SV_InstanceID", hlsl)
    assert re.search(r"int \w+ : SV_StartVertexLocation", hlsl)
    assert re.search(r"uint \w+ : SV_StartInstanceLocation", hlsl)
    assert _bare_reads(hlsl, OPENGL_INPUTS) == []


def test_metal_output_declares_the_opengl_vertex_inputs_of_a_glsl_shader(tmp_path):
    metal = _translate(tmp_path, GLSL, "metal", ".vert")
    for attribute in ("vertex_id", "instance_id", "base_vertex", "base_instance"):
        assert re.search(rf"uint \w+ \[\[{attribute}\]\]", metal), attribute
    assert _bare_reads(metal, OPENGL_INPUTS) == []


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [
        ("directx", r"uint (\w+) : SV_{}"),
        ("metal", r"uint (\w+) \[\[{}\]\]"),
    ],
)
def test_reads_of_the_opengl_vertex_ids_convert_to_glsl_int(
    tmp_path, backend, declaration
):
    code = _translate(tmp_path, GLSL, backend, ".vert")
    # Metal's instance_id includes the base instance, which is subtracted first: see test_metal_draw_base_indices.
    attributes = ("VertexID", "InstanceID") if backend == "directx" else ("vertex_id",)
    for attribute in attributes:
        raw = re.search(declaration.format(attribute), code)
        assert raw is not None, attribute
        # The uint value is converted once, so gl_VertexID - 1 stays signed.
        assert re.search(rf"\bint \w+ = int\({raw.group(1)}\);", code), attribute


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("directx", r"uint (\w+) : SV_VertexID"),
        ("metal", r"uint (\w+) \[\[vertex_id\]\]"),
    ],
)
def test_an_existing_uint_vertex_id_parameter_is_reused_and_read_as_int(
    tmp_path, backend, expected
):
    source = _program(
        "float(gl_VertexID - 1) + float(vid)", parameters=", uint vid @ gl_VertexID"
    )
    code = _translate(tmp_path, source, backend)
    assert re.findall(expected, code) == ["vid"]
    assert "int(vid) - 1" in code


def test_hlsl_output_declares_sv_vertex_id_once_for_both_vertex_index_names(tmp_path):
    source = _program("float(gl_VertexID) + float(gl_VertexIndex)")
    hlsl = _translate(tmp_path, source, "directx")
    assert len(re.findall(r"SV_VertexID", hlsl)) == 1
    assert _bare_reads(hlsl, ("gl_VertexID", "gl_VertexIndex")) == []


@pytest.mark.parametrize(
    ("backend", "declarations"),
    [
        (
            "directx",
            (
                r"int \w+ : SV_StartVertexLocation",
                r"uint \w+ : SV_StartInstanceLocation",
                r"uint \w+ : SV_VertexID",
            ),
        ),
        (
            "metal",
            (
                r"uint \w+ \[\[base_vertex\]\]",
                r"uint \w+ \[\[base_instance\]\]",
                r"uint \w+ \[\[vertex_id\]\]",
            ),
        ),
        (
            "slang",
            (
                r"int \w+ : SV_StartVertexLocation",
                r"uint \w+ : SV_StartInstanceLocation",
                r"uint \w+ : SV_VertexID",
            ),
        ),
    ],
)
def test_a_helper_read_of_the_draw_parameters_is_passed_from_the_entry_point(
    tmp_path, backend, declarations
):
    code = _translate(tmp_path, HELPER, backend)
    for declaration in declarations:
        assert re.search(declaration, code), declaration
    assert re.search(r"float vertexValue\(int \w+, int \w+, int \w+\)", code)
    assert _bare_reads(code, ("gl_VertexID", "gl_BaseVertex", "gl_BaseInstance")) == []


@pytest.mark.parametrize(
    ("backend", "error", "message"),
    [
        ("directx", DirectXLoweringRefused, "@gl_DrawID"),
        ("metal", MetalLoweringRefused, "built-in @gl_DrawID"),
    ],
)
@pytest.mark.parametrize("in_helper", [False, True], ids=["entry", "helper"])
def test_a_draw_index_read_is_refused_without_a_draw_index_value(
    tmp_path, backend, error, message, in_helper
):
    if in_helper:
        source = _program(
            "drawValue()", helper="float drawValue() { return float(gl_DrawID); }"
        )
    else:
        source = _program("float(gl_DrawID)")
    with pytest.raises(error, match=re.escape(message)):
        _translate(tmp_path, source, backend)


def test_slang_output_passes_a_helper_draw_index_read_as_sv_draw_index(tmp_path):
    source = _program(
        "drawValue()", helper="float drawValue() { return float(gl_DrawID); }"
    )
    slang = _translate(tmp_path, source, "slang")
    assert re.search(r"uint \w+ : SV_DrawIndex", slang)
    assert re.search(r"float drawValue\(int \w+\)", slang)
    assert _bare_reads(slang, ("gl_DrawID",)) == []


def test_the_dx11_profile_refuses_a_base_vertex_read(tmp_path):
    source = _program("float(gl_BaseVertex)")
    with pytest.raises(
        DirectXLoweringRefused, match="SV_StartVertexLocation requires Shader Model 6.8"
    ):
        _translate(tmp_path, source, "dx11")
