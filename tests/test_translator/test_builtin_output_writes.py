"""Built-in outputs written by name, ``in``/``out`` stage variables, and ``uint`` system-value outputs on HLSL and MSL."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.stage_interface_structs import (
    StageInterfaceUnsupportedError,
)

GRAPHICS = ("metal", "directx")


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile_metal(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


BARE = """shader S {
    vertex {
        in vec3 pos @location(0);
        void main() {
            gl_PointSize = 2.0;
            gl_Position = vec4(pos, 1.0);
        }
    }
    fragment {
        out vec4 color @location(0);
        void main() {
            color = vec4(1.0);
            gl_FragDepth = 0.5;
            gl_SampleMask[0] = 1;
        }
    }
}
"""


@pytest.mark.parametrize("backend", GRAPHICS)
def test_bare_builtin_output_stores_become_output_members(tmp_path, backend):
    code = _translate(tmp_path, BARE, backend)
    assert not re.search(
        r"\bgl_(Position|PointSize|FragDepth|SampleMask)\b", code
    ), code
    assert re.search(r"output\.\w+ = 0\.5;", code), code
    assert re.search(r"output\.\w+ = uint\(1\);", code), code
    if backend == "metal":
        assert "[[depth(any)]]" in code and "uint sampleMask [[sample_mask]];" in code
        assert "float4 position [[position]];" in code
        assert "float3 pos [[attribute(0)]];" in code
        _compile_metal(tmp_path, code)
    else:
        assert re.search(r"float \w+\s*:\s*SV_DEPTH;", code), code
        assert re.search(r"uint sampleMask\s*:\s*SV_Coverage;", code), code
        assert re.search(r"float4 position\s*:\s*SV_POSITION;", code), code


LOCAL = """shader S {
    fragment {
        in vec2 uv @location(0);
        out vec4 color @location(0);
        void main() {
            color = vec4(uv, 0.0, 1.0);
        }
    }
}
"""


@pytest.mark.parametrize("backend", GRAPHICS)
def test_stage_in_and_out_variables_become_the_entry_interface(tmp_path, backend):
    code = _translate(tmp_path, LOCAL, backend)
    assert "output.color = float4(input.uv, 0.0, 1.0);" in code, code
    assert "return output;" in code
    if backend == "metal":
        # The variables used to be uninitialized locals of a void entry point.
        assert "unused" not in code
        assert (
            "fragment FragmentOutput fragment_main(FragmentInput input [[stage_in]])"
            in code
        )
        _compile_metal(tmp_path, code)
    else:
        assert "FragmentOutput PSMain(FragmentInput input)" in code


GLSL_FRAGMENT = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
float depthBias() { return gl_FragDepth * 0.0; }
void main() {
    color = vec4(uv, gl_FragCoord.x, 1.0);
    gl_FragDepth = gl_FragCoord.z * 0.5;
    gl_FragDepth += depthBias();
    gl_SampleMask[0] = 3;
    gl_SampleMask[0] |= 4;
}
"""


@pytest.mark.parametrize("backend", GRAPHICS)
def test_glsl_fragment_builtin_outputs_reach_the_output_structure(tmp_path, backend):
    code = _translate(tmp_path, GLSL_FRAGMENT, backend, suffix=".frag")
    assert not re.search(r"\bgl_(FragDepth|SampleMask)\b", code), code
    # The sample mask keeps GLSL's int arithmetic and stores its bits as uint.
    assert re.search(r"output\.sampleMask = uint\(3\);", code), code
    assert re.search(
        r"output\.sampleMask = uint\(\(?int\(output\.sampleMask\) \| 4\)?\);", code
    ), code
    # The helper reading the depth output receives the output structure.
    assert re.search(
        r"float depthBias\((inout FragmentOutput|thread FragmentOutput&) output\)", code
    ), code
    if backend == "metal":
        _compile_metal(tmp_path, code)


VERTEX_ARRAYS = """shader S {
    vertex {
        void main() {
            gl_Position = vec4(1.0);
            gl_ClipDistance[0] = 1.0;
            gl_ClipDistance[1] = -1.0;
        }
    }
}
"""


@pytest.mark.parametrize("backend", GRAPHICS)
def test_bare_clip_distances_take_their_largest_constant_index(tmp_path, backend):
    code = _translate(tmp_path, VERTEX_ARRAYS, backend)
    assert not re.search(r"\bgl_ClipDistance\b", code), code
    assert re.search(r"output\.\w+\[1\] = -1\.0;", code), code
    if backend == "metal":
        assert "float clipDistance [[clip_distance]][2];" in code
        _compile_metal(tmp_path, code)
    else:
        assert re.search(r"float clipDistance\[2\]\s*:\s*SV_ClipDistance;", code), code


def test_bare_clip_distance_with_a_variable_index_is_refused(tmp_path):
    source = VERTEX_ARRAYS.replace("gl_ClipDistance[1]", "gl_ClipDistance[gl_VertexID]")
    with pytest.raises(StageInterfaceUnsupportedError, match="not a constant"):
        _translate(tmp_path, source, "metal")


def test_sample_mask_words_beyond_the_first_are_refused(tmp_path):
    source = BARE.replace("gl_SampleMask[0]", "gl_SampleMask[1]")
    with pytest.raises(StageInterfaceUnsupportedError, match="32-bit sample mask"):
        _translate(tmp_path, source, "directx")


FRAG_DATA = """shader S {
    fragment {
        void main() {
            gl_FragData[0] = vec4(1.0);
            gl_FragData[2] = vec4(0.5);
        }
    }
}
"""


@pytest.mark.parametrize(
    "backend, first, third",
    [
        ("metal", r"\[\[color\(0\)\]\]", r"\[\[color\(2\)\]\]"),
        ("directx", r":\s*SV_TARGET0;", r":\s*SV_TARGET2;"),
    ],
)
def test_bare_frag_data_elements_become_colour_attachments(
    tmp_path, backend, first, third
):
    code = _translate(tmp_path, FRAG_DATA, backend)
    assert not re.search(r"\bgl_FragData\b", code), code
    assert re.search(first, code) and re.search(third, code), code
    if backend == "metal":
        _compile_metal(tmp_path, code)


@pytest.mark.parametrize("backend", GRAPHICS)
def test_bare_frag_color_becomes_the_first_colour_attachment(tmp_path, backend):
    source = "shader S { fragment { void main() { gl_FragColor = vec4(1.0); } } }\n"
    code = _translate(tmp_path, source, backend)
    assert not re.search(r"\bgl_FragColor\b", code), code
    assert re.search(r"\[\[color\(0\)\]\]|:\s*SV_TARGET0?;", code), code


SIGNED = """shader S {
    vertex {
        vec4 p @output @gl_Position;
        int layer @output @gl_Layer;
        void main() {
            p = vec4(1.0);
            layer = 2;
            p.x = float(layer);
        }
    }
}
"""


@pytest.mark.parametrize("backend", GRAPHICS)
def test_int_layer_output_is_delivered_as_uint(tmp_path, backend):
    code = _translate(tmp_path, SIGNED, backend)
    assert "output.layer = uint(2);" in code, code
    assert re.search(r"float\(int\(output\.layer\)\)", code), code
    if backend == "metal":
        assert "uint layer [[render_target_array_index]];" in code
        _compile_metal(tmp_path, code)
    else:
        assert re.search(r"uint layer\s*:\s*SV_RenderTargetArrayIndex;", code), code


def test_declared_single_word_sample_mask_becomes_a_scalar(tmp_path):
    source = """shader S {
    out vec4 color @location(0);
    out int mask[1] @gl_SampleMask;
    fragment {
        void main() {
            color = vec4(1.0);
            mask[0] = 1;
        }
    }
}
"""
    code = _translate(tmp_path, source, "metal")
    assert "uint sampleMask [[sample_mask]];" in code, code
    assert "output.sampleMask = uint(1);" in code
    _compile_metal(tmp_path, code)


def test_incrementing_a_uint_builtin_output_is_refused(tmp_path):
    source = SIGNED.replace("layer = 2;", "layer = 2;\n            layer++;")
    with pytest.raises(StageInterfaceUnsupportedError, match="incremented in place"):
        _translate(tmp_path, source, "metal")


def test_bare_stencil_reference_store_uses_the_stencil_output(tmp_path):
    source = """shader S {
    fragment {
        out vec4 color @location(0);
        void main() {
            color = vec4(1.0);
            gl_FragStencilRefARB = 7;
        }
    }
}
"""
    metal = _translate(tmp_path, source, "metal")
    assert re.search(r"uint \w+ \[\[stencil\]\];", metal), metal
    assert re.search(r"output\.\w+ = uint\(7\);", metal), metal
    _compile_metal(tmp_path, metal)
    hlsl = _translate(tmp_path, source, "directx")
    assert re.search(r"uint \w+\s*:\s*SV_StencilRef;", hlsl), hlsl
    assert not re.search(r"\bgl_FragStencilRef(ARB|EXT)\b", metal + hlsl)


def test_glsl_and_spirv_keep_bare_builtin_outputs(tmp_path):
    glsl = _translate(tmp_path, BARE, "opengl")
    assert "gl_FragDepth = 0.5;" in glsl and "gl_SampleMask[0] = 1;" in glsl
    spirv = _translate(tmp_path, BARE, "vulkan")
    assert "BuiltIn FragDepth" in spirv and "BuiltIn SampleMask" in spirv
