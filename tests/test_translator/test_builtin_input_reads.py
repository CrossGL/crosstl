"""Fragment built-in inputs read by name reach HLSL and MSL as entry-point parameters threaded into helpers."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import BuiltinValueUnsupportedError

BUILTIN_READS = """#version 450
layout(location = 0) out vec4 o_color;
vec4 shade(vec2 uv) {
    return vec4(gl_FragCoord.xy * uv, float(gl_SampleID), gl_FrontFacing ? 1.0 : 0.0);
}
vec4 outer() { return shade(vec2(0.5)) + vec4(float(gl_PrimitiveID)); }
void main() { o_color = outer() + vec4(float(gl_SampleID)) + gl_FragCoord; }
"""

POINT_COORD = """#version 450
layout(location = 0) out vec4 o_color;
vec2 sprite() { return gl_PointCoord; }
void main() { o_color = vec4(sprite(), 0.0, 1.0); }
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.frag"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _compile_metal(tmp_path, metal):
    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(metal)
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def test_metal_binds_fragment_builtins_and_threads_them_into_helpers(tmp_path):
    metal = _translate(tmp_path, BUILTIN_READS, "metal")

    assert not re.search(r"\bgl_\w+", metal), metal
    for attribute, type_name in (
        ("position", "float4"),
        ("front_facing", "bool"),
        ("sample_id", "uint"),
        ("primitive_id", "uint"),
    ):
        assert re.search(rf"{type_name} \w+ \[\[{attribute}\]\]", metal), metal
    # shade reads three built-ins; outer reads one and passes on shade's three.
    assert re.search(r"float4 shade\(float2 uv, float4 \w+, bool \w+, int \w+\)", metal)
    assert re.search(r"float4 outer\(float4 \w+, bool \w+, int \w+, int \w+\)", metal)
    assert re.search(r"int (\w+) = int\(\w+_bits\);", metal)
    _compile_metal(tmp_path, metal)


def test_hlsl_binds_fragment_builtins_and_threads_them_into_helpers(tmp_path):
    hlsl = _translate(tmp_path, BUILTIN_READS, "directx")

    assert not re.search(r"\bgl_\w+", hlsl), hlsl
    for semantic, type_name in (
        ("SV_Position", "float4"),
        ("SV_IsFrontFace", "bool"),
        ("SV_SampleIndex", "uint"),
        ("SV_PrimitiveID", "uint"),
    ):
        assert re.search(rf"{type_name} \w+ : {semantic}", hlsl), hlsl
    assert re.search(r"float4 shade\(float2 uv, bool \w+, int \w+, float4 \w+\)", hlsl)


def test_metal_passes_the_point_coordinate_to_a_helper(tmp_path):
    metal = _translate(tmp_path, POINT_COORD, "metal")

    assert re.search(r"float2 (\w+) \[\[point_coord\]\]", metal), metal
    assert re.search(r"float2 sprite\(float2 \w+\)", metal), metal
    _compile_metal(tmp_path, metal)


def test_hlsl_refuses_the_point_coordinate(tmp_path):
    with pytest.raises(BuiltinValueUnsupportedError, match="gl_PointCoord"):
        _translate(tmp_path, POINT_COORD, "directx")


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_targets_with_builtin_globals_keep_reading_them(tmp_path, backend):
    code = _translate(tmp_path, BUILTIN_READS, backend)

    assert "gl_SampleID" in code
    assert "cgl_SampleID" not in code


LAYER_VIEW_SAMPLE_MASK = """#version 460
#extension GL_EXT_multiview : require
layout(location = 0) out vec4 o_color;
float layers() { return float(gl_Layer + gl_ViewportIndex); }
float coverage() { return float(gl_SampleMaskIn[0]); }
void main() {
    o_color = vec4(layers(), coverage(), float(gl_ViewIndex), float(gl_Layer));
}
"""


@pytest.mark.parametrize(
    ("backend", "bindings"),
    [
        (
            "metal",
            (
                r"uint \w+ \[\[render_target_array_index\]\]",
                r"uint \w+ \[\[viewport_array_index\]\]",
                r"uint \w+ \[\[amplification_id\]\]",
                r"uint \w+ \[\[sample_mask\]\]",
            ),
        ),
        (
            "directx",
            (
                r"uint \w+ : SV_RenderTargetArrayIndex",
                r"uint \w+ : SV_ViewportArrayIndex",
                r"uint \w+ : SV_ViewID",
                r"uint \w+ : SV_Coverage",
            ),
        ),
        (
            "slang",
            (
                r"uint \w+ : SV_RenderTargetArrayIndex",
                r"uint \w+ : SV_ViewportArrayIndex",
                r"uint \w+ : SV_ViewID",
                r"uint \w+ : SV_Coverage",
            ),
        ),
    ],
)
def test_layer_viewport_view_and_sample_mask_reads_become_entry_parameters(
    tmp_path, backend, bindings
):
    code = _translate(tmp_path, LAYER_VIEW_SAMPLE_MASK, backend)

    assert not re.search(r"\bgl_\w+", code), code
    for binding in bindings:
        assert len(re.findall(binding, code)) == 1, (binding, code)
    # Each arrives as uint and is GLSL's int once, at the start of the entry point.
    for name in ("Layer", "ViewportIndex", "ViewIndex", "SampleMaskIn"):
        assert re.search(rf"int cgl_{name} = int\(cgl_{name}_bits\);", code), code
    # gl_SampleMaskIn[0] is the one coverage word, a scalar.
    assert re.search(r"float coverage\(int \w+\)", code), code
    assert re.search(r"float layers\(int \w+, int \w+\)", code), code
    assert not re.search(r"cgl_SampleMaskIn\w*\[", code), code
    if backend == "metal":
        _compile_metal(tmp_path, code)


def test_wgsl_reads_the_sample_mask_and_refuses_the_layer(tmp_path):
    source = LAYER_VIEW_SAMPLE_MASK.replace("float(gl_Layer)", "0.0")
    source = source.replace("gl_Layer + gl_ViewportIndex", "0")
    source = source.replace("float(gl_ViewIndex)", "0.0")
    wgsl = _translate(tmp_path, source, "wgsl")
    assert re.search(r"@builtin\(sample_mask\) \w+: u32", wgsl), wgsl
    assert not re.search(r"\bgl_\w+", wgsl), wgsl

    # WGSL has no render-target array index built-in (WGSL section 13.3.1.1).
    with pytest.raises(Exception, match="gl_Layer") as refused:
        _translate(tmp_path, LAYER_VIEW_SAMPLE_MASK, "wgsl")
    assert "wgsl-stage-io-unsupported" in str(refused.value)


@pytest.mark.parametrize("read", ["gl_SampleMaskIn[1]", "gl_SampleMaskIn[index]"])
@pytest.mark.parametrize("backend", ["metal", "directx", "slang", "wgsl"])
def test_a_sample_mask_word_other_than_zero_is_refused(tmp_path, backend, read):
    source = LAYER_VIEW_SAMPLE_MASK.replace(
        "float coverage() { return float(gl_SampleMaskIn[0]); }",
        f"float coverage() {{ int index = 1; return float({read}); }}",
    )
    with pytest.raises(BuiltinValueUnsupportedError, match="gl_SampleMaskIn\\[0\\]"):
        _translate(tmp_path, source.replace("float(gl_Layer)", "0.0"), backend)


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_targets_with_builtin_globals_keep_the_layer_and_sample_mask(
    tmp_path, backend
):
    code = _translate(tmp_path, LAYER_VIEW_SAMPLE_MASK, backend)

    assert "cgl_" not in code
