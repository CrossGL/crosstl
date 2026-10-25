"""GLSL ES 3.00 spellings of desktop-GLSL forms, and the built-ins an OpenGL or WebGL profile lacks.

ESSL 3.00 has no implicit conversions (section 4.1.10), no scalar swizzles, ``length()`` on arrays only and
constant initializers for ``const`` locals (section 4.3.3). Built-ins of other profiles (Vulkan GLSL's view index,
GLSL 4.x clip distances or barycentrics) are refused with a typed diagnostic.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import (
    OpenGLLoweringRefused,
    WebGLLoweringRefused,
)
from crosstl.translator.codegen.texture_residency import (
    TextureResidencyUnsupportedError,
)
from crosstl.translator.semantics import DebugPrintfUnsupportedError

ES_FORMS_CGL = """
shader main {
    cbuffer Params @push_constant {
        uint mode;
        float scale;
    };
    fragment {
        vec4 color @input @location(0);
        vec4 outColor @output @location(0);
        void main() {
            const float halfAlpha = 0.5 * color.w;
            const int taps = 4;
            const vec2 offset = vec2(1.0, float(taps));
            float level = color.x;
            vec4 lights[3];
            lights[0] = color;
            vec4 sum = vec4(0.0);
            for (int i = 0; i < lights[0].length(); i++) {
                sum += color / 32 + vec4(level / 3);
            }
            if (mode == 1) {
                sum.rgb = level.rrr;
                sum.a = level.x + halfAlpha + offset.y;
            }
            outColor = sum * 2 + vec4(scale);
        }
    }
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", "-S", stage, str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_webgl_spells_literal_conversions_swizzles_lengths_and_constants(tmp_path):
    code = _translate(tmp_path, ES_FORMS_CGL, ".cgl", "webgl")

    assert "/ 32.0)" in code and "(level / 3.0)" in code and "* 2.0)" in code
    assert re.search(r"\(mode == 1u\)", code)
    assert "vec3(level)" in code and ".rrr" not in code
    assert re.search(r"\(level \+ halfAlpha\)", code) and "level.x" not in code
    assert re.search(r"i < 4\)", code) and ".length()" not in code
    # Only the local whose initializer reads an input loses ``const``.
    assert re.search(r"(?<!const )float halfAlpha = ", code)
    assert "const int taps = 4;" in code
    assert "const vec2 offset = vec2(1.0, float(taps));" in code
    _validate(tmp_path, code, "frag")


def test_opengl_keeps_desktop_spellings(tmp_path):
    code = _translate(tmp_path, ES_FORMS_CGL, ".cgl", "opengl")

    assert "/ 32)" in code and "const float halfAlpha" in code
    _validate(tmp_path, code, "frag")


ARRAY_LENGTH_CGL = """
shader main {
    fragment {
        vec4 outColor @output @location(0);
        void main() {
            float weights[3];
            weights[0] = 1.0;
            weights[1] = 2.0;
            weights[2] = 3.0;
            float total = 0.0;
            for (int i = 0; i < weights.length(); i++) { total += weights[i]; }
            outColor = vec4(total);
        }
    }
}
"""


def test_webgl_keeps_array_length(tmp_path):
    code = _translate(tmp_path, ARRAY_LENGTH_CGL, ".cgl", "webgl")

    assert "weights.length()" in code
    _validate(tmp_path, code, "frag")


PRINTF_HLSL = """
float4 main(float4 pos : POSITION) : SV_Position {
    printf("Position = %v4f", pos);
    return pos;
}
"""

VIEW_INDEX_HLSL = """
cbuffer UBO : register(b0) { float4x4 views[2]; };
float4 main(float4 pos : POSITION, uint view : SV_ViewID) : SV_Position {
    return mul(views[view], pos);
}
"""


def test_hlsl_printf_is_glsl_debug_printf(tmp_path):
    code = _translate(tmp_path, PRINTF_HLSL, ".hlsl", "opengl")

    assert "#extension GL_EXT_debug_printf : require" in code
    assert 'debugPrintfEXT("Position = %v4f", pos);' in code
    assert not re.search(r"(?<!debugP)printf\(", code)
    _validate(tmp_path, code, "vert")


def test_hlsl_printf_is_refused_for_webgl(tmp_path):
    # GLSL ES 3.00 section 3.5: "#extension ... : require" is an error when the extension is unsupported, and no
    # WebGL 2 implementation exposes GL_EXT_debug_printf.
    with pytest.raises(DebugPrintfUnsupportedError, match="WebGL 2"):
        _translate(tmp_path, PRINTF_HLSL, ".hlsl", "webgl")


@pytest.mark.parametrize(
    "backend, error",
    [("opengl", OpenGLLoweringRefused), ("webgl", WebGLLoweringRefused)],
)
def test_view_index_is_refused_without_a_view_count(tmp_path, backend, error):
    with pytest.raises(error, match="gl_ViewIndex.*num_views"):
        _translate(tmp_path, VIEW_INDEX_HLSL, ".hlsl", backend)


def test_hlsl_view_id_imports_as_the_crossgl_view_index(tmp_path):
    crossgl = _translate(tmp_path, VIEW_INDEX_HLSL, ".hlsl", "cgl")
    hlsl = _translate(tmp_path, VIEW_INDEX_HLSL, ".hlsl", "directx")

    assert "@ gl_ViewIndex" in crossgl and "gl_ViewID" not in crossgl
    assert "SV_ViewID" in hlsl


ES_ONLY_MISSING_GLSL = {
    "gl_BaryCoordEXT": (
        """#version 450
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) out vec4 color;
void main() { color = vec4(gl_BaryCoordEXT, 1.0); }
"""
    ),
    "gl_PrimitiveID": (
        """#version 450
layout(location = 0) out vec4 color;
void main() { color = vec4(float(gl_PrimitiveID)); }
"""
    ),
}


@pytest.mark.parametrize("builtin", sorted(ES_ONLY_MISSING_GLSL))
def test_webgl_refuses_builtins_glsl_es_3_00_lacks(tmp_path, builtin):
    source = ES_ONLY_MISSING_GLSL[builtin]
    with pytest.raises(WebGLLoweringRefused, match=builtin):
        _translate(tmp_path, source, ".frag", "webgl")


SPARSE_TEXTURE_GLSL = """#version 450
#extension GL_ARB_sparse_texture2 : enable
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    vec4 texel = vec4(0.0);
    int code = sparseTextureARB(tex, uv, texel);
    color = texel + vec4(float(code));
}
"""


def test_webgl_refuses_sparse_texture_reads(tmp_path):
    # sparseTextureARB imports as a CrossGL residency query, which GLSL ES 3.00 has no form of.
    with pytest.raises(
        TextureResidencyUnsupportedError, match="GLSL ES 3.00 has no sparse textures"
    ) as raised:
        _translate(tmp_path, SPARSE_TEXTURE_GLSL, ".frag", "webgl")
    assert raised.value.project_diagnostic_code == (
        "project.translate.webgl-texture-residency-unsupported"
    )


SCALAR_CLIP_HLSL = """
cbuffer UBO : register(b0) { float4 plane; };
struct VSOutput {
    float4 Pos : SV_Position;
    float Clip : SV_ClipDistance0;
};
VSOutput main(float4 pos : POSITION) {
    VSOutput output = (VSOutput)0;
    output.Pos = pos;
    output.Clip = dot(pos, plane);
    return output;
}
"""


def test_scalar_hlsl_clip_distance_writes_element_zero(tmp_path):
    code = _translate(tmp_path, SCALAR_CLIP_HLSL, ".hlsl", "opengl")

    assert re.search(r"gl_ClipDistance\[0\] = dot\(", code)
    assert not re.search(r"gl_ClipDistance = ", code)
    assert "out float gl_ClipDistance" not in code
    _validate(tmp_path, code, "vert")


def test_vector_hlsl_clip_distance_fills_consecutive_elements(tmp_path):
    source = SCALAR_CLIP_HLSL.replace("float Clip :", "float2 Clip :").replace(
        "dot(pos, plane)", "pos.xy"
    )
    code = _translate(tmp_path, source, ".hlsl", "opengl")

    assert re.search(r"gl_ClipDistance\[0\] = \w+\[0\];", code)
    assert re.search(r"gl_ClipDistance\[1\] = \w+\[1\];", code)
    _validate(tmp_path, code, "vert")


VIEW_INDEX_GLSL = """#version 450
#extension GL_EXT_multiview : enable
layout(binding = 0) uniform UBO { mat4 views[2]; } ubo;
layout(location = 0) in vec4 pos;
void main() { gl_Position = ubo.views[gl_ViewIndex] * pos; }
"""


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _validate_metal(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_view_id_is_the_spirv_view_index_builtin(tmp_path):
    code = _translate(tmp_path, VIEW_INDEX_HLSL, ".hlsl", "vulkan")

    assert "OpCapability MultiView" in code
    assert 'OpExtension "SPV_KHR_multiview"' in code
    assert re.search(r"OpDecorate %\w+ BuiltIn ViewIndex", code)
    assert not re.search(r"OpDecorate %\w*view\w* Location", code)
    _validate_spirv(tmp_path, code)


def test_hlsl_view_id_is_the_metal_amplification_id(tmp_path):
    code = _translate(tmp_path, VIEW_INDEX_HLSL, ".hlsl", "metal")

    assert re.search(r"uint view \[\[amplification_id\]\]", code)
    assert "[[attribute(1)]]" not in code
    _validate_metal(tmp_path, code)


def test_glsl_view_index_reads_become_stage_parameters(tmp_path):
    metal = _translate(tmp_path, VIEW_INDEX_GLSL, ".vert", "metal")
    hlsl = _translate(tmp_path, VIEW_INDEX_GLSL, ".vert", "directx")

    # GLSL's view index is an int; Metal's amplification_id is a uint.
    assert re.search(r"uint _crossglViewIndex \[\[amplification_id\]\]", metal)
    assert "views[int(_crossglViewIndex)]" in metal
    assert re.search(r"uint gl_ViewIndex : SV_ViewID", hlsl)
    _validate_metal(tmp_path, metal)
