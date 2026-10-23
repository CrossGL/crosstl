"""Texture reads that report tile residency (CrossGL ``textureResidency``/``textureResident``).

The GLSL importer maps the GL_ARB_sparse_texture2 functions onto the CrossGL wrapper around an ordinary read, the
MSL emitter lowers it to the ``sparse_*`` texture members and the desktop GLSL emitter back to the ARB functions,
and every other target refuses it with a typed diagnostic instead of printing the CrossGL call.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.GLSL.sparse_textures import SPARSE_READS
from crosstl.translator.codegen.texture_residency import (
    GLSL_SPARSE_FUNCTIONS,
    TEXTURE_RESIDENCY_READS,
    TextureResidencyUnsupportedError,
)

CROSSGL_FRAGMENT = """
shader Residency {
    struct FragmentInput {
        vec2 uv;
    }

    struct FragmentOutput {
        vec4 color;
    }

    fragment {
        uniform sampler2D colorMap;
        uniform sampler2DShadow shadowMap;
        uniform isampler2DArray layers;

        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            vec4 a;
            vec4 b;
            vec4 c;
            vec4 d;
            vec4 e;
            float s;
            ivec4 l;
            int code = textureResidency(texture(colorMap, input.uv), a);
            code |= textureResidency(texture(colorMap, input.uv, 0.5), b);
            code |= textureResidency(texelFetch(colorMap, ivec2(input.uv), 0), c);
            code |= textureResidency(textureGather(colorMap, input.uv, 1), d);
            code |= textureResidency(textureGradOffset(colorMap, input.uv, vec2(0.1), vec2(0.2), ivec2(1, 0)), e);
            code |= textureResidency(textureCompare(shadowMap, input.uv, 0.5), s);
            code |= textureResidency(textureLod(layers, vec3(input.uv, 1.0), 0.0), l);
            output.color = textureResident(code) ? a + b + c + d + e + vec4(s) + vec4(l) : vec4(0.0);
            return output;
        }
    }
}
"""

GLSL_FRAGMENT = """#version 450
#extension GL_ARB_sparse_texture2 : require
#extension GL_ARB_sparse_texture_clamp : require
layout(binding = 0) uniform sampler2D tex;
layout(binding = 1) uniform sampler2DShadow shadowTex;
layout(binding = 2) uniform samplerCubeArrayShadow cubeShadow;
layout(binding = 3) uniform isampler2DArray itex;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 a, b, c, d, e, f, g, h, k, m;
    float s1, s2;
    ivec4 iv;
    int r = sparseTextureARB(tex, uv, m);
    r |= sparseTextureLodARB(tex, uv, 1.0, a);
    r |= sparseTextureOffsetARB(tex, uv, ivec2(1, 0), b, 0.25);
    r |= sparseTextureGradARB(tex, uv, vec2(0.1), vec2(0.1), c);
    r |= sparseTexelFetchOffsetARB(tex, ivec2(3), 0, ivec2(1), d);
    r |= sparseTextureGatherOffsetARB(tex, uv, ivec2(1), e, 2);
    r |= sparseTextureClampARB(tex, uv, 1.0, f, 0.5);
    r |= sparseTextureGradOffsetClampARB(tex, uv, vec2(0.1), vec2(0.1), ivec2(1), 2.0, g);
    r |= sparseTextureARB(shadowTex, vec3(uv, 0.5), s1);
    r |= sparseTextureGatherARB(shadowTex, uv, 0.5, h);
    r |= sparseTextureARB(cubeShadow, vec4(uv, 1.0, 0.0), 0.5, s2);
    r |= sparseTextureLodARB(itex, vec3(uv, 1.0), 0.0, iv);
    r |= sparseTextureLodOffsetARB(tex, uv, 0.0, ivec2(-1), k);
    bool resident = sparseTexelsResidentARB(r);
    outColor = resident ? a + b + c + d + e + f + g + h + k + m + vec4(s1 + s2) + vec4(iv) : vec4(0.0);
}
"""

VULKAN_GLSL_FRAGMENT = """#version 450
#extension GL_ARB_sparse_texture2 : require
layout(set = 0, binding = 0) uniform texture2D colorTexture;
layout(set = 0, binding = 1) uniform sampler colorSampler;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 texel = vec4(0.0);
    int code = sparseTextureARB(sampler2D(colorTexture, colorSampler), uv, texel);
    outColor = sparseTexelsResidentARB(code) ? texel : vec4(1.0, 0.0, 0.0, 1.0);
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _has_metal_compiler():
    return (
        platform.system() == "Darwin"
        and shutil.which("xcrun") is not None
        and subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode
        == 0
    )


def _validate(tmp_path, backend, code):
    """Compile ``code`` with the target's native checker when it is installed."""
    if backend == "opengl" and shutil.which("glslangValidator"):
        path = tmp_path / "out.frag"
        path.write_text(code)
        _run(["glslangValidator", "-S", "frag", str(path)])
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        path = tmp_path / "out.spvasm"
        path.write_text(code)
        binary = tmp_path / "out.spv"
        _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
        _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    elif backend == "metal" and _has_metal_compiler():
        path = tmp_path / "out.metal"
        path.write_text(code)
        # sparse_color and the sparse_* texture members need MSL 2.3.
        _run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-std=macos-metal2.3",
                "-c",
                str(path),
                "-o",
                str(tmp_path / "out.air"),
            ]
        )


def _compact(code):
    return " ".join(code.split())


def test_crossgl_residency_lowers_to_msl_sparse_members(tmp_path):
    code = _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", "metal")
    flat = _compact(code)

    assert "sparse_color<T> result, thread U& texel" in flat.replace(" &", "& ")
    assert "return result.resident() ? 0 : 1;" in flat
    assert "int code = crosstl_texture_residency(colorMap.sparse_sample(" in flat
    assert "input.uv, bias(0.5)), b)" in flat
    assert "colorMap.sparse_read(uint2(int2(input.uv)), uint(0)), c)" in flat
    assert "colorMap.sparse_gather(" in flat and "component::y), d)" in flat
    assert "gradient2d(float2(0.1), float2(0.2)), int2(1, 0)), e)" in flat
    assert "shadowMap.sparse_sample_compare(" in flat
    assert "layers.sparse_sample(" in flat and "level(0.0)), l)" in flat
    assert "(code == 0) ?" in flat
    assert "textureResidency" not in code and "textureResident" not in code
    _validate(tmp_path, "metal", code)


def test_crossgl_residency_lowers_to_arb_sparse_texture2(tmp_path):
    code = _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", "opengl")
    flat = _compact(code)

    assert "#extension GL_ARB_sparse_texture2 : require" in code
    assert "int code = sparseTextureARB(colorMap, uv, a);" in flat
    # The texel precedes an optional bias or gather component (GL_ARB_sparse_texture2, section 8.9).
    assert "sparseTextureARB(colorMap, uv, b, 0.5)" in flat
    assert "sparseTexelFetchARB(colorMap, ivec2(uv), 0, c)" in flat
    assert "sparseTextureGatherARB(colorMap, uv, d, 1)" in flat
    assert (
        "sparseTextureGradOffsetARB(colorMap, uv, vec2(0.1), vec2(0.2), ivec2(1, 0), e)"
        in flat
    )
    assert "sparseTextureARB(shadowMap, vec3(uv, 0.5), s)" in flat
    assert "sparseTextureLodARB(layers, vec3(uv, 1.0), 0.0, l)" in flat
    assert "sparseTexelsResidentARB(code)" in flat
    _validate(tmp_path, "opengl", code)


@pytest.mark.parametrize(
    ("backend", "reason"),
    [
        ("webgl", "GL_ARB_sparse_texture2 is a desktop OpenGL extension"),
        ("wgsl", "WGSL has no sparse textures"),
        ("cuda", "CUDA texture fetches report no tile residency"),
        ("hip", "HIP texture fetches report no tile residency"),
        ("slang", "no lowering for texture residency feedback"),
        ("rust", "no lowering for texture residency feedback"),
        ("mojo", "no lowering for texture residency feedback"),
    ],
)
def test_targets_without_a_residency_lowering_refuse_it(tmp_path, backend, reason):
    with pytest.raises(TextureResidencyUnsupportedError) as error:
        _translate(tmp_path, CROSSGL_FRAGMENT, ".cgl", backend)

    assert error.value.operation == "textureResidency"
    assert reason in str(error.value)
    assert error.value.code.endswith(f"{backend}-texture-residency-unsupported")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("textureResidency(a, b)", "must be a direct call to a texture read"),
        (
            "textureResidency(textureProj(colorMap, vec3(input.uv, 1.0)), b)",
            "got textureProj",
        ),
        ("textureResidency(texture(colorMap, input.uv))", "got 1 operand(s)"),
    ],
)
def test_malformed_residency_queries_are_refused(tmp_path, replacement, message):
    source = CROSSGL_FRAGMENT.replace(
        "textureResidency(texture(colorMap, input.uv, 0.5), b)", replacement
    )
    for backend in ("metal", "opengl", "directx"):
        with pytest.raises(TextureResidencyUnsupportedError, match=re.escape(message)):
            _translate(tmp_path, source, ".cgl", backend)


def test_resident_test_takes_one_code(tmp_path):
    source = CROSSGL_FRAGMENT.replace(
        "textureResident(code)", "textureResident(code, 1)"
    )
    for backend in ("metal", "opengl", "directx"):
        with pytest.raises(
            TextureResidencyUnsupportedError, match="one residency code"
        ):
            _translate(tmp_path, source, ".cgl", backend)


def test_msl_refuses_a_read_without_a_sparse_member(tmp_path):
    # MSL has no four-offset gather; the read lowers to four gathers, which no sparse member reports on.
    source = CROSSGL_FRAGMENT.replace(
        "textureGather(colorMap, input.uv, 1)",
        "textureGatherOffsets(colorMap, input.uv, ivec2(0), ivec2(1), ivec2(2), ivec2(3))",
    )
    with pytest.raises(
        TextureResidencyUnsupportedError, match="not to a texture member"
    ):
        _translate(tmp_path, source, ".cgl", "metal")


def test_glsl_sparse_functions_import_to_crossgl_residency_queries(tmp_path):
    code = _translate(tmp_path, GLSL_FRAGMENT, ".frag", "cgl")
    flat = _compact(code)

    expected = [
        "int r = textureResidency(texture(tex, input.uv), m);",
        "r |= textureResidency(textureLod(tex, input.uv, 1.0), a);",
        "r |= textureResidency(textureOffset(tex, input.uv, ivec2(1, 0), 0.25), b);",
        "r |= textureResidency(textureGrad(tex, input.uv, vec2(0.1), vec2(0.1)), c);",
        "r |= textureResidency(texelFetchOffset(tex, ivec2(3), 0, ivec2(1)), d);",
        "r |= textureResidency(textureGatherOffset(tex, input.uv, ivec2(1), 2), e);",
        "r |= textureResidency(textureClamp(tex, input.uv, 1.0, 0.5), f);",
        "r |= textureResidency(textureGradOffsetClamp(tex, input.uv, vec2(0.1), vec2(0.1), ivec2(1), 2.0), g);",
        "r |= textureResidency(textureCompare(shadowTex, input.uv, 0.5), s1);",
        "r |= textureResidency(textureGatherCompare(shadowTex, input.uv, 0.5), h);",
        "r |= textureResidency(textureCompare(cubeShadow, vec4(input.uv, 1.0, 0.0), 0.5), s2);",
        "r |= textureResidency(textureLod(itex, vec3(input.uv, 1.0), 0.0), iv);",
        "bool resident = textureResident(r);",
    ]
    for line in expected:
        assert line in flat
    assert "ARB(" not in code


@pytest.mark.parametrize("backend", ["metal", "opengl", "vulkan"])
def test_glsl_sparse_functions_translate_to_valid_targets(tmp_path, backend):
    code = _translate(tmp_path, GLSL_FRAGMENT, ".frag", backend)

    assert "textureResidency" not in code and "textureResident(" not in code
    _validate(tmp_path, backend, code)


def test_glsl_sparse_function_round_trips_its_operand_order(tmp_path):
    code = _compact(_translate(tmp_path, GLSL_FRAGMENT, ".frag", "opengl"))

    assert "sparseTextureOffsetARB(tex, uv, ivec2(1, 0), b, 0.25)" in code
    assert "sparseTextureGatherOffsetARB(tex, uv, ivec2(1), e, 2)" in code
    assert "sparseTextureClampARB(tex, uv, 1.0, f, 0.5)" in code
    assert "sparseTextureGatherARB(shadowTex, uv, 0.5, h)" in code
    assert "sparseTextureARB(cubeShadow, vec4(uv, 1.0, 0.0), 0.5, s2)" in code
    assert code.count("#extension GL_ARB_sparse_texture2") == 1


def test_vulkan_glsl_separate_sampler_reads_report_residency_in_msl(tmp_path):
    cgl = _compact(_translate(tmp_path, VULKAN_GLSL_FRAGMENT, ".frag", "cgl"))
    assert (
        "textureResidency(texture(colorTexture, colorSampler, input.uv), texel)" in cgl
    )

    code = _translate(tmp_path, VULKAN_GLSL_FRAGMENT, ".frag", "metal")
    assert "colorTexture.sparse_sample(colorSampler, input.uv)" in _compact(code)
    _validate(tmp_path, "metal", code)


def test_glsl_es_profile_refuses_sparse_functions(tmp_path):
    source = GLSL_FRAGMENT.replace(
        "#version 450", "#version 310 es\nprecision highp float;"
    )
    with pytest.raises(Exception, match="does not exist in GLSL ES"):
        _translate(tmp_path, source, ".frag", "opengl")


def test_glsl_sparse_function_with_no_matching_overload_is_refused(tmp_path):
    source = GLSL_FRAGMENT.replace(
        "sparseTextureLodARB(tex, uv, 1.0, a)", "sparseTextureOffsetARB(tex, uv, a)"
    )
    with pytest.raises(
        TextureResidencyUnsupportedError, match="no sparseTextureOffsetARB"
    ):
        _translate(tmp_path, source, ".frag", "cgl")


def test_sparse_tables_cover_the_same_reads():
    # Every GLSL sparse function imports to a read the emitters accept and renders back to the same function.
    for function, (read, _before) in SPARSE_READS.items():
        assert GLSL_SPARSE_FUNCTIONS[read][0] == function
        crossgl_read = {
            "textureClampARB": "textureClamp",
            "textureOffsetClampARB": "textureOffsetClamp",
            "textureGradClampARB": "textureGradClamp",
            "textureGradOffsetClampARB": "textureGradOffsetClamp",
        }.get(read, read)
        assert crossgl_read in TEXTURE_RESIDENCY_READS
