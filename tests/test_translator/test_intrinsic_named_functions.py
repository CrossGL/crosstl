"""User functions named like texture built-ins, and GLSL's ``stpq`` swizzles, on every graphics target.

GLSL lets a shader overload a built-in such as ``textureProj`` with its own function; CrossTL renames such a user
function and redirects only the calls that resolve to it, so the built-in calls beside it keep their meaning.
GLSL's texture-coordinate swizzle set (``.stpq``) is spelled with ``.xyzw``, which HLSL and MSL also accept.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics import IntrinsicOverloadAmbiguousError
from crosstl.translator.semantics.intrinsic_named_functions import (
    resolve_intrinsic_named_functions,
)
from crosstl.translator.semantics.texture_swizzles import canonicalize_texture_swizzles
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

USER_TEXTURE_PROJ_GLSL = """#version 450
layout(binding = 0) uniform sampler2D shadowMap;
layout(binding = 1) uniform sampler2D colorMap;
layout(location = 0) in vec4 inShadowCoord;
layout(location = 0) out vec4 outFragColor;

float textureProj(vec4 shadowCoord, vec2 off) {
    float dist = texture(shadowMap, shadowCoord.st + off).r;
    return (shadowCoord.w > 0.0 && dist < shadowCoord.z) ? 0.1 : 1.0;
}

void main() {
    vec4 sc = inShadowCoord / inShadowCoord.w;
    float shadow = textureProj(sc, vec2(0.0));
    vec4 projected = textureProj(colorMap, inShadowCoord.xyw);
    outFragColor = projected * shadow;
}
"""

AMBIGUOUS_CGL = """
shader main {
    sampler2D colorMap @binding(0);
    vec4 textureProj(sampler2D map, vec3 coord) {
        return texture(map, coord.xy);
    }
    fragment {
        vec4 main(vec3 uv @location(0)) @location(0) {
            return textureProj(colorMap, uv);
        }
    }
}
"""

STRUCT_MEMBER_ST_CGL = """
shader main {
    struct Pair {
        vec2 st;
    };
    fragment {
        vec4 main(vec4 coord @location(0)) @location(0) {
            Pair pair;
            pair.st = coord.st;
            return vec4(pair.st, coord.pq);
        }
    }
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _parse_cgl(source):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(source)


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_glsl(code):
    if shutil.which("glslangValidator") is None:
        return
    _run(["glslangValidator", "--stdin", "-S", "frag"], input=code)


def _validate_spirv(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.0", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


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
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")]
    )


def test_user_overload_is_renamed_and_builtin_calls_keep_their_meaning(tmp_path):
    glsl = _translate(tmp_path, USER_TEXTURE_PROJ_GLSL, ".frag", "opengl")
    assert "float textureProj_fn(vec4 shadowCoord, vec2 off)" in glsl
    assert "textureProj_fn(sc, vec2(0.0))" in glsl
    assert "textureProj(colorMap, inShadowCoord.xyw)" in glsl
    _validate_glsl(glsl)


def test_directx_samples_through_the_renamed_user_overload(tmp_path):
    hlsl = _translate(tmp_path, USER_TEXTURE_PROJ_GLSL, ".frag", "directx")
    assert "float textureProj_fn(float4 shadowCoord, float2 off" in hlsl
    assert "textureProj_fn(sc, float2(0.0, 0.0))" in hlsl
    # The built-in projected sample divides by q; the user overload samples its coordinate as given.
    assert "colorMap.Sample(" in hlsl
    assert "shadowCoord.st" not in hlsl and "shadowCoord.xy + off" in hlsl


def test_metal_and_spirv_compile_the_renamed_user_overload(tmp_path):
    metal = _translate(tmp_path, USER_TEXTURE_PROJ_GLSL, ".frag", "metal")
    assert "textureProj_fn(" in metal
    assert ".st" not in metal
    _validate_metal(tmp_path, metal)
    spirv = _translate(tmp_path, USER_TEXTURE_PROJ_GLSL, ".frag", "vulkan")
    # One call to the user function; the user function's sample and the built-in projected sample.
    assert spirv.count("OpFunctionCall") == 1
    assert spirv.count("OpImageSample") == 2
    _validate_spirv(tmp_path, spirv)


def test_user_overload_taking_a_resource_first_is_refused_when_ambiguous(tmp_path):
    with pytest.raises(IntrinsicOverloadAmbiguousError) as raised:
        _translate(tmp_path, AMBIGUOUS_CGL, ".cgl", "directx")
    assert isinstance(raised.value, TranslationDiagnostic)
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.intrinsic-overload-ambiguous"
    )


def test_fresh_name_avoids_every_identifier_in_the_program():
    program = _parse_cgl("""
    shader main {
        float textureProj_fn;
        float textureProj(vec4 coord, vec2 off) {
            return coord.x + off.y + textureProj_fn;
        }
        fragment {
            vec4 main(vec4 coord @location(0)) @location(0) {
                return vec4(textureProj(coord, coord.xy));
            }
        }
    }
    """)
    assert resolve_intrinsic_named_functions(program) == {
        "textureProj": "textureProj_fn2"
    }
    assert [function.name for function in program.functions] == ["textureProj_fn2"]


def test_programs_without_such_functions_are_unchanged():
    program = _parse_cgl("""
    shader main {
        sampler2D colorMap @binding(0);
        fragment {
            vec4 main(vec4 coord @location(0)) @location(0) {
                return textureProj(colorMap, coord);
            }
        }
    }
    """)
    assert resolve_intrinsic_named_functions(program) == {}


def test_texture_swizzles_keep_structure_members_of_the_same_name(tmp_path):
    program = _parse_cgl(STRUCT_MEMBER_ST_CGL)
    assert canonicalize_texture_swizzles(program) == 2
    for backend, suffix in (("directx", "hlsl"), ("metal", "metal")):
        code = _translate(tmp_path, STRUCT_MEMBER_ST_CGL, ".cgl", backend)
        assert "pair.st = coord.xy" in code, suffix
        assert "coord.zw" in code, suffix
    _validate_metal(
        tmp_path, _translate(tmp_path, STRUCT_MEMBER_ST_CGL, ".cgl", "metal")
    )
