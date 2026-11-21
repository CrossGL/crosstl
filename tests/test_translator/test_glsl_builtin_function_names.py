"""GLSL and WebGL output rename program identifiers that spell a GLSL built-in function.

A global named ``texture`` hides the built-in in desktop GLSL, so ``texture(texture, uv)`` fails with "can't use
function syntax on variable", and GLSL ES 3.00 rejects the declaration itself as a redefinition.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_builtin_functions import (
    GLSL_BUILTIN_FUNCTION_NAMES,
    GLSL_CORE_BUILTIN_FUNCTIONS,
)

TARGETS = ("opengl", "webgl")

SAMPLED_TEXTURE = """shader main {
    texture2D texture @set(0) @binding(2);
    sampler samplerTexture @set(0) @binding(3);
    fragment {
        vec4 main(vec2 uv @ TexCoord0) @ gl_FragColor {
            return texture(texture, samplerTexture, uv);
        }
    }
}
"""

BLOCK_MEMBERS_AND_LOCALS = """shader main {
    texture2D image @set(0) @binding(2);
    sampler imageSampler @set(0) @binding(3);
    cbuffer Params { float mix; vec3 normalize; };
    fragment {
        vec4 main(vec2 uv @ TexCoord0) @ gl_FragColor {
            float length = 2.0;
            vec4 c = texture(image, imageSampler, uv);
            float m = mix * length;
            return vec4(mix(c.rgb, normalize, m), length(c));
        }
    }
}
"""

USER_FUNCTION = """shader main {
    fragment {
        float mix(float a, float b) { return a * 0.25 + b * 0.75; }
        vec4 main(vec2 uv @ TexCoord0) @ gl_FragColor {
            float distance = 0.5;
            return vec4(uv, mix(uv.x, uv.y), clamp(distance, 0.0, 1.0));
        }
    }
}
"""

STAGE_INTERFACE = """shader main {
    struct VSIn { vec3 normalize @ POSITION; vec2 distance @ TEXCOORD0; };
    struct VSOut { vec4 pos @ gl_Position; vec2 length @ TEXCOORD0; };
    cbuffer Frame { mat4 transpose; };
    vertex {
        VSOut main(VSIn input) {
            VSOut o;
            o.pos = transpose * vec4(normalize(input.normalize), 1.0);
            o.length = input.distance * length(input.normalize);
            return o;
        }
    }
    fragment {
        vec4 main(VSOut input) @ gl_FragColor {
            return vec4(input.length, 0.0, 1.0);
        }
    }
}
"""

COMPUTE = """shader main {
    struct step { float min; float max; };
    const float sign = -1.0;
    buffer float* round;
    buffer step* clamp;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        float floor(float exp, float pow) { return exp * pow + sign; }
        void main() {
            uint dot = gl_GlobalInvocationID.x;
            step s = clamp[dot];
            round[dot] = floor(s.min, s.max) + abs(round[dot]) + max(s.min, 0.0);
        }
    }
}
"""

HLSL_TEXTURE = """Texture2D texture : register(t0);
SamplerState samplerTexture : register(s0);
cbuffer C : register(b0) { float mix; float step; };
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    float distance = lerp(mix, step, uv.x);
    return texture.Sample(samplerTexture, uv) * distance + length(uv);
}
"""


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _assert_glslang_accepts(tmp_path, glsl, *stages):
    # glslang defines GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, ... for the stage it compiles.
    if not shutil.which("glslangValidator"):
        return
    (tmp_path / "out.glsl").write_text(glsl)
    for stage in stages:
        result = subprocess.run(
            ["glslangValidator", "-S", stage, "out.glsl"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, stage + "\n" + result.stdout + result.stderr


def _identifier_tokens(glsl):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", glsl))


@pytest.mark.parametrize("backend", TARGETS)
def test_texture_named_texture_is_renamed_and_sampled(tmp_path, backend):
    glsl = _translate(tmp_path, SAMPLED_TEXTURE, backend)
    assert "uniform sampler2D texture_;" in glsl
    assert "texture(texture_, uv)" in glsl
    assert "sampler2D texture;" not in glsl
    _assert_glslang_accepts(tmp_path, glsl, "frag")


@pytest.mark.parametrize("backend", TARGETS)
def test_block_members_and_locals_keep_the_builtins_callable(tmp_path, backend):
    glsl = _translate(tmp_path, BLOCK_MEMBERS_AND_LOCALS, backend)
    assert "float mix_;" in glsl and "vec3 normalize_;" in glsl
    assert "float length_ = 2.0;" in glsl
    assert "mix(c.rgb, normalize_, m)" in glsl and "length(c)" in glsl
    _assert_glslang_accepts(tmp_path, glsl, "frag")


@pytest.mark.parametrize("backend", TARGETS)
def test_user_function_named_like_a_builtin_is_renamed_with_its_calls(
    tmp_path, backend
):
    # GLSL ES 3.00 rejects any user function named like a built-in ("redeclaration of existing name").
    glsl = _translate(tmp_path, USER_FUNCTION, backend)
    assert "float mix_(float a, float b)" in glsl
    assert "mix_(uv.x, uv.y)" in glsl
    assert "float distance_ = 0.5;" in glsl and "clamp(distance_, 0.0, 1.0)" in glsl
    assert "mix(" not in glsl.replace("mix_(", "")
    _assert_glslang_accepts(tmp_path, glsl, "frag")


@pytest.mark.parametrize("backend", TARGETS)
def test_stage_interface_members_are_declared_and_read_with_one_name(tmp_path, backend):
    glsl = _translate(tmp_path, STAGE_INTERFACE, backend)
    assert "in vec3 normalize_;" in glsl and "in vec2 distance_;" in glsl
    assert "normalize(normalize_)" in glsl
    assert "length_ = (distance_ * length(normalize_));" in glsl
    assert "mat4 transpose_;" in glsl and "(transpose_ * " in glsl
    _assert_glslang_accepts(tmp_path, glsl, "vert", "frag")


def test_compute_program_names_do_not_hide_builtins(tmp_path):
    glsl = _translate(tmp_path, COMPUTE, "opengl")
    tokens = _identifier_tokens(glsl)
    for renamed in ("sign_", "step_", "round_", "clamp_", "floor_", "dot_"):
        assert renamed in tokens, renamed
    assert "struct step_ {" in glsl
    assert "float floor_(float exp_, float pow_)" in glsl
    assert "(floor_(s.min_, s.max_) + abs(round_[dot_]))" in glsl
    assert "max(s.min_, 0.0)" in glsl
    _assert_glslang_accepts(tmp_path, glsl, "comp")


@pytest.mark.parametrize("backend", TARGETS)
def test_hlsl_texture_named_texture(tmp_path, backend):
    glsl = _translate(tmp_path, HLSL_TEXTURE, backend, suffix=".hlsl")
    assert "uniform sampler2D texture_;" in glsl
    assert "texture(texture_, uv)" in glsl
    assert "float mix_;" in glsl and "float step_;" in glsl
    assert "mix(mix_, step_, uv.x)" in glsl
    _assert_glslang_accepts(tmp_path, glsl, "frag")


def test_core_names_cover_the_pinned_specification():
    upstream = pytest.importorskip("tools.coverage.upstream")
    glsl_features = pytest.importorskip("tools.coverage.features.glsl")
    subgroup_features = pytest.importorskip("tools.coverage.features.glsl_subgroup")
    try:
        core = set(glsl_features.overloads(upstream.fetch("glsl", offline=True)))
        subgroup = set(
            subgroup_features.functions(
                *(
                    upstream.fetch(name, offline=True)
                    for name in (
                        "glsl-subgroup",
                        "glsl-subgroup-rotate",
                        "glsl-quad-control",
                        "glsl-subgroup-partitioned",
                    )
                )
            )
        )
    except Exception as error:  # the pinned sources are not cached offline
        pytest.skip(f"pinned GLSL sources unavailable: {error}")
    assert core <= GLSL_CORE_BUILTIN_FUNCTIONS
    assert subgroup <= GLSL_BUILTIN_FUNCTION_NAMES
