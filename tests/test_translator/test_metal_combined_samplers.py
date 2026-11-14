"""Metal binds a sampler for each combined sampler a program samples, instead of an inline sampler state."""

import re
import shutil
import subprocess

import crosstl


def _translate(tmp_path, source, suffix=".frag"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
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


SAMPLED = """#version 450
layout(set = 0, binding = 1) uniform sampler2D albedo;
layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 3) uniform sampler2D layers[4];
layout(set = 0, binding = 7) uniform sampler2D fetchOnly;
layout(location = 0) in vec2 uv;
layout(location = 1) flat in int layer;
layout(location = 0) out vec4 color;
vec4 shade(sampler2D t, vec2 p) { return textureLod(t, p, 2.0); }
vec4 shadeTwice(sampler2D t, vec2 p) { return shade(t, p) + shade(t, p * 0.5); }
void main() {
    float lit = texture(shadowMap, vec3(uv, 0.5));
    color = shadeTwice(albedo, uv) * lit + texture(layers[layer], uv)
        + texelFetch(fetchOnly, ivec2(0), 0);
}
"""


def test_combined_samplers_take_bound_samplers(tmp_path):
    code = _translate(tmp_path, SAMPLED)
    # The host's sampler state replaces the inline linear sampler, which read level 0 with clamp addressing.
    assert "mag_filter::linear" not in code, code
    assert "sampler albedoSampler [[sampler(1)]]" in code, code
    assert "sampler shadowMapSampler [[sampler(2)]]" in code, code
    assert "array<sampler, 4> layersSampler [[sampler(3)]]" in code, code
    assert "shadowMap.sample_compare(shadowMapSampler, input.uv, 0.5)" in code, code
    assert (
        "layers[input.layer].sample(layersSampler[input.layer], input.uv)" in code
    ), code
    _compile(tmp_path, code)


def test_helpers_receive_the_sampler_with_the_texture(tmp_path):
    code = _translate(tmp_path, SAMPLED)
    assert re.search(
        r"float4 shade\(texture2d<float> t, sampler tSampler, float2 p\)", code
    ), code
    assert "t.sample(tSampler, p, level(2.0))" in code, code
    assert re.search(r"shade\(t, tSampler, p\)", code), code
    assert re.search(r"shadeTwice\(albedo, albedoSampler, input\.uv\)", code), code


def test_texel_fetches_need_no_sampler(tmp_path):
    code = _translate(tmp_path, SAMPLED)
    assert "fetchOnlySampler" not in code, code
    assert "fetchOnly.read(" in code, code


SEPARATE = """#version 450
layout(set = 0, binding = 0) uniform texture2D colorTexture;
layout(set = 0, binding = 1) uniform sampler pointSampler;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() { color = texture(sampler2D(colorTexture, pointSampler), uv); }
"""


def test_separate_samplers_are_kept(tmp_path):
    code = _translate(tmp_path, SEPARATE)
    assert "colorTextureSampler" not in code, code
    assert "pointSampler" in code, code
    _compile(tmp_path, code)


SHARED_BINDING = """shader S {
    sampler2D colorMap @binding(0);
    sampler2D detailMap @binding(1);
    sampler linearSampler @binding(0);
    fragment {
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            return texture(colorMap, linearSampler, uv) + texture(detailMap, uv);
        }
    }
}
"""


def test_a_declared_sampler_keeps_its_slot(tmp_path):
    code = _translate(tmp_path, SHARED_BINDING, suffix=".cgl")
    assert "sampler linearSampler [[sampler(0)]]" in code, code
    assert "colorMapSampler" not in code, code
    assert re.search(r"sampler detailMapSampler \[\[sampler\(1\)\]\]", code), code
    _compile(tmp_path, code)
