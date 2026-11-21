"""WGSL output writes texture gathers, level and sample-count queries, and multisampled textures.

WGSL output used to refuse ``textureGather``, ``textureGatherOffset``, ``textureGatherCompare``,
``textureQueryLevels`` and ``textureSamples``, and every ``sampler2DMS``. Gathers now map to WGSL's
``textureGather(component, t, s, coords, ...)`` and ``textureGatherCompare``, the queries to ``textureNumLevels`` and
``textureNumSamples``, and a multisampled sampler to a ``texture_multisampled_2d<T>`` binding without a sampler. Forms
WGSL cannot express are refused by name.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import WGSLRefusal
from crosstl.translator.semantics import ShadowSampleUnsupportedError

SHADER = """
shader Gathers {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer vec4* outp;
        buffer int* iout;
        uniform sampler2D colorMap;
        uniform texture2D tex;
        uniform sampler samp;
        uniform sampler2DArray layers;
        uniform samplerCube cube;
        uniform samplerCubeArray cubes;
        uniform isampler2D ids;
        uniform sampler2DShadow shadowMap;
        uniform sampler2DArrayShadow shadowLayers;
        uniform samplerCubeShadow shadowCube;
        uniform sampler2DMS msColor;
        uniform isampler2DMS msIds;
        uniform usampler2DMS msMask;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            vec2 uv = vec2(0.25, 0.75);
            vec3 c = vec3(0.5, 0.5, 1.0);
            vec4 q = vec4(0.5, 0.5, 0.5, 1.0);
            {statement}
        }}
    }}
}}
"""


def _translate(tmp_path, statement):
    path = tmp_path / "shader.cgl"
    path.write_text(SHADER.format(statement=statement))
    return crosstl.translate(str(path), backend="wgsl")


def _naga(code, tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    result = subprocess.run(
        [naga, "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr + code


LAYER = "i32(floor(c.z + 0.5))"


@pytest.mark.parametrize(
    "expression, expected",
    [
        (
            "textureGather(colorMap, uv)",
            "textureGather(0, colorMap, colorMap_sampler, uv)",
        ),
        (
            "textureGather(colorMap, uv, 2)",
            "textureGather(2, colorMap, colorMap_sampler, uv)",
        ),
        ("textureGather(tex, samp, uv, 1)", "textureGather(1, tex, samp, uv)"),
        (
            "textureGather(layers, c, 3)",
            f"textureGather(3, layers, layers_sampler, c.xy, {LAYER})",
        ),
        ("textureGather(cube, c)", "textureGather(0, cube, cube_sampler, c)"),
        (
            "textureGather(cubes, q, 1)",
            "textureGather(1, cubes, cubes_sampler, q.xyz, i32(floor(q.w + 0.5)))",
        ),
        (
            "textureGatherOffset(colorMap, uv, ivec2(1, -1))",
            "textureGather(0, colorMap, colorMap_sampler, uv, vec2<i32>(1, -1))",
        ),
        (
            "textureGatherOffset(layers, c, ivec2(-8, 7), 2)",
            f"textureGather(2, layers, layers_sampler, c.xy, {LAYER}, vec2<i32>(-8, 7))",
        ),
        (
            "textureGatherOffset(tex, samp, uv, ivec2(0, 1), 3)",
            "textureGather(3, tex, samp, uv, vec2<i32>(0, 1))",
        ),
        (
            "textureGather(shadowMap, uv, 0.5)",
            "textureGatherCompare(shadowMap, shadowMap_sampler, uv, 0.5)",
        ),
        (
            "textureGather(shadowLayers, c, 0.5)",
            f"textureGatherCompare(shadowLayers, shadowLayers_sampler, c.xy, {LAYER}, 0.5)",
        ),
        (
            "textureGather(shadowCube, c, 0.25)",
            "textureGatherCompare(shadowCube, shadowCube_sampler, c, 0.25)",
        ),
        (
            "textureGatherOffset(shadowMap, uv, 0.5, ivec2(2, 0))",
            "textureGatherCompare(shadowMap, shadowMap_sampler, uv, 0.5, vec2<i32>(2, 0))",
        ),
        (
            "textureGatherCompare(shadowMap, uv, 0.5)",
            "textureGatherCompare(shadowMap, shadowMap_sampler, uv, 0.5)",
        ),
        (
            "textureGatherCompareOffset(shadowLayers, c, 0.5, ivec2(1, 1))",
            "textureGatherCompare(shadowLayers, shadowLayers_sampler, c.xy, "
            f"{LAYER}, 0.5, vec2<i32>(1, 1))",
        ),
    ],
)
def test_gathers_map_to_wgsl_gathers(tmp_path, expression, expected):
    code = _translate(tmp_path, f"outp[i] = {expression};")
    assert expected in code
    _naga(code, tmp_path)


def test_integer_texture_gathers_keep_their_component_type(tmp_path):
    code = _translate(tmp_path, "outp[i] = vec4(textureGather(ids, uv, 3));")
    assert "vec4<f32>(textureGather(3, ids, ids_sampler, uv))" in code
    _naga(code, tmp_path)


def test_a_separate_depth_texture_gathers_with_its_comparison_sampler(tmp_path):
    source = """
    shader SeparateDepth {
        fragment {
            uniform texture2D depth;
            uniform comparison_sampler cmp;
            vec4 main(vec4 position @ gl_FragCoord) @ gl_FragColor {
                return textureGatherCompare(depth, cmp, position.xy, 0.5);
            }
        }
    }
    """
    path = tmp_path / "separate.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="wgsl")
    assert "var depth: texture_depth_2d;" in code
    assert "textureGatherCompare(depth, cmp, position.xy, 0.5)" in code
    _naga(code, tmp_path)


@pytest.mark.parametrize("component", ["4", "-1", "int(uv.x)", "1.0"])
def test_gather_components_must_be_constants_from_0_to_3(tmp_path, component):
    with pytest.raises(WGSLRefusal, match="integer constant from 0 to 3"):
        _translate(tmp_path, f"outp[i] = textureGather(colorMap, uv, {component});")


@pytest.mark.parametrize(
    "expression, message",
    [
        (
            "textureGatherOffset(cube, c, ivec2(1, 0))",
            "gather offset only for 2D and 2D array textures",
        ),
        (
            "textureGatherOffset(shadowCube, c, 0.5, ivec2(1, 0))",
            "gather offset only for 2D and 2D array textures",
        ),
        ("textureGatherCompare(colorMap, uv, 0.5)", "takes a depth texture"),
        (
            "textureGatherOffsets(colorMap, uv, ivec2(0), ivec2(1), ivec2(2), ivec2(3))",
            "texture function textureGatherOffsets",
        ),
    ],
)
def test_gathers_wgsl_cannot_express_are_refused(tmp_path, expression, message):
    with pytest.raises(WGSLRefusal, match=message):
        _translate(tmp_path, f"outp[i] = {expression};")


def test_a_shadow_gather_without_a_reference_is_refused(tmp_path):
    with pytest.raises(ShadowSampleUnsupportedError, match="no textureGather overload"):
        _translate(tmp_path, "outp[i] = textureGather(shadowMap, uv);")


def test_named_constant_components_are_written_as_their_value(tmp_path):
    source = SHADER.replace(
        "shader Gathers {{", "shader Gathers {{\n    const int BLUE = 2;"
    ).format(statement="outp[i] = textureGather(colorMap, uv, BLUE);")
    path = tmp_path / "named.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="wgsl")
    assert "textureGather(2, colorMap, colorMap_sampler, uv)" in code
    _naga(code, tmp_path)


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("iout[i] = textureQueryLevels(colorMap);", "i32(textureNumLevels(colorMap))"),
        ("iout[i] = textureQueryLevels(layers);", "i32(textureNumLevels(layers))"),
        ("iout[i] = textureQueryLevels(cubes);", "i32(textureNumLevels(cubes))"),
        (
            "iout[i] = textureQueryLevels(shadowMap);",
            "i32(textureNumLevels(shadowMap))",
        ),
        (
            "uint levels = textureQueryLevels(colorMap); iout[i] = int(levels);",
            "var levels: u32 = textureNumLevels(colorMap);",
        ),
        ("iout[i] = textureSamples(msColor);", "i32(textureNumSamples(msColor))"),
        ("iout[i] = textureSamples(msMask);", "i32(textureNumSamples(msMask))"),
    ],
)
def test_level_and_sample_counts_map_to_wgsl_queries(tmp_path, statement, expected):
    code = _translate(tmp_path, statement)
    assert expected in code
    _naga(code, tmp_path)


@pytest.mark.parametrize(
    "statement, message",
    [
        ("iout[i] = textureQueryLevels(msColor);", "textureNumLevels takes a sampled"),
        (
            "iout[i] = textureSamples(colorMap);",
            "textureNumSamples takes a multisampled",
        ),
    ],
)
def test_queries_on_the_wrong_texture_kind_are_refused(tmp_path, statement, message):
    with pytest.raises(WGSLRefusal, match=message):
        _translate(tmp_path, statement)


def test_multisampled_textures_declare_only_a_texture_binding(tmp_path):
    code = _translate(
        tmp_path,
        "ivec2 size = textureSize(msColor); vec4 color = texelFetch(msColor, ivec2(1, 2), 3);"
        " ivec4 id = texelFetch(msIds, ivec2(0, 0), 0); uvec4 mask = texelFetch(msMask, ivec2(0, 0), 1);"
        " outp[i] = color + vec4(float(size.x), float(id.x), float(mask.x), 0.0);",
    )
    assert "var msColor: texture_multisampled_2d<f32>;" in code
    assert "var msIds: texture_multisampled_2d<i32>;" in code
    assert "var msMask: texture_multisampled_2d<u32>;" in code
    assert "msColor_sampler" not in code
    assert "vec2<i32>(textureDimensions(msColor))" in code
    assert "textureLoad(msColor, vec2<i32>(1, 2), 3)" in code
    _naga(code, tmp_path)


def test_multisampled_texture_parameters_pass_the_texture_alone(tmp_path):
    source = """
    shader MultisampledParameter {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer uint* outp;
            uniform usampler2DMS mask;
            uint coverage(usampler2DMS t, ivec2 p) {
                uint bits = 0u;
                for (int s = 0; s < textureSamples(t); s++) {
                    bits |= texelFetch(t, p, s).x;
                }
                return bits;
            }
            void main() {
                outp[gl_GlobalInvocationID.x] = coverage(mask, ivec2(0, 0));
            }
        }
    }
    """
    path = tmp_path / "parameter.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend="wgsl")
    assert "fn coverage(t: texture_multisampled_2d<u32>, p: vec2<i32>) -> u32" in code
    assert "coverage(mask, vec2<i32>(0, 0))" in code
    _naga(code, tmp_path)


@pytest.mark.parametrize("sampler_type", ["sampler2DMSArray", "isampler2DMSArray"])
def test_multisampled_array_textures_stay_refused(tmp_path, sampler_type):
    source = f"""
    shader MultisampledArray {{
        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer int* outp;
            uniform {sampler_type} u0;
            void main() {{
                outp[0] = 1;
            }}
        }}
    }}
    """
    path = tmp_path / "array.cgl"
    path.write_text(source)
    with pytest.raises(WGSLRefusal, match=f"resource type {sampler_type}"):
        crosstl.translate(str(path), backend="wgsl")
