"""MSL ``buffer``, ``texture`` and ``sampler`` indices choose the Vulkan descriptor slots of the SPIR-V resources.

MSL keeps a function's buffers, textures and samplers in three argument tables, and each entry point has its own.
A Vulkan descriptor set has one binding namespace, so each resource keeps binding ``n`` when it is free, and a later
table takes the next free binding. Resources are placed in table order (buffers, textures, samplers), then index,
so the slots do not depend on declaration order. Two resources of one table that claim one index are rejected, as
the Metal compiler rejects them. The SPIR-V generator used to ignore ``texture(n)`` and ``sampler(n)`` and number
those resources in declaration order, moving a buffer that came after them off its index.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.spirv_argument_tables import ArgumentTableOverlapError

PRELUDE = """#include <metal_stdlib>
using namespace metal;
struct VOut { float4 position [[position]]; float2 uv; };
"""

INDEXED_FRAGMENT = PRELUDE + textwrap.dedent("""
    fragment float4 fragMain(VOut in [[stage_in]],
                             texture2d<float> albedo [[texture(3)]],
                             texture2d<float> normal [[texture(7)]],
                             sampler samp [[sampler(2)]],
                             constant float4 &tint [[buffer(5)]]) {
        return albedo.sample(samp, in.uv) * normal.sample(samp, in.uv) * tint;
    }
    """)

SHARED_INDICES = """
    vertex VOut vertMain(uint vid [[vertex_id]], {vertex_params}) {{
        VOut o;
        float h = heights.sample(hs, float2(0.5)).r;
        o.position = float4(h, 0, 0, 1) * scale;
        o.uv = float2(h);
        return o;
    }}

    fragment float4 fragMain(VOut in [[stage_in]], texture2d<float> albedo [[texture(0)]],
                             texture2d<float> detail [[texture(1)]], sampler samp [[sampler(0)]]) {{
        return albedo.sample(samp, in.uv) * detail.sample(samp, in.uv);
    }}
    """

VERTEX_PARAMS = [
    "texture2d<float> heights [[texture(0)]]",
    "sampler hs [[sampler(0)]]",
    "constant float4 &scale [[buffer(0)]]",
]


def _translate(tmp_path, source, suffix=".metal", backend="vulkan"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _bindings(spirv):
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', spirv))
    sets = dict(re.findall(r"OpDecorate %(\w+) DescriptorSet (\d+)", spirv))
    return {
        names[ident]: (int(sets[ident]), int(binding))
        for ident, binding in re.findall(r"OpDecorate %(\w+) Binding (\d+)", spirv)
    }


def _validate(tmp_path, spirv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not installed")
    (tmp_path / "shader.spvasm").write_text(spirv, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "shader.spvasm", "-o", "s.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "s.spv"],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, cwd=tmp_path
        )
        assert result.returncode == 0, result.stderr[-2000:]


def test_free_indices_keep_their_binding(tmp_path):
    spirv = _translate(tmp_path, INDEXED_FRAGMENT)
    assert _bindings(spirv) == {
        "albedo": (0, 3),
        "normal": (0, 7),
        "samp": (0, 2),
        "tintUniform": (0, 5),
    }
    _validate(tmp_path, spirv)


def test_tables_share_the_set_in_buffer_texture_sampler_order(tmp_path):
    source = PRELUDE + textwrap.dedent(SHARED_INDICES).format(
        vertex_params=", ".join(VERTEX_PARAMS)
    )
    spirv = _translate(tmp_path, source)
    bindings = _bindings(spirv)
    # buffer(0) keeps 0, texture(0) moves to 1 and sampler(0) to 2.
    assert bindings["scaleUniform"] == (0, 0)
    assert bindings["heights"] == (0, 1)
    assert bindings["hs"] == (0, 2)
    # The fragment entry point has its own tables: texture(0), texture(1), then sampler(0) at 2.
    assert bindings["albedo"] == (0, 0)
    assert bindings["detail"] == (0, 1)
    assert bindings["samp"] == (0, 2)
    _validate(tmp_path, spirv)


def test_slots_do_not_depend_on_parameter_order(tmp_path):
    def slots(params):
        source = PRELUDE + textwrap.dedent(SHARED_INDICES).format(
            vertex_params=", ".join(params)
        )
        return _bindings(_translate(tmp_path, source))

    assert slots(VERTEX_PARAMS) == slots(list(reversed(VERTEX_PARAMS)))


def test_two_textures_at_one_index_are_refused(tmp_path):
    source = PRELUDE + textwrap.dedent("""
        fragment float4 f(texture2d<float> a [[texture(0)]], texture2d<float> b [[texture(0)]],
                          sampler s [[sampler(0)]]) {
            return a.sample(s, float2(0)) + b.sample(s, float2(0));
        }
        """)
    with pytest.raises(ArgumentTableOverlapError) as error:
        _translate(tmp_path, source)
    assert error.value.code == "project.translate.argument-table-overlap"
    assert "texture index 0" in str(error.value)


def test_a_texture_and_a_sampler_may_share_an_index(tmp_path):
    source = PRELUDE + textwrap.dedent("""
        fragment float4 f(VOut in [[stage_in]], texture2d<float> a [[texture(1)]],
                          sampler s [[sampler(1)]]) {
            return a.sample(s, in.uv);
        }
        """)
    spirv = _translate(tmp_path, source)
    assert _bindings(spirv) == {"a": (0, 1), "s": (0, 2)}
    _validate(tmp_path, spirv)


CROSSGL_GLOBALS = """
shader GlobalIndices {{
    struct FIn {{ vec2 uv; }}
    struct FOut {{ vec4 color; }}
    vertex {{
        uniform sampler2D heights @texture(0);
        vec4 main(vec2 p) @gl_Position {{ return vec4(p, texture(heights, p).r, 1.0); }}
    }}
    fragment {{
        uniform sampler2D albedo @texture(4);
        uniform sampler2D detail @texture({detail});
        FOut main(FIn input) {{
            FOut o;
            o.color = texture(albedo, input.uv) * texture(detail, input.uv);
            return o;
        }}
    }}
}}
"""


def test_crossgl_resources_keep_their_texture_index(tmp_path):
    spirv = _translate(tmp_path, CROSSGL_GLOBALS.format(detail=0), suffix=".cgl")
    # Each stage has its own table, so the fragment texture(0) is not refused; binding 0 is held by the vertex
    # stage's texture(0), so it takes the next free binding.
    assert _bindings(spirv) == {
        "heights": (0, 0),
        "albedo": (0, 4),
        "detail": (0, 1),
    }
    _validate(tmp_path, spirv)


def test_crossgl_resources_of_one_stage_at_one_index_are_refused(tmp_path):
    with pytest.raises(ArgumentTableOverlapError):
        _translate(tmp_path, CROSSGL_GLOBALS.format(detail=4), suffix=".cgl")
