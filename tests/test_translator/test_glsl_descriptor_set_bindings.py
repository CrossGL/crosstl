"""OpenGL binding points for resources with Vulkan descriptor sets or HLSL register spaces.

OpenGL has one flat binding array per resource class and small limits (``GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS`` is at
least 80), so ``set = S`` cannot be folded into the binding number. Set 0 keeps its bindings, other sets keep theirs
when the binding is still free in the class, and colliding resources take the lowest free binding points. The plan is
made over the whole program so every stage of it agrees.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_descriptor_sets import plan_descriptor_set_bindings


def bindings(glsl):
    return {
        name: int(binding)
        for binding, name in re.findall(
            r"layout\((?:std140, |std430, )?binding = (\d+)\)[^;{]*?(\w+)(?:\s*\{|;)",
            glsl,
        )
    }


def test_plan_keeps_set_zero_and_free_bindings_and_moves_collisions_to_lowest_free():
    plan = plan_descriptor_set_bindings(
        [
            ("texture binding", 0, 1, 1, "shadowMap"),
            ("texture binding", 1, 0, 1, "colorMap"),
            ("texture binding", 2, 1, 1, "detail"),
            ("texture binding", 3, 0, 4, "array"),
            ("uniform buffer binding", 0, 0, 1, "ubo"),
            ("uniform buffer binding", 1, 0, 1, "material"),
            ("uniform buffer binding", 1, 0, 1, "materialAlias"),
        ]
    )
    assert plan[("shadowMap", 0, 1)] == 1
    assert plan[("colorMap", 1, 0)] == 0  # free: kept
    assert plan[("detail", 2, 1)] == 2  # taken by shadowMap: lowest free
    assert plan[("array", 3, 0)] == 3  # needs four consecutive units: 3..6
    assert plan[("ubo", 0, 0)] == 0
    assert plan[("material", 1, 0)] == 1
    assert plan[("materialAlias", 1, 0)] == 1  # one descriptor, one binding point


def test_plan_is_deterministic_under_input_order():
    entries = [
        ("buffer binding", 2, 0, 1, "c"),
        ("buffer binding", 1, 0, 1, "b"),
        ("buffer binding", 0, 0, 1, "a"),
    ]
    assert plan_descriptor_set_bindings(entries) == plan_descriptor_set_bindings(
        entries[::-1]
    )
    assert plan_descriptor_set_bindings(entries) == {
        ("a", 0, 0): 0,
        ("b", 1, 0): 1,
        ("c", 2, 0): 2,
    }


PROGRAM = """
shader CascadeLike {
    struct UBO { vec4 tint; };
    layout(set = 0, binding = 0) uniform UBO ubo;
    layout(set = 0, binding = 1) uniform sampler2D shadowMap;
    layout(set = 1, binding = 0) uniform sampler2D colorMap;
    layout(set = 1, binding = 1) uniform UBO material;

    vertex {
        vec4 main(vec4 position @ Position) @ gl_Position {
            return position * material.tint;
        }
    }

    fragment {
        vec4 main() @ gl_FragColor {
            vec2 uv = vec2(0.5, 0.5);
            return texture(shadowMap, uv) * texture(colorMap, uv) * ubo.tint;
        }
    }
}
"""


def test_every_stage_of_a_program_gets_the_same_binding_points():
    ast = crosstl.translator.parse(PROGRAM)
    vertex = GLSLCodeGen().generate_stage(ast, "vertex")
    fragment = GLSLCodeGen().generate_stage(ast, "fragment")
    # Texture units 1 and 0 are distinct, so both samplers keep their binding; the set-1 block moves past ubo.
    for glsl in (vertex, fragment):
        assert "layout(std140, binding = 0) uniform UBO {" in glsl
        assert "layout(std140, binding = 1) uniform UBO_material {" in glsl
        assert "layout(binding = 1) uniform sampler2D shadowMap;" in glsl
        assert "layout(binding = 0) uniform sampler2D colorMap;" in glsl


def test_hlsl_register_spaces_get_distinct_opengl_binding_points(tmp_path):
    source = tmp_path / "spaces.hlsl"
    source.write_text("""
        cbuffer Frame : register(b0) { float4 frameTint; };
        cbuffer Material : register(b0, space1) { float4 materialTint; };
        Texture2D albedo : register(t0, space1);
        SamplerState linearSampler : register(s0);
        float4 main(float2 uv : TEXCOORD0) : SV_Target
        {
            return albedo.Sample(linearSampler, uv) * frameTint * materialTint;
        }
        """)
    glsl = crosstl.translate(str(source), backend="opengl", format_output=False)
    found = bindings(glsl)
    assert all(binding < 80 for binding in found.values()), found
    uniform_blocks = re.findall(r"layout\(std140, binding = (\d+)\) uniform", glsl)
    assert sorted(uniform_blocks) == ["0", "1"], glsl
    if shutil.which("glslangValidator"):
        artifact = tmp_path / "spaces.frag"
        artifact.write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "frag", str(artifact)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
