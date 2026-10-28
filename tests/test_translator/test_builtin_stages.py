"""``gl_`` built-in values are refused in the stages and directions GLSL does not declare them for."""

import pytest

import crosstl
from crosstl.translator.semantics.builtin_names import CROSSGL_GL_BUILTIN_NAMES
from crosstl.translator.semantics.builtin_stages import (
    BUILTIN_STAGE_RULES,
    builtin_stage_rule,
)
from crosstl.translator.semantics.builtin_values import BuiltinValueUnsupportedError

GRAPHICS_TARGETS = ["vulkan", "metal", "directx", "opengl", "webgl"]
ALL_TARGETS = GRAPHICS_TARGETS + ["slang", "cuda", "hip", "rust", "mojo", "wgsl"]

VERTEX_OUTPUT_MEMBER = """shader Probe {
    struct VertexOutput {
        vec4 position @gl_Position;
        uint value SEMANTIC;
    }
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.position = vec4(0.0);
            output.value = 1u;
            return output;
        }
    }
}
"""

VERTEX_INPUT_PARAMETER = """shader Probe {
    vertex {
        vec4 main(bool value SEMANTIC) @gl_Position {
            return value ? vec4(1.0) : vec4(0.0);
        }
    }
}
"""

FRAGMENT_INPUT_MEMBER = """shader Probe {
    struct FragmentInput {
        uvec3 value SEMANTIC;
    }
    fragment {
        vec4 main(FragmentInput input) @gl_FragColor {
            return vec4(float(input.value.x));
        }
    }
}
"""

FRAGMENT_RETURN = """shader Probe {
    fragment {
        vec4 main() SEMANTIC {
            return vec4(1.0);
        }
    }
}
"""

COMPUTE_VARIABLE = """shader Probe {
    compute {
        vec4 value @input SEMANTIC;
        buffer float values[];
        @workgroup_size(1, 1, 1)
        void main() {
            values[0] = value.x;
        }
    }
}
"""

MISPLACED = [
    pytest.param(VERTEX_OUTPUT_MEMBER, "@gl_SubgroupSize", "vertex output", id="vertex-output"),
    pytest.param(VERTEX_INPUT_PARAMETER, "@gl_FrontFacing", "vertex input", id="vertex-input"),
    pytest.param(
        FRAGMENT_INPUT_MEMBER, "@gl_LaunchIDEXT", "fragment input", id="fragment-input"
    ),
    pytest.param(FRAGMENT_RETURN, "@gl_FragCoord", "fragment output", id="fragment-output"),
    pytest.param(COMPUTE_VARIABLE, "@gl_FragCoord", "compute input", id="compute-input"),
]

# A vertex output structure the fragment stage also takes, as HLSL shaders commonly declare it: the fragment reads
# gl_Position as its window position and ignores gl_PointSize (CrossGL conventions).
SHARED_VARYINGS = """shader Probe {
    struct Varyings {
        vec4 position @gl_Position;
        float size @gl_PointSize;
        vec2 uv @TEXCOORD0;
    }
    vertex {
        Varyings main(vec3 position @POSITION) {
            Varyings output;
            output.position = vec4(position, 1.0);
            output.size = 1.0;
            output.uv = position.xy;
            return output;
        }
    }
    fragment {
        vec4 main(Varyings input) @gl_FragColor {
            return vec4(input.uv, input.position.z, 1.0);
        }
    }
}
"""

# gl_SampleMask is both a fragment output and, as HLSL's SV_Coverage input, a fragment input.
FRAGMENT_SAMPLE_MASK = """shader Probe {
    struct FragmentInput {
        vec4 position @gl_FragCoord;
        uint coverage @gl_SampleMask;
    }
    struct FragmentOutput {
        vec4 colour @gl_FragColor;
        uint coverage @gl_SampleMask;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.colour = input.position;
            output.coverage = input.coverage;
            return output;
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ALL_TARGETS)
@pytest.mark.parametrize("template, semantic, role", MISPLACED)
def test_misplaced_builtins_are_refused(tmp_path, backend, template, semantic, role):
    source = template.replace("SEMANTIC", semantic)
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, source, backend)
    message = str(raised.value)
    assert f"{semantic} on " in message
    assert f"is not a {role} in GLSL" in message
    assert "GLSL 4.60 section 7.1" in message
    assert raised.value.code == "project.translate.builtin-value-unsupported"


def test_refusal_names_the_stages_glsl_declares(tmp_path):
    source = FRAGMENT_INPUT_MEMBER.replace("SEMANTIC", "@gl_LaunchIDEXT")
    with pytest.raises(BuiltinValueUnsupportedError) as raised:
        _translate(tmp_path, source, "vulkan")
    message = str(raised.value)
    assert "member 'value' of 'FragmentInput'" in message
    assert (
        "GLSL declares gl_LaunchIDEXT as an input of ray_any_hit, ray_callable, ray_closest_hit, "
        "ray_generation, ray_intersection, ray_miss" in message
    )


@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
def test_shared_vertex_output_structure_is_accepted(tmp_path, backend):
    source = SHARED_VARYINGS.replace("        float size @gl_PointSize;\n", "").replace(
        "            output.size = 1.0;\n", ""
    )
    assert "gl_PointSize" not in source
    assert _translate(tmp_path, source, backend)


# SPIR-V refuses a fragment input PointSize built-in itself (spirv_refusals), after this check accepts it.
@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "webgl"])
def test_shared_vertex_output_point_size_is_accepted(tmp_path, backend):
    assert _translate(tmp_path, SHARED_VARYINGS, backend)


# GLSL ES 3.00 has no gl_SampleMask, so WebGL refuses it itself.
@pytest.mark.parametrize("backend", ["vulkan", "metal", "directx", "opengl"])
def test_fragment_sample_mask_input_and_output_are_accepted(tmp_path, backend):
    assert _translate(tmp_path, FRAGMENT_SAMPLE_MASK, backend)


@pytest.mark.parametrize(
    "semantic",
    ["@gl_SubgroupInvocationID", "@gl_WorkGroupID", "@gl_LocalInvocationIndex"],
)
def test_valid_compute_inputs_are_accepted(tmp_path, semantic):
    source = COMPUTE_VARIABLE.replace("vec4 value", "uvec3 value").replace(
        "value.x;", "float(value.x);"
    )
    if semantic != "@gl_WorkGroupID":
        source = source.replace("uvec3 value", "uint value").replace("value.x", "value")
    assert _translate(tmp_path, source.replace("SEMANTIC", semantic), "vulkan")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("gl_FragData[2]", "gl_FragData"),
        ("gl_FragColor1", "gl_FragColor"),
        ("gl_ClipDistance0", "gl_ClipDistance"),
        ("GL_POSITION", "gl_Position"),
        ("gl_fragcoord", "gl_FragCoord"),
    ],
)
def test_rule_lookup_folds_indices_and_case(name, expected):
    assert builtin_stage_rule(name) is BUILTIN_STAGE_RULES[expected]


@pytest.mark.parametrize("name", ["", None, "gl_FragBogus", "SV_Position", "gl_FragColorX"])
def test_rule_lookup_has_no_rule_for_other_names(name):
    assert builtin_stage_rule(name) is None


def test_every_registered_builtin_has_a_stage_rule():
    assert set(BUILTIN_STAGE_RULES) == set(CROSSGL_GL_BUILTIN_NAMES)
    for name, rule in BUILTIN_STAGE_RULES.items():
        assert rule["input"] or rule["output"], name


@pytest.mark.parametrize(
    "name, direction, stage",
    [
        ("gl_FragDepth", "output", "fragment"),
        ("gl_TessLevelOuter", "output", "tessellation_control"),
        ("gl_TessLevelOuter", "input", "tessellation_evaluation"),
        ("gl_PrimitiveIDIn", "input", "geometry"),
        ("gl_HitTEXT", "input", "ray_closest_hit"),
        ("gl_PrimitiveTriangleIndicesEXT", "output", "mesh"),
        ("gl_GlobalInvocationID", "input", "amplification"),
    ],
)
def test_rules_follow_the_glsl_tables(name, direction, stage):
    assert stage in BUILTIN_STAGE_RULES[name][direction]


@pytest.mark.parametrize(
    "name, direction, stage",
    [
        ("gl_FragDepth", "input", "fragment"),
        ("gl_FragCoord", "output", "fragment"),
        ("gl_VertexID", "input", "fragment"),
        ("gl_HitTEXT", "input", "ray_miss"),
        ("gl_TessCoord", "input", "tessellation_control"),
    ],
)
def test_rules_exclude_undeclared_uses(name, direction, stage):
    assert stage not in BUILTIN_STAGE_RULES[name][direction]
