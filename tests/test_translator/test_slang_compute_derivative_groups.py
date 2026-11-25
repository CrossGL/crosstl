"""Slang output refuses derivatives in a compute stage whose thread group is not a derivative group (issue #4657)."""

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused

DERIVATIVE_BUILTINS = [
    "dFdx",
    "dFdy",
    "dFdxCoarse",
    "dFdxFine",
    "dFdyCoarse",
    "dFdyFine",
    "fwidth",
    "fwidthCoarse",
    "fwidthFine",
]

# The issue's probe: derivatives plus a biased sample at 1x1x1.
PROBE = """
shader DerivCompute {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uniform sampler2D tex;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = float(i);
            vec2 uv = vec2(x * 0.25, 0.5);
            float d = dFdx(x);
            vec4 c = texture(tex, uv, 1.0);
            outp[i] = d + c.x + fwidth(x);
        }
    }
}
"""


def compute_cgl(size, body):
    x, y, z = size
    return textwrap.dedent(f"""
        shader DerivCompute {{
            compute {{
                layout(local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;
                buffer float* outp;
                uniform sampler2D tex;
                void main() {{
                    uint i = gl_GlobalInvocationID.x;
                    {body}
                }}
            }}
        }}
        """)


def generate(source):
    return SlangCodeGen().generate(crosstl.translator.parse(textwrap.dedent(source)))


@pytest.mark.parametrize(
    "size",
    [(1, 1, 1), (2, 1, 1), (3, 1, 1), (6, 1, 1), (3, 4, 1), (1, 4, 1)],
    ids=["1x1x1", "2x1x1", "3x1x1", "6x1x1", "3x4x1", "1x4x1"],
)
def test_slang_refuses_derivatives_without_a_derivative_group(size):
    with pytest.raises(SlangLoweringRefused) as exc_info:
        generate(
            PROBE.replace("local_size_x = 1", f"local_size_x = {size[0]}")
            .replace("local_size_y = 1", f"local_size_y = {size[1]}")
            .replace("local_size_z = 1", f"local_size_z = {size[2]}")
        )

    message = str(exc_info.value)
    assert "dFdx" in message
    assert f"{size[0]}x{size[1]}x{size[2]}" in message
    assert "E3121" in message


@pytest.mark.parametrize(
    "size",
    [(4, 1, 1), (8, 1, 1), (2, 2, 1)],
    ids=["4x1x1", "8x1x1", "2x2x1"],
)
def test_slang_keeps_derivatives_with_a_derivative_group(size):
    generated = generate(
        PROBE.replace("local_size_x = 1", f"local_size_x = {size[0]}")
        .replace("local_size_y = 1", f"local_size_y = {size[1]}")
        .replace("local_size_z = 1", f"local_size_z = {size[2]}")
    )

    assert "ddx(x)" in generated
    assert "fwidth(x)" in generated
    assert "SampleBias" in generated


@pytest.mark.parametrize("builtin", DERIVATIVE_BUILTINS)
def test_slang_refuses_each_derivative_builtin_in_compute(builtin):
    source = compute_cgl((1, 1, 1), f"float x = float(i); outp[i] = {builtin}(x);")

    with pytest.raises(SlangLoweringRefused) as exc_info:
        generate(source)

    assert builtin in str(exc_info.value)
    assert "1x1x1" in str(exc_info.value)


def test_slang_refuses_a_biased_sample_without_other_derivatives():
    source = compute_cgl(
        (1, 1, 1),
        "vec2 uv = vec2(0.25, 0.5); vec4 c = texture(tex, uv, 1.0); outp[i] = c.x;",
    )

    with pytest.raises(SlangLoweringRefused) as exc_info:
        generate(source)

    assert "SampleBias" in str(exc_info.value)
    assert "1x1x1" in str(exc_info.value)


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("texture(tex, uv, 1.0)", "SampleBias"),
        ("textureOffset(tex, uv, ivec2(0, 0), 1.0)", "SampleBias"),
        ("textureProj(tex, vec3(0.25, 0.5, 1.0), 1.0)", "SampleBias"),
        (
            "textureProjOffset(tex, vec3(0.25, 0.5, 1.0), ivec2(0, 0), 1.0)",
            "SampleBias",
        ),
    ],
    ids=["texture", "textureOffset", "textureProj", "textureProjOffset"],
)
def test_slang_refuses_each_biased_sample_form(call, operation):
    source = compute_cgl(
        (1, 1, 1),
        f"vec2 uv = vec2(0.25, 0.5); vec4 c = {call}; outp[i] = c.x;",
    )

    with pytest.raises(SlangLoweringRefused) as exc_info:
        generate(source)

    assert operation in str(exc_info.value)


@pytest.mark.parametrize(
    "call",
    [
        "texture(tex, uv)",
        "textureOffset(tex, uv, ivec2(0, 0))",
        "textureProj(tex, vec3(0.25, 0.5, 1.0))",
        "textureGrad(tex, uv, vec2(0.01), vec2(0.01))",
        "textureLod(tex, uv, 1.0)",
    ],
    ids=["unbiassed", "offset-only", "projected", "grad", "lod"],
)
def test_slang_keeps_samples_that_need_no_derivative_group(call):
    source = compute_cgl(
        (1, 1, 1),
        f"vec2 uv = vec2(0.25, 0.5); vec4 c = {call}; outp[i] = c.x;",
    )

    generated = generate(source)

    assert "SampleBias" not in generated


def test_slang_refuses_derivatives_reached_through_a_helper():
    source = """
    shader DerivCompute {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* outp;
            float slope(float x) { return dFdx(x); }
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = slope(float(i));
            }
        }
    }
    """

    with pytest.raises(SlangLoweringRefused) as exc_info:
        generate(source)

    message = str(exc_info.value)
    assert "dFdx" in message
    assert "through 'slope'" in message


def test_slang_keeps_derivatives_in_fragment_stages():
    source = """
    shader DerivFrag {
        fragment {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                vec2 dx = dFdx(uv);
                float edge = fwidth(uv.x);
            }
        }
    }
    """

    generated = generate(source)

    assert "ddx(uv)" in generated
    assert "fwidth(uv.x)" in generated


def test_slang_keeps_derivative_free_compute_at_1x1x1():
    source = compute_cgl((1, 1, 1), "outp[i] = float(i) * 2.0;")

    generated = generate(source)

    assert "outp[i]" in generated


def test_slang_derivative_refusal_reaches_the_public_translate_api(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(textwrap.dedent(PROBE))

    with pytest.raises(SlangLoweringRefused, match="dFdx.*1x1x1"):
        crosstl.translate(str(path), backend="slang")


def test_slang_refuses_derivatives_from_glsl_input(tmp_path):
    path = tmp_path / "probe.comp"
    path.write_text(textwrap.dedent("""
            #version 450
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            layout(std430, binding = 0) buffer Outp { float o[]; };
            void main() {
                uint i = gl_GlobalInvocationID.x;
                o[i] = dFdx(float(i));
            }
            """))

    # The GLSL frontend already spells the builtin ``ddx``.
    with pytest.raises(SlangLoweringRefused, match="ddx.*1x1x1"):
        crosstl.translate(str(path), backend="slang")


def test_slang_leaves_specialized_sizes_to_slangc():
    source = """
    shader DerivCompute {
        compute {
            layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
            buffer float* outp;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = dFdx(float(i));
            }
        }
    }
    """

    generated = generate(source)

    assert "ddx(" in generated
