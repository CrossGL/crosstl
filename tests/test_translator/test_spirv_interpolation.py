"""interpolateAtCentroid, interpolateAtSample and interpolateAtOffset (GLSL 4.60 section 8.13.2) in SPIR-V output.

GLSL.std.450 InterpolateAtCentroid, InterpolateAtSample and InterpolateAtOffset read their interpolant through a
pointer to the Input storage class, so these tests check that the pointer is the fragment input's own variable,
never the entry parameter's local copy, and that every interpolant SPIR-V cannot address is refused.
"""

import re
import shutil
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

STRUCT_INPUT = """shader Interpolation {
    struct FragmentInput {
        vec2 uv;
        float a0;
        vec4 tint;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        buffer float* outp;
        buffer int* samples;
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            float c = interpolateAtCentroid(input.a0);
            float s = interpolateAtSample(input.a0, samples[0]);
            vec2 o = interpolateAtOffset(input.uv, vec2(0.25, -0.25));
            vec2 t = interpolateAtCentroid(input.tint.zw);
            float y = interpolateAtSample(input.tint.y, 1);
            outp[0] = c + s + o.x + t.y + y;
            output.color = vec4(input.uv, input.a0, 1.0);
            return output;
        }
    }
}"""

STAGE_INPUTS = """shader Interpolation {
    fragment {
        float a0 @input @location(0);
        vec3 normals[2] @input @location(1);
        vec4 color @output @location(0);
        void main() {
            float r = interpolateAtOffset(a0, vec2(0.5, 0.5));
            vec3 n = interpolateAtCentroid(normals[1]);
            color = vec4(n, r);
        }
    }
}"""


def _validate(assembly, tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    (tmp_path / "unit.spvasm").write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", "unit.spvasm", "-o", "unit.spv"],
        ["spirv-val", "--target-env", "vulkan1.1", "unit.spv"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr


def _interpolations(assembly):
    """``(result, instruction, interpolant)`` for every interpolation instruction."""
    return re.findall(
        r"(%\w+) = OpExtInst %\w+ %\w+ (InterpolateAt\w+) (%\w+)", assembly
    )


def _input_variables(assembly):
    return set(re.findall(r"(%\w+) = OpVariable %\w+ Input\n", assembly))


def _named(assembly, name):
    """The id ``OpName`` gives ``name``."""
    match = re.search(rf'OpName (%\w+) "{name}"\n', assembly)
    assert match is not None, name
    return match.group(1)


def test_struct_parameter_members_interpolate_their_input_variables(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(STRUCT_INPUT))

    assert "OpCapability InterpolationFunction" in assembly
    found = _interpolations(assembly)
    assert [instruction for _, instruction, _ in found] == [
        "InterpolateAtCentroid",
        "InterpolateAtSample",
        "InterpolateAtOffset",
        "InterpolateAtCentroid",
        "InterpolateAtSample",
    ]
    inputs = _input_variables(assembly)
    a0 = _named(assembly, "CrossGL_fragment_input_input_a0")
    tint = _named(assembly, "CrossGL_fragment_input_input_tint")
    assert found[0][2] == a0 and found[1][2] == a0
    assert found[2][2] == _named(assembly, "CrossGL_fragment_input_input_uv")
    # Two components of tint: the whole vector is interpolated, then shuffled.
    assert found[3][2] == tint
    assert re.search(
        rf"OpVectorShuffle %\w+ {found[3][0]} {found[3][0]} 2 3\n", assembly
    )
    # One component of tint is addressable: an access chain into the Input variable.
    chain = re.search(rf"{found[4][2]} = OpAccessChain %\w+ {tint} %\w+\n", assembly)
    assert chain is not None
    assert {found[0][2], found[2][2], found[3][2]} <= inputs
    _validate(assembly, tmp_path)


def test_stage_inputs_and_their_array_elements_are_interpolated(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(parse(STAGE_INPUTS))

    found = _interpolations(assembly)
    assert [instruction for _, instruction, _ in found] == [
        "InterpolateAtOffset",
        "InterpolateAtCentroid",
    ]
    assert found[0][2] == _named(assembly, "a0")
    normals = _named(assembly, "normals")
    assert re.search(rf"{found[1][2]} = OpAccessChain %\w+ {normals} %\w+\n", assembly)
    _validate(assembly, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        # A local value has no Input variable.
        "float v = input.a0; outp[0] = interpolateAtCentroid(v);",
        # The parameter's copy no longer holds the input once the entry point writes it.
        "input.a0 = 2.0; outp[0] = interpolateAtCentroid(input.a0);",
        # Integer inputs are flat and cannot be interpolated.
        "outp[0] = float(interpolateAtCentroid(input.index));",
    ],
)
def test_interpolants_without_an_input_pointer_are_refused(body):
    source = f"""shader Interpolation {{
        struct FragmentInput {{
            float a0;
            int index;
        }}
        fragment {{
            buffer float* outp;
            vec4 main(FragmentInput input) @ gl_FragColor {{
                {body}
                return vec4(1.0);
            }}
        }}
    }}"""
    with pytest.raises(SpirvLoweringRefused, match="interpolateAtCentroid"):
        VulkanSPIRVCodeGen().generate(parse(source))


def test_interpolation_outside_a_fragment_entry_point_is_refused():
    source = """shader Interpolation {
        vertex {
            float a0 @input @location(0);
            vec4 main() @ gl_Position {
                return vec4(interpolateAtCentroid(a0));
            }
        }
    }"""
    with pytest.raises(SpirvLoweringRefused, match="Fragment execution model"):
        VulkanSPIRVCodeGen().generate(parse(source))
