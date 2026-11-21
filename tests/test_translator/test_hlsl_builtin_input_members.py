"""Entry structure members bound to built-ins that HLSL reads through an intrinsic.

``bool helper @ gl_HelperInvocation`` in a fragment input structure has no HLSL system value: Direct3D reads helper
status with ``IsHelperLane()``. The generator refused the member. The entry now takes a signature structure without
it and builds the program's structure in its body (:mod:`crosstl.translator.codegen.hlsl_builtin_input_members`).
"""

import re
import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

HELPER_PROBE = """shader Probe {
    struct FragmentInput {
        vec2 uv;
        bool helper @ gl_HelperInvocation;
    }
    struct FragmentOutput {
        vec4 color;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, input.helper ? 1.0 : 0.0, 1.0);
            return output;
        }
    }
}
"""

LINKED = """shader Linked {
    struct VertexOutput {
        vec4 position @ gl_Position;
        vec2 uv;
        vec3 normal;
    }
    struct FragmentInput {
        vec2 uv;
        bool helper @ gl_HelperInvocation;
        vec3 normal;
        uint lanes @ gl_SubgroupSize;
    }
    float shade(FragmentInput value) {
        return value.helper ? 0.0 : value.normal.x + float(value.lanes);
    }
    vertex {
        VertexOutput main(vec3 position @ POSITION) {
            VertexOutput output;
            output.position = vec4(position, 1.0);
            output.uv = position.xy;
            output.normal = position;
            return output;
        }
    }
    fragment {
        vec4 main(FragmentInput input) @ gl_FragColor {
            return vec4(input.uv, shade(input), 1.0);
        }
    }
}
"""


def _generate(source):
    return HLSLCodeGen().generate(crosstl.translator.parse(source))


def _compile(tmp_path, hlsl, profile, entry):
    if shutil.which("dxc") is None:
        return
    path = tmp_path / f"{entry}.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        ["dxc", "-T", profile, "-E", entry, str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _struct(hlsl, name):
    match = re.search(rf"struct {name}\s*\{{(.*?)\}};", hlsl, re.S)
    assert match, hlsl
    return re.sub(r"\s*:\s*", " : ", " ".join(match.group(1).split()))


def test_helper_member_reads_is_helper_lane(tmp_path):
    hlsl = _generate(HELPER_PROBE)
    assert "gl_HelperInvocation" not in hlsl
    assert _struct(hlsl, "FragmentInput") == "float2 uv; bool helper;"
    assert _struct(hlsl, "FragmentInputSignature") == "float2 uv : TEXCOORD0;"
    assert "FragmentOutput PSMain(FragmentInputSignature input_signature)" in hlsl
    body = hlsl[hlsl.index("PSMain(") :]
    assert body.index("FragmentInput input;") < body.index("FragmentOutput output;")
    assert "input.uv = input_signature.uv;" in body
    assert "input.helper = IsHelperLane();" in body
    _compile(tmp_path, hlsl, "ps_6_6", "PSMain")


def test_signature_keeps_the_varyings_the_vertex_stage_writes(tmp_path):
    hlsl = _generate(LINKED)
    assert _struct(hlsl, "VertexOutput") == (
        "float4 position : SV_POSITION; float2 uv : TEXCOORD0; float3 normal : TEXCOORD1;"
    )
    assert _struct(hlsl, "FragmentInputSignature") == (
        "float2 uv : TEXCOORD0; float3 normal : TEXCOORD1;"
    )
    # The program's structure is a value type the helper still takes whole.
    assert _struct(hlsl, "FragmentInput") == (
        "float2 uv; bool helper; float3 normal; uint lanes;"
    )
    assert "float shade(FragmentInput value)" in hlsl
    body = hlsl[hlsl.index("PSMain(") :]
    lines = [line.strip() for line in body.splitlines()]
    start = lines.index("FragmentInput input;")
    assert lines[start : start + 5] == [
        "FragmentInput input;",
        "input.uv = input_signature.uv;",
        "input.helper = IsHelperLane();",
        "input.normal = input_signature.normal;",
        "input.lanes = WaveGetLaneCount();",
    ]
    _compile(tmp_path, hlsl, "ps_6_6", "PSMain")
    _compile(tmp_path, hlsl, "vs_6_6", "VSMain")


def test_structure_of_built_ins_only_leaves_the_signature(tmp_path):
    hlsl = _generate("""shader Lanes {
        struct Lanes {
            uvec4 below @ gl_SubgroupLtMask;
            int samples @ gl_NumSamples;
        }
        fragment {
            vec4 main(Lanes lanes) @ gl_FragColor {
                return vec4(float(lanes.below.x), float(lanes.samples), 0.0, 1.0);
            }
        }
    }""")
    assert "LanesSignature" not in hlsl
    assert re.search(r"float4 PSMain\(\)\s*:\s*SV_TARGET", hlsl)
    assert "lanes.samples = int(GetRenderTargetSampleCount());" in hlsl
    assert "lanes.below = " in hlsl and "WaveGetLaneIndex()" in hlsl
    _compile(tmp_path, hlsl, "ps_6_6", "PSMain")


def test_new_names_avoid_the_program_names(tmp_path):
    source = HELPER_PROBE.replace(
        "    struct FragmentOutput {",
        "    struct FragmentInputSignature {\n        float unused;\n    }\n"
        "    struct FragmentOutput {",
    ).replace(
        "FragmentOutput output;",
        "FragmentOutput output;\n            float input_signature = 0.5;",
    )
    hlsl = _generate(source)
    assert "PSMain(FragmentInputSignature2 input_signature2)" in hlsl
    assert "input.uv = input_signature2.uv;" in hlsl
    assert "float input_signature = 0.5;" in hlsl
    _compile(tmp_path, hlsl, "ps_6_6", "PSMain")


def test_the_parsed_program_is_not_changed():
    ast = crosstl.translator.parse(HELPER_PROBE)
    HLSLCodeGen().generate(ast)
    member = ast.structs[0].members[1]
    assert [attribute.name for attribute in member.attributes] == [
        "gl_HelperInvocation"
    ]
    assert len(ast.structs) == 2


@pytest.mark.parametrize(
    "source",
    [
        # A member that cannot hold the value: the parameter path refuses it too.
        HELPER_PROBE.replace("bool helper", "vec2 helper").replace(
            "input.helper ? 1.0 : 0.0", "input.helper.x"
        ),
        # The structure is also an entry's output.
        """shader Both {
            struct Value {
                vec4 position @ gl_Position;
                bool helper @ gl_HelperInvocation;
            }
            vertex {
                Value main() {
                    Value output;
                    output.position = vec4(0.0);
                    output.helper = false;
                    return output;
                }
            }
            fragment {
                vec4 main(Value input) @ gl_FragColor {
                    return vec4(input.helper ? 1.0 : 0.0);
                }
            }
        }""",
    ],
    ids=["mismatched-type", "also-an-output"],
)
def test_other_uses_keep_the_refusal(source):
    with pytest.raises(
        DirectXLoweringRefused,
        match="no system value for the built-in @gl_HelperInvocation",
    ):
        _generate(source)
